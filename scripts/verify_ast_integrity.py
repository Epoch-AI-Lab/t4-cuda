#!/usr/bin/env python3
"""Programmatic AST Verification Tool for T4-CUDA Codebase Integrity.

Scans all Python source files in specified directories (src/, tests/, benchmarks/) to detect:
1. Tautological assertions (literal True/1, identical operands, constant comparisons, dummy loops).
2. Banned timing variables (compute_per_layer_ms, tp_compute_ms, pp_compute_ms, etc.).
3. Artificial latency scaling multipliers (0.65, 0.70, etc.).
4. Mock passes ("else: assert True" in device conditionals).

Exits with code 0 only if 0 violations are found; exits with code 1 otherwise.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import os
import sys
from typing import Dict, List, Optional, Set, Tuple

BANNED_VARIABLES: Set[str] = {
    "compute_per_layer_ms",
    "tp_compute_ms",
    "pp_compute_ms",
    "bfe_count",
    "lop3_count",
    "simulated_latency",
    "fake_latency",
    "dummy_latency",
    "unfused_latency_ms",
    "fused_latency_ms",
    "standard_gemm_stall_cycles",
    "warp_spec_stall_cycles",
    "naive_sass_per_pair",
    "optimized_sass_per_pair",
}

BANNED_MULTIPLIERS: Set[float] = {0.65, 0.70, 0.75, 0.85}


class Violation:
    def __init__(
        self,
        filename: str,
        lineno: int,
        col_offset: int,
        violation_type: str,
        message: str,
        source_line: str = "",
    ):
        self.filename = filename
        self.lineno = lineno
        self.col_offset = col_offset
        self.violation_type = violation_type
        self.message = message
        self.source_line = source_line

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "line": self.lineno,
            "col": self.col_offset,
            "type": self.violation_type,
            "message": self.message,
            "source": self.source_line,
        }

    def __str__(self) -> str:
        s = f"[VIOLATION] {self.filename}:{self.lineno}:{self.col_offset} [{self.violation_type}] {self.message}"
        if self.source_line:
            s += f"\n    >>> {self.source_line}"
        return s


def get_target_names(target: ast.AST) -> List[str]:
    """Extract all variable names targeted by an assignment."""
    if isinstance(target, ast.Name):
        return [target.id]
    elif isinstance(target, (ast.Tuple, ast.List)):
        names: List[str] = []
        for elt in target.elts:
            names.extend(get_target_names(elt))
        return names
    return []


class CodebaseIntegrityVisitor(ast.NodeVisitor):
    def __init__(self, filename: str, lines: List[str]):
        self.filename = filename
        self.lines = lines
        self.violations: List[Violation] = []
        self.const_vars: Dict[str, Tuple[int, str]] = {}
        self.dummy_loop_vars: Set[str] = set()
        self.in_function = False
        self.in_loop = False

    def get_source(self, lineno: int) -> str:
        if 1 <= lineno <= len(self.lines):
            return self.lines[lineno - 1].strip()
        return ""

    def add_violation(self, node: ast.AST, violation_type: str, message: str) -> None:
        lineno = getattr(node, "lineno", 0)
        col_offset = getattr(node, "col_offset", 0)
        src = self.get_source(lineno)
        self.violations.append(
            Violation(
                filename=self.filename,
                lineno=lineno,
                col_offset=col_offset,
                violation_type=violation_type,
                message=message,
                source_line=src,
            )
        )

    def is_constant_expr(self, node: ast.AST) -> bool:
        """Check if an expression AST is statically composed solely of constants."""
        if isinstance(node, ast.Constant):
            return True
        if isinstance(node, ast.BinOp):
            return self.is_constant_expr(node.left) and self.is_constant_expr(node.right)
        if isinstance(node, ast.UnaryOp):
            return self.is_constant_expr(node.operand)
        return False

    def is_const_or_propagated(self, node: ast.AST) -> bool:
        """Check if a node is a constant expression or a tracked constant variable."""
        if self.is_constant_expr(node):
            return True
        if isinstance(node, ast.Name) and node.id in self.const_vars:
            return True
        if isinstance(node, ast.BinOp):
            return self.is_const_or_propagated(node.left) and self.is_const_or_propagated(node.right)
        if isinstance(node, ast.UnaryOp):
            return self.is_const_or_propagated(node.operand)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"abs", "round", "int", "float", "min", "max"}:
            return all(self.is_const_or_propagated(a) for a in node.args)
        return False

    def check_banned_multipliers(self, node: ast.AST) -> None:
        """Check for hardcoded latency scaling multipliers in binary operations."""
        for subnode in ast.walk(node):
            if isinstance(subnode, ast.BinOp) and isinstance(subnode.op, ast.Mult):
                if isinstance(subnode.right, ast.Constant) and isinstance(subnode.right.value, (int, float)):
                    if float(subnode.right.value) in BANNED_MULTIPLIERS:
                        self.add_violation(
                            subnode,
                            "ARTIFICIAL_MULTIPLIER",
                            f"Hardcoded latency scaling multiplier {subnode.right.value}",
                        )
                elif isinstance(subnode.left, ast.Constant) and isinstance(subnode.left.value, (int, float)):
                    if float(subnode.left.value) in BANNED_MULTIPLIERS:
                        self.add_violation(
                            subnode,
                            "ARTIFICIAL_MULTIPLIER",
                            f"Hardcoded latency scaling multiplier {subnode.left.value}",
                        )

    def visit_Module(self, node: ast.Module) -> None:
        nonlocals: Set[str] = set()
        self._check_statements_block(node.body, nonlocals)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function_scope(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function_scope(node)

    def _visit_function_scope(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        saved_consts = self.const_vars.copy()
        saved_loop_vars = self.dummy_loop_vars.copy()
        prev_in_function = self.in_function
        self.in_function = True
        try:
            nonlocals: Set[str] = set()
            for subnode in ast.walk(node):
                if isinstance(subnode, (ast.Nonlocal, ast.Global)):
                    nonlocals.update(subnode.names)

            self._check_statements_block(node.body, nonlocals)
        finally:
            self.in_function = prev_in_function
            self.const_vars = saved_consts
            self.dummy_loop_vars = saved_loop_vars

    def _check_statements_block(self, stmts: List[ast.stmt], nonlocals: Set[str]) -> None:
        for stmt in stmts:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.visit(stmt)
            elif isinstance(stmt, (ast.For, ast.While)):
                self._check_loop(stmt, nonlocals)
            elif isinstance(stmt, ast.If):
                self._check_if(stmt, nonlocals)
            elif isinstance(stmt, ast.Try):
                self._check_statements_block(stmt.body, nonlocals)
                for handler in stmt.handlers:
                    self._check_statements_block(handler.body, nonlocals)
                self._check_statements_block(stmt.orelse, nonlocals)
                self._check_statements_block(stmt.finalbody, nonlocals)
            elif isinstance(stmt, (ast.With, ast.AsyncWith)):
                self._check_statements_block(stmt.body, nonlocals)
            elif isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                self._check_assignment(stmt, nonlocals)
            elif isinstance(stmt, ast.Assert):
                self._check_assert_node(stmt)
            elif isinstance(stmt, ast.Expr):
                self.check_banned_multipliers(stmt.value)

    def _check_loop(self, stmt: ast.For | ast.While, nonlocals: Set[str]) -> None:
        prev_loop = self.in_loop
        self.in_loop = True
        try:
            for sub in stmt.body:
                if isinstance(sub, ast.Assert):
                    if isinstance(sub.test, ast.Constant) and bool(sub.test.value) is True:
                        self.add_violation(
                            sub,
                            "DUMMY_LOOP_ASSERT",
                            f"Asserting literal constant {sub.test.value} in loop",
                        )
                    elif isinstance(sub.test, ast.Compare):
                        left = sub.test.left
                        left_is_const = self.is_const_or_propagated(left)
                        comps_const = all(self.is_const_or_propagated(c) for c in sub.test.comparators)
                        if left_is_const and comps_const:
                            self.add_violation(
                                sub,
                                "DUMMY_LOOP_ASSERT",
                                f"Asserting constant in dummy loop: {ast.unparse(sub.test)}",
                            )
            self._check_statements_block(stmt.body, nonlocals)
            if stmt.orelse:
                self._check_statements_block(stmt.orelse, nonlocals)
        finally:
            self.in_loop = prev_loop

    def _check_if(self, stmt: ast.If, nonlocals: Set[str]) -> None:
        test_src = ast.unparse(stmt.test).lower()
        is_device_check = any(k in test_src for k in ["cuda", "device", "gpu", "t4_kernels"])
        for ostmt in stmt.orelse:
            if isinstance(ostmt, ast.Assert):
                if isinstance(ostmt.test, ast.Constant) and bool(ostmt.test.value) is True:
                    self.add_violation(
                        ostmt,
                        "MOCK_CUDA_PASS",
                        "Tautological 'else: assert True' in conditional",
                    )
                elif is_device_check and isinstance(ostmt.test, ast.Compare):
                    comp_src = ast.unparse(ostmt.test).lower()
                    if "cpu" in comp_src:
                        self.add_violation(
                            ostmt,
                            "MOCK_CUDA_PASS",
                            f"Trivial CPU assertion in device conditional else-branch: {ast.unparse(ostmt.test)}",
                        )

        self._check_statements_block(stmt.body, nonlocals)
        if stmt.orelse:
            self._check_statements_block(stmt.orelse, nonlocals)

    def _check_assignment(self, stmt: ast.Assign | ast.AnnAssign | ast.AugAssign, nonlocals: Set[str]) -> None:
        if isinstance(stmt, ast.Assign):
            targs: List[str] = []
            for t in stmt.targets:
                targs.extend(get_target_names(t))

            for name in targs:
                if name in BANNED_VARIABLES:
                    self.add_violation(
                        stmt,
                        "BANNED_TIMING_VAR",
                        f"Assignment to banned variable '{name}'",
                    )

            self.check_banned_multipliers(stmt.value)

            if self.is_const_or_propagated(stmt.value) and not any(t in nonlocals for t in targs):
                for name in targs:
                    self.const_vars[name] = (stmt.lineno, ast.unparse(stmt.value))
            else:
                for name in targs:
                    self.const_vars.pop(name, None)

        elif isinstance(stmt, ast.AnnAssign):
            targs = get_target_names(stmt.target)
            for name in targs:
                if name in BANNED_VARIABLES:
                    self.add_violation(
                        stmt,
                        "BANNED_TIMING_VAR",
                        f"Annotated assignment to banned variable '{name}'",
                    )
            if stmt.value:
                self.check_banned_multipliers(stmt.value)
                if self.is_const_or_propagated(stmt.value) and not any(t in nonlocals for t in targs):
                    for name in targs:
                        self.const_vars[name] = (stmt.lineno, ast.unparse(stmt.value))
                else:
                    for name in targs:
                        self.const_vars.pop(name, None)

        elif isinstance(stmt, ast.AugAssign):
            targs = get_target_names(stmt.target)
            for name in targs:
                if name in BANNED_VARIABLES:
                    self.add_violation(
                        stmt,
                        "BANNED_TIMING_VAR",
                        f"Augmented assignment to banned variable '{name}'",
                    )
                if self.in_loop and isinstance(stmt.op, ast.Add) and self.is_const_or_propagated(stmt.value):
                    self.dummy_loop_vars.add(name)
                self.const_vars.pop(name, None)
            self.check_banned_multipliers(stmt.value)

    def _check_assert_node(self, stmt: ast.Assert) -> None:
        # Literal truthy asserts: assert True, assert 1
        if isinstance(stmt.test, ast.Constant) and bool(stmt.test.value) is True:
            self.add_violation(
                stmt,
                "TAUTOLOGY_LITERAL",
                f"Tautological assertion: 'assert {stmt.test.value}'",
            )
            return

        if isinstance(stmt.test, ast.UnaryOp) and isinstance(stmt.test.op, ast.Not):
            if isinstance(stmt.test.operand, ast.Constant) and not bool(stmt.test.operand.value):
                self.add_violation(
                    stmt,
                    "TAUTOLOGY_LITERAL",
                    f"Tautological assertion: 'assert not {stmt.test.operand.value}'",
                )
                return

        # Dummy loop variable assertions (e.g. steps == 100)
        if isinstance(stmt.test, ast.Compare):
            left_name = stmt.test.left.id if isinstance(stmt.test.left, ast.Name) else None
            if left_name and left_name in self.dummy_loop_vars:
                if all(self.is_const_or_propagated(c) for c in stmt.test.comparators):
                    self.add_violation(
                        stmt,
                        "DUMMY_LOOP_ASSERT",
                        f"Asserting dummy loop iteration counter: {ast.unparse(stmt.test)}",
                    )
                    return

            left_src = ast.unparse(stmt.test.left)
            for comp in stmt.test.comparators:
                comp_src = ast.unparse(comp)
                if left_src == comp_src:
                    self.add_violation(
                        stmt,
                        "TAUTOLOGY_IDENTITY",
                        f"Comparison of identical operands: {left_src} vs {comp_src}",
                    )
                    return

            left_is_const = self.is_const_or_propagated(stmt.test.left)
            comps_const = [self.is_const_or_propagated(comp) for comp in stmt.test.comparators]

            if left_is_const and all(comps_const):
                self.add_violation(
                    stmt,
                    "TAUTOLOGY_CONSTANT_PROPAGATED",
                    f"Asserting constant comparison without dynamic measurement: {ast.unparse(stmt.test)}",
                )

        # Function calls inside asserts (e.g. torch.equal(x, x))
        if isinstance(stmt.test, ast.Call):
            func_src = ast.unparse(stmt.test.func)
            if func_src in {"torch.equal", "torch.allclose"} and len(stmt.test.args) >= 2:
                arg0 = ast.unparse(stmt.test.args[0])
                arg1 = ast.unparse(stmt.test.args[1])
                if arg0 == arg1:
                    self.add_violation(
                        stmt,
                        "TAUTOLOGY_IDENTITY",
                        f"Assertion of {func_src} with identical arguments: {arg0} vs {arg1}",
                    )


def scan_file(filepath: str, verbose: bool = False) -> List[Violation]:
    """Scan a single Python file for AST integrity violations."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        if verbose:
            print(f"[WARN] Failed to read {filepath}: {e}", file=sys.stderr)
        return []

    try:
        tree = ast.parse(content, filename=filepath)
    except SyntaxError as e:
        return [
            Violation(
                filename=filepath,
                lineno=e.lineno or 1,
                col_offset=e.offset or 0,
                violation_type="SYNTAX_ERROR",
                message=f"Syntax error during parsing: {e.msg}",
                source_line=e.text.strip() if e.text else "",
            )
        ]

    lines = content.splitlines()
    visitor = CodebaseIntegrityVisitor(filename=filepath, lines=lines)
    visitor.visit(tree)
    return visitor.violations


def find_python_files(
    dirs: List[str],
    exclude_patterns: Optional[List[str]] = None,
) -> List[str]:
    """Discover all python files within the target directories, applying exclusions."""
    py_files: List[str] = []
    exclude_patterns = exclude_patterns or []

    for d in dirs:
        if os.path.isfile(d):
            if d.endswith(".py"):
                py_files.append(os.path.normpath(d))
            continue

        for root, dirs_list, files in os.walk(d):
            dirs_list[:] = [
                sub for sub in dirs_list
                if not sub.startswith(".") and sub not in {"__pycache__", "build", "dist"}
            ]
            for file in files:
                if file.endswith(".py"):
                    full_path = os.path.normpath(os.path.join(root, file))
                    excluded = False
                    for pattern in exclude_patterns:
                        if fnmatch.fnmatch(full_path, pattern) or fnmatch.fnmatch(os.path.basename(full_path), pattern):
                            excluded = True
                            break
                    if not excluded:
                        py_files.append(full_path)

    return sorted(list(set(py_files)))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Programmatic AST Verification Tool for Codebase Integrity."
    )
    parser.add_argument(
        "--dirs",
        nargs="+",
        default=["src", "tests", "benchmarks"],
        help="Directories or files to scan for AST integrity.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output of scanned files.",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=[],
        help="Glob patterns to exclude from scanning.",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Optional path to output machine-readable JSON violations.",
    )

    args = parser.parse_args()

    files = find_python_files(args.dirs, args.exclude)
    if not files:
        print(f"[WARN] No Python files found in specified paths: {args.dirs}")
        return 0

    print("=" * 80)
    print("  T4-CUDA CODEBASE AST INTEGRITY SCANNER")
    print("=" * 80)
    print(f"Directories scanned : {args.dirs}")
    print(f"Excluded patterns   : {args.exclude}")
    print(f"Total files located : {len(files)}")
    print("-" * 80)

    all_violations: List[Violation] = []
    files_with_violations: Set[str] = set()

    for filepath in files:
        if args.verbose:
            print(f"Scanning: {filepath}")
        violations = scan_file(filepath, verbose=args.verbose)
        if violations:
            files_with_violations.add(filepath)
            all_violations.extend(violations)
            for v in violations:
                print(str(v))

    print("=" * 80)
    print("  INTEGRITY VERIFICATION SUMMARY")
    print("=" * 80)
    print(f"Files scanned            : {len(files)}")
    print(f"Clean files              : {len(files) - len(files_with_violations)}")
    print(f"Files with violations    : {len(files_with_violations)}")
    print(f"Total violations flagged : {len(all_violations)}")

    if args.json:
        report_data = {
            "total_files": len(files),
            "clean_files": len(files) - len(files_with_violations),
            "total_violations": len(all_violations),
            "violations": [v.to_dict() for v in all_violations],
        }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        print(f"JSON report saved to     : {args.json}")

    print("=" * 80)

    if all_violations:
        print(f"[FAIL] {len(all_violations)} integrity violation(s) detected. Fix required.")
        return 1
    else:
        print("[SUCCESS] 0 AST integrity violations detected. Codebase integrity verified.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
