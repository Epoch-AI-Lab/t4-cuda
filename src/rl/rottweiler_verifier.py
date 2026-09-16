"""Rottweiler Verification: 5-Stage XML Scaffold and Sandboxed SymPy Verifier (Requirement R5).

Enforces:
1. 5-stage XML scaffold (<explore>, <conjecture>, <test_edge_cases>, <lemma_isolate>, <formal_proof>)
   with strict sequence, zero nesting, non-empty blocks, and terminal \\boxed{answer} inside <formal_proof>.
2. AST-sandboxed SymPy execution verifier testing symbolic equivalence and reverse equation substitution.
"""

from typing import Dict, Any, Optional, Tuple, List, Union
import ast
import math
import re
import signal
import sys
import threading
import sympy


TAG_NAMES = [
    "explore",
    "conjecture",
    "test_edge_cases",
    "lemma_isolate",
    "formal_proof"
]

SAFE_AST_NODES = {
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Pow, ast.USub, ast.UAdd,
    ast.Name, ast.Call, ast.Tuple, ast.List, ast.Load, ast.Compare, ast.Eq
}

ALLOWED_MATH_FUNCS = {
    "sqrt", "pi", "sin", "cos", "tan", "log", "exp", "abs",
    "Rational", "Integer", "Matrix", "frac", "Symbol", "factorial", "oo"
}

BANNED_NAMES = {
    "eval", "exec", "open", "__import__", "compile", "globals", "locals",
    "getattr", "setattr", "delattr", "hasattr", "input", "print", "breakpoint",
    "help", "exit", "quit", "system", "popen", "spawn", "os", "sys", "subprocess", "pathlib",
    "shutil", "time", "socket", "urllib", "requests"
}


class TimeoutError(Exception):
    """Raised when symbolic verification exceeds execution timeout."""
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Execution timed out")


def run_with_timeout(fn, args=(), kwargs=None, timeout_seconds: float = 5.0):
    """Executes a function with POSIX signal alarm or direct call."""
    if kwargs is None:
        kwargs = {}

    is_main_thread = (threading.current_thread() is threading.main_thread())
    if is_main_thread and hasattr(signal, "SIGALRM"):
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(int(math.ceil(timeout_seconds)))
        try:
            return fn(*args, **kwargs)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        # Fallback in non-main thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(fn, *args, **kwargs)
            return future.result(timeout=timeout_seconds)


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extracts contents of terminal \\boxed{...} handling nested curly braces."""
    patterns = [r"\boxed{", "boxed{"]
    idx = -1
    lead_len = 0
    for pat in patterns:
        cur_idx = text.rfind(pat)
        if cur_idx > idx:
            idx = cur_idx
            lead_len = len(pat)

    if idx == -1:
        return None

    idx += lead_len
    depth = 1
    chars = []
    for c in text[idx:]:
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                break
        chars.append(c)

    if depth == 0:
        return "".join(chars).strip()
    return None


def clean_latex_math(expr: str, is_equation: bool = False) -> str:
    """Sanitizes LaTeX mathematical expressions for SymPy evaluation."""
    s = expr.strip()
    s = s.replace("$", "")
    s = s.replace(r"\(", "").replace(r"\)", "")
    s = s.replace(r"\[", "").replace(r"\]", "")

    # Convert exponentiation ^ to Python **
    s = s.replace("^", "**")

    # If expression is written as variable assignment (e.g., "x = 42") and is NOT an equation to verify, isolate RHS
    if not is_equation:
        m_eq = re.match(r"^[a-zA-Z]\s*=\s*(.+)$", s)
        if m_eq:
            s = m_eq.group(1).strip()

    # Normalize nested fractions: \frac{num}{den} and \dfrac{num}{den} -> ((num)/(den))
    for frac_cmd in [r"\frac{", r"\dfrac{"]:
        while frac_cmd in s:
            idx = s.find(frac_cmd)
            start_num = idx + len(frac_cmd) - 1
            depth = 0
            end_num = -1
            for i in range(start_num, len(s)):
                if s[i] == "{":
                    depth += 1
                elif s[i] == "}":
                    depth -= 1
                    if depth == 0:
                        end_num = i
                        break
            if end_num == -1:
                break

            # Handle possible whitespace between {num} and {den} in LaTeX
            rest = s[end_num + 1:]
            lstripped = rest.lstrip()
            if not lstripped.startswith("{"):
                break
            start_den = end_num + 1 + (len(rest) - len(lstripped))
            depth = 0
            end_den = -1
            for i in range(start_den, len(s)):
                if s[i] == "{":
                    depth += 1
                elif s[i] == "}":
                    depth -= 1
                    if depth == 0:
                        end_den = i
                        break
            if end_den == -1:
                break
            num = s[start_num + 1:end_num]
            den = s[start_den + 1:end_den]
            s = s[:idx] + f"(({num})/({den}))" + s[end_den + 1:]

    # Normalize nested square roots: \sqrt{arg} -> sqrt(arg)
    while r"\sqrt{" in s:
        idx = s.find(r"\sqrt{")
        depth = 0
        end_idx = -1
        for i in range(idx + 5, len(s)):
            if s[i] == "{":
                depth += 1
            elif s[i] == "}":
                depth -= 1
                if depth == 0:
                    end_idx = i
                    break
        if end_idx != -1:
            arg = s[idx + 6:end_idx]
            s = s[:idx] + f"sqrt({arg})" + s[end_idx + 1:]
        else:
            break

    s = s.replace(r"\pi", "pi")
    s = s.replace(r"\cdot", "*").replace(r"\times", "*")
    s = re.sub(r"\\[,;! ]", "", s)
    s = s.replace(r"\left", "").replace(r"\right", "")

    # Normalize LaTeX mathematical functions
    for fn in [
        "arcsin", "arccos", "arctan",
        "sinh", "cosh", "tanh",
        "sin", "cos", "tan", "sec", "csc", "cot",
        "exp", "abs", "det"
    ]:
        s = s.replace(f"\\{fn}", fn)
    s = s.replace(r"\ln", "log")
    s = s.replace(r"\log", "log")

    # Resolve implicit multiplication
    s = re.sub(r"(\d+)\s*([a-zA-Z\(])", r"\1*\2", s)
    s = re.sub(r"(\))\s*(\()", r"\1*\2", s)
    s = re.sub(r"(\))\s*([a-zA-Z0-9])", r"\1*\2", s)

    return s.strip()


def is_safe_math_expression(expr_str: str) -> bool:
    """Verifies that an expression contains only safe mathematical syntax to prevent RCE."""
    # Split equations if present for checking
    if "=" in expr_str:
        parts = expr_str.split("=")
        return all(is_safe_math_expression(part.strip()) for part in parts if part.strip())

    try:
        tree = ast.parse(expr_str, mode="eval")
    except Exception:
        return False

    for node in ast.walk(tree):
        if type(node) not in SAFE_AST_NODES:
            return False
        if isinstance(node, ast.Name):
            if node.id.startswith("_") or node.id in BANNED_NAMES:
                return False
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                return False
            if node.func.id not in ALLOWED_MATH_FUNCS:
                return False

    return True


class XMLScaffoldValidator:
    """Validates 5-stage XML reasoning scaffold."""

    TAGS = TAG_NAMES

    @classmethod
    def extract_tags(cls, text: str) -> Dict[str, Optional[str]]:
        """Extracts text content within each tag."""
        extracted = {}
        for tag in cls.TAGS:
            open_marker = f"<{tag}>"
            close_marker = f"</{tag}>"
            if open_marker in text and close_marker in text:
                start = text.find(open_marker) + len(open_marker)
                end = text.find(close_marker)
                if start <= end:
                    extracted[tag] = text[start:end].strip()
                else:
                    extracted[tag] = None
            else:
                extracted[tag] = None
        return extracted

    @classmethod
    def validate(cls, text: str) -> Dict[str, Any]:
        """Validates scaffold structure, tag ordering, nesting, and boxed answer placement.

        Returns:
            Dict containing detailed adherence report.
        """
        tags_present: Dict[str, bool] = {}
        error_messages: List[str] = []

        # 1. Exact count of open and close tags
        for tag in cls.TAGS:
            o_cnt = text.count(f"<{tag}>")
            c_cnt = text.count(f"</{tag}>")
            tags_present[tag] = (o_cnt == 1 and c_cnt == 1)
            if o_cnt != 1 or c_cnt != 1:
                error_messages.append(f"Tag <{tag}> count mismatch (open={o_cnt}, close={c_cnt})")

        all_tags_present = all(tags_present.values())
        if not all_tags_present:
            return {
                "adherent": False,
                "all_tags_present": False,
                "strictly_ordered": False,
                "zero_nesting": False,
                "boxed_in_formal_proof": False,
                "no_trailing_text": False,
                "extracted_tags": cls.extract_tags(text),
                "boxed_answer": extract_boxed_answer(text),
                "error_message": "; ".join(error_messages)
            }

        # 2. Strict sequential order and zero nesting
        strictly_ordered = True
        zero_nesting = True
        last_close_pos = -1
        extracted_tags: Dict[str, str] = {}

        for tag in cls.TAGS:
            open_pos = text.find(f"<{tag}>")
            close_pos = text.find(f"</{tag}>")

            if not (last_close_pos <= open_pos < close_pos):
                strictly_ordered = False
                error_messages.append(f"Out of order sequence around <{tag}>")
                break

            content = text[open_pos + len(f"<{tag}>"):close_pos]
            if len(content.strip()) < 5:
                error_messages.append(f"Block <{tag}> is empty or non-substantive")

            # Check for any other tag inside content
            for other_tag in cls.TAGS:
                if f"<{other_tag}>" in content or f"</{other_tag}>" in content:
                    zero_nesting = False
                    error_messages.append(f"Nesting detected inside <{tag}> with <{other_tag}>")
                    break

            extracted_tags[tag] = content.strip()
            last_close_pos = close_pos + len(f"</{tag}>")

        # 3. Terminal boxed answer inside formal_proof
        formal_proof_content = extracted_tags.get("formal_proof", "")
        boxed_inside_formal = extract_boxed_answer(formal_proof_content) is not None
        if not boxed_inside_formal:
            error_messages.append("No \\boxed{...} found inside <formal_proof>")

        # 4. No reasoning text following </formal_proof>
        trailing_text = text[last_close_pos:].strip()
        no_trailing_text = (len(trailing_text) == 0)
        if not no_trailing_text:
            error_messages.append("Trailing reasoning text found after </formal_proof>")

        is_adherent = (
            all_tags_present
            and strictly_ordered
            and zero_nesting
            and boxed_inside_formal
            and no_trailing_text
            and len(error_messages) == 0
        )

        return {
            "adherent": is_adherent,
            "all_tags_present": all_tags_present,
            "strictly_ordered": strictly_ordered,
            "zero_nesting": zero_nesting,
            "boxed_in_formal_proof": boxed_inside_formal,
            "no_trailing_text": no_trailing_text,
            "extracted_tags": extracted_tags,
            "boxed_answer": extract_boxed_answer(formal_proof_content) if boxed_inside_formal else None,
            "error_message": "; ".join(error_messages) if error_messages else None
        }


class RottweilerSymPyVerifier:
    """Sandboxed symbolic verifier checking exact equivalence and reverse substitution."""

    def __init__(self, timeout_seconds: float = 5.0):
        self.timeout_seconds = timeout_seconds

    def clean_latex(self, expr_str: str, is_equation: bool = False) -> str:
        """Sanitizes LaTeX expression."""
        return clean_latex_math(expr_str, is_equation=is_equation)

    def is_safe_expression(self, expr_str: str) -> bool:
        """Validates AST safety to prevent code execution."""
        return is_safe_math_expression(expr_str)

    def verify_symbolic_equivalence(self, pred_raw: Optional[str], gold_raw: str) -> bool:
        """Tests mathematical equivalence between predicted answer and ground truth."""
        if not pred_raw or not gold_raw:
            return False

        pred_clean = self.clean_latex(pred_raw)
        gold_clean = self.clean_latex(gold_raw)

        # 1. Exact string match
        if pred_clean.lower() == gold_clean.lower():
            return True

        # 2. Strict numeric match for single scalars
        num_pattern = r"^[+-]?\d+(?:\.\d+)?$"
        if re.fullmatch(num_pattern, pred_clean) and re.fullmatch(num_pattern, gold_clean):
            try:
                return abs(float(pred_clean) - float(gold_clean)) < 1e-6
            except Exception:
                pass

        # 3. AST sandboxing check
        if not self.is_safe_expression(pred_clean) or not self.is_safe_expression(gold_clean):
            return False

        def _evaluate():
            try:
                sym_pred = sympy.sympify(pred_clean)
                sym_gold = sympy.sympify(gold_clean)
                diff = sympy.simplify(sym_pred - sym_gold)
                if diff == 0 or diff.is_zero:
                    return True
                if diff.is_number and abs(complex(diff.evalf())) < 1e-6:
                    return True
            except Exception:
                return False
            return False

        try:
            return run_with_timeout(_evaluate, timeout_seconds=self.timeout_seconds)
        except (TimeoutError, Exception):
            return False

    def verify_reverse_substitution(
        self,
        candidate_ans: str,
        equation_str: str,
        variable_name: str = "x",
        domain_constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Substitutes candidate answer into equation and verifies residual.

        Detects extraneous roots from division-by-zero denominators and evaluates domain constraints.
        """
        cand_clean = self.clean_latex(candidate_ans, is_equation=False)
        eq_clean = self.clean_latex(equation_str, is_equation=True)

        if not self.is_safe_expression(cand_clean) or not self.is_safe_expression(eq_clean):
            return {
                "valid": False,
                "residual": None,
                "domain_valid": False,
                "error_message": "AST safety check failed for expression or candidate"
            }

        def _evaluate_sub():
            # Parse LHS and RHS
            if "=" in eq_clean:
                parts = eq_clean.split("=")
                lhs_str, rhs_str = parts[0].strip(), parts[1].strip()
            else:
                lhs_str, rhs_str = eq_clean.strip(), "0"

            var = sympy.Symbol(variable_name)
            x_val = sympy.sympify(cand_clean)
            lhs_expr = sympy.sympify(lhs_str)
            rhs_expr = sympy.sympify(rhs_str)

            # Check for division by zero / extraneous roots
            # Check powers with negative exponents or denominators
            for expr in [lhs_expr, rhs_expr]:
                for p in expr.atoms(sympy.core.power.Pow):
                    if p.exp.is_negative:
                        denom_val = sympy.simplify(p.base.subs(var, x_val))
                        if denom_val == 0 or denom_val.is_zero:
                            return {
                                "valid": False,
                                "residual": None,
                                "domain_valid": False,
                                "error_message": f"Extraneous root: denominator {p.base} evaluates to zero"
                            }

            lhs_sub = lhs_expr.subs(var, x_val)
            rhs_sub = rhs_expr.subs(var, x_val)

            # Check if substitution produced zoo or nan
            if lhs_sub.has(sympy.zoo, sympy.nan, sympy.oo) or rhs_sub.has(sympy.zoo, sympy.nan, sympy.oo):
                return {
                    "valid": False,
                    "residual": None,
                    "domain_valid": False,
                    "error_message": "Substitution produced infinite or undefined value"
                }

            # Evaluate domain constraints
            if domain_constraints:
                if domain_constraints.get("integer", False):
                    if not (x_val.is_integer or (x_val.is_number and float(x_val).is_integer())):
                        return {
                            "valid": False,
                            "residual": None,
                            "domain_valid": False,
                            "error_message": f"Candidate {cand_clean} does not satisfy integer constraint"
                        }
                if domain_constraints.get("positive", False):
                    if x_val.is_number and float(x_val) <= 0:
                        return {
                            "valid": False,
                            "residual": None,
                            "domain_valid": False,
                            "error_message": f"Candidate {cand_clean} does not satisfy positive constraint"
                        }

            # Check residual
            residual = sympy.simplify(lhs_sub - rhs_sub)
            is_zero = (residual == 0 or residual.is_zero)
            if not is_zero and residual.is_number:
                is_zero = (abs(complex(residual.evalf())) < 1e-6)

            return {
                "valid": bool(is_zero),
                "residual": str(residual),
                "domain_valid": True,
                "error_message": None if is_zero else f"Non-zero residual: {residual}"
            }

        try:
            return run_with_timeout(_evaluate_sub, timeout_seconds=self.timeout_seconds)
        except TimeoutError:
            return {
                "valid": False,
                "residual": None,
                "domain_valid": False,
                "error_message": f"Timeout of {self.timeout_seconds}s exceeded during substitution"
            }
        except Exception as e:
            return {
                "valid": False,
                "residual": None,
                "domain_valid": False,
                "error_message": str(e)
            }


class RottweilerVerifier:
    """Unified Rottweiler Verifier orchestrating XML scaffold and SymPy verification."""

    def __init__(self, timeout_seconds: float = 5.0):
        self.scaffold_validator = XMLScaffoldValidator()
        self.sympy_verifier = RottweilerSymPyVerifier(timeout_seconds=timeout_seconds)

    def verify_xml_scaffold(self, text: str) -> Tuple[bool, str]:
        """Validates XML scaffold adherence."""
        report = self.scaffold_validator.validate(text)
        return report["adherent"], report.get("error_message") or ""

    def verify_sympy_solution(
        self,
        prediction: str,
        ground_truth: str,
        equation_str: Optional[str] = None
    ) -> Tuple[bool, str]:
        """Verifies solution equivalence and optional reverse substitution."""
        equiv = self.sympy_verifier.verify_symbolic_equivalence(prediction, ground_truth)
        if not equiv:
            return False, "Symbolic equivalence check failed"

        if equation_str:
            sub_res = self.sympy_verifier.verify_reverse_substitution(prediction, equation_str)
            if not sub_res["valid"]:
                return False, sub_res.get("error_message") or "Reverse substitution failed"

        return True, ""

    def verify(
        self,
        completion: str,
        ground_truth: str,
        equation_str: Optional[str] = None
    ) -> Dict[str, Any]:
        """Full verification pipeline returning comprehensive report."""
        scaffold_rep = self.scaffold_validator.validate(completion)
        boxed_ans = scaffold_rep.get("boxed_answer")

        if not scaffold_rep["adherent"] or not boxed_ans:
            return {
                "valid": False,
                "scaffold_adherent": scaffold_rep["adherent"],
                "symbolic_correct": False,
                "boxed_answer": boxed_ans,
                "error": scaffold_rep.get("error_message") or "Scaffold violation"
            }

        sym_valid, sym_err = self.verify_sympy_solution(
            boxed_ans, ground_truth, equation_str=equation_str
        )

        return {
            "valid": sym_valid,
            "scaffold_adherent": scaffold_rep["adherent"],
            "symbolic_correct": sym_valid,
            "boxed_answer": boxed_ans,
            "error": sym_err if not sym_valid else None
        }

    def verify_completion(
        self,
        completion: str,
        ground_truth: str,
        equation_str: Optional[str] = None
    ) -> Dict[str, Any]:
        """Alias for verify pipeline."""
        return self.verify(completion, ground_truth, equation_str=equation_str)
