"""Calibrated Abstention Reward Engine.

Implements asymmetric rewards calibrated so guessing with under 60% confidence
yields a negative expected value:
- Correct symbolic answer: +1.0
- Explicit refusal / <abstain>: 0.0
- Incorrect or hallucinated answer: -1.5
- Hedging penalty (both <abstain> and candidate boxed answer): -1.5

Includes AST-sandboxed SymPy equivalence check with timeout guards.
"""

import ast
import concurrent.futures
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import sympy
import torch


SAFE_AST_NODES = {
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Pow, ast.Mod,
    ast.USub, ast.UAdd,
    ast.Name, ast.Call, ast.Tuple, ast.List, ast.Load
}

ALLOWED_MATH_FUNCS = {
    "sqrt", "pi", "sin", "cos", "tan", "log", "exp", "abs",
    "Rational", "Integer", "Pow", "asin", "acos", "atan",
    "sinh", "cosh", "tanh", "factorial"
}

BANNED_NAMES = {
    "eval", "exec", "open", "__import__", "compile", "globals", "locals",
    "getattr", "setattr", "delattr", "hasattr", "system", "popen", "spawn",
    "os", "sys", "subprocess", "shutil", "builtins", "__builtins__",
    "breakpoint", "help", "input", "print", "quit", "exit"
}

ABSTAIN_SIGNALS = [
    "<abstain>",
    "</abstain>",
    r"\boxed{abstain}",
    r"\boxed{\text{abstain}}",
    r"\boxed{i don't know}",
    r"\boxed{unknown}",
]

ABSTAIN_BOX_VALUES = {
    "abstain",
    r"\text{abstain}",
    "i don't know",
    "unknown",
    "<abstain>",
}


class CalibratedAbstentionRewardEngine:
    """Asymmetric reward engine with 60% confidence calibration and AST sandboxing.

    Args:
        r_correct: Reward for mathematically correct symbolic answer (default +1.0).
        r_abstain: Reward for explicit honest abstention (default 0.0).
        r_incorrect: Penalty for wrong or hallucinated answer (default -1.5).
        r_hedge: Penalty when model outputs both abstain and a guess (default -1.5).
        timeout_seconds: Timeout limit for SymPy symbolic simplification.
    """

    def __init__(
        self,
        r_correct: float = 1.0,
        r_abstain: float = 0.0,
        r_incorrect: float = -1.5,
        r_hedge: float = -1.5,
        timeout_seconds: float = 2.0,
    ):
        self.r_correct = float(r_correct)
        self.r_abstain = float(r_abstain)
        self.r_incorrect = float(r_incorrect)
        self.r_hedge = float(r_hedge)
        self.timeout_seconds = float(timeout_seconds)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    def __del__(self):
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass

    def expected_value_guessing(self, p: float) -> float:
        """Calculate expected reward of guessing at confidence level p.

        Formula:
            E[R | guess] = p * r_correct + (1 - p) * r_incorrect
            With r_correct = 1.0 and r_incorrect = -1.5:
            E[R | guess] = 2.5 * p - 1.5
        """
        return p * self.r_correct + (1.0 - p) * self.r_incorrect

    def breakeven_confidence(self) -> float:
        """Calculate confidence threshold where guessing equals abstaining.

        Formula:
            p* = (r_abstain - r_incorrect) / (r_correct - r_incorrect)
            p* = (0.0 - (-1.5)) / (1.0 - (-1.5)) = 1.5 / 2.5 = 0.60
        """
        return (self.r_abstain - self.r_incorrect) / (self.r_correct - self.r_incorrect)

    def verify_calibration(self) -> Dict[str, Any]:
        """Mathematically verify the 60% confidence calibration curve.

        Returns:
            Dictionary with calibration test points and verification status.
        """
        p_star = self.breakeven_confidence()
        assert abs(p_star - 0.60) < 1e-9, f"Expected breakeven at 0.60, got {p_star}"

        test_points = {}
        for pct in range(0, 101, 5):
            p = round(pct / 100.0, 2)
            ev = self.expected_value_guessing(p)
            expected_formula = 2.5 * p - 1.5
            assert abs(ev - expected_formula) < 1e-9, f"EV mismatch at p={p}"

            if p < 0.60:
                assert ev < self.r_abstain, f"At p={p} (<0.60), guessing must have negative EV vs abstaining"
            elif p > 0.60:
                assert ev > self.r_abstain, f"At p={p} (>0.60), guessing must have positive EV vs abstaining"
            else:
                assert abs(ev - self.r_abstain) < 1e-9, "At p=0.60, EV must equal abstention reward"

            test_points[str(p)] = ev

        return {
            "breakeven_confidence": p_star,
            "is_calibrated": True,
            "test_points": test_points,
        }

    def extract_boxed_answer(self, text: str) -> Optional[str]:
        """Extract content from the last \\boxed{...} tag with nested brace parsing.

        Args:
            text: Completion text string.

        Returns:
            Extracted answer string, or None if no valid box is found.
        """
        idx = text.rfind(r"\\boxed{")
        if idx == -1:
            idx = text.rfind(r"\boxed{")
            if idx == -1:
                idx = text.rfind("boxed{")
                if idx == -1:
                    return None
                idx += len("boxed{")
            else:
                idx += len(r"\boxed{")
        else:
            idx += len(r"\\boxed{")

        depth = 1
        chars = []
        for c in text[idx:]:
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            chars.append(c)

        if depth == 0:
            extracted = "".join(chars).strip()
            return extracted if extracted else None
        return None

    def clean_latex(self, expr: str) -> str:
        """Sanitize LaTeX math expressions into standard Python math syntax.

        Args:
            expr: Raw LaTeX string.

        Returns:
            Sanitized expression string ready for AST parsing and SymPy.
        """
        s = expr.strip().replace("$", "")
        # Remove variable assignment prefix: x = 42 -> 42
        m = re.match(r"^[a-zA-Z]\s*=\s*(.+)$", s)
        if m:
            s = m.group(1).strip()

        # Remove LaTeX text formatting wrappers
        s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
        s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
        s = re.sub(r"\\mathbf\{([^}]*)\}", r"\1", s)

        # Handle \\frac{a}{b} and \\dfrac{a}{b} with nested braces
        for frac_pat in [r"\frac{", r"\dfrac{"]:
            while frac_pat in s:
                idx = s.find(frac_pat)
                depth = 0
                n_end = -1
                for i in range(idx + len(frac_pat) - 1, len(s)):
                    if s[i] == "{":
                        depth += 1
                    elif s[i] == "}":
                        depth -= 1
                        if depth == 0:
                            n_end = i
                            break
                if n_end == -1 or n_end + 1 >= len(s) or s[n_end + 1] != "{":
                    break
                depth = 0
                d_end = -1
                for i in range(n_end + 1, len(s)):
                    if s[i] == "{":
                        depth += 1
                    elif s[i] == "}":
                        depth -= 1
                        if depth == 0:
                            d_end = i
                            break
                if d_end == -1:
                    break
                num = s[idx + len(frac_pat):n_end]
                den = s[n_end + 2:d_end]
                s = s[:idx] + f"(({num})/({den}))" + s[d_end + 1:]

        # Handle \\sqrt{a}
        while r"\sqrt{" in s:
            idx = s.find(r"\sqrt{")
            depth = 0
            end = -1
            for i in range(idx + 5, len(s)):
                if s[i] == "{":
                    depth += 1
                elif s[i] == "}":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            if end != -1:
                arg = s[idx + 6:end]
                s = s[:idx] + f"sqrt({arg})" + s[end + 1:]
            else:
                break

        # Replace mathematical constants and symbols
        s = s.replace(r"\pi", "pi").replace(r"\cdot", "*").replace(r"\times", "*")
        s = re.sub(r"\\[,;! ]", "", s)
        s = s.replace(r"\left", "").replace(r"\right", "")
        s = s.replace("{", "(").replace("}", ")")
        s = s.replace("^", "**")

        # Insert multiplication operators for implicit multiplication
        s = re.sub(r"(\d+)\s*([a-zA-Z\(])", r"\1*\2", s)
        s = re.sub(r"(\))\s*(\()", r"\1*\2", s)
        s = re.sub(r"(\))\s*([a-zA-Z0-9])", r"\1*\2", s)

        return s.strip()

    def is_safe_ast(self, expr_str: str) -> bool:
        """Check expression against AST whitelist to prevent code injection.

        Args:
            expr_str: Python math string.

        Returns:
            True if expression contains only whitelisted nodes and math calls.
        """
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

    def verify_symbolic_equivalence(self, pred_raw: str, gold_raw: str) -> bool:
        """Verify mathematical equivalence using AST-sandboxed SymPy with timeout.

        Args:
            pred_raw: Extracted prediction string.
            gold_raw: Ground truth string.

        Returns:
            True if expressions are mathematically equivalent, False otherwise.
        """
        if not pred_raw or not gold_raw:
            return False

        pred_clean = self.clean_latex(pred_raw)
        gold_clean = self.clean_latex(gold_raw)

        # Exact normalized string match
        if pred_clean.lower() == gold_clean.lower():
            return True

        # Fast numeric float comparison
        num_re = r"^[+-]?\d+(?:\.\d+)?$"
        if re.fullmatch(num_re, pred_clean) and re.fullmatch(num_re, gold_clean):
            try:
                if abs(float(pred_clean) - float(gold_clean)) < 1e-6:
                    return True
            except Exception:
                pass

        # AST sandboxing check on both expressions
        if not self.is_safe_ast(pred_clean) or not self.is_safe_ast(gold_clean):
            return False

        # Symbolic difference check executed in thread pool with timeout
        def _eval_sympy() -> bool:
            try:
                p_sym = sympy.sympify(pred_clean)
                g_sym = sympy.sympify(gold_clean)

                # Check if difference simplifies to zero
                diff = sympy.simplify(p_sym - g_sym)
                if diff == 0 or getattr(diff, "is_zero", False):
                    return True

                # Check numerical evaluation of constant difference
                if getattr(diff, "is_number", False):
                    val = complex(diff.evalf())
                    if abs(val) < 1e-6:
                        return True

                # Rational simplification check
                rat_diff = sympy.ratsimp(p_sym - g_sym)
                if rat_diff == 0 or getattr(rat_diff, "is_zero", False):
                    return True

            except Exception:
                return False
            return False

        try:
            future = self._executor.submit(_eval_sympy)
            return bool(future.result(timeout=self.timeout_seconds))
        except (concurrent.futures.TimeoutError, Exception):
            return False

    def evaluate_completion(
        self,
        completion: str,
        ground_truth: str,
        is_unanswerable: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate completion and return detailed reward result dictionary.

        Args:
            completion: Model rollout text.
            ground_truth: Reference solution or <abstain>.
            is_unanswerable: Flag indicating whether problem is an impossible premise trap.

        Returns:
            Dictionary containing reward, status, extracted prediction, and flags.
        """
        comp_lower = completion.lower()
        has_abstain_signal = any(sig.lower() in comp_lower for sig in ABSTAIN_SIGNALS)
        boxed_pred = self.extract_boxed_answer(completion)

        # Check if the boxed answer itself is an abstention signal
        is_boxed_abstain = False
        if boxed_pred is not None:
            clean_box = boxed_pred.strip().lower()
            if clean_box in ABSTAIN_BOX_VALUES:
                is_boxed_abstain = True
                has_abstain_signal = True

        has_candidate_box = (boxed_pred is not None) and (not is_boxed_abstain)

        # Anti-hedging rule: penalty if completion contains both abstain and a candidate guess
        if has_abstain_signal and has_candidate_box:
            return {
                "reward": self.r_hedge,
                "status": "hedging_penalty",
                "pred": boxed_pred,
                "abstain": True,
                "is_hedging": True,
            }

        # Check if ground truth designates an unanswerable problem trap
        gold_lower = ground_truth.strip().lower()
        unanswerable_gold = is_unanswerable or gold_lower in {"<abstain>", "abstain", "unanswerable"}

        if unanswerable_gold:
            if has_abstain_signal and not has_candidate_box:
                return {
                    "reward": self.r_correct,
                    "status": "honest_abstention",
                    "pred": None,
                    "abstain": True,
                    "is_hedging": False,
                }
            return {
                "reward": self.r_incorrect,
                "status": "hallucinated_answer",
                "pred": boxed_pred,
                "abstain": False,
                "is_hedging": False,
            }

        # Standard problem with solvable ground truth
        if has_abstain_signal and not has_candidate_box:
            return {
                "reward": self.r_abstain,
                "status": "calibrated_abstention",
                "pred": None,
                "abstain": True,
                "is_hedging": False,
            }

        if not has_candidate_box:
            return {
                "reward": self.r_incorrect,
                "status": "missing_or_malformed_answer",
                "pred": None,
                "abstain": False,
                "is_hedging": False,
            }

        # Verify symbolic equivalence of candidate answer
        is_correct = self.verify_symbolic_equivalence(boxed_pred, ground_truth)
        if is_correct:
            return {
                "reward": self.r_correct,
                "status": "correct",
                "pred": boxed_pred,
                "abstain": False,
                "is_hedging": False,
            }

        return {
            "reward": self.r_incorrect,
            "status": "incorrect",
            "pred": boxed_pred,
            "abstain": False,
            "is_hedging": False,
        }

    def compute_reward(
        self,
        completion: str,
        ground_truth: str,
        is_unanswerable: bool = False,
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute calibrated scalar reward for a completion.

        Args:
            completion: Model rollout text.
            ground_truth: Reference solution or <abstain>.
            is_unanswerable: Flag indicating if problem is impossible to solve.

        Returns:
            Tuple of (scalar_reward, info_dict).
        """
        info = self.evaluate_completion(
            completion=completion,
            ground_truth=ground_truth,
            is_unanswerable=is_unanswerable,
        )
        return info["reward"], info

    def compute_batch_rewards(
        self,
        completions: List[List[str]],
        ground_truths: List[str],
        unanswerable_flags: Optional[List[bool]] = None,
    ) -> Tuple[torch.Tensor, List[List[Dict[str, Any]]]]:
        """Compute rewards across a batch of rollout groups.

        Args:
            completions: Nested list of completions [B, G].
            ground_truths: List of ground truth answers [B].
            unanswerable_flags: Optional list of booleans [B].

        Returns:
            rewards_tensor: Float32 Tensor of shape [B, G].
            info_batch: Nested list of per-item result dictionaries [B, G].
        """
        B = len(completions)
        info_batch: List[List[Dict[str, Any]]] = []
        reward_matrix: List[List[float]] = []

        for b in range(B):
            gold = ground_truths[b]
            is_unans = unanswerable_flags[b] if unanswerable_flags is not None else False
            group_comps = completions[b]
            group_info: List[Dict[str, Any]] = []
            group_rewards: List[float] = []

            for comp in group_comps:
                r, info = self.compute_reward(comp, gold, is_unanswerable=is_unans)
                group_rewards.append(r)
                group_info.append(info)

            reward_matrix.append(group_rewards)
            info_batch.append(group_info)

        rewards_tensor = torch.tensor(reward_matrix, dtype=torch.float32)
        return rewards_tensor, info_batch
