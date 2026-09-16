#!/usr/bin/env python3
"""Unit tests for Requirement R5: Rottweiler XML Scaffold and SymPy Verifier.

Tests:
1. XMLScaffoldValidator: positive full 5-stage trace adherence.
2. XMLScaffoldValidator: rejection of nested tags.
3. XMLScaffoldValidator: rejection of \\boxed{...} outside <formal_proof>.
4. XMLScaffoldValidator: rejection of non-whitespace text after </formal_proof>.
5. XMLScaffoldValidator: rejection of empty or non-substantive tag content.
6. RottweilerSymPyVerifier: exact symbolic equivalence across LaTeX formats.
7. RottweilerSymPyVerifier: AST sandboxing rejecting malicious injections.
8. RottweilerSymPyVerifier: reverse substitution on polynomial equations.
9. RottweilerSymPyVerifier: extraneous root detection (division by zero in original equations).
10. RottweilerSymPyVerifier: domain constraints enforcement (integer, positive).
11. RottweilerSymPyVerifier: execution timeout handling without process crash.
12. RottweilerVerifier: unified end-to-end verification report.
"""

import os
import sys
import time
import pytest

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from src.rl.rottweiler_verifier import (
    XMLScaffoldValidator,
    RottweilerSymPyVerifier,
    RottweilerVerifier,
    is_safe_math_expression,
    extract_boxed_answer,
    clean_latex_math,
)


def test_scaffold_positive():
    """Verify compliant 5-stage XML trace satisfies all structural invariants."""
    valid_trace = (
        "<explore>\n"
        "Let the given equation be P(x) = x^2 - 5x + 6 = 0. We analyze its coefficients.\n"
        "</explore>\n"
        "<conjecture>\n"
        "The roots of P(x) are x = 2 and x = 3.\n"
        "</conjecture>\n"
        "<test_edge_cases>\n"
        "Check discriminant Delta = (-5)^2 - 4*1*6 = 25 - 24 = 1 > 0. Two distinct roots.\n"
        "</test_edge_cases>\n"
        "<lemma_isolate>\n"
        "Lemma: (x - r1)(x - r2) = x^2 - (r1+r2)x + r1*r2. Here r1+r2 = 5, r1*r2 = 6.\n"
        "</lemma_isolate>\n"
        "<formal_proof>\n"
        "Factoring gives (x - 2)(x - 3) = 0. The smallest positive root is \\boxed{2}.\n"
        "</formal_proof>"
    )

    report = XMLScaffoldValidator.validate(valid_trace)
    assert report["adherent"] is True
    assert report["all_tags_present"] is True
    assert report["strictly_ordered"] is True
    assert report["zero_nesting"] is True
    assert report["boxed_in_formal_proof"] is True
    assert report["no_trailing_text"] is True
    assert report["boxed_answer"] == "2"
    assert report["error_message"] is None


def test_scaffold_nested_tags():
    """Verify nesting an XML tag inside another block fails zero_nesting invariant."""
    nested_trace = (
        "<explore>\n"
        "Exploring invariants. <conjecture>Premature conjecture</conjecture>\n"
        "</explore>\n"
        "<conjecture>Valid conjecture.</conjecture>\n"
        "<test_edge_cases>Checking cases.</test_edge_cases>\n"
        "<lemma_isolate>Isolating lemma.</lemma_isolate>\n"
        "<formal_proof>Proof gives \\boxed{42}.</formal_proof>"
    )

    report = XMLScaffoldValidator.validate(nested_trace)
    assert report["adherent"] is False
    assert report["zero_nesting"] is False


def test_scaffold_boxed_outside_formal_proof():
    """Verify placement of \\boxed{...} outside <formal_proof> fails validation."""
    trace_boxed_in_conjecture = (
        "<explore>Exploring constraints.</explore>\n"
        "<conjecture>We believe \\boxed{42} is the answer.</conjecture>\n"
        "<test_edge_cases>Testing small parameters.</test_edge_cases>\n"
        "<lemma_isolate>Lemma statement.</lemma_isolate>\n"
        "<formal_proof>Proof establishes validity without repetition.</formal_proof>"
    )

    report = XMLScaffoldValidator.validate(trace_boxed_in_conjecture)
    assert report["adherent"] is False
    assert report["boxed_in_formal_proof"] is False


def test_scaffold_trailing_text_after_formal_proof():
    """Verify non-whitespace reasoning text following </formal_proof> is rejected."""
    trace_with_trailing = (
        "<explore>Exploring constraints.</explore>\n"
        "<conjecture>Candidate answer is 10.</conjecture>\n"
        "<test_edge_cases>Boundary evaluation.</test_edge_cases>\n"
        "<lemma_isolate>Helper lemma.</lemma_isolate>\n"
        "<formal_proof>Proof yields \\boxed{10}.</formal_proof>\n"
        "Post-proof commentary explaining why this answer makes intuitive sense."
    )

    report = XMLScaffoldValidator.validate(trace_with_trailing)
    assert report["adherent"] is False
    assert report["no_trailing_text"] is False


def test_scaffold_empty_tag():
    """Verify empty or non-substantive tag block fails validation."""
    trace_empty_explore = (
        "<explore>   </explore>\n"
        "<conjecture>Candidate answer is 10.</conjecture>\n"
        "<test_edge_cases>Boundary evaluation.</test_edge_cases>\n"
        "<lemma_isolate>Helper lemma.</lemma_isolate>\n"
        "<formal_proof>Proof yields \\boxed{10}.</formal_proof>"
    )

    report = XMLScaffoldValidator.validate(trace_empty_explore)
    assert report["adherent"] is False


def test_sympy_exact_equivalence():
    """Verify SymPy correctly equates equivalent mathematical representations."""
    verifier = RottweilerSymPyVerifier(timeout_seconds=5.0)

    # Fractions
    assert verifier.verify_symbolic_equivalence(r"\frac{7}{32}", "7/32") is True
    assert verifier.verify_symbolic_equivalence(r"\dfrac{1}{2}", "0.5") is True

    # Multiples of pi and transcendental expressions
    assert verifier.verify_symbolic_equivalence(r"36\pi", "36*pi") is True
    assert verifier.verify_symbolic_equivalence(r"\sin(\pi/2)", "1") is True

    # Polynomial expansion
    assert verifier.verify_symbolic_equivalence(r"(x + 1)^2", "x^2 + 2*x + 1") is True

    # Square roots
    assert verifier.verify_symbolic_equivalence(r"\sqrt{12}", "2*sqrt(3)") is True

    # Distinct values must return False
    assert verifier.verify_symbolic_equivalence("41", "42") is False
    assert verifier.verify_symbolic_equivalence("x + 1", "x - 1") is False


def test_sympy_ast_sandbox_security():
    """Verify AST sandboxing rejects RCE attempts and malicious Python calls."""
    verifier = RottweilerSymPyVerifier()

    malicious_inputs = [
        "__import__('os').system('ls')",
        "eval('1 + 1')",
        "open('/etc/passwd').read()",
        "getattr(__import__('os'), 'system')",
        "exec('a = 5')",
        "__class__.__bases__",
        "breakpoint()",
        "exit(0)",
    ]

    for evil_str in malicious_inputs:
        safe = verifier.is_safe_expression(evil_str)
        assert safe is False, f"Malicious expression {evil_str} bypassed AST safety filter"
        # Must also be rejected by verify_symbolic_equivalence
        assert verifier.verify_symbolic_equivalence(evil_str, "42") is False

    # Legitimate expressions must pass safety filter
    valid_math = [
        "sqrt(12)",
        "36*pi",
        "(x + 1)**2",
        "Rational(7, 32)",
        "x**2 - 5*x + 6",
    ]
    for expr in valid_math:
        assert verifier.is_safe_expression(expr) is True, f"Valid expr {expr} rejected by safety filter"


def test_reverse_substitution_polynomial():
    """Verify candidate answers substituted into polynomial equations."""
    verifier = RottweilerSymPyVerifier()
    equation = "x^2 - 5*x + 6 = 0"

    # Root x = 2 satisfies equation
    res_2 = verifier.verify_reverse_substitution("2", equation, variable_name="x")
    assert res_2["valid"] is True
    assert res_2["domain_valid"] is True
    assert res_2["error_message"] is None

    # Root x = 3 satisfies equation
    res_3 = verifier.verify_reverse_substitution("3", equation, variable_name="x")
    assert res_3["valid"] is True

    # Non-root x = 4 fails equation
    res_4 = verifier.verify_reverse_substitution("4", equation, variable_name="x")
    assert res_4["valid"] is False
    assert res_4["residual"] is not None


def test_reverse_substitution_extraneous_roots():
    """Verify extraneous root causing division by zero in equation is detected and rejected."""
    verifier = RottweilerSymPyVerifier()
    # In x / (x - 2) = 2 / (x - 2), multiplying both sides by (x - 2) yields x = 2,
    # but x = 2 makes the denominator 0, so it is an extraneous root.
    equation = r"\frac{x}{x - 2} = \frac{2}{x - 2}"

    res = verifier.verify_reverse_substitution("2", equation, variable_name="x")
    assert res["valid"] is False
    assert res["domain_valid"] is False
    assert "Extraneous root" in (res.get("error_message") or "")


def test_reverse_substitution_domain_constraints():
    """Verify domain constraints (integer, positive) are strictly evaluated."""
    verifier = RottweilerSymPyVerifier()
    # 2*x - 5 = 0 has solution x = 5/2 = 2.5
    equation = "2*x - 5 = 0"

    # With integer constraint, 5/2 must fail
    res_int = verifier.verify_reverse_substitution(
        "5/2", equation, variable_name="x", domain_constraints={"integer": True}
    )
    assert res_int["valid"] is False
    assert "integer" in (res_int.get("error_message") or "").lower()

    # x^2 - 9 = 0 has solutions x = 3 and x = -3
    eq_quad = "x^2 - 9 = 0"
    res_neg = verifier.verify_reverse_substitution(
        "-3", eq_quad, variable_name="x", domain_constraints={"positive": True}
    )
    assert res_neg["valid"] is False
    assert "positive" in (res_neg.get("error_message") or "").lower()


def test_sympy_timeout_resilience():
    """Verify verifier handles execution timeouts gracefully returning False."""
    verifier = RottweilerSymPyVerifier(timeout_seconds=0.2)

    # Pathological nested algebraic expression that takes long to simplify
    pathological_expr = "((x + 1)**50 - (x - 1)**50)**2"
    # Even if an evaluation times out, it must return False without crashing
    res = verifier.verify_symbolic_equivalence(pathological_expr, "0")
    assert res is False


def test_reverse_substitution_linear_and_fraction_whitespace():
    """Verify linear equations with x = c and fraction whitespace parse correctly."""
    verifier = RottweilerSymPyVerifier(timeout_seconds=2.0)

    # Linear equation: x = 2
    res_linear = verifier.verify_reverse_substitution("2", "x = 2", variable_name="x")
    assert res_linear["valid"] is True
    assert res_linear["residual"] == "0"

    # Fraction with whitespace: \frac{1} {2}
    res_frac = verifier.verify_symbolic_equivalence(r"\frac{1} {2}", "0.5")
    assert res_frac is True


def test_unified_rottweiler_verifier():
    """Verify unified RottweilerVerifier pipeline combining scaffold and SymPy checks."""
    verifier = RottweilerVerifier(timeout_seconds=5.0)

    good_completion = (
        "<explore>Analyzing roots of x^2 - 5x + 6.</explore>\n"
        "<conjecture>Roots are 2 and 3.</conjecture>\n"
        "<test_edge_cases>Checking discriminant: Delta = 1 > 0.</test_edge_cases>\n"
        "<lemma_isolate>Lemma: x^2 - (r1+r2)x + r1*r2.</lemma_isolate>\n"
        "<formal_proof>The roots are x = \\boxed{2} and x = 3.</formal_proof>"
    )

    report = verifier.verify(good_completion, ground_truth="2", equation_str="x^2 - 5*x + 6 = 0")
    assert report["valid"] is True
    assert report["scaffold_adherent"] is True
    assert report["symbolic_correct"] is True
    assert report["boxed_answer"] == "2"

    # Wrong answer
    report_wrong = verifier.verify(good_completion, ground_truth="5", equation_str="x^2 - 5*x + 6 = 0")
    assert report_wrong["valid"] is False
    assert report_wrong["scaffold_adherent"] is True
    assert report_wrong["symbolic_correct"] is False


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__]))
