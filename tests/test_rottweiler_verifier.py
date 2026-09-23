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
    LemmaConsistencyValidator,
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


def test_scaffold_corrupted_tags_and_format_penalty():
    """Verify stray punctuation and corrupted tags are rejected with -1.5 format penalty."""
    # Prefix stray closing tag: *</explore>
    trace_prefix_close = (
        "<explore>Analyzing problem.</explore>\n"
        "<conjecture>Answer is 42.</conjecture>\n"
        "<test_edge_cases>Checking boundary.</test_edge_cases>\n"
        "*</explore>\n"
        "<lemma_isolate>Lemma statement.</lemma_isolate>\n"
        "<formal_proof>Proof concludes \\boxed{42}.</formal_proof>"
    )
    rep1 = XMLScaffoldValidator.validate(trace_prefix_close)
    assert rep1["adherent"] is False
    assert rep1["format_penalty"] == -1.5
    assert "Corrupted closing tag detected" in rep1["error_message"] or "mismatch" in rep1["error_message"]

    # Colon prefix closing tag: :</explore>
    trace_colon = (
        "<explore>Analyzing problem.</explore>\n"
        "<conjecture>Answer is 42.</conjecture>\n"
        "<test_edge_cases>Checking boundary.</test_edge_cases>\n"
        " :</explore>\n"
        "<lemma_isolate>Lemma statement.</lemma_isolate>\n"
        "<formal_proof>Proof concludes \\boxed{42}.</formal_proof>"
    )
    rep2 = XMLScaffoldValidator.validate(trace_colon)
    assert rep2["adherent"] is False
    assert rep2["format_penalty"] == -1.5

    # Suffix stray closing tag: </explore>*
    trace_suffix_close = (
        "<explore>Analyzing problem.</explore>*\n"
        "<conjecture>Answer is 42.</conjecture>\n"
        "<test_edge_cases>Checking boundary.</test_edge_cases>\n"
        "<lemma_isolate>Lemma statement.</lemma_isolate>\n"
        "<formal_proof>Proof concludes \\boxed{42}.</formal_proof>"
    )
    rep3 = XMLScaffoldValidator.validate(trace_suffix_close)
    assert rep3["adherent"] is False
    assert rep3["format_penalty"] == -1.5

    # Malformed tag syntax: < /explore>
    trace_malformed = (
        "<explore>Analyzing problem.< /explore>\n"
        "<conjecture>Answer is 42.</conjecture>\n"
        "<test_edge_cases>Checking boundary.</test_edge_cases>\n"
        "<lemma_isolate>Lemma statement.</lemma_isolate>\n"
        "<formal_proof>Proof concludes \\boxed{42}.</formal_proof>"
    )
    rep4 = XMLScaffoldValidator.validate(trace_malformed)
    assert rep4["adherent"] is False
    assert rep4["format_penalty"] == -1.5
    assert "Malformed tag syntax detected" in rep4["error_message"]

    # Embedded malformed tag inside content with valid outer tags
    trace_embedded_malformed = (
        "<explore>Analyzing problem. < /conjecture> notes.</explore>\n"
        "<conjecture>Answer is 42.</conjecture>\n"
        "<test_edge_cases>Checking boundary.</test_edge_cases>\n"
        "<lemma_isolate>Lemma statement.</lemma_isolate>\n"
        "<formal_proof>Proof concludes \\boxed{42}.</formal_proof>"
    )
    rep5 = XMLScaffoldValidator.validate(trace_embedded_malformed)
    assert rep5["adherent"] is False
    assert rep5["format_penalty"] == -1.5
    assert "Malformed tag syntax detected" in rep5["error_message"]


def test_lemma_premise_sanity_checks():
    """Verify Wilson's Theorem and Fermat's Little Theorem are rejected on composite moduli."""
    validator = LemmaConsistencyValidator()

    # Fermat's Little Theorem on composite modulus 99
    trace_flt_composite = (
        "<explore>Examining 10^k mod 99.</explore>\n"
        "<conjecture>Remainder is 72.</conjecture>\n"
        "<test_edge_cases>10^2 = 100 = 1 mod 99.</test_edge_cases>\n"
        "<lemma_isolate>Lemma (Fermat's Little Theorem): If p is prime and gcd(a, p) = 1, then a^{p-1} = 1 mod p.</lemma_isolate>\n"
        "<formal_proof>Applying Fermat mod 99, remainder is \\boxed{72}.</formal_proof>"
    )
    res_flt = validator.validate(trace_flt_composite)
    assert res_flt["consistent"] is False
    assert res_flt["penalty"] == -1.5
    assert "composite" in res_flt["error_message"].lower()

    # Wilson's Theorem on composite modulus 4
    trace_wilson_composite = (
        "<explore>Factorial residues modulo 4.</explore>\n"
        "<conjecture>Residue is 3.</conjecture>\n"
        "<test_edge_cases>Checking 3! mod 4.</test_edge_cases>\n"
        "<lemma_isolate>Lemma (Wilson's Theorem): (p-1)! = -1 mod p.</lemma_isolate>\n"
        "<formal_proof>Evaluating mod 4 yields \\boxed{3}.</formal_proof>"
    )
    res_wilson = validator.validate(trace_wilson_composite)
    assert res_wilson["consistent"] is False
    assert res_wilson["penalty"] == -1.5
    assert "composite" in res_wilson["error_message"].lower()


def test_lemma_wilson_prime_with_secondary_composite_mod():
    """Verify Wilson applied to prime p=11 is valid even if a secondary operation uses composite mod 4."""
    validator = LemmaConsistencyValidator()
    trace = (
        "<explore>Compute 10! mod 11, then remainder mod 4.</explore>\n"
        "<conjecture>The result is 2.</conjecture>\n"
        "<test_edge_cases>Boundary checks on small factorials.</test_edge_cases>\n"
        "<lemma_isolate>Lemma (Wilson's Theorem): If p is prime, (p-1)! = -1 mod p.</lemma_isolate>\n"
        "<formal_proof>Since 11 is prime, by Wilson's theorem, 10! = -1 = 10 mod 11. Now 10 mod 4 = 2. Thus the answer is \\boxed{2}.</formal_proof>"
    )
    res = validator.validate(trace)
    assert res["consistent"] is True
    assert res["penalty"] == 0.0


def test_lemma_euler_totient_generalizing_flt():
    """Verify Euler's Totient Theorem on composite modulus is valid when mentioning FLT generalization."""
    validator = LemmaConsistencyValidator()
    trace = (
        "<explore>Testing 7^100 mod 100.</explore>\n"
        "<conjecture>Remainder is 1.</conjecture>\n"
        "<test_edge_cases>Check phi(100)=40.</test_edge_cases>\n"
        "<lemma_isolate>Euler's Totient Theorem (generalization of Fermat's Little Theorem to composite moduli): if gcd(a, m) = 1, a^{\\phi(m)} = 1 mod m.</lemma_isolate>\n"
        "<formal_proof>By Euler's totient theorem, phi(100) = 40. 7^40 = 1 mod 100. Thus 7^100 = 7^20 = \\boxed{1} mod 100.</formal_proof>"
    )
    res = validator.validate(trace)
    assert res["consistent"] is True
    assert res["penalty"] == 0.0


def test_lemma_crt_stating_flt_does_not_apply():
    """Verify CRT on composite modulus is valid when explaining FLT does not apply."""
    validator = LemmaConsistencyValidator()
    trace = (
        "<explore>Testing mod 99.</explore>\n"
        "<conjecture>Remainder is 1.</conjecture>\n"
        "<test_edge_cases>Checking coprime factors 9 and 11.</test_edge_cases>\n"
        "<lemma_isolate>Since 99 is composite, Fermat's Little Theorem does not apply directly; we use the Chinese Remainder Theorem to decompose mod 99 into mod 9 and mod 11.</lemma_isolate>\n"
        "<formal_proof>Using CRT with coprime moduli 9 and 11, we solve the system to find remainder \\boxed{1} mod 99.</formal_proof>"
    )
    res = validator.validate(trace)
    assert res["consistent"] is True
    assert res["penalty"] == 0.0


def test_lemma_mod_1_and_carmichael_rejection():
    """Verify modulus 1 and composite Carmichael numbers fail prime precondition."""
    validator = LemmaConsistencyValidator()

    # Modulus 1 is not prime
    trace_mod1 = (
        "<explore>Evaluating 0! mod 1.</explore>\n"
        "<conjecture>0</conjecture>\n"
        "<test_edge_cases>Small case.</test_edge_cases>\n"
        "<lemma_isolate>Lemma (Wilson's Theorem): (p-1)! = -1 mod p for prime p.</lemma_isolate>\n"
        "<formal_proof>By Wilson's theorem mod 1, 0! = -1 = 0 mod 1. The result is \\boxed{0}.</formal_proof>"
    )
    res1 = validator.validate(trace_mod1)
    assert res1["consistent"] is False
    assert res1["penalty"] == -1.5

    # Carmichael number 561 (composite: 3 * 11 * 17)
    trace_carmichael = (
        "<explore>Evaluating powers mod 561.</explore>\n"
        "<conjecture>1</conjecture>\n"
        "<test_edge_cases>Small case.</test_edge_cases>\n"
        "<lemma_isolate>Lemma (Fermat's Little Theorem): a^{p-1} = 1 mod p for prime p.</lemma_isolate>\n"
        "<formal_proof>By Fermat's Little Theorem mod 561, a^560 = 1 mod 561. The result is \\boxed{1}.</formal_proof>"
    )
    res_c = validator.validate(trace_carmichael)
    assert res_c["consistent"] is False
    assert res_c["penalty"] == -1.5
    assert "composite" in res_c["error_message"].lower()


def test_scaffold_unclosed_tag_truncation():
    """Verify trace truncated at max_tokens with unclosed tag receives format penalty and truthful lemma telemetry."""
    trace_truncated = (
        "<explore>Analyzing problem.</explore>\n"
        "<conjecture>Answer is 42.</conjecture>\n"
        "<test_edge_cases>Checking boundary.</test_edge_cases>\n"
        "<lemma_isolate>Lemma statement.</lemma_isolate>\n"
        "<formal_proof>Proof starts here and gets truncated without boxed answer"
    )
    rep = XMLScaffoldValidator.validate(trace_truncated)
    assert rep["adherent"] is False
    assert rep["format_penalty"] == -1.5

    verifier = RottweilerVerifier()
    v_rep = verifier.verify(trace_truncated, ground_truth="42")
    assert v_rep["valid"] is False
    assert v_rep["scaffold_adherent"] is False
    assert v_rep["format_penalty"] == -1.5
    # Truthful telemetry: lemma was not the cause of failure
    assert v_rep["lemma_penalty"] == 0.0


def test_lemma_semantic_divergence():
    """Verify semantic divergence between lemma and proof is penalized -1.5."""
    validator = LemmaConsistencyValidator()

    # Declares Chinese Remainder Theorem but proof uses only basic integer arithmetic without modular concepts
    trace_divergent = (
        "<explore>Factoring an integer.</explore>\n"
        "<conjecture>Number of divisors is 8.</conjecture>\n"
        "<test_edge_cases>Testing small values.</test_edge_cases>\n"
        "<lemma_isolate>Lemma (Chinese Remainder Theorem): A system of simultaneous congruences modulo coprime integers has unique solution.</lemma_isolate>\n"
        "<formal_proof>We multiply 2 * 3 = 6. Divisors of 6 are 1, 2, 3, 6, total \\boxed{4}.</formal_proof>"
    )
    res_div = validator.validate(trace_divergent)
    assert res_div["consistent"] is False
    assert res_div["penalty"] == -1.5
    assert "Semantic divergence" in res_div["error_message"]

    # Concordant: declares Pythagorean Theorem and proof uses hypotenuse / right triangle
    trace_concordant = (
        "<explore>Analyzing right triangle sides.</explore>\n"
        "<conjecture>Hypotenuse is 5.</conjecture>\n"
        "<test_edge_cases>Legs are 3 and 4.</test_edge_cases>\n"
        "<lemma_isolate>Lemma (Pythagorean Theorem): In a right triangle, a^2 + b^2 = c^2.</lemma_isolate>\n"
        "<formal_proof>By the Pythagorean theorem on right triangle ABC with legs 3 and 4, hypotenuse c = \\sqrt{9+16} = \\boxed{5}.</formal_proof>"
    )
    res_con = validator.validate(trace_concordant)
    assert res_con["consistent"] is True
    assert res_con["penalty"] == 0.0


def test_scaffold_inner_corrupted_tag_variations():
    """Verify inner punctuations inside tag brackets and casing variations are strictly rejected."""
    corrupted_cases = [
        "<explore>Analyzing problem. </*explore> notes.</explore><conjecture>Conjecture</conjecture><test_edge_cases>Edges</test_edge_cases><lemma_isolate>Lemma</lemma_isolate><formal_proof>Proof \\boxed{1}</formal_proof>",
        "<explore>Analyzing problem. <*explore> notes.</explore><conjecture>Conjecture</conjecture><test_edge_cases>Edges</test_edge_cases><lemma_isolate>Lemma</lemma_isolate><formal_proof>Proof \\boxed{1}</formal_proof>",
        "<explore>Analyzing problem. <:explore> notes.</explore><conjecture>Conjecture</conjecture><test_edge_cases>Edges</test_edge_cases><lemma_isolate>Lemma</lemma_isolate><formal_proof>Proof \\boxed{1}</formal_proof>",
        "<explore>Analyzing problem. <//explore> notes.</explore><conjecture>Conjecture</conjecture><test_edge_cases>Edges</test_edge_cases><lemma_isolate>Lemma</lemma_isolate><formal_proof>Proof \\boxed{1}</formal_proof>",
        "<explore>Analyzing problem. < / * explore > notes.</explore><conjecture>Conjecture</conjecture><test_edge_cases>Edges</test_edge_cases><lemma_isolate>Lemma</lemma_isolate><formal_proof>Proof \\boxed{1}</formal_proof>",
        "<explore>Analyzing problem. <Explore> notes.</explore><conjecture>Conjecture</conjecture><test_edge_cases>Edges</test_edge_cases><lemma_isolate>Lemma</lemma_isolate><formal_proof>Proof \\boxed{1}</formal_proof>",
    ]
    for trace in corrupted_cases:
        rep = XMLScaffoldValidator.validate(trace)
        assert rep["adherent"] is False
        assert rep["format_penalty"] == -1.5
        assert "Malformed tag syntax detected" in rep["error_message"]


def test_scaffold_dash_and_semicolon_stray_punctuation():
    """Verify dashes and semicolons attached to tag boundaries are rejected with format penalty."""
    trace_dash = (
        "<explore>Analyzing problem.</explore>-\n"
        "<conjecture>Answer is 42.</conjecture>\n"
        "<test_edge_cases>Checking boundary.</test_edge_cases>\n"
        "<lemma_isolate>Lemma statement.</lemma_isolate>\n"
        "<formal_proof>Proof concludes \\boxed{42}.</formal_proof>"
    )
    rep_dash = XMLScaffoldValidator.validate(trace_dash)
    assert rep_dash["adherent"] is False
    assert rep_dash["format_penalty"] == -1.5

    trace_semi = (
        "<explore>Analyzing problem.</explore> ;\n"
        "<conjecture>Answer is 42.</conjecture>\n"
        "<test_edge_cases>Checking boundary.</test_edge_cases>\n"
        "<lemma_isolate>Lemma statement.</lemma_isolate>\n"
        "<formal_proof>Proof concludes \\boxed{42}.</formal_proof>"
    )
    rep_semi = XMLScaffoldValidator.validate(trace_semi)
    assert rep_semi["adherent"] is False
    assert rep_semi["format_penalty"] == -1.5


def test_lemma_wilson_in_formal_proof_composite_rejected():
    """Verify Wilson's theorem applied to composite modulus in formal_proof is penalized even if undeclared in lemma_isolate."""
    validator = LemmaConsistencyValidator()
    trace = (
        "<explore>Exploring factorial residues modulo 12.</explore>\n"
        "<conjecture>The residue is 11.</conjecture>\n"
        "<test_edge_cases>Boundary evaluation.</test_edge_cases>\n"
        "<lemma_isolate>We evaluate residues of factorials under modular arithmetic.</lemma_isolate>\n"
        "<formal_proof>By Wilson's theorem modulo 12, 11! = -1 mod 12. Thus \\boxed{11}.</formal_proof>"
    )
    res = validator.validate(trace)
    assert res["consistent"] is False
    assert res["penalty"] == -1.5
    assert "composite" in res["error_message"].lower()


def test_lemma_various_moduli_syntaxes():
    """Verify various LaTeX and textual moduli syntaxes are recognized during precondition checks."""
    validator = LemmaConsistencyValidator()
    syntaxes = [
        r"By Wilson's theorem, we evaluate \pmod(15) and find residue \boxed{1}.",
        r"Applying Fermat's little theorem with \pmod{p=15}, we get \boxed{1}.",
        r"By Wilson's theorem with mod p = 15, we obtain \boxed{1}.",
        r"Using Fermat's little theorem with modulus: 15, we get \boxed{1}.",
        r"By Wilson's theorem with \mod 15, we find \boxed{1}."
    ]
    for proof_body in syntaxes:
        trace = (
            "<explore>Exploring constraints.</explore>\n"
            "<conjecture>Remainder is 1.</conjecture>\n"
            "<test_edge_cases>Boundary evaluation.</test_edge_cases>\n"
            "<lemma_isolate>We analyze modular residues.</lemma_isolate>\n"
            f"<formal_proof>{proof_body}</formal_proof>"
        )
        res = validator.validate(trace)
        assert res["consistent"] is False
        assert res["penalty"] == -1.5
        assert "composite" in res["error_message"].lower()


def test_scaffold_no_false_positive_on_valid_punctuation_and_inequalities():
    """Verify legitimate text ending in %, quotes, commas, inequalities, or bars is not falsely rejected."""
    valid_snippets = [
        "The probability is 50% ",
        "We call it \"symmetric\" ",
        "We compute x, ",
        "where x > 0 ",
        "We have |x| ",
        "We assume 0 < explore(y) for all y > 0.",
        "0 < 1, so explore more.",
    ]
    for snippet in valid_snippets:
        trace = (
            f"<explore>{snippet}</explore>\n"
            "<conjecture>Answer is 42.</conjecture>\n"
            "<test_edge_cases>Checking boundary.</test_edge_cases>\n"
            "<lemma_isolate>Lemma statement.</lemma_isolate>\n"
            "<formal_proof>Proof concludes \\boxed{42}.</formal_proof>"
        )
        rep = XMLScaffoldValidator.validate(trace)
        assert rep["adherent"] is True, f"Falsely rejected: {snippet} -> {rep['error_message']}"
        assert rep["format_penalty"] == 0.0


def test_lemma_wilson_and_flt_variable_parameter_assignments_rejected():
    """Verify Wilson and Fermat precondition checks detect variable assignments p=15, n=15, prime=15, modulus=15."""
    validator = LemmaConsistencyValidator()
    failing_proofs = [
        "Applying Fermat's Little Theorem with p = 15, we compute 2^14 = \\boxed{1}.",
        "Applying Wilson's Theorem for n = 15, we compute 14! = -1 = \\boxed{14}.",
        "We use Fermat's Little Theorem with prime 15, so \\boxed{1}.",
        "By Wilson's theorem with modulus = 15, we get \\boxed{1}.",
        "We apply Fermat's Little Theorem. Here p = 15, so 2^14 = \\boxed{1}."
    ]
    for proof in failing_proofs:
        trace = (
            "<explore>Exploring constraints.</explore>\n"
            "<conjecture>Remainder is 1.</conjecture>\n"
            "<test_edge_cases>Boundary evaluation.</test_edge_cases>\n"
            "<lemma_isolate>We apply modular theorems.</lemma_isolate>\n"
            f"<formal_proof>{proof}</formal_proof>"
        )
        res = validator.validate(trace)
        assert res["consistent"] is False, f"Failed to reject composite modulus in: {proof}"
        assert res["penalty"] == -1.5
        assert "composite" in res["error_message"].lower()


def test_lemma_combinatorial_factorials_not_flagged_as_wilson():
    """Verify purely combinatorial factorials like (n-1)! / 2 with composite n are not misclassified as Wilson's theorem."""
    validator = LemmaConsistencyValidator()
    # Hamiltonian cycles in K_6
    trace_hamiltonian = (
        "<explore>Analyzing Hamiltonian cycles in K_6.</explore>\n"
        "<conjecture>The count is 60.</conjecture>\n"
        "<test_edge_cases>Base case n=3 gives 1 triangle.</test_edge_cases>\n"
        "<lemma_isolate>The number of undirected Hamiltonian cycles in K_n is given by (n-1)! / 2.</lemma_isolate>\n"
        "<formal_proof>For n = 6, we have (6-1)! / 2 = 5! / 2 = \\boxed{60}.</formal_proof>"
    )
    res_h = validator.validate(trace_hamiltonian)
    assert res_h["consistent"] is True
    assert res_h["penalty"] == 0.0

    # Circular permutations of 4 items
    trace_circular = (
        "<explore>Arranging 4 items in a circle.</explore>\n"
        "<conjecture>Count is 6.</conjecture>\n"
        "<test_edge_cases>Small check.</test_edge_cases>\n"
        "<lemma_isolate>The number of circular permutations of n items is (n-1)!.</lemma_isolate>\n"
        "<formal_proof>For n = 4, (4-1)! = 3! = \\boxed{6}.</formal_proof>"
    )
    res_c = validator.validate(trace_circular)
    assert res_c["consistent"] is True
    assert res_c["penalty"] == 0.0


def test_lemma_semantic_divergence_nomenclature_variants():
    """Verify hyphen spacing and terminology variants trigger divergence checks when unutilized in proof."""
    validator = LemmaConsistencyValidator()
    divergent_pairs = [
        ("Lemma: By Euclid's algorithm, we compute gcd.", "We multiply 2 * 3 = 6 to get \\boxed{6}."),
        ("Lemma: We apply AM - GM inequality.", "We multiply 2 * 3 = 6 to get \\boxed{6}."),
        ("Lemma: We apply Cauchy - Schwarz.", "We multiply 2 * 3 = 6 to get \\boxed{6}."),
        ("Lemma: We use Principle of Inclusion - Exclusion.", "We multiply 2 * 3 = 6 to get \\boxed{6}."),
    ]
    for lemma_body, proof_body in divergent_pairs:
        trace = (
            "<explore>Exploring constraints.</explore>\n"
            "<conjecture>Result is 6.</conjecture>\n"
            "<test_edge_cases>Boundary evaluation.</test_edge_cases>\n"
            f"<lemma_isolate>{lemma_body}</lemma_isolate>\n"
            f"<formal_proof>{proof_body}</formal_proof>"
        )
        res = validator.validate(trace)
        assert res["consistent"] is False, f"Expected divergence for: {lemma_body}"
        assert res["penalty"] == -1.5
        assert "Semantic divergence" in res["error_message"]


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__]))
