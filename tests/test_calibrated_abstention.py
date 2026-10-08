"""Unit tests for CalibratedAbstentionRewardEngine.

Verifies:
1. Asymmetric reward schedule: +1.0 (correct), 0.0 (abstain), -1.5 (incorrect), -1.5 (hedging).
2. Mathematical calibration: guessing with <60% confidence has negative expected value,
   with the expected value curve crossing zero at exactly p=0.60.
3. Anti-hedging penalty triggering whenever both <abstain> and a candidate answer are present.
4. AST sandboxing and code execution prevention.
5. Sandboxed SymPy symbolic equivalence across diverse mathematical representations.
"""

import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rl.calibrated_abstention import CalibratedAbstentionRewardEngine


@pytest.fixture
def engine():
    return CalibratedAbstentionRewardEngine(
        r_correct=1.0,
        r_abstain=0.0,
        r_incorrect=-1.5,
        r_hedge=-1.5,
        timeout_seconds=2.0,
    )


def test_asymmetric_reward_schedule(engine):
    """Verify core asymmetric reward values: +1.0, 0.0, -1.5, -1.5 (hedging)."""
    # 1. Correct symbolic answer -> +1.0
    r_corr, info_corr = engine.compute_reward(r"Therefore, the value is \boxed{42}.", "42")
    assert r_corr == 1.0, f"Expected +1.0 for correct answer, got {r_corr}"
    assert info_corr["status"] == "correct"

    # 2. Explicit honest abstention -> 0.0
    r_abs, info_abs = engine.compute_reward("This problem exceeds verification bounds. <abstain>", "42")
    assert r_abs == 0.0, f"Expected 0.0 for abstention, got {r_abs}"
    assert info_abs["status"] == "calibrated_abstention"

    # 3. Incorrect answer -> -1.5
    r_inc, info_inc = engine.compute_reward(r"Answer is \boxed{41}.", "42")
    assert r_inc == -1.5, f"Expected -1.5 for incorrect answer, got {r_inc}"
    assert info_inc["status"] == "incorrect"

    # 4. Hedging (both abstain and boxed guess) -> -1.5
    r_hedge, info_hedge = engine.compute_reward(r"I am not confident <abstain>, but maybe \boxed{42}.", "42")
    assert r_hedge == -1.5, f"Expected -1.5 for hedging, got {r_hedge}"
    assert info_hedge["status"] == "hedging_penalty"


def test_mathematical_calibration_curve(engine):
    """Verify that guessing with <60% confidence yields EV < 0 and crosses zero at p=0.60."""
    cal_res = engine.verify_calibration()
    assert cal_res["is_calibrated"] is True
    assert abs(cal_res["breakeven_confidence"] - 0.60) < 1e-9

    # Verify expected value formula E[R] = 2.5 * p - 1.5 across probability grid
    for pct in range(0, 101, 5):
        p = pct / 100.0
        ev = engine.expected_value_guessing(p)
        expected = 2.5 * p - 1.5
        assert abs(ev - expected) < 1e-9

        if p < 0.60:
            assert ev < 0.0, f"At p={p:.2f} (<0.60), EV must be negative, got {ev}"
            assert ev < engine.r_abstain, f"At p={p:.2f}, abstaining (0.0) must beat guessing ({ev})"
        elif p > 0.60:
            assert ev > 0.0, f"At p={p:.2f} (>0.60), EV must be positive, got {ev}"
            assert ev > engine.r_abstain, f"At p={p:.2f}, guessing ({ev}) must beat abstaining (0.0)"
        else:
            # At exactly p = 0.60, EV must equal 0.0
            assert abs(ev - 0.0) < 1e-9, f"At p=0.60, EV must be exactly 0.0, got {ev}"
            assert abs(ev - engine.r_abstain) < 1e-9


def test_anti_hedging_penalty_patterns(engine):
    """Verify anti-hedging penalty triggers for various hedge phrasing patterns."""
    hedge_cases = [
        r"<abstain> The answer is \boxed{42}",
        r"\boxed{42} though I suspect <abstain>",
        r"Step 1. <abstain> Step 2. \boxed{\frac{7}{32}}",
        r"Maybe unknown <abstain>, guess: \boxed{36\pi}",
    ]
    for text in hedge_cases:
        r, info = engine.compute_reward(text, "42")
        assert r == -1.5, f"Hedging text failed to trigger -1.5 penalty: {text}, got {r}"
        assert info["status"] == "hedging_penalty"
        assert info["is_hedging"] is True


def test_valid_abstention_formats(engine):
    """Verify distinct honest abstention formats without candidate boxes receive 0.0."""
    valid_abstain_cases = [
        "<abstain>",
        "We reach an unresolvable contradiction. <abstain>",
        r"<explore>Testing bounds</explore><formal_proof><abstain></formal_proof>",
        r"Final: \boxed{\text{abstain}}",
        r"Result: \boxed{abstain}",
        r"Conclusion: \boxed{i don't know}",
        r"State: \boxed{unknown}",
    ]
    for text in valid_abstain_cases:
        r, info = engine.compute_reward(text, "42")
        assert r == 0.0, f"Valid abstention format did not receive 0.0: {text}, got {r}"
        assert info["status"] == "calibrated_abstention"


def test_unanswerable_problem_traps(engine):
    """Verify impossible premise trap scoring: +1.0 for honest refusal, -1.5 for hallucination."""
    gold_unans = "<abstain>"

    # Honest abstention on impossible premise -> rewarded with +1.0
    r_honest, info_honest = engine.compute_reward(
        "The premise states x > 5 and x < 2 simultaneously, which is impossible. <abstain>",
        ground_truth=gold_unans,
    )
    assert r_honest == 1.0
    assert info_honest["status"] == "honest_abstention"

    # Hallucinating an answer on impossible premise -> penalized with -1.5
    r_halluc, info_halluc = engine.compute_reward(
        r"I calculated the value to be \boxed{3.5}.",
        ground_truth=gold_unans,
    )
    assert r_halluc == -1.5
    assert info_halluc["status"] == "hallucinated_answer"


def test_symbolic_equivalence_sympy(engine):
    """Verify AST-sandboxed SymPy equivalence across diverse algebraic representations."""
    test_cases = [
        (r"\boxed{42}", "42"),
        (r"\boxed{\frac{7}{32}}", "7/32"),
        (r"\boxed{\dfrac{7}{32}}", "7/32"),
        (r"\boxed{36\pi}", "36*pi"),
        (r"\boxed{x^2 - 1}", "(x - 1)*(x + 1)"),
        (r"\boxed{(x - 1)*(x + 1)}", "x^2 - 1"),
        (r"\boxed{2\sqrt{2}}", r"\sqrt{8}"),
        (r"\boxed{\sqrt{8}}", "2*sqrt(2)"),
        (r"\boxed{0.25}", "1/4"),
        (r"\boxed{\frac{1}{2} + \frac{1}{3}}", "5/6"),
        (r"\boxed{-12}", "-12"),
    ]
    for comp, gold in test_cases:
        r, info = engine.compute_reward(comp, gold)
        assert r == 1.0, f"Equivalence check failed for {comp} vs {gold}, got {r} (status: {info.get('status')})"
        assert info["status"] == "correct"


def test_incorrect_symbolic_answers(engine):
    """Verify wrong answers receive -1.5 penalty."""
    wrong_cases = [
        (r"\boxed{41}", "42"),
        (r"\boxed{\frac{7}{31}}", "7/32"),
        (r"\boxed{-42}", "42"),
        (r"\boxed{x^2 + 1}", "x^2 - 1"),
        (r"\boxed{2\sqrt{3}}", r"\sqrt{8}"),
    ]
    for comp, gold in wrong_cases:
        r, info = engine.compute_reward(comp, gold)
        assert r == -1.5, f"Expected -1.5 for incorrect answer {comp} vs {gold}, got {r}"
        assert info["status"] == "incorrect"


def test_ast_sandbox_blocks_code_injection(engine):
    """Verify that dangerous Python execution attempts are blocked by AST sandboxing."""
    malicious_inputs = [
        r"\boxed{__import__('os').system('ls')}",
        r"\boxed{eval('2 + 2')}",
        r"\boxed{open('/etc/passwd').read()}",
        r"\boxed{globals()}",
        r"\boxed{locals()}",
        r"\boxed{exec('import sys')}",
        r"\boxed{getattr(int, '__doc__')}",
    ]
    for mal in malicious_inputs:
        r, info = engine.compute_reward(mal, "42")
        assert r == -1.5, f"Security injection was not penalized: {mal}, got {r}"
        assert info["status"] in {"missing_or_malformed_answer", "incorrect"}


def test_empty_and_malformed_boxes(engine):
    """Verify empty boxes and unclosed delimiters are safely assigned -1.5."""
    malformed_cases = [
        r"\boxed{}",
        r"\boxed{   }",
        r"\boxed{\frac{7}{32", # Truncated unclosed brace
        "The answer is 42.",    # Missing \boxed{}
    ]
    for text in malformed_cases:
        r, info = engine.compute_reward(text, "42")
        assert r == -1.5, f"Malformed answer should receive -1.5: {text}, got {r}"
        assert info["status"] == "missing_or_malformed_answer"


def test_batch_reward_computation(engine):
    """Verify compute_batch_rewards computes reward tensor for batch of rollout groups."""
    completions = [
        [
            r"\boxed{42}",                                     # Correct (+1.0)
            "<abstain>",                                       # Abstain (0.0)
            r"\boxed{99}",                                     # Incorrect (-1.5)
            r"I think <abstain> but answer is \boxed{42}",     # Hedge (-1.5)
        ],
        [
            r"\boxed{7/32}",                                   # Correct (+1.0)
            r"\boxed{7/31}",                                   # Incorrect (-1.5)
            r"\boxed{\text{abstain}}",                         # Abstain (0.0)
            r"\boxed{\frac{7}{32}}",                           # Correct (+1.0)
        ],
    ]
    ground_truths = ["42", "7/32"]

    rewards, info_batch = engine.compute_batch_rewards(completions, ground_truths)

    assert isinstance(rewards, torch.Tensor)
    assert rewards.shape == (2, 4)
    assert rewards.dtype == torch.float32

    # Verify group 1: [1.0, 0.0, -1.5, -1.5]
    expected_g1 = torch.tensor([1.0, 0.0, -1.5, -1.5])
    assert torch.allclose(rewards[0], expected_g1)

    # Verify group 2: [1.0, -1.5, 0.0, 1.0]
    expected_g2 = torch.tensor([1.0, -1.5, 0.0, 1.0])
    assert torch.allclose(rewards[1], expected_g2)


if __name__ == "__main__":
    print("Running test_calibrated_abstention.py directly...")
    eng = CalibratedAbstentionRewardEngine()
    for fn in [
        lambda: test_asymmetric_reward_schedule(eng),
        lambda: test_mathematical_calibration_curve(eng),
        lambda: test_anti_hedging_penalty_patterns(eng),
        lambda: test_valid_abstention_formats(eng),
        lambda: test_unanswerable_problem_traps(eng),
        lambda: test_symbolic_equivalence_sympy(eng),
        lambda: test_incorrect_symbolic_answers(eng),
        lambda: test_ast_sandbox_blocks_code_injection(eng),
        lambda: test_empty_and_malformed_boxes(eng),
        lambda: test_batch_reward_computation(eng),
    ]:
        try:
            fn()
            print("PASS")
        except Exception as e:
            print(f"FAIL: {e}")
            sys.exit(1)
    print("All calibrated abstention tests PASSED!")
