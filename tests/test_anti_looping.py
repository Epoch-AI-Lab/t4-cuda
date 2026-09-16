#!/usr/bin/env python3
"""Unit tests for Requirement R4: Anti-Looping, XML Progression, and Entropy Floor.

Tests:
1. RollingNGramTracker: clean sequence, 2 repeats allowed, halts on 3rd (>2 occurrences).
2. FourGramRepetitionCriteria: stopping criteria halting and assigning r_rep = -0.5.
3. Prompt masking: prompt tokens excluded from repetition tracking.
4. XMLProgressionTracker: valid trace acceptance, forward monotonicity.
5. XMLProgressionTracker: backtracking detection and rejection.
6. XMLProgressionTracker: stage skipping detection and rejection.
7. XMLProgressionTracker: token budget ceiling enforcement and stall violation.
8. EntropyFloorController: dual ascent beta adjustment (increases when H < floor, decreases when H > floor).
9. EntropyFloorController: compute_entropy correctness on uniform and masked logits.
"""

import math
import os
import sys
import pytest
import torch

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from src.rl.anti_looping import (
    RollingNGramTracker,
    FourGramRepetitionCriteria,
    XMLProgressionTracker,
    EntropyFloorController,
)


def test_ngram_tracker_clean_sequence():
    """Verify distinct tokens never trigger loop detection."""
    tracker = RollingNGramTracker(n=4, max_occurrences=2)
    tokens = list(range(100, 130))

    for tok in tokens:
        triggered = tracker.update(tok)
        assert not triggered, f"Unexpected loop trigger on token {tok}"

    assert tracker.get_max_occurrence() == 1
    assert not tracker.loop_detected
    assert tracker.violating_ngram is None


def test_ngram_tracker_single_repeat():
    """Verify an n-gram appearing exactly 2 times is permitted."""
    tracker = RollingNGramTracker(n=4, max_occurrences=2)
    # 4-gram [1, 2, 3, 4] appears twice
    sequence = [1, 2, 3, 4, 99, 88, 1, 2, 3, 4]

    for tok in sequence:
        triggered = tracker.update(tok)
        assert not triggered, f"Unexpected loop trigger on 2nd occurrence for token {tok}"

    assert tracker.get_max_occurrence() == 2
    assert not tracker.loop_detected


def test_ngram_tracker_exceeds_threshold():
    """Verify an n-gram appearing 3 times halts on the 3rd occurrence (> 2)."""
    tracker = RollingNGramTracker(n=4, max_occurrences=2)
    # [10, 20, 30, 40] appears 3 times
    first_pass = [10, 20, 30, 40]
    intermediate = [999, 888]
    second_pass = [10, 20, 30, 40]
    third_pass = [10, 20, 30, 40]

    for tok in first_pass + intermediate:
        assert not tracker.update(tok)
    assert tracker.get_max_occurrence() == 1

    for tok in second_pass:
        assert not tracker.update(tok)
    assert tracker.get_max_occurrence() == 2

    # Third pass: should trigger on the 4th token of this pass
    triggered_tokens = []
    for tok in third_pass:
        t = tracker.update(tok)
        triggered_tokens.append(t)

    assert triggered_tokens == [False, False, False, True]
    assert tracker.get_max_occurrence() == 3
    assert tracker.loop_detected is True
    assert tracker.violating_ngram == (10, 20, 30, 40)


def test_stopping_criteria_forced_truncation():
    """Verify Hugging Face StoppingCriteria halts generation and sets penalty -0.5."""
    criteria = FourGramRepetitionCriteria(max_occurrences=2, penalty=-0.5, n=4)

    # Batch of 2 sequences:
    # Seq 0: clean sequence
    # Seq 1: repeats [7, 8, 9, 10] three times
    clean_seq = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    looping_seq = [7, 8, 9, 10, 99, 7, 8, 9, 10, 88, 7, 8, 9, 10]

    # Simulate step-by-step autoregressive decoding
    max_len = max(len(clean_seq), len(looping_seq))
    pad_token = 0

    criteria.reset(batch_size=2, prompt_lengths=[0, 0])
    halted_at_step = None

    for step in range(4, max_len + 1):
        s0 = clean_seq[:min(step, len(clean_seq))]
        s1 = looping_seq[:min(step, len(looping_seq))]
        cur_len = max(len(s0), len(s1))

        # Pad to equal length
        s0_padded = s0 + [pad_token] * (cur_len - len(s0))
        s1_padded = s1 + [pad_token] * (cur_len - len(s1))

        input_ids = torch.tensor([s0_padded, s1_padded], dtype=torch.long)
        should_stop = criteria(input_ids)

        if should_stop:
            halted_at_step = step
            break

    assert halted_at_step is not None, "StoppingCriteria failed to halt repeating sequence"
    assert criteria.is_loop(1) is True
    assert criteria.get_penalty(1) == -0.5
    assert criteria.is_loop(0) is False
    assert criteria.get_penalty(0) == 0.0


def test_stopping_criteria_prompt_masking():
    """Verify prompt tokens are ignored and do not count towards repetition threshold."""
    criteria = FourGramRepetitionCriteria(max_occurrences=2, penalty=-0.5, n=4)

    # Prompt contains 2 occurrences of [1, 2, 3, 4], completion contains 1 occurrence
    # With prompt masking, only 1 occurrence is in completion, so it should not halt
    prompt = [1, 2, 3, 4, 55, 1, 2, 3, 4]
    completion = [88, 1, 2, 3, 4]
    full_seq = prompt + completion

    criteria.reset(batch_size=1, prompt_lengths=[len(prompt)])
    input_ids = torch.tensor([full_seq], dtype=torch.long)

    should_stop = criteria(input_ids)
    assert not should_stop
    assert criteria.is_loop(0) is False
    assert criteria.get_penalty(0) == 0.0


def test_tag_progression_fsm_valid():
    """Verify valid 5-stage trace adheres to progression FSM."""
    tracker = XMLProgressionTracker()
    valid_text = (
        "<explore>We inspect the algebraic constraints and symmetries.</explore>\n"
        "<conjecture>We suspect the unique positive integer root is 3.</conjecture>\n"
        "<test_edge_cases>Evaluating n=0, n=1, and boundary values.</test_edge_cases>\n"
        "<lemma_isolate>Lemma 1: f(x) is strictly convex on positive reals.</lemma_isolate>\n"
        "<formal_proof>Deduction establishes that \\boxed{3} holds.</formal_proof>"
    )

    report = tracker.validate_trace(valid_text)
    assert report["valid"] is True, f"Valid trace failed with violations: {report['violations']}"
    assert report["current_state"] == XMLProgressionTracker.STATE_TERMINAL
    assert len(report["violations"]) == 0


def test_tag_progression_fsm_backtracking():
    """Verify reopening an earlier completed tag is detected as backtracking."""
    tracker = XMLProgressionTracker()
    backtrack_text = (
        "<explore>Exploring properties.</explore>\n"
        "<conjecture>Conjecturing value is 5.</conjecture>\n"
        "<explore>Revisiting explore stage.</explore>\n"
        "<test_edge_cases>Testing edge cases.</test_edge_cases>\n"
        "<lemma_isolate>Isolating lemma.</lemma_isolate>\n"
        "<formal_proof>Proof gives \\boxed{5}.</formal_proof>"
    )

    report = tracker.validate_trace(backtrack_text)
    assert report["valid"] is False
    assert any("Backtracking" in v or "Duplicate" in v for v in report["violations"])


def test_tag_progression_fsm_stage_skipped():
    """Verify skipping an intermediate stage is detected and rejected."""
    tracker = XMLProgressionTracker()
    skipped_text = (
        "<explore>Exploring properties.</explore>\n"
        "<test_edge_cases>Jumped straight to test cases without conjecture.</test_edge_cases>\n"
        "<lemma_isolate>Isolating lemma.</lemma_isolate>\n"
        "<formal_proof>Proof gives \\boxed{5}.</formal_proof>"
    )

    report = tracker.validate_trace(skipped_text)
    assert report["valid"] is False
    assert any("conjecture" in v for v in report["violations"])


def test_tag_progression_token_budget_exceeded():
    """Verify exceeding per-stage token limit triggers stall violation."""
    tracker = XMLProgressionTracker(
        per_stage_token_limits={"explore": 10, "conjecture": 50, "test_edge_cases": 50, "lemma_isolate": 50, "formal_proof": 50}
    )

    # 25 tokens inside <explore>, exceeding budget of 10
    bloated_explore = " ".join([f"word{i}" for i in range(25)])
    text = (
        f"<explore>{bloated_explore}</explore>\n"
        "<conjecture>Conjecture content.</conjecture>\n"
        "<test_edge_cases>Edge case content.</test_edge_cases>\n"
        "<lemma_isolate>Lemma content.</lemma_isolate>\n"
        "<formal_proof>Formal proof content \\boxed{42}.</formal_proof>"
    )

    report = tracker.validate_trace(text)
    assert report["valid"] is False
    assert any("Token budget exceeded" in v for v in report["violations"])


def test_entropy_floor_controller_dual_ascent():
    """Verify dual ascent dynamically regulates beta."""
    controller = EntropyFloorController(floor_nats=0.25, lr=0.01, beta_min=0.001, beta_max=0.10, beta_init=0.01)

    initial_beta = controller.beta
    assert initial_beta == 0.01

    # Low entropy: H = 0.10 nats (< 0.25 floor) -> beta must increase
    updated_beta_1 = controller.update(0.10)
    assert updated_beta_1 > initial_beta, f"Beta did not increase: {updated_beta_1} <= {initial_beta}"
    assert controller.history[-1]["delta"] > 0

    # High entropy: H = 0.50 nats (> 0.25 floor) -> beta must decrease
    prev_beta = controller.beta
    for _ in range(5):
        controller.update(0.50)
    assert controller.beta < prev_beta, f"Beta did not decrease on high entropy: {controller.beta} >= {prev_beta}"


def test_entropy_floor_controller_compute_entropy():
    """Verify compute_entropy matches theoretical Shannon entropy."""
    controller = EntropyFloorController(floor_nats=0.25)

    vocab_size = 100
    # Uniform logits have maximum entropy: ln(V) nats
    uniform_logits = torch.zeros(2, 4, vocab_size)
    computed_ent = controller.compute_entropy(uniform_logits)

    expected_ent = math.log(vocab_size)
    assert abs(computed_ent.item() - expected_ent) < 1e-4

    # Test with completion mask
    # 2 sequences: seq 0 has 2 active tokens, seq 1 has 1 active token
    mask = torch.tensor([
        [1, 1, 0, 0],
        [1, 0, 0, 0]
    ], dtype=torch.bool)

    masked_ent = controller.compute_entropy(uniform_logits, mask=mask)
    assert abs(masked_ent.item() - expected_ent) < 1e-4


def test_ngram_tracker_sliding_window():
    """Verify n-grams exiting the sliding window do not trigger false loops."""
    # Window size 10 tokens: n-grams older than 10 tokens are evicted
    tracker = RollingNGramTracker(n=4, max_occurrences=2, window_size=10)

    # 4-gram [1, 2, 3, 4] appears once at beginning
    seq1 = [1, 2, 3, 4]
    for tok in seq1:
        assert not tracker.update(tok)

    # Add 12 distinct tokens to push the first 4-gram completely out of window
    distractor = list(range(100, 112))
    for tok in distractor:
        assert not tracker.update(tok)

    # Now [1, 2, 3, 4] appears twice inside the new window: should NOT trigger loop yet
    for tok in [1, 2, 3, 4, 99, 1, 2, 3, 4]:
        assert not tracker.update(tok)

    assert tracker.get_max_occurrence() == 2
    assert not tracker.loop_detected


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__]))
