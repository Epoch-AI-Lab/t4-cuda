#!/usr/bin/env python3
"""Unit tests for Requirement R5: Format Discrimination Replay Buffer and Reward Engine.

Tests:
1. FormatDiscriminationReplayBuffer: mix ratio bounds (15-20% general tasks, 80-85% math).
2. FormatDiscriminationReplayBuffer: sampling stability across multiple batches.
3. FormatDiscriminationRewardEngine: contest math with valid 5-stage XML yields r_format = 0.0.
4. FormatDiscriminationRewardEngine: contest math with missing or malformed tags yields r_format = -1.0.
5. FormatDiscriminationRewardEngine: general conversational prompt without XML yields r_format = 0.0.
6. FormatDiscriminationRewardEngine: general conversational prompt colonized by XML yields r_format = -1.5.
7. FormatDiscriminationRewardEngine: general coding prompt without XML yields r_format = 0.0.
8. FormatDiscriminationRewardEngine: general coding prompt colonized by XML yields r_format = -1.5.
9. FormatDiscriminationRewardEngine: batch evaluation matches individual results.
"""

import os
import sys
import pytest

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from src.rl.format_replay_buffer import (
    PromptType,
    FormatDiscriminationReplayBuffer,
    FormatDiscriminationRewardEngine,
)


def test_replay_buffer_mix_ratio():
    """Verify replay buffer draws 15-20% general prompts and 80-85% contest math."""
    buffer = FormatDiscriminationReplayBuffer(math_ratio=0.85, general_ratio=0.15, seed=1234)

    # Sample batch of 100 items
    batch_size = 100
    batch = buffer.sample_batch(batch_size)
    assert len(batch) == batch_size

    num_general = sum(1 for item in batch if item["prompt_type"] in [PromptType.GENERAL_CHAT, PromptType.GENERAL_CODE])
    num_math = sum(1 for item in batch if item["prompt_type"] == PromptType.CONTEST_MATH)

    assert num_general + num_math == batch_size
    general_pct = num_general / batch_size
    math_pct = num_math / batch_size

    assert 0.14 <= general_pct <= 0.21, f"General prompt ratio {general_pct} outside [0.14, 0.21]"
    assert 0.79 <= math_pct <= 0.86, f"Math prompt ratio {math_pct} outside [0.79, 0.86]"


def test_replay_buffer_varying_batch_sizes():
    """Verify partition allocation and math/general split across varying batch sizes."""
    buffer = FormatDiscriminationReplayBuffer(math_ratio=0.80, general_ratio=0.20, seed=42)

    for b_size in [15, 23, 37, 50, 64]:
        batch = buffer.sample_batch(b_size)
        assert len(batch) == b_size
        expected_general = int(round(b_size * 0.20))
        expected_math = b_size - expected_general
        actual_general = sum(1 for item in batch if item["prompt_type"] != PromptType.CONTEST_MATH)
        actual_math = sum(1 for item in batch if item["prompt_type"] == PromptType.CONTEST_MATH)
        assert actual_general == expected_general
        assert actual_math == expected_math


def test_format_reward_general_with_think_tags():
    """Verify conversational prompts with <think> or <cot> tags receive format penalty."""
    engine = FormatDiscriminationRewardEngine(general_penalty=-1.5)
    contaminated_chat = "<think>Let me see how to greet the user.</think> Hello! How can I help you today?"
    reward = engine.evaluate_format_compliance(PromptType.GENERAL_CHAT, contaminated_chat)
    assert reward == -1.5


def test_format_reward_math_adherent():
    """Verify compliant 5-stage XML contest math receives 0.0 format penalty."""
    engine = FormatDiscriminationRewardEngine()

    valid_completion = (
        "<explore>Analyzing problem statements.</explore>\n"
        "<conjecture>The answer is 42.</conjecture>\n"
        "<test_edge_cases>Checking edge cases n=1, 2.</test_edge_cases>\n"
        "<lemma_isolate>Lemma statement.</lemma_isolate>\n"
        "<formal_proof>Proof conclusion: \\boxed{42}.</formal_proof>"
    )

    r_format = engine.evaluate_format_compliance(PromptType.CONTEST_MATH, valid_completion)
    assert r_format == 0.0


def test_format_reward_math_non_adherent():
    """Verify missing XML tags on contest math receives -1.0 penalty."""
    engine = FormatDiscriminationRewardEngine()

    # Missing lemma_isolate and test_edge_cases
    broken_completion = (
        "<explore>Analyzing problem.</explore>\n"
        "<conjecture>Answer is 42.</conjecture>\n"
        "<formal_proof>Proof yields \\boxed{42}.</formal_proof>"
    )

    r_format = engine.evaluate_format_compliance(PromptType.CONTEST_MATH, broken_completion)
    assert r_format == -1.0

    # Completely unstructured text on math
    plain_completion = "The answer is clearly 42."
    r_plain = engine.evaluate_format_compliance(PromptType.CONTEST_MATH, plain_completion)
    assert r_plain == -1.0


def test_format_reward_general_clean():
    """Verify natural conversational response receives 0.0 penalty."""
    engine = FormatDiscriminationRewardEngine()

    clean_chat = "Hello! I am doing well, thank you. How can I help you today?"
    r_format = engine.evaluate_format_compliance(PromptType.GENERAL_CHAT, clean_chat)
    assert r_format == 0.0


def test_format_reward_general_colonized():
    """Verify conversational prompt colonized by XML tags receives -1.5 penalty."""
    engine = FormatDiscriminationRewardEngine()

    colonized_chat = (
        "<explore>The user is asking how I am doing.</explore>\n"
        "<conjecture>I should respond politely.</conjecture>\n"
        "I am doing great!"
    )

    r_format = engine.evaluate_format_compliance(PromptType.GENERAL_CHAT, colonized_chat)
    assert r_format == -1.5


def test_format_reward_code_clean():
    """Verify direct coding response without XML receives 0.0 penalty."""
    engine = FormatDiscriminationRewardEngine()

    clean_code = (
        "def binary_search(arr, target):\n"
        "    left, right = 0, len(arr) - 1\n"
        "    while left <= right:\n"
        "        mid = (left + right) // 2\n"
        "        if arr[mid] == target:\n"
        "            return mid\n"
        "        elif arr[mid] < target:\n"
        "            left = mid + 1\n"
        "        else:\n"
        "            right = mid - 1\n"
        "    return -1\n"
    )

    r_format = engine.evaluate_format_compliance(PromptType.GENERAL_CODE, clean_code)
    assert r_format == 0.0


def test_format_reward_code_colonized():
    """Verify code prompt colonized by XML tags receives -1.5 penalty."""
    engine = FormatDiscriminationRewardEngine()

    colonized_code = (
        "<explore>We implement binary search with two pointers.</explore>\n"
        "def binary_search(arr, target): pass"
    )

    r_format = engine.evaluate_format_compliance(PromptType.GENERAL_CODE, colonized_code)
    assert r_format == -1.5


def test_format_reward_batch_evaluation():
    """Verify evaluate_batch evaluates mixed batch correctly."""
    engine = FormatDiscriminationRewardEngine()

    prompt_types = [
        PromptType.CONTEST_MATH,
        PromptType.CONTEST_MATH,
        PromptType.GENERAL_CHAT,
        PromptType.GENERAL_CHAT,
    ]

    valid_math = (
        "<explore>Exploration</explore>\n"
        "<conjecture>Conjecture</conjecture>\n"
        "<test_edge_cases>Edge cases</test_edge_cases>\n"
        "<lemma_isolate>Lemma</lemma_isolate>\n"
        "<formal_proof>Proof \\boxed{1}.</formal_proof>"
    )
    invalid_math = "Just 1"
    clean_chat = "Direct answer."
    colonized_chat = "<explore>Thinking</explore>Direct answer."

    completions = [valid_math, invalid_math, clean_chat, colonized_chat]
    rewards = engine.evaluate_batch(prompt_types, completions)

    assert rewards == [0.0, -1.0, 0.0, -1.5]


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__]))
