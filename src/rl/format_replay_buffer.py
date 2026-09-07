"""Format Discrimination Replay Buffer and Reward Engine (Requirement R5).

Combines:
1. Stratified replay buffer mixing 80-85% competition math with 15-20% general chat/code.
2. Dual-mode format discrimination reward engine penalizing XML tags on general prompts
   and penalizing missing XML tags on contest math prompts.
"""

from typing import List, Dict, Any, Optional, Union
import json
import os
import random
import torch
from torch.utils.data import Dataset

from src.rl.rottweiler_verifier import XMLScaffoldValidator, TAG_NAMES


class PromptType:
    """Prompt category constants."""
    CONTEST_MATH = "contest_math"
    GENERAL_CHAT = "general_chat"
    GENERAL_CODE = "general_code"


DEFAULT_MATH_PROMPTS = [
    {
        "prompt": "Find all real roots of x^2 - 5x + 6 = 0.",
        "prompt_type": PromptType.CONTEST_MATH,
        "ground_truth": "2, 3"
    },
    {
        "prompt": "Compute the value of 7/32 + 1/32.",
        "prompt_type": PromptType.CONTEST_MATH,
        "ground_truth": "1/4"
    },
    {
        "prompt": "Determine the area of a circle with radius 6.",
        "prompt_type": PromptType.CONTEST_MATH,
        "ground_truth": "36*pi"
    },
    {
        "prompt": "Solve for x: (x + 1)^2 = x^2 + 2x + 1.",
        "prompt_type": PromptType.CONTEST_MATH,
        "ground_truth": "x"
    },
]

DEFAULT_GENERAL_PROMPTS = [
    {
        "prompt": "Hi, how are you today?",
        "prompt_type": PromptType.GENERAL_CHAT,
        "ground_truth": ""
    },
    {
        "prompt": "Can you summarize the main differences between Python lists and tuples?",
        "prompt_type": PromptType.GENERAL_CHAT,
        "ground_truth": ""
    },
    {
        "prompt": "Write a Python function to check if a string is a palindrome.",
        "prompt_type": PromptType.GENERAL_CODE,
        "ground_truth": ""
    },
    {
        "prompt": "Explain the concept of recursion in computer science.",
        "prompt_type": PromptType.GENERAL_CHAT,
        "ground_truth": ""
    },
    {
        "prompt": "Implement binary search in Python.",
        "prompt_type": PromptType.GENERAL_CODE,
        "ground_truth": ""
    },
]


class FormatDiscriminationReplayBuffer(Dataset):
    """Stratified replay buffer mixing contest math (80-85%) with general tasks (15-20%)."""

    def __init__(
        self,
        contest_math_data: Optional[List[Dict[str, Any]]] = None,
        general_data: Optional[List[Dict[str, Any]]] = None,
        math_ratio: float = 0.85,
        general_ratio: float = 0.15,
        seed: int = 42,
        math_data: Optional[List[Dict[str, Any]]] = None,
    ):
        if math_ratio <= 0 or general_ratio <= 0:
            raise ValueError("Ratios must be strictly positive")

        total_ratio = math_ratio + general_ratio
        self.math_ratio = math_ratio / total_ratio
        self.general_ratio = general_ratio / total_ratio
        self.rng = random.Random(seed)

        effective_math = contest_math_data if contest_math_data is not None else math_data
        self.contest_math_data: List[Dict[str, Any]] = (
            list(effective_math) if effective_math is not None else list(DEFAULT_MATH_PROMPTS)
        )
        self.general_data: List[Dict[str, Any]] = (
            list(general_data) if general_data is not None else list(DEFAULT_GENERAL_PROMPTS)
        )

        if not self.contest_math_data:
            self.contest_math_data = list(DEFAULT_MATH_PROMPTS)
        if not self.general_data:
            self.general_data = list(DEFAULT_GENERAL_PROMPTS)

    def __len__(self) -> int:
        return len(self.contest_math_data) + len(self.general_data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        math_len = len(self.contest_math_data)
        if idx < math_len:
            item = dict(self.contest_math_data[idx])
            item.setdefault("prompt_type", PromptType.CONTEST_MATH)
            return item
        gen_idx = idx - math_len
        item = dict(self.general_data[gen_idx % len(self.general_data)])
        item.setdefault("prompt_type", PromptType.GENERAL_CHAT)
        return item

    def sample_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """Samples a stratified batch respecting the target mix ratio."""
        if batch_size <= 0:
            return []

        num_general = int(round(batch_size * self.general_ratio))
        num_math = batch_size - num_general

        # Guarantee at least 1 of each if batch_size >= 5
        if batch_size >= 5:
            num_general = max(1, min(batch_size - 1, num_general))
            num_math = batch_size - num_general

        batch: List[Dict[str, Any]] = []

        # Sample math prompts
        for _ in range(num_math):
            item = dict(self.rng.choice(self.contest_math_data))
            item.setdefault("prompt_type", PromptType.CONTEST_MATH)
            batch.append(item)

        # Sample general prompts
        for _ in range(num_general):
            item = dict(self.rng.choice(self.general_data))
            item.setdefault("prompt_type", PromptType.GENERAL_CHAT)
            batch.append(item)

        self.rng.shuffle(batch)
        return batch

    def get_mix_ratio(self) -> Dict[str, float]:
        """Returns the configured mixture ratio."""
        return {
            "math_ratio": self.math_ratio,
            "general_ratio": self.general_ratio,
        }


class FormatDiscriminationRewardEngine:
    """Dual-mode format reward evaluator.

    Penalizes:
    - Missing 5-stage XML tags on contest math prompts (r_format = -1.0)
    - Hallucinated XML tags on general conversational/coding prompts (r_format = -1.5)
    """

    def __init__(
        self,
        math_valid_reward: float = 0.0,
        math_penalty: float = -1.0,
        general_clean_reward: float = 0.0,
        general_penalty: float = -1.5
    ):
        self.math_valid_reward = math_valid_reward
        self.math_penalty = math_penalty
        self.general_clean_reward = general_clean_reward
        self.general_penalty = general_penalty
        self.scaffold_validator = XMLScaffoldValidator()

    def has_xml_tags(self, text: str) -> bool:
        """Returns True if any reasoning tag or standard XML-like tag is present."""
        all_tags = set(TAG_NAMES) | {"think", "cot", "scratchpad", "reasoning", "thought", "solution"}
        for tag in all_tags:
            if f"<{tag}>" in text or f"</{tag}>" in text or f"<{tag}" in text:
                return True
        return False

    def evaluate_format_compliance(self, prompt_type: str, completion: str) -> float:
        """Calculates conditional format reward based on prompt type."""
        pt = str(prompt_type).lower()

        if pt == PromptType.CONTEST_MATH:
            report = self.scaffold_validator.validate(completion)
            if report["adherent"]:
                return self.math_valid_reward
            return self.math_penalty

        # General conversational or coding prompts
        if self.has_xml_tags(completion):
            return self.general_penalty
        return self.general_clean_reward

    def compute_format_reward(self, prompt_type: str, completion_text: str) -> float:
        """Alias for evaluate_format_compliance."""
        return self.evaluate_format_compliance(prompt_type, completion_text)

    def evaluate_batch(
        self,
        prompt_types: List[str],
        completions: List[str]
    ) -> List[float]:
        """Evaluates format rewards across a batch."""
        return [
            self.evaluate_format_compliance(pt, comp)
            for pt, comp in zip(prompt_types, completions)
        ]
