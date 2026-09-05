#!/usr/bin/env python3
"""Unit tests for Item 4: Cold-Start SFT Pipeline & Benchmark Evaluation Suite.

Tests:
1. Strict prompt loss masking (tokens before assistant prompt set to -100).
2. 5-tag format adherence regex parser (positive, missing, out-of-order).
3. Boxed answer extractor with nested LaTeX curly braces.
4. SymPy mathematical symbolic equivalence verification.
5. External benchmark dataset integrity (0% contamination, valid schemas).
6. SFT dataset loading and batch collation.
"""

import json
import os
import re
import sys
import pytest
import torch

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "src"))
sys.path.insert(0, os.path.join(REPO_DIR, "benchmarks"))

from transformers import AutoTokenizer
from benchmarks.train_math_sft import ChalkMathSFTDataset, collate_fn, SYSTEM_PROMPT
from benchmarks.eval_math_benchmark import (
    check_5tag_adherence,
    extract_boxed_answer,
    clean_latex_math,
    verify_math_answer_sympy
)

SNAPSHOT_PATH = "/home/kriday/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2"
DATASET_PATH = os.path.join(REPO_DIR, "data", "chalk_seeds_500.jsonl")
EXTERNAL_EVAL_PATH = os.path.join(REPO_DIR, "data", "external_math_eval.json")


@pytest.fixture(scope="module")
def tokenizer():
    if os.path.exists(SNAPSHOT_PATH):
        tok = AutoTokenizer.from_pretrained(SNAPSHOT_PATH, trust_remote_code=True)
    else:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def test_5tag_adherence_positive():
    valid_trace = (
        "<explore>\nExploring invariants.\n</explore>\n"
        "<conjecture>\nWe conjecture answer is 42.\n</conjecture>\n"
        "<test_edge_cases>\nChecking n=1 and n=2.\n</test_edge_cases>\n"
        "<lemma_isolate>\nIsolating lemma.\n</lemma_isolate>\n"
        "<formal_proof>\nProof completes. \\boxed{42}\n</formal_proof>"
    )
    res = check_5tag_adherence(valid_trace)
    assert res["adherent"] is True
    assert res["all_tags_present"] is True
    assert res["strictly_ordered"] is True


def test_5tag_adherence_missing_tag():
    invalid_trace = (
        "<explore>\nExploring invariants.\n</explore>\n"
        "<test_edge_cases>\nChecking n=1.\n</test_edge_cases>\n"
        "<lemma_isolate>\nIsolating lemma.\n</lemma_isolate>\n"
        "<formal_proof>\nProof completes. \\boxed{42}\n</formal_proof>"
    )
    res = check_5tag_adherence(invalid_trace)
    assert res["adherent"] is False
    assert res["tags"]["conjecture"] is False


def test_5tag_adherence_out_of_order():
    out_of_order_trace = (
        "<conjecture>\nConjecturing first.\n</conjecture>\n"
        "<explore>\nExploring second.\n</explore>\n"
        "<test_edge_cases>\nChecking.\n</test_edge_cases>\n"
        "<lemma_isolate>\nIsolating.\n</lemma_isolate>\n"
        "<formal_proof>\nDone. \\boxed{42}\n</formal_proof>"
    )
    res = check_5tag_adherence(out_of_order_trace)
    assert res["adherent"] is False
    assert res["strictly_ordered"] is False


def test_boxed_extractor():
    assert extract_boxed_answer(r"The result is \boxed{42}.") == "42"
    assert extract_boxed_answer(r"Final: \boxed{\frac{7}{32}}") == r"\frac{7}{32}"
    assert extract_boxed_answer(r"Nested: \boxed{\frac{1}{1 + \sqrt{2}}}") == r"\frac{1}{1 + \sqrt{2}}"
    assert extract_boxed_answer("Fallback: boxed{99}") == "99"
    assert extract_boxed_answer("No boxed here") is None


def test_sympy_verification():
    # Identical
    assert verify_math_answer_sympy("42", "42") is True
    # LaTeX fraction vs standard division
    assert verify_math_answer_sympy("\\frac{7}{32}", "7/32") is True
    assert verify_math_answer_sympy("7/32", "\\frac{7}{32}") is True
    # Pi formatting
    assert verify_math_answer_sympy("36\\pi", "36*pi") is True
    # Algebraic equivalence
    assert verify_math_answer_sympy("2*x + 2", "2*(x + 1)") is True
    # Negative cases
    assert verify_math_answer_sympy("41", "42") is False
    assert verify_math_answer_sympy(None, "42") is False


def test_prompt_masking_strictness(tokenizer):
    dataset = ChalkMathSFTDataset(DATASET_PATH, tokenizer, max_length=512)
    sample = dataset[0]

    input_ids = sample["input_ids"]
    labels = sample["labels"]

    assert input_ids.shape == labels.shape

    # Find the prompt tokens masked to -100
    masked_mask = (labels == -100)
    unmasked_mask = (labels != -100)

    num_masked = masked_mask.sum().item()
    num_unmasked = unmasked_mask.sum().item()

    assert num_masked > 0
    assert num_unmasked > 0

    # Unmasked tokens must begin with the first token of the completion
    first_unmasked_idx = (labels != -100).nonzero(as_tuple=True)[0][0].item()
    first_token_str = tokenizer.decode([input_ids[first_unmasked_idx]])
    assert "<" in first_token_str or "explore" in first_token_str

    # Last unmasked token should be the end-of-sequence token
    last_unmasked_idx = (labels != -100).nonzero(as_tuple=True)[0][-1].item()
    last_token_str = tokenizer.decode([input_ids[last_unmasked_idx]])
    assert "<|im_end|>" in last_token_str or last_token_str == tokenizer.eos_token


def test_collate_fn(tokenizer):
    dataset = ChalkMathSFTDataset(DATASET_PATH, tokenizer, max_length=256)
    batch_raw = [dataset[0], dataset[1]]
    batch = collate_fn(batch_raw, tokenizer.pad_token_id)

    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "labels" in batch

    assert batch["input_ids"].shape[0] == 2
    assert batch["input_ids"].shape == batch["labels"].shape
    assert batch["input_ids"].shape == batch["attention_mask"].shape


def test_external_benchmark_integrity():
    assert os.path.exists(EXTERNAL_EVAL_PATH), "External benchmark JSON not found"
    with open(EXTERNAL_EVAL_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    contest = data["contest_problems"]
    degrad = data["degradation_checks"]

    assert len(contest) == 16, f"Expected 16 contest problems, got {len(contest)}"
    assert len(degrad) == 10, f"Expected 10 degradation checks, got {len(degrad)}"

    # Check 0 overlap with training dataset
    train_problems = set()
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            train_problems.add(rec["problem"].strip().lower())

    for cp in contest:
        prob_text = cp["problem"].strip().lower()
        assert prob_text not in train_problems, f"Contamination detected! Problem '{cp['id']}' in train set."
        assert cp["ground_truth"], f"Missing ground truth for {cp['id']}"
        assert cp["boxed_answer"], f"Missing boxed answer for {cp['id']}"

    for dp in degrad:
        assert dp["ground_truth"], f"Missing ground truth for {dp['id']}"
