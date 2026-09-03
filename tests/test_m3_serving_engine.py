#!/usr/bin/env python3
"""Unit tests for Milestone 3: UnifiedSpeculativeEngine Serving Architecture.

Tests cover:
- Suite 1: Verification Logic Correctness (TestVerificationLogic)
- Suite 2: KV Cache Pointer and Rollback Consistency (TestKVCachePointerAndRollbackConsistency)
- Suite 3: Telemetry and Metrics Calculation (TestTelemetryAndMetrics)
- Suite 4: Step-for-Step Autoregressive Parity (TestAutoregressiveGenerationParity)
- Suite 5: Termination and Boundary Protections (TestTerminationAndBoundaryProtections)

Designed to run fast (<2s) on CPU using synthetic and mock fixtures without external GPU weights.
"""

import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.static_kv_cache import StaticKVCache, build_causal_4d_mask, compute_position_ids
from src.unified_draft_engine import PromptLookupDraftEngine
from src.unified_speculative_engine import UnifiedSpeculativeEngine


# ==============================================================================
# Synthetic Test Fixtures
# ==============================================================================

class MockTokenizer:
    """Minimal tokenizer fixture providing encode and decode interfaces."""

    def __init__(self, vocab_size: int = 1000, eos_token_id: int = 999, pad_token_id: int = 0):
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

    def encode(self, text: str, return_tensors: Optional[str] = None) -> Any:
        tokens = [int(tok) for tok in text.split() if tok.isdigit()]
        if not tokens:
            tokens = [hash(word) % (self.vocab_size - 10) + 1 for word in text.split()]
        if return_tensors == "pt":
            return torch.tensor([tokens], dtype=torch.long)
        return tokens

    def decode(self, token_ids: Any, skip_special_tokens: bool = False) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.reshape(-1).tolist()
        return " ".join(str(t) for t in token_ids)


class DeterministicGroundTruthModel(nn.Module):
    """Target model simulator predicting sequence tokens from a ground-truth list.

    At sequence position p = start_pos + i, predicts ground_truth[p + offset].
    By default offset=1, matching causal LM next-token prediction x_{t+1}.
    """

    def __init__(
        self,
        ground_truth_tokens: List[int],
        vocab_size: int = 1000,
        offset: int = 1,
        num_layers: int = 2,
        num_heads: int = 2,
        head_dim: int = 16,
    ):
        super().__init__()
        self.ground_truth = list(ground_truth_tokens)
        self.vocab_size = vocab_size
        self.offset = offset
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.forward_call_count = 0
        self.evaluated_token_counts: List[int] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[Any] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        **kwargs,
    ):
        self.forward_call_count += 1
        batch_size, query_len = input_ids.shape
        self.evaluated_token_counts.append(query_len)

        if past_key_values is not None and hasattr(past_key_values, "current_pos"):
            start_pos = past_key_values.current_pos
        elif position_ids is not None:
            start_pos = int(position_ids[0, 0].item())
        else:
            start_pos = 0

        logits = torch.zeros((batch_size, query_len, self.vocab_size), dtype=torch.float32, device=input_ids.device)

        for i in range(query_len):
            target_idx = start_pos + i + self.offset
            if 0 <= target_idx < len(self.ground_truth):
                target_tok = self.ground_truth[target_idx]
            else:
                target_tok = 0
            logits[:, i, target_tok] = 10.0

        if past_key_values is not None and use_cache:
            k_synth = torch.ones(
                (batch_size, self.num_heads, query_len, self.head_dim),
                dtype=past_key_values.dtype,
                device=past_key_values.device,
            )
            v_synth = torch.ones(
                (batch_size, self.num_heads, query_len, self.head_dim),
                dtype=past_key_values.dtype,
                device=past_key_values.device,
            )
            for l in range(past_key_values.num_layers):
                past_key_values.update(l, k_synth, v_synth, start_pos=start_pos)

        class ModelOutput:
            def __init__(self, logits, past_key_values):
                self.logits = logits
                self.past_key_values = past_key_values

        return ModelOutput(logits=logits, past_key_values=past_key_values)


class ProgrammableAcceptanceTargetModel(nn.Module):
    """Target model simulator executing programmed verification decisions per round.

    Supports:
    - prefill: emits specified prefill_token (or input_ids[-1])
    - 'accept_all': predicts candidate j at index j, bonus_token at index m
    - 'reject_at': predicts candidate j for j < index, correction_token at index
    - 'custom_next': predicts custom token for single-token fallback
    """

    def __init__(
        self,
        round_actions: List[Dict[str, Any]],
        vocab_size: int = 1000,
        bonus_token_id: int = 777,
        default_correction_id: int = 888,
        prefill_token: int = 3,
        num_layers: int = 2,
        num_heads: int = 2,
        head_dim: int = 16,
    ):
        super().__init__()
        self.round_actions = round_actions
        self.vocab_size = vocab_size
        self.bonus_token_id = bonus_token_id
        self.default_correction_id = default_correction_id
        self.prefill_token = prefill_token
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.forward_call_count = 0
        self.speculative_round_idx = 0
        self.evaluated_token_counts: List[int] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[Any] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        **kwargs,
    ):
        self.forward_call_count += 1
        batch_size, query_len = input_ids.shape
        self.evaluated_token_counts.append(query_len)

        start_pos = past_key_values.current_pos if past_key_values is not None else 0
        logits = torch.zeros((batch_size, query_len, self.vocab_size), dtype=torch.float32, device=input_ids.device)

        # Prefill call (first forward call)
        if self.forward_call_count == 1:
            logits[:, -1, self.prefill_token] = 12.0
        else:
            action = (
                self.round_actions[self.speculative_round_idx]
                if self.speculative_round_idx < len(self.round_actions)
                else {"type": "accept_all"}
            )
            self.speculative_round_idx += 1
            action_type = action.get("type", "accept_all")
            m = query_len - 1  # number of candidates (since query_len = m + 1)

            if m == 0:
                # Single token fallback decode
                next_tok = action.get("next_token", self.prefill_token)
                logits[:, 0, next_tok] = 12.0
            elif action_type == "accept_all":
                # Candidate j is located at input_ids[0, j + 1]. Logit j predicts candidate j.
                for j in range(m):
                    cand_tok = int(input_ids[0, j + 1].item())
                    logits[:, j, cand_tok] = 12.0
                # Final position predicts bonus token
                bonus = action.get("bonus_token", self.bonus_token_id)
                logits[:, m, bonus] = 15.0

            elif action_type == "reject_at":
                reject_idx = action.get("index", 0)
                correction = action.get("correction_token", self.default_correction_id)
                for j in range(m):
                    if j < reject_idx:
                        cand_tok = int(input_ids[0, j + 1].item())
                        logits[:, j, cand_tok] = 12.0
                    elif j == reject_idx:
                        logits[:, j, correction] = 12.0
                    else:
                        logits[:, j, 0] = 5.0

        if past_key_values is not None and use_cache:
            k_synth = torch.ones(
                (batch_size, self.num_heads, query_len, self.head_dim),
                dtype=past_key_values.dtype,
                device=past_key_values.device,
            )
            v_synth = torch.ones(
                (batch_size, self.num_heads, query_len, self.head_dim),
                dtype=past_key_values.dtype,
                device=past_key_values.device,
            )
            for l in range(past_key_values.num_layers):
                past_key_values.update(l, k_synth, v_synth, start_pos=start_pos)

        class ModelOutput:
            def __init__(self, logits, past_key_values):
                self.logits = logits
                self.past_key_values = past_key_values

        return ModelOutput(logits=logits, past_key_values=past_key_values)


class TinyTransformerModel(nn.Module):
    """Tiny neural network testing genuine PyTorch tensor arithmetic without GPU requirements."""

    def __init__(self, vocab_size: int = 64, hidden_dim: int = 32, num_heads: int = 2, num_layers: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.head_dim = hidden_dim // num_heads

        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[Any] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        **kwargs,
    ):
        x = self.embed(input_ids)
        batch_size, seq_len, _ = x.shape
        start_pos = past_key_values.current_pos if past_key_values is not None else 0

        for l_idx, layer in enumerate(self.layers):
            x = layer(x)
            if past_key_values is not None and use_cache:
                k = x.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                v = x.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                past_key_values.update(l_idx, k, v, start_pos=start_pos)

        logits = self.lm_head(x)

        class ModelOutput:
            def __init__(self, logits, past_key_values):
                self.logits = logits
                self.past_key_values = past_key_values

        return ModelOutput(logits=logits, past_key_values=past_key_values)


# ==============================================================================
# Suite 1: Verification Logic Correctness (TestVerificationLogic)
# ==============================================================================

class TestVerificationLogic:
    """Verifies single-pass greedy verification, bonus emission, and rollback logic."""

    def test_verification_100_percent_acceptance(self):
        """100% acceptance emits all K candidates plus 1 bonus token in single pass."""
        # Prompt crafted so trailing 3-gram [1, 2, 3] matches index 0 and proposes [10, 20, 30]
        prompt = [1, 2, 3, 10, 20, 30, 99, 1, 2]
        actions = [{"type": "accept_all", "bonus_token": 777}]

        model = ProgrammableAcceptanceTargetModel(round_actions=actions, prefill_token=3)
        cache = StaticKVCache(max_batch_size=1, max_seq_len=64, num_layers=2, num_heads=2, head_dim=16, device="cpu")
        draft_engine = PromptLookupDraftEngine(max_ngram_size=3, min_ngram_size=1)

        engine = UnifiedSpeculativeEngine(
            target_model=model,
            draft_engine=draft_engine,
            static_cache=cache,
            device="cpu",
        )

        out_tokens, stats = engine.generate(prompt, max_new_tokens=5, k_draft=3)
        # Emitted tokens after prompt: prefill seed [3], then Round 1 [10, 20, 30, 777]
        new_tokens = out_tokens[0, len(prompt):].tolist()

        assert new_tokens == [3, 10, 20, 30, 777]
        assert stats["accepted_tokens"] == 3
        assert stats["draft_tokens"] == 3
        assert stats["acceptance_rate"] == 1.0
        assert stats["target_forward_passes"] == 2  # 1 prefill + 1 speculative round

    def test_verification_0_percent_acceptance(self):
        """0% acceptance rejects candidate 0, emits 1 correction token, rolls back cache."""
        prompt = [1, 2, 3, 10, 20, 30, 99, 1, 2]
        actions = [{"type": "reject_at", "index": 0, "correction_token": 888}]

        model = ProgrammableAcceptanceTargetModel(round_actions=actions, prefill_token=3)
        cache = StaticKVCache(max_batch_size=1, max_seq_len=64, num_layers=2, num_heads=2, head_dim=16, device="cpu")
        draft_engine = PromptLookupDraftEngine(max_ngram_size=3, min_ngram_size=1)

        engine = UnifiedSpeculativeEngine(
            target_model=model,
            draft_engine=draft_engine,
            static_cache=cache,
            device="cpu",
        )

        out_tokens, stats = engine.generate(prompt, max_new_tokens=2, k_draft=3)
        new_tokens = out_tokens[0, len(prompt):].tolist()

        assert new_tokens == [3, 888]
        assert stats["accepted_tokens"] == 0
        assert stats["draft_tokens"] == 3
        assert stats["acceptance_rate"] == 0.0

    def test_verification_partial_acceptance_mid_sequence(self):
        """Partial acceptance accepts candidates 0 and 1, rejects candidate 2, emits correction."""
        prompt = [1, 2, 3, 10, 20, 30, 40, 99, 1, 2]
        actions = [{"type": "reject_at", "index": 2, "correction_token": 888}]

        model = ProgrammableAcceptanceTargetModel(round_actions=actions, prefill_token=3)
        cache = StaticKVCache(max_batch_size=1, max_seq_len=64, num_layers=2, num_heads=2, head_dim=16, device="cpu")
        draft_engine = PromptLookupDraftEngine(max_ngram_size=3, min_ngram_size=1)

        engine = UnifiedSpeculativeEngine(
            target_model=model,
            draft_engine=draft_engine,
            static_cache=cache,
            device="cpu",
        )

        out_tokens, stats = engine.generate(prompt, max_new_tokens=4, k_draft=4)
        new_tokens = out_tokens[0, len(prompt):].tolist()

        assert new_tokens == [3, 10, 20, 888]
        assert stats["accepted_tokens"] == 2
        assert stats["draft_tokens"] == 4
        assert abs(stats["acceptance_rate"] - 0.5) < 1e-6

    def test_verification_lookahead_k_one(self):
        """Single-token lookahead K=1 executes cleanly without indexing anomalies."""
        prompt = [1, 2, 3, 50, 99, 1, 2]

        # Case A: Accepted K=1 candidate + bonus token
        actions_accept = [{"type": "accept_all", "bonus_token": 777}]
        model_a = ProgrammableAcceptanceTargetModel(round_actions=actions_accept, prefill_token=3)
        cache_a = StaticKVCache(max_seq_len=64, device="cpu")
        engine_a = UnifiedSpeculativeEngine(target_model=model_a, static_cache=cache_a, device="cpu")
        out_a, stats_a = engine_a.generate(prompt, max_new_tokens=3, k_draft=1)
        assert out_a[0, len(prompt):].tolist() == [3, 50, 777]
        assert stats_a["accepted_tokens"] == 1
        assert stats_a["draft_tokens"] == 1

        # Case B: Rejected K=1 candidate -> correction token emitted
        actions_reject = [{"type": "reject_at", "index": 0, "correction_token": 666}]
        model_b = ProgrammableAcceptanceTargetModel(round_actions=actions_reject, prefill_token=3)
        cache_b = StaticKVCache(max_seq_len=64, device="cpu")
        engine_b = UnifiedSpeculativeEngine(target_model=model_b, static_cache=cache_b, device="cpu")
        out_b, stats_b = engine_b.generate(prompt, max_new_tokens=2, k_draft=1)
        assert out_b[0, len(prompt):].tolist() == [3, 666]
        assert stats_b["accepted_tokens"] == 0
        assert stats_b["draft_tokens"] == 1

    def test_verification_empty_draft_fallback(self):
        """When prompt lookup finds 0 matches, engine falls back to standard decode step."""
        # Non-repeating unique tokens produce 0 draft proposals
        prompt = [101, 102, 103, 104, 105]
        actions = [{"type": "accept_all", "next_token": 200}]

        model = ProgrammableAcceptanceTargetModel(round_actions=actions, prefill_token=106)
        cache = StaticKVCache(max_seq_len=64, device="cpu")
        engine = UnifiedSpeculativeEngine(target_model=model, static_cache=cache, device="cpu")

        out, stats = engine.generate(prompt, max_new_tokens=2, k_draft=3)
        assert stats["draft_tokens"] == 0
        assert stats["accepted_tokens"] == 0
        assert stats["acceptance_rate"] == 0.0
        assert len(out[0]) == len(prompt) + 2

    def test_verification_multi_round_alternating_pattern(self):
        """Alternating accept/reject pattern across 4 rounds tracks exact tokens and counts."""
        prompt = [1, 2, 3, 10, 20, 99, 1, 2]
        actions = [
            {"type": "accept_all", "bonus_token": 100},                  # Round 1 (K=2): 2 accepted, 1 bonus
            {"type": "reject_at", "index": 0, "correction_token": 200},     # Round 2 (K=2): 0 accepted, 1 correction
            {"type": "accept_all", "bonus_token": 300},                  # Round 3 (K=3): 3 accepted, 1 bonus
            {"type": "reject_at", "index": 1, "correction_token": 400},     # Round 4 (K=3): 1 accepted, 1 correction
        ]

        class SequencedDraftEngine:
            def __init__(self):
                self.proposals = [
                    [10, 20],
                    [11, 21],
                    [30, 40, 50],
                    [31, 41, 51],
                ]
                self.call_idx = 0

            def propose(self, input_ids, k_draft=3, **kwargs):
                if self.call_idx < len(self.proposals):
                    cands = self.proposals[self.call_idx]
                    self.call_idx += 1
                    return torch.tensor([cands], dtype=torch.long), None
                return torch.empty((1, 0), dtype=torch.long), None

        model = ProgrammableAcceptanceTargetModel(round_actions=actions, prefill_token=3)
        cache = StaticKVCache(max_seq_len=128, device="cpu")
        draft_engine = SequencedDraftEngine()
        engine = UnifiedSpeculativeEngine(target_model=model, draft_engine=draft_engine, static_cache=cache, device="cpu")

        out, stats = engine.generate(prompt, max_new_tokens=11, k_draft=3)
        # Expected: Round 1 (2 acc) + Round 2 (0 acc) + Round 3 (3 acc) + Round 4 (1 acc) = 6 accepted
        assert stats["accepted_tokens"] == 6
        assert stats["draft_tokens"] == 10
        assert stats["speculative_rounds"] == 4


# ==============================================================================
# Suite 2: KV Cache Pointer and Rollback Consistency (TestKVCachePointerAndRollbackConsistency)
# ==============================================================================

class TestKVCachePointerAndRollbackConsistency:
    """Verifies static memory invariance, write pointer mechanics, and rollback integrity."""

    def test_prefill_pointer_initialization(self):
        """Prefill advances current_pos and get_seq_length to prompt length P."""
        prompt = list(range(12))
        model = DeterministicGroundTruthModel(ground_truth_tokens=list(range(50)))
        cache = StaticKVCache(max_seq_len=64, num_layers=2, num_heads=2, head_dim=16, device="cpu")
        engine = UnifiedSpeculativeEngine(target_model=model, static_cache=cache, device="cpu")

        engine.generate(prompt, max_new_tokens=1, k_draft=3)
        # Prefill wrote 12 tokens, seed token was emitted without speculative round
        assert cache.current_pos == 12
        assert cache.get_seq_length() == 12

    def test_tentative_write_advances_pointer(self):
        """Tentative candidate evaluation writes to buffer before rollback resets pointer."""
        cache = StaticKVCache(max_seq_len=64, num_layers=2, num_heads=2, head_dim=16, device="cpu")
        # Prefill 10 tokens
        k_prefill = torch.ones((1, 2, 10, 16), dtype=torch.float32)
        cache.update(0, k_prefill, k_prefill, start_pos=0)
        cache.update(1, k_prefill, k_prefill, start_pos=0)
        assert cache.current_pos == 10

        # Tentative write of eval_tokens (last_token + K=3 candidates = 4 tokens)
        k_tentative = torch.ones((1, 2, 4, 16), dtype=torch.float32)
        cache.update(0, k_tentative, k_tentative, start_pos=10)
        cache.update(1, k_tentative, k_tentative, start_pos=10)
        assert cache.current_pos == 14

        # Rollback on 1 accepted candidate: start_pos (10) + 1 (last_token) + 1 (cand 0) = 12
        cache.rollback(12)
        assert cache.current_pos == 12
        assert cache.get_seq_length() == 12

    def test_rollback_on_partial_rejection(self):
        """Partial rejection at candidate r resets pointer to prefix + 1 + r."""
        prompt = [1, 2, 3, 10, 20, 30, 40, 99, 1, 2]
        actions = [{"type": "reject_at", "index": 1, "correction_token": 888}]

        model = ProgrammableAcceptanceTargetModel(round_actions=actions, prefill_token=3)
        cache = StaticKVCache(max_seq_len=64, num_layers=2, num_heads=2, head_dim=16, device="cpu")
        engine = UnifiedSpeculativeEngine(target_model=model, static_cache=cache, device="cpu")

        # Prefill length is 10. In Round 1, start_pos is 10.
        # Candidate 0 accepted (1 accepted). Rejection at candidate 1.
        # Committed pos = 10 + 1 + 1 = 12.
        engine.generate(prompt, max_new_tokens=3, k_draft=4)
        assert cache.current_pos == 12
        assert cache.get_seq_length() == 12

    def test_rollback_on_zero_acceptance(self):
        """Zero acceptance resets pointer to prefix + 1, invalidating all proposed slots."""
        prompt = [1, 2, 3, 10, 20, 30, 99, 1, 2]
        actions = [{"type": "reject_at", "index": 0, "correction_token": 888}]

        model = ProgrammableAcceptanceTargetModel(round_actions=actions, prefill_token=3)
        cache = StaticKVCache(max_seq_len=64, num_layers=2, num_heads=2, head_dim=16, device="cpu")
        engine = UnifiedSpeculativeEngine(target_model=model, static_cache=cache, device="cpu")

        # Prefill length is 9. Rejection at index 0. Committed pos = 9 + 1 + 0 = 10.
        engine.generate(prompt, max_new_tokens=2, k_draft=3)
        assert cache.current_pos == 10
        assert cache.get_seq_length() == 10

    def test_cache_storage_data_ptr_invariance(self):
        """Storage tensor memory address remains strictly invariant across speculative rounds."""
        model = DeterministicGroundTruthModel(ground_truth_tokens=list(range(100)))
        cache = StaticKVCache(max_seq_len=128, num_layers=2, num_heads=2, head_dim=16, device="cpu")
        engine = UnifiedSpeculativeEngine(target_model=model, static_cache=cache, device="cpu")

        initial_storage_ptr = cache.storage.data_ptr()
        initial_layer_ptrs = [cache.key_cache[i].data_ptr() for i in range(cache.num_layers)]

        prompt = [1, 2, 3, 4, 5, 1, 2, 3]
        engine.generate(prompt, max_new_tokens=20, k_draft=3)

        assert cache.storage.data_ptr() == initial_storage_ptr
        for i in range(cache.num_layers):
            assert cache.key_cache[i].data_ptr() == initial_layer_ptrs[i]

    def test_unwritten_buffer_invariance(self):
        """4D causal attention mask excludes rolled-back or unwritten positions with -inf."""
        query_len = 3
        past_kv_len = 10
        total_len = past_kv_len + query_len
        mask = build_causal_4d_mask(1, query_len, past_kv_len, device=torch.device("cpu"), dtype=torch.float32)

        # Keys in past context [0 ... 9] are unmasked (0.0)
        assert torch.all(mask[0, 0, :, :past_kv_len] == 0.0)
        # Query 0 attends to past and pos 10, but not 11 or 12
        assert mask[0, 0, 0, 10] == 0.0
        assert mask[0, 0, 0, 11] < -1e8
        assert mask[0, 0, 0, 12] < -1e8
        # Query 1 attends to past, pos 10, 11, but not 12
        assert mask[0, 0, 1, 11] == 0.0
        assert mask[0, 0, 1, 12] < -1e8


# ==============================================================================
# Suite 3: Telemetry and Metrics Calculation (TestTelemetryAndMetrics)
# ==============================================================================

class TestTelemetryAndMetrics:
    """Verifies schema compliance, mathematical precision, and edge-case safety of telemetry."""

    def test_telemetry_schema_compliance(self):
        """Stats dictionary conforms strictly to PROJECT.md and explorer schema requirements."""
        model = DeterministicGroundTruthModel(ground_truth_tokens=list(range(100)))
        cache = StaticKVCache(max_seq_len=64, device="cpu")
        engine = UnifiedSpeculativeEngine(target_model=model, static_cache=cache, device="cpu")

        prompt = [1, 2, 3, 1, 2, 3]
        _, stats = engine.generate(prompt, max_new_tokens=8, k_draft=3)

        required_keys = [
            "num_tokens",
            "total_time_s",
            "tokens_per_sec",
            "lookahead_k",
            "acceptance_rate",
            "accepted_tokens",
            "draft_tokens",
            "target_forward_passes",
        ]
        for key in required_keys:
            assert key in stats, f"Missing required telemetry key: {key}"
            assert stats[key] is not None

        assert isinstance(stats["num_tokens"], int)
        assert isinstance(stats["total_time_s"], float)
        assert isinstance(stats["tokens_per_sec"], float)
        assert isinstance(stats["acceptance_rate"], float)

    def test_acceptance_rate_mathematical_precision(self):
        """Acceptance rate matches accepted_tokens / draft_tokens with high precision."""
        prompt = [1, 2, 3, 10, 20, 30, 40, 50, 99, 1, 2, 3, 10, 20, 99, 1, 2]
        # Round 1 (K=5): 2 accepted, 3 rejected (index 2 mismatch). Emits 99. 2/5 = 0.40
        # Round 2 (K=5): 3 accepted, 2 rejected (index 3 mismatch). Emits 999. 3/5 = 0.60
        # Combined: 5 accepted / 10 proposed = 0.50 exactly
        actions = [
            {"type": "reject_at", "index": 2, "correction_token": 99},
            {"type": "reject_at", "index": 3, "correction_token": 999},
        ]

        model = ProgrammableAcceptanceTargetModel(round_actions=actions, prefill_token=3)
        cache = StaticKVCache(max_seq_len=128, device="cpu")
        engine = UnifiedSpeculativeEngine(target_model=model, static_cache=cache, device="cpu")

        _, stats = engine.generate(prompt, max_new_tokens=8, k_draft=5)
        assert stats["draft_tokens"] == 10
        assert stats["accepted_tokens"] == 5
        assert abs(stats["acceptance_rate"] - 0.50) < 1e-6
        assert abs(stats["acceptance_rate_pct"] - 50.0) < 1e-4

    def test_acceptance_rate_zero_draft_tokens_safe(self):
        """When draft_tokens == 0, acceptance_rate defaults cleanly to 0.0 without ZeroDivisionError."""
        prompt = [101, 102, 103, 104, 105]
        model = ProgrammableAcceptanceTargetModel(round_actions=[], prefill_token=106)
        cache = StaticKVCache(max_seq_len=64, device="cpu")
        engine = UnifiedSpeculativeEngine(target_model=model, static_cache=cache, device="cpu")

        _, stats = engine.generate(prompt, max_new_tokens=1, k_draft=3)
        assert stats["draft_tokens"] == 0
        assert stats["accepted_tokens"] == 0
        assert stats["acceptance_rate"] == 0.0
        assert stats["acceptance_rate_pct"] == 0.0

    def test_target_forward_passes_count(self):
        """target_forward_passes equals exactly 1 (prefill) + speculative_rounds."""
        prompt = [1, 2, 3, 10, 20, 99, 1, 2]
        actions = [{"type": "accept_all", "bonus_token": 777}]

        model = ProgrammableAcceptanceTargetModel(round_actions=actions, prefill_token=3)
        cache = StaticKVCache(max_seq_len=64, device="cpu")
        engine = UnifiedSpeculativeEngine(target_model=model, static_cache=cache, device="cpu")

        _, stats = engine.generate(prompt, max_new_tokens=4, k_draft=2)
        assert stats["target_forward_passes"] == model.forward_call_count
        assert stats["target_forward_passes"] == 1 + stats["speculative_rounds"]

    def test_tokens_per_second_calculation(self):
        """tokens_per_sec is mathematically consistent with num_tokens / total_time_s."""
        model = DeterministicGroundTruthModel(ground_truth_tokens=list(range(50)))
        cache = StaticKVCache(max_seq_len=64, device="cpu")
        engine = UnifiedSpeculativeEngine(target_model=model, static_cache=cache, device="cpu")

        _, stats = engine.generate([1, 2, 3, 4], max_new_tokens=6, k_draft=2)
        expected_tok_sec = stats["num_tokens"] / stats["total_time_s"] if stats["total_time_s"] > 0 else 0.0
        assert abs(stats["tokens_per_sec"] - expected_tok_sec) < 1e-4


# ==============================================================================
# Suite 4: Step-for-Step Autoregressive Parity (TestAutoregressiveGenerationParity)
# ==============================================================================

class TestAutoregressiveGenerationParity:
    """Verifies that speculative decoding produces token-for-token identical output to autoregressive decoding."""

    def test_greedy_equivalence_repetitive_prompt(self):
        """High speculative hit rate prompt matches autoregressive greedy decode token-for-token."""
        # Ground truth has repeating cycle: 1, 2, 3, 4, 5, 1, 2, 3, 4, 5...
        gt = [1, 2, 3, 4, 5] * 20
        prompt = [1, 2, 3, 4, 5, 1, 2]

        model_spec = DeterministicGroundTruthModel(ground_truth_tokens=gt)
        cache_spec = StaticKVCache(max_seq_len=128, device="cpu")
        engine_spec = UnifiedSpeculativeEngine(target_model=model_spec, static_cache=cache_spec, device="cpu")

        model_ar = DeterministicGroundTruthModel(ground_truth_tokens=gt)
        cache_ar = StaticKVCache(max_seq_len=128, device="cpu")
        engine_ar = UnifiedSpeculativeEngine(target_model=model_ar, static_cache=cache_ar, device="cpu")

        out_spec, stats_spec = engine_spec.generate(prompt, max_new_tokens=15, k_draft=3)
        out_ar, stats_ar = engine_ar.generate_autoregressive(prompt, max_new_tokens=15)

        assert out_spec.tolist() == out_ar.tolist()
        assert stats_spec["num_tokens"] == stats_ar["num_tokens"]

    def test_greedy_equivalence_zero_match_prompt(self):
        """0% draft hit rate prompt matches autoregressive greedy decode token-for-token."""
        gt = list(range(100))
        prompt = [10, 20, 30, 40, 50]  # Non-repeating unique tokens

        model_spec = DeterministicGroundTruthModel(ground_truth_tokens=gt)
        cache_spec = StaticKVCache(max_seq_len=128, device="cpu")
        engine_spec = UnifiedSpeculativeEngine(target_model=model_spec, static_cache=cache_spec, device="cpu")

        model_ar = DeterministicGroundTruthModel(ground_truth_tokens=gt)
        cache_ar = StaticKVCache(max_seq_len=128, device="cpu")
        engine_ar = UnifiedSpeculativeEngine(target_model=model_ar, static_cache=cache_ar, device="cpu")

        out_spec, _ = engine_spec.generate(prompt, max_new_tokens=10, k_draft=3)
        out_ar, _ = engine_ar.generate_autoregressive(prompt, max_new_tokens=10)

        assert out_spec.tolist() == out_ar.tolist()

    def test_greedy_equivalence_varied_k_draft(self):
        """Generated token sequence is identical across all lookahead parameters K in {1, 2, 3, 4, 5, 8}."""
        gt = [1, 2, 3, 4] * 25
        prompt = [1, 2, 3, 4, 1, 2]
        k_values = [1, 2, 3, 4, 5, 8]
        generated_sequences = []

        for k in k_values:
            model = DeterministicGroundTruthModel(ground_truth_tokens=gt)
            cache = StaticKVCache(max_seq_len=128, device="cpu")
            engine = UnifiedSpeculativeEngine(target_model=model, static_cache=cache, device="cpu")
            out, _ = engine.generate(prompt, max_new_tokens=12, k_draft=k)
            generated_sequences.append(out.tolist())

        # All output sequences must be pairwise identical
        first_seq = generated_sequences[0]
        for seq in generated_sequences[1:]:
            assert seq == first_seq

    def test_kv_cache_state_parity_after_generation(self):
        """Committed key and value cache slices match between speculative and autoregressive decode."""
        gt = [1, 2, 3, 4, 5] * 10
        prompt = [1, 2, 3, 4, 1, 2]

        model_spec = DeterministicGroundTruthModel(ground_truth_tokens=gt)
        cache_spec = StaticKVCache(max_seq_len=64, num_layers=2, num_heads=2, head_dim=16, device="cpu")
        engine_spec = UnifiedSpeculativeEngine(target_model=model_spec, static_cache=cache_spec, device="cpu")

        model_ar = DeterministicGroundTruthModel(ground_truth_tokens=gt)
        cache_ar = StaticKVCache(max_seq_len=64, num_layers=2, num_heads=2, head_dim=16, device="cpu")
        engine_ar = UnifiedSpeculativeEngine(target_model=model_ar, static_cache=cache_ar, device="cpu")

        engine_spec.generate(prompt, max_new_tokens=8, k_draft=3)
        engine_ar.generate_autoregressive(prompt, max_new_tokens=8)

        # Committed key and value states match within numerical tolerance
        min_pos = min(cache_spec.current_pos, cache_ar.current_pos)
        k_spec, v_spec = cache_spec.get_valid_cache(0, current_pos=min_pos)
        k_ar, v_ar = cache_ar.get_valid_cache(0, current_pos=min_pos)
        assert torch.allclose(k_spec, k_ar)
        assert torch.allclose(v_spec, v_ar)
        assert min_pos >= len(prompt)


# ==============================================================================
# Suite 5: Termination and Boundary Protections (TestTerminationAndBoundaryProtections)
# ==============================================================================

class TestTerminationAndBoundaryProtections:
    """Verifies EOS halts, budget enforcement, state isolation, and buffer overflow safety."""

    def test_termination_on_eos_in_draft_candidate(self):
        """Generation halts immediately when an accepted draft candidate contains EOS."""
        eos = 999
        gt = [1, 2, 3, 10, eos, 30, 40]
        # Prompt matches 1, 2, 3 proposing [10, eos, 30]
        prompt = [1, 2, 3, 10, eos, 30, 99, 1, 2]

        actions = [{"type": "accept_all", "bonus_token": 777}]
        model = ProgrammableAcceptanceTargetModel(
            round_actions=actions,
            prefill_token=3,
        )
        cache = StaticKVCache(max_seq_len=64, device="cpu")
        tokenizer = MockTokenizer(eos_token_id=eos)
        engine = UnifiedSpeculativeEngine(target_model=model, static_cache=cache, tokenizer=tokenizer, device="cpu")

        out, _ = engine.generate(prompt, max_new_tokens=10, k_draft=3)
        tokens = out[0].tolist()

        # Token sequence ends with EOS, trailing candidate 30 is dropped
        assert tokens[-1] == eos
        assert tokens[len(prompt):] == [3, 10, eos]

    def test_termination_on_eos_in_correction(self):
        """Generation halts immediately when target emits EOS as correction token."""
        eos = 999
        prompt = [1, 2, 3, 10, 20, 30, 99, 1, 2]
        actions = [{"type": "reject_at", "index": 0, "correction_token": eos}]

        model = ProgrammableAcceptanceTargetModel(round_actions=actions, prefill_token=3)
        cache = StaticKVCache(max_seq_len=64, device="cpu")
        tokenizer = MockTokenizer(eos_token_id=eos)
        engine = UnifiedSpeculativeEngine(target_model=model, static_cache=cache, tokenizer=tokenizer, device="cpu")

        out, _ = engine.generate(prompt, max_new_tokens=5, k_draft=3)
        tokens = out[0].tolist()

        assert tokens[-1] == eos
        assert tokens[len(prompt):] == [3, eos]

    def test_termination_on_eos_in_bonus(self):
        """Generation halts immediately when target emits EOS as bonus token."""
        eos = 999
        prompt = [1, 2, 3, 10, 20, 30, 99, 1, 2]
        actions = [{"type": "accept_all", "bonus_token": eos}]

        model = ProgrammableAcceptanceTargetModel(round_actions=actions, prefill_token=3)
        cache = StaticKVCache(max_seq_len=64, device="cpu")
        tokenizer = MockTokenizer(eos_token_id=eos)
        engine = UnifiedSpeculativeEngine(target_model=model, static_cache=cache, tokenizer=tokenizer, device="cpu")

        out, _ = engine.generate(prompt, max_new_tokens=10, k_draft=3)
        tokens = out[0].tolist()

        assert tokens[-1] == eos
        assert tokens[len(prompt):] == [3, 10, 20, 30, eos]

    def test_max_new_tokens_ceiling_enforced(self):
        """Engine stops when exactly max_new_tokens are emitted, never overshooting budget."""
        prompt = [1, 2, 3, 10, 20, 30, 40, 99, 1, 2]
        # Request max_new_tokens = 7 with K = 4
        # Prefill emits 1 (token 3). Round 1 could emit up to 5 (4 + bonus). Round 2 finishes remaining.
        actions = [
            {"type": "accept_all", "bonus_token": 777},
            {"type": "accept_all", "bonus_token": 888},
        ]
        model = ProgrammableAcceptanceTargetModel(round_actions=actions, prefill_token=3)
        cache = StaticKVCache(max_seq_len=64, device="cpu")
        engine = UnifiedSpeculativeEngine(target_model=model, static_cache=cache, device="cpu")

        out, stats = engine.generate(prompt, max_new_tokens=7, k_draft=4)
        emitted_tokens = out[0, len(prompt):].tolist()

        assert len(emitted_tokens) == 7
        assert stats["num_tokens"] == 7

    def test_consecutive_generations_state_isolation(self):
        """Consecutive generations on same engine reset cache cleanly without cross-prompt contamination."""
        gt = [1, 2, 3, 4, 5] * 10
        prompt_1 = [1, 2, 3, 4, 1, 2]
        prompt_2 = [5, 4, 3, 2, 1, 5, 4]

        model = DeterministicGroundTruthModel(ground_truth_tokens=gt)
        cache = StaticKVCache(max_seq_len=64, device="cpu")
        engine = UnifiedSpeculativeEngine(target_model=model, static_cache=cache, device="cpu")

        out_1, _ = engine.generate(prompt_1, max_new_tokens=5, k_draft=2)
        out_2, _ = engine.generate(prompt_2, max_new_tokens=5, k_draft=2)

        # Compare prompt_2 output against a freshly instantiated engine
        fresh_model = DeterministicGroundTruthModel(ground_truth_tokens=gt)
        fresh_cache = StaticKVCache(max_seq_len=64, device="cpu")
        fresh_engine = UnifiedSpeculativeEngine(target_model=fresh_model, static_cache=fresh_cache, device="cpu")
        fresh_out_2, _ = fresh_engine.generate(prompt_2, max_new_tokens=5, k_draft=2)

        assert out_2.tolist() == fresh_out_2.tolist()

    def test_max_seq_len_saturation_protection(self):
        """Writing past max_seq_len raises ValueError buffer overflow protection."""
        gt = list(range(100))
        # max_seq_len is 24, prompt is 20, request 10 new tokens -> total 30 exceeds 24
        model = DeterministicGroundTruthModel(ground_truth_tokens=gt)
        cache = StaticKVCache(max_seq_len=24, device="cpu")
        engine = UnifiedSpeculativeEngine(target_model=model, static_cache=cache, device="cpu")

        prompt = list(range(20))
        with pytest.raises(ValueError):
            engine.generate(prompt, max_new_tokens=10, k_draft=3)
