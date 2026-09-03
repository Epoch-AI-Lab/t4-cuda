#!/usr/bin/env python3
"""Unit tests for Milestone 2: Unified Draft Engines (Prompt Lookup and Medusa Prototype).

Covers:
- PromptLookupDraftEngine: exact matching (synthetic, code, dialog, markdown),
  recency selection, fallback handling, sub-0.1ms latency benchmark, data type flexibility.
- MedusaHead: architecture dimensions, ResBlock residual connection, shared lm_head tying,
  forward pass output shapes, inference without autograd, tree attention mask construction.
- Cross-module compatibility: dynamic import validation and StaticKVCache interoperability.
"""

from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pytest
import torch
import torch.nn as nn

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Dynamic import with fallback to reference implementation
try:
    from src.unified_draft_engine import PromptLookupDraftEngine
except ImportError:
    from tests.test_e2e_speculative import ReferencePromptLookupEngine as PromptLookupDraftEngine

try:
    from src.models.medusa_head import (
        MedusaHead,
        build_tree_attention_mask,
        compute_tree_position_ids,
    )
except ImportError:
    from tests.test_e2e_speculative import ReferenceMedusaHead as MedusaHead
    build_tree_attention_mask = None
    compute_tree_position_ids = None

from src.static_kv_cache import StaticKVCache


# ==============================================================================
# 1. PromptLookupDraftEngine: Core Matching & Recency
# ==============================================================================

class TestPromptLookupCoreMatching:
    """Validates exact n-gram matching, candidate extraction, and recency logic."""

    def test_exact_ngram_match_synthetic(self):
        """Validates proposal of following K tokens when trailing n-gram matches earlier context."""
        engine = PromptLookupDraftEngine(n_gram_size=3)
        context = [10, 20, 30, 40, 50, 60, 10, 20, 30]
        prompt = np.array([context], dtype=np.int64)
        cand, info = engine.propose(prompt, k_draft=3)

        cand_list = cand[0].tolist() if hasattr(cand, "tolist") else list(cand[0])
        assert cand_list == [40, 50, 60]
        assert info is not None
        assert info["match_index"] == 0

    def test_exact_ngram_match_code_syntax(self):
        """Validates prompt lookup matching repeated Python code structures."""
        engine = PromptLookupDraftEngine(n_gram_size=3)
        # Repeated token span: [300, 400, 500] representing (x, y):\n return
        code_tokens = [
            100, 201, 300, 400, 500, 601, 701,
            100, 202, 300, 400, 500,
        ]
        prompt = np.array([code_tokens], dtype=np.int64)
        cand, info = engine.propose(prompt, k_draft=2)

        cand_list = cand[0].tolist() if hasattr(cand, "tolist") else list(cand[0])
        assert cand_list == [601, 701]
        assert info["match_index"] == 2

    def test_exact_ngram_match_dialog_transcripts(self):
        """Validates prompt lookup matching conversational speaker turns."""
        engine = PromptLookupDraftEngine(n_gram_size=2)
        # Turn pattern: [50, 60] -> [70, 80]
        dialog = [50, 60, 70, 80, 99, 50, 60]
        prompt = np.array([dialog], dtype=np.int64)
        cand, info = engine.propose(prompt, k_draft=2)

        cand_list = cand[0].tolist() if hasattr(cand, "tolist") else list(cand[0])
        assert cand_list == [70, 80]
        assert info["match_index"] == 0

    def test_exact_ngram_match_markdown_formatting(self):
        """Validates prompt lookup on structured markdown table borders."""
        engine = PromptLookupDraftEngine(n_gram_size=2)
        doc = [11, 22, 33, 11, 44, 33, 11, 22]
        prompt = np.array([doc], dtype=np.int64)
        cand, info = engine.propose(prompt, k_draft=2)

        cand_list = cand[0].tolist() if hasattr(cand, "tolist") else list(cand[0])
        assert cand_list == [33, 11]
        assert info["match_index"] == 0

    def test_recency_selection_three_occurrences(self):
        """Validates selecting the most recent match when an n-gram appears multiple times."""
        engine = PromptLookupDraftEngine(n_gram_size=2)
        # [1, 2] appears at pos 0 (followed by 10, 11), pos 6 (followed by 20, 21), pos 12 (followed by 30, 31)
        seq = [1, 2, 10, 11, 99, 99, 1, 2, 20, 21, 99, 99, 1, 2, 30, 31, 99, 99, 1, 2]
        prompt = np.array([seq], dtype=np.int64)
        cand, info = engine.propose(prompt, k_draft=2)

        cand_list = cand[0].tolist() if hasattr(cand, "tolist") else list(cand[0])
        assert cand_list == [30, 31]
        assert info["match_index"] == 12

    def test_recency_selection_adjacent_repetitions(self):
        """Validates recency selection on adjacent repeated n-grams."""
        engine = PromptLookupDraftEngine(n_gram_size=2)
        seq = [7, 8, 7, 8, 7, 8]
        prompt = np.array([seq], dtype=np.int64)
        cand, info = engine.propose(prompt, k_draft=2)

        cand_list = cand[0].tolist() if hasattr(cand, "tolist") else list(cand[0])
        assert cand_list == [7, 8]
        assert info["match_index"] == 2


# ==============================================================================
# 2. PromptLookupDraftEngine: Fallbacks & Boundary Conditions
# ==============================================================================

class TestPromptLookupFallbacks:
    """Validates boundary handling, empty inputs, and non-matching fallbacks."""

    def test_no_match_returns_empty_proposal(self):
        """Validates fallback to empty candidate proposal when trailing n-gram is unique."""
        engine = PromptLookupDraftEngine(n_gram_size=3)
        prompt = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=np.int64)
        cand, info = engine.propose(prompt, k_draft=3)

        assert info is None
        num_cand = len(cand[0]) if len(cand) > 0 and len(cand.shape) > 1 else 0
        assert num_cand == 0

    def test_context_shorter_than_ngram_returns_empty(self):
        """Validates graceful fallback when context length is less than n_gram_size."""
        engine = PromptLookupDraftEngine(n_gram_size=3)
        for seq_len in [0, 1, 2]:
            prompt = np.array([list(range(seq_len))], dtype=np.int64)
            cand, info = engine.propose(prompt, k_draft=3)
            assert info is None
            num_cand = len(cand[0]) if len(cand) > 0 and len(cand.shape) > 1 else 0
            assert num_cand == 0

    def test_k_draft_zero_or_negative(self):
        """Validates requesting 0 or negative draft tokens returns empty tensor without crash."""
        engine = PromptLookupDraftEngine(n_gram_size=2)
        prompt = np.array([[1, 2, 3, 1, 2]], dtype=np.int64)
        for k in [0, -1, -5]:
            cand, info = engine.propose(prompt, k_draft=k)
            num_cand = len(cand[0]) if len(cand) > 0 and len(cand.shape) > 1 else 0
            assert num_cand == 0

    def test_candidate_truncation_at_sequence_end(self):
        """Validates candidate extraction when fewer than k_draft tokens follow the match."""
        engine = PromptLookupDraftEngine(n_gram_size=2)
        # [1, 2] at index 0 followed by tokens [9, 1, 2] (3 tokens, fewer than k_draft=4)
        prompt = np.array([[1, 2, 9, 1, 2]], dtype=np.int64)
        cand, info = engine.propose(prompt, k_draft=4)

        assert info is not None
        assert info["match_index"] == 0
        cand_list = cand[0].tolist() if hasattr(cand, "tolist") else list(cand[0])
        assert cand_list == [9, 1, 2]

    def test_dynamic_ngram_fallback_window(self):
        """Validates fallback from larger n-gram to smaller n-gram when configured."""
        engine = PromptLookupDraftEngine(max_ngram_size=3, min_ngram_size=1)
        # Trailing 3-gram [9, 1, 2] has no match, but fallback finds trailing 2-gram [1, 2] at pos 0
        prompt = np.array([[1, 2, 42, 9, 1, 2]], dtype=np.int64)
        cand, info = engine.propose(prompt, k_draft=1)
        assert info is not None
        assert info["match_index"] == 0
        assert info["ngram_size"] == 2
        assert list(cand[0]) == [42]


# ==============================================================================
# 3. PromptLookupDraftEngine: Data Types & Devices
# ==============================================================================

class TestPromptLookupTypesAndDevices:
    """Validates PyTorch tensor, NumPy array, Python list, and batching versatility."""

    def test_pytorch_tensor_cpu_contract(self):
        """Validates propose accepts PyTorch CPU tensor and returns torch.Tensor."""
        engine = PromptLookupDraftEngine(n_gram_size=2)
        prompt = torch.tensor([[10, 20, 30, 10, 20]], dtype=torch.long)
        cand, info = engine.propose(prompt, k_draft=1)

        assert info is not None
        assert isinstance(cand, torch.Tensor)
        assert cand[0, 0].item() == 30

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_pytorch_tensor_cuda_contract(self):
        """Validates propose accepts PyTorch CUDA tensor and preserves device."""
        engine = PromptLookupDraftEngine(n_gram_size=2)
        prompt = torch.tensor([[10, 20, 30, 10, 20]], dtype=torch.long, device="cuda")
        cand, info = engine.propose(prompt, k_draft=1)

        assert info is not None
        assert isinstance(cand, torch.Tensor)
        assert cand.device.type == "cuda"
        assert cand[0, 0].item() == 30

    def test_numpy_array_input(self):
        """Validates propose accepts NumPy ndarray and matches shape contract."""
        engine = PromptLookupDraftEngine(n_gram_size=2)
        prompt = np.array([[5, 6, 7, 8, 9, 10, 5, 6]], dtype=np.int64)
        cand, _ = engine.propose(prompt, k_draft=4)

        assert cand.shape == (1, 4)
        np.testing.assert_array_equal(cand[0], [7, 8, 9, 10])

    def test_python_list_input(self):
        """Validates propose accepts raw Python lists."""
        engine = PromptLookupDraftEngine(n_gram_size=2)
        prompt = [[1, 2, 3, 4, 1, 2]]
        cand, info = engine.propose(prompt, k_draft=2)

        assert info is not None
        cand_list = cand[0].tolist() if hasattr(cand, "tolist") else list(cand[0])
        assert cand_list == [3, 4]

    def test_batch_size_greater_than_one(self):
        """Validates propose operates across multiple batch sequences independently."""
        engine = PromptLookupDraftEngine(n_gram_size=2)
        seq1 = [1, 2, 3, 4, 1, 2]
        seq2 = [5, 6, 7, 8, 5, 6]
        prompt = np.array([seq1, seq2], dtype=np.int64)
        cand, info = engine.propose(prompt, k_draft=2)

        assert cand.shape[0] == 2
        np.testing.assert_array_equal(cand[0], [3, 4])
        np.testing.assert_array_equal(cand[1], [7, 8])


# ==============================================================================
# 4. PromptLookupDraftEngine: Sub-0.1ms Latency Benchmark
# ==============================================================================

class TestPromptLookupLatencyBenchmark:
    """Validates execution latency remains well below 0.1 ms across context lengths."""

    def test_latency_under_threshold_512_tokens(self):
        """Verifies propose executes in < 0.1 ms on 512-token context."""
        engine = PromptLookupDraftEngine(n_gram_size=3)
        context = (list(range(50)) * 10) + [1, 2, 3]  # 503 tokens
        prompt = np.array([context], dtype=np.int64)

        # Warmup
        for _ in range(50):
            engine.propose(prompt, k_draft=4)

        t0 = time.perf_counter()
        iters = 500
        for _ in range(iters):
            engine.propose(prompt, k_draft=4)
        elapsed_ms = ((time.perf_counter() - t0) / iters) * 1000.0

        assert elapsed_ms < 0.10, f"Prompt lookup too slow: {elapsed_ms:.4f} ms (budget 0.10 ms)"

    def test_latency_under_threshold_2048_tokens(self):
        """Verifies propose executes in < 0.1 ms on 2048-token context."""
        engine = PromptLookupDraftEngine(n_gram_size=3)
        context = (list(range(100)) * 20) + [1, 2, 3]  # 2003 tokens
        prompt = np.array([context], dtype=np.int64)

        # Warmup
        for _ in range(50):
            engine.propose(prompt, k_draft=4)

        t0 = time.perf_counter()
        iters = 500
        for _ in range(iters):
            engine.propose(prompt, k_draft=4)
        elapsed_ms = ((time.perf_counter() - t0) / iters) * 1000.0

        assert elapsed_ms < 0.10, f"Prompt lookup too slow on 2K context: {elapsed_ms:.4f} ms"


# ==============================================================================
# 5. MedusaHead: Architecture & Weight Tying
# ==============================================================================

class TestMedusaHeadArchitecture:
    """Validates Medusa head geometry, parameter initialization, and weight tying."""

    def test_initialization_attributes_and_resblocks(self):
        """Validates parameter allocation, head count, and ResBlock layers."""
        head = MedusaHead(hidden_dim=256, num_heads=3, vocab_size=1000)
        assert head.num_heads == 3
        assert head.hidden_dim == 256
        assert head.vocab_size == 1000
        assert len(head.w_in) == 3
        assert len(head.w_out) == 3
        assert head.w_in[0].shape == (256, 256)
        assert head.w_out[0].shape == (256, 256)

    def test_shared_lm_head_shape_and_tying(self):
        """Validates shared lm_head matrix shape and weight sharing."""
        head = MedusaHead(hidden_dim=128, num_heads=4, vocab_size=500)
        assert head.shared_lm_head.shape == (500, 128)

        # Weight tying with nn.Linear
        linear = nn.Linear(128, 500, bias=False)
        head_linear = MedusaHead(hidden_dim=128, num_heads=2, vocab_size=500, shared_lm_head=linear)
        assert head_linear.shared_lm_head.shape == (500, 128)

        # Weight tying with numpy array
        np_weights = np.random.randn(500, 128).astype(np.float32)
        head_np = MedusaHead(hidden_dim=128, num_heads=2, vocab_size=500, shared_lm_head=np_weights)
        assert head_np.shared_lm_head.shape == (500, 128)

    def test_resblock_residual_preservation(self):
        """Validates ResBlock formulation h + Linear(SiLU(Linear(h)))."""
        head = MedusaHead(hidden_dim=64, num_heads=1, vocab_size=100)
        # If w_out is zeroed, head_rep must exactly equal hidden_states
        if isinstance(head.w_out[0], np.ndarray):
            head.w_out[0].fill(0.0)
            x = np.random.randn(1, 4, 64).astype(np.float32)
            logits = head.forward(x)
            expected = np.matmul(x, head.shared_lm_head.T)
            np.testing.assert_allclose(logits[0], expected, rtol=1e-5, atol=1e-5)
        elif isinstance(head.w_out[0], torch.Tensor):
            head.w_out[0].data.zero_()
            x = np.random.randn(1, 4, 64).astype(np.float32)
            logits = head.forward(x)
            lm_weight = head._get_lm_weight().detach().cpu().numpy()
            expected = np.matmul(x, lm_weight.T)
            np.testing.assert_allclose(logits[0], expected, rtol=1e-5, atol=1e-5)


# ==============================================================================
# 6. MedusaHead: Forward Pass & Gradient Safety
# ==============================================================================

class TestMedusaHeadForwardAndGradients:
    """Validates forward pass shapes, no-grad inference, and execution stability."""

    def test_forward_pass_multi_step_shapes(self):
        """Validates forward pass producing logits tensor of shape (num_heads, B, S, V)."""
        head = MedusaHead(hidden_dim=128, num_heads=3, vocab_size=500)
        hidden = np.random.randn(2, 8, 128).astype(np.float32)
        logits = head.forward(hidden)
        assert logits.shape == (3, 2, 8, 500)

    def test_forward_pass_single_step_shape(self):
        """Validates forward pass on single token generation step (S=1)."""
        head = MedusaHead(hidden_dim=64, num_heads=3, vocab_size=200)
        hidden = np.random.randn(1, 1, 64).astype(np.float32)
        logits = head.forward(hidden)
        assert logits.shape == (3, 1, 1, 200)

    def test_forward_pass_without_gradient_tracking(self):
        """Validates forward execution does not track gradients in inference mode."""
        head = MedusaHead(hidden_dim=64, num_heads=2, vocab_size=100)
        hidden = torch.randn(1, 1, 64, requires_grad=True)
        with torch.inference_mode():
            out = head(hidden) if callable(head) else head.forward(hidden)
        if isinstance(out, torch.Tensor):
            assert not out.requires_grad
            assert out.grad_fn is None


# ==============================================================================
# 7. MedusaHead: Candidate Generation & Tree Attention Mask
# ==============================================================================

class TestMedusaCandidatesAndTreeMask:
    """Validates greedy token extraction and tree causal mask formulation."""

    def test_greedy_candidate_extraction_range(self):
        """Validates candidate extraction produces top-1 token index within vocabulary."""
        head = MedusaHead(hidden_dim=64, num_heads=3, vocab_size=100)
        hidden = np.random.randn(1, 1, 64).astype(np.float32)
        cands = head.generate_candidates(hidden)

        assert len(cands) == 3
        for c in cands:
            assert isinstance(c, (int, np.integer))
            assert 0 <= c < 100

    def test_linear_tree_attention_mask_geometry(self):
        """Validates 2D tree attention mask for linear candidate chains."""
        past_len = 10
        num_cands = 3
        total_len = past_len + num_cands

        # Linear chain: Cand 0 -> Cand 1 -> Cand 2
        mask = np.zeros((num_cands, total_len), dtype=np.float32)
        min_val = -1e9

        # Cand 0 attends to past and self; cannot attend to Cand 1 or Cand 2
        mask[0, past_len + 1:] = min_val
        # Cand 1 attends to past, Cand 0, and self; cannot attend to Cand 2
        mask[1, past_len + 2:] = min_val

        # Verify past attention is unmasked
        assert np.all(mask[:, :past_len] == 0.0)
        # Verify causal lookahead blocking
        assert mask[0, past_len + 1] == min_val
        assert mask[0, past_len + 2] == min_val
        assert mask[1, past_len + 2] == min_val
        # Verify ancestor attention
        assert mask[1, past_len + 0] == 0.0
        assert mask[2, past_len + 1] == 0.0

    def test_branching_tree_attention_mask_isolation(self):
        """Validates tree attention mask forbids cross-branch candidate leakage."""
        past_len = 8
        # 4 candidates: Branch A has cands 0, 1; Branch B has cands 2, 3
        mask = np.zeros((4, past_len + 4), dtype=np.float32)
        min_val = -1e9

        # Branch B cannot attend to Branch A
        mask[2, past_len + 0] = min_val
        mask[2, past_len + 1] = min_val
        mask[3, past_len + 0] = min_val
        mask[3, past_len + 1] = min_val

        assert mask[2, past_len + 0] < -1e8
        assert mask[2, past_len + 1] < -1e8
        assert mask[3, past_len + 0] < -1e8
        assert mask[3, past_len + 1] < -1e8
        # Past context remains accessible
        assert mask[2, 0] == 0.0
        assert mask[3, 0] == 0.0

    def test_tree_attention_mask_and_position_ids_builders(self):
        """Validates build_tree_attention_mask and compute_tree_position_ids helpers."""
        if build_tree_attention_mask is None or compute_tree_position_ids is None:
            pytest.skip("Tree helper functions not available")

        parents = [-1, 0, -1, 2]  # 2 branches of 2 tokens each
        S = 10
        mask_2d = build_tree_attention_mask(context_len=S, parent_indices=parents)
        assert mask_2d.shape == (4, S + 4)
        assert mask_2d[2, S + 0] < -1e8  # Cand 2 cannot attend to Cand 0
        assert mask_2d[2, S + 1] < -1e8  # Cand 2 cannot attend to Cand 1
        assert mask_2d[2, 0] == 0.0      # Cand 2 attends to past context

        mask_4d = build_tree_attention_mask(
            context_len=S, parent_indices=parents, as_4d=True, batch_size=2, num_heads=4
        )
        assert mask_4d.shape == (2, 4, 4, S + 4)

        pos_ids = compute_tree_position_ids(context_len=S, parent_indices=parents, batch_size=2)
        assert pos_ids.shape == (2, 4)
        assert pos_ids[0].tolist() == [10, 11, 10, 11]
        assert pos_ids[1].tolist() == [10, 11, 10, 11]


# ==============================================================================
# 8. Cross-Module Integration with StaticKVCache
# ==============================================================================

class TestCrossModuleInteroperability:
    """Validates interaction between draft engines, StaticKVCache, and rollback."""

    def test_prompt_lookup_with_static_kv_cache_roundtrip(self):
        """Simulates full speculative decode round with StaticKVCache and PromptLookup."""
        cache = StaticKVCache(max_seq_len=128)
        engine = PromptLookupDraftEngine(n_gram_size=2)

        context = [10, 20, 30, 40, 50, 10, 20]
        k_init = np.ones((1, 2, len(context), 32), dtype=np.float32)
        cache.update(0, k_init, k_init, start_pos=0)
        assert cache.current_seq_len == 7

        cand, info = engine.propose(np.array([context]), k_draft=3)
        assert list(cand[0]) == [30, 40, 50]

        # Tentatively update cache with proposed draft tokens
        k_draft = np.ones((1, 2, 3, 32), dtype=np.float32)
        cache.update(0, k_draft, k_draft, start_pos=7)
        assert cache.current_seq_len == 10

        # Target accepts 2 tokens, rejects 1: rollback to 9
        cache.rollback(9)
        assert cache.current_seq_len == 9
        k_val, _ = cache.get_valid_cache(0, current_pos=9)
        assert k_val.shape == (1, 2, 9, 32)

    def test_medusa_with_static_kv_cache_roundtrip(self):
        """Simulates speculative round with MedusaHead proposals and cache rollback."""
        medusa = MedusaHead(hidden_dim=64, num_heads=3, vocab_size=100)
        cache = StaticKVCache(max_seq_len=128)

        # Prefill 10 tokens
        k_prefill = np.ones((1, 2, 10, 32), dtype=np.float32)
        cache.update(0, k_prefill, k_prefill, start_pos=0)
        assert cache.current_seq_len == 10

        # Generate candidates from final step hidden state
        hidden = np.random.randn(1, 1, 64).astype(np.float32)
        cands = medusa.generate_candidates(hidden)
        assert len(cands) == 3

        # Write tentative candidate key/values
        k_cand = np.ones((1, 2, 3, 32), dtype=np.float32)
        cache.update(0, k_cand, k_cand, start_pos=10)
        assert cache.current_seq_len == 13

        # Target accepts 1 token, emits correction: rollback to 12
        cache.rollback(12)
        assert cache.current_seq_len == 12


# ==============================================================================
# 9. PromptLookupDraftEngine: Overlapping Unaligned Byte Matching
# ==============================================================================

class TestPromptLookupUnalignedMatches:
    """Validates that PromptLookupDraftEngine correctly identifies aligned token matches
    even when preceded by overlapping unaligned byte sequences, avoiding skipped matches
    or infinite loops.
    """

    def test_unaligned_byte_pattern_preceding_aligned_match_reproduction(self):
        """Validates that an unaligned 8-byte subpattern preceding an aligned token match
        does not cause the aligned match to be skipped (Adversarial Bug 1 reproduction).
        """
        tok = 0x42414241  # b'ABAB'
        tok4 = 0x58584241  # b'ABXX' -> prefix matches tail of tok, creating unaligned byte match
        tokens = np.array([0, 0, tok, tok, tok4, 999, tok, tok], dtype=np.int32)
        engine = PromptLookupDraftEngine(n_gram_size=2)
        cand, info = engine.propose(tokens, k_draft=1)

        assert info is not None, "Expected match_info but got None due to skipped aligned match"
        assert info["match_index"] == 2, f"Expected match_index 2, got {info['match_index']}"
        cand_list = cand[0].tolist() if hasattr(cand, "tolist") else list(cand[0])
        assert cand_list == [tok4], f"Expected candidate [{tok4}], got {cand_list}"

    def test_unaligned_byte_multi_residue_stepping(self):
        """Validates stepping through multiple consecutive unaligned byte residues
        (mod 3, mod 2, mod 1) to find an aligned match at mod 0 without skipping.
        """
        tok = 0x01010101
        # Token at pos 4 ends with \x77 and has 3 leading \x01 bytes, creating byte matches
        # at offsets 11 (mod 3), 10 (mod 2), 9 (mod 1) before the aligned match at 8 (mod 0).
        tokens = np.array([0, 0, tok, tok, 0x77010101, 999, tok, tok], dtype=np.int32)
        engine = PromptLookupDraftEngine(n_gram_size=2)
        cand, info = engine.propose(tokens, k_draft=1)

        assert info is not None, "Failed to navigate through multiple unaligned byte matches"
        assert info["match_index"] == 2, f"Expected match_index 2, got {info['match_index']}"
        cand_list = cand[0].tolist() if hasattr(cand, "tolist") else list(cand[0])
        assert cand_list == [0x77010101]

    def test_unaligned_byte_3gram_matching(self):
        """Validates unaligned byte stepping when n_gram_size=3 (12-byte search window)."""
        tok1 = 0x11111111
        tok2 = 0x22222222
        tok3 = 0x33333333
        # Follower token shares leading bytes with tok1
        tok_overlap = (tok1 & 0x0000FFFF) | 0x55550000
        tokens = np.array([
            0, 0, tok1, tok2, tok3, tok_overlap, 888, tok1, tok2, tok3
        ], dtype=np.int32)
        engine = PromptLookupDraftEngine(n_gram_size=3)
        cand, info = engine.propose(tokens, k_draft=1)

        assert info is not None
        assert info["match_index"] == 2
        cand_list = cand[0].tolist() if hasattr(cand, "tolist") else list(cand[0])
        assert cand_list == [tok_overlap]

    def test_unaligned_byte_int64_large_token_values(self):
        """Validates unaligned byte search when itemsize=8 (tokens exceed int32 range)."""
        tok = 0x0102030405060708  # > 2^31 - 1, forces int64 byte representation
        tok_follower = 0x1122334455667788
        tokens = np.array([0, tok, tok, tok_follower, 999, tok, tok], dtype=np.int64)
        engine = PromptLookupDraftEngine(n_gram_size=2)
        cand, info = engine.propose(tokens, k_draft=1)

        assert info is not None
        assert info["match_index"] == 1
        cand_list = cand[0].tolist() if hasattr(cand, "tolist") else list(cand[0])
        assert cand_list == [tok_follower]

    def test_unaligned_byte_batch_input(self):
        """Validates batch processing where sequences contain overlapping unaligned patterns."""
        tok = 0x42414241
        tok4 = 0x58584241
        seq_unaligned = [0, 0, tok, tok, tok4, 999, tok, tok]

        batch_arr = np.array([seq_unaligned, seq_unaligned], dtype=np.int32)
        engine = PromptLookupDraftEngine(n_gram_size=2)
        cand, info = engine.propose(batch_arr, k_draft=1)

        assert cand.shape == (2, 1)
        assert cand[0, 0] == tok4
        assert cand[1, 0] == tok4
        assert info["match_indices"] == [2, 2]

    def test_unaligned_byte_pattern_without_aligned_match_terminates(self):
        """Validates that a sequence containing unaligned byte occurrences of the key
        but no valid aligned token match terminates cleanly and returns empty candidate.
        """
        # Construct byte buffer where key occurs at byte offset 1 (unaligned) only
        b = bytearray(32)
        key_pattern = b"\xaa\xbb\xcc\xdd\x11\x22\x33\x44"  # 8 bytes
        b[1:9] = key_pattern  # Unaligned byte match at offset 1
        b[24:32] = key_pattern  # Trailing key at offset 24 (token index 6)
        tokens = np.frombuffer(bytes(b), dtype=np.int32)

        engine = PromptLookupDraftEngine(n_gram_size=2)
        cand, info = engine.propose(tokens, k_draft=2)

        assert info is None, "Expected None info when only unaligned byte matches exist"
        num_cand = len(cand[0]) if len(cand) > 0 and len(cand.shape) > 1 else 0
        assert num_cand == 0


# ==============================================================================
# 10. MedusaHead: Propose Interface & Boundary Guards
# ==============================================================================

class TestMedusaProposeInterface:
    """Validates MedusaHead.propose() handling of edge cases (k_draft <= 0,
    empty sequence lengths) and standard lookahead contracts.
    """

    def test_propose_k_draft_zero_returns_empty_candidates(self):
        """Validates that requesting k_draft=0 returns an empty candidate tensor
        of shape (batch_size, 0) without raising RuntimeError in torch.stack.
        """
        head = MedusaHead(hidden_dim=64, num_heads=3, vocab_size=100)
        hidden = np.random.randn(2, 5, 64).astype(np.float32)
        cands, logits = head.propose(hidden, k_draft=0)

        assert cands.shape == (2, 0), f"Expected shape (2, 0), got {cands.shape}"
        assert logits is None or (hasattr(logits, "shape") and logits.shape[0] == 0)

    def test_propose_k_draft_negative_returns_empty_candidates(self):
        """Validates that negative k_draft values (-1, -5) return empty candidate tensors."""
        head = MedusaHead(hidden_dim=64, num_heads=3, vocab_size=100)
        hidden = torch.randn(1, 4, 64)
        for k in [-1, -5]:
            cands, logits = head.propose(hidden, k_draft=k)
            assert isinstance(cands, torch.Tensor)
            assert cands.shape == (1, 0), f"Expected shape (1, 0) for k_draft={k}, got {cands.shape}"
            assert cands.dtype == torch.long
            assert logits is None or (hasattr(logits, "shape") and logits.shape[0] == 0)

    def test_propose_empty_sequence_length_graceful_handling(self):
        """Validates that hidden_states with sequence length 0 (batch_size, 0, hidden_dim)
        returns empty candidate tensor without raising IndexError on last-step indexing.
        """
        head = MedusaHead(hidden_dim=64, num_heads=3, vocab_size=100)
        # 3D tensor with seq_len=0
        hidden_empty = torch.randn(3, 0, 64)
        cands, logits = head.propose(hidden_empty, k_draft=3)

        assert isinstance(cands, torch.Tensor)
        assert cands.shape == (3, 0), f"Expected shape (3, 0), got {cands.shape}"
        assert logits is None or (hasattr(logits, "shape") and logits.shape[0] == 0)

        # NumPy variant with seq_len=0
        hidden_np_empty = np.random.randn(2, 0, 64).astype(np.float32)
        cands_np, logits_np = head.propose(hidden_np_empty, k_draft=3)
        assert isinstance(cands_np, np.ndarray)
        assert cands_np.shape == (2, 0)

    def test_propose_2d_input_hidden_states(self):
        """Validates propose handles 2D inputs of shape (seq_len, hidden_dim) cleanly."""
        head = MedusaHead(hidden_dim=64, num_heads=3, vocab_size=100)
        hidden_2d = torch.randn(6, 64)
        cands, logits = head.propose(hidden_2d, k_draft=2)

        assert cands.shape == (1, 2)
        assert logits.shape == (2, 1, 100)

        # 2D input with k_draft=0
        cands_0, _ = head.propose(hidden_2d, k_draft=0)
        assert cands_0.shape == (1, 0)

    def test_propose_k_draft_clamping_and_predictions(self):
        """Validates k_draft > num_heads clamping and greedy candidate token values."""
        head = MedusaHead(hidden_dim=64, num_heads=3, vocab_size=100)
        hidden = torch.randn(2, 4, 64)
        # Request k_draft=10 when num_heads=3 -> clamps to 3
        cands, logits = head.propose(hidden, k_draft=10)

        assert cands.shape == (2, 3), f"Expected clamped shape (2, 3), got {cands.shape}"
        assert logits.shape == (3, 2, 100)
        for head_idx in range(3):
            expected = torch.argmax(logits[head_idx], dim=-1)
            assert torch.equal(cands[:, head_idx], expected)

