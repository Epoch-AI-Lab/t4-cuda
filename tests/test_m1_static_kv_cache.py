#!/usr/bin/env python3
"""Unit tests for Milestone 1: StaticKVCache correctness and DynamicCache parity.

Verifies:
- Pre-allocated tensor shapes and data pointer invariance (zero reallocations).
- Zero CUDA memory allocations during decode when CUDA is available.
- Strict numerical parity with HuggingFace DynamicCache across prefill, decode,
  multi-token draft chunks, and rollback cycles (<1e-4 FP16, <1e-6 FP32).
- Masked attention computation parity confirming unwritten buffer invariance.
- RoPE relative distance invariance under sequence translation.
- Explicit position_ids arithmetic and rollback sequence position lifecycle.
- 4D causal attention mask equivalence with sequential attention.
- Boundary edge cases (overflow, out-of-bounds rollback, reset, config factory).
"""

import math
from pathlib import Path
import random
import sys
import pytest
import torch
from transformers.cache_utils import DynamicCache

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.static_kv_cache import (
    StaticKVCache,
    build_causal_4d_mask,
    compute_position_ids,
)



def get_dynamic_cache_tensors(dyn_cache: DynamicCache, layer_idx: int):
    """Helper to extract (key, value) tensors across HuggingFace Transformers versions."""
    if hasattr(dyn_cache, "key_cache") and dyn_cache.key_cache:
        return dyn_cache.key_cache[layer_idx], dyn_cache.value_cache[layer_idx]
    if hasattr(dyn_cache, "layers") and len(dyn_cache.layers) > layer_idx:
        return dyn_cache.layers[layer_idx].keys, dyn_cache.layers[layer_idx].values
    raise AttributeError(f"Cannot extract key/value tensors from {type(dyn_cache)}")


def apply_rope(x: torch.Tensor, positions: torch.Tensor, theta: float = 1000000.0) -> torch.Tensor:
    """Standard RoPE rotation for mathematical testing."""
    # x: (batch, heads, seq_len, head_dim)
    # positions: (batch, seq_len)
    batch_size, num_heads, seq_len, head_dim = x.shape
    half_dim = head_dim // 2
    freq_seq = torch.arange(0, half_dim, dtype=torch.float64, device=x.device)
    inv_freq = 1.0 / (theta ** (2.0 * freq_seq / head_dim))

    # angles: (batch, seq_len, half_dim)
    angles = positions.to(torch.float64).unsqueeze(-1) * inv_freq.unsqueeze(0).unsqueeze(0)
    emb = torch.cat([angles, angles], dim=-1)  # (batch, seq_len, head_dim)
    cos = torch.cos(emb).unsqueeze(1).to(x.dtype)
    sin = torch.sin(emb).unsqueeze(1).to(x.dtype)

    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]
    rotated_half = torch.cat([-x2, x1], dim=-1)
    return (x * cos) + (rotated_half * sin)


class TestStaticKVCacheAllocation:
    """Verifies memory allocation stability, shapes, and zero-allocation decode."""

    def test_buffer_allocation_shapes(self):
        """Buffer pre-allocates exactly (max_batch_size, num_heads, max_seq_len, head_dim)."""
        cache = StaticKVCache(
            max_batch_size=2,
            max_seq_len=512,
            num_layers=4,
            num_heads=2,
            head_dim=64,
            device="cpu",
            dtype=torch.float32,
        )
        assert len(cache.key_cache) == 4
        assert len(cache.value_cache) == 4
        assert cache.storage.shape == (4, 2, 2, 2, 512, 64)
        for layer_idx in range(4):
            assert cache.key_cache[layer_idx].shape == (2, 2, 512, 64)
            assert cache.value_cache[layer_idx].shape == (2, 2, 512, 64)

    def test_storage_data_ptr_invariance(self):
        """Data pointers remain constant across updates and rollbacks (zero reallocation)."""
        cache = StaticKVCache(
            max_batch_size=1,
            max_seq_len=256,
            num_layers=2,
            num_heads=2,
            head_dim=64,
            device="cpu",
            dtype=torch.float32,
        )
        initial_k_ptrs = [cache.key_cache[i].data_ptr() for i in range(2)]
        initial_v_ptrs = [cache.value_cache[i].data_ptr() for i in range(2)]
        storage_ptr = cache.storage.data_ptr()

        # Perform 50 decode updates and periodic rollbacks
        for pos in range(50):
            k = torch.randn(1, 2, 1, 64)
            v = torch.randn(1, 2, 1, 64)
            for l in range(2):
                cache.update(l, k, v, start_pos=pos)
            if pos % 5 == 0 and pos > 0:
                cache.rollback(pos - 2)

        final_k_ptrs = [cache.key_cache[i].data_ptr() for i in range(2)]
        final_v_ptrs = [cache.value_cache[i].data_ptr() for i in range(2)]

        assert cache.storage.data_ptr() == storage_ptr
        for i in range(2):
            assert initial_k_ptrs[i] == final_k_ptrs[i], f"Layer {i} key buffer relocated"
            assert initial_v_ptrs[i] == final_v_ptrs[i], f"Layer {i} value buffer relocated"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for zero-alloc check")
    def test_zero_cuda_allocations_during_decode(self):
        """Zero CUDA memory allocations or frees occur during decode update steps."""
        device = torch.device("cuda")
        num_layers = 24
        cache = StaticKVCache(
            max_batch_size=1,
            max_seq_len=512,
            num_layers=num_layers,
            num_heads=2,
            head_dim=128,
            device=device,
            dtype=torch.float16,
        )

        # Warmup step
        k_dummy = torch.randn(1, 2, 1, 128, device=device, dtype=torch.float16)
        v_dummy = torch.randn(1, 2, 1, 128, device=device, dtype=torch.float16)
        for l in range(num_layers):
            cache.update(l, k_dummy, v_dummy, start_pos=0)
        torch.cuda.synchronize(device)

        stats_before = torch.cuda.memory_stats(device)
        for pos in range(1, 50):
            for l in range(num_layers):
                cache.update(l, k_dummy, v_dummy, start_pos=pos)

        torch.cuda.synchronize(device)
        stats_after = torch.cuda.memory_stats(device)

        allocs = stats_after["allocation.all.allocated"] - stats_before["allocation.all.allocated"]
        frees = stats_after["allocation.all.freed"] - stats_before["allocation.all.freed"]
        assert allocs == 0, f"Expected 0 CUDA allocations, got {allocs}"
        assert frees == 0, f"Expected 0 CUDA frees, got {frees}"


class TestStaticKVCacheDynamicParity:
    """Verifies strict numerical equivalence with HuggingFace DynamicCache."""

    @pytest.mark.parametrize("dtype,tol", [
        (torch.float32, 1e-6),
        (torch.float16, 1e-4),
    ])
    def test_prefill_and_decode_parity(self, dtype, tol):
        """Prefill + decode sequence produces identical key-value caches to DynamicCache."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_layers, num_heads, head_dim = 4, 2, 64
        prompt_len, decode_steps = 32, 16

        static_cache = StaticKVCache(1, 256, num_layers, num_heads, head_dim, device, dtype)
        dynamic_cache = DynamicCache()

        # Prefill phase
        k_init = torch.randn(1, num_heads, prompt_len, head_dim, device=device, dtype=dtype)
        v_init = torch.randn(1, num_heads, prompt_len, head_dim, device=device, dtype=dtype)
        for l in range(num_layers):
            static_cache.update(l, k_init, v_init, start_pos=0)
            dynamic_cache.update(k_init, v_init, layer_idx=l)

        # Autoregressive decode phase
        for step in range(decode_steps):
            pos = prompt_len + step
            k_tok = torch.randn(1, num_heads, 1, head_dim, device=device, dtype=dtype)
            v_tok = torch.randn(1, num_heads, 1, head_dim, device=device, dtype=dtype)
            for l in range(num_layers):
                static_cache.update(l, k_tok, v_tok, start_pos=pos)
                dynamic_cache.update(k_tok, v_tok, layer_idx=l)

        # Verify parity across all layers
        assert static_cache.current_pos == prompt_len + decode_steps
        for l in range(num_layers):
            k_stat, v_stat = static_cache.get_valid_cache(l)
            k_dyn, v_dyn = get_dynamic_cache_tensors(dynamic_cache, l)

            assert k_stat.shape == k_dyn.shape
            assert v_stat.shape == v_dyn.shape

            max_err_k = torch.max(torch.abs(k_stat - k_dyn)).item()
            max_err_v = torch.max(torch.abs(v_stat - v_dyn)).item()
            assert max_err_k < tol, f"Layer {l} key cache error {max_err_k} exceeded {tol}"
            assert max_err_v < tol, f"Layer {l} value cache error {max_err_v} exceeded {tol}"

    def test_speculative_rollback_parity(self):
        """Simulate speculative proposal, partial reject, and rollback matching DynamicCache."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_layers, num_heads, head_dim = 2, 2, 64
        static_cache = StaticKVCache(1, 128, num_layers, num_heads, head_dim, device, torch.float32)
        dynamic_cache = DynamicCache()

        # 1. Prefill 10 tokens
        k0 = torch.randn(1, num_heads, 10, head_dim, device=device)
        v0 = torch.randn(1, num_heads, 10, head_dim, device=device)
        for l in range(num_layers):
            static_cache.update(l, k0, v0, start_pos=0)
            dynamic_cache.update(k0, v0, layer_idx=l)

        # 2. Propose K=4 draft tokens (indices 10..13)
        k_draft = torch.randn(1, num_heads, 4, head_dim, device=device)
        v_draft = torch.randn(1, num_heads, 4, head_dim, device=device)
        for l in range(num_layers):
            static_cache.update(l, k_draft, v_draft, start_pos=10)
            dynamic_cache.update(k_draft, v_draft, layer_idx=l)

        # 3. Target accepts 2, rejects 2. Rollback to 12
        static_cache.rollback(12)
        dynamic_cache.crop(12)

        # 4. Append target correction token at pos 12
        k_corr = torch.randn(1, num_heads, 1, head_dim, device=device)
        v_corr = torch.randn(1, num_heads, 1, head_dim, device=device)
        for l in range(num_layers):
            static_cache.update(l, k_corr, v_corr, start_pos=12)
            dynamic_cache.update(k_corr, v_corr, layer_idx=l)

        # 5. Parity check
        for l in range(num_layers):
            k_stat, v_stat = static_cache.get_valid_cache(l)
            k_dyn, v_dyn = get_dynamic_cache_tensors(dynamic_cache, l)
            assert torch.allclose(k_stat, k_dyn, atol=1e-6)
            assert torch.allclose(v_stat, v_dyn, atol=1e-6)

    def test_chunked_draft_update_parity(self):
        """Appending multi-token draft chunks (K in {2, 3, 4, 5}) matches DynamicCache."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_layers, num_heads, head_dim = 2, 2, 64
        static_cache = StaticKVCache(1, 256, num_layers, num_heads, head_dim, device, torch.float32)
        dynamic_cache = DynamicCache()

        current_pos = 0
        for chunk_len in [2, 3, 4, 5]:
            k_chunk = torch.randn(1, num_heads, chunk_len, head_dim, device=device)
            v_chunk = torch.randn(1, num_heads, chunk_len, head_dim, device=device)
            for l in range(num_layers):
                k_out_stat, v_out_stat = static_cache.update(l, k_chunk, v_chunk, start_pos=current_pos)
                k_out_dyn, v_out_dyn = dynamic_cache.update(k_chunk, v_chunk, layer_idx=l)

                assert torch.allclose(k_out_stat, k_out_dyn, atol=1e-6)
                assert torch.allclose(v_out_stat, v_out_dyn, atol=1e-6)
            current_pos += chunk_len
            assert static_cache.current_pos == current_pos

    def test_masked_attention_parity(self):
        """Attention output softmax(QK^T / sqrt(d))V matches DynamicCache."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch, num_heads, q_len, kv_len, head_dim = 1, 2, 1, 45, 64
        q = torch.randn(batch, num_heads, q_len, head_dim, device=device, dtype=torch.float32)

        static_cache = StaticKVCache(batch, 128, 1, num_heads, head_dim, device, torch.float32)
        dynamic_cache = DynamicCache()

        k_init = torch.randn(batch, num_heads, kv_len, head_dim, device=device)
        v_init = torch.randn(batch, num_heads, kv_len, head_dim, device=device)
        static_cache.update(0, k_init, v_init, start_pos=0)
        dynamic_cache.update(k_init, v_init, layer_idx=0)

        k_stat, v_stat = static_cache.get_valid_cache(0)
        k_dyn, v_dyn = get_dynamic_cache_tensors(dynamic_cache, 0)

        scale = 1.0 / math.sqrt(head_dim)
        scores_stat = torch.matmul(q, k_stat.transpose(-1, -2)) * scale
        scores_dyn = torch.matmul(q, k_dyn.transpose(-1, -2)) * scale

        attn_stat = torch.softmax(scores_stat, dim=-1)
        attn_dyn = torch.softmax(scores_dyn, dim=-1)

        out_stat = torch.matmul(attn_stat, v_stat)
        out_dyn = torch.matmul(attn_dyn, v_dyn)

        max_err = torch.max(torch.abs(out_stat - out_dyn)).item()
        assert max_err < 1e-6, f"Attention output diverged: max error {max_err}"


class TestRoPEAndPositionIntegrity:
    """Verifies RoPE invariance and explicit position_ids arithmetic."""

    def test_rope_relative_invariance(self):
        """Relative position dot product remains invariant under absolute translation."""
        device = torch.device("cpu")
        head_dim = 64
        q = torch.randn(1, 1, 1, head_dim, dtype=torch.float64, device=device)
        k = torch.randn(1, 1, 1, head_dim, dtype=torch.float64, device=device)

        m = 15  # query pos
        n = 10  # key pos

        q_rope = apply_rope(q, torch.tensor([[m]], device=device))
        k_rope = apply_rope(k, torch.tensor([[n]], device=device))
        baseline_dot = torch.sum(q_rope * k_rope).item()

        # Shift both positions by delta
        for delta in [1, 5, 20, 100]:
            q_shifted = apply_rope(q, torch.tensor([[m + delta]], device=device))
            k_shifted = apply_rope(k, torch.tensor([[n + delta]], device=device))
            shifted_dot = torch.sum(q_shifted * k_shifted).item()
            err = abs(baseline_dot - shifted_dot)
            assert err < 1e-10, f"RoPE translation invariance failed at delta {delta}: err={err}"

    def test_position_ids_arithmetic(self):
        """compute_position_ids produces correct 2D long tensors."""
        pos_ids = compute_position_ids(start_pos=15, num_tokens=4, batch_size=2)
        assert pos_ids.shape == (2, 4)
        assert pos_ids.dtype == torch.long
        expected = torch.tensor([[15, 16, 17, 18], [15, 16, 17, 18]], dtype=torch.long)
        assert torch.equal(pos_ids, expected)

    def test_cache_get_position_ids(self):
        """cache.get_position_ids respects current_pos and explicit start_pos."""
        cache = StaticKVCache(1, 128, 2, 2, 64, device="cpu")
        cache.rollback(25)

        # Uses current_pos when start_pos=None
        p1 = cache.get_position_ids(num_tokens=3)
        assert torch.equal(p1, torch.tensor([[25, 26, 27]], dtype=torch.long))

        # Explicit start_pos override
        p2 = cache.get_position_ids(num_tokens=2, start_pos=10)
        assert torch.equal(p2, torch.tensor([[10, 11]], dtype=torch.long))

    def test_rollback_position_lifecycle(self):
        """Simulate 100 random speculative cycles with zero drift in sequence pointer."""
        random.seed(42)
        cache = StaticKVCache(1, 4096, 2, 2, 64, device="cpu")

        # Initial prompt length
        prompt_len = 16
        k_dummy = torch.randn(1, 2, prompt_len, 64)
        v_dummy = torch.randn(1, 2, prompt_len, 64)
        for l in range(2):
            cache.update(l, k_dummy, v_dummy, start_pos=0)

        ground_truth_pos = prompt_len
        lookahead_k = 4

        for _ in range(100):
            # Propose lookahead_k tokens
            k_draft = torch.randn(1, 2, lookahead_k, 64)
            v_draft = torch.randn(1, 2, lookahead_k, 64)
            for l in range(2):
                cache.update(l, k_draft, v_draft, start_pos=ground_truth_pos)

            # Random acceptance between 0 and lookahead_k
            num_accepted = random.randint(0, lookahead_k)

            if num_accepted == lookahead_k:
                # All accepted + bonus token
                ground_truth_pos += lookahead_k
                k_bonus = torch.randn(1, 2, 1, 64)
                v_bonus = torch.randn(1, 2, 1, 64)
                for l in range(2):
                    cache.update(l, k_bonus, v_bonus, start_pos=ground_truth_pos)
                ground_truth_pos += 1
            else:
                # Rollback to ground_truth_pos + num_accepted
                new_pos = ground_truth_pos + num_accepted
                cache.rollback(new_pos)
                ground_truth_pos = new_pos

                # Commit 1 target correction token
                k_corr = torch.randn(1, 2, 1, 64)
                v_corr = torch.randn(1, 2, 1, 64)
                for l in range(2):
                    cache.update(l, k_corr, v_corr, start_pos=ground_truth_pos)
                ground_truth_pos += 1

            assert cache.current_pos == ground_truth_pos
            for l in range(2):
                assert cache.get_seq_length(l) == ground_truth_pos


class TestCausal4DMask:
    """Verifies causal mask geometry and attention isolation."""

    def test_causal_mask_structure(self):
        """Past context has 0.0, causal query triangle has 0.0, future is min_val."""
        batch_size = 2
        query_len = 3
        past_kv_len = 4
        mask = build_causal_4d_mask(
            batch_size=batch_size,
            query_len=query_len,
            past_kv_len=past_kv_len,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        assert mask.shape == (2, 1, 3, 7)
        m = mask[0, 0]  # (3, 7)

        # For all queries, past columns [0, 1, 2, 3] must be 0.0
        assert torch.all(m[:, :past_kv_len] == 0.0)

        # Query 0 attends to col 4 (self), future cols 5, 6 are min_val
        assert m[0, 4] == 0.0
        assert m[0, 5] < -1e8
        assert m[0, 6] < -1e8

        # Query 1 attends to cols 4, 5, col 6 is min_val
        assert m[1, 4] == 0.0
        assert m[1, 5] == 0.0
        assert m[1, 6] < -1e8

        # Query 2 attends to cols 4, 5, 6
        assert m[2, 4] == 0.0
        assert m[2, 5] == 0.0
        assert m[2, 6] == 0.0

    def test_causal_attention_sequential_equivalence(self):
        """Parallel multi-token attention with causal 4D mask equals sequential single-token steps."""
        device = torch.device("cpu")
        head_dim = 32
        num_heads = 1
        past_len = 6
        query_len = 3
        total_len = past_len + query_len

        # Shared past keys and values
        k_past = torch.randn(1, num_heads, past_len, head_dim, device=device)
        v_past = torch.randn(1, num_heads, past_len, head_dim, device=device)

        # Query, key, value for the 3 candidate tokens
        q_cand = torch.randn(1, num_heads, query_len, head_dim, device=device)
        k_cand = torch.randn(1, num_heads, query_len, head_dim, device=device)
        v_cand = torch.randn(1, num_heads, query_len, head_dim, device=device)

        k_all = torch.cat([k_past, k_cand], dim=-2)
        v_all = torch.cat([v_past, v_cand], dim=-2)

        # Method A: Parallel forward pass with 4D causal mask
        mask = build_causal_4d_mask(1, query_len, past_len, device=device, dtype=torch.float32)
        scale = 1.0 / math.sqrt(head_dim)
        scores_parallel = torch.matmul(q_cand, k_all.transpose(-1, -2)) * scale
        scores_parallel = scores_parallel + mask
        attn_parallel = torch.softmax(scores_parallel, dim=-1)
        out_parallel = torch.matmul(attn_parallel, v_all)  # (1, 1, 3, head_dim)

        # Method B: Sequential single-token forward passes
        seq_outs = []
        for i in range(query_len):
            q_i = q_cand[:, :, i : i + 1, :]
            # Keys and values visible at step i include past plus candidates up to i
            k_visible = k_all[:, :, : past_len + i + 1, :]
            v_visible = v_all[:, :, : past_len + i + 1, :]

            scores_i = torch.matmul(q_i, k_visible.transpose(-1, -2)) * scale
            attn_i = torch.softmax(scores_i, dim=-1)
            out_i = torch.matmul(attn_i, v_visible)
            seq_outs.append(out_i)

        out_sequential = torch.cat(seq_outs, dim=-2)  # (1, 1, 3, head_dim)

        max_diff = torch.max(torch.abs(out_parallel - out_sequential)).item()
        assert max_diff < 1e-6, f"Parallel causal mask and sequential attention diverged: {max_diff}"


class TestEdgeCasesAndInvariants:
    """Verifies error handling, duck typing, and HF Cache compatibility."""

    def test_out_of_bounds_rollback_raises_error(self):
        cache = StaticKVCache(1, 128, 2, 2, 64, device="cpu")
        with pytest.raises(ValueError):
            cache.rollback(-1)
        with pytest.raises(ValueError):
            cache.rollback(129)

    def test_seq_len_overflow_raises_error(self):
        cache = StaticKVCache(1, 32, 2, 2, 64, device="cpu")
        k_big = torch.randn(1, 2, 33, 64)
        v_big = torch.randn(1, 2, 33, 64)
        with pytest.raises(ValueError):
            cache.update(0, k_big, v_big, start_pos=0)

    def test_batch_size_overflow_raises_error(self):
        cache = StaticKVCache(1, 32, 2, 2, 64, device="cpu")
        k_batch = torch.randn(2, 2, 1, 64)
        v_batch = torch.randn(2, 2, 1, 64)
        with pytest.raises(ValueError):
            cache.update(0, k_batch, v_batch, start_pos=0)

    def test_reset_clears_pointers(self):
        cache = StaticKVCache(1, 128, 2, 2, 64, device="cpu")
        k = torch.randn(1, 2, 10, 64)
        v = torch.randn(1, 2, 10, 64)
        for l in range(2):
            cache.update(l, k, v, start_pos=0)
        assert cache.current_pos == 10

        cache.reset()
        assert cache.current_pos == 0
        assert cache.get_seq_length(0) == 0
        assert cache.get_seq_length(1) == 0

    def test_hf_crop_negative_support(self):
        cache = StaticKVCache(1, 128, 2, 2, 64, device="cpu")
        cache.rollback(20)
        cache.crop(-5)
        assert cache.current_pos == 15

    def test_getitem_duck_typing(self):
        cache = StaticKVCache(1, 128, 2, 2, 64, device="cpu")
        k = torch.randn(1, 2, 5, 64)
        v = torch.randn(1, 2, 5, 64)
        for l in range(2):
            cache.update(l, k, v, start_pos=0)

        k_slice, v_slice = cache[0]
        assert k_slice.shape == (1, 2, 5, 64)
        assert v_slice.shape == (1, 2, 5, 64)

    def test_from_model_config(self):
        class DummyConfig:
            num_hidden_layers = 12
            num_key_value_heads = 4
            num_attention_heads = 8
            hidden_size = 512

        config = DummyConfig()
        cache = StaticKVCache.from_model_config(config, max_batch_size=1, max_seq_len=256, device="cpu")
        assert cache.num_layers == 12
        assert cache.num_heads == 4
        assert cache.head_dim == 64  # 512 // 8
        assert cache.max_seq_len == 256


class TestNewlyFixedEdgeCases:
    """Verifies edge cases and compatibility fixes added in Iteration 2."""

    def test_negative_start_pos_raises_error(self):
        """Negative start_pos raises ValueError in update() and compute_position_ids."""
        cache = StaticKVCache(1, 64, 2, 2, 32, device="cpu")
        k = torch.randn(1, 2, 2, 32)
        v = torch.randn(1, 2, 2, 32)

        with pytest.raises(ValueError, match="start_pos must be non-negative"):
            cache.update(0, k, v, start_pos=-1)

        with pytest.raises(ValueError, match="start_pos must be non-negative"):
            cache.update(0, k, v, start_pos=-5)

        with pytest.raises(ValueError, match="start_pos must be non-negative"):
            compute_position_ids(start_pos=-1, num_tokens=4)

    def test_cache_position_varied_types(self):
        """cache_kwargs['cache_position'] parses 0-D tensors, 1-D tensors, ints, and lists."""
        cache = StaticKVCache(1, 64, 2, 2, 32, device="cpu")
        k = torch.randn(1, 2, 1, 32)
        v = torch.randn(1, 2, 1, 32)

        # 0-D tensor
        k_out, v_out = cache.update(k, v, 0, cache_kwargs={"cache_position": torch.tensor(5)})
        assert cache.current_pos == 6
        assert k_out.shape == (1, 2, 6, 32)

        # 1-D tensor
        k_out, v_out = cache.update(k, v, 0, cache_kwargs={"cache_position": torch.tensor([10])})
        assert cache.current_pos == 11

        # Python integer
        k_out, v_out = cache.update(k, v, 0, cache_kwargs={"cache_position": 15})
        assert cache.current_pos == 16

        # Python list
        k_out, v_out = cache.update(k, v, 0, cache_kwargs={"cache_position": [20]})
        assert cache.current_pos == 21

    def test_from_model_config_none_kv_heads(self):
        """from_model_config defaults num_key_value_heads=None to num_attention_heads."""
        class MockMHAConfig:
            num_hidden_layers = 6
            num_attention_heads = 8
            num_key_value_heads = None
            hidden_size = 512
            head_dim = None

        cache = StaticKVCache.from_model_config(MockMHAConfig, device="cpu")
        assert cache.num_layers == 6
        assert cache.num_heads == 8
        assert cache.head_dim == 64
        assert cache.dtype == torch.float32

    def test_k_cache_v_cache_properties(self):
        """k_cache and v_cache properties return 5D tensor views sharing memory with storage."""
        cache = StaticKVCache(
            max_batch_size=2,
            max_seq_len=128,
            num_layers=3,
            num_heads=4,
            head_dim=32,
            device="cpu",
        )
        assert cache.k_cache.shape == (3, 2, 4, 128, 32)
        assert cache.v_cache.shape == (3, 2, 4, 128, 32)
        assert cache.k_cache.data_ptr() == cache.storage[:, 0].data_ptr()
        assert cache.v_cache.data_ptr() == cache.storage[:, 1].data_ptr()

    def test_numpy_and_3d_input_conversion(self):
        """Numpy arrays and 3D tensors are converted and unsqueezed cleanly."""
        import numpy as np
        cache = StaticKVCache(1, 64, 2, 2, 32, device="cpu")

        # Numpy 4D input
        k_np = np.ones((1, 2, 4, 32), dtype=np.float32)
        v_np = np.full((1, 2, 4, 32), 2.0, dtype=np.float32)
        k_out, v_out = cache.update(0, k_np, v_np, start_pos=0)
        assert isinstance(k_out, torch.Tensor)
        assert k_out.shape == (1, 2, 4, 32)
        assert torch.allclose(k_out[0, 0, 0, :], torch.tensor(1.0))

        # 3D Tensor input (num_heads, seq_len, head_dim)
        k_3d = torch.randn(2, 3, 32)
        v_3d = torch.randn(2, 3, 32)
        k_out3, v_out3 = cache.update(0, k_3d, v_3d, start_pos=4)
        assert k_out3.shape == (1, 2, 7, 32)
        assert cache.current_pos == 7

    def test_default_dtype_float32(self):
        """StaticKVCache defaults to torch.float32 and respects explicit float16 override."""
        cache_default = StaticKVCache(1, 32, 2, 2, 16, device="cpu")
        assert cache_default.dtype == torch.float32
        assert cache_default.storage.dtype == torch.float32

        cache_fp16 = StaticKVCache(1, 32, 2, 2, 16, device="cpu", dtype=torch.float16)
        assert cache_fp16.dtype == torch.float16
        assert cache_fp16.storage.dtype == torch.float16


if __name__ == "__main__":
    pytest.main(["-v", __file__])
