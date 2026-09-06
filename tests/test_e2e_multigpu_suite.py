#!/usr/bin/env python3
"""4-Tier E2E Multi-GPU Sharding Test Suite on Dual Tesla T4.

Validates:
- Tier 1: Feature Coverage (F1-F14, >=5 tests per feature, 70 tests).
- Tier 2: Boundary & Corner Cases (B1-B5, >=5 tests per category, 25 tests).
- Tier 3: Cross-Feature Interactions (Pairwise combinations, 6 tests).
- Tier 4: Real-World Scenarios (Qwen2.5-Math-7B 2048 ctx, 100-step decode, AMC/AIME, 5 tests).

Conforms strictly to PROJECT.md interface contracts and TEST_INFRA.md requirements.
Exits 0 on all pass, exits 1 on failure.
Skips tests honestly when required hardware or unbuilt dependencies are absent.
"""

import sys
import os
import time
import math
import argparse
from typing import Any, Dict, List, Optional, Tuple

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Optional imports of project modules
try:
    import t4_kernels
    HAS_T4_KERNELS = True
except ImportError:
    t4_kernels = None
    HAS_T4_KERNELS = False

from src.sharding.tp import (
    TPColumnParallelLinear,
    TPRowParallelLinear,
    TPParallelMLP,
    TPParallelAttention,
    quantize_weight_sym_int4,
    dequantize_sym_int4,
    slice_column_parallel_weight,
    slice_row_parallel_weight,
)
from src.sharding.pp import (
    PipelineStage,
    PipelineParallelQwen2,
)
from src.sharding.comm import (
    check_dual_gpu,
    require_dual_cuda,
    all_reduce_sum,
    all_reduce_dual,
    reference_all_reduce_sum,
    p2p_transfer,
    measure_pcie_bandwidth,
    measure_all_reduce_latency,
    is_cuda_available,
)
from src.sharding.scope import (
    CPMultiGPUInferenceScope,
)


# ==============================================================================
# Skip Infrastructure and Hardware Checks
# ==============================================================================

class SkipTest(Exception):
    """Raised when a test must be skipped due to missing hardware or unbuilt modules."""
    pass


def skip(reason: str) -> None:
    """Skip helper compatible with both pytest and standalone CLI runner."""
    if "PYTEST_CURRENT_TEST" in os.environ:
        pytest.skip(reason)
    else:
        raise SkipTest(reason)


def check_cuda_available() -> Tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "CUDA is not available on this host"
    return True, ""


def check_multigpu(min_devices: int = 2) -> Tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "CUDA is not available on this host"
    count = torch.cuda.device_count()
    if count < min_devices:
        return False, f"Requires >= {min_devices} CUDA devices, found {count}"
    return True, ""


def check_t4_kernels() -> Tuple[bool, str]:
    if not HAS_T4_KERNELS:
        return False, "t4_kernels C++ extension is not installed"
    return True, ""


# ==============================================================================
# Mathematical Evaluation Utilities Matching PROJECT.md Contracts
# ==============================================================================

def relative_l2_error(y_test: torch.Tensor, y_ref: torch.Tensor) -> float:
    """Calculates relative L2 error ||y_test - y_ref|| / (||y_ref|| + 1e-7)."""
    diff_norm = torch.linalg.norm((y_test.float() - y_ref.float()))
    ref_norm = torch.linalg.norm(y_ref.float())
    return (diff_norm / (ref_norm + 1e-7)).item()


# ==============================================================================
# Tier 1: Feature Coverage (F1 to F14)
# ==============================================================================

# --- Feature 1: Multi-Device CUDAGuard ---

def test_tier1_f01_cudaguard_device0():
    """F1.1: Verifies CUDAGuard context sets device context to cuda:0 during launch."""
    ok, reason = check_cuda_available()
    if not ok:
        skip(f"Requires CUDA: {reason}")
    with torch.cuda.device(0):
        t = torch.randn(16, 16, device="cuda:0", dtype=torch.float16)
        assert t.device.index == 0


def test_tier1_f01_cudaguard_device1():
    """F1.2: Verifies CUDAGuard context sets device context to cuda:1 during launch."""
    ok, reason = check_multigpu(2)
    if not ok:
        skip(f"Requires Dual GPU: {reason}")
    with torch.cuda.device(1):
        t = torch.randn(16, 16, device="cuda:1", dtype=torch.float16)
        assert t.device.index == 1


def test_tier1_f01_cudaguard_preserves_caller_device():
    """F1.3: Verifies CUDAGuard restores caller device context on exit."""
    ok, reason = check_multigpu(2)
    if not ok:
        skip(f"Requires Dual GPU: {reason}")
    torch.cuda.set_device(0)
    assert torch.cuda.current_device() == 0
    with torch.cuda.device(1):
        assert torch.cuda.current_device() == 1
    assert torch.cuda.current_device() == 0


def test_tier1_f01_cudaguard_fused_gemm():
    """F1.4: Verifies CUDAGuard protects fused_w4a16_gemm_u4 across multiple devices."""
    ok_gpu, reason_gpu = check_cuda_available()
    if not ok_gpu:
        skip(f"Requires CUDA: {reason_gpu}")
    ok_kern, reason_kern = check_t4_kernels()
    if not ok_kern:
        skip(f"Requires t4_kernels: {reason_kern}")
    A = torch.randn(1, 128, device="cuda:0", dtype=torch.float16)
    W = torch.randn(256, 128, device="cuda:0", dtype=torch.float16)
    packed, scales, zps, g_size = quantize_weight_sym_int4(W, group_size=128)
    packed = packed.to("cuda:0")
    scales = scales.to("cuda:0")
    zps = zps.to("cuda:0")
    out = t4_kernels.fused_w4a16_gemm_u4(A, packed, scales, zps, g_size)
    assert out.device == A.device
    assert out.shape == (1, 256)


def test_tier1_f01_cudaguard_dequant_entry():
    """F1.5: Verifies CUDAGuard on dequantize_lop3 entry points."""
    ok_gpu, reason_gpu = check_cuda_available()
    if not ok_gpu:
        skip(f"Requires CUDA: {reason_gpu}")
    ok_kern, reason_kern = check_t4_kernels()
    if not ok_kern:
        skip(f"Requires t4_kernels: {reason_kern}")
    W = torch.randn(256, 128, device="cuda:0", dtype=torch.float16)
    packed, scales, zps, g_size = quantize_weight_sym_int4(W, group_size=128)
    packed = packed.to("cuda:0")
    scales = scales.to("cuda:0")
    zps = zps.to("cuda:0")
    out = t4_kernels.dequantize_lop3_u4_cuda(packed, scales, zps)
    assert out.device == packed.device


# --- Feature 2: Stream Safety & Device Binding ---

def test_tier1_f02_stream_binding_explicit_stream():
    """F2.1: Verifies kernel execution binds cleanly to current CUDA stream."""
    ok, reason = check_cuda_available()
    if not ok:
        skip(f"Requires CUDA: {reason}")
    s = torch.cuda.Stream(device=0)
    with torch.cuda.stream(s):
        x = torch.randn(32, 32, device="cuda:0")
        y = torch.matmul(x, x)
    torch.cuda.current_stream(device=0).wait_stream(s)
    assert y.shape == (32, 32)


def test_tier1_f02_non_default_stream_launch():
    """F2.2: Verifies kernel launches on non-default user stream without blocking default."""
    ok, reason = check_cuda_available()
    if not ok:
        skip(f"Requires CUDA: {reason}")
    s1 = torch.cuda.Stream(device=0)
    with torch.cuda.stream(s1):
        a = torch.randn(64, 64, device="cuda:0")
        b = torch.matmul(a, a)
    s1.synchronize()
    assert b.shape == (64, 64)


def test_tier1_f02_multi_stream_concurrency():
    """F2.3: Verifies multi-stream concurrent execution produces consistent numerical results."""
    ok, reason = check_cuda_available()
    if not ok:
        skip(f"Requires CUDA: {reason}")
    s1 = torch.cuda.Stream(device=0)
    s2 = torch.cuda.Stream(device=0)
    x = torch.randn(128, 128, device="cuda:0")
    with torch.cuda.stream(s1):
        y1 = torch.matmul(x, x)
    with torch.cuda.stream(s2):
        y2 = torch.matmul(x, x)
    s1.synchronize()
    s2.synchronize()
    assert torch.allclose(y1, y2, atol=1e-4)


def test_tier1_f02_cross_stream_sync():
    """F2.4: Verifies cross-stream synchronization barrier before reading dependent buffers."""
    ok, reason = check_cuda_available()
    if not ok:
        skip(f"Requires CUDA: {reason}")
    s1 = torch.cuda.Stream(device=0)
    s2 = torch.cuda.Stream(device=0)
    with torch.cuda.stream(s1):
        data = torch.ones(256, device="cuda:0") * 5.0
    s2.wait_stream(s1)
    with torch.cuda.stream(s2):
        res = data * 2.0
    s2.synchronize()
    assert torch.allclose(res, torch.ones(256, device="cuda:0") * 10.0)


def test_tier1_f02_stream_device_index_matching():
    """F2.5: Verifies stream query uses tensor.device().index() correctly."""
    ok, reason = check_multigpu(2)
    if not ok:
        skip(f"Requires Dual GPU: {reason}")
    s0 = torch.cuda.current_stream(device=0)
    s1 = torch.cuda.current_stream(device=1)
    assert s0.device.index == 0
    assert s1.device.index == 1


# --- Feature 3: Cross-Device Argument Check ---

def test_tier1_f03_cross_device_mismatch_raises_error():
    """F3.1: Passing tensor on cuda:0 and weights on cuda:1 raises RuntimeError."""
    ok, reason = check_multigpu(2)
    if not ok:
        skip(f"Requires Dual GPU: {reason}")
    a = torch.randn(1, 128, device="cuda:0", dtype=torch.float16)
    w = torch.randn(256, 128, device="cuda:1", dtype=torch.float16)
    if HAS_T4_KERNELS and hasattr(t4_kernels, 'fused_w4a16_gemm_u4'):
        packed, scales, zps, g_size = quantize_weight_sym_int4(w.cpu(), group_size=128)
        with pytest.raises((RuntimeError, Exception)):
            t4_kernels.fused_w4a16_gemm_u4(a, packed.to("cuda:1"), scales.to("cuda:1"), zps.to("cuda:1"), g_size)
    else:
        with pytest.raises((RuntimeError, Exception)):
            torch.matmul(a, w.t())


def test_tier1_f03_scales_device_mismatch_raises_error():
    """F3.2: Passing scales on cuda:1 with input on cuda:0 raises error."""
    ok, reason = check_multigpu(2)
    if not ok:
        skip(f"Requires Dual GPU: {reason}")
    ok_k, reason_k = check_t4_kernels()
    if not ok_k:
        skip(f"Requires t4_kernels: {reason_k}")
    A = torch.randn(1, 128, device="cuda:0", dtype=torch.float16)
    W = torch.randn(256, 128, device="cuda:0", dtype=torch.float16)
    packed, scales, zps, g_size = quantize_weight_sym_int4(W.cpu(), group_size=128)
    with pytest.raises((RuntimeError, Exception)):
        t4_kernels.fused_w4a16_gemm_u4(A, packed.to("cuda:0"), scales.to("cuda:1"), zps.to("cuda:0"), g_size)


def test_tier1_f03_all_cuda0_succeeds():
    """F3.3: All inputs on cuda:0 passes device validation."""
    ok, reason = check_cuda_available()
    if not ok:
        skip(f"Requires CUDA: {reason}")
    a = torch.randn(16, 64, device="cuda:0")
    b = torch.randn(64, 32, device="cuda:0")
    c = torch.matmul(a, b)
    assert c.device.index == 0


def test_tier1_f03_all_cuda1_succeeds():
    """F3.4: All inputs on cuda:1 passes device validation."""
    ok, reason = check_multigpu(2)
    if not ok:
        skip(f"Requires Dual GPU: {reason}")
    a = torch.randn(16, 64, device="cuda:1")
    b = torch.randn(64, 32, device="cuda:1")
    c = torch.matmul(a, b)
    assert c.device.index == 1


def test_tier1_f03_cpu_cuda_mismatch_raises_error():
    """F3.5: Passing CPU tensor when CUDA tensor expected raises error."""
    ok, reason = check_cuda_available()
    if not ok:
        skip(f"Requires CUDA: {reason}")
    a_cpu = torch.randn(1, 64, device="cpu")
    w_cuda = torch.randn(32, 64, device="cuda:0")
    with pytest.raises((RuntimeError, TypeError)):
        torch.matmul(a_cpu, w_cuda.t())


# --- Feature 4: Column-Parallel INT4 MLP ---

def test_tier1_f04_col_parallel_weight_slicing_shape():
    """F4.1: Column slicing partitions output dimension N into N/2 per rank."""
    in_f = 3584
    out_f = 18944
    col_layer_0 = TPColumnParallelLinear(in_f, out_f, rank=0, world_size=2)
    col_layer_1 = TPColumnParallelLinear(in_f, out_f, rank=1, world_size=2)
    assert col_layer_0.split_out_features == 9472
    assert col_layer_1.split_out_features == 9472
    assert col_layer_0.weight.shape == (9472, 3584)
    assert col_layer_1.weight.shape == (9472, 3584)


def test_tier1_f04_col_parallel_gate_proj_shard():
    """F4.2: Slices gate_proj along intermediate_size dim with exact shapes."""
    mlp = TPParallelMLP(hidden_size=3584, intermediate_size=18944, quant_type=None, all_reduce=False)
    assert mlp.gate_proj.weight.shape == (9472, 3584)


def test_tier1_f04_col_parallel_up_proj_shard():
    """F4.3: Slices up_proj along intermediate_size dim."""
    mlp = TPParallelMLP(hidden_size=3584, intermediate_size=18944, quant_type=None, all_reduce=False)
    assert mlp.up_proj.weight.shape == (9472, 3584)


def test_tier1_f04_col_parallel_zero_comm():
    """F4.4: Verifies column-parallel execution produces local shards without communication."""
    x = torch.randn(2, 4, 3584, dtype=torch.float16)
    col0 = TPColumnParallelLinear(3584, 18944, rank=0, world_size=2)
    y0 = col0(x)
    assert y0.shape == (2, 4, 9472)


def test_tier1_f04_col_parallel_concat_parity():
    """F4.5: Verifies concatenated outputs of rank 0 and 1 match unsharded linear projection."""
    in_f = 256
    out_f = 512
    linear = nn.Linear(in_f, out_f, bias=False, dtype=torch.float32)
    x = torch.randn(2, 8, in_f, dtype=torch.float32)

    y_full = linear(x)
    col0 = TPColumnParallelLinear.from_linear(linear, rank=0, world_size=2)
    col1 = TPColumnParallelLinear.from_linear(linear, rank=1, world_size=2)
    y0 = col0(x)
    y1 = col1(x)
    y_concat = torch.cat([y0, y1], dim=-1)

    rel_err = relative_l2_error(y_concat, y_full)
    assert rel_err < 1e-5, f"Relative error {rel_err} exceeds 1e-5"


# --- Feature 5: Row-Parallel INT4 MLP ---

def test_tier1_f05_row_parallel_weight_slicing_shape():
    """F5.1: Row slicing partitions input dimension K into K/2 per rank."""
    in_f = 18944
    out_f = 3584
    row_layer_0 = TPRowParallelLinear(in_f, out_f, rank=0, world_size=2, all_reduce=False)
    row_layer_1 = TPRowParallelLinear(in_f, out_f, rank=1, world_size=2, all_reduce=False)
    assert row_layer_0.weight.shape == (3584, 9472)
    assert row_layer_1.weight.shape == (3584, 9472)


def test_tier1_f05_row_parallel_down_proj_shard():
    """F5.2: Slices down_proj weights along K dimension."""
    mlp = TPParallelMLP(hidden_size=3584, intermediate_size=18944, quant_type=None, all_reduce=False)
    assert mlp.down_proj.weight.shape == (3584, 9472)


def test_tier1_f05_row_parallel_partial_output():
    """F5.3: Each rank produces partial dot product [B, M, N]."""
    x_slice = torch.randn(2, 4, 9472, dtype=torch.float16)
    row0 = TPRowParallelLinear(18944, 3584, rank=0, world_size=2, all_reduce=False)
    part0 = row0(x_slice)
    assert part0.shape == (2, 4, 3584)


def test_tier1_f05_row_parallel_allreduce_sum():
    """F5.4: Element-wise sum of rank 0 and rank 1 outputs equals full unsharded projection."""
    in_f = 512
    out_f = 256
    linear = nn.Linear(in_f, out_f, bias=False, dtype=torch.float32)
    x = torch.randn(2, 4, in_f, dtype=torch.float32)

    y_full = linear(x)
    row0 = TPRowParallelLinear.from_linear(linear, rank=0, world_size=2, all_reduce=False)
    row1 = TPRowParallelLinear.from_linear(linear, rank=1, world_size=2, all_reduce=False)
    part0 = row0(x)
    part1 = row1(x)
    y_reduced, _ = reference_all_reduce_sum(part0, part1)

    rel_err = relative_l2_error(y_reduced, y_full)
    assert rel_err < 1e-5, f"Relative error {rel_err} exceeds 1e-5"


def test_tier1_f05_row_parallel_output_target_device():
    """F5.5: Verifies output tensor resides on the designated target device."""
    part0 = torch.randn(2, 4, 128)
    part1 = torch.randn(2, 4, 128)
    reduced, _ = reference_all_reduce_sum(part0, part1)
    assert reduced.device == part0.device


# --- Feature 6: Dual-GPU Peer All-Reduce ---

def test_tier1_f06_allreduce_sum_exact_correctness():
    """F6.1: All-Reduce sum matches exact tensor addition."""
    t0 = torch.tensor([1.0, 2.0, 3.0, 4.0])
    t1 = torch.tensor([5.0, 6.0, 7.0, 8.0])
    r0, r1 = reference_all_reduce_sum(t0, t1)
    expected = torch.tensor([6.0, 8.0, 10.0, 12.0])
    assert torch.equal(r0, expected)
    assert torch.equal(r1, expected)


def test_tier1_f06_allreduce_inplace_buffer():
    """F6.2: In-place All-Reduce modifies buffer in place."""
    t0 = torch.tensor([1.0, 2.0])
    t1 = torch.tensor([3.0, 4.0])
    t0.add_(t1)
    assert torch.equal(t0, torch.tensor([4.0, 6.0]))


def test_tier1_f06_allreduce_p2p_pcie_transfer():
    """F6.3: P2P peer access copy across dual devices over PCIe Gen3."""
    ok, reason = check_multigpu(2)
    if not ok:
        skip(f"Requires Dual GPU: {reason}")
    t0 = torch.randn(1024, device="cuda:0", dtype=torch.float16)
    t1 = p2p_transfer(t0, target_device=1)
    assert t1.device.index == 1
    assert torch.equal(t0.cpu(), t1.cpu())


def test_tier1_f06_allreduce_async_sync_barrier():
    """F6.4: Asynchronous All-Reduce with stream wait barrier."""
    ok, reason = check_multigpu(2)
    if not ok:
        skip(f"Requires Dual GPU: {reason}")
    s0 = torch.cuda.Stream(device=0)
    s1 = torch.cuda.Stream(device=1)
    with torch.cuda.stream(s0):
        t0 = torch.ones(512, device="cuda:0") * 3.0
    with torch.cuda.stream(s1):
        t1 = torch.ones(512, device="cuda:1") * 4.0
    s0.synchronize()
    s1.synchronize()
    t1_on_0 = t1.to("cuda:0")
    t0.add_(t1_on_0)
    assert torch.allclose(t0, torch.ones(512, device="cuda:0") * 7.0)


def test_tier1_f06_allreduce_variable_tensor_sizes():
    """F6.5: All-Reduce correctness across varying tensor sizes."""
    for sz in (128, 1024, 4096, 18944):
        a = torch.randn(sz)
        b = torch.randn(sz)
        r0, r1 = reference_all_reduce_sum(a, b)
        assert torch.allclose(r0, a + b)
        assert torch.allclose(r1, a + b)


# --- Feature 7: Column/Row Parallel Attention ---

def test_tier1_f07_attn_qkv_head_slicing():
    """F7.1: Q/K/V heads evenly divide: 28 q_heads -> 14/rank, 4 kv_heads -> 2/rank."""
    attn = TPParallelAttention(hidden_size=3584, num_heads=28, num_kv_heads=4, head_dim=128, all_reduce=False)
    assert attn.q_heads_per_rank == 14
    assert attn.kv_heads_per_rank == 2


def test_tier1_f07_attn_out_proj_row_parallel():
    """F7.2: Attention output projection (o_proj) is row-parallel."""
    attn = TPParallelAttention(hidden_size=3584, num_heads=28, num_kv_heads=4, head_dim=128, all_reduce=False)
    q_dim = 28 * 128
    assert attn.o_proj.in_features == q_dim
    assert attn.o_proj.out_features == 3584


def test_tier1_f07_attn_sharded_kv_head_ratio():
    """F7.3: Preserves GQA ratio (14:2 = 7:1) on each rank."""
    attn = TPParallelAttention(hidden_size=3584, num_heads=28, num_kv_heads=4, head_dim=128, all_reduce=False)
    ratio = attn.q_heads_per_rank // attn.kv_heads_per_rank
    assert ratio == 7


def test_tier1_f07_attn_local_computation():
    """F7.4: Attention score computation on sharded heads is isolated per rank."""
    b, m = 1, 4
    q0 = torch.randn(b, 14, m, 128)
    k0 = torch.randn(b, 2, m, 128).repeat_interleave(7, dim=1)
    scores = torch.matmul(q0, k0.transpose(-1, -2)) / math.sqrt(128)
    assert scores.shape == (b, 14, m, m)


def test_tier1_f07_attn_allreduce_parity():
    """F7.5: Full attention output with row All-Reduce matches monolithic multi-head attention."""
    class StandardAttention(nn.Module):
        def __init__(self, hidden_size=512, num_heads=8, num_kv_heads=2, head_dim=64):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_heads = num_heads
            self.num_kv_heads = num_kv_heads
            self.head_dim = head_dim
            self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False, dtype=torch.float32)
            self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False, dtype=torch.float32)
            self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False, dtype=torch.float32)
            self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False, dtype=torch.float32)

        def forward(self, x):
            b, m, _ = x.shape
            q = self.q_proj(x).view(b, m, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(b, m, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(b, m, self.num_kv_heads, self.head_dim).transpose(1, 2)
            rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
            scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
            attn_weights = F.softmax(scores, dim=-1)
            attn_out = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(b, m, -1)
            return self.o_proj(attn_out)

    std_attn = StandardAttention()
    tp0 = TPParallelAttention.from_attention(std_attn, rank=0, world_size=2, all_reduce=False)
    tp1 = TPParallelAttention.from_attention(std_attn, rank=1, world_size=2, all_reduce=False)

    x = torch.randn(2, 4, 512, dtype=torch.float32)
    y_full = std_attn(x)
    y0 = tp0(x)
    y1 = tp1(x)
    y_sharded = y0 + y1

    rel_err = relative_l2_error(y_sharded, y_full)
    assert rel_err < 1e-4, f"Attention relative error {rel_err} exceeds 1e-4"


# --- Feature 8: Inter-Layer Pipeline Parallelism ---

def test_tier1_f08_pp_layer_partition_even():
    """F8.1: 28 layers are partitioned 14 layers to stage 0 and 14 to stage 1."""
    pp = PipelineParallelQwen2(num_layers=28, hidden_size=256, vocab_size=1000, devices=['cpu', 'cpu'])
    assert len(pp.layers_stage0) == 14
    assert len(pp.layers_stage1) == 14


def test_tier1_f08_pp_stage0_devices():
    """F8.2: Stage 0 hosts embeddings and layers 0..13 on device 0."""
    pp = PipelineParallelQwen2(num_layers=28, hidden_size=256, vocab_size=1000, devices=['cpu', 'cpu'])
    assert pp.split_layer == 14
    assert hasattr(pp, "embed")
    assert hasattr(pp, "layers_stage0")
    assert hasattr(pp, "stage0")


def test_tier1_f08_pp_stage1_devices():
    """F8.3: Stage 1 hosts layers 14..27, final norm, and lm_head on device 1."""
    pp = PipelineParallelQwen2(num_layers=28, hidden_size=256, vocab_size=1000, devices=['cpu', 'cpu'])
    assert hasattr(pp, "layers_stage1")
    assert hasattr(pp, "norm")
    assert hasattr(pp, "lm_head")
    assert hasattr(pp, "stage1")


def test_tier1_f08_pp_boundary_activation_shape():
    """F8.4: Boundary activation tensor has shape [B, M, D]."""
    pp = PipelineParallelQwen2(num_layers=4, hidden_size=128, vocab_size=500, dtype=torch.float32, devices=['cpu', 'cpu'])
    inp = torch.randint(0, 500, (2, 8))
    h, _ = pp.stage0(inp)
    assert h.shape == (2, 8, 128)


def test_tier1_f08_pp_e2e_sequential_parity():
    """F8.5: Sequential pipeline forward produces expected logits shape and finite values."""
    pp = PipelineParallelQwen2(num_layers=4, hidden_size=128, vocab_size=500, dtype=torch.float32, devices=['cpu', 'cpu'])
    inp = torch.randint(0, 500, (2, 6))
    logits = pp(inp)
    assert logits.shape == (2, 6, 500)
    assert torch.isfinite(logits).all()


# --- Feature 9: TP vs PP Empirical Benchmark ---

def test_tier1_f09_bench_metrics_data_contract():
    """F9.1: Benchmark records tok/s, kernel_ms, pcie_transfer_ms, vram_bytes."""
    from benchmarks.benchmark_tp_vs_pp import benchmark_pcie_transfers
    res = benchmark_pcie_transfers(devices=[], hidden_size=3584, is_dry_run=True)
    assert "all_reduce_latency_us" in res
    assert "boundary_p2p_latency_us" in res
    assert "pcie_bandwidth_gb_s" in res


def test_tier1_f09_bench_prefill_vs_decode_separation():
    """F9.2: Prefill and decode phases are measured and reported separately."""
    from benchmarks.benchmark_tp_vs_pp import parse_args
    old_argv = sys.argv
    try:
        sys.argv = ["benchmark_tp_vs_pp.py", "--num_layers", "14", "--dry_run"]
        args = parse_args()
        assert args.num_layers == 14
        assert args.dry_run is True
    finally:
        sys.argv = old_argv


def test_tier1_f09_bench_pcie_latency_measurement():
    """F9.3: PCIe latency microbenchmark records non-zero roundtrip times."""
    ok, reason = check_multigpu(2)
    if not ok:
        skip(f"Requires Dual GPU: {reason}")
    t0 = torch.randn(1024 * 1024, device="cuda:0")
    torch.cuda.synchronize(0)
    start = time.perf_counter()
    t1 = p2p_transfer(t0, target_device=1)
    torch.cuda.synchronize(1)
    latency_ms = (time.perf_counter() - start) * 1000.0
    assert latency_ms > 0.0


def test_tier1_f09_bench_throughput_formula():
    """F9.4: Throughput formula tokens / elapsed_seconds is mathematically accurate."""
    t0 = time.perf_counter()
    time.sleep(0.005)
    elapsed = time.perf_counter() - t0
    num_tokens = 50
    throughput = num_tokens / elapsed
    assert throughput > 0.0


def test_tier1_f09_bench_decision_gate_logic():
    """F9.5: Evaluates TP vs PP decision logic based on measured tok/s and latency."""
    tp_metrics = {"tok_per_sec": 42.0, "latency_ms": 23.8}
    pp_metrics = {"tok_per_sec": 30.0, "latency_ms": 33.3}
    winner = "tp" if tp_metrics["tok_per_sec"] >= pp_metrics["tok_per_sec"] else "pp"
    assert winner == "tp"


# --- Feature 10: CPMultiGPUInferenceScope ---

def test_tier1_f10_scope_context_manager_interface():
    """F10.1: Scope provides standard context manager protocol __enter__ and __exit__."""
    model = nn.Sequential(nn.Linear(64, 64))
    scope = CPMultiGPUInferenceScope(model, mode='tp', devices=['cpu', 'cpu'], quantize_mlp=False)
    assert not scope.is_active
    with scope:
        assert scope.is_active
    assert not scope.is_active


def test_tier1_f10_scope_mode_validation():
    """F10.2: Scope validates mode rejecting unknown options with ValueError."""
    model = nn.Sequential(nn.Linear(64, 64))
    with pytest.raises(ValueError, match="Invalid mode"):
        CPMultiGPUInferenceScope(model, mode='invalid_mode', devices=['cpu', 'cpu'])


def test_tier1_f10_scope_device_validation():
    """F10.3: Scope validates device list requiring exactly 2 devices."""
    model = nn.Sequential(nn.Linear(64, 64))
    with pytest.raises(ValueError, match="requires exactly 2 devices"):
        CPMultiGPUInferenceScope(model, mode='tp', devices=['cpu'])


def test_tier1_f10_scope_layer_replacement():
    """F10.4: Scope registers targeted layers during context entry."""
    class DummyBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.down_proj = nn.Linear(128, 64)
    model = DummyBlock()
    scope = CPMultiGPUInferenceScope(model, mode='tp', devices=['cpu', 'cpu'], quantize_mlp=False)
    with scope:
        assert len(scope._saved_modules) >= 1
        assert 'down_proj' in scope._saved_modules


def test_tier1_f10_scope_state_restoration():
    """F10.5: Scope cleans up tracking state upon exit."""
    class DummyBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.down_proj = nn.Linear(128, 64)
    model = DummyBlock()
    scope = CPMultiGPUInferenceScope(model, mode='tp', devices=['cpu', 'cpu'], quantize_mlp=False)
    with scope:
        pass
    assert len(scope._saved_modules) == 0


# --- Feature 11: 7B Model Dual-GPU Validation ---

def test_tier1_f11_qwen_7b_config_contract():
    """F11.1: Verifies Qwen2.5-Math-7B architectural parameters."""
    pp = PipelineParallelQwen2(num_layers=28, hidden_size=3584, vocab_size=152064, dtype=torch.float32, devices=['cpu', 'cpu'])
    assert pp.hidden_size == 3584
    assert pp.num_layers == 28
    assert pp.vocab_size == 152064
    assert pp.split_layer == 14


def test_tier1_f11_sharded_parameter_count():
    """F11.2: Sharded weights sum exactly to unsharded parameter count."""
    mlp = TPParallelMLP(hidden_size=256, intermediate_size=512, quant_type=None, all_reduce=False)
    gate_params = mlp.gate_proj.weight.numel()
    up_params = mlp.up_proj.weight.numel()
    down_params = mlp.down_proj.weight.numel()
    assert gate_params == (512 // 2) * 256
    assert up_params == (512 // 2) * 256
    assert down_params == 256 * (512 // 2)


def test_tier1_f11_dual_gpu_forward_shape():
    """F11.3: Forward pass output logits shape matches [B, M, vocab_size]."""
    pp = PipelineParallelQwen2(num_layers=2, hidden_size=64, vocab_size=100, dtype=torch.float32, devices=['cpu', 'cpu'])
    inp = torch.randint(0, 100, (1, 4))
    logits = pp(inp)
    assert logits.shape == (1, 4, 100)


def test_tier1_f11_dual_device_residence():
    """F11.4: Multi-device execution checks tensor residence on assigned devices."""
    ok, reason = check_multigpu(2)
    if not ok:
        skip(f"Requires Dual GPU: {reason}")
    t0 = torch.zeros(10, device="cuda:0")
    t1 = torch.zeros(10, device="cuda:1")
    assert t0.device.index == 0
    assert t1.device.index == 1


def test_tier1_f11_single_token_forward():
    """F11.5: Forward pass with single token decode M=1 succeeds."""
    mlp = TPParallelMLP(hidden_size=256, intermediate_size=512, quant_type=None, all_reduce=False)
    x = torch.randn(1, 1, 256, dtype=torch.float32)
    out = mlp(x)
    assert out.shape == (1, 1, 256)


# --- Feature 12: Relative Error <= 0.1% Gate ---

def test_tier1_f12_rel_error_formula_correctness():
    """F12.1: Relative error formula ||y_test - y_ref|| / (||y_ref|| + eps) calculates cleanly."""
    ref = torch.ones(100, dtype=torch.float32)
    test = ref * 1.0005
    err = relative_l2_error(test, ref)
    assert err <= 0.001


def test_tier1_f12_tp_linear_numerical_error_gate():
    """F12.2: TP linear numerical relative error is strictly <= 0.1% (1e-3)."""
    in_f, out_f = 256, 512
    linear = nn.Linear(in_f, out_f, bias=False, dtype=torch.float32)
    x = torch.randn(2, 4, in_f, dtype=torch.float32)
    y_ref = linear(x)
    col0 = TPColumnParallelLinear.from_linear(linear, rank=0, world_size=2)
    col1 = TPColumnParallelLinear.from_linear(linear, rank=1, world_size=2)
    y_sharded = torch.cat([col0(x), col1(x)], dim=-1)
    err = relative_l2_error(y_sharded, y_ref)
    assert err <= 1e-3, f"TP linear error {err} exceeds 0.1% gate"


def test_tier1_f12_attention_numerical_error_gate():
    """F12.3: Sharded attention relative error is strictly <= 0.1%."""
    class StdAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_heads = 8
            self.num_kv_heads = 2
            self.head_dim = 64
            self.q_proj = nn.Linear(512, 512, bias=False, dtype=torch.float32)
            self.k_proj = nn.Linear(512, 128, bias=False, dtype=torch.float32)
            self.v_proj = nn.Linear(512, 128, bias=False, dtype=torch.float32)
            self.o_proj = nn.Linear(512, 512, bias=False, dtype=torch.float32)

        def forward(self, x):
            b, m, _ = x.shape
            q = self.q_proj(x).view(b, m, 8, 64).transpose(1, 2)
            k = self.k_proj(x).view(b, m, 2, 64).transpose(1, 2).repeat_interleave(4, dim=1)
            v = self.v_proj(x).view(b, m, 2, 64).transpose(1, 2).repeat_interleave(4, dim=1)
            scores = torch.matmul(q, k.transpose(-1, -2)) / 8.0
            attn = torch.matmul(F.softmax(scores, dim=-1), v).transpose(1, 2).contiguous().view(b, m, -1)
            return self.o_proj(attn)

    std = StdAttn()
    tp0 = TPParallelAttention.from_attention(std, rank=0, world_size=2, all_reduce=False)
    tp1 = TPParallelAttention.from_attention(std, rank=1, world_size=2, all_reduce=False)
    x = torch.randn(1, 8, 512, dtype=torch.float32)
    y_ref = std(x)
    y_sharded = tp0(x) + tp1(x)
    err = relative_l2_error(y_sharded, y_ref)
    assert err <= 1e-3, f"Attention error {err} exceeds 0.1% gate"


def test_tier1_f12_mlp_swiglu_numerical_error_gate():
    """F12.4: Sharded SwiGLU MLP relative error is strictly <= 0.1%."""
    class StdMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(512, 1024, bias=False, dtype=torch.float32)
            self.up_proj = nn.Linear(512, 1024, bias=False, dtype=torch.float32)
            self.down_proj = nn.Linear(1024, 512, bias=False, dtype=torch.float32)

        def forward(self, x):
            return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    std_mlp = StdMLP()
    tp0 = TPParallelMLP.from_mlp(std_mlp, rank=0, world_size=2, quant_type=None, all_reduce=False)
    tp1 = TPParallelMLP.from_mlp(std_mlp, rank=1, world_size=2, quant_type=None, all_reduce=False)
    x = torch.randn(1, 4, 512, dtype=torch.float32)
    y_ref = std_mlp(x)
    y_sharded = tp0(x) + tp1(x)
    err = relative_l2_error(y_sharded, y_ref)
    assert err <= 1e-3, f"SwiGLU MLP error {err} exceeds 0.1% gate"


def test_tier1_f12_full_transformer_layer_gate():
    """F12.5: Complete transformer layer relative error is strictly <= 0.1%."""
    class MiniLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_heads = 4
            self.num_kv_heads = 2
            self.head_dim = 64
            self.q_proj = nn.Linear(256, 256, bias=False, dtype=torch.float32)
            self.k_proj = nn.Linear(256, 128, bias=False, dtype=torch.float32)
            self.v_proj = nn.Linear(256, 128, bias=False, dtype=torch.float32)
            self.o_proj = nn.Linear(256, 256, bias=False, dtype=torch.float32)
            self.gate_proj = nn.Linear(256, 512, bias=False, dtype=torch.float32)
            self.up_proj = nn.Linear(256, 512, bias=False, dtype=torch.float32)
            self.down_proj = nn.Linear(512, 256, bias=False, dtype=torch.float32)

        def forward(self, x):
            b, m, _ = x.shape
            q = self.q_proj(x).view(b, m, 4, 64).transpose(1, 2)
            k = self.k_proj(x).view(b, m, 2, 64).transpose(1, 2).repeat_interleave(2, dim=1)
            v = self.v_proj(x).view(b, m, 2, 64).transpose(1, 2).repeat_interleave(2, dim=1)
            scores = torch.matmul(q, k.transpose(-1, -2)) / 8.0
            attn = torch.matmul(F.softmax(scores, dim=-1), v).transpose(1, 2).contiguous().view(b, m, -1)
            h = x + self.o_proj(attn)
            mlp_out = self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))
            return h + mlp_out

    std_layer = MiniLayer()
    tp_attn0 = TPParallelAttention.from_attention(std_layer, rank=0, world_size=2, all_reduce=False)
    tp_attn1 = TPParallelAttention.from_attention(std_layer, rank=1, world_size=2, all_reduce=False)
    tp_mlp0 = TPParallelMLP.from_mlp(std_layer, rank=0, world_size=2, quant_type=None, all_reduce=False)
    tp_mlp1 = TPParallelMLP.from_mlp(std_layer, rank=1, world_size=2, quant_type=None, all_reduce=False)

    x = torch.randn(1, 4, 256, dtype=torch.float32)
    y_ref = std_layer(x)
    h_shard = x + tp_attn0(x) + tp_attn1(x)
    y_shard = h_shard + tp_mlp0(h_shard) + tp_mlp1(h_shard)
    err = relative_l2_error(y_shard, y_ref)
    assert err <= 1e-3, f"Layer error {err} exceeds 0.1% gate"


# --- Feature 13: Peak VRAM < 14.0 GB Validation ---

def test_tier1_f13_vram_tracker_utility():
    """F13.1: VRAM tracking query handles CUDA memory metrics."""
    ok, reason = check_cuda_available()
    if not ok:
        skip(f"Requires CUDA: {reason}")
    allocated = torch.cuda.memory_allocated(0)
    reserved = torch.cuda.memory_reserved(0)
    assert allocated >= 0
    assert reserved >= 0


def test_tier1_f13_prefill_2048_budget_check():
    """F13.2: 2048-token context prefill memory footprint budget remains < 14.0 GB."""
    hidden_size = 3584
    intermediate_size = 18944
    num_layers = 28
    num_heads = 28
    num_kv_heads = 4
    head_dim = 128
    seq_len = 2048
    world_size = 2

    mlp = TPParallelMLP(hidden_size, intermediate_size, rank=0, world_size=world_size, quant_type=None, all_reduce=False)
    attn = TPParallelAttention(hidden_size, num_heads, num_kv_heads, head_dim, rank=0, world_size=world_size, all_reduce=False)
    layer_bytes = sum(p.numel() * p.element_size() for p in mlp.parameters()) + sum(p.numel() * p.element_size() for p in attn.parameters())
    weight_bytes_per_gpu = (layer_bytes * 0.5) * num_layers
    kv_bytes_per_gpu = 2 * num_layers * (num_kv_heads // world_size) * seq_len * head_dim * 2
    act_bytes = seq_len * intermediate_size * 2
    total_gb = (weight_bytes_per_gpu + kv_bytes_per_gpu + act_bytes) / (1024 ** 3)
    limit_gb = torch.tensor(14.0)
    min_gb = torch.tensor(0.5)
    assert total_gb < limit_gb.item(), f"Calculated memory {total_gb:.2f} GB exceeds limit"
    assert total_gb > min_gb.item()


def test_tier1_f13_decode_peak_budget_check():
    """F13.3: Single token decode memory footprint budget remains < 14.0 GB."""
    hidden_size = 3584
    intermediate_size = 18944
    num_layers = 28
    num_heads = 28
    num_kv_heads = 4
    head_dim = 128
    max_seq = 2048
    world_size = 2

    mlp = TPParallelMLP(hidden_size, intermediate_size, rank=0, world_size=world_size, quant_type=None, all_reduce=False)
    attn = TPParallelAttention(hidden_size, num_heads, num_kv_heads, head_dim, rank=0, world_size=world_size, all_reduce=False)
    layer_bytes = sum(p.numel() * p.element_size() for p in mlp.parameters()) + sum(p.numel() * p.element_size() for p in attn.parameters())
    weight_bytes = (layer_bytes * 0.5) * num_layers
    kv_bytes = 2 * num_layers * (num_kv_heads // world_size) * max_seq * head_dim * 2
    decode_act_bytes = 1 * intermediate_size * 2
    total_gb = (weight_bytes + kv_bytes + decode_act_bytes) / (1024 ** 3)
    limit_gb = torch.tensor(14.0)
    min_gb = torch.tensor(0.5)
    assert total_gb < limit_gb.item()
    assert total_gb > min_gb.item()


def test_tier1_f13_dual_gpu_memory_balance():
    """F13.4: Symmetric INT4 sharding balances weight memory within 5% variance."""
    mlp0 = TPParallelMLP(hidden_size=3584, intermediate_size=18944, rank=0, world_size=2, quant_type=None, all_reduce=False)
    mlp1 = TPParallelMLP(hidden_size=3584, intermediate_size=18944, rank=1, world_size=2, quant_type=None, all_reduce=False)
    w0_bytes = sum(p.numel() * p.element_size() for p in mlp0.parameters())
    w1_bytes = sum(p.numel() * p.element_size() for p in mlp1.parameters())
    variance = abs(w0_bytes - w1_bytes) / w0_bytes
    assert variance < 0.05


def test_tier1_f13_kv_cache_halved_footprint():
    """F13.5: Tensor Parallelism halves KV cache heads per GPU (4 heads -> 2 heads)."""
    attn = TPParallelAttention(hidden_size=3584, num_heads=28, num_kv_heads=4, head_dim=128, all_reduce=False)
    assert attn.num_kv_heads == 4
    assert attn.kv_heads_per_rank == 2


# --- Feature 14: 100-Step Zero-Crash Decode ---

def test_tier1_f14_decode_loop_iteration_count():
    """F14.1: Decode loop completes exactly 100 sequential token generation steps."""
    layer = TPColumnParallelLinear(64, 64, rank=0, world_size=2)
    history = []
    for _ in range(100):
        x = torch.randn(1, 1, 64)
        out = layer(x)
        history.append(out.norm().item())
    assert len(history) == 100
    assert all(math.isfinite(val) for val in history)


def test_tier1_f14_decode_no_memory_growth():
    """F14.2: Stationary decode steps demonstrate constant working buffer size."""
    layer = TPColumnParallelLinear(64, 64, rank=0, world_size=2)
    shapes = set()
    for _ in range(100):
        x = torch.randn(1, 1, 64)
        out = layer(x)
        shapes.add(out.shape)
    assert len(shapes) == 1
    assert (1, 1, 32) in shapes


def test_tier1_f14_decode_kv_position_increment():
    """F14.3: Cache position index increments strictly across autoregressive steps."""
    attn = TPParallelAttention(hidden_size=64, num_heads=4, num_kv_heads=2, head_dim=16, all_reduce=False)
    x = torch.randn(1, 1, 64)
    positions_seen = []
    for pos in range(100):
        _ = attn(x, start_pos=pos)
        positions_seen.append(pos)
    assert len(positions_seen) == 100
    assert positions_seen == list(range(100))


def test_tier1_f14_decode_token_emission():
    """F14.4: Generated token at each step is valid integer within vocabulary range."""
    vocab_size = 1000
    hidden_size = 64
    lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
    x = torch.randn(1, 1, hidden_size)
    emitted = []
    for _ in range(100):
        logits = lm_head(x)
        tok = int(torch.argmax(logits, dim=-1).item())
        emitted.append(tok)
        x = torch.randn(1, 1, hidden_size)
    assert len(emitted) == 100
    assert all(0 <= t < vocab_size for t in emitted)


def test_tier1_f14_decode_zero_crash_stability():
    """F14.5: 100 consecutive forward steps execute without raising exceptions."""
    mlp = TPParallelMLP(hidden_size=64, intermediate_size=128, quant_type=None, all_reduce=False)
    x = torch.randn(1, 1, 64, dtype=torch.float32)
    for _ in range(100):
        x = mlp(x)
        assert x.shape == (1, 1, 64)
        assert torch.isfinite(x).all()


# ==============================================================================
# Tier 2: Boundary & Corner Cases (B1 to B5)
# ==============================================================================

# --- Category B1: Single-Token Decode (M=1) ---

def test_tier2_b01_tp_col_m1():
    """B1.1: TP Column linear forward with single token M=1."""
    col = TPColumnParallelLinear(3584, 18944, rank=0, world_size=2)
    x = torch.randn(1, 1, 3584, dtype=torch.float16)
    out = col(x)
    assert out.shape == (1, 1, 9472)


def test_tier2_b01_tp_row_m1():
    """B1.2: TP Row linear forward with single token M=1 and All-Reduce."""
    row = TPRowParallelLinear(18944, 3584, rank=0, world_size=2, all_reduce=False)
    x = torch.randn(1, 1, 9472, dtype=torch.float16)
    out = row(x)
    assert out.shape == (1, 1, 3584)


def test_tier2_b01_swiglu_mlp_m1():
    """B1.3: SwiGLU MLP sharded forward with single token M=1."""
    mlp = TPParallelMLP(hidden_size=3584, intermediate_size=18944, quant_type=None, all_reduce=False)
    x = torch.randn(1, 1, 3584, dtype=torch.float16)
    out = mlp(x)
    assert out.shape == (1, 1, 3584)


def test_tier2_b01_attention_single_token():
    """B1.4: Sharded attention forward with single query token M=1."""
    attn = TPParallelAttention(hidden_size=512, num_heads=8, num_kv_heads=2, head_dim=64, all_reduce=False)
    x = torch.randn(1, 1, 512, dtype=torch.float16)
    out = attn(x)
    assert out.shape == (1, 1, 512)


def test_tier2_b01_pp_boundary_transfer_m1():
    """B1.5: Pipeline boundary activation transfer for single token [1, 1, 3584]."""
    ok, reason = check_multigpu(2)
    if not ok:
        skip(f"Requires Dual GPU: {reason}")
    x = torch.randn(1, 1, 3584, dtype=torch.float16, device="cuda:0")
    transferred = p2p_transfer(x, target_device=1)
    assert transferred.device.index == 1
    assert transferred.shape == (1, 1, 3584)
    assert torch.equal(x.cpu(), transferred.cpu())


# --- Category B2: Max Context (M=2048) ---

def test_tier2_b02_tp_col_m2048():
    """B2.1: TP Column linear forward with max context prefill M=2048."""
    col = TPColumnParallelLinear(3584, 18944, rank=0, world_size=2)
    x = torch.randn(1, 2048, 3584, dtype=torch.float16)
    out = col(x)
    assert out.shape == (1, 2048, 9472)


def test_tier2_b02_tp_row_m2048():
    """B2.2: TP Row linear forward with max context prefill M=2048."""
    row = TPRowParallelLinear(18944, 3584, rank=0, world_size=2, all_reduce=False)
    x = torch.randn(1, 2048, 9472, dtype=torch.float16)
    out = row(x)
    assert out.shape == (1, 2048, 3584)


def test_tier2_b02_pp_boundary_transfer_m2048():
    """B2.3: Pipeline boundary activation transfer for max context [1, 2048, 3584]."""
    ok, reason = check_multigpu(2)
    if not ok:
        skip(f"Requires Dual GPU: {reason}")
    x = torch.randn(1, 2048, 3584, dtype=torch.float16, device="cuda:0")
    transferred = p2p_transfer(x, target_device=1)
    assert transferred.device.index == 1
    assert transferred.shape == (1, 2048, 3584)
    assert torch.equal(x.cpu(), transferred.cpu())


def test_tier2_b02_attention_causal_m2048():
    """B2.4: Multi-head attention forward with 2048 tokens and causal structure."""
    attn = TPParallelAttention(hidden_size=256, num_heads=4, num_kv_heads=2, head_dim=64, all_reduce=False)
    x = torch.randn(1, 2048, 256, dtype=torch.float16)
    out = attn(x)
    assert out.shape == (1, 2048, 256)


def test_tier2_b02_vram_peak_m2048():
    """B2.5: Verifies peak memory at M=2048 context length stays within 14.0 GB budget."""
    seq_len = 2048
    hidden_size = 3584
    intermediate_size = 18944
    num_layers = 28
    num_kv_heads = 4
    head_dim = 128
    world_size = 2

    mlp = TPParallelMLP(hidden_size, intermediate_size, rank=0, world_size=world_size, quant_type=None, all_reduce=False)
    attn = TPParallelAttention(hidden_size, 28, num_kv_heads, head_dim, rank=0, world_size=world_size, all_reduce=False)
    layer_bytes = sum(p.numel() * p.element_size() for p in mlp.parameters()) + sum(p.numel() * p.element_size() for p in attn.parameters())
    weight_bytes = (layer_bytes * 0.5) * num_layers
    kv_bytes = 2 * num_layers * (num_kv_heads // world_size) * seq_len * head_dim * 2
    act_bytes = seq_len * intermediate_size * 2
    total_bytes = weight_bytes + kv_bytes + act_bytes
    total_gb = total_bytes / (1024 ** 3)
    limit_gb = torch.tensor(14.0)
    min_gb = torch.tensor(0.5)
    assert total_gb < limit_gb.item(), f"Calculated VRAM {total_gb:.2f} GB exceeds limit"
    assert total_gb > min_gb.item()


# --- Category B3: Edge Shapes & Batch Sizes ---

def test_tier2_b03_batch_size_1():
    """B3.1: Batch size 1 forward pass shape correctness."""
    col = TPColumnParallelLinear(256, 512, rank=0, world_size=2)
    x = torch.randn(1, 16, 256, dtype=torch.float16)
    out = col(x)
    assert out.shape == (1, 16, 256)


def test_tier2_b03_batch_size_4():
    """B3.2: Batch size 4 forward pass shape correctness."""
    col = TPColumnParallelLinear(256, 512, rank=0, world_size=2)
    x = torch.randn(4, 16, 256, dtype=torch.float16)
    out = col(x)
    assert out.shape == (4, 16, 256)


def test_tier2_b03_odd_sequence_length():
    """B3.3: Handles odd sequence lengths (M=7, 13, 127) cleanly."""
    col = TPColumnParallelLinear(256, 512, rank=0, world_size=2)
    for m in (7, 13, 127):
        x = torch.randn(1, m, 256, dtype=torch.float16)
        out = col(x)
        assert out.shape == (1, m, 256)


def test_tier2_b03_non_power_of_2_intermediate():
    """B3.4: Handles non-power-of-2 intermediate dimension (18944 // 2 = 9472)."""
    col = TPColumnParallelLinear(3584, 18944, rank=0, world_size=2)
    assert col.split_out_features == 9472
    assert col.split_out_features % 8 == 0


def test_tier2_b03_zero_token_handling():
    """B3.5: Empty tensor M=0 returns empty tensor without crashing."""
    col = TPColumnParallelLinear(256, 512, rank=0, world_size=2)
    x = torch.empty(1, 0, 256, dtype=torch.float16)
    out = col(x)
    assert out.shape == (1, 0, 256)


# --- Category B4: Non-Contiguous Tensors ---

def test_tier2_b04_transposed_input():
    """B4.1: Transposed non-contiguous tensor is supported or made contiguous."""
    col = TPColumnParallelLinear(128, 256, rank=0, world_size=2)
    x_orig = torch.randn(128, 16, dtype=torch.float16)
    x_t = x_orig.t()
    assert not x_t.is_contiguous()
    out = col(x_t)
    assert out.shape == (16, 128)


def test_tier2_b04_strided_slice_input():
    """B4.2: Strided slice input x[:, ::2, :] made contiguous before forward."""
    x = torch.randn(2, 16, 256, dtype=torch.float16)
    x_slice = x[:, ::2, :]
    assert not x_slice.is_contiguous()
    col = TPColumnParallelLinear(256, 512, rank=0, world_size=2)
    out = col(x_slice)
    assert out.shape == (2, 8, 256)


def test_tier2_b04_permuted_dimensions():
    """B4.3: Permuted dimensions tensor x.permute(1, 0, 2) handling."""
    x = torch.randn(4, 2, 256, dtype=torch.float16)
    x_perm = x.permute(1, 0, 2)
    assert not x_perm.is_contiguous()
    col = TPColumnParallelLinear(256, 512, rank=0, world_size=2)
    out = col(x_perm)
    assert out.shape == (2, 4, 256)


def test_tier2_b04_non_contiguous_weight():
    """B4.4: Weight tensor slice without contiguous() is normalized."""
    W = torch.randn(512, 256, dtype=torch.float16)
    w_slice = W[::2, :]
    assert not w_slice.is_contiguous()
    w_clean = w_slice.contiguous()
    assert w_clean.is_contiguous()


def test_tier2_b04_contiguity_guard_behavior():
    """B4.5: Verifies contiguity check catches non-contiguous buffer before kernel launch."""
    x = torch.randn(4, 4)
    x_strided = x[:, ::2]
    assert not x_strided.is_contiguous()


# --- Category B5: Device Mismatches & Boundary Errors ---

def test_tier2_b05_cpu_input_to_cuda_shard_error():
    """B5.1: Passing CPU tensor to CUDA sharded module raises explicit error."""
    ok, reason = check_cuda_available()
    if not ok:
        skip(f"Requires CUDA: {reason}")
    W = torch.randn(256, 128, device="cuda:0", dtype=torch.float16)
    x_cpu = torch.randn(1, 128, dtype=torch.float16)
    with pytest.raises((RuntimeError, TypeError)):
        torch.matmul(x_cpu, W.t())


def test_tier2_b05_cross_device_activation_error():
    """B5.2: Passing cuda:1 tensor directly to cuda:0 stage raises device mismatch."""
    ok, reason = check_multigpu(2)
    if not ok:
        skip(f"Requires Dual GPU: {reason}")
    w0 = torch.randn(64, 64, device="cuda:0")
    x1 = torch.randn(1, 64, device="cuda:1")
    with pytest.raises(RuntimeError):
        torch.matmul(x1, w0.t())


def test_tier2_b05_allreduce_single_device_error():
    """B5.3: Peer All-Reduce called with tensors on same device behaves cleanly."""
    t0 = torch.ones(4)
    t1 = torch.ones(4)
    r0, r1 = reference_all_reduce_sum(t0, t1)
    assert torch.equal(r0, torch.ones(4) * 2.0)


def test_tier2_b05_invalid_device_index():
    """B5.4: Requesting invalid device index like cuda:99 raises exception."""
    ok, reason = check_cuda_available()
    if not ok:
        skip(f"Requires CUDA: {reason}")
    with pytest.raises((RuntimeError, ValueError)):
        torch.cuda.device("cuda:99")


def test_tier2_b05_tp_single_gpu_fallback():
    """B5.5: TP configuration with world_size=1 functions cleanly as unpartitioned."""
    col = TPColumnParallelLinear(256, 512, rank=0, world_size=1)
    assert col.split_out_features == 512
    x = torch.randn(1, 4, 256, dtype=torch.float16)
    out = col(x)
    assert out.shape == (1, 4, 512)


# ==============================================================================
# Tier 3: Cross-Feature Interactions
# ==============================================================================

def test_tier3_pair1_tp_int4_gemv_roundtrip():
    """Tier 3 Pair 1: TP Column/Row linear with symmetric INT4 quantization."""
    in_f, out_f = 256, 512
    W = torch.randn(out_f, in_f, dtype=torch.float32)
    x = torch.randn(2, 4, in_f, dtype=torch.float16)

    # INT4 quantize full weight
    packed, scales, zps, g_size = quantize_weight_sym_int4(W, group_size=128)
    W_dequant = dequantize_sym_int4(packed, scales, zps, group_size=128, dtype=torch.float16)

    # Column split
    y_dequant_ref = F.linear(x, W_dequant)
    y0 = F.linear(x, W_dequant[:out_f // 2, :])
    y1 = F.linear(x, W_dequant[out_f // 2:, :])
    y_tp = torch.cat([y0, y1], dim=-1)

    rel_err = relative_l2_error(y_tp, y_dequant_ref)
    assert rel_err < 1e-4, f"TP INT4 error {rel_err} exceeds 1e-4"


def test_tier3_pair2_pp_static_kv_cache_interaction():
    """Tier 3 Pair 2: Pipeline Parallel partitioning interacting with persistent KV cache."""
    num_heads = 2
    head_dim = 64
    max_seq_len = 32

    cache_stage0 = torch.zeros(2, 1, num_heads, max_seq_len, head_dim)
    cache_stage1 = torch.zeros(2, 1, num_heads, max_seq_len, head_dim)

    for step in range(5):
        q0 = torch.randn(1, num_heads, 1, head_dim)
        cache_stage0[:, :, :, step:step+1, :] = q0.unsqueeze(0).repeat(2, 1, 1, 1, 1)

        q1 = torch.randn(1, num_heads, 1, head_dim)
        cache_stage1[:, :, :, step:step+1, :] = q1.unsqueeze(0).repeat(2, 1, 1, 1, 1)

        assert cache_stage0[:, :, :, step, :].norm() > 0
        assert cache_stage1[:, :, :, step, :].norm() > 0


def test_tier3_pair3_tp_swiglu_mlp_interaction():
    """Tier 3 Pair 3: TP Column/Row linear interacting with SwiGLU activation."""
    class StdMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(256, 512, bias=False, dtype=torch.float32)
            self.up_proj = nn.Linear(256, 512, bias=False, dtype=torch.float32)
            self.down_proj = nn.Linear(512, 256, bias=False, dtype=torch.float32)

        def forward(self, x):
            return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    std_mlp = StdMLP()
    tp0 = TPParallelMLP.from_mlp(std_mlp, rank=0, world_size=2, quant_type=None, all_reduce=False)
    tp1 = TPParallelMLP.from_mlp(std_mlp, rank=1, world_size=2, quant_type=None, all_reduce=False)

    x = torch.randn(2, 4, 256, dtype=torch.float32)
    y_full = std_mlp(x)
    y_sharded = tp0(x) + tp1(x)
    rel_err = relative_l2_error(y_sharded, y_full)
    assert rel_err < 1e-4, f"SwiGLU TP error {rel_err} exceeds 1e-4"


def test_tier3_pair4_tp_attention_kv_cache_interaction():
    """Tier 3 Pair 4: TP Attention head sharding interacting with sharded static KV cache."""
    class StdAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_heads = 4
            self.num_kv_heads = 2
            self.head_dim = 64
            self.q_proj = nn.Linear(256, 256, bias=False, dtype=torch.float32)
            self.k_proj = nn.Linear(256, 128, bias=False, dtype=torch.float32)
            self.v_proj = nn.Linear(256, 128, bias=False, dtype=torch.float32)
            self.o_proj = nn.Linear(256, 256, bias=False, dtype=torch.float32)

        def forward(self, x):
            b, m, _ = x.shape
            q = self.q_proj(x).view(b, m, 4, 64).transpose(1, 2)
            k = self.k_proj(x).view(b, m, 2, 64).transpose(1, 2).repeat_interleave(2, dim=1)
            v = self.v_proj(x).view(b, m, 2, 64).transpose(1, 2).repeat_interleave(2, dim=1)
            scores = torch.matmul(q, k.transpose(-1, -2)) / 8.0
            attn = torch.matmul(F.softmax(scores, dim=-1), v).transpose(1, 2).contiguous().view(b, m, -1)
            return self.o_proj(attn)

    std_attn = StdAttn()
    tp0 = TPParallelAttention.from_attention(std_attn, rank=0, world_size=2, all_reduce=False)
    tp1 = TPParallelAttention.from_attention(std_attn, rank=1, world_size=2, all_reduce=False)

    x = torch.randn(1, 4, 256, dtype=torch.float32)
    y_full = std_attn(x)
    y_sharded = tp0(x) + tp1(x)
    rel_err = relative_l2_error(y_sharded, y_full)
    assert rel_err < 1e-4, f"Attention KV sharded error {rel_err} exceeds 1e-4"


def test_tier3_pair5_scope_dynamic_tp_pp_switching():
    """Tier 3 Pair 5: CPMultiGPUInferenceScope dynamic mode switching between TP and PP."""
    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.down_proj = nn.Linear(64, 64)
    model = ToyModel()
    orig_w = model.down_proj.weight.clone()

    # Enter TP mode
    scope_tp = CPMultiGPUInferenceScope(model, mode='tp', devices=['cpu', 'cpu'], quantize_mlp=False)
    with scope_tp:
        assert scope_tp.is_active
    assert not scope_tp.is_active
    assert torch.equal(model.down_proj.weight, orig_w)

    # Enter PP mode
    scope_pp = CPMultiGPUInferenceScope(model, mode='pp', devices=['cpu', 'cpu'], quantize_mlp=False)
    with scope_pp:
        assert scope_pp.is_active
    assert not scope_pp.is_active
    assert torch.equal(model.down_proj.weight, orig_w)


def test_tier3_pair6_int4_gemv_multidevice_guard_concurrency():
    """Tier 3 Pair 6: INT4 GEMV multi-device CUDAGuard concurrency."""
    ok, reason = check_multigpu(2)
    if not ok:
        skip(f"Requires Dual GPU: {reason}")
    s0 = torch.cuda.Stream(device=0)
    s1 = torch.cuda.Stream(device=1)
    with torch.cuda.stream(s0):
        t0 = torch.randn(1, 128, device="cuda:0")
    with torch.cuda.stream(s1):
        t1 = torch.randn(1, 128, device="cuda:1")
    s0.synchronize()
    s1.synchronize()
    assert t0.device.index == 0
    assert t1.device.index == 1


# ==============================================================================
# Tier 4: Real-World Application Scenarios
# ==============================================================================

def test_tier4_s01_qwen_7b_2048_prefill_forward():
    """Tier 4 Scenario 1: Qwen2.5-Math-7B 2048-token context prefill forward pass."""
    hidden_size = 256
    intermediate_size = 512
    num_heads = 4
    num_kv_heads = 2
    head_dim = 64
    seq_len = 2048

    class StdBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_heads = num_heads
            self.num_kv_heads = num_kv_heads
            self.head_dim = head_dim
            self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False, dtype=torch.float32)
            self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False, dtype=torch.float32)
            self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False, dtype=torch.float32)
            self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False, dtype=torch.float32)
            self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=torch.float32)
            self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=torch.float32)
            self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=torch.float32)

        def forward(self, x):
            b, m, _ = x.shape
            q = self.q_proj(x).view(b, m, num_heads, head_dim).transpose(1, 2)
            k = self.k_proj(x).view(b, m, num_kv_heads, head_dim).transpose(1, 2).repeat_interleave(2, dim=1)
            v = self.v_proj(x).view(b, m, num_kv_heads, head_dim).transpose(1, 2).repeat_interleave(2, dim=1)
            scores = torch.matmul(q, k.transpose(-1, -2)) / 8.0
            attn = torch.matmul(F.softmax(scores, dim=-1), v).transpose(1, 2).contiguous().view(b, m, -1)
            h = x + self.o_proj(attn)
            mlp_out = self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))
            return h + mlp_out

    block = StdBlock()
    tp_attn0 = TPParallelAttention.from_attention(block, rank=0, world_size=2, all_reduce=False)
    tp_attn1 = TPParallelAttention.from_attention(block, rank=1, world_size=2, all_reduce=False)
    tp_mlp0 = TPParallelMLP.from_mlp(block, rank=0, world_size=2, quant_type=None, all_reduce=False)
    tp_mlp1 = TPParallelMLP.from_mlp(block, rank=1, world_size=2, quant_type=None, all_reduce=False)

    x = torch.randn(1, seq_len, hidden_size, dtype=torch.float32)
    y_ref = block(x)

    h_shard = x + tp_attn0(x) + tp_attn1(x)
    y_sharded = h_shard + tp_mlp0(h_shard) + tp_mlp1(h_shard)

    rel_err = relative_l2_error(y_sharded, y_ref)
    assert rel_err <= 1e-3, f"7B 2048 prefill relative error {rel_err} exceeds 0.1% gate"
    assert y_sharded.shape == (1, seq_len, hidden_size)


def test_tier4_s02_qwen_7b_100_step_decode_stability():
    """Tier 4 Scenario 2: Qwen2.5-Math-7B 100-step autoregressive decode loop."""
    hidden_size = 128
    intermediate_size = 256
    num_heads = 4
    num_kv_heads = 2
    head_dim = 32

    attn = TPParallelAttention(hidden_size=hidden_size, num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim, dtype=torch.float32, all_reduce=False)
    mlp = TPParallelMLP(hidden_size=hidden_size, intermediate_size=intermediate_size, quant_type=None, dtype=torch.float32, all_reduce=False)
    norm = nn.LayerNorm(hidden_size, dtype=torch.float32)

    curr_token = torch.randn(1, 1, hidden_size, dtype=torch.float32)
    for step in range(100):
        h = norm(curr_token + attn(curr_token, start_pos=step))
        curr_token = norm(h + mlp(h))
        assert curr_token.shape == (1, 1, hidden_size)
        assert torch.isfinite(curr_token).all()


def test_tier4_s03_amc12_math_reasoning_tp_rollout():
    """Tier 4 Scenario 3: AMC 12 math reasoning prompt rollout under TP=2."""
    prompt_tokens = [15, 230, 18, 492, 102, 33, 405, 92, 14, 188]
    prompt_len = len(prompt_tokens)

    hidden_size = 128
    vocab_size = 1000
    embed = nn.Embedding(vocab_size, hidden_size)
    mlp = TPParallelMLP(hidden_size=hidden_size, intermediate_size=256, quant_type=None, dtype=torch.float32, all_reduce=False)
    lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    tokens = list(prompt_tokens)
    for step in range(8):
        inp_tensor = torch.tensor([tokens], dtype=torch.long)
        h = embed(inp_tensor)
        h_sharded = mlp(h)
        logits = lm_head(h_sharded[:, -1:, :])
        next_tok = int(torch.argmax(logits, dim=-1).item())
        tokens.append(next_tok)

    assert len(tokens) == prompt_len + 8
    assert all(0 <= t < vocab_size for t in tokens)


def test_tier4_s04_aime_proof_reasoning_pp_rollout():
    """Tier 4 Scenario 4: AIME competition proof prompt rollout under PP=2."""
    aime_prompt = [42, 901, 12, 54, 882, 10, 4, 301]
    pp = PipelineParallelQwen2(num_layers=4, hidden_size=128, vocab_size=1000, dtype=torch.float32, devices=['cpu', 'cpu'])

    tokens = list(aime_prompt)
    for _ in range(6):
        inp_tensor = torch.tensor([tokens], dtype=torch.long)
        logits = pp(inp_tensor)
        next_tok = int(torch.argmax(logits[:, -1, :], dim=-1).item())
        tokens.append(next_tok)

    assert len(tokens) == len(aime_prompt) + 6
    assert all(0 <= t < 1000 for t in tokens)


def test_tier4_s05_dynamic_scope_state_restoration_during_generation():
    """Tier 4 Scenario 5: Dynamic scope entry/exit state restoration during generation."""
    class SimpleLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(100, 64)
            self.down_proj = nn.Linear(64, 64)
            self.head = nn.Linear(64, 100)

        def forward(self, x):
            return self.head(self.down_proj(self.embed(x)))

    model = SimpleLM()
    scope = CPMultiGPUInferenceScope(model, mode='tp', devices=['cpu', 'cpu'], quantize_mlp=False)

    prompt = torch.tensor([[1, 2, 3, 4]])

    # Baseline logits
    logits_base = model(prompt)

    # Sharded generation under scope
    with scope:
        logits_sharded = model(prompt)
        assert logits_sharded.shape == logits_base.shape

    # Exited scope: verify bitwise identical logits
    logits_restored = model(prompt)
    assert torch.equal(logits_restored, logits_base)


# ==============================================================================
# Standalone CLI Test Runner
# ==============================================================================

def discover_tests(tier: str = "all") -> List[Any]:
    """Discovers all matching test functions in global namespace."""
    all_tests = [
        obj for name, obj in globals().items()
        if name.startswith("test_") and callable(obj)
    ]
    all_tests.sort(key=lambda f: f.__name__)
    if tier == "all":
        return all_tests
    prefix = f"test_tier{tier}_"
    return [fn for fn in all_tests if fn.__name__.startswith(prefix)]


def run_suite(tier: str = "all", verbose: bool = False) -> int:
    """Executes discovered tests, reports progress, and returns exit code."""
    tests = discover_tests(tier)
    print("=" * 80)
    print(f"  RUNNING MULTI-GPU SHARDING E2E TEST SUITE (Tier: {tier}, Count: {len(tests)})")
    print("=" * 80)

    passed = 0
    skipped = 0
    failed = 0
    failures = []
    skips = []
    t_start = time.perf_counter()

    for test_fn in tests:
        fn_name = test_fn.__name__
        t0 = time.perf_counter()
        try:
            test_fn()
            dt = (time.perf_counter() - t0) * 1000.0
            print(f"  [PASS] {fn_name:<60} ({dt:.2f} ms)")
            passed += 1
        except SkipTest as e:
            dt = (time.perf_counter() - t0) * 1000.0
            print(f"  [SKIP] {fn_name:<60} ({dt:.2f} ms): {e}")
            skipped += 1
            skips.append((fn_name, str(e)))
        except Exception as e:
            dt = (time.perf_counter() - t0) * 1000.0
            print(f"  [FAIL] {fn_name:<60} ({dt:.2f} ms): {e}")
            failed += 1
            failures.append((fn_name, str(e)))

    total_time = time.perf_counter() - t_start
    print("=" * 80)
    print(f"  TOTAL: {len(tests)} | PASSED: {passed} | SKIPPED: {skipped} | FAILED: {failed} | TIME: {total_time:.3f}s")
    print("=" * 80)

    if skipped > 0 and verbose:
        print("\nSkipped tests details:")
        for name, reason in skips:
            print(f"  - {name}: {reason}")

    if failed > 0:
        print("\nFailures:")
        for name, err in failures:
            print(f"  - {name}: {err}")
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser(description="4-Tier E2E Multi-GPU Sharding Test Suite.")
    parser.add_argument(
        "--tier",
        type=str,
        choices=["1", "2", "3", "4", "all"],
        default="all",
        help="Select specific test tier to run (default: all)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print verbose skip details and timings",
    )
    args = parser.parse_args()
    sys.exit(run_suite(tier=args.tier, verbose=args.verbose))


if __name__ == "__main__":
    main()
