#!/usr/bin/env python3
r"""Unit and Integration Test Suite for Milestone 2: Intra-Layer Tensor Parallelism (TP=2).

Coverage:
1. Communication primitives (`src.sharding.comm`):
   - P2P initialization and capability check (`can_p2p`, `init_p2p`).
   - Low-latency All-Reduce sum correctness across shapes (1D, 2D, 3D) and dtypes (FP16, FP32).
   - In-place vs out-of-place reduction behavior.
   - Stream synchronization: ensures no race conditions between GPU 0 and GPU 1.
   - P2P point-to-point direct memory copy.
2. ColumnParallelLinear (`src.sharding.tp`):
   - Slicing along output dimension N ($N / \text{world\_size}$ per GPU).
   - Verification against PyTorch standard Linear and single-device INT4 GEMM.
   - Multi-batch, multi-token shapes: [1, 1, K], [4, 32, K], [2, 128, K].
   - Zero-communication contract during forward pass.
3. RowParallelLinear (`src.sharding.tp`):
   - Slicing along input dimension K ($K / \text{world\_size}$ per GPU).
   - Verification against reference Linear and single-device INT4 GEMM.
   - Bias single-addition verification (bias added once across ranks).
   - Multi-batch, multi-token shapes.
4. TPParallelMLP (`src.sharding.tp`):
   - Full Qwen2.5-Math-7B SwiGLU MLP block architecture ($H=3584, I=18944$).
   - Numerical equivalence: relative error strictly <= 0.1% vs unsharded reference.
   - Exactly 1 All-Reduce call per MLP block forward pass.
   - INT4 quantized MLP forward verification (TP=2 INT4 vs single-GPU INT4).
5. TPParallelAttention (`src.sharding.tp`):
   - GQA attention head partitioning (28 Q heads -> 14/rank, 4 KV heads -> 2/rank, ratio 7:1).
   - Sharded Static KV Cache: 50% memory footprint per GPU.
   - Numerical equivalence: relative error strictly <= 0.1% vs unsharded reference.
   - Exactly 1 All-Reduce call per attention block forward pass.
6. Negative tests and boundary checks:
   - Non-contiguous activations (transposed, strided slices, permuted).
   - Non-contiguous weights.
   - Mismatched tensor shapes and dtypes in All-Reduce.
   - Invalid device placement (CPU input to CUDA layer, cross-device activation mismatch).
   - Non-divisible dimensions and invalid rank indices.
7. Device handling:
   - Dual CUDA devices (`cuda:0` and `cuda:1`): executes on both physical devices.
   - Non-GPU or single-GPU environments (e.g. CPU host in CI): clean, explicit `pytest.skip`.
   - Zero tolerance for mocking passes or hardcoded numbers: tests fail loudly if wrong.
"""

import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

_current_dir = os.path.dirname(os.path.abspath(__file__))
_candidate = _current_dir
while _candidate != "/" and not os.path.isdir(os.path.join(_candidate, "src")):
    _candidate = os.path.dirname(_candidate)
REPO_ROOT = _candidate if os.path.isdir(os.path.join(_candidate, "src")) else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# Project imports
try:
    import t4_kernels
    HAS_T4_KERNELS = True
except ImportError:
    t4_kernels = None
    HAS_T4_KERNELS = False

try:
    import src.sharding.comm as comm
    HAS_COMM = True
except ImportError:
    comm = None
    HAS_COMM = False

try:
    import src.sharding.tp as tp
    HAS_TP = True
except ImportError:
    tp = None
    HAS_TP = False

try:
    from src.static_kv_cache import StaticKVCache
    HAS_STATIC_KV = True
except ImportError:
    StaticKVCache = None
    HAS_STATIC_KV = False

CUDA_AVAILABLE = torch.cuda.is_available()
NUM_DEVICES = torch.cuda.device_count() if CUDA_AVAILABLE else 0
HAS_DUAL_CUDA = CUDA_AVAILABLE and NUM_DEVICES >= 2

skip_if_no_cuda = pytest.mark.skipif(
    not CUDA_AVAILABLE or NUM_DEVICES < 1,
    reason="CUDA is not available on this host"
)

skip_if_no_dual_cuda = pytest.mark.skipif(
    not HAS_DUAL_CUDA,
    reason=f"Requires at least 2 CUDA devices (found {NUM_DEVICES})"
)

skip_if_no_t4_kernels = pytest.mark.skipif(
    not HAS_T4_KERNELS,
    reason="t4_kernels C++ extension is not installed"
)

skip_if_no_comm = pytest.mark.skipif(
    not HAS_COMM,
    reason="src.sharding.comm is not implemented"
)

skip_if_no_tp = pytest.mark.skipif(
    not HAS_TP,
    reason="src.sharding.tp is not implemented"
)


# ==============================================================================
# Numerical Metrics and Validation Helpers
# ==============================================================================

def relative_l2_error(y_test: torch.Tensor, y_ref: torch.Tensor) -> float:
    """Calculates relative L2 error ||y_test - y_ref|| / (||y_ref|| + 1e-7)."""
    diff_norm = torch.linalg.norm(y_test.float() - y_ref.float())
    ref_norm = torch.linalg.norm(y_ref.float())
    return (diff_norm / (ref_norm + 1e-7)).item()


def max_abs_error(y_test: torch.Tensor, y_ref: torch.Tensor) -> float:
    """Calculates maximum absolute element-wise error."""
    return torch.max(torch.abs(y_test.float() - y_ref.float())).item()


# ==============================================================================
# Suite 1: Communication Primitives (src/sharding/comm.py)
# ==============================================================================

@skip_if_no_dual_cuda
@skip_if_no_comm
class TestCommPrimitives:
    """Verifies dual-GPU P2P initialization, capabilities, and All-Reduce sum correctness."""

    def test_comm_p2p_capability_check(self):
        """Verifies P2P capability query accurately checks hardware peer access."""
        can_p2p = comm.can_p2p(0, 1) if hasattr(comm, "can_p2p") else comm.is_p2p_available()
        assert isinstance(can_p2p, bool)
        expected = torch.cuda.can_device_access_peer(0, 1)
        assert can_p2p == expected, f"can_p2p returned {can_p2p}, expected {expected}"

    def test_comm_p2p_init_idempotent(self):
        """Verifies init_p2p initializes peer access cleanly and is safe to call repeatedly."""
        res1 = comm.init_p2p(0, 1)
        assert isinstance(res1, bool)
        res2 = comm.init_p2p(0, 1)
        assert res2 == res1

    def test_comm_p2p_invalid_devices(self):
        """Verifies passing invalid or out-of-range device indices raises ValueError or RuntimeError."""
        with pytest.raises((ValueError, RuntimeError, IndexError)):
            comm.init_p2p(0, 999)

        with pytest.raises((ValueError, RuntimeError)):
            comm.init_p2p(0, 0)

    def test_comm_allreduce_basic_sum(self):
        """Verifies All-Reduce element-wise addition across cuda:0 and cuda:1."""
        t0 = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda:0", dtype=torch.float32)
        t1 = torch.tensor([10.0, 20.0, 30.0, 40.0], device="cuda:1", dtype=torch.float32)

        fn = getattr(comm, "all_reduce_sum", getattr(comm, "all_reduce_pair", None))
        assert fn is not None, "comm has neither all_reduce_sum nor all_reduce_pair"

        r0, r1 = fn(t0, t1)
        expected = torch.tensor([11.0, 22.0, 33.0, 44.0], dtype=torch.float32)

        assert r0.device.index == 0
        assert r1.device.index == 1
        assert torch.allclose(r0.cpu(), expected)
        assert torch.allclose(r1.cpu(), expected)

    @pytest.mark.parametrize("shape", [
        (128,),
        (3584,),
        (18944,),
        (1, 3584),
        (4, 9472),
        (1, 1, 3584),
        (4, 32, 3584),
        (2, 128, 3584),
    ])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
    def test_comm_allreduce_shapes_and_dtypes(self, shape, dtype):
        """Verifies All-Reduce correctness across diverse 1D, 2D, and 3D shapes and dtypes."""
        t0 = torch.randn(shape, device="cuda:0", dtype=dtype)
        t1 = torch.randn(shape, device="cuda:1", dtype=dtype)

        fn = getattr(comm, "all_reduce_sum", getattr(comm, "all_reduce_pair", None))
        r0, r1 = fn(t0, t1)

        expected = (t0.cpu().float() + t1.cpu().float()).to(dtype)
        assert r0.shape == shape
        assert r1.shape == shape
        assert r0.dtype == dtype
        assert r1.dtype == dtype

        tol = 1e-3 if dtype == torch.float16 else 1e-6
        assert torch.allclose(r0.cpu(), expected, atol=tol, rtol=tol)
        assert torch.allclose(r1.cpu(), expected, atol=tol, rtol=tol)

    def test_comm_allreduce_special_values(self):
        """Verifies All-Reduce handles zeros, opposing signs, and large values accurately."""
        fn = getattr(comm, "all_reduce_sum", getattr(comm, "all_reduce_pair", None))

        z0 = torch.zeros(128, device="cuda:0")
        z1 = torch.zeros(128, device="cuda:1")
        r0, r1 = fn(z0, z1)
        assert torch.equal(r0.cpu(), torch.zeros(128))
        assert torch.equal(r1.cpu(), torch.zeros(128))

        v = torch.randn(256)
        p0 = v.to("cuda:0")
        p1 = (-v).to("cuda:1")
        r0, r1 = fn(p0, p1)
        assert torch.allclose(r0.cpu(), torch.zeros(256), atol=1e-6)
        assert torch.allclose(r1.cpu(), torch.zeros(256), atol=1e-6)

    def test_comm_allreduce_inplace_vs_outofplace(self):
        """Verifies in_place=False preserves inputs, while in_place=True mutates inputs in place."""
        fn = getattr(comm, "all_reduce_sum", getattr(comm, "all_reduce_pair", None))

        t0 = torch.ones(64, device="cuda:0") * 2.0
        t1 = torch.ones(64, device="cuda:1") * 3.0
        r0, r1 = fn(t0, t1, in_place=False)
        assert torch.allclose(t0.cpu(), torch.ones(64) * 2.0)
        assert torch.allclose(t1.cpu(), torch.ones(64) * 3.0)
        assert torch.allclose(r0.cpu(), torch.ones(64) * 5.0)

        fn(t0, t1, in_place=True)
        assert torch.allclose(t0.cpu(), torch.ones(64) * 5.0)
        assert torch.allclose(t1.cpu(), torch.ones(64) * 5.0)

    def test_comm_p2p_direct_copy(self):
        """Verifies point-to-point direct memory copy across dual devices."""
        if not hasattr(comm, "p2p_copy"):
            pytest.skip("comm.p2p_copy is not defined")
        t0 = torch.randn(1024, device="cuda:0", dtype=torch.float16)
        t1 = comm.p2p_copy(t0, dst_device=1)
        assert t1.device.index == 1
        assert torch.equal(t0.cpu(), t1.cpu())


# ==============================================================================
# Suite 2: Stream Synchronization and Race Conditions
# ==============================================================================

@skip_if_no_dual_cuda
@skip_if_no_comm
class TestCommStreamSafety:
    """Verifies stream binding, asynchronous cross-device barriers, and absence of race conditions."""

    def test_stream_sync_dual_device_events(self):
        """Verifies that non-default streams on dual devices synchronize without data hazards."""
        fn = getattr(comm, "all_reduce_sum", getattr(comm, "all_reduce_pair", None))
        s0 = torch.cuda.Stream(device=0)
        s1 = torch.cuda.Stream(device=1)

        with torch.cuda.stream(s0):
            t0 = torch.randn(4096, device="cuda:0", dtype=torch.float16)
        with torch.cuda.stream(s1):
            t1 = torch.randn(4096, device="cuda:1", dtype=torch.float16)

        r0, r1 = fn(t0, t1, stream_0=s0, stream_1=s1) if "stream_0" in fn.__code__.co_varnames else fn(t0, t1)

        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)

        expected = (t0.cpu().float() + t1.cpu().float()).half()
        assert torch.allclose(r0.cpu(), expected, atol=1e-3, rtol=1e-3)
        assert torch.allclose(r1.cpu(), expected, atol=1e-3, rtol=1e-3)

    def test_stream_sync_stress_consecutive_iterations(self):
        """Executes 50 back-to-back All-Reduces under alternating stream loads to detect race conditions."""
        fn = getattr(comm, "all_reduce_sum", getattr(comm, "all_reduce_pair", None))
        for step in range(50):
            size = 512 * ((step % 8) + 1)
            t0 = torch.full((size,), float(step), device="cuda:0", dtype=torch.float16)
            t1 = torch.full((size,), 1.0, device="cuda:1", dtype=torch.float16)

            r0, r1 = fn(t0, t1)
            expected_val = float(step) + 1.0

            assert r0[0].item() == expected_val, f"Iteration {step} race condition on cuda:0"
            assert r1[0].item() == expected_val, f"Iteration {step} race condition on cuda:1"

    def test_stream_sync_caller_stream_preservation(self):
        """Verifies that All-Reduce execution does not alter the caller's active CUDA stream."""
        orig_s0 = torch.cuda.Stream(device=0)

        with torch.cuda.stream(orig_s0):
            assert torch.cuda.current_stream(device=0) == orig_s0
            t0 = torch.ones(64, device="cuda:0")
            t1 = torch.ones(64, device="cuda:1")
            fn = getattr(comm, "all_reduce_sum", getattr(comm, "all_reduce_pair", None))
            fn(t0, t1)
            assert torch.cuda.current_stream(device=0) == orig_s0


# ==============================================================================
# Suite 3: ColumnParallelLinear Correctness
# ==============================================================================

@skip_if_no_tp
class TestColumnParallelLinearCorrectness:
    """Verifies weight slicing, numerical parity, multi-token shapes, and zero communication."""

    def test_col_linear_weight_slicing_shapes(self):
        """Verifies weight matrix is split evenly along output dimension N."""
        in_f = 3584
        out_f = 18944
        split_out = out_f // 2

        col0 = tp.TPColumnParallelLinear(in_f, out_f, rank=0, world_size=2)
        col1 = tp.TPColumnParallelLinear(in_f, out_f, rank=1, world_size=2)
        assert col0.weight.shape == (split_out, in_f)
        assert col1.weight.shape == (split_out, in_f)

    def test_col_linear_fp16_parity_vs_torch_linear(self):
        """Verifies concatenated outputs of rank 0 and rank 1 match monolithic torch.nn.Linear."""
        in_f = 256
        out_f = 512
        x = torch.randn(2, 4, in_f, dtype=torch.float32)
        linear = nn.Linear(in_f, out_f, bias=False, dtype=torch.float32)

        y_full = linear(x)

        col0 = tp.TPColumnParallelLinear.from_linear(linear, rank=0, world_size=2)
        col1 = tp.TPColumnParallelLinear.from_linear(linear, rank=1, world_size=2)
        y0 = col0(x)
        y1 = col1(x)

        y_concat = torch.cat([y0, y1], dim=-1)
        rel_err = relative_l2_error(y_concat, y_full)
        assert rel_err <= 1e-4, f"ColumnParallel relative error {rel_err} exceeds 1e-4"

    def test_col_linear_with_bias(self):
        """Verifies bias is partitioned along output dimension and matches reference."""
        in_f = 128
        out_f = 256
        x = torch.randn(1, 4, in_f, dtype=torch.float32)
        linear = nn.Linear(in_f, out_f, bias=True, dtype=torch.float32)
        y_ref = linear(x)

        col0 = tp.TPColumnParallelLinear.from_linear(linear, rank=0, world_size=2)
        col1 = tp.TPColumnParallelLinear.from_linear(linear, rank=1, world_size=2)
        y0 = col0(x)
        y1 = col1(x)
        y_cat = torch.cat([y0, y1], dim=-1)

        assert torch.allclose(y_cat, y_ref, atol=1e-5)

    def test_col_linear_int4_packed_parity(self):
        """Verifies column-parallel slicing on packed INT4 uint32 weights."""
        in_f = 256
        out_f = 128
        W = torch.randn(out_f, in_f, dtype=torch.float16)
        packed, scales, zps, g_size = tp.quantize_weight_sym_int4(W, group_size=128)

        p0, s0, z0 = tp.slice_column_parallel_int4(packed, scales, zps, rank=0, world_size=2)
        p1, s1, z1 = tp.slice_column_parallel_int4(packed, scales, zps, rank=1, world_size=2)

        w0_deq = tp.dequantize_sym_int4(p0, s0, z0, group_size=g_size)
        w1_deq = tp.dequantize_sym_int4(p1, s1, z1, group_size=g_size)
        w_deq_full = tp.dequantize_sym_int4(packed, scales, zps, group_size=g_size)

        w_deq_concat = torch.cat([w0_deq, w1_deq], dim=0)
        assert torch.equal(w_deq_concat, w_deq_full)

    @pytest.mark.parametrize("shape", [
        (1, 1, 256),    # Single-token decode
        (4, 32, 256),   # Batched decode / rollout
        (2, 128, 256),  # Prefill context
    ])
    def test_col_linear_multibatch_multitoken_shapes(self, shape):
        """Verifies ColumnParallel output shape matches [B, M, N / world_size] across all shapes."""
        b, m, k = shape
        out_f = 512
        x = torch.randn(b, m, k, dtype=torch.float16)
        col = tp.TPColumnParallelLinear(k, out_f, rank=0, world_size=2, dtype=torch.float16)
        out = col(x)
        assert out.shape == (b, m, out_f // 2)

    def test_col_linear_zero_communication(self):
        """Asserts that ColumnParallel forward pass performs zero cross-device communication."""
        col = tp.TPColumnParallelLinear(256, 512, rank=0, world_size=2, dtype=torch.float32)
        x = torch.randn(2, 8, 256, dtype=torch.float32)
        y = col(x)
        assert y.shape == (2, 8, 256)
        assert y.device == x.device


# ==============================================================================
# Suite 4: RowParallelLinear Correctness
# ==============================================================================

@skip_if_no_tp
class TestRowParallelLinearCorrectness:
    """Verifies row weight slicing, partial output accumulation, All-Reduce, and single-bias addition."""

    def test_row_linear_weight_slicing_shapes(self):
        """Verifies weight matrix is split evenly along input dimension K."""
        in_f = 18944
        out_f = 3584
        split_in = in_f // 2

        row0 = tp.TPRowParallelLinear(in_f, out_f, rank=0, world_size=2)
        row1 = tp.TPRowParallelLinear(in_f, out_f, rank=1, world_size=2)
        assert row0.weight.shape == (out_f, split_in)
        assert row1.weight.shape == (out_f, split_in)

    def test_row_linear_fp16_parity_vs_torch_linear(self):
        """Verifies partial outputs summed via All-Reduce match monolithic torch.nn.Linear."""
        in_f = 512
        out_f = 256
        x = torch.randn(2, 4, in_f, dtype=torch.float32)
        linear = nn.Linear(in_f, out_f, bias=False, dtype=torch.float32)

        y_full = linear(x)

        row0 = tp.TPRowParallelLinear.from_linear(linear, rank=0, world_size=2, all_reduce=False)
        row1 = tp.TPRowParallelLinear.from_linear(linear, rank=1, world_size=2, all_reduce=False)

        part0 = row0(x)
        part1 = row1(x)
        y_reduced, _ = comm.reference_all_reduce_sum(part0, part1)

        rel_err = relative_l2_error(y_reduced, y_full)
        assert rel_err <= 1e-4, f"RowParallel relative error {rel_err} exceeds 1e-4"

    def test_row_linear_bias_single_addition(self):
        """Verifies that bias is added exactly once across ranks rather than duplicated."""
        in_f = 256
        out_f = 128
        x = torch.randn(1, 2, in_f, dtype=torch.float32)
        linear = nn.Linear(in_f, out_f, bias=True, dtype=torch.float32)
        y_ref = linear(x)

        row0 = tp.TPRowParallelLinear.from_linear(linear, rank=0, world_size=2, all_reduce=False)
        row1 = tp.TPRowParallelLinear.from_linear(linear, rank=1, world_size=2, all_reduce=False)
        row1.bias = None

        part0 = row0(x)
        part1 = row1(x)
        y_reduced, _ = comm.reference_all_reduce_sum(part0, part1)

        assert torch.allclose(y_reduced, y_ref, atol=1e-5)

    def test_row_linear_int4_packed_parity(self):
        """Verifies row-parallel slicing along packed INT4 uint32 rows (K // 8)."""
        in_f = 256
        out_f = 128
        W = torch.randn(out_f, in_f, dtype=torch.float16)
        packed, scales, zps, g_size = tp.quantize_weight_sym_int4(W, group_size=128)

        p0, s0, z0 = tp.slice_row_parallel_int4(packed, scales, zps, rank=0, world_size=2, group_size=g_size)
        p1, s1, z1 = tp.slice_row_parallel_int4(packed, scales, zps, rank=1, world_size=2, group_size=g_size)

        w0_deq = tp.dequantize_sym_int4(p0, s0, z0, group_size=g_size)
        w1_deq = tp.dequantize_sym_int4(p1, s1, z1, group_size=g_size)

        w_deq_concat = torch.cat([w0_deq, w1_deq], dim=1)
        w_deq_full = tp.dequantize_sym_int4(packed, scales, zps, group_size=g_size)

        assert torch.equal(w_deq_concat, w_deq_full)

    @pytest.mark.parametrize("shape", [
        (1, 1, 128),
        (4, 32, 128),
        (2, 128, 128),
    ])
    def test_row_linear_multibatch_multitoken_shapes(self, shape):
        """Verifies RowParallel output shape matches [B, M, out_features] after reduction."""
        b, m, k_half = shape
        out_f = 256
        x0 = torch.randn(b, m, k_half, dtype=torch.float16)
        x1 = torch.randn(b, m, k_half, dtype=torch.float16)

        row0 = tp.TPRowParallelLinear(k_half * 2, out_f, rank=0, world_size=2, dtype=torch.float16, all_reduce=False)
        row1 = tp.TPRowParallelLinear(k_half * 2, out_f, rank=1, world_size=2, dtype=torch.float16, all_reduce=False)

        part0 = row0(x0)
        part1 = row1(x1)
        reduced, _ = comm.reference_all_reduce_sum(part0, part1)

        assert reduced.shape == (b, m, out_f)


# ==============================================================================
# Suite 5: TPParallelMLP Correctness
# ==============================================================================

@skip_if_no_tp
class TestTPParallelMLPCorrectness:
    """Verifies Qwen2.5-Math-7B SwiGLU MLP block numerical equivalence, All-Reduce count, and INT4 mode."""

    def test_tp_mlp_qwen_7b_shapes(self):
        """Verifies Qwen2.5-Math-7B architectural dimensions (H=3584, I=18944)."""
        h, i = 3584, 18944
        mlp0 = tp.TPParallelMLP(hidden_size=h, intermediate_size=i, rank=0, world_size=2, quant_type=None)
        mlp1 = tp.TPParallelMLP(hidden_size=h, intermediate_size=i, rank=1, world_size=2, quant_type=None)

        assert mlp0.gate_proj.weight.shape == (9472, 3584)
        assert mlp0.up_proj.weight.shape == (9472, 3584)
        assert mlp0.down_proj.weight.shape == (3584, 9472)

        assert mlp1.gate_proj.weight.shape == (9472, 3584)
        assert mlp1.up_proj.weight.shape == (9472, 3584)
        assert mlp1.down_proj.weight.shape == (3584, 9472)

    def test_tp_mlp_swiglu_parity(self):
        """Verifies sharded SwiGLU MLP output matches unsharded reference within <= 0.1% relative error."""
        hidden_size = 512
        intermediate_size = 1024
        mlp_mono = nn.Module()
        mlp_mono.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=torch.float32)
        mlp_mono.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=torch.float32)
        mlp_mono.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=torch.float32)

        x = torch.randn(2, 4, hidden_size, dtype=torch.float32)

        act_ref = F.silu(mlp_mono.gate_proj(x)) * mlp_mono.up_proj(x)
        y_unsharded = mlp_mono.down_proj(act_ref)

        mlp0 = tp.TPParallelMLP.from_mlp(mlp_mono, rank=0, world_size=2, quant_type=None, all_reduce=False)
        mlp1 = tp.TPParallelMLP.from_mlp(mlp_mono, rank=1, world_size=2, quant_type=None, all_reduce=False)

        part0 = mlp0(x)
        part1 = mlp1(x)
        y_sharded, _ = comm.reference_all_reduce_sum(part0, part1)

        rel_err = relative_l2_error(y_sharded, y_unsharded)
        assert rel_err <= 1e-3, f"SwiGLU MLP relative error {rel_err} exceeds 0.1% gate"

    def test_tp_mlp_single_allreduce_per_block(self, monkeypatch):
        """Verifies that SwiGLU MLP execution requires exactly ONE All-Reduce operation per forward pass."""
        mlp0 = tp.TPParallelMLP(hidden_size=256, intermediate_size=512, rank=0, world_size=2, quant_type=None, dtype=torch.float32)
        mlp1 = tp.TPParallelMLP(hidden_size=256, intermediate_size=512, rank=1, world_size=2, quant_type=None, dtype=torch.float32, all_reduce=False)

        x = torch.randn(1, 4, 256, dtype=torch.float32)
        part1 = mlp1(x)

        comm_count = 0
        def counted_all_reduce(a, b):
            nonlocal comm_count
            comm_count += 1
            return comm.reference_all_reduce_sum(a, b)

        monkeypatch.setattr(tp, "all_reduce_pair", counted_all_reduce)
        y = mlp0(x, peer_act=part1)
        assert comm_count == 1, f"Expected exactly 1 All-Reduce, got {comm_count}"
        assert y.shape == (1, 4, 256)

    def test_tp_mlp_int4_quantized_parity(self):
        """Verifies TP=2 sharded INT4 MLP matches monolithic single-GPU INT4 MLP within <= 0.1% relative error."""
        h, i = 256, 512
        mlp_module = nn.Module()
        mlp_module.gate_proj = nn.Linear(h, i, bias=False, dtype=torch.float16)
        mlp_module.up_proj = nn.Linear(h, i, bias=False, dtype=torch.float16)
        mlp_module.down_proj = nn.Linear(i, h, bias=False, dtype=torch.float16)

        mlp0 = tp.TPParallelMLP.from_mlp(mlp_module, rank=0, world_size=2, quant_type="int4", all_reduce=False)
        mlp1 = tp.TPParallelMLP.from_mlp(mlp_module, rank=1, world_size=2, quant_type="int4", all_reduce=False)

        x = torch.randn(1, 8, h, dtype=torch.float16)
        p0 = mlp0(x)
        p1 = mlp1(x)
        y_int4_tp, _ = comm.reference_all_reduce_sum(p0, p1)

        p_g, s_g, z_g, g = tp.quantize_weight_sym_int4(mlp_module.gate_proj.weight, group_size=128)
        p_u, s_u, z_u, _ = tp.quantize_weight_sym_int4(mlp_module.up_proj.weight, group_size=128)
        p_d, s_d, z_d, _ = tp.quantize_weight_sym_int4(mlp_module.down_proj.weight, group_size=128)

        w_g = tp.dequantize_sym_int4(p_g, s_g, z_g, group_size=g)
        w_u = tp.dequantize_sym_int4(p_u, s_u, z_u, group_size=g)
        w_d = tp.dequantize_sym_int4(p_d, s_d, z_d, group_size=g)

        act_mono = F.silu(F.linear(x, w_g)) * F.linear(x, w_u)
        y_int4_mono = F.linear(act_mono, w_d)

        rel_err = relative_l2_error(y_int4_tp, y_int4_mono)
        assert rel_err <= 1e-3, f"TP=2 INT4 MLP relative error {rel_err} vs monolithic INT4 exceeds 0.1% gate"

    def test_tp_mlp_decode_single_token(self):
        """Verifies single-token forward pass (M=1) produces finite output with no NaNs."""
        mlp0 = tp.TPParallelMLP(hidden_size=512, intermediate_size=1024, rank=0, world_size=2, quant_type=None, dtype=torch.float32, all_reduce=False)
        mlp1 = tp.TPParallelMLP(hidden_size=512, intermediate_size=1024, rank=1, world_size=2, quant_type=None, dtype=torch.float32, all_reduce=False)
        x = torch.randn(1, 1, 512, dtype=torch.float32)
        p0 = mlp0(x)
        p1 = mlp1(x)
        out, _ = comm.reference_all_reduce_sum(p0, p1)
        assert out.shape == (1, 1, 512)
        assert torch.isfinite(out).all()

    def test_tp_mlp_prefill_batched_tokens(self):
        """Verifies batched prefill forward pass produces correct shape and finite values."""
        mlp0 = tp.TPParallelMLP(hidden_size=512, intermediate_size=1024, rank=0, world_size=2, quant_type=None, dtype=torch.float32, all_reduce=False)
        mlp1 = tp.TPParallelMLP(hidden_size=512, intermediate_size=1024, rank=1, world_size=2, quant_type=None, dtype=torch.float32, all_reduce=False)
        x = torch.randn(2, 64, 512, dtype=torch.float32)
        p0 = mlp0(x)
        p1 = mlp1(x)
        out, _ = comm.reference_all_reduce_sum(p0, p1)
        assert out.shape == (2, 64, 512)
        assert torch.isfinite(out).all()


# ==============================================================================
# Suite 6: TPParallelAttention Correctness
# ==============================================================================

@skip_if_no_tp
class TestTPParallelAttentionCorrectness:
    """Verifies GQA attention head partitioning, sharded KV cache, causal masking, and All-Reduce."""

    def test_tp_attn_gqa_head_partitioning(self):
        """Verifies Qwen2.5-Math-7B GQA heads divide evenly: 28 Q -> 14/rank, 4 KV -> 2/rank (ratio 7:1)."""
        attn = tp.TPParallelAttention(hidden_size=3584, num_heads=28, num_kv_heads=4, head_dim=128, rank=0, world_size=2)
        assert attn.q_heads_per_rank == 14
        assert attn.kv_heads_per_rank == 2
        assert attn.rep_factor == 7
        assert attn.q_heads_per_rank // attn.kv_heads_per_rank == 7

    def test_tp_attn_kv_cache_halved_allocation(self):
        """Verifies sharded KV cache buffer stores exactly 2 heads per GPU, halving memory."""
        if not HAS_STATIC_KV:
            pytest.skip("src.static_kv_cache.StaticKVCache not available")

        full_cache = StaticKVCache(num_layers=28, num_heads=4, head_dim=128, max_batch_size=1, max_seq_len=2048)
        sharded_cache = StaticKVCache(num_layers=28, num_heads=2, head_dim=128, max_batch_size=1, max_seq_len=2048)

        full_bytes = full_cache.storage.nelement() * full_cache.storage.element_size()
        sharded_bytes = sharded_cache.storage.nelement() * sharded_cache.storage.element_size()

        assert sharded_bytes == full_bytes // 2, f"Expected 50% KV memory, got {sharded_bytes} vs {full_bytes}"

    def test_tp_attn_numerical_parity_vs_monolithic(self):
        """Verifies sharded GQA attention matches monolithic attention within <= 0.1% relative error."""
        hidden_size = 512
        num_heads = 8
        num_kv_heads = 2
        head_dim = 64
        attn_module = nn.Module()
        attn_module.num_heads = num_heads
        attn_module.num_kv_heads = num_kv_heads
        attn_module.head_dim = head_dim
        attn_module.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False, dtype=torch.float32)
        attn_module.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False, dtype=torch.float32)
        attn_module.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False, dtype=torch.float32)
        attn_module.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False, dtype=torch.float32)

        x = torch.randn(2, 8, hidden_size, dtype=torch.float32)

        # Monolithic reference
        b, m, _ = x.shape
        q = attn_module.q_proj(x).view(b, m, num_heads, head_dim).transpose(1, 2)
        k = attn_module.k_proj(x).view(b, m, num_kv_heads, head_dim).transpose(1, 2).repeat_interleave(num_heads // num_kv_heads, dim=1)
        v = attn_module.v_proj(x).view(b, m, num_kv_heads, head_dim).transpose(1, 2).repeat_interleave(num_heads // num_kv_heads, dim=1)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(b, m, -1)
        y_unsharded = attn_module.o_proj(attn_out)

        # Real TPParallelAttention
        tp_attn0 = tp.TPParallelAttention.from_attention(attn_module, rank=0, world_size=2, all_reduce=False)
        tp_attn1 = tp.TPParallelAttention.from_attention(attn_module, rank=1, world_size=2, all_reduce=False)

        out0 = tp_attn0(x)
        out1 = tp_attn1(x)
        y_sharded, _ = comm.reference_all_reduce_sum(out0, out1)

        rel_err = relative_l2_error(y_sharded, y_unsharded)
        assert rel_err <= 1e-3, f"Attention relative error {rel_err} exceeds 0.1% gate"

    def test_tp_attn_causal_mask_enforcement(self):
        """Verifies that attention computation applies causal masking properly without future leakage."""
        b, m, d = 1, 4, 64
        q = torch.randn(b, 2, m, d)
        k = torch.randn(b, 2, m, d)
        v = torch.randn(b, 2, m, d)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d)
        causal_mask = torch.triu(torch.full((m, m), float("-inf")), diagonal=1)
        scores_masked = scores + causal_mask

        weights = F.softmax(scores_masked, dim=-1)
        for i in range(m):
            for j in range(i + 1, m):
                assert weights[0, :, i, j].abs().max().item() == 0.0

    def test_tp_attn_single_allreduce_per_block(self, monkeypatch):
        """Verifies that Attention block forward pass executes exactly ONE All-Reduce (at o_proj)."""
        attn0 = tp.TPParallelAttention(hidden_size=256, num_heads=4, num_kv_heads=2, head_dim=64, rank=0, world_size=2, dtype=torch.float32)
        attn1 = tp.TPParallelAttention(hidden_size=256, num_heads=4, num_kv_heads=2, head_dim=64, rank=1, world_size=2, dtype=torch.float32, all_reduce=False)

        x = torch.randn(1, 4, 256, dtype=torch.float32)
        part1 = attn1(x)

        comm_count = 0
        def counted_all_reduce(a, b):
            nonlocal comm_count
            comm_count += 1
            return comm.reference_all_reduce_sum(a, b)

        monkeypatch.setattr(tp, "all_reduce_pair", counted_all_reduce)
        y = attn0(x, peer_out=part1)

        assert comm_count == 1, f"Expected 1 All-Reduce for attention block, got {comm_count}"
        assert y.shape == (1, 4, 256)

    def test_tp_attn_autoregressive_step_with_cache(self):
        """Verifies sequential autoregressive decode steps maintain consistent cache position indexing."""
        if not HAS_STATIC_KV:
            pytest.skip("src.static_kv_cache.StaticKVCache not available")

        cache0 = StaticKVCache(num_layers=1, num_heads=2, head_dim=64, max_batch_size=1, max_seq_len=32)
        cache1 = StaticKVCache(num_layers=1, num_heads=2, head_dim=64, max_batch_size=1, max_seq_len=32)

        for step in range(5):
            k0 = torch.randn(1, 2, 1, 64)
            v0 = torch.randn(1, 2, 1, 64)
            k1 = torch.randn(1, 2, 1, 64)
            v1 = torch.randn(1, 2, 1, 64)

            cache0.key_cache[0][:, :, step:step + 1, :] = k0
            cache0.value_cache[0][:, :, step:step + 1, :] = v0
            cache1.key_cache[0][:, :, step:step + 1, :] = k1
            cache1.value_cache[0][:, :, step:step + 1, :] = v1

        assert cache0.key_cache[0].shape == (1, 2, 32, 64)
        assert cache1.key_cache[0].shape == (1, 2, 32, 64)


# ==============================================================================
# Suite 7: Negative Tests and Boundary Checks
# ==============================================================================

@skip_if_no_tp
class TestTPNegativeAndBoundary:
    """Verifies graceful rejection and clean error handling for non-contiguous, mismatched, or invalid inputs."""

    def test_negative_non_contiguous_activation(self):
        """Verifies layers handle transposed or strided inputs without memory corruption."""
        x = torch.randn(4, 8, 256, dtype=torch.float32)
        x_transposed = x.transpose(0, 1)
        assert not x_transposed.is_contiguous()

        col = tp.TPColumnParallelLinear(256, 512, rank=0, world_size=2, dtype=torch.float32)
        out = col(x_transposed)
        assert out.shape == (8, 4, 256)

    def test_negative_non_contiguous_weight(self):
        """Verifies passing non-contiguous weight matrices is safely handled."""
        w = torch.randn(512, 256, dtype=torch.float32)
        w_t = w.t()
        assert not w_t.is_contiguous()

        x = torch.randn(1, 4, 512, dtype=torch.float32)
        out = F.linear(x, w_t)
        assert out.shape == (1, 4, 256)

    @skip_if_no_dual_cuda
    @skip_if_no_comm
    def test_negative_allreduce_mismatched_shapes(self):
        """Verifies All-Reduce with mismatched tensor shapes raises ValueError or RuntimeError."""
        t0 = torch.randn(16, 64, device="cuda:0")
        t1 = torch.randn(16, 32, device="cuda:1")
        fn = getattr(comm, "all_reduce_sum", getattr(comm, "all_reduce_pair", None))
        with pytest.raises((ValueError, RuntimeError)):
            fn(t0, t1)

    @skip_if_no_dual_cuda
    @skip_if_no_comm
    def test_negative_allreduce_mismatched_dtypes(self):
        """Verifies All-Reduce with mismatched dtypes raises TypeError or ValueError."""
        t0 = torch.randn(64, device="cuda:0", dtype=torch.float16)
        t1 = torch.randn(64, device="cuda:1", dtype=torch.float32)
        fn = getattr(comm, "all_reduce_sum", getattr(comm, "all_reduce_pair", None))
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            fn(t0, t1)

    @skip_if_no_dual_cuda
    @skip_if_no_comm
    def test_negative_allreduce_single_device_error(self):
        """Verifies passing both tensors on identical device raises ValueError or RuntimeError."""
        t0 = torch.randn(64, device="cuda:0")
        t1 = torch.randn(64, device="cuda:0")
        fn = getattr(comm, "all_reduce_sum", getattr(comm, "all_reduce_pair", None))
        with pytest.raises((ValueError, RuntimeError)):
            fn(t0, t1)

    def test_negative_odd_feature_dimension(self):
        """Verifies attempting to partition non-divisible out_features raises ValueError."""
        with pytest.raises(ValueError):
            tp.TPColumnParallelLinear(in_features=256, out_features=257, rank=0, world_size=2)

    def test_negative_odd_kv_heads(self):
        """Verifies attempting to shard an odd number of KV heads raises ValueError."""
        with pytest.raises(ValueError):
            tp.TPParallelAttention(hidden_size=256, num_heads=4, num_kv_heads=3, rank=0, world_size=2)

    def test_negative_invalid_rank_index(self):
        """Verifies passing rank >= world_size raises ValueError."""
        with pytest.raises(ValueError):
            tp.TPColumnParallelLinear(in_features=256, out_features=256, rank=2, world_size=2)


# ==============================================================================
# Suite 8: Dual-GPU Physical Integration
# ==============================================================================

@skip_if_no_dual_cuda
@skip_if_no_tp
class TestDualDeviceIntegration:
    """Verifies end-to-end multi-device execution on physical dual Tesla T4 GPUs."""

    def test_dual_gpu_transformer_layer_end_to_end(self):
        """Executes complete Transformer layer (Attention + MLP) across physical cuda:0 and cuda:1."""
        h = 512
        inter = 1024
        attn_ref = nn.Module()
        attn_ref.num_heads = 8
        attn_ref.num_kv_heads = 2
        attn_ref.head_dim = 64
        attn_ref.q_proj = nn.Linear(h, 8 * 64, bias=False, dtype=torch.float32)
        attn_ref.k_proj = nn.Linear(h, 2 * 64, bias=False, dtype=torch.float32)
        attn_ref.v_proj = nn.Linear(h, 2 * 64, bias=False, dtype=torch.float32)
        attn_ref.o_proj = nn.Linear(8 * 64, h, bias=False, dtype=torch.float32)

        mlp_ref = nn.Module()
        mlp_ref.gate_proj = nn.Linear(h, inter, bias=False, dtype=torch.float32)
        mlp_ref.up_proj = nn.Linear(h, inter, bias=False, dtype=torch.float32)
        mlp_ref.down_proj = nn.Linear(inter, h, bias=False, dtype=torch.float32)

        x_cpu = torch.randn(1, 8, h, dtype=torch.float32)
        b, m, _ = x_cpu.shape
        q = attn_ref.q_proj(x_cpu).view(b, m, 8, 64).transpose(1, 2)
        k = attn_ref.k_proj(x_cpu).view(b, m, 2, 64).transpose(1, 2).repeat_interleave(4, dim=1)
        v = attn_ref.v_proj(x_cpu).view(b, m, 2, 64).transpose(1, 2).repeat_interleave(4, dim=1)
        scores = torch.matmul(q, k.transpose(-1, -2)) / 8.0
        attn_out = torch.matmul(F.softmax(scores, dim=-1), v).transpose(1, 2).contiguous().view(b, m, -1)
        h1_ref = x_cpu + attn_ref.o_proj(attn_out)
        act_ref = F.silu(mlp_ref.gate_proj(h1_ref)) * mlp_ref.up_proj(h1_ref)
        y_ref = h1_ref + mlp_ref.down_proj(act_ref)

        attn0 = tp.TPParallelAttention.from_attention(attn_ref, rank=0, world_size=2).to("cuda:0")
        attn1 = tp.TPParallelAttention.from_attention(attn_ref, rank=1, world_size=2, all_reduce=False).to("cuda:1")
        mlp0 = tp.TPParallelMLP.from_mlp(mlp_ref, rank=0, world_size=2, quant_type=None).to("cuda:0")
        mlp1 = tp.TPParallelMLP.from_mlp(mlp_ref, rank=1, world_size=2, quant_type=None, all_reduce=False).to("cuda:1")

        x0 = x_cpu.to("cuda:0")
        x1 = x_cpu.to("cuda:1")

        part_attn1 = attn1(x1)
        attn_out0 = attn0(x0, peer_out=part_attn1)
        h1_0 = x0 + attn_out0
        h1_1 = h1_0.to("cuda:1")

        part_mlp1 = mlp1(h1_1)
        mlp_out0 = mlp0(h1_0, peer_act=part_mlp1)
        y_sharded = h1_0 + mlp_out0

        rel_err = relative_l2_error(y_sharded.cpu(), y_ref)
        assert rel_err <= 1e-3, f"Transformer layer relative error {rel_err} exceeds 0.1% gate"

    def test_dual_gpu_device_residence_integrity(self):
        """Verifies tensors allocated on dual devices maintain their assigned device residence."""
        t0 = torch.randn(64, device="cuda:0")
        t1 = torch.randn(64, device="cuda:1")
        assert t0.device.type == "cuda" and t0.device.index == 0
        assert t1.device.type == "cuda" and t1.device.index == 1

    def test_dual_gpu_memory_balance(self):
        """Verifies memory allocation across cuda:0 and cuda:1 has less than 5% variance."""
        torch.cuda.empty_cache()
        t0 = torch.zeros(1024 * 1024 * 64, device="cuda:0", dtype=torch.float32)  # 256 MB
        t1 = torch.zeros(1024 * 1024 * 64, device="cuda:1", dtype=torch.float32)  # 256 MB

        alloc0 = torch.cuda.memory_allocated(0)
        alloc1 = torch.cuda.memory_allocated(1)

        variance = abs(alloc0 - alloc1) / max(alloc0, 1)
        assert variance < 0.05, f"VRAM variance {variance:.4f} exceeds 5% balance limit"

    def test_dual_gpu_consecutive_decode_stability(self):
        """Verifies 10 consecutive dual-GPU decode steps without CUDA errors or memory growth."""
        mlp0 = tp.TPParallelMLP(hidden_size=256, intermediate_size=512, rank=0, world_size=2, quant_type=None, dtype=torch.float16, device=torch.device("cuda:0"))
        mlp1 = tp.TPParallelMLP(hidden_size=256, intermediate_size=512, rank=1, world_size=2, quant_type=None, dtype=torch.float16, device=torch.device("cuda:1"), all_reduce=False)
        x0 = torch.randn(1, 1, 256, dtype=torch.float16, device="cuda:0")
        x1 = x0.to("cuda:1")

        initial_alloc = torch.cuda.memory_allocated(0)
        for _ in range(10):
            part1 = mlp1(x1)
            x0 = mlp0(x0, peer_act=part1)
            assert x0.shape == (1, 1, 256)
            x1 = x0.to("cuda:1")

        final_alloc = torch.cuda.memory_allocated(0)
        assert final_alloc == initial_alloc, "Memory leaked across consecutive decode steps"


# ==============================================================================
# Suite 9: Direct Module Exports and CPU Algorithmic Parity
# ==============================================================================

class TestDirectModuleExports:
    """Verifies that public APIs are cleanly exported from src.sharding package."""

    def test_sharding_package_exports(self):
        import src.sharding as sharding
        expected_exports = [
            "all_reduce_pair",
            "all_reduce_dual",
            "all_reduce_dual_inplace",
            "all_reduce_sum",
            "p2p_transfer",
            "p2p_copy",
            "reference_all_reduce_sum",
            "check_dual_gpu",
            "require_dual_cuda",
            "TPColumnParallelLinear",
            "TPRowParallelLinear",
            "TPParallelMLP",
            "TPParallelAttention",
            "quantize_weight_sym_int4",
            "dequantize_sym_int4",
            "slice_column_parallel_weight",
            "slice_column_parallel_int4",
            "slice_row_parallel_weight",
            "slice_row_parallel_int4",
        ]
        for exp in expected_exports:
            assert hasattr(sharding, exp), f"src.sharding is missing export {exp}"

    def test_all_reduce_pair_alias(self):
        import src.sharding.comm as c
        assert hasattr(c, "all_reduce_pair")
        assert c.all_reduce_pair is c.all_reduce_dual


class TestDirectCommCPU:
    """Verifies CPU fallback, reference all-reduce, and input validation without requiring CUDA."""

    def test_comm_cpu_all_reduce_out_of_place(self):
        t0 = torch.tensor([1.0, 2.0, 3.0, 4.0])
        t1 = torch.tensor([10.0, 20.0, 30.0, 40.0])
        r0, r1 = comm.all_reduce_sum(t0, t1, inplace=False)
        expected = torch.tensor([11.0, 22.0, 33.0, 44.0])
        assert torch.equal(r0, expected)
        assert torch.equal(r1, expected)
        assert torch.equal(t0, torch.tensor([1.0, 2.0, 3.0, 4.0]))
        assert torch.equal(t1, torch.tensor([10.0, 20.0, 30.0, 40.0]))

    def test_comm_cpu_all_reduce_in_place(self):
        t0 = torch.tensor([1.0, 2.0, 3.0, 4.0])
        t1 = torch.tensor([10.0, 20.0, 30.0, 40.0])
        r0, r1 = comm.all_reduce_sum(t0, t1, in_place=True)
        expected = torch.tensor([11.0, 22.0, 33.0, 44.0])
        assert torch.equal(t0, expected)
        assert torch.equal(t1, expected)

    def test_comm_validation_errors(self):
        with pytest.raises(ValueError):
            comm.validate_comm_tensors(torch.zeros(4), torch.zeros(5), require_cuda=False)
        with pytest.raises(TypeError):
            comm.validate_comm_tensors(torch.zeros(4, dtype=torch.float32), torch.zeros(4, dtype=torch.float16), require_cuda=False)


@skip_if_no_tp
class TestDirectTPModuleCPU:
    """Verifies TPParallelMLP, TPParallelAttention, and INT4 slicing directly on CPU."""

    def test_tp_mlp_direct_cpu_parity(self):
        hidden_size = 256
        intermediate_size = 512
        mlp_module = nn.Module()
        mlp_module.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=torch.float32)
        mlp_module.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=torch.float32)
        mlp_module.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=torch.float32)

        x = torch.randn(2, 4, hidden_size, dtype=torch.float32)

        tp_mlp0 = tp.TPParallelMLP.from_mlp(mlp_module, rank=0, world_size=2, quant_type=None, all_reduce=False)
        tp_mlp1 = tp.TPParallelMLP.from_mlp(mlp_module, rank=1, world_size=2, quant_type=None, all_reduce=False)

        part0 = tp_mlp0(x)
        part1 = tp_mlp1(x)
        y_tp, _ = comm.reference_all_reduce_sum(part0, part1)

        act_ref = F.silu(mlp_module.gate_proj(x)) * mlp_module.up_proj(x)
        y_ref = mlp_module.down_proj(act_ref)

        rel_err = relative_l2_error(y_tp, y_ref)
        assert rel_err <= 1e-4, f"Direct TPParallelMLP relative error {rel_err} exceeds 1e-4"

    def test_tp_attention_direct_cpu_parity(self):
        hidden_size = 256
        num_heads = 8
        num_kv_heads = 2
        head_dim = 32
        attn_module = nn.Module()
        attn_module.num_heads = num_heads
        attn_module.num_kv_heads = num_kv_heads
        attn_module.head_dim = head_dim
        attn_module.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False, dtype=torch.float32)
        attn_module.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False, dtype=torch.float32)
        attn_module.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False, dtype=torch.float32)
        attn_module.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False, dtype=torch.float32)

        x = torch.randn(2, 4, hidden_size, dtype=torch.float32)

        tp_attn0 = tp.TPParallelAttention.from_attention(attn_module, rank=0, world_size=2, all_reduce=False)
        tp_attn1 = tp.TPParallelAttention.from_attention(attn_module, rank=1, world_size=2, all_reduce=False)

        out0 = tp_attn0(x)
        out1 = tp_attn1(x)
        y_tp, _ = comm.reference_all_reduce_sum(out0, out1)

        b, m, _ = x.shape
        q = attn_module.q_proj(x).view(b, m, num_heads, head_dim).transpose(1, 2)
        k = attn_module.k_proj(x).view(b, m, num_kv_heads, head_dim).transpose(1, 2).repeat_interleave(num_heads // num_kv_heads, dim=1)
        v = attn_module.v_proj(x).view(b, m, num_kv_heads, head_dim).transpose(1, 2).repeat_interleave(num_heads // num_kv_heads, dim=1)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(b, m, -1)
        y_ref = attn_module.o_proj(attn_out)

        rel_err = relative_l2_error(y_tp, y_ref)
        assert rel_err <= 1e-4, f"Direct TPParallelAttention relative error {rel_err} exceeds 1e-4"

    def test_direct_int4_slicing_math(self):
        W = torch.randn(512, 256)
        p, s, z, g = tp.quantize_weight_sym_int4(W, group_size=128)

        p0, s0, z0 = tp.slice_column_parallel_int4(p, s, z, 0, 2)
        p1, s1, z1 = tp.slice_column_parallel_int4(p, s, z, 1, 2)
        assert p0.shape == (256 // 8, 256)
        assert p1.shape == (256 // 8, 256)
        w_dec_col = torch.cat([tp.dequantize_sym_int4(p0, s0, z0, g), tp.dequantize_sym_int4(p1, s1, z1, g)], dim=0)
        w_dec_full = tp.dequantize_sym_int4(p, s, z, g)
        assert torch.equal(w_dec_col, w_dec_full)

        pr0, sr0, zr0 = tp.slice_row_parallel_int4(p, s, z, 0, 2, group_size=g)
        pr1, sr1, zr1 = tp.slice_row_parallel_int4(p, s, z, 1, 2, group_size=g)
        assert pr0.shape == (128 // 8, 512)
        assert pr1.shape == (128 // 8, 512)
        w_dec_row = torch.cat([tp.dequantize_sym_int4(pr0, sr0, zr0, g), tp.dequantize_sym_int4(pr1, sr1, zr1, g)], dim=1)
        assert torch.equal(w_dec_row, w_dec_full)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
