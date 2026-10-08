#!/usr/bin/env python3
"""
tests/test_multi_gpu_guard.py

Multi-device unit test suite for Milestone 1: Multi-Device CUDAGuard and Stream Safety.

Coverage:
1. Single-device execution on cuda:0 and cuda:1 independently for:
   - fused_w4a16_gemm_u4
   - fused_w4a16_gemm_s4
   - dequantize_u4
   - dequantize_s4
   - dequantize_s3
   - dequantize_fp8
   - fused_h17_gemv_s3
2. Negative tests: cross-device mismatched tensors raise RuntimeError
3. Contiguity tests: non-contiguous inputs raise RuntimeError
4. Clean multi-device detection: skips cleanly with [SKIP] if device_count < 2
5. Numerical verification: cuda:1 matches cuda:0 bit-for-bit on identical inputs
6. Thread-safety: concurrent multi-threaded execution on dual devices
"""

import sys
import threading
import pytest
import torch
import numpy as np

try:
    import t4_kernels
    HAS_T4_KERNELS = True
except ImportError:
    HAS_T4_KERNELS = False

CUDA_AVAILABLE = torch.cuda.is_available()
NUM_DEVICES = torch.cuda.device_count() if CUDA_AVAILABLE else 0

skip_if_no_cuda = pytest.mark.skipif(
    not CUDA_AVAILABLE or NUM_DEVICES < 1,
    reason="CUDA is not available on this system"
)

skip_if_no_dual_cuda = pytest.mark.skipif(
    not CUDA_AVAILABLE or NUM_DEVICES < 2,
    reason=f"Requires at least 2 CUDA devices (found {NUM_DEVICES})"
)

skip_if_no_t4_kernels = pytest.mark.skipif(
    not HAS_T4_KERNELS,
    reason="t4_kernels extension is not installed"
)


# ==============================================================================
# Pure CPU Reference Implementations
# ==============================================================================

def cpu_dequantize_u4(packed_tensor, scales_tensor, zps_tensor):
    """Pure CPU reference for unsigned 4-bit dequantization."""
    packed = packed_tensor.detach().cpu().numpy().astype(np.uint32)
    scales = scales_tensor.detach().cpu().numpy().astype(np.float32)
    zps = zps_tensor.detach().cpu().numpy().astype(np.float32)

    if scales.size == 1:
        scales = np.broadcast_to(scales, packed.shape)
    if zps.size == 1:
        zps = np.broadcast_to(zps, packed.shape)

    unpacked = np.zeros((*packed.shape, 8), dtype=np.float32)
    for i in range(8):
        nibble = ((packed >> (i * 4)) & 0xF).astype(np.int32)
        unpacked[..., i] = (nibble.astype(np.float32) - zps) * scales

    return torch.tensor(unpacked, dtype=torch.float16)


def cpu_dequantize_s4(packed_tensor, scales_tensor, zps_tensor):
    """Pure CPU reference for signed 4-bit dequantization."""
    packed = packed_tensor.detach().cpu().numpy().astype(np.uint32)
    scales = scales_tensor.detach().cpu().numpy().astype(np.float32)
    zps = zps_tensor.detach().cpu().numpy().astype(np.float32)

    if scales.size == 1:
        scales = np.broadcast_to(scales, packed.shape)
    if zps.size == 1:
        zps = np.broadcast_to(zps, packed.shape)

    unpacked = np.zeros((*packed.shape, 8), dtype=np.float32)
    for i in range(8):
        nibble = ((packed >> (i * 4)) & 0xF).astype(np.int32)
        nibble = np.where(nibble >= 8, nibble - 16, nibble)
        unpacked[..., i] = (nibble.astype(np.float32) - zps) * scales

    return torch.tensor(unpacked, dtype=torch.float16)


def cpu_dequantize_s3(packed_tensor, scales_tensor, zps_tensor):
    """Pure CPU reference for signed 3-bit dequantization (10 values per uint32)."""
    packed = packed_tensor.detach().cpu().numpy().astype(np.uint32)
    scales = scales_tensor.detach().cpu().numpy().astype(np.float32)
    zps = zps_tensor.detach().cpu().numpy().astype(np.float32)

    if scales.size == 1:
        scales = np.broadcast_to(scales, packed.shape)
    if zps.size == 1:
        zps = np.broadcast_to(zps, packed.shape)

    unpacked = np.zeros((*packed.shape, 10), dtype=np.float32)
    bit_shifts = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]
    for i, shift in enumerate(bit_shifts):
        val3 = (packed >> shift) & 0x7
        s3_val = np.where(val3 >= 4, val3 - 8, val3)
        unpacked[..., i] = (s3_val - zps) * scales

    return torch.tensor(unpacked, dtype=torch.float16)


def cpu_dequantize_fp8(packed_tensor, scales_tensor):
    """Pure CPU reference for FP8 E4M3 dequantization (4 values per uint32)."""
    packed = packed_tensor.detach().cpu().numpy().astype(np.uint32)
    scales = scales_tensor.detach().cpu().numpy().astype(np.float32)

    if scales.size == 1:
        scales = np.broadcast_to(scales, packed.shape)

    unpacked = np.zeros((*packed.shape, 4), dtype=np.float32)
    for i in range(4):
        byte_val = (packed >> (i * 8)) & 0xFF
        sign = (byte_val >> 7) & 0x1
        exp = (byte_val >> 3) & 0xF
        mant = byte_val & 0x7

        float_val = np.where(exp > 0, ((-1.0)**sign) * (2.0**(exp - 7)) * (1.0 + mant / 8.0), 0.0)
        unpacked[..., i] = float_val * scales

    return torch.tensor(unpacked, dtype=torch.float16)


def unpack_w4(W_packed, K, N, is_signed=False):
    """Unpacks INT4 matrix on CPU."""
    unpacked = torch.zeros((K, N), dtype=torch.int32)
    for i in range(8):
        unpacked[i::8, :] = (W_packed >> (i * 4)) & 0xF
    if is_signed:
        unpacked = torch.where(unpacked >= 8, unpacked - 16, unpacked)
    return unpacked


def cpu_reference_gemm(A, W_packed, scales, zero_points, is_signed=False, group_size=0):
    """Pure CPU reference for W4A16 GEMM."""
    K_8, N = W_packed.shape
    K = K_8 * 8
    unpacked = unpack_w4(W_packed, K, N, is_signed=is_signed).float()

    if scales.dim() == 2 and scales.shape[0] > 1:
        num_groups = scales.shape[0]
        g_size = K // num_groups if group_size <= 0 else group_size
        unpacked_g = unpacked.reshape(num_groups, g_size, N)
        sc_3d = scales.float().unsqueeze(1)
        zp_3d = zero_points.float().unsqueeze(1)
        W_dequant = ((unpacked_g - zp_3d) * sc_3d).reshape(K, N)
    else:
        W_dequant = (unpacked - zero_points.float()) * scales.float()

    C_ref = torch.matmul(A.float(), W_dequant)
    return C_ref.half()


# ==============================================================================
# 1. Single-Device Execution Tests
# ==============================================================================

@skip_if_no_cuda
@skip_if_no_t4_kernels
class TestSingleDeviceExecution:
    """Verifies all kernels run on valid CUDA devices independently."""

    def _check_device_available(self, dev_idx):
        if dev_idx >= NUM_DEVICES:
            pytest.skip(f"Device cuda:{dev_idx} is not available (device_count={NUM_DEVICES})")

    @pytest.mark.parametrize("dev_idx", [0, 1])
    def test_dequantize_u4_execution(self, dev_idx):
        self._check_device_available(dev_idx)
        device = torch.device(f"cuda:{dev_idx}")
        n_words = 128
        packed = torch.randint(-2147483648, 2147483647, (n_words,), dtype=torch.int32, device=device)
        scales = torch.full((n_words,), 0.05, dtype=torch.float16, device=device)
        zps = torch.full((n_words,), 2.0, dtype=torch.float16, device=device)

        out = t4_kernels.dequantize_u4(packed, scales, zps)

        assert out.device == device
        assert out.dtype == torch.float16
        assert out.numel() == n_words * 8
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

        ref = cpu_dequantize_u4(packed, scales, zps).view(-1)
        diff = torch.abs(out.cpu() - ref)
        assert diff.max().item() <= 0.05

    @pytest.mark.parametrize("dev_idx", [0, 1])
    def test_dequantize_s4_execution(self, dev_idx):
        self._check_device_available(dev_idx)
        device = torch.device(f"cuda:{dev_idx}")
        n_words = 128
        packed = torch.randint(-2147483648, 2147483647, (n_words,), dtype=torch.int32, device=device)
        scales = torch.full((n_words,), 0.05, dtype=torch.float16, device=device)
        zps = torch.full((n_words,), 0.0, dtype=torch.float16, device=device)

        out = t4_kernels.dequantize_s4(packed, scales, zps)

        assert out.device == device
        assert out.dtype == torch.float16
        assert out.numel() == n_words * 8

        ref = cpu_dequantize_s4(packed, scales, zps).view(-1)
        diff = torch.abs(out.cpu() - ref)
        assert diff.max().item() <= 0.05

    @pytest.mark.parametrize("dev_idx", [0, 1])
    def test_dequantize_s3_execution(self, dev_idx):
        self._check_device_available(dev_idx)
        device = torch.device(f"cuda:{dev_idx}")
        n_words = 100
        packed = torch.randint(-2147483648, 2147483647, (n_words,), dtype=torch.int32, device=device)
        scales = torch.full((n_words,), 0.05, dtype=torch.float16, device=device)
        zps = torch.full((n_words,), 0.0, dtype=torch.float16, device=device)

        out = t4_kernels.dequantize_s3(packed, scales, zps)

        assert out.device == device
        assert out.dtype == torch.float16
        assert out.numel() == n_words * 10

        ref = cpu_dequantize_s3(packed, scales, zps).view(-1)
        diff = torch.abs(out.cpu() - ref)
        assert diff.max().item() <= 0.05

    @pytest.mark.parametrize("dev_idx", [0, 1])
    def test_dequantize_fp8_execution(self, dev_idx):
        self._check_device_available(dev_idx)
        device = torch.device(f"cuda:{dev_idx}")
        n_words = 128
        packed = torch.randint(-2147483648, 2147483647, (n_words,), dtype=torch.int32, device=device)
        scales = torch.full((n_words,), 0.1, dtype=torch.float16, device=device)

        out = t4_kernels.dequantize_fp8(packed, scales)

        assert out.device == device
        assert out.dtype == torch.float16
        assert out.numel() == n_words * 4

        ref = cpu_dequantize_fp8(packed, scales).view(-1)
        diff = torch.abs(out.cpu() - ref)
        assert diff.max().item() <= 0.05

    @pytest.mark.parametrize("dev_idx", [0, 1])
    @pytest.mark.parametrize("is_signed", [False, True])
    def test_fused_w4a16_gemm_per_channel(self, dev_idx, is_signed):
        self._check_device_available(dev_idx)
        device = torch.device(f"cuda:{dev_idx}")
        M, K, N = 1, 256, 128
        A = torch.randn(M, K, dtype=torch.float16, device=device)
        W_packed = torch.randint(0, 2**31 - 1, (K // 8, N), dtype=torch.int32, device=device)
        scales = torch.randn(1, N, dtype=torch.float16, device=device) * 0.05
        zero_points = torch.randint(0, 16, (1, N), dtype=torch.float16, device=device)

        fn = t4_kernels.fused_w4a16_gemm_s4 if is_signed else t4_kernels.fused_w4a16_gemm_u4
        out = fn(A, W_packed, scales, zero_points, 0)

        assert out.device == device
        assert out.shape == (M, N)
        assert out.dtype == torch.float16

        ref = cpu_reference_gemm(A.cpu(), W_packed.cpu(), scales.cpu(), zero_points.cpu(), is_signed=is_signed, group_size=0)
        assert torch.allclose(out.cpu(), ref, atol=0.08, rtol=0.02)

    @pytest.mark.parametrize("dev_idx", [0, 1])
    @pytest.mark.parametrize("is_signed", [False, True])
    def test_fused_w4a16_gemm_per_group(self, dev_idx, is_signed):
        self._check_device_available(dev_idx)
        device = torch.device(f"cuda:{dev_idx}")
        M, K, N = 2, 256, 128
        group_size = 128
        num_groups = K // group_size
        A = torch.randn(M, K, dtype=torch.float16, device=device)
        W_packed = torch.randint(0, 2**31 - 1, (K // 8, N), dtype=torch.int32, device=device)
        scales = torch.randn(num_groups, N, dtype=torch.float16, device=device) * 0.05
        zero_points = torch.randint(0, 16, (num_groups, N), dtype=torch.float16, device=device)

        fn = t4_kernels.fused_w4a16_gemm_s4 if is_signed else t4_kernels.fused_w4a16_gemm_u4
        out = fn(A, W_packed, scales, zero_points, group_size)

        assert out.device == device
        assert out.shape == (M, N)
        assert out.dtype == torch.float16

        ref = cpu_reference_gemm(A.cpu(), W_packed.cpu(), scales.cpu(), zero_points.cpu(), is_signed=is_signed, group_size=group_size)
        assert torch.allclose(out.cpu(), ref, atol=0.08, rtol=0.02)

    @pytest.mark.parametrize("dev_idx", [0, 1])
    def test_fused_h17_gemv_s3_execution(self, dev_idx):
        self._check_device_available(dev_idx)
        device = torch.device(f"cuda:{dev_idx}")
        M, K, N = 1, 100, 64
        A = torch.randn(M, K, dtype=torch.float16, device=device)
        W_packed = torch.randint(0, 2**31 - 1, (K // 10, N), dtype=torch.int32, device=device)
        num_groups = (K + 99) // 100
        scales = torch.randn(num_groups * N, dtype=torch.float16, device=device) * 0.05
        zero_points = torch.zeros(num_groups * N, dtype=torch.float16, device=device)

        out = t4_kernels.fused_h17_gemv_s3(A, W_packed, scales, zero_points)
        assert out.device == device
        assert out.shape == (M, N)
        assert out.dtype == torch.float16

    @skip_if_no_dual_cuda
    def test_cudaguard_active_context_independence(self):
        """Verifies kernel switches device context via CUDAGuard and restores caller context."""
        # Active device set to 0, but execute on cuda:1
        with torch.cuda.device(0):
            assert torch.cuda.current_device() == 0
            dev1 = torch.device("cuda:1")
            A = torch.randn(1, 128, dtype=torch.float16, device=dev1)
            W = torch.randint(0, 2**31 - 1, (16, 64), dtype=torch.int32, device=dev1)
            s = torch.randn(1, 64, dtype=torch.float16, device=dev1) * 0.05
            z = torch.zeros(1, 64, dtype=torch.float16, device=dev1)

            out = t4_kernels.fused_w4a16_gemm_u4(A, W, s, z, 0)
            assert out.device == dev1
            assert torch.cuda.current_device() == 0

        # Active device set to 1, but execute on cuda:0
        with torch.cuda.device(1):
            assert torch.cuda.current_device() == 1
            dev0 = torch.device("cuda:0")
            A0 = torch.randn(1, 128, dtype=torch.float16, device=dev0)
            W0 = torch.randint(0, 2**31 - 1, (16, 64), dtype=torch.int32, device=dev0)
            s0 = torch.randn(1, 64, dtype=torch.float16, device=dev0) * 0.05
            z0 = torch.zeros(1, 64, dtype=torch.float16, device=dev0)

            out0 = t4_kernels.fused_w4a16_gemm_u4(A0, W0, s0, z0, 0)
            assert out0.device == dev0
            assert torch.cuda.current_device() == 1

    @pytest.mark.parametrize("dev_idx", [0, 1])
    def test_custom_stream_execution(self, dev_idx):
        """Verifies kernel binds to input device stream properly."""
        self._check_device_available(dev_idx)
        device = torch.device(f"cuda:{dev_idx}")
        stream = torch.cuda.Stream(device=device)

        with torch.cuda.stream(stream):
            packed = torch.randint(-2147483648, 2147483647, (64,), dtype=torch.int32, device=device)
            scales = torch.full((64,), 0.05, dtype=torch.float16, device=device)
            zps = torch.full((64,), 0.0, dtype=torch.float16, device=device)
            out = t4_kernels.dequantize_u4(packed, scales, zps)

        stream.synchronize()
        assert out.device == device
        assert out.numel() == 64 * 8

    @pytest.mark.parametrize("dev_idx", [0, 1])
    def test_fused_backward_gemm_adamw_execution(self, dev_idx):
        self._check_device_available(dev_idx)
        device = torch.device(f"cuda:{dev_idx}")
        K, M, N = 64, 64, 64
        dY = torch.randn(K, M, dtype=torch.float16, device=device)
        X = torch.randn(K, N, dtype=torch.float16, device=device)
        W_master = torch.randn(M, N, dtype=torch.float32, device=device)
        W_active = W_master.half().clone()
        exp_avg = torch.zeros(M, N, dtype=torch.float32, device=device)
        exp_avg_sq = torch.zeros(M, N, dtype=torch.float32, device=device)

        t4_kernels.fused_backward_gemm_adamw(
            dY, X, W_master, W_active, exp_avg, exp_avg_sq,
            1e-3, 0.9, 0.999, 1e-8, 0.01, 1.0, 1.0
        )

        assert W_master.device == device
        assert W_active.device == device
        assert exp_avg.device == device
        assert exp_avg_sq.device == device
        assert not torch.isnan(W_master).any()
        assert not torch.isnan(exp_avg).any()
        assert torch.count_nonzero(exp_avg) > 0
        assert torch.count_nonzero(exp_avg_sq) > 0

    @pytest.mark.parametrize("dev_idx", [0, 1])
    def test_fused_ellie_rmsnorm_execution(self, dev_idx):
        self._check_device_available(dev_idx)
        device = torch.device(f"cuda:{dev_idx}")
        M, D = 4, 128
        x = torch.randn(M, D, dtype=torch.float16, device=device)
        gamma = torch.ones(D, dtype=torch.float16, device=device)

        out = t4_kernels.fused_ellie_rmsnorm(x, gamma, 1e-6)

        assert out.device == device
        assert out.shape == (M, D)
        assert out.dtype == torch.float16
        assert not torch.isnan(out).any()

        rms = torch.sqrt(torch.mean(x.cpu().float() ** 2, dim=-1, keepdim=True) + 1e-6)
        ref = (x.cpu().float() / rms * gamma.cpu().float()).half()
        assert torch.allclose(out.cpu(), ref, atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("dev_idx", [0, 1])
    def test_fused_ellie_swiglu_execution(self, dev_idx):
        self._check_device_available(dev_idx)
        device = torch.device(f"cuda:{dev_idx}")
        M, H = 4, 256
        gate = torch.randn(M, H, dtype=torch.float16, device=device)
        up = torch.randn(M, H, dtype=torch.float16, device=device)

        out = t4_kernels.fused_ellie_swiglu(gate, up)

        assert out.device == device
        assert out.shape == (M, H)
        assert out.dtype == torch.float16
        assert not torch.isnan(out).any()

        ref = (torch.nn.functional.silu(gate.cpu().float()) * up.cpu().float()).half()
        assert torch.allclose(out.cpu(), ref, atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("dev_idx", [0, 1])
    def test_fused_ellie_rmsnorm_w4a16_gemv_swiglu_execution(self, dev_idx):
        self._check_device_available(dev_idx)
        device = torch.device(f"cuda:{dev_idx}")
        D, H = 256, 128
        group_size = 128
        num_groups = D // group_size

        x = torch.randn(1, D, dtype=torch.float16, device=device)
        gamma = torch.ones(D, dtype=torch.float16, device=device)
        W_gate = torch.randint(0, 2**31 - 1, (D // 8, H), dtype=torch.int32, device=device)
        scale_gate = torch.randn(num_groups, H, dtype=torch.float16, device=device) * 0.05
        zp_gate = torch.zeros(num_groups, H, dtype=torch.float16, device=device)
        W_up = torch.randint(0, 2**31 - 1, (D // 8, H), dtype=torch.int32, device=device)
        scale_up = torch.randn(num_groups, H, dtype=torch.float16, device=device) * 0.05
        zp_up = torch.zeros(num_groups, H, dtype=torch.float16, device=device)

        out = t4_kernels.fused_ellie_rmsnorm_w4a16_gemv_swiglu(
            x, gamma, W_gate, scale_gate, zp_gate, W_up, scale_up, zp_up, group_size, 1e-6
        )

        assert out.device == device
        assert out.shape == (1, H)
        assert out.dtype == torch.float16
        assert not torch.isnan(out).any()

        zp_empty = torch.empty(0, dtype=torch.float16, device=device)
        out_no_zp = t4_kernels.fused_ellie_rmsnorm_w4a16_gemv_swiglu(
            x, gamma, W_gate, scale_gate, zp_empty, W_up, scale_up, zp_empty, group_size, 1e-6
        )
        assert out_no_zp.device == device
        assert out_no_zp.shape == (1, H)

    @pytest.mark.parametrize("dev_idx", [0, 1])
    def test_fused_swiglu_backward_execution(self, dev_idx):
        self._check_device_available(dev_idx)
        device = torch.device(f"cuda:{dev_idx}")
        M, H = 4, 256
        dY = torch.randn(M, H, dtype=torch.float16, device=device)
        gate = torch.randn(M, H, dtype=torch.float16, device=device)
        up = torch.randn(M, H, dtype=torch.float16, device=device)

        d_gate, d_up = t4_kernels.fused_swiglu_backward(dY, gate, up)

        assert d_gate.device == device
        assert d_up.device == device
        assert d_gate.shape == (M, H)
        assert d_up.shape == (M, H)
        assert not torch.isnan(d_gate).any()
        assert not torch.isnan(d_up).any()

        g_ref = gate.cpu().float().clone().requires_grad_(True)
        u_ref = up.cpu().float().clone().requires_grad_(True)
        y_ref = torch.nn.functional.silu(g_ref) * u_ref
        y_ref.backward(dY.cpu().float())

        assert torch.allclose(d_gate.cpu(), g_ref.grad.half(), atol=1e-2, rtol=1e-2)
        assert torch.allclose(d_up.cpu(), u_ref.grad.half(), atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("dev_idx", [0, 1])
    def test_fused_sft_lora_backward_adamw_execution(self, dev_idx):
        self._check_device_available(dev_idx)
        device = torch.device(f"cuda:{dev_idx}")
        M, d_in, d_out, r = 4, 32, 64, 16
        dY = torch.randn(M, d_out, dtype=torch.float16, device=device)
        X = torch.randn(M, d_in, dtype=torch.float16, device=device)
        H_lora = torch.randn(M, r, dtype=torch.float16, device=device)
        A_master = torch.randn(r, d_in, dtype=torch.float32, device=device)
        A_active = A_master.half().clone()
        m_A = torch.zeros(r, d_in, dtype=torch.float32, device=device)
        v_A = torch.zeros(r, d_in, dtype=torch.float32, device=device)
        B_master = torch.randn(d_out, r, dtype=torch.float32, device=device)
        B_active = B_master.half().clone()
        m_B = torch.zeros(d_out, r, dtype=torch.float32, device=device)
        v_B = torch.zeros(d_out, r, dtype=torch.float32, device=device)

        dX = t4_kernels.fused_sft_lora_backward_adamw(
            dY, X, H_lora, A_master, A_active, m_A, v_A, B_master, B_active, m_B, v_B,
            1e-3, 0.9, 0.999, 1e-8, 0.01, 1.0, 1.0
        )

        assert dX.device == device
        assert dX.shape == (M, d_in)
        assert dX.dtype == torch.float16
        assert not torch.isnan(dX).any()
        assert torch.count_nonzero(m_A) > 0
        assert torch.count_nonzero(m_B) > 0



# ==============================================================================
# 2. Negative Tests: Cross-Device Mismatched Tensors
# ==============================================================================

@skip_if_no_dual_cuda
@skip_if_no_t4_kernels
class TestCrossDeviceMismatches:
    """Verifies that passing tensors residing on differing devices raises an exception."""

    def test_gemm_u4_mismatched_activation_and_weights(self):
        dev0 = torch.device("cuda:0")
        dev1 = torch.device("cuda:1")
        A = torch.randn(1, 128, dtype=torch.float16, device=dev0)
        W = torch.randint(0, 2**31 - 1, (16, 64), dtype=torch.int32, device=dev1)
        scales = torch.randn(1, 64, dtype=torch.float16, device=dev0)
        zps = torch.zeros(1, 64, dtype=torch.float16, device=dev0)

        with pytest.raises(RuntimeError) as exc_info:
            t4_kernels.fused_w4a16_gemm_u4(A, W, scales, zps, 0)
        err_msg = str(exc_info.value).lower()
        assert "device" in err_msg or "cuda" in err_msg

    def test_gemm_u4_mismatched_scales(self):
        dev0 = torch.device("cuda:0")
        dev1 = torch.device("cuda:1")
        A = torch.randn(1, 128, dtype=torch.float16, device=dev0)
        W = torch.randint(0, 2**31 - 1, (16, 64), dtype=torch.int32, device=dev0)
        scales = torch.randn(1, 64, dtype=torch.float16, device=dev1)
        zps = torch.zeros(1, 64, dtype=torch.float16, device=dev0)

        with pytest.raises(RuntimeError) as exc_info:
            t4_kernels.fused_w4a16_gemm_u4(A, W, scales, zps, 0)
        err_msg = str(exc_info.value).lower()
        assert "device" in err_msg or "cuda" in err_msg

    def test_gemm_u4_mismatched_zero_points(self):
        dev0 = torch.device("cuda:0")
        dev1 = torch.device("cuda:1")
        A = torch.randn(1, 128, dtype=torch.float16, device=dev0)
        W = torch.randint(0, 2**31 - 1, (16, 64), dtype=torch.int32, device=dev0)
        scales = torch.randn(1, 64, dtype=torch.float16, device=dev0)
        zps = torch.zeros(1, 64, dtype=torch.float16, device=dev1)

        with pytest.raises(RuntimeError) as exc_info:
            t4_kernels.fused_w4a16_gemm_u4(A, W, scales, zps, 0)
        err_msg = str(exc_info.value).lower()
        assert "device" in err_msg or "cuda" in err_msg

    def test_gemm_s4_mismatched_activation_and_weights(self):
        dev0 = torch.device("cuda:0")
        dev1 = torch.device("cuda:1")
        A = torch.randn(1, 128, dtype=torch.float16, device=dev1)
        W = torch.randint(0, 2**31 - 1, (16, 64), dtype=torch.int32, device=dev0)
        scales = torch.randn(1, 64, dtype=torch.float16, device=dev1)
        zps = torch.zeros(1, 64, dtype=torch.float16, device=dev1)

        with pytest.raises(RuntimeError) as exc_info:
            t4_kernels.fused_w4a16_gemm_s4(A, W, scales, zps, 0)
        err_msg = str(exc_info.value).lower()
        assert "device" in err_msg or "cuda" in err_msg

    def test_dequantize_u4_mismatched_scales(self):
        dev0 = torch.device("cuda:0")
        dev1 = torch.device("cuda:1")
        packed = torch.randint(0, 100, (64,), dtype=torch.int32, device=dev0)
        scales = torch.full((64,), 0.05, dtype=torch.float16, device=dev1)
        zps = torch.full((64,), 0.0, dtype=torch.float16, device=dev0)

        with pytest.raises(RuntimeError) as exc_info:
            t4_kernels.dequantize_u4(packed, scales, zps)
        err_msg = str(exc_info.value).lower()
        assert "device" in err_msg or "cuda" in err_msg

    def test_dequantize_s4_mismatched_zero_points(self):
        dev0 = torch.device("cuda:0")
        dev1 = torch.device("cuda:1")
        packed = torch.randint(0, 100, (64,), dtype=torch.int32, device=dev0)
        scales = torch.full((64,), 0.05, dtype=torch.float16, device=dev0)
        zps = torch.full((64,), 0.0, dtype=torch.float16, device=dev1)

        with pytest.raises(RuntimeError) as exc_info:
            t4_kernels.dequantize_s4(packed, scales, zps)
        err_msg = str(exc_info.value).lower()
        assert "device" in err_msg or "cuda" in err_msg

    def test_dequantize_s3_mismatched_devices(self):
        dev0 = torch.device("cuda:0")
        dev1 = torch.device("cuda:1")
        packed = torch.randint(0, 100, (50,), dtype=torch.int32, device=dev1)
        scales = torch.full((50,), 0.05, dtype=torch.float16, device=dev0)
        zps = torch.full((50,), 0.0, dtype=torch.float16, device=dev1)

        with pytest.raises(RuntimeError) as exc_info:
            t4_kernels.dequantize_s3(packed, scales, zps)
        err_msg = str(exc_info.value).lower()
        assert "device" in err_msg or "cuda" in err_msg

    def test_dequantize_fp8_mismatched_scales(self):
        dev0 = torch.device("cuda:0")
        dev1 = torch.device("cuda:1")
        packed = torch.randint(0, 100, (64,), dtype=torch.int32, device=dev0)
        scales = torch.full((64,), 0.1, dtype=torch.float16, device=dev1)

        with pytest.raises(RuntimeError) as exc_info:
            t4_kernels.dequantize_fp8(packed, scales)
        err_msg = str(exc_info.value).lower()
        assert "device" in err_msg or "cuda" in err_msg

    def test_fused_h17_gemv_s3_mismatched_weights(self):
        dev0 = torch.device("cuda:0")
        dev1 = torch.device("cuda:1")
        A = torch.randn(1, 100, dtype=torch.float16, device=dev0)
        W = torch.randint(0, 2**31 - 1, (10, 64), dtype=torch.int32, device=dev1)
        scales = torch.randn(64, dtype=torch.float16, device=dev0) * 0.05
        zps = torch.zeros(64, dtype=torch.float16, device=dev0)

        with pytest.raises(RuntimeError) as exc_info:
            t4_kernels.fused_h17_gemv_s3(A, W, scales, zps)
        err_msg = str(exc_info.value).lower()
        assert "device" in err_msg or "cuda" in err_msg

    def test_cpu_gpu_mismatch(self):
        dev0 = torch.device("cuda:0")
        A_cpu = torch.randn(1, 128, dtype=torch.float16, device="cpu")
        W_cuda = torch.randint(0, 2**31 - 1, (16, 64), dtype=torch.int32, device=dev0)
        scales = torch.randn(1, 64, dtype=torch.float16, device=dev0)
        zps = torch.zeros(1, 64, dtype=torch.float16, device=dev0)

        with pytest.raises(RuntimeError) as exc_info:
            t4_kernels.fused_w4a16_gemm_u4(A_cpu, W_cuda, scales, zps, 0)
        err_msg = str(exc_info.value).lower()
        assert "cuda" in err_msg

    def test_backward_gemm_adamw_mismatched_devices(self):
        dev0 = torch.device("cuda:0")
        dev1 = torch.device("cuda:1")
        K, M, N = 64, 64, 64
        dY = torch.randn(K, M, dtype=torch.float16, device=dev0)
        X = torch.randn(K, N, dtype=torch.float16, device=dev1)
        W_master = torch.randn(M, N, dtype=torch.float32, device=dev0)
        W_active = W_master.half().clone()
        exp_avg = torch.zeros(M, N, dtype=torch.float32, device=dev0)
        exp_avg_sq = torch.zeros(M, N, dtype=torch.float32, device=dev0)

        with pytest.raises(RuntimeError) as exc_info:
            t4_kernels.fused_backward_gemm_adamw(
                dY, X, W_master, W_active, exp_avg, exp_avg_sq,
                1e-3, 0.9, 0.999, 1e-8, 0.01, 1.0, 1.0
            )
        err = str(exc_info.value).lower()
        assert "device" in err or "cuda" in err

    def test_ellie_rmsnorm_mismatched_devices(self):
        dev0 = torch.device("cuda:0")
        dev1 = torch.device("cuda:1")
        x = torch.randn(4, 128, dtype=torch.float16, device=dev0)
        gamma = torch.ones(128, dtype=torch.float16, device=dev1)

        with pytest.raises(RuntimeError) as exc_info:
            t4_kernels.fused_ellie_rmsnorm(x, gamma)
        err = str(exc_info.value).lower()
        assert "device" in err or "cuda" in err

    def test_ellie_swiglu_mismatched_devices(self):
        dev0 = torch.device("cuda:0")
        dev1 = torch.device("cuda:1")
        gate = torch.randn(4, 256, dtype=torch.float16, device=dev0)
        up = torch.randn(4, 256, dtype=torch.float16, device=dev1)

        with pytest.raises(RuntimeError) as exc_info:
            t4_kernels.fused_ellie_swiglu(gate, up)
        err = str(exc_info.value).lower()
        assert "device" in err or "cuda" in err

    def test_ellie_rmsnorm_w4a16_gemv_swiglu_mismatched_devices(self):
        dev0 = torch.device("cuda:0")
        dev1 = torch.device("cuda:1")
        D, H = 256, 128
        x = torch.randn(1, D, dtype=torch.float16, device=dev0)
        gamma = torch.ones(D, dtype=torch.float16, device=dev0)
        W_gate = torch.randint(0, 2**31 - 1, (D // 8, H), dtype=torch.int32, device=dev1)
        scale_gate = torch.randn(2, H, dtype=torch.float16, device=dev0) * 0.05
        zp_gate = torch.zeros(2, H, dtype=torch.float16, device=dev0)
        W_up = torch.randint(0, 2**31 - 1, (D // 8, H), dtype=torch.int32, device=dev0)
        scale_up = torch.randn(2, H, dtype=torch.float16, device=dev0) * 0.05
        zp_up = torch.zeros(2, H, dtype=torch.float16, device=dev0)

        with pytest.raises(RuntimeError) as exc_info:
            t4_kernels.fused_ellie_rmsnorm_w4a16_gemv_swiglu(
                x, gamma, W_gate, scale_gate, zp_gate, W_up, scale_up, zp_up, 128, 1e-6
            )
        err = str(exc_info.value).lower()
        assert "device" in err or "cuda" in err

    def test_swiglu_backward_mismatched_devices(self):
        dev0 = torch.device("cuda:0")
        dev1 = torch.device("cuda:1")
        dY = torch.randn(4, 256, dtype=torch.float16, device=dev0)
        gate = torch.randn(4, 256, dtype=torch.float16, device=dev1)
        up = torch.randn(4, 256, dtype=torch.float16, device=dev0)

        with pytest.raises(RuntimeError) as exc_info:
            t4_kernels.fused_swiglu_backward(dY, gate, up)
        err = str(exc_info.value).lower()
        assert "device" in err or "cuda" in err

    def test_sft_lora_backward_adamw_mismatched_devices(self):
        dev0 = torch.device("cuda:0")
        dev1 = torch.device("cuda:1")
        M, d_in, d_out, r = 4, 32, 64, 16
        dY = torch.randn(M, d_out, dtype=torch.float16, device=dev0)
        X = torch.randn(M, d_in, dtype=torch.float16, device=dev1)
        H_lora = torch.randn(M, r, dtype=torch.float16, device=dev0)
        A_master = torch.randn(r, d_in, dtype=torch.float32, device=dev0)
        A_active = A_master.half().clone()
        m_A = torch.zeros(r, d_in, dtype=torch.float32, device=dev0)
        v_A = torch.zeros(r, d_in, dtype=torch.float32, device=dev0)
        B_master = torch.randn(d_out, r, dtype=torch.float32, device=dev0)
        B_active = B_master.half().clone()
        m_B = torch.zeros(d_out, r, dtype=torch.float32, device=dev0)
        v_B = torch.zeros(d_out, r, dtype=torch.float32, device=dev0)

        with pytest.raises(RuntimeError) as exc_info:
            t4_kernels.fused_sft_lora_backward_adamw(
                dY, X, H_lora, A_master, A_active, m_A, v_A, B_master, B_active, m_B, v_B,
                1e-3, 0.9, 0.999, 1e-8, 0.01, 1.0, 1.0
            )
        err = str(exc_info.value).lower()
        assert "device" in err or "cuda" in err



# ==============================================================================
# 3. Contiguity Validation Tests
# ==============================================================================

@skip_if_no_cuda
@skip_if_no_t4_kernels
class TestContiguityValidation:
    """Verifies that non-contiguous inputs are rejected with a RuntimeError."""

    def test_gemm_noncontiguous_activation(self):
        device = torch.device("cuda:0")
        M, K, N = 2, 128, 64
        A_raw = torch.randn(K, M, dtype=torch.float16, device=device)
        A_noncontig = A_raw.t()
        assert not A_noncontig.is_contiguous()

        W = torch.randint(0, 2**31 - 1, (K // 8, N), dtype=torch.int32, device=device)
        scales = torch.randn(1, N, dtype=torch.float16, device=device)
        zps = torch.zeros(1, N, dtype=torch.float16, device=device)

        with pytest.raises(RuntimeError) as exc_info:
            t4_kernels.fused_w4a16_gemm_u4(A_noncontig, W, scales, zps, 0)
        assert "contiguous" in str(exc_info.value).lower()

    def test_gemm_noncontiguous_weights(self):
        device = torch.device("cuda:0")
        M, K, N = 1, 128, 64
        A = torch.randn(M, K, dtype=torch.float16, device=device)
        W_raw = torch.randint(0, 2**31 - 1, (N, K // 8), dtype=torch.int32, device=device)
        W_noncontig = W_raw.t()
        assert not W_noncontig.is_contiguous()

        scales = torch.randn(1, N, dtype=torch.float16, device=device)
        zps = torch.zeros(1, N, dtype=torch.float16, device=device)

        with pytest.raises(RuntimeError) as exc_info:
            t4_kernels.fused_w4a16_gemm_u4(A, W_noncontig, scales, zps, 0)
        assert "contiguous" in str(exc_info.value).lower()

    def test_gemm_noncontiguous_scales(self):
        device = torch.device("cuda:0")
        M, K, N = 1, 256, 128
        group_size = 128
        num_groups = K // group_size
        A = torch.randn(M, K, dtype=torch.float16, device=device)
        W = torch.randint(0, 2**31 - 1, (K // 8, N), dtype=torch.int32, device=device)

        scales_raw = torch.randn(N, num_groups, dtype=torch.float16, device=device)
        scales_noncontig = scales_raw.t()
        assert not scales_noncontig.is_contiguous()
        zps = torch.zeros(num_groups, N, dtype=torch.float16, device=device)

        with pytest.raises(RuntimeError) as exc_info:
            t4_kernels.fused_w4a16_gemm_u4(A, W, scales_noncontig, zps, group_size)
        assert "contiguous" in str(exc_info.value).lower()

    def test_gemm_noncontiguous_zero_points(self):
        device = torch.device("cuda:0")
        M, K, N = 1, 256, 128
        group_size = 128
        num_groups = K // group_size
        A = torch.randn(M, K, dtype=torch.float16, device=device)
        W = torch.randint(0, 2**31 - 1, (K // 8, N), dtype=torch.int32, device=device)
        scales = torch.randn(num_groups, N, dtype=torch.float16, device=device)

        zps_raw = torch.zeros(N, num_groups, dtype=torch.float16, device=device)
        zps_noncontig = zps_raw.t()
        assert not zps_noncontig.is_contiguous()

        with pytest.raises(RuntimeError) as exc_info:
            t4_kernels.fused_w4a16_gemm_s4(A, W, scales, zps_noncontig, group_size)
        assert "contiguous" in str(exc_info.value).lower()

    def test_dequantize_noncontiguous_packed_weights(self):
        device = torch.device("cuda:0")
        base = torch.randint(0, 100, (128,), dtype=torch.int32, device=device)
        packed_noncontig = base[::2]
        assert not packed_noncontig.is_contiguous()

        scales = torch.full((64,), 0.05, dtype=torch.float16, device=device)
        zps = torch.full((64,), 0.0, dtype=torch.float16, device=device)

        with pytest.raises(RuntimeError) as exc_info:
            t4_kernels.dequantize_u4(packed_noncontig, scales, zps)
        assert "contiguous" in str(exc_info.value).lower()

    def test_dequantize_noncontiguous_scales(self):
        device = torch.device("cuda:0")
        packed = torch.randint(0, 100, (64,), dtype=torch.int32, device=device)
        scales_base = torch.full((128,), 0.05, dtype=torch.float16, device=device)
        scales_noncontig = scales_base[::2]
        assert not scales_noncontig.is_contiguous()
        zps = torch.full((64,), 0.0, dtype=torch.float16, device=device)

        with pytest.raises(RuntimeError) as exc_info:
            t4_kernels.dequantize_s4(packed, scales_noncontig, zps)
        assert "contiguous" in str(exc_info.value).lower()

    def test_dequantize_noncontiguous_zero_points(self):
        device = torch.device("cuda:0")
        packed = torch.randint(0, 100, (64,), dtype=torch.int32, device=device)
        scales = torch.full((64,), 0.05, dtype=torch.float16, device=device)
        zps_base = torch.full((128,), 0.0, dtype=torch.float16, device=device)
        zps_noncontig = zps_base[::2]
        assert not zps_noncontig.is_contiguous()

        with pytest.raises(RuntimeError) as exc_info:
            t4_kernels.dequantize_u4(packed, scales, zps_noncontig)
        assert "contiguous" in str(exc_info.value).lower()

    def test_backward_gemm_adamw_noncontiguous_inputs(self):
        device = torch.device("cuda:0")
        K, M, N = 64, 64, 64
        dY_raw = torch.randn(M, K, dtype=torch.float16, device=device)
        dY = dY_raw.t()
        assert not dY.is_contiguous()

        X = torch.randn(K, N, dtype=torch.float16, device=device)
        W_master = torch.randn(M, N, dtype=torch.float32, device=device)
        W_active = W_master.half().clone()
        exp_avg = torch.zeros(M, N, dtype=torch.float32, device=device)
        exp_avg_sq = torch.zeros(M, N, dtype=torch.float32, device=device)

        with pytest.raises(RuntimeError) as exc_info:
            t4_kernels.fused_backward_gemm_adamw(
                dY, X, W_master, W_active, exp_avg, exp_avg_sq,
                1e-3, 0.9, 0.999, 1e-8, 0.01, 1.0, 1.0
            )
        assert "contiguous" in str(exc_info.value).lower()

    def test_ellie_rmsnorm_noncontiguous_inputs(self):
        device = torch.device("cuda:0")
        M, D = 4, 128
        x_raw = torch.randn(D, M, dtype=torch.float16, device=device)
        x = x_raw.t()
        assert not x.is_contiguous()
        gamma = torch.ones(D, dtype=torch.float16, device=device)

        with pytest.raises(RuntimeError) as exc_info:
            t4_kernels.fused_ellie_rmsnorm(x, gamma)
        assert "contiguous" in str(exc_info.value).lower()

    def test_ellie_swiglu_noncontiguous_inputs(self):
        device = torch.device("cuda:0")
        M, H = 4, 256
        gate_raw = torch.randn(H, M, dtype=torch.float16, device=device)
        gate = gate_raw.t()
        assert not gate.is_contiguous()
        up = torch.randn(M, H, dtype=torch.float16, device=device)

        with pytest.raises(RuntimeError) as exc_info:
            t4_kernels.fused_ellie_swiglu(gate, up)
        assert "contiguous" in str(exc_info.value).lower()

    def test_ellie_rmsnorm_w4a16_gemv_swiglu_noncontiguous_inputs(self):
        device = torch.device("cuda:0")
        D, H = 256, 128
        x = torch.randn(1, D, dtype=torch.float16, device=device)
        gamma = torch.ones(D, dtype=torch.float16, device=device)
        W_gate_raw = torch.randint(0, 2**31 - 1, (H, D // 8), dtype=torch.int32, device=device)
        W_gate = W_gate_raw.t()
        assert not W_gate.is_contiguous()
        scale_gate = torch.randn(2, H, dtype=torch.float16, device=device) * 0.05
        zp_gate = torch.zeros(2, H, dtype=torch.float16, device=device)
        W_up = torch.randint(0, 2**31 - 1, (D // 8, H), dtype=torch.int32, device=device)
        scale_up = torch.randn(2, H, dtype=torch.float16, device=device) * 0.05
        zp_up = torch.zeros(2, H, dtype=torch.float16, device=device)

        with pytest.raises(RuntimeError) as exc_info:
            t4_kernels.fused_ellie_rmsnorm_w4a16_gemv_swiglu(
                x, gamma, W_gate, scale_gate, zp_gate, W_up, scale_up, zp_up, 128, 1e-6
            )
        assert "contiguous" in str(exc_info.value).lower()

    def test_swiglu_backward_noncontiguous_inputs(self):
        device = torch.device("cuda:0")
        M, H = 4, 256
        dY_raw = torch.randn(H, M, dtype=torch.float16, device=device)
        dY = dY_raw.t()
        assert not dY.is_contiguous()
        gate = torch.randn(M, H, dtype=torch.float16, device=device)
        up = torch.randn(M, H, dtype=torch.float16, device=device)

        with pytest.raises(RuntimeError) as exc_info:
            t4_kernels.fused_swiglu_backward(dY, gate, up)
        assert "contiguous" in str(exc_info.value).lower()

    def test_sft_lora_backward_adamw_noncontiguous_inputs(self):
        device = torch.device("cuda:0")
        M, d_in, d_out, r = 4, 32, 64, 16
        dY_raw = torch.randn(d_out, M, dtype=torch.float16, device=device)
        dY = dY_raw.t()
        assert not dY.is_contiguous()
        X = torch.randn(M, d_in, dtype=torch.float16, device=device)
        H_lora = torch.randn(M, r, dtype=torch.float16, device=device)
        A_master = torch.randn(r, d_in, dtype=torch.float32, device=device)
        A_active = A_master.half().clone()
        m_A = torch.zeros(r, d_in, dtype=torch.float32, device=device)
        v_A = torch.zeros(r, d_in, dtype=torch.float32, device=device)
        B_master = torch.randn(d_out, r, dtype=torch.float32, device=device)
        B_active = B_master.half().clone()
        m_B = torch.zeros(d_out, r, dtype=torch.float32, device=device)
        v_B = torch.zeros(d_out, r, dtype=torch.float32, device=device)

        with pytest.raises(RuntimeError) as exc_info:
            t4_kernels.fused_sft_lora_backward_adamw(
                dY, X, H_lora, A_master, A_active, m_A, v_A, B_master, B_active, m_B, v_B,
                1e-3, 0.9, 0.999, 1e-8, 0.01, 1.0, 1.0
            )
        assert "contiguous" in str(exc_info.value).lower()


# ==============================================================================
# Shape and Edge Case Validation Tests
# ==============================================================================

@skip_if_no_cuda
@skip_if_no_t4_kernels
class TestShapeAndEdgeValidation:
    """Verifies edge cases: 3D RMSNorm, 2D GEMM dimension checks, and 0-size tensors."""

    def test_ellie_rmsnorm_3d_activation_edge_case(self):
        device = torch.device("cuda:0")
        B, S, D = 2, 16, 128
        x_3d = torch.randn(B, S, D, dtype=torch.float16, device=device)
        gamma = torch.ones(D, dtype=torch.float16, device=device)

        try:
            out = t4_kernels.fused_ellie_rmsnorm(x_3d, gamma)
            assert out.shape == (B, S, D)
            rms = torch.sqrt(torch.mean(x_3d.cpu().float() ** 2, dim=-1, keepdim=True) + 1e-6)
            ref = (x_3d.cpu().float() / rms * gamma.cpu().float()).half()
            assert torch.allclose(out.cpu(), ref, atol=1e-2, rtol=1e-2)
        except RuntimeError as e:
            err = str(e).lower()
            assert "dim" in err or "dimension" in err or "shape" in err

    def test_ellie_rmsnorm_gamma_dimension_mismatch(self):
        device = torch.device("cuda:0")
        x = torch.randn(4, 128, dtype=torch.float16, device=device)
        gamma = torch.ones(64, dtype=torch.float16, device=device)
        with pytest.raises(RuntimeError) as exc_info:
            t4_kernels.fused_ellie_rmsnorm(x, gamma)
        assert "gamma size must match hidden dimension" in str(exc_info.value).lower()

    def test_gemm_1d_tensor_rejected(self):
        device = torch.device("cuda:0")
        A_1d = torch.randn(128, dtype=torch.float16, device=device)
        W = torch.randint(0, 2**31 - 1, (16, 64), dtype=torch.int32, device=device)
        scales = torch.randn(1, 64, dtype=torch.float16, device=device) * 0.05
        zps = torch.zeros(1, 64, dtype=torch.float16, device=device)

        with pytest.raises(RuntimeError) as exc_info:
            t4_kernels.fused_w4a16_gemm_u4(A_1d, W, scales, zps, 0)
        err = str(exc_info.value).lower()
        assert "dim" in err or "2d" in err or "dimension" in err or "range" in err

    def test_gemm_3d_tensor_rejected(self):
        device = torch.device("cuda:0")
        A_3d = torch.randn(2, 4, 128, dtype=torch.float16, device=device)
        W = torch.randint(0, 2**31 - 1, (16, 64), dtype=torch.int32, device=device)
        scales = torch.randn(1, 64, dtype=torch.float16, device=device) * 0.05
        zps = torch.zeros(1, 64, dtype=torch.float16, device=device)

        with pytest.raises(RuntimeError) as exc_info:
            t4_kernels.fused_w4a16_gemm_u4(A_3d, W, scales, zps, 0)
        err = str(exc_info.value).lower()
        assert "dim" in err or "2d" in err or "dimension" in err

    def test_gemm_empty_activation_rejected(self):
        device = torch.device("cuda:0")
        A_empty = torch.empty(0, 128, dtype=torch.float16, device=device)
        W = torch.randint(0, 2**31 - 1, (16, 64), dtype=torch.int32, device=device)
        scales = torch.randn(1, 64, dtype=torch.float16, device=device) * 0.05
        zps = torch.zeros(1, 64, dtype=torch.float16, device=device)

        with pytest.raises(RuntimeError) as exc_info:
            t4_kernels.fused_w4a16_gemm_u4(A_empty, W, scales, zps, 0)
        err = str(exc_info.value).lower()
        assert "empty" in err or "m > 0" in err or "numel" in err or "dimension" in err or "invalid" in err

    def test_backward_gemm_adamw_batch_mismatch(self):
        device = torch.device("cuda:0")
        dY = torch.randn(64, 64, dtype=torch.float16, device=device)
        X = torch.randn(32, 64, dtype=torch.float16, device=device)
        W_master = torch.randn(64, 64, dtype=torch.float32, device=device)
        W_active = W_master.half().clone()
        exp_avg = torch.zeros(64, 64, dtype=torch.float32, device=device)
        exp_avg_sq = torch.zeros(64, 64, dtype=torch.float32, device=device)

        with pytest.raises(RuntimeError) as exc_info:
            t4_kernels.fused_backward_gemm_adamw(
                dY, X, W_master, W_active, exp_avg, exp_avg_sq,
                1e-3, 0.9, 0.999, 1e-8, 0.01, 1.0, 1.0
            )
        assert "batch dimension must match" in str(exc_info.value).lower()

    def test_backward_gemm_adamw_weight_shape_mismatch(self):
        device = torch.device("cuda:0")
        dY = torch.randn(64, 64, dtype=torch.float16, device=device)
        X = torch.randn(64, 64, dtype=torch.float16, device=device)
        W_master = torch.randn(32, 64, dtype=torch.float32, device=device)
        W_active = torch.randn(32, 64, dtype=torch.float16, device=device)
        exp_avg = torch.zeros(32, 64, dtype=torch.float32, device=device)
        exp_avg_sq = torch.zeros(32, 64, dtype=torch.float32, device=device)

        with pytest.raises(RuntimeError) as exc_info:
            t4_kernels.fused_backward_gemm_adamw(
                dY, X, W_master, W_active, exp_avg, exp_avg_sq,
                1e-3, 0.9, 0.999, 1e-8, 0.01, 1.0, 1.0
            )
        assert "shape must be [m, n]" in str(exc_info.value).lower()

    def test_ellie_swiglu_dimension_mismatch(self):
        device = torch.device("cuda:0")
        gate = torch.randn(4, 256, dtype=torch.float16, device=device)
        up = torch.randn(4, 128, dtype=torch.float16, device=device)

        with pytest.raises(RuntimeError) as exc_info:
            t4_kernels.fused_ellie_swiglu(gate, up)
        assert "identical dimensions" in str(exc_info.value).lower()

    def test_swiglu_backward_element_count_mismatch(self):
        device = torch.device("cuda:0")
        dY = torch.randn(4, 256, dtype=torch.float16, device=device)
        gate = torch.randn(4, 128, dtype=torch.float16, device=device)
        up = torch.randn(4, 256, dtype=torch.float16, device=device)

        with pytest.raises(RuntimeError) as exc_info:
            t4_kernels.fused_swiglu_backward(dY, gate, up)
        err = str(exc_info.value).lower()
        assert "numel" in err or "size" in err or "match" in err or "dimension" in err

    def test_ellie_rmsnorm_w4a16_gemv_swiglu_packing_dimension_mismatch(self):
        device = torch.device("cuda:0")
        D, H = 256, 128
        x = torch.randn(1, D, dtype=torch.float16, device=device)
        gamma = torch.ones(D, dtype=torch.float16, device=device)
        W_gate_bad = torch.randint(0, 2**31 - 1, (16, H), dtype=torch.int32, device=device)
        scale_gate = torch.randn(2, H, dtype=torch.float16, device=device) * 0.05
        zp_gate = torch.zeros(2, H, dtype=torch.float16, device=device)
        W_up = torch.randint(0, 2**31 - 1, (D // 8, H), dtype=torch.int32, device=device)
        scale_up = torch.randn(2, H, dtype=torch.float16, device=device) * 0.05
        zp_up = torch.zeros(2, H, dtype=torch.float16, device=device)

        with pytest.raises(RuntimeError) as exc_info:
            t4_kernels.fused_ellie_rmsnorm_w4a16_gemv_swiglu(
                x, gamma, W_gate_bad, scale_gate, zp_gate, W_up, scale_up, zp_up, 128, 1e-6
            )
        assert "w_gate rows must equal d/8" in str(exc_info.value).lower()

    def test_dequantize_empty_weights_rejected(self):
        device = torch.device("cuda:0")
        empty_packed = torch.empty(0, dtype=torch.int32, device=device)
        scales = torch.empty(0, dtype=torch.float16, device=device)
        zps = torch.empty(0, dtype=torch.float16, device=device)

        with pytest.raises(RuntimeError) as exc_info:
            t4_kernels.dequantize_u4(empty_packed, scales, zps)
        err = str(exc_info.value).lower()
        assert "empty" in err or "numel" in err or "num_uint32s > 0" in err or "invalid" in err



# ==============================================================================
# 4. Numerical Verification: cuda:1 Matches cuda:0 Bit-for-Bit
# ==============================================================================

@skip_if_no_dual_cuda
@skip_if_no_t4_kernels
class TestNumericalBitForBitCrossDevice:
    """Verifies outputs on cuda:1 match cuda:0 bit-for-bit on identical inputs."""

    def test_dequantize_u4_bit_identical(self):
        torch.manual_seed(1337)
        n_words = 1024
        p_cpu = torch.randint(-2147483648, 2147483647, (n_words,), dtype=torch.int32)
        s_cpu = torch.rand(n_words, dtype=torch.float16) * 0.099 + 0.001
        z_cpu = torch.randint(-8, 8, (n_words,), dtype=torch.float16)

        p0, s0, z0 = p_cpu.to("cuda:0"), s_cpu.to("cuda:0"), z_cpu.to("cuda:0")
        p1, s1, z1 = p_cpu.to("cuda:1"), s_cpu.to("cuda:1"), z_cpu.to("cuda:1")

        torch.cuda.synchronize("cuda:0")
        torch.cuda.synchronize("cuda:1")

        out0 = t4_kernels.dequantize_u4(p0, s0, z0)
        out1 = t4_kernels.dequantize_u4(p1, s1, z1)

        torch.cuda.synchronize("cuda:0")
        torch.cuda.synchronize("cuda:1")

        out0_cpu = out0.cpu()
        out1_cpu = out1.cpu()

        assert torch.equal(out0_cpu.view(torch.int16), out1_cpu.view(torch.int16)), \
            "Bit mismatch between cuda:0 and cuda:1 dequantize_u4 outputs"
        assert torch.max(torch.abs(out0_cpu - out1_cpu)).item() == 0.0

    def test_dequantize_s4_bit_identical(self):
        torch.manual_seed(2026)
        n_words = 1024
        p_cpu = torch.randint(-2147483648, 2147483647, (n_words,), dtype=torch.int32)
        s_cpu = torch.rand(n_words, dtype=torch.float16) * 0.099 + 0.001
        z_cpu = torch.randint(-8, 8, (n_words,), dtype=torch.float16)

        p0, s0, z0 = p_cpu.to("cuda:0"), s_cpu.to("cuda:0"), z_cpu.to("cuda:0")
        p1, s1, z1 = p_cpu.to("cuda:1"), s_cpu.to("cuda:1"), z_cpu.to("cuda:1")

        out0 = t4_kernels.dequantize_s4(p0, s0, z0)
        out1 = t4_kernels.dequantize_s4(p1, s1, z1)

        torch.cuda.synchronize("cuda:0")
        torch.cuda.synchronize("cuda:1")

        out0_cpu = out0.cpu()
        out1_cpu = out1.cpu()

        assert torch.equal(out0_cpu.view(torch.int16), out1_cpu.view(torch.int16)), \
            "Bit mismatch between cuda:0 and cuda:1 dequantize_s4 outputs"
        assert torch.max(torch.abs(out0_cpu - out1_cpu)).item() == 0.0

    def test_dequantize_s3_bit_identical(self):
        torch.manual_seed(777)
        n_words = 1000
        p_cpu = torch.randint(-2147483648, 2147483647, (n_words,), dtype=torch.int32)
        s_cpu = torch.rand(n_words, dtype=torch.float16) * 0.05
        z_cpu = torch.zeros(n_words, dtype=torch.float16)

        p0, s0, z0 = p_cpu.to("cuda:0"), s_cpu.to("cuda:0"), z_cpu.to("cuda:0")
        p1, s1, z1 = p_cpu.to("cuda:1"), s_cpu.to("cuda:1"), z_cpu.to("cuda:1")

        out0 = t4_kernels.dequantize_s3(p0, s0, z0)
        out1 = t4_kernels.dequantize_s3(p1, s1, z1)

        torch.cuda.synchronize("cuda:0")
        torch.cuda.synchronize("cuda:1")

        out0_cpu = out0.cpu()
        out1_cpu = out1.cpu()

        assert torch.equal(out0_cpu.view(torch.int16), out1_cpu.view(torch.int16))
        assert torch.max(torch.abs(out0_cpu - out1_cpu)).item() == 0.0

    def test_dequantize_fp8_bit_identical(self):
        torch.manual_seed(888)
        n_words = 1024
        p_cpu = torch.randint(-2147483648, 2147483647, (n_words,), dtype=torch.int32)
        s_cpu = torch.rand(n_words, dtype=torch.float16) * 0.1

        p0, s0 = p_cpu.to("cuda:0"), s_cpu.to("cuda:0")
        p1, s1 = p_cpu.to("cuda:1"), s_cpu.to("cuda:1")

        out0 = t4_kernels.dequantize_fp8(p0, s0)
        out1 = t4_kernels.dequantize_fp8(p1, s1)

        torch.cuda.synchronize("cuda:0")
        torch.cuda.synchronize("cuda:1")

        out0_cpu = out0.cpu()
        out1_cpu = out1.cpu()

        assert torch.equal(out0_cpu.view(torch.int16), out1_cpu.view(torch.int16))
        assert torch.max(torch.abs(out0_cpu - out1_cpu)).item() == 0.0

    @pytest.mark.parametrize("shape,group_size", [
        ((1, 256, 128), 0),      # Decode vector (per-channel)
        ((1, 896, 896), 128),    # 7B/1.5B layer shape (group=128)
        ((4, 512, 256), 128),    # Batched decode (group=128)
    ])
    def test_fused_gemm_u4_bit_identical(self, shape, group_size):
        torch.manual_seed(999)
        M, K, N = shape
        A_cpu = torch.randn(M, K, dtype=torch.float16)
        W_cpu = torch.randint(0, 2**31 - 1, (K // 8, N), dtype=torch.int32)

        if group_size > 0:
            num_groups = K // group_size
            s_cpu = torch.randn(num_groups, N, dtype=torch.float16) * 0.05
            z_cpu = torch.randint(0, 16, (num_groups, N), dtype=torch.float16)
        else:
            s_cpu = torch.randn(1, N, dtype=torch.float16) * 0.05
            z_cpu = torch.randint(0, 16, (1, N), dtype=torch.float16)

        A0, W0, s0, z0 = A_cpu.to("cuda:0"), W_cpu.to("cuda:0"), s_cpu.to("cuda:0"), z_cpu.to("cuda:0")
        A1, W1, s1, z1 = A_cpu.to("cuda:1"), W_cpu.to("cuda:1"), s_cpu.to("cuda:1"), z_cpu.to("cuda:1")

        torch.cuda.synchronize("cuda:0")
        torch.cuda.synchronize("cuda:1")

        out0 = t4_kernels.fused_w4a16_gemm_u4(A0, W0, s0, z0, group_size)
        out1 = t4_kernels.fused_w4a16_gemm_u4(A1, W1, s1, z1, group_size)

        torch.cuda.synchronize("cuda:0")
        torch.cuda.synchronize("cuda:1")

        out0_cpu = out0.cpu()
        out1_cpu = out1.cpu()

        assert torch.equal(out0_cpu.view(torch.int16), out1_cpu.view(torch.int16)), \
            f"Bit mismatch on shape {shape} group_size {group_size}"
        assert torch.max(torch.abs(out0_cpu - out1_cpu)).item() == 0.0

    @pytest.mark.parametrize("shape,group_size", [
        ((1, 256, 128), 0),
        ((1, 896, 896), 128),
        ((4, 512, 256), 128),
    ])
    def test_fused_gemm_s4_bit_identical(self, shape, group_size):
        torch.manual_seed(1001)
        M, K, N = shape
        A_cpu = torch.randn(M, K, dtype=torch.float16)
        W_cpu = torch.randint(0, 2**31 - 1, (K // 8, N), dtype=torch.int32)

        if group_size > 0:
            num_groups = K // group_size
            s_cpu = torch.randn(num_groups, N, dtype=torch.float16) * 0.05
            z_cpu = torch.randint(0, 16, (num_groups, N), dtype=torch.float16)
        else:
            s_cpu = torch.randn(1, N, dtype=torch.float16) * 0.05
            z_cpu = torch.randint(0, 16, (1, N), dtype=torch.float16)

        A0, W0, s0, z0 = A_cpu.to("cuda:0"), W_cpu.to("cuda:0"), s_cpu.to("cuda:0"), z_cpu.to("cuda:0")
        A1, W1, s1, z1 = A_cpu.to("cuda:1"), W_cpu.to("cuda:1"), s_cpu.to("cuda:1"), z_cpu.to("cuda:1")

        torch.cuda.synchronize("cuda:0")
        torch.cuda.synchronize("cuda:1")

        out0 = t4_kernels.fused_w4a16_gemm_s4(A0, W0, s0, z0, group_size)
        out1 = t4_kernels.fused_w4a16_gemm_s4(A1, W1, s1, z1, group_size)

        torch.cuda.synchronize("cuda:0")
        torch.cuda.synchronize("cuda:1")

        out0_cpu = out0.cpu()
        out1_cpu = out1.cpu()

        assert torch.equal(out0_cpu.view(torch.int16), out1_cpu.view(torch.int16)), \
            f"Bit mismatch on signed shape {shape}"
        assert torch.max(torch.abs(out0_cpu - out1_cpu)).item() == 0.0

    def test_fused_h17_gemv_s3_bit_identical(self):
        torch.manual_seed(1002)
        M, K, N = 1, 100, 64
        A_cpu = torch.randn(M, K, dtype=torch.float16)
        W_cpu = torch.randint(0, 2**31 - 1, (K // 10, N), dtype=torch.int32)
        scales_cpu = torch.randn(N, dtype=torch.float16) * 0.05
        zps_cpu = torch.zeros(N, dtype=torch.float16)

        A0, W0, s0, z0 = A_cpu.to("cuda:0"), W_cpu.to("cuda:0"), scales_cpu.to("cuda:0"), zps_cpu.to("cuda:0")
        A1, W1, s1, z1 = A_cpu.to("cuda:1"), W_cpu.to("cuda:1"), scales_cpu.to("cuda:1"), zps_cpu.to("cuda:1")

        torch.cuda.synchronize("cuda:0")
        torch.cuda.synchronize("cuda:1")

        out0 = t4_kernels.fused_h17_gemv_s3(A0, W0, s0, z0)
        out1 = t4_kernels.fused_h17_gemv_s3(A1, W1, s1, z1)

        torch.cuda.synchronize("cuda:0")
        torch.cuda.synchronize("cuda:1")

        out0_cpu = out0.cpu()
        out1_cpu = out1.cpu()

        assert torch.equal(out0_cpu.view(torch.int16), out1_cpu.view(torch.int16))
        assert torch.max(torch.abs(out0_cpu - out1_cpu)).item() == 0.0

    def test_fused_backward_gemm_adamw_bit_identical(self):
        torch.manual_seed(1003)
        K, M, N = 64, 64, 64
        dY_cpu = torch.randn(K, M, dtype=torch.float16)
        X_cpu = torch.randn(K, N, dtype=torch.float16)
        W_cpu = torch.randn(M, N, dtype=torch.float32)

        W0 = W_cpu.clone().to("cuda:0")
        W1 = W_cpu.clone().to("cuda:1")
        W_act0 = W0.half().clone()
        W_act1 = W1.half().clone()
        m0 = torch.zeros(M, N, dtype=torch.float32, device="cuda:0")
        m1 = torch.zeros(M, N, dtype=torch.float32, device="cuda:1")
        v0 = torch.zeros(M, N, dtype=torch.float32, device="cuda:0")
        v1 = torch.zeros(M, N, dtype=torch.float32, device="cuda:1")

        t4_kernels.fused_backward_gemm_adamw(
            dY_cpu.to("cuda:0"), X_cpu.to("cuda:0"), W0, W_act0, m0, v0,
            1e-3, 0.9, 0.999, 1e-8, 0.01, 1.0, 1.0
        )
        t4_kernels.fused_backward_gemm_adamw(
            dY_cpu.to("cuda:1"), X_cpu.to("cuda:1"), W1, W_act1, m1, v1,
            1e-3, 0.9, 0.999, 1e-8, 0.01, 1.0, 1.0
        )

        torch.cuda.synchronize("cuda:0")
        torch.cuda.synchronize("cuda:1")

        assert torch.equal(W0.cpu(), W1.cpu())
        assert torch.equal(m0.cpu(), m1.cpu())

    def test_fused_ellie_rmsnorm_bit_identical(self):
        torch.manual_seed(1004)
        M, D = 4, 128
        x_cpu = torch.randn(M, D, dtype=torch.float16)
        g_cpu = torch.randn(D, dtype=torch.float16)

        out0 = t4_kernels.fused_ellie_rmsnorm(x_cpu.to("cuda:0"), g_cpu.to("cuda:0"))
        out1 = t4_kernels.fused_ellie_rmsnorm(x_cpu.to("cuda:1"), g_cpu.to("cuda:1"))

        torch.cuda.synchronize("cuda:0")
        torch.cuda.synchronize("cuda:1")

        assert torch.equal(out0.cpu().view(torch.int16), out1.cpu().view(torch.int16))

    def test_fused_ellie_swiglu_bit_identical(self):
        torch.manual_seed(1005)
        M, H = 4, 256
        g_cpu = torch.randn(M, H, dtype=torch.float16)
        u_cpu = torch.randn(M, H, dtype=torch.float16)

        out0 = t4_kernels.fused_ellie_swiglu(g_cpu.to("cuda:0"), u_cpu.to("cuda:0"))
        out1 = t4_kernels.fused_ellie_swiglu(g_cpu.to("cuda:1"), u_cpu.to("cuda:1"))

        torch.cuda.synchronize("cuda:0")
        torch.cuda.synchronize("cuda:1")

        assert torch.equal(out0.cpu().view(torch.int16), out1.cpu().view(torch.int16))

    def test_fused_ellie_rmsnorm_w4a16_gemv_swiglu_bit_identical(self):
        torch.manual_seed(1006)
        D, H = 256, 128
        group_size = 128
        num_groups = D // group_size

        x_cpu = torch.randn(1, D, dtype=torch.float16)
        g_cpu = torch.ones(D, dtype=torch.float16)
        Wg_cpu = torch.randint(0, 2**31 - 1, (D // 8, H), dtype=torch.int32)
        sg_cpu = torch.randn(num_groups, H, dtype=torch.float16) * 0.05
        zg_cpu = torch.zeros(num_groups, H, dtype=torch.float16)
        Wu_cpu = torch.randint(0, 2**31 - 1, (D // 8, H), dtype=torch.int32)
        su_cpu = torch.randn(num_groups, H, dtype=torch.float16) * 0.05
        zu_cpu = torch.zeros(num_groups, H, dtype=torch.float16)

        out0 = t4_kernels.fused_ellie_rmsnorm_w4a16_gemv_swiglu(
            x_cpu.to("cuda:0"), g_cpu.to("cuda:0"),
            Wg_cpu.to("cuda:0"), sg_cpu.to("cuda:0"), zg_cpu.to("cuda:0"),
            Wu_cpu.to("cuda:0"), su_cpu.to("cuda:0"), zu_cpu.to("cuda:0"),
            group_size, 1e-6
        )
        out1 = t4_kernels.fused_ellie_rmsnorm_w4a16_gemv_swiglu(
            x_cpu.to("cuda:1"), g_cpu.to("cuda:1"),
            Wg_cpu.to("cuda:1"), sg_cpu.to("cuda:1"), zg_cpu.to("cuda:1"),
            Wu_cpu.to("cuda:1"), su_cpu.to("cuda:1"), zu_cpu.to("cuda:1"),
            group_size, 1e-6
        )

        torch.cuda.synchronize("cuda:0")
        torch.cuda.synchronize("cuda:1")

        assert torch.equal(out0.cpu().view(torch.int16), out1.cpu().view(torch.int16))

    def test_fused_swiglu_backward_bit_identical(self):
        torch.manual_seed(1007)
        M, H = 4, 256
        dY_cpu = torch.randn(M, H, dtype=torch.float16)
        g_cpu = torch.randn(M, H, dtype=torch.float16)
        u_cpu = torch.randn(M, H, dtype=torch.float16)

        dg0, du0 = t4_kernels.fused_swiglu_backward(dY_cpu.to("cuda:0"), g_cpu.to("cuda:0"), u_cpu.to("cuda:0"))
        dg1, du1 = t4_kernels.fused_swiglu_backward(dY_cpu.to("cuda:1"), g_cpu.to("cuda:1"), u_cpu.to("cuda:1"))

        torch.cuda.synchronize("cuda:0")
        torch.cuda.synchronize("cuda:1")

        assert torch.equal(dg0.cpu().view(torch.int16), dg1.cpu().view(torch.int16))
        assert torch.equal(du0.cpu().view(torch.int16), du1.cpu().view(torch.int16))

    def test_fused_sft_lora_backward_adamw_bit_identical(self):
        torch.manual_seed(1008)
        M, d_in, d_out, r = 4, 32, 64, 16
        dY_cpu = torch.randn(M, d_out, dtype=torch.float16)
        X_cpu = torch.randn(M, d_in, dtype=torch.float16)
        H_cpu = torch.randn(M, r, dtype=torch.float16)
        A_cpu = torch.randn(r, d_in, dtype=torch.float32)
        B_cpu = torch.randn(d_out, r, dtype=torch.float32)

        A0, A1 = A_cpu.clone().to("cuda:0"), A_cpu.clone().to("cuda:1")
        A_act0, A_act1 = A0.half().clone(), A1.half().clone()
        mA0, mA1 = torch.zeros(r, d_in, dtype=torch.float32, device="cuda:0"), torch.zeros(r, d_in, dtype=torch.float32, device="cuda:1")
        vA0, vA1 = torch.zeros(r, d_in, dtype=torch.float32, device="cuda:0"), torch.zeros(r, d_in, dtype=torch.float32, device="cuda:1")

        B0, B1 = B_cpu.clone().to("cuda:0"), B_cpu.clone().to("cuda:1")
        B_act0, B_act1 = B0.half().clone(), B1.half().clone()
        mB0, mB1 = torch.zeros(d_out, r, dtype=torch.float32, device="cuda:0"), torch.zeros(d_out, r, dtype=torch.float32, device="cuda:1")
        vB0, vB1 = torch.zeros(d_out, r, dtype=torch.float32, device="cuda:0"), torch.zeros(d_out, r, dtype=torch.float32, device="cuda:1")

        dX0 = t4_kernels.fused_sft_lora_backward_adamw(
            dY_cpu.to("cuda:0"), X_cpu.to("cuda:0"), H_cpu.to("cuda:0"),
            A0, A_act0, mA0, vA0, B0, B_act0, mB0, vB0,
            1e-3, 0.9, 0.999, 1e-8, 0.01, 1.0, 1.0
        )
        dX1 = t4_kernels.fused_sft_lora_backward_adamw(
            dY_cpu.to("cuda:1"), X_cpu.to("cuda:1"), H_cpu.to("cuda:1"),
            A1, A_act1, mA1, vA1, B1, B_act1, mB1, vB1,
            1e-3, 0.9, 0.999, 1e-8, 0.01, 1.0, 1.0
        )

        torch.cuda.synchronize("cuda:0")
        torch.cuda.synchronize("cuda:1")

        assert torch.equal(dX0.cpu().view(torch.int16), dX1.cpu().view(torch.int16))
        assert torch.equal(A0.cpu(), A1.cpu())
        assert torch.equal(B0.cpu(), B1.cpu())



# ==============================================================================
# 5. Concurrent Multi-Threaded Execution Across Dual Devices
# ==============================================================================

@skip_if_no_dual_cuda
@skip_if_no_t4_kernels
class TestConcurrentDualDeviceExecution:
    """Verifies that concurrent multi-threaded execution across dual devices is thread-safe."""

    def test_concurrent_gemm_execution(self):
        errors = []

        def worker(dev_idx, iterations=20):
            try:
                device = torch.device(f"cuda:{dev_idx}")
                torch.cuda.set_device(dev_idx)
                for _ in range(iterations):
                    A = torch.randn(1, 128, dtype=torch.float16, device=device)
                    W = torch.randint(0, 2**31 - 1, (16, 64), dtype=torch.int32, device=device)
                    scales = torch.randn(1, 64, dtype=torch.float16, device=device) * 0.05
                    zps = torch.zeros(1, 64, dtype=torch.float16, device=device)
                    out = t4_kernels.fused_w4a16_gemm_u4(A, W, scales, zps, 0)
                    torch.cuda.synchronize(device)
                    assert out.device == device
            except Exception as e:
                errors.append((dev_idx, str(e)))

        t0 = threading.Thread(target=worker, args=(0,))
        t1 = threading.Thread(target=worker, args=(1,))

        t0.start()
        t1.start()
        t0.join()
        t1.join()

        assert len(errors) == 0, f"Concurrent execution failed with errors: {errors}"


# ==============================================================================
# Standalone CLI Test Runner
# ==============================================================================

def main():
    print("=" * 80)
    print("Multi-Device CUDAGuard & Stream Safety Test Suite")
    print(f"CUDA Available: {CUDA_AVAILABLE} | Devices Found: {NUM_DEVICES}")
    if CUDA_AVAILABLE and NUM_DEVICES >= 1:
        for idx in range(NUM_DEVICES):
            print(f"  Device {idx}: {torch.cuda.get_device_name(idx)}")
    print(f"t4_kernels Extension Installed: {HAS_T4_KERNELS}")
    print("=" * 80)

    if not HAS_T4_KERNELS:
        print("[SKIP] t4_kernels extension not found. Build with 'pip install -e src'")
        sys.exit(0)

    if not CUDA_AVAILABLE or NUM_DEVICES < 1:
        print("[SKIP] No CUDA hardware detected. Skipping all tests cleanly.")
        sys.exit(0)

    if NUM_DEVICES < 2:
        print("[NOTICE] Single CUDA device detected. Running single-device tests on cuda:0.")
        print("Dual-device tests (cuda:1, cross-device negative tests) will be cleanly skipped with [SKIP].")

    exit_code = pytest.main([__file__, "-v", "-s"])
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
