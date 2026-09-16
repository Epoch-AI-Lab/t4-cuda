#!/usr/bin/env python3
"""
Test Suite for Ellie 4B Custom T4 CUDA Kernels
- Tests Fused RMSNorm vs PyTorch reference
- Tests Fused SwiGLU vs PyTorch reference
- Tests Fused RMSNorm + W4A16 GEMV (LOP3 0x6A) + SwiGLU Mega-Kernel
- Includes simulation mode when running on CPU/host and hardware execution when running on NVIDIA Tesla T4.
"""

import math
import time
import torch
import torch.nn.functional as F
import pytest

def pack_int4_signed(tensor_2d, group_size=128):
    """
    Packs a 2D float tensor (D x H) into column-major INT4 (D/8 x H) with per-group scale and zero point.
    """
    D, H = tensor_2d.shape
    assert D % 8 == 0, "D must be multiple of 8"
    assert D % group_size == 0, "D must be multiple of group_size"

    num_groups = D // group_size
    scales = torch.empty((num_groups, H), dtype=torch.float16)
    zero_points = torch.empty((num_groups, H), dtype=torch.float16)
    packed = torch.zeros((D // 8, H), dtype=torch.int32)

    for g in range(num_groups):
        sub = tensor_2d[g*group_size : (g+1)*group_size, :] # (group_size x H)
        max_val = sub.max(dim=0).values
        min_val = sub.min(dim=0).values
        
        # S4 dynamic range [-8, 7]
        scale = (max_val - min_val).clamp(min=1e-5) / 15.0
        zp = min_val + 8.0 * scale
        
        scales[g, :] = scale.to(torch.float16)
        zero_points[g, :] = zp.to(torch.float16)

        # Quantize to [-8, 7]
        q = torch.clamp(torch.round((sub - zp) / scale), -8, 7).to(torch.int32) & 0x0F # 4-bit unsigned representation

        # Pack 8 values along D into one uint32
        for k in range(group_size // 8):
            k_global = (g * group_size // 8) + k
            word = torch.zeros(H, dtype=torch.int32)
            for nibble in range(8):
                elem = q[k * 8 + nibble, :]
                word = word | (elem << (nibble * 4))
            packed[k_global, :] = word

    return packed, scales, zero_points

def reference_ellie_rmsnorm(x, gamma, eps=1e-6):
    """PyTorch Reference RMSNorm"""
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * gamma

def reference_ellie_swiglu(gate, up):
    """PyTorch Reference SwiGLU"""
    return F.silu(gate.float()).half() * up

def reference_dequant_s4(packed, scales, zero_points, group_size=128):
    """PyTorch Reference S4 Dequantization"""
    num_words, H = packed.shape
    D = num_words * 8
    num_groups = D // group_size
    dequant = torch.empty((D, H), dtype=torch.float16)

    for g in range(num_groups):
        scale = scales[g, :].float()
        zp = zero_points[g, :].float()
        for k in range(group_size // 8):
            word_idx = g * (group_size // 8) + k
            word = packed[word_idx, :]
            for nibble in range(8):
                val_u4 = (word >> (nibble * 4)) & 0x0F
                # Sign extend 4-bit to signed int [-8, 7]
                val_s4 = torch.where(val_u4 >= 8, val_u4 - 16, val_u4).float()
                real_val = val_s4 * scale + zp
                dequant[g * group_size + k * 8 + nibble, :] = real_val.half()

    return dequant

def run_tests():
    print("=" * 70)
    print("  ELLIE 4B CUSTOM T4 CUDA KERNEL SUITE: VERIFICATION & AUDIT")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Compute Device: {device.upper()}")

    # Ellie 4B Dimensions (Qwen 3.5 / 3.6-4B architecture)
    D = 2560   # Hidden Dimension
    H = 6912   # SwiGLU Intermediate Dimension
    group_size = 128
    eps = 1e-6

    print(f"[*] Architecture Dimensions: Hidden D={D}, FFN Intermediate H={H}, Quant Group={group_size}")

    # Generate synthetic activation and weight tensors
    torch.manual_seed(2026)
    x = torch.randn((1, D), dtype=torch.float16)
    gamma = torch.ones((D,), dtype=torch.float16) + 0.1 * torch.randn((D,), dtype=torch.float16)

    W_gate_raw = torch.randn((D, H), dtype=torch.float16) * 0.02
    W_up_raw   = torch.randn((D, H), dtype=torch.float16) * 0.02

    # Pack weights for W4A16 INT4 LOP3 dequant
    W_gate_packed, scale_gate, zp_gate = pack_int4_signed(W_gate_raw, group_size=group_size)
    W_up_packed, scale_up, zp_up = pack_int4_signed(W_up_raw, group_size=group_size)

    # 1. Reference Computation
    print("\n[1/3] Computing High-Precision PyTorch Reference...")
    norm_ref = reference_ellie_rmsnorm(x, gamma, eps=eps)
    
    W_gate_dequant_ref = reference_dequant_s4(W_gate_packed, scale_gate, zp_gate, group_size=group_size)
    W_up_dequant_ref   = reference_dequant_s4(W_up_packed, scale_up, zp_up, group_size=group_size)

    gate_act_ref = torch.matmul(norm_ref.float(), W_gate_dequant_ref.float()).half()
    up_act_ref   = torch.matmul(norm_ref.float(), W_up_dequant_ref.float()).half()
    swiglu_ref   = reference_ellie_swiglu(gate_act_ref, up_act_ref)

    print(f"  [✓] Reference Output Shape: {swiglu_ref.shape}")
    print(f"  [✓] Reference Output Stats: mean={swiglu_ref.float().mean().item():.5f}, std={swiglu_ref.float().std().item():.5f}")

    if device == "cuda":
        import t4_kernels
        print("\n[2/3] Executing Hardware Fused Ellie Kernels on CUDA...")
        x_cu = x.to(device)
        gamma_cu = gamma.to(device)
        W_gate_cu = W_gate_packed.to(device)
        scale_gate_cu = scale_gate.to(device)
        zp_gate_cu = zp_gate.to(device)
        W_up_cu = W_up_packed.to(device)
        scale_up_cu = scale_up.to(device)
        zp_up_cu = zp_up.to(device)

        # Test Standalone RMSNorm Kernel
        norm_cuda = t4_kernels.fused_ellie_rmsnorm(x_cu, gamma_cu, eps)
        diff_norm = (norm_cuda.cpu() - norm_ref).abs().max().item()
        print(f"  [✓] Fused RMSNorm Max Abs Diff: {diff_norm:.6e} (Tolerance: < 5e-3)")
        assert diff_norm < 5e-3, f"RMSNorm diff {diff_norm} exceeds tolerance!"

        # Test Standalone SwiGLU Kernel
        gate_cu = gate_act_ref.to(device)
        up_cu = up_act_ref.to(device)
        swiglu_cuda = t4_kernels.fused_ellie_swiglu(gate_cu, up_cu)
        diff_swiglu = (swiglu_cuda.cpu() - swiglu_ref).abs().max().item()
        print(f"  [✓] Fused SwiGLU Max Abs Diff: {diff_swiglu:.6e} (Tolerance: < 5e-3)")
        assert diff_swiglu < 5e-3, f"SwiGLU diff {diff_swiglu} exceeds tolerance!"

        # Test Fused Mega-Kernel (RMSNorm + Dual W4A16 GEMV + SwiGLU)
        fused_out = t4_kernels.fused_ellie_rmsnorm_w4a16_gemv_swiglu(
            x_cu, gamma_cu,
            W_gate_cu, scale_gate_cu, zp_gate_cu,
            W_up_cu, scale_up_cu, zp_up_cu,
            group_size, eps
        )
        diff_mega_max = (fused_out.cpu() - swiglu_ref).abs().max().item()
        diff_mega_mean = (fused_out.cpu() - swiglu_ref).abs().mean().item()
        cos_sim = torch.nn.functional.cosine_similarity(
            fused_out.cpu().flatten().float(),
            swiglu_ref.flatten().float(),
            dim=0,
        ).item()
        print(f"  [✓] Fused Mega-Kernel Max Abs Diff: {diff_mega_max:.4f} (Gate: < 0.15), Mean Abs Diff: {diff_mega_mean:.4f} (Gate: < 0.02), Cos Sim: {cos_sim:.4f}")
        assert diff_mega_max < 0.15, f"Mega-kernel max diff {diff_mega_max} exceeds tolerance 0.15!"
        assert diff_mega_mean < 0.02, f"Mega-kernel mean diff {diff_mega_mean} exceeds tolerance 0.02!"
        assert cos_sim >= 0.999, f"Mega-kernel cosine similarity {cos_sim} < 0.999!"


        print("\n[3/3] Benchmarking Kernel Latency on Tesla T4...")
        torch.cuda.synchronize()
        # Warmup
        for _ in range(50):
            _ = t4_kernels.fused_ellie_rmsnorm_w4a16_gemv_swiglu(
                x_cu, gamma_cu,
                W_gate_cu, scale_gate_cu, zp_gate_cu,
                W_up_cu, scale_up_cu, zp_up_cu,
                group_size, eps
            )
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(200):
            _ = t4_kernels.fused_ellie_rmsnorm_w4a16_gemv_swiglu(
                x_cu, gamma_cu,
                W_gate_cu, scale_gate_cu, zp_gate_cu,
                W_up_cu, scale_up_cu, zp_up_cu,
                group_size, eps
            )
        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end) / 200.0

        # Compute Memory Traffic & Throughput
        # Unfused reads: x (5KB) + gamma (5KB) + W_gate (35MB FP16) + W_up (35MB FP16) + intermediates (27KB) = ~70 MB
        dram_bytes = (D * H * 2) * 0.5 + (D * 2) # ~17.68 MB
        achieved_gbps = (dram_bytes / (elapsed_ms / 1000.0)) / 1e9
        print(f"  [⚡] Fused Mega-Kernel Token Latency: {elapsed_ms:.3f} ms")
        print(f"  [⚡] Effective Memory Bandwidth: {achieved_gbps:.1f} GB/s ({achieved_gbps/320.0*100:.1f}% of T4 Roofline)")

    else:
        print("\n[2/3] [SKIP] CUDA device or t4_kernels not available. Running CPU bitwise verification...")
        # Validate mathematical properties of LOP3 0x6A dequantization
        # LUT 0x6A = (A ^ C) & B | C & ~B where B=0xF, C=0x6408 (FP16 1032.0)
        # Check all 16 signed 4-bit nibbles [-8..7]
        for nib in range(16):
            s4_val = nib - 16 if nib >= 8 else nib
            c = 0x6408
            b = 0x000F
            a = nib
            raw = ((a ^ c) & b) | (c & (~b & 0xFFFF))
            f16_val = torch.tensor(raw, dtype=torch.int16).view(torch.float16).item()
            recovered = f16_val - 1032.0
            assert abs(recovered - s4_val) < 1e-4, f"Mismatch on nibble {nib}: got {recovered}, expected {s4_val}"

        print("  [✓] LOP3.b32 0x6A Logic: 16/16 Bit-Exact Signed INT4 States Verified on CPU!")
        print("  [SKIP] CUDA Kernel Latency and Hardware Benchmarks Skipped (No GPU/Extension).")
        print("\n" + "=" * 70)
        print("  ELLIE 4B CPU BITWISE VERIFICATION PASSED (CUDA Kernels Skipped)")
        print("=" * 70)
        return

    print("\n" + "=" * 70)
    print("  ALL ELLIE 4B CUSTOM T4 KERNEL VERIFICATIONS: PASSED (100%)")
    print("=" * 70)


def test_ellie_lop3_bitwise_cpu():
    """Validate mathematical properties of LOP3 0x6A dequantization on CPU."""
    for nib in range(16):
        s4_val = nib - 16 if nib >= 8 else nib
        c = 0x6408
        b = 0x000F
        a = nib
        raw = ((a ^ c) & b) | (c & (~b & 0xFFFF))
        f16_val = torch.tensor(raw, dtype=torch.int16).view(torch.float16).item()
        recovered = f16_val - 1032.0
        assert abs(recovered - s4_val) < 1e-4, f"Mismatch on nibble {nib}: got {recovered}, expected {s4_val}"


def test_ellie_custom_t4_cuda_kernels():
    """Live CUDA verification for Ellie custom T4 kernels."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA device not available")
    try:
        import t4_kernels
    except ImportError:
        pytest.skip("t4_kernels C++ extension not available")
    run_tests()


if __name__ == "__main__":
    run_tests()
