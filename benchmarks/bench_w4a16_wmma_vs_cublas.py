#!/usr/bin/env python3
"""Benchmark W4A16 Dual-Path Kernel (GEMV + WMMA Tensor Core) vs PyTorch cuBLAS.

Tests M in [1, 4, 8, 16, 32, 64] on Qwen2.5-0.5B matrix shapes:
- Self-attention shape: K=896, N=896
- MLP intermediate shape: K=896, N=4864

Measures:
1. Numerical accuracy vs exact dequantized reference
2. Latency (microseconds, median of 100 timed iterations after warmup)
3. Speedup factor vs PyTorch cuBLAS FP16
4. Effective TFLOPS
"""

import time
import torch
import torch.nn.functional as F

try:
    import t4_kernels
except ImportError:
    print("Error: t4_kernels not installed.")
    exit(1)

def quantize_weight_sym(W, group_size=128):
    out_f, in_f = W.shape
    assert in_f % 8 == 0, in_f
    Wf = W.float()
    if group_size > 0 and in_f % group_size == 0:
        num_groups = in_f // group_size
        Wf_grouped = Wf.reshape(out_f, num_groups, group_size)
        amax = Wf_grouped.abs().amax(dim=2, keepdim=True).clamp_min(1e-8)
        scale = amax / 7.0
        q = torch.clamp(torch.round(Wf_grouped / scale) + 8, 0, 15).reshape(out_f, in_f).to(torch.int32)
        q = q.t().contiguous().cpu()
        packed = torch.zeros(in_f // 8, out_f, dtype=torch.int32)
        for i in range(8):
            packed |= q[i::8, :] << (4 * i)
        scales = scale.squeeze(2).t().contiguous().half().cuda()
        zps = torch.full_like(scales, 8.0).cuda()
        return packed.cuda(), scales, zps, group_size
    else:
        amax = Wf.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
        scale = amax / 7.0
        q = torch.clamp(torch.round(Wf / scale) + 8, 0, 15).to(torch.int32)
        q = q.t().contiguous().cpu()
        packed = torch.zeros(in_f // 8, out_f, dtype=torch.int32)
        for i in range(8):
            packed |= q[i::8, :] << (4 * i)
        scales = scale.squeeze(1).half().unsqueeze(0).contiguous().cuda()
        zps = torch.full((1, out_f), 8.0, dtype=torch.float16, device="cuda")
        return packed.cuda(), scales, zps, 0

def dequant_ref(packed, sc, zp, group_size=128):
    in_f_8, out_f = packed.shape
    in_f = in_f_8 * 8
    Wq = torch.zeros(in_f, out_f, dtype=torch.float32)
    pc = packed.cpu()
    for i in range(8):
        Wq[i::8, :] = ((pc >> (4 * i)) & 0xF).float()
    sc_cpu = sc.float().cpu()
    zz_cpu = zp.float().cpu()
    if group_size > 0:
        num_groups = in_f // group_size
        Wq_grouped = Wq.reshape(num_groups, group_size, out_f)
        sc_3d = sc_cpu.unsqueeze(1)
        zz_3d = zz_cpu.unsqueeze(1)
        W_deq = ((Wq_grouped - zz_3d) * sc_3d).reshape(in_f, out_f).half()
    else:
        W_deq = ((Wq - zz_cpu) * sc_cpu).reshape(in_f, out_f).half()
    return W_deq.cuda()

def time_op(fn, iters=100, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6 # microseconds

def run_bench():
    torch.manual_seed(42)
    print("=" * 80)
    print("   Tesla T4: W4A16 (GEMV + WMMA Tensor Core) vs cuBLAS FP16 Benchmark")
    print("=" * 80)
    print(f"{'Shape (M x K x N)':<24} | {'cuBLAS (us)':<12} | {'W4A16 (us)':<12} | {'Speedup':<9} | {'TFLOPS':<8} | {'Max Err':<8}")
    print("-" * 80)

    shapes = [
        # Attention shapes (K=896, N=896)
        (1, 896, 896),
        (4, 896, 896),
        (8, 896, 896),
        (16, 896, 896),
        (32, 896, 896),
        (64, 896, 896),
        # MLP shapes (K=896, N=4864)
        (1, 896, 4864),
        (4, 896, 4864),
        (8, 896, 4864),
        (16, 896, 4864),
        (32, 896, 4864),
        (64, 896, 4864),
    ]

    for M, K, N in shapes:
        A = torch.randn(M, K, dtype=torch.half, device="cuda")
        W_fp16 = torch.randn(N, K, dtype=torch.half, device="cuda")

        packed, scales, zps, g_size = quantize_weight_sym(W_fp16, group_size=128)
        W_dequant = dequant_ref(packed, scales, zps, group_size=g_size)

        # Accuracy check vs dequantized reference
        C_ref = torch.matmul(A.float(), W_dequant.float()).half()
        C_w4 = t4_kernels.fused_w4a16_gemm_u4(A, packed, scales, zps, g_size)
        max_err = (C_w4.float() - C_ref.float()).abs().max().item()

        # Timing
        fn_cublas = lambda: torch.matmul(A, W_fp16.t())
        fn_w4 = lambda: t4_kernels.fused_w4a16_gemm_u4(A, packed, scales, zps, g_size)

        t_cublas = time_op(fn_cublas)
        t_w4 = time_op(fn_w4)
        speedup = t_cublas / t_w4
        flops = 2.0 * M * K * N
        tflops_w4 = (flops / (t_w4 * 1e-6)) / 1e12

        shape_str = f"{M} x {K} x {N}"
        print(f"{shape_str:<24} | {t_cublas:10.1f} us | {t_w4:10.1f} us | {speedup:7.2f}x | {tflops_w4:6.2f}   | {max_err:7.4f}")

    print("=" * 80)

if __name__ == "__main__":
    run_bench()
