#!/usr/bin/env python3
"""3-Way Head-to-Head Benchmark: cuBLAS FP16 vs Custom Pure INT4 vs CP-Hybrid.

Benchmarks across universal model dimensions:
- Qwen2.5-0.5B: K=896, N=896 (Attn) and N=4864 (MLP)
- Llama-3-8B / Mistral-7B: K=4096, N=4096 (Attn) and N=14336 (MLP)
- Gemma-2-2B: K=2048, N=2048 (Attn) and N=16384 (MLP)

Measures across M in [1, 4, 8, 16, 32, 64]:
- cuBLAS FP16 latency (us)
- Custom Pure INT4 latency (us)
- CP-Hybrid latency (us)
- Hybrid vs cuBLAS speedup
- Numerical correctness / max error
"""

import sys
import os
import time
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.hybrid_linear import HybridLinear, quantize_weight_sym_int4

try:
    import t4_kernels
except ImportError:
    print("Error: t4_kernels not installed.")
    sys.exit(1)

def time_op(fn, iters=50, warmup=15):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6 # microseconds

def run_3way_benchmark():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA required for benchmark.")
        return

    print("=" * 105)
    print("      3-WAY HEAD-TO-HEAD BENCHMARK: cuBLAS FP16 vs Pure INT4 vs CP-Hybrid (Tesla T4)")
    print("=" * 105)
    print(f"{'Model / Layer':<20} | {'Shape (M x K x N)':<18} | {'cuBLAS':<10} | {'Pure INT4':<10} | {'CP-Hybrid':<10} | {'Speedup':<9} | {'Max Err':<8}")
    print("-" * 105)

    test_configs = [
        # (ModelName, LayerType, K, N)
        ("Qwen-0.5B Attn", "attn", 896, 896),
        ("Qwen-0.5B MLP",  "mlp",  896, 4864),
        ("Llama-3-8B Attn", "attn", 4096, 4096),
        ("Llama-3-8B MLP",  "mlp",  4096, 14336),
        ("Gemma-2-2B Attn", "attn", 2048, 2048),
        ("Gemma-2-2B MLP",  "mlp",  2048, 16384),
    ]

    batch_sizes = [1, 4, 8, 16, 32, 64]

    for model_label, layer_type, K, N in test_configs:
        # Create standard PyTorch linear
        lin = nn.Linear(K, N, bias=True).half().to(device)
        packed, scales, zps, g_size = quantize_weight_sym_int4(lin.weight, group_size=128)
        hybrid_lin = HybridLinear(lin, group_size=128).to(device)

        for M in batch_sizes:
            x = torch.randn(M, K, dtype=torch.half, device=device)

            # 1. cuBLAS FP16
            fn_cublas = lambda: lin(x)
            t_cublas = time_op(fn_cublas)

            # 2. Pure INT4 Custom Kernel
            fn_pure_int4 = lambda: t4_kernels.fused_w4a16_gemm_u4(x, packed, scales, zps, g_size) + lin.bias
            t_pure_int4 = time_op(fn_pure_int4)

            # 3. CP-Hybrid Layer
            fn_hybrid = lambda: hybrid_lin(x)
            t_hybrid = time_op(fn_hybrid)

            speedup = t_cublas / t_hybrid
            out_hybrid = hybrid_lin(x)
            out_cublas = lin(x)
            max_err = (out_hybrid - out_cublas).abs().max().item()

            shape_str = f"{M}x{K}x{N}"
            print(f"{model_label:<20} | {shape_str:<18} | {t_cublas:7.1f} us | {t_pure_int4:7.1f} us | {t_hybrid:7.1f} us | {speedup:6.2f}x   | {max_err:7.4f}")

        print("-" * 105)

    print("=" * 105)

if __name__ == "__main__":
    run_3way_benchmark()
