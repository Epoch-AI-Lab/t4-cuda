#!/usr/bin/env python3
"""3-Way Head-to-Head Kernel Benchmark: t4_kernels vs bitsandbytes vs Marlin.

Evaluates 4-bit low-precision projection execution on NVIDIA Turing (sm_75, Tesla T4):
1. Custom Kernel: t4_kernels (W4A16 GEMV / WMMA Tensor Core)
2. Industry Baseline 1: bitsandbytes (Linear4bit NF4 / FP4)
3. Industry Baseline 2: Marlin (W4A16 FP16xINT4 GEMM via AutoGPTQ or marlin)
4. Unquantized Reference: PyTorch cuBLAS FP16

Evaluates across:
- Decode regime (M = 1, 4): Memory bandwidth bound GEMV
- Batched / Prefill regime (M = 16, 64, 256): Compute bound GEMM
- Production model shapes: Qwen2.5-7B (Gate/Up, Down, QKV, TP-sharded) and Llama-3-8B

Strict benchmarking protocol:
- Authentic CUDA event timing with torch.cuda.synchronize()
- Real warmup (>= 20) and repeated timed iterations (>= 50)
- Effective memory bandwidth (GB/s against T4 320 GB/s peak)
- Computational throughput (TFLOP/s against T4 65 TFLOP/s peak)
- Numerical fidelity (Cosine similarity and Max Error vs FP16 ground truth)
- Zero mocked passes or hardcoded numbers
- Machine-readable JSON logs containing individual iteration times and device info
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add repository root and src to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
for p in (REPO_ROOT, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

# Hardware baseline specifications (NVIDIA Tesla T4 sm_75)
T4_PEAK_BANDWIDTH_GB_S = 320.0
T4_PEAK_FP16_TC_TFLOPS = 65.0

# Try importing custom kernel
try:
    import t4_kernels
    HAS_T4_KERNELS = True
except ImportError:
    t4_kernels = None
    HAS_T4_KERNELS = False

# Try importing bitsandbytes
try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    bnb = None
    HAS_BNB = False

# Try importing Marlin (via auto_gptq or standalone marlin)
HAS_MARLIN = False
marlin_module = None
try:
    import marlin
    marlin_module = marlin
    HAS_MARLIN = True
except ImportError:
    try:
        from auto_gptq.nn_modules.qlinear import qlinear_marlin
        marlin_module = qlinear_marlin
        HAS_MARLIN = True
    except ImportError:
        HAS_MARLIN = False

# Try importing CP-Hybrid helper from repo
try:
    from src.hybrid_linear import quantize_weight_sym_int4
    HAS_QUANT_HELPER = True
except ImportError:
    HAS_QUANT_HELPER = False


def fallback_quantize_sym_int4(W: torch.Tensor, group_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Fallback symmetric INT4 quantization if src.hybrid_linear is not importable."""
    out_f, in_f = W.shape
    assert in_f % 8 == 0, f"in_features ({in_f}) must be divisible by 8"
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
        scales = scale.squeeze(2).t().contiguous().half().to(W.device)
        zps = torch.full_like(scales, 8.0)
        return packed.to(W.device), scales, zps, group_size
    else:
        amax = Wf.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
        scale = amax / 7.0
        q = torch.clamp(torch.round(Wf / scale) + 8, 0, 15).to(torch.int32)
        q = q.t().contiguous().cpu()
        packed = torch.zeros(in_f // 8, out_f, dtype=torch.int32)
        for i in range(8):
            packed |= q[i::8, :] << (4 * i)
        scales = scale.squeeze(1).half().unsqueeze(0).contiguous().to(W.device)
        zps = torch.full((1, out_f), 8.0, dtype=torch.float16, device=W.device)
        return packed.to(W.device), scales, zps, 0


def prepare_marlin_linear(W: torch.Tensor, group_size: int = 128) -> Optional[Callable[[torch.Tensor], torch.Tensor]]:
    """Prepares a callable Marlin forward function if Marlin is available."""
    if not HAS_MARLIN or marlin_module is None:
        return None
    out_f, in_f = W.shape
    if in_f % 128 != 0 or out_f % 64 != 0:
        return None
    try:
        # Check for AutoGPTQ QuantLinear
        if hasattr(marlin_module, "QuantLinear"):
            layer = marlin_module.QuantLinear(bits=4, group_size=group_size, infeatures=in_f, outfeatures=out_f, bias=False).to(W.device)
            return lambda x: layer(x)
        # Check for standalone marlin quantize_and_pack + mul
        if hasattr(marlin_module, "quantize_and_pack") and hasattr(marlin_module, "mul"):
            packed_b, scales = marlin_module.quantize_and_pack(W, group_size=group_size)
            workspace = torch.zeros(out_f // 128 * 16, dtype=torch.int32, device=W.device)
            def _marlin_call(x: torch.Tensor) -> torch.Tensor:
                out = torch.empty((x.shape[0], out_f), dtype=torch.float16, device=x.device)
                marlin_module.mul(x, packed_b, out, scales, workspace)
                return out
            return _marlin_call
    except Exception:
        pass
    return None


def benchmark_cuda_op(fn: Callable[[], Any], iters: int = 50, warmup: int = 20) -> Dict[str, Any]:
    """Measures kernel execution time using CUDA events with warmup and synchronization."""
    if not torch.cuda.is_available():
        # Dry-run fallback on CPU
        for _ in range(warmup):
            fn()
        timings_us: List[float] = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            timings_us.append((time.perf_counter() - t0) * 1e6)
        timings_us.sort()
        n = len(timings_us)
        return {
            "raw_timings_us": timings_us,
            "median_us": timings_us[n // 2],
            "p95_us": timings_us[min(int(0.95 * n), n - 1)],
            "min_us": timings_us[0],
            "max_us": timings_us[-1],
        }

    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Timed runs using individual CUDA event pairs to compute percentiles
    timings_ms: List[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        timings_ms.append(start.elapsed_time(end))

    timings_ms.sort()
    n = len(timings_ms)
    median_ms = timings_ms[n // 2]
    p95_idx = min(int(0.95 * n), n - 1)
    p95_ms = timings_ms[p95_idx]
    min_ms = timings_ms[0]
    max_ms = timings_ms[-1]

    return {
        "raw_timings_us": [t * 1e3 for t in timings_ms],
        "median_us": median_ms * 1e3,
        "p95_us": p95_ms * 1e3,
        "min_us": min_ms * 1e3,
        "max_us": max_ms * 1e3,
    }


def compute_metrics(
    latency_us: float,
    M: int,
    K: int,
    N: int,
    is_4bit: bool = True,
    group_size: int = 128,
) -> Tuple[float, float]:
    """Computes effective memory bandwidth (GB/s) and compute throughput (TFLOP/s)."""
    latency_sec = max(latency_us * 1e-6, 1e-12)

    # Arithmetic FLOPs for GEMM/GEMV: 2 * M * N * K
    flops = 2.0 * M * N * K
    tflops = (flops / latency_sec) / 1e12

    # Memory bytes transferred:
    # Input X: M * K * 2 bytes (FP16)
    # Output Y: M * N * 2 bytes (FP16)
    # Weight W:
    if is_4bit:
        weight_bytes = (K * N) * 0.5
        if group_size > 0:
            num_groups = (K + group_size - 1) // group_size
            weight_bytes += num_groups * N * 4
    else:
        weight_bytes = K * N * 2.0

    total_bytes = (M * K * 2.0) + (M * N * 2.0) + weight_bytes
    bandwidth_gb_s = (total_bytes / latency_sec) / 1e9

    return bandwidth_gb_s, tflops


def compute_numerical_parity(
    out_test: torch.Tensor,
    out_ref: torch.Tensor,
) -> Tuple[float, float]:
    """Computes max absolute error and cosine similarity against reference tensor."""
    with torch.no_grad():
        a = out_test.float().flatten()
        b = out_ref.float().flatten()
        max_err = (a - b).abs().max().item()
        cos_sim = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    return max_err, cos_sim


def get_benchmark_shapes() -> List[Dict[str, Any]]:
    """Defines matrix shapes across target LLM architectures and parallel sharding."""
    return [
        # --- Qwen2.5-7B Shapes (D = 3584, H = 18944) ---
        {
            "category": "Qwen2.5-7B Full",
            "name": "Qwen-7B Gate/Up Proj",
            "layer_type": "mlp_up",
            "K": 3584,
            "N": 18944,
        },
        {
            "category": "Qwen2.5-7B Full",
            "name": "Qwen-7B Down Proj",
            "layer_type": "mlp_down",
            "K": 18944,
            "N": 3584,
        },
        {
            "category": "Qwen2.5-7B Full",
            "name": "Qwen-7B QKV / Q-Proj",
            "layer_type": "attn_qkv",
            "K": 3584,
            "N": 3584,
        },
        {
            "category": "Qwen2.5-7B Full",
            "name": "Qwen-7B KV-Proj",
            "layer_type": "attn_kv",
            "K": 3584,
            "N": 512,
        },
        # --- Qwen2.5-7B Sharded Shapes (Dual T4 Tensor Parallel Rank 0/1) ---
        {
            "category": "Qwen2.5-7B TP Sharded",
            "name": "TP Gate/Up (Col-Parallel)",
            "layer_type": "tp_col",
            "K": 3584,
            "N": 9472,
        },
        {
            "category": "Qwen2.5-7B TP Sharded",
            "name": "TP Down (Row-Parallel)",
            "layer_type": "tp_row",
            "K": 9472,
            "N": 3584,
        },
        # --- Llama-3-8B Shapes (D = 4096, H = 14336) ---
        {
            "category": "Llama-3-8B Full",
            "name": "Llama-3-8B Gate/Up",
            "layer_type": "mlp_up",
            "K": 4096,
            "N": 14336,
        },
        {
            "category": "Llama-3-8B Full",
            "name": "Llama-3-8B Down Proj",
            "layer_type": "mlp_down",
            "K": 14336,
            "N": 4096,
        },
    ]


def run_3way_benchmark(
    batch_sizes: List[int],
    shapes_filter: Optional[str] = None,
    iters: int = 50,
    warmup: int = 20,
    group_size: int = 128,
    dry_run: bool = False,
    output_json: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Runs the 3-way kernel match across all requested shapes and batch sizes."""
    if not torch.cuda.is_available() and not dry_run:
        raise RuntimeError(
            "CUDA is not available on this host. Real GPU kernel benchmark requires CUDA. "
            "Pass --dry-run to test benchmark pipeline structure on CPU."
        )

    device = torch.device("cuda" if torch.cuda.is_available() and not dry_run else "cpu")

    print("\n" + "=" * 115)
    print("      3-WAY HEAD-TO-HEAD KERNEL BENCHMARK: t4_kernels vs bitsandbytes vs Marlin (Turing T4)")
    print("=" * 115)
    print(f"Hardware Device: {device} | Dry Run: {dry_run}")
    print(f"Kernel Status: t4_kernels={HAS_T4_KERNELS} | bitsandbytes={HAS_BNB} | Marlin={HAS_MARLIN}")
    print(f"Timing Config: {warmup} warmup iterations, {iters} timed iterations (CUDA events)")
    print(f"Quantization: Symmetric INT4 / NF4 with group_size={group_size}")
    print("=" * 115)

    all_shapes = get_benchmark_shapes()
    if shapes_filter:
        selected_shapes = [s for s in all_shapes if shapes_filter.lower() in s["name"].lower() or shapes_filter.lower() in s["category"].lower()]
    else:
        selected_shapes = all_shapes

    if not selected_shapes:
        print(f"No shapes matched filter: {shapes_filter}")
        return []

    quant_fn = quantize_weight_sym_int4 if HAS_QUANT_HELPER else fallback_quantize_sym_int4

    results: List[Dict[str, Any]] = []

    for cfg in selected_shapes:
        cat = cfg["category"]
        name = cfg["name"]
        K = cfg["K"]
        N = cfg["N"]

        print(f"\n>>> [{cat}] {name} (K={K}, N={N})")
        print("-" * 115)
        print(f"{'Batch M':<8} | {'cuBLAS FP16':<12} | {'t4_kernels':<12} | {'bitsandbytes':<14} | {'Marlin':<12} | {'Speedup (T4/cuB)':<18} | {'Parity (CosSim)':<15}")
        print("-" * 115)

        # Instantiate unquantized baseline linear layer
        torch.manual_seed(42)
        lin_fp16 = nn.Linear(K, N, bias=False).half().to(device)

        # 1. Prepare t4_kernels weights
        t4_data = None
        t4_init_status = "READY" if HAS_T4_KERNELS and device.type == "cuda" else ("SKIPPED: Dry-run CPU mode" if dry_run else "SKIPPED: Not compiled")
        if HAS_T4_KERNELS and device.type == "cuda":
            try:
                packed_w, scales, zps, g_size = quant_fn(lin_fp16.weight, group_size=group_size)
                t4_data = (packed_w, scales, zps, g_size)
            except Exception as e:
                t4_data = None
                t4_init_status = f"FAILED: Quantization error ({e})"

        # 2. Prepare bitsandbytes layer
        bnb_layer = None
        bnb_init_status = "READY" if HAS_BNB and device.type == "cuda" else ("SKIPPED: Dry-run CPU mode" if dry_run else "SKIPPED: Not installed")
        if HAS_BNB and device.type == "cuda":
            try:
                bnb_layer = bnb.nn.Linear4bit(
                    K,
                    N,
                    bias=False,
                    compute_dtype=torch.float16,
                    quant_type="nf4",
                )
                if hasattr(bnb.nn, "Params4bit"):
                    bnb_layer.weight = bnb.nn.Params4bit(
                        lin_fp16.weight.data.clone().cpu(),
                        requires_grad=False,
                        quant_type="nf4",
                    )
                    bnb_layer = bnb_layer.to(device)
                else:
                    bnb_layer = bnb_layer.to(device)
                    with torch.no_grad():
                        bnb_layer.weight.copy_(lin_fp16.weight.data)
                bnb_init_status = "READY"
            except Exception as e:
                bnb_layer = None
                bnb_init_status = f"FAILED: BNB init error ({e})"

        # 3. Prepare Marlin layer
        marlin_fn = None
        marlin_init_status = "READY" if HAS_MARLIN and device.type == "cuda" else ("SKIPPED: Dry-run CPU mode" if dry_run else "SKIPPED: Not installed")
        if HAS_MARLIN and device.type == "cuda":
            try:
                marlin_fn = prepare_marlin_linear(lin_fp16.weight, group_size=group_size)
                if marlin_fn is None:
                    marlin_init_status = "SKIPPED: Marlin packing incompatible shape or unsupported"
            except Exception as e:
                marlin_fn = None
                marlin_init_status = f"FAILED: Marlin prep error ({e})"

        for M in batch_sizes:
            x = torch.randn(M, K, dtype=torch.half, device=device)

            # Reference: cuBLAS FP16
            fn_cublas = lambda: lin_fp16(x)
            timing_cublas = benchmark_cuda_op(fn_cublas, iters=iters, warmup=warmup)
            t_cublas_us = timing_cublas["median_us"]
            bw_cublas, tflops_cublas = compute_metrics(t_cublas_us, M, K, N, is_4bit=False)
            out_ref = lin_fp16(x)

            # Kernel 1: t4_kernels
            timing_t4: Optional[Dict[str, Any]] = None
            t_t4_us: Optional[float] = None
            bw_t4: Optional[float] = None
            tflops_t4: Optional[float] = None
            cos_sim_t4: Optional[float] = None
            max_err_t4: Optional[float] = None
            t4_status = t4_init_status

            if t4_data is not None and HAS_T4_KERNELS and device.type == "cuda":
                p, s, z, g = t4_data
                fn_t4 = lambda: t4_kernels.fused_w4a16_gemm_u4(x, p, s, z, g)
                try:
                    timing_t4 = benchmark_cuda_op(fn_t4, iters=iters, warmup=warmup)
                    t_t4_us = timing_t4["median_us"]
                    bw_t4, tflops_t4 = compute_metrics(t_t4_us, M, K, N, is_4bit=True, group_size=g)
                    out_t4 = t4_kernels.fused_w4a16_gemm_u4(x, p, s, z, g)
                    max_err_t4, cos_sim_t4 = compute_numerical_parity(out_t4, out_ref)
                    t4_status = "COMPLETED"
                except Exception as e:
                    t_t4_us = None
                    t4_status = f"FAILED: Kernel execution error ({e})"

            # Kernel 2: bitsandbytes NF4
            timing_bnb: Optional[Dict[str, Any]] = None
            t_bnb_us: Optional[float] = None
            bw_bnb: Optional[float] = None
            tflops_bnb: Optional[float] = None
            cos_sim_bnb: Optional[float] = None
            max_err_bnb: Optional[float] = None
            bnb_status = bnb_init_status

            if bnb_layer is not None and HAS_BNB and device.type == "cuda":
                fn_bnb = lambda: bnb_layer(x)
                try:
                    timing_bnb = benchmark_cuda_op(fn_bnb, iters=iters, warmup=warmup)
                    t_bnb_us = timing_bnb["median_us"]
                    bw_bnb, tflops_bnb = compute_metrics(t_bnb_us, M, K, N, is_4bit=True, group_size=group_size)
                    out_bnb = bnb_layer(x)
                    max_err_bnb, cos_sim_bnb = compute_numerical_parity(out_bnb, out_ref)
                    bnb_status = "COMPLETED"
                except Exception as e:
                    t_bnb_us = None
                    bnb_status = f"FAILED: BNB execution error ({e})"

            # Kernel 3: Marlin
            timing_marlin: Optional[Dict[str, Any]] = None
            t_marlin_us: Optional[float] = None
            bw_marlin: Optional[float] = None
            tflops_marlin: Optional[float] = None
            cos_sim_marlin: Optional[float] = None
            max_err_marlin: Optional[float] = None
            marlin_status = marlin_init_status

            if marlin_fn is not None and HAS_MARLIN and device.type == "cuda":
                try:
                    timing_marlin = benchmark_cuda_op(marlin_fn, iters=iters, warmup=warmup)
                    t_marlin_us = timing_marlin["median_us"]
                    bw_marlin, tflops_marlin = compute_metrics(t_marlin_us, M, K, N, is_4bit=True, group_size=group_size)
                    out_marlin = marlin_fn(x)
                    max_err_marlin, cos_sim_marlin = compute_numerical_parity(out_marlin, out_ref)
                    marlin_status = "COMPLETED"
                except Exception as e:
                    t_marlin_us = None
                    marlin_status = f"FAILED: Marlin execution error ({e})"

            # Formatting table outputs
            cublas_str = f"{t_cublas_us:7.1f} us"
            t4_str = f"{t_t4_us:7.1f} us" if t_t4_us is not None else ("N/A (dry)" if dry_run else "Not compiled")
            if t_bnb_us is not None:
                bnb_str = f"{t_bnb_us:7.1f} us"
            elif not HAS_BNB:
                bnb_str = "Not installed"
            elif dry_run:
                bnb_str = "N/A (dry)"
            else:
                bnb_str = "Init error"
            marlin_str = f"{t_marlin_us:7.1f} us" if t_marlin_us is not None else ("N/A (dry)" if dry_run else "Not installed")

            if t_t4_us is not None and t_cublas_us > 0:
                speedup_t4 = t_cublas_us / t_t4_us
                speedup_str = f"{speedup_t4:6.2f}x"
            else:
                speedup_str = "--"

            cos_sim_str = f"{cos_sim_t4:8.4f}" if cos_sim_t4 is not None else "--"

            print(f"M={M:<6} | {cublas_str:<12} | {t4_str:<12} | {bnb_str:<14} | {marlin_str:<12} | {speedup_str:<18} | {cos_sim_str:<15}")

            row = {
                "category": cat,
                "name": name,
                "M": M,
                "K": K,
                "N": N,
                "cuBLAS_FP16": {
                    "status": "COMPLETED",
                    "latency_us": t_cublas_us,
                    "raw_timings_us": timing_cublas["raw_timings_us"],
                    "median_us": timing_cublas["median_us"],
                    "p95_us": timing_cublas["p95_us"],
                    "min_us": timing_cublas["min_us"],
                    "max_us": timing_cublas["max_us"],
                    "bandwidth_gb_s": bw_cublas,
                    "pct_t4_bandwidth": round(bw_cublas / T4_PEAK_BANDWIDTH_GB_S * 100.0, 2),
                    "tflops": tflops_cublas,
                    "pct_t4_tflops": round(tflops_cublas / T4_PEAK_FP16_TC_TFLOPS * 100.0, 2),
                },
                "t4_kernels": {
                    "status": t4_status,
                    "latency_us": t_t4_us,
                    "raw_timings_us": timing_t4["raw_timings_us"] if timing_t4 else [],
                    "median_us": timing_t4["median_us"] if timing_t4 else None,
                    "p95_us": timing_t4["p95_us"] if timing_t4 else None,
                    "min_us": timing_t4["min_us"] if timing_t4 else None,
                    "max_us": timing_t4["max_us"] if timing_t4 else None,
                    "bandwidth_gb_s": bw_t4,
                    "pct_t4_bandwidth": round(bw_t4 / T4_PEAK_BANDWIDTH_GB_S * 100.0, 2) if bw_t4 else None,
                    "tflops": tflops_t4,
                    "pct_t4_tflops": round(tflops_t4 / T4_PEAK_FP16_TC_TFLOPS * 100.0, 2) if tflops_t4 else None,
                    "max_err": max_err_t4,
                    "cos_sim": cos_sim_t4,
                    "speedup_vs_cublas": (t_cublas_us / t_t4_us) if t_t4_us else None,
                    "speedup_vs_bnb": (t_bnb_us / t_t4_us) if (t_t4_us and t_bnb_us) else None,
                },
                "bitsandbytes": {
                    "status": bnb_status,
                    "latency_us": t_bnb_us,
                    "raw_timings_us": timing_bnb["raw_timings_us"] if timing_bnb else [],
                    "median_us": timing_bnb["median_us"] if timing_bnb else None,
                    "p95_us": timing_bnb["p95_us"] if timing_bnb else None,
                    "min_us": timing_bnb["min_us"] if timing_bnb else None,
                    "max_us": timing_bnb["max_us"] if timing_bnb else None,
                    "bandwidth_gb_s": bw_bnb,
                    "pct_t4_bandwidth": round(bw_bnb / T4_PEAK_BANDWIDTH_GB_S * 100.0, 2) if bw_bnb else None,
                    "tflops": tflops_bnb,
                    "pct_t4_tflops": round(tflops_bnb / T4_PEAK_FP16_TC_TFLOPS * 100.0, 2) if tflops_bnb else None,
                    "max_err": max_err_bnb,
                    "cos_sim": cos_sim_bnb,
                },
                "marlin": {
                    "status": marlin_status,
                    "latency_us": t_marlin_us,
                    "raw_timings_us": timing_marlin["raw_timings_us"] if timing_marlin else [],
                    "median_us": timing_marlin["median_us"] if timing_marlin else None,
                    "p95_us": timing_marlin["p95_us"] if timing_marlin else None,
                    "min_us": timing_marlin["min_us"] if timing_marlin else None,
                    "max_us": timing_marlin["max_us"] if timing_marlin else None,
                    "bandwidth_gb_s": bw_marlin,
                    "pct_t4_bandwidth": round(bw_marlin / T4_PEAK_BANDWIDTH_GB_S * 100.0, 2) if bw_marlin else None,
                    "tflops": tflops_marlin,
                    "pct_t4_tflops": round(tflops_marlin / T4_PEAK_FP16_TC_TFLOPS * 100.0, 2) if tflops_marlin else None,
                    "max_err": max_err_marlin,
                    "cos_sim": cos_sim_marlin,
                },
            }
            results.append(row)

    print("-" * 115)
    print("\nBenchmark summary:")
    print("1. Single-token decode (M=1): Memory-bandwidth bound. Target 180+ GB/s (T4 peak 320 GB/s).")
    print("2. Prefill GEMM (M>=64): Compute bound. Target 30+ TFLOP/s (T4 FP16 Tensor Core peak 65 TFLOP/s).")
    print("=" * 115 + "\n")

    if output_json:
        os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
        device_info: Dict[str, Any] = {
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "pytorch_version": torch.__version__,
        }
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            device_info.update({
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory_mb": round(props.total_memory / (1024 * 1024), 2),
            })

        payload = {
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "hardware_device": str(device),
                "dry_run": dry_run,
                "device_info": device_info,
                "config": {
                    "iters": iters,
                    "warmup": warmup,
                    "group_size": group_size,
                    "batch_sizes": batch_sizes,
                },
                "hardware_baselines": {
                    "t4_peak_bandwidth_gb_s": T4_PEAK_BANDWIDTH_GB_S,
                    "t4_peak_fp16_tc_tflops": T4_PEAK_FP16_TC_TFLOPS,
                },
                "kernel_availability": {
                    "t4_kernels": HAS_T4_KERNELS,
                    "bitsandbytes": HAS_BNB,
                    "marlin": HAS_MARLIN,
                },
            },
            "results": results,
        }
        with open(output_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Results successfully saved to {output_json}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="3-Way Head-to-Head Kernel Benchmark (t4_kernels vs bitsandbytes vs Marlin)")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16, 64, 256], help="Token batch sizes M to evaluate (default: 1 4 16 64 256)")
    parser.add_argument("--shapes", type=str, default=None, help="Filter shapes by substring (e.g., 'Qwen', 'TP', 'Gate', 'Down', 'QKV')")
    parser.add_argument("--iters", type=int, default=50, help="Number of timed iterations (default: 50)")
    parser.add_argument("--warmup", type=int, default=20, help="Number of warmup iterations (default: 20)")
    parser.add_argument("--group-size", type=int, default=128, help="Quantization group size (default: 128)")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode on CPU to verify pipeline structure without GPU")
    parser.add_argument("--output-json", type=str, default=None, help="Optional output JSON file path for benchmark results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_3way_benchmark(
        batch_sizes=args.batch_sizes,
        shapes_filter=args.shapes,
        iters=args.iters,
        warmup=args.warmup,
        group_size=args.group_size,
        dry_run=args.dry_run,
        output_json=args.output_json,
    )
