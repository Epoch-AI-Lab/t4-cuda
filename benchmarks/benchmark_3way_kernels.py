#!/usr/bin/env python3
"""3-Way Head-to-Head Kernel Benchmark: t4_kernels vs bitsandbytes vs Marlin.

Evaluates 4-bit low-precision projection execution on NVIDIA Turing (sm_75, Tesla T4):
1. Custom Kernel: t4_kernels (Re-tiled 100% Coalesced W4A16 GEMV / WMMA Tensor Core)
2. Industry Baseline 1: bitsandbytes NF4 (Linear4bit Normalized Float 4)
3. Industry Baseline 2: bitsandbytes FP4 (Linear4bit Standard 4-bit FP/INT4)
4. Industry Baseline 3: Marlin (W4A16 FP16xINT4 GEMM - requires Compute Capability >= 8.0)
5. Unquantized Reference: PyTorch cuBLAS FP16 (Standard unquantized baseline)

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
- Automatic logging: saves exact raw JSON, terminal log, and markdown table reports
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


class DualLogger:
    """Tees stdout to both terminal and a log file simultaneously."""
    def __init__(self, filepath: str):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        self.log_file = open(filepath, "w", encoding="utf-8")

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self) -> None:
        self.terminal.flush()
        self.log_file.flush()

    def close(self) -> None:
        self.log_file.close()


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
        if hasattr(marlin_module, "QuantLinear"):
            layer = marlin_module.QuantLinear(bits=4, group_size=group_size, infeatures=in_f, outfeatures=out_f, bias=False).to(W.device)
            return lambda x: layer(x)
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

    # Timed runs using individual CUDA event pairs
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

    raw_us = [t * 1e3 for t in timings_ms]

    return {
        "raw_timings_us": raw_us,
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
    if is_4bit:
        weight_bytes = (K * N) * 0.5
        if group_size > 0:
            num_groups = (K + group_size - 1) // group_size
            weight_bytes += num_groups * N * 4  # scales + zero_points in FP16
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
            "layer_type": "attn_q",
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
    output_json: Optional[str] = "results/3way_kernel_match_t4.json",
    output_log: Optional[str] = "results/3way_kernel_match_t4.log",
    output_md: Optional[str] = "results/3way_kernel_match_t4.md",
) -> List[Dict[str, Any]]:
    """Runs the 3-way kernel match across all requested shapes and batch sizes."""
    if not torch.cuda.is_available() and not dry_run:
        raise RuntimeError(
            "CUDA is not available on this host. Real GPU kernel benchmark requires CUDA. "
            "To inspect or test benchmark pipeline logic on CPU, run with --dry-run."
        )

    # Setup DualLogger if output_log is specified
    logger = None
    if output_log:
        logger = DualLogger(output_log)
        sys.stdout = logger

    try:
        device = torch.device("cuda:0" if torch.cuda.is_available() and not dry_run else "cpu")

        # Hardware inspection
        is_turing = False
        device_name = "CPU"
        if torch.cuda.is_available() and not dry_run:
            device_name = torch.cuda.get_device_name(0)
            cc = torch.cuda.get_device_capability(0)
            is_turing = (cc[0] < 8)

        print("\n" + "=" * 125)
        print("      3-WAY HEAD-TO-HEAD KERNEL BENCHMARK: t4_kernels vs bitsandbytes vs Marlin (Turing T4)")
        print("=" * 125)
        print(f"Hardware Device: {device_name} ({device}) | Dry Run: {dry_run}")
        print(f"Kernel Status: t4_kernels={HAS_T4_KERNELS} | bitsandbytes={HAS_BNB} | Marlin={HAS_MARLIN}")
        if is_turing:
            print("Note on Marlin: IST-DASLab Marlin requires Compute Capability >= 8.0 (Ampere+ cp.async).")
            print("                On Turing sm_75 (Tesla T4), bitsandbytes (NF4/FP4) is the verified baseline.")
        print(f"Timing Config: {warmup} warmup iterations, {iters} timed iterations (CUDA events)")
        print(f"Quantization: Symmetric INT4 / NF4 with group_size={group_size}")
        print("=" * 125)

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
            print("-" * 125)
            print(f"{'Batch M':<8} | {'cuBLAS FP16':<12} | {'t4_kernels':<12} | {'bnb (NF4)':<12} | {'bnb (FP4)':<12} | {'Marlin':<11} | {'Speedup(T4/cuB)':<16} | {'Speedup(T4/BNB)':<16}")
            print("-" * 125)

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

            # 2. Prepare bitsandbytes layers (NF4 and FP4)
            bnb_nf4_layer = None
            bnb_fp4_layer = None
            bnb_init_status = "READY" if HAS_BNB and device.type == "cuda" else ("SKIPPED: Dry-run CPU mode" if dry_run else "SKIPPED: Not installed")
            if HAS_BNB and device.type == "cuda":
                try:
                    bnb_nf4_layer = bnb.nn.Linear4bit(
                        K, N, bias=False, compute_dtype=torch.float16, quant_type="nf4"
                    ).to(device)
                except Exception as e:
                    bnb_nf4_layer = None
                    bnb_init_status = f"FAILED: BNB NF4 init ({e})"

                try:
                    bnb_fp4_layer = bnb.nn.Linear4bit(
                        K, N, bias=False, compute_dtype=torch.float16, quant_type="fp4"
                    ).to(device)
                except Exception:
                    bnb_fp4_layer = None

            # 3. Prepare Marlin layer
            marlin_fn = None
            marlin_init_status = "READY" if HAS_MARLIN and device.type == "cuda" else ("SKIPPED: Dry-run CPU mode" if dry_run else "SKIPPED: Not installed")
            if HAS_MARLIN and device.type == "cuda":
                try:
                    marlin_fn = prepare_marlin_linear(lin_fp16.weight, group_size=group_size)
                    if marlin_fn is None:
                        marlin_init_status = "SKIPPED: Marlin packing unsupported"
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

                # Kernel 1: t4_kernels (Re-tiled)
                timing_t4 = None
                t_t4_us = None
                bw_t4, tflops_t4, cos_sim_t4, max_err_t4 = None, None, None, None
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

                # Kernel 2a: bitsandbytes NF4
                timing_bnb_nf4 = None
                t_bnb_nf4_us = None
                bw_bnb_nf4, tflops_bnb_nf4 = None, None
                if bnb_nf4_layer is not None and HAS_BNB and device.type == "cuda":
                    fn_bnb_nf4 = lambda: bnb_nf4_layer(x)
                    try:
                        timing_bnb_nf4 = benchmark_cuda_op(fn_bnb_nf4, iters=iters, warmup=warmup)
                        t_bnb_nf4_us = timing_bnb_nf4["median_us"]
                        bw_bnb_nf4, tflops_bnb_nf4 = compute_metrics(t_bnb_nf4_us, M, K, N, is_4bit=True, group_size=group_size)
                    except Exception:
                        t_bnb_nf4_us = None

                # Kernel 2b: bitsandbytes FP4
                timing_bnb_fp4 = None
                t_bnb_fp4_us = None
                bw_bnb_fp4, tflops_bnb_fp4 = None, None
                if bnb_fp4_layer is not None and HAS_BNB and device.type == "cuda":
                    fn_bnb_fp4 = lambda: bnb_fp4_layer(x)
                    try:
                        timing_bnb_fp4 = benchmark_cuda_op(fn_bnb_fp4, iters=iters, warmup=warmup)
                        t_bnb_fp4_us = timing_bnb_fp4["median_us"]
                        bw_bnb_fp4, tflops_bnb_fp4 = compute_metrics(t_bnb_fp4_us, M, K, N, is_4bit=True, group_size=group_size)
                    except Exception:
                        t_bnb_fp4_us = None

                # Kernel 3: Marlin
                timing_marlin = None
                t_marlin_us = None
                bw_marlin, tflops_marlin = None, None
                if marlin_fn is not None and HAS_MARLIN and device.type == "cuda":
                    try:
                        timing_marlin = benchmark_cuda_op(marlin_fn, iters=iters, warmup=warmup)
                        t_marlin_us = timing_marlin["median_us"]
                        bw_marlin, tflops_marlin = compute_metrics(t_marlin_us, M, K, N, is_4bit=True, group_size=group_size)
                    except Exception:
                        t_marlin_us = None

                # String formatters for display
                cublas_str = f"{t_cublas_us:7.1f} us"
                t4_str = f"{t_t4_us:7.1f} us" if t_t4_us is not None else ("N/A (dry)" if dry_run else "Not compiled")
                bnb_nf4_str = f"{t_bnb_nf4_us:7.1f} us" if t_bnb_nf4_us is not None else ("N/A (dry)" if dry_run else ("Not installed" if not HAS_BNB else "Init error"))
                bnb_fp4_str = f"{t_bnb_fp4_us:7.1f} us" if t_bnb_fp4_us is not None else ("N/A (dry)" if dry_run else ("Not installed" if not HAS_BNB else "Init error"))

                if t_marlin_us is not None:
                    marlin_str = f"{t_marlin_us:7.1f} us"
                elif is_turing:
                    marlin_str = "Req sm_80+"
                elif not HAS_MARLIN:
                    marlin_str = "Not installed"
                elif dry_run:
                    marlin_str = "N/A (dry)"
                else:
                    marlin_str = "Unsupported"

                speedup_cub_str = f"{(t_cublas_us / t_t4_us):6.2f}x" if (t_t4_us and t_cublas_us > 0) else "--"
                speedup_bnb_str = f"{(t_bnb_nf4_us / t_t4_us):6.2f}x" if (t_t4_us and t_bnb_nf4_us) else "--"

                print(f"M={M:<6} | {cublas_str:<12} | {t4_str:<12} | {bnb_nf4_str:<12} | {bnb_fp4_str:<12} | {marlin_str:<11} | {speedup_cub_str:<16} | {speedup_bnb_str:<16}")

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
                        "speedup_vs_bnb_nf4": (t_bnb_nf4_us / t_t4_us) if (t_t4_us and t_bnb_nf4_us) else None,
                        "speedup_vs_bnb_fp4": (t_bnb_fp4_us / t_t4_us) if (t_t4_us and t_bnb_fp4_us) else None,
                    },
                    "bitsandbytes": {
                        "status": "COMPLETED" if t_bnb_nf4_us else bnb_init_status,
                        "latency_us": t_bnb_nf4_us,
                        "raw_timings_us": timing_bnb_nf4["raw_timings_us"] if timing_bnb_nf4 else [],
                        "median_us": timing_bnb_nf4["median_us"] if timing_bnb_nf4 else None,
                        "p95_us": timing_bnb_nf4["p95_us"] if timing_bnb_nf4 else None,
                        "min_us": timing_bnb_nf4["min_us"] if timing_bnb_nf4 else None,
                        "max_us": timing_bnb_nf4["max_us"] if timing_bnb_nf4 else None,
                        "bandwidth_gb_s": bw_bnb_nf4,
                        "pct_t4_bandwidth": round(bw_bnb_nf4 / T4_PEAK_BANDWIDTH_GB_S * 100.0, 2) if bw_bnb_nf4 else None,
                        "tflops": tflops_bnb_nf4,
                        "pct_t4_tflops": round(tflops_bnb_nf4 / T4_PEAK_FP16_TC_TFLOPS * 100.0, 2) if tflops_bnb_nf4 else None,
                        "cos_sim": None,
                        "max_err": None,
                    },
                    "bitsandbytes_nf4": {
                        "status": "COMPLETED" if t_bnb_nf4_us else bnb_init_status,
                        "latency_us": t_bnb_nf4_us,
                        "raw_timings_us": timing_bnb_nf4["raw_timings_us"] if timing_bnb_nf4 else [],
                        "median_us": timing_bnb_nf4["median_us"] if timing_bnb_nf4 else None,
                        "p95_us": timing_bnb_nf4["p95_us"] if timing_bnb_nf4 else None,
                        "min_us": timing_bnb_nf4["min_us"] if timing_bnb_nf4 else None,
                        "max_us": timing_bnb_nf4["max_us"] if timing_bnb_nf4 else None,
                        "bandwidth_gb_s": bw_bnb_nf4,
                        "pct_t4_bandwidth": round(bw_bnb_nf4 / T4_PEAK_BANDWIDTH_GB_S * 100.0, 2) if bw_bnb_nf4 else None,
                        "tflops": tflops_bnb_nf4,
                        "pct_t4_tflops": round(tflops_bnb_nf4 / T4_PEAK_FP16_TC_TFLOPS * 100.0, 2) if tflops_bnb_nf4 else None,
                        "cos_sim": None,
                        "max_err": None,
                    },
                    "bitsandbytes_fp4": {
                        "status": "COMPLETED" if t_bnb_fp4_us else bnb_init_status,
                        "latency_us": t_bnb_fp4_us,
                        "raw_timings_us": timing_bnb_fp4["raw_timings_us"] if timing_bnb_fp4 else [],
                        "median_us": timing_bnb_fp4["median_us"] if timing_bnb_fp4 else None,
                        "p95_us": timing_bnb_fp4["p95_us"] if timing_bnb_fp4 else None,
                        "min_us": timing_bnb_fp4["min_us"] if timing_bnb_fp4 else None,
                        "max_us": timing_bnb_fp4["max_us"] if timing_bnb_fp4 else None,
                        "bandwidth_gb_s": bw_bnb_fp4,
                        "pct_t4_bandwidth": round(bw_bnb_fp4 / T4_PEAK_BANDWIDTH_GB_S * 100.0, 2) if bw_bnb_fp4 else None,
                        "tflops": tflops_bnb_fp4,
                        "pct_t4_tflops": round(tflops_bnb_fp4 / T4_PEAK_FP16_TC_TFLOPS * 100.0, 2) if tflops_bnb_fp4 else None,
                        "cos_sim": None,
                        "max_err": None,
                    },
                    "marlin": {
                        "status": "SKIPPED: Requires sm_80+" if is_turing else marlin_init_status,
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
                        "cos_sim": None,
                        "max_err": None,
                    },
                }
                results.append(row)

        print("-" * 125)
        print("\nBenchmark Summary & Key Findings:")
        print("1. Single-token decode (M=1): Memory-bandwidth bound. Target 180+ GB/s (T4 peak 320 GB/s).")
        print("2. Prefill GEMM (M>=64): Compute bound. Target 30+ TFLOP/s (T4 FP16 Tensor Core peak 65 TFLOP/s).")
        print("=" * 125 + "\n")

        # Save Markdown Report
        if output_md:
            os.makedirs(os.path.dirname(os.path.abspath(output_md)), exist_ok=True)
            with open(output_md, "w", encoding="utf-8") as fmd:
                fmd.write("# 3-Way Head-to-Head Kernel Benchmark Report (Tesla T4)\n\n")
                fmd.write(f"- **Timestamp**: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
                fmd.write(f"- **Device**: {device_name} ({device})\n")
                fmd.write(f"- **Warmup**: {warmup} iterations | **Timed**: {iters} iterations (CUDA events)\n\n")
                fmd.write("| Model Projection | M | cuBLAS FP16 | t4_kernels | bnb (NF4) | bnb (FP4) | Speedup (T4/cuB) | Speedup (T4/BNB) | Parity |\n")
                fmd.write("|---|---|---|---|---|---|---|---|---|\n")
                for r in results:
                    c_us = f"{r['cuBLAS_FP16']['latency_us']:.1f} µs"
                    t_us = f"{r['t4_kernels']['latency_us']:.1f} µs" if r['t4_kernels']['latency_us'] else "--"
                    bnf_us = f"{r['bitsandbytes_nf4']['latency_us']:.1f} µs" if r['bitsandbytes_nf4']['latency_us'] else "--"
                    bfp_us = f"{r['bitsandbytes_fp4']['latency_us']:.1f} µs" if r['bitsandbytes_fp4']['latency_us'] else "--"
                    sp_cub = f"{r['t4_kernels']['speedup_vs_cublas']:.2f}x" if r['t4_kernels']['speedup_vs_cublas'] else "--"
                    sp_bnb = f"{r['t4_kernels']['speedup_vs_bnb_nf4']:.2f}x" if r['t4_kernels']['speedup_vs_bnb_nf4'] else "--"
                    cs = f"{r['t4_kernels']['cos_sim']:.4f}" if r['t4_kernels']['cos_sim'] else "--"
                    fmd.write(f"| {r['name']} | {r['M']} | {c_us} | {t_us} | {bnf_us} | {bfp_us} | {sp_cub} | {sp_bnb} | {cs} |\n")
            print(f"[SAVED] Markdown Report: {output_md}")

        # Save Raw JSON
        if output_json:
            os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
            device_info: Dict[str, Any] = {
                "cuda_available": torch.cuda.is_available(),
                "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "device_name": device_name,
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
                "pytorch_version": torch.__version__,
            }
            if torch.cuda.is_available() and not dry_run:
                props = torch.cuda.get_device_properties(0)
                device_info.update({
                    "compute_capability": f"{props.major}.{props.minor}",
                    "total_memory_mb": round(props.total_memory / (1024 * 1024), 2),
                })

            payload = {
                "metadata": {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "hardware_device": str(device),
                    "device_name": device_name,
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
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"[SAVED] Raw JSON Logs:   {output_json}")

        if output_log:
            print(f"[SAVED] Console Log:      {output_log}")

        return results

    finally:
        if logger is not None:
            sys.stdout = logger.terminal
            logger.close()


def parse_args():
    parser = argparse.ArgumentParser(description="3-Way Head-to-Head Kernel Benchmark (t4_kernels vs bitsandbytes vs Marlin)")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16, 64, 256], help="Token batch sizes M to evaluate (default: 1 4 16 64 256)")
    parser.add_argument("--shapes", type=str, default=None, help="Filter shapes by substring (e.g., 'Qwen', 'TP', 'Gate', 'Down', 'QKV')")
    parser.add_argument("--iters", type=int, default=50, help="Number of timed iterations (default: 50)")
    parser.add_argument("--warmup", type=int, default=20, help="Number of warmup iterations (default: 20)")
    parser.add_argument("--group-size", type=int, default=128, help="Quantization group size (default: 128)")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode on CPU to verify pipeline structure without GPU")
    parser.add_argument("--output-json", type=str, default="results/3way_kernel_match_t4.json", help="Output JSON file path")
    parser.add_argument("--output-log", type=str, default="results/3way_kernel_match_t4.log", help="Output text log file path")
    parser.add_argument("--output-md", type=str, default="results/3way_kernel_match_t4.md", help="Output Markdown report file path")
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
        output_log=args.output_log,
        output_md=args.output_md,
    )
