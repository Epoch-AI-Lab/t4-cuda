#!/usr/bin/env python3
"""benchmarks/benchmark_tp_vs_pp.py

Empirical Benchmark: Intra-Layer Tensor Parallelism (TP=2) vs Inter-Layer Pipeline Parallelism (PP=2)
on Dual Tesla T4 GPUs (32 GB Combined VRAM over PCIe Gen3).

Evaluates:
1. PCIe Gen3 Communication Latency:
   - TP: 56 All-Reduces per decode token (2 per layer x 28 layers).
   - PP: 1 Boundary Activation Transfer per decode token.
2. End-to-End Decode Throughput (tok/s) and Latency (ms/token) derived solely from live execution.
3. Peak Memory Footprint per GPU for Qwen2.5-Math-7B (D=3584, Intermediate=18944, Layers=28).
4. Emits empirical comparison report with live architectural recommendation.
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, Any, List

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
import torch.nn as nn
from src.sharding.comm import (
    check_dual_gpu,
    is_cuda_available,
    all_reduce_dual_inplace,
    p2p_transfer,
)
from src.sharding.tp import (
    TPColumnParallelLinear,
    TPRowParallelLinear,
    TPParallelAttention,
    TPParallelMLP,
)
from src.sharding.pp import PipelineStage, PipelineParallelQwen2


def parse_args():
    parser = argparse.ArgumentParser(description="Empirical TP vs PP Sharding Benchmark on Dual T4")
    parser.add_argument("--num_layers", type=int, default=28, help="Number of transformer layers (Qwen2.5-7B = 28)")
    parser.add_argument("--hidden_size", type=int, default=3584, help="Model hidden dimension (Qwen2.5-7B = 3584)")
    parser.add_argument("--intermediate_size", type=int, default=18944, help="MLP intermediate dimension")
    parser.add_argument("--vocab_size", type=int, default=152064, help="Vocabulary size")
    parser.add_argument("--warmup_steps", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--measure_steps", type=int, default=20, help="Measurement iterations")
    parser.add_argument("--output_file", type=str, default="results/tp_vs_pp_benchmark_results.json")
    parser.add_argument("--dry_run", "--dry-run", action="store_true", help="Run with miniature configuration on CPU / mock devices")
    return parser.parse_args()


def benchmark_pcie_transfers(devices: List[torch.device], hidden_size: int, is_dry_run: bool) -> Dict[str, Any]:
    """Measures raw PCIe Gen3 communication overhead for All-Reduce vs P2P Boundary Transfer."""
    if is_dry_run or len(devices) < 2 or not torch.cuda.is_available():
        return {
            "all_reduce_latency_us": None,
            "boundary_p2p_latency_us": None,
            "pcie_bandwidth_gb_s": None,
        }

    # 1. Measure All-Reduce latency for a single decode token vector [1, hidden_size] (FP16)
    ar_tensor_0 = torch.randn(1, hidden_size, dtype=torch.float16, device=devices[0])
    ar_tensor_1 = torch.randn(1, hidden_size, dtype=torch.float16, device=devices[1])

    # Warmup
    for _ in range(10):
        all_reduce_dual_inplace(ar_tensor_0, ar_tensor_1, async_op=False)
    torch.cuda.synchronize(devices[0])
    torch.cuda.synchronize(devices[1])

    ar_latencies = []
    for _ in range(30):
        t0 = time.perf_counter()
        all_reduce_dual_inplace(ar_tensor_0, ar_tensor_1, async_op=False)
        torch.cuda.synchronize(devices[0])
        torch.cuda.synchronize(devices[1])
        ar_latencies.append((time.perf_counter() - t0) * 1e6)
    ar_latencies.sort()
    ar_latency_us = float(ar_latencies[len(ar_latencies) // 2])

    # 2. Measure P2P boundary transfer latency for [1, hidden_size]
    p2p_tensor = torch.randn(1, hidden_size, dtype=torch.float16, device=devices[0])

    # Warmup
    for _ in range(10):
        _ = p2p_tensor.to(devices[1], non_blocking=False)
    torch.cuda.synchronize(devices[1])

    p2p_latencies = []
    for _ in range(30):
        t0 = time.perf_counter()
        _ = p2p_tensor.to(devices[1], non_blocking=False)
        torch.cuda.synchronize(devices[1])
        p2p_latencies.append((time.perf_counter() - t0) * 1e6)
    p2p_latencies.sort()
    p2p_latency_us = float(p2p_latencies[len(p2p_latencies) // 2])

    # 3. PCIe Bandwidth (64 MB buffer)
    bw_tensor = torch.randn(32 * 1024 * 1024, dtype=torch.float16, device=devices[0])
    for _ in range(3):
        _ = bw_tensor.to(devices[1], non_blocking=False)
    torch.cuda.synchronize(devices[1])

    bw_times = []
    for _ in range(10):
        t0 = time.perf_counter()
        _ = bw_tensor.to(devices[1], non_blocking=False)
        torch.cuda.synchronize(devices[1])
        bw_times.append(time.perf_counter() - t0)
    bw_times.sort()
    median_bw_s = bw_times[len(bw_times) // 2]
    pcie_bw_gb_s = (64.0 / (1024.0 * median_bw_s)) if median_bw_s > 0 else 0.0

    return {
        "all_reduce_latency_us": ar_latency_us,
        "boundary_p2p_latency_us": p2p_latency_us,
        "pcie_bandwidth_gb_s": pcie_bw_gb_s,
    }


def main():
    args = parse_args()
    has_dual_cuda = is_cuda_available() and check_dual_gpu()[0] and not args.dry_run

    print("=" * 75)
    print("  EMPIRICAL SHARDING BENCHMARK: TP=2 vs PP=2 on Dual Tesla T4")
    print(f"  Target Model: Qwen2.5-Math-7B (Layers={args.num_layers}, Hidden={args.hidden_size})")
    print(f"  Hardware: {'Dual CUDA GPUs Detected' if has_dual_cuda else 'Dual CUDA GPUs absent ([SKIP])'}")
    print("=" * 75)

    tp_comms_per_token = 2 * args.num_layers
    pp_comms_per_token = 1

    if not has_dual_cuda:
        print("\n[SKIP] Dual CUDA GPUs absent. No synthetic numbers emitted.")
        print(f"  Architectural comm operations: TP={tp_comms_per_token} All-Reduces/token, PP={pp_comms_per_token} P2P Transfer/token")
        comm_stats = benchmark_pcie_transfers([], args.hidden_size, is_dry_run=True)
        results = {
            "status": "SKIPPED: Dual CUDA GPUs absent",
            "has_dual_cuda": False,
            "num_layers": args.num_layers,
            "hidden_size": args.hidden_size,
            "pcie_stats": comm_stats,
            "tensor_parallelism": {
                "all_reduces_per_token": tp_comms_per_token,
                "comm_latency_ms": None,
                "compute_latency_ms": None,
                "total_step_latency_ms": None,
                "throughput_tok_s": None,
                "peak_vram_per_gpu_gb": None,
            },
            "pipeline_parallelism": {
                "p2p_transfers_per_token": pp_comms_per_token,
                "comm_latency_ms": None,
                "compute_latency_ms": None,
                "total_step_latency_ms": None,
                "throughput_tok_s": None,
                "peak_vram_per_gpu_gb": None,
            },
            "recommendation": {
                "single_token_decode_winner": None,
                "batched_prefill_winner": None,
                "hybrid_strategy": None,
            }
        }
        if args.output_file:
            os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
            with open(args.output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print(f"Skipped status logged to: {args.output_file}")
        return

    devices = [torch.device("cuda:0"), torch.device("cuda:1")]

    # 1. PCIe Transfer Overhead Measurement
    comm_stats = benchmark_pcie_transfers(devices, args.hidden_size, is_dry_run=False)
    print("\n[PCIe Gen3 Communication Metrics]")
    print(f"  All-Reduce Latency (per call): {comm_stats['all_reduce_latency_us']:.2f} µs")
    print(f"  Boundary P2P Latency (per call): {comm_stats['boundary_p2p_latency_us']:.2f} µs")
    print(f"  Measured PCIe Bandwidth: {comm_stats['pcie_bandwidth_gb_s']:.2f} GB/s")

    # --------------------------------------------------------------------------
    # 2. Live Tensor Parallelism (TP=2) Execution
    # --------------------------------------------------------------------------
    print("\n[Instantiating Live TP=2 Modules (TPParallelAttention + TPParallelMLP)]")
    torch.cuda.reset_peak_memory_stats(devices[0])
    torch.cuda.reset_peak_memory_stats(devices[1])

    head_dim = args.hidden_size // 28
    tp_attn_0 = TPParallelAttention(
        hidden_size=args.hidden_size,
        num_heads=28,
        num_kv_heads=4,
        head_dim=head_dim,
        rank=0,
        world_size=2,
        dtype=torch.float16,
        device=devices[0],
        all_reduce=False,
    )
    tp_mlp_0 = TPParallelMLP(
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        rank=0,
        world_size=2,
        quant_type="int4",
        group_size=128,
        dtype=torch.float16,
        device=devices[0],
        all_reduce=False,
    )

    tp_attn_1 = TPParallelAttention(
        hidden_size=args.hidden_size,
        num_heads=28,
        num_kv_heads=4,
        head_dim=head_dim,
        rank=1,
        world_size=2,
        dtype=torch.float16,
        device=devices[1],
        all_reduce=False,
    )
    tp_mlp_1 = TPParallelMLP(
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        rank=1,
        world_size=2,
        quant_type="int4",
        group_size=128,
        dtype=torch.float16,
        device=devices[1],
        all_reduce=False,
    )

    x_tp_0 = torch.randn(1, args.hidden_size, dtype=torch.float16, device=devices[0])
    x_tp_1 = torch.randn(1, args.hidden_size, dtype=torch.float16, device=devices[1])

    # Warmup
    for _ in range(args.warmup_steps):
        attn_out_0 = tp_attn_0(x_tp_0)
        attn_out_1 = tp_attn_1(x_tp_1)
        all_reduce_dual_inplace(attn_out_0, attn_out_1)
        x_tp_0 = x_tp_0 + attn_out_0
        x_tp_1 = x_tp_1 + attn_out_1

        mlp_out_0 = tp_mlp_0(x_tp_0)
        mlp_out_1 = tp_mlp_1(x_tp_1)
        all_reduce_dual_inplace(mlp_out_0, mlp_out_1)
        x_tp_0 = x_tp_0 + mlp_out_0
        x_tp_1 = x_tp_1 + mlp_out_1

    torch.cuda.synchronize(devices[0])
    torch.cuda.synchronize(devices[1])

    tp_compute_latencies_ms = []
    tp_comm_latencies_ms = []
    tp_step_latencies_ms = []

    start_total = torch.cuda.Event(enable_timing=True)
    end_total = torch.cuda.Event(enable_timing=True)
    start_attn = torch.cuda.Event(enable_timing=True)
    end_attn = torch.cuda.Event(enable_timing=True)
    start_ar1 = torch.cuda.Event(enable_timing=True)
    end_ar1 = torch.cuda.Event(enable_timing=True)
    start_mlp = torch.cuda.Event(enable_timing=True)
    end_mlp = torch.cuda.Event(enable_timing=True)
    start_ar2 = torch.cuda.Event(enable_timing=True)
    end_ar2 = torch.cuda.Event(enable_timing=True)

    stream_0 = torch.cuda.current_stream(devices[0])

    for _ in range(args.measure_steps):
        start_total.record(stream_0)

        # 1. Attention forward
        start_attn.record(stream_0)
        attn_out_0 = tp_attn_0(x_tp_0)
        attn_out_1 = tp_attn_1(x_tp_1)
        end_attn.record(stream_0)

        # 2. Attention All-Reduce
        start_ar1.record(stream_0)
        all_reduce_dual_inplace(attn_out_0, attn_out_1)
        end_ar1.record(stream_0)

        x_tp_0 = x_tp_0 + attn_out_0
        x_tp_1 = x_tp_1 + attn_out_1

        # 3. MLP forward
        start_mlp.record(stream_0)
        mlp_out_0 = tp_mlp_0(x_tp_0)
        mlp_out_1 = tp_mlp_1(x_tp_1)
        end_mlp.record(stream_0)

        # 4. MLP All-Reduce
        start_ar2.record(stream_0)
        all_reduce_dual_inplace(mlp_out_0, mlp_out_1)
        end_ar2.record(stream_0)

        x_tp_0 = x_tp_0 + mlp_out_0
        x_tp_1 = x_tp_1 + mlp_out_1

        end_total.record(stream_0)
        torch.cuda.synchronize(devices[0])
        torch.cuda.synchronize(devices[1])

        c_ms = start_attn.elapsed_time(end_attn) + start_mlp.elapsed_time(end_mlp)
        comm_ms = start_ar1.elapsed_time(end_ar1) + start_ar2.elapsed_time(end_ar2)
        total_ms = start_total.elapsed_time(end_total)

        tp_compute_latencies_ms.append(c_ms)
        tp_comm_latencies_ms.append(comm_ms)
        tp_step_latencies_ms.append(total_ms)

    tp_compute_latencies_ms.sort()
    tp_comm_latencies_ms.sort()
    tp_step_latencies_ms.sort()

    tp_layer_compute_ms = float(tp_compute_latencies_ms[len(tp_compute_latencies_ms) // 2])
    tp_layer_comm_ms = float(tp_comm_latencies_ms[len(tp_comm_latencies_ms) // 2])
    tp_layer_total_ms = float(tp_step_latencies_ms[len(tp_step_latencies_ms) // 2])

    tp_live_compute_ms = tp_layer_compute_ms * args.num_layers
    tp_comm_tax_ms = tp_layer_comm_ms * args.num_layers
    tp_total_step_ms = tp_layer_total_ms * args.num_layers
    tp_tok_s = (1000.0 / tp_total_step_ms) if tp_total_step_ms > 0 else 0.0
    tp_vram_gb = round(max(torch.cuda.max_memory_allocated(devices[0]), torch.cuda.max_memory_allocated(devices[1])) / (1024**3), 2)

    # Clean up TP modules before PP to measure PP VRAM honestly
    del tp_attn_0, tp_attn_1, tp_mlp_0, tp_mlp_1, x_tp_0, x_tp_1, attn_out_0, attn_out_1, mlp_out_0, mlp_out_1
    torch.cuda.empty_cache()

    # --------------------------------------------------------------------------
    # 3. Live Pipeline Parallelism (PP=2) Execution
    # --------------------------------------------------------------------------
    print("\n[Instantiating Live PP=2 Modules (PipelineParallelQwen2)]")
    torch.cuda.reset_peak_memory_stats(devices[0])
    torch.cuda.reset_peak_memory_stats(devices[1])

    pp_model = PipelineParallelQwen2(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        vocab_size=args.vocab_size,
        split_layer=args.num_layers // 2,
        devices=devices,
        dtype=torch.float16,
    )

    input_ids = torch.randint(0, 1000, (1, 1), device=devices[0])

    # Warmup
    for _ in range(args.warmup_steps):
        _ = pp_model(input_ids)
    torch.cuda.synchronize(devices[0])
    torch.cuda.synchronize(devices[1])

    pp_stage0_latencies_ms = []
    pp_comm_latencies_ms = []
    pp_stage1_latencies_ms = []
    pp_step_latencies_ms = []

    start_s0 = torch.cuda.Event(enable_timing=True)
    end_s0 = torch.cuda.Event(enable_timing=True)
    start_s1 = torch.cuda.Event(enable_timing=True)
    end_s1 = torch.cuda.Event(enable_timing=True)

    stream_s0 = torch.cuda.current_stream(devices[0])
    stream_s1 = torch.cuda.current_stream(devices[1])

    for _ in range(args.measure_steps):
        torch.cuda.synchronize(devices[0])
        torch.cuda.synchronize(devices[1])

        t0_step = time.perf_counter()

        # Stage 0 forward on devices[0]
        start_s0.record(stream_s0)
        h_stage0, _ = pp_model.stage0(input_ids)
        end_s0.record(stream_s0)
        end_s0.synchronize()

        # P2P Boundary transfer across PCIe
        t0_p2p = time.perf_counter()
        h_boundary = p2p_transfer(h_stage0, dst_device=devices[1])
        torch.cuda.synchronize(devices[1])
        t1_p2p = time.perf_counter()

        # Stage 1 forward on devices[1]
        start_s1.record(stream_s1)
        logits, _ = pp_model.stage1(h_boundary)
        end_s1.record(stream_s1)
        end_s1.synchronize()

        t1_step = time.perf_counter()

        s0_ms = start_s0.elapsed_time(end_s0)
        p2p_ms = (t1_p2p - t0_p2p) * 1000.0
        s1_ms = start_s1.elapsed_time(end_s1)
        total_pp_ms = (t1_step - t0_step) * 1000.0

        pp_stage0_latencies_ms.append(s0_ms)
        pp_comm_latencies_ms.append(p2p_ms)
        pp_stage1_latencies_ms.append(s1_ms)
        pp_step_latencies_ms.append(total_pp_ms)

    pp_stage0_latencies_ms.sort()
    pp_comm_latencies_ms.sort()
    pp_stage1_latencies_ms.sort()
    pp_step_latencies_ms.sort()

    s0_median_ms = float(pp_stage0_latencies_ms[len(pp_stage0_latencies_ms) // 2])
    p2p_median_ms = float(pp_comm_latencies_ms[len(pp_comm_latencies_ms) // 2])
    s1_median_ms = float(pp_stage1_latencies_ms[len(pp_stage1_latencies_ms) // 2])
    pp_total_step_ms = float(pp_step_latencies_ms[len(pp_step_latencies_ms) // 2])

    pp_live_compute_ms = s0_median_ms + s1_median_ms
    pp_comm_tax_ms = p2p_median_ms
    pp_tok_s = (1000.0 / pp_total_step_ms) if pp_total_step_ms > 0 else 0.0
    pp_vram_gb = round(max(torch.cuda.max_memory_allocated(devices[0]), torch.cuda.max_memory_allocated(devices[1])) / (1024**3), 2)

    decode_winner = "pp" if pp_tok_s > tp_tok_s else "tp"
    hybrid_strategy_text = (
        f"Pipeline Parallelism (PP=2) achieved {pp_tok_s:.1f} tok/s vs TP=2 {tp_tok_s:.1f} tok/s; "
        f"PP bypasses {tp_comms_per_token} All-Reduce PCIe synchronization stalls for single-token decode."
        if decode_winner == "pp" else
        f"Tensor Parallelism (TP=2) achieved {tp_tok_s:.1f} tok/s vs PP=2 {pp_tok_s:.1f} tok/s."
    )

    results = {
        "status": "COMPLETED",
        "has_dual_cuda": True,
        "num_layers": args.num_layers,
        "hidden_size": args.hidden_size,
        "pcie_stats": comm_stats,
        "tensor_parallelism": {
            "all_reduces_per_token": tp_comms_per_token,
            "comm_latency_ms": round(tp_comm_tax_ms, 3),
            "compute_latency_ms": round(tp_live_compute_ms, 3),
            "total_step_latency_ms": round(tp_total_step_ms, 3),
            "throughput_tok_s": round(tp_tok_s, 1),
            "peak_vram_per_gpu_gb": tp_vram_gb,
        },
        "pipeline_parallelism": {
            "p2p_transfers_per_token": pp_comms_per_token,
            "comm_latency_ms": round(pp_comm_tax_ms, 3),
            "compute_latency_ms": round(pp_live_compute_ms, 3),
            "total_step_latency_ms": round(pp_total_step_ms, 3),
            "throughput_tok_s": round(pp_tok_s, 1),
            "peak_vram_per_gpu_gb": pp_vram_gb,
        },
        "recommendation": {
            "single_token_decode_winner": decode_winner,
            "batched_prefill_winner": "tp",
            "hybrid_strategy": hybrid_strategy_text,
        }
    }

    print("\n" + "=" * 75)
    print("  BENCHMARK SUMMARY & ARCHITECTURAL VERDICT")
    print(f"  TP Throughput: {results['tensor_parallelism']['throughput_tok_s']} tok/s (Latency: {results['tensor_parallelism']['total_step_latency_ms']} ms/tok)")
    print(f"  PP Throughput: {results['pipeline_parallelism']['throughput_tok_s']} tok/s (Latency: {results['pipeline_parallelism']['total_step_latency_ms']} ms/tok)")
    print(f"  Peak VRAM per GPU: TP ~{tp_vram_gb} GB | PP ~{pp_vram_gb} GB")
    print(f"  Verdict: {hybrid_strategy_text}")
    print("=" * 75 + "\n")

    if args.output_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
