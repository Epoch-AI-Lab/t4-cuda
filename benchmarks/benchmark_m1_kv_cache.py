#!/usr/bin/env python3
"""Micro-benchmarks for Milestone 1: StaticKVCache vs DynamicCache.

Evaluates:
1. Rollback Latency: O(1) integer pointer decrement vs dynamic tensor slicing / cropping.
2. Update Latency and Memory Traffic: In-place buffer copy vs dynamic torch.cat reallocation.
3. Memory Allocation Tracking: Zero CUDA allocations during decode for StaticKVCache.

Outputs results to results/m1_kv_cache_benchmark.json and prints human-readable tables.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict
import torch
from transformers.cache_utils import DynamicCache

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.static_kv_cache import StaticKVCache


def benchmark_rollback(
    num_layers: int = 28,
    num_heads: int = 2,
    head_dim: int = 128,
    seq_len: int = 1024,
    drop_k: int = 3,
    iters: int = 2000,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Dict[str, Any]:
    """Measures rollback latency between StaticKVCache pointer rollback and DynamicCache cropping."""
    dev = torch.device(device)
    warmup_iters = 200

    # 1. Setup StaticKVCache
    static_cache = StaticKVCache(
        max_batch_size=1,
        max_seq_len=seq_len + 64,
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        device=dev,
        dtype=dtype,
    )
    k_init = torch.randn(1, num_heads, seq_len, head_dim, device=dev, dtype=dtype)
    v_init = torch.randn(1, num_heads, seq_len, head_dim, device=dev, dtype=dtype)
    for l in range(num_layers):
        static_cache.update(l, k_init, v_init, start_pos=0)

    # 2. Setup DynamicCache baseline
    dyn_cache = DynamicCache()
    for l in range(num_layers):
        dyn_cache.update(k_init, v_init, layer_idx=l)

    # Warmup
    for _ in range(warmup_iters):
        static_cache.rollback(seq_len - drop_k)
        static_cache.current_pos = seq_len

    if dev.type == "cuda":
        torch.cuda.synchronize(dev)

    # Timed StaticKVCache rollback (only measure rollback operation)
    static_times = []
    for _ in range(iters):
        static_cache.current_pos = seq_len
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        static_cache.rollback(seq_len - drop_k)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        static_times.append(time.perf_counter() - t0)
    static_latency_us = (sum(static_times) / iters) * 1e6

    # Timed DynamicCache crop (only measure crop operation without timing restoration)
    dyn_times = []
    for _ in range(iters):
        # Reset state outside the timing window
        dyn_cache = DynamicCache()
        for l in range(num_layers):
            dyn_cache.update(k_init, v_init, layer_idx=l)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        dyn_cache.crop(-drop_k)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        dyn_times.append(time.perf_counter() - t0)
    dyn_latency_us = (sum(dyn_times) / iters) * 1e6

    speedup = dyn_latency_us / max(1e-9, static_latency_us)


    return {
        "num_layers": num_layers,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "seq_len": seq_len,
        "drop_k": drop_k,
        "iterations": iters,
        "static_latency_us": round(static_latency_us, 4),
        "dynamic_latency_us": round(dyn_latency_us, 4),
        "speedup_factor": round(speedup, 2),
    }


def benchmark_update(
    num_layers: int = 28,
    num_heads: int = 2,
    head_dim: int = 128,
    seq_len: int = 512,
    decode_steps: int = 100,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Dict[str, Any]:
    """Measures single-token decode update latency and memory traffic."""
    dev = torch.device(device)
    bytes_per_elem = 2 if dtype in (torch.float16, torch.bfloat16) else 4

    static_cache = StaticKVCache(
        max_batch_size=1,
        max_seq_len=seq_len + decode_steps + 32,
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        device=dev,
        dtype=dtype,
    )
    dyn_cache = DynamicCache()

    k_init = torch.randn(1, num_heads, seq_len, head_dim, device=dev, dtype=dtype)
    v_init = torch.randn(1, num_heads, seq_len, head_dim, device=dev, dtype=dtype)
    for l in range(num_layers):
        static_cache.update(l, k_init, v_init, start_pos=0)
        dyn_cache.update(k_init, v_init, layer_idx=l)

    k_tok = torch.randn(1, num_heads, 1, head_dim, device=dev, dtype=dtype)
    v_tok = torch.randn(1, num_heads, 1, head_dim, device=dev, dtype=dtype)

    # Measure StaticKVCache in-place update
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    for step in range(decode_steps):
        pos = seq_len + step
        for l in range(num_layers):
            static_cache.update(l, k_tok, v_tok, start_pos=pos)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t1 = time.perf_counter()
    static_time_us_per_token = ((t1 - t0) / decode_steps) * 1e6
    static_time_us_per_layer = static_time_us_per_token / num_layers

    # Measure DynamicCache torch.cat update
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    for step in range(decode_steps):
        for l in range(num_layers):
            dyn_cache.update(k_tok, v_tok, layer_idx=l)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t1 = time.perf_counter()
    dyn_time_us_per_token = ((t1 - t0) / decode_steps) * 1e6
    dyn_time_us_per_layer = dyn_time_us_per_token / num_layers

    # Calculate theoretical memory traffic per token
    static_traffic_bytes = 2 * num_layers * 1 * num_heads * 1 * head_dim * bytes_per_elem
    # Dynamic torch.cat reads current history + new token, writes new concatenated buffer
    avg_seq = seq_len + (decode_steps // 2)
    dynamic_traffic_bytes = 2 * num_layers * 1 * num_heads * (2 * avg_seq + 1) * head_dim * bytes_per_elem

    return {
        "num_layers": num_layers,
        "seq_len": seq_len,
        "decode_steps": decode_steps,
        "static_time_us_per_token": round(static_time_us_per_token, 3),
        "dynamic_time_us_per_token": round(dyn_time_us_per_token, 3),
        "static_time_us_per_layer": round(static_time_us_per_layer, 3),
        "dynamic_time_us_per_layer": round(dyn_time_us_per_layer, 3),
        "update_speedup": round(dyn_time_us_per_token / max(1e-9, static_time_us_per_token), 2),
        "static_traffic_kb_per_token": round(static_traffic_bytes / 1024.0, 2),
        "dynamic_traffic_kb_per_token": round(dynamic_traffic_bytes / 1024.0, 2),
        "traffic_reduction_ratio": round(dynamic_traffic_bytes / max(1, static_traffic_bytes), 1),
    }


def benchmark_memory_allocations(
    num_layers: int = 28,
    num_heads: int = 2,
    head_dim: int = 128,
    seq_len: int = 512,
    decode_steps: int = 50,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Dict[str, Any]:
    """Tracks physical memory allocations during decode steps."""
    dev = torch.device(device)
    if dev.type != "cuda":
        return {
            "status": "skipped_on_cpu",
            "reason": "torch.cuda.memory_stats only available on CUDA devices",
            "guarantee": "data_ptr invariance verified in unit tests",
        }

    static_cache = StaticKVCache(1, seq_len + decode_steps + 16, num_layers, num_heads, head_dim, dev, dtype)
    k_init = torch.randn(1, num_heads, seq_len, head_dim, device=dev, dtype=dtype)
    v_init = torch.randn(1, num_heads, seq_len, head_dim, device=dev, dtype=dtype)
    for l in range(num_layers):
        static_cache.update(l, k_init, v_init, start_pos=0)

    k_tok = torch.randn(1, num_heads, 1, head_dim, device=dev, dtype=dtype)
    v_tok = torch.randn(1, num_heads, 1, head_dim, device=dev, dtype=dtype)

    torch.cuda.synchronize(dev)
    stats_before = torch.cuda.memory_stats(dev)
    mem_before = torch.cuda.memory_allocated(dev)

    for step in range(decode_steps):
        pos = seq_len + step
        for l in range(num_layers):
            static_cache.update(l, k_tok, v_tok, start_pos=pos)

    torch.cuda.synchronize(dev)
    stats_after = torch.cuda.memory_stats(dev)
    mem_after = torch.cuda.memory_allocated(dev)

    allocs = stats_after["allocation.all.allocated"] - stats_before["allocation.all.allocated"]
    frees = stats_after["allocation.all.freed"] - stats_before["allocation.all.freed"]
    mem_delta = mem_after - mem_before

    return {
        "decode_steps": decode_steps,
        "num_layers": num_layers,
        "cuda_allocations": allocs,
        "cuda_frees": frees,
        "memory_delta_bytes": mem_delta,
        "zero_allocation_verified": (allocs == 0 and frees == 0 and mem_delta == 0),
    }


def main():
    parser = argparse.ArgumentParser(description="Micro-benchmarks for StaticKVCache vs DynamicCache")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device (cpu/cuda)")
    parser.add_argument("--output", default="results/m1_kv_cache_benchmark.json", help="Path to output JSON")
    parser.add_argument("--iters", type=int, default=1000, help="Number of benchmark iterations")
    parser.add_argument("--seq-len", type=int, default=1024, help="Context sequence length")
    args = parser.parse_args()

    # Determine dtype
    dtype = torch.float16 if (args.device.startswith("cuda") and torch.cuda.is_available()) else torch.float32

    print(f"=== Milestone 1: StaticKVCache Micro-Benchmark ===")
    print(f"Device: {args.device} | Dtype: {dtype} | Seq Len: {args.seq_len} | Iterations: {args.iters}")
    print("-" * 65)

    print("1. Benchmarking Rollback Latency (drop 3 tokens)...")
    rollback_res = benchmark_rollback(
        num_layers=28,
        num_heads=2,
        head_dim=128,
        seq_len=args.seq_len,
        drop_k=3,
        iters=args.iters,
        device=args.device,
        dtype=dtype,
    )
    print(f"   StaticKVCache Rollback: {rollback_res['static_latency_us']:.3f} µs")
    print(f"   DynamicCache Rollback:  {rollback_res['dynamic_latency_us']:.3f} µs")
    print(f"   Speedup:                {rollback_res['speedup_factor']}x")

    print("\n2. Benchmarking Token Update Latency and Memory Traffic...")
    update_res = benchmark_update(
        num_layers=28,
        num_heads=2,
        head_dim=128,
        seq_len=args.seq_len,
        decode_steps=100,
        device=args.device,
        dtype=dtype,
    )
    print(f"   Static Update Latency:  {update_res['static_time_us_per_token']:.2f} µs/token ({update_res['static_time_us_per_layer']:.2f} µs/layer)")
    print(f"   Dynamic Update Latency: {update_res['dynamic_time_us_per_token']:.2f} µs/token ({update_res['dynamic_time_us_per_layer']:.2f} µs/layer)")
    print(f"   Update Speedup:         {update_res['update_speedup']}x")
    print(f"   Memory Traffic:         {update_res['static_traffic_kb_per_token']} KB (Static) vs {update_res['dynamic_traffic_kb_per_token']} KB (Dynamic)")
    print(f"   Traffic Reduction:      {update_res['traffic_reduction_ratio']}x")

    print("\n3. Benchmarking Memory Allocations during Decode...")
    alloc_res = benchmark_memory_allocations(
        num_layers=28,
        num_heads=2,
        head_dim=128,
        seq_len=args.seq_len,
        decode_steps=50,
        device=args.device,
        dtype=dtype,
    )
    print(f"   Allocations result: {json.dumps(alloc_res, indent=2)}")

    results = {
        "hardware": {
            "device": args.device,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        },
        "config": {
            "num_layers": 28,
            "num_heads": 2,
            "head_dim": 128,
            "seq_len": args.seq_len,
            "dtype": str(dtype),
        },
        "rollback_benchmark": rollback_res,
        "update_benchmark": update_res,
        "allocation_benchmark": alloc_res,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved benchmark results to {output_path.resolve()}")


if __name__ == "__main__":
    main()
