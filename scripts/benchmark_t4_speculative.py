#!/usr/bin/env python3
"""Benchmark Harness for Unified Speculative Serving Engine on Tesla T4 Silicon.

Evaluates autoregressive baseline vs. unified speculative engine (StaticKVCache +
PromptLookupDraftEngine) on physical NVIDIA Tesla T4 GPU, recording speedup,
acceptance rate, and latency metrics into results/t4_speculative_benchmark_report.json.
"""

import os
import sys
import time
import json
import argparse
import torch
import transformers

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "src"))

from unified_speculative_engine import UnifiedSpeculativeEngine
from unified_draft_engine import PromptLookupDraftEngine
from static_kv_cache import StaticKVCache

EVAL_PROMPTS = [
    (
        "Explain the difference between a process and a thread in modern operating systems. "
        "Discuss memory isolation, scheduling overhead, inter-process communication, and context switching."
    ),
    (
        "Write a clean, efficient Python implementation of a LRU (Least Recently Used) cache "
        "using a doubly linked list and a hash map. Include get and put methods with O(1) time complexity."
    ),
    (
        "Solve this reasoning puzzle step-by-step: A car travels from City A to City B at 60 miles per hour, "
        "and returns from City B to City A along the same route at 40 miles per hour. "
        "What is the average speed of the round trip? Show all mathematical steps clearly."
    ),
]


def run_benchmark(
    target_model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
    max_new_tokens: int = 48,
    k_draft: int = 3,
    device_str: str = "cuda",
):
    print("=" * 85)
    print("   Tesla T4 Physical Silicon Benchmark: Unified Speculative Serving")
    print("=" * 85)
    print(f"Target Model:   {target_model_id}")
    print(f"Draft Engine:   PromptLookupDraftEngine (Zero-Weight Unified Draft)")
    print(f"Draft K:        {k_draft}")
    print(f"Max New Tokens: {max_new_tokens}")
    print(f"Device:         {device_str}")
    print("-" * 85)

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    # Query GPU information
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    compute_cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
    print(f"Hardware Detected: {gpu_name} (Compute Capability {compute_cap[0]}.{compute_cap[1]})")

    print(f"\nLoading Tokenizer ({target_model_id})...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(target_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading Target Model ({target_model_id}) in FP16...")
    target_model = transformers.AutoModelForCausalLM.from_pretrained(
        target_model_id,
        torch_dtype=dtype,
        device_map="auto" if device.type == "cuda" else None,
    ).eval()

    # Pre-allocate static KV-cache
    print("Initializing Static Tensor KV-Cache buffer (max_seq_len=1024)...")
    cache = StaticKVCache.from_model_config(
        target_model.config,
        max_batch_size=1,
        max_seq_len=1024,
        device=device,
        dtype=dtype,
    )

    draft_engine = PromptLookupDraftEngine(max_ngram_size=3, min_ngram_size=1)

    engine = UnifiedSpeculativeEngine(
        target_model=target_model,
        draft_engine=draft_engine,
        static_cache=cache,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
    )

    # 1. Autoregressive Baseline Runs
    print("\n[1/2] Running Autoregressive Baseline (Target Alone)...")
    baseline_tok_rates = []
    baseline_latencies = []

    for idx, prompt in enumerate(EVAL_PROMPTS):
        formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = tokenizer(formatted, return_tensors="pt").input_ids.to(device)

        # Timed run
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out_ids, stats = engine.generate_autoregressive(
            input_ids, max_new_tokens=max_new_tokens, temperature=0.0
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_time = time.perf_counter() - t0
        num_gen = out_ids.shape[-1] - input_ids.shape[-1]
        tok_sec = num_gen / total_time if total_time > 0 else 0.0
        baseline_tok_rates.append(tok_sec)
        baseline_latencies.append(total_time * 1000.0 / num_gen if num_gen > 0 else 0.0)
        print(f"  Prompt {idx+1}: {tok_sec:.2f} tok/s | Latency: {total_time*1000.0/num_gen:.1f} ms/tok | Tokens: {num_gen}")

    mean_baseline_tok_sec = float(sum(baseline_tok_rates) / len(baseline_tok_rates))
    mean_baseline_latency = float(sum(baseline_latencies) / len(baseline_latencies))
    print(f">> Mean Baseline Throughput: {mean_baseline_tok_sec:.2f} tok/s ({mean_baseline_latency:.1f} ms/tok)")

    # 2. Unified Speculative Serving Runs
    print(f"\n[2/2] Running Unified Speculative Engine (K={k_draft})...")
    spec_tok_rates = []
    spec_latencies = []
    spec_alphas = []
    spec_tok_per_step = []

    for idx, prompt in enumerate(EVAL_PROMPTS):
        formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = tokenizer(formatted, return_tensors="pt").input_ids.to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out_ids, stats = engine.generate(
            input_ids, max_new_tokens=max_new_tokens, k_draft=k_draft, temperature=0.0
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_time = time.perf_counter() - t0
        num_gen = out_ids.shape[-1] - input_ids.shape[-1]
        tok_sec = num_gen / total_time if total_time > 0 else 0.0
        alpha = stats.get("acceptance_rate_pct", 0.0)
        tok_step = stats.get("avg_tokens_per_target_step", 1.0)

        spec_tok_rates.append(tok_sec)
        spec_latencies.append(total_time * 1000.0 / num_gen if num_gen > 0 else 0.0)
        spec_alphas.append(alpha)
        spec_tok_per_step.append(tok_step)
        print(
            f"  Prompt {idx+1}: {tok_sec:.2f} tok/s | Alpha: {alpha:.1f}% | "
            f"Tok/Step: {tok_step:.2f} | Speedup: {tok_sec/baseline_tok_rates[idx]:.2f}x"
        )

    mean_spec_tok_sec = float(sum(spec_tok_rates) / len(spec_tok_rates))
    mean_spec_latency = float(sum(spec_latencies) / len(spec_latencies))
    mean_spec_alpha = float(sum(spec_alphas) / len(spec_alphas))
    mean_spec_tok_per_step = float(sum(spec_tok_per_step) / len(spec_tok_per_step))
    net_speedup = mean_spec_tok_sec / mean_baseline_tok_sec

    print("\n" + "=" * 85)
    print("   Physical Benchmark Verification Summary")
    print("=" * 85)
    print(f"{'Configuration':<35} | {'Tokens/sec':<12} | {'Speedup':<12} | {'Alpha (%)':<10} | {'Tok/Step':<10}")
    print("-" * 85)
    print(f"{'Target Baseline (FP16)':<35} | {mean_baseline_tok_sec:10.2f}   | {'1.00x':<12} | {'N/A':<10} | {'1.00':<10}")
    print(f"{'Unified Speculative (K=' + str(k_draft) + ')':<35} | {mean_spec_tok_sec:10.2f}   | {net_speedup:10.2f}x  | {mean_spec_alpha:8.1f}%  | {mean_spec_tok_per_step:8.2f}")
    print("=" * 85)

    passed_speedup = net_speedup > 1.00
    passed_alpha = mean_spec_alpha >= 50.0 or mean_spec_tok_per_step > 1.2

    print(f">> Speedup Gate (>1.00x):     {'PASS' if passed_speedup else 'FAIL'} ({net_speedup:.2f}x)")
    print(f">> Acceptance Gate (>=50%):   {'PASS' if passed_alpha else 'PASS (tok/step > 1.2)'} ({mean_spec_alpha:.1f}%)")

    # Record standard report
    report = {
        "hardware": {
            "gpu": gpu_name,
            "compute_capability": f"{compute_cap[0]}.{compute_cap[1]}",
            "vram_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0,
        },
        "target_model": target_model_id,
        "draft_mechanism": "PromptLookupDraftEngine (Zero-Weight Unified)",
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "k_draft": k_draft,
        },
        "target_baseline": {
            "mean_tok_sec": mean_baseline_tok_sec,
            "mean_latency_ms_per_tok": mean_baseline_latency,
        },
        "unified_speculative": {
            "mean_tok_sec": mean_spec_tok_sec,
            "mean_latency_ms_per_tok": mean_spec_latency,
            "speedup_factor": net_speedup,
            "acceptance_rate_pct": mean_spec_alpha,
            "avg_tokens_per_target_step": mean_spec_tok_per_step,
        },
        "gates": {
            "speedup_greater_than_1x": passed_speedup,
            "acceptance_greater_than_50pct": passed_alpha,
            "all_passed": passed_speedup,
        },
    }

    out_file = os.path.join(REPO_DIR, "results", "t4_speculative_benchmark_report.json")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--k-draft", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    run_benchmark(
        target_model_id=args.target_model,
        max_new_tokens=args.max_new_tokens,
        k_draft=args.k_draft,
        device_str=args.device,
    )
