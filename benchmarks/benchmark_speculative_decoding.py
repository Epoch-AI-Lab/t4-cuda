#!/usr/bin/env python3
"""Tesla T4 Speculative Decoding Benchmark (H20 / Milestone Phase 2).

Evaluates real wall-clock speedup and acceptance rate of sub-4-bit speculative decoding
on physical Tesla T4 silicon:
  - Draft Model: Qwen2.5-0.5B-Instruct quantized to W4A16 INT4 via CP-Hybrid kernel.
  - Target Model: Qwen2.5-1.5B-Instruct (or Qwen2.5-0.5B FP16 baseline).

Measures:
  1. Tokens/sec throughput
  2. Latency percentiles (p50, p90, p99)
  3. Acceptance rate (alpha %) across draft lengths K in {2, 3, 4}
  4. Speedup factor vs standalone target decoding
"""

import argparse
import json
import os
import sys
import time
import torch
import transformers

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "src"))

from hybrid_linear import CPHybridScope
from speculative_engine import SpeculativeEngine

TEST_PROMPTS = [
    "Explain the difference between a mutex and a semaphore in operating systems.",
    "Solve step-by-step: If a train travels at 60 mph for 2.5 hours, how far does it travel?",
    "Write a Python function to check whether a given binary tree is symmetric.",
    "What causes the northern lights (aurora borealis)?",
    "Describe how a transformer attention mechanism computes query-key dot products.",
]


def run_benchmark(
    target_model_id: str = "Qwen/Qwen2.5-7B-Instruct",
    draft_model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    max_tokens: int = 24,
    lookahead_k_list: list = [2, 3],
    load_in_4bit: bool = True,
    compile_draft: bool = False,
    num_prompts: int = 2,
):
    print("=" * 85)
    print("   Tesla T4 Sub-4-Bit Speculative Decoding Engine Benchmark")
    print("=" * 85)
    print(f"Target Model: {target_model_id} ({'4-bit BNB' if load_in_4bit else 'FP16'})")
    print(f"Draft Model:  {draft_model_id} (INT4 CP-Hybrid)")
    print(f"Tokens/run:   {max_tokens}")
    print("-" * 85)

    tokenizer = transformers.AutoTokenizer.from_pretrained(target_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading Target Model ({target_model_id})...")
    if load_in_4bit:
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        target_model = transformers.AutoModelForCausalLM.from_pretrained(
            target_model_id,
            quantization_config=bnb_config,
            device_map="cuda",
        ).eval()
    else:
        target_model = transformers.AutoModelForCausalLM.from_pretrained(
            target_model_id,
            torch_dtype=torch.float16,
            device_map="cuda",
        ).eval()

    print(f"Loading Draft Model ({draft_model_id})...")
    draft_model = transformers.AutoModelForCausalLM.from_pretrained(
        draft_model_id,
        torch_dtype=torch.float16,
        device_map="cuda",
    ).eval()

    # Wrap draft model with CP-Hybrid INT4
    print("Quantizing Draft Model MLPs to symmetric INT4 (group=128)...")
    draft_scope = CPHybridScope(draft_model, group_size=128)
    draft_scope.__enter__()

    if compile_draft:
        print("Compiling Draft Model with torch.compile(mode='reduce-overhead') for CUDA Graph execution...")
        try:
            draft_model = torch.compile(draft_model, mode="reduce-overhead")
            dummy = torch.tensor([[100]], dtype=torch.long, device="cuda")
            with torch.no_grad():
                for _ in range(3):
                    _ = draft_model(dummy)
            torch.cuda.synchronize()
            print("CUDA Graph capture successful!")
        except Exception as e:
            print(f"torch.compile warning: {e}. Falling back to eager.")

    engine = SpeculativeEngine(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
    )

    results = {
        "target_model": target_model_id,
        "draft_model": draft_model_id,
        "runs": {},
    }

    # 1. Target Autoregressive Baseline
    print("\n[1/2] Running Target Autoregressive Baseline...")
    target_tok_rates = []
    target_latencies_p50 = []
    active_prompts = TEST_PROMPTS[:num_prompts]

    for idx, prompt in enumerate(active_prompts):
        formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = tokenizer(formatted, return_tensors="pt").input_ids.cuda()
        _, stats = engine.generate_autoregressive(input_ids, max_new_tokens=max_tokens)
        target_tok_rates.append(stats["tokens_per_sec"])
        target_latencies_p50.append(stats["latency_p50_ms"])
        print(f"  Prompt {idx+1}: {stats['tokens_per_sec']:.1f} tok/s | p50: {stats['latency_p50_ms']:.1f}ms")

    baseline_speed = sum(target_tok_rates) / len(target_tok_rates)
    baseline_p50 = sum(target_latencies_p50) / len(target_latencies_p50)
    results["runs"]["target_baseline"] = {
        "mean_tok_sec": baseline_speed,
        "mean_p50_ms": baseline_p50,
    }
    print(f">> Target Baseline Speed: {baseline_speed:.1f} tok/s | p50: {baseline_p50:.1f} ms")

    # 2. Speculative Decoding with varying K
    for K in lookahead_k_list:
        print(f"\n[2/2] Running Speculative Decoding (K={K})...")
        spec_rates = []
        spec_alphas = []
        spec_tokens_per_step = []

        for idx, prompt in enumerate(active_prompts):
            formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            input_ids = tokenizer(formatted, return_tensors="pt").input_ids.cuda()
            _, stats = engine.generate_speculative(input_ids, max_new_tokens=max_tokens, lookahead_k=K)
            spec_rates.append(stats["tokens_per_sec"])
            spec_alphas.append(stats["acceptance_rate_pct"])
            spec_tokens_per_step.append(stats["avg_tokens_per_step"])
            print(f"  Prompt {idx+1}: {stats['tokens_per_sec']:.1f} tok/s | alpha: {stats['acceptance_rate_pct']:.1f}% | tok/step: {stats['avg_tokens_per_step']:.2f}")

        mean_speed = sum(spec_rates) / len(spec_rates)
        mean_alpha = sum(spec_alphas) / len(spec_alphas)
        mean_tok_step = sum(spec_tokens_per_step) / len(spec_tokens_per_step)
        speedup = mean_speed / baseline_speed

        results["runs"][f"speculative_k_{K}"] = {
            "mean_tok_sec": mean_speed,
            "speedup_factor": speedup,
            "acceptance_rate_pct": mean_alpha,
            "avg_tokens_per_target_step": mean_tok_step,
        }
        print(f">> Speculative (K={K}): {mean_speed:.1f} tok/s | Speedup: {speedup:.2f}x | Alpha: {mean_alpha:.1f}%")

    out_dir = os.path.join(REPO_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "speculative_decoding_benchmark.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 85)
    print(f"{'Configuration':<25} | {'Tokens/sec':<12} | {'Speedup':<10} | {'Alpha (%)':<10} | {'Tok/Step':<10}")
    print("-" * 85)
    print(f"{'Target Baseline (FP16)':<25} | {baseline_speed:10.1f}   | {'1.00x':<10} | {'N/A':<10} | {'1.00':<10}")
    for K in lookahead_k_list:
        run_data = results["runs"][f"speculative_k_{K}"]
        print(f"{'Speculative (K=' + str(K) + ')':<25} | {run_data['mean_tok_sec']:10.1f}   | {run_data['speedup_factor']:8.2f}x  | {run_data['acceptance_rate_pct']:8.1f}%  | {run_data['avg_tokens_per_target_step']:8.2f}")
    print("=" * 85)
    print(f"Results saved to: {out_file}")

    draft_scope.__exit__(None, None, None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--draft-model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--no-4bit", dest="load_in_4bit", action="store_false")
    parser.add_argument("--compile", action="store_true", default=False)
    parser.add_argument("--num-prompts", type=int, default=2)
    args = parser.parse_args()

    run_benchmark(
        target_model_id=args.target_model,
        draft_model_id=args.draft_model,
        max_tokens=args.max_tokens,
        load_in_4bit=args.load_in_4bit,
        compile_draft=args.compile,
        num_prompts=args.num_prompts,
    )
