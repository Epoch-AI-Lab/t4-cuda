#!/usr/bin/env python3
"""Calibrated Low-Precision GRPO Reasoning Benchmark on Tesla T4.

Evaluates reward integrity and rollout step time across:
1. FP16 Baseline: Standard PyTorch cuBLAS.
2. Naive INT4 Rollouts: All linear layers quantized (the Item 2 regime with 2.5% accuracy).
3. Calibrated CP-Hybrid Rollouts: Attention preserved in FP16, MLPs in group=128 INT4,
   with WMMA Tensor Core execution for batched rollout sequences (M=64).

Demonstrates that Calibrated CP-Hybrid recovers math reasoning accuracy on GSM8K
while preserving low-precision memory savings and accelerating batched decode.
"""

import argparse
import json
import os
import re
import sys
import time

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "src"))

import torch
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
import transformers

from calibrated_rollout import CalibratedRolloutScope

ANSWER_RE = re.compile(r"####\s*(-?[0-9][0-9,\.]*)")


def extract_answer(text: str) -> str | None:
    """Pull the final numeric answer from a GSM8K completion."""
    m = ANSWER_RE.search(text)
    if m:
        return m.group(1).replace(",", "").rstrip(".")
    nums = re.findall(r"-?[0-9][0-9,\.]*", text)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def reward_correct(completions, answer, **kwargs):
    """Exact match reward against gold GSM8K answer."""
    rewards = []
    for comp, gold in zip(completions, answer):
        if isinstance(comp, list):
            comp = comp[-1]["content"] if comp and isinstance(comp[-1], dict) else str(comp)
        pred = extract_answer(comp)
        gold = gold.replace(",", "").rstrip(".")
        rewards.append(1.0 if pred == gold else 0.0)
    return rewards


def run_benchmark(mode: str, max_steps: int = 15, seed: int = 42):
    print("=" * 80)
    print(f"  Tesla T4 GRPO Rollout Reasoning Benchmark: Mode = {mode.upper()}")
    print("=" * 80)

    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Loading {model_id} in FP16...")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load GSM8K
    print("Loading GSM8K train split...")
    ds = load_dataset("openai/gsm8k", "main", split="train")

    def format_prompt(x):
        prompt = (
            f"<|im_start|>system\nYou are a helpful math assistant. Answer clearly with reasoning, "
            f"ending with #### <number>.<|im_end|>\n"
            f"<|im_start|>user\n{x['question']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        return {"prompt": prompt, "answer": x["answer"]}

    ds = ds.map(format_prompt)

    out_dir = os.path.join(REPO_DIR, f"results/grpo_reasoning_{mode}")
    os.makedirs(out_dir, exist_ok=True)

    training_args = GRPOConfig(
        output_dir=out_dir,
        learning_rate=1e-6,
        max_steps=max_steps,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        num_generations=8,
        max_prompt_length=256,
        max_completion_length=256,
        temperature=0.7,
        seed=seed,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_correct,
        args=training_args,
        train_dataset=ds,
    )

    # Wire Rollout Scope if requested
    orig_generate = transformers.GenerationMixin.generate

    if mode == "calibrated_cp_hybrid":
        print("--> Engaging Calibrated CP-Hybrid Rollout Scope (Selective Attention FP16 + MLP INT4)...")
        def patched_generate(self, *args, **kwargs):
            with CalibratedRolloutScope(self, group_size=128, quantize_attention=False):
                return orig_generate(self, *args, **kwargs)
        transformers.GenerationMixin.generate = patched_generate

    elif mode == "naive_int4":
        print("--> Engaging Naive INT4 Rollout Scope (All Linear Layers Quantized)...")
        def patched_generate(self, *args, **kwargs):
            with CalibratedRolloutScope(self, group_size=128, quantize_attention=True):
                return orig_generate(self, *args, **kwargs)
        transformers.GenerationMixin.generate = patched_generate

    else:
        print("--> Using Standard FP16 Baseline Rollout...")

    step_times = []
    rewards_history = []

    class TelemetryCallback(transformers.TrainerCallback):
        def __init__(self):
            self.t_start = None

        def on_step_begin(self, args, state, control, **kwargs):
            torch.cuda.synchronize()
            self.t_start = time.perf_counter()

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "reward" in logs:
                rewards_history.append(logs["reward"])
                print(f"  [Step {state.global_step:02d}] Mean Reward: {logs['reward']:.4f}")

        def on_step_end(self, args, state, control, **kwargs):
            torch.cuda.synchronize()
            dt = time.perf_counter() - self.t_start
            step_times.append(dt)
            vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            print(f"  [Step {state.global_step:02d}] Step Time: {dt:.2f}s | Peak VRAM: {vram_gb:.2f} GB")

    trainer.add_callback(TelemetryCallback())

    print(f"\nStarting {max_steps} GRPO steps under {mode}...")
    t0_total = time.perf_counter()
    trainer.train()
    torch.cuda.synchronize()
    total_time = time.perf_counter() - t0_total

    # Restore original generate
    transformers.GenerationMixin.generate = orig_generate

    avg_step_time = sum(step_times) / len(step_times) if step_times else 0.0
    mean_reward = sum(rewards_history) / len(rewards_history) if rewards_history else 0.0
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    summary = {
        "mode": mode,
        "max_steps": max_steps,
        "total_time_s": total_time,
        "avg_step_time_s": avg_step_time,
        "mean_reward": mean_reward,
        "peak_vram_gb": peak_vram_gb,
        "step_times": step_times,
        "rewards_history": rewards_history,
    }

    out_file = os.path.join(out_dir, "summary.json")
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Benchmark Complete ({mode.upper()}):")
    print(f"  Mean Reward:   {mean_reward:.4f}")
    print(f"  Avg Step Time: {avg_step_time:.2f}s")
    print(f"  Peak VRAM:     {peak_vram_gb:.2f} GB")
    print(f"  Summary saved: {out_file}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline_fp16", "naive_int4", "calibrated_cp_hybrid"], default="calibrated_cp_hybrid")
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_benchmark(mode=args.mode, max_steps=args.steps, seed=args.seed)
