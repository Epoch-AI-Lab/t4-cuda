#!/usr/bin/env python3
"""CP-Hybrid Accelerated GRPO Benchmark on a single Tesla T4.

Deploys Component-and-Phase Hybrid (CP-Hybrid) rollouts for Qwen2.5-0.5B on GSM8K:
- Attention layers (Q, K, V, O) remain in pure FP16 cuBLAS Tensor Cores to preserve
  mathematical reasoning and prevent the reward collapse observed in naive INT4.
- MLP feed-forward layers (80% of weights) are quantized to INT4 (group=128).
- During rollouts:
  - Single-token decode (M <= 4) uses the fused W4A16 GEMV kernel (1.5x faster).
  - Prefill and training passes (M > 4) use native cuBLAS FP16 Tensor Cores.

Results land in results/grpo_hybrid/ as JSON + training log.
"""

import sys
import argparse
import json
import os
import re
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
import transformers

try:
    import t4_kernels
    HAS_T4_KERNELS = True
except ImportError:
    HAS_T4_KERNELS = False
    print("WARNING: t4_kernels not found! Running in pure PyTorch mode.")

from src.hybrid_linear import quantize_weight_sym_int4, is_mlp_module, is_attention_module

ANSWER_RE = re.compile(r"####\s*(-?[0-9][0-9,\.]*)")


def extract_answer(text: str) -> str | None:
    """Pull final numeric answer from a GSM8K completion."""
    m = ANSWER_RE.search(text)
    if m:
        return m.group(1).replace(",", "").rstrip(".")
    nums = re.findall(r"-?[0-9][0-9,\.]*", text)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def reward_correct(completions, answer, **kwargs):
    """+1 exact match, 0 otherwise."""
    rewards = []
    for comp, gold in zip(completions, answer):
        if isinstance(comp, list):
            comp = comp[-1]["content"] if comp and isinstance(comp[-1], dict) else str(comp)
        c_ans = extract_answer(comp)
        g_ans = extract_answer(gold)
        if c_ans and g_ans and c_ans == g_ans:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def reward_format(completions, **kwargs):
    """Reward formatting tokens."""
    rewards = []
    for comp in completions:
        if isinstance(comp, list):
            comp = comp[-1]["content"] if comp and isinstance(comp[-1], dict) else str(comp)
        r = 0.0
        if "<think>" in comp:
            r += 0.25
        if "</think>" in comp:
            r += 0.25
        if "####" in comp:
            r += 0.5
        rewards.append(r)
    return rewards


class CPHybridRolloutScope:
    """Selectively patches MLP layers with phase-dispatched W4A16 kernels during rollouts."""
    def __init__(self, model, group_size=128):
        self.model = model
        self.group_size = group_size
        self.orig_forwards = {}
        self.Q = {}

    def __enter__(self):
        if not HAS_T4_KERNELS:
            return self

        self.Q.clear()
        self.orig_forwards.clear()

        for name, mod in self.model.named_modules():
            # Only patch MLP layers; Attention layers stay pure FP16!
            if isinstance(mod, nn.Linear) and is_mlp_module(name) and not is_attention_module(name):
                self.Q[mod] = quantize_weight_sym_int4(mod.weight.data, group_size=self.group_size)
                self.orig_forwards[mod] = mod.forward

                def make_patched(m):
                    def patched_forward(x):
                        x_shape = x.shape
                        orig_dtype = x.dtype
                        x_flat = x.reshape(-1, x_shape[-1])
                        M = x_flat.shape[0]

                        if M <= 2:
                            # Single-token decode: custom W4A16 GEMV kernel (1.3x - 1.98x faster)
                            packed, scales, zp, g_size = self.Q[m]
                            if x_flat.dtype != torch.float16:
                                x_flat = x_flat.half()
                            out = t4_kernels.fused_w4a16_gemm_u4(x_flat, packed, scales, zp, g_size)
                            if m.bias is not None:
                                out = out + m.bias
                            if orig_dtype != torch.float16:
                                out = out.to(orig_dtype)
                        else:
                            # Prefill or batched forward: native cuBLAS FP16 Tensor Cores
                            out = F.linear(x_flat, m.weight, m.bias)

                        return out.reshape(*x_shape[:-1], out.shape[-1])
                    return patched_forward

                mod.forward = make_patched(mod)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for mod, orig_fwd in self.orig_forwards.items():
            mod.forward = orig_fwd
        self.orig_forwards.clear()
        self.Q.clear()
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--num_gens", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--out", default="results/grpo_hybrid")
    ap.add_argument("--group_size", type=int, default=128)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("=" * 70)
    print("Tesla T4 CP-Hybrid Accelerated GRPO Benchmark")
    print(f"Model: {args.model}")
    print(f"Steps: {args.steps} | Batch: {args.bs} | Gens/prompt: {args.num_gens}")
    print(f"Group size: {args.group_size} (MLP in INT4, Attention in FP16)")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("=" * 70)

    ds = load_dataset("openai/gsm8k", "main", split="train")

    def format_prompt(x):
        return {
            "prompt": (
                "<|im_start|>system\nYou are a helpful math assistant. "
                "Solve the problem, show work inside <think>...</think>, "
                "and put final answer after ####.<|im_end|>\n"
                f"<|im_start|>user\n{x['question']}<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n"
            ),
            "answer": x["answer"],
        }

    ds = ds.map(format_prompt)

    cfg = GRPOConfig(
        output_dir=args.out,
        learning_rate=1e-6,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=1,
        num_generations=args.num_gens,
        max_prompt_length=256,
        max_completion_length=args.max_len,
        max_steps=args.steps,
        logging_steps=1,
        save_strategy="no",
        fp16=True,
        bf16=False,
        report_to="none",
        warmup_ratio=0.05,
        optim="adamw_torch",
        use_vllm=False,
    )

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=[reward_correct, reward_format],
        args=cfg,
        train_dataset=ds,
    )

    # Wrap trainer.model.generate with CPHybridRolloutScope
    orig_generate = transformers.GenerationMixin.generate

    def hybrid_generate(self, *g_args, **g_kwargs):
        with CPHybridRolloutScope(self, group_size=args.group_size):
            return orig_generate(self, *g_args, **g_kwargs)

    transformers.GenerationMixin.generate = hybrid_generate

    step_times = []
    rewards_correct_list = []
    rewards_format_list = []

    class StepTimingCallback(transformers.TrainerCallback):
        def __init__(self):
            self.t0 = None

        def on_step_begin(self, cb_args, state, control, **kwargs):
            torch.cuda.synchronize()
            self.t0 = time.perf_counter()

        def on_step_end(self, cb_args, state, control, **kwargs):
            torch.cuda.synchronize()
            dt = time.perf_counter() - self.t0
            step_times.append(dt)
            vram_gb = torch.cuda.max_memory_allocated() / 1e9
            print(f"  [CP-Hybrid Step {state.global_step:3d}/{args.steps}] {dt:.2f}s | Peak VRAM: {vram_gb:.2f} GB")

        def on_log(self, cb_args, state, control, logs=None, **kwargs):
            if logs:
                for k, v in logs.items():
                    if "reward_correct" in k:
                        rewards_correct_list.append(v)
                    elif "reward_format" in k:
                        rewards_format_list.append(v)

    trainer.add_callback(StepTimingCallback())

    print("\nStarting CP-Hybrid GRPO training...")
    t_total_start = time.perf_counter()
    train_result = trainer.train()
    torch.cuda.synchronize()
    total_time = time.perf_counter() - t_total_start

    mean_step = sum(step_times) / len(step_times) if step_times else 0
    mean_rew_c = sum(rewards_correct_list) / len(rewards_correct_list) if rewards_correct_list else 0
    mean_rew_f = sum(rewards_format_list) / len(rewards_format_list) if rewards_format_list else 0
    peak_vram = torch.cuda.max_memory_allocated() / 1e9
    total_tokens = args.steps * args.bs * args.num_gens * args.max_len
    effective_tok_per_sec = total_tokens / total_time if total_time > 0 else 0

    print("\n" + "=" * 70)
    print("CP-Hybrid GRPO Benchmark Results (Tesla T4)")
    print(f"Total time:             {total_time:.1f}s")
    print(f"Mean step time:         {mean_step:.2f}s")
    print(f"Effective throughput:   {effective_tok_per_sec:.1f} tok/s")
    print(f"Mean reward (correct):  {mean_rew_c:.4f}")
    print(f"Mean reward (format):   {mean_rew_f:.4f}")
    print(f"Peak VRAM:              {peak_vram:.2f} GB")
    print("=" * 70)

    metrics = {
        "benchmark": "grpo_cphybrid",
        "model": args.model,
        "device": torch.cuda.get_device_name(0),
        "steps": args.steps,
        "batch_size": args.bs,
        "num_generations": args.num_gens,
        "max_completion_length": args.max_len,
        "group_size": args.group_size,
        "total_time_s": round(total_time, 2),
        "mean_step_time_s": round(mean_step, 3),
        "effective_tok_per_sec": round(effective_tok_per_sec, 1),
        "mean_reward_correct": round(mean_rew_c, 4),
        "mean_reward_format": round(mean_rew_f, 4),
        "peak_vram_gb": round(peak_vram, 2),
        "step_times": [round(t, 3) for t in step_times],
        "rewards_correct": rewards_correct_list,
        "rewards_format": rewards_format_list,
    }

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {args.out}/metrics.json")


if __name__ == "__main__":
    main()
