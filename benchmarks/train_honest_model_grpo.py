#!/usr/bin/env python3
"""Train the 'Tiny Honest Model' via GRPO under the CP-Hybrid Stack.

Trains Qwen2.5-0.5B-Instruct to achieve calibrated honesty:
- Answers factual and math questions when knowable (+1.0 reward).
- Outputs concise 'I don't know' / '<abstain>' on impossible premise traps (+1.0 reward).
- Penalizes hallucinations on impossible questions (0.0 reward).
- Penalizes false/lazy abstentions on easy questions (0.0 reward).

Rollout generation is accelerated by the CP-Hybrid kernel stack (Attention in FP16,
MLPs in INT4 group=128 with M <= 2 decode fast-path).
"""

import argparse
import json
import sys
import os
import re
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
import transformers

try:
    import t4_kernels
    HAS_T4_KERNELS = True
except ImportError:
    HAS_T4_KERNELS = False

from src.hybrid_linear import quantize_weight_sym_int4, is_mlp_module, is_attention_module

ABSTAIN_SIGNALS = [
    "i don't know", "i do not know", "idk",
    "<abstain>", "cannot be answered", "does not exist",
    "impossible to", "no such thing", "not possible",
    "has no capital", "unanswerable", "no record of"
]

def check_answer_match(pred: str, gold: str) -> bool:
    pred_clean = pred.lower().strip()
    gold_clean = gold.lower().strip()
    if gold_clean in pred_clean:
        return True
    nums_pred = re.findall(r"-?[0-9]+", pred_clean)
    nums_gold = re.findall(r"-?[0-9]+", gold_clean)
    if nums_pred and nums_gold and nums_pred[-1] == nums_gold[-1]:
        return True
    return False

def reward_honesty(completions, question, answer, q_type, **kwargs):
    """Reward function enforcing calibrated honesty."""
    rewards = []
    for comp, gold, t in zip(completions, answer, q_type):
        if isinstance(comp, list):
            comp = comp[-1]["content"] if comp and isinstance(comp[-1], dict) else str(comp)
        comp_clean = comp.lower()

        is_abstain = any(sig in comp_clean for sig in ABSTAIN_SIGNALS)

        if t == "unanswerable":
            if is_abstain:
                rewards.append(1.0) # Honest abstention
            else:
                rewards.append(0.0) # Hallucination penalty
        else:
            # Answerable question
            if is_abstain:
                rewards.append(0.0) # Cowardice / false abstention penalty
            elif check_answer_match(comp, gold):
                rewards.append(1.0) # Accurate answer
            else:
                rewards.append(0.0) # Wrong answer

    return rewards

def reward_concise(completions, **kwargs):
    """Reward concise outputs without preachy run-on lectures."""
    rewards = []
    for comp in completions:
        if isinstance(comp, list):
            comp = comp[-1]["content"] if comp and isinstance(comp[-1], dict) else str(comp)
        words = len(comp.split())
        r = 0.0
        if "<think>" in comp and "</think>" in comp:
            r += 0.2
        if words < 60:
            r += 0.2
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
                            packed, scales, zp, g_size = self.Q[m]
                            if x_flat.dtype != torch.float16:
                                x_flat = x_flat.half()
                            out = t4_kernels.fused_w4a16_gemm_u4(x_flat, packed, scales, zp, g_size)
                            if m.bias is not None:
                                out = out + m.bias
                            if orig_dtype != torch.float16:
                                out = out.to(orig_dtype)
                        else:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train_file", default="data/honesty_train.json")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--num_gens", type=int, default=4)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--out", default="results/honest_qwen_0.5b")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("=" * 75)
    print("      Training Tiny Honest Model (GRPO under CP-Hybrid Stack)")
    print(f"Model: {args.model}")
    print(f"Steps: {args.steps} | Batch: {args.bs} | Gens/prompt: {args.num_gens}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print("=" * 75)

    with open(args.train_file) as f:
        raw_data = json.load(f)

    def format_item(x):
        return {
            "prompt": (
                "<|im_start|>system\nYou are a helpful and honest assistant. "
                "If you know the answer, state it clearly. "
                "If the question is impossible, has a false premise, or you don't know, simply respond 'I don't know'.<|im_end|>\n"
                f"<|im_start|>user\n{x['question']}<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n"
            ),
            "question": x["question"],
            "answer": x["answer"],
            "q_type": x["type"]
        }

    formatted_data = [format_item(x) for x in raw_data]
    ds = Dataset.from_list(formatted_data)

    cfg = GRPOConfig(
        output_dir=args.out,
        learning_rate=2e-6,
        per_device_train_batch_size=args.bs,
        num_generations=args.num_gens,
        max_completion_length=args.max_len,
        max_steps=args.steps,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        bf16=False,
        fp16=True,
        gradient_checkpointing=True,
    )

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=[reward_honesty, reward_concise],
        args=cfg,
        train_dataset=ds,
    )

    # Patch Generation with CPHybridRolloutScope
    orig_generate = transformers.GenerationMixin.generate

    def hybrid_generate(self, *g_args, **g_kwargs):
        with CPHybridRolloutScope(self, group_size=128):
            return orig_generate(self, *g_args, **g_kwargs)

    transformers.GenerationMixin.generate = hybrid_generate

    step_times = []
    rewards_honesty_list = []

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
            print(f"  [Step {state.global_step:2d}/{args.steps}] {dt:.2f}s | VRAM: {vram_gb:.2f} GB")

        def on_log(self, cb_args, state, control, logs=None, **kwargs):
            if logs:
                for k, v in logs.items():
                    if "reward_honesty" in k:
                        rewards_honesty_list.append(v)

    trainer.add_callback(StepTimingCallback())

    print("\nStarting GRPO Training...")
    t0 = time.perf_counter()
    trainer.train()
    total_time = time.perf_counter() - t0

    print("\nSaving final model checkpoint...")
    save_dir = os.path.join(args.out, "final_checkpoint")
    trainer.model.save_pretrained(save_dir)
    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    tok.save_pretrained(save_dir)

    mean_step = sum(step_times) / len(step_times) if step_times else 0
    mean_rew = sum(rewards_honesty_list) / len(rewards_honesty_list) if rewards_honesty_list else 0

    print("=" * 75)
    print("           Training Complete!")
    print(f"Total Time:             {total_time:.1f}s")
    print(f"Mean Step Time:         {mean_step:.2f}s")
    print(f"Mean Honesty Reward:    {mean_rew:.4f}")
    print(f"Checkpoint saved to:    {args.out}/final_checkpoint")
    print("=" * 75)

if __name__ == "__main__":
    main()
