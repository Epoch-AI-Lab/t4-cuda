#!/usr/bin/env python3
"""Baseline GRPO benchmark on a single Tesla T4.

Purpose: measure tokens/sec, VRAM peak, and reward curve for on-policy RL
(TRL GRPOTrainer, Qwen2.5-0.5B-Instruct, GSM8K verifiable rewards) with NO
t4-cuda kernels yet. This is the baseline the fused-kernel run has to beat.

Run on Colab/Kaggle T4:
    pip install trl datasets transformers accelerate
    python benchmarks/benchmark_grpo_t4.py --steps 100

Results land in results/grpo_baseline/ as JSON + training log.
"""

import argparse
import json
import os
import re
import time
import torch
import transformers
from transformers import AutoTokenizer

from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

ANSWER_RE = re.compile(r"####\s*(-?[0-9][0-9,\.]*)")
_tokenized_gen_tokens = 0
_tokenizer_cache = None


def extract_answer(text: str) -> str | None:
    """Pull the final numeric answer from a GSM8K completion (or gold)."""
    m = ANSWER_RE.search(text)
    if m:
        return m.group(1).replace(",", "").rstrip(".")
    # fallback: last number in the text (models rarely use #### unprompted)
    nums = re.findall(r"-?[0-9][0-9,\.]*", text)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def reward_correct(completions, answer, **kwargs):
    """+1 exact match, 0 otherwise. Verifiable, no judge needed."""
    global _tokenized_gen_tokens, _tokenizer_cache
    rewards = []
    for comp, gold in zip(completions, answer):
        # TRL 1.x returns conversational completions as message lists
        if isinstance(comp, list):
            comp = comp[-1]["content"] if comp and isinstance(comp[-1], dict) else str(comp)
        pred = extract_answer(comp)
        gold = gold.replace(",", "").rstrip(".")
        rewards.append(1.0 if pred == gold else 0.0)
        if _tokenizer_cache is not None:
            try:
                _tokenized_gen_tokens += len(_tokenizer_cache.encode(comp, add_special_tokens=False))
            except Exception:
                pass
    return rewards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8,
                    help="prompts per step")
    ap.add_argument("--num-generations", type=int, default=8,
                    help="completions per prompt (GRPO group size)")
    ap.add_argument("--max-completion-len", type=int, default=256)
    ap.add_argument("--out-dir", default="results/grpo_baseline")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    ds = load_dataset("openai/gsm8k", "main", split="train")
    ds = ds.shuffle(seed=42)
    # strip the #### answer from the prompt, keep it in `answer` column for the reward fn
    ds = ds.map(
        lambda ex: {
            "prompt": [
                {"role": "user",
                 "content": ex["question"] +
                 "\nEnd your solution with: #### <answer>"}
            ],
            "answer": ex["answer"].split("####")[-1].strip(),
        },
        remove_columns=["question", "answer"],
    )

    cfg = GRPOConfig(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.batch_size * args.num_generations // 4,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_len,
        learning_rate=1e-6,
        max_steps=args.steps,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        bf16=False,  # T4 is Turing: no bf16
        fp16=True,
        gradient_checkpointing=True,
    )

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=reward_correct,
        args=cfg,
        train_dataset=ds,
    )

    global _tokenizer_cache
    try:
        _tokenizer_cache = AutoTokenizer.from_pretrained(args.model)
    except Exception:
        _tokenizer_cache = None

    _orig_generate = transformers.GenerationMixin.generate
    actual_gen_tokens = 0

    def hooked_generate(self, *gen_args, **gen_kwargs):
        nonlocal actual_gen_tokens
        res = _orig_generate(self, *gen_args, **gen_kwargs)
        try:
            input_ids = gen_kwargs.get("input_ids", None)
            if input_ids is None and len(gen_args) > 0 and isinstance(gen_args[0], torch.Tensor):
                input_ids = gen_args[0]
            seqs = res.sequences if hasattr(res, "sequences") else res
            if isinstance(seqs, torch.Tensor):
                in_len = input_ids.shape[1] if (input_ids is not None and isinstance(input_ids, torch.Tensor)) else 0
                gen = seqs[:, in_len:]
                pad_id = getattr(self.generation_config, "pad_token_id", None)
                if pad_id is None:
                    pad_id = getattr(self.config, "pad_token_id", None)
                if pad_id is not None:
                    actual_gen_tokens += int((gen != pad_id).sum().item())
                else:
                    actual_gen_tokens += int(gen.numel())
        except Exception:
            pass
        return res

    transformers.GenerationMixin.generate = hooked_generate

    t0 = time.time()
    trainer.train()
    wall = time.time() - t0

    # Count actual non-pad generated tokens dynamically
    total_gen_tokens = actual_gen_tokens if actual_gen_tokens > 0 else _tokenized_gen_tokens

    metrics = {
        "model": args.model,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "num_generations": args.num_generations,
        "wall_seconds": round(wall, 1),
        "approx_generated_tokens": total_gen_tokens,
        "approx_tokens_per_sec": round(total_gen_tokens / wall, 1),
        "train_runtime": trainer.state.log_history[-1].get("train_runtime"),
        "final_reward": trainer.state.log_history[-1].get("reward", None),
        "reward_history": [
            {"step": h.get("step"), "reward": h.get("reward", h.get("rewards/reward_correct/mean", h.get("reward/mean")))}
            for h in trainer.state.log_history
            if "reward" in h or "rewards/reward_correct/mean" in h or "reward/mean" in h
        ],
    }
    out_path = os.path.join(args.out_dir, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"\nWrote {out_path}")
    print("NOTE: peak VRAM is visible in nvidia-smi / Colab GPU panel; "
          "add torch.cuda.max_memory_allocated() capture in the next pass.")


if __name__ == "__main__":
    main()
