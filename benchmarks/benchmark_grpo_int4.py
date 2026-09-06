#!/usr/bin/env python3
"""INT4-Accelerated GRPO benchmark on a single Tesla T4.

Wires the verified per-group (group=128) INT4 W4A16 GEMM kernel into the rollout
phase of TRL GRPOTrainer for Qwen2.5-0.5B-Instruct on GSM8K.
Rollout generation runs in INT4 with fused kernels, while backward pass and
optimizer steps remain in FP16.

Results land in results/grpo_int4/ as JSON + training log.
"""

import argparse
import json
import os
import re
import time

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
import transformers

try:
    import t4_kernels
    HAS_T4_KERNELS = True
except ImportError:
    HAS_T4_KERNELS = False
    print("WARNING: t4_kernels not found! Running in pure PyTorch mode.")

ANSWER_RE = re.compile(r"####\s*(-?[0-9][0-9,\.]*)")
_tokenized_gen_tokens = 0
_tokenizer_cache = None


def extract_answer(text: str) -> str | None:
    """Pull the final numeric answer from a GSM8K completion (or gold)."""
    m = ANSWER_RE.search(text)
    if m:
        return m.group(1).replace(",", "").rstrip(".")
    nums = re.findall(r"-?[0-9][0-9,\.]*", text)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def reward_correct(completions, answer, **kwargs):
    """+1 exact match, 0 otherwise. Verifiable, no judge needed."""
    global _tokenized_gen_tokens, _tokenizer_cache
    rewards = []
    for comp, gold in zip(completions, answer):
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


def quantize_weight_sym(W, group_size=128):
    """Symmetric per-group INT4 quantization (group=128, u4 nibbles, zp=8)."""
    out_f, in_f = W.shape
    assert in_f % 8 == 0, in_f
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
        scales = scale.squeeze(2).t().contiguous().half().cuda()
        zps = torch.full_like(scales, 8.0).cuda()
        return packed.cuda(), scales, zps, group_size
    else:
        amax = Wf.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
        scale = amax / 7.0
        q = torch.clamp(torch.round(Wf / scale) + 8, 0, 15).to(torch.int32)
        q = q.t().contiguous().cpu()
        packed = torch.zeros(in_f // 8, out_f, dtype=torch.int32)
        for i in range(8):
            packed |= q[i::8, :] << (4 * i)
        scales = scale.squeeze(1).half().unsqueeze(0).contiguous().cuda()
        zps = torch.full((1, out_f), 8.0, dtype=torch.float16, device="cuda")
        return packed.cuda(), scales, zps, 0


class Int4RolloutScope:
    """Context manager that temporarily patches linear layers with fused INT4 kernels."""
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
            if isinstance(mod, nn.Linear) and "lm_head" not in name:
                self.Q[mod] = quantize_weight_sym(mod.weight.data, group_size=self.group_size)
                self.orig_forwards[mod] = mod.forward

                def make_patched(m):
                    def patched_forward(x):
                        packed, scales, zp, g_size = self.Q[m]
                        shape = x.shape
                        orig_dtype = x.dtype
                        x2 = x.reshape(-1, shape[-1])
                        if x2.dtype != torch.float16:
                            x2 = x2.half()
                        out = t4_kernels.fused_w4a16_gemm_u4(x2, packed, scales, zp, g_size)
                        if m.bias is not None:
                            out = out + m.bias
                        if orig_dtype != torch.float16:
                            out = out.to(orig_dtype)
                        return out.reshape(*shape[:-1], out.shape[-1])
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
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8,
                    help="prompts per step")
    ap.add_argument("--num-generations", type=int, default=8,
                    help="completions per prompt (GRPO group size)")
    ap.add_argument("--max-completion-len", type=int, default=256)
    ap.add_argument("--group-size", type=int, default=128,
                    help="K group size for INT4 quantization")
    ap.add_argument("--out-dir", default="results/grpo_int4")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Hook transformers.GenerationMixin.generate to run rollouts under Int4RolloutScope
    _orig_generate = transformers.GenerationMixin.generate

    generation_time_total = 0.0
    actual_gen_tokens = 0

    def int4_hooked_generate(self, *gen_args, **gen_kwargs):
        nonlocal generation_time_total, actual_gen_tokens
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with Int4RolloutScope(self, group_size=args.group_size):
            res = _orig_generate(self, *gen_args, **gen_kwargs)
        torch.cuda.synchronize()
        generation_time_total += time.perf_counter() - t0
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

    transformers.GenerationMixin.generate = int4_hooked_generate

    ds = load_dataset("openai/gsm8k", "main", split="train")
    ds = ds.shuffle(seed=42)
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
        bf16=False,
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
        from transformers import AutoTokenizer
        _tokenizer_cache = AutoTokenizer.from_pretrained(args.model)
    except Exception:
        _tokenizer_cache = None

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    wall = time.time() - t0

    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)

    # Count actual non-pad generated tokens dynamically
    total_gen_tokens = actual_gen_tokens if actual_gen_tokens > 0 else _tokenized_gen_tokens

    metrics = {
        "model": args.model,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "num_generations": args.num_generations,
        "group_size": args.group_size,
        "wall_seconds": round(wall, 1),
        "generation_seconds": round(generation_time_total, 1),
        "generation_fraction": round(generation_time_total / max(wall, 1e-6), 3),
        "approx_generated_tokens": total_gen_tokens,
        "approx_tokens_per_sec": round(total_gen_tokens / wall, 1),
        "peak_vram_gb": round(peak_vram, 2),
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


if __name__ == "__main__":
    main()
