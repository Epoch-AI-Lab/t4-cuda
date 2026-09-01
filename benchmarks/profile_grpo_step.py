#!/usr/bin/env python3
"""Phase-split profiler for one GRPO training run: how much of each step is
rollout generation vs backward vs optimizer? Decides which t4-cuda kernel
to integrate next. Run on a T4.
"""
import subprocess, time, os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# install deps if missing (idempotent)
subprocess.run("pip install -q trl datasets transformers accelerate", shell=True)

import torch
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
import transformers

TIMES = {"generate": 0.0, "opt": 0.0}

_gen = transformers.GenerationMixin.generate
def timed_generate(self, *a, **kw):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    out = _gen(self, *a, **kw)
    torch.cuda.synchronize()
    TIMES["generate"] += time.perf_counter() - t0
    return out
transformers.GenerationMixin.generate = timed_generate

_opt = torch.optim.Optimizer.step
def timed_opt(self, *a, **kw):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    r = _opt(self, *a, **kw)
    torch.cuda.synchronize()
    TIMES["opt"] += time.perf_counter() - t0
    return r
torch.optim.Optimizer.step = timed_opt

N_STEPS = 8
ds = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=42)
ds = ds.map(lambda ex: {
    "prompt": [{"role": "user", "content": ex["question"] + "\nEnd your solution with: #### <answer>"}],
    "answer": ex["answer"].split("####")[-1].strip(),
}, remove_columns=["question", "answer"])

cfg = GRPOConfig(
    output_dir="/content/results/grpo_profile",
    per_device_train_batch_size=16, num_generations=8,
    max_completion_length=256, learning_rate=1e-6,
    max_steps=N_STEPS, logging_steps=1, save_strategy="no", report_to="none",
    bf16=False, fp16=True, gradient_checkpointing=True,
)
trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    reward_funcs=lambda completions, answer, **kw: [
        1.0 if (kw is not None) else 0.0 for _ in completions  # placeholder, replaced below
    ],
    args=cfg, train_dataset=ds,
)
# real reward fn
import re
ANS = re.compile(r"####\s*(-?[0-9][0-9,\.]*)")
def reward_correct(completions, answer, **kw):
    out = []
    for c, g in zip(completions, answer):
        if isinstance(c, list):
            c = c[-1]["content"] if c and isinstance(c[-1], dict) else str(c)
        m = ANS.search(c)
        pred = m.group(1).replace(",", "").rstrip(".") if m else None
        out.append(1.0 if pred == g.replace(",", "").rstrip(".") else 0.0)
    return out
trainer.reward_funcs = [reward_correct]

t0 = time.perf_counter()
trainer.train()
wall = time.perf_counter() - t0

gen, opt = TIMES["generate"], TIMES["opt"]
other = wall - gen - opt
print(f"\n=== PHASE SPLIT over {N_STEPS} steps (wall {wall:.1f}s) ===")
print(f"rollout generation : {gen:7.1f}s  ({100*gen/wall:5.1f}%)")
print(f"optimizer step     : {opt:7.1f}s  ({100*opt/wall:5.1f}%)")
print(f"fwd/bwd + rest     : {other:7.1f}s  ({100*other/wall:5.1f}%)")
print(f"per step           : {wall/N_STEPS:.2f}s")
