#!/usr/bin/env python3
"""Self-contained Colab training runner for Baby-Chalk RL.

Designed to be run via `colab exec -s <session> -f scripts/colab_train_rl.py`.
All output saved to /content/chalk_rl_out/.

Steps:
  1. Clone / pull repo on Colab VM
  2. pip install deps
  3. Cold-start SFT (1 epoch on chalk_seeds_500.jsonl, ~2 min on T4)
  4. RL post-training (200 steps on rl_pool.jsonl with competition problems)
"""

import os
import subprocess
import sys

REPO_URL = "https://github.com/Epoch-AI-Lab/t4-cuda.git"
BRANCH = "feat-big-chalk-math-7b"
REPO_DIR = "/content/t4-cuda"
OUT_DIR = "/content/chalk_rl_out"


def run(cmd: str, **kwargs) -> int:
    print(f"\n$ {cmd}")
    r = subprocess.run(cmd, shell=True, **kwargs)
    return r.returncode


def die(msg: str):
    print(f"[FATAL] {msg}", file=sys.stderr)
    sys.exit(1)


# ── 1. Clone or pull ──────────────────────────────────────────────────────────
if not os.path.exists(REPO_DIR):
    rc = run(f"git clone -b {BRANCH} {REPO_URL} {REPO_DIR}")
    if rc != 0:
        die("git clone failed")
else:
    run(f"git -C {REPO_DIR} fetch origin {BRANCH}")
    run(f"git -C {REPO_DIR} reset --hard origin/{BRANCH}")

os.chdir(REPO_DIR)
print(f"[INFO] Working dir: {os.getcwd()}")

# ── 2. Install deps ───────────────────────────────────────────────────────────
rc = run("pip install -q transformers peft accelerate sympy datasets")
if rc != 0:
    die("pip install failed")

# Colab ships torchao 0.10.0 which new peft versions refuse to work with
# (dispatch_torchao raises ImportError inside get_peft_model). Nothing in
# our stack uses torchao, so remove it and let peft use the default LoRA path.
run("pip uninstall -y torchao 2>/dev/null || true")

# ── 3. Hardware check ─────────────────────────────────────────────────────────
run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'no nvidia-smi'")

# ── 4. Cold-start SFT ─────────────────────────────────────────────────────────
SFT_OUT = os.path.join(OUT_DIR, "sft")
os.makedirs(SFT_OUT, exist_ok=True)

print("\n" + "=" * 60)
print("  PHASE 1: Cold-start SFT (1 epoch, ~2 min on T4)")
print("=" * 60)

rc = run(
    f"python3 -u benchmarks/train_math_sft.py "
    f"--model_name_or_path Qwen/Qwen2.5-Math-1.5B "
    f"--data_path data/chalk_seeds_500.jsonl "
    f"--output_dir {SFT_OUT} "
    f"--epochs 1 "
    f"--max_length 1024 "
    f"--batch_size 2 "
    f"--grad_accum 8 "
    f"--lora_r 16 "
    f"--lora_alpha 32"
)
if rc != 0:
    print("[WARN] SFT failed — proceeding to RL from base model (no SFT adapter)")
    SFT_ADAPTER = ""
else:
    adapter_path = os.path.join(SFT_OUT, "lora_adapter")
    SFT_ADAPTER = f"--sft_adapter_path {adapter_path}" if os.path.exists(adapter_path) else ""

# ── 5. RL Post-training ───────────────────────────────────────────────────────
RL_OUT = os.path.join(OUT_DIR, "rl")
os.makedirs(RL_OUT, exist_ok=True)

print("\n" + "=" * 60)
print("  PHASE 2: RL Post-training (200 steps on competition pool)")
print("=" * 60)

rc = run(
    f"python3 -u benchmarks/train_baby_chalk_rl.py "
    f"--model_name_or_path Qwen/Qwen2.5-Math-1.5B "
    f"{SFT_ADAPTER} "
    f"--data_path data/rl_pool.jsonl "
    f"--output_dir {RL_OUT} "
    f"--max_steps 200 "
    f"--batch_size 2 "
    f"--num_generations 4 "
    f"--learning_rate 2e-5 "
    f"--max_prompt_len 512 "
    f"--max_completion_len 1024 "
    f"--save_steps 10"
)

if rc != 0:
    die(f"RL training failed with exit code {rc}")

print("\n[DONE] Training complete.")
print(f"[DONE] Checkpoints at: {RL_OUT}")
print("[DONE] Download them with: colab download -s <session> /content/chalk_rl_out ./results/chalk_rl_out")
