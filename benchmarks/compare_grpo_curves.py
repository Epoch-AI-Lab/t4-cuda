#!/usr/bin/env python3
"""Compare GRPO baseline vs INT4-accelerated rollout metrics."""

import json
import os
import sys

def load_metrics(path):
    if not os.path.exists(path):
        print(f"Error: {path} does not exist.")
        return None
    with open(path, "r") as f:
        return json.load(f)

def main():
    base_path = "results/grpo_baseline/metrics.json"
    int4_path = "results/grpo_int4/metrics.json"

    base = load_metrics(base_path)
    int4 = load_metrics(int4_path)

    if base is None or int4 is None:
        sys.exit(1)

    print("===============================================================")
    print("       GRPO Training Comparison: FP16 Baseline vs INT4 Rollouts")
    print("===============================================================")

    base_time = base.get("wall_seconds", 0)
    int4_time = int4.get("wall_seconds", 0)
    speedup = (base_time / int4_time) if int4_time > 0 else 0

    base_tok_s = base.get("approx_tokens_per_sec", 0)
    int4_tok_s = int4.get("approx_tokens_per_sec", 0)

    print(f"Wall time:        FP16: {base_time:.1f}s | INT4: {int4_time:.1f}s (Speedup: {speedup:.2f}x)")
    print(f"Tokens/sec:       FP16: {base_tok_s:.1f} | INT4: {int4_tok_s:.1f}")

    if "peak_vram_gb" in int4:
        print(f"Peak VRAM:        INT4: {int4['peak_vram_gb']} GB")

    base_rewards = [h["reward"] for h in base.get("reward_history", []) if h.get("reward") is not None]
    int4_rewards = [h["reward"] for h in int4.get("reward_history", []) if h.get("reward") is not None]

    if base_rewards and int4_rewards:
        min_len = min(len(base_rewards), len(int4_rewards))
        avg_base = sum(base_rewards[:min_len]) / min_len
        avg_int4 = sum(int4_rewards[:min_len]) / min_len
        print(f"\nMean Reward (first {min_len} steps):")
        print(f"  FP16 Baseline: {avg_base:.4f}")
        print(f"  INT4 Rollouts: {avg_int4:.4f}")
        diff = avg_int4 - avg_base
        print(f"  Reward Delta:  {diff:+.4f} ({'PASS' if abs(diff) < 0.1 else 'DRIFT'})")

if __name__ == "__main__":
    main()
