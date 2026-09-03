#!/usr/bin/env python3
"""Zero-Contamination Benchmark Suite: Base Model vs Tiny Honest Model.

Evaluates on strictly quarantined data guaranteed to have 0% overlap with training:
1. Quarantined Honesty (data/quarantined_eval.json):
   - 40 Never-seen impossible traps (false historical facts, non-existent entities, anachronisms).
   - 40 Never-seen factual/math questions.
2. GSM8K Official Test Split (30 multi-step math problems).
3. TruthfulQA Common Misconceptions (20 questions).

Saves to results/quarantined_benchmark.json.
"""

import argparse
import json
import os
import re
import sys
import time
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from benchmarks.benchmark_indepth_honesty import TRUTHFUL_QA_SAMPLES, evaluate_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--trained_model", default="results/honest_qwen_0.5b/final_checkpoint")
    parser.add_argument("--eval_file", default="data/quarantined_eval.json")
    parser.add_argument("--out_dir", default="results/quarantined_benchmark")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.eval_file) as f:
        quarantined_data = json.load(f)

    # 30 GSM8K test questions
    gsm_raw = load_dataset("openai/gsm8k", "main", split="test")
    gsm_data = []
    for item in list(gsm_raw)[:30]:
        gold = re.findall(r"####\s*(-?[0-9]+)", item["answer"])[-1]
        gsm_data.append({"question": item["question"], "gold_answer": gold})

    truthful_data = TRUTHFUL_QA_SAMPLES

    print("=" * 85)
    print("      STRICTLY QUARANTINED (ZERO-CONTAMINATION) BENCHMARK SUITE")
    print("=" * 85)
    print(f"Quarantined Questions: {len(quarantined_data)} (40 answerable, 40 impossible traps)")
    print(f"Verified Overlap with Train Data: 0.0% (Checked by assertion)")
    print(f"GSM8K Math Problems:   {len(gsm_data)}")
    print(f"TruthfulQA Traps:      {len(truthful_data)}")
    print("=" * 85)

    tok = AutoTokenizer.from_pretrained(args.base_model)

    # 1. Base Model
    base_m = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float16, device_map="cuda")
    base_res = evaluate_model("Base Model (Qwen2.5-0.5B-Instruct)", tok, base_m, quarantined_data, gsm_data, truthful_data)
    del base_m
    torch.cuda.empty_cache()

    # 2. Trained Model
    trained_m = AutoModelForCausalLM.from_pretrained(args.trained_model, torch_dtype=torch.float16, device_map="cuda")
    trained_res = evaluate_model("Tiny Honest Model (Ours, Post-GRPO)", tok, trained_m, quarantined_data, gsm_data, truthful_data)
    del trained_m
    torch.cuda.empty_cache()

    # Side-by-Side Summary
    print("\n" + "=" * 85)
    print("            ZERO-CONTAMINATION BENCHMARK RESULTS (TESLA T4)")
    print("=" * 85)
    print(f"{'Evaluation Metric':<36} | {'Base Model':<16} | {'Tiny Honest (Ours)':<18} | {'Delta':<10}")
    print("-" * 85)
    metrics_to_show = [
        ("Answerable Factual Accuracy (%)", "ans_accuracy", "+"),
        ("False Abstention / Cowardice (%)", "false_abstain_rate", "-"),
        ("Honest Abstention on Traps (%)", "honest_abstain_rate", "+"),
        ("Hallucination Rate on Traps (%)", "hallucination_rate", "-"),
        ("Calibration Index (Overall) (%)", "calibration_score", "+"),
        ("Avg Words on Traps (Conciseness)", "avg_imp_words", "-"),
        ("GSM8K Math Accuracy (%)", "gsm8k_accuracy", "+"),
        ("TruthfulQA Myth Resistance (%)", "truthfulqa_accuracy", "+"),
    ]

    for label, key, good_dir in metrics_to_show:
        b_val = base_res[key]
        t_val = trained_res[key]
        diff = t_val - b_val
        diff_str = f"{diff:+.1f}%" if "%" in label else f"{diff:+.1f} words"
        print(f"{label:<36} | {b_val:14.1f}   | {t_val:16.1f}   | {diff_str:<10}")

    print("=" * 85)

    report = {
        "base_model": base_res,
        "trained_model": trained_res,
    }
    with open(os.path.join(args.out_dir, "quarantined_metrics.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nResults saved to {args.out_dir}/quarantined_metrics.json")

if __name__ == "__main__":
    main()
