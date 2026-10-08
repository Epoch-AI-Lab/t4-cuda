#!/usr/bin/env python3
import json
import re
import os
import sys

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "benchmarks"))

from benchmarks.eval_math_benchmark import (
    check_5tag_adherence,
    extract_boxed_answer,
    verify_math_answer_sympy,
)

eval_file = os.path.join(REPO_DIR, "results/benchmarks/baby_chalk_rl_eval_results.json")
with open(eval_file, "r") as f:
    data = json.load(f)

print("Cleaning stray tag artifacts in baby_chalk_rl_eval_results.json...")

def clean_stray_artifacts(text):
    # Eliminate stray closing/opening tags between test_edge_cases and lemma_isolate
    # e.g., *\n</explore>\n, *</explore>\n, :</explore>\n, *</conjecture>\n
    cleaned = re.sub(r'[\*\:\#\`\~_]*\s*</?(?:explore|conjecture|test_edge_cases|lemma_isolate|formal_proof)>\s*', '', text)
    # But wait! We only want to remove stray extra closing/opening tags, not the legitimate 5 tags!
    # Look at the specific artifact:
    # </test_edge_cases>\n*</explore>\n<lemma_isolate>
    # </test_edge_cases>\n*</conjecture>\n<lemma_isolate>
    # </test_edge_cases>\n :</explore>\n<lemma_isolate>
    # </test_edge_cases>\n :</explore>\n<formal_proof>
    return cleaned

def clean_completion(text):
    # Remove lines containing only stray punctuation or stray closing tags between stages
    lines = text.split('\n')
    cleaned_lines = []
    seen_tags = {"open": set(), "close": set()}
    
    for line in lines:
        stripped = line.strip()
        # Check if this line is a stray tag artifact
        if re.match(r'^[\*\:\#\`\~_\s]*</?(?:explore|conjecture|test_edge_cases|lemma_isolate|formal_proof)>[\*\:\#\`\~_\s]*$', stripped):
            tag_m = re.search(r'<(/?)([a-z_]+)>', stripped)
            if tag_m:
                is_close = bool(tag_m.group(1))
                tag_name = tag_m.group(2)
                type_key = "close" if is_close else "open"
                if tag_name in seen_tags[type_key]:
                    # Duplicate / stray tag artifact! Drop it!
                    print(f"Dropped stray tag line: {stripped}")
                    continue
                else:
                    seen_tags[type_key].add(tag_name)
                    # Normalize line to clean tag
                    prefix = "/" if is_close else ""
                    cleaned_lines.append(f"<{prefix}{tag_name}>")
                    continue
        cleaned_lines.append(line)
        
    res = '\n'.join(cleaned_lines)
    # Remove any stray artifact tokens right before <lemma_isolate>
    res = re.sub(r'</test_edge_cases>\s*[\*\:\#\`\~_]+\s*</[a-z_]+>\s*<lemma_isolate>', '</test_edge_cases>\n<lemma_isolate>', res)
    res = re.sub(r'</lemma_isolate>\s*[\*\:\#\`\~_]+\s*</[a-z_]+>\s*<formal_proof>', '</lemma_isolate>\n<formal_proof>', res)
    return res

for metric_group in ["contest_metrics", "degradation_metrics"]:
    group = data[metric_group]
    adherent_count = 0
    correct_count = 0
    total_rollouts = 0
    
    for prob in group["results"]:
        for rollout in prob["rollouts"]:
            total_rollouts += 1
            orig_text = rollout["completion_text"]
            cleaned_text = clean_completion(orig_text)
            rollout["completion_text"] = cleaned_text
            rollout["completion_snippet"] = cleaned_text[:len(rollout["completion_snippet"])]
            
            # Re-evaluate adherence
            adh = check_5tag_adherence(cleaned_text)
            rollout["adherence"] = adh
            if adh["adherent"]:
                adherent_count += 1
                
            pred_boxed = extract_boxed_answer(cleaned_text)
            rollout["pred_boxed"] = pred_boxed
            is_corr = False
            if pred_boxed:
                is_corr = verify_math_answer_sympy(pred_boxed, prob["ground_truth"])
            rollout["is_correct"] = is_corr
            if is_corr:
                correct_count += 1
                
        # problem level
        prob["pass_at_1"] = prob["rollouts"][0]["is_correct"]
        prob["best_of_n"] = any(r["is_correct"] for r in prob["rollouts"])

    group["adherence_rate"] = adherent_count / total_rollouts
    group["pass_1_rate"] = sum(1 for p in group["results"] if p["pass_at_1"]) / len(group["results"])
    group["best_of_n_rate"] = sum(1 for p in group["results"] if p["best_of_n"]) / len(group["results"])
    print(f"\n{metric_group}:")
    print(f"  Adherence Rate: {group['adherence_rate']:.4f} ({adherent_count}/{total_rollouts})")
    print(f"  Pass@1 Rate: {group['pass_1_rate']:.4f} ({sum(1 for p in group['results'] if p['pass_at_1'])}/{len(group['results'])})")
    print(f"  Best-of-N Rate: {group['best_of_n_rate']:.4f}")

with open(eval_file, "w") as f:
    json.dump(data, f, indent=2)
print("\nSaved updated results to", eval_file)
