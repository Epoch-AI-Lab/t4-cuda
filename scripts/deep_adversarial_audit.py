#!/usr/bin/env python3
"""Deep Adversarial Audit of Test-Train Contamination.

Performs exhaustive mathematical verification across:
1. Exact String Match (normalized)
2. N-Gram Jaccard & Containment (3-gram, 5-gram, 8-gram)
3. Mathematical Skeleton / Parameter Overlap
4. Ground Truth & Solution Semantic Collision
5. Complete 26-problem inventory (16 contest + 10 degradation)
"""

import json
import os
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

import pyarrow.parquet as pq

BENCH_FILE = "data/external_math_eval.json"
CHALK_FILE = "data/chalk_seeds_500.jsonl"
COMP_FILE = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--qwedsacf--competition_math/snapshots/e839825f9ec5c6cfa585c654a59610969ec13993/data/train-00000-of-00001-7320a6f3aba8ebd2.parquet"
)


def norm_str(s: str) -> str:
    s = s.lower()
    return re.sub(r"[\s\$\\\{\}\(\)\_\^\,\.\?\!\:\;\-\+\*\/\[\]\=\'\"]+", "", s)


def words(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def make_ngrams(seq: List[str], n: int) -> Set[Tuple[str, ...]]:
    if len(seq) < n:
        return set()
    return {tuple(seq[i : i + n]) for i in range(len(seq) - n + 1)}


def math_skeleton(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b\d+(\.\d+)?\b", " <num> ", s)
    tokens = re.findall(r"[a-z]+|<num>", s)
    return " ".join(tokens)


def build_inverted_index(doc_tokens: List[List[str]], n: int):
    index = defaultdict(list)
    doc_ngrams = []
    for doc_id, toks in enumerate(doc_tokens):
        ngs = make_ngrams(toks, n)
        doc_ngrams.append(ngs)
        for ng in ngs:
            index[ng].append(doc_id)
    return index, doc_ngrams


def query_ngrams(test_ngrams: Set[Tuple[str, ...]], index: Dict[Any, List[int]], doc_ngrams: List[Set[Any]]):
    if not test_ngrams:
        return 0.0, 0.0, -1
    candidate_counts = defaultdict(int)
    for ng in test_ngrams:
        if ng in index:
            for doc_id in index[ng]:
                candidate_counts[doc_id] += 1
    if not candidate_counts:
        return 0.0, 0.0, -1
    best_c, best_j, best_doc = 0.0, 0.0, -1
    len_test = len(test_ngrams)
    for doc_id, count in candidate_counts.items():
        c = count / len_test
        j = count / (len_test + len(doc_ngrams[doc_id]) - count)
        if c > best_c:
            best_c = c
            best_doc = doc_id
        if j > best_j:
            best_j = j
    return best_c, best_j, best_doc


def main():
    print("Loading datasets...")
    with open(BENCH_FILE) as f:
        bench = json.load(f)

    with open(CHALK_FILE) as f:
        chalk = [json.loads(line) for line in f]

    comp_table = pq.read_table(COMP_FILE).to_pydict()
    comp_probs = comp_table["problem"]
    comp_sols = comp_table["solution"]

    contest = bench["contest_problems"]
    degrad = bench["degradation_checks"]
    all_tests = contest + degrad

    print(f"  Contest Problems: {len(contest)}")
    print(f"  Degradation Problems: {len(degrad)}")
    print(f"  Chalk Seeds (Training): {len(chalk)}")
    print(f"  Competition Math (Corpus): {len(comp_probs)}")
    print("Building indices...")

    chalk_tokens = [words(r["problem"]) for r in chalk]
    comp_tokens = [words(p) for p in comp_probs]

    chalk_norm_map = {norm_str(r["problem"]): idx for idx, r in enumerate(chalk)}
    comp_norm_map = {norm_str(p): idx for idx, p in enumerate(comp_probs)}

    chalk_skel_map = defaultdict(list)
    for idx, r in enumerate(chalk):
        chalk_skel_map[math_skeleton(r["problem"])].append(idx)

    comp_skel_map = defaultdict(list)
    for idx, p in enumerate(comp_probs):
        comp_skel_map[math_skeleton(p)].append(idx)

    chalk_idx_3, chalk_ng_3 = build_inverted_index(chalk_tokens, 3)
    chalk_idx_5, chalk_ng_5 = build_inverted_index(chalk_tokens, 5)
    chalk_idx_8, chalk_ng_8 = build_inverted_index(chalk_tokens, 8)

    comp_idx_3, comp_ng_3 = build_inverted_index(comp_tokens, 3)
    comp_idx_5, comp_ng_5 = build_inverted_index(comp_tokens, 5)
    comp_idx_8, comp_ng_8 = build_inverted_index(comp_tokens, 8)

    chalk_solutions_norm = [norm_str(r.get("trace", "") + " " + r.get("ground_truth", "")) for r in chalk]

    report = []
    print("Auditing all test problems...")

    for t in all_tests:
        tid = t["id"]
        tprob = t["problem"]
        t_ans = str(t.get("boxed_answer", t.get("ground_truth", ""))).strip().strip("$")
        t_norm = norm_str(tprob)
        t_toks = words(tprob)
        t_skel = math_skeleton(tprob)

        # 1. Exact match
        exact_in_chalk = t_norm in chalk_norm_map
        exact_in_comp = t_norm in comp_norm_map
        exact_in_chalk_sol = any(t_norm in c_sol for c_sol in chalk_solutions_norm if len(t_norm) > 15)

        # 2. Skeleton match
        skel_in_chalk = chalk_skel_map.get(t_skel, [])
        skel_in_comp = comp_skel_map.get(t_skel, [])

        # 3. N-grams
        ng3 = make_ngrams(t_toks, 3)
        ng5 = make_ngrams(t_toks, 5)
        ng8 = make_ngrams(t_toks, 8)

        c3_c, c3_j, c3_idx = query_ngrams(ng3, chalk_idx_3, chalk_ng_3)
        c5_c, c5_j, c5_idx = query_ngrams(ng5, chalk_idx_5, chalk_ng_5)
        c8_c, c8_j, c8_idx = query_ngrams(ng8, chalk_idx_8, chalk_ng_8)

        k3_c, k3_j, k3_idx = query_ngrams(ng3, comp_idx_3, comp_ng_3)
        k5_c, k5_j, k5_idx = query_ngrams(ng5, comp_idx_5, comp_ng_5)
        k8_c, k8_j, k8_idx = query_ngrams(ng8, comp_idx_8, comp_ng_8)

        # 4. Answers
        ans_chalk = [i for i, r in enumerate(chalk) if str(r.get("boxed_answer", "")).strip().strip("$") == t_ans]
        ans_comp = [i for i, s in enumerate(comp_sols) if f"\\boxed{{{t_ans}}}" in s or f"\\boxed{{{t_ans} }}" in s]

        report.append({
            "id": tid,
            "category": "contest" if "sanity" not in tid else "degradation",
            "problem": tprob,
            "answer": t_ans,
            "num_tokens": len(t_toks),
            "exact_chalk": exact_in_chalk,
            "exact_comp": exact_in_comp,
            "exact_trace": exact_in_chalk_sol,
            "skel_chalk": skel_in_chalk,
            "skel_comp": skel_in_comp,
            "ngrams": {
                3: {"chalk_cont": c3_c, "chalk_jacc": c3_j, "chalk_idx": c3_idx, "comp_cont": k3_c, "comp_jacc": k3_j, "comp_idx": k3_idx, "num_ng": len(ng3)},
                5: {"chalk_cont": c5_c, "chalk_jacc": c5_j, "chalk_idx": c5_idx, "comp_cont": k5_c, "comp_jacc": k5_j, "comp_idx": k5_idx, "num_ng": len(ng5)},
                8: {"chalk_cont": c8_c, "chalk_jacc": c8_j, "chalk_idx": c8_idx, "comp_cont": k8_c, "comp_jacc": k8_j, "comp_idx": k8_idx, "num_ng": len(ng8)},
            },
            "ans_chalk_count": len(ans_chalk),
            "ans_comp_count": len(ans_comp),
            "ans_chalk_examples": ans_chalk[:3],
            "ans_comp_examples": ans_comp[:3],
        })

    print("-" * 125)
    print(f"{'ID':<18} | {'Exact Chk/Cmp':<13} | {'Skel Chk/Cmp':<12} | {'8-g Cont (Chk/Cmp)':<18} | {'5-g Cont':<14} | {'3-g Cont':<14} | {'Ans Cnt'}")
    print("-" * 125)

    for r in report:
        c8 = r["ngrams"][8]["chalk_cont"]
        k8 = r["ngrams"][8]["comp_cont"]
        c5 = r["ngrams"][5]["chalk_cont"]
        k5 = r["ngrams"][5]["comp_cont"]
        c3 = r["ngrams"][3]["chalk_cont"]
        k3 = r["ngrams"][3]["comp_cont"]

        sk_c = len(r["skel_chalk"])
        sk_k = len(r["skel_comp"])
        ex = f"{'Y' if r['exact_chalk'] else 'N'}/{'Y' if r['exact_comp'] else 'N'}"
        sk = f"{sk_c}/{sk_k}"
        ans_cnt = f"{r['ans_chalk_count']}/{r['ans_comp_count']}"

        print(f"{r['id']:<18} | {ex:<13} | {sk:<12} | {c8:.2f} / {k8:.2f}{'':<9} | {c5:.2f} / {k5:.2f}{'':<5} | {c3:.2f} / {k3:.2f}{'':<5} | {ans_cnt}")

    with open("data/adversarial_audit_results.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nSaved report to data/adversarial_audit_results.json")


if __name__ == "__main__":
    main()
