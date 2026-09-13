# Data Directory & Dataset Manifest

This directory contains the training, fine-tuning, and quarantined evaluation datasets used across the T4-CUDA research program, alongside their deterministic construction scripts.

---

## 1. Datasets Manifest

| Dataset File | Records | Tokens | Purpose | Provenance & Quality |
|---|---|---|---|---|
| [`chalk_seeds_500.jsonl`](chalk_seeds_500.jsonl) | 605 | ~445k | Cold-start SFT training on `Qwen2.5-Math-1.5B`. | Certified zero-slop, audited reasoning traces containing explicit `<explore>`, `<conjecture>`, `<test_edge_cases>`, `<lemma_isolate>`, `<formal_proof>` tags. |
| [`chalk_seeds_2k.jsonl`](chalk_seeds_2k.jsonl) | 2,000 | ~1.5M | Scaled synthetic reasoning corpus for Phase 2 scaling. | Generated via structured multi-agent proof distillation. |
| [`chalk_seeds_bootstrap.jsonl`](chalk_seeds_bootstrap.jsonl) | 120 | ~85k | Seed bootstrap corpus for initial reasoning schema validation. | Audited seed prompts and formal solutions. |
| [`external_math_eval.json`](external_math_eval.json) | 26 | ~35k | Out-of-dataset evaluation suite (16 AIME/AMC 12 + 10 grounding checks). | 0% overlap with `qwedsacf/competition_math`; includes AIME 2023/2024 and AMC 12 2023 problems with canonical SymPy solutions. |
| [`honesty_train.json`](honesty_train.json) | 300 | ~45k | Training dataset for GRPO honesty alignment (Milestone 3). | Balanced 50/50 mix: 150 answerable factual/math questions and 150 impossible premise traps (e.g. "President of the Atlantic Ocean in 1850"). |
| [`honesty_val.json`](honesty_val.json) | 60 | ~10k | Validation split for honesty alignment monitoring. | Non-overlapping split with identical 50/50 balance. |
| [`quarantined_eval.json`](quarantined_eval.json) | 150 | ~25k | Quarantined zero-contamination evaluation set. | Strictly held-out impossible traps and factual queries to measure honest abstention without training contamination. |
| [`adversarial_audit_results.json`](adversarial_audit_results.json) | — | — | Full adversarial contamination and quality audit log. | Verifies n-gram and semantic decontamination across train/eval boundaries. |

---

## 2. Dataset Construction Scripts

| Script | Outputs | Description |
|---|---|---|
| [`build_external_math_benchmark.py`](build_external_math_benchmark.py) | `external_math_eval.json` | Extracts held-out AIME and AMC 12 problems, pairs them with ground-truth boxed answers, and appends the 10-problem grounding sanity suite. |
| [`build_honesty_dataset.py`](build_honesty_dataset.py) | `honesty_train.json`, `honesty_val.json` | Deterministically compiles the balanced honesty dataset with explicit abstention targets for impossible premises. |
| [`build_quarantined_test_set.py`](build_quarantined_test_set.py) | `quarantined_eval.json` | Compiles the 150-question quarantined benchmark for measuring hallucination vs honest abstention. |
