#!/usr/bin/env python3
"""
test_paper_and_data_integrity.py — Pure Python verification suite for Paper Provenance & Dataset Schemas.

Runs in any Python 3 environment with zero external dependencies (no GPU, no PyTorch, no pytest required).
Verifies:
1. Paper manuscript completeness, 100% BibTeX citation resolution, and empirical metric provenance.
2. SFT reasoning training dataset schema and 5-tag coverage (chalk_seeds_500.jsonl).
3. External competition benchmark integrity and 0% train/test contamination (external_math_eval.json).
4. Honesty datasets structure and balance (honesty_train, honesty_val, quarantined_eval).
5. Raw benchmark JSON artifacts integrity.
"""

import json
import os
import re
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from paper.compile_paper import validate_structure, validate_citations, validate_metrics


class TestPaperAndDataIntegrity(unittest.TestCase):
    """Integrity suite for Paper, Citations, Benchmarks, and Datasets."""

    def setUp(self):
        self.paper_dir = REPO_ROOT / "paper"
        self.tex_file = self.paper_dir / "t4_cuda_paper.tex"
        self.bib_file = self.paper_dir / "references.bib"
        self.data_dir = REPO_ROOT / "data"
        self.benchmarks_dir = REPO_ROOT / "results" / "benchmarks"

    def test_01_paper_structure(self):
        """Verify presence of all 7 sections, theorems, lemmas, tables, and figures."""
        self.assertTrue(self.tex_file.exists(), f"Missing {self.tex_file}")
        tex_content = self.tex_file.read_text(encoding="utf-8")
        self.assertTrue(validate_structure(tex_content), "Paper document structure validation failed")

    def test_02_paper_citations(self):
        """Verify 100% of LaTeX citations resolve in references.bib."""
        self.assertTrue(self.bib_file.exists(), f"Missing {self.bib_file}")
        tex_content = self.tex_file.read_text(encoding="utf-8")
        bib_content = self.bib_file.read_text(encoding="utf-8")
        self.assertTrue(validate_citations(tex_content, bib_content), "BibTeX citation resolution failed")

    def test_03_paper_empirical_metrics_provenance(self):
        """Verify all paper numbers match raw benchmark JSONs and verified logs on disk."""
        tex_content = self.tex_file.read_text(encoding="utf-8")
        self.assertTrue(validate_metrics(tex_content), "Empirical metric provenance check failed")

    def test_04_sft_dataset_integrity(self):
        """Verify chalk_seeds_500.jsonl format, schema, and 5-tag presence."""
        sft_path = self.data_dir / "chalk_seeds_500.jsonl"
        self.assertTrue(sft_path.exists(), f"Missing {sft_path}")

        records = []
        with open(sft_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    self.fail(f"Invalid JSON at line {line_idx}: {e}")
                self.assertIn("problem", rec, f"Missing 'problem' at line {line_idx}")
                solution_text = rec.get("trace") or rec.get("solution")
                self.assertIsNotNone(solution_text, f"Missing 'trace'/'solution' at line {line_idx}")
                records.append(rec)

        self.assertEqual(len(records), 605, f"Expected 605 seeds, found {len(records)}")

        # Verify 5-tag coverage
        tags = ["<explore>", "<conjecture>", "<test_edge_cases>", "<lemma_isolate>", "<formal_proof>"]
        for tag in tags:
            covered = sum(1 for r in records if tag in (r.get("trace") or r.get("solution", "")))
            self.assertGreater(covered, 0, f"Tag {tag} has 0 occurrences in dataset")

    def test_05_external_eval_zero_contamination(self):
        """Verify external_math_eval.json integrity and 0% train contamination."""
        eval_path = self.data_dir / "external_math_eval.json"
        sft_path = self.data_dir / "chalk_seeds_500.jsonl"
        self.assertTrue(eval_path.exists(), f"Missing {eval_path}")

        with open(eval_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)

        contest = eval_data.get("contest_problems", [])
        degrad = eval_data.get("degradation_checks", [])

        self.assertEqual(len(contest), 16, f"Expected 16 contest problems, got {len(contest)}")
        self.assertEqual(len(degrad), 10, f"Expected 10 degradation checks, got {len(degrad)}")

        # Collect train problems
        train_problems = set()
        with open(sft_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                train_problems.add(rec["problem"].strip().lower())

        # Check 0 overlap
        for cp in contest:
            prob_norm = cp["problem"].strip().lower()
            self.assertNotIn(prob_norm, train_problems, f"Contamination detected! Problem '{cp['id']}' in train set.")
            self.assertTrue(cp.get("ground_truth"), f"Missing ground truth for {cp['id']}")
            self.assertTrue(cp.get("boxed_answer"), f"Missing boxed answer for {cp['id']}")

        for dp in degrad:
            self.assertTrue(dp.get("ground_truth"), f"Missing ground truth for {dp['id']}")

    def test_06_honesty_datasets_structure(self):
        """Verify honesty training, validation, and quarantined evaluation sets."""
        for filename, expected_len in [
            ("honesty_train.json", 600),
            ("honesty_val.json", 60),
            ("quarantined_eval.json", 80),
        ]:
            filepath = self.data_dir / filename
            self.assertTrue(filepath.exists(), f"Missing {filepath}")
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(len(data), expected_len, f"Expected {expected_len} in {filename}, got {len(data)}")
            for item in data:
                self.assertIn("question", item)
                self.assertIn("type", item)
                self.assertIn(item["type"], {"answerable", "unanswerable"})

    def test_07_raw_benchmark_artifacts_exist(self):
        """Verify key benchmark JSONs are valid and non-empty."""
        expected_jsons = [
            "t4_speculative_benchmark_report.json",
            "external_eval_results.json",
            "base_eval_results.json",
            "fused_3b_benchmark.json",
            "speculative_decoding_benchmark.json",
            "m1_kv_cache_benchmark.json",
        ]
        for name in expected_jsons:
            p = self.benchmarks_dir / name
            self.assertTrue(p.exists(), f"Missing benchmark JSON: {p}")
            data = json.loads(p.read_text(encoding="utf-8"))
            self.assertIsInstance(data, dict, f"Benchmark {name} should be a JSON object")


if __name__ == "__main__":
    unittest.main(verbosity=2)
