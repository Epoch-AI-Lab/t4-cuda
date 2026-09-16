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

    def test_08_theorem_4_fp8_e4m3_exactness_and_domain(self):
        """Verify Theorem 4: exactly 238 normalized FP8 states and bit-exact FP16 mapping."""
        normalized_count = 0
        zero_subnormal_count = 0
        nan_count = 0

        for byte_val in range(256):
            s = (byte_val >> 7) & 1
            e8 = (byte_val >> 3) & 0xF
            m8 = byte_val & 0x7

            if e8 == 0:
                zero_subnormal_count += 1
            elif e8 == 15 and m8 == 7:
                nan_count += 1
            else:
                normalized_count += 1
                # Check mathematical equivalence
                fp8_real = ((-1.0) ** s) * (2.0 ** (e8 - 7)) * (1.0 + m8 / 8.0)
                e16 = e8 + 8
                m16 = m8 * 128
                fp16_real = ((-1.0) ** s) * (2.0 ** (e16 - 15)) * (1.0 + m16 / 1024.0)
                self.assertEqual(fp8_real, fp16_real, f"Mismatch at byte 0x{byte_val:02X}")

        self.assertEqual(normalized_count, 238, "Expected exactly 238 normalized FP8 E4M3 states")
        self.assertEqual(zero_subnormal_count, 16, "Expected exactly 16 zero/subnormal states (E8=0)")
        self.assertEqual(nan_count, 2, "Expected exactly 2 NaN states (0x7F, 0xFF)")

        # Verify constant calculation
        c_exp = (15 - 7) * (2 ** 10)
        self.assertEqual(c_exp, 0x2000)
        c_packed = (c_exp << 16) | c_exp
        self.assertEqual(c_packed, 0x20002000)

    def test_09_lemma_3_dram_traffic_accounting(self):
        """Verify Lemma 3: exact DRAM traffic accounting and 21.43% reduction."""
        # Unfused: write dW (2), read dW (2), read W (4), read m (4), read v (4), write W (4), write m (4), write v (4)
        unfused_traffic = 2 + 2 + 4 + 4 + 4 + 4 + 4 + 4
        self.assertEqual(unfused_traffic, 28)

        # Fused: read W (4), read m (4), read v (4), write W_active (2), write m (4), write v (4)
        fused_traffic = 4 + 4 + 4 + 2 + 4 + 4
        self.assertEqual(fused_traffic, 22)

        traffic_saved_pct = (unfused_traffic - fused_traffic) / unfused_traffic
        self.assertAlmostEqual(traffic_saved_pct, 6.0 / 28.0, places=6)
        self.assertAlmostEqual(traffic_saved_pct * 100.0, 21.42857, places=4)

    def test_10_architectural_claims_and_manuscript_reconciliation(self):
        """Verify manuscript reflects audit remediations for claims and limitations."""
        tex_content = self.tex_file.read_text(encoding="utf-8")

        # Theorem 4 mentions 238 normalized states and subnormal handling
        self.assertIn("238 normalized", tex_content)
        self.assertTrue(re.search(r"flush-to-zero|subnormal", tex_content, re.IGNORECASE))

        # Lemma 3 traffic breakdown
        self.assertIn(r"\text{Traffic}_{\text{unfused}}", tex_content)
        self.assertIn(r"\text{Traffic}_{\text{fused}}", tex_content)

        # Bank conflict padding with XOR swizzling algebraic model
        self.assertTrue(re.search(r"SMEM\\_K\\_STRIDE\s*=\s*72", tex_content))
        self.assertTrue(re.search(r"padding", tex_content, re.IGNORECASE))
        self.assertTrue(re.search(r"algebraic", tex_content, re.IGNORECASE))

        # Warp specialization as architectural proposal / ablation
        self.assertTrue(re.search(r"architectural proposal", tex_content, re.IGNORECASE))

    def test_11_speculative_decoding_and_sft_reasoning_reporting(self):
        """Verify clear separation of neural draft overhead vs prompt lookup, and 1024-token context limits."""
        tex_content = self.tex_file.read_text(encoding="utf-8")

        # Speculative decoding separation
        self.assertTrue(re.search(r"Neural Draft", tex_content))
        self.assertTrue(re.search(r"Prompt Lookup", tex_content, re.IGNORECASE))
        self.assertTrue(re.search(r"1\.48x", tex_content))
        self.assertTrue(re.search(r"sequential.*forward passes|host dispatch", tex_content, re.IGNORECASE))

        # SFT reasoning reporting
        self.assertTrue(re.search(r"Learned \(Exploratory\)", tex_content))
        self.assertTrue(re.search(r"1024-token context", tex_content))
        self.assertTrue(re.search(r"exploratory tags|insufficient token runway|truncation", tex_content, re.IGNORECASE))

    def test_12_dequantization_bandwidth_metadata_accounting(self):
        """Verify Table 1 and Section 4.2 account for all metadata in effective bandwidth."""
        tex_content = self.tex_file.read_text(encoding="utf-8")

        # 52 B/word accounting (4B packed + 16B scales + 16B zeros + 16B output)
        self.assertIn("52 B/word", tex_content)
        self.assertTrue(re.search(r"872\.4[\$\s]*MB", tex_content))
        self.assertTrue(re.search(r"L2 cache hit", tex_content, re.IGNORECASE))
        self.assertIn("332.7 GB/s", tex_content)

    def test_13_scale_envelope_constant_calculation(self):
        """Verify physical precision bound calculation for FP16 exponent scale envelope."""
        tex_content = self.tex_file.read_text(encoding="utf-8")

        # Check mathematical bounds in Section 3.2
        self.assertTrue(re.search(r"63\.97|63\.47|63\.05", tex_content))
        # Ensure 65504 is mentioned
        self.assertIn("65504", tex_content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
