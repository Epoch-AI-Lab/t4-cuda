#!/usr/bin/env python3
"""Adversarial Pipeline & Code Auditor.

Audits 'benchmarks/train_math_sft.py' and 'benchmarks/eval_math_benchmark.py' against:
1. Token boundary & loss masking fuzzing (subword merging, truncation edge cases).
2. Collation & right-padding integrity (attention mask leaks, label -100 verification).
3. LoRA gradient flow verification (parameter coverage, step 1 zero-init vs step 2 flow).
4. Eval metric exploit testing (arbitrary code execution in sympify, regex false positives, tag loopholes).
"""

import json
import os
import re
import sys
from typing import Dict, List, Any

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "src"))

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import sympy

from benchmarks.train_math_sft import SYSTEM_PROMPT, ChalkMathSFTDataset, collate_fn
from benchmarks.eval_math_benchmark import (
    check_5tag_adherence,
    extract_boxed_answer,
    clean_latex_math,
    verify_math_answer_sympy,
)

MODEL_SNAPSHOT = "/home/kriday/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2"
DATA_PATH = os.path.join(REPO_DIR, "data/chalk_seeds_500.jsonl")


class PipelineAuditor:
    def __init__(self):
        self.tok_path = MODEL_SNAPSHOT if os.path.exists(MODEL_SNAPSHOT) else "Qwen/Qwen2.5-Math-1.5B"
        self.tokenizer = AutoTokenizer.from_pretrained(self.tok_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.results = {}

    def audit_token_boundary_and_masking(self) -> Dict[str, Any]:
        """Test 1: Fuzz prompt masking logic in ChalkMathSFTDataset."""
        print("\n" + "=" * 60)
        print("TEST 1: Token Boundary & Loss Masking Fuzzing")
        print("=" * 60)

        import tempfile

        # 1.1 Test on real records via ChalkMathSFTDataset
        ds = ChalkMathSFTDataset(DATA_PATH, self.tokenizer, max_length=1024)
        exact_prefix_matches = 0
        prompt_masking_checks = 0
        num_to_test = min(50, len(ds))

        for idx in range(num_to_test):
            item = ds[idx]
            input_ids = item["input_ids"]
            labels = item["labels"]

            masked_count = (labels == -100).sum().item()
            unmasked_count = (labels != -100).sum().item()

            if masked_count > 0 and unmasked_count > 0:
                prompt_masking_checks += 1

            first_unmasked = (labels != -100).nonzero(as_tuple=True)[0][0].item()
            first_tok_str = self.tokenizer.decode([input_ids[first_unmasked]])
            if "<" in first_tok_str or "explore" in first_tok_str:
                exact_prefix_matches += 1

        print(f"Dataset 50 records prefix matches: {exact_prefix_matches}/{num_to_test}")
        print(f"Dataset 50 records prompt masking checks: {prompt_masking_checks}/{num_to_test}")

        # 1.2 Boundary Subword Merging Fuzzing on ChalkMathSFTDataset
        fuzz_completions = [
            "\n",
            "\n\n",
            " ",
            "  ",
            "\t",
            "\n<explore>",
            " <explore>",
            "\r\n",
            " \n ",
            "Let's solve",
            "<explore>",
            "123",
            "\\frac{1}{2}",
        ]

        merge_failures = []
        with tempfile.NamedTemporaryFile("w+", suffix=".jsonl") as f:
            for comp in fuzz_completions:
                f.write(json.dumps({"problem": "Find x.", "trace": comp}) + "\n")
            f.flush()

            ds_fuzz = ChalkMathSFTDataset(f.name, self.tokenizer, max_length=256)
            for idx, comp in enumerate(fuzz_completions):
                item = ds_fuzz[idx]
                labels = item["labels"]
                input_ids = item["input_ids"]
                unmasked = (labels != -100).sum().item()
                if unmasked == 0:
                    merge_failures.append({
                        "completion": repr(comp),
                        "impact": "All tokens masked; completion lost."
                    })

        print(f"Boundary subword merge failures: {len(merge_failures)}/{len(fuzz_completions)}")
        for mf in merge_failures:
            print(f"  FAILED on {mf['completion']}: boundary fusion observed.")

        # 1.3 Truncation Edge Case: Prompt >= max_length causing all labels = -100
        # Tested on ChalkMathSFTDataset with safe NaN prevention
        with tempfile.NamedTemporaryFile("w+", suffix=".jsonl") as f:
            f.write(json.dumps({"problem": "x " * 2100, "trace": "test"}) + "\n")
            f.flush()

            ds_long = ChalkMathSFTDataset(f.name, self.tokenizer, max_length=128)
            long_item = ds_long[0]
            lbls = long_item["labels"]
            all_masked = (lbls == -100).all().item()

            dummy_logits = torch.randn(1, 128, self.tokenizer.vocab_size)
            loss_val = nn.CrossEntropyLoss(ignore_index=-100)(
                dummy_logits.view(-1, dummy_logits.size(-1)),
                lbls.view(-1)
            ).item()
            is_loss_nan = torch.isnan(torch.tensor(loss_val)).item()

        print(f"Truncation overflow all labels masked: {all_masked}")
        print(f"Loss when prompt saturates max_length: {loss_val} (is NaN: {is_loss_nan})")

        return {
            "dataset_prefix_matches": exact_prefix_matches,
            "dataset_masking_checks": prompt_masking_checks,
            "subword_merge_vulnerabilities": merge_failures,
            "truncation_nan_loss_vulnerability": is_loss_nan,
        }

    def audit_collation_and_padding(self) -> Dict[str, Any]:
        """Test 2: Test collate_fn with variable length batches and attention invariance."""
        print("\n" + "=" * 60)
        print("TEST 2: Collation & Padding Integrity")
        print("=" * 60)

        # Build mock variable length batch
        batch = [
            {
                "input_ids": torch.tensor([101, 202, 303, 404, 505], dtype=torch.long),
                "attention_mask": torch.tensor([1, 1, 1, 1, 1], dtype=torch.long),
                "labels": torch.tensor([-100, -100, 303, 404, 505], dtype=torch.long)
            },
            {
                "input_ids": torch.tensor([101, 202, 303], dtype=torch.long),
                "attention_mask": torch.tensor([1, 1, 1], dtype=torch.long),
                "labels": torch.tensor([-100, -100, 303], dtype=torch.long)
            },
            {
                "input_ids": torch.tensor([101, 202, 303, 404, 505, 606, 707], dtype=torch.long),
                "attention_mask": torch.tensor([1, 1, 1, 1, 1, 1, 1], dtype=torch.long),
                "labels": torch.tensor([-100, -100, -100, 404, 505, 606, 707], dtype=torch.long)
            }
        ]

        pad_token_id = self.tokenizer.pad_token_id or 0
        collated = collate_fn(batch, pad_token_id)

        input_ids = collated["input_ids"]
        attention_mask = collated["attention_mask"]
        labels = collated["labels"]

        # Check shapes
        shape_correct = (input_ids.shape == (3, 7)) and (attention_mask.shape == (3, 7)) and (labels.shape == (3, 7))

        # Check item 1 padding
        item1_pad_ids = (input_ids[0, 5:] == pad_token_id).all().item()
        item1_pad_mask = (attention_mask[0, 5:] == 0).all().item()
        item1_pad_labels = (labels[0, 5:] == -100).all().item()

        # Check item 2 padding
        item2_pad_ids = (input_ids[1, 3:] == pad_token_id).all().item()
        item2_pad_mask = (attention_mask[1, 3:] == 0).all().item()
        item2_pad_labels = (labels[1, 3:] == -100).all().item()

        padding_strictly_minus_100 = item1_pad_labels and item2_pad_labels
        mask_strictly_zero = item1_pad_mask and item2_pad_mask

        print(f"Shapes matched: {shape_correct}")
        print(f"Padding labels strictly -100: {padding_strictly_minus_100}")
        print(f"Padding attention masks strictly 0: {mask_strictly_zero}")

        # Test causal attention invariance on miniature model
        cfg = AutoConfig.from_pretrained(self.tok_path)
        cfg.num_hidden_layers = 2
        cfg.hidden_size = 64
        cfg.intermediate_size = 128
        cfg.num_attention_heads = 4
        cfg.num_key_value_heads = 2
        model = AutoModelForCausalLM.from_config(cfg)
        model.eval()

        with torch.no_grad():
            # Item 2 unpadded
            item2_unpadded_out = model(
                input_ids=batch[1]["input_ids"].unsqueeze(0),
                attention_mask=batch[1]["attention_mask"].unsqueeze(0)
            )
            # Item 2 padded in batch
            batch_out = model(input_ids=input_ids, attention_mask=attention_mask)

            item2_padded_logits = batch_out.logits[1, :3]
            max_logit_diff = (item2_unpadded_out.logits[0] - item2_padded_logits).abs().max().item()

        print(f"Right-padding causal attention max logit delta: {max_logit_diff:.8e}")
        causal_invariant = max_logit_diff < 1e-5

        return {
            "shape_correct": shape_correct,
            "padding_labels_minus_100": padding_strictly_minus_100,
            "mask_strictly_zero": mask_strictly_zero,
            "max_logit_diff": max_logit_diff,
            "causal_invariant": causal_invariant
        }

    def audit_lora_gradient_flow(self) -> Dict[str, Any]:
        """Test 3: Verify LoRA target coverage, Step 1 zero-init behavior, and Step 2 gradient flow."""
        print("\n" + "=" * 60)
        print("TEST 3: LoRA Gradient Flow Verification")
        print("=" * 60)

        cfg = AutoConfig.from_pretrained(self.tok_path)
        cfg.num_hidden_layers = 2
        cfg.hidden_size = 128
        cfg.intermediate_size = 256
        cfg.num_attention_heads = 4
        cfg.num_key_value_heads = 2
        base_model = AutoModelForCausalLM.from_config(cfg)

        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=target_modules,
            lora_dropout=0.0,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        peft_model = get_peft_model(base_model, lora_config)
        peft_model.train()

        # Check coverage of modules
        target_counts = {mod: 0 for mod in target_modules}
        for name, param in peft_model.named_parameters():
            if param.requires_grad:
                for mod in target_modules:
                    if mod in name:
                        target_counts[mod] += 1

        print("Trainable LoRA parameter module distribution:")
        for mod, count in target_counts.items():
            print(f"  {mod}: {count} parameter tensors")

        # Step 1 forward + backward
        optimizer = torch.optim.AdamW(peft_model.parameters(), lr=1e-3)
        input_ids = torch.randint(10, 500, (2, 32))
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        labels[:, :10] = -100

        optimizer.zero_grad()
        out1 = peft_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        out1.loss.backward()

        step1_none_grads = []
        step1_zero_grads = []
        step1_active_grads = []

        for name, param in peft_model.named_parameters():
            if param.requires_grad:
                if param.grad is None:
                    step1_none_grads.append(name)
                elif (param.grad == 0).all():
                    step1_zero_grads.append(name)
                else:
                    step1_active_grads.append(name)

        print(f"\nStep 1 (First Backward Pass):")
        print(f"  Active gradients: {len(step1_active_grads)}")
        print(f"  None gradients: {len(step1_none_grads)}")
        print(f"  Zero gradients: {len(step1_zero_grads)} (All lora_A parameters due to lora_B=0 initialization)")

        # Optimizer step
        optimizer.step()

        # Step 2 forward + backward
        optimizer.zero_grad()
        out2 = peft_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        out2.loss.backward()

        step2_none_grads = []
        step2_zero_grads = []
        step2_active_grads = []

        for name, param in peft_model.named_parameters():
            if param.requires_grad:
                if param.grad is None:
                    step2_none_grads.append(name)
                elif (param.grad == 0).all():
                    step2_zero_grads.append(name)
                else:
                    step2_active_grads.append(name)

        print(f"\nStep 2 (After First Optimizer Step):")
        print(f"  Active gradients: {len(step2_active_grads)}")
        print(f"  None gradients: {len(step2_none_grads)}")
        print(f"  Zero gradients: {len(step2_zero_grads)}")

        full_flow_established = (len(step2_active_grads) > 0) and (len(step2_none_grads) == 0) and (len(step2_zero_grads) == 0)

        return {
            "target_counts": target_counts,
            "step1_zero_grads_count": len(step1_zero_grads),
            "step1_active_grads_count": len(step1_active_grads),
            "step2_none_grads_count": len(step2_none_grads),
            "step2_zero_grads_count": len(step2_zero_grads),
            "step2_active_grads_count": len(step2_active_grads),
            "full_flow_established": full_flow_established
        }

    def audit_eval_metric_exploits(self) -> Dict[str, Any]:
        """Test 4: Eval metric exploit testing on verify_math_answer_sympy, regex, and 5-tag adherence."""
        print("\n" + "=" * 60)
        print("TEST 4: Eval Metric Exploit Testing")
        print("=" * 60)

        # 4.1 Exploiting verify_math_answer_sympy regex fallback
        false_positive_exploits = [
            {
                "name": "Fraction Denominator Fallback Match",
                "pred": "2",
                "gold": r"\frac{1}{2}",
                "expected": False,
                "reason": "Gold fraction denominator '2' matches incorrect model prediction '2'"
            },
            {
                "name": "Coordinate / Tuple Component Match",
                "pred": "5",
                "gold": "(3, 5)",
                "expected": False,
                "reason": "Gold tuple coordinate '5' matches model prediction '5'"
            },
            {
                "name": "Qualifying Text / Negation Matching",
                "pred": "The answer is definitely not 42",
                "gold": "42",
                "expected": False,
                "reason": "Model outputs sentence ending with 42, regex extracts 42 and marks correct"
            },
            {
                "name": "Nested LaTeX Frac Failure Fallback",
                "pred": "3",
                "gold": r"\frac{\sqrt{2}}{3}",
                "expected": False,
                "reason": "Nested braces fail clean_latex_math regex, regex extracts 3 and awards credit"
            },
            {
                "name": "Equation Statement Match",
                "pred": "x = 42",
                "gold": "42",
                "expected": True,  # Common in math, but demonstrates sympy failure + regex bypass
                "reason": "Equation fails sympify, passes through regex fallback"
            }
        ]

        verified_exploits = []
        for test in false_positive_exploits:
            result = verify_math_answer_sympy(test["pred"], test["gold"])
            passed_exploit = (result != test["expected"])
            if passed_exploit:
                verified_exploits.append({
                    "test": test["name"],
                    "pred": test["pred"],
                    "gold": test["gold"],
                    "actual_result": result,
                    "expected_result": test["expected"],
                    "reason": test["reason"]
                })
            print(f"  [{'EXPLOIT CONFIRMED' if passed_exploit else 'OK'}] {test['name']}: pred='{test['pred']}' vs gold='{test['gold']}' -> {result}")

        # 4.2 Arbitrary Code Execution in sympy.sympify
        canary_file = "/tmp/audit_canary_exploit.txt"
        if os.path.exists(canary_file):
            os.remove(canary_file)

        malicious_input = f'__import__("pathlib").Path("{canary_file}").write_text("PWNED")'
        try:
            verify_math_answer_sympy(malicious_input, "42")
        except Exception:
            pass

        rce_vulnerability_confirmed = os.path.exists(canary_file)
        if rce_vulnerability_confirmed:
            os.remove(canary_file)
            print("  [CRITICAL SECURITY FLAW] Arbitrary code execution confirmed via sympy.sympify!")
        else:
            print("  [SECURE] sympy.sympify blocked malicious string execution.")

        # 4.3 5-Tag Adherence Parser Loopholes
        tag_tests = {
            "normal_sequential": (
                "<explore>a</explore><conjecture>b</conjecture><test_edge_cases>c</test_edge_cases>"
                "<lemma_isolate>d</lemma_isolate><formal_proof>e</formal_proof>",
                True
            ),
            "duplicate_tags_loophole": (
                "<explore>first</explore><explore>duplicate</explore><conjecture>b</conjecture>"
                "<test_edge_cases>c</test_edge_cases><lemma_isolate>d</lemma_isolate><formal_proof>e</formal_proof>",
                False
            ),
            "empty_tags_loophole": (
                "<explore></explore><conjecture></conjecture><test_edge_cases></test_edge_cases>"
                "<lemma_isolate></lemma_isolate><formal_proof></formal_proof>",
                False
            ),
            "swapped_order_detection": (
                "<conjecture>b</conjecture><explore>a</explore><test_edge_cases>c</test_edge_cases>"
                "<lemma_isolate>d</lemma_isolate><formal_proof>e</formal_proof>",
                False
            ),
            "unclosed_tag_detection": (
                "<explore>unclosed<conjecture>b</conjecture><test_edge_cases>c</test_edge_cases>"
                "<lemma_isolate>d</lemma_isolate><formal_proof>e</formal_proof>",
                False
            ),
            "nested_tags_detection": (
                "<explore><conjecture>nested</conjecture></explore><test_edge_cases>c</test_edge_cases>"
                "<lemma_isolate>d</lemma_isolate><formal_proof>e</formal_proof>",
                False
            ),
        }

        tag_loopholes = []
        for name, (text, expected_adherent) in tag_tests.items():
            res = check_5tag_adherence(text)
            is_adherent = res["adherent"]
            if is_adherent != expected_adherent:
                tag_loopholes.append({
                    "test": name,
                    "actual": is_adherent,
                    "expected": expected_adherent,
                    "details": res
                })
            status_str = "LOOPHOLE DETECTED" if is_adherent != expected_adherent else "OK"
            print(f"  [{status_str}] {name}: adherent={is_adherent} (expected={expected_adherent})")

        # 4.4 extract_boxed_answer edge cases
        boxed_tests = [
            (r"The answer is \boxed{\frac{1}{2}}.", r"\frac{1}{2}"),
            (r"First attempt \boxed{10}, final is \boxed{42}.", "42"),
            (r"Malformed \boxed{42", None),
            (r"Nested \boxed{\boxed{42}}.", "42"),
            (r"Escaped \x08oxed{42}.", "42"),
            (r"Empty \boxed{}.", ""),
        ]

        boxed_mismatches = []
        for text, expected in boxed_tests:
            extracted = extract_boxed_answer(text)
            if extracted != expected:
                boxed_mismatches.append({"text": text, "actual": extracted, "expected": expected})

        print(f"Boxed extraction edge case failures: {len(boxed_mismatches)}")

        return {
            "verified_exploits": verified_exploits,
            "rce_vulnerability_confirmed": rce_vulnerability_confirmed,
            "tag_loopholes": tag_loopholes,
            "boxed_mismatches": boxed_mismatches
        }

    def run_all(self) -> Dict[str, Any]:
        t1 = self.audit_token_boundary_and_masking()
        t2 = self.audit_collation_and_padding()
        t3 = self.audit_lora_gradient_flow()
        t4 = self.audit_eval_metric_exploits()

        has_critical = (
            t4["rce_vulnerability_confirmed"]
            or len(t4["verified_exploits"]) > 0
            or t1["truncation_nan_loss_vulnerability"]
        )
        verdict = "RED" if has_critical else ("YELLOW" if len(t1["subword_merge_vulnerabilities"]) > 0 else "GREEN")

        report = {
            "verdict": verdict,
            "test_1_boundary_and_masking": t1,
            "test_2_collation_and_padding": t2,
            "test_3_lora_gradient_flow": t3,
            "test_4_eval_metric_exploits": t4,
        }

        print("\n" + "=" * 60)
        print(f"AUDIT VERDICT: {verdict}")
        print("=" * 60)
        return report


if __name__ == "__main__":
    auditor = PipelineAuditor()
    report = auditor.run_all()
    out_path = os.path.join(REPO_DIR, "results/pipeline_adversarial_audit.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved full audit report to {out_path}")
