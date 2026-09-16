#!/usr/bin/env python3
"""Adversarial Verification Suite for RL Pipeline and LoRA Adapter Loader.

Tests:
1. Shape and layer compatibility detection for LoRA adapters.
2. Incompatible adapter rejection and safe fallback to fresh LoRA.
3. Anti-looping criteria and sequence repetition penalty evaluation.
4. GRPO loss computation and micro-batched backward pass.
5. End-to-end dry run execution of train_baby_chalk_rl.
"""

import os
import sys
import tempfile
import unittest
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "src"))

from benchmarks.train_baby_chalk_rl import (
    check_adapter_compatibility,
    load_model_and_tokenizer,
    build_prompts_and_evaluators,
    evaluate_rollout_rewards,
    run_training_loop,
)
from src.rl.anti_looping import FourGramRepetitionCriteria
from src.rl.modified_grpo_loss import ModifiedGRPOLoss


class TestAdapterAuditorAndRLLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.local_snapshot = "/home/kriday/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2"
        cls.adapter_path = os.path.join(REPO_DIR, "results/chalk_math_1.5b_sft/lora_adapter")

    def test_01_dry_run_adapter_mismatch_detection(self):
        """Verify check_adapter_compatibility detects hidden_size and layer mismatches."""
        if not os.path.exists(self.adapter_path):
            self.skipTest(f"Adapter path not found: {self.adapter_path}")

        # Instantiate full 1.5B config
        config = AutoConfig.from_pretrained(self.local_snapshot)
        self.assertEqual(config.hidden_size, 1536)
        self.assertEqual(config.num_hidden_layers, 28)

        is_compat, reason = check_adapter_compatibility(self.adapter_path, config)
        self.assertFalse(is_compat, "Incompatible adapter must NOT be reported as compatible")
        self.assertIsNotNone(reason)
        self.assertTrue(
            "Hidden dimension mismatch" in reason or "Layer count mismatch" in reason,
            f"Reason must indicate hidden_size or layer count mismatch, got: {reason}"
        )

    def test_02_compatible_adapter_detection(self):
        """Verify check_adapter_compatibility accepts matching configuration."""
        if not os.path.exists(self.adapter_path):
            self.skipTest(f"Adapter path not found: {self.adapter_path}")

        # Config matching the dry-run adapter (hidden_size=128, 2 layers)
        mini_config = AutoConfig.from_pretrained(self.local_snapshot)
        mini_config.hidden_size = 128
        mini_config.num_hidden_layers = 2

        is_compat, reason = check_adapter_compatibility(self.adapter_path, mini_config)
        self.assertTrue(is_compat, f"Matching adapter should be accepted, but got: {reason}")
        self.assertIsNone(reason)

    def test_03_safe_fallback_on_incompatible_adapter(self):
        """Verify load_model_and_tokenizer safely falls back to fresh LoRA without RuntimeError."""
        model, tokenizer = load_model_and_tokenizer(
            model_name_or_path=self.local_snapshot,
            sft_adapter_path="/non/existent/path/for/testing",
            lora_r=16,
            lora_alpha=32,
            device="cpu",
            dry_run=True,
        )
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, "peft_config"), "Model must be wrapped in PeftModel")

    def test_04_anti_looping_criteria_and_penalty(self):
        """Verify FourGramRepetitionCriteria sequence evaluation and penalty."""
        criteria = FourGramRepetitionCriteria(max_occurrences=2, penalty=-0.5, n=4)

        # Repeating sequence: [1, 2, 3, 4] repeated 3 times
        repeating_seq = torch.tensor([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4], dtype=torch.long)
        has_looped, penalty, max_occ = criteria.evaluate_sequence(repeating_seq)
        self.assertTrue(has_looped, "Repeating 4-gram > 2 times must trigger loop flag")
        self.assertEqual(penalty, -0.5)

        # Non-repeating sequence
        clean_seq = torch.tensor([10, 20, 30, 40, 50, 60, 70, 80], dtype=torch.long)
        has_looped_clean, penalty_clean, _ = criteria.evaluate_sequence(clean_seq)
        self.assertFalse(has_looped_clean)
        self.assertEqual(penalty_clean, 0.0)

    def test_05_grpo_micro_batched_gradient_flow(self):
        """Verify micro-batched forward/backward pass produces valid gradients without NaNs."""
        loss_fn = ModifiedGRPOLoss(clip_eps_low=0.2, clip_eps_high=0.2, beta_kl=0.0)

        # 2 rollouts of length 8
        log_probs = torch.randn(1, 2, 8, requires_grad=True)
        old_log_probs = log_probs.detach().clone()
        advantages = torch.tensor([[1.0, -1.0]])
        mask = torch.ones(1, 2, 8)

        loss, metrics = loss_fn(
            log_probs=log_probs,
            old_log_probs=old_log_probs,
            advantages=advantages,
            completion_mask=mask,
        )

        self.assertFalse(torch.isnan(loss).item(), "GRPO loss must not be NaN")
        loss.backward()
        self.assertIsNotNone(log_probs.grad)
        self.assertFalse(torch.isnan(log_probs.grad).any().item(), "Gradients must not contain NaNs")

    def test_06_e2e_dry_run_pipeline(self):
        """Verify full run_training_loop executes cleanly without exceptions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            class DummyArgs:
                model_name_or_path = self.local_snapshot
                sft_adapter_path = None
                data_path = os.path.join(REPO_DIR, "data/chalk_seeds_500.jsonl")
                output_dir = tmpdir
                max_steps = 2
                batch_size = 1
                num_generations = 2
                learning_rate = 1e-4
                max_prompt_len = 64
                max_completion_len = 32
                temperature = 0.8
                lora_r = 8
                lora_alpha = 16
                save_steps = 2
                dry_run = True
                force_cpu = False

            args = DummyArgs()
            run_training_loop(args)

            summary_file = os.path.join(tmpdir, "training_summary.json")
            self.assertTrue(os.path.exists(summary_file), "training_summary.json must exist")
            ckpt_dir = os.path.join(tmpdir, "checkpoint_step_2")
            self.assertTrue(os.path.exists(ckpt_dir), "Checkpoint directory must be saved")


if __name__ == "__main__":
    unittest.main()
