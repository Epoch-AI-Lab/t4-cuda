#!/usr/bin/env python3
"""Baby-Chalk (1.5B) RL Post-Training Runner on Tesla T4 (Google Colab Compatible).

Integrates:
1. Qwen2.5-Math-1.5B base model or fine-tuned SFT checkpoint with LoRA.
2. 16k context window and memory-efficient SDPA attention via src.rope_scaling.
3. Cursor-style ModifiedGRPOLoss with unscaled advantages (A_i = r_i - r_bar) and detached CISPO clipping.
4. Rottweiler SymPy verifier with 5-tag XML scaffold validation and reverse equation substitution.
5. Calibrated abstention engine (+1.0 correct, 0.0 abstain, -1.5 incorrect).
6. Anti-looping 4-gram repetition stopping criteria with sliding window.
7. Format discrimination replay buffer (15-20% general prompts, -1.5 penalty on leaked XML).
8. Automatic checkpoint saving and metric logging designed for Colab runs.
"""

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Any, Optional

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "src"))

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)
from peft import LoraConfig, get_peft_model, PeftModel

from src.rope_scaling import configure_rope_scaling, execute_sdpa_gqa_attention
from src.rl.modified_grpo_loss import ModifiedGRPOLoss
from src.rl.rottweiler_verifier import RottweilerVerifier
from src.rl.calibrated_abstention import CalibratedAbstentionRewardEngine
from src.rl.anti_looping import FourGramRepetitionCriteria
from src.rl.format_replay_buffer import (
    FormatDiscriminationReplayBuffer,
    FormatDiscriminationRewardEngine,
    PromptType,
)

SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician and theorem prover. "
    "Reason step by step using strict tagged blocks: <explore>, <conjecture>, "
    "<test_edge_cases>, <lemma_isolate>, and <formal_proof>. "
    "End your formal proof with the final answer in \\boxed{}."
)

SYSTEM_PROMPT_GENERAL = (
    "You are a helpful, direct assistant. Answer the user's request clearly and concisely."
)


def load_model_and_tokenizer(
    model_name_or_path: str,
    sft_adapter_path: Optional[str] = None,
    lora_r: int = 16,
    lora_alpha: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    print(f"Loading tokenizer from {model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model from {model_name_or_path}...")
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    configure_rope_scaling(config, max_position_embeddings=8192, rope_type="yarn", factor=2.0)

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        config=config,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )

    # If an existing SFT LoRA checkpoint exists, load it
    if sft_adapter_path and os.path.exists(sft_adapter_path):
        print(f"Loading existing SFT adapter from {sft_adapter_path}...")
        model = PeftModel.from_pretrained(model, sft_adapter_path, is_trainable=True)
    else:
        print(f"Initializing new LoRA adapter (r={lora_r}, alpha={lora_alpha})...")
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()
    return model, tokenizer


def build_prompts_and_evaluators(data_path: str):
    # Initialize replay buffer with 80% math and 20% general prompts
    math_items = []
    if os.path.exists(data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    math_items.append({
                        "prompt": rec["problem"],
                        "prompt_type": PromptType.CONTEST_MATH,
                        "ground_truth": rec.get("boxed_answer") or rec.get("ground_truth", ""),
                    })

    buffer = FormatDiscriminationReplayBuffer(
        math_data=math_items,
        math_ratio=0.80,
        general_ratio=0.20,
        seed=42,
    )

    rottweiler = RottweilerVerifier(timeout_seconds=3.0)
    abstention = CalibratedAbstentionRewardEngine()
    format_engine = FormatDiscriminationRewardEngine(general_penalty=-1.5)

    return buffer, rottweiler, abstention, format_engine


def evaluate_rollout_rewards(
    batch_items: List[Dict[str, Any]],
    rollout_texts: List[List[str]],
    rottweiler: RottweilerVerifier,
    abstention: CalibratedAbstentionRewardEngine,
    format_engine: FormatDiscriminationRewardEngine,
) -> torch.Tensor:
    """Computes composite rewards across all rollouts in batch.

    Returns tensor of shape [batch_size, num_generations].
    """
    batch_size = len(batch_items)
    num_gen = len(rollout_texts[0])
    rewards = torch.zeros((batch_size, num_gen), dtype=torch.float32)

    for i, item in enumerate(batch_items):
        p_type = item["prompt_type"]
        gold = item.get("ground_truth", "")

        for g in range(num_gen):
            comp = rollout_texts[i][g]

            if p_type == PromptType.CONTEST_MATH:
                # Math item: verify 5-tag scaffold and symbolic correctness
                eval_res = rottweiler.verify_completion(
                    completion=comp,
                    ground_truth=gold,
                )
                r_correct = 1.0 if eval_res["symbolic_correct"] else -1.5
                # Scaffold compliance bonus/penalty
                r_format = 0.0 if eval_res["scaffold_adherent"] else -1.0

                # Check abstention
                r_abstain, _ = abstention.compute_reward(comp, gold)
                # If model cleanly abstained on an impossible/unsolvable problem
                if "<abstain>" in comp.lower() or "cannot be determined" in comp.lower():
                    final_r = r_abstain
                else:
                    final_r = r_correct + 0.2 * r_format
            else:
                # General conversation: penalize leaked reasoning XML tags
                final_r = format_engine.evaluate_format_compliance(p_type, comp)

            rewards[i, g] = float(final_r)

    return rewards


def run_training_loop(args):
    print("=" * 70)
    print("  BABY-CHALK (1.5B) RL POST-TRAINING ON TESLA T4")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(
        model_name_or_path=args.model_name_or_path,
        sft_adapter_path=args.sft_adapter_path,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        device=device,
    )

    buffer, rottweiler, abstention, format_engine = build_prompts_and_evaluators(args.data_path)
    loss_fn = ModifiedGRPOLoss(clip_eps_low=0.2, clip_eps_high=0.2, beta_kl=0.02)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)

    print(f"Starting GRPO training for {args.max_steps} steps (batch_size={args.batch_size}, G={args.num_generations})...")
    metrics_history = []

    for step in range(1, args.max_steps + 1):
        step_start_time = time.time()
        batch_items = buffer.sample_batch(args.batch_size)

        # 1. Prepare formatted prompts
        formatted_prompts = []
        for item in batch_items:
            sys_p = SYSTEM_PROMPT_MATH if item["prompt_type"] == PromptType.CONTEST_MATH else SYSTEM_PROMPT_GENERAL
            formatted = f"<|im_start|>system\n{sys_p}<|im_end|>\n<|im_start|>user\n{item['prompt']}<|im_end|>\n<|im_start|>assistant\n"
            formatted_prompts.append(formatted)

        # 2. Generate rollouts with anti-looping stopping criteria
        model.eval()
        rollout_texts: List[List[str]] = [[] for _ in range(args.batch_size)]
        rollout_ids_list: List[List[torch.Tensor]] = [[] for _ in range(args.batch_size)]

        with torch.no_grad():
            for b_idx, prompt_str in enumerate(formatted_prompts):
                inputs = tokenizer(prompt_str, return_tensors="pt", truncation=True, max_length=args.max_prompt_len).to(device)
                prompt_len = inputs.input_ids.shape[1]

                stopping_criteria = [
                    FourGramRepetitionCriteria(max_occurrences=2, penalty=-0.5, window_size=256, prompt_lengths=[prompt_len])
                ]

                # Generate G rollouts
                gen_out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_completion_len,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=0.95,
                    num_return_sequences=args.num_generations,
                    stopping_criteria=stopping_criteria,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

                for g in range(args.num_generations):
                    full_seq = gen_out[g]
                    comp_seq = full_seq[prompt_len:]
                    text = tokenizer.decode(comp_seq, skip_special_tokens=True)
                    rollout_texts[b_idx].append(text)
                    rollout_ids_list[b_idx].append(comp_seq)

        # 3. Evaluate rewards
        rewards = evaluate_rollout_rewards(
            batch_items=batch_items,
            rollout_texts=rollout_texts,
            rottweiler=rottweiler,
            abstention=abstention,
            format_engine=format_engine,
        ).to(device)

        # 4. Forward pass: compute differentiable log probabilities for GRPOLoss
        model.train()
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        # Compute advantages via tied group fix
        advantages = loss_fn.compute_advantages(rewards)

        # Differentiable log_probs computation
        for b_idx in range(args.batch_size):
            p_inputs = tokenizer(formatted_prompts[b_idx], return_tensors="pt", truncation=True, max_length=args.max_prompt_len).to(device)
            p_len = p_inputs.input_ids.shape[1]

            # Batch G completions together
            comp_seqs = rollout_ids_list[b_idx]
            max_c_len = max(len(s) for s in comp_seqs)
            if max_c_len == 0:
                continue

            padded_input_ids = []
            comp_masks = []
            for s in comp_seqs:
                full_ids = torch.cat([p_inputs.input_ids[0], s])
                padded_input_ids.append(full_ids)

            # Pad batch
            padded_batch = torch.nn.utils.rnn.pad_sequence(padded_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
            logits = model(padded_batch).logits

            # Gather log_probs on completion tokens
            comp_logits = logits[:, p_len - 1 : -1, :]
            targets = padded_batch[:, p_len:]
            mask = (targets != tokenizer.pad_token_id).float()

            log_p = F.log_softmax(comp_logits, dim=-1)
            gathered_log_p = torch.gather(log_p, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

            # Old log probs for off-policy ratio
            old_log_p = gathered_log_p.detach().clone()

            step_loss, _ = loss_fn(
                log_probs=gathered_log_p.unsqueeze(0),
                old_log_probs=old_log_p.unsqueeze(0),
                advantages=advantages[b_idx].unsqueeze(0),
                completion_mask=mask.unsqueeze(0),
            )
            total_loss = total_loss + (step_loss / args.batch_size)

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        step_time = time.time() - step_start_time
        vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0

        mean_r = rewards.mean().item()
        std_r = rewards.std().item()
        print(
            f"Step {step:03d}/{args.max_steps:03d} | "
            f"Loss: {total_loss.item():.4f} | "
            f"Reward: {mean_r:+.3f} ± {std_r:.3f} | "
            f"Peak VRAM: {vram_gb:.2f} GB | "
            f"Time: {step_time:.2f}s"
        )

        metrics_history.append({
            "step": step,
            "loss": float(total_loss.item()),
            "mean_reward": mean_r,
            "std_reward": std_r,
            "peak_vram_gb": vram_gb,
            "step_time_sec": step_time,
        })

        # Checkpoint every save_steps
        if step % args.save_steps == 0 or step == args.max_steps:
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint_step_{step}")
            print(f"Saving checkpoint to {ckpt_dir}...")
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
                json.dump(metrics_history, f, indent=2)

    print("=" * 70)
    print(f"  TRAINING COMPLETE! Output saved to: {args.output_dir}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Baby-Chalk (1.5B) RL post-training on Tesla T4")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--sft_adapter_path", type=str, default=os.path.join(REPO_DIR, "results/chalk_math_1.5b_sft/lora_adapter"))
    parser.add_argument("--data_path", type=str, default=os.path.join(REPO_DIR, "data/chalk_seeds_500.jsonl"))
    parser.add_argument("--output_dir", type=str, default=os.path.join(REPO_DIR, "results/baby_chalk_rl"))
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_prompt_len", type=int, default=512)
    parser.add_argument("--max_completion_len", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--save_steps", type=int, default=10)
    args = parser.parse_args()

    run_training_loop(args)


if __name__ == "__main__":
    main()
