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
import re
import sys
import time
from typing import Dict, List, Any, Optional, Tuple

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
    DEFAULT_MATH_PROMPTS,
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


def check_adapter_compatibility(adapter_path: str, config: Any) -> Tuple[bool, Optional[str]]:
    """Inspects adapter weights without throwing unhandled exceptions.

    Checks:
    1. Existence of adapter files (adapter_config.json and weights).
    2. Tensor shape compatibility against base model dimensions (hidden_size).
    3. Layer count compatibility.
    """
    if not adapter_path or not os.path.exists(adapter_path):
        return False, f"Path does not exist: {adapter_path}"
    if not os.path.isdir(adapter_path):
        return False, f"Not a directory: {adapter_path}"

    cfg_path = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.exists(cfg_path):
        return False, f"Missing adapter_config.json in {adapter_path}"

    safetensors_path = os.path.join(adapter_path, "adapter_model.safetensors")
    bin_path = os.path.join(adapter_path, "adapter_model.bin")
    if not os.path.exists(safetensors_path) and not os.path.exists(bin_path):
        return False, f"Missing adapter weights (neither safetensors nor bin found) in {adapter_path}"

    if os.path.exists(safetensors_path):
        try:
            from safetensors import safe_open
            with safe_open(safetensors_path, framework="pt") as f:
                tensor_names = list(f.keys())
                layers = set()
                for name in tensor_names:
                    m = re.search(r"layers\.(\d+)\.", name)
                    if m:
                        layers.add(int(m.group(1)))
                    if "lora_A.weight" in name and ("q_proj" in name or "gate_proj" in name):
                        t_shape = f.get_slice(name).get_shape()
                        in_features = t_shape[1]
                        if hasattr(config, "hidden_size") and in_features != config.hidden_size:
                            return False, (
                                f"Hidden dimension mismatch: adapter tensor '{name}' has in_features={in_features}, "
                                f"but base model has hidden_size={config.hidden_size}."
                            )
                if hasattr(config, "num_hidden_layers") and layers:
                    max_layer = max(layers) + 1
                    if max_layer < config.num_hidden_layers:
                        return False, (
                            f"Layer count mismatch: adapter only contains {max_layer} layers, "
                            f"but base model requires {config.num_hidden_layers} layers."
                        )
        except Exception as e:
            return False, f"Error inspecting safetensors: {e}"
    elif os.path.exists(bin_path):
        try:
            state = torch.load(bin_path, map_location="cpu")
            layers = set()
            for name, tensor in state.items():
                m = re.search(r"layers\.(\d+)\.", name)
                if m:
                    layers.add(int(m.group(1)))
                if "lora_A.weight" in name and ("q_proj" in name or "gate_proj" in name):
                    in_features = tensor.shape[1]
                    if hasattr(config, "hidden_size") and in_features != config.hidden_size:
                        return False, (
                            f"Hidden dimension mismatch: adapter tensor '{name}' has in_features={in_features}, "
                            f"but base model has hidden_size={config.hidden_size}."
                        )
            if hasattr(config, "num_hidden_layers") and layers:
                max_layer = max(layers) + 1
                if max_layer < config.num_hidden_layers:
                    return False, (
                        f"Layer count mismatch: adapter only contains {max_layer} layers, "
                        f"but base model requires {config.num_hidden_layers} layers."
                    )
        except Exception as e:
            return False, f"Error inspecting bin file: {e}"

    return True, None


def load_model_and_tokenizer(
    model_name_or_path: str,
    sft_adapter_path: Optional[str] = None,
    lora_r: int = 16,
    lora_alpha: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    dry_run: bool = False,
):
    local_snapshot = "/home/kriday/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2"
    effective_path = local_snapshot if os.path.exists(local_snapshot) and "1.5B" in model_name_or_path else model_name_or_path

    print(f"Loading tokenizer from {effective_path}...")
    tokenizer = AutoTokenizer.from_pretrained(effective_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model config from {effective_path}...")
    config = AutoConfig.from_pretrained(effective_path, trust_remote_code=True)

    if dry_run:
        print("[DRY RUN] Instantiating miniature test model from configuration...")
        config.num_hidden_layers = 2
        config.hidden_size = 128
        config.intermediate_size = 256
        config.num_attention_heads = 4
        config.num_key_value_heads = 2
        model = AutoModelForCausalLM.from_config(config)
        model = model.to(device)
    else:
        configure_rope_scaling(config, max_position_embeddings=8192, rope_type="yarn", factor=2.0)
        dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else (torch.float16 if device == "cuda" else torch.float32)
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            config=config,
            torch_dtype=dtype,
            device_map=device if device == "cuda" else None,
            trust_remote_code=True,
        )
        if device == "cpu":
            model = model.to(torch.float32)

        if device == "cuda":
            model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()

    # Robust SFT adapter loading with dimension audit
    model_is_peft = False
    if sft_adapter_path:
        is_compat, reason = check_adapter_compatibility(sft_adapter_path, config)
        if is_compat:
            print(f"Loading certified SFT adapter from {sft_adapter_path}...")
            try:
                model = PeftModel.from_pretrained(model, sft_adapter_path, is_trainable=True)
                model_is_peft = True
                print("Successfully loaded SFT adapter.")
            except Exception as e:
                print(f"[WARNING] PeftModel.from_pretrained failed ({e}). Falling back to fresh LoRA.")
                model_is_peft = False
        else:
            print("=" * 75)
            print(f"[ADAPTER AUDIT WARNING] Cannot load adapter from '{sft_adapter_path}':")
            print(f"  Reason: {reason}")
            print("  Safely bypassing incompatible checkpoint and initializing a fresh LoRA adapter.")
            print("=" * 75)

    if not model_is_peft:
        print(f"Initializing fresh LoRA adapter (r={lora_r}, alpha={lora_alpha}, target=all linear projections)...")
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
    if data_path and os.path.exists(data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    math_items.append({
                        "prompt": rec["problem"],
                        "prompt_type": PromptType.CONTEST_MATH,
                        "ground_truth": rec.get("boxed_answer") or rec.get("ground_truth", ""),
                    })

    if not math_items:
        print("[INFO] No math items loaded from data_path; using DEFAULT_MATH_PROMPTS fallback.")
        math_items = None

    buffer = FormatDiscriminationReplayBuffer(
        contest_math_data=math_items,
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
    rollout_ids_list: List[List[torch.Tensor]],
    rottweiler: RottweilerVerifier,
    abstention: CalibratedAbstentionRewardEngine,
    format_engine: FormatDiscriminationRewardEngine,
    repetition_checker: Optional[FourGramRepetitionCriteria] = None,
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
            comp_ids = rollout_ids_list[i][g]

            if p_type == PromptType.CONTEST_MATH:
                # Math item: verify 5-tag scaffold and symbolic correctness
                eval_res = rottweiler.verify(
                    completion=comp,
                    ground_truth=gold,
                )
                r_correct = 1.0 if eval_res["symbolic_correct"] else -1.5

                # Granular scaffold compliance: reward partial tag progress to break rollout ties
                tags = ["explore", "conjecture", "test_edge_cases", "lemma_isolate", "formal_proof"]
                tags_found = sum(1 for t in tags if f"<{t}>" in comp and f"</{t}>" in comp)
                # Formatted tags yield -1.0 for 0 tags up to 0.0 for all 5 tags
                r_format = (tags_found / 5.0) - 1.0
                if eval_res.get("boxed_answer"):
                    r_format += 0.2

                # Check abstention
                r_abstain, _ = abstention.compute_reward(comp, gold)
                # If model cleanly abstained on an impossible/unsolvable problem
                if "<abstain>" in comp.lower() or "cannot be determined" in comp.lower():
                    final_r = r_abstain
                else:
                    final_r = r_correct + 0.3 * r_format
            else:
                # General conversation: penalize leaked reasoning XML tags
                final_r = format_engine.evaluate_format_compliance(p_type, comp)

            # Check and apply 4-gram repetition penalty
            if repetition_checker is not None:
                has_looped, rep_penalty, _ = repetition_checker.evaluate_sequence(comp_ids)
                if has_looped:
                    final_r += rep_penalty

            rewards[i, g] = float(final_r)

    return rewards


def run_training_loop(args):
    print("=" * 75)
    print("  BABY-CHALK (1.5B) RL POST-TRAINING ON TESLA T4")
    print("=" * 75)

    device = "cuda" if torch.cuda.is_available() and not args.dry_run else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    if device == "cpu" and not args.dry_run and not args.force_cpu:
        try:
            with open("/proc/meminfo", "r") as f:
                meminfo = f.read()
            m = re.search(r"MemAvailable:\s+(\d+)", meminfo)
            if m:
                avail_gb = int(m.group(1)) / (1024 * 1024)
                if avail_gb < 10.0:
                    print("\n" + "!" * 75)
                    print(f"  [RAM SAFETY GUARD] Available system memory is {avail_gb:.1f} GB.")
                    print("  Loading and training Qwen2.5-Math-1.5B on CPU requires >= 10 GB RAM.")
                    print("  Pass --dry_run for lightweight local CPU testing or run on a CUDA GPU.")
                    print("!" * 75 + "\n")
                    sys.exit(1)
        except Exception as e:
            if "RAM SAFETY GUARD" in str(e):
                raise e

    model, tokenizer = load_model_and_tokenizer(
        model_name_or_path=args.model_name_or_path,
        sft_adapter_path=args.sft_adapter_path,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        device=device,
        dry_run=args.dry_run,
    )

    buffer, rottweiler, abstention, format_engine = build_prompts_and_evaluators(args.data_path)
    loss_fn = ModifiedGRPOLoss(clip_eps_low=0.2, clip_eps_high=0.2, beta_kl=0.02)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)

    # Set up EOS token list: handle both <|endoftext|> and ChatML <|im_end|>
    eos_token_ids = [tokenizer.eos_token_id]
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end_id, int) and im_end_id not in eos_token_ids and im_end_id != tokenizer.unk_token_id:
        eos_token_ids.append(im_end_id)

    print(f"Starting GRPO training for {args.max_steps} steps (batch_size={args.batch_size}, G={args.num_generations})...")
    metrics_history = []
    start_train_time = time.time()

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
        step_stopping_criteria: Optional[FourGramRepetitionCriteria] = None

        with torch.no_grad():
            for b_idx, prompt_str in enumerate(formatted_prompts):
                inputs = tokenizer(prompt_str, return_tensors="pt", truncation=True, max_length=args.max_prompt_len).to(device)
                prompt_len = inputs.input_ids.shape[1]

                stopping_criteria = [
                    FourGramRepetitionCriteria(
                        max_occurrences=2,
                        penalty=-0.5,
                        window_size=256,
                        prompt_lengths=[prompt_len] * args.num_generations,
                        stop_on_any=False,
                    )
                ]
                step_stopping_criteria = stopping_criteria[0]

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
                    eos_token_id=eos_token_ids,
                )

                for g in range(args.num_generations):
                    full_seq = gen_out[g]
                    comp_seq = full_seq[prompt_len:]
                    text = tokenizer.decode(comp_seq, skip_special_tokens=True)
                    rollout_texts[b_idx].append(text)
                    rollout_ids_list[b_idx].append(comp_seq)

        if device == "cuda":
            torch.cuda.empty_cache()

        # 3. Evaluate rewards
        rewards = evaluate_rollout_rewards(
            batch_items=batch_items,
            rollout_texts=rollout_texts,
            rollout_ids_list=rollout_ids_list,
            rottweiler=rottweiler,
            abstention=abstention,
            format_engine=format_engine,
            repetition_checker=step_stopping_criteria,
        ).to(device)

        # 4. Forward pass: compute differentiable log probabilities for GRPOLoss
        model.train()
        optimizer.zero_grad()
        accum_loss = 0.0

        # Compute advantages via tied group fix
        advantages = loss_fn.compute_advantages(rewards)

        # Micro-batched differentiable log_probs computation to bound VRAM
        for b_idx in range(args.batch_size):
            p_inputs = tokenizer(formatted_prompts[b_idx], return_tensors="pt", truncation=True, max_length=args.max_prompt_len).to(device)
            p_len = p_inputs.input_ids.shape[1]

            comp_seqs = rollout_ids_list[b_idx]
            max_c_len = max(len(s) for s in comp_seqs)
            if max_c_len == 0:
                continue

            padded_input_ids = []
            comp_lens = []
            for s in comp_seqs:
                full_ids = torch.cat([p_inputs.input_ids[0], s])
                padded_input_ids.append(full_ids)
                comp_lens.append(len(s))

            padded_batch = torch.nn.utils.rnn.pad_sequence(
                padded_input_ids,
                batch_first=True,
                padding_value=tokenizer.pad_token_id
            )
            logits = model(padded_batch).logits

            # Gather log_probs on completion tokens
            comp_logits = logits[:, p_len - 1 : -1, :]
            targets = padded_batch[:, p_len:]

            # Construct exact mask based on true completion lengths
            seq_len = targets.shape[1]
            mask = torch.zeros((len(comp_lens), seq_len), dtype=torch.float32, device=device)
            for g_i, c_len in enumerate(comp_lens):
                mask[g_i, :c_len] = 1.0

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

            # Backward immediately per prompt micro-batch to prevent VRAM accumulation
            micro_loss = step_loss / args.batch_size
            micro_loss.backward()
            accum_loss += micro_loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if device == "cuda":
            torch.cuda.empty_cache()

        step_time = time.time() - step_start_time
        vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0

        mean_r = rewards.mean().item()
        std_r = rewards.std().item()
        print(
            f"Step {step:03d}/{args.max_steps:03d} | "
            f"Loss: {accum_loss:.4f} | "
            f"Reward: {mean_r:+.3f} ± {std_r:.3f} | "
            f"Peak VRAM: {vram_gb:.2f} GB | "
            f"Time: {step_time:.2f}s"
        )

        metrics_history.append({
            "step": step,
            "loss": float(accum_loss),
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

    total_time = time.time() - start_train_time
    summary = {
        "model": args.model_name_or_path,
        "sft_adapter_path": args.sft_adapter_path,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "num_generations": args.num_generations,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "final_loss": metrics_history[-1]["loss"] if metrics_history else 0.0,
        "final_mean_reward": metrics_history[-1]["mean_reward"] if metrics_history else 0.0,
        "elapsed_seconds": total_time,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0,
        "device": str(device),
        "dry_run": args.dry_run,
    }
    with open(os.path.join(args.output_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 75)
    print(f"  TRAINING COMPLETE! Output saved to: {args.output_dir}")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="Baby-Chalk (1.5B) RL post-training on Tesla T4")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--sft_adapter_path", type=str, default=None, help="Path to pre-trained SFT LoRA checkpoint (default: None for fresh LoRA post-training)")
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
    parser.add_argument("--dry_run", action="store_true", help="Run with miniature test model on CPU for testing")
    parser.add_argument("--force_cpu", action="store_true", help="Force execution of full 1.5B model on CPU")
    args = parser.parse_args()

    run_training_loop(args)


if __name__ == "__main__":
    main()
