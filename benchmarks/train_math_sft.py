#!/usr/bin/env python3
"""Cold-Start SFT on Qwen2.5-Math-1.5B with LoRA and Strict Prompt Masking.

Trains Qwen2.5-Math-1.5B on the 605 certified bootstrap records (data/chalk_seeds_500.jsonl):
1. Applies ChatML conversation format.
2. Applies strict loss masking on the 5 reasoning tags completion tokens (label = -100 on prompts).
3. LoRA rank 32, alpha 64 on all linear projections (q, k, v, o, gate, up, down).
4. Effective batch size 16 (per-device 2 x grad accum 8) over 3 epochs with cosine decay.
5. Tesla T4 memory management: expandable segments, gradient checkpointing, peak VRAM ~8.5 GB.
"""

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Any

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "src"))

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType

try:
    import t4_kernels
    HAS_T4_KERNELS = True
except ImportError:
    HAS_T4_KERNELS = False

SYSTEM_PROMPT = (
    "You are an expert mathematician and theorem prover. "
    "Reason step by step using strict tagged blocks: <explore>, <conjecture>, "
    "<test_edge_cases>, <lemma_isolate>, and <formal_proof>. "
    "End your formal proof with the final answer in \\boxed{}."
)

class ChalkMathSFTDataset(Dataset):
    """Dataset for 5-tag Cold-Start SFT with strict prompt loss masking."""

    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.records = []

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        problem = rec["problem"]
        trace = rec["trace"]

        prompt = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{problem}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        completion = trace + "<|im_end|>"
        full_text = prompt + completion

        enc_full = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        enc_prompt = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )

        input_ids = enc_full["input_ids"][0]
        attention_mask = enc_full["attention_mask"][0]
        labels = input_ids.clone()

        # Dynamic token boundary detection for ChatML assistant turn
        # Locate the assistant prefix tokens (<|im_start|>assistant) in input_ids
        assistant_header_ids = self.tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)
        m_len = len(assistant_header_ids)
        boundary = None
        for i in range(len(input_ids) - m_len, -1, -1):
            if input_ids[i : i + m_len].tolist() == assistant_header_ids:
                # The token immediately following assistant header is the newline
                boundary = i + m_len + 1
                break

        if boundary is None or boundary >= len(input_ids):
            boundary = min(enc_prompt["input_ids"].shape[1], len(input_ids) - 1)

        # Strict loss masking: do not calculate cross-entropy on the prompt tokens
        labels[:boundary] = -100

        # Safety: guarantee at least one valid completion label exists to prevent NaN cross-entropy loss
        if (labels != -100).sum() == 0:
            labels[-1] = input_ids[-1]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    max_len = max(item["input_ids"].size(0) for item in batch)

    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []

    for item in batch:
        cur_len = item["input_ids"].size(0)
        pad_len = max_len - cur_len

        if pad_len > 0:
            padded_input = torch.cat([item["input_ids"], torch.full((pad_len,), pad_token_id, dtype=torch.long)])
            padded_mask = torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
            padded_labels = torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
        else:
            padded_input = item["input_ids"]
            padded_mask = item["attention_mask"]
            padded_labels = item["labels"]

        batch_input_ids.append(padded_input)
        batch_attention_mask.append(padded_mask)
        batch_labels.append(padded_labels)

    return {
        "input_ids": torch.stack(batch_input_ids),
        "attention_mask": torch.stack(batch_attention_mask),
        "labels": torch.stack(batch_labels)
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Cold-Start SFT for Qwen2.5-Math-1.5B")
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--data_path", default="data/chalk_seeds_500.jsonl")
    parser.add_argument("--output_dir", default="results/chalk_math_1.5b_sft")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true", help="Run with miniature configuration for CI / CPU testing")
    parser.add_argument("--force_cpu", action="store_true", help="Force full 1.5B model execution on CPU despite low system RAM")
    return parser.parse_args()


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.dry_run else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    if device.type == "cpu" and not args.dry_run and not args.force_cpu:
        try:
            with open("/proc/meminfo", "r") as f:
                meminfo = f.read()
            m = re.search(r"MemAvailable:\s+(\d+)", meminfo)
            if m:
                avail_gb = int(m.group(1)) / (1024 * 1024)
                if avail_gb < 10.0:
                    print("\n" + "!" * 75)
                    print(f"  [RAM SAFETY GUARD] Available system memory is {avail_gb:.1f} GB.")
                    print("  Loading and training Qwen2.5-Math-1.5B on CPU in FP32 requires >= 10 GB RAM.")
                    print("  Proceeding would trigger the Linux Out-Of-Memory (OOM) killer and crash")
                    print("  system applications (including IDE/desktop).")
                    print("  Options:")
                    print("    1. Run on a CUDA GPU (e.g. Tesla T4 in Google Colab / cloud instance)")
                    print("    2. Pass --dry_run for lightweight local pipeline verification")
                    print("    3. Pass --force_cpu if you have configured sufficient swap")
                    print("!" * 75 + "\n")
                    sys.exit(1)
        except Exception as e:
            if "RAM SAFETY GUARD" in str(e):
                raise e

    print("=" * 75)
    print("  COLD-START SFT: Qwen2.5-Math-1.5B (LoRA + 5-Tag Loss Masking)")
    print(f"Device: {device} | t4_kernels Available: {HAS_T4_KERNELS}")
    print(f"Epochs: {args.epochs} | Batch: {args.batch_size} | Grad Accum: {args.grad_accum} | Effective BS: {args.batch_size * args.grad_accum}")
    print(f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}, target=all linear projections")
    print("=" * 75)

    # 1. Load Tokenizer
    local_snapshot = "/home/kriday/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2"
    tokenizer_path = local_snapshot if os.path.exists(local_snapshot) else args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Model
    if args.dry_run:
        print("[DRY RUN] Instantiating miniature test model from configuration...")
        cfg = AutoConfig.from_pretrained(tokenizer_path)
        cfg.num_hidden_layers = 2
        cfg.hidden_size = 128
        cfg.intermediate_size = 256
        cfg.num_attention_heads = 4
        cfg.num_key_value_heads = 2
        model = AutoModelForCausalLM.from_config(cfg)
    else:
        print(f"Loading base model: {args.model_name_or_path} in FP16...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            device_map="auto" if device.type == "cuda" else None
        )
        if device.type == "cuda":
            model.gradient_checkpointing_enable()

    # 3. Setup LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    model.to(device)
    model.print_trainable_parameters()

    # 4. Prepare Dataset & DataLoader
    if args.dry_run:
        args.max_length = min(args.max_length, 128)
        args.grad_accum = 1
        args.logging_steps = 1
        dataset = ChalkMathSFTDataset(args.data_path, tokenizer, max_length=args.max_length)
        dataset.records = dataset.records[:8]
    else:
        dataset = ChalkMathSFTDataset(args.data_path, tokenizer, max_length=args.max_length)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id)
    )

    total_steps = len(dataloader) // args.grad_accum * args.epochs
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)

    warmup_steps = int(total_steps * args.warmup_ratio)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # 5. Training Loop
    model.train()
    global_step = 0
    step_losses = []
    start_time = time.time()

    print(f"Beginning training: Total Planned Steps = {total_steps}, Warmup Steps = {warmup_steps}")

    accumulated_loss = 0.0
    optimizer.zero_grad()

    for epoch in range(args.epochs):
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss / args.grad_accum
            loss.backward()

            accumulated_loss += loss.item() * args.grad_accum

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                step_losses.append(accumulated_loss)

                if global_step % args.logging_steps == 0 or global_step == total_steps:
                    lr_curr = scheduler.get_last_lr()[0]
                    vram_mb = torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else 0.0
                    print(
                        f"Epoch {epoch+1}/{args.epochs} | Step {global_step}/{total_steps} | "
                        f"Loss: {accumulated_loss:.4f} | LR: {lr_curr:.2e} | Peak VRAM: {vram_mb:.1f} MB"
                    )

                accumulated_loss = 0.0

                if args.max_steps > 0 and global_step >= args.max_steps:
                    break

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    elapsed_time = time.time() - start_time
    print("=" * 75)
    print(f"Training Complete in {elapsed_time:.1f}s | Steps: {global_step} | Final Loss: {step_losses[-1] if step_losses else 0.0:.4f}")
    print("=" * 75)

    # 6. Save LoRA weights & metadata
    adapter_path = os.path.join(args.output_dir, "lora_adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(args.output_dir)

    summary = {
        "model": args.model_name_or_path,
        "dataset": args.data_path,
        "records": len(dataset),
        "epochs": args.epochs,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "final_loss": step_losses[-1] if step_losses else None,
        "elapsed_seconds": elapsed_time,
        "peak_vram_mb": torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else 0.0,
        "device": str(device)
    }

    with open(os.path.join(args.output_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved LoRA weights to {adapter_path}")
    print(f"Saved summary to {os.path.join(args.output_dir, 'training_summary.json')}")


if __name__ == "__main__":
    train(parse_args())
