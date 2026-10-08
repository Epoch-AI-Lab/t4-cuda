#!/usr/bin/env bash
# scripts/run_kaggle_7b_sft.sh
# Multi-GPU Cold-Start SFT on Qwen2.5-Math-7B for Dual Tesla T4 GPUs (32 GB Combined VRAM)
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM="false"

echo "=========================================================================="
echo "  BIG CHALK SFT: Qwen2.5-Math-7B on Kaggle Dual Tesla T4 Silicon"
echo "=========================================================================="

nvidia-smi

python3 -c "
import torch
print('CUDA Available:', torch.cuda.is_available())
print('Device Count:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB)')
"

echo "Starting Multi-GPU Cold-Start SFT..."
python3 benchmarks/train_math_sft.py \
    --model_name_or_path "Qwen/Qwen2.5-Math-7B" \
    --data_path "data/chalk_seeds_500.jsonl" \
    --output_dir "results/big_chalk_7b_sft" \
    --epochs 3 \
    --batch_size 1 \
    --grad_accum 16 \
    --lr 1.5e-4 \
    --max_length 2048 \
    --lora_r 32 \
    --lora_alpha 64 \
    --logging_steps 5

echo "Training complete. Running external contest evaluation..."
python3 benchmarks/eval_math_benchmark.py \
    --model_name_or_path "Qwen/Qwen2.5-Math-7B" \
    --adapter_path "results/big_chalk_7b_sft/lora_adapter" \
    --benchmark_path "data/external_math_eval.json" \
    --output_path "results/big_chalk_7b_eval_results.json" \
    --max_new_tokens 2048

echo "All tasks finished successfully."
