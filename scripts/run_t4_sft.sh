#!/usr/bin/env bash
# Item 4: Cold-Start SFT on Tesla T4 & External Benchmark Suite
set -e

echo "=========================================================================="
echo "  Item 4: Cold-Start SFT for Qwen2.5-Math-1.5B on Tesla T4"
echo "=========================================================================="

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

export PYTHONPATH="$REPO_DIR:$REPO_DIR/src:$PYTHONPATH"

# 1. Sanity Check: Confirm CUDA GPU is available and is a Tesla T4
python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA is not available! Connect a T4 GPU runtime.'
device_name = torch.cuda.get_device_name(0)
print(f'CUDA Device: {device_name}')
print(f'VRAM Total: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB')
"

# 2. Build and install t4_kernels extension if needed
if [ -d "$REPO_DIR/src" ]; then
    echo "--> Building & Installing t4_kernels PyTorch extension..."
    cd "$REPO_DIR/src"
    pip install -e . --quiet
    cd "$REPO_DIR"
fi

# 3. Run Pipeline Unit Tests
echo "--> Running Pipeline & Metric Unit Tests..."
python3 -m pytest tests/test_math_sft_pipeline.py -v

# 4. Execute Full Cold-Start SFT Training on Tesla T4
echo "--> Launching Full Cold-Start LoRA SFT on Tesla T4..."
python3 benchmarks/train_math_sft.py \
    --model_name_or_path "Qwen/Qwen2.5-Math-1.5B" \
    --data_path "data/chalk_seeds_500.jsonl" \
    --output_dir "results/chalk_math_1.5b_sft" \
    --epochs 3 \
    --batch_size 2 \
    --grad_accum 8 \
    --lr 2e-4 \
    --max_length 2048 \
    --lora_r 32 \
    --lora_alpha 64 \
    --logging_steps 5

# 5. Execute External Benchmark Evaluation on Held-Out Problems
echo "--> Running External Benchmark Evaluation (Base vs SFT with CP-Hybrid Kernels)..."
python3 benchmarks/eval_math_benchmark.py \
    --model_name_or_path "Qwen/Qwen2.5-Math-1.5B" \
    --adapter_path "results/chalk_math_1.5b_sft/lora_adapter" \
    --benchmark_path "data/external_math_eval.json" \
    --output_path "results/external_eval_results.json" \
    --use_kernels \
    --device "cuda"

echo "=========================================================================="
echo "  [ITEM 4 COMPLETE] SFT & Evaluation Finished Successfully on Tesla T4"
echo "=========================================================================="
