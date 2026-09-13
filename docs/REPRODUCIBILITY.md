# Reproducibility Guide: Physical Tesla T4 Execution

This guide provides step-by-step instructions to reproduce all empirical benchmarks, hardware traces, low-precision RL training runs, and reasoning SFT evaluations on physical **NVIDIA Tesla T4** hardware.

---

## 1. Google Colab Environment Setup

Because local development environments may lack physical Tesla T4 silicon, Google Colab provides an accessible platform with identical hardware specifications:

```bash
# 1. Provision a free or paid Colab instance with Tesla T4 GPU
colab new --gpu T4 -s t4-bench

# 2. Package and upload the repository
tar --exclude='.git' --exclude='*.pyc' -czf /tmp/t4-cuda-repo.tar.gz .
colab upload -s t4-bench /tmp/t4-cuda-repo.tar.gz /content/t4-cuda-repo.tar.gz

# 3. Extract and install the custom CUDA extension
colab exec -s t4-bench "mkdir -p /content/t4-cuda && tar -xzf /content/t4-cuda-repo.tar.gz -C /content/t4-cuda && pip install -e /content/t4-cuda/src"
```

---

## 2. Automated Master Verification Pipeline

To execute the entire 5-stage verification suite across all kernels, serving engines, and reasoning pipelines:

```bash
# Execute master verification pipeline
colab exec -s t4-bench "bash /content/t4-cuda/verify_colab.sh"
```

The master script executes:
1. Bit-exact Known Answer Tests (KAT) for LOP3 signed/unsigned INT4, INT3, and FP8 dequantization.
2. Microarchitectural roofline modeling and SASS instruction count audits.
3. Standalone `%clock64` timer microbenchmarks for cache and memory latencies.
4. PyTorch CUDA C++ extension build and live differential testing.
5. Fused backward GEMM + AdamW comparative benchmark (verifying 1.94x speedup and 21.43% DRAM reduction).
6. Dual-path W4A16 WMMA vs cuBLAS sweep across $M \in \{1, 4, 8, 16, 32, 64\}$.
7. Speculative decoding smoke tests.

---

## 3. Reproducing Low-Precision RL Training (GRPO)

To reproduce the 30-step GRPO RL training of the Tiny Honest Model:

```bash
colab exec -s t4-bench "python3 /content/t4-cuda/benchmarks/train_honest_model_grpo.py --steps 30 --dataset /content/t4-cuda/data/honesty_train.json"
```

To run the quarantined zero-contamination evaluation:

```bash
colab exec -s t4-bench "python3 /content/t4-cuda/benchmarks/eval_quarantined.py"
```

---

## 4. Reproducing Cold-Start SFT & AIME/AMC 12 Evaluation

To reproduce the 111-step Cold-Start SFT on `Qwen2.5-Math-1.5B`:

```bash
colab exec -s t4-bench "bash /content/t4-cuda/scripts/run_t4_sft.sh"
```

To run the out-of-dataset competition evaluation:

```bash
colab exec -s t4-bench "python3 /content/t4-cuda/benchmarks/eval_math_benchmark.py --adapter /content/t4-cuda/results/chalk_math_1.5b_sft/lora_adapter"
```

---

## 5. Paper Verification & Building

To validate paper structure, citations, and empirical numbers against the raw result JSON files:

```bash
python3 paper/compile_paper.py
```
