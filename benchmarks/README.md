# Benchmark, Training & Evaluation Suite

This directory contains the benchmarking harnesses, reinforcement learning / SFT training scripts, and quarantined evaluation suites executed on physical **NVIDIA Tesla T4** hardware.

---

## 1. Core Benchmarks

| Script | Focus & Hardware Metric | Typical Hardware Target |
|---|---|---|
| [`bench_w4a16_wmma_vs_cublas.py`](bench_w4a16_wmma_vs_cublas.py) | Dual-path W4A16 GEMM vs cuBLAS FP16 across $M \in \{1, 4, 8, 16, 32, 64\}$. Identifies the CP-Hybrid dispatch boundary ($M \le 4$). | Tesla T4 (sm_75) |
| [`bench_gen_fp16_vs_int4.py`](bench_gen_fp16_vs_int4.py) | Generation throughput and VRAM comparison for Qwen2.5-0.5B (279.1 tok/s vs 232.5 tok/s, 56% VRAM cut). | Tesla T4 (sm_75) |
| [`benchmark_speculative_decoding.py`](benchmark_speculative_decoding.py) | Multi-model speculative decoding (0.5B INT4 draft + 1.5B/7B target) measuring acceptance rate $\alpha$ and token yield. | Tesla T4 (sm_75) |
| [`benchmark_speculative_fused_serving.py`](benchmark_speculative_fused_serving.py) | Unified serving engine benchmark measuring net wall-clock tokens/sec and speedup over autoregressive baseline. | Tesla T4 (sm_75) |
| [`benchmark_m1_kv_cache.py`](benchmark_m1_kv_cache.py) | Static pre-allocated KV-cache latency vs dynamic PyTorch slicing ($14,388\times$ faster pointer rollback). | Tesla T4 (sm_75) |
| [`benchmark_grpo_t4.py`](benchmark_grpo_t4.py) | Standard FP16 baseline for 30-step GRPO RL on GSM8K (0.2917 mean reward). | Tesla T4 (sm_75) |
| [`benchmark_grpo_int4.py`](benchmark_grpo_int4.py) | Full INT4 rollout GRPO run identifying the reward integrity drift (-0.2667) under temperature exploration. | Tesla T4 (sm_75) |
| [`benchmark_grpo_calibrated_reasoning.py`](benchmark_grpo_calibrated_reasoning.py) | CP-Hybrid precision rollout verifying preservation of mathematical reasoning and reward integrity. | Tesla T4 (sm_75) |
| [`benchmark_indepth_honesty.py`](benchmark_indepth_honesty.py) | In-depth evaluation of abstention vs hallucination on impossible premise traps. | Tesla T4 (sm_75) |
| [`test_3way_benchmark.py`](test_3way_benchmark.py) | 3-way latency and memory comparison (PyTorch FP16 vs BitsAndBytes NF4 vs Fused INT4). | Tesla T4 (sm_75) |
| [`run_nsight_profiling.sh`](run_nsight_profiling.sh) | Automated Nsight Compute (`ncu`) script capturing instruction mix, warp stall reasons, and DRAM saturation. | Tesla T4 (sm_75) |

---

## 2. Training Pipelines

| Script | Description | Key Output Artifact |
|---|---|---|
| [`train_honest_model_grpo.py`](train_honest_model_grpo.py) | 30-step GRPO RL training of Qwen2.5-0.5B under CP-Hybrid kernels on a balanced honesty dataset. | Checkpoint in `results/` |
| [`train_math_sft.py`](train_math_sft.py) | Cold-start SFT of `Qwen2.5-Math-1.5B` on 605 certified reasoning traces (`data/chalk_seeds_500.jsonl`) with strict 5-tag loss masking. | `results/chalk_math_1.5b_sft/lora_adapter` |

---

## 3. Evaluation Pipelines

| Script | Description | Evaluation Target |
|---|---|---|
| [`eval_math_benchmark.py`](eval_math_benchmark.py) | Held-out AMC 12 and AIME evaluation measuring 5-tag schema adherence, SymPy exact match, and grounding. | `results/external_eval_results.json` |
| [`eval_honesty.py`](eval_honesty.py) | Evaluation runner measuring honest abstention vs hallucination rate. | `results/quarantined_benchmark.json` |
| [`eval_quarantined.py`](eval_quarantined.py) | Standalone runner for the quarantined zero-contamination trap dataset. | Terminal report |

---

## 4. Debugging & Development Tools

The `debug/` subdirectory houses diagnostic scripts for kernel shapes, layer dimensions, and weight unpacking:
- [`debug/debug_qproj.py`](debug/debug_qproj.py): Attention query projection shapes and weight packing verification.
- [`debug/debug_kernel_shape.py`](debug/debug_kernel_shape.py): General GEMM/GEMV shape verification.
- [`debug/debug_layers.py`](debug/debug_layers.py): Layer-by-layer inspection of Qwen2.5 model architectures.
