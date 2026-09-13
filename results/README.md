# Experimental Results, Hardware Telemetry & Nsight Traces

This directory contains empirical benchmark logs, Nsight Compute microarchitectural execution traces, JSON metrics, and hardware telemetry recorded on physical **NVIDIA Tesla T4** hardware (TU104, sm_75, 70W TDP).

---

## 1. Directory Structure

```
results/
├── benchmarks/         # Structured JSON evaluation and benchmark metric records
├── logs/               # Raw execution logs from physical Google Colab T4 hardware runs
├── traces/             # Binary NVIDIA Nsight Compute (.ncu-rep) profiling traces
├── t4_colab_runtime_logs/ # Timestamped stage-by-stage verification logs
├── grpo_baseline/      # FP16 GRPO training metrics on GSM8K
├── grpo_int4/          # INT4 GRPO training metrics on GSM8K
├── grpo_profile/       # Detailed timing breakdown of GRPO training steps
└── 2026-07-30_165546/  # Baseline hardware telemetry and kernel artifacts
```

---

## 2. Flagship Hardware Traces (`results/traces/`)

| Trace Artifact | Kernel | Size | Physical Hardware Findings |
|---|---|---|---|
| [`traces/w4a16_gemv_t4.ncu-rep`](traces/w4a16_gemv_t4.ncu-rep) | Fused W4A16 GEMV ($M=1$) | 8.1 MB | **1590 MHz locked boost clock**, **94.2% DRAM throughput saturation**, 0 SMEM bank conflicts via $\mathbb{F}_2^5$ XOR swizzling, 1.8% warp barrier stall cycles. |
| [`traces/h6_fused_adamw_t4.ncu-rep`](traces/h6_fused_adamw_t4.ncu-rep) | Fused Backward GEMM + AdamW | 7.0 MB | **1590 MHz locked boost clock**, **88.7% DRAM saturation**, **91.2% compute throughput**, 21.43% DRAM traffic reduction (28→22 B/param). |

---

## 3. Benchmark Summaries (`results/benchmarks/`)

| Metric File | Milestone / Focus | Key Empirical Finding |
|---|---|---|
| [`benchmarks/external_eval_results.json`](benchmarks/external_eval_results.json) | Milestone 4: Math SFT vs Base | Baby-Chalk achieves **60.0% vs 10.0% grounding pass** (6x boost) on held-out questions; solves 4/16 AIME/AMC problems; learns 5-tag schema. |
| [`benchmarks/base_eval_results.json`](benchmarks/base_eval_results.json) | Base `Qwen2.5-Math-1.5B` | 0.0% 5-tag adherence; severe degradation (10% pass) under strict system prompt instructions. |
| [`benchmarks/t4_speculative_benchmark_report.json`](benchmarks/t4_speculative_benchmark_report.json) | Unified Speculative Engine | **1.48x net wall-clock speedup** (39.34 vs 26.59 tok/s); $K=3$ lookahead; 0 parameter overhead. |
| [`benchmarks/speculative_7b_benchmark.json`](benchmarks/speculative_7b_benchmark.json) | Qwen2.5-7B + INT4 0.5B Draft | **68.3% acceptance rate** ($K=2$), **2.07 accepted tokens/step** on 16 GB T4. |
| [`benchmarks/quarantined_benchmark.json`](benchmarks/quarantined_benchmark.json) | Tiny Honest Model RL | **8x jump in honest abstention** (6.7% $\to$ 53.3%) on impossible premise traps; 0% false abstention. |
| [`benchmarks/m1_kv_cache_benchmark.json`](benchmarks/m1_kv_cache_benchmark.json) | Static Pre-Allocated KV-Cache | $O(1)$ rollback in 0.0003 ms ($14,388\times$ faster than PyTorch dynamic slice reallocation). |

---

## 4. Hardware Verification Logs (`results/logs/`)

- [`logs/t4_colab_verified_run_20260816.log`](logs/t4_colab_verified_run_20260816.log): Full 5-stage verification passing 100% on live physical T4 silicon.
- [`logs/t4_h6_empirical_run_20260813.log`](logs/t4_h6_empirical_run_20260813.log): Fused AdamW hardware comparison verifying 1.94x speedup over PyTorch baseline.
- [`logs/training_kernels_t4_benchmark.log`](logs/training_kernels_t4_benchmark.log): Microbenchmarks for fused SFT LoRA backward kernels and activations.
- [`logs/serving_engine_t4_benchmark_milestone_a.log`](logs/serving_engine_t4_benchmark_milestone_a.log): Token serving engine latency across batch sizes.
