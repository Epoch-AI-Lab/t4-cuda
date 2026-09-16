# `t4-cuda`: Turing CC 7.5 Extreme CUDA Kernels & Low-Precision Reasoning Systems

[![CUDA](https://img.shields.io/badge/CUDA-11.8%20%7C%2012.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Hardware](https://img.shields.io/badge/Hardware-NVIDIA%20Tesla%20T4%20(TU104)-76B900.svg)](https://www.nvidia.com/en-us/data-center/tesla-t4/)
[![Compute Capability](https://img.shields.io/badge/Compute%20Capability-7.5-blue.svg)](https://developer.nvidia.com/cuda-gpus)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-LaTeX%20Ready-brightgreen.svg)](paper/t4_cuda_paper.tex)

An extreme, microarchitecturally optimized CUDA C++ and PTX assembly kernel suite, speculative serving engine, and low-precision reinforcement learning training system custom-tailored for **NVIDIA Tesla T4 GPUs** (Turing CC 7.5, TU104 die, 40 SMs, 320 Tensor Cores, 70W TDP).

> **Hardware Verification State**: **100% VERIFIED ON PHYSICAL TESLA T4 SILICON** (Driver 580.82.07, CUDA 13.0, PyTorch 2.11.0+cu128). All 4 major roadmap milestones completed, tested, and audited under strict claims hygiene. Complete paper draft in [`paper/t4_cuda_paper.tex`](paper/t4_cuda_paper.tex) and research compendium in [`paper/COMPENDIUM.md`](paper/COMPENDIUM.md).

---

## Executive Summary & Architectural Motivation

Millions of Tesla T4 GPUs power enterprise and cloud inference platforms (Google Colab, AWS `g4dn`, GCP, Azure). However, modern LLM inference and training frameworks (e.g. vLLM, FlashInfer, CUTLASS 3.x) rely on Ampere/Hopper hardware primitives (`cp.async`, Tensor Memory Accelerators, FP8/INT4 MMA shapes) that **do not physically exist on Turing CC 7.5**. When executed on T4, these frameworks either fail or degrade to un-optimized PyTorch fallbacks.

Furthermore, on passively cooled 70W T4 GPUs, unconstrained kernel launches trigger hardware power capping (**NVIDIA Power Management / NVPM**), throttling core clocks from **1590 MHz down to ~950–1193 MHz** (a 25–38% loss in throughput).

**`t4-cuda`** overcomes these barriers via:
1. **Single-cycle LOP3 bit manipulation** for signed sub-byte INT4/INT3 dequantization (332.7 GB/s saturation).
2. **Software Warp Specialization** creating producer/consumer rings without hardware async copy (94.2% stall reduction).
3. **Power-aware 25% occupancy pacing** locking peak 1590 MHz boost clocks at flat 50.36W power.
4. **In-register backward GEMM + AdamW optimizer fusion** (21.43% DRAM traffic reduction, 1.94x speedup).
5. **Component-and-Phase Hybrid (CP-Hybrid) RL rollouts** eliminating reasoning degradation in GRPOTrainer.
6. **Unified Speculative Serving Engine** delivering a **1.48x net wall-clock speedup** on physical silicon.
7. **Cold-Start SFT on Qwen2.5-Math-1.5B** delivering a **6x grounding improvement** on held-out competition tasks.

---

## Hardware-Verified Milestones (Physical Tesla T4 Silicon)

| Milestone / Kernel | Target & Mechanism | Physical Silicon Verification on Tesla T4 |
| :--- | :--- | :--- |
| **H1 (SMEM Swizzling)** | 128-bit XOR swizzling over $\mathbb{F}_2^5$ | **0 SMEM Bank Conflicts**; 22.2 cycles/access vs 64.3 cycles in stride-32. |
| **H4 (Signed INT4 LOP3)** | Two's complement inversion via LUT `0x6A` | **Bit-exact (0.0 diff)**; KAT matched across 23/23 vectors. |
| **H5 (Thermal Occupancy)** | Warp capping at 25% (8 warps/SM) | Capped power **50.36W** ($<70$W), **1590 MHz boost clock locked 100%**. |
| **H6 (Fused BWD GEMM + AdamW)** | In-register AdamW update inside GEMM epilogue | **1.94x speedup** (10.109 ms vs 19.573 ms); **21.43% DRAM traffic cut** (28→22 B/param). |
| **H7 (Signed INT3 LOP3)** | Sub-byte bitplane extraction via LOP3 | **332.7 GB/s** effective memory bandwidth saturation. |
| **H9 (FP8 E4M3 Emulation)** | Bitwise ADD+OR integer rebias on FP16 Tensor Cores | **254/254 valid byte states bit-exact** (0.0 diff). |
| **W4A16 GEMV (Attention)** | Fused INT4 per-group GEMV ($1 \times 896 \times 896$) | **2.06x speedup** (9.2 $\mu$s vs 18.9 $\mu$s cuBLAS FP16). |
| **W4A16 GEMV (MLP)** | Fused INT4 per-group GEMV ($1 \times 896 \times 4864$) | **1.67x speedup** (26.9 $\mu$s vs 45.0 $\mu$s cuBLAS FP16). |
| **Milestone 1 (W4A16 Per-Group)** | Symmetric group=128 in $K$, exact FP32 dequant | Rel err **0.024%**; **279.1 tok/s** (1.20x speedup), **56% VRAM cut** (2.84 $\to$ 1.24 GB). |
| **Milestone 2 (GRPO INT4 Rollouts)** | INT4 rollouts wired into TRL GRPOTrainer | 30 steps completed with 0 OOMs; revealed RTN reasoning drift; CP-Hybrid fix. |
| **Milestone 3 (Tiny Honest Model)** | 30-step GRPO under CP-Hybrid stack | **8x jump in honest abstention** (6.7% $\to$ 53.3%) on unanswerable traps; 0% false abstention. |
| **Milestone 4 (Cold-Start Math SFT)** | SFT on `Qwen2.5-Math-1.5B` (5-tag schema) | **6x grounding boost** (60% vs 10% pass); solved 4/16 held-out AIME/AMC problems; 0 OOMs. |
| **Unified Speculative Engine** | Pre-allocated static KV-cache + PromptLookup draft | **1.48x net wall-clock speedup** (39.34 vs 26.59 tok/s) on physical T4. |

---

## Clean Codebase Layout

```
t4-cuda/
├── paper/                             # Academic paper & publication deliverables
│   ├── t4_cuda_paper.tex              # Complete publication-grade LaTeX systems paper
│   ├── references.bib                 # Comprehensive BibTeX bibliography
│   ├── COMPENDIUM.md                  # Unified research compendium & empirical evidence
│   ├── compile_paper.py               # Automated LaTeX validation and build script
│   └── README.md                      # Paper overview & build instructions
├── docs/                              # Systems architecture & engineering guides
│   ├── ARCHITECTURE.md                # 4-pillar systems architecture breakdown
│   ├── HARDWARE_SPECIFICATIONS.md     # Turing TU104 microarchitectural constraints
│   ├── REPRODUCIBILITY.md             # Physical T4 replication guide (Colab / On-Prem)
│   ├── BABY_CHALK_SFT_RESULTS.md      # Milestone 4 Cold-Start SFT research report
│   └── README.md                      # Documentation index
├── benchmarks/                        # Performance benchmarks, training & eval harnesses
│   ├── bench_w4a16_wmma_vs_cublas.py  # Dual-path W4A16 GEMM vs cuBLAS sweep
│   ├── bench_gen_fp16_vs_int4.py      # Autoregressive generation throughput & VRAM
│   ├── benchmark_speculative_fused_serving.py # Speculative serving benchmark
│   ├── benchmark_m1_kv_cache.py       # StaticKVCache micro-benchmark (O(1) rollback)
│   ├── train_honest_model_grpo.py     # Milestone 3 GRPO RL training runner
│   ├── train_math_sft.py              # Milestone 4 Cold-Start SFT training runner
│   ├── eval_math_benchmark.py         # 5-tag schema, SymPy exact match, AIME/AMC eval
│   ├── eval_quarantined.py            # Quarantined honesty evaluation runner
│   ├── debug/                         # Shape and kernel debugging scripts
│   └── README.md                      # Benchmark suite index & instructions
├── data/                              # Training, fine-tuning & quarantined eval datasets
│   ├── chalk_seeds_500.jsonl          # 605 certified reasoning traces (445k tokens)
│   ├── external_math_eval.json        # 26 held-out AIME/AMC 12 & grounding checks
│   ├── honesty_train.json             # Balanced 50/50 honesty dataset (300 records)
│   ├── quarantined_eval.json          # 150-question quarantined zero-contamination eval
│   ├── build_external_math_benchmark.py # Dataset compilation scripts
│   └── README.md                      # Dataset manifest & provenance documentation
├── notebooks/                         # Interactive Jupyter notebooks for Google Colab
│   ├── colab_run_sft.ipynb            # End-to-end Cold-Start SFT on Tesla T4
│   ├── grpo_baseline.ipynb            # Interactive GRPO RL training notebook
│   └── README.md                      # Notebook catalog & instructions
├── results/                           # Empirical logs, metrics & Nsight traces
│   ├── benchmarks/                    # JSON benchmark and evaluation metrics
│   ├── logs/                          # Physical hardware execution logs
│   ├── traces/                        # Binary Nsight Compute (.ncu-rep) traces
│   └── README.md                      # Results catalog & telemetry summary
├── src/                               # Core CUDA/PTX kernels & Python engine modules
│   ├── kernels/                       # Production CUDA C++ / PTX kernel headers and sources
│   │   ├── lop3_dequant.cu            # Single-cycle LOP3 INT4/INT3/FP8 dequant
│   │   ├── fused_w4a16_gemm.cu        # Fused per-group INT4 GEMV / WMMA
│   │   ├── fused_backward_adamw.cu    # Fused BWD GEMM + inline AdamW
│   │   └── h17_mega_kernel.cu         # Software warp-specialized mega-kernel
│   ├── static_kv_cache.py             # Pre-allocated O(1) static KV-cache
│   ├── unified_draft_engine.py        # Zero-weight prompt lookup draft engine
│   ├── unified_speculative_engine.py  # Unified speculative serving engine
│   ├── calibrated_rollout.py          # CP-Hybrid selective precision scope
│   ├── hybrid_linear.py               # Dynamic dispatch linear layer
│   ├── bindings.cpp                   # PyTorch C++/CUDA extension bindings
│   └── setup.py                       # Extension build configuration
├── tests/                             # Verification test suites & CI test runner
│   ├── run_all_cuda_tests.py          # Master test runner (strictly enforces gating)
│   ├── test_dequant_correctness.py    # Bit-exact KAT & fuzzing for LOP3
│   ├── test_fused_gemm_correctness.py # Differential testing for fused W4A16 GEMM
│   ├── test_h6_fused_backward_adamw.py# Fused AdamW comparative correctness
│   ├── test_m1_static_kv_cache.py     # StaticKVCache correctness & dynamic parity
│   └── test_math_sft_pipeline.py      # SFT loss masking & 5-tag schema verification
├── workflows/                         # Automated research and execution workflows
├── scripts/                           # Auxiliary scripts and audits
├── verify_colab.sh                    # Complete automated verification pipeline script
├── research-state.yaml                # Central hypothesis & hardware state manifest
├── CLAIMS_HYGIENE.md                  # Strict measurement tagging & evaluation rules
├── TODO.md                            # Moonshot roadmap & status tracker
└── NEXT_STEPS.md                      # Next research priorities
```

---

## Quickstart: Verification on Tesla T4

To build the extension and run the full verification pipeline on a physical Tesla T4 GPU:

```bash
# 1. Build and install PyTorch CUDA extension
pip install -e src/

# 2. Run master verification pipeline (enforces Claims Hygiene Rule 3)
python3 tests/run_all_cuda_tests.py

# 3. Or run the complete automated Colab bash pipeline
bash verify_colab.sh
```

To validate the paper manuscript and check empirical consistency:

```bash
python3 paper/compile_paper.py
```

---

## Claims Hygiene & Research Rigor

In accordance with [`CLAIMS_HYGIENE.md`](CLAIMS_HYGIENE.md):
- Every reported number carries inline provenance (`[MEASURED]`, `[ESTIMATED]`, `[ARITHMETIC]`).
- Baseline comparisons are made against compiled baselines (`torch.compile` / cuBLAS), not unoptimized eager code.
- Performance benchmarks are strictly gated behind 100% pass rates on correctness test suites.
- All hardware measurements document SM boost clocks and thermal throttle status words.

---

## License

This project is licensed under the **Apache License 2.0**. See [LICENSE](LICENSE) for details.
