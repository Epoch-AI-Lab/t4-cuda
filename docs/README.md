# Documentation Index

This directory contains architectural specifications, microarchitectural hardware analysis, reproducibility guides, and milestone empirical results for the **`t4-cuda`** project.

---

## Documentation Manifest

| Document | Description | Focus Area |
|---|---|---|
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Full architectural overview of the 4 core systems pillars (CUDA kernels, low-precision RL, speculative serving, and mathematical reasoning SFT). | Systems Architecture |
| [`HARDWARE_SPECIFICATIONS.md`](HARDWARE_SPECIFICATIONS.md) | Turing TU104 / sm_75 hardware constraints, memory bus latencies, 70W TDP power scaling, and PTX instruction envelopes. | Hardware Engineering |
| [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) | Complete step-by-step instructions for reproducing all physical T4 silicon benchmarks and training passes on Google Colab or on-prem servers. | Verification & Reproduction |
| [`BABY_CHALK_SFT_RESULTS.md`](BABY_CHALK_SFT_RESULTS.md) | Detailed empirical research report on Milestone 4: Cold-Start SFT of `Qwen2.5-Math-1.5B` and held-out AIME/AMC 12 benchmarks. | Research Report |
