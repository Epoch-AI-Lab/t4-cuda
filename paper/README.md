# Systems Paper: Sub-Byte Arithmetic & Low-Precision Reasoning Systems for Tesla T4

This directory contains the publication manuscript, BibTeX bibliography, empirical research compendium, and automated validation harness for our paper:

> **"Sub-Byte Arithmetic, Asynchronous Pipeline Emulation, and Low-Precision Reasoning Systems for Legacy Turing GPUs"**

---

## 1. Directory Layout

```
paper/
├── t4_cuda_paper.tex   # Full conference manuscript source (LaTeX)
├── references.bib      # Complete BibTeX bibliography (12 core citations)
├── COMPENDIUM.md       # Empirical evidence compendium & theorem proofs
├── compile_paper.py    # Automated LaTeX validation and build runner
└── README.md           # This document
```

---

## 2. Validation & Build Instructions

To validate document structure, citations, and numerical consistency against empirical results:

```bash
python3 paper/compile_paper.py
```

To compile the LaTeX source into a publication-grade PDF using TeX Live:

```bash
cd paper
pdflatex t4_cuda_paper.tex
bibtex t4_cuda_paper
pdflatex t4_cuda_paper.tex
pdflatex t4_cuda_paper.tex
```

---

## 3. Core Contributions Covered in the Paper

1. **Signed Sub-Byte LOP3 Bit-Inversion (Theorems 1 & 4)**: Single-cycle signed INT3/INT4 and FP8 $E4M3$ dequantization via LUT `0x6A` saturating GDDR6 memory bandwidth at 332.7 GB/s.
2. **Conflict-Free Shared Memory Swizzling (Theorem 2)**: 128-bit XOR swizzling over $\mathbb{F}_2^5$ with zero bank conflicts.
3. **Software Warp Specialization**: Producer/consumer split-K architecture reducing warp fetch stalls by 94.2% on Turing hardware without hardware asynchronous copy units.
4. **Power-Aware Occupancy Pacing (Lemma 5)**: 25% active warp capping locking maximum 1590 MHz boost clock under a 70W TDP ceiling.
5. **In-Register Optimizer Fusion (Lemma 3)**: Fused backward GEMM + AdamW optimizer kernel reducing DRAM traffic by 21.43% with a 1.94x speedup over PyTorch.
6. **Low-Precision RL Rollout (GRPO)**: CP-Hybrid kernel dispatch eliminating reasoning drift on GSM8K and training an honest model with 8x higher abstention on impossible traps.
7. **Unified Speculative Serving Engine**: Pre-allocated static KV-caches with $O(1)$ rollback and zero-weight prompt lookup, delivering a 1.48x net wall-clock speedup.
8. **Cold-Start SFT on Qwen2.5-Math-1.5B**: 5-tag scientific discovery schema delivering a 6x grounding improvement (60.0% vs 10.0%) on held-out AIME and AMC 12 problems.
