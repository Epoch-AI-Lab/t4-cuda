# Model Thing: Mathematical Foundation Models & Pure Math Research Dossier

This directory houses the comprehensive research investigation into next-generation open-weights mathematical foundation models, training methodology pipelines, radical reasoning architectures, and epistemological insights from pure mathematics literature.

---

## Deliverables Index

| Document | Size | Description |
| :--- | :--- | :--- |
| **[`RESEARCH_DOSSIER.md`](./RESEARCH_DOSSIER.md)** | 28 KB | **Master Executive Synthesis & Architecture Blueprint.** Integrated 5-phase production roadmap for training model Chock, comparative matrices, and system flow diagrams. |
| **[`AUDIT_LEDGER.md`](./AUDIT_LEDGER.md)** | 37 KB | **Independent Base Model Audit.** Bypasses contaminated leaderboard benchmarks (GSM8K/MATH). Evaluates 10 open-weights models (0.5B to 671B) across tokenization efficiency on LaTeX, pre-training corpus mix, attention mechanics, and stability. |
| **[`METHODOLOGY_TRACK_A.md`](./METHODOLOGY_TRACK_A.md)** | 65 KB | **Standard Training Best Practices.** Production-proven recipes across 12 open-source repos: 2-tier decontamination, backward synthetic data generation, rejection sampling fine-tuning (RFT), process reward models (PRMs), RLVR (GRPO with unbiased KL), and formal Lean 4 workflows. |
| **[`METHODOLOGY_TRACK_B.md`](./METHODOLOGY_TRACK_B.md)** | 152 KB | **Radical Training Innovations.** 24 bold, uncapped hypothesis cards with concrete mechanisms, risk profiles, and Gate 1 falsification tests (continuous proof manifolds, SRAM Gröbner basis kernels, Lakatosian minimax counterexample co-evolution, dynamic entropy annealing, and automated lemma distillation). |
| **[`PURE_MATH_LITERATURE_SYNTHESIS.md`](./PURE_MATH_LITERATURE_SYNTHESIS.md)** | 72 KB | **Pure Mathematics Discovery vs. Presentation.** Epistemological survey of Poincaré, Hadamard, Pólya, Lakatos, Grothendieck, Thurston, and Tao. Analyzes why autoregressive models fail when forcing chaotic, heuristic real-time exploration into a single linear Bourbaki-style token stream. |

---

## Quick Findings Summary

1. **Base Model Selection:** Single-digit tokenization is mandatory to stop arithmetic carry collapse (Qwen2.5-Math and DeepSeek-Math). Llama-3.1 merges up to 3 digits per token, causing high failure rates on multi-digit calculations. Qwen2.5-Math-7B is the recommended 7B base for RLVR; Qwen2.5-Math-72B and DeepSeek-V3 are the optimal teacher models for synthetic distillation.
2. **Standard Training (Track A):** PRMs suffer from length decay and filler reward hacking. RLVR (via GRPO) using rule-based Python/SymPy/Lean verifiers eliminates the critic network to save 50% VRAM while providing rock-solid reward signals.
3. **Radical Training (Track B):** 24 testable hypotheses spanning continuous proof manifolds, intra-layer CAS kernels, category-theoretic attention, and adversarial counterexample generation.
4. **Pure Mathematics (Track C):** Human mathematical discovery is messy, iterative, and adversarial (Pólya / Lakatos), whereas published proofs are sanitized and deductive (Bourbaki). Models need tagged reasoning scratchpads (`<explore>`, `<conjecture>`, `<counterexample_test>`, `<refutation_analysis>`, `<lemma_isolate>`, `<formal_synthesis>`) to decouple chaotic scratchpad search from disciplined final proof compilation.
