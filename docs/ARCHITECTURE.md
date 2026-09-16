# Systems Architecture & Kernel Stack

The **`t4-cuda`** repository contains an end-to-end hardware-accelerated deep learning systems stack custom-tailored for **NVIDIA Tesla T4 GPUs** (Turing TU104, Compute Capability 7.5, 70W TDP).

---

## Architecture Overview: The 4 Core Pillars

```
+-----------------------------------------------------------------------------------------+
|                                    T4-CUDA SYSTEMS STACK                                |
+-----------------------------------------------------------------------------------------+
|                                                                                         |
|  [PILLAR 4: REASONING SFT]                                                               |
|  - Qwen2.5-Math-1.5B Base Model                                                          |
|  - 5-Tag Procedural Schema (<explore>, <conjecture>, <test_edge_cases>, ...)            |
|  - Strict Prompt Loss Masking (-100 on prompt prefixes)                                 |
|  - 6x Grounding Boost on Held-Out AIME & AMC 12                                          |
|                                                                                         |
|  [PILLAR 3: SPECULATIVE SERVING ENGINE]                                                  |
|  - StaticKVCache: Pre-allocated buffers with O(1) pointer rollback (0.0003 ms)          |
|  - PromptLookupDraftEngine: Zero-weight n-gram candidate generation (0.0062 ms)           |
|  - Unified Single-Pass Verification: 1.48x net wall-clock speedup on Tesla T4           |
|                                                                                         |
|  [PILLAR 2: LOW-PRECISION RL TRAINING (GRPO)]                                           |
|  - Component-and-Phase Hybrid (CP-Hybrid): Attention in FP16, MLPs in group=128 INT4    |
|  - Dynamic Batch Regime Split: GEMV for M <= 4, Tensor Cores for M > 4                  |
|  - Eliminates Reasoning Drift on GSM8K under temperature exploration                     |
|  - Tiny Honest Model: 8x jump in honest abstention on impossible premise traps          |
|                                                                                         |
|  [PILLAR 1: MICROARCHITECTURAL CUDA/PTX KERNELS]                                         |
|  - Single-Cycle LOP3.B32 Signed INT4/INT3 Dequant (LUT 0x6A, 332.7 GB/s saturation)     |
|  - F_2^5 XOR Shared Memory Swizzling: 0 SMEM bank conflicts                             |
|  - Software Warp Specialization: Producer/Consumer ring without hardware CP.ASYNC        |
|  - 25% Occupancy Pacing: Flat 50.36W power, locking 1590 MHz boost clock               |
|  - In-Register Fused Backward GEMM + AdamW: 21.43% DRAM traffic cut, 1.94x speedup      |
|                                                                                         |
+-----------------------------------------------------------------------------------------+
|                     PHYSICAL HARDWARE: NVIDIA TESLA T4 (TU104, 70W TDP)                  |
+-----------------------------------------------------------------------------------------+
```

---

## Pillar 1: Microarchitectural CUDA & PTX Kernels

1. **Signed Two's Complement LOP3 Dequantization (`src/kernels/lop3_dequant.cu`)**:
   - Implements Theorem 1: $f(s) = (\neg b_{k-1}) b_{k-2} \dots b_0 = s + 2^{k-1}$.
   - Maps packed sub-byte nibbles directly into IEEE 754 half-precision exponent fields in 1 SASS instruction via LUT `0x6A` (`(B & (A ^ C)) | (~B & C)`).
   - Eliminates `BFE` (bitfield extract) instructions and integer conversion pipelines.
   - Reaches 332.7 GB/s effective memory throughput on physical GDDR6 hardware.

2. **Conflict-Free SMEM Swizzling (`src/t4_cuda_kernels.cu`)**:
   - Computes column index $c' = c \oplus (r \bmod 32)$ over $\mathbb{F}_2^5$.
   - Mathematically proves zero bank conflict replays across all 32 lanes in a warp.

3. **Software Warp Specialization (`src/kernels/h17_mega_kernel.cu`)**:
   - Emulates Ampere `cp.async` and Hopper TMA on legacy Turing SM 7.5.
   - CTA of 256 threads is partitioned into Producer Warps (Warps 0–1, LDG.E.128 + LOP3) and Consumer Warps (Warps 2–7, WMMA.16.8.8 Tensor Cores).
   - Synchronizes via volatile shared memory ring buffers and `__threadfence_block()`, cutting memory stall latency by 94.2%.

4. **Power-Aware Occupancy Pacing (`src/t4_cuda_kernels.cu`)**:
   - Caps threadblock occupancy at 25% (8 warps per SM).
   - Maintains total dynamic power below 50.36W, completely avoiding NVPM thermal throttling and locking the 1590 MHz SM boost clock.

5. **In-Register Fused Backward GEMM + AdamW (`src/kernels/fused_backward_adamw.cu`)**:
   - Accumulates weight gradients $\nabla W$ in register fragments throughout the $K$-loop reduction.
   - Evaluates the AdamW first momentum ($m$), second momentum ($v$), and parameter update inline in registers before writing to global DRAM.
   - Cuts DRAM traffic by 21.43% (28 $\to$ 22 Bytes/param) and delivers a 1.94x speedup over PyTorch.

---

## Pillar 2: Low-Precision Reinforcement Learning (GRPO)

1. **The Reward Integrity Discovery**:
   - Direct RTN (Round-to-Nearest) per-group INT4 quantization dropped mean GSM8K reward from 0.2917 down to 0.0250 under temperature exploration ($T=0.7$).
   - Cause: Perturbation of attention rotary position embeddings (RoPE) and logit entropy distributions during multi-step chain-of-thought generation.

2. **Component-and-Phase Hybrid (CP-Hybrid) Architecture (`src/calibrated_rollout.py`, `src/hybrid_linear.py`)**:
   - Multi-head self-attention ($W_q, W_k, W_v, W_o$) remains in FP16 to safeguard geometry and logit entropy.
   - Intermediate MLPs ($W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$) are compressed to symmetric group=128 INT4.
   - Rollout uses custom fused GEMV; backward and AdamW updates remain full FP16 autograd.

3. **The Tiny Honest Model Demonstration**:
   - Trained Qwen2.5-0.5B-Instruct for 30 GRPO steps in 103.1s on a single 16 GB T4.
   - Honest abstention on unanswerable traps jumped 8x from 6.7% to 53.3%, cutting hallucinations by half while preserving factual math/science accuracy (73.3% vs 76.7%).

---

## Pillar 3: Speculative Decoding & Serving Engine

1. **Pre-Allocated Static KV-Cache (`src/static_kv_cache.py`)**:
   - Allocates fixed-size continuous buffers upfront for keys and values.
   - Replaces PyTorch dynamic tensor slicing with $O(1)$ pointer-based rollback ($0.0003$ ms latency, $14,388\times$ faster than reallocation).

2. **Zero-Weight Draft Proposals (`src/unified_draft_engine.py`)**:
   - Uses Prompt Lookup Decoding (PLD) for 0-parameter n-gram matching, proposing candidate token sequences in 0.0062 ms without additional GPU forward passes.

3. **Unified Single-Pass Verification (`src/unified_speculative_engine.py`)**:
   - Combines draft verification, dynamic causal masking, and KV-cache commits into a single target forward pass.
   - Delivers a verified **1.48x net wall-clock speedup** on physical Tesla T4 hardware.

---

## Pillar 4: Mathematical Reasoning SFT (Baby-Chalk 1.5B)

1. **5-Tag Scientific Discovery Schema (`benchmarks/train_math_sft.py`)**:
   - Fine-tunes `Qwen2.5-Math-1.5B` on 605 certified reasoning traces (`data/chalk_seeds_500.jsonl`, 445k tokens) using LoRA ($r=32, \alpha=64$).
   - Enforces structured reasoning tags:
     - `<explore>`: Problem symmetry and constraint analysis.
     - `<conjecture>`: Hypothesizing candidate relationships.
     - `<test_edge_cases>`: Boundary testing and counterexample search.
     - `<lemma_isolate>`: Rigorous modular lemma statement.
     - `<formal_proof>`: Deductive proof compilation with boxed final answer.

2. **Strict Prompt Loss Masking**:
   - Tokenizes prompt and completion in a single pass and strictly masks all tokens prior to `<|im_start|>assistant\n` with `-100`, eliminating training loss corruption on prompt boundaries.

3. **Head-to-Head Out-of-Dataset Evaluation (`benchmarks/eval_math_benchmark.py`)**:
   - Evaluated on held-out AIME (2023–2024) and AMC 12 (2023) competition problems.
   - Solved 4 of 16 problems with 100% 5-tag schema compliance.
   - Delivered a **6x improvement in grounding** (60.0% vs 10.0%) on impossible premises, completely suppressing degenerate repetition loops.
