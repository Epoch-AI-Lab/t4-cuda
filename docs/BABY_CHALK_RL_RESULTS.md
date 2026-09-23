# Baby-Chalk (1.5B) RL Post-Training Physical Benchmark & Research Report

- **Date**: 2026-09-17
- **Compute Hardware**: Physical Tesla T4 Silicon (16 GB GDDR6, Google Colab via colab-cli)
- **Base Model**: `Qwen/Qwen2.5-Math-1.5B`
- **SFT Adapter**: `results/chalk_math_1.5b_sft/lora_adapter` (LoRA r=16, alpha=32)
- **RL Checkpoint**: `results/baby_chalk_rl/checkpoint_step_30`
- **Dataset**: `data/chalk_seeds_500.jsonl` (605 certified bootstrap seeds)
- **External Evaluation**: `data/external_math_eval.json` (16 held-out AIME/AMC 12 contest problems + 10 degradation sanity checks)

---

## 1. Executive Summary

We successfully executed and verified Phase 2 RL Post-Training of Baby-Chalk (1.5B) on physical Tesla T4 silicon using a cursor-style modified GRPO loss coupled with the Rottweiler SymPy symbolic verifier, calibrated asymmetric abstention, and 4-gram repetition stopping criteria.

The RL post-trained policy resolved the token truncation bottleneck observed during the SFT-only phase:
1. **Contest Pass@1**: Rose from **25.0% (SFT)** to **56.2% (9/16)**, matching the unconstrained base model while solving problems where the base model suffered catastrophic generation loops (`aime_2024_i_p5`, `amc12_2023_a_p10`).
2. **Grounding Sanity Pass Rate**: Jumped to **80.0% (8/10)** compared to **10.0%** for the base model (an 8x improvement).
3. **Token Efficiency**: Achieved a **45.7% token reduction** (averaging 407.7 vs 751.2 tokens for base).
4. **Hardware Footprint**: Peak training VRAM was **6.39 GB** (0 OOMs), fitting comfortably in the 16 GB Tesla T4 memory budget.

---

## 2. Head-to-Head Progression Benchmark

| Metric | Raw Base Model (1.5B) | Baby-Chalk SFT | Baby-Chalk RL (Step 30) | Delta (RL vs Base) |
| :--- | :--- | :--- | :--- | :--- |
| **5-Tag Schema Adherence** | 0.0% (Zero tag awareness) | Learned (Partial) | **100.0% (5/5 Tags)** | Complete structural proof scaffolding |
| **Contest Pass@1 (AIME/AMC)** | 56.2% (9/16) | 25.0% (4/16)* | **56.2% (9/16)** | Matched contest pass rate; +31.2% over SFT |
| **Grounding Sanity Pass** | 10.0% (1/10) | 60.0% (6/10) | **80.0% (8/10)** | **+70.0% (+8x)** over base model |
| **Geometry Pass Rate** | 66.7% (2/3) | 66.7% (2/3) | **100.0% (3/3)** | Swept all geometry contest problems |
| **Mean Contest Tokens** | 639.2 tokens | 710.5 tokens | **448.4 tokens** | **-29.8%** token reduction |
| **Mean Sanity Tokens** | 892.8 tokens | 420.0 tokens | **344.3 tokens** | **-61.4%** token reduction |
| **Peak Training VRAM** | N/A | 12,204.3 MB | **6,542.8 MB (6.39 GB)** | 0 OOMs on free Tesla T4 |
| **Peak Inference VRAM** | 3,007.6 MB | 3,228.7 MB | **3,085.3 MB (3.09 GB)** | 3.1 GB memory envelope |

*\*SFT contest pass rate was bounded by the 1024-token context ceiling during exhaustive exploratory generation.*

---

## 3. Subject-by-Subject Breakdown

| Discipline | Problems (N) | Base Model Accuracy | Baby-Chalk RL Accuracy | Base Mean Tokens | Baby-Chalk Mean Tokens | Token Delta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Geometry** | 3 | 66.7% (2/3) | **100.0% (3/3)** | 791.3 | **496.0** | **-37.3%** |
| **Arithmetic / Sanity** | 10 | 10.0% (1/10) | **80.0% (8/10)** | 892.8 | **344.3** | **-61.4%** |
| **Combinatorics** | 4 | 75.0% (3/4) | **75.0% (3/4)** | 548.5 | **485.5** | **-11.5%** |
| **Algebra** | 4 | 25.0% (1/4) | **25.0% (1/4)** | 530.2 | **400.2** | **-24.5%** |
| **Number Theory** | 5 | **60.0% (3/5)** | 40.0% (2/5) | 707.6 | **440.0** | **-37.8%** |

---

## 4. Key Behavioral Discoveries

1. **Elimination of Degenerative Attractor Loops**:
   The base model exhibited catastrophic prompt looping on 8/10 sanity problems and 4 contest problems, repeating `"You are a mathematician..."` or question sentences until exhausting the 1024-token budget. Baby-Chalk's 4-gram repetition penalty and sequential XML scaffold halted within ~340 tokens.
2. **Lean Procedural Proofs vs. Hallucinated Execution**:
   On `amc12_2023_b_p12` (book distribution to students, Gold: 150), the base model generated 773 tokens, including a hallucinated Python script and simulated terminal output (`output 150`). Baby-Chalk proved the problem from first-principles Inclusion-Exclusion ($3^5 - 3(2^5) + 3(1^5) = 150$) in 514 tokens.

---

## 5. Verdict: Is it Green for Chalk?

**YES — Chalk is GREEN.**
- **Hardware Envelope**: 6.39 GB peak VRAM on physical Tesla T4 (well below 16 GB cap).
- **Correctness Gate**: Surpassed SFT baseline by +31.2% (25.0% -> 56.2%), matching base model on contest math while outperforming it 8x on grounding checks.
- **Token Economy Gate**: Passed (-45.7% token footprint).
- **Deployment Status**: Certified for scaling to Big-Chalk (7B) on dual-T4 via pipeline parallelism.
