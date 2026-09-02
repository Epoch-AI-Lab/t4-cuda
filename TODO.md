# T4-CUDA Moonshot Roadmap

The lab's moonshot: the first rigorous, kernel-native, low-precision RL
training system on free T4s, demonstrated end to end. Kernels are the
identity. Every claim stays machine-verifiable.

## 1. Per-group INT4 W4A16 GEMM (kernel chokepoint — DONE ✅ 2026-09-02)

- **Problem:** per-channel INT4 is too lossy for Qwen2.5-0.5B (~8-9% activation
  error → incoherent generation). Ruled out (2026-09-02).
- **Fix:** per-group symmetric quantization, group=128 in K, GPTQ-style. Exact in-loop
  FP32 dequantization (LOP3 unpack - 1024.0f, scaled by (q - z)*s).
- **Differential Result:** rel err **0.024%** on real Qwen2.5-0.5B activations (gate target was <= 0.1%).
- **Gate Result:** PASSED on Tesla T4 silicon. 8 prompts, 256 tok: **279.1 tok/s** (1.20x speedup vs fp16 232.5 tok/s),
  VRAM down from 2.84 GB to **1.24 GB** (56% reduction), 100% coherent outputs across all 8 prompts.

## 2. Wire INT4 into GRPO rollouts

- Quantized rollout policy (INT4 part) inside TRL GRPOTrainer, trainer stays
  fp16. Rollouts are 85.4% of step time (measured), so this attacks the bulk.
- **Measure reward integrity:** the reward curve of a quantized-policy run vs
  the fp16 baseline on the same GSM8K bench. Lossy kernels can silently break
  training; we measure it exactly. This is the part nobody does.

## 3. Train a tiny honest model under the stack (the demo)

- 0.5B-1.5B, free T4s, GRPO with an abstain token + calibrated honesty reward.
- The kernels power the training; the model is the demo; calibrated honesty is
  the research cherry. "Smallest model that knows when it doesn't know."

## 4. Conjecture loop (stretch, rides the same kernels)

- Old-model sees post-cutoff mathlib/Lean-Workbook + recent arXiv, proposes
  lemmas, Lean 4 checks truth, embeddings check novelty.
- Machine-verified = credible. RL rewards on Lean-pass + novelty + abstain.
  Only after 1-3 land.

---

Principle: measure twice, cut once. Kernel work first; the rest rides it.