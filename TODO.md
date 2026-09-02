# T4-CUDA Moonshot Roadmap

The lab's moonshot: the first rigorous, kernel-native, low-precision RL
training system on free T4s, demonstrated end to end. Kernels are the
identity. Every claim stays machine-verifiable.

## 1. Per-group INT4 W4A16 GEMM (kernel chokepoint — do first)

- **Problem:** per-channel INT4 is too lossy for Qwen2.5-0.5B (~8-9% activation
  error → incoherent generation). Ruled out (2026-09-02).
- **Fix:** per-group quantization, group=128 in K, GPTQ-style. Scale/zp per
  k-group. Reuse the two-accumulator exact-reconstruction scheme from the u4
  kernel fix.
- **First:** KAT + differential vs exact python ref, target rel err ~0.1% on
  real q_proj activations.
- **Gate:** same bench (8 prompts, 256 tok, greedy). Pass = coherent output,
  tokens/sec measured vs fp16.

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