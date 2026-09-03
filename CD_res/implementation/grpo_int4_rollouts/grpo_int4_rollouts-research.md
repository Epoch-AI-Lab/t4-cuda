# Implementation Research: Wiring INT4 into GRPO Rollouts

## The Task
Integrate our verified per-group (group=128) INT4 W4A16 GEMM kernel into the rollout phase of TRL GRPOTrainer for Qwen2.5-0.5B-Instruct on a single Tesla T4 GPU. The trainer's optimization and backward pass remain in FP16, while token generation (which accounts for 85.4% of step latency) runs with INT4 fused kernels. Verify reward integrity against the FP16 baseline on GSM8K.

## 1. Common Gotchas
- **Off-Policy Distribution Drift in Importance Sampling**: GRPO computes policy ratios $r_t(\theta) = \frac{\pi_\theta(y_t | x, y_{<t})}{\pi_{\text{old}}(y_t | x, y_{<t})}$. If rollouts come from an INT4 policy $\pi_{\text{int4}}$ while loss evaluates $\pi_\theta$ in FP16, large quantization noise would create severe distribution mismatch ($r_t \gg 1$ or $r_t \ll 1$), corrupting advantage weighting.
  - *Source:* Schulman et al. (PPO, 2017); Shao et al. (DeepSeekMath / GRPO, 2024).
  - *Mitigation:* We use our group=128 symmetric INT4 kernel which demonstrated 0.024% relative error on real activations. We will log the reward trajectory and compare against `results/grpo_baseline/metrics.json`.
- **Restoring FP16 Forward Cleanly**: The model must use INT4 ONLY during `generate()`. During loss computation (`forward`) and gradient backpropagation (`backward`), the original FP16 weights and autograd graph must remain active.
  - *Source:* PyTorch Autograd Engine & TRL GRPOTrainer execution flow.
  - *Mitigation:* Implement a clean context manager `Int4RolloutScope` that patches `forward` on linear layers before `generate()` and restores the original forwards in a `finally:` block.
- **Bias and Layer Invariants**: As proven in Item 1, Qwen2.5 attention `q_proj`, `k_proj`, and `v_proj` have `bias=True`. The patched forward must add `self.bias`, and `lm_head` must remain in FP16.

## 2. Best Practices
- **In-Situ Quantization without Multi-GPU Overhead**: Because quantizing 450M weights on GPU takes < 10 ms while generation takes ~10 seconds per step, we can re-quantize the updated weights at each step with negligible overhead (< 0.1%), eliminating the need for complex Ray/vLLM multi-process setups on single T4 instances.
- **Identical Random Seeds and Data Pipeline**: The benchmark must evaluate the identical GSM8K split, prompt format, reward function, and hyperparameters as `benchmarks/benchmark_grpo_t4.py` to make reward curves directly comparable.

## 3. Pitfalls & Language Quirks
- **TRL 1.x Generation Hooking**: `GRPOTrainer` calls `self.model.generate(...)` inside `_generate_and_score_completions`. Intercepting `GenerationMixin.generate` or wrapping `trainer.model.generate` guarantees all rollout tokens go through the fused INT4 kernels while subsequent trainer passes remain standard FP16.

## 4. Differentiation
- **Industry Standard**: Large training clusters separate rollout actors (running on quantized inference engines like vLLM) from training actors (running on FP16/BF16 Megatron/FSDP) over NCCL.
- **Our Approach**: In-situ single-GPU hybrid training. Token rollouts are accelerated with our Turing-native INT4 GEMV kernel; backprop and AdamW happen in-place on the same GPU.

## Recommendation
1. Build `benchmarks/benchmark_grpo_int4.py` mirroring `benchmark_grpo_t4.py` with an `Int4RolloutScope` wrapping `GenerationMixin.generate`.
2. Save metrics and reward history to `results/grpo_int4/metrics.json`.
3. Compare reward curves and step times against `results/grpo_baseline/metrics.json`.
