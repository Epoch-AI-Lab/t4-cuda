# CD Derivation: RL Post-Training for Chalk

Derived from the compilation survey across Z.ai, Kimi, MiniMax, and Cursor.
Ordered by confidence and impact. Each traces back to its source.

---

### Idea 1: Replace GRPO Clipping with CISPO

- **Source trace:** MiniMax-M1 (arXiv:2506.13585), Section on CISPO formulation.
- **Observation:** Standard GRPO's clipped surrogate zeros out gradients on tokens that hit the clip boundary. With small group sizes ($G=4$ or $G=8$, which is what we can afford on T4), a large fraction of reasoning-critical tokens lose their gradient signal.
- **Implication:** Our chalk model's structured tags (`<explore>`, `<conjecture>`, `<test_edge_cases>`, `<formal_proof>`) are low-frequency pivot tokens. Standard clipping drops them first.
- **Proposal:** Implement CISPO in the GRPO training loop. Detach the importance ratio with stop-gradient, flow gradient only through $\log \pi_\theta$. Upper clip at 4.0, no lower bound.
- **Steps:**
  1. In `benchmarks/train_math_sft.py` (or wherever GRPO loss is computed), replace the standard clipped surrogate with:
     ```python
     ratio = (new_logprobs - old_logprobs).exp()
     clipped_ratio = ratio.clamp(max=4.0).detach()
     loss = -(advantages * clipped_ratio * new_logprobs).mean()
     ```
  2. Remove any existing length normalization on rewards.
  3. Add uniform-group detection: if `std(rewards_in_group) < 1e-8`, skip advantage normalization for that group.
- **Confidence:** inferred (CISPO verified on MiniMax-M1 at 456B scale, untested on 1.5B-7B, but the gradient zeroing problem is worse at small scale)
- **Risk:** One-sided clipping might allow too-large updates on rare exploratory tokens, destabilizing training. Monitor KL divergence from reference policy.

---

### Idea 2: IcePop-Style Token Masking for INT4/FP16 Mismatch

- **Source trace:** Z.ai / Ant Group IcePop, used in GLM-5 and GLM-5.2 RL.
- **Observation:** Chalk runs INT4 W4A16 GEMV kernels during rollout decode and FP16 during backward. The 0.024% relative error we measured (TODO.md, Section 1) means logits differ between the policy that generated the tokens and the policy computing gradients.
- **Implication:** Importance ratios $r_t = \frac{\pi_\theta}{\pi_{old}}$ can spike on tokens where INT4 and FP16 logit distributions disagree, injecting noise into the advantage calculation.
- **Proposal:** After computing log-probabilities in both INT4 (generation) and FP16 (training), mask out tokens where $|\log \pi_{FP16}(o_t) - \log \pi_{INT4}(o_t)| > \delta$ (suggested $\delta = 0.1$). Zero those tokens' contributions to the policy gradient.
- **Steps:**
  1. During rollout, store per-token log-probs from the INT4 forward pass.
  2. During the training forward pass (FP16), compute per-token log-probs.
  3. Compute discrepancy: `disc = (fp16_logprobs - int4_logprobs).abs()`
  4. Create mask: `valid = disc < 0.1`
  5. Apply mask to loss: `loss = -(advantages * logprobs * valid.float()).sum() / valid.sum()`
- **Confidence:** inferred (IcePop verified at Z.ai on MoE models with FP8/BF16 mismatch; INT4/FP16 is a larger gap but same mechanism)
- **Risk:** If INT4 error is correlated across token positions (systematic bias rather than random noise), masking alone won't fix it. May need GPTQ calibration on the rollout model.

---

### Idea 3: Shortest Rejection Sampling for Cold-Start SFT v2

- **Source trace:** Kimi k1.5 (arXiv:2501.12599), Long2Short distillation pipeline.
- **Observation:** Our 605-seed bootstrap dataset contains traces of varying length. Some problems have 1500-token proofs where a 400-token proof suffices. The cold-start SFT trains on all of them, biasing the model toward verbose reasoning.
- **Implication:** When GRPO starts, the policy initialization is already verbose, consuming more T4 VRAM per rollout and requiring more RL steps to compress.
- **Proposal:** Before GRPO Phase 2, re-sample the seed dataset. For each problem, generate 8 completions from the current SFT checkpoint, verify with SymPy, keep only the shortest correct completion. Re-run SFT on this compressed dataset.
- **Steps:**
  1. Load SFT checkpoint from `results/chalk_math_1.5b_sft/lora_adapter`.
  2. For each of the 605 problems, generate 8 completions (temperature 0.7).
  3. Verify each with SymPy symbolic matching.
  4. Select shortest correct. If none correct, keep original seed.
  5. Write `data/chalk_seeds_shortest.jsonl`.
  6. Re-SFT for 1 epoch on the shortened dataset.
- **Confidence:** confirmed (Kimi reports Short-CoT MATH-500 94.6 vs 96.2 Long-CoT, nearly matching accuracy with far fewer tokens)
- **Risk:** Over-compression might remove necessary exploratory steps. Gate: verify SFT loss doesn't increase by more than 0.3 nats.

---

### Idea 4: Curriculum Staging by Difficulty

- **Source trace:** Z.ai RLCS (arXiv:2507.01006), Kimi k1.5 staged context curriculum.
- **Observation:** Chalk's current plan (TODO.md Section 4) jumps straight from SFT to GRPO on mixed AIME/AMC problems. 1.5B models get 25% pass@1 on contest problems but 60% on sanity checks. The hard problems produce zero-reward rollout groups.
- **Implication:** Zero-reward groups = zero gradient. Compute wasted. Worse, the model learns nothing from repeated failure.
- **Proposal:** Implement RLCS-style difficulty binning:
  - **Tier 1 (warmup):** GSM8K-level problems where model passes 40-70% of the time. Run 50 GRPO steps.
  - **Tier 2 (ramp):** AMC 10/12 problems, 15-50% pass rate. Run 100 GRPO steps.
  - **Tier 3 (competition):** AIME problems, 5-25% pass rate. Run 200 GRPO steps.
  - Track moving pass@4 per problem. Promote to next tier when median pass rate exceeds 50%.
- **Confidence:** confirmed (RLCS verified on GLM-4.1V-9B, which is comparable scale to our 7B target)
- **Risk:** Curriculum might be too slow. Could skip Tier 1 if the SFT checkpoint already handles GSM8K at >70%.

---

### Idea 5: Adam Epsilon = 1e-15

- **Source trace:** MiniMax-M1 (arXiv:2506.13585), optimizer configuration section.
- **Observation:** Math reasoning RL generates gradients spanning $10^{-18}$ to $10^{-5}$. Default Adam `eps=1e-8` drowns signals below $10^{-8}$.
- **Implication:** The structured reasoning tags (`<explore>`, `<conjecture>`) are rare tokens with small gradients. They're the most important tokens for chalk's structured reasoning. Default Adam kills them.
- **Proposal:** Set `adam_epsilon = 1e-15` in the optimizer config for GRPO training. Keep $\beta_1 = 0.9$, $\beta_2 = 0.95$.
- **Confidence:** confirmed (MiniMax verified on M1 at scale; straightforward hyperparameter change)
- **Risk:** If any gradient is genuinely zero (which happens with gradient masking), dividing by $\epsilon = 10^{-15}$ can produce NaN. Add gradient clipping (`max_grad_norm = 1.0`) as a safety net. Monitor for loss spikes in the first 10 steps.

---

### Idea 6: Asymmetric Reward for Calibrated Abstention

- **Source trace:** Cursor Tab RL (cursor.com/blog/tab-rl, Sep 2025), Kimi k1.5 length penalty.
- **Observation:** Chalk already showed 8x improvement in honest abstention (6.7% to 53.3% "I don't know" on impossible traps, TODO.md Section 3). But that was GRPO on a tiny honesty dataset. For math, the model needs to learn when to backtrack mid-proof rather than completing a wrong derivation.
- **Implication:** Binary reward (1.0 correct, 0.0 wrong) treats a confidently wrong proof the same as an uncertain one. The model has no incentive to stop and reconsider.
- **Proposal:** Three-tier reward:
  - Correct final answer: +1.0
  - Wrong final answer: -0.5
  - Explicit `<backtrack>` token before wrong answer, followed by a correction attempt: +0.3 (if final answer correct), 0.0 (if still wrong)
- **Steps:**
  1. Add `<backtrack>` to the structured tag vocabulary.
  2. Modify the reward function to detect backtrack tokens and adjust rewards.
  3. In SFT cold-start, include 50-100 examples demonstrating backtracking.
- **Confidence:** speculative (Cursor verified asymmetric rewards for code suggestions, not for math proofs. The backtrack token idea is novel.)
- **Risk:** The model might learn to always emit `<backtrack>` as a hedge (reward hacking). Mitigate: if `<backtrack>` appears more than 3 times per solution, apply a -0.1 penalty per extra occurrence.

---

### Idea 7: Length-Differentiated DPO After GRPO

- **Source trace:** Kimi k1.5 Long2Short pipeline (arXiv:2501.12599).
- **Observation:** After GRPO training, the model will produce correct but potentially verbose proofs. A second pass of DPO can compress reasoning without losing accuracy.
- **Implication:** Shorter proofs = less KV cache = faster inference on T4 = more rollouts per training step in subsequent RL iterations.
- **Proposal:** After the GRPO phase stabilizes:
  1. Generate 8 completions per problem from the GRPO checkpoint.
  2. Verify all with SymPy.
  3. For problems with multiple correct completions, create DPO pairs: chosen = shortest correct, rejected = longest correct (or any incorrect).
  4. Run 1 epoch of DPO on these pairs.
- **Confidence:** confirmed (Kimi verified on k1.5, Short-CoT MATH-500 was 94.6 vs 96.2 Long-CoT)
- **Risk:** DPO can cause catastrophic forgetting on problems the model barely solved. Use a small learning rate ($1 \times 10^{-6}$) and limit to 1 epoch.

---

### Idea 8: Prompt-Lookup Speculative Decoding for Rollout Speedup

- **Source trace:** Cursor speculative editing (cursor.com/blog, May 2024).
- **Observation:** Math rollouts contain significant chunks of verbatim problem restatement, standard definitions, and boilerplate. These are predictable from the prompt.
- **Implication:** On T4, rollout generation is the bottleneck (80-90% of RL step time). Speeding up generation by 2-4x directly reduces wall-clock training time.
- **Proposal:** During GRPO rollout generation, use prompt-lookup speculative decoding. When generated tokens match the input prompt text, validate entire chunks in a single forward pass instead of autoregressive token-by-token.
- **Confidence:** inferred (Cursor verified 13x speedup on code editing; math has less verbatim repetition but still significant boilerplate, so 2-4x is conservative)
- **Risk:** Implementation complexity. Need to modify the generation loop in TRL's GRPOTrainer. Might conflict with INT4 rollout scope.

---

### Scratch / Raw Notes

- **Kimi MoBA (arXiv:2502.13189):** Block sparse attention for long-context. Not directly relevant to chalk's 2048-4096 token context, but becomes relevant if we scale to longer proofs.

- **MiniMax Lightning Attention 7:1 hybrid:** Interesting for inference serving but not for training on T4. Our bottleneck is VRAM, not attention FLOPs at 2k-4k tokens.

- **Cursor's 5-hour online RL loop:** We can't do this (no production traffic), but the principle of small frequent on-policy updates is applicable. Instead of accumulating a giant rollout dataset then training, generate 32-64 rollouts, update immediately, discard. Keeps policy on-distribution.

- **Z.ai's return to PPO with critic for GLM-5.2:** Worth monitoring. If chalk's GRPO shows high variance on long proofs (>4k tokens), adding a small critic (even a linear probe on the last hidden state) might stabilize training. But only if memory allows.

- **Muon optimizer (Moonlight, arXiv:2502.16982):** 2x compute efficiency over AdamW. Needs matrix orthogonalization which adds complexity. Could be worth testing after GRPO pipeline is stable, as an optimizer swap.

- **SFT/RL data separation (MiniMax):** Critical check for chalk. Verify that `data/chalk_seeds_500.jsonl` problems are NOT in the GRPO prompt pool. If they overlap, policy entropy will collapse.
