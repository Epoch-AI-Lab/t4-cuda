# RL and Post-Training Survey for Chalk

Sources: Z.ai (Zhipu AI / THUDM), Kimi (Moonshot AI), MiniMax, Cursor (Anysphere).
Compiled: 2026-09-06.

---

## Phase 1: Verified Compilation

### DeepSeek: GRPO Origin and Reasoning Distillation

**Source:** DeepSeekMath (arXiv:2402.03300, Feb 2024), DeepSeek-R1 (arXiv:2501.12948, Jan 2025).

**What they built:** GRPO removes the critic model from PPO entirely. Instead of a learned value baseline, it samples $G$ rollouts per prompt and normalizes advantage by group mean/std:
$$\hat{A}_i = \frac{r_i - \text{mean}(\{r_1, \dots, r_G\})}{\text{std}(\{r_1, \dots, r_G\}) + \epsilon}$$

**Key results (verified):**
- DeepSeekMath-7B with GRPO: 51.7% MATH, 88.2% GSM8K (beats Minerva 540B).
- R1-Distill-Qwen-1.5B: 28.8% AIME, 82.8% MATH-500.
- R1-Distill-Qwen-7B: 55.5% AIME, 92.8% MATH-500.

**Critical finding for small models:** Pure RL on base Qwen-2.5-32B underperformed distillation from the 671B teacher. Small models lack the capacity to discover complex search patterns from scratch. Distillation transfers reasoning priors, RL then polishes.

**R1 four-stage pipeline:**
1. Cold-start SFT (few thousand curated long CoT demos)
2. Reasoning RL (GRPO + accuracy reward + format reward + language consistency penalty)
3. Rejection sampling SFT (600k reasoning + 200k non-reasoning = 800k)
4. Secondary RL (rule-based + preference reward models)

---

### Z.ai (Zhipu AI / THUDM): Infrastructure and Mismatch Stabilization

**Source:** THUDM/slime (GitHub), GLM-4.5 (arXiv:2508.06471), RLCS (arXiv:2507.01006).

#### Slime RL Framework
Async RL infra coupling Megatron-LM (policy updates, TP/PP/DP) with SGLang (rollout decoding). Standard RL trainers run generation and backward sequentially, underutilizing GPUs since rollout = 80-90% of step time.

#### IcePop: Rollout-Training Distribution Mismatch
When rollouts use FP8/quantized inference and gradients accumulate in FP16/BF16, tiny logit differences cause importance weights $r_t(\theta) = \frac{\pi_\theta}{\pi_{old}}$ to spike. IcePop applies token-level discrepancy masking: filters out tokens where $|\log \pi_\theta - \log \pi_{old}| > \delta$ before they corrupt the advantage calculation.

**Directly relevant to chalk:** We run INT4 W4A16 GEMV during rollout decode and FP16 during backward. This is exactly the mismatch IcePop was built to fix.

#### RLCS: Curriculum Sampling
Problems binned by difficulty using moving pass@k. Rollout workers sample prompts where current success rate is 10-70%. Prevents reward starvation (all rollouts fail, zero gradient) on hard problems.

**Result:** GLM-4.1V-9B-Thinking matched 72B baselines on STEM/coding reasoning.

#### SLIME Algorithm (arXiv:2602.02383)
Reference-free preference optimization. Fixes DPO's likelihood displacement and formatting collapse via anchoring term + stabilizing penalty + dual-margin constraint.

---

### Kimi (Moonshot AI): Long-CoT RL and Long2Short Distillation

**Source:** Kimi k1.5 (arXiv:2501.12599, Jan 2025), Kimi K2 (arXiv:2507.20534, Jul 2025), Moonlight (arXiv:2502.16982, Feb 2025).

#### Critic-Free Policy via Online Policy Mirror Descent (OPMD)
Same family as GRPO but formalized as online mirror descent with KL regularization:
$$\max_{\pi} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi} [r(x, y)] - \beta D_{KL}(\pi \parallel \pi_{ref})$$

Kimi explicitly argues critics are destructive on long reasoning chains. A value network penalizes intermediate dead-ends even though backtracking is mandatory for correct final answers. Group baselines evaluate complete chain outcomes without punishing exploration.

**Numbers:** AIME 77.5, MATH-500 96.2, Codeforces 94th percentile (all Long-CoT).

#### Partial Rollouts for Long-Context RL
Rollouts broken into chunked segments. Unfinished trajectories suspended, cached in replay buffer, resumed later. Scales RL from 32k to 128k context without OOM.

#### Length Penalty for Overthinking
$$r_{final}(x, y) = r_{correctness}(x, y) - \lambda \cdot \max(0, |y| - L_{target})$$
Penalty only activates past an expected solution threshold, preventing premature stops while bounding runaway verbosity.

#### Long2Short Distillation (the killer technique)
Four methods to compress 128k long-CoT reasoning into compact short-CoT:

1. **Shortest Rejection Sampling:** Sample multiple paths, verify with SymPy, keep only the shortest correct path for SFT.
2. **Length-Differentiated DPO:** Chosen = concise correct. Rejected = verbose correct (or verbose incorrect).
3. **Long2Short RL:** Second RL stage with aggressive length penalty.
4. **Model Merging:** SLERP between long-CoT and short-CoT checkpoints.

**Results:** Short-CoT AIME 60.8 (vs 77.5 long), MATH-500 94.6 (vs 96.2 long) with a fraction of inference tokens.

#### Staged Context Curriculum
- Stage 1: 32k context, LR $2 \times 10^{-5} \to 2 \times 10^{-6}$
- Stage 2: 128k context, LR re-warmed $1 \times 10^{-5} \to 1 \times 10^{-6}$

---

### MiniMax: CISPO and the Adam Epsilon Discovery

**Source:** MiniMax-M1 (arXiv:2506.13585, Jun 2025), MiniMax-M2 (arXiv:2605.26494, May 2026).

#### CISPO (Clipped Importance Sampling Policy Optimization)
The most technically interesting RL algorithm in this survey. Fixes a fundamental flaw in GRPO/PPO clipping.

**The problem:** Standard GRPO clips $r_t(\theta)$ within $[1-\epsilon, 1+\epsilon]$. When the ratio hits the threshold, the clipped term has zero derivative. This zeros out gradients on critical low-frequency reasoning tokens ("Wait", "Alternatively", "Let me check") when policy shifts slightly.

**CISPO formulation:**
$$\mathcal{L}_{CISPO} = - A_t \cdot \text{sg}\left(\text{clip}(r_t(\theta), 1 - \epsilon_{low}, 1 + \epsilon_{high})\right) \cdot \log \pi_\theta(o_t | q, o_{<t})$$

Where $\text{sg}(\cdot)$ is stop-gradient (detach). The importance ratio is bounded but detached. Gradient flows entirely through $\log \pi_\theta$. Every token retains its policy gradient, weighted by a bounded importance coefficient.

**Result:** Reaches target accuracy with 2x fewer steps vs DAPO.

#### One-Sided Clipping
Lower bound on importance weight harms exploration. Omit it ($\epsilon_{low} = \infty$), apply only upper clip ($clip_{high} = 4.0$).

#### The Adam Epsilon Discovery
In long reasoning trajectories (10K-80K tokens), gradients span $10^{-18}$ to $10^{-5}$. Default PyTorch Adam `eps = 1e-8` drowns signals below $10^{-8}$, causing training stagnation.

**Fix:** `adam_epsilon = 1e-15`, $\beta_1 = 0.9$, $\beta_2 = 0.95$.

#### SFT/RL Data Separation
SFT data teaches format scaffolds only (XML tags, CoT structure). SFT prompts and RL prompts must be disjoint. Overlap kills exploratory diversity and causes policy entropy collapse.

#### Lightning Attention (inference efficiency)
7:1 hybrid ratio: 70 linear attention layers, 10 softmax layers (every 8th). Pure linear attention fails on exact recall. Periodic softmax preserves it. At 100K tokens, 75% FLOP reduction vs pure softmax.

---

### Cursor (Anysphere): Modified GRPO for Multi-Turn Agent RL

**Source:** Composer 2 (arXiv:2603.24477, Mar 2026), Tab RL blog (cursor.com/blog/tab-rl, Sep 2025).

#### Modified GRPO Fixes
Three targeted modifications:

1. **Drop length normalization.** Standard GRPO normalizes rewards by sequence length. Creates artificial length bias rewarding verbose padding. Remove it to force clean, direct outputs.

2. **Skip advantage std normalization on tied groups.** When all $K$ rollouts in a group get identical scores (all pass or all fail), dividing by near-zero std amplifies noise into massive gradients. Skip normalization entirely for uniform groups.

3. **Self-summarization for long credit assignment.** In trajectories spanning 100K+ tokens, force periodic self-summarization. Terminal reward backpropagates through summaries, crediting both the final action and earlier summarization decisions.

#### Asymmetric Reward for Calibrated Abstention
From the Tab model:
- Accepted: +0.75
- Rejected: -0.25
- Silent (abstain): 0.0

Expected reward: $\mathbb{E}[R] = 0.75p - 0.25(1-p) = p - 0.25$. Positive only when $p > 0.25$. The model learns when to shut up without external confidence thresholding.

**Result:** 28% higher acceptance rate, 21% fewer suggestions shown.

#### Prompt-Lookup Speculative Decoding
Use the input text as the draft source. When output matches input (problem restatement, boilerplate), validate entire chunks in one forward pass. ~1000 tok/s, 13x faster than vanilla Llama-3-70B.

#### Environment Bootstrapping (Autoinstall)
Older model (Composer 1.5) prepares the sandbox environment before the RL agent begins. Installs dependencies, mocks external APIs, runs baseline tests. The RL agent only encounters solvable environments.

---

## Design Space Map

| Axis | DeepSeek | Z.ai | Kimi | MiniMax | Cursor |
|---|---|---|---|---|---|
| **RL Algorithm** | GRPO | PPO + IcePop (GLM-5.2), GRPO (GLM-4.5) | OPMD (equiv. to GRPO) | CISPO | Modified GRPO |
| **Critic Model** | None (group baseline) | Value model in GLM-5.2 | None (group baseline) | None (group baseline) | None (group baseline) |
| **Clipping** | Standard PPO clip | Standard + IcePop masking | Standard | One-sided upper only, detached | Skip on uniform groups |
| **Length Control** | Implicit via reward | Not disclosed | Thresholded penalty | Not disclosed | Remove length norm |
| **Curriculum** | Not disclosed | RLCS (10-70% pass rate) | Staged context (32k->128k) | Not disclosed | N/A (agentic) |
| **Long2Short** | Distillation (800k dataset) | Not disclosed | 4 methods (rejection, DPO, RL, merge) | Not disclosed | Self-summarization |
| **Verifier** | SymPy + compiler | Rule-based | SymPy + sandbox | Rule-based + GenRM | User feedback + tests |
| **Mixed-Precision Fix** | N/A | IcePop token masking | Not disclosed | Adam eps=1e-15 | N/A |

---

## Contradictions and Tensions

1. **Critic vs no critic.** Z.ai returned to PPO with a value model for GLM-5.2, claiming GRPO had too much variance on long agent interactions. Everyone else (DeepSeek, Kimi, MiniMax, Cursor) eliminated the critic. Tension: is critic-free viable for multi-turn agent RL, or only for single-turn math? Needs manual verification on chalk's multi-step proof traces.

2. **Length normalization.** Cursor explicitly removes it. Kimi adds a thresholded penalty. These are opposite approaches to the same problem (verbose rollouts). Which one works better for math specifically is an open question.

3. **Distillation vs direct RL on small models.** DeepSeek says distillation from a large teacher beats direct RL on small models. Kimi's Long2Short pipeline also implies a teacher. But our setup has no 671B teacher. Tension: can GRPO alone on a 7B base reach competitive math performance, or is distillation initialization mandatory?

4. **Adam epsilon.** MiniMax prescribes `1e-15`. This is 7 orders of magnitude below the PyTorch default. Risk of numerical instability if gradients are genuinely zero (division by near-zero). Needs careful testing before adopting.

---

## Sources

- DeepSeekMath: arXiv:2402.03300
- DeepSeek-R1: arXiv:2501.12948
- THUDM/slime: github.com/THUDM/slime
- GLM-4.5: arXiv:2508.06471
- RLCS / GLM-4.1V-Thinking: arXiv:2507.01006
- IcePop / Ring-1T: Ant Group / Bailing
- SLIME algorithm: arXiv:2602.02383
- Kimi k1.5: arXiv:2501.12599
- Kimi K2: arXiv:2507.20534
- Moonlight / Muon: arXiv:2502.16982
- MoBA: arXiv:2502.13189
- Kimi K3: arXiv:2607.24653
- MiniMax-01: arXiv:2501.08313
- MiniMax-M1: arXiv:2506.13585
- MiniMax-M2: arXiv:2605.26494
- One-RL-to-See-Them-All: arXiv:2505.18129
- Lightning Attention-2: arXiv:2401.04658
- Composer 2: arXiv:2603.24477
- Cursor Tab RL: cursor.com/blog/tab-rl (Sep 2025)
- Cursor online RL: cursor.com/blog (Mar 2026)
- Cursor autoinstall: cursor.com/blog (May 2026)
