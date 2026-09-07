# Project: Chalk 16,384-Token Context Expansion & RL Post-Training Pipeline

## Architecture
This project implements Chalk's 16,384-token context expansion and RL post-training pipeline with behavioral verifiers and controls under zero tolerance.
The system is partitioned into two tracks:
1. Implementation Track:
   - Context Scaling Layer (R1): YaRN and base frequency RoPE scaling to 16,384 tokens with PyTorch SDPA (Cutlass memory-efficient attention) and INT4 W4A16 MLP quantization (CalibratedRolloutScope), ensuring rollouts fit within 16GB T4 VRAM.
   - RL Optimization Loop (R2): Cursor-style modified GRPO loss with unscaled mean-centered advantages, removal of per-token length normalization, and detached CISPO upper-bound clipping.
   - Calibrated Abstention Engine (R3): Asymmetric reward schedule (+1.0 correct, 0.0 abstain, -1.5 incorrect) calibrated so guessing below 60% confidence yields negative expected value.
   - Anti-Looping & Attractor Prevention (R4): 4-gram repetition penalty stopping criteria (halting at 3 occurrences with penalty -0.5), forward-only 7-state XML tag progression FSM with token limits, and entropy floor controller.
   - Rottweiler Verification & Format Discrimination (R5): 5-stage XML scaffold (<explore>, <conjecture>, <test_edge_cases>, <lemma_isolate>, <formal_proof>), SymPy execution verifier with exact equivalence and reverse equation substitution, and 15-20% general replay buffer penalizing format leaks.
2. E2E Verification Track:
   - Comprehensive multi-tier test suite (Tiers 1-4) verifying memory bounds, numerical stability, reward calibration, anti-looping stops, and format discrimination.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | 16k RoPE Scaling | YaRN and base frequency adjustment for 16,384 token sequences | M1 | Survey (Explorer 1) |
| 2 | Memory-Efficient Attention | PyTorch SDPA integration eliminating 15GB DRAM score matrix | M1 | Survey (Explorer 1) |
| 3 | Scaled Static KV Cache | FP16 KV cache buffer scaling to 16,384 tokens within 3.5GB VRAM | M1 | Survey (Explorer 1) |
| 4 | INT4 W4A16 Weight Scope | CalibratedRolloutScope integration fitting 16k rollouts in 11.1GB VRAM | M1 | Survey (Explorer 1) |
| 5 | Unscaled Mean-Centered Advantages | Advantage calculation A_i = r_i - r_bar with zero gradient on tied groups | M2 | Survey (Explorer 2) |
| 6 | Unnormalized Token Loss | Removal of 1/|o_i| token length normalization for elastic proof depth | M2 | Survey (Explorer 2) |
| 7 | Detached CISPO Clipping | Detached importance ratio upper-bound clipping to maintain active gradients | M2 | Survey (Explorer 2) |
| 8 | Asymmetric Reward Engine | Correct (+1.0), Abstain (0.0), Incorrect (-1.5) reward assignment | M3 | Survey (Explorer 2) |
| 9 | 60% Confidence Calibration | Proof and verification that guessing below 60% confidence yields EV < 0 | M3 | Survey (Explorer 2) |
| 10 | Anti-Hedging Reward Guard | Penalty -1.5 on completions that hedge with both abstain and answer | M3 | Survey (Explorer 2) |
| 11 | 4-Gram Repetition Penalty | Stopping criteria halting rollout at 3 occurrences with penalty -0.5 | M4 | Survey (Explorer 3) |
| 12 | XML Tag Progression FSM | Forward-only 7-state FSM preventing exploratory stalling | M4 | Survey (Explorer 3) |
| 13 | Training Entropy Floor | Dual ascent controller keeping policy entropy >= 0.25 nats | M4 | Survey (Explorer 3) |
| 14 | 5-Stage XML Scaffold | Strict sequential validation of <explore> through <formal_proof> | M5 | Survey (Explorer 3) |
| 15 | SymPy Reverse Substitution | Symbolic equivalence and residual check simplify(LHS - RHS) == 0 | M5 | Survey (Explorer 3) |
| 16 | Format Discrimination Buffer | 15-20% general replay buffer penalizing XML on casual prompts | M5 | Survey (Explorer 3) |
| 17 | E2E Post-Training Pipeline | Integrated training and validation runner on 16k context | M6 | Survey (All Explorers) |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | M1: 16k RoPE Scaling & Memory Attention | Implement RoPE scaling config, scaled static KV cache, SDPA integration, and VRAM verification | none | PLANNED |
| 2 | M2: Cursor-Style Modified GRPO Loop | Implement ModifiedGRPOLoss with unscaled advantages, no length norm, and detached CISPO clipping | none | PLANNED |
| 3 | M3: Calibrated Abstention Engine | Implement CalibratedAbstentionRewardEngine with asymmetric rewards, EV calibration, and anti-hedging | none | PLANNED |
| 4 | M4: Anti-Looping & Tag Progression | Implement 4-gram stopping criteria, XML progression FSM, and entropy regularization floor | none | PLANNED |
| 5 | M5: Rottweiler Verifier & Format Replay | Implement 5-stage XML validator, SymPy reverse substitution verifier, and format replay buffer | none | PLANNED |
| 6 | M6: E2E Integration & Verification | Integrated pipeline runner, 4-tier test suite, and forensic verification gate | M1, M2, M3, M4, M5 | PLANNED |

## Interface Contracts
### Context Scaling (`src/rope_scaling.py`)
- `configure_rope_scaling(config, max_position_embeddings=16384, rope_type="yarn", factor=4.0)`:
  - Updates model config in place with standardized RoPE parameters.
- `apply_16k_context_scope(model, max_seq_len=16384)`:
  - Configures rotary embeddings and registers SDPA attention forwards.

### Modified GRPO (`src/rl/modified_grpo_loss.py`)
- `ModifiedGRPOLoss(clip_eps_low=0.2, clip_eps_high=0.2, beta_kl=0.01)`:
  - Inputs: `log_probs` [B, G, T], `old_log_probs` [B, G, T], `rewards` [B, G], `attention_mask` [B, G, T], `completion_mask` [B, G, T].
  - Returns: `loss` scalar, `metrics` dict (advantages, clipped ratio, kl divergence).

### Calibrated Abstention (`src/rl/calibrated_abstention.py`)
- `CalibratedAbstentionRewardEngine(r_correct=1.0, r_abstain=0.0, r_incorrect=-1.5, r_hedge=-1.5)`:
  - Method `compute_reward(completion: str, ground_truth: str) -> Tuple[float, Dict[str, Any]]`.
  - Guarantees $E[R] < 0$ when accuracy $p < 0.60$.

### Anti-Looping & Tag Progression (`src/rl/anti_looping.py`)
- `FourGramRepetitionCriteria(max_occurrences=2, penalty=-0.5)`:
  - Hugging Face `StoppingCriteria` tracking rolling 4-grams.
- `XMLProgressionTracker(tag_order=[...], per_stage_token_limits={...})`:
  - Enforces forward-only tag state transitions and per-stage token caps.
- `EntropyFloorController(floor_nats=0.25, lr=0.01)`:
  - Dynamically regulates entropy bonus multiplier to keep mean entropy above floor.

### Rottweiler Verifier & Format Replay (`src/rl/rottweiler_verifier.py`, `src/rl/format_replay_buffer.py`)
- `RottweilerVerifier()`:
  - `verify_xml_scaffold(text: str) -> Tuple[bool, str]`.
  - `verify_sympy_solution(prediction: str, ground_truth: str, equation_str: Optional[str] = None) -> Tuple[bool, str]`.
- `FormatDiscriminationReplayBuffer(math_ratio=0.85, general_ratio=0.15)`:
  - `sample_batch(batch_size: int)`: Yields mixed math and conversational/coding prompts.
  - `evaluate_format_compliance(prompt_type: str, completion: str) -> float`.

## Code Layout
- `src/rope_scaling.py`: RoPE configuration and 16k context setup.
- `src/static_kv_cache.py`: Scaled static KV cache supporting 16,384 tokens in FP16.
- `src/rl/`:
  - `__init__.py`: RL package exports.
  - `modified_grpo_loss.py`: Cursor-style modified GRPO loss function.
  - `calibrated_abstention.py`: Calibrated abstention reward engine.
  - `anti_looping.py`: 4-gram repetition penalty, XML progression FSM, and entropy floor.
  - `rottweiler_verifier.py`: 5-stage XML validator and SymPy reverse substitution verifier.
  - `format_replay_buffer.py`: Format discrimination replay buffer.
- `benchmarks/run_chalk_rl_pipeline.py`: Complete post-training rollout and training pipeline harness.
- `tests/test_context_expansion_16k.py`: Unit and VRAM verification tests for R1.
- `tests/test_modified_grpo_loss.py`: Unit and numerical tests for R2.
- `tests/test_calibrated_abstention.py`: Unit and mathematical calibration tests for R3.
- `tests/test_anti_looping.py`: Unit and stopping criteria tests for R4.
- `tests/test_rottweiler_verifier.py`: Unit and AST execution tests for R5.
- `tests/test_format_replay.py`: Replay buffer and format discrimination tests.
- `tests/test_e2e_chalk_pipeline.py`: Comprehensive multi-tier opaque-box integration suite.
