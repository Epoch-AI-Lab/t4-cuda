# E2E Test Infra: Multi-GPU Sharding on Dual Tesla T4

## Test Philosophy
- Opaque-box, requirement-driven. Derives from ORIGINAL_REQUEST.md and user specifications without coupling to internal class details.
- Methodology: Category-Partition + Boundary Value Analysis + Pairwise Combinatorial + Real-World Workload Testing.
- Strict Zero-Tolerance: No hardcoding, no mock/stub execution, all tests fail loudly.

## Feature Inventory
| # | Feature | Source (requirement) | Tier 1 | Tier 2 | Tier 3 | Tier 4 |
|---|---------|---------------------|:------:|:------:|:------:|:------:|
| 1 | Multi-Device CUDAGuard | ORIGINAL_REQUEST §R1 | 5 | 5 | ✓ | ✓ |
| 2 | Stream Safety & Device Binding | ORIGINAL_REQUEST §R1 | 5 | 5 | ✓ | ✓ |
| 3 | Cross-Device Argument Check | ORIGINAL_REQUEST §R1 | 5 | 5 | ✓ | ✓ |
| 4 | Column-Parallel INT4 MLP | ORIGINAL_REQUEST §R2 | 5 | 5 | ✓ | ✓ |
| 5 | Row-Parallel INT4 MLP | ORIGINAL_REQUEST §R2 | 5 | 5 | ✓ | ✓ |
| 6 | Dual-GPU Peer All-Reduce | ORIGINAL_REQUEST §R2 | 5 | 5 | ✓ | ✓ |
| 7 | Column/Row Parallel Attention | ORIGINAL_REQUEST §R2 | 5 | 5 | ✓ | ✓ |
| 8 | Inter-Layer Pipeline Parallelism | ORIGINAL_REQUEST §R2 | 5 | 5 | ✓ | ✓ |
| 9 | TP vs PP Empirical Benchmark | ORIGINAL_REQUEST §R2 | 5 | 5 | ✓ | ✓ |
| 10 | CPMultiGPUInferenceScope | ORIGINAL_REQUEST §R2 | 5 | 5 | ✓ | ✓ |
| 11 | 7B Model Dual-GPU Validation | ORIGINAL_REQUEST §R3 | 5 | 5 | ✓ | ✓ |
| 12 | Relative Error <= 0.1% | ORIGINAL_REQUEST §R3 | 5 | 5 | ✓ | ✓ |
| 13 | Peak VRAM < 14.0 GB | ORIGINAL_REQUEST §R3 | 5 | 5 | ✓ | ✓ |
| 14 | 100-Step Zero-Crash Decode | ORIGINAL_REQUEST §Acceptance | 5 | 5 | ✓ | ✓ |

## Test Architecture
- Test runner: `python3 tests/test_e2e_multigpu_suite.py`
- Pass/Fail semantics: Non-zero exit code on any failure (`sys.exit(1)`).
- Input format: Synthetic random activations, real token IDs, and quantized weights.
- Expected output: Numerically exact or relative error <= 0.1% vs FP16 baseline, zero memory faults.

## Real-World Application Scenarios (Tier 4)
| # | Scenario | Features Exercised | Complexity |
|---|----------|--------------------|------------|
| 1 | Qwen2.5-Math-7B 2048-token context prefill forward pass | F1-F7, F10-F13 | High |
| 2 | Qwen2.5-Math-7B 100-step autoregressive decode loop | F1-F7, F10-F14 | High |
| 3 | AMC 12 math reasoning prompt rollout under TP=2 | F4-F7, F10-F12 | High |
| 4 | AIME competition proof prompt rollout under PP=2 | F8, F10-F12 | High |
| 5 | Dynamic scope entry/exit state restoration during generation | F10, F14 | Medium |

## Coverage Thresholds
- Tier 1: >= 5 test cases per feature (Happy-path isolation)
- Tier 2: >= 5 test cases per feature (Boundaries, single-token, max-token, device edge cases)
- Tier 3: Pairwise coverage across feature interactions (TP + INT4, PP + INT4, TP + KV Cache)
- Tier 4: >= 5 realistic end-to-end application scenarios
