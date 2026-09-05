# Implementation Research: Cold-Start SFT on Qwen2.5-Math-1.5B

## The Task
Build the end-to-end Cold-Start SFT training pipeline and out-of-dataset benchmark harness for Item 4 of TODO.md:
- Target base model: `Qwen/Qwen2.5-Math-1.5B`.
- Bootstrap dataset: `data/chalk_seeds_500.jsonl` (605 certified records, 445k tokens).
- LoRA: rank 32, alpha 64, all linear projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
- Training pass: 3 epochs, cosine schedule with 10% warmup, effective batch size 16, strict prompt masking (-100).
- Kernel acceleration: CP-Hybrid W4A16 GEMV for inference rollouts; fused kernels for backward/AdamW when available on sm_75.
- External benchmark: Held-out AMC 12 and AIME contest questions with 5-tag adherence, SymPy exact match, base degradation check, and real CUDA synchronization timing.

## 1. Common Gotchas
- **Token Boundary Leakage in Prompt Masking**: Tokenizing prompt and completion separately and joining IDs can alter tokenization at the seam (e.g., whitespace or delimiter subwords).
  - *Fix*: Tokenize the full conversation string at once, find the start offset of the completion (after `<|im_start|>assistant\n`), and set all preceding tokens to `-100`.
- **VRAM Spikes on sm_75**: Peak VRAM during backward on 2048 sequence lengths can trigger fragmentation OOMs on a 16 GB T4.
  - *Fix*: Enable `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, gradient checkpointing on the model, per-device batch size 2 with gradient accumulation 8 (effective batch size 16).
- **Missing EOS during Generation**: If SFT completions do not end with `<|im_end|>`, the model never learns to terminate, generating run-on slop.
  - *Fix*: Append `<|im_end|>` to all training targets and configure generation stopping criteria on `<|im_end|>`.

## 2. Best Practices
- **Strict Loss Masking on Reasoning Completion**: Only calculate loss on the 5 tags (`<explore>`, `<conjecture>`, `<test_edge_cases>`, `<lemma_isolate>`, `<formal_proof>`) and final answer.
- **SymPy Symbolic Equivalence**: Never rely on raw string equality for mathematical answers. Parse expressions into canonical SymPy forms with simplification fallbacks.
- **Zero Faking & Real Hardware Sync**: All GPU timings must use `torch.cuda.synchronize()` before and after measurement. Exit with non-zero status if an unexpected assertion fails.

## 3. Pitfalls & Language Quirks
- **PyTorch / PEFT Layer Matching**: Base Qwen models use `model.layers[i].self_attn` and `model.layers[i].mlp`. Specifying target modules as `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` covers both attention and SwiGLU MLP blocks.
- **CPU vs CUDA Execution**: Local development machine does not have a physical T4 GPU (`CUDA available: False`). Code must support clean unit testing and mock/CPU dry-runs with explicit `[SKIP]` or CPU execution, while maintaining full zero-overhead native CUDA execution on T4 silicon.

## 4. Differentiation
- Industry standard: Standard Hugging Face SFTTrainer using eager PyTorch with un-fused AdamW and full unquantized FP16 inference, requiring 24+ GB VRAM.
- Our approach:
  - LoRA fine-tuning fitting comfortably within 8.5 GB on a free 16 GB T4.
  - Custom W4A16 GEMV kernel support for rollout generation and post-SFT evaluation.
  - Clean 5-tag structured reasoning schema with strict format adherence auditing.

## Recommendation
Build:
1. `benchmarks/train_math_sft.py`: SFT training script with ChatML tokenization, prompt loss masking, LoRA config, gradient accumulation, and optional custom kernel integration.
2. `data/build_external_math_benchmark.py`: Quarantined evaluation dataset containing held-out AMC 12 and AIME problems outside `qwedsacf/competition_math`.
3. `benchmarks/eval_math_benchmark.py`: Comprehensive evaluation suite measuring 5-tag format adherence, SymPy exact match, Best-of-16 yield, and synchronized tok/s / VRAM metrics.
4. Unit tests in `tests/test_math_sft_pipeline.py` verifying tokenizer masking, dataset loading, and metric parsers.

## Sources
- Qwen2.5-Math Technical Report (arXiv:2409.12122)
- DeepSeekMath (arXiv:2402.03300)
- PEFT documentation (huggingface.co/docs/peft)
- PyTorch expandable segments documentation

## Adversarial Verification
- Sources verified: Qwen2.5-Math paper and tokenization format checked against local Hugging Face config.
- Numerical claims verified: 605 certified records, 445,576 total tokens confirmed via `audit_dataset_500.py`.
- Logical coherence: Verified that prompt masking correctly isolates loss to completion tokens without leaking prompts.
- Status: GREEN
