import os
import sys
import time
import json
import argparse
import torch
import transformers

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "src"))

from hybrid_linear import CPHybridScope
from fused_speculative_engine import FusedSpeculativeServingEngine

TEST_PROMPTS = [
    "Explain the difference between a mutex and a semaphore in operating systems.",
    "Solve step-by-step: If a train travels at 60 mph for 2.5 hours, how far does it travel?",
]


def run_benchmark(
    target_model_id: str = "Qwen/Qwen2.5-3B-Instruct",
    draft_model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    max_tokens: int = 24,
    lookahead_k: int = 2,
    load_in_4bit: bool = False,
):
    print("=" * 85)
    print("   Tesla T4 Fused CUDA-Graph Speculative Serving Benchmark")
    print("=" * 85)
    print(f"Target Model: {target_model_id} ({'4-bit NF4' if load_in_4bit else 'FP16'})")
    print(f"Draft Model:  {draft_model_id} (INT4 CP-Hybrid)")
    print(f"Lookahead K:  {lookahead_k}")
    print(f"Max Tokens:   {max_tokens}")
    print("-" * 85)

    tokenizer = transformers.AutoTokenizer.from_pretrained(target_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading Target Model ({target_model_id})...")
    if load_in_4bit:
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        target_model = transformers.AutoModelForCausalLM.from_pretrained(
            target_model_id,
            quantization_config=bnb_config,
            device_map="cuda",
        ).eval()
    else:
        target_model = transformers.AutoModelForCausalLM.from_pretrained(
            target_model_id,
            torch_dtype=torch.float16,
            device_map="cuda",
        ).eval()

    print("Loading Draft Model in FP16...")
    draft_model = transformers.AutoModelForCausalLM.from_pretrained(
        draft_model_id,
        torch_dtype=torch.float16,
        device_map="cuda",
    ).eval()

    print("Quantizing Draft Model MLPs with CP-Hybrid (INT4 group=128)...")
    draft_scope = CPHybridScope(draft_model, group_size=128)
    draft_scope.__enter__()

    # 1. Target Baseline
    print(f"\n[1/2] Benchmarking Target Model ({target_model_id}) Autoregressive Baseline...")
    target_rates = []
    for idx, prompt in enumerate(TEST_PROMPTS):
        formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = tokenizer(formatted, return_tensors="pt").input_ids.cuda()
        t0 = time.perf_counter()
        with torch.no_grad():
            gen = target_model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        num_tok = gen.shape[-1] - input_ids.shape[-1]
        speed = num_tok / dt if dt > 0 else 0.0
        target_rates.append(speed)
        print(f"  Prompt {idx+1}: {speed:.1f} tok/s ({dt*1000/num_tok:.1f} ms/tok)")

    baseline_speed = sum(target_rates) / len(target_rates)
    print(f">> Target Baseline: {baseline_speed:.1f} tok/s")

    # 2. Eager Speculative Engine
    print("\n[2/3] Benchmarking Eager Speculative Engine (Python dispatch)...")
    eager_engine = FusedSpeculativeServingEngine(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        use_cuda_graphs=False,
    )
    eager_rates = []
    eager_alphas = []
    eager_tok_steps = []
    for idx, prompt in enumerate(TEST_PROMPTS):
        formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = tokenizer(formatted, return_tensors="pt").input_ids.cuda()
        _, stats = eager_engine.generate(input_ids, max_new_tokens=max_tokens, lookahead_k=lookahead_k)
        eager_rates.append(stats["tokens_per_sec"])
        eager_alphas.append(stats["acceptance_rate_pct"])
        eager_tok_steps.append(stats["avg_tokens_per_target_step"])
        print(f"  Prompt {idx+1}: {stats['tokens_per_sec']:.1f} tok/s | Alpha: {stats['acceptance_rate_pct']:.1f}% | Tok/Step: {stats['avg_tokens_per_target_step']:.2f}")

    eager_speed = sum(eager_rates) / len(eager_rates)
    eager_alpha = sum(eager_alphas) / len(eager_alphas)
    eager_tok_step = sum(eager_tok_steps) / len(eager_tok_steps)
    print(f">> Eager Speculative: {eager_speed:.1f} tok/s ({eager_speed/baseline_speed:.2f}x baseline)")

    # 3. CUDA-Graph Speculative Engine
    print("\n[3/3] Benchmarking CUDA-Graph Accelerated Speculative Engine...")
    graph_engine = FusedSpeculativeServingEngine(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        use_cuda_graphs=True,
    )
    print("Capturing CUDA Graph for draft forward passes...")
    graph_engine.warmup_and_capture(sample_prompt_len=16)

    graph_rates = []
    graph_alphas = []
    graph_tok_steps = []
    for idx, prompt in enumerate(TEST_PROMPTS):
        formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = tokenizer(formatted, return_tensors="pt").input_ids.cuda()
        _, stats = graph_engine.generate(input_ids, max_new_tokens=max_tokens, lookahead_k=lookahead_k)
        graph_rates.append(stats["tokens_per_sec"])
        graph_alphas.append(stats["acceptance_rate_pct"])
        graph_tok_steps.append(stats["avg_tokens_per_target_step"])
        print(f"  Prompt {idx+1}: {stats['tokens_per_sec']:.1f} tok/s | Alpha: {stats['acceptance_rate_pct']:.1f}% | Tok/Step: {stats['avg_tokens_per_target_step']:.2f}")

    graph_speed = sum(graph_rates) / len(graph_rates)
    graph_alpha = sum(graph_alphas) / len(graph_alphas)
    graph_tok_step = sum(graph_tok_steps) / len(graph_tok_steps)
    print(f">> CUDA-Graph Speculative: {graph_speed:.1f} tok/s ({graph_speed/baseline_speed:.2f}x baseline)")

    # Summary
    results = {
        "target_model": target_model_id,
        "draft_model": draft_model_id,
        "baseline_tok_sec": baseline_speed,
        "eager_speculative": {
            "tok_sec": eager_speed,
            "speedup": eager_speed / baseline_speed,
            "acceptance_pct": eager_alpha,
            "tok_per_target_step": eager_tok_step,
        },
        "cuda_graph_speculative": {
            "tok_sec": graph_speed,
            "speedup": graph_speed / baseline_speed,
            "acceptance_pct": graph_alpha,
            "tok_per_target_step": graph_tok_step,
        },
    }

    out_file = os.path.join(REPO_DIR, "results", "fused_speculative_benchmark.json")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 85)
    print(f"{'Engine Configuration':<30} | {'Tokens/sec':<12} | {'Speedup vs 7B':<16} | {'Tok/Step':<10}")
    print("-" * 85)
    print(f"{'Target Baseline':<30} | {baseline_speed:10.1f}   | {'1.00x':<16} | {'1.00':<10}")
    print(f"{'Eager Speculative (K=' + str(lookahead_k) + ')':<30} | {eager_speed:10.1f}   | {eager_speed/baseline_speed:14.2f}x | {eager_tok_step:8.2f}")
    print(f"{'CUDA Graph Speculative (K=' + str(lookahead_k) + ')':<30} | {graph_speed:10.1f}   | {graph_speed/baseline_speed:14.2f}x | {graph_tok_step:8.2f}")
    print("=" * 85)
    print(f"Results saved to: {out_file}")

    draft_scope.__exit__(None, None, None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--draft-model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--lookahead-k", type=int, default=2)
    parser.add_argument("--load-in-4bit", action="store_true", default=False)
    args = parser.parse_args()

    run_benchmark(
        target_model_id=args.target_model,
        draft_model_id=args.draft_model,
        max_tokens=args.max_tokens,
        lookahead_k=args.lookahead_k,
        load_in_4bit=args.load_in_4bit,
    )
