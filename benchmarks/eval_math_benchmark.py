#!/usr/bin/env python3
"""External Math Benchmark Evaluation Suite.

Evaluates base Qwen2.5-Math-1.5B vs Cold-Start LoRA SFT model on held-out contest math:
1. 5-Tag Format Adherence (<explore>, <conjecture>, <test_edge_cases>, <lemma_isolate>, <formal_proof>).
2. Symbolic Mathematical Accuracy via SymPy exact match on \boxed{} solutions.
3. Base Degradation Check: Evaluates basic arithmetic/algebra sanity checks to prevent forgetting.
4. Best-of-16 Rejection Sampling Yield for Phase 2 RL viability.
5. Authentic Kernel Throughput & VRAM measurement with torch.cuda.synchronize().
"""

import argparse
import json
import os
import re
import sys
import time
from typing import Dict, List, Tuple, Any, Optional

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "src"))

import ast
import torch
import sympy
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)
from peft import PeftModel

from src.rope_scaling import configure_rope_scaling

try:
    import t4_kernels
    HAS_T4_KERNELS = True
except ImportError:
    HAS_T4_KERNELS = False

from src.hybrid_linear import quantize_weight_sym_int4, is_mlp_module, is_attention_module

SYSTEM_PROMPT = (
    "You are an expert mathematician and theorem prover. "
    "Reason step by step using strict tagged blocks: <explore>, <conjecture>, "
    "<test_edge_cases>, <lemma_isolate>, and <formal_proof>. "
    "End your formal proof with the final answer in \\boxed{}."
)

TAG_NAMES = ["explore", "conjecture", "test_edge_cases", "lemma_isolate", "formal_proof"]
TAG_ORDER = [
    ("explore", r"<explore>(.*?)</explore>"),
    ("conjecture", r"<conjecture>(.*?)</conjecture>"),
    ("test_edge_cases", r"<test_edge_cases>(.*?)</test_edge_cases>"),
    ("lemma_isolate", r"<lemma_isolate>(.*?)</lemma_isolate>"),
    ("formal_proof", r"<formal_proof>(.*?)</formal_proof>")
]


def check_5tag_adherence(text: str) -> Dict[str, Any]:
    """Verifies strict sequential adherence and non-empty content of all 5 reasoning tags."""
    tags_present = {}
    is_strictly_ordered = True

    for tag in TAG_NAMES:
        open_c = text.count(f"<{tag}>")
        close_c = text.count(f"</{tag}>")
        tags_present[tag] = (open_c == 1 and close_c == 1)

    all_present = all(tags_present.values())
    if not all_present:
        return {
            "adherent": False,
            "all_tags_present": all_present,
            "strictly_ordered": False,
            "tags": tags_present
        }

    last_pos = -1
    for tag in TAG_NAMES:
        open_pos = text.find(f"<{tag}>")
        close_pos = text.find(f"</{tag}>")
        if not (last_pos <= open_pos < close_pos):
            is_strictly_ordered = False
            break

        content = text[open_pos + len(f"<{tag}>"):close_pos]
        if not content.strip():
            is_strictly_ordered = False
            break

        # Disallow nested or duplicate tags inside block
        for other_tag in TAG_NAMES:
            if f"<{other_tag}>" in content or f"</{other_tag}>" in content:
                is_strictly_ordered = False
                break
        if not is_strictly_ordered:
            break
        last_pos = close_pos + len(f"</{tag}>")

    return {
        "adherent": all_present and is_strictly_ordered,
        "all_tags_present": all_present,
        "strictly_ordered": is_strictly_ordered,
        "tags": tags_present
    }


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extracts the innermost or outermost contents of \\boxed{...}."""
    patterns = [
        r"\boxed{",
        "boxed{",
        r"\x08oxed{",
        "\x08oxed{",
    ]
    idx = -1
    lead_len = 0
    for pat in patterns:
        cur_idx = text.rfind(pat)
        if cur_idx > idx:
            idx = cur_idx
            lead_len = len(pat)

    if idx == -1:
        return None
    idx += lead_len

    depth = 1
    chars = []
    for c in text[idx:]:
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                break
        chars.append(c)

    if depth == 0:
        return "".join(chars).strip()
    return None


def clean_latex_math(expr: str) -> str:
    """Normalizes LaTeX math expressions for SymPy parsing."""
    s = expr.strip()
    s = s.replace("$", "")
    # If expression is written as variable assignment (e.g., "x = 42"), isolate RHS
    m_eq = re.match(r"^[a-zA-Z]\s*=\s*(.+)$", s)
    if m_eq:
        s = m_eq.group(1).strip()

    # Handle nested \sqrt{...} -> sqrt(...)
    while r"\sqrt{" in s:
        idx = s.find(r"\sqrt{")
        depth = 0
        end_idx = -1
        for i in range(idx + 5, len(s)):
            if s[i] == "{":
                depth += 1
            elif s[i] == "}":
                depth -= 1
                if depth == 0:
                    end_idx = i
                    break
        if end_idx != -1:
            arg = s[idx + 6:end_idx]
            s = s[:idx] + f"sqrt({arg})" + s[end_idx + 1:]
        else:
            break

    # Handle nested \frac{...}{...} and \dfrac{...}{...} -> ((...)/(...))
    for frac_cmd in [r"\frac{", r"\dfrac{"]:
        while frac_cmd in s:
            idx = s.find(frac_cmd)
            start_num = idx + len(frac_cmd) - 1
            depth = 0
            end_num = -1
            for i in range(start_num, len(s)):
                if s[i] == "{":
                    depth += 1
                elif s[i] == "}":
                    depth -= 1
                    if depth == 0:
                        end_num = i
                        break
            if end_num == -1 or end_num + 1 >= len(s) or s[end_num + 1] != "{":
                break
            start_den = end_num + 1
            depth = 0
            end_den = -1
            for i in range(start_den, len(s)):
                if s[i] == "{":
                    depth += 1
                elif s[i] == "}":
                    depth -= 1
                    if depth == 0:
                        end_den = i
                        break
            if end_den == -1:
                break
            num = s[start_num + 1:end_num]
            den = s[start_den + 1:end_den]
            s = s[:idx] + f"(({num})/({den}))" + s[end_den + 1:]

    s = s.replace("\\pi", "pi")
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    s = re.sub(r"\\[,;! ]", "", s)
    s = s.replace("\\left", "").replace("\\right", "")
    # Handle implicit multiplication: e.g. "36pi" -> "36*pi", "2(x+1)" -> "2*(x+1)"
    s = re.sub(r"(\d+)\s*([a-zA-Z\(])", r"\1*\2", s)
    s = re.sub(r"(\))\s*(\()", r"\1*\2", s)
    s = re.sub(r"(\))\s*([a-zA-Z0-9])", r"\1*\2", s)
    return s.strip()


SAFE_AST_NODES = {
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Pow, ast.USub, ast.UAdd,
    ast.Name, ast.Call, ast.Tuple, ast.List, ast.Load
}
ALLOWED_MATH_FUNCS = {"sqrt", "pi", "sin", "cos", "tan", "log", "exp", "abs", "Rational", "Integer", "Matrix", "frac"}
BANNED_NAMES = {
    "eval", "exec", "open", "__import__", "compile", "globals", "locals",
    "getattr", "setattr", "delattr", "hasattr", "input", "print", "breakpoint",
    "help", "exit", "quit", "system", "popen", "spawn", "os", "sys", "subprocess", "pathlib"
}


def is_safe_math_expression(expr_str: str) -> bool:
    """Verifies that an expression contains only safe mathematical syntax to prevent RCE."""
    try:
        tree = ast.parse(expr_str, mode="eval")
    except Exception:
        return False
    for node in ast.walk(tree):
        if type(node) not in SAFE_AST_NODES:
            return False
        if isinstance(node, ast.Name):
            if node.id.startswith("_") or node.id in BANNED_NAMES:
                return False
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                return False
            if node.func.id not in ALLOWED_MATH_FUNCS:
                return False
    return True


def verify_math_answer_sympy(pred_raw: Optional[str], gold_raw: str) -> bool:
    """Symbolic exact match verification via SymPy with AST sandboxing."""
    if not pred_raw:
        return False

    pred_clean = clean_latex_math(pred_raw)
    gold_clean = clean_latex_math(gold_raw)

    # Fast path: identical strings
    if pred_clean.lower() == gold_clean.lower():
        return True

    # Strict numeric check when both are single scalar numbers
    num_re = r"^[+-]?\d+(?:\.\d+)?$"
    if re.fullmatch(num_re, pred_clean) and re.fullmatch(num_re, gold_clean):
        try:
            return abs(float(pred_clean) - float(gold_clean)) < 1e-6
        except Exception:
            pass

    # Sandboxed SymPy symbolic equivalence check
    if is_safe_math_expression(pred_clean) and is_safe_math_expression(gold_clean):
        try:
            sym_pred = sympy.sympify(pred_clean)
            sym_gold = sympy.sympify(gold_clean)
            diff = sympy.simplify(sym_pred - sym_gold)
            if diff == 0 or diff.is_zero:
                return True
            if diff.is_number and abs(complex(diff.evalf())) < 1e-6:
                return True
        except Exception:
            pass

    return False


import contextlib


class CPHybridInferenceScope:
    """Dispatches W4A16 GEMV kernel on MLP linear layers during inference."""

    def __init__(self, model, group_size=128):
        self.model = model
        self.group_size = group_size
        self.orig_forwards = {}
        self.Q = {}

    def __enter__(self):
        if not HAS_T4_KERNELS or not torch.cuda.is_available():
            return self

        if not self.Q:
            print("  [CP-Hybrid] Pre-quantizing MLP weights to INT4...")
            for name, mod in self.model.named_modules():
                if isinstance(mod, torch.nn.Linear) and is_mlp_module(name) and not is_attention_module(name):
                    self.Q[mod] = quantize_weight_sym_int4(mod.weight.data, group_size=self.group_size)

        for mod, (packed, scales, zp, g_size) in self.Q.items():
            self.orig_forwards[mod] = mod.forward

            def make_patched(m, p, s, z, g):
                def patched_forward(x):
                    x_shape = x.shape
                    orig_dtype = x.dtype
                    x_flat = x.reshape(-1, x_shape[-1])
                    M = x_flat.shape[0]

                    if M <= 2:
                        if x_flat.dtype != torch.float16:
                            x_flat = x_flat.half()
                        out = t4_kernels.fused_w4a16_gemm_u4(x_flat, p, s, z, g)
                        if m.bias is not None:
                            out = out + m.bias
                        if orig_dtype != torch.float16:
                            out = out.to(orig_dtype)
                    else:
                        out = torch.nn.functional.linear(x_flat, m.weight, m.bias)

                    return out.reshape(*x_shape[:-1], out.shape[-1])
                return patched_forward

            mod.forward = make_patched(mod, packed, scales, zp, g_size)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for mod, orig_fwd in self.orig_forwards.items():
            mod.forward = orig_fwd
        self.orig_forwards.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class StopOnFormalProof(StoppingCriteria):
    def __init__(self, tokenizer, prompt_len: int = 0):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        new_tokens = input_ids[0, self.prompt_len:]
        if new_tokens.shape[0] < 8:
            return False
        tail = new_tokens[-20:]
        text = self.tokenizer.decode(tail, skip_special_tokens=False)
        return "</formal_proof>" in text or "<|im_end|>" in text


def evaluate_model_on_dataset(
    model,
    tokenizer,
    dataset: List[Dict[str, Any]],
    num_samples_per_problem: int = 1,
    max_new_tokens: int = 1024,
    use_kernels: bool = False,
    device: str = "cpu"
) -> Dict[str, Any]:
    """Runs evaluation and collects metrics."""
    results = []
    total_tokens_generated = 0
    total_generation_time = 0.0

    scope = CPHybridInferenceScope(model) if use_kernels else contextlib.nullcontext()

    # Build comprehensive EOS tokens
    eos_token_ids = [tokenizer.eos_token_id]
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end_id, int) and im_end_id not in eos_token_ids:
        eos_token_ids.append(im_end_id)

    with scope:
        for idx, item in enumerate(dataset):
            prompt = (
                f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{item['problem']}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_len = inputs["input_ids"].shape[1]

            problem_rollouts = []
            any_correct = False

            for s_idx in range(num_samples_per_problem):
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()

                stopping_criteria = StoppingCriteriaList([StopOnFormalProof(tokenizer, prompt_len=input_len)])

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=(num_samples_per_problem > 1),
                    temperature=0.7 if num_samples_per_problem > 1 else 1.0,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    eos_token_id=eos_token_ids,
                    stopping_criteria=stopping_criteria,
                )

                if device == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()

                gen_tokens = outputs[0][input_len:]
                num_tokens = len(gen_tokens)
                total_tokens_generated += num_tokens
                total_generation_time += (t1 - t0)

                completion_text = tokenizer.decode(gen_tokens, skip_special_tokens=False)

                adherence = check_5tag_adherence(completion_text)
                pred_boxed = extract_boxed_answer(completion_text)
                is_correct = verify_math_answer_sympy(pred_boxed, item["ground_truth"])
                if is_correct:
                    any_correct = True

                problem_rollouts.append({
                    "sample_idx": s_idx,
                    "adherence": adherence,
                    "pred_boxed": pred_boxed,
                    "is_correct": is_correct,
                    "num_tokens": num_tokens,
                    "tok_s": num_tokens / max(t1 - t0, 1e-4),
                    "completion_text": completion_text,
                    "completion_snippet": completion_text
                })

            results.append({
                "problem_id": item.get("id", f"prob_{idx}"),
                "ground_truth": item["ground_truth"],
                "pass_at_1": problem_rollouts[0]["is_correct"],
                "best_of_n": any_correct,
                "rollouts": problem_rollouts
            })

            status_icon = "PASS" if any_correct else "FAIL"
            tok_s = problem_rollouts[0]["tok_s"]
            num_tokens = problem_rollouts[0]["num_tokens"]
            last_rollout = problem_rollouts[0]
            pred_val = str(last_rollout['pred_boxed'] or 'None')
            gold_val = str(item['ground_truth'])
            tag_cnt = sum(1 for t in TAG_NAMES if f"<{t}>" in last_rollout['completion_text'] and f"</{t}>" in last_rollout['completion_text'])
            print(f"  [{idx+1:2d}/{len(dataset):2d}] {item.get('id', 'prob')[:18]:18s} | {status_icon:4s} | Pred: {pred_val[:10]:10s} (Gold: {gold_val[:10]:10s}) | Tags: {tag_cnt}/5 | {num_tokens:4d} toks | {tok_s:4.1f} t/s", flush=True)

    adherence_rate = sum(r["rollouts"][0]["adherence"]["adherent"] for r in results) / max(len(results), 1)
    pass_1_rate = sum(r["pass_at_1"] for r in results) / max(len(results), 1)
    best_of_n_rate = sum(r["best_of_n"] for r in results) / max(len(results), 1)
    avg_tok_s = total_tokens_generated / max(total_generation_time, 1e-4)

    return {
        "num_problems": len(results),
        "adherence_rate": adherence_rate,
        "pass_1_rate": pass_1_rate,
        "best_of_n_rate": best_of_n_rate,
        "avg_tok_s": avg_tok_s,
        "total_tokens": total_tokens_generated,
        "total_time_s": total_generation_time,
        "results": results
    }


def parse_args():
    parser = argparse.ArgumentParser(description="External Math Benchmark Evaluation")
    parser.add_argument("--base_model", "--model_name_or_path", "--model_path", dest="base_model", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--adapter_path", "--lora_path", dest="adapter_path", default=None, help="Path to fine-tuned LoRA adapter checkpoint")
    parser.add_argument("--eval_dataset", "--benchmark_path", "--eval_path", dest="eval_dataset", default="data/external_math_eval.json")
    parser.add_argument("--num_samples", type=int, default=1, help="Samples per problem (e.g. 16 for Best-of-16)")
    parser.add_argument("--max_tokens", "--max_new_tokens", dest="max_tokens", type=int, default=1024)
    parser.add_argument("--use_kernels", action="store_true", help="Enable CP-Hybrid W4A16 GEMV kernel")
    parser.add_argument("--output_file", "--output_path", dest="output_file", default="results/external_eval_summary.json")
    parser.add_argument("--device", default=None, help="Target device (default: cuda if available)")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load base model in 4-bit NF4 (QLoRA) for low-memory environments")
    parser.add_argument("--dry_run", action="store_true", help="Dry run on mock/miniature config for pipeline verification")
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() and not args.dry_run else "cpu")
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    print("=" * 75)
    print("  EXTERNAL COMPETITION MATH BENCHMARK EVALUATION (AMC 12 / AIME)")
    print(f"Device: {device} | Use Kernels: {args.use_kernels} | Samples: {args.num_samples}")
    print("=" * 75)

    with open(args.eval_dataset, "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    contest_probs = bench_data.get("contest_problems", [])
    degrad_probs = bench_data.get("degradation_checks", bench_data.get("degradation_problems", []))

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.dry_run:
        print("[DRY RUN] Instantiating miniature test model from configuration...")
        cfg = AutoConfig.from_pretrained(args.base_model)
        cfg.num_hidden_layers = 2
        cfg.hidden_size = 128
        cfg.intermediate_size = 256
        cfg.num_attention_heads = 4
        cfg.num_key_value_heads = 2
        model = AutoModelForCausalLM.from_config(cfg)
        contest_probs = contest_probs[:2]
        degrad_probs = degrad_probs[:2]
    else:
        print(f"Loading Base Model: {args.base_model}...")
        bnb_config = None
        if args.load_in_4bit:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            quantization_config=bnb_config,
            device_map="auto" if device == "cuda" else None
        )
        if args.adapter_path and os.path.exists(args.adapter_path):
            print(f"Applying LoRA Adapter from: {args.adapter_path}...")
            model = PeftModel.from_pretrained(model, args.adapter_path)
            if args.use_kernels:
                print("Merging LoRA adapter into base weights for INT4 kernel execution...")
                model = model.merge_and_unload()
        model.eval()

    if not hasattr(model, "hf_device_map"):
        model.to(device)

    print(f"--> Evaluating Contest Problems (N={len(contest_probs)})...")
    contest_metrics = evaluate_model_on_dataset(
        model, tokenizer, contest_probs,
        num_samples_per_problem=args.num_samples,
        max_new_tokens=args.max_tokens if not args.dry_run else 32,
        use_kernels=args.use_kernels,
        device=device
    )

    print(f"--> Evaluating Degradation Checks (N={len(degrad_probs)})...")
    degrad_metrics = evaluate_model_on_dataset(
        model, tokenizer, degrad_probs,
        num_samples_per_problem=1,
        max_new_tokens=args.max_tokens if not args.dry_run else 32,
        use_kernels=False,
        device=device
    )

    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2) if device == "cuda" else 0.0

    print("\n" + "=" * 75)
    print("  EVALUATION RESULTS REPORT")
    print("=" * 75)
    print(f"Model: {args.base_model} (Adapter: {args.adapter_path})")
    print(f"Contest Problems: {contest_metrics['num_problems']}")
    print(f"  5-Tag Format Adherence: {contest_metrics['adherence_rate'] * 100:.1f}%")
    print(f"  Pass@1 Symbolic Accuracy: {contest_metrics['pass_1_rate'] * 100:.1f}%")
    print(f"  Best-of-{args.num_samples} Yield: {contest_metrics['best_of_n_rate'] * 100:.1f}%")
    print(f"  Throughput: {contest_metrics['avg_tok_s']:.1f} tok/s")
    print(f"Degradation Check Pass Rate: {degrad_metrics['pass_1_rate'] * 100:.1f}% ({len(degrad_probs)} problems)")
    print(f"Peak VRAM: {peak_vram_mb:.1f} MB")
    print("=" * 75)

    summary = {
        "timestamp": time.time(),
        "model": args.base_model,
        "adapter": args.adapter_path,
        "contest_metrics": contest_metrics,
        "degradation_metrics": degrad_metrics,
        "peak_vram_mb": peak_vram_mb,
        "device": device
    }

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Detailed results logged to: {args.output_file}")


if __name__ == "__main__":
    main()
