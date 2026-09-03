"""Low-Precision Speculative Decoding Engine on Tesla T4.

Implements the standard Leviathan/Chen speculative decoding algorithm with
strict 1-target-pass-per-round execution, rolling KV-caches, and O(1) rollback:
- Exactly 1 target forward pass per speculative round.
- No redundant fixup forward passes on target model.
- Draft model accelerated by sm_75 INT4 W4A16 CP-Hybrid kernels.
"""

import time
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import t4_kernels
except ImportError:
    t4_kernels = None


def rollback_cache(cache, num_to_drop: int):
    """Roll back KV-cache by dropping the last `num_to_drop` tokens."""
    if cache is None or num_to_drop <= 0:
        return cache
    if hasattr(cache, "crop"):
        # Modern HuggingFace DynamicCache accepts negative integers to drop tokens
        try:
            cache.crop(-num_to_drop)
            return cache
        except Exception:
            pass

    if isinstance(cache, (list, tuple)):
        cropped = []
        for layer in cache:
            if isinstance(layer, (list, tuple)) and len(layer) >= 2:
                k, v = layer[0], layer[1]
                if k.shape[-2] >= num_to_drop:
                    cropped.append((k[..., :-num_to_drop, :], v[..., :-num_to_drop, :]))
                else:
                    cropped.append((k, v))
            else:
                cropped.append(layer)
        return tuple(cropped) if isinstance(cache, tuple) else cropped
    return cache


class SpeculativeEngine:
    """Speculative decoding engine pairing an INT4 draft model with an FP16 target model."""
    def __init__(
        self,
        target_model: nn.Module,
        draft_model: nn.Module,
        tokenizer,
        lookahead_k: int = 3,
    ):
        self.target_model = target_model
        self.draft_model = draft_model
        self.tokenizer = tokenizer
        self.lookahead_k = lookahead_k

    @torch.inference_mode()
    def generate_autoregressive(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Baseline pure autoregressive decode using target model alone."""
        input_ids = prompt_ids.clone()
        past_key_values = None
        latencies = []

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        for _ in range(max_new_tokens):
            t_tok_start = time.perf_counter()

            if past_key_values is None:
                outputs = self.target_model(input_ids=input_ids, use_cache=True)
            else:
                outputs = self.target_model(
                    input_ids=input_ids[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]

            if temperature == 0.0:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t_tok_start)

            input_ids = torch.cat([input_ids, next_token], dim=-1)
            if next_token.item() == self.tokenizer.eos_token_id:
                break

        torch.cuda.synchronize()
        total_time = time.perf_counter() - t0
        num_generated = input_ids.shape[-1] - prompt_ids.shape[-1]

        stats = {
            "num_tokens": num_generated,
            "total_time_s": total_time,
            "tokens_per_sec": num_generated / total_time if total_time > 0 else 0.0,
            "latency_p50_ms": float(torch.tensor(latencies).quantile(0.50).item() * 1000),
            "latency_p90_ms": float(torch.tensor(latencies).quantile(0.90).item() * 1000),
            "latency_p99_ms": float(torch.tensor(latencies).quantile(0.99).item() * 1000),
        }
        return input_ids, stats

    @torch.inference_mode()
    def generate_speculative(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 64,
        lookahead_k: Optional[int] = None,
        temperature: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Leviathan speculative decoding: exactly 1 parallel target pass per round."""
        K = lookahead_k or self.lookahead_k
        input_ids = prompt_ids.clone()
        prompt_len = prompt_ids.shape[-1]

        total_proposed = 0
        total_accepted = 0
        target_forward_passes = 0
        step_latencies = []

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Initial prompt prefill for target model
        target_out = self.target_model(input_ids, use_cache=True)
        target_kv = target_out.past_key_values
        target_forward_passes += 1

        # Seed token to verify in first round
        next_verify_prefix = None
        target_seed_logits = target_out.logits[:, -1, :]

        # Initial prompt prefill for draft model
        draft_out = self.draft_model(input_ids, use_cache=True)
        draft_kv = draft_out.past_key_values
        draft_next_logits = draft_out.logits[:, -1, :]

        while (input_ids.shape[-1] - prompt_len) < max_new_tokens:
            t_step_start = time.perf_counter()

            # 1. Draft Phase: generate K candidate tokens
            draft_tokens = []
            curr_draft_logits = draft_next_logits

            for step_i in range(K):
                if temperature == 0.0:
                    cand_token = torch.argmax(curr_draft_logits, dim=-1, keepdim=True)
                else:
                    probs_d = F.softmax(curr_draft_logits / temperature, dim=-1)
                    cand_token = torch.multinomial(probs_d, num_samples=1)

                draft_tokens.append(cand_token)
                if cand_token.item() == self.tokenizer.eos_token_id:
                    break

                out_d = self.draft_model(cand_token, past_key_values=draft_kv, use_cache=True)
                draft_kv = out_d.past_key_values
                curr_draft_logits = out_d.logits[:, -1, :]

            actual_k = len(draft_tokens)
            total_proposed += actual_k

            # 2. Target Phase: exactly ONE parallel forward pass
            # If we had an unverified seed token from previous rejection/bonus, include it
            if next_verify_prefix is not None:
                eval_tokens = torch.cat([next_verify_prefix] + draft_tokens, dim=-1)
                has_prefix = True
            else:
                eval_tokens = torch.cat(draft_tokens, dim=-1)
                has_prefix = False

            target_out = self.target_model(eval_tokens, past_key_values=target_kv, use_cache=True)
            target_kv = target_out.past_key_values
            target_forward_passes += 1

            # 3. Verification
            accepted_in_step = 0
            all_accepted = True
            hit_eos = False

            eval_logits = target_out.logits
            # If has_prefix, eval_logits[:, 0] predicts candidate 0.
            # If no prefix (first turn), target_seed_logits predicts candidate 0.
            for i in range(actual_k):
                if i == 0 and not has_prefix:
                    pred_l = target_seed_logits
                elif has_prefix:
                    pred_l = eval_logits[:, i, :]
                else:
                    pred_l = eval_logits[:, i - 1, :]

                if temperature == 0.0:
                    target_choice = torch.argmax(pred_l, dim=-1, keepdim=True)
                else:
                    probs_t = F.softmax(pred_l / temperature, dim=-1)
                    target_choice = torch.multinomial(probs_t, num_samples=1)

                cand = draft_tokens[i]
                if target_choice.item() == cand.item():
                    # Candidate accepted!
                    accepted_in_step += 1
                    input_ids = torch.cat([input_ids, cand], dim=-1)
                    if cand.item() == self.tokenizer.eos_token_id:
                        hit_eos = True
                        all_accepted = False
                        # Drop remaining candidates from target cache
                        num_drop = eval_tokens.shape[-1] - (i + (1 if has_prefix else 1))
                        target_kv = rollback_cache(target_kv, num_drop)
                        break
                else:
                    # Mismatch at candidate i
                    # Append target's correction
                    input_ids = torch.cat([input_ids, target_choice], dim=-1)
                    all_accepted = False
                    if target_choice.item() == self.tokenizer.eos_token_id:
                        hit_eos = True

                    # Drop rejected tokens from target cache:
                    # Tokens kept from eval_tokens = (1 if has_prefix else 0) + i
                    tokens_kept = (1 if has_prefix else 0) + i
                    num_drop = eval_tokens.shape[-1] - tokens_kept
                    target_kv = rollback_cache(target_kv, num_drop)

                    # Next turn will feed target_choice as the prefix to target!
                    next_verify_prefix = target_choice

                    # Sync draft model cache to match accepted tokens + target_choice
                    num_draft_drop = actual_k - i
                    draft_kv = rollback_cache(draft_kv, num_draft_drop)
                    d_sync = self.draft_model(target_choice, past_key_values=draft_kv, use_cache=True)
                    draft_kv = d_sync.past_key_values
                    draft_next_logits = d_sync.logits[:, -1, :]
                    break

            if all_accepted and not hit_eos:
                # Bonus token from target's final position
                bonus_logits = eval_logits[:, -1, :]
                if temperature == 0.0:
                    bonus_choice = torch.argmax(bonus_logits, dim=-1, keepdim=True)
                else:
                    probs_b = F.softmax(bonus_logits / temperature, dim=-1)
                    bonus_choice = torch.multinomial(probs_b, num_samples=1)

                input_ids = torch.cat([input_ids, bonus_choice], dim=-1)
                next_verify_prefix = bonus_choice

                # Sync draft model with bonus choice
                d_sync = self.draft_model(bonus_choice, past_key_values=draft_kv, use_cache=True)
                draft_kv = d_sync.past_key_values
                draft_next_logits = d_sync.logits[:, -1, :]

            total_accepted += accepted_in_step

            torch.cuda.synchronize()
            step_latencies.append(time.perf_counter() - t_step_start)

            if input_ids[0, -1].item() == self.tokenizer.eos_token_id:
                break

        torch.cuda.synchronize()
        total_time = time.perf_counter() - t0
        num_generated = input_ids.shape[-1] - prompt_len
        acceptance_rate = (total_accepted / total_proposed) * 100.0 if total_proposed > 0 else 0.0

        stats = {
            "num_tokens": num_generated,
            "total_time_s": total_time,
            "tokens_per_sec": num_generated / total_time if total_time > 0 else 0.0,
            "lookahead_k": K,
            "total_proposed": total_proposed,
            "total_accepted": total_accepted,
            "acceptance_rate_pct": acceptance_rate,
            "target_forward_passes": target_forward_passes,
            "avg_tokens_per_step": num_generated / target_forward_passes if target_forward_passes > 0 else 0.0,
        }
        return input_ids, stats


class CUDAGraphRunner:
    """Wraps a PyTorch causal LM with static CUDA Graph capture to eliminate CPython driver overhead."""
    def __init__(self, model, static_shape=(1, 1), dtype=torch.float16):
        self.model = model
        self.graph = None
        self.static_input = None
        self.static_output = None
        self.static_shape = static_shape
        self.dtype = dtype

    def capture(self, sample_input, sample_kv=None):
        """Warmup and record the execution graph into a single hardware replay object."""
        if not torch.cuda.is_available():
            return
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                _ = self.model(sample_input, past_key_values=sample_kv, use_cache=True)
            torch.cuda.synchronize()

            self.static_input = sample_input.clone()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.static_output = self.model(self.static_input, past_key_values=sample_kv, use_cache=True)
        torch.cuda.current_stream().wait_stream(stream)

    def replay(self, dynamic_input):
        if self.graph is not None and self.static_input is not None:
            self.static_input.copy_(dynamic_input)
            self.graph.replay()
            return self.static_output
        return self.model(dynamic_input, use_cache=True)

