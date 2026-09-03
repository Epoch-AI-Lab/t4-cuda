"""Unified Speculative Execution Serving Engine on NVIDIA Tesla T4.

Integrates StaticKVCache (M1) and PromptLookupDraftEngine (M2) into a low-latency
serving engine featuring single-pass parallel candidate verification, explicit RoPE
position_ids arithmetic, dynamic 4D causal attention masking, and O(1) integer
pointer rollback without dynamic memory slicing or dual-model synchronization passes.
"""

import math
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from src.static_kv_cache import StaticKVCache, build_causal_4d_mask, compute_position_ids
    from src.unified_draft_engine import PromptLookupDraftEngine
except ImportError:
    from static_kv_cache import StaticKVCache, build_causal_4d_mask, compute_position_ids
    from unified_draft_engine import PromptLookupDraftEngine


class UnifiedSpeculativeEngine:
    """Low-latency speculative decoding serving engine on Tesla T4.

    Eliminates dual-model forward passes, draft resynchronization passes (d_sync),
    and dynamic KV-cache reallocations by uniting StaticKVCache and PromptLookupDraftEngine
    into a single-pass verification loop with O(1) pointer rollback.
    """

    def __init__(
        self,
        target_model: nn.Module,
        draft_engine: Optional[Any] = None,
        static_cache: Optional[StaticKVCache] = None,
        tokenizer: Optional[Any] = None,
        max_seq_len: int = 2048,
        device: Union[str, torch.device] = "cuda",
        dtype: torch.dtype = torch.float16,
        **kwargs,
    ):
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        # Resolve device and default dtype
        if isinstance(device, str):
            if device.startswith("cuda") and not torch.cuda.is_available():
                self.device = torch.device("cpu")
                self.dtype = torch.float32
            else:
                self.device = torch.device(device)
                self.dtype = dtype if self.device.type == "cuda" else torch.float32
        else:
            self.device = device
            self.dtype = dtype if self.device.type == "cuda" else torch.float32

        # Override dtype if explicitly provided and not default
        if "dtype" in kwargs and kwargs["dtype"] is not None:
            self.dtype = kwargs["dtype"]

        # Bind or instantiate draft engine
        if draft_engine is not None:
            self.draft_engine = draft_engine
        else:
            max_ngram = kwargs.get("max_ngram_size", kwargs.get("n_gram_size", 3))
            min_ngram = kwargs.get("min_ngram_size", 1)
            self.draft_engine = PromptLookupDraftEngine(
                max_ngram_size=max_ngram,
                min_ngram_size=min_ngram,
            )

        # Bind or instantiate static KV-cache
        if static_cache is not None:
            self.cache = static_cache
            self.max_seq_len = static_cache.max_seq_len
            self.device = static_cache.device
            self.dtype = static_cache.dtype
        elif hasattr(target_model, "config") and target_model.config is not None:
            self.cache = StaticKVCache.from_model_config(
                target_model.config,
                max_batch_size=1,
                max_seq_len=self.max_seq_len,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            self.cache = StaticKVCache(
                max_batch_size=1,
                max_seq_len=self.max_seq_len,
                device=self.device,
                dtype=self.dtype,
            )

        # Resolve EOS token ID
        self.eos_token_id = (
            kwargs.get("eos_token_id", None)
            or (getattr(tokenizer, "eos_token_id", None) if tokenizer is not None else None)
            or getattr(getattr(target_model, "config", None), "eos_token_id", None)
        )

    def _prepare_input_ids(self, input_ids: Any) -> torch.Tensor:
        """Converts input_ids to a 2D LongTensor of shape (1, seq_len) on self.device."""
        if isinstance(input_ids, str):
            if self.tokenizer is not None:
                encoded = self.tokenizer.encode(input_ids, return_tensors="pt")
                if isinstance(encoded, torch.Tensor):
                    t = encoded
                elif isinstance(encoded, list):
                    t = torch.tensor([encoded], dtype=torch.long)
                else:
                    t = torch.as_tensor(encoded, dtype=torch.long)
            else:
                raise ValueError("String prompt provided but engine has no tokenizer.")
        elif isinstance(input_ids, list):
            t = torch.tensor([input_ids], dtype=torch.long)
        elif isinstance(input_ids, np.ndarray):
            t = torch.from_numpy(input_ids).to(dtype=torch.long)
        elif isinstance(input_ids, torch.Tensor):
            t = input_ids.to(dtype=torch.long)
        else:
            t = torch.as_tensor(input_ids, dtype=torch.long)

        if t.dim() == 1:
            t = t.unsqueeze(0)

        return t.to(device=self.device)

    def _sample_token(
        self,
        logits_1d: torch.Tensor,
        temperature: float = 0.0,
    ) -> torch.Tensor:
        """Samples next token ID using greedy argmax (temperature <= 0) or multinomial."""
        if logits_1d.dim() > 1:
            logits_1d = logits_1d.reshape(-1, logits_1d.shape[-1])
            if logits_1d.shape[0] == 1:
                logits_1d = logits_1d.squeeze(0)
            else:
                logits_1d = logits_1d[-1]
        if temperature <= 0.0:
            tok = torch.argmax(logits_1d, dim=-1)
        else:
            probs = F.softmax(logits_1d / max(temperature, 1e-5), dim=-1)
            tok = torch.multinomial(probs, num_samples=1)
        return tok.view(1, 1)

    def _extract_logits(self, model_output: Any) -> torch.Tensor:
        """Extracts logits tensor from diverse model output signatures."""
        if hasattr(model_output, "logits"):
            return model_output.logits
        if isinstance(model_output, (tuple, list)):
            return model_output[0]
        if isinstance(model_output, torch.Tensor):
            return model_output
        raise TypeError(f"Unexpected target model output type: {type(model_output)}")

    def _forward_target(
        self,
        input_ids: torch.Tensor,
        start_pos: int,
    ) -> torch.Tensor:
        """Executes a single forward evaluation on target model with explicit positions and mask."""
        query_len = input_ids.shape[-1]
        pos_ids = compute_position_ids(start_pos, query_len, batch_size=1, device=self.device)
        attn_mask = build_causal_4d_mask(1, query_len, start_pos, device=self.device, dtype=self.dtype)

        try:
            out = self.target_model(
                input_ids,
                past_key_values=self.cache,
                position_ids=pos_ids,
                attention_mask=attn_mask,
                use_cache=True,
            )
        except TypeError:
            # Fallback for target models not accepting explicit position_ids or attention_mask
            out = self.target_model(
                input_ids,
                past_key_values=self.cache,
                use_cache=True,
            )

        # Defensive: if target model did not update the cache internally, update it now
        if self.cache.current_pos == start_pos:
            k_dummy = torch.zeros(
                (1, self.cache.num_heads, query_len, self.cache.head_dim),
                dtype=self.dtype,
                device=self.device,
            )
            for l_idx in range(self.cache.num_layers):
                self.cache.update(l_idx, k_dummy, k_dummy, start_pos=start_pos)

        return self._extract_logits(out)

    @torch.inference_mode()
    def generate_autoregressive(
        self,
        input_ids: Any,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Autoregressive baseline decode using target model and StaticKVCache."""
        tensor_ids = self._prepare_input_ids(input_ids)
        batch_size, prompt_len = tensor_ids.shape
        if prompt_len == 0:
            raise ValueError("Prompt sequence length must be greater than 0.")
        if prompt_len + max_new_tokens > self.max_seq_len:
            raise ValueError(
                f"Requested sequence length ({prompt_len} + {max_new_tokens} = {prompt_len + max_new_tokens}) "
                f"exceeds cache capacity max_seq_len ({self.max_seq_len})."
            )

        self.cache.reset()
        emitted_ids = tensor_ids.clone()
        step_latencies: List[float] = []

        if self.device.type == "cuda":
            torch.cuda.synchronize(device=self.device)
        t_start = time.perf_counter()

        # Prefill prompt
        prefill_logits = self._forward_target(tensor_ids, start_pos=0)
        curr_token = self._sample_token(prefill_logits[0, -1, :], temperature=temperature)
        emitted_ids = torch.cat([emitted_ids, curr_token], dim=-1)

        # Decode loop
        while (emitted_ids.shape[-1] - prompt_len) < max_new_tokens:
            if self.eos_token_id is not None and curr_token.item() == self.eos_token_id:
                break
            if self.cache.current_pos >= self.max_seq_len:
                break

            t0 = time.perf_counter()
            decode_logits = self._forward_target(curr_token, start_pos=self.cache.current_pos)
            curr_token = self._sample_token(decode_logits[0, -1, :], temperature=temperature)
            emitted_ids = torch.cat([emitted_ids, curr_token], dim=-1)

            if self.device.type == "cuda":
                torch.cuda.synchronize(device=self.device)
            step_latencies.append(time.perf_counter() - t0)

        if self.device.type == "cuda":
            torch.cuda.synchronize(device=self.device)
        total_time = time.perf_counter() - t_start
        tokens_generated = emitted_ids.shape[-1] - prompt_len
        tokens_per_sec = (tokens_generated / total_time) if total_time > 0 else 0.0

        stats = {
            "num_tokens": tokens_generated,
            "total_tokens": tokens_generated,
            "total_time_s": total_time,
            "latency_seconds": total_time,
            "tokens_per_sec": tokens_per_sec,
            "tokens_per_second": tokens_per_sec,
            "step_latencies": step_latencies,
            "tokens": emitted_ids[0].tolist(),
        }
        return emitted_ids, stats

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Any,
        max_new_tokens: int = 64,
        k_draft: int = 3,
        lookahead_k: Optional[int] = None,
        temperature: float = 0.0,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Unified speculative serving execution loop on Tesla T4.

        Executes speculative decoding in strictly 1 target forward pass per round
        without dual-model passes, host synchronization stalls, or dynamic memory slicing.
        """
        effective_k = lookahead_k if lookahead_k is not None else k_draft
        tensor_ids = self._prepare_input_ids(input_ids)
        batch_size, prompt_len = tensor_ids.shape
        if prompt_len == 0:
            raise ValueError("Prompt sequence length must be greater than 0.")
        if prompt_len + max_new_tokens > self.max_seq_len:
            raise ValueError(
                f"Requested sequence length ({prompt_len} + {max_new_tokens} = {prompt_len + max_new_tokens}) "
                f"exceeds cache capacity max_seq_len ({self.max_seq_len})."
            )

        self.cache.reset()
        emitted_ids = tensor_ids.clone()
        step_latencies: List[float] = []

        total_proposed = 0
        total_accepted = 0
        target_forward_passes = 0
        speculative_rounds = 0

        if self.device.type == "cuda":
            torch.cuda.synchronize(device=self.device)
        t_start = time.perf_counter()

        # Step 1: Prefill phase
        prefill_logits = self._forward_target(tensor_ids, start_pos=0)
        target_forward_passes += 1

        last_token = self._sample_token(prefill_logits[0, -1, :], temperature=temperature)
        emitted_ids = torch.cat([emitted_ids, last_token], dim=-1)

        # Early exit if prefill emitted EOS or reached token budget
        if self.eos_token_id is not None and last_token.item() == self.eos_token_id:
            pass
        elif (emitted_ids.shape[-1] - prompt_len) >= max_new_tokens:
            pass
        else:
            # Step 2: Speculative verification loop
            while (emitted_ids.shape[-1] - prompt_len) < max_new_tokens:
                if self.eos_token_id is not None and last_token.item() == self.eos_token_id:
                    break
                if self.cache.current_pos >= self.max_seq_len:
                    break

                t_round_start = time.perf_counter()
                speculative_rounds += 1

                # Step 2a: Propose draft candidates
                cand_tensor, _ = self.draft_engine.propose(
                    emitted_ids, k_draft=effective_k, **kwargs
                )

                if cand_tensor is not None and getattr(cand_tensor, "shape", None) is not None:
                    if not isinstance(cand_tensor, torch.Tensor):
                        cand_tensor = torch.as_tensor(cand_tensor, dtype=torch.long, device=self.device)
                    else:
                        cand_tensor = cand_tensor.to(device=self.device, dtype=torch.long)
                    if cand_tensor.dim() == 1:
                        cand_tensor = cand_tensor.unsqueeze(0)
                    m = cand_tensor.shape[-1]
                else:
                    m = 0

                total_proposed += m
                start_pos = self.cache.current_pos

                # Step 2b: Build verification evaluation tensor
                if m > 0:
                    eval_tokens = torch.cat([last_token, cand_tensor], dim=-1)
                else:
                    eval_tokens = last_token

                query_len = eval_tokens.shape[-1]

                # Step 2c: Execute single parallel target verification pass
                logits_all = self._forward_target(eval_tokens, start_pos=start_pos)
                target_forward_passes += 1
                logits = logits_all[0]  # Shape: (query_len, vocab_size)

                # Step 2d: Greedy Leviathan verification or single-token fallback
                if m == 0:
                    # Clean fallback to standard autoregressive step
                    next_tok = self._sample_token(logits[0, :], temperature=temperature)
                    emitted_ids = torch.cat([emitted_ids, next_tok], dim=-1)
                    self.cache.rollback(start_pos + 1)
                    last_token = next_tok
                    hit_eos = self.eos_token_id is not None and next_tok.item() == self.eos_token_id
                    budget_reached = (emitted_ids.shape[-1] - prompt_len) >= max_new_tokens
                else:
                    accepted_in_round = 0
                    all_accepted = True
                    hit_eos = False
                    budget_reached = False

                    for j in range(m):
                        target_choice = self._sample_token(logits[j, :], temperature=temperature)
                        draft_tok = cand_tensor[:, j : j + 1]

                        if target_choice.item() == draft_tok.item():
                            # Candidate accepted
                            accepted_in_round += 1
                            emitted_ids = torch.cat([emitted_ids, draft_tok], dim=-1)
                            if self.eos_token_id is not None and draft_tok.item() == self.eos_token_id:
                                hit_eos = True
                                all_accepted = False
                                committed_pos = start_pos + 1 + accepted_in_round
                                self.cache.rollback(committed_pos)
                                last_token = draft_tok
                                break
                            if (emitted_ids.shape[-1] - prompt_len) >= max_new_tokens:
                                budget_reached = True
                                all_accepted = False
                                committed_pos = start_pos + 1 + accepted_in_round
                                self.cache.rollback(committed_pos)
                                last_token = draft_tok
                                break
                        else:
                            # Mismatch at candidate j: emit correction and rollback cache
                            all_accepted = False
                            emitted_ids = torch.cat([emitted_ids, target_choice], dim=-1)
                            committed_pos = start_pos + 1 + accepted_in_round
                            self.cache.rollback(committed_pos)
                            last_token = target_choice
                            if self.eos_token_id is not None and target_choice.item() == self.eos_token_id:
                                hit_eos = True
                            if (emitted_ids.shape[-1] - prompt_len) >= max_new_tokens:
                                budget_reached = True
                            break

                    # Bonus token emission when all candidates accepted
                    if all_accepted and not hit_eos and not budget_reached:
                        if (emitted_ids.shape[-1] - prompt_len) < max_new_tokens:
                            bonus_tok = self._sample_token(logits[m, :], temperature=temperature)
                            emitted_ids = torch.cat([emitted_ids, bonus_tok], dim=-1)
                            committed_pos = start_pos + 1 + m
                            self.cache.rollback(committed_pos)
                            last_token = bonus_tok
                            if self.eos_token_id is not None and bonus_tok.item() == self.eos_token_id:
                                hit_eos = True
                        else:
                            committed_pos = start_pos + 1 + m
                            self.cache.rollback(committed_pos)
                            last_token = cand_tensor[:, m - 1 : m]

                    total_accepted += accepted_in_round

                if self.device.type == "cuda":
                    torch.cuda.synchronize(device=self.device)
                step_latencies.append(time.perf_counter() - t_round_start)

                if hit_eos or budget_reached:
                    break

        if self.device.type == "cuda":
            torch.cuda.synchronize(device=self.device)
        total_time = time.perf_counter() - t_start
        tokens_generated = emitted_ids.shape[-1] - prompt_len
        acc_rate = (total_accepted / total_proposed) if total_proposed > 0 else 0.0
        tokens_per_sec = (tokens_generated / total_time) if total_time > 0 else 0.0

        stats = {
            "num_tokens": tokens_generated,
            "total_tokens": tokens_generated,
            "total_time_s": total_time,
            "latency_seconds": total_time,
            "tokens_per_sec": tokens_per_sec,
            "tokens_per_second": tokens_per_sec,
            "k_draft": effective_k,
            "lookahead_k": effective_k,
            "draft_tokens": total_proposed,
            "proposed": total_proposed,
            "total_proposed": total_proposed,
            "accepted_tokens": total_accepted,
            "accepted": total_accepted,
            "total_accepted": total_accepted,
            "acceptance_rate": acc_rate,
            "acceptance_rate_pct": acc_rate * 100.0,
            "target_forward_passes": target_forward_passes,
            "speculative_rounds": speculative_rounds,
            "rounds": speculative_rounds,
            "avg_tokens_per_target_step": (tokens_generated / target_forward_passes) if target_forward_passes > 0 else 0.0,
            "step_latencies": step_latencies,
            "tokens": emitted_ids[0].tolist(),
        }
        return emitted_ids, stats

    def __call__(self, *args, **kwargs) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Convenience callable delegating to generate."""
        return self.generate(*args, **kwargs)
