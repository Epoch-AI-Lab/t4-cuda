"""Modified GRPO Policy Gradient Loss.

Cursor-style stability fixes for long-context mathematical reasoning:
1. Unscaled mean-centered advantages: A_i = r_i - mean(r). Bypasses division by
   sample standard deviation, guaranteeing zero gradient on tied groups without NaNs.
2. Removal of per-token length normalization: Each token contributes equally to
   gradient updates, preventing bias toward short answers.
3. Detached CISPO-style upper-bound clipping: Ratio is detached before multiplying
   policy log-probabilities, preventing gradient zeroing on critical reasoning tokens.
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class GRPOLossOutput(tuple):
    """Container holding scalar loss and metrics dictionary.

    Supports both tuple unpacking:
        loss, metrics = loss_fn(...)
    and direct tensor usage:
        loss = loss_fn(...)
        loss.backward()
    """

    def __new__(cls, loss: torch.Tensor, metrics: Dict[str, Union[torch.Tensor, float]]):
        return super().__new__(cls, (loss, metrics))

    @property
    def loss(self) -> torch.Tensor:
        return self[0]

    @property
    def metrics(self) -> Dict[str, Union[torch.Tensor, float]]:
        return self[1]

    def backward(self, *args, **kwargs):
        return self[0].backward(*args, **kwargs)

    def item(self) -> float:
        return self[0].item()

    def __float__(self) -> float:
        return float(self[0])

    def __getattr__(self, name: str):
        return getattr(self[0], name)

    def __repr__(self) -> str:
        return f"GRPOLossOutput(loss={self[0].item():.6f}, metrics={list(self[1].keys())})"


class ModifiedGRPOLoss(nn.Module):
    """Modified Group Relative Policy Optimization (GRPO) loss function.

    Args:
        clip_eps_low: Lower clipping threshold for importance sampling ratio.
        clip_eps_high: Upper clipping threshold for CISPO importance ratio.
        beta_kl: Weight for reference policy KL divergence penalty.
    """

    def __init__(
        self,
        clip_eps_low: float = 0.2,
        clip_eps_high: float = 0.2,
        beta_kl: float = 0.01,
    ):
        super().__init__()
        self.clip_eps_low = clip_eps_low
        self.clip_eps_high = clip_eps_high
        self.beta_kl = beta_kl

    def compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """Compute unscaled mean-centered advantages.

        Formula:
            A_i = r_i - (1 / G) * sum_{j=1}^G r_j

        No division by sample standard deviation is performed. When all rollouts
        in a group tie (e.g. all 1.0, 0.0, or -1.5), advantages are exactly zero,
        preventing division by zero or NaN explosions.

        Args:
            rewards: Scalar rewards with shape [B, G] or [G] or [N].

        Returns:
            Mean-centered advantages with matching shape.
        """
        if rewards.dim() == 1:
            mean_r = rewards.mean()
            return rewards - mean_r
        mean_r = rewards.mean(dim=-1, keepdim=True)
        return rewards - mean_r

    def forward(
        self,
        *args,
        log_probs: Optional[torch.Tensor] = None,
        old_log_probs: Optional[torch.Tensor] = None,
        rewards: Optional[torch.Tensor] = None,
        advantages: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        completion_mask: Optional[torch.Tensor] = None,
        ref_log_probs: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        old_logprobs: Optional[torch.Tensor] = None,
        ref_logits: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> GRPOLossOutput:
        """Calculate modified GRPO surrogate policy loss.

        Args:
            log_probs: Policy log-probabilities [B, G, T] or [N, T].
            old_log_probs: Rollout policy log-probabilities [B, G, T] or [N, T].
            rewards: Rollout scalar rewards [B, G] or [G].
            advantages: Precomputed advantages [B, G] or [N, 1] or [B, G, 1].
            attention_mask: Mask for non-padding tokens.
            completion_mask: Binary mask (1.0 for completion, 0.0 for prompt/pad).
            ref_log_probs: Reference policy log-probabilities for KL divergence.
            logits: Optional raw model logits [N, S, V].
            input_ids: Optional sequence token IDs [N, S].
            old_logprobs: Alias for old_log_probs.
            ref_logits: Optional reference model logits [N, S, V].

        Returns:
            GRPOLossOutput containing scalar loss and metrics dictionary.
        """
        if old_log_probs is None and old_logprobs is not None:
            old_log_probs = old_logprobs

        # Parse positional arguments if provided
        if len(args) > 0:
            first_arg = args[0]
            # Check if first argument is logits [N, S, V] with V > 1
            if first_arg.dim() == 3 and first_arg.shape[-1] > 1 and len(args) > 1 and args[1].dtype == torch.int64:
                logits = first_arg
                input_ids = args[1]
                if len(args) > 2 and completion_mask is None:
                    completion_mask = args[2]
                if len(args) > 3 and old_log_probs is None:
                    old_log_probs = args[3]
                if len(args) > 4 and advantages is None:
                    advantages = args[4]
                if len(args) > 5 and ref_logits is None:
                    ref_logits = args[5]
            else:
                if log_probs is None:
                    log_probs = first_arg
                if len(args) > 1 and old_log_probs is None:
                    old_log_probs = args[1]
                if len(args) > 2 and rewards is None and advantages is None:
                    if args[2].dim() <= 2 and args[2].dtype in (torch.float32, torch.float16, torch.bfloat16):
                        rewards = args[2]
                if len(args) > 3 and attention_mask is None:
                    attention_mask = args[3]
                if len(args) > 4 and completion_mask is None:
                    completion_mask = args[4]
                if len(args) > 5 and ref_log_probs is None:
                    ref_log_probs = args[5]

        # Convert logits and input_ids to next-token log-probs if provided
        if logits is not None and input_ids is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = input_ids[:, 1:].contiguous()
            log_softmax_probs = F.log_softmax(shift_logits.float(), dim=-1)
            log_probs = log_softmax_probs.gather(
                dim=-1, index=shift_targets.unsqueeze(-1)
            ).squeeze(-1)

            if old_log_probs is not None and old_log_probs.shape == input_ids.shape:
                old_log_probs = old_log_probs[:, 1:].contiguous()

            if completion_mask is not None and completion_mask.shape == input_ids.shape:
                completion_mask = completion_mask[:, 1:].contiguous()

            if ref_logits is not None:
                with torch.no_grad():
                    shift_ref = ref_logits[:, :-1, :].contiguous()
                    ref_lp = F.log_softmax(shift_ref.float(), dim=-1)
                    ref_log_probs = ref_lp.gather(
                        dim=-1, index=shift_targets.unsqueeze(-1)
                    ).squeeze(-1)

        if log_probs is None:
            raise ValueError("log_probs or logits+input_ids must be provided.")
        if old_log_probs is None:
            raise ValueError("old_log_probs must be provided.")

        # Compute or reshape advantages
        if advantages is None:
            if rewards is None:
                raise ValueError("Either rewards or advantages must be provided.")
            advantages = self.compute_advantages(rewards)

        # Broadcast advantages to match log_probs token dimensions
        if log_probs.dim() == 3:
            if advantages.dim() == 2:
                adv_expanded = advantages.unsqueeze(-1)
            elif advantages.dim() == 3:
                adv_expanded = advantages
            else:
                adv_expanded = advantages.view(log_probs.shape[0], log_probs.shape[1], 1)
        elif log_probs.dim() == 2:
            if advantages.dim() == 1:
                adv_expanded = advantages.unsqueeze(-1)
            elif advantages.dim() == 2:
                if advantages.shape[-1] == 1:
                    adv_expanded = advantages
                else:
                    adv_expanded = advantages.reshape(-1, 1)
            else:
                adv_expanded = advantages.reshape(-1, 1)
        else:
            adv_expanded = advantages

        # Importance sampling ratio
        log_ratio = log_probs - old_log_probs.detach()
        ratio = torch.exp(log_ratio)

        # CISPO-style detached ratio with upper-bound clipping
        min_ratio = 1.0 - self.clip_eps_low if self.clip_eps_low is not None else 0.0
        max_ratio = 1.0 + self.clip_eps_high
        clipped_weight = torch.clamp(ratio, min=min_ratio, max=max_ratio).detach()

        # Token-level surrogate loss
        token_loss = - clipped_weight * log_probs * adv_expanded

        # Token mask application
        mask = None
        if completion_mask is not None:
            mask = completion_mask
        elif attention_mask is not None:
            mask = attention_mask

        if mask is not None:
            token_loss = token_loss * mask

        # Removal of per-token length normalization:
        # Sum over sequence tokens, then average across sequences in batch/group.
        # No division by sequence length 1 / |o_i| is performed.
        seq_loss = token_loss.sum(dim=-1)
        policy_loss = seq_loss.mean()

        # Reference policy KL divergence penalty (Schulman k3 estimator)
        kl_loss = torch.tensor(0.0, device=log_probs.device, dtype=torch.float32)
        if ref_log_probs is not None:
            log_kl_ratio = ref_log_probs.detach() - log_probs
            token_kl = torch.exp(log_kl_ratio) - log_kl_ratio - 1.0
            if mask is not None:
                token_kl = token_kl * mask
            seq_kl = token_kl.sum(dim=-1)
            kl_loss = seq_kl.mean()

        total_loss = policy_loss + self.beta_kl * kl_loss

        # Collect metrics
        with torch.no_grad():
            clip_mask = (ratio < min_ratio) | (ratio > max_ratio)
            if mask is not None:
                valid_count = mask.sum().clamp(min=1.0)
                clip_fraction = (clip_mask.float() * mask).sum() / valid_count
                mean_ratio = (ratio * mask).sum() / valid_count
            else:
                clip_fraction = clip_mask.float().mean()
                mean_ratio = ratio.mean()

        metrics = {
            "advantages": advantages.detach(),
            "clipped_ratio": clipped_weight,
            "mean_ratio": mean_ratio.item(),
            "clip_fraction": clip_fraction.item(),
            "policy_loss": policy_loss.detach().item(),
            "kl_loss": kl_loss.detach().item(),
            "total_loss": total_loss.detach().item(),
        }

        return GRPOLossOutput(total_loss, metrics)
