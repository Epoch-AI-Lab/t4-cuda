import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
"""Unit tests for ModifiedGRPOLoss.

Verifies:
1. Unscaled mean-centered advantages on tied groups (all 1.0, all 0.0, all -1.5)
   produce exactly zero loss and zero gradients without NaNs.
2. Removal of per-token length normalization: token gradients are invariant to
   completion sequence length.
3. Detached CISPO upper clipping keeps gradient active on tokens with high ratios.
4. Correct KL penalty calculation and interface unpacking support.
"""

import sys
import pytest
import torch
import torch.nn.functional as F

from src.rl.modified_grpo_loss import ModifiedGRPOLoss, GRPOLossOutput


def test_unscaled_advantages_tied_groups_zero_loss_and_gradients():
    """Verify that tied rollout groups produce zero advantage, zero loss, and zero gradient."""
    loss_fn = ModifiedGRPOLoss(clip_eps_low=0.2, clip_eps_high=0.2, beta_kl=0.01)

    tied_rewards = [
        torch.full((2, 4), 1.0),    # All correct
        torch.full((2, 4), 0.0),    # All abstain
        torch.full((2, 4), -1.5),   # All incorrect
        torch.full((1, 8), 0.75),   # Arbitrary tied value
    ]

    for rewards in tied_rewards:
        # Check advantage calculation
        adv = loss_fn.compute_advantages(rewards)
        assert torch.all(adv == 0.0), f"Advantages for tied rewards must be 0.0, got {adv}"

        # Setup differentiable log_probs
        B, G = rewards.shape
        T = 16
        log_probs = torch.randn(B, G, T, requires_grad=True)
        old_log_probs = log_probs.detach().clone()

        # Forward pass
        loss, metrics = loss_fn(
            log_probs=log_probs,
            old_log_probs=old_log_probs,
            rewards=rewards,
        )

        assert abs(loss.item()) < 1e-9, f"Loss on tied group must be 0.0, got {loss.item()}"
        assert torch.all(metrics["advantages"] == 0.0)

        # Backward pass
        loss.backward()

        assert log_probs.grad is not None, "Gradient must be populated"
        assert torch.all(log_probs.grad == 0.0), f"Gradients must be 0.0 on tied group, got {log_probs.grad}"
        assert not torch.isnan(log_probs.grad).any(), "Gradients must not contain NaNs"
        assert not torch.isinf(log_probs.grad).any(), "Gradients must not contain Infs"


def test_unscaled_advantages_zero_sum_and_scale():
    """Verify advantage calculation maintains unscaled reward scale and zero-sum property."""
    loss_fn = ModifiedGRPOLoss()

    # 1 success (+1.0), 3 failures (-1.5)
    rewards = torch.tensor([[1.0, -1.5, -1.5, -1.5]])
    adv = loss_fn.compute_advantages(rewards).squeeze()

    # Group mean = (1.0 - 4.5) / 4 = -0.875
    # adv[0] = 1.0 - (-0.875) = +1.875
    # adv[1:] = -1.5 - (-0.875) = -0.625
    assert abs(adv[0].item() - 1.875) < 1e-6, f"Expected adv[0]=1.875, got {adv[0].item()}"
    for i in range(1, 4):
        assert abs(adv[i].item() - (-0.625)) < 1e-6, f"Expected adv[{i}]=-0.625, got {adv[i].item()}"

    # Zero-sum property
    assert abs(adv.sum().item()) < 1e-6, f"Sum of advantages must be 0.0, got {adv.sum().item()}"


def test_no_per_token_length_normalization():
    """Verify that token gradients are not penalized by sequence length."""
    loss_fn = ModifiedGRPOLoss()

    # Sequence A: short (10 completion tokens)
    # Sequence B: long (100 completion tokens)
    # Both have the same advantage A = 1.0
    T_short = 10
    T_long = 100

    log_p_short = torch.zeros(1, 1, T_short, requires_grad=True)
    old_log_p_short = torch.zeros(1, 1, T_short)
    adv_short = torch.tensor([[1.0]])

    log_p_long = torch.zeros(1, 1, T_long, requires_grad=True)
    old_log_p_long = torch.zeros(1, 1, T_long)
    adv_long = torch.tensor([[1.0]])

    loss_short, _ = loss_fn(log_probs=log_p_short, old_log_probs=old_log_p_short, advantages=adv_short)
    loss_long, _ = loss_fn(log_probs=log_p_long, old_log_probs=old_log_p_long, advantages=adv_long)

    loss_short.backward()
    loss_long.backward()

    # Under standard length normalization, long tokens would have 1/10 the gradient.
    # Without length normalization, every token has identical gradient weight.
    grad_short_token = log_p_short.grad[0, 0, 0].item()
    grad_long_token = log_p_long.grad[0, 0, 0].item()

    assert abs(grad_short_token - grad_long_token) < 1e-6, (
        f"Gradients per token must be equal regardless of sequence length! "
        f"Got short={grad_short_token}, long={grad_long_token}"
    )


def test_cispo_upper_clipping_keeps_gradient_active():
    """Verify CISPO keeps gradients active when importance ratio exceeds upper clip threshold."""
    eps_high = 0.2
    loss_fn = ModifiedGRPOLoss(clip_eps_low=0.2, clip_eps_high=eps_high)

    # Setup token where ratio = exp(new - old) = exp(1.0) ~ 2.718 > 1 + eps_high (1.2)
    adv = torch.tensor([[1.5]])
    log_probs = torch.tensor([[-0.5]], requires_grad=True)
    old_log_probs = torch.tensor([[-1.5]]) # ratio = exp(1.0) > 1.2

    loss, metrics = loss_fn(log_probs=log_probs, old_log_probs=old_log_probs, advantages=adv)
    loss.backward()

    # The detached weight is clamped at 1.0 + eps_high = 1.2
    # The gradient is - (1 + eps_high) * adv = - 1.2 * 1.5 = -1.8
    expected_grad = - (1.0 + eps_high) * 1.5
    actual_grad = log_probs.grad[0, 0].item()

    assert abs(actual_grad - expected_grad) < 1e-5, (
        f"CISPO gradient should be {expected_grad}, got {actual_grad}"
    )
    assert abs(actual_grad) > 0.0, "CISPO gradient must be strictly non-zero when ratio exceeds clip bound"

    # Contrast with standard PPO objective where gradient zeros out on the clipped branch
    ratio_ppo = torch.exp(log_probs.detach() - old_log_probs)
    surr1 = ratio_ppo * adv
    surr2 = torch.clamp(ratio_ppo, 0.8, 1.2) * adv
    ppo_clip_val = torch.min(surr1, surr2)
    # The derivative of constant min(2.718*1.5, 1.2*1.5) with respect to ratio is 0.0


def test_completion_mask_masks_prompt_tokens():
    """Verify that tokens outside completion_mask receive zero gradient."""
    loss_fn = ModifiedGRPOLoss()

    B, G, T = 1, 2, 8
    log_probs = torch.randn(B, G, T, requires_grad=True)
    old_log_probs = torch.randn(B, G, T)
    advantages = torch.tensor([[1.0, -1.0]])

    # Mask first 3 tokens as prompt (0.0) and rest as completion (1.0)
    completion_mask = torch.ones(B, G, T)
    completion_mask[:, :, :3] = 0.0

    loss, _ = loss_fn(
        log_probs=log_probs,
        old_log_probs=old_log_probs,
        advantages=advantages,
        completion_mask=completion_mask,
    )
    loss.backward()

    # Tokens 0, 1, 2 must have zero gradient
    assert torch.all(log_probs.grad[:, :, :3] == 0.0), "Prompt tokens must receive zero gradient"
    # Tokens 3..7 must have non-zero gradient
    assert torch.all(log_probs.grad[:, :, 3:] != 0.0), "Completion tokens must receive active gradient"


def test_reference_policy_kl_divergence():
    """Verify KL divergence penalty behavior."""
    beta_kl = 0.05
    loss_fn = ModifiedGRPOLoss(beta_kl=beta_kl)

    log_probs = torch.tensor([[-1.0, -2.0]], requires_grad=True)
    old_log_probs = torch.tensor([[-1.0, -2.0]])
    advantages = torch.tensor([[0.0]]) # Zero policy loss to isolate KL

    # 1. Identical reference policy -> zero KL
    ref_log_probs_same = torch.tensor([[-1.0, -2.0]])
    loss_same, metrics_same = loss_fn(
        log_probs=log_probs,
        old_log_probs=old_log_probs,
        advantages=advantages,
        ref_log_probs=ref_log_probs_same,
    )
    assert abs(metrics_same["kl_loss"]) < 1e-6
    assert abs(loss_same.item()) < 1e-6

    # 2. Divergent reference policy -> positive KL
    ref_log_probs_diff = torch.tensor([[-0.5, -3.0]])
    loss_diff, metrics_diff = loss_fn(
        log_probs=log_probs,
        old_log_probs=old_log_probs,
        advantages=advantages,
        ref_log_probs=ref_log_probs_diff,
    )
    assert metrics_diff["kl_loss"] > 0.0
    assert abs(loss_diff.item() - beta_kl * metrics_diff["kl_loss"]) < 1e-6


def test_raw_logits_and_input_ids_flow():
    """Verify ModifiedGRPOLoss accepts raw model logits and computes next-token log-probs."""
    loss_fn = ModifiedGRPOLoss()

    N, S, V = 2, 6, 32
    logits = torch.randn(N, S, V, requires_grad=True)
    input_ids = torch.randint(0, V, (N, S))
    old_log_probs = torch.randn(N, S)
    completion_mask = torch.ones(N, S)
    advantages = torch.tensor([[1.0], [-1.0]])

    loss = loss_fn(
        logits=logits,
        input_ids=input_ids,
        completion_mask=completion_mask,
        old_logprobs=old_log_probs,
        advantages=advantages,
    )

    loss.backward()
    assert logits.grad is not None
    assert logits.grad.norm().item() > 0.0


def test_grpoloss_output_container():
    """Verify GRPOLossOutput supports both tuple unpacking and direct tensor operations."""
    loss_fn = ModifiedGRPOLoss()
    log_probs = torch.randn(1, 2, 4, requires_grad=True)
    old_log_probs = torch.randn(1, 2, 4)
    advantages = torch.tensor([[1.0, -1.0]])

    # Tuple unpacking
    out_loss, out_metrics = loss_fn(log_probs=log_probs, old_log_probs=old_log_probs, advantages=advantages)
    assert isinstance(out_loss, torch.Tensor)
    assert isinstance(out_metrics, dict)
    assert "advantages" in out_metrics
    assert "mean_ratio" in out_metrics

    # Direct backward on output container
    out = loss_fn(log_probs=log_probs, old_log_probs=old_log_probs, advantages=advantages)
    out.backward()
    assert log_probs.grad is not None


if __name__ == "__main__":
    print("Running test_modified_grpo_loss.py directly...")
    for fn in [
        test_unscaled_advantages_tied_groups_zero_loss_and_gradients,
        test_unscaled_advantages_zero_sum_and_scale,
        test_no_per_token_length_normalization,
        test_cispo_upper_clipping_keeps_gradient_active,
        test_completion_mask_masks_prompt_tokens,
        test_reference_policy_kl_divergence,
        test_raw_logits_and_input_ids_flow,
        test_grpoloss_output_container,
    ]:
        try:
            fn()
            print(f"PASS: {fn.__name__}")
        except Exception as e:
            print(f"FAIL: {fn.__name__} - {e}")
            sys.exit(1)
    print("All modified GRPO tests PASSED!")
