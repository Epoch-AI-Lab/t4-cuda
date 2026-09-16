"""Unit tests for 16,384 token context window expansion and memory efficiency."""

import math
import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace

from src.rope_scaling import (
    build_yarn_rope_config,
    build_ntk_base_theta,
    compute_yarn_inv_freq,
    compute_ntk_inv_freq,
    configure_rope_scaling,
    rotate_half,
    apply_rotary_pos_emb_yarn,
    YaRNRotaryEmbedding,
    patch_model_rope_16k,
    apply_16k_context_scope,
    execute_sdpa_gqa_attention,
)


def test_build_yarn_rope_config():
    cfg = build_yarn_rope_config(
        original_max_position_embeddings=4096,
        target_max_position_embeddings=16384,
        base_theta=10000.0,
    )
    assert cfg["rope_type"] == "yarn"
    assert cfg["factor"] == 4.0
    assert cfg["original_max_position_embeddings"] == 4096
    assert cfg["rope_theta"] == 10000.0
    expected_attn_factor = 0.1 * math.log(4.0) + 1.0
    assert math.isclose(cfg["attention_factor"], expected_attn_factor, rel_tol=1e-5)


def test_build_ntk_base_theta():
    base_theta = 10000.0
    head_dim = 128
    scaled_theta = build_ntk_base_theta(
        original_max_position_embeddings=4096,
        target_max_position_embeddings=16384,
        base_theta=base_theta,
        head_dim=head_dim,
    )
    expected = base_theta * (4.0 ** (128.0 / 126.0))
    assert math.isclose(scaled_theta, expected, rel_tol=1e-5)


def test_compute_yarn_inv_freq():
    dim = 128
    base_theta = 10000.0
    factor = 4.0
    orig_max = 4096
    inv_freq, attn_factor = compute_yarn_inv_freq(
        dim=dim,
        base_theta=base_theta,
        factor=factor,
        original_max_position_embeddings=orig_max,
    )
    assert inv_freq.shape == (64,)
    # Mathematical boundary 1: index 0 (high frequency) must strictly equal pure extrapolation
    expected_extrap_0 = 1.0 / (base_theta ** (0.0 / dim))
    assert math.isclose(inv_freq[0].item(), expected_extrap_0, rel_tol=1e-5)

    # Mathematical boundary 2: index -1 (low frequency) must strictly equal pure interpolation
    last_dim_val = (dim - 2) / dim
    expected_interp_last = 1.0 / (factor * (base_theta ** last_dim_val))
    assert math.isclose(inv_freq[-1].item(), expected_interp_last, rel_tol=1e-5)

    assert inv_freq[0] > inv_freq[-1]
    expected_attn_factor = 0.1 * math.log(factor) + 1.0
    assert math.isclose(attn_factor, expected_attn_factor, rel_tol=1e-5)

    # Numerical stability check with Qwen2.5 base theta (1,000,000)
    qwen_inv_freq, _ = compute_yarn_inv_freq(
        dim=128,
        base_theta=1000000.0,
        factor=4.0,
        original_max_position_embeddings=orig_max,
    )
    assert not torch.isinf(qwen_inv_freq).any()
    assert not torch.isnan(qwen_inv_freq).any()
    assert (qwen_inv_freq > 0.0).all()


def test_compute_ntk_inv_freq():
    dim = 128
    base_theta = 10000.0
    factor = 4.0
    inv_freq, attn_factor = compute_ntk_inv_freq(dim=dim, base_theta=base_theta, factor=factor)
    assert inv_freq.shape == (64,)
    assert attn_factor == 1.0

    # Verify scaled theta in NTK formula
    scaled_theta = base_theta * (factor ** (128.0 / 126.0))
    expected_freq_0 = 1.0 / (scaled_theta ** (0.0 / 128.0))
    expected_freq_last = 1.0 / (scaled_theta ** (126.0 / 128.0))
    assert math.isclose(inv_freq[0].item(), expected_freq_0, rel_tol=1e-5)
    assert math.isclose(inv_freq[-1].item(), expected_freq_last, rel_tol=1e-5)


def test_configure_rope_scaling():
    dummy_cfg = SimpleNamespace(
        max_position_embeddings=4096,
        rope_theta=10000.0,
        head_dim=128,
    )
    res_cfg = configure_rope_scaling(
        dummy_cfg,
        max_position_embeddings=16384,
        rope_type="yarn",
    )
    assert res_cfg.max_position_embeddings == 16384
    assert res_cfg.rope_scaling["rope_type"] == "yarn"
    assert res_cfg.rope_scaling["factor"] == 4.0


def test_rotate_half():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    res = rotate_half(x)
    assert torch.equal(res, torch.tensor([[-3.0, -4.0, 1.0, 2.0]]))


def test_apply_rotary_pos_emb_yarn():
    bs, seq, num_heads, head_dim = 1, 4, 2, 64
    q = torch.randn(bs, num_heads, seq, head_dim)
    k = torch.randn(bs, num_heads, seq, head_dim)

    # Test with non-trivial angle theta = pi/2: cos = 0, sin = 1
    # For a vector [x1, x2], rotation by pi/2 yields [-x2, x1]
    cos_90 = torch.zeros(bs, seq, head_dim)
    sin_90 = torch.ones(bs, seq, head_dim)

    q_rot90, k_rot90 = apply_rotary_pos_emb_yarn(q, k, cos_90, sin_90, unsqueeze_dim=1)
    half_d = head_dim // 2
    expected_q_rot = torch.cat([-q[..., half_d:], q[..., :half_d]], dim=-1)
    assert torch.allclose(q_rot90, expected_q_rot, atol=1e-5)

    # Verify norm preservation under arbitrary rotation angle (theta = pi/4)
    angle = math.pi / 4.0
    cos_45 = torch.full((bs, seq, head_dim), math.cos(angle))
    sin_45 = torch.full((bs, seq, head_dim), math.sin(angle))
    q_rot45, _ = apply_rotary_pos_emb_yarn(q, k, cos_45, sin_45, unsqueeze_dim=1)
    q_norm_orig = torch.norm(q, dim=-1)
    q_norm_rot = torch.norm(q_rot45, dim=-1)
    assert torch.allclose(q_norm_orig, q_norm_rot, atol=1e-5)


def test_yarn_rotary_embedding_module():
    yarn_emb = YaRNRotaryEmbedding(
        dim=64,
        max_position_embeddings=16384,
        factor=4.0,
        original_max_position_embeddings=4096,
    )
    x = torch.randn(1, 16, 64)
    pos_ids = torch.arange(0, 16).unsqueeze(0)
    cos, sin = yarn_emb(x, pos_ids)

    # YaRN scales both cos and sin by attention_scaling (t = 0.1*ln(s) + 1.0)
    # Therefore, cos^2 + sin^2 must mathematically equal attention_scaling^2 everywhere
    identity = (cos ** 2) + (sin ** 2)
    expected_identity = yarn_emb.attention_scaling ** 2
    assert torch.allclose(identity, torch.full_like(identity, expected_identity), atol=1e-5)

    # Position 0 identity check: cos(0) = attention_scaling, sin(0) = 0
    assert torch.allclose(cos[:, 0, :], torch.full((1, 64), yarn_emb.attention_scaling), atol=1e-5)
    assert torch.allclose(sin[:, 0, :], torch.zeros(1, 64), atol=1e-5)


def test_execute_sdpa_gqa_attention_prefill_and_decode():
    bs = 1
    q_heads = 8
    kv_heads = 2
    head_dim = 64

    # 1. Prefill test (q_len == kv_len == 32)
    seq_len = 32
    q = torch.randn(bs, q_heads, seq_len, head_dim)
    k = torch.randn(bs, kv_heads, seq_len, head_dim)
    v = torch.randn(bs, kv_heads, seq_len, head_dim)

    out_prefill = execute_sdpa_gqa_attention(q, k, v)
    assert out_prefill.shape == (bs, q_heads, seq_len, head_dim)
    assert not torch.isnan(out_prefill).any()

    # 2. Decode test (q_len == 1, kv_len == 128)
    # Query is all 1s. Key at index 127 is all 1s (dot product = 64, scaled = 64 / sqrt(64) = 8.0).
    # Keys at indices 0..126 are all 0s (dot product = 0).
    # Value at index 127 is 5.0, others are 0.
    q_dec = torch.ones(bs, q_heads, 1, head_dim)
    k_dec = torch.zeros(bs, kv_heads, 128, head_dim)
    k_dec[:, :, 127, :] = 1.0
    v_dec = torch.zeros(bs, kv_heads, 128, head_dim)
    v_dec[:, :, 127, :] = 5.0

    # In single-token decode, attention must not mask out past KV positions
    out_decode = execute_sdpa_gqa_attention(q_dec, k_dec, v_dec)
    assert out_decode.shape == (bs, q_heads, 1, head_dim)
    assert not torch.isnan(out_decode).any()

    # Analytical softmax check:
    # logit_127 = 64 / 8.0 = 8.0; 127 logits = 0.0
    # prob_127 = exp(8.0) / (127 * exp(0.0) + exp(8.0)) = 2980.958 / (127 + 2980.958) = 0.959137
    # Expected output = prob_127 * 5.0 = 4.79568
    expected_prob_127 = math.exp(8.0) / (127.0 * math.exp(0.0) + math.exp(8.0))
    expected_output_val = expected_prob_127 * 5.0
    assert math.isclose(out_decode[0, 0, 0, 0].item(), expected_output_val, rel_tol=1e-4)


def test_apply_16k_context_scope():
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                max_position_embeddings=4096,
                rope_theta=10000.0,
                head_dim=128,
                _attn_implementation="eager",
            )

    model = DummyModel()
    with apply_16k_context_scope(model, max_seq_len=16384):
        assert model.config.max_position_embeddings == 16384
        assert model.config._attn_implementation == "sdpa"
    assert model.config._attn_implementation == "eager"
