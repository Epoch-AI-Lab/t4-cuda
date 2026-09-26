"""Context Window Expansion to 16,384 Tokens via RoPE Scaling and PyTorch SDPA.

Implements YaRN (Yet another RoPE extensioN) and NTK-aware base frequency scaling
for Qwen2.5 models on NVIDIA Tesla T4 GPUs. Routes attention computation to
PyTorch scaled_dot_product_attention (SDPA) with enable_gqa=True and Cutlass
MemoryEfficientAttention on sm_75, eliminating intermediate O(N^2) score matrices.
"""

import math
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_yarn_rope_config(
    original_max_position_embeddings: int = 4096,
    target_max_position_embeddings: int = 16384,
    base_theta: float = 10000.0,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
) -> Dict[str, Any]:
    """Generates standard YaRN RoPE configuration dictionary for Qwen2.5 models."""
    factor = float(target_max_position_embeddings) / float(original_max_position_embeddings)
    attention_factor = 0.1 * math.log(factor) + 1.0 if factor > 1.0 else 1.0
    return {
        "rope_type": "yarn",
        "factor": factor,
        "original_max_position_embeddings": original_max_position_embeddings,
        "rope_theta": float(base_theta),
        "beta_fast": float(beta_fast),
        "beta_slow": float(beta_slow),
        "attention_factor": attention_factor,
    }


def build_ntk_base_theta(
    original_max_position_embeddings: int = 4096,
    target_max_position_embeddings: int = 16384,
    base_theta: float = 10000.0,
    head_dim: int = 128,
) -> float:
    """Calculates adjusted base frequency theta using the NTK-aware formula."""
    factor = float(target_max_position_embeddings) / float(original_max_position_embeddings)
    exponent = float(head_dim) / float(head_dim - 2)
    return float(base_theta * (factor ** exponent))


def compute_yarn_inv_freq(
    dim: int = 128,
    base_theta: float = 10000.0,
    factor: float = 4.0,
    original_max_position_embeddings: int = 4096,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, float]:
    """Computes YaRN inverse frequencies and attention scaling factor mathematically."""
    half_dim = dim // 2

    def find_correction_dim(num_rotations: float) -> float:
        return (dim * math.log(original_max_position_embeddings / (num_rotations * 2.0 * math.pi))) / (2.0 * math.log(base_theta))

    def find_correction_range(low_rot: float, high_rot: float) -> Tuple[int, int]:
        low = math.floor(find_correction_dim(low_rot))
        high = math.ceil(find_correction_dim(high_rot))
        return max(low, 0), min(high, half_dim - 1)

    # Compute in float32 to prevent FP16 overflow with large base_theta (e.g. 1e6 for Qwen2.5)
    dim_range = torch.arange(0, dim, 2, dtype=torch.float32, device=device)
    pos_freqs = float(base_theta) ** (dim_range / dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)

    low, high = find_correction_range(beta_fast, beta_slow)
    if low >= high:
        high = low + 1

    idx = torch.arange(half_dim, dtype=torch.float32, device=device)
    ramp = torch.clamp((idx - low) / (high - low), 0.0, 1.0)
    inv_freq = inv_freq_interpolation * ramp + inv_freq_extrapolation * (1.0 - ramp)

    attention_factor = 0.1 * math.log(factor) + 1.0 if factor > 1.0 else 1.0
    return inv_freq.to(dtype=dtype), float(attention_factor)


def compute_ntk_inv_freq(
    dim: int = 128,
    base_theta: float = 10000.0,
    factor: float = 4.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, float]:
    """Computes NTK-aware inverse frequencies with adjusted base theta."""
    new_theta = base_theta * (factor ** (float(dim) / float(dim - 2)))
    dim_range = torch.arange(0, dim, 2, dtype=dtype, device=device)
    inv_freq = 1.0 / (new_theta ** (dim_range / dim))
    return inv_freq, 1.0


def configure_rope_scaling(
    config: Any,
    max_position_embeddings: int = 16384,
    rope_type: str = "yarn",
    factor: Optional[float] = None,
    original_max_position_embeddings: Optional[int] = None,
    base_theta: Optional[float] = None,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
) -> Any:
    """Configures RoPE scaling on a HuggingFace or custom model configuration.

    Supports YaRN, dynamic NTK, and static base-theta scaling. Standardizes parameters
    across both config.rope_parameters and config.rope_scaling for compatibility.
    """
    orig_max = (
        original_max_position_embeddings
        if original_max_position_embeddings is not None
        else getattr(config, "original_max_position_embeddings", None)
    )
    if orig_max is None:
        orig_max = getattr(config, "max_position_embeddings", 4096)
        if orig_max is None or orig_max == max_position_embeddings:
            orig_max = 4096

    if factor is None:
        factor = float(max_position_embeddings) / float(orig_max)

    theta = (
        base_theta
        if base_theta is not None
        else getattr(config, "rope_theta", None)
    )
    if theta is None and hasattr(config, "rope_parameters") and isinstance(config.rope_parameters, dict):
        theta = config.rope_parameters.get("rope_theta", None)
    if theta is None:
        theta = 10000.0

    r_type = rope_type.lower()
    if r_type == "yarn":
        attention_factor = 0.1 * math.log(factor) + 1.0 if factor > 1.0 else 1.0
        scaling_dict = {
            "rope_type": "yarn",
            "factor": float(factor),
            "original_max_position_embeddings": int(orig_max),
            "rope_theta": float(theta),
            "beta_fast": float(beta_fast),
            "beta_slow": float(beta_slow),
            "attention_factor": float(attention_factor),
        }
    elif r_type in ("ntk", "base_theta", "default"):
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None and hasattr(config, "hidden_size") and hasattr(config, "num_attention_heads"):
            head_dim = config.hidden_size // config.num_attention_heads
        if head_dim is None:
            head_dim = 128
        new_theta = build_ntk_base_theta(
            original_max_position_embeddings=orig_max,
            target_max_position_embeddings=max_position_embeddings,
            base_theta=theta,
            head_dim=head_dim,
        )
        scaling_dict = {
            "rope_type": "default",
            "rope_theta": float(new_theta),
            "factor": float(factor),
            "original_max_position_embeddings": int(orig_max),
        }
    elif r_type in ("dynamic", "dynamic_ntk"):
        scaling_dict = {
            "rope_type": "dynamic",
            "factor": float(factor),
            "original_max_position_embeddings": int(orig_max),
            "rope_theta": float(theta),
        }
    else:
        scaling_dict = {
            "rope_type": rope_type,
            "factor": float(factor),
            "original_max_position_embeddings": int(orig_max),
            "rope_theta": float(theta),
        }

    config.max_position_embeddings = max_position_embeddings
    config.rope_scaling = scaling_dict

    if hasattr(config, "rope_parameters") and isinstance(config.rope_parameters, dict):
        config.rope_parameters.update(scaling_dict)
    else:
        config.rope_parameters = dict(scaling_dict)

    if hasattr(config, "standardize_rope_params"):
        try:
            config.standardize_rope_params()
        except Exception:
            pass

    return config


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input tensor."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_yarn(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies rotary position embedding to query and key tensors with gradient support."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class YaRNRotaryEmbedding(nn.Module):
    """YaRN rotary embedding module supporting 16k context window and backward gradients."""

    def __init__(
        self,
        dim: int = 128,
        max_position_embeddings: int = 16384,
        base_theta: float = 10000.0,
        factor: float = 4.0,
        original_max_position_embeddings: int = 4096,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base_theta = base_theta
        self.factor = factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow

        inv_freq, attention_scaling = compute_yarn_inv_freq(
            dim=dim,
            base_theta=base_theta,
            factor=factor,
            original_max_position_embeddings=original_max_position_embeddings,
            beta_fast=beta_fast,
            beta_slow=beta_slow,
            device=device,
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.attention_scaling = float(attention_scaling)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generates cos and sin positional embedding tensors matching input dtype and device."""
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def patch_model_rope_16k(
    model: nn.Module,
    max_seq_len: int = 16384,
    rope_type: str = "yarn",
    factor: Optional[float] = None,
) -> nn.Module:
    """Updates rotary embeddings and sequence limits across all model modules."""
    if hasattr(model, "config") and model.config is not None:
        configure_rope_scaling(
            model.config,
            max_position_embeddings=max_seq_len,
            rope_type=rope_type,
            factor=factor,
        )

    for mod in model.modules():
        mod_type_name = mod.__class__.__name__
        if "RotaryEmbedding" in mod_type_name:
            if hasattr(mod, "config") and mod.config is not None:
                mod.config = model.config
            if hasattr(mod, "max_seq_len_cached"):
                mod.max_seq_len_cached = max_seq_len
            if hasattr(mod, "original_max_seq_len"):
                mod.original_max_seq_len = max_seq_len

            dim = getattr(mod, "dim", 128)
            base_theta = getattr(mod, "base_theta", None)
            orig_max_pos = 4096
            if hasattr(model, "config") and model.config is not None:
                dim = getattr(model.config, "head_dim", None) or (
                    model.config.hidden_size // model.config.num_attention_heads
                    if hasattr(model.config, "hidden_size") and hasattr(model.config, "num_attention_heads")
                    else dim
                )
                if base_theta is None:
                    rope_params = getattr(model.config, "rope_parameters", None)
                    if isinstance(rope_params, dict):
                        base_theta = rope_params.get("rope_theta")
                if base_theta is None:
                    rope_scaling = getattr(model.config, "rope_scaling", None)
                    if isinstance(rope_scaling, dict):
                        base_theta = rope_scaling.get("rope_theta")
                if base_theta is None:
                    base_theta = getattr(model.config, "rope_theta", None)
                orig_max_pos = getattr(model.config, "original_max_position_embeddings", None) or getattr(model.config, "max_position_embeddings", 4096)

            if base_theta is None:
                base_theta = 10000.0

            calc_factor = factor if factor is not None else float(max_seq_len) / float(orig_max_pos)
            target_device = getattr(mod.inv_freq, "device", None) if hasattr(mod, "inv_freq") else None

            if rope_type.lower() == "ntk":
                inv_freq, attention_scaling = compute_ntk_inv_freq(
                    dim=dim,
                    base_theta=base_theta,
                    factor=calc_factor,
                    device=target_device,
                )
            else:
                inv_freq, attention_scaling = compute_yarn_inv_freq(
                    dim=dim,
                    base_theta=base_theta,
                    factor=calc_factor,
                    original_max_position_embeddings=orig_max_pos,
                    device=target_device,
                )

            if hasattr(mod, "inv_freq"):
                mod.inv_freq = nn.Buffer(inv_freq.to(mod.inv_freq.device), persistent=False)
            if hasattr(mod, "original_inv_freq"):
                mod.original_inv_freq = nn.Buffer(inv_freq.clone().to(mod.original_inv_freq.device), persistent=False)
            if hasattr(mod, "attention_scaling"):
                mod.attention_scaling = attention_scaling

    return model


class apply_16k_context_scope:
    """Configures 16k context window and routes attention to PyTorch SDPA with enable_gqa=True.

    Can be used both as a function and as a context manager.
    On sm_75 (Tesla T4), disables unsupported FlashAttention-2 and enables Cutlass
    MemoryEfficientAttention to eliminate O(N^2) dense DRAM attention matrices.
    """

    def __init__(
        self,
        model: nn.Module,
        max_seq_len: int = 16384,
        rope_type: str = "yarn",
        factor: Optional[float] = None,
    ):
        self.model = model
        self.max_seq_len = max_seq_len
        self.rope_type = rope_type
        self.factor = factor

        self._orig_attn_impl = None
        self._orig_flash_sdp = None
        self._orig_mem_efficient_sdp = None
        self._orig_math_sdp = None

        self.apply()

    def apply(self) -> "apply_16k_context_scope":
        """Executes the RoPE and SDPA memory-efficient attention patching."""
        patch_model_rope_16k(
            self.model,
            max_seq_len=self.max_seq_len,
            rope_type=self.rope_type,
            factor=self.factor,
        )

        if hasattr(self.model, "config") and self.model.config is not None:
            self._orig_attn_impl = getattr(self.model.config, "_attn_implementation", None)
            self.model.config._attn_implementation = "sdpa"

        if torch.cuda.is_available():
            if hasattr(torch.backends.cuda, "flash_sdp_enabled"):
                self._orig_flash_sdp = torch.backends.cuda.flash_sdp_enabled()
                self._orig_mem_efficient_sdp = torch.backends.cuda.mem_efficient_sdp_enabled()
                self._orig_math_sdp = torch.backends.cuda.math_sdp_enabled()

                # On Turing sm_75: Cutlass memory-efficient attention is supported, FlashAttention is not
                torch.backends.cuda.enable_flash_sdp(False)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
                torch.backends.cuda.enable_math_sdp(True)

        return self

    def __enter__(self) -> "apply_16k_context_scope":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._orig_attn_impl is not None and hasattr(self.model, "config") and self.model.config is not None:
            self.model.config._attn_implementation = self._orig_attn_impl

        if torch.cuda.is_available() and self._orig_flash_sdp is not None:
            torch.backends.cuda.enable_flash_sdp(self._orig_flash_sdp)
            torch.backends.cuda.enable_mem_efficient_sdp(self._orig_mem_efficient_sdp)
            torch.backends.cuda.enable_math_sdp(self._orig_math_sdp)


def execute_sdpa_gqa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: Optional[bool] = None,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Executes PyTorch SDPA with native GQA broadcasting and memory-efficient tiling.

    Correctly handles causal masking:
    - If attn_mask is supplied, is_causal must be False.
    - If is_causal is not specified, True is used only during prefill (q_len == kv_len > 1).
    - During autoregressive decode (q_len == 1, kv_len > 1), causal mask is False so query attends to all past KV tokens.
    """
    q_len = q.shape[-2]
    kv_len = k.shape[-2]

    if attn_mask is not None:
        causal_flag = False
    elif is_causal is not None:
        # In single-token decode, setting is_causal=True masks out tokens 1..kv_len-1 in PyTorch SDPA
        causal_flag = is_causal if q_len > 1 else False
    else:
        causal_flag = (q_len == kv_len and q_len > 1)

    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=causal_flag,
        scale=scale,
        enable_gqa=True,
    )
