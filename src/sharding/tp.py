"""src/sharding/tp.py

Intra-Layer Tensor Parallelism (TP=2) for INT4 MLP and FP16 Attention.

Provides:
- TPColumnParallelLinear: Slices weight matrix along output dimension N (zero communication).
- TPRowParallelLinear: Slices weight matrix along input dimension K (1 All-Reduce).
- TPParallelMLP: Full INT4 SwiGLU MLP block with exactly 1 All-Reduce.
- TPParallelAttention: Full FP16 GQA Attention block with sharded KV cache and 1 All-Reduce.
- Weight slicing utilities for standard FP16 and packed INT4 tensors.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import t4_kernels
    HAS_T4_KERNELS = True
except ImportError:
    t4_kernels = None
    HAS_T4_KERNELS = False

from src.sharding.comm import all_reduce_sum, all_reduce_pair


def quantize_weight_sym_int4(
    W: torch.Tensor,
    group_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Quantize 2D weight matrix [out_features, in_features] to symmetric INT4.
    
    Packs 8 INT4 values along the in_features dimension into each int32 word.
    Returns:
        packed: [in_features // 8, out_features], dtype int32
        scales: [num_groups, out_features], dtype matches W.dtype
        zps: [num_groups, out_features], dtype matches W.dtype (filled with 8.0)
        group_size: effective group size
    """
    out_f, in_f = W.shape
    if in_f % 8 != 0:
        raise ValueError(f"in_features ({in_f}) must be divisible by 8")
    Wf = W.float()
    target_dtype = W.dtype if W.dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float16

    if group_size > 0 and in_f % group_size == 0:
        num_groups = in_f // group_size
        Wf_grouped = Wf.reshape(out_f, num_groups, group_size)
        amax = Wf_grouped.abs().amax(dim=2, keepdim=True).clamp_min(1e-8)
        scale = amax / 7.0
        q = torch.clamp(torch.round(Wf_grouped / scale) + 8, 0, 15).reshape(out_f, in_f).to(torch.int32)
        q = q.t().contiguous()
        packed = torch.zeros(in_f // 8, out_f, dtype=torch.int32, device=W.device)
        for i in range(8):
            packed |= q[i::8, :] << (4 * i)
        scales = scale.squeeze(2).t().contiguous().to(dtype=target_dtype, device=W.device)
        zps = torch.full_like(scales, 8.0)
        return packed, scales, zps, group_size
    else:
        amax = Wf.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
        scale = amax / 7.0
        q = torch.clamp(torch.round(Wf / scale) + 8, 0, 15).to(torch.int32)
        q = q.t().contiguous()
        packed = torch.zeros(in_f // 8, out_f, dtype=torch.int32, device=W.device)
        for i in range(8):
            packed |= q[i::8, :] << (4 * i)
        scales = scale.squeeze(1).to(target_dtype).unsqueeze(0).contiguous().to(W.device)
        zps = torch.full((1, out_f), 8.0, dtype=target_dtype, device=W.device)
        return packed, scales, zps, 0


def dequantize_sym_int4(
    packed: torch.Tensor,
    scales: torch.Tensor,
    zps: torch.Tensor,
    group_size: int = 128,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Pure PyTorch reference dequantization matching t4_kernels convention.
    
    Returns:
        W_dequant: [out_features, in_features], dtype matches scales.dtype (or explicit dtype)
    """
    in_div_8, out_f = packed.shape
    in_f = in_div_8 * 8
    target_dtype = dtype if dtype is not None else scales.dtype

    q = torch.zeros((in_f, out_f), dtype=torch.float32, device=packed.device)
    for i in range(8):
        nibble = (packed >> (4 * i)) & 0xF
        q[i::8, :] = nibble.float()
    q = q.t()

    if group_size > 0:
        if in_f % group_size != 0:
            raise ValueError(f"in_features ({in_f}) must be divisible by group_size ({group_size})")
        num_groups = in_f // group_size
        if scales.shape[0] != num_groups:
            raise ValueError(
                f"scales.shape[0] ({scales.shape[0]}) does not match expected num_groups ({num_groups})"
            )
        q_grouped = q.reshape(out_f, num_groups, group_size)
        scale_grouped = scales.t().unsqueeze(2).float()
        unquant = (q_grouped - 8.0) * scale_grouped
        return unquant.reshape(out_f, in_f).to(target_dtype)
    else:
        scale_f = scales.t().float()
        unquant = (q - 8.0) * scale_f
        return unquant.to(target_dtype)


def slice_column_parallel_weight(
    weight: torch.Tensor,
    rank: int,
    world_size: int = 2,
) -> torch.Tensor:
    """Slices unquantized weight [out_features, in_features] along dimension 0."""
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank ({rank}) must be strictly less than world_size ({world_size}) and non-negative")
    out_f = weight.shape[0]
    if out_f % world_size != 0:
        raise ValueError(f"out_features ({out_f}) must be divisible by world_size ({world_size})")
    split_n = out_f // world_size
    start = rank * split_n
    end = (rank + 1) * split_n
    return weight[start:end, :].contiguous()


def slice_column_parallel_int4(
    packed: torch.Tensor,
    scales: torch.Tensor,
    zps: torch.Tensor,
    rank: int,
    world_size: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Slices packed INT4 weight and scales along dimension 1 (out_features N)."""
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank ({rank}) must be strictly less than world_size ({world_size}) and non-negative")
    out_f = packed.shape[1]
    if out_f % world_size != 0:
        raise ValueError(f"out_features ({out_f}) must be divisible by world_size ({world_size})")
    split_n = out_f // world_size
    start = rank * split_n
    end = (rank + 1) * split_n
    return (
        packed[:, start:end].contiguous(),
        scales[:, start:end].contiguous(),
        zps[:, start:end].contiguous(),
    )


def slice_row_parallel_weight(
    weight: torch.Tensor,
    rank: int,
    world_size: int = 2,
) -> torch.Tensor:
    """Slices unquantized weight [out_features, in_features] along dimension 1."""
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank ({rank}) must be strictly less than world_size ({world_size}) and non-negative")
    in_f = weight.shape[1]
    if in_f % world_size != 0:
        raise ValueError(f"in_features ({in_f}) must be divisible by world_size ({world_size})")
    split_k = in_f // world_size
    start = rank * split_k
    end = (rank + 1) * split_k
    return weight[:, start:end].contiguous()


def slice_row_parallel_int4(
    packed: torch.Tensor,
    scales: torch.Tensor,
    zps: torch.Tensor,
    rank: int,
    world_size: int = 2,
    group_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Slices packed INT4 weight along dimension 0 (in_features K words)."""
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank ({rank}) must be strictly less than world_size ({world_size}) and non-negative")
    in_div_8 = packed.shape[0]
    if in_div_8 % world_size != 0:
        raise ValueError(f"packed rows ({in_div_8}) must be divisible by world_size ({world_size})")
    in_features = in_div_8 * 8
    split_in_features = in_features // world_size

    split_words = in_div_8 // world_size
    start_w = rank * split_words
    end_w = (rank + 1) * split_words
    packed_shard = packed[start_w:end_w, :].contiguous()

    if group_size > 0:
        num_groups = scales.shape[0]
        if split_in_features % group_size != 0 or num_groups % world_size != 0:
            raise ValueError(
                f"Cannot shard INT4 row-parallel weight: split_in_features ({split_in_features}) "
                f"must be divisible by group_size ({group_size}) without dropping groups. "
                f"num_groups ({num_groups}) must be divisible by world_size ({world_size})."
            )
        split_groups = num_groups // world_size
        start_g = rank * split_groups
        end_g = (rank + 1) * split_groups
        scales_shard = scales[start_g:end_g, :].contiguous()
        zps_shard = zps[start_g:end_g, :].contiguous()
    else:
        scales_shard = scales.clone()
        zps_shard = zps.clone()

    return packed_shard, scales_shard, zps_shard


class TPColumnParallelLinear(nn.Module):
    """Column-Parallel Linear layer sharded across dual GPUs.
    
    Partitions the output feature dimension N into N / world_size per rank.
    Forward pass takes replicated input X and computes local output slice Y_i = X W_i.
    Requires zero inter-GPU communication.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        rank: int = 0,
        world_size: int = 2,
        quant_type: Optional[str] = None,
        group_size: int = 128,
        dtype: torch.dtype = torch.float16,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        if rank < 0 or rank >= world_size:
            raise ValueError(f"rank ({rank}) must be strictly less than world_size ({world_size}) and non-negative")
        if out_features % world_size != 0:
            raise ValueError(f"out_features ({out_features}) must be divisible by world_size ({world_size})")

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.world_size = world_size
        self.split_out_features = out_features // world_size
        self.quant_type = quant_type
        self.group_size = group_size
        self.dtype = dtype

        if quant_type == "int4":
            if in_features % 8 != 0:
                raise ValueError(f"in_features ({in_features}) must be divisible by 8")
            self.register_buffer(
                "packed",
                torch.zeros(in_features // 8, self.split_out_features, dtype=torch.int32, device=device),
            )
            num_groups = in_features // group_size if group_size > 0 else 1
            self.register_buffer(
                "scales",
                torch.ones(num_groups, self.split_out_features, dtype=dtype, device=device),
            )
            self.register_buffer(
                "zps",
                torch.full((num_groups, self.split_out_features), 8.0, dtype=dtype, device=device),
            )
            self.weight = None
            self.weight_fp16 = None
        else:
            self.weight = nn.Parameter(
                torch.empty((self.split_out_features, in_features), dtype=dtype, device=device)
            )
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            self.weight_fp16 = None

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(self.split_out_features, dtype=dtype, device=device)
            )
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int = 0,
        world_size: int = 2,
        quant_type: Optional[str] = None,
        group_size: int = 128,
    ) -> TPColumnParallelLinear:
        """Instantiates a sharded TPColumnParallelLinear from an existing nn.Linear."""
        in_f = linear.in_features
        out_f = linear.out_features
        has_bias = linear.bias is not None

        layer = cls(
            in_features=in_f,
            out_features=out_f,
            bias=has_bias,
            rank=rank,
            world_size=world_size,
            quant_type=quant_type,
            group_size=group_size,
            dtype=linear.weight.dtype,
            device=linear.weight.device,
        )
        split_n = out_f // world_size
        start = rank * split_n
        end = (rank + 1) * split_n

        with torch.no_grad():
            if quant_type == "int4":
                packed_full, scales_full, zps_full, g_size = quantize_weight_sym_int4(
                    linear.weight, group_size=group_size
                )
                p_shard, s_shard, z_shard = slice_column_parallel_int4(
                    packed_full, scales_full, zps_full, rank, world_size
                )
                layer.packed.copy_(p_shard)
                layer.scales.copy_(s_shard)
                layer.zps.copy_(z_shard)
                layer.group_size = g_size
            else:
                w_shard = slice_column_parallel_weight(linear.weight, rank, world_size)
                layer.weight.copy_(w_shard)

            if has_bias:
                layer.bias.copy_(linear.bias[start:end].contiguous())

        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_clean = x.contiguous()

        if x_clean.dim() == 0:
            raise ValueError("Input tensor must have at least 1 dimension")
        if x_clean.shape[-1] != self.in_features:
            raise ValueError(
                f"Input trailing feature dimension ({x_clean.shape[-1]}) must match in_features ({self.in_features})"
            )

        x_2d = x_clean.view(-1, self.in_features) if x_clean.dim() != 2 else x_clean
        M = x_2d.shape[0]

        if M == 0:
            out_shape = (*orig_shape[:-1], self.split_out_features)
            return torch.empty(out_shape, dtype=self.dtype, device=x.device)

        if self.quant_type == "int4":
            if HAS_T4_KERNELS and x.is_cuda:
                out = t4_kernels.fused_w4a16_gemm_u4(
                    x_2d, self.packed, self.scales, self.zps, self.group_size
                )
                if self.bias is not None:
                    out = out + self.bias
            else:
                target_dtype = x_2d.dtype if x_2d.dtype in (torch.float16, torch.bfloat16, torch.float32) else self.dtype
                W_dequant = dequantize_sym_int4(
                    self.packed, self.scales, self.zps, self.group_size, dtype=target_dtype
                )
                if x_2d.dtype != W_dequant.dtype:
                    x_2d = x_2d.to(W_dequant.dtype)
                out = F.linear(x_2d, W_dequant, self.bias)
        else:
            if self.weight is not None and x_2d.dtype != self.weight.dtype:
                x_2d = x_2d.to(self.weight.dtype)
            out = F.linear(x_2d, self.weight, self.bias)

        return out.view(*orig_shape[:-1], self.split_out_features) if orig_shape[:-1] else out.view(self.split_out_features)


class TPRowParallelLinear(nn.Module):
    """Row-Parallel Linear layer sharded across dual GPUs.
    
    Partitions the input feature dimension K into K / world_size per rank.
    Computes local partial sum Z_i = X_i W_i, then executes fast All-Reduce
    to yield final reduced output Y = Z_0 + Z_1.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        rank: int = 0,
        world_size: int = 2,
        quant_type: Optional[str] = None,
        group_size: int = 128,
        dtype: torch.dtype = torch.float16,
        device: Optional[torch.device] = None,
        all_reduce: bool = True,
    ):
        super().__init__()
        if rank < 0 or rank >= world_size:
            raise ValueError(f"rank ({rank}) must be strictly less than world_size ({world_size}) and non-negative")
        if in_features % world_size != 0:
            raise ValueError(f"in_features ({in_features}) must be divisible by world_size ({world_size})")

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.world_size = world_size
        self.split_in_features = in_features // world_size
        self.quant_type = quant_type
        self.group_size = group_size
        self.dtype = dtype
        self.all_reduce = all_reduce

        if quant_type == "int4":
            if self.split_in_features % 8 != 0:
                raise ValueError(f"split_in_features ({self.split_in_features}) must be divisible by 8")
            if group_size > 0 and self.split_in_features % group_size != 0:
                raise ValueError(
                    f"split_in_features ({self.split_in_features}) must be divisible by group_size ({group_size})"
                )
            self.register_buffer(
                "packed",
                torch.zeros(
                    self.split_in_features // 8, out_features, dtype=torch.int32, device=device
                ),
            )
            num_groups = self.split_in_features // group_size if group_size > 0 else 1
            self.register_buffer(
                "scales",
                torch.ones(num_groups, out_features, dtype=dtype, device=device),
            )
            self.register_buffer(
                "zps",
                torch.full((num_groups, out_features), 8.0, dtype=dtype, device=device),
            )
            self.weight = None
            self.weight_fp16 = None
        else:
            self.weight = nn.Parameter(
                torch.empty((out_features, self.split_in_features), dtype=dtype, device=device)
            )
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            self.weight_fp16 = None

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int = 0,
        world_size: int = 2,
        quant_type: Optional[str] = None,
        group_size: int = 128,
        all_reduce: bool = True,
    ) -> TPRowParallelLinear:
        """Instantiates a sharded TPRowParallelLinear from an existing nn.Linear."""
        in_f = linear.in_features
        out_f = linear.out_features
        has_bias = linear.bias is not None

        layer = cls(
            in_features=in_f,
            out_features=out_f,
            bias=has_bias,
            rank=rank,
            world_size=world_size,
            quant_type=quant_type,
            group_size=group_size,
            dtype=linear.weight.dtype,
            device=linear.weight.device,
            all_reduce=all_reduce,
        )

        with torch.no_grad():
            if quant_type == "int4":
                packed_full, scales_full, zps_full, g_size = quantize_weight_sym_int4(
                    linear.weight, group_size=group_size
                )
                p_shard, s_shard, z_shard = slice_row_parallel_int4(
                    packed_full, scales_full, zps_full, rank, world_size, group_size=g_size
                )
                layer.packed.copy_(p_shard)
                layer.scales.copy_(s_shard)
                layer.zps.copy_(z_shard)
                layer.group_size = g_size
            else:
                w_shard = slice_row_parallel_weight(linear.weight, rank, world_size)
                layer.weight.copy_(w_shard)

            if has_bias:
                layer.bias.copy_(linear.bias.contiguous())

        return layer

    def forward(
        self,
        x: torch.Tensor,
        peer_output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        orig_shape = x.shape
        x_clean = x.contiguous()

        if x_clean.dim() == 0:
            raise ValueError("Input tensor must have at least 1 dimension")

        if x_clean.shape[-1] == self.in_features:
            start_k = self.rank * self.split_in_features
            end_k = (self.rank + 1) * self.split_in_features
            x_slice = x_clean[..., start_k:end_k].contiguous()
        elif x_clean.shape[-1] == self.split_in_features:
            x_slice = x_clean
        else:
            raise ValueError(
                f"Input trailing feature dimension ({x_clean.shape[-1]}) must match "
                f"in_features ({self.in_features}) or split_in_features ({self.split_in_features})"
            )

        x_2d = x_slice.view(-1, self.split_in_features) if x_slice.dim() != 2 else x_slice
        M = x_2d.shape[0]

        if M == 0:
            out_shape = (*orig_shape[:-1], self.out_features)
            return torch.empty(out_shape, dtype=self.dtype, device=x.device)

        if self.quant_type == "int4":
            if HAS_T4_KERNELS and x.is_cuda:
                part = t4_kernels.fused_w4a16_gemm_u4(
                    x_2d, self.packed, self.scales, self.zps, self.group_size
                )
            else:
                target_dtype = x_2d.dtype if x_2d.dtype in (torch.float16, torch.bfloat16, torch.float32) else self.dtype
                W_dequant = dequantize_sym_int4(
                    self.packed, self.scales, self.zps, self.group_size, dtype=target_dtype
                )
                if x_2d.dtype != W_dequant.dtype:
                    x_2d = x_2d.to(W_dequant.dtype)
                part = F.linear(x_2d, W_dequant, None)
        else:
            if self.weight is not None and x_2d.dtype != self.weight.dtype:
                x_2d = x_2d.to(self.weight.dtype)
            part = F.linear(x_2d, self.weight, None)

        part_out = part.view(*orig_shape[:-1], self.out_features) if orig_shape[:-1] else part.view(self.out_features)

        # Communication step: All-Reduce across ranks
        if peer_output is not None:
            reduced, _ = all_reduce_pair(part_out, peer_output)
            if self.bias is not None:
                reduced = reduced + self.bias
            return reduced
        elif self.all_reduce and torch.distributed.is_available() and torch.distributed.is_initialized():
            reduced = all_reduce_sum(part_out)
            if self.bias is not None:
                reduced = reduced + self.bias
            return reduced
        else:
            # Single-rank without communication or all_reduce=False
            if self.bias is not None:
                if self.all_reduce or self.rank == 0 or self.world_size == 1:
                    part_out = part_out + self.bias
            return part_out


class TPParallelMLP(nn.Module):
    """Intra-Layer Tensor Parallel SwiGLU MLP Block for Qwen2.5-Math-7B.
    
    Structure:
    - gate_proj: TPColumnParallelLinear (INT4)
    - up_proj: TPColumnParallelLinear (INT4)
    - activation: fused Ellie SwiGLU elementwise (zero communication)
    - down_proj: TPRowParallelLinear (INT4) + fast PCIe All-Reduce
    
    Executes exactly 1 All-Reduce per MLP block.
    """

    def __init__(
        self,
        hidden_size: int = 3584,
        intermediate_size: int = 18944,
        bias: bool = False,
        rank: int = 0,
        world_size: int = 2,
        quant_type: Optional[str] = "int4",
        group_size: int = 128,
        dtype: torch.dtype = torch.float16,
        device: Optional[torch.device] = None,
        all_reduce: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.rank = rank
        self.world_size = world_size
        self.quant_type = quant_type
        self.group_size = group_size
        self.dtype = dtype

        self.gate_proj = TPColumnParallelLinear(
            in_features=hidden_size,
            out_features=intermediate_size,
            bias=bias,
            rank=rank,
            world_size=world_size,
            quant_type=quant_type,
            group_size=group_size,
            dtype=dtype,
            device=device,
        )
        self.up_proj = TPColumnParallelLinear(
            in_features=hidden_size,
            out_features=intermediate_size,
            bias=bias,
            rank=rank,
            world_size=world_size,
            quant_type=quant_type,
            group_size=group_size,
            dtype=dtype,
            device=device,
        )
        self.down_proj = TPRowParallelLinear(
            in_features=intermediate_size,
            out_features=hidden_size,
            bias=bias,
            rank=rank,
            world_size=world_size,
            quant_type=quant_type,
            group_size=group_size,
            dtype=dtype,
            device=device,
            all_reduce=all_reduce,
        )

    @classmethod
    def from_mlp(
        cls,
        mlp: nn.Module,
        rank: int = 0,
        world_size: int = 2,
        quant_type: Optional[str] = "int4",
        group_size: int = 128,
        all_reduce: bool = True,
    ) -> TPParallelMLP:
        """Instantiates a sharded TPParallelMLP from a standard MLP block."""
        hidden_size = mlp.gate_proj.in_features
        intermediate_size = mlp.gate_proj.out_features
        has_bias = mlp.gate_proj.bias is not None

        layer = cls(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            bias=has_bias,
            rank=rank,
            world_size=world_size,
            quant_type=quant_type,
            group_size=group_size,
            dtype=mlp.gate_proj.weight.dtype,
            device=mlp.gate_proj.weight.device,
            all_reduce=all_reduce,
        )
        layer.gate_proj = TPColumnParallelLinear.from_linear(
            mlp.gate_proj, rank, world_size, quant_type, group_size
        )
        layer.up_proj = TPColumnParallelLinear.from_linear(
            mlp.up_proj, rank, world_size, quant_type, group_size
        )
        layer.down_proj = TPRowParallelLinear.from_linear(
            mlp.down_proj, rank, world_size, quant_type, group_size, all_reduce=all_reduce
        )
        return layer

    def forward(
        self,
        x: torch.Tensor,
        peer_act: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)

        # SwiGLU activation
        if HAS_T4_KERNELS and gate.is_cuda and gate.dtype == torch.float16:
            act = t4_kernels.fused_ellie_swiglu(gate.contiguous(), up.contiguous())
        else:
            act = F.silu(gate) * up

        return self.down_proj(act, peer_output=peer_act)


class TPParallelAttention(nn.Module):
    """Intra-Layer Tensor Parallel Attention with GQA and Sharded KV Cache.
    
    Structure:
    - q_proj: TPColumnParallelLinear (FP16/FP32/BF16, 28 -> 14 heads per rank)
    - k_proj: TPColumnParallelLinear (FP16/FP32/BF16, 4 -> 2 heads per rank)
    - v_proj: TPColumnParallelLinear (FP16/FP32/BF16, 4 -> 2 heads per rank)
    - static_kv_cache: local 2-head cache (50% VRAM saving)
    - local GQA scaled dot product attention (ratio 14:2 = 7:1 preserved)
    - o_proj: TPRowParallelLinear (FP16) + fast PCIe All-Reduce
    
    Executes exactly 1 All-Reduce per Attention block.
    """

    def __init__(
        self,
        hidden_size: int = 3584,
        num_heads: int = 28,
        num_kv_heads: int = 4,
        head_dim: int = 128,
        bias: bool = False,
        rank: int = 0,
        world_size: int = 2,
        dtype: torch.dtype = torch.float16,
        device: Optional[torch.device] = None,
        all_reduce: bool = True,
    ):
        super().__init__()
        if rank < 0 or rank >= world_size:
            raise ValueError(f"rank ({rank}) must be strictly less than world_size ({world_size}) and non-negative")
        if num_heads % world_size != 0:
            raise ValueError(f"num_heads ({num_heads}) must be divisible by world_size ({world_size})")
        if num_kv_heads % world_size != 0:
            raise ValueError(f"num_kv_heads ({num_kv_heads}) must be divisible by world_size ({world_size})")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rank = rank
        self.world_size = world_size
        self.dtype = dtype

        self.q_heads_per_rank = num_heads // world_size
        self.kv_heads_per_rank = num_kv_heads // world_size

        if self.q_heads_per_rank % self.kv_heads_per_rank != 0:
            raise ValueError(
                f"GQA ratio indivisible: q_heads_per_rank ({self.q_heads_per_rank}) "
                f"must be divisible by kv_heads_per_rank ({self.kv_heads_per_rank}) "
                f"(from num_heads={num_heads}, num_kv_heads={num_kv_heads}, world_size={world_size})"
            )
        self.rep_factor = self.q_heads_per_rank // self.kv_heads_per_rank

        self.q_proj = TPColumnParallelLinear(
            in_features=hidden_size,
            out_features=num_heads * head_dim,
            bias=bias,
            rank=rank,
            world_size=world_size,
            quant_type=None,
            dtype=dtype,
            device=device,
        )
        self.k_proj = TPColumnParallelLinear(
            in_features=hidden_size,
            out_features=num_kv_heads * head_dim,
            bias=bias,
            rank=rank,
            world_size=world_size,
            quant_type=None,
            dtype=dtype,
            device=device,
        )
        self.v_proj = TPColumnParallelLinear(
            in_features=hidden_size,
            out_features=num_kv_heads * head_dim,
            bias=bias,
            rank=rank,
            world_size=world_size,
            quant_type=None,
            dtype=dtype,
            device=device,
        )
        self.o_proj = TPRowParallelLinear(
            in_features=num_heads * head_dim,
            out_features=hidden_size,
            bias=bias,
            rank=rank,
            world_size=world_size,
            quant_type=None,
            dtype=dtype,
            device=device,
            all_reduce=all_reduce,
        )

    @classmethod
    def from_attention(
        cls,
        attn: nn.Module,
        rank: int = 0,
        world_size: int = 2,
        num_heads: Optional[int] = None,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        all_reduce: bool = True,
    ) -> TPParallelAttention:
        """Instantiates a sharded TPParallelAttention from a standard Attention module."""
        hidden_size = attn.q_proj.in_features
        q_out = attn.q_proj.out_features
        k_out = attn.k_proj.out_features

        if num_heads is None:
            num_heads = getattr(attn, "num_heads", getattr(attn, "num_attention_heads", None))
        if head_dim is None:
            head_dim = getattr(attn, "head_dim", None)
        if num_kv_heads is None:
            num_kv_heads = getattr(attn, "num_kv_heads", getattr(attn, "num_key_value_heads", None))

        if head_dim is None and num_heads is not None:
            head_dim = q_out // num_heads
        elif num_heads is None and head_dim is not None:
            num_heads = q_out // head_dim
        elif head_dim is None and num_heads is None:
            head_dim = 128
            num_heads = q_out // head_dim

        if num_kv_heads is None:
            num_kv_heads = max(1, k_out // head_dim)

        has_bias = attn.q_proj.bias is not None

        layer = cls(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            bias=has_bias,
            rank=rank,
            world_size=world_size,
            dtype=attn.q_proj.weight.dtype,
            device=attn.q_proj.weight.device,
            all_reduce=all_reduce,
        )
        layer.q_proj = TPColumnParallelLinear.from_linear(attn.q_proj, rank, world_size)
        layer.k_proj = TPColumnParallelLinear.from_linear(attn.k_proj, rank, world_size)
        layer.v_proj = TPColumnParallelLinear.from_linear(attn.v_proj, rank, world_size)
        layer.o_proj = TPRowParallelLinear.from_linear(
            attn.o_proj, rank, world_size, all_reduce=all_reduce
        )
        return layer

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[Any] = None,
        layer_idx: int = 0,
        attention_mask: Optional[torch.Tensor] = None,
        start_pos: Optional[int] = None,
        peer_out: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        orig_shape = x.shape
        if x.dim() == 0:
            raise ValueError("Input tensor must have at least 1 dimension")
        if x.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Input trailing feature dimension ({x.shape[-1]}) must match hidden_size ({self.hidden_size})"
            )

        is_1d = x.dim() == 1
        is_2d = x.dim() == 2

        if is_1d:
            x_3d = x.unsqueeze(0).unsqueeze(0)
        elif is_2d:
            x_3d = x.unsqueeze(0)
        elif x.dim() == 3:
            x_3d = x
        else:
            x_3d = x.view(-1, orig_shape[-2], self.hidden_size)

        b, m, _ = x_3d.shape

        if b == 0 or m == 0:
            out_shape = (*orig_shape[:-1], self.hidden_size)
            return torch.empty(out_shape, dtype=self.dtype, device=x.device)

        q = self.q_proj(x_3d).view(b, m, self.q_heads_per_rank, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_3d).view(b, m, self.kv_heads_per_rank, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_3d).view(b, m, self.kv_heads_per_rank, self.head_dim).transpose(1, 2)

        # Update sharded KV cache (stores 2 heads per rank, saving 50% memory)
        if past_key_value is not None:
            if hasattr(past_key_value, "update"):
                try:
                    k, v = past_key_value.update(
                        layer_idx=layer_idx, k_new=k, v_new=v, start_pos=start_pos
                    )
                except TypeError:
                    k, v = past_key_value.update(k, v, layer_idx, kwargs.get("cache_kwargs", None))

        # GQA repeat interleave (e.g. 2 KV heads repeated 7 times -> 14 heads)
        k_rep = k.repeat_interleave(self.rep_factor, dim=1)
        v_rep = v.repeat_interleave(self.rep_factor, dim=1)

        # Scaled dot-product attention
        scores = torch.matmul(q, k_rep.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            scores = scores + attention_mask
        attn_weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v_rep).transpose(1, 2).contiguous().view(
            b, m, self.q_heads_per_rank * self.head_dim
        )

        # Row-parallel output projection + All-Reduce
        out = self.o_proj(attn_out, peer_output=peer_out)

        if is_1d:
            return out.view(self.hidden_size)
        if is_2d:
            return out.view(orig_shape[0], self.hidden_size)
        if orig_shape[:-1]:
            return out.view(*orig_shape[:-1], self.hidden_size)
        return out
