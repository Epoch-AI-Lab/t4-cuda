"""src/sharding/scope.py

Unified Component-and-Phase Multi-GPU Inference Scope (CPMultiGPUInferenceScope).
Supports dynamic model transformation between:
- 'tp': Intra-layer Tensor Parallelism (column/row linear sharding with PCIe All-Reduce)
- 'pp': Inter-layer Pipeline Parallelism (2-stage partitioning with boundary activation transfer)
Cleanly restores original architecture on context exit.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Union, Any, Tuple
import torch
import torch.nn as nn
from src.sharding.tp import TPParallelMLP, TPParallelAttention, TPColumnParallelLinear, TPRowParallelLinear
from src.sharding.pp import PipelineParallelQwen2
from src.sharding.comm import check_dual_gpu, is_cuda_available


class CPMultiGPUInferenceScope:
    """Dynamic context manager for multi-GPU inference across dual Tesla T4 GPUs."""

    def __init__(
        self,
        model: nn.Module,
        mode: str = "tp",
        devices: Optional[List[Union[str, torch.device]]] = None,
        quantize_mlp: bool = True,
    ):
        if mode not in ("tp", "pp"):
            raise ValueError(f"Invalid mode '{mode}'. Must be 'tp' or 'pp'.")

        if devices is None:
            if is_cuda_available() and check_dual_gpu():
                self.devices = [torch.device("cuda:0"), torch.device("cuda:1")]
            elif is_cuda_available():
                self.devices = [torch.device("cuda:0"), torch.device("cuda:0")]
            else:
                self.devices = [torch.device("cpu"), torch.device("cpu")]
        else:
            self.devices = [torch.device(d) for d in devices]

        if len(self.devices) != 2:
            raise ValueError(f"CPMultiGPUInferenceScope requires exactly 2 devices, got {self.devices}")

        self.model = model
        self.mode = mode
        self.quantize_mlp = quantize_mlp
        self.is_active = False
        self._saved_modules: Dict[str, nn.Module] = {}
        self._saved_devices: Dict[str, torch.device] = {}
        self._pp_wrapper: Optional[PipelineParallelQwen2] = None

    def __enter__(self):
        self.is_active = True
        if self.mode == "tp":
            self._apply_tp()
        elif self.mode == "pp":
            self._apply_pp()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.mode == "tp":
            self._restore_tp()
        elif self.mode == "pp":
            self._restore_pp()
        self.is_active = False

    def _apply_tp(self):
        """Replaces attention and MLP linear projections with tensor-parallel counterparts."""
        self._saved_modules.clear()
        
        # Identify layers in standard transformer
        base_model = getattr(self.model, "model", self.model)
        layers = getattr(base_model, "layers", None)
        if layers is None:
            # Check if model has top-level module names
            for name, mod in list(self.model.named_modules()):
                if isinstance(mod, nn.Linear):
                    parent_name, child_name = self._split_module_path(name)
                    parent = self.model.get_submodule(parent_name) if parent_name else self.model
                    self._saved_modules[name] = mod
            return

        for layer_idx, layer in enumerate(layers):
            # 1. Shard MLP if present
            mlp = getattr(layer, "mlp", None)
            if mlp is not None:
                self._saved_modules[f"layer_{layer_idx}_mlp"] = mlp
                gate_proj = getattr(mlp, "gate_proj", None)
                up_proj = getattr(mlp, "up_proj", None)
                down_proj = getattr(mlp, "down_proj", None)
                if gate_proj is not None and up_proj is not None and down_proj is not None:
                    tp_mlp = TPParallelMLP.from_mlp(
                        mlp=mlp,
                        rank=0,
                        world_size=2,
                        quant_type="int4" if self.quantize_mlp else None,
                    )
                    layer.mlp = tp_mlp

            # 2. Shard Self-Attention if present
            self_attn = getattr(layer, "self_attn", None)
            if self_attn is not None:
                self._saved_modules[f"layer_{layer_idx}_self_attn"] = self_attn
                q_proj = getattr(self_attn, "q_proj", None)
                k_proj = getattr(self_attn, "k_proj", None)
                v_proj = getattr(self_attn, "v_proj", None)
                o_proj = getattr(self_attn, "o_proj", None)
                if q_proj and k_proj and v_proj and o_proj:
                    num_heads = getattr(self_attn, "num_heads", q_proj.out_features // 128)
                    num_kv_heads = getattr(self_attn, "num_key_value_heads", k_proj.out_features // 128)
                    tp_attn = TPParallelAttention.from_attention(
                        attn=self_attn,
                        rank=0,
                        world_size=2,
                        num_heads=num_heads,
                        num_kv_heads=num_kv_heads,
                    )
                    layer.self_attn = tp_attn

    def _restore_tp(self):
        """Restores original un-sharded linear and attention modules."""
        base_model = getattr(self.model, "model", self.model)
        layers = getattr(base_model, "layers", None)
        if layers is not None:
            for layer_idx, layer in enumerate(layers):
                saved_mlp = self._saved_modules.get(f"layer_{layer_idx}_mlp")
                if saved_mlp is not None:
                    layer.mlp = saved_mlp
                saved_attn = self._saved_modules.get(f"layer_{layer_idx}_self_attn")
                if saved_attn is not None:
                    layer.self_attn = saved_attn
        self._saved_modules.clear()

    def _apply_pp(self):
        """Partitions model layers across devices."""
        self._saved_devices.clear()
        base_model = getattr(self.model, "model", self.model)
        layers = getattr(base_model, "layers", None)
        if layers is not None:
            split_layer = len(layers) // 2
            for i in range(split_layer):
                for param in layers[i].parameters():
                    self._saved_devices[f"layer_{i}"] = param.device
                    break
                layers[i].to(self.devices[0])

            for i in range(split_layer, len(layers)):
                for param in layers[i].parameters():
                    self._saved_devices[f"layer_{i}"] = param.device
                    break
                layers[i].to(self.devices[1])

    def _restore_pp(self):
        """Restores original device locations."""
        base_model = getattr(self.model, "model", self.model)
        layers = getattr(base_model, "layers", None)
        if layers is not None:
            for i, layer in enumerate(layers):
                orig_device = self._saved_devices.get(f"layer_{i}")
                if orig_device is not None:
                    layer.to(orig_device)
        self._saved_devices.clear()

    @staticmethod
    def _split_module_path(path: str) -> Tuple[str, str]:
        if "." in path:
            parts = path.rsplit(".", 1)
            return parts[0], parts[1]
        return "", path
