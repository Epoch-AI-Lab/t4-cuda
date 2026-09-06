"""src/sharding/pp.py

Pipeline Parallelism Partitioning for Dual Tesla T4 GPUs.
Partitions 28-layer Qwen2.5-Math-7B into 2 stages:
- Stage 0 (GPU 0): Embedding + Layers 0..13
- Stage 1 (GPU 1): Layers 14..27 + Final RMSNorm + LM Head
Boundary transfer uses asynchronous CUDA stream copy across PCIe Gen3.
"""

from typing import List, Optional, Tuple, Union, Dict, Any
import torch
import torch.nn as nn
from src.sharding.comm import p2p_transfer, check_dual_gpu, is_cuda_available


class PipelineStage(nn.Module):
    """Executes a sequential stage of layers on a designated device."""

    def __init__(
        self,
        stage_idx: int,
        device: torch.device,
        layers: nn.ModuleList,
        pre_module: Optional[nn.Module] = None,
        post_module: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.stage_idx = stage_idx
        self.device = torch.device(device)
        self.layers = layers
        self.pre_module = pre_module
        self.post_module = post_module

        # Move components to designated device
        if self.pre_module is not None:
            self.pre_module.to(self.device)
        self.layers.to(self.device)
        if self.post_module is not None:
            self.post_module.to(self.device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        use_cache: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        """Forward pass through the local stage layers."""
        # Ensure input is on this stage's device
        if hidden_states.device != self.device:
            hidden_states = hidden_states.to(self.device, non_blocking=True)

        if self.pre_module is not None:
            hidden_states = self.pre_module(hidden_states)

        next_past = [] if use_cache else None

        for idx, layer in enumerate(self.layers):
            layer_past = past_key_values[idx] if past_key_values is not None else None
            
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=layer_past,
                use_cache=use_cache,
                **kwargs
            )

            if isinstance(layer_outputs, tuple):
                hidden_states = layer_outputs[0]
                if use_cache and len(layer_outputs) > 1:
                    next_past.append(layer_outputs[1])
            else:
                hidden_states = layer_outputs

        if self.post_module is not None:
            hidden_states = self.post_module(hidden_states)

        return hidden_states, next_past


class PipelineParallelQwen2(nn.Module):
    """Partitions a 28-layer Qwen2 model across dual devices for PP=2 execution."""

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        num_layers: int = 28,
        hidden_size: int = 3584,
        vocab_size: int = 152064,
        split_layer: Optional[int] = None,
        devices: Optional[List[Union[str, torch.device]]] = None,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.split_layer = split_layer if split_layer is not None else num_layers // 2
        self.dtype = dtype

        if devices is None:
            if is_cuda_available() and check_dual_gpu()[0]:
                self.devices = [torch.device("cuda:0"), torch.device("cuda:1")]
            elif is_cuda_available():
                self.devices = [torch.device("cuda:0"), torch.device("cuda:0")]
            else:
                self.devices = [torch.device("cpu"), torch.device("cpu")]
        else:
            self.devices = [torch.device(d) for d in devices]

        if len(self.devices) != 2:
            raise ValueError(f"PipelineParallelQwen2 requires exactly 2 devices, got {self.devices}")

        if model is not None:
            self._init_from_model(model)
        else:
            self._init_mock_architecture()

    def _init_from_model(self, model: nn.Module):
        """Partition an existing Qwen2ForCausalLM model across 2 stages."""
        base_model = getattr(model, "model", model)

        # Stage 0: Embeddings + Layers 0 .. split_layer - 1
        self.embed = getattr(base_model, "embed_tokens", None)
        all_layers = getattr(base_model, "layers", None)
        if all_layers is None:
            raise ValueError("Model does not have a recognizable .layers or .model.layers structure")

        self.num_layers = len(all_layers)
        self.split_layer = self.num_layers // 2

        self.layers_stage0 = nn.ModuleList([all_layers[i] for i in range(self.split_layer)])
        self.layers_stage1 = nn.ModuleList([all_layers[i] for i in range(self.split_layer, self.num_layers)])

        self.norm = getattr(base_model, "norm", None)
        self.lm_head = getattr(model, "lm_head", None)

        self.stage0 = PipelineStage(
            stage_idx=0,
            device=self.devices[0],
            layers=self.layers_stage0,
            pre_module=self.embed,
            post_module=None,
        )

        class PostHead(nn.Module):
            def __init__(self, norm, lm_head):
                super().__init__()
                self.norm = norm
                self.lm_head = lm_head

            def forward(self, x):
                if self.norm is not None:
                    x = self.norm(x)
                if self.lm_head is not None:
                    x = self.lm_head(x)
                return x

        self.stage1 = PipelineStage(
            stage_idx=1,
            device=self.devices[1],
            layers=self.layers_stage1,
            pre_module=None,
            post_module=PostHead(self.norm, self.lm_head),
        )

    def _init_mock_architecture(self):
        """Initializes architectural modules for testing/benchmarking without full weights."""
        self.embed = nn.Embedding(self.vocab_size, self.hidden_size, dtype=self.dtype)
        
        class SimpleBlock(nn.Module):
            def __init__(self, d_model: int, dtype: torch.dtype):
                super().__init__()
                self.linear1 = nn.Linear(d_model, d_model * 2, bias=False, dtype=dtype)
                self.linear2 = nn.Linear(d_model * 2, d_model, bias=False, dtype=dtype)

            def forward(self, x, **kwargs):
                return x + self.linear2(torch.relu(self.linear1(x)))

        self.layers_stage0 = nn.ModuleList([
            SimpleBlock(self.hidden_size, self.dtype) for _ in range(self.split_layer)
        ])
        self.layers_stage1 = nn.ModuleList([
            SimpleBlock(self.hidden_size, self.dtype) for _ in range(self.num_layers - self.split_layer)
        ])
        self.norm = nn.LayerNorm(self.hidden_size, dtype=self.dtype)
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False, dtype=self.dtype)

        self.stage0 = PipelineStage(
            stage_idx=0,
            device=self.devices[0],
            layers=self.layers_stage0,
            pre_module=self.embed,
            post_module=None,
        )

        class PostHead(nn.Module):
            def __init__(self, norm, lm_head):
                super().__init__()
                self.norm = norm
                self.lm_head = lm_head

            def forward(self, x):
                return self.lm_head(self.norm(x))

        self.stage1 = PipelineStage(
            stage_idx=1,
            device=self.devices[1],
            layers=self.layers_stage1,
            pre_module=None,
            post_module=PostHead(self.norm, self.lm_head),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        use_cache: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """Executes forward pass through Stage 0, transfers boundary activations, and completes Stage 1."""
        past_s0 = None
        past_s1 = None
        if past_key_values is not None:
            past_s0 = past_key_values[:self.split_layer]
            past_s1 = past_key_values[self.split_layer:]

        input_ids_s0 = input_ids.to(self.devices[0], non_blocking=True)
        h_stage0, next_past_s0 = self.stage0(
            input_ids_s0,
            attention_mask=attention_mask.to(self.devices[0]) if attention_mask is not None else None,
            position_ids=position_ids.to(self.devices[0]) if position_ids is not None else None,
            past_key_values=past_s0,
            use_cache=use_cache,
            **kwargs
        )

        if self.devices[0] != self.devices[1]:
            h_boundary = p2p_transfer(h_stage0, dst_device=self.devices[1])
        else:
            h_boundary = h_stage0

        logits, next_past_s1 = self.stage1(
            h_boundary,
            attention_mask=attention_mask.to(self.devices[1]) if attention_mask is not None else None,
            position_ids=position_ids.to(self.devices[1]) if position_ids is not None else None,
            past_key_values=past_s1,
            use_cache=use_cache,
            **kwargs
        )

        return logits
