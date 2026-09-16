"""Static Tensor Key-Value Cache with In-Place Updates and O(1) Rollback.

Provides pre-allocated fixed-shape GPU buffers for transformer attention,
eliminating dynamic memory allocations, allocator lock contention, and
enabling stable memory pointers for CUDA Graph execution on NVIDIA Tesla T4.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import torch


def build_causal_4d_mask(
    batch_size: int,
    query_len: int,
    past_kv_len: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Constructs dynamic 4D causal attention mask for speculative verification.

    The mask tensor has shape (batch_size, 1, query_len, past_kv_len + query_len).
    Positions corresponding to past tokens and causally valid query positions
    have value 0.0. Future query positions have value min_val (-inf).
    """
    total_kv_len = past_kv_len + query_len
    if dtype.is_floating_point:
        min_val = torch.finfo(dtype).min
    else:
        min_val = -1e9

    q_pos = torch.arange(past_kv_len, total_kv_len, device=device).unsqueeze(1)
    k_pos = torch.arange(0, total_kv_len, device=device).unsqueeze(0)

    # Valid if key position <= query position
    causal_mask = torch.where(k_pos <= q_pos, 0.0, min_val).to(dtype)
    return causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, query_len, total_kv_len)


def compute_position_ids(
    start_pos: int,
    num_tokens: int,
    batch_size: int = 1,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Constructs explicit 2D position_ids tensor of shape (batch_size, num_tokens)."""
    if start_pos < 0:
        raise ValueError(f"start_pos must be non-negative, got {start_pos}")
    if num_tokens < 0:
        raise ValueError(f"num_tokens must be non-negative, got {num_tokens}")
    pos = torch.arange(start_pos, start_pos + num_tokens, dtype=torch.long, device=device)
    return pos.unsqueeze(0).expand(batch_size, -1)


def _parse_cache_position(pos_val: Any, fallback: Optional[int] = None) -> Optional[int]:
    """Extracts integer start_pos from cache_position supporting tensors, scalars, and lists."""
    if pos_val is None:
        return fallback
    if isinstance(pos_val, (int, float)):
        return int(pos_val)
    if isinstance(pos_val, torch.Tensor):
        if pos_val.numel() == 0:
            return fallback
        return int(pos_val.reshape(-1)[0].item())
    try:
        import numpy as np
        if isinstance(pos_val, np.ndarray):
            if pos_val.size == 0:
                return fallback
            return int(pos_val.reshape(-1)[0].item())
    except ImportError:
        pass
    if isinstance(pos_val, (list, tuple)):
        if len(pos_val) == 0:
            return fallback
        elem = pos_val[0]
        if isinstance(elem, torch.Tensor):
            return int(elem.reshape(-1)[0].item())
        return int(elem)
    if hasattr(pos_val, "__int__"):
        return int(pos_val)
    raise TypeError(f"Unsupported cache_position type: {type(pos_val)}")


class StaticKVCache:
    """Pre-allocated static tensor KV cache with in-place updates and O(1) rollback."""

    def __init__(
        self,
        max_batch_size: int = 1,
        max_seq_len: int = 16384,
        num_layers: int = 28,
        num_heads: int = 2,
        head_dim: int = 128,
        device: Union[str, torch.device] = "cuda",
        dtype: Optional[torch.dtype] = None,
    ):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Resolve device with CPU fallback if CUDA is unavailable
        if isinstance(device, str):
            if device.startswith("cuda") and not torch.cuda.is_available():
                self.device = torch.device("cpu")
            else:
                self.device = torch.device(device)
        elif isinstance(device, torch.device):
            if device.type == "cuda" and not torch.cuda.is_available():
                self.device = torch.device("cpu")
            else:
                self.device = device
        else:
            self.device = torch.device("cpu")

        if dtype is None:
            # Default to float16 for 16k context scalability,
            # retaining float32 default for legacy short sequences
            self.dtype = torch.float16 if max_seq_len >= 16384 else torch.float32
        else:
            self.dtype = dtype

        # Integer write pointer tracking the current committed sequence length
        self.current_pos = 0

        # Per-layer write pointers to support independent layer updates
        self._layer_seen_tokens: List[int] = [0] * num_layers

        # Monolithic contiguous buffer: [num_layers, 2 (K/V), batch, heads, seq, dim]
        self.storage = torch.zeros(
            (num_layers, 2, max_batch_size, num_heads, max_seq_len, head_dim),
            dtype=self.dtype,
            device=self.device,
        )

        # Pre-computed slice views for fast per-layer access without tensor creation
        self.key_cache: List[torch.Tensor] = [
            self.storage[i, 0] for i in range(num_layers)
        ]
        self.value_cache: List[torch.Tensor] = [
            self.storage[i, 1] for i in range(num_layers)
        ]

    @property
    def k_cache(self) -> torch.Tensor:
        """Tensor view of key cache with shape (num_layers, max_batch_size, num_heads, max_seq_len, head_dim)."""
        return self.storage[:, 0]

    @property
    def v_cache(self) -> torch.Tensor:
        """Tensor view of value cache with shape (num_layers, max_batch_size, num_heads, max_seq_len, head_dim)."""
        return self.storage[:, 1]

    @property
    def current_seq_len(self) -> int:
        """Alias property matching current_pos for compatibility."""
        return self.current_pos

    @current_seq_len.setter
    def current_seq_len(self, value: int) -> None:
        self.rollback(value)

    @classmethod
    def from_model_config(
        cls,
        config: Any,
        max_batch_size: int = 1,
        max_seq_len: Optional[int] = None,
        device: Union[str, torch.device] = "cuda",
        dtype: Optional[torch.dtype] = None,
    ) -> "StaticKVCache":
        """Constructs StaticKVCache from a HuggingFace PretrainedConfig."""
        if max_seq_len is None:
            max_seq_len = getattr(config, "max_position_embeddings", None)
            if max_seq_len is None:
                max_seq_len = 2048

        num_layers = getattr(config, "num_hidden_layers", None)
        if num_layers is None:
            num_layers = getattr(config, "n_layer", 28)

        num_attn_heads = getattr(config, "num_attention_heads", None)
        if num_attn_heads is None:
            num_attn_heads = getattr(config, "n_head", 2)

        num_kv_heads = getattr(config, "num_key_value_heads", None)
        if num_kv_heads is None:
            num_kv_heads = num_attn_heads

        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(config, "n_embd", 1536)

        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = getattr(config, "kv_channels", None)
        if head_dim is None:
            head_dim = hidden_size // num_attn_heads

        if dtype is None:
            dtype = torch.float16 if max_seq_len >= 16384 else torch.float32

        return cls(
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            num_layers=num_layers,
            num_heads=num_kv_heads,
            head_dim=head_dim,
            device=device,
            dtype=dtype,
        )

    def update(
        self,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """In-place cache update supporting both serving engine and HF Cache calling conventions.

        Convention 1 (Serving Engine):
            update(layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor, start_pos: int = None)

        Convention 2 (HuggingFace Transformers Cache):
            update(key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, cache_kwargs=None)
        """
        if len(args) > 0 and isinstance(args[0], int):
            # Serving Engine convention: layer_idx, k_new, v_new, [start_pos]
            layer_idx = args[0]
            k_new = args[1]
            v_new = args[2]
            if len(args) > 3:
                start_pos = args[3]
            elif "start_pos" in kwargs:
                start_pos = kwargs["start_pos"]
            else:
                start_pos = self._layer_seen_tokens[layer_idx]
        elif len(args) >= 2 and not isinstance(args[0], int):
            # HuggingFace Cache convention: key_states, value_states, layer_idx, [cache_kwargs]
            k_new = args[0]
            v_new = args[1]
            layer_idx = args[2] if len(args) > 2 else kwargs.get("layer_idx", 0)
            cache_kwargs = args[3] if len(args) > 3 else kwargs.get("cache_kwargs", None)
            if "start_pos" in kwargs and kwargs["start_pos"] is not None:
                start_pos = kwargs["start_pos"]
            elif cache_kwargs and "cache_position" in cache_kwargs:
                start_pos = _parse_cache_position(
                    cache_kwargs["cache_position"],
                    fallback=self._layer_seen_tokens[layer_idx],
                )
            else:
                start_pos = self._layer_seen_tokens[layer_idx]
        elif "layer_idx" in kwargs:
            layer_idx = kwargs["layer_idx"]
            k_new = kwargs.get("k_new", kwargs.get("key_states"))
            v_new = kwargs.get("v_new", kwargs.get("value_states"))
            if "start_pos" in kwargs and kwargs["start_pos"] is not None:
                start_pos = kwargs["start_pos"]
            elif "cache_kwargs" in kwargs and kwargs["cache_kwargs"] and "cache_position" in kwargs["cache_kwargs"]:
                start_pos = _parse_cache_position(
                    kwargs["cache_kwargs"]["cache_position"],
                    fallback=self._layer_seen_tokens[layer_idx],
                )
            else:
                start_pos = self._layer_seen_tokens[layer_idx]
        else:
            raise TypeError("Unsupported argument combination for StaticKVCache.update()")

        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise IndexError(f"layer_idx {layer_idx} out of bounds for num_layers {self.num_layers}")

        # Resolve start_pos fallback and type
        if start_pos is None:
            start_pos = self._layer_seen_tokens[layer_idx]
        elif not isinstance(start_pos, int):
            start_pos = int(start_pos)

        # Validate non-negative start_pos to avoid negative slice wrap-around
        if start_pos < 0:
            raise ValueError(f"start_pos must be non-negative, got {start_pos}")

        # Convert numpy or non-tensor inputs
        if not isinstance(k_new, torch.Tensor):
            k_new = torch.as_tensor(k_new, dtype=self.dtype, device=self.device)
        elif k_new.dtype != self.dtype or k_new.device != self.device:
            k_new = k_new.to(dtype=self.dtype, device=self.device)

        if not isinstance(v_new, torch.Tensor):
            v_new = torch.as_tensor(v_new, dtype=self.dtype, device=self.device)
        elif v_new.dtype != self.dtype or v_new.device != self.device:
            v_new = v_new.to(dtype=self.dtype, device=self.device)

        # Expand 3D inputs: (num_heads, seq_len, head_dim) -> (1, num_heads, seq_len, head_dim)
        if k_new.dim() == 3:
            k_new = k_new.unsqueeze(0)
        if v_new.dim() == 3:
            v_new = v_new.unsqueeze(0)

        # Adapt head_dim if initial unpopulated write has different head dimension
        if self.current_pos == 0 and k_new.shape[-1] != self.head_dim:
            self.head_dim = k_new.shape[-1]
            self.storage = torch.zeros(
                (self.num_layers, 2, self.max_batch_size, self.num_heads, self.max_seq_len, self.head_dim),
                dtype=self.dtype,
                device=self.device,
            )
            self.key_cache = [self.storage[i, 0] for i in range(self.num_layers)]
            self.value_cache = [self.storage[i, 1] for i in range(self.num_layers)]

        batch_size, num_heads, num_new_tokens, head_dim = k_new.shape
        end_pos = start_pos + num_new_tokens

        if end_pos > self.max_seq_len:
            raise ValueError(
                f"Sequence length overflow: end_pos ({end_pos}) > max_seq_len ({self.max_seq_len})"
            )
        if batch_size > self.max_batch_size:
            raise ValueError(
                f"Batch size overflow: batch_size ({batch_size}) > max_batch_size ({self.max_batch_size})"
            )

        # In-place write into pre-allocated contiguous memory buffer
        self.key_cache[layer_idx][:batch_size, :num_heads, start_pos:end_pos, :head_dim].copy_(k_new)
        self.value_cache[layer_idx][:batch_size, :num_heads, start_pos:end_pos, :head_dim].copy_(v_new)

        # Update layer tracking pointer and global committed position
        self._layer_seen_tokens[layer_idx] = end_pos
        self.current_pos = max(self._layer_seen_tokens)

        # Return active slice view up to end_pos
        return (
            self.key_cache[layer_idx][:batch_size, :num_heads, :end_pos, :head_dim],
            self.value_cache[layer_idx][:batch_size, :num_heads, :end_pos, :head_dim],
        )

    def rollback(self, new_pos: int) -> None:
        """O(1) rollback resetting the write pointer without memory copying or reallocation."""
        if new_pos < 0 or new_pos > self.max_seq_len:
            raise ValueError(
                f"Invalid rollback position: {new_pos}. Valid range: [0, {self.max_seq_len}]."
            )
        self.current_pos = new_pos
        for i in range(self.num_layers):
            self._layer_seen_tokens[i] = new_pos

    def crop(self, max_length: int) -> None:
        """HF Cache compatible crop method supporting negative drop lengths and absolute lengths."""
        if max_length < 0:
            target_pos = max(0, self.current_pos + max_length)
            self.rollback(target_pos)
        else:
            self.rollback(max_length)

    def get_valid_cache(
        self,
        layer_idx: int,
        current_pos: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the active slice view of keys and values up to current_pos."""
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise IndexError(f"layer_idx {layer_idx} out of bounds for num_layers {self.num_layers}")
        pos = current_pos if current_pos is not None else self.current_pos
        bs = batch_size if batch_size is not None else self.max_batch_size
        return (
            self.key_cache[layer_idx][:bs, :, :pos, :],
            self.value_cache[layer_idx][:bs, :, :pos, :],
        )

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the valid sequence length for the specified layer."""
        if layer_idx is not None and 0 <= layer_idx < self.num_layers:
            return self._layer_seen_tokens[layer_idx]
        return self.current_pos

    def get_max_length(self) -> int:
        """Returns the maximum pre-allocated sequence length."""
        return self.max_seq_len

    def get_max_cache_shape(self) -> int:
        """Returns the maximum cache capacity for HF Cache compatibility."""
        return self.max_seq_len

    def reset(self) -> None:
        """Resets the cache write pointer for a new generation sequence in O(1) time."""
        self.current_pos = 0
        for i in range(self.num_layers):
            self._layer_seen_tokens[i] = 0

    def build_causal_4d_mask(
        self,
        query_len: int,
        past_kv_len: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Builds a 4D causal attention mask for the current cache state."""
        past_len = past_kv_len if past_kv_len is not None else self.current_pos
        bs = batch_size if batch_size is not None else self.max_batch_size
        return build_causal_4d_mask(
            batch_size=bs,
            query_len=query_len,
            past_kv_len=past_len,
            device=self.device,
            dtype=self.dtype,
        )

    def get_position_ids(
        self,
        num_tokens: int,
        start_pos: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Builds explicit position_ids for new candidate tokens."""
        pos = start_pos if start_pos is not None else self.current_pos
        bs = batch_size if batch_size is not None else self.max_batch_size
        return compute_position_ids(
            start_pos=pos,
            num_tokens=num_tokens,
            batch_size=bs,
            device=self.device,
        )

    def __len__(self) -> int:
        return self.num_layers

    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.get_valid_cache(layer_idx)
