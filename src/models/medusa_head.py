"""Medusa Multi-Head Self-Speculative Draft Architecture.

Implements multi-token speculative lookahead heads attached to the base model's
final hidden states with shared lm_head weight tying, tree attention mask
construction, and fast candidate generation for NVIDIA Tesla T4.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Residual block for Medusa speculative decoding head.

    Computes: h + Linear_out(SiLU(Linear_in(h)))
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.linear_in = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.act = nn.SiLU()
        self.linear_out = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Initialize weights with standard deviation 0.02
        nn.init.normal_(self.linear_in.weight, std=0.02)
        nn.init.normal_(self.linear_out.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.linear_out(self.act(self.linear_in(x)))


class SharedLMHeadLinearWrapper(nn.Module):
    """Wrapper around an nn.Linear layer exposing .shape and weight attributes."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.linear = linear

    @property
    def shape(self) -> torch.Size:
        return self.linear.weight.shape

    @property
    def weight(self) -> nn.Parameter:
        return self.linear.weight

    @property
    def dtype(self) -> torch.dtype:
        return self.linear.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.linear.weight.device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.linear, name)


def build_tree_attention_mask(
    context_len: int,
    parent_indices: List[int],
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float32,
    as_4d: bool = False,
    batch_size: int = 1,
    num_heads: int = 1,
) -> torch.Tensor:
    """Constructs 2D or 4D tree causal mask forbidding cross-branch candidate leakage.

    Args:
        context_len: Length of past verified context S.
        parent_indices: Parent node index for each candidate in [0..M-1], or -1 for root.
        device: PyTorch device.
        dtype: Floating point precision for mask values.
        as_4d: Whether to expand to (batch_size, num_heads, M, S + M).
        batch_size: Batch size dimension for 4D mask.
        num_heads: Attention heads dimension for 4D mask.

    Returns:
        Tensor of shape (M, S + M) or (batch_size, num_heads, M, S + M).
    """
    M = len(parent_indices)
    total_len = context_len + M
    min_val = -1e9

    # Determine ancestors for each node: ancestors[i][j] is True if j is an ancestor of i
    ancestors = [[False] * M for _ in range(M)]
    for i in range(M):
        curr = i
        while curr != -1:
            ancestors[i][curr] = True
            curr = parent_indices[curr]

    mask = torch.zeros((M, total_len), dtype=dtype, device=device)
    # Positions 0..context_len - 1 attend to past context (remain 0.0)
    # Positions context_len..total_len - 1 attend only to ancestors in tree
    for i in range(M):
        for j in range(M):
            if not ancestors[i][j]:
                mask[i, context_len + j] = min_val

    if as_4d:
        return mask.unsqueeze(0).unsqueeze(1).expand(batch_size, num_heads, M, total_len)
    return mask


def compute_tree_position_ids(
    context_len: int,
    parent_indices: List[int],
    batch_size: int = 1,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """Computes explicit position_ids tensor matching candidate tree node depths.

    Args:
        context_len: Length of past verified context S.
        parent_indices: Parent node index for each candidate in [0..M-1], or -1 for root.
        batch_size: Batch size dimension.
        device: PyTorch device.

    Returns:
        LongTensor of shape (batch_size, M).
    """
    M = len(parent_indices)
    depths = [0] * M
    for i in range(M):
        d = 1
        curr = parent_indices[i]
        while curr != -1:
            d += 1
            curr = parent_indices[curr]
        depths[i] = d

    pos = [context_len - 1 + d for d in depths]
    pos_tensor = torch.tensor(pos, dtype=torch.long, device=device)
    return pos_tensor.unsqueeze(0).expand(batch_size, -1)


class MedusaHead(nn.Module):
    """Self-speculative multi-token prediction heads sharing base model weights."""

    def __init__(
        self,
        hidden_dim: int = 1536,
        num_heads: int = 3,
        vocab_size: int = 32000,
        shared_lm_head: Optional[Any] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.vocab_size = vocab_size

        # In-head ResBlocks
        self.blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(num_heads)])

        # Weight tying with base model lm_head
        if shared_lm_head is not None:
            if isinstance(shared_lm_head, nn.Linear):
                self.shared_lm_head = SharedLMHeadLinearWrapper(shared_lm_head)
            elif isinstance(shared_lm_head, nn.Parameter):
                self.shared_lm_head = shared_lm_head
            elif isinstance(shared_lm_head, torch.Tensor):
                self.shared_lm_head = nn.Parameter(shared_lm_head, requires_grad=False)
            elif isinstance(shared_lm_head, np.ndarray):
                t_shared = torch.from_numpy(shared_lm_head).float()
                self.shared_lm_head = nn.Parameter(t_shared, requires_grad=False)
            elif hasattr(shared_lm_head, "weight") and isinstance(shared_lm_head, nn.Module):
                self.shared_lm_head = SharedLMHeadLinearWrapper(shared_lm_head)
            elif hasattr(shared_lm_head, "shape"):
                self.shared_lm_head = shared_lm_head
            else:
                raise TypeError(f"Unsupported shared_lm_head type: {type(shared_lm_head)}")
        else:
            self.shared_lm_head = nn.Parameter(
                torch.randn(vocab_size, hidden_dim) * 0.02, requires_grad=False
            )

    @property
    def w_in(self) -> List[torch.Tensor]:
        return [b.linear_in.weight for b in self.blocks]

    @w_in.setter
    def w_in(self, values: List[Any]) -> None:
        for b, w in zip(self.blocks, values):
            if isinstance(w, torch.Tensor):
                b.linear_in.weight = nn.Parameter(w)
            elif isinstance(w, np.ndarray):
                b.linear_in.weight = nn.Parameter(torch.from_numpy(w).float())

    @property
    def w_out(self) -> List[torch.Tensor]:
        return [b.linear_out.weight for b in self.blocks]

    @w_out.setter
    def w_out(self, values: List[Any]) -> None:
        for b, w in zip(self.blocks, values):
            if isinstance(w, torch.Tensor):
                b.linear_out.weight = nn.Parameter(w)
            elif isinstance(w, np.ndarray):
                b.linear_out.weight = nn.Parameter(torch.from_numpy(w).float())

    def _get_lm_weight(self) -> torch.Tensor:
        if hasattr(self.shared_lm_head, "weight"):
            return self.shared_lm_head.weight
        return self.shared_lm_head

    def forward(
        self, hidden_states: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """Forward pass projecting hidden state through K heads.

        Args:
            hidden_states: Tensor or array of shape (batch_size, seq_len, hidden_dim) or (seq_len, hidden_dim).

        Returns:
            Logits tensor of shape (num_heads, batch_size, seq_len, vocab_size) or (num_heads, seq_len, vocab_size).
        """
        is_numpy = isinstance(hidden_states, np.ndarray)
        weight = self._get_lm_weight()

        if is_numpy:
            t_hidden = torch.from_numpy(hidden_states).to(
                dtype=weight.dtype, device=weight.device
            )
        else:
            t_hidden = hidden_states

        all_logits = []
        for b in self.blocks:
            rep = b(t_hidden)
            logits = F.linear(rep, weight)
            all_logits.append(logits)

        out = torch.stack(all_logits, dim=0)
        if is_numpy:
            return out.detach().cpu().numpy()
        return out

    def generate_candidates(
        self, hidden_states: Union[torch.Tensor, np.ndarray]
    ) -> List[int]:
        """Greedy extraction of top-1 token candidate per head.

        Args:
            hidden_states: Tensor or array of shape (1, seq_len, hidden_dim) or (seq_len, hidden_dim).

        Returns:
            List of length num_heads containing greedy token predictions.
        """
        with torch.no_grad():
            logits = self.forward(hidden_states)
            if isinstance(logits, torch.Tensor):
                if logits.dim() == 3:
                    last_step_logits = logits[:, -1, :]
                else:
                    last_step_logits = logits[:, 0, -1, :]
                return torch.argmax(last_step_logits, dim=-1).tolist()
            else:
                if logits.ndim == 3:
                    last_step_logits = logits[:, -1, :]
                else:
                    last_step_logits = logits[:, 0, -1, :]
                return [int(np.argmax(last_step_logits[k])) for k in range(self.num_heads)]

    def propose(
        self,
        hidden_states: Union[torch.Tensor, np.ndarray],
        k_draft: int = 3,
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, np.ndarray], Optional[Any]]:
        """Candidate generation method adhering to PROJECT.md interface contracts.

        Args:
            hidden_states: Tensor of shape (batch_size, seq_len, hidden_dim) or (seq_len, hidden_dim).
            k_draft: Number of speculative lookahead tokens requested.

        Returns:
            Tuple of (candidate_tokens, stacked_logits):
            - candidate_tokens: Tensor or array of shape (batch_size, k_eval).
            - stacked_logits: Logits tensor of shape (k_eval, batch_size, vocab_size).
        """
        is_numpy = isinstance(hidden_states, np.ndarray)
        weight = self._get_lm_weight()

        if is_numpy:
            if hidden_states.ndim == 1:
                hidden_states = hidden_states[np.newaxis, np.newaxis, :]
            elif hidden_states.ndim == 2:
                hidden_states = hidden_states[np.newaxis, :]
            t_hidden = torch.from_numpy(hidden_states).to(
                dtype=weight.dtype, device=weight.device
            )
        else:
            if hidden_states.dim() == 1:
                hidden_states = hidden_states.unsqueeze(0).unsqueeze(0)
            elif hidden_states.dim() == 2:
                hidden_states = hidden_states.unsqueeze(0)
            t_hidden = hidden_states

        batch_size = t_hidden.size(0)
        seq_len = t_hidden.size(1)
        k_eval = self.num_heads if k_draft is None else min(int(k_draft), self.num_heads)

        # Clean guard for non-positive lookahead, empty heads, or empty sequence
        if k_eval <= 0 or seq_len == 0:
            if is_numpy:
                return np.empty((batch_size, 0), dtype=np.int64), None
            return torch.empty((batch_size, 0), dtype=torch.long, device=t_hidden.device), None

        last_step = t_hidden[:, -1:, :]

        with torch.no_grad():
            logits_list = []
            cand_list = []
            for i in range(k_eval):
                rep = self.blocks[i](last_step)
                logits_step = F.linear(rep, weight)
                logits_2d = logits_step[:, -1, :]
                logits_list.append(logits_2d)
                cand = torch.argmax(logits_2d, dim=-1)
                cand_list.append(cand)

            cands = torch.stack(cand_list, dim=1)
            stacked_logits = torch.stack(logits_list, dim=0)

        if is_numpy:
            return cands.detach().cpu().numpy(), stacked_logits.detach().cpu().numpy()
        return cands, stacked_logits
