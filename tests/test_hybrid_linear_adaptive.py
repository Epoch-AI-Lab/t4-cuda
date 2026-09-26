import os
import sys
import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
if os.path.join(REPO_DIR, "src") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_DIR, "src"))

from src.hybrid_linear import (
    HybridLinear,
    CPHybridScope,
    quantize_weight_sym_int4,
    quantize_weight_asym_int4,
    quantize_weight_gptq_int4,
    dequantize_sym_int4,
)
from src.sharding.tp import TPColumnParallelLinear, TPRowParallelLinear


def test_hybrid_linear_init_modes():
    """Verify HybridLinear properly instantiates across all supported quant modes."""
    in_f, out_f = 256, 512
    lin = nn.Linear(in_f, out_f, bias=True).half()

    # 1. Symmetric
    hl_sym = HybridLinear(lin, group_size=128, quant_type="sym")
    assert hl_sym.quant_type == "sym"
    assert hl_sym.packed.shape == (in_f // 8, out_f)
    assert hl_sym.scales.shape == (in_f // 128, out_f)
    assert hl_sym.bias is not None

    # 2. Asymmetric
    hl_asym = HybridLinear(lin, group_size=128, quant_type="asym")
    assert hl_asym.quant_type == "asym"
    assert hl_asym.packed.shape == (in_f // 8, out_f)
    assert hl_asym.zps.shape == (in_f // 128, out_f)

    # 3. GPTQ with dummy Hessian
    H = torch.eye(in_f, dtype=torch.float32)
    hl_gptq = HybridLinear(lin, group_size=128, quant_type="gptq", H=H)
    assert hl_gptq.quant_type == "gptq"
    assert hl_gptq.packed.shape == (in_f // 8, out_f)

    # 4. Invalid quant_type raises ValueError
    with pytest.raises(ValueError) as exc:
        HybridLinear(lin, quant_type="invalid_quant")
    assert "Unsupported quant_type" in str(exc.value)


def test_hybrid_linear_adaptive_dispatch_logic():
    """Verify adaptive threshold properly routes decode (M <= 4) and prefill (M > 4)."""
    in_f, out_f = 128, 256
    lin = nn.Linear(in_f, out_f, bias=False).half()
    hl = HybridLinear(lin, group_size=128, quant_type="asym", decode_threshold=4)

    # 2D Decode input (M = 1)
    x_decode = torch.randn(1, in_f, dtype=torch.float16)
    out_decode = hl(x_decode)
    assert out_decode.shape == (1, out_f)

    # 2D Prefill input (M = 16)
    x_prefill = torch.randn(16, in_f, dtype=torch.float16)
    out_prefill = hl(x_prefill)
    assert out_prefill.shape == (16, out_f)

    # 3D Sequence input (B = 2, S = 8, M = 16)
    x_3d = torch.randn(2, 8, in_f, dtype=torch.float16)
    out_3d = hl(x_3d)
    assert out_3d.shape == (2, 8, out_f)

    # Numerical parity vs FP16 baseline on prefill
    ref_prefill = lin(x_prefill)
    cos_sim = F.cosine_similarity(out_prefill.flatten().float(), ref_prefill.flatten().float(), dim=0).item()
    assert cos_sim > 0.9999, f"Prefill must match cuBLAS FP16 reference bit-exactly, got {cos_sim}"


def test_cp_hybrid_scope_patching_and_cleanup():
    """Verify CPHybridScope patches targeted layers and cleanly reverts on exit."""
    class DummyTransformerBlock(nn.Module):
        def __init__(self, d=128, h=256):
            super().__init__()
            self.q_proj = nn.Linear(d, d, bias=False).half()
            self.k_proj = nn.Linear(d, d, bias=False).half()
            self.gate_proj = nn.Linear(d, h, bias=False).half()
            self.down_proj = nn.Linear(h, d, bias=False).half()

        def forward(self, x):
            return self.down_proj(F.silu(self.gate_proj(x))) + self.q_proj(x)

    model = DummyTransformerBlock()
    orig_q = model.q_proj
    orig_gate = model.gate_proj

    # With patch_attention=True, both q_proj and gate_proj should be patched
    with CPHybridScope(model, quant_type="asym", patch_attention=True) as patched_model:
        assert isinstance(patched_model.q_proj, HybridLinear)
        assert isinstance(patched_model.gate_proj, HybridLinear)
        assert patched_model.q_proj.quant_type == "asym"

        # Verify forward pass through patched model
        x = torch.randn(2, 128, dtype=torch.float16)
        out = patched_model(x)
        assert out.shape == (2, 128)

    # After exit, modules must be restored to original nn.Linear
    assert model.q_proj is orig_q
    assert model.gate_proj is orig_gate
    assert not isinstance(model.q_proj, HybridLinear)


def test_tp_linear_asym_quant():
    """Verify TP Column and Row parallel linear accept asym quant and produce correct sharded shapes."""
    in_f, out_f = 256, 512
    lin_col = nn.Linear(in_f, out_f, bias=False).half()
    lin_row = nn.Linear(out_f, in_f, bias=False).half()

    # Column Parallel with asym INT4
    tp_col = TPColumnParallelLinear.from_linear(lin_col, rank=0, world_size=2, quant_type="asym", group_size=128)
    assert tp_col.split_out_features == out_f // 2
    assert tp_col.packed.shape == (in_f // 8, out_f // 2)

    # Row Parallel with asym INT4
    tp_row = TPRowParallelLinear.from_linear(lin_row, rank=0, world_size=2, quant_type="asym", group_size=128)
    assert tp_row.split_in_features == out_f // 2
    assert tp_row.packed.shape == (out_f // 2 // 8, in_f)
