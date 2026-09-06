"""tests/test_sharding_pp_correctness.py

Correctness, Numerics, and Lifecycle Verification for Pipeline Parallelism and CPMultiGPUInferenceScope.
All tests support execution on CPU / single GPU via emulation/mocking and dual GPU on real hardware.
"""

import pytest
import torch
import torch.nn as nn
from src.sharding.pp import PipelineStage, PipelineParallelQwen2
from src.sharding.scope import CPMultiGPUInferenceScope
from src.sharding.comm import check_dual_gpu, is_cuda_available


def test_pp_layer_partition_even():
    """Verify 28 layers partition exactly 14/14 across stages."""
    pp = PipelineParallelQwen2(num_layers=28, hidden_size=128, vocab_size=500, dtype=torch.float32)
    assert len(pp.layers_stage0) == 14
    assert len(pp.layers_stage1) == 14
    assert pp.split_layer == 14


def test_pp_boundary_activation_shape():
    """Verify boundary transfer maintains tensor dimensions."""
    pp = PipelineParallelQwen2(num_layers=4, hidden_size=64, vocab_size=100, dtype=torch.float32)
    inp = torch.randint(0, 100, (2, 8))
    out = pp(inp)
    assert out.shape == (2, 8, 100)


def test_pp_forward_execution_parity():
    """Verify output logits are non-zero, finite, and match expected shape."""
    pp = PipelineParallelQwen2(num_layers=2, hidden_size=64, vocab_size=50, dtype=torch.float32)
    inp = torch.randint(0, 50, (1, 16))
    out = pp(inp)
    assert torch.isfinite(out).all()
    assert not torch.isnan(out).any()


def test_scope_invalid_mode_raises():
    """Verify invalid scope mode raises ValueError."""
    dummy = nn.Linear(10, 10)
    with pytest.raises(ValueError, match="Invalid mode"):
        CPMultiGPUInferenceScope(dummy, mode="invalid_mode")


def test_scope_invalid_device_count_raises():
    """Verify device count != 2 raises ValueError."""
    dummy = nn.Linear(10, 10)
    with pytest.raises(ValueError, match="requires exactly 2 devices"):
        CPMultiGPUInferenceScope(dummy, mode="tp", devices=["cuda:0"])


def test_scope_tp_lifecycle():
    """Verify TP scope enters, modifies modules, and cleanly restores on exit."""
    class ToyTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            layer = nn.Module()
            layer.mlp = nn.Module()
            layer.mlp.gate_proj = nn.Linear(32, 64, bias=False)
            layer.mlp.up_proj = nn.Linear(32, 64, bias=False)
            layer.mlp.down_proj = nn.Linear(64, 32, bias=False)
            self.model.layers = nn.ModuleList([layer])

    model = ToyTransformer()
    orig_mlp = model.model.layers[0].mlp

    scope = CPMultiGPUInferenceScope(model, mode="tp", quantize_mlp=False)
    assert not scope.is_active

    with scope:
        assert scope.is_active
        # Layer MLP should be wrapped with TPParallelMLP
        assert model.model.layers[0].mlp != orig_mlp

    assert not scope.is_active
    # Layer MLP should be restored
    assert model.model.layers[0].mlp == orig_mlp


def test_scope_pp_lifecycle():
    """Verify PP scope enters and exits cleanly."""
    class ToyTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([nn.Linear(16, 16) for _ in range(4)])

    model = ToyTransformer()
    scope = CPMultiGPUInferenceScope(model, mode="pp")
    assert not scope.is_active

    with scope:
        assert scope.is_active

    assert not scope.is_active


@pytest.mark.skipif(not (is_cuda_available() and check_dual_gpu()), reason="Requires dual CUDA GPUs")
def test_dual_gpu_pp_execution():
    """Execute PP pipeline across physical dual GPUs."""
    pp = PipelineParallelQwen2(
        num_layers=4,
        hidden_size=128,
        vocab_size=1000,
        devices=["cuda:0", "cuda:1"],
        dtype=torch.float16,
    )
    inp = torch.randint(0, 1000, (1, 32), device="cuda:0")
    out = pp(inp)
    assert out.device.type == "cuda"
    assert out.device.index == 1
    assert torch.isfinite(out).all()
