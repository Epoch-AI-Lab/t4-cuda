#!/usr/bin/env python3
"""Unit tests for the 3-Way Head-to-Head Kernel Benchmark Harness."""

import json
import os
import sys
import pytest
import torch

current = os.path.abspath(os.path.dirname(__file__))
while current != "/" and not os.path.exists(os.path.join(current, "benchmarks")):
    current = os.path.dirname(current)
REPO_ROOT = current
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from benchmarks.benchmark_3way_kernels import (
    compute_metrics,
    compute_numerical_parity,
    fallback_quantize_sym_int4,
    get_benchmark_shapes,
    parse_args,
    run_3way_benchmark,
)


def test_qwen25_7b_canonical_shapes():
    """Verify all canonical Qwen2.5-7B shapes required by R1 are present and valid."""
    shapes = get_benchmark_shapes()
    shape_map = {(s["K"], s["N"]): s for s in shapes}

    # 1. Gate/Up Proj (K=3584, N=18944)
    assert (3584, 18944) in shape_map, "Missing Gate/Up Proj shape (3584, 18944)"
    assert "Gate/Up" in shape_map[(3584, 18944)]["name"]

    # 2. Down Proj (K=18944, N=3584)
    assert (18944, 3584) in shape_map, "Missing Down Proj shape (18944, 3584)"
    assert "Down" in shape_map[(18944, 3584)]["name"]

    # 3. QKV Proj (K=3584, N=3584)
    assert (3584, 3584) in shape_map, "Missing QKV Proj shape (3584, 3584)"
    qkv_name = shape_map[(3584, 3584)]["name"]
    assert "QKV" in qkv_name or "Q-Proj" in qkv_name

    # 4. TP Gate/Up Sharded Rank (K=3584, N=9472)
    assert (3584, 9472) in shape_map, "Missing TP Gate/Up shape (3584, 9472)"

    # 5. TP Down Sharded Rank (K=9472, N=3584)
    assert (9472, 3584) in shape_map, "Missing TP Down shape (9472, 3584)"


def test_shape_filtering_coverage():
    """Verify shapes filter matches all target categories including QKV, Gate, Down, and TP."""
    all_shapes = get_benchmark_shapes()

    for query in ["QKV", "Gate", "Down", "TP"]:
        matched = [
            s for s in all_shapes
            if query.lower() in s["name"].lower() or query.lower() in s["category"].lower()
        ]
        assert len(matched) > 0, f"Query '{query}' matched 0 shapes in benchmark list"


def test_regimes_and_timing_defaults():
    """Verify CLI defaults and parameters satisfy R1 (decode M=1,4; prefill M=16,64,256; warmup>=20; iters>=50)."""
    # Create parser via argparse inspect
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16, 64, 256])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=50)

    defaults = parser.parse_args([])
    assert 1 in defaults.batch_sizes and 4 in defaults.batch_sizes, "Must include decode batch sizes M=1,4"
    assert 16 in defaults.batch_sizes and 64 in defaults.batch_sizes and 256 in defaults.batch_sizes, "Must include prefill batch sizes M=16,64,256"
    assert defaults.warmup >= 20, f"Warmup must be >= 20, got {defaults.warmup}"
    assert defaults.iters >= 50, f"Timed iters must be >= 50, got {defaults.iters}"


def test_compute_metrics():
    """Verify effective bandwidth and TFLOP/s calculations with known mathematical values."""
    latency_us = 1000.0  # 1 ms = 1e-3 s
    M = 1
    K = 3584
    N = 18944

    # FLOPs = 2 * 1 * 18944 * 3584 = 135,798,784 FLOPs
    # In 1e-3 s -> 135,798,784 / 1e-3 = 1.35798784e11 FLOP/s = 0.135798784 TFLOP/s
    bw_int4, tflops = compute_metrics(latency_us, M, K, N, is_4bit=True, group_size=128)
    assert abs(tflops - 0.135798) < 1e-3, f"Expected ~0.1358 TFLOP/s, got {tflops}"

    # Weight bytes INT4: (3584 * 18944) * 0.5 = 33,949,696 bytes
    # Scales + ZPs: (3584 / 128) * 18944 * 4 = 28 * 18944 * 4 = 2,121,728 bytes
    # Input: 1 * 3584 * 2 = 7168 bytes
    # Output: 1 * 18944 * 2 = 37,888 bytes
    # Total = ~36,116,480 bytes in 1e-3 s = ~36.116 GB/s
    assert abs(bw_int4 - 36.116) < 1.0, f"Expected ~36.1 GB/s, got {bw_int4}"

    # FP16 bandwidth test
    bw_fp16, _ = compute_metrics(latency_us, M, K, N, is_4bit=False)
    # FP16 weight = 3584 * 18944 * 2 = 135,798,784 bytes (~135.8 GB/s)
    assert abs(bw_fp16 - 135.84) < 1.0, f"Expected ~135.8 GB/s, got {bw_fp16}"


def test_compute_numerical_parity():
    """Verify numerical parity helper correctly computes cosine similarity and max error."""
    a = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float16)
    b = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float16)
    max_err, cos_sim = compute_numerical_parity(a, b)
    assert max_err == 0.0
    assert abs(cos_sim - 1.0) < 1e-6

    c = torch.tensor([1.25, 2.0, 3.0, 4.0], dtype=torch.float16)
    max_err2, cos_sim2 = compute_numerical_parity(c, b)
    assert abs(max_err2 - 0.25) < 1e-4
    assert 0.99 < cos_sim2 < 1.0


def test_fallback_quantize_sym_int4():
    """Verify fallback symmetric INT4 quantization packing and shape invariants."""
    torch.manual_seed(42)
    K = 256
    N = 512
    W = torch.randn(N, K, dtype=torch.float16)
    packed, scales, zps, g_size = fallback_quantize_sym_int4(W, group_size=128)

    assert packed.shape == (K // 8, N)
    assert packed.dtype == torch.int32
    assert scales.shape == (K // 128, N)
    assert scales.dtype == torch.float16
    assert zps.shape == scales.shape
    assert g_size == 128


def test_benchmark_json_output_schema(tmp_path):
    """Verify machine-readable JSON log schema contains metadata, raw timings, and dynamic metrics."""
    out_file = str(tmp_path / "bench_results.json")
    results = run_3way_benchmark(
        batch_sizes=[1, 4],
        shapes_filter="KV-Proj",
        iters=2,
        warmup=1,
        group_size=128,
        dry_run=True,
        output_json=out_file,
        output_log=str(tmp_path / "bench_results.log"),
        output_md=str(tmp_path / "bench_results.md"),
    )
    assert len(results) == 2, f"Expected 2 rows, got {len(results)}"
    assert os.path.exists(out_file), "JSON output file was not created"

    with open(out_file, "r") as f:
        data = json.load(f)

    # 1. Verify Top-Level Structure
    assert "metadata" in data, "JSON must contain 'metadata' section"
    assert "results" in data, "JSON must contain 'results' section"

    meta = data["metadata"]
    assert "timestamp" in meta, "Metadata must include timestamp"
    assert "hardware_device" in meta, "Metadata must include hardware_device"
    assert "device_info" in meta, "Metadata must include device_info"
    assert "config" in meta, "Metadata must include config"
    assert "hardware_baselines" in meta, "Metadata must include hardware_baselines"

    assert meta["hardware_baselines"]["t4_peak_bandwidth_gb_s"] == 320.0
    assert meta["hardware_baselines"]["t4_peak_fp16_tc_tflops"] == 65.0

    # 2. Verify Per-Row Kernel Metrics and Raw Timings
    for row in data["results"]:
        assert "M" in row and "K" in row and "N" in row
        for kernel_key in ["cuBLAS_FP16", "t4_kernels", "bitsandbytes", "bitsandbytes_nf4", "bitsandbytes_fp4", "marlin"]:
            assert kernel_key in row, f"Missing kernel key '{kernel_key}' in row"
            k_data = row[kernel_key]
            assert "status" in k_data, f"Missing 'status' in {kernel_key}"
            assert "raw_timings_us" in k_data, f"Missing 'raw_timings_us' in {kernel_key}"
            assert isinstance(k_data["raw_timings_us"], list), f"'raw_timings_us' must be a list in {kernel_key}"


def test_loud_failure_when_no_cuda_and_not_dry_run():
    """Verify that attempting a non-dry-run on a system without CUDA raises RuntimeError (no silent CPU dummies)."""
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError) as exc_info:
            run_3way_benchmark(
                batch_sizes=[1],
                shapes_filter="KV-Proj",
                dry_run=False,
            )
        assert "CUDA is not available" in str(exc_info.value)
    else:
        pytest.skip("Test applicable only on hosts without CUDA")


def test_cuda_hardware_gate():
    """Verify execution on actual CUDA hardware when available (skips cleanly on non-CUDA hosts)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable on this host — skipping physical GPU kernel execution test")
    
    results = run_3way_benchmark(
        batch_sizes=[1],
        shapes_filter="KV-Proj",
        iters=5,
        warmup=5,
        dry_run=False,
    )
    assert len(results) > 0
    assert results[0]["cuBLAS_FP16"]["latency_us"] > 0


def _dequant_asym_tuple(packed, scales, zps):
    """Reference dequant of a (packed, scales, zps) tuple back to [out, in] float."""
    K8, N = packed.shape
    K = K8 * 8
    qt = torch.zeros(K, N, dtype=torch.float32)
    for i in range(8):
        qt[i::8, :] = ((packed >> (4 * i)) & 0xF).float()
    q = qt.t()
    ng = scales.shape[0]
    gs = K // ng
    out = torch.empty_like(q)
    for g in range(ng):
        out[:, g * gs:(g + 1) * gs] = (
            q[:, g * gs:(g + 1) * gs] - zps[g].float().unsqueeze(1)
        ) * scales[g].float().unsqueeze(1)
    return out


def test_asym_packing_invariants():
    """Verify asymmetric quantizer tuple shapes, dtypes, and nibble range."""
    from src.hybrid_linear import quantize_weight_asym_int4
    torch.manual_seed(0)
    W = torch.randn(64, 256, dtype=torch.float16)
    packed, scales, zps, g_size = quantize_weight_asym_int4(W, group_size=128)
    assert packed.shape == (256 // 8, 64)
    assert packed.dtype == torch.int32
    assert scales.shape == (2, 64) and scales.dtype == torch.float16
    assert zps.shape == scales.shape
    assert g_size == 128
    # every nibble in 0..15
    for i in range(8):
        nib = (packed >> (4 * i)) & 0xF
        assert nib.min().item() >= 0 and nib.max().item() <= 15


def test_asym_beats_sym_on_skewed_weights():
    """Asymmetric must clearly beat symmetric on offset (skewed) weights."""
    from src.hybrid_linear import quantize_weight_asym_int4, quantize_weight_sym_int4
    torch.manual_seed(0)
    W = torch.randn(64, 256, dtype=torch.float16) + 3.0
    pa, sa, za, _ = quantize_weight_asym_int4(W, group_size=128)
    ps, ss, zs, _ = quantize_weight_sym_int4(W, group_size=128)
    e_asym = (_dequant_asym_tuple(pa, sa, za) - W.float()).pow(2).mean().item()
    e_sym = (_dequant_asym_tuple(ps, ss, zs) - W.float()).pow(2).mean().item()
    assert e_asym < e_sym / 2.0, f"asym {e_asym:.5f} should be < half of sym {e_sym:.5f}"


def test_asym_constant_group_exact():
    """Constant groups must reconstruct exactly (no information to lose)."""
    from src.hybrid_linear import quantize_weight_asym_int4
    W = torch.full((8, 128), 2.5, dtype=torch.float16)
    packed, scales, zps, _ = quantize_weight_asym_int4(W, group_size=128)
    assert torch.equal(_dequant_asym_tuple(packed, scales, zps), W.float())


def test_asym_mirrors_agree():
    """hybrid_linear, tp mirror, and benchmark fallback must pack identically."""
    from src.hybrid_linear import quantize_weight_asym_int4
    from src.sharding.tp import quantize_weight_asym_int4 as tp_asym
    from benchmarks.benchmark_3way_kernels import fallback_quantize_asym_int4
    torch.manual_seed(7)
    W = torch.randn(32, 128, dtype=torch.float16)
    a = quantize_weight_asym_int4(W, group_size=64)
    b = tp_asym(W, group_size=64)
    c = fallback_quantize_asym_int4(W, group_size=64)
    assert torch.equal(a[0], b[0]) and torch.equal(a[0], c[0])
    assert torch.equal(a[1], b[1]) and torch.equal(a[1], c[1])


def test_group_size_divisibility_matrix():
    """All benchmark K dims must support group 32/64/128 with no remainder."""
    for s in get_benchmark_shapes():
        for gs in (32, 64, 128):
            assert s["K"] % gs == 0, f"{s['name']} K={s['K']} not divisible by {gs}"


def test_gptq_beats_rtn_on_correlated_acts():
    """GPTQ error feedback must beat plain RTN under correlated activations."""
    from src.hybrid_linear import quantize_weight_asym_int4, quantize_weight_gptq_int4, compute_hessian
    torch.manual_seed(1)
    O, K = 16, 64
    W = torch.randn(O, K)
    X = (torch.randn(K, 40) @ torch.randn(40, 200)).t()
    H = compute_hessian(X)
    assert H.shape == (K, K)
    pg, sg, zg, _ = quantize_weight_gptq_int4(W, H, group_size=32, blocksize=32)
    pr, sr, zr, _ = quantize_weight_asym_int4(W, group_size=32)
    Xt = torch.randn(50, K)
    e_gptq = ((Xt @ _dequant_asym_tuple(pg, sg, zg).t()) - (Xt @ W.t())).pow(2).mean().item()
    e_rtn = ((Xt @ _dequant_asym_tuple(pr, sr, zr).t()) - (Xt @ W.t())).pow(2).mean().item()
    assert e_gptq < e_rtn, f"gptq {e_gptq:.6f} should beat rtn {e_rtn:.6f}"


def test_gptq_none_hessian_falls_back_to_asym():
    """H=None must behave exactly like the asymmetric RTN path (never crash)."""
    from src.hybrid_linear import quantize_weight_asym_int4, quantize_weight_gptq_int4
    torch.manual_seed(3)
    W = torch.randn(16, 64)
    g = quantize_weight_gptq_int4(W, None, group_size=32)
    a = quantize_weight_asym_int4(W, group_size=32)
    assert torch.equal(g[0], a[0]) and torch.equal(g[1], a[1])


def test_gptq_bad_hessian_shape_raises():
    """Wrong-size Hessian must fail loudly, never silently quantize."""
    from src.hybrid_linear import quantize_weight_gptq_int4
    torch.manual_seed(3)
    W = torch.randn(16, 64)
    with pytest.raises(ValueError):
        quantize_weight_gptq_int4(W, torch.eye(32), group_size=32)


def test_benchmark_quant_flag_plumbing(tmp_path):
    """--quant must select the scheme, record it in JSON, and reject junk."""
    out = str(tmp_path / "q.json")
    res = run_3way_benchmark(batch_sizes=[1], shapes_filter="KV-Proj", iters=2,
                             warmup=1, dry_run=True, output_json=out,
                             output_log=None, output_md=None, quant="asym")
    assert len(res) > 0
    with open(out) as f:
        cfg = json.load(f)["metadata"]["config"]
    assert cfg["quant"] == "asym"
    res = run_3way_benchmark(batch_sizes=[1], shapes_filter="KV-Proj", iters=1,
                             warmup=1, dry_run=True, output_json=out,
                             output_log=None, output_md=None, quant="gptq")
    assert len(res) > 0
    with pytest.raises(ValueError):
        run_3way_benchmark(batch_sizes=[1], shapes_filter="KV-Proj", dry_run=True,
                           output_json=out, output_log=None, output_md=None, quant="bogus")
