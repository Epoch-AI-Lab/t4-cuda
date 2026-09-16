#!/usr/bin/env python3
"""Adversarial stress tests for 3-way kernel benchmark harness and TP vs PP benchmark."""

import json
import os
import sys
import tempfile
import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from benchmarks.benchmark_3way_kernels import (
    benchmark_cuda_op,
    compute_metrics,
    compute_numerical_parity,
    fallback_quantize_sym_int4,
    get_benchmark_shapes,
    run_3way_benchmark,
)
from benchmarks.benchmark_tp_vs_pp import benchmark_pcie_transfers


class TestEdgeParameters:
    """Stress tests on edge parameters for run_3way_benchmark."""

    def test_invalid_shapes_filter_returns_empty_list(self):
        """Shapes filter with nonexistent name should cleanly return empty list without crashing."""
        res = run_3way_benchmark(
            batch_sizes=[1],
            shapes_filter="NonExistentLayerName_XYZ123",
            dry_run=True,
        )
        assert res == []

    def test_case_insensitive_and_partial_shapes_filter(self):
        """Shapes filter should handle case variation and partial keywords."""
        res_lower = run_3way_benchmark(batch_sizes=[1], shapes_filter="qkv", dry_run=True)
        assert len(res_lower) == 1
        assert "QKV" in res_lower[0]["name"] or "Q-Proj" in res_lower[0]["name"]

        res_upper = run_3way_benchmark(batch_sizes=[1], shapes_filter="GATE", dry_run=True)
        assert len(res_upper) >= 2  # Qwen full + Llama + TP

    def test_special_regex_characters_in_filter(self):
        """Shapes filter uses substring 'in', so special regex characters should not crash."""
        res = run_3way_benchmark(batch_sizes=[1], shapes_filter="[", dry_run=True)
        # Should not raise re.error
        assert isinstance(res, list)

    def test_empty_shapes_filter_matches_all(self):
        """Empty string shapes_filter should match all 8 benchmark shapes."""
        res = run_3way_benchmark(batch_sizes=[1], shapes_filter="", dry_run=True)
        assert len(res) == 8

    def test_odd_and_prime_batch_sizes(self):
        """Odd batch sizes (1, 3, 5, 7, 13, 31, 63, 127) should succeed in dry-run mode."""
        odd_batches = [1, 3, 5, 7, 13, 31, 63, 127]
        res = run_3way_benchmark(
            batch_sizes=odd_batches,
            shapes_filter="KV-Proj",
            iters=2,
            warmup=1,
            dry_run=True,
        )
        assert len(res) == len(odd_batches)
        for idx, row in enumerate(res):
            assert row["M"] == odd_batches[idx]
            assert row["cuBLAS_FP16"]["latency_us"] > 0
            assert row["cuBLAS_FP16"]["bandwidth_gb_s"] > 0
            assert row["cuBLAS_FP16"]["tflops"] > 0

    def test_empty_batch_sizes_list(self):
        """Empty list of batch sizes should return empty results without crashing."""
        res = run_3way_benchmark(
            batch_sizes=[],
            shapes_filter="KV-Proj",
            dry_run=True,
        )
        assert res == []

    def test_batch_size_zero(self):
        """Batch size 0: PyTorch allows 0-dim tensors; FLOPs should be 0, latency measured."""
        res = run_3way_benchmark(
            batch_sizes=[0],
            shapes_filter="KV-Proj",
            iters=2,
            warmup=1,
            dry_run=True,
        )
        assert len(res) == 1
        row = res[0]
        assert row["M"] == 0
        assert row["cuBLAS_FP16"]["tflops"] == 0.0

    def test_zero_warmup_iterations(self):
        """warmup=0 should execute cleanly with 0 warmup passes."""
        res = run_3way_benchmark(
            batch_sizes=[1],
            shapes_filter="KV-Proj",
            iters=2,
            warmup=0,
            dry_run=True,
        )
        assert len(res) == 1
        assert len(res[0]["cuBLAS_FP16"]["raw_timings_us"]) == 2

    def test_negative_warmup_iterations(self):
        """warmup < 0 is handled safely by range(warmup) which produces 0 iterations."""
        res = run_3way_benchmark(
            batch_sizes=[1],
            shapes_filter="KV-Proj",
            iters=2,
            warmup=-5,
            dry_run=True,
        )
        assert len(res) == 1
        assert len(res[0]["cuBLAS_FP16"]["raw_timings_us"]) == 2

    def test_zero_iters_raises_error(self):
        """iters=0 cannot compute percentiles and must raise IndexError in benchmark_cuda_op."""
        with pytest.raises(IndexError):
            run_3way_benchmark(
                batch_sizes=[1],
                shapes_filter="KV-Proj",
                iters=0,
                warmup=1,
                dry_run=True,
            )

    def test_negative_iters_raises_error(self):
        """iters < 0 cannot compute percentiles and must raise IndexError in benchmark_cuda_op."""
        with pytest.raises(IndexError):
            run_3way_benchmark(
                batch_sizes=[1],
                shapes_filter="KV-Proj",
                iters=-5,
                warmup=1,
                dry_run=True,
            )

    def test_group_size_zero_per_channel(self):
        """group_size=0 (per-channel quantization) should run fallback quantize successfully."""
        torch.manual_seed(42)
        W = torch.randn(512, 256, dtype=torch.float16)
        packed, scales, zps, g_size = fallback_quantize_sym_int4(W, group_size=0)
        assert packed.shape == (256 // 8, 512)
        assert scales.shape == (1, 512)
        assert zps.shape == (1, 512)
        assert g_size == 0

    def test_group_size_powers_of_two(self):
        """group_size=32, 64, 128 should produce valid packed tensors."""
        torch.manual_seed(42)
        W = torch.randn(256, 256, dtype=torch.float16)
        for g in [32, 64, 128]:
            packed, scales, zps, g_size = fallback_quantize_sym_int4(W, group_size=g)
            assert packed.shape == (256 // 8, 256)
            assert scales.shape == (256 // g, 256)
            assert g_size == g


class TestJsonSchemaAndDataIntegrity:
    """Stress testing JSON schema conformance and data integrity."""

    def test_json_output_with_deeply_nested_path(self, tmp_path):
        """output_json with nonexistent nested directory structure should be created automatically."""
        out_file = str(tmp_path / "deep" / "nested" / "dir" / "results.json")
        res = run_3way_benchmark(
            batch_sizes=[1],
            shapes_filter="KV-Proj",
            iters=3,
            warmup=1,
            dry_run=True,
            output_json=out_file,
        )
        assert os.path.exists(out_file)
        with open(out_file, "r") as f:
            data = json.load(f)

        assert "metadata" in data
        assert "results" in data
        meta = data["metadata"]
        assert meta["dry_run"] is True
        assert meta["hardware_device"] == "cpu"
        assert meta["config"]["iters"] == 3
        assert meta["config"]["warmup"] == 1

        # Check raw timings integrity
        cublas = data["results"][0]["cuBLAS_FP16"]
        raw = cublas["raw_timings_us"]
        assert len(raw) == 3
        assert all(isinstance(t, float) and t > 0 for t in raw)
        # Raw timings must be sorted ascending
        assert raw == sorted(raw)
        assert cublas["min_us"] == raw[0]
        assert cublas["max_us"] == raw[-1]
        assert cublas["median_us"] == raw[1]

    def test_skipped_kernels_have_empty_raw_timings_and_null_metrics(self, tmp_path):
        """When kernels are skipped, raw_timings_us must be empty list and metrics must be None."""
        out_file = str(tmp_path / "skipped_check.json")
        run_3way_benchmark(
            batch_sizes=[1],
            shapes_filter="KV-Proj",
            iters=2,
            warmup=1,
            dry_run=True,
            output_json=out_file,
        )
        with open(out_file, "r") as f:
            data = json.load(f)

        row = data["results"][0]
        for k_name in ["t4_kernels", "bitsandbytes", "marlin"]:
            k = row[k_name]
            assert "SKIPPED" in k["status"]
            assert k["raw_timings_us"] == []
            assert k["latency_us"] is None
            assert k["median_us"] is None
            assert k["p95_us"] is None
            assert k["min_us"] is None
            assert k["max_us"] is None
            assert k["bandwidth_gb_s"] is None
            assert k["pct_t4_bandwidth"] is None
            assert k["tflops"] is None
            assert k["pct_t4_tflops"] is None
            assert k["cos_sim"] is None
            assert k["max_err"] is None


class TestLoudCpuFailureEnforcement:
    """Stress testing that running on CPU without dry_run raises loudly and fails with exit code 1."""

    def test_run_3way_benchmark_raises_runtime_error(self):
        """Calling run_3way_benchmark with dry_run=False on CPU host must raise RuntimeError."""
        if torch.cuda.is_available():
            pytest.skip("Host has CUDA")
        with pytest.raises(RuntimeError) as exc:
            run_3way_benchmark(
                batch_sizes=[1],
                shapes_filter="KV-Proj",
                dry_run=False,
            )
        assert "CUDA is not available on this host" in str(exc.value)

    def test_cli_without_dry_run_exits_nonzero(self):
        """CLI execution without --dry-run on CPU must exit with code 1."""
        if torch.cuda.is_available():
            pytest.skip("Host has CUDA")
        import subprocess
        cmd = [
            sys.executable,
            os.path.join(REPO_ROOT, "benchmarks", "benchmark_3way_kernels.py"),
            "--batch-sizes", "1",
            "--shapes", "KV-Proj",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode != 0
        assert "RuntimeError: CUDA is not available on this host" in proc.stderr


class TestTpVsPpDryRunAuthenticity:
    """Stress testing benchmarks/benchmark_tp_vs_pp.py dry-run and CPU behavior."""

    def test_benchmark_pcie_transfers_returns_none_on_cpu(self):
        """benchmark_pcie_transfers must return all None on CPU host or dry-run."""
        stats = benchmark_pcie_transfers([], 3584, is_dry_run=True)
        assert stats["all_reduce_latency_us"] is None
        assert stats["boundary_p2p_latency_us"] is None
        assert stats["pcie_bandwidth_gb_s"] is None

    def test_cli_tp_vs_pp_emits_no_synthetic_numbers(self, tmp_path):
        """Running benchmark_tp_vs_pp.py on CPU host must emit SKIPPED status and null metrics."""
        import subprocess
        out_file = str(tmp_path / "tp_pp_output.json")
        cmd = [
            sys.executable,
            os.path.join(REPO_ROOT, "benchmarks", "benchmark_tp_vs_pp.py"),
            "--output_file", out_file,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode == 0
        assert "[SKIP] Dual CUDA GPUs absent. No synthetic numbers emitted." in proc.stdout

        with open(out_file, "r") as f:
            data = json.load(f)

        assert data["has_dual_cuda"] is False
        assert "SKIPPED" in data["status"]

        # Ensure PCIe stats are null
        assert data["pcie_stats"]["all_reduce_latency_us"] is None
        assert data["pcie_stats"]["boundary_p2p_latency_us"] is None
        assert data["pcie_stats"]["pcie_bandwidth_gb_s"] is None

        # Ensure TP metrics are null
        tp = data["tensor_parallelism"]
        assert tp["comm_latency_ms"] is None
        assert tp["compute_latency_ms"] is None
        assert tp["total_step_latency_ms"] is None
        assert tp["throughput_tok_s"] is None
        assert tp["peak_vram_per_gpu_gb"] is None

        # Ensure PP metrics are null
        pp = data["pipeline_parallelism"]
        assert pp["comm_latency_ms"] is None
        assert pp["compute_latency_ms"] is None
        assert pp["total_step_latency_ms"] is None
        assert pp["throughput_tok_s"] is None
        assert pp["peak_vram_per_gpu_gb"] is None


class TestPercentileAndMetricsMath:
    """Stress testing mathematical formulas in compute_metrics and benchmark_cuda_op."""

    def test_zero_latency_does_not_divide_by_zero(self):
        """latency_us=0 should use clamp floor of 1e-12 without crashing."""
        bw, tflops = compute_metrics(0.0, M=1, K=3584, N=18944)
        assert bw > 0
        assert tflops > 0

    def test_numerical_parity_identical(self):
        """Identical tensors must produce cos_sim=1.0 and max_err=0.0."""
        t = torch.tensor([1.0, -2.0, 3.5, 0.0], dtype=torch.float16)
        max_err, cos_sim = compute_numerical_parity(t, t)
        assert max_err == 0.0
        assert abs(cos_sim - 1.0) < 1e-6

    def test_numerical_parity_orthogonal(self):
        """Orthogonal tensors must produce cos_sim=0.0."""
        a = torch.tensor([1.0, 0.0], dtype=torch.float16)
        b = torch.tensor([0.0, 1.0], dtype=torch.float16)
        max_err, cos_sim = compute_numerical_parity(a, b)
        assert max_err == 1.0
        assert abs(cos_sim) < 1e-6
