#!/usr/bin/env bash
set -eo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${REPO_DIR}/results/traces"
mkdir -p "${RESULTS_DIR}"

echo "================================================================="
echo "   Tesla T4 Physical Nsight Hardware Trace Capture Suite"
echo "================================================================="

# Pre-flight check: CUDA availability and t4_kernels extension
CUDA_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
if [ "${CUDA_AVAILABLE}" != "True" ]; then
    echo "[SKIP] No CUDA GPU device detected. Physical Nsight profiling requires an NVIDIA GPU (Tesla T4)."
    exit 0
fi

KERNELS_AVAILABLE=$(python3 -c "import t4_kernels; print(True)" 2>/dev/null || echo "False")
if [ "${KERNELS_AVAILABLE}" != "True" ]; then
    echo "[SKIP] t4_kernels extension is not compiled or installed. Build via: pip install -e src/"
    exit 0
fi

# Locate ncu and nsys
NCU_BIN="$(command -v ncu || echo /usr/local/cuda/bin/ncu)"
NSYS_BIN="$(command -v nsys || echo /usr/local/cuda/bin/nsys)"

if [ ! -x "${NCU_BIN}" ]; then
    echo "Warning: ncu not found in standard PATH, searching /usr/local/cuda*/bin..."
    NCU_BIN="$(find /usr/local/cuda* -name ncu -type f 2>/dev/null | head -n 1 || true)"
fi

if [ ! -x "${NSYS_BIN}" ]; then
    echo "Warning: nsys not found in standard PATH, searching /usr/local/cuda*/bin..."
    NSYS_BIN="$(find /usr/local/cuda* -name nsys -type f 2>/dev/null | head -n 1 || true)"
fi

echo "NCU binary:  ${NCU_BIN:-Not Found}"
echo "NSYS binary: ${NSYS_BIN:-Not Found}"
echo "Output dir:  ${RESULTS_DIR}"
echo "-----------------------------------------------------------------"

# Coalescing & Throughput metrics for Turing T4 (sm_75)
COALESCING_METRICS="sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,\
dram__bytes_read.sum,\
dram__bytes_write.sum,\
smsp__warp_issue_stalled_barrier_per_warp_active.pct,\
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct"

# 1. Profile Fused W4A16 GEMV Kernel on Canonical Qwen2.5-7B Gate/Up Proj (M=1, K=3584, N=18944)
if [ -x "${NCU_BIN}" ]; then
    echo "[1/3] Profiling Fused W4A16 GEMV (Qwen2.5-7B Gate/Up M=1, K=3584, N=18944) via Nsight Compute..."
    "${NCU_BIN}" \
        --target-processes all \
        --metrics "${COALESCING_METRICS}" \
        --kernel-name-base demangled \
        -f -o "${RESULTS_DIR}/w4a16_gemv_qwen7b_gate" \
        python3 -c "
import torch, t4_kernels
from src.hybrid_linear import quantize_weight_sym_int4
M, K, N = 1, 3584, 18944
x = torch.randn((M, K), dtype=torch.float16, device='cuda')
W = torch.randn((N, K), dtype=torch.float16, device='cuda')
W_packed, scales, zps, g_size = quantize_weight_sym_int4(W, group_size=128)
for _ in range(20):
    _ = t4_kernels.fused_w4a16_gemm_u4(x, W_packed, scales, zps, g_size)
torch.cuda.synchronize()
for _ in range(10):
    _ = t4_kernels.fused_w4a16_gemm_u4(x, W_packed, scales, zps, g_size)
torch.cuda.synchronize()
"

    # 2. Profile Fused W4A16 WMMA Prefill Kernel on Qwen2.5-7B Gate/Up (M=64, K=3584, N=18944)
    echo "[2/3] Profiling Fused W4A16 WMMA (Qwen2.5-7B Prefill M=64, K=3584, N=18944) via Nsight Compute..."
    "${NCU_BIN}" \
        --target-processes all \
        --metrics "${COALESCING_METRICS}" \
        --kernel-name-base demangled \
        -f -o "${RESULTS_DIR}/w4a16_wmma_qwen7b_prefill" \
        python3 -c "
import torch, t4_kernels
from src.hybrid_linear import quantize_weight_sym_int4
M, K, N = 64, 3584, 18944
x = torch.randn((M, K), dtype=torch.float16, device='cuda')
W = torch.randn((N, K), dtype=torch.float16, device='cuda')
W_packed, scales, zps, g_size = quantize_weight_sym_int4(W, group_size=128)
for _ in range(20):
    _ = t4_kernels.fused_w4a16_gemm_u4(x, W_packed, scales, zps, g_size)
torch.cuda.synchronize()
for _ in range(10):
    _ = t4_kernels.fused_w4a16_gemm_u4(x, W_packed, scales, zps, g_size)
torch.cuda.synchronize()
"
else
    echo "Skipping NCU profiles (ncu binary unavailable)."
fi

# 3. Profile 3-Way Benchmark Execution Timeline via NSYS
if [ -x "${NSYS_BIN}" ]; then
    echo "[3/3] Profiling Authentic 3-Way Benchmark Timeline via Nsight Systems..."
    "${NSYS_BIN}" profile \
        -t cuda,nvtx,osrt \
        -f true \
        -o "${RESULTS_DIR}/benchmark_3way_timeline_t4" \
        python3 "${REPO_DIR}/benchmarks/benchmark_3way_kernels.py" \
            --batch-sizes 1 4 16 64 \
            --shapes "Gate" \
            --iters 20 \
            --warmup 10 \
            --output-json "${RESULTS_DIR}/nsys_benchmark_run.json" || echo "NSYS profile completed with return code $?"
else
    echo "Skipping NSYS profile (nsys binary unavailable)."
fi

echo "================================================================="
echo "   Nsight Tracing Complete. Output artifacts in: ${RESULTS_DIR}"
echo "================================================================="
ls -lh "${RESULTS_DIR}"
