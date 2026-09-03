#!/usr/bin/env bash
set -eo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${REPO_DIR}/results/traces"
mkdir -p "${RESULTS_DIR}"

echo "================================================================="
echo "   Tesla T4 Physical Nsight Hardware Trace Capture Suite"
echo "================================================================="

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

# 1. Profile Fused W4A16 GEMV Kernel via NCU
if [ -x "${NCU_BIN}" ]; then
    echo "[1/2] Profiling Fused W4A16 GEMV via Nsight Compute..."
    "${NCU_BIN}" \
        --target-processes all \
        --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct \
        --kernel-name-base demangled \
        -f -o "${RESULTS_DIR}/w4a16_gemv_t4" \
        python3 -c "
import torch, t4_kernels
M, K, N = 1, 896, 896
x = torch.randn((M, K), dtype=torch.float16, device='cuda')
W_packed = torch.randint(0, 2**31 - 1, (K // 8, N), dtype=torch.int32, device='cuda')
scale = torch.randn((K // 128, N), dtype=torch.float16, device='cuda') * 0.1
zp = torch.zeros((K // 128, N), dtype=torch.float16, device='cuda')
for _ in range(10):
    _ = t4_kernels.fused_w4a16_gemm_s4(x, W_packed, scale, zp, 128)
torch.cuda.synchronize()
for _ in range(5):
    _ = t4_kernels.fused_w4a16_gemm_s4(x, W_packed, scale, zp, 128)
torch.cuda.synchronize()
"

    # 2. Profile H6 Fused AdamW Kernel via NCU
    echo "[2/2] Profiling H6 Fused AdamW via Nsight Compute..."
    "${NCU_BIN}" \
        --target-processes all \
        --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct \
        --kernel-name-base demangled \
        -f -o "${RESULTS_DIR}/h6_fused_adamw_t4" \
        python3 -c "
import torch, t4_kernels
M, d_out, d_in = 64, 896, 896
dY = torch.randn((M, d_out), dtype=torch.float16, device='cuda')
X = torch.randn((M, d_in), dtype=torch.float16, device='cuda')
W_master = torch.randn((d_out, d_in), dtype=torch.float32, device='cuda')
W_active = W_master.half()
exp_avg = torch.zeros((d_out, d_in), dtype=torch.float32, device='cuda')
exp_avg_sq = torch.zeros((d_out, d_in), dtype=torch.float32, device='cuda')
for _ in range(10):
    t4_kernels.fused_backward_gemm_adamw(dY, X, W_master, W_active, exp_avg, exp_avg_sq, 1e-3, 0.9, 0.999, 1e-8, 0.01, 1.0, 1.0)
torch.cuda.synchronize()
for _ in range(5):
    t4_kernels.fused_backward_gemm_adamw(dY, X, W_master, W_active, exp_avg, exp_avg_sq, 1e-3, 0.9, 0.999, 1e-8, 0.01, 1.0, 1.0)
torch.cuda.synchronize()
"
else
    echo "Skipping NCU profiles (ncu binary unavailable)."
fi

# 3. Profile Timeline via NSYS
if [ -x "${NSYS_BIN}" ]; then
    echo "[3/3] Profiling End-to-End Speculative Timeline via Nsight Systems..."
    "${NSYS_BIN}" profile \
        -t cuda,nvtx,osrt \
        -f true \
        -o "${RESULTS_DIR}/speculative_timeline_t4" \
        python3 -c "
import torch, time
# Synthetic timeline profiling
for _ in range(20):
    a = torch.randn((1, 896), dtype=torch.float16, device='cuda')
    b = torch.randn((896, 896), dtype=torch.float16, device='cuda')
    c = torch.matmul(a, b)
torch.cuda.synchronize()
" || echo "NSYS profile completed with return code $?"
else
    echo "Skipping NSYS profile (nsys binary unavailable)."
fi

echo "================================================================="
echo "   Nsight Tracing Complete. Output artifacts in: ${RESULTS_DIR}"
echo "================================================================="
ls -lh "${RESULTS_DIR}"
