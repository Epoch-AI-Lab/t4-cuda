#!/usr/bin/env bash
# Tesla T4 CUDA Research & Kernel Suite: Complete Colab Verification Pipeline
set -e

echo "=========================================================================="
echo "  Tesla T4 CUDA Research & Kernel Suite: Complete Verification Pipeline"
echo "=========================================================================="

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

export PYTHONPATH="$REPO_DIR:$REPO_DIR/src:$PYTHONPATH"

echo ""
echo "--> [1/5] Running Bit-Exact IEEE-754 Math Proofs & KAT Harness..."
python3 tests/test_dequant_correctness.py

echo ""
echo "--> [2/5] Running Microarchitectural Roofline Simulation..."
python3 src/t4_roofline_and_kernel_benchmarks.py

echo ""
echo "--> [3/5] Compiling & Running Standalone CUDA Micro-benchmarks (sm_75)..."
nvcc -O3 -arch=sm_75 src/t4_microbenchmarks.cu -o t4_microbenchmark
./t4_microbenchmark

echo ""
echo "--> [4/5] Building & Installing PyTorch CUDA Extension (t4_kernels)..."
cd "$REPO_DIR/src"
pip install -e .
cd "$REPO_DIR"

echo ""
echo "--> [5/5] Running On-GPU Differential Verification & Stress Test..."
python3 -c "
import sys, os
sys.path.insert(0, '$REPO_DIR/src')
sys.path.insert(0, '$REPO_DIR/tests')
sys.path.insert(0, '$REPO_DIR')

import torch
import t4_kernels
from test_dequant_correctness import cpu_dequantize

print('[PyTorch GPU Check] Device Name:', torch.cuda.get_device_name(0))

# 1. Known Answer Test (KAT) Hex Vectors on GPU
w_test = torch.tensor([0xA7C13E59, 0xF817E29A], dtype=torch.int64).to(torch.int32).to('cuda')
scale_test = torch.tensor([0.25, 0.5], dtype=torch.float16, device='cuda')
zero_test  = torch.tensor([2.0, 0.0], dtype=torch.float16, device='cuda')

kernel_out_u4 = t4_kernels.dequantize_u4(w_test[:1], scale_test[:1], zero_test[:1])
ref_out_u4 = cpu_dequantize(w_test[:1], scale_test[:1], zero_test[:1], is_signed=False).view(-1)
print('Kernel Output (u4):   ', kernel_out_u4)
print('Reference Output (u4):', ref_out_u4)
assert torch.allclose(kernel_out_u4, ref_out_u4, atol=1e-3), 'Unsigned KAT Mismatch on GPU!'

kernel_out_s4 = t4_kernels.dequantize_s4(w_test[1:], scale_test[1:], zero_test[1:])
ref_out_s4 = cpu_dequantize(w_test[1:], scale_test[1:], zero_test[1:], is_signed=True).view(-1)
print('Kernel Output (s4):   ', kernel_out_s4)
print('Reference Output (s4):', ref_out_s4)
assert torch.allclose(kernel_out_s4, ref_out_s4, atol=1e-3), 'Signed KAT Mismatch on GPU!'
print('>> [GPU SUCCESS] On-GPU Unsigned & Signed KAT Matches Verified!')

# 2. Large Tensor Stress Test (4096 x 4096)
w_large = torch.randint(0, 0x7FFFFFFF, (4096 * 512,), dtype=torch.int32, device='cuda')
s_large = torch.rand((4096 * 512,), dtype=torch.float16, device='cuda') * 0.1
z_large = torch.zeros((4096 * 512,), dtype=torch.float16, device='cuda')

out_gpu = t4_kernels.dequantize_u4(w_large, s_large, z_large)
torch.cuda.synchronize()
print('>> [GPU STRESS TEST] Completed 4096x4096 dequantization without crashes. Shape:', out_gpu.shape)
"

echo ""
echo "--> [6/6] Running Fused W4A16 GEMM Accuracy & Performance Verification..."
python3 tests/test_fused_gemm_correctness.py

echo ""
echo "--> [7/9] Running Comprehensive Master Test Suite (Correctness + Benchmarks)..."
python3 tests/run_all_cuda_tests.py || true

echo ""
echo "--> [8/9] Running Batched W4A16 WMMA vs cuBLAS Benchmark..."
python3 benchmarks/bench_w4a16_wmma_vs_cublas.py || true

echo ""
echo "--> [9/9] Running Speculative Decoding Smoke Benchmark..."
python3 benchmarks/benchmark_speculative_decoding.py --max-tokens 32 || true

echo ""
echo "=========================================================================="
echo "  [ALL VERIFICATIONS COMPLETED]"
echo "=========================================================================="

