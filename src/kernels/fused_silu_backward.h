#ifndef FUSED_SILU_BACKWARD_H
#define FUSED_SILU_BACKWARD_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Launch vectorized FP16 SiLU backward derivative kernel on Turing sm_75:
// Computes dX = dY * SiLU'(X) inline using half2 SIMD instructions.
void launch_fused_silu_backward(
    const half* dY,
    const half* X,
    half* dX,
    int num_elements,
    cudaStream_t stream = 0);

#endif // FUSED_SILU_BACKWARD_H
