#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Fused SwiGLU Backward Kernel for LLM Training (Pre-Training and SFT).
 * 
 * Computes:
 *   d_up   = dY * SiLU(gate)
 *   d_gate = dY * up * SiLU'(gate)
 * 
 * Vectorized with 128-bit (uint4 / 8 halfs) memory transactions per thread.
 */
void launch_fused_swiglu_backward(
    const half* dY,
    const half* gate,
    const half* up,
    half* d_gate,
    half* d_up,
    int num_elements,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif
