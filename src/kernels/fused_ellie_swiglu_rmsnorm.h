#ifndef FUSED_ELLIE_SWIGLU_RMSNORM_H
#define FUSED_ELLIE_SWIGLU_RMSNORM_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Fused Ellie 4B RMSNorm + SwiGLU Kernel for NVIDIA Turing T4 (Compute Capability 7.5)
 *
 * Mathematically computes:
 *   1. RMSNorm: norm_x[i] = (x[i] / sqrt(mean(x^2) + eps)) * weight[i]
 *   2. Fused Gate + Up Projection (W4A16 with LOP3.b32 0x6A or FP16)
 *   3. SwiGLU: output[j] = silu(gate[j]) * up[j] = (gate[j] / (1 + exp(-gate[j]))) * up[j]
 *
 * Eliminates 3 memory round-trips to HBM GDDR6.
 */

void launch_fused_ellie_rmsnorm_forward(
    const half* __restrict__ x,           // (M x D) input activations
    const half* __restrict__ gamma,       // (D) RMSNorm gain weight
    half* __restrict__ out_norm,          // (M x D) normalized activations
    int M, int D, float eps,
    cudaStream_t stream);

void launch_fused_ellie_swiglu_forward(
    const half* __restrict__ gate,        // (M x H) Gate projection activations
    const half* __restrict__ up,          // (M x H) Up projection activations
    half* __restrict__ out,               // (M x H) SwiGLU output
    int M, int H,
    cudaStream_t stream);

void launch_fused_ellie_rmsnorm_w4a16_gemv_swiglu(
    const half* __restrict__ x,           // (1 x D) input token activation
    const half* __restrict__ gamma,       // (D) RMSNorm weight
    const uint32_t* __restrict__ W_gate,  // (D/8 x H) packed INT4 Gate weights
    const half* __restrict__ scale_gate,  // (num_groups x H) Gate scale
    const half* __restrict__ zp_gate,     // (num_groups x H) Gate zero point
    const uint32_t* __restrict__ W_up,    // (D/8 x H) packed INT4 Up weights
    const half* __restrict__ scale_up,    // (num_groups x H) Up scale
    const half* __restrict__ zp_up,       // (num_groups x H) Up zero point
    half* __restrict__ out_swiglu,        // (1 x H) SwiGLU output
    int D, int H, int group_size, float eps,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif // FUSED_ELLIE_SWIGLU_RMSNORM_H
