#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Fused SFT LoRA Backward + Inline AdamW Optimizer Kernel for Tesla T4 (sm_75).
 * 
 * Computes in a single fused pass:
 *   1. dB = dY^T * H_lora [d_out, r] -> Updates B, m_B, v_B with AdamW inline
 *   2. dH = dY * B        [M, r]
 *   3. dA = dH^T * X      [r, d_in] -> Updates A, m_A, v_A with AdamW inline
 *   4. dX = dH * A        [M, d_in] -> Returns backpropagated gradient to previous layer
 * 
 * Eliminates all intermediate DRAM write/read roundtrips for dA and dB.
 */
void launch_fused_sft_lora_backward_adamw(
    const half* dY,              // [M, d_out]
    const half* X,               // [M, d_in]
    const half* H_lora,          // [M, r]
    float* A_master,             // [r, d_in]
    half* A_active,              // [r, d_in]
    float* m_A,                  // [r, d_in]
    float* v_A,                  // [r, d_in]
    float* B_master,             // [d_out, r]
    half* B_active,              // [d_out, r]
    float* m_B,                  // [d_out, r]
    float* v_B,                  // [d_out, r]
    half* dX,                    // [M, d_in]
    int M, int d_in, int d_out, int r,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    float bias_correction1, float bias_correction2,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif
