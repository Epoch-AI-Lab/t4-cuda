#ifndef FUSED_BACKWARD_ADAMW_H
#define FUSED_BACKWARD_ADAMW_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>

void launch_fused_backward_gemm_adamw(
    const half* dY,
    const half* X,
    float* W_master,
    half* W_active,
    float* exp_avg,
    float* exp_avg_sq,
    int M, int N, int K,
    float lr, float beta1, float beta2,
    float eps, float weight_decay,
    float bias_correction1, float bias_correction2,
    cudaStream_t stream = 0);

#endif // FUSED_BACKWARD_ADAMW_H
