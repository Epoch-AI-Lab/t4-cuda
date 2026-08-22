#include "fused_silu_backward.h"
#include <cuda_fp16.h>

__device__ __forceinline__ half2 fast_sigmoid_half2(half2 x) {
    const half2 one = __float2half2_rn(1.0f);
    half2 neg_x = __hneg2(x);
    half2 exp_neg_x = h2exp(neg_x);
    half2 denom = __hadd2(one, exp_neg_x);
    return __h2div(one, denom);
}

__device__ __forceinline__ half2 silu_backward_half2(half2 dy, half2 x) {
    const half2 one = __float2half2_rn(1.0f);
    const half2 zero = __float2half2_rn(0.0f);

    half2 s = fast_sigmoid_half2(x);
    half2 one_minus_s = __hsub2(one, s);
    half2 x_one_minus_s = __hmul2(x, one_minus_s);
    half2 inner = __hadd2(one, x_one_minus_s);
    half2 d_silu = __hmul2(s, inner);

    return __hfma2(dy, d_silu, zero);
}

__device__ __forceinline__ half silu_backward_half(half dy, half x) {
    float x_f = __half2float(x);
    float dy_f = __half2float(dy);
    float s = 1.0f / (1.0f + expf(-x_f));
    float d_silu = s * (1.0f + x_f * (1.0f - s));
    return __float2half(dy_f * d_silu);
}

// Vectorized 128-bit (8 halfs per thread) kernel for maximum memory bandwidth
__global__ void __launch_bounds__(256, 4) fused_silu_backward_vector8_kernel(
    const half* __restrict__ dY,
    const half* __restrict__ X,
    half* __restrict__ dX,
    int num_elements)
{
    int num_vec8 = num_elements / 8;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_vec8) {
        int offset = idx * 8;
        uint4 dy_u4 = *reinterpret_cast<const uint4*>(dY + offset);
        uint4 x_u4  = *reinterpret_cast<const uint4*>(X + offset);

        const half2* dy_h2 = reinterpret_cast<const half2*>(&dy_u4);
        const half2* x_h2  = reinterpret_cast<const half2*>(&x_u4);

        uint4 dx_u4;
        half2* dx_h2 = reinterpret_cast<half2*>(&dx_u4);

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            dx_h2[i] = silu_backward_half2(dy_h2[i], x_h2[i]);
        }

        *reinterpret_cast<uint4*>(dX + offset) = dx_u4;
    }

    // Handle remaining tail elements
    int tail_start = num_vec8 * 8;
    int tail_idx = tail_start + idx;
    if (tail_idx < num_elements) {
        dX[tail_idx] = silu_backward_half(dY[tail_idx], X[tail_idx]);
    }
}

void launch_fused_silu_backward(
    const half* dY,
    const half* X,
    half* dX,
    int num_elements,
    cudaStream_t stream)
{
    int num_vec8 = num_elements / 8;
    int block_dim = 256;
    int grid_dim = (num_vec8 + block_dim - 1) / block_dim;
    if (grid_dim == 0) grid_dim = 1;

    fused_silu_backward_vector8_kernel<<<grid_dim, block_dim, 0, stream>>>(
        dY, X, dX, num_elements);
}
