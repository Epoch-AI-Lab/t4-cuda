#include "fused_swiglu_backward.h"
#include <cuda_fp16.h>

__device__ __forceinline__ half2 fast_sigmoid_half2(half2 x) {
    const half2 one = __float2half2_rn(1.0f);
    half2 neg_x = __hneg2(x);
    half2 exp_neg_x = h2exp(neg_x);
    half2 denom = __hadd2(one, exp_neg_x);
    return __h2div(one, denom);
}

__device__ __forceinline__ void swiglu_backward_step_half2(
    half2 dy, half2 g, half2 u,
    half2& dg, half2& du)
{
    const half2 one = __float2half2_rn(1.0f);
    const half2 zero = __float2half2_rn(0.0f);

    // sig_g = sigmoid(g)
    half2 sig_g = fast_sigmoid_half2(g);
    // silu_g = g * sig_g
    half2 silu_g = __hmul2(g, sig_g);

    // du = dy * silu(g)
    du = __hmul2(dy, silu_g);

    // d_silu = sig_g * (1.0 + g * (1.0 - sig_g))
    half2 one_minus_sig = __hsub2(one, sig_g);
    half2 g_one_minus_sig = __hmul2(g, one_minus_sig);
    half2 inner = __hadd2(one, g_one_minus_sig);
    half2 d_silu = __hmul2(sig_g, inner);

    // dg = dy * u * d_silu
    half2 dy_u = __hmul2(dy, u);
    dg = __hmul2(dy_u, d_silu);
}

__device__ __forceinline__ void swiglu_backward_step_scalar(
    half dy, half g, half u,
    half& dg, half& du)
{
    float dy_f = __half2float(dy);
    float g_f  = __half2float(g);
    float u_f  = __half2float(u);

    float sig_g = 1.0f / (1.0f + expf(-g_f));
    float silu_g = g_f * sig_g;

    float du_f = dy_f * silu_g;
    float d_silu = sig_g * (1.0f + g_f * (1.0f - sig_g));
    float dg_f = dy_f * u_f * d_silu;

    du = __float2half(du_f);
    dg = __float2half(dg_f);
}

// 128-bit Vectorized Kernel (8 halfs / 16 bytes per thread)
__global__ void __launch_bounds__(256, 4) fused_swiglu_backward_vector8_kernel(
    const half* __restrict__ dY,
    const half* __restrict__ gate,
    const half* __restrict__ up,
    half* __restrict__ d_gate,
    half* __restrict__ d_up,
    int num_elements)
{
    int num_vec8 = num_elements / 8;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_vec8) {
        int offset = idx * 8;
        uint4 dy_u4   = *reinterpret_cast<const uint4*>(dY + offset);
        uint4 gate_u4 = *reinterpret_cast<const uint4*>(gate + offset);
        uint4 up_u4   = *reinterpret_cast<const uint4*>(up + offset);

        const half2* dy_h2   = reinterpret_cast<const half2*>(&dy_u4);
        const half2* gate_h2 = reinterpret_cast<const half2*>(&gate_u4);
        const half2* up_h2   = reinterpret_cast<const half2*>(&up_u4);

        uint4 dg_u4, du_u4;
        half2* dg_h2 = reinterpret_cast<half2*>(&dg_u4);
        half2* du_h2 = reinterpret_cast<half2*>(&du_u4);

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            swiglu_backward_step_half2(
                dy_h2[i], gate_h2[i], up_h2[i],
                dg_h2[i], du_h2[i]
            );
        }

        *reinterpret_cast<uint4*>(d_gate + offset) = dg_u4;
        *reinterpret_cast<uint4*>(d_up + offset)   = du_u4;
    }

    // Remainder tail handling
    int tail_start = num_vec8 * 8;
    int tail_idx = tail_start + idx;
    if (tail_idx < num_elements) {
        swiglu_backward_step_scalar(
            dY[tail_idx], gate[tail_idx], up[tail_idx],
            d_gate[tail_idx], d_up[tail_idx]
        );
    }
}

void launch_fused_swiglu_backward(
    const half* dY,
    const half* gate,
    const half* up,
    half* d_gate,
    half* d_up,
    int num_elements,
    cudaStream_t stream)
{
    int num_vec8 = num_elements / 8;
    int block_dim = 256;
    int grid_dim = (num_vec8 + block_dim - 1) / block_dim;
    if (grid_dim == 0) grid_dim = 1;

    fused_swiglu_backward_vector8_kernel<<<grid_dim, block_dim, 0, stream>>>(
        dY, gate, up, d_gate, d_up, num_elements);
}
