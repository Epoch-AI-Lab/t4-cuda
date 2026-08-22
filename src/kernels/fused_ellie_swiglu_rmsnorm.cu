#include "fused_ellie_swiglu_rmsnorm.h"
#include "lop3_dequant.h"
#include <cuda_fp16.h>
#include <math.h>

// -------------------------------------------------------------------------
// Helper: 128-bit Vector Load
// -------------------------------------------------------------------------
__device__ __forceinline__ float4 load_float4_const(const half* ptr) {
    return *reinterpret_cast<const float4*>(ptr);
}

__device__ __forceinline__ uint32_t pack_half_dup_u32_ellie(half val) {
    uint16_t bits = __half_as_ushort(val);
    return ((uint32_t)bits << 16) | (uint32_t)bits;
}

// Fast SiLU for half2: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
__device__ __forceinline__ half2 h2silu(half2 x) {
    float x0 = __half2float(x.x);
    float x1 = __half2float(x.y);
    float s0 = x0 / (1.0f + __expf(-x0));
    float s1 = x1 / (1.0f + __expf(-x1));
    return __halves2half2(__float2half(s0), __float2half(s1));
}

// -------------------------------------------------------------------------
// 1. Standalone Fused RMSNorm Forward Kernel
// -------------------------------------------------------------------------
// Grid: (M, 1, 1), Block: 256 or 512 threads
__global__ void fused_ellie_rmsnorm_kernel(
    const half* __restrict__ x,
    const half* __restrict__ gamma,
    half* __restrict__ out_norm,
    int M, int D, float eps)
{
    int m = blockIdx.x;
    if (m >= M) return;

    const half* x_row = x + m * D;
    half* out_row = out_norm + m * D;

    int tid = threadIdx.x;
    int stride = blockDim.x;

    float sum_sq = 0.0f;

    // Vectorized 8-half (float4 = 128-bit) accumulation
    int D_vec = (D / 8) * 8;
    for (int i = tid * 8; i < D_vec; i += stride * 8) {
        float4 v = load_float4_const(x_row + i);
        const half2* h2 = reinterpret_cast<const half2*>(&v);
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            float a = __half2float(h2[k].x);
            float b = __half2float(h2[k].y);
            sum_sq += a * a + b * b;
        }
    }

    // Handle tail if D is not a multiple of 8
    for (int i = D_vec + tid; i < D; i += stride) {
        float val = __half2float(x_row[i]);
        sum_sq += val * val;
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, offset);
    }

    // Inter-warp shared memory reduction
    __shared__ float s_warp_sums[16];
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    if (lane_id == 0) {
        s_warp_sums[warp_id] = sum_sq;
    }
    __syncthreads();

    __shared__ float s_rrms;
    if (tid == 0) {
        float total_sum_sq = 0.0f;
        int num_warps = blockDim.x / 32;
        for (int w = 0; w < num_warps; w++) {
            total_sum_sq += s_warp_sums[w];
        }
        float variance = total_sum_sq / (float)D;
        s_rrms = rsqrtf(variance + eps);
    }
    __syncthreads();

    float rrms = s_rrms;

    // Write out normalized & scaled output
    for (int i = tid * 8; i < D_vec; i += stride * 8) {
        float4 v_x = load_float4_const(x_row + i);
        float4 v_g = load_float4_const(gamma + i);

        const half2* h2_x = reinterpret_cast<const half2*>(&v_x);
        const half2* h2_g = reinterpret_cast<const half2*>(&v_g);

        half2 out_h2[4];
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            float x_0 = __half2float(h2_x[k].x) * rrms * __half2float(h2_g[k].x);
            float x_1 = __half2float(h2_x[k].y) * rrms * __half2float(h2_g[k].y);
            out_h2[k] = __halves2half2(__float2half(x_0), __float2half(x_1));
        }

        *reinterpret_cast<float4*>(out_row + i) = *reinterpret_cast<float4*>(out_h2);
    }

    for (int i = D_vec + tid; i < D; i += stride) {
        float val = __half2float(x_row[i]) * rrms * __half2float(gamma[i]);
        out_row[i] = __float2half(val);
    }
}

// -------------------------------------------------------------------------
// 2. Standalone Fused SwiGLU Forward Kernel: silu(gate) * up
// -------------------------------------------------------------------------
// Grid: (ceil((M * H) / (256 * 2)), 1), Block: 256 threads
__global__ void fused_ellie_swiglu_kernel(
    const half* __restrict__ gate,
    const half* __restrict__ up,
    half* __restrict__ out,
    int total_elements)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = tid * 2; // Process 2 elements (half2) per thread

    if (idx + 1 < total_elements) {
        half2 g = *reinterpret_cast<const half2*>(gate + idx);
        half2 u = *reinterpret_cast<const half2*>(up + idx);
        half2 silu_g = h2silu(g);
        half2 res = __hmul2(silu_g, u);
        *reinterpret_cast<half2*>(out + idx) = res;
    } else if (idx < total_elements) {
        float g_f = __half2float(gate[idx]);
        float u_f = __half2float(up[idx]);
        float silu_g = g_f / (1.0f + __expf(-g_f));
        out[idx] = __float2half(silu_g * u_f);
    }
}

// -------------------------------------------------------------------------
// 3. Fused Ellie RMSNorm + Dual INT4 GEMV (Gate + Up) + SwiGLU Mega-Kernel
// Optimized for M=1 Autoregressive Token Decoding on Tesla T4 (Turing CC 7.5)
// -------------------------------------------------------------------------
__global__ void fused_ellie_rmsnorm_w4a16_gemv_swiglu_kernel(
    const half* __restrict__ x,           // (1 x D)
    const half* __restrict__ gamma,       // (D)
    const uint32_t* __restrict__ W_gate,  // (D/8 x H)
    const half* __restrict__ scale_gate,  // (num_groups x H)
    const half* __restrict__ zp_gate,     // (num_groups x H)
    const uint32_t* __restrict__ W_up,    // (D/8 x H)
    const half* __restrict__ scale_up,    // (num_groups x H)
    const half* __restrict__ zp_up,       // (num_groups x H)
    half* __restrict__ out_swiglu,        // (1 x H)
    int D, int H, int group_size, float eps)
{
    extern __shared__ char s_mem[];
    half* s_norm_x = reinterpret_cast<half*>(s_mem); // size: D * sizeof(half)

    int tid = threadIdx.x;
    int bdim = blockDim.x;

    // --- PHASE 1: Cooperative RMSNorm in Shared Memory ---
    float sum_sq = 0.0f;
    for (int i = tid * 8; i < D; i += bdim * 8) {
        if (i + 7 < D) {
            float4 v = load_float4_const(x + i);
            const half2* h2 = reinterpret_cast<const half2*>(&v);
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                float a = __half2float(h2[k].x);
                float b = __half2float(h2[k].y);
                sum_sq += a * a + b * b;
            }
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, offset);
    }

    __shared__ float s_rms_red[8];
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    if (lane_id == 0 && warp_id < 8) {
        s_rms_red[warp_id] = sum_sq;
    }
    __syncthreads();

    __shared__ float s_rrms;
    if (tid == 0) {
        float total = 0.0f;
        int num_warps = bdim / 32;
        for (int w = 0; w < num_warps; w++) {
            total += s_rms_red[w];
        }
        s_rrms = rsqrtf((total / (float)D) + eps);
    }
    __syncthreads();

    float rrms = s_rrms;

    // Stage normalized activations into shared memory
    for (int i = tid * 8; i < D; i += bdim * 8) {
        if (i + 7 < D) {
            float4 vx = load_float4_const(x + i);
            float4 vg = load_float4_const(gamma + i);
            const half2* h2x = reinterpret_cast<const half2*>(&vx);
            const half2* h2g = reinterpret_cast<const half2*>(&vg);
            half2 res[4];
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                float v0 = __half2float(h2x[k].x) * rrms * __half2float(h2g[k].x);
                float v1 = __half2float(h2x[k].y) * rrms * __half2float(h2g[k].y);
                res[k] = __halves2half2(__float2half(v0), __float2half(v1));
            }
            *reinterpret_cast<float4*>(s_norm_x + i) = *reinterpret_cast<float4*>(res);
        }
    }
    __syncthreads();

    // --- PHASE 2: Fused Dual W4A16 GEMV (Gate + Up) with LOP3.b32 (LUT 0x6A) ---
    // Grid: (ceil(H/4), 1)  Block: 256 threads -> 64 threads per output column
    int col = blockIdx.x * 4 + (tid % 4);
    int tid_k = tid / 4;
    int stride_k = bdim / 4;

    float accum_gate = 0.0f;
    float accum_up = 0.0f;
    int d_uint32_total = D / 8; // Each uint32 contains 8 INT4 weights

    if (col < H) {
        uint32_t scale_gate_32 = 0, bias_gate_32 = 0;
        uint32_t scale_up_32 = 0, bias_up_32 = 0;
        int cur_group = -1;

        #pragma unroll 2
        for (int k_idx = tid_k; k_idx < d_uint32_total; k_idx += stride_k) {
            int k_pos = k_idx * 8;
            int g = k_pos / group_size;

            if (g != cur_group) {
                cur_group = g;
                int gi = g * H + col;

                half sg = scale_gate[gi];
                half zg = (zp_gate != nullptr) ? zp_gate[gi] : __float2half(0.0f);
                float sg_f = __half2float(sg);
                float zg_f = __half2float(zg);
                float bg_f = (-1032.0f - zg_f) * sg_f;
                scale_gate_32 = pack_half_dup_u32_ellie(sg);
                bias_gate_32 = pack_half_dup_u32_ellie(__float2half(bg_f));

                half su = scale_up[gi];
                half zu = (zp_up != nullptr) ? zp_up[gi] : __float2half(0.0f);
                float su_f = __half2float(su);
                float zu_f = __half2float(zu);
                float bu_f = (-1032.0f - zu_f) * su_f;
                scale_up_32 = pack_half_dup_u32_ellie(su);
                bias_up_32 = pack_half_dup_u32_ellie(__float2half(bu_f));
            }

            // 1. Load INT4 packed weights
            uint32_t pw_g = W_gate[k_idx * H + col];
            uint32_t pw_u = W_up[k_idx * H + col];

            // 2. LOP3 0x6A Single-Cycle Dequantization
            uint32_t g_04, g_15, g_26, g_37;
            lop3_unpack_s4_ptx(pw_g, g_04, g_15, g_26, g_37, scale_gate_32, bias_gate_32);

            uint32_t u_04, u_15, u_26, u_37;
            lop3_unpack_s4_ptx(pw_u, u_04, u_15, u_26, u_37, scale_up_32, bias_up_32);

            // 3. Load 8 activations from Shared Memory
            const half* norm_ptr = s_norm_x + k_pos;
            half2 a_04 = __halves2half2(norm_ptr[0], norm_ptr[4]);
            half2 a_15 = __halves2half2(norm_ptr[1], norm_ptr[5]);
            half2 a_26 = __halves2half2(norm_ptr[2], norm_ptr[6]);
            half2 a_37 = __halves2half2(norm_ptr[3], norm_ptr[7]);

            // 4. Multiply-Accumulate for Gate
            half2 pg_04 = __hmul2(reinterpret_cast<const half2&>(g_04), a_04);
            half2 pg_15 = __hmul2(reinterpret_cast<const half2&>(g_15), a_15);
            half2 pg_26 = __hmul2(reinterpret_cast<const half2&>(g_26), a_26);
            half2 pg_37 = __hmul2(reinterpret_cast<const half2&>(g_37), a_37);

            accum_gate += __half2float(pg_04.x) + __half2float(pg_04.y);
            accum_gate += __half2float(pg_15.x) + __half2float(pg_15.y);
            accum_gate += __half2float(pg_26.x) + __half2float(pg_26.y);
            accum_gate += __half2float(pg_37.x) + __half2float(pg_37.y);

            // 5. Multiply-Accumulate for Up
            half2 pu_04 = __hmul2(reinterpret_cast<const half2&>(u_04), a_04);
            half2 pu_15 = __hmul2(reinterpret_cast<const half2&>(u_15), a_15);
            half2 pu_26 = __hmul2(reinterpret_cast<const half2&>(u_26), a_26);
            half2 pu_37 = __hmul2(reinterpret_cast<const half2&>(u_37), a_37);

            accum_up += __half2float(pu_04.x) + __half2float(pu_04.y);
            accum_up += __half2float(pu_15.x) + __half2float(pu_15.y);
            accum_up += __half2float(pu_26.x) + __half2float(pu_26.y);
            accum_up += __half2float(pu_37.x) + __half2float(pu_37.y);
        }
    }

    // Warp-level column reduction
    #pragma unroll
    for (int offset = 16; offset >= 4; offset /= 2) {
        accum_gate += __shfl_xor_sync(0xFFFFFFFF, accum_gate, offset);
        accum_up   += __shfl_xor_sync(0xFFFFFFFF, accum_up, offset);
    }

    // Inter-warp reduction via shared memory
    __shared__ float smem_gate[8][4];
    __shared__ float smem_up[8][4];

    int w_id = tid / 32;
    int l_id = tid % 32;
    int col_in_warp = l_id % 4;

    if (l_id < 4) {
        smem_gate[w_id][col_in_warp] = accum_gate;
        smem_up[w_id][col_in_warp]   = accum_up;
    }
    __syncthreads();

    // Final thread write with inline SwiGLU activation
    if (w_id == 0 && l_id < 4) {
        float final_gate = 0.0f;
        float final_up   = 0.0f;
        int num_warps = bdim / 32;
        for (int w = 0; w < num_warps; w++) {
            final_gate += smem_gate[w][l_id];
            final_up   += smem_up[w][l_id];
        }

        int final_col = blockIdx.x * 4 + l_id;
        if (final_col < H) {
            // Compute SwiGLU: silu(final_gate) * final_up
            float silu_val = final_gate / (1.0f + __expf(-final_gate));
            float swiglu_out = silu_val * final_up;
            out_swiglu[final_col] = __float2half(swiglu_out);
        }
    }
}

// -------------------------------------------------------------------------
// Host Launch Wrappers
// -------------------------------------------------------------------------
void launch_fused_ellie_rmsnorm_forward(
    const half* x,
    const half* gamma,
    half* out_norm,
    int M, int D, float eps,
    cudaStream_t stream)
{
    dim3 grid(M);
    dim3 block(256);
    fused_ellie_rmsnorm_kernel<<<grid, block, 0, stream>>>(
        x, gamma, out_norm, M, D, eps);
}

void launch_fused_ellie_swiglu_forward(
    const half* gate,
    const half* up,
    half* out,
    int M, int H,
    cudaStream_t stream)
{
    int total = M * H;
    int block = 256;
    int grid = (total + block * 2 - 1) / (block * 2);
    fused_ellie_swiglu_kernel<<<grid, block, 0, stream>>>(
        gate, up, out, total);
}

void launch_fused_ellie_rmsnorm_w4a16_gemv_swiglu(
    const half* x,
    const half* gamma,
    const uint32_t* W_gate,
    const half* scale_gate,
    const half* zp_gate,
    const uint32_t* W_up,
    const half* scale_up,
    const half* zp_up,
    half* out_swiglu,
    int D, int H, int group_size, float eps,
    cudaStream_t stream)
{
    dim3 grid((H + 3) / 4);
    dim3 block(256);
    size_t smem_bytes = D * sizeof(half);

    fused_ellie_rmsnorm_w4a16_gemv_swiglu_kernel<<<grid, block, smem_bytes, stream>>>(
        x, gamma,
        W_gate, scale_gate, zp_gate,
        W_up, scale_up, zp_up,
        out_swiglu,
        D, H, group_size, eps);
}
