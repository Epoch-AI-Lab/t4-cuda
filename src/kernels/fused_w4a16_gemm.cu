#include "fused_w4a16_gemm.h"
#include "lop3_dequant.h"
#include <cuda_fp16.h>
#include <cstdlib>

__device__ __forceinline__ uint32_t pack_half_dup_u32_fused(half val) {
    uint16_t bits = __half_as_ushort(val);
    return ((uint32_t)bits << 16) | (uint32_t)bits;
}

// -------------------------------------------------------------------------
// Re-Tiled Fused Unsigned INT4 GEMV Kernel
// 100% Memory Coalesced along N dimension using vectorized 128-bit loads
// Eliminates 87.5% memory bus waste and restores full GDDR6 throughput
// -------------------------------------------------------------------------
#define GEMV_WARPS_PER_BLOCK 4
#define GEMV_COLS_PER_BLOCK 128

__device__ __forceinline__ void fused_w4a16_gemv_u4_retiled_impl(
    const half* __restrict__ A,           // Shape: M x K
    const uint32_t* __restrict__ W_packed,// Shape: K/8 x N
    const half* __restrict__ scale,       // Shape: num_groups x N or 1 x N
    const half* __restrict__ zero_point,  // Shape: num_groups x N or 1 x N
    half* __restrict__ C,                 // Shape: M x N
    int M, int N, int K,
    int group_size)
{
    int m = blockIdx.y;
    if (m >= M) return;

    int block_col_base = blockIdx.x * GEMV_COLS_PER_BLOCK;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    int k_uint32_total = K / 8;
    const half* A_row = A + m * K;

    // Each thread in a warp handles 4 consecutive columns along N:
    // col_0 = block_col_base + lane_id * 4 + 0
    // col_1 = block_col_base + lane_id * 4 + 1
    // col_2 = block_col_base + lane_id * 4 + 2
    // col_3 = block_col_base + lane_id * 4 + 3
    int col_t = block_col_base + lane_id * 4;

    float accum[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    uint32_t one_32 = pack_half_dup_u32_fused(__float2half(1.0f));
    uint32_t zero_32 = pack_half_dup_u32_fused(__float2half(0.0f));

    bool is_group_128 = (group_size == 128);

    // Fast path: block completely within matrix N dimension
    if (block_col_base + GEMV_COLS_PER_BLOCK <= N) {
        float s_f[4], z_f[4];
        if (group_size == 0) {
            #pragma unroll
            for (int v = 0; v < 4; ++v) {
                s_f[v] = __half2float(scale[col_t + v]);
                z_f[v] = __half2float(zero_point[col_t + v]);
            }
        }

        // K loop: Warps split K dimension with stride GEMV_WARPS_PER_BLOCK
        for (int k_idx = warp_id; k_idx < k_uint32_total; k_idx += GEMV_WARPS_PER_BLOCK) {
            if (group_size > 0) {
                int g = is_group_128 ? (k_idx >> 4) : ((k_idx * 8) / group_size);
                int g_stride = g * N;
                #pragma unroll
                for (int v = 0; v < 4; ++v) {
                    s_f[v] = __half2float(scale[g_stride + col_t + v]);
                    z_f[v] = __half2float(zero_point[g_stride + col_t + v]);
                }
            }

            // 1. Vectorized 128-bit load of 4 uint32 weights
            // Across all 32 threads in the warp, this loads 512 contiguous bytes:
            // exactly four 128-byte cache lines with 100% bus utilization.
            const uint4* w_row_u4 = reinterpret_cast<const uint4*>(W_packed + k_idx * N + block_col_base);
            uint4 packed_vec = w_row_u4[lane_id];

            // 2. Vectorized 128-bit load of 8 FP16 activations
            // Shared across all threads in the warp for this K step.
            const float4* a_vec_ptr = reinterpret_cast<const float4*>(A_row + k_idx * 8);
            float4 a_raw = *a_vec_ptr;
            const half* a_half = reinterpret_cast<const half*>(&a_raw);

            float a_f[8];
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                a_f[i] = __half2float(a_half[i]);
            }

            // 3. Process each of the 4 columns in the uint4 vector
            uint32_t w_arr[4] = {packed_vec.x, packed_vec.y, packed_vec.z, packed_vec.w};

            #pragma unroll
            for (int v = 0; v < 4; ++v) {
                uint32_t raw_04, raw_15, raw_26, raw_37;
                lop3_unpack_u4_ptx(w_arr[v], raw_04, raw_15, raw_26, raw_37, one_32, zero_32);

                const half* ra04 = reinterpret_cast<const half*>(&raw_04);
                const half* ra15 = reinterpret_cast<const half*>(&raw_15);
                const half* ra26 = reinterpret_cast<const half*>(&raw_26);
                const half* ra37 = reinterpret_cast<const half*>(&raw_37);

                float q0 = __half2float(ra04[0]) - 1024.0f;
                float q4 = __half2float(ra04[1]) - 1024.0f;
                float q1 = __half2float(ra15[0]) - 1024.0f;
                float q5 = __half2float(ra15[1]) - 1024.0f;
                float q2 = __half2float(ra26[0]) - 1024.0f;
                float q6 = __half2float(ra26[1]) - 1024.0f;
                float q3 = __half2float(ra37[0]) - 1024.0f;
                float q7 = __half2float(ra37[1]) - 1024.0f;

                float z = z_f[v];
                float dot = (q0 - z) * a_f[0] + (q4 - z) * a_f[4]
                          + (q1 - z) * a_f[1] + (q5 - z) * a_f[5]
                          + (q2 - z) * a_f[2] + (q6 - z) * a_f[6]
                          + (q3 - z) * a_f[3] + (q7 - z) * a_f[7];

                accum[v] += s_f[v] * dot;
            }
        }
    } else {
        // Safe fallback path for edge columns where N is not a multiple of 128
        for (int k_idx = warp_id; k_idx < k_uint32_total; k_idx += GEMV_WARPS_PER_BLOCK) {
            int g = 0;
            if (group_size > 0) {
                g = is_group_128 ? (k_idx >> 4) : ((k_idx * 8) / group_size);
            }
            int g_stride = g * N;

            float a_f[8];
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                a_f[i] = (k_idx * 8 + i < K) ? __half2float(A_row[k_idx * 8 + i]) : 0.0f;
            }

            #pragma unroll
            for (int v = 0; v < 4; ++v) {
                int col = col_t + v;
                if (col < N) {
                    float s = __half2float(scale[g_stride + col]);
                    float z = __half2float(zero_point[g_stride + col]);
                    uint32_t packed_w = W_packed[k_idx * N + col];

                    uint32_t raw_04, raw_15, raw_26, raw_37;
                    lop3_unpack_u4_ptx(packed_w, raw_04, raw_15, raw_26, raw_37, one_32, zero_32);

                    const half* ra04 = reinterpret_cast<const half*>(&raw_04);
                    const half* ra15 = reinterpret_cast<const half*>(&raw_15);
                    const half* ra26 = reinterpret_cast<const half*>(&raw_26);
                    const half* ra37 = reinterpret_cast<const half*>(&raw_37);

                    float q0 = __half2float(ra04[0]) - 1024.0f;
                    float q4 = __half2float(ra04[1]) - 1024.0f;
                    float q1 = __half2float(ra15[0]) - 1024.0f;
                    float q5 = __half2float(ra15[1]) - 1024.0f;
                    float q2 = __half2float(ra26[0]) - 1024.0f;
                    float q6 = __half2float(ra26[1]) - 1024.0f;
                    float q3 = __half2float(ra37[0]) - 1024.0f;
                    float q7 = __half2float(ra37[1]) - 1024.0f;

                    float dot = (q0 - z) * a_f[0] + (q4 - z) * a_f[4]
                              + (q1 - z) * a_f[1] + (q5 - z) * a_f[5]
                              + (q2 - z) * a_f[2] + (q6 - z) * a_f[6]
                              + (q3 - z) * a_f[3] + (q7 - z) * a_f[7];

                    accum[v] += s * dot;
                }
            }
        }
    }

    // --- Reduction across Warps ---
    __shared__ float smem_red[GEMV_WARPS_PER_BLOCK][GEMV_COLS_PER_BLOCK];

    #pragma unroll
    for (int v = 0; v < 4; ++v) {
        smem_red[warp_id][lane_id * 4 + v] = accum[v];
    }
    __syncthreads();

    // 128 threads reduce 128 columns in parallel with zero bank conflicts
    int tid = threadIdx.x;
    if (tid < GEMV_COLS_PER_BLOCK) {
        float sum = 0.0f;
        #pragma unroll
        for (int w = 0; w < GEMV_WARPS_PER_BLOCK; ++w) {
            sum += smem_red[w][tid];
        }
        int final_col = block_col_base + tid;
        if (final_col < N) {
            C[m * N + final_col] = __float2half(sum);
        }
    }
}

__global__ void __launch_bounds__(128, 8) fused_w4a16_gemv_u4_retiled_kernel(
    const half* __restrict__ A,           // Shape: M x K
    const uint32_t* __restrict__ W_packed,// Shape: K/8 x N
    const half* __restrict__ scale,       // Shape: num_groups x N or 1 x N
    const half* __restrict__ zero_point,  // Shape: num_groups x N or 1 x N
    half* __restrict__ C,                 // Shape: M x N
    int M, int N, int K,
    int group_size)
{
    fused_w4a16_gemv_u4_retiled_impl(A, W_packed, scale, zero_point, C, M, N, K, group_size);
}

__global__ void __launch_bounds__(128, 8) fused_w4a16_gemv_u4_kernel(
    const half* __restrict__ A,           // Shape: M x K
    const uint32_t* __restrict__ W_packed,// Shape: K/8 x N
    const half* __restrict__ scale,       // Shape: num_groups x N or 1 x N
    const half* __restrict__ zero_point,  // Shape: num_groups x N or 1 x N
    half* __restrict__ C,                 // Shape: M x N
    int M, int N, int K,
    int group_size)
{
    fused_w4a16_gemv_u4_retiled_impl(A, W_packed, scale, zero_point, C, M, N, K, group_size);
}

// -------------------------------------------------------------------------
// Fused Signed INT4 (S4A16) GEMV Kernel for Small M (M = 1 .. 8)
// -------------------------------------------------------------------------
__global__ void fused_w4a16_gemv_s4_kernel(
    const half* __restrict__ A,          // (M x K)
    const uint32_t* __restrict__ W_packed,// (K/8 x N)
    const half* __restrict__ scale,      // (num_groups x N) or (1 x N)
    const half* __restrict__ zero_point, // (num_groups x N) or (1 x N)
    half* __restrict__ C,                // (M x N)
    int M, int N, int K,
    int group_size)
{
    int col = blockIdx.x * 4 + (threadIdx.x % 4);
    int m = blockIdx.y;

    if (col >= N || m >= M) return;

    int tid_k = threadIdx.x / 4;
    int stride_k = blockDim.x / 4;

    const half* A_row = A + m * K;
    const uint32_t* W_col = W_packed + col;

    float accum_f = 0.0f;
    int k_uint32_total = K / 8;
    bool is_group_128 = (group_size == 128);

    for (int k_idx = tid_k; k_idx < k_uint32_total; k_idx += stride_k) {
        int scale_idx = col;
        if (group_size > 0) {
            int g = is_group_128 ? (k_idx >> 4) : ((k_idx * 8) / group_size);
            scale_idx = g * N + col;
        }
        half s = scale[scale_idx];
        half z = zero_point[scale_idx];

        float s_f = __half2float(s);
        float z_f = __half2float(z);
        float bias_f = (-1032.0f - z_f) * s_f;
        half bias_h = __float2half(bias_f);

        uint32_t scale_32 = pack_half_dup_u32_fused(s);
        uint32_t neg_bias_1032_32 = pack_half_dup_u32_fused(bias_h);

        uint32_t packed_w = W_col[k_idx * N];

        // LOP3 0x6A single-cycle sign-flip + exponent injection
        uint32_t raw_04, raw_15, raw_26, raw_37;
        lop3_unpack_s4_ptx(packed_w, raw_04, raw_15, raw_26, raw_37, scale_32, neg_bias_1032_32);

        const half* A_ptr = A_row + k_idx * 8;
        half2 a_04 = __halves2half2(A_ptr[0], A_ptr[4]);
        half2 a_15 = __halves2half2(A_ptr[1], A_ptr[5]);
        half2 a_26 = __halves2half2(A_ptr[2], A_ptr[6]);
        half2 a_37 = __halves2half2(A_ptr[3], A_ptr[7]);

        half2 prod04 = __hmul2(reinterpret_cast<const half2&>(raw_04), a_04);
        half2 prod15 = __hmul2(reinterpret_cast<const half2&>(raw_15), a_15);
        half2 prod26 = __hmul2(reinterpret_cast<const half2&>(raw_26), a_26);
        half2 prod37 = __hmul2(reinterpret_cast<const half2&>(raw_37), a_37);

        accum_f += __half2float(prod04.x) + __half2float(prod04.y);
        accum_f += __half2float(prod15.x) + __half2float(prod15.y);
        accum_f += __half2float(prod26.x) + __half2float(prod26.y);
        accum_f += __half2float(prod37.x) + __half2float(prod37.y);
    }

    #pragma unroll
    for (int offset = 16; offset >= 4; offset /= 2) {
        accum_f += __shfl_xor_sync(0xFFFFFFFF, accum_f, offset);
    }

    __shared__ float smem_red[64][4];
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int col_in_warp = lane_id % 4;

    if (lane_id < 4) {
        smem_red[warp_id][col_in_warp] = accum_f;
    }
    __syncthreads();

    if (warp_id == 0 && lane_id < 4) {
        float final_sum = 0.0f;
        int num_warps = blockDim.x / 32;
        for (int w = 0; w < num_warps; w++) {
            final_sum += smem_red[w][lane_id];
        }
        int final_col = blockIdx.x * 4 + lane_id;
        if (final_col < N) {
            C[m * N + final_col] = __float2half(final_sum);
        }
    }
}

// ============================================================================
// Tensor Core WMMA Batched Kernel for M > 4 (sm_75)
// ============================================================================
// ============================================================================
// Tensor Core WMMA Batched Kernel for M > 4 (sm_75)
// 64x64 Tile with 2-Stage Double-Buffering & Zero Warp Divergence
// ============================================================================
#include <mma.h>
using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define BLOCK_M 64
#define BLOCK_N 64
#define BLOCK_K 32
#define PADDED_K_A 40 // 32 + 8 padding to avoid bank conflicts
#define PADDED_N_B 72 // 64 + 8 padding to avoid bank conflicts
#define PADDED_N_C 72 // 64 + 8 padding to avoid bank conflicts

__global__ void fused_w4a16_wmma_gemm_u4_kernel(
    const half* __restrict__ A,
    const uint32_t* __restrict__ W_packed,
    const half* __restrict__ scale,
    const half* __restrict__ zero_point,
    half* __restrict__ C,
    int M, int N, int K,
    int group_size)
{
    int block_m = blockIdx.y * BLOCK_M;
    int block_n = blockIdx.x * BLOCK_N;

    int warp_id = threadIdx.x / 32;
    int warp_m = (warp_id / 2) * 32; // 0 or 32
    int warp_n = (warp_id % 2) * 32; // 0 or 32

    // Aligned shared memory union for double buffers (Stage 0/1) and Epilogue (Matrix C)
    union alignas(16) SharedStorage {
        struct {
            half A[2][BLOCK_M * PADDED_K_A]; // 2 x 64 x 40 x 2B = 10,240B
            half B[2][BLOCK_K * PADDED_N_B]; // 2 x 32 x 72 x 2B = 9,216B
        } stages;
        float C[BLOCK_M * PADDED_N_C];       // 64 x 72 x 4B = 18,432B (fits inside 19,456B)
    };
    __shared__ SharedStorage shmem;
    half (*shmem_A)[BLOCK_M * PADDED_K_A] = shmem.stages.A;
    half (*shmem_B)[BLOCK_K * PADDED_N_B] = shmem.stages.B;

    // Accumulators for 32x32 sub-tile per warp (4 fragments of 16x16)
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[2][2];
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            wmma::fill_fragment(c_frag[i][j], 0.0f);
        }
    }

    int num_stages = K / BLOCK_K;

    // --- Prologue: Load Stage 0 ---
    {
        // Load Matrix A for stage 0 (64 x 32 = 2048 elements / 128 threads = 16 elements/thread)
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            int elem = threadIdx.x * 16 + i;
            int r = elem / BLOCK_K;
            int c = elem % BLOCK_K;
            int gm = block_m + r;
            int gk = c;
            half val = __float2half(0.0f);
            if (gm < M && gk < K) {
                val = A[gm * K + gk];
            }
            shmem_A[0][r * PADDED_K_A + c] = val;
        }

        // Load & Dequantize Matrix B for stage 0 (32 x 64 elements = 256 uint32_t / 128 threads = 2 uint32_t/thread)
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int elem = threadIdx.x * 2 + i;
            int r_pack = elem / BLOCK_N;
            int c_col = elem % BLOCK_N;
            int gn = block_n + c_col;

            if (gn < N) {
                uint32_t p = W_packed[r_pack * N + gn];
                int k_act = r_pack * 8;
                int g = (group_size == 128) ? (k_act >> 7) : ((group_size > 0) ? (k_act / group_size) : 0);
                float s = __half2float(scale[g * N + gn]);
                float z = __half2float(zero_point[g * N + gn]);

                #pragma unroll
                for (int w = 0; w < 8; ++w) {
                    int q = (p >> (4 * w)) & 0xF;
                    float deq = ((float)q - z) * s;
                    shmem_B[0][(r_pack * 8 + w) * PADDED_N_B + c_col] = __float2half(deq);
                }
            } else {
                #pragma unroll
                for (int w = 0; w < 8; ++w) {
                    shmem_B[0][(r_pack * 8 + w) * PADDED_N_B + c_col] = __float2half(0.0f);
                }
            }
        }
    }

    __syncthreads();

    // --- Main Loop: 2-Stage Pipelining ---
    for (int s = 0; s < num_stages; ++s) {
        int curr = s % 2;
        int next = (s + 1) % 2;
        int next_k_base = (s + 1) * BLOCK_K;

        // 1. Prefetch next stage if available
        if (s + 1 < num_stages) {
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                int elem = threadIdx.x * 16 + i;
                int r = elem / BLOCK_K;
                int c = elem % BLOCK_K;
                int gm = block_m + r;
                int gk = next_k_base + c;
                half val = __float2half(0.0f);
                if (gm < M && gk < K) {
                    val = A[gm * K + gk];
                }
                shmem_A[next][r * PADDED_K_A + c] = val;
            }

            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                int elem = threadIdx.x * 2 + i;
                int r_pack = elem / BLOCK_N;
                int c_col = elem % BLOCK_N;
                int gn = block_n + c_col;

                if (gn < N) {
                    int k_pack_idx = (next_k_base / 8) + r_pack;
                    uint32_t p = W_packed[k_pack_idx * N + gn];
                    int k_act = next_k_base + r_pack * 8;
                    int g = (group_size == 128) ? (k_act >> 7) : ((group_size > 0) ? (k_act / group_size) : 0);
                    float sc = __half2float(scale[g * N + gn]);
                    float zp = __half2float(zero_point[g * N + gn]);

                    #pragma unroll
                    for (int w = 0; w < 8; ++w) {
                        int q = (p >> (4 * w)) & 0xF;
                        float deq = ((float)q - zp) * sc;
                        shmem_B[next][(r_pack * 8 + w) * PADDED_N_B + c_col] = __float2half(deq);
                    }
                } else {
                    #pragma unroll
                    for (int w = 0; w < 8; ++w) {
                        shmem_B[next][(r_pack * 8 + w) * PADDED_N_B + c_col] = __float2half(0.0f);
                    }
                }
            }
        }

        // 2. Tensor Core Math on current stage (2 sub-steps of K=16)
        #pragma unroll
        for (int k_sub = 0; k_sub < 2; ++k_sub) {
            int k_off = k_sub * WMMA_K;

            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a0, a1;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b0, b1;

            wmma::load_matrix_sync(a0, &shmem_A[curr][warp_m * PADDED_K_A + k_off], PADDED_K_A);
            wmma::load_matrix_sync(a1, &shmem_A[curr][(warp_m + 16) * PADDED_K_A + k_off], PADDED_K_A);

            wmma::load_matrix_sync(b0, &shmem_B[curr][k_off * PADDED_N_B + warp_n], PADDED_N_B);
            wmma::load_matrix_sync(b1, &shmem_B[curr][k_off * PADDED_N_B + warp_n + 16], PADDED_N_B);

            wmma::mma_sync(c_frag[0][0], a0, b0, c_frag[0][0]);
            wmma::mma_sync(c_frag[0][1], a0, b1, c_frag[0][1]);
            wmma::mma_sync(c_frag[1][0], a1, b0, c_frag[1][0]);
            wmma::mma_sync(c_frag[1][1], a1, b1, c_frag[1][1]);
        }

        // Single barrier per loop to synchronize shared memory for next stage
        __syncthreads();
    }

    // --- Epilogue: Store results to global memory ---
    float* shmem_C = shmem.C;

    wmma::store_matrix_sync(&shmem_C[warp_m * PADDED_N_C + warp_n], c_frag[0][0], PADDED_N_C, wmma::mem_row_major);
    wmma::store_matrix_sync(&shmem_C[warp_m * PADDED_N_C + warp_n + 16], c_frag[0][1], PADDED_N_C, wmma::mem_row_major);
    wmma::store_matrix_sync(&shmem_C[(warp_m + 16) * PADDED_N_C + warp_n], c_frag[1][0], PADDED_N_C, wmma::mem_row_major);
    wmma::store_matrix_sync(&shmem_C[(warp_m + 16) * PADDED_N_C + warp_n + 16], c_frag[1][1], PADDED_N_C, wmma::mem_row_major);

    __syncthreads();

    // Write out to C in FP16 (64 x 64 = 4096 elements / 128 threads = 32 elements/thread)
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        int elem = threadIdx.x * 32 + i;
        int r = elem / BLOCK_N;
        int c = elem % BLOCK_N;
        int gm = block_m + r;
        int gn = block_n + c;

        if (gm < M && gn < N) {
            C[gm * N + gn] = __float2half(shmem_C[r * PADDED_N_C + c]);
        }
    }
}

// -------------------------------------------------------------------------
// Split-K GEMV: tiles the grid across K so narrow-N shapes (Down-Proj)
// saturate all 40 T4 SMs. Each (nblock, m, split) block reduces its own
// K-slice into float partials; a tiny reduce kernel sums the splits.
// K-slicing is done at uint32 granularity so 128-bit vector loads stay
// aligned, and scale group indices use the global k_idx (bit-exact vs
// the non-split path up to FP add order).
// -------------------------------------------------------------------------
#define GEMV_SPLITK_MAX 16
#define GEMV_TARGET_BLOCKS 96

__global__ void __launch_bounds__(128, 8) fused_w4a16_gemv_u4_splitk_kernel(
    const half* __restrict__ A,
    const uint32_t* __restrict__ W_packed,
    const half* __restrict__ scale,
    const half* __restrict__ zero_point,
    float* __restrict__ partials,  // [split_k, M, N] float
    int M, int N, int K,
    int group_size,
    int split_k)
{
    int m = blockIdx.y;
    int split = blockIdx.z;
    if (m >= M || split >= split_k) return;

    int block_col_base = blockIdx.x * GEMV_COLS_PER_BLOCK;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    int k_uint32_total = K / 8;
    // Partition K at uint32 granularity: every global k_idx is owned by
    // exactly one split, so the union of slices covers [0, k_total).
    int k_begin = (k_uint32_total * split) / split_k;
    int k_end = (k_uint32_total * (split + 1)) / split_k;

    const half* A_row = A + m * K;
    int col_t = block_col_base + lane_id * 4;

    float accum[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    uint32_t one_32 = pack_half_dup_u32_fused(__float2half(1.0f));
    uint32_t zero_32 = pack_half_dup_u32_fused(__float2half(0.0f));

    bool is_group_128 = (group_size == 128);

    if (block_col_base + GEMV_COLS_PER_BLOCK <= N) {
        float s_f[4], z_f[4];
        if (group_size == 0) {
            #pragma unroll
            for (int v = 0; v < 4; ++v) {
                s_f[v] = __half2float(scale[col_t + v]);
                z_f[v] = __half2float(zero_point[col_t + v]);
            }
        }

        for (int k_idx = k_begin + warp_id; k_idx < k_end; k_idx += GEMV_WARPS_PER_BLOCK) {
            if (group_size > 0) {
                int g = is_group_128 ? (k_idx >> 4) : ((k_idx * 8) / group_size);
                int g_stride = g * N;
                #pragma unroll
                for (int v = 0; v < 4; ++v) {
                    s_f[v] = __half2float(scale[g_stride + col_t + v]);
                    z_f[v] = __half2float(zero_point[g_stride + col_t + v]);
                }
            }

            const uint4* w_row_u4 = reinterpret_cast<const uint4*>(W_packed + k_idx * N + block_col_base);
            uint4 packed_vec = w_row_u4[lane_id];

            const float4* a_vec_ptr = reinterpret_cast<const float4*>(A_row + k_idx * 8);
            float4 a_raw = *a_vec_ptr;
            const half* a_half = reinterpret_cast<const half*>(&a_raw);

            float a_f[8];
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                a_f[i] = __half2float(a_half[i]);
            }

            uint32_t w_arr[4] = {packed_vec.x, packed_vec.y, packed_vec.z, packed_vec.w};

            #pragma unroll
            for (int v = 0; v < 4; ++v) {
                uint32_t raw_04, raw_15, raw_26, raw_37;
                lop3_unpack_u4_ptx(w_arr[v], raw_04, raw_15, raw_26, raw_37, one_32, zero_32);

                const half* ra04 = reinterpret_cast<const half*>(&raw_04);
                const half* ra15 = reinterpret_cast<const half*>(&raw_15);
                const half* ra26 = reinterpret_cast<const half*>(&raw_26);
                const half* ra37 = reinterpret_cast<const half*>(&raw_37);

                float q0 = __half2float(ra04[0]) - 1024.0f;
                float q4 = __half2float(ra04[1]) - 1024.0f;
                float q1 = __half2float(ra15[0]) - 1024.0f;
                float q5 = __half2float(ra15[1]) - 1024.0f;
                float q2 = __half2float(ra26[0]) - 1024.0f;
                float q6 = __half2float(ra26[1]) - 1024.0f;
                float q3 = __half2float(ra37[0]) - 1024.0f;
                float q7 = __half2float(ra37[1]) - 1024.0f;

                float z = z_f[v];
                float dot = (q0 - z) * a_f[0] + (q4 - z) * a_f[4]
                          + (q1 - z) * a_f[1] + (q5 - z) * a_f[5]
                          + (q2 - z) * a_f[2] + (q6 - z) * a_f[6]
                          + (q3 - z) * a_f[3] + (q7 - z) * a_f[7];

                accum[v] += s_f[v] * dot;
            }
        }
    } else {
        // Edge columns: scalar fallback, same K-slice ownership as above.
        for (int k_idx = k_begin + warp_id; k_idx < k_end; k_idx += GEMV_WARPS_PER_BLOCK) {
            int g = 0;
            if (group_size > 0) {
                g = is_group_128 ? (k_idx >> 4) : ((k_idx * 8) / group_size);
            }
            int g_stride = g * N;

            float a_f[8];
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                a_f[i] = (k_idx * 8 + i < K) ? __half2float(A_row[k_idx * 8 + i]) : 0.0f;
            }

            #pragma unroll
            for (int v = 0; v < 4; ++v) {
                int col = col_t + v;
                if (col < N) {
                    float s = __half2float(scale[g_stride + col]);
                    float z = __half2float(zero_point[g_stride + col]);
                    uint32_t packed_w = W_packed[k_idx * N + col];

                    uint32_t raw_04, raw_15, raw_26, raw_37;
                    lop3_unpack_u4_ptx(packed_w, raw_04, raw_15, raw_26, raw_37, one_32, zero_32);

                    const half* ra04 = reinterpret_cast<const half*>(&raw_04);
                    const half* ra15 = reinterpret_cast<const half*>(&raw_15);
                    const half* ra26 = reinterpret_cast<const half*>(&raw_26);
                    const half* ra37 = reinterpret_cast<const half*>(&raw_37);

                    float q0 = __half2float(ra04[0]) - 1024.0f;
                    float q4 = __half2float(ra04[1]) - 1024.0f;
                    float q1 = __half2float(ra15[0]) - 1024.0f;
                    float q5 = __half2float(ra15[1]) - 1024.0f;
                    float q2 = __half2float(ra26[0]) - 1024.0f;
                    float q6 = __half2float(ra26[1]) - 1024.0f;
                    float q3 = __half2float(ra37[0]) - 1024.0f;
                    float q7 = __half2float(ra37[1]) - 1024.0f;

                    float dot = (q0 - z) * a_f[0] + (q4 - z) * a_f[4]
                              + (q1 - z) * a_f[1] + (q5 - z) * a_f[5]
                              + (q2 - z) * a_f[2] + (q6 - z) * a_f[6]
                              + (q3 - z) * a_f[3] + (q7 - z) * a_f[7];

                    accum[v] += s * dot;
                }
            }
        }
    }

    __shared__ float smem_red[GEMV_WARPS_PER_BLOCK][GEMV_COLS_PER_BLOCK];

    #pragma unroll
    for (int v = 0; v < 4; ++v) {
        smem_red[warp_id][lane_id * 4 + v] = accum[v];
    }
    __syncthreads();

    int tid = threadIdx.x;
    if (tid < GEMV_COLS_PER_BLOCK) {
        float sum = 0.0f;
        #pragma unroll
        for (int w = 0; w < GEMV_WARPS_PER_BLOCK; ++w) {
            sum += smem_red[w][tid];
        }
        int final_col = block_col_base + tid;
        if (final_col < N) {
            partials[((size_t)split * M + m) * N + final_col] = sum;
        }
    }
}

__global__ void gemv_splitk_reduce_kernel(
    const float* __restrict__ partials,  // [split_k, M, N] float
    half* __restrict__ C,                // [M, N] half
    int M, int N, int split_k)
{
    int m = blockIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M || n >= N) return;
    float sum = 0.0f;
    for (int s = 0; s < split_k; ++s) {
        sum += partials[((size_t)s * M + m) * N + n];
    }
    C[(size_t)m * N + n] = __float2half(sum);
}

// Host Launcher Wrappers
void launch_fused_w4a16_gemv_u4_retiled(
    const half* d_A,
    const uint32_t* d_W_packed,
    const half* d_scale,
    const half* d_zero,
    half* d_C,
    int M, int N, int K,
    int group_size,
    cudaStream_t stream)
{
    int nblocks = (N + GEMV_COLS_PER_BLOCK - 1) / GEMV_COLS_PER_BLOCK;
    int base_blocks = nblocks * M;

    // Pick a split that fills ~96 blocks (2+ waves on 40 T4 SMs).
    // Shapes that already saturate (e.g. Gate/Up, 148 blocks) keep
    // split=1, so their fast path is bit-identical to before.
    int split = 1;
    const char* env = std::getenv("T4_GEMV_SPLITK");
    if (env != nullptr && env[0] != '\0') {
        split = atoi(env);
        if (split < 1) split = 1;
        if (split > GEMV_SPLITK_MAX) split = GEMV_SPLITK_MAX;
    } else if (M <= 4 && base_blocks < 40) {
        int max_split = (base_blocks <= 8) ? GEMV_SPLITK_MAX : 8;
        split = (GEMV_TARGET_BLOCKS + base_blocks - 1) / base_blocks;
        if (split < 1) split = 1;
        if (split > max_split) split = max_split;
        // Keep each K-slice thick enough to stay memory-bound, not
        // launch-bound (>=16 uint32 = 128 K-elems per slice).
        while (split > 1 && (K / 8) / split < 16) --split;
    }

    dim3 block(GEMV_WARPS_PER_BLOCK * 32); // 128 threads
    if (split <= 1) {
        dim3 grid(nblocks, M);
        fused_w4a16_gemv_u4_retiled_kernel<<<grid, block, 0, stream>>>(
            d_A, d_W_packed, d_scale, d_zero, d_C, M, N, K, group_size);
        return;
    }

    // Cached workspace: per-device [split, M, N] float buffer, reused across
    // calls to avoid per-launch malloc latency on the decode path and eliminate
    // cross-device thrashing on dual T4.
    static float* s_partials[8] = {nullptr};
    static size_t s_cap_bytes[8] = {0};
    int cur_dev = 0;
    cudaGetDevice(&cur_dev);
    if (cur_dev < 0 || cur_dev >= 8) cur_dev = 0;

    size_t need_bytes = (size_t)split * M * N * sizeof(float);
    if (need_bytes > s_cap_bytes[cur_dev]) {
        if (s_partials[cur_dev] != nullptr) cudaFree(s_partials[cur_dev]);
        s_partials[cur_dev] = nullptr;
        s_cap_bytes[cur_dev] = 0;
        if (cudaMalloc(&s_partials[cur_dev], need_bytes) != cudaSuccess || s_partials[cur_dev] == nullptr) {
            // Alloc failed: fall back to the non-split path, never fail the call.
            s_partials[cur_dev] = nullptr;
            s_cap_bytes[cur_dev] = 0;
            dim3 grid(nblocks, M);
            fused_w4a16_gemv_u4_retiled_kernel<<<grid, block, 0, stream>>>(
                d_A, d_W_packed, d_scale, d_zero, d_C, M, N, K, group_size);
            return;
        }
        s_cap_bytes[cur_dev] = need_bytes;
    }

    dim3 grid(nblocks, M, (unsigned)split);
    fused_w4a16_gemv_u4_splitk_kernel<<<grid, block, 0, stream>>>(
        d_A, d_W_packed, d_scale, d_zero, s_partials[cur_dev], M, N, K, group_size, split);

    dim3 rgrid((N + 255) / 256, M);
    dim3 rblock(256);
    gemv_splitk_reduce_kernel<<<rgrid, rblock, 0, stream>>>(
        s_partials[cur_dev], d_C, M, N, split);
}

void launch_fused_w4a16_gemm_u4(
    const half* d_A,
    const uint32_t* d_W_packed,
    const half* d_scale,
    const half* d_zero,
    half* d_C,
    int M, int N, int K,
    int group_size,
    cudaStream_t stream)
{
    if (M <= 4) {
        // Re-tiled 100% coalesced GEMV kernel for single-sequence decode
        launch_fused_w4a16_gemv_u4_retiled(
            d_A, d_W_packed, d_scale, d_zero, d_C, M, N, K, group_size, stream);
    } else {
        // Tensor Core WMMA kernel for batched rollouts (M > 4)
        dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
        dim3 block(128);
        fused_w4a16_wmma_gemm_u4_kernel<<<grid, block, 0, stream>>>(
            d_A, d_W_packed, d_scale, d_zero, d_C, M, N, K, group_size);
    }
}

void launch_fused_w4a16_gemm_s4(
    const half* d_A,
    const uint32_t* d_W_packed,
    const half* d_scale,
    const half* d_zero,
    half* d_C,
    int M, int N, int K,
    int group_size,
    cudaStream_t stream)
{
    dim3 grid((N + 3) / 4, M);
    dim3 block(256);
    fused_w4a16_gemv_s4_kernel<<<grid, block, 0, stream>>>(
        d_A, d_W_packed, d_scale, d_zero, d_C, M, N, K, group_size);
}
