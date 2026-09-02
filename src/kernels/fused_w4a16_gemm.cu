#include "fused_w4a16_gemm.h"
#include "lop3_dequant.h"
#include <cuda_fp16.h>

__device__ __forceinline__ uint32_t pack_half_dup_u32_fused(half val) {
    uint16_t bits = __half_as_ushort(val);
    return ((uint32_t)bits << 16) | (uint32_t)bits;
}

// -------------------------------------------------------------------------
// Fused Unsigned INT4 GEMV Kernel for Small M (M = 1 .. 8)
// -------------------------------------------------------------------------
// Grid: (N / 4, M)
// Block: (256 threads) -> 8 warps work on 4 columns of output
__global__ void fused_w4a16_gemv_u4_kernel(
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

    int tid_k = threadIdx.x / 4; // 0 .. 63
    int stride_k = blockDim.x / 4; // 64 threads working along K per column

    uint32_t one_32 = pack_half_dup_u32_fused(__float2half(1.0f));
    uint32_t zero_32 = pack_half_dup_u32_fused(__float2half(0.0f));

    const half* A_row = A + m * K;
    const uint32_t* W_col = W_packed + col; // Column stride N uint32s

    float accum_f = 0.0f;
    int k_uint32_total = K / 8;
    bool is_group_128 = (group_size == 128);

    for (int k_idx = tid_k; k_idx < k_uint32_total; k_idx += stride_k) {
        int scale_idx = col;
        if (group_size > 0) {
            int g = is_group_128 ? (k_idx >> 4) : ((k_idx * 8) / group_size);
            scale_idx = g * N + col;
        }
        float s_f = __half2float(scale[scale_idx]);
        float z_f = __half2float(zero_point[scale_idx]);

        // 1. Load packed uint32 weight (8 INT4 weights)
        uint32_t packed_w = W_col[k_idx * N];

        // 2. LOP3 0xEA single-cycle register dequantization (kept exact: (1024+q) in fp16)
        uint32_t raw_04, raw_15, raw_26, raw_37;
        lop3_unpack_u4_ptx(packed_w, raw_04, raw_15, raw_26, raw_37, one_32, zero_32);

        // 3. Load 8 FP16 activations corresponding to these K positions
        const half* A_ptr = A_row + k_idx * 8;

        const half* ra04 = reinterpret_cast<const half*>(&raw_04);
        const half* ra15 = reinterpret_cast<const half*>(&raw_15);
        const half* ra26 = reinterpret_cast<const half*>(&raw_26);
        const half* ra37 = reinterpret_cast<const half*>(&raw_37);

        // Subtract 1024.0f to recover exact integer q in [0, 15] in FP32
        float q0 = __half2float(ra04[0]) - 1024.0f;
        float q4 = __half2float(ra04[1]) - 1024.0f;
        float q1 = __half2float(ra15[0]) - 1024.0f;
        float q5 = __half2float(ra15[1]) - 1024.0f;
        float q2 = __half2float(ra26[0]) - 1024.0f;
        float q6 = __half2float(ra26[1]) - 1024.0f;
        float q3 = __half2float(ra37[0]) - 1024.0f;
        float q7 = __half2float(ra37[1]) - 1024.0f;

        float dot = (q0 - z_f) * __half2float(A_ptr[0]) + (q4 - z_f) * __half2float(A_ptr[4])
                  + (q1 - z_f) * __half2float(A_ptr[1]) + (q5 - z_f) * __half2float(A_ptr[5])
                  + (q2 - z_f) * __half2float(A_ptr[2]) + (q6 - z_f) * __half2float(A_ptr[6])
                  + (q3 - z_f) * __half2float(A_ptr[3]) + (q7 - z_f) * __half2float(A_ptr[7]);

        accum_f += s_f * dot;
    }

    // Parallel reduction across threads assigned to the same column
    #pragma unroll
    for (int offset = 16; offset >= 4; offset /= 2) {
        accum_f += __shfl_xor_sync(0xFFFFFFFF, accum_f, offset);
    }

    // Shared memory reduction across warps
    __shared__ float smem_red[64][4]; // 16 warps x 4 cols
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
#include <mma.h>
using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define BLOCK_M 32
#define BLOCK_N 32
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

    // Double buffers for Matrix A and Matrix B
    __shared__ half shmem_A[2][BLOCK_M * PADDED_K_A]; // 2 x 64 x 40 x 2B = 10,240B
    __shared__ half shmem_B[2][BLOCK_K * PADDED_N_B]; // 2 x 32 x 72 x 2B = 9,216B

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
    // Overlay shmem_C on shmem_A & shmem_B (which are no longer needed)
    float* shmem_C = reinterpret_cast<float*>(shmem_A);

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

// Host Launcher Wrappers
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
        // Scalar GEMV kernel optimized for single-sequence decode
        dim3 grid((N + 3) / 4, M);
        dim3 block(256);
        fused_w4a16_gemv_u4_kernel<<<grid, block, 0, stream>>>(
            d_A, d_W_packed, d_scale, d_zero, d_C, M, N, K, group_size);
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
