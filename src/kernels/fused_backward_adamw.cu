#include "fused_backward_adamw.h"
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

constexpr int SMEM_K_STRIDE = 72;
constexpr int GRAD_STRIDE = 68;

// Launch bounds: 128 threads/block, max 2 blocks/SM (256 threads = 25% occupancy for H5 70W power pacing)
__global__ void __launch_bounds__(128, 2) fused_backward_gemm_adamw_kernel(
    const half* __restrict__ dY,       // [K, M]
    const half* __restrict__ X,        // [K, N]
    float* __restrict__ W_master,      // [M, N]
    half* __restrict__ W_active,       // [M, N]
    float* __restrict__ exp_avg,       // [M, N]
    float* __restrict__ exp_avg_sq,    // [M, N]
    const int M, const int N, const int K,
    const float lr, const float beta1, const float beta2,
    const float eps, const float weight_decay,
    const float bias_correction1, const float bias_correction2)
{
    __shared__ half s_dY[2][16][SMEM_K_STRIDE];
    __shared__ half s_X[2][16][SMEM_K_STRIDE];
    __shared__ float s_grad[64][GRAD_STRIDE];

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;

    const int warp_row = warp_id / 2;
    const int warp_col = warp_id % 2;

    const int block_m = blockIdx.y * 64;
    const int block_n = blockIdx.x * 64;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[2][2];
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            wmma::fill_fragment(c_frag[i][j], 0.0f);
        }
    }

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[2];

    int write_buf = 0;

    // Load initial 16x64 tile into shared memory
    #pragma unroll
    for (int i = tid; i < (16 * 64) / 8; i += blockDim.x) {
        int k_idx = i / 8;
        int m_idx = (i % 8) * 8;
        if (k_idx < K && (block_m + m_idx + 7) < M) {
            *reinterpret_cast<uint4*>(&s_dY[write_buf][k_idx][m_idx]) =
                *reinterpret_cast<const uint4*>(&dY[k_idx * M + block_m + m_idx]);
        } else {
            #pragma unroll
            for (int v = 0; v < 8; ++v) {
                s_dY[write_buf][k_idx][m_idx + v] = (k_idx < K && (block_m + m_idx + v) < M) ?
                    dY[k_idx * M + block_m + m_idx + v] : __float2half(0.0f);
            }
        }
    }

    #pragma unroll
    for (int i = tid; i < (16 * 64) / 8; i += blockDim.x) {
        int k_idx = i / 8;
        int n_idx = (i % 8) * 8;
        if (k_idx < K && (block_n + n_idx + 7) < N) {
            *reinterpret_cast<uint4*>(&s_X[write_buf][k_idx][n_idx]) =
                *reinterpret_cast<const uint4*>(&X[k_idx * N + block_n + n_idx]);
        } else {
            #pragma unroll
            for (int v = 0; v < 8; ++v) {
                s_X[write_buf][k_idx][n_idx + v] = (k_idx < K && (block_n + n_idx + v) < N) ?
                    X[k_idx * N + block_n + n_idx + v] : __float2half(0.0f);
            }
        }
    }
    __syncthreads();

    // Main K-loop accumulation
    for (int k_tile_idx = 0; k_tile_idx < K; k_tile_idx += 16) {
        int read_buf = write_buf;
        write_buf ^= 1;

        int next_k = k_tile_idx + 16;
        if (next_k < K) {
            #pragma unroll
            for (int i = tid; i < (16 * 64) / 8; i += blockDim.x) {
                int k_idx = i / 8;
                int m_idx = (i % 8) * 8;
                if ((next_k + k_idx) < K && (block_m + m_idx + 7) < M) {
                    *reinterpret_cast<uint4*>(&s_dY[write_buf][k_idx][m_idx]) =
                        *reinterpret_cast<const uint4*>(&dY[(next_k + k_idx) * M + block_m + m_idx]);
                } else {
                    #pragma unroll
                    for (int v = 0; v < 8; ++v) {
                        s_dY[write_buf][k_idx][m_idx + v] = ((next_k + k_idx) < K && (block_m + m_idx + v) < M) ?
                            dY[(next_k + k_idx) * M + block_m + m_idx + v] : __float2half(0.0f);
                    }
                }
            }

            #pragma unroll
            for (int i = tid; i < (16 * 64) / 8; i += blockDim.x) {
                int k_idx = i / 8;
                int n_idx = (i % 8) * 8;
                if ((next_k + k_idx) < K && (block_n + n_idx + 7) < N) {
                    *reinterpret_cast<uint4*>(&s_X[write_buf][k_idx][n_idx]) =
                        *reinterpret_cast<const uint4*>(&X[(next_k + k_idx) * N + block_n + n_idx]);
                } else {
                    #pragma unroll
                    for (int v = 0; v < 8; ++v) {
                        s_X[write_buf][k_idx][n_idx + v] = ((next_k + k_idx) < K && (block_n + n_idx + v) < N) ?
                            X[(next_k + k_idx) * N + block_n + n_idx + v] : __float2half(0.0f);
                    }
                }
            }
        }

        wmma::load_matrix_sync(a_frag[0], &s_dY[read_buf][0][warp_row * 32], SMEM_K_STRIDE);
        wmma::load_matrix_sync(a_frag[1], &s_dY[read_buf][0][warp_row * 32 + 16], SMEM_K_STRIDE);

        wmma::load_matrix_sync(b_frag[0], &s_X[read_buf][0][warp_col * 32], SMEM_K_STRIDE);
        wmma::load_matrix_sync(b_frag[1], &s_X[read_buf][0][warp_col * 32 + 16], SMEM_K_STRIDE);

        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
            }
        }

        __syncthreads();
    }

    // Store accumulated gradients to SMEM with guaranteed layout
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            wmma::store_matrix_sync(
                &s_grad[warp_row * 32 + i * 16][warp_col * 32 + j * 16],
                c_frag[i][j],
                GRAD_STRIDE,
                wmma::mem_row_major
            );
        }
    }
    __syncthreads();

    // Coalesced elementwise AdamW update directly from SMEM
    #pragma unroll 4
    for (int idx = tid; idx < 64 * 64; idx += blockDim.x) {
        int r = idx / 64;
        int c = idx % 64;
        int global_r = block_m + r;
        int global_c = block_n + c;

        if (global_r < M && global_c < N) {
            float grad = s_grad[r][c];
            int offset = global_r * N + global_c;

            float w_val = W_master[offset];
            float m_val = exp_avg[offset];
            float v_val = exp_avg_sq[offset];

            // AdamW Decoupled Weight Decay
            w_val -= lr * weight_decay * w_val;

            // First & second moment updates
            m_val = beta1 * m_val + (1.0f - beta1) * grad;
            v_val = beta2 * v_val + (1.0f - beta2) * (grad * grad);

            // Bias correction
            float m_hat = m_val / bias_correction1;
            float v_hat = v_val / bias_correction2;

            // Parameter update
            w_val -= lr * (m_hat / (sqrtf(v_hat) + eps));

            // Write updated master and active weights + moments
            W_master[offset] = w_val;
            W_active[offset] = __float2half(w_val);
            exp_avg[offset] = m_val;
            exp_avg_sq[offset] = v_val;
        }
    }
}

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
    cudaStream_t stream)
{
    dim3 grid((N + 63) / 64, (M + 63) / 64);
    dim3 block(128);

    fused_backward_gemm_adamw_kernel<<<grid, block, 0, stream>>>(
        dY, X, W_master, W_active, exp_avg, exp_avg_sq,
        M, N, K,
        lr, beta1, beta2, eps, weight_decay,
        bias_correction1, bias_correction2);
}
