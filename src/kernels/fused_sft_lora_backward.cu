#include "fused_sft_lora_backward.h"
#include <cuda_fp16.h>
#include <math.h>

// Step 1: Accumulate dB [d_out, r] across M and update B inline with AdamW
__global__ void __launch_bounds__(256, 2) fused_sft_db_adamw_kernel(
    const half* __restrict__ dY,        // [M, d_out]
    const half* __restrict__ H_lora,   // [M, r]
    float* __restrict__ B_master,      // [d_out, r]
    half* __restrict__ B_active,       // [d_out, r]
    float* __restrict__ m_B,           // [d_out, r]
    float* __restrict__ v_B,           // [d_out, r]
    int M, int d_out, int r,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    float bc1, float bc2)
{
    // Each thread processes one (out_idx, r_idx) coordinate
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int r_idx   = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_idx < d_out && r_idx < r) {
        float grad_b = 0.0f;

        #pragma unroll 4
        for (int m = 0; m < M; ++m) {
            float dy_val = __half2float(dY[m * d_out + out_idx]);
            float h_val  = __half2float(H_lora[m * r + r_idx]);
            grad_b += dy_val * h_val;
        }

        // Inline AdamW update for B
        int b_offset = out_idx * r + r_idx;
        float param_val = B_master[b_offset];
        float m_val = m_B[b_offset];
        float v_val = v_B[b_offset];

        m_val = beta1 * m_val + (1.0f - beta1) * grad_b;
        v_val = beta2 * v_val + (1.0f - beta2) * (grad_b * grad_b);

        float m_hat = m_val / bc1;
        float v_hat = v_val / bc2;

        float update = (m_hat / (sqrtf(v_hat) + eps)) + weight_decay * param_val;
        param_val -= lr * update;

        // Write updated optimizer states & weights
        B_master[b_offset] = param_val;
        B_active[b_offset] = __float2half(param_val);
        m_B[b_offset]      = m_val;
        v_B[b_offset]      = v_val;
    }
}

// Step 2: Compute dH [M, r] = dY [M, d_out] * B [d_out, r] in FP32
__global__ void __launch_bounds__(256, 4) sft_compute_dh_kernel(
    const half* __restrict__ dY,
    const float* __restrict__ B_master,
    float* __restrict__ dH,
    int M, int d_out, int r)
{
    int m_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int r_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (m_idx < M && r_idx < r) {
        float acc = 0.0f;
        #pragma unroll 4
        for (int j = 0; j < d_out; ++j) {
            float dy_val = __half2float(dY[m_idx * d_out + j]);
            float b_val  = B_master[j * r + r_idx];
            acc += dy_val * b_val;
        }
        dH[m_idx * r + r_idx] = acc;
    }
}

// Step 3: Accumulate dA [r, d_in] across M and update A inline with AdamW
__global__ void __launch_bounds__(256, 2) fused_sft_da_adamw_kernel(
    const float* __restrict__ dH,      // [M, r]
    const half* __restrict__ X,        // [M, d_in]
    float* __restrict__ A_master,      // [r, d_in]
    half* __restrict__ A_active,       // [r, d_in]
    float* __restrict__ m_A,           // [r, d_in]
    float* __restrict__ v_A,           // [r, d_in]
    int M, int d_in, int r,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    float bc1, float bc2)
{
    int r_idx  = blockIdx.y * blockDim.y + threadIdx.y;
    int in_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (r_idx < r && in_idx < d_in) {
        float grad_a = 0.0f;

        #pragma unroll 4
        for (int m = 0; m < M; ++m) {
            float dh_val = dH[m * r + r_idx];
            float x_val  = __half2float(X[m * d_in + in_idx]);
            grad_a += dh_val * x_val;
        }

        // Inline AdamW update for A
        int a_offset = r_idx * d_in + in_idx;
        float param_val = A_master[a_offset];
        float m_val = m_A[a_offset];
        float v_val = v_A[a_offset];

        m_val = beta1 * m_val + (1.0f - beta1) * grad_a;
        v_val = beta2 * v_val + (1.0f - beta2) * (grad_a * grad_a);

        float m_hat = m_val / bc1;
        float v_hat = v_val / bc2;

        float update = (m_hat / (sqrtf(v_hat) + eps)) + weight_decay * param_val;
        param_val -= lr * update;

        A_master[a_offset] = param_val;
        A_active[a_offset] = __float2half(param_val);
        m_A[a_offset]      = m_val;
        v_A[a_offset]      = v_val;
    }
}

// Step 4: Compute dX [M, d_in] = dH [M, r] * A [r, d_in]
__global__ void __launch_bounds__(256, 4) sft_compute_dx_kernel(
    const float* __restrict__ dH,
    const float* __restrict__ A_master,
    half* __restrict__ dX,
    int M, int d_in, int r)
{
    int m_idx  = blockIdx.y * blockDim.y + threadIdx.y;
    int in_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (m_idx < M && in_idx < d_in) {
        float acc = 0.0f;
        #pragma unroll
        for (int k = 0; k < r; ++k) {
            float dh_val = dH[m_idx * r + k];
            float a_val  = A_master[k * d_in + in_idx];
            acc += dh_val * a_val;
        }
        dX[m_idx * d_in + in_idx] = __float2half(acc);
    }
}

void launch_fused_sft_lora_backward_adamw(
    const half* dY,
    const half* X,
    const half* H_lora,
    float* A_master,
    half* A_active,
    float* m_A,
    float* v_A,
    float* B_master,
    half* B_active,
    float* m_B,
    float* v_B,
    half* dX,
    int M, int d_in, int d_out, int r,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    float bc1, float bc2,
    cudaStream_t stream)
{
    // Temporary intermediate dH allocation on stream (FP32)
    float* dH = nullptr;
    cudaMallocAsync(&dH, M * r * sizeof(float), stream);

    // 1. Launch dH computation (using unmodified B_master)
    dim3 block_dh(16, 16);
    dim3 grid_dh((M + block_dh.x - 1) / block_dh.x, (r + block_dh.y - 1) / block_dh.y);
    sft_compute_dh_kernel<<<grid_dh, block_dh, 0, stream>>>(
        dY, B_master, dH, M, d_out, r
    );

    // 2. Launch dX computation (using dH and unmodified A_master)
    dim3 block_dx(16, 16);
    dim3 grid_dx((d_in + block_dx.x - 1) / block_dx.x, (M + block_dx.y - 1) / block_dx.y);
    sft_compute_dx_kernel<<<grid_dx, block_dx, 0, stream>>>(
        dH, A_master, dX, M, d_in, r
    );

    // 3. Launch dB accumulation + AdamW update for B (in-place)
    dim3 block_b(16, 16);
    dim3 grid_b((d_out + block_b.x - 1) / block_b.x, (r + block_b.y - 1) / block_b.y);
    fused_sft_db_adamw_kernel<<<grid_b, block_b, 0, stream>>>(
        dY, H_lora, B_master, B_active, m_B, v_B,
        M, d_out, r, lr, beta1, beta2, eps, weight_decay, bc1, bc2
    );

    // 4. Launch dA accumulation + AdamW update for A (in-place)
    dim3 block_a(16, 16);
    dim3 grid_a((d_in + block_a.x - 1) / block_a.x, (r + block_a.y - 1) / block_a.y);
    fused_sft_da_adamw_kernel<<<grid_a, block_a, 0, stream>>>(
        dH, X, A_master, A_active, m_A, v_A,
        M, d_in, r, lr, beta1, beta2, eps, weight_decay, bc1, bc2
    );

    cudaFreeAsync(dH, stream);
}
