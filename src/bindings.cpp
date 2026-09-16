#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>

#include "kernels/lop3_dequant.h"
#include "kernels/fused_w4a16_gemm.h"
#include "kernels/h17_mega_kernel.h"
#include "kernels/fused_backward_adamw.h"
#include "kernels/fused_ellie_swiglu_rmsnorm.h"
#include "kernels/fused_swiglu_backward.h"
#include "kernels/fused_sft_lora_backward.h"

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_SAME_DEVICE(x, primary) TORCH_CHECK((x).device() == (primary).device(), \
    #x " device mismatch: expected device ", (primary).device(), " but got ", (x).device())
#define CHECK_HALF(x) TORCH_CHECK((x).scalar_type() == torch::kHalf, #x " must be FP16")
#define CHECK_INT32(x) TORCH_CHECK((x).scalar_type() == torch::kInt32, #x " must be Int32")
#define CHECK_FLOAT(x) TORCH_CHECK((x).scalar_type() == torch::kFloat, #x " must be FP32")

torch::Tensor dequantize_lop3_u4_cuda(
    torch::Tensor packed_weights,
    torch::Tensor scales,
    torch::Tensor zero_points)
{
    CHECK_CUDA(packed_weights);
    CHECK_CUDA(scales);
    CHECK_CUDA(zero_points);

    CHECK_CONTIGUOUS(packed_weights);
    CHECK_CONTIGUOUS(scales);
    CHECK_CONTIGUOUS(zero_points);

    CHECK_SAME_DEVICE(scales, packed_weights);
    CHECK_SAME_DEVICE(zero_points, packed_weights);

    CHECK_INT32(packed_weights);
    CHECK_HALF(scales);
    CHECK_HALF(zero_points);

    const at::cuda::CUDAGuard device_guard(packed_weights.device());

    int num_uint32s = packed_weights.numel();
    TORCH_CHECK(num_uint32s > 0, "packed_weights must not be empty");
    TORCH_CHECK(scales.numel() >= num_uint32s, "scales dimension mismatch");
    TORCH_CHECK(zero_points.numel() >= num_uint32s, "zero_points dimension mismatch");
    auto options = torch::TensorOptions().dtype(torch::kHalf).device(packed_weights.device());
    torch::Tensor output = torch::empty({num_uint32s * 8}, options);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(packed_weights.device().index()).stream();

    launch_lop3_dequant_u4(
        reinterpret_cast<const uint32_t*>(packed_weights.data_ptr<int32_t>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(zero_points.data_ptr<at::Half>()),
        num_uint32s,
        stream);

    return output;
}

torch::Tensor dequantize_lop3_s4_cuda(
    torch::Tensor packed_weights,
    torch::Tensor scales,
    torch::Tensor zero_points)
{
    CHECK_CUDA(packed_weights);
    CHECK_CUDA(scales);
    CHECK_CUDA(zero_points);

    CHECK_CONTIGUOUS(packed_weights);
    CHECK_CONTIGUOUS(scales);
    CHECK_CONTIGUOUS(zero_points);

    CHECK_SAME_DEVICE(scales, packed_weights);
    CHECK_SAME_DEVICE(zero_points, packed_weights);

    CHECK_INT32(packed_weights);
    CHECK_HALF(scales);
    CHECK_HALF(zero_points);

    const at::cuda::CUDAGuard device_guard(packed_weights.device());

    int num_uint32s = packed_weights.numel();
    TORCH_CHECK(num_uint32s > 0, "packed_weights must not be empty");
    TORCH_CHECK(scales.numel() >= num_uint32s, "scales dimension mismatch");
    TORCH_CHECK(zero_points.numel() >= num_uint32s, "zero_points dimension mismatch");
    auto options = torch::TensorOptions().dtype(torch::kHalf).device(packed_weights.device());
    torch::Tensor output = torch::empty({num_uint32s * 8}, options);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(packed_weights.device().index()).stream();

    launch_lop3_dequant_s4(
        reinterpret_cast<const uint32_t*>(packed_weights.data_ptr<int32_t>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(zero_points.data_ptr<at::Half>()),
        num_uint32s,
        stream);

    return output;
}

torch::Tensor fused_w4a16_gemm_u4_cuda(
    torch::Tensor A,
    torch::Tensor W_packed,
    torch::Tensor scales,
    torch::Tensor zero_points,
    int64_t group_size = 0)
{
    CHECK_CUDA(A);
    CHECK_CUDA(W_packed);
    CHECK_CUDA(scales);
    CHECK_CUDA(zero_points);

    CHECK_CONTIGUOUS(A);
    CHECK_CONTIGUOUS(W_packed);
    CHECK_CONTIGUOUS(scales);
    CHECK_CONTIGUOUS(zero_points);

    CHECK_SAME_DEVICE(W_packed, A);
    CHECK_SAME_DEVICE(scales, A);
    CHECK_SAME_DEVICE(zero_points, A);

    CHECK_HALF(A);
    CHECK_INT32(W_packed);
    CHECK_HALF(scales);
    CHECK_HALF(zero_points);

    TORCH_CHECK(A.dim() == 2, "A must be 2D [M, K]");
    TORCH_CHECK(W_packed.dim() == 2, "W_packed must be 2D [K/8, N]");

    const at::cuda::CUDAGuard device_guard(A.device());

    int M = A.size(0);
    int K = A.size(1);
    int N = W_packed.size(1);
    TORCH_CHECK(M > 0, "M must be positive");
    if (group_size > 0) {
        TORCH_CHECK(scales.size(0) * group_size >= K, "scales dimension mismatch");
    }

    int g_size = static_cast<int>(group_size);
    if (g_size <= 0 && scales.dim() == 2 && scales.size(0) > 1) {
        g_size = K / scales.size(0);
    }

    auto options = torch::TensorOptions().dtype(torch::kHalf).device(A.device());
    torch::Tensor C = torch::empty({M, N}, options);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(A.device().index()).stream();

    launch_fused_w4a16_gemm_u4(
        reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const uint32_t*>(W_packed.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(zero_points.data_ptr<at::Half>()),
        reinterpret_cast<half*>(C.data_ptr<at::Half>()),
        M, N, K,
        g_size,
        stream);

    return C;
}

torch::Tensor fused_w4a16_gemm_s4_cuda(
    torch::Tensor A,
    torch::Tensor W_packed,
    torch::Tensor scales,
    torch::Tensor zero_points,
    int64_t group_size = 0)
{
    CHECK_CUDA(A);
    CHECK_CUDA(W_packed);
    CHECK_CUDA(scales);
    CHECK_CUDA(zero_points);

    CHECK_CONTIGUOUS(A);
    CHECK_CONTIGUOUS(W_packed);
    CHECK_CONTIGUOUS(scales);
    CHECK_CONTIGUOUS(zero_points);

    CHECK_SAME_DEVICE(W_packed, A);
    CHECK_SAME_DEVICE(scales, A);
    CHECK_SAME_DEVICE(zero_points, A);

    CHECK_HALF(A);
    CHECK_INT32(W_packed);
    CHECK_HALF(scales);
    CHECK_HALF(zero_points);

    TORCH_CHECK(A.dim() == 2, "A must be 2D [M, K]");
    TORCH_CHECK(W_packed.dim() == 2, "W_packed must be 2D [K/8, N]");

    const at::cuda::CUDAGuard device_guard(A.device());

    int M = A.size(0);
    int K = A.size(1);
    int N = W_packed.size(1);
    TORCH_CHECK(M > 0, "M must be positive");
    if (group_size > 0) {
        TORCH_CHECK(scales.size(0) * group_size >= K, "scales dimension mismatch");
    }

    int g_size = static_cast<int>(group_size);
    if (g_size <= 0 && scales.dim() == 2 && scales.size(0) > 1) {
        g_size = K / scales.size(0);
    }

    auto options = torch::TensorOptions().dtype(torch::kHalf).device(A.device());
    torch::Tensor C = torch::empty({M, N}, options);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(A.device().index()).stream();

    launch_fused_w4a16_gemm_s4(
        reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const uint32_t*>(W_packed.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(zero_points.data_ptr<at::Half>()),
        reinterpret_cast<half*>(C.data_ptr<at::Half>()),
        M, N, K,
        g_size,
        stream);

    return C;
}

torch::Tensor dequantize_lop3_s3_cuda(
    torch::Tensor packed_weights,
    torch::Tensor scales,
    torch::Tensor zero_points)
{
    CHECK_CUDA(packed_weights);
    CHECK_CUDA(scales);
    CHECK_CUDA(zero_points);

    CHECK_CONTIGUOUS(packed_weights);
    CHECK_CONTIGUOUS(scales);
    CHECK_CONTIGUOUS(zero_points);

    CHECK_SAME_DEVICE(scales, packed_weights);
    CHECK_SAME_DEVICE(zero_points, packed_weights);

    CHECK_INT32(packed_weights);
    CHECK_HALF(scales);
    CHECK_HALF(zero_points);

    const at::cuda::CUDAGuard device_guard(packed_weights.device());

    int num_uint32s = packed_weights.numel();
    TORCH_CHECK(num_uint32s > 0, "packed_weights must not be empty");
    TORCH_CHECK(scales.numel() >= num_uint32s, "scales dimension mismatch");
    TORCH_CHECK(zero_points.numel() >= num_uint32s, "zero_points dimension mismatch");
    auto options = torch::TensorOptions().dtype(torch::kHalf).device(packed_weights.device());
    torch::Tensor output = torch::empty({num_uint32s * 10}, options);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(packed_weights.device().index()).stream();

    launch_lop3_dequant_s3(
        reinterpret_cast<const uint32_t*>(packed_weights.data_ptr<int32_t>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(zero_points.data_ptr<at::Half>()),
        num_uint32s,
        stream);

    return output;
}

torch::Tensor dequantize_lop3_fp8_cuda(
    torch::Tensor packed_weights,
    torch::Tensor scales)
{
    CHECK_CUDA(packed_weights);
    CHECK_CUDA(scales);

    CHECK_CONTIGUOUS(packed_weights);
    CHECK_CONTIGUOUS(scales);

    CHECK_SAME_DEVICE(scales, packed_weights);

    CHECK_INT32(packed_weights);
    CHECK_HALF(scales);

    const at::cuda::CUDAGuard device_guard(packed_weights.device());

    int num_uint32s = packed_weights.numel();
    TORCH_CHECK(num_uint32s > 0, "packed_weights must not be empty");
    TORCH_CHECK(scales.numel() >= num_uint32s, "scales dimension mismatch");
    auto options = torch::TensorOptions().dtype(torch::kHalf).device(packed_weights.device());
    torch::Tensor output = torch::empty({num_uint32s * 4}, options);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(packed_weights.device().index()).stream();

    launch_lop3_dequant_fp8(
        reinterpret_cast<const uint32_t*>(packed_weights.data_ptr<int32_t>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
        num_uint32s,
        stream);

    return output;
}

torch::Tensor fused_h17_gemv_s3_cuda(
    torch::Tensor A,
    torch::Tensor W_packed,
    torch::Tensor scales,
    torch::Tensor zero_points = torch::Tensor())
{
    CHECK_CUDA(A);
    CHECK_CUDA(W_packed);
    CHECK_CUDA(scales);

    CHECK_CONTIGUOUS(A);
    CHECK_CONTIGUOUS(W_packed);
    CHECK_CONTIGUOUS(scales);

    CHECK_SAME_DEVICE(W_packed, A);
    CHECK_SAME_DEVICE(scales, A);

    if (zero_points.defined() && zero_points.numel() > 0) {
        CHECK_CUDA(zero_points);
        CHECK_CONTIGUOUS(zero_points);
        CHECK_SAME_DEVICE(zero_points, A);
        CHECK_HALF(zero_points);
    }

    CHECK_HALF(A);
    CHECK_INT32(W_packed);
    CHECK_HALF(scales);

    TORCH_CHECK(A.dim() == 1 || A.dim() == 2, "A must be 1D or 2D tensor");
    TORCH_CHECK(W_packed.dim() == 2, "W_packed must be a 2D tensor");

    int M = A.dim() == 1 ? 1 : A.size(0);
    int K = A.dim() == 1 ? A.size(0) : A.size(1);
    int N = W_packed.size(1);

    TORCH_CHECK(M > 0 && N > 0 && K > 0, "Dimensions M, N, K must be positive");
    // Canonical H17 packing: 10 INT3 per uint32, W_packed is (K/10) x N.
    TORCH_CHECK(K % 10 == 0, "K must be a multiple of 10 for H17 int3 packing");
    TORCH_CHECK(W_packed.size(0) == K / 10,
                "W_packed first dim must equal K/10 (canonical H17 layout)");

    // Quant group = 100 K-positions; scales/zps are per-group.
    const int num_groups = (K + 99) / 100;
    TORCH_CHECK(scales.numel() == (int64_t)num_groups * N,
                "scales must have num_groups * N elements (num_groups = ceil(K/100))");
    if (zero_points.defined() && zero_points.numel() > 0) {
        TORCH_CHECK(zero_points.numel() == (int64_t)num_groups * N,
                    "zero_points must have num_groups * N elements");
    }

    const at::cuda::CUDAGuard device_guard(A.device());

    auto options = torch::TensorOptions().dtype(torch::kHalf).device(A.device());
    torch::Tensor C = torch::empty({M, N}, options);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(A.device().index()).stream();

    const half* zp_ptr = zero_points.defined() && zero_points.numel() > 0 ?
                         reinterpret_cast<const half*>(zero_points.data_ptr<at::Half>()) : nullptr;

    launch_h17_gemv_s3(
        reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const uint32_t*>(W_packed.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
        zp_ptr,
        reinterpret_cast<half*>(C.data_ptr<at::Half>()),
        M, N, K, num_groups,
        stream);

    return C;
}

void fused_backward_gemm_adamw_cuda(
    torch::Tensor dY,
    torch::Tensor X,
    torch::Tensor W_master,
    torch::Tensor W_active,
    torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float bias_correction1,
    float bias_correction2)
{
    CHECK_CUDA(dY);
    CHECK_CUDA(X);
    CHECK_CUDA(W_master);
    CHECK_CUDA(W_active);
    CHECK_CUDA(exp_avg);
    CHECK_CUDA(exp_avg_sq);

    CHECK_CONTIGUOUS(dY);
    CHECK_CONTIGUOUS(X);
    CHECK_CONTIGUOUS(W_master);
    CHECK_CONTIGUOUS(W_active);
    CHECK_CONTIGUOUS(exp_avg);
    CHECK_CONTIGUOUS(exp_avg_sq);

    CHECK_SAME_DEVICE(X, dY);
    CHECK_SAME_DEVICE(W_master, dY);
    CHECK_SAME_DEVICE(W_active, dY);
    CHECK_SAME_DEVICE(exp_avg, dY);
    CHECK_SAME_DEVICE(exp_avg_sq, dY);

    CHECK_HALF(dY);
    CHECK_HALF(X);
    CHECK_FLOAT(W_master);
    CHECK_HALF(W_active);
    CHECK_FLOAT(exp_avg);
    CHECK_FLOAT(exp_avg_sq);

    int K = dY.size(0);
    int M = dY.size(1);
    int N = X.size(1);
    TORCH_CHECK(X.size(0) == K, "X batch dimension must match dY batch dimension");
    TORCH_CHECK(W_master.size(0) == M && W_master.size(1) == N, "W_master shape must be [M, N]");

    const at::cuda::CUDAGuard device_guard(dY.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(dY.device().index()).stream();

    launch_fused_backward_gemm_adamw(
        reinterpret_cast<const half*>(dY.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(X.data_ptr<at::Half>()),
        W_master.data_ptr<float>(),
        reinterpret_cast<half*>(W_active.data_ptr<at::Half>()),
        exp_avg.data_ptr<float>(),
        exp_avg_sq.data_ptr<float>(),
        M, N, K,
        lr, beta1, beta2, eps, weight_decay,
        bias_correction1, bias_correction2,
        stream);
}

torch::Tensor fused_ellie_rmsnorm_cuda(
    torch::Tensor x,
    torch::Tensor gamma,
    float eps = 1e-6f)
{
    CHECK_CUDA(x);
    CHECK_CUDA(gamma);

    CHECK_CONTIGUOUS(x);
    CHECK_CONTIGUOUS(gamma);

    CHECK_SAME_DEVICE(gamma, x);

    CHECK_HALF(x);
    CHECK_HALF(gamma);

    TORCH_CHECK(x.dim() >= 1, "x must have at least 1 dimension");
    int D = x.size(-1);
    TORCH_CHECK(D > 0, "hidden dimension D must be positive");
    int M = x.numel() / D;
    TORCH_CHECK(M > 0, "M must be positive");
    TORCH_CHECK(gamma.numel() == D, "gamma size must match hidden dimension D");

    const at::cuda::CUDAGuard device_guard(x.device());
    torch::Tensor out = torch::empty_like(x);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(x.device().index()).stream();

    launch_fused_ellie_rmsnorm_forward(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(gamma.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        M, D, eps, stream);

    return out;
}

torch::Tensor fused_ellie_swiglu_cuda(
    torch::Tensor gate,
    torch::Tensor up)
{
    CHECK_CUDA(gate);
    CHECK_CUDA(up);

    CHECK_CONTIGUOUS(gate);
    CHECK_CONTIGUOUS(up);

    CHECK_SAME_DEVICE(up, gate);

    CHECK_HALF(gate);
    CHECK_HALF(up);

    TORCH_CHECK(gate.sizes() == up.sizes(), "gate and up must have identical dimensions");

    int M = gate.dim() == 1 ? 1 : gate.size(0);
    int H = gate.dim() == 1 ? gate.size(0) : gate.size(1);

    const at::cuda::CUDAGuard device_guard(gate.device());
    torch::Tensor out = torch::empty_like(gate);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(gate.device().index()).stream();

    launch_fused_ellie_swiglu_forward(
        reinterpret_cast<const half*>(gate.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(up.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        M, H, stream);

    return out;
}

torch::Tensor fused_ellie_rmsnorm_w4a16_gemv_swiglu_cuda(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor W_gate,
    torch::Tensor scale_gate,
    torch::Tensor zp_gate,
    torch::Tensor W_up,
    torch::Tensor scale_up,
    torch::Tensor zp_up,
    int group_size = 128,
    float eps = 1e-6f)
{
    CHECK_CUDA(x);
    CHECK_CUDA(gamma);
    CHECK_CUDA(W_gate);
    CHECK_CUDA(scale_gate);
    CHECK_CUDA(W_up);
    CHECK_CUDA(scale_up);

    CHECK_CONTIGUOUS(x);
    CHECK_CONTIGUOUS(gamma);
    CHECK_CONTIGUOUS(W_gate);
    CHECK_CONTIGUOUS(scale_gate);
    CHECK_CONTIGUOUS(W_up);
    CHECK_CONTIGUOUS(scale_up);

    CHECK_SAME_DEVICE(gamma, x);
    CHECK_SAME_DEVICE(W_gate, x);
    CHECK_SAME_DEVICE(scale_gate, x);
    CHECK_SAME_DEVICE(W_up, x);
    CHECK_SAME_DEVICE(scale_up, x);

    if (zp_gate.defined() && zp_gate.numel() > 0) {
        CHECK_CUDA(zp_gate);
        CHECK_CONTIGUOUS(zp_gate);
        CHECK_SAME_DEVICE(zp_gate, x);
        CHECK_HALF(zp_gate);
    }
    if (zp_up.defined() && zp_up.numel() > 0) {
        CHECK_CUDA(zp_up);
        CHECK_CONTIGUOUS(zp_up);
        CHECK_SAME_DEVICE(zp_up, x);
        CHECK_HALF(zp_up);
    }

    CHECK_HALF(x);
    CHECK_HALF(gamma);
    CHECK_INT32(W_gate);
    CHECK_HALF(scale_gate);
    CHECK_INT32(W_up);
    CHECK_HALF(scale_up);

    int D = x.numel();
    int H = W_gate.size(1);
    TORCH_CHECK(W_gate.size(0) == D / 8, "W_gate rows must equal D/8 for INT4 packing");
    TORCH_CHECK(W_up.size(0) == D / 8 && W_up.size(1) == H, "W_up dimensions must match W_gate");

    const at::cuda::CUDAGuard device_guard(x.device());
    auto options = torch::TensorOptions().dtype(torch::kHalf).device(x.device());
    torch::Tensor out = torch::empty({1, H}, options);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(x.device().index()).stream();

    const half* zp_g_ptr = zp_gate.defined() && zp_gate.numel() > 0 ? reinterpret_cast<const half*>(zp_gate.data_ptr<at::Half>()) : nullptr;
    const half* zp_u_ptr = zp_up.defined() && zp_up.numel() > 0 ? reinterpret_cast<const half*>(zp_up.data_ptr<at::Half>()) : nullptr;

    launch_fused_ellie_rmsnorm_w4a16_gemv_swiglu(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(gamma.data_ptr<at::Half>()),
        reinterpret_cast<const uint32_t*>(W_gate.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(scale_gate.data_ptr<at::Half>()),
        zp_g_ptr,
        reinterpret_cast<const uint32_t*>(W_up.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(scale_up.data_ptr<at::Half>()),
        zp_u_ptr,
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        D, H, group_size, eps, stream);

    return out;
}

std::vector<torch::Tensor> fused_swiglu_backward_cuda(
    torch::Tensor dY,
    torch::Tensor gate,
    torch::Tensor up)
{
    CHECK_CUDA(dY);
    CHECK_CUDA(gate);
    CHECK_CUDA(up);

    CHECK_CONTIGUOUS(dY);
    CHECK_CONTIGUOUS(gate);
    CHECK_CONTIGUOUS(up);

    CHECK_SAME_DEVICE(gate, dY);
    CHECK_SAME_DEVICE(up, dY);

    CHECK_HALF(dY);
    CHECK_HALF(gate);
    CHECK_HALF(up);

    TORCH_CHECK(gate.numel() == dY.numel() && up.numel() == dY.numel(), "gate and up must match dY numel");

    int numel = dY.numel();
    const at::cuda::CUDAGuard device_guard(dY.device());
    auto options = torch::TensorOptions().dtype(torch::kHalf).device(dY.device());
    torch::Tensor d_gate = torch::empty_like(gate);
    torch::Tensor d_up = torch::empty_like(up);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(dY.device().index()).stream();

    launch_fused_swiglu_backward(
        reinterpret_cast<const half*>(dY.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(gate.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(up.data_ptr<at::Half>()),
        reinterpret_cast<half*>(d_gate.data_ptr<at::Half>()),
        reinterpret_cast<half*>(d_up.data_ptr<at::Half>()),
        numel,
        stream);

    return {d_gate, d_up};
}

torch::Tensor fused_sft_lora_backward_adamw_cuda(
    torch::Tensor dY,
    torch::Tensor X,
    torch::Tensor H_lora,
    torch::Tensor A_master,
    torch::Tensor A_active,
    torch::Tensor m_A,
    torch::Tensor v_A,
    torch::Tensor B_master,
    torch::Tensor B_active,
    torch::Tensor m_B,
    torch::Tensor v_B,
    float lr,
    float beta1 = 0.9f,
    float beta2 = 0.999f,
    float eps = 1e-8f,
    float weight_decay = 0.01f,
    float bias_correction1 = 1.0f,
    float bias_correction2 = 1.0f)
{
    CHECK_CUDA(dY);
    CHECK_CUDA(X);
    CHECK_CUDA(H_lora);
    CHECK_CUDA(A_master);
    CHECK_CUDA(A_active);
    CHECK_CUDA(m_A);
    CHECK_CUDA(v_A);
    CHECK_CUDA(B_master);
    CHECK_CUDA(B_active);
    CHECK_CUDA(m_B);
    CHECK_CUDA(v_B);

    CHECK_CONTIGUOUS(dY);
    CHECK_CONTIGUOUS(X);
    CHECK_CONTIGUOUS(H_lora);
    CHECK_CONTIGUOUS(A_master);
    CHECK_CONTIGUOUS(A_active);
    CHECK_CONTIGUOUS(m_A);
    CHECK_CONTIGUOUS(v_A);
    CHECK_CONTIGUOUS(B_master);
    CHECK_CONTIGUOUS(B_active);
    CHECK_CONTIGUOUS(m_B);
    CHECK_CONTIGUOUS(v_B);

    CHECK_SAME_DEVICE(X, dY);
    CHECK_SAME_DEVICE(H_lora, dY);
    CHECK_SAME_DEVICE(A_master, dY);
    CHECK_SAME_DEVICE(A_active, dY);
    CHECK_SAME_DEVICE(m_A, dY);
    CHECK_SAME_DEVICE(v_A, dY);
    CHECK_SAME_DEVICE(B_master, dY);
    CHECK_SAME_DEVICE(B_active, dY);
    CHECK_SAME_DEVICE(m_B, dY);
    CHECK_SAME_DEVICE(v_B, dY);

    CHECK_HALF(dY);
    CHECK_HALF(X);
    CHECK_HALF(H_lora);
    CHECK_FLOAT(A_master);
    CHECK_HALF(A_active);
    CHECK_FLOAT(m_A);
    CHECK_FLOAT(v_A);
    CHECK_FLOAT(B_master);
    CHECK_HALF(B_active);
    CHECK_FLOAT(m_B);
    CHECK_FLOAT(v_B);

    int M = dY.size(0);
    int d_out = dY.size(1);
    int d_in = X.size(1);
    int r = H_lora.size(1);

    const at::cuda::CUDAGuard device_guard(dY.device());
    auto options = torch::TensorOptions().dtype(torch::kHalf).device(dY.device());
    torch::Tensor dX = torch::empty({M, d_in}, options);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(dY.device().index()).stream();

    launch_fused_sft_lora_backward_adamw(
        reinterpret_cast<const half*>(dY.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(X.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(H_lora.data_ptr<at::Half>()),
        A_master.data_ptr<float>(),
        reinterpret_cast<half*>(A_active.data_ptr<at::Half>()),
        m_A.data_ptr<float>(),
        v_A.data_ptr<float>(),
        B_master.data_ptr<float>(),
        reinterpret_cast<half*>(B_active.data_ptr<at::Half>()),
        m_B.data_ptr<float>(),
        v_B.data_ptr<float>(),
        reinterpret_cast<half*>(dX.data_ptr<at::Half>()),
        M, d_in, d_out, r,
        lr, beta1, beta2, eps, weight_decay, bias_correction1, bias_correction2,
        stream);

    return dX;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dequantize_u4", &dequantize_lop3_u4_cuda, "LOP3 Fast Unsigned INT4 Dequantization (CUDA)");
    m.def("dequantize_s4", &dequantize_lop3_s4_cuda, "LOP3 Fast Signed INT4 Dequantization (CUDA)");
    m.def("dequantize_s3", &dequantize_lop3_s3_cuda, "LOP3 Fast Signed INT3 Dequantization (CUDA)");
    m.def("dequantize_fp8", &dequantize_lop3_fp8_cuda, "LOP3 Fast FP8 E4M3 Dequantization (CUDA)");
    m.def("fused_w4a16_gemm_u4", &fused_w4a16_gemm_u4_cuda, "Fused Unsigned W4A16 GEMM with LOP3 0xEA Dequant (CUDA)",
          py::arg("A"), py::arg("W_packed"), py::arg("scales"), py::arg("zero_points"), py::arg("group_size") = 0);
    m.def("fused_w4a16_gemm_s4", &fused_w4a16_gemm_s4_cuda, "Fused Signed S4A16 GEMM with LOP3 0x6A Dequant (CUDA)",
          py::arg("A"), py::arg("W_packed"), py::arg("scales"), py::arg("zero_points"), py::arg("group_size") = 0);
    m.def("fused_h17_gemv_s3", &fused_h17_gemv_s3_cuda, "Flagship H17 Fused INT3 Dequant + GEMV Decode Mega-Kernel (CUDA)",
          py::arg("A"), py::arg("W_packed"), py::arg("scales"), py::arg("zero_points") = torch::Tensor());
    m.def("fused_backward_gemm_adamw", &fused_backward_gemm_adamw_cuda, "H6 Fused Backward GEMM + Inline AdamW Optimizer Kernel (CUDA)");
    m.def("fused_ellie_rmsnorm", &fused_ellie_rmsnorm_cuda, "Fused Ellie 4B RMSNorm Kernel (CUDA)");
    m.def("fused_ellie_swiglu", &fused_ellie_swiglu_cuda, "Fused Ellie 4B SwiGLU Elementwise Kernel (CUDA)");
    m.def("fused_ellie_rmsnorm_w4a16_gemv_swiglu", &fused_ellie_rmsnorm_w4a16_gemv_swiglu_cuda, "Fused Ellie 4B RMSNorm + W4A16 GEMV + SwiGLU Mega-Kernel (CUDA)");
    m.def("fused_swiglu_backward", &fused_swiglu_backward_cuda, "Fused SwiGLU Backward Elementwise Kernel for Training (CUDA)");
    m.def("fused_sft_lora_backward_adamw", &fused_sft_lora_backward_adamw_cuda, "Fused SFT LoRA Backward + Inline AdamW Kernel for SFT (CUDA)");
}
