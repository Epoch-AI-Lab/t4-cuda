import torch
import sys

try:
    import t4_kernels
    HAS_T4_KERNELS = True
except ImportError:
    HAS_T4_KERNELS = False
    print("WARNING: t4_kernels not found. Tests will skip CUDA executions.")

def unpack_u4(W_packed, K, N):
    unpacked = torch.zeros((K, N), dtype=torch.uint8)
    for i in range(8):
        unpacked[i::8, :] = (W_packed >> (i * 4)) & 0xF
    return unpacked

def unpack_s4(W_packed, K, N):
    unpacked = unpack_u4(W_packed, K, N).to(torch.int8)
    unpacked[unpacked >= 8] -= 16
    return unpacked

def dequantize_ref(W_packed, scales, zero_points, signed=False, group_size=0):
    K_8, N = W_packed.shape
    K = K_8 * 8
    if signed:
        unpacked = unpack_s4(W_packed, K, N).float()
    else:
        unpacked = unpack_u4(W_packed, K, N).float()
    
    if scales.dim() == 2 and scales.shape[0] > 1:
        num_groups = scales.shape[0]
        g_size = K // num_groups if group_size <= 0 else group_size
        unpacked_g = unpacked.reshape(num_groups, g_size, N)
        sc_3d = scales.float().unsqueeze(1)
        zp_3d = zero_points.float().unsqueeze(1)
        W_dequant = ((unpacked_g - zp_3d) * sc_3d).reshape(K, N)
    else:
        W_dequant = (unpacked - zero_points.float()) * scales.float()
    return W_dequant.half()

def cpu_reference_gemm(A, W_packed, scales, zero_points, signed=False, group_size=0):
    W_dequant = dequantize_ref(W_packed, scales, zero_points, signed=signed, group_size=group_size)
    C_ref = torch.matmul(A.float(), W_dequant.float())
    return C_ref.half()

def run_test_case(M, K, N, name, signed=False, tol=None, mean_tol=None, group_size=0):
    print(f"Running Test: {name} (M={M}, K={K}, N={N}, Signed={signed}, GroupSize={group_size})")
    
    if tol is None:
        # Standard tight tolerance for FP16 accumulation
        tol = 0.05
    if mean_tol is None:
        mean_tol = 0.01
    
    # Generate random inputs
    A = torch.randn(M, K, dtype=torch.float16, device='cuda' if HAS_T4_KERNELS else 'cpu')
    W_packed = torch.randint(0, 2**31 - 1, (K // 8, N), dtype=torch.int32, device='cuda' if HAS_T4_KERNELS else 'cpu')
    
    if group_size > 0:
        num_groups = K // group_size
        scales = torch.randn(num_groups, N, dtype=torch.float16, device='cuda' if HAS_T4_KERNELS else 'cpu') * 0.1
        zero_points = torch.randint(0, 16, (num_groups, N), dtype=torch.float16, device='cuda' if HAS_T4_KERNELS else 'cpu')
    else:
        scales = torch.randn(1, N, dtype=torch.float16, device='cuda' if HAS_T4_KERNELS else 'cpu') * 0.1
        zero_points = torch.randint(0, 16, (1, N), dtype=torch.float16, device='cuda' if HAS_T4_KERNELS else 'cpu')

    C_ref = cpu_reference_gemm(A.cpu(), W_packed.cpu(), scales.cpu(), zero_points.cpu(), signed=signed, group_size=group_size)

    if not HAS_T4_KERNELS or not torch.cuda.is_available():
        print(f"  [SKIPPED] Missing t4_kernels or CUDA device")
        return True

    if signed:
        C_out = t4_kernels.fused_w4a16_gemm_s4(A, W_packed, scales, zero_points, group_size)
    else:
        C_out = t4_kernels.fused_w4a16_gemm_u4(A, W_packed, scales, zero_points, group_size)

    max_err = torch.max(torch.abs(C_out.cpu() - C_ref)).item()
    mean_err = torch.mean(torch.abs(C_out.cpu() - C_ref)).item()
    
    if max_err > tol or mean_err > mean_tol:
        print(f"  [FAIL] Max Err: {max_err:.4f} (tol={tol:.2f}), Mean Err: {mean_err:.4f} (tol={mean_tol:.2f})")
        return False
    else:
        print(f"  [PASS] Max Err: {max_err:.4f} (tol={tol:.2f}), Mean Err: {mean_err:.4f} (tol={mean_tol:.2f})")
        return True

def test_identity():
    print("Running Test: Identity Matrix")
    if not HAS_T4_KERNELS or not torch.cuda.is_available():
        print("  [SKIP] CUDA or t4_kernels not available for identity test")
        return True

    K = 8
    N = 8
    A = torch.eye(K, dtype=torch.float16, device='cuda')
    W_packed = torch.randint(0, 2**31 - 1, (K // 8, N), dtype=torch.int32, device='cuda')
    scales = torch.ones(1, N, dtype=torch.float16, device='cuda')
    zero_points = torch.zeros(1, N, dtype=torch.float16, device='cuda')
    
    C_out = t4_kernels.fused_w4a16_gemm_u4(A, W_packed, scales, zero_points, 0)
    W_dequant = dequantize_ref(W_packed.cpu(), scales.cpu(), zero_points.cpu(), signed=False).cuda()
    
    max_err = torch.max(torch.abs(C_out - W_dequant)).item()
    if max_err < 1e-2:
        print("  [PASS] Identity matched weights exactly on CUDA kernel")
        return True
    else:
        print(f"  [FAIL] Identity mismatch on CUDA kernel: max err {max_err}")
        return False

def run_all_tests():
    passed = []
    passed.append(test_identity())
    passed.append(run_test_case(1, 8, 4, "Known Small Matrix (Per-Channel)", signed=False, group_size=0))
    passed.append(run_test_case(1, 3584, 3584, "Eli Model Sizes (Decode, Per-Channel)", signed=False, group_size=0))
    for b in [1, 2, 4, 8]:
        passed.append(run_test_case(b, 2048, 4096, f"Batch Decode M={b} (Per-Channel)", signed=True, group_size=0))
    
    # Per-group tests (group=128, GPTQ-style)
    passed.append(run_test_case(1, 896, 896, "Qwen2.5-0.5B q_proj Decode (Group=128)", signed=False, group_size=128))
    passed.append(run_test_case(1, 896, 4864, "Qwen2.5-0.5B gate_proj Decode (Group=128)", signed=False, group_size=128))
    for b in [1, 2, 4]:
        passed.append(run_test_case(b, 896, 896, f"Qwen2.5-0.5B Batch={b} (Group=128)", signed=False, group_size=128))
    
    print("\nRunning Random Fuzz...")
    max_errors = []
    mean_errors = []
    for _ in range(100):
        A = torch.randn(1, 512, dtype=torch.float16)
        W_packed = torch.randint(0, 2**31 - 1, (512 // 8, 512), dtype=torch.int32)
        scales = torch.randn(1, 512, dtype=torch.float16) * 0.1
        zero_points = torch.randint(0, 16, (1, 512), dtype=torch.float16)
        C_ref = cpu_reference_gemm(A.cpu(), W_packed.cpu(), scales.cpu(), zero_points.cpu(), signed=False)
        
        if HAS_T4_KERNELS and torch.cuda.is_available():
            C_out = t4_kernels.fused_w4a16_gemm_u4(A.cuda(), W_packed.cuda(), scales.cuda(), zero_points.cuda())
            max_err = torch.max(torch.abs(C_out.cpu() - C_ref)).item()
            mean_err = torch.mean(torch.abs(C_out.cpu() - C_ref)).item()
            max_errors.append(max_err)
            mean_errors.append(mean_err)
    if HAS_T4_KERNELS and torch.cuda.is_available():
        fuzz_pass = max(max_errors) <= 0.08
        passed.append(fuzz_pass)
        print(f"  Fuzz 100 iterations -> Max Err: {max(max_errors):.4f}, Avg Mean Err: {sum(mean_errors)/100:.4f}, Pass: {fuzz_pass}")
    
    passed.append(run_test_case(1, 8, 1, "Edge Case: Minimum K, Single N"))
    
    all_ok = all(passed)
    print("\n" + "=" * 60)
    print(f"ALL TESTS SUMMARY: {'PASSED' if all_ok else 'FAILED'}")
    print("=" * 60)
    if not all_ok:
        sys.exit(1)

if __name__ == '__main__':
    run_all_tests()

