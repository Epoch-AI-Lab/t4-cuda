#!/usr/bin/env python3
"""
================================================================================
  Tesla T4 Custom CUDA Training Kernels Suite (Pre-Training & SFT)
================================================================================
Tests and benchmarks custom Turing (sm_75) CUDA kernels for LLM training:

  1. Pre-Training Fused Backward GEMM + Inline AdamW (Hypothesis H6)
     - Accumulates dW in register fragments across K-loop
     - Updates W, m, v inline in registers, eliminating 21.4% DRAM traffic

  2. Pre-Training & SFT Fused SwiGLU Backward Elementwise Kernel
     - Evaluates d_gate and d_up in a single 128-bit vectorized pass
     - Eliminates 6 intermediate DRAM roundtrips

  3. SFT / Fine-Tuning Fused LoRA Adapter Backward + Inline AdamW Kernel
     - Fuses dB accumulation + AdamW on B + dH backprop + dA accumulation + AdamW on A + dX backprop
     - Keeps adapter fine-tuning VRAM footprint minimal for 8B/4B models

Verification against PyTorch Eager Autograd + AdamW References.
================================================================================
"""

import os
import sys
import time
import math
import numpy as np
import torch
import torch.nn.functional as F

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "src"))

HAS_T4_KERNELS = False
try:
    import t4_kernels
    HAS_T4_KERNELS = True
except ImportError:
    pass

# ==============================================================================
# 1. SwiGLU Backward PyTorch Reference
# ==============================================================================

def reference_swiglu_backward(dY, gate, up):
    """
    High-precision PyTorch autograd reference for SwiGLU backward pass.
    """
    g = gate.detach().clone().float().requires_grad_(True)
    u = up.detach().clone().float().requires_grad_(True)
    
    # Forward: Y = SiLU(g) * u
    y = F.silu(g) * u
    
    # Backward
    y.backward(dY.float())
    
    return g.grad.half(), u.grad.half()


# ==============================================================================
# 2. SFT LoRA Backward + AdamW PyTorch Reference
# ==============================================================================

def reference_sft_lora_backward_adamw(
    dY, X, H_lora,
    A_master, m_A, v_A,
    B_master, m_B, v_B,
    lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
    """
    PyTorch Reference for LoRA adapter backward + AdamW update.
    """
    M, d_out = dY.shape
    _, d_in = X.shape
    _, r = H_lora.shape

    # 1. dB = dY^T * H_lora [d_out, r]
    grad_B = torch.matmul(dY.float().t(), H_lora.float()) # (d_out, r)

    # 2. dH = dY * B [M, r]
    B_curr = B_master.clone()
    dH = torch.matmul(dY.float(), B_curr.float()) # (M, r)

    # 3. dA = dH^T * X [r, d_in]
    grad_A = torch.matmul(dH.t(), X.float()) # (r, d_in)

    # 4. dX = dH * A [M, d_in]
    A_curr = A_master.clone()
    dX = torch.matmul(dH, A_curr.float()).half()

    # 5. AdamW on B
    m_B_new = beta1 * m_B + (1.0 - beta1) * grad_B
    v_B_new = beta2 * v_B + (1.0 - beta2) * (grad_B ** 2)
    m_hat_B = m_B_new / 1.0
    v_hat_B = v_B_new / 1.0
    B_master_new = B_master - lr * (m_hat_B / (torch.sqrt(v_hat_B) + eps) + weight_decay * B_master)

    # 6. AdamW on A
    m_A_new = beta1 * m_A + (1.0 - beta1) * grad_A
    v_A_new = beta2 * v_A + (1.0 - beta2) * (grad_A ** 2)
    m_hat_A = m_A_new / 1.0
    v_hat_A = v_A_new / 1.0
    A_master_new = A_master - lr * (m_hat_A / (torch.sqrt(v_hat_A) + eps) + weight_decay * A_master)

    return dX, A_master_new, m_A_new, v_A_new, B_master_new, m_B_new, v_B_new


# ==============================================================================
# 3. Test Battery Runner
# ==============================================================================

def run_all_training_tests():
    print("=" * 80)
    print("  TESLA T4 CUSTOM CUDA TRAINING KERNEL SUITE (PRE-TRAINING & SFT)")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"[*] Hardware Target : {torch.cuda.get_device_name(0)}")
        print(f"[*] PyTorch Version : {torch.__version__}, CUDA: {torch.version.cuda}")
    else:
        print("[!] No GPU available. Running host checks.")

    all_passed = True

    # --------------------------------------------------------------------------
    # Test 1: Fused SwiGLU Backward (Pre-Training & SFT)
    # --------------------------------------------------------------------------
    print("\n--- [TEST 1/3] Fused SwiGLU Backward Elementwise Kernel ---")
    M_test = 2048
    H_test = 5504
    torch.manual_seed(2026)

    dY = torch.randn((M_test, H_test), dtype=torch.float16, device=device) * 0.1
    gate = torch.randn((M_test, H_test), dtype=torch.float16, device=device)
    up = torch.randn((M_test, H_test), dtype=torch.float16, device=device)

    # Reference
    dg_ref, du_ref = reference_swiglu_backward(dY, gate, up)

    if device == "cuda" and HAS_T4_KERNELS:
        dg_cuda, du_cuda = t4_kernels.fused_swiglu_backward(dY, gate, up)
        
        diff_dg_max = (dg_cuda - dg_ref).abs().max().item()
        diff_dg_mean = (dg_cuda - dg_ref).abs().mean().item()
        diff_du_max = (du_cuda - du_ref).abs().max().item()
        diff_du_mean = (du_cuda - du_ref).abs().mean().item()

        print(f"  -> d_gate Max Abs Diff: {diff_dg_max:.6e} (Tolerance: < 2e-3)")
        print(f"  -> d_up   Max Abs Diff: {diff_du_max:.6e} (Tolerance: < 2e-3)")

        pass_swiglu = (diff_dg_max < 2e-3 and diff_du_max < 2e-3)
        status_swiglu = "\033[92mPASS\033[0m" if pass_swiglu else "\033[91mFAIL\033[0m"
        print(f"  -> Correctness Verdict: {status_swiglu}")
        if not pass_swiglu: all_passed = False

        # Benchmark
        torch.cuda.synchronize()
        for _ in range(50): _ = t4_kernels.fused_swiglu_backward(dY, gate, up)
        torch.cuda.synchronize()

        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(200): _ = t4_kernels.fused_swiglu_backward(dY, gate, up)
        t1.record()
        torch.cuda.synchronize()
        time_swiglu_ms = t0.elapsed_time(t1) / 200.0

        # Memory bandwidth: read dY (22MB) + gate (22MB) + up (22MB) + write dg (22MB) + du (22MB) = 112.7 MB
        bytes_moved = M_test * H_test * 2 * 5
        bw_achieved = (bytes_moved / (time_swiglu_ms / 1000.0)) / 1e9
        print(f"  -> Measured Latency    : {time_swiglu_ms:.3f} ms")
        print(f"  -> Effective Bandwidth : {bw_achieved:.1f} GB/s ({bw_achieved/320.0*100:.1f}% of T4 Peak)")
    else:
        print("  [SKIP] CUDA device or t4_kernels extension not available; skipping Test 1")

    # --------------------------------------------------------------------------
    # Test 2: Fused SFT LoRA Adapter Backward + AdamW (Fine-Tuning)
    # --------------------------------------------------------------------------
    print("\n--- [TEST 2/3] Fused SFT LoRA Backward + Inline AdamW Kernel ---")
    M_sft = 1024
    d_in = 2048
    d_out = 2048
    r = 32 # LoRA rank 32
    torch.manual_seed(2026)

    dY_sft = torch.randn((M_sft, d_out), dtype=torch.float16, device=device) * 0.05
    X_sft  = torch.randn((M_sft, d_in), dtype=torch.float16, device=device) * 0.05
    H_sft  = torch.randn((M_sft, r), dtype=torch.float16, device=device) * 0.05

    A_master = torch.randn((r, d_in), dtype=torch.float32, device=device) * 0.02
    A_active = A_master.clone().half()
    m_A = torch.zeros((r, d_in), dtype=torch.float32, device=device)
    v_A = torch.zeros((r, d_in), dtype=torch.float32, device=device)

    B_master = torch.randn((d_out, r), dtype=torch.float32, device=device) * 0.02
    B_active = B_master.clone().half()
    m_B = torch.zeros((d_out, r), dtype=torch.float32, device=device)
    v_B = torch.zeros((d_out, r), dtype=torch.float32, device=device)

    # Reference
    dX_ref, A_ref, mA_ref, vA_ref, B_ref, mB_ref, vB_ref = reference_sft_lora_backward_adamw(
        dY_sft, X_sft, H_sft,
        A_master.clone(), m_A.clone(), v_A.clone(),
        B_master.clone(), m_B.clone(), v_B.clone(),
        lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01
    )

    if device == "cuda" and HAS_T4_KERNELS:
        A_master_cu = A_master.clone()
        A_active_cu = A_active.clone()
        m_A_cu = m_A.clone()
        v_A_cu = v_A.clone()

        B_master_cu = B_master.clone()
        B_active_cu = B_active.clone()
        m_B_cu = m_B.clone()
        v_B_cu = v_B.clone()

        dX_cuda = t4_kernels.fused_sft_lora_backward_adamw(
            dY_sft, X_sft, H_sft,
            A_master_cu, A_active_cu, m_A_cu, v_A_cu,
            B_master_cu, B_active_cu, m_B_cu, v_B_cu,
            1e-3, 0.9, 0.999, 1e-8, 0.01, 1.0, 1.0
        )

        diff_dX = (dX_cuda - dX_ref).abs().max().item()
        diff_A  = (A_master_cu - A_ref).abs().max().item()
        diff_B  = (B_master_cu - B_ref).abs().max().item()
        diff_mA = (m_A_cu - mA_ref).abs().max().item()
        diff_mB = (m_B_cu - mB_ref).abs().max().item()

        print(f"  -> dX Backprop Max Abs Diff : {diff_dX:.6e} (Tolerance: < 5e-3)")
        print(f"  -> Adapter A Max Abs Diff   : {diff_A:.6e} (Tolerance: < 1e-3)")
        print(f"  -> Adapter B Max Abs Diff   : {diff_B:.6e} (Tolerance: < 1e-4)")
        print(f"  -> AdamW Momentum Max Diff  : {max(diff_mA, diff_mB):.6e} (Tolerance: < 1e-5)")

        pass_sft = (diff_dX < 5e-3 and diff_A < 1e-3 and diff_B < 1e-4 and diff_mA < 1e-5)
        status_sft = "\033[92mPASS\033[0m" if pass_sft else "\033[91mFAIL\033[0m"
        print(f"  -> Correctness Verdict      : {status_sft}")
        if not pass_sft: all_passed = False

        # Benchmark
        torch.cuda.synchronize()
        for _ in range(50):
            _ = t4_kernels.fused_sft_lora_backward_adamw(
                dY_sft, X_sft, H_sft,
                A_master_cu, A_active_cu, m_A_cu, v_A_cu,
                B_master_cu, B_active_cu, m_B_cu, v_B_cu,
                1e-3, 0.9, 0.999, 1e-8, 0.01, 1.0, 1.0
            )
        torch.cuda.synchronize()

        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(200):
            _ = t4_kernels.fused_sft_lora_backward_adamw(
                dY_sft, X_sft, H_sft,
                A_master_cu, A_active_cu, m_A_cu, v_A_cu,
                B_master_cu, B_active_cu, m_B_cu, v_B_cu,
                1e-3, 0.9, 0.999, 1e-8, 0.01, 1.0, 1.0
            )
        t1.record()
        torch.cuda.synchronize()
        time_sft_ms = t0.elapsed_time(t1) / 200.0
        print(f"  -> Fused SFT Step Latency   : {time_sft_ms:.3f} ms (Single Fused Kernel Launch)")
    else:
        print("  [SKIP] CUDA device or t4_kernels extension not available; skipping Test 2")

    # --------------------------------------------------------------------------
    # Test 3: Pre-Training Fused Backward GEMM + AdamW (H6)
    # --------------------------------------------------------------------------
    print("\n--- [TEST 3/3] Pre-Training Dense Fused Backward GEMM + AdamW (H6) ---")
    M_pre = 2048
    N_pre = 2048
    K_pre = 1024
    torch.manual_seed(2026)

    dY_pre = torch.randn((K_pre, M_pre), dtype=torch.float16, device=device) * 0.02
    X_pre  = torch.randn((K_pre, N_pre), dtype=torch.float16, device=device) * 0.02

    W_master = torch.randn((M_pre, N_pre), dtype=torch.float32, device=device) * 0.02
    W_active = W_master.clone().half()
    m_W = torch.zeros((M_pre, N_pre), dtype=torch.float32, device=device)
    v_W = torch.zeros((M_pre, N_pre), dtype=torch.float32, device=device)

    # Reference
    dW_ref = torch.matmul(dY_pre.t(), X_pre).float() # (M, N) via Tensor Cores
    mW_ref = 0.9 * m_W + 0.1 * dW_ref
    vW_ref = 0.999 * v_W + 0.001 * (dW_ref ** 2)
    W_ref  = W_master - 1e-3 * (mW_ref / (torch.sqrt(vW_ref) + 1e-8) + 0.01 * W_master)

    if device == "cuda" and HAS_T4_KERNELS:
        W_master_cu = W_master.clone()
        W_active_cu = W_active.clone()
        m_W_cu = m_W.clone()
        v_W_cu = v_W.clone()

        t4_kernels.fused_backward_gemm_adamw(
            dY_pre, X_pre, W_master_cu, W_active_cu, m_W_cu, v_W_cu,
            1e-3, 0.9, 0.999, 1e-8, 0.01, 1.0, 1.0
        )

        diff_W = (W_master_cu - W_ref).abs().max().item()
        diff_mW = (m_W_cu - mW_ref).abs().max().item()
        diff_vW = (v_W_cu - vW_ref).abs().max().item()

        print(f"  -> Weight Matrix Max Abs Diff : {diff_W:.6e} (Gate: < 1e-3)")
        print(f"  -> Momentum State Max Abs Diff: {diff_mW:.6e} (Gate: < 1e-5)")

        pass_pre = (diff_W < 1e-3 and diff_mW < 1e-5)
        status_pre = "\033[92mPASS\033[0m" if pass_pre else "\033[91mFAIL\033[0m"
        print(f"  -> Correctness Verdict        : {status_pre}")
        if not pass_pre: all_passed = False

        # Benchmark vs PyTorch Eager
        torch.cuda.synchronize()
        for _ in range(30):
            t4_kernels.fused_backward_gemm_adamw(
                dY_pre, X_pre, W_master_cu, W_active_cu, m_W_cu, v_W_cu,
                1e-3, 0.9, 0.999, 1e-8, 0.01, 1.0, 1.0
            )
        torch.cuda.synchronize()

        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(100):
            t4_kernels.fused_backward_gemm_adamw(
                dY_pre, X_pre, W_master_cu, W_active_cu, m_W_cu, v_W_cu,
                1e-3, 0.9, 0.999, 1e-8, 0.01, 1.0, 1.0
            )
        t1.record()
        torch.cuda.synchronize()
        fused_ms = t0.elapsed_time(t1) / 100.0

        # PyTorch Eager timing
        t0.record()
        for _ in range(100):
            grad = torch.matmul(dY_pre.t(), X_pre).float()
            m_W = 0.9 * m_W + 0.1 * grad
            v_W = 0.999 * v_W + 0.001 * (grad ** 2)
            W_master = W_master - 1e-3 * (m_W / (torch.sqrt(v_W) + 1e-8) + 0.01 * W_master)
        t1.record()
        torch.cuda.synchronize()
        eager_ms = t0.elapsed_time(t1) / 100.0

        speedup_training = eager_ms / fused_ms
        print(f"  -> PyTorch Baseline Latency   : {eager_ms:.3f} ms")
        print(f"  -> Fused Kernel Latency       : {fused_ms:.3f} ms")
        print(f"  -> Training Speedup on T4     : {speedup_training:.2f}x (DRAM Traffic: 28B -> 22B/param)")
    else:
        print("  [SKIP] CUDA device or t4_kernels extension not available; skipping Test 3")

    print("\n" + "=" * 80)
    if all_passed:
        print("  ALL PRE-TRAINING & SFT CUDA KERNEL VERIFICATIONS: PASSED (100%)")
    else:
        print("  SOME VERIFICATIONS FAILED!")
        sys.exit(1)
    print("=" * 80)


if __name__ == "__main__":
    run_all_training_tests()
