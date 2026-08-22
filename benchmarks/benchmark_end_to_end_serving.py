#!/usr/bin/env python3
"""
================================================================================
  Tesla T4 End-to-End Autoregressive Serving Engine Benchmark (Milestone A)
================================================================================
Evaluates real token generation throughput (tokens/sec) and per-token latency 
(p50, p90, p99) on physical NVIDIA Tesla T4 across 3 execution backends:

  1. PyTorch FP16 Eager Baseline (Unfused Linear + RMSNorm + SwiGLU + KV-Cache)
  2. PyTorch Inductor (torch.compile mode="reduce-overhead" with CUDA Graphs)
  3. T4 Custom Fused Quantized Engine (LOP3 W4A16 GEMV + Fused RMSNorm + SwiGLU)

Architecture Evaluated:
  - Model Target: Qwen/Ellie-style 0.5B to 4B Decoder Layer Profile
  - Layers: 24 Layers
  - Hidden Dimension (D): 2048
  - FFN Dimension (H): 5504 (SwiGLU intermediate)
  - Attention Heads: 16 (Head Dim: 128)
  - Batch Size: B=1 (Interactive single-stream generation)
  - Quantization: W4A16 Signed INT4 (LOP3 LUT 0x6A) with Group Size = 128
================================================================================
"""

import sys
import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure local modules are loadable
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "src"))

# Try importing custom CUDA extensions
HAS_T4_KERNELS = False
try:
    import t4_kernels
    HAS_T4_KERNELS = True
except ImportError:
    pass

# ==============================================================================
# 1. Quantization Helper (W4A16 Signed INT4 LOP3)
# ==============================================================================

def pack_s4_weight_matrix(weight_fp16, group_size=128):
    """
    Packs a 2D weight matrix (Out_Features x In_Features) into column-major INT4 (In_Features/8 x Out_Features)
    compatible with t4_kernels.fused_ellie_rmsnorm_w4a16_gemv_swiglu and fused_w4a16_gemv_s4.
    """
    # weight_fp16 is (Out x In), we want (In x Out) for GEMV x * W
    W_2d = weight_fp16.t().contiguous() # (In x Out)
    D, H = W_2d.shape
    assert D % group_size == 0, f"D ({D}) must be divisible by group_size ({group_size})"
    assert D % 8 == 0

    num_groups = D // group_size
    scales = torch.empty((num_groups, H), dtype=torch.float16, device=weight_fp16.device)
    zero_points = torch.empty((num_groups, H), dtype=torch.float16, device=weight_fp16.device)
    packed = torch.zeros((D // 8, H), dtype=torch.int32, device=weight_fp16.device)

    for g in range(num_groups):
        sub = W_2d[g * group_size : (g + 1) * group_size, :]
        max_val = sub.max(dim=0).values
        min_val = sub.min(dim=0).values

        scale = (max_val - min_val).clamp(min=1e-5) / 15.0
        zp = min_val + 8.0 * scale

        scales[g, :] = scale.to(torch.float16)
        zero_points[g, :] = zp.to(torch.float16)

        q = torch.clamp(torch.round((sub - zp) / scale), -8, 7).to(torch.int32) & 0x0F

        for k in range(group_size // 8):
            k_global = (g * group_size // 8) + k
            word = torch.zeros(H, dtype=torch.int32, device=weight_fp16.device)
            for nibble in range(8):
                elem = q[k * 8 + nibble, :]
                word = word | (elem << (nibble * 4))
            packed[k_global, :] = word

    return packed, scales, zero_points


# ==============================================================================
# 2. Baseline Transformer Decoder Block (PyTorch FP16 / torch.compile)
# ==============================================================================

class PyTorchRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float16))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class PyTorchAttentionDecode(nn.Module):
    """Attention layer with pre-allocated static KV-cache for single-token decode"""
    def __init__(self, dim, num_heads=16, max_seq_len=2048):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False, dtype=torch.float16)
        self.out_proj = nn.Linear(dim, dim, bias=False, dtype=torch.float16)
        self.max_seq_len = max_seq_len

    def forward(self, x, k_cache, v_cache, seq_pos):
        # x is (1, 1, dim)
        B, M, D = x.shape
        qkv = self.qkv_proj(x) # (1, 1, 3*dim)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        q = q.view(B, self.num_heads, 1, self.head_dim)
        k = k.view(B, self.num_heads, 1, self.head_dim)
        v = v.view(B, self.num_heads, 1, self.head_dim)

        # Update KV cache inplace
        k_cache[:, :, seq_pos:seq_pos+1, :] = k
        v_cache[:, :, seq_pos:seq_pos+1, :] = v

        # Read historical KV
        k_hist = k_cache[:, :, :seq_pos+1, :]
        v_hist = v_cache[:, :, :seq_pos+1, :]

        # Attention calculation
        scores = torch.matmul(q, k_hist.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1, dtype=torch.float32).half()
        context = torch.matmul(attn, v_hist).view(B, 1, D)
        return self.out_proj(context)


class PyTorchMLP(nn.Module):
    """SwiGLU FFN layer"""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float16)
        self.up_proj   = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float16)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False, dtype=torch.float16)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class PyTorchDecoderLayer(nn.Module):
    def __init__(self, dim, hidden_dim, num_heads):
        super().__init__()
        self.attn_norm = PyTorchRMSNorm(dim)
        self.attn = PyTorchAttentionDecode(dim, num_heads)
        self.mlp_norm = PyTorchRMSNorm(dim)
        self.mlp = PyTorchMLP(dim, hidden_dim)

    def forward(self, x, k_cache, v_cache, seq_pos):
        h = x + self.attn(self.attn_norm(x), k_cache, v_cache, seq_pos)
        out = h + self.mlp(self.mlp_norm(h))
        return out


# ==============================================================================
# 3. T4 Custom Fused Quantized Decoder Block
# ==============================================================================

class T4FusedDecoderLayer(nn.Module):
    """
    Decoder layer executing with custom Turing (sm_75) kernels:
      - Attn Norm: Standard / Fused
      - Attention Projections: W4A16 LOP3 GEMV
      - MLP: Fused RMSNorm + Dual W4A16 (Gate+Up) + Inline SwiGLU Mega-Kernel
      - Down Proj: W4A16 LOP3 GEMV
    """
    def __init__(self, dim, hidden_dim, num_heads, group_size=128):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.group_size = group_size

        # Norm weights
        self.attn_norm_weight = nn.Parameter(torch.ones(dim, dtype=torch.float16))
        self.mlp_norm_weight  = nn.Parameter(torch.ones(dim, dtype=torch.float16))

        # Packed QKV and Out weights (INT4 LOP3)
        qkv_raw = torch.randn((dim, 3 * dim), dtype=torch.float16) * 0.02
        out_raw = torch.randn((dim, dim), dtype=torch.float16) * 0.02
        gate_raw = torch.randn((dim, hidden_dim), dtype=torch.float16) * 0.02
        up_raw   = torch.randn((dim, hidden_dim), dtype=torch.float16) * 0.02
        down_raw = torch.randn((hidden_dim, dim), dtype=torch.float16) * 0.02

        qkv_packed, qkv_scale, qkv_zp = pack_s4_weight_matrix(qkv_raw.t(), group_size)
        out_packed, out_scale, out_zp = pack_s4_weight_matrix(out_raw.t(), group_size)
        gate_packed, gate_scale, gate_zp = pack_s4_weight_matrix(gate_raw.t(), group_size)
        up_packed, up_scale, up_zp = pack_s4_weight_matrix(up_raw.t(), group_size)
        down_packed, down_scale, down_zp = pack_s4_weight_matrix(down_raw.t(), group_size)

        self.register_buffer("qkv_packed", qkv_packed)
        self.register_buffer("qkv_scale", qkv_scale)
        self.register_buffer("qkv_zp", qkv_zp)

        self.register_buffer("out_packed", out_packed)
        self.register_buffer("out_scale", out_scale)
        self.register_buffer("out_zp", out_zp)

        self.register_buffer("gate_packed", gate_packed)
        self.register_buffer("gate_scale", gate_scale)
        self.register_buffer("gate_zp", gate_zp)

        self.register_buffer("up_packed", up_packed)
        self.register_buffer("up_scale", up_scale)
        self.register_buffer("up_zp", up_zp)

        self.register_buffer("down_packed", down_packed)
        self.register_buffer("down_scale", down_scale)
        self.register_buffer("down_zp", down_zp)

    def forward(self, x, k_cache, v_cache, seq_pos):
        B, M, D = x.shape
        x_2d = x.view(B * M, D)

        # 1. Attn Norm (Fused RMSNorm)
        if HAS_T4_KERNELS and x.is_cuda:
            norm_attn = t4_kernels.fused_ellie_rmsnorm(x_2d, self.attn_norm_weight, 1e-6)
            # 2. QKV Proj via Fused W4A16 GEMV
            qkv = t4_kernels.fused_w4a16_gemm_s4(
                norm_attn, self.qkv_packed, self.qkv_scale, self.qkv_zp
            ).view(B, 1, 3 * self.dim)
        else:
            norm_attn = x_2d * torch.rsqrt(x_2d.pow(2).mean(-1, keepdim=True) + 1e-6) * self.attn_norm_weight
            qkv = norm_attn.view(B, 1, -1) # CPU fallback mock

        q, k, v = torch.chunk(qkv, 3, dim=-1)
        q = q.view(B, self.num_heads, 1, self.head_dim)
        k = k.view(B, self.num_heads, 1, self.head_dim)
        v = v.view(B, self.num_heads, 1, self.head_dim)

        k_cache[:, :, seq_pos:seq_pos+1, :] = k
        v_cache[:, :, seq_pos:seq_pos+1, :] = v
        k_hist = k_cache[:, :, :seq_pos+1, :]
        v_hist = v_cache[:, :, :seq_pos+1, :]

        scores = torch.matmul(q, k_hist.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1, dtype=torch.float32).half()
        ctx = torch.matmul(attn, v_hist).view(B, 1, D)

        if HAS_T4_KERNELS and x.is_cuda:
            attn_out = t4_kernels.fused_w4a16_gemm_s4(
                ctx.view(B*M, D), self.out_packed, self.out_scale, self.out_zp
            ).view(B, 1, D)
        else:
            attn_out = ctx

        h = x + attn_out
        h_2d = h.view(B * M, D)

        # 3. Fused Mega-Kernel: RMSNorm + Dual W4A16 (Gate+Up) + Inline SwiGLU
        if HAS_T4_KERNELS and x.is_cuda:
            swiglu_act = t4_kernels.fused_ellie_rmsnorm_w4a16_gemv_swiglu(
                h_2d, self.mlp_norm_weight,
                self.gate_packed, self.gate_scale, self.gate_zp,
                self.up_packed, self.up_scale, self.up_zp,
                self.group_size, 1e-6
            )
            # 4. Down Projection (W4A16 GEMV)
            mlp_out = t4_kernels.fused_w4a16_gemm_s4(
                swiglu_act, self.down_packed, self.down_scale, self.down_zp
            ).view(B, 1, D)
        else:
            mlp_out = h

        return h + mlp_out


# ==============================================================================
# 4. Full Decoder Stack Model Wrapper
# ==============================================================================

class FullDecoderStack(nn.Module):
    def __init__(self, num_layers, dim, hidden_dim, num_heads, layer_cls, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            layer_cls(dim, hidden_dim, num_heads, **kwargs) for _ in range(num_layers)
        ])
        self.final_norm = PyTorchRMSNorm(dim)

    def forward(self, x, kv_k_list, kv_v_list, seq_pos):
        for i, layer in enumerate(self.layers):
            x = layer(x, kv_k_list[i], kv_v_list[i], seq_pos)
        return self.final_norm(x)


# ==============================================================================
# 5. Serving Engine Benchmarking Battery
# ==============================================================================

def run_serving_benchmark():
    print("=" * 80)
    print("  TESLA T4 END-TO-END AUTOREGRESSIVE SERVING ENGINE BENCHMARK (MILESTONE A)")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        print(f"[*] Compute Target : {gpu_name} (Compute Capability {capability[0]}.{capability[1]})")
        print(f"[*] Driver / CUDA  : PyTorch {torch.version.__version__}, CUDA {torch.version.cuda}")
    else:
        print("[!] No CUDA GPU detected. Running in host simulation mode.")

    # Architecture Profile: 24 Layers, D=2048, H=5504, 16 Heads (0.5B-1B LLM Decode Profile)
    NUM_LAYERS = 24
    DIM = 2048
    HIDDEN_DIM = 5504
    NUM_HEADS = 16
    HEAD_DIM = DIM // NUM_HEADS
    MAX_SEQ_LEN = 512
    WARMUP_TOKENS = 30
    BENCHMARK_TOKENS = 120

    print(f"[*] Model Architecture: {NUM_LAYERS} Layers | Hidden D={DIM} | FFN H={HIDDEN_DIM} | Heads={NUM_HEADS}")
    print(f"[*] Serving Mode      : Batch Size = 1, Single-Token Step Decode (B=1, M=1)")
    print(f"[*] Warmup Tokens     : {WARMUP_TOKENS} | Timed Decode Tokens: {BENCHMARK_TOKENS}")
    print("-" * 80)

    # Initialize KV-Caches for 24 layers
    def make_kv_caches():
        k_list = [torch.zeros((1, NUM_HEADS, MAX_SEQ_LEN, HEAD_DIM), dtype=torch.float16, device=device) for _ in range(NUM_LAYERS)]
        v_list = [torch.zeros((1, NUM_HEADS, MAX_SEQ_LEN, HEAD_DIM), dtype=torch.float16, device=device) for _ in range(NUM_LAYERS)]
        return k_list, v_list

    results = {}

    # --------------------------------------------------------------------------
    # Benchmark 1: PyTorch FP16 Eager Baseline
    # --------------------------------------------------------------------------
    print("\n[Engine 1/3] Benchmarking PyTorch FP16 Eager Baseline...")
    eager_model = FullDecoderStack(NUM_LAYERS, DIM, HIDDEN_DIM, NUM_HEADS, PyTorchDecoderLayer).to(device).half().eval()
    k_caches, v_caches = make_kv_caches()
    cur_token = torch.randn((1, 1, DIM), dtype=torch.float16, device=device)

    # Warmup
    with torch.no_grad():
        for pos in range(WARMUP_TOKENS):
            cur_token = eager_model(cur_token, k_caches, v_caches, pos)
        if device == "cuda":
            torch.cuda.synchronize()

    # Timed Loop
    latencies_eager = []
    with torch.no_grad():
        for pos in range(WARMUP_TOKENS, WARMUP_TOKENS + BENCHMARK_TOKENS):
            if device == "cuda":
                t0 = torch.cuda.Event(enable_timing=True)
                t1 = torch.cuda.Event(enable_timing=True)
                t0.record()
                cur_token = eager_model(cur_token, k_caches, v_caches, pos)
                t1.record()
                torch.cuda.synchronize()
                latencies_eager.append(t0.elapsed_time(t1))
            else:
                t0 = time.perf_counter()
                cur_token = eager_model(cur_token, k_caches, v_caches, pos)
                latencies_eager.append((time.perf_counter() - t0) * 1000.0)

    lat_eager = np.array(latencies_eager)
    tok_per_sec_eager = 1000.0 / np.mean(lat_eager)
    results["PyTorch FP16 Eager"] = {
        "mean_ms": np.mean(lat_eager),
        "p50_ms": np.percentile(lat_eager, 50),
        "p90_ms": np.percentile(lat_eager, 90),
        "p99_ms": np.percentile(lat_eager, 99),
        "tok_sec": tok_per_sec_eager,
        "vram_mb": torch.cuda.max_memory_allocated() / (1024**2) if device == "cuda" else 0.0
    }
    print(f"  -> Mean Latency: {np.mean(lat_eager):.2f} ms | p50: {np.percentile(lat_eager, 50):.2f} ms | p99: {np.percentile(lat_eager, 99):.2f} ms")
    print(f"  -> Throughput  : {tok_per_sec_eager:.2f} tokens/sec")

    # --------------------------------------------------------------------------
    # Benchmark 2: PyTorch Inductor (torch.compile)
    # --------------------------------------------------------------------------
    if device == "cuda":
        print("\n[Engine 2/3] Benchmarking PyTorch Inductor (torch.compile mode='reduce-overhead')...")
        try:
            k_caches, v_caches = make_kv_caches()
            cur_token = torch.randn((1, 1, DIM), dtype=torch.float16, device=device)
            compiled_model = torch.compile(eager_model, mode="reduce-overhead")

            # Warmup + Compilation
            with torch.no_grad():
                for pos in range(WARMUP_TOKENS):
                    cur_token = compiled_model(cur_token, k_caches, v_caches, pos)
                torch.cuda.synchronize()

            latencies_compile = []
            with torch.no_grad():
                for pos in range(WARMUP_TOKENS, WARMUP_TOKENS + BENCHMARK_TOKENS):
                    t0 = torch.cuda.Event(enable_timing=True)
                    t1 = torch.cuda.Event(enable_timing=True)
                    t0.record()
                    cur_token = compiled_model(cur_token, k_caches, v_caches, pos)
                    t1.record()
                    torch.cuda.synchronize()
                    latencies_compile.append(t0.elapsed_time(t1))

            lat_compile = np.array(latencies_compile)
            tok_per_sec_compile = 1000.0 / np.mean(lat_compile)
            results["torch.compile (Inductor)"] = {
                "mean_ms": np.mean(lat_compile),
                "p50_ms": np.percentile(lat_compile, 50),
                "p90_ms": np.percentile(lat_compile, 90),
                "p99_ms": np.percentile(lat_compile, 99),
                "tok_sec": tok_per_sec_compile,
                "vram_mb": torch.cuda.max_memory_allocated() / (1024**2)
            }
            print(f"  -> Mean Latency: {np.mean(lat_compile):.2f} ms | p50: {np.percentile(lat_compile, 50):.2f} ms | p99: {np.percentile(lat_compile, 99):.2f} ms")
            print(f"  -> Throughput  : {tok_per_sec_compile:.2f} tokens/sec ({tok_per_sec_compile/tok_per_sec_eager:.2f}x speedup vs Eager)")
        except Exception as e:
            print(f"  [!] torch.compile encountered an error: {e}")
            results["torch.compile (Inductor)"] = {"tok_sec": 0.0, "mean_ms": 0.0}

    # --------------------------------------------------------------------------
    # Benchmark 3: T4 Custom Fused Quantized Engine (W4A16 LOP3 + Mega-Kernel)
    # --------------------------------------------------------------------------
    print("\n[Engine 3/3] Benchmarking T4 Custom Fused Quantized Engine (LOP3 W4A16 + Mega-Kernel)...")
    if device == "cuda" and HAS_T4_KERNELS:
        torch.cuda.reset_peak_memory_stats()
        t4_model = FullDecoderStack(NUM_LAYERS, DIM, HIDDEN_DIM, NUM_HEADS, T4FusedDecoderLayer, group_size=128).to(device).half().eval()
        k_caches, v_caches = make_kv_caches()
        cur_token = torch.randn((1, 1, DIM), dtype=torch.float16, device=device)

        # Warmup
        with torch.no_grad():
            for pos in range(WARMUP_TOKENS):
                cur_token = t4_model(cur_token, k_caches, v_caches, pos)
            torch.cuda.synchronize()

        latencies_t4 = []
        with torch.no_grad():
            for pos in range(WARMUP_TOKENS, WARMUP_TOKENS + BENCHMARK_TOKENS):
                t0 = torch.cuda.Event(enable_timing=True)
                t1 = torch.cuda.Event(enable_timing=True)
                t0.record()
                cur_token = t4_model(cur_token, k_caches, v_caches, pos)
                t1.record()
                torch.cuda.synchronize()
                latencies_t4.append(t0.elapsed_time(t1))

        lat_t4 = np.array(latencies_t4)
        tok_per_sec_t4 = 1000.0 / np.mean(lat_t4)
        speedup_vs_eager = tok_per_sec_t4 / tok_per_sec_eager
        speedup_vs_compile = tok_per_sec_t4 / results.get("torch.compile (Inductor)", {}).get("tok_sec", 1e-5)

        results["T4 Custom Fused INT4 Engine"] = {
            "mean_ms": np.mean(lat_t4),
            "p50_ms": np.percentile(lat_t4, 50),
            "p90_ms": np.percentile(lat_t4, 90),
            "p99_ms": np.percentile(lat_t4, 99),
            "tok_sec": tok_per_sec_t4,
            "vram_mb": torch.cuda.max_memory_allocated() / (1024**2)
        }
        print(f"  -> Mean Latency: {np.mean(lat_t4):.2f} ms | p50: {np.percentile(lat_t4, 50):.2f} ms | p99: {np.percentile(lat_t4, 99):.2f} ms")
        print(f"  -> Throughput  : {tok_per_sec_t4:.2f} tokens/sec")
        print(f"  -> Speedup vs Eager     : {speedup_vs_eager:.2f}x")
        print(f"  -> Speedup vs Inductor  : {speedup_vs_compile:.2f}x")
    else:
        print("  [!] Running in mathematical estimation mode (T4 kernels not linked locally).")

    # --------------------------------------------------------------------------
    # Final Comparison Table & Gate Report
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"  FINAL SERVING ENGINE SCORECARD (TESLA T4, B=1, 24 LAYERS, D={DIM}, H={HIDDEN_DIM})")
    print("=" * 80)
    print(f"{'Execution Engine':<32} | {'Tokens/Sec':<12} | {'p50 (ms)':<10} | {'p99 (ms)':<10} | {'Speedup':<8}")
    print("-" * 80)

    eager_tok = results.get("PyTorch FP16 Eager", {}).get("tok_sec", 1.0)
    for name, data in results.items():
        if data.get("tok_sec", 0) > 0:
            speedup = data["tok_sec"] / eager_tok
            print(f"{name:<32} | {data['tok_sec']:>10.2f}  | {data['p50_ms']:>8.2f}ms | {data['p99_ms']:>8.2f}ms | {speedup:>6.2f}x")

    print("=" * 80)
    print("  [MILESTONE A EVALUATION COMPLETED]")
    print("=" * 80)


if __name__ == "__main__":
    run_serving_benchmark()
