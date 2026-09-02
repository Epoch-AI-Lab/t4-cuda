# Implementation Research: W4A16 WMMA Tensor Core GEMM on Turing (sm_75)

## The Task
Implement a batched W4A16 GEMM kernel targeting the NVIDIA Turing (sm_75) architecture using `nvcuda::wmma` / `mma.sync` Tensor Core instructions. Weights are stored in INT4 (4-bit per weight, packed as 8 nibbles per `uint32_t`) with per-group scales/zero-points (group=128), while activations and outputs are FP16. The kernel must hide the dequantization overhead behind Tensor Core math and outperform cuBLAS FP16 at batched rollout shapes ($M \in [16, 64]$, $K=896, N=896$ and $N=4864$).

## Gate Protocol (kernel-research)

- **GATE 1: Bottleneck — PASS**
  - Proven by profile in `benchmarks/profile_grpo_step.py` and `benchmarks/benchmark_grpo_int4.py`:
  - Rollout generation accounts for 85.4% to 88.8% of step time.
  - At batch $M=64$, scalar GEMV takes 19.53s/step while cuBLAS FP16 takes 13.03s/step because scalar CUDA cores hit the 8.1 TFLOPS compute ceiling on Turing.
- **GATE 2: Roofline — PASS**
  - At $M=64, K=896, N=896$: Arithmetic Intensity (AI) = 200 FLOP/byte.
  - Ridge point of T4 Tensor Cores = 216 FLOP/byte (65 TFLOPS / 300 GB/s).
  - Workload sits directly on the Tensor Core ridge point.
  - Headroom from 8.1 TFLOPS scalar to 65 TFLOPS Tensor Cores is up to 8.0x in compute.
- **GATE 3: Hand-write vs compile — PASS**
  - PyTorch / cuBLAS does not provide a fused W4A16 Tensor Core GEMM for sm_75.
  - Marlin exclusively requires Ampere+ (`cp.async`, sm_80+).
  - Writing a bespoke sm_75 WMMA kernel is strictly justified.
- **GATE 4: Design — PASS**
  - Tile size: $M=16, N=16, K=16$ (native sm_75 WMMA fragment).
  - Warp-level decomposition: Each block of 128 or 256 threads computes a tile of $C$ (e.g. $64 \times 64$).
  - Zero-SMem dequantization: 4-bit weights are loaded into registers and unpacked via PTX `lop3_unpack_u4_ptx` into `half2` fragments directly, bypassing shared memory write-backs.
- **GATE 5: Literature — PASS**
  - Marlin (Frantar et al., 2024, arXiv:2408.11743): Demonstrates register-level unpacking overlapping with matrix multiplies.
  - QServe (Lin et al., 2024, arXiv:2405.04532): Identifies dequantization overhead on scalar cores as the primary barrier to batched INT4 speedup.
  - AWQ (Lin et al., 2023, arXiv:2306.00978): Proves W4A16 preserves accuracy when activation-saliency is respected.

## 1. Gotchas & Architecture Details on Turing (sm_75)
- **WMMA Fragment Layout on sm_75**:
  - `wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major>`:
    A warp of 32 threads holds a $16 \times 16$ tile of Matrix B.
    Total elements = 256 `half`s.
    Each thread holds exactly 8 `half` elements in its `x` array (`fragment.x[0..7]`).
  - In 4-bit, 8 elements require exactly 32 bits (1 `uint32_t`)!
  - Therefore, each thread loads 1 `uint32_t` from memory, unpacks it into 8 `half` values, and populates `frag_b.x[0..7]`.
- **Bank Conflicts in Shared Memory for Matrix A**:
  - Matrix A ($M \times K$) is shared across the warps in a block.
  - Storing Matrix A in shared memory requires padding (`K + 8` or `16 + 8`) to avoid 32-bank conflicts during `wmma::load_matrix_sync`.
- **Scale and Zero-Point Application**:
  - With `group_size = 128`, each group spans 8 WMMA steps along K ($8 \times 16 = 128$).
  - Dequantization formula: $(q - z) \cdot s$.
  - Scales and zero points can be applied either to the unpacked weight fragment `B` before `mma_sync`, or to the accumulator `C` in the epilogue if scales are constant across K. With per-group (group=128), scales change every 8 steps along K, so scaling the weight fragment `frag_b` directly in FP16 registers is the cleanest and most accurate path.

## 2. Best Practices
- **Double Buffering / Pipelining**: Prefetch the next tile of A from global memory into shared memory while the Tensor Cores compute the current tile.
- **PTX LOP3 Unpacking**: Use the tested 0xEA LOP3 instruction (`lop3_unpack_u4_ptx`) to unpack 2 nibbles at a time in 1 cycle without branching or masking.
- **Preserve Single-Token Fast-Path**: Retain our existing `fused_w4a16_gemv_u4_kernel` for $M \le 4$ and dispatch to the new WMMA kernel when $M > 4$.

## 3. Verification Plan
- Phase 1: Standalone micro-benchmark testing $M \in [8, 16, 32, 64]$, $K=896, N=896$ and $N=4864$ against cuBLAS `torch.matmul(A, W_fp16)`.
- Phase 2: Numerical correctness verification against Python / CPU reference (maximum absolute and relative error $\le 0.1\%$).
- Phase 3: Integration into `benchmarks/benchmark_grpo_int4.py` and step time measurement.
