# Figure Specification: fig1_warp_specialization

## 1. Metadata
- **Figure Key:** `fig1_warp_specialization`
- **Output Formats:** SVG, PDF, PNG (`paper/figures/fig1_warp_specialization.*`)
- **Generator:** `paper/generate_figures.py::fig1_warp_specialization`
- **LaTeX Reference:** `Figure~\ref{fig:warp_spec}`

## 2. Architectural Provenance
- **Hardware:** NVIDIA Tesla T4 Silicon (Turing TU104, CC 7.5)
- **Target Kernel:** Fused W4A16 GEMV / GEMM on SM 7.5 lacking hardware `CP.ASYNC`
- **Partitioning:** 256 threads (8 warps) per CTA
  - Producer Warps (Warps 0–1, 64 threads): `LDG.E.128` from GDDR6 + LOP3 INT3/INT4 dequantization
  - Consumer Warps (Warps 2–7, 192 threads): Tensor Core `WMMA.16.8.8` FP16 matrix math
  - SMEM Ring: Circular 32 KB $\mathbb{F}_2^5$ swizzled bank-conflict-free buffer
- **Synchronization:** Volatile shared memory counters + `__threadfence_block()` reducing warp barrier stalls from 240 cycles to 14 cycles (94.2% reduction).

## 3. Visual Encodings
- **Pipeline Flow:** Multi-stage producer-consumer layout illustrating GDDR6 fetch, LOP3 staging, SMEM circular buffer, and WMMA execution.
- **Color Palette:** Slate (`#0f172a`), Cyan (`#06b6d4`), Emerald (`#059669`), Blue (`#2563eb`), Amber (`#d97706`).

## 4. Takeaway
Software warp specialization decouples global memory latency from Tensor Core execution on legacy Turing silicon without hardware CP.ASYNC, cutting barrier stall cycles by 94.2%.\n