# Figure Specification: fig2_lop3_dequant_pipeline

## 1. Metadata
- **Figure Key:** `fig2_lop3_dequant_pipeline`
- **Output Formats:** SVG, PDF, PNG (`paper/figures/fig2_lop3_dequant_pipeline.*`)
- **Generator:** `paper/generate_figures.py::fig2_lop3_dequant_pipeline`
- **LaTeX Reference:** `Figure~\ref{fig:lop3_pipeline}`

## 2. Microarchitectural Provenance
- **Hardware:** NVIDIA Tesla T4 Silicon (Turing TU104, CC 7.5, LOP3 instruction set)
- **Instruction:** Single-cycle `LOP3.LUT` with truth table `0x6A` (`(A & B) | C`)
- **Datapath:**
  - Mask: `0x000F000F` isolating signed nibbles `b0, b1`
  - Magic Exponent: `0x64006400` inserting IEEE 754 FP16 biased exponent (bias 15 + 10 = 25)
  - FMA Sub-Scale: fused `(x - 1024.0) * scale`
- **Zero Spill:** Reconstructs FP16 values in-register without temporary shared memory spills or arithmetic shift bottlenecks.

## 3. Visual Encodings
- **Bit-Field Diagram:** Low and high nibble bit packing (`b0, b1`), bitwise LOP3 insertion into FP16 format, and subsequent FMA scale step.
- **Color Palette:** Dark Slate (`#0f172a`), Cyan (`#0284c7`), Purple (`#7c3aed`), Emerald (`#059669`).

## 4. Takeaway
Single-cycle LOP3 bit-manipulation directly populates IEEE 754 floating-point exponent fields, performing signed sub-byte dequantization with zero register spills.\n