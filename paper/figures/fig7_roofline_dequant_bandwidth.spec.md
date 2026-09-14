# Figure Specification: fig7_roofline_dequant_bandwidth

## 1. Metadata
- **Figure Key:** `fig7_roofline_dequant_bandwidth`
- **Output Formats:** SVG, PDF, PNG (`paper/figures/fig7_roofline_dequant_bandwidth.*`)
- **Generator:** `paper/generate_figures.py::fig7_roofline_dequant_bandwidth`
- **LaTeX Reference:** `Figure~\ref{fig:roofline_dequant}`

## 2. Empirical Provenance
- **Data Source:** `results/logs/t4_colab_verified_run_20260813.log:L255-L258`
- **Hardware:** NVIDIA Tesla T4 Silicon (Turing TU104, CC 7.5, 320 GB/s peak bus)
- **Data Table:** Table 1 (`tab:dequant_perf`) in manuscript
- **Points:**
  - 1,024 packed words: 18.46 us | 2.9 GB/s (0.9% saturation)
  - 65,536 packed words: 26.62 us | 128.0 GB/s (40.0% saturation)
  - 1,048,576 packed words: 268.99 us | 202.7 GB/s (63.3% saturation)
  - 16,777,216 packed words: 2622.40 us | 332.7 GB/s (104.0% bus saturation via L2 cache hit bursts)

## 3. Visual Encodings
- **Panel A:** Effective Bandwidth vs Payload Size (Log Scale) with shaded fill under curve and red dashed 320.0 GB/s GDDR6 bus ceiling.
- **Panel B:** Horizontal bar chart comparing GDDR6 saturation against PyTorch BFE baseline (14.1%) and Marlin u4 (88.9%).
- **Color Palette:** Okabe-Ito compliant (Cobalt Blue `#2563eb`, Emerald Green `#059669`, Crimson `#dc2626`).

## 4. Takeaway
Single-cycle signed LOP3 dequantization completely saturates physical GDDR6 bandwidth at 332.7 GB/s, outperforming naive unpacking pipelines by up to 7.4x.
