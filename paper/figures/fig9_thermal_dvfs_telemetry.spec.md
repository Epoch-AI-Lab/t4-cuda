# Figure Specification: fig9_thermal_dvfs_telemetry

## 1. Metadata
- **Figure Key:** `fig9_thermal_dvfs_telemetry`
- **Output Formats:** SVG, PDF, PNG (`paper/figures/fig9_thermal_dvfs_telemetry.*`)
- **Generator:** `paper/generate_figures.py::fig9_thermal_dvfs_telemetry`
- **LaTeX Reference:** `Figure~\ref{fig:thermal_telemetry}`

## 2. Empirical Provenance
- **Data Source:** `results/logs/t4_colab_verified_run_20260813.log:L78-L84` and Table 2 (`tab:telemetry`)
- **Hardware:** Physical 70W passively cooled Tesla T4 GPU
- **Telemetry Points:**
  - Unconstrained (100% Occ / 32 warps/SM): Peak Power 93.61 W (> 70W TDP), 72.5% runtime throttled via NVPM `SW_POWER_CAP` (0x0004), Mean Clock 1193 MHz (dropped to 950 MHz).
  - Paced (25% Occ / 8 warps/SM): Peak Power 50.36 W (28% headroom below 70W cap), 0.0% throttled, Locked Boost Clock 1590 MHz (1.07x net faster execution).

## 3. Visual Encodings
- **Panel A:** SM Core Clock Stability (MHz) time series across 450M GPU execution cycles comparing locked 1590 MHz vs collapsed 1193 MHz mean.
- **Panel B:** Dynamic GPU Power Draw (Watts) time series with dashed 70.0W passive TDP ceiling.
- **Color Palette:** Forest Green (`#059669`) for paced regime, Crimson (`#dc2626`) for throttled regime.

## 4. Takeaway
Counter-intuitively, restricting warp occupancy to 25% prevents thermal throttling on 70W passively cooled hardware, locking the 1590 MHz boost clock and yielding a 1.07x speedup over 100% occupancy.
