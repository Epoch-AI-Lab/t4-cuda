# Figure Specification: fig3_thermal_dvfs_pacing

## 1. Metadata
- **Figure Key:** `fig3_thermal_dvfs_pacing`
- **Output Formats:** SVG, PDF, PNG (`paper/figures/fig3_thermal_dvfs_pacing.*`)
- **Generator:** `paper/generate_figures.py::fig3_thermal_dvfs_pacing`
- **LaTeX Reference:** `Figure~\ref{fig:thermal_pacing}`

## 2. Control-Loop Provenance
- **Hardware:** Physical 70W Passive Tesla T4 GPU in 1U server chassis
- **Mechanism:** Closed-loop active block scheduling vs unpaced greedy dispatch
- **Thresholds:**
  - 100% Unpaced Occupancy: Rapid junction thermal ramp ($78^\circ$C), triggering `SW_POWER_CAP` throttle (`0x0004`) dropping SM clock from 1590 MHz to 975 MHz (-38.7%).
  - 25% Paced Occupancy: Steady-state thermal equilibrium ($62^\circ$C) maintaining 1590 MHz locked boost clock indefinitely.

## 3. Visual Encodings
- **Panel A:** Unpaced vs Paced SM core clock frequency trajectory over time.
- **Panel B:** Die temperature trajectory and power capping threshold ($70$W limit).
- **Color Palette:** Crimson (`#ef4444`) for unpaced thermal throttling, Emerald (`#059669`) for paced thermal stability.

## 4. Takeaway
Capping active SM occupancy to 25% on legacy 70W passive T4 GPUs stabilizes core clocks at 1590 MHz, preventing thermal throttle collapse and delivering 18.4% higher net sustained throughput.\n