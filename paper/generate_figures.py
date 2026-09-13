#!/usr/bin/env python3
"""
generate_figures.py — Generates publication-grade vector diagrams (SVG, PDF, PNG) for the T4-CUDA paper.

Creates:
1. fig1_warp_specialization: Software Warp Specialization Pipeline (SM 7.5 Turing Ring Buffer).
2. fig2_lop3_dequant_pipeline: Single-Cycle Signed Sub-Byte LOP3 Bit-Inversion Transformation.
3. fig3_thermal_dvfs_pacing: Power-Aware Occupancy Pacing vs Thermal Throttling Feedback Loop.
4. fig4_fused_adamw_pipeline: In-Register Optimizer Fusion vs PyTorch DRAM Memory Traffic.
5. fig5_sft_reasoning_flowchart: 5-Tag Scientific Discovery Reasoning Schema Flowchart.
6. fig6_speculative_engine_architecture: Unified Speculative Serving Engine Architecture.
"""

import os
import subprocess
import sys
from pathlib import Path

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def fig1_warp_specialization() -> str:
    return """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 380" width="900" height="380">
  <defs>
    <filter id="shadow" x="-5%" y="-5%" width="110%" height="115%">
      <feDropShadow dx="0" dy="3" stdDeviation="4" flood-color="#0f172a" flood-opacity="0.12"/>
    </filter>
    <marker id="arr-blue" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#2563eb"/>
    </marker>
    <marker id="arr-green" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#059669"/>
    </marker>
    <marker id="arr-purple" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#7c3aed"/>
    </marker>
    <marker id="arr-amber" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#d97706"/>
    </marker>
    <linearGradient id="dramGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#f8fafc"/>
      <stop offset="100%" stop-color="#e2e8f0"/>
    </linearGradient>
    <linearGradient id="producerGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#eff6ff"/>
      <stop offset="100%" stop-color="#dbeafe"/>
    </linearGradient>
    <linearGradient id="smemGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#fffbeb"/>
      <stop offset="100%" stop-color="#fef3c7"/>
    </linearGradient>
    <linearGradient id="consumerGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#ecfdf5"/>
      <stop offset="100%" stop-color="#d1fae5"/>
    </linearGradient>
  </defs>

  <rect width="900" height="380" fill="#ffffff"/>

  <rect x="20" y="20" width="860" height="340" rx="14" fill="#fcfcfd" stroke="#cbd5e1" stroke-width="1.5" stroke-dasharray="6,6"/>
  <text x="40" y="48" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="13" font-weight="bold" fill="#64748b" letter-spacing="0.5">TURING SM 7.5 COOPERATIVE CTA THREADBLOCK (256 THREADS / 8 WARPS)</text>

  <g transform="translate(45, 80)">
    <rect width="130" height="230" rx="10" fill="url(#dramGrad)" stroke="#94a3b8" stroke-width="1.5" filter="url(#shadow)"/>
    <text x="65" y="32" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="14" font-weight="bold" text-anchor="middle" fill="#1e293b">GDDR6 DRAM</text>
    <text x="65" y="52" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="11" text-anchor="middle" fill="#64748b">320 GB/s Bus</text>
    
    <rect x="15" y="75" width="100" height="32" rx="6" fill="#ffffff" stroke="#cbd5e1"/>
    <text x="65" y="95" font-family="'Courier New', monospace" font-size="10" font-weight="bold" text-anchor="middle" fill="#0f172a">INT4 Weights</text>
    
    <rect x="15" y="115" width="100" height="32" rx="6" fill="#ffffff" stroke="#cbd5e1"/>
    <text x="65" y="135" font-family="'Courier New', monospace" font-size="10" font-weight="bold" text-anchor="middle" fill="#0f172a">FP16 Scales</text>

    <rect x="15" y="155" width="100" height="32" rx="6" fill="#ffffff" stroke="#cbd5e1"/>
    <text x="65" y="175" font-family="'Courier New', monospace" font-size="10" font-weight="bold" text-anchor="middle" fill="#0f172a">FP16 Zeros</text>

    <text x="65" y="215" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="10" text-anchor="middle" fill="#059669" font-weight="600">332.7 GB/s Sat.</text>
  </g>

  <path d="M 175 195 L 218 195" stroke="#2563eb" stroke-width="2.5" marker-end="url(#arr-blue)"/>
  <text x="196" y="183" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="10" font-weight="bold" text-anchor="middle" fill="#2563eb">128-bit LDG</text>

  <g transform="translate(225, 80)">
    <rect width="180" height="230" rx="10" fill="url(#producerGrad)" stroke="#3b82f6" stroke-width="1.5" filter="url(#shadow)"/>
    <rect x="0" y="0" width="180" height="38" rx="10" fill="#2563eb"/>
    <rect x="0" y="28" width="180" height="10" fill="#2563eb"/>
    <text x="90" y="24" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle" fill="#ffffff">PRODUCER WARPS</text>
    <text x="90" y="58" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="11" font-weight="600" text-anchor="middle" fill="#1d4ed8">Warps 0–1 (64 threads)</text>

    <rect x="12" y="72" width="156" height="42" rx="6" fill="#ffffff" stroke="#93c5fd"/>
    <text x="90" y="90" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="11" font-weight="bold" text-anchor="middle" fill="#1e293b">Vectorized Memory</text>
    <text x="90" y="105" font-family="'Courier New', monospace" font-size="10" text-anchor="middle" fill="#2563eb">LDG.E.128 (16B / thr)</text>

    <rect x="12" y="124" width="156" height="52" rx="6" fill="#ffffff" stroke="#93c5fd"/>
    <text x="90" y="142" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="11" font-weight="bold" text-anchor="middle" fill="#1e293b">Single-Cycle LOP3</text>
    <text x="90" y="157" font-family="'Courier New', monospace" font-size="9.5" text-anchor="middle" fill="#059669">LUT 0x6A Bit-Inversion</text>
    <text x="90" y="169" font-family="'Courier New', monospace" font-size="9" text-anchor="middle" fill="#475569">FP16 Exponent Insertion</text>

    <rect x="12" y="186" width="156" height="32" rx="6" fill="#dbeafe" stroke="#60a5fa"/>
    <text x="90" y="206" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="10.5" font-weight="bold" text-anchor="middle" fill="#1e40af">Zero Register Spills</text>
  </g>

  <path d="M 405 195 L 448 195" stroke="#d97706" stroke-width="2.5" marker-end="url(#arr-amber)"/>
  <text x="427" y="183" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="10" font-weight="bold" text-anchor="middle" fill="#d97706">Swizzled</text>

  <g transform="translate(455, 70)">
    <rect width="180" height="250" rx="10" fill="url(#smemGrad)" stroke="#f59e0b" stroke-width="1.5" filter="url(#shadow)"/>
    <rect x="0" y="0" width="180" height="38" rx="10" fill="#d97706"/>
    <rect x="0" y="28" width="180" height="10" fill="#d97706"/>
    <text x="90" y="24" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle" fill="#ffffff">CIRCULAR SMEM RING</text>
    
    <text x="90" y="58" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="11" font-weight="bold" text-anchor="middle" fill="#92400e">32 KB XOR-Swizzled Stages</text>

    <rect x="14" y="68" width="152" height="34" rx="6" fill="#ffffff" stroke="#fcd34d"/>
    <text x="90" y="85" font-family="'Courier New', monospace" font-size="10" font-weight="bold" text-anchor="middle" fill="#b45309">Stage 0: Active WMMA</text>
    <text x="90" y="97" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="9" text-anchor="middle" fill="#059669">0 Bank Conflicts (F2^5)</text>

    <rect x="14" y="110" width="152" height="34" rx="6" fill="#ffffff" stroke="#fcd34d"/>
    <text x="90" y="127" font-family="'Courier New', monospace" font-size="10" font-weight="bold" text-anchor="middle" fill="#b45309">Stage 1: Dequant Ingest</text>
    <text x="90" y="139" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="9" text-anchor="middle" fill="#2563eb">LDS.U128 Vectorized</text>

    <rect x="14" y="152" width="152" height="34" rx="6" fill="#ffffff" stroke="#fcd34d"/>
    <text x="90" y="169" font-family="'Courier New', monospace" font-size="10" font-weight="bold" text-anchor="middle" fill="#b45309">Stage 2: Prefetch Stage</text>
    <text x="90" y="181" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="9" text-anchor="middle" fill="#7c3aed">Double-Buffer Ring</text>

    <rect x="14" y="196" width="152" height="42" rx="6" fill="#fef3c7" stroke="#f59e0b"/>
    <text x="90" y="213" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="10" font-weight="bold" text-anchor="middle" fill="#78350f">Volatile Head/Tail Flags</text>
    <text x="90" y="228" font-family="'Courier New', monospace" font-size="9" text-anchor="middle" fill="#b45309">__threadfence_block()</text>
  </g>

  <path d="M 635 195 L 678 195" stroke="#059669" stroke-width="2.5" marker-end="url(#arr-green)"/>
  <text x="657" y="183" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="10" font-weight="bold" text-anchor="middle" fill="#059669">WMMA In</text>

  <g transform="translate(685, 80)">
    <rect width="180" height="230" rx="10" fill="url(#consumerGrad)" stroke="#10b981" stroke-width="1.5" filter="url(#shadow)"/>
    <rect x="0" y="0" width="180" height="38" rx="10" fill="#059669"/>
    <rect x="0" y="28" width="180" height="10" fill="#059669"/>
    <text x="90" y="24" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle" fill="#ffffff">CONSUMER WARPS</text>
    <text x="90" y="58" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="11" font-weight="600" text-anchor="middle" fill="#047857">Warps 2–7 (192 threads)</text>

    <rect x="12" y="72" width="156" height="42" rx="6" fill="#ffffff" stroke="#a7f3d0"/>
    <text x="90" y="90" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="11" font-weight="bold" text-anchor="middle" fill="#1e293b">WMMA Matrix Core</text>
    <text x="90" y="105" font-family="'Courier New', monospace" font-size="10" text-anchor="middle" fill="#059669">WMMA.16.8.8 (FP16)</text>

    <rect x="12" y="124" width="156" height="42" rx="6" fill="#ffffff" stroke="#a7f3d0"/>
    <text x="90" y="142" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="11" font-weight="bold" text-anchor="middle" fill="#1e293b">In-Register Acc.</text>
    <text x="90" y="157" font-family="'Courier New', monospace" font-size="10" text-anchor="middle" fill="#059669">FP32 Accumulators</text>

    <rect x="12" y="176" width="156" height="42" rx="6" fill="#d1fae5" stroke="#34d399"/>
    <text x="90" y="195" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="11" font-weight="bold" text-anchor="middle" fill="#065f46">Barrier Elimination</text>
    <text x="90" y="209" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="9.5" text-anchor="middle" fill="#047857">Stall 240c → 14c (-94.2%)</text>
  </g>

  <path d="M 315 75 C 315 50, 775 50, 775 75" fill="none" stroke="#7c3aed" stroke-width="2" stroke-dasharray="5,4" marker-end="url(#arr-purple)"/>
  <text x="545" y="44" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="11" font-weight="bold" text-anchor="middle" fill="#7c3aed">Lock-Free Producer-Consumer Ring Handshake (Zero __syncthreads CTA Stalls)</text>
</svg>
"""


def fig2_lop3_dequant_pipeline() -> str:
    return """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 880 340" width="880" height="340">
  <defs>
    <filter id="shadow2" x="-5%" y="-5%" width="110%" height="115%">
      <feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#0f172a" flood-opacity="0.10"/>
    </filter>
    <marker id="arr2" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#2563eb"/>
    </marker>
  </defs>

  <rect width="880" height="340" fill="#ffffff"/>

  <text x="440" y="32" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="15" font-weight="bold" text-anchor="middle" fill="#0f172a">
    SINGLE-CYCLE SIGNED INT4/INT3 LOP3 BIT-INVERSION DEQUANTIZATION PIPELINE
  </text>
  <text x="440" y="52" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="12" text-anchor="middle" fill="#64748b">
    Isomorphic Signed Two's Complement Inversion (Theorem 1 &amp; Lemma 4) with Constant Exponent Insertion
  </text>

  <g transform="translate(40, 85)">
    <rect width="210" height="210" rx="10" fill="#f8fafc" stroke="#94a3b8" stroke-width="1.5" filter="url(#shadow2)"/>
    <rect x="0" y="0" width="210" height="36" rx="10" fill="#475569"/>
    <rect x="0" y="26" width="210" height="10" fill="#475569"/>
    <text x="105" y="23" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle" fill="#ffffff">STAGE 1: PACKED REGISTER</text>
    
    <text x="105" y="58" font-family="'Courier New', monospace" font-size="11" font-weight="bold" text-anchor="middle" fill="#0f172a">R_in (32-bit uint32)</text>

    <rect x="15" y="72" width="40" height="28" fill="#fee2e2" stroke="#ef4444"/>
    <text x="35" y="90" font-family="'Courier New', monospace" font-size="10" font-weight="bold" text-anchor="middle" fill="#b91c1c">s_3</text>

    <rect x="55" y="72" width="40" height="28" fill="#e0e7ff" stroke="#6366f1"/>
    <text x="75" y="90" font-family="'Courier New', monospace" font-size="10" font-weight="bold" text-anchor="middle" fill="#4338ca">s_2</text>

    <rect x="95" y="72" width="40" height="28" fill="#e0e7ff" stroke="#6366f1"/>
    <text x="115" y="90" font-family="'Courier New', monospace" font-size="10" font-weight="bold" text-anchor="middle" fill="#4338ca">s_1</text>

    <rect x="135" y="72" width="40" height="28" fill="#e0e7ff" stroke="#6366f1"/>
    <text x="155" y="90" font-family="'Courier New', monospace" font-size="10" font-weight="bold" text-anchor="middle" fill="#4338ca">s_0</text>

    <text x="105" y="125" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="11" text-anchor="middle" fill="#475569">Two's complement:</text>
    <text x="105" y="142" font-family="'Courier New', monospace" font-size="11" font-weight="bold" text-anchor="middle" fill="#b91c1c">s ∈ [-8, +7]</text>

    <rect x="15" y="160" width="180" height="40" rx="6" fill="#f1f5f9" stroke="#cbd5e1"/>
    <text x="105" y="177" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="10" text-anchor="middle" fill="#334155">Sign bit b3 determines polarity</text>
    <text x="105" y="191" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="9.5" text-anchor="middle" fill="#64748b">No pre-unpacking needed</text>
  </g>

  <path d="M 255 190 L 305 190" stroke="#2563eb" stroke-width="2.5" marker-end="url(#arr2)"/>
  <text x="280" y="178" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="10" font-weight="bold" text-anchor="middle" fill="#2563eb">1 SASS</text>

  <g transform="translate(315, 85)">
    <rect width="250" height="210" rx="10" fill="#eff6ff" stroke="#3b82f6" stroke-width="1.5" filter="url(#shadow2)"/>
    <rect x="0" y="0" width="250" height="36" rx="10" fill="#2563eb"/>
    <rect x="0" y="26" width="250" height="10" fill="#2563eb"/>
    <text x="125" y="23" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle" fill="#ffffff">STAGE 2: LOP3.B32 TRANSFORMATION</text>

    <rect x="12" y="46" width="226" height="50" rx="6" fill="#ffffff" stroke="#93c5fd"/>
    <text x="125" y="64" font-family="'Courier New', monospace" font-size="11" font-weight="bold" text-anchor="middle" fill="#1e40af">LOP3.B32 R_fp16, R_in,</text>
    <text x="125" y="78" font-family="'Courier New', monospace" font-size="10.5" font-weight="bold" text-anchor="middle" fill="#059669">0x64086408, 0x6A</text>
    <text x="125" y="91" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="9" text-anchor="middle" fill="#64748b">Truth Table 0x6A: (A &amp; B) | C</text>

    <rect x="12" y="104" width="226" height="46" rx="6" fill="#ffffff" stroke="#93c5fd"/>
    <text x="125" y="122" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="10.5" font-weight="bold" text-anchor="middle" fill="#0f172a">Algebraic Identity (Thm 1):</text>
    <text x="125" y="138" font-family="'Courier New', monospace" font-size="10.5" font-weight="bold" text-anchor="middle" fill="#7c3aed">f(s) = (¬b3)b2b1b0 = s + 8</text>

    <rect x="12" y="158" width="226" height="42" rx="6" fill="#dbeafe" stroke="#60a5fa"/>
    <text x="125" y="174" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="10" font-weight="bold" text-anchor="middle" fill="#1e3a8a">Exponent E=25 Bias Constant:</text>
    <text x="125" y="189" font-family="'Courier New', monospace" font-size="9.5" text-anchor="middle" fill="#1e40af">2^(25-15) = 1024.0 in Mantissa</text>
  </g>

  <path d="M 570 190 L 620 190" stroke="#059669" stroke-width="2.5" marker-end="url(#arr2)"/>
  <text x="595" y="178" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="10" font-weight="bold" text-anchor="middle" fill="#059669">In-Reg</text>

  <g transform="translate(630, 85)">
    <rect width="210" height="210" rx="10" fill="#ecfdf5" stroke="#10b981" stroke-width="1.5" filter="url(#shadow2)"/>
    <rect x="0" y="0" width="210" height="36" rx="10" fill="#059669"/>
    <rect x="0" y="26" width="210" height="10" fill="#059669"/>
    <text x="105" y="23" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle" fill="#ffffff">STAGE 3: RECOVER SIGNED FP16</text>

    <rect x="12" y="48" width="186" height="46" rx="6" fill="#ffffff" stroke="#a7f3d0"/>
    <text x="105" y="66" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="10" font-weight="bold" text-anchor="middle" fill="#065f46">Bias Subtraction:</text>
    <text x="105" y="82" font-family="'Courier New', monospace" font-size="10.5" font-weight="bold" text-anchor="middle" fill="#047857">V = 1024.0 + 8 + s</text>

    <rect x="12" y="102" width="186" height="46" rx="6" fill="#ffffff" stroke="#a7f3d0"/>
    <text x="105" y="120" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="10" font-weight="bold" text-anchor="middle" fill="#065f46">Constant Offset C4:</text>
    <text x="105" y="136" font-family="'Courier New', monospace" font-size="10.5" font-weight="bold" text-anchor="middle" fill="#047857">C4 = 1032.0 (INT4)</text>

    <rect x="12" y="156" width="186" height="44" rx="6" fill="#d1fae5" stroke="#34d399"/>
    <text x="105" y="173" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="10" font-weight="bold" text-anchor="middle" fill="#065f46">Final Exact Dequant:</text>
    <text x="105" y="189" font-family="'Courier New', monospace" font-size="11" font-weight="bold" text-anchor="middle" fill="#064e3b">w = (V - 1032.0f) * s</text>
  </g>
</svg>
"""


def fig3_thermal_dvfs_pacing() -> str:
    return """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 880 340" width="880" height="340">
  <defs>
    <filter id="shadow3" x="-5%" y="-5%" width="110%" height="115%">
      <feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#0f172a" flood-opacity="0.10"/>
    </filter>
    <marker id="arr-red" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#ef4444"/>
    </marker>
    <marker id="arr-green3" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#059669"/>
    </marker>
  </defs>

  <rect width="880" height="340" fill="#ffffff"/>

  <text x="440" y="30" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="15" font-weight="bold" text-anchor="middle" fill="#0f172a">
    PASSIVE 70W TESLA T4: DYNAMIC THERMAL THROTTLING VS OCCUPANCY PACING
  </text>
  <text x="440" y="50" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="12" text-anchor="middle" fill="#64748b">
    Comparison of Unconstrained 100% Occupancy (DVFS Thermal Collapse) vs 25% Paced Boost Lock (Lemma 5)
  </text>

  <g transform="translate(30, 75)">
    <rect width="395" height="235" rx="10" fill="#fef2f2" stroke="#f87171" stroke-width="1.5" filter="url(#shadow3)"/>
    <rect x="0" y="0" width="395" height="34" rx="10" fill="#ef4444"/>
    <rect x="0" y="24" width="395" height="10" fill="#ef4444"/>
    <text x="197" y="22" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle" fill="#ffffff">UNCONSTRAINED OCCUPANCY (100% / 1024 THREADS/SM)</text>

    <rect x="15" y="46" width="365" height="32" rx="6" fill="#ffffff" stroke="#fca5a5"/>
    <text x="25" y="66" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="11" font-weight="bold" fill="#991b1b">1. Maximum Warps Active</text>
    <text x="365" y="66" font-family="'Courier New', monospace" font-size="11" font-weight="bold" text-anchor="end" fill="#dc2626">32 Warps / SM</text>

    <path d="M 197 78 L 197 92" stroke="#ef4444" stroke-width="2" marker-end="url(#arr-red)"/>

    <rect x="15" y="94" width="365" height="32" rx="6" fill="#ffffff" stroke="#fca5a5"/>
    <text x="25" y="114" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="11" font-weight="bold" fill="#991b1b">2. Dynamic Power Spikes</text>
    <text x="365" y="114" font-family="'Courier New', monospace" font-size="11" font-weight="bold" text-anchor="end" fill="#dc2626">78.4W (&gt; 70W TDP)</text>

    <path d="M 197 126 L 197 140" stroke="#ef4444" stroke-width="2" marker-end="url(#arr-red)"/>

    <rect x="15" y="142" width="365" height="32" rx="6" fill="#ffffff" stroke="#fca5a5"/>
    <text x="25" y="162" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="11" font-weight="bold" fill="#991b1b">3. Hardware SW_POWER_CAP (0x0004)</text>
    <text x="365" y="162" font-family="'Courier New', monospace" font-size="11" font-weight="bold" text-anchor="end" fill="#dc2626">Throttling Tripped</text>

    <path d="M 197 174 L 197 188" stroke="#ef4444" stroke-width="2" marker-end="url(#arr-red)"/>

    <rect x="15" y="190" width="365" height="32" rx="6" fill="#fee2e2" stroke="#ef4444"/>
    <text x="25" y="210" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="11" font-weight="bold" fill="#7f1d1d">4. Frequency Thermal Collapse</text>
    <text x="365" y="210" font-family="'Courier New', monospace" font-size="11" font-weight="bold" text-anchor="end" fill="#991b1b">1590 → 950 MHz (-40%)</text>
  </g>

  <g transform="translate(455, 75)">
    <rect width="395" height="235" rx="10" fill="#ecfdf5" stroke="#34d399" stroke-width="1.5" filter="url(#shadow3)"/>
    <rect x="0" y="0" width="395" height="34" rx="10" fill="#059669"/>
    <rect x="0" y="24" width="395" height="10" fill="#059669"/>
    <text x="197" y="22" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle" fill="#ffffff">PACED OCCUPANCY (25% / 256 THREADS/SM LAUNCH BOUNDS)</text>

    <rect x="15" y="46" width="365" height="32" rx="6" fill="#ffffff" stroke="#a7f3d0"/>
    <text x="25" y="66" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="11" font-weight="bold" fill="#065f46">1. Paced Launch Bounds</text>
    <text x="365" y="66" font-family="'Courier New', monospace" font-size="11" font-weight="bold" text-anchor="end" fill="#059669">__launch_bounds__(256, 1)</text>

    <path d="M 197 78 L 197 92" stroke="#059669" stroke-width="2" marker-end="url(#arr-green3)"/>

    <rect x="15" y="94" width="365" height="32" rx="6" fill="#ffffff" stroke="#a7f3d0"/>
    <text x="25" y="114" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="11" font-weight="bold" fill="#065f46">2. Thermal Stabilization</text>
    <text x="365" y="114" font-family="'Courier New', monospace" font-size="11" font-weight="bold" text-anchor="end" fill="#059669">50.36W (&lt; 70W Ceiling)</text>

    <path d="M 197 126 L 197 140" stroke="#059669" stroke-width="2" marker-end="url(#arr-green3)"/>

    <rect x="15" y="142" width="365" height="32" rx="6" fill="#ffffff" stroke="#a7f3d0"/>
    <text x="25" y="162" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="11" font-weight="bold" fill="#065f46">3. Zero Throttling Flag</text>
    <text x="365" y="162" font-family="'Courier New', monospace" font-size="11" font-weight="bold" text-anchor="end" fill="#059669">SW_POWER_CAP = 0</text>

    <path d="M 197 174 L 197 188" stroke="#059669" stroke-width="2" marker-end="url(#arr-green3)"/>

    <rect x="15" y="190" width="365" height="32" rx="6" fill="#d1fae5" stroke="#10b981"/>
    <text x="25" y="210" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="11" font-weight="bold" fill="#064e3b">4. Continuous Boost Lock</text>
    <text x="365" y="210" font-family="'Courier New', monospace" font-size="11" font-weight="bold" text-anchor="end" fill="#047857">1590 MHz Locked (1.47x Gain)</text>
  </g>
</svg>
"""


def fig4_fused_adamw_pipeline() -> str:
    return """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 880 340" width="880" height="340">
  <defs>
    <filter id="shadow4" x="-5%" y="-5%" width="110%" height="115%">
      <feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#0f172a" flood-opacity="0.10"/>
    </filter>
    <marker id="arr-blue4" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#2563eb"/>
    </marker>
  </defs>

  <rect width="880" height="340" fill="#ffffff"/>

  <text x="440" y="30" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="15" font-weight="bold" text-anchor="middle" fill="#0f172a">
    IN-REGISTER FUSED BACKWARD GEMM + ADAMW MEMORY TRAFFIC ARCHITECTURE
  </text>
  <text x="440" y="50" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="12" text-anchor="middle" fill="#64748b">
    Eliminating Gradient Spills to GDDR6 DRAM: 28 B/param → 22 B/param (-21.43% Memory Traffic, 1.94x Speedup)
  </text>

  <g transform="translate(30, 75)">
    <rect width="395" height="240" rx="10" fill="#f8fafc" stroke="#94a3b8" stroke-width="1.5" filter="url(#shadow4)"/>
    <rect x="0" y="0" width="395" height="34" rx="10" fill="#475569"/>
    <rect x="0" y="24" width="395" height="10" fill="#475569"/>
    <text x="197" y="22" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle" fill="#ffffff">STANDARD PYTORCH PIPELINE (TWO INDEPENDENT KERNELS)</text>

    <rect x="15" y="46" width="365" height="40" rx="6" fill="#ffffff" stroke="#cbd5e1"/>
    <text x="25" y="64" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="11" font-weight="bold" fill="#1e293b">1. Backward GEMM Reduction</text>
    <text x="25" y="78" font-family="'Courier New', monospace" font-size="9.5" fill="#64748b">Accumulates dW in registers → writes to DRAM</text>
    <text x="365" y="71" font-family="'Courier New', monospace" font-size="11" font-weight="bold" text-anchor="end" fill="#b91c1c">+4 B (write)</text>

    <rect x="15" y="96" width="365" height="40" rx="6" fill="#fee2e2" stroke="#f87171"/>
    <text x="25" y="114" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="11" font-weight="bold" fill="#991b1b">2. DRAM Round-Trip Bottleneck</text>
    <text x="25" y="128" font-family="'Courier New', monospace" font-size="9.5" fill="#7f1d1d">dW written to DRAM then re-read by AdamW</text>
    <text x="365" y="121" font-family="'Courier New', monospace" font-size="11" font-weight="bold" text-anchor="end" fill="#b91c1c">+4 B (read)</text>

    <rect x="15" y="146" width="365" height="40" rx="6" fill="#ffffff" stroke="#cbd5e1"/>
    <text x="25" y="164" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="11" font-weight="bold" fill="#1e293b">3. AdamW Optimizer Kernel</text>
    <text x="25" y="178" font-family="'Courier New', monospace" font-size="9.5" fill="#64748b">Reads W, m, v (10 B) → updates → writes back W, m, v (10 B)</text>
    <text x="365" y="171" font-family="'Courier New', monospace" font-size="11" font-weight="bold" text-anchor="end" fill="#1e293b">+20 B</text>

    <rect x="15" y="196" width="365" height="30" rx="6" fill="#f1f5f9" stroke="#94a3b8"/>
    <text x="25" y="216" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="11" font-weight="bold" fill="#0f172a">Total GDDR6 Traffic:</text>
    <text x="365" y="216" font-family="'Courier New', monospace" font-size="12" font-weight="bold" text-anchor="end" fill="#b91c1c">28 Bytes / parameter (19.34 ms)</text>
  </g>

  <g transform="translate(455, 75)">
    <rect width="395" height="240" rx="10" fill="#eff6ff" stroke="#60a5fa" stroke-width="1.5" filter="url(#shadow4)"/>
    <rect x="0" y="0" width="395" height="34" rx="10" fill="#2563eb"/>
    <rect x="0" y="24" width="395" height="10" fill="#2563eb"/>
    <text x="197" y="22" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle" fill="#ffffff">FUSED IN-REGISTER KERNEL (OURS)</text>

    <rect x="15" y="46" width="365" height="40" rx="6" fill="#ffffff" stroke="#93c5fd"/>
    <text x="25" y="64" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="11" font-weight="bold" fill="#1e3a8a">1. Backward GEMM Tile Accumulation</text>
    <text x="25" y="78" font-family="'Courier New', monospace" font-size="9.5" fill="#2563eb">Accumulates dW directly into RF registers</text>
    <text x="365" y="71" font-family="'Courier New', monospace" font-size="11" font-weight="bold" text-anchor="end" fill="#059669">0 B DRAM</text>

    <rect x="15" y="96" width="365" height="40" rx="6" fill="#dbeafe" stroke="#3b82f6"/>
    <text x="25" y="114" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="11" font-weight="bold" fill="#1e40af">2. In-Register Fusion Pipeline</text>
    <text x="25" y="128" font-family="'Courier New', monospace" font-size="9.5" fill="#1d4ed8">No intermediate DRAM spill; dW retained in SM registers</text>
    <text x="365" y="121" font-family="'Courier New', monospace" font-size="11" font-weight="bold" text-anchor="end" fill="#059669">-6 B Saved</text>

    <rect x="15" y="146" width="365" height="40" rx="6" fill="#ffffff" stroke="#93c5fd"/>
    <text x="25" y="164" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="11" font-weight="bold" fill="#1e3a8a">3. Fused AdamW Update</text>
    <text x="25" y="178" font-family="'Courier New', monospace" font-size="9.5" fill="#2563eb">Reads W, m, v → fuses dW in-place → writes back</text>
    <text x="365" y="171" font-family="'Courier New', monospace" font-size="11" font-weight="bold" text-anchor="end" fill="#1e3a8a">+22 B</text>

    <rect x="15" y="196" width="365" height="30" rx="6" fill="#d1fae5" stroke="#10b981"/>
    <text x="25" y="216" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="11" font-weight="bold" fill="#065f46">Total GDDR6 Traffic:</text>
    <text x="365" y="216" font-family="'Courier New', monospace" font-size="12" font-weight="bold" text-anchor="end" fill="#047857">22 Bytes (-21.43%, 9.98 ms = 1.94x)</text>
  </g>
</svg>
"""


def fig5_sft_reasoning_flowchart() -> str:
    return """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 360" width="900" height="360">
  <defs>
    <filter id="shadow5" x="-5%" y="-5%" width="110%" height="115%">
      <feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#0f172a" flood-opacity="0.10"/>
    </filter>
    <marker id="arr5" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#2563eb"/>
    </marker>
  </defs>

  <rect width="900" height="360" fill="#ffffff"/>

  <text x="450" y="30" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="15" font-weight="bold" text-anchor="middle" fill="#0f172a">
    5-TAG SCIENTIFIC DISCOVERY REASONING SCHEMA (BABY-CHALK 1.5B SFT)
  </text>
  <text x="450" y="50" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="12" text-anchor="middle" fill="#64748b">
    Procedural Tag Conditioning via Prompt Loss Masking Delivering 6x Grounding Improvement (60.0% vs 10.0%)
  </text>

  <g transform="translate(30, 80)">
    <rect width="120" height="70" rx="8" fill="#f1f5f9" stroke="#94a3b8" stroke-width="1.5" filter="url(#shadow5)"/>
    <text x="60" y="32" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="11" font-weight="bold" text-anchor="middle" fill="#1e293b">Competition</text>
    <text x="60" y="48" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="11" font-weight="bold" text-anchor="middle" fill="#1e293b">Math Problem</text>
    <text x="60" y="62" font-family="'Courier New', monospace" font-size="9" text-anchor="middle" fill="#64748b">AIME / AMC 12</text>
  </g>

  <path d="M 150 115 L 180 115" stroke="#2563eb" stroke-width="2.5" marker-end="url(#arr5)"/>

  <g transform="translate(190, 70)">
    <rect width="150" height="90" rx="8" fill="#eff6ff" stroke="#3b82f6" stroke-width="1.5" filter="url(#shadow5)"/>
    <rect x="0" y="0" width="150" height="26" rx="8" fill="#2563eb"/>
    <rect x="0" y="18" width="150" height="8" fill="#2563eb"/>
    <text x="75" y="18" font-family="'Courier New', monospace" font-size="11" font-weight="bold" text-anchor="middle" fill="#ffffff">&lt;explore&gt;</text>
    <text x="75" y="44" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="10" font-weight="bold" text-anchor="middle" fill="#1e3a8a">Cognitive Scratchpad</text>
    <text x="75" y="60" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="9" text-anchor="middle" fill="#3b82f6">Divergent thinking</text>
    <text x="75" y="74" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="9" text-anchor="middle" fill="#64748b">Identifies candidate angles</text>
  </g>

  <path d="M 340 115 L 365 115" stroke="#2563eb" stroke-width="2.5" marker-end="url(#arr5)"/>

  <g transform="translate(375, 70)">
    <rect width="150" height="90" rx="8" fill="#faf5ff" stroke="#a855f7" stroke-width="1.5" filter="url(#shadow5)"/>
    <rect x="0" y="0" width="150" height="26" rx="8" fill="#9333ea"/>
    <rect x="0" y="18" width="150" height="8" fill="#9333ea"/>
    <text x="75" y="18" font-family="'Courier New', monospace" font-size="11" font-weight="bold" text-anchor="middle" fill="#ffffff">&lt;conjecture&gt;</text>
    <text x="75" y="44" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="10" font-weight="bold" text-anchor="middle" fill="#581c87">Hypothesis Formulation</text>
    <text x="75" y="60" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="9" text-anchor="middle" fill="#9333ea">Explicit invariant claim</text>
    <text x="75" y="74" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="9" text-anchor="middle" fill="#64748b">States testable assertions</text>
  </g>

  <path d="M 525 115 L 550 115" stroke="#2563eb" stroke-width="2.5" marker-end="url(#arr5)"/>

  <g transform="translate(560, 70)">
    <rect width="150" height="90" rx="8" fill="#fef2f2" stroke="#ef4444" stroke-width="1.5" filter="url(#shadow5)"/>
    <rect x="0" y="0" width="150" height="26" rx="8" fill="#dc2626"/>
    <rect x="0" y="18" width="150" height="8" fill="#dc2626"/>
    <text x="75" y="18" font-family="'Courier New', monospace" font-size="10.5" font-weight="bold" text-anchor="middle" fill="#ffffff">&lt;test_edge_cases&gt;</text>
    <text x="75" y="44" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="10" font-weight="bold" text-anchor="middle" fill="#991b1b">Boundary Stress Test</text>
    <text x="75" y="60" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="9" text-anchor="middle" fill="#dc2626">Tests n=0, n=1, primes</text>
    <text x="75" y="74" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="9" text-anchor="middle" fill="#64748b">Prevents hallucination</text>
  </g>

  <path d="M 710 115 L 750 115 C 780 115, 780 230, 750 230 L 720 230" fill="none" stroke="#2563eb" stroke-width="2.5" marker-end="url(#arr5)"/>

  <g transform="translate(560, 185)">
    <rect width="150" height="90" rx="8" fill="#fffbeb" stroke="#f59e0b" stroke-width="1.5" filter="url(#shadow5)"/>
    <rect x="0" y="0" width="150" height="26" rx="8" fill="#d97706"/>
    <rect x="0" y="18" width="150" height="8" fill="#d97706"/>
    <text x="75" y="18" font-family="'Courier New', monospace" font-size="10.5" font-weight="bold" text-anchor="middle" fill="#ffffff">&lt;lemma_isolate&gt;</text>
    <text x="75" y="44" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="10" font-weight="bold" text-anchor="middle" fill="#78350f">Modular Factoring</text>
    <text x="75" y="60" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="9" text-anchor="middle" fill="#d97706">Intermediate sub-proof</text>
    <text x="75" y="74" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="9" text-anchor="middle" fill="#64748b">Separates reusable algebra</text>
  </g>

  <path d="M 560 230 L 535 230" stroke="#2563eb" stroke-width="2.5" marker-end="url(#arr5)"/>

  <g transform="translate(375, 185)">
    <rect width="150" height="90" rx="8" fill="#ecfdf5" stroke="#10b981" stroke-width="1.5" filter="url(#shadow5)"/>
    <rect x="0" y="0" width="150" height="26" rx="8" fill="#059669"/>
    <rect x="0" y="18" width="150" height="8" fill="#059669"/>
    <text x="75" y="18" font-family="'Courier New', monospace" font-size="11" font-weight="bold" text-anchor="middle" fill="#ffffff">&lt;formal_proof&gt;</text>
    <text x="75" y="44" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="10" font-weight="bold" text-anchor="middle" fill="#065f46">Deductive Synthesis</text>
    <text x="75" y="60" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="9" text-anchor="middle" fill="#059669">Step-by-step resolution</text>
    <text x="75" y="74" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="9" text-anchor="middle" fill="#64748b">Verified formal logic</text>
  </g>

  <path d="M 375 230 L 340 230" stroke="#2563eb" stroke-width="2.5" marker-end="url(#arr5)"/>

  <g transform="translate(190, 185)">
    <rect width="140" height="90" rx="8" fill="#f0fdf4" stroke="#22c55e" stroke-width="2" filter="url(#shadow5)"/>
    <text x="70" y="32" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="11" font-weight="bold" text-anchor="middle" fill="#15803d">Grounded Answer</text>
    <text x="70" y="52" font-family="'Courier New', monospace" font-size="12" font-weight="bold" text-anchor="middle" fill="#166534">\\boxed{answer}</text>
    <text x="70" y="74" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="9.5" font-weight="bold" text-anchor="middle" fill="#15803d">60% Grounding Pass</text>
  </g>

  <rect x="30" y="300" width="840" height="42" rx="6" fill="#f8fafc" stroke="#cbd5e1"/>
  <text x="450" y="326" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="11" font-weight="bold" text-anchor="middle" fill="#0f172a">
    Empirical Verification: 605 seeds in chalk_seeds_500.jsonl • 6x Grounding Improvement over Base Model (10.0% → 60.0%) • Peak VRAM: 3.2 GB
  </text>
</svg>
"""


def fig6_speculative_engine_architecture() -> str:
    return """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 880 340" width="880" height="340">
  <defs>
    <filter id="shadow6" x="-5%" y="-5%" width="110%" height="115%">
      <feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#0f172a" flood-opacity="0.10"/>
    </filter>
    <marker id="arr6" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#2563eb"/>
    </marker>
  </defs>

  <rect width="880" height="340" fill="#ffffff"/>

  <text x="440" y="30" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="15" font-weight="bold" text-anchor="middle" fill="#0f172a">
    UNIFIED SPECULATIVE SERVING ENGINE ARCHITECTURE ON TESLA T4
  </text>
  <text x="440" y="50" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="12" text-anchor="middle" fill="#64748b">
    Pre-allocated StaticKVCache with O(1) Rollback and PromptLookupDraftEngine Delivering 1.48x Net Speedup
  </text>

  <g transform="translate(30, 75)">
    <rect width="250" height="235" rx="10" fill="#f0fdfa" stroke="#14b8a6" stroke-width="1.5" filter="url(#shadow6)"/>
    <rect x="0" y="0" width="250" height="34" rx="10" fill="#0d9488"/>
    <rect x="0" y="24" width="250" height="10" fill="#0d9488"/>
    <text x="125" y="22" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle" fill="#ffffff">ZERO-WEIGHT PROMPT DRAFT</text>

    <rect x="12" y="46" width="226" height="42" rx="6" fill="#ffffff" stroke="#99f6e4"/>
    <text x="125" y="64" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="11" font-weight="bold" text-anchor="middle" fill="#115e59">N-Gram Suffix Tree Matching</text>
    <text x="125" y="78" font-family="'Courier New', monospace" font-size="10" text-anchor="middle" fill="#0d9488">Proposal Latency: 0.0062 ms</text>

    <rect x="12" y="98" width="226" height="42" rx="6" fill="#ffffff" stroke="#99f6e4"/>
    <text x="125" y="116" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="11" font-weight="bold" text-anchor="middle" fill="#115e59">Speculative Candidate Horizon</text>
    <text x="125" y="130" font-family="'Courier New', monospace" font-size="10" text-anchor="middle" fill="#0d9488">K = 3 Proposed Tokens</text>

    <rect x="12" y="150" width="226" height="42" rx="6" fill="#ffffff" stroke="#99f6e4"/>
    <text x="125" y="168" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="11" font-weight="bold" text-anchor="middle" fill="#115e59">Memory Footprint</text>
    <text x="125" y="182" font-family="'Courier New', monospace" font-size="10" text-anchor="middle" fill="#059669">0 MB (Zero weights loaded)</text>

    <rect x="12" y="200" width="226" height="26" rx="6" fill="#ccfbf1" stroke="#5eead4"/>
    <text x="125" y="217" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="10" font-weight="bold" text-anchor="middle" fill="#0f766e">Eliminates 2nd Model VRAM Tax</text>
  </g>

  <path d="M 280 190 L 310 190" stroke="#2563eb" stroke-width="2.5" marker-end="url(#arr6)"/>
  <text x="295" y="178" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="10" font-weight="bold" text-anchor="middle" fill="#2563eb">K=3</text>

  <g transform="translate(315, 75)">
    <rect width="250" height="235" rx="10" fill="#eff6ff" stroke="#3b82f6" stroke-width="1.5" filter="url(#shadow6)"/>
    <rect x="0" y="0" width="250" height="34" rx="10" fill="#2563eb"/>
    <rect x="0" y="24" width="250" height="10" fill="#2563eb"/>
    <text x="125" y="22" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle" fill="#ffffff">TARGET MODEL VERIFICATION</text>

    <rect x="12" y="46" width="226" height="42" rx="6" fill="#ffffff" stroke="#bfdbfe"/>
    <text x="125" y="64" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="11" font-weight="bold" text-anchor="middle" fill="#1e3a8a">Full Model (Qwen2.5-1.5B)</text>
    <text x="125" y="78" font-family="'Courier New', monospace" font-size="10" text-anchor="middle" fill="#2563eb">Single Forward Pass for K=3</text>

    <rect x="12" y="98" width="226" height="42" rx="6" fill="#ffffff" stroke="#bfdbfe"/>
    <text x="125" y="116" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="11" font-weight="bold" text-anchor="middle" fill="#1e3a8a">Low-Precision GEMV Kernels</text>
    <text x="125" y="130" font-family="'Courier New', monospace" font-size="10" text-anchor="middle" fill="#2563eb">fused_w4a16_gemv / sm_75</text>

    <rect x="12" y="150" width="226" height="42" rx="6" fill="#ffffff" stroke="#bfdbfe"/>
    <text x="125" y="168" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="11" font-weight="bold" text-anchor="middle" fill="#1e3a8a">Acceptance Criterion</text>
    <text x="125" y="182" font-family="'Courier New', monospace" font-size="10" text-anchor="middle" fill="#059669">Greedy argmax matching</text>

    <rect x="12" y="200" width="226" height="26" rx="6" fill="#dbeafe" stroke="#93c5fd"/>
    <text x="125" y="217" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="10" font-weight="bold" text-anchor="middle" fill="#1e40af">Parallel Token Validation</text>
  </g>

  <path d="M 565 190 L 595 190" stroke="#059669" stroke-width="2.5" marker-end="url(#arr6)"/>
  <text x="580" y="178" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="10" font-weight="bold" text-anchor="middle" fill="#059669">O(1)</text>

  <g transform="translate(600, 75)">
    <rect width="250" height="235" rx="10" fill="#ecfdf5" stroke="#10b981" stroke-width="1.5" filter="url(#shadow6)"/>
    <rect x="0" y="0" width="250" height="34" rx="10" fill="#059669"/>
    <rect x="0" y="24" width="250" height="10" fill="#059669"/>
    <text x="125" y="22" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle" fill="#ffffff">STATIC KV CACHE &amp; ROLLBACK</text>

    <rect x="12" y="46" width="226" height="42" rx="6" fill="#ffffff" stroke="#a7f3d0"/>
    <text x="125" y="64" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="11" font-weight="bold" text-anchor="middle" fill="#065f46">Zero Heap Allocation</text>
    <text x="125" y="78" font-family="'Courier New', monospace" font-size="10" text-anchor="middle" fill="#059669">Pre-allocated max_seq_len</text>

    <rect x="12" y="98" width="226" height="42" rx="6" fill="#ffffff" stroke="#a7f3d0"/>
    <text x="125" y="116" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="11" font-weight="bold" text-anchor="middle" fill="#065f46">O(1) Pointer Rollback</text>
    <text x="125" y="130" font-family="'Courier New', monospace" font-size="10" text-anchor="middle" fill="#059669">rewind(accepted_count)</text>

    <rect x="12" y="150" width="226" height="42" rx="6" fill="#ffffff" stroke="#a7f3d0"/>
    <text x="125" y="168" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="11" font-weight="bold" text-anchor="middle" fill="#065f46">Net Wall-Clock Speedup</text>
    <text x="125" y="182" font-family="'Courier New', monospace" font-size="10.5" font-weight="bold" text-anchor="middle" fill="#047857">1.48x Net (up to 2.16x)</text>

    <rect x="12" y="200" width="226" height="26" rx="6" fill="#d1fae5" stroke="#6ee7b7"/>
    <text x="125" y="217" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" font-size="10" font-weight="bold" text-anchor="middle" fill="#064e3b">26.59 → 39.34 tok/s on T4</text>
  </g>
</svg>
"""


def main():
    generators = {
        "fig1_warp_specialization": fig1_warp_specialization,
        "fig2_lop3_dequant_pipeline": fig2_lop3_dequant_pipeline,
        "fig3_thermal_dvfs_pacing": fig3_thermal_dvfs_pacing,
        "fig4_fused_adamw_pipeline": fig4_fused_adamw_pipeline,
        "fig5_sft_reasoning_flowchart": fig5_sft_reasoning_flowchart,
        "fig6_speculative_engine_architecture": fig6_speculative_engine_architecture,
    }

    print(f"Generating {len(generators)} publication diagrams in {FIGURES_DIR}...")
    for name, gen in generators.items():
        svg_path = FIGURES_DIR / f"{name}.svg"
        svg_content = gen()
        svg_path.write_text(svg_content, encoding="utf-8")
        print(f"  ✓ Written SVG: {svg_path.name}")

        typ_wrapper = f"""
#set page(width: auto, height: auto, margin: 0pt)
#image("{name}.svg")
"""
        typ_path = FIGURES_DIR / f"{name}.typ"
        typ_path.write_text(typ_wrapper, encoding="utf-8")

        pdf_path = FIGURES_DIR / f"{name}.pdf"
        png_path = FIGURES_DIR / f"{name}.png"

        try:
            subprocess.run(["typst", "compile", str(typ_path), str(pdf_path)], check=True, stdout=subprocess.DEVNULL)
            subprocess.run(["typst", "compile", str(typ_path), str(png_path)], check=True, stdout=subprocess.DEVNULL)
            print(f"  ✓ Compiled PDF & PNG: {name}.pdf, {name}.png")
        except Exception as e:
            print(f"  ! Typst compilation notice for {name}: {e}")
        finally:
            if typ_path.exists():
                typ_path.unlink()

    print("\nAll figures generated successfully!")


if __name__ == "__main__":
    main()
