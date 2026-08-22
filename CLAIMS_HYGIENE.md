# T4-CUDA Claims Hygiene Guide

A short checklist to stop the four failure classes found in the audit from recurring.

---

## Rule 1: Tag every number at birth

Every printed number must carry its provenance inline. No exceptions.

```python
# ✅ Do this
print(f"Attn proj fused: {median:.2f} μs  [MEASURED: CUDA events, {N} reps, median, T4 Colab]")
print(f"INT3 decode B=1: {lat:.2f} μs  [ESTIMATED: bytes÷bandwidth model, 320 GB/s spec]")
print(f"DRAM traffic: {saved:.1f}%  [ARITHMETIC: 28→22 B/param byte-count derivation]")

# 🚫 Not this
print(f"INT3 decode B=1: {lat:.2f} μs")
```

**The rule:** if the source isn't in the same line as the number, it will get lost by the time someone writes a summary table. Three tags are enough:

| Tag | Meaning | Can go in a paper? |
|---|---|---|
| `MEASURED` | CUDA events or profiler, real kernel, real GPU | Yes |
| `ESTIMATED` | Model/formula applied to measured inputs | Yes, if labeled |
| `ARITHMETIC` | Pure byte/FLOP counting, no kernel exists | Only as "theoretical" |

---

## Rule 2: Baseline is compiled, not eager

Add this to every benchmark script:

```python
# --- Baseline: compiled ---
compiled_fn = torch.compile(torch.nn.functional.linear)
# Warmup compile
for _ in range(10):
    compiled_fn(A, W_fp16.t())
torch.cuda.synchronize()

compiled_stats = benchmark_latency(compiled_fn, A, W_fp16.t())
print_row(f"Compiled {name}", compiled_stats, memory_mb=compiled_mem)

# --- Baseline: eager (for reference only) ---
eager_stats = benchmark_latency(torch.nn.functional.linear, A, W_fp16.t())
print_row(f"Eager {name}", eager_stats, memory_mb=eager_mem)

# --- Your kernel ---
fused_stats = benchmark_latency(t4_kernels.fused_w4a16_gemm_u4, A, W_packed, scales, zeros)
speedup_vs_compiled = compiled_stats['median'] / fused_stats['median']
speedup_vs_eager = eager_stats['median'] / fused_stats['median']
print_row(f"Fused {name}", fused_stats, speedup=speedup_vs_compiled)
print(f"  (also {speedup_vs_eager:.2f}× vs eager, for reference)")
```

**The rule:** the speedup number you report is always vs compiled. The eager number goes in parentheses for context. If your kernel only beats eager but not compiled, that's Gate 3 telling you the kernel shouldn't exist.

---

## Rule 3: Correctness gates benchmarks — enforce it in CI

The test suite currently exits code 1 (s4 dequant failures) but the benchmark still runs and the summary still prints speedup numbers. Fix:

```python
# In run_all_cuda_tests.py or a wrapper script:
import subprocess, sys

# Stage 1: Correctness (must pass before benchmarks run)
result = subprocess.run(["python3", "tests/test_dequant_correctness.py"])
if result.returncode != 0:
    print("❌ CORRECTNESS FAILED — benchmarks BLOCKED")
    print("Fix the kernel before measuring how fast it is.")
    sys.exit(1)

# Stage 2: Benchmarks (only reached if Stage 1 passes)
subprocess.run(["python3", "tests/benchmark_kernels.py"])
```

**The rule:** a benchmark number from a kernel with known correctness failures is not a speedup — it's how fast wrong answers are produced. Gate the benchmark behind the correctness suite passing at 100%.

For the specific s4 `inf` failures: the bug is likely in the signed dequant path's handling of the sign-extension when the packed word has certain bit patterns. Track it down before reporting any s4 speedups.

---

## Rule 4: State the thermal context

Since Colab won't let you `nvidia-smi -lgc`, document the occupancy state of each benchmark arm:

```python
def benchmark_with_thermal_context(name, func, *args, **kwargs):
    """Benchmark with thermal state logging."""
    import subprocess
    
    # Snapshot clocks before
    pre = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=clocks.sm,power.draw", "--format=csv,noheader"]
    ).decode().strip()
    
    stats = benchmark_latency(func, *args, **kwargs)
    
    # Snapshot clocks after
    post = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=clocks.sm,power.draw", "--format=csv,noheader"]
    ).decode().strip()
    
    print(f"{name}: {stats['median']:.2f} μs  [clocks: {pre} → {post}]")
    return stats
```

**The rule:** if the baseline ran at 1100 MHz (throttled) and your kernel ran at 1500 MHz (light load), 30% of your "speedup" is the fan curve, not your code. Print the clock state for both arms so this is visible.

> [!TIP]
> A cheap alternative: **interleave** baseline and kernel runs within the same loop instead of running all baseline first, then all kernel. This ensures both arms see the same thermal state.

```python
# Interleaved timing — both arms same thermal window
for i in range(iters):
    start_base[i].record(); base_fn(*args); end_base[i].record()
    start_fused[i].record(); fused_fn(*args); end_fused[i].record()
```

---

## Rule 5: Separate simulated results visually

When a test prints both measured and simulated results (like `test_h17_fused_int3_gemv.py` does), use a visual separator:

```python
print("=" * 60)
print("  MEASURED ON GPU (CUDA events)")
print("=" * 60)
# ... real kernel timings ...

print("\n" + "=" * 60)
print("  SPEED-OF-LIGHT ESTIMATES (bandwidth model)")
print("=" * 60)
# ... bytes ÷ bandwidth calculations ...
```

**The rule:** a reader scanning the log should never have to read the code to know whether a number came from a GPU or a calculator.

---

## Summary: The 5 Rules

| # | Rule | Prevents |
|---|---|---|
| 1 | Tag every number `MEASURED` / `ESTIMATED` / `ARITHMETIC` | Simulated numbers presented as benchmarks |
| 2 | Baseline is `torch.compile`, eager is secondary | Inflated speedups vs a baseline nobody uses |
| 3 | Correctness gates benchmarks — block if tests fail | Publishing speeds of wrong kernels |
| 4 | Print clock state for both arms of every A/B comparison | Thermal bias inflating speedups |
| 5 | Visual separator between measured and modeled output | Log readers confusing calculations for measurements |

Apply these five and no future summary table will have the problems the audit found.
