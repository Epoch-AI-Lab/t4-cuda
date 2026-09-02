#!/usr/bin/env python3
"""Isolated generation benchmark: fp16 policy vs INT4-per-channel policy
using t4-cuda fused_w4a16_gemm kernels. No TRL. This is the go/no-go gate for
putting the quantized policy inside GRPO rollouts.

Measures: tokens/sec over greedy generation of 256 new tokens, 8 concurrent
prompts (matching the GRPO rollout shape), plus output drift vs fp16.
"""
import subprocess, time
subprocess.run("pip install -q transformers accelerate", shell=True)

import torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import t4_kernels

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPTS = [
    "What is 17*24? End with: #### <answer>",
    "Solve step by step: If a train travels 60km in 45 minutes, what is its speed in km/h? End with: #### <answer>",
    "What is the sum of the first 10 positive integers? End with: #### <answer>",
    "If x+3=10 and y=x*2, what is y? End with: #### <answer>",
    "What is 15% of 200? End with: #### <answer>",
    "A shirt costs $40 and is discounted 25%. What is the final price? End with: #### <answer>",
    "What is 2^10? End with: #### <answer>",
    "Tom has 3 boxes with 12 apples each. He eats 5. How many left? End with: #### <answer>",
]
NEW_TOKENS = 256

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda")
tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model.eval()

inputs = tok(PROMPTS, return_tensors="pt", padding=True).to("cuda")

def bench(m, tag):
    # warmup (CUDA graphs/autotune) then timed
    with torch.inference_mode():
        m.generate(**inputs, max_new_tokens=32, do_sample=False)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = m.generate(**inputs, max_new_tokens=NEW_TOKENS, do_sample=False,
                         pad_token_id=tok.pad_token_id)
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    # count non-pad generated tokens
    gen = out[:, inputs["input_ids"].shape[1]:]
    ntok = (gen != tok.pad_token_id).sum().item()
    print(f"[{tag}] {dt:.2f}s for {ntok} tokens = {ntok/dt:.1f} tok/s")
    return out

fp16_out = bench(model, "fp16 baseline")

# ---- quantize every nn.Linear to per-group INT4 (group=128, u4 nibbles, GPTQ-style asymmetric) ----
def quantize_weight(W, group_size=128):
    """W [out, in] fp16 -> (W_packed int32 [in/8, out], scales [num_groups, out], zp [num_groups, out], group_size)"""
    out_f, in_f = W.shape
    assert in_f % 8 == 0, in_f
    Wf = W.float()
    if group_size > 0 and in_f % group_size == 0:
        num_groups = in_f // group_size
        Wf_grouped = Wf.reshape(out_f, num_groups, group_size)
        maxv = Wf_grouped.amax(dim=2, keepdim=True)
        minv = Wf_grouped.amin(dim=2, keepdim=True)
        scale = ((maxv - minv) / 15.0).clamp_min(1e-10)
        zp = (-minv / scale).clamp(0, 15)
        q = torch.clamp(torch.round(Wf_grouped / scale) + zp, 0, 15).reshape(out_f, in_f).to(torch.int32)
        q = q.t().contiguous().cpu()
        packed = torch.zeros(in_f // 8, out_f, dtype=torch.int32)
        for i in range(8):
            packed |= q[i::8, :] << (4 * i)
        scales = scale.squeeze(2).t().contiguous().half()  # [num_groups, out]
        zps = zp.squeeze(2).t().contiguous().half()        # [num_groups, out]
        return packed.cuda(), scales.cuda(), zps.cuda(), group_size
    else:
        # Fallback to per-channel if in_f is not divisible by group_size
        amax = Wf.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
        scale = amax / 7.0
        q = torch.clamp(torch.round(Wf / scale) + 8, 0, 15).to(torch.int32)
        q = q.t().contiguous().cpu()
        packed = torch.zeros(in_f // 8, out_f, dtype=torch.int32)
        for i in range(8):
            packed |= q[i::8, :] << (4 * i)
        scales = scale.squeeze(1).half().unsqueeze(0).contiguous()
        zps = torch.full((1, out_f), 8.0, dtype=torch.float16)
        return packed.cuda(), scales.cuda(), zps.cuda(), 0

Q = {}
hits = 0
def patched_forward(self, x):
    packed, scales, zp, g_size = Q[self]
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])  # binding takes 2D A only
    out = t4_kernels.fused_w4a16_gemm_u4(x2, packed, scales, zp, g_size)
    return out.reshape(*shape[:-1], out.shape[-1])

for name, mod in list(model.named_modules()):
    if isinstance(mod, nn.Linear):
        Q[mod] = quantize_weight(mod.weight.data, group_size=128)
        mod.forward = patched_forward.__get__(mod)
        hits += 1
print(f"quantized {hits} Linear layers to INT4 per-group (group=128)")

int4_out = bench(model, "int4 fused")

# drift: exact-match rate of generated continuations vs fp16 (greedy)
same = (fp16_out == int4_out).float().mean().item()
print(f"token agreement fp16 vs int4: {100*same:.1f}%")
print("sample fp16 :", repr(tok.decode(fp16_out[0][-40:], skip_special_tokens=True))[-120:])
print("sample int4 :", repr(tok.decode(int4_out[0][-40:], skip_special_tokens=True))[-120:])

vram = torch.cuda.max_memory_allocated() / 1e9
print(f"peak VRAM allocated: {vram:.2f} GB")
