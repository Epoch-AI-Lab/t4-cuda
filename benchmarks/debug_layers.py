import torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import t4_kernels

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

def captures_for(model, input_ids, max_new=4, linear_caps=None):
    caps = {}
    orig = [l.forward for l in model.model.layers]
    def make(i):
        def f(*a, **kw):
            out = orig[i](*a, **kw)
            caps[i] = out[0].clone()
            return out
        return f
    for i, l in enumerate(model.model.layers):
        l.forward = make(i)
    if linear_caps is not None:
        lin_orig = {}
        def make_lin(mod, tag):
            def g(*a, **kw):
                out = mod._orig_forward(*a, **kw)
                linear_caps.setdefault(tag, []).append(out.clone())
                return out
            return g
        for tag, mod in linear_caps["_mods"].items():
            mod._orig_forward = mod.forward
            mod.forward = make_lin(mod, tag)
    with torch.inference_mode():
        model.generate(input_ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
    for i, l in enumerate(model.model.layers):
        l.forward = orig[i]
    if linear_caps is not None:
        for tag, mod in linear_caps["_mods"].items():
            mod.forward = mod._orig_forward
    return caps

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda").eval()
ids = tok(["What is 17*24?"], return_tensors="pt")["input_ids"].cuda()

lc0 = {"_mods": {}}
for i, l in enumerate(model.model.layers[:1]):
    for name, m in l.named_modules():
        if isinstance(m, nn.Linear):
            lc0["_mods"][f"L{i}.{name}"] = m
base = captures_for(model, ids, 4, lc0)
print("base done")
for tag in lc0["_mods"]:
    outs = lc0[tag]
    print(tag, "shape", [tuple(o.shape) for o in outs][:3])

def quantize_weight(W):
    out_f, in_f = W.shape
    Wf = W.float()
    amax = Wf.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    scale = (amax / 7.0)
    q = torch.clamp(torch.round(Wf / scale) + 8, 0, 15).to(torch.int32).t().contiguous().cpu()
    packed = torch.zeros(in_f // 8, out_f, dtype=torch.int32)
    for i in range(8):
        packed |= q[i::8, :] << (4 * i)
    return packed.cuda(), scale.squeeze(1).half().unsqueeze(0).contiguous().cuda(), \
           torch.full((1, out_f), 8.0, dtype=torch.float16).cuda()

Q = {}
def patched_forward(self, x):
    packed, scales, zp = Q[self]
    shape = x.shape
    out = t4_kernels.fused_w4a16_gemm_u4(x.reshape(-1, shape[-1]), packed, scales, zp)
    return out.reshape(*shape[:-1], out.shape[-1])

for name, mod in model.named_modules():
    if isinstance(mod, nn.Linear):
        Q[mod] = quantize_weight(mod.weight.data)
        mod.forward = patched_forward.__get__(mod)
print("quantized", len(Q), "linears")

lc1 = {"_mods": {}}
for i, l in enumerate(model.model.layers[:1]):
    for name, m in l.named_modules():
        if isinstance(m, nn.Linear):
            lc1["_mods"][f"L{i}.{name}"] = m
q = captures_for(model, ids, 4, lc1)
print("quant done")

for tag in lc0["_mods"]:
    b = lc0[tag][0].float()
    qo = lc1[tag][0].float()
    d = (qo - b).abs().max().item()
    print(f"{tag}: max_abs_diff={d:.4f}  base_max={b.abs().max().item():.3f}")