import torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import t4_kernels

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda").eval()
ids = tok(["What is 17*24?"], return_tensors="pt")["input_ids"].cuda()

# get layer-0 q_proj real input (post input_layernorm)
import torch.nn.functional as F
h = model.model.embed_tokens(ids)
h = model.model.layers[0].input_layernorm(h)
x = h.reshape(-1, h.shape[-1])
print("x shape", x.shape, "x max", x.float().abs().max().item())

W = model.model.layers[0].self_attn.q_proj.weight  # [out, in]
print("W shape", W.shape)

# torch reference on real activations
torch_out = F.linear(x, W)

# quantize + kernel
def quantize_weight(Wt):
    out_f, in_f = Wt.shape
    Wf = Wt.float()
    maxv = Wf.amax(dim=1, keepdim=True)
    minv = Wf.amin(dim=1, keepdim=True)
    scale = ((maxv - minv) / 15.0).clamp_min(1e-10)
    zp = (-minv / scale).clamp(0, 15)
    q = torch.clamp(torch.round(Wf / scale) + zp, 0, 15).to(torch.int32).t().contiguous().cpu()
    packed = torch.zeros(in_f // 8, out_f, dtype=torch.int32)
    for i in range(8):
        packed |= q[i::8, :] << (4 * i)
    sc = scale.squeeze(1).half()  # [out]
    zz = zp.squeeze(1).half()     # [out]
    return packed.cuda(), sc.unsqueeze(0).contiguous().cuda(), zz.unsqueeze(0).cuda()

packed, sc, zp = quantize_weight(W)
ker = t4_kernels.fused_w4a16_gemm_u4(x, packed, sc, zp)
print("kernel out shape", ker.shape)
d = (ker.float() - torch_out.float()).abs()
print("kernel vs torch: max_abs_diff", d.max().item(), "mean", d.mean().item())
print("torch max", torch_out.float().abs().max().item())

# direct reference via python for the EXACT scale used (all CPU, then move)
in_f, out_f = W.shape
Wq = torch.zeros(in_f, out_f, dtype=torch.float32)
pc = packed.cpu()
for i in range(8):
    Wq[i::8, :] = ((pc >> (4 * i)) & 0xF).float()
sc_cpu = sc.float().cpu().squeeze(0)
zz_cpu = zp.float().cpu().squeeze(0)
W_deq = ((Wq - zz_cpu) * sc_cpu).half().t().cuda()  # [out, in]
ref = x.float() @ W_deq.float().t()
d2 = (ker.float() - ref).abs().max().item()
print("kernel vs python-ref: max_abs_diff", d2, " rel", d2 / ref.abs().max().item())
print("torch vs python-ref : max_abs_diff", (torch_out.float() - ref).abs().max().item())