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

# quantize + kernel (supports per-group group_size=128)
def quantize_weight(Wt, group_size=128):
    out_f, in_f = Wt.shape
    Wf = Wt.float()
    if group_size > 0:
        num_groups = in_f // group_size
        Wf_grouped = Wf.reshape(out_f, num_groups, group_size)
        maxv = Wf_grouped.amax(dim=2, keepdim=True)
        minv = Wf_grouped.amin(dim=2, keepdim=True)
        scale = ((maxv - minv) / 15.0).clamp_min(1e-10)
        zp = (-minv / scale).clamp(0, 15)
        q = torch.clamp(torch.round(Wf_grouped / scale) + zp, 0, 15).reshape(out_f, in_f).to(torch.int32).t().contiguous().cpu()
        packed = torch.zeros(in_f // 8, out_f, dtype=torch.int32)
        for i in range(8):
            packed |= q[i::8, :] << (4 * i)
        sc = scale.squeeze(2).t().contiguous().half()  # [num_groups, out_f]
        zz = zp.squeeze(2).t().contiguous().half()     # [num_groups, out_f]
    else:
        maxv = Wf.amax(dim=1, keepdim=True)
        minv = Wf.amin(dim=1, keepdim=True)
        scale = ((maxv - minv) / 15.0).clamp_min(1e-10)
        zp = (-minv / scale).clamp(0, 15)
        q = torch.clamp(torch.round(Wf / scale) + zp, 0, 15).to(torch.int32).t().contiguous().cpu()
        packed = torch.zeros(in_f // 8, out_f, dtype=torch.int32)
        for i in range(8):
            packed |= q[i::8, :] << (4 * i)
        sc = scale.squeeze(1).half().unsqueeze(0).contiguous()  # [1, out_f]
        zz = zp.squeeze(1).half().unsqueeze(0).contiguous()     # [1, out_f]
    return packed.cuda(), sc.cuda(), zz.cuda()

def dequant_ref(packed, sc, zp, group_size=128):
    in_f_8, out_f = packed.shape
    in_f = in_f_8 * 8
    Wq = torch.zeros(in_f, out_f, dtype=torch.float32)
    pc = packed.cpu()
    for i in range(8):
        Wq[i::8, :] = ((pc >> (4 * i)) & 0xF).float()
    sc_cpu = sc.float().cpu()
    zz_cpu = zp.float().cpu()
    if group_size > 0:
        num_groups = in_f // group_size
        Wq_grouped = Wq.reshape(num_groups, group_size, out_f)
        sc_3d = sc_cpu.unsqueeze(1) # [num_groups, 1, out_f]
        zz_3d = zz_cpu.unsqueeze(1) # [num_groups, 1, out_f]
        W_deq = ((Wq_grouped - zz_3d) * sc_3d).reshape(in_f, out_f).half().t()
    else:
        sc_1d = sc_cpu.squeeze(0)
        zz_1d = zz_cpu.squeeze(0)
        W_deq = ((Wq - zz_1d) * sc_1d).half().t()
    return W_deq.cuda()

print("--- [Per-Group group=128 Test] ---")
packed, sc, zp = quantize_weight(W, group_size=128)
ker = t4_kernels.fused_w4a16_gemm_u4(x, packed, sc, zp, 128)
print("kernel out shape", ker.shape)
d = (ker.float() - torch_out.float()).abs()
print("kernel vs torch: max_abs_diff", d.max().item(), "mean", d.mean().item())
print("torch max", torch_out.float().abs().max().item())

W_deq = dequant_ref(packed, sc, zp, group_size=128)
ref = x.float() @ W_deq.float().t()
d2 = (ker.float() - ref).abs().max().item()
print("kernel vs python-ref: max_abs_diff", d2, " rel", d2 / ref.abs().max().item())
rel_err_vs_torch = (torch_out.float() - ref).abs().max().item() / torch_out.float().abs().max().item()
print("torch vs python-ref : max_abs_diff", (torch_out.float() - ref).abs().max().item(), "rel", rel_err_vs_torch)

print("\n--- [Legacy Per-Channel Test] ---")
packed_c, sc_c, zp_c = quantize_weight(W, group_size=0)
ker_c = t4_kernels.fused_w4a16_gemm_u4(x, packed_c, sc_c, zp_c, 0)
W_deq_c = dequant_ref(packed_c, sc_c, zp_c, group_size=0)
ref_c = x.float() @ W_deq_c.float().t()
rel_c = (torch_out.float() - ref_c).abs().max().item() / torch_out.float().abs().max().item()
print("Per-channel torch vs python-ref rel err:", rel_c)
print("Improvement factor with per-group:", rel_c / max(rel_err_vs_torch, 1e-9))