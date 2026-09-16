import torch, torch.nn as nn
from transformers import AutoModelForCausalLM
import t4_kernels

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=torch.float16, device_map="cuda").eval()
W = model.model.layers[0].self_attn.q_proj.weight.data  # [out=1536, in=896]
out_f, in_f = W.shape
Wf = W.float()
amax = Wf.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
scale = amax / 7.0
q = torch.clamp(torch.round(Wf / scale) + 8, 0, 15).to(torch.int32).t().contiguous().cpu()  # [in, out]
packed = torch.zeros(in_f // 8, out_f, dtype=torch.int32)
for i in range(8):
    packed |= q[i::8, :] << (4 * i)
packed = packed.cuda()
scales = scale.squeeze(1).half().unsqueeze(0).contiguous().cuda()
zp = torch.full((1, out_f), 8.0, dtype=torch.float16).cuda()

# dequant reference straight from packed (using repo's own unpack semantics)
unp = torch.zeros(in_f, out_f)
for i in range(8):
    unp[i::8, :] = ((packed.cpu() >> (4 * i)) & 0xF).float()
W_deq = ((unp - 8.0) * scale.t().cpu().float()).half().cuda()
print("quant err vs orig W:", (W_deq.float() - Wf.cuda()).abs().mean().item())

torch.manual_seed(0)
for M in (1, 16, 280):
    A = torch.randn(M, in_f, dtype=torch.float16, device="cuda")
    C = t4_kernels.fused_w4a16_gemm_u4(A, packed, scales, zp)
    C_ref = A.float() @ W_deq.float()  # kernel computes A @ W with W [K, N]
    err = (C.float() - C_ref).abs().max().item()
    rel = err / C_ref.abs().max().item()
    print(f"M={M:4d}  max_abs_err={err:.4f}  rel={rel:.5f}")
