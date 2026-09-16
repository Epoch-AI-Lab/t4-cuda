import torch
import t4_kernels

torch.manual_seed(3)
K, N, M = 128, 32, 1
# known nibbles 0..15, realistic per-channel scale & zp
q = torch.randint(0, 16, (K, N), dtype=torch.int32)
s = (torch.rand(1, N) * 0.1 + 0.01).half()
zp = torch.full((1, N), 8.0, dtype=torch.float16)

packed = torch.zeros(K // 8, N, dtype=torch.int32)
for i in range(8):
    packed |= q[i::8, :] << (4 * i)
packed = packed.cuda()
scales, zps = s.cuda(), zp.cuda()
A = torch.randn(M, K, dtype=torch.half).cuda()

# reference: C = A @ ((q - 8) * s)   -- kernel computes A @ W, W [K, N]
C_ref = A.half().float() @ ((q.cuda().float() - 8.0) * s.cuda().float())
C = t4_kernels.fused_w4a16_gemm_u4(A, packed, scales, zps)

diff = (C.float() - C_ref).abs()
rel = diff / C_ref.abs().clamp_min(1e-6)
print("max_abs_diff:", diff.max().item())
print("mean_rel    :", rel.mean().item())
print("pred[:8]    :", C[0, :8].tolist())
print("ref[:8]     :", C_ref[0, :8].tolist())
print("ratio pred/ref:", (C.float()[0, :8] / C_ref[0, :8]).tolist())
print("s[:8]       :", s.cuda().float()[0, :8].tolist())