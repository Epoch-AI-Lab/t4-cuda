import torch
import t4_kernels

torch.manual_seed(1)
K, N, M = 64, 16, 2
# known nibbles 0..15
q = torch.randint(0, 16, (K, N), dtype=torch.int32)
packed = torch.zeros(K // 8, N, dtype=torch.int32)
for i in range(8):
    packed |= q[i::8, :] << (4 * i)
packed = packed.cuda()
scales = torch.ones(1, N, dtype=torch.float16).cuda()
zp = torch.zeros(1, N, dtype=torch.float16).cuda()
A = torch.randn(M, K, dtype=torch.float16).cuda()
C = t4_kernels.fused_w4a16_gemm_u4(A, packed, scales, zp)
ref = A.float() @ q.cuda().float()  # kernel computes A @ W, W [K,N]
print("max err:", (C.float() - ref).abs().max().item())
print("C[0,:4] :", C[0, :4].tolist())
print("ref[0,:4]:", ref[0, :4].tolist())
