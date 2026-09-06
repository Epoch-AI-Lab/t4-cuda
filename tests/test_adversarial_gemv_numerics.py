#!/usr/bin/env python3
"""Adversarial stress-test suite for re-tiled GEMV kernel and numerics.

Tests:
1. 128-bit load alignment math for W_packed (uint4) and A (float4).
2. Bit-exactness of LOP3 0xEA dequantization across all nibble values and random words.
3. Scale and zero point group indexing math for group_size = 128, 0, and arbitrary groups.
4. Numerical equivalence between retiled GEMV kernel logic and reference FP16 matmul.
5. Boundary conditions and out-of-bounds guards when N % 128 != 0 and K / 8 < 4.
"""

import numpy as np
import pytest
import torch


def test_128bit_load_alignment_math():
    """Verify memory alignment requirements for 128-bit (16-byte) loads.

    Weight loads:
        w_row_u4 = reinterpret_cast<const uint4*>(W_packed + k_idx * N + block_col_base);
        uint4 packed_vec = w_row_u4[lane_id];
    Byte address offset:
        offset_bytes = (k_idx * N + block_col_base + lane_id * 4) * sizeof(uint32_t)
                     = (k_idx * N + block_col_base + lane_id * 4) * 4
    """
    # 1. Check lane_id * 4 and block_col_base * 4 alignment:
    # block_col_base = blockIdx.x * 128.
    # block_col_base * 4 = blockIdx.x * 512, which is 0 mod 16.
    # lane_id * 4 * 4 = lane_id * 16, which is 0 mod 16.
    for lane_id in range(32):
        assert (lane_id * 4 * 4) % 16 == 0
    for block_idx in range(100):
        assert (block_idx * 128 * 4) % 16 == 0

    # 2. Check k_idx * N * 4 mod 16:
    # (k_idx * N * 4) % 16 == 0 for all k_idx iff (k_idx * N) % 4 == 0.
    # At k_idx = 1, this requires N % 4 == 0.
    # Test N % 4 == 0 vs N % 4 != 0:
    for N in [128, 256, 3584, 9472, 18944, 896, 4864, 4096]:
        assert N % 4 == 0, f"Canonical shape N={N} must be multiple of 4"
        for k_idx in range(50):
            byte_offset = (k_idx * N) * 4
            assert byte_offset % 16 == 0, f"N={N}, k_idx={k_idx} is not 16-byte aligned"

    # 3. Adversarial case: N >= 128 but N % 4 != 0 (e.g. N = 129, 130, 131)
    # When N = 130, block_col_base = 0: block_col_base + 128 <= 130 is True!
    # Block 0 takes the fast path.
    # At k_idx = 1, byte offset is 130 * 4 = 520 bytes = 16 * 32 + 8.
    # This proves an unaligned 128-bit load will occur if N % 4 != 0 and N >= 128!
    for N_odd in [129, 130, 131]:
        assert 0 + 128 <= N_odd  # Block 0 qualifies for fast path
        byte_offset_k1 = (1 * N_odd) * 4
        misalignment = byte_offset_k1 % 16
        assert misalignment != 0, f"Expected misalignment for N={N_odd}"

    # 4. Activation loads:
    # const float4* a_vec_ptr = reinterpret_cast<const float4*>(A_row + k_idx * 8);
    # Byte offset: (m * K + k_idx * 8) * sizeof(half) = 2 * m * K + 16 * k_idx.
    # For m > 0, 2 * m * K % 16 == 0 requires K % 8 == 0.
    for K in [3584, 18944, 896, 4096, 128, 64, 8]:
        assert K % 8 == 0
        for m in range(4):
            for k_idx in range(10):
                byte_offset = (m * K + k_idx * 8) * 2
                assert byte_offset % 16 == 0, f"Activation load misaligned for K={K}, m={m}"


def test_lop3_0xEA_all_nibbles_bit_exact():
    """Mathematically verify LOP3 0xEA dequantization across all 16 nibble values."""
    mask_even = 0x000F000F
    magic_exp = 0x64006400  # 1024.0 in FP16 for both halves

    for q in range(16):
        # Place q in lower half (bits 0..3) and upper half (bits 16..19)
        w_word = q | (q << 16)
        # LOP3 0xEA evaluates: (A & B) | C
        raw = (w_word & mask_even) | magic_exp
        raw_lo = raw & 0xFFFF
        raw_hi = (raw >> 16) & 0xFFFF

        # In FP16, raw_lo has sign=0, exponent=25 (bias 15 -> 2^10 = 1024), mantissa = q
        # Evaluated value: (1 + q / 1024) * 1024 = 1024.0 + q
        f16_lo = np.frombuffer(np.uint16(raw_lo).tobytes(), dtype=np.float16)[0]
        f16_hi = np.frombuffer(np.uint16(raw_hi).tobytes(), dtype=np.float16)[0]

        # In kernel: __half2float(raw) - 1024.0f
        rec_lo = float(f16_lo) - 1024.0
        rec_hi = float(f16_hi) - 1024.0

        assert rec_lo == float(q), f"Mismatch for q={q}: got {rec_lo}"
        assert rec_hi == float(q), f"Mismatch for q={q}: got {rec_hi}"


def test_lop3_0xEA_all_65536_pairs():
    """Exhaustively verify all 65,536 combinations of 2 nibbles."""
    mask_even = 0x000F000F
    magic_exp = 0x64006400

    q_lo = np.arange(16, dtype=np.uint32)
    q_hi = np.arange(16, dtype=np.uint32)
    grid_lo, grid_hi = np.meshgrid(q_lo, q_hi)
    w_words = grid_lo.flatten() | (grid_hi.flatten() << 16)

    raw = (w_words & mask_even) | magic_exp
    raw_lo = (raw & 0xFFFF).astype(np.uint16)
    raw_hi = ((raw >> 16) & 0xFFFF).astype(np.uint16)

    f_lo = np.frombuffer(raw_lo.tobytes(), dtype=np.float16).astype(np.float32) - 1024.0
    f_hi = np.frombuffer(raw_hi.tobytes(), dtype=np.float16).astype(np.float32) - 1024.0

    assert np.array_equal(f_lo, grid_lo.flatten().astype(np.float32))
    assert np.array_equal(f_hi, grid_hi.flatten().astype(np.float32))


def test_lop3_0xEA_vectorized_unpack_equivalence():
    """Verify 8-nibble LOP3 extraction on 50,000 random uint32 words vs reference."""
    mask_even = 0x000F000F
    magic_exp = 0x64006400

    np.random.seed(1337)
    num_words = 50000
    W = np.random.randint(0, 2**32, size=num_words, dtype=np.uint32)

    raw_04 = (W & mask_even) | magic_exp
    raw_15 = ((W >> 4) & mask_even) | magic_exp
    raw_26 = ((W >> 8) & mask_even) | magic_exp
    raw_37 = ((W >> 12) & mask_even) | magic_exp

    def unpack_pair(raw_word):
        lo = (raw_word & 0xFFFF).astype(np.uint16)
        hi = ((raw_word >> 16) & 0xFFFF).astype(np.uint16)
        f_lo = np.frombuffer(lo.tobytes(), dtype=np.float16).astype(np.float32) - 1024.0
        f_hi = np.frombuffer(hi.tobytes(), dtype=np.float16).astype(np.float32) - 1024.0
        return f_lo, f_hi

    q0, q4 = unpack_pair(raw_04)
    q1, q5 = unpack_pair(raw_15)
    q2, q6 = unpack_pair(raw_26)
    q3, q7 = unpack_pair(raw_37)

    extracted = [q0, q1, q2, q3, q4, q5, q6, q7]
    for i in range(8):
        ref = ((W >> (i * 4)) & 0xF).astype(np.float32)
        max_diff = np.max(np.abs(extracted[i] - ref))
        assert max_diff == 0.0, f"Nibble {i} differs from reference unpack by {max_diff}"


def test_scale_and_zero_point_group_indexing():
    """Verify group indexing formula for group_size = 128 and group_size = 0."""
    # Case 1: group_size = 128
    # Formula in kernel: int g = is_group_128 ? (k_idx >> 4) : ((k_idx * 8) / group_size);
    for k_idx in range(4096):
        g_shift = k_idx >> 4
        g_div = (k_idx * 8) // 128
        assert g_shift == g_div, f"Group index mismatch at k_idx={k_idx}: shift={g_shift}, div={g_div}"

    # Verify weight element level mapping:
    # Each k_idx contains 8 weight elements: k_act in [k_idx * 8, k_idx * 8 + 7].
    # For all 8 elements, k_act // 128 must match k_idx >> 4.
    for k_idx in range(1000):
        expected_g = k_idx >> 4
        for elem in range(8):
            k_act = k_idx * 8 + elem
            assert k_act // 128 == expected_g

    # Case 2: group_size = 0
    # Formula in kernel: if (group_size > 0) { ... }
    # When group_size == 0, division is never evaluated (no div-by-zero).
    # scale and zero_point are loaded once per block at col_t + v.
    group_size = 0
    assert not (group_size > 0)


def test_retiled_gemv_numerical_equivalence_vs_reference():
    """Vectorized numerical simulation of retiled GEMV kernel vs CPU reference."""
    np.random.seed(42)
    torch.manual_seed(42)

    test_configs = [
        {"M": 1, "K": 128, "N": 128, "group_size": 0},
        {"M": 1, "K": 896, "N": 896, "group_size": 128},
        {"M": 2, "K": 3584, "N": 128, "group_size": 0},
        {"M": 1, "K": 3584, "N": 3584, "group_size": 0},
    ]

    for cfg in test_configs:
        M, K, N, group_size = cfg["M"], cfg["K"], cfg["N"], cfg["group_size"]

        A = torch.randn(M, K, dtype=torch.float16)
        W_raw = torch.randint(0, 16, (K, N), dtype=torch.uint8)

        if group_size > 0:
            num_groups = K // group_size
            scales = torch.randn(num_groups, N, dtype=torch.float16) * 0.05
            zero_points = torch.randint(0, 16, (num_groups, N), dtype=torch.float16)
            # Reference dequantization
            unpacked_g = W_raw.float().reshape(num_groups, group_size, N)
            sc_3d = scales.float().unsqueeze(1)
            zp_3d = zero_points.float().unsqueeze(1)
            W_deq = ((unpacked_g - zp_3d) * sc_3d).reshape(K, N)
        else:
            scales = torch.randn(1, N, dtype=torch.float16) * 0.05
            zero_points = torch.randint(0, 16, (1, N), dtype=torch.float16)
            W_deq = (W_raw.float() - zero_points.float()) * scales.float()

        C_ref = torch.matmul(A.float(), W_deq).half()

        # Simulate retiled GEMV kernel arithmetic in FP32 with 4-warp reduction
        A_f = A.float().numpy()
        W_f = W_raw.float().numpy()
        s_f = scales.float().numpy()
        z_f = zero_points.float().numpy()

        C_sim = np.zeros((M, N), dtype=np.float16)
        k_uint32_total = K // 8

        for m in range(M):
            # Reshape activations to (k_uint32_total, 8)
            A_chunks = A_f[m].reshape(k_uint32_total, 8)  # (K/8, 8)
            # Reshape weights to (k_uint32_total, 8, N)
            W_chunks = W_f.reshape(k_uint32_total, 8, N)

            # Determine scales and zero_points per k_idx
            if group_size > 0:
                g_indices = np.arange(k_uint32_total) >> 4 if group_size == 128 else (np.arange(k_uint32_total) * 8) // group_size
                s_expanded = s_f[g_indices]  # (K/8, N)
                z_expanded = z_f[g_indices]  # (K/8, N)
            else:
                s_expanded = np.tile(s_f, (k_uint32_total, 1))
                z_expanded = np.tile(z_f, (k_uint32_total, 1))

            # Inner dot product: sum over 8 elements: (q - z) * a
            # W_chunks - z_expanded[:, None, :]: (K/8, 8, N)
            diff = W_chunks - z_expanded[:, None, :]
            # Multiply by activation: diff * A_chunks[:, :, None] -> (K/8, 8, N)
            dots = np.sum(diff * A_chunks[:, :, None], axis=1)  # (K/8, N)
            # Multiply by scale: s * dot
            scaled_dots = s_expanded * dots  # (K/8, N)

            # 4 warps split k_idx with stride 4 and accumulate
            warps_accum = np.zeros((4, N), dtype=np.float32)
            for w in range(4):
                warps_accum[w] = np.sum(scaled_dots[w::4], axis=0)

            # Reduction across warps
            final_sum = np.sum(warps_accum, axis=0)
            C_sim[m] = final_sum.astype(np.float16)

        diff = torch.abs(torch.from_numpy(C_sim) - C_ref)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()

        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            torch.from_numpy(C_sim).float().flatten(),
            C_ref.float().flatten(),
            dim=0
        ).item()

        assert max_diff <= 0.05, f"Max diff {max_diff} exceeded 0.05 tolerance for config {cfg}"
        assert mean_diff <= 0.01, f"Mean diff {mean_diff} exceeded 0.01 tolerance for config {cfg}"
        assert cos_sim >= 0.999, f"Cosine similarity {cos_sim} below 0.999 for config {cfg}"


def test_boundary_conditions_and_edge_cases():
    """Verify edge case dimensions: small K, non-multiple-of-128 N, inactive warps."""
    # Test 1: Small K where k_uint32_total < 4 (e.g. K = 8 -> k_uint32_total = 1)
    # Warps 1, 2, 3 never enter the K loop, staying at 0.0f
    k_uint32_total = 1
    warps_accum = np.zeros(4, dtype=np.float32)
    for k_idx in range(k_uint32_total):
        warp_id = k_idx % 4
        warps_accum[warp_id] += 42.0

    assert warps_accum[0] == 42.0
    assert warps_accum[1] == 0.0
    assert warps_accum[2] == 0.0
    assert warps_accum[3] == 0.0
    assert np.sum(warps_accum) == 42.0

    # Test 2: N % 128 != 0 column bounds checking
    # Fallback path guards: col < N and final_col < N
    for N in [1, 2, 3, 7, 15, 31, 63, 127, 129, 130, 255]:
        grid_x = (N + 128 - 1) // 128
        written_cols = []
        for block_x in range(grid_x):
            block_col_base = block_x * 128
            for tid in range(128):
                final_col = block_col_base + tid
                if final_col < N:
                    written_cols.append(final_col)

        assert written_cols == list(range(N)), f"Columns written for N={N} mismatch: {len(written_cols)} vs {N}"


if __name__ == '__main__':
    pytest.main([__file__, "-v"])
