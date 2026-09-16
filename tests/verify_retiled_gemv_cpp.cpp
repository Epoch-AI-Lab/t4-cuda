#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <vector>

// Define CUDA-like vector types for C++ simulation
struct uint4_sim {
    uint32_t x, y, z, w;
};

struct float4_sim {
    float x, y, z, w;
};

// Simulation of half precision for CPU verification
// FP16 bit layout: 1 sign, 5 exponent (bias 15), 10 mantissa
static inline float half_bits_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        // Subnormal
        float val = std::ldexp((float)mant / 1024.0f, -14);
        return sign ? -val : val;
    } else if (exp == 31) {
        return mant ? NAN : (sign ? -INFINITY : INFINITY);
    }
    float val = (1.0f + (float)mant / 1024.0f) * std::ldexp(1.0f, (int)exp - 15);
    return sign ? -val : val;
}

static inline uint16_t float_to_half_bits(float f) {
    // Basic conversion for test numbers
    union { float f; uint32_t u; } u = { f };
    uint32_t sign = (u.u >> 16) & 0x8000;
    int32_t exp = ((u.u >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (u.u >> 13) & 0x3FF;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | (exp << 10) | mant);
}

// Bitwise LOP3 0xEA emulation: (A & B) | C
static inline uint32_t lop3_0xEA(uint32_t a, uint32_t b, uint32_t c) {
    return (a & b) | c;
}

int main() {
    printf("=== Starting C++ GEMV Kernel & Numerics Empirical Stress Harness ===\n");

    // -------------------------------------------------------------
    // Test 1: LOP3 0xEA Nibble Dequantization
    // -------------------------------------------------------------
    const uint32_t mask_even = 0x000F000F;
    const uint32_t magic_exp = 0x64006400; // 1024.0 in FP16 x 2

    for (uint32_t q = 0; q < 16; ++q) {
        uint32_t word = q | (q << 16);
        uint32_t raw = lop3_0xEA(word, mask_even, magic_exp);

        uint16_t lo_bits = raw & 0xFFFF;
        uint16_t hi_bits = (raw >> 16) & 0xFFFF;

        float f_lo = half_bits_to_float(lo_bits) - 1024.0f;
        float f_hi = half_bits_to_float(hi_bits) - 1024.0f;

        assert(std::fabs(f_lo - (float)q) < 1e-5f);
        assert(std::fabs(f_hi - (float)q) < 1e-5f);
    }
    printf("[PASS] LOP3 0xEA: All 16 nibble values produce exact integer floats.\n");

    // -------------------------------------------------------------
    // Test 2: Full 8-nibble extraction on 1,000,000 random words
    // -------------------------------------------------------------
    srand(12345);
    for (int iter = 0; iter < 1000000; ++iter) {
        uint32_t W = ((uint32_t)rand() << 16) | ((uint32_t)rand() & 0xFFFF);

        uint32_t raw_04 = lop3_0xEA(W, mask_even, magic_exp);
        uint32_t raw_15 = lop3_0xEA(W >> 4, mask_even, magic_exp);
        uint32_t raw_26 = lop3_0xEA(W >> 8, mask_even, magic_exp);
        uint32_t raw_37 = lop3_0xEA(W >> 12, mask_even, magic_exp);

        float extracted[8];
        extracted[0] = half_bits_to_float(raw_04 & 0xFFFF) - 1024.0f;
        extracted[4] = half_bits_to_float((raw_04 >> 16) & 0xFFFF) - 1024.0f;
        extracted[1] = half_bits_to_float(raw_15 & 0xFFFF) - 1024.0f;
        extracted[5] = half_bits_to_float((raw_15 >> 16) & 0xFFFF) - 1024.0f;
        extracted[2] = half_bits_to_float(raw_26 & 0xFFFF) - 1024.0f;
        extracted[6] = half_bits_to_float((raw_26 >> 16) & 0xFFFF) - 1024.0f;
        extracted[3] = half_bits_to_float(raw_37 & 0xFFFF) - 1024.0f;
        extracted[7] = half_bits_to_float((raw_37 >> 16) & 0xFFFF) - 1024.0f;

        for (int i = 0; i < 8; ++i) {
            uint32_t ref = (W >> (i * 4)) & 0xF;
            assert(extracted[i] == (float)ref);
        }
    }
    printf("[PASS] 1,000,000 words (8,000,000 nibbles) verified bit-exact with canonical unpack.\n");

    // -------------------------------------------------------------
    // Test 3: 128-bit uint4 Alignment Analysis
    // -------------------------------------------------------------
    // W_packed shape: (K/8, N) in row-major layout
    // Row k_idx starts at: W_packed + k_idx * N
    // Thread lane_id loads uint4 at: w_row_u4[lane_id] = W_packed + k_idx * N + block_col_base + lane_id * 4
    // Check if (k_idx * N + block_col_base + lane_id * 4) * sizeof(uint32_t) is 16-byte aligned.
    bool all_aligned = true;
    for (int N : {128, 256, 3584, 9472, 18944}) {
        for (int blockIdx_x = 0; blockIdx_x < N / 128; ++blockIdx_x) {
            int block_col_base = blockIdx_x * 128;
            for (int k_idx = 0; k_idx < 100; ++k_idx) {
                for (int lane_id = 0; lane_id < 32; ++lane_id) {
                    uintptr_t elem_offset = (uintptr_t)(k_idx * N + block_col_base + lane_id * 4);
                    uintptr_t byte_offset = elem_offset * sizeof(uint32_t);
                    if (byte_offset % 16 != 0) {
                        all_aligned = false;
                    }
                }
            }
        }
    }
    assert(all_aligned);
    printf("[PASS] All canonical Qwen2.5-7B shapes guarantee 16-byte alignment for uint4 loads.\n");

    // -------------------------------------------------------------
    // Test 4: Scale and Zero Point Group Indexing
    // -------------------------------------------------------------
    for (int k_idx = 0; k_idx < 4096; ++k_idx) {
        int g_fast = k_idx >> 4;
        int g_exact = (k_idx * 8) / 128;
        assert(g_fast == g_exact);
    }
    printf("[PASS] Group indexing identity (k_idx >> 4) == ((k_idx * 8) / 128) verified.\n");

    // -------------------------------------------------------------
    // Test 5: Boundary Columns and Reduction Correctness
    // -------------------------------------------------------------
    for (int N : {1, 7, 15, 63, 127, 129, 130, 255}) {
        int num_blocks = (N + 128 - 1) / 128;
        std::vector<bool> col_written(N, false);
        for (int bx = 0; bx < num_blocks; ++bx) {
            int block_col_base = bx * 128;
            for (int tid = 0; tid < 128; ++tid) {
                int final_col = block_col_base + tid;
                if (final_col < N) {
                    assert(!col_written[final_col]);
                    col_written[final_col] = true;
                }
            }
        }
        for (int c = 0; c < N; ++c) {
            assert(col_written[c]);
        }
    }
    printf("[PASS] Boundary column masking strictly writes each column [0..N-1] exactly once.\n");

    printf("=== ALL C++ EMPIRICAL CHECKS PASSED SUCCESSFULLY ===\n");
    return 0;
}
