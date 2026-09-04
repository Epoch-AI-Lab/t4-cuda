def simulate_smem_access():
    warp_size = 32
    banks = 32
    bytes_per_bank = 4
    bytes_per_thread = 16  # 128-bit vector load (8 FP16 elements)
    words_per_thread = bytes_per_thread // bytes_per_bank  # 4 words per thread

    # Simulate shared memory 2D matrix tile transposed load:
    # 32 threads in a warp access a 32x32 half-precision tile (each row is 64 bytes = 16 banks).
    # When loading column-wise (transposed read):
    # Without swizzling, multiple threads in the same warp map to identical banks.

    # 1. Without Swizzling
    unswizzled_conflicts = 0
    # A 128-bit load executes across 4 phases (1 word per phase per thread)
    for phase in range(words_per_thread):
        bank_counts = [0] * banks
        for thread_idx in range(warp_size):
            # Transposed access: thread_idx selects row, phase selects column offset
            byte_addr = (thread_idx * 64) + (phase * bytes_per_bank)
            bank_id = (byte_addr // bytes_per_bank) % banks
            bank_counts[bank_id] += 1
        
        max_conflict_phase = max(bank_counts)
        if max_conflict_phase > 1:
            unswizzled_conflicts += (max_conflict_phase - 1)

    # 2. With XOR Swizzling
    # Standard CUTLASS XOR swizzle pattern: bank = (bank ^ (row >> 0)) % banks
    swizzled_conflicts = 0
    for phase in range(words_per_thread):
        bank_counts = [0] * banks
        for thread_idx in range(warp_size):
            row = thread_idx
            raw_bank = ((row * 64) + (phase * bytes_per_bank)) // bytes_per_bank
            # Apply 128-bit XOR swizzle
            swizzled_bank = (raw_bank ^ (row % banks)) % banks
            bank_counts[swizzled_bank] += 1
        
        max_conflict_phase = max(bank_counts)
        if max_conflict_phase > 1:
            swizzled_conflicts += (max_conflict_phase - 1)

    conflict_reduction = 100.0 if unswizzled_conflicts > 0 and swizzled_conflicts == 0 else (
        (unswizzled_conflicts - swizzled_conflicts) / unswizzled_conflicts * 100.0
    )

    print("=== Shared Memory Bank Conflict Simulation (Turing SM 7.5) ===")
    print(f"Warp size: {warp_size}, Banks: {banks}, Vector fetch: {bytes_per_thread} bytes/thread")
    print(f"Unswizzled bank conflicts per tile access: {unswizzled_conflicts}")
    print(f"XOR-swizzled bank conflicts:               {swizzled_conflicts}")
    print(f"Conflict elimination:                      {conflict_reduction:.1f}%")

if __name__ == "__main__":
    simulate_smem_access()

