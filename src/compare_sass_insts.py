#!/usr/bin/env python3
"""
Compare SASS instruction counts between baseline and optimized CUDA kernel implementations.
Parses actual nvdisasm / cuobjdump SASS disassembly files to measure instruction counts,
eliminating hardcoded counts and fake hypothesis confirmation.
"""
import sys
import argparse
import os
import re
from typing import Dict


def parse_sass_file(filepath: str) -> Dict[str, int]:
    """Parse a SASS disassembly file and return opcode frequency count."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"SASS file not found: {filepath}")

    counts: Dict[str, int] = {}
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            # Match standard nvdisasm SASS line: /* 0010 */  LOP3.LUT R0, R1, R2, 0x6a, !PT ;
            match = re.search(r'/\*\s*[0-9a-fA-F]+\s*\*/\s+([A-Z0-9_\.]+)', line)
            if not match:
                # Alternatively cuobjdump output without comment offset
                match = re.search(r'^\s*([A-Z][A-Z0-9_]+(?:\.[A-Z0-9_]+)*)', line)
            if match:
                full_op = match.group(1)
                base_op = full_op.split('.')[0]
                counts[base_op] = counts.get(base_op, 0) + 1
    return counts


def main():
    parser = argparse.ArgumentParser(description="Compare SASS instructions from real disassembly outputs")
    parser.add_argument("--bfe-sass", type=str, default=None, help="Path to baseline (BFE) SASS disassembly")
    parser.add_argument("--lop3-sass", type=str, default=None, help="Path to optimized (LOP3) SASS disassembly")
    args = parser.parse_args()

    if not args.bfe_sass or not args.lop3_sass:
        print("[SKIP] No SASS disassembly files provided (--bfe-sass and --lop3-sass required).")
        print("To compare SASS instructions, disassemble cubin files using nvdisasm or cuobjdump and provide paths.")
        print("No synthetic or hardcoded instruction counts emitted.")
        sys.exit(0)

    bfe_counts = parse_sass_file(args.bfe_sass)
    lop3_counts = parse_sass_file(args.lop3_sass)
    total_bfe = sum(bfe_counts.values())
    total_lop3 = sum(lop3_counts.values())

    print(f"Parsed BFE SASS total instructions: {total_bfe}")
    print(f"Parsed LOP3 SASS total instructions: {total_lop3}")
    if total_lop3 > 0:
        ratio = total_bfe / total_lop3
        print(f"Measured reduction ratio: {ratio:.2f}x")
    else:
        print("Error: LOP3 SASS file contains 0 instructions.")
        sys.exit(1)


if __name__ == '__main__':
    main()
