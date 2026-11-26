#!/usr/bin/env python3
"""Writes mtok-iq2_xs.ptx: IQ2_XS's fast multi-token expert kernel (C1138, code row 28r3) as hand-written PTX, the whole
kernel of mmvq-pascal-mtok-iq2_xs-fast.cu for the products Qwen3.8-Flash-Next's verify pass makes (the fused gate/up
launch with SWIGLU, rows of 2,560 weights: 10 super-blocks of 74 bytes, 2 lanes a row, 5 iterations; no shared expert,
no biases).

Every instruction is written here, in gen_mtok_iq2_xxs.py or in gen_mtok_iq2_s.py, whose parts this script shares:
the parameters, the block's group, the activations' staging, the skipped slots, the row loop's frame, the row loads and
the SiLU epilogue are gen_mtok_iq2_xxs.py's (row_start, segments, row_end); the two scales per 32 weights (their bytes,
their floats, the two DP4A chains and the exact FFMAs) and the value PRMTs are gen_mtok_iq2_s.py's (scale_bytes,
scale_floats, sums, values). This script brings IQ2_XS's code table (512 entries in 8 copies), its row loads (10 8-byte
units a lane: lane 1's super-block sits 2 words further into a 16-byte unit than lane 0's, so 16-byte loads would break
the warp-uniform choice of words) and its decoding: per 16-bit entry the 9-bit index as one BFI into the lane's copy
offset (an entry in a word's high half shifted down first) and the 7 sign bits and their parity by IQ2_XXS's path. The
float operations are the CUDA C kernel's, each exact, in its order and with the forms nvcc gives them under
-use_fast_math, so the results are bit-identical to it and to the old kernel. Run: python3 -I gen_mtok_iq2_xs.py >
mtok-iq2_xs.ptx (from this folder).

Parameters (byte offsets, checked by static_asserts in mmvq-pascal-mtok-iq-ptx.cu): mtok_iq::args (168 bytes) and
ggml_cuda_mm_fusion_args_device (80 bytes; the gate matrix at 8).
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_mtok_iq2_xxs as base  # noqa: E402
import gen_mtok_iq2_s as two     # noqa: E402

PI = base.PI                   # nibble n of a selector word holds weight PI[n] of its group of 8
SB = 74                        # bytes per super-block
NENT = 512                     # grid entries
COPIES = 8                     # copies of the code table, interleaved by word


def grid_codes():
    src = open(os.path.join(HERE, "..", "..", "ggml-common.h")).read()
    m = re.search(r"GGML_TABLE_BEGIN\(uint64_t, iq2xs_grid, 512\)(.*?)GGML_TABLE_END", src, re.S)
    vals = [int(x, 16) for x in re.findall(r"0x([0-9a-fA-F]{16})", m.group(1))]
    assert len(vals) == NENT
    codes = []
    for g in vals:
        c = 0
        for n in range(8):
            v = (g >> (8 * PI[n])) & 0xFF
            assert v in (0x08, 0x19, 0x2B)
            c |= {0x08: 0, 0x19: 1, 0x2B: 2}[v] << (4 * n)
        codes.append(c)
    return codes


def table(k):
    """The code table in shared memory, from the global table mtok_iq2xs_codes: thread tid copies entries tid + 128 i,
    each as two 16-byte stores of its 8 copies (all 4 loads first)."""
    b32 = lambda *n: k.r(".b32", *n)
    b64 = lambda *n: k.r(".b64", *n)
    k.c("the code table: entry e's eight codes (0, 1, 2 for 8, 25, 43) as nibbles, 8 copies interleaved by word")
    b32("tabb", "e", "sa", *[f"cw{i}" for i in range(4)])
    b64("cods")
    k("mov.u32 tabb, tab;")
    k("mov.u64 cods, mtok_iq2xs_codes;")
    k("mul.wide.u32 o64, tid, 4;")
    k("add.u64 a64, cods, o64;")
    for i in range(4):
        k(f"ld.global.nc.u32 cw{i}, [a64+{512 * i}];")
    k("shl.b32 sa, tid, 5;")
    k("add.u32 sa, sa, tabb;")
    for i in range(4):
        k(f"st.shared.v4.u32 [sa+{4096 * i}], {{cw{i}, cw{i}, cw{i}, cw{i}}};")
        k(f"st.shared.v4.u32 [sa+{4096 * i + 16}], {{cw{i}, cw{i}, cw{i}, cw{i}}};")


def consts(k):
    b32 = lambda *n: k.r(".b32", *n)
    b32("K1E", "K01", "KEXP", "coff")
    k("and.b32 coff, lane, 7;")
    k("shl.b32 coff, coff, 2;      // the lane's copy of the code table: bits 2-4 of its word address")
    k("mov.b32 K1E, 0x1E1E1E1E;   // 2 s in every byte")
    k("mov.b32 K01, 0x01010101;")
    k("mov.b32 KEXP, 0x4B000000;  // the exponent of 2^23: a byte b under it is the float 2^23 + b")


def body(k, nt):
    """The rows of the block for groups of exactly nt members (mtok_iq2xs_fast::rows<true, nt>)."""
    b32 = lambda *n: k.r(".b32", *n)
    f32 = lambda *n: k.r(".f32", *n)
    base.row_start(k, nt, unit=8)
    base.segments(k, nt, SB, 8, 10, 72, 18)
    b32("s7", "t6", "x", "pc", "sn", "hi", "ea", "cw", "sv", "sh", "ia", "ib1", "ts")
    b32(*[f"q{m}_{n}" for m in range(2) for n in range(8)])
    b32(*[f"ua{n}" for n in range(4)], *[f"ub{n}" for n in range(4)])
    b32("e0_0", "e0_1", "e1_0", "e1_1")
    f32("fl0_0", "fl0_1", "fl1_0", "fl1_1", "nc1_0", "nc1_1", "fa", "fi", "fp", "fr")
    for h in range(2):
        k.c(f"blocks {4 * h}..{4 * h + 3}: the scales as bytes 2 s + 1")
        for m in range(2):
            two.scale_bytes(k, m, f"w{m}_{16 + h}")
        for b in range(4):
            ib = 4 * h + b
            k.c(f"block {ib}: weights {32 * ib}..{32 * ib + 31}")
            for m in range(2):
                for l in range(4):
                    ew = f"w{m}_{2 * ib + l // 2}"  # the word of group l's 16-bit entry: the low half for l even
                    hb = 16 * (l & 1)
                    k(f"bfe.u32 s7, {ew}, {hb + 9}, 7;")
                    k("shl.b32 t6, s7, 6;")
                    k("lop3.b32 x, s7, t6, K3333, 0xA8;   // (s7 | s7 << 6) & 0x3333")
                    k("popc.b32 pc, s7;")
                    k("bfi.b32 x, pc, x, 13, 1;          // the eighth sign: the parity of the seven")
                    k("prmt.b32 sn, LS, ZERO, x;")
                    if hb:
                        k(f"shr.b32 hi, {ew}, 16;")
                        k("bfi.b32 ea, hi, coff, 5, 9;       // the 9-bit grid index at bits 5-13, the copy at 2-4")
                    else:
                        k(f"bfi.b32 ea, {ew}, coff, 5, 9;       // the 9-bit grid index at bits 5-13, the copy at 2-4")
                    k("add.u32 ea, ea, tabb;")
                    two.values(k, m, l)
                two.scale_floats(k, m, b)
            two.sums(k, nt, ib)
    base.row_end(k, nt)


class IQ2XS:
    prefix = "iq2xs"
    copies = COPIES
    tab_words = NENT * COPIES
    unit_off = 72
    table = staticmethod(table)
    consts = staticmethod(consts)
    body = staticmethod(body)


def main():
    codes = grid_codes()
    print("//")
    print("// mtok-iq2_xs.ptx: IQ2_XS's fast multi-token expert kernel (C1138) in hand-written PTX, written by")
    print("// gen_mtok_iq2_xs.py (do not edit; edit the script). See mmvq-pascal-mtok-iq2_xs-fast.cu for the method")
    print("// and mmvq-pascal-mtok-iq-ptx.cu for how it is loaded and which products it takes.")
    print("//")
    print(".version 7.0")
    print(".target sm_61")
    print(".address_size 64")
    print()
    print("// the IQ2_XS grid as codes: entry e's eight values (8, 25, 43 -> 0, 1, 2) as nibbles, nibble n weight")
    print(f"// {PI}[n] of the group")
    print(f".global .align 4 .b32 mtok_iq2xs_codes[{NENT}] = {{")
    for i in range(0, NENT, 8):
        print("    " + ", ".join(f"0x{c:08x}" for c in codes[i:i + 8]) + ("," if i + 8 < NENT else ""))
    print("};")
    print(".extern .shared .align 16 .b8 mtok_iq2xs_dyn[];")
    print()
    for ntm in (2, 4, 8):
        print(base.entry(ntm, IQ2XS))
        print()


if __name__ == "__main__":
    main()
