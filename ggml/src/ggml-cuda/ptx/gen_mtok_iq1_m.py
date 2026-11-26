#!/usr/bin/env python3
"""Writes mtok-iq1_m.ptx: IQ1_M's fast multi-token expert kernel (C1138, code row 28r4) as hand-written PTX, the whole
kernel of mmvq-pascal-mtok-iq1_m-fast.cu for the products Qwen3.8-Flash-Next's verify pass makes (the fused gate/up
launch with SWIGLU, rows of 2,560 weights: 10 super-blocks of 56 bytes, 2 lanes a row, 5 iterations; no shared expert,
no biases).

Every instruction is written here, in gen_mtok_iq2_xxs.py or in gen_mtok_iq2_s.py, whose parts this script shares:
the parameters, the block's group, the activations' staging (in IQ1_M's order, stage_sel), the skipped slots, the row
loop's frame and the SiLU epilogue are gen_mtok_iq2_xxs.py's (row_start without kappa, segments in its direct mode,
row_end with IQ1_M's d); the two DP4A chains, their exact FFMAs and the scale floats are gen_mtok_iq2_s.py's (sums,
scale_floats). This script brings IQ1_M's code table (the grid's 2,048 entries and the same with bit 2 of every nibble
set: 4,096 words, one copy), its decoding (per 8 weights one PRMT of the qs byte and the qh nibble, one BFI of their 12
bits into the table's byte address, the lookup, the value PRMTs), its scale bytes (two 3-bit scales per 32 weights) and
its d (the four top nibbles of the scale words). The float operations are the CUDA C kernel's, each exact, in its order
and with the forms nvcc gives them under -use_fast_math, so the results are bit-identical to it and to the old kernel.
Run: python3 -I gen_mtok_iq1_m.py > mtok-iq1_m.ptx (from this folder).

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

SB = 56                        # bytes per super-block
NENT = 2048                    # grid entries
NEG = 0x44444444               # bit 2 of every nibble: a negative delta's codes


def grid():
    """ggml's iq1s_grid_gpu: per entry grid + 1 (0, 1, 2) as nibbles, weights 0-3 in the low nibbles of bytes 0-3 and
    4-7 in the high ones."""
    src = open(os.path.join(HERE, "..", "..", "ggml-common.h")).read()
    m = re.search(r"GGML_TABLE_BEGIN\(uint32_t, iq1s_grid_gpu, NGRID_IQ1S\)(.*?)GGML_TABLE_END", src, re.S)
    vals = [int(x, 16) for x in re.findall(r"0x([0-9a-fA-F]{8})", m.group(1))]
    assert len(vals) == NENT
    for g in vals:
        assert all(((g >> (4 * n)) & 0xF) <= 2 for n in range(8))  # bit 2 of every nibble free for the sign
    return vals


def table(k):
    """The code table in shared memory, from the global table mtok_iq1m_grid: thread tid copies 16-byte units tid + 128 i
    (all 4 loads first), then the same words with bit 2 of every nibble set 8 KB further."""
    b32 = lambda *n: k.r(".b32", *n)
    b64 = lambda *n: k.r(".b64", *n)
    k.c("the code table: the grid's 2,048 entries (grid + 1 as nibbles), then the same with bit 2 of every nibble set"
        " (a negative delta): the 11-bit grid index and the delta's sign above it address it")
    gw = [f"gw{i}" for i in range(16)]
    b32("tabb", "sa", *gw)
    b64("cods")
    k("mov.u32 tabb, tab;")
    k("mov.u64 cods, mtok_iq1m_grid;")
    k("mul.wide.u32 o64, tid, 16;")
    k("add.u64 a64, cods, o64;")
    for i in range(4):
        k(f"ld.global.nc.v4.u32 {{{', '.join(gw[4 * i:4 * i + 4])}}}, [a64+{2048 * i}];")
    k("shl.b32 sa, tid, 4;")
    k("add.u32 sa, sa, tabb;")
    for i in range(4):
        k(f"st.shared.v4.u32 [sa+{2048 * i}], {{{', '.join(gw[4 * i:4 * i + 4])}}};")
    for g in gw:
        k(f"or.b32 {g}, {g}, 0x{NEG:08X};")
    for i in range(4):
        k(f"st.shared.v4.u32 [sa+{4 * NENT + 2048 * i}], {{{', '.join(gw[4 * i:4 * i + 4])}}};")


def consts(k):
    b32 = lambda *n: k.r(".b32", *n)
    b32("KS0", "KS1", "K01", "KF0")
    k("mov.b32 KS0, 0x000E000E;   // 2 s of the scales at bits 0-2 of each 16-bit scale word, after a shift")
    k("mov.b32 KS1, 0x0E000E00;   // 2 s of the scales at bits 6-8 (and 9-11)")
    k("mov.b32 K01, 0x01010101;")
    k("mov.b32 KF0, 0xF0F0F0F0;   // the high nibbles of every byte")


def scale_bytes(k, m, sw):
    """Four blocks' scales (sw: two 16-bit scale words, the low one blocks 4h and 4h + 1 with ls0, ls1 at bits 0-2, 3-5
    and 6-8, 9-11, the high one blocks 4h + 2 and 4h + 3) as bytes 2 s + 1 of e0_<m> (ls0) and e1_<m> (ls1), byte b
    block 4h + b."""
    k(f"shl.b32 ts, {sw}, 1;")
    k("lop3.b32 tz, ts, KS0, K01, 0xEA;        // ((sw << 1) & 0x000E000E) | 0x01010101: 2 ls0 + 1, blocks 4h, 4h + 2")
    k(f"shl.b32 ts, {sw}, 3;")
    k(f"lop3.b32 e0_{m}, ts, KS1, tz, 0xEA;       // | ((sw << 3) & 0x0E000E00): blocks 4h + 1, 4h + 3")
    k(f"shr.b32 ts, {sw}, 2;")
    k("lop3.b32 tz, ts, KS0, K01, 0xEA;        // ((sw >> 2) & 0x000E000E) | 0x01010101: 2 ls1 + 1, blocks 4h, 4h + 2")
    k(f"lop3.b32 e1_{m}, {sw}, KS1, tz, 0xEA;    // | (sw & 0x0E000E00): blocks 4h + 1, 4h + 3")


def dscale(k, m):
    """Matrix m's super-block factor 0.125 d into fd: d's four nibbles are the top nibbles of the scale words (bits
    12-15 and 28-31 of w<m>_12 and w<m>_13), d = n0 | n1 << 4 | n2 << 8 | n3 << 12 (the old kernel's sblock_scale)."""
    k(f"prmt.b32 t0, w{m}_12, w{m}_13, 0x7531;  // bytes 1, 3 of both words: n0 .. n3 in their high nibbles")
    k("shl.b32 t1, t0, 4;")
    k("lop3.b32 t0, t0, t1, KF0, 0xE4;         // high nibbles from t0, low from t1: bytes 1, 3 = n1 n0, n3 n2")
    k("prmt.b32 dsh, t0, ZERO, 0x0031;         // d in the low half")
    k("mov.b32 {h0, h1}, dsh;")
    k("cvt.f32.f16 fd, h0;")
    k("mul.ftz.f32 fd, fd, 0f3E000000;")


def body(k, nt):
    """The rows of the block for groups of exactly nt members (mtok_iq1m_fast::rows<true, nt>)."""
    b32 = lambda *n: k.r(".b32", *n)
    f32 = lambda *n: k.r(".f32", *n)
    base.row_start(k, nt, unit=8, kappa=False)
    base.segments(k, nt, SB, 8, 7, SB, 14, direct=True)
    b32("px", "ea", "cw", "sh", "ia", "ib1", "ts", "tz", "qh4_0", "qh4_1")
    b32(*[f"q{m}_{n}" for m in range(2) for n in range(8)])
    b32(*[f"ua{n}" for n in range(4)], *[f"ub{n}" for n in range(4)])
    b32("e0_0", "e0_1", "e1_0", "e1_1")
    f32("fl0_0", "fl0_1", "fl1_0", "fl1_1", "nc1_0", "nc1_1", "fa", "fi", "fp", "fr")
    for h in range(2):
        k.c(f"blocks {4 * h}..{4 * h + 3}: the scales as bytes 2 s + 1")
        for m in range(2):
            scale_bytes(k, m, f"w{m}_{12 + h}")
        for b in range(4):
            ib = 4 * h + b
            k.c(f"block {ib}: weights {32 * ib}..{32 * ib + 31}")
            for m in range(2):
                qs, qh = f"w{m}_{ib}", f"w{m}_{8 + ib // 2}"
                if ib % 2 == 0:
                    k(f"shr.b32 qh4_{m}, {qh}, 4;            // the qh nibbles of odd groups as low nibbles")
                for l in range(4):
                    src = qh if l % 2 == 0 else f"qh4_{m}"
                    byte = 2 * (ib % 2) + l // 2
                    k(f"prmt.b32 px, {qs}, {src}, 0x{(4 + byte) << 4 | l:04x};  // group {l}: the qs byte, the qh nibble above it")
                    k("bfi.b32 ea, px, ZERO, 2, 12;      // the 11-bit grid index and the delta's sign: the entry's word")
                    k("add.u32 ea, ea, tabb;")
                    k("ld.shared.u32 cw, [ea];")
                    k(f"prmt.b32 q{m}_{2 * l}, LP, LN, cw;      // weights 0, 4, 1, 5: 8 (grid + delta)")
                    k("shr.b32 sh, cw, 16;")
                    k(f"prmt.b32 q{m}_{2 * l + 1}, LP, LN, sh;      // weights 2, 6, 3, 7")
                two.scale_floats(k, m, b)
            two.sums(k, nt, ib)
    base.row_end(k, nt, dscale=dscale)


class IQ1M:
    prefix = "iq1m"
    copies = 1
    tab_words = 2 * NENT
    unit_off = SB
    stage_sel = ("0x5140", "0x7362")
    stage_doc = "(0, 4, 1, 5), (2, 6, 3, 7)"
    lut = (("0x000901F9", "bytes -7, 1, 9, 0: 8 (grid + 1/8) by code grid + 1"),
           ("0x0007FFF7", "-9, -1, 7, 0: 8 (grid - 1/8)"))
    word_pick = False
    table = staticmethod(table)
    consts = staticmethod(consts)
    body = staticmethod(body)


def main():
    g = grid()
    print("//")
    print("// mtok-iq1_m.ptx: IQ1_M's fast multi-token expert kernel (C1138) in hand-written PTX, written by")
    print("// gen_mtok_iq1_m.py (do not edit; edit the script). See mmvq-pascal-mtok-iq1_m-fast.cu for the method")
    print("// and mmvq-pascal-mtok-iq-ptx.cu for how it is loaded and which products it takes.")
    print("//")
    print(".version 7.0")
    print(".target sm_61")
    print(".address_size 64")
    print()
    print("// the IQ1_M grid (ggml's iq1s_grid_gpu): entry e's eight values grid + 1 (0, 1, 2) as nibbles, weights 0-3 in")
    print("// the low nibbles of bytes 0-3, 4-7 in the high ones")
    print(f".global .align 16 .b32 mtok_iq1m_grid[{NENT}] = {{")
    for i in range(0, NENT, 8):
        print("    " + ", ".join(f"0x{c:08x}" for c in g[i:i + 8]) + ("," if i + 8 < NENT else ""))
    print("};")
    print(".extern .shared .align 16 .b8 mtok_iq1m_dyn[];")
    print()
    for ntm in (2, 4, 8):
        print(base.entry(ntm, IQ1M))
        print()


if __name__ == "__main__":
    main()
