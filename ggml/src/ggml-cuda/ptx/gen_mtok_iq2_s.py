#!/usr/bin/env python3
"""Writes mtok-iq2_s.ptx: IQ2_S's fast multi-token expert kernel (C1138, code row 28r2) as hand-written PTX, the whole
kernel of mmvq-pascal-mtok-iq2_s-fast.cu for the products Qwen3.8-Flash-Next's verify pass makes (the fused gate/up
launch with SWIGLU, rows of 2,560 weights: 10 super-blocks of 82 bytes, 2 lanes a row, 5 iterations; no shared expert,
no biases).

Every instruction is written here or in gen_mtok_iq2_xxs.py, whose parts this script shares: the parameters, the
block's group, the activations' staging, the skipped slots, the row loop's frame and the SiLU epilogue are IQ2_XXS's;
this script brings IQ2_S's code table (1,024 entries in 4 copies), its row loads (6 units a lane) and its decoding (a
10-bit index from qs and qh, 8 explicit sign bits, two scales per 32 weights), with two instruction choices of its own
(BFI_INDEX, I2F_SCALES below). The float operations are the CUDA C kernel's, each exact, in its order and with the
forms nvcc gives them under -use_fast_math, so the results are bit-identical to it and to the old kernel. Run:
python3 -I gen_mtok_iq2_s.py > mtok-iq2_s.ptx (from this folder).

Parameters (byte offsets, checked by static_asserts in mmvq-pascal-mtok-iq-ptx.cu): mtok_iq::args (168 bytes) and
ggml_cuda_mm_fusion_args_device (80 bytes; the gate matrix at 8).
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_mtok_iq2_xxs as base  # noqa: E402

PI = base.PI                   # nibble n of a selector word holds weight PI[n] of its group of 8
ACT_RUN = base.ACT_RUN
NITER = base.NITER
SB = 82                        # bytes per super-block
NENT = 1024                    # grid entries
COPIES = 4                     # copies of the code table, interleaved by word
# Two choices nvcc did not make for the CUDA C version (each saves instructions; the other path is kept for a timing):
BFI_INDEX = True    # the table address: one BFI of the PRMT's 10 index bits into the lane's copy offset (bits 4-13 and
                    # 2-3 of the word address; ptxas folds the table's base into the load), not AND, shift and add
I2F_SCALES = True   # each scale's float: one conversion of its byte (ptxas folds the byte pick into I2F.F32.U8), not a
                    # PRMT under the exponent of 2^23 and an FADD; I2F runs at a quarter of the FP32 rate on Pascal


def grid_codes():
    src = open(os.path.join(HERE, "..", "..", "ggml-common.h")).read()
    m = re.search(r"GGML_TABLE_BEGIN\(uint64_t, iq2s_grid, 1024\)(.*?)GGML_TABLE_END", src, re.S)
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
    """The code table in shared memory, from the global table mtok_iq2s_codes: thread tid copies entries tid + 128 i,
    each as one 16-byte store of its 4 copies (all 8 loads first)."""
    b32 = lambda *n: k.r(".b32", *n)
    b64 = lambda *n: k.r(".b64", *n)
    k.c("the code table: entry e's eight codes (0, 1, 2 for 8, 25, 43) as nibbles, 4 copies interleaved by word")
    b32("tabb", "e", "sa", *[f"cw{i}" for i in range(8)])
    b64("cods")
    k("mov.u32 tabb, tab;")
    k("mov.u64 cods, mtok_iq2s_codes;")
    k("mul.wide.u32 o64, tid, 4;")
    k("add.u64 a64, cods, o64;")
    for i in range(8):
        k(f"ld.global.nc.u32 cw{i}, [a64+{512 * i}];")
    k("shl.b32 sa, tid, 4;")
    k("add.u32 sa, sa, tabb;")
    for i in range(8):
        k(f"st.shared.v4.u32 [sa+{2048 * i}], {{cw{i}, cw{i}, cw{i}, cw{i}}};")


def consts(k):
    b32 = lambda *n: k.r(".b32", *n)
    b32("K33", "K1E", "K01", "KEXP", "coff")
    k("and.b32 coff, lane, 3;")
    k("shl.b32 coff, coff, 2;      // the lane's copy of the code table: bits 2-3 of its word address")
    k("mov.b32 K33, 0x33333333;   // the sign pairs' bits in every nibble")
    k("mov.b32 K1E, 0x1E1E1E1E;   // 2 s in every byte")
    k("mov.b32 K01, 0x01010101;")
    k("mov.b32 KEXP, 0x4B000000;  // the exponent of 2^23: a byte b under it is the float 2^23 + b")


def scale_bytes(k, m, sw):
    """Four blocks' scales (sw: a byte per block, the low nibble for weights 0-15) as bytes 2 s + 1 of e0_<m> and e1_<m>
    (IQ2_S and IQ2_XS)."""
    k(f"shl.b32 ts, {sw}, 1;")
    k(f"lop3.b32 e0_{m}, ts, K1E, K01, 0xEA;  // ((sw << 1) & 0x1E1E1E1E) | 0x01010101: 2 ls0 + 1")
    k(f"shr.b32 ts, {sw}, 3;")
    k(f"lop3.b32 e1_{m}, ts, K1E, K01, 0xEA;  // ((sw >> 3) & 0x1E1E1E1E) | 0x01010101: 2 ls1 + 1")


def scale_floats(k, m, b):
    """Block b's two scales as floats fl0_<m>, fl1_<m> from byte b of e0_<m>, e1_<m>, and nc1_<m> = -fl1 * 1.5 * 2^23
    (IQ2_S and IQ2_XS)."""
    if I2F_SCALES:
        k(f"bfe.u32 ts, e0_{m}, {8 * b}, 8;")
        k(f"cvt.rn.f32.u32 fl0_{m}, ts;           // 2 ls0 + 1")
        k(f"bfe.u32 ts, e1_{m}, {8 * b}, 8;")
        k(f"cvt.rn.f32.u32 fl1_{m}, ts;           // 2 ls1 + 1")
    else:
        k(f"prmt.b32 ts, e0_{m}, KEXP, 0x{0x7440 | b:04x};")
        k(f"mov.b32 fl0_{m}, ts;")
        k(f"add.ftz.f32 fl0_{m}, fl0_{m}, 0fCB000000;  // 2 ls0 + 1")
        k(f"prmt.b32 ts, e1_{m}, KEXP, 0x{0x7440 | b:04x};")
        k(f"mov.b32 fl1_{m}, ts;")
        k(f"add.ftz.f32 fl1_{m}, fl1_{m}, 0fCB000000;  // 2 ls1 + 1")
    k(f"mul.ftz.f32 nc1_{m}, fl1_{m}, 0fCB400000;  // -(2 ls1 + 1) * 1.5 * 2^23")


def values(k, m, l):
    """Group l's eight int8 weights q<m>_<2l>, q<m>_<2l+1> from its code word (the table word at ea) and its sign
    nibbles sn."""
    k("ld.shared.u32 cw, [ea];")
    k("or.b32 sv, cw, sn;")
    k(f"prmt.b32 q{m}_{2 * l}, LP, LN, sv;")
    k("shr.b32 sh, sv, 16;")
    k(f"prmt.b32 q{m}_{2 * l + 1}, LP, LN, sh;")


def sums(k, nt, ib):
    """Block ib times every member: two DP4A chains per matrix (weights 0-15 and 16-31, each started at 1.5 * 2^23),
    (float) (ls0 * i0 + ls1 * i1) exactly, times the member's activation scale into sf<j>_<m> (IQ2_S and IQ2_XS)."""
    for j in range(nt):
        aj = "au" if j == 0 else f"au{j}"
        dj = "ad" if j == 0 else f"ad{j}"
        k(f"ld.shared.v4.u32 {{ua0, ua1, ua2, ua3}}, [{aj}+{32 * ib}];")
        k(f"ld.shared.v4.u32 {{ub0, ub1, ub2, ub3}}, [{aj}+{32 * ib + 16}];")
        k(f"ld.shared.f32 fa, [{dj}+{4 * ib}];")
        for m in range(2):
            k(f"dp4a.s32.s32 ia, q{m}_0, ua0, DBASE;")
            k(f"dp4a.s32.s32 ia, q{m}_1, ua1, ia;")
            k(f"dp4a.s32.s32 ia, q{m}_2, ua2, ia;")
            k(f"dp4a.s32.s32 ia, q{m}_3, ua3, ia;")
            k(f"dp4a.s32.s32 ib1, q{m}_4, ub0, DBASE;")
            k(f"dp4a.s32.s32 ib1, q{m}_5, ub1, ib1;")
            k(f"dp4a.s32.s32 ib1, q{m}_6, ub2, ib1;")
            k(f"dp4a.s32.s32 ib1, q{m}_7, ub3, ib1;")
            k("mov.b32 fi, ib1;")
            k(f"fma.rn.ftz.f32 fp, fl1_{m}, fi, nc1_{m};     // ls1 * i1, exactly")
            k("mov.b32 fi, ia;")
            k("add.ftz.f32 fi, fi, 0fCB400000;              // i0, exactly")
            k(f"fma.rn.ftz.f32 fr, fl0_{m}, fi, fp;         // (float) (ls0 * i0 + ls1 * i1), exactly")
            k(f"fma.rn.ftz.f32 sf{j}_{m}, fa, fr, sf{j}_{m};")


def body(k, nt):
    """The rows of the block for groups of exactly nt members (mtok_iq2s_fast::rows<true, nt>)."""
    b32 = lambda *n: k.r(".b32", *n)
    f32 = lambda *n: k.r(".f32", *n)
    base.row_start(k, nt)
    base.segments(k, nt, SB, 16, 6, 80, 20)
    b32("z", "z2", "x", "sn", "px", "ix", "ea", "cw", "sv", "sh", "ia", "ib1", "ts")
    b32(*[f"q{m}_{n}" for m in range(2) for n in range(8)])
    b32(*[f"ua{n}" for n in range(4)], *[f"ub{n}" for n in range(4)])
    b32(*[f"qh{m}_{l}" for m in range(2) for l in range(1, 4)], "e0_0", "e0_1", "e1_0", "e1_1")
    f32("fl0_0", "fl0_1", "fl1_0", "fl1_1", "nc1_0", "nc1_1", "fa", "fi", "fp", "fr")
    for h in range(2):
        k.c(f"blocks {4 * h}..{4 * h + 3}: the high index bits shifted by 2 l (group l), the scales as bytes 2 s + 1")
        for m in range(2):
            qw, sw = f"w{m}_{16 + h}", f"w{m}_{18 + h}"
            for l in range(1, 4):
                k(f"shr.b32 qh{m}_{l}, {qw}, {2 * l};")
            scale_bytes(k, m, sw)
        for b in range(4):
            ib = 4 * h + b
            k.c(f"block {ib}: weights {32 * ib}..{32 * ib + 31}")
            for m in range(2):
                qs, sg = f"w{m}_{ib}", f"w{m}_{8 + ib}"
                qhw = [f"w{m}_{16 + h}"] + [f"qh{m}_{l}" for l in range(1, 4)]
                k(f"and.b32 z, {sg}, K33;             // sign pairs (0, 1), (4, 5) of group l in nibbles 2l, 2l + 1")
                k(f"shr.b32 ts, {sg}, 2;")
                k("and.b32 z2, ts, K33;               // the pairs (2, 3), (6, 7)")
                for l in range(4):
                    k(f"prmt.b32 x, z, z2, 0x{(4 + l) << 4 | l:04x};       // group {l}'s pairs as a selector")
                    k("prmt.b32 sn, LS, ZERO, x;")
                    k(f"prmt.b32 px, {qs}, {qhw[l]}, 0x{(4 + b) << 4 | l:04x};  // the qs byte, the qh byte (bits 2l, 2l + 1)")
                    if BFI_INDEX:
                        k("bfi.b32 ea, px, coff, 4, 10;      // the 10-bit grid index at bits 4-13, the copy at 2-3")
                        k("add.u32 ea, ea, tabb;")
                    else:
                        k("and.b32 ix, px, 0x3FF;            // the 10-bit grid index")
                        k("shl.b32 ea, ix, 4;")
                        k("add.u32 ea, tabl, ea;")
                    values(k, m, l)
                scale_floats(k, m, b)
            sums(k, nt, ib)
    base.row_end(k, nt)


class IQ2S:
    prefix = "iq2s"
    copies = COPIES
    tab_words = NENT * COPIES
    unit_off = 80
    table = staticmethod(table)
    consts = staticmethod(consts)
    body = staticmethod(body)


def main():
    codes = grid_codes()
    print("//")
    print("// mtok-iq2_s.ptx: IQ2_S's fast multi-token expert kernel (C1138) in hand-written PTX, written by")
    print("// gen_mtok_iq2_s.py (do not edit; edit the script). See mmvq-pascal-mtok-iq2_s-fast.cu for the method")
    print("// and mmvq-pascal-mtok-iq-ptx.cu for how it is loaded and which products it takes.")
    print("//")
    print(".version 7.0")
    print(".target sm_61")
    print(".address_size 64")
    print()
    print("// the IQ2_S grid as codes: entry e's eight values (8, 25, 43 -> 0, 1, 2) as nibbles, nibble n weight")
    print(f"// {PI}[n] of the group")
    print(f".global .align 4 .b32 mtok_iq2s_codes[{NENT}] = {{")
    for i in range(0, NENT, 8):
        print("    " + ", ".join(f"0x{c:08x}" for c in codes[i:i + 8]) + ("," if i + 8 < NENT else ""))
    print("};")
    print(".extern .shared .align 16 .b8 mtok_iq2s_dyn[];")
    print()
    for ntm in (2, 4, 8):
        print(base.entry(ntm, IQ2S))
        print()


if __name__ == "__main__":
    main()
