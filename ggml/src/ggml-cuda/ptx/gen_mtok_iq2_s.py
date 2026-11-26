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


def body(k, nt):
    """The rows of the block for groups of exactly nt members (mtok_iq2s_fast::rows<true, nt>)."""
    b32 = lambda *n: k.r(".b32", *n)
    b64 = lambda *n: k.r(".b64", *n)
    f32 = lambda *n: k.r(".f32", *n)
    prd = lambda *n: k.r(".pred", *n)
    b16 = lambda *n: k.r(".b16", *n)
    L = f"N{nt}_"
    k.label(f"BODY{nt}")
    k.c(f"groups of {nt} member(s): member j's output offset dc<j>")
    for j in range(nt):
        b32(f"dc{j}")
        k(f"shfl.sync.idx.b32 dc{j}, doff, {j}, 31, -1;")
    b32("rbi", "rblk", "row0", "rowc", "kap0", "kap", "itr", "cs", "au", "ad", "sego")
    b64("rw0", "rw1", "ro64", "seg")
    prd("p0", "pst")
    k("mov.u32 rbi, 0;")
    k.label(L + "ROWBLOCK")
    k("setp.ge.u32 p0, rbi, rb;")
    k("@p0 bra.uni DONE;")
    k("add.u32 rblk, rbx, rbi;")
    k("setp.ge.u32 p0, rblk, nrowblocks;")
    k("@p0 bra.uni DONE;")
    k.c("rows 4 apart: warp w takes rows w, w + 4, ..., w + 60 of the row block; lane pair grp row w + 4 grp")
    k("shl.b32 row0, rblk, 6;")
    k("add.u32 row0, row0, warp;")
    k("shl.b32 t0, grp, 2;")
    k("add.u32 row0, row0, t0;")
    k("add.u32 t0, nrows, -1;")
    k("min.u32 rowc, row0, t0;")
    k("mul.wide.u32 ro64, rowc, row_bytes;")
    k("add.u64 rw0, xu, ro64;")
    k("add.u64 rw1, xg, ro64;")
    k.c("kappa: the word of its first 16-byte unit at which the warp's segments start (lane 0's row)")
    k("cvt.u32.u64 t0, rw0;")
    k("bfe.u32 t0, t0, 2, 2;")
    k("shfl.sync.idx.b32 kap0, t0, 0, 31, -1;")
    for j in range(nt):
        for m in range(2):
            f32(f"acc{j}_{m}")
            k(f"mov.f32 acc{j}_{m}, 0f00000000;")
    k("mov.u32 itr, 0;")
    k.label(L + "ITER")
    k.c("the lane's super-block c = 2 it + sl: its activations au (member 0) and scales ad")
    k("shl.b32 cs, itr, 1;")
    k("add.u32 cs, cs, sl;")
    k(f"mad.lo.u32 au, cs, {ACT_RUN}, act;")
    k("shl.b32 t0, cs, 5;")
    k("add.u32 ad, ad0, t0;")
    for j in range(1, nt):
        b32(f"au{j}", f"ad{j}")
        k(f"mad.lo.u32 au{j}, act_tok, {j}, au;")
        k(f"mad.lo.u32 ad{j}, act_tok, {j}, ad;")
    k.c("both matrices' segments: lane sl loads the 6 aligned 16-byte units from (segment & ~15) + 80 sl")
    k(f"mul.lo.u32 sego, itr, {2 * SB};")
    k("cvt.u64.u32 w64, sego;")
    for m, rw in ((0, "rw0"), (1, "rw1")):
        k(f"add.u64 seg, {rw}, w64;")
        k("and.b64 seg, seg, -16;")
        k("add.u64 seg, seg, sl64;")
        for u in range(6):
            regs = [f"u{m}_{4 * u + q}" for q in range(4)]
            b32(*regs)
            k(f"ld.global.nc.v4.u32 {{{', '.join(regs)}}}, [seg+{16 * u}];")
    k.c("the words of each super-block after its d, by kappa (warp-uniform): w<m>_k = bytes 2 + 4k .. 5 + 4k")
    k("add.u32 kap, kap0, itr;")
    k("and.b32 kap, kap, 3;")
    for kv in range(3):
        k(f"setp.eq.u32 p0, kap, {kv};")
        k(f"@p0 bra.uni {L}KAPPA{kv};")
    for kv in (3, 0, 1, 2):
        if kv != 3:
            k.label(f"{L}KAPPA{kv}")
        for m in range(2):
            for w in range(20):
                b32(f"w{m}_{w}")
                k(f"prmt.b32 w{m}_{w}, u{m}_{kv + w}, u{m}_{kv + w + 1}, sel;")
            b32(f"dw{m}")
            k(f"mov.b32 dw{m}, u{m}_{kv};")
        if kv != 2:
            k(f"bra.uni {L}WORDS;")
    k.label(f"{L}WORDS")
    for j in range(nt):
        for m in range(2):
            f32(f"sf{j}_{m}")
            k(f"mov.f32 sf{j}_{m}, 0f00000000;")
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
            k(f"shl.b32 ts, {sw}, 1;")
            k(f"lop3.b32 e0_{m}, ts, K1E, K01, 0xEA;  // ((sw << 1) & 0x1E1E1E1E) | 0x01010101: 2 ls0 + 1")
            k(f"shr.b32 ts, {sw}, 3;")
            k(f"lop3.b32 e1_{m}, ts, K1E, K01, 0xEA;  // ((sw >> 3) & 0x1E1E1E1E) | 0x01010101: 2 ls1 + 1")
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
                    k("ld.shared.u32 cw, [ea];")
                    k("or.b32 sv, cw, sn;")
                    k(f"prmt.b32 q{m}_{2 * l}, LP, LN, sv;")
                    k("shr.b32 sh, sv, 16;")
                    k(f"prmt.b32 q{m}_{2 * l + 1}, LP, LN, sh;")
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
    k.c("the super-block's d: lane 0's in the low half of dw, lane 1's in the high half")
    b32("dsh")
    f32("fd")
    for m in range(2):
        k(f"shr.b32 t0, dw{m}, 16;")
        k(f"selp.b32 dsh, t0, dw{m}, psl;")
        k("mov.b32 {h0, h1}, dsh;")
        k("cvt.f32.f16 fd, h0;")
        k("mul.ftz.f32 fd, fd, 0f3E000000;")
        for j in range(nt):
            k(f"fma.rn.ftz.f32 acc{j}_{m}, fd, sf{j}_{m}, acc{j}_{m};")
    k("add.u32 itr, itr, 1;")
    k(f"setp.lt.u32 p0, itr, {NITER};")
    k(f"@p0 bra.uni {L}ITER;")
    k.c("the row's two lanes summed, then up * SiLU(gate) (ggml_cuda_op_silu_single under -use_fast_math)")
    b32("rt0", "rt1")
    f32("fs", "fu", "fg", "fe")
    for j in range(nt):
        for m in range(2):
            k(f"mov.b32 rt0, acc{j}_{m};")
            k("shfl.sync.bfly.b32 rt1, rt0, 1, 31, -1;")
            k("mov.b32 fs, rt1;")
            k(f"add.ftz.f32 acc{j}_{m}, acc{j}_{m}, fs;")
    k("setp.eq.u32 p0, sl, 0;")
    k("setp.lt.u32 pst, row0, nrows;")
    k("and.pred pst, pst, p0;")
    for j in range(nt):
        k(f"mul.ftz.f32 fe, acc{j}_1, 0fBFB8AA3B;")
        k("ex2.approx.ftz.f32 fe, fe;")
        k("add.ftz.f32 fe, fe, 0f3F800000;")
        k(f"div.approx.ftz.f32 fg, acc{j}_1, fe;")
        k(f"mul.ftz.f32 fu, acc{j}_0, fg;")
        k(f"add.u32 t0, dc{j}, row0;")
        k("mul.wide.u32 o64, t0, 4;")
        k("add.u64 a64, dstp, o64;")
        k("@pst st.global.f32 [a64], fu;")
    k("add.u32 rbi, rbi, 1;")
    k(f"bra.uni {L}ROWBLOCK;")
    k.blank()


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
