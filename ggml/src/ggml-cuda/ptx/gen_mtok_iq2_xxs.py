#!/usr/bin/env python3
"""Writes mtok-iq2_xxs.ptx: IQ2_XXS's fast multi-token expert kernel (C1138) as hand-written PTX, the whole kernel of
mmvq-pascal-mtok-iq2_xxs-fast.cu for the products Qwen3.8-Flash-Next's verify pass makes (the fused gate/up launch with
SWIGLU, rows of 2,560 weights: 10 super-blocks, 2 lanes a row, 5 iterations; no shared expert, no biases).

Every instruction is written here; this script only repeats the unrolled parts (8 blocks x 2 matrices x 4 groups of a
super-block, the members, the three entry points for groups of up to 2, 4 and 8 members) and fills in the IQ2_XXS grid's
code table from ggml-common.h. The float operations are the CUDA C kernel's, in its order and with the forms nvcc gives
them under -use_fast_math (.ftz, ex2.approx and div.approx for SiLU), so the results are bit-identical to it and to the
old kernel. Run: python3 -I gen_mtok_iq2_xxs.py > mtok-iq2_xxs.ptx (from this folder).

Parameters (byte offsets, checked by static_asserts in mmvq-pascal-mtok-iq-ptx.cu): mtok_iq::args (168 bytes) and
ggml_cuda_mm_fusion_args_device (80 bytes; the gate matrix at 8).
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PI = [0, 1, 4, 5, 2, 3, 6, 7]  # nibble n of a selector word holds weight PI[n] of its group of 8
ACT_RUN = 272                  # bytes of a member's staged int8 values per super-block (256 and 16 of padding)
NCHUNK = 10                    # super-blocks per row
NITER = 5                      # iterations per lane (2 lanes a row)
SCALES = NCHUNK * ACT_RUN      # where a member's activation scales start


def grid_codes():
    src = open(os.path.join(HERE, "..", "..", "ggml-common.h")).read()
    m = re.search(r"GGML_TABLE_BEGIN\(uint64_t, iq2xxs_grid, 256\)(.*?)GGML_TABLE_END", src, re.S)
    vals = [int(x, 16) for x in re.findall(r"0x([0-9a-fA-F]{16})", m.group(1))]
    assert len(vals) == 256
    codes = []
    for g in vals:
        c = 0
        for n in range(8):
            v = (g >> (8 * PI[n])) & 0xFF
            assert v in (0x08, 0x19, 0x2B)
            c |= {0x08: 0, 0x19: 1, 0x2B: 2}[v] << (4 * n)
        codes.append(c)
    return codes


class K:
    """One entry point: its lines and its registers."""

    def __init__(self):
        self.lines = []
        self.regs = {}

    def r(self, typ, *names):
        for n in names:
            assert self.regs.get(n, typ) == typ, n
            self.regs[n] = typ
        return names[0] if len(names) == 1 else names

    def __call__(self, s):
        self.lines.append("    " + s)

    def label(self, name):
        self.lines.append(name + ":")

    def c(self, text):
        self.lines.append("    // " + text)

    def blank(self):
        self.lines.append("")


class IQ2XXS:
    """What an entry point of this type differs in (gen_mtok_iq2_s.py brings its own: the generator's other parts are
    shared)."""
    prefix = "iq2xxs"    # names: mtok_<prefix>_fast_ptx_ntm<N>, mtok_<prefix>_codes, mtok_<prefix>_dyn
    copies = 8           # copies of the code table, interleaved by word
    tab_words = 2048     # the shared code table's words
    unit_off = 64        # bytes from lane 0's first 16-byte unit to lane 1's

    @staticmethod
    def table(k):
        table_iq2xxs(k)

    @staticmethod
    def consts(k):
        pass

    @staticmethod
    def body(k, nt):
        body(k, nt)


def entry(ntm, kt=IQ2XXS):
    k = K()
    name = f"mtok_{kt.prefix}_fast_ptx_ntm{ntm}"
    b32 = lambda *n: k.r(".b32", *n)
    b64 = lambda *n: k.r(".b64", *n)
    f32 = lambda *n: k.r(".f32", *n)
    prd = lambda *n: k.r(".pred", *n)
    b16 = lambda *n: k.r(".b16", *n)

    k.c("thread, block and parameters")
    b32("lane", "warp", "bx", "by", "bz", "tid")
    k("mov.u32 lane, %tid.x;")
    k("mov.u32 warp, %tid.y;")
    k("mov.u32 bx, %ctaid.x;")
    k("mov.u32 by, %ctaid.y;")
    k("mov.u32 bz, %ctaid.z;")
    k("mad.lo.u32 tid, warp, 32, lane;")
    b64("vx", "vy", "ids", "dstp", "gatep")
    for reg, off in (("vx", 0), ("vy", 8), ("ids", 16), ("dstp", 24)):
        k(f"ld.param.u64 {reg}, [P+{off}];")
        k(f"cvta.to.global.u64 {reg}, {reg};")
    k("ld.param.u64 gatep, [F+8];")
    k("cvta.to.global.u64 gatep, gatep;")
    ints = (("nrows", 32), ("row_bytes", 36), ("nrowblocks", 48), ("rb", 52), ("gcap", 56), ("ncols", 60),
            ("nslots", 64), ("ids_stride", 68), ("act_tok", 76), ("ncy_x", 80), ("ncy_y", 84), ("ncy_z", 88),
            ("sr_x", 92), ("sr_y", 96))
    for reg, off in ints:
        b32(reg)
        k(f"ld.param.u32 {reg}, [P+{off}];")
    longs = (("chbx", 104), ("smbx", 112), ("chby", 120), ("smby", 128), ("colby", 136), ("scd", 144), ("ssd", 152),
             ("scold", 160))
    for reg, off in longs:
        b64(reg)
        k(f"ld.param.u64 {reg}, [P+{off}];")
    k.c("sample_x = fastdiv(blockIdx.z, sample_ratio)")
    b32("t0", "t1", "t2", "t3", "sx")
    k("mul.hi.u32 t0, bz, sr_x;")
    k("add.u32 t0, t0, bz;")
    k("shr.b32 sx, t0, sr_y;")
    k.blank()

    k.c("the block's group (mtok_iq::group): entry blockIdx.y = token ty, slot ky; its expert ey")
    b32("ne", "ty", "ky", "ey", "idx")
    b64("a64", "o64")
    prd("p0", "p1", "p2", "p3", "p4")
    k("mul.lo.u32 ne, ncols, nslots;")
    k("div.u32 ty, by, nslots;")
    k("mul.lo.u32 t0, ty, nslots;")
    k("sub.u32 ky, by, t0;")
    k("mad.lo.u32 idx, ty, ids_stride, ky;")
    k("mul.wide.u32 o64, idx, 4;")
    k("add.u64 a64, ids, o64;")
    k("ld.global.s32 ey, [a64];")
    k("setp.lt.s32 p0, ey, 0;")
    k("@p0 bra.uni SKIPPED;")
    k.c("rank: the entries before this one with the same expert; mk<h>: those from it on, 32 entries a word")
    b32("rank", "i", "v", "bal", "cnt", "mk0", "mk1", "mk2", "mk3")
    k("mov.u32 rank, 0;")
    for h in range(4):
        k(f"add.u32 i, lane, {32 * h};")
        k("div.u32 t0, i, nslots;")
        k("mul.lo.u32 t1, t0, nslots;")
        k("sub.u32 t1, i, t1;")
        k("mad.lo.u32 idx, t0, ids_stride, t1;")
        k("mul.wide.u32 o64, idx, 4;")
        k("add.u64 a64, ids, o64;")
        k("setp.lt.u32 p1, i, ne;")
        k("mov.u32 v, -1;")
        k("@p1 ld.global.s32 v, [a64];")
        k("setp.eq.s32 p2, v, ey;")
        k("setp.lt.u32 p3, i, by;")
        k("and.pred p4, p2, p3;")
        k("vote.sync.ballot.b32 bal, p4, -1;")
        k("popc.b32 cnt, bal;")
        k("add.u32 rank, rank, cnt;")
        k("setp.ge.u32 p3, i, by;")
        k("and.pred p4, p2, p3;")
        k(f"vote.sync.ballot.b32 mk{h}, p4, -1;")
    k("rem.u32 t0, rank, gcap;")
    k("setp.ne.u32 p0, t0, 0;")
    k("@p0 bra.uni DONE;")
    k.c("lane j < nt finds the group's member j: the j-th set bit of mk0..mk3")
    b32("total", "hsel", "rr", "msk", "nt", "yoff", "doff", "bit", "kk", "chy")
    b64("w64", "x64")
    k("mov.u32 total, 0;")
    k("mov.u32 hsel, 4;")
    k("mov.u32 rr, 0;")
    k("mov.u32 msk, 0;")
    for h in range(4):
        k(f"popc.b32 cnt, mk{h};")
        k("add.u32 t0, total, cnt;")
        k("setp.eq.u32 p1, hsel, 4;")
        k("setp.lt.u32 p2, lane, t0;")
        k("and.pred p3, p1, p2;")
        k(f"@p3 mov.u32 hsel, {h};")
        k("@p3 sub.u32 rr, lane, total;")
        k(f"@p3 mov.u32 msk, mk{h};")
        k("mov.u32 total, t0;")
    k("min.u32 nt, gcap, total;")
    k("mov.u32 yoff, 0;")
    k("mov.u32 doff, 0;")
    k("setp.ge.u32 p1, lane, nt;")
    k("@p1 bra MEMBER_DONE;")
    k.label("DROP_BITS")
    k("setp.eq.u32 p2, rr, 0;")
    k("@p2 bra DROPPED;")
    k("add.u32 t0, msk, -1;")
    k("and.b32 msk, msk, t0;")
    k("sub.u32 rr, rr, 1;")
    k("bra DROP_BITS;")
    k.label("DROPPED")
    k("neg.s32 t0, msk;")
    k("and.b32 t0, msk, t0;")
    k("bfind.u32 bit, t0;")
    k("shl.b32 i, hsel, 5;")
    k("add.u32 i, i, bit;")
    k("div.u32 t0, i, nslots;")
    k("mul.lo.u32 t1, t0, nslots;")
    k("sub.u32 kk, i, t1;")
    k.c("channel_y = fastmodulo(k, nchannels_y)")
    k("mul.hi.u32 t2, kk, ncy_x;")
    k("add.u32 t2, t2, kk;")
    k("shr.b32 t2, t2, ncy_y;")
    k("mul.lo.u32 t2, t2, ncy_z;")
    k("sub.u32 chy, kk, t2;")
    k.c("yoff = sample * sample_bytes_y + channel_y * channel_bytes_y + token * col_bytes_y (fits an int)")
    k("cvt.u64.u32 w64, bz;")
    k("mul.lo.u64 x64, w64, smby;")
    k("cvt.u64.u32 w64, chy;")
    k("mad.lo.u64 x64, w64, chby, x64;")
    k("cvt.u64.u32 w64, t0;")
    k("mad.lo.u64 x64, w64, colby, x64;")
    k("cvt.u32.u64 yoff, x64;")
    k.c("doff = sample * stride_sample_dst + slot * stride_channel_dst + token * stride_col_dst")
    k("cvt.u64.u32 w64, bz;")
    k("mul.lo.u64 x64, w64, ssd;")
    k("cvt.u64.u32 w64, kk;")
    k("mad.lo.u64 x64, w64, scd, x64;")
    k("cvt.u64.u32 w64, t0;")
    k("mad.lo.u64 x64, w64, scold, x64;")
    k("cvt.u32.u64 doff, x64;")
    k.label("MEMBER_DONE")
    k.blank()

    kt.table(k)
    k.c("the members' activations: each group of 8 int8 values as (0, 1, 4, 5), (2, 3, 6, 7); thread q < 80 stages")
    k.c("Q8_1 block q of each member: 256 values and 16 bytes of padding per super-block, the FP32 scales after them")
    b32("act", "jm", "yj", "aj", "drel", "srel", "pa0", "pa1", "pa2", "pa3")
    b32(*[f"y{n}" for n in range(9)])
    b16("h0", "h1")
    f32("fsc")
    prd("pq")
    k(f"mov.u32 act, mtok_{kt.prefix}_dyn;")
    k(f"setp.lt.u32 pq, tid, {8 * NCHUNK};")
    k("shr.b32 t0, tid, 3;")
    k("and.b32 t1, tid, 7;")
    k("shl.b32 t1, t1, 5;")
    k(f"mad.lo.u32 drel, t0, {ACT_RUN}, t1;")
    k(f"shl.b32 srel, tid, 2;")
    k(f"add.u32 srel, srel, {SCALES};")
    k("mul.wide.u32 w64, tid, 36;")
    k("mov.u32 jm, 0;")
    k.label("STAGE_MEMBER")
    k("setp.ge.u32 p0, jm, nt;")
    k("@p0 bra.uni STAGED;")
    k("shfl.sync.idx.b32 yj, yoff, jm, 31, -1;")
    k("mad.lo.u32 aj, jm, act_tok, act;")
    k("cvt.s64.s32 x64, yj;")
    k("add.u64 a64, vy, x64;")
    k("add.u64 a64, a64, w64;")
    for n in range(9):
        k(f"@pq ld.global.nc.u32 y{n}, [a64+{4 * n}];")
    k("add.u32 sa, aj, drel;")
    k("prmt.b32 pa0, y1, y2, 0x5410;")
    k("prmt.b32 pa1, y1, y2, 0x7632;")
    k("prmt.b32 pa2, y3, y4, 0x5410;")
    k("prmt.b32 pa3, y3, y4, 0x7632;")
    k("@pq st.shared.v4.u32 [sa], {pa0, pa1, pa2, pa3};")
    k("prmt.b32 pa0, y5, y6, 0x5410;")
    k("prmt.b32 pa1, y5, y6, 0x7632;")
    k("prmt.b32 pa2, y7, y8, 0x5410;")
    k("prmt.b32 pa3, y7, y8, 0x7632;")
    k("@pq st.shared.v4.u32 [sa+16], {pa0, pa1, pa2, pa3};")
    k("mov.b32 {h0, h1}, y0;")
    k("cvt.f32.f16 fsc, h0;")
    k("add.u32 sa, aj, srel;")
    k("@pq st.shared.f32 [sa], fsc;")
    k("add.u32 jm, jm, 1;")
    k("bra.uni STAGE_MEMBER;")
    k.label("STAGED")
    k("bar.sync 0;")
    k.blank()

    k.c("the expert's up and gate matrices; constants")
    b64("xo", "xu", "xg", "sl64")
    k("cvt.u64.u32 w64, sx;")
    k("mul.lo.u64 xo, w64, smbx;")
    k("cvt.u64.u32 w64, ey;")
    k("mad.lo.u64 xo, w64, chbx, xo;")
    k("add.u64 xu, vx, xo;")
    k("add.u64 xg, gatep, xo;")
    b32("sl", "grp", "sel", "tabl", "ad0", "rbx", "LP", "LN", "LS", "K3333", "ZERO", "DBASE")
    prd("psl")
    k("and.b32 sl, lane, 1;")
    k("shr.b32 grp, lane, 1;")
    k("setp.ne.u32 psl, sl, 0;")
    k("selp.b32 sel, 0x7654, 0x5432, psl;")
    k(f"and.b32 t0, lane, {kt.copies - 1};")
    k("shl.b32 t0, t0, 2;")
    k("add.u32 tabl, tabb, t0;")
    k(f"add.u32 ad0, act, {SCALES};")
    k("mul.lo.u32 rbx, bx, rb;")
    k(f"mul.wide.u32 sl64, sl, {kt.unit_off};")
    k("mov.b32 LP, 0x002b1908;    // bytes 8, 25, 43, 0: the grid values by code")
    k("mov.b32 LN, 0x00d5e7f8;    // -8, -25, -43, 0")
    k("mov.b32 LS, 0x44400400;    // two sign bits (index b0 + 2 b1) -> bit 2 of two nibbles")
    k("mov.b32 K3333, 0x3333;")
    k("mov.b32 ZERO, 0;")
    k("mov.b32 DBASE, 0x4B400000; // the float 1.5 * 2^23: a DP4A chain started there is that float + its sum")
    kt.consts(k)
    k.blank()
    for nt in range(1, ntm):
        k(f"setp.eq.u32 p0, nt, {nt};")
        k(f"@p0 bra.uni BODY{nt};")
    k(f"bra.uni BODY{ntm};")
    k.blank()

    for nt in range(1, ntm + 1):
        kt.body(k, nt)

    k.label("SKIPPED")
    k.c("a skipped slot (id -1): zeros in its rows of the output")
    b32("rbi", "rblk", "row0")
    b64("dsk")
    k("cvt.u64.u32 w64, bz;")
    k("mul.lo.u64 x64, w64, ssd;")
    k("cvt.u64.u32 w64, ky;")
    k("mad.lo.u64 x64, w64, scd, x64;")
    k("cvt.u64.u32 w64, ty;")
    k("mad.lo.u64 x64, w64, scold, x64;")
    k("shl.b64 x64, x64, 2;")
    k("add.u64 dsk, dstp, x64;")
    k("and.b32 sl, lane, 1;")
    k("shr.b32 grp, lane, 1;")
    k("mov.u32 rbi, 0;")
    k.label("SKIP_ROWS")
    k("setp.ge.u32 p0, rbi, rb;")
    k("@p0 bra.uni DONE;")
    k("mad.lo.u32 rblk, bx, rb, rbi;")
    k("setp.ge.u32 p0, rblk, nrowblocks;")
    k("@p0 bra.uni DONE;")
    k("shl.b32 row0, rblk, 6;")
    k("add.u32 row0, row0, warp;")
    k("shl.b32 t0, grp, 2;")
    k("add.u32 row0, row0, t0;")
    k("setp.eq.u32 p1, sl, 0;")
    k("setp.lt.u32 p2, row0, nrows;")
    k("and.pred p1, p1, p2;")
    k("mul.wide.u32 o64, row0, 4;")
    k("add.u64 a64, dsk, o64;")
    k("@p1 st.global.f32 [a64], 0f00000000;")
    k("add.u32 rbi, rbi, 1;")
    k("bra.uni SKIP_ROWS;")
    k.label("DONE")
    k("ret;")

    out = [f".visible .entry {name}(", "    .param .align 8 .b8 P[168],", "    .param .align 8 .b8 F[80]", ")",
           ".maxntid 128, 1, 1", f".minnctapersm {2 if ntm >= 8 else 3}", "{"]
    by_type = {}
    for n, t in k.regs.items():
        by_type.setdefault(t, []).append(n)
    for t in (".pred", ".b16", ".b32", ".b64", ".f32"):
        names = by_type.get(t, [])
        for i in range(0, len(names), 12):
            out.append(f"    .reg {t} " + ", ".join(names[i:i + 12]) + ";")
    out.append(f"    .shared .align 16 .b32 tab[{kt.tab_words}];")
    out.append("")
    out += k.lines
    out.append("}")
    return "\n".join(out)


def table_iq2xxs(k):
    """The code table in shared memory, from the global table mtok_iq2xxs_codes (256 entries, 8 copies)."""
    b32 = lambda *n: k.r(".b32", *n)
    b64 = lambda *n: k.r(".b64", *n)
    k.c("the code table: entry e's eight codes (0, 1, 2 for 8, 25, 43) as nibbles, 8 copies interleaved by word")
    b32("tabb", "e", "cw", "sa")
    b64("cods")
    k("mov.u32 tabb, tab;")
    k("mov.u64 cods, mtok_iq2xxs_codes;")
    for add in (0, 128):
        k(f"add.u32 e, tid, {add};")
        k("mul.wide.u32 o64, e, 4;")
        k("add.u64 a64, cods, o64;")
        k("ld.global.nc.u32 cw, [a64];")
        k("shl.b32 sa, e, 5;")
        k("add.u32 sa, sa, tabb;")
        k("st.shared.v4.u32 [sa], {cw, cw, cw, cw};")
        k("st.shared.v4.u32 [sa+16], {cw, cw, cw, cw};")


def row_start(k, nt, unit=16):
    """A body's start, shared by every type: BODY<nt>, the members' output offsets, the row-block loop, the rows of the
    lane pair, kappa (the word of its first unit-byte unit at which the warp's segments start), the accumulators and
    the iteration's activation pointers."""
    b32 = lambda *n: k.r(".b32", *n)
    b64 = lambda *n: k.r(".b64", *n)
    f32 = lambda *n: k.r(".f32", *n)
    prd = lambda *n: k.r(".pred", *n)
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
    k.c(f"kappa: the word of its first {unit}-byte unit at which the warp's segments start (lane 0's row)")
    k("cvt.u32.u64 t0, rw0;")
    k(f"bfe.u32 t0, t0, 2, {(unit // 4).bit_length() - 1};")
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


def segments(k, nt, sb, unit, nunits, unit_off, nwords):
    """Both matrices' row segments into registers, shared by every type: lane sl loads nunits aligned unit-byte units
    (16 or 8) from (segment & ~(unit - 1)) + unit_off sl into u<m>_*, a warp-uniform branch on kappa (unit / 4 cases)
    picks the nwords words after each super-block's d (w<m>_k = bytes 2 + 4k .. 5 + 4k) and the word holding d (dw<m>);
    then the members' sums are zeroed."""
    b32 = lambda *n: k.r(".b32", *n)
    f32 = lambda *n: k.r(".f32", *n)
    L = f"N{nt}_"
    per = unit // 4  # words per unit
    k.c(f"both matrices' segments: lane sl loads the {nunits} aligned {unit}-byte units from (segment & ~{unit - 1})"
        f" + {unit_off} sl")
    k(f"mul.lo.u32 sego, itr, {2 * sb};")
    k("cvt.u64.u32 w64, sego;")
    for m, rw in ((0, "rw0"), (1, "rw1")):
        k(f"add.u64 seg, {rw}, w64;")
        k(f"and.b64 seg, seg, -{unit};")
        k("add.u64 seg, seg, sl64;")
        for u in range(nunits):
            regs = [f"u{m}_{per * u + q}" for q in range(per)]
            b32(*regs)
            k(f"ld.global.nc.v{per}.u32 {{{', '.join(regs)}}}, [seg+{unit * u}];")
    k.c("the words of each super-block after its d, by kappa (warp-uniform): w<m>_k = bytes 2 + 4k .. 5 + 4k")
    k("add.u32 kap, kap0, itr;")
    k(f"and.b32 kap, kap, {per - 1};")
    for kv in range(per - 1):
        k(f"setp.eq.u32 p0, kap, {kv};")
        k(f"@p0 bra.uni {L}KAPPA{kv};")
    order = [per - 1] + list(range(per - 1))
    for kv in order:
        if kv != per - 1:
            k.label(f"{L}KAPPA{kv}")
        for m in range(2):
            for w in range(nwords):
                b32(f"w{m}_{w}")
                k(f"prmt.b32 w{m}_{w}, u{m}_{kv + w}, u{m}_{kv + w + 1}, sel;")
            b32(f"dw{m}")
            k(f"mov.b32 dw{m}, u{m}_{kv};")
        if kv != order[-1]:
            k(f"bra.uni {L}WORDS;")
    k.label(f"{L}WORDS")
    for j in range(nt):
        for m in range(2):
            f32(f"sf{j}_{m}")
            k(f"mov.f32 sf{j}_{m}, 0f00000000;")


def row_end(k, nt):
    """A body's end, shared by every type: each matrix's super-block sums times its d, the iteration loop, the row's two
    lanes summed, up * SiLU(gate) stored, the row-block loop."""
    b32 = lambda *n: k.r(".b32", *n)
    f32 = lambda *n: k.r(".f32", *n)
    L = f"N{nt}_"
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


def body(k, nt):
    """The rows of the block for groups of exactly nt members (mtok_iq2xxs_fast::rows<true, nt>)."""
    b32 = lambda *n: k.r(".b32", *n)
    f32 = lambda *n: k.r(".f32", *n)
    row_start(k, nt)
    segments(k, nt, 66, 16, 5, 64, 16)
    b32("s7", "t6", "x", "pc", "sg", "eo", "ea", "cw2", "sv", "sh", "ia")
    b32(*[f"q{m}_{n}" for m in range(2) for n in range(8)])
    b32(*[f"ua{n}" for n in range(4)], *[f"ub{n}" for n in range(4)])
    f32("fl0", "fl1", "nc0", "nc1", "fa", "fi", "fp")
    for ib in range(8):
        k.c(f"block {ib}: weights {32 * ib}..{32 * ib + 31}")
        for m in range(2):
            qi, aux = f"w{m}_{2 * ib}", f"w{m}_{2 * ib + 1}"
            for l in range(4):
                k(f"bfe.u32 s7, {aux}, {7 * l}, 7;")
                k("shl.b32 t6, s7, 6;")
                k("lop3.b32 x, s7, t6, K3333, 0xA8;   // (s7 | s7 << 6) & 0x3333")
                k("popc.b32 pc, s7;")
                k("bfi.b32 x, pc, x, 13, 1;          // the eighth sign: the parity of the seven")
                k("prmt.b32 sg, LS, ZERO, x;")
                k(f"bfe.u32 eo, {qi}, {8 * l}, 8;")
                k("shl.b32 eo, eo, 5;")
                k("add.u32 ea, tabl, eo;")
                k("ld.shared.u32 cw2, [ea];")
                k("or.b32 sv, cw2, sg;")
                k(f"prmt.b32 q{m}_{2 * l}, LP, LN, sv;")
                k("shr.b32 sh, sv, 16;")
                k(f"prmt.b32 q{m}_{2 * l + 1}, LP, LN, sh;")
            k(f"shr.b32 t0, {aux}, 27;")
            k("or.b32 t0, t0, 0x4B000001;")
            k(f"mov.b32 fl{m}, t0;")
            k(f"add.ftz.f32 fl{m}, fl{m}, 0fCB000000;  // 2 scale + 1")
            k(f"mul.ftz.f32 nc{m}, fl{m}, 0fCB400000;  // -(2 scale + 1) * 1.5 * 2^23")
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
                k(f"dp4a.s32.s32 ia, q{m}_4, ub0, ia;")
                k(f"dp4a.s32.s32 ia, q{m}_5, ub1, ia;")
                k(f"dp4a.s32.s32 ia, q{m}_6, ub2, ia;")
                k(f"dp4a.s32.s32 ia, q{m}_7, ub3, ia;")
                k("mov.b32 fi, ia;")
                k(f"fma.rn.ftz.f32 fp, fl{m}, fi, nc{m};     // (float) (ls * isum), exactly")
                k(f"fma.rn.ftz.f32 sf{j}_{m}, fa, fp, sf{j}_{m};")
    row_end(k, nt)


def main():
    codes = grid_codes()
    print("//")
    print("// mtok-iq2_xxs.ptx: IQ2_XXS's fast multi-token expert kernel (C1138) in hand-written PTX, written by")
    print("// gen_mtok_iq2_xxs.py (do not edit; edit the script). See mmvq-pascal-mtok-iq2_xxs-fast.cu for the method")
    print("// and mmvq-pascal-mtok-iq-ptx.cu for how it is loaded and which products it takes.")
    print("//")
    print(".version 7.0")
    print(".target sm_61")
    print(".address_size 64")
    print()
    print("// the IQ2_XXS grid as codes: entry e's eight values (8, 25, 43 -> 0, 1, 2) as nibbles, nibble n weight")
    print(f"// {PI}[n] of the group")
    print(".global .align 4 .b32 mtok_iq2xxs_codes[256] = {")
    for i in range(0, 256, 8):
        print("    " + ", ".join(f"0x{c:08x}" for c in codes[i:i + 8]) + ("," if i + 8 < 256 else ""))
    print("};")
    print(".extern .shared .align 16 .b8 mtok_iq2xxs_dyn[];")
    print()
    for ntm in (2, 4, 8):
        print(entry(ntm))
        print()


if __name__ == "__main__":
    main()
