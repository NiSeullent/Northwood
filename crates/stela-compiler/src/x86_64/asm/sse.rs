//! Typed SSE and AVX forms (Intel SDM Vol. 2) on the generic legacy and SSE
//! encodings.
use super::{Asm, Gpr, Map, Pp, Rm, Size, Vex};

/// Legacy SSE forms `name xmm, xmm/m`: an optional mandatory prefix and the
/// opcode bytes.
macro_rules! sse {
    ($($(#[$doc:meta])* $name:ident => $prefix:expr, [$($opcode:literal),+];)+) => {
        impl Asm<'_> {
            $(
                $(#[$doc])*
                pub(crate) fn $name(&mut self, xmm: u8, rm: impl Into<Rm>) {
                    self.sse($prefix, &[$($opcode),+], xmm, rm);
                }
            )+
        }
    };
}

/// Legacy SSE forms `name xmm, xmm/m, imm8`.
macro_rules! sse_imm {
    ($($(#[$doc:meta])* $name:ident => $prefix:expr, [$($opcode:literal),+];)+) => {
        impl Asm<'_> {
            $(
                $(#[$doc])*
                pub(crate) fn $name(&mut self, xmm: u8, rm: impl Into<Rm>, immediate: u8) {
                    self.sse_imm($prefix, &[$($opcode),+], xmm, rm, immediate);
                }
            )+
        }
    };
}

sse! {
    /// `movlhps xmm, xmm`: the low qword of the source to the high qword.
    movlhps => None, [0x0f, 0x16];
    /// `movhps xmm, m64`: memory to the high qword.
    movhps => None, [0x0f, 0x16];
    /// `movhlps xmm, xmm`: the high qword of the source to the low qword.
    movhlps => None, [0x0f, 0x12];
    /// `movq m64, xmm` (66 0F D6), named by its register first.
    movq_store => Some(0x66), [0x0f, 0xd6];
    /// `movq xmm, xmm/m64`: the low qword, zeroing the high one.
    movq_xmm => Some(0xf3), [0x0f, 0x7e];
    /// `movdqu xmm, xmm/m128`.
    movdqu_load => Some(0xf3), [0x0f, 0x6f];
    /// `movdqa xmm, xmm/m128`: one 16-byte access, which faults unless
    /// 16-byte aligned.
    movdqa_load => Some(0x66), [0x0f, 0x6f];
    /// `punpcklqdq xmm, xmm/m128`: the low qwords of both, the source's
    /// high.
    punpcklqdq => Some(0x66), [0x0f, 0x6c];
    /// `punpckhqdq xmm, xmm/m128`: the high qwords of both, the source's
    /// high.
    punpckhqdq => Some(0x66), [0x0f, 0x6d];
    /// `unpcklps xmm, xmm/m128`.
    unpcklps => None, [0x0f, 0x14];
    /// `unpcklpd xmm, xmm/m128`.
    unpcklpd => Some(0x66), [0x0f, 0x14];
    /// `unpckhpd xmm, xmm/m128`.
    unpckhpd => Some(0x66), [0x0f, 0x15];
    /// `cvtss2sd xmm, xmm/m32`.
    cvtss2sd => Some(0xf3), [0x0f, 0x5a];
    /// `cvtsd2ss xmm, xmm/m64`.
    cvtsd2ss => Some(0xf2), [0x0f, 0x5a];
    /// `cvtps2pd xmm, xmm/m64`.
    cvtps2pd => None, [0x0f, 0x5a];
    /// `cvtpd2ps xmm, xmm/m128`.
    cvtpd2ps => Some(0x66), [0x0f, 0x5a];
    /// `cvtdq2pd xmm, xmm/m64`.
    cvtdq2pd => Some(0xf3), [0x0f, 0xe6];
    /// `cvttps2dq xmm, xmm/m128`.
    cvttps2dq => Some(0xf3), [0x0f, 0x5b];
    /// `ucomiss xmm, xmm/m32`: unordered compare into ZF/PF/CF.
    ucomiss => None, [0x0f, 0x2e];
    /// `ucomisd xmm, xmm/m64`: unordered compare into ZF/PF/CF.
    ucomisd => Some(0x66), [0x0f, 0x2e];
    /// `psadbw xmm, xmm/m128`: sums of absolute byte differences per qword.
    psadbw => Some(0x66), [0x0f, 0xf6];
    /// `pblendvb xmm, xmm/m128, <xmm0>` (SSE4.1): bytes of `rm` where XMM0
    /// bytes have their sign set.
    pblendvb => Some(0x66), [0x0f, 0x38, 0x10];
}

sse_imm! {
    /// `pinsrw xmm, r32/m16, imm8`.
    pinsrw => Some(0x66), [0x0f, 0xc4];
    /// `pinsrd xmm, r/m32, imm8` (SSE4.1).
    pinsrd => Some(0x66), [0x0f, 0x3a, 0x22];
    /// `shufps xmm, xmm/m128, imm8`.
    shufps => None, [0x0f, 0xc6];
}

/// The lanes an SSE floating-point operation covers, which select its
/// mandatory prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FloatLanes {
    PackedSingle,
    ScalarSingle,
    PackedDouble,
    ScalarDouble,
}

impl FloatLanes {
    const fn prefix(self) -> Option<u8> {
        match self {
            Self::PackedSingle => None,
            Self::ScalarSingle => Some(0xf3),
            Self::PackedDouble => Some(0x66),
            Self::ScalarDouble => Some(0xf2),
        }
    }
}

/// SSE floating-point operations that share the `0F op` opcode across the
/// four lane forms (`ps`, `ss`, `pd`, `sd`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum FloatOp {
    Sqrt = 0x51,
    Add = 0x58,
    Mul = 0x59,
    Sub = 0x5c,
    Min = 0x5d,
    Div = 0x5e,
    Max = 0x5f,
}

impl Asm<'_> {
    /// `op{ps,ss,pd,sd} xmm, xmm/m`. MIN and MAX return the second operand
    /// when either is NaN or both are zero.
    pub(crate) fn float(&mut self, op: FloatOp, lanes: FloatLanes, xmm: u8, rm: impl Into<Rm>) {
        self.sse(lanes.prefix(), &[0x0f, op as u8], xmm, rm);
    }
    /// `cmp{ps,ss,pd,sd} xmm, xmm/m, predicate`.
    pub(crate) fn float_compare(
        &mut self,
        lanes: FloatLanes,
        xmm: u8,
        rm: impl Into<Rm>,
        predicate: u8,
    ) {
        self.sse_imm(lanes.prefix(), &[0x0f, 0xc2], xmm, rm, predicate);
    }
    /// `cvtsi2ss`/`cvtsi2sd xmm, r/m32/64` (F3/F2 [REX.W] 0F 2A) for a scalar
    /// `lanes` form.
    pub(crate) fn cvtsi2float(
        &mut self,
        lanes: FloatLanes,
        size: Size,
        xmm: u8,
        rm: impl Into<Rm>,
    ) {
        self.legacy(size, lanes.prefix(), &[0x0f, 0x2a], xmm, false, rm.into());
    }
    /// `cvtss2si`/`cvtsd2si r32/64, xmm/m` (0F 2D), or the truncating
    /// `cvtt` forms (0F 2C), for a scalar `lanes` form.
    pub(crate) fn cvtfloat2si(
        &mut self,
        lanes: FloatLanes,
        truncate: bool,
        size: Size,
        destination: Gpr,
        rm: impl Into<Rm>,
    ) {
        let opcode = if truncate { 0x2c } else { 0x2d };
        self.legacy(
            size,
            lanes.prefix(),
            &[0x0f, opcode],
            destination as u8,
            false,
            rm.into(),
        );
    }
    /// `movdqu xmm/m128, xmm`.
    pub(crate) fn movdqu_store(&mut self, rm: impl Into<Rm>, xmm: u8) {
        self.sse_reading(Some(0xf3), &[0x0f, 0x7f], xmm, rm);
    }
    /// `movdqa xmm/m128, xmm`: one 16-byte access, which faults unless
    /// 16-byte aligned.
    pub(crate) fn movdqa_store(&mut self, rm: impl Into<Rm>, xmm: u8) {
        self.sse_reading(Some(0x66), &[0x0f, 0x7f], xmm, rm);
    }
    /// `pextrw r32, xmm, imm8` (66 0F C5): zero-extends word `lane`.
    pub(crate) fn pextrw(&mut self, destination: Gpr, xmm: u8, lane: u8) {
        self.sse_imm(
            Some(0x66),
            &[0x0f, 0xc5],
            destination as u8,
            Rm::Xmm(xmm),
            lane,
        );
    }
    /// `pextrd r/m32, xmm, imm8` (66 0F 3A 16, SSE4.1).
    pub(crate) fn pextrd(&mut self, rm: impl Into<Rm>, xmm: u8, lane: u8) {
        self.sse_imm(Some(0x66), &[0x0f, 0x3a, 0x16], xmm, rm, lane);
    }
    /// `pextrq r/m64, xmm, imm8` (66 REX.W 0F 3A 16, SSE4.1).
    pub(crate) fn pextrq(&mut self, rm: impl Into<Rm>, xmm: u8, lane: u8) {
        self.legacy(
            Size::Qword,
            Some(0x66),
            &[0x0f, 0x3a, 0x16],
            xmm,
            false,
            rm.into(),
        );
        self.data(&[lane]);
    }
    /// `movd xmm, r/m32`, zeroing the rest of `xmm`.
    pub(crate) fn movd_to_xmm(&mut self, xmm: u8, rm: impl Into<Rm>) {
        self.sse(Some(0x66), &[0x0f, 0x6e], xmm, rm);
    }
    /// `movd r/m32, xmm`.
    pub(crate) fn movd_from_xmm(&mut self, rm: impl Into<Rm>, xmm: u8) {
        self.sse_reading(Some(0x66), &[0x0f, 0x7e], xmm, rm);
    }
    /// `stmxcsr m32`.
    pub(crate) fn stmxcsr(&mut self, memory: impl Into<Rm>) {
        self.legacy(Size::Dword, None, &[0x0f, 0xae], 3, false, memory.into());
    }
    /// `ldmxcsr m32`.
    pub(crate) fn ldmxcsr(&mut self, memory: impl Into<Rm>) {
        self.legacy(Size::Dword, None, &[0x0f, 0xae], 2, false, memory.into());
    }
    /// `vzeroupper` (AVX).
    pub(crate) fn vzeroupper(&mut self) {
        self.fixed(&[0xc5, 0xf8, 0x77]);
    }
    /// `kmovw k, r32` (AVX-512F): the low word of `source` into opmask `mask`.
    pub(crate) fn kmovw(&mut self, mask: u8, source: Gpr) {
        self.vex(
            Vex {
                map: Map::M0f,
                pp: Pp::None,
                w: false,
                l: false,
            },
            0x92,
            mask,
            0,
            Rm::Reg(source),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::x86_64::asm::{Code, Gpr::*, Mem};

    fn one(build: impl FnOnce(&mut Asm)) -> Code {
        let mut code = Code::new();
        build(&mut code.asm());
        assert_eq!(code.instructions(), 1);
        code
    }

    #[test]
    fn forms_match_the_sdm_encodings() {
        use FloatLanes::*;
        use Size::*;
        for (expected, code) in [
            (&[0xc5, 0xf8, 0x77][..], one(|a| a.vzeroupper())),
            (&[0xc5, 0xf8, 0x92, 0xc8], one(|a| a.kmovw(1, Gpr::Rax))),
            (&[0x0f, 0x16, 0xca], one(|a| a.movlhps(1, Rm::Xmm(2)))),
            (&[0x66, 0x0f, 0x7e, 0xf0], one(|a| a.movd_from_xmm(Rax, 6))),
            (&[0x66, 0x0f, 0x6e, 0xe0], one(|a| a.movd_to_xmm(4, Rax))),
            (&[0x66, 0x0f, 0x6e, 0xf1], one(|a| a.movd_to_xmm(6, Rcx))),
            (
                &[0x66, 0x0f, 0xd6, 0x48, 0x08],
                one(|a| a.movq_store(1, Mem::base(Rax, 8))),
            ),
            (
                &[0xf3, 0x0f, 0x6f, 0x44, 0x24, 0x10],
                one(|a| a.movdqu_load(0, Mem::base_disp8(Rsp, 0x10))),
            ),
            (
                &[0xf3, 0x0f, 0x7f, 0x9c, 0x24, 0x20, 0, 0, 0],
                one(|a| a.movdqu_store(Mem::base_disp32(Rsp, 0x20), 3)),
            ),
            (
                &[0x66, 0x0f, 0xc4, 0xc0, 0x01],
                one(|a| a.pinsrw(0, Rax, 1)),
            ),
            (
                &[0x66, 0x0f, 0x3a, 0x22, 0xc8, 2],
                one(|a| a.pinsrd(1, Rax, 2)),
            ),
            (
                &[0x0f, 0xae, 0x1c, 0x24],
                one(|a| a.stmxcsr(Mem::base(Rsp, 0))),
            ),
            (
                &[0x0f, 0xae, 0x54, 0x24, 0x04],
                one(|a| a.ldmxcsr(Mem::base(Rsp, 4))),
            ),
            (
                &[0x0f, 0xc6, 0xc1, 0x88],
                one(|a| a.shufps(0, Rm::Xmm(1), 0x88)),
            ),
            (&[0x0f, 0x14, 0xc1], one(|a| a.unpcklps(0, Rm::Xmm(1)))),
            (
                &[0x66, 0x0f, 0x14, 0xea],
                one(|a| a.unpcklpd(5, Rm::Xmm(2))),
            ),
            (
                &[0x66, 0x0f, 0x15, 0xc6],
                one(|a| a.unpckhpd(0, Rm::Xmm(6))),
            ),
            (&[0x66, 0x0f, 0x2e, 0xc1], one(|a| a.ucomisd(0, Rm::Xmm(1)))),
            (&[0x0f, 0x2e, 0xc7], one(|a| a.ucomiss(0, Rm::Xmm(7)))),
            (
                &[0xf3, 0x0f, 0x5a, 0xf6],
                one(|a| a.cvtss2sd(6, Rm::Xmm(6))),
            ),
            (
                &[0xf2, 0x0f, 0x5a, 0xc1],
                one(|a| a.cvtsd2ss(0, Rm::Xmm(1))),
            ),
            (&[0x0f, 0x5a, 0xf0], one(|a| a.cvtps2pd(6, Rm::Xmm(0)))),
            (
                &[0x66, 0x0f, 0x5a, 0xc1],
                one(|a| a.cvtpd2ps(0, Rm::Xmm(1))),
            ),
            (
                &[0xf2, 0x48, 0x0f, 0x2a, 0xc0],
                one(|a| a.cvtsi2float(ScalarDouble, Qword, 0, Rax)),
            ),
            (
                &[0xf3, 0x48, 0x0f, 0x2a, 0xf0],
                one(|a| a.cvtsi2float(ScalarSingle, Qword, 6, Rax)),
            ),
            (
                &[0xf3, 0x0f, 0x2c, 0xc0],
                one(|a| a.cvtfloat2si(ScalarSingle, true, Dword, Rax, Rm::Xmm(0))),
            ),
            (
                &[0xf2, 0x48, 0x0f, 0x2c, 0xc2],
                one(|a| a.cvtfloat2si(ScalarDouble, true, Qword, Rax, Rm::Xmm(2))),
            ),
            (&[0x66, 0x0f, 0xc5, 0xc0, 0], one(|a| a.pextrw(Rax, 0, 0))),
            (
                &[0x66, 0x41, 0x0f, 0xc5, 0xc9, 3],
                one(|a| a.pextrw(Rcx, 9, 3)),
            ),
            (
                &[0x66, 0x49, 0x0f, 0x3a, 0x16, 0x88, 0x40, 0, 0, 0, 1],
                one(|a| a.pextrq(Mem::base_disp32(R8, 0x40), 1, 1)),
            ),
            (
                &[0x41, 0x0f, 0x16, 0x90, 8, 0, 0, 0],
                one(|a| a.movhps(2, Mem::base_disp32(R8, 8))),
            ),
            (
                &[0xf3, 0x0f, 0xc2, 0xc8, 2],
                one(|a| a.float_compare(ScalarSingle, 1, Rm::Xmm(0), 2)),
            ),
            (
                &[0x66, 0x41, 0x0f, 0xc2, 0xc0, 0],
                one(|a| a.float_compare(PackedDouble, 0, Rm::Xmm(8), 0)),
            ),
            (
                &[0xf2, 0x0f, 0x5d, 0xc1],
                one(|a| a.float(FloatOp::Min, ScalarDouble, 0, Rm::Xmm(1))),
            ),
            (
                &[0x0f, 0x5f, 0xf1],
                one(|a| a.float(FloatOp::Max, PackedSingle, 6, Rm::Xmm(1))),
            ),
            (
                &[0x66, 0x0f, 0x51, 0xc0],
                one(|a| a.float(FloatOp::Sqrt, PackedDouble, 0, Rm::Xmm(0))),
            ),
            (
                &[0x0f, 0x58, 0xd4],
                one(|a| a.float(FloatOp::Add, PackedSingle, 2, Rm::Xmm(4))),
            ),
        ] {
            assert_eq!(code, expected);
        }
    }
}
