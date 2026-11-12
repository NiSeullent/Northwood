//! Capability-checked packed bit operations (Intel SDM Vol. 2).
use super::{Code, Evex, Gpr, Map, Mem, Pp, Rm, Vex};
use crate::TargetCpu;
use crate::aarch64::floating::{
    Format,
    integer::{IntegerFormat, IntegerWidth},
};
use crate::x86_64::isa::{ShiftOpcode as Shift, SimdOpcode as Host};

/// Transform each byte of `destination` by the bit matrix `matrix`
/// (GF2P8AFFINEQB): result bit i is the parity of the source bits selected by
/// matrix byte 7 - i. The matrix goes through `gpr` into `scratch`. The DIT
/// profile keeps its reviewed recipes.
pub(crate) fn byte_affine(
    code: &mut Code,
    destination: u8,
    scratch: u8,
    matrix: u64,
    gpr: Gpr,
    target: TargetCpu,
) -> bool {
    if !target.supports(TargetCpu::Gfni) || target.data_independent_timing() {
        return false;
    }
    debug_assert_ne!(destination, scratch);
    let mut asm = code.asm();
    asm.mov_imm64(gpr, matrix);
    asm.movq_to_xmm(scratch, gpr);
    asm.simd(
        Host::UnpackLowQword,
        scratch,
        Rm::Xmm(scratch),
        TargetCpu::Baseline,
    );
    asm.gf2p8affineqb(destination, Rm::Xmm(scratch), 0);
    true
}

/// The [`byte_affine`] matrix whose result bit i is source bit `source(i)`,
/// or zero where it is `None`.
pub(crate) fn byte_matrix(source: impl Fn(u8) -> Option<u8>) -> u64 {
    (0..8).fold(0, |matrix, bit| match source(bit) {
        Some(from) => matrix | 1 << from << (8 * (7 - bit)),
        None => matrix,
    })
}

/// Reverse each byte in `destination`, using `scratch` and RAX for the
/// anti-diagonal affine matrix.
pub(crate) fn reverse_bytes(
    code: &mut Code,
    destination: u8,
    scratch: u8,
    target: TargetCpu,
) -> bool {
    byte_affine(
        code,
        destination,
        scratch,
        byte_matrix(|bit| Some(7 - bit)),
        Gpr::Rax,
        target,
    )
}

/// VPSRAQ: shift each quadword of `xmm` right by `amount`, filling with its
/// sign; 64 or more leaves only the sign.
pub(crate) fn arithmetic_qword_right(
    code: &mut Code,
    xmm: u8,
    amount: u8,
    target: TargetCpu,
) -> bool {
    if !merge_supported(target) {
        return false;
    }
    let mut asm = code.asm();
    // EVEX.vvvv names the destination; ModRM.reg is the /4 extension.
    asm.evex(
        Evex {
            map: Map::M0f,
            pp: Pp::P66,
            w: true,
            length: 0,
            mask: 0,
            zeroing: false,
            broadcast: false,
            scale: 16,
        },
        0x72,
        4,
        xmm,
        Rm::Xmm(xmm),
    );
    asm.data(&[amount]);
    true
}

/// EVEX.128 needs usable opmask/ZMM state and VL in addition to its lane
/// extension. The feature declarations are independent for explicit targets.
pub(crate) fn population_supported(lane_bytes: u8, target: TargetCpu) -> bool {
    let extension = match lane_bytes {
        1 | 2 => TargetCpu::Avx512bitalg,
        4 | 8 => TargetCpu::Avx512vpopcntdq,
        _ => return false,
    };
    target.supports(TargetCpu::Avx512f)
        && target.supports(TargetCpu::Avx512vl)
        && target.supports(extension)
        && !target.data_independent_timing()
}

/// EVEX.128 ternary logic and rotates need F and VL. The DIT profile keeps
/// its recipe.
pub(crate) fn merge_supported(target: TargetCpu) -> bool {
    target.supports(TargetCpu::Avx512f)
        && target.supports(TargetCpu::Avx512vl)
        && !target.data_independent_timing()
}

/// VPTERNLOGD: each bit of `destination` becomes bit A:B:C of `table`, where
/// A is the old `destination` bit, B the `second` bit and C the `third`.
/// k0 disables opmasking.
pub(crate) fn ternary(
    code: &mut Code,
    destination: u8,
    second: u8,
    third: impl Into<Rm>,
    table: u8,
    target: TargetCpu,
) -> bool {
    if !merge_supported(target) {
        return false;
    }
    let mut asm = code.asm();
    asm.evex(
        Evex {
            map: Map::M0f3a,
            pp: Pp::P66,
            w: false,
            length: 0,
            mask: 0,
            zeroing: false,
            broadcast: false,
            scale: 16,
        },
        0x25,
        destination,
        second,
        third.into(),
    );
    asm.data(&[table]);
    true
}

/// Select each bit from the new destination where mask is set, else from old:
/// table 0xe4 over destination:old:mask.
pub(crate) fn merge(
    code: &mut Code,
    destination: u8,
    old: u8,
    mask: u8,
    target: TargetCpu,
) -> bool {
    ternary(code, destination, old, Rm::Xmm(mask), 0xe4, target)
}

/// VPRORQ: rotate each quadword of `source` right by `rotation` into
/// `destination`.
pub(crate) fn rotate_qwords_right(
    code: &mut Code,
    destination: u8,
    source: u8,
    rotation: u8,
    target: TargetCpu,
) -> bool {
    if !merge_supported(target) {
        return false;
    }
    let mut asm = code.asm();
    // EVEX.vvvv names the destination; ModRM.reg is the /0 extension.
    asm.evex(
        Evex {
            map: Map::M0f,
            pp: Pp::P66,
            w: true,
            length: 0,
            mask: 0,
            zeroing: false,
            broadcast: false,
            scale: 16,
        },
        0x72,
        0,
        destination,
        Rm::Xmm(source),
    );
    asm.data(&[rotation % 64]);
    true
}

/// VEX.128 VPBLENDVB needs AVX; the represented AVX2 target is sufficient.
/// Each mask byte must be 0 or255, as produced by the predicate expander.
/// Unlike the legacy form, the mask is explicit and cannot clobber XMM0.
#[cfg(test)]
pub(crate) fn blend_bytes(
    code: &mut Code,
    destination: u8,
    old: u8,
    mask: u8,
    target: TargetCpu,
) -> bool {
    if !target.supports(TargetCpu::Avx2) || target.data_independent_timing() {
        return false;
    }
    assert!(
        destination < 16 && old < 16 && mask < 16,
        "VEX blend uses legacy XMM register encodings"
    );
    let mut asm = code.asm();
    asm.vex(
        Vex {
            map: Map::M0f3a,
            pp: Pp::P66,
            w: false,
            l: false,
        },
        0x4c,
        destination,
        old,
        Rm::Xmm(destination),
    );
    asm.data(&[mask << 4]);
    true
}

/// Byte dot products need AVX-VNNI-INT8, AVX-VNNI, or AVX-512 VNNI with F
/// and VL. The DIT profile keeps its reviewed recipe.
pub(crate) fn byte_dot_supported(target: TargetCpu) -> bool {
    (target.supports(TargetCpu::AvxVnniInt8)
        || target.supports(TargetCpu::AvxVnni)
        || target.supports(TargetCpu::Avx512f.with_avx512vl().with_avx512vnni()))
        && !target.data_independent_timing()
}

/// AVX512-FP16's packed forms on XMM registers need VL. The DIT profile keeps
/// its reviewed recipes.
pub(crate) fn half_supported(target: TargetCpu) -> bool {
    target.supports(TargetCpu::Avx512fp16.with_avx512vl()) && !target.data_independent_timing()
}

/// The writemask of [`half_arithmetic`].
const HALF_WRITEMASK: u8 = 1;

/// K1 selects the low `lanes` Half lanes, through EAX.
pub(crate) fn half_writemask(code: &mut Code, lanes: u8) {
    let mut asm = code.asm();
    asm.mov_imm32(Gpr::Rax, (1 << lanes) - 1);
    asm.kmovw(HALF_WRITEMASK, Gpr::Rax);
}

/// An AVX512-FP16 packed binary operation, by its map 5 opcode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HalfOpcode {
    Add = 0x58,
    Multiply = 0x59,
    Subtract = 0x5c,
    Divide = 0x5e,
}

/// VADDPH, VMULPH, VSUBPH or VDIVPH of the XMM registers `destination` and `source`
/// under MXCSR.RC, in the [`half_writemask`] lanes: the others clear and raise
/// no status. The instructions ignore MXCSR.DAZ and FTZ.
pub(crate) fn half_arithmetic(code: &mut Code, opcode: HalfOpcode, destination: u8, source: u8) {
    code.asm().evex(
        Evex {
            map: Map::M5,
            pp: Pp::None,
            w: false,
            length: 0,
            mask: HALF_WRITEMASK,
            zeroing: true,
            broadcast: false,
            scale: 16,
        },
        opcode as u8,
        destination,
        destination,
        Rm::Xmm(source),
    );
}

/// AVX10.2's VMINMAX. The DIT profile keeps its reviewed recipe.
pub(crate) fn extrema_supported(target: TargetCpu) -> bool {
    target.supports(TargetCpu::Avx102) && !target.data_independent_timing()
}

/// VMINMAX's immediate: the result's sign follows the selection, so -0 orders
/// below +0, and a NaN operand propagates; bit 0 selects maximum.
const EXTREMA_SIGN_FROM_SELECTION: u8 = 1 << 2;

/// VMINMAXPS/PD, or SS/SD when `scalar`: the IEEE 754-2019 minimum or `maximum`
/// of `destination` and `right` into `destination`.
pub(crate) fn extrema(
    code: &mut Code,
    format: Format,
    scalar: bool,
    maximum: bool,
    destination: u8,
    right: u8,
) {
    let mut asm = code.asm();
    asm.evex(
        Evex {
            map: Map::M0f3a,
            pp: Pp::P66,
            w: format == Format::Double,
            length: 0,
            mask: 0,
            zeroing: false,
            broadcast: false,
            scale: 16,
        },
        0x52 + u8::from(scalar),
        destination,
        destination,
        Rm::Xmm(right),
    );
    asm.data(&[EXTREMA_SIGN_FROM_SELECTION | u8::from(maximum)]);
}

/// VFMADD132PH, VFMSUB132PH, VFNMADD132PH or VFNMSUB132PH, by the map 6 opcode
/// they share with FMA3: `left` * `right` +/- `addend` into `left`, rounded once
/// under MXCSR.RC, in the [`half_writemask`] lanes as [`half_arithmetic`] does.
pub(crate) fn half_fused(code: &mut Code, opcode: u8, left: u8, addend: u8, right: u8) {
    code.asm().evex(
        Evex {
            map: Map::M6,
            pp: Pp::P66,
            w: false,
            length: 0,
            mask: HALF_WRITEMASK,
            zeroing: true,
            broadcast: false,
            scale: 16,
        },
        opcode,
        left,
        addend,
        Rm::Xmm(right),
    );
}

/// VPDPWUUD (AVX-VNNI-INT16): add to each dword of `sum` the two products of
/// its unsigned words in `left` and `right`, wrapping. The DIT profile keeps
/// its reviewed recipe.
pub(crate) fn unsigned_word_dot(
    code: &mut Code,
    sum: u8,
    left: u8,
    right: u8,
    target: TargetCpu,
) -> bool {
    if !target.supports(TargetCpu::AvxVnniInt16) || target.data_independent_timing() {
        return false;
    }
    code.asm().vex(
        Vex {
            map: Map::M0f38,
            pp: Pp::None,
            w: false,
            l: false,
        },
        0xd2,
        sum,
        left,
        Rm::Xmm(right),
    );
    true
}

/// The host instruction behind a [`ByteDot`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DotForm {
    /// AVX-VNNI-INT8: VPDPBUUD, VPDPBSUD or VPDPBSSD.
    Int8,
    /// AVX-VNNI's VEX VPDPBUSD.
    Vex,
    /// AVX-512 VNNI's EVEX.128 VPDPBUSD.
    Evex,
}

/// A byte dot product recipe: each dword gains the wrapping sum of the four
/// products of its bytes in a left and a right operand, each signed as the
/// recipe's signedness says. VPDPBUSD multiplies unsigned by signed bytes, so
/// equal signedness without AVX-VNNI-INT8 is [`Self::biased`]: one operand's
/// sign bits are flipped, (x ^ 0x80) - 0x80 = x, and a correction from the
/// other operand's byte sum is subtracted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ByteDot {
    signed: [bool; 2],
    form: DotForm,
}

impl ByteDot {
    /// The recipe for `signed` operands on `target`, if it has one.
    pub(crate) fn new(signed: [bool; 2], target: TargetCpu) -> Option<Self> {
        if !byte_dot_supported(target) {
            return None;
        }
        let form = if target.supports(TargetCpu::AvxVnniInt8) {
            DotForm::Int8
        } else if target.supports(TargetCpu::AvxVnni) {
            DotForm::Vex
        } else {
            DotForm::Evex
        };
        Some(Self { signed, form })
    }

    /// Whether one operand is flipped and the result corrected.
    pub(crate) fn biased(self) -> bool {
        self.form != DotForm::Int8 && self.signed[0] == self.signed[1]
    }

    /// 0x80 in every byte of `xmm`: all-ones words shifted to 0x8000
    /// saturate to 0x80 bytes.
    pub(crate) fn bias(code: &mut Code, xmm: u8) {
        code.asm()
            .simd(Host::EqualByte, xmm, Rm::Xmm(xmm), TargetCpu::Baseline);
        Shift::LeftWord.emit(code, xmm, 15);
        code.asm().simd(
            Host::PackSignedWordToByte,
            xmm,
            Rm::Xmm(xmm),
            TargetCpu::Baseline,
        );
    }

    /// Flip the sign bit of each byte of `xmm` with [`Self::bias`] `bias`.
    pub(crate) fn flip(code: &mut Code, xmm: u8, bias: u8) {
        code.asm()
            .simd(Host::Xor, xmm, Rm::Xmm(bias), TargetCpu::Baseline);
    }

    /// VPDPBUSD, `unsigned` bytes by `signed` bytes, into `sum`.
    fn unsigned_by_signed(self, code: &mut Code, sum: u8, unsigned: u8, signed: u8) {
        if self.form == DotForm::Vex {
            code.asm().vex(
                Vex {
                    map: Map::M0f38,
                    pp: Pp::P66,
                    w: false,
                    l: false,
                },
                0x50,
                sum,
                unsigned,
                Rm::Xmm(signed),
            );
        } else {
            code.asm().evex(
                Evex {
                    map: Map::M0f38,
                    pp: Pp::P66,
                    w: false,
                    length: 0,
                    mask: 0,
                    zeroing: false,
                    broadcast: false,
                    scale: 16,
                },
                0x50,
                sum,
                unsigned,
                Rm::Xmm(signed),
            );
        }
    }

    /// Add to `sum` the dot products of `[left, right]`. A biased recipe
    /// expects operand `flipped` (0 or 1) to have been flipped.
    pub(crate) fn accumulate(
        self,
        code: &mut Code,
        sum: u8,
        [left, right]: [u8; 2],
        flipped: usize,
    ) {
        if self.form == DotForm::Int8 {
            // VEX.pp names the signedness: VPDPBUUD, VPDPBSUD (signed first
            // source), VPDPBSSD.
            let (pp, [first, second]) = match self.signed {
                [false, false] => (Pp::None, [left, right]),
                [true, true] => (Pp::Pf2, [left, right]),
                [true, false] => (Pp::Pf3, [left, right]),
                [false, true] => (Pp::Pf3, [right, left]),
            };
            code.asm().vex(
                Vex {
                    map: Map::M0f38,
                    pp,
                    w: false,
                    l: false,
                },
                0x50,
                sum,
                first,
                Rm::Xmm(second),
            );
            return;
        }
        let operands = [left, right];
        // A flipped signed operand becomes unsigned, a flipped unsigned one
        // signed; otherwise each keeps its own signedness.
        let unsigned = match self.signed {
            [true, true] => flipped,
            [false, false] => 1 - flipped,
            [left_signed, _] => usize::from(left_signed),
        };
        self.unsigned_by_signed(code, sum, operands[unsigned], operands[1 - unsigned]);
    }

    /// Add to `correction` what the flip of a biased recipe adds to every
    /// product sum, from the unflipped operand `other` and [`Self::bias`]
    /// `bias`: 128 times its byte sum for signed operands, as
    /// (x + 128)·y - 128·y = x·y, and -128 times it for unsigned ones, as
    /// x·(y - 128) + 128·x = x·y. Subtracting it completes the dot products.
    pub(crate) fn correction(self, code: &mut Code, correction: u8, other: u8, bias: u8) {
        debug_assert!(self.biased());
        if self.signed[0] {
            self.unsigned_by_signed(code, correction, bias, other);
        } else {
            self.unsigned_by_signed(code, correction, other, bias);
        }
    }
}

/// Set each dword of `sum` to the wrapping sum of the four products of the
/// bytes in that dword of each `[left, right]` pair, each operand's bytes
/// signed as `signed` says (SDOT, UDOT, USDOT, SUDOT and the matrix forms).
/// Callers add the old destination afterwards, so a chain of accumulations
/// waits on one PADDD rather than on the multiply. A [`ByteDot::biased`]
/// recipe flips each left operand in place, with the bias in `scratch[0]` and
/// the correction collected in `scratch[1]`.
pub(crate) fn byte_dot(
    code: &mut Code,
    sum: u8,
    pairs: &[[u8; 2]],
    signed: [bool; 2],
    scratch: [u8; 2],
    target: TargetCpu,
) -> bool {
    let Some(dot) = ByteDot::new(signed, target) else {
        return false;
    };
    let [bias, correction] = scratch;
    if dot.biased() {
        ByteDot::bias(code, bias);
        code.asm().simd(
            Host::Xor,
            correction,
            Rm::Xmm(correction),
            TargetCpu::Baseline,
        );
    }
    code.asm()
        .simd(Host::Xor, sum, Rm::Xmm(sum), TargetCpu::Baseline);
    for &[left, right] in pairs {
        if dot.biased() {
            dot.correction(code, correction, right, bias);
            ByteDot::flip(code, left, bias);
        }
        dot.accumulate(code, sum, [left, right], 0);
    }
    if dot.biased() {
        code.asm().simd(
            Host::SubDword,
            sum,
            Rm::Xmm(correction),
            TargetCpu::Baseline,
        );
    }
    true
}

/// TBL and TBX on `lane_bytes` lanes need AVX-512 with VL: BW and VBMI for
/// bytes, BW for halfwords and F alone for dwords and quadwords. The DIT
/// profile keeps its reviewed recipe.
pub(crate) fn table_supported(lane_bytes: u8, target: TargetCpu) -> bool {
    let halfwords = TargetCpu::Avx512f.with_avx512vl().with_avx512bw();
    let required = match lane_bytes {
        1 => halfwords.with_avx512vbmi(),
        2 => halfwords,
        4 | 8 => TargetCpu::Avx512f.with_avx512vl(),
        _ => return false,
    };
    target.supports(required) && !target.data_independent_timing()
}

/// EVEX.128 with opmask `mask`, zeroing the inactive lanes when `zeroing`.
const fn masked(map: Map, pp: Pp, w: bool, mask: u8, zeroing: bool) -> Evex {
    Evex {
        map,
        pp,
        w,
        length: 0,
        mask,
        zeroing,
        broadcast: false,
        scale: 16,
    }
}

const IN_RANGE: u8 = 1;
const UPPER_PAIR: u8 = 2;
const LESS_THAN: u8 = 1;
const TABLE_BYTES: u32 = 16;

/// `value` in every `lane_bytes` lane of `xmm` through EAX:
/// VPBROADCASTB/W/D/Q.
fn broadcast(code: &mut Code, lane_bytes: u8, xmm: u8, value: u32) {
    let opcode = match lane_bytes {
        1 => 0x7a,
        2 => 0x7b,
        _ => 0x7c,
    };
    let mut asm = code.asm();
    asm.mov_imm32(Gpr::Rax, value);
    asm.evex(
        masked(Map::M0f38, Pp::P66, lane_bytes == 8, 0, false),
        opcode,
        xmm,
        0,
        Rm::Reg(Gpr::Rax),
    );
}

/// Look up each `lane_bytes` lane of `indices` in the concatenation of
/// `tables`, one to four 16-byte operands. An index beyond them gives zero,
/// or with `preserve` leaves the lane already in `result` (TBX). VPCMPU{B,W,
/// D,Q} marks the in-range indices in K1; VPERMB or VPERMW reads one table,
/// VPERMT2{B,W,D,Q} a pair (a lone D or Q table pairs with itself), and a
/// second pair is blended in by the index bit above the first (VPTESTM into
/// K2). EVEX.W selects the H and Q forms. Clobbers `scratch`, EAX, K1 and
/// K2.
#[expect(
    clippy::too_many_arguments,
    reason = "the lane size, both operands, tables, merge mode and scratch are independent"
)]
pub(crate) fn table(
    code: &mut Code,
    lane_bytes: u8,
    result: u8,
    indices: u8,
    tables: &[Mem],
    preserve: bool,
    scratch: [u8; 2],
    target: TargetCpu,
) -> bool {
    if !table_supported(lane_bytes, target) {
        return false;
    }
    assert!(
        (1..=4).contains(&tables.len()),
        "TBL reads one to four tables"
    );
    let w = matches!(lane_bytes, 2 | 8);
    let dwords = lane_bytes >= 4;
    let permute = |mask, zeroing| masked(Map::M0f38, Pp::P66, w, mask, zeroing);
    let blend = |mask| {
        masked(
            Map::M0f,
            if dwords { Pp::Pf3 } else { Pp::Pf2 },
            w,
            mask,
            false,
        )
    };
    let [lookup, other] = scratch;
    let lanes = TABLE_BYTES / u32::from(lane_bytes);
    broadcast(code, lane_bytes, other, lanes * tables.len() as u32);
    code.asm().evex(
        masked(Map::M0f3a, Pp::P66, w, 0, false),
        if dwords { 0x1e } else { 0x3e },
        IN_RANGE,
        indices,
        Rm::Xmm(other),
    );
    code.asm().data(&[LESS_THAN]);
    if let ([table], false) = (tables, dwords) {
        code.asm().evex(
            permute(IN_RANGE, !preserve),
            0x8d,
            result,
            indices,
            (*table).into(),
        );
        return true;
    }
    let lookup = if preserve { lookup } else { result };
    let pair = |code: &mut Code, xmm: u8, low: Mem, high: Mem| {
        Host::LoadUnaligned.emit(code, xmm, low, TargetCpu::Baseline);
        code.asm().evex(
            permute(IN_RANGE, true),
            if dwords { 0x7e } else { 0x7d },
            xmm,
            indices,
            high.into(),
        );
    };
    pair(code, lookup, tables[0], tables[tables.len().min(2) - 1]);
    if let Some([.., third]) = tables.get(2..) {
        broadcast(code, lane_bytes, other, 2 * lanes);
        code.asm().evex(
            permute(0, false),
            if dwords { 0x27 } else { 0x26 },
            UPPER_PAIR,
            indices,
            Rm::Xmm(other),
        );
        // Three tables repeat the third; its indices 48..63 are out of range.
        pair(code, other, tables[2], *third);
        code.asm()
            .evex(blend(UPPER_PAIR), 0x6f, lookup, 0, Rm::Xmm(other));
    }
    if preserve {
        code.asm()
            .evex(blend(IN_RANGE), 0x6f, result, 0, Rm::Xmm(lookup));
    }
    true
}

/// Quadword minima, maxima and absolute values need AVX-512 F and VL. The
/// DIT profile keeps its reviewed recipe.
pub(crate) fn quadword_supported(target: TargetCpu) -> bool {
    target.supports(TargetCpu::Avx512f.with_avx512vl()) && !target.data_independent_timing()
}

/// EVEX.128.66.0F38.W1, unmasked.
const QUADWORD: Evex = masked(Map::M0f38, Pp::P66, true, 0, false);

/// VPMAXSQ, VPMINSQ, VPMAXUQ or VPMINUQ: each quadword of `destination`
/// becomes the larger or smaller of it and the one in `source`.
pub(crate) fn quadword_extremum(
    code: &mut Code,
    destination: u8,
    source: u8,
    signed: bool,
    maximum: bool,
    target: TargetCpu,
) -> bool {
    if !quadword_supported(target) {
        return false;
    }
    let opcode = match (signed, maximum) {
        (true, false) => 0x39,
        (false, false) => 0x3b,
        (true, true) => 0x3d,
        (false, true) => 0x3f,
    };
    code.asm()
        .evex(QUADWORD, opcode, destination, destination, Rm::Xmm(source));
    true
}

/// VPMULLQ: the low 64 bits of each quadword product of `destination` and
/// `source`, into `destination`. It needs AVX-512 DQ as well; the DIT
/// profile keeps its reviewed recipe.
pub(crate) fn quadword_multiply(
    code: &mut Code,
    destination: u8,
    source: u8,
    target: TargetCpu,
) -> bool {
    if !quadword_supported(target) || !target.supports(TargetCpu::Avx512dq) {
        return false;
    }
    code.asm()
        .evex(QUADWORD, 0x40, destination, destination, Rm::Xmm(source));
    true
}

/// VPABSQ: the absolute value of each quadword of `xmm`, in place.
pub(crate) fn quadword_absolute(code: &mut Code, xmm: u8, target: TargetCpu) -> bool {
    if !quadword_supported(target) {
        return false;
    }
    code.asm().evex(QUADWORD, 0x1f, xmm, 0, Rm::Xmm(xmm));
    true
}

/// Lane comparisons into all-ones or zero lanes need AVX-512 VL with BW for
/// bytes and halfwords or DQ for dwords and quadwords (VPMOVM2*). The DIT
/// profile keeps its reviewed recipe.
pub(crate) fn compare_supported(lane_bytes: u8, target: TargetCpu) -> bool {
    let required = match lane_bytes {
        1 | 2 => TargetCpu::Avx512f.with_avx512vl().with_avx512bw(),
        4 | 8 => TargetCpu::Avx512f.with_avx512vl().with_avx512dq(),
        _ => return false,
    };
    target.supports(required) && !target.data_independent_timing()
}

/// VPCMP immediates: whether the left lane is equal to, below, at most,
/// unequal to, at least or above the right one.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Predicate {
    Equal = 0,
    Less = 1,
    LessEqual = 2,
    NotEqual = 4,
    GreaterEqual = 5,
    Greater = 6,
}

/// The mask in K1 as all-ones or zero `lane_bytes` lanes of `xmm`:
/// VPMOVM2B/W/D/Q.
fn lanes_from_mask(code: &mut Code, lane_bytes: u8, xmm: u8) {
    code.asm().evex(
        masked(Map::M0f38, Pp::Pf3, matches!(lane_bytes, 2 | 8), 0, false),
        if lane_bytes <= 2 { 0x28 } else { 0x38 },
        xmm,
        0,
        Rm::Xmm(IN_RANGE),
    );
}

/// All ones in each `lane_bytes` lane of `left` where `predicate` holds
/// against `right`, signed or unsigned, and zero elsewhere:
/// VPCMP[U]{B,W,D,Q} into K1, then VPMOVM2. Clobbers K1.
pub(crate) fn compare(
    code: &mut Code,
    lane_bytes: u8,
    signed: bool,
    predicate: Predicate,
    left: u8,
    right: u8,
    target: TargetCpu,
) -> bool {
    if !compare_supported(lane_bytes, target) {
        return false;
    }
    let opcode = if lane_bytes <= 2 { 0x3e } else { 0x1e } + u8::from(signed);
    code.asm().evex(
        masked(Map::M0f3a, Pp::P66, matches!(lane_bytes, 2 | 8), 0, false),
        opcode,
        IN_RANGE,
        left,
        Rm::Xmm(right),
    );
    code.asm().data(&[predicate as u8]);
    lanes_from_mask(code, lane_bytes, left);
    true
}

/// All ones in each `lane_bytes` lane of `left` that shares a set bit with
/// `right`, and zero elsewhere: VPTESTM{B,W,D,Q} into K1, then VPMOVM2.
/// Clobbers K1.
pub(crate) fn test(
    code: &mut Code,
    lane_bytes: u8,
    left: u8,
    right: u8,
    target: TargetCpu,
) -> bool {
    if !compare_supported(lane_bytes, target) {
        return false;
    }
    code.asm().evex(
        masked(Map::M0f38, Pp::P66, matches!(lane_bytes, 2 | 8), 0, false),
        if lane_bytes <= 2 { 0x26 } else { 0x27 },
        IN_RANGE,
        left,
        Rm::Xmm(right),
    );
    lanes_from_mask(code, lane_bytes, left);
    true
}

/// Truncate each `source_lane_bytes` lane of `source` to its low half, into
/// consecutive lanes of the low quadword of `destination`, clearing its high
/// quadword: VPMOVWB (AVX-512 BW and VL), VPMOVDW or VPMOVQD (F and VL).
/// The DIT profile keeps its reviewed recipe. No EFLAGS or memory.
pub(crate) fn truncating_narrow(
    code: &mut Code,
    source_lane_bytes: u8,
    destination: u8,
    source: u8,
    target: TargetCpu,
) -> bool {
    let required = match source_lane_bytes {
        2 => TargetCpu::Avx512f.with_avx512vl().with_avx512bw(),
        4 | 8 => TargetCpu::Avx512f.with_avx512vl(),
        _ => return false,
    };
    if !target.supports(required) || target.data_independent_timing() {
        return false;
    }
    let opcode = match source_lane_bytes {
        2 => 0x30,
        4 => 0x33,
        _ => 0x35,
    };
    // ModRM.reg names the wide source and r/m the narrow destination.
    code.asm().evex(
        masked(Map::M0f38, Pp::Pf3, false, 0, false),
        opcode,
        source,
        0,
        Rm::Xmm(destination),
    );
    true
}

/// Saturating narrows of `source_lane_bytes` lanes need AVX-512 VL with BW
/// for halfword sources (VPMOVSWB) and F for wider ones, the lane compare's
/// extensions for the saturation mask and SSE4.1 to re-extend.
pub(crate) fn saturating_narrow_supported(source_lane_bytes: u8, target: TargetCpu) -> bool {
    matches!(source_lane_bytes, 2 | 4 | 8)
        && compare_supported(source_lane_bytes, target)
        && target.supports(TargetCpu::Sse41)
}

/// Saturate each `source_lane_bytes` lane of `source`, signed or unsigned as
/// `signed[0]`, to the half-width range that `signed[1]` names, into
/// consecutive lanes of the low quadword of `destination`, clearing its high
/// quadword: VPMOVS{WB,DW,QD} or VPMOVUS{WB,DW,QD}, after a signed maximum
/// with zero for a signed source and unsigned result. With `saturated`, each
/// source lane of that register is all ones where the result differs from
/// the source (PMOVSX or PMOVZX back, then VPCMP) and zero elsewhere.
/// `destination` and `saturated` must differ from `source`. Clobbers K1.
pub(crate) fn saturating_narrow(
    code: &mut Code,
    source_lane_bytes: u8,
    signed: [bool; 2],
    destination: u8,
    source: u8,
    saturated: Option<u8>,
    target: TargetCpu,
) -> bool {
    if !saturating_narrow_supported(source_lane_bytes, target) {
        return false;
    }
    let [signed_source, signed_result] = signed;
    assert!(
        signed_source || !signed_result,
        "Arm has no unsigned-to-signed narrow"
    );
    assert!(
        destination != source && saturated != Some(source),
        "the source is read last"
    );
    let mut narrowed = source;
    if signed_source && !signed_result {
        Host::Xor.register(code, destination, destination, target);
        match source_lane_bytes {
            2 => Host::MaxSignedWord.register(code, destination, source, target),
            4 => Host::MaxSignedDword.register(code, destination, source, target),
            _ => {
                quadword_extremum(code, destination, source, true, true, target);
            }
        }
        narrowed = destination;
    }
    let opcode = match source_lane_bytes {
        2 => 0x10,
        4 => 0x13,
        _ => 0x15,
    } + if signed_result { 0x10 } else { 0 };
    // ModRM.reg names the wide source and r/m the narrow destination.
    code.asm().evex(
        masked(Map::M0f38, Pp::Pf3, false, 0, false),
        opcode,
        narrowed,
        0,
        Rm::Xmm(destination),
    );
    if let Some(saturated) = saturated {
        let extend = match (source_lane_bytes, signed_result) {
            (2, true) => Host::SignExtendByteToWord,
            (2, false) => Host::ZeroExtendByteToWord,
            (4, true) => Host::SignExtendWordToDword,
            (4, false) => Host::ZeroExtendWordToDword,
            (_, true) => Host::SignExtendDwordToQword,
            (_, false) => Host::ZeroExtendDwordToQword,
        };
        extend.register(code, saturated, destination, target);
        compare(
            code,
            source_lane_bytes,
            true,
            Predicate::NotEqual,
            saturated,
            source,
            target,
        );
    }
    true
}

/// AVX-512 conversions between S/D lanes and W/X integers: unsigned dwords
/// need F and VL, quadwords DQ as well. SSE2 already converts signed dwords.
/// Half lanes convert with H integers on AVX512-FP16. The DIT profile keeps
/// its reviewed recipe.
pub(crate) fn packed_conversion_supported(
    format: Format,
    integer: IntegerFormat,
    target: TargetCpu,
) -> bool {
    let required = match (format, integer.width, integer.signed) {
        (Format::Half, IntegerWidth::Bits16, _) => return half_supported(target),
        (Format::Half, ..) | (_, IntegerWidth::Bits16, _) | (_, IntegerWidth::Bits32, true) => {
            return false;
        }
        (_, IntegerWidth::Bits32, false) => TargetCpu::Avx512f.with_avx512vl(),
        (_, IntegerWidth::Bits64, _) => TargetCpu::Avx512f.with_avx512vl().with_avx512dq(),
    };
    target.supports(required) && !target.data_independent_timing()
}

/// EVEX.128 for a packed conversion.
const fn packed_conversion(pp: Pp, w: bool) -> Evex {
    Evex {
        map: Map::M0f,
        pp,
        w,
        length: 0,
        mask: 0,
        zeroing: false,
        broadcast: false,
        scale: 16,
    }
}

/// EVEX.128 for a packed Half conversion.
const fn packed_half_conversion(pp: Pp) -> Evex {
    Evex {
        map: Map::M5,
        pp,
        w: false,
        length: 0,
        mask: 0,
        zeroing: false,
        broadcast: false,
        scale: 16,
    }
}

/// Convert the `integer` lanes of `xmm` in place to float lanes rounded by
/// MXCSR: VCVTW2PH, VCVTUW2PH, VCVTUDQ2PS/PD, VCVTQQ2PS/PD or VCVTUQQ2PS/PD.
/// Inexact results set PE. Between widths the two narrow lanes are the low
/// quadword; a narrow result clears the high one.
pub(crate) fn packed_from_integer(
    code: &mut Code,
    xmm: u8,
    format: Format,
    integer: IntegerFormat,
    target: TargetCpu,
) -> bool {
    if !packed_conversion_supported(format, integer, target) {
        return false;
    }
    if format == Format::Half {
        let pp = if integer.signed { Pp::Pf3 } else { Pp::Pf2 };
        let prefix = packed_half_conversion(pp);
        code.asm().evex(prefix, 0x7d, xmm, 0, Rm::Xmm(xmm));
        return true;
    }
    let (pp, opcode) = match (format, integer.signed) {
        (Format::Single, true) => (Pp::None, 0x5b),
        (Format::Single, false) => (Pp::Pf2, 0x7a),
        (_, true) => (Pp::Pf3, 0xe6),
        (_, false) => (Pp::Pf3, 0x7a),
    };
    let quadword = integer.width == IntegerWidth::Bits64;
    code.asm().evex(
        packed_conversion(pp, quadword),
        opcode,
        xmm,
        0,
        Rm::Xmm(xmm),
    );
    true
}

/// Convert the float lanes of `xmm` in place to `integer` lanes, truncated or
/// rounded by MXCSR: VCVT(T)PH2(U)W, VCVT(T)PS2UDQ/QQ/UQQ or
/// VCVT(T)PD2UDQ/QQ/UQQ, with lanes placed as for [`packed_from_integer`]. An
/// unrepresentable result sets IE and PE marks an inexact one, as for the
/// scalar forms.
pub(crate) fn packed_to_integer(
    code: &mut Code,
    xmm: u8,
    format: Format,
    integer: IntegerFormat,
    truncate: bool,
    target: TargetCpu,
) -> bool {
    if !packed_conversion_supported(format, integer, target) {
        return false;
    }
    if format == Format::Half {
        let pp = if integer.signed { Pp::P66 } else { Pp::None };
        let opcode = 0x7d - u8::from(truncate);
        let prefix = packed_half_conversion(pp);
        code.asm().evex(prefix, opcode, xmm, 0, Rm::Xmm(xmm));
        return true;
    }
    let pp = if integer.width == IntegerWidth::Bits64 {
        Pp::P66
    } else {
        Pp::None
    };
    let opcode = if integer.signed { 0x7b } else { 0x79 } - u8::from(truncate);
    code.asm().evex(
        packed_conversion(pp, format == Format::Double),
        opcode,
        xmm,
        0,
        Rm::Xmm(xmm),
    );
    true
}

/// Scalar unsigned conversions need F. The DIT profile keeps its recipe.
pub(crate) fn unsigned_scalar_supported(target: TargetCpu) -> bool {
    target.supports(TargetCpu::Avx512f) && !target.data_independent_timing()
}

/// EVEX.LIG for a scalar conversion; a 64-bit integer sets W.
fn scalar_conversion(format: Format, wide: bool) -> Evex {
    Evex {
        map: if format == Format::Half {
            Map::M5
        } else {
            Map::M0f
        },
        pp: match format {
            Format::Half | Format::Single => Pp::Pf3,
            Format::Double => Pp::Pf2,
        },
        w: wide,
        length: 0,
        mask: 0,
        zeroing: false,
        broadcast: false,
        scale: (format.bits() / u8::BITS) as u8,
    }
}

/// VCVTSI2SH: the low Half lane of `xmm` from signed `source` (64-bit when
/// `wide`), rounded by MXCSR; the other lanes of `xmm` are kept.
pub(crate) fn half_from_signed(code: &mut Code, xmm: u8, wide: bool, source: Gpr) {
    code.asm().evex(
        scalar_conversion(Format::Half, wide),
        0x2a,
        xmm,
        xmm,
        Rm::Reg(source),
    );
}

/// VCVT(T)SH2SI: signed `destination` (64-bit when `wide`) from the low Half
/// lane of `xmm`, truncated or rounded by MXCSR.
pub(crate) fn half_to_signed(
    code: &mut Code,
    destination: Gpr,
    xmm: u8,
    wide: bool,
    truncate: bool,
) {
    code.asm().evex(
        scalar_conversion(Format::Half, wide),
        0x2d - u8::from(truncate),
        destination as u8,
        0,
        Rm::Xmm(xmm),
    );
}

/// VCVTSD2SH: the low Half lane of `xmm` from the low Double lane of `source`,
/// rounded by MXCSR.
pub(crate) fn half_from_double(code: &mut Code, xmm: u8, source: u8) {
    code.asm().evex(
        Evex {
            map: Map::M5,
            pp: Pp::Pf2,
            w: true,
            length: 0,
            mask: 0,
            zeroing: false,
            broadcast: false,
            scale: 8,
        },
        0x5a,
        xmm,
        xmm,
        Rm::Xmm(source),
    );
}

/// VCVTUSI2SH/SS/SD: the low float lane of `xmm` from unsigned `source`
/// (64-bit when `wide`), rounded by MXCSR; the other lanes of `xmm` are kept.
pub(crate) fn scalar_from_unsigned(
    code: &mut Code,
    xmm: u8,
    format: Format,
    wide: bool,
    source: Gpr,
    target: TargetCpu,
) -> bool {
    if !unsigned_scalar_supported(target) {
        return false;
    }
    code.asm().evex(
        scalar_conversion(format, wide),
        0x7b,
        xmm,
        xmm,
        Rm::Reg(source),
    );
    true
}

/// VCVT(T)SH2USI, VCVT(T)SS2USI and VCVT(T)SD2USI: unsigned `destination`
/// (64-bit when `wide`) from the low float lane of `xmm`, truncated or
/// rounded by MXCSR. An unrepresentable result, negative ones included, sets
/// IE.
pub(crate) fn scalar_to_unsigned(
    code: &mut Code,
    destination: Gpr,
    xmm: u8,
    format: Format,
    wide: bool,
    truncate: bool,
    target: TargetCpu,
) -> bool {
    if !unsigned_scalar_supported(target) {
        return false;
    }
    code.asm().evex(
        scalar_conversion(format, wide),
        0x79 - u8::from(truncate),
        destination as u8,
        0,
        Rm::Xmm(xmm),
    );
    true
}

/// Packed dword/qword leading-zero counts require CD in addition to F and VL.
pub(crate) fn leading_supported(lane_bytes: u8, target: TargetCpu) -> bool {
    matches!(lane_bytes, 4 | 8)
        && target.supports(TargetCpu::Avx512f)
        && target.supports(TargetCpu::Avx512vl)
        && target.supports(TargetCpu::Avx512cd)
        && !target.data_independent_timing()
}

/// Count leading zeros, including the architectural element width for zero.
pub(crate) fn leading_zeros(code: &mut Code, lane_bytes: u8, xmm: u8, target: TargetCpu) -> bool {
    if !leading_supported(lane_bytes, target) {
        return false;
    }
    code.asm().evex(
        Evex {
            map: Map::M0f38,
            pp: Pp::P66,
            w: lane_bytes == 8,
            length: 0,
            mask: 0,
            zeroing: false,
            broadcast: false,
            scale: 16,
        },
        0x44,
        xmm,
        0,
        Rm::Xmm(xmm),
    );
    true
}

/// Count set bits independently in each lane of `xmm`, without flags,
/// floating-point status or opmask state changes.
pub(crate) fn population_count(
    code: &mut Code,
    lane_bytes: u8,
    xmm: u8,
    target: TargetCpu,
) -> bool {
    if !population_supported(lane_bytes, target) {
        return false;
    }
    code.asm().evex(
        Evex {
            map: Map::M0f38,
            pp: Pp::P66,
            w: matches!(lane_bytes, 2 | 8),
            length: 0,
            mask: 0,
            zeroing: false,
            broadcast: false,
            scale: 16,
        },
        if lane_bytes <= 2 { 0x54 } else { 0x55 },
        xmm,
        0,
        Rm::Xmm(xmm),
    );
    true
}

#[cfg(all(
    test,
    target_arch = "x86_64",
    any(target_os = "macos", target_os = "linux")
))]
pub(crate) mod execution {
    use crate::{GuestState, context::ExecutionContext};
    use core::ffi::c_void;
    unsafe extern "C" {
        fn mmap(
            address: *mut c_void,
            bytes: usize,
            protection: i32,
            flags: i32,
            fd: i32,
            offset: i64,
        ) -> *mut c_void;
        fn mprotect(address: *mut c_void, bytes: usize, protection: i32) -> i32;
        fn munmap(address: *mut c_void, bytes: usize) -> i32;
    }
    pub(crate) struct Native {
        page: *mut c_void,
        bytes: usize,
    }
    impl Native {
        pub(crate) fn new(code: &[u8]) -> Self {
            #[cfg(target_os = "macos")]
            const ANONYMOUS: i32 = 0x1000;
            #[cfg(target_os = "linux")]
            const ANONYMOUS: i32 = 0x20;
            let bytes = code.len().next_multiple_of(4096);
            // SAFETY: private anonymous memory becomes RX after the copy and
            // remains mapped until this Native is dropped.
            let page = unsafe {
                let page = mmap(core::ptr::null_mut(), bytes, 3, 2 | ANONYMOUS, -1, 0);
                assert_ne!(page as usize, usize::MAX);
                core::ptr::copy_nonoverlapping(code.as_ptr(), page.cast(), code.len());
                assert_eq!(mprotect(page, bytes, 5), 0);
                page
            };
            Self { page, bytes }
        }
        pub(crate) fn run(&self, state: &mut GuestState, context: &mut ExecutionContext) {
            // SAFETY: callers supply a complete generated function using the
            // state/context ABI, with every requested host feature available.
            unsafe {
                let entry: unsafe extern "sysv64" fn(*mut GuestState, *mut ExecutionContext) =
                    core::mem::transmute(self.page);
                entry(state, context);
            }
        }
    }
    impl Drop for Native {
        fn drop(&mut self) {
            // SAFETY: this object owns the mapping returned by mmap.
            assert_eq!(unsafe { munmap(self.page, self.bytes) }, 0);
        }
    }
}

#[cfg(test)]
mod scalable_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_arithmetic_matches_gnu_encodings() {
        let target = TargetCpu::X86_64_V4.with_avx512fp16();
        assert!(half_supported(target));
        assert!(!half_supported(TargetCpu::X86_64_V4));
        assert!(!half_supported(target.with_data_independent_timing()));
        let mut code = Code::new();
        half_writemask(&mut code, 4);
        half_arithmetic(&mut code, HalfOpcode::Add, 0, 1);
        half_arithmetic(&mut code, HalfOpcode::Subtract, 9, 10);
        half_arithmetic(&mut code, HalfOpcode::Multiply, 2, 3);
        half_fused(&mut code, 0x98, 0, 6, 1);
        half_fused(&mut code, 0x9c, 9, 10, 11);
        half_from_signed(&mut code, 0, true, Gpr::Rax);
        half_to_signed(&mut code, Gpr::Rax, 0, false, true);
        half_from_double(&mut code, 0, 1);
        extrema(&mut code, Format::Single, false, false, 0, 1);
        extrema(&mut code, Format::Double, true, true, 0, 1);
        assert_eq!(
            code,
            [
                &[0xb8, 0x0f, 0, 0, 0][..],                  // mov eax,0xf
                &[0xc5, 0xf8, 0x92, 0xc8],                   // kmovw k1,eax
                &[0x62, 0xf5, 0x7c, 0x89, 0x58, 0xc1],       // vaddph xmm0{k1}{z},xmm0,xmm1
                &[0x62, 0x55, 0x34, 0x89, 0x5c, 0xca],       // vsubph xmm9{k1}{z},xmm9,xmm10
                &[0x62, 0xf5, 0x6c, 0x89, 0x59, 0xd3],       // vmulph xmm2{k1}{z},xmm2,xmm3
                &[0x62, 0xf6, 0x4d, 0x89, 0x98, 0xc1],       // vfmadd132ph xmm0{k1}{z},xmm6,xmm1
                &[0x62, 0x56, 0x2d, 0x89, 0x9c, 0xcb],       // vfnmadd132ph xmm9{k1}{z},xmm10,xmm11
                &[0x62, 0xf5, 0xfe, 0x08, 0x2a, 0xc0],       // vcvtsi2sh xmm0,xmm0,rax
                &[0x62, 0xf5, 0x7e, 0x08, 0x2c, 0xc0],       // vcvttsh2si eax,xmm0
                &[0x62, 0xf5, 0xff, 0x08, 0x5a, 0xc1],       // vcvtsd2sh xmm0,xmm0,xmm1
                &[0x62, 0xf3, 0x7d, 0x08, 0x52, 0xc1, 0x04], // vminmaxps xmm0,xmm0,xmm1,4
                &[0x62, 0xf3, 0xfd, 0x08, 0x53, 0xc1, 0x05], // vminmaxsd xmm0,xmm0,xmm1,5
            ]
            .concat()
        );
    }

    #[test]
    fn byte_blend_has_explicit_mask_and_refuses_missing_state_or_dit() {
        let target = TargetCpu::Avx2;
        for (destination, old, mask, expected) in [
            (0, 7, 6, [0xc4, 0xe3, 0x41, 0x4c, 0xc0, 0x60]),
            (8, 15, 14, [0xc4, 0x43, 0x01, 0x4c, 0xc0, 0xe0]),
        ] {
            let mut code = Code::new();
            assert!(blend_bytes(&mut code, destination, old, mask, target));
            assert_eq!(*code, expected);
            assert_eq!(code.instructions(), 1);
        }
        for target in [
            TargetCpu::Baseline,
            TargetCpu::Gfni,
            TargetCpu::Avx512f.with_avx512vl(),
            TargetCpu::Avx2.with_data_independent_timing(),
        ] {
            let mut code = Code::new();
            assert!(!blend_bytes(&mut code, 0, 7, 6, target));
            assert!(code.is_empty());
        }
    }

    #[test]
    fn merge_requires_vector_state_and_preserves_dit() {
        for target in [
            TargetCpu::Baseline,
            TargetCpu::Avx512f,
            TargetCpu::Avx512vl,
            TargetCpu::Avx512f
                .with_avx512vl()
                .with_data_independent_timing(),
        ] {
            let mut code = Code::new();
            assert!(!merge(&mut code, 0, 7, 6, target));
            assert!(code.is_empty());
        }
    }

    #[test]
    fn merge_truth_table_and_gnu_encodings() {
        let target = TargetCpu::Avx512f.with_avx512vl();
        for (destination, old, mask, expected) in [
            (0, 7, 6, [0x62, 0xf3, 0x45, 0x08, 0x25, 0xc6, 0xe4]),
            (8, 15, 14, [0x62, 0x53, 0x05, 0x08, 0x25, 0xc6, 0xe4]),
            (31, 30, 29, [0x62, 0x03, 0x0d, 0x00, 0x25, 0xfd, 0xe4]),
        ] {
            let mut code = Code::new();
            assert!(merge(&mut code, destination, old, mask, target));
            assert_eq!(*code, expected);
            assert_eq!(code.instructions(), 1);
        }
        for new in 0u8..2 {
            for old in 0u8..2 {
                for mask in 0u8..2 {
                    let index = (new << 2) | (old << 1) | mask;
                    assert_eq!((0xe4u8 >> index) & 1, (new & mask) | (old & (mask ^ 1)));
                }
            }
        }
    }

    #[test]
    fn leading_zero_count_requires_cd_and_uses_evex_128() {
        for lane in [4, 8] {
            for target in [
                TargetCpu::Avx512cd,
                TargetCpu::Avx512cd.with_avx512f(),
                TargetCpu::Avx512cd.with_avx512vl(),
                TargetCpu::Avx512f.with_avx512vl(),
                TargetCpu::Avx512f
                    .with_avx512vl()
                    .with_avx512cd()
                    .with_data_independent_timing(),
            ] {
                let mut code = Code::new();
                assert!(!leading_zeros(&mut code, lane, 0, target));
                assert!(code.is_empty());
            }
            let mut code = Code::new();
            assert!(leading_zeros(
                &mut code,
                lane,
                0,
                TargetCpu::Avx512f.with_avx512vl().with_avx512cd()
            ));
            assert_eq!(
                code.as_slice(),
                &[
                    0x62,
                    0xf2,
                    if lane == 8 { 0xfd } else { 0x7d },
                    0x08,
                    0x44,
                    0xc0
                ]
            );
            assert_eq!(code.instructions(), 1);
        }
    }

    #[test]
    fn population_count_checks_each_lane_extension_and_vector_state() {
        for lane in [1, 2, 4, 8] {
            let feature = if lane <= 2 {
                TargetCpu::Avx512bitalg
            } else {
                TargetCpu::Avx512vpopcntdq
            };
            for target in [
                TargetCpu::Baseline,
                feature,
                feature.with_avx512f(),
                feature.with_avx512vl(),
                TargetCpu::Avx512f.with_avx512vl(),
                feature
                    .with_avx512f()
                    .with_avx512vl()
                    .with_data_independent_timing(),
            ] {
                let mut code = Code::new();
                assert!(!population_count(&mut code, lane, 0, target));
                assert!(code.is_empty());
            }
            let target = feature.with_avx512f().with_avx512vl();
            let mut code = Code::new();
            assert!(population_count(&mut code, lane, 0, target));
            assert_eq!(
                code.as_slice(),
                &[
                    0x62,
                    0xf2,
                    if matches!(lane, 2 | 8) { 0xfd } else { 0x7d },
                    0x08,
                    if lane <= 2 { 0x54 } else { 0x55 },
                    0xc0
                ]
            );
            assert_eq!(code.instructions(), 1);
        }
    }

    #[test]
    fn byte_dot_checks_vector_state_and_encodes_each_signedness() {
        let vnni = TargetCpu::Avx512f.with_avx512vl().with_avx512vnni();
        for target in [
            TargetCpu::Baseline,
            TargetCpu::Avx512vnni,
            TargetCpu::Avx512f.with_avx512vnni(),
            TargetCpu::AvxVnni.with_data_independent_timing(),
            TargetCpu::AvxVnniInt8.with_data_independent_timing(),
            vnni.with_data_independent_timing(),
        ] {
            let mut code = Code::new();
            assert!(!byte_dot(
                &mut code,
                2,
                &[[0, 1]],
                [true; 2],
                [3, 4],
                target
            ));
            assert!(code.is_empty());
        }
        let encoded = |signed, target| {
            let mut code = Code::new();
            assert!(byte_dot(&mut code, 2, &[[0, 1]], signed, [3, 4], target));
            code.into_bytes()
        };
        const CLEAR_SUM: [u8; 4] = [0x66, 0x0f, 0xef, 0xd2];
        // GNU as: vpdpbuud, vpdpbsud and vpdpbssd xmm2, xmm0, xmm1;
        // vpdpbsud xmm2, xmm1, xmm0 for unsigned by signed.
        for (signed, product) in [
            ([false, false], [0xc4, 0xe2, 0x78, 0x50, 0xd1]),
            ([true, false], [0xc4, 0xe2, 0x7a, 0x50, 0xd1]),
            ([true, true], [0xc4, 0xe2, 0x7b, 0x50, 0xd1]),
            ([false, true], [0xc4, 0xe2, 0x72, 0x50, 0xd0]),
        ] {
            assert_eq!(
                encoded(signed, TargetCpu::AvxVnniInt8),
                [&CLEAR_SUM[..], &product].concat()
            );
        }
        // {vex} and {evex} vpdpbusd xmm2, xmm0, xmm1 multiply unsigned by
        // signed alone; signed by unsigned swaps the sources.
        for (target, prefix) in [
            (TargetCpu::AvxVnni, &[0xc4, 0xe2, 0x79][..]),
            (vnni, &[0x62, 0xf2, 0x7d, 0x08][..]),
        ] {
            let expected = [&CLEAR_SUM[..], prefix, &[0x50, 0xd1]].concat();
            assert_eq!(encoded([false, true], target), expected);
            let swapped = encoded([true, false], target);
            assert_eq!(swapped.len(), expected.len());
            assert_eq!(swapped[swapped.len() - 1], 0xd0);
            for signed in [[true, true], [false, false]] {
                let mut code = Code::new();
                assert!(byte_dot(&mut code, 2, &[[0, 1]], signed, [3, 4], target));
                // Bias, cleared correction and sum, the two products, the
                // flip and the subtraction.
                assert_eq!(code.instructions(), 9);
            }
        }
    }

    #[test]
    fn byte_dots_match_the_lane_model_on_each_host_form() {
        use crate::{GuestAddress, GuestState, context::ExecutionContext};
        use alloc::vec::Vec;
        use execution::Native;
        #[derive(Clone, Copy)]
        enum Form {
            Dot { index: Option<u8>, q: bool },
            Matrix,
        }
        let mut forms = Vec::new();
        for q in [false, true] {
            for (base, signed) in [
                (0x0e809400, [true, true]),
                (0x2e809400, [false, false]),
                (0x0e809c00, [false, true]),
            ] {
                forms.push((base, signed, Form::Dot { index: None, q }));
            }
            for (base, signed) in [
                (0x0f80e000, [true, true]),
                (0x2f80e000, [false, false]),
                (0x0f80f000, [false, true]),
                (0x0f00f000, [true, false]),
            ] {
                for index in 0..4 {
                    forms.push((
                        base,
                        signed,
                        Form::Dot {
                            index: Some(index),
                            q,
                        },
                    ));
                }
            }
        }
        for (base, signed) in [
            (0x4e80a400, [true, true]),
            (0x6e80a400, [false, false]),
            (0x4e80ac00, [false, true]),
        ] {
            forms.push((base, signed, Form::Matrix));
        }
        let host = crate::x86_64::modeled::host();
        let mut extended = 0;
        let mut random = 0x9e3779b97f4a7c15_u64;
        for target in [
            TargetCpu::X86_64_V3,
            TargetCpu::X86_64_V3.with_avx_vnni(),
            TargetCpu::X86_64_V3.with_avx_vnni_int8(),
            TargetCpu::Avx512f.with_avx512vl().with_avx512vnni(),
        ] {
            if !host.supports(target) {
                continue;
            }
            extended += usize::from(byte_dot_supported(target));
            for &(base, signed, form) in &forms {
                for (destination, left, right) in [(3u32, 1u32, 2u32), (4, 4, 4), (31, 17, 17)] {
                    let mut word = base | right << 16 | left << 5 | destination;
                    if let Form::Dot { index, q } = form {
                        word |= u32::from(q) << 30;
                        if let Some(index) = index {
                            word |= u32::from(index & 1) << 21 | u32::from(index >> 1) << 11;
                        }
                    }
                    let source: Vec<u8> = [word, 0xd65f03c0]
                        .into_iter()
                        .flat_map(u32::to_le_bytes)
                        .collect();
                    let block =
                        crate::testing::leaf_for(&source, GuestAddress(0x1000), target).unwrap();
                    let native = Native::new(&block.x86_64_code);
                    for sample in 0..16 {
                        let mut state = GuestState::default();
                        for vector in &mut state.v {
                            for limb in vector {
                                random ^= random << 13;
                                random ^= random >> 7;
                                random ^= random << 17;
                                *limb = match sample {
                                    0 => 0x8080_8080_8080_8080,
                                    1 => u64::MAX,
                                    2 => 0x7f80_7f80_ff01_8081,
                                    _ => random,
                                };
                            }
                        }
                        let value = |register: u32, at: usize, signed: bool| {
                            let vector = state.v[register as usize];
                            let byte = (vector[at / 8] >> (at % 8 * 8)) as u8;
                            if signed {
                                i64::from(byte as i8)
                            } else {
                                i64::from(byte)
                            }
                        };
                        let old = state.v[destination as usize];
                        let mut lanes: [u32; 4] =
                            core::array::from_fn(|i| (old[i / 2] >> (i % 2 * 32)) as u32);
                        let products = |left_at: usize, right_at: usize, count: usize| {
                            (0..count)
                                .map(|k| {
                                    value(left, left_at + k, signed[0])
                                        * value(right, right_at + k, signed[1])
                                })
                                .sum::<i64>() as u32
                        };
                        match form {
                            Form::Dot { index, q } => {
                                for (lane, sum) in lanes.iter_mut().enumerate() {
                                    *sum = if lane < 2 || q {
                                        let group = index.map_or(lane, usize::from);
                                        sum.wrapping_add(products(4 * lane, 4 * group, 4))
                                    } else {
                                        0
                                    };
                                }
                            }
                            Form::Matrix => {
                                for (lane, sum) in lanes.iter_mut().enumerate() {
                                    let row = lane / 2;
                                    let column = lane % 2;
                                    *sum = sum.wrapping_add(products(8 * row, 8 * column, 8));
                                }
                            }
                        }
                        let mut expected = state.clone();
                        expected.v[destination as usize] = [
                            u64::from(lanes[0]) | u64::from(lanes[1]) << 32,
                            u64::from(lanes[2]) | u64::from(lanes[3]) << 32,
                        ];
                        native.run(&mut state, &mut ExecutionContext::default());
                        assert_eq!(state, expected, "{target:?} {word:#010x} sample={sample}");
                    }
                }
            }
        }
        std::println!("BYTE_DOT_EXTENDED_TARGETS={extended}");
    }

    #[test]
    fn integer_conversions_check_each_extension_and_match_gnu_encodings() {
        let quadword = TargetCpu::Avx512f.with_avx512vl().with_avx512dq();
        let integer = |width, signed| IntegerFormat { width, signed };
        let [w, x] = [IntegerWidth::Bits32, IntegerWidth::Bits64];
        for (format, integer, target) in [
            (Format::Single, integer(w, true), quadword),
            (Format::Double, integer(w, true), quadword),
            (Format::Half, integer(IntegerWidth::Bits16, false), quadword),
            (Format::Single, integer(w, false), TargetCpu::Avx512f),
            (
                Format::Single,
                integer(x, true),
                TargetCpu::Avx512f.with_avx512vl(),
            ),
            (
                Format::Double,
                integer(x, false),
                quadword.with_data_independent_timing(),
            ),
        ] {
            let mut code = Code::new();
            assert!(!packed_from_integer(&mut code, 0, format, integer, target));
            assert!(!packed_to_integer(
                &mut code, 0, format, integer, true, target
            ));
            assert!(code.is_empty());
        }
        let mut code = Code::new();
        assert!(!scalar_from_unsigned(
            &mut code,
            0,
            Format::Double,
            true,
            Gpr::Rax,
            TargetCpu::Avx512f.with_data_independent_timing()
        ));
        assert!(code.is_empty());
        let packed = |emit: &dyn Fn(&mut Code) -> bool| {
            let mut code = Code::new();
            assert!(emit(&mut code));
            assert_eq!(code.instructions(), 1);
            code.into_bytes()
        };
        // GNU as, on XMM0 and RAX/EAX.
        for (format, integer, from, rounded, truncated) in [
            (Format::Single, integer(w, false), 0x7f7a, 0x7c79, 0x7c78),
            (Format::Single, integer(x, true), 0xfc5b, 0x7d7b, 0x7d7a),
            (Format::Single, integer(x, false), 0xff7a, 0x7d79, 0x7d78),
            (Format::Double, integer(w, false), 0x7e7a, 0xfc79, 0xfc78),
            (Format::Double, integer(x, true), 0xfee6, 0xfd7b, 0xfd7a),
            (Format::Double, integer(x, false), 0xfe7a, 0xfd79, 0xfd78),
        ] {
            let encoding = |fields: u16| {
                let [pp, opcode] = fields.to_be_bytes();
                [0x62, 0xf1, pp, 0x08, opcode, 0xc0]
            };
            assert_eq!(
                packed(&|code| packed_from_integer(code, 0, format, integer, quadword)),
                encoding(from)
            );
            for (truncate, expected) in [(false, rounded), (true, truncated)] {
                assert_eq!(
                    packed(&|code| packed_to_integer(code, 0, format, integer, truncate, quadword)),
                    encoding(expected)
                );
            }
        }
        let target = TargetCpu::Avx512f;
        assert_eq!(
            packed(&|code| scalar_from_unsigned(code, 0, Format::Double, true, Gpr::Rax, target)),
            [0x62, 0xf1, 0xff, 0x08, 0x7b, 0xc0]
        );
        assert_eq!(
            packed(&|code| scalar_from_unsigned(code, 0, Format::Single, false, Gpr::Rax, target)),
            [0x62, 0xf1, 0x7e, 0x08, 0x7b, 0xc0]
        );
        assert_eq!(
            packed(&|code| scalar_to_unsigned(
                code,
                Gpr::Rax,
                0,
                Format::Double,
                true,
                false,
                target
            )),
            [0x62, 0xf1, 0xff, 0x08, 0x79, 0xc0]
        );
        assert_eq!(
            packed(&|code| scalar_to_unsigned(
                code,
                Gpr::Rax,
                0,
                Format::Single,
                false,
                true,
                target
            )),
            [0x62, 0xf1, 0x7e, 0x08, 0x78, 0xc0]
        );
    }

    #[test]
    fn quadword_extrema_and_multiplies_match_gnu_encodings() {
        let target = TargetCpu::Avx512f.with_avx512vl();
        let mut code = Code::new();
        assert!(!quadword_absolute(&mut code, 0, TargetCpu::Avx512f));
        assert!(!quadword_extremum(
            &mut code,
            0,
            1,
            true,
            true,
            target.with_data_independent_timing()
        ));
        assert!(code.is_empty());
        assert!(!quadword_multiply(&mut code, 2, 1, target));
        // GNU as: VPABSQ xmm2, xmm2; VPMINSQ, VPMINUQ, VPMAXSQ and VPMAXUQ
        // xmm0, xmm0, xmm1; VPMULLQ xmm2, xmm2, xmm1.
        assert!(quadword_absolute(&mut code, 2, target));
        for (signed, maximum) in [(true, false), (false, false), (true, true), (false, true)] {
            assert!(quadword_extremum(&mut code, 0, 1, signed, maximum, target));
        }
        assert!(quadword_multiply(&mut code, 2, 1, target.with_avx512dq()));
        assert_eq!(
            code.into_bytes(),
            [
                0x62, 0xf2, 0xfd, 0x08, 0x1f, 0xd2, 0x62, 0xf2, 0xfd, 0x08, 0x39, 0xc1, 0x62, 0xf2,
                0xfd, 0x08, 0x3b, 0xc1, 0x62, 0xf2, 0xfd, 0x08, 0x3d, 0xc1, 0x62, 0xf2, 0xfd, 0x08,
                0x3f, 0xc1, 0x62, 0xf2, 0xed, 0x08, 0x40, 0xd1,
            ]
        );
    }

    #[test]
    fn lane_compares_check_each_extension_and_match_gnu_encodings() {
        let mut code = Code::new();
        for (lane_bytes, target) in [
            (1, TargetCpu::Avx512f.with_avx512vl().with_avx512dq()),
            (8, TargetCpu::Avx512f.with_avx512vl().with_avx512bw()),
            (4, TargetCpu::X86_64_V4.with_data_independent_timing()),
            (16, TargetCpu::X86_64_V4),
        ] {
            assert!(!compare(
                &mut code,
                lane_bytes,
                true,
                Predicate::Equal,
                0,
                1,
                target
            ));
            assert!(!test(&mut code, lane_bytes, 0, 1, target));
        }
        assert!(code.is_empty());
        // GNU as: VPCMPNLTUB, VPCMPNLEQ, VPCMPLTUD and VPCMPLEW into K1 of
        // xmm0 against xmm1 and VPTESTMW of xmm2 and xmm3, each followed by
        // VPMOVM2 back into its left register.
        let target = TargetCpu::X86_64_V4;
        for (lane_bytes, signed, predicate) in [
            (1, false, Predicate::GreaterEqual),
            (8, true, Predicate::Greater),
            (4, false, Predicate::Less),
            (2, true, Predicate::LessEqual),
        ] {
            assert!(compare(
                &mut code, lane_bytes, signed, predicate, 0, 1, target
            ));
        }
        assert!(test(&mut code, 2, 2, 3, target));
        assert_eq!(
            code.into_bytes(),
            [
                0x62, 0xf3, 0x7d, 0x08, 0x3e, 0xc9, 0x05, 0x62, 0xf2, 0x7e, 0x08, 0x28, 0xc1, 0x62,
                0xf3, 0xfd, 0x08, 0x1f, 0xc9, 0x06, 0x62, 0xf2, 0xfe, 0x08, 0x38, 0xc1, 0x62, 0xf3,
                0x7d, 0x08, 0x1e, 0xc9, 0x01, 0x62, 0xf2, 0x7e, 0x08, 0x38, 0xc1, 0x62, 0xf3, 0xfd,
                0x08, 0x3f, 0xc9, 0x02, 0x62, 0xf2, 0xfe, 0x08, 0x28, 0xc1, 0x62, 0xf2, 0xed, 0x08,
                0x26, 0xcb, 0x62, 0xf2, 0xfe, 0x08, 0x28, 0xd1,
            ]
        );
    }

    #[test]
    fn lane_compares_match_the_sse_recipes() {
        use crate::{GuestAddress, GuestState, context::ExecutionContext};
        use alloc::vec::Vec;
        use execution::Native;
        if !TargetCpu::detect().supports(TargetCpu::X86_64_V4) {
            return;
        }
        let mut random = 0x2545_f491_4f6c_dd1d_u64;
        // CMHI, CMHS and CMTST on vector and scalar lanes, and UABD, UQXTN
        // and UQSHRN, which compare unsigned lanes inside.
        for word in [
            0x6e22_3423u32,
            0x6e62_3c23,
            0x6ee2_3423,
            0x6ee2_3c23,
            0x0e62_8c23,
            0x4ee2_8c23,
            0x7ee2_3423,
            0x7ee2_3c23,
            0x5ee2_8c23,
            0x6ea2_7423,
            0x2e21_4823,
            0x2f3d_9423,
        ] {
            let source: Vec<u8> = [word, 0xd65f03c0]
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect();
            let compile = |target| {
                crate::testing::leaf_for(&source, GuestAddress(0x1000), target)
                    .unwrap()
                    .x86_64_code
            };
            let baseline = compile(TargetCpu::X86_64_V3);
            let extended = compile(TargetCpu::X86_64_V4);
            assert_ne!(baseline, extended, "{word:#010x} uses an AVX-512 compare");
            let [baseline, extended] = [&baseline, &extended].map(|code| Native::new(code));
            for sample in 0..48 {
                let mut state = GuestState::default();
                for vector in &mut state.v {
                    for limb in vector {
                        random ^= random << 13;
                        random ^= random >> 7;
                        random ^= random << 17;
                        // Small lanes and equal operands in two thirds.
                        *limb = if sample % 3 == 0 {
                            random
                        } else {
                            random & 0x8181_8181_8181_8181
                        };
                    }
                }
                if sample % 2 == 0 {
                    state.v[2] = state.v[1];
                }
                let mut expected = state.clone();
                baseline.run(&mut expected, &mut ExecutionContext::default());
                extended.run(&mut state, &mut ExecutionContext::default());
                assert_eq!(state, expected, "{word:#010x} sample={sample}");
            }
        }
    }

    #[test]
    fn saturating_narrows_check_each_extension_and_match_gnu_encodings() {
        let mut code = Code::new();
        for (lane_bytes, target) in [
            (2, TargetCpu::Avx512f.with_avx512vl().with_avx512dq()),
            (8, TargetCpu::X86_64_V4.with_data_independent_timing()),
            (1, TargetCpu::X86_64_V4),
        ] {
            assert!(!saturating_narrow(
                &mut code,
                lane_bytes,
                [true, true],
                1,
                0,
                Some(4),
                target
            ));
        }
        assert!(code.is_empty());
        // GNU as: VPMOVSWB xmm1, xmm0; VPMOVUSDW xmm2, xmm0; and VPMOVSQD
        // xmm1, xmm0, PMOVSXDQ xmm4, xmm1, VPCMPNEQQ k1, xmm4, xmm0 and
        // VPMOVM2Q xmm4, k1.
        let target = TargetCpu::X86_64_V4;
        for (lane_bytes, signed, destination, saturated) in [
            (2, [true, true], 1, None),
            (4, [false, false], 2, None),
            (8, [true, true], 1, Some(4)),
        ] {
            assert!(saturating_narrow(
                &mut code,
                lane_bytes,
                signed,
                destination,
                0,
                saturated,
                target
            ));
        }
        assert_eq!(
            code.into_bytes(),
            [
                0x62, 0xf2, 0x7e, 0x08, 0x20, 0xc1, 0x62, 0xf2, 0x7e, 0x08, 0x13, 0xc2, 0x62, 0xf2,
                0x7e, 0x08, 0x25, 0xc1, 0x66, 0x0f, 0x38, 0x25, 0xe1, 0x62, 0xf3, 0xdd, 0x08, 0x1f,
                0xc8, 0x04, 0x62, 0xf2, 0xfe, 0x08, 0x38, 0xe1,
            ]
        );
    }

    #[test]
    fn saturating_narrows_match_the_sse_recipes() {
        use crate::{GuestAddress, GuestState, context::ExecutionContext};
        use alloc::vec::Vec;
        use execution::Native;
        if !TargetCpu::detect().supports(TargetCpu::X86_64_V4) {
            return;
        }
        // Lanes at and beyond every B/H/S bound, of each signedness.
        let bounds: [u64; 12] = [
            0x00ff_00ff_00ff_00ff,
            0x0100_0100_0100_0100,
            0x007f_0080_ff80_ff7f,
            0x0000_ffff_0001_0000,
            0x0000_7fff_ffff_8000,
            0xffff_7fff_0000_8000,
            0x0000_0000_ffff_ffff,
            0x0000_0001_0000_0000,
            0x0000_0000_7fff_ffff,
            0xffff_ffff_8000_0000,
            0xffff_ffff_7fff_ffff,
            0x8000_0000_0000_0000,
        ];
        let mut random = 0x9e37_79b9_7f4a_7c15_u64;
        // SQXTN, UQXTN and SQXTUN on each lane size, their second-part and
        // scalar forms, and SQSHRN, UQSHRN, SQRSHRUN, SQRSHRN2, UQRSHRN2 and
        // scalar SQSHRUN.
        for word in [
            0x0e21_4823u32,
            0x2e61_4823,
            0x2ea1_2823,
            0x4ea1_4823,
            0x6e21_4823,
            0x6e61_2823,
            0x5e21_4823,
            0x7ea1_4823,
            0x7e61_2823,
            0x0f0d_9423,
            0x2f11_9423,
            0x2f39_8c23,
            0x4f20_9c23,
            0x6f0f_9c23,
            0x7f3b_8423,
        ] {
            let source: Vec<u8> = [word, 0xd65f03c0]
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect();
            let compile = |target| {
                crate::testing::leaf_for(&source, GuestAddress(0x1000), target)
                    .unwrap()
                    .x86_64_code
            };
            let baseline = compile(TargetCpu::X86_64_V3);
            let extended = compile(TargetCpu::X86_64_V4);
            // Vector shift narrows of signed H and S lanes use one host pack
            // and a VPTEST for QC on both tiers.
            const VPTEST: [u8; 4] = [0xc4, 0xe2, 0x79, 0x17];
            let packed = extended.windows(4).any(|bytes| bytes == VPTEST);
            assert!(
                baseline != extended || packed,
                "{word:#010x} uses VPMOVS or VPMOVUS"
            );
            let [baseline, extended] = [&baseline, &extended].map(|code| Native::new(code));
            for sample in 0..48 {
                let mut state = GuestState::default();
                for vector in &mut state.v {
                    for limb in vector {
                        random ^= random << 13;
                        random ^= random >> 7;
                        random ^= random << 17;
                        *limb = if sample % 2 == 0 {
                            bounds[(random % bounds.len() as u64) as usize]
                        } else {
                            random
                        };
                    }
                }
                let mut expected = state.clone();
                baseline.run(&mut expected, &mut ExecutionContext::default());
                extended.run(&mut state, &mut ExecutionContext::default());
                assert_eq!(state, expected, "{word:#010x} sample={sample}");
            }
        }
    }

    #[test]
    fn quadword_absolute_values_match_the_sse2_recipe() {
        use crate::{GuestAddress, GuestState, context::ExecutionContext};
        use alloc::vec::Vec;
        use execution::Native;
        let candidate = TargetCpu::X86_64_V3.with_avx512f().with_avx512vl();
        if !TargetCpu::detect().supports(candidate) {
            return;
        }
        let mut random = 0x9e37_79b9_7f4a_7c15_u64;
        // ABS V3.2D, V1.2D and ABS D3, D1.
        for word in [0x4ee0_b823u32, 0x5ee0_b823] {
            let source: Vec<u8> = [word, 0xd65f03c0]
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect();
            let compile = |target| {
                crate::testing::leaf_for(&source, GuestAddress(0x1000), target)
                    .unwrap()
                    .x86_64_code
            };
            let baseline = compile(TargetCpu::X86_64_V3);
            let extended = compile(candidate);
            assert_ne!(baseline, extended, "{word:#010x} uses VPABSQ");
            let [baseline, extended] = [&baseline, &extended].map(|code| Native::new(code));
            for sample in 0..32 {
                let mut state = GuestState::default();
                for vector in &mut state.v {
                    for limb in vector {
                        random ^= random << 13;
                        random ^= random >> 7;
                        random ^= random << 17;
                        *limb = random;
                    }
                }
                state.v[1][sample % 2] = [0, i64::MIN as u64, u64::MAX][sample % 3];
                let mut expected = state.clone();
                baseline.run(&mut expected, &mut ExecutionContext::default());
                extended.run(&mut state, &mut ExecutionContext::default());
                assert_eq!(state, expected, "{word:#010x} sample={sample}");
            }
        }
    }

    #[test]
    fn integer_conversions_match_the_baseline_recipes() {
        use crate::{GuestAddress, GuestState, context::ExecutionContext};
        use alloc::vec::Vec;
        use execution::Native;
        let candidate = TargetCpu::X86_64_V4;
        if !TargetCpu::detect().supports(candidate) {
            return;
        }
        let mut words = Vec::new();
        // SCVTF/UCVTF and FCVT{N,M,P,Z}{S,U} on vectors, with and without
        // fixed-point bits (Rd = 3, Rn = 1).
        for q in [0u32, 1] {
            for (base, double) in [
                (0x2e21d800u32, false),
                (0x2e61d800, true),
                (0x0e61d800, true),
                (0x2e21a800, false),
                (0x2e21b800, false),
                (0x2ea1a800, false),
                (0x2ea1b800, false),
                (0x0e61a800, true),
                (0x0ee1b800, true),
                (0x2e61a800, true),
                (0x2e61b800, true),
                (0x2ee1a800, true),
                (0x2ee1b800, true),
                (0x2f3de400, false),
                (0x2f3dfc00, false),
                (0x0f7be400, true),
                (0x2f7bfc00, true),
            ] {
                if !double || q == 1 {
                    words.push(base | q << 30 | 1 << 5 | 3);
                }
            }
        }
        // UCVTF from X (a W source zero-extends into CVTSI2SS/SD),
        // FCVT{N,P,M,Z}U to X and W, and the SIMD scalar UCVTF/FCVTZU on D.
        for word in [
            0x9e630000u32,
            0x9e230000,
            0x9e610000,
            0x9e690000,
            0x9e710000,
            0x9e790000,
            0x1e390000,
            0x9e390000,
            0x1e790000,
            0x7ee1b800,
            0x7e61d800,
        ] {
            words.push(word | 1 << 5 | 3);
        }
        let inputs: [u64; 12] = [
            0,
            0x8000_0000_0000_0000,
            u64::MAX,
            0x7fff_ffff_ffff_ffff,
            0x3fe0_0000_0000_0000, // 0.5
            0xbfe0_0000_0000_0000, // -0.5
            0x41f0_0000_0000_0000, // 2^32
            0x43f0_0000_0000_0000, // 2^64
            0x43ef_ffff_ffff_ffff, // just below 2^64
            0x4f80_0000_bf00_0000, // 2^32 and -0.5 as singles
            0x4f7f_ffff_3fc0_0000, // just below 2^32 and 1.5
            0x0000_0001_cf00_0000, // a subnormal and -2^31
        ];
        let mut random = 0x2545f4914f6cdd1d_u64;
        for word in words {
            let source: Vec<u8> = [word, 0xd65f03c0]
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect();
            // Ordinary blocks may exit to the numerical runtime.
            let compile = |target| {
                crate::testing::block_for(
                    &source,
                    GuestAddress(0x1000),
                    crate::block::BlockProfile::ORDINARY,
                    target,
                )
                .unwrap()
                .native
                .x86_64_code
            };
            let baseline = compile(TargetCpu::X86_64_V3);
            let extended = compile(candidate);
            assert_ne!(baseline, extended, "{word:#010x} uses an AVX-512 form");
            let [baseline, extended] = [&baseline, &extended].map(|code| Native::new(code));
            for sample in 0..64 {
                let mut state = GuestState::default();
                for vector in &mut state.v {
                    for limb in vector {
                        random ^= random << 13;
                        random ^= random >> 7;
                        random ^= random << 17;
                        *limb = if sample < 48 {
                            inputs[(random % inputs.len() as u64) as usize]
                        } else {
                            random
                        };
                    }
                }
                state.x[1] = state.v[1][0];
                state.fpcr = (sample as u32 % 4) << 22;
                let mut expected = state.clone();
                baseline.run(&mut expected, &mut ExecutionContext::default());
                extended.run(&mut state, &mut ExecutionContext::default());
                assert_eq!(state, expected, "{word:#010x} sample={sample}");
            }
        }
    }

    #[test]
    fn byte_tables_check_vbmi_state_and_match_gnu_encodings() {
        let vbmi = TargetCpu::Avx512f
            .with_avx512vl()
            .with_avx512bw()
            .with_avx512vbmi();
        let memory = Mem::base_disp32(Gpr::Rdi, 0x128);
        for target in [
            TargetCpu::Avx512f.with_avx512vl().with_avx512bw(),
            TargetCpu::Avx512f.with_avx512vl().with_avx512vbmi(),
            vbmi.with_data_independent_timing(),
        ] {
            let mut code = Code::new();
            assert!(!table_supported(1, target));
            assert!(!table(&mut code, 1, 0, 1, &[memory], false, [2, 3], target));
            assert!(code.is_empty());
        }
        let halfwords = TargetCpu::Avx512f.with_avx512vl().with_avx512bw();
        for (lane_bytes, target) in [
            (2, TargetCpu::Avx512f.with_avx512vl()),
            (4, TargetCpu::Avx512f),
            (8, halfwords.with_data_independent_timing()),
            (16, vbmi),
        ] {
            assert!(!table_supported(lane_bytes, target));
        }
        // GNU as: VPERMW on one halfword table, VPERMT2D pairing one dword
        // table with itself, and TBX over four quadword tables at
        // [rdi+0x100..0x130]: VPCMPLTUQ, a VPERMT2Q pair per half, VPTESTMQ
        // of index bit 2 and two VMOVDQU64 blends.
        let emit = |lane_bytes, tables: &[Mem], preserve| {
            let mut code = Code::new();
            assert!(table(
                &mut code,
                lane_bytes,
                0,
                1,
                tables,
                preserve,
                [2, 3],
                TargetCpu::X86_64_V4
            ));
            code.into_bytes()
        };
        assert_eq!(
            emit(2, &[memory], false),
            [
                0xb8, 0x08, 0x00, 0x00, 0x00, 0x62, 0xf2, 0x7d, 0x08, 0x7b, 0xd8, 0x62, 0xf3, 0xf5,
                0x08, 0x3e, 0xcb, 0x01, 0x62, 0xf2, 0xf5, 0x89, 0x8d, 0x87, 0x28, 0x01, 0x00, 0x00,
            ]
        );
        assert_eq!(
            emit(4, &[memory], false),
            [
                0xb8, 0x04, 0x00, 0x00, 0x00, 0x62, 0xf2, 0x7d, 0x08, 0x7c, 0xd8, 0x62, 0xf3, 0x75,
                0x08, 0x1e, 0xcb, 0x01, 0xf3, 0x0f, 0x6f, 0x87, 0x28, 0x01, 0x00, 0x00, 0x62, 0xf2,
                0x75, 0x89, 0x7e, 0x87, 0x28, 0x01, 0x00, 0x00,
            ]
        );
        let tables = [0x100, 0x110, 0x120, 0x130].map(|at| Mem::base_disp32(Gpr::Rdi, at));
        assert_eq!(
            emit(8, &tables, true),
            [
                0xb8, 0x08, 0x00, 0x00, 0x00, 0x62, 0xf2, 0xfd, 0x08, 0x7c, 0xd8, 0x62, 0xf3, 0xf5,
                0x08, 0x1e, 0xcb, 0x01, 0xf3, 0x0f, 0x6f, 0x97, 0x00, 0x01, 0x00, 0x00, 0x62, 0xf2,
                0xf5, 0x89, 0x7e, 0x97, 0x10, 0x01, 0x00, 0x00, 0xb8, 0x04, 0x00, 0x00, 0x00, 0x62,
                0xf2, 0xfd, 0x08, 0x7c, 0xd8, 0x62, 0xf2, 0xf5, 0x08, 0x27, 0xd3, 0xf3, 0x0f, 0x6f,
                0x9f, 0x20, 0x01, 0x00, 0x00, 0x62, 0xf2, 0xf5, 0x89, 0x7e, 0x9f, 0x30, 0x01, 0x00,
                0x00, 0x62, 0xf1, 0xfe, 0x0a, 0x6f, 0xd3, 0x62, 0xf1, 0xfe, 0x09, 0x6f, 0xc2,
            ]
        );
        // GNU as: mov eax, 16; vpbroadcastb xmm3, eax; vpcmpltub k1, xmm1,
        // xmm3; vpermb xmm0{k1}{z} (TBL) or xmm0{k1} (TBX), xmm1, [rdi+0x128].
        for (preserve, masking) in [(false, 0x89), (true, 0x09)] {
            let mut code = Code::new();
            assert!(table(&mut code, 1, 0, 1, &[memory], preserve, [2, 3], vbmi));
            assert_eq!(code.instructions(), 4);
            let bytes = code.into_bytes();
            assert_eq!(
                bytes[5..],
                [
                    0x62, 0xf2, 0x7d, 0x08, 0x7a, 0xd8, 0x62, 0xf3, 0x75, 0x08, 0x3e, 0xcb, 0x01,
                    0x62, 0xf2, 0x75, masking, 0x8d, 0x87, 0x28, 0x01, 0x00, 0x00,
                ]
            );
        }
    }

    #[test]
    fn byte_tables_match_the_lane_model() {
        use crate::{GuestAddress, GuestState, context::ExecutionContext};
        use alloc::vec::Vec;
        use execution::Native;
        let host = TargetCpu::detect();
        let mut random = 0x853c_49e6_748f_ea9b_u64;
        let mut next = move || {
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            random
        };
        let mut extended = 0;
        for target in [TargetCpu::X86_64_V3, TargetCpu::X86_64_V4.with_avx512vbmi()] {
            if !host.supports(target) {
                continue;
            }
            extended += usize::from(table_supported(1, target));
            for registers in 1..=4u32 {
                for (destination, first, indices) in [(3u32, 4u32, 2u32), (2, 30, 2), (5, 4, 5)] {
                    for (q, tbx) in [(0u32, false), (1, false), (0, true), (1, true)] {
                        let word = 0x0e00_0000
                            | q << 30
                            | indices << 16
                            | (registers - 1) << 13
                            | u32::from(tbx) << 12
                            | first << 5
                            | destination;
                        let source: Vec<u8> = [word, 0xd65f03c0]
                            .into_iter()
                            .flat_map(u32::to_le_bytes)
                            .collect();
                        let block = crate::testing::leaf_for(&source, GuestAddress(0x1000), target)
                            .unwrap();
                        let native = Native::new(&block.x86_64_code);
                        for sample in 0..32 {
                            let mut state = GuestState::default();
                            for vector in &mut state.v {
                                *vector = [next(), next()];
                            }
                            if sample % 2 == 0 {
                                // Mostly in-range indices, with the bound itself.
                                let bound = 16 * registers as u64 + 1;
                                let bytes: Vec<u8> =
                                    (0..16).map(|_| (next() % bound) as u8).collect();
                                state.v[indices as usize] = [
                                    u64::from_le_bytes(bytes[..8].try_into().unwrap()),
                                    u64::from_le_bytes(bytes[8..].try_into().unwrap()),
                                ];
                            }
                            let byte = |vector: [u64; 2], at: usize| {
                                (vector[at / 8] >> (at % 8 * 8)) as u8
                            };
                            let old = state.v[destination as usize];
                            let lanes: [u8; 16] = core::array::from_fn(|lane| {
                                let index = usize::from(byte(state.v[indices as usize], lane));
                                if lane >= 8 && q == 0 {
                                    0
                                } else if index < 16 * registers as usize {
                                    let table = (first as usize + index / 16) % 32;
                                    byte(state.v[table], index % 16)
                                } else if tbx {
                                    byte(old, lane)
                                } else {
                                    0
                                }
                            });
                            let mut expected = state.clone();
                            expected.v[destination as usize] = [
                                u64::from_le_bytes(lanes[..8].try_into().unwrap()),
                                u64::from_le_bytes(lanes[8..].try_into().unwrap()),
                            ];
                            native.run(&mut state, &mut ExecutionContext::default());
                            assert_eq!(state, expected, "{target:?} {word:#010x} sample={sample}");
                        }
                    }
                }
            }
        }
        std::println!("BYTE_TABLE_EXTENDED_TARGETS={extended}");
    }

    #[test]
    fn byte_reversal_keeps_baseline_and_dit_paths_unchanged() {
        for target in [
            TargetCpu::Baseline,
            TargetCpu::Gfni.with_data_independent_timing(),
        ] {
            let mut code = Code::new();
            assert!(!reverse_bytes(&mut code, 0, 2, target));
            assert!(code.is_empty());
        }
        let mut code = Code::new();
        assert!(reverse_bytes(&mut code, 0, 2, TargetCpu::Gfni));
        assert_eq!(code.instructions(), 4);
        assert_eq!(&code[code.len() - 6..], &[0x66, 0x0f, 0x3a, 0xce, 0xc2, 0]);
    }
}
