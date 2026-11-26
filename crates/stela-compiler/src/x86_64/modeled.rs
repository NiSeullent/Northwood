//! Test-only models of host instructions this machine lacks.
//!
//! Native equivalence tests run emitted code on the host. For the features
//! [`host`] adds, a SIGILL handler decodes the faulting instruction, applies
//! its model (Intel SDM Vol. 2) to the interrupted registers in the Linux
//! signal frame and resumes after it. `models_match_the_host_instructions`
//! checks each model against the instruction wherever the host has it,
//! which Intel SDE provides for each Intel extension.
use crate::TargetCpu;
use core::ffi::c_void;
use std::sync::{Once, OnceLock, atomic::AtomicUsize, atomic::Ordering};

/// `ucontext_t.uc_mcontext.gregs`, after `uc_flags`, `uc_link` and the
/// 24-byte `stack_t`.
const GREGS: usize = 40;
/// `REG_RIP` in `gregs`.
const RIP: usize = 16;
/// `uc_mcontext.fpregs`, after the 23 general registers.
const FPREGS: usize = GREGS + 23 * 8;
/// XMM0 in the FXSAVE image `fpregs` points to.
const XMM: usize = 160;
/// `_fpx_sw_bytes.magic1`: the kernel saved an XSAVE image.
const MAGIC1: usize = 464;
const FP_XSTATE_MAGIC1: u32 = 0x4650_5853;
/// `_fpx_sw_bytes.xfeatures`: the state components the image holds.
const XFEATURES: usize = 472;
/// XSAVE header XSTATE_BV: a clear bit means the component is in its initial
/// (zero) state and its bytes are stale.
const XSTATE_BV: usize = 512;
/// MXCSR in the FXSAVE image.
const MXCSR: usize = 24;
/// XSAVE components: the upper halves of YMM0..15, the opmask registers, the
/// upper halves of ZMM0..15, and ZMM16..31.
const AVX: u32 = 2;
const OPMASK: u32 = 5;
const ZMM_HI256: u32 = 6;
const HI16_ZMM: u32 = 7;
const SIGILL: i32 = 4;
const SA_SIGINFO: i32 = 4;
const SIG_DFL: usize = 0;

/// The host's features plus the modeled ones when the host can hold their
/// state (AVX) under Linux's signal frame, with the handler installed.
pub fn host() -> TargetCpu {
    let host = TargetCpu::detect();
    if !cfg!(target_os = "linux") || !host.supports(TargetCpu::Avx2) {
        return host;
    }
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        layout();
        let action = SigAction {
            handler: trap as unsafe extern "C" fn(i32, *mut c_void, *mut c_void) as usize,
            mask: [0; 16],
            flags: SA_SIGINFO,
            restorer: 0,
        };
        let mut prior = SigAction {
            handler: SIG_DFL,
            mask: [0; 16],
            flags: 0,
            restorer: 0,
        };
        // SAFETY: a valid action for SIGILL; the handler is async-signal-safe.
        let installed = unsafe { sigaction(SIGILL, &action, &mut prior) };
        assert_eq!(installed, 0);
        if prior.flags & SA_SIGINFO != 0 {
            PRIOR.store(prior.handler, Ordering::Relaxed);
        }
    });
    let host = host
        .with_avx_vnni_int8()
        .with_avx_vnni_int16()
        .with_sha512();
    // EVEX forms need the host's opmask and ZMM state.
    if host.supports(TargetCpu::Avx512f) {
        host.with_avx512fp16().with_avx10_2()
    } else {
        host
    }
}

/// The SA_SIGINFO handler SIGILL had, zero for none.
static PRIOR: AtomicUsize = AtomicUsize::new(0);

#[repr(C)]
struct SigAction {
    handler: usize,
    mask: [u64; 16],
    flags: i32,
    restorer: usize,
}

unsafe extern "C" {
    fn sigaction(signal: i32, action: *const SigAction, old: *mut SigAction) -> i32;
    fn write(fd: i32, bytes: *const c_void, count: usize) -> isize;
}

/// One XSAVE component's place in the standard-format image (CPUID 0DH).
#[derive(Clone, Copy)]
struct Component {
    bit: u32,
    offset: usize,
    bytes: usize,
}

struct Layout {
    avx: Component,
    opmask: Component,
    zmm_hi256: Component,
    hi16_zmm: Component,
}

fn layout() -> &'static Layout {
    static LAYOUT: OnceLock<Layout> = OnceLock::new();
    LAYOUT.get_or_init(|| {
        let component = |bit| {
            let leaf = core::arch::x86_64::__cpuid_count(0xd, bit);
            Component {
                bit,
                offset: leaf.ebx as usize,
                bytes: leaf.eax as usize,
            }
        };
        Layout {
            avx: component(AVX),
            opmask: component(OPMASK),
            zmm_hi256: component(ZMM_HI256),
            hi16_zmm: component(HI16_ZMM),
        }
    })
}

/// The interrupted vector registers: an XSAVE image in a signal frame.
struct Frame(*mut u8);

impl Frame {
    fn header(&self, at: usize) -> *mut u64 {
        // SAFETY: the kernel's image covers the FXSAVE area and XSAVE header.
        unsafe { self.0.add(at).cast() }
    }
    fn holds(&self, component: Component) -> bool {
        // SAFETY: as above.
        unsafe { *self.header(XFEATURES) & 1 << component.bit != 0 }
    }
    fn live(&self, component: Component) -> bool {
        // SAFETY: as above.
        self.holds(component) && unsafe { *self.header(XSTATE_BV) & 1 << component.bit != 0 }
    }
    /// The parts of ZMM `register`: its byte range, the component holding it
    /// (`None` for the legacy XMM area) and its offset in the image.
    fn parts(
        register: u8,
    ) -> impl Iterator<Item = (core::ops::Range<usize>, Option<Component>, usize)> {
        let layout = layout();
        let r = usize::from(register);
        let parts = if r < 16 {
            [
                Some((0..16, None, XMM + 16 * r)),
                Some((16..32, Some(layout.avx), layout.avx.offset + 16 * r)),
                Some((
                    32..64,
                    Some(layout.zmm_hi256),
                    layout.zmm_hi256.offset + 32 * r,
                )),
            ]
        } else {
            [
                Some((
                    0..64,
                    Some(layout.hi16_zmm),
                    layout.hi16_zmm.offset + 64 * (r - 16),
                )),
                None,
                None,
            ]
        };
        parts.into_iter().flatten()
    }
    fn read(&self, register: u8) -> [u8; 64] {
        let mut value = [0; 64];
        for (range, component, at) in Self::parts(register) {
            if component.is_none_or(|component| self.live(component)) {
                // SAFETY: the part lies inside the image the frame holds.
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        self.0.add(at),
                        value[range.clone()].as_mut_ptr(),
                        range.len(),
                    );
                }
            }
        }
        value
    }
    /// Opmask register `k`, zero when the component is in its initial state.
    fn opmask(&self, k: u8) -> u64 {
        let opmask = layout().opmask;
        if !self.live(opmask) {
            return 0;
        }
        // SAFETY: the component lies inside the image.
        unsafe {
            self.0
                .add(opmask.offset + 8 * usize::from(k))
                .cast::<u64>()
                .read_unaligned()
        }
    }
    fn mxcsr(&self) -> *mut u32 {
        // SAFETY: MXCSR lies inside the FXSAVE area.
        unsafe { self.0.add(MXCSR).cast() }
    }
    /// Write all of ZMM `register`; false when a nonzero part has no place in
    /// the image.
    fn write(&mut self, register: u8, value: &[u8; 64]) -> bool {
        for (range, component, at) in Self::parts(register) {
            let part = &value[range];
            if let Some(component) = component {
                let zero = part.iter().all(|&byte| byte == 0);
                if !self.holds(component) {
                    if zero {
                        continue;
                    }
                    return false;
                }
                if !self.live(component) {
                    if zero {
                        continue;
                    }
                    // Its stale bytes become every register's live state.
                    // SAFETY: the component lies inside the image.
                    unsafe {
                        core::ptr::write_bytes(self.0.add(component.offset), 0, component.bytes);
                        *self.header(XSTATE_BV) |= 1 << component.bit;
                    }
                }
            }
            // SAFETY: the part lies inside the image.
            unsafe { core::ptr::copy_nonoverlapping(part.as_ptr(), self.0.add(at), part.len()) };
        }
        true
    }
}

/// A register-only VEX3 or EVEX instruction, with an immediate in EVEX map 3.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Instruction {
    evex: bool,
    /// The opcode map: 2 for 0F38 (VEX), 5 or 6 for the FP16 maps (EVEX).
    map: u8,
    opcode: u8,
    /// 0: none, 1: 66, 2: F3, 3: F2.
    pp: u8,
    w: bool,
    /// The vector length in bytes.
    bytes: usize,
    reg: u8,
    vvvv: u8,
    rm: u8,
    /// EVEX.aaa, the writemask register; k0 writes every lane.
    opmask: u8,
    /// EVEX.z: masked lanes clear instead of keeping the destination.
    zeroing: bool,
    /// The immediate byte of a map 3 form.
    imm: u8,
}

impl Instruction {
    /// C4 or 62, the payload bytes, the opcode, ModRM and any immediate, from
    /// the first two bytes.
    fn length(first: u8, payload: u8) -> usize {
        match (first == 0x62, payload & 7) {
            (true, 3) => 7,
            (true, _) => 6,
            (false, _) => 5,
        }
    }
    fn decode(bytes: &[u8]) -> Option<Self> {
        let invert = |byte: u8, bit: u8| (!byte >> bit & 1) << 3;
        match *bytes {
            [0xc4, first, second, opcode, modrm] if first & 0x1f == 2 && modrm >> 6 == 3 => {
                Some(Self {
                    evex: false,
                    map: 2,
                    opcode,
                    pp: second & 3,
                    w: second & 0x80 != 0,
                    bytes: if second & 4 != 0 { 32 } else { 16 },
                    reg: (modrm >> 3 & 7) | invert(first, 7),
                    vvvv: !second >> 3 & 15,
                    rm: (modrm & 7) | invert(first, 5),
                    opmask: 0,
                    zeroing: false,
                    imm: 0,
                })
            }
            // EVEX.b (rounding or broadcast) and L'L = 3 are not modeled.
            [0x62, p0, p1, p2, opcode, modrm, ref rest @ ..]
                if p0 & 8 == 0
                    && p1 & 4 != 0
                    && p2 & 0x10 == 0
                    && p2 >> 5 & 3 != 3
                    && modrm >> 6 == 3
                    && rest.len() == usize::from(p0 & 7 == 3) =>
            {
                Some(Self {
                    evex: true,
                    map: p0 & 7,
                    opcode,
                    pp: p1 & 3,
                    w: p1 & 0x80 != 0,
                    bytes: 16 << (p2 >> 5 & 3),
                    reg: (modrm >> 3 & 7) | invert(p0, 7) | invert(p0, 4) << 1,
                    vvvv: (!p1 >> 3 & 15) | invert(p2, 3) << 1,
                    rm: (modrm & 7) | invert(p0, 5) | invert(p0, 6) << 1,
                    opmask: p2 & 7,
                    zeroing: p2 & 0x80 != 0,
                    imm: rest.first().copied().unwrap_or(0),
                })
            }
            _ => None,
        }
    }
}

fn qwords(value: &[u8; 64]) -> [u64; 4] {
    core::array::from_fn(|i| u64::from_le_bytes(value[8 * i..][..8].try_into().unwrap()))
}
fn from_qwords(qwords: [u64; 4]) -> [u8; 64] {
    let mut value = [0; 64];
    for (i, qword) in qwords.into_iter().enumerate() {
        value[8 * i..][..8].copy_from_slice(&qword.to_le_bytes());
    }
    value
}

/// VPDPWUUD, VPDPWUSD and VPDPWSUD (AVX-VNNI-INT16): each dword of
/// `accumulator` plus the two products of its words of `left` and `right`,
/// wrapping; bytes above the vector length clear.
fn word_dots(
    accumulator: &[u8; 64],
    left: &[u8; 64],
    right: &[u8; 64],
    signed: [bool; 2],
    bytes: usize,
) -> [u8; 64] {
    let word = |value: &[u8; 64], at: usize, signed: bool| {
        let word = u16::from_le_bytes([value[at], value[at + 1]]);
        if signed {
            i64::from(word as i16)
        } else {
            i64::from(word)
        }
    };
    let mut result = [0; 64];
    for lane in (0..bytes).step_by(4) {
        let sum = [lane, lane + 2].into_iter().fold(
            u32::from_le_bytes(accumulator[lane..][..4].try_into().unwrap()),
            |sum, at| {
                sum.wrapping_add((word(left, at, signed[0]) * word(right, at, signed[1])) as u32)
            },
        );
        result[lane..][..4].copy_from_slice(&sum.to_le_bytes());
    }
    result
}

/// VPDPBUUD, VPDPBSUD and VPDPBSSD (AVX-VNNI-INT8): each dword of
/// `accumulator` plus the four products of its bytes of `left` and `right`,
/// wrapping; bytes above the vector length clear.
fn byte_dots(
    accumulator: &[u8; 64],
    left: &[u8; 64],
    right: &[u8; 64],
    signed: [bool; 2],
    bytes: usize,
) -> [u8; 64] {
    let extend = |byte: u8, signed: bool| {
        if signed {
            i32::from(byte as i8)
        } else {
            i32::from(byte)
        }
    };
    let mut result = [0; 64];
    for lane in (0..bytes).step_by(4) {
        let sum = (lane..lane + 4).fold(
            u32::from_le_bytes(accumulator[lane..][..4].try_into().unwrap()),
            |sum, i| {
                sum.wrapping_add((extend(left[i], signed[0]) * extend(right[i], signed[1])) as u32)
            },
        );
        result[lane..][..4].copy_from_slice(&sum.to_le_bytes());
    }
    result
}

fn large_sigma0(x: u64) -> u64 {
    x.rotate_right(28) ^ x.rotate_right(34) ^ x.rotate_right(39)
}
fn large_sigma1(x: u64) -> u64 {
    x.rotate_right(14) ^ x.rotate_right(18) ^ x.rotate_right(41)
}
fn small_sigma0(x: u64) -> u64 {
    x.rotate_right(1) ^ x.rotate_right(8) ^ x >> 7
}
fn small_sigma1(x: u64) -> u64 {
    x.rotate_right(19) ^ x.rotate_right(61) ^ x >> 6
}

/// VSHA512RNDS2: two rounds on the state C, D, G, H in `state` (qwords 3, 2,
/// 1, 0) and A, B, E, F in `source` with the summed K+W pair `wk`, leaving A,
/// B, E, F there.
fn sha512_rounds(state: [u64; 4], source: [u64; 4], wk: [u64; 4]) -> [u64; 4] {
    let [mut h, mut g, mut d, mut c] = state;
    let [mut f, mut e, mut b, mut a] = source;
    for &k in &wk[..2] {
        let t1 = (e & f ^ !e & g)
            .wrapping_add(large_sigma1(e))
            .wrapping_add(k)
            .wrapping_add(h);
        let t2 = (a & b ^ a & c ^ b & c).wrapping_add(large_sigma0(a));
        (h, g, f, e) = (g, f, e, d.wrapping_add(t1));
        (d, c, b, a) = (c, b, a, t1.wrapping_add(t2));
    }
    [f, e, b, a]
}

/// VSHA512MSG1: each qword of `state` plus the small sigma0 of the next, with
/// `source` qword 0 after qword 3.
fn sha512_message1(state: [u64; 4], source: [u64; 4]) -> [u64; 4] {
    let next = [state[1], state[2], state[3], source[0]];
    core::array::from_fn(|i| state[i].wrapping_add(small_sigma0(next[i])))
}

/// VSHA512MSG2: W16..W19 from `state` and W14, W15 in `source` qwords 2 and
/// 3, each new word feeding the one two after it.
fn sha512_message2(state: [u64; 4], source: [u64; 4]) -> [u64; 4] {
    let mut w = [source[2], source[3], 0, 0, 0, 0];
    for i in 0..4 {
        w[i + 2] = state[i].wrapping_add(small_sigma1(w[i]));
    }
    [w[2], w[3], w[4], w[5]]
}

/// IEEE binary16 arithmetic as the AVX512-FP16 instructions perform it,
/// with MXCSR rounding and status (SDM Vol. 1 4.9 and Vol. 2). The
/// instructions ignore MXCSR.DAZ and FTZ and report no denormal operands
/// (DE), as Intel SDE does.
mod half {
    pub(super) const INVALID: u32 = 1;
    pub(super) const DIVIDE_BY_ZERO: u32 = 4;
    pub(super) const OVERFLOW: u32 = 8;
    pub(super) const UNDERFLOW: u32 = 16;
    pub(super) const INEXACT: u32 = 32;
    const SIGN: u16 = 0x8000;
    const EXPONENT: u16 = 0x7c00;
    const QUIET: u16 = 0x0200;
    /// The QNaN floating-point indefinite.
    const INDEFINITE: u16 = 0xfe00;
    const MAXIMUM: u16 = 0x7bff;

    #[derive(Clone, Copy)]
    pub(super) enum Operation {
        Add,
        Subtract,
        Multiply,
        Divide,
    }

    /// The sum `p` * 2^-48 as an f64 that rounds like it: sign and magnitude, with
    /// any bits past 53 significant ones jammed into a sticky bit far below
    /// every binary16 rounding position.
    fn lattice(sum: i128) -> f64 {
        let magnitude = sum.unsigned_abs();
        let excess = (128 - magnitude.leading_zeros()).saturating_sub(53);
        let kept = magnitude >> excess;
        let sticky = u128::from(kept << excess != magnitude);
        let value = (kept | sticky) as f64 * power(excess as i32 - 48);
        if sum < 0 { -value } else { value }
    }

    fn power(exponent: i32) -> f64 {
        f64::from_bits(((exponent + 1023) as u64) << 52)
    }
    fn nan(bits: u16) -> bool {
        bits & EXPONENT == EXPONENT && bits & !(SIGN | EXPONENT) != 0
    }
    fn value(bits: u16) -> f64 {
        let magnitude = bits & !SIGN;
        let value = if magnitude & EXPONENT == EXPONENT {
            f64::INFINITY
        } else if magnitude & EXPONENT == 0 {
            f64::from(magnitude) * power(-24)
        } else {
            f64::from(magnitude & 0x3ff | 0x400) * power(i32::from(magnitude >> 10) - 25)
        };
        if bits & SIGN != 0 { -value } else { value }
    }

    /// Round the nonnegative `scaled` to an integer under MXCSR.RC `rounding`
    /// for a value of sign `negative`.
    fn integer(scaled: f64, rounding: u32, negative: bool) -> f64 {
        let floor = scaled.floor();
        let fraction = scaled - floor;
        let up = match rounding {
            0 => fraction > 0.5 || fraction == 0.5 && floor % 2.0 == 1.0,
            1 => negative && fraction != 0.0,
            2 => !negative && fraction != 0.0,
            _ => false,
        };
        floor + f64::from(u8::from(up))
    }

    /// Round the finite nonzero `exact` to binary16. Quotients of binary16
    /// values are either exact in f64 or farther than 2^-22 relative from
    /// every binary16 value and midpoint, so rounding the f64 quotient
    /// rounds the exact one. Tininess is detected after rounding.
    fn round(exact: f64, rounding: u32) -> (u16, u32) {
        let negative = exact < 0.0;
        let magnitude = exact.abs();
        let exponent = ((magnitude.to_bits() >> 52) as i32) - 1023;
        let unbounded =
            integer(magnitude / power(exponent - 10), rounding, negative) * power(exponent - 10);
        let quantum = power(exponent.max(-14) - 10);
        let rounded = integer(magnitude / quantum, rounding, negative) * quantum;
        let mut status = if rounded == magnitude { 0 } else { INEXACT };
        let sign = if negative { SIGN } else { 0 };
        if rounded > value(MAXIMUM) {
            let infinity = match rounding {
                0 => true,
                1 => negative,
                2 => !negative,
                _ => false,
            };
            let bits = if infinity { EXPONENT } else { MAXIMUM };
            return (sign | bits, OVERFLOW | INEXACT);
        }
        if unbounded < power(-14) && status != 0 {
            status |= UNDERFLOW;
        }
        let bits = if rounded >= power(-14) {
            let exponent = ((rounded.to_bits() >> 52) as i32) - 1023;
            ((exponent + 15) as u16) << 10 | ((rounded / power(exponent - 10)) as u16 - 0x400)
        } else {
            (rounded / power(-24)) as u16
        };
        (sign | bits, status)
    }

    /// The magnitude of the finite `bits` in units of 2^-24.
    fn units(bits: u16) -> i128 {
        let magnitude = bits & !SIGN;
        if magnitude & EXPONENT == 0 {
            i128::from(magnitude)
        } else {
            i128::from(magnitude & 0x3ff | 0x400) << ((magnitude >> 10) - 1)
        }
    }

    /// One lane of `left` * `right` + `addend`, each sign changed first as the
    /// form says and the exact sum rounded once. NaN operands propagate the
    /// first NaN of `left`, `right` and `addend`, quiet with its sign unchanged.
    /// An infinity times zero signals invalid in the absence of NaNs, as in
    /// the Intel SDM's FMA numeric behavior table.
    pub(super) fn fused(
        [left, addend, right]: [u16; 3],
        [negate_product, negate_addend]: [bool; 2],
        rounding: u32,
    ) -> (u16, u32) {
        if let Some(&first) = [left, right, addend].iter().find(|&&bits| nan(bits)) {
            let status = [left, right, addend]
                .iter()
                .any(|&bits| nan(bits) && bits & QUIET == 0);
            return (first | QUIET, if status { INVALID } else { 0 });
        }
        let invalid = (INDEFINITE, INVALID);
        let (x, y) = (value(left), value(right));
        if x == 0.0 && y.is_infinite() || x.is_infinite() && y == 0.0 {
            return invalid;
        }
        let sign = |negative: bool| if negative { SIGN } else { 0 };
        let product_sign = (left ^ right) & SIGN ^ sign(negate_product);
        let addend = addend ^ sign(negate_addend);
        let c = value(addend);
        if x.is_infinite() || y.is_infinite() {
            if c.is_infinite() && addend & SIGN != product_sign {
                return invalid;
            }
            return (product_sign | EXPONENT, 0);
        }
        if c.is_infinite() {
            return (addend, 0);
        }
        let signed = |magnitude: i128, sign: u16| if sign != 0 { -magnitude } else { magnitude };
        let product = units(left) * units(right);
        let shifted = units(addend) << 24;
        let sum = signed(product, product_sign) + signed(shifted, addend & SIGN);
        if sum == 0 {
            let alike = product == 0 && shifted == 0 && product_sign == addend & SIGN;
            let negative = if alike {
                product_sign != 0
            } else {
                rounding == 1
            };
            return (sign(negative), 0);
        }
        round(lattice(sum), rounding)
    }

    /// The integer of `width` bits nearest `bits` in the rounding direction,
    /// or the integer indefinite with IE.
    pub(super) fn to_integer(
        bits: u16,
        [signed, truncate]: [bool; 2],
        width: u32,
        rounding: u32,
    ) -> (u64, u32) {
        let mask = u64::MAX >> (64 - width);
        let indefinite = if signed { 1 << (width - 1) } else { mask };
        if bits & EXPONENT == EXPONENT {
            return (indefinite, INVALID);
        }
        let exact = value(bits);
        let negative = exact < 0.0;
        let magnitude = exact.abs();
        let rounded = integer(magnitude, if truncate { 3 } else { rounding }, negative);
        let result = if negative {
            -(rounded as i128)
        } else {
            rounded as i128
        };
        let range = if signed {
            -(1 << (width - 1))..=(1 << (width - 1)) - 1
        } else {
            0..=i128::from(mask)
        };
        if !range.contains(&result) {
            return (indefinite, INVALID);
        }
        let status = if rounded == magnitude { 0 } else { INEXACT };
        (result as u64 & mask, status)
    }

    /// The Half nearest the integer `value`.
    pub(super) fn from_integer(value: i128, rounding: u32) -> (u16, u32) {
        if value == 0 {
            (0, 0)
        } else {
            round(lattice(value << 48), rounding)
        }
    }

    /// The Half nearest `value`.
    pub(super) fn from_double(value: f64, rounding: u32) -> (u16, u32) {
        let bits = value.to_bits();
        let sign = (bits >> 48) as u16 & SIGN;
        if value.is_nan() {
            let fraction = (bits >> 42) as u16 & 0x3ff;
            let status = if bits >> 51 & 1 == 0 { INVALID } else { 0 };
            return (sign | EXPONENT | QUIET | fraction, status);
        }
        if value.is_infinite() || value == 0.0 {
            return (sign | if value == 0.0 { 0 } else { EXPONENT }, 0);
        }
        // Beyond these bounds every rounding sees only the sign and inexactness.
        let bound = value.abs().clamp(power(-40), power(20));
        round(if value < 0.0 { -bound } else { bound }, rounding)
    }

    /// One lane: the result bits and the MXCSR status it raises.
    pub(super) fn arithmetic(
        operation: Operation,
        left: u16,
        right: u16,
        rounding: u32,
    ) -> (u16, u32) {
        if nan(left) || nan(right) {
            let signaling = |bits| nan(bits) && bits & QUIET == 0;
            let status = if signaling(left) || signaling(right) {
                INVALID
            } else {
                0
            };
            return (if nan(left) { left } else { right } | QUIET, status);
        }
        let status = 0;
        let (x, y) = (value(left), value(right));
        let invalid = (INDEFINITE, status | INVALID);
        let exact = match operation {
            Operation::Add | Operation::Subtract => {
                let y = if matches!(operation, Operation::Subtract) {
                    -y
                } else {
                    y
                };
                if x.is_infinite() && y.is_infinite() && x != y {
                    return invalid;
                }
                if x + y == 0.0 {
                    // Zeros of one sign keep it; any other exact zero sum is
                    // -0 when rounding down and +0 otherwise (IEEE 754 6.3).
                    let alike =
                        x == 0.0 && y == 0.0 && x.is_sign_negative() == y.is_sign_negative();
                    let negative = if alike {
                        x.is_sign_negative()
                    } else {
                        rounding == 1
                    };
                    return (if negative { SIGN } else { 0 }, status);
                }
                x + y
            }
            Operation::Multiply => {
                if x == 0.0 && y.is_infinite() || x.is_infinite() && y == 0.0 {
                    return invalid;
                }
                x * y
            }
            Operation::Divide => {
                if x == 0.0 && y == 0.0 || x.is_infinite() && y.is_infinite() {
                    return invalid;
                }
                if y == 0.0 && !x.is_infinite() {
                    let sign = (left ^ right) & SIGN;
                    return (sign | EXPONENT, status | DIVIDE_BY_ZERO);
                }
                x / y
            }
        };
        if exact.is_infinite() || exact == 0.0 {
            let sign = if exact.is_sign_negative() { SIGN } else { 0 };
            return (sign | if exact == 0.0 { 0 } else { EXPONENT }, status);
        }
        let (bits, rounded) = round(exact, rounding);
        (bits, status | rounded)
    }
}

/// MXCSR and the opmask registers the instruction reads.
struct Control {
    mxcsr: u32,
    opmask: u64,
}

/// The Half at byte `at` of `value`.
fn half_word(value: &[u8; 64], at: usize) -> u16 {
    u16::from_le_bytes([value[at], value[at + 1]])
}

/// FP16 lanes of `bytes`, each from `lane` given its byte offset, under
/// `instruction`'s writemask with MXCSR status; `None` when an unmasked
/// exception would trap.
fn half_lanes(
    instruction: Instruction,
    destination: &[u8; 64],
    lane: impl Fn(usize) -> (u16, u32),
    control: &mut Control,
) -> Option<[u8; 64]> {
    let mut result = [0; 64];
    let mut status = 0;
    for index in 0..instruction.bytes / 2 {
        let at = 2 * index;
        let active = instruction.opmask == 0 || control.opmask >> index & 1 != 0;
        let bits = if active {
            let (bits, raised) = lane(at);
            status |= raised;
            bits
        } else if instruction.zeroing {
            0
        } else {
            half_word(destination, at)
        };
        result[at..at + 2].copy_from_slice(&bits.to_le_bytes());
    }
    // Masked exceptions only: the status bits whose MXCSR mask is clear trap.
    if status & !(control.mxcsr >> 7) & 0x3f != 0 {
        return None;
    }
    control.mxcsr |= status;
    Some(result)
}

/// VMINMAXPS/PD with bytes per lane `lane`, or SS/SD when `scalar`, at the
/// immediates [`crate::x86_64::asm::vector_extensions::extrema`] uses: -0
/// orders below +0, the first NaN returns quiet and an SNaN is invalid.
fn extrema(
    instruction: Instruction,
    [first, second]: [&[u8; 64]; 2],
    lane: usize,
    scalar: bool,
    control: &mut Control,
) -> [u8; 64] {
    let bits = |value: &[u8; 64], at: usize| {
        let mut bytes = [0; 8];
        bytes[..lane].copy_from_slice(&value[at..at + lane]);
        u64::from_le_bytes(bytes)
    };
    let quiet = 1u64 << if lane == 4 { 22 } else { 51 };
    let float = |bits: u64| {
        if lane == 4 {
            f64::from(f32::from_bits(bits as u32))
        } else {
            f64::from_bits(bits)
        }
    };
    let mut result = if scalar { *first } else { [0; 64] };
    let lanes = if scalar { 1 } else { instruction.bytes / lane };
    for index in 0..lanes {
        let at = index * lane;
        let (a, b) = (bits(first, at), bits(second, at));
        let (x, y) = (float(a), float(b));
        let chosen = if x.is_nan() || y.is_nan() {
            let signaling = |value: f64, bits: u64| value.is_nan() && bits & quiet == 0;
            if signaling(x, a) || signaling(y, b) {
                control.mxcsr |= half::INVALID;
            }
            quiet | if x.is_nan() { a } else { b }
        } else {
            let maximum = instruction.imm & 1 != 0;
            match (x < y, y < x) {
                (true, _) => {
                    if maximum {
                        b
                    } else {
                        a
                    }
                }
                (_, true) => {
                    if maximum {
                        a
                    } else {
                        b
                    }
                }
                _ if maximum => a & b,
                _ => a | b,
            }
        };
        result[at..at + lane].copy_from_slice(&chosen.to_le_bytes()[..lane]);
    }
    if scalar {
        result[16..].fill(0);
    }
    result
}

/// The new value of the destination register from its old value and the
/// vvvv and ModRM.rm sources, or `None` when the form is not modeled.
fn apply(
    instruction: Instruction,
    destination: [u8; 64],
    first: [u8; 64],
    second: [u8; 64],
    control: &mut Control,
) -> Option<[u8; 64]> {
    let bytes = instruction.bytes;
    if instruction.evex {
        if let (3, opcode @ (0x52 | 0x53), 1, 0, 4 | 5) = (
            instruction.map,
            instruction.opcode,
            instruction.pp,
            instruction.opmask,
            instruction.imm,
        ) {
            let lane = if instruction.w { 8 } else { 4 };
            let sources = [&first, &second];
            return Some(extrema(instruction, sources, lane, opcode == 0x53, control));
        }
        let rounding = control.mxcsr >> 13 & 3;
        if instruction.map == 6 {
            let signs = match (instruction.opcode, instruction.pp, instruction.w) {
                (0x98, 1, false) => [false, false],
                (0x9a, 1, false) => [false, true],
                (0x9c, 1, false) => [true, false],
                (0x9e, 1, false) => [true, true],
                _ => return None,
            };
            let lane = |at| {
                let word = |value| half_word(value, at);
                half::fused(
                    [word(&destination), word(&first), word(&second)],
                    signs,
                    rounding,
                )
            };
            return half_lanes(instruction, &destination, lane, control);
        }
        match (
            instruction.map,
            instruction.opcode,
            instruction.pp,
            instruction.w,
        ) {
            (5, opcode @ (0x7c | 0x7d), pp @ (0 | 1), false) => {
                let forms = [pp == 1, opcode == 0x7c];
                let lane = |at| {
                    let (bits, status) =
                        half::to_integer(half_word(&second, at), forms, 16, rounding);
                    (bits as u16, status)
                };
                return half_lanes(instruction, &destination, lane, control);
            }
            (5, 0x7d, pp @ (2 | 3), false) => {
                let lane = |at| {
                    let word = half_word(&second, at);
                    let value = if pp == 2 {
                        i128::from(word as i16)
                    } else {
                        i128::from(word)
                    };
                    half::from_integer(value, rounding)
                };
                return half_lanes(instruction, &destination, lane, control);
            }
            (5, 0x5a, 3, true) if instruction.opmask == 0 => {
                let double = f64::from_bits(qwords(&second)[0]);
                let (bits, status) = half::from_double(double, rounding);
                if status & !(control.mxcsr >> 7) & 0x3f != 0 {
                    return None;
                }
                control.mxcsr |= status;
                let mut result = [0; 64];
                result[..16].copy_from_slice(&first[..16]);
                result[..2].copy_from_slice(&bits.to_le_bytes());
                return Some(result);
            }
            _ => {}
        }
        let operation = match (
            instruction.map,
            instruction.opcode,
            instruction.pp,
            instruction.w,
        ) {
            (5, 0x58, 0, false) => half::Operation::Add,
            (5, 0x5c, 0, false) => half::Operation::Subtract,
            (5, 0x59, 0, false) => half::Operation::Multiply,
            (5, 0x5e, 0, false) => half::Operation::Divide,
            _ => return None,
        };
        let lane = |at| {
            let (left, right) = (half_word(&first, at), half_word(&second, at));
            half::arithmetic(operation, left, right, rounding)
        };
        return half_lanes(instruction, &destination, lane, control);
    }
    let vex = instruction;
    let wide = bytes == 32;
    match (vex.opcode, vex.pp, vex.w) {
        (0x50, pp, false) if pp != 1 => {
            let signed = [pp != 0, pp == 3];
            Some(byte_dots(&destination, &first, &second, signed, bytes))
        }
        (0xd2, pp, false) => {
            // VEX.pp: VPDPWUUD, VPDPWUSD (signed second source), VPDPWSUD.
            let signed = [pp == 2, pp == 1];
            Some(word_dots(&destination, &first, &second, signed, bytes))
        }
        (0xcb, 3, false) if wide => Some(from_qwords(sha512_rounds(
            qwords(&destination),
            qwords(&first),
            [qwords(&second)[0], qwords(&second)[1], 0, 0],
        ))),
        (0xcc, 3, false) if wide && vex.vvvv == 0 => Some(from_qwords(sha512_message1(
            qwords(&destination),
            qwords(&second),
        ))),
        (0xcd, 3, false) if wide && vex.vvvv == 0 => Some(from_qwords(sha512_message2(
            qwords(&destination),
            qwords(&second),
        ))),
        _ => None,
    }
}

/// A VCVT(T)SH2(U)SI or VCVT(U)SI2SH: [signed, truncate] for one to a general
/// register, or [signed] for one from it.
#[derive(Clone, Copy)]
enum General {
    To([bool; 2]),
    From(bool),
}

impl General {
    fn of(instruction: Instruction) -> Option<Self> {
        if !instruction.evex || instruction.map != 5 || instruction.pp != 2 {
            return None;
        }
        match instruction.opcode {
            0x2c => Some(Self::To([true, true])),
            0x2d => Some(Self::To([true, false])),
            0x78 => Some(Self::To([false, true])),
            0x79 => Some(Self::To([false, false])),
            0x2a => Some(Self::From(true)),
            0x7b => Some(Self::From(false)),
            _ => None,
        }
    }
}

/// `ucontext_t` gregs slots of RAX..R15, by x86 register number.
const GENERAL_SLOTS: [usize; 16] = [13, 14, 12, 11, 15, 10, 9, 8, 0, 1, 2, 3, 4, 5, 6, 7];

/// The bits of the [`General`] form between `value` and the Half of `rm`,
/// or `None` when an unmasked exception would trap.
fn general_result(
    form: General,
    instruction: Instruction,
    value: u64,
    half_bits: u16,
    control: &mut Control,
) -> Option<u64> {
    let rounding = control.mxcsr >> 13 & 3;
    let width = if instruction.w { 64 } else { 32 };
    let (bits, status) = match form {
        General::To(forms) => half::to_integer(half_bits, forms, width, rounding),
        General::From(signed) => {
            let value = match (signed, instruction.w) {
                (true, true) => i128::from(value as i64),
                (true, false) => i128::from(value as u32 as i32),
                (false, true) => i128::from(value),
                (false, false) => i128::from(value as u32),
            };
            let (bits, status) = half::from_integer(value, rounding);
            (u64::from(bits), status)
        }
    };
    if status & !(control.mxcsr >> 7) & 0x3f != 0 {
        return None;
    }
    control.mxcsr |= status;
    Some(bits)
}

/// Apply a [`General`] form to the frame and `gregs`.
///
/// # Safety
/// `gregs` addresses the interrupted general registers.
unsafe fn apply_general(
    form: General,
    instruction: Instruction,
    frame: &mut Frame,
    gregs: *mut u64,
    control: &mut Control,
) -> Option<()> {
    match form {
        General::To(_) => {
            let bits = half_word(&frame.read(instruction.rm), 0);
            let result = general_result(form, instruction, 0, bits, control)?;
            // SAFETY: the caller passes the register bank; a slot is in it.
            unsafe { *gregs.add(GENERAL_SLOTS[usize::from(instruction.reg)]) = result };
        }
        General::From(_) => {
            // SAFETY: as above.
            let value = unsafe { *gregs.add(GENERAL_SLOTS[usize::from(instruction.rm)]) };
            let bits = general_result(form, instruction, value, 0, control)? as u16;
            let mut result = [0; 64];
            result[..16].copy_from_slice(&frame.read(instruction.vvvv)[..16]);
            result[..2].copy_from_slice(&bits.to_le_bytes());
            frame.write(instruction.reg, &result).then_some(())?;
        }
    }
    Some(())
}

unsafe extern "C" fn trap(signal: i32, info: *mut c_void, context: *mut c_void) {
    // SAFETY: the kernel passes a ucontext_t whose RIP addresses the
    // instruction that faulted, and fpregs its XSAVE image.
    unsafe {
        let context = context.cast::<u8>();
        let rip = context.add(GREGS + 8 * RIP).cast::<u64>();
        let image = *context.add(FPREGS).cast::<*mut u8>();
        let start = *rip as *const u8;
        // The faulting instruction is mapped; read only its own bytes.
        let length = Instruction::length(start.read(), start.add(1).read());
        let bytes = core::slice::from_raw_parts(start, length);
        let mut frame = Frame(image);
        if !image.is_null() && image.add(MAGIC1).cast::<u32>().read() == FP_XSTATE_MAGIC1 {
            let mut control = Control {
                mxcsr: frame.mxcsr().read(),
                opmask: 0,
            };
            let decoded = Instruction::decode(bytes);
            if let Some((instruction, form)) =
                decoded.and_then(|instruction| Some((instruction, General::of(instruction)?)))
            {
                let gregs = context.add(GREGS).cast::<u64>();
                if apply_general(form, instruction, &mut frame, gregs, &mut control).is_some() {
                    frame.mxcsr().write(control.mxcsr);
                    *rip += length as u64;
                    return;
                }
            }
            let result = decoded.and_then(|instruction| {
                control.opmask = frame.opmask(instruction.opmask);
                let [destination, first, second] =
                    [instruction.reg, instruction.vvvv, instruction.rm].map(|r| frame.read(r));
                Some((
                    instruction,
                    apply(instruction, destination, first, second, &mut control)?,
                ))
            });
            if let Some((instruction, result)) = result
                && frame.write(instruction.reg, &result)
            {
                frame.mxcsr().write(control.mxcsr);
                *rip += length as u64;
                return;
            }
        }
        let prior = PRIOR.load(Ordering::Relaxed);
        if prior != 0 {
            let prior: unsafe extern "C" fn(i32, *mut c_void, *mut c_void) =
                core::mem::transmute(prior);
            prior(signal, info, context.cast());
            return;
        }
        // Fault again without the handler.
        let message = b"modeled: unmodeled instruction, restoring SIGILL\n";
        write(2, message.as_ptr().cast(), message.len());
        let default = SigAction {
            handler: SIG_DFL,
            mask: [0; 16],
            flags: 0,
            restorer: 0,
        };
        sigaction(SIGILL, &default, core::ptr::null_mut());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        GuestState, context::ExecutionContext, x86_64::asm::vector_extensions::execution::Native,
    };
    use alloc::vec::Vec;

    impl Instruction {
        fn encode(self) -> Vec<u8> {
            let bit = |register: u8, bit: u8| !register >> bit & 1;
            if !self.evex {
                return alloc::vec![
                    0xc4,
                    bit(self.reg, 3) << 7 | 1 << 6 | bit(self.rm, 3) << 5 | self.map,
                    u8::from(self.w) << 7
                        | (!self.vvvv & 15) << 3
                        | u8::from(self.bytes == 32) << 2
                        | self.pp,
                    self.opcode,
                    0xc0 | (self.reg & 7) << 3 | (self.rm & 7),
                ];
            }
            let mut bytes = alloc::vec![
                0x62,
                bit(self.reg, 3) << 7
                    | bit(self.rm, 4) << 6
                    | bit(self.rm, 3) << 5
                    | bit(self.reg, 4) << 4
                    | self.map,
                u8::from(self.w) << 7 | (!self.vvvv & 15) << 3 | 4 | self.pp,
                u8::from(self.zeroing) << 7
                    | ((self.bytes / 32) as u8) << 5
                    | bit(self.vvvv, 4) << 3
                    | self.opmask,
                self.opcode,
                0xc0 | (self.reg & 7) << 3 | (self.rm & 7),
            ];
            if self.map == 3 {
                bytes.push(self.imm);
            }
            bytes
        }
    }

    /// An instruction addressing RDI + `offset` (ModRM 10 reg 111).
    fn at_rdi(prefix: &[u8], reg: u8, offset: usize) -> Vec<u8> {
        let mut bytes = prefix.to_vec();
        bytes.push(0x87 | reg << 3);
        bytes.extend((offset as u32).to_le_bytes());
        bytes
    }

    /// YMM `register` as the test loads it: guest vectors 2r and 2r + 1.
    fn ymm(state: &GuestState, register: usize) -> [u8; 64] {
        let mut value = [0; 64];
        for (i, limb) in state.v.as_flattened()[4 * register..][..4]
            .iter()
            .enumerate()
        {
            value[8 * i..][..8].copy_from_slice(&limb.to_le_bytes());
        }
        value
    }

    /// A binary16 value from every class: normals across the exponent range,
    /// zeros, denormals, infinities and both kinds of NaN.
    fn half(random: u64) -> u16 {
        let sign = (random >> 63) as u16 * 0x8000;
        let fraction = (random >> 20) as u16 & 0x3ff;
        sign | match random % 16 {
            0 => 0,
            1 => fraction.max(1),
            2 => 0x7c00,
            3 => 0x7c00 | fraction.max(1),
            4 => 0x7e00 | fraction,
            5 => 0x3c00,
            6 => 0x0400 | fraction,
            7 => 0x7800 | fraction,
            8 => 0x3800 | fraction,
            _ => ((random >> 32) as u16 & 0x7fff) % 0x7c00,
        }
    }

    #[test]
    fn models_match_the_host_instructions() {
        let host = TargetCpu::detect();
        let int8 = TargetCpu::X86_64_V3.with_avx_vnni_int8();
        let int16 = TargetCpu::X86_64_V3.with_avx_vnni_int16();
        let sha512 = TargetCpu::X86_64_V3.with_sha512();
        let fp16 = TargetCpu::X86_64_V4.with_avx512fp16();
        let avx10_2 = TargetCpu::X86_64_V4.with_avx10_2();
        // YMM1 is the destination, YMM2 the vvvv source (0 when the form has
        // none) and YMM3 the ModRM.rm source.
        let vex = |opcode, pp, bytes, vvvv| Instruction {
            evex: false,
            map: 2,
            opcode,
            pp,
            w: false,
            bytes,
            reg: 1,
            vvvv,
            rm: 3,
            opmask: 0,
            zeroing: false,
            imm: 0,
        };
        let mut forms = Vec::new();
        for bytes in [16, 32] {
            for pp in [0, 2, 3] {
                forms.push((int8, vex(0x50, pp, bytes, 2)));
            }
            for pp in [0, 1, 2] {
                forms.push((int16, vex(0xd2, pp, bytes, 2)));
            }
            for opcode in [0x58, 0x5c, 0x59, 0x5e] {
                for (opmask, zeroing) in [(0, false), (1, true), (1, false)] {
                    forms.push((
                        fp16,
                        Instruction {
                            evex: true,
                            map: 5,
                            opmask,
                            zeroing,
                            ..vex(opcode, 0, bytes, 2)
                        },
                    ));
                }
            }
            for opcode in [0x98, 0x9a, 0x9c, 0x9e] {
                for (opmask, zeroing) in [(0, false), (1, true), (1, false)] {
                    forms.push((
                        fp16,
                        Instruction {
                            evex: true,
                            map: 6,
                            opmask,
                            zeroing,
                            ..vex(opcode, 1, bytes, 2)
                        },
                    ));
                }
            }
            for (w, imm) in [(false, 4), (false, 5), (true, 4), (true, 5)] {
                forms.push((
                    avx10_2,
                    Instruction {
                        evex: true,
                        map: 3,
                        w,
                        imm,
                        ..vex(0x52, 1, bytes, 2)
                    },
                ));
            }
            for (opcode, pp) in [
                (0x7d, 1),
                (0x7c, 1),
                (0x7d, 0),
                (0x7c, 0),
                (0x7d, 2),
                (0x7d, 3),
            ] {
                for (opmask, zeroing) in [(0, false), (1, true), (1, false)] {
                    forms.push((
                        fp16,
                        Instruction {
                            evex: true,
                            map: 5,
                            opmask,
                            zeroing,
                            ..vex(opcode, pp, bytes, 0)
                        },
                    ));
                }
            }
        }
        forms.push((
            fp16,
            Instruction {
                evex: true,
                map: 5,
                w: true,
                ..vex(0x5a, 3, 16, 2)
            },
        ));
        for (w, imm) in [(false, 4), (false, 5), (true, 4), (true, 5)] {
            forms.push((
                avx10_2,
                Instruction {
                    evex: true,
                    map: 3,
                    w,
                    imm,
                    ..vex(0x53, 1, 16, 2)
                },
            ));
        }
        forms.extend([
            (sha512, vex(0xcb, 3, 32, 2)),
            (sha512, vex(0xcc, 3, 32, 0)),
            (sha512, vex(0xcd, 3, 32, 0)),
        ]);
        let vectors = core::mem::offset_of!(GuestState, v);
        let x = |i: usize| core::mem::offset_of!(GuestState, x) + 8 * i;
        let mut random = 0x243f_6a88_85a3_08d3_u64;
        let mut next = move || {
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            random
        };
        let mut checked = 0;
        for (target, instruction) in forms {
            if !host.supports(target) {
                continue;
            }
            // X0 holds the MXCSR to run under, X1 receives the resulting one,
            // X2 holds the caller's and X3 the K1 writemask.
            let mut code = at_rdi(&[0x0f, 0xae], 2, x(0));
            if instruction.evex {
                code.extend(at_rdi(&[0xc5, 0xf8, 0x90], 1, x(3)));
            }
            for register in 1..4 {
                let offset = vectors + 32 * usize::from(register);
                code.extend(at_rdi(&[0xc5, 0xfe, 0x6f], register, offset));
            }
            code.extend(instruction.encode());
            code.extend(at_rdi(&[0xc5, 0xfe, 0x7f], 1, vectors + 32));
            code.extend(at_rdi(&[0x0f, 0xae], 3, x(1)));
            code.extend(at_rdi(&[0x0f, 0xae], 2, x(2)));
            code.extend([0xc5, 0xf8, 0x77, 0xc3]); // vzeroupper; ret
            let native = Native::new(&code);
            for sample in 0..256 {
                let mut state = GuestState::default();
                for limb in state.v.as_flattened_mut() {
                    *limb = match sample {
                        0 => 0x8080_8080_8080_8080,
                        1 => u64::MAX,
                        _ if instruction.evex => (0..4).fold(0, |limb, lane| {
                            limb | u64::from(half(next())) << (16 * lane)
                        }),
                        _ => next(),
                    };
                }
                let mut caller = 0u32;
                // SAFETY: STMXCSR writes four bytes to the local.
                unsafe { core::arch::asm!("stmxcsr [{}]", in(reg) &mut caller) };
                let rounding = (sample % 4) << 13;
                state.x[0] = 0x1f80 | rounding;
                state.x[2] = u64::from(caller);
                state.x[3] = next() & 0xffff;
                let first = if instruction.vvvv == 0 {
                    [0; 64]
                } else {
                    ymm(&state, 2)
                };
                let mut control = Control {
                    mxcsr: 0x1f80 | rounding as u32,
                    opmask: state.x[3],
                };
                let model = apply(
                    instruction,
                    ymm(&state, 1),
                    first,
                    ymm(&state, 3),
                    &mut control,
                )
                .unwrap();
                native.run(&mut state, &mut ExecutionContext::default());
                // Intel SDE 10.13.1 raises only IE and ZE for VADDPH, VSUBPH,
                // VMULPH and VF*132PH, adds only PE for VDIVPH (no OE or UE) and
                // raises nothing for VCVTSD2SH, so
                // the rest of their status rests on IEEE 754 and
                // `half_rounding_status_and_special_values`.
                let emulated = half::INVALID | half::DIVIDE_BY_ZERO;
                let compared = match (instruction.evex, instruction.opcode) {
                    (false, _) => !0,
                    (true, 0x5e) => !0x3f | emulated | half::INEXACT,
                    (true, 0x52 | 0x53) => !0x3f | half::INVALID,
                    (true, 0x5a) => !0x3f,
                    (true, _) => !0x3f | emulated,
                };
                assert_eq!(
                    (ymm(&state, 1)[..32].to_vec(), state.x[1] as u32 & compared),
                    (model[..32].to_vec(), control.mxcsr & compared),
                    "{instruction:x?} sample={sample}"
                );
                checked += 1;
            }
        }
        std::println!("MODELED_FORMS_CHECKED={checked}");
    }

    #[test]
    fn general_register_models_match_the_host_instructions() {
        if !TargetCpu::detect().supports(TargetCpu::X86_64_V4.with_avx512fp16()) {
            return;
        }
        let vector = |register: usize| core::mem::offset_of!(GuestState, v) + 32 * register;
        let x = |i: usize| core::mem::offset_of!(GuestState, x) + 8 * i;
        let mut random = 0x3956_c25b_f348_b538_u64;
        let mut next = move || {
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            random
        };
        let mut checked = 0;
        for opcode in [0x2c, 0x2d, 0x78, 0x79, 0x2a, 0x7b] {
            for w in [false, true] {
                let to = opcode != 0x2a && opcode != 0x7b;
                let instruction = Instruction {
                    evex: true,
                    map: 5,
                    opcode,
                    pp: 2,
                    w,
                    bytes: 16,
                    reg: if to { 0 } else { 1 },
                    vvvv: if to { 0 } else { 2 },
                    rm: if to { 3 } else { 0 },
                    opmask: 0,
                    zeroing: false,
                    imm: 0,
                };
                // X0 holds the MXCSR to run under, X1 receives the resulting
                // one, X2 holds the caller's, X4 RAX before and X5 RAX after.
                let mut code = at_rdi(&[0x0f, 0xae], 2, x(0));
                code.extend(at_rdi(&[0x48, 0x8b], 0, x(4)));
                for register in 1..4 {
                    code.extend(at_rdi(
                        &[0xc5, 0xfe, 0x6f],
                        register,
                        vector(usize::from(register)),
                    ));
                }
                code.extend(instruction.encode());
                code.extend(at_rdi(&[0x48, 0x89], 0, x(5)));
                code.extend(at_rdi(&[0xc5, 0xfe, 0x7f], 1, vector(1)));
                code.extend(at_rdi(&[0x0f, 0xae], 3, x(1)));
                code.extend(at_rdi(&[0x0f, 0xae], 2, x(2)));
                code.extend([0xc5, 0xf8, 0x77, 0xc3]); // vzeroupper; ret
                let native = Native::new(&code);
                let form = General::of(instruction).unwrap();
                for sample in 0..256 {
                    let mut state = GuestState::default();
                    for limb in state.v.as_flattened_mut() {
                        *limb = (0..4).fold(0, |limb, lane| {
                            limb | u64::from(half(next())) << (16 * lane)
                        });
                    }
                    let mut caller = 0u32;
                    // SAFETY: STMXCSR writes four bytes to the local.
                    unsafe { core::arch::asm!("stmxcsr [{}]", in(reg) &mut caller) };
                    let rounding = (sample % 4) << 13;
                    let integer = next() >> (next() % 64);
                    state.x[0] = 0x1f80 | rounding;
                    state.x[2] = u64::from(caller);
                    state.x[4] = if next() & 1 == 0 {
                        integer
                    } else {
                        integer.wrapping_neg()
                    };
                    let mut control = Control {
                        mxcsr: 0x1f80 | rounding as u32,
                        opmask: 0,
                    };
                    let value = state.x[4];
                    let source = half_word(&ymm(&state, 3), 0);
                    let model =
                        general_result(form, instruction, value, source, &mut control).unwrap();
                    let vvvv = ymm(&state, 2);
                    native.run(&mut state, &mut ExecutionContext::default());
                    let compared = !0x3f | half::INVALID;
                    if to {
                        // Intel SDE 10.13.1 sign-extends a dword VCVTSH2SI into RAX.
                        let result = if w {
                            state.x[5]
                        } else {
                            state.x[5] as u32 as u64
                        };
                        assert_eq!(result, model, "{instruction:x?} sample={sample}");
                    } else {
                        let mut expected = vvvv[..16].to_vec();
                        expected[..2].copy_from_slice(&(model as u16).to_le_bytes());
                        assert_eq!(ymm(&state, 1)[..16], expected, "{instruction:x?}");
                    }
                    assert_eq!(
                        state.x[1] as u32 & compared,
                        control.mxcsr & compared,
                        "{instruction:x?} sample={sample}"
                    );
                    checked += 1;
                }
            }
        }
        std::println!("GENERAL_MODELS_CHECKED={checked}");
    }

    #[test]
    fn instruction_fields_round_trip_through_gnu_encodings() {
        // GNU as 2.42.
        for (bytes, map, opcode, pp, width, [reg, vvvv, rm], opmask, zeroing) in [
            (
                &[0xc4, 0xe2, 0x7b, 0x50, 0xd1][..],
                2,
                0x50,
                3,
                16,
                [2, 0, 1],
                0,
                false,
            ),
            (
                &[0xc4, 0x42, 0x2c, 0xd2, 0xcb],
                2,
                0xd2,
                0,
                32,
                [9, 10, 11],
                0,
                false,
            ),
            (
                &[0x62, 0xf5, 0x6c, 0x89, 0x58, 0xcb],
                5,
                0x58,
                0,
                16,
                [1, 2, 3],
                1,
                true,
            ),
            (
                &[0x62, 0xf5, 0x6c, 0x29, 0x59, 0xcb],
                5,
                0x59,
                0,
                32,
                [1, 2, 3],
                1,
                false,
            ),
            (
                &[0x62, 0xa5, 0x6c, 0xa2, 0x5e, 0xcb],
                5,
                0x5e,
                0,
                32,
                [17, 18, 19],
                2,
                true,
            ),
            (
                &[0x62, 0x55, 0x2c, 0x0f, 0x5e, 0xcb],
                5,
                0x5e,
                0,
                16,
                [9, 10, 11],
                7,
                false,
            ),
        ] {
            let instruction = Instruction {
                evex: bytes[0] == 0x62,
                map,
                opcode,
                pp,
                w: false,
                bytes: width,
                reg,
                vvvv,
                rm,
                opmask,
                zeroing,
                imm: 0,
            };
            assert_eq!(
                Instruction::decode(bytes),
                Some(instruction),
                "{bytes:02x?}"
            );
            assert_eq!(instruction.encode(), bytes);
        }
    }

    #[test]
    fn half_rounding_status_and_special_values() {
        use half::*;
        let add = |left, right, rounding| arithmetic(Operation::Add, left, right, rounding);
        // 1 + 2^-11 is a tie: to even down, toward +infinity up.
        assert_eq!(add(0x3c00, 0x1000, 0), (0x3c00, INEXACT));
        assert_eq!(add(0x3c00, 0x1000, 2), (0x3c01, INEXACT));
        // 65504 + 16 ties to 65536, overflowing to infinity, but is 65504
        // toward zero; 65504 + 32 overflows to 65504 toward zero.
        assert_eq!(add(0x7bff, 0x4c00, 0), (0x7c00, OVERFLOW | INEXACT));
        assert_eq!(add(0x7bff, 0x4c00, 3), (0x7bff, INEXACT));
        assert_eq!(add(0x7bff, 0x5000, 3), (0x7bff, OVERFLOW | INEXACT));
        // x - x is -0 only when rounding down.
        assert_eq!(
            arithmetic(Operation::Subtract, 0x3c00, 0x3c00, 1),
            (0x8000, 0)
        );
        assert_eq!(arithmetic(Operation::Subtract, 0x3c00, 0x3c00, 0), (0, 0));
        // The first NaN operand returns quiet; an SNaN anywhere is invalid.
        assert_eq!(add(0x7d00, 0xfe01, 0), (0x7f00, INVALID));
        assert_eq!(add(0x3c00, 0xfd01, 0), (0xff01, INVALID));
        assert_eq!(
            arithmetic(Operation::Multiply, 0, 0x7c00, 0),
            (0xfe00, INVALID)
        );
        assert_eq!(
            arithmetic(Operation::Divide, 0xbc00, 0, 0),
            (0xfc00, DIVIDE_BY_ZERO)
        );
        // 2^-14 * (1 - 2^-11) has 11 significant bits below 2^-14, so it is
        // tiny after rounding to an unbounded exponent and underflows even
        // where it rounds to 2^-14.
        assert_eq!(
            arithmetic(Operation::Multiply, 0x0400, 0x3bff, 0),
            (0x0400, UNDERFLOW | INEXACT)
        );
        assert_eq!(
            arithmetic(Operation::Multiply, 0x0400, 0x3bff, 3),
            (0x03ff, UNDERFLOW | INEXACT)
        );
        // Denormal operands raise nothing; an exact denormal result does not
        // underflow.
        assert_eq!(add(0x0001, 0x0001, 0), (0x0002, 0));
        // (1 + 2^-10)^2 - 1 keeps the tie at 2^-20 that a rounded product loses.
        assert_eq!(
            fused([0x3c01, 0xbc00, 0x3c01], [false, false], 0),
            (0x1800, INEXACT)
        );
        assert_eq!(
            fused([0x3c01, 0xbc00, 0x3c01], [false, false], 2),
            (0x1801, INEXACT)
        );
        // The smallest product and the largest addend still round exactly:
        // 65504 + 2^-48 is inexact, and rounds up only toward +infinity.
        assert_eq!(fused([1, 0x7bff, 1], [false, false], 0), (0x7bff, INEXACT));
        assert_eq!(
            fused([1, 0x7bff, 1], [false, false], 2),
            (0x7c00, OVERFLOW | INEXACT)
        );
        // x * 1 - x is -0 only when rounding down; zeros of one sign keep it.
        assert_eq!(
            fused([0x3c00, 0x3c00, 0x3c00], [false, true], 1),
            (0x8000, 0)
        );
        assert_eq!(
            fused([0x8000, 0x8000, 0x3c00], [false, false], 0),
            (0x8000, 0)
        );
        // Conversions truncate or round by MXCSR; out of range is indefinite.
        assert_eq!(to_integer(0x4110, [true, true], 16, 0), (2, INEXACT));
        assert_eq!(to_integer(0xc110, [true, false], 16, 1), (0xfffd, INEXACT));
        assert_eq!(to_integer(0x7bff, [true, true], 16, 0), (0x8000, INVALID));
        assert_eq!(to_integer(0x7bff, [false, true], 16, 0), (0xffe0, 0));
        assert_eq!(
            to_integer(0xbc00, [false, true], 32, 0),
            (u32::MAX as u64, INVALID)
        );
        assert_eq!(to_integer(0xb800, [false, true], 64, 0), (0, INEXACT));
        assert_eq!(from_integer(3, 0), (0x4200, 0));
        assert_eq!(from_integer(2049, 0), (0x6800, INEXACT));
        assert_eq!(from_integer(65535, 0), (0x7c00, OVERFLOW | INEXACT));
        assert_eq!(from_double(1.5, 0), (0x3e00, 0));
        assert_eq!(from_double(1e300, 3), (0x7bff, OVERFLOW | INEXACT));
        // Infinity times zero is invalid, as is infinities of opposite sign.
        assert_eq!(fused([0, 0, 0x7c00], [false, false], 0), (0xfe00, INVALID));
        // A NaN addend takes precedence and retains its payload. Only a
        // signaling NaN makes this case invalid.
        assert_eq!(fused([0, 0x7ff6, 0x7c00], [false, false], 0), (0x7ff6, 0));
        assert_eq!(
            fused([0x7c00, 0x7df6, 0], [false, false], 0),
            (0x7ff6, INVALID)
        );
        assert_eq!(
            fused([0x7c00, 0x7c00, 0x3c00], [false, true], 0),
            (0xfe00, INVALID)
        );
    }
}
