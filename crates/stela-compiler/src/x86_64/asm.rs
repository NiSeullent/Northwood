//! A typed x86-64 encoder (Intel SDM Vol. 2, chapter 2). Operands are typed
//! registers and memory; operand-size prefixes, REX, VEX, ModRM, SIB and
//! displacements are derived from them, and [`Asm`] counts every instruction
//! it emits.
//!
//! Emitters take a [`Code`], which carries the instruction count across
//! every [`Asm`] built over it.
//!
use alloc::vec::Vec;

#[cfg(test)]
pub(crate) mod apx;
pub(crate) mod control;
mod integer;
pub(crate) mod memory;
mod sse;
pub(crate) mod vector_extensions;
pub(crate) use integer::BitTest;
pub(crate) use sse::{FloatLanes, FloatOp};

mod code;
mod operands;
mod prefixes;
use code::EmissionSite;
pub(crate) use code::{Checkpoint, Code, Span};
pub(crate) use operands::*;

/// An encoder that appends to `code` and counts the instructions it emits,
/// into the [`Code`] it was built over as well.
pub(crate) struct Asm<'a> {
    code: &'a mut Vec<u8>,
    instructions: usize,
    shared: Option<&'a mut usize>,
    sites: Option<&'a mut Vec<EmissionSite>>,
    pinned: bool,
    locals: Option<super::pinned::LocalRegisters>,
    /// The locals a fault recovers from their hosts ([`super::pinned::Span`]).
    dirty: super::pinned::Dirty,
    linked_block: Option<u64>,
    /// Host XMM registers that hold guest vectors
    /// ([`super::vector_cache`]): no instruction emitted here writes one.
    guarded: u32,
    /// Whether a guest vector's only current value is in a host register,
    /// which no fault site may then be.
    vectors_dirty: bool,
}

impl<'a> Asm<'a> {
    /// Whether this encoder appends to pinned code.
    pub(crate) const fn is_pinned(&self) -> bool {
        self.pinned
    }
    /// The block whose exits keep a link slot, if this code's do.
    pub(crate) const fn linked_block(&self) -> Option<u64> {
        self.linked_block
    }
    /// `emit` given the frame's NZCV word from byte `offset`, which pinned
    /// code reaches through [`super::flag_word`] alone.
    pub(crate) fn with_flags(&mut self, offset: u32, emit: impl FnOnce(&mut Self, Mem)) {
        const NZCV: u32 = core::mem::offset_of!(crate::GuestState, nzcv) as u32;
        let pinned = core::mem::replace(&mut self.pinned, false);
        emit(self, super::frame(NZCV + offset));
        self.pinned = pinned;
    }
    pub(crate) fn new(code: &'a mut Vec<u8>) -> Self {
        Self {
            code,
            instructions: 0,
            shared: None,
            sites: None,
            pinned: false,
            locals: None,
            dirty: Default::default(),
            linked_block: None,
            guarded: 0,
            vectors_dirty: false,
        }
    }
    /// [`Self::new`] for code that keeps guest registers in host registers
    /// ([`super::pinned`]).
    pub(crate) fn pinned(code: &'a mut Vec<u8>) -> Self {
        Self {
            pinned: true,
            ..Self::new(code)
        }
    }
    /// The storage of `register`, which XZR lacks: its host register in
    /// pinned code, else its frame slot.
    pub(crate) fn guest(&self, register: crate::aarch64::Register) -> Option<Rm> {
        let slot = super::offset(register)? as i32;
        Some(if !self.pinned {
            Mem::base_disp32(Gpr::Rdi, slot).into()
        } else if let Some(host) = super::pinned::host_or_local(register, self.locals) {
            host.into()
        } else {
            Mem::base(Gpr::Rdi, slot).into()
        })
    }
    /// Mark exactly one guest memory operand, including its prefixes and
    /// immediate bytes. Frame/staging/address instructions must remain outside.
    pub(crate) fn guest_access(&mut self, emit: impl FnOnce(&mut Self)) {
        self.fault_site(crate::block::RecoverySite::guest_access(), None, emit);
    }
    /// [`Self::guest_access`] of an instruction that needs the host register
    /// of SP, which `stack` holds meanwhile.
    pub(crate) fn guest_access_moving_stack(&mut self, stack: Gpr, emit: impl FnOnce(&mut Self)) {
        let site = crate::block::RecoverySite::guest_access();
        self.fault_site(site, Some(stack.number()), emit);
    }
    /// Intentional alignment/range failure, rather than an unsupported opcode.
    pub(crate) fn guest_check(&mut self) {
        self.fault_site(crate::block::RecoverySite::guest_check(), None, |asm| {
            asm.ud2()
        });
    }
    /// [`Self::guest_check`] out of line, for the instruction at `source` in
    /// the block's source map.
    pub(crate) fn guest_check_of(&mut self, source: usize) {
        self.guest_check();
        if let Some(site) = self.sites.as_deref_mut().and_then(|sites| sites.last_mut()) {
            site.source = Some(source);
        }
    }
    /// Emit the unchanged translated descriptor and mark ONLY its first byte as
    /// a deliberate helper-return token. The CALL opcode and descriptor tail
    /// remain internal; neither is a hardware guest-access/check instruction.
    pub(crate) fn translated_helper_return(&mut self, bytes: u8, flags: u8) {
        let start = self.len();
        self.data(&[bytes, flags]);
        if let Some(sites) = self.sites.as_deref_mut() {
            assert!(
                sites.last().is_none_or(|previous| previous.end <= start),
                "helper sites do not overlap"
            );
            sites.push(EmissionSite {
                start,
                end: start + 1,
                site: crate::block::RecoverySite::translated_helper(bytes, flags),
                stack: None,
                source: None,
                locals: self.dirty,
            });
        }
    }
    /// Refuse to write `xmm` where it holds a guest vector
    /// ([`super::vector_cache`]): a recipe names it only as a source.
    pub(crate) fn guard_vector(&self, xmm: u8) {
        assert!(
            self.guarded & 1 << xmm == 0,
            "xmm{xmm} holds a guest vector of the block"
        );
    }
    fn fault_site(
        &mut self,
        site: crate::block::RecoverySite,
        stack: Option<u8>,
        emit: impl FnOnce(&mut Self),
    ) {
        assert!(
            !self.vectors_dirty,
            "a fault site with V registers only in XMM"
        );
        let start = self.len();
        let instructions = self.instructions;
        emit(self);
        assert_eq!(
            self.instructions,
            instructions + 1,
            "one exact faulting instruction"
        );
        let end = self.len();
        if let Some(sites) = self.sites.as_deref_mut() {
            assert!(
                sites.last().is_none_or(|previous| previous.end <= start),
                "fault sites do not nest"
            );
            sites.push(EmissionSite {
                start,
                end,
                site,
                stack,
                source: None,
                locals: self.dirty,
            });
        }
    }
    fn count(&mut self) {
        self.instructions += 1;
        if let Some(shared) = self.shared.as_deref_mut() {
            *shared += 1;
        }
    }
    /// The number of instructions emitted through this encoder.
    #[cfg(test)]
    pub(crate) const fn instructions(&self) -> usize {
        self.instructions
    }
    /// The code length, the offset of the next instruction.
    pub(crate) fn len(&self) -> usize {
        self.code.len()
    }
    /// A prefix byte (LOCK, REP, a segment override) of the next
    /// instruction, which counts it.
    pub(crate) fn prefix(&mut self, byte: u8) {
        self.code.push(byte);
    }
    /// `lock` for the next instruction.
    pub(crate) fn lock(&mut self) {
        self.prefix(0xf0);
    }
    /// Data or immediates that belong to the instruction just emitted.
    pub(crate) fn data(&mut self, bytes: &[u8]) {
        self.code.extend(bytes);
    }

    /// A legacy SSE instruction on XMM registers: the mandatory `prefix`,
    /// REX when a register needs it, `opcode` and ModRM.
    pub(crate) fn sse(&mut self, prefix: Option<u8>, opcode: &[u8], xmm: u8, rm: impl Into<Rm>) {
        assert!(xmm < 16, "legacy SSE cannot name extended vector registers");
        self.guard_vector(xmm);
        let rm = rm.into();
        assert!(!matches!(rm, Rm::Xmm(register) if register >= 16));
        self.legacy(Size::Dword, prefix, opcode, xmm, false, rm);
    }
    /// [`Self::sse`] of a form whose ModRM.reg register is a source (a
    /// store), which may name a register holding a guest vector.
    pub(crate) fn sse_reading(
        &mut self,
        prefix: Option<u8>,
        opcode: &[u8],
        xmm: u8,
        rm: impl Into<Rm>,
    ) {
        let guarded = core::mem::replace(&mut self.guarded, 0);
        self.sse(prefix, opcode, xmm, rm);
        self.guarded = guarded;
    }
    /// [`Self::sse`] with an 8-bit immediate.
    pub(crate) fn sse_imm(
        &mut self,
        prefix: Option<u8>,
        opcode: &[u8],
        xmm: u8,
        rm: impl Into<Rm>,
        immediate: u8,
    ) {
        self.sse(prefix, opcode, xmm, rm);
        self.code.push(immediate);
    }
    /// An opcode that names its register in its low three bits (`+r`).
    pub(crate) fn opcode_reg(&mut self, size: Size, opcode: u8, register: Gpr) {
        if size == Size::Word {
            self.code.push(0x66);
        }
        self.opcode_prefix(size, false, &[opcode], 0, register.into());
        self.code.push(opcode | register.low());
        self.count();
    }
    /// An instruction of fixed bytes and no operands (`ret`, `ud2`, `mfence`).
    pub(crate) fn fixed(&mut self, bytes: &[u8]) {
        self.code.extend(bytes);
        self.count();
    }

    // Stack and control transfer.

    /// `push r64`.
    pub(crate) fn push(&mut self, register: Gpr) {
        self.opcode_reg(Size::Dword, 0x50, register);
    }
    /// `push rm64` (FF /6). Only the old probes' differential test uses it.
    #[cfg(all(test, target_arch = "x86_64", target_os = "linux"))]
    pub(crate) fn push_rm(&mut self, rm: impl Into<Rm>) {
        self.legacy(Size::Dword, None, &[0xff], 6, false, rm.into());
    }
    /// `pop r64`.
    pub(crate) fn pop(&mut self, register: Gpr) {
        self.opcode_reg(Size::Dword, 0x58, register);
    }
    /// `ret`.
    pub(crate) fn ret(&mut self) {
        self.fixed(&[0xc3]);
    }
    /// `call rm64`.
    pub(crate) fn call_rm(&mut self, rm: impl Into<Rm>) {
        self.legacy(Size::Dword, None, &[0xff], 2, false, rm.into());
    }
    /// `jmp rm64`.
    pub(crate) fn jmp_rm(&mut self, rm: impl Into<Rm>) {
        self.legacy(Size::Dword, None, &[0xff], 4, false, rm.into());
    }
    /// `call rel32`, resolved later by [`Self::bind`] or [`resolve`].
    pub(crate) fn call(&mut self) -> Patch {
        self.code.push(0xe8);
        self.patch()
    }

    // Moves.

    /// `mov rm, src`.
    pub(crate) fn mov_store(&mut self, size: Size, rm: impl Into<Rm>, source: Gpr) {
        let opcode = if size == Size::Byte { 0x88 } else { 0x89 };
        self.legacy(size, None, &[opcode], source.number(), true, rm.into());
    }
    /// `mov dst, rm`.
    pub(crate) fn mov_load(&mut self, size: Size, destination: Gpr, rm: impl Into<Rm>) {
        let opcode = if size == Size::Byte { 0x8a } else { 0x8b };
        self.legacy(size, None, &[opcode], destination.number(), true, rm.into());
    }
    /// `mov dst, src` in the 89 direction (`mov rm, reg`).
    pub(crate) fn mov(&mut self, size: Size, destination: Gpr, source: Gpr) {
        self.mov_store(size, destination, source);
    }
    /// `mov r32, imm32` (B8+r), which zero-extends to 64 bits.
    pub(crate) fn mov_imm32(&mut self, destination: Gpr, value: u32) {
        self.opcode_reg(Size::Dword, 0xb8, destination);
        self.data(&value.to_le_bytes());
    }
    /// Load the address of `place` in the running PE's data.
    pub(crate) fn place(&mut self, destination: Gpr, place: crate::Place) {
        let anchor = core::mem::offset_of!(crate::GuestState, anchors) + place.anchor as usize * 8;
        self.mov_load(Size::Qword, destination, Mem::base(Gpr::Rdi, anchor as i32));
        if place.offset != 0 {
            self.lea(
                Size::Qword,
                destination,
                Mem::base(destination, place.offset as i32),
            );
        }
    }

    /// `movabs r64, imm64` (REX.W B8+r, or REX2 for R16..R31).
    /// `mov destination16,value`, which keeps the bits above.
    pub(crate) fn mov_imm16(&mut self, destination: Gpr, value: u16) {
        self.opcode_reg(Size::Word, 0xb8, destination);
        self.data(&value.to_le_bytes());
    }
    pub(crate) fn mov_imm64(&mut self, destination: Gpr, value: u64) {
        self.opcode_reg(Size::Qword, 0xb8, destination);
        self.data(&value.to_le_bytes());
    }
    /// `mov rm, imm32` (C7 /0), sign-extended for a quadword.
    pub(crate) fn mov_rm_imm32(&mut self, size: Size, rm: impl Into<Rm>, value: i32) {
        debug_assert!(size != Size::Byte && size != Size::Word);
        self.legacy(size, None, &[0xc7], 0, false, rm.into());
        self.code.extend(value.to_le_bytes());
    }
    /// `movzx dst, rm8/rm16`.
    pub(crate) fn movzx(&mut self, size: Size, destination: Gpr, from: Size, rm: impl Into<Rm>) {
        let opcode = if from == Size::Byte { 0xb6 } else { 0xb7 };
        let rm = rm.into();
        let byte_rex = from == Size::Byte && rm.needs_byte_rex();
        self.extended(
            size,
            byte_rex,
            None,
            &[0x0f, opcode],
            destination.number(),
            rm,
        );
    }
    /// `movsx dst, rm8/rm16`.
    pub(crate) fn movsx(&mut self, size: Size, destination: Gpr, from: Size, rm: impl Into<Rm>) {
        let opcode = if from == Size::Byte { 0xbe } else { 0xbf };
        let rm = rm.into();
        let byte_rex = from == Size::Byte && rm.needs_byte_rex();
        self.extended(
            size,
            byte_rex,
            None,
            &[0x0f, opcode],
            destination.number(),
            rm,
        );
    }
    /// `movsxd r64, rm32`.
    pub(crate) fn movsxd(&mut self, destination: Gpr, rm: impl Into<Rm>) {
        self.legacy(
            Size::Qword,
            None,
            &[0x63],
            destination.number(),
            false,
            rm.into(),
        );
    }
    /// `lea dst, [mem]`.
    pub(crate) fn lea(&mut self, size: Size, destination: Gpr, memory: Mem) {
        self.legacy(
            size,
            None,
            &[0x8d],
            destination.number(),
            false,
            Rm::Mem(memory),
        );
    }

    /// A legacy instruction whose operand size is `size` but whose REX also
    /// carries `byte_rex` for a byte-sized r/m source (MOVZX, MOVSX).
    fn extended(
        &mut self,
        size: Size,
        byte_rex: bool,
        prefix: Option<u8>,
        opcode: &[u8],
        reg: u8,
        rm: Rm,
    ) {
        self.legacy_extended(size, prefix, opcode, reg, byte_rex, rm);
    }

    // Arithmetic.

    /// `op rm, src` (the 01-direction ALU forms).
    pub(crate) fn alu(&mut self, op: Alu, size: Size, rm: impl Into<Rm>, source: Gpr) {
        let opcode = (op as u8) << 3 | u8::from(size != Size::Byte);
        self.legacy(size, None, &[opcode], source.number(), true, rm.into());
    }
    /// `op dst, rm` (the 03-direction ALU forms).
    pub(crate) fn alu_load(&mut self, op: Alu, size: Size, destination: Gpr, rm: impl Into<Rm>) {
        let opcode = (op as u8) << 3 | 2 | u8::from(size != Size::Byte);
        self.legacy(size, None, &[opcode], destination.number(), true, rm.into());
    }
    /// `op rm, imm8` sign-extended (83 /digit).
    pub(crate) fn alu_imm8(&mut self, op: Alu, size: Size, rm: impl Into<Rm>, value: i8) {
        debug_assert!(size != Size::Byte);
        self.legacy(size, None, &[0x83], op as u8, false, rm.into());
        self.code.push(value as u8);
    }
    /// `op rm, imm32` (81 /digit), sign-extended for a quadword.
    pub(crate) fn alu_imm32(&mut self, op: Alu, size: Size, rm: impl Into<Rm>, value: i32) {
        debug_assert!(size == Size::Dword || size == Size::Qword);
        self.legacy(size, None, &[0x81], op as u8, false, rm.into());
        self.code.extend(value.to_le_bytes());
    }
    /// `op eax/rax, imm32`, the accumulator short form.
    pub(crate) fn alu_accumulator_imm32(&mut self, op: Alu, size: Size, value: i32) {
        debug_assert!(size == Size::Dword || size == Size::Qword);
        if size == Size::Qword {
            self.code.push(0x48);
        }
        self.code.push((op as u8) << 3 | 5);
        self.code.extend(value.to_le_bytes());
        self.count();
    }
    /// `op rm, imm`: the sign-extended imm8 form when it fits, else imm32.
    pub(crate) fn alu_imm(&mut self, op: Alu, size: Size, rm: impl Into<Rm>, value: i32) {
        match i8::try_from(value) {
            Ok(short) => self.alu_imm8(op, size, rm, short),
            Err(_) => self.alu_imm32(op, size, rm, value),
        }
    }
    /// `test rm, src`.
    pub(crate) fn test(&mut self, size: Size, rm: impl Into<Rm>, source: Gpr) {
        let opcode = if size == Size::Byte { 0x84 } else { 0x85 };
        self.legacy(size, None, &[opcode], source.number(), true, rm.into());
    }
    /// Group 3 `op rm`: NOT, NEG, MUL, IMUL, DIV, IDIV.
    pub(crate) fn unary(&mut self, op: Unary, size: Size, rm: impl Into<Rm>) {
        let opcode = if size == Size::Byte { 0xf6 } else { 0xf7 };
        self.legacy(size, None, &[opcode], op as u8, false, rm.into());
    }
    /// `imul dst, rm`.
    pub(crate) fn imul(&mut self, size: Size, destination: Gpr, rm: impl Into<Rm>) {
        self.legacy(
            size,
            None,
            &[0x0f, 0xaf],
            destination.number(),
            false,
            rm.into(),
        );
    }
    /// `cdq` or `cqo`: sign-extend EAX/RAX into EDX/RDX.
    pub(crate) fn sign_extend_accumulator(&mut self, size: Size) {
        if size == Size::Qword {
            self.code.push(0x48);
        }
        self.code.push(0x99);
        self.count();
    }
    /// A group 2 shift or rotate by CL.
    pub(crate) fn shift_cl(&mut self, op: ShiftOp, size: Size, rm: impl Into<Rm>) {
        let opcode = if size == Size::Byte { 0xd2 } else { 0xd3 };
        self.legacy(size, None, &[opcode], op as u8, false, rm.into());
    }
    /// A group 2 shift or rotate by one (D1 /digit).
    pub(crate) fn shift_one(&mut self, op: ShiftOp, size: Size, rm: impl Into<Rm>) {
        let opcode = if size == Size::Byte { 0xd0 } else { 0xd1 };
        self.legacy(size, None, &[opcode], op as u8, false, rm.into());
    }
    /// `imul dst, rm, imm32`.
    pub(crate) fn imul_imm32(
        &mut self,
        size: Size,
        destination: Gpr,
        rm: impl Into<Rm>,
        value: i32,
    ) {
        self.legacy(size, None, &[0x69], destination.number(), false, rm.into());
        self.code.extend(value.to_le_bytes());
    }
    /// `nop`.
    pub(crate) fn nop(&mut self) {
        self.code.push(0x90);
        self.count();
    }
    /// A group 2 shift or rotate by an immediate (C1 /digit ib).
    pub(crate) fn shift_imm(&mut self, op: ShiftOp, size: Size, rm: impl Into<Rm>, amount: u8) {
        let opcode = if size == Size::Byte { 0xc0 } else { 0xc1 };
        self.legacy(size, None, &[opcode], op as u8, false, rm.into());
        self.code.push(amount);
    }
    /// `shrd rm, src, imm8`.
    pub(crate) fn shrd_imm(&mut self, size: Size, rm: impl Into<Rm>, source: Gpr, amount: u8) {
        self.legacy(size, None, &[0x0f, 0xac], source.number(), false, rm.into());
        self.code.push(amount);
    }

    // Bit operations.

    /// `bsr dst, rm`.
    pub(crate) fn bsr(&mut self, size: Size, destination: Gpr, rm: impl Into<Rm>) {
        self.legacy(
            size,
            None,
            &[0x0f, 0xbd],
            destination.number(),
            false,
            rm.into(),
        );
    }
    /// `bsf dst, rm`.
    pub(crate) fn bsf(&mut self, size: Size, destination: Gpr, rm: impl Into<Rm>) {
        self.legacy(
            size,
            None,
            &[0x0f, 0xbc],
            destination.number(),
            false,
            rm.into(),
        );
    }
    /// `lzcnt dst, rm` (LZCNT; ABM on AMD).
    pub(crate) fn lzcnt(&mut self, size: Size, destination: Gpr, rm: impl Into<Rm>) {
        self.legacy(
            size,
            Some(0xf3),
            &[0x0f, 0xbd],
            destination.number(),
            false,
            rm.into(),
        );
    }
    /// `tzcnt dst, rm` (BMI1).
    pub(crate) fn tzcnt(&mut self, size: Size, destination: Gpr, rm: impl Into<Rm>) {
        self.legacy(
            size,
            Some(0xf3),
            &[0x0f, 0xbc],
            destination.number(),
            false,
            rm.into(),
        );
    }
    /// `popcnt dst, rm`.
    pub(crate) fn popcnt(&mut self, size: Size, destination: Gpr, rm: impl Into<Rm>) {
        self.legacy(
            size,
            Some(0xf3),
            &[0x0f, 0xb8],
            destination.number(),
            false,
            rm.into(),
        );
    }
    /// `bswap reg`.
    pub(crate) fn bswap(&mut self, size: Size, register: Gpr) {
        debug_assert!(size == Size::Dword || size == Size::Qword);
        let opcode = [0x0f, 0xc8 | register.low()];
        let skip = self.opcode_prefix(size, false, &opcode, 0, register.into());
        self.code.extend(&opcode[skip..]);
        self.count();
    }
    /// `cmovcc dst, rm`.
    pub(crate) fn cmov(&mut self, cond: Cond, size: Size, destination: Gpr, rm: impl Into<Rm>) {
        self.legacy(
            size,
            None,
            &[0x0f, 0x40 | cond as u8],
            destination.number(),
            false,
            rm.into(),
        );
    }
    /// `setcc rm8`.
    pub(crate) fn setcc(&mut self, cond: Cond, rm: impl Into<Rm>) {
        self.legacy(
            Size::Byte,
            None,
            &[0x0f, 0x90 | cond as u8],
            0,
            false,
            rm.into(),
        );
    }

    // SSE moves and GFNI.

    /// `movq xmm, r64`.
    pub(crate) fn movq_to_xmm(&mut self, xmm: u8, source: Gpr) {
        self.legacy(
            Size::Qword,
            Some(0x66),
            &[0x0f, 0x6e],
            xmm,
            false,
            Rm::Reg(source),
        );
    }
    /// `movq r64, xmm`.
    pub(crate) fn movq_from_xmm(&mut self, destination: Gpr, xmm: u8) {
        self.legacy(
            Size::Qword,
            Some(0x66),
            &[0x0f, 0x7e],
            xmm,
            false,
            Rm::Reg(destination),
        );
    }
    /// `gf2p8affineqb xmm, xmm/m128, imm8`: each byte of `xmm` becomes the
    /// GF(2) affine transform by the matching qword's 8x8 bit matrix in
    /// `matrix`, XOR `constant` (GFNI).
    pub(crate) fn gf2p8affineqb(&mut self, xmm: u8, matrix: Rm, constant: u8) {
        self.legacy(
            Size::Dword,
            Some(0x66),
            &[0x0f, 0x3a, 0xce],
            xmm,
            false,
            matrix,
        );
        self.code.push(constant);
    }

    /// GFNI inverse in the AES field followed by the supplied affine matrix.
    pub(crate) fn gf2p8affineinvqb(&mut self, xmm: u8, matrix: Rm, constant: u8) {
        self.legacy(
            Size::Dword,
            Some(0x66),
            &[0x0f, 0x3a, 0xcf],
            xmm,
            false,
            matrix,
        );
        self.code.push(constant);
    }

    // BMI2 (VEX-encoded general-purpose forms).

    /// `pext dst, src, mask`: pack the selected bits without affecting flags.
    pub(crate) fn pext(&mut self, destination: Gpr, source: Gpr, mask: Gpr) {
        self.vex(
            Vex {
                map: Map::M0f38,
                pp: Pp::Pf3,
                w: false,
                l: false,
            },
            0xf5,
            destination.number(),
            source.number(),
            mask.into(),
        );
    }

    /// `shlx`, `shrx` or `sarx dst, rm, count`: a shift by the count register
    /// modulo the operand width, without flags (BMI2).
    pub(crate) fn shift_x(
        &mut self,
        op: ShiftOp,
        size: Size,
        destination: Gpr,
        rm: impl Into<Rm>,
        count: Gpr,
    ) {
        let pp = match op {
            ShiftOp::Shl => Pp::P66,
            ShiftOp::Shr => Pp::Pf2,
            ShiftOp::Sar => Pp::Pf3,
            _ => unreachable!("SHLX, SHRX and SARX only"),
        };
        self.vex(
            Vex {
                map: Map::M0f38,
                pp,
                w: size == Size::Qword,
                l: false,
            },
            0xf7,
            destination.number(),
            count.number(),
            rm.into(),
        );
    }
    /// `rorx dst, rm, imm8`: rotate right without flags (BMI2).
    pub(crate) fn rorx(&mut self, size: Size, destination: Gpr, rm: impl Into<Rm>, amount: u8) {
        self.vex(
            Vex {
                map: Map::M0f3a,
                pp: Pp::Pf2,
                w: size == Size::Qword,
                l: false,
            },
            0xf0,
            destination.number(),
            0,
            rm.into(),
        );
        self.code.push(amount);
    }

    // Control flow.

    /// `jmp rel32`, resolved later by [`Self::bind`].
    pub(crate) fn jmp(&mut self) -> Patch {
        self.code.push(0xe9);
        self.patch()
    }
    /// `jcc rel32`, resolved later by [`Self::bind`].
    pub(crate) fn jcc(&mut self, cond: Cond) -> Patch {
        self.code.extend([0x0f, 0x80 | cond as u8]);
        self.patch()
    }
    /// `jmp rel8`, resolved later by [`Self::bind`].
    pub(crate) fn jmp8(&mut self) -> Patch {
        self.code.push(0xeb);
        self.short_patch()
    }
    /// `jcc rel8`, resolved later by [`Self::bind`].
    pub(crate) fn jcc8(&mut self, cond: Cond) -> Patch {
        self.code.push(0x70 | cond as u8);
        self.short_patch()
    }
    fn patch(&mut self) -> Patch {
        let at = self.code.len();
        self.code.extend([0; 4]);
        self.count();
        Patch { at, short: false }
    }
    fn short_patch(&mut self) -> Patch {
        let at = self.code.len();
        self.code.push(0);
        self.count();
        Patch { at, short: true }
    }
    /// Point `patch` at the current position.
    pub(crate) fn bind(&mut self, patch: Patch) {
        let target = self.code.len();
        resolve_bytes(self.code, patch, target);
    }
    /// Point `patch` at the earlier position `target` ([`Self::len`]).
    pub(crate) fn bind_back(&mut self, patch: Patch, target: usize) {
        resolve_bytes(self.code, patch, target);
    }
}

/// Point `patch` in `code` at `target`.
pub(crate) fn resolve(code: &mut Code, patch: Patch, target: usize) {
    resolve_bytes(&mut code.bytes, patch, target);
}

fn resolve_bytes(code: &mut [u8], patch: Patch, target: usize) {
    let distance = target as i64 - patch.end() as i64;
    if patch.short {
        code[patch.at] = i8::try_from(distance).expect("rel8 branch in range") as u8;
    } else {
        let distance = i32::try_from(distance).expect("rel32 branch in range");
        code[patch.at..patch.at + 4].copy_from_slice(&distance.to_le_bytes());
    }
}

#[cfg(test)]
mod tests {
    mod span;
    use super::*;
    use alloc::vec;

    fn bytes(build: impl FnOnce(&mut Asm)) -> (Vec<u8>, usize) {
        let mut code = vec![];
        let mut asm = Asm::new(&mut code);
        build(&mut asm);
        let count = asm.instructions();
        (code, count)
    }

    #[test]
    fn integer_forms_match_the_sdm_encodings() {
        use Gpr::*;
        use Size::*;
        for (expected, build) in [
            (
                &[0x89, 0xc1][..],
                (|a: &mut Asm| a.mov(Dword, Rcx, Rax)) as fn(&mut Asm),
            ),
            (&[0x48, 0x89, 0xc1], |a| a.mov(Qword, Rcx, Rax)),
            (&[0x4d, 0x89, 0xc8], |a| a.mov(Qword, R8, R9)),
            (&[0x48, 0x8b, 0x47, 0x08], |a| {
                a.mov_load(Qword, Rax, Mem::base(Rdi, 8))
            }),
            (&[0x48, 0x8b, 0x87, 0x00, 0x01, 0, 0], |a| {
                a.mov_load(Qword, Rax, Mem::base(Rdi, 256))
            }),
            (&[0x48, 0x89, 0x87, 0x08, 0, 0, 0], |a| {
                a.mov_store(Qword, Mem::base_disp32(Rdi, 8), Rax)
            }),
            (&[0x48, 0x8b, 0x04, 0x24], |a| {
                a.mov_load(Qword, Rax, Mem::base(Rsp, 0))
            }),
            (&[0x49, 0x8b, 0x45, 0x00], |a| {
                a.mov_load(Qword, Rax, Mem::base(R13, 0))
            }),
            (&[0x41, 0x33, 0x04, 0x90], |a| {
                a.alu_load(Alu::Xor, Dword, Rax, Mem::indexed(R8, Rdx, 4, 0))
            }),
            (&[0x40, 0x88, 0xf0], |a| a.mov_store(Byte, Rax, Rsi)),
            (&[0x0f, 0xb6, 0xc9], |a| a.movzx(Dword, Rcx, Byte, Rcx)),
            (&[0x40, 0x0f, 0xb6, 0xc6], |a| {
                a.movzx(Dword, Rax, Byte, Rsi)
            }),
            (&[0x48, 0x63, 0xc0], |a| a.movsxd(Rax, Rax)),
            (&[0x48, 0x0f, 0xbd, 0xc0], |a| a.bsr(Qword, Rax, Rax)),
            (&[0xf3, 0x48, 0x0f, 0xbd, 0xc0], |a| {
                a.lzcnt(Qword, Rax, Rax)
            }),
            (&[0x0f, 0xc8], |a| a.bswap(Dword, Rax)),
            (&[0x49, 0x0f, 0xc9], |a| a.bswap(Qword, R9)),
            (&[0x48, 0x0f, 0x44, 0xc1], |a| {
                a.cmov(Cond::E, Qword, Rax, Rcx)
            }),
            (&[0x83, 0xf0, 0x1f], |a| {
                a.alu_imm8(Alu::Xor, Dword, Rax, 31)
            }),
            (&[0x81, 0xe1, 0x55, 0x55, 0x55, 0x55], |a| {
                a.alu_imm32(Alu::And, Dword, Rcx, 0x5555_5555)
            }),
            (&[0x25, 0x55, 0x55, 0x55, 0x55], |a| {
                a.alu_accumulator_imm32(Alu::And, Dword, 0x5555_5555)
            }),
            (&[0x48, 0xd3, 0xf8], |a| {
                a.shift_cl(ShiftOp::Sar, Qword, Rax)
            }),
            (&[0xc1, 0xc8, 0x10], |a| {
                a.shift_imm(ShiftOp::Ror, Dword, Rax, 16)
            }),
            (&[0x48, 0x0f, 0xac, 0xc1, 0x07], |a| {
                a.shrd_imm(Qword, Rcx, Rax, 7)
            }),
            (&[0x48, 0xf7, 0xe9], |a| a.unary(Unary::Imul, Qword, Rcx)),
            (&[0x48, 0x99], |a| a.sign_extend_accumulator(Qword)),
            (&[0xb9, 0x3f, 0, 0, 0], |a| a.mov_imm32(Rcx, 63)),
            (&[0x48, 0xba, 1, 2, 3, 4, 5, 6, 7, 8], |a| {
                a.mov_imm64(Rdx, 0x0807_0605_0403_0201)
            }),
            (&[0x4c, 0x8d, 0x05, 0, 0, 0, 0], |a| {
                a.lea(Qword, R8, Mem::rip(0))
            }),
            (&[0xc4, 0xe2, 0xf1, 0xf7, 0xc0], |a| {
                a.shift_x(ShiftOp::Shl, Qword, Rax, Rax, Rcx)
            }),
            (&[0xc4, 0xe2, 0x72, 0xf7, 0xc0], |a| {
                a.shift_x(ShiftOp::Sar, Dword, Rax, Rax, Rcx)
            }),
            (&[0xc4, 0xe2, 0x73, 0xf7, 0xc0], |a| {
                a.shift_x(ShiftOp::Shr, Dword, Rax, Rax, Rcx)
            }),
            (&[0xc4, 0xe3, 0xfb, 0xf0, 0xc0, 0x05], |a| {
                a.rorx(Qword, Rax, Rax, 5)
            }),
            (&[0x66, 0x48, 0x0f, 0x6e, 0xc0], |a| a.movq_to_xmm(0, Rax)),
            (&[0x66, 0x48, 0x0f, 0x7e, 0xc8], |a| a.movq_from_xmm(Rax, 1)),
            (&[0x66, 0x49, 0x0f, 0x6e, 0xca], |a| a.movq_to_xmm(1, R10)),
            (&[0x66, 0x0f, 0x3a, 0xce, 0xc1, 0x00], |a| {
                a.gf2p8affineqb(0, Rm::Xmm(1), 0)
            }),
            (&[0x66, 0x44, 0x0f, 0x3a, 0xce, 0xc1, 0x00], |a| {
                a.gf2p8affineqb(8, Rm::Xmm(1), 0)
            }),
        ] {
            let (code, count) = bytes(build);
            assert_eq!(code, expected);
            assert_eq!(count, 1);
        }
    }

    #[test]
    fn core_forms_match_the_sdm_encodings() {
        use Gpr::*;
        for (expected, build) in [
            (
                &[0x48, 0x8b, 0x47, 0x00][..],
                (|a: &mut Asm| a.mov_load(Size::Qword, Rax, Mem::base_disp8(Rdi, 0)))
                    as fn(&mut Asm),
            ),
            (&[0x41, 0x57], |a| a.push(R15)),
            (&[0x5d], |a| a.pop(Rbp)),
            (&[0xff, 0xd0], |a| a.call_rm(Rax)),
            (&[0x41, 0xff, 0xe3], |a| a.jmp_rm(R11)),
            (&[0x66, 0x0f, 0xef, 0xc1], |a| {
                a.sse(Some(0x66), &[0x0f, 0xef], 0, Rm::Xmm(1))
            }),
            (&[0x66, 0x45, 0x0f, 0xef, 0xc1], |a| {
                a.sse(Some(0x66), &[0x0f, 0xef], 8, Rm::Xmm(9))
            }),
            (&[0x66, 0x0f, 0x70, 0xc1, 0x1b], |a| {
                a.sse_imm(Some(0x66), &[0x0f, 0x70], 0, Rm::Xmm(1), 0x1b)
            }),
            (&[0x0f, 0xae, 0xf0], |a| a.fixed(&[0x0f, 0xae, 0xf0])),
            // vpaddd zmm0{k1}, zmm1, zmm2
            (&[0x62, 0xf1, 0x75, 0x49, 0xfe, 0xc2], |a| {
                a.evex(
                    Evex {
                        map: Map::M0f,
                        pp: Pp::P66,
                        w: false,
                        length: 2,
                        mask: 1,
                        zeroing: false,
                        broadcast: false,
                        scale: 64,
                    },
                    0xfe,
                    0,
                    1,
                    Rm::Xmm(2),
                )
            }),
            // vpaddd zmm16, zmm17, [rdi+0x40]: disp8*64 = 1.
            (&[0x62, 0xe1, 0x75, 0x40, 0xfe, 0x47, 0x01], |a| {
                a.evex(
                    Evex {
                        map: Map::M0f,
                        pp: Pp::P66,
                        w: false,
                        length: 2,
                        mask: 0,
                        zeroing: false,
                        broadcast: false,
                        scale: 64,
                    },
                    0xfe,
                    16,
                    17,
                    Rm::Mem(Mem::base(Rdi, 0x40)),
                )
            }),
        ] {
            let (code, count) = bytes(build);
            assert_eq!(code, expected);
            assert_eq!(count, 1);
        }
    }

    #[test]
    fn checkpoints_preserve_counts_when_moving_cold_code() {
        let mut code = Code::new();
        code.asm().mov_imm64(Gpr::Rax, 0x1234);
        let hot = code.checkpoint();
        let skip = code.asm().jmp();
        let cold = code.checkpoint();
        resolve(&mut code, skip, cold.offset());
        code.asm().mov_imm32(Gpr::Rcx, 1);
        code.asm().ret();
        let tail = code.split_off(cold);
        assert_eq!(tail.instructions(), 2);
        assert_eq!(code.instructions(), 2);
        code.truncate(hot);
        assert_eq!(code.instructions(), 1);
        let bytes = code.len();
        code.append(tail);
        assert_eq!(code.instructions(), 3);
        code.remove_instruction(0..bytes);
        assert_eq!(code.instructions(), 2);
        assert_eq!(*code, [0xb9, 1, 0, 0, 0, 0xc3]);
    }

    #[test]
    fn code_counts_across_encoders_and_prefixes_join_their_instruction() {
        let mut code = Code::new();
        code.asm().mov(Size::Dword, Gpr::Rax, Gpr::Rcx);
        let mut asm = code.asm();
        asm.lock();
        asm.alu(Alu::Add, Size::Dword, Mem::base(Gpr::Rdi, 0), Gpr::Rax);
        assert_eq!(code.instructions(), 2);
        assert_eq!(*code, [0x89, 0xc8, 0xf0, 0x01, 0x07]);
        code.clear();
        assert_eq!((code.len(), code.instructions()), (0, 0));
    }

    #[test]
    fn short_branches_resolve_relative_to_the_next_instruction() {
        let (code, count) = bytes(|a| {
            let forward = a.jcc8(Cond::Ne);
            a.mov(Size::Dword, Gpr::Rax, Gpr::Rax);
            a.bind(forward);
            let back = a.jmp8();
            resolve_bytes(a.code, back, 0);
        });
        assert_eq!(code, [0x75, 2, 0x89, 0xc0, 0xeb, 0xfa]);
        assert_eq!(count, 3);
    }

    #[test]
    fn branches_resolve_relative_to_the_next_instruction() {
        let (code, count) = bytes(|a| {
            let forward = a.jcc(Cond::E);
            a.mov(Size::Dword, Gpr::Rax, Gpr::Rax);
            a.bind(forward);
            let back = a.jmp();
            resolve_bytes(a.code, back, 0);
        });
        assert_eq!(
            code,
            [
                0x0f, 0x84, 2, 0, 0, 0, 0x89, 0xc0, 0xe9, 0xf3, 0xff, 0xff, 0xff
            ]
        );
        assert_eq!(count, 3);
    }
}
