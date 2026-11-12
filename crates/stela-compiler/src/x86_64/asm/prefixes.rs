//! Operand-derived legacy, REX2, VEX and vector EVEX prefixes.
use super::{Asm, Disp, Evex, Gpr, Map, Mem, Rm, Size, Vex};

impl Asm<'_> {
    pub(super) fn modrm(&mut self, reg: u8, rm: Rm) {
        self.modrm_scaled(reg, rm, 1);
    }

    /// ModRM, SIB and displacement, with an 8-bit displacement counted in
    /// units of `scale` bytes (EVEX disp8*N; 1 elsewhere).
    fn modrm_scaled(&mut self, reg: u8, rm: Rm, scale: u8) {
        let reg = (reg & 7) << 3;
        match rm {
            Rm::Reg(register) => self.code.push(0xc0 | reg | register.low()),
            Rm::Xmm(register) => self.code.push(0xc0 | reg | (register & 7)),
            Rm::Mem(Mem {
                base: None,
                index,
                displacement,
                ..
            }) => {
                debug_assert!(index.is_none(), "RIP-relative addressing has no index");
                self.code.push(reg | 5);
                self.code.extend(displacement.to_le_bytes());
            }
            Rm::Mem(Mem {
                base: Some(base),
                index,
                displacement,
                width,
            }) => {
                #[cfg(debug_assertions)]
                debug_assert!(
                    !(self.pinned
                        && base == Gpr::Rdi
                        && index.is_none()
                        && super::super::pinned::slot(displacement)),
                    "the frame slot of a pinned register"
                );
                // RBP and R13 have no mod-00 form; RSP and R12 need a SIB.
                let scale = i32::from(scale);
                let short = displacement % scale == 0 && i8::try_from(displacement / scale).is_ok();
                let mode = match width {
                    Disp::Dword => 2,
                    Disp::Byte => {
                        debug_assert!(short, "displacement fits its 8-bit field");
                        1
                    }
                    Disp::Auto if displacement == 0 && base.low() != 5 => 0,
                    Disp::Auto if short => 1,
                    Disp::Auto => 2,
                };
                match index {
                    None if base.low() != 4 => self.code.push(mode << 6 | reg | base.low()),
                    None => self.code.extend([mode << 6 | reg | 4, 0x24]),
                    Some((index, scale)) => {
                        debug_assert!(index != Gpr::Rsp, "RSP cannot be an index");
                        let scale = match scale {
                            1 => 0,
                            2 => 1,
                            4 => 2,
                            8 => 3,
                            _ => unreachable!("scale of 1, 2, 4 or 8"),
                        };
                        self.code.extend([
                            mode << 6 | reg | 4,
                            scale << 6 | index.low() << 3 | base.low(),
                        ]);
                    }
                }
                match mode {
                    1 => self.code.push((displacement / scale) as u8),
                    2 => self.code.extend(displacement.to_le_bytes()),
                    _ => {}
                }
            }
        }
    }

    /// A legacy-encoded instruction: the operand-size prefix for words, the
    /// mandatory `prefix`, REX when any field needs it, `opcode` and ModRM.
    pub(crate) fn legacy(
        &mut self,
        size: Size,
        prefix: Option<u8>,
        opcode: &[u8],
        reg: u8,
        reg_byte: bool,
        rm: Rm,
    ) {
        let byte_rex =
            size == Size::Byte && (rm.needs_byte_rex() || (reg_byte && (4..8).contains(&reg)));
        self.legacy_extended(size, prefix, opcode, reg, byte_rex, rm);
    }

    pub(super) fn legacy_extended(
        &mut self,
        size: Size,
        prefix: Option<u8>,
        opcode: &[u8],
        reg: u8,
        byte_rex: bool,
        rm: Rm,
    ) {
        if size == Size::Word {
            self.code.push(0x66);
        }
        if let Some(prefix) = prefix {
            self.code.push(prefix);
        }
        let skip = self.opcode_prefix(size, byte_rex, opcode, reg, rm);
        self.code.extend(&opcode[skip..]);
        self.modrm(reg, rm);
        self.count();
    }

    /// Emit REX or REX2, returning the opcode escape bytes consumed by REX2.
    pub(super) fn opcode_prefix(
        &mut self,
        size: Size,
        force: bool,
        opcode: &[u8],
        reg: u8,
        rm: Rm,
    ) -> usize {
        assert!(reg < 32, "integer register encoding is at most five bits");
        let (x, b) = rm.extensions();
        let (x4, b4) = rm.extended_extensions();
        let w = size == Size::Qword;
        let r = reg & 8 != 0;
        let r4 = reg & 16 != 0;
        if r4 || x4 || b4 {
            let map1 = opcode.first() == Some(&0x0f);
            assert!(
                (!map1 && opcode.len() == 1) || (map1 && opcode.len() == 2),
                "REX2 supports only legacy maps zero and one"
            );
            self.code.extend([
                0xd5,
                u8::from(map1) << 7
                    | u8::from(r4) << 6
                    | u8::from(x4) << 5
                    | u8::from(b4) << 4
                    | u8::from(w) << 3
                    | u8::from(r) << 2
                    | u8::from(x) << 1
                    | u8::from(b),
            ]);
            usize::from(map1)
        } else {
            if w || r || x || b || force {
                self.code.push(
                    0x40 | u8::from(w) << 3 | u8::from(r) << 2 | u8::from(x) << 1 | u8::from(b),
                );
            }
            0
        }
    }

    /// A VEX-encoded instruction: two-byte C5 when the fields allow it,
    /// otherwise three-byte C4.
    pub(crate) fn vex(&mut self, prefix: Vex, opcode: u8, reg: u8, vvvv: u8, rm: Rm) {
        assert!(
            reg < 16 && vvvv < 16,
            "VEX registers have four encoding bits"
        );
        assert_eq!(
            rm.extended_extensions(),
            (false, false),
            "VEX cannot encode APX registers"
        );
        if let Rm::Xmm(_) = rm {
            // Register-register VEX forms here are vector instructions.
            self.guard_vector(reg);
        }
        let Vex { map, pp, w, l } = prefix;
        let (x, b) = rm.extensions();
        let r = reg & 8 != 0;
        let tail = u8::from(l) << 2 | pp as u8 | (!vvvv & 15) << 3;
        if map == Map::M0f && !w && !x && !b {
            self.code.extend([0xc5, u8::from(!r) << 7 | tail]);
        } else {
            self.code.extend([
                0xc4,
                u8::from(!r) << 7 | u8::from(!x) << 6 | u8::from(!b) << 5 | map as u8,
                u8::from(w) << 7 | tail,
            ]);
        }
        self.code.push(opcode);
        self.modrm(reg, rm);
        self.count();
    }

    /// An EVEX-encoded instruction. Registers number 0 to 31; `rm` names
    /// an XMM register or memory.
    pub(crate) fn evex(&mut self, prefix: Evex, opcode: u8, reg: u8, vvvv: u8, rm: Rm) {
        assert!(
            reg < 32 && vvvv < 32,
            "EVEX vector registers have five encoding bits"
        );
        assert_eq!(
            rm.extended_extensions(),
            (false, false),
            "APX extended memory addressing requires its dedicated EVEX form"
        );
        self.guard_vector(reg);
        let Evex {
            map,
            pp,
            w,
            length,
            mask,
            zeroing,
            broadcast,
            scale,
        } = prefix;
        let (x, b) = match rm {
            // EVEX.X extends a register r/m to 32 registers.
            Rm::Xmm(register) => (register & 16 != 0, register & 8 != 0),
            other => other.extensions(),
        };
        let r = reg & 8 != 0;
        let r2 = reg & 16 != 0;
        let v2 = vvvv & 16 != 0;
        self.code.extend([
            0x62,
            u8::from(!r) << 7
                | u8::from(!x) << 6
                | u8::from(!b) << 5
                | u8::from(!r2) << 4
                | map as u8,
            u8::from(w) << 7 | (!vvvv & 15) << 3 | 4 | pp as u8,
            u8::from(zeroing) << 7
                | (length & 3) << 5
                | u8::from(broadcast) << 4
                | u8::from(!v2) << 3
                | (mask & 7),
            opcode,
        ]);
        let scale = if matches!(rm, Rm::Mem(_)) {
            scale.max(1)
        } else {
            1
        };
        self.modrm_scaled(reg, rm, scale);
        self.count();
    }
}

#[cfg(test)]
mod tests;
