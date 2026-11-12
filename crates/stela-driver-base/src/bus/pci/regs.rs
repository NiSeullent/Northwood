// SPDX-License-Identifier: GPL-2.0-only
// PCI encodings follow Linux include/uapi/linux/pci_regs.h, copyright
// 1994 Drew Eckhardt and 1997-1999 Martin Mares; see NOTICE.
//! The configuration header registers every function shares, and their bits.

pub const VENDOR_ID: u16 = 0x00;
pub const COMMAND: u16 = 0x04;
pub const COMMAND_IO: u16 = 1 << 0;
pub const COMMAND_MEMORY: u16 = 1 << 1;
pub const COMMAND_MASTER: u16 = 1 << 2;
pub const COMMAND_INVALIDATE: u16 = 1 << 4;
pub const COMMAND_INTX_DISABLE: u16 = 1 << 10;
pub const STATUS: u16 = 0x06;
pub const STATUS_CAP_LIST: u16 = 1 << 4;
/// Revision ID, then the class code's three bytes.
pub const CLASS_REVISION: u16 = 0x08;
pub const HEADER_TYPE: u16 = 0x0e;
pub const BASE_ADDRESS_0: u16 = 0x10;
pub const SUBSYSTEM_VENDOR_ID: u16 = 0x2c;
pub const CAPABILITY_LIST: u16 = 0x34;
/// Interrupt Pin: 0 when the function uses no INTx pin, 1-4 for INTA-INTD.
/// Type 0 and type 1 headers both place it here.
pub const INTERRUPT_PIN: u16 = 0x3d;

/// Whether a function must read back COMMAND's INTx Disable as set before
/// its INTx is known to be quiet. A function whose Interrupt Pin is 0 has no
/// INTx pin to assert, and the PCI and PCI Express specifications let it
/// hardwire INTx Disable to 0 (host bridges commonly implement no COMMAND bit
/// at all). Any other pin value, including a failed read's 0xff, requires the
/// bit.
pub const fn intx_disable_required(interrupt_pin: u8) -> bool {
    interrupt_pin != 0
}

/// COMMAND bits a stop asks for that a function may leave clear without
/// any effect on containment: INTx Disable when no INTx pin is implemented.
/// Bus Master is never among them: a function that keeps it set can DMA.
pub const fn command_optional_bits(interrupt_pin: u8) -> u16 {
    if intx_disable_required(interrupt_pin) {
        0
    } else {
        COMMAND_INTX_DISABLE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intx_disable_is_optional_only_without_an_interrupt_pin() {
        assert_eq!(command_optional_bits(0), COMMAND_INTX_DISABLE);
        for pin in [1, 2, 3, 4, 5, 0xff] {
            assert!(intx_disable_required(pin));
            assert_eq!(command_optional_bits(pin), 0);
        }
        assert_eq!(command_optional_bits(0) & COMMAND_MASTER, 0);
    }
}
