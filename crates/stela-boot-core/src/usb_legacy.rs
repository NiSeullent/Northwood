// SPDX-License-Identifier: GPL-2.0-or-later
// Adapted from Tyler Smith's Stambha commit 6de44a1a4.

//! Taking an xHCI controller from the firmware through its USB Legacy
//! Support capability (xHCI 7.1), as the firmware exits.
//!
//! Some firmware keeps driving USB from system management mode after
//! ExitBootServices, for a keyboard in a boot loader with no USB driver,
//! until the operating system claims the controller. Linux claims every
//! controller in an early PCI fixup (`quirk_usb_handoff_xhci`), before
//! anything else touches it. Quiescing devices before taking ownership
//! can hang a controller still driven by the firmware's handler. Take
//! ownership on the firmware's remaining map immediately after its exit.

/// Dword register accesses of one controller.
pub trait Dwords {
    fn read32(&mut self, offset: usize) -> u32;
    fn write32(&mut self, offset: usize, value: u32);
}

const HCCPARAMS1: usize = 0x10;
const FIRST_EXTENDED_CAPABILITY: usize = 0x20;
/// The capability ID of USB Legacy Support.
const LEGACY: u32 = 1;
const BIOS_OWNED: u32 = 1 << 16;
const OS_OWNED: u32 = 1 << 24;
/// `XHCI_LEGACY_DISABLE_SMI`: the bits of USBLEGCTLSTS that are no SMI
/// enable and no write-one-to-clear event.
const DISABLE_SMI: u32 = (0x7 << 1) + (0xff << 5) + (0x7 << 17);
/// `XHCI_LEGACY_SMI_EVENTS`: the write-one-to-clear SMI events.
const SMI_EVENTS: u32 = 0x7 << 29;
/// The longest extended capability list walked.
const CAPABILITIES: usize = 256;

/// How long the firmware has to let go, as Linux gives it.
pub const HANDOFF_US: u64 = 1_000_000;

/// The offset of the USB Legacy Support capability in a register window of
/// `bytes`, or none when the controller has none, its list is malformed or
/// its registers read as absent.
pub fn find(registers: &mut impl Dwords, bytes: usize) -> Option<usize> {
    if bytes < HCCPARAMS1 + 4 {
        return None;
    }
    let parameters = registers.read32(HCCPARAMS1);
    if parameters == u32::MAX {
        return None;
    }
    let mut offset = (parameters >> 16) as usize * 4;
    for _ in 0..CAPABILITIES {
        if offset < FIRST_EXTENDED_CAPABILITY || offset.checked_add(8)? > bytes {
            return None;
        }
        let header = registers.read32(offset);
        if header == u32::MAX {
            return None;
        }
        if header & 0xff == LEGACY {
            return Some(offset);
        }
        let next = (header >> 8 & 0xff) as usize;
        if next == 0 {
            return None;
        }
        offset += next * 4;
    }
    None
}

/// What [`take`] found and did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Taken {
    /// The capability or its control word is inaccessible.
    Gone,
    /// The firmware did not own the controller. Its SMIs are off.
    Free,
    /// The firmware owned the controller and let go. Its SMIs are off.
    Released,
    /// The firmware owned the controller and did not let go in
    /// [`HANDOFF_US`]; ownership was taken regardless, with its SMIs off.
    Forced,
}

/// Claim the controller whose USB Legacy Support capability is at `at`:
/// set the OS Owned semaphore, wait for the firmware to clear its own, take
/// it when the wait ends without that, then clear the SMI enables and the
/// SMI events. `wait` polls its argument until it is true or
/// [`HANDOFF_US`] have passed, and returns whether it became true.
pub fn take<R: Dwords>(
    registers: &mut R,
    at: usize,
    wait: impl FnOnce(&mut dyn FnMut() -> bool) -> bool,
) -> Taken {
    let header = registers.read32(at);
    let is_legacy = |value: u32| value != u32::MAX && value & 0xff == LEGACY;
    if !is_legacy(header) {
        return Taken::Gone;
    }
    let mut taken = Taken::Free;
    if header & BIOS_OWNED != 0 {
        registers.write32(at, header | OS_OWNED);
        let _ = wait(&mut || {
            let header = registers.read32(at);
            !is_legacy(header) || header & BIOS_OWNED == 0
        });
        let header = registers.read32(at);
        if !is_legacy(header) {
            return Taken::Gone;
        }
        if header & BIOS_OWNED == 0 {
            taken = Taken::Released;
        } else {
            registers.write32(at, header & !BIOS_OWNED | OS_OWNED);
            taken = Taken::Forced;
        }
    }
    let control = registers.read32(at + 4);
    if control == u32::MAX {
        return Taken::Gone;
    }
    registers.write32(at + 4, control & DISABLE_SMI | SMI_EVENTS);
    taken
}

/// Whether the page tables at `root`, of `levels` levels (4 or 5), hold a
/// present, writable identity mapping for `address`. `entry` reads entry `index` of the table
/// at a physical address. UEFI leaves the mapping of memory-mapped
/// registers after ExitBootServices to the firmware, so a capability is
/// written only where this holds.
pub fn mapped_writable(
    entry: impl Fn(u64, usize) -> u64,
    root: u64,
    levels: u32,
    address: u64,
) -> bool {
    const PRESENT: u64 = 1;
    const WRITABLE: u64 = 1 << 1;
    const LARGE: u64 = 1 << 7;
    const FRAME: u64 = 0x000f_ffff_ffff_f000;
    if !matches!(levels, 4 | 5) || address >= 1 << (11 + 9 * levels) {
        return false;
    }
    let mut table = root & FRAME;
    for level in (0..levels).rev() {
        let found = entry(table, (address >> (12 + 9 * level) & 0x1ff) as usize);
        if found & (PRESENT | WRITABLE) != PRESENT | WRITABLE {
            return false;
        }
        // A 4 KiB page, or a 2 MiB or 1 GiB one.
        if level == 0 || (level <= 2 && found & LARGE != 0) {
            let page_mask = !((1 << (12 + 9 * level)) - 1);
            return found & FRAME & page_mask == address & page_mask;
        }
        table = found & FRAME;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    /// Registers as dwords, with a firmware that clears its semaphore after
    /// `releases_after` reads of the capability once the OS asks.
    struct Model {
        words: Vec<u32>,
        legacy: usize,
        releases_after: Option<u32>,
        disappears_after_claim: bool,
        writes: Vec<(usize, u32)>,
    }

    impl Model {
        fn new(owned: bool, releases_after: Option<u32>) -> Self {
            let mut words = alloc::vec![0u32; 0x400];
            // xECP at 0x800: a protocol capability, then legacy support.
            words[HCCPARAMS1 / 4] = (0x800 / 4) << 16;
            words[0x800 / 4] = 2 | 4 << 8;
            let legacy = 0x810;
            words[legacy / 4] = LEGACY | if owned { BIOS_OWNED } else { 0 };
            // Every SMI enable on and one event pending.
            words[legacy / 4 + 1] = 1 | 1 << 4 | 0x7 << 13 | 1 << 29;
            Self {
                words,
                legacy,
                releases_after,
                disappears_after_claim: false,
                writes: Vec::new(),
            }
        }
    }

    impl Dwords for Model {
        fn read32(&mut self, offset: usize) -> u32 {
            let word = self.words[offset / 4];
            if offset == self.legacy && word & OS_OWNED != 0 && self.disappears_after_claim {
                return u32::MAX;
            }
            if offset == self.legacy
                && word & OS_OWNED != 0
                && let Some(left) = self.releases_after.as_mut()
            {
                if *left == 0 {
                    self.words[offset / 4] &= !BIOS_OWNED;
                } else {
                    *left -= 1;
                }
            }
            word
        }
        fn write32(&mut self, offset: usize, value: u32) {
            self.writes.push((offset, value));
            self.words[offset / 4] = if offset == self.legacy + 4 {
                // The events clear where a one is written.
                value & !SMI_EVENTS
            } else {
                value
            };
        }
    }

    #[test]
    fn only_a_present_writable_mapping_at_every_level_counts() {
        // Tables at 0x1000 (root), 0x2000, 0x3000, 0x4000; entry 0 of each
        // leads down, and the leaf maps one 4 KiB page. Root entry 1 is a
        // read-only branch, entry 2 of the 0x2000 table a 1 GiB page.
        let entry = |table: u64, index: usize| match (table, index) {
            (0x1000, 0) => 0x2000 | 3,
            (0x1000, 1) => 0x2000 | 1,
            (0x2000, 0) => 0x3000 | 3,
            (0x2000, 2) => 0x8000_0000 | 3 | 1 << 7,
            (0x3000, 0) => 0x4000 | 3,
            (0x4000, 5) => 0x5000 | 3,
            (0x4000, 7) => 0xfee0_7000 | 3,
            _ => 0,
        };
        assert!(mapped_writable(entry, 0x1000, 4, 0x5000));
        assert!(mapped_writable(entry, 0x1000, 4, 0x5ffc));
        // The next page is not present.
        assert!(!mapped_writable(entry, 0x1000, 4, 0x6000));
        // A present and writable mapping to a different physical address.
        assert!(!mapped_writable(entry, 0x1000, 4, 0x7000));
        // A non-canonical address cannot be used as a pointer.
        assert!(!mapped_writable(entry, 0x1000, 4, 1 << 47 | 0x5000));
        // Inside the 1 GiB page.
        assert!(mapped_writable(entry, 0x1000, 4, 0x8765_4320));
        // Below a read-only entry nothing is writable.
        assert!(!mapped_writable(entry, 0x1000, 4, 1 << 39));
        // A root with flags in its low bits, as CR3 may carry.
        assert!(mapped_writable(entry, 0x1018, 4, 0x5000));
    }

    /// Polls a bounded number of times, as a deadline does.
    fn wait(polls: u32) -> impl FnOnce(&mut dyn FnMut() -> bool) -> bool {
        move |done| (0..polls).any(|_| done())
    }

    #[test]
    fn the_capability_is_found_behind_another_and_absent_lists_are_refused() {
        let mut model = Model::new(true, None);
        assert_eq!(find(&mut model, 0x1000), Some(0x810));
        // A window that ends before the capability.
        assert_eq!(find(&mut model, 0x810), None);
        // No extended capabilities.
        model.words[HCCPARAMS1 / 4] = 0;
        assert_eq!(find(&mut model, 0x1000), None);
        // A malformed pointer into the fixed capability registers.
        model.words[HCCPARAMS1 / 4] = 1 << 16;
        model.words[1] = LEGACY;
        assert_eq!(find(&mut model, 0x1000), None);
        // Registers that read as absent.
        model.words[HCCPARAMS1 / 4] = u32::MAX;
        assert_eq!(find(&mut model, 0x1000), None);
        // A list that ends before it.
        let mut model = Model::new(true, None);
        model.words[0x800 / 4] = 2;
        assert_eq!(find(&mut model, 0x1000), None);
        assert!(model.writes.is_empty());
    }

    #[test]
    fn a_firmware_that_lets_go_is_waited_for_and_its_smis_end() {
        let mut model = Model::new(true, Some(3));
        assert_eq!(take(&mut model, 0x810, wait(10)), Taken::Released);
        let header = model.words[0x810 / 4];
        assert!(header & OS_OWNED != 0 && header & BIOS_OWNED == 0);
        // No SMI enable is left and the pending event was acknowledged.
        assert_eq!(model.words[0x814 / 4], 0);
        assert_eq!(model.writes.last(), Some(&(0x814, SMI_EVENTS)));
    }

    #[test]
    fn a_firmware_that_keeps_its_semaphore_loses_it_after_the_wait() {
        let mut model = Model::new(true, None);
        assert_eq!(take(&mut model, 0x810, wait(10)), Taken::Forced);
        let header = model.words[0x810 / 4];
        assert!(header & OS_OWNED != 0 && header & BIOS_OWNED == 0);
        assert_eq!(model.words[0x814 / 4], 0);
    }

    #[test]
    fn a_free_controller_is_not_waited_for_and_a_missing_capability_is_left() {
        let mut model = Model::new(false, None);
        let taken = take(&mut model, 0x810, |_| panic!("no wait without an owner"));
        assert_eq!(taken, Taken::Free);
        // Only the SMI word is written.
        assert_eq!(model.writes, [(0x814, SMI_EVENTS)]);
        // The offset of another capability, or absent registers.
        let mut model = Model::new(true, None);
        assert_eq!(take(&mut model, 0x800, wait(1)), Taken::Gone);
        model.words[0x810 / 4] = u32::MAX;
        assert_eq!(take(&mut model, 0x810, wait(1)), Taken::Gone);
        assert!(model.writes.is_empty());
    }

    #[test]
    fn disappearing_capability_is_left_alone_after_the_claim_request() {
        let mut model = Model::new(true, None);
        model.disappears_after_claim = true;
        assert_eq!(take(&mut model, 0x810, wait(10)), Taken::Gone);
        assert_eq!(model.writes, [(0x810, LEGACY | BIOS_OWNED | OS_OWNED)]);
    }
}
