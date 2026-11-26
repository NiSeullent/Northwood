// SPDX-License-Identifier: GPL-2.0-or-later
// Adapted from Tyler Smith's Stambha commit 6de44a1a4.

//! The claim of each xHCI controller from the firmware as the firmware
//! exits ([`stela_boot_core::usb_legacy`]).
use stela_boot_core::usb_legacy::{Dwords, HANDOFF_US, Taken, mapped_writable, take};
use stela_drivers::bus::pci::Address;

/// More xHCI controllers than a PC has; one past these is claimed later,
/// as its driver starts.
const CONTROLLERS: usize = 16;

/// The USB Legacy Support capabilities found while the firmware ran.
#[derive(Clone, Copy)]
pub(crate) struct Controllers {
    found: [(Address, u64); CONTROLLERS],
    count: usize,
}

/// Find them through the firmware's own PCI protocol. Boot services are
/// active and the loader's heap is in use.
pub(crate) fn find() -> Controllers {
    let mut controllers = Controllers {
        found: [(
            Address {
                segment: 0,
                bus: 0,
                device: 0,
                function: 0,
            },
            0,
        ); CONTROLLERS],
        count: 0,
    };
    for found in crate::pci_io::xhci_legacy_capabilities()
        .into_iter()
        .take(CONTROLLERS)
    {
        controllers.found[controllers.count] = found;
        controllers.count += 1;
    }
    controllers
}

/// One capability at its physical address, which the firmware's identity
/// map still maps after its exit.
struct Capability(u64);

impl Dwords for Capability {
    fn read32(&mut self, offset: usize) -> u32 {
        // SAFETY: `Controllers::take`'s contract.
        unsafe { ((self.0 + offset as u64) as *const u32).read_volatile() }
    }
    fn write32(&mut self, offset: usize, value: u32) {
        // SAFETY: `Controllers::take`'s contract.
        unsafe { ((self.0 + offset as u64) as *mut u32).write_volatile(value) }
    }
}

impl Controllers {
    /// Claim every controller found, and say what each one's firmware did.
    ///
    /// # Safety
    /// Boot services have just ended, the firmware's identity map is still
    /// loaded, interrupts are off and nothing else drives a controller.
    pub(crate) unsafe fn take(&self, tsc_hz: u64) {
        let ticks = tsc_hz / 1_000_000 * HANDOFF_US;
        for &(address, at) in &self.found[..self.count] {
            // UEFI does not promise that the firmware's map holds a
            // controller's registers once boot services end. One it leaves
            // out is claimed as its driver starts, as before.
            if !(mapped(at) && mapped(at + 4)) {
                crate::output::arguments(format_args!(
                    "HOST USB xHCI {address} registers are not in the firmware's map, claimed later\n"
                ));
                continue;
            }
            let taken = take(&mut Capability(at), 0, |released| {
                let start = rdtsc();
                loop {
                    if released() {
                        return true;
                    }
                    if rdtsc().wrapping_sub(start) >= ticks {
                        return false;
                    }
                    core::hint::spin_loop();
                }
            });
            let what = match taken {
                Taken::Gone => "capability no longer there, left alone",
                Taken::Free => "not owned by the firmware",
                Taken::Released => "released by the firmware",
                Taken::Forced => "not released in 1 s, taken with its SMIs off",
            };
            crate::output::arguments(format_args!("HOST USB xHCI {address} {what}\n"));
        }
    }
}

/// Whether the loaded page tables map `address` present and writable.
fn mapped(address: u64) -> bool {
    const LA57: u64 = 1 << 12;
    let (root, control): (u64, u64);
    // SAFETY: CPL0 control register reads have no side effects.
    unsafe {
        core::arch::asm!("mov {}, cr3", out(reg) root, options(nomem, nostack, preserves_flags));
        core::arch::asm!("mov {}, cr4", out(reg) control, options(nomem, nostack, preserves_flags));
    }
    let levels = if control & LA57 != 0 { 5 } else { 4 };
    mapped_writable(
        // SAFETY: the firmware's tables are in memory its own identity map
        // holds, and each address read comes from a present entry of them.
        |table, index| unsafe { ((table + index as u64 * 8) as *const u64).read_volatile() },
        root,
        levels,
        address,
    )
}

fn rdtsc() -> u64 {
    let (low, high): (u32, u32);
    // SAFETY: RDTSC has no side effects.
    unsafe {
        core::arch::asm!("rdtsc", out("eax") low, out("edx") high, options(nomem, nostack));
    }
    (u64::from(high) << 32) | u64::from(low)
}
