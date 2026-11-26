// SPDX-License-Identifier: GPL-2.0-or-later
// Adapted from Tyler Smith's Stambha AP parking at 8746557de45bf39e3f3f6a3ca5fb93dd6479d500.

//! Keep APs available for firmware SMI rendezvous during device takeover.
//! An AP in wait-for-SIPI cannot service an SMI. After firmware AP code is
//! stopped, the BSP starts each AP into an owned, interrupt-masked HLT loop.
use super::{StartupFailure, bootstrap, firmware, topology};
use crate::{
    apic::{self, Apic},
    engine::interrupts,
    handoff::Handoff,
    layout,
    tables::PAGE,
};
use alloc::{boxed::Box, vec};
use core::{
    arch::asm,
    cell::UnsafeCell,
    sync::atomic::{AtomicBool, Ordering},
};
use stela_boot_core::memory_types::Image;
use stela_engine::machine::mmu::placement::Placement;

static STARTUP_PAGE_HELD: AtomicBool = AtomicBool::new(false);
const STACK_BYTES: usize = 32 * 1024;

struct Parking {
    tables: UnsafeCell<Option<&'static mut interrupts::Tables>>,
    x2apic: bool,
    ready: AtomicBool,
}

// SAFETY: the BSP prepares the record before SIPI. Only its owning AP
// consumes `tables`, and only `ready` is accessed by both processors.
unsafe impl Sync for Parking {}

extern "sysv64" fn fatal(frame: &mut interrupts::Frame, _: &interrupts::SavedState<'_>) {
    interrupts::fatal(frame)
}

unsafe extern "sysv64" fn enter(argument: u64) -> ! {
    // SAFETY: the BSP retains the record, stack, and descriptor storage.
    let parking = unsafe { &*(argument as *const Parking) };
    // SAFETY: only this AP takes its own descriptor storage.
    let tables = unsafe { (&mut *parking.tables.get()).take() }.expect("parking descriptors");
    // SAFETY: IF is clear since SIPI, and the private tables stay mapped.
    unsafe { interrupts::install_on(tables, fatal) };
    if parking.x2apic {
        // SAFETY: CPL0, interrupts masked, and no local APIC handle exists.
        unsafe { apic::prefer_x2apic() };
    }
    parking.ready.store(true, Ordering::Release);
    loop {
        // SAFETY: the AP owns persistent tables for NMI and machine checks.
        // SMI is still accepted with IF clear. INIT can restart this AP later.
        unsafe { asm!("hlt", options(nomem, nostack)) };
    }
}

pub(super) fn page_held() -> bool {
    STARTUP_PAGE_HELD.load(Ordering::Acquire)
}

/// Park firmware APs before the OS invokes AML or takes over PCI devices.
///
/// # Safety
/// Caller is the BSP after EBS, with its host root and descriptors installed
/// and IF clear. All firmware APs had INIT sent and none use the SIPI page.
pub(crate) unsafe fn before_firmware(handoff: &Handoff, capture: &firmware::Capture) {
    let Ok(targets) = topology::enabled_unselected(&capture.processors, &capture.selected[..1])
    else {
        return;
    };
    let count = targets.clone().count();
    if count == 0 {
        return;
    }
    let physical = handoff.native_resources.sipi;
    if physical == 0 {
        crate::output::arguments(format_args!(
            "NATIVE PARKED BEFORE FIRMWARE 0 of {count}: no startup page\n"
        ));
        return;
    }
    // SAFETY: the caller has installed the BSP's CPL0 context.
    let memory = unsafe { firmware::observe() }
        .map_err(|error| error.message())
        .and_then(|source| {
            let image = Image::from_snapshot(&source)
                .map_err(|_| "BSP memory types could not be copied")?;
            if image.amd.is_some() {
                image
                    .writeback_ram(physical, PAGE)
                    .and_then(|()| image.writeback_ram(handoff.heap.physical, handoff.heap.bytes))
                    .map_err(|_| "startup page or heap is outside write-back RAM")?;
            }
            Ok((image, source.pat))
        });
    let (image, pat) = match memory {
        Ok(memory) => memory,
        Err(error) => {
            crate::output::arguments(format_args!(
                "NATIVE PARKED BEFORE FIRMWARE 0 of {count}: {error}\n"
            ));
            return;
        }
    };
    let sender = Apic::new(
        handoff.apic_base,
        layout::physical(handoff.apic_base & apic::BASE_ADDRESS),
    );
    let mut parked = 0;
    for index in targets {
        let source = capture.processors[index];
        // Every enabled AP already passed this check before the first INIT.
        let Ok(destination) = u32::try_from(source.apic_id) else {
            continue;
        };
        let signature = capture.selected[..capture.core_count as usize]
            .iter()
            .position(|&selected| selected as usize == index)
            .and_then(|slot| capture.snapshots[slot].get(1, 0));
        let signature = signature.or_else(|| capture.snapshots[0].get(1, 0));
        let Some(signature) = signature else {
            break;
        };
        // SAFETY: each previous AP acknowledged leaving the shared page.
        match unsafe { park_one(handoff, &sender, &image, pat, signature.eax, destination) } {
            Ok(()) => parked += 1,
            Err(error) => {
                crate::output::arguments(format_args!(
                    "NATIVE PARK failed apic={destination:#x} firmware_cpu={}: {error:?}\n",
                    source.processor_number,
                ));
                // INIT delivery does not acknowledge that the AP has left the
                // trampoline. Preserve it even when the reset command succeeds.
                STARTUP_PAGE_HELD.store(true, Ordering::Release);
                // SAFETY: the BSP owns the ICR and the validated AP target.
                if let Err(error) = unsafe { sender.init_only(destination, handoff.tsc_hz) } {
                    crate::output::arguments(format_args!(
                        "NATIVE PARK reset failed apic={destination:#x}: {error:?}\n"
                    ));
                }
                break;
            }
        }
    }
    crate::output::arguments(format_args!(
        "NATIVE PARKED BEFORE FIRMWARE {parked} of {count}; wait_for_sipi={}\n",
        count - parked,
    ));
}

/// All startup data stays pinned across a successful park or uncertain entry.
unsafe fn park_one(
    handoff: &Handoff,
    sender: &Apic,
    image: &Image,
    pat: u64,
    signature: u32,
    destination: u32,
) -> Result<(), StartupFailure> {
    let stack = Box::leak(vec![0u8; STACK_BYTES].into_boxed_slice());
    let stack_top = (stack.as_mut_ptr() as u64 + STACK_BYTES as u64) & !15;
    let parking = Box::leak(Box::new(Parking {
        tables: UnsafeCell::new(Some(Box::leak(Box::new(interrupts::Tables::new())))),
        x2apic: sender.is_x2apic(),
        ready: AtomicBool::new(false),
    }));
    let physical = handoff.native_resources.sipi;
    let mapped = layout::physical(physical) as *mut u8;
    // SAFETY: the BSP exclusively holds the low page; all entry data and the
    // mapped root remain resident through the AP's lifetime.
    let vector = unsafe {
        bootstrap::prepare(
            mapped,
            physical,
            bootstrap::Entry {
                processor_signature: signature,
                root: handoff.root,
                five_level: matches!(handoff.placement, Placement::FiveLevel),
                pat,
                memory_types: image,
                stack_top,
                argument: parking as *const Parking as u64,
                entry: enter,
            },
        )
    }
    .map_err(StartupFailure::Bootstrap)?;
    // SAFETY: the BSP owns the ICR, the target, and its full startup context.
    unsafe { sender.startup(destination, vector, handoff.tsc_hz) }
        .map_err(StartupFailure::Delivery)?;
    let started = apic::rdtsc();
    let slot = destination as usize;
    loop {
        // SAFETY: the page is still owned by this startup attempt.
        let failure = unsafe { bootstrap::failure(mapped) }
            .map_err(|cause| StartupFailure::FaultRecord { slot, cause })?;
        if let Some(cause) = failure {
            return Err(StartupFailure::EarlyException { slot, cause });
        }
        if parking.ready.load(Ordering::Acquire) {
            return Ok(());
        }
        if apic::rdtsc().wrapping_sub(started) >= handoff.tsc_hz {
            return Err(StartupFailure::Timeout { slot });
        }
        core::hint::spin_loop();
    }
}
