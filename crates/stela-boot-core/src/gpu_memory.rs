// SPDX-License-Identifier: GPL-2.0-or-later
//! Guest GPU staging through registered RAM's canonical copy path.

use alloc::{boxed::Box, sync::Arc};
use stela_contracts::gate::MemoryRange;
use stela_drivers::gpu::metal::{
    error::{Error, Result},
    native::{GuestCopyLease, GuestCopyProvider},
};
use stela_engine::memory::{RegisteredSystemRam, WiredRam};

/// Resolves the physical ranges from a guest wire without exporting an
/// alternate virtual address for those frames.
pub struct GuestRamCopy {
    ram: Arc<RegisteredSystemRam>,
}

impl GuestRamCopy {
    pub fn new(ram: Arc<RegisteredSystemRam>) -> Self {
        Self { ram }
    }
}

impl GuestCopyProvider for GuestRamCopy {
    unsafe fn lease(&self, ranges: &[MemoryRange]) -> Result<Box<dyn GuestCopyLease>> {
        let wire = WiredRam::new(Arc::clone(&self.ram), ranges)
            .map_err(|_| Error::Invalid("writable guest GPU memory ranges"))?;
        Ok(Box::new(Lease(wire)))
    }
}

struct Lease(WiredRam);

impl GuestCopyLease for Lease {
    fn read_into(&self, offset: u64, output: &mut [u8]) -> Result<()> {
        // SAFETY: the provider's lease contract keeps the descriptor wired.
        // The GPU backend gives this call idle, private staging bytes.
        unsafe { self.0.read_into(offset, output) }.map_err(|_| Error::DeviceLost)
    }

    fn publish_from(&self, offset: u64, source: &[u8]) -> Result<()> {
        // SAFETY: the same lease is bidirectional. The backend waits for
        // completion of device writes before borrowing the staging bytes.
        // A failed copy may have published a prefix, so treat it as lost.
        unsafe { self.0.publish_from(offset, source) }.map_err(|_| Error::DeviceLost)
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    use core::{cell::UnsafeCell, ptr::NonNull};
    use stela_engine::memory::{ReservedSystemRamRange, system_ram::fixture};

    #[repr(align(16))]
    struct Storage(UnsafeCell<[u8; 64]>);

    // SAFETY: only the registered RAM owner reads and writes these raw bytes.
    unsafe impl Sync for Storage {}

    #[test]
    fn the_lease_keeps_fragmented_ram_alive_and_maps_copy_failure_to_loss() {
        let bytes = Arc::new(Storage(UnsafeCell::new([0x3a; 64])));
        let pointer = bytes.0.get().cast::<u8>();
        let ranges = [
            ReservedSystemRamRange {
                source_pa: 0x4000,
                host_pa: 0x10000,
                host_base: NonNull::new(pointer).unwrap(),
                bytes: 32,
            },
            ReservedSystemRamRange {
                source_pa: 0x8000,
                host_pa: 0x10020,
                host_base: NonNull::new(pointer.wrapping_add(32)).unwrap(),
                bytes: 32,
            },
        ];
        // SAFETY: the extents are disjoint and the fixture retains the bytes.
        let ram = Arc::new(unsafe { fixture(&ranges, bytes) });
        let owner = Arc::downgrade(&ram);
        let provider = GuestRamCopy::new(ram);
        // SAFETY: both ranges remain wired in the fixture for the whole test.
        let lease = unsafe {
            provider.lease(&[
                MemoryRange {
                    address: 0x8000,
                    bytes: 32,
                },
                MemoryRange {
                    address: 0x4000,
                    bytes: 32,
                },
            ])
        }
        .unwrap();
        drop(provider);
        let patch = [0x65; 16];
        lease.publish_from(24, &patch).unwrap();
        let mut data = [0; 64];
        lease.read_into(0, &mut data).unwrap();
        assert_eq!(data[..24], [0x3a; 24]);
        assert_eq!(data[24..40], patch);
        assert_eq!(data[40..], [0x3a; 24]);
        assert_eq!(lease.publish_from(60, &patch), Err(Error::DeviceLost));
        lease.read_into(0, &mut data).unwrap();
        assert_eq!(data[40..], [0x3a; 24]);
        assert!(owner.upgrade().is_some());
        drop(lease);
        assert!(owner.upgrade().is_none());
    }
}
