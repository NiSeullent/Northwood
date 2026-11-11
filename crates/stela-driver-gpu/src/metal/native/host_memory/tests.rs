// SPDX-License-Identifier: GPL-2.0-or-later

use super::*;
use core::cell::UnsafeCell;

const PAGE: u64 = 4096;

#[repr(align(4096))]
struct Page(UnsafeCell<[u8; PAGE as usize]>);

// The tests take a host reference only when the single submitting thread has
// observed the Vulkan queue idle. This also permits holding the lease in Arc.
unsafe impl Sync for Page {}

struct Exporter(alloc::sync::Arc<Page>);

impl HostMemoryExporter for Exporter {
    fn lease(&self, _: &[MemoryRange]) -> Result<HostAllocation> {
        let pointer = NonNull::new(self.0.0.get().cast::<u8>()).unwrap();
        // SAFETY: the page is pinned by the Arc and accessed only as above.
        Ok(unsafe {
            HostAllocation::new(
                pointer,
                NonZeroU64::new(PAGE).unwrap(),
                Box::new(self.0.clone()),
            )
        })
    }
}

fn exporter() -> Exporter {
    Exporter(alloc::sync::Arc::new(Page(UnsafeCell::new(
        [0; PAGE as usize],
    ))))
}

#[test]
fn ranges_and_leases_require_exact_aligned_length() {
    let ranges = [MemoryRange {
        address: PAGE,
        bytes: PAGE,
    }];
    let total = range_bytes(&ranges).unwrap();
    assert_eq!(total.get(), PAGE);
    assert!(range_bytes(&[]).is_err());
    assert!(
        range_bytes(&[MemoryRange {
            address: 0,
            bytes: 0
        }])
        .is_err()
    );
    assert!(
        range_bytes(&[MemoryRange {
            address: u64::MAX - 1,
            bytes: 2
        }])
        .is_err()
    );
    assert!(
        range_bytes(&[
            MemoryRange {
                address: 0,
                bytes: u64::MAX
            },
            ranges[0]
        ])
        .is_err()
    );
    let owner = exporter();
    let mut allocation = owner.lease(&ranges).unwrap();
    check_allocation(&allocation, total, total).unwrap();
    assert!(check_allocation(&allocation, NonZeroU64::new(2 * PAGE).unwrap(), total).is_err());
    allocation.pointer = NonNull::new(allocation.pointer.as_ptr().wrapping_add(1)).unwrap();
    assert!(check_allocation(&allocation, total, total).is_err());
}

#[test]
fn binding_refuses_out_of_bounds_alignment_and_memory_type() {
    let imported = Import {
        memory: vk::DeviceMemory::null(),
        allocation: exporter().lease(&[]).unwrap(),
        memory_type: 2,
        token: NonZeroU64::MIN,
    };
    let requirements = vk::MemoryRequirements {
        size: 1024,
        alignment: 256,
        memory_type_bits: 4,
    };
    imported.check_binding(requirements, 3072).unwrap();
    assert!(imported.check_binding(requirements, 3584).is_err());
    assert!(imported.check_binding(requirements, 1).is_err());
    assert!(
        imported
            .check_binding(requirements, u64::MAX - 255)
            .is_err()
    );
    assert!(
        imported
            .check_binding(
                vk::MemoryRequirements {
                    memory_type_bits: 2,
                    ..requirements
                },
                0
            )
            .is_err()
    );
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod linux;
