//! Firmware task-page ownership, moved once into the engine handoff.
use crate::{handoff::Region, tables::PAGE};
use core::sync::atomic::{AtomicBool, Ordering};
use uefi::boot::{self, AllocateType, MemoryType};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AllocationError {
    InvalidSize(u64),
    PageCount(u64),
    Firmware(uefi::Status),
}
impl core::fmt::Display for AllocationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidSize(bytes) => {
                write!(f, "task allocation size {bytes} is not whole pages")
            }
            Self::PageCount(bytes) => {
                write!(f, "task allocation size {bytes} exceeds page-count range")
            }
            Self::Firmware(status) => write!(f, "firmware task allocation: {status:?}"),
        }
    }
}
impl core::error::Error for AllocationError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ClaimError {
    DescriptorMismatch,
    AlreadyOwned,
}
impl core::fmt::Display for ClaimError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::DescriptorMismatch => "task allocation descriptor differs from its owner",
            Self::AlreadyOwned => "task allocation was already claimed",
        })
    }
}
impl core::error::Error for ClaimError {}

/// Created only by the firmware allocating call below. Dropping this owner
/// keeps LOADER_DATA: these pages belong to Stela for the entire boot.
#[repr(C)]
struct Allocation {
    region: Region,
    claimed: AtomicBool,
}
impl Allocation {
    fn reserve(bytes: u64) -> Result<Self, AllocationError> {
        if bytes == 0 || !bytes.is_multiple_of(PAGE) {
            return Err(AllocationError::InvalidSize(bytes));
        }
        let count = usize::try_from(bytes / PAGE).map_err(|_| AllocationError::PageCount(bytes))?;
        // The engine runs its stacks only in write-back memory; firmware
        // whose MTRRs leave some RAM otherwise gets the pages taken again.
        // SAFETY: the loader runs at CPL0 on the BSP, as the processor
        // capture's own observation of it does.
        let types = unsafe { crate::native_smp::firmware::observe_boxed() }
            .ok()
            .and_then(|snapshot| {
                crate::native_smp::memory_types::Image::from_snapshot(&snapshot).ok()
            });
        let placed = stela_boot_core::writeback_placement::place(
            || {
                boot::allocate_pages(AllocateType::AnyPages, MemoryType::LOADER_DATA, count)
                    .map(|page| page.as_ptr() as u64)
                    .map_err(|error| AllocationError::Firmware(error.status()))
            },
            |base| Some(types.as_ref()?.writeback_ram(base, bytes).is_ok()),
        )?;
        for base in &placed.rejected {
            // SAFETY: these pages were allocated just above and never used.
            let _ = unsafe {
                boot::free_pages(
                    core::ptr::NonNull::new(*base as *mut u8).expect("allocated pages"),
                    count,
                )
            };
        }
        if !placed.rejected.is_empty() || placed.writeback == Some(false) {
            crate::output::arguments(format_args!(
                "TASK MEMORY write-back={:?} after {} allocations in non-write-back memory\n",
                placed.writeback,
                placed.rejected.len() + usize::from(placed.writeback == Some(false)),
            ));
        }
        Ok(Self {
            region: Region {
                physical: placed.base,
                bytes,
            },
            claimed: AtomicBool::new(false),
        })
    }
}

/// Non-Copy allocation owners; Region in Resources remains a description.
/// Appended to Handoff, preserving its earlier field offsets. The firmware
/// reserves these pages before EBS; engine claim never reconstructs an owner.
#[repr(C)]
pub(crate) struct TaskAllocations {
    tasks: Allocation,
}
impl TaskAllocations {
    pub(crate) fn reserve(bytes: u64) -> Result<Self, AllocationError> {
        Ok(Self {
            tasks: Allocation::reserve(bytes)?,
        })
    }
    pub(crate) fn region(&self) -> Region {
        self.tasks.region
    }
    pub(crate) fn claim(&'static self, expected: Region) -> Result<AllocationClaim, ClaimError> {
        let owner = &self.tasks;
        let region = owner.region;
        if region.physical != expected.physical || region.bytes != expected.bytes {
            return Err(ClaimError::DescriptorMismatch);
        }
        owner
            .claimed
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| ClaimError::AlreadyOwned)?;
        Ok(AllocationClaim { owner })
    }
}

/// The one engine claim keeps the firmware allocation owner. No Drop path
/// clears its claim or returns its pages to firmware, the guest or an allocator.
pub(crate) struct AllocationClaim {
    owner: &'static Allocation,
}
impl AllocationClaim {
    pub(crate) fn contains(&self, physical: u64, bytes: u64) -> bool {
        physical >= self.owner.region.physical
            && physical
                .checked_add(bytes)
                .zip(
                    self.owner
                        .region
                        .physical
                        .checked_add(self.owner.region.bytes),
                )
                .is_some_and(|(end, allocation_end)| bytes != 0 && end <= allocation_end)
    }
}
