//! Guest memory mappings.
//!
//! Guest code is data to the host: no guest mapping is ever host-executable.
//! Mappings are placed exactly at the requested address without replacing an
//! existing allocation, and may be bound into a physical memory domain.

mod access;
mod allocation;
pub mod backing;
mod binding;
pub mod domain;
mod mapping;
pub mod monitor;
pub mod policy;
mod protection;
mod published;
mod service;
pub mod system_ram;
use stela_contracts::{refuse, require};
pub use system_ram::{
    CanonicalRamService, RegisteredSystemRam, RegistrationError as SystemRamRegistrationError,
    ReservedSystemRamRange, WiredRam,
};
pub mod stage1;
mod storage;
pub mod tag_execution;
pub mod tag_store;
pub mod tag_transfer;
pub mod tags;

pub use allocation::{
    AllocateDataError, AllocateDataRequest, AllocationOperation, DataAllocation, DataInheritance,
    DataPermissions, DataPlacement, PendingDataAllocation,
};
pub use binding::{SharedBindingError, SharedMemoryConfig};
#[cfg(host_memory)]
pub use mapping::FileMemoryRegionSpec;
pub use protection::{DataProtectionLimit, ProtectionError};
pub use published::PublishedReadOnlyRegion;
pub use service::GuestServiceMemory;

use crate::Result;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use alloc::{rc::Rc, sync::Arc};
use binding::SharedBinding;
use core::marker::PhantomData;
#[cfg(host_memory)]
use mapping::page_rounded;
#[cfg(host_memory)]
use protection::host_protection;
use serde::Serialize;
use storage::NativeStorage;

pub const MAX_IMAGE_MEMORY_BYTES: u64 = 256 * 1024 * 1024;
/// Virtual-space limit for file-backed mappings. Snapshot and write operations
/// keep the smaller MAX_IMAGE_MEMORY_BYTES limit.
pub const MAX_FILE_MEMORY_BYTES: u64 = 64 * 1024 * 1024 * 1024;
/// End of the lower canonical user range in the direct-RAM profile. It does not
/// define the guest's VM maximum.
pub const DIRECT_RAM_USER_END: u64 = 1 << 47;

#[derive(Debug, Clone)]
pub struct MemoryRegionSpec {
    pub guest_address: u64,
    pub virtual_bytes: u64,
    pub initial_bytes: Vec<u8>,
    pub readable: bool,
    pub writable: bool,
    pub executable: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MemoryRegion {
    pub guest_address: u64,
    /// Exact guest-visible extent. Allocation padding is not guest memory.
    pub virtual_bytes: u64,
    /// Host-page extent of this protection view; views never cross allocations.
    pub mapped_bytes: u64,
    pub initial_bytes: usize,
    pub readable: bool,
    pub writable: bool,
    /// Guest instruction-fetch permission, never host execute permission.
    pub executable: bool,
}

impl MemoryRegion {
    fn end(&self) -> u64 {
        // Construction checks addition and the native address range.
        self.guest_address + self.virtual_bytes
    }
}

#[derive(Debug)]
struct Allocation {
    shared_backing: Option<Arc<backing::MemoryBacking>>,
    storage: Arc<NativeStorage>,
    base: u64,
    bytes: u64,
    logical_bytes: u64,
    maximum_readable: bool,
    maximum_writable: bool,
    maximum_executable: bool,
    #[cfg_attr(
        not(host_memory),
        expect(dead_code, reason = "only host-memory protection reads it")
    )]
    executable: bool,
    alias_tag: u8,
    inheritance: DataInheritance,
}
/// Private, non-shareable mappings with kept native storage lifetimes.
///
/// Safe operations copy bytes; they never lend Rust references into memory that
/// native guest code may mutate. Callers executing native code remain responsible
/// for its unsafe execution contract and must keep this owner alive throughout.
/// Internal kept storage can outlive this logical owner; retention alone does
/// not pin host rights or admit concurrent/shared execution.
#[derive(Debug)]
pub struct OwnedMemory {
    shared: Option<Arc<SharedBinding>>,
    regions: Vec<MemoryRegion>,
    allocations: Vec<Allocation>,
    page_bytes: u64,
    usable: bool,
    _not_send_sync: PhantomData<Rc<()>>,
}

#[cfg(host_memory)]
pub fn host_page_size() -> Result<usize> {
    // SAFETY: sysconf has no pointer arguments or mutable Rust state.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    require!(page > 0, "cannot determine host page size");
    let page = page as usize;
    require!(page.is_power_of_two(), "unsupported host page size");
    Ok(page)
}

#[cfg(not(feature = "std"))]
pub fn host_page_size() -> Result<usize> {
    Err(refuse!("UNSUPPORTED guest memory: the freestanding build maps no host pages").into())
}

/// Hosts without POSIX mappings back no guest memory.
#[cfg(all(feature = "std", not(host_memory)))]
pub fn host_page_size() -> Result<usize> {
    Err(refuse!("UNSUPPORTED guest memory: it needs a host with POSIX memory mappings").into())
}

impl OwnedMemory {
    pub fn regions(&self) -> &[MemoryRegion] {
        &self.regions
    }

    /// False only if a failed host operation could not restore prior rights.
    /// Safe byte APIs then decline; native executors must also refuse resumption.
    pub fn is_usable(&self) -> bool {
        self.usable
    }

    fn validate_access(&self, address: u64, length: usize, write: bool) -> Result<()> {
        require!(
            self.usable,
            "guest address space is unusable after failed protection rollback"
        );
        require!(
            length as u64 <= MAX_IMAGE_MEMORY_BYTES,
            "guest access exceeds memory limit"
        );
        let end = address
            .checked_add(length as u64)
            .ok_or_else(|| refuse!("guest access address overflow", address, length))?;
        if length == 0 {
            return Ok(());
        }
        let mut cursor = address;
        for region in &self.regions {
            if region.end() <= cursor {
                continue;
            }
            require!(
                region.guest_address <= cursor,
                "guest access reaches unmapped memory"
            );
            require!(
                if write {
                    region.writable
                } else {
                    region.readable
                },
                "guest access violates region permissions"
            );
            cursor = region.end().min(end);
            if cursor == end {
                return Ok(());
            }
        }
        Err(refuse!("guest access reaches unmapped memory", address, length).into())
    }

    /// Validate one complete private data write without changing any bytes.
    /// Platform copyout owners select whether whole-range preflight is appropriate.
    pub fn validate_write(&self, address: u64, length: usize) -> Result<()> {
        self.validate_access(address, length, true)
    }

    pub fn read_snapshot(&self, address: u64, length: usize) -> Result<Vec<u8>> {
        self.validate_access(address, length, false)?;
        if let Some(shared) = &self.shared {
            return shared
                .domain
                .service_loan()
                .map_err(SharedBindingError::Domain)?
                .snapshot(address, length)
                .map_err(SharedBindingError::Domain)
                .map_err(Into::into);
        }
        let mut bytes = vec![0; length];
        for (start, count, offset) in self.access_spans(address, length) {
            // SAFETY: all requested bytes belong to live readable regions, the
            // destination is independent, and this owner cannot cross threads.
            // Each copy stays within one mmap allocation.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    start as usize as *const u8,
                    bytes.as_mut_ptr().add(offset),
                    count,
                )
            };
        }
        Ok(bytes)
    }

    /// Validate the entire write first, including crossings of adjacent regions.
    pub fn write(&mut self, address: u64, bytes: &[u8]) -> Result<()> {
        self.validate_access(address, bytes.len(), true)?;
        if self.shared.is_some() {
            return self.write_shared(address, bytes);
        }
        for (start, count, offset) in self.access_spans(address, bytes.len()) {
            // SAFETY: the exclusive owner validated every byte as writable;
            // callers cannot obtain references into these allocations safely.
            // Each copy stays within one mmap allocation.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    bytes.as_ptr().add(offset),
                    start as usize as *mut u8,
                    count,
                )
            };
        }
        Ok(())
    }

    // Only called after validate_access has checked the end and permissions.
    fn access_spans(
        &self,
        address: u64,
        length: usize,
    ) -> impl Iterator<Item = (u64, usize, usize)> + '_ {
        let end = address + length as u64;
        self.regions.iter().filter_map(move |region| {
            let start = address.max(region.guest_address);
            let stop = end.min(region.end());
            (start < stop).then(|| (start, (stop - start) as usize, (start - address) as usize))
        })
    }
}

#[cfg(all(
    test,
    target_arch = "x86_64",
    any(target_os = "macos", target_os = "linux")
))]
mod tests;
