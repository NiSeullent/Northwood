//! The mount's native page source over its heap, and the Linux page catalog
//! bound to it. The source itself is `stela_boot_core::native_memory`.
use core::ptr::NonNull;
use stela_boot_core::native_memory::{NativeMemory, PAGE, Registration};
use stela_drivers::bus::dma::memory::{
    CacheAttribute, MemoryError, PageGeometry, PageSource, PinnedPages,
};

pub(crate) mod cache;

pub static SOURCE: NativeMemory = NativeMemory::new(&crate::ALLOCATOR, cache::verify_writeback);

static AMD_TABLE_HEAP: stela_boot_core::heap::Heap = stela_boot_core::heap::Heap::new();
pub(crate) static AMD_TABLES: NativeMemory =
    NativeMemory::new(&AMD_TABLE_HEAP, cache::verify_writeback);

/// Register the separate low physical memory reserved for AMD-Vi tables.
///
/// # Safety
/// The loader's exclusive LOADER_DATA reservation remains mapped in every
/// native root, outside the guest and driver DMA ranges. Call once on the BSP
/// before remapper startup.
pub(crate) unsafe fn initialize_amd_tables(
    region: crate::handoff::Region,
) -> Result<(), MemoryError> {
    if region.bytes == 0 {
        return Ok(());
    }
    let bytes = usize::try_from(region.bytes).map_err(|_| MemoryError::InvalidSize)?;
    let cpu = crate::layout::physical(region.physical);
    // SAFETY: the caller retains this reservation and all shared mappings.
    unsafe { qualify_reserved_writeback(cpu, region.physical, bytes) }?;
    // SAFETY: unused firmware-owned RAM; the AMD table heap is its only owner.
    unsafe {
        AMD_TABLE_HEAP.initialize(cpu as *mut u8, bytes);
        AMD_TABLES.initialize(
            cpu as usize,
            region.physical,
            bytes,
            crate::layout::physical(0),
        );
    }
    Ok(())
}

// Adapted from Tyler Smith's Stambha commit 951e92cf0.
static LOW_DMA_HEAP: stela_boot_core::heap::Heap = stela_boot_core::heap::Heap::new();
/// Exclusive firmware RAM below 4 GiB for devices that reach only 32-bit
/// addresses when no IOMMU translates for them: the heap is one firmware
/// allocation of several GiB, which the firmware may place wholly above
/// 4 GiB, where `allocate_below` then finds nothing (an RTL8852CE's rings on
/// an AMD laptop with AMD-Vi left off). Linux's DMA API takes such pages from
/// ZONE_DMA32 or bounces them; this pool is that zone.
static LOW_DMA: NativeMemory = NativeMemory::new(&LOW_DMA_HEAP, cache::verify_writeback);

/// Pages of the low DMA pool to ask the firmware for, largest first: an
/// RTL8852CE's rings, RX buffers and TX pools take about 1.1 MiB, and the
/// rest leaves room for other 32-bit requesters (HDA, tg3) on a
/// passthrough domain.
const LOW_DMA_PAGES: [usize; 3] = [4096, 2048, 1024];

/// Try progressively smaller reservations below 4 GiB that fit the boot
/// memory budget.
pub(crate) fn reserve_low_dma(fits: impl Fn(u64) -> bool) -> crate::handoff::Region {
    use uefi::boot::{self, AllocateType, MemoryType};
    for pages in LOW_DMA_PAGES
        .into_iter()
        .filter(|pages| fits((pages * PAGE) as u64))
    {
        if let Ok(allocation) = boot::allocate_pages(
            AllocateType::MaxAddress(u32::MAX.into()),
            MemoryType::LOADER_DATA,
            pages,
        ) {
            let region = crate::handoff::Region {
                physical: allocation.as_ptr() as u64,
                bytes: (pages * PAGE) as u64,
            };
            crate::output::arguments(format_args!(
                "HOST LOW DMA reserved {} KiB below 4 GiB at {:#x}\n",
                region.bytes >> 10,
                region.physical
            ));
            return region;
        }
    }
    crate::output::arguments(format_args!(
        "HOST LOW DMA unavailable within the boot memory budget below 4 GiB\n"
    ));
    crate::handoff::Region::default()
}

/// Register the low DMA pool the loader reserved (`reserve_low_dma`).
///
/// # Safety
/// The loader's exclusive LOADER_DATA reservation remains mapped in every
/// native root and outside guest RAM. Call once on the BSP before any
/// driver starts.
pub(crate) unsafe fn initialize_low_dma(region: crate::handoff::Region) -> Result<(), MemoryError> {
    if region.bytes == 0 {
        return Ok(());
    }
    let bytes = usize::try_from(region.bytes).map_err(|_| MemoryError::InvalidSize)?;
    region
        .physical
        .checked_add(region.bytes)
        .filter(|&end| end <= 1 << 32)
        .ok_or(MemoryError::InvalidSize)?;
    let cpu = crate::layout::physical(region.physical);
    // SAFETY: the caller retains this reservation and all shared mappings.
    unsafe { qualify_reserved_writeback(cpu, region.physical, bytes) }?;
    // SAFETY: unused firmware-owned RAM; the low DMA heap is its only owner.
    unsafe {
        LOW_DMA_HEAP.initialize(cpu as *mut u8, bytes);
        LOW_DMA.initialize(
            cpu as usize,
            region.physical,
            bytes,
            crate::layout::physical(0),
        );
    }
    Ok(())
}

/// The low DMA pool's physical range, which passthrough domains admit; empty
/// while the pool is absent.
pub(crate) fn low_dma_range() -> Option<core::ops::Range<u64>> {
    let Registration {
        physical, bytes, ..
    } = LOW_DMA.registration();
    (bytes != 0).then(|| physical..physical + bytes as u64)
}

/// The drivers' page source: the heap, and for a request below a physical
/// end that the heap has no room under, the low DMA pool.
pub(crate) struct DriverPages;

/// The page source the engine's services hand to drivers.
pub(crate) static DRIVER_PAGES: DriverPages = DriverPages;

impl PageSource for DriverPages {
    fn page_geometry(&self) -> Result<PageGeometry, MemoryError> {
        SOURCE.page_geometry()
    }

    fn allocate(
        &self,
        bytes: usize,
        alignment: usize,
        attribute: CacheAttribute,
    ) -> Result<PinnedPages, MemoryError> {
        SOURCE.allocate(bytes, alignment, attribute)
    }

    fn allocate_below(
        &self,
        bytes: usize,
        alignment: usize,
        attribute: CacheAttribute,
        physical_end_exclusive: u64,
    ) -> Result<PinnedPages, MemoryError> {
        match SOURCE.allocate_below(bytes, alignment, attribute, physical_end_exclusive) {
            Err(MemoryError::OutOfMemory) if low_dma_range().is_some() => {
                LOW_DMA.allocate_below(bytes, alignment, attribute, physical_end_exclusive)
            }
            other => other,
        }
    }
}

/// Register the heap's CPU/physical mapping (`NativeMemory::initialize`).
///
/// # Safety
/// As `NativeMemory::initialize`, for the mount's heap.
pub unsafe fn initialize(cpu: usize, physical: u64, bytes: usize, table_offset: u64) {
    // SAFETY: the caller's contract.
    unsafe { SOURCE.initialize(cpu, physical, bytes, table_offset) }
}

/// Keeps the same boot heap generation registered through every consumer.
pub(crate) struct HeapPin {
    registration: Registration,
}

impl HeapPin {
    pub(crate) fn registration(&self) -> Registration {
        self.registration
    }
}

impl Drop for HeapPin {
    fn drop(&mut self) {
        SOURCE.release_catalog();
    }
}

/// Retain the currently registered physical heap and prevent a new generation.
///
/// # Safety
/// Caller owns the original heap/root lifetime and excludes registration
/// changes until the pin is acquired. Retiring the last pin is process-only.
pub(crate) unsafe fn pin_heap() -> Result<HeapPin, MemoryError> {
    let registration = SOURCE.registration();
    if registration.generation == 0 || registration.bytes == 0 {
        return Err(MemoryError::Unavailable);
    }
    SOURCE.retain_catalog()?;
    Ok(HeapPin { registration })
}

pub struct LinuxPages {
    heap: HeapPin,
}

// SAFETY: initialize registered the heap and its WB mapping. The catalog
// count prevents retargeting the generation; each allocation goes through the
// zeroed page source and keeps its backing until the last drop. Unsupported
// flags and contexts are refused before any heap or cache work.
unsafe impl stela_linux::pages::Provider for LinuxPages {
    fn validate(&self, request: stela_linux::pages::Request) -> Result<(), MemoryError> {
        // The pinned heap never sleeps or reclaims, so a request that may
        // not block (TTM's GFP_DMA32 page, say) gets what it asked for. Only
        // atomic context, where the heap's lock could be held, is refused.
        if SOURCE.registration().generation != self.heap.registration().generation
            || request.atomic
            || !(request.basic_ram() || request.reclaimable_ram())
        {
            return Err(MemoryError::Unavailable);
        }
        request.bytes().ok_or(MemoryError::InvalidSize)?;
        Ok(())
    }

    fn allocate(&self, request: stela_linux::pages::Request) -> Result<PinnedPages, MemoryError> {
        self.validate(request)?;
        let bytes = request.bytes().ok_or(MemoryError::InvalidSize)?;
        // The same pinned WB heap supplies Linux's unmovable fallback for
        // reclaimable-class requests. Cache accounting keeps the original class.
        SOURCE.allocate(bytes, bytes, CacheAttribute::WriteBack)
    }
}

/// Bind the registered heap to Linux's native page service.
///
/// # Safety
/// Process or deferred-work context excludes migration, root and cache
/// writers and native IRQ reentry for registration, use and retirement. The
/// heap and its mapping stay registered through every pin and any quarantined
/// catalog; the caller retires the catalog before re-registering the heap.
pub unsafe fn linux_pages()
-> Result<stela_linux::pages::Catalog<LinuxPages>, stela_linux::pages::Error> {
    // SAFETY: the caller's contract.
    let (provider, reservation) = unsafe { linux_page_source() }?;
    // SAFETY: the provider holds the catalog count for this generation.
    unsafe { stela_linux::pages::Catalog::register(provider, reservation) }
}

/// The heap generation and extent for one composed catalog. Returning the
/// reservation allocates nothing.
///
/// # Safety
/// As `linux_pages`.
pub(crate) unsafe fn linux_page_source()
-> Result<(LinuxPages, stela_linux::pages::Reservation), stela_linux::pages::Error> {
    use stela_linux::pages::{Error, Reservation};
    // SAFETY: the original catalog source owns this heap/root lifetime.
    let heap = unsafe { pin_heap() }.map_err(|error| match error {
        MemoryError::Unavailable => Error::InvalidReservation,
        _ => Error::OutOfMemory,
    })?;
    let registration = heap.registration();
    let cpu = NonNull::new(registration.cpu as *mut u8).ok_or(Error::InvalidReservation)?;
    // SAFETY: the registration records this mapping of the heap's RAM.
    let reservation =
        unsafe { Reservation::new(registration.physical, cpu, registration.bytes / PAGE)? };
    let provider = LinuxPages { heap };
    Ok((provider, reservation))
}

/// Check that a loader reservation is mapped write-back, with the decoder
/// native pages use. This allocates nothing and adds no alias.
///
/// # Safety
/// The reservation and its CPU mapping last for the boot; root, cache
/// controls and physical ownership do not change during the check.
pub(crate) unsafe fn qualify_reserved_writeback(
    cpu: u64,
    physical: u64,
    bytes: usize,
) -> Result<(), MemoryError> {
    cache::verify_writeback(cpu, physical, bytes, SOURCE.table_offset())
}

/// Exercise the registered mapping and allocator through the Linux page
/// catalog and the page source. Called by the mount's host check.
pub fn verify() -> Result<(), MemoryError> {
    // SAFETY: the host check runs in process context, with root and cache
    // writers excluded.
    let mut catalog = unsafe { linux_pages() }.map_err(|_| MemoryError::Unavailable)?;
    catalog.verify().map_err(|error| {
        crate::output::arguments(format_args!("CHECK Linux page catalog FAIL {error:?}\n"));
        MemoryError::InvalidPageList
    })?;
    catalog.try_retire().map_err(|_| MemoryError::Unavailable)?;
    SOURCE.verify()?;
    #[cfg(feature = "diagnostics")]
    frame_reuse::verify()?;
    Ok(())
}

#[cfg(feature = "diagnostics")]
mod frame_reuse;
