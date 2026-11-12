//! Canonical physical identity of the full-system loader's existing RAM.
//! Registration neither allocates nor copies guest bytes. Guest VA aliases
//! resolve through one physical view/backing and one all-writer memory system.
use crate::{
    memory::{
        access,
        backing::{AccessKind, AccessPermissions, MemoryBacking, MemoryView},
        domain::{MappingAttributes, MemoryDomain, MemorySystem},
    },
    platform::cpu::system::NativeSystemTarget,
};
use alloc::{sync::Arc, vec::Vec};
use core::ptr::NonNull;
use stela_compiler::{TargetCpu, context::memory::MemoryStatus};
use stela_contracts::gate::MemoryRange;

mod binding;
mod completion;
mod descriptor_frames;
mod transfer;
use binding::RamBinding;
pub use binding::{CpuRamBinding, RamPageKind};
pub use completion::{CanonicalRamService, ClosedDescriptorFrame, NonresidentRamFrame};
pub use transfer::{DescriptorReader, WiredRam};

pub(crate) const REGISTERED_FRAME_BYTES: usize = descriptor_frames::FRAME_BYTES;

const GRANULE: usize = 16;
const MAX_RANGES: usize = 4096;

/// One reservation, with both the guest's physical placement and the
/// host physical/CPU mapping identities. Metadata never lends guest references.
#[derive(Debug, Clone, Copy)]
pub struct ReservedSystemRamRange {
    pub source_pa: u64,
    pub host_pa: u64,
    pub host_base: NonNull<u8>,
    pub bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegistrationError {
    Geometry,
    PhysicalAlias,
    NativeTarget,
    Backing,
    FrameTracking,
    Domain(MemoryStatus),
}

struct Region {
    source_pa: u64,
    host_pa: u64,
    bytes: usize,
    backing: Arc<MemoryBacking>,
    wire_base: usize,
}

impl Region {
    fn wire_index(&self, physical: u64) -> usize {
        let frame = descriptor_frames::FRAME_BYTES as u64;
        self.wire_base + (physical / frame - self.source_pa / frame) as usize
    }
}

/// Prepared access to the exact ranges of one raw DMA export. Drop releases
/// an unpublished preparation and keeps the exact frames of a possible
/// publication. The registration must outlive every published external use.
#[must_use]
pub struct ExternalDmaLoan<'a> {
    binding: RamBinding<'a>,
    ranges: &'a [MemoryRange],
    _pins: Option<descriptor_frames::RangePin<'a>>,
    publication_started: bool,
}

impl ExternalDmaLoan<'_> {
    /// Mark before the first possible externally visible mapping effect.
    pub fn begin_publication(&mut self) {
        self.publication_started = true;
    }

    /// Permit returning the temporary pins.
    ///
    /// # Safety
    /// No external reference escaped, and all CPU aliases and their cached
    /// translations have been removed. A published DMA buffer needs its own
    /// final-reference and DMA teardown evidence instead.
    pub unsafe fn cancel_publication(&mut self) {
        self.publication_started = false;
    }
}

impl Drop for ExternalDmaLoan<'_> {
    fn drop(&mut self) {
        let ram = self.binding.ram;
        let Some(frames) = ram.descriptor_frames.as_ref() else {
            return;
        };
        if !self.publication_started {
            return;
        }
        // The scoped pins remain held until the exact bits are visible.
        for range in self.ranges {
            let mut address = range.address;
            let mut remaining = range.bytes;
            while remaining != 0 {
                let (region, offset) = ram.region(address).expect("validated DMA range");
                let bytes = remaining.min((region.bytes - offset) as u64);
                frames.wire(
                    region.wire_index(address),
                    region.wire_index(address + bytes - 1),
                );
                address += bytes;
                remaining -= bytes;
            }
        }
    }
}

/// The only canonical backing identities for this reserved physical RAM.
/// Every guest/native/host/table-walker writer must use its memory system;
/// direct RAM and DMA writers are excluded until separately joined.
pub struct RegisteredSystemRam {
    regions: Vec<Region>,
    domain: Arc<MemoryDomain>,
    system: Arc<MemorySystem>,
    target: TargetCpu,
    erg_bytes: u64,
    descriptor_frames: Option<descriptor_frames::DescriptorFrames>,
}

impl RegisteredSystemRam {
    /// Bind the loader's real reservations without modifying historical bytes.
    ///
    /// # Safety
    /// Each CPU mapping is coherent Normal-WB RAM with stable readable/writable
    /// host rights, exactly backed by the given host physical extent. `owner`
    /// keeps these reservations/mappings through their last backing,
    /// every native loan, recovery and final source-PC settlement. Ranges are
    /// disjoint from Rust references, Stela code/data/stacks and other consumers.
    /// This transfers all access discipline: every writer, including aliases,
    /// AF/DBM updates and host services, must participate in this one system.
    /// No firmware, SMI, external DMA or unjoined direct writer may access it.
    /// Called before any admitted concurrent execution with interrupts disabled
    /// at CPL0. Per-worker effective guest controls and native target
    /// admission are additional requirements; registration alone grants neither.
    pub unsafe fn register<O: Send + Sync + 'static>(
        ranges: &[ReservedSystemRamRange],
        owner: Arc<O>,
        erg_bytes: u64,
        stripes: usize,
    ) -> Result<Self, RegistrationError> {
        // SAFETY: forwarded unchanged reservation/all-writer contract.
        unsafe { Self::register_inner(ranges, owner, erg_bytes, stripes, None) }
    }

    /// Register with a conservative common-worker target ceiling. A limit
    /// cannot promote the BSP's admitted capabilities or enable DIT,
    /// AVX state or another unknown native recipe. Every destination still
    /// requires its own live target and exact-control proof before entry.
    ///
    /// # Safety
    /// The complete reservation, mapping and all-writer contract of `register`
    /// applies. A supplied limit is policy only, never execution permission.
    pub unsafe fn register_with_target_limit<O: Send + Sync + 'static>(
        ranges: &[ReservedSystemRamRange],
        owner: Arc<O>,
        erg_bytes: u64,
        stripes: usize,
        limit: TargetCpu,
    ) -> Result<Self, RegistrationError> {
        // SAFETY: forwarded unchanged reservation/all-writer contract.
        unsafe { Self::register_inner(ranges, owner, erg_bytes, stripes, Some(limit)) }
    }

    unsafe fn register_inner<O: Send + Sync + 'static>(
        ranges: &[ReservedSystemRamRange],
        owner: Arc<O>,
        erg_bytes: u64,
        stripes: usize,
        limit: Option<TargetCpu>,
    ) -> Result<Self, RegistrationError> {
        validate(ranges)?;
        let authority =
            NativeSystemTarget::capture().map_err(|_| RegistrationError::NativeTarget)?;
        let target = limit.map_or(authority.target(), |limit| {
            crate::memory::domain::native_target::restrict(authority.target(), limit)
        });
        // SAFETY: validate checked range geometry. The caller keeps initialized
        // pinned mappings and all writers; bindings clones owner into each backing.
        let (regions, views, system) = unsafe { bindings(ranges, owner, erg_bytes, stripes) }?;
        let domain =
            MemoryDomain::new_native_system_ram(Arc::clone(&system), views, &authority, target)
                .map_err(RegistrationError::Domain)?;
        Ok(Self {
            regions,
            domain,
            system,
            target,
            erg_bytes,
            descriptor_frames: None,
        })
    }

    /// Enable 16 KiB frame admission before publishing this RAM owner.
    /// The physical backing and guest-visible geometry stay fixed.
    pub fn enable_descriptor_frame_leases(&mut self) -> Result<(), RegistrationError> {
        if self.descriptor_frames.is_some() {
            return Ok(());
        }
        let mut count = 0usize;
        for region in &mut self.regions {
            region.wire_base = count;
            let last = region.wire_index(region.source_pa + region.bytes as u64 - 1);
            count = last
                .checked_add(1)
                .ok_or(RegistrationError::FrameTracking)?;
        }
        self.descriptor_frames = Some(descriptor_frames::DescriptorFrames::new(count)?);
        Ok(())
    }

    fn pin_descriptor(
        &self,
        physical: u64,
    ) -> Result<Option<descriptor_frames::DescriptorPin<'_>>, MemoryStatus> {
        let Some(frames) = &self.descriptor_frames else {
            return Ok(None);
        };
        let pin = frames.try_pin(physical).ok_or(MemoryStatus::Domain)?;
        self.check_frame_residency(physical, 1)?;
        Ok(Some(pin))
    }

    fn pin_range(
        &self,
        physical: u64,
        bytes: usize,
    ) -> Result<Option<descriptor_frames::RangePin<'_>>, MemoryStatus> {
        let Some(frames) = &self.descriptor_frames else {
            return Ok(None);
        };
        if bytes == 0 {
            return Ok(None);
        }
        let last = physical
            .checked_add(bytes as u64 - 1)
            .ok_or(MemoryStatus::Translation)?;
        let pin = frames
            .try_pin_range(physical, last)
            .ok_or(MemoryStatus::Domain)?;
        self.check_frame_residency(physical, bytes)?;
        Ok(Some(pin))
    }

    /// Observe the first nonresident 16 KiB frame intersecting a range.
    /// Unregistered gaps are ignored. An empty or overflowing range has no
    /// result. This observation does not pin the range or validate its access.
    pub fn nonresident_frame(&self, physical: u64, bytes: usize) -> Option<u64> {
        let frames = self.descriptor_frames.as_ref()?;
        if !frames.has_nonresident() {
            return None;
        }
        let last = physical.checked_add(u64::try_from(bytes.checked_sub(1)?).ok()?)?;
        let frame_bytes = descriptor_frames::FRAME_BYTES as u64;
        let first_region = self
            .regions
            .partition_point(|region| region.source_pa + region.bytes as u64 <= physical);
        for region in &self.regions[first_region..] {
            if region.source_pa > last {
                break;
            }
            let start = physical.max(region.source_pa);
            let end = last.min(region.source_pa + region.bytes as u64 - 1);
            let first_index = region.wire_index(start);
            let last_index = region.wire_index(end);
            if let Some(index) = (first_index..=last_index).find(|&i| frames.is_nonresident(i)) {
                let frame = start / frame_bytes + (index - first_index) as u64;
                return Some(frame * frame_bytes);
            }
        }
        None
    }

    fn check_frame_residency(&self, physical: u64, bytes: usize) -> Result<(), MemoryStatus> {
        if self.nonresident_frame(physical, bytes).is_some() {
            return Err(MemoryStatus::Nonresident);
        }
        Ok(())
    }

    /// Source/host physical extents. Metadata does not lend host
    /// pointers or authorize another RAM store/direct native writer.
    pub fn placements(&self) -> impl Iterator<Item = (u64, u64, usize)> + '_ {
        self.regions
            .iter()
            .map(|region| (region.source_pa, region.host_pa, region.bytes))
    }
    pub fn reservation_granule(&self) -> u64 {
        self.erg_bytes
    }
    pub fn target(&self) -> TargetCpu {
        self.target
    }
    /// Current generation of the canonical RAM domain.
    /// A failed or retired domain refuses rather than supplying stale identity.
    pub fn generation(&self) -> Result<u64, MemoryStatus> {
        self.domain.generation()
    }
    /// Keep a real native-execution loan in reference tests so adapters can
    /// verify that physical service acquisition refuses while execution is live.
    #[cfg(any(test, feature = "test-support"))]
    pub fn reference_execution_loan(
        &self,
    ) -> Result<crate::memory::domain::MemoryExecutionLoan, MemoryStatus> {
        self.domain.execution_loan()
    }
    pub(crate) fn domain(&self) -> &Arc<MemoryDomain> {
        &self.domain
    }

    fn region(&self, physical: u64) -> Option<(&Region, usize)> {
        self.region_at(physical, None)
            .map(|(_, region, offset)| (region, offset))
    }

    fn region_at(&self, physical: u64, hint: Option<usize>) -> Option<(usize, &Region, usize)> {
        let at = |index| {
            let region: &Region = self.regions.get(index)?;
            let offset = usize::try_from(physical.checked_sub(region.source_pa)?).ok()?;
            (offset < region.bytes).then_some((index, region, offset))
        };
        if let Some(found) = hint.and_then(at) {
            return Some(found);
        }
        let index = self
            .regions
            .partition_point(|region| region.source_pa <= physical);
        at(index.checked_sub(1)?)
    }

    /// Observe initialized physical RAM through native scalar reads.
    /// Instruction words and descriptors do not use racing Rust byte copies.
    /// Refused whole-range admission leaves the caller's buffer unchanged.
    ///
    /// # Safety
    /// The output is private/disjoint from this registered RAM, live native
    /// frames and host metadata. The kept source mapping and all-writer
    /// contract remain valid. This does not make a multiword snapshot atomic;
    /// instruction/cache publication still requires its own stopped-cohort
    /// protocol and immutable source-byte identity at native preparation.
    pub unsafe fn read_physical(
        &self,
        physical: u64,
        output: &mut [u8],
    ) -> Result<(), MemoryStatus> {
        // SAFETY: the caller keeps the same private output and live owner.
        unsafe { self.read_physical_progress(physical, output) }.map_err(|fault| fault.status)
    }

    /// Read through the same canonical scalar access path, keeping the exact
    /// output prefix if a later access encounters a terminal owner failure.
    ///
    /// # Safety
    /// The `read_physical` private-output and live-registration contract applies.
    pub unsafe fn read_physical_progress(
        &self,
        physical: u64,
        output: &mut [u8],
    ) -> Result<(), crate::memory::domain::PublicationFault> {
        let _pin = self.pin_range(physical, output.len()).map_err(|status| {
            crate::memory::domain::PublicationFault {
                status,
                completed: 0,
            }
        })?;
        // SAFETY: same private output and live backing, with admission held.
        unsafe { self.read_physical_unpinned(physical, output) }
    }

    /// The caller keeps registered admission or the exact closed frame.
    unsafe fn read_physical_unpinned(
        &self,
        physical: u64,
        output: &mut [u8],
    ) -> Result<(), crate::memory::domain::PublicationFault> {
        use crate::memory::domain::PublicationFault;
        // Most reads are a descriptor or a block's source, in one region.
        let mut current = self.region(physical);
        let admitted = match current {
            Some((region, offset)) if output.len() <= region.bytes - offset => {
                self.domain.check_execution()
            }
            _ => self.preflight(physical, output.len()),
        };
        admitted.map_err(|status| PublicationFault {
            status,
            completed: 0,
        })?;
        let mut done = 0;
        while done < output.len() {
            let fault = |status| PublicationFault {
                status,
                completed: done,
            };
            let (region, offset) = current.ok_or_else(|| fault(MemoryStatus::Domain))?;
            let count = (output.len() - done).min(region.bytes - offset);
            let pointer = region
                .backing
                .host_pointer(offset, count, AccessKind::Read)
                .map_err(|_| fault(MemoryStatus::Domain))?;
            // Registration keeps this region and its backing envelope immutable.
            // Reuse only that checked envelope within this call; each
            // scalar read still observes the canonical, concurrently mutable RAM.
            let mut advanced = 0;
            while advanced < count {
                // SAFETY: this nonempty suffix stays inside the checked envelope.
                let part = unsafe { pointer.as_ptr().add(advanced) };
                if (part as usize).is_multiple_of(8) && count - advanced >= 8 {
                    let words = (count - advanced) & !7;
                    // SAFETY: as below, for these aligned doublewords.
                    let read =
                        unsafe { access::read_doublewords(part, &mut output[done..done + words]) };
                    advanced += read;
                    done += read;
                    if read != 0 {
                        continue;
                    }
                }
                let remaining = count - advanced;
                let width = [8, 4, 2, 1]
                    .into_iter()
                    .find(|width| remaining >= *width && (part as usize).is_multiple_of(*width))
                    .expect("a byte fits the admitted range");
                // SAFETY: a pinned exact aligned scalar part, checked baseline
                // host, no Rust reference into concurrently mutable guest storage.
                let value = unsafe { access::read_part(part, width, TargetCpu::Baseline) }
                    .map_err(|_| PublicationFault {
                        status: MemoryStatus::Domain,
                        completed: done,
                    })?;
                output[done..done + width].copy_from_slice(&value.to_le_bytes()[..width]);
                advanced += width;
                done += width;
            }
            if done == output.len() {
                break;
            }
            current = self.region(physical + done as u64);
        }
        Ok(())
    }

    /// Store `new` at `physical` if RAM there holds `expected`, in one locked
    /// compare-and-exchange of the smallest aligned quantity holding both;
    /// whether it stored. When not, `expected` receives what RAM held. The
    /// access lies within one aligned 16-byte block; bytes of the quantity
    /// outside it that change meanwhile are compared again, never stored.
    ///
    /// # Safety
    /// The `read_physical` live-registration contract applies; this is a
    /// writer of the canonical RAM.
    pub unsafe fn compare_exchange_physical(
        &self,
        physical: u64,
        expected: &mut [u8],
        new: &[u8],
    ) -> Result<bool, MemoryStatus> {
        let bytes = expected.len();
        let offset = (physical % GRANULE as u64) as usize;
        if bytes == 0 || bytes != new.len() || offset + bytes > GRANULE {
            return Err(MemoryStatus::Alignment);
        }
        self.preflight(physical, bytes)?;
        let _pin = self.pin_range(physical, bytes)?;
        let width = [1, 2, 4, 8, 16]
            .into_iter()
            .find(|&width| offset / width == (offset + bytes - 1) / width)
            .expect("a 16-byte block holds the access");
        let start = physical - (offset % width) as u64;
        let (region, at) = self.region(start).ok_or(MemoryStatus::Domain)?;
        let pointer = region
            .backing
            .host_pointer(at, width, AccessKind::Write)
            .map_err(|_| MemoryStatus::Permission)?
            .as_ptr();
        let shift = (offset % width) * 8;
        let mask = if bytes == 16 {
            u128::MAX
        } else {
            ((1u128 << (bytes * 8)) - 1) << shift
        };
        let place = |value: &[u8]| {
            let mut wide = [0u8; 16];
            wide[..bytes].copy_from_slice(value);
            u128::from_le_bytes(wide) << shift
        };
        let (expected_bits, new_bits) = (place(expected), place(new));
        // SAFETY: a pinned aligned quantity inside the checked envelope; the
        // caller's contract covers concurrent canonical writers.
        let mut current = unsafe { access::read_part(pointer, width, self.target) }
            .map_err(|_| MemoryStatus::Domain)?;
        loop {
            let compare = current & !mask | expected_bits;
            // SAFETY: as above, one locked exchange of the same quantity.
            let old = unsafe {
                access::compare_exchange_part(
                    pointer,
                    width,
                    compare,
                    current & !mask | new_bits,
                    self.target,
                )
            }
            .map_err(|_| MemoryStatus::Domain)?;
            if old == compare {
                return Ok(true);
            }
            if old & mask != expected_bits {
                expected.copy_from_slice(&((old & mask) >> shift).to_le_bytes()[..bytes]);
                return Ok(false);
            }
            current = old;
        }
    }

    /// Check residency, physical geometry and kept backing permissions for a
    /// DMA descriptor. This does not pin the range for later use.
    pub fn check_dma_range(
        &self,
        physical: u64,
        bytes: usize,
        write: bool,
    ) -> Result<(), MemoryStatus> {
        self.check_physical(physical, bytes)?;
        self.check_dma_permissions(physical, bytes, write)
    }

    fn check_dma_permissions(
        &self,
        physical: u64,
        bytes: usize,
        write: bool,
    ) -> Result<(), MemoryStatus> {
        let mut done = 0;
        while done < bytes {
            let (region, offset) = self
                .region(physical + done as u64)
                .ok_or(MemoryStatus::Translation)?;
            let count = (bytes - done).min(region.bytes - offset);
            region
                .backing
                .host_pointer(
                    offset,
                    count,
                    if write {
                        AccessKind::Write
                    } else {
                        AccessKind::Read
                    },
                )
                .map_err(|_| MemoryStatus::Permission)?;
            done += count;
        }
        Ok(())
    }

    /// Validate each range and keep the union of its temporary pins.
    /// The loan supplies placement metadata; it grants no DMA permission
    /// and gives no completion proof.
    pub fn prepare_external_dma<'a>(
        &'a self,
        ranges: &'a [MemoryRange],
    ) -> Result<ExternalDmaLoan<'a>, MemoryStatus> {
        let binding = self.binding()?;
        let mut stripes = 0;
        for range in ranges.iter().filter(|range| range.bytes != 0) {
            let bytes = usize::try_from(range.bytes).map_err(|_| MemoryStatus::Translation)?;
            self.preflight_geometry(range.address, bytes)?;
            let last = range
                .address
                .checked_add(range.bytes - 1)
                .ok_or(MemoryStatus::Translation)?;
            stripes |= descriptor_frames::DescriptorFrames::range_stripes(range.address, last);
        }
        let pins = self
            .descriptor_frames
            .as_ref()
            .map(|frames| frames.try_pin_stripes(stripes).ok_or(MemoryStatus::Domain))
            .transpose()?;
        binding.validate()?;
        for range in ranges.iter().filter(|range| range.bytes != 0) {
            let bytes = usize::try_from(range.bytes).map_err(|_| MemoryStatus::Translation)?;
            binding.check_range(range.address, bytes)?;
            self.check_dma_permissions(range.address, bytes, true)?;
        }
        Ok(ExternalDmaLoan {
            binding,
            ranges,
            _pins: pins,
            publication_started: false,
        })
    }

    pub(crate) fn check_physical(&self, physical: u64, bytes: usize) -> Result<(), MemoryStatus> {
        self.binding()?.check_range(physical, bytes)
    }

    fn preflight(&self, physical: u64, bytes: usize) -> Result<(), MemoryStatus> {
        self.domain.check_execution()?;
        self.preflight_geometry(physical, bytes)
    }

    fn preflight_geometry(&self, physical: u64, bytes: usize) -> Result<(), MemoryStatus> {
        if bytes == 0 {
            return Ok(());
        }
        physical
            .checked_add(bytes as u64 - 1)
            .ok_or(MemoryStatus::Translation)?;
        let mut done = 0;
        while done < bytes {
            let (region, offset) = self
                .region(physical + done as u64)
                .ok_or(MemoryStatus::Translation)?;
            done += (bytes - done).min(region.bytes - offset);
        }
        Ok(())
    }
}

fn validate(ranges: &[ReservedSystemRamRange]) -> Result<(), RegistrationError> {
    if ranges.is_empty() || ranges.len() > MAX_RANGES {
        return Err(RegistrationError::Geometry);
    }
    for (index, range) in ranges.iter().enumerate() {
        if range.bytes == 0
            || range.bytes > isize::MAX as usize
            || !range.bytes.is_multiple_of(GRANULE)
            || !range.source_pa.is_multiple_of(GRANULE as u64)
            || !range.host_pa.is_multiple_of(GRANULE as u64)
            || !(range.host_base.as_ptr() as usize).is_multiple_of(GRANULE)
            || range.source_pa.checked_add(range.bytes as u64).is_none()
            || range.host_pa.checked_add(range.bytes as u64).is_none()
            || (range.host_base.as_ptr() as usize)
                .checked_add(range.bytes)
                .is_none()
        {
            return Err(RegistrationError::Geometry);
        }
        for other in &ranges[..index] {
            let overlap = |left: u64, left_bytes: usize, right: u64, right_bytes: usize| {
                left < right + right_bytes as u64 && right < left + left_bytes as u64
            };
            if overlap(range.source_pa, range.bytes, other.source_pa, other.bytes)
                || overlap(range.host_pa, range.bytes, other.host_pa, other.bytes)
                || overlap(
                    range.host_base.as_ptr() as u64,
                    range.bytes,
                    other.host_base.as_ptr() as u64,
                    other.bytes,
                )
            {
                return Err(RegistrationError::PhysicalAlias);
            }
        }
    }
    Ok(())
}

type Views = Vec<(Arc<MemoryView>, MappingAttributes)>;
/// # Safety
/// Caller validated geometry and keeps WB reservations under the full
/// register contract. This never reads or changes guest bytes or creates a
/// caller-selectable native admission shortcut.
unsafe fn bindings<O: Send + Sync + 'static>(
    ranges: &[ReservedSystemRamRange],
    owner: Arc<O>,
    erg_bytes: u64,
    stripes: usize,
) -> Result<(Vec<Region>, Views, Arc<MemorySystem>), RegistrationError> {
    let system = MemorySystem::new(erg_bytes, stripes).map_err(RegistrationError::Domain)?;
    let permissions = AccessPermissions {
        readable: true,
        writable: true,
        executable: true,
    };
    let mut regions = Vec::with_capacity(ranges.len());
    let mut views = Vec::with_capacity(ranges.len());
    for range in ranges {
        // SAFETY: the exact reservation contract is the caller's.
        let backing = unsafe {
            MemoryBacking::new(
                range.host_base,
                range.bytes,
                permissions,
                Arc::clone(&owner),
            )
        }
        .map_err(|_| RegistrationError::Backing)?;
        let view = MemoryView::new(
            Arc::clone(&backing),
            range.source_pa,
            0,
            range.bytes,
            permissions,
        )
        .map_err(|_| RegistrationError::Backing)?;
        views.push((
            view,
            MappingAttributes {
                tagged: false,
                shareable: true,
            },
        ));
        regions.push(Region {
            source_pa: range.source_pa,
            host_pa: range.host_pa,
            bytes: range.bytes,
            backing,
            wire_base: 0,
        });
    }
    regions.sort_unstable_by_key(|region| region.source_pa);
    Ok((regions, views, system))
}

/// Source-MMU fixtures use real pinned bytes/native scalar transfers but the
/// legacy validation domain. This cannot construct production CPL0 entry permission.
///
/// # Safety
/// Every source and host range must describe initialized, pinned mappings
/// by `owner`; no Rust references or external writers may alias the
/// registered bytes outside the canonical memory synchronization protocol.
#[cfg(any(test, feature = "test-support"))]
pub unsafe fn fixture<O: Send + Sync + 'static>(
    ranges: &[ReservedSystemRamRange],
    owner: Arc<O>,
) -> RegisteredSystemRam {
    // SAFETY: the caller upholds this function's contract.
    unsafe { fixture_for(ranges, owner, TargetCpu::Baseline) }
}

/// [`fixture`] on an admitted `target`, for parts only some hosts transfer
/// atomically.
///
/// # Safety
/// As [`fixture`].
#[cfg(any(test, feature = "test-support"))]
pub unsafe fn fixture_for<O: Send + Sync + 'static>(
    ranges: &[ReservedSystemRamRange],
    owner: Arc<O>,
    target: TargetCpu,
) -> RegisteredSystemRam {
    validate(ranges).unwrap();
    // SAFETY: The fixture caller keeps initialized pinned mappings and excludes
    // external writers. Checked ranges and cloned owner pins outlive every backing.
    let (regions, views, system) = unsafe { bindings(ranges, owner, 64, 8) }.unwrap();
    let domain = MemoryDomain::new(Arc::clone(&system), views, target).unwrap();
    RegisteredSystemRam {
        regions,
        domain,
        system,
        target,
        erg_bytes: 64,
        descriptor_frames: None,
    }
}

#[cfg(test)]
mod tests;
