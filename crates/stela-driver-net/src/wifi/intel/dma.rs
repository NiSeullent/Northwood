// SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause
// Copyright (C) 2017 Intel Deutschland GmbH
// Copyright (C) 2018-2026 Intel Corporation
//! Source-ordered firmware DMA using kept native RAM and device-domain
//! leases. No CPU/physical pointer becomes an IOVA. The ROM context and RX/TX
//! rings are separate allocations owned by the eventual transport.
use super::{
    context::{Images, Region},
    firmware::{Section, SelfLoadPlan},
};
use crate::bus::dma::{
    mapping::{Direction, DmaDomain, DmaError, DmaMapping},
    memory::{CacheAttribute, CpuMapping, MemoryError, PageSource},
};
use alloc::{sync::Arc, vec::Vec};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    Dma(DmaError),
    Memory(MemoryError),
    PayloadSize,
    AllocationContract,
    BoundaryRetries,
    State,
}
impl From<DmaError> for Error {
    fn from(value: DmaError) -> Self {
        Self::Dma(value)
    }
}
impl From<MemoryError> for Error {
    fn from(value: MemoryError) -> Self {
        Self::Memory(value)
    }
}

/// The largest allocation: a 1024-entry descriptor ring.
const MAX_CHUNK_BYTES: usize = 1024 * 256;

pub(in crate::wifi::intel) struct Chunk {
    mapping: Option<DmaMapping>,
    cpu: CpuMapping,
    region: Region,
    exposed: bool,
}
impl Chunk {
    pub(super) fn allocate(
        source: &dyn PageSource,
        domain: Arc<dyn DmaDomain>,
        data: &[u8],
        direction: Direction,
    ) -> Result<Self, Error> {
        Self::allocate_bounded(source, domain, data, direction, u64::MAX)
    }

    pub(super) fn allocate_bounded(
        source: &dyn PageSource,
        domain: Arc<dyn DmaDomain>,
        data: &[u8],
        direction: Direction,
        address_limit: u64,
    ) -> Result<Self, Error> {
        if data.is_empty() || data.len() > MAX_CHUNK_BYTES {
            return Err(Error::PayloadSize);
        }
        // Keep rejected allocations reserved while trying another address, as
        // the donor does. Releasing first could hand out the same bad IOVA.
        let mut rejected = Vec::new();
        for _ in 0..3 {
            let pages = source.allocate(data.len(), 4096, CacheAttribute::WriteBack)?;
            if pages.logical_bytes() != data.len()
                || pages.cache_attribute() != CacheAttribute::WriteBack
            {
                return Err(Error::AllocationContract);
            }
            let cpu = pages.cpu_mapping(CacheAttribute::WriteBack)?;
            let mapping = DmaMapping::map(domain.clone(), pages, direction, address_limit)?;
            let region = Region {
                address: mapping.address()?,
                bytes: data.len(),
            };
            let chunk = Self {
                mapping: Some(mapping),
                cpu,
                region,
                exposed: false,
            };
            if (region.address >> 32) != ((region.address + data.len() as u64) >> 32) {
                rejected.push(chunk);
                continue;
            }
            let pointer = chunk.cpu.address(0, data.len())?;
            // SAFETY: this new, exclusive allocation has never been submitted
            // to the device; the source and bounded CPU mapping do not alias.
            unsafe {
                core::ptr::copy_nonoverlapping(data.as_ptr(), pointer.as_ptr(), data.len());
            }
            chunk
                .mapping
                .as_ref()
                .expect("new mapping")
                .sync_for_device(0..data.len())?;
            return Ok(chunk);
        }
        Err(Error::BoundaryRetries)
    }
    pub(super) fn region(&self) -> Region {
        self.region
    }
    pub(super) fn expose(&mut self) {
        self.exposed = true;
    }
    /// Shared status observation checked by the actual mapping backend.
    pub(super) fn observe_u32(&self, offset: usize) -> Result<u32, Error> {
        Ok(self
            .mapping
            .as_ref()
            .ok_or(Error::State)?
            .read_coherent_u32(offset)?)
    }
    /// Shared producer/descriptor access only for the owning transport's
    /// explicitly checked coherent requester. Never call on streaming DMA.
    pub(super) unsafe fn read16_coherent(&self, offset: usize) -> Result<u16, Error> {
        if !offset.is_multiple_of(2) {
            return Err(Error::AllocationContract);
        }
        let pointer = self.cpu.address(offset, 2)?.as_ptr().cast::<u16>();
        // SAFETY: covered by this function's own safety contract.
        Ok(u16::from_le(unsafe { pointer.read_volatile() }))
    }
    pub(super) unsafe fn read32_coherent(&self, offset: usize) -> Result<u32, Error> {
        if !offset.is_multiple_of(4) {
            return Err(Error::AllocationContract);
        }
        let pointer = self.cpu.address(offset, 4)?.as_ptr().cast::<u32>();
        // SAFETY: covered by this function's own safety contract.
        Ok(u32::from_le(unsafe { pointer.read_volatile() }))
    }
    /// CPU exclusively owns this range until its next protocol doorbell.
    pub(super) unsafe fn write_owned(&self, offset: usize, bytes: &[u8]) -> Result<(), Error> {
        let pointer = self.cpu.address(offset, bytes.len())?;
        // SAFETY: covered by this function's own safety contract.
        unsafe {
            core::ptr::copy_nonoverlapping(bytes.as_ptr(), pointer.as_ptr(), bytes.len());
        }
        self.mapping
            .as_ref()
            .ok_or(Error::State)?
            .sync_for_device(offset..offset + bytes.len())?;
        Ok(())
    }
    /// Device completion has ended all device access to this range,
    /// transferring exclusive ownership back to the CPU before further writes.
    /// Other live descriptor slots remain outside this completed range.
    pub(super) unsafe fn complete_for_cpu(
        &self,
        range: core::ops::Range<usize>,
    ) -> Result<(), Error> {
        self.cpu.address(range.start, range.len())?;
        // SAFETY: covered by this function's own safety contract.
        unsafe {
            self.mapping
                .as_ref()
                .ok_or(Error::State)?
                .sync_for_cpu(range)?;
        }
        Ok(())
    }
    /// The entire allocation is exclusively CPU-owned after real completion;
    /// return it to device visibility before a new descriptor publication.
    pub(super) unsafe fn ready_for_device(&self) -> Result<(), Error> {
        self.mapping
            .as_ref()
            .ok_or(Error::State)?
            .sync_for_device(0..self.region.bytes)?;
        Ok(())
    }
    /// Device completion must transfer this entire payload to the CPU.
    pub(super) unsafe fn snapshot_completed(&self) -> Result<Vec<u8>, Error> {
        let mapping = self.mapping.as_ref().ok_or(Error::State)?;
        // SAFETY: covered by this function's own safety contract.
        unsafe {
            mapping.sync_for_cpu(0..self.region.bytes)?;
        }
        let pointer = self.cpu.address(0, self.region.bytes)?;
        let mut bytes = alloc::vec![0;self.region.bytes];
        // SAFETY: covered by this function's own safety contract.
        unsafe {
            core::ptr::copy_nonoverlapping(pointer.as_ptr(), bytes.as_mut_ptr(), bytes.len());
        }
        Ok(bytes)
    }
    /// # Safety
    /// Hardware no longer accesses this allocation and submissions/IRQ work
    /// referencing it are drained. Failed invalidations keep the lease.
    pub(super) unsafe fn release(&mut self) -> Result<(), DmaError> {
        if let Some(mapping) = self.mapping.as_mut() {
            // SAFETY: hardware quiescence is established by this method's caller.
            unsafe {
                mapping.unmap()?;
            }
            self.mapping = None;
        }
        Ok(())
    }
}
impl Drop for Chunk {
    fn drop(&mut self) {
        if !self.exposed {
            // SAFETY: no address has escaped to a hardware submission. Failure
            // leaves DmaMapping's own Drop to quarantine the pins and domain.
            let _ = unsafe { self.release() };
        }
    }
}

#[derive(Default)]
struct Bank {
    chunks: Vec<Chunk>,
    regions: Vec<Region>,
}
impl Bank {
    fn prepare(
        sections: &[Section<'_>],
        source: &dyn PageSource,
        domain: &Arc<dyn DmaDomain>,
        direction: Direction,
    ) -> Result<Self, Error> {
        let mut bank = Self::default();
        for section in sections {
            let chunk = Chunk::allocate(source, domain.clone(), section.data, direction)?;
            bank.regions.push(chunk.region);
            bank.chunks.push(chunk);
        }
        Ok(bank)
    }
    fn expose(&mut self) {
        for chunk in &mut self.chunks {
            chunk.exposed = true;
        }
    }
    unsafe fn release(&mut self) -> Result<(), DmaError> {
        for chunk in &mut self.chunks {
            // SAFETY: the caller establishes quiescence of the complete bank.
            unsafe {
                chunk.release()?;
            }
        }
        self.chunks.clear();
        self.regions.clear();
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    Prepared,
    Exposed,
    Alive,
    Stopped,
}

/// Keeps every uploaded section's native RAM pin and isolated DMA domain.
/// Dropping after exposure quarantines potentially active mappings. Explicit
/// release requires device completion, never a timer or guessed ALIVE.
pub struct MappedFirmware {
    lmac: Bank,
    umac: Bank,
    paging: Bank,
    state: State,
}
impl MappedFirmware {
    /// # Safety
    /// The source returns fresh, exclusive allocations without CPU/device
    /// aliases. No other source user may write or submit these returned pins.
    pub unsafe fn prepare(
        plan: &SelfLoadPlan<'_>,
        source: &dyn PageSource,
        domain: Option<Arc<dyn DmaDomain>>,
    ) -> Result<Self, Error> {
        let domain = domain.ok_or(DmaError::IommuUnavailable)?;
        Ok(Self {
            lmac: Bank::prepare(plan.lmac(), source, &domain, Direction::ToDevice)?,
            umac: Bank::prepare(plan.umac(), source, &domain, Direction::ToDevice)?,
            paging: Bank::prepare(plan.paging(), source, &domain, Direction::Bidirectional)?,
            state: State::Prepared,
        })
    }
    /// Internal preparation only: these addresses must not reach hardware
    /// until expose_for_context marks all leases potentially active.
    pub(super) fn prepared_regions(&self) -> Result<Images<'_>, Error> {
        if self.state != State::Prepared {
            return Err(Error::State);
        }
        Ok(Images {
            lmac: &self.lmac.regions,
            umac: &self.umac.regions,
            paging: &self.paging.regions,
        })
    }
    /// Transfer all sections to potentially active device ownership *before*
    /// serializing/publishing their addresses in a context. Call at most once.
    /// All section writes have passed the real domain's visibility operation.
    pub fn expose_for_context(&mut self) -> Result<Images<'_>, Error> {
        if self.state != State::Prepared {
            return Err(Error::State);
        }
        self.lmac.expose();
        self.umac.expose();
        self.paging.expose();
        self.state = State::Exposed;
        Ok(Images {
            lmac: &self.lmac.regions,
            umac: &self.umac.regions,
            paging: &self.paging.regions,
        })
    }
    /// The donor releases LMAC/UMAC after ALIVE, while paging remains resident.
    /// A failed invalidation keeps unreleased mappings for explicit retry.
    ///
    /// # Safety
    /// A validated ALIVE response for this exact boot/firmware epoch
    /// proves ROM self-load no longer reads LMAC/UMAC. No outstanding context
    /// submission or IRQ worker may still reference these allocations.
    pub unsafe fn release_after_alive(&mut self) -> Result<(), Error> {
        if self.state != State::Exposed {
            return Err(Error::State);
        }
        // SAFETY: caller supplied the donor's image-read completion.
        unsafe {
            self.lmac.release()?;
            self.umac.release()?;
        }
        self.state = State::Alive;
        Ok(())
    }
    /// # Safety
    /// Device DMA is stopped/acknowledged and all submissions/IRQ workers are
    /// drained for this boot epoch, including persistent firmware paging reads.
    pub unsafe fn release_after_stop(&mut self) -> Result<(), Error> {
        if self.state == State::Stopped {
            return Err(Error::State);
        }
        // SAFETY: caller establishes hardware quiescence for every section.
        unsafe {
            self.lmac.release()?;
            self.umac.release()?;
            self.paging.release()?;
        }
        self.state = State::Stopped;
        Ok(())
    }
}

#[cfg(test)]
pub(in crate::wifi::intel) mod tests {
    extern crate std;
    use super::*;
    use crate::bus::dma::{
        mapping::DomainMapping,
        memory::{PageBacking, PinnedPages},
    };
    use alloc::{
        alloc::{alloc_zeroed, dealloc},
        boxed::Box,
        collections::BTreeMap,
    };
    /// What the device would read from `chunk`.
    pub(in crate::wifi::intel) fn read_back(chunk: &Chunk, offset: usize, bytes: usize) -> Vec<u8> {
        let pointer = chunk.cpu.address(offset, bytes).unwrap();
        let mut copy = alloc::vec![0; bytes];
        // SAFETY: model memory no concurrent writer holds.
        unsafe {
            core::ptr::copy_nonoverlapping(pointer.as_ptr(), copy.as_mut_ptr(), bytes);
        }
        copy
    }
    use core::{
        alloc::Layout,
        ops::Range,
        ptr::NonNull,
        sync::atomic::{AtomicUsize, Ordering},
    };
    use std::sync::Mutex;

    struct Backing {
        pointer: NonNull<u8>,
        layout: Layout,
        pages: Vec<u64>,
        pub(in crate::wifi::intel) released: Arc<AtomicUsize>,
    }
    // SAFETY: owned aligned model RAM; access is serialized by the test domain.
    unsafe impl Send for Backing {}
    // SAFETY: no safe slices or concurrent model CPU/DMA writers are exposed.
    unsafe impl Sync for Backing {}
    // SAFETY: uniquely owned host RAM models reserved physical pages;
    // metadata stays immutable and the reservation outlives all kept pins.
    unsafe impl PageBacking for Backing {
        fn cpu_base(&self) -> NonNull<u8> {
            self.pointer
        }
        fn physical_pages(&self) -> &[u64] {
            &self.pages
        }
        fn page_shift(&self) -> u8 {
            12
        }
        fn cache_attribute(&self) -> CacheAttribute {
            CacheAttribute::WriteBack
        }
    }
    impl Drop for Backing {
        fn drop(&mut self) {
            self.released.fetch_add(1, Ordering::Relaxed);
            // SAFETY: precisely this backing's unique still-live allocation.
            unsafe {
                dealloc(self.pointer.as_ptr(), self.layout);
            }
        }
    }
    #[derive(Default)]
    pub(in crate::wifi::intel) struct Source {
        pub(in crate::wifi::intel) calls: AtomicUsize,
        pub(in crate::wifi::intel) fail_at: Option<usize>,
        pub(in crate::wifi::intel) released: Arc<AtomicUsize>,
    }
    impl PageSource for Source {
        fn allocate(
            &self,
            bytes: usize,
            alignment: usize,
            attribute: CacheAttribute,
        ) -> Result<PinnedPages, MemoryError> {
            let call = self.calls.fetch_add(1, Ordering::Relaxed);
            if self.fail_at == Some(call) {
                return Err(MemoryError::OutOfMemory);
            }
            assert_eq!(alignment, 4096);
            assert_eq!(attribute, CacheAttribute::WriteBack);
            let padded = bytes.next_multiple_of(4096);
            let layout = Layout::from_size_align(padded, 4096).unwrap();
            // SAFETY: real model allocation, owned until Backing::drop.
            let pointer =
                NonNull::new(unsafe { alloc_zeroed(layout) }).ok_or(MemoryError::OutOfMemory)?;
            let pages = (0..padded / 4096)
                .map(|n| 0x900000 + (call as u64) * 0x10000 + (n as u64) * 4096)
                .collect();
            PinnedPages::new(
                Box::new(Backing {
                    pointer,
                    layout,
                    pages,
                    released: self.released.clone(),
                }),
                bytes,
                4096,
            )
        }
    }
    #[derive(Default)]
    pub(in crate::wifi::intel) struct Records {
        pub(in crate::wifi::intel) device_sync_hook: Option<Arc<dyn Fn() + Send + Sync>>,
        pub(in crate::wifi::intel) maps: usize,
        pub(in crate::wifi::intel) syncs: usize,
        pub(in crate::wifi::intel) unmaps: usize,
        pub(in crate::wifi::intel) completed: alloc::collections::BTreeSet<u64>,
        pub(in crate::wifi::intel) completed_ranges:
            alloc::collections::BTreeSet<(u64, usize, usize)>,
        pub(in crate::wifi::intel) cpu_syncs: usize,
        pub(in crate::wifi::intel) cpu_sync_fail: Option<usize>,
        pub(in crate::wifi::intel) coherent: bool,
        pub(in crate::wifi::intel) coherent_reads: Vec<(u64, usize)>,
        pub(in crate::wifi::intel) map_fail: Option<usize>,
        pub(in crate::wifi::intel) sync_fail: Option<usize>,
        pub(in crate::wifi::intel) unmap_fail: Option<usize>,
        pub(in crate::wifi::intel) bad_boundaries: usize,
        pub(in crate::wifi::intel) live: BTreeMap<u64, CpuMapping>,
        pub(in crate::wifi::intel) permissions: BTreeMap<u64, Direction>,
        pub(in crate::wifi::intel) copied: Vec<Vec<u8>>,
        pub(in crate::wifi::intel) directions: Vec<Direction>,
    }
    #[derive(Default)]
    pub(in crate::wifi::intel) struct Domain {
        pub(in crate::wifi::intel) records: Mutex<Records>,
    }
    // SAFETY: synchronous IOMMU fixture with distinct device IOVAs independent
    // of CPU/physical addresses. Live pins and failed invalidations are held.
    unsafe impl DmaDomain for Domain {
        unsafe fn map(
            &self,
            pages: &crate::bus::dma::pages::DmaPages,
            direction: Direction,
            limit: u64,
        ) -> Result<DomainMapping, DmaError> {
            let pages = pages.as_single().ok_or(DmaError::MappingFailed)?;
            let mut r = self.records.lock().unwrap();
            let index = r.maps;
            r.maps += 1;
            if r.map_fail == Some(index) {
                return Err(DmaError::MappingFailed);
            }
            let id = index as u64 + 1;
            let address = if r.bad_boundaries != 0 && pages.logical_bytes() == 4096 {
                r.bad_boundaries -= 1;
                (id << 32) - 4096
            } else {
                0x100000 + id * 0x10000
            };
            assert!(address + pages.bytes() as u64 - 1 <= limit);
            assert_ne!(address, pages.physical_pages()[0]);
            r.live
                .insert(id, pages.cpu_mapping(CacheAttribute::WriteBack).unwrap());
            r.directions.push(direction);
            r.permissions.insert(id, direction);
            Ok(DomainMapping {
                id,
                address,
                bytes: pages.bytes(),
            })
        }
        unsafe fn unmap(&self, mapping: DomainMapping) -> Result<(), DmaError> {
            let mut r = self.records.lock().unwrap();
            let index = r.unmaps;
            r.unmaps += 1;
            if r.unmap_fail == Some(index) {
                return Err(DmaError::InvalidationFailed);
            }
            assert!(r.live.remove(&mapping.id).is_some());
            r.permissions.remove(&mapping.id);
            Ok(())
        }
        unsafe fn sync_for_device(
            &self,
            mapping: DomainMapping,
            range: Range<usize>,
            direction: Direction,
        ) -> Result<(), DmaError> {
            let mut r = self.records.lock().unwrap();
            if let Some(hook) = r.device_sync_hook.as_ref() {
                hook();
            }
            let index = r.syncs;
            r.syncs += 1;
            if r.sync_fail == Some(index) {
                return Err(DmaError::VisibilityFailed);
            }
            assert_eq!(Some(&direction), r.permissions.get(&mapping.id));
            assert!(range.start <= range.end && range.end <= mapping.bytes);
            let cpu = r.live.get(&mapping.id).unwrap();
            let ptr = cpu.address(range.start, range.len()).unwrap();
            // SAFETY: fixture executes synchronously before any device exposure,
            // and its real allocation was initialized by the driver.
            let copied = unsafe { core::slice::from_raw_parts(ptr.as_ptr(), range.len()) }.to_vec();
            r.copied.push(copied);
            Ok(())
        }
        unsafe fn sync_for_cpu(
            &self,
            mapping: DomainMapping,
            range: Range<usize>,
            direction: Direction,
        ) -> Result<(), DmaError> {
            let mut r = self.records.lock().unwrap();
            let index = r.cpu_syncs;
            r.cpu_syncs += 1;
            if r.cpu_sync_fail == Some(index) {
                return Err(DmaError::VisibilityFailed);
            }
            assert!(
                r.completed_ranges
                    .remove(&(mapping.id, range.start, range.end))
                    || r.completed.remove(&mapping.id),
                "actual model device completion required before CPU sync"
            );
            assert_eq!(r.permissions.get(&mapping.id), Some(&direction));
            assert!(matches!(
                direction,
                Direction::FromDevice | Direction::Bidirectional
            ));
            assert!(range.start <= range.end && range.end <= mapping.bytes);
            Ok(())
        }
        unsafe fn read_coherent_u32(
            &self,
            mapping: DomainMapping,
            offset: usize,
            direction: Direction,
        ) -> Result<u32, DmaError> {
            let mut r = self.records.lock().unwrap();
            if !r.coherent {
                return Err(DmaError::CoherenceUnavailable);
            }
            assert_eq!(r.permissions.get(&mapping.id), Some(&direction));
            assert!(matches!(
                direction,
                Direction::FromDevice | Direction::Bidirectional
            ));
            assert!(offset.is_multiple_of(4) && offset + 4 <= mapping.bytes);
            let pointer = r.live[&mapping.id].address(offset, 4).unwrap();
            r.coherent_reads.push((mapping.id, offset));
            // SAFETY: the mapping names aligned model RAM, and its domain
            // lock serializes the fixture's device writes with this read.
            Ok(u32::from_le(unsafe {
                pointer.as_ptr().cast::<u32>().read_volatile()
            }))
        }
    }
    fn image() -> Vec<u8> {
        let mut bytes = vec![0; 88];
        bytes[4..8].copy_from_slice(&0x0a4c5749u32.to_le_bytes());
        let mut tlv = |kind: u32, data: &[u8]| {
            bytes.extend(kind.to_le_bytes());
            bytes.extend((data.len() as u32).to_le_bytes());
            bytes.extend(data);
            while !bytes.len().is_multiple_of(4) {
                bytes.push(0);
            }
        };
        tlv(27, &2u32.to_le_bytes());
        tlv(32, &4096u32.to_le_bytes());
        for (offset, len, seed) in [
            (0u32, 16, 1u8),
            (0xffffcccc, 0, 0),
            (0x1000, 16, 2),
            (0xaaaabbbb, 0, 0),
            (0, 16, 3),
            (0x2000, 4096, 4),
        ] {
            let mut data = offset.to_le_bytes().to_vec();
            data.extend((0..len).map(|n| seed.wrapping_add(n as u8)));
            tlv(19, &data);
        }
        bytes
    }
    fn prepare(
        plan: &SelfLoadPlan<'_>,
        source: &Source,
        domain: &Arc<Domain>,
    ) -> Result<MappedFirmware, Error> {
        // SAFETY: fixture source hands out unique RAM with no external aliases.
        unsafe { MappedFirmware::prepare(plan, source, Some(domain.clone())) }
    }
    #[test]
    fn domain_addresses_copies_visibility_and_split_lifetimes() {
        if crate::test_process::isolate_model() {
            return;
        }
        let bytes = image();
        let fw = super::super::firmware::Firmware::parse(&bytes).unwrap();
        let plan = fw.self_load_plan().unwrap();
        let source = Source::default();
        let domain = Arc::new(Domain::default());
        let mut mapped = prepare(&plan, &source, &domain).unwrap();
        {
            let r = domain.records.lock().unwrap();
            assert_eq!(r.maps, 4);
            assert_eq!(r.syncs, 4);
            assert_eq!(
                r.directions,
                vec![
                    Direction::ToDevice,
                    Direction::ToDevice,
                    Direction::Bidirectional,
                    Direction::Bidirectional
                ]
            );
            let expected: Vec<_> = plan
                .lmac()
                .iter()
                .chain(plan.umac())
                .chain(plan.paging())
                .map(|s| s.data.to_vec())
                .collect();
            assert_eq!(r.copied, expected);
        }
        let regions = mapped.expose_for_context().unwrap();
        assert_eq!(regions.lmac[0].address, 0x110000);
        assert_eq!(regions.umac[0].address, 0x120000);
        assert_eq!(regions.paging.len(), 2);
        assert!(matches!(mapped.expose_for_context(), Err(Error::State)));
        // SAFETY: this synchronous fixture has no hardware references.
        unsafe {
            mapped.release_after_alive().unwrap();
        }
        assert_eq!(domain.records.lock().unwrap().live.len(), 2);
        assert_eq!(source.released.load(Ordering::Relaxed), 2);
        {
            let r = domain.records.lock().unwrap();
            for (id, cpu) in &r.live {
                assert_eq!(r.permissions.get(id), Some(&Direction::Bidirectional));
                let pointer = cpu.address(0, 8).unwrap();
                // SAFETY: model device owns live bidirectional paging after
                // exposure/ALIVE; the driver never reads these pages as CPU.
                unsafe {
                    pointer
                        .as_ptr()
                        .cast::<u64>()
                        .write_volatile(0x656469766564);
                }
            }
        }
        // SAFETY: no device or asynchronous IRQ worker exists.
        unsafe {
            mapped.release_after_stop().unwrap();
        }
        assert!(domain.records.lock().unwrap().live.is_empty());
        assert_eq!(source.released.load(Ordering::Relaxed), 4);
        assert!(matches!(mapped.expose_for_context(), Err(Error::State)));
    }
    #[test]
    fn every_prepublication_allocation_mapping_visibility_failure_unwinds() {
        if crate::test_process::isolate_model() {
            return;
        }
        let bytes = image();
        let fw = super::super::firmware::Firmware::parse(&bytes).unwrap();
        let plan = fw.self_load_plan().unwrap();
        for stage in 0..3 {
            for fail in 0..4 {
                let source = Source {
                    fail_at: (stage == 0).then_some(fail),
                    ..Default::default()
                };
                let domain = Arc::new(Domain {
                    records: Mutex::new(Records {
                        map_fail: (stage == 1).then_some(fail),
                        sync_fail: (stage == 2).then_some(fail),
                        ..Default::default()
                    }),
                });
                assert!(prepare(&plan, &source, &domain).is_err());
                assert!(domain.records.lock().unwrap().live.is_empty());
                let allocations = source.calls.load(Ordering::Relaxed) - usize::from(stage == 0);
                assert_eq!(source.released.load(Ordering::Relaxed), allocations);
            }
        }
        let source = Source::default();
        assert!(matches!(
            // SAFETY: unique fixture allocations; absent domain is refused first.
            unsafe { MappedFirmware::prepare(&plan, &source, None) },
            Err(Error::Dma(DmaError::IommuUnavailable))
        ));
        assert_eq!(source.calls.load(Ordering::Relaxed), 0);
        let domain = Arc::new(Domain::default());
        drop(prepare(&plan, &source, &domain).unwrap());
        assert_eq!(source.released.load(Ordering::Relaxed), 4);
        assert!(domain.records.lock().unwrap().live.is_empty());
    }
    #[test]
    fn boundary_retries_preserve_reservations_and_have_a_finite_budget() {
        if crate::test_process::isolate_model() {
            return;
        }
        let bytes = image();
        let fw = super::super::firmware::Firmware::parse(&bytes).unwrap();
        let plan = fw.self_load_plan().unwrap();
        for bad in [1, 2, 3] {
            let source = Source::default();
            let domain = Arc::new(Domain {
                records: Mutex::new(Records {
                    bad_boundaries: bad,
                    ..Default::default()
                }),
            });
            let result = prepare(&plan, &source, &domain);
            if bad < 3 {
                drop(result.unwrap());
                assert_eq!(domain.records.lock().unwrap().maps, 4 + bad);
            } else {
                assert!(matches!(result, Err(Error::BoundaryRetries)));
            }
            assert!(domain.records.lock().unwrap().live.is_empty());
            assert_eq!(
                source.released.load(Ordering::Relaxed),
                source.calls.load(Ordering::Relaxed)
            );
        }
    }
    #[test]
    fn failed_invalidation_keeps_pins_for_retry_and_exposed_drop_quarantines() {
        if crate::test_process::isolate_model() {
            return;
        }
        let bytes = image();
        let fw = super::super::firmware::Firmware::parse(&bytes).unwrap();
        let plan = fw.self_load_plan().unwrap();
        let source = Source::default();
        let domain = Arc::new(Domain {
            records: Mutex::new(Records {
                unmap_fail: Some(0),
                ..Default::default()
            }),
        });
        let mut mapped = prepare(&plan, &source, &domain).unwrap();
        mapped.expose_for_context().unwrap();
        assert_eq!(
            // SAFETY: no real device exists; this fails invalidation.
            unsafe { mapped.release_after_alive() },
            Err(Error::Dma(DmaError::InvalidationFailed))
        );
        assert_eq!(source.released.load(Ordering::Relaxed), 0);
        assert_eq!(domain.records.lock().unwrap().live.len(), 4);
        // SAFETY: same quiescent fixture, next invalidation succeeds.
        unsafe {
            mapped.release_after_alive().unwrap();
        }
        assert_eq!(source.released.load(Ordering::Relaxed), 2);
        drop(mapped);
        assert_eq!(source.released.load(Ordering::Relaxed), 2);
        let r = domain.records.lock().unwrap();
        assert_eq!(r.live.len(), 2);
        assert_eq!(r.unmaps, 3);
    }

    #[test]
    #[ignore = "oracle: STELA_IWLWIFI_IMAGES"]
    fn images_preserve_every_mapped_payload_and_paging_lifetime() {
        if crate::test_process::isolate_model() {
            return;
        }
        use super::super::firmware::{Family, Firmware};
        use sha2::{Digest, Sha256};
        let directory = std::env::var("STELA_IWLWIFI_IMAGES").expect("firmware directory");
        for (family, digest) in [
            (
                Family::Ax200,
                "1866a58a1daee56b120362122d5f7548b443b53d33f2558a7e4407d58fd27dab",
            ),
            (
                Family::Ax210,
                "792478ef7752595525379a5068e2d47e310a2c344636b7df866c54746ced1dc0",
            ),
        ] {
            let image =
                std::fs::read(std::path::Path::new(&directory).join(family.firmware_name()))
                    .unwrap();
            let actual: alloc::string::String = Sha256::digest(&image)
                .iter()
                .map(|b| alloc::format!("{b:02x}"))
                .collect();
            assert_eq!(actual, digest);
            let fw = Firmware::parse(&image).unwrap();
            fw.require_family(family).unwrap();
            let plan = fw.self_load_plan().unwrap();
            let source = Source::default();
            let domain = Arc::new(Domain::default());
            let mut mapped = prepare(&plan, &source, &domain).unwrap();
            {
                let r = domain.records.lock().unwrap();
                let expected: Vec<_> = plan
                    .lmac()
                    .iter()
                    .chain(plan.umac())
                    .chain(plan.paging())
                    .map(|s| s.data.to_vec())
                    .collect();
                assert_eq!(r.copied, expected);
            }
            let images = mapped.expose_for_context().unwrap();
            assert_eq!(
                [images.lmac.len(), images.umac.len(), images.paging.len()],
                [plan.lmac().len(), plan.umac().len(), plan.paging().len()]
            );
            // SAFETY: synchronous no-device model, source-derived ALIVE ownership.
            unsafe {
                mapped.release_after_alive().unwrap();
            }
            assert_eq!(
                domain.records.lock().unwrap().live.len(),
                plan.paging().len()
            );
            // SAFETY: no device accesses or IRQ workers exist.
            unsafe {
                mapped.release_after_stop().unwrap();
            }
            assert!(domain.records.lock().unwrap().live.is_empty());
            assert_eq!(
                source.released.load(Ordering::Relaxed),
                plan.lmac().len() + plan.umac().len() + plan.paging().len()
            );
        }
    }

    #[test]
    fn prepublication_cleanup_invalidation_failure_keeps_the_uncertain_pin() {
        if crate::test_process::isolate_model() {
            return;
        }
        let bytes = image();
        let fw = super::super::firmware::Firmware::parse(&bytes).unwrap();
        let plan = fw.self_load_plan().unwrap();
        let source = Source::default();
        let domain = Arc::new(Domain {
            records: Mutex::new(Records {
                sync_fail: Some(3),
                unmap_fail: Some(0),
                ..Default::default()
            }),
        });
        assert!(matches!(
            prepare(&plan, &source, &domain),
            Err(Error::Dma(DmaError::VisibilityFailed))
        ));
        assert_eq!(source.released.load(Ordering::Relaxed), 3);
        let r = domain.records.lock().unwrap();
        assert_eq!(r.live.len(), 1);
        assert_eq!(r.unmaps, 4);
    }
}
