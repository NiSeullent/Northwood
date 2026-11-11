//! A live canonical RAM borrow, not an MCC/DRAM/DMA completion receipt.
use super::*;
use crate::memory::domain::{AllWriterService, PublicationFault};
use descriptor_frames::{ClosedStripe, DescriptorFrames, FRAME_BYTES};

/// Keeps this exact registered reservation/view generation and every canonical
/// physical writer gate borrowed. It neither accesses bytes nor changes a
/// monitor. Native execution is excluded until drop. Unregistered DMA/direct
/// access is never admitted by this capability.
#[must_use]
pub struct CanonicalRamService<'a> {
    ram: &'a RegisteredSystemRam,
    service: AllWriterService<'a>,
}

/// Keeps registered frame access closed while the canonical all-writer
/// service remains held. A snapshot supplies bytes without any host pointer
/// or physical reuse authority. Direct mappings have separate owners.
#[must_use]
pub struct ClosedDescriptorFrame<'a> {
    // Reopen frame admission before unfreezing writers.
    _stripe: ClosedStripe<'a>,
    service: FrameService<'a>,
    physical: u64,
}

enum FrameService<'a> {
    Owned(CanonicalRamService<'a>),
    Borrowed(&'a CanonicalRamService<'a>),
}

impl FrameService<'_> {
    fn memory(&self) -> &CanonicalRamService<'_> {
        match self {
            Self::Owned(service) => service,
            Self::Borrowed(service) => service,
        }
    }
}

impl ClosedDescriptorFrame<'_> {
    pub fn physical(&self) -> u64 {
        self.physical
    }

    pub(crate) fn service(&self) -> &CanonicalRamService<'_> {
        self.service.memory()
    }

    /// Inspect the full frame while ordinary canonical reads and writes
    /// remain excluded. This grants no permission to retire raw mappings.
    ///
    /// # Safety
    /// The output must be private and disjoint from live RAM/native frames.
    pub unsafe fn copy_into(&self, output: &mut [u8; FRAME_BYTES]) -> Result<(), MemoryStatus> {
        let service = self.service();
        service.validate_current()?;
        // SAFETY: same live owner, the full frame was validated before
        // closing, and the caller keeps the output private.
        unsafe { service.ram.read_physical_unpinned(self.physical, output) }
            .map_err(|fault| fault.status)
    }
}

impl<'a> ClosedDescriptorFrame<'a> {
    /// # Safety
    /// All direct aliases must have been retired and their owners must stay
    /// held until the original backing is restored.
    pub(crate) unsafe fn mark_nonresident(self) -> Result<NonresidentRamFrame<'a>, MemoryStatus> {
        let service = self.service();
        service.validate_current()?;
        let (frames, index) = service.ram.frame_table(self.physical)?;
        frames.mark_nonresident(index);
        let Self {
            _stripe: stripe,
            service,
            physical,
        } = self;
        drop(stripe);
        Ok(NonresidentRamFrame {
            service,
            physical,
            pending: true,
        })
    }
}

/// Holds one nonresident frame and its original backing. The caller must
/// prove its contents before reopening admission and releasing the workers.
#[must_use]
pub struct NonresidentRamFrame<'a> {
    service: FrameService<'a>,
    physical: u64,
    pending: bool,
}

impl NonresidentRamFrame<'_> {
    pub fn physical(&self) -> u64 {
        self.physical
    }

    /// Reopen admission only if the retained backing matches the entire
    /// independently recovered page. A mismatch leaves the page nonresident
    /// and unchanged. Returns false if the stripe is temporarily busy.
    pub fn try_restore(&mut self, expected: &[u8; FRAME_BYTES]) -> Result<bool, MemoryStatus> {
        let service = self.service.memory();
        let Some((frames, index, _stripe)) = close_pending(service, self.physical, self.pending)?
        else {
            return Ok(false);
        };
        let mut observed = [0; 512];
        for (chunk, expected) in expected.chunks_exact(observed.len()).enumerate() {
            let address = self.physical + (chunk * observed.len()) as u64;
            // SAFETY: original backing, exact closed frame, private stack buffer.
            unsafe { service.ram.read_physical_unpinned(address, &mut observed) }
                .map_err(|fault| fault.status)?;
            if observed.as_slice() != expected {
                return Err(MemoryStatus::InvalidRequest);
            }
        }
        frames.mark_resident(index);
        self.pending = false;
        Ok(true)
    }

    /// Replace the retained bytes while this frame stays nonresident.
    /// Returns false if the stripe is busy. A copy fault reports the written
    /// prefix, which stays inaccessible through registered RAM.
    ///
    /// # Safety
    /// The caller must keep the exact frame's complete durable copy and the
    /// original stopped owners through restoration. `source` must be private,
    /// disjoint from registered RAM, and unavailable to device writers.
    pub unsafe fn try_overwrite_backing(
        &mut self,
        source: &[u8; FRAME_BYTES],
    ) -> Result<bool, PublicationFault> {
        // SAFETY: the caller maintains the exact frame and source contract.
        unsafe { self.publish_backing(source, false) }
    }

    /// Replace the retained bytes and reopen admission after a full copy.
    /// A busy stripe or copy failure leaves this frame nonresident.
    ///
    /// # Safety
    /// The overwrite contract applies. `source` must be the authenticated
    /// original bytes of this exact frame from its current durable copy.
    pub unsafe fn try_restore_backing(
        &mut self,
        source: &[u8; FRAME_BYTES],
    ) -> Result<bool, PublicationFault> {
        // SAFETY: the caller authenticated the exact frame and owns the source.
        unsafe { self.publish_backing(source, true) }
    }

    unsafe fn publish_backing(
        &mut self,
        source: &[u8; FRAME_BYTES],
        restore: bool,
    ) -> Result<bool, PublicationFault> {
        let before = |status| PublicationFault {
            status,
            completed: 0,
        };
        let service = self.service.memory();
        let Some((frames, index, _stripe)) =
            close_pending(service, self.physical, self.pending).map_err(before)?
        else {
            return Ok(false);
        };
        let (region, offset) = service
            .ram
            .region(self.physical)
            .ok_or_else(|| before(MemoryStatus::Domain))?;
        // SAFETY: one exact writable registered frame, closed to new pins,
        // with all original writer gates and CPU owners still held.
        unsafe {
            service
                .service
                .publish_closed(&region.backing, offset, source)
        }?;
        service
            .validate_current()
            .map_err(|status| PublicationFault {
                status,
                completed: FRAME_BYTES,
            })?;
        if restore {
            frames.mark_resident(index);
            self.pending = false;
        }
        Ok(true)
    }

    /// Inspect the original backing through the hosted model's held frame.
    ///
    /// # Safety
    /// The output must be private and disjoint from the fixture's RAM.
    #[cfg(all(feature = "std", feature = "test-support"))]
    pub unsafe fn reference_copy_backing(
        &self,
        output: &mut [u8; FRAME_BYTES],
    ) -> Result<bool, MemoryStatus> {
        let service = self.service.memory();
        let Some((_, _, _stripe)) = close_pending(service, self.physical, self.pending)? else {
            return Ok(false);
        };
        // SAFETY: the held stripe excludes access and output is private.
        unsafe { service.ram.read_physical_unpinned(self.physical, output) }
            .map_err(|fault| fault.status)?;
        Ok(true)
    }
}

fn close_pending<'a>(
    service: &'a CanonicalRamService<'_>,
    physical: u64,
    pending: bool,
) -> Result<Option<(&'a DescriptorFrames, usize, ClosedStripe<'a>)>, MemoryStatus> {
    if !pending {
        return Err(MemoryStatus::InvalidRequest);
    }
    service.validate_current()?;
    let (frames, index) = service.ram.frame_table(physical)?;
    let Some(stripe) = frames.try_close(physical) else {
        return Ok(None);
    };
    if !frames.is_nonresident(index) {
        return Err(MemoryStatus::Domain);
    }
    Ok(Some((frames, index, stripe)))
}

impl Drop for NonresidentRamFrame<'_> {
    fn drop(&mut self) {
        if self.pending {
            self.service.memory().fail();
        }
    }
}

fn close_resident(
    frames: &DescriptorFrames,
    index: usize,
    physical: u64,
) -> Result<Option<ClosedStripe<'_>>, MemoryStatus> {
    let Some(stripe) = frames.try_close(physical) else {
        return Ok(None);
    };
    if frames.is_nonresident(index) {
        return Err(MemoryStatus::Nonresident);
    }
    Ok((!frames.is_wired(index)).then_some(stripe))
}

impl CanonicalRamService<'_> {
    pub(crate) fn registration(&self) -> &RegisteredSystemRam {
        self.ram
    }

    pub(crate) fn has_nonresident_frames(&self) -> bool {
        self.ram
            .descriptor_frames
            .as_ref()
            .is_some_and(DescriptorFrames::has_nonresident)
    }

    pub fn validate_current(&self) -> Result<(), MemoryStatus> {
        self.service.validate_current()
    }
    pub fn generation(&self) -> u64 {
        self.service.generation()
    }
    /// Identity comparison of the owner, not matching geometry.
    pub fn belongs_to(&self, ram: &RegisteredSystemRam) -> bool {
        core::ptr::eq(self.ram, ram)
    }

    /// Borrow the existing writer service to close one complete frame.
    /// On contention release the outer service before retrying so a pinned
    /// writer waiting for its gate can finish.
    pub fn try_close_frame(
        &self,
        physical: u64,
    ) -> Result<Option<ClosedDescriptorFrame<'_>>, MemoryStatus> {
        self.validate_current()?;
        let (frames, index) = self.ram.frame_table(physical)?;
        Ok(
            close_resident(frames, index, physical)?.map(|stripe| ClosedDescriptorFrame {
                _stripe: stripe,
                service: FrameService::Borrowed(self),
                physical,
            }),
        )
    }

    /// Permanently fail this exact RAM domain and its physical-writer
    /// gate after an unchecked native fault. Future entry/service acquisition
    /// refuses; this cannot restore permission or manufacture completion.
    pub fn fail(&self) {
        self.service.fail();
    }
}
impl RegisteredSystemRam {
    /// Close one complete 16 KiB frame to registered accesses. Park the
    /// issuers before closing on a running guest. A busy or colliding
    /// stripe returns `None` after releasing the writer service.
    pub fn try_close_descriptor_frame(
        &self,
        physical: u64,
    ) -> Result<Option<ClosedDescriptorFrame<'_>>, MemoryStatus> {
        let (frames, index) = self.frame_table(physical)?;
        let Some(service) = self.try_service()? else {
            return Ok(None);
        };
        Ok(
            close_resident(frames, index, physical)?.map(|stripe| ClosedDescriptorFrame {
                _stripe: stripe,
                service: FrameService::Owned(service),
                physical,
            }),
        )
    }

    fn frame_table(&self, physical: u64) -> Result<(&DescriptorFrames, usize), MemoryStatus> {
        if !physical.is_multiple_of(FRAME_BYTES as u64) {
            return Err(MemoryStatus::Alignment);
        }
        let frames = self
            .descriptor_frames
            .as_ref()
            .ok_or(MemoryStatus::Unsupported)?;
        let (region, offset) = self.region(physical).ok_or(MemoryStatus::Translation)?;
        if FRAME_BYTES > region.bytes - offset {
            return Err(MemoryStatus::Translation);
        }
        if !(region.host_pa + offset as u64).is_multiple_of(FRAME_BYTES as u64) {
            return Err(MemoryStatus::Alignment);
        }
        Ok((frames, region.wire_index(physical)))
    }

    /// Acquire an all-writer service borrow only when no registered native
    /// execution or physical writer is active. A busy owner returns `None`.
    /// This serializes canonical memory; it performs no controller or DMA
    /// completion and grants no access to unregistered physical writers.
    pub fn try_service(&self) -> Result<Option<CanonicalRamService<'_>>, MemoryStatus> {
        self.domain
            .try_all_writer_service()
            .map(|service| service.map(|service| CanonicalRamService { ram: self, service }))
    }

    /// [`Self::try_service`] for an owner that has stopped every PE: it
    /// waits out the holders left, which finish without it (another PE's
    /// monitor or writer gate while it applies its mail, or the native
    /// slice of one that woke from a wait before it parks).
    pub fn service(&self) -> Result<CanonicalRamService<'_>, MemoryStatus> {
        loop {
            if let Some(service) = self.try_service()? {
                return Ok(service);
            }
            core::hint::spin_loop();
        }
    }
}
