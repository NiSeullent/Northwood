//! Native coherent host/table-walker transfers through the canonical RAM owner.
use super::RegisteredSystemRam;
use crate::memory::{access, backing::AccessKind, domain::PublicationFault};
use alloc::{sync::Arc, vec::Vec};
use core::ops::Range;
use stela_compiler::{TargetCpu, context::memory::MemoryStatus};
use stela_contracts::gate::MemoryRange;

/// Ordered guest ranges behind a wired buffer. It lends no host address for
/// a guest frame, so a device can use a separate allocation for its pixels.
pub struct WiredRam {
    ram: Arc<RegisteredSystemRam>,
    spans: Vec<WiredSpan>,
    bytes: u64,
}

struct WiredSpan {
    logical: u64,
    physical: u64,
    bytes: u64,
}

impl WiredRam {
    /// Checks the whole writable layout before a transfer can change bytes.
    /// Logical order is preserved, including when physical ranges go
    /// backwards or have holes between them.
    pub fn new(
        ram: Arc<RegisteredSystemRam>,
        ranges: &[MemoryRange],
    ) -> Result<Self, MemoryStatus> {
        let mut spans = Vec::new();
        spans
            .try_reserve_exact(ranges.len())
            .map_err(|_| MemoryStatus::Domain)?;
        let mut bytes = 0u64;
        for range in ranges {
            let length = usize::try_from(range.bytes).map_err(|_| MemoryStatus::Translation)?;
            if length == 0 {
                return Err(MemoryStatus::InvalidRequest);
            }
            range
                .address
                .checked_add(range.bytes)
                .ok_or(MemoryStatus::Translation)?;
            ram.check_dma_range(range.address, length, true)?;
            spans.push(WiredSpan {
                logical: bytes,
                physical: range.address,
                bytes: range.bytes,
            });
            bytes = bytes
                .checked_add(range.bytes)
                .ok_or(MemoryStatus::Translation)?;
        }
        spans.sort_unstable_by_key(|span| span.physical);
        if bytes == 0
            || spans
                .windows(2)
                .any(|pair| pair[0].physical + pair[0].bytes > pair[1].physical)
        {
            return Err(MemoryStatus::InvalidRequest);
        }
        spans.sort_unstable_by_key(|span| span.logical);
        Ok(Self { ram, spans, bytes })
    }

    /// Read a logical span into a private device allocation.
    ///
    /// # Safety
    /// The guest's original descriptor keeps the ranges wired. The output
    /// is private, disjoint from registered RAM and idle on the device.
    pub unsafe fn read_into(&self, offset: u64, output: &mut [u8]) -> Result<(), PublicationFault> {
        self.transfer(offset, output.len(), |address, range| {
            // SAFETY: the caller keeps the wire and output through the copy.
            unsafe { self.ram.read_physical_progress(address, &mut output[range]) }
        })
    }

    /// Publish a logical span from a completed device operation.
    /// On failure `completed` counts the logical bytes already published.
    ///
    /// # Safety
    /// The guest's bidirectional descriptor keeps the ranges wired. The
    /// source is private and coherent, and the device has completed its writes.
    pub unsafe fn publish_from(&self, offset: u64, source: &[u8]) -> Result<(), PublicationFault> {
        self.transfer(offset, source.len(), |address, range| {
            // SAFETY: the source is idle and the caller keeps the descriptor.
            unsafe { self.ram.publish_physical(address, &source[range]) }
        })
    }

    fn transfer(
        &self,
        offset: u64,
        bytes: usize,
        mut copy: impl FnMut(u64, Range<usize>) -> Result<(), PublicationFault>,
    ) -> Result<(), PublicationFault> {
        let end = offset
            .checked_add(bytes as u64)
            .filter(|end| *end <= self.bytes)
            .ok_or(PublicationFault {
                status: MemoryStatus::Translation,
                completed: 0,
            })?;
        if bytes == 0 {
            return Ok(());
        }
        let first = self
            .spans
            .partition_point(|span| span.logical + span.bytes <= offset);
        for span in self.spans[first..]
            .iter()
            .take_while(|span| span.logical < end)
        {
            let start = offset.max(span.logical);
            let stop = end.min(span.logical + span.bytes);
            let address = span.physical + (start - span.logical);
            let done = (start - offset) as usize;
            copy(address, done..(stop - offset) as usize).map_err(|fault| PublicationFault {
                status: fault.status,
                completed: done + fault.completed,
            })?;
        }
        Ok(())
    }
}

/// A page-table walk's view of one registered RAM owner.
/// Caches an extent index in one binding generation, never a descriptor value.
pub struct DescriptorReader<'a> {
    ram: &'a RegisteredSystemRam,
    last: Option<(u64, usize)>,
}

impl RegisteredSystemRam {
    /// Publish private host bytes through every reached backing's same writer
    /// owner. Whole physical geometry is checked before the first byte effect;
    /// a later terminal coherence failure reports the exact committed prefix.
    ///
    /// # Safety
    /// Source bytes are private and disjoint from registered RAM/native frames.
    /// The registration all-writer/owner contract remains live. This is host
    /// copyout, not a source instruction or a code-publication acknowledgement.
    pub unsafe fn publish_physical(
        &self,
        physical: u64,
        bytes: &[u8],
    ) -> Result<(), PublicationFault> {
        self.preflight(physical, bytes.len())
            .map_err(|status| PublicationFault {
                status,
                completed: 0,
            })?;
        let _pin = self
            .pin_range(physical, bytes.len())
            .map_err(|status| PublicationFault {
                status,
                completed: 0,
            })?;
        let mut done = 0;
        while done < bytes.len() {
            let (region, offset) = self
                .region(physical + done as u64)
                .ok_or(PublicationFault {
                    status: MemoryStatus::Domain,
                    completed: done,
                })?;
            let count = (bytes.len() - done).min(region.bytes - offset);
            // SAFETY: preflight checks the whole immutable physical placement;
            // registration keeps the same native writer and private source.
            unsafe {
                self.system
                    .publish(&region.backing, offset, &bytes[done..done + count])
            }
            .map_err(|fault| PublicationFault {
                status: fault.status,
                completed: done + fault.completed,
            })?;
            done += count;
        }
        Ok(())
    }

    /// Read a single aligned descriptor through the canonical RAM owner.
    pub fn read_descriptor(&self, physical: u64) -> Result<u64, MemoryStatus> {
        self.descriptor_reader().read_descriptor(physical)
    }

    /// Read a succession of descriptors through this registered owner.
    pub fn descriptor_reader(&self) -> DescriptorReader<'_> {
        DescriptorReader {
            ram: self,
            last: None,
        }
    }

    pub fn update_descriptor(
        &self,
        physical: u64,
        old: u64,
        new: u64,
    ) -> Result<bool, MemoryStatus> {
        self.preflight(physical, 8)?;
        if !physical.is_multiple_of(8) {
            return Err(MemoryStatus::InvalidRequest);
        }
        let _pin = self.pin_descriptor(physical)?;
        let binding = self.binding()?;
        let (region, offset) = binding
            .ram
            .region(physical)
            .ok_or(MemoryStatus::Translation)?;
        region
            .backing
            .host_pointer(offset, 8, AccessKind::Write)
            .map_err(|_| MemoryStatus::Permission)?;
        // SAFETY: the registration closes every physical writer over
        // this exact WB backing; no external DMA/direct writer participates.
        unsafe {
            self.system
                .compare_update_descriptor(&region.backing, offset, old, new)
        }
    }
}

impl DescriptorReader<'_> {
    /// Read one aligned descriptor through its live canonical backing.
    pub fn read_descriptor(&mut self, physical: u64) -> Result<u64, MemoryStatus> {
        if !physical.is_multiple_of(8) {
            return Err(MemoryStatus::InvalidRequest);
        }
        let _pin = self.ram.pin_descriptor(physical)?;
        let binding = self.ram.binding()?;
        let hint = self
            .last
            .filter(|&(generation, _)| generation == binding.generation)
            .map(|(_, index)| index);
        let (index, region, offset) = binding
            .ram
            .region_at(physical, hint)
            .ok_or(MemoryStatus::Translation)?;
        self.last = Some((binding.generation, index));
        let pointer = region
            .backing
            .host_pointer(offset, 8, AccessKind::Read)
            .map_err(|_| MemoryStatus::Domain)?;
        // SAFETY: registration aligns both ends of each extent to at least
        // eight bytes. The pointer covers one aligned doubleword in its
        // pinned, coherent backing, with no Rust reference into guest RAM.
        unsafe { access::read_part(pointer.as_ptr(), 8, TargetCpu::Baseline) }
            .map(|value| value as u64)
            .map_err(|_| MemoryStatus::Domain)
    }

    /// Publish through the same canonical owner as a standalone update.
    pub fn update_descriptor(
        &self,
        physical: u64,
        old: u64,
        new: u64,
    ) -> Result<bool, MemoryStatus> {
        self.ram.update_descriptor(physical, old, new)
    }
}
