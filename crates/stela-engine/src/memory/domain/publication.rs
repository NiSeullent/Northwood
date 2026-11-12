//! Publishing host writes through the domain's physical writer.

use super::{
    AccessKind, Arc, MemoryBacking, MemoryStatus, MemorySystem, Ordering, access, monitor,
    service::FrozenPhysicalWriters,
};
use alloc::vec::Vec;
use stela_compiler::TargetCpu;

const HOST_QUANTITY_BYTES: usize = 8;

/// Failed host publication, including the prefix written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PublicationFault {
    pub status: MemoryStatus,
    pub completed: usize,
}

impl core::fmt::Display for PublicationFault {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "host publication failed after {} bytes: {:?}",
            self.completed, self.status
        )
    }
}

impl core::error::Error for PublicationFault {}

impl MemorySystem {
    /// Conditionally update one aligned physical descriptor under its exact
    /// all-writer reservation gate. A comparison miss performs no store and
    /// does not invalidate reservations; a same-value successful write does.
    ///
    /// # Safety
    /// The backing allocation stays live, pinned, coherent write-back raw RAM,
    /// with no aliasing Rust references. For an admitted descriptor access, the
    /// caller owns the descriptor and every physical writer (including source
    /// and host aliases) uses this same system; there is no external DMA writer.
    /// This function checks coherence identity, alignment, bounds and rights
    /// before accessing descriptor bytes. Source/code publication remains the
    /// outer lifecycle owner's responsibility.
    pub(in crate::memory) unsafe fn compare_update_descriptor(
        &self,
        backing: &Arc<MemoryBacking>,
        offset: usize,
        expected: u64,
        replacement: u64,
    ) -> Result<bool, MemoryStatus> {
        self.reservations
            .check_execution()
            .map_err(|_| MemoryStatus::Domain)?;
        if self.failed.load(Ordering::Acquire) {
            return Err(MemoryStatus::Domain);
        }
        backing
            .bind_coherence(&self.identity)
            .map_err(|_| MemoryStatus::Domain)?;
        if !offset.is_multiple_of(HOST_QUANTITY_BYTES) {
            return Err(MemoryStatus::InvalidRequest);
        }
        let pointer = backing
            .host_pointer(offset, HOST_QUANTITY_BYTES, AccessKind::Read)
            .map_err(|_| MemoryStatus::Translation)?;
        backing
            .host_pointer(offset, HOST_QUANTITY_BYTES, AccessKind::Write)
            .map_err(|_| MemoryStatus::Permission)?;
        if !(pointer.as_ptr() as usize).is_multiple_of(HOST_QUANTITY_BYTES) {
            return Err(MemoryStatus::Unsupported);
        }
        access::read_footprint(
            pointer.as_ptr() as usize,
            HOST_QUANTITY_BYTES,
            TargetCpu::Baseline,
        )
        .map_err(|_| MemoryStatus::Unsupported)?;
        let range = monitor::WriteRange::new(
            Arc::clone(backing),
            offset as u64,
            HOST_QUANTITY_BYTES as u64,
        )
        .map_err(|_| MemoryStatus::Domain)?;
        let mut mutation = self.prepare_source_write(backing, offset, HOST_QUANTITY_BYTES)?;
        enum Attempt {
            Mismatch,
            Native,
        }
        let result = self
            .reservations
            .with_writer(&range, monitor::Invalidation::OnCommit, || {
                // SAFETY: exact aligned private read while the same physical gate
                // excludes every source/host/alias writer. No AtomicLoad16 needed.
                let observed = unsafe {
                    access::read_part(pointer.as_ptr(), HOST_QUANTITY_BYTES, TargetCpu::Baseline)
                }
                .map_err(|_| Attempt::Native)? as u64;
                if observed != expected {
                    return Err(Attempt::Mismatch);
                }
                mutation.attempting();
                // SAFETY: the same pinned exact extent; XCHG publishes one SCA
                // write for ungated ordinary readers while our writer gate is held.
                unsafe {
                    access::write_part(
                        pointer.as_ptr(),
                        HOST_QUANTITY_BYTES,
                        u128::from(replacement),
                        TargetCpu::Baseline,
                    )
                }
                .map_err(|_| Attempt::Native)?;
                mutation.complete();
                Ok(())
            });
        match result {
            Ok(()) => Ok(true),
            Err(monitor::AccessError::Memory(Attempt::Mismatch)) => Ok(false),
            Err(_) => Err(MemoryStatus::Domain),
        }
    }

    /// Publish private host bytes to the backing, invalidating reservations.
    ///
    /// Guest permissions and whole-operation checks are the caller's, so a host
    /// publisher with writable rights may update a guest read-only view. Every
    /// fragment invalidates reservations, including same-value writes and ABA
    /// updates through aliases. Bounds, rights and recipe failures are reported
    /// before any byte is written; later errors report the published prefix.
    /// This is a host copyout, not a guest-instruction transaction. An empty
    /// publication still validates and binds the coherence owner.
    ///
    /// # Safety
    /// The caller may publish to this backing and has validated the operation.
    /// The whole extent stays live, pinned, readable and writable coherent
    /// write-back RAM for the call. The source bytes are private and disjoint
    /// from shared guest RAM, and no Rust reference aliases the destination.
    /// Every physical writer and reservation uses this system and backing. Do
    /// not re-enter the system or call while holding a monitor gate. Host
    /// faults are not caught.
    pub unsafe fn publish(
        &self,
        backing: &Arc<MemoryBacking>,
        offset: usize,
        bytes: &[u8],
    ) -> Result<(), PublicationFault> {
        // SAFETY: the same contract as `publish_tracked`.
        unsafe { self.publish_tracked(backing, offset, bytes, None) }
    }

    /// The same publication, recording source writes for the write session.
    pub(in crate::memory) unsafe fn publish_tracked(
        &self,
        backing: &Arc<MemoryBacking>,
        offset: usize,
        bytes: &[u8],
        receipts: Option<&mut Vec<super::SourceWriteReceipt>>,
    ) -> Result<(), PublicationFault> {
        // SAFETY: the caller supplies pinned storage and private staging.
        unsafe { self.publish_with(backing, offset, bytes, receipts, None) }
    }

    /// `writers` must come from this system when the caller already holds
    /// the physical writer gates.
    pub(super) unsafe fn publish_with(
        &self,
        backing: &Arc<MemoryBacking>,
        offset: usize,
        bytes: &[u8],
        mut receipts: Option<&mut Vec<super::SourceWriteReceipt>>,
        writers: Option<&FrozenPhysicalWriters<'_>>,
    ) -> Result<(), PublicationFault> {
        let preflight_fault = |status| PublicationFault {
            status,
            completed: 0,
        };
        if self.failed.load(Ordering::Acquire) {
            return Err(preflight_fault(MemoryStatus::Domain));
        }
        self.reservations
            .check_execution()
            .map_err(|_| preflight_fault(MemoryStatus::Domain))?;
        backing
            .bind_coherence(&self.identity)
            .map_err(|_| preflight_fault(MemoryStatus::Domain))?;
        offset
            .checked_add(bytes.len())
            .filter(|end| *end <= backing.bytes())
            .ok_or_else(|| preflight_fault(MemoryStatus::Translation))?;
        if bytes.is_empty() {
            return Ok(());
        }
        let map_extent_error = |error| {
            preflight_fault(match error {
                super::super::backing::AccessError::PermissionDenied(_) => MemoryStatus::Permission,
                _ => MemoryStatus::Translation,
            })
        };
        let pointer = backing
            .host_pointer(offset, bytes.len(), AccessKind::Write)
            .map_err(map_extent_error)?;
        // Scalar XCHG reads the old value as well as writing the replacement.
        backing
            .host_pointer(offset, bytes.len(), AccessKind::Read)
            .map_err(map_extent_error)?;
        // The scalar read/write helpers share width and quantity validation.
        // Preflight every recipe before the first reservation or byte effect.
        for part in PublicationParts::new(pointer.as_ptr() as usize, bytes.len()) {
            access::read_footprint(part.address, part.bytes, TargetCpu::Baseline)
                .map_err(|_| preflight_fault(MemoryStatus::Unsupported))?;
        }
        if let Some(receipts) = receipts.as_mut() {
            // Every later push is allocation-free, including partial publication.
            receipts
                .try_reserve(PublicationParts::new(pointer.as_ptr() as usize, bytes.len()).count())
                .map_err(|_| preflight_fault(MemoryStatus::Domain))?;
        }
        for part in PublicationParts::new(pointer.as_ptr() as usize, bytes.len()) {
            let mut mutation = self
                .prepare_source_write(backing, offset + part.offset, part.bytes)
                .map_err(|status| PublicationFault {
                    status,
                    completed: part.offset,
                })?;
            let mut staging = [0; HOST_QUANTITY_BYTES];
            staging[..part.bytes].copy_from_slice(&bytes[part.offset..part.offset + part.bytes]);
            let value = u128::from(u64::from_le_bytes(staging));
            let range = monitor::WriteRange::new(
                Arc::clone(backing),
                (offset + part.offset) as u64,
                part.bytes as u64,
            )
            .map_err(|_| PublicationFault {
                status: MemoryStatus::Domain,
                completed: part.offset,
            })?;
            let write = || {
                mutation.attempting();
                // SAFETY: whole host extent and scalar recipes were checked;
                // the caller pins rights and lifetime, with private staging.
                unsafe {
                    access::write_part(
                        part.address as *mut u8,
                        part.bytes,
                        value,
                        TargetCpu::Baseline,
                    )
                }?;
                if let Some(receipt) = mutation.complete()
                    && let Some(receipts) = receipts.as_mut()
                {
                    receipts.push(receipt);
                }
                Ok::<_, access::AccessError>(())
            };
            let result = match writers {
                Some(writers) => writers.with_writer(&range, write),
                None => self.reservations.with_writer(
                    &range,
                    monitor::Invalidation::BeforeAttempt,
                    write,
                ),
            };
            result.map_err(|error| PublicationFault {
                status: match error {
                    monitor::AccessError::Monitor(_) => MemoryStatus::Domain,
                    monitor::AccessError::Memory(_) => MemoryStatus::Unsupported,
                },
                completed: part.offset,
            })?;
        }
        Ok(())
    }
}

struct PublicationPart {
    address: usize,
    offset: usize,
    bytes: usize,
}

struct PublicationParts {
    address: usize,
    remaining: usize,
    offset: usize,
}

impl PublicationParts {
    fn new(address: usize, bytes: usize) -> Self {
        Self {
            address,
            remaining: bytes,
            offset: 0,
        }
    }
}

impl Iterator for PublicationParts {
    type Item = PublicationPart;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let available = HOST_QUANTITY_BYTES - (self.address & (HOST_QUANTITY_BYTES - 1));
        let bytes = 1 << self.remaining.min(available).ilog2();
        let part = PublicationPart {
            address: self.address,
            offset: self.offset,
            bytes,
        };
        self.address += bytes;
        self.offset += bytes;
        self.remaining -= bytes;
        Some(part)
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests;
