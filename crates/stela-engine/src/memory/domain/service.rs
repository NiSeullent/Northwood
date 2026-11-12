//! Exclusive mapping service and all physical writer gates of one live domain.
use super::*;

pub(super) type FrozenPhysicalWriters<'a> =
    monitor::FrozenWriters<'a, MemoryBacking, MemoryView, MappingAttributes>;

pub(in crate::memory) struct AllWriterService<'a> {
    domain: &'a Arc<MemoryDomain>,
    generation: u64,
    // Drop gates before ending mapping/native exclusion.
    writers: FrozenPhysicalWriters<'a>,
    _loan: MemoryServiceLoan,
}
impl AllWriterService<'_> {
    pub(in crate::memory) fn validate_current(&self) -> Result<(), MemoryStatus> {
        if self.domain.generation()? != self.generation {
            return Err(MemoryStatus::Domain);
        }
        Ok(())
    }
    pub(in crate::memory) fn generation(&self) -> u64 {
        self.generation
    }
    pub(in crate::memory) fn fail(&self) {
        self.domain.system.failed.store(true, Ordering::Release);
        self.writers.fail();
    }

    /// Publish through the physical writer gates already held by this service.
    ///
    /// # Safety
    /// The caller owns the exact registered nonresident stripe and excludes
    /// all readers and unjoined CPU/DMA aliases to the destination. The
    /// pinned backing and private, disjoint source meet `MemorySystem::publish`'s
    /// extent contract.
    pub(in crate::memory) unsafe fn publish_closed(
        &self,
        backing: &Arc<MemoryBacking>,
        offset: usize,
        bytes: &[u8],
    ) -> Result<(), PublicationFault> {
        self.validate_current().map_err(|status| PublicationFault {
            status,
            completed: 0,
        })?;
        // SAFETY: the caller keeps the admitted closed stripe and private
        // source; these gates came from this exact domain's physical writer.
        unsafe {
            self.domain
                .system
                .publish_with(backing, offset, bytes, None, Some(&self.writers))
        }
    }
}
impl MemoryDomain {
    pub(in crate::memory) fn try_all_writer_service(
        self: &Arc<Self>,
    ) -> Result<Option<AllWriterService<'_>>, MemoryStatus> {
        let loan = match self.try_service_loan()? {
            ServiceLoanAttempt::Acquired(loan) => loan,
            ServiceLoanAttempt::ExecutionBusy | ServiceLoanAttempt::ServiceBusy => return Ok(None),
        };
        let generation = self.generation()?;
        let Some(writers) = self
            .system
            .reservations
            .try_freeze_writers()
            .map_err(|_| MemoryStatus::Domain)?
        else {
            return Ok(None);
        };
        let service = AllWriterService {
            domain: self,
            generation,
            writers,
            _loan: loan,
        };
        service.validate_current()?;
        Ok(Some(service))
    }
}
