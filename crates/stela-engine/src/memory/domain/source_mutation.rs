//! Accounting of writes to mutable instruction RAM.
use super::{MemoryStatus, MemorySystem};
use crate::memory::backing::MemoryBacking;
use crate::platform::sync::{Mutex, MutexGuard};
use alloc::sync::{Arc, Weak};
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

pub(in crate::memory) struct SourceMutationTracker {
    enabled: AtomicBool,
    generation: AtomicU64,
    failed: AtomicBool,
    state: Mutex<Option<SourceMutationState>>,
}
struct SourceMutationState {
    system: Weak<()>,
    generation: u64,
}
impl SourceMutationTracker {
    pub(in crate::memory) fn new() -> Self {
        Self {
            enabled: AtomicBool::new(false),
            generation: AtomicU64::new(0),
            failed: AtomicBool::new(false),
            state: Mutex::new(None),
        }
    }
}

/// Created only right after a tracked byte write. Its private fields bind the
/// backing and the epochs before and after. Not Clone.
pub(crate) struct SourceWriteReceipt {
    backing: Arc<MemoryBacking>,
    offset: usize,
    bytes: usize,
    before: u64,
    after: u64,
}

impl core::fmt::Debug for SourceWriteReceipt {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SourceWriteReceipt")
            .field("backing", &self.backing.id())
            .field("offset", &self.offset)
            .field("bytes", &self.bytes)
            .field("before", &self.before)
            .field("after", &self.after)
            .finish()
    }
}

pub(in crate::memory) struct SourceWriteGuard<'a> {
    backing: &'a Arc<MemoryBacking>,
    system: &'a MemorySystem,
    state: Option<MutexGuard<'a, Option<SourceMutationState>>>,
    offset: usize,
    bytes: usize,
    before: u64,
    after: u64,
    attempted: bool,
    finished: bool,
}
impl SourceWriteGuard<'_> {
    /// Invoke immediately before the native write, inside its existing
    /// reservation gate. No fallible work separates this marker from that write.
    pub(in crate::memory) fn attempting(&mut self) {
        if self.state.is_some() {
            self.backing
                .source_mutation
                .generation
                .store(self.after, Ordering::Release);
        }
        self.attempted = true;
    }
    /// Infallible accounting immediately after known completion. Arc retention
    /// does not allocate. Receipt publication uses pre-reserved caller storage.
    pub(in crate::memory) fn complete(&mut self) -> Option<SourceWriteReceipt> {
        self.finished = true;
        let state = self
            .state
            .as_mut()?
            .as_mut()
            .expect("retained source tracker");
        state.generation = self.after;
        Some(SourceWriteReceipt {
            backing: Arc::clone(self.backing),
            offset: self.offset,
            bytes: self.bytes,
            before: self.before,
            after: self.after,
        })
    }
}
impl Drop for SourceWriteGuard<'_> {
    fn drop(&mut self) {
        if self.state.is_some() && self.attempted && !self.finished {
            // An unknown host-effect outcome never becomes an unchanged source.
            self.backing
                .source_mutation
                .failed
                .store(true, Ordering::Release);
            self.system.failed.store(true, Ordering::Release);
        }
    }
}

impl MemorySystem {
    #[cfg(test)]
    pub(in crate::memory) fn seed_source_mutation_for_test(
        &self,
        backing: &Arc<MemoryBacking>,
        generation: u64,
    ) {
        let tracker = &backing.source_mutation;
        let mut state = tracker.state.lock().unwrap();
        assert!(state.is_none());
        *state = Some(SourceMutationState {
            system: Arc::downgrade(&self.identity),
            generation,
        });
        tracker.generation.store(generation, Ordering::Release);
        tracker.enabled.store(true, Ordering::Release);
    }

    pub(in crate::memory) fn prepare_source_write<'a>(
        &'a self,
        backing: &'a Arc<MemoryBacking>,
        offset: usize,
        bytes: usize,
    ) -> Result<SourceWriteGuard<'a>, MemoryStatus> {
        if self.failed.load(Ordering::Acquire) {
            return Err(MemoryStatus::Domain);
        }
        let mut guard = SourceWriteGuard {
            backing,
            system: self,
            state: None,
            offset,
            bytes,
            before: 0,
            after: 0,
            attempted: false,
            finished: false,
        };
        if !backing.source_mutation.enabled.load(Ordering::Acquire) {
            return Ok(guard);
        }
        if bytes == 0
            || offset
                .checked_add(bytes)
                .is_none_or(|end| end > backing.bytes())
        {
            return Err(MemoryStatus::InvalidRequest);
        }
        let state = backing
            .source_mutation
            .state
            .try_lock()
            .map_err(|_| MemoryStatus::Domain)?;
        let current = state.as_ref().ok_or(MemoryStatus::Domain)?;
        if backing.source_mutation.failed.load(Ordering::Acquire)
            || !current
                .system
                .upgrade()
                .is_some_and(|owner| Arc::ptr_eq(&owner, &self.identity))
        {
            return Err(MemoryStatus::Domain);
        }
        guard.before = current.generation;
        guard.after = current
            .generation
            .checked_add(1)
            .ok_or(MemoryStatus::Domain)?;
        guard.state = Some(state);
        Ok(guard)
    }
}
