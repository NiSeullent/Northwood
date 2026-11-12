//! Exclusive-monitor bookkeeping for a memory domain.
//!
//! Callers own the backing bytes, permission and translation checks, and mapping
//! pins. Each callback performs exactly one architectural access and must not
//! re-enter the domain. A Rust panic fails the whole domain.

use crate::platform::sync::{Mutex, MutexGuard};
use crate::system::events::{EventProducer, MonitorBinding};
use alloc::collections::BTreeMap;
use alloc::sync::{Arc, Weak};
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, Ordering};
use core::{cell::RefCell, marker::PhantomData};

const MIN_ERG_BYTES: u64 = 16;
const MAX_ERG_BYTES: u64 = 2048;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shape {
    Scalar,
    Pair,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MonitorError {
    InvalidErg,
    InvalidStripeCount,
    InvalidRange,
    InvalidShape,
    CrossesGranule,
    WrongDomain,
    DeadMapping,
    Poisoned,
    EventsUnbound,
    EventsAlreadyBound,
    ReservationActive,
    EventBindingUnavailable,
    WrongEventEndpoint,
}

#[derive(Debug, PartialEq, Eq)]
pub enum AccessError<E> {
    Monitor(MonitorError),
    Memory(E),
}

impl<E> From<MonitorError> for AccessError<E> {
    fn from(error: MonitorError) -> Self {
        Self::Monitor(error)
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum ConditionalWrite<T> {
    Failed,
    Stored(T),
}

/// Source-selected invalidation timing; this is distinct from committed bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Invalidation {
    /// Ordinary reached fragment: a callback error means no bytes committed.
    OnCommit,
    /// A source-permitted clear-before-tag-check/attempt policy, including faults.
    BeforeAttempt,
}

/// Immutable view identity keeps the real mapping payload for reservations.
/// Protection/backing/attribute changes revoke this identity and create another.
/// Root must also hold stable memory pins while invoking an access callback.
pub struct Mapping<M, A> {
    payload: Arc<M>,
    attributes: A,
    live: AtomicBool,
}

impl<M, A> Mapping<M, A> {
    pub fn new(payload: Arc<M>, attributes: A) -> Arc<Self> {
        Arc::new(Self {
            payload,
            attributes,
            live: AtomicBool::new(true),
        })
    }

    pub fn payload(&self) -> &Arc<M> {
        &self.payload
    }

    pub fn attributes(&self) -> &A {
        &self.attributes
    }

    pub fn is_live(&self) -> bool {
        self.live.load(Ordering::Acquire)
    }
}

/// Already resolved physical identity. It contains no guest/native RAM pointer.
pub struct Access<B, M, A> {
    backing: Arc<B>,
    mapping: Arc<Mapping<M, A>>,
    offset: u64,
    bytes: u8,
    shape: Shape,
}

impl<B, M, A> Access<B, M, A> {
    pub fn new(
        backing: Arc<B>,
        mapping: Arc<Mapping<M, A>>,
        offset: u64,
        bytes: u8,
        shape: Shape,
    ) -> Result<Self, MonitorError> {
        let valid_shape = match shape {
            Shape::Scalar => matches!(bytes, 1 | 2 | 4 | 8),
            Shape::Pair => matches!(bytes, 8 | 16),
        };
        if !valid_shape {
            return Err(MonitorError::InvalidShape);
        }
        offset
            .checked_add(u64::from(bytes))
            .ok_or(MonitorError::InvalidRange)?;
        Ok(Self {
            backing,
            mapping,
            offset,
            bytes,
            shape,
        })
    }
}

/// One reached writer event, not an entire speculative store footprint.
pub struct WriteRange<B> {
    backing: Arc<B>,
    offset: u64,
    bytes: u64,
}

impl<B> WriteRange<B> {
    pub fn new(backing: Arc<B>, offset: u64, bytes: u64) -> Result<Self, MonitorError> {
        if bytes == 0 || offset.checked_add(bytes).is_none() {
            return Err(MonitorError::InvalidRange);
        }
        Ok(Self {
            backing,
            offset,
            bytes,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Key {
    backing: usize,
    granule: u64,
}

impl Key {
    fn new<B>(backing: &Arc<B>, granule: u64) -> Self {
        Self {
            backing: Arc::as_ptr(backing) as usize,
            granule,
        }
    }
}

/// No integer version or epoch counter exists. A live reservation keeps this
/// allocation and its backing; allocator address reuse cannot revive that mark.
struct Epoch<B> {
    backing: Arc<B>,
    granule: u64,
    subscribers: Mutex<Vec<Weak<EventCell<B>>>>,
}

/// Event scope is chosen by the physical mapping owner, not an opcode name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReservationScope {
    Local,
    Global,
}

struct EventState<B> {
    epoch: Weak<Epoch<B>>,
    // Active GlobalMark keeps this Mapping allocation; no address reuse.
    mapping: usize,
}

struct EventCell<B> {
    endpoint: MonitorBinding,
    current: Mutex<EventState<B>>,
}

/// Global lifetime is independent from local reservation validity.
struct GlobalMark<B, M, A> {
    epoch: Arc<Epoch<B>>,
    mapping: Arc<Mapping<M, A>>,
}

type Entries<B> = BTreeMap<Key, Weak<Epoch<B>>>;
// Only the frozen writer borrows entries through a shared mutex guard.
// Ordinary callers use get_mut() through their exclusive mutex guard.
type Stripe<B> = RefCell<Entries<B>>;

struct Reservation<B, M, A> {
    epoch: Arc<Epoch<B>>,
    mapping: Arc<Mapping<M, A>>,
    offset: u64,
    bytes: u8,
    shape: Shape,
}

/// Rust unwinding is fatal for execution across every stripe, not just the
/// Mutex that happens to observe poisoning. This is not native fault recovery.
struct FailureBoundary<'a>(&'a AtomicBool);

impl Drop for FailureBoundary<'_> {
    fn drop(&mut self) {
        if crate::platform::unwind::panicking() {
            self.0.store(true, Ordering::Release);
        }
    }
}

/// Gates serialize bounded callbacks, never a whole
/// load-exclusive/store-exclusive sequence. Keys in one stripe share a gate but
/// never a reservation. Fairness is whatever std::Mutex provides.
pub struct MonitorDomain<B, M, A> {
    erg_bytes: u64,
    stripes: Vec<Mutex<Stripe<B>>>,
    failed: AtomicBool,
    mapping: PhantomData<fn() -> (M, A)>,
}

pub struct ThreadMonitor<B, M, A> {
    domain: Arc<MonitorDomain<B, M, A>>,
    reservation: Option<Reservation<B, M, A>>,
    global: Option<GlobalMark<B, M, A>>,
    event: Option<Arc<EventCell<B>>>,
}

/// A kept exclusive borrow of the real local monitor. Dropping this
/// preparation changes nothing; commit cannot substitute another reservation.
pub(crate) struct PreparedClear<'a, B, M, A> {
    domain: &'a MonitorDomain<B, M, A>,
    thread: &'a mut ThreadMonitor<B, M, A>,
}
impl<B, M, A> PreparedClear<'_, B, M, A> {
    pub(crate) fn commit(self) {
        self.domain.retire(self.thread.reservation.take());
    }
}

/// Declare before stripe guards so error/unwind cleanup cannot self-deadlock.
struct LocalRetirement<'a, B, M, A> {
    domain: &'a MonitorDomain<B, M, A>,
    saved: Option<Reservation<B, M, A>>,
}
impl<B, M, A> Drop for LocalRetirement<'_, B, M, A> {
    fn drop(&mut self) {
        self.domain.retire(self.saved.take());
    }
}

struct GlobalRetirement<'a, B, M, A> {
    domain: &'a MonitorDomain<B, M, A>,
    saved: Option<GlobalMark<B, M, A>>,
    cell: Option<Arc<EventCell<B>>>,
    open: bool,
}
impl<B, M, A> Drop for GlobalRetirement<'_, B, M, A> {
    fn drop(&mut self) {
        self.domain
            .finish_global(self.saved.take(), self.cell.as_ref(), self.open);
    }
}

/// Keeps every physical writer gate. Acquisition changes no reservation.
/// Only the canonical memory owner uses this while native frames are recovered.
/// Held writes use these guards directly and cannot reenter the monitor.
pub(in crate::memory) struct FrozenWriters<'a, B, M, A> {
    domain: &'a MonitorDomain<B, M, A>,
    gates: Vec<MutexGuard<'a, Stripe<B>>>,
}

impl<B, M, A> FrozenWriters<'_, B, M, A> {
    /// Fail before releasing gates, including writers which already passed an
    /// outer memory-system check and are now waiting for their physical gate.
    pub(in crate::memory) fn fail(&self) {
        self.domain.failed.store(true, Ordering::Release);
    }

    pub(in crate::memory) fn validate_current(&self) -> Result<(), MonitorError> {
        self.domain.check_execution()
    }

    fn invalidate(&self, stripe: usize, range: &WriteRange<B>) -> Result<(), MonitorError> {
        let mut entries = self.gates[stripe].try_borrow_mut().map_err(|_| {
            self.fail();
            MonitorError::Poisoned
        })?;
        self.domain.invalidate(&mut entries, range);
        Ok(())
    }

    /// Invalidate before the attempted write, using the gates already held.
    pub(in crate::memory) fn with_writer<T, E>(
        &self,
        range: &WriteRange<B>,
        write: impl FnOnce() -> Result<T, E>,
    ) -> Result<T, AccessError<E>> {
        let domain = self.domain;
        let _failure = FailureBoundary(&domain.failed);
        self.validate_current()?;
        let first = range.offset / domain.erg_bytes;
        let last = (range.offset + range.bytes - 1) / domain.erg_bytes;
        if first == last {
            let stripe = domain.stripe(Key::new(&range.backing, first));
            self.invalidate(stripe, range)?;
        } else {
            for stripe in domain.writer_stripes(range) {
                self.invalidate(stripe, range)?;
            }
        }
        self.validate_current()?;
        write().map_err(AccessError::Memory)
    }
}

impl<B, M, A> MonitorDomain<B, M, A> {
    pub fn new(erg_bytes: u64, stripe_count: usize) -> Result<Arc<Self>, MonitorError> {
        if !(MIN_ERG_BYTES..=MAX_ERG_BYTES).contains(&erg_bytes) || !erg_bytes.is_power_of_two() {
            return Err(MonitorError::InvalidErg);
        }
        if stripe_count == 0 {
            return Err(MonitorError::InvalidStripeCount);
        }
        Ok(Arc::new(Self {
            erg_bytes,
            stripes: (0..stripe_count)
                .map(|_| Mutex::new(RefCell::new(BTreeMap::new())))
                .collect(),
            failed: AtomicBool::new(false),
            mapping: PhantomData,
        }))
    }

    pub fn thread(self: &Arc<Self>) -> ThreadMonitor<B, M, A> {
        ThreadMonitor {
            domain: Arc::clone(self),
            reservation: None,
            global: None,
            event: None,
        }
    }

    /// Bind one PE endpoint while the context has no reservation or binding.
    /// The endpoint can be bound to only one monitor domain.
    pub fn bind_events(
        &self,
        thread: &mut ThreadMonitor<B, M, A>,
        producer: &EventProducer,
    ) -> Result<(), MonitorError> {
        self.owner(thread)?;
        self.check_execution()?;
        if thread.event.is_some() {
            return Err(MonitorError::EventsAlreadyBound);
        }
        if thread.reservation.is_some() || thread.global.is_some() {
            return Err(MonitorError::ReservationActive);
        }
        let endpoint = producer
            .try_bind_monitor()
            .map_err(|_| MonitorError::EventBindingUnavailable)?;
        thread.event = Some(Arc::new(EventCell {
            endpoint,
            current: Mutex::new(EventState {
                epoch: Weak::new(),
                mapping: 0,
            }),
        }));
        Ok(())
    }

    /// Verify kept real endpoint identity; never rebind or mutate a mark.
    /// The higher owner borrows the live EventEndpoint for the complete entry.
    pub fn check_event_endpoint(
        &self,
        thread: &ThreadMonitor<B, M, A>,
        producer: &EventProducer,
    ) -> Result<(), MonitorError> {
        self.owner(thread)?;
        self.check_execution()?;
        let cell = thread.event.as_ref().ok_or(MonitorError::EventsUnbound)?;
        if cell.endpoint.same_endpoint(producer) {
            Ok(())
        } else {
            Err(MonitorError::WrongEventEndpoint)
        }
    }

    // Only monitor metadata mutexes; endpoint publication itself takes no lock.
    // Poison recovery is terminal teardown, never resumable guest execution.
    fn event_lock<'a, T>(&self, mutex: &'a Mutex<T>) -> MutexGuard<'a, T> {
        mutex.lock().unwrap_or_else(|error| {
            self.failed.store(true, Ordering::Release);
            error.into_inner()
        })
    }

    fn current_is(state: &EventState<B>, epoch: &Arc<Epoch<B>>) -> bool {
        core::ptr::eq(state.epoch.as_ptr(), Arc::as_ptr(epoch))
    }

    fn open_cell(
        &self,
        cell: &EventCell<B>,
        epoch: &Arc<Epoch<B>>,
        mapping: Option<usize>,
    ) -> bool {
        let mut current = self.event_lock(&cell.current);
        if Self::current_is(&current, epoch)
            && mapping.is_none_or(|mapping| current.mapping == mapping)
        {
            current.epoch = Weak::new();
            current.mapping = 0;
            // Concrete infallible atomic publication, no endpoint locks/callbacks.
            cell.endpoint.signal_local();
            true
        } else {
            false
        }
    }

    fn open_epoch(&self, epoch: &Arc<Epoch<B>>, mapping: Option<usize>) {
        let mut subscribers = self.event_lock(&epoch.subscribers);
        subscribers.retain(|subscriber| {
            let Some(cell) = subscriber.upgrade() else {
                return false;
            };
            if self.open_cell(&cell, epoch, mapping) {
                return false;
            }
            Self::current_is(&self.event_lock(&cell.current), epoch)
        });
    }

    fn subscribe(
        &self,
        epoch: &Arc<Epoch<B>>,
        cell: &Arc<EventCell<B>>,
        mapping: &Arc<Mapping<M, A>>,
    ) {
        let mut subscribers = self.event_lock(&epoch.subscribers);
        // Prune stale/dead entries; same reusable cell is never appended twice.
        subscribers.retain(|subscriber| {
            subscriber.upgrade().is_some_and(|other| {
                Arc::ptr_eq(&other, cell)
                    || Self::current_is(&self.event_lock(&other.current), epoch)
            })
        });
        {
            let mut current = self.event_lock(&cell.current);
            current.epoch = Arc::downgrade(epoch);
            current.mapping = Arc::as_ptr(mapping) as usize;
        }
        if !subscribers
            .iter()
            .any(|subscriber| Weak::ptr_eq(subscriber, &Arc::downgrade(cell)))
        {
            subscribers.push(Arc::downgrade(cell));
        }
    }

    fn finish_global(
        &self,
        mark: Option<GlobalMark<B, M, A>>,
        cell: Option<&Arc<EventCell<B>>>,
        open: bool,
    ) {
        let Some(mark) = mark else {
            return;
        };
        let key = Key::new(&mark.epoch.backing, mark.epoch.granule);
        let entries = self.stripes[self.stripe(key)]
            .lock()
            .unwrap_or_else(crate::platform::sync::PoisonError::into_inner);
        if let Some(cell) = cell {
            let mut subscribers = self.event_lock(&mark.epoch.subscribers);
            if open {
                self.open_cell(cell, &mark.epoch, Some(Arc::as_ptr(&mark.mapping) as usize));
            }
            // Same-Epoch retarget is still current: keep its reused subscriber.
            let still_current = Self::current_is(&self.event_lock(&cell.current), &mark.epoch);
            if !still_current {
                subscribers.retain(|subscriber| {
                    !Weak::ptr_eq(subscriber, &Arc::downgrade(cell))
                        && subscriber.strong_count() != 0
                });
            }
        }
        drop(entries);
        drop(mark);
        let mut entries = self.stripes[self.stripe(key)]
            .lock()
            .unwrap_or_else(crate::platform::sync::PoisonError::into_inner);
        let entries = entries.get_mut();
        if entries
            .get(&key)
            .is_some_and(|epoch| epoch.strong_count() == 0)
        {
            entries.remove(&key);
        }
    }

    fn stripe(&self, key: Key) -> usize {
        (key.backing.rotate_left(13) ^ key.granule as usize) % self.stripes.len()
    }

    /// Dispatch/helpers, including CLREX, must check execution viability. Clear
    /// and revocation remain available for teardown after a terminal failure.
    pub fn check_execution(&self) -> Result<(), MonitorError> {
        if self.failed.load(Ordering::Acquire) {
            return Err(MonitorError::Poisoned);
        }
        Ok(())
    }

    fn lock(&self, stripe: usize) -> Result<MutexGuard<'_, Stripe<B>>, MonitorError> {
        self.check_execution()?;
        let guard = self.stripes[stripe].lock().map_err(|_| {
            self.failed.store(true, Ordering::Release);
            MonitorError::Poisoned
        })?;
        // A callback on another stripe may have unwound while this thread waited.
        self.check_execution()?;
        Ok(guard)
    }

    fn owner(&self, thread: &ThreadMonitor<B, M, A>) -> Result<(), MonitorError> {
        if !core::ptr::eq(self, Arc::as_ptr(&thread.domain)) {
            return Err(MonitorError::WrongDomain);
        }
        Ok(())
    }

    fn key(&self, access: &Access<B, M, A>) -> Result<Key, MonitorError> {
        if access.offset % self.erg_bytes + u64::from(access.bytes) > self.erg_bytes {
            return Err(MonitorError::CrossesGranule);
        }
        Ok(Key::new(&access.backing, access.offset / self.erg_bytes))
    }

    fn retire(&self, reservation: Option<Reservation<B, M, A>>) {
        let Some(reservation) = reservation else {
            return;
        };
        let key = Key::new(&reservation.epoch.backing, reservation.epoch.granule);
        // Drop before cleanup. Two simultaneous final clears must not both see
        // strong_count>1, leave the entry, then drop the last two references.
        drop(reservation);
        let mut entries = self.stripes[self.stripe(key)]
            .lock()
            .unwrap_or_else(crate::platform::sync::PoisonError::into_inner);
        let entries = entries.get_mut();
        if entries
            .get(&key)
            .is_some_and(|epoch| epoch.strong_count() == 0)
        {
            entries.remove(&key);
        }
    }

    pub(crate) fn prepare_clear<'a>(
        &'a self,
        thread: &'a mut ThreadMonitor<B, M, A>,
    ) -> Result<PreparedClear<'a, B, M, A>, MonitorError> {
        self.check_execution()?;
        self.owner(thread)?;
        Ok(PreparedClear {
            domain: self,
            thread,
        })
    }

    pub fn clear(&self, thread: &mut ThreadMonitor<B, M, A>) -> Result<(), MonitorError> {
        let _failure = FailureBoundary(&self.failed);
        self.owner(thread)?;
        self.retire(thread.reservation.take());
        Ok(())
    }

    pub fn load<T, E>(
        &self,
        thread: &mut ThreadMonitor<B, M, A>,
        access: &Access<B, M, A>,
        read: impl FnOnce() -> Result<T, E>,
    ) -> Result<T, AccessError<E>> {
        self.load_with_scope(thread, access, ReservationScope::Local, read)
    }

    pub fn load_with_scope<T, E>(
        &self,
        thread: &mut ThreadMonitor<B, M, A>,
        access: &Access<B, M, A>,
        scope: ReservationScope,
        read: impl FnOnce() -> Result<T, E>,
    ) -> Result<T, AccessError<E>> {
        let _failure = FailureBoundary(&self.failed);
        self.owner(thread)?;
        let key = self.key(access)?;
        if scope == ReservationScope::Global && thread.event.is_none() {
            return Err(MonitorError::EventsUnbound.into());
        }
        // Local replacement never itself opens the independent global mark.
        self.retire(thread.reservation.take());
        let mut old_global = GlobalRetirement {
            domain: self,
            saved: None,
            cell: thread.event.clone(),
            open: false,
        };
        let mut pending = LocalRetirement {
            domain: self,
            saved: None,
        };
        let value = {
            let mut entries = self.lock(self.stripe(key))?;
            let entries = entries.get_mut();
            let _gate_failure = FailureBoundary(&self.failed);
            if !access.mapping.is_live() {
                return Err(MonitorError::DeadMapping.into());
            }
            // Error/panic leaves the prior global hold and current cell intact.
            let value = read().map_err(AccessError::Memory)?;
            let epoch = entries
                .get(&key)
                .and_then(Weak::upgrade)
                .unwrap_or_else(|| {
                    let epoch = Arc::new(Epoch {
                        backing: Arc::clone(&access.backing),
                        granule: key.granule,
                        subscribers: Mutex::new(Vec::new()),
                    });
                    entries.insert(key, Arc::downgrade(&epoch));
                    epoch
                });
            pending.saved = Some(Reservation {
                epoch: Arc::clone(&epoch),
                mapping: Arc::clone(&access.mapping),
                offset: access.offset,
                bytes: access.bytes,
                shape: access.shape,
            });
            if scope == ReservationScope::Global {
                let cell = thread
                    .event
                    .as_ref()
                    .expect("global scope preflighted binding");
                self.subscribe(&epoch, cell, &access.mapping);
                old_global.saved = thread.global.replace(GlobalMark {
                    epoch,
                    mapping: Arc::clone(&access.mapping),
                });
            }
            self.check_execution()?;
            value
        };
        thread.reservation = pending.saved.take();
        drop(old_global); // Old stripe reacquisition occurs after new gate release.
        Ok(value)
    }

    pub fn store<T, E>(
        &self,
        thread: &mut ThreadMonitor<B, M, A>,
        access: &Access<B, M, A>,
        write: impl FnOnce() -> Result<T, E>,
    ) -> Result<ConditionalWrite<T>, AccessError<E>>
    where
        A: PartialEq,
    {
        let _failure = FailureBoundary(&self.failed);
        self.owner(thread)?;
        let key = self.key(access)?;
        let saved = LocalRetirement {
            domain: self,
            saved: thread.reservation.take(),
        };
        let result = {
            let mut entries = self.lock(self.stripe(key))?;
            let entries = entries.get_mut();
            let _gate_failure = FailureBoundary(&self.failed);
            if !access.mapping.is_live() {
                Err(MonitorError::DeadMapping.into())
            } else {
                let valid = saved.saved.as_ref().is_some_and(|saved| {
                    saved.mapping.is_live()
                        && saved.mapping.attributes == access.mapping.attributes
                        && saved.offset == access.offset
                        && saved.bytes == access.bytes
                        && saved.shape == access.shape
                        && entries
                            .get(&key)
                            .and_then(Weak::upgrade)
                            .is_some_and(|epoch| Arc::ptr_eq(&saved.epoch, &epoch))
                });
                if valid {
                    // Consume and invalidate before the attempted conditional
                    // write; a callback fault cannot leave a reusable mark.
                    entries.remove(&key);
                    self.open_epoch(
                        &saved.saved.as_ref().expect("valid saved reservation").epoch,
                        None,
                    );
                    self.check_execution()?;
                    write()
                        .map(ConditionalWrite::Stored)
                        .map_err(AccessError::Memory)
                } else {
                    Ok(ConditionalWrite::Failed)
                }
            }
        };
        drop(saved);
        // Stored and Failed are the only retired conditional outcomes. Errors
        // are not silently classified as retired StoreExcl or taken DataAbort.
        if result.is_ok() {
            self.finish_global(thread.global.take(), thread.event.as_ref(), true);
        }
        result
    }

    fn writer_stripes(&self, range: &WriteRange<B>) -> Vec<usize> {
        let first = range.offset / self.erg_bytes;
        let last = (range.offset + range.bytes - 1) / self.erg_bytes;
        if last - first >= self.stripes.len() as u64 - 1 {
            return (0..self.stripes.len()).collect();
        }
        let mut stripes: Vec<_> = (first..=last)
            .map(|granule| self.stripe(Key::new(&range.backing, granule)))
            .collect();
        stripes.sort_unstable();
        stripes.dedup();
        stripes
    }

    fn invalidate(&self, entries: &mut Entries<B>, range: &WriteRange<B>) {
        let backing = Arc::as_ptr(&range.backing) as usize;
        let first = range.offset / self.erg_bytes;
        let last = (range.offset + range.bytes - 1) / self.erg_bytes;
        // Bound work by live reservation metadata, not bytes/ERG in a large
        // copyout. Gate collisions never invalidate unrelated physical keys.
        entries.retain(|key, epoch| {
            if key.backing == backing && (first..=last).contains(&key.granule) {
                if let Some(epoch) = epoch.upgrade() {
                    self.open_epoch(&epoch, None);
                }
                false
            } else {
                epoch.strong_count() != 0
            }
        });
    }

    /// Hold every physical writer gate for an observer with no architectural
    /// store. A native atomic-read recipe may perform permitted unchanged-value
    /// host writeback, but must not mutate data or retire a source reservation.
    /// Source writes must continue to use `with_writer`/`store` instead.
    pub(in crate::memory) fn with_observer<T, E>(
        &self,
        range: &WriteRange<B>,
        read: impl FnOnce() -> Result<T, E>,
    ) -> Result<T, AccessError<E>> {
        let _failure = FailureBoundary(&self.failed);
        let first = range.offset / self.erg_bytes;
        let last = (range.offset + range.bytes - 1) / self.erg_bytes;
        if first == last {
            let key = Key::new(&range.backing, first);
            let _gate = self.lock(self.stripe(key))?;
            let _gate_failure = FailureBoundary(&self.failed);
            let result = read().map_err(AccessError::Memory)?;
            self.check_execution()?;
            return Ok(result);
        }
        // Same stable order as writers, including aliases and shared ERGs.
        let _gates: Vec<_> = self
            .writer_stripes(range)
            .into_iter()
            .map(|stripe| self.lock(stripe))
            .collect::<Result<_, _>>()?;
        let _gate_failure = FailureBoundary(&self.failed);
        let result = read().map_err(AccessError::Memory)?;
        self.check_execution()?;
        Ok(result)
    }

    pub fn with_writer<T, E>(
        &self,
        range: &WriteRange<B>,
        policy: Invalidation,
        write: impl FnOnce() -> Result<T, E>,
    ) -> Result<T, AccessError<E>> {
        let _failure = FailureBoundary(&self.failed);
        let first = range.offset / self.erg_bytes;
        let last = (range.offset + range.bytes - 1) / self.erg_bytes;
        if first == last {
            // The usual architectural part touches one reservation granule.
            // Keep its guard on the stack instead of allocating two vectors.
            let key = Key::new(&range.backing, first);
            let mut entries = self.lock(self.stripe(key))?;
            let _gate_failure = FailureBoundary(&self.failed);
            if policy == Invalidation::BeforeAttempt {
                self.invalidate(entries.get_mut(), range);
                self.check_execution()?;
            }
            let result = write();
            if policy == Invalidation::OnCommit && result.is_ok() {
                self.invalidate(entries.get_mut(), range);
                self.check_execution()?;
            }
            return result.map_err(AccessError::Memory);
        }
        let stripes = self.writer_stripes(range);
        // Stable acquisition order prevents cycles for cross-granule fragments.
        let mut guards: Vec<_> = stripes
            .into_iter()
            .map(|stripe| self.lock(stripe))
            .collect::<Result<_, _>>()?;
        let _gate_failure = FailureBoundary(&self.failed);
        if policy == Invalidation::BeforeAttempt {
            for entries in &mut guards {
                self.invalidate(entries.get_mut(), range);
            }
            self.check_execution()?;
        }
        let result = write();
        if policy == Invalidation::OnCommit && result.is_ok() {
            for entries in &mut guards {
                self.invalidate(entries.get_mut(), range);
            }
            self.check_execution()?;
        }
        result.map_err(AccessError::Memory)
    }

    /// Nonblocking acquisition in the same ascending order as cross-granule
    /// writers. Busy acquisition releases its prefix, performs no callback and
    /// does not invalidate a reservation. Poisoning remains terminal.
    pub(in crate::memory) fn try_freeze_writers(
        &self,
    ) -> Result<Option<FrozenWriters<'_, B, M, A>>, MonitorError> {
        self.check_execution()?;
        let mut gates = Vec::with_capacity(self.stripes.len());
        for stripe in &self.stripes {
            #[cfg(feature = "std")]
            let gate = match stripe.try_lock() {
                Ok(gate) => gate,
                Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
                Err(std::sync::TryLockError::Poisoned(_)) => {
                    self.failed.store(true, Ordering::Release);
                    return Err(MonitorError::Poisoned);
                }
            };
            #[cfg(not(feature = "std"))]
            let gate = match stripe.try_lock() {
                Ok(gate) => gate,
                Err(_) => return Ok(None),
            };
            gates.push(gate);
        }
        self.check_execution()?;
        Ok(Some(FrozenWriters {
            domain: self,
            gates,
        }))
    }

    pub fn revoke_mapping(&self, mapping: &Mapping<M, A>) -> Result<(), MonitorError> {
        let _failure = FailureBoundary(&self.failed);
        // Declare before gates so generic backing payload destruction is outside all gates.
        let mut deferred_epochs = Vec::new();
        let mut guards = self
            .stripes
            .iter()
            .map(|stripe| stripe.lock().map_err(|_| MonitorError::Poisoned))
            .collect::<Result<Vec<_>, _>>()?;
        mapping.live.store(false, Ordering::Release);
        // Finite mapping-lifetime spurious opening; not universal ISA semantics.
        for entries in &mut guards {
            for weak in entries.get_mut().values() {
                if let Some(epoch) = weak.upgrade() {
                    self.open_epoch(&epoch, Some(core::ptr::from_ref(mapping) as usize));
                    deferred_epochs.push(epoch);
                }
            }
        }
        Ok(())
    }

    #[cfg(test)]
    fn entry_count(&self) -> usize {
        self.stripes
            .iter()
            .map(|stripe| stripe.lock().unwrap().get_mut().len())
            .sum()
    }
}

impl<B, M, A> Drop for ThreadMonitor<B, M, A> {
    fn drop(&mut self) {
        let _failure = FailureBoundary(&self.domain.failed);
        self.domain.retire(self.reservation.take());
        // Finite PE-owner teardown opening under B2.12.5, with mandatory event.
        self.domain
            .finish_global(self.global.take(), self.event.as_ref(), true);
    }
}

#[cfg(test)]
mod tests;
