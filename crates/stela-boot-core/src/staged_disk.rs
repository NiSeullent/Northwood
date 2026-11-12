//! A host disk whose commands a device model starts under its devices' lock,
//! and whose host I/O runs outside that lock.
//!
//! A storage model starts and finishes its commands under the lock every
//! processor's device accesses wait for, and a host disk can take
//! milliseconds to take or give back a command: its drivers' lock, its
//! bounce copies, and its doorbell and register accesses, each an exit to a
//! hypervisor. So the model's disk is a [`StagedDisk`]: a start only queues
//! the command and a finish only takes a result the host disk already gave,
//! each under a short lock of its own. A processor that holds no device lock
//! gives the host disk the queued commands and takes its completions, in
//! bounded steps ([`DiskWorker::step`]). A start, or a finish that finds no
//! result while the host disk has work, calls the stage's `wake` once until
//! the next step, so a processor comes to take that step.
//!
//! Until a stepping processor exists ([`DiskWorker::set_staged`]), and
//! again once none does, the model's own starts and finishes take the steps:
//! a caller that polls for a result before the PEs run, or after they stop,
//! makes progress by itself, as an unstaged disk would.
//!
//! The host disk takes the commands in the order the model started them, and
//! a result reaches the model only once the host disk completed its command:
//! a write the model saw complete is on the host disk before any command the
//! model started after that, so a later read sees it and a later flush makes
//! it durable, and a flush completes only once the host disk's has. The
//! model's synchronous calls ([`BlockDevice::read`] and the like, a shutdown's
//! flush) wait for every queued command first, then run on the host disk.
use alloc::{boxed::Box, collections::BTreeMap, collections::VecDeque, sync::Arc, vec::Vec};
use core::cell::UnsafeCell;
use core::ops::{Deref, DerefMut};
use core::sync::atomic::{AtomicBool, Ordering};
use stela_drivers::block::{BlockDevice, BlockError, Command, Finished, Identity};

/// The commands the model may have queued, at the host disk or finished and
/// not yet taken: as many as a block layer window keeps.
pub const DEPTH: usize = 64;

/// The commands one step gives a host disk that queues them. A disk that
/// runs each where it is asked runs one a step.
const STARTS: usize = 16;

/// One processor at a time; never held across a wait for another lock.
struct Spin<T> {
    held: AtomicBool,
    value: UnsafeCell<T>,
}

// SAFETY: a host disk's drivers reach their device through their own state
// and its registers, not through the processor that made them (as for
// `block::Shared`), and `held` admits one processor at a time.
unsafe impl<T> Send for Spin<T> {}
// SAFETY: as for `Send`; every access holds `held`.
unsafe impl<T> Sync for Spin<T> {}

struct Held<'a, T>(&'a Spin<T>);

impl<T> Spin<T> {
    fn new(value: T) -> Self {
        Self {
            held: AtomicBool::new(false),
            value: UnsafeCell::new(value),
        }
    }

    /// The guard, made only once this processor took `held`: a guard's drop
    /// releases it, so a failed attempt must make none.
    fn try_lock(&self) -> Option<Held<'_, T>> {
        self.held
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
            .then(|| Held(self))
    }

    /// Wait for the holder, answering at each turn what a holder inside a
    /// driver may wait on ([`stela_driver_base::on_spin`]).
    fn lock(&self) -> Held<'_, T> {
        loop {
            if let Some(held) = self.try_lock() {
                return held;
            }
            stela_driver_base::spinning();
        }
    }
}

impl<T> Deref for Held<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        // SAFETY: `held` excludes every other reference while this lives.
        unsafe { &*self.0.value.get() }
    }
}

impl<T> DerefMut for Held<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: as for `deref`.
        unsafe { &mut *self.0.value.get() }
    }
}

impl<T> Drop for Held<'_, T> {
    fn drop(&mut self) {
        self.0.held.store(false, Ordering::Release);
    }
}

/// What the model and the host side share: the queue, under a lock no one
/// holds while calling the host disk, and the host disk, under its own.
struct Staging {
    queue: Spin<Queue>,
    host: Spin<Host>,
    /// Whether a result waits for the model, read without a lock.
    ready: AtomicBool,
    /// Whether a command waits for the host disk or runs on it.
    work: AtomicBool,
    /// Asks a processor for a step; `called` since the last step.
    wake: fn(),
    called: AtomicBool,
    /// Whether processors step the host side; until then the model does.
    staged: AtomicBool,
}

#[derive(Default)]
struct Queue {
    /// Commands the model started that the host disk has not taken, in the
    /// order they were started, by tag.
    waiting: VecDeque<(u32, Command)>,
    /// Results of commands the host disk completed, by tag, that the model
    /// has not taken.
    finished: BTreeMap<u32, Finished>,
    /// Tags given and not yet taken back: waiting, at the host or finished.
    owned: usize,
    next: u32,
}

struct Host {
    device: Box<dyn BlockDevice>,
    /// Commands the host disk runs: its tag and the model's.
    running: Vec<(u32, u32)>,
    /// When a step last looked for completions the disk had not posted.
    looked: u64,
}

/// What a [`DiskWorker::step`] did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Step {
    /// Commands it gave the host disk and results it took from it.
    pub started: usize,
    pub finished: usize,
    /// Whether commands still wait for the host disk or run on it.
    pub work: bool,
    /// Whether another processor held the host disk, so the step did nothing.
    pub held: bool,
}

impl Step {
    pub fn progressed(&self) -> bool {
        self.started != 0 || self.finished != 0
    }
}

/// The model's side of a staged host disk, under the devices' lock.
pub struct StagedDisk {
    identity: Identity,
    staging: Arc<Staging>,
}

/// The host side, stepped outside the devices' lock.
#[derive(Clone)]
pub struct DiskWorker {
    staging: Arc<Staging>,
}

/// Stage `device`: the model's disk and its host side, which `wake` asks a
/// processor to step. `wake` runs under the model's lock: it must not wait.
pub fn stage(device: Box<dyn BlockDevice>, wake: fn()) -> (StagedDisk, DiskWorker) {
    let identity = device.identity().clone();
    let staging = Arc::new(Staging {
        queue: Spin::new(Queue::default()),
        host: Spin::new(Host {
            device,
            running: Vec::new(),
            looked: 0,
        }),
        ready: AtomicBool::new(false),
        work: AtomicBool::new(false),
        wake,
        called: AtomicBool::new(false),
        staged: AtomicBool::new(false),
    });
    (
        StagedDisk {
            identity,
            staging: Arc::clone(&staging),
        },
        DiskWorker { staging },
    )
}

impl Staging {
    /// Take the host disk's completions (`reap`), then give it up to `most`
    /// waiting commands, in order, while it has room; post what completed.
    /// A disk whose one finish collects every completion is reached once a
    /// reap; the other results come from what it collected.
    fn advance(&self, host: &mut Host, most: usize, reap: bool) -> Step {
        self.advance_within(host, most, reap, &mut || false)
    }

    /// [`Self::advance`], ending once `over` says the step's time is spent,
    /// after at least one call of the host disk: what is left keeps for the
    /// next step, so no step holds its processor for long whatever one
    /// call of the disk takes.
    fn advance_within(
        &self,
        host: &mut Host,
        most: usize,
        reap: bool,
        over: &mut dyn FnMut() -> bool,
    ) -> Step {
        let mut done = Vec::new();
        let together = reap && host.device.reaps_together();
        let mut looked = false;
        let mut index = 0;
        let mut calls = 0usize;
        while reap && let Some(&(tag, ours)) = host.running.get(index) {
            if calls != 0 && over() {
                // The next step looks again at once: a disk that collects
                // its completions together may hold the rest unposted.
                host.looked = 0;
                break;
            }
            calls += 1;
            let result = if together && looked {
                host.device.finished(tag)
            } else {
                looked = true;
                host.device.finish(tag)
            };
            match result {
                Some(result) => {
                    host.running.swap_remove(index);
                    done.push((ours, result));
                }
                None => index += 1,
            }
        }
        let finished = done.len();
        let queued = host.device.depth() > 0;
        let mut started = 0;
        while started < most && (!queued || host.device.room()) {
            if calls != 0 && over() {
                break;
            }
            calls += 1;
            let Some((ours, command)) = self.queue.lock().waiting.pop_front() else {
                break;
            };
            started += 1;
            if !queued {
                done.push((ours, host.device.execute(command)));
                break;
            }
            match host.device.start(command) {
                Ok(tag) => host.running.push((tag, ours)),
                Err(error) => done.push((ours, Err(error))),
            }
        }
        let mut queue = self.queue.lock();
        for (ours, result) in done {
            queue.finished.insert(ours, result);
        }
        let work = !queue.waiting.is_empty() || !host.running.is_empty();
        self.ready
            .store(!queue.finished.is_empty(), Ordering::Release);
        self.work.store(work, Ordering::Release);
        Step {
            started,
            finished,
            work,
            held: false,
        }
    }

    /// Ask for a step while the host disk has work, once until it is taken;
    /// with no stepping processor, take it here.
    fn call(&self) {
        if !self.work.load(Ordering::Acquire) {
            return;
        }
        if !self.staged.load(Ordering::Acquire) {
            if let Some(mut host) = self.host.try_lock() {
                self.advance(&mut host, STARTS, true);
            }
        } else if !self.called.swap(true, Ordering::AcqRel) {
            (self.wake)();
        }
    }

    /// Run every queued command to completion, then `run` on the host disk.
    fn direct<T>(&self, run: impl FnOnce(&mut dyn BlockDevice) -> T) -> T {
        let mut host = self.host.lock();
        while self.advance(&mut host, usize::MAX, true).work {
            stela_driver_base::waiting();
            core::hint::spin_loop();
        }
        run(&mut *host.device)
    }
}

impl DiskWorker {
    /// Whether processors step the host side from now on (`on`), or the
    /// model's own calls do, as before the PEs run and after they stop.
    pub fn set_staged(&self, on: bool) {
        self.staging.staged.store(on, Ordering::Release);
    }

    /// Whether a command waits for the host disk or runs on it.
    pub fn work(&self) -> bool {
        self.staging.work.load(Ordering::Acquire)
    }

    /// One bounded step at time `now`: take the host disk's completions and
    /// give it the next commands. Another processor's step leaves this one
    /// nothing. Completions are looked for once the disk has posted one (a
    /// look at memory), or `lost` after the last look without, for one whose
    /// signal was lost; a step with nothing to do reaches no register.
    pub fn step(&self, now: u64, lost: u64) -> Step {
        self.step_within(now, lost, &mut || false)
    }

    /// [`Self::step`], ending once `over` says its time is spent, after
    /// at least one call of the host disk; what is left keeps for the next
    /// step, which `work` still shows.
    pub fn step_within(&self, now: u64, lost: u64, over: &mut dyn FnMut() -> bool) -> Step {
        let Some(mut host) = self.staging.host.try_lock() else {
            return Step {
                work: self.work(),
                held: true,
                ..Step::default()
            };
        };
        // A look after this asks for the next step.
        self.staging.called.store(false, Ordering::Release);
        let reap = !host.running.is_empty()
            && (host.device.posted() || now.saturating_sub(host.looked) >= lost);
        if reap {
            host.looked = now;
        }
        self.staging.advance_within(&mut host, STARTS, reap, over)
    }
}

/// When a look at a signalling host disk's completion queue may be skipped.
///
/// A disk that signals its completions moves the host's doorbell count, so
/// a look that found nothing need not be repeated until the count moves.
/// But the count moves only when a core takes the disk's interrupt, and a
/// busy core runs guest code with interrupts off: it takes the signal only
/// at its next interrupt window, milliseconds away, while the completion is
/// already in memory. So a fruitless look excuses the next ones only for
/// `window` counter ticks; after that a look reads memory again, whatever
/// the count says. A look costs a root change and a read of one word.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LookGate {
    /// The doorbell count and the counter of the last look that found
    /// nothing.
    fruitless: Option<(u64, u64)>,
}

impl LookGate {
    /// Whether a look at counter `now`, with the doorbell count at
    /// `doorbells`, can be skipped for a disk that signals (`signals`).
    pub fn skip(&self, signals: bool, doorbells: u64, now: u64, window: u64) -> bool {
        signals
            && self
                .fruitless
                .is_some_and(|(count, at)| count == doorbells && now.wrapping_sub(at) < window)
    }

    /// What a look at `now`, the count read before it at `doorbells`, found.
    pub fn looked(&mut self, posted: bool, doorbells: u64, now: u64) {
        self.fruitless = (!posted).then_some((doorbells, now));
    }

    /// A start or a finish changed what the disk may post.
    pub fn reset(&mut self) {
        self.fruitless = None;
    }
}

/// Whether a core that runs guest code (`whole` false) or waits for one
/// (`whole`) steps the staged disks now, at counter `now`: a waiting one
/// always; a busy one when no core stepped since `fallback` ticks before
/// (`stepped`), or when the doorbell count moved past `seen`, the count at
/// the last step: a host disk's completion signal has come since, and the
/// completion waits in memory for a step to take it.
pub fn step_due(
    whole: bool,
    now: u64,
    stepped: u64,
    fallback: u64,
    doorbells: u64,
    seen: u64,
) -> bool {
    whole || now.saturating_sub(stepped) >= fallback || doorbells != seen
}

impl BlockDevice for StagedDisk {
    fn identity(&self) -> &Identity {
        &self.identity
    }
    fn read(&mut self, lba: u64, buffer: &mut [u8]) -> Result<(), BlockError> {
        self.staging.direct(|device| device.read(lba, buffer))
    }
    fn write(&mut self, lba: u64, data: &[u8]) -> Result<(), BlockError> {
        self.staging.direct(|device| device.write(lba, data))
    }
    fn flush_cache(&mut self) -> Result<(), BlockError> {
        self.staging.direct(|device| device.flush_cache())
    }
    fn write_fua(&mut self, lba: u64, data: &[u8]) -> Result<(), BlockError> {
        self.staging.direct(|device| device.write_fua(lba, data))
    }
    fn discard(&mut self, lba: u64, blocks: u64) -> Result<(), BlockError> {
        self.staging.direct(|device| device.discard(lba, blocks))
    }
    fn write_zeroes(&mut self, lba: u64, blocks: u64, unmap: bool) -> Result<(), BlockError> {
        self.staging
            .direct(|device| device.write_zeroes(lba, blocks, unmap))
    }
    fn flush(&mut self) -> Result<(), BlockError> {
        self.staging.direct(|device| device.flush())
    }
    fn shutdown(&mut self, power_off: bool) -> Result<(), BlockError> {
        self.staging.direct(|device| device.shutdown(power_off))
    }
    fn execute(&mut self, command: Command) -> Finished {
        self.staging.direct(|device| device.execute(command))
    }

    fn depth(&self) -> usize {
        DEPTH
    }

    fn room(&self) -> bool {
        self.staging.queue.lock().owned < DEPTH
    }

    /// Queue `command` for the host disk; it never waits for the disk.
    fn start(&mut self, command: Command) -> Result<u32, BlockError> {
        let mut queue = self.staging.queue.lock();
        if queue.owned >= DEPTH {
            return Err(BlockError::Protocol("the staged disk queue is full"));
        }
        let tag = queue.next;
        queue.next = queue.next.wrapping_add(1);
        queue.owned += 1;
        queue.waiting.push_back((tag, command));
        self.staging.work.store(true, Ordering::Release);
        drop(queue);
        self.staging.call();
        Ok(tag)
    }

    fn finish(&mut self, tag: u32) -> Option<Finished> {
        if !self.staging.ready.load(Ordering::Acquire) {
            self.staging.call();
            if !self.staging.ready.load(Ordering::Acquire) {
                return None;
            }
        }
        let mut queue = self.staging.queue.lock();
        let Some(result) = queue.finished.remove(&tag) else {
            drop(queue);
            self.staging.call();
            return None;
        };
        queue.owned -= 1;
        self.staging
            .ready
            .store(!queue.finished.is_empty(), Ordering::Release);
        Some(result)
    }

    /// A result waits for the model: a look at one flag.
    fn posted(&self) -> bool {
        self.staging.ready.load(Ordering::Acquire)
    }

    /// A step outside the lock, not a signal, posts a completion, so the
    /// model looks for it as for a disk that sends none.
    fn signals(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests;
