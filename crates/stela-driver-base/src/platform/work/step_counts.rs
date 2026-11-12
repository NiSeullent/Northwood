//! Where the dispatcher's steps spend their time: relaxed counters a report
//! swaps out. Four counter reads and a few relaxed adds per step.
use core::sync::atomic::{AtomicU64, Ordering};

/// Time-stamp counter ticks and event counts since the last [`take`].
pub struct StepCounts {
    /// Entry to the context switch of steps that ran a task.
    pub select: AtomicU64,
    /// The task's run, from the switch to its return.
    pub body: AtomicU64,
    /// From the task's return to the step's exit.
    pub after: AtomicU64,
    /// Entry to exit of steps that found nothing to run.
    pub empty: AtomicU64,
    /// Steps that found nothing to run.
    pub empty_steps: AtomicU64,
    /// Job-slot scans for idle tasks.
    pub scans: AtomicU64,
    /// Queue lock acquisitions.
    pub locks: AtomicU64,
}

/// The process-wide counters.
pub static STEP_COUNTS: StepCounts = StepCounts {
    select: AtomicU64::new(0),
    body: AtomicU64::new(0),
    after: AtomicU64::new(0),
    empty: AtomicU64::new(0),
    empty_steps: AtomicU64::new(0),
    scans: AtomicU64::new(0),
    locks: AtomicU64::new(0),
};

/// One snapshot of [`StepCounts`], reset by [`take`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StepSnapshot {
    pub select: u64,
    pub body: u64,
    pub after: u64,
    pub empty: u64,
    pub empty_steps: u64,
    pub scans: u64,
    pub locks: u64,
}

/// Swap every counter to zero and return what it held.
pub fn take() -> StepSnapshot {
    let take = |counter: &AtomicU64| counter.swap(0, Ordering::Relaxed);
    let counts = &STEP_COUNTS;
    StepSnapshot {
        select: take(&counts.select),
        body: take(&counts.body),
        after: take(&counts.after),
        empty: take(&counts.empty),
        empty_steps: take(&counts.empty_steps),
        scans: take(&counts.scans),
        locks: take(&counts.locks),
    }
}

pub(super) fn add(counter: &AtomicU64, value: u64) {
    counter.fetch_add(value, Ordering::Relaxed);
}

pub(super) fn now() -> u64 {
    // SAFETY: RDTSC has no memory effects; x86-64 always has it.
    unsafe { core::arch::x86_64::_rdtsc() }
}
