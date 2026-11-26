// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2006, Red Hat, Inc., Ingo Molnar
//! Each AP's time-stamp counter, brought into agreement with the BSP's
//! before the AP takes work, after Linux's `arch/x86/kernel/tsc_sync.c`.
//!
//! The engine and the drivers measure intervals as differences of counter
//! reads that may be taken on different cores: a deadline started on one
//! PE's core and checked on another's. Firmware may leave the cores'
//! counters apart. As `check_tsc_sync_source` and `check_tsc_sync_target`
//! do, the BSP and the AP take turns reading their counters under one lock
//! for a round (`stela_boot_core::tsc::sync::round_millis`), each noting
//! the most the shared last value went back. A clean round ends it only
//! when both processors observed readings from their peer.
//! Otherwise the AP moves its own counter by the warp through
//! `IA32_TSC_ADJUST`, forward when it saw the counter go back and so ran
//! behind, else back, and tries again. Linux finds the sync mechanism's own
//! overhead leaves at most a third round needed. Where the AP has no
//! `IA32_TSC_ADJUST`, or its writes do not converge in [`ROUNDS`], the AP
//! instead adds a software offset to the counter it reads in further
//! rounds, and once those agree counts its guest clock from a base moved by
//! that offset (`tsc::sync::corrected_base`). An AP that agrees in neither
//! way is refused.
use crate::Result;
use crate::cpu::msr::TSC_ADJUST;
use core::{
    arch::{
        asm,
        x86_64::{__cpuid_count, _rdtsc},
    },
    sync::atomic::{AtomicBool, AtomicI64, AtomicU32, Ordering},
};
use stela_boot_core::tsc::sync::{Samples, Side, correction};
use stela_contracts::refuse;

/// Rounds of each correction, through `IA32_TSC_ADJUST` and in software.
const ROUNDS: u32 = 4;
/// A sum of delayed, non-overlapping rounds that never gave usable samples.
const APART_ROUNDS: u32 = 8;

/// One AP's synchronization with the BSP.
pub(super) struct Sync {
    round_ticks: u64,
    rendezvous_ticks: u64,
    /// Arrivals at the rendezvous points; each point needs both cores.
    arrivals: AtomicU32,
    aborted: AtomicBool,
    lock: AtomicBool,
    samples: Samples,
    /// Set by the AP when it has decided, before the last rendezvous.
    finished: AtomicBool,
    /// What the AP added to its `IA32_TSC_ADJUST`, its software offset,
    /// and its rounds.
    adjusted: AtomicI64,
    offset: AtomicI64,
    rounds: AtomicU32,
    apart: AtomicU32,
}

/// How an AP's counter came to agree with the BSP's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Outcome {
    /// Added to its `IA32_TSC_ADJUST`.
    pub(super) adjusted: i64,
    /// Added in software to every value it reads: its counter base moves by
    /// this much.
    pub(super) offset: i64,
    pub(super) rounds: u32,
    pub(super) apart: u32,
}

/// The counter, read after every earlier load and store, as Linux's
/// `rdtsc_ordered`.
fn ordered() -> u64 {
    // SAFETY: fences and RDTSC have no effect beyond ordering.
    unsafe { asm!("mfence", "lfence", options(nostack, preserves_flags)) };
    // SAFETY: RDTSC has no side effects; x86-64 always has it.
    unsafe { _rdtsc() }
}

impl Sync {
    /// A round of `round_millis` (`tsc::sync::round_millis`).
    pub(super) fn new(tsc_hz: u64, round_millis: u64) -> Self {
        Self {
            round_ticks: tsc_hz / 1000 * round_millis,
            // The AP can already be waiting while the BSP brings others up.
            rendezvous_ticks: tsc_hz.saturating_mul(2),
            arrivals: AtomicU32::new(0),
            aborted: AtomicBool::new(false),
            lock: AtomicBool::new(false),
            samples: Samples::default(),
            finished: AtomicBool::new(false),
            adjusted: AtomicI64::new(0),
            offset: AtomicI64::new(0),
            rounds: AtomicU32::new(0),
            apart: AtomicU32::new(0),
        }
    }

    /// Poison the exchange if a processor stops making progress.
    fn cancelled(&self, started: u64) -> bool {
        if self.aborted.load(Ordering::Acquire) {
            return true;
        }
        if ordered().wrapping_sub(started) < self.rendezvous_ticks {
            return false;
        }
        self.aborted.store(true, Ordering::Release);
        true
    }

    /// Wait until both cores reach rendezvous `point`, counted from one.
    fn meet(&self, point: &mut u32) -> bool {
        *point += 1;
        let started = ordered();
        self.arrivals.fetch_add(1, Ordering::AcqRel);
        while self.arrivals.load(Ordering::Acquire) < 2 * *point {
            if self.cancelled(started) {
                return false;
            }
            core::hint::spin_loop();
        }
        !self.aborted.load(Ordering::Acquire)
    }

    /// Read the counter, moved by `offset`, in turn with the other core for
    /// a round, noting the most the shared value went back as `side` saw it.
    fn measure(&self, side: Side, offset: i64) -> bool {
        let started = ordered();
        loop {
            // Give each processor the next sample after its peer, while
            // keeping a preempted peer from holding the round open.
            while !self.samples.next_is(side) {
                if ordered().wrapping_sub(started) >= self.round_ticks {
                    return !self.aborted.load(Ordering::Acquire);
                }
                core::hint::spin_loop();
            }
            while self
                .lock
                .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
                .is_err()
            {
                if self.cancelled(started) {
                    return false;
                }
                core::hint::spin_loop();
            }
            let now = ordered();
            self.samples.record(side, now.wrapping_add(offset as u64));
            self.lock.store(false, Ordering::Release);
            if now.wrapping_sub(started) >= self.round_ticks {
                return !self.aborted.load(Ordering::Acquire);
            }
        }
    }

    /// The BSP's part: every round the AP asks for.
    pub(super) fn source(&self) -> bool {
        let mut point = 0;
        loop {
            if !(self.meet(&mut point)
                && self.measure(Side::Bsp, 0)
                && self.meet(&mut point)
                && self.meet(&mut point))
            {
                return false;
            }
            if self.finished.load(Ordering::Acquire) {
                return true;
            }
        }
    }

    /// The AP's part, on the AP: rounds until its counter agrees with the
    /// BSP's, correcting it between them through `IA32_TSC_ADJUST` while
    /// that converges, else in software. Returns the software offset.
    pub(super) fn target(&self) -> Result<i64> {
        // CPUID.(EAX=7,ECX=0):EBX[1] enumerates IA32_TSC_ADJUST.
        let adjustable = __cpuid_count(0, 0).eax >= 7 && __cpuid_count(7, 0).ebx & 1 << 1 != 0;
        // SAFETY: enumerated above, and read only then.
        let base = adjustable.then(|| unsafe { TSC_ADJUST.read() });
        let (mut adjusted, mut offset) = (0i64, 0i64);
        let mut point = 0;
        let (mut rounds, mut apart) = (0, 0);
        loop {
            self.samples.reset();
            if !(self.meet(&mut point) && self.measure(Side::Ap, offset) && self.meet(&mut point)) {
                break;
            }
            let outcome = match self.samples.interleaved_warps() {
                Some((seen_here, seen_by_bsp)) => {
                    rounds += 1;
                    if seen_here == 0 && seen_by_bsp == 0 {
                        Some(Ok(offset))
                    } else if rounds == 2 * ROUNDS {
                        Some(Err(refuse!(
                            "AP TSC still differs from the BSP's after hardware and software correction"
                        )))
                    } else {
                        let step = correction(seen_here, seen_by_bsp);
                        match base {
                            Some(base) if rounds < ROUNDS => {
                                adjusted = adjusted.wrapping_add(step);
                                // SAFETY: enumerated; Linux writes the same register.
                                unsafe { TSC_ADJUST.write(base.wrapping_add(adjusted as u64)) };
                            }
                            _ => offset = offset.wrapping_add(step),
                        }
                        None
                    }
                }
                None => {
                    apart += 1;
                    (apart == APART_ROUNDS).then(|| {
                        Err(refuse!(
                            "AP and BSP TSC samples repeatedly failed to overlap"
                        ))
                    })
                }
            };
            if let Some(outcome) = outcome {
                self.adjusted.store(adjusted, Ordering::Relaxed);
                self.offset.store(offset, Ordering::Relaxed);
                self.rounds.store(rounds, Ordering::Relaxed);
                self.apart.store(apart, Ordering::Relaxed);
                self.finished.store(true, Ordering::Release);
                if self.meet(&mut point) {
                    return outcome;
                }
                break;
            }
            if !self.meet(&mut point) {
                break;
            }
        }
        Err(refuse!("AP TSC synchronization rendezvous timed out"))
    }

    /// How the AP's counter came to agree, once [`Self::source`] returned.
    pub(super) fn outcome(&self) -> Outcome {
        Outcome {
            adjusted: self.adjusted.load(Ordering::Relaxed),
            offset: self.offset.load(Ordering::Relaxed),
            rounds: self.rounds.load(Ordering::Relaxed),
            apart: self.apart.load(Ordering::Relaxed),
        }
    }
}
