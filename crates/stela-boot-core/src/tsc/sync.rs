// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2006, Red Hat, Inc., Ingo Molnar
//! Arithmetic of each AP's TSC synchronization with the BSP, after Linux's
//! `arch/x86/kernel/tsc_sync.c`: how long a round reads, which way a warp
//! moves the AP's counter, and the counter base an AP corrected in software
//! counts from.

use core::sync::atomic::{AtomicU8, AtomicU64, Ordering};

/// The sampling processors in a synchronization round.
#[derive(Clone, Copy)]
pub enum Side {
    Bsp,
    Ap,
}

impl Side {
    fn bit(self) -> u8 {
        1 << self as u8
    }
}

/// Shared counter samples and the largest backwards warp seen by each side.
pub struct Samples {
    previous: AtomicU64,
    previous_side: AtomicU8,
    crossed: AtomicU8,
    warps: [AtomicU64; 2],
}

impl Default for Samples {
    fn default() -> Self {
        Self {
            previous: AtomicU64::new(0),
            previous_side: AtomicU8::new(0),
            crossed: AtomicU8::new(0),
            warps: [AtomicU64::new(0), AtomicU64::new(0)],
        }
    }
}

impl Samples {
    /// Give the peer the next sample, so a processor cannot immediately
    /// reacquire the sampling lock until its peer misses the entire round.
    pub fn next_is(&self, side: Side) -> bool {
        self.previous_side.load(Ordering::Relaxed) != side.bit()
    }

    /// Forget the previous round before either processor starts sampling.
    pub fn reset(&self) {
        self.previous_side.store(0, Ordering::Relaxed);
        self.crossed.store(0, Ordering::Relaxed);
        for warp in &self.warps {
            warp.store(0, Ordering::Relaxed);
        }
    }

    /// Record a counter read. Callers serialize their reads and submissions
    /// with the same lock, so each sample sees the last read in real order.
    pub fn record(&self, side: Side, now: u64) {
        let bit = side.bit();
        let previous_side = self.previous_side.load(Ordering::Relaxed);
        let previous = self.previous.load(Ordering::Relaxed);
        self.previous_side.store(bit, Ordering::Relaxed);
        self.previous.store(now, Ordering::Relaxed);
        if previous_side != 0 && previous_side != bit {
            let crossed = self.crossed.load(Ordering::Relaxed);
            self.crossed.store(crossed | bit, Ordering::Relaxed);
            // Signed subtraction also covers a counter or correction crossing zero.
            let forward = now.wrapping_sub(previous) as i64;
            if forward < 0 {
                self.warps[side as usize].fetch_max(forward.unsigned_abs(), Ordering::Relaxed);
            }
        }
    }

    /// The AP and BSP warps, once each has read after the other. A quiet
    /// round in which one processor only sampled after its peer stopped
    /// provides no proof of agreement.
    pub fn interleaved_warps(&self) -> Option<(u64, u64)> {
        (self.crossed.load(Ordering::Relaxed) == 0b11).then(|| {
            (
                self.warps[Side::Ap as usize].load(Ordering::Relaxed),
                self.warps[Side::Bsp as usize].load(Ordering::Relaxed),
            )
        })
    }
}

/// Linux's `loop_timeout`: 2 ms when a CPU of the AP's package already
/// synchronized, whose counters the package's own clock already keeps
/// together, else 20 ms for a new package or module.
pub fn round_millis(package: u32, synchronized: impl IntoIterator<Item = u32>) -> u64 {
    if synchronized.into_iter().any(|other| other == package) {
        2
    } else {
        20
    }
}

/// What to add to the AP's counter after a round with warps: the most the
/// shared value went back as the AP saw it (the AP ran behind), else as the
/// BSP saw it, negated (the AP ran ahead).
pub fn correction(seen_here: u64, seen_by_bsp: u64) -> i64 {
    let clamp = |warp: u64| warp.min(i64::MAX as u64) as i64;
    if seen_here != 0 {
        clamp(seen_here)
    } else {
        -clamp(seen_by_bsp)
    }
}

/// The base an AP whose counter reads `offset` ticks from the BSP's, and
/// which corrects that in software, counts its 24 MHz counter from, so that
/// it reads the same count as the BSP at the same instant: a raw value
/// `offset` behind the BSP's pairs with a base `offset` behind `tsc_base`.
pub fn corrected_base(tsc_base: u64, offset: i64) -> u64 {
    tsc_base.wrapping_sub(offset as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_delayed_ap_does_not_make_a_quiet_round_agree() {
        let samples = Samples::default();
        samples.record(Side::Bsp, 100);
        samples.record(Side::Bsp, 200);
        samples.record(Side::Ap, 4_592_093_650);
        samples.record(Side::Ap, 4_592_093_750);
        assert_eq!(samples.interleaved_warps(), None);

        samples.reset();
        samples.record(Side::Bsp, 100);
        samples.record(Side::Ap, 4_592_093_650);
        samples.record(Side::Bsp, 300);
        let (ap, bsp) = samples.interleaved_warps().unwrap();
        assert_eq!(correction(ap, bsp), -4_592_093_350);
    }

    #[test]
    fn an_interleaved_round_can_agree_across_a_counter_wrap() {
        let samples = Samples::default();
        samples.record(Side::Ap, u64::MAX);
        samples.record(Side::Bsp, 0);
        samples.record(Side::Ap, 1);
        assert_eq!(samples.interleaved_warps(), Some((0, 0)));
    }

    #[test]
    fn a_core_of_a_synchronized_package_reads_for_2_ms() {
        assert_eq!(round_millis(0, [0]), 2);
        assert_eq!(round_millis(1, [0, 0, 1]), 2);
        assert_eq!(round_millis(1, [0]), 20);
        assert_eq!(round_millis(0, []), 20);
    }

    #[test]
    fn warps_move_the_ap_toward_the_bsp() {
        assert_eq!(correction(500, 0), 500);
        assert_eq!(correction(0, 700), -700);
        assert_eq!(correction(500, 700), 500);
        assert_eq!(correction(0, 0), 0);
        assert_eq!(correction(u64::MAX, 0), i64::MAX);
    }

    #[test]
    fn a_software_correction_gives_the_bsps_count() {
        let base = 1_000_000;
        for (bsp_now, offset) in [(5_000_000u64, 12_345i64), (5_000_000, -9_876), (base, 0)] {
            // The AP's raw counter reads `offset` behind the BSP's.
            let ap_raw = bsp_now.wrapping_sub(offset as u64);
            let ap_base = corrected_base(base, offset);
            assert_eq!(ap_raw.wrapping_sub(ap_base), bsp_now - base);
        }
    }
}
