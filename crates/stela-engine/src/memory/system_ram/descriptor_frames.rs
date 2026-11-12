// SPDX-License-Identifier: GPL-2.0-or-later
//! Canonical admission for reads, writes, and wired DMA on 16 KiB frames.

use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

pub(super) const FRAME_BYTES: usize = 16 * 1024;
const STRIPES: usize = 64;
const CLOSED: usize = usize::MAX;

/// A stripe collision can delay closing an idle frame. It can never hide a
/// live access. The bounded table needs no allocation on the fault path.
pub(super) struct DescriptorFrames {
    stripes: [AtomicUsize; STRIPES],
    wired: Vec<AtomicU64>,
    nonresident: Vec<AtomicU64>,
    nonresident_count: AtomicUsize,
}

impl DescriptorFrames {
    pub(super) fn new(frame_count: usize) -> Result<Self, super::RegistrationError> {
        let words = frame_count.div_ceil(64);
        let bitmap = || -> Result<Vec<AtomicU64>, super::RegistrationError> {
            let mut bits = Vec::new();
            bits.try_reserve_exact(words)
                .map_err(|_| super::RegistrationError::FrameTracking)?;
            bits.resize_with(words, AtomicU64::default);
            Ok(bits)
        };
        Ok(Self {
            stripes: [const { AtomicUsize::new(0) }; STRIPES],
            wired: bitmap()?,
            nonresident: bitmap()?,
            nonresident_count: AtomicUsize::new(0),
        })
    }

    fn stripe(&self, physical: u64) -> &AtomicUsize {
        &self.stripes[((physical / FRAME_BYTES as u64) % STRIPES as u64) as usize]
    }

    pub(super) fn try_pin(&self, physical: u64) -> Option<DescriptorPin<'_>> {
        let stripe = self.stripe(physical);
        try_pin(stripe).then(|| DescriptorPin(stripe))
    }

    /// Acquire every distinct stripe covered by a checked nonempty range.
    /// If any stripe is closed, the partially acquired guard rolls back.
    pub(super) fn try_pin_range(&self, start: u64, last: u64) -> Option<RangePin<'_>> {
        self.try_pin_stripes(Self::range_stripes(start, last))
    }

    pub(super) fn range_stripes(start: u64, last: u64) -> u64 {
        let first_frame = start / FRAME_BYTES as u64;
        let count = last / FRAME_BYTES as u64 - first_frame + 1;
        if count >= STRIPES as u64 {
            u64::MAX
        } else {
            ((1u64 << count) - 1).rotate_left((first_frame % STRIPES as u64) as u32)
        }
    }

    pub(super) fn try_pin_stripes(&self, mut pending: u64) -> Option<RangePin<'_>> {
        let mut pin = RangePin {
            frames: self,
            stripes: 0,
        };
        while pending != 0 {
            let stripe = pending.trailing_zeros() as usize;
            if !try_pin(&self.stripes[stripe]) {
                return None;
            }
            pin.stripes |= 1 << stripe;
            pending &= pending - 1;
        }
        Some(pin)
    }

    /// Commit while holding pins for all the corresponding source stripes.
    pub(super) fn wire(&self, mut first: usize, last: usize) {
        while first <= last {
            let bit = first % 64;
            let count = (last - first + 1).min(64 - bit);
            let mask = (u64::MAX >> (64 - count)) << bit;
            self.wired[first / 64].fetch_or(mask, Ordering::Relaxed);
            first += count;
        }
    }

    /// Observe after closing the source stripe. Its acquire pairs with the
    /// temporary pin's release, including the permanent bits it recorded.
    pub(super) fn is_wired(&self, frame: usize) -> bool {
        self.wired[frame / 64].load(Ordering::Acquire) & (1 << (frame % 64)) != 0
    }

    pub(super) fn has_nonresident(&self) -> bool {
        self.nonresident_count.load(Ordering::Acquire) != 0
    }

    pub(super) fn is_nonresident(&self, frame: usize) -> bool {
        self.nonresident[frame / 64].load(Ordering::Acquire) & (1 << (frame % 64)) != 0
    }

    /// Change an exact frame's state while its stripe is closed. The count
    /// is nonzero throughout the interval in which a bit is observable.
    pub(super) fn mark_nonresident(&self, frame: usize) {
        debug_assert!(!self.is_nonresident(frame));
        self.nonresident_count.fetch_add(1, Ordering::Release);
        self.nonresident[frame / 64].fetch_or(1 << (frame % 64), Ordering::Release);
    }

    pub(super) fn mark_resident(&self, frame: usize) {
        debug_assert!(self.is_nonresident(frame));
        self.nonresident[frame / 64].fetch_and(!(1 << (frame % 64)), Ordering::Release);
        self.nonresident_count.fetch_sub(1, Ordering::Release);
    }

    pub(super) fn try_close(&self, physical: u64) -> Option<ClosedStripe<'_>> {
        let stripe = self.stripe(physical);
        stripe
            .compare_exchange(0, CLOSED, Ordering::Acquire, Ordering::Relaxed)
            .ok()?;
        Some(ClosedStripe(stripe))
    }
}

fn try_pin(stripe: &AtomicUsize) -> bool {
    let mut held = stripe.load(Ordering::Relaxed);
    while held < CLOSED - 1 {
        match stripe.compare_exchange_weak(held, held + 1, Ordering::AcqRel, Ordering::Relaxed) {
            Ok(_) => return true,
            Err(actual) => held = actual,
        }
    }
    false
}

pub(super) struct DescriptorPin<'a>(&'a AtomicUsize);

impl Drop for DescriptorPin<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

pub(super) struct RangePin<'a> {
    frames: &'a DescriptorFrames,
    stripes: u64,
}

impl RangePin<'_> {
    fn acquired(&self) -> impl Iterator<Item = &AtomicUsize> {
        let mut stripes = self.stripes;
        core::iter::from_fn(move || {
            if stripes == 0 {
                return None;
            }
            let stripe = stripes.trailing_zeros() as usize;
            stripes &= stripes - 1;
            Some(&self.frames.stripes[stripe])
        })
    }
}

impl Drop for RangePin<'_> {
    fn drop(&mut self) {
        for stripe in self.acquired() {
            stripe.fetch_sub(1, Ordering::AcqRel);
        }
    }
}

pub(super) struct ClosedStripe<'a>(&'a AtomicUsize);

impl Drop for ClosedStripe<'_> {
    fn drop(&mut self) {
        self.0.store(0, Ordering::Release);
    }
}
