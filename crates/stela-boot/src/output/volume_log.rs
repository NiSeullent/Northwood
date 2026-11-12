// SPDX-License-Identifier: GPL-2.0-or-later

//! Block-aligned writes to the engine section of a preallocated text slot.
use alloc::{vec, vec::Vec};
use core::ops::Range;
use stela_drivers::block::{BlockError, Finished, Request};
use stela_runtime::pc::boot_log::{PENDING_BYTES, Pending};

pub(crate) struct Writer {
    bytes: Vec<u8>,
    block: usize,
    first_lba: u64,
    record_start: u64,
    dirty: Option<Range<usize>>,
    storing: bool,
}

impl Writer {
    pub(crate) fn new(first_lba: u64, block: usize, capacity: usize, record_start: u64) -> Self {
        Self {
            bytes: vec![b' '; capacity],
            block,
            first_lba,
            record_start,
            dirty: None,
            storing: false,
        }
    }

    /// Append incrementally while space is available. Once full, keep the
    /// newest complete lines in the engine section. The loader's section
    /// and the ESP's circular binary log are kept separately.
    pub(super) fn stage(&mut self, record: &Pending<[u8; PENDING_BYTES]>, from: u64) -> u64 {
        let end = record.pushed();
        if end <= from {
            return from;
        }
        let capacity = self.bytes.len();
        let length = end - self.record_start;
        let (start, through) = if length <= capacity as u64 {
            let start = (from - self.record_start) as usize;
            let through = length as usize;
            copy(record, from, &mut self.bytes[start..through]);
            (start, through)
        } else {
            let notice = b"[earlier engine output omitted]\n";
            let from = end - (capacity - notice.len()) as u64;
            let tail = &mut self.bytes[notice.len()..];
            copy(record, from, tail);
            let len = tail.len();
            let skip = tail
                .iter()
                .position(|&byte| byte == b'\n')
                .map_or(len, |index| index + 1);
            tail.copy_within(skip.., 0);
            tail[len - skip..].fill(b' ');
            self.bytes[..notice.len()].copy_from_slice(notice);
            (0, capacity)
        };
        self.dirty = Some(start / self.block * self.block..through.next_multiple_of(self.block));
        end
    }

    pub(super) fn storing(&self) -> bool {
        self.storing
    }

    pub(super) fn begin_store(&mut self) -> bool {
        self.storing = self.dirty.is_some();
        self.storing
    }

    pub(super) fn next_request(&mut self) -> Option<Request> {
        let range = self.dirty.take()?;
        Some(Request::Write {
            lba: self.first_lba + (range.start / self.block) as u64,
            data: self.bytes[range].to_vec(),
            durable: true,
        })
    }

    pub(super) fn completed(&mut self, result: Finished) -> Result<(), BlockError> {
        self.storing = false;
        result.map(|_| ())
    }
}

fn copy(record: &Pending<[u8; PENDING_BYTES]>, from: u64, mut out: &mut [u8]) {
    let _ = record.copy_since(from, |bytes| {
        let (head, tail) = core::mem::take(&mut out)
            .split_at_mut_checked(bytes.len())
            .ok_or(())?;
        head.copy_from_slice(bytes);
        out = tail;
        Ok::<(), ()>(())
    });
}
