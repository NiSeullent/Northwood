//! A crash-safe journal inside a preallocated file of fixed size.
//!
//! The file begins with two commit slots of [`SLOT_BYTES`]; the rest is the
//! data area. A [`Commit`] names the blobs that make up the file's content:
//! each an extent of the data area with a kind, its length and its SHA-256.
//! Commit `g` lives in slot `g % 2`, so a new commit never overwrites the
//! newest one, and its own checksum tells a whole slot from a torn one.
//!
//! An update ([`plan`]) writes its blobs only into space the newest commit
//! does not name, makes them durable, writes the next commit into the other
//! slot and makes that durable. A write cut anywhere leaves the newest
//! commit and every blob it names as they were, so a reader ([`candidates`],
//! [`read_entries`]) finds either the old content or the new. An update that
//! replaces content and finds no room beside it first commits without the
//! replaced blobs, then writes the replacement where they were: a cut
//! between the two leaves that intermediate content, which is also whole.
//!
//! The format knows nothing of what the blobs hold: each user names its own
//! domain and kinds, and a key that a reader of another key ignores, so a
//! store of pipelines and a store of translations can share it.
use crate::payload::pieces::Extent;
use alloc::vec::Vec;
use stela_crypto::{Sha256, digest::Digest};

/// Bytes of each commit slot.
pub const SLOT_BYTES: u64 = 64 << 10;
/// Where the data area begins: after both slots.
pub const DATA_START: u64 = 2 * SLOT_BYTES;
/// The alignment of every blob, a multiple of any disk block.
pub const ALIGN: u64 = 4096;

const MAGIC: [u8; 8] = *b"STLJRNL1";
const VERSION: u32 = 1;
/// Magic, version, entry count, domain, generation, key.
const HEAD_BYTES: usize = 64;
/// Kind, a zero word, offset, length, SHA-256.
const ENTRY_BYTES: usize = 56;
const CHECKSUM_AT: usize = SLOT_BYTES as usize - 32;
/// The most blobs one commit names.
pub const MAX_ENTRIES: usize = (CHECKSUM_AT - HEAD_BYTES) / ENTRY_BYTES;

/// A blob's kind and bytes.
pub type Blob = (u32, Vec<u8>);

/// Where [`layout`] places blobs: by index and offset in the first commit
/// and in the second, and the indices of those placed in neither.
type Layout = (Vec<(usize, u64)>, Vec<(usize, u64)>, Vec<usize>);

/// A blob a commit names.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Entry {
    pub kind: u32,
    /// From the start of the file, a multiple of [`ALIGN`] in the data area.
    pub offset: u64,
    pub length: u64,
    pub digest: [u8; 32],
}

impl Entry {
    /// The aligned range the blob occupies.
    fn span(&self) -> (u64, u64) {
        (self.offset, self.offset + aligned(self.length))
    }
}

/// A file's content as one commit names it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Commit {
    /// What kind of store the file is.
    pub domain: [u8; 8],
    /// Higher is newer; the slot is `generation % 2`.
    pub generation: u64,
    /// What the content is valid for, the user's own.
    pub key: [u8; 32],
    pub entries: Vec<Entry>,
}

fn aligned(length: u64) -> u64 {
    length.div_ceil(ALIGN) * ALIGN
}

fn word(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().expect("four bytes"))
}

fn double(bytes: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(bytes[at..at + 8].try_into().expect("eight bytes"))
}

impl Commit {
    /// The slot this commit lives in.
    pub fn slot(&self) -> u64 {
        self.generation % 2
    }

    /// Its offset in the file.
    pub fn slot_offset(&self) -> u64 {
        self.slot() * SLOT_BYTES
    }

    /// The slot's bytes. A commit names at most [`MAX_ENTRIES`] blobs.
    pub fn encode(&self) -> Vec<u8> {
        assert!(
            self.entries.len() <= MAX_ENTRIES,
            "a commit names too many blobs"
        );
        let mut slot = alloc::vec![0; SLOT_BYTES as usize];
        slot[..8].copy_from_slice(&MAGIC);
        slot[8..12].copy_from_slice(&VERSION.to_le_bytes());
        slot[12..16].copy_from_slice(&(self.entries.len() as u32).to_le_bytes());
        slot[16..24].copy_from_slice(&self.domain);
        slot[24..32].copy_from_slice(&self.generation.to_le_bytes());
        slot[32..64].copy_from_slice(&self.key);
        for (index, entry) in self.entries.iter().enumerate() {
            let at = HEAD_BYTES + index * ENTRY_BYTES;
            slot[at..at + 4].copy_from_slice(&entry.kind.to_le_bytes());
            slot[at + 8..at + 16].copy_from_slice(&entry.offset.to_le_bytes());
            slot[at + 16..at + 24].copy_from_slice(&entry.length.to_le_bytes());
            slot[at + 24..at + 56].copy_from_slice(&entry.digest);
        }
        let checksum = Sha256::digest(&slot[..CHECKSUM_AT]);
        slot[CHECKSUM_AT..].copy_from_slice(&checksum);
        slot
    }

    /// The commit slot `index` of a file of `file_bytes` holds for
    /// `domain`: `None` for a blank, torn or foreign slot, one in the wrong
    /// slot, or one whose blobs leave the data area or overlap.
    pub fn decode(slot: &[u8], index: u64, domain: [u8; 8], file_bytes: u64) -> Option<Self> {
        if slot.len() != SLOT_BYTES as usize
            || slot[..8] != MAGIC
            || word(slot, 8) != VERSION
            || slot[16..24] != domain
            || Sha256::digest(&slot[..CHECKSUM_AT])[..] != slot[CHECKSUM_AT..]
        {
            return None;
        }
        let count = word(slot, 12) as usize;
        let generation = double(slot, 24);
        if count > MAX_ENTRIES || generation % 2 != index {
            return None;
        }
        let entries: Vec<Entry> = (0..count)
            .map(|index| {
                let at = HEAD_BYTES + index * ENTRY_BYTES;
                Entry {
                    kind: word(slot, at),
                    offset: double(slot, at + 8),
                    length: double(slot, at + 16),
                    digest: slot[at + 24..at + 56].try_into().expect("a digest"),
                }
            })
            .collect();
        let end = data_end(file_bytes);
        let mut spans: Vec<(u64, u64)> = Vec::with_capacity(entries.len());
        for entry in &entries {
            if entry.offset < DATA_START
                || !entry.offset.is_multiple_of(ALIGN)
                || entry.length == 0
                || entry.offset.checked_add(aligned(entry.length))? > end
            {
                return None;
            }
            spans.push(entry.span());
        }
        spans.sort_unstable();
        if spans.windows(2).any(|pair| pair[0].1 > pair[1].0) {
            return None;
        }
        Some(Self {
            domain,
            generation,
            key: slot[32..64].try_into().expect("a key"),
            entries,
        })
    }
}

/// The end of the data area of a file of `file_bytes`.
fn data_end(file_bytes: u64) -> u64 {
    file_bytes / ALIGN * ALIGN
}

/// The whole commits of `slots`, the file's first [`DATA_START`] bytes,
/// newest first: a reader takes the first whose blobs all read back whole
/// ([`read_entries`]).
pub fn candidates(slots: &[u8], domain: [u8; 8], file_bytes: u64) -> Vec<Commit> {
    let mut found: Vec<Commit> = (0..2u64)
        .filter_map(|index| {
            let start = (index * SLOT_BYTES) as usize;
            let slot = slots.get(start..start + SLOT_BYTES as usize)?;
            Commit::decode(slot, index, domain, file_bytes)
        })
        .collect();
    found.sort_unstable_by_key(|commit| core::cmp::Reverse(commit.generation));
    found
}

/// The blobs `commit` names, read with `read_at` (an offset in the file
/// and the bytes to fill), each checked against its digest: `None` when one
/// does not read back whole.
pub fn read_entries(
    commit: &Commit,
    mut read_at: impl FnMut(u64, &mut [u8]) -> Option<()>,
) -> Option<Vec<Blob>> {
    commit
        .entries
        .iter()
        .map(|entry| {
            let mut bytes = alloc::vec![0; usize::try_from(entry.length).ok()?];
            read_at(entry.offset, &mut bytes)?;
            (Sha256::digest(&bytes)[..] == entry.digest).then_some((entry.kind, bytes))
        })
        .collect()
}

/// New content for a file: blobs to add, and the kinds whose committed blobs
/// those of the same kind replace.
#[derive(Clone, Debug, Default)]
pub struct Change {
    pub replace: Vec<u32>,
    pub blobs: Vec<Blob>,
}

/// One write of an update, in order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Step {
    /// Blob `blob` of the plan at `offset`, padded with zeros to [`ALIGN`].
    Data { blob: usize, offset: u64 },
    /// Every write before it durable.
    Flush,
    /// A commit into its slot.
    Commit(Commit),
}

/// An update's writes and what it leaves.
#[derive(Debug)]
pub struct Plan {
    pub blobs: Vec<Blob>,
    pub steps: Vec<Step>,
    /// The blobs no room was found for, by index: their content is not kept,
    /// and a replaced kind with such a blob keeps its committed blobs.
    pub dropped: Vec<usize>,
    /// The commit the steps end with, the current one when they are none.
    pub last: Option<Commit>,
}

/// First fit for `length` bytes among the data area of `end` bytes less
/// `taken`, kept sorted.
fn place(taken: &mut Vec<(u64, u64)>, end: u64, length: u64) -> Option<u64> {
    let size = aligned(length);
    let mut at = DATA_START;
    let mut index = 0;
    for &(start, stop) in taken.iter() {
        if start >= at + size {
            break;
        }
        at = at.max(stop);
        index += 1;
    }
    if at + size > end {
        return None;
    }
    taken.insert(index, (at, at + size));
    Some(at)
}

fn entry(kind: u32, offset: u64, bytes: &[u8]) -> Entry {
    Entry {
        kind,
        offset,
        length: bytes.len() as u64,
        digest: Sha256::digest(bytes).into(),
    }
}

/// Where each blob goes when `current` is the newest commit and `change`
/// replaces `replace`: blobs placed beside every committed blob first, then
/// the rest where the replaced ones were, after a commit without them.
/// Answers the first commit's new entries, the second's, and the blobs that
/// fit in neither.
fn layout(current: &[Entry], replace: &[u32], blobs: &[Blob], end: u64, room: usize) -> Layout {
    let mut taken: Vec<(u64, u64)> = current.iter().map(Entry::span).collect();
    taken.sort_unstable();
    let kept = current
        .iter()
        .filter(|entry| !replace.contains(&entry.kind))
        .count();
    let mut first = Vec::new();
    let mut left = Vec::new();
    for (index, (_, bytes)) in blobs.iter().enumerate() {
        if bytes.is_empty() {
            continue;
        }
        match (kept + first.len() < room)
            .then(|| place(&mut taken, end, bytes.len() as u64))
            .flatten()
        {
            Some(offset) => first.push((index, offset)),
            None => left.push(index),
        }
    }
    if left.is_empty() {
        return (first, Vec::new(), Vec::new());
    }
    // Freed by the first commit: the replaced blobs.
    let mut taken: Vec<(u64, u64)> = current
        .iter()
        .filter(|entry| !replace.contains(&entry.kind))
        .map(Entry::span)
        .chain(
            first
                .iter()
                .map(|&(index, offset)| (offset, offset + aligned(blobs[index].1.len() as u64))),
        )
        .collect();
    taken.sort_unstable();
    let mut second = Vec::new();
    let mut dropped = Vec::new();
    for index in left {
        match (kept + first.len() + second.len() < room)
            .then(|| place(&mut taken, end, blobs[index].1.len() as u64))
            .flatten()
        {
            Some(offset) => second.push((index, offset)),
            None => dropped.push(index),
        }
    }
    (first, second, dropped)
}

/// The writes that take a file of `file_bytes`, whose newest commit is
/// `current`, to `change` for `domain` and `key`. Committed blobs of another
/// domain or key are not kept, and their space is reused. A replaced kind
/// whose new blobs do not all fit keeps its committed blobs, and those new
/// blobs are dropped.
pub fn plan(
    current: Option<&Commit>,
    domain: [u8; 8],
    key: [u8; 32],
    file_bytes: u64,
    change: Change,
) -> Plan {
    let Change { mut replace, blobs } = change;
    let end = data_end(file_bytes);
    let generation = current.map_or(0, |commit| commit.generation);
    let committed: &[Entry] = match current {
        Some(commit) if commit.domain == domain && commit.key == key => &commit.entries,
        _ => &[],
    };
    // A replacement that cannot fit whole keeps what it would replace.
    let mut withdrawn: Vec<usize> = Vec::new();
    let (first, second, dropped) = loop {
        let candidates: Vec<Blob> = blobs
            .iter()
            .enumerate()
            .map(|(index, (kind, bytes))| {
                let bytes = if withdrawn.contains(&index) {
                    Vec::new()
                } else {
                    bytes.clone()
                };
                (*kind, bytes)
            })
            .collect();
        let (first, second, dropped) = layout(committed, &replace, &candidates, end, MAX_ENTRIES);
        let short: Vec<u32> = replace
            .iter()
            .copied()
            .filter(|kind| dropped.iter().any(|&index| blobs[index].0 == *kind))
            .collect();
        if short.is_empty() {
            break (first, second, dropped);
        }
        replace.retain(|kind| !short.contains(kind));
        withdrawn.extend(
            blobs
                .iter()
                .enumerate()
                .filter(|(_, (kind, _))| short.contains(kind))
                .map(|(index, _)| index),
        );
    };
    let mut dropped: Vec<usize> = dropped.into_iter().chain(withdrawn).collect();
    dropped.sort_unstable();
    dropped.dedup();
    let kept: Vec<Entry> = committed
        .iter()
        .filter(|entry| !replace.contains(&entry.kind))
        .copied()
        .collect();
    let mut steps = Vec::new();
    let mut last = current.cloned();
    if first.is_empty() && second.is_empty() && kept.len() == committed.len() {
        return Plan {
            blobs,
            steps,
            dropped,
            last,
        };
    }
    let mut entries = kept;
    let mut commit = |placed: &[(usize, u64)], entries: &mut Vec<Entry>, generation: u64| {
        for &(index, offset) in placed {
            steps.push(Step::Data {
                blob: index,
                offset,
            });
            entries.push(entry(blobs[index].0, offset, &blobs[index].1));
        }
        if !placed.is_empty() {
            steps.push(Step::Flush);
        }
        let next = Commit {
            domain,
            generation,
            key,
            entries: entries.clone(),
        };
        steps.push(Step::Commit(next.clone()));
        steps.push(Step::Flush);
        next
    };
    let next = commit(&first, &mut entries, generation + 1);
    last = Some(next);
    if !second.is_empty() {
        last = Some(commit(&second, &mut entries, generation + 2));
    }
    Plan {
        blobs,
        steps,
        dropped,
        last,
    }
}

/// Where an update's writes go: the file, as offsets from its start.
pub trait Target {
    type Error;
    /// Write `bytes` at `offset`, both multiples of [`ALIGN`] apart from a
    /// commit's whole slot. Answers `Ok(false)` when the write was queued
    /// and has not completed: the same call is made again later.
    fn write_at(&mut self, offset: u64, bytes: &[u8]) -> Result<bool, Self::Error>;
    /// Make every completed write durable; `Ok(false)` as for a write.
    fn flush(&mut self) -> Result<bool, Self::Error>;
}

/// A plan being carried out a piece at a time, so no call writes more than
/// its budget.
#[derive(Debug)]
pub struct Execution {
    plan: Plan,
    step: usize,
    /// Bytes of the current data step done.
    done: u64,
    /// The piece written last, while its write is queued.
    pending: Option<Vec<u8>>,
}

impl Execution {
    pub fn new(plan: Plan) -> Self {
        Self {
            plan,
            step: 0,
            done: 0,
            pending: None,
        }
    }

    pub fn plan(&self) -> &Plan {
        &self.plan
    }

    /// Whether every step is done.
    pub fn finished(&self) -> bool {
        self.step == self.plan.steps.len()
    }

    /// Carries out the next pieces of at most `budget` bytes in all (at
    /// least one piece), stopping at a write the target queued. Answers
    /// whether every step is done.
    pub fn advance<T: Target>(&mut self, target: &mut T, budget: u64) -> Result<bool, T::Error> {
        let budget = aligned(budget.max(1));
        let mut spent = 0;
        while let Some(step) = self.plan.steps.get(self.step) {
            if spent != 0 && spent >= budget {
                break;
            }
            match step {
                Step::Data { blob, offset } => {
                    let bytes = &self.plan.blobs[*blob].1;
                    let total = aligned(bytes.len() as u64);
                    let piece = match self.pending.take() {
                        Some(piece) => piece,
                        None => {
                            let take = (total - self.done).min(budget);
                            let from = self.done as usize;
                            let to = (self.done + take).min(bytes.len() as u64) as usize;
                            let mut piece = bytes.get(from..to).unwrap_or(&[]).to_vec();
                            piece.resize(take as usize, 0);
                            piece
                        }
                    };
                    if !target.write_at(offset + self.done, &piece)? {
                        self.pending = Some(piece);
                        return Ok(false);
                    }
                    spent += piece.len() as u64;
                    self.done += piece.len() as u64;
                    if self.done == total {
                        self.done = 0;
                        self.step += 1;
                    }
                }
                Step::Flush => {
                    if !target.flush()? {
                        return Ok(false);
                    }
                    self.step += 1;
                }
                Step::Commit(commit) => {
                    let slot = match self.pending.take() {
                        Some(slot) => slot,
                        None => commit.encode(),
                    };
                    if !target.write_at(commit.slot_offset(), &slot)? {
                        self.pending = Some(slot);
                        return Ok(false);
                    }
                    spent += SLOT_BYTES;
                    self.step += 1;
                }
            }
        }
        Ok(self.finished())
    }
}

/// The device writes of file range `offset..offset + length` of a file at
/// `extents` on a disk of `block`-byte blocks, in order: each its first
/// block, its bytes and where they begin in the range. `None` when the range
/// passes the file or a piece is not whole blocks.
pub fn device_ranges(
    extents: &[Extent],
    block: u64,
    offset: u64,
    length: u64,
) -> Option<Vec<(u64, u64, usize)>> {
    let mut ranges = Vec::new();
    let mut file_at = 0u64;
    let (mut want, end) = (offset, offset.checked_add(length)?);
    for extent in extents {
        let extent_end = file_at + extent.bytes;
        if want < extent_end && want < end {
            let take = extent_end.min(end) - want;
            let device = extent.offset + (want - file_at);
            if block == 0 || !device.is_multiple_of(block) || !take.is_multiple_of(block) {
                return None;
            }
            ranges.push((device / block, take, (want - offset) as usize));
            want += take;
        }
        file_at = extent_end;
    }
    (want == end).then_some(ranges)
}

#[cfg(test)]
mod tests;
