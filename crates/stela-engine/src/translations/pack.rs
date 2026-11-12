//! The pack container `stela.efi` keeps its translations in, and the store a
//! boot reads from it and plans the next pack with.
//!
//! A pack is one file per translator id: a header, an index sorted by page
//! key with the SHA-256 of the header and index, and the page records
//! (each checked by its own trailer when it is first used). Its length
//! follows from the index, so a file allocated larger than its pack (to be
//! rewritten in place) holds bytes past it that are never read.
//!
//! The store hands the engine the pack's records and keeps the blocks the
//! engine compiled anew. A store shared by machines that run at once (one
//! per PE) also keeps those blocks' native code, within a bound, for the
//! other machines to install instead of compiling them again. At a quiescent point it proves them (each compiled
//! again at its page's probe addresses) and lays out the next pack within a
//! bound: the records that gained entries, then the records this boot used,
//! then the others, as long as they fit.
mod codec;

use super::{Digest, FormatError, Record};
use crate::machine::engine::SharedTranslation;
use crate::machine::{Offer, Preparation, Translations};
use alloc::{
    collections::{BTreeMap, VecDeque},
    sync::{Arc, Weak},
    vec::Vec,
};
pub use codec::{DecodeError, DecodeScratch};
use core::{mem::size_of, ops::Range};
use stela_crypto::Sha256;
use stela_crypto::digest::Digest as _;

pub const MAGIC: [u8; 8] = *b"STLTPACK";
pub const SCHEMA: u32 = 1;
const MIXED_SCHEMA: u32 = 2;
/// Magic, schema, record count and translator id.
pub const HEADER_BYTES: usize = MAGIC.len() + 4 + 4 + DIGEST_BYTES;
const DIGEST_BYTES: usize = 32;
/// Page key, offset and length.
const INDEX_ENTRY_BYTES: usize = DIGEST_BYTES + 8 + 8;
const CHECKSUM_BYTES: usize = DIGEST_BYTES;
/// The most records a pack holds.
pub const MAX_RECORDS: usize = 1 << 22;
/// The most memory the blocks waiting for proof take by default; later
/// blocks are counted as dropped ([`Store::set_bounds`]).
const MAX_PENDING_BYTES: usize = 16 << 20;
/// The default bound of the decoded-record cache, least recently used first
/// out ([`Store::set_bounds`]). One record takes at most an eighth of it.
const DECODED_BYTES: usize = 8 << 20;
/// The default bound of the entries proven during the run
/// ([`Store::prove_batch`]).
const PROVEN_BYTES: usize = 64 << 20;
/// What a page waiting for proof costs beyond its buffers: an entry in a
/// B-tree node about half full.
const PENDING_PAGE_BYTES: usize = 2 * size_of::<(PendingKey, PendingPage)>();
/// The direct-mapped tables that name why a shared lookup missed: offers
/// being compiled, offers evicted, and the last offer of each PC and page.
const MISS_SLOTS: usize = 4096;
/// A full relayout leaves room for this many more index entries (at least
/// [`MIN_INDEX_SLACK`]) before the records, so later boots that add records
/// rewrite only the index and the new records.
const INDEX_SLACK_DIVISOR: usize = 2;
const MIN_INDEX_SLACK: usize = 1024;
/// Records start on this boundary after a full relayout.
const RECORD_ALIGNMENT: u64 = 4096;
/// A pack whose unused bytes between records pass this share of its length
/// is laid out again from the start.
const MAX_WASTE_PERCENT: u64 = 25;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackError {
    Truncated,
    Magic,
    Schema,
    Translator,
    Count,
    Checksum,
    /// An index entry is out of order or names bytes outside the pack.
    Index,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
enum PackFormat {
    Raw = SCHEMA,
    Mixed = MIXED_SCHEMA,
}

impl PackFormat {
    fn read(header: &[u8]) -> Result<Self, PackError> {
        match u32::from_le_bytes(
            header[MAGIC.len()..][..4]
                .try_into()
                .expect("a schema word"),
        ) {
            SCHEMA => Ok(Self::Raw),
            MIXED_SCHEMA => Ok(Self::Mixed),
            _ => Err(PackError::Schema),
        }
    }
}

/// The bytes of the header, index and index checksum of the pack whose
/// header is `header`: what to read before its records.
pub fn prefix_bytes(header: &[u8], translator: &Digest) -> Result<usize, PackError> {
    read_header(header, translator).map(|(_, count)| prefix_length(count))
}

fn read_header(header: &[u8], translator: &Digest) -> Result<(PackFormat, usize), PackError> {
    if header.len() < HEADER_BYTES {
        return Err(PackError::Truncated);
    }
    if header[..MAGIC.len()] != MAGIC {
        return Err(PackError::Magic);
    }
    let word = |at: usize| u32::from_le_bytes(header[at..at + 4].try_into().expect("a word"));
    let format = PackFormat::read(header)?;
    let count = word(MAGIC.len() + 4) as usize;
    if count > MAX_RECORDS {
        return Err(PackError::Count);
    }
    if header[MAGIC.len() + 8..HEADER_BYTES] != translator.0 {
        return Err(PackError::Translator);
    }
    Ok((format, count))
}

/// A verified index over a pack's header, index and checksum. The record
/// ranges are relative to the start of the pack. Their contents are checked
/// separately when decoded.
#[derive(Debug, Clone, Copy)]
pub struct PackIndex<'a> {
    bytes: &'a [u8],
    count: usize,
    length: usize,
    format: PackFormat,
}

impl<'a> PackIndex<'a> {
    /// Check the header, index checksum, key order and each record's range.
    /// The prefix may end before the first record. A reader can compare
    /// [`Self::length`] with the actual file size before reading a record.
    pub fn parse(prefix: &'a [u8], translator: &Digest) -> Result<Self, PackError> {
        let (format, count) = read_header(prefix, translator)?;
        let records_start = prefix_length(count);
        if prefix.len() < records_start {
            return Err(PackError::Truncated);
        }
        let index_end = records_start - CHECKSUM_BYTES;
        if Sha256::digest(&prefix[..index_end]).as_slice() != &prefix[index_end..records_start] {
            return Err(PackError::Checksum);
        }
        let mut length = records_start;
        let mut previous: Option<Digest> = None;
        for index in 0..count {
            let (key, offset, bytes) = entry(prefix, index);
            let end = offset.checked_add(bytes).ok_or(PackError::Index)?;
            if previous.is_some_and(|previous| previous >= key)
                || offset < records_start as u64
                || bytes == 0
            {
                return Err(PackError::Index);
            }
            length = length.max(usize::try_from(end).map_err(|_| PackError::Index)?);
            previous = Some(key);
        }
        Ok(Self {
            bytes: &prefix[..records_start],
            count,
            length,
            format,
        })
    }

    /// The pack's byte length as declared by its index.
    pub fn length(&self) -> u64 {
        self.length as u64
    }

    /// The size of the verified header, index and checksum.
    pub fn prefix_bytes(&self) -> usize {
        prefix_length(self.count)
    }

    /// The byte range of the record for page key `key`.
    pub fn record_range(&self, key: &Digest) -> Option<Range<u64>> {
        let (mut low, mut high) = (0, self.count);
        while low < high {
            let middle = (low + high) / 2;
            let (found, offset, bytes) = entry(self.bytes, middle);
            match found.cmp(key) {
                core::cmp::Ordering::Less => low = middle + 1,
                core::cmp::Ordering::Greater => high = middle,
                core::cmp::Ordering::Equal => return Some(offset..offset + bytes),
            }
        }
        None
    }

    /// Every record's key and byte range, in key order.
    pub fn iter(&self) -> impl Iterator<Item = (Digest, Range<u64>)> + '_ {
        (0..self.count).map(|index| {
            let (key, offset, bytes) = entry(self.bytes, index);
            (key, offset..offset + bytes)
        })
    }
}

/// A pack whose prefix and all of its record bytes are resident.
#[derive(Debug, Clone, Copy)]
pub struct Pack<'a> {
    // The resident pack retains the entire byte range on which the verified
    // index is based. Index access still uses only the header and index.
    index: PackIndex<'a>,
}

impl<'a> Pack<'a> {
    /// The length declared by the verified index at the start of `prefix`.
    pub fn length(prefix: &[u8], translator: &Digest) -> Result<u64, PackError> {
        Ok(PackIndex::parse(prefix, translator)?.length())
    }

    /// Verify the index and require all the referenced record bytes.
    /// Bytes after the last record are ignored.
    pub fn parse(bytes: &'a [u8], translator: &Digest) -> Result<Self, PackError> {
        let mut index = PackIndex::parse(bytes, translator)?;
        index.bytes = bytes.get(..index.length).ok_or(PackError::Truncated)?;
        Ok(Self { index })
    }

    pub fn records(&self) -> usize {
        self.index.count
    }

    /// The pack's length in bytes.
    pub fn bytes(&self) -> usize {
        self.index.length
    }

    /// The record bytes of page key `key`, unchecked.
    pub fn get(&self, key: &Digest) -> Option<&'a [u8]> {
        Some(self.body(self.index.record_range(key)?))
    }

    fn body(&self, range: Range<u64>) -> &'a [u8] {
        &self.index.bytes[range.start as usize..range.end as usize]
    }

    fn record(&self, key: &Digest) -> Option<PackedRecord<'a>> {
        Some(PackedRecord {
            bytes: self.get(key)?,
            format: self.index.format,
            checked: false,
        })
    }

    /// Every record's page key and bytes, in key order.
    pub fn iter(&self) -> impl Iterator<Item = (Digest, &'a [u8])> + '_ {
        self.index
            .iter()
            .map(|(key, range)| (key, self.body(range)))
    }
}

/// The stored bytes and the verified index's record format.
#[derive(Debug, Clone, Copy)]
pub struct PackedRecord<'a> {
    bytes: &'a [u8],
    format: PackFormat,
    /// These bytes decoded whole once already, with every check: they are
    /// immutable, so a decode again checks only the format's bounds.
    checked: bool,
}

impl PackedRecord<'_> {
    pub fn is_compressed(self) -> bool {
        self.format == PackFormat::Mixed && self.bytes.starts_with(codec::MAGIC)
    }

    fn same_storage(self, other: Self) -> bool {
        self.format == other.format && core::ptr::eq(self.bytes, other.bytes)
    }

    fn decode(
        self,
        translator: &Digest,
        key: &Digest,
        scratch: &mut DecodeScratch,
    ) -> Result<Record, DecodeError> {
        let raw = if self.is_compressed() {
            scratch.inflate(self.bytes)?
        } else {
            self.bytes
        };
        Record::decode_checked(raw, translator, key, !self.checked).map_err(|error| match error {
            FormatError::Capacity => DecodeError::Capacity,
            _ => DecodeError::Invalid,
        })
    }
}

/// Index entry `index` of the pack starting `bytes`: page key, offset and
/// length.
fn entry(bytes: &[u8], index: usize) -> (Digest, u64, u64) {
    let at = HEADER_BYTES + index * INDEX_ENTRY_BYTES;
    let field = |offset: usize| {
        u64::from_le_bytes(
            bytes[at + offset..at + offset + 8]
                .try_into()
                .expect("a field"),
        )
    };
    (
        Digest(bytes[at..at + DIGEST_BYTES].try_into().expect("a key")),
        field(DIGEST_BYTES),
        field(DIGEST_BYTES + 8),
    )
}

/// A record of a pack being laid out: one of the read pack's, or new.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Piece<'a> {
    Kept(&'a [u8]),
    New(Vec<u8>),
}

impl Piece<'_> {
    fn bytes(&self) -> &[u8] {
        match self {
            Self::Kept(bytes) => bytes,
            Self::New(bytes) => bytes,
        }
    }
}

/// A pack to write: its header, index and checksum, then its records at
/// their offsets, with zeros in any room between them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout<'a> {
    prefix: Vec<u8>,
    /// Each record's offset and bytes, by offset.
    records: Vec<(u64, Piece<'a>)>,
    length: u64,
    /// The byte ranges that differ from the pack this one was planned over,
    /// in order; `None` when every byte may.
    changed: Option<Vec<core::ops::Range<u64>>>,
}

/// Zeros for the room between records.
static ZEROS: [u8; 4096] = [0; 4096];

impl<'a> Layout<'a> {
    /// The pack of `records` (page keys and their record bytes, each key
    /// once) for `translator`, its records in key order after the index.
    pub fn new(translator: &Digest, records: Vec<(Digest, Piece<'a>)>) -> Self {
        Self::with_format(translator, records, PackFormat::Raw)
    }

    fn with_format(
        translator: &Digest,
        records: Vec<(Digest, Piece<'a>)>,
        format: PackFormat,
    ) -> Self {
        let start = prefix_length(records.len()) as u64;
        Self::compact(translator, records, format, start)
    }

    /// Every record laid out anew in key order from `start`.
    fn compact(
        translator: &Digest,
        mut records: Vec<(Digest, Piece<'a>)>,
        format: PackFormat,
        start: u64,
    ) -> Self {
        records.sort_by_key(|record| record.0);
        let mut offset = start;
        let placed = records
            .into_iter()
            .map(|(key, piece)| {
                let at = offset;
                offset += piece.bytes().len() as u64;
                (key, at, piece)
            })
            .collect();
        Self::placed(translator, placed, format, None)
    }

    /// The layout of records at the offsets given, which lie after the
    /// prefix and do not overlap.
    fn placed(
        translator: &Digest,
        mut records: Vec<(Digest, u64, Piece<'a>)>,
        format: PackFormat,
        changed: Option<Vec<core::ops::Range<u64>>>,
    ) -> Self {
        records.sort_by_key(|record| record.0);
        let count = records.len();
        let mut prefix = Vec::with_capacity(prefix_length(count));
        prefix.extend(MAGIC);
        prefix.extend((format as u32).to_le_bytes());
        prefix.extend((count as u32).to_le_bytes());
        prefix.extend(translator.0);
        for (key, offset, piece) in &records {
            prefix.extend(key.0);
            prefix.extend(offset.to_le_bytes());
            prefix.extend((piece.bytes().len() as u64).to_le_bytes());
        }
        let checksum = Sha256::digest(&prefix);
        prefix.extend(checksum);
        let mut records: Vec<(u64, Piece<'a>)> = records
            .into_iter()
            .map(|(_, offset, piece)| (offset, piece))
            .collect();
        records.sort_by_key(|record| record.0);
        let length = records
            .last()
            .map_or(prefix.len() as u64, |(offset, piece)| {
                offset + piece.bytes().len() as u64
            });
        debug_assert!(
            records
                .first()
                .is_none_or(|(offset, _)| *offset >= prefix.len() as u64),
            "records after the prefix"
        );
        Self {
            prefix,
            records,
            length,
            changed,
        }
    }

    pub fn records(&self) -> usize {
        self.records.len()
    }

    /// The pack's length in bytes.
    pub fn bytes(&self) -> u64 {
        self.length
    }

    /// The pack's bytes, in order.
    pub fn chunks(&self) -> impl Iterator<Item = &[u8]> + '_ {
        let mut chunks: Vec<&[u8]> = Vec::with_capacity(1 + 2 * self.records.len());
        chunks.push(&self.prefix);
        let mut at = self.prefix.len() as u64;
        for (offset, piece) in &self.records {
            for gap in zeros(offset - at) {
                chunks.push(gap);
            }
            chunks.push(piece.bytes());
            at = offset + piece.bytes().len() as u64;
        }
        chunks.into_iter()
    }

    pub fn to_vec(&self) -> Vec<u8> {
        self.chunks().flatten().copied().collect()
    }

    /// The ranges of [`Self::chunks`] that differ from the pack it was planned
    /// over, in order: a writer may write only them, the index last. `None`
    /// when every byte may differ.
    pub fn changed(&self) -> Option<&[core::ops::Range<u64>]> {
        self.changed.as_deref()
    }

    /// Fill `buffer` with the pack's bytes from `offset`, zeros past its end.
    pub fn read_at(&self, offset: u64, buffer: &mut [u8]) {
        buffer.fill(0);
        let end = offset + buffer.len() as u64;
        let copy = |buffer: &mut [u8], start: u64, bytes: &[u8]| {
            let from = offset.max(start);
            let to = end.min(start + bytes.len() as u64);
            if from < to {
                buffer[(from - offset) as usize..(to - offset) as usize]
                    .copy_from_slice(&bytes[(from - start) as usize..(to - start) as usize]);
            }
        };
        copy(buffer, 0, &self.prefix);
        let first = self
            .records
            .partition_point(|(start, piece)| start + piece.bytes().len() as u64 <= offset);
        for (start, piece) in &self.records[first..] {
            if *start >= end {
                break;
            }
            copy(buffer, *start, piece.bytes());
        }
    }
}

/// `bytes` zeros, in slices of [`ZEROS`].
fn zeros(bytes: u64) -> impl Iterator<Item = &'static [u8]> {
    let whole = (bytes / ZEROS.len() as u64) as usize;
    let rest = (bytes % ZEROS.len() as u64) as usize;
    core::iter::repeat_n(ZEROS.as_slice(), whole).chain((rest != 0).then(|| &ZEROS[..rest]))
}

/// The bytes of the header, index and checksum of a pack of `records`.
fn prefix_length(records: usize) -> usize {
    HEADER_BYTES + records * INDEX_ENTRY_BYTES + CHECKSUM_BYTES
}

/// The pack bytes a layout of `records` records holding `bytes` record
/// bytes takes.
fn layout_bytes(records: usize, bytes: u64) -> u64 {
    prefix_length(records) as u64 + bytes
}

/// The layout of `records`, the plan's choice within `bound`, over the pack
/// read, `previous`: each record it keeps unchanged stays where it was, the
/// others follow its last, and only they and the index are written. When
/// the index outgrows the room before the records, the unused room passes
/// [`MAX_WASTE_PERCENT`] or the pack would pass `bound`, every record is
/// laid out again, with room for the index to grow.
fn layout_over<'a>(
    translator: &Digest,
    previous: Option<Pack<'a>>,
    records: Vec<(Digest, Piece<'a>)>,
    format: PackFormat,
    bound: u64,
) -> Layout<'a> {
    let relayout = |records: Vec<(Digest, Piece<'a>)>| {
        let used: u64 = records
            .iter()
            .map(|(_, piece)| piece.bytes().len() as u64)
            .sum();
        let slack = (records.len() / INDEX_SLACK_DIVISOR).max(MIN_INDEX_SLACK);
        let roomy =
            (prefix_length(records.len() + slack) as u64).next_multiple_of(RECORD_ALIGNMENT);
        // Without room in the bound for the slack, the records follow the
        // index directly.
        let start = if roomy + used <= bound {
            roomy
        } else {
            prefix_length(records.len()) as u64
        };
        Layout::compact(translator, records, format, start)
    };
    let Some(pack) = previous else {
        return relayout(records);
    };
    let base = pack.index.bytes.as_ptr() as usize;
    let first_record = pack
        .iter()
        .map(|(_, bytes)| (bytes.as_ptr() as usize - base) as u64)
        .min()
        .unwrap_or(pack.index.length());
    let prefix = prefix_length(records.len()) as u64;
    if prefix > first_record {
        return relayout(records);
    }
    // A kept record that is the read pack's slot for its key keeps its
    // offset.
    let mut placed = Vec::with_capacity(records.len());
    let mut appended = Vec::new();
    for (key, piece) in records {
        let kept = match &piece {
            Piece::Kept(bytes) => pack
                .get(&key)
                .filter(|slot| core::ptr::eq(*slot, *bytes))
                .map(|slot| (slot.as_ptr() as usize - base) as u64),
            Piece::New(_) => None,
        };
        match kept {
            Some(offset) => placed.push((key, offset, piece)),
            None => appended.push((key, piece)),
        }
    }
    let mut end = placed
        .iter()
        .map(|(_, offset, piece)| offset + piece.bytes().len() as u64)
        .max()
        .unwrap_or(first_record)
        .max(first_record);
    let mut changed = Vec::new();
    appended.sort_by_key(|record| record.0);
    for (key, piece) in appended {
        let bytes = piece.bytes().len() as u64;
        changed.push(end..end + bytes);
        placed.push((key, end, piece));
        end += bytes;
    }
    let used: u64 = placed
        .iter()
        .map(|(_, _, piece)| piece.bytes().len() as u64)
        .sum();
    // Kept records of a corrupt index can overlap and use more than the
    // span: lay those out again too.
    let waste = (end - first_record).checked_sub(used);
    if end > bound || waste.is_none_or(|waste| waste * 100 > end * MAX_WASTE_PERCENT) {
        return relayout(
            placed
                .into_iter()
                .map(|(key, _, piece)| (key, piece))
                .collect(),
        );
    }
    // The index changes whenever a record comes, goes or moves; a pack the
    // same in every byte is not written at all.
    let same = placed.len() == pack.records() && changed.is_empty();
    let mut ranges = Vec::new();
    if !same {
        ranges.push(0..prefix);
        ranges.extend(changed);
    }
    Layout::placed(translator, placed, format, Some(ranges))
}

/// What a boot's store did.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Counters {
    /// Records looked up and valid, absent, or invalid.
    pub loaded: u64,
    pub absent: u64,
    pub invalid: u64,
    /// Lookups of a key looked up before (its page let go and read again,
    /// or another PE's), those the decoded cache served, and records
    /// decoded (inflated and checked).
    pub repeated: u64,
    pub cached: u64,
    pub decoded: u64,
    /// Blocks offered and kept for proof, and those past the memory bound.
    pub offered: u64,
    pub dropped: u64,
    /// Blocks proven during the run ([`Store::prove_batch`]), and proven
    /// entries left out because the proven bound was full.
    pub proven: u64,
    pub unprovable: u64,
    pub unkept: u64,
    /// Shared lookups that found a block, and why the others missed: no
    /// offer of it, an offer of its PC and page in another context, its
    /// offer evicted, or another machine compiling it at that moment.
    pub shared_hits: u64,
    pub shared_absent: u64,
    pub shared_other_context: u64,
    pub shared_evicted: u64,
    pub shared_in_flight: u64,
}

impl Counters {
    /// Add `other`'s counts, as of another part of one store.
    pub fn add(&mut self, other: &Self) {
        let Self {
            loaded,
            absent,
            invalid,
            repeated,
            cached,
            decoded,
            offered,
            dropped,
            proven,
            unprovable,
            unkept,
            shared_hits,
            shared_absent,
            shared_other_context,
            shared_evicted,
            shared_in_flight,
        } = other;
        self.loaded += loaded;
        self.absent += absent;
        self.invalid += invalid;
        self.repeated += repeated;
        self.cached += cached;
        self.decoded += decoded;
        self.offered += offered;
        self.dropped += dropped;
        self.proven += proven;
        self.unprovable += unprovable;
        self.unkept += unkept;
        self.shared_hits += shared_hits;
        self.shared_absent += shared_absent;
        self.shared_other_context += shared_other_context;
        self.shared_evicted += shared_evicted;
        self.shared_in_flight += shared_in_flight;
    }
}

/// A block offered for proof: its place in its page and its input's in the
/// page's [`PendingPage::inputs`]. The proof compiles the block again.
#[derive(Debug, Clone, Copy)]
struct PendingBlock {
    offset: u16,
    length: u16,
    start: u32,
}

/// A page's blocks of one context waiting for proof, by offset. What the
/// blocks share is held once, so a block costs little more than its words.
#[derive(Debug)]
struct PendingPage {
    preparation: Arc<Preparation>,
    blocks: Vec<PendingBlock>,
    inputs: Vec<u8>,
}

impl PendingPage {
    /// What the page is charged against the pending bound.
    fn charge(&self) -> usize {
        self.blocks.capacity() * size_of::<PendingBlock>()
            + self.inputs.capacity()
            + PENDING_PAGE_BYTES
    }
}

/// A pending page's key, guest page and context.
type PendingKey = (Digest, u64, Digest);

/// Stage bounded growth without changing a retained allocation. Either
/// buffer may fail, so the owner installs their replacements together.
fn grow_pending<T: Copy>(
    current: Option<&Vec<T>>,
    additional: usize,
    minimum: usize,
    limit: usize,
) -> Result<Option<Vec<T>>, ()> {
    let held = current.map_or(0, Vec::capacity);
    let used = current.map_or(0, Vec::len);
    let required = used.checked_add(additional).ok_or(())?;
    if required.max(held) > limit {
        return Err(());
    }
    if required <= held {
        return Ok(None);
    }
    let capacity = held.saturating_mul(2).max(minimum).max(required).min(limit);
    let mut replacement = Vec::new();
    replacement.try_reserve_exact(capacity).map_err(|_| ())?;
    if replacement.capacity() > limit {
        return Err(());
    }
    if let Some(current) = current {
        replacement.extend_from_slice(current);
    }
    Ok(Some(replacement))
}

/// A shared block's PC and the leading words of its page key and context:
/// a compact index whose rare collision only replaces an entry, as a hit
/// still matches the whole key and context ([`SharedTranslation::matches`]).
type SharedKey = (u64, u64, u64);

fn shared_key(key: &Digest, pc: u64, context: &Digest) -> SharedKey {
    let word = |digest: &Digest| u64::from_le_bytes(digest.0[..8].try_into().expect("a word"));
    (pc, word(key), word(context))
}

/// A slot of the [`MissTables`] for `words`.
fn miss_slot(words: &[u64]) -> usize {
    let mixed = words.iter().fold(0u64, |hash, word| {
        (hash ^ word).wrapping_mul(0x9e37_79b9_7f4a_7c15)
    });
    (mixed >> 32) as usize % MISS_SLOTS
}

/// Why shared lookups miss, in direct-mapped tables a collision only
/// misnames: offers being compiled since their lookup missed, offers
/// evicted, and the context of the last offer of each PC and page.
#[derive(Debug, Default)]
struct MissTables {
    in_flight: Vec<SharedKey>,
    evicted: Vec<SharedKey>,
    offered: Vec<SharedKey>,
}

impl MissTables {
    fn ready(&mut self) -> bool {
        if self.in_flight.is_empty() {
            let mut tables = [Vec::new(), Vec::new(), Vec::new()];
            for table in &mut tables {
                if table.try_reserve_exact(MISS_SLOTS).is_err() {
                    return false;
                }
                table.resize(MISS_SLOTS, (u64::MAX, 0, 0));
            }
            [self.in_flight, self.evicted, self.offered] = tables;
        }
        true
    }

    /// Why the lookup of `key` missed, marking it in flight.
    fn miss(&mut self, key: SharedKey, counters: &mut Counters) {
        if !self.ready() {
            counters.shared_absent += 1;
            return;
        }
        let slot = miss_slot(&[key.0, key.1, key.2]);
        let place = miss_slot(&[key.0, key.1]);
        let reason = if self.in_flight[slot] == key {
            &mut counters.shared_in_flight
        } else if self.evicted[slot] == key {
            &mut counters.shared_evicted
        } else if self.offered[place].0 == key.0
            && self.offered[place].1 == key.1
            && self.offered[place].2 != key.2
        {
            &mut counters.shared_other_context
        } else {
            &mut counters.shared_absent
        };
        *reason += 1;
        self.in_flight[slot] = key;
    }

    fn offered(&mut self, key: SharedKey) {
        if !self.ready() {
            return;
        }
        let slot = miss_slot(&[key.0, key.1, key.2]);
        if self.in_flight[slot] == key {
            self.in_flight[slot] = (u64::MAX, 0, 0);
        }
        if self.evicted[slot] == key {
            self.evicted[slot] = (u64::MAX, 0, 0);
        }
        self.offered[miss_slot(&[key.0, key.1])] = key;
    }

    fn evicted(&mut self, key: SharedKey) {
        if self.ready() {
            self.evicted[miss_slot(&[key.0, key.1, key.2])] = key;
        }
    }
}

#[derive(Debug)]
struct SharedEntry {
    translation: Arc<SharedTranslation>,
    /// Immutable owner charge, computed once before admission.
    bytes: usize,
}

/// One local producer's shared offer, allocated and charged before the
/// producer acquires the store lock.
pub struct StagedOffer {
    key: SharedKey,
    entry: Option<SharedEntry>,
}

impl StagedOffer {
    /// `bound` is the producer's snapshot of the shared cache limit.
    /// Admission rechecks the store's limit.
    pub fn new(offer: Offer, bound: usize) -> Option<Self> {
        let (key, bytes) = Self::charge(&offer, bound)?;
        Some(Self::entry(key, bytes, offer.shared()))
    }

    /// Stage only the shared emission while leaving its producer's proof
    /// input available. Admission still checks the live shared bound.
    pub fn for_writable(offer: &Offer, bound: usize) -> Option<Self> {
        let (key, bytes) = Self::charge(offer, bound)?;
        Some(Self::entry(key, bytes, offer.shared_for_proof()))
    }

    fn charge(offer: &Offer, bound: usize) -> Option<(SharedKey, usize)> {
        if bound == 0 {
            return None;
        }
        let bytes = offer
            .retained_bytes()
            .saturating_add(size_of::<(SharedKey, SharedEntry)>());
        (bytes <= bound).then(|| {
            (
                shared_key(&offer.key(), offer.pc(), &offer.context()),
                bytes,
            )
        })
    }

    fn entry(key: SharedKey, bytes: usize, translation: Arc<SharedTranslation>) -> Self {
        Self {
            key,
            entry: Some(SharedEntry { translation, bytes }),
        }
    }
}

/// At most one displaced owner per producer can outlive the store lock.
/// The cap covers charged cache bytes, with the same exclusions as the
/// shared cache budget. Larger or additional owners are released inside.
const RETIRED_OWNER_BYTES: usize = 64 << 10;

fn retire(entry: SharedEntry, deferred: &mut Option<Arc<SharedTranslation>>) {
    if deferred.is_none() && entry.bytes <= RETIRED_OWNER_BYTES {
        *deferred = Some(entry.translation);
    }
}

/// A validated record paired with the immutable pack bytes it came from.
pub struct DecodedRecord<'a> {
    stored: PackedRecord<'a>,
    record: Arc<Record>,
    retained_bytes: usize,
}

impl<'a> DecodedRecord<'a> {
    pub fn decode(
        stored: PackedRecord<'a>,
        translator: &Digest,
        key: &Digest,
        scratch: &mut DecodeScratch,
    ) -> Result<Self, DecodeError> {
        let record = Arc::new(stored.decode(translator, key, scratch)?);
        let retained_bytes = record.retained_bytes();
        Ok(Self {
            stored,
            record,
            retained_bytes,
        })
    }

    pub fn into_record(self) -> Arc<Record> {
        self.record
    }
}

/// Decoded pack records, least recently used first out, within a bound of
/// their retained bytes.
#[derive(Debug, Default)]
struct DecodedCache {
    /// Each record, its charge and its last use.
    records: BTreeMap<Digest, (Arc<Record>, usize, u64)>,
    /// Keys by last use.
    uses: BTreeMap<u64, Digest>,
    bytes: usize,
    bound: usize,
    clock: u64,
    /// Every record decoded, while a machine still holds it with a page it
    /// keeps: the cache holds a small part of what the machines hold
    /// between them, and a record one of them still has need not be
    /// decoded again for another.
    held: BTreeMap<Digest, Weak<Record>>,
    /// The size of `held` at which the records nobody holds leave it.
    held_sweep: usize,
}

/// The fewest records [`DecodedCache::held`] sweeps at.
const HELD_SWEEP: usize = 1024;

impl DecodedCache {
    fn new(bound: usize) -> Self {
        Self {
            bound,
            ..Self::default()
        }
    }

    fn get(&mut self, key: &Digest) -> Option<Arc<Record>> {
        let Some((record, _, used)) = self.records.get_mut(key) else {
            return self.held.get(key)?.upgrade();
        };
        self.clock += 1;
        self.uses.remove(used);
        *used = self.clock;
        self.uses.insert(self.clock, *key);
        Some(Arc::clone(record))
    }

    /// Note that `record` was handed out, for [`Self::get`] to find while
    /// anyone holds it.
    fn hold(&mut self, record: &Arc<Record>) {
        if self.held.len() >= self.held_sweep.max(HELD_SWEEP) {
            self.held.retain(|_, held| held.strong_count() != 0);
            self.held_sweep = 2 * self.held.len();
        }
        self.held.insert(record.key, Arc::downgrade(record));
    }

    /// Keep `record`, letting the least recently used go until it fits:
    /// those go back to the caller, to drop outside the store's lock.
    fn insert(&mut self, record: &Arc<Record>, bytes: usize) -> Vec<Arc<Record>> {
        let mut evicted = Vec::new();
        self.hold(record);
        if bytes > self.bound / 8 || self.records.contains_key(&record.key) {
            return evicted;
        }
        while self.bytes + bytes > self.bound {
            let Some((_, key)) = self.uses.pop_first() else {
                break;
            };
            if let Some((old, charge, _)) = self.records.remove(&key) {
                self.bytes -= charge;
                evicted.push(old);
            }
        }
        self.clock += 1;
        self.records
            .insert(record.key, (Arc::clone(record), bytes, self.clock));
        self.uses.insert(self.clock, record.key);
        self.bytes += bytes;
        evicted
    }

    /// Let records go until the cache fits `bound`.
    fn shrink(&mut self, bound: usize) -> Vec<Arc<Record>> {
        self.bound = bound;
        let mut evicted = Vec::new();
        while self.bytes > bound {
            let Some((_, key)) = self.uses.pop_first() else {
                break;
            };
            if let Some((old, charge, _)) = self.records.remove(&key) {
                self.bytes -= charge;
                evicted.push(old);
            }
        }
        evicted
    }
}

/// Pending pages taken from a store to prove outside its lock
/// ([`Store::take_batch`]).
pub struct ProofBatch {
    pages: Vec<(PendingKey, PendingPage)>,
}

/// What proving a [`ProofBatch`] made: each page key's new entries, and
/// the blocks proven and refused.
pub struct ProvenBatch {
    records: Vec<(Digest, Vec<super::Entry>)>,
    proven: u64,
    unprovable: u64,
}

impl ProofBatch {
    /// The blocks in the batch.
    pub fn blocks(&self) -> usize {
        self.pages.iter().map(|(_, page)| page.blocks.len()).sum()
    }

    /// Compile each block again at its page's probe addresses and keep the
    /// entries proven to move only by them. Needs no store.
    pub fn prove(self) -> ProvenBatch {
        let (mut proven, mut unprovable) = (0, 0);
        let mut records: Vec<(Digest, Vec<super::Entry>)> = Vec::new();
        for ((key, page, context), held) in self.pages {
            let mut entries = Vec::new();
            for block in &held.blocks {
                let start = block.start as usize;
                let input = &held.inputs[start..start + usize::from(block.length)];
                let pc = page + u64::from(block.offset);
                match Offer::prove_input(input, page, pc, context, &held.preparation) {
                    Some(entry) => {
                        proven += 1;
                        entries.push(entry);
                    }
                    None => unprovable += 1,
                }
            }
            match records.last_mut() {
                Some((last, held)) if *last == key => held.extend(entries),
                _ => records.push((key, entries)),
            }
        }
        ProvenBatch {
            records,
            proven,
            unprovable,
        }
    }
}

/// One boot's store over the pack it read.
#[derive(Debug)]
pub struct Store<'a> {
    translator: Digest,
    pack: Option<Pack<'a>>,
    /// The keys looked up, and whether the pack held a valid record for
    /// each. The records themselves are the engine's to keep: decoded, they
    /// take several times the pack's bytes.
    loaded: BTreeMap<Digest, bool>,
    decoded: DecodedCache,
    decode_scratch: DecodeScratch,
    pending: BTreeMap<PendingKey, PendingPage>,
    /// The blocks in `pending`.
    pending_blocks: usize,
    /// What `pending` holds: a charge for each page's entry and tree node,
    /// and the capacity of its buffers.
    pending_bytes: usize,
    pending_bound: usize,
    /// Entries proven during the run, by page key, and their retained bytes.
    proven: BTreeMap<Digest, Record>,
    proven_bytes: usize,
    proven_bound: usize,
    /// A fold left an entry out for the proven bound: a batch proven now
    /// would be left out too, so none is taken until the bound grows
    /// ([`Self::take_batch`]). Its pages wait for the stop's proof instead.
    proven_full: bool,
    /// The blocks offered, by page key, PC and context, for the other
    /// machines sharing the store, oldest first. The bound charges kept
    /// source/template buffers, Arc payloads, index entries and FIFO capacity;
    /// allocator node slack/bookkeeping and shared Preparation are excluded.
    shared: BTreeMap<SharedKey, SharedEntry>,
    shared_order: VecDeque<SharedKey>,
    shared_bytes: usize,
    shared_bound: usize,
    misses: MissTables,
    pub counters: Counters,
}

/// The next pack, and what laying it out did.
#[derive(Debug)]
pub struct Planned<'a> {
    pub layout: Layout<'a>,
    /// Blocks proven and refused by their proofs, and those left unproven
    /// when the room to prove them ran out.
    pub proven: u64,
    pub unprovable: u64,
    pub unproven: u64,
    /// Records of the read pack left out to stay within the bound, and new
    /// ones that did not fit.
    pub evicted: u64,
}

/// What a store keeps beyond its pack, in bytes ([`Store::set_bounds`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bounds {
    /// Inputs of blocks waiting for proof.
    pub pending: usize,
    /// Entries proven during the run.
    pub proven: usize,
    /// Decoded pack records.
    pub decoded: usize,
}

impl Default for Bounds {
    fn default() -> Self {
        Self {
            pending: MAX_PENDING_BYTES,
            proven: PROVEN_BYTES,
            decoded: DECODED_BYTES,
        }
    }
}

impl Bounds {
    /// Each of `parts` stores' share of these bounds.
    pub fn share(self, parts: usize) -> Self {
        let parts = parts.max(1);
        Self {
            pending: self.pending / parts,
            proven: self.proven / parts,
            decoded: self.decoded / parts,
        }
    }
}

impl<'a> Store<'a> {
    /// A store of `translator` over the pack `pack` holds, if any; `None`
    /// or a damaged pack (whose error comes back too) starts it empty.
    pub fn new(translator: Digest, pack: Option<&'a [u8]>) -> (Self, Option<PackError>) {
        let parsed = pack
            .map(|bytes| Pack::parse(bytes, &translator))
            .transpose();
        match parsed {
            Ok(pack) => (Self::over(translator, pack), None),
            Err(error) => (Self::over(translator, None), Some(error)),
        }
    }

    /// A store of `translator` over `pack`, already parsed for it: one part
    /// of a store split by page key.
    pub fn over(translator: Digest, pack: Option<Pack<'a>>) -> Self {
        let bounds = Bounds::default();
        Self {
            translator,
            pack,
            loaded: BTreeMap::new(),
            decoded: DecodedCache::new(bounds.decoded),
            decode_scratch: DecodeScratch::default(),
            pending: BTreeMap::new(),
            pending_blocks: 0,
            pending_bytes: 0,
            pending_bound: bounds.pending,
            proven: BTreeMap::new(),
            proven_bytes: 0,
            proven_bound: bounds.proven,
            proven_full: false,
            shared: BTreeMap::new(),
            shared_order: VecDeque::new(),
            shared_bytes: 0,
            shared_bound: 0,
            misses: MissTables::default(),
            counters: Counters::default(),
        }
    }

    /// Bound what the store keeps beyond its pack. Records the decoded
    /// cache lets go come back, to drop outside a lock.
    pub fn set_bounds(&mut self, bounds: Bounds) -> Vec<Arc<Record>> {
        self.pending_bound = bounds.pending;
        if bounds.proven > self.proven_bound {
            self.proven_full = false;
        }
        self.proven_bound = bounds.proven;
        self.decoded.shrink(bounds.decoded)
    }

    /// Take `other`'s run (another part of one store over the same pack):
    /// what it looked up, its blocks waiting for proof, its proven entries
    /// and its counts. Its shared blocks are dropped.
    pub fn absorb(&mut self, other: Self) {
        for (key, valid) in other.loaded {
            self.loaded.entry(key).or_insert(valid);
        }
        for (group, page) in other.pending {
            match self.pending.entry(group) {
                alloc::collections::btree_map::Entry::Vacant(slot) => {
                    self.pending_blocks += page.blocks.len();
                    self.pending_bytes += page.charge();
                    slot.insert(page);
                }
                alloc::collections::btree_map::Entry::Occupied(mut slot) => {
                    // One page's offers reach one part: merge them anyway.
                    let held = slot.get_mut();
                    let before = held.charge();
                    for block in page.blocks {
                        if let Err(place) = held
                            .blocks
                            .binary_search_by_key(&block.offset, |held| held.offset)
                        {
                            let start = held.inputs.len();
                            let from = block.start as usize;
                            held.inputs.extend_from_slice(
                                &page.inputs[from..from + usize::from(block.length)],
                            );
                            held.blocks.insert(
                                place,
                                PendingBlock {
                                    start: start as u32,
                                    ..block
                                },
                            );
                            self.pending_blocks += 1;
                        }
                    }
                    self.pending_bytes += held.charge() - before;
                }
            }
        }
        for (key, record) in other.proven {
            match self.proven.entry(key) {
                alloc::collections::btree_map::Entry::Vacant(slot) => {
                    slot.insert(record);
                }
                alloc::collections::btree_map::Entry::Occupied(mut slot) => {
                    slot.get_mut().merge(record);
                }
            }
        }
        self.proven_bytes += other.proven_bytes;
        self.proven_full |= other.proven_full;
        self.counters.add(&other.counters);
    }

    /// Keep offered blocks and their buffers, up to `bound` bytes, for
    /// [`Translations::prepared`]: for machines that share the store. Full,
    /// it forgets the oldest until a new one fits.
    pub fn share(&mut self, bound: usize) {
        self.shared_bound = bound;
        while self.shared_usage() > bound {
            if self.evict_shared().is_none() {
                self.shared_order = VecDeque::new();
                break;
            }
        }
    }

    fn shared_usage(&self) -> usize {
        self.shared_bytes
            .saturating_add(self.shared_order.capacity() * size_of::<SharedKey>())
    }

    fn evict_shared(&mut self) -> Option<SharedEntry> {
        while let Some(oldest) = self.shared_order.pop_front() {
            if let Some(evicted) = self.shared.remove(&oldest) {
                self.shared_bytes -= evicted.bytes;
                self.misses.evicted(oldest);
                return Some(evicted);
            }
        }
        None
    }

    /// Admit a preallocated shared block. The producer can drop a displaced
    /// owner after releasing the lock. The producer still owns a declined
    /// block, so declining it does not drop its buffers while locked.
    pub fn accept_shared(&mut self, staged: &mut StagedOffer) -> Option<Arc<SharedTranslation>> {
        use alloc::collections::btree_map::Entry;

        let key = staged.key;
        self.misses.offered(key);
        let bytes = staged.entry.as_ref()?.bytes;
        if bytes > self.shared_bound {
            return None;
        }
        let mut deferred = None;
        loop {
            let usage = self.shared_usage();
            match self.shared.entry(key) {
                Entry::Occupied(mut slot)
                    if usage.saturating_sub(slot.get().bytes).saturating_add(bytes)
                        <= self.shared_bound =>
                {
                    self.shared_bytes = self.shared_bytes - slot.get().bytes + bytes;
                    let old = slot.insert(staged.entry.take().expect("staged entry"));
                    retire(old, &mut deferred);
                    return deferred;
                }
                Entry::Vacant(slot)
                    if usage.saturating_add(bytes) <= self.shared_bound
                        && self.shared_order.try_reserve(1).is_ok() =>
                {
                    // A FIFO allocation can itself exceed the budget.
                    let fifo = self.shared_order.capacity() * size_of::<SharedKey>();
                    if self.shared_bytes.saturating_add(fifo).saturating_add(bytes)
                        <= self.shared_bound
                    {
                        slot.insert(staged.entry.take().expect("staged entry"));
                        self.shared_order.push_back(key);
                        self.shared_bytes += bytes;
                        return deferred;
                    }
                }
                _ => {}
            }
            let Some(evicted) = self.evict_shared() else {
                self.shared_order = VecDeque::new();
                return deferred;
            };
            retire(evicted, &mut deferred);
        }
    }

    /// Admit the original input for durable proof independently of shared
    /// emission admission. A writable owner may call both under one lock.
    pub fn keep_pending(&mut self, offer: &Offer) {
        self.misses
            .offered(shared_key(&offer.key(), offer.pc(), &offer.context()));
        if !self.reserve_pending(offer) {
            self.counters.dropped += 1;
        }
    }

    /// Keep `offer` for proof unless its page holds the block already:
    /// whether it fit.
    fn reserve_pending(&mut self, offer: &Offer) -> bool {
        const BLOCK_BYTES: usize = size_of::<PendingBlock>();
        let (Ok(offset), Ok(length)) = (
            u16::try_from(offer.pc() - offer.page()),
            u16::try_from(offer.input().len()),
        ) else {
            return false;
        };
        let group = (offer.key(), offer.page(), offer.context());
        if self
            .proven
            .get(&offer.key())
            .is_some_and(|record| record.find(offset, &offer.context()).is_some())
        {
            // Proven during the run already.
            return true;
        }
        let known = self.pending.get(&group);
        let place = known.map_or(Err(0), |page| {
            page.blocks
                .binary_search_by_key(&offset, |block| block.offset)
        });
        let Err(place) = place else {
            return true;
        };
        let new_page = if known.is_some() {
            0
        } else {
            PENDING_PAGE_BYTES
        };
        let held_slots = known.map_or(0, |page| page.blocks.capacity());
        let held_inputs = known.map_or(0, |page| page.inputs.capacity());
        let held = held_slots * BLOCK_BYTES + held_inputs;
        let Some(available) = self
            .pending_bound
            .checked_sub(self.pending_bytes)
            .and_then(|available| available.checked_sub(new_page))
        else {
            return false;
        };
        let limit = held + available;
        // A page's inputs stay far below 4 GiB: each offset is unique and
        // both it and the associated block length fit in 16 bits.
        let start = known.map_or(0, |page| page.inputs.len());
        let required_inputs = (start + usize::from(length)).max(held_inputs);
        let Some(slot_limit) = limit.checked_sub(required_inputs) else {
            return false;
        };
        let Ok(blocks) = grow_pending(
            known.map(|page| &page.blocks),
            1,
            4,
            slot_limit / BLOCK_BYTES,
        ) else {
            return false;
        };
        let slots = blocks.as_ref().map_or(held_slots, Vec::capacity);
        let Ok(inputs) = grow_pending(
            known.map(|page| &page.inputs),
            usize::from(length),
            8,
            limit - slots * BLOCK_BYTES,
        ) else {
            return false;
        };
        let page = self.pending.entry(group).or_insert_with(|| PendingPage {
            preparation: Arc::clone(offer.preparation()),
            blocks: Vec::new(),
            inputs: Vec::new(),
        });
        if let Some(blocks) = blocks {
            page.blocks = blocks;
        }
        if let Some(inputs) = inputs {
            page.inputs = inputs;
        }
        page.blocks.insert(
            place,
            PendingBlock {
                offset,
                length,
                start: start as u32,
            },
        );
        page.inputs.extend_from_slice(offer.input());
        let grown = page.blocks.capacity() * BLOCK_BYTES + page.inputs.capacity() - held;
        self.pending_bytes += grown + new_page;
        self.pending_blocks += 1;
        self.counters.offered += 1;
        true
    }

    /// Pending pages holding at least `blocks` blocks (all, when fewer
    /// wait), oldest key first, to prove outside the store's lock; `None`
    /// when none wait or the proven bound is full: reached, or short of an
    /// entry a fold left out. A batch then would be proven only to be
    /// dropped, and its pages with it.
    pub fn take_batch(&mut self, blocks: usize) -> Option<ProofBatch> {
        if self.pending.is_empty() || self.proven_full || self.proven_bytes >= self.proven_bound {
            return None;
        }
        let mut pages = Vec::new();
        let mut taken = 0;
        while taken < blocks.max(1)
            && let Some((group, page)) = self.pending.pop_first()
        {
            taken += page.blocks.len();
            self.pending_blocks -= page.blocks.len();
            self.pending_bytes -= page.charge();
            pages.push((group, page));
        }
        Some(ProofBatch { pages })
    }

    /// Keep what a [`ProofBatch`] of this store proved, within the proven
    /// bound; the rest is counted as unkept.
    pub fn fold(&mut self, batch: ProvenBatch) {
        self.counters.proven += batch.proven;
        self.counters.unprovable += batch.unprovable;
        for (key, entries) in batch.records {
            for entry in entries {
                let bytes = entry.retained_bytes();
                if self.proven_bytes + bytes > self.proven_bound {
                    self.counters.unkept += 1;
                    self.proven_full = true;
                    continue;
                }
                self.proven_bytes += bytes;
                self.proven
                    .entry(key)
                    .or_insert_with(|| Record::new(self.translator, key))
                    .insert(entry);
            }
        }
    }

    /// The translator whose records the store keeps.
    pub fn translator(&self) -> &Digest {
        &self.translator
    }

    /// A record already decoded and checked against this pack.
    pub fn cached_record(&mut self, key: &Digest) -> Option<Arc<Record>> {
        let record = self.decoded.get(key)?;
        self.counters.cached += 1;
        self.counters.repeated += 1;
        Some(record)
    }

    /// The pack's slot for `key`, unless an earlier lookup found it invalid.
    /// Its immutable bytes may be decoded after releasing the store lock.
    pub fn stored(&self, key: &Digest) -> Option<PackedRecord<'a>> {
        let pack = self.pack?;
        let loaded = self.loaded.get(key).copied();
        if loaded == Some(false) {
            return None;
        }
        let mut stored = pack.record(key)?;
        // A first lookup that decoded the slot checked its bytes whole.
        stored.checked = loaded == Some(true);
        Some(stored)
    }

    pub fn take_decode_scratch(&mut self, stored: PackedRecord<'a>) -> DecodeScratch {
        if stored.is_compressed() {
            core::mem::take(&mut self.decode_scratch)
        } else {
            DecodeScratch::default()
        }
    }

    /// A transient capacity failure can be retried. Only record a result
    /// for the exact stored slot we gave the decoder. The records the cache
    /// let go come back, for the caller to drop with the scratch outside
    /// the store lock.
    pub fn finish_decode(
        &mut self,
        key: &Digest,
        stored: PackedRecord<'a>,
        decoded: &Result<DecodedRecord<'a>, DecodeError>,
        scratch: &mut DecodeScratch,
    ) -> Vec<Arc<Record>> {
        self.decode_scratch.retain(scratch);
        if !self
            .stored(key)
            .is_some_and(|current| current.same_storage(stored))
        {
            return Vec::new();
        }
        match decoded {
            Ok(decoded)
                if decoded.stored.same_storage(stored)
                    && decoded.record.key == *key
                    && decoded.record.translator == self.translator =>
            {
                self.counters.decoded += 1;
                self.looked_up(key, true, true);
                self.decoded.insert(&decoded.record, decoded.retained_bytes)
            }
            Err(DecodeError::Invalid) => {
                self.looked_up(key, true, false);
                Vec::new()
            }
            _ => Vec::new(),
        }
    }

    /// Count `key`'s lookup: on the first, whether the pack held bytes for
    /// it, and whether they decoded.
    pub fn looked_up(&mut self, key: &Digest, found: bool, valid: bool) {
        let alloc::collections::btree_map::Entry::Vacant(slot) = self.loaded.entry(*key) else {
            self.counters.repeated += 1;
            return;
        };
        match (found, valid) {
            (true, true) => self.counters.loaded += 1,
            (true, false) => self.counters.invalid += 1,
            (false, _) => self.counters.absent += 1,
        }
        slot.insert(valid);
    }

    /// The pack read, if valid.
    pub fn pack(&self) -> Option<&Pack<'a>> {
        self.pack.as_ref()
    }

    /// Blocks waiting for proof.
    pub fn pending(&self) -> usize {
        self.pending_blocks
    }

    /// Prove the offered blocks while `room` says there is memory to, and
    /// lay out the next pack within `bound` bytes: the records that gained
    /// entries, then those this boot used, then the others, as long as
    /// they fit. Records kept unchanged stay where they were in the pack
    /// read ([`Layout::changed`]).
    pub fn plan(&mut self, bound: u64, room: impl Fn() -> bool) -> Planned<'a> {
        let mut proven = core::mem::take(&mut self.proven);
        self.proven_bytes = 0;
        self.proven_full = false;
        let (mut count, mut unprovable) = (0, 0);
        let pending = core::mem::take(&mut self.pending);
        let mut unproven = self.pending_blocks as u64;
        'pages: for ((key, page, context), held) in pending {
            for block in held.blocks {
                if !room() {
                    break 'pages;
                }
                unproven -= 1;
                let start = block.start as usize;
                let input = &held.inputs[start..start + usize::from(block.length)];
                let pc = page + u64::from(block.offset);
                let entry = Offer::prove_input(input, page, pc, context, &held.preparation);
                match entry {
                    Some(entry) => {
                        count += 1;
                        proven
                            .entry(key)
                            .or_insert_with(|| Record::new(self.translator, key))
                            .insert(entry);
                    }
                    None => unprovable += 1,
                }
            }
        }
        self.pending_blocks = 0;
        self.pending_bytes = 0;
        // Gained entries first, then this boot's records, then the others.
        let mut scratch = core::mem::take(&mut self.decode_scratch);
        let mut candidates: Vec<(Digest, Piece<'a>)> = Vec::new();
        for (key, record) in &mut proven {
            if let Some(stored) = self.pack.and_then(|pack| pack.record(key)) {
                match stored.decode(&self.translator, key, &mut scratch) {
                    Ok(previous) => {
                        let before = previous.entries.len();
                        record.merge(previous);
                        if record.entries.len() == before {
                            // Nothing new: the stored slot stays as it is.
                            candidates.push((*key, Piece::Kept(stored.bytes)));
                            continue;
                        }
                    }
                    Err(DecodeError::Invalid) => {}
                    Err(DecodeError::Capacity) => {
                        candidates.push((*key, Piece::Kept(stored.bytes)));
                        continue;
                    }
                }
            }
            candidates.push((*key, Piece::New(record.encode())));
        }
        self.decode_scratch.retain(&mut scratch);
        drop(scratch);
        if let Some(pack) = self.pack {
            for (key, valid) in &self.loaded {
                if *valid
                    && !proven.contains_key(key)
                    && let Some(bytes) = pack.get(key)
                {
                    candidates.push((*key, Piece::Kept(bytes)));
                }
            }
            for (key, bytes) in pack.iter() {
                if !self.loaded.contains_key(&key) && !proven.contains_key(&key) {
                    candidates.push((key, Piece::Kept(bytes)));
                }
            }
        }
        let (mut kept, mut kept_bytes, mut evicted) = (Vec::new(), 0u64, 0);
        for (key, piece) in candidates {
            let bytes = piece.bytes().len() as u64;
            if kept.len() < MAX_RECORDS && layout_bytes(kept.len() + 1, kept_bytes + bytes) <= bound
            {
                kept_bytes += bytes;
                kept.push((key, piece));
            } else {
                evicted += 1;
            }
        }
        let format = self.pack.map_or(PackFormat::Raw, |pack| pack.index.format);
        Planned {
            layout: layout_over(&self.translator, self.pack, kept, format, bound),
            proven: count,
            unprovable,
            unproven,
            evicted,
        }
    }
}

impl Translations for Store<'_> {
    fn record(&mut self, key: &Digest) -> Option<Arc<Record>> {
        if let Some(record) = self.cached_record(key) {
            return Some(record);
        }
        let Some(stored) = self.stored(key) else {
            self.looked_up(key, false, false);
            return None;
        };
        let mut scratch = self.take_decode_scratch(stored);
        let decoded = DecodedRecord::decode(stored, &self.translator, key, &mut scratch);
        drop(self.finish_decode(key, stored, &decoded, &mut scratch));
        decoded.ok().map(DecodedRecord::into_record)
    }

    fn prepared(
        &mut self,
        key: &Digest,
        pc: u64,
        context: &Digest,
    ) -> Option<Arc<SharedTranslation>> {
        let shared = shared_key(key, pc, context);
        match self.shared.get(&shared) {
            Some(entry) => {
                self.counters.shared_hits += 1;
                Some(Arc::clone(&entry.translation))
            }
            None => {
                self.misses.miss(shared, &mut self.counters);
                None
            }
        }
    }

    fn offer(&mut self, offer: Offer) {
        self.keep_pending(&offer);
        self.lend(offer);
    }

    fn lend(&mut self, offer: Offer) {
        if let Some(mut staged) = StagedOffer::new(offer, self.shared_bound) {
            drop(self.accept_shared(&mut staged));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::machine::test_preparation;
    use crate::translations::{PAGE_BYTES, translator_id};

    /// An offer of `add x0, x0, #1; ret` at `pc`, in a page whose bytes
    /// hash to `key`.
    fn offer(pc: u64, key: Digest) -> Offer {
        let preparation = Arc::new(test_preparation());
        let page = pc & !(PAGE_BYTES as u64 - 1);
        let mut input: Vec<u8> = [0x9100_0400u32, 0xd65f_03c0]
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect();
        preparation.admit(&mut input);
        Offer::compile(key, page, pc, preparation.context(page), input, preparation).unwrap()
    }

    #[test]
    fn decoded_pack_records_share_an_owner_and_the_least_recently_used_goes_first() {
        let translator = translator_id();
        let keys: Vec<Digest> = (0u8..9)
            .map(|index| {
                // Keys alike but for their last byte.
                let mut key = Digest([1; 32]);
                key.0[31] = index;
                key
            })
            .collect();
        let bytes = Layout::new(
            &translator,
            keys.iter()
                .map(|&key| (key, Piece::New(Record::new(translator, key).encode())))
                .collect(),
        )
        .to_vec();
        let mut store = Store::new(translator, Some(&bytes)).0;
        let one = Record::new(translator, keys[0]).retained_bytes();
        drop(store.set_bounds(Bounds {
            decoded: 8 * one,
            ..Bounds::default()
        }));
        assert!(!store.stored(&keys[0]).unwrap().checked);
        let held: Vec<Arc<Record>> = keys[..8]
            .iter()
            .map(|key| store.record(key).unwrap())
            .collect();
        for (key, held) in keys.iter().zip(&held) {
            assert!(Arc::ptr_eq(held, &store.record(key).unwrap()));
            assert_eq!(held.key, *key);
        }
        // The first is used again; the ninth then evicts the second, which
        // its holder still shares.
        assert!(Arc::ptr_eq(&held[0], &store.record(&keys[0]).unwrap()));
        assert_eq!(store.record(&keys[8]).unwrap().key, keys[8]);
        assert!(Arc::ptr_eq(&held[0], &store.record(&keys[0]).unwrap()));
        assert!(Arc::ptr_eq(&held[1], &store.record(&keys[1]).unwrap()));
        assert_eq!(store.counters.loaded, 9);
        assert_eq!(store.counters.decoded, 9);
        assert_eq!(store.counters.repeated, 11);
        // Once nobody holds it, it is decoded again, from bytes already
        // checked whole.
        assert!(store.stored(&keys[1]).unwrap().checked);
        drop(held);
        assert_eq!(store.record(&keys[1]).unwrap().key, keys[1]);
        assert_eq!(store.counters.decoded, 10);
    }

    #[test]
    fn records_nobody_holds_leave_the_held_index() {
        let translator = translator_id();
        let mut cache = DecodedCache::new(0);
        let key = |index: usize| {
            let mut key = Digest([2; 32]);
            key.0[..8].copy_from_slice(&(index as u64).to_le_bytes());
            key
        };
        let kept = Arc::new(Record::new(translator, key(0)));
        assert!(cache.insert(&kept, 1).is_empty());
        for index in 1..=2 * HELD_SWEEP {
            assert!(
                cache
                    .insert(&Arc::new(Record::new(translator, key(index))), 1)
                    .is_empty()
            );
        }
        // Nothing fits the cache itself; only the record still held is found.
        assert!(cache.records.is_empty());
        assert!(cache.held.len() <= HELD_SWEEP + 1);
        assert!(Arc::ptr_eq(&kept, &cache.get(&key(0)).unwrap()));
        assert!(cache.get(&key(1)).is_none());
    }

    #[test]
    fn decoded_cache_declines_oversized_and_foreign_records() {
        let translator = translator_id();
        let key = Digest::of(b"page");
        let mut large = Record::new(translator, key);
        let mut entry = offer(0x1000, key).prove().unwrap();
        entry.code.resize(DECODED_BYTES / 8, 0);
        large.insert(entry);
        let bytes = Layout::new(&translator, vec![(key, Piece::New(large.encode()))]).to_vec();
        let mut store = Store::new(translator, Some(&bytes)).0;
        assert!(store.record(&key).is_some());
        assert!(store.cached_record(&key).is_none());

        let bytes = Layout::new(
            &translator,
            vec![(key, Piece::New(Record::new(translator, key).encode()))],
        )
        .to_vec();
        let foreign = bytes.clone();
        let mut store = Store::new(translator, Some(&bytes)).0;
        let foreign = Pack::parse(&foreign, &translator)
            .unwrap()
            .record(&key)
            .unwrap();
        let mut scratch = DecodeScratch::default();
        let decoded = DecodedRecord::decode(foreign, &translator, &key, &mut scratch);
        assert!(decoded.is_ok());
        assert!(
            store
                .finish_decode(&key, foreign, &decoded, &mut scratch)
                .is_empty()
        );
        assert!(store.cached_record(&key).is_none());
    }

    #[test]
    fn mixed_records_bind_their_original_index_slot_and_survive_rewrite() {
        let translator = translator_id();
        let first = Digest::of(b"compressed");
        let second = Digest::of(b"raw");
        let mut compressed_record = Record::new(translator, first);
        let mut entry = offer(0x1000, first).prove().unwrap();
        entry.code.resize(4096, 0);
        compressed_record.insert(entry);
        let raw = compressed_record.encode();
        let compressed = codec::test_envelope(&raw);
        let records = || {
            vec![
                (first, Piece::New(compressed.clone())),
                (second, Piece::New(Record::new(translator, second).encode())),
            ]
        };
        let bytes = Layout::with_format(&translator, records(), PackFormat::Mixed).to_vec();
        let foreign = bytes.clone();
        let mut store = Store::new(translator, Some(&bytes)).0;
        let mut scratch = DecodeScratch::default();
        let stored = store.stored(&first).unwrap();
        store.finish_decode(&first, stored, &Err(DecodeError::Capacity), &mut scratch);
        assert_eq!(store.counters.invalid, 0);
        assert!(store.stored(&first).is_some());
        let foreign = Pack::parse(&foreign, &translator)
            .unwrap()
            .record(&first)
            .unwrap();
        let decoded = DecodedRecord::decode(foreign, &translator, &first, &mut scratch);
        assert!(decoded.is_ok());
        assert!(
            store
                .finish_decode(&first, stored, &decoded, &mut scratch)
                .is_empty()
        );
        assert_eq!(store.counters.loaded, 0);
        assert!(store.cached_record(&first).is_none());

        let held = store.record(&first).unwrap();
        assert_eq!(*held, compressed_record);
        assert!(Arc::ptr_eq(&held, &store.record(&first).unwrap()));
        assert_eq!(store.record(&second).unwrap().key, second);
        let output = store.plan(u64::MAX, || true).layout.to_vec();
        let pack = Pack::parse(&output, &translator).unwrap();
        assert_eq!(pack.index.format, PackFormat::Mixed);
        assert_eq!(pack.get(&first), Some(compressed.as_slice()));

        let old = Layout::new(&translator, records()).to_vec();
        let mut store = Store::new(translator, Some(&old)).0;
        assert!(store.record(&first).is_none());
        assert_eq!(store.counters.invalid, 1);
        assert_eq!(store.record(&second).unwrap().key, second);
    }

    #[test]
    fn shared_cache_budget_includes_source_and_recovery_owners() {
        let translator = Digest::of(b"translator");
        let key = Digest::of(b"page");
        let first = offer(0x1000, key);
        let context = first.context();
        let code = first.block().native.x86_64_code.len();
        let (mut store, _) = Store::new(translator, None);
        // The native bytes fit, but their source/template/collection owners do not.
        store.share(code);
        store.offer(first);
        assert!(store.prepared(&key, 0x1000, &context).is_none());
        assert!(store.shared_order.is_empty());
    }

    #[test]
    fn shared_cache_budget_charges_emitted_recovery_buffers() {
        let key = Digest::of(b"page");
        let preparation = Arc::new(test_preparation());
        let context = preparation.context(0x1000);
        let mut input: Vec<u8> = core::iter::repeat_n(0xf900_0020u32, 16)
            .chain(core::iter::once(0xd65f_03c0))
            .flat_map(u32::to_le_bytes)
            .collect();
        preparation.admit(&mut input);
        let offered = Offer::compile(key, 0x1000, 0x1000, context, input, preparation).unwrap();
        let code = offered.block().native.x86_64_code.len();
        assert!(!offered.block().fault_recovery.is_empty());
        assert!(
            offered.block().fault_recovery.len()
                * size_of::<stela_compiler::block::FaultRecovery>()
                > code
        );
        let (mut store, _) = Store::new(Digest::of(b"translator"), None);
        store.share(2 * code);
        store.offer(offered);
        assert!(store.prepared(&key, 0x1000, &context).is_none());
        assert!(store.shared_order.is_empty());
    }

    #[test]
    fn an_offer_keeps_no_reserved_source_capacity_against_the_shared_bound() {
        let key = Digest::of(b"page");
        let first = offer(0x1000, key);
        let context = first.context();
        let preparation = Arc::clone(first.preparation());
        let mut input = Vec::with_capacity(2 << 20);
        input.extend_from_slice(first.input());
        let replacement = Offer::compile(key, 0x1000, 0x1000, context, input, preparation).unwrap();
        let (mut store, _) = Store::new(Digest::of(b"translator"), None);
        store.share(1 << 20);
        store.offer(replacement);
        assert!(store.prepared(&key, 0x1000, &context).is_some());
        assert_eq!(store.shared_order.len(), 1);
    }

    #[test]
    fn reducing_shared_cache_budget_releases_kept_entries_and_fifo_capacity() {
        let key = Digest::of(b"page");
        let first = offer(0x1000, key);
        let context = first.context();
        let (mut store, _) = Store::new(Digest::of(b"translator"), None);
        store.share(usize::MAX);
        store.offer(first);
        assert!(store.prepared(&key, 0x1000, &context).is_some());
        store.share(0);
        assert!(store.prepared(&key, 0x1000, &context).is_none());
        assert_eq!(store.shared_order.capacity(), 0);
    }

    #[test]
    fn staged_admission_defers_eviction_and_keeps_a_declined_owner() {
        let translator = Digest::of(b"translator");
        let key = Digest::of(b"page");
        let first = offer(0x1000, key);
        let second = offer(0x1008, key);
        let context = first.context();
        let charge = first.retained_bytes() + size_of::<(SharedKey, SharedEntry)>();
        let mut queue = VecDeque::<SharedKey>::new();
        queue.try_reserve(1).unwrap();
        let bound = charge + queue.capacity() * size_of::<SharedKey>();

        let (mut store, _) = Store::new(translator, None);
        store.share(bound);
        let mut staged = StagedOffer::new(first, bound).unwrap();
        assert!(store.accept_shared(&mut staged).is_none());
        assert!(staged.entry.is_none());
        let old = Arc::downgrade(&store.prepared(&key, 0x1000, &context).unwrap());

        let mut staged = StagedOffer::new(second, bound).unwrap();
        let retired = store.accept_shared(&mut staged).unwrap();
        assert!(store.prepared(&key, 0x1000, &context).is_none());
        assert!(Arc::ptr_eq(&retired, &old.upgrade().unwrap()));
        assert!(store.prepared(&key, 0x1008, &context).is_some());
        assert!(store.shared_usage() <= bound);
        assert_eq!(store.pending(), 0);
        drop(retired);
        assert!(old.upgrade().is_none());

        let mut staged = StagedOffer::new(offer(0x1000, key), bound).unwrap();
        store.share(0);
        assert!(store.accept_shared(&mut staged).is_none());
        assert!(staged.entry.is_some());
        assert!(store.prepared(&key, 0x1000, &context).is_none());
    }

    #[test]
    fn pending_budget_caps_capacity_growth_and_keeps_exact_input() {
        let key = Digest::of(b"page");
        let base = 0x1_0000_4000;
        let mut store = Store::new(translator_id(), None).0;
        for index in 0..4 {
            store.keep_pending(&offer(base + index * 8, key));
        }
        let page = store.pending.values().next().unwrap();
        assert_eq!((page.blocks.len(), page.inputs.len()), (4, 32));
        assert_eq!((page.blocks.capacity(), page.inputs.capacity()), (4, 32));

        // The other pages consume the rest of the budget. There is room
        // for one exact block and its input, but no geometric slack.
        let extra = size_of::<PendingBlock>() + 8;
        store.pending_bytes = MAX_PENDING_BYTES - extra;
        let next = offer(base + 32, key);
        store.keep_pending(&next);
        assert_eq!(store.pending_bytes, MAX_PENDING_BYTES);
        let page = store.pending.values().next().unwrap();
        assert_eq!((page.blocks.capacity(), page.inputs.capacity()), (5, 40));
        assert_eq!(&page.inputs[32..], next.input());

        // A duplicate has no charge. A fresh block cannot replace an old
        // allocation or corrupt the input offsets at the exhausted limit.
        let previous = page.inputs.clone();
        store.keep_pending(&next);
        store.keep_pending(&offer(base + 40, key));
        assert_eq!(store.pending_bytes, MAX_PENDING_BYTES);
        assert_eq!((store.pending(), store.counters.dropped), (5, 1));
        let page = store.pending.values().next().unwrap();
        assert_eq!(page.inputs, previous);
        assert_eq!(
            page.blocks
                .iter()
                .map(|block| block.start)
                .collect::<Vec<_>>(),
            [0, 8, 16, 24, 32]
        );
    }

    #[test]
    fn writable_staging_preserves_proof_with_and_without_shared_admission() {
        let translator = translator_id();
        let key = Digest::of(b"page");
        let first = offer(0x1_0000_4008, key);
        let second = offer(0x1_0000_4010, key);
        let charge = first.retained_bytes() + size_of::<(SharedKey, SharedEntry)>();
        let mut queue = VecDeque::<SharedKey>::new();
        queue.try_reserve(1).unwrap();
        let bound = charge + queue.capacity() * size_of::<SharedKey>();
        let mut store = Store::new(translator, None).0;
        store.share(bound);

        let mut staged = StagedOffer::for_writable(&first, bound).unwrap();
        store.keep_pending(&first);
        assert!(store.accept_shared(&mut staged).is_none());
        let owner = Arc::downgrade(&store.prepared(&key, first.pc(), &first.context()).unwrap());

        let mut staged = StagedOffer::for_writable(&second, bound).unwrap();
        store.keep_pending(&second);
        let retired = store.accept_shared(&mut staged).unwrap();
        assert!(Arc::ptr_eq(&retired, &owner.upgrade().unwrap()));
        assert!(store.prepared(&key, first.pc(), &first.context()).is_none());
        assert!(store.shared_usage() <= bound);
        drop(retired);
        assert!(owner.upgrade().is_none());

        // A live store change may refuse shared publication after staging.
        let mut staged = StagedOffer::for_writable(&second, bound).unwrap();
        store.share(0);
        store.keep_pending(&second);
        assert!(store.accept_shared(&mut staged).is_none());
        assert!(staged.entry.is_some());
        assert!(StagedOffer::for_writable(&first, 0).is_none());
        assert_eq!(store.pending(), 2);
        assert_eq!((store.counters.offered, store.counters.dropped), (2, 0));

        // Reuse and eviction are independent of durable proof input.
        let planned = store.plan(u64::MAX, || true);
        assert_eq!(
            (planned.proven, planned.unprovable, planned.unproven),
            (2, 0, 0)
        );
        let bytes = planned.layout.to_vec();
        let mut next = Store::new(translator, Some(&bytes)).0;
        let record = next.record(&key).unwrap();
        assert!(record.find(8, &first.context()).is_some());
        assert!(record.find(16, &second.context()).is_some());
    }

    #[test]
    fn a_shared_store_hands_offered_blocks_to_other_machines_and_forgets_the_oldest_when_full() {
        let translator = Digest::of(b"translator");
        let key = Digest::of(b"page");
        let (first, second) = (offer(0x1000, key), offer(0x1008, key));
        let code = first.block().native.x86_64_code.len();
        let bytes = first.clone().retained_bytes() + size_of::<(SharedKey, SharedEntry)>();
        let mut queue = VecDeque::<SharedKey>::new();
        queue.try_reserve(1).unwrap();
        let fifo = queue.capacity() * size_of::<SharedKey>();
        let (mut store, _) = Store::new(translator, None);
        store.offer(first.clone());
        // A store no machine shares keeps no native code.
        assert!(store.prepared(&key, 0x1000, &first.context()).is_none());
        store.share(bytes + bytes / 2 + fifo);
        store.offer(first.clone());
        store.offer(first.clone());
        store.offer(second.clone());
        assert!(
            store
                .prepared(&key, 0x1000, &Digest::of(b"other"))
                .is_none()
        );
        assert!(store.prepared(&key, 0x1008, &second.context()).is_some());
        assert!(store.prepared(&key, 0x1000, &first.context()).is_none());
        store.share(4 * bytes + fifo);
        store.offer(first.clone());
        assert_eq!(
            store
                .prepared(&key, 0x1000, &first.context())
                .map(|block| block.block().native.x86_64_code.clone()),
            Some(first.block().native.x86_64_code.clone())
        );
        // Offered again, a block keeps its one place and its bytes.
        store.offer(first.clone());
        assert_eq!(store.shared_bytes, 2 * bytes);
        // Full, the oldest goes first: the second, offered before the first
        // came back.
        let third = offer(0x1010, key);
        let fourth = offer(0x1018, key);
        store.offer(third.clone());
        store.offer(fourth.clone());
        let preparation = Arc::new(test_preparation());
        let pc = 0x1020;
        let page = pc & !(PAGE_BYTES as u64 - 1);
        let fifth = Offer::compile(
            key,
            page,
            pc,
            preparation.context(page),
            0xd65f03c0_u32.to_le_bytes().to_vec(),
            preparation,
        )
        .unwrap();
        assert!(
            fifth.block().native.x86_64_code.len() < code,
            "a genuine shorter emitted block"
        );
        store.offer(fifth.clone());
        assert!(store.prepared(&key, 0x1008, &second.context()).is_none());
        assert!(store.prepared(&key, 0x1000, &first.context()).is_some());
        assert!(store.prepared(&key, 0x1020, &fifth.context()).is_some());
        assert!(store.shared_usage() <= 4 * bytes + fifo);
        // Each distinct block waits for proof once, however often offered.
        assert_eq!(store.pending(), 5);
    }

    #[test]
    fn a_pack_indexes_its_records_by_key_and_ignores_bytes_past_its_end() {
        let translator = Digest::of(b"translator");
        let names: [&[u8]; 3] = [b"b", b"a", b"c"];
        let records: Vec<(Digest, Piece)> = names
            .iter()
            .map(|name| (Digest::of(name), Piece::New(name.repeat(10))))
            .collect();
        let layout = Layout::new(&translator, records);
        let mut bytes = layout.to_vec();
        assert_eq!(bytes.len() as u64, layout.bytes());
        let length = Pack::length(&bytes, &translator).unwrap();
        assert_eq!(length, layout.bytes());
        // A file allocated larger than its pack.
        bytes.extend([0xa5; 100]);
        let pack = Pack::parse(&bytes, &translator).unwrap();
        assert_eq!((pack.records(), pack.bytes() as u64), (3, length));
        assert_eq!(pack.get(&Digest::of(b"c")), Some(&b"cccccccccc"[..]));
        assert_eq!(pack.get(&Digest::of(b"d")), None);
        let mut keys: Vec<Digest> = names.iter().map(|name| Digest::of(name)).collect();
        keys.sort();
        assert_eq!(pack.iter().map(|(key, _)| key).collect::<Vec<_>>(), keys);
        let prefix = prefix_bytes(&bytes, &translator).unwrap();
        assert_eq!(
            prefix,
            HEADER_BYTES + 3 * INDEX_ENTRY_BYTES + CHECKSUM_BYTES
        );
        assert_eq!(
            Pack::parse(&bytes, &Digest::default()).unwrap_err(),
            PackError::Translator
        );
        assert_eq!(
            Pack::parse(&bytes[..prefix - 1], &translator).unwrap_err(),
            PackError::Truncated
        );
        assert_eq!(
            Pack::parse(&bytes[..length as usize - 1], &translator).unwrap_err(),
            PackError::Truncated
        );
        let mut damaged = bytes.clone();
        damaged[HEADER_BYTES + DIGEST_BYTES] ^= 1;
        assert_eq!(
            Pack::parse(&damaged, &translator).unwrap_err(),
            PackError::Checksum
        );
        let mut magic = bytes;
        magic[0] ^= 1;
        assert_eq!(
            Pack::parse(&magic, &translator).unwrap_err(),
            PackError::Magic
        );
    }

    #[test]
    fn the_prefix_indexes_nonresident_records_at_their_file_offsets() {
        let translator = Digest::of(b"translator");
        let first = Digest([1; DIGEST_BYTES]);
        let second = Digest([2; DIGEST_BYTES]);
        let prefix_len = prefix_length(2);
        let low = prefix_len as u64 + 64;
        let high = low + 4096;
        let bytes = Layout::placed(
            &translator,
            vec![
                (first, high, Piece::New(b"first".to_vec())),
                (second, low, Piece::New(b"second".to_vec())),
            ],
            PackFormat::Mixed,
            None,
        )
        .to_vec();
        let prefix = &bytes[..prefix_len];
        let index = PackIndex::parse(prefix, &translator).unwrap();
        assert_eq!(index.length(), high + 5);
        assert_eq!(Pack::length(prefix, &translator).unwrap(), index.length());
        assert_eq!(index.record_range(&first), Some(high..high + 5));
        assert_eq!(index.record_range(&second), Some(low..low + 6));
        assert_eq!(index.record_range(&Digest::default()), None);
        assert_eq!(
            index.iter().collect::<Vec<_>>(),
            [(first, high..high + 5), (second, low..low + 6)]
        );
        assert_eq!(
            PackIndex::parse(&prefix[..prefix_len - 1], &translator).unwrap_err(),
            PackError::Truncated
        );
        assert_eq!(
            Pack::parse(prefix, &translator).unwrap_err(),
            PackError::Truncated
        );
        let resident = Pack::parse(&bytes, &translator).unwrap();
        assert_eq!(resident.get(&first), Some(&b"first"[..]));
        assert_eq!(resident.get(&second), Some(&b"second"[..]));
        assert_eq!(resident.index.format, PackFormat::Mixed);
    }

    #[test]
    fn prefix_lookup_refuses_corrupt_checksums_and_invalid_ranges() {
        let translator = Digest::of(b"translator");
        let key = Digest::of(b"page");
        let bytes = Layout::new(&translator, vec![(key, Piece::New(vec![0]))]).to_vec();
        let prefix_len = prefix_length(1);
        let mut prefix = bytes[..prefix_len].to_vec();
        prefix[prefix_len - 1] ^= 1;
        assert_eq!(
            PackIndex::parse(&prefix, &translator).unwrap_err(),
            PackError::Checksum
        );
        let offset_at = HEADER_BYTES + DIGEST_BYTES;
        let index_end = prefix_len - CHECKSUM_BYTES;
        for (offset, bytes) in [
            (prefix_len as u64 - 1, 1_u64),
            (prefix_len as u64, 0),
            (u64::MAX, 1),
        ] {
            prefix[offset_at..offset_at + 8].copy_from_slice(&offset.to_le_bytes());
            prefix[offset_at + 8..index_end].copy_from_slice(&bytes.to_le_bytes());
            let checksum = Sha256::digest(&prefix[..index_end]);
            prefix[index_end..].copy_from_slice(checksum.as_slice());
            assert_eq!(
                PackIndex::parse(&prefix, &translator).unwrap_err(),
                PackError::Index
            );
        }
    }

    #[test]
    fn blocks_past_the_room_to_prove_them_stay_unproven() {
        let translator = translator_id();
        let (first, second) = (Digest::of(b"first page"), Digest::of(b"second page"));
        let mut store = Store::new(translator, None).0;
        store.offer(offer(0x1_0000_4008, first));
        store.offer(offer(0x1_0000_8010, second));
        let tried = core::cell::Cell::new(0);
        let planned = store.plan(u64::MAX, || {
            tried.set(tried.get() + 1);
            tried.get() == 1
        });
        assert_eq!((planned.proven, planned.unproven), (1, 1));
        assert_eq!(planned.layout.records(), 1);
    }

    #[test]
    fn a_boot_proves_its_blocks_into_the_next_pack_and_the_one_after_reads_them() {
        let translator = translator_id();
        let (first, second) = (Digest::of(b"first page"), Digest::of(b"second page"));
        let mut store = Store::new(translator, None).0;
        store.offer(offer(0x1_0000_4008, first));
        store.offer(offer(0x1_0000_8010, second));
        assert_eq!(store.record(&first), None);
        let planned = store.plan(u64::MAX, || true);
        assert_eq!(
            (planned.proven, planned.unprovable, planned.evicted),
            (2, 0, 0)
        );
        assert_eq!(store.pending(), 0);
        let bytes = planned.layout.to_vec();

        let mut next = Store::new(translator, Some(&bytes)).0;
        let record = next.record(&first).expect("the first page's record");
        let context = offer(0x1_0000_4008, first).context();
        assert!(
            record
                .find(8, &context)
                .unwrap()
                .rebase(0x1_0000_4000, offer(0x1_0000_4008, first).input())
                .is_some()
        );
        assert_eq!(next.counters.loaded, 1);
        // The record this boot used is kept first under a tight bound.
        let one = layout_bytes(1, next.pack().unwrap().get(&first).unwrap().len() as u64);
        let planned = next.plan(one, || true);
        assert_eq!((planned.layout.records(), planned.evicted), (1, 1));
        let kept = planned.layout.to_vec();
        let pack = Pack::parse(&kept, &translator).unwrap();
        assert!(pack.get(&first).is_some() && pack.get(&second).is_none());

        // A damaged pack starts the store empty.
        let (mut empty, error) = Store::new(translator, Some(&bytes[..10]));
        assert_eq!(error, Some(PackError::Truncated));
        assert_eq!(empty.record(&first), None);
        assert_eq!(empty.counters.absent, 1);
    }

    #[test]
    fn new_entries_of_a_stored_page_are_merged_with_its_record() {
        let translator = translator_id();
        let key = Digest::of(b"page");
        let mut store = Store::new(translator, None).0;
        store.offer(offer(0x1_0000_4008, key));
        let bytes = store.plan(u64::MAX, || true).layout.to_vec();
        let mut next = Store::new(translator, Some(&bytes)).0;
        next.offer(offer(0x1_0000_4010, key));
        let merged = next.plan(u64::MAX, || true).layout.to_vec();
        let pack = Pack::parse(&merged, &translator).unwrap();
        let record = Record::decode(pack.get(&key).unwrap(), &translator, &key).unwrap();
        let mut offsets: Vec<u16> = record.entries.iter().map(|entry| entry.offset).collect();
        offsets.sort();
        assert_eq!(offsets, [8, 0x10]);
    }

    /// A pack whose records `layout` lays out, read back by key.
    fn records_of(bytes: &[u8], translator: &Digest) -> BTreeMap<Digest, Vec<u8>> {
        Pack::parse(bytes, translator)
            .unwrap()
            .iter()
            .map(|(key, bytes)| (key, bytes.to_vec()))
            .collect()
    }

    #[test]
    fn a_later_pack_keeps_its_unchanged_records_in_place_and_writes_only_the_rest() {
        let translator = translator_id();
        let keys: Vec<Digest> = (0u8..6).map(|index| Digest::of(&[index])).collect();
        let mut store = Store::new(translator, None).0;
        for (index, key) in keys[..4].iter().enumerate() {
            store.offer(offer(0x1_0000_4008 + 0x1000 * index as u64, *key));
        }
        let first = store.plan(u64::MAX, || true).layout;
        assert_eq!(first.changed(), None);
        let old = first.to_vec();
        // The next boot uses one record and offers a new block in another
        // page and in a page the pack holds.
        let mut next = Store::new(translator, Some(&old)).0;
        assert!(next.record(&keys[1]).is_some());
        next.offer(offer(0x1_0000_9010, keys[4]));
        next.offer(offer(0x1_0000_6010, keys[2]));
        let layout = next.plan(u64::MAX, || true).layout;
        let new = layout.to_vec();
        assert_eq!(new.len() as u64, layout.bytes());
        let changed = layout.changed().expect("an incremental layout").to_vec();
        // Every record outside the changed ranges is where it was, so the
        // bytes the old pack holds there stand.
        let offsets = |bytes: &[u8]| -> BTreeMap<Digest, (usize, usize)> {
            let base = bytes.as_ptr() as usize;
            Pack::parse(bytes, &translator)
                .unwrap()
                .iter()
                .map(|(key, record)| (key, (record.as_ptr() as usize - base, record.len())))
                .collect()
        };
        let (before, after) = (offsets(&old), offsets(&new));
        for (key, (at, length)) in &after {
            let written = changed
                .iter()
                .any(|range| range.start as usize <= *at && *at + length <= range.end as usize);
            if !written {
                assert_eq!(before.get(key), Some(&(*at, *length)));
                assert_eq!(new[*at..at + length], old[*at..at + length]);
            }
        }
        // The index and the two records that gained entries are written;
        // the three others are not.
        assert_eq!(changed.len(), 3);
        assert_eq!(changed[0].start, 0);
        let written: u64 = changed.iter().map(|range| range.end - range.start).sum();
        assert!(written < new.len() as u64 / 2 + changed[0].end);
        let read = records_of(&new, &translator);
        assert_eq!(read.len(), 5);
        for key in [keys[0], keys[1], keys[3]] {
            assert_eq!(read[&key], records_of(&old, &translator)[&key]);
        }
        let record = Record::decode(&read[&keys[2]], &translator, &keys[2]).unwrap();
        assert_eq!(record.entries().len(), 2);
        // Random access reads what the chunks hold.
        let mut buffer = vec![0xa5; 777];
        layout.read_at(100, &mut buffer);
        assert_eq!(buffer[..], new[100..877]);
        layout.read_at(new.len() as u64 - 10, &mut buffer[..20]);
        assert_eq!(buffer[..10], new[new.len() - 10..]);
        assert!(buffer[10..20].iter().all(|&byte| byte == 0));

        // A boot that changes nothing writes nothing.
        let mut idle = Store::new(translator, Some(&new)).0;
        assert!(idle.record(&keys[0]).is_some());
        let same = idle.plan(u64::MAX, || true).layout;
        assert_eq!(same.changed(), Some(&[][..]));
        assert_eq!(same.to_vec(), new);
    }

    #[test]
    fn a_pack_whose_index_outgrows_its_room_is_laid_out_again_with_room_to_grow() {
        let translator = translator_id();
        let keys: Vec<Digest> = (0u8..3).map(|index| Digest::of(&[index])).collect();
        // A pack laid out without room after its index.
        let old = Layout::new(
            &translator,
            vec![(
                keys[0],
                Piece::New(Record::new(translator, keys[0]).encode()),
            )],
        )
        .to_vec();
        let mut store = Store::new(translator, Some(&old)).0;
        store.offer(offer(0x1_0000_4008, keys[1]));
        let layout = store.plan(u64::MAX, || true).layout;
        assert_eq!(layout.changed(), None);
        let roomy = layout.to_vec();
        let pack = Pack::parse(&roomy, &translator).unwrap();
        assert_eq!(pack.records(), 2);
        // Its records start past room for many more index entries.
        let first = pack
            .iter()
            .map(|(_, bytes)| bytes.as_ptr() as usize - roomy.as_ptr() as usize)
            .min()
            .unwrap();
        assert!(first >= prefix_length(2 + MIN_INDEX_SLACK));
        let mut next = Store::new(translator, Some(&roomy)).0;
        next.offer(offer(0x1_0000_5008, keys[2]));
        let grown = next.plan(u64::MAX, || true).layout;
        assert!(grown.changed().is_some());
        assert_eq!(records_of(&grown.to_vec(), &translator).len(), 3);
        // Under a bound without room for it, the records follow the index.
        let tight = Store::new(translator, Some(&old))
            .0
            .plan(old.len() as u64, || true)
            .layout;
        assert_eq!(tight.to_vec(), old);
    }

    /// A checksummed index whose entries share bytes uses more than their
    /// span; the pack is laid out again instead of computing negative waste.
    #[test]
    fn kept_records_that_overlap_are_laid_out_again() {
        let translator = translator_id();
        let keys: Vec<Digest> = (0u8..2).map(|index| Digest::of(&[index])).collect();
        let mut old = Layout::new(
            &translator,
            keys.iter()
                .map(|key| (*key, Piece::New(Record::new(translator, *key).encode())))
                .collect(),
        )
        .to_vec();
        // Point the second entry at the first's bytes and checksum the index.
        let (_, offset, bytes) = entry(&old, 0);
        let second = HEADER_BYTES + INDEX_ENTRY_BYTES + DIGEST_BYTES;
        old[second..second + 8].copy_from_slice(&offset.to_le_bytes());
        old[second + 8..second + 16].copy_from_slice(&bytes.to_le_bytes());
        let index_end = HEADER_BYTES + 2 * INDEX_ENTRY_BYTES;
        let checksum = Sha256::digest(&old[..index_end]);
        old[index_end..index_end + CHECKSUM_BYTES].copy_from_slice(checksum.as_slice());
        let pack = Pack::parse(&old, &translator).unwrap();
        let records = pack
            .iter()
            .map(|(key, bytes)| (key, Piece::Kept(bytes)))
            .collect();
        let layout = layout_over(
            &translator,
            Some(pack),
            records,
            pack.index.format,
            u64::MAX,
        );
        assert_eq!(layout.changed(), None);
    }

    #[test]
    fn blocks_proven_in_batches_during_the_run_reach_the_next_pack() {
        let translator = translator_id();
        let (first, second) = (Digest::of(b"first page"), Digest::of(b"second page"));
        let mut store = Store::new(translator, None).0;
        store.offer(offer(0x1_0000_4008, first));
        store.offer(offer(0x1_0000_4010, first));
        store.offer(offer(0x1_0000_8010, second));
        let batch = store.take_batch(1).unwrap();
        assert_eq!(batch.blocks(), 2);
        assert_eq!(store.pending(), 1);
        store.fold(batch.prove());
        assert_eq!(store.counters.proven, 2);
        // An offer already proven waits no more, and frees nothing twice.
        store.offer(offer(0x1_0000_4008, first));
        assert_eq!(store.pending(), 1);
        let planned = store.plan(u64::MAX, || true);
        assert_eq!((planned.proven, planned.unproven), (1, 0));
        let bytes = planned.layout.to_vec();
        let mut next = Store::new(translator, Some(&bytes)).0;
        let context = offer(0x1_0000_4008, first).context();
        let record = next.record(&first).unwrap();
        assert!(record.find(8, &context).is_some() && record.find(16, &context).is_some());
        assert!(next.record(&second).unwrap().find(16, &context).is_some());

        // Past the proven bound a batch's entries are counted, not kept.
        let mut full = Store::new(translator, None).0;
        drop(full.set_bounds(Bounds {
            proven: 0,
            ..Bounds::default()
        }));
        full.offer(offer(0x1_0000_4008, first));
        assert!(full.take_batch(1).is_none());
        drop(full.set_bounds(Bounds {
            proven: 1,
            ..Bounds::default()
        }));
        let batch = full.take_batch(1).unwrap();
        full.fold(batch.prove());
        assert_eq!((full.counters.proven, full.counters.unkept), (1, 1));
    }

    /// Once a fold left an entry out for the proven bound, a batch proven
    /// then would be left out too: none is taken, and the pages wait for
    /// the stop's proof, which keeps them. Before, every batch was proven
    /// and dropped with its pages for the rest of the run.
    #[test]
    fn a_store_short_of_an_entry_proves_no_batch_until_its_bound_grows() {
        let translator = translator_id();
        let keys: Vec<Digest> = (0u8..4).map(|index| Digest::of(&[index])).collect();
        let mut store = Store::new(translator, None).0;
        drop(store.set_bounds(Bounds {
            proven: 1,
            ..Bounds::default()
        }));
        for (index, key) in keys.iter().enumerate() {
            store.offer(offer(0x1_0000_4008 + 0x4000 * index as u64, *key));
        }
        assert_eq!(store.pending(), 4);
        let batch = store.take_batch(1).unwrap();
        assert_eq!(batch.blocks(), 1);
        store.fold(batch.prove());
        assert_eq!((store.counters.proven, store.counters.unkept), (1, 1));
        // Room is left under the bound, but not for an entry: no batch.
        for _ in 0..8 {
            assert!(store.take_batch(1).is_none());
        }
        assert_eq!((store.counters.proven, store.pending()), (1, 3));
        // A smaller bound changes nothing; a larger one proves again.
        drop(store.set_bounds(Bounds {
            proven: 0,
            ..Bounds::default()
        }));
        assert!(store.take_batch(1).is_none());
        drop(store.set_bounds(Bounds::default()));
        let batch = store.take_batch(1).unwrap();
        store.fold(batch.prove());
        assert_eq!((store.counters.proven, store.counters.unkept), (2, 1));
        assert_eq!(store.pending(), 2);
        // The stop proves the two that still wait, and the next pack holds
        // them with the one proven during the run.
        let planned = store.plan(u64::MAX, || true);
        assert_eq!((planned.proven, planned.unproven), (2, 0));
        let bytes = planned.layout.to_vec();
        let mut next = Store::new(translator, Some(&bytes)).0;
        // All but the page the first, unkept batch took.
        let context = offer(0x1_0000_4008, keys[0]).context();
        let kept = keys
            .iter()
            .filter(|key| {
                next.record(key)
                    .is_some_and(|record| record.find(8, &context).is_some())
            })
            .count();
        assert_eq!(kept, 3);
    }

    #[test]
    fn a_larger_pending_bound_keeps_offers_a_smaller_one_drops() {
        let translator = translator_id();
        let key = Digest::of(b"page");
        let offers: Vec<Offer> = (0..64)
            .map(|index| offer(0x1_0000_4000 + 8 * index, key))
            .collect();
        let mut small = Store::new(translator, None).0;
        drop(small.set_bounds(Bounds {
            pending: 256,
            ..Bounds::default()
        }));
        let mut large = Store::new(translator, None).0;
        for offer in &offers {
            small.keep_pending(offer);
            large.keep_pending(offer);
        }
        assert!(small.counters.dropped > 0);
        assert_eq!((large.pending(), large.counters.dropped), (64, 0));
    }

    #[test]
    fn shared_misses_are_counted_by_reason() {
        let translator = translator_id();
        let key = Digest::of(b"page");
        let first = offer(0x1_0000_4008, key);
        let context = first.context();
        let other = Digest::of(b"other context");
        let mut store = Store::new(translator, None).0;
        store.share(usize::MAX);
        // Never offered, then compiling elsewhere.
        assert!(store.prepared(&key, first.pc(), &context).is_none());
        assert!(store.prepared(&key, first.pc(), &context).is_none());
        assert_eq!(
            (
                store.counters.shared_absent,
                store.counters.shared_in_flight
            ),
            (1, 1)
        );
        store.offer(first.clone());
        assert!(store.prepared(&key, first.pc(), &context).is_some());
        // Offered in another context.
        assert!(store.prepared(&key, first.pc(), &other).is_none());
        assert_eq!(store.counters.shared_other_context, 1);
        // Evicted.
        store.share(0);
        assert!(store.prepared(&key, first.pc(), &context).is_none());
        assert_eq!(
            (store.counters.shared_hits, store.counters.shared_evicted),
            (1, 1)
        );
    }

    #[test]
    fn parts_of_a_split_store_plan_the_pack_one_store_plans() {
        let translator = translator_id();
        let keys: Vec<Digest> = (0u8..4).map(|index| Digest::of(&[index])).collect();
        let offers: Vec<Offer> = keys
            .iter()
            .enumerate()
            .map(|(index, key)| offer(0x1_0000_4008 + 0x1000 * index as u64, *key))
            .collect();
        let mut whole = Store::new(translator, None).0;
        let mut parts: Vec<Store> = (0..2).map(|_| Store::over(translator, None)).collect();
        for offer in &offers {
            whole.offer(offer.clone());
            parts[usize::from(offer.key().0[0] & 1)].offer(offer.clone());
        }
        let mut merged = parts.remove(0);
        for part in parts {
            merged.absorb(part);
        }
        assert_eq!(merged.pending(), whole.pending());
        assert_eq!(merged.counters.offered, whole.counters.offered);
        assert_eq!(
            merged.plan(u64::MAX, || true).layout.to_vec(),
            whole.plan(u64::MAX, || true).layout.to_vec()
        );
    }
}
