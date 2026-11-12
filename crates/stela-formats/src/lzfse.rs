//! LZFSE and LZVN decoding in `no_std`, for the compressed payloads of
//! Image4 files (kernel collections, DeviceTrees, firmware).
//!
//! A stream is a sequence of blocks, each starting with a magic: `bvx-`
//! (stored bytes), `bvxn` (LZVN), `bvx1` (LZFSE with plain frequency
//! tables), `bvx2` (LZFSE with packed frequency tables), and `bvx$` ending
//! the stream. LZFSE blocks carry literals and literal/match/distance
//! triples, each coded with finite-state entropy and read backwards from the
//! end of their payload. This follows Apple's reference decoder
//! ([lzfse](https://github.com/lzfse/lzfse) commit `e634ca5`,
//! `lzfse_decode_base.c`, `lzfse_fse.[ch]`, `lzvn_decode_base.c` and
//! `lzfse_internal.h`), decoding a whole stream into one buffer.
//!
//! Decoding is stricter than the reference where the reference only bounds
//! writes to its output buffer: every block must produce exactly its
//! declared raw size, every literal and match must stay inside its block's
//! literals and the bytes decoded so far, and nothing may follow `bvx$`.
use alloc::vec::Vec;
use stela_contracts::bytes::{get_le16, get_le32, get_le64};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LzfseError {
    /// The stream ends inside a block or before `bvx$`.
    Truncated,
    Malformed(&'static str),
    /// Decoding would produce more than the caller's limit.
    Limit,
}

impl core::fmt::Display for LzfseError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Truncated => write!(f, "truncated LZFSE stream"),
            Self::Malformed(what) => write!(f, "malformed LZFSE stream: {what}"),
            Self::Limit => write!(f, "LZFSE output exceeds the configured limit"),
        }
    }
}

impl core::error::Error for LzfseError {}

pub type Result<T> = core::result::Result<T, LzfseError>;

const END_OF_STREAM: u32 = u32::from_le_bytes(*b"bvx$");
const UNCOMPRESSED: u32 = u32::from_le_bytes(*b"bvx-");
const COMPRESSED_V1: u32 = u32::from_le_bytes(*b"bvx1");
const COMPRESSED_V2: u32 = u32::from_le_bytes(*b"bvx2");
const COMPRESSED_LZVN: u32 = u32::from_le_bytes(*b"bvxn");

/// Whether `bytes` starts with an LZFSE block magic.
pub fn is_lzfse(bytes: &[u8]) -> bool {
    bytes.get(..4).is_some_and(|magic| {
        [UNCOMPRESSED, COMPRESSED_V1, COMPRESSED_V2, COMPRESSED_LZVN]
            .contains(&u32::from_le_bytes(magic.try_into().unwrap()))
    })
}

const L_SYMBOLS: usize = 20;
const M_SYMBOLS: usize = 20;
const D_SYMBOLS: usize = 64;
const LITERAL_SYMBOLS: usize = 256;
const L_STATES: usize = 64;
const M_STATES: usize = 64;
const D_STATES: usize = 256;
const LITERAL_STATES: usize = 1024;
const MATCHES_PER_BLOCK: u32 = 10_000;
const LITERALS_PER_BLOCK: u32 = 4 * MATCHES_PER_BLOCK;
/// `sizeof(lzfse_compressed_block_header_v1)`: 770 bytes of fields padded to
/// the structure's 4-byte alignment.
const V1_HEADER_BYTES: usize = 772;
/// `offsetof(lzfse_compressed_block_header_v2, freq)`.
const V2_FREQ_OFFSET: usize = 32;

const L_EXTRA_BITS: [u8; L_SYMBOLS] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 5, 8];
const L_BASE_VALUE: [i32; L_SYMBOLS] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 28, 60,
];
const M_EXTRA_BITS: [u8; M_SYMBOLS] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 5, 8, 11];
const M_BASE_VALUE: [i32; M_SYMBOLS] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 24, 56, 312,
];
const D_EXTRA_BITS: [u8; D_SYMBOLS] = [
    0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7,
    8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14,
    14, 14, 15, 15, 15, 15,
];
const D_BASE_VALUE: [i32; D_SYMBOLS] = [
    0, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 28, 36, 44, 52, 60, 76, 92, 108, 124, 156, 188, 220,
    252, 316, 380, 444, 508, 636, 764, 892, 1020, 1276, 1532, 1788, 2044, 2556, 3068, 3580, 4092,
    5116, 6140, 7164, 8188, 10236, 12284, 14332, 16380, 20476, 24572, 28668, 32764, 40956, 49148,
    57340, 65532, 81916, 98300, 114684, 131068, 163836, 196604, 229372,
];

fn u16_at(bytes: &[u8], at: usize) -> Result<u16> {
    get_le16(bytes, at).ok_or(LzfseError::Truncated)
}

fn u32_at(bytes: &[u8], at: usize) -> Result<u32> {
    get_le32(bytes, at).ok_or(LzfseError::Truncated)
}

fn u64_at(bytes: &[u8], at: usize) -> Result<u64> {
    get_le64(bytes, at).ok_or(LzfseError::Truncated)
}

/// The decoded output, bounded by the caller's limit.
struct Output<'a> {
    bytes: &'a mut Vec<u8>,
    start: usize,
    limit: usize,
    /// Where the current block must end.
    block_end: usize,
}

impl<'a> Output<'a> {
    fn new(bytes: &'a mut Vec<u8>, limit: usize) -> Self {
        let start = bytes.len();
        Self {
            bytes,
            start,
            limit,
            block_end: start,
        }
    }

    fn begin_block(&mut self, raw_bytes: u32) -> Result<()> {
        let end = self
            .bytes
            .len()
            .checked_add(raw_bytes as usize)
            .ok_or(LzfseError::Limit)?;
        if end - self.start > self.limit {
            return Err(LzfseError::Limit);
        }
        self.bytes.reserve(raw_bytes as usize);
        self.block_end = end;
        Ok(())
    }

    fn room(&self, bytes: usize) -> Result<()> {
        if self.bytes.len() + bytes > self.block_end {
            return Err(LzfseError::Malformed("block decodes past its raw size"));
        }
        Ok(())
    }

    fn literals(&mut self, literals: &[u8]) -> Result<()> {
        self.room(literals.len())?;
        self.bytes.extend_from_slice(literals);
        Ok(())
    }

    /// Copy a match from the output, repeating its pattern on overlap.
    fn copy_match(&mut self, distance: usize, length: usize) -> Result<()> {
        let start = self
            .bytes
            .len()
            .checked_sub(distance)
            .filter(|&at| distance != 0 && at >= self.start)
            .ok_or(LzfseError::Malformed("match distance"))?;
        self.room(length)?;
        if distance >= length {
            self.bytes.extend_from_within(start..start + length);
        } else if distance == 1 {
            self.bytes
                .resize(self.bytes.len() + length, self.bytes[start]);
        } else {
            // Each copy makes twice as much of the repeated pattern available.
            let mut copied = 0;
            let mut available = distance;
            while copied < length {
                let count = available.min(length - copied);
                self.bytes.extend_from_within(start..start + count);
                copied += count;
                available += count;
            }
        }
        Ok(())
    }

    fn end_block(&self) -> Result<()> {
        if self.bytes.len() != self.block_end {
            return Err(LzfseError::Malformed("block decodes short of its raw size"));
        }
        Ok(())
    }
}

/// The byte ranges advertised by a block, before entropy validation.
struct Frame {
    raw_bytes: u32,
    payload: usize,
    lmd: usize,
    end: usize,
}

impl Frame {
    fn read(src: &[u8], at: usize, magic: u32) -> Result<Self> {
        Self::from_header(&src[at..], at, magic, src.len())
    }

    fn from_header(block: &[u8], at: usize, magic: u32, total: usize) -> Result<Self> {
        let (header, literal, lmd_bytes) = match magic {
            UNCOMPRESSED => (8, None, 0),
            COMPRESSED_LZVN => (12, Some(u32_at(block, 8)?), 0),
            COMPRESSED_V1 => (
                V1_HEADER_BYTES,
                Some(u32_at(block, 20)?),
                u32_at(block, 24)?,
            ),
            COMPRESSED_V2 => {
                let header = field(u64_at(block, 24)?, 0, 32) as usize;
                if header < V2_FREQ_OFFSET {
                    return Err(LzfseError::Malformed("bvx2 header size"));
                }
                (
                    header,
                    Some(field(u64_at(block, 8)?, 20, 20)),
                    field(u64_at(block, 16)?, 40, 20),
                )
            }
            _ => return Err(LzfseError::Malformed("unknown block magic")),
        };
        let raw_bytes = u32_at(block, 4)?;
        let payload = at.checked_add(header).ok_or(LzfseError::Truncated)?;
        let lmd = payload
            .checked_add(literal.unwrap_or(raw_bytes) as usize)
            .ok_or(LzfseError::Truncated)?;
        let end = lmd
            .checked_add(lmd_bytes as usize)
            .filter(|&end| end <= total)
            .ok_or(LzfseError::Truncated)?;
        Ok(Self {
            raw_bytes,
            payload,
            lmd,
            end,
        })
    }
}

/// Size a multi-block output without inspecting its frequency tables or
/// decoding any data. A failed hint leaves validation to the decoder.
fn framed_output_size(src: &[u8], limit: usize) -> Option<usize> {
    let mut at = 0;
    let mut multiple = false;
    let mut total = 0usize;
    loop {
        let magic = u32_at(src, at).ok()?;
        if magic == END_OF_STREAM {
            return (at + 4 == src.len() && multiple).then_some(total);
        }
        let frame = Frame::read(src, at, magic).ok()?;
        total = total.checked_add(frame.raw_bytes as usize)?;
        if total > limit {
            return None;
        }
        multiple |= at != 0;
        at = frame.end;
    }
}

/// Decode a whole LZFSE stream, producing at most `limit` bytes.
pub fn decode(src: &[u8], limit: usize) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    Decoder::default().append(src, limit, &mut bytes)?;
    Ok(bytes)
}

impl Decoder {
    /// Append one complete stream. Its matches can use only this stream's output.
    pub(crate) fn append(&mut self, src: &[u8], limit: usize, bytes: &mut Vec<u8>) -> Result<()> {
        let mut out = Output::new(bytes, limit);
        let result = decode_stream(src, &mut out, self);
        if result.is_err() {
            out.bytes.truncate(out.start);
        }
        result
    }
}

fn decode_stream(src: &[u8], out: &mut Output<'_>, decoder: &mut Decoder) -> Result<()> {
    // A heap without in-place growth copies all preceding output on each
    // resize. Reserve once when all block ranges fit the stream and limit.
    if out.bytes.capacity() - out.start < out.limit
        && let Some(bytes) = framed_output_size(src, out.limit)
    {
        let _ = out.bytes.try_reserve_exact(bytes);
    }
    let mut at = 0;
    loop {
        let magic = u32_at(src, at)?;
        if magic == END_OF_STREAM {
            if at + 4 != src.len() {
                return Err(LzfseError::Malformed("bytes follow the end of stream"));
            }
            return Ok(());
        }
        let frame = Frame::read(src, at, magic)?;
        decode_frame(src, at, magic, &frame, out, decoder)?;
        at = frame.end;
    }
}

fn decode_frame(
    src: &[u8],
    at: usize,
    magic: u32,
    frame: &Frame,
    out: &mut Output<'_>,
    decoder: &mut Decoder,
) -> Result<()> {
    match magic {
        UNCOMPRESSED => {
            out.begin_block(frame.raw_bytes)?;
            out.literals(&src[frame.payload..frame.end])?;
        }
        COMPRESSED_LZVN => {
            out.begin_block(frame.raw_bytes)?;
            lzvn(&src[frame.payload..frame.end], out, true)?;
        }
        COMPRESSED_V1 | COMPRESSED_V2 => {
            let encoded = &src[at..frame.payload];
            let header = if magic == COMPRESSED_V1 {
                Header::v1(encoded)?
            } else {
                Header::v2(encoded)?
            };
            header.check()?;
            out.begin_block(frame.raw_bytes)?;
            block(src, frame, &header, out, decoder)?;
        }
        _ => unreachable!("validated by Frame::from_header"),
    }
    out.end_block()
}

/// Read one stream in order, retaining only a bounded encoded frame.
/// The input closure fills each requested slice exactly, with the next bytes.
/// The caller supplies a checked maximum output size for the reservation.
///
/// `Ok(None)` declines this path when a frame exceeds the staging envelope
/// or the output allocation cannot be made. The ordinary slice decoder can
/// still handle streams outside this staging envelope.
pub fn decode_bounded<E: From<LzfseError>>(
    read: impl FnMut(&mut [u8]) -> core::result::Result<(), E>,
    remaining: usize,
    output_limit: usize,
) -> core::result::Result<Option<Vec<u8>>, E> {
    decode_bounded_observed(read, remaining, output_limit, |_| {})
}

/// [`decode_bounded`], handing `decoded` each frame's output in order once
/// it is final, so a caller can hash the output while it is still in the
/// cache instead of in a second pass. A stream this path declines may
/// have handed out some frames already.
pub fn decode_bounded_observed<E: From<LzfseError>>(
    mut read: impl FnMut(&mut [u8]) -> core::result::Result<(), E>,
    mut remaining: usize,
    output_limit: usize,
    mut decoded: impl FnMut(&[u8]),
) -> core::result::Result<Option<Vec<u8>>, E> {
    const MAX_FRAME_BYTES: usize = 1 << 20;
    let mut output = Vec::new();
    let mut input = Vec::new();
    let mut decoder = Decoder::default();
    let mut out = Output::new(&mut output, output_limit);
    loop {
        if remaining < 4 {
            return Err(LzfseError::Truncated.into());
        }
        input.resize(4, 0);
        read(&mut input)?;
        let magic = u32_at(&input, 0)?;
        let header = match magic {
            END_OF_STREAM => {
                if remaining != 4 {
                    return Err(LzfseError::Malformed("bytes follow the end of stream").into());
                }
                return Ok(Some(output));
            }
            UNCOMPRESSED => 8,
            COMPRESSED_LZVN => 12,
            COMPRESSED_V1 => 28,
            COMPRESSED_V2 => V2_FREQ_OFFSET,
            _ => return Err(LzfseError::Malformed("unknown block magic").into()),
        };
        if remaining < header {
            return Err(LzfseError::Truncated.into());
        }
        input.resize(header, 0);
        read(&mut input[4..])?;
        let frame = Frame::from_header(&input, 0, magic, remaining)?;
        if frame.end > MAX_FRAME_BYTES {
            return Ok(None);
        }
        input.resize(frame.end, 0);
        read(&mut input[header..])?;
        // Defer reserving the complete output until the first frame has
        // passed its size and input checks.
        if out.bytes.capacity() == 0 && out.bytes.try_reserve_exact(output_limit).is_err() {
            return Ok(None);
        }
        let before = out.bytes.len();
        decode_frame(&input, 0, magic, &frame, &mut out, &mut decoder)?;
        decoded(&out.bytes[before..]);
        remaining -= frame.end;
    }
}

/// `lzfse_compressed_block_header_v1`, the form both LZFSE block kinds
/// decode to.
struct Header {
    n_literals: u32,
    n_matches: u32,
    literal_bits: i32,
    literal_state: [u16; 4],
    lmd_bits: i32,
    l_state: u16,
    m_state: u16,
    d_state: u16,
    l_freq: [u16; L_SYMBOLS],
    m_freq: [u16; M_SYMBOLS],
    d_freq: [u16; D_SYMBOLS],
    literal_freq: [u16; LITERAL_SYMBOLS],
}

/// One frequency of a packed `bvx2` table from the low bits of `bits`, and
/// how many bits it used (`lzfse_decode_v1_freq_value`).
fn packed_frequency(bits: u32) -> (u16, u32) {
    const NBITS: [u8; 32] = [
        2, 3, 2, 5, 2, 3, 2, 8, 2, 3, 2, 5, 2, 3, 2, 14, 2, 3, 2, 5, 2, 3, 2, 8, 2, 3, 2, 5, 2, 3,
        2, 14,
    ];
    const VALUE: [u8; 32] = [
        0, 2, 1, 4, 0, 3, 1, 0, 0, 2, 1, 5, 0, 3, 1, 0, 0, 2, 1, 6, 0, 3, 1, 0, 0, 2, 1, 7, 0, 3,
        1, 0,
    ];
    let low = (bits & 31) as usize;
    match NBITS[low] {
        8 => (8 + ((bits >> 4) & 0xf) as u16, 8),
        14 => (24 + ((bits >> 4) & 0x3ff) as u16, 14),
        n => (u16::from(VALUE[low]), u32::from(n)),
    }
}

fn field(value: u64, offset: u32, bits: u32) -> u32 {
    ((value >> offset) & ((1u64 << bits) - 1)) as u32
}

impl Header {
    fn v1(block: &[u8]) -> Result<Self> {
        if block.len() < V1_HEADER_BYTES {
            return Err(LzfseError::Truncated);
        }
        let mut header = Self {
            n_literals: u32_at(block, 12)?,
            n_matches: u32_at(block, 16)?,
            literal_bits: u32_at(block, 28)? as i32,
            literal_state: [
                u16_at(block, 32)?,
                u16_at(block, 34)?,
                u16_at(block, 36)?,
                u16_at(block, 38)?,
            ],
            lmd_bits: u32_at(block, 40)? as i32,
            l_state: u16_at(block, 44)?,
            m_state: u16_at(block, 46)?,
            d_state: u16_at(block, 48)?,
            l_freq: [0; L_SYMBOLS],
            m_freq: [0; M_SYMBOLS],
            d_freq: [0; D_SYMBOLS],
            literal_freq: [0; LITERAL_SYMBOLS],
        };
        let mut at = 50;
        for frequency in header
            .l_freq
            .iter_mut()
            .chain(&mut header.m_freq)
            .chain(&mut header.d_freq)
            .chain(&mut header.literal_freq)
        {
            *frequency = u16_at(block, at)?;
            at += 2;
        }
        Ok(header)
    }

    /// The header of a `bvx2` block, bounded by its validated frame.
    fn v2(block: &[u8]) -> Result<Self> {
        let v0 = u64_at(block, 8)?;
        let v1 = u64_at(block, 16)?;
        let v2 = u64_at(block, 24)?;
        let tables = &block[V2_FREQ_OFFSET..];
        let mut header = Self {
            n_literals: field(v0, 0, 20),
            n_matches: field(v0, 40, 20),
            literal_bits: field(v0, 60, 3) as i32 - 7,
            literal_state: [
                field(v1, 0, 10) as u16,
                field(v1, 10, 10) as u16,
                field(v1, 20, 10) as u16,
                field(v1, 30, 10) as u16,
            ],
            lmd_bits: field(v1, 60, 3) as i32 - 7,
            l_state: field(v2, 32, 10) as u16,
            m_state: field(v2, 42, 10) as u16,
            d_state: field(v2, 52, 10) as u16,
            l_freq: [0; L_SYMBOLS],
            m_freq: [0; M_SYMBOLS],
            d_freq: [0; D_SYMBOLS],
            literal_freq: [0; LITERAL_SYMBOLS],
        };
        // An empty table area means every frequency is zero.
        if !tables.is_empty() {
            let mut source = tables.iter();
            let mut accum = 0u32;
            let mut accum_bits = 0u32;
            for frequency in header
                .l_freq
                .iter_mut()
                .chain(&mut header.m_freq)
                .chain(&mut header.d_freq)
                .chain(&mut header.literal_freq)
            {
                while accum_bits + 8 <= 32 {
                    let Some(&byte) = source.next() else { break };
                    accum |= u32::from(byte) << accum_bits;
                    accum_bits += 8;
                }
                let (value, bits) = packed_frequency(accum);
                if bits > accum_bits {
                    return Err(LzfseError::Malformed("bvx2 frequency tables"));
                }
                *frequency = value;
                accum >>= bits;
                accum_bits -= bits;
            }
            if accum_bits >= 8 || source.next().is_some() {
                return Err(LzfseError::Malformed("bvx2 frequency tables"));
            }
        }
        Ok(header)
    }

    /// `lzfse_check_block_header_v1`.
    fn check(&self) -> Result<()> {
        let sums = |freq: &[u16], states: usize| {
            freq.iter().map(|&f| usize::from(f)).sum::<usize>() <= states
        };
        if self.n_literals > LITERALS_PER_BLOCK
            || self.n_matches > MATCHES_PER_BLOCK
            || self
                .literal_state
                .iter()
                .any(|&s| usize::from(s) >= LITERAL_STATES)
            || usize::from(self.l_state) >= L_STATES
            || usize::from(self.m_state) >= M_STATES
            || usize::from(self.d_state) >= D_STATES
            || !sums(&self.l_freq, L_STATES)
            || !sums(&self.m_freq, M_STATES)
            || !sums(&self.d_freq, D_STATES)
            || !sums(&self.literal_freq, LITERAL_STATES)
        {
            return Err(LzfseError::Malformed("LZFSE block header"));
        }
        Ok(())
    }
}

/// A finite-state-entropy bit stream read backwards from the end of its
/// payload; the accumulator holds 56 to 63 bits after each refill
/// (`fse_in_stream64`).
struct BitStream<'a> {
    src: &'a [u8],
    /// The lowest byte not yet loaded.
    position: usize,
    /// The lowest byte the stream may load.
    start: usize,
    accum: u64,
    accum_bits: u32,
}

impl<'a> BitStream<'a> {
    /// `fse_in_checked_init64`: `bits` (in -7..=0) is the number of unused
    /// bits in the final byte, as a negative count.
    fn new(src: &'a [u8], start: usize, end: usize, bits: i32) -> Result<Self> {
        let (take, total) = match bits {
            0 => (7usize, 56i32),
            _ => (8, 64 + bits),
        };
        if !(56..64).contains(&total) || end < start + take {
            return Err(LzfseError::Malformed("FSE stream start"));
        }
        let position = end - take;
        let mut word = [0u8; 8];
        word[..take].copy_from_slice(&src[position..end]);
        let accum = u64::from_le_bytes(word);
        let accum_bits = total as u32;
        if accum >> accum_bits != 0 {
            return Err(LzfseError::Malformed("FSE stream start"));
        }
        Ok(Self {
            src,
            position,
            start,
            accum,
            accum_bits,
        })
    }

    /// `fse_in_checked_flush64`.
    fn refill(&mut self) -> Result<()> {
        let bits = (63 - self.accum_bits) & !7;
        let bytes = (bits / 8) as usize;
        if self.position < self.start + bytes {
            return Err(LzfseError::Malformed("FSE stream underflow"));
        }
        if bytes == 0 {
            return Ok(());
        }
        self.position -= bytes;
        // Initialization loaded at least seven bytes and we moved back at
        // least one. Eight bytes from the new position therefore fit within
        // the stream's end. Only the newly loaded low bytes enter the buffer.
        let word = u64::from_le_bytes(
            self.src[self.position..self.position + 8]
                .try_into()
                .unwrap(),
        );
        self.accum = (self.accum << bits) | (word & ((1u64 << bits) - 1));
        self.accum_bits += bits;
        Ok(())
    }

    /// `fse_in_pull64`. Each decode group starts with at least 56 bits
    /// after refilling. A group of four literals consumes at most 40 bits,
    /// and an L, M, D triple at most (6 + 8) + (6 + 11) + (8 + 15) bits.
    /// Only the low `accum_bits` bits are live between pulls and refills.
    fn pull(&mut self, bits: u32) -> u64 {
        debug_assert!(bits <= self.accum_bits);
        self.accum_bits -= bits;
        (self.accum >> self.accum_bits) & ((1u64 << bits) - 1)
    }
}

/// The shift `k` and threshold `j0` for a symbol of frequency `f` among
/// `states` states: states `j < j0` read `k` bits, the rest `k - 1`.
fn spread(states: usize, f: usize) -> (u32, usize) {
    let k = (f as u32).leading_zeros() - (states as u32).leading_zeros();
    (k, ((2 * states) >> k) - f)
}

/// `fse_decoder_entry`: one literal decoder state.
#[derive(Clone, Copy, Default)]
struct LiteralEntry {
    bits: u8,
    symbol: u8,
    delta: i16,
}

/// `fse_init_decoder_table`.
fn fill_literal_table(table: &mut Vec<LiteralEntry>, freq: &[u16; LITERAL_SYMBOLS]) {
    table.clear();
    table.reserve_exact(LITERAL_STATES);
    for (symbol, &f) in freq.iter().enumerate() {
        let f = usize::from(f);
        if f == 0 {
            continue;
        }
        let (k, j0) = spread(LITERAL_STATES, f);
        for j in 0..f {
            table.push(if j < j0 {
                LiteralEntry {
                    bits: k as u8,
                    symbol: symbol as u8,
                    delta: (((f + j) << k) as i16) - LITERAL_STATES as i16,
                }
            } else {
                LiteralEntry {
                    bits: (k - 1) as u8,
                    symbol: symbol as u8,
                    delta: ((j - j0) << (k - 1)) as i16,
                }
            });
        }
    }
    table.resize(LITERAL_STATES, LiteralEntry::default());
}

/// `fse_value_decoder_entry`: one L, M or D decoder state.
#[derive(Clone, Copy, Default)]
struct ValueEntry {
    total_bits: u8,
    value_bits: u8,
    delta: i16,
    base: i32,
}

/// `fse_init_value_decoder_table`.
fn fill_value_table(
    table: &mut Vec<ValueEntry>,
    states: usize,
    freq: &[u16],
    extra_bits: &[u8],
    base: &[i32],
) {
    table.clear();
    table.reserve_exact(states);
    for (symbol, &f) in freq.iter().enumerate() {
        let f = usize::from(f);
        if f == 0 {
            continue;
        }
        let (k, j0) = spread(states, f);
        let value_bits = extra_bits[symbol];
        for j in 0..f {
            table.push(if j < j0 {
                ValueEntry {
                    total_bits: k as u8 + value_bits,
                    value_bits,
                    delta: (((f + j) << k) as i16) - states as i16,
                    base: base[symbol],
                }
            } else {
                ValueEntry {
                    total_bits: (k - 1) as u8 + value_bits,
                    value_bits,
                    delta: ((j - j0) << (k - 1)) as i16,
                    base: base[symbol],
                }
            });
        }
    }
    table.resize(states, ValueEntry::default());
}

fn next_state(delta: i16, bits: u64) -> u16 {
    (delta as u16).wrapping_add(bits as u16)
}

/// `fse_decode`.
fn decode_literal(state: &mut u16, table: &[LiteralEntry], input: &mut BitStream) -> Result<u8> {
    let entry = table
        .get(usize::from(*state))
        .ok_or(LzfseError::Malformed("FSE state"))?;
    *state = next_state(entry.delta, input.pull(u32::from(entry.bits)));
    Ok(entry.symbol)
}

/// `fse_value_decode`.
fn decode_value(state: &mut u16, table: &[ValueEntry], input: &mut BitStream) -> Result<i32> {
    let entry = table
        .get(usize::from(*state))
        .ok_or(LzfseError::Malformed("FSE state"))?;
    let bits = input.pull(u32::from(entry.total_bits));
    *state = next_state(entry.delta, bits >> entry.value_bits);
    Ok(entry.base + (bits & ((1u64 << entry.value_bits) - 1)) as i32)
}

/// Scratch for one framed stream or a sequence of independent streams.
/// The input header bounds every vector, and each block resets its tables.
#[derive(Default)]
pub(crate) struct Decoder {
    literal_table: Vec<LiteralEntry>,
    l_table: Vec<ValueEntry>,
    m_table: Vec<ValueEntry>,
    d_table: Vec<ValueEntry>,
    literals: Vec<u8>,
}

/// Decode the literals and then the L, M, D triples of one LZFSE block
/// (`lzfse_decode` and `lzfse_decode_lmd`).
fn block(
    src: &[u8],
    frame: &Frame,
    header: &Header,
    out: &mut Output<'_>,
    decoder: &mut Decoder,
) -> Result<()> {
    let Decoder {
        literal_table,
        l_table,
        m_table,
        d_table,
        literals,
    } = decoder;
    fill_literal_table(literal_table, &header.literal_freq);
    fill_value_table(
        l_table,
        L_STATES,
        &header.l_freq,
        &L_EXTRA_BITS,
        &L_BASE_VALUE,
    );
    fill_value_table(
        m_table,
        M_STATES,
        &header.m_freq,
        &M_EXTRA_BITS,
        &M_BASE_VALUE,
    );
    fill_value_table(
        d_table,
        D_STATES,
        &header.d_freq,
        &D_EXTRA_BITS,
        &D_BASE_VALUE,
    );

    // The literal stream may refill from any earlier byte of the input, as
    // the reference allows; a valid stream never consumes those bits.
    let mut input = BitStream::new(src, 0, frame.lmd, header.literal_bits)?;
    literals.clear();
    literals.reserve((header.n_literals as usize).next_multiple_of(4));
    let mut states = header.literal_state;
    for _ in (0..header.n_literals).step_by(4) {
        input.refill()?;
        let [a, b, c, d] = &mut states;
        literals.extend_from_slice(&[
            decode_literal(a, literal_table, &mut input)?,
            decode_literal(b, literal_table, &mut input)?,
            decode_literal(c, literal_table, &mut input)?,
            decode_literal(d, literal_table, &mut input)?,
        ]);
    }

    let mut input = BitStream::new(src, frame.lmd, frame.end, header.lmd_bits)?;
    let (mut l_state, mut m_state, mut d_state) = (header.l_state, header.m_state, header.d_state);
    let mut literal = 0usize;
    let mut distance: Option<usize> = None;
    for _ in 0..header.n_matches {
        input.refill()?;
        let l = decode_value(&mut l_state, l_table, &mut input)? as usize;
        let m = decode_value(&mut m_state, m_table, &mut input)? as usize;
        let d = decode_value(&mut d_state, d_table, &mut input)? as usize;
        if d != 0 {
            distance = Some(d);
        }
        let run = literals
            .get(literal..literal + l)
            .ok_or(LzfseError::Malformed(
                "literal run past the block's literals",
            ))?;
        out.literals(run)?;
        literal += l;
        // The reference checks the distance even for a match of length 0.
        let distance = distance.ok_or(LzfseError::Malformed("match distance"))?;
        out.copy_match(distance, m)?;
    }
    Ok(())
}

/// The kinds of LZVN opcode (`lzvn_decode_base.c`'s `opc_tbl`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Opcode {
    SmallDistance,
    MediumDistance,
    LargeDistance,
    PreviousDistance,
    SmallMatch,
    LargeMatch,
    SmallLiteral,
    LargeLiteral,
    Nop,
    EndOfStream,
    Undefined,
}

fn opcode(op: u8) -> Opcode {
    match op {
        0x06 => Opcode::EndOfStream,
        0x0e | 0x16 => Opcode::Nop,
        0x1e | 0x26 | 0x2e | 0x36 | 0x3e => Opcode::Undefined,
        0x70..=0x7f | 0xd0..=0xdf => Opcode::Undefined,
        0xa0..=0xbf => Opcode::MediumDistance,
        0xe0 => Opcode::LargeLiteral,
        0xe1..=0xef => Opcode::SmallLiteral,
        0xf0 => Opcode::LargeMatch,
        0xf1..=0xff => Opcode::SmallMatch,
        _ => match op & 7 {
            6 => Opcode::PreviousDistance,
            7 => Opcode::LargeDistance,
            _ => Opcode::SmallDistance,
        },
    }
}

/// Decode one bare LZVN stream, without a `bvxn` header, as decmpfs stores
/// its chunks: it must produce exactly `raw_bytes` and end with its
/// end-of-stream opcode.
pub fn decode_lzvn(src: &[u8], raw_bytes: usize) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    append_lzvn(src, raw_bytes, &mut bytes)?;
    Ok(bytes)
}

pub(crate) fn append_lzvn(src: &[u8], raw_bytes: usize, bytes: &mut Vec<u8>) -> Result<()> {
    let mut out = Output::new(bytes, raw_bytes);
    out.begin_block(u32::try_from(raw_bytes).map_err(|_| LzfseError::Limit)?)?;
    let result = lzvn(src, &mut out, false).and_then(|()| out.end_block());
    if result.is_err() {
        out.bytes.truncate(out.start);
    }
    result
}

/// Decode one LZVN payload up to its end-of-stream opcode, which a block
/// must end with exactly, padding included. A decmpfs chunk drops the
/// padding when its bytes round its stored size.
fn lzvn(src: &[u8], out: &mut Output<'_>, block: bool) -> Result<()> {
    let mut at = 0usize;
    let mut distance = 0usize;
    // Each opcode needs its operands, its literals and one more byte (the
    // next opcode) to follow.
    let need = |at: usize, bytes: usize| {
        if src.len() - at <= bytes {
            Err(LzfseError::Truncated)
        } else {
            Ok(())
        }
    };
    loop {
        let op = *src.get(at).ok_or(LzfseError::Truncated)?;
        let (operands, literals, length) = match opcode(op) {
            Opcode::EndOfStream => {
                if block && src.len() - at != 8 {
                    return Err(LzfseError::Malformed("LZVN end of stream"));
                }
                return Ok(());
            }
            Opcode::Nop => {
                need(at, 1)?;
                at += 1;
                continue;
            }
            Opcode::Undefined => return Err(LzfseError::Malformed("undefined LZVN opcode")),
            Opcode::SmallDistance => {
                let l = usize::from(op >> 6);
                need(at, 2 + l)?;
                distance = usize::from(op & 7) << 8 | usize::from(src[at + 1]);
                (2, l, usize::from((op >> 3) & 7) + 3)
            }
            Opcode::MediumDistance => {
                let l = usize::from((op >> 3) & 3);
                need(at, 3 + l)?;
                let operand = usize::from(u16_at(src, at + 1)?);
                distance = operand >> 2;
                (3, l, (usize::from(op & 7) << 2 | (operand & 3)) + 3)
            }
            Opcode::LargeDistance => {
                let l = usize::from(op >> 6);
                need(at, 3 + l)?;
                distance = usize::from(u16_at(src, at + 1)?);
                (3, l, usize::from((op >> 3) & 7) + 3)
            }
            Opcode::PreviousDistance => {
                let l = usize::from(op >> 6);
                need(at, 1 + l)?;
                (1, l, usize::from((op >> 3) & 7) + 3)
            }
            Opcode::SmallMatch => {
                need(at, 1)?;
                (1, 0, usize::from(op & 0xf))
            }
            Opcode::LargeMatch => {
                need(at, 2)?;
                (2, 0, usize::from(src[at + 1]) + 16)
            }
            Opcode::SmallLiteral => {
                let l = usize::from(op & 0xf);
                need(at, 1 + l)?;
                (1, l, 0)
            }
            Opcode::LargeLiteral => {
                need(at, 2)?;
                let l = usize::from(src[at + 1]) + 16;
                need(at, 2 + l)?;
                (2, l, 0)
            }
        };
        at += operands;
        out.literals(&src[at..at + literals])?;
        at += literals;
        if length != 0 {
            out.copy_match(distance, length)?;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn stored(raw: &[u8]) -> Vec<u8> {
        [
            b"bvx-".as_slice(),
            &(raw.len() as u32).to_le_bytes(),
            raw,
            b"bvx$",
        ]
        .concat()
    }

    fn lzvn_block(raw_bytes: u32, payload: &[u8]) -> Vec<u8> {
        [
            b"bvxn".as_slice(),
            &raw_bytes.to_le_bytes(),
            &(payload.len() as u32).to_le_bytes(),
            payload,
        ]
        .concat()
    }

    const LZVN_END: [u8; 8] = [0x06, 0, 0, 0, 0, 0, 0, 0];

    #[test]
    fn bare_lzvn_streams_decode_to_exactly_their_size() {
        // "abc", then a match of 3 at distance 3: "abcabc".
        let stream = [&[0xe3, b'a', b'b', b'c', 0b00_000_111, 3, 0][..], &LZVN_END].concat();
        assert_eq!(decode_lzvn(&stream, 6).unwrap(), b"abcabc");
        assert!(decode_lzvn(&stream, 7).is_err());
        assert!(decode_lzvn(&stream, 5).is_err());
        // A decmpfs chunk may stop at the opcode, without the padding.
        let chunk = [0xe3, b'a', b'b', b'c', 0b00_000_111, 3, 0, 0x06];
        assert_eq!(decode_lzvn(&chunk, 6).unwrap(), b"abcabc");
    }

    #[test]
    fn rebuilding_underfilled_tables_clears_previous_states() {
        let mut decoder = Decoder::default();
        let mut literals = [0; LITERAL_SYMBOLS];
        literals[42] = LITERAL_STATES as u16;
        fill_literal_table(&mut decoder.literal_table, &literals);
        assert_eq!(decoder.literal_table[LITERAL_STATES - 1].symbol, 42);
        literals[42] = 1;
        fill_literal_table(&mut decoder.literal_table, &literals);
        assert_eq!(decoder.literal_table[0].symbol, 42);
        assert_eq!(decoder.literal_table[LITERAL_STATES - 1].symbol, 0);

        let mut lengths = [0; L_SYMBOLS];
        lengths[3] = L_STATES as u16;
        fill_value_table(
            &mut decoder.l_table,
            L_STATES,
            &lengths,
            &L_EXTRA_BITS,
            &L_BASE_VALUE,
        );
        assert_eq!(decoder.l_table[L_STATES - 1].base, 3);
        lengths[3] = 1;
        fill_value_table(
            &mut decoder.l_table,
            L_STATES,
            &lengths,
            &L_EXTRA_BITS,
            &L_BASE_VALUE,
        );
        assert_eq!(decoder.l_table[0].base, 3);
        assert_eq!(decoder.l_table[L_STATES - 1].base, 0);
    }

    #[test]
    fn appended_streams_cannot_match_the_existing_prefix() {
        let mut bytes = b"prefix".to_vec();
        let valid = [0xe3, b'a', b'b', b'c', 0b00_000_111, 3, 0, 0x06];
        append_lzvn(&valid, 6, &mut bytes).unwrap();
        assert_eq!(bytes, b"prefixabcabc");

        // A literal is written before the malformed match. The match
        // would reach the previous stream and must roll that literal back.
        let invalid = [0b01_000_111, 3, 0, b'x', 0x06];
        assert_eq!(
            append_lzvn(&invalid, 4, &mut bytes),
            Err(LzfseError::Malformed("match distance"))
        );
        assert_eq!(bytes, b"prefixabcabc");

        let mut framed = lzvn_block(4, &[&invalid[..4], &LZVN_END].concat());
        framed.extend(b"bvx$");
        assert_eq!(
            Decoder::default().append(&framed, 4, &mut bytes),
            Err(LzfseError::Malformed("match distance"))
        );
        assert_eq!(bytes, b"prefixabcabc");

        let mut truncated = stored(b"new");
        truncated.truncate(truncated.len() - 4);
        assert_eq!(
            Decoder::default().append(&truncated, 3, &mut bytes),
            Err(LzfseError::Truncated)
        );
        assert_eq!(bytes, b"prefixabcabc");
    }

    #[test]
    fn overlapping_matches_repeat_until_the_declared_end() {
        let mut bytes = Vec::new();
        let mut out = Output::new(&mut bytes, 21);
        out.begin_block(21).unwrap();
        out.literals(b"abc").unwrap();
        out.copy_match(3, 13).unwrap();
        out.copy_match(1, 5).unwrap();
        assert_eq!(out.bytes.as_slice(), b"abcabcabcabcabcaaaaaa");
        out.end_block().unwrap();
    }

    /// The observed decode hands out each frame's final output in order,
    /// a compressed frame's included, and the pieces make the output.
    #[test]
    fn the_observed_bounded_decode_hands_out_every_frame_in_order() {
        let lzvn = [&[0xe3, b'a', b'b', b'c', 0b00_000_111, 3, 0][..], &LZVN_END].concat();
        let mut encoded = stored(b"first");
        encoded.truncate(encoded.len() - 4);
        encoded.extend_from_slice(&lzvn_block(6, &lzvn));
        encoded.extend_from_slice(&stored(b"last"));
        let mut at = 0;
        let mut read = |part: &mut [u8]| -> Result<()> {
            part.copy_from_slice(&encoded[at..at + part.len()]);
            at += part.len();
            Ok(())
        };
        let mut pieces = Vec::new();
        let bytes = decode_bounded_observed(&mut read, encoded.len(), 15, |piece| {
            pieces.push(piece.to_vec())
        })
        .unwrap()
        .unwrap();
        assert_eq!(bytes, b"firstabcabclast");
        assert_eq!(pieces, [&b"first"[..], b"abcabc", b"last"]);
    }

    #[test]
    fn decodes_stored_blocks_and_enforces_the_limit() {
        assert_eq!(decode(&stored(b"stela"), 5).unwrap(), b"stela");
        assert_eq!(decode(&stored(b"stela"), 4), Err(LzfseError::Limit));
        let mut truncated = stored(b"stela");
        truncated.pop();
        assert_eq!(decode(&truncated, 64), Err(LzfseError::Truncated));
        let mut trailing = stored(b"stela");
        trailing.push(0);
        assert!(matches!(
            decode(&trailing, 64),
            Err(LzfseError::Malformed(_))
        ));
        assert!(matches!(
            decode(b"bvx3\0\0\0\0", 64),
            Err(LzfseError::Malformed(_))
        ));
        assert!(is_lzfse(b"bvx2....") && !is_lzfse(b"bvx$") && !is_lzfse(b"bv"));
    }

    #[test]
    fn sizes_only_a_complete_multiblock_stream_within_the_limit() {
        let mut encoded = stored(b"first");
        assert_eq!(framed_output_size(&encoded, 12), None);
        encoded.truncate(encoded.len() - 4);
        encoded.extend_from_slice(&stored(b"second"));
        assert_eq!(framed_output_size(&encoded, 11), Some(11));
        assert_eq!(decode(&encoded, 11).unwrap(), b"firstsecond");
        assert_eq!(framed_output_size(&encoded, 10), None);
        assert_eq!(decode(&encoded, 10), Err(LzfseError::Limit));
        assert_eq!(framed_output_size(&encoded[..encoded.len() - 1], 11), None);
    }

    #[test]
    fn decodes_each_lzvn_opcode_kind() {
        let payload = [
            // Small literal: "abc".
            &[0xe3, b'a', b'b', b'c'][..],
            // Small distance, L=1 'd', M=3, D=4: "dabc".
            &[0b01_000_000, 4, b'd'],
            // Previous distance, L=1 'e', M=3: "eabc".
            &[0b01_000_110, b'e'],
            // Nop.
            &[0x0e],
            // Large distance, L=0, M=3, D=1: "ccc".
            &[0b00_000_111, 1, 0],
            // Medium distance, L=0, M=3+((1<<2)|1)=8, D=2: "cccccccc".
            &[0xa1, (2 << 2 | 1) as u8, 0],
            // Small match, M=2 at the previous distance: "cc".
            &[0xf2],
            // Large literal: 16 + 1 bytes of 'x'.
            &[0xe0, 1],
            &[b'x'; 17],
            // Large match, M=16 at D=2 (still 2): 16 more 'x'.
            &[0xf0, 0],
            &LZVN_END,
        ]
        .concat();
        let mut expected = b"abcdabceabcccc".to_vec();
        expected.extend_from_slice(b"cccccccccc");
        expected.extend_from_slice(&[b'x'; 33]);
        let stream = [
            lzvn_block(expected.len() as u32, &payload),
            b"bvx$".to_vec(),
        ]
        .concat();
        assert_eq!(decode(&stream, 1 << 20).unwrap(), expected);

        let short = [
            lzvn_block(expected.len() as u32 + 1, &payload),
            b"bvx$".to_vec(),
        ]
        .concat();
        assert!(matches!(
            decode(&short, 1 << 20),
            Err(LzfseError::Malformed(_))
        ));
        let far = [&[0xe1, b'a'][..], &[0b00_000_000, 2], &LZVN_END].concat();
        let far = [lzvn_block(4, &far), b"bvx$".to_vec()].concat();
        assert_eq!(
            decode(&far, 64),
            Err(LzfseError::Malformed("match distance"))
        );
        let undefined = [lzvn_block(0, &[0x70, 0, 0]), b"bvx$".to_vec()].concat();
        assert!(matches!(
            decode(&undefined, 64),
            Err(LzfseError::Malformed(_))
        ));
    }

    /// Deterministic text-like bytes with repeats at many distances.
    fn sample(bytes: usize, seed: u64) -> Vec<u8> {
        let words: [&[u8]; 8] = [
            b"kernel ",
            b"collection ",
            b"device ",
            b"tree ",
            b"\0\0\0\0",
            b"stela ",
            b"\xff\xfe",
            b"preboot ",
        ];
        let mut state = seed;
        let mut out = vec![];
        while out.len() < bytes {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let pick = (state >> 33) as usize;
            if pick.is_multiple_of(5) {
                out.push((pick >> 8) as u8);
            } else {
                out.extend_from_slice(words[pick % words.len()]);
            }
        }
        out.truncate(bytes);
        out
    }

    #[test]
    fn matches_the_reference_encoder_on_lzfse_and_lzvn_streams() {
        for (bytes, seed) in [
            (0usize, 1u64),
            (100, 2),
            (3000, 3),
            (70_000, 4),
            (400_000, 5),
        ] {
            let raw = sample(bytes, seed);
            let mut encoded = vec![];
            lzfse_rust::LzfseEncoder::default()
                .encode_bytes(&raw, &mut encoded)
                .unwrap();
            assert_eq!(decode(&encoded, raw.len()).unwrap(), raw, "{bytes} bytes");
            if !raw.is_empty() {
                assert_eq!(decode(&encoded, raw.len() - 1), Err(LzfseError::Limit));
            }
        }
    }

    #[test]
    fn decodes_apple_encoded_fixtures() {
        // Encoded by macOS `compression_tool -encode -a lzfse` from
        // `sample(N, 7)`: N = 3000 makes one LZVN block, N = 300000 LZFSE
        // blocks with packed tables.
        for (name, bytes, encoded) in [
            (
                "lzvn",
                3000,
                &include_bytes!("../tests/fixtures/lzfse/sample-3000.lzfse")[..],
            ),
            (
                "lzfse",
                300_000,
                &include_bytes!("../tests/fixtures/lzfse/sample-300000.lzfse")[..],
            ),
        ] {
            assert_eq!(decode(encoded, bytes).unwrap(), sample(bytes, 7), "{name}");
        }
    }

    #[test]
    fn rejects_corrupt_lzfse_blocks() {
        let raw = sample(70_000, 9);
        let mut encoded = vec![];
        lzfse_rust::LzfseEncoder::default()
            .encode_bytes(&raw, &mut encoded)
            .unwrap();
        assert_eq!(&encoded[..4], b"bvx2");
        for cut in [10, 40, encoded.len() / 2, encoded.len() - 1] {
            assert!(decode(&encoded[..cut], 1 << 20).is_err(), "cut at {cut}");
        }
        for flip in [12, 30, 200, encoded.len() / 2] {
            let mut corrupt = encoded.clone();
            corrupt[flip] ^= 0x5a;
            assert_ne!(
                decode(&corrupt, 1 << 20).ok(),
                Some(raw.clone()),
                "flip at {flip}"
            );
        }
    }
}
