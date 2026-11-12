//! Reading a remote object by byte ranges, with read-ahead, for readers
//! that jump around it: the restore's install image on a CDN.
//!
//! [`Ranges`] keeps up to [`STREAMS`] streams. Each has one open range,
//! from where its reads are to the object's end (an ordinary download from
//! there), filling a read-ahead buffer: a read at or just past a stream's
//! buffer continues that stream, and a read elsewhere starts a free stream
//! there or, with none free, restarts the least recently read one. So two
//! transfers that read far apart, as a restore that streams the system
//! image while it reads other members does, each keep their own range and
//! read-ahead instead of trading one range between them.
//!
//! [`Ranges::pump`] keeps the buffers filling between reads, [`AHEAD`] in
//! all, split among the streams. A range that fails resumes from its last
//! byte received, a few times. Once [`Ranges::bound`] is set, a read waits
//! at most [`BUDGET_MS`] for the network and otherwise fails as blocked
//! (`ReadAt::blocked`); the same read later goes on from what arrived. A
//! restarted stream keeps the bytes its reads used as a segment
//! ([`SEGMENT_BYTES`] in all), so a request that reads several members
//! again from its first finds those it already read.
//!
//! The transport is a [`RangeSource`], so the boot's network and tests
//! serve the same logic.
use super::NetError;
use alloc::{collections::VecDeque, vec::Vec};
use core::fmt;
use stela_formats::payload::ReadAt;

/// Streams with their own range and read-ahead.
pub const STREAMS: usize = 2;
/// How far [`Ranges::pump`] fills ahead of the reads, all streams together.
pub const AHEAD: usize = 32 << 20;
/// A read up to this far past a stream's buffered bytes continues it.
pub const SKIP: u64 = 8 << 20;
/// Bytes already read that stay buffered in each stream for a re-read.
pub const KEEP: usize = 8 << 20;
/// Room for what one step takes past a stream's read-ahead (about a
/// socket's worth), so its ring rarely grows.
pub const SLACK: usize = 8 << 20;
/// Consecutive failures of one range's progress before a read fails; the
/// waits between them double from a second, about a minute in all.
pub const ATTEMPTS: u32 = 7;
/// The longest one wait for an open range's next frame lasts.
pub const WAIT_US: u64 = 100_000;
/// The longest a bounded read waits for the network, and so holds the
/// processor that serves it.
pub const BUDGET_MS: u64 = 20;
/// The bytes kept from restarted streams for reads that come back to them.
pub const SEGMENT_BYTES: usize = 64 << 20;

/// A transport of byte ranges of one object.
pub trait RangeSource {
    /// One open range.
    type Range;
    /// Requests bytes `first..=last`.
    fn open(&mut self, first: u64, last: u64) -> Result<Self::Range, NetError>;
    /// Without waiting, hands what arrived of the range's body to `take`.
    /// Returns whether the range is complete.
    fn read(
        &mut self,
        range: &mut Self::Range,
        take: &mut dyn FnMut(&[u8]),
    ) -> Result<bool, NetError>;
    fn close(&mut self, range: Self::Range);
    /// Waits up to `micros` for the network's next event.
    fn wait(&mut self, micros: u64) -> Result<(), NetError>;
    /// A monotonic millisecond clock.
    fn millis(&mut self) -> u64;
    /// Records a progress line.
    fn note(&mut self, line: fmt::Arguments<'_>);
}

struct Stream<R> {
    /// Object bytes from `start`; the open range continues at their end.
    start: u64,
    buffer: VecDeque<u8>,
    range: Option<R>,
    /// The farthest byte a read asked for since the buffer started.
    wanted: u64,
    /// When a read last used the stream, on [`Ranges::tick`].
    used: u64,
    /// Consecutive failures of the open range's progress, and when the
    /// last one was, for a bounded read's backoff.
    failures: u32,
    failed_at: u64,
}

impl<R> Stream<R> {
    fn end(&self) -> u64 {
        self.start + self.buffer.len() as u64
    }

    /// Whether a read at `offset` continues this stream.
    fn holds(&self, offset: u64) -> bool {
        offset >= self.start && offset <= self.end().saturating_add(SKIP)
    }
}

pub struct Ranges<S: RangeSource> {
    source: S,
    len: u64,
    streams: Vec<Stream<S::Range>>,
    /// Counts reads, to find the least recently read stream.
    tick: u64,
    /// Ranges requested so far.
    ranges: u64,
    /// Whether [`Ranges::pump`] reads ahead: from a read that succeeded
    /// until a read-ahead fails.
    ahead: bool,
    /// Earlier buffers, by their first byte, oldest first: each up to
    /// where its reads reached, without what was read ahead past them.
    segments: VecDeque<(u64, Vec<u8>)>,
    /// Whether reads wait for the network at most [`BUDGET_MS`].
    bounded: bool,
    /// The last read failed only because its time ran out.
    blocked: bool,
}

impl<S: RangeSource> Ranges<S> {
    pub fn new(source: S, len: u64) -> Self {
        Self {
            source,
            len,
            streams: Vec::with_capacity(STREAMS),
            tick: 0,
            ranges: 0,
            ahead: false,
            segments: VecDeque::new(),
            bounded: false,
            blocked: false,
        }
    }

    pub fn source(&self) -> &S {
        &self.source
    }

    pub fn source_mut(&mut self) -> &mut S {
        &mut self.source
    }

    /// Ranges requested so far.
    pub fn ranges(&self) -> u64 {
        self.ranges
    }

    /// Bytes held in the streams' buffers.
    pub fn buffered(&self) -> usize {
        self.streams.iter().map(|stream| stream.buffer.len()).sum()
    }

    /// Bytes held in kept segments.
    pub fn kept(&self) -> usize {
        self.segments.iter().map(|(_, bytes)| bytes.len()).sum()
    }

    /// Have every later read wait for the network at most [`BUDGET_MS`].
    pub fn bound(&mut self) {
        self.bounded = true;
    }

    /// Closes every range and frees the buffers, returning the capacity
    /// the buffers had.
    pub fn retire(&mut self) -> usize {
        self.ahead = false;
        let mut capacity = 0;
        for mut stream in core::mem::take(&mut self.streams) {
            if let Some(range) = stream.range.take() {
                self.source.close(range);
            }
            capacity += stream.buffer.capacity();
        }
        self.segments = VecDeque::new();
        capacity
    }

    fn stop(&mut self, at: usize) {
        if let Some(range) = self.streams[at].range.take() {
            self.source.close(range);
        }
    }

    /// The stream a read at `offset` uses: the one it continues, else a new
    /// one while fewer than [`STREAMS`] exist, else the least recently read
    /// one, restarted at `offset`.
    fn choose(&mut self, offset: u64) -> usize {
        if let Some(at) = self.streams.iter().position(|stream| stream.holds(offset)) {
            return at;
        }
        if self.streams.len() < STREAMS {
            // The first stream may read ahead alone; later ones share.
            let ahead = match self.streams.is_empty() {
                true => AHEAD,
                false => AHEAD / STREAMS,
            };
            self.streams.push(Stream {
                start: offset,
                buffer: VecDeque::with_capacity(ahead + KEEP + SLACK),
                range: None,
                wanted: offset,
                used: 0,
                failures: 0,
                failed_at: 0,
            });
            return self.streams.len() - 1;
        }
        let at = (0..self.streams.len())
            .min_by_key(|&at| self.streams[at].used)
            .expect("STREAMS is not zero");
        self.stop(at);
        self.keep_segment(at);
        let stream = &mut self.streams[at];
        stream.start = offset;
        stream.wanted = offset;
        stream.failures = 0;
        at
    }

    /// Keep a stream's buffer as a segment before it starts over elsewhere,
    /// dropping the oldest segments past [`SEGMENT_BYTES`].
    fn keep_segment(&mut self, at: usize) {
        let stream = &mut self.streams[at];
        if stream.buffer.is_empty() {
            return;
        }
        let reached = stream.wanted.saturating_sub(stream.start) as usize;
        let mut bytes: Vec<u8> = stream.buffer.drain(..).take(reached).collect();
        bytes.truncate(SEGMENT_BYTES);
        if bytes.is_empty() {
            return;
        }
        self.segments.push_back((stream.start, bytes));
        let mut total = self.kept();
        while total > SEGMENT_BYTES
            && let Some((_, dropped)) = self.segments.pop_front()
        {
            total -= dropped.len();
        }
    }

    /// Copy `out` from a kept segment that holds all of it.
    fn read_segment(&self, offset: u64, out: &mut [u8]) -> bool {
        let found = self.segments.iter().rev().find(|(start, bytes)| {
            offset >= *start && offset + out.len() as u64 <= *start + bytes.len() as u64
        });
        if let Some((start, bytes)) = found {
            let at = (offset - start) as usize;
            out.copy_from_slice(&bytes[at..at + out.len()]);
        }
        found.is_some()
    }

    /// Takes what has arrived for a stream, opening a range at its buffer's
    /// end first when none is open.
    fn step(&mut self, at: usize) -> Result<(), NetError> {
        if self.streams[at].range.is_none() {
            let (first, last) = (self.streams[at].end(), self.len - 1);
            let range = self.source.open(first, last)?;
            self.streams[at].range = Some(range);
            self.ranges += 1;
            let ranges = self.ranges;
            self.source.note(format_args!(
                "REMOTE IMAGE range {ranges} stream {at} bytes={first}-{last}\n"
            ));
        }
        let stream = &mut self.streams[at];
        let range = stream.range.as_mut().expect("opened above");
        let buffer = &mut stream.buffer;
        let done = self.source.read(range, &mut |body| buffer.extend(body))?;
        if done {
            self.stop(at);
        }
        Ok(())
    }

    /// Fills a stream up to `until`: bounded, within [`BUDGET_MS`], and
    /// after a failure not before its backoff ends; unbounded, waiting
    /// those out. A failure's count goes on across bounded reads.
    fn fill(&mut self, at: usize, until: u64) -> Result<(), NetError> {
        let started = self.source.millis();
        while self.streams[at].end() < until {
            let spent = self.source.millis().saturating_sub(started);
            if self.bounded && spent >= BUDGET_MS {
                return Err(self.block("the read's time ran out"));
            }
            let (failures, failed_at) = (self.streams[at].failures, self.streams[at].failed_at);
            if failures > 0 {
                let backoff = 1000 << (failures - 1);
                if self.source.millis().saturating_sub(failed_at) < backoff {
                    if self.bounded {
                        return Err(self.block("the range waits before it is asked again"));
                    }
                    while self.source.millis().saturating_sub(failed_at) < backoff {
                        core::hint::spin_loop();
                    }
                }
            }
            let before = self.streams[at].end();
            if let Err(error) = self.step(at) {
                self.stop(at);
                let stream = &mut self.streams[at];
                if !error.retryable() {
                    stream.failures = 0;
                    return Err(error);
                }
                stream.failures += 1;
                let failures = stream.failures;
                self.source.note(format_args!(
                    "REMOTE IMAGE range at {before} failed ({failures}/{ATTEMPTS}): {error}\n"
                ));
                if failures == ATTEMPTS {
                    self.streams[at].failures = 0;
                    return Err(error);
                }
                self.streams[at].failed_at = self.source.millis();
            } else if self.streams[at].end() > before {
                self.streams[at].failures = 0;
            } else if self.streams[at].range.is_some() {
                let wait = match self.bounded {
                    true => BUDGET_MS.saturating_sub(spent) * 1000,
                    false => WAIT_US,
                };
                if let Err(error) = self.source.wait(wait.min(WAIT_US)) {
                    self.stop(at);
                    return Err(error);
                }
            }
        }
        Ok(())
    }

    /// A bounded read that ran out of time: the ranges keep filling between
    /// requests, and the same read later goes on.
    fn block(&mut self, why: &'static str) -> NetError {
        self.blocked = true;
        self.ahead = true;
        NetError::Timeout(why)
    }

    /// Without waiting, takes what has arrived for each stream while its
    /// share of [`AHEAD`] is not buffered, opening the next range when one
    /// ends. A failure leaves the next read to retry.
    pub fn pump(&mut self) {
        if !self.ahead || self.streams.is_empty() {
            return;
        }
        let share = AHEAD / self.streams.len();
        for at in 0..self.streams.len() {
            let stream = &self.streams[at];
            if stream.buffer.len() < share && stream.end() < self.len && self.step(at).is_err() {
                self.stop(at);
                self.ahead = false;
                return;
            }
        }
    }
}

impl<S: RangeSource> ReadAt for Ranges<S> {
    type Error = NetError;

    fn len(&self) -> u64 {
        self.len
    }

    fn blocked(&mut self) -> bool {
        core::mem::take(&mut self.blocked)
    }

    fn read_at(&mut self, offset: u64, out: &mut [u8]) -> Result<(), NetError> {
        let until = offset
            .checked_add(out.len() as u64)
            .ok_or(NetError::Invalid("remote image offset overflow"))?;
        if until > self.len {
            return Err(NetError::Invalid("read past the remote image's end"));
        }
        if self.read_segment(offset, out) {
            return Ok(());
        }
        let at = self.choose(offset);
        self.tick += 1;
        let stream = &mut self.streams[at];
        stream.used = self.tick;
        stream.wanted = stream.wanted.max(until);
        self.fill(at, until)?;
        self.ahead = true;
        let stream = &mut self.streams[at];
        // The ring's two halves, from the read's first byte.
        let (mut skip, mut filled) = ((offset - stream.start) as usize, 0);
        for part in <[&[u8]; 2]>::from(stream.buffer.as_slices()) {
            if skip < part.len() {
                let take = (part.len() - skip).min(out.len() - filled);
                out[filled..filled + take].copy_from_slice(&part[skip..skip + take]);
                filled += take;
                skip = 0;
            } else {
                skip -= part.len();
            }
        }
        // Bytes before this read are done with, but the last few stay for
        // a re-read.
        let done = (until - stream.start) as usize;
        if done > KEEP {
            stream.buffer.drain(..done - KEEP);
            stream.start = until - KEEP as u64;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
