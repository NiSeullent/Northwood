use super::*;
use alloc::collections::VecDeque;

const GIB: u64 = 1 << 30;
const LEN: u64 = 30 * GIB;
const READ: usize = 256 << 10;

/// The object's byte at `at`, so every read can be checked.
fn byte(at: u64) -> u8 {
    (at ^ (at >> 11) ^ (at >> 23) ^ (at >> 31)) as u8
}

struct Range {
    next: u64,
    last: u64,
}

/// An object served by ranges: each read hands over up to `chunk` bytes
/// unless stalled; a clock that moves a millisecond per look and by each
/// wait.
struct Mock {
    chunk: usize,
    stalled: bool,
    opens: Vec<u64>,
    live: usize,
    open_errors: VecDeque<NetError>,
    read_errors: VecDeque<NetError>,
    now: u64,
}

impl Mock {
    fn new() -> Self {
        Self {
            chunk: 1 << 20,
            stalled: false,
            opens: Vec::new(),
            live: 0,
            open_errors: VecDeque::new(),
            read_errors: VecDeque::new(),
            now: 0,
        }
    }
}

impl RangeSource for Mock {
    type Range = Range;

    fn open(&mut self, first: u64, last: u64) -> Result<Range, NetError> {
        if let Some(error) = self.open_errors.pop_front() {
            return Err(error);
        }
        self.opens.push(first);
        self.live += 1;
        Ok(Range { next: first, last })
    }

    fn read(&mut self, range: &mut Range, take: &mut dyn FnMut(&[u8])) -> Result<bool, NetError> {
        if let Some(error) = self.read_errors.pop_front() {
            return Err(error);
        }
        if self.stalled {
            return Ok(false);
        }
        let n = (self.chunk as u64).min(range.last + 1 - range.next);
        let bytes: Vec<u8> = (range.next..range.next + n).map(byte).collect();
        range.next += n;
        take(&bytes);
        Ok(range.next > range.last)
    }

    fn close(&mut self, _: Range) {
        self.live -= 1;
    }

    fn wait(&mut self, micros: u64) -> Result<(), NetError> {
        self.now += (micros / 1000).max(1);
        Ok(())
    }

    fn millis(&mut self) -> u64 {
        self.now += 1;
        self.now
    }

    fn note(&mut self, _: fmt::Arguments<'_>) {}
}

fn ranges() -> Ranges<Mock> {
    Ranges::new(Mock::new(), LEN)
}

fn read(reader: &mut Ranges<Mock>, offset: u64, len: usize) -> Result<(), NetError> {
    let mut out = vec![0; len];
    reader.read_at(offset, &mut out)?;
    assert!(
        out.iter().zip(offset..).all(|(&got, at)| got == byte(at)),
        "bytes at {offset}"
    );
    Ok(())
}

fn bounded_memory(reader: &Ranges<Mock>) {
    assert!(reader.buffered() <= AHEAD + STREAMS * (KEEP + SLACK));
    assert!(reader.kept() <= SEGMENT_BYTES);
    assert!(reader.source().live <= STREAMS);
}

#[test]
fn two_interleaved_sequential_readers_keep_their_two_ranges() {
    let mut reader = ranges();
    let (mut system, mut other) = (2 * GIB, 24 * GIB);
    for _ in 0..400 {
        read(&mut reader, system, READ).unwrap();
        reader.pump();
        read(&mut reader, other, READ / 4).unwrap();
        reader.pump();
        bounded_memory(&reader);
        system += READ as u64;
        // The other reader skips a little now and then.
        other += READ as u64 / 4 + 4096;
    }
    assert_eq!(reader.ranges(), 2);
    assert_eq!(reader.source().opens, [2 * GIB, 24 * GIB]);
}

#[test]
fn a_third_far_reader_replaces_the_least_recently_read_stream() {
    let mut reader = ranges();
    let (a, b, c) = (2 * GIB, 24 * GIB, 10 * GIB);
    read(&mut reader, a, READ).unwrap();
    read(&mut reader, b, READ).unwrap();
    read(&mut reader, a + READ as u64, READ).unwrap();
    // `b` was read least recently.
    read(&mut reader, c, READ).unwrap();
    assert_eq!(reader.source().opens, [a, b, c]);
    // `a` goes on in its own range.
    read(&mut reader, a + 2 * READ as u64, READ).unwrap();
    assert_eq!(reader.ranges(), 3);
    // What `b`'s reads used stays as a segment; past it needs a range,
    // which replaces `c`, now the least recently read.
    read(&mut reader, b, READ).unwrap();
    assert_eq!(reader.ranges(), 3);
    read(&mut reader, b + READ as u64, READ).unwrap();
    assert_eq!(reader.source().opens, [a, b, c, b + READ as u64]);
    // `a` still goes on in its own range.
    read(&mut reader, a + 3 * READ as u64, READ).unwrap();
    assert_eq!(reader.ranges(), 4);
    bounded_memory(&reader);
}

#[test]
fn memory_stays_bounded_while_readers_jump_around() {
    let mut reader = ranges();
    for round in 0..64u64 {
        let base = (round * 7 % 29) * GIB;
        for step in 0..8 {
            read(&mut reader, base + step * READ as u64, READ).unwrap();
            for _ in 0..8 {
                reader.pump();
            }
            bounded_memory(&reader);
        }
    }
    assert!(reader.retire() > 0);
    assert_eq!(reader.buffered(), 0);
    assert_eq!(reader.source().live, 0);
}

#[test]
fn a_refused_range_passes_its_error_through() {
    let mut reader = ranges();
    reader
        .source_mut()
        .open_errors
        .push_back(NetError::Status(404));
    assert_eq!(read(&mut reader, GIB, READ), Err(NetError::Status(404)));
    // The next read asks again.
    read(&mut reader, GIB, READ).unwrap();
    assert_eq!(reader.ranges(), 1);
}

#[test]
fn a_dropped_range_resumes_from_its_last_byte() {
    let mut reader = ranges();
    read(&mut reader, GIB, READ).unwrap();
    reader.source_mut().read_errors.push_back(NetError::Reset);
    read(&mut reader, GIB + (2 << 20), READ).unwrap();
    let opens = &reader.source().opens;
    assert_eq!(opens.len(), 2);
    assert!(opens[1] > GIB, "resumed at {}", opens[1]);
    assert_eq!(reader.source().live, 1);
}

#[test]
fn a_bounded_read_that_waits_too_long_is_pending_and_goes_on_later() {
    let mut reader = ranges();
    reader.bound();
    reader.source_mut().stalled = true;
    assert!(matches!(
        read(&mut reader, 3 * GIB, READ),
        Err(NetError::Timeout(_))
    ));
    assert!(reader.blocked());
    assert!(!reader.blocked(), "asking clears it");
    // The range stays open and fills between requests.
    reader.source_mut().stalled = false;
    reader.pump();
    read(&mut reader, 3 * GIB, READ).unwrap();
    assert_eq!(reader.ranges(), 1);
    assert!(!reader.blocked());
}

#[test]
fn a_bounded_read_after_a_failure_waits_out_its_backoff_as_pending() {
    let mut reader = ranges();
    reader.bound();
    reader.source_mut().open_errors.push_back(NetError::Reset);
    assert!(matches!(
        read(&mut reader, GIB, READ),
        Err(NetError::Timeout(_))
    ));
    assert!(reader.blocked());
    reader.source_mut().now += 1000;
    read(&mut reader, GIB, READ).unwrap();
    assert_eq!(reader.ranges(), 1);
}

#[test]
fn reads_past_the_end_are_refused() {
    let mut reader = ranges();
    assert!(matches!(
        read(&mut reader, LEN - 10, 11),
        Err(NetError::Invalid(_))
    ));
    read(&mut reader, LEN - 10, 10).unwrap();
}
