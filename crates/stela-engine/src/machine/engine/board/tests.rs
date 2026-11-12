//! Several threads, each a PE, take away and reuse what the others read,
//! under [`Grace`]: no reader ever sees what it holds change.

use super::*;
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU32, AtomicUsize},
    },
    thread,
    time::{Duration, Instant},
};

/// A deadline for one wait, counted from its start: a stuck grace period
/// fails the test, a slow host does not. Each check yields, so a waiter on
/// a loaded host leaves the CPU to the readers it waits for.
fn give_up() -> impl FnMut() -> bool {
    let started = Instant::now();
    move || {
        thread::yield_now();
        started.elapsed() > Duration::from_secs(60)
    }
}

/// A small pseudo-random step, so each thread reads its own pattern.
fn next(seed: &mut u64) -> u64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed
}

#[test]
fn grace_waits_for_readers_and_not_for_outside_pes() {
    let grace = Grace::new(3);
    grace.enter(0);
    let snapshot = grace.snapshot();
    // PE 0 is inside since before; PEs 1 and 2 are outside.
    assert!(!grace.passed(&snapshot, None));
    assert!(grace.passed(&snapshot, Some(0)));
    grace.pass(0);
    assert!(grace.passed(&snapshot, None));
    // A PE that enters after the snapshot holds nothing of before.
    grace.enter(1);
    assert!(grace.passed(&snapshot, None));
    let later = grace.snapshot();
    assert!(!grace.passed(&later, None));
    grace.leave(1);
    grace.leave(0);
    assert!(grace.passed(&later, None));
    assert!(!grace.inside(0) && !grace.inside(1));
}

/// Readers hold one of a few buffers inside a section; a writer replaces
/// it, waits the grace period, then poisons and reuses the old one.
#[test]
fn grace_keeps_buffers_until_readers_left() {
    const READERS: usize = 4;
    const BUFFERS: usize = 4;
    const WORDS: usize = 8;
    let grace = Arc::new(Grace::new(READERS + 2));
    let buffers: Arc<Vec<[AtomicU64; WORDS]>> = Arc::new(
        (0..BUFFERS)
            .map(|_| core::array::from_fn(|_| AtomicU64::new(1)))
            .collect(),
    );
    let current = Arc::new(AtomicUsize::new(0));
    let done = Arc::new(AtomicBool::new(false));
    let readers: Vec<_> = (0..READERS)
        .map(|pe| {
            let (grace, buffers, current, done) = (
                grace.clone(),
                buffers.clone(),
                current.clone(),
                done.clone(),
            );
            thread::spawn(move || {
                let mut reads = 0u64;
                while !done.load(Ordering::Relaxed) {
                    grace.enter(pe);
                    for _ in 0..16 {
                        let held = &buffers[current.load(Ordering::Acquire)];
                        let value = held[0].load(Ordering::Relaxed);
                        assert_ne!(value, u64::MAX, "a reader saw a poisoned buffer");
                        for _ in 0..4 {
                            for word in held {
                                assert_eq!(
                                    word.load(Ordering::Relaxed),
                                    value,
                                    "reused under a reader"
                                );
                            }
                        }
                        reads += 1;
                    }
                    // Readers alternate between leaving and passing a point.
                    if reads.is_multiple_of(64) {
                        grace.pass(pe);
                    }
                    grace.leave(pe);
                }
                reads
            })
        })
        .collect();
    // The last PE is parked throughout: it never holds anything up.
    let writer = READERS;
    grace.enter(writer);
    let mut value = 1;
    for _ in 0..20_000 {
        let old = current.load(Ordering::Relaxed);
        let new = (old + 1) % BUFFERS;
        value += 1;
        for word in &buffers[new] {
            word.store(value, Ordering::Relaxed);
        }
        current.store(new, Ordering::Release);
        let snapshot = grace.snapshot();
        assert!(
            grace.wait(writer, &snapshot, give_up()),
            "a grace period never passed"
        );
        for word in &buffers[old] {
            word.store(u64::MAX, Ordering::Relaxed);
        }
    }
    grace.leave(writer);
    done.store(true, Ordering::Relaxed);
    for reader in readers {
        assert!(reader.join().unwrap() > 0);
    }
}

#[test]
fn ring_moves_on_and_retires_one_ahead() {
    let grace = Grace::new(1);
    let ring = Ring::new(3, 256);
    // Three parts keep no tenured generation: its code is young.
    assert_eq!(ring.young_parts, 0);
    grace.enter(0);
    let first = ring.take(YOUNG, 200, 16).unwrap();
    // Each install publishes before the ring needs its part again.
    ring.release(first);
    assert_eq!(
        first,
        Place {
            generation: 0,
            part: 0,
            offset: 0
        }
    );
    let full = ring.head(YOUNG);
    assert_eq!(ring.take(TENURED, 100, 16), Err(full));
    // Parts 1 and 2 never held code: nothing to retire yet.
    assert_eq!(ring.advance(YOUNG, full, &grace, Some(0)), Advance::Moved);
    let place = ring.take(YOUNG, 64, 16).unwrap();
    ring.release(place);
    assert_eq!(
        place,
        Place {
            generation: 1,
            part: 1,
            offset: 0
        }
    );
    assert!(ring.live(first));
    // Opening the last free part chooses the oldest to go.
    let second = ring.head(YOUNG);
    assert_eq!(
        ring.advance(YOUNG, second, &grace, Some(0)),
        Advance::Retire(0, Why::Young)
    );
    assert!(!ring.live(first));
    let third = ring.head(YOUNG);
    let place = ring.take(YOUNG, 256, 16).unwrap();
    ring.release(place);
    assert_eq!(place.part, 2);
    // Until it is retired another advance waits for the retirer.
    assert_eq!(ring.advance(YOUNG, third, &grace, Some(0)), Advance::Busy);
    ring.retired(0, grace.snapshot());
    assert_eq!(ring.retired_through(0), 1);
    // PE 0 was inside at the snapshot: it waits unless it holds nothing.
    assert!(matches!(
        ring.advance(YOUNG, third, &grace, None),
        Advance::Wait(_)
    ));
    grace.pass(0);
    assert_eq!(
        ring.advance(YOUNG, third, &grace, None),
        Advance::Retire(1, Why::Young)
    );
    let place = ring.take(YOUNG, 16, 16).unwrap();
    ring.release(place);
    assert_eq!(
        place,
        Place {
            generation: 3,
            part: 0,
            offset: 0
        }
    );
    // An older caller's full part is stale: the ring moved already.
    assert_eq!(ring.advance(YOUNG, third, &grace, None), Advance::Moved);
    grace.leave(0);
}

/// Take what is left of `stream`'s open part in small places, published
/// at once, and give its head.
fn fill(ring: &Ring, stream: usize) -> u32 {
    while let Ok(place) = ring.take(stream, 16, 16) {
        ring.release(place);
    }
    ring.head(stream)
}

/// A part an install still writes may be chosen when nothing else can go,
/// but is not opened again until the install gives its place back: its
/// writes would land in the part's next code.
#[test]
fn a_part_an_install_still_writes_is_not_opened_again() {
    let grace = Grace::new(1);
    let ring = Ring::new(3, 256);
    let held = ring.take(YOUNG, 200, 16).unwrap();
    assert_eq!(
        ring.advance(YOUNG, fill(&ring, YOUNG), &grace, None),
        Advance::Moved
    );
    // Part 0 alone is settled: the last resort chooses it, writer and all.
    let full = fill(&ring, YOUNG);
    assert_eq!(
        ring.advance(YOUNG, full, &grace, None),
        Advance::Retire(0, Why::Young)
    );
    assert!(!ring.live(held));
    ring.retired(0, grace.snapshot());
    let full = fill(&ring, YOUNG);
    assert_eq!(ring.advance(YOUNG, full, &grace, None), Advance::Busy);
    ring.release(held);
    assert_eq!(
        ring.advance(YOUNG, full, &grace, None),
        Advance::Retire(1, Why::Young)
    );
    let place = ring.take(YOUNG, 16, 16).unwrap();
    assert_eq!((place.part, place.generation), (0, 3));
    ring.release(place);
}

/// A part with a writer is passed over while another can go.
#[test]
fn a_part_an_install_still_writes_goes_after_the_others() {
    let grace = Grace::new(1);
    let ring = Ring::new(4, 256);
    let held = ring.take(YOUNG, 200, 16).unwrap();
    for _ in 0..2 {
        assert_eq!(
            ring.advance(YOUNG, fill(&ring, YOUNG), &grace, None),
            Advance::Moved
        );
    }
    // Parts 0 and 1 are settled; 0, the oldest, is still written.
    let full = fill(&ring, YOUNG);
    assert_eq!(
        ring.advance(YOUNG, full, &grace, None),
        Advance::Retire(1, Why::Young)
    );
    assert!(ring.live(held));
    ring.release(held);
}

/// A reference into a part already chosen to go is refused, and noted
/// into a live part it holds that part.
#[test]
fn a_reference_into_a_part_already_chosen_is_refused() {
    let grace = Grace::new(1);
    let ring = Ring::new(3, 256);
    assert_eq!(
        ring.advance(YOUNG, fill(&ring, YOUNG), &grace, None),
        Advance::Moved
    );
    let full = fill(&ring, YOUNG);
    assert_eq!(
        ring.advance(YOUNG, full, &grace, None),
        Advance::Retire(0, Why::Young)
    );
    assert!(!ring.refer(1, 0), "into a part going");
    assert!(ring.refer(0, 1), "from a part going, into a full one");
    assert!(ring.refer(2, 2));
    ring.retired(0, grace.snapshot());
    assert!(!ring.refer(1, 0), "into a free part");
}

/// A tenured part that stayed open while young code turned the ring past
/// the age cap is old when it fills, but has only just filled: installs
/// that took its last bytes may still be publishing, so the advance that
/// closes it chooses another part (the audit's ring-level repro of
/// finding 1, where it chose that part at once).
#[test]
fn a_tenured_part_is_not_chosen_the_moment_it_fills() {
    const PART: usize = 256;
    let grace = Grace::new(1);
    let ring = Ring::new(8, PART);
    assert_eq!(ring.young_parts, 2);
    let step = |stream: usize, full: u32| match ring.advance(stream, full, &grace, None) {
        Advance::Retire(part, why) => {
            ring.retired(part, grace.snapshot());
            Some((part, why))
        }
        _ => None,
    };
    step(TENURED, ring.head(TENURED));
    let first = ring.take(TENURED, 16, 16).unwrap();
    ring.release(first);
    let tenured = first.part;
    while ring.generation() < ring.holds(tenured as usize) + AGE_TURNS * 8 {
        step(YOUNG, fill(&ring, YOUNG));
    }
    let in_flight = ring.take(TENURED, PART - 16, 16).unwrap();
    assert_eq!(in_flight.part, tenured);
    let full = ring.take(TENURED, 16, 16).unwrap_err();
    let chosen = step(TENURED, full).expect("a part chosen");
    assert_ne!(chosen.0, tenured, "the part just filled was chosen");
    assert!(ring.live(in_flight));
    ring.release(in_flight);
    // A quarter turn later it goes, by age.
    let mut outs = Vec::new();
    while !outs.iter().any(|&(part, _)| part == tenured) {
        outs.extend(step(YOUNG, fill(&ring, YOUNG)));
    }
    assert!(outs.contains(&(tenured, Why::Aged)));
}

/// PEs install records into a small ring and publish them in a table as
/// the engine publishes blocks: a record's place is taken (and the ring
/// moved on, a part retired or a grace waited for as needed), written, and
/// a thunk's place taken for it as an install holding the record's place
/// does (never waiting), and the record is published under the lock only
/// while its place's part is live ([`Ring::live`]), then the places are
/// released. Half the PEs install outside their section, as a compile
/// does. The others read published records, as cores run code, and check
/// that each still holds what its install wrote for that place: a record
/// published after its part's retirement, or written into a part opened
/// again, fails. Parts turn over thousands of times.
fn ring_reuses_parts_only_after_readers_left(parts: usize) {
    const PES: usize = 6;
    const PART: usize = 512;
    const RECORD: usize = 64;
    const THUNK: usize = 32;
    const PUBLISHED: usize = 32;
    let grace = Arc::new(Grace::new(PES));
    let ring = Arc::new(Ring::new(parts, PART));
    let arena: Arc<Vec<AtomicU64>> =
        Arc::new((0..parts * PART / 8).map(|_| AtomicU64::new(0)).collect());
    // Published places plus one; the lock is the retirer's and installers'.
    let table: Arc<Vec<AtomicU64>> = Arc::new((0..PUBLISHED).map(|_| AtomicU64::new(0)).collect());
    let lock = Arc::new(std::sync::Mutex::new(()));
    let waits = Arc::new(AtomicU32::new(0));
    let redone = Arc::new(AtomicU32::new(0));
    fn word(place: Place) -> usize {
        (place.part as usize * PART + place.offset as usize) / 8
    }
    fn pack(place: Place) -> u64 {
        u64::from(place.generation) << 40 | u64::from(place.part) << 32 | u64::from(place.offset)
    }
    fn unpack(packed: u64) -> Place {
        Place {
            generation: (packed >> 40) as u32,
            part: (packed >> 32) as u32 & 0xff,
            offset: packed as u32,
        }
    }
    // What a retirement does under the lock: its part's records go.
    fn retire(table: &[AtomicU64], lock: &std::sync::Mutex<()>, part: u32) {
        let _held = lock.lock().unwrap();
        for entry in table.iter() {
            let held = entry.load(Ordering::Relaxed);
            if held != 0 && unpack(held - 1).part == part {
                entry.store(0, Ordering::Relaxed);
            }
        }
    }
    let threads: Vec<_> = (0..PES)
        .map(|pe| {
            let (grace, ring, arena, table, lock, waits, redone) = (
                grace.clone(),
                ring.clone(),
                arena.clone(),
                table.clone(),
                lock.clone(),
                waits.clone(),
                redone.clone(),
            );
            thread::spawn(move || {
                let mut seed = pe as u64 * 0x9e37_79b9 + 1;
                let mut installed = 0u64;
                let outside = pe % 3 == 0;
                let write = |place: Place, bytes: usize| {
                    let first = word(place);
                    for index in first..first + bytes / 8 {
                        arena[index].store(pack(place) | (pe as u64) << 56, Ordering::Relaxed);
                    }
                };
                grace.enter(pe);
                for _ in 1..=20_000u64 {
                    if outside {
                        grace.leave(pe);
                    }
                    let skip = (!outside).then_some(pe);
                    // Tenured records are few: their parts stay open long
                    // and are old when they fill, as the guest's hot code.
                    let stream = match pe % 2 == 1 && next(&mut seed).is_multiple_of(16) {
                        true => TENURED,
                        false => YOUNG,
                    };
                    // Install a record.
                    let place = loop {
                        match ring.take(stream, RECORD, 16) {
                            Ok(place) => break place,
                            Err(full) => match ring.advance(stream, full, &grace, skip) {
                                Advance::Moved => {}
                                Advance::Busy => thread::yield_now(),
                                Advance::Retire(part, _) => {
                                    retire(&table, &lock, part);
                                    ring.retired(part, grace.snapshot());
                                }
                                Advance::Wait(snapshot) => {
                                    waits.fetch_add(1, Ordering::Relaxed);
                                    let mut give = give_up();
                                    let passed = match outside {
                                        true => loop {
                                            if grace.passed(&snapshot, None) {
                                                break true;
                                            }
                                            if give() {
                                                break false;
                                            }
                                        },
                                        false => grace.wait(pe, &snapshot, give),
                                    };
                                    assert!(passed, "a part's grace never passed");
                                }
                            },
                        }
                    };
                    // All but its last word, which the jump to its thunk
                    // patches after the thunk is installed.
                    write(place, RECORD - 8);
                    // Its thunk, which may close the record's part and
                    // choose a part to go, but waits for nothing.
                    let mut thunk = None;
                    if next(&mut seed).is_multiple_of(2) {
                        for _ in 0..2 {
                            match ring.take(stream, THUNK, 16) {
                                Ok(taken) => {
                                    write(taken, THUNK);
                                    thunk = Some(taken);
                                    break;
                                }
                                Err(full) => match ring.advance(stream, full, &grace, skip) {
                                    Advance::Moved => {}
                                    Advance::Retire(part, _) => {
                                        retire(&table, &lock, part);
                                        ring.retired(part, grace.snapshot());
                                    }
                                    Advance::Busy | Advance::Wait(_) => break,
                                },
                            }
                        }
                    }
                    for _ in 0..next(&mut seed) % 64 {
                        core::hint::spin_loop();
                    }
                    arena[word(place) + RECORD / 8 - 1]
                        .store(pack(place) | (pe as u64) << 56, Ordering::Relaxed);
                    {
                        let _held = lock.lock().unwrap();
                        if ring.live(place) {
                            let slot = next(&mut seed) as usize % PUBLISHED;
                            table[slot].store(pack(place) + 1, Ordering::Release);
                            installed += 1;
                        } else {
                            redone.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    ring.release(place);
                    if let Some(thunk) = thunk {
                        ring.release(thunk);
                    }
                    if outside {
                        grace.enter(pe);
                    }
                    // Run a few published records.
                    for _ in 0..4 {
                        let held =
                            table[next(&mut seed) as usize % PUBLISHED].load(Ordering::Acquire);
                        if held == 0 {
                            continue;
                        }
                        let place = unpack(held - 1);
                        let first = word(place);
                        let tag = arena[first].load(Ordering::Relaxed);
                        assert_eq!(
                            tag & ((1 << 56) - 1),
                            pack(place),
                            "a published record holds another install's bytes"
                        );
                        for _ in 0..64 {
                            for index in first..first + RECORD / 8 {
                                assert_eq!(
                                    arena[index].load(Ordering::Relaxed),
                                    tag,
                                    "code reused under a reader"
                                );
                            }
                        }
                    }
                    grace.pass(pe);
                }
                grace.leave(pe);
                installed
            })
        })
        .collect();
    let installed: u64 = threads
        .into_iter()
        .map(|thread| thread.join().unwrap())
        .sum();
    assert!(installed > 0);
    // Six PEs over a few parts of 8 records each turn the ring constantly.
    assert!(ring.generation() as usize > 10 * parts);
    std::eprintln!(
        "ring of {parts}: {} generations, {installed} installs, {} redone, {} waits",
        ring.generation(),
        redone.load(Ordering::Relaxed),
        waits.load(Ordering::Relaxed)
    );
}

#[test]
fn ring_reuses_parts_only_after_readers_left_and_publishes_only_live_parts() {
    ring_reuses_parts_only_after_readers_left(MIN_PARTS);
}

/// The same with a tenured generation, whose parts stay open long and are
/// old when they fill.
#[test]
fn a_tenured_ring_reuses_parts_only_after_readers_left_and_publishes_only_live_parts() {
    ring_reuses_parts_only_after_readers_left(8);
}

#[test]
fn slots_wait_in_limbo_and_never_block() {
    let grace = Grace::new(2);
    let slots = Slots::new(4);
    grace.enter(0);
    grace.enter(1);
    let taken: Vec<u32> = (0..2)
        .map(|_| slots.take(&grace, Some(0)).unwrap())
        .collect();
    assert_eq!(taken, [0, 1]);
    slots.release(taken, grace.snapshot());
    // PE 1 stays inside: new numbers are given instead, up to the capacity.
    assert_eq!(slots.take(&grace, Some(0)), Some(2));
    assert_eq!(slots.take(&grace, Some(0)), Some(3));
    assert_eq!(slots.take(&grace, Some(0)), None);
    assert_eq!(slots.issued(), 4);
    grace.pass(1);
    let mut again = [slots.take(&grace, Some(0)), slots.take(&grace, Some(0))];
    again.sort();
    assert_eq!(again, [Some(0), Some(1)]);
    grace.leave(1);
    grace.leave(0);
}

/// PEs take slots, write their records, publish them and later release
/// them; the others read published slots' records and check they stay
/// unchanged while held.
#[test]
fn slots_reused_only_after_readers_left() {
    const PES: usize = 6;
    const CAPACITY: u32 = 256;
    const PUBLISHED: usize = 16;
    let grace = Arc::new(Grace::new(PES + 1));
    let slots = Arc::new(Slots::new(CAPACITY));
    let records: Arc<Vec<AtomicU64>> = Arc::new((0..CAPACITY).map(|_| AtomicU64::new(0)).collect());
    let table: Arc<Vec<AtomicU32>> = Arc::new((0..PUBLISHED).map(|_| AtomicU32::new(0)).collect());
    let threads: Vec<_> = (0..PES)
        .map(|pe| {
            let (grace, slots, records, table) =
                (grace.clone(), slots.clone(), records.clone(), table.clone());
            thread::spawn(move || {
                let mut seed = pe as u64 * 0x2545_f491 + 7;
                let mut full = 0;
                grace.enter(pe);
                for sequence in 1..=20_000u64 {
                    if let Some(slot) = slots.take(&grace, Some(pe)) {
                        records[slot as usize]
                            .store((pe as u64) << 48 | sequence, Ordering::Relaxed);
                        let old = table[next(&mut seed) as usize % PUBLISHED]
                            .swap(slot + 1, Ordering::AcqRel);
                        if old != 0 {
                            slots.release(alloc::vec![old - 1], grace.snapshot());
                        }
                    } else {
                        full += 1;
                    }
                    for _ in 0..2 {
                        let held =
                            table[next(&mut seed) as usize % PUBLISHED].load(Ordering::Acquire);
                        if held == 0 {
                            continue;
                        }
                        let record = &records[held as usize - 1];
                        let value = record.load(Ordering::Relaxed);
                        for _ in 0..256 {
                            assert_eq!(
                                record.load(Ordering::Relaxed),
                                value,
                                "a slot reused under a reader"
                            );
                        }
                    }
                    if sequence.is_multiple_of(16) {
                        grace.pass(pe);
                    }
                }
                grace.leave(pe);
                full
            })
        })
        .collect();
    for thread in threads {
        thread.join().unwrap();
    }
    // The published slots and those in limbo stay within the capacity.
    assert!(slots.issued() <= CAPACITY);
}

/// PEs take slots, fill their records and publish them; readers on other
/// threads find records whole as chunks are allocated beside them.
#[test]
fn records_are_read_without_locks_while_others_grow() {
    #[derive(Default)]
    struct Record {
        tag: AtomicU64,
        check: AtomicU64,
    }
    const PES: usize = 4;
    const CAPACITY: u32 = 64 * 1024;
    let grace = Arc::new(Grace::new(PES));
    let slots = Arc::new(Slots::new(CAPACITY));
    let records = Arc::new(Records::<Record>::new(CAPACITY));
    let published: Arc<Vec<AtomicU32>> = Arc::new((0..256).map(|_| AtomicU32::new(0)).collect());
    assert!(records.get(0).is_none());
    let threads: Vec<_> = (0..PES)
        .map(|pe| {
            let (grace, slots, records, published) = (
                grace.clone(),
                slots.clone(),
                records.clone(),
                published.clone(),
            );
            thread::spawn(move || {
                let mut seed = pe as u64 + 11;
                grace.enter(pe);
                for sequence in 1..=12_000u64 {
                    let Some(slot) = slots.take(&grace, Some(pe)) else {
                        break;
                    };
                    let record = records.reserve(slot);
                    let tag = (pe as u64) << 48 | sequence;
                    record.tag.store(tag, Ordering::Relaxed);
                    record.check.store(!tag, Ordering::Relaxed);
                    published[next(&mut seed) as usize % 256].store(slot + 1, Ordering::Release);
                    let held = published[next(&mut seed) as usize % 256].load(Ordering::Acquire);
                    if held != 0 {
                        let record = records.get(held - 1).expect("a published slot's chunk");
                        let tag = record.tag.load(Ordering::Relaxed);
                        assert_eq!(record.check.load(Ordering::Relaxed), !tag);
                    }
                }
                grace.leave(pe);
            })
        })
        .collect();
    for thread in threads {
        thread.join().unwrap();
    }
    // Nothing was released: every slot is new, 48,000 of them.
    assert_eq!(slots.issued(), 48_000);
    assert!(records.heap_bytes() >= 48_000 * size_of::<Record>());
}

#[test]
fn the_code_lock_counts_its_waits_and_the_report_starts_the_longest_over() {
    // The waiter announces itself before it asks for the lock, so a loaded
    // host can hold it off until the lock is free again and its take goes
    // uncontended. Such a round proves nothing; run a fresh one, boundedly.
    let (code, report) = (0..20)
        .map(|_| {
            let code = Arc::new(Code::new(1 << 16, 1 << 16, None));
            let held = code.cache();
            let waiting = Arc::clone(&code);
            let started = Arc::new(AtomicBool::new(false));
            let seen = Arc::clone(&started);
            let waiter = thread::spawn(move || {
                seen.store(true, Ordering::Release);
                drop(waiting.cache());
            });
            while !started.load(Ordering::Acquire) {
                thread::yield_now();
            }
            thread::sleep(Duration::from_millis(20));
            drop(held);
            waiter.join().unwrap();
            let report = code.report();
            // The holder, the waiter and the report's own.
            assert_eq!(report.lock.acquired, 3);
            assert!(report.lock.contended <= 1);
            (code, report)
        })
        .find(|(_, report)| report.lock.contended == 1)
        .expect("the waiter met the held lock in one of 20 rounds");
    assert!(report.lock.longest > 0 && report.lock.longest == report.lock.waited);
    let again = code.report();
    assert_eq!(again.lock.acquired, 4);
    assert_eq!(again.lock.longest, 0, "the longest wait is per report");
    assert_eq!(again.lock.waited, report.lock.waited);
}
