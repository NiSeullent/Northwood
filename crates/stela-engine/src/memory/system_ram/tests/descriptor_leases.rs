// SPDX-License-Identifier: GPL-2.0-or-later
use super::*;
use crate::memory::system_ram::descriptor_frames::{DescriptorFrames, FRAME_BYTES};
use std::sync::mpsc;
use stela_contracts::gate::MemoryRange;

const FIRST: u64 = 0x8000;
const NEXT: u64 = FIRST + FRAME_BYTES as u64;
const SAME_STRIPE: u64 = FIRST + 64 * FRAME_BYTES as u64;

#[repr(align(16384))]
struct FrameStorage(UnsafeCell<[u8; 4 * FRAME_BYTES]>);

// SAFETY: the fixture publishes its bytes only through the canonical RAM owner.
unsafe impl Sync for FrameStorage {}

fn ram(tracked: bool, bytes: usize) -> RegisteredSystemRam {
    ram_with_layout(tracked, &[(FIRST, 0, bytes)])
}

fn ram_with_layout(tracked: bool, layout: &[(u64, usize, usize)]) -> RegisteredSystemRam {
    let storage = Arc::new(FrameStorage(UnsafeCell::new([0x53; 4 * FRAME_BYTES])));
    let ranges: Vec<_> = layout
        .iter()
        .map(|&(source_pa, offset, bytes)| {
            assert!(offset <= 4 * FRAME_BYTES && bytes <= 4 * FRAME_BYTES - offset);
            ReservedSystemRamRange {
                source_pa,
                host_pa: 0x1000_0000 + offset as u64,
                // SAFETY: the fixture checks every extent against the allocation.
                host_base: NonNull::new(unsafe { storage.0.get().cast::<u8>().add(offset) })
                    .unwrap(),
                bytes,
            }
        })
        .collect();
    // SAFETY: the initialized allocation and stable address stay owned by RAM.
    let mut ram = unsafe { fixture(&ranges, storage) };
    if tracked {
        ram.enable_descriptor_frame_leases().unwrap();
    }
    ram
}

fn dma_page(address: u64) -> MemoryRange {
    MemoryRange {
        address,
        bytes: 4096,
    }
}

#[test]
fn real_descriptor_reads_and_updates_release_on_success_and_error() {
    let ram = ram(true, 2 * FRAME_BYTES);
    let mut reader = ram.descriptor_reader();
    let original = u64::from_le_bytes([0x53; 8]);
    assert_eq!(reader.read_descriptor(NEXT - 8), Ok(original));
    assert_eq!(reader.read_descriptor(NEXT), Ok(original));
    assert_eq!(reader.update_descriptor(NEXT, original, 17), Ok(true));
    assert_eq!(reader.update_descriptor(NEXT, original, 42), Ok(false));
    assert_eq!(reader.read_descriptor(NEXT), Ok(17));
    assert_eq!(
        reader.read_descriptor(NEXT + 1),
        Err(MemoryStatus::InvalidRequest)
    );
    assert_eq!(
        reader.read_descriptor(NEXT + FRAME_BYTES as u64),
        Err(MemoryStatus::Translation)
    );
    let first = ram.try_close_descriptor_frame(FIRST).unwrap().unwrap();
    assert_eq!(first.physical(), FIRST);
    assert!(ram.try_close_descriptor_frame(NEXT).unwrap().is_none());
    drop(first);
    let next = ram.try_close_descriptor_frame(NEXT).unwrap().unwrap();
    assert_eq!(next.physical(), NEXT);
    assert_eq!(reader.read_descriptor(NEXT), Err(MemoryStatus::Domain));
    assert!(ram.try_service().unwrap().is_none());
    drop(next);
    assert_eq!(reader.read_descriptor(NEXT), Ok(17));
}

#[test]
fn in_flight_descriptor_prevents_close_until_the_access_returns() {
    let ram = ram(true, 2 * FRAME_BYTES);
    std::thread::scope(|scope| {
        let (ready, waiting) = mpsc::channel();
        let (release, released) = mpsc::channel();
        let worker = scope.spawn({
            let ram = &ram;
            move || {
                // The canonical reader and updater acquire this same guard
                // before resolving the backing pointer.
                let pin = ram.pin_descriptor(NEXT - 8).unwrap().unwrap();
                ready.send(()).unwrap();
                let _ = released.recv();
                drop(pin);
            }
        });
        waiting.recv().unwrap();
        assert!(ram.try_close_descriptor_frame(FIRST).unwrap().is_none());
        // A refused close must return the writer freeze before the updater
        // can finish, and it must not close an adjacent frame's stripe.
        assert!(ram.try_service().unwrap().is_some());
        assert!(ram.try_close_descriptor_frame(NEXT).unwrap().is_some());
        release.send(()).unwrap();
        worker.join().unwrap();
        let closed = ram.try_close_descriptor_frame(FIRST).unwrap().unwrap();
        assert!(matches!(
            ram.pin_descriptor(NEXT - 8),
            Err(MemoryStatus::Domain)
        ));
        drop(closed);
        assert!(ram.pin_descriptor(NEXT - 8).unwrap().is_some());
    });
}

#[test]
fn untracked_or_partial_registrations_cannot_close_a_frame() {
    let disabled = ram(false, 2 * FRAME_BYTES);
    assert!(matches!(
        disabled.try_close_descriptor_frame(FIRST),
        Err(MemoryStatus::Unsupported)
    ));
    let partial = ram(true, FRAME_BYTES + 16);
    assert!(matches!(
        partial.try_close_descriptor_frame(FIRST + 8),
        Err(MemoryStatus::Alignment)
    ));
    assert!(matches!(
        partial.try_close_descriptor_frame(NEXT),
        Err(MemoryStatus::Translation)
    ));
    let native = partial.reference_execution_loan().unwrap();
    assert!(partial.try_close_descriptor_frame(FIRST).unwrap().is_none());
    drop(native);
    assert!(partial.try_close_descriptor_frame(FIRST).unwrap().is_some());
}

#[test]
fn shared_stripe_refuses_until_every_pin_returns() {
    let frames = DescriptorFrames::new(0).unwrap();
    let first = frames.try_pin(FIRST).unwrap();
    let second = frames.try_pin(FIRST + 8).unwrap();
    drop(first);
    assert!(frames.try_close(SAME_STRIPE).is_none());
    drop(second);
    let closed = frames.try_close(SAME_STRIPE).unwrap();
    assert!(frames.try_pin(FIRST).is_none());
    assert!(frames.try_close(SAME_STRIPE).is_none());
    drop(closed);
    assert!(frames.try_pin(FIRST).is_some());
}

#[test]
fn spanning_copy_and_cas_refuse_closed_frames_before_any_effect() {
    let ram = ram(true, 2 * FRAME_BYTES);
    let closed = ram.try_close_descriptor_frame(NEXT).unwrap().unwrap();
    let start = NEXT - 8;
    let mut output = [0xa1; 16];
    let mut expected = [0x53; 8];
    // SAFETY: all caller buffers are private and this owns the fixture RAM.
    let (read, write, exchange) = unsafe {
        (
            ram.read_physical_progress(start, &mut output).unwrap_err(),
            ram.publish_physical(start, &[0x92; 16]).unwrap_err(),
            ram.compare_exchange_physical(NEXT, &mut expected, &[0x11; 8]),
        )
    };
    assert_eq!((read.status, read.completed), (MemoryStatus::Domain, 0));
    assert_eq!((write.status, write.completed), (MemoryStatus::Domain, 0));
    assert_eq!(exchange, Err(MemoryStatus::Domain));
    assert_eq!(output, [0xa1; 16]);
    assert_eq!(expected, [0x53; 8]);
    drop(closed);
    // A failed multi-stripe acquisition must return the first pin too.
    assert!(ram.try_close_descriptor_frame(FIRST).unwrap().is_some());
    // SAFETY: same private buffers and fixture RAM.
    unsafe {
        ram.read_physical(start, &mut output).unwrap();
        assert_eq!(output, [0x53; 16]);
        ram.publish_physical(start, &[0x92; 16]).unwrap();
        ram.read_physical(start, &mut output).unwrap();
    }
    assert_eq!(output, [0x92; 16]);
}

#[test]
fn a_wired_copy_reports_the_exact_prefix_when_a_later_extent_closes() {
    let ram = Arc::new(ram(true, 2 * FRAME_BYTES));
    let wires = [dma_page(FIRST), dma_page(NEXT)];
    let wired = WiredRam::new(Arc::clone(&ram), &wires).unwrap();
    // Close the later stripe in the fixture without holding the global
    // writer gate, so the first extent can still be copied and examined.
    let frames = ram.descriptor_frames.as_ref().unwrap();
    let later = frames.try_close(NEXT).unwrap();
    let mut output = [0xa1; 16];
    // SAFETY: these private bytes and the fixture remain alive throughout.
    let (read, write) = unsafe {
        (
            wired.read_into(4096 - 8, &mut output).unwrap_err(),
            wired.publish_from(4096 - 8, &[0x92; 16]).unwrap_err(),
        )
    };
    assert_eq!((read.status, read.completed), (MemoryStatus::Domain, 8));
    assert_eq!((write.status, write.completed), (MemoryStatus::Domain, 8));
    assert_eq!(output[..8], [0x53; 8]);
    assert_eq!(output[8..], [0xa1; 8]);
    drop(later);
    // SAFETY: same private bytes and fixture.
    unsafe { wired.read_into(4096 - 8, &mut output) }.unwrap();
    assert_eq!(output[..8], [0x92; 8]);
    assert_eq!(output[8..], [0x53; 8]);
}

#[test]
fn in_flight_copy_releases_the_frame_and_busy_close_releases_writers() {
    let ram = ram(true, 2 * FRAME_BYTES);
    std::thread::scope(|scope| {
        let (ready, waiting) = mpsc::channel();
        let (release, released) = mpsc::channel();
        let worker = scope.spawn({
            let ram = &ram;
            move || {
                let held = ram.pin_range(NEXT - 8, 16).unwrap().unwrap();
                ready.send(()).unwrap();
                released.recv().unwrap();
                // SAFETY: source bytes are private and the fixture owns RAM.
                unsafe { ram.publish_physical(NEXT - 8, &[0x82; 16]) }.unwrap();
                drop(held);
            }
        });
        waiting.recv().unwrap();
        assert!(ram.try_close_descriptor_frame(FIRST).unwrap().is_none());
        assert!(ram.try_close_descriptor_frame(NEXT).unwrap().is_none());
        assert!(ram.try_service().unwrap().is_some());
        release.send(()).unwrap();
        worker.join().unwrap();
        let closed = ram.try_close_descriptor_frame(NEXT).unwrap().unwrap();
        let mut snapshot = [0; FRAME_BYTES];
        // SAFETY: snapshot is private and is not backed by the registered RAM.
        unsafe { closed.copy_into(&mut snapshot) }.unwrap();
        assert_eq!(snapshot[..8], [0x82; 8]);
        assert_eq!(snapshot[8..], [0x53; FRAME_BYTES - 8]);
    });
}

#[test]
fn raw_dma_exports_wire_validated_ranges_without_blocking_cpu_access() {
    let ram = ram(true, 2 * FRAME_BYTES);
    let ranges = [dma_page(NEXT)];
    for _ in 0..2 {
        let mut loan = ram.prepare_external_dma(&ranges).unwrap();
        loan.begin_publication();
    }
    assert!(ram.try_close_descriptor_frame(FIRST).unwrap().is_some());
    assert!(ram.try_close_descriptor_frame(NEXT).unwrap().is_none());
    assert_eq!(ram.read_descriptor(NEXT), Ok(u64::from_le_bytes([0x53; 8])));
}

#[test]
fn cpu_and_dma_resolve_exact_registered_runs_in_the_same_generation() {
    const FRAME: u64 = FRAME_BYTES as u64;
    const HOST: u64 = 0x1000_0000;
    const FAR: u64 = NEXT + 2 * FRAME;
    let ram = ram_with_layout(
        true,
        &[
            (FIRST, 2 * FRAME_BYTES, FRAME_BYTES),
            (NEXT, 0, FRAME_BYTES),
            (FAR, 3 * FRAME_BYTES, FRAME_BYTES),
        ],
    );
    let initial = u64::from_le_bytes([0x53; 8]);
    assert_eq!(ram.update_descriptor(NEXT - 8, initial, 17), Ok(true));
    assert_eq!(ram.update_descriptor(NEXT, initial, 29), Ok(true));
    let mut descriptors = ram.descriptor_reader();
    assert_eq!(descriptors.read_descriptor(NEXT - 8), Ok(17));

    let ranges = [
        dma_page(FAR + 4096),
        MemoryRange {
            address: NEXT - 4096,
            bytes: 8192,
        },
    ];
    let dma = ram.prepare_external_dma(&ranges).unwrap();
    {
        let execution = ram.reference_execution_loan().unwrap();
        let cpu = ram.cpu_binding(&execution).unwrap();
        assert_eq!(cpu.generation(), ram.generation().unwrap());
        assert_eq!(dma.generation(), cpu.generation());
        assert_eq!(
            cpu.physical_run(NEXT - 4096, 8192),
            Ok(Some((HOST + 3 * FRAME - 4096, 4096)))
        );
        assert_eq!(cpu.physical_run(NEXT, 4096), Ok(Some((HOST, 4096))));
        assert_eq!(
            cpu.physical_run(NEXT + FRAME - 8, 16),
            Ok(Some((HOST + FRAME - 8, 8)))
        );
        assert_eq!(cpu.physical_run(NEXT + FRAME, 4096), Ok(None));

        let mut runs = Vec::new();
        dma.visit_physical_runs(|host, bytes| runs.push((host, bytes)))
            .unwrap();
        assert_eq!(
            runs,
            [
                (HOST + 3 * FRAME + 4096, 4096),
                (HOST + 3 * FRAME - 4096, 4096),
                (HOST, 4096),
            ]
        );
        assert_eq!(descriptors.read_descriptor(NEXT - 8), Ok(17));
        assert_eq!(descriptors.read_descriptor(NEXT), Ok(29));
        assert_eq!(ram.update_descriptor(NEXT, 29, 41), Ok(true));
        assert_eq!(descriptors.read_descriptor(NEXT), Ok(41));
    }
    for address in [FIRST, NEXT, FAR] {
        assert!(ram.try_close_descriptor_frame(address).unwrap().is_none());
    }
    drop(dma);
    for address in [FIRST, NEXT, FAR] {
        assert!(ram.try_close_descriptor_frame(address).unwrap().is_some());
    }
}

#[test]
fn a_dma_range_crossing_a_hole_leaves_registered_frames_unpinned() {
    const AFTER_HOLE: u64 = NEXT + FRAME_BYTES as u64;
    let ram = ram_with_layout(
        true,
        &[
            (FIRST, 0, FRAME_BYTES),
            (AFTER_HOLE, FRAME_BYTES, FRAME_BYTES),
        ],
    );
    let ranges = [
        dma_page(AFTER_HOLE),
        MemoryRange {
            address: NEXT - 4096,
            bytes: FRAME_BYTES as u64 + 8192,
        },
    ];
    assert!(matches!(
        ram.prepare_external_dma(&ranges),
        Err(MemoryStatus::Translation)
    ));
    for address in [FIRST, AFTER_HOLE] {
        assert!(ram.try_close_descriptor_frame(address).unwrap().is_some());
    }
}

#[test]
fn cpu_binding_rejects_a_foreign_owner_at_the_same_generation() {
    let first = ram(true, FRAME_BYTES);
    let other = ram(true, FRAME_BYTES);
    let execution = first.reference_execution_loan().unwrap();
    assert_eq!(
        first.cpu_binding(&execution).unwrap().generation(),
        other.generation().unwrap()
    );
    assert!(matches!(
        other.cpu_binding(&execution),
        Err(MemoryStatus::Domain)
    ));
}

#[test]
fn invalid_later_dma_range_leaves_the_valid_prefix_closable() {
    let ram = ram(true, 2 * FRAME_BYTES);
    let missing = dma_page(NEXT + FRAME_BYTES as u64);
    let overflow = MemoryRange {
        address: u64::MAX - 4095,
        bytes: 8192,
    };
    for invalid in [missing, overflow] {
        assert!(matches!(
            ram.prepare_external_dma(&[dma_page(FIRST), invalid]),
            Err(MemoryStatus::Translation)
        ));
        assert!(ram.try_close_descriptor_frame(FIRST).unwrap().is_some());
        assert!(ram.try_close_descriptor_frame(NEXT).unwrap().is_some());
    }
}

#[test]
fn dma_prepare_rolls_back_earlier_stripes_when_a_later_stripe_is_closed() {
    let ram = ram(true, 2 * FRAME_BYTES);
    let ranges = [dma_page(FIRST), dma_page(NEXT)];
    let service = ram.try_service().unwrap().unwrap();
    let closed = service.try_close_frame(NEXT).unwrap().unwrap();
    assert!(matches!(
        ram.prepare_external_dma(&ranges),
        Err(MemoryStatus::Domain)
    ));
    assert!(service.try_close_frame(FIRST).unwrap().is_some());
    drop(closed);
    drop(service);
    drop(ram.prepare_external_dma(&ranges).unwrap());
}

#[test]
fn unstarted_and_confirmed_cancelled_dma_exports_release_every_frame() {
    let ram = ram(true, 2 * FRAME_BYTES);
    let ranges = [dma_page(FIRST), dma_page(NEXT)];
    let loan = ram.prepare_external_dma(&ranges).unwrap();
    assert!(ram.try_close_descriptor_frame(FIRST).unwrap().is_none());
    assert!(ram.try_close_descriptor_frame(NEXT).unwrap().is_none());
    drop(loan);
    assert!(ram.try_close_descriptor_frame(FIRST).unwrap().is_some());
    assert!(ram.try_close_descriptor_frame(NEXT).unwrap().is_some());

    {
        let mut loan = ram.prepare_external_dma(&ranges).unwrap();
        loan.begin_publication();
        // SAFETY: the test installed no CPU mapping and submitted no DMA.
        unsafe { loan.cancel_publication() };
    }
    assert!(ram.try_close_descriptor_frame(FIRST).unwrap().is_some());
    assert!(ram.try_close_descriptor_frame(NEXT).unwrap().is_some());
}

#[test]
fn unconfirmed_dma_publication_retains_only_the_requested_frames() {
    let ram = ram_with_layout(
        true,
        &[
            (FIRST, 0, 2 * FRAME_BYTES),
            (SAME_STRIPE, 2 * FRAME_BYTES, 2 * FRAME_BYTES),
        ],
    );
    let last = SAME_STRIPE + FRAME_BYTES as u64;
    let ranges = [dma_page(FIRST + 4096), dma_page(last + 4096)];
    let mut loan = ram.prepare_external_dma(&ranges).unwrap();
    loan.begin_publication();
    // Leaving a publication attempt without a confirmed rollback retains the frames.
    drop(loan);

    assert!(ram.try_close_descriptor_frame(FIRST).unwrap().is_none());
    assert!(ram.try_close_descriptor_frame(last).unwrap().is_none());
    assert!(ram.try_close_descriptor_frame(NEXT).unwrap().is_some());
    assert!(
        ram.try_close_descriptor_frame(SAME_STRIPE)
            .unwrap()
            .is_some()
    );
}

#[test]
fn partial_frame_publication_never_reopens_nonresident_admission() {
    const ORIGINAL: [u8; FRAME_BYTES] = [0x53; FRAME_BYTES];
    const OVERWRITTEN: [u8; FRAME_BYTES] = [0x92; FRAME_BYTES];
    for restore in [false, true] {
        let ram = ram_with_layout(
            true,
            &[(FIRST, 0, FRAME_BYTES), (NEXT, FRAME_BYTES, FRAME_BYTES)],
        );
        let closed = ram.try_close_descriptor_frame(FIRST).unwrap().unwrap();
        // SAFETY: the fixture has no raw aliases or external DMA owner.
        let mut frame = unsafe { closed.mark_nonresident() }.unwrap();
        if restore {
            // SAFETY: the source is private and the fixture keeps the frame held.
            assert!(unsafe { frame.try_overwrite_backing(&OVERWRITTEN) }.unwrap());
        }
        let (region, _) = ram.region(FIRST).unwrap();
        // Exactly one 8-byte source publication can advance this generation.
        ram.system
            .seed_source_mutation_for_test(&region.backing, u64::MAX - 1);

        // SAFETY: both sources are private and disjoint from the fixture, and
        // ORIGINAL matches the fixture's bytes before the frame was sealed.
        let error = unsafe {
            if restore {
                frame.try_restore_backing(&ORIGINAL)
            } else {
                frame.try_overwrite_backing(&OVERWRITTEN)
            }
        }
        .unwrap_err();
        assert_eq!((error.status, error.completed), (MemoryStatus::Domain, 8));
        assert_eq!(ram.nonresident_frame(FIRST, FRAME_BYTES), Some(FIRST));
        assert!(matches!(
            ram.reference_execution_loan(),
            Err(MemoryStatus::Domain)
        ));

        let mut output = [0xa5; 16];
        // SAFETY: the output is private and the fixture keeps the backing live.
        let read = unsafe { ram.read_physical_progress(FIRST, &mut output) }.unwrap_err();
        assert_eq!(
            (read.status, read.completed),
            (MemoryStatus::Nonresident, 0)
        );
        assert_eq!(output, [0xa5; 16]);

        let mut observed = [0; FRAME_BYTES];
        {
            let frames = ram.descriptor_frames.as_ref().unwrap();
            let _stripe = frames.try_close(FIRST).unwrap();
            // SAFETY: the live frame keeps its writer service, the exact
            // stripe is closed, and the output is private.
            unsafe { ram.read_physical_unpinned(FIRST, &mut observed) }.unwrap();
        }
        let (written, previous) = if restore {
            (&ORIGINAL, &OVERWRITTEN)
        } else {
            (&OVERWRITTEN, &ORIGINAL)
        };
        assert_eq!(observed[..8], written[..8]);
        assert_eq!(observed[8..], previous[8..]);
        // SAFETY: the destination is private and NEXT is a separate backing.
        unsafe { ram.read_physical(NEXT, &mut observed) }.unwrap();
        assert_eq!(observed, ORIGINAL);

        drop(frame);
        assert_eq!(ram.nonresident_frame(FIRST, FRAME_BYTES), Some(FIRST));
        assert_eq!(ram.domain().check_execution(), Err(MemoryStatus::Domain));
    }
}
