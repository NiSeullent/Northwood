use super::*;
use core::cell::UnsafeCell;
use stela_contracts::gate::MemoryRange;

#[repr(C, align(16))]
struct Storage(UnsafeCell<[u8; 128]>);
// SAFETY: tests use no Rust references into these bytes once registered;
// all reached accesses use the existing scalar native access/publication.
unsafe impl Sync for Storage {}
impl Storage {
    fn new() -> Arc<Self> {
        Arc::new(Self(UnsafeCell::new([0x53; 128])))
    }
    fn pointer(&self) -> NonNull<u8> {
        NonNull::new(self.0.get().cast::<u8>()).unwrap()
    }
}

fn range(storage: &Storage, source_pa: u64, offset: usize, bytes: usize) -> ReservedSystemRamRange {
    ReservedSystemRamRange {
        source_pa,
        host_pa: 0x1000_0000 + offset as u64,
        // SAFETY: fixture geometry stays inside the test allocation.
        host_base: unsafe { NonNull::new_unchecked(storage.pointer().as_ptr().add(offset)) },
        bytes,
    }
}

// A RAM ownership/access model using the real POSIX platform's target
// admission. It does not manufacture a CPL0 native-system proof or assert
// freestanding/AP execution, which has its own boot validation.
unsafe fn model(ranges: &[ReservedSystemRamRange], owner: Arc<Storage>) -> RegisteredSystemRam {
    validate(ranges).unwrap();
    // SAFETY: the caller validated these ranges and `owner` keeps their storage.
    let (regions, views, system) = unsafe { bindings(ranges, owner, 64, 8) }.unwrap();
    let domain = MemoryDomain::new(Arc::clone(&system), views, TargetCpu::Baseline).unwrap();
    RegisteredSystemRam {
        regions,
        domain,
        system,
        target: TargetCpu::Baseline,
        erg_bytes: 64,
        descriptor_frames: None,
    }
}

mod descriptor_leases;

#[test]
fn a_wired_buffer_copies_in_logical_order_across_fragmented_ram() {
    let storage = Storage::new();
    let layout = [
        range(&storage, 0x4000, 0, 64),
        range(&storage, 0x8000, 64, 64),
    ];
    // SAFETY: storage remains with the registration and has no other writer.
    let ram = Arc::new(unsafe { model(&layout, storage) });
    let wired = WiredRam::new(
        Arc::clone(&ram),
        &[
            MemoryRange {
                address: 0x8010,
                bytes: 32,
            },
            MemoryRange {
                address: 0x4010,
                bytes: 32,
            },
        ],
    )
    .unwrap();
    let mut output = [0x71; 16];
    let mut first = [0; 64];
    let mut second = [0; 64];
    // SAFETY: the fixture retains the registered bytes. The local buffers
    // are private and act as an independent device allocation.
    unsafe {
        ram.publish_physical(0x4000, &[0x11; 64]).unwrap();
        ram.publish_physical(0x8000, &[0x22; 64]).unwrap();
        wired.read_into(24, &mut output).unwrap();
        assert_eq!(output[..8], [0x22; 8]);
        assert_eq!(output[8..], [0x11; 8]);
        wired.publish_from(24, &[0x9d; 16]).unwrap();
        ram.read_physical(0x4000, &mut first).unwrap();
        ram.read_physical(0x8000, &mut second).unwrap();
    }
    assert_eq!(first[..16], [0x11; 16]);
    assert_eq!(first[16..24], [0x9d; 8]);
    assert_eq!(first[24..], [0x11; 40]);
    assert_eq!(second[..40], [0x22; 40]);
    assert_eq!(second[40..48], [0x9d; 8]);
    assert_eq!(second[48..], [0x22; 16]);
    let registration = Arc::downgrade(&ram);
    drop(ram);
    // SAFETY: the owned plan keeps the fixture RAM and the output is private.
    unsafe { wired.read_into(24, &mut output) }.unwrap();
    assert_eq!(output, [0x9d; 16]);
    drop(wired);
    assert!(registration.upgrade().is_none());
}

#[test]
fn a_wired_copy_refuses_a_hole_overlap_or_out_of_bounds_span() {
    let storage = Storage::new();
    let layout = [range(&storage, 0x4000, 0, 64)];
    // SAFETY: storage remains with the registration and has no other writer.
    let ram = Arc::new(unsafe { model(&layout, storage) });
    let valid = MemoryRange {
        address: 0x4000,
        bytes: 32,
    };
    let hole = MemoryRange {
        address: 0x4040,
        bytes: 32,
    };
    assert_eq!(
        WiredRam::new(Arc::clone(&ram), &[valid, hole]).err(),
        Some(MemoryStatus::Translation)
    );
    assert_eq!(
        WiredRam::new(Arc::clone(&ram), &[valid, valid]).err(),
        Some(MemoryStatus::InvalidRequest)
    );
    let wires = [valid];
    let wired = WiredRam::new(Arc::clone(&ram), &wires).unwrap();
    let mut output = [0x71; 8];
    let mut ram_bytes = [0; 32];
    // SAFETY: the fixture retains the registered bytes and the slices are
    // private device-side allocations.
    unsafe {
        assert_eq!(wired.read_into(25, &mut output).unwrap_err().completed, 0);
        assert_eq!(
            wired
                .publish_from(u64::MAX - 3, &[0x92; 8])
                .unwrap_err()
                .completed,
            0
        );
        ram.read_physical(valid.address, &mut ram_bytes).unwrap();
    }
    assert_eq!(output, [0x71; 8]);
    assert_eq!(ram_bytes, [0x53; 32]);
}

#[test]
fn canonical_service_excludes_native_and_mapping_publication_until_drop() {
    let storage = Storage::new();
    let ranges = [range(&storage, 0x4000, 0, 128)];
    // SAFETY: the test's storage backs every range, stays alive with the registration and has
    // no other writer.
    let ram = unsafe { model(&ranges, storage) };
    let other_storage = Storage::new();
    // SAFETY: the test's storage backs every range, stays alive with the registration and has
    // no other writer.
    let other = unsafe { model(&[range(&other_storage, 0x4000, 0, 128)], other_storage) };
    let native = ram.domain.execution_loan().unwrap();
    assert!(ram.try_service().unwrap().is_none());
    drop(native);
    let service = ram.try_service().unwrap().unwrap();
    assert!(service.belongs_to(&ram));
    assert!(!service.belongs_to(&other));
    assert_eq!(service.generation(), ram.domain.generation().unwrap());
    service.validate_current().unwrap();
    assert!(matches!(
        ram.domain.try_execution_loan().unwrap(),
        crate::memory::domain::ExecutionLoanAttempt::ServiceBusy
    ));
    assert!(ram.try_service().unwrap().is_none());
    // An independent physical domain still works; geometry is not permission.
    drop(other.try_service().unwrap().unwrap());
    drop(service);
    drop(ram.domain.execution_loan().unwrap());
    drop(ram.try_service().unwrap().unwrap());
    let mut bytes = [0; 128];
    // SAFETY: the buffers are private locals, disjoint from the registered RAM, which stays
    // registered.
    unsafe { ram.read_physical(0x4000, &mut bytes) }.unwrap();
    assert_eq!(bytes, [0x53; 128]);
}

#[test]
fn a_waiting_service_starts_once_the_native_holder_returns() {
    let storage = Storage::new();
    let ranges = [range(&storage, 0x4000, 0, 128)];
    // SAFETY: the test's storage backs every range, stays alive with the
    // registration and has no other writer.
    let ram = unsafe { model(&ranges, storage) };
    let native = ram.domain.execution_loan().unwrap();
    assert!(ram.try_service().unwrap().is_none());
    let holder = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(20));
        drop(native);
    });
    let service = ram.service().unwrap();
    service.validate_current().unwrap();
    holder.join().unwrap();
}

#[test]
fn canonical_service_blocks_real_alias_copyout_and_descriptor_writers() {
    use std::{sync::mpsc, time::Duration};
    let storage = Storage::new();
    // SAFETY: the test's storage backs every range, stays alive with the registration and has
    // no other writer.
    let ram = Arc::new(unsafe { model(&[range(&storage, 0x4000, 0, 128)], storage) });
    let service = ram.try_service().unwrap().unwrap();
    let (started, ready) = mpsc::channel();
    let (finished, done) = mpsc::channel();
    std::thread::scope(|scope| {
        for descriptor in [false, true] {
            let ram = Arc::clone(&ram);
            let started = started.clone();
            let finished = finished.clone();
            scope.spawn(move || {
                started.send(()).unwrap();
                if descriptor {
                    assert!(
                        ram.update_descriptor(0x4040, u64::from_le_bytes([0x53; 8]), 7)
                            .unwrap()
                    );
                } else {
                    // Same kept physical backing through the host writer.
                    // SAFETY: physical-system writes go through the registered owner, which
                    // keeps these bytes.
                    unsafe { ram.system.publish(&ram.regions[0].backing, 0, &[9; 8]) }.unwrap();
                }
                finished.send(()).unwrap();
            });
        }
        for _ in 0..2 {
            ready.recv_timeout(Duration::from_secs(2)).unwrap();
        }
        service.validate_current().unwrap();
        assert!(done.try_recv().is_err());
        let mut before = [0; 128];
        // SAFETY: the buffers are private locals, disjoint from the registered RAM, which stays
        // registered.
        unsafe { ram.read_physical(0x4000, &mut before) }.unwrap();
        assert_eq!(before, [0x53; 128]);
        drop(service);
        for _ in 0..2 {
            done.recv_timeout(Duration::from_secs(2)).unwrap();
        }
    });
    let mut after = [0; 128];
    // SAFETY: the buffers are private locals, disjoint from the registered RAM, which stays
    // registered.
    unsafe { ram.read_physical(0x4000, &mut after) }.unwrap();
    assert_eq!(&after[..8], &[9; 8]);
    assert_eq!(&after[64..72], &7u64.to_le_bytes());
    assert!(
        after[8..64]
            .iter()
            .chain(after[72..].iter())
            .all(|byte| *byte == 0x53)
    );
}

#[test]
fn compare_exchange_stores_only_while_ram_holds_the_expected_bytes() {
    let storage = Storage::new();
    // SAFETY: the test's storage backs every range, stays alive with the registration and has
    // no other writer.
    let ram = unsafe { model(&[range(&storage, 0x4000, 0, 128)], storage) };
    // Two bytes inside a four-byte quantity whose other bytes stay.
    let mut expected = [0x53; 2];
    // SAFETY: the buffers are private locals, disjoint from the registered RAM, which stays
    // registered.
    assert!(unsafe { ram.compare_exchange_physical(0x4001, &mut expected, &[1, 2]) }.unwrap());
    let mut expected = [0x53; 2];
    // SAFETY: the buffers are private locals, disjoint from the registered RAM, which stays
    // registered.
    assert!(!unsafe { ram.compare_exchange_physical(0x4001, &mut expected, &[3, 4]) }.unwrap());
    assert_eq!(expected, [1, 2], "RAM's bytes");
    let mut bytes = [0; 4];
    // SAFETY: the buffers are private locals, disjoint from the registered RAM, which stays
    // registered.
    unsafe { ram.read_physical(0x4000, &mut bytes) }.unwrap();
    assert_eq!(bytes, [0x53, 1, 2, 0x53]);
    assert!(matches!(
        // SAFETY: the buffers are private locals, disjoint from the registered RAM, which stays
        // registered.
        unsafe { ram.compare_exchange_physical(0x400e, &mut [0; 4], &[0; 4]) },
        Err(MemoryStatus::Alignment)
    ));
    // Threads that each add one with a compare-and-exchange lose none.
    std::thread::scope(|scope| {
        for _ in 0..4 {
            let ram = &ram;
            scope.spawn(move || {
                for _ in 0..1000 {
                    let mut current = [0; 4];
                    // SAFETY: the buffers are private locals, disjoint from the registered RAM,
                    // which stays registered.
                    unsafe { ram.read_physical(0x4024, &mut current) }.unwrap();
                    loop {
                        let next = (u32::from_le_bytes(current) + 1).to_le_bytes();
                        // SAFETY: the buffers are private locals, disjoint from the registered
                        // RAM, which stays registered.
                        if unsafe { ram.compare_exchange_physical(0x4024, &mut current, &next) }
                            .unwrap()
                        {
                            break;
                        }
                    }
                }
            });
        }
    });
    let mut counter = [0; 4];
    // SAFETY: the buffers are private locals, disjoint from the registered RAM, which stays
    // registered.
    unsafe { ram.read_physical(0x4024, &mut counter) }.unwrap();
    assert_eq!(u32::from_le_bytes(counter), 0x5353_5353 + 4000);
}

#[test]
fn canonical_registration_keeps_original_bytes_and_mapping_owner() {
    let storage = Storage::new();
    let weak = Arc::downgrade(&storage);
    let ranges = [range(&storage, 0x4000, 0, 128)];
    // SAFETY: the test's storage backs every range, stays alive with the registration and has
    // no other writer.
    let registered = unsafe { model(&ranges, Arc::clone(&storage)) };
    drop(storage);
    assert!(weak.upgrade().is_some());
    let mut bytes = [0; 128];
    // SAFETY: the buffers are private locals, disjoint from the registered RAM, which stays
    // registered.
    unsafe { registered.read_physical(0x4000, &mut bytes) }.unwrap();
    assert_eq!(bytes, [0x53; 128]);
    drop(registered);
    assert!(weak.upgrade().is_none());
}

#[test]
fn physical_cpu_aliases_overlaps_and_overflow_refuse_before_binding() {
    let storage = Storage::new();
    let first = range(&storage, 0x4000, 0, 64);
    let mut second = range(&storage, 0x4080, 64, 64);
    assert_eq!(validate(&[first, second]), Ok(()));
    second.source_pa = first.source_pa;
    assert_eq!(
        validate(&[first, second]),
        Err(RegistrationError::PhysicalAlias)
    );
    second = range(&storage, 0x4080, 64, 64);
    second.host_pa = first.host_pa;
    assert_eq!(
        validate(&[first, second]),
        Err(RegistrationError::PhysicalAlias)
    );
    second = range(&storage, 0x4080, 64, 64);
    second.host_base = first.host_base;
    assert_eq!(
        validate(&[first, second]),
        Err(RegistrationError::PhysicalAlias)
    );
    second = range(&storage, u64::MAX - 15, 64, 64);
    assert_eq!(validate(&[second]), Err(RegistrationError::Geometry));
    assert_eq!(validate(&[]), Err(RegistrationError::Geometry));
}

#[test]
fn a_hole_or_outer_boundary_never_exposes_a_read_prefix() {
    let storage = Storage::new();
    let ranges = [
        range(&storage, 0x4000, 0, 64),
        range(&storage, 0x4080, 64, 64),
    ];
    // SAFETY: the test's storage backs every range, stays alive with the registration and has
    // no other writer.
    let registered = unsafe { model(&ranges, storage) };
    for physical in [0x3ff0, 0x4038, 0x40b8, u64::MAX - 7] {
        let mut bytes = [0xa5; 16];
        let fault =
                // SAFETY: the buffers are private locals, disjoint from the registered RAM, which
                // stays registered.
                unsafe { registered.read_physical_progress(physical, &mut bytes) }.unwrap_err();
        assert_eq!(fault.status, MemoryStatus::Translation);
        assert_eq!(fault.completed, 0);
        assert_eq!(bytes, [0xa5; 16]);
    }
    // Empty reads still perform the existing domain-health preflight but
    // need no region or backing, even at a hole or the address-space end.
    for physical in [0x3ff0, 0x4040, 0x4080, u64::MAX] {
        // SAFETY: the buffers are private locals, disjoint from the registered RAM, which stays
        // registered.
        unsafe { registered.read_physical_progress(physical, &mut []) }.unwrap();
    }
}

#[test]
fn unaligned_physical_reads_preserve_all_reached_scalar_tails() {
    let storage = Storage::new();
    // SAFETY: the test's storage backs every range, stays alive with the registration and has
    // no other writer.
    let registered = unsafe { model(&[range(&storage, 0x4000, 0, 128)], storage) };
    let expected: [u8; 128] = core::array::from_fn(|index| u8::try_from(index).unwrap() ^ 0xa5);
    // SAFETY: the buffers are private locals, disjoint from the registered RAM, which stays
    // registered.
    unsafe { registered.publish_physical(0x4000, &expected) }.unwrap();
    for start in 0..16 {
        for bytes in 0..=expected.len() - start {
            let mut output = [0x3c; 128];
            // SAFETY: the buffers are private locals, disjoint from the registered RAM, which
            // stays registered.
            unsafe {
                registered.read_physical_progress(0x4000 + start as u64, &mut output[..bytes])
            }
            .unwrap();
            assert_eq!(&output[..bytes], &expected[start..start + bytes]);
            assert!(output[bytes..].iter().all(|byte| *byte == 0x3c));
        }
    }
}

#[test]
fn physical_reads_cross_adjacent_source_regions_with_distinct_backings() {
    let first = Storage::new();
    let second = Storage::new();
    let first_range = range(&first, 0x4000, 16, 64);
    let mut second_range = range(&second, 0x4040, 32, 64);
    second_range.host_pa = 0x2000_0020;
    // These source-adjacent regions keep different allocations,
    // host offsets and physical placements; pointer arithmetic cannot cross
    // their boundary even though the requested source range is contiguous.
    // SAFETY: the test's storage backs every range, stays alive with the registration and has
    // no other writer.
    let registered = unsafe {
        fixture(
            &[second_range, first_range],
            Arc::new((Arc::clone(&first), Arc::clone(&second))),
        )
    };
    let expected: [u8; 128] = core::array::from_fn(|index| u8::try_from(index).unwrap() ^ 0x69);
    // SAFETY: the buffers are private locals, disjoint from the registered RAM, which stays
    // registered.
    unsafe { registered.publish_physical(0x4000, &expected) }.unwrap();
    for start in 0..expected.len() {
        for bytes in [
            0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 63, 64, 65, 127, 128,
        ] {
            if bytes > expected.len() - start {
                continue;
            }
            let mut output = [0xc3; 128];
            // SAFETY: the buffers are private locals, disjoint from the registered RAM, which
            // stays registered.
            unsafe {
                registered.read_physical_progress(0x4000 + start as u64, &mut output[..bytes])
            }
            .unwrap();
            assert_eq!(&output[..bytes], &expected[start..start + bytes]);
            assert!(output[bytes..].iter().all(|byte| *byte == 0xc3));
        }
    }
}

#[test]
fn later_backing_refusal_reports_only_the_completed_output_prefix() {
    let storage = Storage::new();
    let ranges = [
        range(&storage, 0x4000, 0, 64),
        range(&storage, 0x4040, 64, 64),
    ];
    // SAFETY: the test's storage backs every range, stays alive with the registration and has
    // no other writer.
    let mut registered = unsafe { model(&ranges, Arc::clone(&storage)) };
    // Inconsistent private metadata models a later envelope
    // refusal. Production registration never changes a backing's rights.
    // No failed read may expose bytes from that backing or erase the prefix.
    // SAFETY: the replacement backing covers the same storage bytes.
    registered.regions[1].backing = unsafe {
        MemoryBacking::new(
            ranges[1].host_base,
            ranges[1].bytes,
            AccessPermissions {
                readable: false,
                writable: true,
                executable: false,
            },
            storage,
        )
    }
    .unwrap();
    for start in 0..64 {
        let mut output = [0xa5; 128];
        let bytes = 128 - start;
        // SAFETY: the buffers are private locals, disjoint from the registered RAM, which stays
        // registered.
        let refused = unsafe {
            registered.read_physical_progress(0x4000 + start as u64, &mut output[..bytes])
        }
        .unwrap_err();
        assert_eq!(refused.status, MemoryStatus::Domain);
        assert_eq!(refused.completed, 64 - start);
        assert!(output[..refused.completed].iter().all(|byte| *byte == 0x53));
        assert!(output[refused.completed..].iter().all(|byte| *byte == 0xa5));
    }
}

#[test]
fn long_physical_read_keeps_one_backing_and_the_original_scalar_observations() {
    #[repr(C, align(16))]
    struct LongStorage(UnsafeCell<[u8; 1024]>);
    // SAFETY: the storage is accessed only through the canonical
    // native scalar reader and same-system publication after registration.
    unsafe impl Sync for LongStorage {}
    let storage = Arc::new(LongStorage(UnsafeCell::new([0; 1024])));
    let range = ReservedSystemRamRange {
        source_pa: 0x4000,
        host_pa: 0x1000_0000,
        host_base: NonNull::new(storage.0.get().cast::<u8>()).unwrap(),
        bytes: 1024,
    };
    // SAFETY: the test's storage backs every range, stays alive with the registration and has
    // no other writer.
    let registered = unsafe { fixture(&[range], storage) };
    let expected: [u8; 1024] = core::array::from_fn(|index| u8::try_from(index % 251).unwrap());
    // SAFETY: the buffers are private locals, disjoint from the registered RAM, which stays
    // registered.
    unsafe { registered.publish_physical(0x4000, &expected) }.unwrap();
    let mut output = [0; 1024];
    // SAFETY: the buffers are private locals, disjoint from the registered RAM, which stays
    // registered.
    unsafe { registered.read_physical_progress(0x4000, &mut output) }.unwrap();
    assert_eq!(output, expected);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn native_scalar_fetches_are_coherent_with_the_same_host_writer_system() {
    let storage = Storage::new();
    let ranges = [range(&storage, 0x4000, 0, 128)];
    // SAFETY: the test's storage backs every range, stays alive with the registration and has
    // no other writer.
    let registered = unsafe { model(&ranges, storage) };
    std::thread::scope(|scope| {
        scope.spawn(|| {
            for value in (0..10_000).map(|iteration| if iteration & 1 == 0 { 0x55 } else { 0xaa }) {
                // SAFETY: physical-system writes go through the registered owner, which keeps
                // these bytes.
                unsafe {
                    registered
                        .system
                        .publish(&registered.regions[0].backing, 0, &[value; 8])
                }
                .unwrap();
            }
        });
        for _ in 0..10_000 {
            for bytes in [8, 128] {
                // Both the original one-scalar read and a multi-scalar
                // read keep the real aligned 8-byte prefix non-torn while
                // the joined writer changes it. Later scalar reads are
                // independent observations, not an atomic snapshot.
                let mut value = [0; 128];
                // SAFETY: the buffers are private locals, disjoint from the registered RAM,
                // which stays registered.
                unsafe { registered.read_physical(0x4000, &mut value[..bytes]) }.unwrap();
                assert!(matches!(value[0], 0x53 | 0x55 | 0xaa));
                assert!(value[..8].iter().all(|byte| *byte == value[0]));
                assert!(value[8..bytes].iter().all(|byte| *byte == 0x53));
            }
        }
    });
}

#[test]
fn physical_host_copy_preflights_holes_and_keeps_descriptor_prefix_precise() {
    let storage = Storage::new();
    let ranges = [
        range(&storage, 0x4000, 0, 64),
        range(&storage, 0x4080, 64, 64),
    ];
    // SAFETY: the test's storage backs every range, stays alive with the registration and has
    // no other writer.
    let registered = unsafe { model(&ranges, storage) };
    // SAFETY: the buffers are private locals, disjoint from the registered RAM, which stays
    // registered.
    let refused = unsafe { registered.publish_physical(0x4038, &[0; 16]) }.unwrap_err();
    assert_eq!(refused.completed, 0);
    assert_eq!(refused.status, MemoryStatus::Translation);
    assert_eq!(
        registered.read_descriptor(0x4038),
        Ok(0x5353_5353_5353_5353)
    );
    let mut descriptors = registered.descriptor_reader();
    assert_eq!(
        descriptors.read_descriptor(0x4000),
        Ok(0x5353_5353_5353_5353)
    );
    assert_eq!(
        descriptors.read_descriptor(0x4001),
        Err(MemoryStatus::InvalidRequest)
    );
    assert_eq!(
        descriptors.read_descriptor(0x4040),
        Err(MemoryStatus::Translation)
    );
    assert_eq!(
        registered.update_descriptor(0x4001, 0, 1),
        Err(MemoryStatus::InvalidRequest)
    );
    assert_eq!(
        registered.update_descriptor(0x4040, 0, 1),
        Err(MemoryStatus::Translation)
    );
    assert_eq!(
        registered.update_descriptor(0x4000, 0x5353_5353_5353_5353, 0x1234),
        Ok(true)
    );
    assert_eq!(
        registered.update_descriptor(0x4000, 0x5353_5353_5353_5353, 0x9999),
        Ok(false)
    );
    assert_eq!(registered.read_descriptor(0x4000), Ok(0x1234));
    assert_eq!(descriptors.read_descriptor(0x4000), Ok(0x1234));
    assert_eq!(
        registered.read_descriptor(0x4008),
        Ok(0x5353_5353_5353_5353)
    );
    // SAFETY: the buffers are private locals, disjoint from the registered RAM, which stays
    // registered.
    unsafe { registered.publish_physical(0x4080, &42u64.to_le_bytes()) }.unwrap();
    assert_eq!(registered.read_descriptor(0x4080), Ok(42));
    assert_eq!(descriptors.read_descriptor(0x4080), Ok(42));
    assert_eq!(descriptors.read_descriptor(0x4000), Ok(0x1234));
}

#[test]
fn posix_ram_layout_does_not_mint_a_freestanding_target_proof() {
    let storage = Storage::new();
    let ranges = [range(&storage, 0x4000, 0, 128)];
    assert!(matches!(
        // SAFETY: the test's storage backs every range, stays alive with the registration and
        // has no other writer.
        unsafe { RegisteredSystemRam::register(&ranges, storage, 64, 8) },
        Err(RegistrationError::NativeTarget)
    ));
}
