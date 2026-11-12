use super::super::{DataAccess, DataPolicy, MappingAttributes, MemoryDomain};
use super::*;
use crate::memory::backing::{AccessPermissions, MemoryView};
use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::ptr::NonNull;
use stela_compiler::context::memory::{MemoryContextView, MemoryRequest};

const STORAGE_BYTES: usize = 64;
const GUEST_BASE: u64 = 0x1000;
const ALIAS_BASE: u64 = 0x2000;
const RIGHTS: AccessPermissions = AccessPermissions {
    readable: true,
    writable: true,
    executable: false,
};
const READONLY: AccessPermissions = AccessPermissions {
    writable: false,
    ..RIGHTS
};
const ATTRIBUTES: MappingAttributes = MappingAttributes {
    tagged: false,
    shareable: false,
};

struct Storage {
    pointer: NonNull<u8>,
    layout: Layout,
}

// SAFETY: storage only owns raw allocation; byte access uses native helpers.
unsafe impl Send for Storage {}
// SAFETY: storage lends no Rust references and accesses use one writer domain.
unsafe impl Sync for Storage {}

impl Drop for Storage {
    fn drop(&mut self) {
        // SAFETY: this owner keeps its allocation and layout.
        unsafe { dealloc(self.pointer.as_ptr(), self.layout) };
    }
}

fn backing(rights: AccessPermissions) -> Arc<MemoryBacking> {
    let layout = Layout::from_size_align(STORAGE_BYTES, HOST_QUANTITY_BYTES).unwrap();
    // SAFETY: nonempty valid layout kept by Storage for deallocation.
    let pointer = NonNull::new(unsafe { alloc_zeroed(layout) }).unwrap();
    let storage = Arc::new(Storage { pointer, layout });
    // SAFETY: owner pins raw coherent RAM, with no aliasing Rust references.
    unsafe { MemoryBacking::new(pointer, STORAGE_BYTES, rights, storage) }.unwrap()
}

fn system() -> Arc<MemorySystem> {
    MemorySystem::new(16, 4).unwrap()
}

fn domain(
    system: &Arc<MemorySystem>,
    backing: &Arc<MemoryBacking>,
    base: u64,
) -> Arc<MemoryDomain> {
    let view = MemoryView::new(Arc::clone(backing), base, 0, STORAGE_BYTES, READONLY).unwrap();
    MemoryDomain::new(
        Arc::clone(system),
        vec![(view, ATTRIBUTES)],
        TargetCpu::Baseline,
    )
    .unwrap()
}

fn publish(
    system: &MemorySystem,
    backing: &Arc<MemoryBacking>,
    offset: usize,
    bytes: &[u8],
) -> Result<(), PublicationFault> {
    // SAFETY: the test may publish; inputs are private, host storage is pinned
    // and disjoint, and every write uses this domain.
    unsafe { system.publish(backing, offset, bytes) }
}

fn snapshot(backing: &MemoryBacking) -> Vec<u8> {
    (0..STORAGE_BYTES)
        .map(|offset| {
            let pointer = backing.host_pointer(offset, 1, AccessKind::Read).unwrap();
            // SAFETY: synchronous exact raw reads have pinned rights/lifetime.
            unsafe { access::read_part(pointer.as_ptr(), 1, TargetCpu::Baseline) }.unwrap() as u8
        })
        .collect()
}

struct Policy;

impl DataPolicy for Policy {
    fn validate_execution(&self, _required_flags: u32) -> Result<(), MemoryStatus> {
        Ok(())
    }

    fn translate(&self, raw_address: u64) -> Result<u64, MemoryStatus> {
        Ok(raw_address)
    }
    fn check(&mut self, _: DataAccess<'_>) -> Result<(), MemoryStatus> {
        Ok(())
    }
}

fn native_read(domain: &Arc<MemoryDomain>, address: u64, bytes: &mut [u8]) -> u32 {
    let mut thread = domain.thread(Policy);
    let request = MemoryRequest {
        address,
        data: bytes.as_mut_ptr(),
        bytes: bytes.len() as u32,
        flags: 0,
        pc: 0x1234,
    };
    // SAFETY: raw native call uses the C ABI, synchronous valid context/request,
    // and private staging disjoint from guest RAM; no pointers escape.
    unsafe {
        thread.with_view(|context: &mut MemoryContextView| {
            let status: u32;
            std::arch::asm!(
                "call {callback}",
                callback = in(reg) context.callback,
                in("rdi") std::ptr::from_mut(context),
                in("rsi") std::ptr::from_ref(&request),
                lateout("eax") status,
                clobber_abi("C"),
            );
            status
        })
    }
}

fn reservation_access(
    domain: &MemoryDomain,
    offset: u64,
) -> monitor::Access<MemoryBacking, MemoryView, MappingAttributes> {
    let registered = &domain.registered_views()[0];
    monitor::Access::new(
        Arc::clone(registered.view.backing()),
        Arc::clone(&registered.monitor),
        offset,
        1,
        monitor::Shape::Scalar,
    )
    .unwrap()
}

#[test]
fn conditional_descriptor_miss_has_no_store_or_reservation_effect() {
    let system = system();
    let backing = backing(RIGHTS);
    let domain = domain(&system, &backing, GUEST_BASE);
    let access = reservation_access(&domain, 0);
    let mut reservation = system.reservations.thread();
    system
        .reservations
        .load(&mut reservation, &access, || Ok::<_, ()>(()))
        .unwrap();
    assert_eq!(
        // SAFETY: Storage pins the aligned writable eight-byte descriptor; the only
        // writer and this reservation both use system, without Rust byte references.
        unsafe { system.compare_update_descriptor(&backing, 0, 1, 2) },
        Ok(false)
    );
    assert_eq!(snapshot(&backing), vec![0; STORAGE_BYTES]);
    assert_eq!(
        system
            .reservations
            .store(&mut reservation, &access, || Ok::<_, ()>(()))
            .unwrap(),
        monitor::ConditionalWrite::Stored(())
    );
    system
        .reservations
        .load(&mut reservation, &access, || Ok::<_, ()>(()))
        .unwrap();
    // Successful same-value publication is still a physical writer event.
    assert_eq!(
        // SAFETY: The descriptor stays pinned and writable, and all stores use system;
        // even this same-value update participates in its reservation gate.
        unsafe { system.compare_update_descriptor(&backing, 0, 0, 0) },
        Ok(true)
    );
    assert_eq!(
        system
            .reservations
            .store(&mut reservation, &access, || -> Result<(), ()> {
                panic!("same-value descriptor write must invalidate")
            })
            .unwrap(),
        monitor::ConditionalWrite::Failed
    );
}

#[test]
fn descriptor_updates_preflight_exact_rights_bounds_and_coherence() {
    let system = system();
    let backing = backing(RIGHTS);
    publish(&system, &backing, 0, &[0x5a; STORAGE_BYTES]).unwrap();
    assert_eq!(
        // SAFETY: The raw allocation remains pinned and unaliased; descriptor preflight
        // rejects this unaligned offset before any memory access.
        unsafe { system.compare_update_descriptor(&backing, 1, 0, 1) },
        Err(MemoryStatus::InvalidRequest)
    );
    assert_eq!(
        // SAFETY: The backing allocation remains live; descriptor preflight rejects
        // this out-of-bounds offset before deriving or accessing a pointer.
        unsafe { system.compare_update_descriptor(&backing, STORAGE_BYTES, 0, 1) },
        Err(MemoryStatus::Translation)
    );
    let other = MemorySystem::new(16, 4).unwrap();
    assert_eq!(
        // SAFETY: The allocation is live and already bound to system. The other
        // coherence identity is rejected before any descriptor access.
        unsafe { other.compare_update_descriptor(&backing, 0, 0, 1) },
        Err(MemoryStatus::Domain)
    );
    assert_eq!(snapshot(&backing), vec![0x5a; STORAGE_BYTES]);
    let readonly = self::backing(READONLY);
    assert_eq!(
        // SAFETY: The backing allocation is live raw RAM, but its declared write right
        // is absent; preflight rejects it before any store.
        unsafe { system.compare_update_descriptor(&readonly, 0, 0, 1) },
        Err(MemoryStatus::Permission)
    );
    assert_eq!(snapshot(&readonly), vec![0; STORAGE_BYTES]);
    assert_eq!(
        // SAFETY: Bytes 8..16 lie in the pinned aligned writable allocation. This thread
        // is the sole writer and uses system; no Rust reference aliases the bytes.
        unsafe { system.compare_update_descriptor(&backing, 8, 0x5a5a_5a5a_5a5a_5a5a, 0x1234) },
        Ok(true)
    );
    let mut expected = vec![0x5a; STORAGE_BYTES];
    expected[8..16].copy_from_slice(&0x1234u64.to_le_bytes());
    assert_eq!(snapshot(&backing), expected);
}

#[test]
fn native_concurrent_descriptor_updates_share_the_same_alias_writer_gate() {
    let system = system();
    let backing = backing(RIGHTS);
    let first = domain(&system, &backing, GUEST_BASE);
    let alias = domain(&system, &backing, ALIAS_BASE);
    let alias_access = reservation_access(&alias, 0);
    let mut reservation = system.reservations.thread();
    system
        .reservations
        .load(&mut reservation, &alias_access, || Ok::<_, ()>(()))
        .unwrap();
    std::thread::scope(|scope| {
        for _ in 0..2 {
            scope.spawn(|| {
                let pointer = backing.host_pointer(0, 8, AccessKind::Read).unwrap();
                for _ in 0..10_000 {
                    loop {
                        let old =
                            // SAFETY: host_pointer validated eight readable bytes at the aligned allocation
                            // base. Storage stays pinned, and concurrent writers use descriptor CAS.
                            unsafe { access::read_part(pointer.as_ptr(), 8, TargetCpu::Baseline) }
                                .unwrap() as u64;
                        // SAFETY: The descriptor is pinned, aligned, and writable. Both threads use
                        // this system for CAS, and no Rust references alias the raw bytes.
                        if unsafe { system.compare_update_descriptor(&backing, 0, old, old + 1) }
                            .unwrap()
                        {
                            break;
                        }
                    }
                }
            });
        }
    });
    assert_eq!(&snapshot(&backing)[..8], &20_000u64.to_le_bytes());
    assert_eq!(
        system
            .reservations
            .store(&mut reservation, &alias_access, || -> Result<(), ()> {
                panic!("concurrent aliased writer must invalidate")
            })
            .unwrap(),
        monitor::ConditionalWrite::Failed
    );
    assert!(first.check_execution().is_ok());
}

#[test]
fn fragments_fit_host_quantities_and_choose_largest_exact_scalar() {
    for residue in 0..HOST_QUANTITY_BYTES {
        for length in 0..=STORAGE_BYTES {
            let mut completed = 0;
            for part in PublicationParts::new(residue, length) {
                assert_eq!(part.offset, completed);
                assert_eq!(part.address, residue + completed);
                assert!(matches!(part.bytes, 1 | 2 | 4 | 8));
                let available = HOST_QUANTITY_BYTES - (part.address & (HOST_QUANTITY_BYTES - 1));
                assert!(part.bytes <= available);
                assert!(part.bytes * 2 > available.min(length - completed));
                assert_eq!(
                    access::read_footprint(part.address, part.bytes, TargetCpu::Baseline),
                    Ok((part.address, part.bytes))
                );
                completed += part.bytes;
            }
            assert_eq!(completed, length);
        }
    }
}

#[test]
fn native_publications_preserve_all_neighbors_for_every_residue_and_length() {
    for residue in 0..HOST_QUANTITY_BYTES {
        for length in 0..=32 {
            let system = system();
            let backing = backing(RIGHTS);
            let mut expected = vec![0xa5; STORAGE_BYTES];
            publish(&system, &backing, 0, &expected).unwrap();
            let bytes: Vec<_> = (0..length).map(|index| index as u8).collect();
            let offset = HOST_QUANTITY_BYTES + residue;
            publish(&system, &backing, offset, &bytes).unwrap();
            expected[offset..offset + length].copy_from_slice(&bytes);
            assert_eq!(
                snapshot(&backing),
                expected,
                "residue={residue}, length={length}"
            );
        }
    }
}

#[test]
fn guest_readonly_alias_reads_host_publication_through_native_callback() {
    let system = system();
    let backing = backing(RIGHTS);
    let domain = domain(&system, &backing, GUEST_BASE);
    let expected = [0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0];
    publish(&system, &backing, 0, &expected).unwrap();
    let mut actual = [0; 8];
    assert_eq!(
        native_read(&domain, GUEST_BASE, &mut actual),
        MemoryStatus::Success as u32
    );
    assert_eq!(actual, expected);
}

#[test]
fn publication_invalidates_cross_address_space_alias_reservations_including_aba() {
    let system = system();
    let backing = backing(RIGHTS);
    let first = domain(&system, &backing, GUEST_BASE);
    let alias = domain(&system, &backing, ALIAS_BASE);
    let mut reservation = system.reservations.thread();
    let first_access = reservation_access(&first, 0);
    let alias_access = reservation_access(&alias, 0);
    for values in [&[0][..], &[0x5a, 0][..]] {
        system
            .reservations
            .load(&mut reservation, &first_access, || {
                let pointer = backing.host_pointer(0, 1, AccessKind::Read).unwrap();
                // SAFETY: reservation gate protects this live raw read.
                unsafe { access::read_part(pointer.as_ptr(), 1, TargetCpu::Baseline) }
            })
            .unwrap();
        for value in values {
            publish(&system, &backing, 0, &[*value]).unwrap();
        }
        let stored = system
            .reservations
            .store(&mut reservation, &alias_access, || -> Result<(), ()> {
                panic!("host publication must invalidate the alias reservation")
            })
            .unwrap();
        assert_eq!(stored, monitor::ConditionalWrite::Failed);
        assert_eq!(snapshot(&backing)[0], 0);
    }
}

#[test]
fn publication_under_frozen_gates_invalidates_only_reached_reservations() {
    let system = system();
    let backing = backing(RIGHTS);
    let first = domain(&system, &backing, GUEST_BASE);
    let alias = domain(&system, &backing, ALIAS_BASE);
    let mut overwritten = system.reservations.thread();
    let mut neighbor = system.reservations.thread();
    let neighbor_access = reservation_access(&first, 16);
    system
        .reservations
        .load(&mut overwritten, &reservation_access(&first, 0), || {
            Ok::<_, ()>(())
        })
        .unwrap();
    system
        .reservations
        .load(&mut neighbor, &neighbor_access, || Ok::<_, ()>(()))
        .unwrap();

    let writers = system.reservations.try_freeze_writers().unwrap().unwrap();
    // SAFETY: the fixture owns both aliases and all readers, the source
    // is private, and these are this system's retained writer guards.
    unsafe { system.publish_with(&backing, 0, &[0, 0x5a], None, Some(&writers)) }.unwrap();
    drop(writers);

    assert_eq!(
        system
            .reservations
            .store(&mut overwritten, &reservation_access(&alias, 0), || {
                Ok::<_, ()>(())
            })
            .unwrap(),
        monitor::ConditionalWrite::Failed
    );
    assert_eq!(
        system
            .reservations
            .store(&mut neighbor, &neighbor_access, || Ok::<_, ()>(()))
            .unwrap(),
        monitor::ConditionalWrite::Stored(())
    );
    assert_eq!(snapshot(&backing)[..2], [0, 0x5a]);
}

#[test]
fn wrong_coherence_owner_and_invalid_bounds_have_no_byte_effects() {
    let system = system();
    let backing = backing(RIGHTS);
    publish(&system, &backing, 0, &[0x5a; STORAGE_BYTES]).unwrap();
    let other = MemorySystem::new(16, 4).unwrap();
    assert_eq!(
        publish(&other, &backing, 0, &[0xff]),
        Err(PublicationFault {
            status: MemoryStatus::Domain,
            completed: 0,
        })
    );
    for (offset, bytes) in [
        (STORAGE_BYTES - 1, &[1, 2][..]),
        (usize::MAX, &[1][..]),
        (STORAGE_BYTES + 1, &[][..]),
    ] {
        assert_eq!(
            publish(&system, &backing, offset, bytes),
            Err(PublicationFault {
                status: MemoryStatus::Translation,
                completed: 0,
            })
        );
    }
    assert_eq!(snapshot(&backing), [0x5a; STORAGE_BYTES]);
}

#[test]
fn preflight_failures_and_empty_publications_preserve_reservations() {
    let system = system();
    let backing = backing(RIGHTS);
    let domain = domain(&system, &backing, GUEST_BASE);
    let access = reservation_access(&domain, 0);
    let mut reservation = system.reservations.thread();
    system
        .reservations
        .load(&mut reservation, &access, || Ok::<(), ()>(()))
        .unwrap();
    assert!(publish(&system, &backing, STORAGE_BYTES - 1, &[1, 2]).is_err());
    let other = MemorySystem::new(16, 4).unwrap();
    assert!(publish(&other, &backing, 0, &[1]).is_err());
    publish(&system, &backing, 0, &[]).unwrap();
    let stored = system
        .reservations
        .store(&mut reservation, &access, || Ok::<(), ()>(()))
        .unwrap();
    assert_eq!(stored, monitor::ConditionalWrite::Stored(()));
    assert_eq!(snapshot(&backing), [0; STORAGE_BYTES]);
}

#[test]
fn complete_host_read_and_write_rights_are_required_before_any_write() {
    for rights in [
        READONLY,
        AccessPermissions {
            readable: false,
            ..RIGHTS
        },
    ] {
        let system = system();
        let backing = backing(rights);
        assert_eq!(
            publish(&system, &backing, 0, &[0x5a; STORAGE_BYTES]),
            Err(PublicationFault {
                status: MemoryStatus::Permission,
                completed: 0,
            })
        );
        // The allocation is readable natively even when backing metadata denies
        // reads; its owner pins those rights for this post-failure observation.
        let pointer = backing
            .host_pointer(
                0,
                1,
                if rights.readable {
                    AccessKind::Read
                } else {
                    AccessKind::Write
                },
            )
            .unwrap();
        for offset in 0..STORAGE_BYTES {
            assert_eq!(
                // SAFETY: host_pointer covers STORAGE_BYTES readable bytes, so this offset is
                // in bounds. All writes are complete and the allocation remains pinned.
                unsafe { access::read_part(pointer.as_ptr().add(offset), 1, TargetCpu::Baseline) }
                    .unwrap(),
                0
            );
        }
    }
}

#[test]
fn empty_publication_is_noop_but_terminal_system_still_declines() {
    let system = system();
    let backing = backing(RIGHTS);
    publish(&system, &backing, 0, &[]).unwrap();
    publish(&system, &backing, STORAGE_BYTES, &[]).unwrap();
    assert_eq!(snapshot(&backing), [0; STORAGE_BYTES]);
    system.failed.store(true, Ordering::Release);
    for bytes in [&[][..], &[0x5a][..]] {
        assert_eq!(
            publish(&system, &backing, 0, bytes),
            Err(PublicationFault {
                status: MemoryStatus::Domain,
                completed: 0,
            })
        );
    }
    assert_eq!(snapshot(&backing), [0; STORAGE_BYTES]);
}

#[test]
fn poisoned_monitor_declines_before_publication() {
    let system = system();
    let backing = backing(RIGHTS);
    let range = monitor::WriteRange::new(Arc::clone(&backing), 0, 1).unwrap();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = system.reservations.with_writer(
            &range,
            monitor::Invalidation::BeforeAttempt,
            || -> Result<(), ()> { panic!("poison the existing physical writer monitor") },
        );
    }));
    assert!(result.is_err());
    for bytes in [&[][..], &[0x5a][..]] {
        assert_eq!(
            publish(&system, &backing, 0, bytes),
            Err(PublicationFault {
                status: MemoryStatus::Domain,
                completed: 0,
            })
        );
    }
    assert_eq!(snapshot(&backing), [0; STORAGE_BYTES]);
}

#[cfg(host_memory)]
mod binding_reservation {
    //! Reservation behavior of bound mappings.
    use super::super::super::{MappingAttributes, MemoryDomain, MemorySystem};
    use crate::memory::{
        backing::MemoryView,
        binding::tests::{config, native, owned, published},
        monitor,
    };
    use std::sync::Arc;
    use stela_compiler::TargetCpu;

    #[test]
    fn service_write_invalidates_the_original_backing_reservation() {
        use crate::memory::policy::{LowerAddressProfile, MemoryControls, TagCheckFaultMode};
        let (mut owner, base) = owned();
        let system = MemorySystem::new(16, 8).unwrap();
        // SAFETY: new owned anonymous storage with every writer.
        let domain =
            unsafe { owner.bind_shared_memory(config(Arc::clone(&system), 75), &mut []) }.unwrap();
        let mapping = &domain.registered_views()[0];
        let backing = Arc::clone(mapping.view.backing());
        let access = monitor::Access::new(
            Arc::clone(&backing),
            Arc::clone(&mapping.monitor),
            0,
            8,
            monitor::Shape::Scalar,
        )
        .unwrap();
        let mut reservation = system.reservations.thread();
        system
            .reservations
            .load(&mut reservation, &access, || {
                Ok::<_, ()>(native(&backing, 0, 8))
            })
            .unwrap();
        let mut actual_thread = domain.thread(MemoryControls::new(
            LowerAddressProfile::new(47, false).unwrap(),
            false,
            false,
            false,
            TagCheckFaultMode::Ignore,
            false,
        ));
        {
            let mut service = owner.service_memory(&mut actual_thread).unwrap();
            service.validate_write(base, 8).unwrap();
            assert_eq!(service.read_snapshot(base, 8).unwrap(), [0; 8]);
            // Even a same-value write is a real physical writer effect.
            service.write(base, &0u64.to_le_bytes()).unwrap();
        }
        assert!(matches!(
            system
                .reservations
                .store(&mut reservation, &access, || Ok::<_, ()>(())),
            Ok(monitor::ConditionalWrite::Failed)
        ));
        assert_eq!(domain.generation().unwrap(), 0);
    }

    #[test]
    fn bound_host_same_value_and_aba_invalidate_existing_system_reservation() {
        let (mut owner, base) = owned();
        let system = MemorySystem::new(16, 8).unwrap();
        let domain =
            // SAFETY: owned creates private anonymous RAM; this exclusive owner is its
            // only writer and the resulting domain keeps its allocation.
            unsafe { owner.bind_shared_memory(config(Arc::clone(&system), 11), &mut []) }.unwrap();
        let mapping = &domain.registered_views()[0];
        let backing = Arc::clone(mapping.view.backing());
        let access = monitor::Access::new(
            Arc::clone(&backing),
            Arc::clone(&mapping.monitor),
            0,
            8,
            monitor::Shape::Scalar,
        )
        .unwrap();
        let mut thread = system.reservations.thread();
        for aba in [false, true] {
            system
                .reservations
                .load(&mut thread, &access, || Ok::<_, ()>(native(&backing, 0, 8)))
                .unwrap();
            if aba {
                owner.write(base, &1u64.to_le_bytes()).unwrap();
            }
            owner.write(base, &0u64.to_le_bytes()).unwrap();
            assert_eq!(
                system
                    .reservations
                    .store(&mut thread, &access, || -> Result<(), ()> {
                        panic!("invalidated reservation must not write")
                    })
                    .unwrap(),
                monitor::ConditionalWrite::<()>::Failed
            );
        }
    }

    #[test]
    fn guest_ro_publisher_invalidates_reservation_in_another_address_space() {
        let (mut owner, _) = owned();
        let mut publisher = published();
        let system = MemorySystem::new(16, 8).unwrap();
        // SAFETY: owner and publisher keep disjoint live mappings with no running
        // guest. This call binds their complete writer set to the same system.
        let domain = unsafe {
            owner.bind_shared_memory(config(Arc::clone(&system), 12), &mut [&mut publisher])
        }
        .unwrap();
        let backing = Arc::clone(publisher.shared_backing.as_ref().unwrap());
        let rights = crate::memory::backing::AccessPermissions {
            readable: true,
            writable: false,
            executable: false,
        };
        let alias = MemoryView::new(Arc::clone(&backing), 0x1000, 0, 32, rights).unwrap();
        let other = MemoryDomain::new_for_address_space(
            13,
            Arc::clone(&system),
            vec![(
                alias,
                MappingAttributes {
                    tagged: false,
                    shareable: true,
                },
            )],
            TargetCpu::Baseline,
        )
        .unwrap();
        let access = monitor::Access::new(
            Arc::clone(&backing),
            Arc::clone(&other.registered_views()[0].monitor),
            0,
            8,
            monitor::Shape::Scalar,
        )
        .unwrap();
        let mut thread = system.reservations.thread();
        system
            .reservations
            .load(&mut thread, &access, || Ok::<_, ()>(native(&backing, 0, 8)))
            .unwrap();
        publisher.publish(0, &[7; 8]).unwrap();
        assert_eq!(
            system
                .reservations
                .store(&mut thread, &access, || -> Result<(), ()> {
                    panic!("alias writer must invalidate")
                })
                .unwrap(),
            monitor::ConditionalWrite::<()>::Failed
        );
        assert_ne!(domain.address_space(), other.address_space());
    }

    #[test]
    fn canonical_snapshots_and_unrelated_mapping_changes_preserve_live_reservation() {
        let (mut owner, base) = owned();
        let (mut other_owner, other_base) = owned();
        let system = MemorySystem::new(16, 8).unwrap();
        // SAFETY: Both owners exclusively keep separate anonymous mappings. There is
        // no running guest; the group binds every writer before any access.
        let domain = unsafe {
            owner.bind_shared_memory_group(
                config(Arc::clone(&system), 14),
                &mut [&mut other_owner],
                &mut [],
            )
        }
        .unwrap();
        let views = domain.registered_views();
        let mapping = views
            .iter()
            .find(|mapping| mapping.view.guest_base() == base)
            .unwrap();
        let backing = Arc::clone(mapping.view.backing());
        let access = monitor::Access::new(
            Arc::clone(&backing),
            Arc::clone(&mapping.monitor),
            0,
            8,
            monitor::Shape::Scalar,
        )
        .unwrap();
        let mut thread = system.reservations.thread();
        system
            .reservations
            .load(&mut thread, &access, || Ok::<_, ()>(native(&backing, 0, 8)))
            .unwrap();
        assert_eq!(owner.read_snapshot(base, 8).unwrap(), [0; 8]);
        other_owner
            .protect_data(
                other_base,
                crate::memory::host_page_size().unwrap() as u64,
                true,
                false,
            )
            .unwrap();
        assert_eq!(
            system
                .reservations
                .store(&mut thread, &access, || Ok::<_, ()>(()))
                .unwrap(),
            monitor::ConditionalWrite::Stored(())
        );
        system
            .reservations
            .load(&mut thread, &access, || Ok::<_, ()>(native(&backing, 0, 8)))
            .unwrap();
        owner
            .protect_data(
                base,
                crate::memory::host_page_size().unwrap() as u64,
                true,
                false,
            )
            .unwrap();
        let current = domain.registered_views();
        let mapping = current
            .iter()
            .find(|mapping| mapping.view.guest_base() == base)
            .unwrap();
        let replacement = monitor::Access::new(
            Arc::clone(&backing),
            Arc::clone(&mapping.monitor),
            0,
            8,
            monitor::Shape::Scalar,
        )
        .unwrap();
        assert_eq!(
            system
                .reservations
                .store(&mut thread, &replacement, || -> Result<(), ()> {
                    panic!("retired view reservation must not reach write")
                })
                .unwrap(),
            monitor::ConditionalWrite::Failed
        );
    }
}
