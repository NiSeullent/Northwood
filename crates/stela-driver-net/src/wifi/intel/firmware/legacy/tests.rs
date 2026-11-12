// SPDX-License-Identifier: GPL-2.0-or-later
use super::*;
use crate::wifi::intel::firmware::{
    API_NEW_VERSION, CPUS, HEADER_BYTES, MAGIC, PAGING, SEC_INIT, SEC_RT,
};

mod alive;

fn tlv(image: &mut Vec<u8>, kind: u32, bytes: &[u8]) {
    image.extend(kind.to_le_bytes());
    image.extend((bytes.len() as u32).to_le_bytes());
    image.extend(bytes);
    image.resize(image.len().next_multiple_of(4), 0);
}

pub(super) fn section(image: &mut Vec<u8>, kind: u32, offset: u32, data: &[u8]) {
    let mut bytes = offset.to_le_bytes().to_vec();
    bytes.extend(data);
    tlv(image, kind, &bytes);
}

pub(super) fn image() -> Vec<u8> {
    let mut bytes = alloc::vec![0; HEADER_BYTES];
    bytes[4..8].copy_from_slice(&MAGIC.to_le_bytes());
    bytes[72..76].copy_from_slice(&API.to_le_bytes());
    let mut api = 0u32.to_le_bytes().to_vec();
    api.extend((1u32 << API_NEW_VERSION).to_le_bytes());
    tlv(&mut bytes, super::super::API, &api);
    tlv(&mut bytes, CPUS, &2u32.to_le_bytes());
    tlv(&mut bytes, PAGING, &4096u32.to_le_bytes());
    section(&mut bytes, SEC_INIT, 0x404000, &[1]);
    section(&mut bytes, SEC_RT, 0x456000, &[2; 65536]);
    section(&mut bytes, SEC_RT, CPU_SEPARATOR, &[0; 4]);
    section(&mut bytes, SEC_RT, 0x80448000, &[3]);
    section(&mut bytes, SEC_RT, PAGING_SEPARATOR, &[0; 4]);
    section(&mut bytes, SEC_RT, 0, &[4]);
    section(&mut bytes, SEC_RT, 0x1000000, &[5; 4096]);
    section(&mut bytes, SEC_RT, CPU_SEPARATOR, &[0; 4]);
    bytes
}

fn init_steps<'a>(plan: &LoadPlan<'a>) -> Vec<InitStep<'a>> {
    let mut steps = Vec::new();
    plan.upload_init(NonZeroUsize::new(2).unwrap(), |step| {
        steps.push(step);
        Ok::<(), core::convert::Infallible>(())
    })
    .unwrap();
    steps
}

#[test]
fn init_steps_order_chunks_status_and_interrupts_for_both_cpus() {
    let mut bytes = image();
    section(&mut bytes, SEC_INIT, 0x41000, b"abc");
    section(&mut bytes, SEC_INIT, CPU_SEPARATOR, &[]);
    section(&mut bytes, SEC_INIT, 0xc0080000, b"de");
    section(&mut bytes, SEC_INIT, 0xc0080010, b"f");
    let firmware = Firmware::parse(&bytes).unwrap();
    let plan = LoadPlan::from_firmware(&firmware).unwrap();
    let chunk = |cpu, section, destination, data| InitStep::Transfer {
        cpu,
        section,
        transfer: Transfer { destination, data },
    };
    let progress = |cpu, section, before, after| InitStep::SectionLoaded {
        cpu,
        section,
        status: InitStatus { before, after },
    };
    let complete = |cpu, before, after| InitStep::CpuComplete {
        cpu,
        status: InitStatus { before, after },
    };
    assert_eq!(
        init_steps(&plan),
        [
            InitStep::CheckFresh,
            chunk(Cpu::First, 0, 0x404000, &[1]),
            progress(Cpu::First, 0, 0, 1),
            chunk(Cpu::First, 1, 0x41000, b"ab"),
            chunk(Cpu::First, 1, 0x41002, b"c"),
            progress(Cpu::First, 1, 1, 3),
            InitStep::EnableInterrupts(Cpu::First),
            complete(Cpu::First, 3, 0xffff),
            chunk(Cpu::Second, 0, 0xc0080000, b"de"),
            progress(Cpu::Second, 0, 0xffff, 0x0001_ffff),
            chunk(Cpu::Second, 1, 0xc0080010, b"f"),
            progress(Cpu::Second, 1, 0x0001_ffff, 0x0003_ffff),
            InitStep::EnableInterrupts(Cpu::Second),
            complete(Cpu::Second, 0x0003_ffff, u32::MAX),
        ]
    );
}

#[test]
fn init_stops_at_any_failed_step_and_completes_an_empty_cpu2_on_success() {
    let bytes = image();
    let firmware = Firmware::parse(&bytes).unwrap();
    let plan = LoadPlan::from_firmware(&firmware).unwrap();
    let expected = init_steps(&plan);
    assert_eq!(
        expected[expected.len() - 2..],
        [
            InitStep::EnableInterrupts(Cpu::Second),
            InitStep::CpuComplete {
                cpu: Cpu::Second,
                status: InitStatus {
                    before: 0xffff,
                    after: u32::MAX
                },
            },
        ]
    );
    assert!(!expected.iter().any(|step| matches!(
        step,
        InitStep::Transfer {
            cpu: Cpu::Second,
            ..
        }
    )));
    for fail in 0..expected.len() {
        let mut seen = Vec::new();
        let result = plan.upload_init(NonZeroUsize::new(2).unwrap(), |step| {
            seen.push(step);
            if seen.len() == fail + 1 {
                Err(fail)
            } else {
                Ok(())
            }
        });
        assert_eq!(result, Err(InitError::Step(fail)));
        assert_eq!(seen, expected[..=fail]);
    }
}

#[test]
fn init_rejects_a_completion_collision_or_runtime_sections_before_io() {
    let mut bytes = image();
    for index in 1..15 {
        section(&mut bytes, SEC_INIT, 0x41000 + index, &[8]);
    }
    let firmware = Firmware::parse(&bytes).unwrap();
    let mut plan = LoadPlan::from_firmware(&firmware).unwrap();
    let steps = init_steps(&plan);
    assert!(steps.contains(&InitStep::SectionLoaded {
        cpu: Cpu::First,
        section: 14,
        status: InitStatus {
            before: 0x3fff,
            after: 0x7fff
        },
    }));
    core::mem::swap(&mut plan.init, &mut plan.runtime);
    fn no_io(_: InitStep<'_>) -> Result<(), ()> {
        panic!("invalid INIT started an upload")
    }
    assert_eq!(
        plan.upload_init(NonZeroUsize::MIN, no_io),
        Err(InitError::Plan(Error::SectionLayout))
    );
    section(&mut bytes, SEC_INIT, 0x42000, &[9]);
    let firmware = Firmware::parse(&bytes).unwrap();
    let plan = LoadPlan::from_firmware(&firmware).unwrap();
    assert_eq!(
        plan.upload_init(NonZeroUsize::MIN, no_io),
        Err(InitError::Plan(Error::ResourceLimit))
    );

    let mut bytes = image();
    section(&mut bytes, SEC_INIT, CPU_SEPARATOR, &[]);
    for index in 0..16 {
        section(&mut bytes, SEC_INIT, 0xc0080000 + index, &[10]);
    }
    let firmware = Firmware::parse(&bytes).unwrap();
    let plan = LoadPlan::from_firmware(&firmware).unwrap();
    assert_eq!(
        plan.upload_init(NonZeroUsize::MIN, no_io),
        Err(InitError::Plan(Error::ResourceLimit))
    );
}

#[test]
fn regular_completes_both_resident_cpus_without_publishing_paging() {
    let bytes = image();
    let firmware = Firmware::parse(&bytes).unwrap();
    let plan = LoadPlan::from_firmware(&firmware).unwrap();
    let mut steps = Vec::new();
    let capacity = NonZeroUsize::new(FH_MAX_TRANSFER).unwrap();
    plan.upload_regular(capacity, |step| {
        steps.push(step);
        Ok::<(), core::convert::Infallible>(())
    })
    .unwrap();
    assert_eq!(
        steps,
        [
            InitStep::CheckFresh,
            InitStep::Transfer {
                cpu: Cpu::First,
                section: 0,
                transfer: Transfer {
                    destination: 0x456000,
                    data: plan.runtime.cpu1[0].data
                },
            },
            InitStep::SectionLoaded {
                cpu: Cpu::First,
                section: 0,
                status: InitStatus {
                    before: 0,
                    after: 1
                },
            },
            InitStep::EnableInterrupts(Cpu::First),
            InitStep::CpuComplete {
                cpu: Cpu::First,
                status: InitStatus {
                    before: 1,
                    after: 0xffff
                },
            },
            InitStep::Transfer {
                cpu: Cpu::Second,
                section: 0,
                transfer: Transfer {
                    destination: 0x80448000,
                    data: &[3]
                },
            },
            InitStep::SectionLoaded {
                cpu: Cpu::Second,
                section: 0,
                status: InitStatus {
                    before: 0xffff,
                    after: 0x0001_ffff
                },
            },
            InitStep::EnableInterrupts(Cpu::Second),
            InitStep::CpuComplete {
                cpu: Cpu::Second,
                status: InitStatus {
                    before: 0x0001_ffff,
                    after: u32::MAX
                },
            },
        ]
    );

    let mut seen = Vec::new();
    let failed = plan.upload_regular(capacity, |step| {
        seen.push(step);
        if matches!(
            step,
            InitStep::Transfer {
                cpu: Cpu::Second,
                ..
            }
        ) {
            Err(())
        } else {
            Ok(())
        }
    });
    assert_eq!(failed, Err(InitError::Step(())));
    assert_eq!(seen, steps[..6]);
}

#[test]
fn regular_refuses_unprepared_cpu2_before_transport_effects() {
    let bytes = image();
    let firmware = Firmware::parse(&bytes).unwrap();
    let mut plan = LoadPlan::from_firmware(&firmware).unwrap();
    let cpu2 = plan.runtime.cpu2[0];
    plan.runtime.cpu2.clear();
    fn no_io(_: InitStep<'_>) -> Result<(), ()> {
        panic!("invalid REGULAR started an upload")
    }
    assert_eq!(
        plan.upload_regular(NonZeroUsize::MIN, no_io),
        Err(InitError::Plan(Error::SectionLayout))
    );
    plan.runtime.cpu2.resize(MAX_CPU_SECTIONS, cpu2);
    assert_eq!(
        plan.upload_regular(NonZeroUsize::MIN, no_io),
        Err(InitError::Plan(Error::ResourceLimit))
    );
}

#[test]
fn only_legacy_intel_pci_ids_select_candidate_firmware() {
    for id in [0x30dc, 0x31dc, 0x9df0, 0xa370] {
        assert_eq!(Sku::from_pci(0x8086, id), Some(Sku::Cnvi9000));
    }
    for id in [0x2526, 0x271b, 0x271c] {
        assert_eq!(Sku::from_pci(0x8086, id), Some(Sku::WirelessAc9260));
    }
    assert_eq!(Sku::from_pci(0x8086, 0x2723), None);
    assert_eq!(Sku::from_pci(0x14c3, 0x31dc), None);
}

#[test]
fn large_cpu_payloads_and_paging_are_separate_from_init() {
    let bytes = image();
    let firmware = Firmware::parse(&bytes).unwrap();
    let plan = LoadPlan::from_firmware(&firmware).unwrap();
    assert_eq!(plan.init.cpu1[0].data, &[1]);
    assert!(plan.init.cpu2.is_empty());
    assert_eq!(plan.runtime.cpu1[0].data.len(), 65536);
    assert_eq!(plan.runtime.cpu2[0].offset, 0x80448000);
    assert_eq!(plan.paging_css.data, &[4]);
    assert_eq!(plan.paging[0].data, &[5; 4096]);
    assert_eq!(
        plan.runtime.cpu1[0].data.as_ptr(),
        firmware.sections()[1].data.as_ptr()
    );
}

#[test]
fn fh_transfers_split_address_modes_and_encode_a_bounded_iova() {
    let mut bytes = image();
    section(&mut bytes, SEC_INIT, EXTENDED_START - 16, &[6; 0x18020]);
    let firmware = Firmware::parse(&bytes).unwrap();
    let plan = LoadPlan::from_firmware(&firmware).unwrap();
    let transfers: Vec<_> = plan
        .init
        .transfers(Cpu::First, 1, NonZeroUsize::new(0x40000).unwrap())
        .unwrap()
        .collect();
    assert_eq!(
        transfers.iter().map(|t| t.data().len()).collect::<Vec<_>>(),
        [16, 0x18000, 16]
    );
    assert!(!transfers[0].extended_sram());
    assert!(transfers[1].extended_sram());
    assert!(!transfers[2].extended_sram());
    assert_eq!(
        transfers[1].register_writes(0xf_0123_4000),
        Some([
            (0x1e20, 0),
            (0x19c8, EXTENDED_START),
            (0x1948, 0x0123_4000),
            (0x194c, 0xf001_8000),
            (0x1e28, 0x0010_1003),
            (0x1e20, 0x8010_0000),
        ])
    );
    assert!(transfers[1].register_writes(FH_DMA_LIMIT - 1).is_none());
    assert!(transfers[1].register_writes(u64::MAX).is_none());
    assert!(
        transfers[1]
            .register_writes(FH_DMA_LIMIT - 0x18000)
            .is_some()
    );
    assert!(
        plan.init
            .transfers(Cpu::Second, 0, NonZeroUsize::MIN)
            .is_none()
    );
}

#[test]
fn rejects_missing_pages_extra_cpu_delimiters_and_wrong_api() {
    let mut bytes = image();
    section(&mut bytes, SEC_RT, CPU_SEPARATOR, &[0; 4]);
    assert_eq!(
        LoadPlan::from_firmware(&Firmware::parse(&bytes).unwrap()),
        Err(Error::SectionLayout)
    );
    let mut bytes = image();
    bytes.truncate(bytes.len() - 16 - 4108);
    assert_eq!(
        LoadPlan::from_firmware(&Firmware::parse(&bytes).unwrap()),
        Err(Error::SectionLayout)
    );
    let mut bytes = image();
    bytes[72..76].copy_from_slice(&77u32.to_le_bytes());
    assert_eq!(
        LoadPlan::from_firmware(&Firmware::parse(&bytes).unwrap()),
        Err(Error::FirmwareApi {
            expected: 46,
            actual: 77
        })
    );
}

#[test]
#[ignore = "oracle: STELA_IWLWIFI_IMAGES"]
fn pinned_api_46_images_plan_init_runtime_and_paging() {
    extern crate std;
    use crate::bytes::le32;
    let root = std::env::var("STELA_IWLWIFI_IMAGES").expect("actual pinned image directory");
    for sku in [Sku::Cnvi9000, Sku::WirelessAc9260] {
        let bytes = std::fs::read(std::path::Path::new(&root).join(sku.firmware_name())).unwrap();
        let firmware = Firmware::parse(&bytes).unwrap();
        let plan = LoadPlan::from_firmware(&firmware).unwrap();
        alive::check_pinned_alive_sequence(&firmware);
        let bt = crate::wifi::intel::radio::BluetoothConfig::from_firmware(&firmware).unwrap();
        let id = bt.id().wire();
        assert_eq!((id.group, id.opcode, id.version), (1, 0x9b, 0));
        assert_eq!(bt.payload(), &[1, 0, 0, 0, 0x15, 0, 0, 0]);
        let shared_clock = sku == Sku::Cnvi9000;
        let commands = phy::Config::from_firmware(&firmware)
            .unwrap()
            .commands(3, 3, shared_clock)
            .unwrap();
        assert_eq!(commands.antenna().payload(), &3u32.to_le_bytes());
        let config = commands.phy();
        let payload = config.payload();
        assert_eq!(
            le32(payload, 0),
            0x0033_0018 | u32::from(shared_clock) << 31
        );
        assert_eq!(le32(payload, 4), 0x1f8e_51db);
        assert_eq!(le32(payload, 8), 1);
        assert_eq!((plan.init.cpu1.len(), plan.init.cpu2.len()), (9, 0));
        assert_eq!((plan.runtime.cpu1.len(), plan.runtime.cpu2.len()), (4, 4));
        assert_eq!(plan.paging_css.data.len(), 664);
        assert_eq!(plan.paging_css.offset, 0);
        assert_eq!(plan.paging.len(), 1);
        assert_eq!(plan.paging[0].data.len(), 241664);
        assert_eq!(plan.paging[0].offset, 0x01000000);
        assert!(paging::Layout::from_plan(&firmware, &plan).is_ok());
        assert!(
            firmware
                .command_version(phy_db::ID.group, phy_db::ID.opcode)
                .is_none()
        );
        let mut completed = Vec::new();
        plan.upload_init(NonZeroUsize::new(FH_MAX_TRANSFER).unwrap(), |step| {
            match step {
                InitStep::SectionLoaded {
                    cpu,
                    section,
                    status,
                } => {
                    assert_eq!(cpu, Cpu::First);
                    assert_eq!(status.after, (1 << (section + 1)) - 1);
                    completed.push(section);
                }
                InitStep::Transfer { cpu, .. } => assert_eq!(cpu, Cpu::First),
                _ => {}
            }
            Ok::<(), core::convert::Infallible>(())
        })
        .unwrap();
        assert_eq!(completed, [0, 1, 2, 3, 4, 5, 6, 7, 8]);
        let mut bytes_by_cpu = [0, 0];
        let mut completed = Vec::new();
        let mut sentinels = Vec::new();
        plan.upload_regular(NonZeroUsize::new(FH_MAX_TRANSFER).unwrap(), |step| {
            match step {
                InitStep::Transfer { cpu, transfer, .. } => {
                    bytes_by_cpu[usize::from(cpu == Cpu::Second)] += transfer.data().len();
                }
                InitStep::SectionLoaded {
                    cpu,
                    section,
                    status,
                } => {
                    completed.push((cpu, section, status.after));
                }
                InitStep::CpuComplete { cpu, status } => sentinels.push((cpu, status.after)),
                _ => {}
            }
            Ok::<(), core::convert::Infallible>(())
        })
        .unwrap();
        assert_eq!(
            completed,
            [
                (Cpu::First, 0, 0x1),
                (Cpu::First, 1, 0x3),
                (Cpu::First, 2, 0x7),
                (Cpu::First, 3, 0xf),
                (Cpu::Second, 0, 0x0001_ffff),
                (Cpu::Second, 1, 0x0003_ffff),
                (Cpu::Second, 2, 0x0007_ffff),
                (Cpu::Second, 3, 0x000f_ffff),
            ]
        );
        assert_eq!(sentinels, [(Cpu::First, 0xffff), (Cpu::Second, u32::MAX)]);
        for (index, cpu) in [Cpu::First, Cpu::Second].into_iter().enumerate() {
            assert_eq!(
                bytes_by_cpu[index],
                plan.runtime
                    .sections(cpu)
                    .iter()
                    .map(|section| section.data.len())
                    .sum::<usize>()
            );
        }
        let resident = plan.runtime.sections(Cpu::First)[2];
        let chunks: Vec<_> = plan
            .runtime
            .transfers(Cpu::First, 2, NonZeroUsize::new(usize::MAX).unwrap())
            .unwrap()
            .collect();
        assert_eq!(
            chunks
                .iter()
                .map(|chunk| chunk.data().len())
                .collect::<Vec<_>>(),
            [0x20000, resident.data.len() - 0x20000]
        );
    }
}
