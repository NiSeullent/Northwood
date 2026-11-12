// SPDX-License-Identifier: GPL-2.0-or-later
use super::*;
use crate::wifi::intel::firmware::legacy::{alive, phy_db::Kind, service};

pub(super) const STAGING: NonZeroUsize = NonZeroUsize::new(0x10000).unwrap();
const SHARED_MASK: u32 = FH_MASK | RX_MASK;
const RF_ENABLED: u32 = 1 << 27;
const CPU2_SYNTHETIC: u32 = 0x80448000;
const RX_PINS: usize = 507;
const TX_PINS: usize = 35;

pub(super) fn run_init(fixture: &mut Fixture<'_>, firmware: &Firmware<'_>) {
    calibration::ready_with_firmware(fixture, firmware, STAGING, true);
    calibration::complete_current_init(fixture);
}

pub(super) fn restart(
    io: &mut Io,
    fixture: &Fixture<'_>,
    fail_destination: Option<u32>,
) -> Result<(), service::Error> {
    assert!(Arc::ptr_eq(&io.device, &fixture.device));
    assert!(Arc::ptr_eq(&io.domain, &fixture.domain));
    let mut device = io.device.lock().unwrap();
    assert!(device.reset_settled && device.drained);
    assert_ne!(device.csr(RESET) & MASTER_DISABLED, 0);
    assert_eq!(device.csr(MASK), 0);
    // The explicitly verified platform restart opens the same quiesced
    // requester in a fresh device epoch. It publishes no new DMA addresses.
    let events = core::mem::take(&mut device.events);
    *device = Device::new();
    device.events = events;
    device.regular_upload = true;
    device.fail_fh_destination = fail_destination;
    device.csr.insert(GP, 1 | RF_ENABLED);
    device.peripheral.insert(0x02c00, scheduler::SRAM_BASE);
    Ok(())
}

fn ring_bases(device: &Device) -> [u64; 3] {
    [FREE_BASE, USED_BASE, STATUS_BASE].map(|register| device.address(register))
}

fn expected_pinned_upload() -> Vec<Event> {
    const CPU1: [(u32, u32); 4] = [
        (0x404000, 696),
        (0x800000, 98304),
        (0, 229376),
        (0x456000, 230624),
    ];
    const CPU2: [(u32, u32); 4] = [
        (0x405000, 696),
        (0xc0080000, 65536),
        (0xc0880000, 32768),
        (0x80448000, 56208),
    ];
    let mut events = vec![
        Event::Write(MASK, FH_MASK),
        Event::PeripheralWrite(CPU_RELEASE, 1 << 24),
    ];
    for (sections, mask, base, shift) in [(CPU1, FH_MASK, 0, 0), (CPU2, SHARED_MASK, 0xffff, 16)] {
        for (index, (start, size)) in sections.into_iter().enumerate() {
            for offset in (0..size).step_by(STAGING.get()) {
                events.push(Event::FhSubmission {
                    destination: start + offset,
                    bytes: (size - offset).min(STAGING.get() as u32),
                    mask,
                });
                events.push(Event::FhComplete);
            }
            events.push(Event::Write(
                LOAD_STATUS,
                base | (((1 << (index + 1)) - 1) << shift),
            ));
        }
        if mask == FH_MASK {
            events.push(Event::Write(MASK, SHARED_MASK));
            events.push(Event::Write(LOAD_STATUS, 0xffff));
        } else {
            events.push(Event::Write(LOAD_STATUS, u32::MAX));
            events.push(Event::Write(MASK, RX_MASK));
        }
    }
    events
}

#[test]
#[ignore = "oracle: STELA_IWLWIFI_IMAGES"]
fn pinned_31dc_handoff_uploads_only_resident_regular_sections_then_receives_fresh_alive() {
    let _clock = test_clock::virtualize();
    let bytes = pinned_cnvi9000_api46_image();
    let firmware = Firmware::parse(&bytes).unwrap();
    let source = Source::default();
    let mut fixture = Fixture::new(&source);
    run_init(&mut fixture, &firmware);

    let old_bases = ring_bases(&fixture.device.lock().unwrap());
    fixture.publish_alive_with_scheduler(scheduler::SRAM_BASE);
    let stale = fixture.adapter.receiver.poll().unwrap().unwrap();
    assert_eq!(
        stale
            .completion(&stale.packets().unwrap().next().unwrap().unwrap())
            .generation,
        1
    );
    fixture.adapter.stop().unwrap();
    assert!(fixture.domain.records.lock().unwrap().live.is_empty());
    let handoff = fixture.adapter.take_regular_handoff().unwrap();
    let before_prepare = fixture.domain.records.lock().unwrap().maps;
    let before_restart = fixture.device.lock().unwrap().events.len();

    let Err(rejected) = handoff.prepare_regular(1, &firmware) else {
        panic!("a REGULAR attempt cannot reuse the INIT generation");
    };
    assert_eq!(
        rejected.error,
        Error::Alive(alive::Error::Generation {
            previous: 1,
            next: 1
        })
    );
    let mut wrong_bytes = bytes.clone();
    wrong_bytes[8] ^= 1;
    let wrong_firmware = Firmware::parse(&wrong_bytes).unwrap();
    let Err(rejected) = rejected.handoff.prepare_regular(2, &wrong_firmware) else {
        panic!("a different firmware image cannot reuse the completed INIT");
    };
    assert_eq!(rejected.error, Error::FirmwareChanged);
    assert_eq!(fixture.domain.records.lock().unwrap().maps, before_prepare);
    assert_eq!(fixture.device.lock().unwrap().events.len(), before_restart);

    let Ok(mut regular) = rejected.handoff.prepare_regular(2, &firmware) else {
        panic!("the stopped owner can prepare a new generation for the same firmware");
    };
    assert_eq!(regular.generation(), 2);
    assert!(regular.poll_alive().is_err());
    assert!(regular.finish().is_err());
    // SAFETY: the callback models a fresh, exclusive, already stopped NIC
    // with the same requester identity and no DMA publication.
    unsafe { regular.start(STAGING, |io| restart(io, &fixture, None)) }.unwrap();
    let expected = expected_pinned_upload();
    {
        let device = fixture.device.lock().unwrap();
        let events = &device.events[before_restart..];
        let upload: Vec<_> = events
            .iter()
            .copied()
            .filter(|event| {
                matches!(
                    event,
                    Event::Write(LOAD_STATUS | MASK, _)
                        | Event::PeripheralWrite(CPU_RELEASE, _)
                        | Event::FhSubmission { .. }
                        | Event::FhComplete
                )
            })
            .collect();
        assert_eq!(upload, expected);
        let fresh_bases = ring_bases(&device);
        for address in fresh_bases {
            assert_ne!(address, 0);
            assert!(!old_bases.contains(&address));
        }
        assert_eq!(device.peripheral(SCD_TXFACT), 0);
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, Event::Write(COMMAND_DOORBELL, _)))
        );
        // RX and FH-RX causes injected during CPU2 must survive service-TX
        // acknowledgement and the final switch to the RX-only mask.
        assert_eq!(device.csr(INT), 1 << 31);
        assert_eq!(device.csr(FH_INT), 1 << 16);
    }
    let chunks = expected
        .iter()
        .filter(|event| matches!(event, Event::FhSubmission { .. }))
        .count();
    {
        let records = fixture.domain.records.lock().unwrap();
        assert_eq!(records.maps - before_prepare, RX_PINS + TX_PINS + chunks);
        assert_eq!(records.live.len(), RX_PINS + TX_PINS);
    }
    let sections: Vec<_> = regular
        .phy_db()
        .sections()
        .map(|section| (section.kind, section.data))
        .collect();
    assert_eq!(sections, [(Kind::Configuration, &[0x10, 0x11][..])]);

    // A retained INIT ALIVE and pending RX causes provide no completion on
    // the new ring. An actual fresh slot must close before ALIVE is accepted.
    assert!(matches!(regular.poll_alive(), Ok(Progress::Pending)));
    assert_eq!(fixture.device.lock().unwrap().csr(INT), 0);
    assert_eq!(fixture.device.lock().unwrap().csr(FH_INT), 0);
    let mut index = 0;
    publish_rx_frame(
        &fixture.device,
        &fixture.domain,
        &mut index,
        &alive_frame(scheduler::SRAM_BASE),
    );
    assert!(matches!(
        regular.poll_alive(),
        Ok(Progress::Alive {
            image: Image::Regular,
            ..
        })
    ));
    assert!(regular.poll_alive().is_err());
    regular.stop().unwrap();
    let io = regular.finish().unwrap();
    assert!(Arc::ptr_eq(&io.device, &fixture.device));
    assert!(fixture.domain.records.lock().unwrap().live.is_empty());
}

#[test]
fn partial_restart_failure_never_exposes_new_dma_and_retains_the_owner_for_stop() {
    let _clock = test_clock::virtualize();
    let bytes = calibration::calibration_image();
    let firmware = Firmware::parse(&bytes).unwrap();
    let source = Source::default();
    let mut fixture = Fixture::new(&source);
    run_init(&mut fixture, &firmware);
    fixture.adapter.stop().unwrap();
    let handoff = fixture.adapter.take_regular_handoff().unwrap();
    let mut regular = handoff.prepare_regular(2, &firmware).unwrap();
    let before = fixture.device.lock().unwrap().events.len();
    let (maps, unmaps) = {
        let records = fixture.domain.records.lock().unwrap();
        assert_eq!(records.live.len(), RX_PINS);
        (records.maps, records.unmaps)
    };
    // SAFETY: the callback performs a partial restart on the same synthetic
    // stopped owner, reports the injected failure, and publishes no DMA.
    let result = unsafe {
        regular.start(STAGING, |io| {
            restart(io, &fixture, None)?;
            Err(AccessError::Mapping.into())
        })
    };
    assert_eq!(
        result,
        Err(Error::Service(service::Error::Access(AccessError::Mapping)))
    );
    {
        let mut device = fixture.device.lock().unwrap();
        assert!(device.regular_upload);
        assert_eq!(ring_bases(&device), [0; 3]);
        assert!(device.events[before..].is_empty());
        device.ack_master = false;
    }
    assert_eq!(fixture.domain.records.lock().unwrap().maps, maps);
    assert!(regular.finish().is_err());
    assert!(regular.stop().is_err());
    assert_eq!(fixture.domain.records.lock().unwrap().unmaps, unmaps);
    fixture.device.lock().unwrap().ack_master = true;
    regular.stop().unwrap();
    assert!(
        !fixture.device.lock().unwrap().events[before..]
            .iter()
            .any(|event| matches!(event, Event::Drain(_)))
    );
    let io = regular.finish().unwrap();
    assert!(Arc::ptr_eq(&io.device, &fixture.device));
    assert!(fixture.domain.records.lock().unwrap().live.is_empty());
}

#[test]
fn cpu2_posted_failure_keeps_the_regular_requester_and_all_pins_until_stop() {
    let _clock = test_clock::virtualize();
    let bytes = calibration::calibration_image();
    let firmware = Firmware::parse(&bytes).unwrap();
    let source = Source::default();
    let mut fixture = Fixture::new(&source);
    run_init(&mut fixture, &firmware);
    fixture.adapter.stop().unwrap();
    let handoff = fixture.adapter.take_regular_handoff().unwrap();
    let Ok(mut regular) = handoff.prepare_regular(2, &firmware) else {
        panic!("same stopped owner and firmware must prepare REGULAR");
    };
    // SAFETY: same stopped fake requester, with one posted CPU2 failure.
    assert!(
        unsafe { regular.start(STAGING, |io| restart(io, &fixture, Some(CPU2_SYNTHETIC))) }
            .is_err()
    );
    {
        let device = fixture.device.lock().unwrap();
        assert!(device.active_fh);
        assert_eq!(device.csr(LOAD_STATUS), 0xffff);
        assert_eq!(device.csr(MASK), SHARED_MASK);
        assert!(device.events.contains(&Event::FhSubmission {
            destination: CPU2_SYNTHETIC,
            bytes: 1,
            mask: SHARED_MASK
        }));
    }
    assert!(regular.poll_alive().is_err());
    assert!(regular.finish().is_err());
    // SAFETY: an already attempted transport cannot call the callback again.
    assert_eq!(
        unsafe { regular.start(STAGING, |_| panic!("faulted owner restarted")) },
        Err(Error::State)
    );
    let (unmaps, live) = {
        let records = fixture.domain.records.lock().unwrap();
        (records.unmaps, records.live.len())
    };
    assert_eq!(live, RX_PINS + TX_PINS + 1);

    fixture.device.lock().unwrap().ack_master = false;
    assert!(regular.stop().is_err());
    assert_eq!(fixture.domain.records.lock().unwrap().unmaps, unmaps);
    {
        let mut device = fixture.device.lock().unwrap();
        device.ack_master = true;
        device.fail_drain = true;
    }
    assert_eq!(
        regular.stop(),
        Err(Error::Drain(DmaError::DrainUnavailable))
    );
    assert!(regular.finish().is_err());
    assert_eq!(fixture.domain.records.lock().unwrap().live.len(), live);
    fixture.domain.records.lock().unwrap().unmap_fail = Some(unmaps);
    assert!(regular.stop().is_err());
    assert!(regular.finish().is_err());
    regular.stop().unwrap();
    let io = regular.finish().unwrap();
    assert!(Arc::ptr_eq(&io.device, &fixture.device));
    assert!(fixture.domain.records.lock().unwrap().live.is_empty());
}
