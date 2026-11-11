// SPDX-License-Identifier: GPL-2.0-or-later
use super::*;
use crate::wifi::intel::firmware::legacy::{
    LoadPlan,
    adapter::{PagingProgress, RegularAdapter},
};

const OPCODE: u8 = 0x4f;
const PAGE: usize = 4096;
const BLOCK: usize = 32768;
const COMMAND: usize = 152;

pub(super) fn start_regular<'a, 'fw>(
    fixture: &mut Fixture<'a>,
    firmware: &Firmware<'fw>,
) -> RegularAdapter<'a, 'fw, Io> {
    regular::run_init(fixture, firmware);
    handoff_to_regular(fixture, firmware)
}

pub(super) fn handoff_to_regular<'a, 'fw>(
    fixture: &mut Fixture<'a>,
    firmware: &Firmware<'fw>,
) -> RegularAdapter<'a, 'fw, Io> {
    fixture.adapter.stop().unwrap();
    let handoff = fixture.adapter.take_regular_handoff().unwrap();
    let mut regular = handoff.prepare_regular(2, firmware).unwrap();
    // SAFETY: this exclusively restarts the same synthetic, already stopped
    // requester and publishes no DMA addresses in the callback.
    unsafe { regular.start(regular::STAGING, |io| regular::restart(io, fixture, None)) }.unwrap();
    regular
}

pub(super) fn accept_alive(
    fixture: &Fixture<'_>,
    regular: &mut RegularAdapter<'_, '_, Io>,
    index: &mut u16,
) {
    publish_rx_frame(
        &fixture.device,
        &fixture.domain,
        index,
        &alive_frame(scheduler::SRAM_BASE),
    );
    fixture.device.lock().unwrap().scheduler.alive = true;
    assert!(matches!(
        regular.poll_alive(),
        Ok(Progress::Alive {
            image: Image::Regular,
            ..
        })
    ));
}

pub(super) fn post(fixture: &Fixture<'_>, index: &mut u16, packets: &[calibration::Packet<'_>]) {
    publish_rx_frame(
        &fixture.device,
        &fixture.domain,
        index,
        &calibration::buffer(packets),
    );
}

pub(super) fn stop_with_retries(fixture: &Fixture<'_>, regular: &mut RegularAdapter<'_, '_, Io>) {
    let (unmaps, live) = {
        let records = fixture.domain.records.lock().unwrap();
        (records.unmaps, records.live.len())
    };
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
    {
        let mut records = fixture.domain.records.lock().unwrap();
        assert_eq!(records.unmaps, unmaps);
        assert_eq!(records.live.len(), live);
        records.unmap_fail = Some(unmaps);
    }
    assert!(regular.stop().is_err());
    assert!(regular.finish().is_err());
    assert_eq!(fixture.domain.records.lock().unwrap().live.len(), live);
    regular.stop().unwrap();
    let io = regular.finish().unwrap();
    assert!(Arc::ptr_eq(&io.device, &fixture.device));
    assert!(fixture.domain.records.lock().unwrap().live.is_empty());
}

#[test]
#[ignore = "oracle: STELA_IWLWIFI_IMAGES"]
fn pinned_31dc_paging_publishes_exact_css_data_and_keeps_backings_after_ack() {
    let _clock = test_clock::virtualize();
    let bytes = pinned_cnvi9000_api46_image();
    let firmware = Firmware::parse(&bytes).unwrap();
    let plan = LoadPlan::from_firmware(&firmware).unwrap();
    let source = Source::default();
    let mut fixture = Fixture::new(&source);
    let mut regular = start_regular(&mut fixture, &firmware);
    let (maps, unmaps, live) = {
        let records = fixture.domain.records.lock().unwrap();
        (records.maps, records.unmaps, records.live.len())
    };
    let events = fixture.device.lock().unwrap().events.len();
    assert_eq!(regular.start_paging(), Err(Error::State));
    assert!(regular.poll_paging().is_err());
    assert_eq!(fixture.device.lock().unwrap().events.len(), events);
    assert_eq!(fixture.domain.records.lock().unwrap().maps, maps);

    let mut index = 0;
    accept_alive(&fixture, &mut regular, &mut index);
    // A preclosed exact reply in this generation must not admit paging.
    post(&fixture, &mut index, &[(OPCODE, 1, 0, &[])]);
    regular.start_paging().unwrap();
    let pending = nvm::submitted(&fixture);
    assert_eq!(pending.bytes().len(), COMMAND);
    assert_eq!(&pending.bytes()[..8], [OPCODE, 1, 0, 0, 144, 0, 0, 0]);
    let fields: Vec<_> = pending.bytes()[8..]
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
        .collect();
    assert_eq!(&fields[..3], [0x303, 15, 8]);
    let addresses: Vec<_> = fields[3..12]
        .iter()
        .map(|&page| u64::from(page) << 12)
        .collect();
    assert_eq!(fields[12..], [0; 24]);
    assert!(
        addresses
            .iter()
            .all(|&address| address != 0 && address.is_multiple_of(PAGE as u64))
    );
    assert_eq!(
        addresses
            .iter()
            .copied()
            .collect::<alloc::collections::BTreeSet<_>>()
            .len(),
        9
    );
    let css = read_dma::<PAGE>(&fixture.domain, addresses[0], 0);
    assert_eq!(plan.paging_css.data.len(), 664);
    assert_eq!(&css[..664], plan.paging_css.data);
    assert_eq!(&css[664..], [0; PAGE - 664]);
    assert_eq!(plan.paging.len(), 1);
    assert_eq!(plan.paging[0].data.len(), 241664);
    for (data, &address) in plan.paging[0].data.chunks(BLOCK).zip(&addresses[1..]) {
        let block = read_dma::<BLOCK>(&fixture.domain, address, 0);
        assert_eq!(&block[..data.len()], data);
        assert!(block[data.len()..].iter().all(|byte| *byte == 0));
    }
    let remainder = dma_offset(pending.remainder().unwrap()).0;
    {
        let records = fixture.domain.records.lock().unwrap();
        assert_eq!(records.maps - maps, 10);
        assert_eq!(records.live.len(), live + 10);
        assert_eq!(records.permissions[&remainder], Direction::ToDevice);
        for &address in &addresses {
            assert_eq!(
                records.permissions[&dma_offset(address).0],
                Direction::Bidirectional
            );
        }
    }
    assert!(matches!(regular.poll_paging(), Ok(PagingProgress::Pending)));
    post(
        &fixture,
        &mut index,
        &[
            (OPCODE, 0, 0, &[]),
            (OPCODE, 1, 0x2000, &[]),
            (OPCODE, 1, 0x8000, &[]),
            (0x6a, 1, 0, &[]),
        ],
    );
    assert!(matches!(regular.poll_paging(), Ok(PagingProgress::Pending)));
    assert_eq!(fixture.domain.records.lock().unwrap().unmaps, unmaps);

    fixture
        .device
        .lock()
        .unwrap()
        .command
        .complete(&fixture.domain, pending.sequence());
    post(
        &fixture,
        &mut index,
        &[(OPCODE, 1, pending.sequence(), &[])],
    );
    assert!(matches!(
        regular.poll_paging(),
        Ok(PagingProgress::Complete)
    ));
    {
        let records = fixture.domain.records.lock().unwrap();
        assert_eq!(
            records.unmaps,
            unmaps + 1,
            "only the command remainder retires"
        );
        assert_eq!(records.live.len(), live + 9);
        assert!(!records.live.contains_key(&remainder));
        assert!(
            addresses
                .iter()
                .all(|&address| records.live.contains_key(&dma_offset(address).0))
        );
    }
    assert_eq!(regular.start_paging(), Err(Error::State));
    stop_with_retries(&fixture, &mut regular);
}

#[test]
fn duplicate_paging_ack_faults_without_releasing_any_new_dma() {
    let _clock = test_clock::virtualize();
    let bytes = calibration::calibration_image();
    let firmware = Firmware::parse(&bytes).unwrap();
    let source = Source::default();
    let mut fixture = Fixture::new(&source);
    let mut regular = start_regular(&mut fixture, &firmware);
    let mut index = 0;
    accept_alive(&fixture, &mut regular, &mut index);
    regular.start_paging().unwrap();
    let seq = nvm::submitted(&fixture).sequence();
    let unmaps = fixture.domain.records.lock().unwrap().unmaps;
    post(
        &fixture,
        &mut index,
        &[(OPCODE, 1, seq, &[]), (OPCODE, 1, seq, &[])],
    );
    assert!(matches!(
        regular.poll_paging(),
        Err(Error::DuplicateCommandReply)
    ));
    assert_eq!(fixture.domain.records.lock().unwrap().unmaps, unmaps);
    assert_eq!(nvm::submitted(&fixture).sequence(), seq);
    assert!(regular.finish().is_err());
    regular.stop().unwrap();
    assert!(regular.finish().is_ok());
}

#[test]
fn unsupported_paging_abi_has_no_queue_or_dma_publication() {
    let _clock = test_clock::virtualize();
    let mut bytes = calibration::calibration_image();
    let abi = bytes
        .windows(4)
        .rposition(|row| row == [OPCODE, 1, 1, 0])
        .unwrap();
    bytes[abi + 2] = 2;
    let firmware = Firmware::parse(&bytes).unwrap();
    let source = Source::default();
    let mut fixture = Fixture::new(&source);
    let mut regular = start_regular(&mut fixture, &firmware);
    let mut index = 0;
    accept_alive(&fixture, &mut regular, &mut index);
    let maps = fixture.domain.records.lock().unwrap().maps;
    let events = fixture.device.lock().unwrap().events.len();
    assert!(regular.start_paging().is_err());
    assert_eq!(fixture.domain.records.lock().unwrap().maps, maps);
    {
        let device = fixture.device.lock().unwrap();
        assert_eq!(device.peripheral(SCD_TXFACT), 0);
        assert!(
            device.events[events..]
                .iter()
                .all(|event| matches!(event, Event::Read(..)))
        );
    }
    regular.stop().unwrap();
    assert!(regular.finish().is_ok());
}

#[test]
fn posted_paging_doorbell_failure_retains_all_new_dma_until_stop_and_drain() {
    let _clock = test_clock::virtualize();
    let bytes = calibration::calibration_image();
    let firmware = Firmware::parse(&bytes).unwrap();
    let source = Source::default();
    let mut fixture = Fixture::new(&source);
    let mut regular = start_regular(&mut fixture, &firmware);
    let mut index = 0;
    accept_alive(&fixture, &mut regular, &mut index);
    let (unmaps, live) = {
        let records = fixture.domain.records.lock().unwrap();
        (records.unmaps, records.live.len())
    };
    fixture.device.lock().unwrap().command.fail_next_doorbell();
    assert!(matches!(regular.start_paging(), Err(Error::Command(_))));
    let pending = nvm::submitted(&fixture);
    assert_eq!(pending.bytes()[0], OPCODE);
    {
        let records = fixture.domain.records.lock().unwrap();
        assert_eq!(records.unmaps, unmaps);
        assert_eq!(
            records.live.len(),
            live + 3,
            "CSS, one paging block and TB1"
        );
    }
    assert_eq!(regular.start_paging(), Err(Error::State));
    assert!(regular.poll_paging().is_err());
    assert!(regular.finish().is_err());
    stop_with_retries(&fixture, &mut regular);
}
