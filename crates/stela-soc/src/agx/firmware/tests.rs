use super::*;
use crate::Result;
use crate::agx::initdata::tests::{BASE, RINGS, ROOT, kext_root};
use crate::agx::ring::tests::Flat;
use crate::agx::ring::{READ, WRITE};
use core::ops::Range;
use stela_contracts::refuse;
use stela_soc_data::agx::generation::{G13G, G13X, G14G, G14X, G15G, MAGIC_G13_G14G};

fn doorbell_payload(kind: u64, low: u64) -> u64 {
    ((DOORBELL_FLAG | kind) << TYPE_SHIFT) | low
}

fn work(pipe: u64, index: u64) -> u64 {
    doorbell_payload(DOORBELL_WORK, (index & 7) << 2 | pipe)
}

fn booted_gfx(family: &'static Family) -> Coprocessor {
    let mut firmware = Coprocessor::new(family, stela_formats::builds::DEFAULT).unwrap();
    firmware.start(&mut Vec::new()).unwrap();
    for payload in [
        2 << 52 | 12 | 12 << 16,
        8 << 52,
        8 << 52 | 1 << 51 | 1 << 32,
    ] {
        firmware
            .receive(
                Mail {
                    endpoint: 0,
                    payload,
                },
                &mut Vec::new(),
            )
            .unwrap();
    }
    for endpoint in [MESSAGE_ENDPOINT, DOORBELL_ENDPOINT] {
        firmware
            .receive(
                Mail {
                    endpoint: 0,
                    payload: 5 << 52 | u64::from(endpoint) << 32 | 2,
                },
                &mut Vec::new(),
            )
            .unwrap();
    }
    firmware
}

#[test]
fn native_power_constructor_keeps_the_concrete_family_and_callback_refusal() {
    use crate::agx::{generation::GPUS, sgx::Sgx};
    for family in [&G13G, &G13X, &G14G, &G14X] {
        let gpu = GPUS
            .iter()
            .find(|gpu| gpu.family == family && gpu.admitted().is_ok())
            .unwrap();
        let mut sgx = Sgx::new(gpu).unwrap();
        let power = sgx.firmware_power(&[(0x204000000, 0x4000000, 0)]).unwrap();
        let mut firmware =
            Coprocessor::with_power(family, power.clone(), stela_formats::builds::DEFAULT).unwrap();
        let other = if family == &G14G { &G13G } else { &G14G };
        assert!(
            Coprocessor::with_power(other, power.clone(), stela_formats::builds::DEFAULT).is_err()
        );
        // A truthful target binding alone does not run the unimplemented
        // config-dependent callback effects.
        firmware.protocol.0.borrow_mut().initdata_va = Some(ROOT);
        let mut outbox = Vec::new();
        let sleep = Mail {
            endpoint: 0,
            payload: 6 << 52 | 0x201,
        };
        let reason = firmware.receive(sleep, &mut outbox).unwrap_err();
        assert_eq!(firmware.error_reply(sleep, &reason), None);
        firmware.protocol.0.borrow_mut().initdata_checked = true;
        assert_eq!(
            firmware.receive(sleep, &mut outbox),
            Err(refuse!(
                "GFX initialized IOP sleep runs no native power callbacks"
            ))
        );
        assert!(outbox.is_empty());
        if family == &G13G || family == &G14G {
            use stela_contracts::interrupt::InterruptLatch;
            use {
                stela_contracts::bus::Access, stela_contracts::bus::MmioDevice,
                stela_contracts::bus::Width,
            };
            let mut latch = InterruptLatch::new();
            let mut access = Access::new(0, 0, &mut latch);
            sgx.write(0xd04074, Width::Word, 1, &mut access).unwrap();
            assert!(
                firmware
                    .reset()
                    .unwrap_err()
                    .to_string()
                    .contains("restore ownership")
            );
            assert_eq!(
                firmware.protocol.with(|state| state.initdata_va),
                Some(ROOT)
            );
            sgx.write(0xd04074, Width::Word, 0, &mut access).unwrap();
            sgx.write(0xd06020, Width::Word, 1, &mut access).unwrap();
            firmware.reset().unwrap();
            assert_eq!(firmware.protocol.with(|state| state.initdata_va), None);
        }
    }
}

#[test]
fn initialized_sleep_waits_for_startup_then_acknowledges_with_state_kept_on_every_family() {
    for family in [&G13G, &G13X, &G14G, &G14X, &G15G] {
        for sleep in [0, 0x201, 0x202] {
            let mut firmware = booted_gfx(family);
            let mut outbox = Vec::new();
            let power = |state| Mail {
                endpoint: 0,
                payload: 6 << 52 | state,
            };
            // Before initdata there are no registered native callbacks.
            firmware.receive(power(sleep), &mut outbox).unwrap();
            assert_eq!(
                outbox,
                [Mail {
                    endpoint: 0,
                    payload: 7 << 52 | sleep
                }]
            );
            outbox.clear();
            for message in [
                Mail {
                    endpoint: MESSAGE_ENDPOINT,
                    payload: MSG_INIT << TYPE_SHIFT | ROOT & INIT_VA_BITS,
                },
                Mail {
                    endpoint: DOORBELL_ENDPOINT,
                    payload: work(0, 0),
                },
            ] {
                assert_eq!(
                    firmware.receive(message, &mut outbox),
                    Err(refuse!(
                        "GFX application message while the IOP is not awake"
                    ))
                );
            }
            assert!(!firmware.dma_pending());
            assert!(outbox.is_empty());
            firmware.receive(power(0x220), &mut outbox).unwrap();
            outbox.clear();
            let mut memory = Flat::new(BASE, 0x20000);
            kext_root(&mut memory);
            memory.put_u64(ROOT, family.initdata_magic.unwrap());
            firmware
                .receive(
                    Mail {
                        endpoint: MESSAGE_ENDPOINT,
                        payload: MSG_INIT << TYPE_SHIFT | ROOT & INIT_VA_BITS,
                    },
                    &mut outbox,
                )
                .unwrap();
            for initialized in [false, true] {
                if initialized {
                    firmware
                        .protocol
                        .0
                        .borrow_mut()
                        .dma(&mut memory, &mut outbox)
                        .unwrap();
                }
                let bytes = memory.bytes.clone();
                let reason = firmware.receive(power(sleep), &mut outbox).unwrap_err();
                assert!(outbox.is_empty());
                let ack = Mail {
                    endpoint: 0,
                    payload: 7 << 52 | sleep,
                };
                assert_eq!(
                    firmware.error_reply(power(sleep), &reason),
                    initialized.then(|| alloc::vec![ack])
                );
                let state = if initialized { sleep } else { 0x220 };
                assert_eq!(firmware.rtkit.iop_power(), state);
                assert_eq!(memory.bytes, bytes);
                assert_eq!(firmware.dma_pending(), !initialized);
                // A repeated wake has no native transition or generation.
                firmware.receive(power(0x220), &mut outbox).unwrap();
                assert_eq!(
                    outbox,
                    [Mail {
                        endpoint: 0,
                        payload: 7 << 52 | 0x220
                    }]
                );
                outbox.clear();
                assert!(firmware.receive(power(0x10), &mut outbox).is_err());
                assert!(outbox.is_empty());
                assert_eq!(firmware.rtkit.iop_power(), 0x220);
                assert_eq!(memory.bytes, bytes);
            }
        }
    }
}

#[test]
fn a_sleep_after_a_refused_start_is_acknowledged_and_a_later_valid_root_starts() {
    for sleep in [0, 0x201, 0x202] {
        let mut firmware = booted_gfx(&G14G);
        let mut outbox = Vec::new();
        let power = |state| Mail {
            endpoint: 0,
            payload: 6 << 52 | state,
        };
        firmware.receive(power(0x220), &mut outbox).unwrap();
        outbox.clear();
        // The root's first word is not the kext's magic.
        let mut memory = Flat::new(BASE, 0x20000);
        kext_root(&mut memory);
        memory.put_u64(ROOT, 0);
        for message in [
            Mail {
                endpoint: MESSAGE_ENDPOINT,
                payload: MSG_INIT << TYPE_SHIFT | ROOT & INIT_VA_BITS,
            },
            Mail {
                endpoint: DOORBELL_ENDPOINT,
                payload: doorbell_payload(DOORBELL_FWCTL, 0x11),
            },
        ] {
            firmware.receive(message, &mut outbox).unwrap();
        }
        // Before the first DMA step the start is still in progress.
        let reason = firmware.receive(power(sleep), &mut outbox).unwrap_err();
        assert_eq!(firmware.error_reply(power(sleep), &reason), None);
        let bytes = memory.bytes.clone();
        let refusal = firmware
            .protocol
            .0
            .borrow_mut()
            .dma(&mut memory, &mut outbox)
            .unwrap_err();
        assert!(refusal.to_string().contains("lacks its magic"));
        // The refused start wrote nothing and still asks for DMA steps, yet a
        // sleep does not wait for it or for the doorbell it cannot run.
        assert_eq!(memory.bytes, bytes);
        assert!(firmware.dma_pending());
        let reason = firmware.receive(power(sleep), &mut outbox).unwrap_err();
        assert!(outbox.is_empty());
        assert_eq!(
            firmware.error_reply(power(sleep), &reason),
            Some(alloc::vec![Mail {
                endpoint: 0,
                payload: 7 << 52 | sleep,
            }])
        );
        assert_eq!(firmware.rtkit.iop_power(), sleep);
        // A root that becomes valid starts the firmware, and a sleep then
        // waits for the queued doorbell again.
        firmware.receive(power(0x220), &mut outbox).unwrap();
        outbox.clear();
        memory.put_u64(ROOT, G14G.initdata_magic.unwrap());
        let _ = firmware
            .protocol
            .0
            .borrow_mut()
            .dma(&mut memory, &mut outbox);
        assert!(firmware.protocol.with(|state| state.initdata_checked));
        assert!(!firmware.protocol.with(|state| state.start_refused));
    }
}

const CONTROL_STATE: u64 = BASE + 0x8000;
const CONTROL_ENTRIES: u64 = BASE + 0xa000;

/// A checked root whose device-control ring holds `entries`.
fn with_device_control(entries: &[[u8; 0x40]]) -> (State, Flat) {
    with_device_control_on(&G14G, entries).unwrap()
}

/// `family`'s firmware, having read the root, with `entries` on its
/// device-control ring.
fn with_device_control_on(
    family: &'static Family,
    entries: &[[u8; 0x40]],
) -> Result<(State, Flat)> {
    let mut memory = Flat::new(BASE, 0x20000);
    kext_root(&mut memory);
    if let Some(magic) = family.initdata_magic {
        memory.put_u64(ROOT, magic);
    }
    memory.put_descriptor(RINGS + 0x1a0, CONTROL_STATE, CONTROL_ENTRIES);
    for (n, entry) in entries.iter().enumerate() {
        memory.put(CONTROL_ENTRIES + n as u64 * 0x40, entry);
    }
    memory.put_u32(CONTROL_STATE + WRITE, entries.len() as u32);
    let mut state = State::new(family);
    state
        .init((MSG_INIT << TYPE_SHIFT) | (ROOT & INIT_VA_BITS))
        .unwrap();
    state.dma(&mut memory, &mut Vec::new())?;
    Ok((state, memory))
}

fn power(word: u32) -> [u8; 0x40] {
    let mut entry = [0; 0x40];
    entry[..4].copy_from_slice(&DEVICE_CONTROL_POWER.to_le_bytes());
    entry[4..8].copy_from_slice(&word.to_le_bytes());
    entry
}

/// Fail before a selected startup write; record only committed writes.
struct StartupMemory {
    flat: Flat,
    fail_at: Option<usize>,
    writes: Vec<(u64, u32)>,
    reads: usize,
}

impl Shared for StartupMemory {
    fn read(&mut self, va: u64, data: &mut [u8]) -> Result<()> {
        self.reads += 1;
        self.flat.read(va, data)
    }
    fn write(&mut self, va: u64, data: &[u8]) -> Result<()> {
        if self.fail_at == Some(self.writes.len()) {
            return Err(refuse!("startup status write fault"));
        }
        self.flat.write(va, data)?;
        self.writes
            .push((va, u32::from_le_bytes(data.try_into().unwrap())));
        Ok(())
    }
    fn gpu_read(
        &mut self,
        context: usize,
        va: u64,
        data: &mut [u8],
    ) -> Result<(), stela_contracts::dma::DmaError> {
        self.flat.gpu_read(context, va, data)
    }
    fn gpu_write(
        &mut self,
        context: usize,
        va: u64,
        data: &[u8],
    ) -> Result<(), stela_contracts::dma::DmaError> {
        self.flat.gpu_write(context, va, data)
    }
}

#[test]
fn cold_start_status_owns_each_committed_word_across_faults_for_every_modeled_family() {
    let status = BASE + 0x4000;
    for family in [&G13G, &G13X, &G14G, &G14X, &G15G] {
        for fault in 0..STARTUP_STATUS.len() {
            let mut flat = Flat::new(BASE, 0x20000);
            kext_root(&mut flat);
            flat.put_u64(ROOT, family.initdata_magic.unwrap());
            flat.put(status, &[0xa5; 0x20]);
            let mut memory = StartupMemory {
                flat,
                fail_at: Some(fault),
                writes: Vec::new(),
                reads: 0,
            };
            let protocol = Protocol(Rc::new(RefCell::new(State::new(family))));
            let mut endpoint = Message(protocol.clone());
            let mut outbox = Vec::new();
            endpoint
                .receive(
                    MSG_INIT << TYPE_SHIFT | (ROOT & INIT_VA_BITS),
                    &mut Vec::new(),
                )
                .unwrap();
            assert_eq!(
                protocol.0.borrow_mut().dma(&mut memory, &mut outbox),
                Err(refuse!("startup status write fault"))
            );
            assert_eq!(memory.writes.len(), fault);
            assert!(outbox.is_empty());
            assert!(protocol.with(State::dma_pending));
            assert!(!protocol.with(|state| state.initdata_checked));
            assert!(endpoint.reset().is_err());
            // An accepted root and committed prefix survive repeated faults.
            let reads = memory.reads;
            memory.flat.put_u64(ROOT, 0);
            memory.flat.put_u64(ROOT + 0xa8, BASE + 0x8000);
            assert!(
                protocol
                    .0
                    .borrow_mut()
                    .dma(&mut memory, &mut outbox)
                    .is_err()
            );
            assert_eq!(memory.reads, reads);
            assert_eq!(memory.writes.len(), fault);
            memory.fail_at = None;
            protocol
                .0
                .borrow_mut()
                .dma(&mut memory, &mut outbox)
                .unwrap();
            let expected = STARTUP_STATUS.map(|(offset, value)| (status + offset, value));
            assert_eq!(memory.writes, expected);
            assert_eq!(memory.reads, reads);
            assert!(outbox.is_empty());
            assert!(!protocol.with(State::dma_pending));
            assert!(protocol.with(|state| state.initdata_checked));
            assert_eq!(memory.flat.read_u32(status + STATUS_POWERED).unwrap(), 1);
            assert_eq!(memory.flat.read_u32(status + STATUS_READY).unwrap(), 1);
            assert_eq!(memory.flat.read_u32(status + STATUS_GENERATION).unwrap(), 0);
            let mut untouched = [0; 0x10];
            memory.flat.read(status, &mut untouched).unwrap();
            assert_eq!(untouched, [0xa5; 0x10]);
            assert_eq!(memory.flat.read_u32(status + 0x18).unwrap(), 0xa5a5_a5a5);
            // Idle cold-reset starts root validation and publication again.
            endpoint.reset().unwrap();
            endpoint
                .receive(
                    MSG_INIT << TYPE_SHIFT | (ROOT & INIT_VA_BITS),
                    &mut Vec::new(),
                )
                .unwrap();
            assert_eq!(
                protocol
                    .0
                    .borrow_mut()
                    .dma(&mut memory, &mut outbox)
                    .unwrap_err()
                    .what(),
                "GFX initdata root lacks its magic"
            );
            assert_eq!(memory.writes, expected);
            kext_root(&mut memory.flat);
            memory.flat.put_u64(ROOT, family.initdata_magic.unwrap());
            protocol
                .0
                .borrow_mut()
                .dma(&mut memory, &mut outbox)
                .unwrap();
            assert_eq!(&memory.writes[4..], &expected);
        }
    }
}

#[test]
fn physical_reset_preserves_unprocessed_channel_ownership() {
    let (state, mut memory) = with_device_control(&[power(0)]);
    let protocol = Protocol(Rc::new(RefCell::new(state)));
    let mut endpoint = Message(protocol.clone());
    protocol
        .0
        .borrow_mut()
        .doorbell((0x84 << TYPE_SHIFT) | 0x11)
        .unwrap();
    assert!(endpoint.check_reset().is_err());
    assert!(endpoint.reset().is_err());
    assert!(protocol.with(|state| state.initdata_checked));
    protocol
        .0
        .borrow_mut()
        .dma(&mut memory, &mut Vec::new())
        .unwrap();
    assert_eq!(protocol.with(|state| state.power.len()), 1);
    endpoint.check_reset().unwrap();
    endpoint.reset().unwrap();
    assert_eq!(protocol.with(|state| state.initdata_va), None);
    // History remains inspectable; a new initdata session is accepted.
    assert_eq!(protocol.with(|state| state.power.len()), 1);
    endpoint
        .receive(
            MSG_INIT << TYPE_SHIFT | (ROOT & INIT_VA_BITS),
            &mut Vec::new(),
        )
        .unwrap();
    assert!(protocol.with(State::dma_pending));
}

#[test]
fn the_message_endpoint_captures_the_initdata_pointer_once() {
    let mut state = State::new(&G14G);
    let va = 0x4_1234_5000;
    assert!(state.init((MSG_INIT << TYPE_SHIFT) | va).is_ok());
    assert_eq!(state.initdata_va, Some(va));
    // A second initdata, or any other message type, is refused.
    assert!(state.init((MSG_INIT << TYPE_SHIFT) | va).is_err());
    let mut fresh = State::new(&G14G);
    assert!(fresh.init(0x42 << TYPE_SHIFT).is_err());
    // Reserved low bits are not another way to spell the same VA.
    for bit in 44..48 {
        let mut wide = State::new(&G14G);
        assert!(
            wide.init((MSG_INIT << TYPE_SHIFT) | INIT_VA_BITS | (1 << bit))
                .is_err()
        );
        assert_eq!(wide.initdata_va, None);
        assert!(!wide.dma_pending());
    }
}

#[test]
fn doorbells_take_the_kexts_forms_and_queue_firmware_work() {
    let (endpoints, protocol) = endpoints(&G14G);
    let mut by_number = endpoints
        .into_iter()
        .collect::<alloc::collections::BTreeMap<_, _>>();
    let door = by_number.get_mut(&DOORBELL_ENDPOINT).unwrap();
    let mut outbox = Vec::new();
    // submit3DChannel at index 2, then submitDeviceControl's 0x84 << 48 | 0x11.
    door.receive(work(1, 2), &mut outbox).unwrap();
    door.receive(doorbell_payload(DOORBELL_FWCTL, 0x11), &mut outbox)
        .unwrap();
    // The model never answers a doorbell with a message of its own.
    assert!(outbox.is_empty());
    protocol.with(|state| {
        assert_eq!(state.work, [(Pipe::Fragment, 2)]);
        assert_eq!(state.fwctl, 1);
        assert!(state.dma_pending());
    });
}

#[test]
fn malformed_or_unmodeled_doorbells_refuse_explicitly() {
    let mut state = State::new(&G14G);
    // Missing the 0x80 flag.
    assert!(state.doorbell(DOORBELL_WORK << TYPE_SHIFT).is_err());
    // An unmodeled type.
    assert!(state.doorbell(doorbell_payload(5, 0)).is_err());
    // Kind 3 names no pipe; index 4 no submission ring.
    assert!(state.doorbell(work(3, 0)).is_err());
    assert!(state.doorbell(work(0, 4)).is_err());
    // Bits outside the type, index and kind.
    assert!(state.doorbell(work(0, 0) | 0x20).is_err());
    // A firmware-control doorbell other than device control.
    assert!(state.doorbell(doorbell_payload(DOORBELL_FWCTL, 0)).is_err());
    assert!(state.work.is_empty() && !state.dma_pending());
}

#[test]
fn every_reserved_doorbell_bit_refuses_without_queueing_or_counting_work() {
    // kickFirmware masks the type to six bits, fixes bit 55, and uses
    // only bits 4:0 for index/pipe. In particular bit 54 is reserved,
    // even though simply masking to the type would accept 0xc3/0xc4.
    let allowed = (DOORBELL_FLAG | TYPE_MASK) << TYPE_SHIFT | 0x1f;
    for family in [&G13G, &G13X, &G14G, &G14X, &G15G] {
        for bit in 0..64 {
            if allowed & (1 << bit) != 0 {
                continue;
            }
            let mut state = State::new(family);
            for payload in [work(2, 3), doorbell_payload(DOORBELL_FWCTL, 0x11)] {
                assert!(state.doorbell(payload | 1 << bit).is_err());
                assert!(state.work.is_empty());
                assert_eq!(state.fwctl, 0);
                assert!(state.jobs.is_empty());
            }
            // A rejected message must leave the endpoint usable.
            state.doorbell(work(2, 3)).unwrap();
            assert_eq!(state.work, [(Pipe::Compute, 3)]);
            assert_eq!(state.jobs.len(), 1);
        }
    }
}

#[test]
fn power_notices_are_consumed_from_the_device_control_ring() {
    let (mut state, mut memory) = with_device_control(&[power(1), power(0)]);
    state
        .doorbell(doorbell_payload(DOORBELL_FWCTL, 0x11))
        .unwrap();
    state.dma(&mut memory, &mut Vec::new()).unwrap();
    assert_eq!(
        state.power,
        [PowerNotice::PoweringOff, PowerNotice::PoweredOn]
    );
    // waitForDeviceControlCompletion sees the read index reach the write
    // index.
    assert_eq!(memory.read_u32(CONTROL_STATE + READ), Ok(2));
    assert!(!state.dma_pending());
}

const OBJECT_928: u64 = BASE + 0x5000;
const OBJECT_358: u64 = BASE + 0x3000;

/// The G14G interface, which the tests' root carries.
fn layout() -> &'static Layout {
    &G14G.model().unwrap().layout
}

/// The handled-entry count.
fn count() -> u64 {
    OBJECT_928 + layout().device_control_count
}

/// A device-control entry of `kind` whose bytes in `fields` are `fill`.
fn filled(kind: u32, fields: Range<usize>, fill: u8) -> [u8; 0x40] {
    let mut entry = [0; 0x40];
    entry[..4].copy_from_slice(&kind.to_le_bytes());
    entry[fields].fill(fill);
    entry
}

fn of_type(kind: u32) -> [u8; 0x40] {
    filled(kind, 4..4, 0)
}

/// Ring the device-control doorbell and run the firmware's DMA.
fn ring(state: &mut State, memory: &mut Flat) -> Result<()> {
    state.doorbell(doorbell_payload(DOORBELL_FWCTL, 0x11))?;
    state.dma(memory, &mut Vec::new())
}

#[test]
fn the_device_state_query_leaves_the_report_as_the_ap_left_it() {
    let (mut state, mut memory) = with_device_control(&[of_type(7)]);
    // A mask the AP left; the 26A434 firmware has no type-7 handler.
    let sections = OBJECT_928 + 0xe82c;
    memory.put(sections, &[1]);
    ring(&mut state, &mut memory).unwrap();
    let mut mask = [0xff];
    memory.read(sections, &mut mask).unwrap();
    assert_eq!(mask, [1]);
    assert_eq!(state.device_control.get(&7), Some(&1));
    assert_eq!(memory.read_u32(count()), Ok(1));
    assert_eq!(memory.read_u32(CONTROL_STATE + READ), Ok(1));
    assert!(device_control_entry(&filled(7, 8..9, 1)).is_err());
}

#[test]
fn the_configuration_snapshot_is_taken_and_left_uncounted() {
    let snapshot = of_type(0x15);
    let (mut state, mut memory) = with_device_control(&[power(0), snapshot, power(1)]);
    memory.put_u32(count(), 7);
    ring(&mut state, &mut memory).unwrap();
    assert_eq!(state.device_control.get(&0x15), Some(&1));
    // The two power notices are counted; the snapshot is not.
    assert_eq!(memory.read_u32(count()), Ok(9));
    assert_eq!(memory.read_u32(CONTROL_STATE + READ), Ok(3));
    assert!(device_control_entry(&filled(0x15, 4..5, 1)).is_err());
}

/// Every kext form whose firmware handler changes nothing the AP shares.
fn private_forms() -> Vec<[u8; 0x40]> {
    let mut forms: Vec<_> = [
        0x7, 0x12, 0x23, 0x2d, 0xb, 0xc, 0xd, 0x10, 0x15, 0x1b, 0x22, 0x30, 0x38,
    ]
    .into_iter()
    .map(of_type)
    .collect();
    forms.extend([
        filled(0x11, 4..0x14, 0x5a),
        filled(0x1c, 4..8, 0x5a),
        filled(0x31, 4..0xd, 0x5a),
        filled(0xa, 4..5, 1),
        filled(0xf, 4..0x1c, 0x5a),
        filled(0x13, 8..0x14, 0x5a),
        filled(0x16, 4..0xc, 0x5a),
        filled(0x17, 4..0xc, 0x5a),
        filled(0x18, 8..0x10, 0x5a),
        filled(0x19, 4..5, 1),
        filled(0x24, 4..8, 0x5a),
        filled(0x25, 4..0xa, 2),
        filled(0x33, 4..0xc, 0x5a),
        filled(0x3b, 4..0x10, 0x5a),
    ]);
    forms
}

#[test]
fn private_device_control_is_consumed_and_counted_by_the_firmware_table() {
    let forms = private_forms();
    let (mut state, mut memory) = with_device_control(&forms);
    let before = memory.bytes.clone();
    ring(&mut state, &mut memory).unwrap();
    let entries = forms.len() as u32;
    assert_eq!(memory.read_u32(CONTROL_STATE + READ), Ok(entries));
    // 0x13, 0x15, 0x1b and 0x22 are the uncounted ones among them.
    assert_eq!(memory.read_u32(count()), Ok(entries - 4));
    assert_eq!(
        state.device_control.values().sum::<u64>(),
        u64::from(entries)
    );
    // Nothing else the AP shares changed.
    let mut after = memory.bytes.clone();
    for at in [CONTROL_STATE + READ, count()] {
        let at = (at - BASE) as usize;
        after[at..at + 4].copy_from_slice(&before[at..at + 4]);
    }
    assert!(after == before);
}

#[test]
fn private_device_control_refuses_bytes_outside_its_form() {
    for form in private_forms() {
        let mut extra = form;
        extra[0x3f] = 1;
        assert!(device_control_entry(&extra).is_err());
    }
    // Flags are 0 or 1, and the QoS knob's data master is 0..2.
    assert!(device_control_entry(&filled(0xa, 4..5, 2)).is_err());
    assert!(device_control_entry(&filled(0x18, 4..5, 2)).is_err());
    assert!(device_control_entry(&filled(0x25, 4..5, 3)).is_err());
}

/// A HWDSID table named by the ring directory, and an entry of `kind`
/// for DSID 3 whose completion word is at `word` with value 0x700.
fn dsid_entry(memory: &mut Flat, kind: u32, word: u64) -> ([u8; 0x40], u64) {
    let table = BASE + 0x1e000;
    memory.put_u64(RINGS + DSID_TABLE, table);
    let mut entry = of_type(kind);
    entry[4..0xc].copy_from_slice(&word.to_le_bytes());
    entry[0xc..0x10].copy_from_slice(&0x700u32.to_le_bytes());
    entry[0x14] = 3;
    (entry, table + 3 * layout().dsid_slot_bytes)
}

#[test]
fn dsid_priority_sets_the_table_and_releases_a_matched_slot() {
    let (mut state, mut memory) = with_device_control(&[]);
    let word = BASE + 0x1f000;
    let (mut entry, slot) = dsid_entry(&mut memory, 0xe, word);
    entry[0x10..0x14].copy_from_slice(&5u32.to_le_bytes());
    // The firmware's word does not match yet: the request stays.
    memory.put(CONTROL_ENTRIES, &entry);
    memory.put_u32(slot + 4, 4);
    memory.put_u32(CONTROL_STATE + WRITE, 1);
    ring(&mut state, &mut memory).unwrap();
    assert_eq!(memory.read_u32(slot), Ok(5));
    assert_eq!(memory.read_u32(slot + 4), Ok(4));
    assert_eq!(memory.read_u32(word), Ok(0x700));
    // A matching word releases the slot.
    memory.put(CONTROL_ENTRIES + 0x40, &entry);
    memory.put_u32(slot + 4, 5);
    memory.put_u32(word, 0);
    memory.put_u32(CONTROL_STATE + WRITE, 2);
    ring(&mut state, &mut memory).unwrap();
    assert_eq!(memory.read_u64(slot), Ok(0));
    assert_eq!(memory.read_u32(word), Ok(0x700));
    // 0xe is uncounted.
    assert_eq!(memory.read_u32(count()), Ok(0));
    assert_eq!(state.device_control.get(&0xe), Some(&2));
    entry[0x15] = 1;
    assert!(device_control_entry(&entry).is_err());
    entry[0x15] = 0;
    entry[4..0xc].fill(0);
    assert!(device_control_entry(&entry).is_err());
}

#[test]
fn dsid_deprioritize_clears_the_slot_and_writes_the_completion_word() {
    let (mut state, mut memory) = with_device_control(&[]);
    let word = BASE + 0x1f000;
    let (mut entry, slot) = dsid_entry(&mut memory, 0x14, word);
    memory.put(CONTROL_ENTRIES, &entry);
    memory.put_u64(slot, 0x0000_0006_0000_0006);
    memory.put_u32(CONTROL_STATE + WRITE, 1);
    ring(&mut state, &mut memory).unwrap();
    assert_eq!(memory.read_u64(slot), Ok(0));
    assert_eq!(memory.read_u32(word), Ok(0x700));
    assert_eq!(memory.read_u32(count()), Ok(0));
    // The kext leaves +0x10..+0x14 zero.
    entry[0x10] = 1;
    assert!(device_control_entry(&entry).is_err());
    // Without a table the entry refuses and stays.
    let (mut bare, mut memory) = with_device_control(&[]);
    entry[0x10] = 0;
    memory.put(CONTROL_ENTRIES, &entry);
    memory.put_u32(CONTROL_STATE + WRITE, 1);
    assert_eq!(
        ring(&mut bare, &mut memory),
        Err(refuse!("GFX ring directory names no HWDSID table"))
    );
    assert_eq!(memory.read_u32(CONTROL_STATE + READ), Ok(0));
}

#[test]
fn the_context_switch_histogram_family_follows_the_firmware() {
    // The boot run's entry: get (1) for data master 0.
    let get = filled(0x1a, 8..9, 1);
    let mut enable = filled(0x1a, 8..9, 3);
    enable[4] = 2;
    let mut disable = filled(0x1a, 8..9, 4);
    disable[4] = 1;
    let reset = filled(0x1a, 8..9, 2);
    let (mut state, mut memory) = with_device_control(&[get, enable, disable, reset]);
    let enables = RINGS + layout().histogram_enables;
    memory.put(enables, &[0b011]);
    // A bin the AP reads back after the get.
    memory.put_u32(enables + 1, 9);
    ring(&mut state, &mut memory).unwrap();
    let mut bits = [0];
    memory.read(enables, &mut bits).unwrap();
    assert_eq!(bits, [0b101]);
    assert_eq!(memory.read_u32(enables + 1), Ok(9));
    assert_eq!(state.device_control.get(&0x1a), Some(&4));
    // 0x1a is uncounted.
    assert_eq!(memory.read_u32(count()), Ok(0));
    // No fourth data master, no operation 0, nothing beyond +8.
    let mut master = get;
    master[4] = 3;
    assert!(device_control_entry(&master).is_err());
    assert!(device_control_entry(&filled(0x1a, 8..9, 0)).is_err());
    assert!(device_control_entry(&filled(0x1a, 8..0xa, 1)).is_err());
}

#[test]
fn a_late_eval_timeout_marks_its_wait() {
    let wait = BASE + 0x1f000;
    let mut entry = filled(0x27, 0xc..0x20, 0x5a);
    entry[4..0xc].copy_from_slice(&wait.to_le_bytes());
    let (mut state, mut memory) = with_device_control(&[entry]);
    ring(&mut state, &mut memory).unwrap();
    assert_eq!(memory.read_u32(wait + LATE_EVAL_TIMED_OUT), Ok(1));
    assert_eq!(memory.read_u32(count()), Ok(1));
    entry[0x20] = 1;
    assert!(device_control_entry(&entry).is_err());
    assert!(device_control_entry(&filled(0x27, 0xc..0x20, 1)).is_err());
}

#[test]
fn the_power_counter_gather_empties_its_buffer() {
    let (mut state, mut memory) = with_device_control(&[of_type(0x32)]);
    let samples = OBJECT_928 + layout().power_counters;
    let sampled = OBJECT_928 + layout().power_counter_samples;
    memory.put(samples, &[0x5a; POWER_COUNTER_BYTES]);
    memory.put_u32(sampled, 3);
    // The selection list beside it is the kext's and stays.
    memory.put_u32(OBJECT_928 + 0xca34, 0x11);
    ring(&mut state, &mut memory).unwrap();
    let mut buffer = [0xff; POWER_COUNTER_BYTES];
    memory.read(samples, &mut buffer).unwrap();
    assert!(buffer.iter().all(|&b| b == 0));
    assert_eq!(memory.read_u32(sampled), Ok(0));
    assert_eq!(memory.read_u32(OBJECT_928 + 0xca34), Ok(0x11));
    assert_eq!(memory.read_u32(count()), Ok(1));
    assert!(device_control_entry(&filled(0x32, 4..5, 1)).is_err());
}

#[test]
fn riart_settings_land_in_the_root_object() {
    let mut mask = of_type(0x34);
    mask[4..8].copy_from_slice(&0x6u32.to_le_bytes());
    let disabled = filled(0x35, 4..5, 1);
    let (mut state, mut memory) = with_device_control(&[mask, disabled]);
    ring(&mut state, &mut memory).unwrap();
    let mut riart = |offset| memory.read_u32(OBJECT_358 + offset);
    assert_eq!(riart(layout().riart_u_levels_disabled), Ok(6));
    assert_eq!(riart(layout().riart_t_disabled), Ok(1));
    assert_eq!(memory.read_u32(count()), Ok(2));
    assert!(device_control_entry(&filled(0x35, 4..5, 2)).is_err());
    assert!(device_control_entry(&filled(0x34, 8..9, 1)).is_err());
}

#[test]
fn device_control_needing_a_missing_unit_refuses_by_name() {
    for (kind, reason) in [
        (0x4, "performance-counter"),
        (0x5, "performance-counter"),
        (0x9, "scheduler thread"),
        (0x1f, "context registry"),
        (0x29, "teardown thread"),
        (0x2c, "fatal"),
        (0x26, "host-allocation"),
    ] {
        let (mut state, mut memory) = with_device_control(&[of_type(kind)]);
        assert!(
            ring(&mut state, &mut memory)
                .unwrap_err()
                .to_string()
                .contains(reason)
        );
        assert_eq!(memory.read_u32(CONTROL_STATE + READ), Ok(0));
        assert_eq!(memory.read_u32(count()), Ok(0));
    }
    // Types the kext never sends.
    for kind in [0, 1, 2, 3, 6, 0x1d, 0x20, 0x3c, 0x40] {
        assert_eq!(
            device_control_entry(&of_type(kind)),
            Err(refuse!(
                "GFX device-control type is not one the 26A434 kext sends"
            ))
        );
    }
}

#[test]
fn unmodeled_device_control_stays_unconsumed_and_refuses_by_name() {
    let mut event = [0; 0x40];
    event[0] = 8;
    let (mut state, mut memory) = with_device_control(&[power(0), event]);
    state
        .doorbell(doorbell_payload(DOORBELL_FWCTL, 0x11))
        .unwrap();
    assert!(
        state
            .dma(&mut memory, &mut Vec::new())
            .unwrap_err()
            .to_string()
            .contains("host-allocation")
    );
    // The power notice before it was retired; the memory event was not.
    assert_eq!(memory.read_u32(CONTROL_STATE + READ), Ok(1));
    let mut extra = power(0);
    extra[0x20] = 1;
    assert!(device_control_entry(&extra).is_err());
    assert!(device_control_entry(&power(2)).is_err());
}

/// The 3D ring at submission index 0 with one entry naming the channel
/// at `channel_state`, whose queue holds one command at `command`.
fn one_3d_entry(memory: &mut Flat, channel_state: u64, command: u64) -> (u64, u64) {
    let (ring_state, entries) = (BASE + 0x9000, BASE + 0xc000);
    memory.put_descriptor(RINGS + 0x20 + 0x20, ring_state, entries);
    memory.put_u32(entries + 0x10, 1);
    memory.put(entries + 0x14, &1u16.to_le_bytes());
    memory.put_u32(ring_state + WRITE, 1);
    if channel_state != 0 {
        crate::agx::channel::tests::one_command(memory, channel_state, command);
        memory.put_u64(entries + 0x08, channel_state);
    }
    (ring_state, entries)
}

#[test]
fn unexecuted_work_refuses_rather_than_completing() {
    let (mut state, mut memory) = with_device_control(&[]);
    let (ring_state, _) = one_3d_entry(&mut memory, 0, 0);
    state.doorbell(work(1, 0)).unwrap();
    // The entry names no channel yet.
    assert_eq!(
        state.dma(&mut memory, &mut Vec::new()),
        Err(refuse!("GFX data-master entry names no channel state"))
    );
    // A channel whose next command is a 3D command.
    let (channel_state, command) = (BASE + 0xd000, BASE + 0x10000);
    one_3d_entry(&mut memory, channel_state, command);
    memory.put_u32(command, 1);
    memory.put_u64(command + 0x14, command + 0x1000);
    memory.put_u32(command + 0x1c, 0x200);
    memory.put_u32(command + 0x1000, 7);
    assert_eq!(
        state.dma(&mut memory, &mut Vec::new()),
        Err(refuse!("GFX 3D command (type 1) is not executed"))
    );
    assert_eq!(memory.read_u32(ring_state + READ), Ok(0));
    assert_eq!(memory.read_u32(channel_state + 0x1000), Ok(0));
    // An empty ring has nothing to refuse.
    let mut idle = with_device_control(&[]);
    memory.put_descriptor(RINGS + 0x20, ring_state + 0x100, BASE + 0xc000);
    idle.1 = memory;
    idle.0.doorbell(work(0, 0)).unwrap();
    idle.0.dma(&mut idle.1, &mut Vec::new()).unwrap();
}

#[test]
fn the_firmware_publishes_its_magic_in_the_uat_handoff_as_it_starts() {
    let handoff = BASE + 0x1_8000;
    let mut memory = Flat::new(BASE, 0x20000);
    kext_root(&mut memory);
    let mut state = State::new(&G14G);
    state.handoff = Some(handoff);
    assert!(!state.dma_pending());
    state
        .init((MSG_INIT << TYPE_SHIFT) | (ROOT & INIT_VA_BITS))
        .unwrap();
    assert!(state.dma_pending());
    state.dma(&mut memory, &mut Vec::new()).unwrap();
    assert_eq!(memory.read_u64(handoff + 8), Ok(0x4b1d_0000_0000_0002));
    assert!(!state.dma_pending());
}

#[test]
fn the_stop_drops_held_work_and_follows_the_kexts_recovery() {
    let (mut state, mut memory) = with_device_control(&[]);
    let (channel_state, command) = (BASE + 0xd000, BASE + 0x10000);
    one_3d_entry(&mut memory, channel_state, command);
    memory.put_u32(command, 1);
    memory.put_u64(command + 0x14, command + 0x1000);
    memory.put_u32(command + 0x1c, 0x200);
    memory.put_u32(command + 0x1000, 7);
    state.doorbell(work(1, 0)).unwrap();
    assert!(state.dma(&mut memory, &mut Vec::new()).is_err());

    let object = state.root.as_ref().unwrap().object_928;
    let words = G14G.model().unwrap().layout.recovery.unwrap();
    let (generation, stop_state, stopped) = (
        object + words.generation,
        object + words.stop_state,
        object + words.stopped,
    );
    state.doorbell(doorbell_payload(DOORBELL_STOP, 41)).unwrap();
    state.dma(&mut memory, &mut Vec::new()).unwrap();
    assert_eq!(memory.read_u64(generation), Ok(42));
    assert_eq!(memory.read_u32(stop_state), Ok(1));
    assert_eq!(memory.read_u32(stopped), Ok(1));

    // Stopped until the kext says otherwise, and idle again after.
    state.dma(&mut memory, &mut Vec::new()).unwrap();
    assert!(state.dma_pending());
    memory.put_u32(stop_state, 2);
    state.dma(&mut memory, &mut Vec::new()).unwrap();
    assert_eq!(memory.read_u32(stop_state), Ok(3));
    assert_eq!(memory.read_u32(stopped), Ok(0));
    memory.put_u32(stop_state, 0);
    state.dma(&mut memory, &mut Vec::new()).unwrap();
    assert!(!state.dma_pending());
}

#[test]
fn a_family_without_recovered_recovery_words_refuses_the_stop() {
    let (mut state, mut memory) = with_device_control_on(&G14X, &[]).unwrap();
    state.doorbell(doorbell_payload(DOORBELL_STOP, 0)).unwrap();
    assert_eq!(
        state.dma(&mut memory, &mut Vec::new()),
        Err(refuse!("the GPU family's recovery words are not recovered"))
    );
}

#[test]
fn each_family_reads_its_own_root_and_g13_refuses_dashboard_signals() {
    use stela_soc_data::agx::generation::{G14X, G16X};
    // A G14X firmware refuses the G13G/G13X/G14G root.
    let mut memory = Flat::new(BASE, 0x20000);
    kext_root(&mut memory);
    let mut g14x = State::new(&G14X);
    g14x.init((MSG_INIT << TYPE_SHIFT) | (ROOT & INIT_VA_BITS))
        .unwrap();
    assert_eq!(
        g14x.dma(&mut memory, &mut Vec::new()).unwrap_err().what(),
        "GFX initdata root lacks its magic"
    );
    assert_eq!(
        with_device_control_on(&G16X, &[]).err(),
        Some(refuse!(
            "the GPU family's initdata root magic is not recovered"
        ))
    );
    let (mut state, mut memory) = with_device_control_on(&G13G, &[]).unwrap();
    assert!(state.root.is_some());
    let (channel_state, command) = (BASE + 0xd000, BASE + 0x10000);
    let (ring_state, _) = one_3d_entry(&mut memory, channel_state, command);
    memory.put_u32(command, 0xc);
    state.doorbell(work(1, 0)).unwrap();
    assert_eq!(
        state.dma(&mut memory, &mut Vec::new()),
        Err(refuse!(
            "GFX dashboard signal command: this GPU family's firmware has no dashboard-signal handler"
        ))
    );
    assert_eq!(memory.read_u32(ring_state + READ), Ok(0));
    assert_eq!(memory.read_u32(channel_state + 0x1000), Ok(0));
}

#[test]
fn g14x_device_control_writes_its_own_offsets() {
    use stela_soc_data::agx::generation::G14X;
    let mut mask = of_type(0x34);
    mask[4..8].copy_from_slice(&0x6u32.to_le_bytes());
    let (mut state, mut memory) = with_device_control_on(&G14X, &[]).unwrap();
    let word = BASE + 0x1f000;
    let (deprioritize, _) = dsid_entry(&mut memory, 0x14, word);
    let slot = BASE + 0x1e000 + 3 * 8;
    memory.put_u64(slot, 0x0000_0006_0000_0006);
    memory.put_u64(slot + 8, 0x0000_0007_0000_0007);
    for (n, entry) in [of_type(0x32), mask, deprioritize].iter().enumerate() {
        memory.put(CONTROL_ENTRIES + n as u64 * 0x40, entry);
    }
    memory.put_u32(CONTROL_STATE + WRITE, 3);
    let g14x = &G14X.model().unwrap().layout;
    memory.put_u32(OBJECT_928 + g14x.power_counter_samples, 3);
    ring(&mut state, &mut memory).unwrap();
    assert_eq!(memory.read_u32(OBJECT_928 + 0xcfc0), Ok(0));
    assert_eq!(memory.read_u32(OBJECT_358 + 0x123c), Ok(6));
    // DSID slots are 8 bytes: DSID 3's neighbour stays.
    assert_eq!(memory.read_u64(slot), Ok(0));
    assert_eq!(memory.read_u64(slot + 8), Ok(0x0000_0007_0000_0007));
    assert_eq!(memory.read_u32(word), Ok(0x700));
    // 0x32 and 0x34 are counted at the G14X offset; G14G's stays empty.
    assert_eq!(memory.read_u32(OBJECT_928 + 0xedc8), Ok(2));
    assert_eq!(memory.read_u32(count()), Ok(0));
}

#[test]
fn g15g_device_control_writes_its_own_offsets() {
    use stela_soc_data::agx::generation::G15G;
    let mut mask = of_type(0x34);
    mask[4..8].copy_from_slice(&0x6u32.to_le_bytes());
    let mut disabled = of_type(0x35);
    disabled[4..8].copy_from_slice(&1u32.to_le_bytes());
    let mut enable = filled(0x1a, 8..9, 3);
    enable[4] = 2;
    let (mut state, mut memory) = with_device_control_on(&G15G, &[]).unwrap();
    for (n, entry) in [of_type(0x32), mask, disabled, enable].iter().enumerate() {
        memory.put(CONTROL_ENTRIES + n as u64 * 0x40, entry);
    }
    memory.put_u32(CONTROL_STATE + WRITE, 4);
    memory.put_u32(OBJECT_928 + 0xcfc0, 3);
    ring(&mut state, &mut memory).unwrap();
    assert_eq!(memory.read_u32(OBJECT_928 + 0xcfc0), Ok(0));
    // The RIART words are packed at +0xe35 and +0xe39.
    assert_eq!(memory.read_u32(OBJECT_358 + 0xe35), Ok(6));
    assert_eq!(memory.read_u32(OBJECT_358 + 0xe39), Ok(1));
    assert_eq!(memory.read_u32(OBJECT_358 + 0x123c), Ok(0));
    // The histogram enables sit four bytes above G14X's.
    let mut bits = [0];
    memory.read(RINGS + 0x3d8, &mut bits).unwrap();
    assert_eq!(bits, [0b100]);
    memory.read(RINGS + 0x3d4, &mut bits).unwrap();
    assert_eq!(bits, [0]);
    // 0x32, 0x34 and 0x35 are counted at G14X's offset; 0x1a is not.
    assert_eq!(memory.read_u32(OBJECT_928 + 0xedc8), Ok(3));
}

const OWNED_CHANNEL: u64 = BASE + 0xd000;
const OWNED_COMMAND: u64 = BASE + 0x10000;
const OWNED_STREAM: u64 = OWNED_COMMAND + 0x400;
const OWNED_SHARED: u64 = OWNED_COMMAND + 0x800;
const OWNED_DRIVER: u64 = OWNED_COMMAND + 0x900;
const OWNED_FIRMWARE: u64 = OWNED_COMMAND + 0x908;

fn owned_signal(family: &'static Family) -> (State, Flat, u64) {
    let (mut state, mut memory) = with_device_control_on(family, &[]).unwrap();
    event::tests::event_ring(&mut memory);
    let (master, _) = one_3d_entry(&mut memory, OWNED_CHANNEL, OWNED_COMMAND);
    memory.put_u32(OWNED_COMMAND, 9);
    memory.put_u64(OWNED_COMMAND + 0x1c, OWNED_STREAM);
    memory.put_u64(OWNED_COMMAND + 0x30, OWNED_DRIVER);
    memory.put_u64(OWNED_COMMAND + 0x38, OWNED_FIRMWARE);
    memory.put_u32(OWNED_COMMAND + 0x40, 0x100);
    memory.put_u32(OWNED_COMMAND + 0x44, 2);
    memory.put_u32(OWNED_STREAM, 0xe);
    memory.put_u64(OWNED_STREAM + 0x10, 0x55);
    memory.put_u64(OWNED_STREAM + 0x18, 1);
    memory.put_u64(OWNED_STREAM + 0x28, OWNED_SHARED);
    memory.put_u64(OWNED_STREAM + 0x30, OWNED_FIRMWARE);
    memory.put_u32(OWNED_STREAM + 0x38, 0x100);
    memory.put_u32(OWNED_STREAM + 0x4c, 0x4000_0002);
    state.doorbell(work(1, 0)).unwrap();
    (state, memory, master)
}

struct WorkMemory {
    flat: Flat,
    fail_write: Option<usize>,
    fail_read: Option<u64>,
    writes: Vec<(u64, Vec<u8>)>,
}

impl Shared for WorkMemory {
    fn read(&mut self, va: u64, out: &mut [u8]) -> Result<()> {
        if self.fail_read == Some(va) {
            self.fail_read = None;
            return Err(refuse!("injected work input read fault"));
        }
        self.flat.read(va, out)
    }
    fn write(&mut self, va: u64, data: &[u8]) -> Result<()> {
        if self.fail_write == Some(self.writes.len()) {
            self.fail_write = None;
            return Err(refuse!("injected work publication write fault"));
        }
        self.flat.write(va, data)?;
        self.writes.push((va, data.to_vec()));
        Ok(())
    }
    fn gpu_read(
        &mut self,
        _: usize,
        _: u64,
        _: &mut [u8],
    ) -> Result<(), stela_contracts::dma::DmaError> {
        panic!("a signal command cannot execute GPU reads");
    }
    fn gpu_write(
        &mut self,
        _: usize,
        _: u64,
        _: &[u8],
    ) -> Result<(), stela_contracts::dma::DmaError> {
        panic!("a signal command cannot execute GPU writes");
    }
}

fn owned_control(family: &'static Family, kind: u32) -> (State, Flat, u64) {
    let (mut state, mut memory) = with_device_control_on(family, &[]).unwrap();
    let layout = &family.model().unwrap().layout;
    let count = OBJECT_928 + layout.device_control_count;
    memory.put_u32(count, 7);
    let mut entry = of_type(kind);
    match kind {
        0xe | 0x14 => {
            memory.put_u64(RINGS + DSID_TABLE, BASE + 0x1e000);
            let slot = BASE + 0x1e000 + 3 * layout.dsid_slot_bytes;
            memory.put_u32(slot + 4, 5);
            entry[4..0xc].copy_from_slice(&(BASE + 0x1f000).to_le_bytes());
            entry[0xc..0x10].copy_from_slice(&0x700u32.to_le_bytes());
            entry[0x14] = 3;
            if kind == 0xe {
                entry[0x10..0x14].copy_from_slice(&5u32.to_le_bytes());
            }
        }
        0x1a => {
            entry[4..8].copy_from_slice(&1u32.to_le_bytes());
            entry[8] = 3;
        }
        0x27 => {
            entry[4..0xc].copy_from_slice(&(BASE + 0x1f000).to_le_bytes());
        }
        0x34 => {
            entry[4..8].copy_from_slice(&6u32.to_le_bytes());
        }
        0x35 => {
            entry[4..8].copy_from_slice(&1u32.to_le_bytes());
        }
        _ => {}
    }
    memory.put(CONTROL_ENTRIES, &entry);
    memory.put_u32(CONTROL_STATE + WRITE, 1);
    state
        .doorbell(doorbell_payload(DOORBELL_FWCTL, 0x11))
        .unwrap();
    (state, memory, count)
}

#[test]
fn accepted_device_controls_keep_all_effects_count_and_retirement_on_every_family() {
    for family in [&G13G, &G13X, &G14G, &G14X, &G15G] {
        for kind in [
            7,
            DEVICE_CONTROL_POWER,
            0xe,
            0x14,
            0x1a,
            0x27,
            0x32,
            0x34,
            0x35,
        ] {
            let (mut baseline, flat, _) = owned_control(family, kind);
            let mut complete = WorkMemory {
                flat,
                fail_write: None,
                fail_read: None,
                writes: Vec::new(),
            };
            baseline.dma(&mut complete, &mut Vec::new()).unwrap();
            for failed_write in 0..complete.writes.len() {
                let (mut state, flat, count) = owned_control(family, kind);
                let mut memory = WorkMemory {
                    flat,
                    fail_write: Some(failed_write),
                    fail_read: None,
                    writes: Vec::new(),
                };
                assert_eq!(
                    state.dma(&mut memory, &mut Vec::new()),
                    Err(refuse!("injected work publication write fault"))
                );
                assert!(state.dma_pending());
                assert_eq!(memory.writes.len(), failed_write);
                // Every form, target, DSID comparison and derived bitmask
                // was accepted before the first command write. A later
                // error cannot redispatch a changed AP entry.
                memory.flat.put_u32(CONTROL_ENTRIES, 0xffff);
                state.dma(&mut memory, &mut Vec::new()).unwrap();
                assert_eq!(memory.writes, complete.writes);
                assert_eq!(state.device_control.get(&kind), Some(&1));
                assert_eq!(state.power.len(), usize::from(kind == DEVICE_CONTROL_POWER));
                let expected = if matches!(kind, 0xe | 0x14 | 0x1a) {
                    7
                } else {
                    8
                };
                assert_eq!(memory.flat.read_u32(count), Ok(expected));
                assert_eq!(memory.flat.read_u32(CONTROL_STATE), Ok(1));
                assert!(!state.dma_pending());
                state.dma(&mut memory, &mut Vec::new()).unwrap();
                assert_eq!(memory.writes, complete.writes);
                assert_eq!(state.device_control.get(&kind), Some(&1));
            }
        }
    }
}

#[test]
fn dsid_priority_read_fault_preserves_the_source_ordered_store_and_accepted_identity() {
    for family in [&G13G, &G13X, &G14G, &G14X, &G15G] {
        let (mut state, flat, _) = owned_control(family, 0xe);
        let slot = BASE + 0x1e000 + 3 * family.model().unwrap().layout.dsid_slot_bytes;
        let mut memory = WorkMemory {
            flat,
            fail_write: None,
            fail_read: Some(slot + 4),
            writes: Vec::new(),
        };
        assert_eq!(
            state.dma(&mut memory, &mut Vec::new()),
            Err(refuse!("injected work input read fault"))
        );
        assert_eq!(memory.writes, [(slot, 5u32.to_le_bytes().to_vec())]);
        assert_eq!(memory.flat.read_u32(slot), Ok(5));
        assert!(state.active_control.is_some() && state.dma_pending());
        memory.flat.put_u32(CONTROL_ENTRIES + 0xc, 0x800);
        memory.flat.put_u32(CONTROL_ENTRIES + 0x10, 99);
        memory.flat.put_u32(slot, 42); // must not replay the priority store
        state.dma(&mut memory, &mut Vec::new()).unwrap();
        assert_eq!(memory.flat.read_u64(slot), Ok(0));
        assert_eq!(memory.flat.read_u32(BASE + 0x1f000), Ok(0x700));
        assert_eq!(
            memory
                .writes
                .iter()
                .filter(|(at, bytes)| *at == slot && bytes.len() == 4)
                .count(),
            1
        );
        assert_eq!(state.device_control.get(&0xe), Some(&1));
    }
}

#[test]
fn dsid_control_refuses_wrapped_table_pointers_and_complete_slot_extents_without_effects() {
    for family in [&G13G, &G13X, &G14G, &G14X, &G15G] {
        let layout = &family.model().unwrap().layout;
        for kind in [0xe, 0x14] {
            let (state, flat, _) = owned_control(family, kind);
            let mut memory = WorkMemory {
                flat,
                fail_write: None,
                fail_read: None,
                writes: Vec::new(),
            };
            let descriptor = ring::Descriptor {
                state: CONTROL_STATE,
                entries: CONTROL_ENTRIES,
            };
            let mut entry = [0; 0x40];
            memory.read(CONTROL_ENTRIES, &mut entry).unwrap();
            let original = state.root.as_ref().unwrap();
            for rings in [u64::MAX - DSID_TABLE + 1, u64::MAX - DSID_TABLE - 3] {
                let mut root = original.clone();
                root.rings = rings;
                assert_eq!(
                    ActiveControl::accept(&mut memory, &root, layout, descriptor, 0, &entry)
                        .unwrap_err(),
                    refuse!("GFX HWDSID table pointer wraps")
                );
                assert!(memory.writes.is_empty());
            }
            let offset = 3 * layout.dsid_slot_bytes;
            for table in [
                u64::MAX - offset + 1,
                u64::MAX - offset - layout.dsid_slot_bytes + 2,
            ] {
                memory.flat.put_u64(RINGS + DSID_TABLE, table);
                assert!(
                    ActiveControl::accept(&mut memory, original, layout, descriptor, 0, &entry)
                        .unwrap_err()
                        .to_string()
                        .contains("table slot")
                );
                assert!(memory.writes.is_empty());
            }
        }
    }
}

#[test]
fn dsid_priority_captured_match_survives_faults_with_overlapping_completion_targets() {
    for matched in [4u32, 5] {
        let (mut state, flat, _) = owned_control(&G14G, 0xe);
        let slot = BASE + 0x1e000 + 3 * layout().dsid_slot_bytes;
        let mut memory = WorkMemory {
            flat,
            fail_write: Some(1),
            fail_read: None,
            writes: Vec::new(),
        };
        memory.flat.put_u32(slot + 4, matched);
        // Completion overwrites the matching word. Its value must not
        // be observed as a new match while retrying a later phase.
        memory.flat.put_u64(CONTROL_ENTRIES + 4, slot + 4);
        memory.flat.put_u32(CONTROL_ENTRIES + 0xc, 0x900);
        assert_eq!(
            state.dma(&mut memory, &mut Vec::new()),
            Err(refuse!("injected work publication write fault"))
        );
        memory
            .flat
            .put_u32(slot + 4, if matched == 4 { 5 } else { 4 });
        state.dma(&mut memory, &mut Vec::new()).unwrap();
        assert_eq!(
            memory.flat.read_u32(slot),
            Ok(if matched == 5 { 0 } else { 5 })
        );
        assert_eq!(memory.flat.read_u32(slot + 4), Ok(0x900));
        let expected = if matched == 5 {
            alloc::vec![slot, slot, slot + 4, CONTROL_STATE]
        } else {
            alloc::vec![slot, slot + 4, CONTROL_STATE]
        };
        assert_eq!(
            memory.writes.iter().map(|(at, _)| *at).collect::<Vec<_>>(),
            expected
        );
    }
}

#[test]
fn a_device_control_pre_effect_input_fault_accepts_no_entry() {
    let (mut state, flat, _) = owned_control(&G14G, 0x1a);
    let enables = RINGS + layout().histogram_enables;
    let mut memory = WorkMemory {
        flat,
        fail_write: None,
        fail_read: Some(enables),
        writes: Vec::new(),
    };
    assert_eq!(
        state.dma(&mut memory, &mut Vec::new()),
        Err(refuse!("injected work input read fault"))
    );
    assert!(memory.writes.is_empty() && state.active_control.is_none());
    assert!(state.device_control.is_empty());
    state.dma(&mut memory, &mut Vec::new()).unwrap();
    assert_eq!(state.device_control.get(&0x1a), Some(&1));
}

#[test]
fn device_control_counter_read_and_write_have_separate_kept_phases() {
    let (mut state, flat, count) = owned_control(&G14G, 0x27);
    let mut memory = WorkMemory {
        flat,
        fail_write: None,
        fail_read: Some(count),
        writes: Vec::new(),
    };
    assert_eq!(
        state.dma(&mut memory, &mut Vec::new()),
        Err(refuse!("injected work input read fault"))
    );
    assert_eq!(memory.writes.len(), 1); // timeout effect precedes common-tail count read
    assert_eq!(memory.flat.read_u32(BASE + 0x1f038), Ok(1));
    memory.flat.put_u32(count, u32::MAX);
    memory.fail_write = Some(1); // fail the counter write after capturing its wraparound result
    assert_eq!(
        state.dma(&mut memory, &mut Vec::new()),
        Err(refuse!("injected work publication write fault"))
    );
    memory.flat.put_u32(count, 777);
    memory.fail_write = Some(2); // commit captured count, then fail final retirement
    assert_eq!(
        state.dma(&mut memory, &mut Vec::new()),
        Err(refuse!("injected work publication write fault"))
    );
    assert_eq!(memory.flat.read_u32(count), Ok(0));
    assert_eq!(state.device_control.get(&0x27), Some(&1));
    memory.flat.put_u32(count, 888);
    let protocol = Protocol(Rc::new(RefCell::new(state)));
    assert!(Message(protocol.clone()).check_reset().is_err());
    protocol
        .0
        .borrow_mut()
        .dma(&mut memory, &mut Vec::new())
        .unwrap();
    assert_eq!(memory.flat.read_u32(count), Ok(888)); // no repeated count after retirement fault
    assert_eq!(
        memory.writes.iter().map(|(at, _)| *at).collect::<Vec<_>>(),
        [BASE + 0x1f038, count, CONTROL_STATE]
    );
    assert_eq!(
        protocol.with(|state| state.device_control.get(&0x27).copied()),
        Some(1)
    );
    Message(protocol).check_reset().unwrap();
}

#[test]
fn device_control_count_reads_after_the_original_effect_even_when_targets_overlap() {
    let (mut state, mut memory, count) = owned_control(&G14G, 0x27);
    memory.put_u64(CONTROL_ENTRIES + 4, count - LATE_EVAL_TIMED_OUT);
    state.dma(&mut memory, &mut Vec::new()).unwrap();
    assert_eq!(memory.read_u32(count), Ok(2)); // timeout writes1, common tail reads1 then writes2
}

#[test]
fn accepted_signal_actions_survive_every_publication_fault_without_replay() {
    use event::tests::{EVENT_ENTRIES, EVENT_STATE, entry};
    for family in [&G13G, &G13X, &G14G, &G14X, &G15G] {
        for failed_write in 0..11 {
            let (mut state, flat, master) = owned_signal(family);
            let mut memory = WorkMemory {
                flat,
                fail_write: Some(failed_write),
                fail_read: None,
                writes: Vec::new(),
            };
            let mut outbox = Vec::new();
            assert_eq!(
                state.dma(&mut memory, &mut outbox),
                Err(refuse!("injected work publication write fault"))
            );
            assert_eq!(memory.writes.len(), failed_write);
            assert!(state.dma_pending());
            // The accepted command owns these original values. Altering
            // source memory during recovery cannot redispatch it, lose
            // its event, or change the target completion identity.
            // The entry is already accepted even if its tail publication
            // faults. The signal itself is accepted only after both words.
            memory.flat.put(BASE + 0xc014, &2u16.to_le_bytes());
            memory.flat.put_u32(OWNED_CHANNEL + 0x1040, 2);
            if failed_write >= 2 {
                memory.flat.put_u64(OWNED_STREAM + 0x10, 0xdead);
                memory.flat.put_u64(OWNED_STREAM + 0x18, 77);
                memory
                    .flat
                    .put_u64(OWNED_STREAM + 0x30, OWNED_COMMAND + 0xa00);
                memory.flat.put_u32(OWNED_COMMAND + 0x40, 0xbeef);
                memory.flat.put_u32(OWNED_COMMAND + 0x50, 9);
            }
            if failed_write == 10 {
                // Final master retirement has already been accepted;
                // it must not reopen the channel before publishing it.
                memory.fail_read = Some(OWNED_CHANNEL + 0x1000);
            }
            state.dma(&mut memory, &mut outbox).unwrap();
            assert!(!state.dma_pending());
            assert_eq!((state.commands, state.events), (1, 2));
            assert_eq!(outbox.len(), 2);
            assert_eq!(
                memory.writes.iter().map(|(va, _)| *va).collect::<Vec<_>>(),
                [
                    OWNED_CHANNEL + 0x28,
                    OWNED_CHANNEL + 0x1030,
                    OWNED_SHARED,
                    EVENT_ENTRIES,
                    EVENT_STATE + WRITE,
                    OWNED_FIRMWARE,
                    OWNED_DRIVER,
                    EVENT_ENTRIES + event::ENTRY_BYTES as u64,
                    EVENT_STATE + WRITE,
                    OWNED_CHANNEL + 0x1000,
                    master,
                ]
            );
            assert_eq!(memory.flat.read_u64(OWNED_SHARED), Ok(1));
            assert_eq!(memory.flat.read_u32(OWNED_FIRMWARE), Ok(0x100));
            assert_eq!(memory.flat.read_u32(OWNED_DRIVER), Ok(0x100));
            assert_eq!(memory.flat.read_u32(OWNED_COMMAND + 0xa00), Ok(0));
            assert_eq!(
                entry(&mut memory.flat, 0),
                event::Event::SharedSignal {
                    id: 0x55,
                    value: 1,
                    flag: 0
                }
                .encode()
            );
            assert_eq!(
                entry(&mut memory.flat, 1),
                event::Event::Flag(1 << 2).encode()
            );
            assert_eq!(memory.flat.read_u32(EVENT_STATE + WRITE), Ok(2));
            assert_eq!(memory.flat.read_u32(OWNED_CHANNEL + 0x1000), Ok(1));
            assert_eq!(memory.flat.read_u32(master), Ok(1));
            if failed_write == 10 {
                assert_eq!(memory.fail_read, Some(OWNED_CHANNEL + 0x1000));
            }
            // An idle later poll cannot replay the completed command.
            state.dma(&mut memory, &mut outbox).unwrap();
            assert_eq!((memory.writes.len(), outbox.len()), (11, 2));
        }
    }
}

#[test]
fn submitted_tail_stops_before_later_commands_including_wrap_and_empty_intervals() {
    for family in [&G13G, &G13X, &G14G, &G14X, &G15G] {
        for wrapping in [false, true] {
            let (mut state, mut memory, master) = owned_signal(family);
            let queue = OWNED_CHANNEL + 0x1000;
            let pointers = OWNED_CHANNEL + 0x2000;
            let entries = BASE + 0xc000;
            let (read, target, write) = if wrapping {
                (127, 0u16, 1)
            } else {
                (0, 1u16, 2)
            };
            memory.put_u32(queue, read);
            memory.put_u32(queue + 0x40, write);
            memory.put(entries + 0x14, &target.to_le_bytes());
            memory.put_u64(pointers + u64::from(read) * 8, OWNED_COMMAND);
            // A second AP command is already published in the channel,
            // but belongs to the next data-master entry. It must not be
            // executed or block completion of the accepted interval.
            let later = OWNED_COMMAND + 0xb00;
            memory.put_u32(later, 1); // source-checked but unexecuted 3D
            memory.put_u64(later + 0x14, later + 0x1000);
            memory.put_u32(later + 0x1c, 0x200);
            memory.put_u32(later + 0x1000, 7);
            memory.put_u64(pointers + u64::from(target) * 8, later);
            let mut outbox = Vec::new();
            state.dma(&mut memory, &mut outbox).unwrap();
            assert_eq!((state.commands, state.events), (1, 2));
            assert_eq!(memory.read_u32(queue), Ok(u32::from(target)));
            assert_eq!(memory.read_u32(OWNED_CHANNEL + 0x28), Ok(u32::from(target)));
            assert_eq!(memory.read_u32(queue + 0x30), Ok(u32::from(target)));
            assert_eq!(memory.read_u32(master), Ok(1));

            // Repeated empty intervals cannot drain a live WRITE either.
            let second = entries + Kind::DataMaster.entry_bytes() as u64;
            memory.put_u64(second + 8, OWNED_CHANNEL);
            memory.put_u32(second + 0x10, 1);
            memory.put(second + 0x14, &target.to_le_bytes());
            memory.put_u32(master + WRITE, 2);
            state.doorbell(work(1, 0)).unwrap();
            state.dma(&mut memory, &mut outbox).unwrap();
            assert_eq!(memory.read_u32(master), Ok(2));
            assert_eq!((state.commands, outbox.len()), (1, 2));

            // Only the next entry carrying the new tail can acquire it.
            let third = second + Kind::DataMaster.entry_bytes() as u64;
            memory.put_u64(third + 8, OWNED_CHANNEL);
            memory.put_u32(third + 0x10, 1);
            memory.put(third + 0x14, &(write as u16).to_le_bytes());
            memory.put_u32(master + WRITE, 3);
            state.doorbell(work(1, 0)).unwrap();
            assert_eq!(
                state.dma(&mut memory, &mut outbox),
                Err(refuse!("GFX 3D command (type 1) is not executed"))
            );
            assert_eq!(memory.read_u32(master), Ok(2));
            assert_eq!(memory.read_u32(queue), Ok(u32::from(target)));
            assert_eq!((state.commands, outbox.len()), (1, 2));
        }
    }
}

#[test]
fn wrapped_submission_keeps_its_tail_after_final_retirement_fault_and_late_append() {
    let (mut state, flat, master) = owned_signal(&G14G);
    let mut memory = WorkMemory {
        flat,
        fail_write: Some(10),
        fail_read: None,
        writes: Vec::new(),
    };
    let queue = OWNED_CHANNEL + 0x1000;
    memory.flat.put_u32(queue, 127);
    memory.flat.put_u32(queue + 0x40, 0);
    memory
        .flat
        .put_u64(OWNED_CHANNEL + 0x2000 + 127 * 8, OWNED_COMMAND);
    memory.flat.put(BASE + 0xc014, &0u16.to_le_bytes());
    let mut outbox = Vec::new();
    assert_eq!(
        state.dma(&mut memory, &mut outbox),
        Err(refuse!("injected work publication write fault"))
    );
    assert_eq!(memory.flat.read_u32(queue), Ok(0));
    assert_eq!(memory.flat.read_u32(master), Ok(0));
    let later = OWNED_COMMAND + 0xb00;
    memory.flat.put_u32(later, 1);
    memory.flat.put_u64(OWNED_CHANNEL + 0x2000, later);
    memory.flat.put_u32(queue + 0x40, 1);
    memory.flat.put(BASE + 0xc014, &1u16.to_le_bytes());
    memory.fail_read = Some(queue);
    state.dma(&mut memory, &mut outbox).unwrap();
    assert_eq!(memory.fail_read, Some(queue));
    assert_eq!(memory.flat.read_u32(queue), Ok(0));
    assert_eq!(memory.flat.read_u32(master), Ok(1));
    assert_eq!(
        (state.commands, outbox.len(), memory.writes.len()),
        (1, 2, 11)
    );
}

#[test]
fn submitted_tail_refuses_unpublished_or_out_of_range_pointers_before_effects() {
    for target in [2u16, 128] {
        let (mut state, flat, master) = owned_signal(&G14G);
        let mut memory = WorkMemory {
            flat,
            fail_write: None,
            fail_read: None,
            writes: Vec::new(),
        };
        memory.flat.put(BASE + 0xc014, &target.to_le_bytes());
        assert!(
            state
                .dma(&mut memory, &mut Vec::new())
                .unwrap_err()
                .to_string()
                .contains("submitted command tail")
        );
        assert!(memory.writes.is_empty());
        assert_eq!(memory.flat.read_u32(master), Ok(0));
        assert_eq!((state.commands, state.events), (0, 0));
    }
}

#[test]
fn signal_scalar_crossing_a_uat_page_refuses_before_signal_effects() {
    let (mut state, flat, _) = owned_signal(&G14G);
    let mut memory = WorkMemory {
        flat,
        fail_write: None,
        fail_read: None,
        writes: Vec::new(),
    };
    memory.flat.put_u64(OWNED_STREAM + 0x28, BASE + 0x1bffc);
    assert_eq!(
        state.dma(&mut memory, &mut Vec::new()),
        Err(refuse!("GFX scalar publication crosses a UAT page"))
    );
    assert_eq!(memory.writes.len(), 2); // accepted master tail only
    assert_eq!((state.commands, state.events), (0, 0));
}

#[test]
fn full_event_ring_keeps_work_between_shared_signal_and_completion() {
    use event::tests::{EVENT_STATE, entry};
    let (mut state, flat, master) = owned_signal(&G14G);
    let mut memory = WorkMemory {
        flat,
        fail_write: None,
        fail_read: None,
        writes: Vec::new(),
    };
    memory.flat.put_u32(EVENT_STATE + WRITE, 0xff);
    let mut outbox = Vec::new();
    state.dma(&mut memory, &mut outbox).unwrap();
    assert_eq!(memory.writes.len(), 3);
    assert_eq!(memory.flat.read_u64(OWNED_SHARED), Ok(1));
    assert_eq!(memory.flat.read_u32(OWNED_FIRMWARE), Ok(0));
    assert_eq!(memory.flat.read_u32(OWNED_DRIVER), Ok(0));
    assert!(outbox.is_empty() && state.dma_pending());
    // The existing reset contract observes the still-owned work.
    let protocol = Protocol(Rc::new(RefCell::new(state)));
    assert!(Message(protocol.clone()).check_reset().is_err());
    memory.flat.put_u32(EVENT_STATE + READ, 1);
    protocol
        .0
        .borrow_mut()
        .dma(&mut memory, &mut outbox)
        .unwrap();
    assert_eq!(outbox.len(), 1);
    assert_eq!(memory.flat.read_u32(OWNED_FIRMWARE), Ok(0x100));
    assert_eq!(memory.flat.read_u32(OWNED_DRIVER), Ok(0x100));
    assert_eq!(memory.flat.read_u32(OWNED_CHANNEL + 0x1000), Ok(0));
    assert_eq!(
        entry(&mut memory.flat, 0xff),
        event::Event::SharedSignal {
            id: 0x55,
            value: 1,
            flag: 0
        }
        .encode()
    );
    assert!(Message(protocol.clone()).check_reset().is_err());
    memory.flat.put_u32(EVENT_STATE + READ, 2);
    protocol
        .0
        .borrow_mut()
        .dma(&mut memory, &mut outbox)
        .unwrap();
    assert_eq!(
        entry(&mut memory.flat, 0),
        event::Event::Flag(1 << 2).encode()
    );
    assert_eq!(outbox.len(), 2);
    assert_eq!(memory.flat.read_u32(master), Ok(1));
    assert_eq!(
        memory
            .writes
            .iter()
            .filter(|(va, _)| *va == OWNED_SHARED)
            .count(),
        1
    );
    assert!(!protocol.with(State::dma_pending));
    Message(protocol).check_reset().unwrap();
}

#[test]
fn failed_signal_acceptance_keeps_only_the_accepted_master_tail() {
    let (mut state, flat, _) = owned_signal(&G14G);
    let mut memory = WorkMemory {
        flat,
        fail_write: None,
        fail_read: Some(OWNED_COMMAND + 0x50),
        writes: Vec::new(),
    };
    let mut outbox = Vec::new();
    assert_eq!(
        state.dma(&mut memory, &mut outbox),
        Err(refuse!("injected work input read fault"))
    );
    assert_eq!(memory.writes.len(), 2);
    assert!(outbox.is_empty());
    assert_eq!(memory.flat.read_u64(OWNED_SHARED), Ok(0));
    memory.flat.put_u64(OWNED_STREAM + 0x20, 1 << 3);
    assert!(
        state
            .dma(&mut memory, &mut outbox)
            .unwrap_err()
            .to_string()
            .contains("mask bit 3")
    );
    assert_eq!(memory.writes.len(), 2);
    assert!(outbox.is_empty());
    memory.flat.put_u64(OWNED_STREAM + 0x20, 0);
    state.dma(&mut memory, &mut outbox).unwrap();
    assert_eq!((state.commands, state.events), (1, 2));
}

#[test]
fn a_shared_event_signal_completes_and_posts_its_events() {
    use crate::agx::event::Event;
    use crate::agx::event::tests::{EVENT_STATE, entry, event_ring};
    let (mut state, mut memory) = with_device_control(&[]);
    event_ring(&mut memory);
    let (channel_state, command) = (BASE + 0xd000, BASE + 0x10000);
    let (ring_state, _) = one_3d_entry(&mut memory, channel_state, command);
    let (stream, shared) = (command + 0x400, command + 0x800);
    memory.put_u32(command, 9);
    memory.put_u64(command + 0x1c, stream);
    memory.put_u64(command + 0x30, command + 0x900);
    memory.put_u64(command + 0x38, command + 0x908);
    memory.put_u32(command + 0x40, 0x100);
    memory.put_u32(command + 0x44, 2);
    memory.put_u32(stream, 0xe);
    memory.put_u64(stream + 0x10, 0x55);
    memory.put_u64(stream + 0x18, 1);
    memory.put_u64(stream + 0x28, shared);
    memory.put_u64(stream + 0x30, command + 0x908);
    memory.put_u32(stream + 0x38, 0x100);
    memory.put_u32(stream + 0x4c, 0x4000_0002);
    state.doorbell(work(1, 0)).unwrap();
    let mut outbox = Vec::new();
    state.dma(&mut memory, &mut outbox).unwrap();
    // Two events, each announced by the 0x42 message on 0x20.
    assert_eq!(
        outbox,
        [Mail {
            endpoint: MESSAGE_ENDPOINT,
            payload: EVENT_TYPE << TYPE_SHIFT
        }; 2]
    );
    assert_eq!(memory.read_u32(EVENT_STATE + WRITE), Ok(2));
    assert_eq!(
        entry(&mut memory, 0),
        Event::SharedSignal {
            id: 0x55,
            value: 1,
            flag: 0
        }
        .encode()
    );
    assert_eq!(entry(&mut memory, 1), Event::Flag(1 << 2).encode());
    assert_eq!(memory.read_u64(shared), Ok(1));
    assert_eq!(memory.read_u32(command + 0x900), Ok(0x100));
    // The command and the data-master entry are retired.
    assert_eq!(memory.read_u32(channel_state + 0x1000), Ok(1));
    assert_eq!(memory.read_u32(ring_state + READ), Ok(1));
    assert_eq!((state.commands, state.events), (1, 2));
}

#[test]
fn the_root_must_be_the_kexts_before_any_doorbell_runs() {
    let mut memory = Flat::new(BASE, 0x20000);
    let mut state = State::new(&G14G);
    state
        .init((MSG_INIT << TYPE_SHIFT) | (ROOT & INIT_VA_BITS))
        .unwrap();
    assert_eq!(
        state.dma(&mut memory, &mut Vec::new()).unwrap_err().what(),
        "GFX initdata root lacks its magic"
    );
    assert!(state.dma_pending());
    let mut early = State::new(&G14G);
    early
        .doorbell(doorbell_payload(DOORBELL_FWCTL, 0x11))
        .unwrap();
    assert!(early.dma(&mut memory, &mut Vec::new()).is_err());
}

#[test]
fn the_initdata_root_is_read_through_the_uat() {
    use stela_contracts::bus::Access;
    use stela_contracts::dma::test_support::Ram;
    use stela_contracts::interrupt::InterruptLatch;
    use uat::{Access as Ap, Agent, Flags, MemoryAttr, Uat};
    // TTBAT, PCE, PDE, PTE and the initdata page.
    let base = 0x8_0000_0000;
    let mut ram = Ram::new(base, 0x4000 * 6);
    let va = 0xffff_ffa0_0000_0000;
    let (_, idx) = uat::indices(va);
    let [ttbr0, ttbr1] = uat::encode_gptbat(0, 0, base + 0x4000);
    ram.put_u64(base, ttbr0);
    ram.put_u64(base + 8, ttbr1);
    ram.put_u64(
        base + 0x4000 + idx[0] as u64 * 8,
        uat::encode_table(base + 0x8000),
    );
    ram.put_u64(
        base + 0x8000 + idx[1] as u64 * 8,
        uat::encode_table(base + 0xc000),
    );
    let flags = Flags {
        access: Ap::Firmware,
        memory: MemoryAttr::Uncached,
        global: true,
        pxn: false,
        uxn: true,
        gpu_access: true,
    };
    ram.put_u64(
        base + 0xc000 + idx[2] as u64 * 8,
        uat::encode_pte(base + 0x1_0000, &flags),
    );
    // The status object root+a8 names is a distinct mapped page.
    let (_, status_idx) = uat::indices(va + 0x4000);
    ram.put_u64(
        base + 0xc000 + status_idx[2] as u64 * 8,
        uat::encode_pte(base + 0x1_4000, &flags),
    );
    let mut table = Uat::new(40);
    table.configure(base).unwrap();
    let mut latch = InterruptLatch::new();
    let mut state = State::new(&G14G);
    state
        .init((MSG_INIT << TYPE_SHIFT) | (va & INIT_VA_BITS))
        .unwrap();
    assert!(state.dma_pending());
    let mut run = |state: &mut State, ram: &mut Ram, table: &mut Uat| {
        let mut access = Access::new(0, 0, &mut latch);
        let mut view = table.agent(Agent::Firmware);
        let mut port = Port {
            iommu: &mut view,
            stream: uat::FIRMWARE_CONTEXT,
            memory: ram,
            access: &mut access,
        };
        state.dma(&mut port, &mut Vec::new())
    };
    // A root without the magic refuses and stays pending.
    assert!(run(&mut state, &mut ram, &mut table).is_err());
    assert!(state.dma_pending());
    put_root(&mut ram, base + 0x1_0000, va);
    run(&mut state, &mut ram, &mut table).unwrap();
    assert!(state.initdata_checked && !state.dma_pending());
    assert_eq!(
        state.root.as_ref().map(|root| root.rings),
        Some(va + 0x1000)
    );
}

/// Write a kext-shaped root at physical `at` whose objects follow `va`.
pub(crate) fn put_root(ram: &mut stela_contracts::dma::test_support::Ram, at: u64, va: u64) {
    ram.put_u64(at, MAGIC_G13_G14G);
    ram.put_u64(at + 0x28, 1 << 32);
    for (n, field) in [0x08, 0x18, 0x20, 0xa8, 0xb0, 0xb8, 0xc0]
        .into_iter()
        .enumerate()
    {
        ram.put_u64(at + field, va + 0x1000 * (n as u64 + 1));
    }
    // +0x18, the ring directory, is the second object.
    ram.put_u64(at + 0x18, va + 0x1000);
}
