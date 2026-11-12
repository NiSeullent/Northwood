use super::super::tests::{Io, Model, Timer, fixture, fixture_lpt, start, start_lpt};
use super::*;
use crate::bus::dma::{DmaBuffer, RegionAllocator};
use crate::ring::BUFFER;
use alloc::{rc::Rc, sync::Arc, vec};
use core::{cell::RefCell, sync::atomic::AtomicUsize};

struct Memory {
    // Native CPU RAM backs the model's published DMA addresses.
    // No asynchronous actor accesses it after the model is stopped/dropped.
    _ram: Vec<u128>,
    allocator: RegionAllocator,
}
impl Memory {
    fn new() -> Self {
        let mut ram = vec![0u128; (4096 * 1024 + 4096) / 16];
        let start = ram.as_mut_ptr() as u64;
        // SAFETY: exclusively owned native test RAM, used only by the
        // synchronous software device model while this owner is alive.
        let allocator = unsafe { RegionAllocator::new(start..start + (ram.len() * 16) as u64) };
        Self {
            _ram: ram,
            allocator,
        }
    }
}
impl DmaAllocator for Memory {
    fn allocate(
        &mut self,
        bytes: usize,
        align: usize,
    ) -> Result<DmaBuffer, crate::block::BlockError> {
        self.allocator.allocate(bytes, align)
    }
}
type Net = Ethernet<Io, Timer>;
type Setup = (Rc<RefCell<Model>>, Arc<AtomicUsize>, Controller<Io, Timer>);
fn setup() -> Setup {
    let (state, released, io, timer) = fixture();
    let controller = start(io, timer).unwrap();
    {
        let mut m = state.borrow_mut();
        m.mac
            .insert(super::super::STATUS, LINK_UP | FULL_DUPLEX | (1 << 7));
        m.phy.insert((2, 0, 1), PHY_LINK_READY);
        m.phy.insert((2, 0, 26), 0x1240);
        m.phy.insert((1, 770, 17), 1 << 14);
        // Preserve manageability-owned receive addresses verbatim.
        for index in 1..7 {
            m.mac.insert(RAL + index * 8, 0x1234_0000 | index);
            m.mac.insert(RAH + index * 8, 0xabcd_0000 | index);
        }
    }
    (state, released, controller)
}
fn setup_lpt(device: u16, lock: u32) -> Setup {
    let (state, released, io, timer) = fixture_lpt();
    let fw = state.borrow().mac[&super::super::super::FWSM] | (lock << 7);
    state.borrow_mut().mac.insert(super::super::super::FWSM, fw);
    let controller = start_lpt(device, io, timer).unwrap();
    {
        let mut m = state.borrow_mut();
        m.mac
            .insert(super::super::STATUS, LINK_UP | FULL_DUPLEX | (1 << 7));
        m.phy.insert((2, 0, 1), PHY_LINK_READY);
        m.phy.insert((1, 772, 20), 0xffff);
        m.phy.insert((1, 772, 28), 0xf800);
        for index in 0..11 {
            m.mac.insert(SHRAL_LPT + index * 8, 0x1234_0000 | index);
            m.mac.insert(SHRAH_LPT + index * 8, 0xabcd_0000 | index);
        }
    }
    (state, released, controller)
}
fn open(controller: Controller<Io, Timer>, memory: &mut Memory) -> Net {
    // SAFETY: the synchronous model's same native CPU RAM grant.
    unsafe { Ethernet::new(controller, memory) }.unwrap()
}
fn inject(net: &mut Net, index: usize, status: u32, length: usize, fill: u8) {
    net.receive.buffer(index).fill(fill);
    let base = index * DESCRIPTOR;
    net.receive.descriptors.write(base + 12, length as u16);
    fence(Ordering::Release);
    net.receive.descriptors.write(base + 8, status);
}
#[test]
#[ignore = "oracle: STELA_E1000E_PCH_LPT_ORACLE"]
fn unchanged_linux_lpt_shared_address_ownership_and_register_effects() {
    replay_shared(false);
}
fn replay_shared(spt: bool) {
    let cases = if spt {
        spt::tests::oracle_cases(2)
    } else {
        lpt::tests::oracle_cases(2)
    };
    replay_shared_cases(spt.then_some(0x156f), cases);
}
fn replay_shared_family(family: i219::Family) {
    replay_shared_cases(
        Some(i219::tests::sample_device(family)),
        i219::tests::oracle_cases(family, 2),
    );
}
pub(in crate::e1000e::pch::cold) fn replay_shared_cases(
    profile: Option<u16>,
    cases: Vec<([u32; 4], Vec<[u32; 3]>)>,
) {
    assert_eq!(cases.len(), 8);
    for ([lock, count, _, _], expected) in cases {
        assert_eq!(count, lpt::shared_count(lock << 7) + 1);
        let (state, _, controller) = if let Some(device) = profile {
            setup_integrated(device, lock)
        } else {
            setup_lpt(0x153a, lock)
        };
        let mut memory = Memory::new();
        let mut net = open(controller, &mut memory);
        {
            let mut m = state.borrow_mut();
            m.mac.insert(super::super::STATUS, 0x83);
            for i in 0..11 {
                m.mac.insert(SHRAL_LPT + i * 8, 0x1234_0000 | i);
                m.mac.insert(SHRAH_LPT + i * 8, 0xabcd_0000 | i);
            }
            m.operations.clear();
        }
        net.station_filters().unwrap();
        let actual: Vec<_> = state
            .borrow()
            .operations
            .iter()
            .copied()
            .filter(|o| {
                o[0] <= 1 && (o[1] == super::super::STATUS || (0x5400..0x5460).contains(&o[1]))
            })
            .collect();
        assert_eq!(actual, expected, "FWSM lock={lock}");
        net.shutdown().unwrap();
    }
}
#[test]
#[ignore = "oracle: STELA_E1000E_PCH_LPT_ORACLE"]
fn unchanged_linux_lpt_eee_disabled_phy_and_emi_effects() {
    replay_eee(false);
}
fn replay_eee(spt: bool) {
    let cases = if spt {
        spt::tests::oracle_cases(3)
    } else {
        lpt::tests::oracle_cases(3)
    };
    replay_eee_cases(spt.then_some(0x156f), cases);
}
fn replay_eee_family(family: i219::Family) {
    replay_eee_cases(
        Some(i219::tests::sample_device(family)),
        i219::tests::oracle_cases(family, 3),
    );
}
pub(in crate::e1000e::pch::cold) fn replay_eee_cases(
    profile: Option<u16>,
    cases: Vec<([u32; 4], Vec<[u32; 3]>)>,
) {
    assert_eq!(cases.len(), 3);
    for ([control, _, _, _], expected) in cases {
        let (state, _, controller) = if let Some(device) = profile {
            setup_integrated(device, 0)
        } else {
            setup_lpt(0x153b, 0)
        };
        let mut memory = Memory::new();
        let mut net = open(controller, &mut memory);
        let ext;
        {
            let mut m = state.borrow_mut();
            m.phy.insert((1, 772, 20), control as u16);
            m.emi.insert(0x9401, 0x3c00);
            m.emi.insert(0x8001, 0xffff);
            ext = m.mac[&super::super::super::EXTCNF_CTRL];
            m.operations.clear();
        }
        net.disable_eee().unwrap();
        let actual = super::super::tests::post_effects(&state.borrow().operations, ext);
        assert_eq!(actual, expected, "initial LPI control={control:x}");
        net.shutdown().unwrap();
    }
}
#[test]
fn only_the_32_mta_words_are_cleared_before_packet_initialization() {
    // Prove the hardware model refuses every word beyond the MTA
    // table instead of accepting all BAR-relative addresses as registers.
    for address in (0x5280..0x5400).step_by(4) {
        let (state, _, mut io, _) = fixture();
        assert_eq!(
            Registers::write32(&mut io, address, 0),
            Err(super::super::super::Fault::OutsideBar)
        );
        assert!(state.borrow().operations.is_empty());
    }
    let mut memory = Memory::new();
    let (state, released, controller) = setup();
    {
        let mut model = state.borrow_mut();
        for address in (MTA..0x5400).step_by(4) {
            model.mac.insert(address, 0xa55a_0000 | address);
        }
    }
    let mut net = open(controller, &mut memory);
    assert!(net.carrier());
    let model = state.borrow();
    let actual: Vec<_> = model
        .operations
        .iter()
        .copied()
        .filter(|o| o[0] == 1 && (MTA..0x5400).contains(&o[1]))
        .collect();
    let expected: Vec<_> = (0..32).map(|index| [1, MTA + index * 4, 0]).collect();
    assert_eq!(actual, expected);
    for address in (0x5280..0x5400).step_by(4) {
        assert_eq!(model.mac[&address], 0xa55a_0000 | address);
    }
    assert_eq!(model.mac[&super::super::RCTL], RECEIVE_POLICY);
    assert_eq!(model.mac[&super::super::TCTL], TRANSMIT_POLICY);
    assert_eq!(model.mac[&RFCTL] & EXTENDED_RX, EXTENDED_RX);
    assert_eq!(model.mac[&RDLEN], (DESCRIPTORS * DESCRIPTOR) as u32);
    assert_eq!(model.mac[&TDLEN], (DESCRIPTORS * DESCRIPTOR) as u32);
    drop(model);
    net.transmit(&[0x66; 60]).unwrap();
    inject(&mut net, 0, RX_DONE | RX_EOP, 64, 0x77);
    assert_eq!(net.receive().unwrap(), Some(vec![0x77; 60]));
    net.shutdown().unwrap();
    drop(net);
    assert_eq!(released.load(Ordering::SeqCst), 1);
}
#[test]
fn native_extended_rx_and_legacy_tx_payloads_wrap_without_offloads() {
    let mut memory = Memory::new();
    let (state, released, controller) = setup();
    let mut net = open(controller, &mut memory);
    assert!(net.carrier());
    for n in 0..97 {
        let size = if n % 3 == 0 { 14 } else { MAX_FRAME };
        let frame = vec![n as u8; size];
        let index = net.next_tx;
        net.transmit(&frame).unwrap();
        assert_eq!(&net.transmit.buffer(index)[..size], frame.as_slice());
        assert!(
            net.transmit.buffer(index)[size..size.max(MIN_FRAME)]
                .iter()
                .all(|b| *b == 0)
        );
        assert_eq!(
            net.transmit.descriptors.read::<u32>(index * DESCRIPTOR + 8),
            size.max(MIN_FRAME) as u32 | TX_COMMAND
        );
        // Hardware DD is the only retirement evidence. Head/tail
        // equality and a doorbell write never substitute for completion.
        net.transmit
            .descriptors
            .write(index * DESCRIPTOR + 12, TX_DONE);
        let index = net.next_rx;
        inject(&mut net, index, RX_DONE | RX_EOP, size + FCS_BYTES, n as u8);
        assert_eq!(net.receive().unwrap().unwrap(), frame);
        assert_eq!(
            net.receive.descriptors.read::<u64>(index * DESCRIPTOR),
            net.receive.address(index)
        );
        assert_eq!(
            net.receive.descriptors.read::<u64>(index * DESCRIPTOR + 8),
            0
        );
    }
    let m = state.borrow();
    assert_eq!(m.mac[&RFCTL] & EXTENDED_RX, EXTENDED_RX);
    assert_eq!(
        m.mac[&super::super::RCTL] & (1 << 26),
        0,
        "FCS is retained for software removal"
    );
    assert_eq!(m.mac[&TXDCTL], m.mac[&(TXDCTL + QUEUE_STRIDE)]);
    assert_eq!(m.phy[&(2, 0, 4)] & ADVERTISEMENT_PAUSE, 0);
    assert_eq!(m.phy[&(1, 772, 20)] & LPI_ENABLE, 0);
    assert_eq!(m.phy[&(1, 776, 23)], 0x7e00);
    for index in 1..7 {
        assert_eq!(m.mac[&(RAL + index * 8)], 0x1234_0000 | index);
        assert_eq!(m.mac[&(RAH + index * 8)], 0xabcd_0000 | index);
    }
    drop(m);
    net.shutdown().unwrap();
    assert_eq!(state.borrow().mac[&CTRL_EXT] & DRIVER_LOADED, 0);
    let accesses = state.borrow().accesses;
    assert!(net.transmit(&[0; 60]).is_err());
    assert_eq!(state.borrow().accesses, accesses);
    drop(net);
    assert_eq!(released.load(Ordering::SeqCst), 1);
}
#[test]
fn fragments_bad_status_and_bad_lengths_are_never_delivered() {
    let mut memory = Memory::new();
    let (_, _, controller) = setup();
    let mut net = open(controller, &mut memory);
    inject(&mut net, 0, RX_DONE, 64, 1);
    inject(&mut net, 1, RX_DONE | RX_EOP, 64, 2);
    inject(&mut net, 2, RX_DONE | RX_EOP | 0x0100_0000, 64, 3);
    inject(&mut net, 3, RX_DONE | RX_EOP, BUFFER + 1, 4);
    inject(&mut net, 4, RX_DONE | RX_EOP, 3, 5);
    inject(&mut net, 5, RX_DONE | RX_EOP, 64, 6);
    // The stack's reused buffer keeps its allocation and holds the frame.
    let mut reused = vec![0x52; 1514];
    let buffer = reused.as_ptr();
    assert_eq!(net.receive_reusing(&mut reused), Ok(true));
    assert_eq!(reused, [6; 60]);
    assert_eq!(reused.as_ptr(), buffer);
    assert_eq!(net.next_rx, 6);
    assert_eq!(net.receive_reusing(&mut reused), Ok(false));
    assert_eq!(reused, [6; 60]);
    inject(&mut net, 6, RX_DONE | RX_EOP, 64, 7);
    assert_eq!(net.receive().unwrap(), Some(vec![7; 60]));
    assert_eq!(net.receive().unwrap(), None);
}
#[test]
fn link_reconnect_and_speed_changes_apply_82579_workarounds() {
    let mut memory = Memory::new();
    let (state, _, controller) = setup();
    let mut net = open(controller, &mut memory);
    assert_eq!(state.borrow().phy[&(1, 770, 17)] & (1 << 14), 0);
    state.borrow_mut().mac.insert(super::super::STATUS, 0);
    assert!(!net.carrier());
    state.borrow_mut().mac.insert(super::super::STATUS, LINK_UP);
    state.borrow_mut().phy.insert((2, 0, 26), 0x1040);
    assert!(net.carrier());
    assert_eq!(state.borrow().mac[&TIPG] & 0x3ff, 0xff);
    assert_eq!(state.borrow().mac[&FEXTNVM4] & 7, 3);
    assert_eq!(state.borrow().emi[&RX_CONFIG], 0);
    assert_eq!(state.borrow().phy[&(1, 770, 16)] & 0x7000, 0x1000);
    state
        .borrow_mut()
        .mac
        .insert(super::super::STATUS, LINK_UP | FULL_DUPLEX | (1 << 6));
    state.borrow_mut().phy.insert((2, 0, 26), 0x1140);
    assert!(net.carrier());
    assert_eq!(state.borrow().mac[&TIPG] & 0x3ff, 8);
    assert_eq!(state.borrow().emi[&RX_CONFIG], 1);
    assert_eq!(state.borrow().phy[&(1, 770, 16)] & 0x7000, 0);
    // A same-speed LSC can arrive while PHY negotiation is still settling.
    // Do not cache a mismatched MAC/PHY observation as fully configured.
    state.borrow_mut().mac.insert(super::super::ICR, LINK_EVENT);
    state.borrow_mut().phy.insert((2, 0, 26), 0x1040);
    assert!(!net.carrier());
    assert_eq!(net.link, None);
    state.borrow_mut().phy.insert((2, 0, 26), 0x1140);
    assert!(net.carrier());
    assert!(!net.controller.is_poisoned());
}
#[test]
fn unretired_tx_cannot_be_overwritten_and_stall_poison_keeps_owner() {
    let mut memory = Memory::new();
    let (state, released, controller) = setup();
    let mut net = open(controller, &mut memory);
    for _ in 0..DESCRIPTORS - 1 {
        net.transmit(&[0x55; 60]).unwrap();
    }
    let before = net.transmit.descriptors.bytes().to_vec();
    let time = state.borrow().time;
    assert!(net.transmit(&[0x99; 60]).is_err());
    assert_eq!(state.borrow().time - time, 100_000);
    assert_eq!(net.transmit.descriptors.bytes(), before);
    assert_eq!(net.failure(), Some(&Error::QueueStalled));
    let accesses = state.borrow().accesses;
    assert!(net.receive().is_err());
    assert!(net.shutdown().is_err());
    assert_eq!(state.borrow().accesses, accesses);
    drop(net);
    assert_eq!(released.load(Ordering::SeqCst), 0);
}
#[test]
fn uncertain_tx_rx_doorbells_and_stop_drain_never_release_owner() {
    for lpt in [false, true] {
        for failure in 0..5 {
            let mut memory = Memory::new();
            let (state, released, controller) = if lpt { setup_lpt(0x153a, 0) } else { setup() };
            let mut net = open(controller, &mut memory);
            match failure {
                0 => {
                    // update_link's STATUS/ICR reads then the TDT store.
                    let at = state.borrow().accesses + 2;
                    state.borrow_mut().fail_at = Some(at);
                    assert!(net.transmit(&[0; 60]).is_err());
                    assert_eq!(net.next_tx, 0);
                }
                1 => {
                    // The batch's last frame carries the RDT store.
                    let last = crate::ring::RECEIVE_BATCH - 1;
                    for index in 0..=last {
                        inject(&mut net, index, RX_DONE | RX_EOP, 64, 0x44);
                    }
                    for _ in 0..last {
                        assert!(net.receive().unwrap().is_some());
                    }
                    let at = state.borrow().accesses + 2;
                    state.borrow_mut().fail_at = Some(at);
                    assert!(net.receive().is_err());
                    assert_eq!(net.next_rx, last);
                }
                2 => {
                    state.borrow_mut().master_busy = 10_000;
                    assert_eq!(net.shutdown(), Err(Error::MasterTimeout));
                }
                3 => {
                    let at = state.borrow().accesses;
                    state.borrow_mut().fail_at = Some(at);
                    assert!(net.shutdown().is_err());
                }
                4 => {
                    state
                        .borrow_mut()
                        .mac
                        .insert(super::super::STATUS, u32::MAX);
                    assert!(!net.carrier());
                    assert!(matches!(
                        net.failure(),
                        Some(Error::Phy(super::super::super::Error {
                            primary: super::super::super::Fault::Removed,
                            ..
                        }))
                    ));
                }
                _ => unreachable!(),
            }
            assert!(net.controller.is_poisoned());
            let accesses = state.borrow().accesses;
            assert!(!net.carrier());
            assert!(net.transmit(&[0; 60]).is_err());
            assert_eq!(state.borrow().accesses, accesses);
            drop(net);
            assert_eq!(released.load(Ordering::SeqCst), 0);
        }
    }
}
#[test]
fn every_queue_configuration_access_failure_prevents_reuse_and_keeps_owner() {
    let mut memory = Memory::new();
    let (state, _, controller) = setup();
    let first = state.borrow().accesses;
    let net = open(controller, &mut memory);
    let last = state.borrow().accesses;
    drop(net);
    for at in first..last {
        let mut memory = Memory::new();
        let (state, released, controller) = setup();
        state.borrow_mut().fail_at = Some(at);
        assert!(
            // SAFETY: the same isolated native-RAM model grant.
            unsafe { Ethernet::new(controller, &mut memory) }.is_err(),
            "lost queue fault at {at}"
        );
        assert_eq!(
            released.load(Ordering::SeqCst),
            0,
            "released actual owner at {at}"
        );
    }
}
#[test]
fn every_shutdown_access_failure_keeps_the_owner() {
    for lpt in [false, true] {
        let mut memory = Memory::new();
        let (state, _, controller) = if lpt { setup_lpt(0x153a, 0) } else { setup() };
        let mut net = open(controller, &mut memory);
        let first = state.borrow().accesses;
        net.shutdown().unwrap();
        let last = state.borrow().accesses;
        drop(net);
        for at in first..last {
            let mut memory = Memory::new();
            let (state, released, controller) = if lpt { setup_lpt(0x153a, 0) } else { setup() };
            let mut net = open(controller, &mut memory);
            state.borrow_mut().fail_at = Some(at);
            assert!(net.shutdown().is_err(), "lost stop fault at {at}");
            let accesses = state.borrow().accesses;
            drop(net);
            assert_eq!(
                state.borrow().accesses,
                accesses,
                "drop retried failed stop"
            );
            assert_eq!(
                released.load(Ordering::SeqCst),
                0,
                "released owner after stop fault {at}"
            );
        }
    }
}
#[test]
fn lpt_exact_devices_honor_shra_locks_ecc_and_packet_lifecycle() {
    for device in [0x153a, 0x153b, 0x15a0, 0x15a1] {
        for lock in 0..8 {
            let mut memory = Memory::new();
            let (state, released, controller) = setup_lpt(device, lock);
            let mut net = open(controller, &mut memory);
            assert!(net.carrier());
            {
                let m = state.borrow();
                let count = match lock {
                    0 => 11,
                    1 => 0,
                    _ => lock,
                };
                for index in 0..11 {
                    assert_eq!(
                        m.mac[&(SHRAL_LPT + index * 8)],
                        if index < count {
                            0
                        } else {
                            0x1234_0000 | index
                        }
                    );
                    assert_eq!(
                        m.mac[&(SHRAH_LPT + index * 8)],
                        if index < count {
                            0
                        } else {
                            0xabcd_0000 | index
                        }
                    );
                }
                assert_eq!(m.mac[&PBECCSTS] & ECC_ENABLE, ECC_ENABLE);
                assert_eq!(
                    m.mac[&super::super::CTRL] & MEMORY_ERROR_HANDLING,
                    MEMORY_ERROR_HANDLING
                );
                assert_eq!(m.mac[&0xf8], 0xc8b9_88b9);
                assert_eq!(m.phy[&(1, 772, 28)], 0xf8fa);
                assert_eq!(m.phy[&(1, 770, 17)] & (1 << 9), 1 << 9);
                assert_eq!(m.phy[&(1, 772, 20)] & LPI_ENABLE, 0);
                assert_eq!(m.emi[&0x8001], 0);
                assert!(!m.emi.contains_key(&LPI_PLL_SHUT));
                assert!(!m.kmrn.contains_key(&4));
                assert!(!m.kmrn.contains_key(&9));
            }
            for n in 0..65 {
                let frame = vec![n as u8; 60];
                let tx = net.next_tx;
                net.transmit(&frame).unwrap();
                assert_eq!(&net.transmit.buffer(tx)[..60], frame.as_slice());
                net.transmit
                    .descriptors
                    .write(tx * DESCRIPTOR + 12, TX_DONE);
                let rx = net.next_rx;
                inject(&mut net, rx, RX_DONE | RX_EOP, 64, n as u8);
                assert_eq!(net.receive().unwrap(), Some(frame));
            }
            net.shutdown().unwrap();
            assert_eq!(state.borrow().mac[&0xf8], 1 << 30);
            drop(net);
            assert_eq!(released.load(Ordering::SeqCst), 1);
        }
    }
}
#[test]
fn lpt_link_events_update_pll_emi_and_ltr() {
    let mut memory = Memory::new();
    let (state, _, controller) = setup_lpt(0x15a0, 0);
    let mut net = open(controller, &mut memory);
    for (status, tipg, ltr, emi) in [
        (LINK_UP, 0xff, 0xd003_9003, 0),
        (LINK_UP | FULL_DUPLEX | (1 << 6), 8, 0xcc3a_8c3a, 1),
        (LINK_UP | FULL_DUPLEX | (1 << 7), 8, 0xc8b9_88b9, 1),
    ] {
        state.borrow_mut().mac.insert(super::super::STATUS, status);
        assert!(net.carrier());
        let m = state.borrow();
        assert_eq!(m.mac[&TIPG] & 0x3ff, tipg);
        assert_eq!(m.mac[&0xf8], ltr);
        assert_eq!(m.emi[&0xb20c], emi);
        assert_eq!(
            m.phy[&(1, 772, 28)] & 0x7ff,
            if status & (1 << 7) != 0 { 0xfa } else { 0x3e8 }
        );
        assert_eq!(m.mac[&FEXTNVM4] & 7, 7);
    }
    state.borrow_mut().mac.insert(super::super::ICR, LINK_EVENT);
    state.borrow_mut().phy.insert((1, 772, 28), 0xffff);
    assert!(net.carrier());
    assert_eq!(state.borrow().phy[&(1, 772, 28)] & 0x7ff, 0xfa);
    state.borrow_mut().mac.insert(super::super::STATUS, 0);
    assert!(!net.carrier());
    assert_eq!(state.borrow().mac[&0xf8], 1 << 30);
}
#[test]
fn every_lpt_queue_config_and_link_transition_access_failure_keeps_owner() {
    let mut memory = Memory::new();
    let (state, _, controller) = setup_lpt(0x153a, 0);
    let first = state.borrow().accesses;
    let mut net = open(controller, &mut memory);
    let configured = state.borrow().accesses;
    state.borrow_mut().mac.insert(super::super::STATUS, LINK_UP);
    assert!(net.carrier());
    let linked = state.borrow().accesses;
    drop(net);
    for at in first..linked {
        let mut memory = Memory::new();
        let (state, released, controller) = setup_lpt(0x153a, 0);
        if at < configured {
            state.borrow_mut().fail_at = Some(at);
            assert!(
                // SAFETY: isolated same-function model's native RAM.
                unsafe { Ethernet::new(controller, &mut memory) }.is_err(),
                "lost LPT queue fault at{at}"
            );
        } else {
            let mut net = open(controller, &mut memory);
            state.borrow_mut().mac.insert(super::super::STATUS, LINK_UP);
            state.borrow_mut().fail_at = Some(at);
            assert!(!net.carrier(), "lost LPT link fault at{at}");
            let accesses = state.borrow().accesses;
            assert!(net.transmit(&[0; 60]).is_err());
            assert!(net.shutdown().is_err());
            assert_eq!(state.borrow().accesses, accesses);
            drop(net);
        }
        assert_eq!(
            released.load(Ordering::SeqCst),
            0,
            "released LPT owner at{at}"
        );
    }
}
fn setup_spt(device: u16, lock: u32) -> Setup {
    let (state, released, io, timer) = spt::tests::fixture_spt();
    let fw = state.borrow().mac[&super::super::super::FWSM] | (lock << 7);
    state.borrow_mut().mac.insert(super::super::super::FWSM, fw);
    let controller = spt::tests::start_spt(device, io, timer).unwrap();
    {
        let mut m = state.borrow_mut();
        m.mac
            .insert(super::super::STATUS, LINK_UP | FULL_DUPLEX | (1 << 7));
        m.phy.insert((2, 0, 1), PHY_LINK_READY);
        for i in 0..11 {
            m.mac.insert(SHRAL_LPT + i * 8, 0x1234_0000 | i);
            m.mac.insert(SHRAH_LPT + i * 8, 0xabcd_0000 | i);
        }
    }
    (state, released, controller)
}
#[test]
fn spt_cohort_ecc_shra_errata_link_payload_wrap_and_live_stop_status() {
    for device in [0x156f, 0x1570] {
        for lock in 0..8 {
            let (state, released, controller) = setup_spt(device, lock);
            let mut memory = Memory::new();
            let mut net = open(controller, &mut memory);
            {
                let m = state.borrow();
                let count = lpt::shared_count(lock << 7);
                for i in 0..11 {
                    assert_eq!(
                        m.mac[&(SHRAL_LPT + i * 8)],
                        if i < count { 0 } else { 0x1234_0000 | i }
                    );
                }
                assert_ne!(m.mac[&PBECCSTS] & ECC_ENABLE, 0);
                assert_ne!(m.mac[&super::super::CTRL] & MEMORY_ERROR_HANDLING, 0);
                assert_ne!(m.mac[&0xf28] & 0x1_0000, 0);
                assert_eq!(m.mac[&TARC] & 0x3000_0000, 0x2000_0000);
                assert_ne!(m.mac[&0xe4] & 4, 0);
                assert_eq!(m.mac[&0x5bb4] & 0x1800, 0x1800);
                assert!((m.phy[&(1, 776, 20)] >> 2) & 0x3ff >= 0x18);
            }
            for n in 0..65 {
                let frame = vec![n as u8; 60];
                let tx = net.next_tx;
                net.transmit(&frame).unwrap();
                assert_eq!(&net.transmit.buffer(tx)[..60], frame.as_slice());
                net.transmit
                    .descriptors
                    .write(tx * DESCRIPTOR + 12, TX_DONE);
                let rx = net.next_rx;
                inject(&mut net, rx, RX_DONE | RX_EOP, 64, n as u8);
                assert_eq!(net.receive().unwrap(), Some(frame));
            }
            net.shutdown().unwrap();
            drop(net);
            assert_eq!(released.load(Ordering::SeqCst), 1);
        }
    }
}
#[test]
fn spt_all_speeds_apply_ipg_gap_and_analog_k1_policy() {
    let (state, _, controller) = setup_spt(0x1570, 0);
    let mut memory = Memory::new();
    let mut net = open(controller, &mut memory);
    for (status, ipg, gap) in [
        (LINK_UP, 0xff, 0xc023),
        (LINK_UP | FULL_DUPLEX, 0xc, 0xc023),
        (LINK_UP | (1 << 6), 8, 0xc023),
        (LINK_UP | FULL_DUPLEX | (1 << 6), 0xc, 0xc023),
        (LINK_UP | FULL_DUPLEX | (1 << 7), 8, 0x60),
    ] {
        let mut m = state.borrow_mut();
        m.mac.insert(super::super::STATUS, status);
        m.phy.insert((1, 776, 20), 0);
        m.mac.insert(0xf18, 1 << 31);
        drop(m);
        assert!(net.carrier());
        let m = state.borrow();
        assert_eq!(m.mac[&TIPG] & 0x3ff, ipg);
        assert_eq!(m.phy[&(1, 776, 20)], gap);
        assert_eq!(m.mac[&0x10] & (1 << 31), 1 << 31);
    }
    let mut m = state.borrow_mut();
    m.mac.insert(0xf18, 0);
    m.mac.insert(super::super::ICR, LINK_EVENT);
    drop(m);
    assert!(net.carrier());
    assert_eq!(state.borrow().mac[&0x10] & (1 << 31), 0);
}
#[test]
fn spt_owned_ring_flush_uses_dummy_dma_and_rx_fallback_before_vendor_drain() {
    for rx in [false, true] {
        for tail in [0, 31] {
            let (state, released, controller) = setup_spt(0x156f, 0);
            let mut memory = Memory::new();
            let mut net = open(controller, &mut memory);
            net.next_tx = tail;
            net.clean_tx = tail;
            {
                let mut m = state.borrow_mut();
                m.mac.insert(TDT, tail as u32);
                m.config.insert(0xe4, 0x100);
                m.flush_after_tx = !rx;
                m.flush_after_rx = rx;
                m.operations.clear();
            }
            net.shutdown().unwrap();
            let base = tail * DESCRIPTOR;
            assert_eq!(
                net.transmit.descriptors.read::<u64>(base),
                net.transmit.descriptors.bus_address()
            );
            assert_eq!(
                net.transmit.descriptors.read::<u32>(base + 8),
                512 | (1 << 25)
            );
            assert_eq!(net.transmit.descriptors.read::<u32>(base + 12), 0);
            assert_eq!(state.borrow().config[&0xe4] & 0x100, 0);
            assert!(state.borrow().operations.contains(&[2, 200, 0]));
            let m = state.borrow();
            let doorbell = m
                .operations
                .iter()
                .position(|o| o[0] == 1 && o[1] == TDT)
                .unwrap();
            let drained = m
                .operations
                .iter()
                .position(|o| {
                    o[0] == 1
                        && o[1] == super::super::CTRL
                        && o[2] & super::super::MASTER_DISABLE != 0
                })
                .unwrap();
            assert!(doorbell < drained);
            drop(m);
            drop(net);
            assert_eq!(released.load(Ordering::SeqCst), 1);
        }
    }
}
#[test]
fn unresolved_flush_tail_mismatch_or_independent_master_timeout_never_release_owner() {
    for failure in 0..3 {
        let (state, released, controller) = setup_spt(0x156f, 0);
        let mut memory = Memory::new();
        let mut net = open(controller, &mut memory);
        {
            let mut m = state.borrow_mut();
            m.config.insert(0xe4, 0x100);
            m.flush_after_tx = failure != 0;
            if failure == 1 {
                m.mac.insert(TDT, 7);
            }
            if failure == 2 {
                m.master_busy = 2000;
            }
        }
        assert!(net.shutdown().is_err());
        let accesses = state.borrow().accesses;
        assert!(net.transmit(&[0; 60]).is_err());
        assert!(!net.carrier());
        drop(net);
        assert_eq!(state.borrow().accesses, accesses);
        assert_eq!(released.load(Ordering::SeqCst), 0);
    }
}
#[test]
fn every_spt_config_link_and_flushing_stop_access_failure_keeps_owner() {
    let (state, _, controller) = setup_spt(0x156f, 0);
    let mut memory = Memory::new();
    let first = state.borrow().accesses;
    let mut net = open(controller, &mut memory);
    let configured = state.borrow().accesses;
    state
        .borrow_mut()
        .mac
        .insert(super::super::STATUS, LINK_UP | FULL_DUPLEX | (1 << 6));
    assert!(net.carrier());
    let linked = state.borrow().accesses;
    {
        let mut m = state.borrow_mut();
        m.config.insert(0xe4, 0x100);
        m.flush_after_rx = true;
    }
    net.shutdown().unwrap();
    let last = state.borrow().accesses;
    drop(net);
    for at in first..last {
        let (state, released, controller) = setup_spt(0x156f, 0);
        let mut memory = Memory::new();
        if at < configured {
            state.borrow_mut().fail_at = Some(at); /* SAFETY: actual isolated native RAM grant. */
            assert!(
                // SAFETY: the model device, its pages and its domain belong to this test alone;
                // no other driver, worker or DMA touches them.
                unsafe { Ethernet::new(controller, &mut memory) }.is_err(),
                "lost SPT config{at}"
            );
        } else {
            let mut net = open(controller, &mut memory);
            state
                .borrow_mut()
                .mac
                .insert(super::super::STATUS, LINK_UP | FULL_DUPLEX | (1 << 6));
            if at < linked {
                state.borrow_mut().fail_at = Some(at);
                assert!(!net.carrier(), "lost SPT link{at}");
            } else {
                assert!(net.carrier());
                let mut m = state.borrow_mut();
                m.config.insert(0xe4, 0x100);
                m.flush_after_rx = true;
                m.fail_at = Some(at);
                drop(m);
                assert!(net.shutdown().is_err(), "lost SPT stop{at}");
            }
            let count = state.borrow().accesses;
            assert!(net.transmit(&[0; 60]).is_err());
            assert!(!net.carrier());
            drop(net);
            assert_eq!(state.borrow().accesses, count);
        }
        assert_eq!(released.load(Ordering::SeqCst), 0, "released SPT owner{at}");
    }
}
#[test]
#[ignore = "oracle: STELA_E1000E_PCH_SPT_ORACLE"]
fn unchanged_linux_spt_shared_address_and_eee_disabled_effects() {
    replay_shared(true);
    replay_eee(true);
}
#[test]
#[ignore = "oracle: STELA_E1000E_PCH_SPT_ORACLE"]
fn unchanged_linux_spt_link_phy_mac_and_platform_workarounds() {
    replay_i219_link(0x156f, spt::tests::oracle_cases(6));
}
pub(in crate::e1000e::pch::cold) fn replay_i219_link(
    device: u16,
    cases: Vec<([u32; 4], Vec<[u32; 3]>)>,
) {
    assert_eq!(cases.len(), 48);
    for ([status, analog, gap, _], expected) in cases {
        let (state, _, controller) = setup_integrated(device, 0);
        let mut memory = Memory::new();
        let mut net = open(controller, &mut memory);
        net.link = None;
        {
            let mut m = state.borrow_mut();
            m.mac.insert(super::super::STATUS, status);
            m.mac.insert(TIPG, 0x76543210);
            m.mac.insert(FEXTNVM4, 0xabcdef00);
            m.mac
                .insert(0x10, 0x23456789 | if analog != 0 { 0 } else { 1 << 31 });
            m.mac.insert(0xf18, if analog != 0 { 1 << 31 } else { 0 });
            m.mac.insert(PBA, 26);
            m.phy
                .insert((2, 0, 1), if status & 2 != 0 { 0x24 } else { 0 });
            m.phy.insert((1, 772, 28), 0xf812);
            m.phy.insert((1, 770, 17), 0xa533);
            m.phy.insert((1, 776, 20), gap as u16);
            m.phy.insert((1, 772, 20), 0xffff);
            m.emi.insert(0x9401, 0x3c00);
            m.operations.clear();
        }
        assert_eq!(net.carrier(), status & 2 != 0);
        let actual = spt::tests::source_effects(&state.borrow().operations, 6);
        assert_eq!(
            actual, expected,
            "status={status:x} analog={analog} gap={gap:x}"
        );
        net.shutdown().unwrap();
    }
}
#[test]
#[ignore = "oracle: STELA_E1000E_PCH_SPT_ORACLE"]
fn unchanged_linux_spt_owned_descriptor_flush_effects_and_exact_dma_bytes() {
    replay_i219_flush(0x156f, spt::tests::oracle_cases(7));
}
pub(in crate::e1000e::pch::cold) fn replay_i219_flush(
    device: u16,
    cases: Vec<([u32; 4], Vec<[u32; 3]>)>,
) {
    assert_eq!(cases.len(), 15);
    for ([scenario, tail, _, _], expected) in cases {
        let (state, _, controller) = setup_integrated(device, 0);
        let mut memory = Memory::new();
        let mut net = open(controller, &mut memory);
        net.next_tx = tail as usize;
        net.clean_tx = tail as usize;
        {
            let mut m = state.borrow_mut();
            m.mac.insert(spt::FEXTNVM11, 0x9234);
            m.mac.insert(TDLEN, if scenario == 4 { 0 } else { 512 });
            m.mac.insert(super::super::TCTL, 0x12345670);
            m.mac.insert(TDT, tail);
            m.mac.insert(super::super::RCTL, 0x76543212);
            m.mac.insert(RXDCTL, 0xa5a53fff);
            m.mac.insert(super::super::STATUS, 0x83);
            m.config.insert(0xe4, if scenario == 0 { 0 } else { 0x100 });
            m.flush_after_tx = scenario == 1;
            m.flush_after_rx = scenario == 2;
            m.operations.clear();
        }
        let result = net.spt_flush_descriptors();
        assert_eq!(
            result,
            if scenario == 3 {
                Err(Error::DescriptorFlushRequired)
            } else {
                Ok(())
            }
        );
        let mut cfg_reads = 0;
        let mut actual: Vec<_> = state
            .borrow()
            .operations
            .iter()
            .copied()
            .filter(|o| {
                if o[0] == 7 {
                    cfg_reads += 1;
                    return cfg_reads <= 2;
                }
                o[0] <= 2
            })
            .collect();
        if (1..=3).contains(&scenario) {
            let base = tail as usize * DESCRIPTOR;
            actual.extend_from_slice(&[
                [
                    21,
                    0,
                    u32::from(
                        net.transmit.descriptors.read::<u64>(base)
                            == net.transmit.descriptors.bus_address(),
                    ),
                ],
                [21, 8, net.transmit.descriptors.read::<u32>(base + 8)],
                [21, 12, net.transmit.descriptors.read::<u32>(base + 12)],
                [21, 16, net.next_tx as u32],
            ]);
        }
        assert_eq!(actual, expected, "scenario={scenario} tail={tail}");
        if result.is_err() {
            net.poison();
        } else {
            state.borrow_mut().config.insert(0xe4, 0);
            net.shutdown().unwrap();
        }
    }
}
#[test]
fn spt_flush_uses_the_reserved_slot_without_overwriting_unretired_packet_descriptors() {
    let (state, released, controller) = setup_spt(0x156f, 0);
    let mut memory = Memory::new();
    let mut net = open(controller, &mut memory);
    for n in 0..DESCRIPTORS - 1 {
        net.transmit(&[n as u8; 60]).unwrap();
    }
    assert_eq!(net.next_tx, DESCRIPTORS - 1);
    assert_eq!(net.clean_tx, 0);
    let before: Vec<_> = (0..DESCRIPTORS - 1)
        .map(|i| {
            (
                net.transmit.descriptors.read::<u64>(i * DESCRIPTOR),
                net.transmit.descriptors.read::<u64>(i * DESCRIPTOR + 8),
            )
        })
        .collect();
    {
        let mut m = state.borrow_mut();
        m.config.insert(0xe4, 0x100);
        m.flush_after_tx = true;
    }
    net.shutdown().unwrap();
    for (i, &(address, command)) in before.iter().enumerate() {
        assert_eq!(
            net.transmit.descriptors.read::<u64>(i * DESCRIPTOR),
            address
        );
        assert_eq!(
            net.transmit.descriptors.read::<u64>(i * DESCRIPTOR + 8),
            command
        );
        assert_eq!(&net.transmit.buffer(i)[..60], vec![i as u8; 60]);
    }
    drop(net);
    assert_eq!(released.load(Ordering::SeqCst), 1);
}
fn setup_i219(device: u16, lock: u32) -> Setup {
    setup_integrated(device, lock)
}
fn setup_integrated(device: u16, lock: u32) -> Setup {
    let (state, released, io, timer) = i219::tests::fixture();
    let fw = state.borrow().mac[&super::super::super::FWSM] | (lock << 7);
    state.borrow_mut().mac.insert(super::super::super::FWSM, fw);
    let controller = super::super::mtp::tests::start_any(device, io, timer).unwrap();
    {
        let mut m = state.borrow_mut();
        m.mac
            .insert(super::super::STATUS, LINK_UP | FULL_DUPLEX | (1 << 7));
        m.phy.insert((2, 0, 1), PHY_LINK_READY);
        for i in 0..11 {
            m.mac.insert(SHRAL_LPT + i * 8, 0x1234_0000 | i);
            m.mac.insert(SHRAH_LPT + i * 8, 0xabcd_0000 | i);
        }
    }
    (state, released, controller)
}
#[test]
#[ignore = "oracle: STELA_E1000E_PCH_I219_ADP_ORACLE, STELA_E1000E_PCH_I219_CNP_ORACLE, STELA_E1000E_PCH_I219_SPT_ORACLE, STELA_E1000E_PCH_I219_TGP_ORACLE"]
fn unchanged_family_shra_eee_link_flush_and_dma_clock_effects() {
    for family in i219::tests::families() {
        replay_shared_family(family);
        replay_eee_family(family);
        replay_i219_link(
            i219::tests::sample_device(family),
            i219::tests::oracle_cases(family, 6),
        );
        replay_i219_flush(
            i219::tests::sample_device(family),
            i219::tests::oracle_cases(family, 7),
        );
        let cases = i219::tests::oracle_cases(family, 8);
        assert_eq!(cases.len(), 2);
        for ([initial, _, _, _], expected) in cases {
            let (state, _, controller) = setup_i219(i219::tests::sample_device(family), 0);
            let mut memory = Memory::new();
            let mut net = open(controller, &mut memory);
            state.borrow_mut().mac.insert(FFLT_DBG, initial);
            state.borrow_mut().operations.clear();
            net.i219_clock_policy().unwrap();
            let actual: Vec<_> = state
                .borrow()
                .operations
                .iter()
                .copied()
                .filter(|o| o[0] <= 1 && o[1] == FFLT_DBG)
                .collect();
            assert_eq!(actual, expected, "DMA clock{family:?} initial{initial:x}");
            assert_eq!(
                state.borrow().mac[&FFLT_DBG] & !i219::DONT_GATE_WAKE_DMA_CLOCK,
                initial & !i219::DONT_GATE_WAKE_DMA_CLOCK
            );
            net.shutdown().unwrap();
        }
    }
}
#[test]
fn all_declared_i219_ids_complete_cold_packet_wrap_and_live_flushing_stop() {
    let devices = i219::tests::inventory();
    assert_eq!(devices.len(), 39);
    for (n, (device, family)) in devices.into_iter().enumerate() {
        let lock = n as u32 % 8;
        let (state, released, controller) = setup_i219(device, lock);
        {
            let mut m = state.borrow_mut();
            m.mac.insert(
                spt::IOSFPC,
                if family == i219::Family::Spt {
                    0x4000
                } else {
                    u32::MAX
                },
            );
            m.mac.insert(TARC, 0x3000_0000);
            m.mac.insert(FFLT_DBG, 0x2345_0abc);
            m.mac.insert(spt::PCIE_ANALOG, 1 << 31);
            m.mac.insert(0x10, 0x23456789);
        }
        let mut memory = Memory::new();
        let mut net = open(controller, &mut memory);
        {
            let m = state.borrow();
            assert_eq!(
                m.mac[&spt::IOSFPC],
                if family == i219::Family::Spt {
                    0x4000 | spt::CORRUPTION_THRESHOLD
                } else {
                    u32::MAX
                }
            );
            assert_eq!(
                m.mac[&TARC] & spt::TARC_REQUEST_MASK,
                if family == i219::Family::Spt {
                    spt::TARC_TWO_REQUESTS
                } else {
                    0x3000_0000
                }
            );
            assert_eq!(
                m.mac[&0x10] & (1 << 31),
                if family == i219::Family::Spt {
                    1 << 31
                } else {
                    0
                }
            );
            assert_eq!(
                m.mac[&FFLT_DBG] & 0x1000,
                if family.dma_clock_workaround() {
                    0x1000
                } else {
                    0
                }
            );
            let count = lpt::shared_count(lock << 7);
            for i in 0..11 {
                assert_eq!(
                    m.mac[&(SHRAL_LPT + i * 8)],
                    if i < count { 0 } else { 0x1234_0000 | i }
                );
            }
        }
        for packet in 0..65 {
            let frame = vec![packet as u8; 60];
            let tx = net.next_tx;
            net.transmit(&frame).unwrap();
            assert_eq!(&net.transmit.buffer(tx)[..60], frame.as_slice());
            net.transmit
                .descriptors
                .write(tx * DESCRIPTOR + 12, TX_DONE);
            let rx = net.next_rx;
            inject(&mut net, rx, RX_DONE | RX_EOP, 64, packet as u8);
            assert_eq!(net.receive().unwrap(), Some(frame));
        }
        {
            let mut m = state.borrow_mut();
            m.config.insert(0xe4, 0x100);
            m.flush_after_tx = n % 2 == 0;
            m.flush_after_rx = n % 2 != 0;
        }
        net.shutdown().unwrap();
        assert_eq!(state.borrow().config[&0xe4] & 0x100, 0);
        assert_eq!(state.borrow().mac[&0xf8], 1 << 30);
        drop(net);
        assert_eq!(released.load(Ordering::SeqCst), 1, "retained {device:04x}");
    }
}
#[test]
fn each_new_family_cold_config_link_and_flushing_stop_fault_keeps_owner() {
    for family in i219::tests::families() {
        let device = i219::tests::sample_device(family);
        let (state, _, io, timer) = i219::tests::fixture();
        let mut memory = Memory::new();
        let controller = super::super::mtp::tests::start_any(device, io, timer).unwrap();
        let cold = state.borrow().accesses;
        {
            let mut m = state.borrow_mut();
            m.mac
                .insert(super::super::STATUS, LINK_UP | FULL_DUPLEX | (1 << 7));
            m.phy.insert((2, 0, 1), PHY_LINK_READY);
        }
        let mut net = open(controller, &mut memory);
        let configured = state.borrow().accesses;
        state
            .borrow_mut()
            .mac
            .insert(super::super::STATUS, LINK_UP | FULL_DUPLEX | (1 << 6));
        assert!(net.carrier());
        let linked = state.borrow().accesses;
        {
            let mut m = state.borrow_mut();
            m.config.insert(0xe4, 0x100);
            m.flush_after_rx = true;
        }
        net.shutdown().unwrap();
        let last = state.borrow().accesses;
        drop(net);
        for at in 0..last {
            let (state, released, io, timer) = i219::tests::fixture();
            let mut memory = Memory::new();
            if at < cold {
                state.borrow_mut().fail_at = Some(at);
                assert!(
                    i219::tests::start(device, io, timer).is_err(),
                    "lost cold{family:?}:{at}"
                );
            } else {
                let controller = super::super::mtp::tests::start_any(device, io, timer).unwrap();
                {
                    let mut m = state.borrow_mut();
                    m.mac
                        .insert(super::super::STATUS, LINK_UP | FULL_DUPLEX | (1 << 7));
                    m.phy.insert((2, 0, 1), PHY_LINK_READY);
                }
                if at < configured {
                    state.borrow_mut().fail_at = Some(at);
                    assert!(
                        // SAFETY: the same isolated model's native RAM.
                        unsafe { Ethernet::new(controller, &mut memory) }.is_err(),
                        "lost config{family:?}:{at}"
                    );
                } else {
                    let mut net = open(controller, &mut memory);
                    state
                        .borrow_mut()
                        .mac
                        .insert(super::super::STATUS, LINK_UP | FULL_DUPLEX | (1 << 6));
                    if at < linked {
                        state.borrow_mut().fail_at = Some(at);
                        assert!(!net.carrier(), "lost link{family:?}:{at}");
                    } else {
                        assert!(net.carrier());
                        let mut m = state.borrow_mut();
                        m.config.insert(0xe4, 0x100);
                        m.flush_after_rx = true;
                        m.fail_at = Some(at);
                        drop(m);
                        assert!(net.shutdown().is_err(), "lost stop{family:?}:{at}");
                    }
                    let count = state.borrow().accesses;
                    assert!(net.transmit(&[0; 60]).is_err());
                    assert!(!net.carrier());
                    drop(net);
                    assert_eq!(state.borrow().accesses, count);
                }
            }
            assert_eq!(
                released.load(Ordering::SeqCst),
                0,
                "released {family:?}:{at}"
            );
        }
    }
}
#[test]
#[ignore = "oracle: STELA_E1000E_PCH_I219_ADP_ORACLE, STELA_E1000E_PCH_I219_CNP_ORACLE, STELA_E1000E_PCH_I219_SPT_ORACLE, STELA_E1000E_PCH_I219_TGP_ORACLE"]
fn source_nvm_outcomes_compose_through_every_cold_reset_packet_and_stop() {
    for family in i219::tests::families() {
        for ([marker, check, expected, _], _) in i219::tests::oracle_cases(family, 9) {
            let (state, released, io, timer) = i219::tests::fixture();
            {
                let mut m = state.borrow_mut();
                m.words[3] = (m.words[3] & !1) | marker as u16;
                let sum = m.words[..63].iter().copied().fold(0u16, u16::wrapping_add);
                m.words[63] = 0xbabau16.wrapping_sub(sum);
                if check == 1 {
                    m.words[63] ^= 1;
                }
                if check == 2 {
                    m.words[63] = 0xffff;
                }
            }
            let result = i219::tests::start(i219::tests::sample_device(family), io, timer);
            if expected != 0 {
                let kind = match result {
                    Err(Error::Nvm(nvm::Error::Checksum(_))) => 1,
                    Err(Error::Nvm(nvm::Error::RepairRequired)) => 2,
                    _ => panic!(
                        "accepted invalid family image{family:?} marker{marker} check{check}"
                    ),
                };
                assert_eq!(kind, expected);
                assert_eq!(released.load(Ordering::SeqCst), 0);
                continue;
            }
            let controller = result.unwrap();
            if family == i219::Family::Tgp && marker == 0 {
                assert!(
                    !state.borrow().operations.contains(&[6, 8, 0x7c]),
                    "TGP skip incorrectly read checksum word"
                );
            }
            {
                let mut m = state.borrow_mut();
                m.mac
                    .insert(super::super::STATUS, LINK_UP | FULL_DUPLEX | (1 << 7));
                m.phy.insert((2, 0, 1), PHY_LINK_READY);
            }
            let mut memory = Memory::new();
            let mut net = open(controller, &mut memory);
            let frame = [0x71; 60];
            net.transmit(&frame).unwrap();
            net.transmit.descriptors.write(12, TX_DONE);
            inject(&mut net, 0, RX_DONE | RX_EOP, 64, 0x71);
            assert_eq!(net.receive().unwrap(), Some(frame.to_vec()));
            net.shutdown().unwrap();
            drop(net);
            assert_eq!(released.load(Ordering::SeqCst), 1);
        }
    }
}
#[test]
fn every_mtp_device_completes_packets() {
    use super::super::mtp;
    let devices: Vec<_> = (0..=u16::MAX)
        .filter(|id| mtp::Family::from_device(*id).is_some())
        .collect();
    assert_eq!(devices.len(), 14);
    for device in devices {
        for disable in [false, true] {
            let (state, released, io, timer) = spt::tests::fixture_spt();
            {
                let mut m = state.borrow_mut();
                m.kmrn.insert(7, 0xa5a7);
                m.phy.insert((1, 770, 21), 0x1234);
                m.mac.insert(0x18, 0x5a5a0008);
            }
            // SAFETY: one source-selected host/Function/reset model grant.
            let controller =
                unsafe { Controller::new_mtp(device, io, timer, mtp::tests::host(disable)) }
                    .unwrap();
            let ptp = mtp::Family::from_device(device) == Some(mtp::Family::Ptp);
            assert_eq!(state.borrow().mac[&0x18] & 8, if ptp { 0 } else { 8 });
            {
                let mut m = state.borrow_mut();
                m.mac
                    .insert(super::super::STATUS, LINK_UP | FULL_DUPLEX | (1 << 7));
                m.phy.insert((2, 0, 1), PHY_LINK_READY);
                m.mac.insert(spt::IOSFPC, u32::MAX);
            }
            let mut memory = Memory::new();
            let mut net = open(controller, &mut memory);
            {
                let m = state.borrow();
                assert_eq!(m.kmrn[&7] & 2, if disable { 0 } else { 2 });
                assert_eq!(m.phy[&(1, 770, 21)] & 0xfc0, 0xf00);
                assert_eq!(m.mac[&0x5bc0] & 0xc00000, 0x800000);
                assert_eq!(m.mac[&FFLT_DBG] & 0x1000, 0x1000);
                assert_eq!(m.phy[&(1, 772, 28)] & lpt::PLL_CLOCK_GATE_MASK, 0x1d5);
                assert_eq!(m.mac[&spt::IOSFPC], u32::MAX);
            }
            for n in 0..65 {
                let frame = vec![n as u8; 60];
                let tx = net.next_tx;
                net.transmit(&frame).unwrap();
                assert_eq!(&net.transmit.buffer(tx)[..60], frame.as_slice());
                net.transmit
                    .descriptors
                    .write(tx * DESCRIPTOR + 12, TX_DONE);
                let rx = net.next_rx;
                inject(&mut net, rx, RX_DONE | RX_EOP, 64, n as u8);
                assert_eq!(net.receive().unwrap(), Some(frame));
            }
            {
                let mut m = state.borrow_mut();
                m.config.insert(0xe4, 0x100);
                m.flush_after_tx = disable;
                m.flush_after_rx = !disable;
            }
            net.shutdown().unwrap();
            assert_eq!(state.borrow().config[&0xe4] & 0x100, 0);
            assert_eq!(state.borrow().mac[&0xf8], 1 << 30);
            drop(net);
            assert_eq!(released.load(Ordering::SeqCst), 1);
        }
    }
}
#[test]
#[ignore = "oracle: STELA_E1000E_PCH_MTP_ORACLE, STELA_E1000E_PCH_PTP_ORACLE"]
fn mandatory_mtp_nvm_source_outcomes_and_clock_policy_compose_through_packet_stop() {
    use super::super::mtp;
    for (family, device) in [(mtp::Family::Mtp, 0x550a), (mtp::Family::Ptp, 0x57b3)] {
        for ([marker, check, expected, _], _) in mtp::tests::cases(family, 9) {
            for disable in [false, true] {
                let (state, released, io, timer) = spt::tests::fixture_spt();
                {
                    let mut m = state.borrow_mut();
                    m.words[3] = (m.words[3] & !1) | marker as u16;
                    let sum = m.words[..63].iter().copied().fold(0u16, u16::wrapping_add);
                    m.words[63] = 0xbabau16.wrapping_sub(sum);
                    if check == 1 {
                        m.words[63] ^= 1;
                    }
                    if check == 2 {
                        m.words[63] = 0xffff;
                    }
                }
                // SAFETY: same-host modeled identity and Function grant.
                let result =
                    unsafe { Controller::new_mtp(device, io, timer, mtp::tests::host(disable)) };
                if expected != 0 {
                    assert_eq!(expected, 1);
                    assert!(matches!(result, Err(Error::Nvm(nvm::Error::Checksum(_)))));
                    assert_eq!(released.load(Ordering::SeqCst), 0);
                    continue;
                }
                let controller = result.unwrap();
                {
                    let mut m = state.borrow_mut();
                    m.mac
                        .insert(super::super::STATUS, LINK_UP | FULL_DUPLEX | (1 << 7));
                    m.phy.insert((2, 0, 1), PHY_LINK_READY);
                }
                let mut memory = Memory::new();
                let mut net = open(controller, &mut memory);
                net.transmit(&[0x71; 60]).unwrap();
                net.transmit.descriptors.write(12, TX_DONE);
                inject(&mut net, 0, RX_DONE | RX_EOP, 64, 0x71);
                assert_eq!(net.receive().unwrap(), Some(vec![0x71; 60]));
                net.shutdown().unwrap();
                drop(net);
                assert_eq!(released.load(Ordering::SeqCst), 1);
            }
        }
        for ([initial, _, _, _], expected) in mtp::tests::cases(family, 8) {
            let (state, _, controller) = setup_integrated(device, 0);
            let mut memory = Memory::new();
            let mut net = open(controller, &mut memory);
            state.borrow_mut().mac.insert(FFLT_DBG, initial);
            state.borrow_mut().operations.clear();
            net.i219_clock_policy().unwrap();
            assert_eq!(state.borrow().operations, expected);
            net.shutdown().unwrap();
        }
    }
}
#[test]
fn every_mtp_ptp_cold_k1_queue_link_and_flushing_stop_access_fault_keeps_owner() {
    use super::super::mtp;
    for device in [0x550a, 0x57b3] {
        for disable in [false, true] {
            let (state, _, io, timer) = spt::tests::fixture_spt();
            // SAFETY: one same-host model and source-selected family.
            let controller =
                unsafe { Controller::new_mtp(device, io, timer, mtp::tests::host(disable)) }
                    .unwrap();
            let cold = state.borrow().accesses;
            {
                let mut m = state.borrow_mut();
                m.mac
                    .insert(super::super::STATUS, LINK_UP | FULL_DUPLEX | (1 << 7));
                m.phy.insert((2, 0, 1), PHY_LINK_READY);
            }
            let mut memory = Memory::new();
            let mut net = open(controller, &mut memory);
            let configured = state.borrow().accesses;
            state
                .borrow_mut()
                .mac
                .insert(super::super::STATUS, LINK_UP | FULL_DUPLEX | (1 << 6));
            assert!(net.carrier());
            let linked = state.borrow().accesses;
            {
                let mut m = state.borrow_mut();
                m.config.insert(0xe4, 0x100);
                m.flush_after_rx = true;
            }
            net.shutdown().unwrap();
            let last = state.borrow().accesses;
            drop(net);
            for at in 0..last {
                let (state, released, io, timer) = spt::tests::fixture_spt();
                let mut memory = Memory::new();
                if at < cold {
                    state.borrow_mut().fail_at = Some(at);
                    assert!(
                        // SAFETY: fresh same-host model; failure never retried as a new owner.
                        unsafe {
                            Controller::new_mtp(device, io, timer, mtp::tests::host(disable))
                        }
                        .is_err(),
                        "cold{device:x}/{disable}/{at}"
                    );
                } else {
                    // SAFETY: fresh same-host source-selected model.
                    let controller = unsafe {
                        Controller::new_mtp(device, io, timer, mtp::tests::host(disable))
                    }
                    .unwrap();
                    {
                        let mut m = state.borrow_mut();
                        m.mac
                            .insert(super::super::STATUS, LINK_UP | FULL_DUPLEX | (1 << 7));
                        m.phy.insert((2, 0, 1), PHY_LINK_READY);
                    }
                    if at < configured {
                        state.borrow_mut().fail_at = Some(at);
                        assert!(
                            // SAFETY: live model requester and owned coherent model RAM.
                            unsafe { Ethernet::new(controller, &mut memory) }.is_err(),
                            "queue{device:x}/{disable}/{at}"
                        );
                    } else {
                        let mut net = open(controller, &mut memory);
                        state
                            .borrow_mut()
                            .mac
                            .insert(super::super::STATUS, LINK_UP | FULL_DUPLEX | (1 << 6));
                        if at < linked {
                            state.borrow_mut().fail_at = Some(at);
                            assert!(!net.carrier(), "link{device:x}/{disable}/{at}");
                        } else {
                            assert!(net.carrier());
                            {
                                let mut m = state.borrow_mut();
                                m.config.insert(0xe4, 0x100);
                                m.flush_after_rx = true;
                                m.fail_at = Some(at);
                            }
                            assert!(net.shutdown().is_err(), "stop{device:x}/{disable}/{at}");
                        }
                        let count = state.borrow().accesses;
                        assert!(net.transmit(&[0; 60]).is_err());
                        assert!(!net.carrier());
                        drop(net);
                        assert_eq!(state.borrow().accesses, count);
                    }
                }
                assert_eq!(
                    released.load(Ordering::SeqCst),
                    0,
                    "released{device:x}/{disable}/{at}"
                );
            }
        }
    }
}
#[test]
fn mtp_flush_success_does_not_replace_vendor_drain_and_unresolved_flush_never_releases() {
    for device in [0x550a, 0x57b3] {
        for unresolved in [false, true] {
            let (state, released, controller) = setup_integrated(device, 0);
            let mut memory = Memory::new();
            let mut net = open(controller, &mut memory);
            {
                let mut m = state.borrow_mut();
                m.config.insert(0xe4, 0x100);
                m.flush_after_tx = !unresolved;
                m.master_busy = 2000;
            }
            assert_eq!(
                net.shutdown(),
                Err(if unresolved {
                    Error::DescriptorFlushRequired
                } else {
                    Error::MasterTimeout
                })
            );
            if !unresolved {
                assert_eq!(state.borrow().config[&0xe4] & 0x100, 0);
            }
            let accesses = state.borrow().accesses;
            assert!(net.transmit(&[0; 60]).is_err());
            assert!(!net.carrier());
            drop(net);
            assert_eq!(state.borrow().accesses, accesses);
            assert_eq!(released.load(Ordering::SeqCst), 0);
        }
    }
}

#[test]
fn lost_minimum_is_typed_through_link_and_cannot_resume_device_effects() {
    let (state, released, controller) = setup();
    let mut memory = Memory::new();
    let mut net = open(controller, &mut memory);
    let cause = crate::bus::pci::power::Error::Clock;
    state.borrow_mut().timing_failure = Some(cause);
    let observed = state.borrow().operations.clone();
    assert!(!net.carrier());
    let expected = LinkError::Pci(crate::block::BlockError::PciPower(cause));
    assert_eq!(net.failure().and_then(Error::minimum_cause), Some(cause));
    state.borrow_mut().timing_failure = None;
    for _ in 0..3 {
        assert_eq!(net.transmit(&[1; 60]), Err(expected.clone()));
        assert_eq!(net.receive(), Err(expected.clone()));
    }
    assert_eq!(state.borrow().operations, observed);
    drop(net);
    assert_eq!(released.load(Ordering::SeqCst), 0);
    // NVM's unused clone may drop; PHY keeps the actual shared provider alive.
    assert_eq!(state.borrow().timer_drops.load(Ordering::SeqCst), 1);
}
