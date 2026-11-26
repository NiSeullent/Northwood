use super::super::tests::{Io, Model, Timer, fixture_lpt};
use super::*;
use alloc::{rc::Rc, sync::Arc};
use core::{
    cell::RefCell,
    sync::atomic::{AtomicUsize, Ordering},
};

pub(in crate::e1000e::pch::cold) fn fixture_spt()
-> (Rc<RefCell<Model>>, Arc<AtomicUsize>, Io, Timer) {
    let (state, released, io, timer) = fixture_lpt();
    {
        let mut m = state.borrow_mut();
        m.spt = true;
        m.phy.insert((1, 779, 16), 0xffff);
        m.phy.insert((1, 769, 23), 0x7f00);
        m.mac.insert(FEXTNVM7, 0xaaaa_003f);
    }
    (state, released, io, timer)
}
pub(in crate::e1000e::pch::cold) fn start_spt(
    device: u16,
    io: Io,
    timer: Timer,
) -> Result<Controller<Io, Timer>, Error> {
    // SAFETY: one synchronous modeled Function/config/PHY/flash.
    unsafe { Controller::new_spt(device, io, timer) }
}
pub(in crate::e1000e::pch::cold) fn oracle_cases(
    mode: u32,
) -> alloc::vec::Vec<([u32; 4], alloc::vec::Vec<[u32; 3]>)> {
    super::super::lpt::tests::oracle_cases_for(mode, "STELA_E1000E_PCH_SPT_ORACLE")
}
#[test]
#[ignore = "oracle: STELA_E1000E_PCH_SPT_ORACLE"]
fn unchanged_linux_spt_post_reset_ltr_and_global_reset_effects() {
    super::super::lpt::tests::replay_post(true);
    super::super::lpt::tests::replay_ltr(true);
    super::super::lpt::tests::replay_reset(true);
}

pub(in crate::e1000e::pch::cold) fn source_effects(
    operations: &[[u32; 3]],
    mode: u32,
) -> alloc::vec::Vec<[u32; 3]> {
    let mut result = alloc::vec::Vec::new();
    let mut page = 0;
    let mut pending = None;
    for &[kind, at, value] in operations {
        if kind == 2
            && ((mode == 5 && (at == 10 || at >= 5000))
                || (mode == 11 && matches!(at, 2 | 20 | 1000)))
        {
            result.push([kind, at, value]);
        }
        let mac = match mode {
            5 => matches!(at, 0 | 8 | 0x18 | 0x3c | 0xe4),
            11 => matches!(at, 0 | 8 | 0x18 | 0x34 | 0x5bc0),
            6 => matches!(at, 0x410 | 0x24 | 0x10 | 0xf18 | 0x1000 | 0xf8),
            _ => false,
        };
        if kind <= 1 && mac {
            result.push([kind, at, value]);
        }
        if at != super::super::super::MDIC {
            continue;
        }
        if kind == 1 {
            let index = (value >> 16) & 31;
            if index == 31 {
                page = (value & 0xffff) >> 5;
                continue;
            }
            let address = (value >> 21) & 31;
            let logical = if index <= 15 {
                index
            } else {
                ((if address == 1 && page == 0 { 768 } else { page }) << 5) | index
            };
            if value & super::super::super::MDIC_WRITE != 0 {
                result.push([11, logical, value & 0xffff]);
            } else {
                pending = Some(logical);
            }
        } else if kind == 0
            && let Some(logical) = pending.take()
        {
            result.push(if value & (1 << 30) != 0 {
                [12, logical, 0]
            } else {
                [10, logical, value & 0xffff]
            });
        }
    }
    if mode == 6 {
        // Stela reasserts the previously source-compared EEE advertisement
        // disable policy on each link change. The source check-link helper
        // only runs its LPI disable operation; advertisement was set at reset.
        let mut filtered = alloc::vec::Vec::new();
        let mut at = 0;
        while at < result.len() {
            if result[at] == [11, 16, 0x8001] && result.get(at + 1) == Some(&[11, 17, 0]) {
                at += 2;
            } else {
                filtered.push(result[at]);
                at += 1;
            }
        }
        result = filtered;
    }
    result
}
#[test]
#[ignore = "oracle: STELA_E1000E_PCH_SPT_ORACLE"]
fn unchanged_linux_spt_forced_ulp_exit_ordered_effects() {
    replay_ulp(0x156f, oracle_cases(5));
}
pub(in crate::e1000e::pch::cold) fn replay_ulp(
    device: u16,
    cases: alloc::vec::Vec<([u32; 4], alloc::vec::Vec<[u32; 3]>)>,
) {
    assert_eq!(cases.len(), 36);
    for ([busy, hidden, force, ulp], expected) in cases {
        let (state, _, io, timer) = fixture_spt();
        // SAFETY: same-function model with integrated flash geometry.
        let mut controller =
            unsafe { super::super::mtp::tests::claim_any(device, io, timer) }.unwrap();
        {
            let mut m = state.borrow_mut();
            m.power_busy = busy as usize;
            m.smb_hidden_after_power = hidden != 0;
            m.mac.insert(CTRL_EXT, 0x1000);
            m.mac.insert(super::super::FEXTNVM3, 0x5abc1234);
            m.mac.insert(FEXTNVM7, 0xaaaa00ff);
            m.phy.insert((1, 769, 23), 0xa500 | force as u16);
            m.phy.insert((1, 770, 17), 0x9332);
            m.phy.insert((1, 779, 16), ulp as u16);
            m.operations.clear();
        }
        controller.spt_ulp_exit().unwrap();
        let actual = source_effects(&state.borrow().operations, 5);
        assert_eq!(
            actual, expected,
            "busy={busy} hidden={hidden} force={force} ulp={ulp:x}"
        );
    }
}
#[test]
fn exact_initial_i219_cohort_uses_integrated_nvm_and_forced_ulp_exit() {
    for device in [0x156f, 0x1570] {
        let (state, released, io, timer) = fixture_spt();
        let controller = start_spt(device, io, timer).unwrap();
        assert_eq!(controller.identity().id, 0x0154_00a0);
        assert_eq!(controller.station().0, [2, 0x11, 0x22, 0x33, 0x44, 0x55]);
        assert_eq!(controller.nvm.geometry().bank_words, 1024);
        let m = state.borrow();
        assert_eq!(m.phy[&(1, 779, 16)] & ULP_CLEAR, 0);
        assert_ne!(m.phy[&(1, 770, 17)] & K1_ENABLE, 0);
        assert_eq!(m.mac[&FEXTNVM7] & DISABLE_SMB_PERST, 0);
        assert_ne!(m.mac[&FEXTNVM11] & DISABLE_MULR_FIX, 0);
        assert!(
            m.operations
                .iter()
                .any(|o| o[0] == 6 && o[1] == 4 && o[2] >> 16 != 0)
        );
        assert!(!m.operations.iter().any(|o| o[0] == 4 && o[1] == 0));
        assert!(!m.operations.iter().any(|o| o[0] == 5 && o[1] == 6));
        let power = m
            .operations
            .iter()
            .position(|o| o[0] == 1 && o[1] == super::super::CTRL && o[2] & LANPHYPC_OVERRIDE != 0)
            .unwrap();
        assert_eq!(m.operations[power + 1][..2], [0, super::super::STATUS]);
        assert_eq!(m.operations[power + 2], [2, 10, 0]);
        let reset = m
            .operations
            .iter()
            .position(|o| {
                o[0] == 1 && o[1] == super::super::CTRL && o[2] & super::super::RESET != 0
            })
            .unwrap();
        assert_eq!(m.operations[reset + 1], [2, 20_000, 0]);
        drop(m);
        drop(controller);
        assert_eq!(released.load(Ordering::SeqCst), 1);
    }
    for device in [
        0x1502, 0x153a, 0x1559, 0x15b7, 0x15b8, 0x15b9, 0x15d7, 0x15e3, 0x15bd,
    ] {
        let (state, released, io, timer) = fixture_spt();
        assert!(
            matches!(start_spt(device,io,timer),Err(Error::UnsupportedDevice(id))if id==device)
        );
        assert!(state.borrow().operations.is_empty());
        assert_eq!(released.load(Ordering::SeqCst), 1);
    }
}
#[test]
fn foreign_required_flush_and_live_config_failures_refuse_without_foreign_dma() {
    for failure in 0..3 {
        let (state, released, io, timer) = fixture_spt();
        {
            let mut m = state.borrow_mut();
            m.mac.insert(0x3808, 512);
            match failure {
                0 => {
                    m.config.insert(DESCRIPTOR_STATUS, FLUSH_REQUIRED);
                }
                1 => {
                    m.config.insert(DESCRIPTOR_STATUS, 0xffff);
                }
                2 => {
                    m.fail_at = Some(5);
                }
                _ => unreachable!(),
            }
        }
        assert!(start_spt(0x156f, io, timer).is_err());
        let m = state.borrow();
        assert!(
            !m.operations
                .iter()
                .any(|o| o[0] == 1 && matches!(o[1], super::super::CTRL | 0x3818))
        );
        assert_eq!(released.load(Ordering::SeqCst), 0);
    }
}
#[test]
fn ulp_exit_clears_settings_even_without_the_power_cycle_status() {
    let (state, _, io, timer) = fixture_spt();
    state.borrow_mut().power_stuck = true;
    let controller = start_spt(0x156f, io, timer).unwrap();
    let model = state.borrow();
    assert_eq!(model.phy[&(1, 779, 16)] & ULP_CLEAR, 0);
    assert_eq!(model.mac[&CTRL_EXT] & FORCE_SMBUS, 0);
    assert_eq!(model.mac[&FEXTNVM7] & DISABLE_SMB_PERST, 0);
    drop(model);
    drop(controller);
}

#[test]
fn ulp_mode_switch_recovers_only_the_specific_mdi_transition_and_verifies_state() {
    let (state, released, io, timer) = fixture_spt();
    state.borrow_mut().smb_hidden_after_power = true;
    let controller = start_spt(0x156f, io, timer).unwrap();
    assert_eq!(state.borrow().phy[&(1, 769, 23)] & 1, 0);
    assert_eq!(state.borrow().mac[&CTRL_EXT] & FORCE_SMBUS, 0);
    assert!(
        state
            .borrow()
            .operations
            .iter()
            .any(|o| o[0] == 1 && o[1] == CTRL_EXT && o[2] & FORCE_SMBUS != 0)
    );
    assert!(
        state
            .borrow()
            .operations
            .iter()
            .filter(|o| **o == [2, 50_000, 0])
            .count()
            >= 2
    );
    drop(controller);
    assert_eq!(released.load(Ordering::SeqCst), 1);
    for stuck in [false, true] {
        let (state, released, io, timer) = fixture_spt();
        if stuck {
            let mut m = state.borrow_mut();
            m.smb_hidden_after_power = true;
            m.smb_stuck = true;
        } else {
            state.borrow_mut().ulp_stuck = true;
        }
        assert!(matches!(
            start_spt(0x1570, io, timer),
            Err(Error::SmbusMode | Error::UlpState)
        ));
        assert_eq!(released.load(Ordering::SeqCst), 0);
    }
    let expected = (1 << 28) | (1 << 30) | (23 << 16) | (1 << 21);
    assert!(smbus_response_error(Fault::MdicResponse(expected)));
    for bad in [
        expected ^ (1 << 30),
        expected ^ (1 << 28),
        expected ^ (1 << 16),
        expected ^ (1 << 21),
    ] {
        assert!(!smbus_response_error(Fault::MdicResponse(bad)));
    }
    for bad in [
        Fault::Access,
        Fault::Removed,
        Fault::OutsideBar,
        Fault::ResetBlocked,
    ] {
        assert!(!smbus_read_unavailable(bad));
    }
}
#[test]
fn each_mode_switch_access_failure_is_terminal_and_keeps_owner() {
    let (state, _, io, timer) = fixture_spt();
    state.borrow_mut().smb_hidden_after_power = true;
    let controller = start_spt(0x156f, io, timer).unwrap();
    let count = state.borrow().accesses;
    drop(controller);
    for at in 0..count {
        let (state, released, io, timer) = fixture_spt();
        let mut m = state.borrow_mut();
        m.smb_hidden_after_power = true;
        m.fail_at = Some(at);
        drop(m);
        assert!(
            start_spt(0x156f, io, timer).is_err(),
            "lost mode-switch fault{at}"
        );
        assert_eq!(released.load(Ordering::SeqCst), 0);
    }
}
#[test]
fn every_spt_cold_access_failure_keeps_owner_and_prevents_packet_dma() {
    let (state, _, io, timer) = fixture_spt();
    let controller = start_spt(0x1570, io, timer).unwrap();
    let count = state.borrow().accesses;
    drop(controller);
    for at in 0..count {
        let (state, released, io, timer) = fixture_spt();
        state.borrow_mut().fail_at = Some(at);
        assert!(
            start_spt(0x1570, io, timer).is_err(),
            "lost SPT cold access{at}"
        );
        assert_eq!(released.load(Ordering::SeqCst), 0, "released SPT owner{at}");
        assert!(!state.borrow().operations.iter().any(|o| o[0] == 1
            && matches!(o[1], super::super::RCTL | super::super::TCTL)
            && o[2] & 2 != 0));
    }
}
#[test]
fn spt_firmware_reset_phy_nvm_and_readiness_failures_never_enable_packet_dma() {
    for failure in 0..7 {
        let (state, released, io, timer) = fixture_spt();
        {
            let mut m = state.borrow_mut();
            match failure {
                0 => {
                    m.mac.insert(
                        super::super::super::FWSM,
                        super::super::super::FW_VALID | super::super::super::RESET_PERMITTED,
                    );
                }
                1 => {
                    m.mac.insert(super::super::super::FWSM, 0);
                }
                2 => {
                    m.phy.insert((2, 0, 3), 0x93);
                }
                3 => {
                    m.words[63] ^= 1;
                }
                4 => m.init_missing = true,
                5 => m.reset_missing = true,
                6 => m.reset_stuck = true,
                _ => unreachable!(),
            }
        }
        assert!(start_spt(0x156f, io, timer).is_err());
        assert_eq!(released.load(Ordering::SeqCst), 0);
        assert!(!state.borrow().operations.iter().any(|o| o[0] == 1
            && matches!(o[1], super::super::RCTL | super::super::TCTL)
            && o[2] & 2 != 0));
    }
}
