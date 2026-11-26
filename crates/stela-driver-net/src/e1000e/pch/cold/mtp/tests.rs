use super::super::{
    spt,
    tests::{Io, Timer},
};
use super::*;
use alloc::vec::Vec;
use core::sync::atomic::Ordering;
pub(in crate::e1000e::pch::cold) fn host(disable: bool) -> HostIdentity<'static> {
    if disable {
        HostIdentity {
            system_vendor: Observation::Value(b"Dell Inc."),
            product_name: Observation::Value(b"Dell Pro 16 Plus PB16250"),
            product_version: Observation::Absent,
        }
    } else {
        HostIdentity {
            system_vendor: Observation::Value(b"Other"),
            ..HostIdentity::default()
        }
    }
}
pub(in crate::e1000e::pch::cold) unsafe fn claim_any(
    device: u16,
    io: Io,
    timer: Timer,
) -> Result<Controller<Io, Timer>, Error> {
    if Family::from_device(device).is_some() {
        // SAFETY: the caller owns the one synchronous same-host model.
        unsafe { Controller::claim_mtp(device, io, timer, host(false)) }
    } else {
        // SAFETY: the caller owns the one synchronous same-host model.
        unsafe { Controller::claim_i219(device, io, timer) }
    }
}
pub(in crate::e1000e::pch::cold) fn start_any(
    device: u16,
    io: Io,
    timer: Timer,
) -> Result<Controller<Io, Timer>, Error> {
    if Family::from_device(device).is_some() {
        // SAFETY: fresh exact physical-host/Function model epoch.
        unsafe { Controller::new_mtp(device, io, timer, host(false)) }
    } else {
        super::super::i219::tests::start(device, io, timer)
    }
}
pub(in crate::e1000e::pch::cold) fn cases(
    family: Family,
    mode: u32,
) -> Vec<([u32; 4], Vec<[u32; 3]>)> {
    super::super::lpt::tests::oracle_cases_for(
        mode,
        match family {
            Family::Mtp => "STELA_E1000E_PCH_MTP_ORACLE",
            Family::Ptp => "STELA_E1000E_PCH_PTP_ORACLE",
        },
    )
}
#[test]
fn unknown_board_identity_refuses_before_config_and_observed_absence_is_not_unknown() {
    for id in [0x550a, 0x57b3] {
        for h in [
            HostIdentity::default(),
            HostIdentity {
                system_vendor: Observation::Value(b"Dell Inc."),
                ..HostIdentity::default()
            },
            HostIdentity {
                system_vendor: Observation::Value(b"LENOVO"),
                ..HostIdentity::default()
            },
        ] {
            let (state, released, io, timer) = spt::tests::fixture_spt();
            assert!(matches!(
                // SAFETY: modeled identity observation, incomplete.
                unsafe { Controller::new_mtp(id, io, timer, h) },
                Err(Error::HostIdentityNotObserved)
            ));
            assert!(state.borrow().operations.is_empty());
            assert_eq!(released.load(Ordering::SeqCst), 1);
        }
    }
    assert_eq!(
        HostIdentity {
            system_vendor: Observation::Absent,
            ..HostIdentity::default()
        }
        .decision(),
        Decision::NoMatch
    );
    assert_eq!(
        HostIdentity {
            system_vendor: Observation::Value(b"Dell Inc."),
            product_name: Observation::Absent,
            ..HostIdentity::default()
        }
        .decision(),
        Decision::NoMatch
    );
    assert_eq!(
        HostIdentity {
            system_vendor: Observation::NotObserved,
            product_name: Observation::Absent,
            product_version: Observation::Absent
        }
        .decision(),
        Decision::NotObserved
    );
    assert_eq!(
        HostIdentity {
            system_vendor: Observation::Value(b"Dell\0 Inc."),
            ..HostIdentity::default()
        }
        .decision(),
        Decision::NoMatch
    );
}
#[test]
#[ignore = "oracle: STELA_E1000E_PCH_MTP_DMI_ORACLE"]
fn unchanged_dmi_matches_and_quirk_table_preserve_byte_string_semantics() {
    const V: [Option<&[u8]>; 8] = [
        None,
        Some(b""),
        Some(b"Dell Inc."),
        Some(b"LENOVO"),
        Some(b"prefixDell Inc.suffix"),
        Some(b"lenovo"),
        Some(b"Dell\0 Inc."),
        Some(b"\xffLENOVO"),
    ];
    const N: [Option<&[u8]>; 8] = [
        None,
        Some(b""),
        Some(b"Dell Pro 16 Plus PB16250"),
        Some(b"prefix Dell Pro 16 Plus PB16250 suffix"),
        Some(b"dell Pro 16 Plus PB16250"),
        Some(b"Dell Pro 16 Plus\0 PB16250"),
        Some(b"\xffDell Pro 16 Plus PB16250"),
        Some(b"ThinkPad P14s Gen 5"),
    ];
    const P: [Option<&[u8]>; 8] = [
        None,
        Some(b""),
        Some(b"ThinkPad P14s Gen 5"),
        Some(b"prefixThinkPad P14s Gen 5suffix"),
        Some(b"thinkPad P14s Gen 5"),
        Some(b"ThinkPad\0 P14s Gen 5"),
        Some(b"\xfeThinkPad P14s Gen 5"),
        Some(b"Dell Pro 16 Plus PB16250"),
    ];
    let rows = super::super::lpt::tests::oracle_cases_for(14, "STELA_E1000E_PCH_MTP_DMI_ORACLE");
    assert_eq!(rows.len(), 512);
    for ([v, n, p, expected], ops) in rows {
        assert!(ops.is_empty());
        let observed = |v: Option<&'static [u8]>| v.map_or(Observation::Absent, Observation::Value);
        let h = HostIdentity {
            system_vendor: observed(V[v as usize]),
            product_name: observed(N[n as usize]),
            product_version: observed(P[p as usize]),
        };
        assert_eq!(
            h.decision(),
            if expected == 0 {
                Decision::NoMatch
            } else {
                Decision::Match
            },
            "{v}/{n}/{p}"
        );
    }
}
#[test]
#[ignore = "oracle: STELA_E1000E_PCH_MTP_ORACLE, STELA_E1000E_PCH_PTP_ORACLE"]
fn exact_remaining_fourteen_ids_match_source_table_without_broadening_old_constructor() {
    let mut seen = Vec::new();
    for (family, code) in [(Family::Mtp, 4), (Family::Ptp, 5)] {
        for ([id, actual, _, _], ops) in cases(family, 10) {
            assert!(ops.is_empty());
            assert_eq!(actual, code);
            assert_eq!(Family::from_device(id as u16), Some(family));
            seen.push(id as u16);
        }
    }
    assert_eq!(seen.len(), 14);
    for id in 0..=u16::MAX {
        assert_eq!(Family::from_device(id).is_some(), seen.contains(&id));
    }
    for id in seen {
        let (state, _, io, timer) = spt::tests::fixture_spt();
        assert!(matches!(
            super::super::i219::tests::start(id, io, timer),
            Err(Error::UnsupportedDevice(_))
        ));
        assert!(state.borrow().operations.is_empty());
    }
}
#[test]
#[ignore = "oracle: STELA_E1000E_PCH_MTP_ORACLE, STELA_E1000E_PCH_PTP_ORACLE"]
fn unchanged_mtp_ptp_reset_post_ulp_ltr_shra_eee_link_flush_and_k1() {
    for (family, id) in [(Family::Mtp, 0x550a), (Family::Ptp, 0x57b3)] {
        super::super::lpt::tests::replay_post_cases(Some(id), cases(family, 0));
        super::super::lpt::tests::replay_ltr_cases(Some(id), cases(family, 1));
        super::super::lpt::tests::replay_reset_cases(Some(id), cases(family, 4));
        spt::tests::replay_ulp(id, cases(family, 5));
        super::super::queue::tests::replay_shared_cases(Some(id), cases(family, 2));
        super::super::queue::tests::replay_eee_cases(Some(id), cases(family, 3));
        super::super::queue::tests::replay_i219_link(id, cases(family, 6));
        super::super::queue::tests::replay_i219_flush(id, cases(family, 7));
        let rows = cases(family, 11);
        assert_eq!(rows.len(), 54);
        for ([fext, disable, kmrn, timeouts], expected) in rows {
            let (state, _, io, timer) = spt::tests::fixture_spt();
            // SAFETY: one model and complete board observation.
            let mut c =
                unsafe { Controller::claim_mtp(id, io, timer, host(disable != 0)) }.unwrap();
            {
                let mut m = state.borrow_mut();
                m.mac.insert(0x5bc0, fext);
                m.mac.insert(0x18, 0x1234a5a5);
                m.mac.insert(0, 0x98765b21);
                m.kmrn.insert(7, kmrn as u16);
                m.phy.insert((1, 770, 21), timeouts as u16);
                m.operations.clear();
            }
            c.mtp_init_k1().unwrap();
            assert_eq!(
                spt::tests::source_effects(&state.borrow().operations, 11),
                expected,
                "{family:?}/{disable}/{fext:x}/{kmrn:x}/{timeouts:x}"
            );
        }
    }
}
#[test]
#[ignore = "oracle: STELA_E1000E_PCH_MTP_ORACLE, STELA_E1000E_PCH_PTP_ORACLE"]
fn unchanged_reset_dpg_tail_is_ptp_only_and_preserves_other_bits() {
    for (family, id) in [(Family::Mtp, 0x550a), (Family::Ptp, 0x57b3)] {
        let rows = cases(family, 12);
        assert_eq!(rows.len(), 3);
        for ([ext, _, _, _], expected) in rows {
            let expected: Vec<_> = expected
                .into_iter()
                .filter(|o| o[0] <= 1 && o[1] == 0x18)
                .collect();
            let (state, _, io, timer) = spt::tests::fixture_spt();
            // SAFETY: same-host cold model; no reset wrapped in PHY transaction.
            let mut c = unsafe { Controller::claim_mtp(id, io, timer, host(false)) }.unwrap();
            state.borrow_mut().mac.insert(0x18, ext);
            state.borrow_mut().operations.clear();
            c.mtp_reset_finish().unwrap();
            assert_eq!(state.borrow().operations, expected);
            assert_eq!(
                state.borrow().mac[&0x18],
                if family == Family::Ptp { ext & !8 } else { ext }
            );
        }
    }
}
#[test]
fn retries_never_swallow_access_removal_or_uncertain_owner_cleanup() {
    for fault in [Fault::Access, Fault::Removed, Fault::OutsideBar] {
        for write in [false, true] {
            let (state, released, io, timer) = spt::tests::fixture_spt();
            // SAFETY: same-host model with exact retry-enabled family.
            let mut c = unsafe { Controller::claim_mtp(0x550a, io, timer, host(false)) }.unwrap();
            let before = state.borrow().accesses;
            c.phy.acquire().unwrap();
            let acquired = state.borrow().accesses - before;
            c.phy.release().unwrap();
            let at = state.borrow().accesses + acquired + 1;
            {
                let mut m = state.borrow_mut();
                m.operations.clear();
                m.fail_at = Some(at);
                m.fail_with = fault;
            }
            let error = c
                .phy
                .transaction(|p| p.mdic(1, 21, write.then_some(0x5a5a)))
                .unwrap_err();
            assert_eq!(error.primary, fault);
            assert_eq!(error.cleanup, None);
            assert_eq!(
                state
                    .borrow()
                    .operations
                    .iter()
                    .filter(|o| o[0] == 1 && o[1] == 0x20)
                    .count(),
                1
            );
            assert!(!state.borrow().operations.contains(&[2, 10_000, 0]));
            drop(c);
            assert_eq!(released.load(Ordering::SeqCst), 0);
        }
    }
}
#[test]
fn cold_retry_success_and_firmware_phy_nvm_power_refusals_are_distinct() {
    for device in [0x550a, 0x57b3] {
        for kind in 0..3 {
            let (state, released, io, timer) = spt::tests::fixture_spt();
            {
                let mut m = state.borrow_mut();
                m.mdic_failures = 2;
                m.mdic_kind = kind;
            }
            let c = start_any(device, io, timer).unwrap();
            {
                let m = state.borrow();
                let completed = m
                    .operations
                    .iter()
                    .position(|o| {
                        o[0] == 0
                            && o[1] == 0x20
                            && o[2] & (1 << 28) != 0
                            && o[2] & (1 << 30) == 0
                            && (o[2] >> 16) & 31 == 2
                    })
                    .unwrap();
                assert_eq!(
                    m.operations[..completed]
                        .iter()
                        .filter(|o| **o == [2, 10_000, 0])
                        .count(),
                    2
                );
                assert_eq!(
                    m.operations[..completed]
                        .iter()
                        .filter(|o| o[0] == 1 && o[1] == 0x20)
                        .count(),
                    3
                );
            }
            drop(c);
            assert_eq!(released.load(Ordering::SeqCst), 1);
        }
        for failure in 0..8 {
            let (state, released, io, timer) = spt::tests::fixture_spt();
            {
                let mut m = state.borrow_mut();
                match failure {
                    0 => {
                        m.words[19] = 0xffff;
                        m.words[1043] = 0xffff;
                    }
                    1 => {
                        m.words[..3].fill(0);
                        let sum = m.words[..63].iter().copied().fold(0u16, u16::wrapping_add);
                        m.words[63] = 0xbabau16.wrapping_sub(sum);
                    }
                    2 => {
                        m.phy.insert((2, 0, 3), 0x93);
                    }
                    3 => {
                        m.mac.insert(
                            super::super::super::FWSM,
                            super::super::super::FW_VALID | super::super::super::RESET_PERMITTED,
                        );
                    }
                    4 => {
                        m.smb_stuck = true;
                        m.phy.insert((1, 769, 23), 0x7f01);
                    }
                    5 => m.ulp_stuck = true,
                    6 => {
                        m.mac.insert(0x3808, 512);
                        m.config.insert(0xe4, 0x100);
                    }
                    7 => m.reset_stuck = true,
                    _ => unreachable!(),
                }
            }
            assert!(
                start_any(device, io, timer).is_err(),
                "accepted{device:x}/{failure}"
            );
            assert_eq!(released.load(Ordering::SeqCst), 0);
            assert!(
                !state
                    .borrow()
                    .operations
                    .iter()
                    .any(|o| o[0] == 1 && matches!(o[1], 0x100 | 0x400) && o[2] & 2 != 0)
            );
        }
    }
    // Existing constructor cohorts never gain MTP's retry policy by PHY ID.
    let (state, released, io, timer) = spt::tests::fixture_spt();
    state.borrow_mut().mdic_failures = 1;
    assert!(super::super::i219::tests::start(0x156f, io, timer).is_err());
    assert_eq!(state.borrow().mdic_attempts, 1);
    assert_eq!(released.load(Ordering::SeqCst), 0);
}

#[test]
fn both_cohorts_can_leave_ulp_without_a_power_cycle_completion_bit() {
    for device in [0x550a, 0x57b3] {
        let (state, released, io, timer) = spt::tests::fixture_spt();
        // Withhold the status indication while the PHY remains accessible.
        state.borrow_mut().power_stuck = true;
        let controller = start_any(device, io, timer).unwrap();
        assert!(!controller.is_poisoned());
        drop(controller);
        assert_eq!(released.load(Ordering::SeqCst), 1);
    }
}

#[test]
#[ignore = "oracle: STELA_E1000E_PCH_MTP_ORACLE, STELA_E1000E_PCH_PTP_ORACLE"]
fn unchanged_mdic_protocol_retry_limit_delays_and_both_directions() {
    for (family, id) in [(Family::Mtp, 0x550a), (Family::Ptp, 0x57b3)] {
        let rows = cases(family, 13);
        assert_eq!(rows.len(), 48);
        for ([write, enabled, failures, kind], expected) in rows {
            let (state, _, io, timer) = spt::tests::fixture_spt();
            // SAFETY: isolated source-selected model owner.
            let mut c = unsafe { Controller::claim_mtp(id, io, timer, host(false)) }.unwrap();
            c.phy.retry_enabled = enabled != 0;
            {
                let mut m = state.borrow_mut();
                m.operations.clear();
                m.mdic_attempts = 0;
                m.mdic_failures = failures as usize;
                m.mdic_kind = (kind >> 16) as u8;
            }
            let result = c.phy.mdic(1, 21, (write != 0).then_some(0x5a5a));
            assert_eq!(result.is_err(), kind & 1 != 0);
            let actual: Vec<_> = state
                .borrow()
                .operations
                .iter()
                .copied()
                .filter(|o| o[0] == 2 || o[1] == 0x20)
                .collect();
            assert_eq!(
                actual, expected,
                "{family:?} write{write} enabled{enabled} failures{failures} kind{kind:x}"
            );
        }
    }
}
