// SPDX-License-Identifier: GPL-2.0-only
// Copyright (c) 2026 Stela contributors
//! Nbase admission cases.
use super::*;

#[test]
fn nbase_aperture_irq_width_and_doorbell_are_checked_before_extended_io() {
    // The shared map_registers boundary checks both BAR size and
    // mapping before returning MMIO. The constructor requests the
    // complete range through the 16-bit timer at 6048 before reading it.
    assert_eq!(register_aperture(revision(61)), 0x604a);
    assert_eq!(register_aperture(revision(70)), 0x604a);
    assert_eq!(register_aperture(revision(34)), 256);
    for version in [61, 63, 64, 65, 66, 70] {
        let time = Time::default();
        let model = Model::nbase(&time);
        let hw = Hardware::new(&model, &time, revision(version), 0);
        hw.reset().unwrap();
        kick_transmit(&model, revision(version));
        let s = model.state.borrow();
        assert!(s.writes.contains(&(NBASE_INTERRUPT_MASK, 0, 4)));
        assert!(s.writes.contains(&(NBASE_INTERRUPT_STATUS, u32::MAX, 4)));
        assert!(s.writes.contains(&(NBASE_TX_POLL, 1, 2)));
        assert!(
            !s.writes
                .iter()
                .any(|(r, _, w)| (*r == TX_POLL && *w == 1) || (*r == INTERRUPT_STATUS && *w == 2))
        );
    }
}

#[test]
fn nbase_initial_fifo_and_new_b_shared_status_require_real_completion() {
    for version in [61, 63, 64, 65, 66, 70] {
        let time = Time::default();
        let model = Model::nbase(&time);
        Model::store32(&mut model.state.borrow_mut(), TX_CONFIG, 0); // No old TXCFG_EMPTY condition.
        let hw = Hardware::new(&model, &time, revision(version), 0);
        hw.initialize_modern().unwrap();
        let s = model.state.borrow();
        assert_eq!(s.mac[&0xc0aa], 0x7d0);
        assert_eq!(s.mac[&0xc0a6], 0x150);
        assert_eq!(s.mac[&0xc01e], 0x5555);
        assert!(time.now() >= 3700);
        assert_eq!(
            s.writes
                .iter()
                .any(|(r, v, _)| *r == COMMAND && *v & 0x80 != 0),
            version >= 63
        );
        drop(s);
        for broken in [MCU, INTERRUPT_MITIGATION, MAC_OCP] {
            if version == 61 && broken == INTERRUPT_MITIGATION {
                continue;
            }
            let time = Time::default();
            let mut model = Model::nbase(&time);
            if broken == MCU {
                model.state.borrow_mut().bytes[MCU] = 0x20;
            }
            if broken == INTERRUPT_MITIGATION {
                model.state.borrow_mut().bytes[INTERRUPT_MITIGATION] = 0;
            }
            if broken == MAC_OCP {
                model.shared_ready_delay = None;
            }
            let hw = Hardware::new(&model, &time, revision(version), 0);
            assert_eq!(hw.initialize_modern(), Err(BlockError::Timeout));
            assert!(
                !model
                    .state
                    .borrow()
                    .writes
                    .iter()
                    .any(|(r, _, _)| matches!(*r, PHY_OCP | TX_HIGH | TX_LOW | RX_HIGH | RX_LOW))
            );
        }
    }
}

#[test]
fn nbase_polled_carrier_preserves_latch_low_an_restart_and_cache_expiry() {
    for version in [61, 63, 64, 65, 66, 70] {
        let time = Time::default();
        let model = Model::nbase(&time);
        {
            let mut s = model.state.borrow_mut();
            s.phy_ocp.insert(0xa400, 0x1000);
            s.phy_ocp.insert(0xa402, 0x792d);
            s.bmsr_latch_low_once = true;
        }
        let hw = Hardware::new(&model, &time, revision(version), 0);
        let mut link = RuntimeLink::default();
        assert_eq!(link.update(&hw), Ok(true)); // Down state retries its low latch sample.
        assert_eq!(model.state.borrow().phy_ocp_commands, 3);
        let writes = model.state.borrow().writes.len();
        time.delay(NBASE_LINK_POLL_US - 1).unwrap();
        assert_eq!(link.update(&hw), Ok(true));
        assert_eq!(model.state.borrow().writes.len(), writes);
        time.delay(1).unwrap();
        assert_eq!(link.update(&hw), Ok(true));
        assert_eq!(model.state.borrow().phy_ocp_commands, 5);
        model.state.borrow_mut().bmsr_latch_low_once = true;
        time.delay(NBASE_LINK_POLL_US).unwrap();
        assert_eq!(link.update(&hw), Ok(false)); // Up state preserves the brief latched drop.
        time.delay(NBASE_LINK_POLL_US).unwrap();
        assert_eq!(link.update(&hw), Ok(true));
        model.state.borrow_mut().phy_ocp.insert(0xa400, 0x1200);
        time.delay(NBASE_LINK_POLL_US).unwrap();
        assert_eq!(link.update(&hw), Ok(false)); // Restart overrides an old link indication.
        {
            let mut s = model.state.borrow_mut();
            s.phy_ocp.insert(0xa400, 0x1000);
            s.phy_ocp.insert(0xa402, 0x790d);
        }
        time.delay(NBASE_LINK_POLL_US).unwrap();
        assert_eq!(link.update(&hw), Ok(false)); // AN not complete.
        // A register that reads all ones is taken as read, as Linux's
        // genphy_update_link takes it, and polling goes on: a BMCR of all
        // ones has its restart bit set and reads the link down.
        model.state.borrow_mut().phy_ocp.insert(0xa400, u16::MAX);
        time.delay(NBASE_LINK_POLL_US).unwrap();
        assert_eq!(link.update(&hw), Ok(false));
        model.state.borrow_mut().phy_ocp.insert(0xa400, 0x1000);
        model.state.borrow_mut().phy_ocp.insert(0xa402, 0x792d);
        time.delay(NBASE_LINK_POLL_US).unwrap();
        assert_eq!(link.update(&hw), Ok(true));
        let mut model = Model::nbase(&time);
        model.fail_phy_ocp_at = 1;
        let mut link = RuntimeLink::default();
        {
            let hw = Hardware::new(&model, &time, revision(version), 0);
            assert_eq!(link.update(&hw), Err(BlockError::Timeout));
        }
        let writes = model.state.borrow().writes.len();
        model.fail_phy_ocp_at = usize::MAX;
        time.delay(50_000).unwrap();
        model.read32(PHY_OCP);
        let hw = Hardware::new(&model, &time, revision(version), 0);
        assert_eq!(link.update(&hw), Err(BlockError::Timeout));
        assert_eq!(model.state.borrow().writes.len(), writes);
    }
}
