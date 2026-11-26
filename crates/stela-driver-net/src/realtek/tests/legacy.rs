// SPDX-License-Identifier: GPL-2.0-only
// Copyright (c) 2026 Stela contributors
//! Legacy cases.
use super::*;

#[test]
fn every_implemented_phy_branch_executes_analogue_sequence_and_restarts_copper_aneg() {
    for version in [2, 3, 4, 5, 6, 17, 18, 19, 20, 21, 22, 23, 24, 34] {
        let time = Time::default();
        let model = Model::ready(&time);
        let mut hw = Hardware::new(&model, &time, revision(version), 0);
        let bytes = 0x801f_0000u32.to_le_bytes();
        let script = Script::parse(&bytes).unwrap();
        hw.configure_phy(Some(&script), version == 5).unwrap();
        let s = model.state.borrow();
        assert_eq!(s.page, 0, "version{version}");
        assert_eq!(s.phy[&(0, 4)], 0x0de1);
        assert_eq!(s.phy[&(0, 9)] & 0xfb00, 0x0200);
        assert_eq!(s.phy[&(0, 0)], 0x1200);
        let expected = match version {
            2 | 3 => ((1, 0x18), 0x65c7),
            4 => ((2, 1), 0x90d0),
            5 => ((1, 0x10), 0xf01b),
            6 => ((1, 0x18), 0x67c7),
            17 => ((1, 0x10), 0xf41b),
            18 => ((2, 0xc), 0x1ec8),
            19 => ((3, 0x12), 0xc096),
            20 => ((2, 6), 0x0761),
            21 | 22 => ((2, 6), 0x5461),
            23 | 24 => ((1, 0x1d), 0x3d98),
            34 => ((3, 9), 0xa20f),
            _ => unreachable!(),
        };
        assert_eq!(s.phy[&expected.0], expected.1, "version{version}");
        assert!(time.now() >= 70_000);
        assert!(
            s.writes
                .iter()
                .all(|(o, v, _)| *o != COMMAND || *v != u32::from(ENABLE))
        );
    }
}

#[test]
fn reset_mdio_ephy_eri_csi_faults_are_bounded_and_configuration_lock_restores() {
    for port in [PHY_ACCESS, EPHY_ACCESS, ERI_ADDRESS, CSI_ADDRESS] {
        let time = Time::default();
        let mut model = Model::ready(&time);
        model.fault = Some(port);
        let mut hw = Hardware::new(&model, &time, revision(34), 3);
        let result = match port {
            PHY_ACCESS => hw.phy_read(2).map(|_| ()),
            EPHY_ACCESS => hw.ephy(1, 0, 1),
            ERI_ADDRESS => hw.eri_read(0xe0).map(|_| ()),
            CSI_ADDRESS => hw.latency(),
            _ => unreachable!(),
        };
        assert_eq!(result, Err(BlockError::Timeout));
        assert!(time.now() <= 10_000);
        if port != PHY_ACCESS {
            assert_eq!(
                hw.configure_mac(None),
                Err(Error::Device(BlockError::Timeout))
            );
            assert_eq!(model.read8(CONFIG_LOCK), 0);
        }
    }
    let time = Time::default();
    let mut model = Model::ready(&time);
    model.stuck_reset = true;
    let hw = Hardware::new(&model, &time, revision(2), 0);
    assert_eq!(hw.reset(), Err(BlockError::Timeout));
    assert_eq!(time.now(), 10_000);
    let time = Time::default();
    let mut model = Model::ready(&time);
    model.stuck_phy_reset = true;
    let hw = Hardware::new(&model, &time, revision(18), 0);
    assert_eq!(hw.phy_reset(), Err(BlockError::Timeout));
    assert!(time.now() < 700_000);
    // Fail partway through an analogue sequence, not only at its first ID read.
    for nth in [8, 14, 25] {
        let time = Time::default();
        let mut model = Model::ready(&time);
        model.fail_phy_at = nth;
        let mut hw = Hardware::new(&model, &time, revision(19), 0);
        assert!(hw.configure_phy(None, false).is_err());
        assert!(time.now() < 100_000);
    }
}

#[test]
fn each_mac_revision_uses_its_real_ephy_and_fifo_configuration_before_dma() {
    for version in [2, 3, 4, 5, 6, 17, 18, 19, 20, 21, 22, 23, 24, 34] {
        let time = Time::default();
        let model = Model::ready(&time);
        let mut hw = Hardware::new(&model, &time, revision(version), 2);
        hw.reset().unwrap();
        hw.configure_mac(None).unwrap();
        let s = model.state.borrow();
        assert_eq!(s.bytes[CONFIG_LOCK], 0);
        assert_eq!(
            u16::from_le_bytes(s.bytes[CPLUS..CPLUS + 2].try_into().unwrap())
                & ((1 << 5) | (1 << 6)),
            0
        );
        assert!(
            !s.writes
                .iter()
                .any(|(reg, _, _)| matches!(*reg, TX_LOW | TX_HIGH | RX_LOW | RX_HIGH))
        );
        match version {
            18 => {
                assert_eq!(s.ephy[2], 0x1000);
                assert_eq!(s.ephy[7], 0x2000);
            }
            19 => {
                assert_eq!(s.ephy[3], 2);
                assert_eq!(s.bytes[DEBUG], 0x1e);
            }
            20 | 21 => assert_eq!(s.ephy[3], 0x20),
            24 => assert_eq!(s.bytes[DEBUG], 0x20),
            34 => {
                assert_eq!(s.ephy[0xc], 0x0200);
                assert_eq!(s.eri[&0xc8], 0x0010_0002);
                assert_eq!(s.eri[&0xe8], 0x0010_0006);
                assert_eq!(s.eri[&0xd0], 0x07ff_0060);
            }
            _ => (),
        }
        if version >= 18 {
            let writes: Vec<_> = s
                .writes
                .iter()
                .filter(|(reg, _, _)| *reg == CSI_ADDRESS)
                .collect();
            assert_eq!(writes[0].1, 0x0002_f70c);
            assert_eq!(writes[1].1, 0x8002_f70c);
            assert!(s.writes.contains(&(CSI_DATA, 0x2778_5634, 4)));
        }
    }
}

#[test]
fn register_validation_rejects_bad_addresses_without_io_and_mac_mcu_page_is_restored() {
    use firmware::ScriptIo;
    let time = Time::default();
    let model = Model::ready(&time);
    let mut hw = Hardware::new(&model, &time, revision(34), 0);
    assert!(hw.phy_read(32).is_err());
    assert!(hw.eri_read(1).is_err());
    assert!(hw.eri_write(0, 0, 0).is_err());
    assert!(hw.ephy(32, 0, 0).is_err());
    assert!(model.state.borrow().writes.is_empty());
    hw.write(true, 31, 0x0a00).unwrap();
    hw.write(true, 0x10, 0x55aa).unwrap();
    assert_eq!(hw.read(true, 0x10).unwrap(), 0x55aa);
    hw.restore();
    hw.write(true, 0, 0x1234).unwrap();
    assert_eq!(model.state.borrow().mac[&OCP_PHY_BASE], 0x1234);
}
