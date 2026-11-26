// SPDX-License-Identifier: GPL-2.0-only
// Copyright (c) 2026 Stela contributors
//! H revisions cases.
use super::*;

#[test]
fn h_phy_adc_bias_sentinel_rlen_and_masks_use_calibration_values() {
    for xid in [0x541, 0x6c0] {
        let detected = revision::detect(xid << 20).unwrap();
        assert_eq!(detected.version, 46);
        assert!(detected.implemented());
        assert_eq!(detected.firmware, Some("rtl_nic/rtl8168h-2.fw"));
        assert_eq!(u32::from(detected.xid), xid);
    }
    let bytes = 0x8016_a55au32.to_le_bytes();
    let script = Script::parse(&bytes).unwrap();
    for (adc1, adc2, bias) in [
        (0x80, 0x1234, Some(0x891c)),
        (0, 0, Some(0)),
        (0x80, 0xffff, None),
    ] {
        for (rlen, expected) in [(0, 0), (3, 0), (4, 0x1111), (15, 0xcccc)] {
            let time = Time::default();
            let mut model = Model::modern(&time);
            model.h_adc = Some((adc1, adc2));
            {
                let mut s = model.state.borrow_mut();
                s.g_parameters.extend([(0x808a, 0xffff), (0x0811, 1)]);
                s.phy_ocp.extend([
                    (0xbcdc, 0xfff0 | rlen),
                    (0xbcfc, 0x7654),
                    (0xa442, 0xffff),
                    (0xa430, 0xffff),
                    (0xa5d0, 0xffff),
                ]);
            }
            let mut hw = Hardware::new(&model, &time, revision(46), 0);
            assert_eq!(hw.configure_phy(Some(&script), false).unwrap(), None);
            let s = model.state.borrow();
            assert_eq!(s.g_parameters[&0x808a], 0xffca);
            assert_eq!(s.g_parameters[&0x0811], 0x0801);
            assert_eq!(s.phy_ocp[&0xbcfc], bias.unwrap_or(0x7654));
            assert_eq!(s.phy_ocp[&0xbcde], expected);
            assert_eq!(s.phy_ocp[&0xa442], 0xff7f);
            assert_eq!(s.phy_ocp[&0xa430], 0xfffa);
            assert_eq!(s.phy_ocp[&0xa5d0], 0);
            assert_eq!(s.phy_ocp[&0xa400], 0x1200);
            // Pause advertised, as Linux: flow control under bursts.
            assert_eq!(s.phy_ocp[&0xa408], 0x0de1);
            assert_eq!(s.mac[&0xdd02], 0x807d);
            if bias.is_none() {
                assert!(!s.writes.iter().any(|(r, v, _)| *r == PHY_OCP
                    && *v & ACCESS != 0
                    && (*v >> 15) as u16 & !1 == 0xbcfc));
            }
        }
    }
}

#[test]
fn h_mac_saw_count_timer_masks_and_ephy_complete_before_dma() {
    use crate::bus::dma::TestRegion;
    for saw in [0, 1, 1000, 0x3fff] {
        let time = Time::default();
        let model = Model::modern(&time);
        {
            let mut s = model.state.borrow_mut();
            s.ephy.fill(0xffff);
            s.bytes[DLL_POWER] = 0xfe;
            s.bytes[MISC_1] = 0xfe;
            s.bytes[CONFIG3] = 0x7f;
            s.phy_ocp.insert(0xc426, saw | 0xc000);
            s.mac.extend([
                (0xd412, 0xf812),
                (0xe092, 0xabc4),
                (0xe052, 0x7fff),
                (0xe056, 0xffff),
                (0xe0d6, 0xffff),
                (0xd420, 0xffff),
            ]);
            s.eri.extend([(0x5f0, 0xabcd_0000), (0x1b0, 0x1003)]);
        }
        let mut hw = Hardware::new(&model, &time, revision(46), 0);
        hw.initialize_modern().unwrap();
        hw.configure_mac(None).unwrap();
        {
            let s = model.state.borrow();
            assert_eq!(s.ephy[0x1e], 0xf7ff);
            assert_eq!(s.ephy[0x1d], 0xffff);
            for (reg, value) in [(5, 0x2089), (6, 0x5881), (4, 0x854a), (1, 0x068b)] {
                assert_eq!(s.ephy[reg], value);
            }
            assert_eq!(s.eri[&0x5f0], 0xabcd_4f87);
            assert_eq!(s.eri[&0x1b0], 0);
            assert_eq!(s.bytes[DLL_POWER], 0x3e);
            assert_eq!(s.bytes[MISC_1], 0xb6); // MISC DWORD's RXDV gate shares byte f2 bit3.
            assert_eq!(s.mac[&0xe092], 0xab00);
            assert_eq!(s.mac[&0xe048], 1546);
            assert_eq!(
                s.mac[&0xd412],
                if saw == 0 {
                    0xf812
                } else {
                    0xf000 | ((16_000_000 / u32::from(saw)) & 0xfff) as u16
                }
            );
            assert_eq!(s.mac[&0xe052], 0x9fff);
            assert_eq!(s.mac[&0xe056], 0xff0f);
            assert_eq!(s.mac[&0xe0d6], 0xff7f);
            assert_eq!(s.mac[&0xd420], 0xf47f);
            let pulse: Vec<_> = s
                .writes
                .iter()
                .filter(|(r, v, _)| {
                    *r == MAC_OCP && (*v >> 15) as u16 & !1 == 0xe63e && *v & ACCESS != 0
                })
                .map(|(_, v, _)| *v as u16)
                .collect();
            assert_eq!(pulse, [1, 0]);
            assert!(
                !s.writes
                    .iter()
                    .any(|(r, _, _)| matches!(*r, TX_HIGH | TX_LOW | RX_HIGH | RX_LOW))
            );
        }
        let mut dma = TestRegion::new(200_000 * 8);
        let mut rx = Ring::new(&mut dma, true).unwrap();
        let mut tx = Ring::new(&mut dma, true).unwrap();
        rx.initialize(true);
        tx.initialize(false);
        publish(&hw, &rx, &tx, read_station(&hw).unwrap()).unwrap();
        assert_eq!(
            model.read32(RX_CONFIG),
            (1 << 15) | (1 << 14) | (1 << 11) | (7 << 8) | 0x0e
        );
    }
}

#[test]
fn h_calibration_failed_bus_read_and_runtime_tx_lpi_failure_remain_errors() {
    let bytes = 0x8016_a55au32.to_le_bytes();
    let script = Script::parse(&bytes).unwrap();
    let time = Time::default();
    let mut model = Model::modern(&time);
    model.failed_read = Some(MAC_OCP);
    let mut hw = Hardware::new(&model, &time, revision(46), 0);
    assert_eq!(
        hw.configure_phy(Some(&script), false),
        Err(Error::Device(BlockError::Device { status: u32::MAX }))
    );
    assert_ne!(model.state.borrow().phy_ocp[&0xa400], 0x1200);
    for version in [46, 51, 52] {
        let time = Time::default();
        let mut model = Model::modern(&time);
        model.fail_eri_at = 2;
        model.state.borrow_mut().bytes[PHY_STATUS] = 0x12;
        let mut link = RuntimeLink::default();
        {
            let hw = Hardware::new(&model, &time, revision(version), 0);
            assert_eq!(link.update(&hw), Err(BlockError::Timeout));
        }
        let writes = model.state.borrow().writes.len();
        model.fail_eri_at = usize::MAX;
        time.delay(50_000).unwrap();
        model.read32(ERI_ADDRESS);
        let hw = Hardware::new(&model, &time, revision(version), 0);
        assert_eq!(link.update(&hw), Err(BlockError::Timeout));
        assert_eq!(model.state.borrow().writes.len(), writes);
    }
}
