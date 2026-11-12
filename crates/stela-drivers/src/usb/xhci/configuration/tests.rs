use super::*;
use crate::bus::dma::TestRegion;
use crate::usb::xhci::model::*;

#[test]
fn superspeed_companions_associate_only_with_immediately_preceding_endpoint() {
    let mut bytes = vec![
        9, 2, 0, 0, 1, 7, 0, 0x80, 50, 9, 4, 0, 0, 2, 0xff, 0, 0, 0, 7, 5, 0x81, 2, 0, 4, 0, 6,
        0x30, 7, 0, 0, 0, 7, 5, 2, 3, 64, 0, 4, 6, 0x30, 3, 0, 37, 0,
    ];
    let total = bytes.len() as u16;
    bytes[2..4].copy_from_slice(&total.to_le_bytes());
    assert_eq!(sanitize(&bytes, SUPER_SPEED).unwrap(), bytes);
    let all = interfaces(&bytes);
    assert_eq!(
        all[0].endpoints[0].companion,
        Some(SuperSpeedCompanion {
            max_burst: 7,
            attributes: 0,
            bytes_per_interval: 0,
        })
    );
    assert_eq!(
        all[0].endpoints[1].companion,
        Some(SuperSpeedCompanion {
            max_burst: 3,
            attributes: 0,
            bytes_per_interval: 37,
        })
    );
    assert_eq!(
        all[0].extra,
        [vec![6, 0x30, 7, 0, 0, 0], vec![6, 0x30, 3, 0, 37, 0]]
    );
    // A short companion, or one a class descriptor separates from its
    // endpoint, gives way to Linux's default companion.
    let default = Some(SuperSpeedCompanion::default());
    for length in 2..6 {
        let mut bad = bytes.clone();
        bad.splice(
            25..31,
            [length, 0x30, 7, 0, 0, 0][..usize::from(length)]
                .iter()
                .copied(),
        );
        let total = bad.len() as u16;
        bad[2..4].copy_from_slice(&total.to_le_bytes());
        let fixed = sanitize(&bad, SUPER_SPEED).unwrap();
        assert_eq!(interfaces(&fixed)[0].endpoints[0].companion, default);
    }
    let mut separated = bytes.clone();
    separated.splice(25..25, [2, 0x25]);
    let total = separated.len() as u16;
    separated[2..4].copy_from_slice(&total.to_le_bytes());
    let fixed = sanitize(&separated, SUPER_SPEED).unwrap();
    assert_eq!(interfaces(&fixed)[0].endpoints[0].companion, default);
    // A periodic companion cannot claim more than its bursts carry.
    let mut greedy = bytes.clone();
    greedy[42..44].copy_from_slice(&1000u16.to_le_bytes());
    let fixed = sanitize(&greedy, SUPER_SPEED).unwrap();
    assert_eq!(
        interfaces(&fixed)[0].endpoints[1]
            .companion
            .unwrap()
            .bytes_per_interval,
        256
    );
    // Orphan and duplicate companions are dropped.
    for at in [18, 31] {
        let mut stray = bytes.clone();
        stray.splice(at..at, [6, 0x30, 9, 0, 0, 0]);
        let total = stray.len() as u16;
        stray[2..4].copy_from_slice(&total.to_le_bytes());
        assert_eq!(sanitize(&stray, SUPER_SPEED).unwrap(), bytes);
    }
}

pub(crate) fn vendor(value: u8) -> Vec<u8> {
    vec![
        9, 2, 18, 0, 1, value, 0, 0x80, 50, 9, 4, 0, 0, 0, 0xff, 0, 0, 0,
    ]
}

pub(crate) fn ecm(value: u8) -> Vec<u8> {
    let mut bytes = vec![
        9, 2, 0, 0, 2, value, 0, 0x80, 50, 9, 4, 0, 0, 1, 2, 6, 0, 0, 5, 0x24, 6, 0,
        1, // Union: communications 0, data 1.
        13, 0x24, 0x0f, 1, 0, 0, 0, 0, 0xea, 5, 0, 0, 0, 7, 5, 0x83, 3, 16, 0, 1, 9, 4, 1, 0, 0,
        10, 0, 0, 0, // Inactive data setting.
        9, 4, 1, 3, 2, 10, 0, 0, 0, // Nonsequential alternate value.
        7, 5, 0x81, 2, 64, 0, 0, 7, 5, 0x02, 2, 64, 0, 0,
    ];
    let total = bytes.len() as u16;
    bytes[2..4].copy_from_slice(&total.to_le_bytes());
    bytes
}

fn storage(value: u8) -> Vec<u8> {
    vec![
        9, 2, 32, 0, 1, value, 0, 0x80, 50, 9, 4, 0, 0, 2, 8, 6, 0x50, 0, 7, 5, 0x81, 2, 64, 0, 0,
        7, 5, 0x02, 2, 64, 0, 0,
    ]
}

/// Descriptor-only selection fixture. Its synthetic identity
/// grants no access to an addressed controller.
pub(crate) fn device(configurations: Vec<Vec<u8>>) -> Device {
    let count = configurations.len() as u8;
    Device {
        id: SlotId::fixture(1),
        vendor_ethernet_revision: None,
        vendor_ethernet_tcr: None,
        slot: 1,
        speed: FULL_SPEED,
        location: "configuration model".into(),
        descriptor: [
            18, 1, 0, 2, 0, 0, 0, 64, 0x34, 0x12, 0x78, 0x56, 0, 1, 0, 0, 0, count,
        ],
        configurations,
        configuration: Vec::new(),
        manufacturer: None,
        product: None,
        serial: None,
        language: None,
    }
}

/// Class-binding fixture whose permission belongs to the controller that
/// completed Address Device. Never substitute a hardware slot
/// number or another controller's otherwise identical descriptor fixture.
pub(crate) fn addressed_device(controller: &Controller, configurations: Vec<Vec<u8>>) -> Device {
    let mut device = device(configurations);
    device.id = controller.test_device_id(device.slot).unwrap();
    device
}

#[test]
fn source_listed_broadcom_vendor_hci_is_selected_before_vendor_ethernet() {
    let mut configuration = vec![
        9, 2, 39, 0, 1, 7, 0, 0x80, 50, 9, 4, 0, 0, 3, 0xff, 1, 1, 0, 7, 5, 0x81, 3, 16, 0, 1, 7,
        5, 0x82, 2, 64, 0, 0, 7, 5, 0x02, 2, 64, 0, 0,
    ];
    let length = configuration.len() as u16;
    configuration[2..4].copy_from_slice(&length.to_le_bytes());
    let mut device = device(vec![configuration]);
    device.descriptor[8..12].copy_from_slice(&[0x5c, 0x0a, 0xe8, 0x21]);
    select(&mut device).unwrap();
    assert_eq!(device.configuration_value(), 7);
    let all = interfaces(&device.configuration);
    assert!(matches!(
        plan(&device, &all[0], &all),
        Ok(Plan::Bluetooth(_))
    ));
    device.descriptor[8..12].copy_from_slice(&[0x34, 0x12, 0x78, 0x56]);
    assert!(plan(&device, &all[0], &all).is_err());
}

#[test]
fn supported_binding_beats_standard_but_unsupported_rndis_without_losing_inventory() {
    let mut rndis = vendor(9);
    rndis[14..17].copy_from_slice(&[2, 2, 0xff]);
    let originals = vec![rndis, ecm(42), vendor(5)];
    for bytes in &originals {
        sanitize(bytes, SUPER_SPEED).unwrap();
    }
    let mut device = device(originals.clone());
    select(&mut device).unwrap();
    assert_eq!(device.configuration_value(), 42);
    assert_eq!(device.configuration, originals[1]);
    assert_eq!(device.configurations, originals);
    let all = interfaces(&device.configuration);
    let binding = plan(&device, &all[0], &all).unwrap();
    assert!(matches!(binding, Plan::Ethernet(_)));
    assert_eq!(binding.endpoints().len(), 3);
    let Plan::Ethernet(binding) = binding else {
        unreachable!()
    };
    assert_eq!(binding.data_setting(), (1, 3));
}

fn superspeed(bytes: &[u8]) -> Vec<u8> {
    let mut ss = Vec::new();
    let mut at = 0;
    while at < bytes.len() {
        let length = usize::from(bytes[at]);
        let mut d = bytes[at..at + length].to_vec();
        if d[1] == ENDPOINT_DESCRIPTOR {
            if d[3] & 3 == 2 {
                d[4..6].copy_from_slice(&1024u16.to_le_bytes());
            }
            ss.extend_from_slice(&d);
            ss.extend_from_slice(&[
                6,
                SS_ENDPOINT_COMPANION,
                0,
                0,
                if d[3] & 3 == 3 { d[4] } else { 0 },
                0,
            ]);
        } else {
            ss.extend_from_slice(&d);
        }
        at += length;
    }
    let total = ss.len() as u16;
    ss[2..4].copy_from_slice(&total.to_le_bytes());
    sanitize(&ss, SUPER_SPEED).unwrap();
    ss
}

#[test]
fn refused_superspeed_functions_never_tie_or_beat_usable_configuration() {
    let valid = superspeed(&storage(42));
    // Linux fixes a missing companion, a reserved stream count and an
    // oversized burst; each such function then counts.
    let mut missing = storage(7);
    missing[22..24].copy_from_slice(&1024u16.to_le_bytes());
    missing[29..31].copy_from_slice(&1024u16.to_le_bytes());
    let mut streams = superspeed(&storage(7));
    streams[28] = 17;
    let mut burst = superspeed(&storage(7));
    burst[27] = 16;
    for fixed in [missing, streams, burst] {
        let mut dev = device(vec![fixed]);
        dev.speed = SUPER_SPEED;
        select(&mut dev).unwrap();
        let all = interfaces(&dev.configuration);
        assert!(plan(&dev, &all[0], &all).is_ok());
    }
    // A 512-byte SuperSpeed bulk endpoint stays refused.
    let mut small = superspeed(&storage(7));
    small[22..24].copy_from_slice(&512u16.to_le_bytes());
    for refused in [small] {
        for order in [false, true] {
            let mut configurations = vec![refused.clone(), valid.clone()];
            if order {
                configurations.reverse();
            }
            let mut dev = device(configurations);
            dev.speed = SUPER_SPEED;
            select(&mut dev).unwrap();
            assert_eq!(dev.configuration_value(), 42);
            assert_eq!(dev.configurations.len(), 2);
            let fixed = sanitize(&refused, SUPER_SPEED).unwrap();
            let all = interfaces(&fixed);
            assert!(plan(&dev, &all[0], &all).is_err());
        }
    }
    // Even the supported, observed vendor-mode preference cannot count a
    // vendor function whose transport is explicitly refused by this host.
    let mut vendor = superspeed(&storage(7));
    vendor[14..17].copy_from_slice(&[0xff, 0, 0]);
    vendor[22..24].copy_from_slice(&512u16.to_le_bytes());
    let valid = superspeed(&ecm(42));
    for order in [false, true] {
        let mut configurations = vec![vendor.clone(), valid.clone()];
        if order {
            configurations.reverse();
        }
        let mut dev = device(configurations);
        dev.speed = SUPER_SPEED;
        dev.descriptor[8..12].copy_from_slice(&[0xda, 0x0b, 0x53, 0x81]);
        dev.vendor_ethernet_revision = Some(0x4c10);
        select(&mut dev).unwrap();
        assert_eq!(dev.configuration_value(), 42);
    }
}

#[test]
fn ordinary_bulk_configuration_accepts_a_valid_advertised_stream_capability() {
    for exponent in 1..=16 {
        let mut configuration = superspeed(&storage(42));
        configuration[28] = exponent;
        let mut dev = device(vec![configuration.clone()]);
        dev.speed = SUPER_SPEED;
        select(&mut dev).unwrap();
        assert_eq!(dev.configuration_value(), 42);
        let all = interfaces(&configuration);
        assert!(plan(&dev, &all[0], &all).is_ok());
    }
}

#[test]
fn hub_configuration_coverage_requires_a_usable_interrupt_endpoint() {
    let valid = vec![
        9, 2, 25, 0, 1, 42, 0, 0x80, 50, 9, 4, 0, 0, 1, HUB_CLASS, 0, 0, 0, 7, 5, 0x81, 3, 8, 0, 1,
    ];
    // An OUT status endpoint cannot report port changes.
    let mut invalid = valid.clone();
    invalid[5] = 7;
    invalid[20] = 0x01;
    for order in [false, true] {
        let mut configurations = vec![invalid.clone(), valid.clone()];
        if order {
            configurations.reverse();
        }
        let mut dev = device(configurations);
        dev.descriptor[4] = HUB_CLASS;
        select(&mut dev).unwrap();
        assert_eq!(dev.configuration_value(), 42);
    }
}

#[test]
fn equal_supported_configurations_choose_the_first_like_linux() {
    let mut device = device(vec![storage(7), ecm(42)]);
    select(&mut device).unwrap();
    assert_eq!(device.configuration_value(), 7);
    assert_eq!(device.configurations.len(), 2);
}

#[test]
fn greatest_supported_function_coverage_is_unique_and_descriptor_order_independent() {
    let single = storage(7);
    let mut composite = storage(42);
    composite.extend_from_slice(&[
        9, 4, 1, 0, 2, 8, 6, 0x50, 0, 7, 5, 0x83, 2, 64, 0, 0, 7, 5, 0x04, 2, 64, 0, 0,
    ]);
    composite[4] = 2;
    let total = composite.len() as u16;
    composite[2..4].copy_from_slice(&total.to_le_bytes());
    sanitize(&composite, HIGH_SPEED).unwrap();
    for configurations in [
        vec![single.clone(), composite.clone()],
        vec![composite, single],
    ] {
        let mut device = device(configurations);
        select(&mut device).unwrap();
        assert_eq!(device.configuration_value(), 42);
    }
}

#[test]
fn unsupported_interfaces_never_become_a_successful_plan() {
    let mut device = device(vec![vendor(19), vendor(3)]);
    select(&mut device).unwrap();
    let all = interfaces(&device.configuration);
    assert!(plan(&device, &all[0], &all).is_err());
    assert_eq!(device.configuration_value(), 19);
}

#[test]
fn rndis_pairs_only_the_explicit_control_and_data_interfaces() {
    use crate::usb::cdc::tests::rndis_configuration;

    for signature in [[2, 2, 0xff], [0xe0, 1, 3]] {
        let configuration = rndis_configuration(42, signature);
        let dev = device(vec![configuration.clone()]);
        let all = interfaces(&configuration);
        let binding = plan(&dev, &all[0], &all).unwrap();
        assert_eq!(binding.settings(0), [(0, 0), (1, 0)]);
        assert_eq!(binding.endpoints().len(), 3);
        assert!(!primary(&all[1]));
        for (offset, wrong) in [(16, 2), (21, 1), (22, 0), (25, 1), (36, 1), (48, 0x82)] {
            let mut malformed = configuration.clone();
            malformed[offset] = wrong;
            let all = interfaces(&malformed);
            assert!(plan(&dev, &all[0], &all).is_err(), "offset {offset}");
        }
        let mut duplicate_union = configuration;
        duplicate_union.splice(23..23, [5, 0x24, 6, 0, 1]);
        let all = interfaces(&duplicate_union);
        assert!(plan(&dev, &all[0], &all).is_err());
    }
}

#[test]
fn wireless_class_bluetooth_still_uses_the_hci_plan() {
    let configuration = vec![
        9, 2, 39, 0, 1, 7, 0, 0x80, 50, 9, 4, 0, 0, 3, 0xe0, 1, 1, 0, 7, 5, 0x81, 3, 16, 0, 1, 7,
        5, 0x82, 2, 64, 0, 0, 7, 5, 0x02, 2, 64, 0, 0,
    ];
    let mut dev = device(vec![configuration]);
    select(&mut dev).unwrap();
    let all = interfaces(&dev.configuration);
    assert!(matches!(plan(&dev, &all[0], &all), Ok(Plan::Bluetooth(_))));
}

#[test]
fn equal_cdc_coverage_prefers_ncm_then_ecm_then_rndis() {
    use crate::usb::cdc::tests::rndis_configuration;

    let mut ncm = ecm(9);
    ncm[15] = cdc::NCM;
    ncm.splice(18..18, [6, 0x24, 0x1a, 0, 1, 1]);
    let total = ncm.len() as u16;
    ncm[2..4].copy_from_slice(&total.to_le_bytes());
    let rndis = rndis_configuration(42, [0xe0, 1, 3]);
    for (configurations, value) in [
        (vec![rndis.clone()], 42),
        (vec![rndis.clone(), ecm(7)], 7),
        (vec![rndis.clone(), ecm(7), ncm.clone()], 9),
        (vec![ncm, ecm(7), rndis.clone()], 9),
        (vec![storage(6), rndis], 6),
    ] {
        let mut dev = device(configurations);
        select(&mut dev).unwrap();
        assert_eq!(dev.configuration_value(), value);
    }
}

#[test]
fn unsupported_or_unobserved_realtek_revision_preserves_compatible_cdc_choice() {
    let mut vendor = storage(7);
    vendor[14..17].copy_from_slice(&[0xff, 0, 0]);
    for revision in [None, Some(0x4c20), Some(0x7010), Some(0x1030), Some(0x2020)] {
        let mut device = device(vec![vendor.clone(), ecm(42)]);
        device.descriptor[8..12].copy_from_slice(&[0xda, 0x0b, 0x56, 0x81]);
        device.vendor_ethernet_revision = revision;
        select(&mut device).unwrap();
        assert_eq!(device.configuration_value(), 42);
        assert_eq!(device.configurations.len(), 2);
    }
    let mut device = device(vec![vendor]);
    device.descriptor[8..12].copy_from_slice(&[0xda, 0x0b, 0x52, 0x81]);
    device.vendor_ethernet_revision = Some(0x4c10);
    select(&mut device).unwrap();
    let all = interfaces(&device.configuration);
    assert!(matches!(
        plan(&device, &all[0], &all),
        Ok(Plan::VendorEthernet(_))
    ));
}

#[test]
fn implemented_realtek_vendor_mode_wins_equal_cdc_coverage_independent_of_order() {
    let mut vendor = storage(7);
    vendor[14..17].copy_from_slice(&[0xff, 0, 0]);
    for revision in [0x4c00, 0x4c10, 0x4800] {
        for configurations in [vec![vendor.clone(), ecm(42)], vec![ecm(42), vendor.clone()]] {
            let mut device = device(configurations);
            device.descriptor[8..12].copy_from_slice(&[0xda, 0x0b, 0x52, 0x81]);
            device.vendor_ethernet_revision = Some(revision);
            select(&mut device).unwrap();
            assert_eq!(device.configuration_value(), 7);
        }
    }
    let mut other = vendor.clone();
    other[5] = 9;
    let mut device = device(vec![vendor, other, ecm(42)]);
    device.descriptor[8..12].copy_from_slice(&[0xda, 0x0b, 0x52, 0x81]);
    device.vendor_ethernet_revision = Some(0x4c10);
    // Linux takes the first of equally good configurations.
    select(&mut device).unwrap();
    assert_eq!(device.configuration_value(), 7);
}

#[test]
fn exact_oem_candidates_need_observed_supported_chip_and_fixed_bulk_pipes() {
    for &(vid, pid) in ethernet::USB_IDS {
        for revision in [None, Some(0x4c20), Some(0x1030), Some(0x2020), Some(0x5c10)] {
            for variant in 0..4 {
                let mut vendor = storage(7);
                vendor[14..17].copy_from_slice(&[0xff, 0, 0]);
                match variant {
                    1 => vendor[20] = 0x83,
                    2 => vendor[27] = 0x01,
                    3 => vendor[14] = 0x02,
                    _ => {}
                }
                for configurations in [vec![vendor.clone(), ecm(42)], vec![ecm(42), vendor.clone()]]
                {
                    let mut device = device(configurations);
                    device.descriptor[8..10].copy_from_slice(&vid.to_le_bytes());
                    device.descriptor[10..12].copy_from_slice(&pid.to_le_bytes());
                    device.vendor_ethernet_revision = revision;
                    select(&mut device).unwrap();
                    assert_eq!(
                        device.configuration_value(),
                        if revision == Some(0x5c10) && variant == 0 {
                            7
                        } else {
                            42
                        },
                        "{vid:04x}:{pid:04x} {revision:?}/{variant}"
                    );
                }
            }
        }
    }
}

#[test]
fn asix_exact_interface_candidates_preserve_cdc_and_select_supported_alternate() {
    for (vid, pid) in ethernet::asix_ids() {
        for mismatch in 0..4 {
            let mut vendor = storage(7);
            vendor[14..17].copy_from_slice(&[0xff, 0xff, 0]);
            match mismatch {
                1 => vendor[15] = 0,
                2 => vendor[16] = 1,
                3 => vendor[14] = 2,
                _ => {}
            }
            for configurations in [vec![vendor.clone(), ecm(42)], vec![ecm(42), vendor.clone()]] {
                let mut dev = device(configurations);
                dev.descriptor[8..10].copy_from_slice(&vid.to_le_bytes());
                dev.descriptor[10..12].copy_from_slice(&pid.to_le_bytes());
                select(&mut dev).unwrap();
                assert_eq!(dev.configuration_value(), 42);
                let all = interfaces(&vendor);
                assert_eq!(plan(&dev, &all[0], &all).is_ok(), mismatch == 0);
            }
        }
    }
    let mut bytes = storage(7);
    bytes.truncate(18);
    bytes[13] = 0;
    bytes[14..17].copy_from_slice(&[0xff, 0xff, 0]);
    bytes.extend_from_slice(&[
        9, 4, 0, 2, 2, 0xff, 0xff, 0, 0, 7, 5, 0x81, 2, 64, 0, 0, 7, 5, 2, 2, 64, 0, 0,
    ]);
    let total = bytes.len() as u16;
    bytes[2..4].copy_from_slice(&total.to_le_bytes());
    let mut dev = device(vec![bytes]);
    dev.descriptor[8..12].copy_from_slice(&[0x95, 0x0b, 0x90, 0x17]);
    select(&mut dev).unwrap();
    let all = interfaces(&dev.configuration);
    let Plan::VendorEthernet(binding) = plan(&dev, &all[0], &all).unwrap() else {
        panic!()
    };
    assert_eq!(binding.setting(), (0, 2));
    assert!(binding.asix_candidate() && !binding.observed_vendor());
}

#[test]
fn ax88772_exact_candidates_keep_cdc_and_refuse_superspeed_before_binding() {
    for (vid, pid) in ethernet::asix_legacy_ids() {
        for (speed, packet) in [(FULL_SPEED, 64u16), (HIGH_SPEED, 512), (SUPER_SPEED, 1024)] {
            let mut vendor = storage(7);
            vendor[14..17].copy_from_slice(&[0xff, 0, 0]);
            vendor[22..24].copy_from_slice(&packet.to_le_bytes());
            vendor[29..31].copy_from_slice(&packet.to_le_bytes());
            let vendor = if speed == SUPER_SPEED {
                superspeed(&vendor)
            } else {
                vendor
            };
            let mut compatible = ecm(42);
            if speed != FULL_SPEED {
                let mut at = 0;
                while at < compatible.len() {
                    let n = compatible[at] as usize;
                    if compatible[at + 1] == ENDPOINT_DESCRIPTOR && compatible[at + 3] & 3 == 2 {
                        compatible[at + 4..at + 6].copy_from_slice(&packet.to_le_bytes());
                    }
                    at += n;
                }
            }
            if speed == SUPER_SPEED {
                compatible = superspeed(&compatible);
            }
            for configurations in [
                vec![vendor.clone(), compatible.clone()],
                vec![compatible.clone(), vendor.clone()],
            ] {
                let mut dev = device(configurations);
                dev.speed = speed;
                dev.descriptor[8..10].copy_from_slice(&vid.to_le_bytes());
                dev.descriptor[10..12].copy_from_slice(&pid.to_le_bytes());
                select(&mut dev).unwrap();
                assert_eq!(dev.configuration_value(), 42);
                let all = interfaces(&vendor);
                assert_eq!(plan(&dev, &all[0], &all).is_ok(), speed != SUPER_SPEED);
            }
        }
    }
}

#[test]
fn asix_alternate_search_validates_superspeed_contexts_before_claiming() {
    let mut vendor = superspeed(&storage(7));
    vendor[14..17].copy_from_slice(&[0xff, 0xff, 0]);
    let mut alternate = vendor.clone();
    alternate[12] = 2;
    // A 512-byte SuperSpeed bulk endpoint invalidates the default setting.
    vendor[22..24].copy_from_slice(&512u16.to_le_bytes());
    vendor.extend_from_slice(&alternate[9..]);
    let length = vendor.len() as u16;
    vendor[2..4].copy_from_slice(&length.to_le_bytes());
    let mut dev = device(vec![vendor.clone()]);
    dev.speed = SUPER_SPEED;
    dev.descriptor[8..12].copy_from_slice(&[0x95, 0x0b, 0x90, 0x17]);
    select(&mut dev).unwrap();
    let all = interfaces(&dev.configuration);
    let Plan::VendorEthernet(binding) = plan(&dev, &all[0], &all).unwrap() else {
        panic!()
    };
    assert_eq!(binding.setting(), (0, 2));
    for configurations in [
        vec![vendor.clone(), superspeed(&ecm(42))],
        vec![superspeed(&ecm(42)), vendor.clone()],
    ] {
        dev.configurations = configurations;
        select(&mut dev).unwrap();
        assert_eq!(dev.configuration_value(), 42);
    }
    let mut other = vendor.clone();
    other[5] = 9;
    dev.configurations = vec![vendor, other];
    // Linux takes the first of equally good configurations.
    select(&mut dev).unwrap();
    assert_eq!(dev.configuration_value(), 7);
}

#[test]
fn an_endpoint_another_interface_owns_is_skipped_like_linux() {
    let mut bytes = ecm(42);
    bytes.extend_from_slice(&[9, 4, 2, 0, 1, 0xff, 0, 0, 0, 7, 5, 0x81, 2, 64, 0, 0]);
    bytes[4] = 3;
    let total = bytes.len() as u16;
    bytes[2..4].copy_from_slice(&total.to_le_bytes());
    let mut device = device(vec![bytes]);
    select(&mut device).unwrap();
    let all = interfaces(&device.configuration);
    let vendor = all.iter().find(|interface| interface.number == 2).unwrap();
    assert!(vendor.endpoints.is_empty());
    assert!(active_settings(&all, &[(1, 1)]).is_ok());
}

#[test]
fn two_cdc_masters_cannot_claim_one_data_interface() {
    let mut bytes = ecm(42);
    // A second communications interface, naming the original data interface.
    bytes.extend_from_slice(&[
        9, 4, 2, 0, 0, 2, 6, 0, 0, 5, 0x24, 6, 2, 1, 13, 0x24, 0x0f, 1, 0, 0, 0, 0, 0xea, 5, 0, 0,
        0,
    ]);
    bytes[4] = 3;
    let total = bytes.len() as u16;
    bytes[2..4].copy_from_slice(&total.to_le_bytes());
    sanitize(&bytes, HIGH_SPEED).unwrap();
    let mut device = device(vec![bytes]);
    assert_eq!(
        select(&mut device),
        Err("supported USB functions claim the same interface")
    );
}

#[test]
fn malformed_configurations_are_read_as_linux_reads_them() {
    let valid = ecm(42);
    assert_eq!(sanitize(&valid, HIGH_SPEED).unwrap(), valid);
    // Only the header is fatal.
    for (offset, value) in [(0, 8), (1, 1), (5, 0)] {
        let mut malformed = valid.clone();
        malformed[offset] = value;
        assert!(sanitize(&malformed, HIGH_SPEED).is_err(), "offset {offset}");
    }
    assert!(sanitize(&valid[..8], HIGH_SPEED).is_err());
    // A truncated stream keeps every whole descriptor and fixes counts.
    let truncated = sanitize(&valid[..valid.len() - 3], HIGH_SPEED).unwrap();
    assert_eq!(usize::from(truncated[2]), truncated.len());
    // Wrong interface and endpoint counts are corrected.
    let mut miscounted = valid.clone();
    miscounted[4] = 7;
    assert_eq!(sanitize(&miscounted, HIGH_SPEED).unwrap(), valid);
    // Duplicate endpoints in one setting, endpoint zero and empty packets
    // are skipped; the device keeps the rest.
    let mut duplicate = storage(1);
    duplicate[27] = 0x81;
    let fixed = sanitize(&duplicate, HIGH_SPEED).unwrap();
    assert_eq!(interfaces(&fixed)[0].endpoints.len(), 1);
    let mut orphan = vendor(1);
    orphan.extend_from_slice(&[7, 5, 0x81, 2, 64, 0, 0]);
    orphan.drain(9..18);
    orphan[2] = orphan.len() as u8;
    assert!(interfaces(&sanitize(&orphan, HIGH_SPEED).unwrap()).is_empty());
}

#[test]
fn endpoint_intervals_and_packet_sizes_are_corrected_like_linux() {
    let fix = |bytes: [u8; 7], speed| {
        let mut endpoint = bytes.to_vec();
        fix_endpoint(&mut endpoint, speed);
        (u16::from_le_bytes([endpoint[4], endpoint[5]]), endpoint[6])
    };
    // A full-speed period in a high-speed interrupt endpoint: 32 frames.
    assert_eq!(fix([7, 5, 0x81, 3, 8, 0, 32], HIGH_SPEED), (8, 9));
    assert_eq!(fix([7, 5, 0x81, 3, 8, 0, 0], HIGH_SPEED), (8, 7));
    assert_eq!(fix([7, 5, 0x81, 3, 8, 0, 0], FULL_SPEED), (8, 10));
    assert_eq!(fix([7, 5, 0x81, 1, 0, 2, 0], FULL_SPEED), (512, 4));
    // Oversized packets are clamped; high-speed periodic keeps its
    // additional transactions.
    assert_eq!(fix([7, 5, 0x81, 2, 0, 4, 0], HIGH_SPEED), (512, 0));
    assert_eq!(fix([7, 5, 0x81, 3, 0, 0x14, 1], HIGH_SPEED), (0x1400, 1));
    assert_eq!(fix([7, 5, 0x81, 3, 64, 0, 1], LOW_SPEED), (8, 1));
    // Linux only warns about a high-speed bulk endpoint below 512.
    assert_eq!(fix([7, 5, 0x81, 2, 64, 0, 0], HIGH_SPEED), (64, 0));
}

#[test]
fn uac2_selection_keeps_zero_bandwidth_until_explicit_pcm_start() {
    let configuration = audio::test_configuration();
    sanitize(&configuration, HIGH_SPEED).unwrap();
    let mut device = device(vec![configuration]);
    device.speed = 3;
    select(&mut device).unwrap();
    let all = interfaces(&device.configuration);
    let selected = plan(&device, &all[0], &all).unwrap();
    assert!(matches!(selected, Plan::Audio(_)));
    assert!(selected.endpoints().is_empty());
    assert_eq!(selected.settings(0), vec![(0, 0), (1, 0), (2, 0)]);
    active_settings(&all, &selected.settings(0)).unwrap();
}

#[test]
fn vendor_class_interfaces_reach_ethernet_only_for_listed_adapters() {
    // A vendor-class interface without endpoints, as LED controllers have.
    let configuration = vec![
        9, 2, 18, 0, 1, 1, 0, 0x80, 50, 9, 4, 0, 0, 0, 0xff, 0xff, 0xff, 0,
    ];
    let mut device = device(vec![configuration]);
    select(&mut device).unwrap();
    let all = interfaces(&device.configuration);
    assert_eq!(
        plan(&device, &all[0], &all).err(),
        Some("no class driver for this interface")
    );
    // A listed RTL815x ID without a revision read keeps its own reason.
    device.descriptor[8..12].copy_from_slice(&[0xda, 0x0b, 0x53, 0x81]);
    assert_eq!(
        plan(&device, &all[0], &all).err(),
        Some("USB Ethernet chip revision not identified")
    );
}

#[test]
fn uac2_board_codec_binds_its_association_and_leaves_hid_to_hid() {
    let configuration = audio::board_codec_configuration();
    assert_eq!(sanitize(&configuration, HIGH_SPEED).unwrap(), configuration);
    let mut device = device(vec![configuration]);
    device.speed = HIGH_SPEED;
    select(&mut device).unwrap();
    let all = interfaces(&device.configuration);
    let selected = plan(&device, &all[0], &all).unwrap();
    assert!(matches!(selected, Plan::Audio(_)));
    assert!(selected.endpoints().is_empty());
    let settings = selected.settings(0);
    assert_eq!(settings, (0..7).map(|n| (n, 0)).collect::<Vec<_>>());
    active_settings(&all, &settings).unwrap();
    let hid = all.iter().find(|i| i.number == 7 && primary(i)).unwrap();
    assert!(matches!(plan(&device, hid, &all), Ok(Plan::Hid(..))));
}

fn storage_modes(value: u8, superspeed: bool, bot: bool, uas: bool) -> Vec<u8> {
    let mut bytes = vec![9, 2, 0, 0, 1, value, 0, 0x80, 50];
    let packet = if superspeed { 1024u16 } else { 512u16 };
    for (enabled, alternate, protocol, pipes) in [
        (bot, 0, 0x50, vec![(0x81u8, 0u8), (2, 0)]),
        (
            uas,
            u8::from(bot),
            0x62,
            vec![(3u8, 1u8), (0x84, 2), (0x85, 3), (6, 4)],
        ),
    ] {
        if !enabled {
            continue;
        }
        bytes.extend_from_slice(&[9, 4, 0, alternate, pipes.len() as u8, 8, 6, protocol, 0]);
        for (address, pipe) in pipes {
            bytes.extend_from_slice(&[7, 5, address, 2, packet as u8, (packet >> 8) as u8, 0]);
            if superspeed {
                bytes.extend_from_slice(&[
                    6,
                    SS_ENDPOINT_COMPANION,
                    0,
                    if pipe > 1 { 4 } else { 0 },
                    0,
                    0,
                ]);
            }
            if protocol == 0x62 {
                bytes.extend_from_slice(&[4, 0x24, pipe, 0]);
            }
        }
    }
    let length = bytes.len() as u16;
    bytes[2..4].copy_from_slice(&length.to_le_bytes());
    configuration::sanitize(&bytes, SUPER_SPEED).unwrap();
    bytes
}

#[test]
fn storage_selector_uses_stream_policy_before_configuration_and_bot_fallback() {
    let mut dma = TestRegion::new(128 * 1024 * 8);
    let (mut controller, model) = unconfigured_cancellation_controller(&mut dma);
    controller.max_primary_streams = 16;
    controller.slots[1].as_mut().unwrap().speed = SUPER_SPEED;
    let mut dev = configuration::tests::addressed_device(
        &controller,
        vec![storage_modes(7, true, true, true)],
    );
    dev.speed = SUPER_SPEED;
    for (policy, expected) in [
        (streams::Policy::Unknown, (0, 0)),
        (streams::Policy::Broken, (0, 0)),
        (streams::Policy::Standard, (0, 1)),
        (streams::Policy::Cadence, (0, 1)),
    ] {
        controller.stream_policy = policy;
        configuration::select_with_controller(&mut dev, Some(&controller)).unwrap();
        let all = interfaces(&dev.configuration);
        let Plan::Storage(binding) =
            configuration::plan_with_controller(&dev, &all[0], &all, Some(&controller)).unwrap()
        else {
            panic!()
        };
        assert_eq!(binding.setting(), expected);
        assert_eq!(binding.uas(), expected.1 == 1);
    }
    controller.stream_policy = streams::Policy::Standard;
    controller.max_primary_streams = 0;
    let all = interfaces(&dev.configuration);
    let Plan::Storage(binding) =
        configuration::plan_with_controller(&dev, &all[0], &all, Some(&controller)).unwrap()
    else {
        panic!()
    };
    assert!(!binding.uas());
    assert_eq!(model.borrow().configuration_value, 0);
    assert!(
        model
            .borrow()
            .configuration_seen_before_endpoints
            .is_empty()
    );
    assert!(model.borrow().active.iter().skip(2).all(Option::is_none));
}

#[test]
fn storage_configuration_prefers_usable_uas_and_preserves_quirk_bot_fallback() {
    for superspeed in [false, true] {
        let mut dma = TestRegion::new(128 * 1024 * 8);
        let (mut controller, _) = unconfigured_cancellation_controller(&mut dma);
        let speed = if superspeed { SUPER_SPEED } else { HIGH_SPEED };
        controller.slots[1].as_mut().unwrap().speed = speed;
        controller.stream_policy = streams::Policy::Standard;
        controller.max_primary_streams = 16;
        for reverse in [false, true] {
            let mut configs = vec![
                storage_modes(7, superspeed, true, false),
                storage_modes(9, superspeed, false, true),
            ];
            if reverse {
                configs.reverse();
            }
            let mut dev = configuration::tests::addressed_device(&controller, configs);
            dev.speed = speed;
            configuration::select_with_controller(&mut dev, Some(&controller)).unwrap();
            assert_eq!(dev.configuration_value(), 9);
            // This source-listed SanDisk enclosure refuses UAS even
            // on a capable stream host; keep the usable BOT alternative.
            dev.descriptor[8..12].copy_from_slice(&[0x81, 0x07, 0xe8, 0x55]);
            configuration::select_with_controller(&mut dev, Some(&controller)).unwrap();
            assert_eq!(dev.configuration_value(), 7);
        }
    }
}
