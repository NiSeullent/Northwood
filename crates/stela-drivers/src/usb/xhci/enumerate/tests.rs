use super::*;
use crate::bus::dma::TestRegion;
use crate::usb::xhci::model::*;

#[test]
fn multi_tt_hubs_use_the_status_endpoint_of_alternate_setting_one() {
    // Interface 0 alternate 0 (single TT) and 1 (multi TT), each with
    // an interrupt IN status endpoint of its own.
    let configuration = [
        9, 2, 41, 0, 1, 1, 0, 0xe0, 0, //
        9, 4, 0, 0, 1, HUB_CLASS, 0, 1, 0, //
        7, 5, 0x81, 3, 1, 0, 12, //
        9, 4, 0, 1, 1, HUB_CLASS, 0, 2, 0, //
        7, 5, 0x82, 3, 1, 0, 12,
    ];
    assert_eq!(
        hub_status_endpoint(&configuration, 0).unwrap().address,
        0x81
    );
    assert_eq!(
        hub_status_endpoint(&configuration, MULTI_TT_ALTERNATE)
            .unwrap()
            .address,
        0x82
    );
    assert!(hub_status_endpoint(&configuration, 2).is_none());
}

#[test]
fn port_connections_count_after_a_steady_hundred_milliseconds() {
    // A bounce (connected, changed, gone, back) restarts the window.
    let samples = [(true, true), (false, true), (true, true), (true, false)];
    let mut taken = samples
        .iter()
        .copied()
        .chain(core::iter::repeat((true, false)));
    let mut count = 0;
    let mut waits = 0;
    let connected = debounce_with_delay(
        || {
            count += 1;
            Ok(taken.next().unwrap())
        },
        || {
            waits += 1;
            Ok(())
        },
    );
    assert_eq!(connected, Ok(true));
    // Three changing samples, then four steady ones of 25 ms each.
    assert_eq!(count, 3 + (DEBOUNCE_STABLE_US / DEBOUNCE_STEP_US) as usize);
    assert_eq!(waits, count - 1);
    assert_eq!(
        debounce_with_delay(|| Ok((false, false)), || Ok(())),
        Ok(false)
    );
    let mut flapping = false;
    assert_eq!(
        debounce_with_delay(
            || {
                flapping = !flapping;
                Ok((flapping, true))
            },
            || Ok(())
        ),
        Err(BlockError::Unsupported("USB port connection never settled"))
    );
}

#[test]
fn an_empty_root_port_is_skipped_without_waiting() {
    let mut samples = 0;
    let mut waits = 0;
    let settled = debounce_ports_with_delay(
        3,
        0,
        |_| {
            samples += 1;
            Ok((false, false))
        },
        || {
            waits += 1;
            Ok(())
        },
    );
    assert_eq!(settled, [Ok((false, 0)), Ok((false, 0)), Ok((false, 0))]);
    assert_eq!((samples, waits), (3, 0));
}

/// After a controller reset a device may connect only some time later:
/// one that connects 300 ms after the first sample is debounced and found,
/// and an empty port is declared empty after the settle window.
#[test]
fn a_root_port_that_connects_after_the_reset_is_found_within_the_settle_window() {
    let mut elapsed = [0u64; 2];
    let settled = debounce_ports_with_delay(
        2,
        CONNECT_SETTLE_US,
        |port| {
            let connected = port == 0 && elapsed[port] >= 300_000;
            let changed = connected && elapsed[port] == 300_000;
            elapsed[port] += DEBOUNCE_STEP_US;
            Ok((connected, changed))
        },
        || Ok(()),
    );
    assert_eq!(settled[0], Ok((true, 300_000 + DEBOUNCE_STABLE_US)));
    assert_eq!(settled[1], Ok((false, CONNECT_SETTLE_US)));
}

#[test]
fn connected_root_ports_debounce_in_one_window() {
    let mut first = [true; 4];
    let mut waits = 0;
    let settled = debounce_ports_with_delay(
        4,
        0,
        |port| {
            // Port 2 is empty; the others connect with a change.
            let changed = core::mem::replace(&mut first[port], false);
            Ok((port != 2, changed && port != 2))
        },
        || {
            waits += 1;
            Ok(())
        },
    );
    let stable = Ok((true, DEBOUNCE_STABLE_US));
    assert_eq!(
        settled,
        [stable.clone(), stable.clone(), Ok((false, 0)), stable]
    );
    // Three connected ports share one 100 ms window: 4 waits, not 12.
    assert_eq!(waits, (DEBOUNCE_STABLE_US / DEBOUNCE_STEP_US) as usize);
}

#[test]
fn a_bouncing_root_port_settles_late_and_a_flapping_one_times_out_alone() {
    let mut samples = [0usize; 3];
    let settled = debounce_ports_with_delay(
        3,
        0,
        |port| {
            samples[port] += 1;
            let n = samples[port];
            Ok(match port {
                // Connected, then a bounce at the second sample.
                0 => (n != 2, n <= 3),
                // Never steady.
                1 => (n % 2 == 0, true),
                // A device that was there all along, its change cleared.
                _ => (true, false),
            })
        },
        || Ok(()),
    );
    assert_eq!(
        settled[0],
        Ok((true, DEBOUNCE_STABLE_US + 2 * DEBOUNCE_STEP_US))
    );
    assert_eq!(
        settled[1],
        Err(BlockError::Unsupported("USB port connection never settled"))
    );
    assert_eq!(settled[2], Ok((true, DEBOUNCE_STABLE_US)));
    // A failed wait fails every port still settling.
    assert_eq!(
        debounce_ports_with_delay(2, 0, |_| Ok((true, true)), || Err(BlockError::Timeout)),
        [Err(BlockError::Timeout), Err(BlockError::Timeout)]
    );
}

#[test]
fn a_port_probe_reads_as_the_boot_log_line() {
    let probe = PortProbe {
        port: 3,
        connected: true,
        debounce_us: 100_400,
        reset_us: 61_000,
        enumerate_us: 12_999,
    };
    assert_eq!(
        format!("{probe}"),
        "port=3 connected=true debounce_ms=100 reset_ms=61 enumerate_ms=12"
    );
}

#[test]
fn unavailable_debounce_timing_refuses_before_port_sampling_and_failed_wait_is_explicit() {
    for rate in [0, u64::MAX, (u64::MAX / 2) / 60_000_000 + 1] {
        assert!(
            debounce(
                Minimum::retain(crate::minimum_model::refusing(Clock {
                    ticks_per_microsecond: rate
                })),
                || panic!("unavailable clock sampled real port")
            )
            .is_err()
        );
    }
    assert_eq!(
        debounce_with_delay(|| Ok((true, false)), || Err(BlockError::Timeout)),
        Err(BlockError::Timeout)
    );
}

#[test]
fn hub_ports_past_fifteen_route_as_fifteen() {
    assert_eq!(child_route(0, 0, 3), 3);
    assert_eq!(child_route(0x3, 1, 9), 0x93);
    assert_eq!(child_route(0x3, 1, 20), 0xf3);
    assert_eq!(child_route(0, 4, MAX_HUB_PORTS), 0xf_0000);
}

#[test]
fn the_first_setup_waits_for_set_address_recovery() {
    let mut dma = TestRegion::new(16 * 1024 * 8);
    let (mut controller, model) = address_controller(&mut dma, SUCCESS);
    let attachment = Attachment {
        root_port: 1,
        route: 0,
        tier: 0,
        speed: FULL_SPEED,
        translator: None,
        parent: None,
    };
    let slot = controller.address(attachment, &mut dma).unwrap();
    let request = setup(
        0x80,
        GET_DESCRIPTOR,
        u16::from(DEVICE_DESCRIPTOR) << 8,
        0,
        8,
    );
    assert_eq!(controller.control(slot, request, &[]).unwrap().len(), 8);
    let state = model.borrow();
    assert_eq!(state.commands_seen, [ENABLE_SLOT, ADDRESS_DEVICE]);
    assert_eq!(state.setup_seen, [request]);
    assert!(!state.early_setup);
}

#[test]
fn an_address_device_error_does_not_issue_a_setup() {
    let mut dma = TestRegion::new(16 * 1024 * 8);
    let (mut controller, model) = address_controller(&mut dma, 4);
    let attachment = Attachment {
        root_port: 1,
        route: 0,
        tier: 0,
        speed: FULL_SPEED,
        translator: None,
        parent: None,
    };
    assert_eq!(
        controller
            .identify(attachment, String::from("port 1"), &mut dma)
            .unwrap_err(),
        (
            String::from("slot 1 Address Device"),
            BlockError::Device { status: 4 }
        )
    );
    let state = model.borrow();
    // The slot no driver was given is disabled and free for reuse.
    assert_eq!(
        state.commands_seen,
        [ENABLE_SLOT, ADDRESS_DEVICE, reset::DISABLE_SLOT]
    );
    assert!(state.setup_seen.is_empty());
    assert_eq!(controller.slot_ownership[1], reset::Ownership::Vacant);
    assert_eq!(controller.dcbaa.read::<u64>(8), 0);
}

#[test]
fn descriptor_failure_keeps_the_request_and_completion_code() {
    let mut dma = TestRegion::new(16 * 1024 * 8);
    let (mut controller, model) = address_controller(&mut dma, SUCCESS);
    let attachment = Attachment {
        root_port: 1,
        route: 0,
        tier: 0,
        speed: FULL_SPEED,
        translator: None,
        parent: None,
    };
    let slot = controller.address(attachment, &mut dma).unwrap();
    // Model a descriptor transfer failing after successful addressing.
    model.borrow_mut().ready_at = u64::MAX;
    assert_eq!(
        controller
            .enumeration_descriptor(slot, CONFIGURATION_DESCRIPTOR, 2, 9)
            .unwrap_err(),
        (
            String::from("slot 1 EP0 GET_DESCRIPTOR type=2 index=2 length=9"),
            BlockError::Device { status: 4 }
        )
    );
    let state = model.borrow();
    assert_eq!(
        state.setup_seen,
        [setup(0x80, GET_DESCRIPTOR, 0x0202, 0, 9)]
    );
    assert_eq!(
        state.commands_seen,
        [ENABLE_SLOT, ADDRESS_DEVICE, RESET_ENDPOINT, SET_DEQUEUE]
    );
}

#[test]
fn all_configuration_indices_travel_through_ep0_and_selection_uses_the_wire_value() {
    use configuration::tests::{device, ecm, vendor};
    let mut dma = TestRegion::new(128 * 1024 * 8);
    let (mut controller, model) =
        unconfigured_cancellation_controller_with_configurations(&mut dma, &[]);
    let originals = vec![vendor(9), ecm(42), vendor(5)];
    model.borrow_mut().descriptors = originals
        .iter()
        .enumerate()
        .map(|(index, bytes)| (CONFIGURATION_DESCRIPTOR, index as u8, bytes.clone()))
        .collect();
    let inventory = controller.configurations(1, 3).unwrap();
    assert_eq!(inventory, originals);
    let mut device = device(inventory);
    configuration::select(&mut device).unwrap();
    assert_eq!(device.configuration_value(), 42);
    assert_eq!(
        model
            .borrow()
            .setup
            .iter()
            .map(|(request, _)| request[2])
            .collect::<Vec<_>>(),
        vec![0, 0, 1, 1, 2, 2]
    );
    let all = interfaces(&device.configuration);
    let plan = configuration::plan(&device, &all[0], &all).unwrap();
    assert!(
        controller
            .configure(1, &plan.endpoints(), None, &mut dma)
            .is_err()
    );
    assert!(model.borrow().seen.is_empty());
    controller
        .set_configuration(1, device.configuration_value())
        .unwrap();
    controller
        .configure(1, &plan.endpoints(), None, &mut dma)
        .unwrap();
    assert_eq!(controller.current_configuration(1), Ok(Some(42)));
    assert_eq!(model.borrow().configuration_seen_before_endpoints, vec![42]);
    assert_eq!(
        model.borrow().setup.last().unwrap().0,
        setup(0, SET_CONFIGURATION, 42, 0, 0)
    );
    let requests = model.borrow().setup.len();
    controller.set_configuration(1, 42).unwrap(); // Another class binder must not reset the device.
    assert_eq!(model.borrow().setup.len(), requests);
    assert!(controller.set_configuration(1, 9).is_err());
    assert!(
        controller
            .control(1, setup(0, SET_CONFIGURATION, 42, 0, 0), &[])
            .is_err()
    );
    assert_eq!(model.borrow().setup.len(), requests);
    assert_eq!(controller.current_configuration(1), Ok(Some(42)));
}

#[test]
fn descriptor_inventory_refuses_truncation_duplicate_values_and_silent_size_caps() {
    use configuration::tests::vendor;
    for scenario in 0..5 {
        let mut dma = TestRegion::new(128 * 1024 * 8);
        let (mut controller, model) = unconfigured_cancellation_controller(&mut dma);
        let mut bytes = vendor(7);
        match scenario {
            0 => {
                bytes.pop();
            }
            1 => {
                bytes[2..4].copy_from_slice(&4097u16.to_le_bytes());
            }
            2 => {
                bytes[5] = 0;
            }
            3 => {
                bytes[2..4].copy_from_slice(&8u16.to_le_bytes());
            }
            _ => {}
        }
        model.borrow_mut().descriptors = vec![
            (CONFIGURATION_DESCRIPTOR, 0, bytes.clone()),
            (CONFIGURATION_DESCRIPTOR, 1, bytes),
        ];
        let error = controller.configurations(1, 2).unwrap_err();
        assert!(matches!(error.1, BlockError::Unsupported(_)), "{error:?}");
        assert_eq!(controller.current_configuration(1), Ok(Some(0)));
        assert!(model.borrow().seen.is_empty());
        if scenario == 1 || scenario == 3 {
            assert_eq!(model.borrow().setup.len(), 1); // Reject before requesting capped/full bytes.
        }
    }
}

#[test]
fn configuration_inventory_has_an_explicit_aggregate_resource_limit() {
    use configuration::tests::vendor;
    let mut dma = TestRegion::new(128 * 1024 * 8);
    let (mut controller, model) = unconfigured_cancellation_controller(&mut dma);
    for index in 0..17 {
        let mut bytes = vendor(index + 1);
        while bytes.len() < CONTROL_BYTES {
            bytes.extend_from_slice(&[2, 0x25]); // Preserved unknown class descriptors.
        }
        bytes[2..4].copy_from_slice(&(CONTROL_BYTES as u16).to_le_bytes());
        model
            .borrow_mut()
            .descriptors
            .push((CONFIGURATION_DESCRIPTOR, index, bytes));
    }
    assert_eq!(
        controller.configurations(1, 17).unwrap_err().1,
        BlockError::Unsupported("USB configuration inventory exceeds 64 KiB")
    );
    assert_eq!(model.borrow().setup.len(), 34);
    assert_eq!(controller.current_configuration(1), Ok(Some(0)));
    assert!(model.borrow().seen.is_empty());
}
