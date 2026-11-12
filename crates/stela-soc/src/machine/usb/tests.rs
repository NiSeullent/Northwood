// SPDX-License-Identifier: GPL-2.0-or-later

use super::*;
use crate::pmgr::CLOCK_GATED;

/// Diagnostic-only model without mapped registers or a guest DMA user.
fn add_port(devices: &mut AppleDevices, number: u32) -> (DeviceId, DeviceId) {
    let phy = devices.bus.add(Box::new(AtcPhy::new()));
    let dart = devices.bus.add(Box::new(Dart::new(1, None)));
    let mut controller = Dwc3::new([AicInput::new(0, 1); 4]);
    controller.set_power(false, true, &mut Access::new(0, 0, &mut devices.latch));
    let device = devices.bus.add(Box::new(controller));
    let connector = devices.typec_port(number);
    devices.usb_masters.push(UsbMaster {
        device,
        phy,
        phy_changed: false,
        dart,
        stream: 0,
        session: UsbSession::new(),
        port: UsbPort::of(&connector),
        number,
        connector,
        peripheral: None,
        next_poll: 0,
        attached: false,
        epoch: 0,
        retention: true,
        session_reset: false,
    });
    (device, phy)
}

#[test]
fn unavailable_device_port_reports_only_its_controller_and_phy_power_gates() {
    let mut devices = AppleDevices::new();
    let (controller, phy) = add_port(&mut devices, 1);
    let (hub_controller, _) = add_port(&mut devices, 2);
    devices.attach_input(2, UsbCables::default()).unwrap();

    let baseline = devices.usb_diagnostics();
    assert_eq!(
        baseline[0],
        "port=1 attached=false unavailable=DWC3 power domain is off: USB2 USBCTL is not running out of isolation"
    );
    assert!(baseline[1].starts_with("port=2 attached=true hub=["));

    let link = |owner, device, domains| PowerClient {
        owner,
        device,
        domains,
        resources: String::new(),
        retention: true,
    };
    let state = |power| DomainState {
        power,
        disabled: false,
        reset: false,
    };
    devices.power_clients = vec![
        link(
            PowerOwner::Usb(0),
            controller,
            vec![
                (0x1000, state(ACTIVE)),
                (0x1004, state(POWER_GATED)),
                (0x1008, state(POWER_GATED)),
            ],
        ),
        link(
            PowerOwner::Phy(0),
            phy,
            vec![(0x2000, state(CLOCK_GATED)), (0x2004, state(POWER_GATED))],
        ),
        link(
            PowerOwner::Usb(1),
            hub_controller,
            vec![(0x3000, state(POWER_GATED))],
        ),
    ];
    let report = devices.usb_diagnostics();
    assert_eq!(
        report[0].strip_prefix(&baseline[0]),
        Some(" pmgr_power_gated=[controller=0x1004,0x1008 phy=0x2004]")
    );
    assert_eq!(report[1], baseline[1]);
}
