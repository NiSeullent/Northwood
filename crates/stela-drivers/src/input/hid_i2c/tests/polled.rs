//! A device whose only interrupt is an APIC interrupt, read on a timer.
use super::*;

fn polled() -> (Device<Model, Polled, Time>, Rc<RefCell<Hardware>>) {
    let time = Rc::new(Cell::new(0));
    let h = Rc::new(RefCell::new(Hardware::new(time.clone(), 0x06cb, 0x1234)));
    h.borrow_mut().idle_zero = true;
    let device = Device::new(
        Model(h.clone()),
        Polled(h.clone()),
        Time(time.clone()),
        Rc::new(MinimumTime(time)),
        Address::SevenBit(0x2c),
        0x20,
        100,
    )
    .unwrap();
    (device, h)
}

#[test]
fn a_polled_device_resets_after_the_delay_and_reads_until_empty() {
    let (mut d, h) = polled();
    // A report left from before the reset is read and discarded.
    h.borrow_mut().input.push_back(Hardware::frame(&[17, 1, 2]));
    d.initialize().unwrap();
    assert!(d.is_ready());
    assert_eq!(d.reset_completion(), Some(ResetCompletion::PolledDelay));
    assert_eq!(d.discarded_before_start(), 1);
    assert_eq!(h.borrow().reset_count, 1);
    assert!(h.borrow().input.is_empty());
    assert_eq!(d.report_descriptor(), REPORT_DESCRIPTOR);
}

#[test]
fn an_empty_polled_register_is_no_report_and_data_is_delivered() {
    let (mut d, h) = polled();
    d.initialize().unwrap();
    assert_eq!(d.poll().unwrap(), None);
    assert!(d.is_ready());
    h.borrow_mut().input.push_back(Hardware::frame(&[17, 5, 6]));
    assert_eq!(d.poll().unwrap(), Some(InputReport::Data(vec![17, 5, 6])));
    assert_eq!(d.poll().unwrap(), None);
}

// Independent Precision Touchpad fixture with a Digitizer collection, a
// Configuration collection, and the optional relative Mouse collection.
const TOUCHPAD_REPORT: &[u8] = &[
    0x05, 0x0d, 0x09, 0x05, 0xa1, 0x01, 0x85, 3, 0x09, 0x22, 0xa1, 0x02, 0x09, 0x42, 0x15, 0x00,
    0x25, 0x01, 0x75, 8, 0x95, 1, 0x81, 0x02, 0x05, 0x01, 0x09, 0x30, 0x09, 0x31, 0x25, 127, 0x95,
    2, 0x81, 0x02, 0xc0, 0xc0,
];
const CONFIGURATION_START: &[u8] = &[
    0x05, 0x0d, 0x09, 0x0e, 0xa1, 0x01, 0x85, 0x23, 0x09, 0x22, 0xa1, 0x02, 0x09, 0x52, 0x15, 0,
    0x25, 10, 0x75, 8, 0x95, 1, 0xb1, 0x02,
];
const MOUSE_REPORT: &[u8] = &[
    0x05, 0x01, 0x09, 0x02, 0xa1, 0x01, 0x85, 5, 0x05, 0x09, 0x09, 1, 0x15, 0x00, 0x25, 0x01, 0x75,
    8, 0x95, 1, 0x81, 0x02, 0x05, 0x01, 0x09, 0x30, 0x09, 0x31, 0x15, 0x81, 0x25, 0x7f, 0x95, 2,
    0x81, 0x06, 0xc0,
];

fn precision_pad(
    mouse: bool,
    shared_feature: bool,
) -> (Device<Model, Polled, Time>, Rc<RefCell<Hardware>>) {
    let time = Rc::new(Cell::new(0));
    let mut hardware = Hardware::new(time.clone(), 0x06cb, 0x1234);
    hardware.idle_zero = true;
    hardware.descriptor_register = 0x2c;
    hardware.feature = vec![3];
    hardware.report_desc = [
        TOUCHPAD_REPORT,
        CONFIGURATION_START,
        if shared_feature {
            &[0x09, 0x57, 0xb1, 0x02]
        } else {
            &[]
        },
        &[0xc0, 0xc0],
        if mouse { MOUSE_REPORT } else { &[] },
    ]
    .concat();
    hardware.desc[4..6].copy_from_slice(&(hardware.report_desc.len() as u16).to_le_bytes());
    let h = Rc::new(RefCell::new(hardware));
    let device = Device::new(
        Model(h.clone()),
        Polled(h.clone()),
        Time(time.clone()),
        Rc::new(MinimumTime(time)),
        Address::SevenBit(0x2c),
        0x2c,
        100,
    )
    .unwrap()
    .initialize_owned()
    .unwrap();
    (device, h)
}

#[test]
fn precision_pad_selects_the_mouse_report_before_polled_input() {
    let (device, h) = precision_pad(true, false);
    assert_eq!(
        device.reset_completion(),
        Some(ResetCompletion::PolledDelay)
    );
    assert_eq!(h.borrow().feature, [3]);
    let mut input = HidInput::new(device, String::from("precision fixture")).unwrap();
    {
        let mut hardware = h.borrow_mut();
        let command = hardware.commands.last().unwrap();
        assert_eq!(
            command,
            &[0x24, 0, 0x3f, SET_REPORT, 0x23, 0x25, 0, 4, 0, 0x23, 0]
        );
        let report = if hardware.feature == [0] {
            Hardware::frame(&[5, 1, 7, 0xfd])
        } else {
            Hardware::frame(&[3, 1, 40, 50])
        };
        hardware.input.push_back(report);
        hardware.input.push_back(Hardware::frame(&[5, 0, 0, 0]));
    }
    let mut events = Vec::new();
    input.poll(&mut events).unwrap();
    assert_eq!(
        events,
        [
            Input::Motion { dx: 7, dy: -3 },
            Input::MouseButtons(1),
            Input::MouseButtons(0),
        ]
    );
    assert!(input.device().is_ready());
}

#[test]
fn precision_pad_leaves_digitizer_only_or_shared_feature_devices_alone() {
    for (mouse, shared_feature) in [(false, false), (true, true)] {
        let (device, h) = precision_pad(mouse, shared_feature);
        let mut input = HidInput::new(device, String::from("precision fixture")).unwrap();
        assert_eq!(h.borrow().feature, [3]);
        assert!(
            !h.borrow()
                .commands
                .iter()
                .any(|command| command[3] == SET_REPORT)
        );
        h.borrow_mut()
            .input
            .push_back(Hardware::frame(&[3, 1, 40, 50]));
        let mut events = Vec::new();
        input.poll(&mut events).unwrap();
        assert!(
            events.is_empty(),
            "digitizer coordinates are not mouse motion"
        );
        assert_eq!(input.last_report(), &[3, 1, 40, 50]);
    }
}

#[test]
fn precision_pad_keeps_transport_on_an_uncertain_mode_write() {
    let (device, h) = precision_pad(true, false);
    h.borrow_mut().fail_after_opcode = Some(SET_REPORT);
    let refusal = HidInput::new(device, String::from("precision fixture"))
        .err()
        .unwrap();
    assert!(matches!(refusal.error, RefusalCause::Transport(_)));
    assert!(refusal.device().needs_reset());
    assert!(h.borrow().masked);
}
