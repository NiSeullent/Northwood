//! Independent HID-I2C peripheral model: addresses/registers, pin state,
//! reset FIFO, power-dependent behavior, commands and wire reports.
use super::*;
use alloc::{collections::VecDeque, rc::Rc, vec};
use core::cell::{Cell, RefCell};
mod minimum;
mod ownership;
mod polled;

// A complete vendor-page HID collection: ID17, two eight-bit input fields.
const REPORT_DESCRIPTOR: [u8; 23] = [
    0x06, 0, 0xff, 0x09, 1, 0xa1, 1, 0x85, 17, 0x15, 0, 0x26, 0xff, 0, 0x75, 8, 0x95, 2, 0x09, 2,
    0x81, 2, 0xc0,
];

#[derive(Clone)]
struct Model(Rc<RefCell<Hardware>>);
#[derive(Clone)]
struct Pin(Rc<RefCell<Hardware>>);
struct Time(Rc<Cell<u64>>);
struct MinimumTime(Rc<Cell<u64>>);
// SAFETY: explicit peripheral model only. Waiting advances this same model's
// observable time by the complete interval; it grants no native calibration.
unsafe impl Minimum for MinimumTime {
    fn check(&self) -> Result<(), power::Error> {
        Ok(())
    }
    fn wait_us(&self, us: u64) -> Result<(), power::Error> {
        self.0.set(self.0.get().checked_add(us).unwrap());
        Ok(())
    }
}
struct Hardware {
    time: Rc<Cell<u64>>,
    descriptor_register: u16,
    desc: [u8; 30],
    report_desc: Vec<u8>,
    input: VecDeque<Vec<u8>>,
    feature: Vec<u8>,
    commands: Vec<Vec<u8>>,
    power_on: bool,
    reset_count: usize,
    mask_count: usize,
    masked: bool,
    force_irq_low: bool,
    no_reset_ack: bool,
    repeated_input: Option<Vec<u8>>,
    fail_irq: bool,
    fail_mask: bool,
    fail_next: bool,
    address_nack_next: bool,
    truncate_get: Option<u16>,
    wrong_id: bool,
    writes: Vec<Vec<u8>>,
    reads: usize,
    fail_after_opcode: Option<u8>,
    /// No pin: an empty input register reads as a zero length.
    idle_zero: bool,
}
fn descriptor(vendor: u16, product: u16) -> [u8; 30] {
    let mut bytes = [0; 30];
    for (index, value) in [
        30,
        0x100,
        REPORT_DESCRIPTOR.len() as u16,
        0x21,
        0x22,
        32,
        0x23,
        32,
        0x24,
        0x25,
        vendor,
        product,
        1,
    ]
    .into_iter()
    .enumerate()
    {
        bytes[index * 2..index * 2 + 2].copy_from_slice(&value.to_le_bytes());
    }
    bytes
}
impl Hardware {
    fn new(time: Rc<Cell<u64>>, vendor: u16, product: u16) -> Self {
        Self {
            time,
            descriptor_register: 0x20,
            desc: descriptor(vendor, product),
            report_desc: REPORT_DESCRIPTOR.to_vec(),
            input: VecDeque::new(),
            feature: vec![0x11, 0x22],
            commands: Vec::new(),
            power_on: false,
            reset_count: 0,
            mask_count: 0,
            masked: false,
            force_irq_low: false,
            no_reset_ack: false,
            repeated_input: None,
            fail_irq: false,
            fail_mask: false,
            fail_next: false,
            address_nack_next: false,
            truncate_get: None,
            wrong_id: false,
            writes: Vec::new(),
            reads: 0,
            fail_after_opcode: None,
            idle_zero: false,
        }
    }
    fn frame(body: &[u8]) -> Vec<u8> {
        let mut out = (body.len() as u16 + 2).to_le_bytes().to_vec();
        out.extend_from_slice(body);
        out
    }
    fn command(&mut self, bytes: &[u8], receive: &mut [u8]) {
        assert!(self.masked, "I/O before actual IRQ masking");
        self.commands.push(bytes.to_vec());
        let mut index = 4;
        let kind = bytes[2] >> 4;
        let mut id = bytes[2] & 15;
        if id == 15 {
            id = bytes[index];
            index += 1;
        }
        match bytes[3] {
            SET_POWER => {
                assert_eq!(kind, 0);
                self.power_on = id == 0;
            }
            RESET => {
                assert!(self.power_on);
                assert_eq!(id, 0);
                self.reset_count += 1;
                if !self.no_reset_ack {
                    self.input.push_back(vec![0, 0]);
                }
            }
            GET_REPORT => {
                assert!(self.power_on);
                assert!((1..=3).contains(&kind));
                assert_eq!(&bytes[index..index + 2], &0x25_u16.to_le_bytes());
                let mut body = Vec::new();
                if id != 0 {
                    body.push(if self.wrong_id {
                        id.wrapping_add(1)
                    } else {
                        id
                    });
                }
                body.extend_from_slice(&self.feature);
                let mut frame = Self::frame(&body);
                if let Some(length) = self.truncate_get {
                    frame[..2].copy_from_slice(&length.to_le_bytes());
                }
                for (target, source) in receive.iter_mut().zip(frame) {
                    *target = source;
                }
            }
            SET_REPORT => {
                assert!(self.power_on);
                assert!((2..=3).contains(&kind));
                assert_eq!(&bytes[index..index + 2], &0x25_u16.to_le_bytes());
                index += 2;
                let length = usize::from(u16::from_le_bytes([bytes[index], bytes[index + 1]]));
                assert_eq!(bytes.len() - index, length);
                index += 2;
                if id != 0 {
                    assert_eq!(bytes[index], id);
                    index += 1;
                }
                self.feature = bytes[index..].to_vec();
            }
            _ => panic!("unknown HID command"),
        }
    }
}
fn bus_error(kind: i2c::AbortKind) -> i2c::Error {
    i2c::Error {
        kind: i2c::ErrorKind::Aborted { kind, source: 1 },
        recovery: i2c::Recovery::Disabled,
    }
}
impl Bus for Model {
    fn transaction(
        &mut self,
        address: Address,
        operations: &mut [Operation<'_>],
    ) -> Result<(), i2c::Error> {
        assert_eq!(address, Address::SevenBit(0x2c));
        let mut h = self.0.borrow_mut();
        assert!(h.masked, "I2C before actual IRQ route masking");
        if core::mem::take(&mut h.address_nack_next) {
            return Err(bus_error(i2c::AbortKind::AddressNack));
        }
        if core::mem::take(&mut h.fail_next) {
            return Err(bus_error(i2c::AbortKind::DataNack));
        }
        match operations {
            [Operation::Read(output)] => {
                assert!(h.masked);
                assert!(h.power_on);
                assert!(!h.force_irq_low, "read without actual asserted device pin");
                h.reads += 1;
                let frame = h
                    .input
                    .pop_front()
                    .or_else(|| h.repeated_input.clone())
                    .or_else(|| h.idle_zero.then(|| vec![0, 0]))
                    .expect("read without device output");
                assert_eq!(
                    output.len(),
                    usize::from(u16::from_le_bytes([h.desc[10], h.desc[11]]))
                );
                output.fill(0);
                for (target, source) in output.iter_mut().zip(frame) {
                    *target = source;
                }
            }
            [Operation::Write(register), Operation::Read(output)] if register.len() == 2 => {
                let reg = u16::from_le_bytes([register[0], register[1]]);
                match reg {
                    register if register == h.descriptor_register => {
                        assert_eq!(output.len(), 30);
                        output.copy_from_slice(&h.desc);
                    }
                    0x21 => {
                        assert!(h.reset_count > 0);
                        assert!(
                            h.input.is_empty(),
                            "report descriptor published before reset ACK consumed"
                        );
                        output.copy_from_slice(&h.report_desc);
                    }
                    _ => panic!("unknown register {reg:x}"),
                }
            }
            [Operation::Write(command), Operation::Read(output)] => {
                assert_eq!(&command[..2], &0x24_u16.to_le_bytes());
                h.command(command, output);
            }
            [Operation::Write(bytes)] => {
                h.writes.push(bytes.to_vec());
                if bytes[..2] == 0x24_u16.to_le_bytes() {
                    h.command(bytes, &mut []);
                    if h.fail_after_opcode == Some(bytes[3]) {
                        h.fail_after_opcode = None;
                        return Err(bus_error(i2c::AbortKind::DataNack));
                    }
                } else {
                    assert_eq!(&bytes[..2], &0x23_u16.to_le_bytes());
                    assert_eq!(
                        usize::from(u16::from_le_bytes([bytes[2], bytes[3]])),
                        bytes.len() - 2
                    );
                    h.feature = bytes[4..].to_vec();
                }
            }
            _ => panic!("unexpected I2C operation"),
        }
        Ok(())
    }
}
impl Interrupt for Pin {
    fn mask_and_drain(&mut self) -> Result<(), InterruptError> {
        let mut h = self.0.borrow_mut();
        h.mask_count += 1;
        if h.fail_mask {
            return Err(InterruptError::Hardware(1));
        }
        h.masked = true;
        Ok(())
    }
    fn asserted(&mut self) -> Result<bool, InterruptError> {
        let h = self.0.borrow();
        if h.fail_irq {
            return Err(InterruptError::Hardware(2));
        }
        Ok(!h.force_irq_low && (!h.input.is_empty() || h.repeated_input.is_some()))
    }
}
/// A device without a readable pin: always due, and nothing to mask.
struct Polled(Rc<RefCell<Hardware>>);
impl Interrupt for Polled {
    fn mask_and_drain(&mut self) -> Result<(), InterruptError> {
        self.0.borrow_mut().masked = true;
        Ok(())
    }
    fn asserted(&mut self) -> Result<bool, InterruptError> {
        Ok(true)
    }
    fn polled(&self) -> bool {
        true
    }
}
impl Timer for Time {
    fn start(&self) -> u64 {
        !self.now_us().rotate_left(17)
    }
    fn elapsed_us(&self, marker: u64) -> u64 {
        self.now_us().wrapping_sub((!marker).rotate_right(17))
    }
    fn now_us(&self) -> u64 {
        self.0.get()
    }
    fn relax(&self) {
        self.0.set(self.0.get().wrapping_add(1));
    }
}
fn device(vendor: u16, product: u16) -> (Device<Model, Pin, Time>, Rc<RefCell<Hardware>>) {
    let time = Rc::new(Cell::new(0));
    let h = Rc::new(RefCell::new(Hardware::new(time.clone(), vendor, product)));
    (
        Device::new(
            Model(h.clone()),
            Pin(h.clone()),
            Time(time.clone()),
            Rc::new(MinimumTime(time)),
            Address::SevenBit(0x2c),
            0x20,
            100,
        )
        .unwrap(),
        h,
    )
}
fn ready() -> (Device<Model, Pin, Time>, Rc<RefCell<Hardware>>) {
    let (mut d, h) = device(0x06cb, 0x1234);
    d.initialize().unwrap();
    (d, h)
}
#[test]
fn reset_requires_irq_and_zero_report_before_descriptor_or_ready() {
    let (mut d, h) = ready();
    assert!(d.is_ready());
    assert_eq!(h.borrow().reset_count, 1);
    assert_eq!(h.borrow().reads, 1);
    assert_eq!(d.report_descriptor(), REPORT_DESCRIPTOR);
    assert_eq!(d.descriptor().unwrap().vendor, 0x06cb);
    assert_eq!(
        h.borrow()
            .commands
            .iter()
            .filter(|c| c[3] == SET_POWER)
            .count(),
        2
    );
    assert!(d.poll().unwrap().is_none());
    let reads = h.borrow().reads;
    h.borrow_mut().force_irq_low = true;
    h.borrow_mut().input.push_back(Hardware::frame(&[1, 2, 3]));
    assert!(d.poll().unwrap().is_none());
    assert_eq!(h.borrow().reads, reads);
}
#[test]
fn descriptor_identity_never_substitutes_for_reset_acknowledgement() {
    for force_low in [false, true] {
        let (mut d, h) = device(0x06cb, 0x1234);
        h.borrow_mut().no_reset_ack = !force_low;
        h.borrow_mut().force_irq_low = force_low;
        assert_eq!(d.initialize().unwrap_err().kind, ErrorKind::ResetTimeout);
        assert!(!d.is_ready());
        assert!(d.needs_reset());
        assert!(d.report_descriptor().is_empty());
        let writes = h.borrow().writes.len();
        assert_eq!(d.initialize().unwrap_err().kind, ErrorKind::NeedsReset);
        assert_eq!(h.borrow().writes.len(), writes);
    }
}
#[test]
fn raw_input_keeps_report_ids_payload_and_empty_interrupt_distinct() {
    let (mut d, h) = ready();
    h.borrow_mut()
        .input
        .push_back(Hardware::frame(&[17, 0xaa, 0xbb]));
    assert_eq!(
        d.poll().unwrap(),
        Some(InputReport::Data(vec![17, 0xaa, 0xbb]))
    );
    h.borrow_mut().input.push_back(vec![2, 0]);
    assert_eq!(d.poll().unwrap(), Some(InputReport::Empty));
    h.borrow_mut().input.push_back(vec![0, 0]);
    assert_eq!(d.poll().unwrap_err().kind, ErrorKind::UnexpectedReset);
    assert!(d.needs_reset());
}
#[test]
fn command_report_ids_zero_short_and_extended_preserve_exact_wire_layout() {
    for id in [0, 1, 14, 15, 255] {
        let (mut d, h) = ready();
        d.set_report(ReportType::Feature, id, &[5, 6]).unwrap();
        assert_eq!(h.borrow().feature, vec![5, 6]);
        let got = d.get_report(ReportType::Feature, id, 16).unwrap();
        let expected = if id == 0 { vec![5, 6] } else { vec![id, 5, 6] };
        assert!(got.is_complete());
        assert_eq!(got.data, expected);
        let commands = h.borrow().commands.clone();
        let command = commands.iter().find(|c| c[3] == SET_REPORT).unwrap();
        assert_eq!(command[2], 0x30 | id.min(15));
        let offset = if id >= 15 {
            assert_eq!(command[4], id);
            5
        } else {
            4
        };
        assert_eq!(&command[offset..offset + 2], &[0x25, 0]);
        d.output(id, &[7, 8]).unwrap();
        let last = h.borrow().writes.last().unwrap().clone();
        assert_eq!(&last[..2], &[0x23, 0]);
        assert_eq!(
            &last[4..],
            if id == 0 { vec![7, 8] } else { vec![id, 7, 8] }
        );
    }
}
#[test]
fn malformed_or_wrong_id_reports_never_escape_as_valid_data() {
    for length in [0, 1] {
        let (mut d, h) = ready();
        h.borrow_mut().truncate_get = Some(length);
        assert!(matches!(
            d.get_report(ReportType::Feature, 3, 16).unwrap_err().kind,
            ErrorKind::ReportLength { .. }
        ));
        assert!(d.needs_reset());
    }
    let (mut d, h) = ready();
    h.borrow_mut().wrong_id = true;
    assert_eq!(
        d.get_report(ReportType::Feature, 3, 16).unwrap_err().kind,
        ErrorKind::ReportId {
            expected: 3,
            actual: 4
        }
    );
    let (mut d, h) = ready();
    h.borrow_mut().input.push_back(vec![33, 0]);
    assert!(matches!(
        d.poll().unwrap_err().kind,
        ErrorKind::ReportLength {
            actual: 33,
            maximum: 32
        }
    ));
}
#[test]
fn invalid_user_report_requests_do_not_publish_or_break_the_ready_device() {
    let (mut d, h) = ready();
    let count = h.borrow().commands.len();
    assert_eq!(
        d.set_report(ReportType::Input, 1, &[1]).unwrap_err().kind,
        ErrorKind::UnsupportedReportType
    );
    assert_eq!(
        d.get_report(ReportType::Feature, 1, 0).unwrap_err().kind,
        ErrorKind::InvalidConfiguration
    );
    assert!(matches!(
        d.output(1, &[0; 32]).unwrap_err().kind,
        ErrorKind::ReportLength { .. }
    ));
    assert_eq!(h.borrow().commands.len(), count);
    assert!(d.is_ready());
}
#[test]
fn sleep_stop_and_restart_have_acknowledged_power_effects() {
    let (mut d, h) = ready();
    d.sleep().unwrap();
    assert_eq!(d.power_state(), PowerState::Sleep);
    assert!(!h.borrow().power_on);
    assert_eq!(d.poll().unwrap_err().kind, ErrorKind::NotReady);
    d.wake().unwrap();
    assert!(d.is_ready());
    assert!(h.borrow().power_on);
    d.stop().unwrap();
    assert_eq!(d.power_state(), PowerState::Stopped);
    assert!(!h.borrow().power_on);
    d.initialize().unwrap();
    assert_eq!(h.borrow().reset_count, 2);
    assert!(d.is_ready());
}
#[test]
fn elan_does_not_receive_the_reset_breaking_second_power_on() {
    let (mut d, h) = device(0x04f3, 0x1234);
    d.initialize().unwrap();
    assert_eq!(
        h.borrow()
            .commands
            .iter()
            .filter(|c| c[3] == SET_POWER)
            .count(),
        1
    );
    h.borrow_mut().input.push_back(vec![0xff, 0xff]);
    assert_eq!(d.poll().unwrap(), Some(InputReport::SpuriousInterrupt));
    assert!(d.is_ready());
}
#[test]
fn source_checked_resume_profiles_apply_distinct_physical_effects() {
    for (vendor, product, reset, delay, no_sleep) in [
        (0x044e, 1, true, false, false),
        (0x06cb, 0x7a13, true, false, false),
        (0x27c6, 0x0d42, false, true, false),
        (0x0488, 0x1063, false, false, true),
    ] {
        let (mut d, h) = device(vendor, product);
        d.initialize().unwrap();
        d.sleep().unwrap();
        assert_eq!(h.borrow().power_on, no_sleep);
        assert_eq!(
            d.power_state(),
            if no_sleep {
                PowerState::SuspendedAwake
            } else {
                PowerState::Sleep
            }
        );
        let before = h.borrow().time.get();
        d.wake().unwrap();
        assert_eq!(h.borrow().reset_count, if reset { 2 } else { 1 });
        if delay {
            assert!(h.borrow().time.get() - before >= GOODIX_RESUME_DELAY_US + WAKE_DELAY_US);
        }
    }
}
#[test]
fn interrupt_and_partial_bus_failures_quarantine_without_automatic_replay() {
    let (mut d, h) = device(0x06cb, 1);
    h.borrow_mut().fail_mask = true;
    assert_eq!(
        d.initialize().unwrap_err().recovery,
        Recovery::Failed(InterruptError::Hardware(1))
    );
    assert!(h.borrow().commands.is_empty());
    let (mut d, h) = ready();
    h.borrow_mut().fail_next = true;
    let writes = h.borrow().writes.len();
    assert!(matches!(
        d.set_report(ReportType::Feature, 1, &[1]).unwrap_err().kind,
        ErrorKind::Bus(_)
    ));
    assert!(d.needs_reset());
    assert_eq!(h.borrow().writes.len(), writes);
    let (mut d, h) = ready();
    h.borrow_mut().fail_irq = true;
    assert!(matches!(
        d.poll().unwrap_err().kind,
        ErrorKind::Interrupt(_)
    ));
    assert!(d.needs_reset());
}
#[test]
fn safe_address_nack_wakeup_retry_and_qtec_extra_power_are_real_commands() {
    let (mut d, h) = device(0x6243, 1);
    h.borrow_mut().address_nack_next = true;
    d.initialize().unwrap();
    assert_eq!(
        h.borrow()
            .commands
            .iter()
            .filter(|c| c[3] == SET_POWER)
            .count(),
        3
    );
    assert!(h.borrow().time.get() >= 3 * WAKE_DELAY_US + ADDRESS_WAKE_DELAY_US);
}
#[test]
fn restart_identity_change_is_explicit() {
    let (mut d, h) = ready();
    d.stop().unwrap();
    h.borrow_mut().desc[22..24].copy_from_slice(&99_u16.to_le_bytes());
    assert_eq!(d.initialize().unwrap_err().kind, ErrorKind::IdentityChanged);
}

#[test]
fn late_errors_after_side_effects_never_replay_or_publish_ready() {
    let (mut d, h) = ready();
    h.borrow_mut().fail_after_opcode = Some(SET_REPORT);
    let count = h.borrow().commands.len();
    assert!(matches!(
        d.set_report(ReportType::Feature, 1, &[7, 8])
            .unwrap_err()
            .kind,
        ErrorKind::Bus(_)
    ));
    assert_eq!(h.borrow().feature, vec![7, 8]);
    assert_eq!(h.borrow().commands.len(), count + 1);
    assert_eq!(
        d.set_report(ReportType::Feature, 1, &[7, 8])
            .unwrap_err()
            .kind,
        ErrorKind::NeedsReset
    );
    assert_eq!(h.borrow().commands.len(), count + 1);
    let (mut d, h) = device(0x06cb, 1);
    h.borrow_mut().fail_after_opcode = Some(RESET);
    assert!(matches!(
        d.initialize().unwrap_err().kind,
        ErrorKind::Bus(_)
    ));
    assert_eq!(h.borrow().reset_count, 1);
    assert_eq!(h.borrow().reads, 0);
    assert!(!h.borrow().input.is_empty());
    assert!(d.report_descriptor().is_empty());
    assert!(d.needs_reset());
}

#[test]
fn descriptor_bounds_fail_before_reset_and_missing_output_can_use_set_report() {
    for (offset, value, expected) in [
        (0usize, 29u16, ErrorKind::DescriptorLength(29)),
        (2, 0x101, ErrorKind::Version(0x101)),
        (4, 0, ErrorKind::ReportDescriptorLength(0)),
        (4, 4097, ErrorKind::ReportDescriptorLength(4097)),
        (10, 1, ErrorKind::BufferLength(1)),
        (10, 16385, ErrorKind::BufferLength(16385)),
        (14, 1, ErrorKind::BufferLength(1)),
    ] {
        let (mut d, h) = device(0x06cb, 1);
        h.borrow_mut().desc[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
        assert_eq!(d.initialize().unwrap_err().kind, expected);
        assert_eq!(h.borrow().reset_count, 0);
    }
    let (mut d, h) = device(0x06cb, 1);
    h.borrow_mut().desc[14..16].copy_from_slice(&0_u16.to_le_bytes());
    d.initialize().unwrap();
    let count = h.borrow().commands.len();
    assert_eq!(
        d.output(1, &[2]).unwrap_err().kind,
        ErrorKind::NoOutputRegister
    );
    assert_eq!(h.borrow().commands.len(), count);
    assert!(d.is_ready());
    d.set_report(ReportType::Output, 1, &[2]).unwrap();
    assert_eq!(h.borrow().feature, vec![2]);
}

#[test]
fn bounded_feature_reads_report_truncation_without_treating_valid_device_as_broken() {
    let (mut d, h) = ready();
    h.borrow_mut().feature = (0..64).collect();
    let partial = d.get_report(ReportType::Feature, 17, 16).unwrap();
    assert!(!partial.is_complete());
    assert_eq!(partial.reported_length, 65);
    assert_eq!(
        partial.data,
        core::iter::once(17).chain(0..15).collect::<Vec<_>>()
    );
    assert!(d.is_ready());
    let full = d.get_report(ReportType::Feature, 17, 65).unwrap();
    assert!(full.is_complete());
    assert_eq!(
        full.data,
        core::iter::once(17).chain(0..64).collect::<Vec<_>>()
    );
}

#[test]
fn preexisting_zero_cannot_acknowledge_a_new_silent_reset() {
    let (mut d, h) = device(0x06cb, 1);
    h.borrow_mut().input.push_back(vec![0, 0]);
    h.borrow_mut().input.push_back(Hardware::frame(&[17, 1, 2]));
    h.borrow_mut().no_reset_ack = true;
    assert_eq!(d.initialize().unwrap_err().kind, ErrorKind::ResetTimeout);
    assert_eq!(d.discarded_before_start(), 2);
    assert_eq!(h.borrow().reset_count, 1);
    assert_eq!(h.borrow().reads, 2);
    assert!(d.needs_reset());
    assert!(d.report_descriptor().is_empty());

    let (mut d, h) = device(0x06cb, 1);
    h.borrow_mut().input.push_back(vec![0, 0]);
    d.initialize().unwrap();
    assert_eq!(d.discarded_before_start(), 1);
    assert_eq!(h.borrow().reads, 2);
    assert!(d.is_ready());
}

#[test]
fn continuously_asserted_device_pin_bounds_pre_reset_drain() {
    let (mut d, h) = device(0x06cb, 1);
    h.borrow_mut().repeated_input = Some(Hardware::frame(&[17, 1, 2]));
    assert_eq!(
        d.initialize().unwrap_err().kind,
        ErrorKind::InputDrainTimeout
    );
    assert_eq!(h.borrow().reset_count, 0);
    assert!(d.discarded_before_start() > 0);
    assert!(d.discarded_before_start() <= 101);
    assert!(d.needs_reset());
    assert!(d.report_descriptor().is_empty());
}

#[test]
fn source_checked_no_irq_devices_expose_delay_instead_of_inventing_ack() {
    for (vendor, product) in [
        (0x0911, 0x5288),
        (0x103c, 0x184f),
        (0x048d, 0x83db),
        (0x2386, 0x3118),
        (0x0488, 0xd0c1),
        (0x36b6, 0xc001),
    ] {
        let (mut d, h) = device(vendor, product);
        h.borrow_mut().no_reset_ack = true;
        d.initialize().unwrap();
        assert!(d.is_ready());
        assert_eq!(
            d.reset_completion(),
            Some(ResetCompletion::DeviceQuirkDelay)
        );
        assert_eq!(h.borrow().reads, 0);
        assert_eq!(h.borrow().reset_count, 1);
        assert_eq!(
            h.borrow().time.get(),
            2 * WAKE_DELAY_US + QUIRK_RESET_DELAY_US
        );
        h.borrow_mut().input.push_back(Hardware::frame(&[17, 1, 2]));
        assert_eq!(d.poll().unwrap(), Some(InputReport::Data(vec![17, 1, 2])));
    }
    // Source identity is exact: a neighboring ordinary device still requires
    // its new zero-length response regardless of elapsed time.
    let (mut d, h) = device(0x2386, 0x3119);
    h.borrow_mut().no_reset_ack = true;
    assert_eq!(d.initialize().unwrap_err().kind, ErrorKind::ResetTimeout);
    assert_eq!(d.reset_completion(), None);
    let (d, _) = ready();
    assert_eq!(
        d.reset_completion(),
        Some(ResetCompletion::InterruptAcknowledgement)
    );
}

#[test]
fn no_irq_profile_keeps_irq_and_mutable_command_failure_requirements() {
    let (mut d, h) = device(0x2386, 0x3118);
    h.borrow_mut().no_reset_ack = true;
    h.borrow_mut().fail_irq = true;
    assert!(matches!(
        d.initialize().unwrap_err().kind,
        ErrorKind::Interrupt(_)
    ));
    assert_eq!(h.borrow().reset_count, 0);
    assert_eq!(d.reset_completion(), None);

    let (mut d, h) = device(0x2386, 0x3118);
    h.borrow_mut().no_reset_ack = true;
    h.borrow_mut().fail_after_opcode = Some(RESET);
    assert!(matches!(
        d.initialize().unwrap_err().kind,
        ErrorKind::Bus(_)
    ));
    assert_eq!(h.borrow().reset_count, 1);
    assert_eq!(d.reset_completion(), None);
    assert!(d.needs_reset());

    // Even when a no-IRQ device does emit a real ACK, consume it without
    // relabeling the checked delay as IRQ evidence or leaking it to poll.
    let (mut d, h) = device(0x2386, 0x3118);
    d.initialize().unwrap();
    assert_eq!(
        d.reset_completion(),
        Some(ResetCompletion::DeviceQuirkDelay)
    );
    assert_eq!(d.discarded_before_start(), 1);
    assert_eq!(h.borrow().reads, 1);
    assert_eq!(d.poll().unwrap(), None);
}

#[test]
fn exact_bad_input_size_profiles_expose_original_header_and_all_bytes() {
    for (vendor, product) in [(0x0911, 0x5288), (0x048d, 0x837a)] {
        for reported_length in [1u16, 33, 65535] {
            let (mut d, h) = device(vendor, product);
            h.borrow_mut().no_reset_ack = vendor == 0x0911;
            d.initialize().unwrap();
            let mut bytes: Vec<u8> = (0..32).collect();
            bytes[..2].copy_from_slice(&reported_length.to_le_bytes());
            let expected = bytes[2..].to_vec();
            h.borrow_mut().input.push_back(bytes);
            assert_eq!(
                d.poll().unwrap(),
                Some(InputReport::CorrectedLength {
                    data: expected,
                    reported_length,
                    read_length: 32,
                })
            );
            assert!(d.is_ready());
            h.borrow_mut().input.push_back(vec![0, 0]);
            assert_eq!(d.poll().unwrap_err().kind, ErrorKind::UnexpectedReset);
        }
    }
    let (mut d, h) = device(0x048d, 0x837b);
    d.initialize().unwrap();
    h.borrow_mut().input.push_back(vec![33, 0, 17, 1, 2]);
    assert!(matches!(
        d.poll().unwrap_err().kind,
        ErrorKind::ReportLength { .. }
    ));
}
