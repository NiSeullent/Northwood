//! Queried firmware, sliced commands and real i8042 data/reply lanes.
use super::*;
use alloc::collections::BTreeMap;
pub(super) struct Peripheral {
    pub(super) queries: BTreeMap<u8, [u8; 3]>,
    pub(super) sliced: u8,
    pub(super) mode: Option<u8>,
    pub(super) queried: Vec<u8>,
    pub(super) omit: Option<(u8, usize)>,
    pub(super) before_disable: Vec<u8>,
    pub(super) read_delay_us: u64,
    pub(super) late_mode_ack: bool,
    pub(super) late_query_byte: Option<(u8, usize)>,
    pub(super) query_reads_left: u8,
    pub(super) advanced: bool,
    pub(super) late_advanced_ack: bool,
    pub(super) client_writes: Vec<u8>,
    pub(super) before_client_ack: Vec<[u8; 6]>,
    pub(super) before_mode_ack: Vec<[u8; 6]>,
    pub(super) late_client_ack: bool,
    pub(super) busy_after_mode_zero: bool,
    pub(super) child: Option<ChildPeripheral>,
}
pub(super) struct ChildPeripheral {
    pub(super) wheel: bool,
    pub(super) explorer: bool,
    pub(super) scanning: bool,
    pub(super) enhanced: bool,
    pub(super) writes: Vec<u8>,
    pub(super) no_ack: Option<u8>,
    pub(super) no_id_ack: bool,
    pub(super) force_id: Option<u8>,
    pub(super) bad_bat: Option<u8>,
    pub(super) resend: Option<(u8, usize)>,
    pub(super) late_response: Option<u8>,
    pub(super) omit_reply: Option<usize>,
    pub(super) late_reply: Option<usize>,
    pub(super) reply_number: usize,
    pub(super) replies_before_parent_ack: bool,
    pub(super) get_id_tail: Option<u8>,
    argument: Option<u8>,
    rates: Vec<u8>,
    id: u8,
}
impl ChildPeripheral {
    pub(super) fn write(&mut self, byte: u8) -> Vec<u8> {
        self.writes.push(byte);
        if let Some((value, left)) = &mut self.resend
            && *value == byte
            && *left != 0
        {
            *left -= 1;
            return vec![RESEND];
        }
        let mut result = if self.no_ack == Some(byte)
            || (self.argument.is_none() && byte == GET_ID && self.no_id_ack)
        {
            Vec::new()
        } else {
            vec![ACK]
        };
        if let Some(command) = self.argument.take() {
            match command {
                SET_RATE => {
                    self.rates.push(byte);
                    if self.wheel && self.rates.ends_with(&[200, 100, 80]) {
                        self.id = 3;
                    }
                    if self.explorer && self.rates.ends_with(&[200, 200, 80]) {
                        self.id = 4;
                    }
                    if self.explorer && self.rates.ends_with(&[200, 80, 40]) {
                        self.enhanced = true;
                    }
                }
                SET_RESOLUTION => assert_eq!(byte, 2),
                _ => panic!("unknown child argument"),
            }
        } else {
            match byte {
                RESET => {
                    self.enhanced = false;
                    self.scanning = false;
                    self.id = 0;
                    self.rates.clear();
                    result.extend([self.bad_bat.unwrap_or(BAT_OK), 0]);
                }
                DEFAULTS | DISABLE_REPORTING => self.scanning = false,
                ENABLE_REPORTING => self.scanning = true,
                SET_RATE | SET_RESOLUTION => self.argument = Some(byte),
                GET_ID => {
                    result.push(self.force_id.unwrap_or(self.id));
                    if let Some(tail) = self.get_id_tail {
                        result.push(tail);
                    }
                }
                SCALE_ONE_TO_ONE => (),
                _ => panic!("unknown actual child command {byte:x}"),
            }
        }
        result
    }
}
fn target(new: bool, high_rate: bool) -> (Controller<Model, Time>, Rc<RefCell<Hardware>>) {
    let (base, h) = controller(Config::default());
    let mut queries: BTreeMap<_, _> = [
        (0, [5, 0x47, 7]),
        (3, [0, 0, 0x80]),
        (10, [0x12, 0x34, 0x56]),
        (1, [0x12, 0x34, 0]),
        (16, [0, 0, 0]),
        (2, [0xf4, 0, 3]),
        (9, [0x12, 0, 0]),
        (12, [2, 0x20, 0]),
        (8, [40, 0x80, 50]),
        (13, [187, 0x48, 156]),
        (15, [31, 4, 25]),
    ]
    .into();
    if !new {
        queries.insert(0, [1, 0x47, 4]);
        queries.insert(3, [0, 0, 0]);
        queries.insert(2, [0, 0x47, 0]);
    }
    h.borrow_mut().synaptics = Some(Peripheral {
        queries,
        sliced: 0,
        mode: None,
        queried: Vec::new(),
        omit: None,
        before_disable: Vec::new(),
        read_delay_us: 0,
        late_mode_ack: false,
        late_query_byte: None,
        query_reads_left: 0,
        advanced: false,
        late_advanced_ack: false,
        client_writes: Vec::new(),
        before_client_ack: Vec::new(),
        before_mode_ack: Vec::new(),
        late_client_ack: false,
        busy_after_mode_zero: false,
        child: None,
    });
    let c = Controller::new_synaptics(
        Model(h.clone()),
        Time(h.borrow().time.clone()),
        base.config,
        high_rate,
    )
    .unwrap();
    (c, h)
}
fn ready_synaptics(new: bool) -> (Controller<Model, Time>, Rc<RefCell<Hardware>>) {
    let (mut c, h) = target(new, true);
    c.initialize().unwrap();
    (c, h)
}
fn new_packet(x: u16, y: u16, z: u8, width: u8, buttons: u8) -> [u8; 6] {
    [
        0x80 | ((width & 12) << 2) | ((width & 2) << 1) | (buttons & 3),
        ((x >> 8) as u8 & 15) | ((y >> 4) as u8 & 0xf0),
        z,
        0xc0 | ((x >> 8) as u8 & 0x10)
            | ((y >> 7) as u8 & 0x20)
            | (width & 1) << 2
            | ((buttons & 1) ^ ((buttons >> 2) & 1)),
        x as u8,
        y as u8,
    ]
}
fn old_packet(x: u16, y: u16, z: u8, width: u8, buttons: u8) -> [u8; 6] {
    [
        0xc0 | ((z & 0xc0) >> 2) | ((width & 2) << 1) | (buttons & 3),
        ((x >> 8) as u8 & 31) | ((width & 8) << 4),
        x as u8,
        0x80 | (z & 0x3f),
        (y >> 8) as u8 & 31,
        y as u8,
    ]
}
#[test]
fn real_queried_identity_bounds_capabilities_and_mode_precede_reporting() {
    for new in [false, true] {
        for rate in [false, true] {
            let (mut c, h) = target(new, rate);
            c.initialize().unwrap();
            assert!(c.is_ready());
            assert_eq!(c.mouse_protocol(), None);
            let s = h.borrow();
            let s = s.synaptics.as_ref().unwrap();
            assert_eq!(
                s.queried,
                if new {
                    vec![0, 3, 10, 1, 16, 2, 9, 12, 8, 13, 15]
                } else {
                    vec![0, 3, 10, 2, 8]
                }
            );
            let expected = 0x84 | u8::from(new) | if rate { 0x40 } else { 0 };
            assert_eq!(s.mode, Some(expected));
            let cap = c.synaptics_capabilities().unwrap();
            assert_eq!(cap.mode, Some(expected));
            assert_eq!(cap.firmware, 0x123456);
            assert_eq!(cap.queries.len(), s.queried.len());
            assert_eq!(cap.resolution, Some((40, 50)));
            assert_eq!(cap.minimum, new.then_some((1000, 800)));
            assert_eq!(cap.maximum, new.then_some((6000, 5000)));
        }
    }
}
#[test]
fn the_protocol_probe_selects_synaptics_by_its_identity_with_the_firmware_profile() {
    let (explicit, h) = target(true, true);
    let mut c = Controller::new_probing(
        Model(h.clone()),
        Time(h.borrow().time.clone()),
        explicit.config,
        Some("PNP: LEN0033 PNP0f13"),
    )
    .unwrap();
    c.initialize().unwrap();
    let capabilities = c.synaptics_capabilities().unwrap();
    assert_eq!(
        capabilities.coordinate_quirk,
        Some(synaptics::board::Bounds {
            minimum: (1024, 2258),
            maximum: (5052, 4832)
        })
    );
    assert!(c.synaptics.as_ref().unwrap().high_rate);
    assert_eq!(
        h.borrow().synaptics.as_ref().unwrap().queried[..2],
        [0, 0],
        "the probe's identity query, then the engine's own"
    );
}
fn intertouch(firmware_id: &str) -> (Controller<Model, Time>, Rc<RefCell<Hardware>>) {
    let (explicit, h) = target(true, true);
    // SYN_CAP_INTERTOUCH beside the fixture's other extended capabilities.
    h.borrow_mut()
        .synaptics
        .as_mut()
        .unwrap()
        .queries
        .insert(12, [2, 0x60, 0]);
    let mut c = Controller::new_probing(
        Model(h.clone()),
        Time(h.borrow().time.clone()),
        explicit.config,
        Some(firmware_id),
    )
    .unwrap();
    c.initialize().unwrap();
    (c, h)
}
#[test]
fn intertouch_is_the_companion_of_a_listed_pad_that_has_it() {
    assert_eq!(
        intertouch("PNP: LEN0073 PNP0f13").0.smbus_companion(),
        Some(Companion::Rmi)
    );
    assert_eq!(
        intertouch("PNP: len2004 PNP0f13").0.smbus_companion(),
        Some(Companion::Rmi),
        "a top-button pad, matched without case"
    );
    assert_eq!(intertouch("PNP: SYN1B7F PNP0f13").0.smbus_companion(), None);
    let (c, _) = ready_synaptics(true);
    assert_eq!(c.smbus_companion(), None, "no PNP identifier");
}
#[test]
fn a_handed_over_mouse_port_drops_its_bytes_until_reclaimed() {
    let (mut c, h) = intertouch("PNP: LEN0073 PNP0f13");
    assert!(h.borrow().scanning[1]);
    c.hand_mouse_to_companion().unwrap();
    assert!(!h.borrow().scanning[1]);
    assert_eq!(
        h.borrow().synaptics.as_ref().unwrap().mode,
        None,
        "reset to its defaults"
    );
    emit(&h, Port::Mouse, &new_packet(3000, 3000, 60, 4, 1));
    assert!(c.poll().unwrap().is_empty());
    c.hand_mouse_to_companion().unwrap();
    c.reclaim_mouse().unwrap();
    assert!(h.borrow().scanning[1]);
    assert!(h.borrow().synaptics.as_ref().unwrap().mode.is_some());
    emit(&h, Port::Mouse, &new_packet(3000, 3000, 60, 4, 1));
    assert!(!c.poll().unwrap().is_empty());
}
#[test]
fn absolute_records_decode_every_field_in_both_formats() {
    for new in [false, true] {
        let (mut c, h) = ready_synaptics(new);
        for (x, y) in [
            (0, 1),
            (1, 2),
            (1000, 800),
            (6000, 5000),
            (8176, 8177),
            (8191, 8191),
        ] {
            for width in if new {
                vec![0, 1, 2, 4, 5, 15]
            } else {
                vec![0, 2, 8, 10]
            } {
                for z in [0, 24, 25, 30, 31, 255] {
                    let buttons = if new { 7 } else { 3 };
                    let packet = if new {
                        new_packet(x, y, z, width, buttons)
                    } else {
                        old_packet(x, y, z, width, buttons)
                    };
                    emit(&h, Port::Mouse, &packet);
                    let polled = c.poll_synaptics().unwrap();
                    assert_eq!(polled.frames.len(), 1);
                    let f = polled.frames[0];
                    assert_eq!(f.raw, packet);
                    let coord = |value: u16| {
                        if value == 8176 {
                            6143
                        } else if value > 8176 {
                            value as i16 - 8192
                        } else {
                            value as i16
                        }
                    };
                    assert_eq!(
                        (f.x, f.y, f.pressure, f.width, f.buttons),
                        (coord(x), coord(y), z, width, buttons)
                    );
                    assert!(
                        polled
                            .inputs
                            .iter()
                            .all(|input| !matches!(input, Input::Motion { .. }))
                    );
                }
            }
        }
    }
}
#[test]
fn incompatible_profiles_and_bad_query_identity_never_arm_absolute_mode() {
    for (query, bytes) in [(0, [5, 0x46, 7]), (9, [0x12, 0x90, 0]), (13, [0, 0, 0])] {
        let (mut c, h) = target(true, true);
        h.borrow_mut()
            .synaptics
            .as_mut()
            .unwrap()
            .queries
            .insert(query, bytes);
        assert!(matches!(
            c.initialize().unwrap_err().kind,
            ErrorKind::Synaptics(_)
        ));
        assert!(c.needs_reset());
        assert!(h.borrow().synaptics.as_ref().unwrap().mode.is_none());
        assert!(!c.synaptics_queries().unwrap().is_empty());
        assert!(!h.borrow().scanning[1]);
        let writes = h.borrow().writes.len();
        assert_eq!(c.initialize().unwrap_err().kind, ErrorKind::NeedsReset);
        assert_eq!(h.borrow().writes.len(), writes);
    }
}
#[test]
fn incomplete_queries_are_not_replayed() {
    for query in [0, 3, 10, 1, 16, 2, 9, 12, 8, 13, 15] {
        for count in 0..3 {
            let (mut c, h) = target(true, true);
            h.borrow_mut().synaptics.as_mut().unwrap().omit = Some((query, count));
            assert_eq!(
                c.initialize().unwrap_err().kind,
                ErrorKind::Timeout(Phase::Response)
            );
            let progress = c.synaptics_query_progress().unwrap();
            assert_eq!(
                (progress.id, usize::from(progress.received)),
                (query, count)
            );
            let actual = h.borrow().synaptics.as_ref().unwrap().queries[&query];
            assert_eq!(progress.bytes[..count], actual[..count]);
            assert!(h.borrow().synaptics.as_ref().unwrap().mode.is_none());
            assert!(c.needs_reset());
        }
    }
    let (mut c, h) = target(true, true);
    h.borrow_mut().no_ack = Some((Port::Mouse, 0x14));
    assert_eq!(
        c.initialize().unwrap_err().kind,
        ErrorKind::Timeout(Phase::Response)
    );
    assert_eq!(h.borrow().synaptics.as_ref().unwrap().mode, Some(0xc5));
    assert_eq!(c.synaptics_capabilities().unwrap().mode, None);
    assert_eq!(c.synaptics_queries().unwrap().len(), 11);
    assert!(c.needs_reset());
}
#[test]
fn fa_coordinates_survive_restart() {
    let (mut c, h) = ready_synaptics(true);
    let packet = new_packet(0x0ffa, 0x0ffe, 0xfa, 5, 3);
    // Keyboard command consumes the first mouse prefix into its pending lane.
    emit(&h, Port::Mouse, &packet[..2]);
    h.borrow_mut().synaptics.as_mut().unwrap().before_disable = packet[2..].to_vec();
    c.stop().unwrap();
    assert!(!c.is_ready());
    assert_eq!(c.drain_synaptics_frames()[0].raw, packet);
    let queries = h.borrow().synaptics.as_ref().unwrap().queried.len();
    c.initialize().unwrap();
    assert_eq!(
        h.borrow().synaptics.as_ref().unwrap().queried.len(),
        queries + 11
    );
    let packet = new_packet(2000, 2500, 50, 5, 0);
    h.borrow_mut().inject_led = packet.into_iter().map(|b| (Port::Mouse, b)).collect();
    c.set_leds(7).unwrap();
    assert_eq!(c.poll_synaptics().unwrap().frames[0].raw, packet);
    assert_eq!(h.borrow().leds, 7);
}
#[test]
fn packet_failures_and_idle_prefix_timeouts_preserve_owner_fragment_without_fake_release() {
    for new in [false, true] {
        for index in if new { vec![0, 3] } else { vec![0, 1, 3, 4] } {
            let (mut c, h) = ready_synaptics(new);
            let mut packet = if new {
                new_packet(2000, 2000, 50, 5, 1)
            } else {
                old_packet(2000, 2000, 50, 2, 1)
            };
            packet[index] ^= if new || matches!(index, 0 | 3) {
                0x40
            } else {
                0x20
            };
            emit(&h, Port::Mouse, &packet);
            assert!(matches!(
                c.poll().unwrap_err(),
                super::super::Error {
                    response: None,
                    kind: ErrorKind::Synaptics(synaptics::Error::Packet { .. }),
                    ..
                }
            ));
            assert_eq!(c.synaptics_fragment().unwrap(), &packet[..=index]);
            assert!(c.needs_reset());
            assert!(c.drain_synaptics_frames().is_empty());
        }
        let (mut c, h) = ready_synaptics(new);
        let packet = if new {
            new_packet(2000, 2000, 50, 5, 1)
        } else {
            old_packet(2000, 2000, 50, 2, 1)
        };
        emit(&h, Port::Mouse, &packet[..2]);
        assert!(c.poll().unwrap().is_empty());
        h.borrow().time.set(h.borrow().time.get() + 501);
        assert_eq!(
            c.poll().unwrap_err().kind,
            ErrorKind::PacketTimeout(Port::Mouse)
        );
        assert_eq!(c.synaptics_fragment().unwrap(), &packet[..2]);
    }
}
#[test]
fn queued_raw_frames_apply_bounded_backpressure_and_can_resume_without_reset() {
    let (mut c, h) = ready_synaptics(true);
    let packet = new_packet(2000, 2000, 50, 5, 0);
    for _ in 0..40 {
        emit(&h, Port::Mouse, &packet);
    }
    for _ in 0..4 {
        c.poll().unwrap();
    }
    let consumed = h.borrow().consumed;
    c.poll().unwrap();
    assert_eq!(h.borrow().consumed, consumed);
    assert_eq!(c.drain_synaptics_frames().len(), 32);
    assert_eq!(c.poll_synaptics().unwrap().frames.len(), 8);
    assert!(!c.needs_reset());
}
#[test]
fn restart_requeries_identity_and_refuses_changed_hardware_before_mode_mutation() {
    for query in [0, 3, 10, 1, 16, 2, 9, 8, 13, 15] {
        let (mut c, h) = ready_synaptics(true);
        let original = c.synaptics_capabilities().cloned();
        c.stop().unwrap();
        h.borrow_mut()
            .synaptics
            .as_mut()
            .unwrap()
            .queries
            .get_mut(&query)
            .unwrap()[0] ^= 0x20;
        let error = c.initialize().unwrap_err();
        assert!(
            matches!(
                error.kind,
                ErrorKind::Synaptics(synaptics::Error::IdentityChanged)
            ),
            "query {query}: {error:?}"
        );
        assert_eq!(c.synaptics_capabilities(), original.as_ref());
        assert!(h.borrow().synaptics.as_ref().unwrap().mode.is_none());
        assert!(c.needs_reset());
    }
}
#[test]
fn continuous_same_port_reports_cannot_extend_a_command_response_deadline() {
    let (mut c, h) = target(true, true);
    c.config.keyboard = false;
    c.initialize().unwrap();
    let packet = new_packet(2000, 2000, 50, 5, 0);
    h.borrow_mut().synaptics.as_mut().unwrap().before_disable = packet.repeat(40);
    h.borrow_mut().synaptics.as_mut().unwrap().read_delay_us = 2;
    let consumed = h.borrow().consumed;
    assert_eq!(
        c.stop().unwrap_err().kind,
        ErrorKind::Timeout(Phase::Response)
    );
    assert!(c.needs_reset());
    assert!(h.borrow().consumed - consumed < 150);
    assert!(!c.drain_synaptics_frames().is_empty());
}
#[test]
fn late_acknowledgements_never_publish_mode_or_stopped_success_and_survive_cleanup() {
    let (mut c, h) = target(true, true);
    h.borrow_mut().synaptics.as_mut().unwrap().late_mode_ack = true;
    let error = c.initialize().unwrap_err();
    assert_eq!(error.kind, ErrorKind::Timeout(Phase::Response));
    assert_eq!(
        error.response,
        Some(ObservedResponse {
            port: Port::Mouse,
            data: ACK
        })
    );
    assert_eq!(h.borrow().synaptics.as_ref().unwrap().mode, Some(0xc5));
    assert_eq!(c.synaptics_capabilities().unwrap().mode, None);
    assert!(c.needs_reset());
    for new in [false, true] {
        let (mut c, h) = ready_synaptics(new);
        h.borrow_mut().synaptics.as_mut().unwrap().read_delay_us = 101;
        let error = c.stop().unwrap_err();
        assert_eq!(error.kind, ErrorKind::Timeout(Phase::Response));
        assert_eq!(
            error.response,
            Some(ObservedResponse {
                port: Port::Mouse,
                data: ACK
            })
        );
        assert!(!h.borrow().scanning[1]);
        assert!(c.needs_reset());
        let writes = h.borrow().writes.len();
        assert_eq!(c.stop().unwrap_err().kind, ErrorKind::NeedsReset);
        assert_eq!(h.borrow().writes.len(), writes);
    }
}
#[test]
fn successful_but_late_query_reads_keep_prefix_before_timeout() {
    for query in [0, 3, 10, 1, 16, 2, 9, 12, 8, 13, 15] {
        for index in 0..3 {
            let (mut c, h) = target(true, true);
            h.borrow_mut().synaptics.as_mut().unwrap().late_query_byte = Some((query, index));
            let error = c.initialize().unwrap_err();
            assert_eq!(error.kind, ErrorKind::Timeout(Phase::Response));
            let actual = h.borrow().synaptics.as_ref().unwrap().queries[&query];
            assert_eq!(
                error.response,
                Some(ObservedResponse {
                    port: Port::Mouse,
                    data: actual[index]
                })
            );
            let progress = c.synaptics_query_progress().unwrap();
            assert_eq!(
                (progress.id, usize::from(progress.received)),
                (query, index + 1)
            );
            assert_eq!(progress.bytes[..=index], actual[..=index]);
            assert!(c.needs_reset());
        }
    }
}

fn advanced_target() -> (Controller<Model, Time>, Rc<RefCell<Hardware>>) {
    let (c, h) = target(true, true);
    h.borrow_mut()
        .synaptics
        .as_mut()
        .unwrap()
        .queries
        .get_mut(&12)
        .unwrap()[0] |= 8;
    (c, h)
}
#[test]
fn advanced_mode_requires_its_own_real_command_and_never_publishes_late_ack_as_success() {
    let (mut c, h) = advanced_target();
    c.initialize().unwrap();
    assert!(c.synaptics_capabilities().unwrap().advanced_mode);
    assert!(h.borrow().synaptics.as_ref().unwrap().advanced);
    assert!(
        h.borrow()
            .writes
            .windows(2)
            .any(|window| window[1] == (false, 0xc8))
    );
    let (mut c, h) = advanced_target();
    h.borrow_mut().synaptics.as_mut().unwrap().late_advanced_ack = true;
    let error = c.initialize().unwrap_err();
    assert_eq!(error.kind, ErrorKind::Timeout(Phase::Response));
    assert_eq!(
        error.response,
        Some(ObservedResponse {
            port: Port::Mouse,
            data: ACK
        })
    );
    assert_eq!(c.synaptics_capabilities().unwrap().mode, Some(0xc5));
    assert!(!c.synaptics_capabilities().unwrap().advanced_mode);
    assert!(h.borrow().synaptics.as_ref().unwrap().advanced);
    assert!(c.needs_reset());
}
#[test]
fn advanced_contacts_counts_and_unknown_records_keep_wire_order_without_pointer_fabrication() {
    let (mut c, h) = advanced_target();
    c.initialize().unwrap();
    let first = new_packet(2000, 2500, 60, 5, 1);
    let advanced = [0x84, 0xfa, 0xfe, 0xf0, 0xa5, 0x1f];
    let next = new_packet(2200, 2600, 65, 5, 1);
    for packet in [first, advanced, next] {
        emit(&h, Port::Mouse, &packet);
    }
    let polled = c.poll_synaptics().unwrap();
    assert_eq!(polled.frames.len(), 2);
    assert_eq!(
        (
            polled.frames[0].native_sequence,
            polled.frames[1].native_sequence
        ),
        (0, 2)
    );
    assert_eq!(polled.advanced[0].native_sequence, 1);
    assert_eq!(polled.advanced[0].raw, advanced);
    assert_eq!(
        polled.advanced[0].data,
        synaptics::agm::Data::Secondary {
            x: 3060,
            y: 5628,
            pressure: 126
        }
    );
    for count in 0..=255 {
        let packet = [0x84, count, 0, 0xc0, 0, 0x20];
        emit(&h, Port::Mouse, &packet);
        let polled = c.poll_synaptics().unwrap();
        assert!(polled.frames.is_empty() && polled.inputs.is_empty());
        assert_eq!(polled.advanced[0].data, synaptics::agm::Data::Count(count));
        assert_eq!(polled.advanced[0].raw, packet);
    }
    for kind in [0, 3] {
        let packet = [0x84, 1, 2, 0xc0, 3, kind << 4];
        emit(&h, Port::Mouse, &packet);
        let polled = c.poll_synaptics().unwrap();
        assert!(polled.inputs.is_empty() && polled.frames.is_empty());
        assert_eq!(
            polled.advanced[0].data,
            synaptics::agm::Data::Unrecognized(kind)
        );
    }
}
#[test]
fn advanced_frames_share_bounded_backpressure_stop_and_identity_checked_restart() {
    let (mut c, h) = advanced_target();
    c.initialize().unwrap();
    let packet = [0x84, 3, 0, 0xc0, 0, 0x20];
    for _ in 0..40 {
        emit(&h, Port::Mouse, &packet);
    }
    for _ in 0..4 {
        c.poll().unwrap();
    }
    let consumed = h.borrow().consumed;
    c.poll().unwrap();
    assert_eq!(h.borrow().consumed, consumed);
    assert_eq!(c.drain_synaptics_advanced().len(), 32);
    assert_eq!(c.poll_synaptics().unwrap().advanced.len(), 8);
    c.stop().unwrap();
    c.initialize().unwrap();
    assert!(c.synaptics_capabilities().unwrap().advanced_mode);
    emit(&h, Port::Mouse, &packet);
    assert_eq!(c.poll_synaptics().unwrap().advanced[0].native_sequence, 40);
}
#[test]
fn advanced_mode_reports_multifinger_without_the_legacy_multifinger_capability() {
    let (mut c, h) = advanced_target();
    h.borrow_mut()
        .synaptics
        .as_mut()
        .unwrap()
        .queries
        .get_mut(&2)
        .unwrap()[2] &= !2;
    c.initialize().unwrap();
    assert_eq!(c.synaptics_capabilities().unwrap().capabilities & 2, 0);
    for width in [0, 1] {
        let ordinary = new_packet(2000, 2500, 50, width, 1);
        let secondary = [0x84, 1, 2, 0xc0, 3, 0x10];
        let count = [0x84, width + 2, 0, 0xc0, 0, 0x20];
        for packet in [secondary, ordinary, count] {
            emit(&h, Port::Mouse, &packet);
        }
        let polled = c.poll_synaptics().unwrap();
        assert_eq!(polled.frames[0].fingers, Some(width + 2));
        assert_eq!(
            polled.advanced[1].data,
            synaptics::agm::Data::Count(width + 2)
        );
        assert!(polled.advanced[0].native_sequence < polled.frames[0].native_sequence);
        assert!(polled.frames[0].native_sequence < polled.advanced[1].native_sequence);
    }
}
fn image_target() -> (Controller<Model, Time>, Rc<RefCell<Hardware>>) {
    let (c, h) = target(true, true);
    {
        let mut h = h.borrow_mut();
        let s = h.synaptics.as_mut().unwrap();
        s.queries.get_mut(&12).unwrap()[1] |= 8;
        s.queries.get_mut(&2).unwrap()[2] &= !2;
    }
    (c, h)
}
#[test]
fn image_sensor_uses_source_count_policy_and_current_count_without_legacy_touch_hysteresis() {
    let (mut c, h) = image_target();
    c.initialize().unwrap();
    assert!(c.synaptics_capabilities().unwrap().advanced_mode);
    assert_eq!(
        c.synaptics_capabilities().unwrap().extended_0c & (1 << 19),
        0
    );
    assert_eq!(c.synaptics_count_observation(), None);
    for count in [0, 1, 2, 3, 4, 5, 255] {
        let packet = [0x84, count, 0, 0xc0, 0, 0x20];
        emit(&h, Port::Mouse, &packet);
        let observation = c.poll_synaptics().unwrap().advanced[0];
        assert_eq!(
            c.synaptics_count_observation(),
            Some((observation.native_sequence, count))
        );
        for width in [0, 1, 3, 4, 15] {
            for pressure in [0, 1, 24, 31, 255] {
                let ordinary = new_packet(0, 2500, pressure, width, 0);
                emit(&h, Port::Mouse, &ordinary);
                let frame = c.poll_synaptics().unwrap().frames[0];
                let expected = if pressure == 0 {
                    0
                } else if width >= 4 {
                    1
                } else if width == 0 {
                    2
                } else if width == 1 {
                    if count != 0 { count } else { 3 }
                } else {
                    4
                };
                assert_eq!(frame.fingers, Some(expected));
                assert_eq!(frame.touching, Some(expected != 0));
                assert_eq!(
                    frame.x, 0,
                    "image source does not apply legacy X>1 contact gate"
                );
            }
        }
    }
}
#[test]
fn image_sensor_epoch_reset_refuses_old_absolute_format_and_requires_activation_ack() {
    let (mut c, h) = image_target();
    c.initialize().unwrap();
    emit(&h, Port::Mouse, &[0x84, 5, 0, 0xc0, 0, 0x20]);
    c.poll_synaptics().unwrap();
    assert_eq!(c.synaptics_count_observation().unwrap().1, 5);
    c.stop().unwrap();
    c.initialize().unwrap();
    assert_eq!(c.synaptics_count_observation(), None);
    emit(&h, Port::Mouse, &new_packet(2000, 2500, 50, 1, 0));
    assert_eq!(c.poll_synaptics().unwrap().frames[0].fingers, Some(3));
    let (mut c, h) = image_target();
    h.borrow_mut().synaptics.as_mut().unwrap().late_advanced_ack = true;
    assert_eq!(
        c.initialize().unwrap_err().kind,
        ErrorKind::Timeout(Phase::Response)
    );
    assert!(!c.synaptics_capabilities().unwrap().advanced_mode);
    assert!(c.needs_reset());
    let (mut c, h) = image_target();
    h.borrow_mut()
        .synaptics
        .as_mut()
        .unwrap()
        .queries
        .insert(3, [0, 0, 0]);
    assert!(matches!(
        c.initialize().unwrap_err().kind,
        ErrorKind::Synaptics(synaptics::Error::Unsupported { query: 12, .. })
    ));
    assert!(h.borrow().synaptics.as_ref().unwrap().mode.is_none());
}
fn button_target(
    count: u8,
    stick: bool,
    version: u16,
    four: bool,
) -> (Controller<Model, Time>, Rc<RefCell<Hardware>>) {
    let (c, h) = target(true, true);
    {
        let mut h = h.borrow_mut();
        let s = h.synaptics.as_mut().unwrap();
        s.queries
            .insert(0, [version as u8, 0x47, (version >> 8) as u8]);
        s.queries.get_mut(&9).unwrap()[1] = count << 4;
        s.queries.get_mut(&16).unwrap()[0] = u8::from(stick);
        if four {
            s.queries.get_mut(&2).unwrap()[2] |= 8;
        }
    }
    (c, h)
}
#[test]
fn four_button_packets_preserve_five_button_compatibility_and_acknowledged_release() {
    let (mut c, h) = button_target(0, false, 0x705, true);
    c.initialize().unwrap();
    for switches in 0..4 {
        let mut packet = new_packet(2000, 2500, 50, 5, 3);
        packet[3] |= packet[0] & 2;
        packet[3] ^= switches;
        emit(&h, Port::Mouse, &packet);
        let polled = c.poll_synaptics().unwrap();
        let expected = 3 | ((switches & 1) << 2) | ((switches & 1) << 4) | ((switches & 2) << 2);
        assert_eq!(polled.frames[0].buttons, expected);
        assert!(polled.inputs.contains(&Input::MouseButtons(expected)));
    }
    assert!(c.stop().unwrap().contains(&Input::MouseButtons(0)));
}
#[test]
fn complete_extended_bitmaps_keep_padding_target_mapping_and_firmware_update_validity() {
    for count in 1..=8 {
        for stick in [false, true] {
            for version in [0x705, 0x801, 0x802, 0x900] {
                let (mut c, h) = button_target(count, stick, version, false);
                c.initialize().unwrap();
                let bits = count.div_ceil(2);
                let mask = (1u8 << bits) - 1;
                for bitmap in 0..=255 {
                    let mut packet = new_packet(2000, 2500, 50, 5, 0);
                    packet[3] ^= 2;
                    packet[4] = (packet[4] & !mask) | (bitmap & mask);
                    packet[5] = (packet[5] & !mask) | ((bitmap >> bits) & mask);
                    emit(&h, Port::Mouse, &packet);
                    let frame = c.poll_synaptics().unwrap().frames[0];
                    let extended = frame.extended_buttons.unwrap();
                    let wire = bitmap & ((1u16 << (bits * 2)) - 1) as u8;
                    assert_eq!((extended.count, extended.wire_bitmap), (count, wire));
                    let target = if stick {
                        synaptics::buttons::Target::Trackstick(
                            (wire & 1) | ((wire & 4) >> 1) | ((wire & 2) << 1),
                        )
                    } else {
                        let mut logical = 0;
                        for index in 0..count {
                            let raw_index = if index % 2 == 0 {
                                index / 2
                            } else {
                                bits + index / 2
                            };
                            logical |= ((wire >> raw_index) & 1) << index;
                        }
                        synaptics::buttons::Target::Touchpad(logical)
                    };
                    assert_eq!(extended.target, target);
                    packet[3] ^= 2;
                    emit(&h, Port::Mouse, &packet);
                    let update = c.poll_synaptics().unwrap().frames[0].extended_buttons;
                    if matches!(version, 0x801 | 0x802) {
                        assert_eq!(update, None);
                    } else {
                        assert_eq!(update.unwrap().wire_bitmap, 0);
                    }
                }
            }
        }
    }
}
fn passthrough_target() -> (Controller<Model, Time>, Rc<RefCell<Hardware>>) {
    let (c, h) = target(true, true);
    h.borrow_mut()
        .synaptics
        .as_mut()
        .unwrap()
        .queries
        .get_mut(&2)
        .unwrap()[2] |= 0x80;
    (c, h)
}
fn forcepad_target(new: bool) -> (Controller<Model, Time>, Rc<RefCell<Hardware>>) {
    let (base, h) = target(new, true);
    let c = Controller::new_synaptics_forcepad(
        Model(h.clone()),
        Time(h.borrow().time.clone()),
        base.config,
        true,
        "PNP: SYN300D",
    )
    .unwrap();
    (c, h)
}
fn force_packet(z: u8, width: u8, above: bool) -> [u8; 6] {
    let mut raw = new_packet(2000, 2500, z, width, 0);
    raw[3] ^= u8::from(above);
    raw
}
fn pnp_target(
    firmware: &str,
    board: Option<u16>,
) -> (Controller<Model, Time>, Rc<RefCell<Hardware>>) {
    let (base, h) = target(true, true);
    {
        let mut hw = h.borrow_mut();
        let q = &mut hw.synaptics.as_mut().unwrap().queries;
        if let Some(board) = board {
            q.insert(1, [((board >> 6) as u8 & 0xfc) | 2, board as u8, 0]);
        } else {
            q.insert(0, [1, 0x47, 4]);
            q.get_mut(&2).unwrap()[1] = 0x47;
        }
    }
    let c = Controller::new_synaptics_pnp(
        Model(h.clone()),
        Time(h.borrow().time.clone()),
        base.config,
        true,
        firmware,
    )
    .unwrap();
    (c, h)
}
#[test]
fn pnp_and_queried_board_select_coordinate_rules_without_replacing_raw_bounds() {
    for (firmware, board, min, max) in [
        ("PNP: LEN0033", Some(3000), (1024, 2258), (5052, 4832)),
        ("PNP: LEN0042", Some(3000), (1232, 1156), (5710, 4696)),
        ("PNP: LEN2000", Some(3000), (1024, 2021), (5113, 4832)),
        ("PNP: LEN2001", Some(3000), (1024, 2508), (5022, 4832)),
        ("PNP: LEN2006", Some(2691), (1024, 2457), (5045, 4832)),
        ("PNP: LEN2006", Some(2690), (1264, 1171), (5675, 4688)),
        ("PNP: LEN2006", Some(2692), (1264, 1171), (5675, 4688)),
        ("PNP: other len0034", Some(2961), (1024, 2024), (5112, 4832)),
        ("PNP: LEN0034", None, (1024, 2024), (5112, 4832)),
        (
            "PNP: LEN2006 LEN0033",
            Some(2691),
            (1024, 2258),
            (5052, 4832),
        ),
    ] {
        let (mut c, h) = pnp_target(firmware, board);
        c.initialize().unwrap();
        let cap = c.synaptics_capabilities().unwrap();
        assert_eq!(cap.board, board);
        assert_eq!(cap.minimum, Some((1000, 800)));
        assert_eq!(cap.maximum, Some((6000, 5000)));
        assert_eq!(cap.bounds(), (min, max));
        assert!(cap.coordinate_quirk.is_some());
        if board.is_none() {
            assert!(!cap.queries.iter().any(|q| q.id == 1));
        }
        emit(&h, Port::Mouse, &new_packet(min.0, min.1, 50, 5, 0));
        assert!(c.poll_synaptics().unwrap().inputs.contains(&Input::Move {
            x: 0,
            y: crate::input::POINTER_MAX
        }));
        emit(&h, Port::Mouse, &new_packet(max.0, max.1, 50, 5, 0));
        assert!(c.poll_synaptics().unwrap().inputs.contains(&Input::Move {
            x: crate::input::POINTER_MAX,
            y: 0
        }));
    }
    for id in [
        "LEN0034", "LEN0036", "LEN0037", "LEN0039", "LEN2002", "LEN2004",
    ] {
        for board in [2961, 2962] {
            let (mut c, _) = pnp_target(&alloc::format!("PNP: {id}"), Some(board));
            c.initialize().unwrap();
            assert_eq!(
                c.synaptics_capabilities()
                    .unwrap()
                    .coordinate_quirk
                    .is_some(),
                board == 2961
            );
        }
    }
}
#[test]
fn pnp_rule_is_explicit_and_changed_board_is_refused_before_mode_rearm() {
    let (base, h) = target(true, true);
    assert!(matches!(
        Controller::new_synaptics_pnp(
            Model(h.clone()),
            Time(h.borrow().time.clone()),
            base.config,
            true,
            "LEN0033"
        ),
        Err(super::super::Error {
            kind: ErrorKind::InvalidConfiguration,
            ..
        })
    ));
    assert!(h.borrow().writes.is_empty());
    let (mut c, h) = pnp_target("PNP: LEN0034", Some(2961));
    c.initialize().unwrap();
    let bounds = c.synaptics_capabilities().unwrap().bounds();
    c.stop().unwrap();
    h.borrow_mut()
        .synaptics
        .as_mut()
        .unwrap()
        .queries
        .insert(1, [((2962u16 >> 6) as u8 & 0xfc) | 2, 2962u16 as u8, 0]);
    assert_eq!(
        c.initialize().unwrap_err().kind,
        ErrorKind::Synaptics(synaptics::Error::IdentityChanged)
    );
    assert_eq!(h.borrow().synaptics.as_ref().unwrap().mode, None);
    assert_eq!(c.synaptics_capabilities().unwrap().bounds(), bounds);
}
#[test]
fn known_coordinate_correction_can_repair_reported_geometry_but_unknown_id_cannot() {
    for firmware in ["PNP: LEN0033", "PNP: LEN0033x"] {
        let (mut c, h) = pnp_target(firmware, Some(3000));
        h.borrow_mut()
            .synaptics
            .as_mut()
            .unwrap()
            .queries
            .insert(13, [1, 0, 1]);
        if firmware.ends_with('x') {
            assert_eq!(
                c.initialize().unwrap_err().kind,
                ErrorKind::Synaptics(synaptics::Error::Geometry)
            );
            assert_eq!(h.borrow().synaptics.as_ref().unwrap().mode, None);
        } else {
            c.initialize().unwrap();
            assert_eq!(c.synaptics_capabilities().unwrap().maximum, Some((32, 32)));
            assert_eq!(
                c.synaptics_capabilities().unwrap().bounds(),
                ((1024, 2258), (5052, 4832))
            );
        }
    }
}
fn child_target(wheel: bool, explorer: bool) -> (Controller<Model, Time>, Rc<RefCell<Hardware>>) {
    let (c, h) = passthrough_target();
    h.borrow_mut().synaptics.as_mut().unwrap().child = Some(ChildPeripheral {
        wheel,
        explorer,
        scanning: false,
        enhanced: false,
        writes: Vec::new(),
        no_ack: None,
        no_id_ack: false,
        force_id: None,
        bad_bat: None,
        resend: None,
        late_response: None,
        omit_reply: None,
        late_reply: None,
        reply_number: 0,
        replies_before_parent_ack: false,
        get_id_tail: None,
        argument: None,
        rates: Vec::new(),
        id: 0,
    });
    (c, h)
}
fn wrapped(packet: [u8; 4]) -> [u8; 6] {
    [0x84, packet[0], packet[3], 0xc4, packet[1], packet[2]]
}
#[test]
fn child_reset_identity_negotiation_input_and_stop_remain_separate_from_parent() {
    for (wheel, explorer, protocol) in [
        (false, false, MouseProtocol::Standard),
        (true, false, MouseProtocol::IntelliMouse),
        (true, true, MouseProtocol::Explorer),
    ] {
        let (mut c, h) = child_target(wheel, explorer);
        c.initialize().unwrap();
        assert_eq!(c.synaptics_child_protocol(), None);
        let start = h.borrow().time.get();
        assert_eq!(c.initialize_synaptics_mouse().unwrap(), protocol);
        assert!(h.borrow().time.get().wrapping_sub(start) >= 10_000);
        assert_eq!(c.synaptics_child_protocol(), Some(protocol));
        assert!(
            h.borrow()
                .synaptics
                .as_ref()
                .unwrap()
                .child
                .as_ref()
                .unwrap()
                .scanning
        );
        let expected = if protocol == MouseProtocol::Standard {
            0xc5
        } else {
            0xc7
        };
        assert_eq!(c.synaptics_capabilities().unwrap().mode, Some(expected));
        let command_records = c.poll_synaptics().unwrap();
        assert_eq!(command_records.passthrough.len(), 27);
        assert!(command_records.child_inputs.is_empty() && command_records.inputs.is_empty());
        emit(&h, Port::Mouse, &new_packet(2000, 2500, 50, 5, 1));
        emit(
            &h,
            Port::Mouse,
            &wrapped([0x09, 2, 3, if explorer { 0x31 } else { 1 }]),
        );
        let input = c.poll_synaptics().unwrap();
        assert!(input.inputs.contains(&Input::Buttons(1)));
        assert!(
            input
                .child_inputs
                .contains(&Input::Motion { dx: 2, dy: -3 })
        );
        assert!(
            input
                .child_inputs
                .contains(&Input::MouseButtons(if explorer { 25 } else { 1 }))
        );
        if wheel {
            assert!(input.child_inputs.contains(&Input::MouseWheel(-1)));
        }
        assert_eq!(
            input.frames[0].native_sequence + 1,
            input.passthrough[0].native_sequence
        );
        assert!(c.stop().unwrap().contains(&Input::Buttons(0)));
        assert!(
            c.drain_synaptics_child_inputs()
                .contains(&Input::MouseButtons(0))
        );
        assert!(
            !h.borrow()
                .synaptics
                .as_ref()
                .unwrap()
                .child
                .as_ref()
                .unwrap()
                .scanning
        );
        assert_eq!(c.synaptics_child_protocol(), None);
        c.initialize().unwrap();
        assert_eq!(
            c.synaptics_child_protocol(),
            None,
            "parent reset cannot fabricate child activation"
        );
        assert_eq!(c.initialize_synaptics_mouse().unwrap(), protocol);
    }
}
#[test]
fn bound_child_blocks_raw_mutations_but_explicit_input_release_preserves_live_routing() {
    let (mut c, h) = child_target(false, false);
    c.initialize().unwrap();
    c.initialize_synaptics_mouse().unwrap();
    c.poll_synaptics().unwrap();
    let count = h.borrow().writes.len();
    assert_eq!(
        c.initialize_synaptics_mouse().unwrap_err().kind,
        ErrorKind::AlreadyInitialized
    );
    assert_eq!(
        c.write_synaptics_passthrough(RESET).unwrap_err().kind,
        ErrorKind::Synaptics(synaptics::Error::ChildActive)
    );
    assert_eq!(
        c.configure_synaptics_passthrough(synaptics::passthrough::Format::Four)
            .unwrap_err()
            .kind,
        ErrorKind::Synaptics(synaptics::Error::ChildActive)
    );
    assert_eq!(h.borrow().writes.len(), count);
    emit(&h, Port::Mouse, &wrapped([9, 0, 0, 0]));
    assert!(
        c.poll_synaptics()
            .unwrap()
            .child_inputs
            .contains(&Input::MouseButtons(1))
    );
    c.release_inputs();
    assert!(
        c.drain_synaptics_child_inputs()
            .contains(&Input::MouseButtons(0))
    );
    assert_eq!(c.synaptics_child_protocol(), Some(MouseProtocol::Standard));
    emit(&h, Port::Mouse, &wrapped([9, 0, 0, 0]));
    assert!(
        c.poll_synaptics()
            .unwrap()
            .child_inputs
            .contains(&Input::MouseButtons(1))
    );
}
#[test]
fn child_failures_keep_real_reply_or_late_effect_without_synthetic_ack_or_replay() {
    for phase in 0..4 {
        let (mut c, h) = child_target(true, true);
        c.initialize().unwrap();
        {
            let mut hw = h.borrow_mut();
            let child = hw.synaptics.as_mut().unwrap().child.as_mut().unwrap();
            match phase {
                0 => child.bad_bat = Some(0xfc),
                1 => child.force_id = Some(0x42),
                2 => child.no_ack = Some(ENABLE_REPORTING),
                _ => child.late_response = Some(RESET),
            }
        }
        let error = c.initialize_synaptics_mouse().unwrap_err();
        match phase {
            0 => assert_eq!(
                error.kind,
                ErrorKind::Synaptics(synaptics::Error::ChildReset([0xfc, 0]))
            ),
            1 => assert_eq!(error.kind, ErrorKind::UnsupportedMouseId(0x42)),
            _ => assert_eq!(error.kind, ErrorKind::Timeout(Phase::Response)),
        }
        assert!(c.needs_reset());
        assert_eq!(c.synaptics_child_protocol(), None);
        let progress = c.synaptics_child_command().unwrap();
        if phase == 2 {
            assert_eq!(progress.opcode, ENABLE_REPORTING);
            assert_eq!(progress.acknowledged, 0);
            assert!(
                h.borrow()
                    .synaptics
                    .as_ref()
                    .unwrap()
                    .child
                    .as_ref()
                    .unwrap()
                    .scanning
            );
        } else if phase == 3 {
            assert_eq!(progress.opcode, RESET);
            assert_eq!(progress.acknowledged, 0);
            assert_eq!(progress.observations[0].byte, ACK);
            assert_eq!(c.drain_synaptics_passthrough()[0].first(), ACK);
        } else {
            assert!(!progress.observations.is_empty());
        }
        let count = h.borrow().writes.len();
        assert_eq!(
            c.initialize_synaptics_mouse().unwrap_err().kind,
            ErrorKind::NeedsReset
        );
        assert_eq!(h.borrow().writes.len(), count);
    }
}
#[test]
fn child_resend_retries_only_the_rejected_byte() {
    let (mut c, h) = child_target(true, true);
    c.initialize().unwrap();
    h.borrow_mut()
        .synaptics
        .as_mut()
        .unwrap()
        .child
        .as_mut()
        .unwrap()
        .resend = Some((DEFAULTS, 1));
    assert_eq!(
        c.initialize_synaptics_mouse().unwrap(),
        MouseProtocol::Explorer
    );
    assert_eq!(
        h.borrow()
            .synaptics
            .as_ref()
            .unwrap()
            .child
            .as_ref()
            .unwrap()
            .writes
            .iter()
            .filter(|b| **b == DEFAULTS)
            .count(),
        2
    );
    assert_eq!(
        c.drain_synaptics_passthrough()
            .iter()
            .filter(|r| r.first() == RESEND)
            .count(),
        1
    );
}

#[test]
fn uncertain_parent_mode_never_activates_the_child() {
    let (mut c, h) = child_target(true, true);
    c.initialize().unwrap();
    h.borrow_mut().synaptics.as_mut().unwrap().late_mode_ack = true;
    assert_eq!(
        c.initialize_synaptics_mouse().unwrap_err().kind,
        ErrorKind::Timeout(Phase::Response)
    );
    assert_eq!(c.synaptics_capabilities().unwrap().mode, Some(0xc5));
    assert_eq!(h.borrow().synaptics.as_ref().unwrap().mode, Some(0xc7));
    assert!(
        !h.borrow()
            .synaptics
            .as_ref()
            .unwrap()
            .child
            .as_ref()
            .unwrap()
            .scanning
    );
    assert_eq!(c.synaptics_child_protocol(), None);
}
#[test]
fn child_stop_failure_keeps_held_input_and_unknown_replacement_is_not_rearmed() {
    let (mut c, h) = child_target(true, true);
    c.initialize().unwrap();
    c.initialize_synaptics_mouse().unwrap();
    c.poll_synaptics().unwrap();
    emit(&h, Port::Mouse, &wrapped([9, 0, 0, 0]));
    c.poll_synaptics().unwrap();
    h.borrow_mut()
        .synaptics
        .as_mut()
        .unwrap()
        .child
        .as_mut()
        .unwrap()
        .no_ack = Some(DISABLE_REPORTING);
    assert_eq!(
        c.stop().unwrap_err().kind,
        ErrorKind::Timeout(Phase::Response)
    );
    assert!(c.drain_synaptics_child_inputs().is_empty());
    c.release_inputs();
    assert!(
        c.drain_synaptics_child_inputs()
            .contains(&Input::MouseButtons(0))
    );
    let (mut c, h) = child_target(true, true);
    c.initialize().unwrap();
    c.initialize_synaptics_mouse().unwrap();
    c.poll_synaptics().unwrap();
    c.stop().unwrap();
    {
        let mut hw = h.borrow_mut();
        let child = hw.synaptics.as_mut().unwrap().child.as_mut().unwrap();
        child.wheel = false;
        child.explorer = false;
    }
    c.initialize().unwrap();
    assert_eq!(
        c.initialize_synaptics_mouse().unwrap_err().kind,
        ErrorKind::Synaptics(synaptics::Error::ChildIdentityChanged)
    );
    assert!(
        !h.borrow()
            .synaptics
            .as_ref()
            .unwrap()
            .child
            .as_ref()
            .unwrap()
            .scanning
    );
}
#[test]
fn every_child_reply_is_required_and_late_observations_survive_failure() {
    for reply in 1..=27 {
        for late in [false, true] {
            let (mut c, h) = child_target(true, true);
            c.initialize().unwrap();
            {
                let mut hw = h.borrow_mut();
                let child = hw.synaptics.as_mut().unwrap().child.as_mut().unwrap();
                if late {
                    child.late_reply = Some(reply);
                } else {
                    child.omit_reply = Some(reply);
                }
            }
            let result = c.initialize_synaptics_mouse();
            if !late && matches!(reply, 14 | 22) {
                assert_eq!(result.unwrap(), MouseProtocol::Explorer);
                assert_eq!(
                    c.synaptics_child_identities()
                        .unwrap()
                        .iter()
                        .filter(|i| !i.acknowledged)
                        .count(),
                    1
                );
                continue;
            }
            assert!(result.is_err(), "reply{reply} late{late}");
            assert!(c.needs_reset());
            assert_eq!(c.synaptics_child_protocol(), None);
            if late {
                assert!(!c.synaptics_child_command().unwrap().observations.is_empty());
                assert!(!c.drain_synaptics_passthrough().is_empty());
            }
            let count = h.borrow().writes.len();
            assert_eq!(
                c.initialize_synaptics_mouse().unwrap_err().kind,
                ErrorKind::NeedsReset
            );
            assert_eq!(h.borrow().writes.len(), count);
        }
    }
}
#[test]
fn child_get_id_without_ack_keeps_identity_and_does_not_invent_acknowledgement() {
    for byte in [0, 3, 4] {
        let (mut c, h) = child_target(true, true);
        c.initialize().unwrap();
        h.borrow_mut()
            .synaptics
            .as_mut()
            .unwrap()
            .before_client_ack
            .push([0x84, byte, 0, 0xc4, 0, 0]);
        assert_eq!(
            c.initialize_synaptics_mouse().unwrap_err().kind,
            ErrorKind::Synaptics(synaptics::Error::ChildReply(byte))
        );
        assert_eq!(c.synaptics_child_command().unwrap().opcode, RESET);
        assert_eq!(c.synaptics_child_command().unwrap().acknowledged, 0);
    }
    for (wheel, explorer, protocol) in [
        (false, false, MouseProtocol::Standard),
        (true, false, MouseProtocol::IntelliMouse),
        (true, true, MouseProtocol::Explorer),
    ] {
        let (mut c, h) = child_target(wheel, explorer);
        c.initialize().unwrap();
        h.borrow_mut()
            .synaptics
            .as_mut()
            .unwrap()
            .child
            .as_mut()
            .unwrap()
            .no_id_ack = true;
        assert_eq!(c.initialize_synaptics_mouse().unwrap(), protocol);
        let identities = c.synaptics_child_identities().unwrap();
        assert_eq!(identities.len(), 2);
        assert!(identities.iter().all(|id| !id.acknowledged));
        assert_eq!(identities[0].observation.byte, if wheel { 3 } else { 0 });
        assert_eq!(
            identities[1].observation.byte,
            if explorer {
                4
            } else if wheel {
                3
            } else {
                0
            }
        );
        assert_eq!(c.drain_synaptics_passthrough().len(), 25);
    }
    let (mut c, h) = child_target(true, true);
    c.initialize().unwrap();
    {
        let mut hw = h.borrow_mut();
        let child = hw.synaptics.as_mut().unwrap().child.as_mut().unwrap();
        child.no_id_ack = true;
        child.force_id = Some(0x42);
    }
    assert_eq!(
        c.initialize_synaptics_mouse().unwrap_err().kind,
        ErrorKind::Synaptics(synaptics::Error::ChildReply(0x42))
    );
    let command = c.synaptics_child_command().unwrap();
    assert_eq!((command.opcode, command.acknowledged), (GET_ID, 0));
    assert_eq!(command.observations.last().unwrap().byte, 0x42);
    assert!(c.synaptics_child_identities().unwrap().is_empty());
}

#[test]
fn consumed_child_id_is_distinct_from_later_queued_unsolicited_data() {
    for ackless in [false, true] {
        let (mut c, h) = child_target(true, true);
        c.initialize().unwrap();
        {
            let mut hw = h.borrow_mut();
            let child = hw.synaptics.as_mut().unwrap().child.as_mut().unwrap();
            child.no_id_ack = ackless;
            child.replies_before_parent_ack = true;
            child.get_id_tail = Some(0x55);
        }
        assert_eq!(
            c.initialize_synaptics_mouse().unwrap_err().kind,
            ErrorKind::Synaptics(synaptics::Error::ChildUnsolicited)
        );
        let identity = c.synaptics_child_identities().unwrap()[0];
        assert_eq!(identity.observation.byte, 3);
        assert_eq!(identity.acknowledged, !ackless);
        let command = c.synaptics_child_command().unwrap();
        assert_eq!(
            (command.opcode, command.acknowledged),
            (GET_ID, u8::from(!ackless))
        );
        let tail = *command.observations.last().unwrap();
        assert_eq!(tail.byte, 0x55);
        assert_eq!(
            identity.observation.native_sequence + 1,
            tail.native_sequence
        );
        let records = c.drain_synaptics_passthrough();
        assert_eq!(records.last().unwrap().first(), 0x55);
        assert!(
            records.iter().any(
                |r| r.native_sequence == identity.observation.native_sequence && r.first() == 3
            )
        );
        assert_eq!(c.synaptics_child_protocol(), None);
        assert!(c.needs_reset());
        assert_eq!(
            h.borrow()
                .synaptics
                .as_ref()
                .unwrap()
                .child
                .as_ref()
                .unwrap()
                .writes
                .last(),
            Some(&GET_ID)
        );
        let writes = h.borrow().writes.len();
        assert_eq!(
            c.initialize_synaptics_mouse().unwrap_err().kind,
            ErrorKind::NeedsReset
        );
        assert_eq!(h.borrow().writes.len(), writes);
    }
}

#[test]
fn child_ack_and_data_may_precede_the_later_parent_outer_ack() {
    for (wheel, explorer, protocol) in [
        (false, false, MouseProtocol::Standard),
        (true, false, MouseProtocol::IntelliMouse),
        (true, true, MouseProtocol::Explorer),
    ] {
        let (mut c, h) = child_target(wheel, explorer);
        c.initialize().unwrap();
        h.borrow_mut()
            .synaptics
            .as_mut()
            .unwrap()
            .child
            .as_mut()
            .unwrap()
            .replies_before_parent_ack = true;
        assert_eq!(c.initialize_synaptics_mouse().unwrap(), protocol);
        assert_eq!(c.synaptics_child_protocol(), Some(protocol));
        assert_eq!(c.drain_synaptics_passthrough().len(), 27);
    }
}
#[test]
fn the_host_poll_drops_raw_frames_so_the_keyboard_keeps_working_past_32_packets() {
    let (mut c, h) = ready_synaptics(true);
    let before = h.borrow().consumed;
    let packet = new_packet(2000, 2000, 50, 5, 0);
    for _ in 0..40 {
        emit(&h, Port::Mouse, &packet);
    }
    emit(&h, Port::Keyboard, &[0x1c]);
    let mut inputs = Vec::new();
    for _ in 0..8 {
        inputs.extend(c.poll_host().unwrap());
    }
    assert!(
        inputs.contains(&Input::Key {
            usage: 4,
            pressed: true
        }),
        "{inputs:?}"
    );
    assert_eq!(h.borrow().consumed - before, 241);
    assert!(c.drain_synaptics_frames().is_empty());
}
#[test]
fn the_host_poll_delivers_the_pass_through_pointer() {
    let (mut c, h) = child_target(false, false);
    c.initialize().unwrap();
    c.initialize_synaptics_mouse().unwrap();
    c.poll_synaptics().unwrap();
    for _ in 0..100 {
        emit(&h, Port::Mouse, &wrapped([9, 1, 0, 0]));
    }
    let mut inputs = Vec::new();
    for _ in 0..40 {
        inputs.extend(c.poll_host().unwrap());
    }
    assert!(inputs.contains(&Input::MouseButtons(1)), "{inputs:?}");
    assert!(
        inputs
            .iter()
            .filter(|input| matches!(input, Input::Motion { .. }))
            .count()
            >= 99
    );
}
#[test]
fn child_backpressure_keeps_raw_packets() {
    let (mut c, h) = child_target(false, false);
    c.initialize().unwrap();
    c.initialize_synaptics_mouse().unwrap();
    c.poll_synaptics().unwrap();
    for _ in 0..100 {
        emit(&h, Port::Mouse, &wrapped([9, 1, 0, 0]));
    }
    for _ in 0..20 {
        c.poll().unwrap();
        c.drain_synaptics_passthrough();
    }
    let count = h.borrow().consumed;
    c.poll().unwrap();
    assert_eq!(
        h.borrow().consumed,
        count,
        "child queue imposes backpressure independently of raw draining"
    );
    let inputs = c.drain_synaptics_child_inputs();
    assert_eq!(inputs.len(), 64);
    assert!(inputs.contains(&Input::MouseButtons(1)));
    let mut motions = inputs
        .iter()
        .filter(|v| matches!(v, Input::Motion { dx: 1, dy: 0 }))
        .count();
    for _ in 0..10 {
        let polled = c.poll_synaptics().unwrap();
        motions += polled
            .child_inputs
            .iter()
            .filter(|v| matches!(v, Input::Motion { dx: 1, dy: 0 }))
            .count();
    }
    assert_eq!(motions, 100);
    emit(&h, Port::Mouse, &wrapped([0, 1, 0, 0]));
    assert_eq!(c.poll().unwrap_err().kind, ErrorKind::MouseSync(0));
    assert_eq!(c.drain_synaptics_passthrough()[0].fields()[0], 0);
    assert!(c.drain_synaptics_child_inputs().is_empty());
    c.release_inputs();
    assert!(
        c.drain_synaptics_child_inputs()
            .contains(&Input::MouseButtons(0))
    );
}
#[test]
fn bound_child_extra_buttons_are_reported_and_kept_across_physical_packets() {
    let (mut c, h) = child_target(true, true);
    {
        let mut hw = h.borrow_mut();
        let q = &mut hw.synaptics.as_mut().unwrap().queries;
        q.get_mut(&9).unwrap()[1] = 0x30;
        q.get_mut(&16).unwrap()[0] |= 1;
    }
    c.initialize().unwrap();
    c.initialize_synaptics_mouse().unwrap();
    c.poll_synaptics().unwrap();
    let mut raw = new_packet(2000, 2500, 50, 5, 0);
    raw[3] ^= 2;
    raw[4] = (raw[4] & !3) | 3;
    raw[5] = (raw[5] & !3) | 1;
    emit(&h, Port::Mouse, &raw);
    let first = c.poll_synaptics().unwrap();
    assert!(first.child_inputs.contains(&Input::MouseButtons(7)));
    emit(&h, Port::Mouse, &wrapped([8, 1, 0, 0x30]));
    let physical = c.poll_synaptics().unwrap();
    assert!(physical.child_inputs.contains(&Input::MouseButtons(31)));
    raw[3] ^= 2;
    emit(&h, Port::Mouse, &raw);
    assert!(
        c.poll_synaptics()
            .unwrap()
            .child_inputs
            .contains(&Input::MouseButtons(24))
    );
    c.stop().unwrap();
    assert!(
        c.drain_synaptics_child_inputs()
            .contains(&Input::MouseButtons(0))
    );
}
#[test]
fn explorer_scroll_requires_magic_acks() {
    let (mut c, h) = child_target(true, true);
    c.initialize().unwrap();
    c.initialize_synaptics_mouse().unwrap();
    c.poll_synaptics().unwrap();
    let first = h
        .borrow()
        .synaptics
        .as_ref()
        .unwrap()
        .child
        .as_ref()
        .unwrap()
        .writes
        .len();
    c.enable_synaptics_explorer_scroll().unwrap();
    assert_eq!(
        c.synaptics_child_scroll_mode(),
        Some(synaptics::child::ScrollMode::Explorer4)
    );
    assert_eq!(
        &h.borrow()
            .synaptics
            .as_ref()
            .unwrap()
            .child
            .as_ref()
            .unwrap()
            .writes[first..],
        &[
            DISABLE_REPORTING,
            SET_RATE,
            200,
            SET_RATE,
            80,
            SET_RATE,
            40,
            SET_RATE,
            100,
            ENABLE_REPORTING
        ]
    );
    assert!(
        h.borrow()
            .synaptics
            .as_ref()
            .unwrap()
            .child
            .as_ref()
            .unwrap()
            .enhanced
    );
    c.poll_synaptics().unwrap();
    let writes = h.borrow().writes.len();
    c.enable_synaptics_explorer_scroll().unwrap();
    assert_eq!(h.borrow().writes.len(), writes);
    let mut actual_buttons = 0;
    let mut expected_buttons = 0;
    for extra in 0..=255 {
        emit(&h, Port::Mouse, &wrapped([9, 2, 3, extra]));
        let result = c.poll_synaptics().unwrap();
        let mode = extra & 0xc0;
        let (vertical, horizontal) = if matches!(mode, 0 | 0xc0) {
            expected_buttons = 1 | ((extra & 0x30) >> 1);
            (
                -(i16::from(extra & 15) - if extra & 8 != 0 { 16 } else { 0 }),
                0,
            )
        } else {
            expected_buttons = 1 | (expected_buttons & 24);
            let value = -(i16::from(extra & 63) - if extra & 32 != 0 { 64 } else { 0 });
            if mode == 0x40 { (0, value) } else { (value, 0) }
        };
        let mut observed_vertical = 0;
        for input in result.child_inputs {
            match input {
                Input::MouseButtons(buttons) => actual_buttons = buttons,
                Input::MouseWheel(value) => observed_vertical += i16::from(value),
                Input::Motion { dx, dy } => assert_eq!((dx, dy), (2, -3)),
                other => panic!("unexpected child event {other:?}"),
            }
        }
        assert_eq!(actual_buttons, expected_buttons, "byte{extra:02x}");
        assert_eq!(observed_vertical, vertical);
        assert_eq!(
            result
                .child_horizontal
                .iter()
                .map(|v| i16::from(*v))
                .sum::<i16>(),
            horizontal
        );
    }
    c.stop().unwrap();
    assert!(
        c.drain_synaptics_child_inputs()
            .contains(&Input::MouseButtons(0))
    );
    c.initialize().unwrap();
    c.initialize_synaptics_mouse().unwrap();
    assert_eq!(
        c.synaptics_child_scroll_mode(),
        Some(synaptics::child::ScrollMode::Ordinary)
    );
    assert!(
        !h.borrow()
            .synaptics
            .as_ref()
            .unwrap()
            .child
            .as_ref()
            .unwrap()
            .enhanced
    );
}
#[test]
fn each_explorer_scroll_reply_is_required_and_uncertain_magic_is_never_replayed() {
    for reply in 28..=37 {
        for late in [false, true] {
            let (mut c, h) = child_target(true, true);
            c.initialize().unwrap();
            c.initialize_synaptics_mouse().unwrap();
            c.poll_synaptics().unwrap();
            {
                let mut hw = h.borrow_mut();
                let child = hw.synaptics.as_mut().unwrap().child.as_mut().unwrap();
                if late {
                    child.late_reply = Some(reply);
                } else {
                    child.omit_reply = Some(reply);
                }
            }
            assert_eq!(
                c.enable_synaptics_explorer_scroll().unwrap_err().kind,
                ErrorKind::Timeout(Phase::Response),
                "reply{reply} late{late}"
            );
            assert!(c.needs_reset());
            assert_eq!(c.synaptics_child_protocol(), None);
            assert_eq!(
                c.synaptics_child_scroll_mode(),
                Some(if reply > 34 {
                    synaptics::child::ScrollMode::Explorer4
                } else {
                    synaptics::child::ScrollMode::Ordinary
                })
            );
            assert_eq!(
                h.borrow()
                    .synaptics
                    .as_ref()
                    .unwrap()
                    .child
                    .as_ref()
                    .unwrap()
                    .enhanced,
                reply >= 34
            );
            let writes = h.borrow().writes.len();
            assert_eq!(
                c.enable_synaptics_explorer_scroll().unwrap_err().kind,
                ErrorKind::NeedsReset
            );
            assert_eq!(h.borrow().writes.len(), writes);
        }
    }
}
#[test]
fn horizontal_child_queue_has_its_own_backpressure_and_unqualified_profiles_cannot_enable_it() {
    for (wheel, explorer) in [(false, false), (true, false)] {
        let (mut c, h) = child_target(wheel, explorer);
        c.initialize().unwrap();
        c.initialize_synaptics_mouse().unwrap();
        let writes = h.borrow().writes.len();
        assert_eq!(
            c.enable_synaptics_explorer_scroll().unwrap_err().kind,
            ErrorKind::Synaptics(synaptics::Error::ChildScrollUnsupported)
        );
        assert_eq!(h.borrow().writes.len(), writes);
    }
    let (mut c, h) = child_target(true, true);
    c.initialize().unwrap();
    c.initialize_synaptics_mouse().unwrap();
    c.poll_synaptics().unwrap();
    c.enable_synaptics_explorer_scroll().unwrap();
    c.poll_synaptics().unwrap();
    for _ in 0..100 {
        emit(&h, Port::Mouse, &wrapped([8, 0, 0, 0x41]));
    }
    for _ in 0..20 {
        c.poll().unwrap();
        c.drain_synaptics_passthrough();
        c.drain_synaptics_child_inputs();
    }
    let consumed = h.borrow().consumed;
    c.poll().unwrap();
    assert_eq!(h.borrow().consumed, consumed);
    let first = c.drain_synaptics_child_horizontal();
    assert_eq!(first, vec![-1; 64]);
    let mut count = first.len();
    for _ in 0..10 {
        let result = c.poll_synaptics().unwrap();
        assert!(result.child_horizontal.iter().all(|v| *v == -1));
        count += result.child_horizontal.len();
    }
    assert_eq!(count, 100);
}

#[test]
fn cleanup_acknowledges_relative_mode_and_child_stop_before_retiring_held_input() {
    let (mut c, h) = child_target(true, true);
    h.borrow_mut()
        .synaptics
        .as_mut()
        .unwrap()
        .queries
        .get_mut(&12)
        .unwrap()[0] |= 8;
    c.initialize().unwrap();
    c.initialize_synaptics_mouse().unwrap();
    c.poll_synaptics().unwrap();
    c.enable_synaptics_explorer_scroll().unwrap();
    c.poll_synaptics().unwrap();
    emit(&h, Port::Mouse, &new_packet(2000, 2500, 50, 5, 1));
    emit(&h, Port::Mouse, &wrapped([9, 0, 0, 0]));
    emit(&h, Port::Keyboard, &[0x1c]);
    c.poll_synaptics().unwrap();
    let record = wrapped([0xfa, 2, 3, 4]);
    h.borrow_mut()
        .synaptics
        .as_mut()
        .unwrap()
        .before_mode_ack
        .push(record);
    let releases = c.cleanup_synaptics().unwrap();
    assert!(releases.contains(&Input::Buttons(0)));
    assert!(releases.contains(&Input::Key {
        usage: 4,
        pressed: false
    }));
    assert!(
        c.drain_synaptics_child_inputs()
            .contains(&Input::MouseButtons(0))
    );
    assert!(!c.is_ready());
    assert!(!c.needs_reset());
    assert_eq!(c.synaptics_capabilities().unwrap().mode, Some(0));
    assert!(!c.synaptics_capabilities().unwrap().advanced_mode);
    assert_eq!(h.borrow().synaptics.as_ref().unwrap().mode, Some(0));
    assert_eq!(
        h.borrow().config & (KEYBOARD_DISABLED | MOUSE_DISABLED),
        KEYBOARD_DISABLED | MOUSE_DISABLED
    );
    assert_eq!(h.borrow().scanning, [false, false]);
    assert!(
        !h.borrow()
            .synaptics
            .as_ref()
            .unwrap()
            .child
            .as_ref()
            .unwrap()
            .scanning
    );
    let retained = c.drain_synaptics_passthrough();
    assert_eq!(retained.last().unwrap().raw, record);
    assert_eq!(retained.last().unwrap().programmed_format, None);
    c.initialize().unwrap();
    assert_eq!(c.synaptics_capabilities().unwrap().mode, Some(0xc5));
    assert!(c.synaptics_capabilities().unwrap().advanced_mode);
    assert_eq!(c.synaptics_child_protocol(), None);
}
#[test]
fn cleanup_failures_keep_held_state() {
    for late_mode in [false, true] {
        let (mut c, h) = ready_synaptics(true);
        emit(&h, Port::Mouse, &new_packet(2000, 2500, 50, 5, 1));
        c.poll_synaptics().unwrap();
        {
            let mut hw = h.borrow_mut();
            let s = hw.synaptics.as_mut().unwrap();
            if late_mode {
                s.late_mode_ack = true;
            } else {
                s.busy_after_mode_zero = true;
            }
        }
        assert!(matches!(
            c.cleanup_synaptics().unwrap_err().kind,
            ErrorKind::Timeout(_)
        ));
        assert!(c.needs_reset());
        assert_eq!(h.borrow().synaptics.as_ref().unwrap().mode, Some(0));
        assert_eq!(
            c.synaptics_capabilities().unwrap().mode,
            Some(if late_mode { 0xc5 } else { 0 })
        );
        assert!(c.drain_events().is_empty());
        assert!(c.release_inputs().contains(&Input::Buttons(0)));
        let writes = h.borrow().writes.len();
        assert_eq!(
            c.cleanup_synaptics().unwrap_err().kind,
            ErrorKind::NeedsReset
        );
        assert_eq!(h.borrow().writes.len(), writes);
    }
    let (mut c, h) = controller(Config::default());
    c.initialize().unwrap();
    let writes = h.borrow().writes.len();
    assert_eq!(
        c.cleanup_synaptics().unwrap_err().kind,
        ErrorKind::NotInitialized
    );
    assert_eq!(h.borrow().writes.len(), writes);
}

#[test]
fn runtime_rates_use_source_threshold_preserve_other_modes_and_survive_clean_restart() {
    for new in [false, true] {
        let (mut c, h) = target(new, true);
        c.initialize().unwrap();
        let fixed = c.synaptics_capabilities().unwrap().mode.unwrap() & !0x40;
        for requested in [0, 1, 79, 80, 81, 200, u32::MAX, 40] {
            assert_eq!(
                c.set_synaptics_rate(requested).unwrap(),
                if requested >= 80 { 80 } else { 40 }
            );
            let mode = fixed | if requested >= 80 { 0x40 } else { 0 };
            assert_eq!(c.synaptics_capabilities().unwrap().mode, Some(mode));
            assert_eq!(h.borrow().synaptics.as_ref().unwrap().mode, Some(mode));
            assert!(h.borrow().scanning[1]);
        }
        c.stop().unwrap();
        c.initialize().unwrap();
        assert_eq!(c.synaptics_capabilities().unwrap().mode, Some(fixed));
    }
    let (mut c, h) = passthrough_target();
    h.borrow_mut()
        .synaptics
        .as_mut()
        .unwrap()
        .queries
        .get_mut(&12)
        .unwrap()[0] |= 8;
    c.initialize().unwrap();
    c.configure_synaptics_passthrough(synaptics::passthrough::Format::Four)
        .unwrap();
    let raw = [0x84, 0xfa, 1, 0xc4, 2, 3];
    h.borrow_mut()
        .synaptics
        .as_mut()
        .unwrap()
        .before_mode_ack
        .push(raw);
    c.set_synaptics_rate(40).unwrap();
    assert_eq!(c.synaptics_capabilities().unwrap().mode, Some(0x87));
    assert!(c.synaptics_capabilities().unwrap().advanced_mode);
    assert_eq!(
        c.drain_synaptics_passthrough()[0].programmed_format,
        Some(synaptics::passthrough::Format::Four)
    );
}
#[test]
fn runtime_rate_failures_keep_acknowledged_mode_and_never_replay_uncertain_commands() {
    for phase in 0..3 {
        let (mut c, h) = ready_synaptics(true);
        match phase {
            0 => h.borrow_mut().no_ack = Some((Port::Mouse, DISABLE_REPORTING)),
            1 => h.borrow_mut().synaptics.as_mut().unwrap().late_mode_ack = true,
            _ => h.borrow_mut().no_ack = Some((Port::Mouse, ENABLE_REPORTING)),
        }
        assert_eq!(
            c.set_synaptics_rate(40).unwrap_err().kind,
            ErrorKind::Timeout(Phase::Response)
        );
        assert!(c.needs_reset());
        assert_eq!(
            c.synaptics_capabilities().unwrap().mode,
            Some(if phase == 2 { 0x85 } else { 0xc5 })
        );
        assert_eq!(
            h.borrow().synaptics.as_ref().unwrap().mode,
            Some(if phase == 0 { 0xc5 } else { 0x85 })
        );
        let count = h.borrow().writes.len();
        assert_eq!(
            c.set_synaptics_rate(40).unwrap_err().kind,
            ErrorKind::NeedsReset
        );
        assert_eq!(h.borrow().writes.len(), count);
    }
    let (mut c, h) = controller(Config::default());
    c.initialize().unwrap();
    let count = h.borrow().writes.len();
    assert_eq!(
        c.set_synaptics_rate(80).unwrap_err().kind,
        ErrorKind::NotInitialized
    );
    assert_eq!(h.borrow().writes.len(), count);
}

#[test]
fn forcepad_requires_source_firmware_identifier_and_new_absolute_profile() {
    for firmware in [
        "SYN300D",
        "pnp: SYN300D",
        "PNP: ASYN300D",
        "PNP: SYN300D0",
        "PNP: SYN0000",
    ] {
        let (base, h) = target(true, true);
        let result = Controller::new_synaptics_forcepad(
            Model(h.clone()),
            Time(h.borrow().time.clone()),
            base.config,
            true,
            firmware,
        );
        assert!(matches!(
            result,
            Err(super::super::Error {
                kind: ErrorKind::InvalidConfiguration,
                ..
            })
        ));
        assert!(h.borrow().writes.is_empty());
    }
    for firmware in ["PNP: SYN300D", "PNP: SYN3014", "PNP: other syn300d other"] {
        let (base, h) = target(true, true);
        let mut c = Controller::new_synaptics_forcepad(
            Model(h.clone()),
            Time(h.borrow().time.clone()),
            base.config,
            true,
            firmware,
        )
        .unwrap();
        c.initialize().unwrap();
        assert!(c.is_ready());
    }
    let (mut c, h) = forcepad_target(false);
    assert_eq!(
        c.initialize().unwrap_err().kind,
        ErrorKind::Synaptics(synaptics::Error::ForcepadFormat)
    );
    assert_eq!(h.borrow().synaptics.as_ref().unwrap().mode, None);
}
#[test]
fn forcepad_pressure_requires_sustained_single_finger_then_latches_until_no_contact() {
    let (mut c, h) = forcepad_target(true);
    c.initialize().unwrap();
    let start = h.borrow().time.get() + 100;
    let cases = [
        (0, 50, 5, true, 0),
        (50_000, 50, 5, true, 0),
        (50_001, 50, 5, true, 1),
        (51_000, 50, 0, true, 1),
        (52_000, 1, 5, false, 1),
        (53_000, 0, 5, true, 0),
        (54_000, 50, 5, true, 0),
        (103_999, 50, 1, true, 0),
        (104_001, 50, 5, true, 0),
        (154_002, 50, 5, true, 1),
    ];
    for (time, z, width, above, expected) in cases {
        h.borrow().time.set(start + time);
        emit(&h, Port::Mouse, &force_packet(z, width, above));
        let polled = c.poll_synaptics().unwrap();
        assert_eq!(polled.frames[0].buttons, expected, "t={time}");
        assert_eq!(polled.frames[0].scroll, 0);
        assert!(
            !polled
                .inputs
                .iter()
                .any(|v| matches!(v, Input::Wheel(_) | Input::MouseWheel(_)))
        );
    }
    assert!(c.stop().unwrap().contains(&Input::Buttons(0)));
    c.initialize().unwrap();
    emit(&h, Port::Mouse, &force_packet(50, 5, true));
    assert_eq!(c.poll_synaptics().unwrap().frames[0].buttons, 0);
}
#[test]
fn forcepad_interleaved_raw_records_preserve_pressure_epoch_and_fault_release() {
    let (mut c, h) = forcepad_target(true);
    {
        let mut hw = h.borrow_mut();
        let queries = &mut hw.synaptics.as_mut().unwrap().queries;
        queries.get_mut(&2).unwrap()[2] |= 0x80;
        queries.get_mut(&12).unwrap()[0] |= 8;
    }
    c.initialize().unwrap();
    h.borrow().time.set(u64::MAX - 40_000);
    emit(&h, Port::Mouse, &force_packet(50, 5, true));
    assert_eq!(c.poll_synaptics().unwrap().frames[0].buttons, 0);
    h.borrow().time.set(10_001);
    emit(&h, Port::Mouse, &[0x84, 2, 0, 0xc0, 0, 0x20]);
    emit(&h, Port::Mouse, &[0x84, 0xfa, 0, 0xc4, 0, 0]);
    let raw = c.poll_synaptics().unwrap();
    assert!(raw.frames.is_empty());
    assert_eq!((raw.advanced.len(), raw.passthrough.len()), (1, 1));
    emit(&h, Port::Mouse, &force_packet(50, 5, true));
    assert_eq!(c.poll_synaptics().unwrap().frames[0].buttons, 1);
    emit(&h, Port::Mouse, &[0]);
    assert!(c.poll().is_err());
    assert!(c.drain_events().is_empty());
    assert!(c.release_inputs().contains(&Input::Buttons(0)));
    assert!(c.needs_reset());
}
#[test]
fn client_bytes_use_sliced_transport_without_fabricating_child_ack_or_identity() {
    let (mut c, h) = passthrough_target();
    c.initialize().unwrap();
    for byte in 0..=255 {
        let first = h.borrow().writes.len();
        c.write_synaptics_passthrough(byte).unwrap();
        let wire: Vec<_> = h.borrow().writes[first..]
            .iter()
            .filter(|(command, _)| !*command)
            .map(|(_, data)| *data)
            .collect();
        assert_eq!(
            wire,
            [
                0xe6,
                0xe8,
                (byte >> 6) & 3,
                0xe8,
                (byte >> 4) & 3,
                0xe8,
                (byte >> 2) & 3,
                0xe8,
                byte & 3,
                0xf3,
                0x28
            ]
        );
        assert_eq!(
            h.borrow().synaptics.as_ref().unwrap().client_writes.last(),
            Some(&byte)
        );
    }
    assert!(
        c.poll_synaptics().unwrap().passthrough.is_empty(),
        "outer ACK alone creates no child response"
    );
    let (mut c, h) = ready_synaptics(true);
    let first = h.borrow().writes.len();
    assert_eq!(
        c.write_synaptics_passthrough(0xff).unwrap_err().kind,
        ErrorKind::Synaptics(synaptics::Error::NoPassthrough)
    );
    assert_eq!(h.borrow().writes.len(), first);
    assert!(c.is_ready());
}
#[test]
fn wrapped_child_records_interleave_with_parent_replies_without_pointer_aliasing() {
    let (mut c, h) = passthrough_target();
    c.initialize().unwrap();
    let record = [0x84, 0xfa, 0xbb, 0xc4, 0xcc, 0xdd];
    h.borrow_mut()
        .synaptics
        .as_mut()
        .unwrap()
        .before_client_ack
        .push(record);
    c.write_synaptics_passthrough(0xff).unwrap();
    let polled = c.poll_synaptics().unwrap();
    assert!(polled.inputs.is_empty() && polled.frames.is_empty() && polled.advanced.is_empty());
    let received = polled.passthrough[0];
    assert_eq!(received.raw, record);
    assert_eq!(received.first(), 0xfa);
    assert_eq!(received.fields(), [0xfa, 0xcc, 0xdd, 0xbb]);
    assert_eq!(
        received.programmed_format,
        Some(synaptics::passthrough::Format::Three)
    );
    emit(&h, Port::Mouse, &new_packet(2000, 2500, 50, 5, 0));
    assert_eq!(
        c.poll_synaptics().unwrap().frames[0].native_sequence,
        received.native_sequence + 1
    );
}
#[test]
fn client_format_changes_require_all_acks_and_preserve_uncertain_raw_record_format() {
    let (mut c, h) = passthrough_target();
    c.initialize().unwrap();
    let packet = [0x84, 0xfa, 0x11, 0xc4, 0x22, 0x33];
    h.borrow_mut()
        .synaptics
        .as_mut()
        .unwrap()
        .before_mode_ack
        .push(packet);
    c.configure_synaptics_passthrough(synaptics::passthrough::Format::Four)
        .unwrap();
    assert_eq!(c.synaptics_capabilities().unwrap().mode, Some(0xc7));
    assert_eq!(h.borrow().synaptics.as_ref().unwrap().mode, Some(0xc7));
    assert!(h.borrow().scanning[1]);
    assert_eq!(c.drain_synaptics_passthrough()[0].programmed_format, None);
    emit(&h, Port::Mouse, &packet);
    assert_eq!(
        c.poll_synaptics().unwrap().passthrough[0].programmed_format,
        Some(synaptics::passthrough::Format::Four)
    );
    c.configure_synaptics_passthrough(synaptics::passthrough::Format::Three)
        .unwrap();
    assert_eq!(c.synaptics_capabilities().unwrap().mode, Some(0xc5));
    h.borrow_mut().synaptics.as_mut().unwrap().late_mode_ack = true;
    let error = c
        .configure_synaptics_passthrough(synaptics::passthrough::Format::Four)
        .unwrap_err();
    assert_eq!(error.kind, ErrorKind::Timeout(Phase::Response));
    assert_eq!(
        error.response,
        Some(ObservedResponse {
            port: Port::Mouse,
            data: ACK
        })
    );
    assert!(c.needs_reset());
    assert_eq!(c.synaptics_capabilities().unwrap().mode, Some(0xc5));
    assert_eq!(h.borrow().synaptics.as_ref().unwrap().mode, Some(0xc7));
    assert!(c.drain_synaptics_passthrough().is_empty());
}
#[test]
fn passthrough_failures_keep_ownership() {
    let (mut c, h) = passthrough_target();
    c.initialize().unwrap();
    h.borrow_mut().synaptics.as_mut().unwrap().late_client_ack = true;
    let error = c.write_synaptics_passthrough(0xff).unwrap_err();
    assert_eq!(error.kind, ErrorKind::Timeout(Phase::Response));
    assert_eq!(
        error.response,
        Some(ObservedResponse {
            port: Port::Mouse,
            data: ACK
        })
    );
    assert_eq!(h.borrow().synaptics.as_ref().unwrap().client_writes, [0xff]);
    let writes = h.borrow().writes.len();
    assert_eq!(
        c.write_synaptics_passthrough(0xff).unwrap_err().kind,
        ErrorKind::NeedsReset
    );
    assert_eq!(h.borrow().writes.len(), writes);
    let (mut c, h) = passthrough_target();
    c.initialize().unwrap();
    let packet = [0x84, 1, 2, 0xc4, 3, 4];
    for _ in 0..40 {
        emit(&h, Port::Mouse, &packet);
    }
    for _ in 0..4 {
        c.poll().unwrap();
    }
    let consumed = h.borrow().consumed;
    c.poll().unwrap();
    assert_eq!(h.borrow().consumed, consumed);
    assert_eq!(c.drain_synaptics_passthrough().len(), 32);
    assert_eq!(c.poll_synaptics().unwrap().passthrough.len(), 8);
    c.stop().unwrap();
    c.initialize().unwrap();
    assert_eq!(c.synaptics_capabilities().unwrap().mode, Some(0xc5));
}

#[test]
#[ignore = "oracle: STELA_SYNAPTICS_CLEANUP_ORACLE"]
fn unchanged_linux_cleanup_mode_matches_quiesced_i8042_owner() {
    extern crate std;
    let records =
        std::fs::read_to_string(std::env::var("STELA_SYNAPTICS_CLEANUP_ORACLE").unwrap()).unwrap();
    let mut cases = 0;
    for line in records.lines() {
        let fields: Vec<_> = line.split_whitespace().collect();
        assert_eq!(fields.len(), 2);
        let mode = fields[0].parse::<u8>().unwrap();
        let (mut c, h) = target(mode & 1 != 0, mode & 0x40 != 0);
        if mode & 2 != 0 {
            h.borrow_mut()
                .synaptics
                .as_mut()
                .unwrap()
                .queries
                .get_mut(&2)
                .unwrap()[2] |= 0x80;
        }
        c.initialize().unwrap();
        if mode & 2 != 0 {
            c.configure_synaptics_passthrough(synaptics::passthrough::Format::Four)
                .unwrap();
        }
        assert_eq!(c.synaptics_capabilities().unwrap().mode, Some(mode));
        let first = h.borrow().writes.len();
        c.cleanup_synaptics().unwrap();
        let expected: Vec<_> = fields[1]
            .as_bytes()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|p| u8::from_str_radix(core::str::from_utf8(p).unwrap(), 16).unwrap())
            .collect();
        let mut routed = false;
        let mut mouse_wire = Vec::new();
        for (command, byte) in &h.borrow().writes[first..] {
            if *command {
                routed = *byte == SEND_MOUSE;
            } else if routed {
                mouse_wire.push(*byte);
                routed = false;
            }
        }
        assert_eq!(mouse_wire[0], DISABLE_REPORTING);
        assert_eq!(&mouse_wire[1..], expected);
        assert_eq!(c.synaptics_capabilities().unwrap().mode, Some(0));
        assert!(!c.is_ready());
        cases += 1;
    }
    assert_eq!(cases, 6);
}

#[test]
#[ignore = "oracle: STELA_SYNAPTICS_BOARD_ORACLE"]
fn unchanged_linux_board_coordinates_match_pnp_queried_device() {
    extern crate std;
    let records =
        std::fs::read_to_string(std::env::var("STELA_SYNAPTICS_BOARD_ORACLE").unwrap()).unwrap();
    let mut cases = 0;
    for line in records.lines() {
        let fields: Vec<_> = line.split_whitespace().collect();
        assert_eq!(fields.len(), 6);
        let board = if fields[0] == "N" {
            None
        } else {
            Some(fields[0].parse::<u16>().unwrap())
        };
        let firmware: Vec<_> = fields[1]
            .as_bytes()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|p| u8::from_str_radix(core::str::from_utf8(p).unwrap(), 16).unwrap())
            .collect();
        let (mut c, _) = pnp_target(core::str::from_utf8(&firmware).unwrap(), board);
        c.initialize().unwrap();
        let number = |i: usize| fields[i].parse::<u16>().unwrap();
        let expected = ((number(2), number(3)), (number(4), number(5)));
        let cap = c.synaptics_capabilities().unwrap();
        assert_eq!(cap.board, board);
        assert_eq!(cap.minimum, Some((1000, 800)));
        assert_eq!(cap.maximum, Some((6000, 5000)));
        assert_eq!(cap.bounds(), expected, "source record {line}");
        assert_eq!(
            cap.coordinate_quirk.is_some(),
            expected != ((1000, 800), (6000, 5000))
        );
        cases += 1;
    }
    assert_eq!(cases, 8192);
}

#[test]
#[ignore = "oracle: STELA_SYNAPTICS_CHILD_ORACLE"]
fn unchanged_linux_child_packets_and_oob_match_bound_i8042_mouse() {
    extern crate std;
    let records =
        std::fs::read_to_string(std::env::var("STELA_SYNAPTICS_CHILD_ORACLE").unwrap()).unwrap();
    compare_linux_child_records(&records, false);
}
#[test]
#[ignore = "oracle: STELA_SYNAPTICS_EXPLORER_ORACLE"]
fn unchanged_linux_explorer_scroll_matches_acknowledged_child_mode() {
    extern crate std;
    let records =
        std::fs::read_to_string(std::env::var("STELA_SYNAPTICS_EXPLORER_ORACLE").unwrap()).unwrap();
    compare_linux_child_records(&records, true);
}
fn compare_linux_child_records(records: &str, enhanced: bool) {
    let mut device = None;
    let mut group = None;
    let mut native_buttons = 0;
    let mut cases = 0;
    for line in records.lines() {
        let fields: Vec<_> = line.split_whitespace().collect();
        assert_eq!(fields.len(), if enhanced { 10 } else { 9 });
        let number = |i: usize| fields[i].parse::<i32>().unwrap();
        let id = number(0);
        if group != Some(id) {
            let (mut c, h) = child_target(number(1) != 0, number(1) == 4);
            {
                let mut hw = h.borrow_mut();
                let q = &mut hw.synaptics.as_mut().unwrap().queries;
                q.get_mut(&9).unwrap()[1] = 0x30;
                q.get_mut(&16).unwrap()[0] |= 1;
            }
            c.initialize().unwrap();
            c.initialize_synaptics_mouse().unwrap();
            c.poll_synaptics().unwrap();
            if enhanced && number(1) == 4 {
                c.enable_synaptics_explorer_scroll().unwrap();
                c.poll_synaptics().unwrap();
            }
            device = Some((c, h));
            group = Some(id);
            native_buttons = 0;
        }
        let (c, h) = device.as_mut().unwrap();
        if number(3) >= 0 {
            let buttons = number(3) as u8;
            let wire = (buttons & 1) | ((buttons & 2) << 1) | ((buttons & 4) >> 1);
            let mut raw = new_packet(2000, 2500, 50, 5, 0);
            raw[3] ^= 2;
            raw[4] = (raw[4] & !3) | (wire & 3);
            raw[5] = (raw[5] & !3) | (wire >> 2);
            emit(h, Port::Mouse, &raw);
            for input in c.poll_synaptics().unwrap().child_inputs {
                if let Input::MouseButtons(buttons) = input {
                    native_buttons = buttons;
                }
            }
        }
        assert_eq!(
            i32::from(native_buttons),
            number(4),
            "OOB source record {line}"
        );
        let bytes: Vec<_> = fields[2]
            .as_bytes()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|p| u8::from_str_radix(core::str::from_utf8(p).unwrap(), 16).unwrap())
            .collect();
        emit(h, Port::Mouse, &wrapped(bytes.try_into().unwrap()));
        let mut dx = 0;
        let mut dy = 0;
        let mut wheel = 0;
        let polled = c.poll_synaptics().unwrap();
        let horizontal = polled
            .child_horizontal
            .iter()
            .map(|v| i32::from(*v))
            .sum::<i32>();
        for input in polled.child_inputs {
            match input {
                Input::MouseButtons(buttons) => native_buttons = buttons,
                Input::Motion { dx: x, dy: y } => {
                    dx += i32::from(x);
                    dy += i32::from(y);
                }
                Input::MouseWheel(value) => wheel += i32::from(value),
                other => panic!("unexpected child event {other:?}"),
            }
        }
        assert_eq!(
            (i32::from(native_buttons), dx, dy, wheel),
            (number(5), number(6), number(7), number(8)),
            "source record {line}"
        );
        assert_eq!(
            horizontal,
            if enhanced { number(9) } else { 0 },
            "source record {line}"
        );
        cases += 1;
    }
    assert_eq!(cases, 8192);
}

#[test]
#[ignore = "oracle: STELA_SYNAPTICS_RATE_ORACLE"]
fn unchanged_linux_rate_commands_match_i8042_controls() {
    extern crate std;
    let records =
        std::fs::read_to_string(std::env::var("STELA_SYNAPTICS_RATE_ORACLE").unwrap()).unwrap();
    let mut cases = 0;
    for line in records.lines() {
        let fields: Vec<_> = line.split_whitespace().collect();
        assert_eq!(fields.len(), 5);
        let number = |i: usize| fields[i].parse::<u32>().unwrap();
        let mode = number(0) as u8;
        let (mut c, h) = target(true, mode & 0x40 != 0);
        h.borrow_mut()
            .synaptics
            .as_mut()
            .unwrap()
            .queries
            .get_mut(&2)
            .unwrap()[2] |= 0x80;
        c.initialize().unwrap();
        if mode & 2 != 0 {
            c.configure_synaptics_passthrough(synaptics::passthrough::Format::Four)
                .unwrap();
        }
        assert_eq!(c.synaptics_capabilities().unwrap().mode, Some(mode));
        let first = h.borrow().writes.len();
        assert_eq!(
            u32::from(c.set_synaptics_rate(number(1)).unwrap()),
            number(2)
        );
        assert_eq!(
            c.synaptics_capabilities().unwrap().mode,
            Some(number(3) as u8)
        );
        let expected: Vec<_> = fields[4]
            .as_bytes()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| u8::from_str_radix(core::str::from_utf8(pair).unwrap(), 16).unwrap())
            .collect();
        let wire: Vec<_> = h.borrow().writes[first..]
            .iter()
            .filter(|(command, _)| !*command)
            .map(|(_, byte)| *byte)
            .collect();
        assert_eq!(wire, expected, "source record {line}");
        cases += 1;
    }
    assert_eq!(cases, 1028);
}

#[test]
#[ignore = "oracle: STELA_SYNAPTICS_FORCEPAD_ORACLE"]
fn unchanged_linux_forcepad_temporal_records_match_i8042_reports() {
    extern crate std;
    let records =
        std::fs::read_to_string(std::env::var("STELA_SYNAPTICS_FORCEPAD_ORACLE").unwrap()).unwrap();
    let mut device = None;
    let mut group = None;
    let mut cases = 0;
    for line in records.lines() {
        let fields: Vec<_> = line.split_whitespace().collect();
        assert_eq!(fields.len(), 5);
        let id = fields[0].parse::<u16>().unwrap();
        if group != Some(id) {
            let (mut c, h) = forcepad_target(true);
            if fields[1] == "1" {
                h.borrow_mut()
                    .synaptics
                    .as_mut()
                    .unwrap()
                    .queries
                    .get_mut(&12)
                    .unwrap()[0] |= 0x10;
            }
            c.initialize().unwrap();
            device = Some((c, h));
            group = Some(id);
        }
        let (c, h) = device.as_mut().unwrap();
        h.borrow().time.set(fields[2].parse::<u64>().unwrap());
        let raw: Vec<_> = fields[3]
            .as_bytes()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| u8::from_str_radix(core::str::from_utf8(pair).unwrap(), 16).unwrap())
            .collect();
        emit(h, Port::Mouse, &raw);
        let polled = c.poll_synaptics().unwrap();
        assert_eq!(polled.frames[0].raw.as_slice(), raw);
        assert_eq!(
            polled.frames[0].buttons,
            fields[4].parse::<u8>().unwrap(),
            "source record {line}"
        );
        assert_eq!(polled.frames[0].scroll, 0);
        cases += 1;
    }
    assert_eq!(cases, 8192);
}

#[test]
#[ignore = "oracle: STELA_SYNAPTICS_PT_ORACLE"]
fn unchanged_linux_passthrough_commands_and_fields_match_i8042_transport() {
    extern crate std;
    let records =
        std::fs::read_to_string(std::env::var("STELA_SYNAPTICS_PT_ORACLE").unwrap()).unwrap();
    let hex = |value: &str| -> Vec<u8> {
        value
            .as_bytes()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| u8::from_str_radix(core::str::from_utf8(pair).unwrap(), 16).unwrap())
            .collect()
    };
    let mut cases = 0;
    for line in records.lines() {
        let fields: Vec<_> = line.split_whitespace().collect();
        let number = |i: usize| fields[i].parse::<u8>().unwrap();
        let (mut c, h) = passthrough_target();
        c.initialize().unwrap();
        match fields[0] {
            "W" | "M" => {
                let first = h.borrow().writes.len();
                if fields[0] == "W" {
                    c.write_synaptics_passthrough(number(1)).unwrap();
                } else {
                    c.configure_synaptics_passthrough(if number(1) == 4 {
                        synaptics::passthrough::Format::Four
                    } else {
                        synaptics::passthrough::Format::Three
                    })
                    .unwrap();
                    assert_eq!(c.synaptics_capabilities().unwrap().mode, Some(number(2)));
                }
                let wire: Vec<_> = h.borrow().writes[first..]
                    .iter()
                    .filter(|(command, _)| !*command)
                    .map(|(_, byte)| *byte)
                    .collect();
                assert_eq!(wire, hex(fields[fields.len() - 1]), "source record {line}");
            }
            "R" => {
                let format = if number(1) == 4 {
                    synaptics::passthrough::Format::Four
                } else {
                    synaptics::passthrough::Format::Three
                };
                c.configure_synaptics_passthrough(format).unwrap();
                let raw = hex(fields[3]);
                emit(&h, Port::Mouse, &raw);
                let polled = c.poll_synaptics().unwrap();
                assert!(
                    polled.inputs.is_empty()
                        && polled.frames.is_empty()
                        && polled.advanced.is_empty()
                );
                let received = polled.passthrough[0];
                assert_eq!(received.raw.as_slice(), raw);
                assert_eq!(received.programmed_format, Some(format));
                let count = usize::from(number(4));
                assert_eq!(
                    count,
                    if number(2) == 2 {
                        usize::from(number(1))
                    } else {
                        1
                    }
                );
                // Linux's callback view depends on its bound child state.
                // This comparison selects that view without inventing native binding.
                let bytes = if number(2) != 2 {
                    vec![received.first()]
                } else {
                    received.fields()[..count].to_vec()
                };
                assert_eq!(bytes, hex(fields[5]), "source record {line}");
            }
            _ => panic!("unknown source record"),
        }
        cases += 1;
    }
    assert_eq!(cases, 1794);
}

#[test]
#[ignore = "oracle: STELA_SYNAPTICS_BUTTONS_ORACLE"]
fn unchanged_linux_button_reports_match_i8042_extended_updates() {
    extern crate std;
    let records =
        std::fs::read_to_string(std::env::var("STELA_SYNAPTICS_BUTTONS_ORACLE").unwrap()).unwrap();
    let mut cases = 0;
    for line in records.lines() {
        let fields: Vec<_> = line.split_whitespace().collect();
        assert_eq!(fields.len(), 9);
        let number = |i: usize| fields[i].parse::<u16>().unwrap();
        let (mut c, h) = button_target(number(0) as u8, number(1) != 0, number(2), number(3) != 0);
        c.initialize().unwrap();
        let raw: Vec<_> = fields[4]
            .as_bytes()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| u8::from_str_radix(core::str::from_utf8(pair).unwrap(), 16).unwrap())
            .collect();
        emit(&h, Port::Mouse, &raw);
        let frame = c.poll_synaptics().unwrap().frames[0];
        assert_eq!(frame.raw.as_slice(), raw);
        assert_eq!(frame.buttons, number(6) as u8, "source record {line}");
        if fields[7] == "N" {
            assert_eq!(frame.extended_buttons, None);
        } else {
            let observed = frame.extended_buttons.unwrap();
            assert_eq!(
                observed.wire_bitmap,
                number(5) as u8,
                "source record {line}"
            );
            assert_eq!(
                observed.target,
                match fields[7] {
                    "T" => synaptics::buttons::Target::Touchpad(number(8) as u8),
                    "S" => synaptics::buttons::Target::Trackstick(number(8) as u8),
                    _ => panic!("unknown source target"),
                },
                "source record {line}"
            );
        }
        cases += 1;
    }
    assert_eq!(cases, 8192);
}

#[test]
#[ignore = "oracle: STELA_SYNAPTICS_IMAGE_ORACLE"]
fn unchanged_linux_image_count_policy_matches_i8042_reports() {
    extern crate std;
    let records =
        std::fs::read_to_string(std::env::var("STELA_SYNAPTICS_IMAGE_ORACLE").unwrap()).unwrap();
    let mut cases = 0;
    for line in records.lines() {
        let fields: Vec<_> = line.split_whitespace().collect();
        assert_eq!(fields.len(), 4);
        let raw: Vec<_> = fields[1]
            .as_bytes()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| u8::from_str_radix(core::str::from_utf8(pair).unwrap(), 16).unwrap())
            .collect();
        let count = fields[2].parse::<u8>().unwrap();
        let (mut c, h) = image_target();
        c.initialize().unwrap();
        emit(&h, Port::Mouse, &[0x84, count, 0, 0xc0, 0, 0x20]);
        c.poll_synaptics().unwrap();
        emit(&h, Port::Mouse, &raw);
        let result = c.poll_synaptics().unwrap();
        if fields[0] == "A" {
            assert!(result.frames.is_empty());
            assert_eq!(result.advanced[0].raw.as_slice(), raw);
        } else {
            assert_eq!(fields[0], "F");
            assert_eq!(result.frames[0].raw.as_slice(), raw);
            assert_eq!(
                result.frames[0].fingers,
                Some(fields[3].parse::<u8>().unwrap()),
                "source record {line}"
            );
        }
        cases += 1;
    }
    assert_eq!(cases, 8192);
}

#[test]
#[ignore = "oracle: STELA_SYNAPTICS_AGM_ORACLE"]
fn unchanged_linux_agm_records_match_advanced_mode_input() {
    extern crate std;
    let records =
        std::fs::read_to_string(std::env::var("STELA_SYNAPTICS_AGM_ORACLE").unwrap()).unwrap();
    let mut count = 0;
    for line in records.lines() {
        let fields: Vec<_> = line.split_whitespace().collect();
        assert_eq!(fields.len(), 6);
        let raw: Vec<_> = fields[0]
            .as_bytes()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| u8::from_str_radix(core::str::from_utf8(pair).unwrap(), 16).unwrap())
            .collect();
        let (mut c, h) = advanced_target();
        h.borrow_mut()
            .synaptics
            .as_mut()
            .unwrap()
            .queries
            .get_mut(&2)
            .unwrap()[2] &= !2;
        c.initialize().unwrap();
        emit(&h, Port::Mouse, &raw);
        let polled = c.poll_synaptics().unwrap();
        assert!(polled.inputs.is_empty() && polled.frames.is_empty());
        let observed = polled.advanced[0];
        assert_eq!(observed.raw.as_slice(), raw);
        let number = |i: usize| fields[i].parse::<u16>().unwrap();
        assert_eq!(
            observed.data,
            match number(1) {
                1 => synaptics::agm::Data::Secondary {
                    x: number(2),
                    y: number(3),
                    pressure: number(4) as u8
                },
                2 => synaptics::agm::Data::Count(number(5) as u8),
                other => synaptics::agm::Data::Unrecognized(other as u8),
            },
            "source record {line}"
        );
        count += 1;
    }
    assert_eq!(count, 8192);
}

#[test]
#[ignore = "oracle: STELA_SYNAPTICS_ORACLE"]
fn unchanged_linux_absolute_records_match_i8042_input() {
    extern crate std;
    let records =
        std::fs::read_to_string(std::env::var("STELA_SYNAPTICS_ORACLE").unwrap()).unwrap();
    let mut count = 0;
    for line in records.lines() {
        let fields: Vec<_> = line.split_whitespace().collect();
        assert_eq!(fields.len(), 9);
        let new = fields[0] == "1";
        let (mut c, h) = target(new, true);
        if fields[1] == "1" {
            h.borrow_mut()
                .synaptics
                .as_mut()
                .unwrap()
                .queries
                .get_mut(&12)
                .unwrap()[0] |= 0x10;
        }
        c.initialize().unwrap();
        let packet: Vec<_> = fields[2]
            .as_bytes()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| u8::from_str_radix(core::str::from_utf8(pair).unwrap(), 16).unwrap())
            .collect();
        emit(&h, Port::Mouse, &packet);
        let frame = c.poll_synaptics().unwrap().frames[0];
        assert_eq!(frame.raw.as_slice(), packet);
        let number = |index: usize| fields[index].parse::<i16>().unwrap();
        assert_eq!(
            (
                frame.x,
                frame.y,
                i16::from(frame.pressure),
                i16::from(frame.width),
                i16::from(frame.buttons),
                i16::from(frame.scroll)
            ),
            (
                number(3),
                number(4),
                number(5),
                number(6),
                number(7),
                number(8)
            ),
            "source record {line}"
        );
        count += 1;
    }
    assert_eq!(count, 8192);
}
