// SPDX-License-Identifier: GPL-2.0-or-later
// Linux HID input mapping reference and kept notices: NOTICE
// and the parent HID module.
//! The HID report translator every HID transport shares: a device's report
//! descriptor decodes each input report into descriptor events, and its
//! keyboard and pointer reports become the native peripheral's [`Input`]
//! changes. USB HID, I²C-HID and the HID transports behind ISH and SFH hand
//! their raw reports here.
use super::{
    KEY_USAGES, LED_PAGE, LED_USAGES, MODIFIER_USAGES, VARIABLE,
    descriptor::{DecodeError, Descriptor, Frame, ReportType},
    error::ParseError,
    events,
    report::{self, Field, Pointer, Reports},
};
use crate::input::{Input, POINTER_MAX};
use alloc::vec::Vec;

/// What one input report decoded to.
#[derive(Debug)]
pub enum Translated {
    /// A report the descriptor describes.
    Data(Frame),
    /// A report the descriptor rejects; no translation state changed.
    Rejected(DecodeError),
    /// A boot-protocol report, read with the Appendix B layout.
    Boot,
}

/// One HID device's translation state: what its last reports held.
pub struct Translator {
    descriptor: Option<Descriptor>,
    pub(super) reports: Reports,
    events: events::Decoder,
    /// The keyboard usages the last keyboard report held.
    pub(super) held: Vec<u32>,
    pub(super) buttons: u8,
    /// The absolute pointer's last position.
    pub(super) position: Option<(u16, u16)>,
    /// Keyboard reports holding usages the native keyboard cannot report.
    pub(super) dropped: u64,
    /// The last report received, as it arrived.
    pub(super) last_report: Vec<u8>,
}

impl Translator {
    /// A translator reading reports with `reports`; `descriptor` is the
    /// device's report descriptor, or `None` for boot protocol.
    pub(super) fn new(descriptor: Option<Descriptor>, reports: Reports) -> Self {
        Self {
            descriptor,
            reports,
            events: events::Decoder::default(),
            held: Vec::new(),
            buttons: 0,
            position: None,
            dropped: 0,
            last_report: Vec::new(),
        }
    }

    /// A translator for a report-protocol device whose report descriptor is
    /// `bytes`.
    pub fn describe(bytes: &[u8]) -> Result<Self, ParseError> {
        let descriptor = Descriptor::parse(bytes)?;
        let reports = report::parse(bytes)?;
        if let Some(pointer) = &reports.pointer
            && pointer.x.map(|f| f.relative) != pointer.y.map(|f| f.relative)
        {
            return Err(ParseError::pointer(
                pointer.x.map(|field| field.relative),
                pointer.y.map(|field| field.relative),
            ));
        }
        Ok(Self::new(Some(descriptor), reports))
    }

    pub fn descriptor(&self) -> Option<&Descriptor> {
        self.descriptor.as_ref()
    }

    /// The longest input report, including its report ID byte.
    pub fn input_bytes(&self) -> usize {
        self.descriptor
            .as_ref()
            .map_or(self.reports.input_bytes, Descriptor::input_bytes)
    }

    /// Whether the device describes a keyboard or a pointer report.
    pub fn is_keyboard_or_pointer(&self) -> bool {
        self.reports.keys.is_some() || self.reports.pointer.is_some()
    }

    /// The input report feeding the native relative pointer, if any.
    pub(crate) fn relative_pointer_report(&self) -> Option<u8> {
        let pointer = self.reports.pointer.as_ref()?;
        pointer.x?.relative.then_some(pointer.report.unwrap_or(0))
    }

    /// Keyboard reports holding usages the native keyboard cannot report.
    pub fn dropped(&self) -> u64 {
        self.dropped
    }

    /// The last report received, as it arrived, for diagnostics.
    pub fn last_report(&self) -> &[u8] {
        &self.last_report
    }

    /// Translate one input report, with its report ID byte when the device
    /// numbers its reports: its descriptor events onto `events`, and its key,
    /// pointer and button changes onto `inputs`.
    pub fn translate(
        &mut self,
        bytes: &[u8],
        events: &mut Vec<events::Event>,
        inputs: &mut Vec<Input>,
    ) -> Translated {
        let Some(descriptor) = &self.descriptor else {
            self.report(bytes, inputs);
            return Translated::Boot;
        };
        let frame = match descriptor.decode(bytes) {
            Ok(frame) => frame,
            Err(error) => {
                // Never partially mutate legacy state for a malformed report.
                self.last_report = bytes.to_vec();
                return Translated::Rejected(error);
            }
        };
        let events_at = events.len();
        events.extend(self.events.update(descriptor, &frame));
        // Native keys merge every report ID; a receiver's second keyboard
        // cannot release the first one's keys.
        let mut dropped = false;
        for event in &events[events_at..] {
            if let events::Event::Keyboard { usage, pressed } = event {
                if KEY_USAGES.contains(usage) || MODIFIER_USAGES.contains(usage) {
                    let was = self.held.contains(usage);
                    if was != *pressed {
                        inputs.push(Input::Key {
                            usage: *usage as u8,
                            pressed: *pressed,
                        });
                        if *pressed {
                            self.held.push(*usage);
                        } else {
                            self.held.retain(|u| u != usage);
                        }
                    }
                } else {
                    dropped = true;
                }
            }
        }
        if dropped {
            self.dropped += 1;
        }
        self.report(bytes, inputs);
        Translated::Data(frame)
    }

    /// Release everything the last reports held, as a stopped device's
    /// inputs: its keys, its buttons and its descriptor events' state.
    pub fn release(&mut self) -> (Vec<Input>, Vec<events::Event>) {
        let mut releases = Vec::new();
        for usage in self.held.drain(..) {
            releases.push(Input::Key {
                usage: usage as u8,
                pressed: false,
            });
        }
        if self.buttons != 0 {
            releases.push(
                if self
                    .reports
                    .pointer
                    .as_ref()
                    .is_some_and(|p| p.x.is_some_and(|f| f.relative))
                {
                    Input::MouseButtons(0)
                } else {
                    Input::Buttons(0)
                },
            );
        }
        self.buttons = 0;
        self.position = None;
        (releases, self.events.stop())
    }

    /// Linux `hidinput_led_worker`'s output report for `leds`, as its report
    /// ID and wire bytes. `leds` holds input LED codes as bits (LED_NUML is
    /// bit 0); each LED-page usage `hidinput_configure_usage` maps takes its
    /// bit. The output report holding the first such field is built with
    /// those fields set and every other field zero. `None` when the device
    /// has no LEDs.
    pub fn led_report(&self, leds: u16) -> Option<(u8, Vec<u8>)> {
        let descriptor = self.descriptor.as_ref()?;
        let led = |usage: u32| {
            (usage >> 16 == LED_PAGE)
                .then(|| {
                    LED_USAGES
                        .iter()
                        .position(|&led| u32::from(led) == usage & 0xffff)
                })
                .flatten()
        };
        let layout = descriptor
            .reports
            .iter()
            .filter(|layout| layout.kind == ReportType::Output)
            .find(|layout| {
                layout
                    .fields
                    .iter()
                    .any(|field| field.usages.iter().any(|&usage| led(usage).is_some()))
            })?;
        let id = layout.id;
        let mut report = alloc::vec![0u8; layout.wire_bytes()];
        let body = usize::from(id != 0);
        report[0] |= id;
        for field in layout.fields.iter().filter(|f| f.flags & VARIABLE != 0) {
            for element in 0..field.count {
                let usage = field.usages.get(element).or(field.usages.last());
                let Some(code) = usage.and_then(|&usage| led(usage)) else {
                    continue;
                };
                if leds & 1 << code != 0 {
                    let bit = field.offset + element * field.size;
                    report[body + bit / 8] |= 1 << (bit % 8);
                }
            }
        }
        Some((id, report))
    }

    fn report(&mut self, report: &[u8], inputs: &mut Vec<Input>) {
        self.last_report = report.to_vec();
        let (id, body) = if self.reports.numbered() {
            match report.split_first() {
                Some((&id, body)) => (Some(id), body),
                None => return,
            }
        } else {
            (None, report)
        };
        if self.descriptor.is_none()
            && let Some(keys) = &self.reports.keys
            && keys.report == id
        {
            self.keys(body, inputs);
        }
        if let Some(pointer) = &self.reports.pointer
            && pointer.report == id
        {
            let pointer = pointer.clone();
            self.pointer(&pointer, body, inputs);
        }
    }

    /// Key presses and releases since the last keyboard report: modifiers,
    /// then releases, then presses, each in report order.
    fn keys(&mut self, body: &[u8], inputs: &mut Vec<Input>) {
        let Some(keys) = &self.reports.keys else {
            return;
        };
        let Some(mut held) = keys.pressed_after(body, &self.held) else {
            return;
        };
        let before = held.len();
        held.retain(|usage| KEY_USAGES.contains(usage) || MODIFIER_USAGES.contains(usage));
        // Count each report that holds usages the native keyboard lacks.
        if held.len() != before {
            self.dropped += 1;
        }
        let key = |usage: u32, pressed| Input::Key {
            usage: usage as u8,
            pressed,
        };
        for usage in MODIFIER_USAGES {
            let (was, is) = (self.held.contains(&usage), held.contains(&usage));
            if was != is {
                inputs.push(key(usage, is));
            }
        }
        let modifier = |usage: &&u32| MODIFIER_USAGES.contains(*usage);
        for usage in self.held.iter().filter(|u| !modifier(u)) {
            if !held.contains(usage) {
                inputs.push(key(*usage, false));
            }
        }
        for usage in held.iter().filter(|u| !modifier(u)) {
            if !self.held.contains(usage) {
                inputs.push(key(*usage, true));
            }
        }
        self.held = held;
    }

    fn pointer(&mut self, layout: &Pointer, body: &[u8], inputs: &mut Vec<Input>) {
        let mouse = layout.x.is_some_and(|f| f.relative);
        let value = |field: Option<Field>| field.and_then(|f| f.read(body));
        if value(layout.x).is_none() || value(layout.y).is_none() {
            return;
        }
        if mouse {
            let counts = |v: i32| v.clamp(i16::MIN.into(), i16::MAX.into()) as i16;
            let (dx, dy) = (value(layout.x), value(layout.y));
            if let (Some(dx), Some(dy)) = (dx, dy)
                && (dx, dy) != (0, 0)
            {
                inputs.push(Input::Motion {
                    dx: counts(dx),
                    dy: counts(dy),
                });
            }
        } else {
            let axis = |field: Option<Field>| -> Option<u16> {
                let field = field?;
                let value = field.read(body)?;
                let max = i64::from(POINTER_MAX);
                let span = i64::from(field.maximum) - i64::from(field.minimum);
                if span <= 0 {
                    return None;
                }
                let scaled = (i64::from(value) - i64::from(field.minimum)) * max / span;
                Some(scaled.clamp(0, max) as u16)
            };
            if let (Some(x), Some(y)) = (axis(layout.x), axis(layout.y))
                && Some((x, y)) != self.position
            {
                self.position = Some((x, y));
                inputs.push(Input::Move { x, y });
            }
        }
        let mut buttons = 0;
        for (n, field) in layout.buttons.iter().enumerate() {
            if value(*field).is_some_and(|v| v != 0) {
                buttons |= 1 << n;
            }
        }
        // The legacy absolute-pointer frontend has three buttons. Complete
        // extra-button values remain in poll_hid's descriptor samples.
        if !mouse {
            buttons &= 7;
        }
        if buttons != self.buttons {
            self.buttons = buttons;
            inputs.push(if mouse {
                Input::MouseButtons(buttons)
            } else {
                Input::Buttons(buttons)
            });
        }
        if let Some(wheel) = value(layout.wheel)
            && wheel != 0
        {
            let detents = wheel.clamp(-127, 127) as i8;
            inputs.push(if mouse {
                Input::MouseWheel(detents)
            } else {
                Input::Wheel(detents)
            });
        }
        if let Some(pan) = value(layout.pan)
            && pan != 0
        {
            let detents = pan.clamp(-127, 127) as i8;
            inputs.push(if mouse {
                Input::MouseHWheel(detents)
            } else {
                Input::HWheel(detents)
            });
        }
    }
}
