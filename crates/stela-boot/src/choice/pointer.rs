// SPDX-License-Identifier: GPL-2.0-or-later

//! Firmware input stays open only while the graphical startup picker runs.
use super::{Press, await_event, keyboard_event, pressed};
use alloc::vec::Vec;
use core::time::Duration;
use uefi::{
    Event,
    boot::{self, OpenProtocolAttributes, OpenProtocolParams, ScopedProtocol},
    proto::{
        ProtocolPointer,
        console::pointer::{AbsolutePointer, Pointer},
    },
};

const SUBPIXEL: i128 = 256;
const TOUCH_ACTIVE: u32 = 1;

pub(super) enum Input {
    Key(Press),
    Pointer {
        position: (usize, usize),
        clicked: bool,
    },
    Idle,
    Timeout,
    Unavailable,
}

enum Protocol {
    Relative(ScopedProtocol<Pointer>),
    Absolute(ScopedProtocol<AbsolutePointer>),
}

struct Device {
    protocol: Protocol,
    button: bool,
}

impl Device {
    fn event(&self) -> Option<Event> {
        match &self.protocol {
            Protocol::Relative(pointer) => pointer.wait_for_input_event().ok(),
            Protocol::Absolute(pointer) => pointer.wait_for_input_event().ok(),
        }
    }

    fn sample(&mut self, cursor: &mut [i128; 2], limits: [usize; 2]) -> Result<Option<bool>, ()> {
        let held = match &mut self.protocol {
            Protocol::Relative(pointer) => {
                let Some(state) = pointer.read_state().map_err(|_| ())? else {
                    return Ok(None);
                };
                let mode = pointer.mode();
                for (axis, (motion, resolution)) in [
                    (state.relative_movement_x, mode.resolution_x),
                    (state.relative_movement_y, mode.resolution_y),
                ]
                .into_iter()
                .enumerate()
                {
                    let movement =
                        i128::from(motion) * (4 * SUBPIXEL) / i128::from(resolution.max(1));
                    cursor[axis] =
                        (cursor[axis] + movement).clamp(0, (limits[axis] - 1) as i128 * SUBPIXEL);
                }
                bool::from(state.left_button)
            }
            Protocol::Absolute(pointer) => {
                let Some(state) = pointer.read_state().map_err(|_| ())? else {
                    return Ok(None);
                };
                let mode = pointer.mode();
                for (axis, (position, min, max)) in [
                    (state.current_x, mode.absolute_min_x, mode.absolute_max_x),
                    (state.current_y, mode.absolute_min_y, mode.absolute_max_y),
                ]
                .into_iter()
                .enumerate()
                {
                    let range = max.checked_sub(min).filter(|&range| range != 0).ok_or(())?;
                    let position = position.clamp(min, max) - min;
                    cursor[axis] = (u128::from(position) * (limits[axis] - 1) as u128
                        / u128::from(range)) as i128
                        * SUBPIXEL;
                }
                state.active_buttons & TOUCH_ACTIVE != 0
            }
        };
        let clicked = held && !self.button;
        self.button = held;
        Ok(Some(clicked))
    }
}

fn find<P: ProtocolPointer + ?Sized>() -> Option<ScopedProtocol<P>> {
    let handle = boot::get_handle_for_protocol::<P>().ok()?;
    // SAFETY: these protocol opens are used exclusively by the picker at
    // application TPL and dropped before EFI options or drivers change.
    unsafe {
        boot::open_protocol(
            OpenProtocolParams {
                handle,
                agent: boot::image_handle(),
                controller: None,
            },
            OpenProtocolAttributes::GetProtocol,
        )
    }
    .ok()
}

pub(super) struct InputDevices {
    devices: Vec<Device>,
    cursor: [i128; 2],
    limits: [usize; 2],
}

impl InputDevices {
    pub(super) fn open((width, height): (usize, usize)) -> Self {
        let mut devices = Vec::new();
        if let Some(pointer) = find::<Pointer>() {
            devices.push(Device {
                protocol: Protocol::Relative(pointer),
                button: false,
            });
        }
        if let Some(pointer) = find::<AbsolutePointer>() {
            let mode = pointer.mode();
            if mode.absolute_max_x > mode.absolute_min_x
                && mode.absolute_max_y > mode.absolute_min_y
            {
                devices.push(Device {
                    protocol: Protocol::Absolute(pointer),
                    button: false,
                });
            }
        }
        Self {
            devices,
            cursor: [width as i128 * SUBPIXEL / 2, height as i128 * SUBPIXEL / 2],
            limits: [width, height],
        }
    }

    pub(super) fn wait(&mut self, timeout: Option<Duration>) -> Input {
        let keyboard = keyboard_event();
        let keyboard_events = usize::from(keyboard.is_some());
        let mut events: Vec<Event> = keyboard.into_iter().collect();
        let mut device_indices = Vec::new();
        for (index, device) in self.devices.iter().enumerate() {
            if let Some(event) = device.event() {
                events.push(event);
                device_indices.push(index);
            }
        }
        let Some(index) = await_event(events, timeout) else {
            return Input::Unavailable;
        };
        if index < keyboard_events {
            return pressed().map_or(Input::Idle, Input::Key);
        }
        let Some(&index) = device_indices.get(index - keyboard_events) else {
            return Input::Timeout;
        };
        let old = self.cursor;
        match self.devices[index].sample(&mut self.cursor, self.limits) {
            Ok(Some(clicked)) if self.cursor != old || clicked => Input::Pointer {
                position: (
                    (self.cursor[0] / SUBPIXEL) as usize,
                    (self.cursor[1] / SUBPIXEL) as usize,
                ),
                clicked,
            },
            Ok(_) => Input::Idle,
            Err(()) => {
                self.devices.remove(index);
                Input::Idle
            }
        }
    }
}
