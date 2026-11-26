// SPDX-License-Identifier: GPL-2.0-only
//! One host keyboard or pointer, whatever its transport: probe returns them
//! in one list and each is polled the same way for [`Input`] changes and sent
//! the guest's lock-key LEDs.
use super::{Input, hid_i2c, ish, ps2, touchpad, virtio::VirtioInput};
use crate::{Clock, block::BlockError, bus::i2c::Bus, usb::hid::UsbInput};
use alloc::boxed::Box;
use alloc::{format, string::String, vec::Vec};

/// A host keyboard or pointer.
pub enum InputDevice {
    Virtio(Box<VirtioInput>),
    Usb(Box<UsbInput>),
    /// A boot keyboard or mouse on an EHCI controller.
    Ehci(Box<crate::usb::ehci::input::BootInput>),
    /// The i8042 controller, for both of its ports, shared with the SMBus
    /// companion of its mouse port's touchpad.
    Ps2(ps2::Shared),
    I2cHid(Box<I2cHid>),
    /// An ELAN or Cypress touchpad, read as a mouse.
    Touchpad(Box<touchpad::TouchpadInput>),
    /// An Intel ISH with a keyboard or pointer client.
    Ish(Box<ish::IshInput>),
}

/// An I²C-HID device on whichever bus and interrupt pin probe found it.
pub type I2cHid = hid_i2c::HidInput<Box<dyn Bus>, Box<dyn hid_i2c::Interrupt>, Clock>;

/// Why polling an input device or setting its LEDs failed.
#[derive(Debug)]
pub enum InputError {
    Usb(BlockError),
    Ps2(ps2::Error),
    I2cHid(hid_i2c::Error),
    Touchpad(touchpad::Error),
    Ish(ish::Error),
}

impl core::fmt::Display for InputError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Usb(error) => write!(f, "{}", crate::usb::xhci::explain(error)),
            Self::Ps2(error) => write!(f, "{error}"),
            Self::I2cHid(error) => write!(f, "{error}"),
            Self::Touchpad(error) => write!(f, "{error}"),
            Self::Ish(error) => write!(f, "Intel ISH {error:?}"),
        }
    }
}

impl InputDevice {
    /// The device's name and location, for the boot log.
    pub fn name(&self) -> String {
        match self {
            Self::Virtio(device) => format!("virtio-input {:?}", device.name),
            Self::Usb(device) => format!("USB {}", device.name),
            Self::Ehci(device) => format!("USB {}", device.name),
            Self::Ps2(_) => String::from("i8042 PS/2"),
            Self::I2cHid(device) => device.name.clone(),
            Self::Touchpad(device) => device.name.clone(),
            Self::Ish(device) => device.name.clone(),
        }
    }

    /// Append the device's input changes since the last poll to `inputs`.
    /// On an error, what the device decoded before it is still appended.
    pub fn poll(&mut self, inputs: &mut Vec<Input>) -> Result<(), InputError> {
        match self {
            Self::Virtio(device) => {
                for event in device.poll() {
                    inputs.extend(device.translator.translate(event));
                }
                Ok(())
            }
            Self::Usb(device) => {
                inputs.extend(device.poll());
                Ok(())
            }
            Self::Ehci(device) => {
                inputs.extend(device.poll());
                Ok(())
            }
            Self::Ps2(controller) => {
                let mut controller = controller.borrow_mut();
                match controller.poll_host() {
                    Ok(polled) => {
                        inputs.extend(polled);
                        Ok(())
                    }
                    Err(error) => {
                        inputs.extend(controller.drain_events());
                        Err(InputError::Ps2(error))
                    }
                }
            }
            Self::I2cHid(device) => device.poll(inputs).map_err(InputError::I2cHid),
            Self::Touchpad(device) => device.poll(inputs).map_err(InputError::Touchpad),
            Self::Ish(device) => device.poll(inputs).map_err(InputError::Ish),
        }
    }

    /// Show `leds` on the device's keyboard LEDs: bit 0 Num Lock, bit 1
    /// Caps Lock, bit 2 Scroll Lock, then the other input LED codes. A
    /// device without LEDs ignores it.
    pub fn set_leds(&mut self, leds: u8) -> Result<(), InputError> {
        match self {
            Self::Virtio(_) | Self::Touchpad(_) => Ok(()),
            Self::Usb(device) => device.set_leds(u16::from(leds)).map_err(InputError::Usb),
            Self::Ehci(device) => device.set_leds(u16::from(leds)).map_err(InputError::Usb),
            Self::Ps2(controller) if controller.borrow().has_keyboard() => controller
                .borrow_mut()
                .set_leds(ps2_leds(leds))
                .map_err(InputError::Ps2),
            Self::Ps2(_) => Ok(()),
            Self::I2cHid(device) => device.set_leds(u16::from(leds)).map_err(InputError::I2cHid),
            Self::Ish(device) => device.set_leds(u16::from(leds)).map_err(InputError::Ish),
        }
    }

    /// The last HID report the device sent, as it arrived, for diagnostics;
    /// empty for a device that sends no HID reports.
    pub fn last_report(&self) -> &[u8] {
        match self {
            Self::Usb(device) => device.last_report(),
            Self::Ehci(device) => device.last_report(),
            Self::I2cHid(device) => device.last_report(),
            Self::Ish(device) => device.last_report(),
            Self::Virtio(_) | Self::Ps2(_) | Self::Touchpad(_) => &[],
        }
    }

    /// Whether the device stopped and is no longer read.
    pub fn stopped(&self) -> bool {
        match self {
            Self::Virtio(device) => device.broken,
            Self::Usb(device) => device.failed(),
            Self::Ehci(device) => device.failed(),
            Self::Ps2(controller) => controller.borrow().needs_reset(),
            Self::I2cHid(device) => device.device().needs_reset(),
            Self::Touchpad(device) => device.stopped(),
            Self::Ish(device) => device.stopped(),
        }
    }
}

/// The PS/2 keyboard's Set LEDs byte for input LED codes `leds`: Scroll,
/// Num and Caps Lock in bits 0, 1 and 2, as Linux `atkbd_set_leds` sends.
fn ps2_leds(leds: u8) -> u8 {
    (leds & 1) << 1 | (leds & 2) << 1 | (leds >> 2 & 1)
}

#[cfg(test)]
mod tests {
    use super::ps2_leds;

    #[test]
    fn lock_key_leds_take_their_ps2_bits() {
        assert_eq!(ps2_leds(0b001), 0b010, "Num Lock");
        assert_eq!(ps2_leds(0b010), 0b100, "Caps Lock");
        assert_eq!(ps2_leds(0b100), 0b001, "Scroll Lock");
        assert_eq!(ps2_leds(0b1000_0111), 0b111, "other LEDs are not sent");
    }
}
