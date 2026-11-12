//! Devices the ACPI namespace describes instead of PCI, attached through
//! [`NativeServices::acpi`](super::NativeServices::acpi): the i8042 keyboard
//! and mouse ports.
use super::PlatformError;
use crate::input::ps2;
use alloc::string::String;
use core::ops::Range;

/// The standard i8042 data and command/status ports.
const I8042_PORTS: [Range<u64>; 2] = [0x60..0x61, 0x64..0x65];

/// The controller configuration for the i8042 ports the namespace lists.
///
/// `io` holds the I/O ranges of the first `_CRS` that lists any, the keyboard's
/// before the mouse's, as Linux's i8042 PnP probe reads them: the data port
/// first, then the command port. A `_CRS` without I/O ranges takes the
/// standard ports, as Linux does. USB hosts firmware may still hold do not
/// gate the ports, as in Linux: the controller self-test that
/// initialization runs fails on ports USB legacy emulation traps badly.
pub(crate) fn i8042(
    keyboard: bool,
    mouse: bool,
    io: &[Range<u64>],
) -> Result<Option<ps2::Config>, PlatformError> {
    if !keyboard && !mouse {
        return Ok(None);
    }
    if !io.is_empty() && io.get(..2) != Some(&I8042_PORTS[..]) {
        return Err(PlatformError::Refused(
            "i8042 at ports other than 0x60 and 0x64",
        ));
    }
    Ok(Some(ps2::Config {
        keyboard,
        mouse,
        ..ps2::Config::default()
    }))
}

/// Byte offset of the FADT's IA-PC boot architecture flags.
const FADT_BOOT_ARCH: usize = 109;
/// IAPC_BOOT_ARCH bit: the motherboard has an 8042 or equivalent.
const FADT_8042: u16 = 1 << 1;

/// The firmware's i8042 declaration. This carries no port ownership or
/// evidence that an i8042 answers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum I8042Fadt {
    Absent,
    LegacyRevision(u8),
    Truncated { revision: Option<u8> },
    FlagClear { revision: u8 },
    FlagSet { revision: u8 },
}

impl I8042Fadt {
    /// Inspect the fields of a FADT identified by the caller.
    pub fn observe(fadt: Option<&[u8]>) -> Self {
        let Some(fadt) = fadt else {
            return Self::Absent;
        };
        let Some(&revision) = fadt.get(8) else {
            return Self::Truncated { revision: None };
        };
        if revision < 2 {
            return Self::LegacyRevision(revision);
        }
        let Some(flags) = fadt.get(FADT_BOOT_ARCH..FADT_BOOT_ARCH + 2) else {
            return Self::Truncated {
                revision: Some(revision),
            };
        };
        if u16::from_le_bytes([flags[0], flags[1]]) & FADT_8042 == 0 {
            Self::FlagClear { revision }
        } else {
            Self::FlagSet { revision }
        }
    }

    /// Preserve the direct-probe fallback when there is no modern FADT
    /// declaration and the namespace lists no i8042 device.
    fn permits_direct_probe(self) -> bool {
        matches!(
            self,
            Self::Absent
                | Self::LegacyRevision(_)
                | Self::Truncated { revision: None }
                | Self::FlagSet { .. }
        )
    }
}

/// The mouse port's firmware identifier as Linux's i8042 PnP driver forms it
/// from the device's `ids`, its `_HID` and then its `_CID`s: `PNP:` and each
/// ID in PnP form, the first of them once, with the hex digits lowercase as
/// `pnp_add_id` stores them. None when no ID is in PnP form, as `pnpacpi`
/// then adds no PnP device.
pub(crate) fn firmware_id(ids: &[String]) -> Option<String> {
    let pnp = |id: &&String| {
        let bytes = id.as_bytes();
        bytes.len() == 7
            && bytes[..3].iter().all(u8::is_ascii_uppercase)
            && bytes[3..]
                .iter()
                .all(|c| c.is_ascii_digit() || (b'A'..=b'F').contains(c))
    };
    let primary = ids.iter().find(pnp)?;
    let others = ids.iter().filter(pnp).filter(|id| *id != primary);
    let mut text = String::from("PNP:");
    for id in core::iter::once(primary).chain(others) {
        text.push(' ');
        text.push_str(&id[..3]);
        text.extend(id[3..].chars().map(|c| c.to_ascii_lowercase()));
    }
    Some(text)
}

/// Attach the i8042 when the namespace lists a keyboard or mouse port. When
/// it lists neither and the FADT expects an 8042, probe the keyboard port at
/// the standard ports, as Linux's i8042 probes them directly; an absent
/// controller is then not a failure. The auxiliary port is probed only when
/// the namespace lists a mouse, since a failing mouse port would take the
/// keyboard with it.
#[cfg(feature = "native-acpi")]
pub(super) fn attach_i8042(
    session: &crate::platform::acpi::Session,
    clock: crate::Clock,
    smbus_hosts: &[super::smbus::Host],
    usb_legacy_released: bool,
    devices: &mut super::Devices,
) {
    use super::{i2c::hardware_id, services::ids};
    use crate::platform::acpi::{
        Handle, Value,
        resources::{self, Resource, Space},
    };
    use alloc::vec::Vec;

    let found = |hids: &[&str]| -> Result<Vec<_>, PlatformError> {
        let mut handles = Vec::new();
        for hid in hids {
            handles.extend(
                session
                    .devices(hid)
                    .map_err(|error| PlatformError::Acpi(error.into()))?,
            );
        }
        Ok(handles)
    };
    // The `_HID` and `_CID`s ACPICA reads for a device's PnP IDs.
    let pnp_ids = |device: Handle<'_>| -> Result<Vec<String>, PlatformError> {
        let acpi = |error: crate::platform::acpi::Error| PlatformError::Acpi(error.into());
        let mut ids: Vec<String> = hardware_id(device.evaluate(Some("_HID"), &[]).map_err(acpi)?)
            .into_iter()
            .collect();
        if device.child("_CID").is_ok() {
            match device.evaluate(Some("_CID"), &[]).map_err(acpi)? {
                Some(Value::Package(cids)) => {
                    ids.extend(cids.into_iter().filter_map(|cid| hardware_id(Some(cid))));
                }
                cid => ids.extend(hardware_id(cid)),
            }
        }
        Ok(ids)
    };
    let mut direct = false;
    let mut attempt = || -> Result<Option<ps2::Controller>, PlatformError> {
        let keyboards = found(ids::I8042_KEYBOARD)?;
        let mice = found(ids::I8042_MOUSE)?;
        let mut io = Vec::new();
        for device in keyboards.iter().chain(&mice).take(1) {
            for descriptor in resources::evaluate(*device, "_CRS").map_err(PlatformError::Acpi)? {
                if let Resource::Address(address) = descriptor.resource
                    && address.space == Space::Io
                    && address.minimum_fixed
                {
                    io.push(address.minimum..address.minimum + address.length);
                }
            }
        }
        direct = keyboards.is_empty()
            && mice.is_empty()
            && I8042Fadt::observe(
                session
                    .table(b"FACP")
                    .map_err(|error| PlatformError::Acpi(error.into()))?
                    .as_deref(),
            )
            .permits_direct_probe();
        let Some(config) = i8042(direct || !keyboards.is_empty(), !mice.is_empty(), &io)? else {
            return Ok(None);
        };
        if !usb_legacy_released {
            return Err(PlatformError::Refused(
                "USB legacy interception was not ruled out",
            ));
        }
        // SAFETY: the namespace lists an i8042 at the standard ports, or the
        // FADT says the board has one there, and the mount's NativeServices
        // contract grants those ports with IRQ1 and IRQ12 masked. Every USB
        // host was checked for legacy interception before probing the ports.
        let registers = unsafe { ps2::PortIo::claim() };
        let firmware_id = match mice.first() {
            Some(mouse) => firmware_id(&pnp_ids(*mouse)?),
            None => None,
        };
        let mut controller =
            ps2::Controller::new_probing(registers, clock, config, firmware_id.as_deref())
                .map_err(PlatformError::Ps2)?;
        controller.initialize().map_err(PlatformError::Ps2)?;
        Ok(Some(controller))
    };
    let attached = attempt();
    match attached {
        Ok(Some(controller)) => {
            let companion = controller.smbus_companion();
            let shared = alloc::rc::Rc::new(core::cell::RefCell::new(controller));
            devices
                .inputs
                .push(crate::input::InputDevice::Ps2(shared.clone()));
            if let Some(companion) = companion {
                super::companion::attach(&shared, companion, smbus_hosts, clock, devices);
            }
        }
        Ok(None) => {}
        // Linux's direct probe finds no controller and stays quiet.
        Err(PlatformError::Ps2(error)) if direct && error.kind == ps2::ErrorKind::NoController => {}
        Err(error) => devices.platform_failures.push((
            String::from(if direct {
                "i8042 (FADT 8042 flag, no ACPI device)"
            } else {
                "i8042"
            }),
            error,
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_ports_attach_the_listed_ports() {
        let config = i8042(true, false, &I8042_PORTS).unwrap().unwrap();
        assert!(config.keyboard && !config.mouse);
        let config = i8042(false, true, &[]).unwrap().unwrap();
        assert!(!config.keyboard && config.mouse);
    }

    #[test]
    fn no_listed_port_attaches_nothing() {
        assert!(matches!(i8042(false, false, &[]), Ok(None)));
    }

    #[test]
    fn other_ports_are_refused() {
        for io in [
            &[0x60..0x61, 0x60..0x61][..],
            &[0x64..0x65, 0x60..0x61],
            &[0x60..0x62, 0x64..0x65],
        ] {
            assert!(matches!(
                i8042(true, true, io),
                Err(PlatformError::Refused(_))
            ));
        }
    }

    #[test]
    fn the_firmware_id_lists_the_pnp_ids_hid_first_in_pnp_form() {
        let ids = |ids: &[&str]| {
            ids.iter()
                .map(|id| String::from(*id))
                .collect::<alloc::vec::Vec<_>>()
        };
        assert_eq!(
            firmware_id(&ids(&["LEN0073", "PNP0F13"])).as_deref(),
            Some("PNP: LEN0073 PNP0f13")
        );
        assert_eq!(
            firmware_id(&ids(&["ACPI0003", "SYN3052", "PNP0F13", "SYN3052"])).as_deref(),
            Some("PNP: SYN3052 PNP0f13"),
            "an ID not in PnP form is skipped, and the first PnP ID listed once"
        );
        assert_eq!(firmware_id(&ids(&["MSFT0001"])), None);
    }

    /// A FADT of `revision` with IAPC_BOOT_ARCH `flags`.
    fn fadt(revision: u8, flags: u16) -> alloc::vec::Vec<u8> {
        let mut table = alloc::vec![0; 276];
        table[8] = revision;
        table[FADT_BOOT_ARCH..FADT_BOOT_ARCH + 2].copy_from_slice(&flags.to_le_bytes());
        table
    }

    #[test]
    fn the_fadt_8042_flag_allows_a_direct_probe() {
        let set = fadt(6, FADT_8042 | 1);
        let clear = fadt(6, 1);
        let legacy = fadt(1, 0);
        for (table, observation, permits_direct_probe) in [
            (
                Some(set.as_slice()),
                I8042Fadt::FlagSet { revision: 6 },
                true,
            ),
            (
                Some(clear.as_slice()),
                I8042Fadt::FlagClear { revision: 6 },
                false,
            ),
            (Some(legacy.as_slice()), I8042Fadt::LegacyRevision(1), true),
            (None, I8042Fadt::Absent, true),
            (
                Some(&set[..8]),
                I8042Fadt::Truncated { revision: None },
                true,
            ),
            (
                Some(&set[..FADT_BOOT_ARCH + 1]),
                I8042Fadt::Truncated { revision: Some(6) },
                false,
            ),
        ] {
            assert_eq!(I8042Fadt::observe(table), observation);
            assert_eq!(observation.permits_direct_probe(), permits_direct_probe);
        }
    }

    #[test]
    fn a_keyboard_probed_directly_takes_the_standard_ports() {
        let config = i8042(true, false, &[]).unwrap().unwrap();
        assert!(config.keyboard && !config.mouse);
    }
}
