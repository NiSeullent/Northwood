// SPDX-License-Identifier: GPL-2.0-or-later

use super::{I2cAddress, PlatformError};

/// An exact match on the device's own `_HID`, its full ACPI namespace path,
/// or both. The caller selects the rules for the current host.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HidMatch<'a> {
    Hid(&'a str),
    Path(&'a str),
    Both { hid: &'a str, path: &'a str },
}

impl HidMatch<'_> {
    fn matches(self, hid: Option<&str>, path: Option<&str>) -> bool {
        match self {
            Self::Hid(expected) => hid == Some(expected),
            Self::Path(expected) => path == Some(expected),
            Self::Both { hid: id, path: at } => hid == Some(id) && path == Some(at),
        }
    }
}

/// The three HID-over-I²C settings a host rule can change.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HidSetting {
    Address,
    DescriptorRegister,
    PollIntervalUs,
}

/// A checked setting for a specific device. Address overrides retain the
/// address mode given by the device's `_CRS` serial resource.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HidQuirk<'a> {
    target: HidMatch<'a>,
    setting: HidSetting,
    value: u16,
}

impl<'a> HidQuirk<'a> {
    /// Construct a rule from a numeric config value. The polling interval
    /// is at least one millisecond and at most 50 milliseconds.
    pub fn new(target: HidMatch<'a>, setting: HidSetting, value: u64) -> Option<Self> {
        let value = u16::try_from(value).ok()?;
        let valid = match setting {
            HidSetting::Address => value <= 0x3ff,
            HidSetting::DescriptorRegister => true,
            HidSetting::PollIntervalUs => (1_000..=50_000).contains(&value),
        };
        valid.then_some(Self {
            target,
            setting,
            value,
        })
    }
}

#[derive(Default)]
pub(super) struct HidOverrides {
    address: Option<u16>,
    descriptor: Option<u16>,
    polling: Option<u16>,
}

impl HidOverrides {
    pub(super) fn resolve(
        quirks: &[HidQuirk<'_>],
        hid: Option<&str>,
        path: Option<&str>,
    ) -> Result<Self, PlatformError> {
        let mut overrides = Self::default();
        for rule in quirks.iter().filter(|rule| rule.target.matches(hid, path)) {
            let slot = match rule.setting {
                HidSetting::Address => &mut overrides.address,
                HidSetting::DescriptorRegister => &mut overrides.descriptor,
                HidSetting::PollIntervalUs => &mut overrides.polling,
            };
            if slot.is_some_and(|previous| previous != rule.value) {
                return Err(PlatformError::Refused("conflicting HID-over-I2C quirks"));
            }
            *slot = Some(rule.value);
        }
        Ok(overrides)
    }

    pub(super) fn address(&self, original: I2cAddress) -> Result<I2cAddress, PlatformError> {
        match (self.address, original) {
            (None, _) => Ok(original),
            (Some(address @ 0x08..=0x77), I2cAddress::SevenBit(_)) => {
                Ok(I2cAddress::SevenBit(address as u8))
            }
            (Some(address), I2cAddress::TenBit(_)) => Ok(I2cAddress::TenBit(address)),
            _ => Err(PlatformError::Refused(
                "HID-over-I2C quirk uses an invalid 7-bit address",
            )),
        }
    }

    pub(super) fn descriptor(
        &self,
        evaluate_dsm: impl FnOnce() -> Result<u16, PlatformError>,
    ) -> Result<u16, PlatformError> {
        self.descriptor.map_or_else(evaluate_dsm, Ok)
    }

    pub(super) fn poll_interval_us(&self) -> Option<u64> {
        self.polling.map(u64::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HID: &str = "SYNA2BA6";
    const PATH: &str = "\\_SB.PCI0.I2C1.TPD0";

    #[test]
    fn only_exact_hid_and_namespace_paths_match() {
        let target = HidMatch::Both {
            hid: HID,
            path: PATH,
        };
        assert!(target.matches(Some(HID), Some(PATH)));
        assert!(!target.matches(Some("SYNA2BA7"), Some(PATH)));
        assert!(!target.matches(Some(HID), Some("\\_SB.PCI0.I2C2.TPD0")));
        assert!(!target.matches(Some(HID), None));
        assert!(HidMatch::Path(PATH).matches(None, Some(PATH)));
        assert!(!HidMatch::Path("TPD0").matches(None, Some(PATH)));
        assert!(HidMatch::Hid(HID).matches(Some(HID), None));
        assert!(!HidMatch::Hid("SYNA").matches(Some(HID), Some(PATH)));
    }

    #[test]
    fn empty_rules_retain_firmware_behavior() {
        let settings = HidOverrides::resolve(&[], Some(HID), Some(PATH)).unwrap();
        assert_eq!(
            settings.address(I2cAddress::SevenBit(0x15)).unwrap(),
            I2cAddress::SevenBit(0x15)
        );
        assert_eq!(settings.descriptor(|| Ok(0x20)).unwrap(), 0x20);
        assert_eq!(settings.poll_interval_us(), None);
    }

    #[test]
    fn matching_rules_override_only_their_own_fields() {
        let target = HidMatch::Both {
            hid: HID,
            path: PATH,
        };
        let quirks = [
            HidQuirk::new(target, HidSetting::Address, 0x2c).unwrap(),
            HidQuirk::new(target, HidSetting::DescriptorRegister, 0x20).unwrap(),
            HidQuirk::new(target, HidSetting::PollIntervalUs, 8_000).unwrap(),
        ];
        let settings = HidOverrides::resolve(&quirks, Some(HID), Some(PATH)).unwrap();
        assert_eq!(
            settings.address(I2cAddress::SevenBit(0x15)).unwrap(),
            I2cAddress::SevenBit(0x2c)
        );
        assert_eq!(
            settings.address(I2cAddress::TenBit(0x215)).unwrap(),
            I2cAddress::TenBit(0x2c)
        );
        assert_eq!(
            settings
                .descriptor(|| panic!("_DSM must not run for an override"))
                .unwrap(),
            0x20
        );
        assert_eq!(settings.poll_interval_us(), Some(8_000));
    }

    #[test]
    fn invalid_or_conflicting_values_are_refused() {
        let target = HidMatch::Hid(HID);
        for (setting, value) in [
            (HidSetting::Address, 0x400),
            (HidSetting::DescriptorRegister, 0x1_0000),
            (HidSetting::PollIntervalUs, 999),
            (HidSetting::PollIntervalUs, 50_001),
        ] {
            assert!(HidQuirk::new(target, setting, value).is_none());
        }
        for address in [0, 7, 0x78, 0x7f, 0x100] {
            let quirk = HidQuirk::new(target, HidSetting::Address, address).unwrap();
            let settings = HidOverrides::resolve(&[quirk], Some(HID), None).unwrap();
            assert!(settings.address(I2cAddress::SevenBit(0x15)).is_err());
        }
        let first = HidQuirk::new(target, HidSetting::PollIntervalUs, 1_000).unwrap();
        let second = HidQuirk::new(target, HidSetting::PollIntervalUs, 50_000).unwrap();
        assert!(HidOverrides::resolve(&[first, first], Some(HID), None).is_ok());
        assert!(HidOverrides::resolve(&[first, second], Some(HID), None).is_err());
    }
}
