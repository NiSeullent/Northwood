// SPDX-License-Identifier: GPL-2.0-or-later
//! Explicit native consumer initialization, separate from raw USB attachment.
//! The kept result owns metadata only and never extends a provider borrow.
use super::{Error, Hci, Io, ath3k, broadcom, intel, mediatek, qca, realtek};
use crate::firmware::Provider;
use alloc::boxed::Box;
use core::fmt;

/// Device-specific observed initialization evidence, never universal radio or
/// guest readiness. Inspect transport lifecycle separately before using HCI.
#[derive(Debug)]
pub enum InitializationReport {
    Atheros(ath3k::runtime::OwnedReport),
    Qualcomm(qca::OwnedReport),
    Broadcom(broadcom::OwnedReport),
    BroadcomApple(broadcom::AppleReport),
    Intel(intel::OwnedReport),
    Realtek(realtek::OwnedReport),
    MediaTek(mediatek::OwnedReport),
}
/// Typed failure. Each initializer preserves exact resource/target errors,
/// HCI controller status, transport/cleanup errors and partial receipts.
#[derive(Debug)]
pub enum InitializationError {
    Atheros(ath3k::runtime::Error),
    UnsupportedDevice {
        vendor: u16,
        product: u16,
        revision: u16,
    },
    Qualcomm(qca::Error),
    Broadcom(broadcom::Error),
    Intel(intel::Error),
    Realtek(realtek::Error),
    MediaTek(mediatek::Error),
    Transport(Error),
}
impl fmt::Display for InitializationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::UnsupportedDevice {
                vendor,
                product,
                revision,
            } => {
                write!(
                    f,
                    "unsupported USB Bluetooth controller {vendor:04x}:{product:04x} \
                     revision {revision:04x}"
                )?;
                if let Some(name) = vendor_name(vendor) {
                    write!(f, " ({name})")?;
                }
                f.write_str(": no vendor setup here takes it, so it stays off")
            }
            _ => write!(f, "native USB Bluetooth initialization {self:?}"),
        }
    }
}

/// The USB vendors whose IDs Bluetooth controllers commonly carry: chip
/// makers and the module makers whose IDs Linux btusb lists.
fn vendor_name(vendor: u16) -> Option<&'static str> {
    Some(match vendor {
        0x03f0 => "HP",
        0x043e => "LG",
        0x0489 => "Foxconn",
        0x04ca => "Lite-On",
        0x04f2 => "Chicony",
        0x050d => "Belkin",
        0x05ac => "Apple",
        0x0930 => "Toshiba",
        0x0a12 => "Cambridge Silicon Radio",
        0x0a5c => "Broadcom",
        0x0b05 => "ASUS",
        0x0bda => "Realtek",
        0x0cf3 => "Qualcomm Atheros",
        0x0e8d => "MediaTek",
        0x105b => "Foxconn",
        0x13d3 => "IMC Networks",
        0x1286 => "Marvell",
        0x2357 => "TP-Link",
        0x33fa => "Actions",
        0x413c => "Dell",
        0x8087 => "Intel",
        _ => return None,
    })
}
impl core::error::Error for InitializationError {}

/// The vendor setup a USB Bluetooth controller gets here, chosen by its IDs
/// as initialization chooses it. Its live registers still choose the
/// firmware, or refuse the chip.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Setup {
    Atheros,
    Qualcomm,
    Broadcom,
    BroadcomApple,
    Realtek,
    MediaTek,
    Intel,
}
impl Setup {
    /// The setup for a controller with these IDs, whose HCI interface is of
    /// vendor class FF/01/01 (`vendor_interface`) rather than E0/01/01;
    /// `None` for one no setup here takes.
    pub fn of(vendor: u16, product: u16, vendor_interface: bool) -> Option<Self> {
        if ath3k::SOURCE_USB_IDS
            .iter()
            .any(|&(v, p, _)| (v, p) == (vendor, product))
        {
            return Some(Self::Atheros);
        }
        if qca::SOURCE_USB_IDS
            .iter()
            .any(|&(v, p, _)| (v, p) == (vendor, product))
        {
            return Some(Self::Qualcomm);
        }
        if broadcom::listed(vendor, product, vendor_interface) {
            return Some(Self::Broadcom);
        }
        if vendor_interface && vendor == broadcom::APPLE_VENDOR {
            return Some(Self::BroadcomApple);
        }
        // The others take only the standard HCI interface.
        if vendor_interface {
            return None;
        }
        if (realtek::UsbIdentity {
            vendor,
            product,
            revision: 0,
        })
        .project()
        .is_some()
        {
            Some(Self::Realtek)
        } else if (mediatek::UsbIdentity {
            vendor,
            product,
            revision: 0,
        })
        .candidate()
        {
            Some(Self::MediaTek)
        } else if crate::bluetooth::intel::Identity::listed(vendor, product) {
            Some(Self::Intel)
        } else {
            None
        }
    }
    /// The chip maker's name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Atheros => "Qualcomm Atheros AR3011/AR3012",
            Self::Qualcomm => "Qualcomm",
            Self::Broadcom => "Broadcom",
            Self::BroadcomApple => "Apple Broadcom",
            Self::Realtek => "Realtek",
            Self::MediaTek => "MediaTek",
            Self::Intel => "Intel",
        }
    }
}

impl<B: Io> Hci<B> {
    /// Explicitly initialize an attached native controller against exact supplied
    /// resources. A provider/catalog is borrowed only for this call; the stored
    /// result owns identities and receipt metadata, never firmware bytes.
    ///
    /// Exactly one attempt is held per attachment. Repeated calls return
    /// that historical result without issuing commands or recording new resource
    /// requests, even with another provider/platform input. Failed candidates
    /// require a fresh generic rebind to retry. Unknown devices remain raw and
    /// unverified and return UnsupportedDevice without initializer commands.
    /// A kept success does not supersede subsequent transport lifecycle or
    /// establish image authenticity for RunningUnknown, radio or guest services.
    pub fn initialize_with_provider(
        &mut self,
        provider: &dyn Provider,
        platform: intel::PlatformInputs,
    ) -> Result<&InitializationReport, &InitializationError> {
        if self.initialization.is_none() {
            let result = self.initialize_once(provider, platform);
            self.initialization = Some(Box::new(result));
        }
        self.initialization().unwrap()
    }
    /// Historical result of the explicit native initialization attempt. None
    /// means no attempt, including every newly attached raw transport.
    pub fn initialization(&self) -> Option<Result<&InitializationReport, &InitializationError>> {
        self.initialization.as_deref().map(Result::as_ref)
    }
    #[expect(
        clippy::result_large_err,
        reason = "The shared error preserves exact source and cleanup causes without allocating on propagation."
    )]
    fn initialize_once(
        &mut self,
        provider: &dyn Provider,
        platform: intel::PlatformInputs,
    ) -> Result<InitializationReport, InitializationError> {
        self.open().map_err(InitializationError::Transport)?;
        if ath3k::is_runtime_candidate(self.binding.device()) {
            return ath3k::runtime::initialize(self, provider)
                .map(InitializationReport::Atheros)
                .map_err(InitializationError::Atheros);
        }
        if qca::is_candidate(self.binding.device()) {
            return qca::initialize(self, provider)
                .map(InitializationReport::Qualcomm)
                .map_err(InitializationError::Qualcomm);
        }
        if broadcom::is_candidate(self.binding.device()) {
            return broadcom::initialize(self, provider)
                .map(InitializationReport::Broadcom)
                .map_err(InitializationError::Broadcom);
        }
        if self.binding.apple_rom {
            return broadcom::initialize_apple(self)
                .map(InitializationReport::BroadcomApple)
                .map_err(InitializationError::Broadcom);
        }
        if realtek::is_candidate(self.binding.device()) {
            return realtek::initialize(self, provider)
                .map(InitializationReport::Realtek)
                .map_err(InitializationError::Realtek);
        }
        if mediatek::is_candidate(self.binding.device()) {
            return mediatek::initialize(self, provider)
                .map(InitializationReport::MediaTek)
                .map_err(InitializationError::MediaTek);
        }
        if !intel::is_candidate(self.binding.device()) {
            let descriptor = &self.binding.device().descriptor;
            return Err(InitializationError::UnsupportedDevice {
                vendor: u16::from_le_bytes([descriptor[8], descriptor[9]]),
                product: u16::from_le_bytes([descriptor[10], descriptor[11]]),
                revision: u16::from_le_bytes([descriptor[12], descriptor[13]]),
            });
        }
        intel::initialize_with_platform(self, provider, platform)
            .map(|report| InitializationReport::Intel(report.into()))
            .map_err(InitializationError::Intel)
    }
}

#[cfg(test)]
mod tests {
    use super::{InitializationError, Setup};
    use alloc::string::ToString;

    #[test]
    fn controller_ids_choose_the_setup_initialization_runs() {
        for (vendor, product, vendor_interface, setup) in [
            (0x13d3, 0x3602, false, Some(Setup::MediaTek)),
            (0x0e8d, 0x7922, false, Some(Setup::MediaTek)),
            (0x0bda, 0x8771, false, Some(Setup::Realtek)),
            (0x8087, 0x0029, false, Some(Setup::Intel)),
            (0x0cf3, 0xe300, false, Some(Setup::Qualcomm)),
            (0x0cf3, 0x3000, false, Some(Setup::Atheros)),
            (0x0a5c, 0x21e8, true, Some(Setup::Broadcom)),
            (0x19ff, 0x0239, false, Some(Setup::Broadcom)),
            (0x05ac, 0x8290, true, Some(Setup::BroadcomApple)),
            (0x05ac, 0x8290, false, None),
            // A Broadcom module maker's ID on the standard interface, and a
            // MediaTek ID on a vendor-class one.
            (0x0a5c, 0x21e8, false, None),
            (0x0e8d, 0x7922, true, None),
            // Intel's legacy-ROM controllers and generic ones.
            (0x8087, 0x0a2a, false, None),
            (0x0a12, 0x0001, false, None),
        ] {
            assert_eq!(
                Setup::of(vendor, product, vendor_interface),
                setup,
                "{vendor:04x}:{product:04x}"
            );
        }
    }

    #[test]
    fn an_unsupported_controller_is_named_by_its_ids_and_vendor() {
        let error = InitializationError::UnsupportedDevice {
            vendor: 0x0a12,
            product: 0x0001,
            revision: 0x8891,
        };
        assert_eq!(
            error.to_string(),
            "unsupported USB Bluetooth controller 0a12:0001 revision 8891 \
             (Cambridge Silicon Radio): no vendor setup here takes it, so it stays off"
        );
        let unknown = InitializationError::UnsupportedDevice {
            vendor: 0x1234,
            product: 0x5678,
            revision: 0x0100,
        };
        assert_eq!(
            unknown.to_string(),
            "unsupported USB Bluetooth controller 1234:5678 revision 0100: no vendor setup \
             here takes it, so it stays off"
        );
    }
}
