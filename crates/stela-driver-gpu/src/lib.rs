//! Host GPU discovery. An entry is a PCI inventory item, not an initialized
//! accelerator or a Metal device. The native vendor ports live under this
//! module.
#![no_std]
extern crate alloc;

use stela_driver_base::bus;

use crate::bus::pci::{Address, ConfigSpace, Function};

#[cfg(feature = "metal")]
pub mod metal;

#[cfg(feature = "nouveau-port")]
pub mod nouveau;

pub mod cold;
pub mod intel;
pub mod nouveau_chip;

#[cfg(feature = "linux-kernel")]
pub mod drm;

/// The native ports' hosted C test programs, which `build.rs` compiles.
#[cfg(all(test, feature = "linux-kernel"))]
mod c_tests {
    extern crate std;

    // Clippy's build compiles no C, so it lists no programs.
    #[cfg_attr(clippy, allow(dead_code))]
    fn run(program: &str) {
        let output = std::process::Command::new(program)
            .output()
            .unwrap_or_else(|error| panic!("cannot run {program}: {error}"));
        assert!(
            output.status.success(),
            "{program} failed ({}):\n{}{}",
            output.status,
            std::string::String::from_utf8_lossy(&output.stdout),
            std::string::String::from_utf8_lossy(&output.stderr)
        );
    }

    include!(concat!(env!("OUT_DIR"), "/c_tests.rs"));
}

/// PCI base class for VGA, 3D and other display controllers.
const DISPLAY_CLASS: u8 = 0x03;
const NVIDIA_VENDOR: u16 = 0x10de;
const AMD_VENDOR: u16 = 0x1002;
const INTEL_VENDOR: u16 = 0x8086;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Vendor {
    Nvidia,
    Amd,
    Intel,
    Other(u16),
}

impl Vendor {
    pub fn from_pci(id: u16) -> Self {
        match id {
            NVIDIA_VENDOR => Self::Nvidia,
            AMD_VENDOR => Self::Amd,
            INTEL_VENDOR => Self::Intel,
            other => Self::Other(other),
        }
    }
}

/// An unbound host display controller. Vendor identity does not establish
/// architecture support, discrete/integrated placement, or Metal features.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Adapter {
    pub address: Address,
    pub vendor: Vendor,
    pub device: u16,
    pub class: [u8; 3],
    /// Read-only Gemini Lake graphics memory inventory, if requested.
    pub glk_memory: Option<Result<intel::glk::MemoryLayout, intel::glk::Error>>,
}

impl Adapter {
    /// PCI family identification for GPUs relevant to Intel Mac bring-up.
    /// IDs follow Linux's amdgpu, radeon and drm/intel PCI tables. Shared
    /// device IDs identify a chip family without identifying the OEM board.
    pub fn family(&self) -> Option<&'static str> {
        Some(match (self.vendor, self.device) {
            (Vendor::Amd, 0x6741 | 0x6760) => "TeraScale 2",
            (Vendor::Amd, 0x6821) => "GCN Southern Islands",
            (Vendor::Amd, 0x67df | 0x67eb | 0x67ef) => "GCN Polaris",
            (Vendor::Amd, 0x7340 | 0x7341 | 0x7360) => "RDNA 1",
            (Vendor::Intel, 0x0a2e | 0x0d26) => "Intel Haswell",
            (Vendor::Intel, 0x1622 | 0x1626 | 0x162b) => "Intel Broadwell",
            (Vendor::Intel, 0x191b | 0x1926) => "Intel Skylake",
            (Vendor::Intel, 0x5926) => "Intel Kaby Lake",
            (Vendor::Intel, 0x3e92 | 0x3e9b | 0x3ea5) => "Intel Coffee Lake",
            (Vendor::Intel, 0x8a52 | 0x8a53) => "Intel Ice Lake",
            _ => return None,
        })
    }

    /// Inventory only: does not size BARs, enable bus mastering, reset the
    /// device, change the firmware's scanout or allocate GPU memory.
    pub fn discover<C: ConfigSpace>(function: &Function<C>) -> Option<Self> {
        (function.class[0] == DISPLAY_CLASS).then_some(Self {
            address: function.address,
            vendor: Vendor::from_pci(function.vendor),
            device: function.device,
            class: function.class,
            glk_memory: None,
        })
    }

    /// Inventory with an optional read-only Gemini Lake PCI memory observation.
    /// Even when the observation fails, keep the adapter in the inventory.
    pub fn inspect<C: ConfigSpace>(function: &mut Function<C>) -> Option<Self> {
        let mut adapter = Self::discover(function)?;
        adapter.glk_memory = intel::glk::inspect_memory(function);
        Some(adapter)
    }

    /// Every discovered adapter remains unbound until a real port exists.
    pub fn unavailable_reason(&self) -> &'static str {
        match self.vendor {
            Vendor::Nvidia => "NVIDIA requires nouveau/GSP binding and an NVK-enabled image",
            Vendor::Amd => "AMDGPU is not ported to Stela",
            Vendor::Intel => "Intel i915/xe are not ported to Stela",
            Vendor::Other(_) => "no Stela GPU backend for this PCI vendor",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct ReadOnlyConfig {
        vendor: u16,
        class: [u8; 3],
    }

    impl ConfigSpace for ReadOnlyConfig {
        fn read32(&self, offset: u16) -> Result<u32, crate::bus::pci::ConfigError> {
            Ok(match offset {
                0 => 0x1234_0000 | u32::from(self.vendor),
                8 => u32::from_be_bytes([self.class[0], self.class[1], self.class[2], 0]),
                _ => panic!("GPU inventory must not read device registers"),
            })
        }

        fn write32(&mut self, _: u16, _: u32) -> Result<(), crate::bus::pci::ConfigError> {
            panic!("GPU inventory must not change the controller");
        }

        fn write16(&mut self, _: u16, _: u16) -> Result<(), crate::bus::pci::ConfigError> {
            panic!("GPU inventory must not change the controller");
        }

        fn write8(&mut self, _: u16, _: u8) -> Result<(), crate::bus::pci::ConfigError> {
            panic!("GPU inventory must not change the controller");
        }

        fn read8(&self, offset: u16) -> Result<u8, crate::bus::pci::ConfigError> {
            Ok((self.read32(offset & !3)? >> (8 * (offset & 3))) as u8)
        }

        fn read16(&self, offset: u16) -> Result<u16, crate::bus::pci::ConfigError> {
            Ok((self.read32(offset & !3)? >> (8 * (offset & 2))) as u16)
        }
    }

    fn function(vendor: u16, class: [u8; 3]) -> Function<ReadOnlyConfig> {
        Function::new(
            Address {
                segment: 0,
                bus: 1,
                device: 0,
                function: 0,
            },
            ReadOnlyConfig { vendor, class },
        )
        .unwrap()
    }

    #[test]
    fn display_discovery_does_not_touch_hardware_or_claim_support() {
        for vendor in [NVIDIA_VENDOR, AMD_VENDOR, INTEL_VENDOR, 0x1234] {
            for subclass in [0, 2, 0x80] {
                let function = function(vendor, [DISPLAY_CLASS, subclass, 0]);
                let adapter = Adapter::discover(&function).unwrap();
                assert_eq!(adapter.address, function.address);
                assert_eq!(adapter.device, function.device);
                assert_eq!(adapter.class, function.class);
                assert!(!adapter.unavailable_reason().is_empty());
            }
            assert!(Adapter::discover(&function(vendor, [4, 3, 0])).is_none());
        }
    }

    #[test]
    fn pci_family_identification_requires_matching_vendor_and_device() {
        let mut adapter = Adapter::discover(&function(AMD_VENDOR, [DISPLAY_CLASS, 0, 0])).unwrap();
        for (device, family) in [
            (0x6741, "TeraScale 2"),
            (0x6821, "GCN Southern Islands"),
            (0x67ef, "GCN Polaris"),
            (0x7340, "RDNA 1"),
            (0x7360, "RDNA 1"),
        ] {
            adapter.device = device;
            assert_eq!(adapter.family(), Some(family));
        }
        assert_eq!(
            adapter.unavailable_reason(),
            "AMDGPU is not ported to Stela"
        );
        adapter.vendor = Vendor::Intel;
        assert_eq!(adapter.family(), None);
        adapter.device = 0x3e9b;
        assert_eq!(adapter.family(), Some("Intel Coffee Lake"));
        adapter.device = 0xffff;
        assert_eq!(adapter.family(), None);
    }
}
