// SPDX-License-Identifier: GPL-2.0-or-later
//! Read-only Gemini Lake LPSS I²C PCI observations.
//! PCI IDs checked against Linux cc9de5d4b5ab6459f5774946aa3195ad51236fa8,
//! drivers/mfd/intel-lpss-pci.c (GLK I²C entries).

use super::{BarRegion, observe_memory_bar, original_root_host, root_memory_windows};
use crate::ecam::RootWindow;

pub use super::RootBarError;

/// Page extent for root-window correlation. The BAR descriptor has its own length.
pub const BAR0_PROBE_BYTES: u64 = 4096;

pub fn matches(identity: u32) -> bool {
    matches!(
        identity,
        0x31ac_8086
            | 0x31ae_8086
            | 0x31b0_8086
            | 0x31b2_8086
            | 0x31b4_8086
            | 0x31b6_8086
            | 0x31b8_8086
            | 0x31ba_8086
    )
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Bar0 {
    pub pci_base: u64,
    pub bits: u64,
}

impl Bar0 {
    /// Observe the assigned base and width from the BAR configuration words.
    pub fn observe(layout: u32, low: u32, high: u32) -> Option<Self> {
        let (pci_base, bits) = observe_memory_bar(layout, low, high, BAR0_PROBE_BYTES)?;
        Some(Self { pci_base, bits })
    }

    /// Correlate the exact PCI I/O memory descriptor with the raw BAR.
    /// Preserve its reported length and translation as observations.
    pub fn firmware_region(self, regions: &[BarRegion]) -> Option<&BarRegion> {
        let [region] = regions else { return None };
        let bytes = region.pci.end.checked_sub(region.pci.start)?;
        let in_width = match self.bits {
            32 => region.pci.end <= 1u64 << 32,
            64 => true,
            _ => false,
        };
        (region.resource_type == 0
            && region.granularity == self.bits
            && region.pci.start == self.pci_base
            && bytes != 0
            && region.host.end.checked_sub(region.host.start) == Some(bytes)
            && region.host.start.checked_add(region.translation) == Some(region.pci.start)
            && in_width)
            .then_some(region)
    }

    /// Correlate one diagnostic page with the captured root.
    /// Overlapping descriptors leave the host address unresolved.
    pub fn original_root_host(
        self,
        root: &RootWindow,
        windows: &[RootWindow],
    ) -> Result<u64, RootBarError> {
        let host = original_root_host(self.pci_base, BAR0_PROBE_BYTES, root, windows)?;
        host.checked_add(BAR0_PROBE_BYTES - 1)
            .ok_or(RootBarError::InvalidTranslation)?;
        let last = self.pci_base + (BAR0_PROBE_BYTES - 1);
        let mut partial = false;
        for window in root_memory_windows(root, windows).filter(|window| {
            window.minimum <= window.maximum
                && window.minimum <= last
                && self.pci_base <= window.maximum
                && (self.pci_base < window.minimum || window.maximum < last)
        }) {
            if window.translation != host - self.pci_base {
                return Err(RootBarError::Conflicting);
            }
            partial = true;
        }
        if partial {
            return Err(RootBarError::Ambiguous);
        }
        Ok(host)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PCI: u64 = 0x1_8000_1000;
    const HOST: u64 = 0x8000_1000;

    #[test]
    fn matches_only_the_gemini_lake_lpss_i2c_functions() {
        for device in [
            0x31ac, 0x31ae, 0x31b0, 0x31b2, 0x31b4, 0x31b6, 0x31b8, 0x31ba,
        ] {
            assert!(matches(device << 16 | 0x8086));
            assert!(!matches(device << 16 | 0x1234));
        }
        for device in [0x3185, 0x31aa, 0x31ad, 0x31bc] {
            assert!(!matches(device << 16 | 0x8086));
        }
    }

    #[test]
    fn observes_only_an_assigned_page_aligned_type_zero_memory_bar() {
        assert_eq!(
            Bar0::observe(0, PCI as u32 | 4, 1),
            Some(Bar0 {
                pci_base: PCI,
                bits: 64
            })
        );
        assert_eq!(
            Bar0::observe(0, HOST as u32, u32::MAX),
            Some(Bar0 {
                pci_base: HOST,
                bits: 32
            })
        );
        for (layout, low, high) in [
            (1 << 16, HOST as u32, 0),
            (0, HOST as u32 | 1, 0),
            (0, HOST as u32 | 8, 0),
            (0, HOST as u32 | 0x14, 1),
            (0, HOST as u32 | 4, u32::MAX),
            (0, 0, 0),
        ] {
            assert!(Bar0::observe(layout, low, high).is_none());
        }
    }

    fn region(bytes: u64) -> BarRegion {
        BarRegion {
            resource_type: 0,
            general_flags: 0,
            specific_flags: 0,
            granularity: 64,
            observed_maximum: bytes - 1,
            host: HOST..HOST + bytes,
            translation: PCI - HOST,
            pci: PCI..PCI + bytes,
        }
    }

    #[test]
    fn preserves_matching_descriptor_lengths_independent_of_the_diagnostic_page() {
        let bar = Bar0::observe(0, PCI as u32 | 4, 1).unwrap();
        for bytes in [0x800, BAR0_PROBE_BYTES, 3 * BAR0_PROBE_BYTES] {
            let descriptor = region(bytes);
            assert_eq!(
                bar.firmware_region(core::slice::from_ref(&descriptor)),
                Some(&descriptor)
            );
        }
        let descriptor = region(BAR0_PROBE_BYTES);
        assert!(bar.firmware_region(&[]).is_none());
        assert!(
            bar.firmware_region(&[descriptor.clone(), descriptor.clone()])
                .is_none()
        );
        for invalid in [
            BarRegion {
                resource_type: 1,
                ..descriptor.clone()
            },
            BarRegion {
                granularity: 32,
                ..descriptor.clone()
            },
            BarRegion {
                pci: PCI + 1..PCI + BAR0_PROBE_BYTES + 1,
                ..descriptor.clone()
            },
            BarRegion {
                pci: PCI..PCI,
                host: HOST..HOST,
                ..descriptor.clone()
            },
            BarRegion {
                host: HOST..HOST + BAR0_PROBE_BYTES - 1,
                ..descriptor.clone()
            },
            BarRegion {
                translation: u64::MAX,
                ..descriptor
            },
        ] {
            assert!(bar.firmware_region(&[invalid]).is_none());
        }
        let bar = Bar0::observe(0, 0xffff_f000, 0).unwrap();
        let invalid = BarRegion {
            granularity: 32,
            host: 0xffff_f000..0x1_0000_1000,
            pci: 0xffff_f000..0x1_0000_1000,
            translation: 0,
            ..region(2 * BAR0_PROBE_BYTES)
        };
        assert!(bar.firmware_region(&[invalid]).is_none());
    }

    #[test]
    fn only_one_complete_page_of_the_captured_root_can_supply_a_host_address() {
        let bar = Bar0::observe(0, PCI as u32 | 4, 1).unwrap();
        let root = RootWindow {
            segment: 7,
            bridge: 2,
            space: 2,
            ..Default::default()
        };
        let window = RootWindow {
            space: 0,
            minimum: PCI,
            maximum: PCI + BAR0_PROBE_BYTES - 1,
            ..root
        };
        let distractors = [
            RootWindow {
                bridge: 1,
                ..window
            },
            RootWindow {
                segment: 6,
                ..window
            },
            RootWindow { space: 1, ..window },
            RootWindow {
                maximum: window.maximum - 1,
                ..window
            },
        ];
        assert_eq!(
            bar.original_root_host(&root, &distractors),
            Err(RootBarError::NoWindow)
        );
        let [wrong_root, wrong_segment, io, partial] = distractors;
        assert_eq!(
            bar.original_root_host(&root, &[wrong_root, wrong_segment, io, window]),
            Ok(PCI)
        );
        assert_eq!(
            bar.original_root_host(&root, &[window, partial]),
            Err(RootBarError::Ambiguous)
        );
        let partial = RootWindow {
            translation: BAR0_PROBE_BYTES,
            ..partial
        };
        assert_eq!(
            bar.original_root_host(&root, &[partial, window]),
            Err(RootBarError::Conflicting)
        );
        assert_eq!(
            bar.original_root_host(&root, &[window, window]),
            Err(RootBarError::Ambiguous)
        );
        let translated = RootWindow {
            translation: BAR0_PROBE_BYTES,
            ..window
        };
        assert_eq!(
            bar.original_root_host(&root, &[translated]),
            Ok(PCI + BAR0_PROBE_BYTES)
        );
        assert_eq!(
            bar.original_root_host(&root, &[window, translated]),
            Err(RootBarError::Conflicting)
        );
        let overflowing = RootWindow {
            translation: u64::MAX - PCI,
            ..window
        };
        assert_eq!(
            bar.original_root_host(&root, &[overflowing]),
            Err(RootBarError::InvalidTranslation)
        );

        let top = Bar0 {
            pci_base: u64::MAX - (BAR0_PROBE_BYTES - 1),
            bits: 64,
        };
        let last_page = RootWindow {
            minimum: top.pci_base,
            maximum: u64::MAX,
            ..window
        };
        assert_eq!(
            top.original_root_host(&root, &[last_page]),
            Ok(top.pci_base)
        );
    }
}
