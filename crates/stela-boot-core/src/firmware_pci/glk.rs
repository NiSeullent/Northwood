// SPDX-License-Identifier: GPL-2.0-or-later
//! Read-only observations of Gemini Lake's GTTMMADR BAR.

use super::{BarRegion, observe_memory_bar, original_root_host};
use crate::ecam::RootWindow;

pub use super::RootBarError;

/// The expected Gen8+ GTTMMADR BAR length in i915 intel_ggtt.c,
/// cc9de5d4b5ab6459f5774946aa3195ad51236fa8. A config read supplies no size.
pub const BAR0_BYTES: u64 = 16 << 20;

pub fn matches(identity: u32, class_revision: u32) -> bool {
    matches!(identity, 0x3184_8086 | 0x3185_8086) && class_revision >> 24 == 3
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Bar0 {
    pub pci_base: u64,
    pub bits: u64,
}

impl Bar0 {
    /// Observe an assigned Type 0 non-prefetchable memory BAR from raw
    /// config words. No sizing write or BAR access is performed.
    pub fn observe(layout: u32, low: u32, high: u32) -> Option<Self> {
        let (pci_base, bits) = observe_memory_bar(layout, low, high, BAR0_BYTES)?;
        Some(Self { pci_base, bits })
    }

    /// Correlate a UEFI PCI_IO descriptor with the raw BAR, retaining the
    /// firmware's original address translation as observation data.
    pub fn firmware_region(self, regions: &[BarRegion]) -> Option<&BarRegion> {
        let [region] = regions else { return None };
        (region.resource_type == 0
            && region.granularity == self.bits
            && region.pci.start == self.pci_base
            && region.pci.end.checked_sub(region.pci.start) == Some(BAR0_BYTES)
            && region.host.end.checked_sub(region.host.start) == Some(BAR0_BYTES))
        .then_some(region)
    }

    /// Translate the entire BAR through exactly one memory descriptor of the
    /// root already identified by the function's captured PCI topology.
    /// A duplicate descriptor or conflicting translation leaves the host
    /// address unresolved. This is an observation, with no mapping authority.
    pub fn original_root_host(
        self,
        root: &RootWindow,
        windows: &[RootWindow],
    ) -> Result<u64, RootBarError> {
        original_root_host(self.pci_base, BAR0_BYTES, root, windows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PCI: u64 = 0x1_c000_0000;
    const HOST: u64 = 0xc000_0000;

    #[test]
    fn accepts_only_gemini_lake_display_functions() {
        assert!(matches(0x3185_8086, 0x0300_0001));
        assert!(matches(0x3184_8086, 0x0380_0001));
        assert!(!matches(0x3185_8086, 0x0200_0001));
        assert!(!matches(0x5917_8086, 0x0300_0001));
    }

    #[test]
    fn decodes_raw_bar_without_interpreting_a_32_bit_bars_neighbor() {
        assert_eq!(Bar0::observe(0, PCI as u32 | 4, 1).unwrap().pci_base, PCI);
        assert_eq!(
            Bar0::observe(0, PCI as u32, u32::MAX).unwrap().pci_base,
            PCI & u64::from(u32::MAX)
        );
        for (layout, low, high) in [
            (1 << 16, PCI as u32 | 4, 1),
            (0, PCI as u32 | 1, 1),
            (0, PCI as u32 | 12, 1),
            (0, PCI as u32 | 4, u32::MAX),
            (0, PCI as u32 | 0x1004, 1),
            (0, 4, 0),
        ] {
            assert_eq!(Bar0::observe(layout, low, high), None);
        }
    }

    #[test]
    fn preserves_only_the_matching_firmware_bar_extent_and_translation() {
        let bar = Bar0::observe(0, PCI as u32 | 4, 1).unwrap();
        let descriptor = BarRegion {
            resource_type: 0,
            general_flags: 0,
            specific_flags: 0,
            granularity: 64,
            observed_maximum: BAR0_BYTES - 1,
            host: HOST..HOST + BAR0_BYTES,
            translation: PCI - HOST,
            pci: PCI..PCI + BAR0_BYTES,
        };
        assert_eq!(
            bar.firmware_region(core::slice::from_ref(&descriptor))
                .unwrap()
                .host
                .start,
            HOST
        );
        assert!(
            bar.firmware_region(&[descriptor.clone(), descriptor.clone()])
                .is_none()
        );
        let mut wrong = descriptor;
        wrong.pci.end -= 4096;
        assert!(bar.firmware_region(&[wrong]).is_none());
    }

    #[test]
    fn only_the_captured_root_and_complete_bar_span_count() {
        let bar = Bar0::observe(0, PCI as u32 | 4, 1).unwrap();
        let root = RootWindow {
            bridge: 2,
            segment: 7,
            space: 2,
            ..Default::default()
        };
        let window = RootWindow {
            space: 0,
            minimum: PCI,
            maximum: PCI + BAR0_BYTES - 1,
            translation: 1 << 32,
            ..root
        };
        let distractors = [
            RootWindow {
                bridge: 1,
                translation: 0,
                ..window
            },
            RootWindow {
                segment: 3,
                translation: 0,
                ..window
            },
            RootWindow {
                maximum: window.maximum - 1,
                ..window
            },
            RootWindow { space: 1, ..window },
        ];
        assert_eq!(
            bar.original_root_host(&root, &distractors),
            Err(RootBarError::NoWindow)
        );
        let [other_root, other_segment, partial, io] = distractors;
        assert_eq!(
            bar.original_root_host(&root, &[other_root, other_segment, partial, window, io]),
            Ok(PCI + (1 << 32))
        );
    }

    #[test]
    fn duplicate_or_conflicting_root_windows_never_choose_a_host_address() {
        let bar = Bar0::observe(0, PCI as u32 | 4, 1).unwrap();
        let root = RootWindow {
            bridge: 1,
            space: 2,
            ..Default::default()
        };
        let window = RootWindow {
            space: 0,
            minimum: PCI,
            maximum: PCI + BAR0_BYTES - 1,
            ..root
        };
        let translated = RootWindow {
            translation: 1 << 32,
            ..window
        };
        let invalid = RootWindow {
            translation: u64::MAX,
            ..window
        };
        assert_eq!(
            bar.original_root_host(&root, &[window, window]),
            Err(RootBarError::Ambiguous)
        );
        assert_eq!(
            bar.original_root_host(&root, &[window, window, translated]),
            Err(RootBarError::Conflicting)
        );
        assert_eq!(
            bar.original_root_host(&root, &[translated, window]),
            Err(RootBarError::Conflicting)
        );
        assert_eq!(
            bar.original_root_host(&root, &[invalid]),
            Err(RootBarError::InvalidTranslation)
        );
    }
}
