//! How the UEFI memory map's entry types and attributes bound native device
//! ranges and the identity map of the firmware's runtime regions.
//!
//! EFI_MEMORY_RUNTIME says the OS must keep a region mapped for the runtime
//! services; it does not take the range from the OS. EDK2-derived firmware
//! lists MMIO in its map only with that attribute (DxeCore CoreGetMemoryMap),
//! so the MCFG window, IOAPICs and HPET of real PCs often carry it. Linux
//! uses an MCFG window that its map reserves whatever the attribute.

/// EfiReservedMemoryType.
pub const RESERVED: u32 = 0;
/// EfiRuntimeServicesCode.
pub const RUNTIME_SERVICES_CODE: u32 = 5;
/// EfiMemoryMappedIO.
pub const MMIO: u32 = 11;
/// EfiMemoryMappedIOPortSpace.
pub const MMIO_PORT_SPACE: u32 = 12;
/// EfiPalCode.
pub const PAL_CODE: u32 = 13;

/// EFI_MEMORY_WB: the region supports write-back caching.
pub const MEMORY_WB: u64 = 1 << 3;
/// EFI_MEMORY_RUNTIME.
pub const MEMORY_RUNTIME: u64 = 1 << 63;

/// Whether an entry of `kind` leaves its range to devices: reserved or
/// MMIO, never memory the firmware or the loader can hand out. The
/// attributes, EFI_MEMORY_RUNTIME included, do not change this.
pub const fn leaves_to_devices(kind: u32) -> bool {
    matches!(kind, RESERVED | MMIO)
}

/// The first entry of `entries` (start, end, kind) that overlaps
/// `first..end` and does not leave it to devices. Ranges no entry covers
/// are outside the map's memory and do not conflict.
pub fn first_conflict(
    entries: impl IntoIterator<Item = (u64, u64, u32)>,
    first: u64,
    end: u64,
) -> Option<(u64, u64, u32)> {
    entries
        .into_iter()
        .find(|&(start, stop, kind)| start < end && first < stop && !leaves_to_devices(kind))
}

/// Whether every entry of `entries` that overlaps `first..end` leaves the
/// range to devices.
pub fn excludes_memory(
    entries: impl IntoIterator<Item = (u64, u64, u32)>,
    first: u64,
    end: u64,
) -> bool {
    first_conflict(entries, first, end).is_none()
}

/// The UEFI name of a memory type, for the boot log.
pub const fn kind_name(kind: u32) -> &'static str {
    match kind {
        0 => "reserved",
        1 => "loader code",
        2 => "loader data",
        3 => "boot services code",
        4 => "boot services data",
        5 => "runtime services code",
        6 => "runtime services data",
        7 => "conventional memory",
        8 => "unusable memory",
        9 => "ACPI reclaim memory",
        10 => "ACPI NVS",
        11 => "MMIO",
        12 => "MMIO port space",
        13 => "PAL code",
        14 => "persistent memory",
        15 => "unaccepted memory",
        _ => "an unknown type",
    }
}

/// The identity mapping of one runtime region.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RuntimeMapping {
    /// Runtime code: write-back and executable.
    Code,
    /// Write-back data, not executable.
    WriteBack,
    /// Uncached, not executable.
    Uncached,
}

/// Map MMIO uncached and code executable; map other runtime regions
/// write-back only when the firmware says they support it, as Linux's
/// efi_map_region does. A reserved runtime region without EFI_MEMORY_WB is
/// often device registers such as an ECAM window, and a write-back alias of
/// those would conflict with their uncached native mapping.
pub const fn runtime_mapping(kind: u32, attributes: u64) -> RuntimeMapping {
    match kind {
        MMIO | MMIO_PORT_SPACE => RuntimeMapping::Uncached,
        RUNTIME_SERVICES_CODE | PAL_CODE => RuntimeMapping::Code,
        _ if attributes & MEMORY_WB != 0 => RuntimeMapping::WriteBack,
        _ => RuntimeMapping::Uncached,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ECAM: (u64, u64) = (0xe000_0000, 0xf000_0000);

    #[test]
    fn runtime_reserved_or_mmio_entries_leave_an_ecam_window_to_devices() {
        for kind in [RESERVED, MMIO] {
            assert!(excludes_memory([(ECAM.0, ECAM.1, kind)], ECAM.0, ECAM.1));
            assert!(excludes_memory(
                [(ECAM.0 - 0x1000, ECAM.1 + 0x1000, kind)],
                ECAM.0,
                ECAM.0 + 0x10_0000
            ));
        }
        assert!(excludes_memory([], ECAM.0, ECAM.1));
    }

    #[test]
    fn ram_loader_or_firmware_memory_over_an_ecam_window_refuses() {
        // Loader code/data, boot services, conventional memory, ACPI
        // tables and NVS, runtime services code and data, persistent memory.
        for kind in [1, 2, 3, 4, 5, 6, 7, 9, 10, 13, 14] {
            assert!(!excludes_memory(
                [
                    (ECAM.0, ECAM.0 + 0x1000, MMIO),
                    (ECAM.0 + 0x1000, ECAM.1, kind)
                ],
                ECAM.0,
                ECAM.1
            ));
        }
        assert_eq!(
            first_conflict(
                [
                    (ECAM.0, ECAM.0 + 0x1000, MMIO),
                    (ECAM.0 + 0x1000, ECAM.1, 7)
                ],
                ECAM.0,
                ECAM.1
            ),
            Some((ECAM.0 + 0x1000, ECAM.1, 7))
        );
        assert_eq!(kind_name(7), "conventional memory");
        // Entries that end at or start after the range do not count.
        assert!(excludes_memory(
            [(0, ECAM.0, 7), (ECAM.1, ECAM.1 + 0x1000, 7)],
            ECAM.0,
            ECAM.1
        ));
    }

    #[test]
    fn runtime_regions_map_write_back_only_when_the_firmware_allows_it() {
        let uc = 1;
        assert_eq!(
            runtime_mapping(MMIO, MEMORY_WB | uc),
            RuntimeMapping::Uncached
        );
        assert_eq!(
            runtime_mapping(MMIO_PORT_SPACE, 0),
            RuntimeMapping::Uncached
        );
        assert_eq!(
            runtime_mapping(RUNTIME_SERVICES_CODE, MEMORY_WB),
            RuntimeMapping::Code
        );
        assert_eq!(runtime_mapping(PAL_CODE, MEMORY_WB), RuntimeMapping::Code);
        assert_eq!(
            runtime_mapping(6, MEMORY_WB | uc),
            RuntimeMapping::WriteBack
        );
        assert_eq!(
            runtime_mapping(RESERVED, MEMORY_RUNTIME | uc),
            RuntimeMapping::Uncached
        );
        assert_eq!(
            runtime_mapping(RESERVED, MEMORY_RUNTIME | MEMORY_WB),
            RuntimeMapping::WriteBack
        );
    }
}
