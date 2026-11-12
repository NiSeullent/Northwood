//! Whether the partition Install macOS chose for the Mac's SSD
//! (`EFI/stela/disk.cfg`, `SERIAL:GUID`) is still the one to boot.
//!
//! The USB keeps that record when the disk is later replaced, wiped,
//! repartitioned or unplugged, and when the USB moves to another PC. Such a
//! record is stale: the macOS choice is left out, Install macOS chooses a
//! disk again, and the record is replaced only when a new disk is
//! confirmed. Only a store that cannot be read on a partition that is
//! otherwise the chosen one, with a Mac system on it, still stops the boot.

/// What a search for the chosen partition found: every whole disk
/// reporting the serial, then its GPT partitions with the GUID.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Search {
    /// No whole disk reports the serial.
    NoDisk,
    /// Several whole disks report the serial.
    SeveralDisks,
    /// The disk has no GPT, or none that can be read: a blank or wiped disk.
    NoTable,
    /// The GPT has no partition with the GUID.
    NoPartition,
    /// Several of the GPT's partitions have the GUID.
    SeveralPartitions,
    /// The partition with the GUID is an EFI system partition now.
    EspPartition,
    /// Exactly one partition, which is not an EFI system partition.
    Found,
}

/// Why the chosen partition can no longer be booted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stale {
    /// No disk on this PC reports the chosen serial: the disk was replaced
    /// or unplugged, or the USB moved to another PC.
    Absent,
    /// Several disks report the serial, so none can be told to be it.
    Ambiguous,
    /// The disk is there but has no partition table: it was wiped.
    Wiped,
    /// The disk is there but the chosen partition is not: it was deleted
    /// or the disk repartitioned.
    PartitionGone,
    /// The partition is there but holds no Mac system and its NVRAM store
    /// cannot be read, as an interrupted install leaves it.
    NoMacOs,
}

impl Stale {
    /// The record for a search that did not find exactly the chosen
    /// partition; `None` when it did.
    pub fn of(search: Search) -> Option<Self> {
        match search {
            Search::NoDisk => Some(Self::Absent),
            Search::SeveralDisks => Some(Self::Ambiguous),
            Search::NoTable => Some(Self::Wiped),
            Search::NoPartition | Search::SeveralPartitions | Search::EspPartition => {
                Some(Self::PartitionGone)
            }
            Search::Found => None,
        }
    }

    /// The case, for the one log line (`CHOSEN DISK stale: ...`).
    pub fn case(self) -> &'static str {
        match self {
            Self::Absent => "no disk on this PC has the chosen serial",
            Self::Ambiguous => "several disks on this PC report the chosen serial",
            Self::Wiped => "the chosen disk has no partition table",
            Self::PartitionGone => "the chosen disk no longer has the chosen partition",
            Self::NoMacOs => "the chosen partition holds no macOS and no readable NVRAM",
        }
    }

    /// Why the macOS choice is left out of the menu.
    pub fn reason(self) -> &'static str {
        match self {
            Self::NoMacOs => "the disk chosen before has no macOS",
            _ => "the disk chosen before is not there",
        }
    }

    /// The two lines the menu shows while it waits for Install macOS.
    pub fn screen(self) -> &'static str {
        match self {
            Self::NoMacOs => {
                "The disk chosen before has no macOS.\nInstall macOS will choose again."
            }
            _ => "The disk chosen before is not there.\nInstall macOS will choose again.",
        }
    }
}

/// What a chosen partition whose NVRAM store could not be read is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unreadable {
    /// The partition holds no Mac system container: an install that did not
    /// finish, which Install macOS repeats.
    Unprepared,
    /// The partition holds a Mac system, or whether it does could not be
    /// read: the boot stops with [`UNREADABLE_STORE`].
    Fatal,
}

/// A store that could not be read on the chosen partition, given whether
/// that partition holds no Mac system container (`blank`, `None` when that
/// could not be read either).
pub fn unreadable_store(blank: Option<bool>) -> Unreadable {
    match blank {
        Some(true) => Unreadable::Unprepared,
        Some(false) | None => Unreadable::Fatal,
    }
}

/// Why the boot stops on [`Unreadable::Fatal`], and what to do.
pub const UNREADABLE_STORE: &str = "NVRAM on the disk chosen for macOS cannot be read: choose \
     Install macOS again, or run stela setup repair on this USB";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_found_partition_is_not_stale() {
        assert_eq!(Stale::of(Search::Found), None);
    }

    #[test]
    fn a_missing_disk_or_partition_is_stale() {
        assert_eq!(Stale::of(Search::NoDisk), Some(Stale::Absent));
        assert_eq!(Stale::of(Search::SeveralDisks), Some(Stale::Ambiguous));
        assert_eq!(Stale::of(Search::NoTable), Some(Stale::Wiped));
        for search in [
            Search::NoPartition,
            Search::SeveralPartitions,
            Search::EspPartition,
        ] {
            assert_eq!(Stale::of(search), Some(Stale::PartitionGone));
        }
    }

    #[test]
    fn each_stale_case_has_its_own_line_and_two_plain_screen_lines() {
        let all = [
            Stale::Absent,
            Stale::Ambiguous,
            Stale::Wiped,
            Stale::PartitionGone,
            Stale::NoMacOs,
        ];
        for (n, a) in all.iter().enumerate() {
            for b in &all[n + 1..] {
                assert_ne!(a.case(), b.case());
            }
            assert_eq!(a.screen().lines().count(), 2, "{a:?}");
            assert!(a.screen().ends_with("Install macOS will choose again."));
            assert!(a.screen().lines().all(|line| line.len() <= 40), "{a:?}");
        }
        assert!(Stale::Wiped.screen().contains("is not there"));
        assert!(Stale::NoMacOs.screen().contains("has no macOS"));
    }

    #[test]
    fn an_unreadable_store_without_a_system_is_an_unfinished_install() {
        assert_eq!(unreadable_store(Some(true)), Unreadable::Unprepared);
        assert_eq!(unreadable_store(Some(false)), Unreadable::Fatal);
        assert_eq!(unreadable_store(None), Unreadable::Fatal);
        assert!(UNREADABLE_STORE.contains("Install macOS"));
        assert!(UNREADABLE_STORE.contains("stela setup repair"));
    }
}
