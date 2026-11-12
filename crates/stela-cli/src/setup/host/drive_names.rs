//! How setup names Windows disks and partitions: `\\.\PhysicalDriveN` is
//! `diskN`, and its partition number M is `diskNpM`. The Windows backend
//! opens the devices; this module only reads the names, so it is compiled
//! and tested on every host.
use anyhow::{Context, Result};

/// `diskN` → N.
pub(super) fn disk_number(identifier: &str) -> Result<u32> {
    identifier
        .strip_prefix("disk")
        .filter(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
        .and_then(|n| n.parse().ok())
        .with_context(|| format!("{identifier} is not a whole disk such as disk2"))
}

/// `diskNpM` → (N, M), M from 1.
pub(super) fn partition_number(identifier: &str) -> Result<(u32, u32)> {
    identifier
        .strip_prefix("disk")
        .and_then(|rest| rest.split_once('p'))
        .filter(|(disk, partition)| {
            [disk, partition]
                .iter()
                .all(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
        })
        .and_then(|(disk, partition)| Some((disk.parse().ok()?, partition.parse().ok()?)))
        .filter(|(_, partition)| *partition > 0)
        .with_context(|| format!("{identifier} is not a partition such as disk2p1"))
}

/// The whole disk `--usb` names, as `diskN`: `PhysicalDriveN`,
/// `\\.\PhysicalDriveN` or `diskN`, in any case.
pub(super) fn whole_disk(name: &str) -> Option<String> {
    let name = name.to_ascii_lowercase();
    let name = name.strip_prefix(r"\\.\").unwrap_or(&name);
    let number = name
        .strip_prefix("physicaldrive")
        .or_else(|| name.strip_prefix("disk"))
        .filter(|n| !n.is_empty() && n.len() <= 3 && n.bytes().all(|b| b.is_ascii_digit()))?;
    Some(format!("disk{}", number.parse::<u32>().ok()?))
}

/// The PCI vendor and device IDs in a device's instance ID, such as
/// `PCI\VEN_10DE&DEV_2786&SUBSYS_...`.
pub(super) fn pci_device(instance: &str) -> Option<(u16, u16)> {
    let ids = instance.strip_prefix("PCI\\VEN_")?;
    let vendor = u16::from_str_radix(ids.get(..4)?, 16).ok()?;
    let device = ids.get(4..)?.strip_prefix("&DEV_")?.get(..4)?;
    Some((vendor, u16::from_str_radix(device, 16).ok()?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whole_disks_are_named_by_physical_drive_or_disk_number() {
        for (typed, disk) in [
            ("PhysicalDrive2", "disk2"),
            (r"\\.\PhysicalDrive2", "disk2"),
            ("physicaldrive10", "disk10"),
            ("disk3", "disk3"),
            ("DISK3", "disk3"),
            ("disk03", "disk3"),
        ] {
            assert_eq!(whole_disk(typed).as_deref(), Some(disk), "{typed}");
        }
        for typed in [
            "",
            "disk",
            "PhysicalDrive",
            "disk2p1",
            "E:",
            r"E:\",
            r"\\.\E:",
            "disk-1",
            "disk1234",
            "PhysicalDrive2 ",
        ] {
            assert_eq!(whole_disk(typed), None, "{typed:?}");
        }
    }

    #[test]
    fn disk_and_partition_identifiers_are_checked() {
        assert_eq!(disk_number("disk0").unwrap(), 0);
        assert_eq!(disk_number("disk12").unwrap(), 12);
        for bad in ["", "disk", "disk1p1", "Disk1", "disk+1", "sdb"] {
            assert!(disk_number(bad).is_err(), "{bad}");
        }
        assert_eq!(partition_number("disk2p1").unwrap(), (2, 1));
        assert_eq!(partition_number("disk10p12").unwrap(), (10, 12));
        for bad in [
            "disk2", "disk2p0", "diskp1", "disk2p", "disk2p+1", "disk+2p1",
        ] {
            assert!(partition_number(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn instance_ids_name_pci_devices() {
        let id = "PCI\\VEN_10DE&DEV_2786&SUBSYS_00000000&REV_A1\\4&0123ABCD&0&0008";
        assert_eq!(pci_device(id), Some((0x10de, 0x2786)));
        assert_eq!(
            pci_device("PCI\\VEN_14C3&DEV_7925&SUBSYS_00000000"),
            Some((0x14c3, 0x7925))
        );
        for other in [
            "USB\\VID_0E8D&PID_0616",
            "PCI\\VEN_10DE",
            "PCI\\VEN_10DE&DEV_27",
        ] {
            assert_eq!(pci_device(other), None, "{other}");
        }
    }
}
