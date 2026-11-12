//! Selecting, among the disks the drivers found (`stela_drivers::probe`),
//! the one partition Stela may use.
//!
//! `stela.cfg` names the disk by serial and the partition by unique GPT GUID.
//! If the native drivers report no matching serial, a single partition with
//! that GUID can identify the disk. EFI system partitions are refused as
//! guest storage targets.
//!
//! Each disk's table is read as boot selection reads it ([`gpt::read`]):
//! the primary GPT, or the backup when an interrupted write left the
//! primary damaged. So a disk the mount selected is a disk it can open.
use super::gpt::{self, GptError, Guid, PartitionDevice};
use alloc::{boxed::Box, string::String, vec::Vec};
use core::fmt;
use stela_drivers::block::{BlockDevice, BlockError, Identity, Shared, Window};

/// The disk and partition `stela.cfg` names, as `SERIAL:GUID`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Target {
    pub serial: String,
    pub partition: Guid,
}

impl core::str::FromStr for Target {
    type Err = &'static str;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let (serial, partition) = text
            .rsplit_once(':')
            .ok_or("storage target must be SERIAL:PARTITION-GUID")?;
        if serial.is_empty() {
            return Err("storage target has an empty serial");
        }
        Ok(Self {
            serial: String::from(serial),
            partition: partition.parse()?,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StorageError {
    /// No disk reports `serial`; `found` lists the serials the disks found
    /// do report.
    NoDisk {
        serial: String,
        found: Vec<String>,
    },
    AmbiguousDisk(String, usize),
    Table(GptError),
    NoPartition(Guid),
    AmbiguousPartition(Guid),
    SystemPartition(Guid),
    Device(BlockError),
    Mismatch {
        offset: usize,
    },
}

impl fmt::Display for StorageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoDisk { serial, found } => {
                write!(f, "no disk has serial {serial:?}")?;
                if found.is_empty() {
                    return f.write_str("; no disk was found");
                }
                f.write_str("; found")?;
                for (index, found) in found.iter().enumerate() {
                    let separator = if index == 0 { " " } else { ", " };
                    write!(f, "{separator}{found:?}")?;
                }
                Ok(())
            }
            Self::AmbiguousDisk(serial, count) => {
                write!(f, "{count} disks have serial {serial:?}")
            }
            Self::Table(error) => error.fmt(f),
            Self::NoPartition(guid) => write!(f, "no partition {guid}"),
            Self::AmbiguousPartition(guid) => write!(f, "several partitions are {guid}"),
            Self::SystemPartition(guid) => {
                write!(f, "partition {guid} is an EFI system partition")
            }
            Self::Device(error) => error.fmt(f),
            Self::Mismatch { offset } => write!(f, "read back differs at byte {offset}"),
        }
    }
}

impl From<BlockError> for StorageError {
    fn from(error: BlockError) -> Self {
        Self::Device(error)
    }
}

/// The index of the one disk in `devices` whose serial is `serial`.
fn find(devices: &[Box<dyn BlockDevice>], serial: &str) -> Result<usize, StorageError> {
    let matching: Vec<usize> = devices
        .iter()
        .enumerate()
        .filter(|(_, device)| device.identity().serial == serial)
        .map(|(index, _)| index)
        .collect();
    match matching[..] {
        [index] => Ok(index),
        [] => Err(StorageError::NoDisk {
            serial: String::from(serial),
            found: devices
                .iter()
                .map(|device| device.identity().serial.clone())
                .collect(),
        }),
        _ => Err(StorageError::AmbiguousDisk(
            String::from(serial),
            matching.len(),
        )),
    }
}

/// Take the whole disk with `serial` out of `devices`, as the guest's USB
/// mass storage sees it: every block, its partition table included.
pub fn take(
    devices: &mut Vec<Box<dyn BlockDevice>>,
    serial: &str,
) -> Result<Box<dyn BlockDevice>, StorageError> {
    let index = find(devices, serial)?;
    Ok(devices.swap_remove(index))
}

/// Take the disk identified by `target` out of `devices` and confine it
/// to the selected partition.
pub fn open(
    devices: &mut Vec<Box<dyn BlockDevice>>,
    target: &Target,
) -> Result<PartitionDevice<Box<dyn BlockDevice>>, StorageError> {
    let (index, partition) = find_target(devices, target)?;
    Ok(PartitionDevice::new(devices.swap_remove(index), partition))
}

/// Check the same native disk and partition selection as `open`, using only
/// device identities and GPT reads. Keep all disks in the inventory.
pub fn verify_target(
    devices: &mut [Box<dyn BlockDevice>],
    target: &Target,
) -> Result<(), StorageError> {
    find_target(devices, target).map(|_| ())
}

/// Take the one disk in `devices` whose GPT holds partition `guid` and
/// confine it to that partition: a partition named by its unique GUID
/// alone, such as the install medium's on the USB `stela.efi` started from,
/// whose serial the host that wrote it could not read. Every disk's table
/// is read; a disk without one is passed over. No disk, or several, holding
/// that GUID is refused, as is an EFI system partition.
pub fn open_unique(
    devices: &mut Vec<Box<dyn BlockDevice>>,
    guid: Guid,
) -> Result<PartitionDevice<Box<dyn BlockDevice>>, StorageError> {
    let (index, partition) = find_unique(devices, guid)?;
    Ok(PartitionDevice::new(devices.swap_remove(index), partition))
}

/// A partition to open: by its disk's serial and its GUID, as `stela.cfg`
/// names a target, or by its unique GUID alone, as the install medium and
/// the translation store are named. An EFI system partition is refused,
/// except as [`Named::Host`]: a partition where Stela keeps a file of its
/// own, such as the USB's boot log, which never reaches the guest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Named {
    Disk(Target),
    Unique(Guid),
    Host(Guid),
}

/// Open every partition of `named`, each disk taken once and shared by the
/// partitions found on it: the Mac's internal SSD, the install medium and
/// the translation store `stela.efi` rewrites are partitions of one USB, and
/// one partition may be named twice. When any is refused, no disk is taken
/// and the refused one's index comes back with the reason.
#[expect(
    clippy::result_large_err,
    reason = "preserve the exact partition index and typed device failure without allocating on the error path"
)]
pub fn open_shared(
    devices: &mut Vec<Box<dyn BlockDevice>>,
    named: &[Named],
) -> Result<Vec<PartitionDevice<Window>>, (usize, StorageError)> {
    let found = named
        .iter()
        .enumerate()
        .map(|(at, named)| locate(devices, named).map_err(|error| (at, error)))
        .collect::<Result<Vec<_>, _>>()?;
    let mut indices: Vec<usize> = found.iter().map(|(index, _)| *index).collect();
    indices.sort_unstable();
    indices.dedup();
    // Highest first, so each swap_remove leaves the lower indices in place.
    let disks: Vec<(usize, Shared)> = indices
        .iter()
        .rev()
        .map(|&index| (index, Shared::new(devices.swap_remove(index))))
        .collect();
    Ok(found
        .into_iter()
        .map(|(index, partition)| {
            let (_, disk) = disks
                .iter()
                .find(|(taken, _)| *taken == index)
                .expect("every found disk is taken");
            PartitionDevice::new(disk.whole(), partition)
        })
        .collect())
}

/// The index of the disk holding `named` and its partition, refusing an
/// EFI system partition unless it is [`Named::Host`].
fn locate(
    devices: &mut [Box<dyn BlockDevice>],
    named: &Named,
) -> Result<(usize, gpt::Partition), StorageError> {
    match named {
        Named::Unique(guid) => find_unique(devices, *guid),
        Named::Host(guid) => find_any(devices, *guid),
        Named::Disk(target) => find_target(devices, target),
    }
}

/// The disk and partition `target` names. The serial was read through the
/// firmware when the target was chosen; Stela's own driver may report the
/// same disk under another serial (a USB bridge's INQUIRY serial against
/// its USB string, NVMe padding) or none. When no disk has the serial, the
/// partition's unique GUID, which the chooser wrote fresh, finds it: on
/// exactly one disk, and never an EFI system partition.
fn find_target(
    devices: &mut [Box<dyn BlockDevice>],
    target: &Target,
) -> Result<(usize, gpt::Partition), StorageError> {
    let index = match find(devices, &target.serial) {
        Ok(index) => index,
        Err(error @ StorageError::NoDisk { .. }) => {
            return match find_unique(devices, target.partition) {
                Err(StorageError::NoPartition(_)) => Err(error),
                found => found,
            };
        }
        Err(error) => return Err(error),
    };
    let table = gpt::read(&mut *devices[index]).map_err(StorageError::Table)?;
    let mut matching = table
        .partitions
        .into_iter()
        .filter(|partition| partition.unique_guid == target.partition);
    let partition = matching
        .next()
        .ok_or(StorageError::NoPartition(target.partition))?;
    if matching.next().is_some() {
        return Err(StorageError::AmbiguousPartition(target.partition));
    }
    if partition.type_guid == Guid::ESP {
        return Err(StorageError::SystemPartition(target.partition));
    }
    Ok((index, partition))
}

/// Each of `named`'s disk targets that [`find_target`] finds by its
/// partition GUID on a disk reporting another serial: the serial named and
/// the one the disk reports, for the log.
pub fn serial_fallbacks(
    devices: &mut [Box<dyn BlockDevice>],
    named: &[Named],
) -> Vec<(String, String)> {
    let mut found = Vec::new();
    for named in named {
        let Named::Disk(target) = named else {
            continue;
        };
        if find(devices, &target.serial).is_ok() {
            continue;
        }
        if let Ok((index, _)) = find_target(devices, target) {
            found.push((
                target.serial.clone(),
                devices[index].identity().serial.clone(),
            ));
        }
    }
    found
}

/// The index of the one disk in `devices` whose GPT holds partition `guid`,
/// and that partition, as [`open_unique`] requires them.
fn find_unique(
    devices: &mut [Box<dyn BlockDevice>],
    guid: Guid,
) -> Result<(usize, gpt::Partition), StorageError> {
    let (index, partition) = find_any(devices, guid)?;
    if partition.type_guid == Guid::ESP {
        return Err(StorageError::SystemPartition(guid));
    }
    Ok((index, partition))
}

/// The index of the one disk whose GPT holds partition `guid`, of any type,
/// and that partition.
fn find_any(
    devices: &mut [Box<dyn BlockDevice>],
    guid: Guid,
) -> Result<(usize, gpt::Partition), StorageError> {
    let mut found = None;
    for (index, device) in devices.iter_mut().enumerate() {
        let Ok(table) = gpt::read(&mut **device) else {
            continue;
        };
        let mut matching = table
            .partitions
            .into_iter()
            .filter(|p| p.unique_guid == guid);
        let Some(partition) = matching.next() else {
            continue;
        };
        if matching.next().is_some() || found.is_some() {
            return Err(StorageError::AmbiguousPartition(guid));
        }
        found = Some((index, partition));
    }
    found.ok_or(StorageError::NoPartition(guid))
}

/// How long [`wait_for_named`] keeps looking for a missing partition: a USB
/// device can connect, train its link and answer its first commands well
/// after its controller started, as Linux finds it later through a port
/// change. The wait goes on while devices keep arriving or failing, ends
/// `quiet_us` after the last such news, and never passes `total_us`.
#[derive(Clone, Copy, Debug)]
pub struct Patience {
    pub quiet_us: u64,
    pub total_us: u64,
    /// The pause between two looks at the USB hosts.
    pub poll_us: u64,
}

/// 10 s without news, 25 s in all, a look every 50 ms. A stick that
/// answers its first READ CAPACITY in a few seconds after its port settles
/// is found; a port that never connects ends the wait in 10 s.
pub const PATIENCE: Patience = Patience {
    quiet_us: 10_000_000,
    total_us: 25_000_000,
    poll_us: 50_000,
};

/// What one look at the USB hosts brought: disks that arrived, and whether
/// anything else changed (a device added, removed or failed).
#[derive(Default)]
pub struct Arrival {
    pub disks: Vec<Box<dyn BlockDevice>>,
    pub news: bool,
}

/// How [`wait_for_named`] ended.
#[derive(Debug, PartialEq, Eq)]
pub struct Waited {
    /// The first partition still missing, by its index in `named`.
    pub missing: Option<(usize, StorageError)>,
    pub waited_us: u64,
    pub arrived: usize,
    pub looks: usize,
}

/// The first of `named` that no disk holds. Any other refusal, such as two
/// disks holding one GUID, is not waited for.
fn missing(devices: &mut [Box<dyn BlockDevice>], named: &[Named]) -> Option<(usize, StorageError)> {
    named
        .iter()
        .enumerate()
        .find_map(|(at, named)| match locate(devices, named) {
            Err(error @ (StorageError::NoDisk { .. } | StorageError::NoPartition(_))) => {
                Some((at, error))
            }
            _ => None,
        })
}

/// Wait until every one of `named` is on a disk of `devices`, taking the
/// disks each `look` brings, by [`Patience`]. Nothing is looked at, and no
/// time passes, when all are there already.
pub fn wait_for_named(
    devices: &mut Vec<Box<dyn BlockDevice>>,
    named: &[Named],
    patience: Patience,
    now_us: &mut dyn FnMut() -> u64,
    look: &mut dyn FnMut() -> Arrival,
    pause: &mut dyn FnMut(u64),
) -> Waited {
    let mut waited = Waited {
        missing: missing(devices, named),
        waited_us: 0,
        arrived: 0,
        looks: 0,
    };
    if waited.missing.is_none() {
        return waited;
    }
    let start = now_us();
    let mut news_at = start;
    loop {
        let now = now_us();
        waited.waited_us = now.wrapping_sub(start);
        if waited.waited_us >= patience.total_us || now.wrapping_sub(news_at) >= patience.quiet_us {
            return waited;
        }
        pause(patience.poll_us);
        let arrival = look();
        waited.looks += 1;
        if arrival.news || !arrival.disks.is_empty() {
            news_at = now_us();
        }
        if !arrival.disks.is_empty() {
            waited.arrived += arrival.disks.len();
            devices.extend(arrival.disks);
            waited.missing = missing(devices, named);
            if waited.missing.is_none() {
                waited.waited_us = now_us().wrapping_sub(start);
                return waited;
            }
        }
    }
}

/// The byte at `offset` of the storage check's pattern.
pub fn pattern(offset: u64) -> u8 {
    let mixed = offset.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ offset >> 7;
    (mixed >> 56) as u8
}

/// Write `bytes` of the pattern at the start of `device`, flush, read them
/// back, compare, and confirm the first block past the end is refused.
/// Returns the SHA-256 of what was read.
pub fn check(device: &mut dyn BlockDevice, bytes: usize) -> Result<String, StorageError> {
    let identity: Identity = device.identity().clone();
    let data: Vec<u8> = (0..bytes as u64).map(pattern).collect();
    device.write(0, &data)?;
    device.flush()?;
    let mut read = vec![0u8; bytes];
    device.read(0, &mut read)?;
    if let Some(offset) = read.iter().zip(&data).position(|(a, b)| a != b) {
        return Err(StorageError::Mismatch { offset });
    }
    let mut block = vec![0u8; identity.block_size as usize];
    match device.read(identity.blocks, &mut block) {
        Err(BlockError::OutOfRange { .. }) => {}
        Ok(()) => {
            return Err(StorageError::Device(BlockError::Unsupported(
                "a read past the partition succeeded",
            )));
        }
        Err(error) => return Err(error.into()),
    }
    Ok(stela_formats::sha256(&read))
}

#[cfg(test)]
mod tests {
    use super::super::gpt::tests as table;
    use super::*;
    use stela_drivers::block::RamDisk;

    const TARGET: &str = "AAAAAAAA-0000-0000-0000-000000000002";

    fn disks() -> Vec<Box<dyn BlockDevice>> {
        let data: Guid = table::DATA.parse().unwrap();
        let layout = [
            table::partition(Guid::ESP, "AAAAAAAA-0000-0000-0000-000000000001", 34, 99),
            table::partition(data, TARGET, 100, 199),
        ];
        vec![
            Box::new(table::formatted("disk-a", &layout)),
            Box::new(table::formatted("disk-b", &layout)),
            Box::new(RamDisk::new("blank", 256, 512)),
        ]
    }

    fn target(text: &str) -> Target {
        text.parse().unwrap()
    }

    #[test]
    fn targets_name_a_serial_and_a_partition_guid() {
        let parsed = target("S4EV:NX0R-1:AAAAAAAA-0000-0000-0000-000000000002");
        assert_eq!(parsed.serial, "S4EV:NX0R-1");
        assert_eq!(parsed.partition, TARGET.parse().unwrap());
        assert!(
            "AAAAAAAA-0000-0000-0000-000000000002"
                .parse::<Target>()
                .is_err()
        );
        assert!(
            ":AAAAAAAA-0000-0000-0000-000000000002"
                .parse::<Target>()
                .is_err()
        );
        assert!("disk:not-a-guid".parse::<Target>().is_err());
    }

    #[test]
    fn selection_requires_one_disk_and_one_non_system_partition() {
        let mut devices = disks();
        let mut device = open(&mut devices, &target(&format!("disk-b:{TARGET}"))).unwrap();
        assert_eq!(devices.len(), 2);
        assert_eq!(device.disk().serial, "disk-b");
        assert_eq!(device.identity().blocks, 100);
        let hash = check(&mut device, 4096).unwrap();
        let expected: Vec<u8> = (0..4096).map(pattern).collect();
        assert_eq!(hash, stela_formats::sha256(&expected));

        let mut devices = disks();
        assert_eq!(
            open(
                &mut devices,
                &target("disk-c:BBBBBBBB-0000-0000-0000-000000000000")
            )
            .err(),
            Some(StorageError::NoDisk {
                serial: String::from("disk-c"),
                found: vec![
                    String::from("disk-a"),
                    String::from("disk-b"),
                    String::from("blank")
                ],
            })
        );
        assert_eq!(
            StorageError::NoDisk {
                serial: String::from("disk-c"),
                found: vec![String::from("disk-a"), String::from("blank")],
            }
            .to_string(),
            "no disk has serial \"disk-c\"; found \"disk-a\", \"blank\""
        );
        assert_eq!(
            StorageError::NoDisk {
                serial: String::from("disk-c"),
                found: Vec::new(),
            }
            .to_string(),
            "no disk has serial \"disk-c\"; no disk was found"
        );
        let esp = "AAAAAAAA-0000-0000-0000-000000000001";
        assert_eq!(
            open(&mut devices, &target(&format!("disk-a:{esp}"))).err(),
            Some(StorageError::SystemPartition(esp.parse().unwrap()))
        );
        let other = "BBBBBBBB-0000-0000-0000-000000000000";
        assert_eq!(
            open(&mut devices, &target(&format!("disk-a:{other}"))).err(),
            Some(StorageError::NoPartition(other.parse().unwrap()))
        );
        assert_eq!(
            open(&mut devices, &target(&format!("blank:{TARGET}"))).err(),
            Some(StorageError::Table(GptError::NoTable))
        );
        devices.push(Box::new(RamDisk::new("disk-a", 8, 512)));
        assert_eq!(
            open(&mut devices, &target(&format!("disk-a:{TARGET}"))).err(),
            Some(StorageError::AmbiguousDisk(String::from("disk-a"), 2))
        );
        assert_eq!(devices.len(), 4);
    }

    /// A fake clock and USB host for [`wait_for_named`]: the pause moves
    /// time, and the stick arrives at `arrives_us` (never with `None`).
    struct Late {
        now: u64,
        arrives_us: Option<u64>,
        news_every_us: Option<u64>,
        stick: Option<Box<dyn BlockDevice>>,
    }

    fn stick(medium: &str) -> Box<dyn BlockDevice> {
        let data: Guid = table::DATA.parse().unwrap();
        Box::new(table::formatted(
            "",
            &[table::partition(data, medium, 40, 139)],
        ))
    }

    fn wait_late(
        late: &core::cell::RefCell<Late>,
        devices: &mut Vec<Box<dyn BlockDevice>>,
        named: &[Named],
    ) -> Waited {
        wait_for_named(
            devices,
            named,
            PATIENCE,
            &mut || late.borrow().now,
            &mut || {
                let mut late = late.borrow_mut();
                let news = late
                    .news_every_us
                    .is_some_and(|every| late.now % every < PATIENCE.poll_us);
                let disks = match late.arrives_us {
                    Some(at) if late.now >= at => late.stick.take().into_iter().collect(),
                    _ => Vec::new(),
                };
                Arrival { disks, news }
            },
            &mut |us| late.borrow_mut().now += us,
        )
    }

    #[test]
    fn a_stick_that_arrives_late_is_waited_for_and_found() {
        const MEDIUM: &str = "CCCCCCCC-0000-0000-0000-000000000003";
        let named = [Named::Unique(MEDIUM.parse().unwrap())];
        let late = core::cell::RefCell::new(Late {
            now: 1_000,
            arrives_us: Some(1_000 + 5_000_000),
            news_every_us: None,
            stick: Some(stick(MEDIUM)),
        });
        let mut devices = disks();
        let waited = wait_late(&late, &mut devices, &named);
        assert_eq!(waited.missing, None);
        assert_eq!(waited.arrived, 1);
        assert!(
            (5_000_000..5_100_000).contains(&waited.waited_us),
            "{waited:?}"
        );
        assert!(open_shared(&mut devices, &named).is_ok());
    }

    #[test]
    fn nothing_is_waited_for_when_every_partition_is_there() {
        let named = [Named::Disk(target(&alloc::format!("disk-a:{TARGET}")))];
        let late = core::cell::RefCell::new(Late {
            now: 7,
            arrives_us: None,
            news_every_us: None,
            stick: None,
        });
        let mut devices = disks();
        let waited = wait_late(&late, &mut devices, &named);
        assert_eq!(
            waited,
            Waited {
                missing: None,
                waited_us: 0,
                arrived: 0,
                looks: 0
            }
        );
        assert_eq!(late.borrow().now, 7);
    }

    #[test]
    fn a_stick_that_never_arrives_ends_the_wait_quietly_or_at_the_total_bound() {
        const MEDIUM: &str = "CCCCCCCC-0000-0000-0000-000000000003";
        let guid: Guid = MEDIUM.parse().unwrap();
        let named = [Named::Unique(guid)];
        // No news at all: the quiet bound ends it.
        let late = core::cell::RefCell::new(Late {
            now: 0,
            arrives_us: None,
            news_every_us: None,
            stick: None,
        });
        let waited = wait_late(&late, &mut disks(), &named);
        assert_eq!(waited.missing, Some((0, StorageError::NoPartition(guid))));
        assert!((PATIENCE.quiet_us..PATIENCE.quiet_us + 100_000).contains(&waited.waited_us));
        // A port that keeps failing its reset brings news every second:
        // the total bound ends it.
        let late = core::cell::RefCell::new(Late {
            now: 0,
            arrives_us: None,
            news_every_us: Some(1_000_000),
            stick: None,
        });
        let waited = wait_late(&late, &mut disks(), &named);
        assert_eq!(waited.missing, Some((0, StorageError::NoPartition(guid))));
        assert!((PATIENCE.total_us..PATIENCE.total_us + 100_000).contains(&waited.waited_us));
    }

    /// The chooser read the disk's serial through the firmware; Stela's
    /// driver reports another. The fresh partition GUID still finds it,
    /// once; on two disks it is refused; on none the serial's error stands.
    #[test]
    fn a_target_whose_serial_differs_is_found_by_its_partition_guid_on_one_disk() {
        let data: Guid = table::DATA.parse().unwrap();
        const FRESH: &str = "DDDDDDDD-0000-0000-0000-000000000005";
        let mut devices = disks();
        devices.push(Box::new(table::formatted(
            "INQUIRY-SERIAL",
            &[table::partition(data, FRESH, 40, 139)],
        )));
        let chosen = target(&alloc::format!("FIRMWARE-SERIAL:{FRESH}"));
        let named = [Named::Disk(chosen.clone())];
        assert_eq!(
            serial_fallbacks(&mut devices, &named),
            [(
                String::from("FIRMWARE-SERIAL"),
                String::from("INQUIRY-SERIAL")
            )]
        );
        let opened = open_shared(&mut devices, &named).unwrap();
        assert_eq!(opened[0].identity().blocks, 100);
        // The same GUID on two disks is refused.
        let mut devices = disks();
        for serial in ["one", "two"] {
            devices.push(Box::new(table::formatted(
                serial,
                &[table::partition(data, FRESH, 40, 139)],
            )));
        }
        assert_eq!(
            open_shared(&mut devices, &named).err(),
            Some((0, StorageError::AmbiguousPartition(FRESH.parse().unwrap())))
        );
        assert!(open(&mut devices, &chosen).is_err());
        // Neither the serial nor the GUID: the serial's error, with the
        // serials found, which the account and the wait use.
        let mut devices = disks();
        assert!(matches!(
            open_shared(&mut devices, &named).err(),
            Some((0, StorageError::NoDisk { .. }))
        ));
        assert!(serial_fallbacks(&mut devices, &named).is_empty());
        // An EFI system partition is never taken by its GUID.
        let esp = "AAAAAAAA-0000-0000-0000-000000000001";
        let mut devices = disks();
        devices.remove(1);
        let named = [Named::Disk(target(&alloc::format!("other:{esp}")))];
        assert_eq!(
            open_shared(&mut devices, &named).err(),
            Some((0, StorageError::SystemPartition(esp.parse().unwrap())))
        );
    }

    #[test]
    fn saved_install_target_can_be_verified_on_a_read_only_disk() {
        let esp = "AAAAAAAA-0000-0000-0000-000000000001";
        let data = table::DATA.parse().unwrap();
        let mut disk = table::formatted(
            "selected",
            &[
                table::partition(Guid::ESP, esp, 34, 99),
                table::partition(data, TARGET, 100, 199),
            ],
        );
        disk.identity.read_only = true;
        let mut devices: Vec<Box<dyn BlockDevice>> = vec![Box::new(disk)];
        let selected = target(&format!("selected:{TARGET}"));
        verify_target(&mut devices, &selected).unwrap();
        verify_target(&mut devices, &target(&format!("absent:{TARGET}"))).unwrap();
        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].identity().serial, "selected");
        assert!(matches!(
            verify_target(&mut devices, &target(&format!("absent:{}", Guid::ESP))),
            Err(StorageError::NoDisk { .. })
        ));
        assert_eq!(
            verify_target(&mut devices, &target(&format!("selected:{}", Guid::ESP))),
            Err(StorageError::NoPartition(Guid::ESP))
        );
        assert_eq!(
            verify_target(&mut devices, &target(&format!("selected:{esp}"))),
            Err(StorageError::SystemPartition(esp.parse().unwrap()))
        );
        devices.push(Box::new(RamDisk::new("selected", 8, 512)));
        assert!(matches!(
            verify_target(&mut devices, &selected),
            Err(StorageError::AmbiguousDisk(_, 2))
        ));
    }

    #[test]
    fn a_partition_is_found_by_its_guid_alone_on_exactly_one_disk() {
        let data: Guid = table::DATA.parse().unwrap();
        let medium = "CCCCCCCC-0000-0000-0000-000000000003";
        let mut devices = disks();
        devices.push(Box::new(table::formatted(
            "",
            &[table::partition(data, medium, 40, 139)],
        )));
        let device = open_unique(&mut devices, medium.parse().unwrap()).unwrap();
        assert_eq!((device.identity().blocks, devices.len()), (100, 3));
        // The target GUID is on both disk-a and disk-b.
        assert_eq!(
            open_unique(&mut devices, TARGET.parse().unwrap()).err(),
            Some(StorageError::AmbiguousPartition(TARGET.parse().unwrap()))
        );
        assert_eq!(
            open_unique(&mut devices, medium.parse().unwrap()).err(),
            Some(StorageError::NoPartition(medium.parse().unwrap()))
        );
        let esp = "AAAAAAAA-0000-0000-0000-000000000001";
        devices.remove(0);
        assert_eq!(
            open_unique(&mut devices, esp.parse().unwrap()).err(),
            Some(StorageError::SystemPartition(esp.parse().unwrap()))
        );
        assert_eq!(devices.len(), 2);
        // Stela's own files may sit on the ESP; a guest partition may not.
        let esp: Guid = esp.parse().unwrap();
        assert_eq!(
            open_shared(&mut devices, &[Named::Unique(esp)]).err(),
            Some((0, StorageError::SystemPartition(esp)))
        );
        let opened = open_shared(&mut devices, &[Named::Host(esp)]).unwrap();
        assert_eq!((opened.len(), devices.len()), (1, 1));
    }

    #[test]
    fn partitions_of_one_disk_share_it_and_a_refusal_takes_no_disk() {
        let data: Guid = table::DATA.parse().unwrap();
        let (medium, store) = (
            "CCCCCCCC-0000-0000-0000-000000000003",
            "CCCCCCCC-0000-0000-0000-000000000004",
        );
        let mut devices = disks();
        devices.push(Box::new(table::formatted(
            "usb",
            &[
                table::partition(data, medium, 40, 139),
                table::partition(data, store, 140, 179),
            ],
        )));
        let guids: Vec<Named> = [medium, store, store]
            .iter()
            .map(|guid| Named::Unique(guid.parse().unwrap()))
            .collect();
        assert_eq!(
            open_shared(
                &mut devices,
                &[guids[0].clone(), Named::Unique(TARGET.parse().unwrap())]
            )
            .err(),
            Some((1, StorageError::AmbiguousPartition(TARGET.parse().unwrap())))
        );
        assert_eq!(devices.len(), 4, "a refusal takes no disk");
        let mut opened = open_shared(&mut devices, &guids).unwrap();
        assert_eq!(devices.len(), 3, "the USB is taken once");
        let blocks: Vec<u64> = opened
            .iter()
            .map(|device| device.identity().blocks)
            .collect();
        assert_eq!(blocks, [100, 40, 40]);
        opened[1].write(0, &[7; 512]).unwrap();
        let mut read = [0; 512];
        opened[2].read(0, &mut read).unwrap();
        assert_eq!(read, [7; 512], "one partition opened twice is one");
        opened[0].read(99, &mut read).unwrap();
        assert_ne!(read, [7; 512], "the medium ends before the store");
    }

    #[test]
    fn a_target_shares_its_disk_with_partitions_named_by_guid() {
        let data: Guid = table::DATA.parse().unwrap();
        let (medium, ssd) = (
            "CCCCCCCC-0000-0000-0000-000000000003",
            "CCCCCCCC-0000-0000-0000-000000000005",
        );
        let mut devices = disks();
        devices.push(Box::new(table::formatted(
            "usb",
            &[
                table::partition(data, medium, 40, 139),
                table::partition(data, ssd, 140, 199),
            ],
        )));
        let ssd = Named::Disk(Target {
            serial: "usb".into(),
            partition: ssd.parse().unwrap(),
        });
        let opened =
            open_shared(&mut devices, &[ssd, Named::Unique(medium.parse().unwrap())]).unwrap();
        assert_eq!(devices.len(), 3, "the USB is taken once");
        let blocks: Vec<u64> = opened
            .iter()
            .map(|device| device.identity().blocks)
            .collect();
        assert_eq!(blocks, [60, 100]);
        // A serial no disk reports refuses before any disk is taken.
        let missing = Named::Disk(Target {
            serial: "none".into(),
            partition: TARGET.parse().unwrap(),
        });
        assert_eq!(
            open_shared(
                &mut devices,
                &[Named::Unique(TARGET.parse().unwrap()), missing]
            )
            .err(),
            Some((0, StorageError::AmbiguousPartition(TARGET.parse().unwrap())))
        );
        assert_eq!(devices.len(), 3);
    }

    #[test]
    fn whole_disks_are_taken_by_one_serial() {
        let mut devices = disks();
        let disk = take(&mut devices, "blank").unwrap();
        assert_eq!((disk.identity().blocks, devices.len()), (256, 2));
        assert!(matches!(
            take(&mut devices, "blank"),
            Err(StorageError::NoDisk { .. })
        ));
        devices.push(Box::new(RamDisk::new("disk-a", 8, 512)));
        assert!(matches!(
            take(&mut devices, "disk-a"),
            Err(StorageError::AmbiguousDisk(_, 2))
        ));
    }

    #[test]
    fn a_damaged_primary_gpt_opens_through_the_backup_as_selection_finds_it() {
        let mut devices = disks();
        // An interrupted write: the primary header's CRC no longer matches.
        let mut header = [0; 512];
        devices[1].read(1, &mut header).unwrap();
        header[16] ^= 0xff;
        devices[1].write(1, &header).unwrap();
        let mut device = open(&mut devices, &target(&format!("disk-b:{TARGET}"))).unwrap();
        assert_eq!(device.disk().serial, "disk-b");
        assert_eq!(device.identity().blocks, 100);
        check(&mut device, 4096).unwrap();
        // A damaged primary and backup is still refused.
        let mut devices = disks();
        let mut block = [0; 512];
        for lba in [1, 255] {
            devices[0].read(lba, &mut block).unwrap();
            block[16] ^= 0xff;
            devices[0].write(lba, &block).unwrap();
        }
        assert!(matches!(
            open(&mut devices, &target(&format!("disk-a:{TARGET}"))),
            Err(StorageError::Table(GptError::Invalid(_)))
        ));
    }

    #[test]
    fn the_pattern_is_not_trivially_periodic() {
        let first: Vec<u8> = (0..512).map(pattern).collect();
        let second: Vec<u8> = (512..1024).map(pattern).collect();
        assert_ne!(first, second);
        assert!(first.iter().any(|byte| *byte != first[0]));
    }
}
