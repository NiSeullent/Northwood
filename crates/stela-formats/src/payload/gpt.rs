//! GUID partition tables: the one reader and encoder every Stela tool and
//! the mount use, and the plan of the partitions Stela adds beside an
//! untouched EFI system partition.
//!
//! The Apple region is the one partition the guest's ANS2 namespace exposes;
//! Apple's restore writes its own nested GPT inside it. Staging, machine
//! state and the translation cache stay outside it, so the host ESP and
//! generated code never mix with Apple bytes.
use super::PayloadError;
use alloc::{string::String, vec, vec::Vec};
use core::{fmt, str::FromStr};
use stela_contracts::bytes::{array, le32, le64};

/// The signature that opens a GPT header.
pub const SIGNATURE: &[u8; 8] = b"EFI PART";
const REVISION: u32 = 0x0001_0000;
const HEADER_BYTES: u32 = 92;
const ENTRY_BYTES: u32 = 128;
const ENTRY_COUNT: u32 = 128;
/// The entry array always takes 16 KiB, whatever the block size.
const ENTRY_ARRAY_BYTES: u64 = ENTRY_BYTES as u64 * ENTRY_COUNT as u64;
const MIB: u64 = 1024 * 1024;
/// Partitions start and end on 1 MiB boundaries.
pub const ALIGNMENT: u64 = MIB;
/// Left free at the start of a blank disk for an ESP that a later step
/// creates and formats; nothing is written there.
pub const RESERVED_ESP_BYTES: u64 = 512 * MIB;
pub const MACHINE_STATE_BYTES: u64 = 256 * MIB;
pub const DEFAULT_CACHE_BYTES: u64 = 16 * 1024 * MIB;
/// The smallest Mac's SSD (the Apple region, without a machine state in its
/// tail) Stela installs macOS onto, in decimal bytes as disks are sold:
/// `stela setup`, `stela.efi`'s installer and the docs all use it.
///
/// Measured from macOS 27.2 (26B123)'s erase identities for the
/// MacBookPro18 boards: a 13101 MiB system partition
/// (`MinimumSystemPartition`) with 640 MiB of padding on a 64 GB disk
/// (`SystemPartitionPadding`), 2814 MiB of system and app cryptexes, the
/// 6.5 GB Rosetta cryptex, 0.87 GB of `/private/var` (`OSVarContentSize`)
/// and two 1.3 GB recoveryOS images (paired and System Recovery): about
/// 27 GB installed. A staged update needs a second system and its
/// cryptexes, about 17 GB more. 60 GB leaves room for both and for user
/// data, and a disk sold as 64 GB (61.5 GB or more) holds it.
pub const DEFAULT_MIN_APPLE_BYTES: u64 = 60_000_000_000;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Guid(pub [u8; 16]);

impl Guid {
    /// The EFI system partition type.
    pub const ESP: Self = Self([
        0x28, 0x73, 0x2a, 0xc1, 0x1f, 0xf8, 0xd2, 0x11, 0xba, 0x4b, 0x00, 0xa0, 0xc9, 0x3e, 0xc9,
        0x3b,
    ]);

    /// Parse the canonical text form; the first three groups are stored
    /// little-endian, as on disk.
    pub fn parse(text: &str) -> Option<Self> {
        let groups: Vec<&str> = text.split('-').collect();
        let widths = [8, 4, 4, 4, 12];
        if groups.len() != 5
            || groups.iter().zip(widths).any(|(group, width)| {
                group.len() != width || !group.bytes().all(|byte| byte.is_ascii_hexdigit())
            })
        {
            return None;
        }
        let mut raw = [0u8; 16];
        let mut position = 0;
        for group in &groups {
            for pair in group.as_bytes().chunks(2) {
                let text = core::str::from_utf8(pair).ok()?;
                raw[position] = u8::from_str_radix(text, 16).ok()?;
                position += 1;
            }
        }
        raw[0..4].reverse();
        raw[4..6].reverse();
        raw[6..8].reverse();
        Some(Self(raw))
    }

    /// A random-format (version 4) GUID from 16 bytes of identity entropy.
    pub fn from_entropy(mut bytes: [u8; 16]) -> Self {
        bytes[7] = (bytes[7] & 0x0f) | 0x40;
        bytes[8] = (bytes[8] & 0x3f) | 0x80;
        Self(bytes)
    }

    pub fn is_zero(&self) -> bool {
        self.0 == [0; 16]
    }
}

impl FromStr for Guid {
    type Err = &'static str;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        Self::parse(text).ok_or("GUID must be XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX")
    }
}

impl fmt::Display for Guid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let b = &self.0;
        write!(
            f,
            "{:02X}{:02X}{:02X}{:02X}-{:02X}{:02X}-{:02X}{:02X}-",
            b[3], b[2], b[1], b[0], b[5], b[4], b[7], b[6]
        )?;
        for byte in &b[8..10] {
            write!(f, "{byte:02X}")?;
        }
        f.write_str("-")?;
        for byte in &b[10..] {
            write!(f, "{byte:02X}")?;
        }
        Ok(())
    }
}

impl fmt::Debug for Guid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

/// What a partition is for, by type GUID.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Esp,
    AppleRegion,
    Staging,
    MachineState,
    Cache,
    NorthwoodSwap,
}

impl Role {
    pub const STELA: [Self; 4] = [
        Self::MachineState,
        Self::Staging,
        Self::Cache,
        Self::AppleRegion,
    ];
    /// E9AFCDBE-E70E-40C5-8A5F-AF97B70BD75D in GPT byte order.
    pub const NORTHWOOD_SWAP_TYPE: Guid = Guid([
        0xbe, 0xcd, 0xaf, 0xe9, 0x0e, 0xe7, 0xc5, 0x40, 0x8a, 0x5f, 0xaf, 0x97, 0xb7, 0x0b, 0xd7,
        0x5d,
    ]);
    pub const NORTHWOOD_SWAP_NAME: &str = "Northwood Swap";

    pub fn type_guid(self) -> Guid {
        Guid::parse(match self {
            Self::Esp => "C12A7328-F81F-11D2-BA4B-00A0C93EC93B",
            Self::AppleRegion => "09137F55-F9C2-4DE6-B5DE-5A68DF821BDC",
            Self::Staging => "99D6A603-EC02-4167-A0AB-653442B4A265",
            Self::MachineState => "EC61F267-6C83-493C-9BBE-DD26518023BD",
            Self::Cache => "F4F9CDCC-7F4D-4331-A060-8FB3D87B54B8",
            Self::NorthwoodSwap => return Self::NORTHWOOD_SWAP_TYPE,
        })
        .expect("constant GUID")
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Esp => "EFI System",
            Self::AppleRegion => "Stela Apple region",
            Self::Staging => "Stela staging",
            Self::MachineState => "Stela machine state",
            Self::Cache => "Stela translation cache",
            Self::NorthwoodSwap => Self::NORTHWOOD_SWAP_NAME,
        }
    }

    fn of(type_guid: Guid) -> Option<Self> {
        [Self::Esp, Self::NorthwoodSwap]
            .into_iter()
            .chain(Self::STELA)
            .find(|role| role.type_guid() == type_guid)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Partition {
    pub type_guid: Guid,
    pub unique_guid: Guid,
    pub first_lba: u64,
    /// Inclusive, as in the GPT entry.
    pub last_lba: u64,
    pub attributes: u64,
    pub name: String,
}

impl Partition {
    pub fn role(&self) -> Option<Role> {
        Role::of(self.type_guid)
    }

    pub fn blocks(&self) -> u64 {
        self.last_lba - self.first_lba + 1
    }
}

/// A disk's geometry: logical block size and number of blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Disk {
    pub block_bytes: u64,
    pub blocks: u64,
}

impl Disk {
    fn entry_blocks(&self) -> u64 {
        ENTRY_ARRAY_BYTES.div_ceil(self.block_bytes)
    }

    pub fn first_usable(&self) -> u64 {
        2 + self.entry_blocks()
    }

    pub fn last_usable(&self) -> u64 {
        self.blocks - 2 - self.entry_blocks()
    }

    fn validate(&self) -> Result<(), PayloadError> {
        if !matches!(self.block_bytes, 512 | 1024 | 2048 | 4096)
            || self.blocks < 2 * (2 + self.entry_blocks()) + 1
        {
            return Err(PayloadError::Disk("unsupported block size or tiny disk"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Table {
    pub disk_guid: Guid,
    pub partitions: Vec<Partition>,
}

/// The IEEE CRC-32 of GPT headers and entry arrays.
pub use stela_contracts::crc::crc32_ieee as crc32;

fn guid_at(bytes: &[u8], at: usize) -> Guid {
    Guid(array(bytes, at))
}

/// The location of the entry array a header names, and the blocks its
/// partitions may use.
pub struct HeaderInfo {
    pub disk_guid: Guid,
    alternate_lba: u64,
    supported_header: bool,
    pub first_usable: u64,
    /// Inclusive, as in the header.
    pub last_usable: u64,
    pub entries_lba: u64,
    pub entry_count: u32,
    pub entry_bytes: u32,
    entries_crc: u32,
}

impl HeaderInfo {
    pub fn entry_array_bytes(&self) -> u64 {
        self.entry_count as u64 * self.entry_bytes as u64
    }
}

/// Parse the header block read from `lba`, the primary (1) or the backup
/// (last). `Ok(None)` means there is no GPT signature; a damaged GPT is an
/// error, never treated as blank.
pub fn parse_header(
    disk: Disk,
    lba: u64,
    block: &[u8],
) -> Result<Option<HeaderInfo>, PayloadError> {
    disk.validate()?;
    if block.len() < HEADER_BYTES as usize || &block[..8] != SIGNATURE {
        return Ok(None);
    }
    let size = le32(block, 12);
    if !(HEADER_BYTES..=disk.block_bytes as u32).contains(&size) {
        return Err(PayloadError::Disk("GPT header size"));
    }
    let mut copy = block
        .get(..size as usize)
        .ok_or(PayloadError::Disk("truncated GPT header"))?
        .to_vec();
    copy[16..20].fill(0);
    if crc32(&copy) != le32(block, 16) {
        return Err(PayloadError::Disk("GPT header CRC mismatch"));
    }
    let info = HeaderInfo {
        disk_guid: guid_at(block, 56),
        alternate_lba: le64(block, 32),
        supported_header: le32(block, 8) == REVISION && le32(block, 20) == 0,
        first_usable: le64(block, 40),
        last_usable: le64(block, 48),
        entries_lba: le64(block, 72),
        entry_count: le32(block, 80),
        entry_bytes: le32(block, 84),
        entries_crc: le32(block, 88),
    };
    if le64(block, 24) != lba
        || info.entry_bytes < ENTRY_BYTES
        || !info.entry_bytes.is_power_of_two()
        || info.entry_count == 0
        || info.entry_array_bytes() > 1024 * 1024
        || info.entries_lba < 2
    {
        return Err(PayloadError::Disk("unsupported GPT header fields"));
    }
    if info.first_usable > info.last_usable {
        return Err(PayloadError::Disk("GPT usable range ends before it starts"));
    }
    Ok(Some(info))
}

/// Parse the entry array a header names, checking its CRC and that every
/// partition lies in the header's usable range, so none overlaps a GPT.
pub fn parse_entries(info: &HeaderInfo, bytes: &[u8]) -> Result<Table, PayloadError> {
    let length = info.entry_array_bytes() as usize;
    if bytes.len() < length || crc32(&bytes[..length]) != info.entries_crc {
        return Err(PayloadError::Disk("GPT entry array CRC mismatch"));
    }
    let mut partitions = Vec::new();
    for entry in bytes[..length].chunks(info.entry_bytes as usize) {
        let type_guid = guid_at(entry, 0);
        if type_guid.is_zero() {
            continue;
        }
        let units: Vec<u16> = entry[56..128]
            .chunks(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .take_while(|&u| u != 0)
            .collect();
        let partition = Partition {
            type_guid,
            unique_guid: guid_at(entry, 16),
            first_lba: le64(entry, 32),
            last_lba: le64(entry, 40),
            attributes: le64(entry, 48),
            name: String::from_utf16_lossy(&units),
        };
        if partition.last_lba < partition.first_lba {
            return Err(PayloadError::Disk("GPT entry ends before it starts"));
        }
        if partition.first_lba < info.first_usable || partition.last_lba > info.last_usable {
            return Err(PayloadError::Disk("GPT entry outside the usable range"));
        }
        partitions.push(partition);
    }
    Ok(Table {
        disk_guid: info.disk_guid,
        partitions,
    })
}

/// Sizes of the partitions Stela adds.
#[derive(Debug, Clone, Copy)]
pub struct Request {
    pub staging_bytes: u64,
    pub machine_state_bytes: u64,
    pub cache_bytes: u64,
    pub min_apple_bytes: u64,
}

/// A planned table, and whether it differs from what is on disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    pub table: Table,
    pub write: bool,
}

impl Plan {
    pub fn partition(&self, role: Role) -> Option<&Partition> {
        self.table
            .partitions
            .iter()
            .find(|partition| partition.role() == Some(role))
    }
}

fn align_up(value: u64, unit: u64) -> u64 {
    value.div_ceil(unit) * unit
}

/// Plan Stela's partitions. An existing table may hold only EFI system
/// partitions and Stela's own; any other partition is refused, so another
/// system's data is never at risk. When all four Stela partitions already
/// exist and staging is large enough, the table is kept as it is. A blank
/// disk (`existing` is `None`) gets a fresh table with a reserved, unwritten
/// gap for a later ESP. `entropy` yields 16 bytes for each new GUID.
pub fn plan(
    disk: Disk,
    existing: Option<&Table>,
    request: Request,
    entropy: impl FnMut() -> [u8; 16],
) -> Result<Plan, PayloadError> {
    plan_with_swap(disk, existing, request, None, entropy)
}

/// Plan the same install layout, reserving an optional swap partition when
/// building a fresh Apple region. An existing Apple region is never shrunk.
/// The setup caller is responsible for formatting the swap store itself.
pub fn plan_with_swap(
    disk: Disk,
    existing: Option<&Table>,
    request: Request,
    swap_bytes: Option<core::num::NonZeroU64>,
    mut entropy: impl FnMut() -> [u8; 16],
) -> Result<Plan, PayloadError> {
    disk.validate()?;
    let bytes = |blocks: u64| blocks * disk.block_bytes;
    let blocks = |bytes: u64| bytes / disk.block_bytes;
    let mut has_swap = false;
    if let Some(table) = existing {
        if table.partitions.iter().any(|p| p.role().is_none()) {
            return Err(PayloadError::Disk(
                "target disk has a partition that is neither an ESP nor Stela's",
            ));
        }
        let mut swap = table
            .partitions
            .iter()
            .filter(|p| p.role() == Some(Role::NorthwoodSwap));
        if let Some(partition) = swap.next() {
            if swap.next().is_some() {
                return Err(PayloadError::Disk("several Northwood Swap partitions"));
            }
            if swap_bytes
                .is_some_and(|size| partition.blocks() < size.get().div_ceil(disk.block_bytes))
            {
                return Err(PayloadError::Disk(
                    "existing Northwood Swap partition is too small",
                ));
            }
            has_swap = true;
        }
        if table
            .partitions
            .iter()
            .any(|p| p.role().is_some_and(|role| Role::STELA.contains(&role)))
        {
            let complete = Role::STELA.iter().all(|role| {
                table
                    .partitions
                    .iter()
                    .filter(|p| p.role() == Some(*role))
                    .count()
                    == 1
            });
            let staging = table
                .partitions
                .iter()
                .find(|p| p.role() == Some(Role::Staging));
            return match (complete, staging) {
                (true, Some(_)) if swap_bytes.is_some() && !has_swap => Err(PayloadError::Disk(
                    "existing Apple region must be kept; place swap in an unused GPT gap",
                )),
                (true, Some(staging)) if bytes(staging.blocks()) >= request.staging_bytes => {
                    Ok(Plan {
                        table: table.clone(),
                        write: false,
                    })
                }
                (true, Some(_)) => Err(PayloadError::Disk(
                    "existing staging partition is too small for this payload",
                )),
                _ => Err(PayloadError::Disk(
                    "target disk has an incomplete set of Stela partitions",
                )),
            };
        }
    }
    let alignment = blocks(ALIGNMENT).max(1);
    let mut next = match existing {
        Some(table) => table
            .partitions
            .iter()
            .map(|p| p.last_lba + 1)
            .max()
            .unwrap_or(disk.first_usable()),
        None => disk.first_usable() + blocks(RESERVED_ESP_BYTES),
    };
    let end = disk.last_usable() + 1;
    let mut table = match existing {
        Some(table) => table.clone(),
        None => Table {
            disk_guid: Guid::from_entropy(entropy()),
            partitions: Vec::new(),
        },
    };
    let fixed = [
        (Role::MachineState, request.machine_state_bytes),
        (Role::Staging, request.staging_bytes),
        (Role::Cache, request.cache_bytes),
    ];
    for (role, size) in fixed {
        let first = align_up(next, alignment);
        let count = align_up(blocks(align_up(size, disk.block_bytes)), alignment).max(alignment);
        if first + count > end {
            return Err(PayloadError::Disk("target disk is too small"));
        }
        table
            .partitions
            .push(new_partition(role, first, count, &mut entropy));
        next = first + count;
    }
    if let Some(size) = swap_bytes.filter(|_| !has_swap) {
        let first = align_up(next, alignment);
        let count = size
            .get()
            .div_ceil(ALIGNMENT)
            .checked_mul(alignment)
            .ok_or(PayloadError::Disk("Northwood Swap size overflows"))?;
        next = first
            .checked_add(count)
            .filter(|&after| after <= end)
            .ok_or(PayloadError::Disk(
                "target disk has no room for Northwood Swap",
            ))?;
        table.partitions.push(new_partition(
            Role::NorthwoodSwap,
            first,
            count,
            &mut entropy,
        ));
    }
    let first = align_up(next, alignment);
    let last = (end / alignment) * alignment;
    if last <= first || bytes(last - first) < request.min_apple_bytes {
        return Err(PayloadError::Disk(
            "target disk leaves too little space for the Apple region",
        ));
    }
    table.partitions.push(new_partition(
        Role::AppleRegion,
        first,
        last - first,
        &mut entropy,
    ));
    if table.partitions.len() > ENTRY_COUNT as usize {
        return Err(PayloadError::Disk("too many partitions"));
    }
    Ok(Plan { table, write: true })
}

fn new_partition(
    role: Role,
    first: u64,
    count: u64,
    entropy: &mut impl FnMut() -> [u8; 16],
) -> Partition {
    Partition {
        type_guid: role.type_guid(),
        unique_guid: Guid::from_entropy(entropy()),
        first_lba: first,
        last_lba: first + count - 1,
        attributes: 0,
        name: role.name().into(),
    }
}

/// One write of whole blocks at `lba`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Write {
    pub lba: u64,
    pub bytes: Vec<u8>,
}

/// Encode a table as the protective MBR, primary header and entries, and
/// the backup entries and header. The backup goes first in the list, then
/// the primary entries and header, then the MBR, so an interrupted write
/// never leaves a valid primary header that names unwritten entries.
pub fn encode(disk: Disk, table: &Table) -> Result<Vec<Write>, PayloadError> {
    disk.validate()?;
    if table.partitions.len() > ENTRY_COUNT as usize {
        return Err(PayloadError::Disk("too many partitions"));
    }
    let block = disk.block_bytes as usize;
    let mut entries = vec![0u8; ENTRY_ARRAY_BYTES as usize];
    for (slot, partition) in entries
        .chunks_mut(ENTRY_BYTES as usize)
        .zip(&table.partitions)
    {
        if partition.first_lba < disk.first_usable()
            || partition.last_lba > disk.last_usable()
            || partition.first_lba > partition.last_lba
        {
            return Err(PayloadError::Disk("partition outside the usable range"));
        }
        slot[0..16].copy_from_slice(&partition.type_guid.0);
        slot[16..32].copy_from_slice(&partition.unique_guid.0);
        slot[32..40].copy_from_slice(&partition.first_lba.to_le_bytes());
        slot[40..48].copy_from_slice(&partition.last_lba.to_le_bytes());
        slot[48..56].copy_from_slice(&partition.attributes.to_le_bytes());
        // At most 35 of the 36 UTF-16 units, so the name keeps the
        // terminating NUL UEFI specifies.
        for (unit, out) in partition
            .name
            .encode_utf16()
            .take(35)
            .zip(slot[56..].chunks_mut(2))
        {
            out.copy_from_slice(&unit.to_le_bytes());
        }
    }
    let entries_crc = crc32(&entries);
    let entry_blocks = disk.entry_blocks();
    let last = disk.blocks - 1;
    let header = |current: u64, backup: u64, entries_lba: u64| {
        let mut h = vec![0u8; block];
        h[0..8].copy_from_slice(SIGNATURE);
        h[8..12].copy_from_slice(&REVISION.to_le_bytes());
        h[12..16].copy_from_slice(&HEADER_BYTES.to_le_bytes());
        h[24..32].copy_from_slice(&current.to_le_bytes());
        h[32..40].copy_from_slice(&backup.to_le_bytes());
        h[40..48].copy_from_slice(&disk.first_usable().to_le_bytes());
        h[48..56].copy_from_slice(&disk.last_usable().to_le_bytes());
        h[56..72].copy_from_slice(&table.disk_guid.0);
        h[72..80].copy_from_slice(&entries_lba.to_le_bytes());
        h[80..84].copy_from_slice(&ENTRY_COUNT.to_le_bytes());
        h[84..88].copy_from_slice(&ENTRY_BYTES.to_le_bytes());
        h[88..92].copy_from_slice(&entries_crc.to_le_bytes());
        let crc = crc32(&h[..HEADER_BYTES as usize]);
        h[16..20].copy_from_slice(&crc.to_le_bytes());
        h
    };
    let mut padded = entries;
    padded.resize((entry_blocks * disk.block_bytes) as usize, 0);
    let mut mbr = vec![0u8; block];
    let protective_blocks = (disk.blocks - 1).min(u32::MAX as u64) as u32;
    // One partition of type 0xEE covering the disk, as UEFI specifies.
    mbr[446 + 1..446 + 4].copy_from_slice(&[0x00, 0x02, 0x00]);
    mbr[446 + 4] = 0xee;
    mbr[446 + 5..446 + 8].copy_from_slice(&[0xff, 0xff, 0xff]);
    mbr[446 + 8..446 + 12].copy_from_slice(&1u32.to_le_bytes());
    mbr[446 + 12..446 + 16].copy_from_slice(&protective_blocks.to_le_bytes());
    mbr[510] = 0x55;
    mbr[511] = 0xaa;
    Ok(vec![
        Write {
            lba: last - entry_blocks,
            bytes: padded.clone(),
        },
        Write {
            lba: last,
            bytes: header(last, 1, last - entry_blocks),
        },
        Write {
            lba: 2,
            bytes: padded,
        },
        Write {
            lba: 1,
            bytes: header(1, last, 2),
        },
        Write { lba: 0, bytes: mbr },
    ])
}

/// Blocks at each end of a disk that a GPT occupies at 512-byte blocks (the
/// MBR or backup header, and a header and the 16 KiB entry array). A disk
/// without a GPT is blank only when they are zero, so another system's MBR
/// or filesystem is never overwritten.
const BLANK_SPAN: u64 = 34;

/// What [`read`] found on a disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Found {
    /// No GPT, and the blocks a GPT occupies at either end are zero.
    Blank,
    /// No GPT header, and the only bytes at either end are a backup entry
    /// array holding nothing but ESP and Stela entries: [`encode`]'s first
    /// write to a blank disk, interrupted before the backup header. Nothing
    /// was committed, so the disk is staged as blank.
    Interrupted,
    /// A table: the primary GPT's, or the backup's when the primary is
    /// damaged or unwritten. `repair` names why [`encode`]'s writes must be
    /// made again even when the plan keeps the table.
    Table {
        table: Table,
        repair: Option<&'static str>,
    },
}

/// One copy of the GPT.
enum Side {
    Absent,
    Damaged(PayloadError),
    Valid(Table, HeaderInfo),
}

fn copy_at(
    disk: Disk,
    lba: u64,
    read: &mut impl FnMut(u64, u64) -> Result<Vec<u8>, PayloadError>,
) -> Result<Side, PayloadError> {
    let header = read(lba, 1)?;
    let info = match parse_header(disk, lba, &header) {
        Ok(Some(info)) => info,
        Ok(None) => return Ok(Side::Absent),
        Err(error) => return Ok(Side::Damaged(error)),
    };
    let blocks = info.entry_array_bytes().div_ceil(disk.block_bytes);
    let Some(entries_end) = info.entries_lba.checked_add(blocks) else {
        return Ok(Side::Damaged(PayloadError::Disk(
            "GPT entry array outside the disk",
        )));
    };
    if entries_end > disk.blocks {
        return Ok(Side::Damaged(PayloadError::Disk(
            "GPT entry array outside the disk",
        )));
    }
    let usable = info.first_usable..=info.last_usable;
    if info.first_usable <= 1
        || info.last_usable >= disk.blocks - 1
        || usable.contains(&lba)
        || (info.entries_lba <= info.last_usable && entries_end > info.first_usable)
    {
        return Ok(Side::Damaged(PayloadError::Disk(
            "GPT usable range overlaps disk metadata",
        )));
    }
    let entries = read(info.entries_lba, blocks)?;
    Ok(match parse_entries(&info, &entries) {
        Ok(table) => Side::Valid(table, info),
        Err(error) => Side::Damaged(error),
    })
}

/// A protective (or hybrid) MBR: the boot signature and a type 0xEE entry.
fn protective(block: &[u8]) -> bool {
    block.len() >= 512
        && block[510..512] == [0x55, 0xaa]
        && (0..4).any(|n| block[446 + 16 * n + 4] == 0xee)
}

/// Admit writes within an existing partition only when the primary and
/// backup agree and keep their entry arrays on opposite sides of the
/// usable range. A protective MBR must cover the disk without any
/// additional partition entries.
pub fn read_for_write(
    disk: Disk,
    mut read: impl FnMut(u64, u64) -> Result<Vec<u8>, PayloadError>,
) -> Result<Table, PayloadError> {
    disk.validate()?;
    let last = disk.blocks - 1;
    let Side::Valid(table, primary) = copy_at(disk, 1, &mut read)? else {
        return Err(PayloadError::Disk("invalid primary GPT"));
    };
    let Side::Valid(copy, backup) = copy_at(disk, last, &mut read)? else {
        return Err(PayloadError::Disk("invalid backup GPT"));
    };
    if !primary.supported_header
        || !backup.supported_header
        || primary.first_usable < disk.first_usable()
        || primary.last_usable > disk.last_usable()
    {
        return Err(PayloadError::Disk(
            "unsupported GPT header fields for writing",
        ));
    }
    if table != copy
        || primary.alternate_lba != last
        || backup.alternate_lba != 1
        || primary.first_usable != backup.first_usable
        || primary.last_usable != backup.last_usable
        || primary.entry_count != backup.entry_count
        || primary.entry_bytes != backup.entry_bytes
        || primary.entries_crc != backup.entries_crc
        || primary.entries_lba + primary.entry_array_bytes().div_ceil(disk.block_bytes)
            > primary.first_usable
        || backup.entries_lba <= backup.last_usable
        || backup.entries_lba + backup.entry_array_bytes().div_ceil(disk.block_bytes) > last
    {
        return Err(PayloadError::Disk("the primary and backup GPTs disagree"));
    }
    let mbr = read(0, 1)?;
    if !protective(&mbr) {
        return Err(PayloadError::Disk("the protective MBR is missing"));
    }
    let protected = last.min(u64::from(u32::MAX)) as u32;
    let mut entries = mbr[446..510]
        .chunks_exact(16)
        .filter(|entry| entry.iter().any(|&byte| byte != 0));
    let protection = entries.next().is_some_and(|entry| {
        entry[0] == 0
            && entry[4] == 0xee
            && entry[8..12] == 1u32.to_le_bytes()
            && entry[12..16] == protected.to_le_bytes()
    });
    if !protection || entries.next().is_some() {
        return Err(PayloadError::Disk("hybrid or incomplete protective MBR"));
    }
    Ok(table)
}

/// Every 128-byte entry is zero or names an ESP or Stela role, and one is
/// not zero.
fn stela_entries(bytes: &[u8]) -> bool {
    let mut any = false;
    for entry in bytes.chunks(ENTRY_BYTES as usize) {
        if entry.iter().all(|&byte| byte == 0) {
            continue;
        }
        if entry.len() < 16 || Role::of(guid_at(entry, 0)).is_none() {
            return false;
        }
        any = true;
    }
    any
}

/// Read a disk's GPT as UEFI does after an interrupted write: the primary
/// when it and its entries are valid, else the backup, which must then be
/// written back. [`encode`] writes the backup first and the primary header
/// last but for the MBR, so the primary header is the commit point:
///
/// - before it, a valid primary is the earlier table (the plan is made
///   again), and a damaged or missing primary means the backup holds the
///   new table in full;
/// - after it, only the protective MBR can be missing.
///
/// A disk without either header must be blank, or hold only [`encode`]'s
/// first write ([`Found::Interrupted`]); anything else is refused. `read`
/// returns `blocks` whole blocks from `lba`; its errors pass through.
pub fn read(
    disk: Disk,
    mut read: impl FnMut(u64, u64) -> Result<Vec<u8>, PayloadError>,
) -> Result<Found, PayloadError> {
    disk.validate()?;
    let last = disk.blocks - 1;
    let primary = copy_at(disk, 1, &mut read)?;
    let backup = copy_at(disk, last, &mut read)?;
    match (primary, backup) {
        (Side::Valid(table, _), backup) => {
            let repair = match backup {
                Side::Valid(copy, _) if copy == table => None,
                Side::Valid(..) => Some("the backup GPT differs from the primary"),
                Side::Absent => Some("the backup GPT is missing"),
                Side::Damaged(_) => Some("the backup GPT is damaged"),
            };
            let repair = match repair {
                None if !protective(&read(0, 1)?) => Some("the protective MBR is missing"),
                repair => repair,
            };
            Ok(Found::Table { table, repair })
        }
        (primary, Side::Valid(table, _)) => Ok(Found::Table {
            table,
            repair: Some(match primary {
                Side::Absent => "the primary GPT is missing",
                _ => "the primary GPT is damaged",
            }),
        }),
        (Side::Damaged(error), _) | (Side::Absent, Side::Damaged(error)) => Err(error),
        (Side::Absent, Side::Absent) => {
            let span = BLANK_SPAN.min(disk.blocks);
            let head = read(0, span)?;
            let tail_lba = disk.blocks - span;
            let tail = read(tail_lba, span)?;
            if head.iter().chain(&tail).all(|&byte| byte == 0) {
                return Ok(Found::Blank);
            }
            // The backup entry array sits just below the backup header.
            let block = disk.block_bytes as usize;
            let entries = ((last - disk.entry_blocks() - tail_lba) as usize * block)
                ..((last - tail_lba) as usize * block);
            let outside = tail[..entries.start].iter().chain(&tail[entries.end..]);
            if head.iter().chain(outside).all(|&byte| byte == 0) && stela_entries(&tail[entries]) {
                return Ok(Found::Interrupted);
            }
            Err(PayloadError::Disk("the disk has no GPT and is not blank"))
        }
    }
}

#[cfg(test)]
mod tests;
