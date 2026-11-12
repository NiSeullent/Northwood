//! LLB's boot choice, before boot services end: with `--preboot`,
//! `stela.efi` reads the persistent NOR's current NVRAM bank (`--nor
//! SERIAL:GUID`) through the firmware's block I/O, chooses macOS or
//! recoveryOS as `stela_loader::boot_selection` does, removes the one-shot
//! variables that boot consumes (saving the next bank, as LLB saves NVRAM
//! before it boots) and hands the engine the options for that boot. A macOS
//! boot whose Preboot disk holds no installed system boots System Recovery
//! instead (missing boot media), as a Mac with an erased disk does. A
//! recoveryOS boot takes the disk's own copy, the paired recoveryOS of the
//! Recovery volume and then System Recovery of the RecoveryOSContainer
//! (`stela_loader::recovery::find`), which the engine builds from the disk.
//! A disk that holds neither boots nothing ([`Refusal::NoRecovery`]): Stela
//! has no recoveryOS of its own, and only Install macOS restores the disk's.
//! NVRAM that holds the machine in iBoot's recovery mode, as an interrupted
//! restore leaves it, boots nothing ([`Refusal::RecoveryMode`]).
//!
//! Each partition is found as the engine finds it: exactly one whole disk
//! reports the serial, and exactly one of its GPT partitions has the GUID;
//! an EFI system partition is refused. A blank NOR partition holds no
//! variables yet: the engine initializes it. When `--nor` names the
//! `--storage` partition, the NOR is the start of that partition's tail
//! (`pc::machine_state`), and the Preboot disk is the rest, as the engine
//! splits it.
use crate::Result;
use crate::{
    output,
    payload::{Aligned, device_path, disk_serial},
};
use alloc::string::ToString;
use alloc::{format, string::String, vec, vec::Vec};
use stela_boot_core::chosen_disk::{self, Search, Stale};
use stela_contracts::refusal::Within;
use stela_contracts::refuse;
use stela_loader::{
    boot_selection::{self, BootVolume, Selection},
    mount_interface::ChoiceKind,
    recovery,
};
use uefi::{
    Handle,
    boot::{self, OpenProtocolAttributes, OpenProtocolParams, ScopedProtocol},
    proto::media::block::BlockIO,
};
use {
    stela_formats::apfs, stela_formats::apfs::ApfsError, stela_formats::apfs::Container,
    stela_formats::payload::PayloadError, stela_formats::payload::ReadAt,
    stela_formats::payload::gpt, stela_formats::payload::gpt::Disk,
    stela_formats::payload::gpt::Role, stela_formats::preboot,
    stela_formats::preboot::PrebootError, stela_runtime::pc::failure_record,
    stela_runtime::pc::machine_state, stela_runtime::pc::storage::Target,
    stela_soc_data::nor::CAPACITY, stela_soc_data::nor::Image, stela_soc_data::nor::SECTOR_SIZE,
    stela_soc_data::nor::nvram::Variables, stela_soc_data::nor::nvram::empty_bank,
};

/// One GPT partition, through its whole disk's block I/O.
struct Partition {
    io: ScopedProtocol<BlockIO>,
    media: u32,
    block_bytes: u64,
    first_lba: u64,
    bytes: u64,
}

/// Which part of a partition that also holds the machine state in its tail.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Part {
    Whole,
    /// The guest's namespace before the tail.
    Namespace,
    /// The machine state in the tail.
    State,
}

impl Partition {
    /// Confine the partition to `part`, as the engine splits it.
    fn confine(&mut self, part: Part) -> Result<()> {
        if part == Part::Whole {
            return Ok(());
        }
        let layout = machine_state::split(self.bytes)
            .map_err(|_| refuse!("the partition is too small to also hold the machine state"))?;
        if !machine_state::ALIGNMENT.is_multiple_of(self.block_bytes) {
            return Err(refuse!(
                "the partition's blocks do not divide the machine state's alignment"
            ));
        }
        if part == Part::State {
            self.first_lba += layout.namespace_bytes / self.block_bytes;
            self.bytes = layout.state_bytes;
        } else {
            self.bytes = layout.namespace_bytes;
        }
        Ok(())
    }

    fn read_blocks(&mut self, lba: u64, buffer: &mut [u8]) -> Result<()> {
        let mut aligned = Aligned::new(buffer.len());
        self.io
            .read_blocks(self.media, self.first_lba + lba, aligned.bytes())
            .map_err(|_| refuse!("cannot read the partition"))?;
        buffer.copy_from_slice(aligned.bytes());
        Ok(())
    }

    fn write_sector(&mut self, offset: usize, sector: &[u8]) -> Result<()> {
        let mut buffer = Aligned::new(SECTOR_SIZE);
        buffer.bytes().copy_from_slice(sector);
        let lba = self.first_lba + offset as u64 / self.block_bytes;
        self.io
            .write_blocks(self.media, lba, buffer.bytes())
            .map_err(|_| refuse!("cannot write the NOR partition"))
    }
}

/// The largest firmware read, so a large APFS read needs no huge buffer.
const TRANSFER_BYTES: u64 = 1 << 20;

impl ReadAt for Partition {
    type Error = stela_contracts::Refusal;

    fn len(&self) -> u64 {
        self.bytes
    }

    fn read_at(&mut self, offset: u64, buffer: &mut [u8]) -> Result<()> {
        let end = offset
            .checked_add(buffer.len() as u64)
            .filter(|&end| end <= self.bytes)
            .ok_or(refuse!("read past the partition"))?;
        let mut at = offset;
        while at < end {
            let first = at / self.block_bytes;
            let span = (end - at).min(TRANSFER_BYTES);
            let last = (at + span).div_ceil(self.block_bytes);
            let mut blocks = vec![0u8; ((last - first) * self.block_bytes) as usize];
            self.read_blocks(first, &mut blocks)?;
            let within = (at - first * self.block_bytes) as usize;
            let done = (at - offset) as usize;
            buffer[done..done + span as usize]
                .copy_from_slice(&blocks[within..within + span as usize]);
            at += span;
        }
        Ok(())
    }
}

/// Open `handle`'s block I/O for reading and, on the NOR partition only,
/// writing. Not exclusive: the disk's other partitions, such as the ESP
/// this image was loaded from, stay connected.
fn block_io(handle: Handle) -> Option<ScopedProtocol<BlockIO>> {
    // SAFETY: Stela reads the named partitions and writes only the NOR's.
    unsafe {
        boot::open_protocol::<BlockIO>(
            OpenProtocolParams {
                handle,
                agent: boot::image_handle(),
                controller: None,
            },
            OpenProtocolAttributes::GetProtocol,
        )
    }
    .ok()
}

/// A whole disk's serial, when it reports one, and its block I/O.
type WholeDisk = (Option<String>, ScopedProtocol<BlockIO>);

/// Every whole disk with media.
fn whole_disks() -> Result<Vec<WholeDisk>> {
    let handles = boot::find_handles::<BlockIO>().map_err(|_| refuse!("no block devices"))?;
    let mut disks = Vec::new();
    for handle in handles {
        let Some(io) = block_io(handle) else {
            continue;
        };
        let media = io.media();
        if media.is_logical_partition() || !media.is_media_present() {
            continue;
        }
        let path = device_path(handle).unwrap_or_default();
        disks.push((disk_serial(handle, &path), io));
    }
    Ok(disks)
}

/// The whole disk behind `io` and its GPT.
fn table(io: ScopedProtocol<BlockIO>) -> Result<(Partition, gpt::Table)> {
    let media = io.media();
    let disk = Disk {
        block_bytes: media.block_size() as u64,
        blocks: media.last_block() + 1,
    };
    let media = media.media_id();
    let mut whole = Partition {
        io,
        media,
        block_bytes: disk.block_bytes,
        first_lba: 0,
        bytes: disk.blocks * disk.block_bytes,
    };
    // The backup table stands in for a primary an interrupted write left
    // damaged (`gpt::read`); selecting a boot only reads, so the repair is
    // left to the next staging run.
    let found = gpt::read(disk, |lba, blocks| {
        let mut bytes = vec![0u8; (blocks * disk.block_bytes) as usize];
        whole
            .read_blocks(lba, &mut bytes)
            .map_err(|error| PayloadError::Io(error.to_string()))?;
        Ok(bytes)
    })
    .map_err(|_| refuse!("the disk's GPT is invalid"))?;
    let gpt::Found::Table { table, .. } = found else {
        return Err(refuse!("the disk has no valid GPT"));
    };
    Ok((whole, table))
}

/// The partition `target` names, as the engine finds it, or what was found
/// instead (`chosen_disk::Search`). Fails only when the disks cannot be
/// listed.
fn search(target: &Target) -> Result<core::result::Result<Partition, Search>> {
    let mut disks: Vec<_> = whole_disks()?
        .into_iter()
        .filter(|(serial, _)| serial.as_deref() == Some(target.serial.as_str()))
        .collect();
    let io = match disks.len() {
        0 => return Ok(Err(Search::NoDisk)),
        1 => disks.remove(0).1,
        _ => return Ok(Err(Search::SeveralDisks)),
    };
    // A table that cannot be read is a blank or wiped disk to a boot.
    let Ok((mut whole, table)) = table(io) else {
        return Ok(Err(Search::NoTable));
    };
    let matching: Vec<_> = table
        .partitions
        .iter()
        .filter(|p| p.unique_guid.0 == target.partition.0)
        .collect();
    let partition = match matching[..] {
        [partition] => partition,
        [] => return Ok(Err(Search::NoPartition)),
        _ => return Ok(Err(Search::SeveralPartitions)),
    };
    if partition.role() == Some(Role::Esp) {
        return Ok(Err(Search::EspPartition));
    }
    whole.first_lba = partition.first_lba;
    whole.bytes = partition.blocks() * whole.block_bytes;
    Ok(Ok(whole))
}

fn find(target: &Target) -> Result<Partition> {
    search(target)?.map_err(|found| match found {
        Search::NoDisk | Search::SeveralDisks => refuse!("not exactly one disk has the serial"),
        Search::NoTable => refuse!("the disk has no valid GPT"),
        Search::NoPartition | Search::SeveralPartitions => {
            refuse!("not exactly one partition has the GUID")
        }
        Search::EspPartition => refuse!("the partition is an EFI system partition"),
        Search::Found => refuse!("the partition was found"),
    })
}

/// Copy into the log, once, the record a previous start left in the machine
/// state of the partition `options` name for it (`--storage` and `--nor`
/// alike) when the USB's own log was out of reach of Stela's drivers
/// (`failure_record`), then clear it.
pub fn carry_failure_record(options: &[u8]) {
    let Ok(text) = core::str::from_utf8(options) else {
        return;
    };
    let storage = option(text, "--storage");
    let Some(target) = storage
        .filter(|_| option(text, "--nor") == storage)
        .and_then(|name| name.parse::<Target>().ok())
    else {
        return;
    };
    let Ok(Ok(mut partition)) = search(&target) else {
        return;
    };
    if partition.confine(Part::State).is_err() {
        return;
    }
    let mut region = vec![0u8; failure_record::BYTES as usize];
    if partition
        .read_at(failure_record::OFFSET, &mut region)
        .is_err()
    {
        return;
    }
    let Some(record) = failure_record::decode(&region) else {
        return;
    };
    output::arguments(format_args!(
        "PREVIOUS BOOT failed where the USB's log was out of reach; the chosen disk kept:\n"
    ));
    for line in record.lines() {
        output::arguments(format_args!("PREVIOUS BOOT {line}\n"));
    }
    // A zero header clears the record.
    let block = partition.block_bytes;
    let mut zeros = Aligned::new(block as usize);
    let cleared = partition
        .io
        .write_blocks(
            partition.media,
            partition.first_lba + failure_record::OFFSET / block,
            zeros.bytes(),
        )
        .and_then(|()| partition.io.flush_blocks());
    if cleared.is_err() {
        output::arguments(format_args!(
            "PREVIOUS BOOT record not cleared: the chosen disk refused the write\n"
        ));
    }
}

/// Whether the partition Install macOS chose before (`SERIAL:GUID`) is on
/// this PC: exactly one disk with its serial, and exactly one partition
/// with its GUID that is no EFI system partition.
pub fn chosen_present(disk: &str) -> bool {
    disk.parse::<Target>()
        .is_ok_and(|target| matches!(search(&target), Ok(Ok(_))))
}

/// Why the partition `target` names is no longer there to boot, when it is
/// not (`chosen_disk::Stale::of`).
fn stale(target: &Target) -> Result<Option<Stale>> {
    Ok(search(target)?.err().and_then(Stale::of))
}

/// The external disk's partition NVRAM's boot volume names, as the
/// `--preboot` target that boots it: a whole disk's own GPT partition, as
/// recoveryOS installs macOS on a Mac's external drive. `None` when no
/// disk's table holds it, as for the internal SSD's containers, which lie in
/// the Apple region's own table.
fn external(volume: &BootVolume) -> Result<Option<Target>> {
    let mut found = None;
    for (serial, io) in whole_disks()? {
        let Ok((_, table)) = table(io) else {
            continue;
        };
        if !table
            .partitions
            .iter()
            .any(|p| p.unique_guid.0 == volume.partition)
        {
            continue;
        }
        let serial = serial.ok_or(refuse!(
            "the boot volume's disk reports no serial number to find it by after boot"
        ))?;
        if found.is_some() {
            return Err(refuse!("several disks hold the boot volume's partition"));
        }
        found = Some(Target {
            serial,
            partition: stela_runtime::pc::gpt::Guid(volume.partition),
        });
    }
    Ok(found)
}

/// [`select`]; a store that cannot be read on a Preboot disk without a Mac
/// system container is an install that did not finish
/// (`chosen_disk::unreadable_store`), and any other stops the boot with
/// what to do.
fn select_or_unprepared(
    target: &Target,
    part: Part,
    consume: bool,
    (preboot, preboot_part): &(Target, Part),
) -> Result<Selection, Refusal> {
    let error = match select(target, part, consume) {
        Ok(selection) => return Ok(selection),
        Err(error) => error,
    };
    output::arguments(format_args!("NVRAM boot refusal: {error}\n"));
    match chosen_disk::unreadable_store(blank(preboot, *preboot_part).ok()) {
        chosen_disk::Unreadable::Unprepared => Err(Refusal::Stale(Stale::NoMacOs)),
        chosen_disk::Unreadable::Fatal => Err(Refusal::Other(chosen_disk::UNREADABLE_STORE)),
    }
}

/// The boot NVRAM chooses, with its one-shot variables consumed and saved
/// when `consume` holds.
fn select(target: &Target, part: Part, consume: bool) -> Result<Selection> {
    let mut nor = find(target).within(|| refuse!("NOR partition"))?;
    nor.confine(part).within(|| refuse!("NOR partition"))?;
    if nor.io.media().is_read_only()
        || !SECTOR_SIZE.is_multiple_of(nor.block_bytes as usize)
        || nor.bytes < CAPACITY as u64
    {
        return Err(refuse!(
            "the NOR partition is read-only, smaller than the image or has odd blocks"
        ));
    }
    let mut bytes = vec![0u8; CAPACITY];
    nor.read_blocks(0, &mut bytes)?;
    let select = |variables: &Variables| Ok(boot_selection::select(variables)?);
    if bytes.iter().all(|&b| b == 0) {
        return select(&Variables::parse(&empty_bank())?);
    }
    let mut image = Image::from_bytes(bytes)?;
    let (_, proxy) = image.boot_bank()?;
    let mut variables = Variables::parse(&proxy)?;
    let selection = select(&variables)?;
    if consume && boot_selection::consume(&mut variables, &selection)? {
        let bank = variables.encode()?;
        let (written, sectors) = image.save_bank(&bank)?;
        for offset in sectors {
            let sector = image.bytes()[offset..offset + SECTOR_SIZE].to_vec();
            nor.write_sector(offset, &sector)?;
        }
        nor.io
            .flush_blocks()
            .map_err(|_| refuse!("cannot flush the NOR partition"))?;
        output::arguments(format_args!(
            "NVRAM consumed {} into bank {written}\n",
            selection
                .consumed
                .iter()
                .map(|name| core::str::from_utf8(name).unwrap_or("?"))
                .collect::<Vec<_>>()
                .join(",")
        ));
    }
    Ok(selection)
}

/// Whether the Preboot disk holds an installed system to boot: a container
/// (the boot volume's, when NVRAM names one) whose Preboot volume has a
/// boot directory. Only a missing container, Preboot volume or boot
/// directory counts as absent; any other failure is reported.
fn installed(target: &Target, part: Part, volume: Option<&BootVolume>) -> Result<bool> {
    let mut disk = find(target).within(|| refuse!("Preboot disk"))?;
    disk.confine(part).within(|| refuse!("Preboot disk"))?;
    let partitions = match apfs::container_partitions(&mut disk) {
        Ok(partitions) => partitions,
        Err(ApfsError::NotFound(_)) => return Ok(false),
        Err(error) => return Err(refuse!("Preboot disk: {error}", %error)),
    };
    for (offset, guid) in partitions {
        if volume.is_some_and(|v| guid.is_some_and(|g| g.0 != v.partition)) {
            continue;
        }
        let mut container = match Container::open(&mut disk, offset) {
            Ok(container) => container,
            Err(ApfsError::Malformed(_)) => continue,
            Err(error) => return Err(refuse!("container at {offset:#x}: {error}", offset, %error)),
        };
        match preboot::candidates(&mut container) {
            Ok(_) => return Ok(true),
            Err(PrebootError::NoBootDirectory | PrebootError::Apfs(ApfsError::NotFound(_))) => {}
            Err(error) => return Err(refuse!("container at {offset:#x}: {error}", offset, %error)),
        }
    }
    Ok(false)
}

/// Whether the Preboot disk holds a recoveryOS of `kind`
/// (`stela_loader::recovery::find`): a Recovery volume with a boot
/// directory. A disk that cannot be read for it is reported and counts as
/// holding none, so the boot falls back to the next recoveryOS.
fn holds_recovery(
    target: &Target,
    part: Part,
    kind: recovery::Kind,
    volume: Option<&BootVolume>,
) -> bool {
    let probe = || -> Result<bool> {
        let mut disk = find(target).within(|| refuse!("Preboot disk"))?;
        disk.confine(part).within(|| refuse!("Preboot disk"))?;
        recovery::find(&mut disk, kind, volume)
            .map(|found| !found.is_empty())
            .map_err(|error| refuse!("recovery volume: {error}", %error))
    };
    match probe() {
        Ok(found) => found,
        Err(error) => {
            output::arguments(format_args!(
                "RECOVERY probe of the {} refused: {error}\n",
                kind.label()
            ));
            false
        }
    }
}

/// Whether the internal SSD lacks the iBootSystemContainer every boot of a
/// Mac reads (`apfs::require_system_container`), as a blank one does.
fn blank(target: &Target, part: Part) -> Result<bool> {
    let mut disk = find(target).within(|| refuse!("internal SSD"))?;
    disk.confine(part).within(|| refuse!("internal SSD"))?;
    match apfs::require_system_container(&mut disk) {
        Ok(()) => Ok(false),
        Err(ApfsError::NotFound(_)) => Ok(true),
        Err(error) => Err(refuse!("internal SSD: {error}", %error)),
    }
}

/// The value after `name` in the options.
fn option<'a>(text: &'a str, name: &str) -> Option<&'a str> {
    let mut words = text.split_ascii_whitespace();
    words.find(|w| *w == name)?;
    words.next()
}

/// `text` with the value after `name` replaced by `value`.
fn replace(text: &str, name: &str, value: &str) -> String {
    let mut words: Vec<&str> = text.split_ascii_whitespace().collect();
    if let Some(at) = words.iter().position(|w| *w == name)
        && at + 1 < words.len()
    {
        words[at + 1] = value;
    }
    words.join(" ")
}

/// Whether a startup choice can start now ([`ready`]).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Readiness {
    /// It starts what it names: macOS from an installed system, or the
    /// disk's recoveryOS when NVRAM asks for it.
    Ready,
    /// A Mac choice whose startup disk holds no installed macOS, which
    /// would start the disk's System Recovery instead, or nothing without
    /// one.
    NotInstalled,
    /// Its SSD has no iBootSystemContainer yet, NVRAM holds the machine in
    /// recovery mode, as an unfinished restore leaves it, or NVRAM asks for
    /// a recoveryOS the disk does not hold: only Install macOS starts.
    Unprepared,
    /// The disk Install macOS chose is no longer there, or holds no macOS:
    /// only Install macOS starts, and it chooses again.
    Stale(Stale),
    /// The check could not be made; starting the choice reports why.
    Unknown,
}

/// Whether the startup choice `options` of `kind` can start now. Install
/// and other choices are not checked ([`Readiness::Unknown`]).
pub fn ready(options: &str, kind: ChoiceKind) -> Readiness {
    match kind {
        ChoiceKind::Mac => match route_checked(options.as_bytes().into(), false) {
            Ok((_, true)) => Readiness::Ready,
            Ok((_, false)) | Err(Refusal::NoRecovery { installed: false }) => {
                Readiness::NotInstalled
            }
            Err(
                Refusal::BlankDisk
                | Refusal::RecoveryMode
                | Refusal::NoRecovery { installed: true },
            ) => Readiness::Unprepared,
            Err(Refusal::Stale(stale)) => Readiness::Stale(stale),
            Err(Refusal::Other(_)) => Readiness::Unknown,
        },
        _ => Readiness::Unknown,
    }
}

/// Why no boot options came out of [`route`].
pub enum Refusal {
    /// NVRAM holds the machine in iBoot's recovery mode
    /// (`boot_selection::Target::RecoveryMode`): it waits for a restore
    /// host, Stela's Install macOS startup choice.
    RecoveryMode,
    /// The internal SSD has no iBootSystemContainer yet: only Install macOS
    /// boots.
    BlankDisk,
    /// NVRAM boots recoveryOS, or the boot disk holds no installed system
    /// (`installed` false, missing boot media), and the disk holds no
    /// recoveryOS to boot: only Install macOS restores one.
    NoRecovery {
        installed: bool,
    },
    /// The disk or partition the options name is no longer there, or holds
    /// no macOS and no readable NVRAM: Install macOS chooses again.
    Stale(Stale),
    Other(&'static str),
}

/// The engine's options for this boot (`boot_selection::route`), with
/// NVRAM's one-shot choices consumed.
pub fn route(options: Vec<u8>) -> Result<Vec<u8>, Refusal> {
    route_with(options, true)
}

/// Where [`route`] would send `options`, leaving NVRAM unchanged: the
/// check before the startup menu, which a chosen boot routes again.
pub fn peek(options: Vec<u8>) -> Result<Vec<u8>, Refusal> {
    route_with(options, false)
}

fn route_with(options: Vec<u8>, consume: bool) -> Result<Vec<u8>, Refusal> {
    route_checked(options, consume).map(|(routed, _)| routed)
}

/// [`route_with`], and whether the boot disk holds an installed system: false
/// when a Mac boot falls back to recoveryOS for want of one.
fn route_checked(options: Vec<u8>, consume: bool) -> Result<(Vec<u8>, bool), Refusal> {
    let mut installed_system = true;
    let text =
        core::str::from_utf8(&options).map_err(|_| Refusal::Other("stela.cfg is not UTF-8"))?;
    let refuse = |error: stela_contracts::Refusal| {
        output::arguments(format_args!("NVRAM boot refusal: {error}\n"));
        Refusal::Other("NVRAM's boot choice could not be read or is not modeled")
    };
    let parse = |text: &str| text.parse::<Target>().map_err(Refusal::Other);
    let mut retargeted = None;
    let selection = match option(text, boot_selection::PREBOOT_OPTION) {
        Some(preboot) => {
            let nor = option(text, "--nor");
            // A disk or partition no longer there is a choice to make again,
            // not a store to read.
            let preboot_too = (nor != Some(preboot)).then_some(preboot);
            for name in nor.into_iter().chain(preboot_too) {
                if let Some(stale) = stale(&parse(name)?).map_err(refuse)? {
                    return Err(Refusal::Stale(stale));
                }
            }
            // The engine splits a `--storage` partition that `--nor` also
            // names; the Preboot disk is then its namespace.
            let target = |name: Option<&str>| name.and_then(|name| name.parse::<Target>().ok());
            let (nor_target, storage) = (target(nor), target(option(text, "--storage")));
            let in_storage = |name: &str| {
                nor_target.is_some() && nor_target == storage && target(Some(name)) == storage
            };
            let mut selection = match nor {
                Some(nor) => {
                    let part = if in_storage(nor) {
                        Part::State
                    } else {
                        Part::Whole
                    };
                    let preboot_part = if in_storage(preboot) {
                        Part::Namespace
                    } else {
                        Part::Whole
                    };
                    let preboot = (parse(preboot)?, preboot_part);
                    select_or_unprepared(&parse(nor)?, part, consume, &preboot)?
                }
                None => boot_selection::select(&Variables::parse(&empty_bank()).map_err(refuse)?)
                    .map_err(|error| refuse(error.into()))?,
            };
            // A boot volume on an external disk boots from that disk.
            let outside = match &selection.volume {
                Some(volume) => external(volume).map_err(refuse)?,
                None => None,
            };
            let (disk, part) = match outside {
                Some(target) => {
                    let value = format!("{}:{}", target.serial, target.partition);
                    output::arguments(format_args!("BOOT volume on external disk {value}\n"));
                    retargeted = Some(replace(text, boot_selection::PREBOOT_OPTION, &value));
                    (target, Part::Whole)
                }
                None if in_storage(preboot) => (parse(preboot)?, Part::Namespace),
                None => (parse(preboot)?, Part::Whole),
            };
            // An internal SSD without its iBootSystemContainer boots nothing,
            // recoveryOS included: it waits for the restore that creates it.
            if retargeted.is_none() && blank(&disk, part).map_err(refuse)? {
                return Err(Refusal::BlankDisk);
            }
            if matches!(selection.target, boot_selection::Target::MacOs { .. })
                && !installed(&disk, part, selection.volume.as_ref()).map_err(refuse)?
            {
                selection.missing_boot_media();
                installed_system = false;
            }
            // A recoveryOS boot takes the disk's own copy first, as LLB does.
            let volume = selection.volume;
            match selection.target {
                boot_selection::Target::PairedRecovery => {
                    selection.paired_on_disk =
                        holds_recovery(&disk, part, recovery::Kind::Paired, volume.as_ref());
                    selection.system_on_disk = !selection.paired_on_disk
                        && holds_recovery(&disk, part, recovery::Kind::System, volume.as_ref());
                }
                boot_selection::Target::SystemRecovery => {
                    selection.system_on_disk =
                        holds_recovery(&disk, part, recovery::Kind::System, volume.as_ref());
                }
                _ => {}
            }
            let wants_recovery = matches!(
                selection.target,
                boot_selection::Target::PairedRecovery | boot_selection::Target::SystemRecovery
            );
            if wants_recovery && !selection.paired_on_disk && !selection.system_on_disk {
                if let Err(error) = boot_selection::route(text, Some(&selection)) {
                    output::arguments(format_args!("BOOT {error:#}\n"));
                }
                return Err(Refusal::NoRecovery {
                    installed: installed_system,
                });
            }
            Some(selection)
        }
        None => None,
    };
    let text = retargeted.as_deref().unwrap_or(text);
    if let Some(held) = selection
        .as_ref()
        .filter(|s| s.target == boot_selection::Target::RecoveryMode)
    {
        if let Err(error) = boot_selection::route(text, Some(held)) {
            output::arguments(format_args!("BOOT {error:#}\n"));
        }
        return Err(Refusal::RecoveryMode);
    }
    let (routed, log) = boot_selection::route(text, selection.as_ref()).map_err(|error| {
        output::arguments(format_args!("NVRAM boot refusal: {error:#}\n"));
        Refusal::Other("stela.cfg cannot boot what NVRAM chose")
    })?;
    // Only an installed-system boot has a choice to report, and only the
    // boot that uses it: a `peek` before the menu would repeat the line.
    if consume && selection.is_some() {
        output::arguments(format_args!("BOOT {log}\n"));
    }
    Ok((routed.into_bytes(), installed_system))
}
