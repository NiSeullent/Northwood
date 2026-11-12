//! Install mode: find the restore payload on any volume, verify it, and
//! stage the Erase subset on the target disk, all before boot services end.
//!
//! `stela.cfg` turns this on with `--install-mode verify` or
//! `--install-mode stage`, plus `--install-build BUILD`. Staging writes only
//! the disk `stela.cfg` names: by its serial number (`--install-target-serial`,
//! read from NVMe Identify Controller or ATA IDENTIFY data), by its GPT disk
//! GUID (`--install-target-guid`), or by its device path and exact size
//! (`--install-target-path`, `--install-target-bytes`). The disk that holds
//! the payload is never a target. Without `--install-mode` the
//! options pass through to the mount unchanged.
//!
//! A stage that stops part way through is run again: its GPT writes resume
//! from what `gpt::read` finds (the primary header commits the table), and
//! the staging header is cleared before the first member byte and written
//! after the last, so a partial copy never reads as staged and the members
//! are copied again.
use crate::Refusal;
use crate::Result;
use crate::output;
use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use stela_contracts::refuse;
use stela_crypto::Sha256;
use stela_crypto::digest::Digest;
use uefi::{
    CString16, Handle,
    boot::{self, OpenProtocolAttributes, OpenProtocolParams, ScopedProtocol},
    proto::{
        device_path::{
            DevicePath,
            text::{AllowShortcuts, DisplayOnly},
        },
        media::{
            block::BlockIO,
            file::{Directory, File, FileAttribute, FileInfo, FileMode, RegularFile},
            fs::SimpleFileSystem,
        },
    },
};
use {
    stela_formats::payload::CopyError, stela_formats::payload::Layout,
    stela_formats::payload::MANIFEST_FILE, stela_formats::payload::MemberRead,
    stela_formats::payload::PayloadError, stela_formats::payload::PayloadManifest,
    stela_formats::payload::PayloadSource, stela_formats::payload::ReadAt,
    stela_formats::payload::copy_verified, stela_formats::payload::gpt,
    stela_formats::payload::gpt::Disk, stela_formats::payload::gpt::Guid,
    stela_formats::payload::gpt::Request, stela_formats::payload::gpt::Role,
    stela_formats::payload::images_directory, stela_formats::payload::slice_count,
    stela_formats::payload::slice_path, stela_formats::payload::staging,
    stela_formats::payload::staging::Header, stela_formats::payload::zip::Archive,
    stela_formats::payload::zip::MemberStream,
};

/// Exit codes after an install-mode run; the mount itself does not start.
const VERIFIED: u32 = 0x12;
const STAGED: u32 = 0x13;
const COPY_BYTES: usize = 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Verify,
    Stage,
}

#[derive(Debug, Default)]
struct Install {
    mode: Option<Mode>,
    build: Option<String>,
    target_guid: Option<Guid>,
    target_serial: Option<String>,
    target_path: Option<String>,
    target_bytes: Option<u64>,
    cache_bytes: Option<u64>,
    min_apple_bytes: Option<u64>,
}

fn number(value: &str) -> Result<u64> {
    match value.strip_prefix("0x") {
        Some(hex) => u64::from_str_radix(hex, 16),
        None => value.parse(),
    }
    .map_err(|_| refuse!("invalid install option number"))
}

/// Read the install options in `stela.cfg`, skipping the mount's own.
fn parse(options: &str) -> Result<Install> {
    let mut install = Install::default();
    let mut words = options.split_ascii_whitespace();
    while let Some(name) = words.next() {
        if !name.starts_with("--install-") {
            words.next();
            continue;
        }
        let value = words
            .next()
            .ok_or(refuse!("install option without a value"))?;
        match name {
            "--install-mode" => {
                install.mode = Some(match value {
                    "verify" => Mode::Verify,
                    "stage" => Mode::Stage,
                    _ => return Err(refuse!("--install-mode must be verify or stage")),
                })
            }
            "--install-build" => install.build = Some(value.into()),
            "--install-target-guid" => {
                install.target_guid =
                    Some(Guid::parse(value).ok_or(refuse!("--install-target-guid is not a GUID"))?)
            }
            "--install-target-serial" => install.target_serial = Some(value.into()),
            "--install-target-path" => install.target_path = Some(value.into()),
            "--install-target-bytes" => install.target_bytes = Some(number(value)?),
            "--install-cache-bytes" => install.cache_bytes = Some(number(value)?),
            "--install-min-apple-bytes" => install.min_apple_bytes = Some(number(value)?),
            _ => return Err(refuse!("unknown install option")),
        }
    }
    let other = install.build.is_some()
        || install.target_guid.is_some()
        || install.target_serial.is_some()
        || install.target_path.is_some()
        || install.target_bytes.is_some()
        || install.cache_bytes.is_some()
        || install.min_apple_bytes.is_some();
    if install.mode.is_none() && other {
        return Err(refuse!("install options need --install-mode"));
    }
    Ok(install)
}

/// Run install mode if `stela.cfg` asks for it, which ends in
/// [`output::terminate`]. Otherwise return the mount's own options.
pub fn run(options: Vec<u8>) -> Result<Vec<u8>> {
    let text = core::str::from_utf8(&options).map_err(|_| refuse!("stela.cfg is not UTF-8"))?;
    let install = parse(text)?;
    let Some(mode) = install.mode else {
        return Ok(options);
    };
    let build = install
        .build
        .as_deref()
        .ok_or(refuse!("install mode needs --install-build"))?;
    // Verifying tens of gigabytes outlasts the firmware's 5-minute watchdog.
    boot::set_watchdog_timer(0, 0x10000, None).map_err(|_| refuse!("cannot stop the watchdog"))?;
    let (mut source, payload_path) = Source::find(build)?;
    output::arguments(format_args!(
        "PAYLOAD build={} model={} layout={:?} members={} bytes={} volume={payload_path}\n",
        source.manifest.product_build,
        source.manifest.model,
        source.manifest.layout,
        source.manifest.members.len(),
        source.manifest.total_bytes()
    ));
    let subset = source.manifest.erase_subset().map_err(report)?;
    match mode {
        Mode::Verify => {
            let mut buffer = vec![0u8; COPY_BYTES];
            for member in &subset.members {
                copy_verified(&mut source, member, &mut buffer, |_| Ok::<(), ()>(()))
                    .map_err(copy_report)?;
                output::arguments(format_args!(
                    "VERIFIED {} bytes={}\n",
                    member.name, member.bytes
                ));
            }
            output::arguments(format_args!(
                "STELA PAYLOAD VERIFIED members={} bytes={}; nothing was written\n",
                subset.members.len(),
                subset.total_bytes()
            ));
            output::terminate(VERIFIED)
        }
        Mode::Stage => {
            stage(&install, &mut source, subset, &payload_path)?;
            output::terminate(STAGED)
        }
    }
}

fn report(error: PayloadError) -> Refusal {
    output::arguments(format_args!("payload refusal: {error}\n"));
    refuse!("install mode failed: {error}", %error)
}

fn copy_report<E: core::fmt::Debug>(error: CopyError<E>) -> Refusal {
    output::arguments(format_args!("payload refusal: {error}\n"));
    refuse!("install mode failed: {error}", %error)
}

fn path16(path: &str) -> Result<CString16> {
    CString16::try_from(path.replace('/', "\\").as_str()).map_err(|_| refuse!("path is not UCS-2"))
}

fn open_file(root: &mut Directory, path: &str) -> Result<RegularFile> {
    root.open(&path16(path)?, FileMode::Read, FileAttribute::empty())
        .map_err(|_| refuse!("cannot open a payload file"))?
        .into_regular_file()
        .ok_or(refuse!("a payload path is a directory"))
}

fn file_size(file: &mut RegularFile) -> Result<u64> {
    file.get_boxed_info::<FileInfo>()
        .map(|info| info.file_size())
        .map_err(|_| refuse!("cannot read a payload file's size"))
}

/// A whole file read at offsets, such as an IPSW.
struct UefiFile {
    file: RegularFile,
    length: u64,
}

impl ReadAt for UefiFile {
    type Error = Refusal;
    fn len(&self) -> u64 {
        self.length
    }
    fn read_at(&mut self, offset: u64, buffer: &mut [u8]) -> Result<()> {
        self.file
            .set_position(offset)
            .map_err(|_| refuse!("cannot seek a payload file"))?;
        let mut done = 0;
        while done < buffer.len() {
            let count = self
                .file
                .read(&mut buffer[done..])
                .map_err(|_| refuse!("cannot read a payload file"))?;
            if count == 0 {
                return Err(refuse!("payload file ended early"));
            }
            done += count;
        }
        Ok(())
    }
}

enum Store {
    Split {
        root: Directory,
        directory: String,
        slice_bytes: u64,
    },
    Archive {
        file: UefiFile,
        archive: Archive,
    },
}

/// The payload on one volume, which stays open while it is read.
struct Source {
    manifest: PayloadManifest,
    store: Store,
    _volume: ScopedProtocol<SimpleFileSystem>,
}

impl Source {
    /// The first volume with this build's payload, and its device path.
    fn find(build: &str) -> Result<(Self, String)> {
        let directory = images_directory(build);
        let manifest_path = format!("{directory}/{MANIFEST_FILE}");
        let handles =
            boot::find_handles::<SimpleFileSystem>().map_err(|_| refuse!("no file systems"))?;
        for handle in handles {
            let Ok(mut volume) = boot::open_protocol_exclusive::<SimpleFileSystem>(handle) else {
                continue;
            };
            let Ok(mut root) = volume.open_volume() else {
                continue;
            };
            let Ok(mut file) = open_file(&mut root, &manifest_path) else {
                continue;
            };
            let length = file_size(&mut file)?;
            let mut bytes =
                vec![0u8; usize::try_from(length).map_err(|_| refuse!("manifest size"))?];
            let mut reader = UefiFile { file, length };
            reader.read_at(0, &mut bytes)?;
            let manifest = PayloadManifest::parse(&bytes).map_err(report)?;
            if manifest.product_build != build {
                return Err(refuse!("payload.json is for another build"));
            }
            let store = match &manifest.layout {
                Layout::Split { slice_bytes } => Store::Split {
                    root,
                    directory: directory.clone(),
                    slice_bytes: *slice_bytes,
                },
                Layout::Archive { path } => {
                    let mut file = open_file(&mut root, path)?;
                    let length = file_size(&mut file)?;
                    if length != manifest.source.bytes {
                        return Err(refuse!("the IPSW's size differs from payload.json"));
                    }
                    let mut file = UefiFile { file, length };
                    let archive = Archive::open(&mut file).map_err(report)?;
                    Store::Archive { file, archive }
                }
                Layout::Staged => return Err(refuse!("payload.json describes a staged copy")),
                Layout::Pieces { .. } | Layout::Remote { .. } => {
                    return Err(refuse!(
                        "payload.json describes an install image, which --install reads"
                    ));
                }
            };
            let path = device_path(handle).unwrap_or_default();
            return Ok((
                Self {
                    manifest,
                    store,
                    _volume: volume,
                },
                path,
            ));
        }
        Err(refuse!("no volume has the payload for --install-build"))
    }
}

/// A split member, read slice after slice.
struct Slices<'a> {
    root: &'a mut Directory,
    directory: &'a str,
    name: String,
    bytes: u64,
    slice_bytes: u64,
    index: u64,
    current: Option<(RegularFile, u64)>,
    done: u64,
}

impl MemberRead for Slices<'_> {
    fn read(&mut self, buffer: &mut [u8]) -> Result<usize, PayloadError> {
        let failed = |what: Refusal| PayloadError::Io(what.to_string());
        loop {
            if self.done == self.bytes {
                return Ok(0);
            }
            if let Some((file, left)) = &mut self.current
                && *left > 0
            {
                let take = buffer.len().min(*left as usize);
                let count = file
                    .read(&mut buffer[..take])
                    .map_err(|_| failed(refuse!("cannot read a slice")))?;
                if count == 0 {
                    return Err(failed(refuse!("a slice ended early")));
                }
                *left -= count as u64;
                self.done += count as u64;
                return Ok(count);
            }
            if self.index >= slice_count(self.bytes, self.slice_bytes) {
                return Err(failed(refuse!("a member has too few slices")));
            }
            let path = format!("{}/{}", self.directory, slice_path(&self.name, self.index));
            let mut file = open_file(self.root, &path).map_err(failed)?;
            let expected = (self.bytes - self.index * self.slice_bytes).min(self.slice_bytes);
            if file_size(&mut file).map_err(failed)? != expected {
                return Err(failed(refuse!("a slice has the wrong size")));
            }
            self.current = Some((file, expected));
            self.index += 1;
        }
    }
}

enum Reader<'a> {
    Slices(Slices<'a>),
    Archive(MemberStream<'a, UefiFile>),
}

impl MemberRead for Reader<'_> {
    fn read(&mut self, buffer: &mut [u8]) -> Result<usize, PayloadError> {
        match self {
            Self::Slices(reader) => reader.read(buffer),
            Self::Archive(reader) => reader.read(buffer),
        }
    }
}

impl PayloadSource for Source {
    type Reader<'a>
        = Reader<'a>
    where
        Self: 'a;

    fn manifest(&self) -> &PayloadManifest {
        &self.manifest
    }

    fn open(&mut self, name: &str) -> Result<Reader<'_>, PayloadError> {
        let member = self
            .manifest
            .member(name)
            .ok_or(PayloadError::Manifest("member is not in the payload"))?
            .clone();
        match &mut self.store {
            Store::Split {
                root,
                directory,
                slice_bytes,
            } => Ok(Reader::Slices(Slices {
                root,
                directory,
                name: member.name,
                bytes: member.bytes,
                slice_bytes: *slice_bytes,
                index: 0,
                current: None,
                done: 0,
            })),
            Store::Archive { file, archive } => {
                let entry = archive
                    .find(name)
                    .ok_or(PayloadError::Archive("member is not in the IPSW"))?
                    .clone();
                if entry.bytes != member.bytes {
                    return Err(PayloadError::Verification {
                        member: member.name,
                        reason: "IPSW entry size differs from the manifest",
                    });
                }
                Ok(Reader::Archive(archive.open_member(file, &entry)?))
            }
        }
    }
}

/// The C `wcslen` that `uefi`'s device-path text conversion links against;
/// the UEFI target's builtins have none. UEFI wide characters are 16 bits.
///
/// # Safety
///
/// `text` must point to a NUL-terminated UCS-2 string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn wcslen(text: *const u16) -> usize {
    let mut length = 0;
    // SAFETY: the caller passes a NUL-terminated string.
    while unsafe { text.add(length).read() } != 0 {
        length += 1;
    }
    length
}

pub(crate) fn device_path(handle: Handle) -> Option<String> {
    // SAFETY: the path is only read, while no driver changes this handle.
    let path = unsafe {
        boot::open_protocol::<DevicePath>(
            OpenProtocolParams {
                handle,
                agent: boot::image_handle(),
                controller: None,
            },
            OpenProtocolAttributes::GetProtocol,
        )
    }
    .ok()?;
    path.to_string16(DisplayOnly(false), AllowShortcuts(false))
        .ok()
        .map(|text| text.to_string())
}

/// A page-aligned buffer, which every block device's IoAlign accepts.
pub(crate) struct Aligned {
    storage: Vec<u8>,
    start: usize,
    length: usize,
}

impl Aligned {
    pub(crate) fn new(length: usize) -> Self {
        let storage = vec![0u8; length + 4096];
        let start = storage.as_ptr().align_offset(4096);
        Self {
            storage,
            start,
            length,
        }
    }
    pub(crate) fn bytes(&mut self) -> &mut [u8] {
        &mut self.storage[self.start..self.start + self.length]
    }
}

/// The disk `stela.cfg` names, opened exclusively for writing.
struct Target {
    io: ScopedProtocol<BlockIO>,
    media: u32,
    disk: Disk,
    path: String,
}

impl Target {
    fn read(&mut self, lba: u64, blocks: u64) -> Result<Vec<u8>> {
        let mut buffer = Aligned::new((blocks * self.disk.block_bytes) as usize);
        self.io
            .read_blocks(self.media, lba, buffer.bytes())
            .map_err(|_| refuse!("cannot read the target disk"))?;
        Ok(buffer.bytes().to_vec())
    }

    fn write(&mut self, lba: u64, bytes: &[u8]) -> Result<()> {
        if !(bytes.len() as u64).is_multiple_of(self.disk.block_bytes)
            || lba + bytes.len() as u64 / self.disk.block_bytes > self.disk.blocks
        {
            return Err(refuse!("write outside the target disk"));
        }
        let mut buffer = Aligned::new(bytes.len());
        buffer.bytes().copy_from_slice(bytes);
        self.io
            .write_blocks(self.media, lba, buffer.bytes())
            .map_err(|_| refuse!("cannot write the target disk"))
    }

    /// The disk's GPT, from the backup when an interrupted write left the
    /// primary damaged (`gpt::read`). A disk without one is staged only when
    /// it is blank, so another system's MBR or filesystem is never
    /// overwritten.
    fn table(&mut self) -> Result<gpt::Found> {
        let disk = self.disk;
        gpt::read(disk, |lba, blocks| {
            self.read(lba, blocks)
                .map_err(|error| PayloadError::Io(error.to_string()))
        })
        .map_err(report)
    }
}

fn find_target(install: &Install, payload_path: &str) -> Result<Target> {
    let handles = boot::find_handles::<BlockIO>().map_err(|_| refuse!("no block devices"))?;
    let mut found: Vec<(Handle, String)> = Vec::new();
    for handle in handles {
        // SAFETY: the protocol is only read here, before any write.
        let Ok(io) = (unsafe {
            boot::open_protocol::<BlockIO>(
                OpenProtocolParams {
                    handle,
                    agent: boot::image_handle(),
                    controller: None,
                },
                OpenProtocolAttributes::GetProtocol,
            )
        }) else {
            continue;
        };
        let media = io.media();
        if media.is_logical_partition() || !media.is_media_present() || media.is_read_only() {
            continue;
        }
        let disk = Disk {
            block_bytes: media.block_size() as u64,
            blocks: media.last_block() + 1,
        };
        let media = media.media_id();
        let path = device_path(handle).unwrap_or_default();
        output::arguments(format_args!(
            "DISK path={path} bytes={} serial={:?}\n",
            disk.blocks * disk.block_bytes,
            disk_serial(handle, &path)
        ));
        let matches = match (
            install.target_guid,
            &install.target_serial,
            &install.target_path,
            install.target_bytes,
        ) {
            (Some(guid), None, None, None) => {
                let mut probe = Target {
                    io,
                    media,
                    disk,
                    path: path.clone(),
                };
                matches!(probe.table(), Ok(gpt::Found::Table { table, .. }) if table.disk_guid == guid)
            }
            (None, Some(serial), None, None) => {
                disk_serial(handle, &path).is_some_and(|found| found == *serial)
            }
            (None, None, Some(want), Some(bytes)) => {
                *want == path && bytes == disk.blocks * disk.block_bytes
            }
            _ => {
                return Err(refuse!(
                    "stage needs exactly one of --install-target-serial, \
                     --install-target-guid, or --install-target-path with \
                     --install-target-bytes"
                ));
            }
        };
        if matches {
            found.push((handle, path));
        }
    }
    let [(handle, path)] = &found[..] else {
        output::arguments(format_args!("matching target disks: {}\n", found.len()));
        return Err(refuse!(
            "the named target disk is not exactly one present, writable disk"
        ));
    };
    let holds_payload =
        payload_path == path || payload_path.starts_with(format!("{path}/").as_str());
    if path.is_empty() || holds_payload {
        return Err(refuse!(
            "the target disk holds the payload, or has no device path"
        ));
    }
    let io = boot::open_protocol_exclusive::<BlockIO>(*handle)
        .map_err(|_| refuse!("cannot open the target disk exclusively"))?;
    let media = io.media();
    let disk = Disk {
        block_bytes: media.block_size() as u64,
        blocks: media.last_block() + 1,
    };
    let media = media.media_id();
    Ok(Target {
        io,
        media,
        disk,
        path: path.clone(),
    })
}

/// The NVMe admin command that returns controller data, and the controller
/// data structure's serial number field.
const NVME_IDENTIFY: u8 = 0x06;
const NVME_IDENTIFY_CONTROLLER: u32 = 1;
const NVME_IDENTIFY_BYTES: usize = 4096;
const NVME_SERIAL: core::ops::Range<usize> = 4..24;
/// edk2 treats a zero timeout as already expired; its own admin commands
/// allow 5 seconds.
const NVME_TIMEOUT: core::time::Duration = core::time::Duration::from_secs(5);
/// ATA IDENTIFY DEVICE words 10-19: the serial number, and words 27-46:
/// the model number, two characters per word with the first in the high
/// byte.
const ATA_IDENTIFY_BYTES: usize = 512;
const ATA_SERIAL: core::ops::Range<usize> = 20..40;
const ATA_MODEL: core::ops::Range<usize> = 54..94;
/// NVMe Identify Controller bytes 24-63: the model number.
const NVME_MODEL: core::ops::Range<usize> = 24..64;

fn trimmed(bytes: &[u8]) -> Option<String> {
    let text = core::str::from_utf8(bytes).ok()?.trim_matches([' ', '\0']);
    (!text.is_empty()).then(|| text.into())
}

/// A whole disk's serial number, from its NVMe controller, its ATA
/// IDENTIFY data or its USB device's serial number string. Other buses,
/// such as virtio, report none.
pub(crate) fn disk_serial(handle: Handle, path: &str) -> Option<String> {
    nvme_identify(path)
        .and_then(|data| trimmed(data.get(NVME_SERIAL)?))
        .or_else(|| ata_text(handle, ATA_SERIAL))
        .or_else(|| usb_string(path, |device| device.str_serial_number))
}

/// A whole disk's model name, from the same sources as [`disk_serial`]
/// (the USB device's product string).
pub(crate) fn disk_model(handle: Handle, path: &str) -> Option<String> {
    nvme_identify(path)
        .and_then(|data| trimmed(data.get(NVME_MODEL)?))
        .or_else(|| ata_text(handle, ATA_MODEL))
        .or_else(|| usb_string(path, |device| device.str_product))
}

/// A string of the USB device the disk belongs to, `index` naming it in
/// the device descriptor, in US English and trimmed as Stela's own USB
/// driver reads it (`stela_drivers::usb::xhci`), so both name the disk
/// alike. Hubs' paths prefix their devices', so the device is the one with
/// the longest path the disk's starts with.
fn usb_string(
    path: &str,
    index: impl FnOnce(&uefi::proto::usb::DeviceDescriptor) -> u8,
) -> Option<String> {
    use uefi::proto::usb::io::UsbIo;
    const LANGUAGE_EN_US: u16 = 0x0409;
    let (device, _) = boot::find_handles::<UsbIo>()
        .ok()?
        .into_iter()
        .filter_map(|device| Some((device, device_path(device)?)))
        .filter(|(_, prefix)| path == prefix || path.starts_with(format!("{prefix}/").as_str()))
        .max_by_key(|(_, prefix)| prefix.len())?;
    // SAFETY: descriptors are only read.
    let mut io = unsafe {
        boot::open_protocol::<UsbIo>(
            OpenProtocolParams {
                handle: device,
                agent: boot::image_handle(),
                controller: None,
            },
            OpenProtocolAttributes::GetProtocol,
        )
    }
    .ok()?;
    let index = index(&io.device_descriptor().ok()?);
    if index == 0 {
        return None;
    }
    let text = io
        .string_descriptor(LANGUAGE_EN_US, index)
        .ok()?
        .to_string();
    let text = text.trim();
    (!text.is_empty()).then(|| text.into())
}

/// The Identify Controller data of the NVMe controller `path` is on.
fn nvme_identify(path: &str) -> Option<Vec<u8>> {
    use uefi::proto::nvme::{NvmeQueueType, NvmeRequestBuilder, pass_thru::NvmePassThru};
    for controller in boot::find_handles::<NvmePassThru>().ok()? {
        let Some(prefix) = device_path(controller) else {
            continue;
        };
        if !path.starts_with(format!("{prefix}/NVMe(").as_str()) {
            continue;
        }
        // SAFETY: an admin Identify command reads controller data only.
        let pass_thru = unsafe {
            boot::open_protocol::<NvmePassThru>(
                OpenProtocolParams {
                    handle: controller,
                    agent: boot::image_handle(),
                    controller: None,
                },
                OpenProtocolAttributes::GetProtocol,
            )
        }
        .ok()?;
        let request =
            NvmeRequestBuilder::new(pass_thru.io_align(), NVME_IDENTIFY, NvmeQueueType::ADMIN)
                .with_cdw10(NVME_IDENTIFY_CONTROLLER)
                .with_timeout(NVME_TIMEOUT)
                .with_transfer_buffer(NVME_IDENTIFY_BYTES)
                .ok()?
                .build();
        let response = pass_thru.controller().execute_command(request).ok()?;
        return Some(response.transfer_buffer()?.to_vec());
    }
    None
}

/// The text in `range` of the disk's ATA IDENTIFY data.
fn ata_text(handle: Handle, range: core::ops::Range<usize>) -> Option<String> {
    use uefi::proto::media::disk_info::{DiskInfo, DiskInfoInterface};
    // SAFETY: the identify data is only read.
    let info = unsafe {
        boot::open_protocol::<DiskInfo>(
            OpenProtocolParams {
                handle,
                agent: boot::image_handle(),
                controller: None,
            },
            OpenProtocolAttributes::GetProtocol,
        )
    }
    .ok()?;
    if !matches!(
        info.interface(),
        DiskInfoInterface::AHCI | DiskInfoInterface::IDE
    ) {
        return None;
    }
    let mut data = [0u8; ATA_IDENTIFY_BYTES];
    if info.identify(&mut data).ok()? < range.end {
        return None;
    }
    let mut text = data[range].to_vec();
    for pair in text.as_chunks_mut::<2>().0 {
        pair.swap(0, 1);
    }
    // As Stela's AHCI driver reads the field, so both name the disk alike.
    let text = stela_drivers::block::ascii(&text);
    (!text.is_empty()).then_some(text)
}

/// GUID entropy for a new table: the disk, the time and a counter. These
/// GUIDs must be unique, not secret.
fn entropy(target: &Target, manifest: &str) -> impl FnMut() -> [u8; 16] {
    let time = uefi::runtime::get_time()
        .map(|time| format!("{time:?}"))
        .unwrap_or_default();
    let seed = format!("{}|{}|{time}|{manifest}", target.path, target.disk.blocks);
    let mut counter = 0u64;
    move || {
        counter += 1;
        let digest = Sha256::new()
            .chain_update(seed.as_bytes())
            .chain_update(counter.to_le_bytes())
            .finalize();
        digest[..16].try_into().unwrap()
    }
}

/// Buffered whole-block writes into one partition.
struct PartitionWriter<'a> {
    target: &'a mut Target,
    first: u64,
    end: u64,
    next: u64,
    buffer: Vec<u8>,
}

impl PartitionWriter<'_> {
    fn seek(&mut self, offset: u64) -> Result<()> {
        if !self.buffer.is_empty() || !offset.is_multiple_of(self.target.disk.block_bytes) {
            return Err(refuse!("staging extent is not block aligned"));
        }
        self.next = self.first + offset / self.target.disk.block_bytes;
        Ok(())
    }

    fn put(&mut self, bytes: &[u8]) -> Result<()> {
        self.buffer.extend_from_slice(bytes);
        if self.buffer.len() >= COPY_BYTES {
            let whole = self.buffer.len() / self.target.disk.block_bytes as usize
                * self.target.disk.block_bytes as usize;
            let rest = self.buffer.split_off(whole);
            self.flush_whole()?;
            self.buffer = rest;
        }
        Ok(())
    }

    fn flush_whole(&mut self) -> Result<()> {
        let blocks = self.buffer.len() as u64 / self.target.disk.block_bytes;
        if self.next + blocks > self.end {
            return Err(refuse!("staging overruns its partition"));
        }
        let bytes = core::mem::take(&mut self.buffer);
        self.target.write(self.next, &bytes)?;
        self.next += blocks;
        Ok(())
    }

    /// Write what is left, padded with zeros to a whole block.
    fn finish(&mut self) -> Result<()> {
        let block = self.target.disk.block_bytes as usize;
        let padded = self.buffer.len().next_multiple_of(block);
        self.buffer.resize(padded, 0);
        self.flush_whole()
    }
}

fn stage(
    install: &Install,
    source: &mut Source,
    subset: PayloadManifest,
    payload_path: &str,
) -> Result<()> {
    let source_json = source.manifest.to_json();
    let header = Header::plan(subset, stela_formats::sha256(source_json.as_bytes()));
    let mut target = find_target(install, payload_path)?;
    output::arguments(format_args!(
        "TARGET path={} block={} blocks={}\n",
        target.path, target.disk.block_bytes, target.disk.blocks
    ));
    // An earlier stage that stopped part way through its GPT writes is
    // resumed: before the primary header they read as the earlier table (or
    // blank), after it as the new one with its remaining writes to repeat.
    let (existing, repair) = match target.table()? {
        gpt::Found::Blank => (None, None),
        gpt::Found::Interrupted => (
            None,
            Some("an earlier stage stopped before its first GPT header"),
        ),
        gpt::Found::Table { table, repair } => (Some(table), repair),
    };
    if let Some(repair) = repair {
        output::arguments(format_args!("GPT REPAIR {repair}\n"));
    }
    let request = Request {
        staging_bytes: header.total_bytes(),
        machine_state_bytes: gpt::MACHINE_STATE_BYTES,
        cache_bytes: install.cache_bytes.unwrap_or(gpt::DEFAULT_CACHE_BYTES),
        min_apple_bytes: install
            .min_apple_bytes
            .unwrap_or(gpt::DEFAULT_MIN_APPLE_BYTES),
    };
    let plan = gpt::plan(
        target.disk,
        existing.as_ref(),
        request,
        entropy(&target, &source_json),
    )
    .map_err(report)?;
    if plan.write || repair.is_some() {
        for write in gpt::encode(target.disk, &plan.table).map_err(report)? {
            target.write(write.lba, &write.bytes)?;
        }
        target
            .io
            .flush_blocks()
            .map_err(|_| refuse!("cannot flush the target disk"))?;
    }
    for partition in &plan.table.partitions {
        output::arguments(format_args!(
            "PARTITION {:?} first={} last={} guid={}\n",
            partition.role(),
            partition.first_lba,
            partition.last_lba,
            partition.unique_guid
        ));
    }
    let staging = plan
        .partition(Role::Staging)
        .ok_or(refuse!("the plan has no staging partition"))?
        .clone();
    let mut writer = PartitionWriter {
        first: staging.first_lba,
        end: staging.last_lba + 1,
        next: staging.first_lba,
        buffer: Vec::new(),
        target: &mut target,
    };
    // Invalidate any earlier copy before the first member byte changes.
    writer.put(&vec![0u8; staging::HEADER_BYTES as usize])?;
    writer.finish()?;
    let mut buffer = vec![0u8; COPY_BYTES];
    for (member, extent) in header.manifest.members.iter().zip(&header.extents) {
        writer.seek(extent.offset)?;
        copy_verified(source, member, &mut buffer, |bytes| writer.put(bytes))
            .map_err(copy_report)?;
        writer.finish()?;
        output::arguments(format_args!(
            "STAGED {} bytes={} offset={}\n",
            member.name, member.bytes, extent.offset
        ));
    }
    writer.seek(0)?;
    writer.put(&header.encode().map_err(report)?)?;
    writer.finish()?;
    target
        .io
        .flush_blocks()
        .map_err(|_| refuse!("cannot flush the target disk"))?;
    output::arguments(format_args!(
        "STELA PAYLOAD STAGED members={} bytes={} disk={}\n",
        header.manifest.members.len(),
        header.manifest.total_bytes(),
        plan.table.disk_guid
    ));
    Ok(())
}
