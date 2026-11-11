//! The USB's own boot log beside the boot settings on its volume,
//! which `stela setup` preallocates: the loader writes it through the
//! firmware's file system from its first line until boot services end
//! ([`open`]), and the system engine writes on to the same clusters, wherever
//! they lie, through Stela's own drivers ([`resume`]), so the stick keeps the
//! whole run.
use crate::Result;
use crate::{
    engine::{bundle::Bytes, translations},
    output::{self, Log},
    status,
};
use alloc::{boxed::Box, format, string::String, vec, vec::Vec};
use core::sync::atomic::{AtomicBool, Ordering};
use stela_contracts::refuse;
use stela_crypto::{Sha256, digest::Digest};
use stela_drivers::block::{BlockDevice, BlockError, Features, Identity, Shared, Transport};
use stela_loader::mount_interface::{
    VOLUME_LOG_BYTES, VOLUME_LOG_PREFIX, VOLUME_LOG_SLOTS,
    layout::{BOOT_LOG, DATA_LOGS, Layout},
    volume_log_sequence,
};
use stela_runtime::pc::{
    boot_log::{self, LOG_BYTES, Writer},
    fat32::{Fat32, FatError},
    gpt::Guid,
};
use uefi::{
    CString16, Handle, Status,
    boot::{self, OpenProtocolAttributes, OpenProtocolParams, ScopedProtocol},
    proto::{
        ProtocolPointer,
        device_path::{DevicePath, DeviceSubType, DeviceType},
        loaded_image::LoadedImage,
        media::{
            file::{File as _, FileAttribute, FileInfo, FileMode, RegularFile},
            fs::SimpleFileSystem,
        },
    },
};

/// The blocks the firmware's file is addressed in.
const BLOCK: u32 = 512;
/// Bound the firmware prefix so the engine keeps at least 192 KiB.
const LOADER_BYTES: usize = 64 << 10;
const HEADER_BYTES: usize = 128;
/// The append point is aligned for physical disks with 512 B or 4 KiB blocks.
const HEAD_ALIGNMENT: usize = 4096;

static VOLUME_LOG_ENABLED: AtomicBool = AtomicBool::new(true);

/// Apply the startup choice's logging policy before it opens guest inputs.
pub(crate) fn set_volume_log(enabled: bool) {
    VOLUME_LOG_ENABLED.store(enabled, Ordering::Relaxed);
}

/// The boot device's GPT identity and observed USB path before firmware exit.
#[cfg(feature = "diagnostics")]
pub(crate) fn diagnostic_boot_device() -> (Option<Guid>, bool) {
    let device = shared::<LoadedImage>(boot::image_handle()).and_then(|image| image.device());
    let Some(device) = device else {
        return (None, false);
    };
    let usb = shared::<DevicePath>(device).is_some_and(|path| {
        path.node_iter().any(|node| {
            node.device_type() == DeviceType::MESSAGING
                && matches!(
                    node.sub_type(),
                    DeviceSubType::MESSAGING_USB
                        | DeviceSubType::MESSAGING_USB_CLASS
                        | DeviceSubType::MESSAGING_USB_WWID
                )
        })
    });
    (translations::partition(device).map(Guid), usb)
}

/// Identifies the exact slot opened before ExitBootServices. The engine
/// must rediscover the same partition and verify the header before writing.
#[derive(Clone, Copy)]
pub(crate) struct VolumeLog {
    pub(crate) partition: [u8; 16],
    layout: Layout,
    slot: usize,
    head: usize,
    header: [u8; 32],
    #[cfg(feature = "diagnostics")]
    record_end: u64,
}

/// Store this boot's firmware output on the USB's visible data partition.
/// Only a sibling of the boot partition with prepared slots may be written.
pub(crate) fn save_loader_log() -> Option<VolumeLog> {
    if !VOLUME_LOG_ENABLED.load(Ordering::Relaxed) {
        return None;
    }
    let Some(device) = shared::<LoadedImage>(boot::image_handle()).and_then(|image| image.device())
    else {
        return None;
    };
    let Some(path) = shared::<DevicePath>(device) else {
        return None;
    };
    let Some(boot_disk) = partition_path(&path) else {
        return None;
    };
    let Ok(handles) = boot::find_handles::<SimpleFileSystem>() else {
        return None;
    };
    for handle in handles.into_iter().filter(|&handle| handle != device) {
        // Drop this DevicePath protocol before looking up the same
        // partition's GUID. EDK2 tracks both queries by the same agent.
        let same_disk = shared::<DevicePath>(handle)
            .is_some_and(|path| partition_path(&path).is_some_and(|disk| disk == boot_disk));
        if !same_disk {
            continue;
        }
        let Some(mut volume) = shared::<SimpleFileSystem>(handle) else {
            continue;
        };
        let Ok(mut root) = volume.open_volume() else {
            continue;
        };
        for layout in Layout::PREFERRED {
            if let Some((slot, head, header, _record_end, path)) =
                write_loader_log(&mut root, layout)
            {
                output::arguments(format_args!("BOOT LOG FILE {path} through the firmware\n"));
                return Some(VolumeLog {
                    partition: translations::partition(handle)?,
                    layout,
                    slot,
                    head,
                    header,
                    #[cfg(feature = "diagnostics")]
                    record_end: _record_end,
                });
            }
        }
    }
    None
}

/// A full device path's nonempty prefix up to its terminal partition node.
/// A short hard-drive path alone cannot identify the parent disk.
fn partition_path(path: &DevicePath) -> Option<Vec<&uefi::proto::device_path::DevicePathNode>> {
    let mut nodes: Vec<_> = path.node_iter().collect();
    let last = nodes.pop()?;
    (last.full_type() == (DeviceType::MEDIA, DeviceSubType::MEDIA_HARD_DRIVE) && !nodes.is_empty())
        .then_some(nodes)
}

fn write_loader_log(
    root: &mut uefi::proto::media::file::Directory,
    layout: Layout,
) -> Option<(usize, usize, [u8; 32], u64, String)> {
    let directory = layout.data_path(DATA_LOGS).replace('/', "\\");
    let mut oldest = None;
    let mut newest = 0u64;
    for (slot, name) in VOLUME_LOG_SLOTS.into_iter().enumerate() {
        let path = format!("\\{directory}\\{name}");
        let Some(mut file) = loader_log_file(root, &path, FileMode::Read) else {
            continue;
        };
        let mut header = [0; HEADER_BYTES];
        let Ok(read) = file.read(&mut header) else {
            continue;
        };
        let Some(sequence) = volume_log_sequence(&header[..read]) else {
            continue;
        };
        newest = newest.max(sequence);
        if oldest
            .as_ref()
            .is_none_or(|(oldest, _, _)| sequence < *oldest)
        {
            oldest = Some((sequence, slot, path));
        }
    }
    let (_, slot, path) = oldest?;
    let sequence = newest.checked_add(1)?;
    let time = uefi::runtime::get_time().map_or_else(
        |_| String::from("unknown"),
        |time| {
            format!(
                "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}",
                time.year(),
                time.month(),
                time.day(),
                time.hour(),
                time.minute(),
                time.second(),
            )
        },
    );
    let commit = option_env!("NORTHWOOD_SOURCE_REVISION")
        .or(option_env!("GITHUB_SHA"))
        .and_then(|revision| revision.get(..12))
        .filter(|revision| revision.bytes().all(|byte| byte.is_ascii_hexdigit()))
        .unwrap_or("unknown");
    let cpu = core::arch::x86_64::__cpuid(1).eax;
    let header = format!(
        "{VOLUME_LOG_PREFIX}{sequence} version={} commit={commit} cpu={cpu:08x} time={time}",
        env!("CARGO_PKG_VERSION")
    );
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(VOLUME_LOG_BYTES).ok()?;
    bytes.resize(VOLUME_LOG_BYTES, b' ');
    bytes
        .get_mut(..HEADER_BYTES - 1)?
        .get_mut(..header.len())?
        .copy_from_slice(header.as_bytes());
    bytes[HEADER_BYTES - 1] = b'\n';
    let notice = b"[earlier loader output omitted]\n";
    let start = HEADER_BYTES + notice.len();
    let (written, truncated, record_end) =
        output::loader_log_tail(&mut bytes[start..LOADER_BYTES])?;
    let used = if truncated {
        bytes[HEADER_BYTES..start].copy_from_slice(notice);
        let skip = bytes[start..start + written]
            .iter()
            .position(|&byte| byte == b'\n')
            .map_or(written, |index| index + 1);
        bytes.copy_within(start + skip..start + written, start);
        bytes[start + written - skip..].fill(b' ');
        start + written - skip
    } else {
        bytes.copy_within(start..start + written, HEADER_BYTES);
        bytes[HEADER_BYTES + written..].fill(b' ');
        HEADER_BYTES + written
    };
    let head = used.next_multiple_of(HEAD_ALIGNMENT);
    if head > used {
        bytes[head - 1] = b'\n';
    }
    let mut file = loader_log_file(root, &path, FileMode::ReadWrite)?;
    // Commit this sequence after the body. If power fails midway, the
    // other slots still identify the preceding newest complete boot.
    file.set_position(HEADER_BYTES as u64).ok()?;
    file.write(&bytes[HEADER_BYTES..]).ok()?;
    file.flush().ok()?;
    file.set_position(0).ok()?;
    file.write(&bytes[..HEADER_BYTES]).ok()?;
    file.flush().ok()?;
    let header = Sha256::digest(&bytes[..HEADER_BYTES]).into();
    Some((slot, head, header, record_end, path))
}

fn loader_log_file(
    root: &mut uefi::proto::media::file::Directory,
    path: &str,
    mode: FileMode,
) -> Option<RegularFile> {
    let path = CString16::try_from(path).ok()?;
    let mut file = root
        .open(&path, mode, FileAttribute::empty())
        .ok()?
        .into_regular_file()?;
    (file.get_boxed_info::<FileInfo>().ok()?.file_size() == VOLUME_LOG_BYTES as u64).then_some(file)
}

/// Open the log on the volume Stela was loaded from, when it has one, and
/// return that partition's GUID, which the system engine opens it by.
pub fn open() -> Option<Guid> {
    status::begin("Opening the boot log on the USB");
    let opened = firmware_file().and_then(|(device, partition)| {
        let mut device: Box<dyn BlockDevice> = Box::new(device);
        let writer = Writer::open(&mut *device)
            .map_err(|error| refuse!("firmware boot log: {error}", %error))?;
        report(&writer, "firmware", &device.identity().location);
        output::add_log(writer, device, Log::Firmware)
            .map_err(|error| refuse!("firmware boot log output: {error}", %error))?;
        Ok(partition)
    });
    finish(opened).flatten()
}

/// Write on to the loader's log in its file on `partition`, the volume
/// [`open`] found it on.
pub fn resume(partition: Box<dyn BlockDevice>) {
    status::begin("Opening the boot log on the USB again");
    finish(engine_file(partition));
}

/// Continue the exact data-volume slot committed before firmware exit.
pub(crate) fn resume_volume(partition: Box<dyn BlockDevice>, log: VolumeLog) {
    resume_volume_at(partition, log, 0);
}

/// A standalone diagnostic still has the loader's output record after EBS.
#[cfg(feature = "diagnostics")]
pub(crate) fn resume_diagnostic_volume(partition: Box<dyn BlockDevice>, log: VolumeLog) {
    resume_volume_at(partition, log, log.record_end);
}

fn resume_volume_at(partition: Box<dyn BlockDevice>, log: VolumeLog, record_start: u64) {
    let opened = volume_file(partition, log, record_start);
    if let Err(error) = opened {
        output::arguments(format_args!("BOOT LOG data volume left out: {error}\n"));
    }
}

fn volume_file(partition: Box<dyn BlockDevice>, log: VolumeLog, record_start: u64) -> Result<()> {
    let identity = partition.identity();
    let block = identity.block_size as usize;
    if identity.read_only
        || !(HEADER_BYTES..=HEAD_ALIGNMENT).contains(&block)
        || !VOLUME_LOG_BYTES.is_multiple_of(block)
        || !(HEADER_BYTES..=LOADER_BYTES).contains(&log.head)
        || !log.head.is_multiple_of(block)
    {
        return Err(refuse!("data-volume log has unsupported geometry"));
    }
    let name = VOLUME_LOG_SLOTS
        .get(log.slot)
        .ok_or(refuse!("invalid data-volume log slot"))?;
    let path = log.layout.data_path(&format!("{DATA_LOGS}/{name}"));
    let mut volume = Fat32::open(Bytes::new(partition))
        .map_err(|error| refuse!("data-volume log filesystem: {error}", %error))?;
    let file = volume
        .file(&path)
        .map_err(|error| refuse!("data-volume log file: {error}", %error))?;
    if file.bytes != VOLUME_LOG_BYTES as u64 {
        return Err(refuse!("data-volume log file changed size"));
    }
    let runs = boot_log::file_runs(&file, block as u64, VOLUME_LOG_BYTES as u64)
        .map_err(|error| refuse!("data-volume log file runs: {error}", %error))?;
    let partition = volume.into_inner().into_inner();
    let mut file = Shared::new(partition)
        .runs(runs)
        .map_err(|error| refuse!("data-volume log blocks: {error}", %error))?;
    let mut first = vec![0; block];
    file.read(0, &mut first)
        .map_err(|error| refuse!("data-volume log header: {error}", %error))?;
    let header: [u8; 32] = Sha256::digest(&first[..HEADER_BYTES]).into();
    if header != log.header {
        return Err(refuse!("data-volume log header changed"));
    }
    let writer = output::volume_log::Writer::new(
        (log.head / block) as u64,
        block,
        VOLUME_LOG_BYTES - log.head,
        record_start,
    );
    output::arguments(format_args!(
        "BOOT LOG FILE {path} through the USB driver\n"
    ));
    output::add_volume_log(writer, Box::new(file), record_start);
    Ok(())
}

/// End the step: the log is kept, or why not, which the boot goes on
/// without.
fn finish<T>(opened: Result<T>) -> Option<T> {
    match opened {
        Ok(value) => {
            status::done(BOOT_LOG);
            Some(value)
        }
        Err(why) => {
            output::arguments(format_args!("BOOT LOG FILE off: {why}\n"));
            status::warn(&format!("not kept: {why}"));
            None
        }
    }
}

fn report(writer: &Writer, through: &str, path: &str) {
    if let Some(reason) = writer.fresh() {
        output::arguments(format_args!("BOOT LOG FILE fresh: {reason}\n"));
    }
    output::arguments(format_args!(
        "BOOT LOG FILE {path} through the {through} boot={} head={:#x}\n",
        writer.boots(),
        writer.head()
    ));
}

/// `handle`'s `P`, shared with whoever else holds it.
fn shared<P: ProtocolPointer + ?Sized>(handle: Handle) -> Option<ScopedProtocol<P>> {
    // SAFETY: used only for queries and file access while boot services
    // run; no driver on `handle` is replaced while it is held.
    unsafe {
        boot::open_protocol::<P>(
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

fn firmware_esp_log(
    root: &mut uefi::proto::media::file::Directory,
) -> Result<(String, RegularFile)> {
    for layout in Layout::PREFERRED {
        let path = layout.uefi_path(BOOT_LOG);
        let name = CString16::try_from(path.as_str())
            .map_err(|_| refuse!("the log's path is not UCS-2"))?;
        match root.open(&name, FileMode::ReadWrite, FileAttribute::empty()) {
            Ok(file) => {
                let file = file
                    .into_regular_file()
                    .ok_or(refuse!("the USB log path is a directory"))?;
                return Ok((path, file));
            }
            Err(error) if error.status() == Status::NOT_FOUND => {}
            Err(_) => return Err(refuse!("the USB boot log cannot be opened for writing")),
        }
    }
    Err(refuse!("the USB has no boot.log in either EFI layout"))
}

fn firmware_file() -> Result<(FileDevice, Option<Guid>)> {
    let device = shared::<LoadedImage>(boot::image_handle())
        .ok_or(refuse!("no loaded image"))?
        .device()
        .ok_or(refuse!("Stela was not loaded from a volume"))?;
    let mut volume =
        shared::<SimpleFileSystem>(device).ok_or(refuse!("Stela's volume has no file system"))?;
    let mut root = volume
        .open_volume()
        .map_err(|_| refuse!("Stela's volume cannot be opened"))?;
    let (path, mut file) = firmware_esp_log(&mut root)?;
    let size = file
        .get_boxed_info::<FileInfo>()
        .map_err(|_| refuse!("the log file's size cannot be read"))?
        .file_size();
    if size < LOG_BYTES {
        return Err(refuse!("boot.log is shorter than the log"));
    }
    let identity = Identity {
        transport: Transport::Firmware,
        location: path,
        model: String::new(),
        serial: String::new(),
        blocks: LOG_BYTES / u64::from(BLOCK),
        block_size: BLOCK,
        read_only: false,
        // The firmware's file system may hold writes until a flush.
        features: Features {
            write_cache: true,
            ..Features::default()
        },
    };
    Ok((
        FileDevice { identity, file },
        translations::partition(device).map(Guid),
    ))
}

fn engine_file(partition: Box<dyn BlockDevice>) -> Result<()> {
    let block = u64::from(partition.identity().block_size);
    let mut volume = Fat32::open(Bytes::new(partition))
        .map_err(|error| refuse!("boot log volume: {error}", %error))?;
    let (layout, file) = engine_esp_log(&mut volume)?;
    let runs = boot_log::file_runs(&file, block, LOG_BYTES)
        .map_err(|error| refuse!("boot log file runs: {error}", %error))?;
    let partition = volume.into_inner().into_inner();
    let mut device: Box<dyn BlockDevice> = Box::new(
        Shared::new(partition)
            .runs(runs)
            .map_err(|error| refuse!("boot log device: {error}", %error))?,
    );
    let writer = Writer::resume(&mut *device)
        .map_err(|error| refuse!("boot log resume: {error}", %error))?;
    report(&writer, "USB driver", &layout.uefi_path(BOOT_LOG));
    output::add_log(writer, device, Log::Usb)
        .map_err(|error| refuse!("USB boot log output: {error}", %error))
}

fn engine_esp_log<R: stela_formats::payload::ReadAt>(
    volume: &mut Fat32<R>,
) -> Result<(Layout, stela_runtime::pc::fat32::File)> {
    for layout in Layout::PREFERRED {
        match volume.file(&layout.esp_path(BOOT_LOG)) {
            Ok(file) => return Ok((layout, file)),
            Err(FatError::Missing(_)) => {}
            Err(error) => return Err(refuse!("boot log file: {error}", %error)),
        }
    }
    Err(refuse!("the USB has no boot.log in either EFI layout"))
}

/// The log file as blocks, through the firmware's file system.
struct FileDevice {
    identity: Identity,
    file: RegularFile,
}

impl FileDevice {
    fn seek(&mut self, lba: u64, bytes: usize) -> Result<(), BlockError> {
        if !bytes.is_multiple_of(BLOCK as usize) {
            return Err(BlockError::Misaligned {
                bytes,
                block_size: BLOCK,
            });
        }
        let blocks = (bytes / BLOCK as usize) as u64;
        if lba.saturating_add(blocks) > self.identity.blocks {
            return Err(BlockError::OutOfRange {
                lba,
                blocks,
                capacity: self.identity.blocks,
            });
        }
        self.file
            .set_position(lba * u64::from(BLOCK))
            .map_err(|error| failed(error.status()))
    }
}

fn failed(status: uefi::Status) -> BlockError {
    BlockError::Device {
        status: status.0 as u32,
    }
}

impl BlockDevice for FileDevice {
    fn identity(&self) -> &Identity {
        &self.identity
    }

    fn read(&mut self, lba: u64, buffer: &mut [u8]) -> Result<(), BlockError> {
        self.seek(lba, buffer.len())?;
        let mut done = 0;
        while done < buffer.len() {
            match self.file.read(&mut buffer[done..]) {
                Ok(0) => return Err(failed(uefi::Status::END_OF_FILE)),
                Ok(count) => done += count,
                Err(error) => return Err(failed(error.status())),
            }
        }
        Ok(())
    }

    fn write(&mut self, lba: u64, data: &[u8]) -> Result<(), BlockError> {
        self.seek(lba, data.len())?;
        self.file
            .write(data)
            .map_err(|error| failed(error.status()))
    }

    fn flush_cache(&mut self) -> Result<(), BlockError> {
        self.file.flush().map_err(|error| failed(error.status()))
    }
}
