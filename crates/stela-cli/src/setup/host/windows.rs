//! Windows: disks are `\\.\PhysicalDriveN`, which setup names `diskN`, and
//! their partitions `diskNpM` by partition number. The storage control codes
//! give each disk's bus, serial number and partition table; the volume
//! functions give each volume's disk, file system and mount paths. Partitions
//! are read and written through the physical drive, a sector at a time as
//! Windows requires, and a USB is erased by writing setup's own layout to
//! its whole disk, with its volumes locked and dismounted.
use super::{
    Bus, Cpu, Disk, DiskFacts, FileSystem, Host, Partition, PartitionInfo, Volume,
    drive_names::{self, disk_number, partition_number, pci_device},
};
use anyhow::{Context, Result, ensure};
use std::{
    ffi::OsStr,
    fs::File,
    io,
    mem::{offset_of, size_of},
    os::windows::{
        ffi::{OsStrExt, OsStringExt},
        fs::{FileExt, OpenOptionsExt},
        io::AsRawHandle,
    },
    path::{Path, PathBuf},
    ptr,
};
use windows_sys::{
    Win32::{
        Devices::DeviceAndDriverInstallation::{
            DIGCF_PRESENT, SP_DEVINFO_DATA, SPDRP_COMPATIBLEIDS, SetupDiDestroyDeviceInfoList,
            SetupDiEnumDeviceInfo, SetupDiGetClassDevsW, SetupDiGetDeviceInstanceIdW,
            SetupDiGetDeviceRegistryPropertyW,
        },
        Foundation::{
            ERROR_INSUFFICIENT_BUFFER, ERROR_MORE_DATA, ERROR_NO_MORE_FILES, GENERIC_READ,
            GENERIC_WRITE, INVALID_HANDLE_VALUE,
        },
        Storage::FileSystem::{
            BusType1394, BusTypeFileBackedVirtual, BusTypeNvme, BusTypeUsb,
            DeleteVolumeMountPointW, FILE_SHARE_READ, FILE_SHARE_WRITE, FindFirstVolumeW,
            FindNextVolumeW, FindVolumeClose, GetDiskFreeSpaceExW, GetLogicalDrives,
            GetVolumeInformationW, GetVolumeNameForVolumeMountPointW, GetVolumePathNameW,
            GetVolumePathNamesForVolumeNameW, IOCTL_VOLUME_GET_VOLUME_DISK_EXTENTS,
            SetVolumeMountPointW,
        },
        System::{
            IO::DeviceIoControl,
            Ioctl::{
                DISK_EXTENT, DISK_GEOMETRY_EX, DRIVE_LAYOUT_INFORMATION_EX, FSCTL_DISMOUNT_VOLUME,
                FSCTL_LOCK_VOLUME, IOCTL_DISK_GET_DRIVE_GEOMETRY_EX,
                IOCTL_DISK_GET_DRIVE_LAYOUT_EX, IOCTL_DISK_UPDATE_PROPERTIES,
                IOCTL_STORAGE_QUERY_PROPERTY, NVMeDataTypeIdentify, PARTITION_INFORMATION_EX,
                PARTITION_STYLE_GPT, PARTITION_STYLE_MBR, PropertyStandardQuery, ProtocolTypeNvme,
                STORAGE_DEVICE_DESCRIPTOR, STORAGE_PROPERTY_QUERY,
                STORAGE_PROTOCOL_DATA_DESCRIPTOR, STORAGE_PROTOCOL_SPECIFIC_DATA,
                StorageAdapterProtocolSpecificProperty, StorageDeviceProperty, VOLUME_DISK_EXTENTS,
            },
            SystemInformation::GetSystemWindowsDirectoryW,
        },
    },
    core::GUID,
};

/// How setup gets the rights its raw reads and writes need.
pub(crate) const ADMIN: &str = "a terminal run as administrator";
/// A whole disk as `--usb` names it.
pub(crate) const DISK_EXAMPLE: &str = "PhysicalDrive2";

/// The EFI system partition's GPT type.
const ESP: GUID = GUID::from_u128(0xc12a7328_f81f_11d2_ba4b_00a0c93ec93b);
/// Physical drive numbers setup looks for.
const DRIVES: u32 = 128;

/// The computer `stela setup` runs on.
pub(crate) struct System;

/// `text` as a NUL-terminated wide string.
fn wide(text: impl AsRef<OsStr>) -> Vec<u16> {
    text.as_ref().encode_wide().chain([0]).collect()
}

/// A wide string up to its NUL.
fn narrow(wide: &[u16]) -> String {
    let end = wide.iter().position(|c| *c == 0).unwrap_or(wide.len());
    std::ffi::OsString::from_wide(&wide[..end])
        .to_string_lossy()
        .into_owned()
}

fn last_error() -> io::Error {
    io::Error::last_os_error()
}

/// `path`, a device or volume, opened for `access` with others still
/// reading and writing it; access 0 asks only for its control codes.
fn open(path: &str, access: u32) -> io::Result<File> {
    std::fs::OpenOptions::new()
        .access_mode(access)
        .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE)
        .open(path)
}

/// Read an IOCTL response structure at `offset`, if the complete structure fits.
///
/// # Safety
/// The bytes in this range must encode a valid `T`, including any Boolean
/// fields. Callers use the structure specified by the successful Windows IOCTL.
unsafe fn read<T: Copy>(bytes: &[u8], offset: usize) -> Option<T> {
    let end = offset.checked_add(size_of::<T>())?;
    // SAFETY: the checked range holds T; the caller guarantees its field values
    // are valid, and read_unaligned does not require the byte buffer to align T.
    (end <= bytes.len()).then(|| unsafe { ptr::read_unaligned(bytes[offset..].as_ptr().cast()) })
}

/// Send `code` with `input` to `file`; the answer, growing the buffer from
/// `capacity` while the device asks for more.
fn control(file: &File, code: u32, input: &[u8], capacity: usize) -> io::Result<Vec<u8>> {
    let mut size = capacity.max(64);
    loop {
        let mut output = vec![0u8; size];
        let mut returned = 0u32;
        // SAFETY: file keeps the synchronous handle open; input and the initialized
        // output buffer live through the call and match their supplied lengths.
        // returned is a writable u32, and a null OVERLAPPED requests synchronous I/O.
        let done = unsafe {
            DeviceIoControl(
                file.as_raw_handle(),
                code,
                input.as_ptr().cast(),
                input.len() as u32,
                output.as_mut_ptr().cast(),
                size as u32,
                &mut returned,
                ptr::null_mut(),
            )
        };
        if done != 0 {
            output.truncate(returned as usize);
            return Ok(output);
        }
        let error = last_error();
        let more = matches!(
            error.raw_os_error().map(|code| code as u32),
            Some(ERROR_INSUFFICIENT_BUFFER | ERROR_MORE_DATA)
        );
        if !more || size >= 1 << 20 {
            return Err(error);
        }
        size *= 4;
    }
}

fn guid_text(guid: &GUID) -> String {
    let d = guid.data4;
    format!(
        "{:08X}-{:04X}-{:04X}-{:02X}{:02X}-{:02X}{:02X}{:02X}{:02X}{:02X}{:02X}",
        guid.data1, guid.data2, guid.data3, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]
    )
}

fn same_guid(a: &GUID, b: &GUID) -> bool {
    (a.data1, a.data2, a.data3, a.data4) == (b.data1, b.data2, b.data3, b.data4)
}

fn drive_path(disk: u32) -> String {
    format!(r"\\.\PhysicalDrive{disk}")
}

/// What a disk's storage descriptor says.
struct Descriptor {
    removable: bool,
    bus: i32,
    name: String,
    serial: Option<String>,
}

impl Descriptor {
    /// What decides whether setup may erase the disk. A virtual disk (VHD)
    /// stands in for a USB, as a disk image does on a Mac; SD and MMC
    /// (`BusTypeSd`, `BusTypeMmc`) are other buses, removable by their
    /// media alone, as eMMC reports the same buses.
    fn facts(&self) -> DiskFacts {
        DiskFacts {
            bus: if self.bus == BusTypeUsb {
                Bus::Usb
            } else if self.bus == BusType1394 {
                Bus::FireWire
            } else if self.bus == BusTypeFileBackedVirtual {
                Bus::Image
            } else {
                Bus::Other
            },
            removable_media: self.removable,
            external: false,
        }
    }
}

fn descriptor(drive: &File) -> io::Result<Descriptor> {
    let mut query = [0u8; size_of::<STORAGE_PROPERTY_QUERY>()];
    let property = offset_of!(STORAGE_PROPERTY_QUERY, PropertyId);
    query[property..property + 4].copy_from_slice(&StorageDeviceProperty.to_le_bytes());
    let kind = offset_of!(STORAGE_PROPERTY_QUERY, QueryType);
    query[kind..kind + 4].copy_from_slice(&PropertyStandardQuery.to_le_bytes());
    let bytes = control(drive, IOCTL_STORAGE_QUERY_PROPERTY, &query, 1024)?;
    // SAFETY: the successful StorageDeviceProperty query returns this SDK
    // structure, including valid BOOLEAN fields; read checks its complete size.
    let header: STORAGE_DEVICE_DESCRIPTOR =
        unsafe { read(&bytes, 0) }.ok_or_else(|| io::Error::other("short storage descriptor"))?;
    let text = |offset: u32| -> Option<String> {
        let tail = bytes.get(usize::try_from(offset).ok().filter(|o| *o != 0)?..)?;
        let end = tail.iter().position(|b| *b == 0).unwrap_or(tail.len());
        Some(String::from_utf8_lossy(&tail[..end]).trim().to_owned()).filter(|t| !t.is_empty())
    };
    let name = [text(header.VendorIdOffset), text(header.ProductIdOffset)]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join(" ");
    Ok(Descriptor {
        removable: header.RemovableMedia,
        bus: header.BusType,
        name,
        serial: text(header.SerialNumberOffset),
    })
}

/// An NVMe disk's controller serial number, from its Identify Controller
/// data (bytes 4 to 23), as `stela.efi` reads it.
fn nvme_serial(drive: &File) -> Option<String> {
    const IDENTIFY_BYTES: usize = 4096;
    const CONTROLLER: u32 = 1;
    let header = offset_of!(STORAGE_PROPERTY_QUERY, AdditionalParameters);
    let data = size_of::<STORAGE_PROTOCOL_SPECIFIC_DATA>();
    let mut input = vec![0u8; header + data + IDENTIFY_BYTES];
    input[..4].copy_from_slice(&StorageAdapterProtocolSpecificProperty.to_le_bytes());
    input[4..8].copy_from_slice(&PropertyStandardQuery.to_le_bytes());
    // STORAGE_PROTOCOL_SPECIFIC_DATA is ten consecutive 32-bit fields.
    // Serialize them into initialized bytes rather than exposing C padding.
    const _: () = assert!(size_of::<STORAGE_PROTOCOL_SPECIFIC_DATA>() == 40);
    let fields = [
        ProtocolTypeNvme as u32,
        NVMeDataTypeIdentify as u32,
        CONTROLLER,
        0,
        data as u32,
        IDENTIFY_BYTES as u32,
        0,
        0,
        0,
        0,
    ];
    for (field, bytes) in fields
        .iter()
        .zip(input[header..header + data].as_chunks_mut::<4>().0)
    {
        bytes.copy_from_slice(&field.to_le_bytes());
    }
    let output = control(drive, IOCTL_STORAGE_QUERY_PROPERTY, &input, input.len()).ok()?;
    // SAFETY: this successful protocol query returns the all-integer SDK
    // descriptor; read also requires the complete structure to be present.
    let answer: STORAGE_PROTOCOL_DATA_DESCRIPTOR = unsafe { read(&output, 0) }?;
    let at = offset_of!(STORAGE_PROTOCOL_DATA_DESCRIPTOR, ProtocolSpecificData)
        + answer.ProtocolSpecificData.ProtocolDataOffset as usize;
    let serial = output.get(at + 4..at + 24)?;
    Some(String::from_utf8_lossy(serial).trim().to_owned()).filter(|s| !s.is_empty())
}

/// One partition-table entry.
struct Entry {
    number: u32,
    offset: u64,
    bytes: u64,
    content: String,
    guid: String,
}

fn layout(drive: &File) -> io::Result<Vec<Entry>> {
    let bytes = control(drive, IOCTL_DISK_GET_DRIVE_LAYOUT_EX, &[], 8192)?;
    // SAFETY: this successful IOCTL returns the SDK layout structure, with
    // valid Boolean fields and a union selected by its partition style.
    let header: DRIVE_LAYOUT_INFORMATION_EX =
        unsafe { read(&bytes, 0) }.ok_or_else(|| io::Error::other("short drive layout"))?;
    let first = offset_of!(DRIVE_LAYOUT_INFORMATION_EX, PartitionEntry);
    let mut entries = Vec::new();
    for index in 0..header.PartitionCount as usize {
        // SAFETY: PartitionEntry begins the IOCTL's array of SDK partition
        // records. read checks that this entire record fits in the response.
        let Some(entry) = (unsafe {
            read::<PARTITION_INFORMATION_EX>(
                &bytes,
                first + index * size_of::<PARTITION_INFORMATION_EX>(),
            )
        }) else {
            break;
        };
        if entry.PartitionNumber == 0 || entry.PartitionLength <= 0 {
            continue;
        }
        // SAFETY: the partition style says which member the union holds.
        let (content, guid) = unsafe {
            match entry.PartitionStyle {
                PARTITION_STYLE_GPT => {
                    let gpt = entry.Anonymous.Gpt;
                    let content = if same_guid(&gpt.PartitionType, &ESP) {
                        "EFI".into()
                    } else {
                        guid_text(&gpt.PartitionType)
                    };
                    (content, guid_text(&gpt.PartitionId))
                }
                PARTITION_STYLE_MBR => {
                    let kind = entry.Anonymous.Mbr.PartitionType;
                    let content = if kind == 0xef {
                        "EFI".into()
                    } else {
                        format!("{kind:#04x}")
                    };
                    (content, String::new())
                }
                _ => (String::new(), String::new()),
            }
        };
        entries.push(Entry {
            number: entry.PartitionNumber,
            offset: entry.StartingOffset as u64,
            bytes: entry.PartitionLength as u64,
            content,
            guid,
        });
    }
    Ok(entries)
}

/// A drive's size and logical sector size.
fn geometry(drive: &File) -> io::Result<(u64, u32)> {
    let bytes = control(drive, IOCTL_DISK_GET_DRIVE_GEOMETRY_EX, &[], 256)?;
    // SAFETY: this successful IOCTL returns the all-integer SDK geometry
    // structure; read checks that the complete response header is present.
    let geometry: DISK_GEOMETRY_EX =
        unsafe { read(&bytes, 0) }.ok_or_else(|| io::Error::other("short drive geometry"))?;
    Ok((
        geometry.DiskSize as u64,
        geometry.Geometry.BytesPerSector.max(512),
    ))
}

/// One mounted-or-not volume and where it lies.
struct WinVolume {
    /// `\\?\Volume{GUID}\`.
    name: String,
    disk: u32,
    offset: u64,
    paths: Vec<PathBuf>,
}

impl WinVolume {
    /// The volume's device, without the trailing backslash, to open.
    fn device(&self) -> &str {
        self.name.trim_end_matches('\\')
    }
}

/// The single disk extent of the volume `name`, `\\?\Volume{GUID}\`.
fn extent(name: &str) -> Option<DISK_EXTENT> {
    let volume = open(name.trim_end_matches('\\'), 0).ok()?;
    let bytes = control(&volume, IOCTL_VOLUME_GET_VOLUME_DISK_EXTENTS, &[], 256).ok()?;
    // SAFETY: the successful extent query returns this all-integer SDK
    // header and first extent; read checks that both fit in the response.
    let header: VOLUME_DISK_EXTENTS = unsafe { read(&bytes, 0) }?;
    // A volume spanning disks belongs to none of them.
    (header.NumberOfDiskExtents == 1).then_some(header.Extents[0])
}

fn volumes() -> Result<Vec<WinVolume>> {
    let mut volumes = Vec::new();
    let mut name = [0u16; 64];
    // SAFETY: name is a writable UTF-16 buffer of the supplied capacity;
    // Windows keeps no pointer after creating the independent search handle.
    let search = unsafe { FindFirstVolumeW(name.as_mut_ptr(), name.len() as u32) };
    if search == INVALID_HANDLE_VALUE {
        return Err(last_error()).context("cannot list this computer's volumes");
    }
    loop {
        let volume = narrow(&name);
        if let Some(extent) = extent(&volume) {
            volumes.push(WinVolume {
                paths: mount_paths(&volume),
                disk: extent.DiskNumber,
                offset: extent.StartingOffset as u64,
                name: volume,
            });
        }
        // SAFETY: search is open until enumeration ends, and name remains a
        // writable UTF-16 buffer of the supplied capacity.
        if unsafe { FindNextVolumeW(search, name.as_mut_ptr(), name.len() as u32) } == 0 {
            let error = last_error();
            // SAFETY: the search handle is open.
            unsafe { FindVolumeClose(search) };
            if error.raw_os_error() == Some(ERROR_NO_MORE_FILES as i32) {
                return Ok(volumes);
            }
            return Err(error).context("cannot list this computer's volumes");
        }
    }
}

/// Where volume `name` is mounted: drive letters and folders.
fn mount_paths(name: &str) -> Vec<PathBuf> {
    let volume = wide(name);
    let mut buffer = vec![0u16; 1024];
    let mut needed = 0u32;
    // SAFETY: volume is NUL-terminated and both output pointers remain
    // writable for the duration of this synchronous call.
    let done = unsafe {
        GetVolumePathNamesForVolumeNameW(
            volume.as_ptr(),
            buffer.as_mut_ptr(),
            buffer.len() as u32,
            &mut needed,
        )
    };
    if done == 0 {
        return Vec::new();
    }
    buffer
        .split(|c| *c == 0)
        .take_while(|path| !path.is_empty())
        .map(|path| PathBuf::from(narrow(path)))
        .collect()
}

/// The volume `path` is on, `\\?\Volume{GUID}\`, and its root.
fn volume_of(path: &Path) -> Result<(String, Vec<u16>)> {
    let path = std::path::absolute(path)?;
    let mut root = [0u16; 1024];
    // SAFETY: a NUL-terminated path and a buffer of the length given.
    if unsafe { GetVolumePathNameW(wide(&path).as_ptr(), root.as_mut_ptr(), root.len() as u32) }
        == 0
    {
        return Err(last_error()).with_context(|| format!("cannot inspect {}", path.display()));
    }
    let end = root.iter().position(|c| *c == 0).unwrap_or(root.len());
    let root = root[..=end].to_vec();
    let mut name = [0u16; 64];
    // SAFETY: the successful GetVolumePathNameW call NUL-terminated root;
    // name is a writable UTF-16 buffer with the supplied capacity.
    if unsafe {
        GetVolumeNameForVolumeMountPointW(root.as_ptr(), name.as_mut_ptr(), name.len() as u32)
    } == 0
    {
        return Err(last_error()).with_context(|| format!("cannot inspect {}", path.display()));
    }
    Ok((narrow(&name), root))
}

/// The disk number of the volume holding Windows.
fn system_disk() -> Result<u32> {
    let mut directory = [0u16; 512];
    // SAFETY: a buffer of the length given.
    let length =
        unsafe { GetSystemWindowsDirectoryW(directory.as_mut_ptr(), directory.len() as u32) };
    if length == 0 {
        return Err(last_error()).context("cannot locate the running Windows installation");
    }
    ensure!(
        (length as usize) < directory.len(),
        "the running Windows installation path exceeds the directory buffer"
    );
    let (volume, _) = volume_of(Path::new(&narrow(&directory)))
        .context("cannot identify the volume holding the running Windows installation")?;
    extent(&volume)
        .map(|extent| extent.DiskNumber)
        .context("cannot identify one physical disk holding the running Windows installation")
}

/// Bytes `[offset, offset + length)` of `drive`, read in whole sectors.
fn read_sectors(drive: &File, sector: u64, offset: u64, length: usize) -> Result<Vec<u8>> {
    let first = offset / sector * sector;
    let end = (offset + length as u64).div_ceil(sector) * sector;
    let mut bytes = vec![0u8; usize::try_from(end - first)?];
    let mut done = 0;
    while done < bytes.len() {
        let read = drive.seek_read(&mut bytes[done..], first + done as u64)?;
        ensure!(read > 0, "the disk ended at byte {}", first + done as u64);
        done += read;
    }
    let skip = usize::try_from(offset - first)?;
    Ok(bytes[skip..skip + length].to_vec())
}

/// Write `data` at `offset` of `drive`, merging partial sectors with what
/// they hold.
fn write_sectors(drive: &File, sector: u64, offset: u64, data: &[u8]) -> Result<()> {
    let first = offset / sector * sector;
    let end = (offset + data.len() as u64).div_ceil(sector) * sector;
    let bytes = if first == offset && end == offset + data.len() as u64 {
        data.to_vec()
    } else {
        let mut bytes = read_sectors(drive, sector, first, usize::try_from(end - first)?)?;
        let skip = usize::try_from(offset - first)?;
        bytes[skip..skip + data.len()].copy_from_slice(data);
        bytes
    };
    let mut done = 0;
    while done < bytes.len() {
        let written = drive.seek_write(&bytes[done..], first + done as u64)?;
        ensure!(
            written > 0,
            "the disk took no bytes at {}",
            first + done as u64
        );
        done += written;
    }
    drive.sync_all()?;
    Ok(())
}

impl System {
    /// Partition `identifier`'s disk, table entry and the disk's sector size.
    fn entry(&self, identifier: &str) -> Result<(u32, Entry, u64)> {
        let (disk, number) = partition_number(identifier)?;
        let drive = open(&drive_path(disk), 0)
            .with_context(|| format!("this computer has no disk {disk}"))?;
        let entry = layout(&drive)?
            .into_iter()
            .find(|entry| entry.number == number)
            .with_context(|| format!("disk {disk} has no partition {number}"))?;
        let (_, sector) = geometry(&drive)?;
        Ok((disk, entry, u64::from(sector)))
    }

    /// The volume on partition `identifier`, when Windows made one.
    fn volume_on(&self, identifier: &str) -> Result<Option<WinVolume>> {
        let (disk, entry, _) = self.entry(identifier)?;
        Ok(volumes()?
            .into_iter()
            .find(|volume| volume.disk == disk && volume.offset == entry.offset))
    }

    /// Open partition `identifier`'s drive for `access`; with its start,
    /// length and sector size.
    fn raw(&self, identifier: &str, access: u32) -> Result<(File, Entry, u64)> {
        let (disk, entry, sector) = self.entry(identifier)?;
        let drive = open(&drive_path(disk), access).with_context(|| {
            format!("cannot open disk {disk}; setup reads and writes it from {ADMIN}")
        })?;
        Ok((drive, entry, sector))
    }
}

/// The vendor and device IDs of the present PCI devices of setup class
/// `class`, which `what` names in an error.
fn pci_devices(class: &GUID, what: &str) -> Result<Vec<(u16, u16)>> {
    Ok(class_devices(class, what)?
        .iter()
        .filter_map(|(instance, _)| pci_device(instance))
        .collect())
}

/// The instance and compatible IDs of the present devices of setup class
/// `class`, which `what` names in an error.
fn class_devices(class: &GUID, what: &str) -> Result<Vec<(String, Vec<String>)>> {
    // SAFETY: `class` outlives the call; the other inputs are optional.
    let set = unsafe {
        SetupDiGetClassDevsW(class, std::ptr::null(), std::ptr::null_mut(), DIGCF_PRESENT)
    };
    ensure!(
        set != INVALID_HANDLE_VALUE as isize,
        "cannot list {what}: {}",
        std::io::Error::last_os_error()
    );
    let mut devices = Vec::new();
    for index in 0.. {
        // SAFETY: an all-zero SP_DEVINFO_DATA is valid plain data.
        let mut device: SP_DEVINFO_DATA = unsafe { std::mem::zeroed() };
        device.cbSize = size_of::<SP_DEVINFO_DATA>() as u32;
        // SAFETY: `set` is open and `device` declares its size.
        if unsafe { SetupDiEnumDeviceInfo(set, index, &mut device) } == 0 {
            break;
        }
        let mut id = [0u16; 512];
        let mut length = 0;
        // SAFETY: the buffer's length is passed with it, and `device`
        // came from `set`.
        let named = unsafe {
            SetupDiGetDeviceInstanceIdW(set, &device, id.as_mut_ptr(), id.len() as u32, &mut length)
        };
        if named == 0 {
            continue;
        }
        let end = (length as usize).saturating_sub(1).min(id.len());
        let instance = String::from_utf16_lossy(&id[..end]);
        let mut compatible = [0u16; 1024];
        let mut bytes = 0;
        // SAFETY: the buffer's length in bytes is passed with it, `device`
        // came from `set`, and the type output may be null.
        let read = unsafe {
            SetupDiGetDeviceRegistryPropertyW(
                set,
                &device,
                SPDRP_COMPATIBLEIDS,
                ptr::null_mut(),
                compatible.as_mut_ptr().cast(),
                size_of_val(&compatible) as u32,
                &mut bytes,
            )
        };
        // A REG_MULTI_SZ: strings each ended by a NUL, then another NUL.
        let compatible = if read != 0 {
            let units = (bytes as usize / 2).min(compatible.len());
            compatible[..units]
                .split(|&unit| unit == 0)
                .filter(|id| !id.is_empty())
                .map(String::from_utf16_lossy)
                .collect()
        } else {
            Vec::new()
        };
        devices.push((instance, compatible));
    }
    // SAFETY: `set` is open and not used again.
    unsafe { SetupDiDestroyDeviceInfoList(set) };
    Ok(devices)
}

impl Host for System {
    fn volume(&self, path: &Path) -> Result<Volume> {
        let (name, root) = volume_of(path)?;
        let mut filesystem = [0u16; 64];
        // SAFETY: a NUL-terminated root and a buffer of the length given.
        let done = unsafe {
            GetVolumeInformationW(
                root.as_ptr(),
                ptr::null_mut(),
                0,
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                filesystem.as_mut_ptr(),
                filesystem.len() as u32,
            )
        };
        if done == 0 {
            return Err(last_error()).with_context(|| format!("cannot inspect {}", path.display()));
        }
        let mut free = 0u64;
        // SAFETY: root is NUL-terminated and remains live; free is writable
        // through this synchronous call, and the other outputs may be null.
        if unsafe {
            GetDiskFreeSpaceExW(root.as_ptr(), &mut free, ptr::null_mut(), ptr::null_mut())
        } == 0
        {
            return Err(last_error()).with_context(|| format!("cannot inspect {}", path.display()));
        }
        let filesystem = match narrow(&filesystem).as_str() {
            "FAT32" | "FAT" | "FAT16" | "FAT12" => FileSystem::Fat,
            "exFAT" => FileSystem::ExFat,
            "NTFS" => FileSystem::Ntfs,
            other => FileSystem::Other(other.into()),
        };
        Ok(Volume {
            filesystem,
            free_bytes: free,
            disk: extent(&name).map(|extent| format!("disk{}", extent.DiskNumber)),
        })
    }

    fn disks(&self) -> Result<Vec<Disk>> {
        let volumes = volumes()?;
        let system = system_disk()?;
        let mut disks = Vec::new();
        for number in 0..DRIVES {
            let Ok(drive) = open(&drive_path(number), 0) else {
                continue;
            };
            let Ok(descriptor) = descriptor(&drive) else {
                continue;
            };
            let entries = layout(&drive).unwrap_or_default();
            disks.push(Disk {
                identifier: format!("disk{number}"),
                bytes: geometry(&drive).map_or(0, |(bytes, _)| bytes),
                removal: descriptor.facts().removal(),
                name: descriptor.name,
                system: system == number,
                partitions: entries
                    .into_iter()
                    .map(|entry| Partition {
                        identifier: format!("disk{number}p{}", entry.number),
                        mount: volumes
                            .iter()
                            .find(|v| v.disk == number && v.offset == entry.offset)
                            .and_then(|v| v.paths.first().cloned()),
                        content: entry.content,
                    })
                    .collect(),
            });
        }
        Ok(disks)
    }

    fn partition(&self, identifier: &str) -> Result<PartitionInfo> {
        let (disk, entry, _) = self.entry(identifier)?;
        let mount = volumes()?
            .into_iter()
            .find(|volume| volume.disk == disk && volume.offset == entry.offset)
            .and_then(|volume| volume.paths.into_iter().next());
        Ok(PartitionInfo {
            identifier: identifier.into(),
            disk: format!("disk{disk}"),
            guid: entry.guid,
            bytes: entry.bytes,
            content: entry.content,
            mount,
            container: None,
        })
    }

    fn serial(&self, disk: &str) -> Result<Option<String>> {
        let number = disk_number(disk)?;
        let drive = open(&drive_path(number), 0)
            .with_context(|| format!("this computer has no disk {number}"))?;
        let descriptor = descriptor(&drive)?;
        Ok(if descriptor.bus == BusTypeNvme {
            // The storage descriptor's NVMe serial may be the namespace's
            // EUI-64 instead; the controller's own comes from Identify.
            nvme_serial(&drive)
        } else if descriptor.bus == BusTypeFileBackedVirtual {
            None
        } else {
            descriptor.serial
        })
    }

    fn read(&self, identifier: &str, offset: u64, length: usize) -> Result<Vec<u8>> {
        let (drive, entry, sector) = self.raw(identifier, GENERIC_READ)?;
        ensure!(
            offset + length as u64 <= entry.bytes,
            "{identifier} ends before byte {}",
            offset + length as u64
        );
        read_sectors(&drive, sector, entry.offset + offset, length)
    }

    fn write(&self, identifier: &str, offset: u64, bytes: &[u8]) -> Result<()> {
        let (drive, entry, sector) = self.raw(identifier, GENERIC_READ | GENERIC_WRITE)?;
        ensure!(
            offset + bytes.len() as u64 <= entry.bytes,
            "{identifier} ends before byte {}",
            offset + bytes.len() as u64
        );
        write_sectors(&drive, sector, entry.offset + offset, bytes)
            .with_context(|| format!("cannot write {identifier} (is a volume on it mounted?)"))
    }

    fn unmount(&self, partition: &PartitionInfo) -> Result<()> {
        let Some(volume) = self.volume_on(&partition.identifier)? else {
            return Ok(());
        };
        for path in &volume.paths {
            // SAFETY: a NUL-terminated mount path.
            if unsafe { DeleteVolumeMountPointW(wide(path).as_ptr()) } == 0 {
                return Err(last_error())
                    .with_context(|| format!("cannot unmount {} ({ADMIN})", path.display()));
            }
        }
        let handle = open(volume.device(), GENERIC_READ | GENERIC_WRITE)
            .with_context(|| format!("cannot open {} ({ADMIN})", partition.identifier))?;
        control(&handle, FSCTL_DISMOUNT_VOLUME, &[], 0)
            .with_context(|| format!("cannot dismount {}", partition.identifier))?;
        Ok(())
    }

    fn mount(&self, identifier: &str) -> Result<PathBuf> {
        let volume = self
            .volume_on(identifier)?
            .with_context(|| format!("Windows has no volume on {identifier}"))?;
        if let Some(path) = volume.paths.first() {
            return Ok(path.clone());
        }
        // SAFETY: no arguments.
        let used = unsafe { GetLogicalDrives() };
        let letter = (b'D'..=b'Z')
            .find(|letter| used & (1 << (letter - b'A')) == 0)
            .context("every drive letter is taken")?;
        let path = format!("{}:\\", letter as char);
        // SAFETY: two NUL-terminated paths.
        if unsafe { SetVolumeMountPointW(wide(&path).as_ptr(), wide(&volume.name).as_ptr()) } == 0 {
            return Err(last_error())
                .with_context(|| format!("cannot mount {identifier} at {path} ({ADMIN})"));
        }
        Ok(PathBuf::from(path))
    }

    fn erase(&self, disk: &str, name: &str) -> Result<()> {
        let number = disk_number(disk)?;
        // Every volume on the disk, locked and dismounted until Windows has
        // read the new partition table.
        let mut locked = Vec::new();
        for volume in volumes()?.into_iter().filter(|v| v.disk == number) {
            let handle = open(volume.device(), GENERIC_READ | GENERIC_WRITE)
                .with_context(|| format!("cannot open a volume of {disk} ({ADMIN})"))?;
            control(&handle, FSCTL_LOCK_VOLUME, &[], 0).with_context(|| {
                format!(
                    "a volume of {disk} ({}) is in use; close what has it open",
                    volume
                        .paths
                        .first()
                        .map_or(volume.name.clone(), |p| p.display().to_string())
                )
            })?;
            control(&handle, FSCTL_DISMOUNT_VOLUME, &[], 0)?;
            locked.push(handle);
        }
        let mut drive = open(&drive_path(number), GENERIC_READ | GENERIC_WRITE)
            .with_context(|| format!("cannot open {disk} to erase it ({ADMIN})"))?;
        let (bytes, sector) = geometry(&drive)?;
        let layout = crate::setup::format::layout(bytes, sector, name, &mut |bytes| {
            getrandom::getrandom(bytes).context("the system's random generator failed")
        })?;
        super::write_layout(&mut drive, &layout)?;
        control(&drive, IOCTL_DISK_UPDATE_PROPERTIES, &[], 0)
            .with_context(|| format!("{disk} was erased, but Windows did not reread it"))?;
        drop(drive);
        drop(locked);
        // Windows mounts the new volumes as it finds them.
        let data = format!("{disk}p2");
        let mut last = None;
        for _ in 0..100 {
            match self.mount(&data) {
                Ok(_) => return Ok(()),
                Err(error) => last = Some(error),
            }
            std::thread::sleep(std::time::Duration::from_millis(200));
        }
        Err(last.expect("tried"))
            .context(format!("{disk} was erased, but its volume is not mounted"))
    }

    fn flush(&self, path: &Path) -> Result<()> {
        // FlushFileBuffers on a volume handle writes out every file of the
        // volume; it takes write access, which an administrator has.
        let (name, _) = volume_of(path)?;
        let volume = open(name.trim_end_matches('\\'), GENERIC_READ | GENERIC_WRITE)
            .with_context(|| format!("cannot open the volume of {} ({ADMIN})", path.display()))?;
        volume
            .sync_all()
            .with_context(|| format!("cannot write out {}", path.display()))
    }

    fn check_erase_access(&self, disk: &str) -> Result<()> {
        open(
            &drive_path(disk_number(disk)?),
            GENERIC_READ | GENERIC_WRITE,
        )
        .with_context(|| format!("USB access requires {ADMIN}"))?;
        Ok(())
    }

    fn whole_disk(&self, usb: &Path) -> Result<Option<String>> {
        if usb.is_dir() {
            return Ok(None);
        }
        let Some(identifier) = usb.to_str().and_then(drive_names::whole_disk) else {
            return Ok(None);
        };
        Ok(self
            .disks()?
            .into_iter()
            .any(|disk| disk.identifier == identifier)
            .then_some(identifier))
    }

    fn cpu(&self) -> Result<Cpu> {
        super::cpu()
    }

    fn nvidia_gpus(&self) -> Result<Vec<u16>> {
        // The display device class, which an NVIDIA GPU's audio function is
        // not in.
        const DISPLAY: GUID = GUID::from_u128(0x4d36e968_e325_11ce_bfc1_08002be10318);
        Ok(pci_devices(&DISPLAY, "display devices")?
            .into_iter()
            .filter(|&(vendor, _)| vendor == 0x10de)
            .map(|(_, device)| device)
            .collect())
    }

    fn network_adapters(&self) -> Result<Vec<(u16, u16)>> {
        // This device class includes wired and wireless adapters.
        const NET: GUID = GUID::from_u128(0x4d36e972_e325_11ce_bfc1_08002be10318);
        pci_devices(&NET, "network adapters")
    }
}
