//! Firmware-only PCI_IO observation. This supplies no native hardware grant.
//! Provenance: UEFI 2.11 chapter14, sections14.4.1/.8/.16/.18.
//! https://uefi.org/specs/UEFI/2.11/14_Protocols_PCI_Bus_Support.html
//! All callbacks and returned pool storage must end before ExitBootServices.
use alloc::vec::Vec;
use core::{
    ffi::c_void,
    mem::{offset_of, size_of},
    ptr::{self, NonNull},
};
use uefi::{
    Handle, Status,
    boot::{self, OpenProtocolAttributes, OpenProtocolParams, ScopedProtocol},
    proto::{
        ProtocolPointer,
        device_path::{DevicePath, DeviceSubType, DeviceType},
        unsafe_protocol,
    },
};

// Unused slots are opaque pointer-sized fields, never invoked. They preserve
// the protocol's exact EFI_PCI_IO_PROTOCOL_ACCESS pairs and vtable ordering.
type Read = unsafe extern "efiapi" fn(*mut PciIo, u32, u32, usize, *mut c_void) -> Status;
/// `EFI_PCI_IO_PROTOCOL_IO_MEM`: width, BAR, offset in it, count, buffer.
type MemRead = unsafe extern "efiapi" fn(*mut PciIo, u32, u8, u64, usize, *mut c_void) -> Status;
type Location =
    unsafe extern "efiapi" fn(*mut PciIo, *mut usize, *mut usize, *mut usize, *mut usize) -> Status;
type BarAttributes =
    unsafe extern "efiapi" fn(*mut PciIo, u8, *mut u64, *mut *mut c_void) -> Status;
#[repr(C)]
#[unsafe_protocol("4cf5b200-68b8-4ca5-9eec-b23e3f50029a")]
pub struct PciIo {
    poll_mem_poll_io: [usize; 2],
    mem_read: MemRead,
    mem_write_io: [usize; 3],
    config_read: Read,
    config_write_copy_map_unmap_alloc_free_flush: [usize; 7],
    location: Location,
    attributes: usize,
    bar_attributes: BarAttributes,
    set_bar_attributes: usize,
    rom_size: u64,
    rom_image: *mut c_void,
}
// This binary is x86-64 only; EFIAPI uses that target's UEFI ABI.
const _: () = {
    assert!(size_of::<usize>() == 8);
    assert!(size_of::<PciIo>() == 160);
    assert!(offset_of!(PciIo, mem_read) == 16);
    assert!(offset_of!(PciIo, config_read) == 48);
    assert!(offset_of!(PciIo, location) == 112);
    assert!(offset_of!(PciIo, bar_attributes) == 128);
    assert!(offset_of!(PciIo, rom_size) == 144);
};
use stela_drivers::bus::pci::Address;
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Error {
    Firmware(Status),
    Location,
    BarIndex,
    MissingResources,
    DescriptorTag(u8),
    DescriptorLength,
    DescriptorRange,
    Descriptor(DescriptorError),
    Checksum,
    Allocation,
    Release(Status),
}
impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Firmware(status) => write!(f, "firmware status {status:?}"),
            Self::Location => f.write_str("invalid firmware PCI location"),
            Self::BarIndex => f.write_str("invalid BAR index"),
            Self::MissingResources => f.write_str("successful BAR call returned no pool buffer"),
            Self::DescriptorTag(tag) => write!(f, "unsupported descriptor tag {tag:#x}"),
            Self::DescriptorLength => f.write_str("invalid descriptor wire length"),
            Self::DescriptorRange => f.write_str("descriptor span overflow"),
            Self::Descriptor(error) => write!(f, "resource descriptor refused: {error:?}"),
            Self::Checksum => f.write_str("resource descriptor checksum mismatch"),
            Self::Allocation => f.write_str("resource observation allocation failed"),
            Self::Release(status) => write!(f, "resource pool release failed: {status:?}"),
        }
    }
}
pub use stela_boot_core::firmware_pci::BarRegion;
use stela_boot_core::firmware_pci::{DescriptorError, FramebufferBar, decode_qword};

/// Original GOP association copied while both firmware protocols are live.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FramebufferSource {
    pub address: Address,
    pub bar: FramebufferBar,
}

fn get_protocol<P: ProtocolPointer + ?Sized>(handle: Handle) -> uefi::Result<ScopedProtocol<P>> {
    // SAFETY: a scoped read-only protocol observation during boot services.
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
}

fn gop_pci(handle: Handle) -> Option<ScopedProtocol<PciIo>> {
    if let Ok(pci) = get_protocol(handle) {
        return Some(pci);
    }
    let path = get_protocol::<DevicePath>(handle).ok()?;
    if path.instance_iter().count() != 1 {
        return None;
    }
    let mut remaining = &*path;
    let controller = boot::locate_device_path::<PciIo>(&mut remaining).ok()?;
    // GOP may be on a child framebuffer/output path of the PCI controller.
    // An unconsumed PCI node means the returned handle is only an ancestor.
    if !remaining
        .node_iter()
        .all(|node| match (node.device_type(), node.sub_type()) {
            (DeviceType::ACPI, DeviceSubType::ACPI_ADR) => {
                node.length() >= 8 && node.length().is_multiple_of(4)
            }
            (DeviceType::HARDWARE, DeviceSubType::HARDWARE_CONTROLLER) => node.length() == 8,
            _ => false,
        })
    {
        return None;
    }
    get_protocol(controller).ok()
}

pub(crate) fn framebuffer_source(
    handle: Handle,
    base: u64,
    bytes: u64,
) -> Result<Option<FramebufferSource>, Error> {
    let Some(mut pci) = gop_pci(handle) else {
        return Ok(None);
    };
    let address = pci.address()?;
    let header = pci.read_header()?;
    let end = base.checked_add(bytes).ok_or(Error::DescriptorRange)?;
    let mut bars = Vec::new();
    bars.try_reserve_exact(6).map_err(|_| Error::Allocation)?;
    for index in 0..6 {
        match pci.bar_regions(index) {
            Ok(regions) => bars.push((index, regions)),
            Err(Error::Firmware(Status::UNSUPPORTED)) => {}
            Err(error) => return Err(error),
        }
    }
    Ok(FramebufferBar::observe(
        &header,
        bars.iter().filter_map(|(index, regions)| {
            let [region] = regions.as_slice() else {
                return None;
            };
            Some((*index, region))
        }),
        base..end,
    )
    .map(|bar| FramebufferSource { address, bar }))
}
/// The Bochs-compatible virtual display device (PCI 1234:1111, QEMU's
/// standard VGA, `hw/display/vga-pci.c`), vendor and device. No PC graphics
/// card has this ID, so the paths below run only on a virtual machine.
const BOCHS_DISPLAY: u32 = 0x1111_1234;
/// Its MMIO BAR, whose first bytes hold the EDID with `edid=on`
/// (QEMU's `docs/specs/standard-vga.rst`: 0x0000..0x0400 "edid data blob").
const DISPLAY_MMIO_BAR: u8 = 2;
const EDID_BYTES: usize = 128;

/// The EDID base block the virtual display device on GOP `handle` serves in its
/// registers, which OVMF reads to pick its first mode but does not install
/// as a protocol. `None` for another device or when the read fails; the
/// caller checks the block.
pub(crate) fn bochs_display_edid(handle: Handle) -> Option<Vec<u8>> {
    let mut pci = gop_pci(handle)?;
    let header = pci.read_header().ok()?;
    if u32::from_le_bytes(header[..4].try_into().ok()?) != BOCHS_DISPLAY {
        return None;
    }
    let mut edid = Vec::new();
    edid.try_reserve_exact(EDID_BYTES).ok()?;
    edid.resize(EDID_BYTES, 0);
    // SAFETY: live boot protocol; EfiPciIoWidthUint8 (0) reads `EDID_BYTES`
    // single bytes from the BAR into the writable buffer of that length.
    let read = unsafe {
        (pci.mem_read)(
            &mut *pci,
            0,
            DISPLAY_MMIO_BAR,
            0,
            EDID_BYTES,
            edid.as_mut_ptr().cast(),
        )
    };
    read.is_success().then_some(edid)
}

/// The Bochs DISPI registers in the virtual display's MMIO BAR, 16 bits
/// each (QEMU's `docs/specs/standard-vga.rst`: 0x0500..0x0516), by index
/// (`include/hw/display/bochs-vbe.h`).
pub(crate) const DISPI_REGISTERS: u64 = 0x500;
pub(crate) const DISPI_VIRT_WIDTH: u64 = 6;
pub(crate) const DISPI_X_OFFSET: u64 = 8;
pub(crate) const DISPI_Y_OFFSET: u64 = 9;
const DISPI_VIDEO_MEMORY_64K: u64 = 0xa;

/// The physical address of the DISPI registers of the virtual display on
/// GOP `handle`, when its screen at `base` starts its video memory, shows
/// rows of `stride` pixels from its top and the video memory holds a second
/// frame of `frame_bytes` after it: the screen can then show either.
pub(crate) fn bochs_display_flip(
    handle: Handle,
    base: u64,
    stride: usize,
    frame_bytes: u64,
) -> Option<u64> {
    let mut pci = gop_pci(handle)?;
    let header = pci.read_header().ok()?;
    if u32::from_le_bytes(header[..4].try_into().ok()?) != BOCHS_DISPLAY {
        return None;
    }
    let single = |regions: Vec<BarRegion>| match regions.as_slice() {
        [region] => Some(region.host.clone()),
        _ => None,
    };
    let memory = single(pci.bar_regions(0).ok()?)?;
    let registers = single(pci.bar_regions(DISPLAY_MMIO_BAR).ok()?)?;
    let mut register = |index: u64| {
        let mut value = 0u16;
        // SAFETY: live boot protocol; EfiPciIoWidthUint16 (1) reads one
        // register into the local.
        let read = unsafe {
            (pci.mem_read)(
                &mut *pci,
                1,
                DISPLAY_MMIO_BAR,
                DISPI_REGISTERS + index * 2,
                1,
                ptr::from_mut(&mut value).cast(),
            )
        };
        read.is_success().then_some(value)
    };
    let video_memory = u64::from(register(DISPI_VIDEO_MEMORY_64K)?) << 16;
    let fits = memory.start == base
        && 2 * frame_bytes <= video_memory.min(memory.end - memory.start)
        && usize::from(register(DISPI_VIRT_WIDTH)?) == stride
        && register(DISPI_X_OFFSET)? == 0
        && register(DISPI_Y_OFFSET)? == 0
        && registers.end - registers.start >= DISPI_REGISTERS + 0x20;
    fits.then_some(registers.start + DISPI_REGISTERS)
}

fn status(value: Status) -> Result<(), Error> {
    if value.is_success() {
        Ok(())
    } else {
        Err(Error::Firmware(value))
    }
}
impl PciIo {
    // Only a live ScopedProtocol obtained from firmware can construct a
    // usable borrow; no copy of this callback-bearing object enters a Handoff.
    pub fn address(&mut self) -> Result<Address, Error> {
        let (mut segment, mut bus, mut device, mut function) = (0, 0, 0, 0);
        // SAFETY: original scoped firmware protocol, EFIAPI signature,
        // writable local UINTN outputs; boot services remain active.
        status(unsafe {
            (self.location)(self, &mut segment, &mut bus, &mut device, &mut function)
        })?;
        if device >= 32 || function >= 8 {
            return Err(Error::Location);
        }
        Ok(Address {
            segment: segment.try_into().map_err(|_| Error::Location)?,
            bus: bus.try_into().map_err(|_| Error::Location)?,
            device: device as u8,
            function: function as u8,
        })
    }
    pub fn read_header(&mut self) -> Result<[u8; 64], Error> {
        let mut header = [0; 64];
        // Width0=EfiPciIoWidthUint8, count64 bytes; no raw ECAM access.
        // SAFETY: live boot protocol and bounded writable local output; the
        // firmware performs its own controller/configuration addressing.
        status(unsafe {
            (self.config_read)(self, 0, 0, header.len(), header.as_mut_ptr().cast())
        })?;
        Ok(header)
    }
    pub fn bar_regions(&mut self, index: u8) -> Result<Vec<BarRegion>, Error> {
        if index >= 6 {
            return Err(Error::BarIndex);
        }
        let mut raw = ptr::null_mut();
        // SAFETY: live EFIAPI protocol; Resources is a writable VOID**.
        // Supports is optional and unused. A successful response is a firmware
        // pool allocation containing a complete EndTag-terminated resource list.
        status(unsafe { (self.bar_attributes)(self, index, ptr::null_mut(), &mut raw) })?;
        let raw = NonNull::new(raw.cast::<u8>()).ok_or(Error::MissingResources)?;
        // SAFETY: successful protocol contract supplies the original complete
        // descriptor allocation. Only exact protocol-described records are read.
        let result = unsafe { copy_regions(raw) };
        // FreePool happens even for a refused descriptor; explicit failure is
        // returned instead of hiding release status in an infallible Drop.
        // SAFETY: original successful pool allocation, copied records own all
        // returned DATA and hold no firmware pointers. This is called once even
        // when copy_regions refuses the descriptor. On FreePool failure the
        // pool remains firmware-owned and caller must stop preparation.
        unsafe { uefi::boot::free_pool(raw) }.map_err(|error| Error::Release(error.status()))?;
        result
    }
}
/// # Safety
/// `base` is the original successful GetBarAttributes pool allocation and
/// contains its complete valid readable EndTag-terminated descriptor chain.
/// Firmware callback lifetime is still active. No synthetic slice length is
/// invented; malformed headers are refused before reading a descriptor body.
unsafe fn copy_regions(base: NonNull<u8>) -> Result<Vec<BarRegion>, Error> {
    let mut result = Vec::new();
    let mut offset = 0usize;
    let mut checksum = 0u8;
    loop {
        // SAFETY: caller keeps the protocol's complete descriptor chain;
        // offset advances only by checked lengths of specified descriptors.
        let tag = unsafe { base.as_ptr().add(offset).read() };
        match tag {
            0x79 => {
                let end = offset
                    .checked_add(2)
                    .filter(|n| *n <= isize::MAX as usize)
                    .ok_or(Error::DescriptorRange)?;
                // SAFETY: the small EndTag record has its checksum byte.
                let value = unsafe { base.as_ptr().add(end - 1).read() };
                if value != 0 && checksum.wrapping_add(tag).wrapping_add(value) != 0 {
                    return Err(Error::Checksum);
                }
                return Ok(result);
            }
            0x8a => {
                let end = offset
                    .checked_add(46)
                    .filter(|n| *n <= isize::MAX as usize)
                    .ok_or(Error::DescriptorRange)?;
                // SAFETY: the large-record header is part of the chain.
                let length = unsafe {
                    u16::from_le_bytes([
                        base.as_ptr().add(offset + 1).read(),
                        base.as_ptr().add(offset + 2).read(),
                    ])
                };
                if length != 43 {
                    return Err(Error::DescriptorLength);
                }
                let mut bytes = [0; 46];
                // SAFETY: exact standard-sized record after validated header;
                // local output is distinct and has exactly46 writable bytes.
                unsafe {
                    ptr::copy_nonoverlapping(
                        base.as_ptr().add(offset),
                        bytes.as_mut_ptr(),
                        bytes.len(),
                    );
                }
                let observed = decode_qword(&bytes).map_err(Error::Descriptor)?;
                result.try_reserve(1).map_err(|_| Error::Allocation)?;
                result.push(observed);
                checksum = bytes
                    .iter()
                    .fold(checksum, |sum, byte| sum.wrapping_add(*byte));
                offset = end;
            }
            other => return Err(Error::DescriptorTag(other)),
        }
    }
}

/// No handles/protocol pointers escape this firmware-only census. All native
/// functions (including those without a firmware handle) must still be found
/// through the later root configuration factory and identity rechecked.
pub struct Census {
    pub functions: Vec<FunctionObservation>,
    pub failures: Vec<CaptureFailure>,
}
pub struct FunctionObservation {
    pub address: Address,
    pub header: [u8; 64],
    // A missing/unimplemented BAR is preserved as the firmware status;
    // it is not turned into an empty device or a guessed identity translation.
    pub bars: [Result<Vec<BarRegion>, Error>; 6],
}
impl FunctionObservation {
    /// Copied firmware fields, to be revalidated under a real native owner.
    pub fn identity(&self) -> stela_drivers::bus::pci::Identity {
        let word = |offset| u16::from_le_bytes([self.header[offset], self.header[offset + 1]]);
        stela_drivers::bus::pci::Identity {
            address: self.address,
            vendor: word(0),
            device: word(2),
            class: [self.header[11], self.header[10], self.header[9]],
        }
    }
}

/// Diagnostic snapshot through the firmware's exact controller handle. It
/// supplies neither a controller-stop receipt nor native access permission.
#[cfg(feature = "diagnostics")]
fn observe_nvme(
    handle: Handle,
    protocol: &mut PciIo,
    address: Address,
    header: &[u8; 64],
    bar0: &Result<Vec<BarRegion>, Error>,
) {
    use stela_drivers::{bus::pci::regs, storage::nvme};
    use uefi::proto::nvme::pass_thru::NvmePassThru;

    if [header[11], header[10], header[9]] != nvme::CLASS {
        return;
    }
    let protocol_test = boot::test_protocol::<NvmePassThru>(OpenProtocolParams {
        handle,
        agent: boot::image_handle(),
        controller: None,
    });
    let pass_thru = match &protocol_test {
        Ok(true) => "present",
        Ok(false) => "absent",
        Err(_) => "unavailable",
    };
    let command = u16::from_le_bytes([header[4], header[5]]);
    let mse = command & regs::COMMAND_MEMORY != 0;
    let bme = command & regs::COMMAND_MASTER != 0;
    let registers = if !matches!(protocol_test, Ok(true)) {
        Err("no-pass-thru")
    } else if !mse {
        Err("decode-off")
    } else if !bar0
        .as_ref()
        .is_ok_and(|regions| nvme_bar0(header, regions))
    {
        Err("bar0-unavailable")
    } else {
        let mut read = |offset| {
            let mut value = 0u32;
            // SAFETY: the live firmware protocol and assigned, decoding BAR0
            // cover this aligned 32-bit register and the local output is writable.
            status(unsafe {
                (protocol.mem_read)(protocol, 2, 0, offset, 1, ptr::from_mut(&mut value).cast())
            })?;
            Ok::<_, Error>(value)
        };
        read(0x14)
            .and_then(|cc| read(0x1c).map(|csts| (cc, csts)))
            .map_err(|_| "read-refused")
    };
    match registers {
        Ok((cc, csts)) => crate::output::arguments(format_args!(
            "FW NVME pci={address} pass-thru={pass_thru} mse={mse} bme={bme} cc={cc:#010x} csts={csts:#010x}\n"
        )),
        Err(reason) => crate::output::arguments(format_args!(
            "FW NVME pci={address} pass-thru={pass_thru} mse={mse} bme={bme} registers={reason}\n"
        )),
    }
}

/// Read-only Gemini Lake I²C snapshot. Protocol presence is checked on the
/// exact PCI handle. This census supplies no stop receipt.
#[cfg(feature = "diagnostics")]
fn observe_lpss_i2c(
    handle: Handle,
    address: Address,
    header: &[u8; 64],
    bar0: &Result<Vec<BarRegion>, Error>,
) {
    use stela_boot_core::firmware_pci::lpss::{self, Bar0};
    use stela_drivers::bus::pci::regs;

    let word = |offset| u32::from_le_bytes(header[offset..offset + 4].try_into().unwrap());
    let identity = word(0);
    if !lpss::matches(identity) {
        return;
    }
    let vendor = identity as u16;
    let device = identity >> 16;
    // UEFI PI 1.8, §17.2.2 and §17.2.7. Inspect GUIDs only and do not
    // open the interfaces or infer driver binding or child handle state.
    const I2C_MASTER: uefi::Guid = uefi::guid!("cd72881f-45b5-4feb-98c8-313da8117462");
    const I2C_HOST: uefi::Guid = uefi::guid!("a5aab9e3-c727-48cd-8bbf-427233854948");
    let (master, host) = match boot::protocols_per_handle(handle) {
        Ok(guids) => {
            let presence = |protocol: &uefi::Guid| {
                if guids.iter().any(|&guid| guid == protocol) {
                    "present"
                } else {
                    "absent"
                }
            };
            (presence(&I2C_MASTER), presence(&I2C_HOST))
        }
        Err(_) => ("unavailable", "unavailable"),
    };
    let command = u16::from_le_bytes([header[4], header[5]]);
    let ioe = command & regs::COMMAND_IO != 0;
    let mse = command & regs::COMMAND_MEMORY != 0;
    let bme = command & regs::COMMAND_MASTER != 0;
    let intx_disabled = command & regs::COMMAND_INTX_DISABLE != 0;
    let advertised = match bar0 {
        Ok(regions) if !regions.is_empty() => "advertised",
        Ok(_) | Err(Error::Firmware(Status::UNSUPPORTED)) => "not-advertised",
        Err(_) => "unavailable",
    };
    crate::output::arguments(format_args!(
        "FW LPSS-I2C pci={address} id={vendor:04x}:{device:04x} command={command:#06x} \
         ioe={ioe} mse={mse} bme={bme} intx-disabled={intx_disabled} bar0={advertised} \
         pci-handle-i2c-master={master} pci-handle-i2c-host={host} stop-evidence=unavailable\n"
    ));
    let region = Bar0::observe(word(0x0c), word(0x10), word(0x14))
        .ok_or("raw-bar0-refused")
        .and_then(|bar| {
            let regions = bar0.as_ref().map_err(|_| "descriptor-unavailable")?;
            bar.firmware_region(regions).ok_or("descriptor-mismatch")
        });
    match region {
        Ok(region) => crate::output::arguments(format_args!(
            "FW LPSS-I2C BAR0 pci={address} id={vendor:04x}:{device:04x} bus={:#x} host={:#x} bytes={}\n",
            region.pci.start,
            region.host.start,
            region.pci.end - region.pci.start
        )),
        Err(state) => crate::output::arguments(format_args!(
            "FW LPSS-I2C BAR0 pci={address} id={vendor:04x}:{device:04x} state={state}\n"
        )),
    }
}

#[cfg(feature = "diagnostics")]
fn nvme_bar0(header: &[u8; 64], regions: &[BarRegion]) -> bool {
    let [region] = regions else {
        return false;
    };
    let low = u32::from_le_bytes(header[0x10..0x14].try_into().unwrap());
    let high = u32::from_le_bytes(header[0x14..0x18].try_into().unwrap());
    let (high, bits) = match low & 7 {
        0 => (0, 32),
        4 => (high, 64),
        _ => return false,
    };
    let base = u64::from(low & !15) | u64::from(high) << 32;
    header[0x0e] & 0x7f == 0
        && region.resource_type == 0
        && region.granularity == bits
        && base != 0
        && region.pci.start == base
        && region.pci.end - base >= 0x20
}

#[derive(Debug)]
pub enum CaptureFailure {
    Enumeration(Status),
    Open(Status),
    Function(Error),
    Allocation,
}
pub fn capture() -> Result<Census, CaptureFailure> {
    capture_with_cnvi(false)
}

#[cfg(feature = "diagnostics")]
pub fn capture_host_check() -> Result<Census, CaptureFailure> {
    capture_with_cnvi(true)
}

fn capture_with_cnvi(
    #[cfg_attr(not(feature = "diagnostics"), allow(unused_variables))] observe_cnvi_csr: bool,
) -> Result<Census, CaptureFailure> {
    use uefi::boot::{self, OpenProtocolAttributes, OpenProtocolParams};
    let handles = boot::find_handles::<PciIo>()
        .map_err(|error| CaptureFailure::Enumeration(error.status()))?;
    let mut result = Census {
        functions: Vec::new(),
        failures: Vec::new(),
    };
    // Reserve before starting callbacks, so an allocation refusal never leaves
    // a protocol, returned pool allocation or callback-bearing token kept.
    result
        .functions
        .try_reserve_exact(handles.len())
        .map_err(|_| CaptureFailure::Allocation)?;
    result
        .failures
        .try_reserve_exact(handles.len())
        .map_err(|_| CaptureFailure::Allocation)?;
    for handle in handles {
        // SAFETY: GetProtocol is the firmware's scoped instance. This
        // consumer only observes the function; the firmware keeps controller
        // driving and reset permission throughout this loop.
        // All borrowed callbacks and returned pool storage end in this loop.
        let mut protocol = match unsafe {
            boot::open_protocol::<PciIo>(
                OpenProtocolParams {
                    handle,
                    agent: boot::image_handle(),
                    controller: None,
                },
                OpenProtocolAttributes::GetProtocol,
            )
        } {
            Ok(protocol) => protocol,
            Err(error) => {
                result.failures.push(CaptureFailure::Open(error.status()));
                continue;
            }
        };
        let address = match protocol.address() {
            Ok(address) => address,
            Err(error) => {
                result.failures.push(CaptureFailure::Function(error));
                continue;
            }
        };
        let header = match protocol.read_header() {
            Ok(header) => header,
            Err(error) => {
                result.failures.push(CaptureFailure::Function(error));
                continue;
            }
        };
        let bars = core::array::from_fn(|bar| protocol.bar_regions(bar as u8));
        // FreePool refusal cannot be followed by EBS as though reclamation had
        // succeeded. Caller stops preparation with the status kept.
        if let Some(status) = bars.iter().find_map(|bar| match bar {
            Err(Error::Release(status)) => Some(*status),
            _ => None,
        }) {
            return Err(CaptureFailure::Function(Error::Release(status)));
        }
        #[cfg(feature = "diagnostics")]
        {
            observe_nvme(handle, &mut protocol, address, &header, &bars[0]);
            observe_lpss_i2c(handle, address, &header, &bars[0]);
            observe_glk(&mut protocol, address, &header, &bars[0]);
            if observe_cnvi_csr {
                observe_cnvi(&mut protocol, address, &header, &bars[0]);
            }
        }
        result.functions.push(FunctionObservation {
            address,
            header,
            bars,
        });
    }
    Ok(result)
}

#[cfg(feature = "diagnostics")]
fn observe_cnvi(
    protocol: &mut PciIo,
    address: Address,
    header: &[u8; 64],
    regions: &Result<Vec<BarRegion>, Error>,
) {
    use stela_boot_core::firmware_pci::cnvi;
    use stela_drivers::{bus::pci::regs, net::wifi::intel::firmware::legacy::identity};

    if !cnvi::matches(header) {
        return;
    }
    let command = u16::from_le_bytes([header[4], header[5]]);
    let memory_decode = command & regs::COMMAND_MEMORY != 0;
    let bus_master = command & regs::COMMAND_MASTER != 0;
    let power = cnvi::power_state(header, |offset| {
        let mut word = 0u16;
        // SAFETY: the original scoped PCI_IO, one aligned UINT16 output.
        status(unsafe {
            (protocol.config_read)(protocol, 1, u32::from(offset), 1, (&raw mut word).cast())
        })
        .ok()
        .map(|()| u16::from_le(word))
    });
    let Some(bar) = regions
        .as_ref()
        .ok()
        .and_then(|regions| cnvi::bar0(header, regions))
    else {
        crate::output::arguments(format_args!(
            "HOST WIFI CNVI PCI pci={address} power={power:?} memory_decode={memory_decode} bus_master={bus_master} bar0=unverified\n"
        ));
        return;
    };
    crate::output::arguments(format_args!(
        "HOST WIFI CNVI PCI pci={address} power={power:?} memory_decode={memory_decode} bus_master={bus_master} bar0_bus={:#x} bar0_host={:#x} bar0_bytes={} source=uefi-pci-io\n",
        bar.pci.start,
        bar.host.start,
        bar.host.end - bar.host.start
    ));
    if power != Ok(0) || !memory_decode {
        return;
    }

    let mut read = |offset: u32| {
        let mut word = 0u32;
        // SAFETY: the same handle's BAR0 is readable and covers these
        // aligned identity CSRs. No reset, wake, port or PRPH write is issued.
        status(unsafe {
            (protocol.mem_read)(protocol, 2, 0, u64::from(offset), 1, (&raw mut word).cast())
        })
        .map(|()| u32::from_le(word))
    };
    let snapshot = (|| {
        Ok::<_, Error>(identity::Snapshot {
            hw_revision: read(identity::HW_REVISION)?,
            rf_id: read(identity::RF_ID)?,
            gp_control: read(identity::GP_CONTROL)?,
        })
    })();
    match snapshot {
        Ok(snapshot) => crate::output::arguments(format_args!(
            "HOST WIFI CNVI CSR pci={address} hw_rev={:#010x} rf_id={:#010x} gp={:#010x} identity={:?} awake={:?}\n",
            snapshot.hw_revision,
            snapshot.rf_id,
            snapshot.gp_control,
            snapshot.qualify_31dc(),
            snapshot.qualify_awake(),
        )),
        Err(error) => crate::output::arguments(format_args!(
            "HOST WIFI CNVI CSR pci={address} read=failed {error}\n"
        )),
    }
}

#[cfg(feature = "diagnostics")]
fn observe_glk(
    protocol: &mut PciIo,
    address: Address,
    header: &[u8; 64],
    regions: &Result<Vec<BarRegion>, Error>,
) {
    use stela_boot_core::firmware_pci::glk::{self, Bar0};
    use stela_drivers::bus::pci::regs;

    let word = |offset| u32::from_le_bytes(header[offset..offset + 4].try_into().unwrap());
    let identity = word(0);
    if !glk::matches(identity, word(8)) {
        return;
    }
    let device = identity >> 16;
    let command = word(4) as u16;
    let mse = command & regs::COMMAND_MEMORY != 0;
    let bme = command & regs::COMMAND_MASTER != 0;
    let low = word(0x10);
    let high = word(0x14);
    let region = Bar0::observe(word(0x0c), low, high)
        .ok_or("raw-bar0-refused")
        .and_then(|bar| {
            let regions = regions.as_ref().map_err(|_| "descriptor-unavailable")?;
            bar.firmware_region(regions)
                .map(|region| (bar, region))
                .ok_or("descriptor-mismatch")
        });
    match region {
        Ok((bar, region)) => {
            crate::output::arguments(format_args!(
                "FW GLK ATTACH pci={address} device={device:04x} mse={mse} bme={bme} bar0_bus={:#x} bar0_host={:#x} bar0_bytes={} bar0_source=uefi-pci-io\n",
                region.pci.start,
                region.host.start,
                region.pci.end - region.pci.start
            ));
            observe_glk_ggtt(protocol, address, bar);
        }
        Err(reason) => crate::output::arguments(format_args!(
            "FW GLK ATTACH pci={address} device={device:04x} mse={mse} bme={bme} bar0_low={low:#x} bar0_next={high:#x} bar0={reason}\n"
        )),
    }
}

#[cfg(feature = "diagnostics")]
fn observe_glk_ggtt(
    protocol: &mut PciIo,
    address: Address,
    bar: stela_boot_core::firmware_pci::glk::Bar0,
) {
    use stela_boot_core::firmware_pci::glk::BAR0_BYTES;
    use stela_drivers::{
        bus::pci::Bar,
        gpu::intel::glk::{GRAPHICS_CONTROL, ggtt::Geometry},
    };

    let mut control = 0u16;
    // SAFETY: this is the live PCI_IO borrow used for the same function's
    // header and BAR descriptor. Width 1 reads one UINT16 of configuration.
    let read = unsafe {
        (protocol.config_read)(
            protocol,
            1,
            GRAPHICS_CONTROL.into(),
            1,
            (&mut control as *mut u16).cast(),
        )
    };
    if let Err(error) = status(read) {
        crate::output::arguments(format_args!(
            "FW GLK GGTT pci={address} status=config-unavailable error={error:?}\n"
        ));
        return;
    }
    let bar = Bar::Memory {
        address: bar.pci_base,
        size: BAR0_BYTES,
        prefetchable: false,
        wide: bar.bits == 64,
    };
    match Geometry::from_control(bar, control) {
        Ok(Some(geometry)) => {
            let range = geometry.pci_range();
            crate::output::arguments(format_args!(
                "FW GLK GGTT pci={address} status=observed pci_start={:#x} pci_end_exclusive={:#x} entries={} gpu_va_bytes={} source=uefi-pci-io\n",
                range.start,
                range.end,
                geometry.entry_count(),
                geometry.address_space_bytes()
            ));
        }
        Ok(None) => {
            crate::output::arguments(format_args!("FW GLK GGTT pci={address} status=disabled\n"))
        }
        Err(error) => crate::output::arguments(format_args!(
            "FW GLK GGTT pci={address} status=refused error={error:?}\n"
        )),
    }
}

impl core::fmt::Display for CaptureFailure {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Enumeration(status) => write!(f, "enumerate protocol: {status:?}"),
            Self::Open(status) => write!(f, "open protocol: {status:?}"),
            Self::Function(error) => write!(f, "observe function: {error}"),
            Self::Allocation => f.write_str("protocol census allocation failed"),
        }
    }
}
