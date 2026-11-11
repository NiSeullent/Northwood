// SPDX-License-Identifier: GPL-2.0-or-later
//! Explicit, polled bootstrap for an already bound Intel CNVi requester.
use crate::{
    Clock,
    block::BlockError,
    bus::{
        dma::{
            mapping::{DmaError, DomainKind},
            memory::PageSource,
        },
        mmio::owned::{CacheAttribute, Mapping, Protection, ResourceKind},
        pci::{
            CAPABILITY_EXPRESS, CAPABILITY_POWER, ConfigEffect, ConfigSpace, Function,
            owned::OwnedConfig,
            power::Timer,
            regs,
            requester::{
                ClaimError,
                preparation::{self, Attachment, BoundRequester, PreparingRequester},
                validate_stopped,
            },
        },
    },
    firmware::{Provider, Requirement, Target},
    net::wifi::intel::{
        boot::{self, FunctionOwner, NativeBootIo},
        firmware::{
            self as intel_firmware, Firmware,
            legacy::{Sku, adapter, alive::Progress, nvm},
        },
        identity::DEVICE_REVISION,
        power::{AccessError, Csr},
    },
};
use alloc::{boxed::Box, rc::Rc};
use core::{cell::RefCell, mem::ManuallyDrop, num::NonZeroUsize, ptr::NonNull};

const BAR_BYTES: usize = 16 * 1024;
const BAR_TYPE: u32 = 0b110;
const BAR_64: u32 = 0b100;
const BAR_IO_OR_PREFETCHABLE: u32 = 0b1001;
const PCIE_DEVICE_CONTROL: u16 = 8;
const PCIE_RELAXED_ORDERING: u16 = 1 << 4;
const PCIE_NO_SNOOP: u16 = 1 << 11;
const PM_CONTROL: u16 = 4;
const PM_STATE: u16 = 3;
const POLLED_COMMAND: u16 =
    regs::COMMAND_MEMORY | regs::COMMAND_MASTER | regs::COMMAND_INTX_DISABLE;
const INIT_FIRMWARE: Requirement<'static> = Requirement {
    name: Sku::Cnvi9000.firmware_name(),
    target: Target {
        driver: "wifi-intel",
        hardware: "pci:8086:31dc",
        revision: DEVICE_REVISION,
        board: None,
    },
    minimum_bytes: 96,
};

#[derive(Debug)]
pub enum Error {
    Claim(ClaimError),
    Preparation(preparation::Error),
    Pci(BlockError),
    Attachment(DmaError),
    Detach(DmaError),
    ForeignAttachment,
    Identity,
    Bar,
    Pcie,
    Ordering(u16),
    Power(u16),
    Command(u16),
    Boot(boot::Error),
    FirmwareResource(Box<crate::firmware::Error>),
    FirmwareBinding,
    Firmware(intel_firmware::Error),
    Adapter(adapter::Error),
    State,
}

struct ColdRegisters<'a> {
    bar: &'a Mapping,
    minimum: &'a dyn Timer,
}

impl Csr for ColdRegisters<'_> {
    fn read32(&mut self, register: u32) -> Result<u32, AccessError> {
        self.bar
            .read32(register as usize)
            .map_err(|_| AccessError::Mapping)
    }

    fn write32(&mut self, register: u32, value: u32) -> Result<(), AccessError> {
        self.bar
            .write32(register as usize, value)
            .map_err(|_| AccessError::Mapping)
    }

    fn delay_us(&mut self, microseconds: u32) -> Result<(), AccessError> {
        self.minimum
            .wait_us(u64::from(microseconds))
            .map_err(AccessError::Minimum)
    }
}

/// Stop the inherited engine and bind its original DMA root. Retain the
/// same BAR for the caller's post-reset MAC/RF preparation and INIT.
///
/// # Safety
/// The caller owns this awake, identified 31dc and its original BAR0, with
/// prior firmware, IRQ, worker and competing configuration owners excluded.
/// Bypass is excluded and the inherited DMA backing stays retained through
/// finalization or quarantine. The minimum timer must be valid in this context.
pub unsafe fn stop_and_bind(
    mut requester: PreparingRequester,
    bar0: Mapping,
    minimum: &dyn Timer,
) -> Result<(BoundRequester, Mapping), Error> {
    let bar0 = ManuallyDrop::new(bar0);
    minimum
        .check()
        .map_err(|error| Error::Pci(BlockError::PciPower(error)))?;
    let function = requester.function_mut();
    checked_bar(function, &bar0)?;
    validate_stopped(function).map_err(Error::Claim)?;
    let command = checked_command(function.config())?;
    if command & regs::COMMAND_MEMORY == 0 {
        return Err(Error::Command(command));
    }
    let mut io = ColdRegisters {
        bar: &bar0,
        minimum,
    };
    // SAFETY: the original retained BAR and caller's exclusion proof cover
    // these pre-bind register operations. A failed halt never binds a root.
    unsafe { adapter::InheritedStop::default().halt(&mut io) }.map_err(Error::Adapter)?;
    // SAFETY: the same requester just acknowledged the hardware stop. The
    // caller keeps inherited backing and excludes external producers.
    let bound = unsafe { requester.acknowledge_quiescence() }
        .finalize()
        .map_err(|failure| Error::Preparation(failure.error))?;
    Ok((bound, ManuallyDrop::into_inner(bar0)))
}

struct Resources {
    function: Function<OwnedConfig>,
    attachment: Attachment,
    bar0: Mapping,
}

struct Owner(ManuallyDrop<RefCell<Option<Resources>>>);

impl FunctionOwner for Owner {
    fn reset(&self) -> Result<bool, BlockError> {
        // The cold receipt and bound domain belong to the original PCI epoch.
        // The legacy bootstrap has its own CSR stop/reset sequence.
        Ok(false)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Retirement {
    Bound,
    ClearingCommand,
    Sealed,
    Detached,
    BarRetired,
    Complete,
}

/// Owns the native INIT attempt and the original resources through retirement.
pub struct Bootstrap<'a> {
    adapter: Option<adapter::Adapter<'a, NativeBootIo>>,
    owner: Rc<Owner>,
    retirement: Retirement,
}

impl<'a> Bootstrap<'a> {
    /// Load the boot catalog's exact CNVi image and upload its INIT sections.
    pub fn start_init(
        &mut self,
        provider: &dyn Provider,
        staging_capacity: NonZeroUsize,
    ) -> Result<(), Error> {
        self.with_init_firmware(provider, |adapter, firmware| {
            adapter.start_init(firmware, staging_capacity)
        })
    }

    /// Prepare the Gen1 command DMA rings and start the exact catalog INIT.
    pub fn start_nvm_init(
        &mut self,
        provider: &dyn Provider,
        staging_capacity: NonZeroUsize,
    ) -> Result<(), Error> {
        self.with_init_firmware(provider, |adapter, firmware| {
            adapter.start_nvm_init(firmware, staging_capacity)
        })
    }

    fn with_init_firmware(
        &mut self,
        provider: &dyn Provider,
        start: impl FnOnce(
            &mut adapter::Adapter<'a, NativeBootIo>,
            &Firmware<'_>,
        ) -> Result<(), adapter::Error>,
    ) -> Result<(), Error> {
        let adapter = self.adapter_mut()?;
        let image = provider
            .load(INIT_FIRMWARE)
            .map_err(|error| Error::FirmwareResource(Box::new(error)))?;
        if image.name() != INIT_FIRMWARE.name || image.target() != INIT_FIRMWARE.target {
            return Err(Error::FirmwareBinding);
        }
        let firmware = Firmware::parse(image.bytes()).map_err(Error::Firmware)?;
        start(adapter, &firmware).map_err(Error::Adapter)
    }

    pub fn poll_init_alive(&mut self) -> Result<Progress, Error> {
        self.adapter_mut()?
            .poll_init_alive()
            .map_err(Error::Adapter)
    }

    pub fn activate_nvm_queue(&mut self) -> Result<(), Error> {
        self.adapter_mut()?
            .activate_nvm_queue()
            .map_err(Error::Adapter)
    }

    pub fn start_nvm_read(&mut self, request: nvm::Request) -> Result<(), Error> {
        self.adapter_mut()?
            .start_nvm_read(request)
            .map_err(Error::Adapter)
    }

    pub fn poll_nvm_read(&mut self) -> Result<adapter::NvmProgress, Error> {
        self.adapter_mut()?.poll_nvm_read().map_err(Error::Adapter)
    }

    pub fn nvm_identity(&mut self) -> Result<nvm::Identity, Error> {
        self.adapter_mut()?.nvm_identity().map_err(Error::Adapter)
    }

    fn adapter_mut(&mut self) -> Result<&mut adapter::Adapter<'a, NativeBootIo>, Error> {
        self.adapter.as_mut().ok_or(Error::State)
    }

    /// Stop the device, drain and retire its DMA mappings, then close PCI BME.
    /// The same owner keeps its progress across an interrupted retirement.
    pub fn stop(&mut self) -> Result<(), Error> {
        if let Some(adapter) = self.adapter.as_mut() {
            adapter.stop().map_err(Error::Adapter)?;
            self.adapter = None;
        }
        if !matches!(
            self.retirement,
            Retirement::Bound | Retirement::ClearingCommand
        ) {
            return Ok(());
        }
        let mut resources = self.owner.0.borrow_mut();
        let function = &mut resources.as_mut().unwrap().function;
        let config = function.config();
        let command = checked_command(config)?;
        if command & regs::COMMAND_MASTER != 0 {
            if self.retirement == Retirement::ClearingCommand {
                return Err(Error::Command(command));
            }
            self.retirement = Retirement::ClearingCommand;
            if let Err(error) = config.write16(regs::COMMAND, command & !regs::COMMAND_MASTER) {
                if error.effect == ConfigEffect::None {
                    self.retirement = Retirement::Bound;
                }
                return Err(pci(error));
            }
        }
        validate_stopped(function).map_err(Error::Claim)?;
        function.config().seal_bus_master().map_err(pci)?;
        self.retirement = Retirement::Sealed;
        Ok(())
    }

    /// Stop and detach the original requester, then retire its BAR and config.
    /// A repeated call continues at the first unfinished step. Each native
    /// owner decides whether its outstanding operation can safely be retried.
    pub fn retire(&mut self) -> Result<(), Error> {
        self.stop()?;
        if self.retirement == Retirement::Complete {
            return Ok(());
        }
        let mut resources = self.owner.0.borrow_mut();
        let owned = resources.as_mut().unwrap();
        if self.retirement == Retirement::Sealed {
            // SAFETY: adapter stop acknowledged the device stop and payload
            // drain, all DMA mappings were unmapped, and PCI BME is sealed.
            // This exclusive polled owner has no IRQ grants or workers.
            unsafe { owned.attachment.detach() }.map_err(Error::Detach)?;
            self.retirement = Retirement::Detached;
        }
        if self.retirement == Retirement::Detached {
            // SAFETY: requester detach has acknowledged retirement and the
            // adapter's sole register accessor was dropped after stop.
            unsafe { owned.bar0.retire() }.map_err(pci)?;
            self.retirement = Retirement::BarRetired;
        }
        // SAFETY: all device, DMA, IRQ, MMIO and config users are retired;
        // the original sealed configuration remains owned until acknowledgement.
        unsafe { owned.function.config().retire() }.map_err(pci)?;
        self.retirement = Retirement::Complete;
        drop(resources.take());
        Ok(())
    }
}

/// Consume the original bound requester and its BAR0 into a polled INIT
/// bootstrap. PCI bus mastering is enabled after RFH DMA preflight.
///
/// # Safety
/// The caller has already cold-stopped and finalized this exact requester,
/// validated its MAC/RF identity for the CNVi 9000 image, and prepared its
/// awake NIC for the legacy INIT protocol. `bar0` is the same original owned
/// identity-mapped BAR0, kept throughout the cold stop and binding. Bypass
/// traffic and old submissions are drained, and this attachment has issued no
/// IRQ grants. Firmware, IRQ delivery, work and competing MMIO/configuration
/// users are excluded through retirement. `pages`
/// supplies compatible physical backing. `minimum` and `clock` are the real
/// checked timing grants for this process context, and `generation` is fresh.
/// Abandonment or uncertain construction parks the PCI configuration, DMA
/// attachment and BAR rather than treating Drop as retirement.
pub unsafe fn prepare<'a>(
    bound: BoundRequester,
    bar0: Mapping,
    minimum: Rc<dyn Timer>,
    pages: &'a dyn PageSource,
    generation: u64,
    clock: Clock,
) -> Result<Bootstrap<'a>, Error> {
    // Keep all native owners before any fallible check. In particular, an
    // uncertain command write must never drop the BAR or the bound function.
    let mut inputs = ManuallyDrop::new((bound, bar0));
    inputs.0.requester.check_stopped().map_err(Error::Claim)?;
    let (
        BoundRequester {
            requester,
            attachment,
        },
        bar0,
    ) = ManuallyDrop::into_inner(inputs);
    let owner = Rc::new(Owner(ManuallyDrop::new(RefCell::new(Some(Resources {
        function: requester.into_function(),
        attachment,
        bar0,
    })))));
    let (window, domain) = {
        let mut resources = owner.0.borrow_mut();
        let resources = resources.as_mut().unwrap();
        (checked_window(resources)?, resources.attachment.domain())
    };
    if domain.kind() != DomainKind::Isolated {
        return Err(Error::Attachment(DmaError::IommuUnavailable));
    }
    // SAFETY: the original stopped function, bound root and its same UC BAR
    // stay kept by this owner. No PCI interrupt delivery is enabled.
    let io = unsafe { NativeBootIo::from_window(window, owner.clone(), minimum) }
        .map_err(Error::Boot)?;
    // SAFETY: the caller's completed preparation and exact attachment match
    // keep the same awake CNVi function, process clock and DMA domain.
    let adapter =
        unsafe { adapter::Adapter::new(io, pages, domain, Sku::Cnvi9000, generation, clock) }
            .map_err(Error::Adapter)?;

    {
        let mut resources = owner.0.borrow_mut();
        let resources = resources.as_mut().unwrap();
        resources.bar0.check_current().map_err(pci)?;
        validate_stopped(&mut resources.function).map_err(Error::Claim)?;
        resources.function.enable_polled().map_err(Error::Pci)?;
        let command = checked_command(resources.function.config())?;
        if command & POLLED_COMMAND != POLLED_COMMAND {
            return Err(Error::Command(command));
        }
    }
    Ok(Bootstrap {
        adapter: Some(adapter),
        owner,
        retirement: Retirement::Bound,
    })
}

fn checked_window(resources: &mut Resources) -> Result<NonNull<[u8; BAR_BYTES]>, Error> {
    let function = &mut resources.function;
    let window = checked_bar(function, &resources.bar0)?;
    if !resources
        .attachment
        .matches_config(function.config())
        .map_err(Error::Attachment)?
    {
        return Err(Error::ForeignAttachment);
    }
    Ok(window)
}

fn checked_bar(
    function: &mut Function<OwnedConfig>,
    bar0: &Mapping,
) -> Result<NonNull<[u8; BAR_BYTES]>, Error> {
    if (function.vendor, function.device, function.class) != (0x8086, 0x31dc, [0x02, 0x80, 0]) {
        return Err(Error::Identity);
    }
    let capabilities = function.capabilities().map_err(pci)?;
    let express = capabilities
        .iter()
        .find_map(|&(offset, id)| (id == CAPABILITY_EXPRESS).then_some(u16::from(offset)))
        .ok_or(Error::Pcie)?;
    let config = function.config();
    let ordering = config.read16(express + PCIE_DEVICE_CONTROL).map_err(pci)?;
    if ordering & (PCIE_RELAXED_ORDERING | PCIE_NO_SNOOP) != 0 {
        return Err(Error::Ordering(ordering));
    }
    if let Some(&(offset, _)) = capabilities.iter().find(|&&(_, id)| id == CAPABILITY_POWER) {
        let power = config.read16(u16::from(offset) + PM_CONTROL).map_err(pci)?;
        if power & PM_STATE != 0 {
            return Err(Error::Power(power));
        }
    }

    let low = config.read32(regs::BASE_ADDRESS_0).map_err(pci)?;
    let high = match low & BAR_TYPE {
        0 => 0,
        BAR_64 => config.read32(regs::BASE_ADDRESS_0 + 4).map_err(pci)?,
        _ => return Err(Error::Bar),
    };
    let base = u64::from(high) << 32 | u64::from(low & !0xf);
    if low & BAR_IO_OR_PREFETCHABLE != 0
        || base == 0
        || !base.is_multiple_of(4096)
        || base != bar0.physical_span().map_err(pci)?.start()
        || bar0.attribute().map_err(pci)? != CacheAttribute::Uncached
        || bar0.protection().map_err(pci)? != Protection::ReadWrite
        || bar0.kind().map_err(pci)? != ResourceKind::Registers
    {
        return Err(Error::Bar);
    }
    Ok(bar0.address(0, BAR_BYTES).map_err(pci)?.cast())
}

fn checked_command(config: &OwnedConfig) -> Result<u16, Error> {
    let command = config.read16(regs::COMMAND).map_err(pci)?;
    if command == u16::MAX {
        return Err(Error::Command(command));
    }
    Ok(command)
}

fn pci(error: impl Into<BlockError>) -> Error {
    Error::Pci(error.into())
}
