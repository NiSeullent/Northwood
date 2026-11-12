// SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause
// Copyright (C) 2007-2014, 2018-2026 Intel Corporation
// Copyright (C) 2013-2015 Intel Mobile Communications GmbH
// Copyright (C) 2016-2017 Intel Deutschland GmbH
//! Owned standalone AX200/AX210 ROM publication and polling RX-to-ALIVE.
//! Adapted from pcie/{ctxt-info.c,gen1_2/{trans-gen2.c,rx.c}} and mvm/ops.c.
//! ALIVE is firmware boot metadata, not NVM/calibration/PNVM/radio readiness.
use super::{
    calibration,
    context::{self, Ax200Context, Queues},
    context_v2::{self, Ax210Context},
    dma::{self, Chunk, MappedFirmware},
    firmware::{self, Family, Firmware},
    identity::{Profile, Staged},
    platform::{Observations, StepConfiguration},
    pnvm_dma::{self, MappedTable},
    power::{self, Csr, Power},
    receive::{self, Alive, FreeDescriptor, Imr, Lmac, Packets, Umac},
};
use crate::{
    bus::{
        dma::{
            mapping::{Direction, DmaDomain},
            memory::PageSource,
        },
        mmio::Mmio,
    },
    firmware as resources,
};
use alloc::{rc::Rc, sync::Arc, vec::Vec};
use core::{
    ptr::NonNull,
    sync::atomic::{Ordering, fence},
};
mod ax200;
mod coexistence;
pub use coexistence::CoexistenceError;
mod hcmd;
pub use ax200::{
    Ax200Binding, Ax200BindingError, Ax200BindingPhase, Ax200BoardPolicy, Ax200BoardPolicyError,
    Ax200BoardPolicyPhase, Ax200Coexistence, Ax200CoexistenceError, Ax200CoexistencePhase,
    Ax200Country, Ax200CountryError, Ax200CountryPhase, Ax200LegacyPeer, Ax200LegacyPeerError,
    Ax200LegacyPeerPhase, Ax200Mac, Ax200MacError, Ax200MacPhase, Ax200ManagementQueue,
    Ax200ManagementQueueError, Ax200ManagementQueuePhase, Ax200Mcc, Ax200MccError, Ax200MccPhase,
    Ax200Phy, Ax200PhyError, Ax200PhyPhase, Ax200Scan, Ax200ScanConfiguration,
    Ax200ScanConfigurationError, Ax200ScanConfigurationPhase, Ax200ScanError, Ax200ScanPhase,
    Ax200ScanTraffic, Ax200Setup, Ax200SetupError, Ax200SetupOptions, Ax200SetupPhase,
    Ax200TransmitPower, Ax200TransmitPowerError, Ax200TransmitPowerPhase, BindingCommand,
    BindingCommandKind, BoardCommand, BoardCommandKind, CountryIo, CountryIoError, LegacyPeer,
    MacCommand, MacCommandKind, PeerCommand, PeerCommandKind, PhyCommand, PhyCommandKind, Probe,
    ScanCompletion, ScanRawReceive, ScanRejectedMpdu, ScanResult, ScanResultProgress,
    ScanRetirement, SelectedCountry, SetupDecisions, TransmitPowerLimit,
};
mod lari;
pub use lari::{LariError, LariPhase, LariRadio};
mod radio;
mod recycle;
mod rx;
pub use radio::{ColdRadio, ColdRadioPhase};
pub use recycle::Replenished;
pub use rx::{MpduProgress, NormalizedMpdu, ObservedMpdu, RejectedMpdu};
mod unified;
pub use hcmd::{CommandProgress, CommandResponse, CommandTicket, FirmwareNotification};
pub use unified::{InitializationPhase, InitializationProgress, InitializedNvm, UnifiedNvm};
/// `iwl_pcie_set_ltr`'s ~250 us long-value request.
const LTR_BOOT_VALUE: u32 = 0x88fa88fa;
const LTR_LAST_MESSAGE: u32 = 0xdc;
const FUNC_SCRATCH: u32 = 0x2c;
const FUNC_SCRATCH_INIT: u32 = 0x01010101;
const ROM_START: u32 = 1 << 7;
const HBUS_TARG_WRPTR: u32 = 0x460;
use super::interrupts::{self, HW_CAUSES as MSIX_HW_CAUSES};
const IML_CAUSE: u32 = 1 << 1;
const RECEIVE_BUFFER_BYTES: usize = 4096;

/// Declared platform integration mode, never inferred from an SB flag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProductResetPolicy {
    NoIwlmei,
    ManagedSap,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProductResetCoordination {
    NotPerformedNoIwlmei,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProductResetObservation {
    pub sb_configuration: u32,
    pub coordination: ProductResetCoordination,
}

/// Exclusive requester/BAR owner plus ordered CSR writes.
/// # Safety
/// Implementors keep the exclusive PCI function and uncached BAR
/// reservation through final Drop, excluding other firmware/OS/reset/IRQ users.
/// This is an owned reservation, never a borrowed token. Accesses must use the
/// device and complete ordered MMIO stores; errors may be uncertain.
/// Uncertain transport Drop keeps this entire object. A clone reaches the same
/// function under the same reservation; clones take turns.
pub unsafe trait BootIo: Csr + Clone + 'static {
    fn write8(&mut self, register: u32, value: u8) -> Result<(), power::AccessError>;
    /// Reset the stopped function with its PCI setup kept. `Ok(false)` reports
    /// no reset; `Err` preserves access/ownership failure and never authorizes
    /// another boot or an unchanged-state retry.
    fn reset_function(&mut self) -> Result<bool, crate::block::BlockError>;
}
/// What keeps the PCI function behind [`NativeBootIo`].
pub trait FunctionOwner {
    /// [`BootIo::reset_function`], the claim and BAR mapping kept throughout.
    fn reset(&self) -> Result<bool, crate::block::BlockError>;
}
#[derive(Clone)]
pub struct NativeBootIo {
    registers: Mmio,
    minimum: Rc<dyn crate::bus::pci::power::Timer>,
    owner: Rc<dyn FunctionOwner>,
}
impl NativeBootIo {
    /// # Safety
    /// `window` is the claimed AX200, AX210 or Intel 9000 `8086:31dc`
    /// function's aligned uncached BAR mapping covering its first 16 KiB.
    /// `owner` keeps that real exclusive PCI
    /// claim and BAR reservation, with all old DMA/IRQs stopped and drained and
    /// PCI interrupt routes disabled; no unrelated/dummy owner checks.
    /// The minimum provider is the actual retained platform grant, usable in this
    /// context. Nominal clocks cannot supply it. The window stays mapped through
    /// the owner's final Drop, including deliberate quarantine on failure.
    pub unsafe fn from_window(
        window: NonNull<[u8; 16384]>,
        owner: Rc<dyn FunctionOwner>,
        minimum: Rc<dyn crate::bus::pci::power::Timer>,
    ) -> Result<Self, Error> {
        minimum
            .check()
            .map_err(|error| Error::Pci(crate::block::BlockError::PciPower(error)))?;
        if !(window.as_ptr() as usize).is_multiple_of(4096) {
            return Err(Error::Access(power::AccessError::Mapping));
        }
        Ok(Self {
            // SAFETY: covered by this function's own safety contract.
            registers: unsafe { Mmio::new(window.as_ptr().cast::<u8>() as u64, 16384) },
            minimum,
            owner,
        })
    }
}
impl Csr for NativeBootIo {
    fn read32(&mut self, register: u32) -> Result<u32, power::AccessError> {
        if !register.is_multiple_of(4) || register > 8188 {
            return Err(power::AccessError::Bounds);
        }
        Ok(self.registers.read32(register as usize))
    }
    fn write32(&mut self, register: u32, value: u32) -> Result<(), power::AccessError> {
        if !register.is_multiple_of(4) || register > 8188 {
            return Err(power::AccessError::Bounds);
        }
        self.registers.write32(register as usize, value);
        Ok(())
    }
    fn delay_us(&mut self, microseconds: u32) -> Result<(), power::AccessError> {
        self.minimum
            .wait_us(u64::from(microseconds))
            .map_err(power::AccessError::Minimum)
    }
}
// SAFETY: construction requires the owned exclusive requester/BAR.
unsafe impl BootIo for NativeBootIo {
    fn write8(&mut self, register: u32, value: u8) -> Result<(), power::AccessError> {
        if register >= 8192 {
            return Err(power::AccessError::Bounds);
        }
        self.registers.write8(register as usize, value);
        Ok(())
    }
    fn reset_function(&mut self) -> Result<bool, crate::block::BlockError> {
        self.owner.reset()
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    Pci(crate::block::BlockError),
    Access(power::AccessError),
    Power(power::Error),
    Dma(dma::Error),
    Context(context::Error),
    ContextV2(context_v2::Error),
    Calibration(calibration::Error),
    PnvmDma(pnvm_dma::Error),
    PnvmNotification,
    Command(super::command::Error),
    CommandReclaim(dma::Error),
    CommandTicket,
    CommandPending,
    CommandSequence(u16),
    NotificationLimit,
    ReceiveEpoch,
    Nvm(super::nvm::Error),
    Radio(super::radio::Error),
    InitializationResponse { command: u8, group: u8 },
    ManagedSapUnsupported,
    Firmware(firmware::Error),
    Receive(receive::Error),
    Family,
    FirmwareBinding,
    PhyConfiguration,
    QueueSize,
    State,
    IdentityChanged,
    FailedRead { register: u32, value: u32 },
    Rejected { register: u32 },
    Timeout { register: u32 },
    RfKilled,
    DeviceError(u32),
    CompletionId(u16),
    CompletionCredit,
    Fragmented,
}
impl From<crate::bus::dma::mapping::DmaError> for Error {
    fn from(e: crate::bus::dma::mapping::DmaError) -> Self {
        Self::Dma(dma::Error::Dma(e))
    }
}
impl From<power::AccessError> for Error {
    fn from(e: power::AccessError) -> Self {
        Self::Access(e)
    }
}
impl From<power::Error> for Error {
    fn from(e: power::Error) -> Self {
        Self::Power(e)
    }
}
impl From<dma::Error> for Error {
    fn from(e: dma::Error) -> Self {
        Self::Dma(e)
    }
}
impl From<context::Error> for Error {
    fn from(e: context::Error) -> Self {
        Self::Context(e)
    }
}
impl From<context_v2::Error> for Error {
    fn from(e: context_v2::Error) -> Self {
        Self::ContextV2(e)
    }
}
impl From<calibration::Error> for Error {
    fn from(value: calibration::Error) -> Self {
        Self::Calibration(value)
    }
}
impl From<pnvm_dma::Error> for Error {
    fn from(value: pnvm_dma::Error) -> Self {
        Self::PnvmDma(value)
    }
}
impl From<super::nvm::Error> for Error {
    fn from(value: super::nvm::Error) -> Self {
        Self::Nvm(value)
    }
}
impl From<super::radio::Error> for Error {
    fn from(value: super::radio::Error) -> Self {
        Self::Radio(value)
    }
}
impl From<super::command::Error> for Error {
    fn from(value: super::command::Error) -> Self {
        Self::Command(value)
    }
}
impl From<firmware::Error> for Error {
    fn from(e: firmware::Error) -> Self {
        Self::Firmware(e)
    }
}
impl From<receive::Error> for Error {
    fn from(e: receive::Error) -> Self {
        Self::Receive(e)
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    Prepared,
    Published,
    Restocked,
    Alive,
    PnvmWaiting,
    PnvmObserved,
    Poisoned,
    Quiescent,
    Stopped,
}
#[derive(Clone, Debug)]
pub struct AliveReport {
    pub notification_version: u8,
    pub flags: u16,
    pub lmac: [Lmac; 2],
    pub umac: Umac,
    pub sku: [u32; 3],
    pub imr: Imr,
    pub platform_id: Option<u64>,
    pub raw: Vec<u8>,
}
/// Raw notification status, not a radio/calibration success enum.
/// Linux's helper logs this word without defining a success-value contract.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PnvmNotification {
    pub status: u32,
    pub raw: Vec<u8>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PnvmProgress {
    Skipped,
    Waiting,
    Observed,
}
struct PlatformPhase<'a> {
    receipts: calibration::Receipts<'a>,
    tables: Vec<MappedTable>,
    notification: Option<PnvmNotification>,
    released_tables: usize,
}
struct Buffer {
    allocation: Chunk,
    device_owned: bool,
}
/// One firmware/reset epoch, keeping CSR owner, native pins, CPU views
/// and isolated device DMA mappings. Polling mode never unmasks IRQ sources.
/// Dropping an active/uncertain epoch quarantines the BAR/requester too.
pub struct RomBoot<'a, Q: BootIo> {
    vector: bool,
    io: Option<Q>,
    profile: Profile,
    firmware: Firmware<'a>,
    image: resources::Image<'a>,
    images: MappedFirmware,
    images_stopped: bool,
    domain: Arc<dyn DmaDomain>,
    platform: Option<PlatformPhase<'a>>,
    alive_sku: Option<[u32; 3]>,
    /// `step_urm`, which the STEP says once firmware runs.
    step_urm: bool,
    sb_configuration: Option<u32>,
    product_reset_policy: Option<ProductResetPolicy>,
    context: Chunk,
    ax210: Option<Ax210Memory<'a>>,
    free: Chunk,
    used: Chunk,
    status: Chunk,
    command: Chunk,
    hcmd: hcmd::Queue,
    receive_ring: recycle::Ring,
    buffers: Vec<Buffer>,
    entries: u16,
    read: u16,
    credits: u16,
    // Original MVM quota cache spans MAC/PHY/power continuations in this
    // firmware epoch. Only a real completed empty command establishes it.
    empty_quotas_applied: bool,
    coexistence: coexistence::Coexistence,
    state: State,
}
struct Ax210Memory<'a> {
    scratch: Chunk,
    info: Chunk,
    iml: Chunk,
    step: StepConfiguration<'a>,
}
/// Compatibility name for the original AX200 `start` constructor. New family
/// consumers can name `RomBoot` and choose its explicit family constructor.
/// `CNVI_PMU_STEP_FLOW_BZ`, `CNVI_PMU_STEP_FLOW_SC` and what `step_urm` reads.
const STEP_FLOW_BZ: u32 = 0xa2_d588;
const STEP_FLOW_SC: u32 = 0xa2_d688;
const STEP_FORCE_URM: u32 = 1 << 2;

pub type Ax200Boot<'a, Q> = RomBoot<'a, Q>;
impl<'a, Q: BootIo> RomBoot<'a, Q> {
    /// The device family this boot runs.
    pub fn family(&self) -> Family {
        self.profile.family()
    }
    /// The radio beside this boot's MAC.
    pub fn radio(&self) -> super::identity::Rf {
        self.profile.radio()
    }
    /// What limits a radio with EHT, from ALIVE on.
    pub fn eht_limits(&self) -> Option<super::identity::EhtLimits> {
        self.profile.eht_limits(self.alive_sku?, self.step_urm)
    }
    /// Start the standalone AX200 context protocol.
    /// # Safety
    /// PCI Memory Space is enabled and bus mastering has been enabled only
    /// after isolated domain attachment by the checked platform owner.
    /// Source allocations are fresh/exclusive and domain is attached
    /// to this requester. Platform/requester WB DMA is coherent,
    /// PCIe NoSnoop, ATS and relaxed ordering are disabled, and aligned16/32-bit
    /// volatile completion observations plus acquire ordering see prior device
    /// payload writes. A WB attribute/compiler fence alone proves none of this.
    /// Validation remains true through every quarantined lease. No IRQ route
    /// or competing submission/reset worker can run during this polling epoch.
    /// General production PCI/IOMMU/coherency attachment is not supplied here.
    pub unsafe fn start(
        power: Power<Q>,
        staged: Staged<'a>,
        source: &dyn PageSource,
        domain: Option<Arc<dyn DmaDomain>>,
        entries: u16,
    ) -> Result<Self, Error> {
        // SAFETY: the caller supplies the complete documented validation.
        unsafe { Self::start_inner(power, staged, None, source, domain, entries, None) }
    }
    /// Start the distinct AX210 IML/context-v2 protocol, preserving the
    /// checked STEP resource receipt through this epoch. No absent default is
    /// manufactured; EFI_NOT_FOUND must have been recorded by its loader.
    /// # Safety
    /// The same requester/domain/coherency/exclusive polling obligations as
    /// `start` apply, for the standalone AX210 function. NoIwlmei is
    /// an explicit unmanaged platform mode: the caller keeps
    /// exclusive controller/reset ownership and no kept active SAP owner.
    /// It does not assert AMT is disabled or invent a PLDR acknowledgement.
    /// ManagedSap requires a real session/PLDR/END provider and is refused.
    pub unsafe fn start_ax210(
        power: Power<Q>,
        staged: Staged<'a>,
        step: StepConfiguration<'a>,
        source: &dyn PageSource,
        domain: Option<Arc<dyn DmaDomain>>,
        entries: u16,
        product_reset_policy: ProductResetPolicy,
    ) -> Result<Self, Error> {
        // SAFETY: covered by this function's own safety contract.
        unsafe {
            Self::start_inner(
                power,
                staged,
                Some(step),
                source,
                domain,
                entries,
                Some(product_reset_policy),
            )
        }
    }
    unsafe fn start_inner(
        power: Power<Q>,
        staged: Staged<'a>,
        step: Option<StepConfiguration<'a>>,
        source: &dyn PageSource,
        domain: Option<Arc<dyn DmaDomain>>,
        entries: u16,
        product_reset_policy: Option<ProductResetPolicy>,
    ) -> Result<Self, Error> {
        let vector = power.vector();
        let (io, profile, awake) = power.into_awake()?;
        let (firmware, image) = staged.into_parts();
        let request = profile.firmware_request();
        if !profile.binds(request.name, &image) {
            return Err(Error::FirmwareBinding);
        }
        let family = profile.family();
        if (family >= Family::Ax210) != step.is_some() {
            return Err(Error::Family);
        }
        if product_reset_policy == Some(ProductResetPolicy::ManagedSap) {
            return Err(Error::ManagedSapUnsupported);
        }
        if awake.rf_killed {
            return Err(Error::RfKilled);
        }
        let maximum = if family >= Family::Ax210 { 1024 } else { 2048 };
        if !(16..=maximum).contains(&entries) || !entries.is_power_of_two() {
            return Err(Error::QueueSize);
        }
        firmware.require_family(family)?;
        if firmware.phy_sku().is_none() {
            return Err(Error::PhyConfiguration);
        }
        if firmware
            .command_version(0, 1)
            .is_none_or(|v| !matches!(v.notification_version, 6..=8))
        {
            return Err(Error::Receive(receive::Error::AliveVersion(
                firmware
                    .command_version(0, 1)
                    .map(|v| v.notification_version),
            )));
        }
        if family >= Family::Ax210 {
            let iml = firmware
                .iml()
                .ok_or(firmware::Error::Missing("AX210 IML"))?;
            if iml.len() > 32768 {
                return Err(Error::Dma(dma::Error::PayloadSize));
            }
        }
        let domain = domain.ok_or(Error::Dma(dma::Error::Dma(
            crate::bus::dma::mapping::DmaError::IommuUnavailable,
        )))?;
        let plan = firmware.self_load_plan()?;
        // SAFETY: covered by this function's own safety contract.
        let images = unsafe { MappedFirmware::prepare(&plan, source, Some(domain.clone()))? };
        let allocate = |bytes: usize, direction| {
            Chunk::allocate(source, domain.clone(), &alloc::vec![0;bytes], direction)
        };
        let (free_stride, used_stride, status_bytes, command_entries) = match family {
            Family::Ax200 => (8, 4, 12, 32),
            _ => (16, receive::used_stride(family), 2, 128),
        };
        let free = allocate(usize::from(entries) * free_stride, Direction::Bidirectional)?;
        let used = allocate(usize::from(entries) * used_stride, Direction::Bidirectional)?;
        let status = allocate(status_bytes, Direction::Bidirectional)?;
        let command = allocate(command_entries * 256, Direction::Bidirectional)?;
        // SAFETY: covered by this function's own safety contract.
        let hcmd = unsafe { hcmd::Queue::prepare(&command, &profile, source, domain.clone())? };
        let mut buffers = Vec::new();
        for _ in 0..entries - 1 {
            buffers.push(Buffer {
                allocation: allocate(RECEIVE_BUFFER_BYTES, Direction::FromDevice)?,
                device_owned: false,
            });
        }
        let (context, ax210) = if let Some(step) = step {
            let context = allocate(context_v2::CONTEXT_BYTES, Direction::Bidirectional)?;
            let scratch = allocate(context_v2::SCRATCH_BYTES, Direction::Bidirectional)?;
            let info = allocate(context_v2::INFO_BYTES, Direction::Bidirectional)?;
            // Keep donor dma_alloc_coherent permissions; IML remains pinned
            // until current-epoch ALIVE or acknowledged device stop.
            let iml = Chunk::allocate(
                source,
                domain.clone(),
                firmware.iml().unwrap(),
                Direction::Bidirectional,
            )?;
            let encoded = Ax210Context::encode(
                &profile,
                &firmware,
                context_v2::Queues {
                    free: free.region(),
                    used: used.region(),
                    status: status.region(),
                    command: command.region(),
                    receive_entries: entries,
                },
                images.prepared_regions()?,
                context_v2::Memory {
                    info: info.region(),
                    scratch: scratch.region(),
                    context: context.region(),
                    iml: iml.region(),
                },
                &step,
            )?;
            // SAFETY: every allocation remains exclusively CPU owned before
            // the first context pointer publication below.
            unsafe {
                context.write_owned(0, encoded.context())?;
                scratch.write_owned(0, encoded.scratch())?;
            }
            (
                context,
                Some(Ax210Memory {
                    scratch,
                    info,
                    iml,
                    step,
                }),
            )
        } else {
            let encoded = Ax200Context::encode(
                &profile,
                &plan,
                Queues {
                    free: free.region(),
                    used: used.region(),
                    status: status.region(),
                    command: command.region(),
                    receive_entries: entries,
                },
                images.prepared_regions()?,
            )?;
            (
                Chunk::allocate(
                    source,
                    domain.clone(),
                    encoded.bytes(),
                    Direction::Bidirectional,
                )?,
                None,
            )
        };
        let mut boot = Self {
            io: Some(io),
            vector,
            profile,
            firmware,
            image,
            images,
            images_stopped: false,
            domain,
            platform: None,
            alive_sku: None,
            step_urm: false,
            sb_configuration: None,
            product_reset_policy,
            context,
            ax210,
            free,
            used,
            status,
            command,
            hcmd,
            receive_ring: recycle::Ring::new(entries),
            buffers,
            entries,
            read: 0,
            credits: 0,
            empty_quotas_applied: false,
            coexistence: coexistence::Coexistence::new(),
            state: State::Prepared,
        };
        boot.publish()?;
        Ok(boot)
    }
    /// Exact canonical image identity, integrity and provenance kept across
    /// publication/completion; no parsed-input or diagnostic-image substitution.
    pub fn firmware_source(&self) -> resources::Image<'a> {
        self.image
    }
    pub fn product_reset_observation(&self) -> Option<ProductResetObservation> {
        self.sb_configuration
            .map(|sb_configuration| ProductResetObservation {
                sb_configuration,
                coordination: ProductResetCoordination::NotPerformedNoIwlmei,
            })
    }
    /// Observed STEP receipt used by AX210 scratch, kept without
    /// copying its original resource bytes. AX200 has no context-v2 input.
    pub fn step_configuration(&self) -> Option<&StepConfiguration<'a>> {
        self.ax210.as_ref().map(|memory| &memory.step)
    }
    fn io(&mut self) -> &mut Q {
        self.io.as_mut().expect("retained owner")
    }
    fn read_register(&mut self, register: u32) -> Result<u32, Error> {
        let value = self.io().read32(register)?;
        if value == u32::MAX || matches!(value & !0xf, 0xa5a5a5a0 | 0x5a5a5a50) {
            return Err(Error::FailedRead { register, value });
        }
        Ok(value)
    }
    fn write(&mut self, register: u32, value: u32) -> Result<(), Error> {
        self.io().write32(register, value)?;
        Ok(())
    }
    fn set(&mut self, register: u32, value: u32) -> Result<(), Error> {
        let old = self.read_register(register)?;
        self.write(register, old | value)
    }
    fn poll_register(
        &mut self,
        register: u32,
        mask: u32,
        value: u32,
        microseconds: u32,
    ) -> Result<(), Error> {
        for _ in 0..microseconds.div_ceil(10) {
            if self.read_register(register)? & mask == value {
                return Ok(());
            }
            self.io().delay_us(10)?;
        }
        Err(Error::Timeout { register })
    }
    fn prph_write(&mut self, address: u32, value: u32) -> Result<(), Error> {
        let mask = self.grab()?;
        self.write(0x444, (address & mask) | (3 << 24))?;
        self.write(0x44c, value)?;
        self.release()
    }
    /// `__iwl_trans_pcie_grab_nic_access`, giving the peripheral address mask.
    fn grab(&mut self) -> Result<u32, Error> {
        let family = self.profile.family();
        let access = power::NicAccess::of(family);
        self.set(0x24, access.request)?;
        self.io().delay_us(2)?;
        self.poll_register(0x24, access.mask, access.ready, 15000)?;
        Ok(if family >= Family::Ax210 {
            0xffffff
        } else {
            0xfffff
        })
    }
    fn release(&mut self) -> Result<(), Error> {
        let access = power::NicAccess::of(self.profile.family());
        let gp = self.read_register(0x24)?;
        self.write(0x24, gp & !access.request)
    }
    fn prph_read(&mut self, address: u32) -> Result<u32, Error> {
        let mask = self.grab()?;
        self.write(0x448, (address & mask) | (3 << 24))?;
        let value = self.read_register(0x450)?;
        self.release()?;
        Ok(value)
    }
    fn publish(&mut self) -> Result<(), Error> {
        let observed = self.profile.observed();
        if self.read_register(0x28)? != observed.hardware_revision
            || self.read_register(0x9c)? != observed.radio_id
        {
            return Err(Error::IdentityChanged);
        }
        if self.read_register(0x24)? & 0x08000000 == 0 {
            return Err(Error::RfKilled);
        }
        // CONFIG_IWLMEI-disabled Linux performs no PLDR coordination. Keep
        // the SB value even when0x80, under this explicit unmanaged
        // ownership mode. Never infer mode or a fictional ACK from that flag.
        if self.profile.family() >= Family::Ax210 {
            if self.product_reset_policy != Some(ProductResetPolicy::NoIwlmei) {
                return Err(Error::ManagedSapUnsupported);
            }
            self.sb_configuration = Some(self.prph_read(0xa03088 + 0x300000)?);
        }
        self.write(12, 0)?;
        if self.read_register(12)? != 0 {
            return Err(Error::Rejected { register: 12 });
        }
        self.write(0x5c, 2)?;
        self.write(0x5c, 4)?;
        self.write(8, u32::MAX)?;
        if self.vector {
            self.route_interrupts()?;
        }
        // iwl_mvm_nic_config returns immediately on AX210: CSR0 becomes the
        // context-v2 boot control, never the old PHY configuration fields.
        if self.profile.family() == Family::Ax200 {
            let phy = self.firmware.phy_sku().ok_or(Error::PhyConfiguration)?;
            let config = (observed.hardware_revision & 0xf)
                | ((phy & 3) << 10)
                | (((phy >> 2) & 3) << 14)
                | (((phy >> 4) & 3) << 12);
            let hw = self.read_register(0)?;
            self.write(0, (hw & !0xff0f) | config)?;
            if self.read_register(0)? & 0xff0f != config {
                return Err(Error::Rejected { register: 0 });
            }
        }
        self.io().write8(4, 0x40)?;
        self.set(0xa8, 0x800fffff)?;
        // Keep the complete guard before the FIRST context pointer write;
        // even its low half/error can be an uncertain DMA publication.
        self.images.expose_for_context()?;
        self.context.expose();
        self.free.expose();
        self.used.expose();
        self.status.expose();
        self.command.expose();
        self.hcmd.expose();
        if let Some(memory) = self.ax210.as_mut() {
            memory.scratch.expose();
            memory.info.expose();
            memory.iml.expose();
        }
        self.state = State::Published;
        fence(Ordering::Release);
        let address = self.context.region().address;
        let register = if self.profile.family() >= Family::Ax210 {
            0x118
        } else {
            0x40
        };
        self.write(register, address as u32)?;
        self.write(register + 4, (address >> 32) as u32)?;
        if let Some(memory) = &self.ax210 {
            let iml = memory.iml.region();
            self.write(0x120, iml.address as u32)?;
            self.write(0x124, (iml.address >> 32) as u32)?;
            self.write(0x128, iml.bytes as u32)?;
            self.set(0, 2)?; // CSR_AUTO_FUNC_BOOT_ENA
        }
        // `iwl_pcie_set_ltr`: ROM leaves LTR unset, and an integrated AX210
        // device keeps it out of the host's reach.
        let family = self.profile.family();
        let keep_busy = match (self.profile.platform().integrated, family) {
            (false, _) => {
                self.write(0xd4, LTR_BOOT_VALUE)?;
                false
            }
            (true, Family::Ax200) => {
                self.prph_write(0xa0348c, 0xf)?;
                self.prph_write(0xa03480, LTR_BOOT_VALUE)?;
                false
            }
            (true, Family::Ax210) => {
                self.write(MSIX_HW_CAUSES, IML_CAUSE)?;
                true
            }
            (true, _) => false,
        };
        if family >= Family::Bz {
            self.write(FUNC_SCRATCH, FUNC_SCRATCH_INIT)?;
            self.set(0x24, ROM_START)?;
        } else {
            let offset = if family >= Family::Ax210 { 0x300000 } else { 0 };
            self.prph_write(0xa05c44 + offset, 1)?;
        }
        if keep_busy {
            self.spin_for_iml()?;
        }
        Ok(())
    }
    /// `iwl_pcie_conf_msix_hw` and `_iwl_enable_interrupts`. A cause only
    /// wakes the core that looks for frames, which finds what it asks for in
    /// the rings, so nothing here waits for firmware's own state.
    fn route_interrupts(&mut self) -> Result<(), Error> {
        let family = self.profile.family();
        let offset = if family >= Family::Ax210 { 0x300000 } else { 0 };
        self.prph_write(interrupts::UREG_CHICK + offset, interrupts::MSIX_ENABLE)?;
        let routes = interrupts::routes(family);
        for ivar in routes.ivars {
            self.io().write8(ivar, interrupts::SHARED)?;
        }
        self.write(interrupts::FH_MASK, routes.fh_mask)?;
        self.write(interrupts::HW_MASK, routes.hw_mask)
    }
    /// What booting this function again takes of this boot.
    pub(crate) fn restart_inputs(&self) -> (Q, Profile, Arc<dyn DmaDomain>, u16) {
        let io = self.io.clone().expect("retained owner");
        (io, self.profile.clone(), self.domain.clone(), self.entries)
    }
    /// `CSR_INT`, and on a function in MSI-X mode the causes it reports
    /// through instead.
    fn causes(&mut self) -> Result<u32, Error> {
        let causes = self.read_register(8)?;
        self.write(8, causes | !(1 | (1 << 31)))?;
        if !self.vector {
            return Ok(causes);
        }
        let family = self.profile.family();
        let reported = interrupts::take(self.io(), family).map_err(Error::Access)?;
        Ok(causes | reported)
    }
    /// Whether the function's first MSI-X vector reaches the host.
    pub(crate) fn vector(&self) -> bool {
        self.vector
    }
    /// `iwl_pcie_spin_for_iml`: keep the CPU and device busy for at least
    /// 100 ms or until the image loader finishes. Running out is no failure;
    /// ALIVE decides whether the boot worked.
    fn spin_for_iml(&mut self) -> Result<(), Error> {
        for _ in 0..100_000 {
            if self.read_register(MSIX_HW_CAUSES)? & IML_CAUSE != 0 {
                break;
            }
            self.read_register(LTR_LAST_MESSAGE)?;
            self.io().delay_us(1)?;
        }
        Ok(())
    }
    fn restock(&mut self) -> Result<(), Error> {
        for (index, buffer) in self.buffers.iter_mut().enumerate() {
            let descriptor = receive::free_descriptor(
                self.profile.family(),
                buffer.allocation.region().address,
                index as u16 + 1,
            )?;
            let descriptor = match &descriptor {
                FreeDescriptor::Ax200(bytes) => bytes.as_slice(),
                FreeDescriptor::Ax210(bytes) => bytes.as_slice(),
            };
            buffer.allocation.expose();
            // SAFETY: the initial free descriptors are still CPU-owned; ROM
            // configured RFH, but no free-ring producer doorbell has advanced.
            unsafe {
                self.free
                    .write_owned(index * descriptor.len(), descriptor)?;
            }
        }
        let published = (self.entries - 1) & !7;
        // Mark exact eligible payloads before the potentially uncertain store.
        for buffer in &mut self.buffers[..usize::from(published)] {
            buffer.device_owned = true;
        }
        self.credits = published;
        self.receive_ring.initial(self.entries, published);
        self.state = State::Restocked;
        fence(Ordering::Release);
        self.ring_receive_doorbell(published)
    }
    /// `iwl_pcie_rxq_inc_wr_ptr` for the default queue.
    pub(super) fn ring_receive_doorbell(&mut self, write: u16) -> Result<(), Error> {
        if self.profile.family() >= Family::Bz {
            self.write(HBUS_TARG_WRPTR, u32::from(write))
        } else {
            self.write(0x1c80, u32::from(write))
        }
    }
    fn completed_buffer(&mut self) -> Result<Option<Vec<u8>>, Error> {
        // SAFETY: start checked shared coherent producer polling, kept
        // pins/domain/CPU view, aligned width; this is NOT streaming sync_cpu.
        let family = self.profile.family();
        let mask = if family == Family::Ax200 {
            0xfff
        } else {
            u16::MAX
        };
        // SAFETY: the status page is pinned, coherent and device-written; this reads a
        // complete aligned u16 without claiming CPU ownership.
        let closed = unsafe { self.status.read16_coherent(0)? } & mask & (self.entries - 1);
        fence(Ordering::Acquire);
        let count = closed.wrapping_sub(self.read) & (self.entries - 1);
        if count > self.credits {
            return Err(Error::CompletionCredit);
        }
        if self.read == closed {
            return Ok(None);
        }
        let stride = receive::used_stride(family);
        let mut descriptor = [0; 32];
        for offset in (0..stride).step_by(4) {
            descriptor[offset..offset + 4].copy_from_slice(
                // SAFETY: the used ring is pinned and coherent, and `closed` says the device
                // finished writing this entry.
                &unsafe {
                    self.used
                        .read32_coherent(usize::from(self.read) * stride + offset)?
                }
                .to_le_bytes(),
            );
        }
        let used = receive::used_descriptor(family, &descriptor[..stride])?;
        if used.fragmented {
            return Err(Error::Fragmented);
        }
        let buffer = self
            .buffers
            .get_mut(usize::from(used.id - 1))
            .ok_or(Error::CompletionId(used.id))?;
        if !buffer.device_owned {
            return Err(Error::CompletionId(used.id));
        }
        buffer.device_owned = false;
        self.credits -= 1;
        self.read = (self.read + 1) & (self.entries - 1);
        // SAFETY: the closed producer and unique submitted VID
        // transfer this complete RB to CPU ownership. Explicit rearm is only
        // allowed after this visibility/copy completes successfully.
        let bytes = unsafe { buffer.allocation.snapshot_completed()? };
        self.receive_ring.complete(used.id)?;
        Ok(Some(bytes))
    }
    fn receive(&mut self) -> Result<Option<AliveReport>, Error> {
        let family = self.profile.family();
        while let Some(bytes) = self.completed_buffer()? {
            let mut observed = None;
            for packet in Packets::new(family, 0, &bytes)? {
                let packet = packet?;
                if packet.command != 1 || packet.group != 0 {
                    self.hcmd.retain_notification(&packet)?;
                    continue;
                }
                if observed.is_some() {
                    return Err(Error::State);
                }
                let alive = Alive::decode(&packet, &self.firmware)?;
                let report = AliveReport {
                    notification_version: alive.notification_version,
                    flags: alive.flags,
                    lmac: alive.lmac,
                    umac: alive.umac,
                    sku: alive.sku,
                    imr: alive.imr,
                    platform_id: alive.platform_id,
                    raw: alive.raw().to_vec(),
                };
                // SAFETY: valid ALIVE was decoded only from completed
                // DMA in this exclusively context/firmware/reset epoch.
                unsafe {
                    self.images.release_after_alive()?;
                    if let Some(memory) = self.ax210.as_mut() {
                        memory.iml.release()?;
                    } else {
                        self.context.release()?;
                    }
                }
                self.alive_sku = Some(report.sku);
                self.state = State::Alive;
                self.step_urm = match self.profile.family() {
                    Family::Ax200 | Family::Ax210 => false,
                    Family::Bz => self.prph_read(STEP_FLOW_BZ)? & STEP_FORCE_URM != 0,
                    Family::Sc => self.prph_read(STEP_FLOW_SC)? & STEP_FORCE_URM != 0,
                };
                observed = Some(report);
            }
            if observed.is_some() {
                return Ok(observed);
            }
        }
        Ok(None)
    }
    /// Derive conditional resource requirements only from this transport's
    /// completed ALIVE; caller-mutated report metadata cannot choose
    /// another board/default. Acquisition/upload/status remain a later phase.
    pub fn post_alive_plan(&self) -> Result<super::platform::PostAlivePlan, Error> {
        if !matches!(
            self.state,
            State::Alive | State::PnvmWaiting | State::PnvmObserved
        ) {
            return Err(Error::State);
        }
        Ok(super::platform::post_alive_plan(
            &self.profile,
            self.alive_sku.ok_or(Error::State)?,
        ))
    }
    /// Acquire exactly the current ALIVE SKU's canonical resources, prepare
    /// DMA, then publish scratch and the source PNVM doorbell once. Actual
    /// missing/failed observations remain visible in the receipts.
    /// # Safety
    /// Source returns fresh exclusive native RAM under the same domain
    /// and coherent requester validation as start. Observations are the
    /// pre-EBS GetVariable results, kept in protected canonical
    /// handoff storage; no invented/default observation checks.
    pub unsafe fn begin_pnvm(
        &mut self,
        provider: &'a dyn resources::Provider,
        observations: Observations,
        source: &dyn PageSource,
    ) -> Result<PnvmProgress, Error> {
        if self.state != State::Alive {
            return Err(Error::State);
        }
        let sku = self.alive_sku.ok_or(Error::State)?;
        let prepared = calibration::prepare(&self.profile, sku, observations, provider)?;
        if sku == [0; 3] {
            self.platform = Some(PlatformPhase {
                receipts: prepared.receipts,
                tables: Vec::new(),
                notification: None,
                released_tables: 0,
            });
            self.state = State::PnvmObserved;
            return Ok(PnvmProgress::Skipped);
        }
        if self
            .firmware
            .command_version(0xc, 0xfe)
            .is_none_or(|v| v.notification_version != 1)
        {
            return Err(Error::PnvmNotification);
        }
        let mut tables = Vec::new();
        let mut fields = Vec::new();
        if self.profile.family() >= Family::Ax210 {
            for (offset, chunks) in [(16, &prepared.pnvm), (60, &prepared.reduced_power)] {
                if chunks.is_empty() {
                    continue;
                }
                // SAFETY: covered by this function's own safety contract.
                let table = unsafe {
                    MappedTable::prepare(
                        chunks,
                        self.firmware.supports_capability(32),
                        source,
                        Some(self.domain.clone()),
                    )?
                };
                fields.push(offset);
                tables.push(table);
            }
        }
        // Keep the whole real requester and every table before even the
        // first possibly visible scratch field store (not just the doorbell).
        self.platform = Some(PlatformPhase {
            receipts: prepared.receipts,
            tables,
            notification: None,
            released_tables: 0,
        });
        self.state = State::PnvmWaiting;
        let result = (|| {
            let platform = self.platform.as_mut().unwrap();
            let mut configurations = Vec::new();
            for table in &mut platform.tables {
                configurations.push(table.expose_for_scratch()?);
            }
            if let Some(memory) = &self.ax210 {
                for (offset, config) in fields.into_iter().zip(configurations) {
                    let mut field = [0; 16];
                    field[..8].copy_from_slice(&config.address.to_le_bytes());
                    field[8..12].copy_from_slice(&config.bytes.to_le_bytes());
                    // SAFETY: These PNVM/reduced fields are host-owned until this
                    // protocol's doorbell; other scratch ranges remain live.
                    unsafe {
                        memory.scratch.write_owned(offset, &field)?;
                    }
                }
            }
            fence(Ordering::Release);
            let offset = if self.profile.family() >= Family::Ax210 {
                0x300000
            } else {
                0
            };
            self.prph_write(0xa05c04 + offset, 1 << 20)?;
            Ok(PnvmProgress::Waiting)
        })();
        if result.is_err() {
            self.state = State::Poisoned;
        }
        result
    }
    pub fn platform_receipts(&self) -> Option<&calibration::Receipts<'a>> {
        self.platform.as_ref().map(|p| &p.receipts)
    }
    pub fn pnvm_notification(&self) -> Option<&PnvmNotification> {
        self.platform.as_ref().and_then(|p| p.notification.as_ref())
    }
    /// Observe only the epoch group0xc/op0xfe/version1 packet.
    /// A timeout/requested cancel is never a completion or DMA release proof.
    pub fn poll_pnvm(&mut self) -> Result<PnvmProgress, Error> {
        if self.state != State::PnvmWaiting {
            return Err(Error::State);
        }
        let result = self.poll_pnvm_inner();
        if result.is_err() {
            self.state = State::Poisoned;
        }
        result
    }
    fn poll_pnvm_inner(&mut self) -> Result<PnvmProgress, Error> {
        let causes = self.causes()?;
        if causes & (interrupts::INT_HW_ERR | interrupts::INT_SW_ERR) != 0 {
            return Err(Error::DeviceError(causes));
        }
        if self.read_register(0x24)? & 0x08000000 == 0 {
            return Err(Error::RfKilled);
        }
        while let Some(bytes) = self.completed_buffer()? {
            let mut observed = false;
            for packet in Packets::new(self.profile.family(), 0, &bytes)? {
                let packet = packet?;
                if packet.command != 0xfe || packet.group != 0xc {
                    self.hcmd.retain_notification(&packet)?;
                    continue;
                }
                if observed {
                    return Err(Error::PnvmNotification);
                }
                let raw = packet.payload();
                if raw.len() < 4 {
                    return Err(Error::PnvmNotification);
                }
                let notification = PnvmNotification {
                    status: u32::from_le_bytes(raw[..4].try_into().unwrap()),
                    raw: raw.to_vec(),
                };
                self.platform.as_mut().ok_or(Error::State)?.notification = Some(notification);
                self.state = State::PnvmObserved;
                observed = true;
            }
            if observed {
                return Ok(PnvmProgress::Observed);
            }
        }
        Ok(PnvmProgress::Waiting)
    }
    /// Poll CSR causes and DMA producer state once in process context.
    /// CSR_ALIVE only allows restocking; only completed RX ALIVE returns data.
    pub fn poll(&mut self) -> Result<Option<AliveReport>, Error> {
        if !matches!(self.state, State::Published | State::Restocked) {
            return Err(Error::State);
        }
        let result = self.poll_inner();
        if result.is_err() {
            self.state = State::Poisoned;
        }
        result
    }
    fn poll_inner(&mut self) -> Result<Option<AliveReport>, Error> {
        let causes = self.causes()?;
        if causes & (interrupts::INT_HW_ERR | interrupts::INT_SW_ERR) != 0 {
            return Err(Error::DeviceError(causes));
        }
        if self.read_register(0x24)? & 0x08000000 == 0 {
            return Err(Error::RfKilled);
        }
        if self.state == State::Published && causes & 1 != 0 {
            self.restock()?;
        }
        if self.state == State::Restocked {
            self.receive()
        } else {
            Ok(None)
        }
    }
    /// Source bus-master stop acknowledgement plus reset, followed by
    /// invalidations. Failed acknowledgements never free device-visible pages;
    /// failed invalidations remain retryable with the owner kept.
    pub fn stop(&mut self) -> Result<(), Error> {
        if self.state == State::Stopped {
            return Err(Error::State);
        }
        if self.state != State::Quiescent {
            self.state = State::Poisoned;
            self.write(12, 0)?;
            if self.read_register(12)? != 0 {
                return Err(Error::Rejected { register: 12 });
            }
            self.set(0x20, 0x200)?;
            self.poll_register(0x20, 0x100, 0x100, 100)?;
            self.set(0x20, 0x80)?;
            self.io().delay_us(5000)?;
            let gp = self.read_register(0x24)?;
            self.write(0x24, gp & !4)?;
            self.state = State::Quiescent;
        }
        // SAFETY: an interrupt only wakes a core, the exclusive polling owner
        // has no queued workers and master-disable/reset acknowledged all DMA.
        unsafe {
            if !self.images_stopped {
                self.images.release_after_stop()?;
                self.images_stopped = true;
            }
            self.context.release()?;
            if let Some(memory) = self.ax210.as_mut() {
                memory.iml.release()?;
                memory.scratch.release()?;
                memory.info.release()?;
            }
            self.free.release()?;
            self.used.release()?;
            self.status.release()?;
            self.command.release()?;
            self.hcmd.release_after_stop()?;
            if let Some(platform) = self.platform.as_mut() {
                while platform.released_tables < platform.tables.len() {
                    platform.tables[platform.released_tables].release_after_stop()?;
                    platform.released_tables += 1;
                }
            }
            for buffer in &mut self.buffers {
                buffer.allocation.release()?;
            }
        }
        self.state = State::Stopped;
        Ok(())
    }
}
impl<Q: BootIo> Drop for RomBoot<'_, Q> {
    fn drop(&mut self) {
        if !matches!(self.state, State::Prepared | State::Stopped)
            && let Some(owner) = self.io.take()
        {
            core::mem::forget(owner);
        }
    }
}

/// The ROM boot every owner nests, which device models complete against.
#[cfg(test)]
mod rom {
    use super::*;
    use crate::wifi::{intel::radio::bios::service::BiosIo, regulatory::OsPolicy};
    impl<'a, Q: BootIo> RomBoot<'a, Q> {
        pub(crate) fn rom(&self) -> Option<&RomBoot<'a, Q>> {
            Some(self)
        }
    }
    impl<'a, Q: BootIo> UnifiedNvm<'a, Q> {
        pub(crate) fn rom(&self) -> Option<&RomBoot<'a, Q>> {
            Some(&self.boot)
        }
    }
    impl<'a, Q: BootIo> ColdRadio<'a, Q> {
        pub(crate) fn rom(&self) -> Option<&RomBoot<'a, Q>> {
            self.initialized.rom()
        }
    }
    impl<'a, Q: BootIo, B: BiosIo> LariRadio<'a, Q, B> {
        pub(crate) fn rom(&self) -> Option<&RomBoot<'a, Q>> {
            self.radio.rom()
        }
    }
    impl<'a, Q: BootIo, B: BiosIo> Ax200Setup<'a, Q, B> {
        pub(crate) fn rom(&self) -> Option<&RomBoot<'a, Q>> {
            self.lari.rom()
        }
    }
    impl<'a, Q: BootIo, B: BiosIo> Ax200Mcc<'a, Q, B> {
        pub(crate) fn rom(&self) -> Option<&RomBoot<'a, Q>> {
            self.setup.rom()
        }
    }
    impl<'a, Q: BootIo, B: BiosIo> Ax200ScanConfiguration<'a, Q, B> {
        pub(crate) fn rom(&self) -> Option<&RomBoot<'a, Q>> {
            self.mcc.rom()
        }
    }
    impl<'a, Q: BootIo, B: BiosIo> Ax200BoardPolicy<'a, Q, B> {
        pub(crate) fn rom(&self) -> Option<&RomBoot<'a, Q>> {
            self.scan.rom()
        }
    }
    impl<'a, Q: BootIo, B: BiosIo, O: CountryIo> Ax200Country<'a, Q, B, O> {
        pub(crate) fn rom(&self) -> Option<&RomBoot<'a, Q>> {
            self.board.rom()
        }
    }
    impl<'a, Q: BootIo, B: BiosIo, O: CountryIo> Ax200Mac<'a, Q, B, O> {
        pub(crate) fn rom(&self) -> Option<&RomBoot<'a, Q>> {
            self.country.rom()
        }
    }
    impl<'a, Q: BootIo, B: BiosIo, O: CountryIo + OsPolicy> Ax200Scan<'a, Q, B, O> {
        pub(crate) fn rom(&self) -> Option<&RomBoot<'a, Q>> {
            self.mac.rom()
        }
    }
    impl<'a, Q: BootIo, B: BiosIo, O: CountryIo + OsPolicy> Ax200Phy<'a, Q, B, O> {
        pub(crate) fn rom(&self) -> Option<&RomBoot<'a, Q>> {
            self.scan.rom()
        }
    }
    impl<'a, Q: BootIo, B: BiosIo, O: CountryIo + OsPolicy> Ax200TransmitPower<'a, Q, B, O> {
        pub(crate) fn rom(&self) -> Option<&RomBoot<'a, Q>> {
            self.phy.rom()
        }
    }
    impl<'a, Q: BootIo, B: BiosIo, O: CountryIo + OsPolicy> Ax200Binding<'a, Q, B, O> {
        pub(crate) fn rom(&self) -> Option<&RomBoot<'a, Q>> {
            self.power.rom()
        }
    }
    impl<'a, Q: BootIo, B: BiosIo, O: CountryIo + OsPolicy> Ax200LegacyPeer<'a, Q, B, O> {
        pub(crate) fn rom(&self) -> Option<&RomBoot<'a, Q>> {
            self.binding.rom()
        }
    }
    impl<'a, Q: BootIo, B: BiosIo, O: CountryIo + OsPolicy> Ax200ManagementQueue<'a, Q, B, O> {
        pub(crate) fn rom(&self) -> Option<&RomBoot<'a, Q>> {
            self.peer.as_ref()?.rom()
        }
    }
    impl<'a, Q: BootIo, B: BiosIo, O: CountryIo + OsPolicy> Ax200Coexistence<'a, Q, B, O> {
        pub(crate) fn rom(&self) -> Option<&RomBoot<'a, Q>> {
            self.queue.rom()
        }
    }
}

#[cfg(test)]
mod tests;

impl From<crate::block::BlockError> for Error {
    fn from(error: crate::block::BlockError) -> Self {
        Self::Pci(error)
    }
}

#[cfg(test)]
use crate::Clock;
