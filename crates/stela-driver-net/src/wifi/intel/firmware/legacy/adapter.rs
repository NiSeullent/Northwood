// SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause
// Copyright (C) 2005-2015, 2018-2026 Intel Corporation
// Copyright (C) 2013-2015 Intel Mobile Communications GmbH
// Copyright (C) 2016-2017 Intel Deutschland GmbH
//! One polled Intel 9000 requester through secure INIT and REGULAR loading.
//! The CPU release, load status and IRQ ordering follow Linux iwlwifi
//! cc9de5d4b5ab6459f5774946aa3195ad51236fa8.

use super::{
    Cpu, FH_DMA_LIMIT, Firmware, InitError, InitStatus, InitStep, LoadPlan, Sku,
    alive::{self, Progress, Sequence},
    hcmd, nvm, paging, phy, phy_db,
    rfh::{self, Receiver},
    rx::{self, CompletedBuffer, Queue, ResponseCutoff},
    service::{self, ICT, ICT_ENABLED, INT_FH_TX, INT_MASK, TransferState},
};
use crate::{
    bus::dma::{
        mapping::{DmaDomain, DmaError},
        memory::PageSource,
    },
    wifi::intel::{
        boot::BootIo,
        command::Id,
        nvm::HardwareAddress,
        power::{AccessError, Csr},
        radio,
        receive::Packet,
    },
};
use alloc::{sync::Arc, vec::Vec};
use core::num::NonZeroUsize;
use stela_crypto::{Sha256, digest::Digest};
use stela_driver_base::{Clock, Deadline};

mod calibration;
mod regular;
pub use calibration::CalibrationProgress;
pub use regular::{
    BluetoothCoexistenceProgress, PagingProgress, PhyConfigurationProgress, PhyDbProgress,
    RegularAdapter, RegularHandoff, RegularPrepareError, SocConfigurationProgress,
};

const RELEASE_CPU_RESET: u32 = 0x300c;
const RELEASE_CPU_RESET_BIT: u32 = 1 << 24;
const FH_UCODE_LOAD_STATUS: u32 = 0x1af0;
const UCODE_DRV_GP1_CLEAR: u32 = 0x5c;
const SW_RFKILL: u32 = 1 << 1;
const COMMAND_BLOCKED: u32 = 1 << 2;
const COMMAND_TIMEOUT_US: u64 = 2_000_000;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    Firmware(super::Error),
    Alive(alive::Error),
    Receiver(rfh::Error),
    Service(service::Error),
    Command(hcmd::Error),
    Nvm(nvm::Error),
    Paging(paging::Error),
    Phy(phy::Error),
    PhyDb(phy_db::Error),
    Bluetooth(radio::BluetoothConfigError),
    SocConfiguration(radio::SocConfigurationError),
    DuplicateNvmReply,
    DuplicateCommandReply,
    CommandTimeout,
    RfKilled,
    FirmwareChanged,
    Drain(DmaError),
    Cpu2Sections,
    LoadStatus { expected: u32, actual: u32 },
    State,
}

impl From<alive::Error> for Error {
    fn from(error: alive::Error) -> Self {
        Self::Alive(error)
    }
}

impl From<rfh::Error> for Error {
    fn from(error: rfh::Error) -> Self {
        Self::Receiver(error)
    }
}

impl From<rx::Error> for Error {
    fn from(error: rx::Error) -> Self {
        Self::Receiver(error.into())
    }
}

impl From<service::Error> for Error {
    fn from(error: service::Error) -> Self {
        Self::Service(error)
    }
}

impl From<AccessError> for Error {
    fn from(error: AccessError) -> Self {
        Self::Service(error.into())
    }
}

impl From<hcmd::Error> for Error {
    fn from(error: hcmd::Error) -> Self {
        Self::Command(error)
    }
}

impl From<nvm::Error> for Error {
    fn from(error: nvm::Error) -> Self {
        Self::Nvm(error)
    }
}

impl From<paging::Error> for Error {
    fn from(error: paging::Error) -> Self {
        Self::Paging(error)
    }
}

impl From<phy::Error> for Error {
    fn from(error: phy::Error) -> Self {
        Self::Phy(error)
    }
}

impl From<phy_db::Error> for Error {
    fn from(error: phy_db::Error) -> Self {
        Self::PhyDb(error)
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum NvmProgress {
    Pending,
    Data(Vec<u8>),
    Absent,
    End,
}

#[derive(Clone, Copy)]
struct PendingNvm {
    request: nvm::Request,
    ticket: hcmd::Ticket,
    deadline: Deadline,
    cutoff: ResponseCutoff,
}

#[derive(Clone, Copy)]
struct PendingCommand {
    id: Id,
    ticket: hcmd::Ticket,
    cutoff: ResponseCutoff,
    deadline: Deadline,
}

impl PendingCommand {
    fn matches(self, packet: &Packet<'_>) -> bool {
        !packet.unsolicited()
            && (packet.group, packet.command, packet.sequence)
                == (self.id.group, self.id.opcode, self.ticket.sequence())
    }

    fn check_deadline(self) -> Result<(), Error> {
        if self.deadline.passed() {
            return Err(Error::CommandTimeout);
        }
        Ok(())
    }

    fn reply(self, buffer: Option<&CompletedBuffer>) -> Result<bool, Error> {
        let Some(buffer) = buffer.filter(|buffer| self.cutoff.accepts(buffer)) else {
            return Ok(false);
        };
        let mut reply = false;
        for packet in buffer.checked_packets().map_err(rx::Error::from)? {
            let packet = packet.map_err(rx::Error::from)?;
            record_reply(self.matches(&packet), &mut reply)?;
        }
        Ok(reply)
    }
}

fn record_reply(matches: bool, reply: &mut bool) -> Result<(), Error> {
    if matches && core::mem::replace(reply, true) {
        return Err(Error::DuplicateCommandReply);
    }
    Ok(())
}

#[derive(Default)]
struct IdentitySections {
    software: Option<[u8; 10]>,
    phy_sku: Option<[u8; 8]>,
}

impl IdentitySections {
    fn observe(&mut self, request: nvm::Request, progress: &NvmProgress) {
        if request.offset() != 0 {
            return;
        }
        let bytes = match progress {
            NvmProgress::Data(bytes) => bytes.as_slice(),
            _ => &[],
        };
        match request.section() {
            nvm::Section::SW => self.software = bytes.first_chunk().copied(),
            nvm::Section::PHY_SKU => self.phy_sku = bytes.first_chunk().copied(),
            _ => {}
        }
    }
}

enum State {
    Prepared,
    Faulted,
    Waiting,
    Alive(alive::Report),
    NvmReady,
    NvmWaiting(PendingNvm),
    Calibrating(calibration::Attempt),
    Calibrated,
    PagingWaiting(PendingCommand),
    PagingReady,
    PhyDbWaiting {
        pending: PendingCommand,
        sent: Option<phy_db::Key>,
    },
    PhyDbReady,
    PhyConfigurationWaiting(PendingCommand),
    PhyReady,
    BluetoothWaiting(PendingCommand),
    BluetoothReady,
    SocConfigurationWaiting(PendingCommand),
    SocConfigurationReady,
    Stopped,
}

/// Retains NIC access custody while stopping an inherited Intel 9000 engine.
#[derive(Default)]
pub struct InheritedStop {
    nic_requested: bool,
}

impl InheritedStop {
    /// Mask device interrupts, halt FH TX and RFH, and acknowledge master
    /// disable before reset. The original root must still drain before binding.
    ///
    /// # Safety
    /// `io` accesses the awake, identified original 31dc through its kept
    /// uncached BAR. Exclude prior configuration, interrupt and DMA submitters
    /// and preserve all old DMA backing and the BAR. Retry with the same state
    /// and original accessor if any step fails. The reset requires separate
    /// MAC/RF preparation before a later INIT upload.
    pub unsafe fn halt(&mut self, io: &mut impl Csr) -> Result<(), Error> {
        service::read(io, INT_MASK)?;
        rfh::stop_engines(io, &mut self.nic_requested, |io| {
            hcmd::halt(io).map_err(Error::from)
        })
    }
}

/// Owns one INIT attempt, including the PCI/BAR claim, RFH queue, and any
/// outstanding FH DMA lease.
pub struct Adapter<'a, Q: BootIo> {
    receiver: Receiver<Q>,
    transfer: TransferState,
    commands: Option<hcmd::Queue>,
    paging: Option<paging::Pages>,
    identity_sections: IdentitySections,
    init_sequence: Option<Sequence>,
    phy_config: Option<Result<phy::Config, phy::Error>>,
    init_database: Option<phy_db::Database>,
    init_phy: Option<phy::Commands>,
    init_firmware: Option<[u8; 32]>,
    sku: Sku,
    source: &'a dyn PageSource,
    domain: Arc<dyn DmaDomain>,
    clock: Clock,
    state: State,
    needs_drain: bool,
}

impl<'a, Q: BootIo> Adapter<'a, Q> {
    /// Prepare DMA for a fresh RX generation before touching MMIO.
    ///
    /// # Safety
    /// `io` must hold the exclusive, already awake Intel 9000 function and
    /// BAR for `sku`. The caller must identify the MAC/RF and prepare the
    /// gen1 hardware with no active DMA, direct CSR causes, and no ICT,
    /// MSI-X, PCI interrupt delivery, ATS, PCIe No Snoop or relaxed ordering.
    /// No other owner or worker may access or restart the function. `domain`
    /// must be that exact requester's coherent domain, `source` must supply
    /// compatible retained pages, and `generation` must be fresh. `clock`
    /// must come from the calibrated clock producer for this execution
    /// context.
    pub unsafe fn new(
        io: Q,
        source: &'a dyn PageSource,
        domain: Arc<dyn DmaDomain>,
        sku: Sku,
        generation: u64,
        clock: Clock,
    ) -> Result<Self, Error> {
        clock
            .check_duration(alive::TIMEOUT_US)
            .map_err(alive::Error::Clock)?;
        let queue = Queue::prepare(source, domain.clone(), FH_DMA_LIMIT - 1, generation)?;
        // SAFETY: the caller binds this prepared requester to the queue's domain.
        let receiver = unsafe { Receiver::new(io, queue, sku) };
        Ok(Self::from_receiver(receiver, source, domain, sku, clock))
    }

    fn from_receiver(
        receiver: Receiver<Q>,
        source: &'a dyn PageSource,
        domain: Arc<dyn DmaDomain>,
        sku: Sku,
        clock: Clock,
    ) -> Self {
        Self {
            receiver,
            transfer: TransferState::new(),
            commands: None,
            paging: None,
            identity_sections: IdentitySections::default(),
            init_sequence: None,
            phy_config: None,
            init_database: None,
            init_phy: None,
            init_firmware: None,
            sku,
            source,
            domain,
            clock,
            state: State::Prepared,
            needs_drain: false,
        }
    }

    /// Upload INIT and arm the ALIVE deadline after both CPU status writes.
    /// A failure after device programming begins requires stop before any
    /// new attempt. The direct RX mask supports the released API 46 INIT
    /// layout, where CPU2 has a completion write but no resident sections.
    pub fn start_init(
        &mut self,
        firmware: &Firmware<'_>,
        staging_capacity: NonZeroUsize,
    ) -> Result<(), Error> {
        self.start_init_with(firmware, staging_capacity, |_| Ok(()))
    }

    /// Register the Gen1 command DMA layout before starting INIT firmware.
    /// The scheduler stays inactive until the current INIT ALIVE is received.
    pub fn start_nvm_init(
        &mut self,
        firmware: &Firmware<'_>,
        staging_capacity: NonZeroUsize,
    ) -> Result<(), Error> {
        self.start_nvm_init_with(
            firmware,
            staging_capacity,
            hcmd::Queue::prepare_full,
            Self::prepare_hardware_commands,
        )
    }

    /// Stage an owned command DMA layout for the same requester's INIT.
    ///
    /// # Safety
    /// `prepare` must install only this layout in the exclusive, stopped
    /// Gen1 FH TX engine. Keep the FIFOs and command doorbell inactive until
    /// the INIT ALIVE handoff. Do not publish other DMA addresses, retain an
    /// accessor, enable PCI interrupt delivery, or change the direct CSR
    /// interrupt mode. On failure the adapter retains every published pin.
    pub unsafe fn start_init_with_nvm(
        &mut self,
        firmware: &Firmware<'_>,
        staging_capacity: NonZeroUsize,
        prepare: impl FnOnce(&mut Q, hcmd::Layout) -> Result<(), hcmd::Error>,
    ) -> Result<(), Error> {
        self.start_nvm_init_with(
            firmware,
            staging_capacity,
            hcmd::Queue::prepare,
            |io, _, layout| prepare(io, layout).map_err(Error::from),
        )
    }

    fn start_nvm_init_with(
        &mut self,
        firmware: &Firmware<'_>,
        staging_capacity: NonZeroUsize,
        allocate: impl FnOnce(&dyn PageSource, Arc<dyn DmaDomain>) -> Result<hcmd::Queue, hcmd::Error>,
        prepare: impl FnOnce(&mut Q, &mut bool, hcmd::Layout) -> Result<(), Error>,
    ) -> Result<(), Error> {
        if !matches!(self.state, State::Prepared) {
            return Err(Error::State);
        }
        nvm::validate_firmware(firmware)?;
        self.start_init_with(firmware, staging_capacity, |adapter| {
            adapter.stage_commands(allocate, prepare)
        })
    }

    fn stage_commands(
        &mut self,
        allocate: impl FnOnce(&dyn PageSource, Arc<dyn DmaDomain>) -> Result<hcmd::Queue, hcmd::Error>,
        prepare: impl FnOnce(&mut Q, &mut bool, hcmd::Layout) -> Result<(), Error>,
    ) -> Result<(), Error> {
        let mut commands = allocate(self.source, self.domain.clone())?;
        // SAFETY: the same exclusive requester uses this layout. Keep the
        // owner and pins if preparation fails after exposing an address.
        let layout = unsafe { commands.expose_for_preparation()? };
        self.commands = Some(commands);
        let (io, requested) = self.receiver.control_io()?;
        prepare(io, requested, layout)
    }

    fn prepare_hardware_commands(
        io: &mut Q,
        requested: &mut bool,
        layout: hcmd::Layout,
    ) -> Result<(), Error> {
        service::with_nic(io, requested, |io| {
            hcmd::hardware::prepare(io, layout).map_err(Error::from)
        })
    }

    fn start_init_with(
        &mut self,
        firmware: &Firmware<'_>,
        staging_capacity: NonZeroUsize,
        prepare_commands: impl FnOnce(&mut Self) -> Result<(), Error>,
    ) -> Result<(), Error> {
        if !matches!(self.state, State::Prepared) {
            return Err(Error::State);
        }
        let plan = LoadPlan::from_firmware(firmware).map_err(Error::Firmware)?;
        alive::validate_firmware(firmware)?;
        if !plan.init.sections(Cpu::Second).is_empty() {
            return Err(Error::Cpu2Sections);
        }
        let mut prepare_commands = Some(prepare_commands);
        plan.upload_init(staging_capacity, |step| {
            self.execute(step, alive::Image::Init, &mut prepare_commands)
        })
        .map_err(upload_error)?;
        self.init_sequence = Some(self.receiver.wait_init(firmware, self.clock)?);
        self.init_firmware = Some(firmware_digest(firmware));
        self.phy_config = Some(phy::Config::from_firmware(firmware));
        self.state = State::Waiting;
        Ok(())
    }

    fn execute(
        &mut self,
        step: InitStep<'_>,
        image: alive::Image,
        prepare_commands: &mut Option<impl FnOnce(&mut Self) -> Result<(), Error>>,
    ) -> Result<(), Error> {
        let second_mask = match image {
            alive::Image::Init => rfh::RX_MASK,
            alive::Image::Regular => rfh::FH_RX_MASK,
        };
        match step {
            InitStep::CheckFresh => self.begin_upload(prepare_commands.take().ok_or(Error::State)?),
            InitStep::Transfer { cpu, transfer, .. } => {
                let (io, requested) = self.receiver.control_io()?;
                if cpu == Cpu::Second && image == alive::Image::Regular {
                    return self
                        .transfer
                        .transfer_with_rx(io, requested, self.source, &self.domain, transfer)
                        .map_err(Error::from);
                }
                self.transfer
                    .transfer(io, requested, self.source, &self.domain, transfer)
                    .map_err(Error::from)
            }
            InitStep::SectionLoaded { status, .. } | InitStep::CpuComplete { status, .. } => {
                self.publish_status(status, second_mask)
            }
            InitStep::EnableInterrupts(Cpu::First) => {
                let (io, _) = self.receiver.control_io()?;
                require_direct(io, INT_FH_TX)?;
                match image {
                    alive::Image::Init => self.receiver.enable_interrupts()?,
                    alive::Image::Regular => self.receiver.enable_interrupts_for_fh()?,
                }
                Ok(())
            }
            InitStep::EnableInterrupts(Cpu::Second) => {
                let (io, _) = self.receiver.control_io()?;
                require_direct(io, second_mask)
            }
        }
    }

    fn begin_upload(
        &mut self,
        prepare_commands: impl FnOnce(&mut Self) -> Result<(), Error>,
    ) -> Result<(), Error> {
        self.state = State::Faulted;
        let needs_drain = &mut self.needs_drain;
        self.receiver.start_with_preparation(|io| {
            *needs_drain = true;
            io.write32(UCODE_DRV_GP1_CLEAR, SW_RFKILL)?;
            io.write32(UCODE_DRV_GP1_CLEAR, COMMAND_BLOCKED)?;
            Ok(())
        })?;
        prepare_commands(self)?;
        let (io, requested) = self.receiver.control_io()?;
        service::with_nic(io, requested, |io| {
            expect_status(io, 0)?;
            io.write32(INT_MASK, INT_FH_TX)?;
            require_direct(io, INT_FH_TX)?;
            io.write32(UCODE_DRV_GP1_CLEAR, SW_RFKILL)?;
            io.write32(UCODE_DRV_GP1_CLEAR, SW_RFKILL)?;
            service::write_prph(io, RELEASE_CPU_RESET, RELEASE_CPU_RESET_BIT)?;
            // The GP_CONTROL read when releasing NIC access flushes the
            // posted PRPH write. The donor does not read back this latch.
            Ok(())
        })
    }

    fn publish_status(&mut self, status: InitStatus, final_mask: u32) -> Result<(), Error> {
        let (io, requested) = self.receiver.control_io()?;
        service::with_nic(io, requested, |io| {
            expect_status(io, status.before)?;
            io.write32(FH_UCODE_LOAD_STATUS, status.after)?;
            expect_status(io, status.after)?;
            if status.after == u32::MAX {
                // A read of all ones can also indicate lost MMIO. Require
                // independent CSR observations and successful NIC release.
                require_direct(io, final_mask)?;
            }
            Ok(())
        })
    }

    /// Poll the owned RFH queue and its deadline for a v3 INIT ALIVE.
    /// Only a packet reclaimed from this generation can complete the wait.
    pub fn poll_init_alive(&mut self) -> Result<Progress, Error> {
        let sequence = self.init_sequence.as_mut().ok_or(Error::State)?;
        Self::poll_alive(&mut self.receiver, &mut self.state, sequence)
    }

    fn poll_alive(
        receiver: &mut Receiver<Q>,
        state: &mut State,
        sequence: &mut Sequence,
    ) -> Result<Progress, Error> {
        if !matches!(state, State::Waiting) {
            return Err(Error::State);
        }
        let result = receiver.poll().map_err(Error::from).and_then(|buffer| {
            rx::poll_notification(sequence, buffer.as_ref()).map_err(Error::from)
        });
        match &result {
            Ok(Progress::Alive { report, .. }) => *state = State::Alive(*report),
            Err(_) => *state = State::Faulted,
            _ => {}
        }
        result
    }

    /// Start the Gen1 scheduler for queue zero after a current INIT ALIVE.
    /// Only the NVM command producer is admitted.
    pub fn activate_nvm_queue(&mut self) -> Result<(), Error> {
        self.activate_command_queue_with(Self::activate_hardware_commands)?;
        self.state = State::NvmReady;
        Ok(())
    }

    /// Admit the staged command queue after a valid current INIT ALIVE.
    ///
    /// # Safety
    /// On success `activate` has programmed the same Gen1 queue zero and
    /// command FIFO, initially at slot zero, using this layout. The active
    /// queue has no external producers, no extra published DMA owners and
    /// no callbacks. Keep direct CSR causes and PCI interrupts disabled.
    pub unsafe fn activate_nvm_commands(
        &mut self,
        activate: impl FnOnce(&mut Q, hcmd::Layout) -> Result<(), hcmd::Error>,
    ) -> Result<(), Error> {
        self.activate_command_queue_with(|io, _, layout, _| {
            activate(io, layout).map_err(Error::from)
        })?;
        self.state = State::NvmReady;
        Ok(())
    }

    fn activate_hardware_commands(
        io: &mut Q,
        requested: &mut bool,
        layout: hcmd::Layout,
        report: alive::Report,
    ) -> Result<(), Error> {
        service::with_nic(io, requested, |io| {
            hcmd::hardware::activate(io, layout, report.scheduler_base()).map_err(Error::from)
        })
    }

    fn activate_command_queue_with(
        &mut self,
        activate: impl FnOnce(&mut Q, &mut bool, hcmd::Layout, alive::Report) -> Result<(), Error>,
    ) -> Result<(), Error> {
        let State::Alive(report) = self.state else {
            return Err(Error::State);
        };
        let commands = self.commands.as_mut().ok_or(Error::State)?;
        self.state = State::Faulted;
        let (io, requested) = self.receiver.control_io()?;
        require_direct(io, rfh::RX_MASK)?;
        activate(io, requested, commands.layout(), report)?;
        require_direct(io, rfh::RX_MASK)?;
        // SAFETY: successful activation established the live FIFO for this
        // layout, with all publication restricted to this owner.
        unsafe { commands.admit_prepared()? };
        Ok(())
    }

    fn retire_command(&mut self, pending: PendingCommand) -> Result<(), Error> {
        let commands = self.commands.as_mut().ok_or(Error::State)?;
        // SAFETY: the caller validated exactly one matching solicited reply
        // from the current RFH generation after this command's cutoff.
        unsafe { commands.complete(pending.ticket.sequence())? };
        Ok(())
    }

    /// Submit a bounded read-only request and wait for its exact RFH reply.
    pub fn start_nvm_read(&mut self, request: nvm::Request) -> Result<(), Error> {
        if !matches!(self.state, State::NvmReady) {
            return Err(Error::State);
        }
        let deadline = self
            .clock
            .try_deadline(COMMAND_TIMEOUT_US)
            .map_err(alive::Error::Clock)?;
        self.state = State::Faulted;
        let cutoff = self.receiver.response_cutoff()?;
        let commands = self.commands.as_mut().ok_or(Error::State)?;
        let (io, requested) = self.receiver.control_io()?;
        let ticket = service::with_nic(io, requested, |io| {
            require_direct(io, rfh::RX_MASK)?;
            commands.submit_nvm_read(io, request).map_err(Error::from)
        })?;
        self.state = State::NvmWaiting(PendingNvm {
            request,
            ticket,
            deadline,
            cutoff,
        });
        Ok(())
    }

    pub fn poll_nvm_read(&mut self) -> Result<NvmProgress, Error> {
        let State::NvmWaiting(pending) = self.state else {
            return Err(Error::State);
        };
        self.state = State::Faulted;
        if pending.deadline.passed() {
            return Err(Error::CommandTimeout);
        }
        let buffer = self.receiver.poll()?;
        let result = pending.reply(buffer.as_ref())?;
        if pending.deadline.passed() {
            return Err(Error::CommandTimeout);
        }
        if result == NvmProgress::Pending {
            self.state = State::NvmWaiting(pending);
        } else {
            let commands = self.commands.as_mut().ok_or(Error::State)?;
            // SAFETY: the sole RX owner reclaimed a fully validated buffer
            // in this generation, after the cutoff, with this command ID,
            // exact sequence, and the expected NVM section and offset.
            unsafe { commands.complete(pending.ticket.sequence())? };
            self.identity_sections.observe(pending.request, &result);
            self.state = State::NvmReady;
        }
        Ok(result)
    }

    /// Read the MAC from this requester's strap or OTP CSRs and combine it
    /// with the SW and PHY_SKU prefixes from matched offset-zero NVM replies.
    pub fn nvm_identity(&mut self) -> Result<nvm::Identity, Error> {
        if !matches!(self.state, State::NvmReady) {
            return Err(Error::State);
        }
        let software = self.identity_sections.software.ok_or(Error::State)?;
        let phy_sku = self.identity_sections.phy_sku.ok_or(Error::State)?;
        self.state = State::Faulted;
        let (io, requested) = self.receiver.control_io()?;
        let (strap, otp) = service::with_nic(io, requested, |io| {
            require_direct(io, rfh::RX_MASK)?;
            let strap = read_pair(io, nvm::MAC_STRAP)?;
            let otp = if HardwareAddress::valid(HardwareAddress::decode(strap[0], strap[1])) {
                None
            } else {
                Some(read_pair(io, nvm::MAC_OTP)?)
            };
            // An all-ones word can be a blank strap or OTP. Verify the CSR
            // route independently before interpreting the address bytes.
            require_direct(io, rfh::RX_MASK)?;
            Ok::<_, Error>((strap, otp))
        })?;
        self.state = State::NvmReady;
        Ok(nvm::Identity::decode(&software, &phy_sku, strap, otp)?)
    }

    /// Stop the requester, drain translated writes, then retire all DMA
    /// leases. Retry failed drain or invalidation with the same owner.
    pub fn stop(&mut self) -> Result<(), Error> {
        if matches!(self.state, State::Stopped) {
            return Err(Error::State);
        }
        self.state = State::Faulted;
        if self.commands.is_some() {
            self.receiver
                .quiesce_with(|io| hcmd::halt(io).map_err(Error::from))?;
        } else {
            self.receiver.quiesce()?;
        }
        if self.needs_drain {
            // SAFETY: quiesce confirmed RFH idle and requester master
            // disable/reset. This owner keeps all target mappings and no
            // external IRQ or worker can post DMA in this epoch.
            unsafe { self.domain.drain_posted_writes() }.map_err(Error::Drain)?;
            self.needs_drain = false;
        }
        // SAFETY: quiescence and the translated write drain cover this
        // exact requester's FH mapping as well as RFH.
        unsafe { self.transfer.release_pin()? };
        if let Some(commands) = &mut self.commands {
            // SAFETY: all FH TX channels were observed idle, then requester
            // stop/reset and the translated write drain completed.
            unsafe { commands.release_after_stop()? };
        }
        if let Some(paging) = &mut self.paging {
            // SAFETY: the firmware can no longer reference paging addresses
            // after the same requester-wide stop/reset and domain drain.
            unsafe { paging.release_after_stop()? };
        }
        self.receiver.stop()?;
        self.state = State::Stopped;
        Ok(())
    }

    /// Return the PCI/BAR owner only after successful stop and unmap.
    pub fn finish(&mut self) -> Result<Q, Error> {
        self.receiver.finish().map_err(Error::from)
    }
}

fn read_pair(io: &mut impl Csr, [first, second]: [u32; 2]) -> Result<[u32; 2], AccessError> {
    Ok([io.read32(first)?, io.read32(second)?])
}

impl PendingNvm {
    fn reply(self, buffer: Option<&CompletedBuffer>) -> Result<NvmProgress, Error> {
        let Some(buffer) = buffer.filter(|buffer| self.cutoff.accepts(buffer)) else {
            return Ok(NvmProgress::Pending);
        };
        let expected = self.request.id().wire();
        let mut reply = None;
        for packet in buffer.packets().map_err(rx::Error::from)? {
            let packet = packet.map_err(rx::Error::from)?;
            if (packet.group, packet.command, packet.sequence)
                != (expected.group, expected.opcode, self.ticket.sequence())
            {
                continue;
            }
            if reply.is_some() {
                return Err(Error::DuplicateNvmReply);
            }
            reply = Some(self.request.decode(packet.payload())?);
        }
        Ok(match reply {
            Some(nvm::Read::Data(data)) => NvmProgress::Data(data.to_vec()),
            Some(nvm::Read::Absent) => NvmProgress::Absent,
            Some(nvm::Read::End) => NvmProgress::End,
            None => NvmProgress::Pending,
        })
    }
}

fn require_direct(io: &mut impl Csr, mask: u32) -> Result<(), Error> {
    if service::read(io, ICT)? & ICT_ENABLED != 0 || service::read(io, INT_MASK)? != mask {
        return Err(service::Error::InterruptMode.into());
    }
    service::causes(io)?;
    Ok(())
}

fn firmware_digest(firmware: &Firmware<'_>) -> [u8; 32] {
    Sha256::digest(firmware.bytes).into()
}

fn upload_error(error: InitError<Error>) -> Error {
    match error {
        InitError::Plan(error) => Error::Firmware(error),
        InitError::Step(error) => error,
    }
}

fn expect_status(io: &mut impl Csr, expected: u32) -> Result<(), Error> {
    let actual = if expected == u32::MAX {
        io.read32(FH_UCODE_LOAD_STATUS)?
    } else {
        service::read(io, FH_UCODE_LOAD_STATUS)?
    };
    if actual != expected {
        return Err(Error::LoadStatus { expected, actual });
    }
    Ok(())
}

#[cfg(test)]
mod tests;
