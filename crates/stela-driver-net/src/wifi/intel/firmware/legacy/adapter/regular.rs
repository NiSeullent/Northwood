// SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause
// Copyright (C) 2005-2015, 2018-2026 Intel Corporation
// Copyright (C) 2013-2015 Intel Mobile Communications GmbH
// Copyright (C) 2016-2017 Intel Deutschland GmbH
//! Polled REGULAR firmware load, paging, and radio initialization.
//! The restart and resident-load order follows Linux iwlwifi gen1_2/trans.c
//! and mvm/fw.c at cc9de5d4b5ab6459f5774946aa3195ad51236fa8.

use super::{
    Adapter, BootIo, COMMAND_TIMEOUT_US, Clock, DmaDomain, Error, FH_DMA_LIMIT, Firmware, Id,
    InitStep, LoadPlan, PageSource, PendingCommand, Progress, Queue, Receiver, Sku, State, alive,
    calibration, expect_status, firmware_digest, hcmd, paging, phy, phy_db, radio, require_direct,
    rfh, service, upload_error,
};
use alloc::sync::Arc;
use core::{fmt, num::NonZeroUsize};

/// The stopped INIT owner, its receive sequence and collected PHY DB.
pub struct RegularHandoff<'a, Q: BootIo> {
    pub(super) io: Q,
    pub(super) sequence: alive::Sequence,
    pub(super) database: phy_db::Database,
    pub(super) init_phy: phy::Commands,
    pub(super) firmware: [u8; 32],
    pub(super) sku: Sku,
    pub(super) source: &'a dyn PageSource,
    pub(super) domain: Arc<dyn DmaDomain>,
    pub(super) clock: Clock,
}

impl<'a, Q: BootIo> RegularHandoff<'a, Q> {
    pub fn phy_db(&self) -> &phy_db::Database {
        &self.database
    }

    /// Recover the stopped requester, sequence and PHY DB separately.
    pub fn into_parts(self) -> (Q, alive::Sequence, phy_db::Database) {
        (self.io, self.sequence, self.database)
    }

    /// Allocate a new RFH generation in the same requester's DMA domain.
    /// Validation and allocation failures return the original handoff.
    pub fn prepare_regular<'fw>(
        self,
        generation: u64,
        firmware: &Firmware<'fw>,
    ) -> Result<RegularAdapter<'a, 'fw, Q>, RegularPrepareError<'a, Q>> {
        let prepare = || -> Result<_, Error> {
            self.sequence.check_regular_generation(generation)?;
            if firmware_digest(firmware) != self.firmware {
                return Err(Error::FirmwareChanged);
            }
            let plan = LoadPlan::from_firmware(firmware).map_err(Error::Firmware)?;
            let queue = Queue::prepare(
                self.source,
                self.domain.clone(),
                FH_DMA_LIMIT - 1,
                generation,
            )?;
            Ok((plan, queue))
        };
        let (plan, queue) = match prepare() {
            Ok(prepared) => prepared,
            Err(error) => {
                return Err(RegularPrepareError {
                    error,
                    handoff: self,
                });
            }
        };
        // SAFETY: the stopped handoff owns the same PCI function, clock and
        // DMA domain. Start requires exclusive device preparation before
        // exposing any address from the new RFH generation.
        let receiver = unsafe { Receiver::after_init_stop(self.io, queue, self.sku) };
        let adapter =
            Adapter::from_receiver(receiver, self.source, self.domain, self.sku, self.clock);
        let paging_layout = paging::Layout::from_plan(firmware, &plan);
        let phy_configuration = phy::Config::from_regular_firmware(firmware)
            .map(|config| config.regular_request(self.init_phy));
        Ok(RegularAdapter {
            adapter,
            plan,
            paging_layout,
            sequence: self.sequence,
            database: self.database,
            antenna: self.init_phy.antenna(),
            phy_configuration,
            bluetooth_configuration: radio::BluetoothConfig::from_firmware(firmware),
            // This handoff is issued by the verified 31dc INIT path.
            soc_configuration: radio::SocConfiguration::for_31dc(firmware),
            generation,
        })
    }
}

pub struct RegularPrepareError<'a, Q: BootIo> {
    pub error: Error,
    pub handoff: RegularHandoff<'a, Q>,
}

impl<Q: BootIo> fmt::Debug for RegularPrepareError<'_, Q> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RegularPrepareError")
            .field("error", &self.error)
            .finish_non_exhaustive()
    }
}

/// Owns the REGULAR RFH generation through PHY, Bluetooth coexistence and
/// SoC configuration, before creation of MAC and station contexts.
pub struct RegularAdapter<'a, 'fw, Q: BootIo> {
    adapter: Adapter<'a, Q>,
    plan: LoadPlan<'fw>,
    paging_layout: Result<paging::Layout<'fw>, paging::Error>,
    sequence: alive::Sequence,
    database: phy_db::Database,
    antenna: phy::Request,
    phy_configuration: Result<phy::Request, phy::Error>,
    bluetooth_configuration: Result<radio::BluetoothConfig, radio::BluetoothConfigError>,
    soc_configuration: Result<radio::SocConfiguration, radio::SocConfigurationError>,
    generation: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PagingProgress {
    Pending,
    Complete,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PhyDbProgress {
    Pending,
    Complete,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PhyConfigurationProgress {
    Pending,
    Complete,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BluetoothCoexistenceProgress {
    Pending,
    Complete,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SocConfigurationProgress {
    Pending,
    Complete,
}

impl<Q: BootIo> RegularAdapter<'_, '_, Q> {
    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn phy_db(&self) -> &phy_db::Database {
        &self.database
    }

    /// Restart the stopped requester, upload both resident CPUs, and arm
    /// the ALIVE deadline against the newly exposed RFH generation.
    ///
    /// # Safety
    /// `restart` must exclusively reset and prepare this same function and
    /// BAR with the 9000-series MAC/RF settings. The NIC must be awake with
    /// direct CSR causes, ICT disabled, zero interrupt mask and load status,
    /// and idle DMA. PCI interrupt delivery, ATS, PCIe No Snoop and relaxed
    /// ordering must be disabled. The callback must publish no DMA address,
    /// retain no accessor and leave PCI/BAR ownership intact even on failure.
    pub unsafe fn start(
        &mut self,
        staging_capacity: NonZeroUsize,
        restart: impl FnOnce(&mut Q) -> Result<(), service::Error>,
    ) -> Result<(), Error> {
        if !matches!(self.adapter.state, State::Prepared) {
            return Err(Error::State);
        }
        let adapter = &mut self.adapter;
        let mut restart = Some(restart);
        let mut prepare_commands = Some(|adapter: &mut Adapter<'_, Q>| {
            adapter.stage_commands(
                hcmd::Queue::prepare_full,
                Adapter::prepare_hardware_commands,
            )
        });
        self.plan
            .upload_regular(staging_capacity, |step| {
                if matches!(step, InitStep::CheckFresh) {
                    adapter.state = State::Faulted;
                    let restart = restart.take().ok_or(Error::State)?;
                    adapter.receiver.restart_prepared(|io| {
                        restart(io)?;
                        require_direct(io, 0)?;
                        expect_status(io, 0)?;
                        calibration::require_rf(io)
                    })?;
                }
                adapter.execute(step, alive::Image::Regular, &mut prepare_commands)
            })
            .map_err(upload_error)?;
        adapter.receiver.finish_fh_upload()?;
        calibration::require_rf(adapter.receiver.control_io()?.0)?;
        adapter
            .receiver
            .wait_regular_after_init_stop(&mut self.sequence)?;
        adapter.state = State::Waiting;
        Ok(())
    }

    /// Poll only this attempt's RFH generation for a v3 REGULAR ALIVE.
    pub fn poll_alive(&mut self) -> Result<Progress, Error> {
        Adapter::poll_alive(
            &mut self.adapter.receiver,
            &mut self.adapter.state,
            &mut self.sequence,
        )
    }

    /// After this generation's REGULAR ALIVE, activate the command scheduler
    /// and publish the permanent secure paging pages using queue zero.
    pub fn start_paging(&mut self) -> Result<(), Error> {
        let adapter = &mut self.adapter;
        if !matches!(adapter.state, State::Alive(_)) {
            return Err(Error::State);
        }
        let layout = self.paging_layout.as_ref().map_err(|error| *error)?;
        let pages = paging::Pages::prepare(layout, adapter.source, &adapter.domain)?;
        adapter.activate_command_queue_with(|io, requested, layout, report| {
            calibration::require_rf(io)?;
            Adapter::activate_hardware_commands(io, requested, layout, report)
        })?;
        let cutoff = adapter.receiver.response_cutoff()?;
        let deadline = adapter
            .clock
            .try_deadline(COMMAND_TIMEOUT_US)
            .map_err(alive::Error::Clock)?;
        let (io, requested) = adapter.receiver.control_io()?;
        let commands = adapter.commands.as_mut().ok_or(Error::State)?;
        let pages = adapter.paging.insert(pages);
        let ticket = service::with_nic(io, requested, |io| {
            require_direct(io, rfh::RX_MASK)?;
            calibration::require_rf(io)?;
            // Keep the paging pages under the original owner before the
            // command or its doorbell can make their addresses visible.
            commands
                .submit_paging(io, adapter.source, &adapter.domain, pages.expose())
                .map_err(Error::from)
        })?;
        adapter.state = State::PagingWaiting(PendingCommand {
            id: paging::ID,
            ticket,
            cutoff,
            deadline,
        });
        Ok(())
    }

    /// Retire the command only for its exact solicited response. The paging
    /// pages remain device-owned through successful stop and drain.
    pub fn poll_paging(&mut self) -> Result<PagingProgress, Error> {
        let adapter = &mut self.adapter;
        let State::PagingWaiting(pending) = adapter.state else {
            return Err(Error::State);
        };
        adapter.state = State::Faulted;
        if adapter.poll_regular_command(pending)? {
            adapter.retire_command(pending)?;
            adapter.state = State::PagingReady;
            Ok(PagingProgress::Complete)
        } else {
            adapter.state = State::PagingWaiting(pending);
            Ok(PagingProgress::Pending)
        }
    }

    /// Reapply the original INIT antenna mask after paging. Its matching
    /// reply opens replay of the retained CFG, NCH, PAPD, and TXP sections.
    pub fn start_phy_db_replay(&mut self) -> Result<(), Error> {
        let adapter = &mut self.adapter;
        if !matches!(adapter.state, State::PagingReady) {
            return Err(Error::State);
        }
        // Reserve slots for TX antenna, PHY_CFG, BT_CONFIG and SoC configuration.
        let count = self.database.replay_count()? + 4;
        adapter
            .commands
            .as_ref()
            .ok_or(Error::State)?
            .check_capacity(count)?;
        adapter.state = State::Faulted;
        let antenna = self.antenna;
        let pending = adapter.submit_regular_command(antenna.id(), |queue, io, _, _| {
            queue.submit_phy(io, antenna)
        })?;
        adapter.state = State::PhyDbWaiting {
            pending,
            sent: None,
        };
        Ok(())
    }

    /// Advance one command only after its solicited reply in the current
    /// REGULAR generation. Keep paging active and retain faulted DMA for stop.
    pub fn poll_phy_db_replay(&mut self) -> Result<PhyDbProgress, Error> {
        let adapter = &mut self.adapter;
        let State::PhyDbWaiting { pending, sent } = adapter.state else {
            return Err(Error::State);
        };
        adapter.state = State::Faulted;
        if !adapter.poll_regular_command(pending)? {
            adapter.state = State::PhyDbWaiting { pending, sent };
            return Ok(PhyDbProgress::Pending);
        }
        adapter.retire_command(pending)?;
        let Some((key, request)) = self.database.next_replay(sent) else {
            adapter.state = State::PhyDbReady;
            return Ok(PhyDbProgress::Complete);
        };
        let pending = adapter.submit_regular_command(phy_db::ID, |queue, io, source, domain| {
            queue.submit_phy_db(io, source, domain, request)
        })?;
        adapter.state = State::PhyDbWaiting {
            pending,
            sent: Some(key),
        };
        Ok(PhyDbProgress::Pending)
    }

    /// Send PHY_CFG after the last PHY DB reply, using the REGULAR image's
    /// calibration controls and this requester's verified INIT PHY word.
    pub fn start_phy_configuration(&mut self) -> Result<(), Error> {
        let adapter = &mut self.adapter;
        if !matches!(adapter.state, State::PhyDbReady) {
            return Err(Error::State);
        }
        let request = self.phy_configuration?;
        adapter.state = State::Faulted;
        let pending = adapter.submit_regular_command(request.id(), |queue, io, _, _| {
            queue.submit_phy(io, request)
        })?;
        adapter.state = State::PhyConfigurationWaiting(pending);
        Ok(())
    }

    /// Accept only the solicited PHY_CFG reply for this REGULAR command.
    pub fn poll_phy_configuration(&mut self) -> Result<PhyConfigurationProgress, Error> {
        let adapter = &mut self.adapter;
        let State::PhyConfigurationWaiting(pending) = adapter.state else {
            return Err(Error::State);
        };
        adapter.state = State::Faulted;
        if !adapter.poll_regular_command(pending)? {
            adapter.state = State::PhyConfigurationWaiting(pending);
            return Ok(PhyConfigurationProgress::Pending);
        }
        adapter.retire_command(pending)?;
        adapter.state = State::PhyReady;
        Ok(PhyConfigurationProgress::Complete)
    }

    /// Whether the current REGULAR generation acknowledged PHY_CFG.
    pub fn phy_ready(&self) -> bool {
        matches!(
            self.adapter.state,
            State::PhyReady
                | State::BluetoothWaiting(_)
                | State::BluetoothReady
                | State::SocConfigurationWaiting(_)
                | State::SocConfigurationReady
        )
    }

    /// Configure default Bluetooth coexistence after this generation's PHY_CFG reply.
    pub fn start_bluetooth_coexistence(&mut self) -> Result<(), Error> {
        let adapter = &mut self.adapter;
        if !matches!(adapter.state, State::PhyReady) {
            return Err(Error::State);
        }
        let request = self.bluetooth_configuration.map_err(Error::Bluetooth)?;
        adapter.state = State::Faulted;
        let pending = adapter.submit_regular_command(request.id(), |queue, io, _, _| {
            queue.submit_bluetooth(io, request)
        })?;
        adapter.state = State::BluetoothWaiting(pending);
        Ok(())
    }

    /// Retire BT_CONFIG only for its matching solicited reply from the current RFH generation.
    pub fn poll_bluetooth_coexistence(&mut self) -> Result<BluetoothCoexistenceProgress, Error> {
        let adapter = &mut self.adapter;
        let State::BluetoothWaiting(pending) = adapter.state else {
            return Err(Error::State);
        };
        adapter.state = State::Faulted;
        if !adapter.poll_regular_command(pending)? {
            adapter.state = State::BluetoothWaiting(pending);
            return Ok(BluetoothCoexistenceProgress::Pending);
        }
        adapter.retire_command(pending)?;
        adapter.state = State::BluetoothReady;
        Ok(BluetoothCoexistenceProgress::Complete)
    }

    /// Whether BT_CONFIG completed for this REGULAR generation.
    pub fn bluetooth_coexistence_ready(&self) -> bool {
        matches!(
            self.adapter.state,
            State::BluetoothReady
                | State::SocConfigurationWaiting(_)
                | State::SocConfigurationReady
        )
    }

    /// Submit the verified 31dc SoC settings after this generation's BT_CONFIG reply.
    pub fn start_soc_configuration(&mut self) -> Result<(), Error> {
        let adapter = &mut self.adapter;
        if !matches!(adapter.state, State::BluetoothReady) {
            return Err(Error::State);
        }
        let request = self.soc_configuration.map_err(Error::SocConfiguration)?;
        adapter.state = State::Faulted;
        let pending = adapter.submit_regular_command(request.id(), |queue, io, _, _| {
            queue.submit_soc_configuration(io, request)
        })?;
        adapter.state = State::SocConfigurationWaiting(pending);
        Ok(())
    }

    /// Retire the SoC command only on its matching solicited RFH reply.
    pub fn poll_soc_configuration(&mut self) -> Result<SocConfigurationProgress, Error> {
        let adapter = &mut self.adapter;
        let State::SocConfigurationWaiting(pending) = adapter.state else {
            return Err(Error::State);
        };
        adapter.state = State::Faulted;
        if !adapter.poll_regular_command(pending)? {
            adapter.state = State::SocConfigurationWaiting(pending);
            return Ok(SocConfigurationProgress::Pending);
        }
        adapter.retire_command(pending)?;
        adapter.state = State::SocConfigurationReady;
        Ok(SocConfigurationProgress::Complete)
    }

    /// Whether SOC_CONFIGURATION_CMD completed for this REGULAR generation.
    pub fn soc_configuration_ready(&self) -> bool {
        matches!(self.adapter.state, State::SocConfigurationReady)
    }

    /// Quiesce the requester and retire its DMA leases, with same-owner retry.
    pub fn stop(&mut self) -> Result<(), Error> {
        self.adapter.stop()
    }

    /// Recover the PCI/BAR owner after a successful stop.
    pub fn finish(&mut self) -> Result<Q, Error> {
        self.adapter.finish()
    }
}

impl<Q: BootIo> Adapter<'_, Q> {
    fn poll_regular_command(&mut self, pending: PendingCommand) -> Result<bool, Error> {
        pending.check_deadline()?;
        calibration::require_rf(self.receiver.control_io()?.0)?;
        let buffer = self.receiver.poll()?;
        let reply = pending.reply(buffer.as_ref())?;
        pending.check_deadline()?;
        Ok(reply)
    }

    fn submit_regular_command(
        &mut self,
        id: Id,
        submit: impl FnOnce(
            &mut hcmd::Queue,
            &mut Q,
            &dyn PageSource,
            &Arc<dyn DmaDomain>,
        ) -> Result<hcmd::Ticket, hcmd::Error>,
    ) -> Result<PendingCommand, Error> {
        let cutoff = self.receiver.response_cutoff()?;
        let deadline = self
            .clock
            .try_deadline(COMMAND_TIMEOUT_US)
            .map_err(alive::Error::Clock)?;
        let (io, requested) = self.receiver.control_io()?;
        let queue = self.commands.as_mut().ok_or(Error::State)?;
        let ticket = service::with_nic(io, requested, |io| {
            require_direct(io, rfh::RX_MASK)?;
            calibration::require_rf(io)?;
            submit(queue, io, self.source, &self.domain).map_err(Error::from)
        })?;
        Ok(PendingCommand {
            id: id.wire(),
            ticket,
            cutoff,
            deadline,
        })
    }
}

#[cfg(test)]
impl<Q: BootIo> RegularAdapter<'_, '_, Q> {
    pub(super) fn submit_dqa_for_test(
        &mut self,
        request: &radio::control::Control,
    ) -> Result<PendingCommand, Error> {
        if !self.soc_configuration_ready() {
            return Err(Error::State);
        }
        // Use the established Gen1 owner for an isolated transfer. This
        // test has no platform regulatory observation or DQA-ready state.
        self.adapter.state = State::Faulted;
        self.adapter
            .submit_regular_command(request.id(), |queue, io, _, _| {
                queue.submit_dqa_for_test(io, request)
            })
    }

    pub(super) fn poll_dqa_for_test(&mut self, pending: PendingCommand) -> Result<bool, Error> {
        let reply = self.adapter.poll_regular_command(pending)?;
        if reply {
            self.adapter.retire_command(pending)?;
        }
        Ok(reply)
    }
}
