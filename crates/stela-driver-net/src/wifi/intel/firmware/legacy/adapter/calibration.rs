// SPDX-License-Identifier: GPL-2.0-or-later
use super::{
    Adapter, COMMAND_TIMEOUT_US, Error, PendingCommand, RegularHandoff, State, alive, phy, phy_db,
    record_reply, require_direct, rfh, rx, service,
};
use crate::wifi::intel::{boot::BootIo, firmware::legacy::Sku, power::Csr};

const RFKILL_ALLOWED: u32 = 1 << 27;
const PHY_DB: u8 = 0x6b;
const INIT_COMPLETE: u8 = 0x04;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CalibrationProgress {
    Pending,
    Complete,
}

pub(super) enum Attempt {
    Antenna {
        pending: PendingCommand,
        config: phy::Request,
    },
    Phy {
        pending: Option<PendingCommand>,
        cutoff: rx::ResponseCutoff,
        database: phy_db::Database,
        init_complete: bool,
    },
}

impl<'a, Q: BootIo> Adapter<'a, Q> {
    /// Submit the 31dc INIT antenna command before PHY calibration.
    ///
    /// # Safety
    /// The caller must have verified the actual 8086:31dc shared-clock
    /// board, its full NVM and the antenna configuration for this requester.
    /// Calibration may operate the radio. This method retains the INIT
    /// requester and does not start REGULAR firmware.
    pub unsafe fn start_31dc_init_calibration(&mut self) -> Result<(), Error> {
        if !matches!(self.state, State::NvmReady) || self.sku != Sku::Cnvi9000 {
            return Err(Error::State);
        }
        let config = self.phy_config.ok_or(Error::State)??;
        require_rf(self.receiver.control_io()?.0)?;
        let identity = self.nvm_identity()?;
        let commands = config.commands(identity.tx_antennas, identity.rx_antennas, true)?;

        self.state = State::Faulted;
        let pending = self.submit_calibration(commands.antenna(), false)?;
        self.init_phy = Some(commands);
        self.state = State::Calibrating(Attempt::Antenna {
            pending,
            config: commands.phy(),
        });
        Ok(())
    }

    /// Process the current RFH generation in wire order. The solicited PHY
    /// reply retires command DMA. INIT_COMPLETE ends notification capture.
    pub fn poll_init_calibration(&mut self) -> Result<CalibrationProgress, Error> {
        let attempt = match core::mem::replace(&mut self.state, State::Faulted) {
            State::Calibrating(attempt) => attempt,
            state => {
                self.state = state;
                return Err(Error::State);
            }
        };
        require_rf(self.receiver.control_io()?.0)?;
        let buffer = self.receiver.poll()?;
        match attempt {
            Attempt::Antenna { pending, config } => {
                if self.poll_antenna(pending, buffer.as_ref())? {
                    let pending = self.submit_calibration(config, true)?;
                    self.state = State::Calibrating(Attempt::Phy {
                        pending: Some(pending),
                        cutoff: pending.cutoff,
                        database: phy_db::Database::default(),
                        init_complete: false,
                    });
                } else {
                    self.state = State::Calibrating(Attempt::Antenna { pending, config });
                }
            }
            Attempt::Phy {
                pending,
                cutoff,
                mut database,
                mut init_complete,
            } => {
                let pending = self.poll_phy(
                    pending,
                    cutoff,
                    buffer.as_ref(),
                    &mut database,
                    &mut init_complete,
                )?;
                if pending.is_none() && init_complete {
                    self.init_database = Some(database);
                    self.state = State::Calibrated;
                    return Ok(CalibrationProgress::Complete);
                }
                self.state = State::Calibrating(Attempt::Phy {
                    pending,
                    cutoff,
                    database,
                    init_complete,
                });
            }
        }
        Ok(CalibrationProgress::Pending)
    }

    /// Take the REGULAR handoff only after INIT completion and acknowledged
    /// stop, DMA drain and unmapping. The returned owner has no active queues.
    pub fn take_regular_handoff(&mut self) -> Result<RegularHandoff<'a, Q>, Error> {
        if !matches!(self.state, State::Stopped)
            || self.init_sequence.is_none()
            || self.init_database.is_none()
            || self.init_phy.is_none()
            || self.init_firmware.is_none()
        {
            return Err(Error::State);
        }
        let io = self.receiver.finish()?;
        let sequence = self.init_sequence.take().ok_or(Error::State)?;
        let database = self.init_database.take().ok_or(Error::State)?;
        Ok(RegularHandoff {
            io,
            sequence,
            database,
            init_phy: self.init_phy.take().ok_or(Error::State)?,
            firmware: self.init_firmware.take().ok_or(Error::State)?,
            sku: self.sku,
            source: self.source,
            domain: self.domain.clone(),
            clock: self.clock,
        })
    }

    fn submit_calibration(
        &mut self,
        request: phy::Request,
        arm_init_complete: bool,
    ) -> Result<PendingCommand, Error> {
        let deadline = self
            .clock
            .try_deadline(COMMAND_TIMEOUT_US)
            .map_err(alive::Error::Clock)?;
        require_rf(self.receiver.control_io()?.0)?;
        let cutoff = if arm_init_complete {
            let sequence = self.init_sequence.as_mut().ok_or(Error::State)?;
            self.receiver.arm_init_complete(sequence)?
        } else {
            self.receiver.response_cutoff()?
        };
        let commands = self.commands.as_mut().ok_or(Error::State)?;
        let (io, requested) = self.receiver.control_io()?;
        let ticket = service::with_nic(io, requested, |io| {
            require_direct(io, rfh::RX_MASK)?;
            require_rf(io)?;
            commands.submit_phy(io, request).map_err(Error::from)
        })?;
        Ok(PendingCommand {
            id: request.id().wire(),
            ticket,
            cutoff,
            deadline,
        })
    }

    fn poll_antenna(
        &mut self,
        pending: PendingCommand,
        buffer: Option<&rx::CompletedBuffer>,
    ) -> Result<bool, Error> {
        pending.check_deadline()?;
        let reply = pending.reply(buffer)?;
        if reply {
            pending.check_deadline()?;
            self.retire_command(pending)?;
        }
        Ok(reply)
    }

    fn poll_phy(
        &mut self,
        pending: Option<PendingCommand>,
        cutoff: rx::ResponseCutoff,
        buffer: Option<&rx::CompletedBuffer>,
        database: &mut phy_db::Database,
        init_complete: &mut bool,
    ) -> Result<Option<PendingCommand>, Error> {
        if let Some(command) = pending {
            command.check_deadline()?;
        }
        let sequence = self.init_sequence.as_mut().ok_or(Error::State)?;
        if !*init_complete {
            sequence.poll(None)?;
        }
        let Some(buffer) = buffer.filter(|buffer| cutoff.accepts(buffer)) else {
            return Ok(pending);
        };
        let mut reply = false;
        for packet in buffer.checked_packets().map_err(rx::Error::from)? {
            let packet = packet.map_err(rx::Error::from)?;
            record_reply(
                pending.is_some_and(|command| command.matches(&packet)),
                &mut reply,
            )?;
            if *init_complete || !packet.unsolicited() || !matches!(packet.group, 0 | 1) {
                continue;
            }
            match packet.command {
                PHY_DB => database.record(packet.payload())?,
                INIT_COMPLETE => {
                    *init_complete = sequence.poll(Some(buffer.completion(&packet)))?
                        == alive::Progress::InitComplete;
                }
                _ => {}
            }
        }
        if let Some(command) = pending.filter(|_| reply) {
            command.check_deadline()?;
            self.retire_command(command)?;
            return Ok(None);
        }
        Ok(pending)
    }
}

pub(super) fn require_rf(io: &mut impl Csr) -> Result<(), Error> {
    if service::read(io, service::GP_CONTROL)? & RFKILL_ALLOWED == 0 {
        return Err(Error::RfKilled);
    }
    Ok(())
}
