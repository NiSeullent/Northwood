// SPDX-License-Identifier: GPL-2.0-only
//! The one outstanding command, started and finished without waiting, as
//! Linux drivers/usb/host/xhci-ring.c queues a command and its timer
//! (`xhci_handle_command_timeout`) runs apart from the issuer.
//! Copyright (C) 2008 Intel Corp.
use super::*;

/// The command the ring holds, with its timer.
pub(super) struct Outstanding {
    address: u64,
    control: u32,
    phase: Phase,
    deadline: Deadline,
    /// Aborted into a No Op, whose completion reports a timeout.
    retired: bool,
    /// Times the ring was restarted for this command after it stopped
    /// without completing it.
    restarts: u8,
}

/// Restarts of a stopped command ring one command may take before the
/// controller counts as dead. Linux re-arms `xhci_handle_stopped_cmd_ring`
/// without end while its issuer sleeps; here the issuer is the boot, so a
/// controller that does not run its ring after a restart is given up.
const RING_RESTARTS: u8 = 1;

#[derive(Clone, Copy)]
enum Phase {
    Running,
    /// `xhci_abort_cmd_ring`: CRR must clear.
    Aborting,
    /// Command Ring Stopped must follow.
    Stopping,
}

impl Controller {
    /// Queue a command and ring the command doorbell; [`Self::finish_command`]
    /// gives its completion. One command is outstanding at a time.
    pub(super) fn start_command(
        &mut self,
        parameter: u64,
        status: u32,
        control: u32,
    ) -> Result<u64, BlockError> {
        if self.outstanding.is_some() {
            return Err(BlockError::Protocol("an xHCI command is outstanding"));
        }
        self.ready()?;
        self.sync_command_buffers((control >> 24) as u8)?;
        let deadline = self.event_deadline(COMMAND_US)?;
        let address = self.commands.push(&[(parameter, status, control)])?[0];
        // Linux xhci_ring_cmd_db rings only a running command ring.
        if !self.command_ring_stopped {
            self.doorbell(0, 0);
        }
        self.outstanding = Some(Outstanding {
            address,
            control,
            phase: Phase::Running,
            deadline,
            retired: false,
            restarts: 0,
        });
        Ok(address)
    }

    /// The started command's result once it completed or timed out; `None`
    /// while it runs. Never waits.
    pub(super) fn finish_command(&mut self, address: u64) -> Option<Result<Event, BlockError>> {
        if let Err(error) = self.check_minimum() {
            return Some(Err(error));
        }
        if let Some(at) = self.finished.iter().position(|(a, _)| *a == address) {
            return Some(self.finished.remove(at).1);
        }
        if self
            .outstanding
            .as_ref()
            .is_none_or(|c| c.address != address)
        {
            return Some(Err(BlockError::Protocol("unknown xHCI command")));
        }
        let result = self.advance_command()?;
        if let Some(error) = self.minimum.error() {
            return Some(Err(BlockError::PciPower(error)));
        }
        let retired = self.outstanding.take().is_some_and(|c| c.retired);
        Some(self.conclude_command(result, retired))
    }

    /// Run `control` to completion, finishing any command another owner
    /// started first.
    pub(super) fn command(
        &mut self,
        parameter: u64,
        status: u32,
        control: u32,
    ) -> Result<Event, BlockError> {
        self.check_minimum()?;
        self.settle_command();
        let address = self.start_command(parameter, status, control)?;
        loop {
            if let Some(result) = self.finish_command(address) {
                return result;
            }
            crate::waiting();
            core::hint::spin_loop();
        }
    }

    /// One look at an operation's command: started once no other command
    /// is outstanding, then its result. `None` while either waits.
    pub(super) fn step_command(
        &mut self,
        started: &mut Option<u64>,
        parameter: u64,
        status: u32,
        control: u32,
    ) -> Option<Result<Event, BlockError>> {
        if let Err(error) = self.check_minimum() {
            return Some(Err(error));
        }
        let address = match *started {
            Some(address) => address,
            None => {
                if let Some(other) = self.outstanding.as_ref().map(|c| c.address) {
                    let result = self.finish_command(other)?;
                    if let Some(error) = self.minimum.error() {
                        return Some(Err(BlockError::PciPower(error)));
                    }
                    self.finished.push((other, result));
                }
                match self.start_command(parameter, status, control) {
                    Ok(address) => *started.insert(address),
                    Err(error) => return Some(Err(error)),
                }
            }
        };
        let result = self.finish_command(address)?;
        *started = None;
        Some(result)
    }

    /// Run the outstanding command to its end, keeping the result for its
    /// starter.
    pub(super) fn settle_command(&mut self) {
        if self.check_minimum().is_err() {
            return;
        }
        while let Some(other) = self.outstanding.as_ref().map(|c| c.address) {
            match self.finish_command(other) {
                Some(result) => {
                    if self.minimum.error().is_some() {
                        return;
                    }
                    self.finished.push((other, result));
                }
                None => {
                    crate::waiting();
                    core::hint::spin_loop();
                }
            }
        }
    }

    /// Wait-free progress of the outstanding command: its completion, or an
    /// error ending it, once either is known.
    fn advance_command(&mut self) -> Option<Result<Event, BlockError>> {
        loop {
            if let Err(error) = self.check_minimum() {
                return Some(Err(error));
            }
            self.drain();
            if let Some(memory) = &self.native_memory
                && let Err(error) = memory.ready()
            {
                return Some(Err(error));
            }
            if let Err(error) = self.hardware.ready() {
                return Some(Err(error));
            }
            let current = self.outstanding.as_ref()?;
            let (address, control, phase) = (current.address, current.control, current.phase);
            let expired = current.deadline.passed();
            // Quiet drains read no USBSTS: a controller that died without
            // an event is reported as such at a deadline.
            if expired && self.error.is_none() {
                self.check_status();
            }
            match phase {
                Phase::Running => {
                    if let Some(event) =
                        self.take(|e| e.kind() == COMMAND_COMPLETION && e.parameter == address)
                    {
                        return Some(Ok(event));
                    }
                    if let Some(status) = self.error {
                        return Some(Err(BlockError::Device { status }));
                    }
                    if !expired {
                        return None;
                    }
                    if let Err(error) = self.command_timeout(control) {
                        return Some(Err(error));
                    }
                }
                Phase::Aborting => {
                    if self.registers.read32(self.operational + CRCR) & CRCR_RUNNING == 0 {
                        if let Err(error) = self.set_command_phase(Phase::Stopping, RING_STOP_US) {
                            return Some(Err(error));
                        }
                        continue;
                    }
                    if let Some(status) = self.error {
                        return Some(Err(BlockError::Device { status }));
                    }
                    if expired {
                        self.died();
                        return Some(Err(BlockError::Timeout));
                    }
                    return None;
                }
                Phase::Stopping => {
                    if self
                        .take(|e| {
                            e.kind() == COMMAND_COMPLETION && e.code() == COMMAND_RING_STOPPED
                        })
                        .is_some()
                    {
                        // Its completion precedes Command Ring Stopped unless
                        // the controller had not fetched it; a completion that
                        // raced the abort stands.
                        match self
                            .take(|e| e.kind() == COMMAND_COMPLETION && e.parameter == address)
                        {
                            Some(event) if event.code() == COMMAND_ABORTED => {
                                return Some(Err(BlockError::Timeout));
                            }
                            Some(event) => return Some(Ok(event)),
                            None => {
                                // xhci_handle_stopped_cmd_ring: it becomes a No
                                // Op, which the restarted ring retires.
                                if let Err(error) = self.restart_ring() {
                                    return Some(Err(error));
                                }
                                if let Err(error) = self.commands.no_op(address) {
                                    return Some(Err(error));
                                }
                                self.doorbell(0, 0);
                                if let Err(error) =
                                    self.set_command_phase(Phase::Running, COMMAND_US)
                                {
                                    return Some(Err(error));
                                }
                                self.outstanding.as_mut()?.retired = true;
                                continue;
                            }
                        }
                    }
                    if let Some(status) = self.error {
                        return Some(Err(BlockError::Device { status }));
                    }
                    if expired {
                        // xhci_cleanup_command_queue: the command is reported
                        // aborted.
                        self.command_ring_stopped = true;
                        return Some(Err(BlockError::Timeout));
                    }
                    return None;
                }
            }
        }
    }

    fn set_command_phase(&mut self, phase: Phase, microseconds: u64) -> Result<(), BlockError> {
        let deadline = self.event_deadline(microseconds)?;
        if let Some(current) = self.outstanding.as_mut() {
            current.phase = phase;
            current.deadline = deadline;
        }
        Ok(())
    }

    /// Count one restart of a stopped ring for the outstanding command; one
    /// past [`RING_RESTARTS`] means the controller no longer runs commands.
    fn restart_ring(&mut self) -> Result<(), BlockError> {
        let Some(current) = self.outstanding.as_mut() else {
            return Ok(());
        };
        if current.restarts >= RING_RESTARTS {
            self.died();
            return Err(BlockError::Timeout);
        }
        current.restarts += 1;
        Ok(())
    }

    /// Linux xhci_handle_command_timeout for the one outstanding command.
    fn command_timeout(&mut self, control: u32) -> Result<(), BlockError> {
        // A Stop Endpoint the controller never answers means the host died.
        if control >> 10 & 0x3f == STOP_ENDPOINT {
            self.died();
            return Err(BlockError::Timeout);
        }
        let at = self.operational + CRCR;
        let low = self.registers.read32(at);
        if u64::from(low) | u64::from(self.registers.read32(at + 4)) << 32 == u64::MAX {
            self.died();
            return Err(BlockError::Timeout);
        }
        if self.command_ring_stopped || low & CRCR_RUNNING == 0 {
            // xhci_handle_stopped_cmd_ring: restart the ring and wait again.
            self.restart_ring()?;
            self.command_ring_stopped = false;
            self.doorbell(0, 0);
            return self.set_command_phase(Phase::Running, COMMAND_US);
        }
        // xhci_abort_cmd_ring. The upper half names the next command, so a
        // ring that stops before it is written keeps a valid pointer.
        self.registers
            .write64(at, self.commands.address() | u64::from(CRCR_ABORT));
        self.set_command_phase(Phase::Aborting, ABORT_US)
    }

    /// A command's outcome. A fatal controller state or an unresolved
    /// completion forbids further commands until the controller is reset.
    fn conclude_command(
        &mut self,
        result: Result<Event, BlockError>,
        retired: bool,
    ) -> Result<Event, BlockError> {
        let event = match result {
            Ok(event) => event,
            // A ring abort that got no Command Ring Stopped is not poison;
            // the next command restarts the ring.
            Err(BlockError::Timeout) => return Err(BlockError::Timeout),
            Err(error) => {
                self.command_poisoned = true;
                return Err(error);
            }
        };
        if let Some(status) = self.error {
            self.command_poisoned = true;
            return Err(BlockError::Device { status });
        }
        if retired {
            return Err(BlockError::Timeout);
        }
        match event.code() {
            SUCCESS => Ok(event),
            status => Err(BlockError::Device { status }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bus::dma::TestRegion;
    use crate::usb::xhci::model::*;

    #[test]
    fn commands_finish_without_waiting_and_a_sync_command_settles_another() {
        let mut dma = TestRegion::new(64 * 1024 * 8);
        let (mut controller, model) = cancellation_controller(&mut dma);
        model.borrow_mut().silence = Some(EVALUATE_CONTEXT);
        let address = controller
            .start_command(0, 0, EVALUATE_CONTEXT << 10 | 1 << 24)
            .unwrap();
        assert!(controller.finish_command(address).is_none());
        assert_eq!(
            controller.start_command(0, 0, NO_OP_COMMAND << 10),
            Err(BlockError::Protocol("an xHCI command is outstanding"))
        );
        {
            let mut model = model.borrow_mut();
            let at = model.stuck.take().unwrap();
            model.event(at, SUCCESS, COMMAND_COMPLETION, 0, 0);
        }
        assert!(controller.finish_command(address).unwrap().is_ok());
        let first = controller.start_command(0, 0, NO_OP_COMMAND << 10).unwrap();
        controller.command(0, 0, NO_OP_COMMAND << 10).unwrap();
        assert!(controller.outstanding.is_none());
        assert!(controller.finish_command(first).unwrap().is_ok());
        assert!(controller.finished.is_empty());
    }
}
