// SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause
// Copyright (C) 2003-2015, 2018-2026 Intel Corporation
// Copyright (C) 2013-2015 Intel Mobile Communications GmbH
// Copyright (C) 2016-2017 Intel Deutschland GmbH
//! Polled Intel 9000 RFH register owner. Register layout and stop acknowledgments
//! follow Linux iwlwifi at cc9de5d4b5ab6459f5774946aa3195ad51236fa8.

use super::{
    Firmware, Sku,
    alive::Sequence,
    rx::{self, CompletedBuffer, Layout, Queue, ResponseCutoff},
    service::{
        self, FH_INT, ICT, ICT_ENABLED, INT, INT_FH_TX, INT_MASK, causes, read, read_prph,
        with_nic, write_prph,
    },
};
use crate::{
    bus::dma::mapping::DmaError,
    wifi::intel::{
        boot::BootIo,
        interrupts::{INT_HW_ERR, INT_SW_ERR},
        power::{AccessError, Csr},
    },
};
use stela_driver_base::Clock;

const COALESCING: u32 = 0x04;
const RESET: u32 = 0x20;
const SHADOW: u32 = 0xa8;
const SHADOW_ENABLE: u32 = 0x800f_ffff;
const STOP_MASTER: u32 = 0x200;
const MASTER_DISABLED: u32 = 0x100;
const SW_RESET: u32 = 0x80;
const DOORBELL: u32 = 0x1c80;
const RX_CAUSES: u32 = (1 << 31) | (1 << 3) | 1;
const PERIODIC_RX: u32 = 1 << 28;
const FH_RX_CAUSES: u32 = (1 << 30) | (3 << 16);
pub(super) const RX_MASK: u32 = RX_CAUSES | INT_HW_ERR | INT_SW_ERR;
pub(super) const FH_RX_MASK: u32 = INT_FH_TX | RX_MASK;

const FREE_BASE: u32 = 0xa0_8000;
const FREE_WRITE: u32 = 0xa0_8080;
const FREE_READ: u32 = 0xa0_80c0;
const USED_BASE: u32 = 0xa0_8100;
const USED_WRITE: u32 = 0xa0_8180;
const STATUS_BASE: u32 = 0xa0_8200;
const RFH_GENERAL: u32 = 0xa0_9800;
const RFH_STATUS: u32 = 0xa0_9808;
const RFH_ACTIVE: u32 = 0xa0_980c;
const RFH_DMA: u32 = 0xa0_9820;
const RX_IDLE: u32 = 1 << 31;
const QUEUE_ZERO: u32 = 1 | (1 << 16);
// Enable, drop oversized frames, minimum RB 4/8, 512 entries, 4 KiB RBs.
const DMA_CONFIG: u32 = (1 << 31) | (1 << 26) | (3 << 24) | (9 << 20) | (4 << 16);
const GENERAL_MASK: u32 = 0xf13;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    Registers(service::Error),
    Queue(rx::Error),
    Mapping(DmaError),
    State,
    InterruptMode,
    Rejected {
        register: u32,
        expected: u32,
        actual: u32,
    },
    StopTimeout {
        register: u32,
    },
}

impl From<service::Error> for Error {
    fn from(error: service::Error) -> Self {
        Self::Registers(error)
    }
}

impl From<AccessError> for Error {
    fn from(error: AccessError) -> Self {
        Self::Registers(error.into())
    }
}

impl From<rx::Error> for Error {
    fn from(error: rx::Error) -> Self {
        Self::Queue(error)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    RestartRequired,
    Prepared,
    Restarted,
    Configured,
    ReceivingWithFh,
    Receiving,
    Faulted,
    Quiescent,
    Stopped,
}

/// One exclusive, directly polled requester and its bootstrap receive queue.
pub struct Receiver<Q: BootIo> {
    io: Option<Q>,
    queue: Queue,
    sku: Sku,
    state: State,
    nic_requested: bool,
}

impl<Q: BootIo> Receiver<Q> {
    /// # Safety
    /// `io` is the prepared, awake Intel 9000 function identified by `sku`.
    /// `queue` is fresh, with mappings in this exact requester's coherent DMA
    /// domain. PCI No Snoop, relaxed ordering, ATS, and PCI interrupt delivery
    /// are disabled. No outside owner, IRQ handler or worker may use the BAR,
    /// reuse this generation, post work or restart this requester. The
    /// owner serializes firmware uploads, RFH polling and teardown.
    pub unsafe fn new(io: Q, queue: Queue, sku: Sku) -> Self {
        Self::with_state(io, queue, sku, State::Prepared)
    }

    /// # Safety
    /// The same requester has stopped and drained. `queue` is fresh and
    /// mapped in this exclusive requester's coherent DMA domain.
    pub(super) unsafe fn after_init_stop(io: Q, queue: Queue, sku: Sku) -> Self {
        Self::with_state(io, queue, sku, State::RestartRequired)
    }

    fn with_state(io: Q, queue: Queue, sku: Sku, state: State) -> Self {
        Self {
            io: Some(io),
            queue,
            sku,
            state,
            nic_requested: false,
        }
    }

    /// Configure queue zero and post the bootstrap credits. An existing
    /// direct FH-TX-only mask stays in place until the separate RX handoff.
    pub fn start(&mut self) -> Result<(), Error> {
        self.start_with_preparation(|_| Ok(()))
    }

    /// Restart the requester before programming a fresh receive queue.
    /// An uncertain restart retains the owner and requires hardware stop.
    pub(super) fn restart_prepared<E: From<Error>>(
        &mut self,
        restart: impl FnOnce(&mut Q) -> Result<(), E>,
    ) -> Result<(), E> {
        if self.state != State::RestartRequired {
            return Err(Error::State.into());
        }
        let io = self.io.as_mut().ok_or(Error::State)?;
        self.state = State::Faulted;
        restart(io)?;
        self.state = State::Restarted;
        Ok(())
    }

    pub(super) fn start_with_preparation(
        &mut self,
        prepare: impl FnOnce(&mut Q) -> Result<(), Error>,
    ) -> Result<(), Error> {
        if !matches!(self.state, State::Prepared | State::Restarted) {
            return Err(Error::State);
        }
        if self.state == State::Restarted {
            self.state = State::Faulted;
        }
        let io = self.io.as_mut().ok_or(Error::State)?;
        if read(io, ICT)? & ICT_ENABLED != 0 || !matches!(read(io, INT_MASK)?, 0 | INT_FH_TX) {
            return Err(Error::InterruptMode);
        }
        causes(io)?;
        self.state = State::Faulted;
        prepare(io)?;
        let queue = &mut self.queue;
        let general = match self.sku {
            Sku::Cnvi9000 => 3,
            Sku::WirelessAc9260 => 3 | (1 << 4),
        };
        let credits = with_nic(io, &mut self.nic_requested, |io| {
            halt_rfh(io)?;
            // SAFETY: constructor binds the fresh coherent queue to this
            // exclusive requester. RFH is stopped and no bases have escaped.
            let layout = unsafe { queue.expose()? };
            program_rfh(io, layout, general)?;
            Ok::<_, Error>(layout.write)
        })?;
        let shadow = read(io, SHADOW)? | SHADOW_ENABLE;
        io.write32(SHADOW, shadow)?;
        expect(SHADOW, SHADOW_ENABLE, read(io, SHADOW)? & SHADOW_ENABLE)?;
        io.write8(COALESCING, 0x40)?;
        io.write32(DOORBELL, u32::from(credits))?;
        // Flush the posted doorbell while the requester is still owned.
        read(io, INT_MASK)?;
        self.state = State::Configured;
        Ok(())
    }

    /// Switch from the FH-only window after the serial uploader has drained.
    /// Causes are acknowledged through CSR, with PCI delivery still disabled.
    pub fn enable_interrupts(&mut self) -> Result<(), Error> {
        self.enable_receive_mask(RX_MASK, State::Receiving)
    }

    /// Open RFH reception at CPU1 handoff while CPU2 still needs FH-TX.
    pub(super) fn enable_interrupts_for_fh(&mut self) -> Result<(), Error> {
        self.enable_receive_mask(FH_RX_MASK, State::ReceivingWithFh)
    }

    fn enable_receive_mask(&mut self, mask: u32, next: State) -> Result<(), Error> {
        if self.state != State::Configured {
            return Err(Error::State);
        }
        self.state = State::Faulted;
        let io = self.io.as_mut().ok_or(Error::State)?;
        direct(io)?;
        if next == State::ReceivingWithFh {
            expect(INT_MASK, INT_FH_TX, read(io, INT_MASK)?)?;
        }
        acknowledge_rx(io)?;
        write_mask(io, mask)?;
        self.state = next;
        Ok(())
    }

    /// End the shared FH/RX mask after CPU2 DMA completion. Leave pending
    /// causes for the RFH poller, including an already queued ALIVE.
    pub(super) fn finish_fh_upload(&mut self) -> Result<(), Error> {
        if self.state != State::ReceivingWithFh {
            return Err(Error::State);
        }
        self.state = State::Faulted;
        let io = self.io.as_mut().ok_or(Error::State)?;
        direct(io)?;
        expect(INT_MASK, FH_RX_MASK, read(io, INT_MASK)?)?;
        write_mask(io, RX_MASK)?;
        self.state = State::Receiving;
        Ok(())
    }

    /// A cause is only a hint. The coherent producer and its posted VID
    /// decide whether a buffer can transfer to the CPU, even with no cause.
    pub fn poll(&mut self) -> Result<Option<CompletedBuffer>, Error> {
        let io = self.receiving_io()?;
        let result = direct(io)
            .and_then(|()| expect(INT_MASK, RX_MASK, read(io, INT_MASK)?))
            .and_then(|()| acknowledge_rx(io))
            .and_then(|()| Ok(self.queue.complete()?));
        if result.is_err() {
            self.state = State::Faulted;
        }
        result
    }

    pub fn wait_init(&mut self, firmware: &Firmware<'_>, clock: Clock) -> Result<Sequence, Error> {
        self.receiving_io()?;
        Ok(self.queue.wait_init(firmware, clock)?)
    }

    pub fn wait_init_complete(&mut self, sequence: &mut Sequence) -> Result<(), Error> {
        self.arm_init_complete(sequence).map(|_| ())
    }

    pub(super) fn arm_init_complete(
        &mut self,
        sequence: &mut Sequence,
    ) -> Result<ResponseCutoff, Error> {
        self.receiving_io()?;
        Ok(self.queue.arm_init_complete(sequence)?)
    }

    pub(super) fn response_cutoff(&mut self) -> Result<ResponseCutoff, Error> {
        self.receiving_io()?;
        Ok(self.queue.response_cutoff()?)
    }

    pub fn wait_regular_after_init_stop(&mut self, sequence: &mut Sequence) -> Result<(), Error> {
        self.receiving_io()?;
        Ok(self.queue.wait_regular_after_init_stop(sequence)?)
    }

    fn receiving_io(&mut self) -> Result<&mut Q, Error> {
        if self.state != State::Receiving {
            return Err(Error::State);
        }
        self.io.as_mut().ok_or(Error::State)
    }

    /// Lend CSR and NIC-access custody to the serial firmware uploader.
    pub(super) fn control_io(&mut self) -> Result<(&mut Q, &mut bool), Error> {
        if !matches!(
            self.state,
            State::Configured | State::ReceivingWithFh | State::Receiving
        ) {
            return Err(Error::State);
        }
        let io = self.io.as_mut().ok_or(Error::State)?;
        Ok((io, &mut self.nic_requested))
    }

    /// Mask delivery, stop RFH, acknowledge requester-wide master disable
    /// and reset, then retire the mappings. Retry failed invalidations with
    /// the already quiescent requester still owned.
    pub fn stop(&mut self) -> Result<(), Error> {
        self.quiesce()?;
        // SAFETY: either no address was exposed or RFH idle and requester
        // master-disable/reset completed. Constructor excluded other work.
        unsafe { self.queue.release_after_stop() }.map_err(Error::Mapping)?;
        self.state = State::Stopped;
        Ok(())
    }

    /// Prove hardware stop while retaining every receive mapping. A caller
    /// sharing the requester can drain its root before retiring any mapping.
    pub(super) fn quiesce(&mut self) -> Result<(), Error> {
        self.quiesce_with(|_| Ok(()))
    }

    pub(super) fn quiesce_with<E>(
        &mut self,
        halt_transmit: impl FnOnce(&mut Q) -> Result<(), E>,
    ) -> Result<(), E>
    where
        E: From<Error> + From<service::Error>,
    {
        match self.state {
            State::Stopped => return Err(Error::State.into()),
            State::RestartRequired | State::Prepared | State::Quiescent => {}
            _ => {
                self.state = State::Faulted;
                let io = self.io.as_mut().ok_or(Error::State)?;
                stop_engines(io, &mut self.nic_requested, halt_transmit)?;
            }
        }
        self.state = State::Quiescent;
        Ok(())
    }

    pub fn finish(&mut self) -> Result<Q, Error> {
        if self.state != State::Stopped {
            return Err(Error::State);
        }
        self.io.take().ok_or(Error::State)
    }
}

pub(super) fn stop_engines<Q: Csr, E: From<Error> + From<service::Error>>(
    io: &mut Q,
    nic_requested: &mut bool,
    halt_transmit: impl FnOnce(&mut Q) -> Result<(), E>,
) -> Result<(), E> {
    write_mask(io, 0)?;
    with_nic(io, nic_requested, |io| {
        halt_transmit(io)?;
        halt_rfh(io).map_err(E::from)
    })?;
    stop_master(io)?;
    Ok(())
}

impl<Q: BootIo> Drop for Receiver<Q> {
    fn drop(&mut self) {
        if !matches!(
            self.state,
            State::RestartRequired | State::Prepared | State::Stopped
        ) && let Some(io) = self.io.take()
        {
            // Exposed Chunk leases quarantine their pins. Keep the same PCI
            // claim and BAR with them on uncertain programming or teardown.
            core::mem::forget(io);
        }
    }
}

fn expect(register: u32, expected: u32, actual: u32) -> Result<(), Error> {
    if actual == expected {
        Ok(())
    } else {
        Err(Error::Rejected {
            register,
            expected,
            actual,
        })
    }
}

fn halt_rfh(io: &mut impl Csr) -> Result<(), Error> {
    write_prph(io, RFH_DMA, 0)?;
    write_prph(io, RFH_ACTIVE, 0)?;
    expect(RFH_DMA, 0, read_prph(io, RFH_DMA)?)?;
    expect(RFH_ACTIVE, 0, read_prph(io, RFH_ACTIVE)?)?;
    poll_bit(io, RFH_STATUS, RX_IDLE, 100, |io, register| {
        Ok(read_prph(io, register)?)
    })
}

fn program_rfh(io: &mut impl Csr, layout: Layout, general: u32) -> Result<(), Error> {
    for (register, address) in [
        (FREE_BASE, layout.free),
        (USED_BASE, layout.used),
        (STATUS_BASE, layout.status),
    ] {
        write_prph(io, register, address as u32)?;
        write_prph(io, register + 4, (address >> 32) as u32)?;
    }
    for register in [FREE_WRITE, FREE_READ, USED_WRITE] {
        write_prph(io, register, 0)?;
    }
    for (register, address) in [
        (FREE_BASE, layout.free),
        (USED_BASE, layout.used),
        (STATUS_BASE, layout.status),
    ] {
        expect(register, address as u32, read_prph(io, register)?)?;
        expect(
            register + 4,
            (address >> 32) as u32,
            read_prph(io, register + 4)?,
        )?;
    }
    for register in [FREE_WRITE, FREE_READ, USED_WRITE] {
        expect(register, 0, read_prph(io, register)?)?;
    }
    write_prph(io, RFH_DMA, DMA_CONFIG)?;
    write_prph(io, RFH_GENERAL, general)?;
    write_prph(io, RFH_ACTIVE, QUEUE_ZERO)?;
    expect(RFH_DMA, DMA_CONFIG, read_prph(io, RFH_DMA)?)?;
    expect(
        RFH_GENERAL,
        general,
        read_prph(io, RFH_GENERAL)? & GENERAL_MASK,
    )?;
    expect(RFH_ACTIVE, QUEUE_ZERO, read_prph(io, RFH_ACTIVE)?)
}

fn direct(io: &mut impl Csr) -> Result<(), Error> {
    if read(io, ICT)? & ICT_ENABLED != 0 {
        Err(Error::InterruptMode)
    } else {
        Ok(())
    }
}

fn acknowledge_rx(io: &mut impl Csr) -> Result<(), Error> {
    let (csr, fh) = causes(io)?;
    let receive = csr & (RX_CAUSES | PERIODIC_RX);
    if receive != 0 {
        io.write32(INT, receive)?;
    }
    let receive = fh & FH_RX_CAUSES;
    if receive != 0 {
        io.write32(FH_INT, receive)?;
    }
    Ok(())
}

fn write_mask(io: &mut impl Csr, mask: u32) -> Result<(), Error> {
    io.write32(INT_MASK, mask)?;
    expect(INT_MASK, mask, read(io, INT_MASK)?)
}

fn stop_master(io: &mut impl Csr) -> Result<(), Error> {
    let reset = read(io, RESET)?;
    io.write32(RESET, reset | STOP_MASTER)?;
    poll_bit(io, RESET, MASTER_DISABLED, 10, |io, register| {
        Ok(read(io, register)?)
    })?;
    let reset = read(io, RESET)?;
    io.write32(RESET, reset | SW_RESET)?;
    io.delay_us(5000)?;
    read(io, RESET)?;
    write_mask(io, 0)
}

fn poll_bit<T: Csr>(
    io: &mut T,
    register: u32,
    bit: u32,
    intervals: usize,
    mut observe: impl FnMut(&mut T, u32) -> Result<u32, Error>,
) -> Result<(), Error> {
    for _ in 0..intervals {
        if observe(io, register)? & bit != 0 {
            return Ok(());
        }
        io.delay_us(10)?;
    }
    if observe(io, register)? & bit != 0 {
        Ok(())
    } else {
        Err(Error::StopTimeout { register })
    }
}

#[cfg(test)]
mod tests;
