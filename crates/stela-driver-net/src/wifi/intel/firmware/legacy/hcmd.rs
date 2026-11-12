// SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause
// Copyright (C) 2003-2015, 2018-2026 Intel Corporation
// Copyright (C) 2013-2015 Intel Mobile Communications GmbH
// Copyright (C) 2016-2017 Intel Deutschland GmbH
//! Gen1 FH command DMA for bounded, polled Intel 9000 bootstrap commands.
//! Descriptor and queue semantics follow the pinned Linux iwlwifi
//! iwl-fh.h, iwl-scd.h and pcie/gen1_2/tx.c (347246472df8).

use super::{FH_DMA_LIMIT, nvm::Request, paging, phy, phy_db, service};
use crate::{
    bus::dma::{
        mapping::{Direction, DmaDomain},
        memory::PageSource,
    },
    wifi::intel::{
        command::{self, Command, Id},
        context::Region,
        dma::{self, Chunk},
        power::{AccessError, Csr},
        radio,
        transmit::ring::WRITE_POINTER,
    },
};
use alloc::{sync::Arc, vec, vec::Vec};
use core::sync::atomic::{Ordering, fence};

pub(super) mod hardware;

const QUEUES: usize = 31;
const ENTRIES: u16 = 256;
const TFD_BYTES: usize = 128;
const RING_BYTES: usize = ENTRIES as usize * TFD_BYTES;
const TB0_BYTES: usize = 64;
const KEEP_WARM_BYTES: usize = 4096;
// 9000 has 31 queues with 256 byte counts and 64 duplicated entries each.
const BYTE_COUNT_BYTES: usize = QUEUES * (256 + 64) * 2;
const INVALID_COMMAND: [u8; 8] = [6, 0x0f, 0xff, 0xff, 0, 0, 0, 0];

const SCD_TXFACT: u32 = 0xa02c10;
const FH_TX_CONFIG: u32 = 0x1d00;
const FH_TX_STRIDE: u32 = 0x20;
const FH_TX_CHANNELS: u32 = 8;
const FH_TX_STATUS: u32 = 0x1eb0;
const FH_TX_IDLE: u32 = 0x00ff0000;
const IDLE_POLL_US: u32 = 10;
const IDLE_TIMEOUT_US: u32 = 5000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    pub ring: u64,
    pub rings: [u64; QUEUES],
    pub entries: u16,
    pub entry_bytes: usize,
    pub keep_warm: u64,
    pub byte_count: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ticket(u16);

impl Ticket {
    pub fn sequence(self) -> u16 {
        self.0
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    Dma(dma::Error),
    Command(command::Error),
    Service(service::Error),
    Address,
    State,
    Pending,
    Exhausted,
    SchedulerSram { register: u32, alive: u32 },
    Sequence { expected: u16, observed: u16 },
    Rejected { register: u32, observed: u32 },
    StopTimeout(u32),
}

impl From<dma::Error> for Error {
    fn from(error: dma::Error) -> Self {
        Self::Dma(error)
    }
}

impl From<command::Error> for Error {
    fn from(error: command::Error) -> Self {
        Self::Command(error)
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

#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    Private,
    Exposed,
    Ready,
    Faulted,
    Retiring,
    Stopped,
}

struct Pending {
    ticket: Ticket,
    tfd_cpu: bool,
    tb0_cpu: bool,
    remainder: Option<Chunk>,
}

pub(super) struct Queue {
    ring: Chunk,
    data_rings: Vec<Chunk>,
    tb0: Chunk,
    invalid: Chunk,
    keep_warm: Chunk,
    byte_count: Chunk,
    invalid_tfd: [u8; TFD_BYTES],
    write: u16,
    pending: Option<Pending>,
    state: State,
}

impl Queue {
    pub(super) fn prepare(
        source: &dyn PageSource,
        domain: Arc<dyn DmaDomain>,
    ) -> Result<Self, Error> {
        Self::prepare_rings(source, domain, 0)
    }

    pub(super) fn prepare_full(
        source: &dyn PageSource,
        domain: Arc<dyn DmaDomain>,
    ) -> Result<Self, Error> {
        Self::prepare_rings(source, domain, QUEUES - 1)
    }

    fn prepare_rings(
        source: &dyn PageSource,
        domain: Arc<dyn DmaDomain>,
        extra_rings: usize,
    ) -> Result<Self, Error> {
        let allocate = |bytes: &[u8]| {
            Chunk::allocate_bounded(
                source,
                domain.clone(),
                bytes,
                Direction::Bidirectional,
                FH_DMA_LIMIT - 1,
            )
        };
        let invalid = allocate(&INVALID_COMMAND)?;
        let invalid_tfd = command_tfd(invalid.region(), None)?;
        let mut descriptors = vec![0; RING_BYTES];
        for slot in descriptors.chunks_exact_mut(TFD_BYTES) {
            slot.copy_from_slice(&invalid_tfd);
        }
        let ring = allocate(&descriptors)?;
        let tb0 = allocate(&vec![0; usize::from(ENTRIES) * TB0_BYTES])?;
        let keep_warm = allocate(&vec![0; KEEP_WARM_BYTES])?;
        let byte_count = allocate(&vec![0; BYTE_COUNT_BYTES])?;
        let mut data_rings = Vec::with_capacity(extra_rings);
        for _ in 0..extra_rings {
            data_rings.push(allocate(&descriptors)?);
        }
        for (chunk, alignment) in [
            (&ring, 256),
            (&tb0, 64),
            (&keep_warm, 4096),
            (&byte_count, 1024),
        ]
        .into_iter()
        .chain(data_rings.iter().map(|ring| (ring, 256)))
        {
            let address = chunk.region().address;
            if address == 0 || !address.is_multiple_of(alignment) {
                return Err(Error::Address);
            }
        }
        Ok(Self {
            ring,
            data_rings,
            tb0,
            invalid,
            keep_warm,
            byte_count,
            invalid_tfd,
            write: 0,
            pending: None,
            state: State::Private,
        })
    }

    pub(super) fn layout(&self) -> Layout {
        let mut rings = [0; QUEUES];
        rings[0] = self.ring.region().address;
        for (address, ring) in rings[1..].iter_mut().zip(&self.data_rings) {
            *address = ring.region().address;
        }
        Layout {
            ring: rings[0],
            rings,
            entries: ENTRIES,
            entry_bytes: TFD_BYTES,
            keep_warm: self.keep_warm.region().address,
            byte_count: self.byte_count.region().address,
        }
    }

    /// Mark every pin exposed before the first possible FH/SCD address write.
    ///
    /// # Safety
    /// The original requester owns this exact domain, its command producer and
    /// engine are stopped, and every preparation or fault keeps its original
    /// PCI and BAR owner until requester-wide retirement.
    pub(super) unsafe fn expose_for_preparation(&mut self) -> Result<Layout, Error> {
        if self.state != State::Private {
            return Err(Error::State);
        }
        for chunk in [
            &mut self.invalid,
            &mut self.ring,
            &mut self.tb0,
            &mut self.keep_warm,
            &mut self.byte_count,
        ]
        .into_iter()
        .chain(&mut self.data_rings)
        {
            chunk.expose();
        }
        self.state = State::Exposed;
        Ok(self.layout())
    }

    /// # Safety
    /// The same original, exclusive 9000 requester acknowledged a fresh
    /// ALIVE and its real FH/SCD preparation for this exact exposed layout.
    /// Queue zero is the command queue, with both pointers at zero, and can
    /// reach only posted slots. Every later operation uses the same BAR and DMA
    /// epoch with no other submitter, IRQ worker, or scheduler/config writer.
    pub(super) unsafe fn admit_prepared(&mut self) -> Result<(), Error> {
        if self.state != State::Exposed {
            return Err(Error::State);
        }
        self.state = State::Ready;
        Ok(())
    }

    pub(super) fn submit_nvm_read(
        &mut self,
        io: &mut impl Csr,
        request: Request,
    ) -> Result<Ticket, Error> {
        self.submit(io, request.id(), &request.payload())
    }

    pub(super) fn submit_phy(
        &mut self,
        io: &mut impl Csr,
        request: phy::Request,
    ) -> Result<Ticket, Error> {
        self.submit(io, request.id(), request.payload())
    }

    pub(super) fn submit_bluetooth(
        &mut self,
        io: &mut impl Csr,
        request: radio::BluetoothConfig,
    ) -> Result<Ticket, Error> {
        self.submit(io, request.id(), request.payload())
    }

    pub(super) fn submit_soc_configuration(
        &mut self,
        io: &mut impl Csr,
        request: radio::SocConfiguration,
    ) -> Result<Ticket, Error> {
        self.submit(io, request.id(), request.payload())
    }

    #[cfg(test)]
    pub(super) fn submit_dqa_for_test(
        &mut self,
        io: &mut impl Csr,
        request: &radio::control::Control,
    ) -> Result<Ticket, Error> {
        if request.kind() != radio::control::Kind::Dqa {
            return Err(Error::State);
        }
        self.submit(io, request.id(), request.bytes())
    }

    pub(super) fn submit_paging(
        &mut self,
        io: &mut impl Csr,
        source: &dyn PageSource,
        domain: &Arc<dyn DmaDomain>,
        payload: &[u8; paging::PAYLOAD_BYTES],
    ) -> Result<Ticket, Error> {
        self.submit_with_tail(io, source, domain, paging::ID, payload)
    }

    pub(super) fn submit_phy_db(
        &mut self,
        io: &mut impl Csr,
        source: &dyn PageSource,
        domain: &Arc<dyn DmaDomain>,
        request: phy_db::Request<'_>,
    ) -> Result<Ticket, Error> {
        self.submit_with_tail(io, source, domain, phy_db::ID, &request.payload())
    }

    fn submit_with_tail(
        &mut self,
        io: &mut impl Csr,
        source: &dyn PageSource,
        domain: &Arc<dyn DmaDomain>,
        id: Id,
        payload: &[u8],
    ) -> Result<Ticket, Error> {
        let command = self.encode(id, payload)?;
        // Gen1 puts up to 20 bytes in the coherent bidirectional TB0,
        // including the prefix of a NOCOPY section when it fits there.
        let remainder = if command.remainder().is_empty() {
            None
        } else {
            Some(Chunk::allocate_bounded(
                source,
                domain.clone(),
                command.remainder(),
                Direction::ToDevice,
                FH_DMA_LIMIT - 1,
            )?)
        };
        self.submit_encoded(io, command, remainder)
    }

    fn submit(&mut self, io: &mut impl Csr, id: Id, payload: &[u8]) -> Result<Ticket, Error> {
        let command = self.encode(id, payload)?;
        self.submit_encoded(io, command, None)
    }

    fn encode(&self, id: Id, payload: &[u8]) -> Result<Command, Error> {
        self.check_capacity(1)?;
        Ok(Command::encode(id, 0, self.write, payload)?)
    }

    /// Check the upcoming serial commands against the one-shot producer.
    pub(super) fn check_capacity(&self, commands: usize) -> Result<(), Error> {
        if self.state != State::Ready {
            return Err(Error::State);
        }
        if self.pending.is_some() {
            return Err(Error::Pending);
        }
        // Posting slot 255 would wrap the producer to zero.
        if commands > usize::from(ENTRIES - 1 - self.write) {
            return Err(Error::Exhausted);
        }
        Ok(())
    }

    fn submit_encoded(
        &mut self,
        io: &mut impl Csr,
        command: Command,
        mut remainder: Option<Chunk>,
    ) -> Result<Ticket, Error> {
        let remainder_region = remainder.as_ref().map(Chunk::region);
        if remainder_region.map_or(0, |region| region.bytes) != command.remainder().len() {
            return Err(command::Error::Split.into());
        }
        let slot = usize::from(self.write);
        let tfd_offset = slot * TFD_BYTES;
        let tb0_offset = slot * TB0_BYTES;
        let first = command.first_tb();
        let tfd = command_tfd(
            Region {
                address: self.tb0.region().address + tb0_offset as u64,
                bytes: first.len(),
            },
            remainder_region,
        )?;
        let mut tb0 = [0; TB0_BYTES];
        tb0[..first.len()].copy_from_slice(first);
        let ticket = Ticket(command.sequence());
        // A failed synchronization or posted MMIO may have had an effect.
        // Keep the pending command and every already exposed lease first.
        if let Some(remainder) = &mut remainder {
            remainder.expose();
        }
        self.pending = Some(Pending {
            ticket,
            tfd_cpu: false,
            tb0_cpu: false,
            remainder,
        });
        self.state = State::Faulted;
        // SAFETY: the fresh unposted slot belongs to this exclusive producer.
        unsafe {
            self.tb0.write_owned(tb0_offset, &tb0)?;
            self.ring.write_owned(tfd_offset, &tfd)?;
        }
        fence(Ordering::Release);
        let next = self.write + 1;
        io.write32(WRITE_POINTER, u32::from(next))?; // Gen1 queue 0 << 8.
        service::read(io, service::INT_MASK)?;
        self.write = next;
        self.state = State::Ready;
        Ok(ticket)
    }

    /// # Safety
    /// This exact pending command completed in the same RFH generation. The
    /// adapter has validated a matching, reclaimed response and established
    /// the device ended all access to the corresponding descriptor and TBs.
    /// A failed synchronization or invalidation keeps the pending command.
    pub(super) unsafe fn complete(&mut self, sequence: u16) -> Result<(), Error> {
        if self.state != State::Ready {
            return Err(Error::State);
        }
        let pending = self.pending.as_mut().ok_or(Error::State)?;
        let expected = pending.ticket.sequence();
        if sequence != expected {
            return Err(Error::Sequence {
                expected,
                observed: sequence,
            });
        }
        let slot = usize::from(expected);
        let tfd = slot * TFD_BYTES;
        let tb0 = slot * TB0_BYTES;
        if !pending.tfd_cpu {
            // SAFETY: the caller's exact completion covers this descriptor.
            unsafe { self.ring.complete_for_cpu(tfd..tfd + TFD_BYTES)? };
            pending.tfd_cpu = true;
        }
        if !pending.tb0_cpu {
            // SAFETY: the same completion covers TB0 including writeback.
            unsafe { self.tb0.complete_for_cpu(tb0..tb0 + TB0_BYTES)? };
            pending.tb0_cpu = true;
        }
        // SAFETY: the retired slot remains CPU-owned and is never reposted.
        unsafe { self.ring.write_owned(tfd, &self.invalid_tfd)? };
        if let Some(remainder) = &mut pending.remainder {
            // SAFETY: the matched response also ended DMA reads of the tail.
            unsafe { remainder.release().map_err(dma::Error::from)? };
        }
        self.pending = None;
        Ok(())
    }

    /// # Safety
    /// The original requester has acknowledged FH/SCD and RFH stop, no DMA
    /// or submission can restart, and its exact translated payload root and
    /// all bypass/IRQ/work users have drained. Keep its BAR and original owner
    /// until every invalidation succeeds.
    pub(super) unsafe fn release_after_stop(&mut self) -> Result<(), Error> {
        self.state = State::Retiring;
        for chunk in [
            &mut self.ring,
            &mut self.tb0,
            &mut self.invalid,
            &mut self.keep_warm,
            &mut self.byte_count,
        ]
        .into_iter()
        .chain(&mut self.data_rings)
        {
            // SAFETY: the caller's requester-wide stop covers every address.
            unsafe { chunk.release().map_err(dma::Error::from)? };
        }
        if let Some(remainder) = self
            .pending
            .as_mut()
            .and_then(|pending| pending.remainder.as_mut())
        {
            // SAFETY: the same stop covers pending reads of the command tail.
            unsafe { remainder.release().map_err(dma::Error::from)? };
        }
        self.pending = None;
        self.state = State::Stopped;
        Ok(())
    }
}

fn command_tfd(first: Region, remainder: Option<Region>) -> Result<[u8; TFD_BYTES], Error> {
    let mut bytes = [0; TFD_BYTES];
    bytes[3] = 1;
    bytes[4..10].copy_from_slice(&encode_tb(first)?);
    if let Some(remainder) = remainder {
        bytes[3] = 2;
        bytes[10..16].copy_from_slice(&encode_tb(remainder)?);
    }
    Ok(bytes)
}

fn encode_tb(region: Region) -> Result<[u8; 6], Error> {
    if region.address == 0
        || !(1..=4092).contains(&region.bytes)
        || region
            .address
            .checked_add(region.bytes as u64)
            .is_none_or(|end| end > FH_DMA_LIMIT)
    {
        return Err(Error::Address);
    }
    let mut bytes = [0; 6];
    bytes[..4].copy_from_slice(&(region.address as u32).to_le_bytes());
    let hi_n_len = (region.bytes as u16) << 4 | (region.address >> 32) as u16;
    bytes[4..].copy_from_slice(&hi_n_len.to_le_bytes());
    Ok(bytes)
}

/// Disable the scheduler and all FH TX channels while holding NIC access.
pub(super) fn halt(io: &mut impl Csr) -> Result<(), Error> {
    service::write_prph(io, SCD_TXFACT, 0)?;
    expect_zero(SCD_TXFACT, service::read_prph(io, SCD_TXFACT)?)?;
    for channel in 0..FH_TX_CHANNELS {
        io.write32(FH_TX_CONFIG + channel * FH_TX_STRIDE, 0)?;
    }
    for channel in 0..FH_TX_CHANNELS {
        let register = FH_TX_CONFIG + channel * FH_TX_STRIDE;
        expect_zero(register, service::read(io, register)?)?;
    }
    for _ in 0..IDLE_TIMEOUT_US / IDLE_POLL_US {
        if service::read(io, FH_TX_STATUS)? & FH_TX_IDLE == FH_TX_IDLE {
            return Ok(());
        }
        io.delay_us(IDLE_POLL_US)?;
    }
    let status = service::read(io, FH_TX_STATUS)?;
    if status & FH_TX_IDLE == FH_TX_IDLE {
        Ok(())
    } else {
        Err(Error::StopTimeout(status))
    }
}

fn expect_zero(register: u32, observed: u32) -> Result<(), Error> {
    if observed == 0 {
        Ok(())
    } else {
        Err(Error::Rejected { register, observed })
    }
}

#[cfg(test)]
mod tests;
