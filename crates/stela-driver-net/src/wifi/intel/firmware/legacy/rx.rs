// SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause
// Copyright (C) 2003-2014, 2018-2024, 2026 Intel Corporation
// Copyright (C) 2013-2015 Intel Mobile Communications GmbH
// Copyright (C) 2016-2017 Intel Deutschland GmbH
//! Default RFH RX queue for Intel 9000 firmware startup. Layout and
//! completion follow Linux iwlwifi at cc9de5d4b5ab6459f5774946aa3195ad51236fa8.

use super::{
    FH_DMA_LIMIT, Firmware,
    alive::{self, Completion, Progress, Sequence},
};
use crate::{
    bus::dma::{
        mapping::{Direction, DmaDomain, DmaError},
        memory::PageSource,
    },
    wifi::intel::{
        dma::{self, Chunk},
        receive::{self, Packets},
    },
};
use alloc::{sync::Arc, vec::Vec};
use core::sync::atomic::{Ordering, fence};
use stela_driver_base::Clock;

pub const ENTRIES: u16 = 512;
pub const BUFFER_BYTES: usize = 4096;
const POSTED: u16 = (ENTRIES - 1) & !7;
const STATUS_BYTES: usize = 12;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    pub free: u64,
    pub used: u64,
    pub status: u64,
    pub write: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    Dma(dma::Error),
    Wire(receive::Error),
    Sequence(alive::Error),
    InvalidAddress,
    State,
    Generation,
    Producer { previous: u16, observed: u16 },
    CompletionId(u16),
    NoReceiveCredit,
}

impl From<dma::Error> for Error {
    fn from(error: dma::Error) -> Self {
        Self::Dma(error)
    }
}

impl From<receive::Error> for Error {
    fn from(error: receive::Error) -> Self {
        Self::Wire(error)
    }
}

impl From<alive::Error> for Error {
    fn from(error: alive::Error) -> Self {
        Self::Sequence(error)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    Prepared,
    Exposed,
    Faulted,
    Retiring,
    Stopped,
}

struct Buffer {
    chunk: Chunk,
    consumed: bool,
}

/// Owns the pinned RFH ring and receive buffers for one reset generation.
/// Startup has a finite batch of receive credits, with no VID reuse. A new
/// generation needs newly prepared DMA and an acknowledged requester stop.
pub struct Queue {
    generation: u64,
    free: Chunk,
    used: Chunk,
    status: Chunk,
    buffers: Vec<Buffer>,
    producer: u16,
    read: u16,
    bound: bool,
    state: State,
}

impl Queue {
    /// Allocate the default queue with 4 KiB RBs in the requester's checked
    /// address range. The transport still owns register programming.
    pub fn prepare(
        source: &dyn PageSource,
        domain: Arc<dyn DmaDomain>,
        address_limit: u64,
        generation: u64,
    ) -> Result<Self, Error> {
        let address_limit = address_limit.min(FH_DMA_LIMIT - 1);
        let allocate = |bytes: &[u8], direction| {
            Chunk::allocate_bounded(source, domain.clone(), bytes, direction, address_limit)
        };
        let mut descriptors = [0; ENTRIES as usize * 8];
        let mut buffers = Vec::with_capacity(usize::from(POSTED));
        let empty = [0; BUFFER_BYTES];
        for (index, descriptor) in descriptors
            .chunks_exact_mut(8)
            .take(POSTED.into())
            .enumerate()
        {
            let chunk = allocate(&empty, Direction::FromDevice)?;
            descriptor.copy_from_slice(&receive::pre_ax210_mq_free(
                chunk.region().address,
                index as u16 + 1,
            )?);
            buffers.push(Buffer {
                chunk,
                consumed: false,
            });
        }
        let free = allocate(&descriptors, Direction::ToDevice)?;
        let used = allocate(&[0; ENTRIES as usize * 4], Direction::Bidirectional)?;
        let status = allocate(&[0; STATUS_BYTES], Direction::Bidirectional)?;
        for chunk in [&free, &used, &status] {
            let address = chunk.region().address;
            if address == 0 || !address.is_multiple_of(4096) {
                return Err(Error::InvalidAddress);
            }
        }
        // Require the mapping backend to support shared coherent observation
        // of both DMA records before any address can escape to RFH.
        if status.observe_u32(0)? != 0 || used.observe_u32(0)? != 0 {
            return Err(Error::State);
        }
        Ok(Self {
            generation,
            free,
            used,
            status,
            buffers,
            producer: 0,
            read: 0,
            bound: false,
            state: State::Prepared,
        })
    }

    /// Keep the DMA lease before publishing these addresses to the device.
    /// # Safety
    /// Only the exclusive, stopped Intel 9000 requester bound to this domain
    /// may receive this layout. The caller resets RFH indexes, configures the
    /// default queue for 512 entries and 4 KiB RBs with snooping, disables
    /// PCIe No Snoop and relaxed ordering, and gives it only `write` credits.
    /// The same owner must retain the PCI claim
    /// until requester stop and must never reuse this generation.
    pub unsafe fn expose(&mut self) -> Result<Layout, Error> {
        if self.state != State::Prepared {
            return Err(Error::State);
        }
        for chunk in [&mut self.free, &mut self.used, &mut self.status] {
            chunk.expose();
        }
        for buffer in &mut self.buffers {
            buffer.chunk.expose();
        }
        self.state = State::Exposed;
        fence(Ordering::Release);
        Ok(Layout {
            free: self.free.region().address,
            used: self.used.region().address,
            status: self.status.region().address,
            write: POSTED,
        })
    }

    fn observe_producer(&mut self) -> Result<u16, Error> {
        if self.state != State::Exposed {
            return Err(Error::State);
        }
        // closed_rb_num is the low u16 of the first coherent status word.
        // The queue mask also covers the documented 9000 A0 wrap quirk.
        let result = self
            .status
            .observe_u32(0)
            .map(|word| word as u16 & (ENTRIES - 1));
        let observed = result.inspect_err(|_| self.state = State::Faulted)?;
        if observed < self.producer || observed > POSTED {
            self.state = State::Faulted;
            return Err(Error::Producer {
                previous: self.producer,
                observed,
            });
        }
        self.producer = observed;
        Ok(observed)
    }

    /// Observe one completed DMA buffer. No interrupt bit grants ownership.
    /// The exclusive producer and an outstanding VID must both agree.
    pub fn complete(&mut self) -> Result<Option<CompletedBuffer>, Error> {
        if self.observe_producer()? == self.read {
            return Ok(None);
        }
        let result = self.take_completed();
        if result.is_err() {
            self.state = State::Faulted;
        }
        result.map(Some)
    }

    fn take_completed(&mut self) -> Result<CompletedBuffer, Error> {
        fence(Ordering::Acquire);
        let index = self.read;
        let vid = receive::pre_ax210_mq_vid(self.used.observe_u32(usize::from(index) * 4)?);
        let buffer = vid
            .checked_sub(1)
            .and_then(|index| self.buffers.get_mut(usize::from(index)))
            .filter(|buffer| !buffer.consumed)
            .ok_or(Error::CompletionId(vid))?;
        // SAFETY: the closed index covers this used descriptor and its VID
        // uniquely names one published buffer which is never returned to RFH.
        let bytes = unsafe { buffer.chunk.snapshot_completed()? };
        buffer.consumed = true;
        self.read += 1;
        Ok(CompletedBuffer {
            generation: self.generation,
            index,
            bytes,
        })
    }

    /// Start the ALIVE deadline after INIT firmware start is acknowledged.
    pub fn wait_init(&mut self, firmware: &Firmware<'_>, clock: Clock) -> Result<Sequence, Error> {
        if self.state != State::Exposed || self.bound {
            return Err(Error::State);
        }
        let sequence = Sequence::wait_init(firmware, clock, self.generation)?;
        self.bound = true;
        Ok(sequence)
    }

    /// Arm INIT_COMPLETE immediately before the transport publishes the final
    /// PHY calibration command. Older buffers cannot complete this wait, even
    /// when an already closed buffer has yet to be read or dispatched.
    pub fn wait_init_complete(&mut self, sequence: &mut Sequence) -> Result<(), Error> {
        self.arm_init_complete(sequence).map(|_| ())
    }

    pub(super) fn arm_init_complete(
        &mut self,
        sequence: &mut Sequence,
    ) -> Result<ResponseCutoff, Error> {
        if !sequence.accepts_generation(self.generation) {
            return Err(Error::Generation);
        }
        let cutoff = self.response_cutoff()?;
        sequence.wait_init_complete(cutoff.generation, cutoff.index)?;
        Ok(cutoff)
    }

    /// Fence off descriptors the device had already closed before a command
    /// doorbell. The cutoff is issued only by this queue's coherent producer.
    pub(super) fn response_cutoff(&mut self) -> Result<ResponseCutoff, Error> {
        let index = self.observe_producer()?;
        if index == POSTED {
            return Err(Error::NoReceiveCredit);
        }
        Ok(ResponseCutoff {
            generation: self.generation,
            index,
        })
    }

    /// Start REGULAR's ALIVE deadline on the new queue after the old
    /// requester has stopped and REGULAR firmware start is acknowledged.
    pub fn wait_regular_after_init_stop(&mut self, sequence: &mut Sequence) -> Result<(), Error> {
        if self.state != State::Exposed || self.bound {
            return Err(Error::State);
        }
        sequence.wait_regular_after_init_stop(self.generation)?;
        self.bound = true;
        Ok(())
    }

    /// Retryable cleanup, including after an invalid DMA receipt.
    /// # Safety
    /// The exact requester has stopped DMA and no pending doorbell or IRQ
    /// work references this queue. Timeouts and firmware notifications do not
    /// prove quiescence.
    pub unsafe fn release_after_stop(&mut self) -> Result<(), DmaError> {
        self.state = State::Retiring;
        for chunk in [&mut self.free, &mut self.used, &mut self.status] {
            // SAFETY: the caller established quiescence for this queue.
            unsafe { chunk.release()? };
        }
        for buffer in &mut self.buffers {
            // SAFETY: the same proof covers every published payload.
            unsafe { buffer.chunk.release()? };
        }
        self.state = State::Stopped;
        Ok(())
    }
}

/// A synchronized CPU copy, retaining the generation and used index from the
/// original queue. Callers cannot relabel a delayed snapshot as current RX.
pub struct CompletedBuffer {
    generation: u64,
    index: u16,
    bytes: Vec<u8>,
}

#[derive(Clone, Copy)]
pub(super) struct ResponseCutoff {
    generation: u64,
    index: u16,
}

impl ResponseCutoff {
    pub(super) fn accepts(self, buffer: &CompletedBuffer) -> bool {
        buffer.generation == self.generation && buffer.index >= self.index
    }
}

impl CompletedBuffer {
    /// The same packet framing serves command responses and PHY DB data.
    pub fn packets(&self) -> Result<Packets<'_>, receive::Error> {
        Packets::pre_ax210_mq(0, &self.bytes)
    }

    pub(super) fn checked_packets(&self) -> Result<Packets<'_>, receive::Error> {
        let packets = self.packets()?;
        packets.clone().try_for_each(|packet| packet.map(|_| ()))?;
        Ok(packets)
    }

    pub(super) fn completion<'a>(&self, packet: &receive::Packet<'a>) -> Completion<'a> {
        Completion {
            generation: self.generation,
            index: self.index,
            group: packet.group,
            command: packet.command,
            payload: packet.payload(),
        }
    }
}

/// Dispatch only data reclaimed by the RFH queue. A poll without a buffer
/// still checks the current deadline. Keep other packets available to the
/// command owner through `CompletedBuffer::packets`.
pub fn poll_notification(
    sequence: &mut Sequence,
    buffer: Option<&CompletedBuffer>,
) -> Result<Progress, Error> {
    sequence.poll(None)?;
    let Some(buffer) = buffer.filter(|buffer| sequence.accepts_generation(buffer.generation))
    else {
        return Ok(Progress::Pending);
    };
    let packets = buffer.checked_packets()?;
    let mut observed = Progress::Pending;
    for packet in packets {
        let packet = packet?;
        let progress = sequence.poll(Some(buffer.completion(&packet)))?;
        match progress {
            Progress::Pending => {}
            Progress::Malformed { .. } => observed = progress,
            _ => return Ok(progress),
        }
    }
    Ok(observed)
}

#[cfg(test)]
mod tests;
