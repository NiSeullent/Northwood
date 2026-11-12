// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2014-2019 aQuantia Corporation
// Copyright (C) 2019-2020 Marvell International Ltd.
//! AQtion descriptor rings and packet buffers in the requester's DMA domain.
use super::{Error, MAX_FRAME, MIN_FRAME, Result};
use crate::bus::dma::{
    mapping::{Direction, DmaDomain, DmaMapping},
    memory::{CacheAttribute, CpuMapping, PageSource},
};
use alloc::{sync::Arc, vec::Vec};
pub(super) const COUNT: usize = 64;
pub(super) const BUFFER: usize = 4096;
const DESCRIPTOR: usize = 16;
struct Buffer {
    cpu: CpuMapping,
    mapping: DmaMapping,
}
impl Buffer {
    fn new(pages: &dyn PageSource, domain: Arc<dyn DmaDomain>, bytes: usize) -> Result<Self> {
        let pin = pages.allocate(bytes, 4096, CacheAttribute::WriteBack)?;
        let cpu = pin.cpu_mapping(CacheAttribute::WriteBack)?;
        let mapping = DmaMapping::map(domain, pin, Direction::Bidirectional, u64::MAX)?;
        Ok(Self { cpu, mapping })
    }
    fn write(&self, offset: usize, bytes: &[u8]) -> Result<()> {
        let p = self.cpu.address(offset, bytes.len())?;
        for (i, b) in bytes.iter().enumerate() {
            // SAFETY: address checked the complete CPU-mapped span;
            // i is within bytes, and the serialized queue owns this exact
            // unpublished/completed range. Byte stores preserve its neighbors.
            unsafe { p.as_ptr().add(i).write_volatile(*b) }
        }
        Ok(())
    }
    fn word(&self, offset: usize, value: u64) -> Result<()> {
        self.write(offset, &value.to_le_bytes())
    }
    fn address(&self, offset: usize) -> Result<u64> {
        Ok(self.mapping.address()? + offset as u64)
    }
}
pub(super) struct Memory {
    retiring: bool,
    tx: Buffer,
    rx: Buffer,
    tx_data: Buffer,
    rx_data: Buffer,
}
impl Memory {
    pub fn new(pages: &dyn PageSource, domain: Arc<dyn DmaDomain>) -> Result<Self> {
        Ok(Self {
            retiring: false,
            tx: Buffer::new(pages, domain.clone(), 4096)?,
            rx: Buffer::new(pages, domain.clone(), 4096)?,
            tx_data: Buffer::new(pages, domain.clone(), COUNT * BUFFER)?,
            rx_data: Buffer::new(pages, domain, COUNT * BUFFER)?,
        })
    }
    pub fn is_retiring(&self) -> bool {
        self.retiring
    }
    pub fn tx_address(&self) -> Result<u64> {
        self.tx.address(0)
    }
    pub fn rx_address(&self) -> Result<u64> {
        self.rx.address(0)
    }
    pub fn prepare(&mut self) -> Result<()> {
        for index in 0..COUNT {
            self.tx.word(index * DESCRIPTOR, 0)?;
            self.tx.word(index * DESCRIPTOR + 8, 0)?;
            self.offer(index)?;
        }
        self.tx.mapping.sync_for_device(0..COUNT * DESCRIPTOR)?;
        // Independently checked coherent descriptor observation is mandatory;
        // a legacy domain cannot silently degrade to a CPU fence.
        self.rx.mapping.read_coherent_u32(8)?;
        Ok(())
    }
    pub fn offer(&mut self, index: usize) -> Result<()> {
        self.rx_data
            .mapping
            .sync_for_device(index * BUFFER..(index + 1) * BUFFER)?;
        self.rx
            .word(index * DESCRIPTOR, self.rx_data.address(index * BUFFER)?)?;
        self.rx.word(index * DESCRIPTOR + 8, 0)?;
        self.rx
            .mapping
            .sync_for_device(index * DESCRIPTOR..(index + 1) * DESCRIPTOR)?;
        Ok(())
    }
    pub fn transmit(&mut self, index: usize, frame: &[u8]) -> Result<()> {
        let length = frame.len().max(MIN_FRAME);
        if length > MAX_FRAME || index >= COUNT {
            return Err(Error::Unsupported("AQtion frame bounds"));
        }
        let base = index * BUFFER;
        self.tx_data.write(base, frame)?;
        if length > frame.len() {
            self.tx_data
                .write(base + frame.len(), &[0; MIN_FRAME][..length - frame.len()])?;
        }
        self.tx_data.mapping.sync_for_device(base..base + length)?;
        self.tx
            .word(index * DESCRIPTOR, self.tx_data.address(base)?)?;
        // TXD, buffer length, end-of-packet, writeback. Linux B0 normal path
        // has no FCS command; the MAC appends FCS through ordinary TX setup.
        let ctl = 1 | ((length as u32) << 4) | (1 << 21) | (1 << 27);
        let ctl2 = (length as u32) << 14;
        self.tx.word(
            index * DESCRIPTOR + 8,
            u64::from(ctl) | (u64::from(ctl2) << 32),
        )?;
        self.tx
            .mapping
            .sync_for_device(index * DESCRIPTOR..(index + 1) * DESCRIPTOR)?;
        Ok(())
    }
    pub fn complete_tx(&self, head: usize, count: usize) -> Result<()> {
        for n in 0..count {
            let i = (head + n) % COUNT;
            // SAFETY: hardware TX head passed these exact submitted
            // descriptors; it will not read them again before a new tail.
            unsafe {
                self.tx_data
                    .mapping
                    .sync_for_cpu(i * BUFFER..(i + 1) * BUFFER)?;
            }
        }
        Ok(())
    }
    pub fn receive_status(&self, index: usize) -> Result<Option<(u32, u16, usize)>> {
        let status = self.rx.mapping.read_coherent_u32(index * DESCRIPTOR + 8)?;
        if status & 1 == 0 {
            return Ok(None);
        }
        // DD is the device's last descriptor write. Coherent word reads avoid
        // invalidating cache lines still shared with other active descriptors.
        let kind = self.rx.mapping.read_coherent_u32(index * DESCRIPTOR)?;
        Ok(Some((kind, status as u16, (status >> 16) as usize)))
    }
    /// Copy a completed frame into `frame`, reusing its allocation. On an
    /// error `frame` is unchanged.
    pub fn receive_bytes(&self, index: usize, length: usize, frame: &mut Vec<u8>) -> Result<()> {
        let start = index * BUFFER;
        // SAFETY: receive_status observed DD for this descriptor. The
        // device cannot access this page again before offer() and RX tail.
        unsafe { self.rx_data.mapping.sync_for_cpu(start..start + BUFFER) }?;
        let p = self.rx_data.cpu.address(start, length)?;
        frame
            .try_reserve_exact(length.saturating_sub(frame.len()))
            .map_err(|_| Error::Memory(crate::bus::dma::memory::MemoryError::OutOfMemory))?;
        frame.clear();
        for i in 0..length {
            // SAFETY: completed, checked packet range, no live DMA.
            frame.push(unsafe { p.as_ptr().add(i).read_volatile() });
        }
        Ok(())
    }
    pub fn retire(&mut self) -> Result<()> {
        // Revoke rebuild before any invalidation. A failed unmap may
        // already have retired a prefix; retry may only finish this retirement.
        self.retiring = true;
        for buffer in [
            &mut self.tx,
            &mut self.rx,
            &mut self.tx_data,
            &mut self.rx_data,
        ] {
            if buffer.mapping.address().is_ok() {
                // SAFETY: Aqtion::retire completed stop's engine/BME
                // containment and original-domain write drain before entry.
                // This kept mapping is still live; calls are serialized
                // and there are no registered IRQ workers or new submissions.
                unsafe { buffer.mapping.unmap() }?;
            }
        }
        Ok(())
    }
}
