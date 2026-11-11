// SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause
// Copyright (C) 2005-2015, 2018-2026 Intel Corporation
// Copyright (C) 2013-2015 Intel Mobile Communications GmbH
// Copyright (C) 2016-2017 Intel Deutschland GmbH
//! Polled FH service-channel transfer for legacy Intel firmware loading.
//!
//! Uses the pinned Linux gen1_2 service-channel programming and CSR FH-TX
//! completion protocol. The MAC/PCI preparation, CPU release, section status,
//! paging and ALIVE handshake belong to the surrounding legacy transport.

use super::{FH_DMA_LIMIT, Transfer, rfh::FH_RX_MASK};
use crate::{
    bus::dma::{
        mapping::{Direction, DmaDomain},
        memory::PageSource,
    },
    wifi::intel::{
        boot::BootIo,
        dma::{self, Chunk},
        interrupts::{INT_HW_ERR, INT_SW_ERR},
        power::{AccessError, Csr, is_hardware_error},
    },
};
use alloc::sync::Arc;

pub(super) const INT: u32 = 0x08;
pub(super) const INT_MASK: u32 = 0x0c;
pub(super) const FH_INT: u32 = 0x10;
pub(super) const GP_CONTROL: u32 = 0x24;
pub(super) const ICT: u32 = 0xa0;
pub(super) const ICT_ENABLED: u32 = 1 << 31;
pub(super) const INT_FH_TX: u32 = 1 << 27;
const FH_TX: u32 = 3;
const FH_ERROR: u32 = 1 << 31;
pub(super) const MAC_ACCESS: u32 = 1 << 3;
const MAC_CLOCK: u32 = 1;
const GOING_TO_SLEEP: u32 = 1 << 4;
const PRPH_WADDR: u32 = 0x444;
const PRPH_RADDR: u32 = 0x448;
const PRPH_WDATA: u32 = 0x44c;
const PRPH_RDATA: u32 = 0x450;
const LMPM_CHICK: u32 = 0x1ff8;
const EXTENDED_SRAM: u32 = 1;
const COMPLETION_POLL_US: u32 = 1_000;
const COMPLETION_POLLS: usize = 5_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    Access(AccessError),
    Dma(dma::Error),
    InvalidAddress,
    RegisterUnavailable { register: u32, value: u32 },
    InterruptMode,
    DeviceFault { csr: u32, fh: u32 },
    CompletionNotCleared,
    NicAccess,
    ExtendedSram,
    Timeout,
    Unavailable,
}

impl From<AccessError> for Error {
    fn from(error: AccessError) -> Self {
        Self::Access(error)
    }
}

impl From<dma::Error> for Error {
    fn from(error: dma::Error) -> Self {
        Self::Dma(error)
    }
}

/// Exclusive owner of one prepared FH service channel and its in-flight pin.
pub struct ServiceChannel<'a, Q: BootIo> {
    io: Option<Q>,
    source: &'a dyn PageSource,
    domain: Arc<dyn DmaDomain>,
    transfer: TransferState,
    nic_requested: bool,
}

impl<'a, Q: BootIo> ServiceChannel<'a, Q> {
    /// # Safety
    /// `io` owns the awake Intel 9000 requester, with service-channel DMA
    /// idle and no other channel sharing FH-TX. The device routes causes
    /// through direct CSRs, with no ICT or MSI-X, and enables only FH-TX in
    /// CSR_INT_MASK. PCI interrupt delivery remains disabled. No other owner
    /// may acknowledge these causes or access NIC or reset registers during
    /// the epoch. `domain` is the exact requester's bound DMA domain and
    /// `source` supplies its compatible kept physical pages.
    pub unsafe fn new(io: Q, source: &'a dyn PageSource, domain: Arc<dyn DmaDomain>) -> Self {
        Self {
            io: Some(io),
            source,
            domain,
            transfer: TransferState::new(),
            nic_requested: false,
        }
    }

    /// Complete one resident chunk before another may be submitted.
    pub fn transfer(&mut self, transfer: Transfer<'_>) -> Result<(), Error> {
        let io = self.io.as_mut().ok_or(Error::Unavailable)?;
        self.transfer.transfer(
            io,
            &mut self.nic_requested,
            self.source,
            &self.domain,
            transfer,
        )
    }

    /// Return the function owner when every submitted transfer completed.
    pub fn finish(&mut self) -> Result<Q, Error> {
        if self.transfer.poisoned {
            return Err(Error::Unavailable);
        }
        self.io.take().ok_or(Error::Unavailable)
    }

    /// Reclaim the owner of an uncertain channel after external quiescence.
    /// Failed invalidation keeps the pin and allows another cleanup attempt.
    ///
    /// # Safety
    /// A caller must independently prove that this exact requester has
    /// stopped DMA, completed every outstanding access and drained pending
    /// submissions and IRQ work. An FH timeout or posted stop is insufficient.
    pub unsafe fn reclaim_after_stop(&mut self) -> Result<Q, Error> {
        // SAFETY: the caller supplied quiescence for this exact DMA epoch.
        unsafe { self.transfer.release_pin()? };
        self.transfer.poisoned = false;
        self.finish()
    }
}

impl<Q: BootIo> Drop for ServiceChannel<'_, Q> {
    fn drop(&mut self) {
        if self.transfer.poisoned
            && let Some(io) = self.io.take()
        {
            // Chunk's exposed lease quarantines the pin and IOMMU domain.
            // Keep the BAR and PCI claim with them until quiescence is known.
            core::mem::forget(io);
        }
    }
}

pub(super) struct TransferState {
    pending: Option<Chunk>,
    poisoned: bool,
}

impl TransferState {
    pub(super) const fn new() -> Self {
        Self {
            pending: None,
            poisoned: false,
        }
    }

    pub(super) fn transfer(
        &mut self,
        io: &mut impl BootIo,
        nic_requested: &mut bool,
        source: &dyn PageSource,
        domain: &Arc<dyn DmaDomain>,
        transfer: Transfer<'_>,
    ) -> Result<(), Error> {
        self.transfer_masked(io, nic_requested, source, domain, transfer, INT_FH_TX)
    }

    /// Transfer a CPU2 chunk while preserving any pending RFH causes.
    pub(super) fn transfer_with_rx(
        &mut self,
        io: &mut impl BootIo,
        nic_requested: &mut bool,
        source: &dyn PageSource,
        domain: &Arc<dyn DmaDomain>,
        transfer: Transfer<'_>,
    ) -> Result<(), Error> {
        self.transfer_masked(io, nic_requested, source, domain, transfer, FH_RX_MASK)
    }

    fn transfer_masked(
        &mut self,
        io: &mut impl BootIo,
        nic_requested: &mut bool,
        source: &dyn PageSource,
        domain: &Arc<dyn DmaDomain>,
        transfer: Transfer<'_>,
        mask: u32,
    ) -> Result<(), Error> {
        if self.poisoned {
            return Err(Error::Unavailable);
        }
        let mut chunk = Chunk::allocate_bounded(
            source,
            domain.clone(),
            transfer.data(),
            Direction::ToDevice,
            FH_DMA_LIMIT - 1,
        )?;
        let writes = transfer
            .register_writes(chunk.region().address)
            .ok_or(Error::InvalidAddress)?;

        self.poisoned = true;
        if read(io, INT_MASK)? != mask || read(io, ICT)? & ICT_ENABLED != 0 {
            return Err(Error::InterruptMode);
        }
        let (csr, fh) = causes(io)?;
        if csr & INT_FH_TX != 0 || fh & FH_TX != 0 {
            clear_completion(io)?;
        }

        let old_mode = with_nic(io, nic_requested, |io| {
            let old_mode = extended_sram(io, transfer.extended_sram())?;
            // Even a failed CSR write can have published the descriptor.
            // Keep the lease before giving the channel its IOVA.
            chunk.expose();
            self.pending = Some(chunk);
            for (register, value) in writes {
                io.write32(register, value)?;
            }
            Ok::<_, Error>(old_mode)
        })?;
        wait_completion(io)?;
        if old_mode != transfer.extended_sram() {
            with_nic(io, nic_requested, |io| extended_sram(io, old_mode))?;
        }
        // SAFETY: a fresh and acknowledged FH-TX completed this single
        // service descriptor. Direct polling has no IRQ or worker to drain.
        unsafe { self.release_pin()? };
        self.poisoned = false;
        Ok(())
    }

    /// # Safety
    /// No device access can still reach the pending mapping.
    pub(super) unsafe fn release_pin(&mut self) -> Result<(), Error> {
        if let Some(pending) = &mut self.pending {
            // SAFETY: the caller proved no accesses can still reach the pin.
            unsafe { pending.release() }.map_err(dma::Error::from)?;
        }
        self.pending = None;
        Ok(())
    }
}

pub(super) fn read(io: &mut impl Csr, register: u32) -> Result<u32, Error> {
    let value = io.read32(register)?;
    if value == u32::MAX || is_hardware_error(value) {
        return Err(Error::RegisterUnavailable { register, value });
    }
    Ok(value)
}

pub(super) fn causes(io: &mut impl Csr) -> Result<(u32, u32), Error> {
    let csr = read(io, INT)?;
    let fh = read(io, FH_INT)?;
    if csr & (INT_SW_ERR | INT_HW_ERR) != 0 || fh & FH_ERROR != 0 {
        return Err(Error::DeviceFault { csr, fh });
    }
    Ok((csr, fh))
}

fn clear_completion(io: &mut impl Csr) -> Result<(), Error> {
    io.write32(INT, INT_FH_TX)?;
    io.write32(FH_INT, FH_TX)?;
    let (csr, fh) = causes(io)?;
    if csr & INT_FH_TX != 0 || fh & FH_TX != 0 {
        return Err(Error::CompletionNotCleared);
    }
    Ok(())
}

fn wait_completion(io: &mut impl Csr) -> Result<(), Error> {
    for _ in 0..COMPLETION_POLLS {
        if causes(io)?.0 & INT_FH_TX != 0 {
            return clear_completion(io);
        }
        io.delay_us(COMPLETION_POLL_US)?;
    }
    Err(Error::Timeout)
}

pub(super) fn with_nic<Q: Csr, T, E: From<Error>>(
    io: &mut Q,
    requested: &mut bool,
    access: impl FnOnce(&mut Q) -> Result<T, E>,
) -> Result<T, E> {
    if *requested {
        release_nic(io)?;
        *requested = false;
    }
    let gp = read(io, GP_CONTROL)?;
    if gp & MAC_ACCESS != 0 {
        return Err(Error::NicAccess.into());
    }
    // A failed request write can still take effect. Hold custody until its
    // release is confirmed, including on a later requester-wide stop.
    *requested = true;
    let result = request_nic(io, gp)
        .map_err(E::from)
        .and_then(|()| access(io));
    let cleanup = release_nic(io);
    if cleanup.is_ok() {
        *requested = false;
    }
    let value = result?;
    cleanup?;
    Ok(value)
}

fn request_nic(io: &mut impl Csr, gp: u32) -> Result<(), Error> {
    io.write32(GP_CONTROL, gp | MAC_ACCESS)?;
    io.delay_us(2)?;
    for _ in 0..1500 {
        if read(io, GP_CONTROL)? & (MAC_ACCESS | MAC_CLOCK | GOING_TO_SLEEP)
            == MAC_ACCESS | MAC_CLOCK
        {
            return Ok(());
        }
        io.delay_us(10)?;
    }
    Err(Error::NicAccess)
}

fn release_nic(io: &mut impl Csr) -> Result<(), Error> {
    let gp = read(io, GP_CONTROL)?;
    io.write32(GP_CONTROL, gp & !MAC_ACCESS)?;
    if read(io, GP_CONTROL)? & MAC_ACCESS != 0 {
        return Err(Error::NicAccess);
    }
    Ok(())
}

const fn prph_selector(register: u32) -> u32 {
    (register & 0x000f_ffff) | (3 << 24)
}

/// Access a peripheral register while the caller holds NIC access.
pub(super) fn read_prph(io: &mut impl Csr, register: u32) -> Result<u32, Error> {
    io.write32(PRPH_RADDR, prph_selector(register))?;
    read(io, PRPH_RDATA)
}

pub(super) fn write_prph(io: &mut impl Csr, register: u32, value: u32) -> Result<(), Error> {
    io.write32(PRPH_WADDR, prph_selector(register))?;
    io.write32(PRPH_WDATA, value)?;
    Ok(())
}

/// Return the previous mode, preserving unrelated bits across the change.
fn extended_sram(io: &mut impl Csr, enabled: bool) -> Result<bool, Error> {
    let old = read_prph(io, LMPM_CHICK)?;
    let new = (old & !EXTENDED_SRAM) | u32::from(enabled);
    if new != old {
        write_prph(io, LMPM_CHICK, new)?;
        if read_prph(io, LMPM_CHICK)? & EXTENDED_SRAM != new & EXTENDED_SRAM {
            return Err(Error::ExtendedSram);
        }
    }
    Ok(old & EXTENDED_SRAM != 0)
}

#[cfg(test)]
mod tests;
