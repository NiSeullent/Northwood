// SPDX-License-Identifier: GPL-2.0-or-later
//! Authenticated writeback to the same stopped and nonresident backing.

use super::{Error, ExclusiveDisk, PAGE_BYTES, RetainedSlot, Session, SlotProof};
use alloc::vec;
use core::fmt;
use stela_compiler::context::memory::MemoryStatus;
use stela_engine::{
    machine::engine::CpuMappingsRevokedFrame,
    memory::{domain::PublicationFault, system_ram::NonresidentRamFrame},
};
use zeroize::Zeroizing;

#[derive(Debug)]
pub enum FrameSwapError {
    Storage(Error),
    Frame(MemoryStatus),
    Publication(PublicationFault),
}

impl fmt::Display for FrameSwapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Storage(error) => error.fmt(f),
            Self::Frame(status) => write!(f, "held swap frame refused: {}", status.name()),
            Self::Publication(fault) => fault.fmt(f),
        }
    }
}

/// Holds the same disk session and slot proof until the original frame is
/// restored. Dropping it before restoration fails the RAM owner. The slot
/// stays claimed even if the token is forgotten.
#[must_use]
pub struct StoredFrame<'session, 'frame, D> {
    session: &'session mut Session<D>,
    proof: SlotProof,
    frame: Option<NonresidentRamFrame<'frame>>,
}

impl<D: ExclusiveDisk> Session<D> {
    /// Store a held frame in a slot already assigned to this operation by
    /// the caller. The canonical physical address supplies the page number.
    /// Call in host service context on a backend that does not re-enter the
    /// held RAM. A write failure leaves the original frame resident.
    pub fn store_revoked_frame<'frame>(
        &mut self,
        slot: u64,
        frame: CpuMappingsRevokedFrame<'frame>,
    ) -> Result<StoredFrame<'_, 'frame, D>, FrameSwapError> {
        if let Some(held) = &self.retained_slot {
            return Err(FrameSwapError::Storage(Error::SlotHeld(held.slot)));
        }
        let mut staging = Zeroizing::new(vec![0; PAGE_BYTES]);
        let page: &mut [u8; PAGE_BYTES] = staging.as_mut_slice().try_into().expect("one swap page");
        // SAFETY: this allocation belongs solely to the host and is disjoint
        // from the registered guest RAM and every live native frame.
        unsafe { frame.copy_into(page) }.map_err(FrameSwapError::Frame)?;
        let number = frame.physical() / PAGE_BYTES as u64;
        let proof = self
            .write_slot(slot, number, page)
            .map_err(FrameSwapError::Storage)?;
        let frame = frame.mark_nonresident().map_err(FrameSwapError::Frame)?;
        self.retained_slot = Some(RetainedSlot {
            slot,
            counter: proof.counter,
        });
        Ok(StoredFrame {
            session: self,
            proof,
            frame: Some(frame),
        })
    }
}

impl<D: ExclusiveDisk> StoredFrame<'_, '_, D> {
    pub fn physical(&self) -> u64 {
        self.frame.as_ref().map_or_else(
            || self.proof.page * PAGE_BYTES as u64,
            NonresidentRamFrame::physical,
        )
    }

    /// Clear the original bytes while keeping the same nonresident backing
    /// and its durable slot. A partial publication leaves the frame hidden.
    pub fn try_discard_retained_bytes(&mut self) -> Result<bool, FrameSwapError> {
        let Some(frame) = &mut self.frame else {
            return Err(FrameSwapError::Frame(MemoryStatus::InvalidRequest));
        };
        claimed_slot(self.session, &self.proof, frame.physical())?;
        let zeros = vec![0; PAGE_BYTES];
        let source = zeros.as_slice().try_into().expect("one swap page");
        // SAFETY: the durable proof is bound to this frame, the held owner
        // stays live, and the zero buffer is private and disjoint.
        unsafe { frame.try_overwrite_backing(source) }.map_err(FrameSwapError::Publication)
    }

    /// Read and authenticate the exact saved page before reopening RAM.
    /// Failure or a busy frame preserves the slot, proof and worker hold.
    /// An already completed restore returns true without another read.
    pub fn try_restore(&mut self) -> Result<bool, FrameSwapError> {
        let Some(frame) = &mut self.frame else {
            return Ok(true);
        };
        let physical = frame.physical();
        let slot = claimed_slot(self.session, &self.proof, physical)?;
        let number = physical / PAGE_BYTES as u64;
        let mut staging = Zeroizing::new(vec![0; PAGE_BYTES]);
        let page: &mut [u8; PAGE_BYTES] = staging.as_mut_slice().try_into().expect("one swap page");
        self.session
            .read_slot(slot, number, &self.proof, page)
            .map_err(FrameSwapError::Storage)?;
        // SAFETY: this private buffer came from the authenticated read of
        // this exact stopped frame's current durable slot and write receipt.
        let restored =
            unsafe { frame.try_restore_backing(page) }.map_err(FrameSwapError::Publication)?;
        if restored {
            self.frame = None;
            self.session.retained_slot = None;
        }
        Ok(restored)
    }

    #[cfg(all(test, feature = "std"))]
    pub(super) fn reference_copy_backing(
        &self,
        output: &mut [u8; PAGE_BYTES],
    ) -> Result<bool, MemoryStatus> {
        let frame = self.frame.as_ref().ok_or(MemoryStatus::InvalidRequest)?;
        // SAFETY: hosted fixtures pass private staging and keep registered RAM
        // exclusively behind the original held owner.
        unsafe { frame.reference_copy_backing(output) }
    }

    #[cfg(all(test, feature = "std"))]
    pub(super) fn reference_swap_proof(&mut self, proof: &mut SlotProof) {
        core::mem::swap(&mut self.proof, proof);
    }
}

fn claimed_slot<D>(
    session: &Session<D>,
    proof: &SlotProof,
    physical: u64,
) -> Result<u64, FrameSwapError> {
    let held = session.retained_slot.as_ref().filter(|held| {
        held.slot == proof.slot
            && held.counter == proof.counter
            && proof.generation == session.header.generation()
            && proof.page == physical / PAGE_BYTES as u64
    });
    held.map(|held| held.slot)
        .ok_or(FrameSwapError::Storage(Error::Authentication))
}
