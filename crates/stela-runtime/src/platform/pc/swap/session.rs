// SPDX-License-Identifier: GPL-2.0-or-later
//! Synchronous, encrypted v1 slot I/O on an exclusively owned disk.
//! Call this in host service context. The frame allocator owns slot
//! lifetimes and must retain the RAM copy until a durable write succeeds.

use super::{Error as HeaderError, Header, MAGIC, SLOT_BYTES, partition, volume};
use crate::pc::gpt::{self, Guid, PartitionDevice, geometry};
use alloc::{vec, vec::Vec};
use chacha20poly1305::{AeadInOut, ChaCha20Poly1305, KeyInit};
use core::fmt;
use stela_crypto::random::Generator;
use stela_drivers::block::{BlockDevice, BlockError, Identity, Request};
use stela_formats::payload::pieces::Extent;
use zeroize::Zeroizing;

/// The caller exclusively owns the whole disk for the session, including
/// its partition table. The backend keeps the selected media, logical
/// block size, capacity and durability features fixed or rejects a change
/// before admitting each command. Returning success or failure must end
/// access to the supplied buffers. An unacknowledged command or a media
/// change cannot be retried on a newly attached device.
pub trait ExclusiveDisk: BlockDevice {}

impl ExclusiveDisk for stela_drivers::storage::nvme::ExclusiveNamespace {}

const PAGE_BYTES: usize = SLOT_BYTES as usize;

/// In-memory receipt for one completed write. Keep only the current
/// receipt with the page and discard it when its slot is freed. Nothing
/// from an old boot is read back to recreate a receipt.
#[derive(Debug)]
pub struct SlotProof {
    slot: u64,
    page: u64,
    generation: u64,
    counter: u64,
    tag: [u8; 16],
}

pub struct Session<D> {
    disk: PartitionDevice<D>,
    placement: Placement,
    expected: Identity,
    header: Header,
    cipher: ChaCha20Poly1305,
    counter: u64,
    retained_slot: Option<RetainedSlot>,
}

struct RetainedSlot {
    slot: u64,
    counter: u64,
}

enum Placement {
    Partition,
    Fat32(volume::VerifiedLayout),
}

impl Placement {
    fn runs(&self, offset: u64, bytes: u64) -> impl Iterator<Item = Extent> + '_ {
        let (contiguous, fragmented) = match self {
            Self::Partition => (Some(Extent { offset, bytes }), None),
            Self::Fat32(layout) => (None, Some(layout.runs_at(offset, bytes))),
        };
        contiguous
            .into_iter()
            .chain(fragmented.into_iter().flatten())
    }
}

mod data_volume;
mod held_frame;
pub use held_frame::{FrameSwapError, StoredFrame};

impl<D: ExclusiveDisk> Session<D> {
    /// Open a named partition after rechecking both GPT copies and its
    /// header. Start a new volatile generation with a fresh boot-local key.
    /// The generator must come from an initialized host entropy source.
    /// An error from the header write can leave the header partially
    /// updated. It does not open a session or repair the store.
    pub fn open(mut disk: D, uuid: Guid, random: &mut Generator) -> Result<Self, Error> {
        if disk.identity().read_only {
            return Err(Error::Device(BlockError::ReadOnly));
        }
        let expected = disk.identity().clone();
        let (partition, previous) =
            partition::inspect_for_write(&mut disk, uuid).map_err(Error::Partition)?;
        Self::begin(
            PartitionDevice::new(disk, partition),
            expected,
            Placement::Partition,
            previous,
            random,
        )
    }

    /// Consume the exclusive whole-disk owner before qualifying the FAT32
    /// layout. Both the DATA partition and prepared store must match the
    /// expected GUIDs. The owner must exclude all other access from before
    /// GPT verification until this session is dropped.
    pub fn open_volume(
        mut disk: D,
        data_uuid: Guid,
        store_uuid: Guid,
        maximum_bytes: u64,
        random: &mut Generator,
    ) -> Result<Self, Error> {
        let expected = disk.identity().clone();
        data_volume::check_features(&expected)?;
        let partition = data_volume::partition(&mut disk, data_uuid)?;
        check_disk(disk.identity(), &expected)?;
        let mut disk = PartitionDevice::new(disk, partition);
        let layout = data_volume::verify_layout(&mut disk, &expected, maximum_bytes)?;
        let previous = layout.metadata().header;
        if previous.store_uuid() != store_uuid {
            return Err(Error::StoreIdentity);
        }
        Self::begin(disk, expected, Placement::Fat32(layout), previous, random)
    }

    fn begin(
        disk: PartitionDevice<D>,
        expected: Identity,
        placement: Placement,
        previous: Header,
        random: &mut Generator,
    ) -> Result<Self, Error> {
        check_disk(disk.disk(), &expected)?;
        let mut generation = [0u8; 8];
        random.fill(&mut generation);
        let header = previous
            .begin_session(u64::from_le_bytes(generation), true)
            .map_err(Error::Header)?;
        let mut key = Zeroizing::new([0u8; 32]);
        random.fill(&mut *key);
        let cipher = ChaCha20Poly1305::new((&*key).into());
        let mut session = Self {
            disk,
            placement,
            expected,
            header,
            cipher,
            counter: 0,
            retained_slot: None,
        };
        session.write_store(0, &header.encode())?;
        Ok(session)
    }

    pub fn header(&self) -> Header {
        self.header
    }

    /// Write a page to a free slot, or replace an old copy while a separate
    /// authoritative copy is kept. A failed replacement can tear the old
    /// slot. The caller owns that slot and its retirement.
    pub fn write_slot(
        &mut self,
        slot: u64,
        page: u64,
        plaintext: &[u8; PAGE_BYTES],
    ) -> Result<SlotProof, Error> {
        let offset = self.slot_offset(slot)?;
        if self
            .retained_slot
            .as_ref()
            .is_some_and(|held| held.slot == slot)
        {
            return Err(Error::SlotHeld(slot));
        }
        check_disk(self.disk.disk(), &self.expected)?;
        // Never reuse a nonce, even if a failed write reached the device.
        let counter = self.counter.checked_add(1).ok_or(Error::NonceExhausted)?;
        self.counter = counter;
        let mut bytes = Zeroizing::new(plaintext.to_vec());
        let tag = self
            .cipher
            .encrypt_inout_detached(
                (&nonce(counter)).into(),
                &self.aad(slot, page),
                bytes.as_mut_slice().into(),
            )
            .map_err(|_| Error::Authentication)?;
        // Only ciphertext enters the block layer. It owns this buffer until
        // the write and its durability requirement have both completed.
        self.write_store(offset, &bytes)?;
        Ok(SlotProof {
            slot,
            page,
            generation: self.header.generation(),
            counter,
            tag: tag.into(),
        })
    }

    /// Copy plaintext only after a complete read and tag verification.
    /// On any refusal or failed completion, `output` is unchanged.
    pub fn read_slot(
        &mut self,
        slot: u64,
        page: u64,
        proof: &SlotProof,
        output: &mut [u8; PAGE_BYTES],
    ) -> Result<(), Error> {
        let offset = self.slot_offset(slot)?;
        check_disk(self.disk.disk(), &self.expected)?;
        if proof.generation != self.header.generation() || proof.slot != slot || proof.page != page
        {
            return Err(Error::Authentication);
        }
        let mut data = Zeroizing::new(vec![0; PAGE_BYTES]);
        self.read_store(offset, &mut data)?;
        self.cipher
            .decrypt_inout_detached(
                (&nonce(proof.counter)).into(),
                &self.aad(slot, page),
                data.as_mut_slice().into(),
                (&proof.tag).into(),
            )
            .map_err(|_| Error::Authentication)?;
        output.copy_from_slice(&data);
        Ok(())
    }

    fn slot_offset(&self, slot: u64) -> Result<u64, Error> {
        self.header
            .layout()
            .slot_offset(slot)
            .ok_or(Error::Slot(slot))
    }

    fn read_store(&mut self, offset: u64, mut output: &mut [u8]) -> Result<(), Error> {
        let block = u64::from(self.expected.block_size);
        for run in self.placement.runs(offset, output.len() as u64) {
            let (chunk, rest) = output.split_at_mut(run.bytes as usize);
            let data = read_blocks(
                &mut self.disk,
                &self.expected,
                run.offset / block,
                chunk.len(),
            )?;
            chunk.copy_from_slice(&data);
            output = rest;
        }
        debug_assert!(output.is_empty());
        Ok(())
    }

    fn write_store(&mut self, offset: u64, mut data: &[u8]) -> Result<(), Error> {
        let block = u64::from(self.expected.block_size);
        let features = self.expected.features;
        for run in self.placement.runs(offset, data.len() as u64) {
            let (chunk, rest) = data.split_at(run.bytes as usize);
            check_disk(self.disk.disk(), &self.expected)?;
            Request::Write {
                lba: run.offset / block,
                data: chunk.to_vec(),
                durable: features.fua,
            }
            .run(&mut self.disk)
            .map_err(Error::Device)?;
            check_disk(self.disk.disk(), &self.expected)?;
            data = rest;
        }
        debug_assert!(data.is_empty());
        if features.write_cache && !features.fua {
            // All fragments have completed. One flush covers the entire write.
            Request::Flush.run(&mut self.disk).map_err(Error::Device)?;
            check_disk(self.disk.disk(), &self.expected)?;
        }
        Ok(())
    }

    fn aad(&self, slot: u64, page: u64) -> [u8; 48] {
        let mut bytes = [0; 48];
        bytes[..8].copy_from_slice(MAGIC);
        bytes[8..24].copy_from_slice(&self.header.store_uuid().0);
        bytes[24..32].copy_from_slice(&self.header.generation().to_le_bytes());
        bytes[32..40].copy_from_slice(&slot.to_le_bytes());
        bytes[40..].copy_from_slice(&page.to_le_bytes());
        bytes
    }
}

fn check_disk(identity: &Identity, expected: &Identity) -> Result<(), Error> {
    if geometry(identity) != geometry(expected) {
        return Err(Error::GeometryChanged);
    }
    if identity.read_only {
        return Err(Error::Device(BlockError::ReadOnly));
    }
    if identity != expected {
        return Err(Error::DeviceChanged);
    }
    Ok(())
}

fn read_blocks(
    disk: &mut PartitionDevice<impl ExclusiveDisk>,
    expected: &Identity,
    lba: u64,
    bytes: usize,
) -> Result<Vec<u8>, Error> {
    check_disk(disk.disk(), expected)?;
    let data = Request::Read { lba, bytes }
        .run(disk)
        .map_err(Error::Device)?
        .ok_or(Error::Device(BlockError::Protocol(
            "swap read returned no bytes",
        )))?;
    check_disk(disk.disk(), expected)?;
    Ok(data)
}

fn nonce(counter: u64) -> [u8; 12] {
    let mut bytes = [0; 12];
    bytes[4..].copy_from_slice(&counter.to_le_bytes());
    bytes
}

#[derive(Debug)]
pub enum Error {
    Partition(partition::Error),
    DataTable(gpt::GptError),
    DataPartition,
    Volume(volume::Error),
    StoreIdentity,
    Header(HeaderError),
    Device(BlockError),
    GeometryChanged,
    DeviceChanged,
    Slot(u64),
    SlotHeld(u64),
    Authentication,
    NonceExhausted,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Partition(error) => error.fmt(f),
            Self::DataTable(error) => error.fmt(f),
            Self::DataPartition => {
                f.write_str("swap DATA partition is missing, ambiguous, or invalid")
            }
            Self::Volume(error) => error.fmt(f),
            Self::StoreIdentity => f.write_str("FAT32 swap store UUID does not match"),
            Self::Header(error) => error.fmt(f),
            Self::Device(error) => error.fmt(f),
            Self::GeometryChanged => f.write_str("swap disk geometry changed"),
            Self::DeviceChanged => f.write_str("swap disk identity or I/O features changed"),
            Self::Slot(slot) => write!(f, "swap slot {slot} is out of range"),
            Self::SlotHeld(slot) => write!(f, "swap slot {slot} is held by a nonresident frame"),
            Self::Authentication => f.write_str("swap page authentication failed"),
            Self::NonceExhausted => f.write_str("swap write sequence exhausted"),
        }
    }
}

#[cfg(test)]
mod tests;
