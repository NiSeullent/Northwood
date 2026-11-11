// SPDX-License-Identifier: GPL-2.0-or-later
//! Read-only discovery of a swap candidate during the owned disk probe.
//! The engine still reports resident DRAM while paging is unavailable.

use super::bundle::Bytes;
use crate::output;
use alloc::{boxed::Box, string::String};
use northwood_config::{Swap, SwapLocation};
use stela_boot_core::host_memory::Profile;
use stela_drivers::block::{BlockDevice, BlockError, Identity};
use stela_formats::payload::ReadAt;
use stela_runtime::pc::{
    gpt::{GptError, Guid, PartitionDevice},
    storage::Target,
    swap::{Header, partition, volume},
};

pub(crate) struct Probe {
    location: SwapLocation,
    partition: Option<Guid>,
    target_serial: Option<String>,
    boot_esp: Option<volume::BootEsp>,
    volume_max_bytes: u64,
}

impl Probe {
    pub(super) fn for_boot(
        profile: Profile,
        config: Option<&Swap>,
        target: Option<&Target>,
        boot_esp: Option<volume::BootEsp>,
    ) -> Option<Self> {
        if profile != Profile::Paged {
            return None;
        }
        let default = Swap::default();
        let config = config.unwrap_or(&default);
        if !config.enabled || config.location == SwapLocation::None {
            return None;
        }
        let partition = if config.partition.is_empty() {
            None
        } else if let Some(guid) = Guid::parse(&config.partition) {
            Some(guid)
        } else {
            output::arguments(format_args!("HOST SWAP refused invalid partition UUID\n"));
            return None;
        };
        let target_serial = target.map(|target| target.serial.clone());
        let probe = Self {
            location: config.location,
            partition,
            target_serial,
            boot_esp,
            volume_max_bytes: config.volume_max_bytes,
        };
        if !probe.wants_partition() && probe.boot_esp().is_none() {
            return None;
        }
        Some(probe)
    }

    fn wants_partition(&self) -> bool {
        matches!(self.location, SwapLocation::Auto | SwapLocation::Partition)
            && (self.partition.is_some() || self.target_serial.is_some())
    }

    fn boot_esp(&self) -> Option<volume::BootEsp> {
        match self.location {
            SwapLocation::Auto | SwapLocation::VolumeFile => self.boot_esp,
            SwapLocation::Partition | SwapLocation::None => None,
        }
    }

    /// Find a candidate only on the install target's disk, unless setup
    /// named a unique partition explicitly. Do this before disks are handed
    /// out to any service or guest.
    pub(super) fn inspect_partition(&self, disks: &mut [Box<dyn BlockDevice>]) -> bool {
        if !self.wants_partition() {
            return false;
        }
        let matches_target = |disk: &dyn BlockDevice| {
            self.partition.is_some()
                || self.target_serial.as_deref() == Some(disk.identity().serial.as_str())
        };
        if self.partition.is_none()
            && disks
                .iter()
                .filter(|disk| matches_target(disk.as_ref()))
                .take(2)
                .count()
                > 1
        {
            output::arguments(format_args!("HOST SWAP refused ambiguous target disk\n"));
            return false;
        }
        let mut candidate = None;
        for disk in disks
            .iter_mut()
            .filter(|disk| matches_target(disk.as_ref()))
        {
            match partition::inspect(&mut **disk, self.partition) {
                Ok((_, header)) if candidate.is_none() => candidate = Some(header),
                Ok(_) => {
                    output::arguments(format_args!("HOST SWAP refused ambiguous GPT store\n"));
                    return false;
                }
                Err(partition::Error::Missing | partition::Error::Table(GptError::NoTable)) => {}
                Err(partition::Error::Table(_)) if self.partition.is_some() => {}
                Err(error) => {
                    output::arguments(format_args!("HOST SWAP GPT candidate refused: {error}\n"));
                    return false;
                }
            }
        }
        if let Some(header) = candidate {
            report("partition", header);
            true
        } else {
            false
        }
    }

    /// Read through a partition-bounded borrow. The normal probe still owns
    /// every disk after the scan and can hand it off as configured.
    pub(super) fn inspect_volume(
        &self,
        disks: &mut [Box<dyn BlockDevice>],
        ticks_per_microsecond: u64,
    ) {
        let Some(esp) = self.boot_esp() else {
            output::arguments(format_args!("HOST SWAP no verified store; mode=resident\n"));
            return;
        };
        let (disk, partition) = match volume::boot_data(disks, esp) {
            Ok(found) => found,
            Err(
                error @ (volume::BootDataError::MissingEsp | volume::BootDataError::MissingData),
            ) => {
                output::arguments(format_args!(
                    "HOST SWAP no verified store; mode=resident reason={error}\n"
                ));
                return;
            }
            Err(error) => {
                output::arguments(format_args!(
                    "HOST SWAP volume candidate refused: {error}\n"
                ));
                return;
            }
        };
        let device = PartitionDevice::new(ReadOnlyDisk(&mut *disks[disk]), partition);
        let block_bytes = u64::from(device.identity().block_size);
        let reader = VolumeReader {
            inner: Bytes::new(device),
            start: output::now(),
            budget: ticks_per_microsecond.saturating_mul(4_000_000),
        };
        match volume::inspect(reader, block_bytes, self.volume_max_bytes) {
            Ok(metadata) => report("volume", metadata.header),
            Err(error) => output::arguments(format_args!(
                "HOST SWAP volume candidate refused: {error}\n"
            )),
        }
    }
}

struct ReadOnlyDisk<'a>(&'a mut dyn BlockDevice);

impl BlockDevice for ReadOnlyDisk<'_> {
    fn identity(&self) -> &Identity {
        self.0.identity()
    }

    fn read(&mut self, lba: u64, buffer: &mut [u8]) -> Result<(), BlockError> {
        self.0.read(lba, buffer)
    }

    fn write(&mut self, _: u64, _: &[u8]) -> Result<(), BlockError> {
        Err(BlockError::ReadOnly)
    }
}

/// Bound the optional volume scan between block reads. The device retains
/// ownership of each individual read until completion or timeout.
struct VolumeReader<'a> {
    inner: Bytes<PartitionDevice<ReadOnlyDisk<'a>>>,
    start: u64,
    budget: u64,
}

impl ReadAt for VolumeReader<'_> {
    type Error = BlockError;

    fn len(&self) -> u64 {
        self.inner.len()
    }

    fn read_at(&mut self, offset: u64, bytes: &mut [u8]) -> Result<(), Self::Error> {
        if output::now().wrapping_sub(self.start) >= self.budget {
            return Err(BlockError::Timeout);
        }
        self.inner.read_at(offset, bytes)
    }
}

fn report(location: &str, header: Header) {
    output::arguments(format_args!(
        "HOST SWAP candidate={location} uuid={} state={:?} slots={} bytes={} mode=resident\n",
        header.store_uuid(),
        header.state(),
        header.layout().slot_count(),
        header.layout().used_bytes(),
    ));
}
