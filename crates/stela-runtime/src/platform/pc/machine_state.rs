//! A machine's state kept in the tail of its Apple region instead of on a
//! partition of its own.
//!
//! On a Mac the NOR flash (NVRAM), the SEP's storage and Stela's boot log
//! live off the internal SSD, so the guest must never see or overwrite them.
//! When `--nor` names the same partition as `--storage`, the partition's
//! last [`MACHINE_STATE_BYTES`] or more, starting at a [`ALIGNMENT`]
//! boundary from the partition's start, hold that state in the machine-state
//! partition's own layout: the NOR image at 0, the SEP's area after it, the
//! boot log at [`boot_log::OFFSET`](super::boot_log::OFFSET) and the board
//! devices' flash at [`board_flash::OFFSET`](super::board_flash::OFFSET). The
//! guest's ANS2 namespace is everything before it. Both are windows of one
//! shared device ([`Shared::window`]), so no write of one reaches the other.
use alloc::boxed::Box;
use core::fmt;
use stela_drivers::block::{BlockDevice, BlockError, Shared, Window};

/// The machine state kept at the end of the partition, before alignment.
pub const MACHINE_STATE_BYTES: u64 = 16 << 20;
// The tail always holds the whole layout, the boot log, board flash and RTC
// anchor included.
const _: () = assert!(MACHINE_STATE_BYTES >= super::rtc_anchor::PARTITION_BYTES);
/// Where the machine state starts: a multiple of this from the partition's
/// start, which also aligns it to every block size that divides it.
pub const ALIGNMENT: u64 = 1 << 20;
/// The smallest partition that holds a namespace of [`ALIGNMENT`] bytes and
/// the machine state.
pub const MINIMUM_BYTES: u64 = MACHINE_STATE_BYTES + ALIGNMENT;

/// A partition split into the guest's namespace and the machine state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Split {
    /// The guest's namespace, from the partition's start; the machine state
    /// starts here.
    pub namespace_bytes: u64,
    /// The machine state, to the partition's end: at least
    /// [`MACHINE_STATE_BYTES`] and less than that plus [`ALIGNMENT`].
    pub state_bytes: u64,
}

impl Split {
    /// The machine state's byte offset in the partition.
    pub fn state_offset(&self) -> u64 {
        self.namespace_bytes
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SplitError {
    /// The partition cannot hold a namespace and the machine state.
    TooSmall { partition_bytes: u64 },
    /// The block size does not divide [`ALIGNMENT`].
    Blocks { block_size: u64 },
    /// The partition's windows could not be made.
    Device(BlockError),
}

impl fmt::Display for SplitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooSmall { partition_bytes } => write!(
                f,
                "the partition holds {partition_bytes:#x} bytes; the guest's namespace and the \
                 {MACHINE_STATE_BYTES:#x}-byte machine state in its last bytes need at least \
                 {MINIMUM_BYTES:#x}"
            ),
            Self::Blocks { block_size } => write!(
                f,
                "{block_size}-byte blocks do not divide the machine state's {ALIGNMENT:#x}-byte \
                 alignment"
            ),
            Self::Device(error) => write!(f, "{error}"),
        }
    }
}

/// Where a partition of `partition_bytes` keeps the machine state in its
/// tail.
pub fn split(partition_bytes: u64) -> Result<Split, SplitError> {
    if partition_bytes < MINIMUM_BYTES {
        return Err(SplitError::TooSmall { partition_bytes });
    }
    let namespace_bytes = (partition_bytes - MACHINE_STATE_BYTES) / ALIGNMENT * ALIGNMENT;
    Ok(Split {
        namespace_bytes,
        state_bytes: partition_bytes - namespace_bytes,
    })
}

/// The partition `device` as the guest's namespace and the machine state,
/// two windows of one shared device.
pub fn split_device(device: Box<dyn BlockDevice>) -> Result<(Window, Window, Split), SplitError> {
    let block = u64::from(device.identity().block_size);
    if block == 0 || !ALIGNMENT.is_multiple_of(block) {
        return Err(SplitError::Blocks { block_size: block });
    }
    let layout = split(device.identity().bytes())?;
    let disk = Shared::new(device);
    let first = layout.namespace_bytes / block;
    let namespace = disk.window(0, first).map_err(SplitError::Device)?;
    let state = disk
        .window(first, layout.state_bytes / block)
        .map_err(SplitError::Device)?;
    Ok((namespace, state, layout))
}

#[cfg(test)]
mod tests {
    use super::*;
    use stela_drivers::block::RamDisk;

    #[test]
    fn the_state_takes_the_aligned_tail_and_the_namespace_the_rest() {
        let mib = 1 << 20;
        // 64 GB is not a whole number of MiB: the tail grows to the boundary.
        let layout = split(64_000_000_000).unwrap();
        assert_eq!(layout.namespace_bytes % ALIGNMENT, 0);
        assert_eq!(layout.namespace_bytes + layout.state_bytes, 64_000_000_000);
        assert!(layout.state_bytes >= MACHINE_STATE_BYTES);
        assert!(layout.state_bytes < MACHINE_STATE_BYTES + ALIGNMENT);
        assert_eq!(layout.state_offset(), layout.namespace_bytes);
        // A whole number of MiB keeps exactly the 16 MiB.
        assert_eq!(
            split(100 * mib).unwrap(),
            Split {
                namespace_bytes: 84 * mib,
                state_bytes: 16 * mib
            }
        );
        assert_eq!(
            split(MINIMUM_BYTES).unwrap(),
            Split {
                namespace_bytes: mib,
                state_bytes: 16 * mib
            }
        );
        let error = split(MINIMUM_BYTES - 1).unwrap_err();
        assert_eq!(
            error,
            SplitError::TooSmall {
                partition_bytes: MINIMUM_BYTES - 1
            }
        );
        assert!(error.to_string().contains("machine state"), "{error}");
    }

    #[test]
    fn the_windows_share_one_device_and_the_namespace_never_reaches_the_state() {
        let blocks = (20 << 20) / 512 + 3;
        let (mut namespace, mut state, layout) =
            split_device(Box::new(RamDisk::new("s", blocks, 512))).unwrap();
        assert_eq!(layout.namespace_bytes, 4 << 20);
        assert_eq!(namespace.identity().blocks, (4 << 20) / 512);
        assert_eq!(state.identity().blocks, blocks - (4 << 20) / 512);
        assert_eq!(state.identity().bytes(), layout.state_bytes);
        state.write(0, &[0x5a; 512]).unwrap();
        let last = namespace.identity().blocks;
        let mut block = [0; 512];
        assert!(namespace.read(last, &mut block).is_err());
        assert!(namespace.write(last, &[1; 512]).is_err());
        namespace.read(last - 1, &mut block).unwrap();
        assert_eq!(block, [0; 512]);
        state.read(0, &mut block).unwrap();
        assert_eq!(block, [0x5a; 512]);
        // A partition too small, or blocks that do not divide the alignment,
        // are refused.
        let small = (MINIMUM_BYTES - 4096) / 512;
        assert!(matches!(
            split_device(Box::new(RamDisk::new("s", small, 512))),
            Err(SplitError::TooSmall { .. })
        ));
        assert!(matches!(
            split_device(Box::new(RamDisk::new("s", 8, 3 << 20))),
            Err(SplitError::Blocks { .. })
        ));
    }
}
