//! The Metal device's pipeline store: `stela/store/pipeline.bin`, a file `stela setup`
//! preallocates beside the translation packs.
//!
//! Before boot services end, [`read`] reads the stored pipelines from the
//! store's volume (the first with `store.cfg`). [`install`] hands them to
//! the Metal device, which seeds its compiled pipelines' cache from them
//! when they are its compiler build's and GPU's. When the guest restarts or
//! powers off, [`write`] rewrites the file in place with the device's
//! newest snapshot, over the store's partition, before the translation pack
//! is written. Every failure leaves the boot running: its pipelines compile
//! again.
use super::translations;
use crate::{firmware, handoff::Region, output, paging::PAGE};
use alloc::format;
use stela_drivers::block::BlockDevice;
use stela_drivers::gpu::metal::native::pipeline_store::{self, HEADER_BYTES, MAX_BYTES};
use stela_formats::payload::pieces::Extent;
use uefi::proto::media::file::{File as _, FileInfo, FileMode};

/// The handoff file holding the file's size and the stored pipelines.
pub const FILE: &str = "pipelines";
/// The file's 8.3 name in the store's directory.
const NAME: &str = "pipeline.bin";
/// The placement record at the start of the handoff file: magic and the
/// file's size, the store's bound; the stored bytes follow.
const PLACEMENT_MAGIC: [u8; 8] = *b"STLGPLAC";
const PLACEMENT_BYTES: usize = 16;
/// The bytes a write sends to the device at a time.
const TRANSFER_BYTES: usize = 1 << 20;

/// The pipelines the store's volume holds, as the handoff file's pages:
/// run before boot services end. `None` without a store or its file.
pub fn read() -> Option<Region> {
    translations::with_configured(|_, root, layout, _| {
        let path = translations::uefi_store_path(layout, NAME);
        let Some(mut file) = translations::open(root, &path, FileMode::Read) else {
            output::arguments(format_args!(
                "PIPELINES no {path}: compiled pipelines compile every boot; `stela setup \
                 repair` prepares the file\n"
            ));
            return None;
        };
        let size = file
            .get_boxed_info::<FileInfo>()
            .ok()?
            .file_size()
            .min(MAX_BYTES);
        let mut header = [0; HEADER_BYTES];
        let stored = if size < HEADER_BYTES as u64 {
            0
        } else {
            translations::read_exact(&mut file, &mut header)?;
            // A store this build cannot read keeps its header, so the device
            // reports why it is ignored; a blank file holds nothing.
            match pipeline_store::stored_bytes(&header) {
                Ok(bytes) if bytes <= size => bytes as usize,
                Err(pipeline_store::Rejection::Blank) => 0,
                _ => HEADER_BYTES,
            }
        };
        let bytes = PLACEMENT_BYTES + stored;
        let pages = bytes.div_ceil(PAGE);
        let physical = firmware::pages(pages).ok()?;
        // SAFETY: firmware returned this exclusive allocation of `bytes` or
        // more. Initialize it before taking a mutable slice.
        let region = unsafe {
            let start = physical as *mut u8;
            start.write_bytes(0, bytes);
            core::slice::from_raw_parts_mut(start, bytes)
        };
        region[..8].copy_from_slice(&PLACEMENT_MAGIC);
        region[8..16].copy_from_slice(&size.to_le_bytes());
        if stored != 0 {
            let at = PLACEMENT_BYTES;
            region[at..at + HEADER_BYTES].copy_from_slice(&header);
            if translations::read_exact(&mut file, &mut region[at + HEADER_BYTES..]).is_none() {
                // A short file seeds nothing; the device reports its length.
                region[at + HEADER_BYTES..].fill(0);
            }
        }
        output::arguments(format_args!(
            "PIPELINES {path} file={size} stored={stored}\n"
        ));
        Some(Region {
            physical: physical as u64,
            bytes: bytes as u64,
        })
    })
}

/// Give the Metal device the stored pipelines of the handoff file
/// `contents`, before any GPU opens.
pub fn install(contents: &'static [u8]) {
    if contents.len() < PLACEMENT_BYTES || contents[..8] != PLACEMENT_MAGIC {
        return;
    }
    let size = u64::from_le_bytes(contents[8..16].try_into().expect("a size"));
    pipeline_store::provide(
        &contents[PLACEMENT_BYTES..],
        stela_mesa::COMPILER_SHA256.as_bytes(),
        size,
    );
}

/// At a quiescent point (the guest restarts or powers off), before the
/// translation pack is written: rewrite the file with the device's newest
/// snapshot, when it added pipelines this boot.
pub fn write() {
    let Some(sealed) = pipeline_store::take() else {
        return;
    };
    let bytes = HEADER_BYTES + sealed.payload.len();
    let started = output::now();
    let written = translations::with_store_file(NAME, |device, extents| {
        write_over(device, extents, &[&sealed.header[..], &sealed.payload[..]])
    })
    .and_then(|written| written);
    match written {
        Ok(()) => output::arguments(format_args!(
            "PIPELINES written bytes={bytes} ms={}\n",
            output::millis_since(started)
        )),
        Err(error) => output::arguments(format_args!("PIPELINES not written: {error}\n")),
    }
}

/// Write `parts` in order over the file of device extents `extents`, whole
/// blocks at a time with the last padded with zeros, then make it durable.
/// The file is never grown: parts longer than it are refused first.
fn write_over(
    device: &mut dyn BlockDevice,
    extents: &[Extent],
    parts: &[&[u8]],
) -> Result<(), alloc::string::String> {
    let block = u64::from(device.identity().block_size);
    let file: u64 = extents.iter().map(|extent| extent.bytes).sum();
    let total: u64 = parts.iter().map(|part| part.len() as u64).sum();
    if total > file {
        return Err(format!("the file holds {file} bytes, fewer than {total}"));
    }
    if block == 0 || extents.iter().any(|extent| extent.offset % block != 0) {
        return Err(format!("an extent is not on a {block}-byte block"));
    }
    let mut parts = parts.iter().copied();
    let mut part: &[u8] = &[];
    let mut left = total;
    let mut buffer = alloc::vec::Vec::with_capacity(TRANSFER_BYTES);
    for extent in extents {
        let mut written = 0;
        while written < extent.bytes && left > 0 {
            let take = (extent.bytes - written)
                .min(TRANSFER_BYTES as u64)
                .min(left);
            buffer.clear();
            while buffer.len() < take as usize {
                if part.is_empty() {
                    part = parts.next().expect("the parts hold `total` bytes");
                    continue;
                }
                let (now, later) = part.split_at(part.len().min(take as usize - buffer.len()));
                buffer.extend_from_slice(now);
                part = later;
            }
            buffer.resize(buffer.len().next_multiple_of(block as usize), 0);
            device
                .write((extent.offset + written) / block, &buffer)
                .map_err(|error| format!("{error:?}"))?;
            written += take;
            left -= take;
        }
    }
    device.flush().map_err(|error| format!("{error:?}"))
}
