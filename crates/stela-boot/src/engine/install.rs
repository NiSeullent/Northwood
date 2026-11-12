//! The install medium (`--install MANIFEST --install-source GUID`): the
//! IPSW as `ipsw.NNN` pieces on a FAT32 partition of the USB, beside its
//! `payload.json`, which `stela setup` writes.
//!
//! Before boot services end, [`check_pieces`] confirms through the
//! firmware's own FAT driver that every piece is there with its size. After
//! them, [`open`] finds each piece's clusters through Stela's FAT32 reader
//! on the partition the storage drivers opened by its GUID, and joins them
//! into one IPSW whose every block is checked against the piece hashes the
//! manifest names.
use super::bundle::{self, Bytes};
use crate::Result;
use crate::output;
use alloc::{boxed::Box, format, vec::Vec};
use stela_contracts::refuse;
use stela_drivers::block::BlockDevice;
use uefi::proto::media::file::Directory;
use {
    stela_formats::payload::Layout, stela_formats::payload::PayloadManifest,
    stela_formats::payload::piece_name, stela_formats::payload::pieces::Joined,
    stela_runtime::pc::fat32::Fat32,
};

/// The joined install medium the restore host reads.
pub(super) type Medium = Joined<Bytes<Box<dyn BlockDevice>>>;

/// Check, through the firmware's FAT driver on `root`, that every piece
/// `manifest` lists is in `directory` with its exact size. Each missing or
/// short piece is reported by name.
pub fn check_pieces(
    root: &mut Directory,
    directory: &str,
    manifest: &PayloadManifest,
    has_source: bool,
) -> Result<()> {
    let pieces = match (&manifest.layout, has_source) {
        (Layout::Pieces { pieces, .. }, true) => pieces,
        // Streamed after boot services end, over the mount's own network.
        (Layout::Remote { .. }, false) => return Ok(()),
        (Layout::Pieces { .. }, false) => {
            return Err(refuse!("a local install medium requires --install-source"));
        }
        (Layout::Remote { .. }, true) => {
            return Err(refuse!(
                "a network install inventory cannot use --install-source"
            ));
        }
        _ => {
            return Err(refuse!(
                "--install names a payload that is not an install medium"
            ));
        }
    };
    let mut complete = true;
    for (index, piece) in pieces.iter().enumerate() {
        let path = format!("{directory}\\{}", piece_name(index));
        match bundle::file_size(root, &path) {
            Ok(bytes) if bytes == piece.bytes => {}
            Ok(bytes) => {
                output::arguments(format_args!(
                    "INSTALL MEDIUM {path} holds {bytes} bytes, not {}\n",
                    piece.bytes
                ));
                complete = false;
            }
            Err(_) => {
                output::arguments(format_args!("INSTALL MEDIUM {path} is missing\n"));
                complete = false;
            }
        }
    }
    if !complete {
        return Err(refuse!(
            "the install medium is incomplete; run stela setup again to finish it"
        ));
    }
    output::arguments(format_args!(
        "INSTALL MEDIUM {directory}: {} pieces, {} bytes, build {}\n",
        pieces.len(),
        manifest.source.bytes,
        manifest.product_build
    ));
    Ok(())
}

/// The install medium on `partition`: each piece of `manifest` found
/// through FAT32 in `directory` (the manifest's own, as `stela.cfg` names
/// it) and joined, checked against `hashes`, the piece hash file.
pub(super) fn open(
    partition: Box<dyn BlockDevice>,
    manifest: &PayloadManifest,
    directory: &str,
    hashes: &'static [u8],
) -> Result<Medium> {
    let Layout::Pieces { pieces, .. } = &manifest.layout else {
        return Err(refuse!(
            "--install names a payload that is not an install medium"
        ));
    };
    let identity = partition.identity();
    let sector_bytes = identity.block_size;
    let features = identity.features;
    let read_limit = features
        .max_read_transfer
        .map_or(features.max_transfer, |limit| limit.get());
    let mut volume = Fat32::open(Bytes::new(partition))
        .map_err(|error| refuse!("install medium: {error}", %error))?;
    let directory = directory.replace('\\', "/");
    let mut extents = Vec::with_capacity(pieces.len());
    for index in 0..pieces.len() {
        let file = volume
            .file(&format!("{directory}/{}", piece_name(index)))
            .map_err(|error| refuse!("install medium: {error}", %error))?;
        extents.push(file.extents);
    }
    Joined::from_static_hashes(volume.into_inner(), manifest, extents, hashes)
        .map(|joined| joined.with_two_block_reads(read_limit, sector_bytes))
        .map_err(|error| refuse!("install medium: {error}", %error))
}
