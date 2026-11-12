//! Download straight into FAT32 pieces; no complete IPSW is cached on the PC.
use super::*;
use std::io;

const BLOCK: usize = HASH_BLOCK_BYTES as usize;
const WINDOW: usize = 16 * BLOCK;

pub(in super::super) struct Download {
    pub image: restore::Image,
    pub bytes: u64,
    etag: String,
}

impl Download {
    pub(in super::super) fn prepare(
        image: restore::Image,
        model: &str,
        boot: &Path,
    ) -> Result<Self> {
        let (download, source) = Self::open_source(image)?;
        restore::payload::trim(source, model, boot)?;
        ensure!(
            restore::payload::identity(boot, model)?.1 == download.image.build,
            "the catalog download contains a different macOS build"
        );
        Ok(download)
    }

    pub(in super::super) fn open(image: restore::Image) -> Result<Self> {
        Self::open_source(image).map(|(download, _)| download)
    }

    fn open_source(image: restore::Image) -> Result<(Self, restore::remote::Remote)> {
        ensure!(
            image.sha1.is_some(),
            "the catalog provides no checksum for this full installer"
        );
        let source = restore::open_image(&image)?;
        let download = Self {
            bytes: source.size(),
            etag: source.etag().into(),
            image,
        };
        Ok((download, source))
    }

    pub(in super::super) fn write_in(
        &self,
        volume: &Path,
        layout: VolumeLayout,
        model: &str,
        piece_bytes: u64,
    ) -> Result<Written> {
        let source = restore::open_image(&self.image)?;
        ensure!(
            source.size() == self.bytes && source.etag() == self.etag,
            "the download changed after boot-file preparation; run setup again"
        );
        write_from(
            source,
            volume,
            layout,
            model,
            &self.image,
            self.bytes,
            piece_bytes,
        )
    }

    pub(in super::super) fn write_fresh_in(
        &self,
        volume: &Path,
        model: &str,
        piece_bytes: u64,
    ) -> Result<Written> {
        publish_fresh(volume, &self.image.build, |scratch| {
            self.write_in(scratch, VolumeLayout::Legacy, model, piece_bytes)
        })
    }
}

/// The standalone download never occupies the active install path until
/// both the catalog digest and FAT32 readback have passed.
fn publish_fresh(
    volume: &Path,
    build: &str,
    write: impl FnOnce(&Path) -> Result<Written>,
) -> Result<Written> {
    let destination = directory(volume, build);
    ensure!(
        !destination.try_exists()?,
        "{} already exists",
        destination.display()
    );
    let scratch = tempfile::Builder::new()
        .prefix(".stela-install-")
        .tempdir_in(volume)?;
    let written = write(scratch.path())?;
    fs::create_dir_all(
        destination
            .parent()
            .context("install directory has no parent")?,
    )?;
    ensure!(
        !destination.try_exists()?,
        "{} already exists",
        destination.display()
    );
    fs::rename(directory(scratch.path(), build), destination)?;
    Ok(written)
}

fn write_from(
    mut source: impl Read,
    volume: &Path,
    layout: VolumeLayout,
    model: &str,
    image: &restore::Image,
    total: u64,
    piece_bytes: u64,
) -> Result<Written> {
    ensure!(
        piece_bytes > 0
            && piece_bytes <= stela_formats::payload::MAX_SLICE_BYTES
            && piece_bytes.is_multiple_of(HASH_BLOCK_BYTES),
        "invalid FAT32 piece size"
    );
    let sizes = piece_sizes(total, piece_bytes);
    ensure!(
        !sizes.is_empty() && sizes.len() <= MAX_PIECES,
        "invalid download size or piece size"
    );
    let directory = directory_in(volume, layout, &image.build);
    fs::create_dir_all(&directory)?;
    let manifest_path = directory.join(MANIFEST_FILE);
    if manifest_path.exists() {
        fs::remove_file(&manifest_path)?;
    }
    let mut pieces = Vec::new();
    let mut piece_blocks = Vec::new();
    let mut whole_sha256 = Sha256::new();
    let mut whole_sha1 = Sha1::new();
    let progress = crate::progress::Phase::bytes("Downloading macOS onto USB", total);
    for (index, &size) in sizes.iter().enumerate() {
        let mut file = fs::File::create(partial_path(&directory, index))?;
        let mut hashes = Hashes::default();
        stream(&mut source, size, |bytes| {
            file.write_all(bytes)?;
            hashes.update(bytes);
            whole_sha256.update(bytes);
            whole_sha1.update(bytes);
            progress.advance(bytes.len() as u64);
            Ok(())
        })?;
        file.sync_all()?;
        let (sha256, blocks) = hashes.finish();
        pieces.push(Piece {
            bytes: size,
            sha256,
        });
        piece_blocks.extend(blocks);
    }
    ensure!(
        source.read(&mut [0])? == 0,
        "download exceeds the expected image size"
    );
    let sha256 = hex::encode(whole_sha256.finalize());
    let sha1 = hex::encode(whole_sha1.finalize());
    ensure!(
        image
            .sha1
            .as_ref()
            .is_some_and(|expected| expected.eq_ignore_ascii_case(&sha1)),
        "download differs from the catalog digest"
    );
    progress.finish();
    let mut completing = false;
    let inventory = (|| {
        let joined = Joined::new(&directory, piece_bytes, total, &piece_blocks)?;
        restore::payload::inventory_from(
            joined,
            restore::payload::VerifiedSource {
                image: Path::new(&image.url)
                    .file_name()
                    .map(Path::new)
                    .context("image URL has no file name")?,
                model,
                digest: (total, sha256),
            },
            |joined| {
                completing = true;
                joined.complete(&pieces)
            },
        )
    })();
    if !completing {
        // If inventory failed early, retain the old ordered verification and
        // promotion. A media failure still takes precedence and leaves the
        // same verified prefix promoted before that failure.
        let progress = crate::progress::Phase::bytes("Verifying USB", total);
        promote(&directory, &pieces, |_, piece, partial| {
            ensure!(
                hash_file(partial, piece.bytes, &progress)? == piece.sha256,
                "USB data does not read back as written"
            );
            Ok(())
        })?;
        progress.finish();
    }
    let mut inventory = inventory?;
    ensure!(
        inventory.manifest.product_build == image.build,
        "download contains a different macOS build"
    );
    inventory.manifest.source.catalog_sha1 = Some(sha1);
    inventory.manifest.layout = Layout::Pieces {
        piece_bytes,
        pieces,
        block_hashes: BlockHashes {
            file: PIECE_HASHES_FILE.into(),
            block_bytes: HASH_BLOCK_BYTES,
            sha256: stela_formats::sha256(&piece_blocks),
        },
    };
    inventory.manifest.validate().map_err(anyhow::Error::msg)?;
    efi::write_atomic(&directory.join(PIECE_HASHES_FILE), &piece_blocks, false)?;
    efi::write_atomic(&directory.join(BLOCK_HASHES_FILE), &inventory.blocks, false)?;
    let json = inventory.manifest.to_json();
    efi::write_atomic(&manifest_path, json.as_bytes(), false)?;
    Ok(Written {
        wrote: sizes.len(),
        kept: 0,
        manifest_sha256: stela_formats::sha256(json.as_bytes()),
    })
}

fn partial_path(directory: &Path, index: usize) -> PathBuf {
    directory.join(format!("{}.partial", piece_name(index)))
}

/// Verify and promote each piece in the original order, leaving later
/// partials alone on failure.
fn promote(
    directory: &Path,
    pieces: &[Piece],
    mut verify: impl FnMut(usize, &Piece, &Path) -> Result<()>,
) -> Result<()> {
    for (index, piece) in pieces.iter().enumerate() {
        let path = directory.join(piece_name(index));
        let partial = partial_path(directory, index);
        verify(index, piece, &partial)?;
        if path.exists() {
            fs::remove_file(&path)?;
        }
        fs::rename(partial, path)?;
    }
    Ok(())
}

/// ZIP's seekable view of the synced partials. The bitmap records readback
/// coverage, but every window fetched from disk is verified again.
struct Joined<'a> {
    directory: &'a Path,
    piece_bytes: u64,
    total: u64,
    position: u64,
    current: Option<(usize, fs::File)>,
    hashes: &'a [[u8; 32]],
    checked: Vec<u8>,
    checked_bytes: u64,
    buffer: Vec<u8>,
    window: Option<(u64, usize)>,
}

impl<'a> Joined<'a> {
    fn new(directory: &'a Path, piece_bytes: u64, total: u64, hashes: &'a [u8]) -> Result<Self> {
        let (hashes, tail) = hashes.as_chunks::<32>();
        ensure!(
            tail.is_empty() && hashes.len() as u64 == total.div_ceil(HASH_BLOCK_BYTES),
            "USB readback has the wrong number of block digests"
        );
        Ok(Self {
            directory,
            piece_bytes,
            total,
            position: 0,
            current: None,
            hashes,
            checked: vec![0; hashes.len().div_ceil(8)],
            checked_bytes: 0,
            buffer: vec![0; WINDOW],
            window: None,
        })
    }

    fn checked(&self, block: usize) -> bool {
        self.checked[block / 8] & (1 << (block % 8)) != 0
    }

    fn load(&mut self, start: u64, bytes: usize) -> io::Result<()> {
        self.window = None;
        let index = (start / self.piece_bytes) as usize;
        let offset = start % self.piece_bytes;
        debug_assert!(start.is_multiple_of(HASH_BLOCK_BYTES));
        debug_assert!(bytes <= WINDOW && bytes as u64 <= self.piece_bytes - offset);
        if self.current.as_ref().is_none_or(|(old, _)| *old != index) {
            self.current = Some((index, fs::File::open(partial_path(self.directory, index))?));
        }
        let file = &mut self.current.as_mut().unwrap().1;
        file.seek(SeekFrom::Start(offset))?;
        file.read_exact(&mut self.buffer[..bytes])?;
        let first = (start / HASH_BLOCK_BYTES) as usize;
        for (index, bytes) in self.buffer[..bytes].chunks(BLOCK).enumerate() {
            let block = first + index;
            if Sha256::digest(bytes).as_slice() != self.hashes[block] {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "USB data does not read back as written",
                ));
            }
            let bits = &mut self.checked[block / 8];
            let bit = 1 << (block % 8);
            if *bits & bit == 0 {
                *bits |= bit;
                self.checked_bytes += bytes.len() as u64;
            }
        }
        self.window = Some((start, bytes));
        Ok(())
    }

    fn complete(mut self, pieces: &[Piece]) -> Result<()> {
        self.current = None;
        let directory = self.directory;
        let progress = crate::progress::Phase::bytes(
            "Verifying remaining USB bytes",
            self.total - self.checked_bytes,
        );
        promote(directory, pieces, |index, piece, _| {
            let start = index as u64 * self.piece_bytes;
            let end = start + piece.bytes;
            let mut block = (start / HASH_BLOCK_BYTES) as usize;
            let last = end.div_ceil(HASH_BLOCK_BYTES) as usize;
            while block < last {
                if self.checked(block) {
                    block += 1;
                    continue;
                }
                let blocks = (block..last)
                    .take(WINDOW / BLOCK)
                    .take_while(|&block| !self.checked(block))
                    .count();
                let start = block as u64 * HASH_BLOCK_BYTES;
                let bytes = (end - start).min((blocks * BLOCK) as u64) as usize;
                self.load(start, bytes)?;
                progress.advance(bytes as u64);
                block += blocks;
            }
            self.current = None;
            Ok(())
        })?;
        progress.finish();
        Ok(())
    }
}

impl Read for Joined<'_> {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        if self.position == self.total || bytes.is_empty() {
            return Ok(0);
        }
        if self
            .window
            .is_none_or(|(start, len)| !(start..start + len as u64).contains(&self.position))
        {
            let offset = self.position % self.piece_bytes;
            let start = self.position - offset % WINDOW as u64;
            let bytes = (self.total - start)
                .min(self.piece_bytes - start % self.piece_bytes)
                .min(WINDOW as u64);
            self.load(start, bytes as usize)?;
        }
        let (start, len) = self.window.unwrap();
        let offset = (self.position - start) as usize;
        let count = bytes.len().min(len - offset);
        bytes[..count].copy_from_slice(&self.buffer[offset..offset + count]);
        self.position += count as u64;
        Ok(count)
    }
}

impl Seek for Joined<'_> {
    fn seek(&mut self, from: SeekFrom) -> io::Result<u64> {
        let position = match from {
            SeekFrom::Start(n) => i128::from(n),
            SeekFrom::Current(n) => i128::from(self.position) + i128::from(n),
            SeekFrom::End(n) => i128::from(self.total) + i128::from(n),
        };
        if position < 0 || position > i128::from(self.total) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "seek outside USB image",
            ));
        }
        self.position = position as u64;
        Ok(self.position)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interrupted_or_unverified_download_leaves_the_install_path_free_for_retry() -> Result<()> {
        let work = tempfile::tempdir()?;
        let volume = work.path().join("staged-volume");
        fs::create_dir(&volume)?;
        let ipsw = work.path().join("source.ipsw");
        crate::efi::synthetic_ipsw(&ipsw)?;
        let bytes = fs::read(ipsw)?;
        let model = restore::payload::SYNTHETIC_MODEL;
        let image = restore::Image {
            version: "test".into(),
            build: restore::payload::SYNTHETIC_BUILD.into(),
            models: vec![model.into()],
            url: "https://updates.cdn-apple.com/source.ipsw".into(),
            sha1: Some(hex::encode(Sha1::digest(&bytes))),
        };
        let attempt = |source: &[u8]| {
            publish_fresh(&volume, &image.build, |scratch| {
                write_from(
                    source,
                    scratch,
                    VolumeLayout::Legacy,
                    model,
                    &image,
                    bytes.len() as u64,
                    HASH_BLOCK_BYTES,
                )
            })
        };
        assert!(attempt(&bytes[..bytes.len() / 2]).is_err());
        assert!(fs::read_dir(&volume)?.next().is_none());
        let mut damaged = bytes.clone();
        damaged[0] ^= 1;
        assert!(
            attempt(&damaged)
                .unwrap_err()
                .to_string()
                .contains("catalog digest")
        );
        assert!(fs::read_dir(&volume)?.next().is_none());

        let written = attempt(&bytes)?;
        assert!(written.wrote > 1);
        let destination = directory(&volume, &image.build);
        let manifest_path = destination.join(MANIFEST_FILE);
        let manifest = fs::read(&manifest_path)?;
        assert_eq!(written.manifest_sha256, stela_formats::sha256(&manifest));
        let joined: Vec<u8> = (0..written.wrote)
            .map(|index| fs::read(destination.join(piece_name(index))))
            .collect::<io::Result<Vec<_>>>()?
            .concat();
        assert_eq!(joined, bytes);
        assert!(
            attempt(&bytes)
                .unwrap_err()
                .to_string()
                .contains("already exists")
        );
        assert_eq!(manifest, fs::read(manifest_path)?);
        Ok(())
    }

    fn partials(
        directory: &Path,
        data: &[u8],
        piece_bytes: usize,
    ) -> Result<(Vec<Piece>, Vec<u8>)> {
        let mut pieces = Vec::new();
        let mut blocks = Vec::new();
        for (index, part) in data.chunks(piece_bytes).enumerate() {
            fs::write(partial_path(directory, index), part)?;
            let mut hashes = Hashes::default();
            hashes.update(part);
            let (sha256, digests) = hashes.finish();
            pieces.push(Piece {
                bytes: part.len() as u64,
                sha256,
            });
            blocks.extend(digests);
        }
        Ok((pieces, blocks))
    }

    #[test]
    fn readback_tracks_unique_windows_and_a_short_final_block() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let piece = WINDOW + BLOCK;
        let data: Vec<u8> = (0..2 * piece + 29).map(|at| (at / 103) as u8).collect();
        let (pieces, hashes) = partials(directory.path(), &data, piece)?;
        let mut joined = Joined::new(directory.path(), piece as u64, data.len() as u64, &hashes)?;
        let mut bytes = [0; 4];
        joined.seek(SeekFrom::Start(piece as u64 - 2))?;
        joined.read_exact(&mut bytes)?;
        assert_eq!(bytes, data[piece - 2..piece + 2]);
        assert_eq!(joined.checked_bytes, (WINDOW + BLOCK) as u64);
        joined.seek(SeekFrom::End(-4))?;
        joined.read_exact(&mut bytes)?;
        assert_eq!(bytes, data[data.len() - 4..]);
        assert_eq!(joined.checked_bytes, (WINDOW + BLOCK + 29) as u64);
        joined.seek(SeekFrom::Start(piece as u64 - 2))?;
        joined.read_exact(&mut bytes)?;
        assert_eq!(bytes, data[piece - 2..piece + 2]);
        assert_eq!(joined.checked_bytes, (WINDOW + BLOCK + 29) as u64);
        joined.complete(&pieces)?;
        for (index, part) in data.chunks(piece).enumerate() {
            assert_eq!(fs::read(directory.path().join(piece_name(index)))?, part);
            assert!(!partial_path(directory.path(), index).exists());
        }
        Ok(())
    }

    #[test]
    fn rereading_a_window_checks_for_corruption_before_returning_bytes() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let data = vec![7; 2 * BLOCK];
        let (_, hashes) = partials(directory.path(), &data, BLOCK)?;
        let mut joined = Joined::new(directory.path(), BLOCK as u64, data.len() as u64, &hashes)?;
        let mut byte = [0];
        joined.read_exact(&mut byte)?;
        assert_eq!(byte, [7]);
        joined.seek(SeekFrom::Start(BLOCK as u64))?;
        joined.read_exact(&mut byte)?;
        fs::write(partial_path(directory.path(), 0), vec![8; BLOCK])?;
        joined.seek(SeekFrom::Start(0))?;
        let mut output = [42; 2];
        assert_eq!(
            joined.read(&mut output).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
        assert_eq!(joined.position, 0);
        assert_eq!(output, [42; 2]);
        Ok(())
    }

    #[test]
    fn untouched_readback_failure_keeps_the_original_promotion_prefix() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let data = vec![7; 2 * BLOCK + 11];
        for truncate in [false, true] {
            let path = directory
                .path()
                .join(if truncate { "short" } else { "corrupt" });
            fs::create_dir(&path)?;
            let (pieces, hashes) = partials(&path, &data, BLOCK)?;
            let mut joined = Joined::new(&path, BLOCK as u64, data.len() as u64, &hashes)?;
            let mut byte = [0];
            joined.read_exact(&mut byte)?;
            let old = path.join(piece_name(1));
            fs::write(&old, b"old piece")?;
            fs::write(
                partial_path(&path, 1),
                vec![9; if truncate { 4 } else { BLOCK }],
            )?;
            let error = joined.complete(&pieces).unwrap_err();
            assert_eq!(
                error.downcast_ref::<io::Error>().unwrap().kind(),
                if truncate {
                    io::ErrorKind::UnexpectedEof
                } else {
                    io::ErrorKind::InvalidData
                }
            );
            assert_eq!(fs::read(path.join(piece_name(0)))?, &data[..BLOCK]);
            assert!(!partial_path(&path, 0).exists());
            assert_eq!(fs::read(old)?, b"old piece");
            assert!(partial_path(&path, 1).exists());
            assert!(partial_path(&path, 2).exists());
        }
        Ok(())
    }

    #[test]
    fn invalid_archive_keeps_the_old_fallback_promotion() -> Result<()> {
        struct CorruptAtEof<'a> {
            data: &'a [u8],
            partial: PathBuf,
        }

        impl Read for CorruptAtEof<'_> {
            fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
                if self.data.is_empty() {
                    fs::write(&self.partial, [8; 17])?;
                }
                self.data.read(output)
            }
        }

        let temp = tempfile::tempdir()?;
        let data = vec![9; BLOCK + 17];
        let image = restore::Image {
            version: "test".into(),
            build: restore::payload::SYNTHETIC_BUILD.into(),
            models: vec![restore::payload::SYNTHETIC_MODEL.into()],
            url: "https://updates.cdn-apple.com/source.ipsw".into(),
            sha1: Some(hex::encode(Sha1::digest(&data))),
        };
        let volume = temp.path();
        assert!(
            write_from(
                &data[..],
                volume,
                VolumeLayout::Legacy,
                restore::payload::SYNTHETIC_MODEL,
                &image,
                data.len() as u64,
                BLOCK as u64,
            )
            .is_err()
        );
        let path = directory(volume, &image.build);
        for (index, part) in data.chunks(BLOCK).enumerate() {
            assert_eq!(fs::read(path.join(piece_name(index)))?, part);
            assert!(!partial_path(&path, index).exists());
        }
        assert!(!path.join(MANIFEST_FILE).exists());

        let volume = temp.path().join("corrupt");
        let path = directory(&volume, &image.build);
        let partial = partial_path(&path, 1);
        let error = write_from(
            CorruptAtEof {
                data: &data,
                partial: partial.clone(),
            },
            &volume,
            VolumeLayout::Legacy,
            restore::payload::SYNTHETIC_MODEL,
            &image,
            data.len() as u64,
            BLOCK as u64,
        )
        .unwrap_err();
        assert_eq!(error.to_string(), "USB data does not read back as written");
        assert_eq!(fs::read(path.join(piece_name(0)))?, &data[..BLOCK]);
        assert!(partial.exists());
        assert!(!path.join(piece_name(1)).exists());
        assert!(!path.join(MANIFEST_FILE).exists());
        Ok(())
    }

    #[test]
    fn download_verifies_before_publication() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let ipsw = temp.path().join("source.ipsw");
        crate::efi::synthetic_ipsw(&ipsw)?;
        let bytes = fs::read(&ipsw)?;
        let image = restore::Image {
            version: "test".into(),
            build: restore::payload::SYNTHETIC_BUILD.into(),
            models: vec![restore::payload::SYNTHETIC_MODEL.into()],
            url: "https://updates.cdn-apple.com/source.ipsw".into(),
            sha1: Some(hex::encode(Sha1::digest(&bytes))),
        };
        let volume = temp.path().join("usb");
        let written = write_from(
            &bytes[..],
            &volume,
            VolumeLayout::Northwood,
            restore::payload::SYNTHETIC_MODEL,
            &image,
            bytes.len() as u64,
            HASH_BLOCK_BYTES,
        )?;
        assert!(!written.manifest_sha256.is_empty());
        assert!(
            written.wrote > 1,
            "the fixture must exercise joined-piece reads"
        );
        let written_dir = directory_in(&volume, VolumeLayout::Northwood, &image.build);
        assert!(!directory(&volume, &image.build).exists());
        let manifest = fs::read(written_dir.join(MANIFEST_FILE))?;
        assert_eq!(written.manifest_sha256, stela_formats::sha256(&manifest));
        let manifest = stela_formats::payload::PayloadManifest::parse(&manifest)
            .map_err(anyhow::Error::msg)?;
        assert_eq!(manifest.source.sha256, Some(stela_formats::sha256(&bytes)));
        assert_eq!(manifest.source.catalog_sha1, image.sha1);
        let Layout::Pieces {
            pieces,
            block_hashes,
            ..
        } = manifest.layout
        else {
            panic!("download did not publish a pieces layout");
        };
        let mut expected_blocks = Vec::new();
        assert_eq!(
            pieces.len(),
            bytes.len().div_ceil(HASH_BLOCK_BYTES as usize)
        );
        for (index, (piece, expected)) in pieces
            .iter()
            .zip(bytes.chunks(HASH_BLOCK_BYTES as usize))
            .enumerate()
        {
            let digest = Sha256::digest(expected);
            expected_blocks.extend_from_slice(&digest);
            assert_eq!(piece.sha256, hex::encode(digest));
            assert_eq!(fs::read(written_dir.join(piece_name(index)))?, expected);
        }
        assert_eq!(
            fs::read(written_dir.join(block_hashes.file))?,
            expected_blocks
        );
        let mut damaged = bytes.clone();
        damaged[0] ^= 1;
        let mut oversized = bytes.clone();
        oversized.push(0);
        for (index, bad_bytes) in [damaged, bytes[..bytes.len() - 1].to_vec(), oversized]
            .into_iter()
            .enumerate()
        {
            let bad = temp.path().join(format!("bad-{index}"));
            assert!(
                write_from(
                    &bad_bytes[..],
                    &bad,
                    VolumeLayout::Legacy,
                    restore::payload::SYNTHETIC_MODEL,
                    &image,
                    bytes.len() as u64,
                    HASH_BLOCK_BYTES
                )
                .is_err()
            );
            assert!(!directory(&bad, &image.build).join(MANIFEST_FILE).exists());
        }
        Ok(())
    }
}
