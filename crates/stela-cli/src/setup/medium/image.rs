//! The IPSW a full install medium holds, read back from its pieces with
//! every block checked against `pieces.bin`, as `stela.efi` reads it. An
//! update prepares boot bundles from it instead of downloading the image
//! again.
use super::*;
use std::io;
use stela_formats::payload::{
    PayloadManifest, ReadAt,
    pieces::{Extent, Joined},
};

/// The medium's piece files back to back: a device offset is the IPSW
/// offset.
pub(in super::super) struct PieceFiles {
    /// Each piece's first IPSW offset, its bytes and its file.
    pieces: Vec<(u64, u64, fs::File)>,
    total: u64,
}

impl ReadAt for PieceFiles {
    type Error = io::Error;

    fn len(&self) -> u64 {
        self.total
    }

    fn read_at(&mut self, mut offset: u64, mut buffer: &mut [u8]) -> io::Result<()> {
        while !buffer.is_empty() {
            let index = self
                .pieces
                .partition_point(|(start, _, _)| *start <= offset)
                .checked_sub(1)
                .ok_or_else(|| io::Error::other("read before the first piece"))?;
            let (start, bytes, file) = &mut self.pieces[index];
            let within = offset - *start;
            if within >= *bytes {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "read past the last piece",
                ));
            }
            let count = (buffer.len() as u64).min(*bytes - within) as usize;
            file.seek(SeekFrom::Start(within))?;
            file.read_exact(&mut buffer[..count])?;
            buffer = &mut buffer[count..];
            offset += count as u64;
        }
        Ok(())
    }
}

/// [`Read`] and [`Seek`] over a [`ReadAt`] source, as ZIP readers want.
pub(crate) struct Reader<R> {
    source: R,
    position: u64,
}

impl<R: ReadAt> Reader<R> {
    pub(crate) fn new(source: R) -> Self {
        Self {
            source,
            position: 0,
        }
    }
}

impl<R: ReadAt> Read for Reader<R> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let left = self.source.len().saturating_sub(self.position);
        let count = (buffer.len() as u64).min(left) as usize;
        if count == 0 {
            return Ok(0);
        }
        self.source
            .read_at(self.position, &mut buffer[..count])
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, format!("{error:?}")))?;
        self.position += count as u64;
        Ok(count)
    }
}

impl<R: ReadAt> Seek for Reader<R> {
    fn seek(&mut self, from: SeekFrom) -> io::Result<u64> {
        let position = match from {
            SeekFrom::Start(n) => i128::from(n),
            SeekFrom::Current(n) => i128::from(self.position) + i128::from(n),
            SeekFrom::End(n) => i128::from(self.source.len()) + i128::from(n),
        };
        self.position = u64::try_from(position)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "seek before the start"))?;
        Ok(self.position)
    }
}

/// The IPSW `medium` holds, read from its pieces. Refused for a remote
/// medium, one whose `payload.json` is not the one setup wrote, and one
/// with a missing or short piece; every block read later is checked
/// against `pieces.bin`.
pub(in super::super) fn open(medium: &Medium) -> Result<Reader<Joined<PieceFiles>>> {
    ensure!(
        !medium.remote,
        "the install medium names the image on Apple's servers"
    );
    let problems = medium.problems();
    ensure!(problems.is_empty(), "{}", problems.join("; "));
    let directory = medium.directory();
    let manifest = PayloadManifest::parse(&fs::read(directory.join(MANIFEST_FILE))?)
        .map_err(|error| anyhow::anyhow!("{error}"))?;
    let Layout::Pieces {
        pieces,
        block_hashes,
        ..
    } = &manifest.layout
    else {
        bail!("{} names no pieces", directory.display());
    };
    let hashes = fs::read(directory.join(&block_hashes.file))?;
    let mut files = Vec::new();
    let mut extents = Vec::new();
    let mut start = 0u64;
    for (index, piece) in pieces.iter().enumerate() {
        let path = directory.join(piece_name(index));
        let file =
            fs::File::open(&path).with_context(|| format!("cannot read {}", path.display()))?;
        extents.push(vec![Extent {
            offset: start,
            bytes: piece.bytes,
        }]);
        files.push((start, piece.bytes, file));
        start = start
            .checked_add(piece.bytes)
            .context("install medium size overflow")?;
    }
    let device = PieceFiles {
        pieces: files,
        total: start,
    };
    let joined = Joined::new(device, &manifest, extents, &hashes)
        .map_err(|error| anyhow::anyhow!("{}: {error}", directory.display()))?;
    Ok(Reader::new(joined))
}
