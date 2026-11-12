//! Upload memory of one command buffer: argument tables, `set*Bytes` data
//! and small transfer sources, placed in host-visible chunks the GPU reads
//! by device address.
//!
//! Encoding needs each upload's address immediately, so space is
//! suballocated from chunks as commands are encoded and the bytes are kept
//! per chunk until commit, when the device copies them into the chunks'
//! mapped memory. Chunks return to the device's pool once the command
//! buffer completes.
use super::{
    error::{Error, Result},
    vulkan::object::Buffer,
};
use alloc::vec::Vec;

/// Chunks are at least this large; bigger uploads get a chunk of their own.
pub const CHUNK_BYTES: u64 = 64 << 10;

/// A host-visible buffer with a device address, from the device's pool.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Chunk {
    pub buffer: Buffer,
    pub address: u64,
    pub size: u64,
}

/// The device's chunk pool.
pub trait ChunkSource {
    /// A free chunk of at least `bytes`.
    fn acquire(&mut self, bytes: u64) -> Result<Chunk>;
}

/// Where an upload landed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Upload {
    pub buffer: Buffer,
    pub offset: u64,
    pub address: u64,
}

#[derive(Debug, Default)]
pub struct UploadArena {
    /// Chunks in use, each with the bytes written so far.
    chunks: Vec<(Chunk, Vec<u8>)>,
    /// Chunks of scratch memory, which no upload shares.
    scratch: Vec<Chunk>,
}

impl UploadArena {
    pub fn write(
        &mut self,
        source: &mut dyn ChunkSource,
        bytes: &[u8],
        alignment: u64,
    ) -> Result<Upload> {
        let (upload, data) = self.reserve(source, bytes.len(), alignment)?;
        data.extend_from_slice(bytes);
        Ok(upload)
    }

    /// Device memory of its own chunk that commit writes nothing into: its
    /// contents are undefined.
    pub(super) fn scratch(&mut self, source: &mut dyn ChunkSource, length: u64) -> Result<Upload> {
        let chunk = source.acquire(length.max(CHUNK_BYTES))?;
        if chunk.size < length {
            return Err(Error::OutOfMemory);
        }
        self.scratch.push(chunk);
        Ok(Upload {
            buffer: chunk.buffer,
            offset: 0,
            address: chunk.address,
        })
    }

    /// Appends storage for a payload encoded directly into the arena.
    pub(super) fn zeroed(
        &mut self,
        source: &mut dyn ChunkSource,
        length: usize,
        alignment: u64,
    ) -> Result<(Upload, &mut [u8])> {
        let (upload, data) = self.reserve(source, length, alignment)?;
        let start = data.len();
        data.resize(start + length, 0);
        Ok((upload, &mut data[start..]))
    }

    fn reserve(
        &mut self,
        source: &mut dyn ChunkSource,
        length: usize,
        alignment: u64,
    ) -> Result<(Upload, &mut Vec<u8>)> {
        debug_assert!(alignment.is_power_of_two());
        let length = length as u64;
        let fits = |(chunk, data): &(Chunk, Vec<u8>)| {
            let offset = (data.len() as u64).next_multiple_of(alignment);
            (offset + length <= chunk.size).then_some(offset)
        };
        let offset = match self.chunks.last().and_then(fits) {
            Some(offset) => offset,
            None => {
                let chunk = source.acquire(length.max(CHUNK_BYTES))?;
                if chunk.size < length || !chunk.address.is_multiple_of(alignment) {
                    return Err(Error::OutOfMemory);
                }
                // Room for a whole default chunk (or this upload) up front,
                // so filling it never reallocates and recopies the bytes.
                let capacity = chunk.size.min(length.max(CHUNK_BYTES)) as usize;
                self.chunks.push((chunk, Vec::with_capacity(capacity)));
                0
            }
        };
        let (chunk, data) = self.chunks.last_mut().expect("a chunk was just ensured");
        data.resize(offset as usize, 0);
        Ok((
            Upload {
                buffer: chunk.buffer,
                offset,
                address: chunk.address + offset,
            },
            data,
        ))
    }

    /// The chunks and the bytes to copy into each before submission; a
    /// scratch chunk has none.
    pub fn finish(self) -> Vec<(Chunk, Vec<u8>)> {
        let mut chunks = self.chunks;
        chunks.extend(self.scratch.into_iter().map(|chunk| (chunk, Vec::new())));
        chunks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Pool {
        next: u64,
    }

    impl ChunkSource for Pool {
        fn acquire(&mut self, bytes: u64) -> Result<Chunk> {
            self.next += 1;
            Ok(Chunk {
                buffer: Buffer(self.next),
                address: self.next << 32,
                size: bytes,
            })
        }
    }

    #[test]
    fn uploads_suballocate_aligned_and_spill_to_new_chunks() {
        let mut pool = Pool::default();
        let mut arena = UploadArena::default();
        let first = arena.write(&mut pool, &[1; 10], 16).unwrap();
        let (second, payload) = arena.zeroed(&mut pool, 4, 16).unwrap();
        assert_eq!(payload, &[0; 4]);
        payload.fill(2);
        assert_eq!((first.offset, second.offset), (0, 16));
        assert_eq!(second.address, (1 << 32) + 16);
        let big = arena
            .write(&mut pool, &alloc::vec![3; CHUNK_BYTES as usize + 1], 16)
            .unwrap();
        assert_eq!((big.buffer, big.offset), (Buffer(2), 0));
        let after = arena.write(&mut pool, &[4], 4).unwrap();
        assert_eq!(after.buffer, Buffer(3), "a full chunk is not reused");
        let chunks = arena.finish();
        assert_eq!(chunks.len(), 3);
        assert_eq!(
            &chunks[0].1[..],
            &[[1; 10].as_slice(), &[0; 6], &[2; 4]].concat()[..]
        );
    }

    /// Scratch memory has a chunk of its own that no later upload shares,
    /// even when it is the arena's first.
    #[test]
    fn scratch_memory_is_never_shared() {
        let mut pool = Pool::default();
        let mut arena = UploadArena::default();
        let scratch = arena.scratch(&mut pool, 64).unwrap();
        let table = arena.write(&mut pool, &[5; 8], 16).unwrap();
        assert_ne!(scratch.buffer, table.buffer);
        let more = arena.scratch(&mut pool, 16).unwrap();
        let next = arena.write(&mut pool, &[6; 8], 16).unwrap();
        assert_ne!(more.buffer, next.buffer);
        assert_eq!(next.buffer, table.buffer);
        let chunks = arena.finish();
        assert_eq!(chunks.len(), 3);
        assert!(
            chunks
                .iter()
                .filter(|(chunk, _)| chunk.buffer == scratch.buffer || chunk.buffer == more.buffer)
                .all(|(_, data)| data.is_empty())
        );
    }
}
