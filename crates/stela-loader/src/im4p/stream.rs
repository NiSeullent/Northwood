//! Decode a large Image4 object without retaining its compressed payload.
use super::{DecodedImage, compression_metadata, payload_header};
use crate::{Error, Result};
use alloc::{vec, vec::Vec};
use stela_contracts::{refuse, require};
use stela_crypto::{Sha256, digest::Digest};
use stela_formats::{img4, lzfse, payload::ReadAt};

pub(crate) const PREFIX_BYTES: usize = 256;
pub(crate) const MIN_FILE_BYTES: u64 = 8 << 20;
const MAX_SUFFIX_BYTES: usize = 256 << 10;
// Match the installed boot reader's transfer envelope.
const READ_BYTES: usize = 1 << 20;

/// The prefix has already been read from this exact source at offset zero.
/// Keep the envelope snapshots while reading each compressed byte once.
/// Return `None` to use the complete parser for other layouts or payloads.
pub(crate) fn decode<R: ReadAt>(
    source: &mut R,
    prefix: &[u8],
    maximum: u64,
) -> Result<Option<DecodedImage>>
where
    Error: From<R::Error>,
{
    let Ok(file_bytes) = usize::try_from(source.len()) else {
        return Ok(None);
    };
    let Ok(im4p) = img4::payload_range(prefix, file_bytes) else {
        return Ok(None);
    };
    let Some(im4p_prefix) = prefix.get(im4p.start..) else {
        return Ok(None);
    };
    let Ok(header) = payload_header(im4p_prefix, im4p.len()) else {
        return Ok(None);
    };
    let start = im4p.start + header.payload.start;
    let end = im4p.start + header.payload.end;
    let prefix_end = prefix.len().min(end);
    let Some(initial) = prefix.get(start..prefix_end) else {
        return Ok(None);
    };
    if !lzfse::is_lzfse(initial) || file_bytes - end > MAX_SUFFIX_BYTES {
        return Ok(None);
    }
    let mut suffix = vec![0; file_bytes - end];
    if !suffix.is_empty() {
        source.read_at(end as u64, &mut suffix)?;
    }
    let (metadata, outer) = suffix.split_at(im4p.end - end);
    let manifest = img4::suffix_manifest(outer)?;
    let expected = compression_metadata(metadata)?.lzfse_size(maximum);
    let Some(expected) = expected else {
        return Ok(None);
    };

    let mut input = Input {
        source,
        hash: Sha256::new(),
        buffer: Vec::with_capacity(READ_BYTES),
        consumed: 0,
        next: prefix_end,
        end,
    };
    input.hash.update(&prefix[..prefix_end]);
    input.buffer.extend_from_slice(initial);
    // The decoded bytes are hashed frame by frame as they are written, not
    // in a second pass over the whole image (115 MB for a kernel).
    let mut decoded = Sha256::new();
    let Some(bytes) = lzfse::decode_bounded_observed(
        |part| input.read(part),
        end - start,
        expected,
        |frame| decoded.update(frame),
    )?
    else {
        return Ok(None);
    };
    require!(
        bytes.len() == expected,
        "decoded size disagrees with IM4P metadata"
    );
    input.hash.update(&suffix);
    Ok(Some(DecodedImage {
        payload_type: header.payload_type,
        description: header.description,
        bytes,
        container_sha256: hex::encode(input.hash.finalize()),
        decoded_sha256: Some(hex::encode(decoded.finalize())),
        manifest: manifest.map(<[u8]>::to_vec),
    }))
}

struct Input<'a, R> {
    source: &'a mut R,
    hash: Sha256,
    buffer: Vec<u8>,
    consumed: usize,
    next: usize,
    end: usize,
}

impl<R: ReadAt> Input<'_, R>
where
    Error: From<R::Error>,
{
    fn read(&mut self, mut out: &mut [u8]) -> Result<()> {
        while !out.is_empty() {
            if self.consumed == self.buffer.len() {
                // Align bulk reads after the short prefix, leaving only the
                // first and last refills to need partial disk blocks.
                let until_boundary = READ_BYTES - self.next % READ_BYTES;
                let count = (self.end - self.next).min(until_boundary);
                if count == 0 {
                    return Err(refuse!("read past end of Image4 payload").into());
                }
                self.buffer.resize(count, 0);
                self.source.read_at(self.next as u64, &mut self.buffer)?;
                self.hash.update(&self.buffer);
                self.next += count;
                self.consumed = 0;
            }
            let count = out.len().min(self.buffer.len() - self.consumed);
            out[..count].copy_from_slice(&self.buffer[self.consumed..self.consumed + count]);
            self.consumed += count;
            out = &mut out[count..];
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use stela_contracts::Refusal;
    use stela_formats::der::{IA5_STRING, INTEGER, OCTET_STRING, SEQUENCE, encode};

    struct Source<'a> {
        bytes: &'a [u8],
        seen: Vec<bool>,
    }

    impl ReadAt for Source<'_> {
        type Error = Refusal;

        fn len(&self) -> u64 {
            self.bytes.len() as u64
        }

        fn read_at(&mut self, offset: u64, into: &mut [u8]) -> core::result::Result<(), Refusal> {
            let start = usize::try_from(offset).unwrap();
            let range = start..start + into.len();
            for seen in &mut self.seen[range.clone()] {
                assert!(!*seen, "read an original byte twice");
                *seen = true;
            }
            into.copy_from_slice(&self.bytes[range]);
            Ok(())
        }
    }

    fn fixture(size: u32) -> Vec<u8> {
        let mut compressed = Vec::new();
        for _ in 0..3 {
            compressed.extend_from_slice(b"bvx-");
            compressed.extend_from_slice(&32_768u32.to_le_bytes());
            compressed.resize(compressed.len() + 32_768, 0xab);
        }
        compressed.extend_from_slice(b"bvx$");
        let metadata = [
            encode(INTEGER, &[1]),
            encode(INTEGER, &size.to_be_bytes()[1..]),
        ]
        .concat();
        let inner = [
            encode(IA5_STRING, b"IM4P"),
            encode(IA5_STRING, b"krnl"),
            encode(IA5_STRING, b"real envelope"),
            encode(OCTET_STRING, &compressed),
            encode(SEQUENCE, &metadata),
        ]
        .concat();
        encode(SEQUENCE, &inner)
    }

    fn stream(file: &[u8]) -> Result<Option<DecodedImage>> {
        let mut source = Source {
            bytes: file,
            seen: vec![false; file.len()],
        };
        let mut prefix = [0; PREFIX_BYTES];
        source.read_at(0, &mut prefix)?;
        let decoded = decode(&mut source, &prefix, 1 << 20)?;
        if decoded.is_some() {
            assert!(source.seen.iter().all(|&seen| seen));
        }
        Ok(decoded)
    }

    #[test]
    fn wrapped_and_bare_decode_match_exact_slice_identities() -> Result<()> {
        let im4p = fixture(3 * 32_768);
        let manifest = encode(SEQUENCE, &encode(IA5_STRING, b"IM4M"));
        for file in [img4::compose(&im4p, &manifest), im4p] {
            let streamed = stream(&file)?.expect("streamed");
            let parts = img4::parse(&file)?;
            let original = super::super::parse(parts.payload)?;
            let (plaintext, _) = original.decode_payload(1 << 20)?;
            assert_eq!(streamed.bytes, plaintext);
            assert_eq!(streamed.payload_type, original.payload_type);
            assert_eq!(streamed.description, original.description);
            assert_eq!(streamed.manifest.as_deref(), parts.manifest);
            assert_eq!(streamed.container_sha256, stela_formats::sha256(&file));
            // Taken while decoding, the same as a pass over the output.
            assert_eq!(
                streamed.decoded_sha256.as_deref(),
                Some(stela_formats::sha256(&plaintext).as_str())
            );
        }
        Ok(())
    }

    #[test]
    fn rejects_wrong_output_size_and_duplicate_manifest() {
        assert!(stream(&fixture(3 * 32_768 + 1)).is_err());
        let manifest = encode(SEQUENCE, &encode(IA5_STRING, b"IM4M"));
        let outer = [
            encode(IA5_STRING, b"IMG4"),
            fixture(3 * 32_768),
            encode(0xa0, &manifest),
            encode(0xa0, &manifest),
        ]
        .concat();
        assert!(stream(&encode(SEQUENCE, &outer)).is_err());
    }
}
