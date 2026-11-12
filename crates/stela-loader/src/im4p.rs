//! Bounded IM4P inspection and plaintext payload decoding.
//!
//! An IM4P is the DER `SEQUENCE { IA5 "IM4P", IA5 type, IA5 description,
//! OCTET STRING payload, [SEQUENCE { INTEGER algorithm, INTEGER size }] }`
//! that restore images use for kernels, DeviceTrees, trust caches and
//! firmware ([blacktop/ipsw img4 payload](https://github.com/blacktop/ipsw/blob/master/pkg/img4/payload.go)).
//! This is not an IMG4 signature validator and never decrypts keybag payloads.
use crate::Result;
use alloc::{borrow::Cow, string::String, vec, vec::Vec};
use core::ops::Range;
use serde::Serialize;
use stela_contracts::{refuse, require};

/// IM4P compression algorithm 1 is LZFSE, with a `bvx*` block stream.
const COMPRESSION_LZFSE: u64 = 1;

pub(crate) mod stream;

const DER_SEQUENCE: u8 = 0x30;
const DER_IA5_STRING: u8 = 0x16;
const DER_OCTET_STRING: u8 = 0x04;
const DER_INTEGER: u8 = 0x02;
const DER_CONTEXT_0: u8 = 0xa0;

struct Der<'a> {
    bytes: &'a [u8],
    position: usize,
    end: usize,
}

impl<'a> Der<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self::prefix(bytes, bytes.len())
    }

    fn prefix(bytes: &'a [u8], end: usize) -> Self {
        Self {
            bytes: &bytes[..bytes.len().min(end)],
            position: 0,
            end,
        }
    }

    fn next_range(&mut self) -> Result<(u8, Range<usize>)> {
        let tag = *self
            .bytes
            .get(self.position)
            .ok_or(refuse!("truncated DER tag"))?;
        require!(tag & 31 != 31, "high-number DER tags are unsupported");
        let first = *self
            .bytes
            .get(self.position + 1)
            .ok_or(refuse!("truncated DER length"))?;
        self.position += 2;
        let length = if first < 128 {
            usize::from(first)
        } else {
            let width = usize::from(first & 127);
            require!(
                width > 0 && width <= size_of::<usize>(),
                "indefinite or oversized DER length"
            );
            let end = self
                .position
                .checked_add(width)
                .ok_or(refuse!("DER length overflow"))?;
            let encoded = self
                .bytes
                .get(self.position..end)
                .ok_or(refuse!("truncated DER long length"))?;
            require!(encoded[0] != 0, "non-minimal DER length");
            self.position = end;
            let mut length = 0usize;
            for byte in encoded {
                length = length
                    .checked_mul(256)
                    .and_then(|n| n.checked_add(usize::from(*byte)))
                    .ok_or(refuse!("DER length overflow"))?;
            }
            require!(length >= 128, "non-minimal DER long length");
            length
        };
        let end = self
            .position
            .checked_add(length)
            .ok_or(refuse!("DER range overflow"))?;
        require!(end <= self.end, "DER value is out of bounds");
        let value = self.position..end;
        self.position = end;
        Ok((tag, value))
    }

    fn next(&mut self) -> Result<(u8, &'a [u8])> {
        let (tag, range) = self.next_range()?;
        let value = self
            .bytes
            .get(range)
            .ok_or(refuse!("DER value is out of bounds"))?;
        Ok((tag, value))
    }

    fn value_range(&mut self, tag: u8) -> Result<Range<usize>> {
        let (actual, value) = self.next_range()?;
        require!(
            actual == tag,
            "unexpected DER tag: {actual:#x}, expected {tag:#x}",
            actual,
            tag
        );
        Ok(value)
    }

    fn value(&mut self, tag: u8) -> Result<&'a [u8]> {
        let range = self.value_range(tag)?;
        self.bytes
            .get(range)
            .ok_or_else(|| refuse!("DER value is out of bounds").into())
    }

    fn done(&self) -> bool {
        self.position == self.end
    }
}

fn text(bytes: &[u8]) -> Result<String> {
    require!(bytes.is_ascii(), "IM4P IA5 string is not ASCII");
    Ok(bytes.iter().map(|&b| char::from(b)).collect())
}

fn integer(bytes: &[u8]) -> Result<u64> {
    require!(
        !bytes.is_empty() && bytes[0] & 128 == 0,
        "negative or empty DER integer"
    );
    require!(
        bytes.len() <= 1 || bytes[0] != 0 || bytes[1] & 128 != 0,
        "non-minimal DER integer"
    );
    let mut value = 0u64;
    for byte in bytes {
        value = value
            .checked_mul(256)
            .and_then(|n| n.checked_add(u64::from(*byte)))
            .ok_or(refuse!("DER integer overflow"))?;
    }
    Ok(value)
}

/// An IM4P field this parser keeps opaque, recorded by tag and length.
#[derive(Debug, Clone, Serialize)]
pub struct Extension {
    pub tag: u8,
    pub bytes: usize,
}

#[derive(Debug)]
pub struct Im4p<'a> {
    /// The four-character payload type, such as `krnl`, `dtre` or `trst`.
    pub payload_type: String,
    pub description: String,
    pub payload: &'a [u8],
    pub compression_algorithm: Option<u64>,
    pub declared_uncompressed_bytes: Option<u64>,
    pub opaque_extensions: Vec<Extension>,
}

pub(crate) struct PayloadHeader {
    pub(crate) payload_type: String,
    pub(crate) description: String,
    /// Relative to the start of the original IM4P.
    pub(crate) payload: Range<usize>,
}

/// Validate every field up to the payload value and identify its exact range.
pub(crate) fn payload_header(prefix: &[u8], bytes: usize) -> Result<PayloadHeader> {
    let mut outer = Der::prefix(prefix, bytes);
    let fields = outer.value_range(DER_SEQUENCE)?;
    require!(outer.done(), "trailing bytes after IM4P container");
    let fields_prefix = prefix.get(fields.start..).unwrap_or_default();
    let mut sequence = Der::prefix(fields_prefix, fields.len());
    require!(
        sequence.value(DER_IA5_STRING)? == b"IM4P",
        "expected a bare IM4P container"
    );
    let payload_type = text(sequence.value(DER_IA5_STRING)?)?;
    require!(
        payload_type.len() == 4,
        "payload type must have four characters"
    );
    let description = text(sequence.value(DER_IA5_STRING)?)?;
    let payload = sequence.value_range(DER_OCTET_STRING)?;
    Ok(PayloadHeader {
        payload_type,
        description,
        payload: (fields.start + payload.start)..(fields.start + payload.end),
    })
}

pub(crate) struct Compression {
    algorithm: Option<u64>,
    declared_bytes: Option<u64>,
    extensions: Vec<Extension>,
}

impl Compression {
    /// Whether the metadata qualifies for bounded LZFSE streaming.
    pub(crate) fn lzfse_size(&self, maximum: u64) -> Option<usize> {
        let bytes = self.declared_bytes?;
        if self.algorithm != Some(COMPRESSION_LZFSE) || !(1..=maximum).contains(&bytes) {
            return None;
        }
        usize::try_from(bytes).ok()
    }
}

/// Validate the remaining fields in the original IM4P.
pub(crate) fn compression_metadata(suffix: &[u8]) -> Result<Compression> {
    let mut sequence = Der::new(suffix);
    let mut compression_algorithm = None;
    let mut declared_uncompressed_bytes = None;
    let mut opaque_extensions = vec![];
    while !sequence.done() {
        let (tag, value) = sequence.next()?;
        match tag {
            DER_SEQUENCE => {
                require!(compression_algorithm.is_none(), "duplicate compression metadata");
                let mut compression = Der::new(value);
                compression_algorithm = Some(integer(compression.value(DER_INTEGER)?)?);
                declared_uncompressed_bytes = Some(integer(compression.value(DER_INTEGER)?)?);
                require!(compression.done(), "unexpected compression metadata fields");
            }
            DER_OCTET_STRING => return Err(refuse!("IM4P keybags or trailing hashes are unsupported; encrypted payloads are not decrypted").into()),
            DER_CONTEXT_0 => opaque_extensions.push(Extension {
                tag,
                bytes: value.len(),
            }),
            _ => return Err(refuse!("unsupported IM4P extension tag {tag:#x}", tag).into()),
        }
    }
    Ok(Compression {
        algorithm: compression_algorithm,
        declared_bytes: declared_uncompressed_bytes,
        extensions: opaque_extensions,
    })
}

/// Parse a bare IM4P container, rejecting keybags and trailing bytes.
pub fn parse(bytes: &[u8]) -> Result<Im4p<'_>> {
    let header = payload_header(bytes, bytes.len())?;
    let compression = compression_metadata(&bytes[header.payload.end..])?;
    Ok(Im4p {
        payload_type: header.payload_type,
        description: header.description,
        payload: &bytes[header.payload],
        compression_algorithm: compression.algorithm,
        declared_uncompressed_bytes: compression.declared_bytes,
        opaque_extensions: compression.extensions,
    })
}

/// Owned Image4 output with the enclosing source identity and original manifest.
pub(crate) struct DecodedImage {
    pub(crate) payload_type: String,
    pub(crate) description: String,
    pub(crate) bytes: Vec<u8>,
    pub(crate) container_sha256: String,
    /// The digest of `bytes`, when the decoder took it as it wrote them.
    pub(crate) decoded_sha256: Option<String>,
    pub(crate) manifest: Option<Vec<u8>>,
}

/// Identities of one decoded IM4P; the payload remains Apple's bytes.
#[derive(Debug, Clone, Serialize)]
pub struct DecodeReceipt {
    pub schema_version: u32,
    pub container_bytes: usize,
    pub container_sha256: String,
    pub payload_type: String,
    pub description: String,
    pub compression: String,
    pub payload_bytes: usize,
    pub payload_sha256: String,
    pub declared_uncompressed_bytes: Option<u64>,
    pub decoded_bytes: usize,
    pub decoded_sha256: String,
    pub opaque_extensions: Vec<Extension>,
    pub signature_validated: bool,
}

impl<'a> Im4p<'a> {
    /// Decode this parsed payload. Boot bundle callers record the enclosing
    /// Image4's identity and need only the plaintext and IM4P metadata.
    pub(crate) fn decode_payload(&self, maximum: u64) -> Result<(Vec<u8>, &'static str)> {
        let (decoded, compression) = self.decode_data(maximum)?;
        Ok((decoded.into_owned(), compression))
    }

    fn decode_data(&self, maximum: u64) -> Result<(Cow<'a, [u8]>, &'static str)> {
        require!(
            self.declared_uncompressed_bytes
                .is_none_or(|n| n <= maximum),
            "declared output exceeds configured limit"
        );
        let (decoded, compression) = if stela_formats::lzfse::is_lzfse(self.payload) {
            require!(
                self.compression_algorithm
                    .is_none_or(|n| n == COMPRESSION_LZFSE),
                "compression record disagrees with LZFSE marker"
            );
            let limit = usize::try_from(maximum).unwrap_or(usize::MAX);
            (
                Cow::Owned(stela_formats::lzfse::decode(self.payload, limit)?),
                "lzfse",
            )
        } else {
            require!(
                self.compression_algorithm.is_none(),
                "unrecognized compressed payload format"
            );
            require!(
                self.payload.len() as u64 <= maximum,
                "payload exceeds configured limit"
            );
            (Cow::Borrowed(self.payload), "none")
        };
        require!(!decoded.is_empty(), "decoded payload is empty");
        require!(
            self.declared_uncompressed_bytes
                .is_none_or(|n| n == decoded.len() as u64),
            "decoded size disagrees with IM4P metadata"
        );
        Ok((decoded, compression))
    }
}

/// Decode a plaintext IM4P payload of any type, returning Apple's bytes.
///
/// LZFSE streams must be consumed exactly and match the declared size. Other
/// payloads are copied unchanged and must not carry compression metadata.
pub fn decode(bytes: &[u8], maximum: u64) -> Result<(Vec<u8>, DecodeReceipt)> {
    let (decoded, receipt) = decode_with_receipt(bytes, maximum)?;
    Ok((decoded.into_owned(), receipt))
}

/// Decode an owned IM4P, retaining its allocation when it carries plaintext.
pub fn decode_owned(mut bytes: Vec<u8>, maximum: u64) -> Result<(Vec<u8>, DecodeReceipt)> {
    let (decoded, receipt) = decode_with_receipt(&bytes, maximum)?;
    match decoded {
        Cow::Owned(decoded) => Ok((decoded, receipt)),
        Cow::Borrowed(payload) => {
            // The plaintext payload is a subslice of the original container.
            let start = payload.as_ptr().addr() - bytes.as_ptr().addr();
            let length = payload.len();
            bytes.copy_within(start..start + length, 0);
            bytes.truncate(length);
            Ok((bytes, receipt))
        }
    }
}

fn decode_with_receipt(bytes: &[u8], maximum: u64) -> Result<(Cow<'_, [u8]>, DecodeReceipt)> {
    let im4p = parse(bytes)?;
    let (decoded, compression) = im4p.decode_data(maximum)?;
    let payload_sha256 = stela_formats::sha256(im4p.payload);
    let decoded_sha256 = match &decoded {
        Cow::Borrowed(_) => payload_sha256.clone(),
        Cow::Owned(decoded) => stela_formats::sha256(decoded),
    };
    let receipt = DecodeReceipt {
        schema_version: 1,
        container_bytes: bytes.len(),
        container_sha256: stela_formats::sha256(bytes),
        payload_type: im4p.payload_type,
        description: im4p.description,
        compression: compression.into(),
        payload_bytes: im4p.payload.len(),
        payload_sha256,
        declared_uncompressed_bytes: im4p.declared_uncompressed_bytes,
        decoded_bytes: decoded.len(),
        decoded_sha256,
        opaque_extensions: im4p.opaque_extensions,
        signature_validated: false,
    };
    Ok((decoded, receipt))
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;

    fn field(tag: u8, data: &[u8]) -> Vec<u8> {
        assert!(data.len() < 128);
        [vec![tag, data.len() as u8], data.to_vec()].concat()
    }

    fn fixture(payload_type: &[u8], payload: &[u8], extension: &[u8]) -> Vec<u8> {
        let fields = [
            field(DER_IA5_STRING, b"IM4P"),
            field(DER_IA5_STRING, payload_type),
            field(DER_IA5_STRING, b"original fixture"),
            field(DER_OCTET_STRING, payload),
            extension.to_vec(),
        ]
        .concat();
        field(DER_SEQUENCE, &fields)
    }

    fn stored_lzfse(raw: &[u8]) -> Vec<u8> {
        [
            b"bvx-".to_vec(),
            (raw.len() as u32).to_le_bytes().to_vec(),
            raw.to_vec(),
            b"bvx$".to_vec(),
        ]
        .concat()
    }

    #[test]
    fn decodes_lzfse_and_raw_payloads_with_limits() -> Result<()> {
        let raw = b"original kernel fixture";
        let size = field(
            DER_SEQUENCE,
            &[
                field(DER_INTEGER, &[1]),
                field(DER_INTEGER, &[raw.len() as u8]),
            ]
            .concat(),
        );
        let container = fixture(b"dtre", &stored_lzfse(raw), &size);
        let (decoded, receipt) = decode(&container, 1024)?;
        assert_eq!(decoded, raw);
        assert_eq!(receipt.payload_type, "dtre");
        assert_eq!(receipt.compression, "lzfse");
        assert_eq!(receipt.decoded_sha256, stela_formats::sha256(raw));
        assert!(!receipt.signature_validated);
        assert!(decode(&container, 2).is_err());

        let plaintext = fixture(b"trst", b"\x02\0\0\0", &[]);
        let container_sha256 = stela_formats::sha256(&plaintext);
        let address = plaintext.as_ptr();
        let (decoded, receipt) = decode_owned(plaintext, 1024)?;
        assert_eq!(decoded.as_ptr(), address);
        assert_eq!(decoded, b"\x02\0\0\0");
        assert_eq!(receipt.compression, "none");
        assert_eq!(receipt.container_sha256, container_sha256);
        assert_eq!(receipt.decoded_sha256, stela_formats::sha256(&decoded));
        assert_eq!(receipt.payload_sha256, receipt.decoded_sha256);

        let mut truncated = stored_lzfse(raw);
        truncated.pop();
        assert!(decode(&fixture(b"krnl", &truncated, &[]), 1024).is_err());
        let mut trailing = container;
        trailing.push(0);
        assert!(decode(&trailing, 1024).is_err());
        Ok(())
    }

    #[test]
    fn rejects_keybags_duplicate_and_incorrect_size_metadata() {
        let raw = [0xcf, 0xfa, 0xed, 0xfe];
        assert!(parse(&fixture(b"krnl", &raw, &field(DER_OCTET_STRING, b"keybag"))).is_err());
        let compression = field(
            DER_SEQUENCE,
            &[field(DER_INTEGER, &[1]), field(DER_INTEGER, &[5])].concat(),
        );
        assert!(
            parse(&fixture(
                b"krnl",
                &raw,
                &[compression.clone(), compression.clone()].concat()
            ))
            .is_err()
        );
        assert!(decode(&fixture(b"krnl", &stored_lzfse(&raw), &compression), 100).is_err());
        assert!(decode(&fixture(b"krnl", &raw, &compression), 100).is_err());
    }

    #[test]
    fn rejects_malformed_lengths_and_short_types() {
        for bytes in [
            vec![0x30, 0x80],
            vec![0x30, 0x81, 0x01, 0],
            vec![0x30, 0x82, 0, 128],
            vec![0x30, 0xff],
            vec![0x30, 4, 0x16, 4],
        ] {
            assert!(parse(&bytes).is_err());
        }
        assert!(parse(&fixture(b"krn", b"x", &[])).is_err());
    }
}
