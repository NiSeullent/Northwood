//! The restore payload: the IPSW members an installation needs, how they are
//! stored on removable media, and verified reads of them.
//!
//! `stela-dev efi install --ipsw` writes a payload manifest beside the members,
//! either as split slices for FAT32 or as the whole IPSW. `stela.efi` reads
//! either form before boot services end, verifies every member's SHA-256
//! and copies the Erase subset into the target disk's staging partition.
//! The restore client reads the staged copy through the same
//! [`PayloadSource`] trait afterwards.
//!
//! `stela setup` writes the install medium instead: the whole IPSW as
//! numbered pieces below FAT32's file limit ([`Layout::Pieces`]), which
//! [`pieces::Joined`] presents to the restore host as one IPSW, or only
//! the manifest of the IPSW on Apple's CDN ([`Layout::Remote`]), which
//! `stela.efi` streams during the restore.
use alloc::{
    format,
    string::{String, ToString},
    vec::Vec,
};
use core::fmt;
use serde::{Deserialize, Serialize};
use stela_crypto::Sha256;
use stela_crypto::digest::Digest;

pub mod gpt;
pub mod pieces;
pub mod staging;
pub mod zip;

pub const KIND: &str = "stela_restore_payload";
pub const MANIFEST_FILE: &str = "payload.json";
/// FAT32's largest file.
pub const MAX_SLICE_BYTES: u64 = 4 * 1024 * 1024 * 1024 - 1;
/// The install medium's piece size: below FAT32's largest file, on a
/// 1 MiB boundary.
pub const PIECE_BYTES: u64 = 4 * 1024 * 1024 * 1024 - 1024 * 1024;
/// Pieces an install medium may have: `ipsw.000` to `ipsw.999`.
pub const MAX_PIECES: usize = 1000;
/// The block a member's precomputed hashes cover ([`BlockHashes`]).
pub const HASH_BLOCK_BYTES: u64 = 64 * 1024;
/// The block hashes' file name beside `payload.json`.
pub const BLOCK_HASHES_FILE: &str = "blocks.bin";
/// The install medium's piece block hashes' file name beside
/// `payload.json`.
pub const PIECE_HASHES_FILE: &str = "pieces.bin";
/// The largest block a hash may cover.
const MAX_HASH_BLOCK_BYTES: u64 = 1024 * 1024;
/// The most block hashes a manifest may name: 64 GiB of members.
pub const MAX_HASH_BLOCKS: u64 = 1 << 20;
const MAX_MANIFEST_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PayloadError {
    /// The manifest or a staging header is malformed or inconsistent.
    Manifest(&'static str),
    Archive(&'static str),
    Unsupported(&'static str),
    Disk(&'static str),
    /// A member's bytes differ from the manifest.
    Verification {
        member: String,
        reason: &'static str,
    },
    Io(String),
}

impl PayloadError {
    pub fn io(error: impl fmt::Debug) -> Self {
        Self::Io(format!("{error:?}"))
    }
}

impl fmt::Display for PayloadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Manifest(what) => write!(f, "payload manifest: {what}"),
            Self::Archive(what) => write!(f, "archive: {what}"),
            Self::Unsupported(what) => write!(f, "unsupported: {what}"),
            Self::Disk(what) => write!(f, "target disk: {what}"),
            Self::Verification { member, reason } => write!(f, "{member}: {reason}"),
            Self::Io(what) => write!(f, "read failed: {what}"),
        }
    }
}

/// Random-access reads of a file or partition; each call fills `buffer`.
pub trait ReadAt {
    type Error: fmt::Debug;
    fn len(&self) -> u64;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn read_at(&mut self, offset: u64, buffer: &mut [u8]) -> Result<(), Self::Error>;
    /// Whether the last failed read only ran out of the time it may wait,
    /// as a network source's read bounded for its caller does: the same
    /// read later goes on from what arrived meanwhile. Asking clears it.
    fn blocked(&mut self) -> bool {
        false
    }
}

/// A borrowed source, so one disk can be opened as several containers in
/// turn.
impl<R: ReadAt + ?Sized> ReadAt for &mut R {
    type Error = R::Error;

    fn len(&self) -> u64 {
        (**self).len()
    }

    fn read_at(&mut self, offset: u64, buffer: &mut [u8]) -> Result<(), Self::Error> {
        (**self).read_at(offset, buffer)
    }

    fn blocked(&mut self) -> bool {
        (**self).blocked()
    }
}

/// A member's bytes in order. `Ok(0)` ends the member.
pub trait MemberRead {
    fn read(&mut self, buffer: &mut [u8]) -> Result<usize, PayloadError>;
}

/// Where the restore client gets payload members, whatever stores them.
pub trait PayloadSource {
    type Reader<'a>: MemberRead
    where
        Self: 'a;
    fn manifest(&self) -> &PayloadManifest;
    /// Stream a member named in the manifest. Bytes are verified only by
    /// reading the member to the end, as [`copy_verified`] does.
    fn open(&mut self, name: &str) -> Result<Self::Reader<'_>, PayloadError>;
}

#[derive(Debug)]
pub enum CopyError<E> {
    Payload(PayloadError),
    Sink(E),
}

impl<E: fmt::Debug> fmt::Display for CopyError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Payload(error) => write!(f, "{error}"),
            Self::Sink(error) => write!(f, "write failed: {error:?}"),
        }
    }
}

/// Stream a member through `sink` and check its length and SHA-256 at the
/// end. The sink sees every byte before verification finishes, so a caller
/// that keeps the bytes must commit them only after this returns `Ok`.
pub fn copy_verified<S: PayloadSource, E>(
    source: &mut S,
    member: &Member,
    buffer: &mut [u8],
    mut sink: impl FnMut(&[u8]) -> Result<(), E>,
) -> Result<(), CopyError<E>> {
    let mismatch = |reason| {
        CopyError::Payload(PayloadError::Verification {
            member: member.name.clone(),
            reason,
        })
    };
    let mut reader = source.open(&member.name).map_err(CopyError::Payload)?;
    let mut hash = Sha256::new();
    let mut total = 0u64;
    loop {
        let count = reader.read(buffer).map_err(CopyError::Payload)?;
        if count == 0 {
            break;
        }
        total += count as u64;
        if total > member.bytes {
            return Err(mismatch("longer than the manifest says"));
        }
        hash.update(&buffer[..count]);
        sink(&buffer[..count]).map_err(CopyError::Sink)?;
    }
    if total != member.bytes {
        return Err(mismatch("shorter than the manifest says"));
    }
    if member.sha256.as_deref() != Some(&*hex::encode(hash.finalize())) {
        return Err(mismatch("SHA-256 differs from the manifest"));
    }
    Ok(())
}

/// The source IPSW the payload came from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceImage {
    pub file_name: String,
    pub bytes: u64,
    /// The whole IPSW's SHA-256; absent for a [`Layout::Remote`] image,
    /// which setup reads only in part.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    pub build_manifest_sha256: String,
    /// The SHA-1 Apple's catalog lists for the image, when it was checked.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub catalog_sha1: Option<String>,
}

/// One piece of the install medium's IPSW.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Piece {
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Layout {
    /// `<images>/<build>/<member>.NNN` slices of at most `slice_bytes`.
    Split { slice_bytes: u64 },
    /// The whole IPSW at `path`, relative to the volume root.
    Archive { path: String },
    /// Members in a staging partition.
    Staged,
    /// The whole IPSW as `ipsw.NNN` pieces beside the manifest, each
    /// `piece_bytes` but the last; their concatenation is the IPSW.
    /// `block_hashes` names the SHA-256 of every block of every piece, in
    /// piece order, each piece's last block its remainder, so a reader
    /// checks every block it reads without hashing a whole piece first.
    Pieces {
        piece_bytes: u64,
        pieces: Vec<Piece>,
        block_hashes: BlockHashes,
    },
    /// The IPSW on Apple's CDN at `url` (plain HTTP), read by byte ranges,
    /// each request pinned to `etag` (RFC 9110 13.1.5 If-Range) so a changed
    /// object fails instead of mixing two versions.
    Remote { url: String, etag: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Member {
    /// The ZIP member path in the IPSW.
    pub name: String,
    pub bytes: u64,
    /// Absent for a member of a [`Layout::Remote`] image that the device
    /// verifies as the restore consumes it: an Image4 component, whose
    /// digest the personalized ticket covers, an AEA archive, which
    /// authenticates itself under its key, or a global ticket. Stela then
    /// passes its bytes through unhashed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    /// Its SHA-384, the digest a BuildManifest component names, when the
    /// manifest carries precomputed hashes ([`BlockHashes`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha384: Option<String>,
}

/// The SHA-256 of every [`HASH_BLOCK_BYTES`] block of every member,
/// computed by the host that wrote the manifest, so the restore host checks
/// each block as it reads it instead of scanning the whole payload first.
/// The hashes are a separate file beside the manifest: each member's in
/// manifest order, its last block the remainder, 32 raw bytes each.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockHashes {
    /// Its file name, in the manifest's directory.
    pub file: String,
    pub block_bytes: u64,
    /// The SHA-256 of the whole file.
    pub sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Purpose {
    /// A full restore that erases the target.
    Erase,
    /// A restore that keeps user data.
    Update,
    /// The installed OS's own objects.
    Installed,
}

/// One BuildManifest identity of the payload's model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IdentitySet {
    /// Its index in `BuildIdentities`.
    pub index: u32,
    pub variant: String,
    pub purpose: Purpose,
    pub members: Vec<String>,
}

/// A prefetched public AEA key response, so the restore needs no network.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AeaKey {
    pub member: String,
    pub header_sha256: String,
    pub url: String,
    pub key_pem: String,
    pub key_sha256: String,
    /// SHA-256 of the binary-plist `URLAsset` reply the restore host stages
    /// for this member's `fcs-key-url` GET, so the host serves the recipient
    /// key offline. Absent in inventories written before the staged answer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url_asset_sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PayloadManifest {
    pub schema_version: u32,
    pub kind: String,
    pub product_version: String,
    pub product_build: String,
    pub model: String,
    pub device_class: String,
    pub source: SourceImage,
    pub layout: Layout,
    pub members: Vec<Member>,
    pub identities: Vec<IdentitySet>,
    /// The global `apticket` manifests, which no identity names.
    pub global_manifests: Vec<String>,
    pub aea_keys: Vec<AeaKey>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub block_hashes: Option<BlockHashes>,
}

/// A file name of one ordinary component.
fn safe_name(name: &str) -> bool {
    !name.contains('/') && safe_path(name)
}

/// A relative path made of ordinary components only.
pub fn safe_path(path: &str) -> bool {
    !path.is_empty()
        && path.len() <= 1024
        && !path.contains(['\\', '\0', ':'])
        && path.split('/').all(|part| !matches!(part, "" | "." | ".."))
}

fn sha256_hex(text: &str) -> bool {
    lower_hex(text, 64)
}

fn lower_hex(text: &str, digits: usize) -> bool {
    text.len() == digits && text.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// The install medium's directory for a build, relative to the root of the
/// USB's FAT32 data partition.
pub fn install_directory(build: &str) -> String {
    format!("stela/install/{build}")
}

/// A piece's file name in the install directory: an 8.3 name, as FAT
/// keeps it.
pub fn piece_name(index: usize) -> String {
    format!("ipsw.{index:03}")
}

/// Pieces of `piece_bytes` that hold `bytes`.
pub fn piece_sizes(bytes: u64, piece_bytes: u64) -> Vec<u64> {
    let mut sizes = Vec::new();
    let mut left = bytes;
    while left > 0 && piece_bytes > 0 {
        let size = left.min(piece_bytes);
        sizes.push(size);
        left -= size;
    }
    sizes
}

/// The payload directory for a build, relative to the volume root.
pub fn images_directory(build: &str) -> String {
    format!("EFI/stela/images/{build}")
}

/// Slices a member of `bytes` takes; an empty member still has one.
pub fn slice_count(bytes: u64, slice_bytes: u64) -> u64 {
    bytes.div_ceil(slice_bytes).max(1)
}

/// A slice's path relative to the build's payload directory.
pub fn slice_path(member: &str, index: u64) -> String {
    format!("{member}.{index:03}")
}

impl PayloadManifest {
    pub fn parse(bytes: &[u8]) -> Result<Self, PayloadError> {
        if bytes.len() > MAX_MANIFEST_BYTES {
            return Err(PayloadError::Manifest("exceeds 16 MiB"));
        }
        let manifest: Self =
            serde_json::from_slice(bytes).map_err(|_| PayloadError::Manifest("malformed JSON"))?;
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn validate(&self) -> Result<(), PayloadError> {
        let fail = |what| Err(PayloadError::Manifest(what));
        if self.schema_version != 1 || self.kind != KIND {
            return fail("unknown schema");
        }
        if self.product_build.is_empty()
            || !self
                .product_build
                .bytes()
                .all(|b| b.is_ascii_alphanumeric())
        {
            return fail("invalid build");
        }
        let remote = matches!(self.layout, Layout::Remote { .. });
        if self
            .source
            .sha256
            .as_deref()
            .map_or(!remote, |s| !sha256_hex(s))
            || !sha256_hex(&self.source.build_manifest_sha256)
            || self
                .source
                .catalog_sha1
                .as_ref()
                .is_some_and(|digest| !lower_hex(digest, 40))
        {
            return fail("invalid source digest");
        }
        match &self.layout {
            Layout::Split { slice_bytes } if (1..=MAX_SLICE_BYTES).contains(slice_bytes) => {}
            Layout::Split { .. } => return fail("slice size outside 1..=4 GiB-1"),
            Layout::Archive { path } if safe_path(path) => {}
            Layout::Archive { .. } => return fail("unsafe archive path"),
            Layout::Staged => {}
            Layout::Remote { url, etag } => {
                let printable = |text: &str| text.bytes().all(|b| (0x21..0x7f).contains(&b));
                if !url.starts_with("http://")
                    || !printable(url)
                    || etag.is_empty()
                    || !printable(etag)
                {
                    return fail("remote image URL or ETag");
                }
            }
            Layout::Pieces {
                piece_bytes,
                pieces,
                block_hashes,
            } => {
                if !safe_name(&block_hashes.file) || !sha256_hex(&block_hashes.sha256) {
                    return fail("unsafe piece hash file or invalid digest");
                }
                let block = block_hashes.block_bytes;
                if !(1..=MAX_HASH_BLOCK_BYTES).contains(&block) || piece_bytes % block != 0 {
                    return fail("piece blocks do not divide the piece size");
                }
                if !(1..=MAX_SLICE_BYTES).contains(piece_bytes) {
                    return fail("piece size outside 1..=4 GiB-1");
                }
                if pieces.is_empty() || pieces.len() > MAX_PIECES {
                    return fail("piece count outside 1..=1000");
                }
                let sizes: Vec<u64> = pieces.iter().map(|piece| piece.bytes).collect();
                if sizes != piece_sizes(self.source.bytes, *piece_bytes) {
                    return fail("pieces do not cover the IPSW in piece-size steps");
                }
                if !pieces.iter().all(|piece| sha256_hex(&piece.sha256)) {
                    return fail("invalid piece digest");
                }
            }
        }
        let mut names: Vec<&str> = Vec::with_capacity(self.members.len());
        for member in &self.members {
            if !safe_path(&member.name)
                || member.sha256.as_deref().map_or(!remote, |s| !sha256_hex(s))
                || member
                    .sha384
                    .as_ref()
                    .is_some_and(|digest| !lower_hex(digest, 96))
            {
                return fail("unsafe member name or invalid digest");
            }
            names.push(&member.name);
        }
        if let Some(blocks) = &self.block_hashes {
            if !safe_name(&blocks.file) || !sha256_hex(&blocks.sha256) {
                return fail("unsafe block hash file or invalid digest");
            }
            if blocks.block_bytes != HASH_BLOCK_BYTES {
                return fail("block hashes of another block size");
            }
            if self.hashed().any(|member| member.sha384.is_none()) {
                return fail("block hashes without every hashed member's SHA-384");
            }
            if self.hash_blocks() > MAX_HASH_BLOCKS {
                return fail("more block hashes than 64 GiB of members");
            }
        }
        names.sort_unstable();
        if names.windows(2).any(|pair| pair[0] == pair[1]) {
            return fail("duplicate member");
        }
        let known = |name: &str| names.binary_search(&name).is_ok();
        let referenced = self
            .identities
            .iter()
            .flat_map(|identity| identity.members.iter())
            .chain(&self.global_manifests)
            .chain(self.aea_keys.iter().map(|key| &key.member));
        if !referenced.into_iter().all(|name| known(name)) {
            return fail("names a member it does not list");
        }
        for key in &self.aea_keys {
            if crate::sha256(key.key_pem.as_bytes()) != key.key_sha256
                || !sha256_hex(&key.header_sha256)
                || key
                    .url_asset_sha256
                    .as_deref()
                    .is_some_and(|s| !sha256_hex(s))
            {
                return fail("AEA key digest mismatch");
            }
        }
        Ok(())
    }

    /// The BuildManifest index of the one identity of `purpose`: Erase for
    /// a full restore, Update for one that keeps the installed system's
    /// data.
    pub fn restore_identity(&self, purpose: Purpose) -> Result<u32, PayloadError> {
        let expected = match purpose {
            Purpose::Erase => "expected exactly one Erase identity",
            Purpose::Update => "expected exactly one Update identity",
            Purpose::Installed => {
                return Err(PayloadError::Manifest(
                    "an installed-system identity is no restore",
                ));
            }
        };
        let mut found = self
            .identities
            .iter()
            .filter(|identity| identity.purpose == purpose);
        match (found.next(), found.next()) {
            (Some(identity), None) => Ok(identity.index),
            _ => Err(PayloadError::Manifest(expected)),
        }
    }

    pub fn member(&self, name: &str) -> Option<&Member> {
        self.members.iter().find(|member| member.name == name)
    }

    /// The members whose bytes Stela checks: all but a remote image's
    /// device-verified ones.
    pub fn hashed(&self) -> impl Iterator<Item = &Member> {
        self.members.iter().filter(|member| member.sha256.is_some())
    }

    /// The members a full restore needs: the Erase identity's and the
    /// global manifests, in manifest order, laid out for staging.
    pub fn erase_subset(&self) -> Result<Self, PayloadError> {
        let erase: Vec<&IdentitySet> = self
            .identities
            .iter()
            .filter(|identity| identity.purpose == Purpose::Erase)
            .collect();
        let [identity] = erase[..] else {
            return Err(PayloadError::Manifest(
                "expected exactly one Erase identity",
            ));
        };
        let wanted =
            |name: &String| identity.members.contains(name) || self.global_manifests.contains(name);
        let members: Vec<Member> = self
            .members
            .iter()
            .filter(|member| wanted(&member.name))
            .cloned()
            .collect();
        Ok(Self {
            layout: Layout::Staged,
            identities: alloc::vec![identity.clone()],
            aea_keys: self
                .aea_keys
                .iter()
                .filter(|key| wanted(&key.member))
                .cloned()
                .collect(),
            members,
            ..self.clone()
        })
    }

    /// Blocks of [`HASH_BLOCK_BYTES`] the hashed members take, in the
    /// order [`BlockHashes`] lists them.
    pub fn hash_blocks(&self) -> u64 {
        self.hashed()
            .map(|member| member.bytes.div_ceil(HASH_BLOCK_BYTES))
            .sum()
    }

    /// The install medium's piece blocks, in the order its piece hash file
    /// lists them; zero for another layout.
    pub fn piece_hash_blocks(&self) -> u64 {
        match &self.layout {
            Layout::Pieces {
                pieces,
                block_hashes,
                ..
            } => pieces
                .iter()
                .map(|piece| piece.bytes.div_ceil(block_hashes.block_bytes))
                .sum(),
            _ => 0,
        }
    }

    pub fn total_bytes(&self) -> u64 {
        self.members.iter().map(|member| member.bytes).sum()
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|error| error.to_string())
    }
}

#[cfg(any(test, feature = "test-support"))]
pub mod test_support;

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use alloc::vec::Vec;

    use super::test_support::manifest;

    #[test]
    fn manifests_round_trip_and_reject_unsafe_or_inconsistent_fields() {
        let (manifest, _) = manifest();
        let json = manifest.to_json();
        assert_eq!(PayloadManifest::parse(json.as_bytes()).unwrap(), manifest);
        let broken: [fn(&mut PayloadManifest); 19] = [
            |m| m.members[0].name = "../escape".into(),
            |m| m.members[0].sha256 = Some("A".repeat(64)),
            |m| m.members[0].sha256 = None,
            |m| m.source.sha256 = None,
            |m| m.members[1].name = m.members[0].name.clone(),
            |m| m.identities[0].members.push("missing".into()),
            |m| {
                m.layout = Layout::Split {
                    slice_bytes: MAX_SLICE_BYTES + 1,
                }
            },
            |m| {
                m.layout = Layout::Archive {
                    path: "C:/x.ipsw".into(),
                }
            },
            |m| m.aea_keys[0].key_pem.push('x'),
            |m| m.product_build = "26A434/..".into(),
            |m| m.source.catalog_sha1 = Some("A".repeat(40)),
            |m| m.members[0].sha384 = Some("0".repeat(64)),
            |m| m.block_hashes = Some(blocks("blocks.bin")),
            |m| {
                for member in &mut m.members {
                    member.sha384 = Some("5".repeat(96));
                }
                m.block_hashes = Some(blocks("../blocks.bin"));
            },
            |m| m.layout = pieces(&[1], 0),
            |m| m.layout = pieces(&[], 2),
            |m| m.layout = pieces(&[1, 1], 2),
            |m| {
                m.layout = pieces(&[1], 2);
                if let Layout::Pieces { block_hashes, .. } = &mut m.layout {
                    block_hashes.block_bytes = 3;
                }
            },
            |m| {
                m.layout = pieces(&[1], 2);
                if let Layout::Pieces { pieces, .. } = &mut m.layout {
                    pieces[0].sha256 = "x".into();
                }
            },
        ];
        for change in broken {
            let mut bad = manifest.clone();
            change(&mut bad);
            assert!(bad.validate().is_err(), "{bad:?}");
        }
    }

    fn blocks(file: &str) -> BlockHashes {
        BlockHashes {
            file: file.into(),
            block_bytes: HASH_BLOCK_BYTES,
            sha256: "6".repeat(64),
        }
    }

    #[test]
    fn block_hashes_need_every_members_sha384() {
        let (mut manifest, _) = manifest();
        for member in &mut manifest.members {
            member.sha384 = Some("5".repeat(96));
        }
        manifest.block_hashes = Some(blocks(BLOCK_HASHES_FILE));
        manifest.validate().unwrap();
        let json = manifest.to_json();
        assert_eq!(PayloadManifest::parse(json.as_bytes()).unwrap(), manifest);
        // 3000 bytes, 2,500,000 bytes (39 blocks), 10 bytes and an empty one.
        assert_eq!(manifest.hash_blocks(), 1 + 39 + 1);
        manifest.block_hashes.as_mut().unwrap().block_bytes = 4096;
        assert!(manifest.validate().is_err());
    }

    #[test]
    fn only_a_remote_image_leaves_device_verified_members_unhashed() {
        let (mut manifest, _) = manifest();
        manifest.layout = Layout::Remote {
            url: "http://updates-http.cdn-apple.com/a/UniversalMac.ipsw".into(),
            etag: "\"8faf3ef6-3175\"".into(),
        };
        manifest.source.sha256 = None;
        manifest.members[1].sha256 = None;
        for member in manifest.hashed().collect::<Vec<_>>() {
            assert!(member.sha256.is_some());
        }
        // The unhashed member takes no block hashes.
        manifest.members[0].sha384 = Some("5".repeat(96));
        manifest.members[2].sha384 = Some("5".repeat(96));
        manifest.members[3].sha384 = Some("5".repeat(96));
        manifest.block_hashes = Some(blocks(BLOCK_HASHES_FILE));
        manifest.validate().unwrap();
        assert_eq!(manifest.hash_blocks(), 1 + 1);
        let json = manifest.to_json();
        assert_eq!(PayloadManifest::parse(json.as_bytes()).unwrap(), manifest);
        for (url, etag) in [
            ("https://a/b", "e"),
            ("http://a b", "e"),
            ("http://a/b", ""),
        ] {
            let mut bad = manifest.clone();
            bad.layout = Layout::Remote {
                url: url.into(),
                etag: etag.into(),
            };
            assert!(bad.validate().is_err(), "{url} {etag}");
        }
    }

    /// A pieces layout of these sizes.
    fn pieces(sizes: &[u64], piece_bytes: u64) -> Layout {
        Layout::Pieces {
            block_hashes: BlockHashes {
                file: PIECE_HASHES_FILE.into(),
                block_bytes: 1,
                sha256: "7".repeat(64),
            },
            piece_bytes,
            pieces: sizes
                .iter()
                .map(|&bytes| Piece {
                    bytes,
                    sha256: "3".repeat(64),
                })
                .collect(),
        }
    }

    #[test]
    fn pieces_cover_the_ipsw_in_piece_size_steps() {
        let (mut manifest, _) = manifest();
        manifest.source.bytes = 5;
        manifest.source.catalog_sha1 = Some("4".repeat(40));
        manifest.layout = pieces(&[2, 2, 1], 2);
        manifest.validate().unwrap();
        let json = manifest.to_json();
        assert_eq!(PayloadManifest::parse(json.as_bytes()).unwrap(), manifest);
        assert!(json.contains("\"kind\": \"pieces\""), "{json}");
        manifest.layout = pieces(&[2, 1, 2], 2);
        assert!(manifest.validate().is_err());
        assert_eq!(piece_sizes(5, 2), [2, 2, 1]);
        assert_eq!(piece_sizes(4, 2), [2, 2]);
        assert!(piece_sizes(0, 2).is_empty());
        // The 26A434 IPSW (26,637,307,067 bytes) takes seven pieces.
        assert_eq!(piece_sizes(26_637_307_067, PIECE_BYTES).len(), 7);
        const { assert!(PIECE_BYTES < MAX_SLICE_BYTES) };
        assert_eq!(piece_name(7), "ipsw.007");
        assert_eq!(install_directory("26A434"), "stela/install/26A434");
        assert_eq!(manifest.piece_hash_blocks(), 5);
        assert_eq!(manifest.restore_identity(Purpose::Erase).unwrap(), 20);
        assert_eq!(manifest.restore_identity(Purpose::Update).unwrap(), 80);
        // Even a manifest's one installed-system identity is no restore.
        let mut installed = manifest.clone();
        installed.identities[1].purpose = Purpose::Installed;
        assert!(installed.restore_identity(Purpose::Installed).is_err());
        manifest.identities[1].purpose = Purpose::Erase;
        assert!(manifest.restore_identity(Purpose::Erase).is_err());
        assert!(manifest.restore_identity(Purpose::Update).is_err());
    }

    #[test]
    fn the_erase_subset_keeps_erase_members_global_manifests_and_their_keys() {
        let (manifest, contents) = manifest();
        let subset = manifest.erase_subset().unwrap();
        let names: Vec<&str> = subset.members.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(
            names,
            [&contents[0].0, &contents[1].0, &contents[3].0].map(String::as_str)
        );
        assert_eq!(subset.layout, Layout::Staged);
        assert_eq!(subset.identities.len(), 1);
        assert_eq!(subset.aea_keys.len(), 1);
        subset.validate().unwrap();
        let mut two = manifest.clone();
        two.identities[1].purpose = Purpose::Erase;
        assert!(two.erase_subset().is_err());
    }

    #[test]
    fn slices_cover_members_within_the_fat32_limit() {
        assert_eq!(slice_count(0, MAX_SLICE_BYTES), 1);
        assert_eq!(slice_count(MAX_SLICE_BYTES, MAX_SLICE_BYTES), 1);
        assert_eq!(slice_count(MAX_SLICE_BYTES + 1, MAX_SLICE_BYTES), 2);
        // The 26A434 OS image (11,274,289,152 bytes) takes three slices.
        assert_eq!(slice_count(11_274_289_152, MAX_SLICE_BYTES), 3);
        assert_eq!(slice_path("a/b.dmg", 2), "a/b.dmg.002");
        assert_eq!(images_directory("26A434"), "EFI/stela/images/26A434");
    }
}
