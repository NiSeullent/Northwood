//! The restore payload a USB carries to `stela.efi`: one model's IPSW
//! members with their SHA-256s, the global `apticket` manifests and the
//! public AEA key responses, so an installation needs no network.
//!
//! [`inventory`] hashes members where they are. [`write`] lays the payload
//! out under `EFI/stela/images/<build>/`, as FAT32 slices or as the whole
//! IPSW, and writes `payload.json` last, so a volume without it holds no
//! payload. [`remote_inventory`] describes an IPSW `stela.efi` streams
//! from Apple's servers during the install.
use super::{aea, components, manifest};
use anyhow::{Context, Result, ensure};
use serde_json::json;
use sha2::{Digest, Sha256, Sha384};
use std::{
    collections::BTreeSet,
    fs::{self, File},
    io::{Read, Seek, Write},
    path::{Path, PathBuf},
};
use zip::ZipArchive;
use {
    stela_formats::payload::AeaKey, stela_formats::payload::BLOCK_HASHES_FILE,
    stela_formats::payload::BlockHashes, stela_formats::payload::HASH_BLOCK_BYTES,
    stela_formats::payload::IdentitySet, stela_formats::payload::KIND,
    stela_formats::payload::Layout, stela_formats::payload::MANIFEST_FILE,
    stela_formats::payload::Member, stela_formats::payload::PayloadManifest,
    stela_formats::payload::Purpose, stela_formats::payload::SourceImage,
    stela_formats::payload::images_directory, stela_formats::payload::slice_count,
    stela_formats::payload::slice_path,
};

const BUILD_MANIFEST: &str = "BuildManifest.plist";
const RESTORE_PLIST: &str = "Restore.plist";
const BUFFER: usize = 4 * 1024 * 1024;

/// Which members a model needs, before anything is hashed.
struct Selection {
    restore: manifest::RestoreManifest,
    device_class: String,
    identities: Vec<IdentitySet>,
    global_manifests: Vec<String>,
    names: Vec<String>,
}

/// Members outside the BuildManifest that restored asks the host for in any
/// restore: `SourceBootObjectV4` `__RestoreVersion__` and `__SystemVersion__`,
/// and the `BootabilityBundle` tree (every file, link and trust cache under
/// it; directories carry no bytes).
fn restored_members<'a>(names: impl Iterator<Item = &'a str>) -> Vec<String> {
    names
        .filter(|name| {
            matches!(*name, "RestoreVersion.plist" | "SystemVersion.plist")
                || (name.starts_with("BootabilityBundle/") && !name.ends_with('/'))
        })
        .map(str::to_owned)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

/// The archive comment of a recovery medium [`trim`] wrote. Its identities
/// are the profile's erase restore and installed system, without the
/// components only the macOS system volume holds
/// ([`components::system_volume`]).
pub const RECOVERY_COMMENT: &[u8] = b"Stela recovery members";

/// Whether the IPSW at `image` holds the macOS system image an erase
/// restore installs: every IPSW but a recovery archive [`trim`] wrote.
pub fn holds_system_image(image: &Path) -> Result<bool> {
    let archive = ZipArchive::new(File::open(image)?)
        .with_context(|| format!("{} is not an IPSW", image.display()))?;
    Ok(archive.comment() != RECOVERY_COMMENT)
}

fn select<R: Read + Seek>(archive: &mut ZipArchive<R>, model: &str) -> Result<Selection> {
    let recovery = archive.comment() == RECOVERY_COMMENT;
    select_scope(archive, model, recovery)
}

fn select_scope<R: Read + Seek>(
    archive: &mut ZipArchive<R>,
    model: &str,
    recovery: bool,
) -> Result<Selection> {
    let mut bytes = Vec::new();
    archive
        .by_name(BUILD_MANIFEST)?
        .take(64 * 1024 * 1024 + 1)
        .read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() <= 64 * 1024 * 1024,
        "restore manifest exceeds 64 MiB"
    );
    // The profile's identities, as `components` defines them.
    let profile = components::Profile::of(model)?;
    let mut restore = manifest::inspect(&bytes, None)?;
    restore.identities.retain(|identity| {
        profile.owns(identity)
            && (!recovery
                || matches!(
                    components::purpose(identity),
                    Some(Purpose::Erase | Purpose::Installed)
                ))
    });
    ensure!(
        !restore.identities.is_empty(),
        "macOS {} has no identities for {profile}",
        restore.product_build
    );
    let device_class = profile.device_class.clone();
    let restored = restored_members(archive.file_names());
    let identities: Vec<IdentitySet> = restore
        .identities
        .iter()
        .map(|identity| {
            Ok(IdentitySet {
                index: identity.index,
                variant: identity.variant.clone(),
                purpose: components::purpose(identity).with_context(|| {
                    format!(
                        "macOS {} identity {} ({}) has no RestoreBehavior Stela knows",
                        restore.product_build, identity.index, identity.variant
                    )
                })?,
                members: identity
                    .components
                    .iter()
                    .filter(|(name, _)| !recovery || !components::system_volume(name))
                    .map(|(_, component)| component.path.clone())
                    .chain(restored.iter().cloned())
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect(),
            })
        })
        .collect::<Result<_>>()?;
    let suffix = format!("/apticket.{device_class}.im4m");
    let global_manifests: Vec<String> = archive
        .file_names()
        .filter(|name| name.starts_with("Firmware/Manifests/") && name.ends_with(&suffix))
        .map(str::to_owned)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let names: Vec<String> = identities
        .iter()
        .flat_map(|identity| identity.members.iter().cloned())
        .chain(global_manifests.iter().cloned())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    for name in &names {
        ensure!(
            stela_formats::payload::safe_path(name),
            "unsafe member {name}"
        );
        archive
            .by_name(name)
            .with_context(|| format!("the manifest names {name}, which the IPSW lacks"))?;
    }
    Ok(Selection {
        restore,
        device_class,
        identities,
        global_manifests,
        names,
    })
}

/// A member's hashes: its entry, with its SHA-384, and the SHA-256 of each
/// of its [`HASH_BLOCK_BYTES`] blocks, as [`BlockHashes`] lists them.
struct Hashed {
    member: Member,
    blocks: Vec<u8>,
}

/// Whether the caller publishes the per-block digests alongside the manifest.
#[derive(Clone, Copy)]
enum BlockDigests {
    Record,
    Omit,
}

/// Stream one member through `sink`, returning its length and hashes.
fn stream<R: Read + Seek>(
    archive: &mut ZipArchive<R>,
    name: &str,
    block_digests: BlockDigests,
    mut sink: impl FnMut(&[u8]) -> Result<()>,
) -> Result<Hashed> {
    let mut member = archive.by_name(name)?;
    let expected = member.size();
    let mut hash = Sha256::new();
    let mut hash384 = Sha384::new();
    let mut block = match block_digests {
        BlockDigests::Record => Some(Sha256::new()),
        BlockDigests::Omit => None,
    };
    let mut in_block = 0u64;
    let mut blocks = Vec::new();
    let mut total = 0u64;
    let mut buffer = vec![0u8; BUFFER];
    loop {
        let count = member.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        total += count as u64;
        ensure!(
            total <= expected,
            "{name} is longer than its directory entry"
        );
        hash.update(&buffer[..count]);
        hash384.update(&buffer[..count]);
        if let Some(block) = &mut block {
            let mut rest = &buffer[..count];
            while !rest.is_empty() {
                let take = rest.len().min((HASH_BLOCK_BYTES - in_block) as usize);
                block.update(&rest[..take]);
                in_block += take as u64;
                rest = &rest[take..];
                if in_block == HASH_BLOCK_BYTES {
                    blocks.extend_from_slice(&std::mem::take(block).finalize());
                    in_block = 0;
                }
            }
        }
        sink(&buffer[..count])?;
    }
    ensure!(total == expected, "{name} is truncated");
    if let Some(block) = block
        && in_block > 0
    {
        blocks.extend_from_slice(&block.finalize());
    }
    Ok(Hashed {
        member: Member {
            name: name.into(),
            bytes: total,
            sha256: Some(hex::encode(hash.finalize())),
            sha384: Some(hex::encode(hash384.finalize())),
        },
        blocks,
    })
}

/// The block hash file of `hashed`, in their order, and its entry.
fn block_hashes(hashed: &[Hashed]) -> (Vec<u8>, BlockHashes) {
    let file: Vec<u8> = hashed
        .iter()
        .flat_map(|h| h.blocks.iter().copied())
        .collect();
    let entry = BlockHashes {
        file: BLOCK_HASHES_FILE.into(),
        block_bytes: HASH_BLOCK_BYTES,
        sha256: stela_formats::sha256(&file),
    };
    (file, entry)
}

/// Hash the named members on several threads; each opens the IPSW itself.
fn hash_members(
    image: &Path,
    names: &[String],
    block_digests: BlockDigests,
) -> Result<Vec<Hashed>> {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get().min(8));
    let next = std::sync::atomic::AtomicUsize::new(0);
    let results: Vec<Result<Vec<(usize, Hashed)>>> = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(|| -> Result<Vec<(usize, Hashed)>> {
                    let mut archive = ZipArchive::new(File::open(image)?)?;
                    let mut done = Vec::new();
                    loop {
                        let index = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(name) = names.get(index) else {
                            return Ok(done);
                        };
                        done.push((
                            index,
                            stream(&mut archive, name, block_digests, |_| Ok(()))?,
                        ));
                    }
                })
            })
            .collect();
        workers
            .into_iter()
            .map(|worker| worker.join().expect("hash worker panicked"))
            .collect()
    });
    let mut members: Vec<(usize, Hashed)> = Vec::new();
    for result in results {
        members.extend(result?);
    }
    members.sort_by_key(|(index, _)| *index);
    Ok(members.into_iter().map(|(_, member)| member).collect())
}

/// AEA metadata of each `.aea` member, read from the archive in place.
fn aea_metadata<R: Read + Seek>(
    archive: &mut ZipArchive<R>,
    image: &Path,
    names: &[String],
) -> Result<Vec<(String, aea::Metadata)>> {
    let mut found = Vec::new();
    for name in names.iter().filter(|name| name.ends_with(".aea")) {
        let mut member = archive.by_name(name)?;
        let length = member.size();
        let source = image.join(name);
        found.push((
            name.clone(),
            aea::inspect_stream(source, length, &mut member)?,
        ));
    }
    Ok(found)
}

/// Record an AEA member's recipient key: confirm it unwraps the image and
/// record the restore host's offline `URLAsset` answer digest.
fn make_key(
    member: &str,
    metadata: &aea::Metadata,
    url: String,
    key_pem: String,
) -> Result<AeaKey> {
    aea::verify_key(metadata, &key_pem)
        .with_context(|| format!("the AEA key for {member} does not unwrap its image"))?;
    let url_asset_sha256 =
        stela_runtime::restore::answers::staged_system_image_key_digest(&key_pem)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(AeaKey {
        member: member.into(),
        header_sha256: metadata.header_sha256.clone(),
        url,
        key_sha256: stela_formats::sha256(key_pem.as_bytes()),
        url_asset_sha256: Some(url_asset_sha256),
        key_pem,
    })
}

fn fetch_keys(metadata: &[(String, aea::Metadata)]) -> Result<Vec<AeaKey>> {
    metadata
        .iter()
        .map(|(member, metadata)| {
            let key = aea::fetch_public_key(metadata)
                .with_context(|| format!("public AEA key for {member}"))?;
            let key_pem = String::from_utf8(key.body)?;
            make_key(member, metadata, key.request_url, key_pem)
        })
        .collect()
}

/// Read each member's recipient key from a local directory, named by the
/// image's AEA header SHA-256 (the user's offline inputs), instead of
/// fetching it from Apple.
fn local_keys(metadata: &[(String, aea::Metadata)], dir: &Path) -> Result<Vec<AeaKey>> {
    metadata
        .iter()
        .map(|(member, metadata)| {
            let url = metadata
                .public_key_url
                .clone()
                .with_context(|| format!("{member} names no fcs-key-url"))?;
            let path = dir.join(format!("{}.pem", metadata.header_sha256));
            let key_pem = fs::read_to_string(&path)
                .with_context(|| format!("offline AEA key {}", path.display()))?;
            make_key(member, metadata, url, key_pem)
        })
        .collect()
}

/// A file's length and SHA-256, read in pieces.
pub(crate) fn file_sha256(path: &Path) -> Result<(u64, String)> {
    file_sha256_with(path, |_| {})
}

/// [`file_sha256`], telling `read` each piece's length as it is hashed.
fn file_sha256_with(path: &Path, read: impl Fn(u64)) -> Result<(u64, String)> {
    let mut file = File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0u8; BUFFER];
    let mut total = 0u64;
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            return Ok((total, hex::encode(hash.finalize())));
        }
        total += count as u64;
        hash.update(&buffer[..count]);
        read(count as u64);
    }
}

fn manifest_for(
    image: &Path,
    selection: &Selection,
    layout: Layout,
    members: Vec<Member>,
    aea_keys: Vec<AeaKey>,
    (bytes, sha256): (u64, Option<String>),
) -> Result<PayloadManifest> {
    let manifest = PayloadManifest {
        schema_version: 1,
        kind: KIND.into(),
        product_version: selection.restore.product_version.clone(),
        product_build: selection.restore.product_build.clone(),
        model: selection.restore.identities[0]
            .product_type
            .clone()
            .context("identity has no product type")?,
        device_class: selection.device_class.clone(),
        source: SourceImage {
            file_name: image
                .file_name()
                .and_then(|name| name.to_str())
                .context("IPSW path has no UTF-8 file name")?
                .into(),
            bytes,
            sha256,
            build_manifest_sha256: selection.restore.manifest_sha256.clone(),
            catalog_sha1: None,
        },
        layout,
        members,
        identities: selection.identities.clone(),
        global_manifests: selection.global_manifests.clone(),
        aea_keys,
        block_hashes: None,
    };
    manifest
        .validate()
        .map_err(|error| anyhow::anyhow!("{error}"))?;
    Ok(manifest)
}

/// An IPSW's restore inventory, read where it is.
pub struct Inventory {
    /// The `image-manifest --payload` report.
    pub report: serde_json::Value,
    /// The archive manifest, carrying the fetched public AEA keys and
    /// naming the block hash file.
    pub manifest: PayloadManifest,
    /// The block hash file (`BLOCK_HASHES_FILE`), written beside the
    /// manifest so the restore host needs no scan of the payload.
    pub blocks: Vec<u8>,
}

/// The `image-manifest --payload` report: member sets with SHA-256s, global
/// manifests and each AEA member's key URL. The options select local or fetched
/// recipient keys and optional 0600 copies named by header SHA-256. Key bodies
/// are omitted from the report. The returned manifest binds the in-place IPSW
/// and each member's block hashes.
pub fn inventory(image: &Path, model: &str, keys: KeyOptions<'_>) -> Result<Inventory> {
    let mut archive = ZipArchive::new(File::open(image)?)?;
    let selection = select(&mut archive, model)?;
    let metadata = aea_metadata(&mut archive, image, &selection.names)?;
    // Hashing a whole IPSW takes minutes from a slow disk: its progress is
    // the whole file's, which the members are read beside.
    let progress = crate::progress::Phase::bytes(
        "Checking the restore image",
        std::fs::metadata(image)?.len(),
    );
    let (ipsw, members) = std::thread::scope(|scope| {
        let whole = scope.spawn(|| file_sha256_with(image, |bytes| progress.advance(bytes)));
        let members = hash_members(image, &selection.names, BlockDigests::Record);
        (whole.join().expect("hash worker panicked"), members)
    });
    progress.finish();
    let hashed = members?;
    finish_inventory(image, selection, metadata, hashed, ipsw?, keys)
}

/// Identity and verified whole-file digest of a joined IPSW reader.
pub(crate) struct VerifiedSource<'a> {
    pub(crate) image: &'a Path,
    pub(crate) model: &'a str,
    pub(crate) digest: (u64, String),
}

/// Inventory a verified, joined IPSW without creating a full host-side file.
pub(crate) fn inventory_from<R: Read + Seek>(
    source: R,
    identity: VerifiedSource<'_>,
    complete_verification: impl FnOnce(R) -> Result<()>,
) -> Result<Inventory> {
    let VerifiedSource {
        image,
        model,
        digest,
    } = identity;
    let mut archive = ZipArchive::new(source)?;
    let selection = select(&mut archive, model)?;
    let metadata = aea_metadata(&mut archive, image, &selection.names)?;
    let total = selection
        .names
        .iter()
        .map(|name| Ok(archive.by_name(name)?.size()))
        .collect::<Result<Vec<u64>>>()?
        .iter()
        .sum();
    let progress = crate::progress::Phase::bytes("Checking macOS files", total);
    let hashed = selection
        .names
        .iter()
        .map(|name| {
            stream(&mut archive, name, BlockDigests::Record, |bytes| {
                progress.advance(bytes.len() as u64);
                Ok(())
            })
        })
        .collect::<Result<Vec<_>>>()?;
    progress.finish();
    // The medium must finish readback before requesting the AEA keys.
    complete_verification(archive.into_inner())?;
    finish_inventory(
        image,
        selection,
        metadata,
        hashed,
        digest,
        KeyOptions {
            fetch: true,
            output: None,
            input: None,
        },
    )
}

/// Public AEA recipient key sources and optional owner-only output copies.
/// Requesting output also fetches keys when no input directory is supplied.
#[derive(Default)]
pub struct KeyOptions<'a> {
    /// Fetch public keys unless an input directory is supplied.
    pub fetch: bool,
    /// Write key copies here with owner-only permissions.
    pub output: Option<&'a Path>,
    /// Read keys here instead of fetching them.
    pub input: Option<&'a Path>,
}

fn finish_inventory(
    image: &Path,
    selection: Selection,
    metadata: Vec<(String, aea::Metadata)>,
    hashed: Vec<Hashed>,
    digest: (u64, String),
    options: KeyOptions<'_>,
) -> Result<Inventory> {
    let (blocks, block_entry) = block_hashes(&hashed);
    let mut keys = Vec::new();
    // Prefer the user's offline key directory; fetch from Apple only when
    // asked and no offline keys are given.
    let aea_keys = if let Some(dir) = options.input {
        local_keys(&metadata, dir)?
    } else if options.fetch || options.output.is_some() {
        self::fetch_keys(&metadata)?
    } else {
        Vec::new()
    };
    if let Some(directory) = options.output {
        fs::create_dir_all(directory)?;
        for key in &aea_keys {
            let path = directory.join(format!("{}.pem", key.header_sha256));
            if !path.exists() {
                let mut file = tempfile::NamedTempFile::new_in(directory)?;
                file.write_all(key.key_pem.as_bytes())?;
                file.as_file().sync_all()?;
                file.persist_noclobber(&path).map_err(|error| error.error)?;
            }
            keys.push(
                json!({"member": key.member, "header_sha256": key.header_sha256,
                "url": key.url, "key_sha256": key.key_sha256, "file": path}),
            );
        }
    }
    let manifest = manifest_for(
        image,
        &selection,
        Layout::Archive {
            path: images_directory(&selection.restore.product_build)
                + "/"
                + &image.file_name().unwrap().to_string_lossy(),
        },
        hashed.into_iter().map(|h| h.member).collect(),
        Vec::new(),
        (digest.0, Some(digest.1)),
    )?;
    let keyed = PayloadManifest {
        aea_keys,
        block_hashes: Some(block_entry),
        ..manifest.clone()
    };
    keyed
        .validate()
        .map_err(|error| anyhow::anyhow!("{error}"))?;
    let report = json!({
        "payload": manifest,
        "aea": metadata.iter().map(|(member, metadata)| json!({
            "member": member, "header_sha256": metadata.header_sha256,
            "public_key_url": metadata.public_key_url})).collect::<Vec<_>>(),
        "aea_keys": keys,
        "limits": "Member SHA-256s are of the bytes in this IPSW; the BuildManifest's own digests are opaque and not reinterpreted. Key bodies are omitted from this report.",
    });
    Ok(Inventory {
        report,
        manifest: keyed,
        blocks,
    })
}

/// Remote IPSW identity, source layout and optional local recipient keys.
pub struct RemoteInventoryOptions<'a> {
    /// IPSW name recorded in source and AEA member identities.
    pub image: &'a Path,
    /// Whole archive length reported by the range transport.
    pub bytes: u64,
    /// Remote URL and ETag used by the restore host.
    pub layout: Layout,
    /// Product type selecting the restore identity.
    pub model: &'a str,
    /// Local recipient keys; absence requests public key fetching.
    pub keys_in: Option<&'a Path>,
}

/// The restore inventory of the IPSW `source` of `bytes` named `image`,
/// read by ranges where it is, for `stela.efi` to stream from `layout`'s
/// URL during the install ([`Layout::Remote`]). Only members no device
/// check covers are hashed (the restored plists, the bootability bundle,
/// the global manifests): an Image4 component's digest is in the
/// personalized ticket and an AEA archive authenticates itself, so those
/// pass through unhashed, and the sealed system volume checks itself.
pub fn remote_inventory<R: Read + Seek>(
    source: R,
    options: RemoteInventoryOptions<'_>,
) -> Result<Inventory> {
    let RemoteInventoryOptions {
        image,
        bytes,
        layout,
        model,
        keys_in,
    } = options;
    ensure!(
        matches!(layout, Layout::Remote { .. }),
        "a remote inventory needs a remote layout"
    );
    let mut archive = ZipArchive::new(source)?;
    let selection = select(&mut archive, model)?;
    let metadata = aea_metadata(&mut archive, image, &selection.names)?;
    let verified: BTreeSet<&str> = selection
        .restore
        .identities
        .iter()
        .flat_map(|identity| identity.components.values())
        .filter(|component| component.declared_digest_hex.is_some())
        .map(|component| component.path.as_str())
        .chain(metadata.iter().map(|(name, _)| name.as_str()))
        .collect();
    let mut members = Vec::new();
    let mut hashed = Vec::new();
    for name in &selection.names {
        if verified.contains(name.as_str()) {
            members.push(Member {
                name: name.clone(),
                bytes: archive.by_name(name)?.size(),
                sha256: None,
                sha384: None,
            });
        } else {
            let member = stream(&mut archive, name, BlockDigests::Record, |_| Ok(()))?;
            members.push(member.member.clone());
            hashed.push(member);
        }
    }
    let (blocks, block_entry) = block_hashes(&hashed);
    let aea_keys = match keys_in {
        Some(dir) => local_keys(&metadata, dir)?,
        None => fetch_keys(&metadata)?,
    };
    let manifest = PayloadManifest {
        aea_keys,
        block_hashes: Some(block_entry),
        ..manifest_for(
            image,
            &selection,
            layout,
            members,
            Vec::new(),
            (bytes, None),
        )?
    };
    manifest
        .validate()
        .map_err(|error| anyhow::anyhow!("{error}"))?;
    let report = json!({
        "payload": manifest,
        "hashed": hashed.len(),
        "limits": "Members without a SHA-256 are verified by the device: Image4 components by the personalized ticket, AEA archives by their own authentication.",
    });
    Ok(Inventory {
        report,
        manifest,
        blocks,
    })
}

/// The product version and build of an IPSW's restore identity for
/// `model`, from its ZIP directory and `Restore.plist` alone.
pub fn identity(image: &Path, model: &str) -> Result<(String, String)> {
    let mut archive = ZipArchive::new(File::open(image)?)?;
    let selection = select(&mut archive, model)?;
    Ok((
        selection.restore.product_version,
        selection.restore.product_build,
    ))
}

/// Copy the recovery members of the IPSW `source` for `model` into a new
/// archive at `output`, with [`RECOVERY_COMMENT`] as its comment. Every
/// member is stored uncompressed, so `stela.efi` reads any range of one
/// directly; a deflated member is inflated here once, its CRC-32 checked as
/// it is read. `source` may be a local file or a remote one read by ranges;
/// only the members copied are read. The archive is written beside
/// `output` and renamed when complete, so a partial one is never taken for
/// a medium. Returns the bytes written for the members.
pub fn trim<R: Read + Seek>(source: R, model: &str, output: &Path, what: &str) -> Result<u64> {
    let mut archive = ZipArchive::new(source)?;
    let (names, total) = recovery_members(&mut archive, model)?;
    let names: Vec<&str> = names.iter().map(String::as_str).collect();
    let directory = output.parent().unwrap_or(Path::new("."));
    crate::setup::ensure_room(directory, what, total)?;
    let progress = crate::progress::Phase::bytes(what, total);
    let part = output.with_extension("part");
    // A partial archive cannot be resumed: it is removed on any failure.
    let written = copy_members(&mut archive, &names, &part, &progress).inspect_err(|_| {
        let _ = fs::remove_file(&part);
    })?;
    fs::rename(&part, output)?;
    progress.finish();
    Ok(written)
}

/// The members [`trim`] copies from `archive` for `model`, and their bytes.
fn recovery_members<R: Read + Seek>(
    archive: &mut ZipArchive<R>,
    model: &str,
) -> Result<(Vec<String>, u64)> {
    let selection = select_scope(archive, model, true)?;
    let mut names = vec![BUILD_MANIFEST.to_owned()];
    if archive.index_for_name(RESTORE_PLIST).is_some() {
        names.push(RESTORE_PLIST.into());
    }
    names.extend(selection.names);
    let total = names.iter().try_fold(0u64, |total, name| -> Result<u64> {
        total
            .checked_add(archive.by_name(name)?.size())
            .context("boot archive size overflow")
    })?;
    Ok((names, total))
}

/// The bytes [`trim`] would copy from the IPSW `source` for `model`, read
/// from its ZIP directory alone; and the whole IPSW's.
pub fn recovery_bytes<R: Read + Seek>(mut source: R, model: &str) -> Result<(u64, u64)> {
    let whole = source.seek(std::io::SeekFrom::End(0))?;
    let mut archive = ZipArchive::new(source)?;
    Ok((recovery_members(&mut archive, model)?.1, whole))
}

/// Copy `names` of `archive` into a new stored archive `part`.
fn copy_members<R: Read + Seek>(
    archive: &mut ZipArchive<R>,
    names: &[&str],
    part: &Path,
    progress: &crate::progress::Phase,
) -> Result<u64> {
    use zip::{CompressionMethod, write::SimpleFileOptions};
    let mut writer = zip::ZipWriter::new(File::create(part)?);
    let mut written = 0;
    for &name in names {
        let index = archive
            .index_for_name(name)
            .with_context(|| format!("the IPSW lacks {name}"))?;
        let mut member = archive.by_index(index)?;
        let options = SimpleFileOptions::default()
            .compression_method(CompressionMethod::Stored)
            .large_file(member.size() >= u32::MAX as u64)
            .last_modified_time(member.last_modified().unwrap_or_default());
        writer.start_file(name, options)?;
        let mut count = 0u64;
        let mut buffer = vec![0; 1024 * 1024];
        loop {
            let size = member
                .read(&mut buffer)
                .with_context(|| format!("read {name}"))?;
            if size == 0 {
                break;
            }
            writer.write_all(&buffer[..size])?;
            count += size as u64;
            progress.advance(size as u64);
        }
        ensure!(count == member.size(), "{name} is truncated");
        written += count;
    }
    writer.set_raw_comment(RECOVERY_COMMENT.into());
    writer.finish()?.sync_all()?;
    Ok(written)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum PayloadLayout {
    /// FAT32-safe slices of each member.
    Split,
    /// The whole IPSW, for volumes without FAT32's 4 GiB limit.
    Archive,
}

pub struct WriteOptions<'a> {
    pub image: &'a Path,
    pub model: &'a str,
    pub root: &'a Path,
    pub layout: PayloadLayout,
    pub slice_bytes: u64,
    pub fetch_keys: bool,
}

/// Write the payload under `root` and return a report.
pub fn write(options: &WriteOptions<'_>) -> Result<serde_json::Value> {
    let image = options.image;
    let mut archive = ZipArchive::new(File::open(image)?)?;
    let selection = select(&mut archive, options.model)?;
    let directory = options
        .root
        .join(images_directory(&selection.restore.product_build));
    let manifest_path = directory.join(MANIFEST_FILE);
    ensure!(
        !manifest_path.exists(),
        "{} exists; remove the old payload first",
        manifest_path.display()
    );
    let metadata = aea_metadata(&mut archive, image, &selection.names)?;
    let aea_keys = if options.fetch_keys {
        fetch_keys(&metadata)?
    } else {
        Vec::new()
    };
    fs::create_dir_all(&directory)?;
    let (layout, members, whole) = match options.layout {
        PayloadLayout::Split => {
            ensure!(
                (1..=stela_formats::payload::MAX_SLICE_BYTES).contains(&options.slice_bytes),
                "slice size must be 1..=4 GiB-1"
            );
            let whole = std::thread::scope(|scope| {
                let whole = scope.spawn(|| file_sha256(image));
                let members = selection
                    .names
                    .iter()
                    .map(|name| write_slices(&mut archive, name, &directory, options.slice_bytes))
                    .collect::<Result<Vec<_>>>();
                (whole.join().expect("hash worker panicked"), members)
            });
            (
                Layout::Split {
                    slice_bytes: options.slice_bytes,
                },
                whole.1?,
                whole.0?,
            )
        }
        PayloadLayout::Archive => {
            let name = image.file_name().context("IPSW path has no file name")?;
            let copy = directory.join(name);
            ensure!(!copy.exists(), "{} exists", copy.display());
            let whole = copy_hashed(image, &copy)?;
            let members = hash_members(&copy, &selection.names, BlockDigests::Omit)?
                .into_iter()
                .map(|h| h.member)
                .collect();
            let path = format!(
                "{}/{}",
                images_directory(&selection.restore.product_build),
                name.to_str().context("IPSW file name is not UTF-8")?
            );
            (Layout::Archive { path }, members, whole)
        }
    };
    let manifest = manifest_for(
        image,
        &selection,
        layout,
        members,
        aea_keys,
        (whole.0, Some(whole.1)),
    )?;
    let mut file = tempfile::NamedTempFile::new_in(&directory)?;
    file.write_all(manifest.to_json().as_bytes())?;
    file.write_all(b"\n")?;
    file.as_file().sync_all()?;
    file.persist_noclobber(&manifest_path)
        .map_err(|error| error.error)?;
    Ok(json!({
        "manifest": manifest_path,
        "build": manifest.product_build,
        "model": manifest.model,
        "members": manifest.members.len(),
        "bytes": manifest.total_bytes(),
        "erase_bytes": manifest.erase_subset().map_err(|e| anyhow::anyhow!("{e}"))?.total_bytes(),
        "aea_keys": manifest.aea_keys.len(),
        "aea_members": metadata.len(),
        "ipsw_sha256": manifest.source.sha256,
    }))
}

fn write_slices(
    archive: &mut ZipArchive<File>,
    name: &str,
    directory: &Path,
    slice_bytes: u64,
) -> Result<Member> {
    let base = directory.join(name);
    fs::create_dir_all(base.parent().expect("member path has a parent"))?;
    let expected = archive.by_name(name)?.size();
    let count = slice_count(expected, slice_bytes);
    let paths: Vec<PathBuf> = (0..count)
        .map(|index| directory.join(slice_path(name, index)))
        .collect();
    for path in &paths {
        ensure!(!path.exists(), "{} exists", path.display());
    }
    let mut index = 0usize;
    let mut file = File::create(&paths[0])?;
    let mut in_slice = 0u64;
    let member = stream(archive, name, BlockDigests::Omit, |mut chunk| {
        while !chunk.is_empty() {
            if in_slice == slice_bytes {
                file.sync_all()?;
                index += 1;
                file = File::create(&paths[index])?;
                in_slice = 0;
            }
            let take = chunk.len().min((slice_bytes - in_slice) as usize);
            file.write_all(&chunk[..take])?;
            in_slice += take as u64;
            chunk = &chunk[take..];
        }
        Ok(())
    })?;
    file.sync_all()?;
    ensure!(
        index + 1 == paths.len(),
        "{name} ended before its last slice"
    );
    Ok(member.member)
}

fn copy_hashed(from: &Path, to: &Path) -> Result<(u64, String)> {
    let mut input = File::open(from)?;
    let mut output = File::create_new(to)?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0u8; BUFFER];
    let mut total = 0u64;
    loop {
        let count = input.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
        output.write_all(&buffer[..count])?;
        total += count as u64;
    }
    output.sync_all()?;
    Ok((total, hex::encode(hash.finalize())))
}

/// The synthetic IPSW's model, which no real BuildManifest uses.
pub const SYNTHETIC_MODEL: &str = "StelaTest1,1";
pub const SYNTHETIC_BUILD: &str = "STELA1";

/// Write a small IPSW-shaped archive with no Apple bytes, for tests and the
/// QEMU payload scenario. It has an Erase, an Update and an installed
/// identity for [`SYNTHETIC_MODEL`], one identity for another model, a
/// global manifest, stored and deflate members, and one ZIP64 member.
pub fn synthetic_ipsw(path: &Path) -> Result<()> {
    synthetic_ipsw_of(path, SYNTHETIC_BUILD)
}

/// [`synthetic_ipsw`] of another build, as a later macOS release would be.
pub fn synthetic_ipsw_of(path: &Path, build: &str) -> Result<()> {
    use zip::{CompressionMethod, write::SimpleFileOptions};
    let identity =
        |model: &str, class: &str, variant: &str, behavior: &str, members: &[(&str, &str)]| {
            let entries: String = members
                .iter()
                .map(|(component, path)| {
                    format!(
                        "<key>{component}</key><dict><key>Digest</key><data>AAECAw==</data>\
                     <key>Info</key><dict><key>Path</key><string>{path}</string></dict></dict>"
                    )
                })
                .collect();
            let (_, chip, board, synthetic_class) = components::SYNTHETIC;
            let board = if class == synthetic_class {
                board
            } else {
                board + 1
            };
            format!(
                "<dict><key>Ap,ProductType</key><string>{model}</string>\
             <key>ApChipID</key><string>{chip:#x}</string>\
             <key>ApBoardID</key><string>{board:#x}</string><key>Info</key><dict>\
             <key>DeviceClass</key><string>{class}</string><key>Variant</key>\
             <string>{variant}</string><key>RestoreBehavior</key><string>{behavior}</string>\
             </dict><key>Manifest</key><dict>{entries}</dict></dict>"
            )
        };
    let kernel = ("KernelCache", "kernelcache.release.stelatest");
    let os = ("OS", "Firmware/stelatest-os.dmg");
    let ramdisk = ("RestoreRamDisk", "stelatest-ramdisk.dmg");
    let update = ("RestoreRamDisk", "stelatest-update.dmg");
    let identities = [
        identity(
            "Other1,1",
            "otherap",
            "Customer Erase Install (IPSW)",
            "Erase",
            &[kernel, ramdisk],
        ),
        identity(
            SYNTHETIC_MODEL,
            "stelatestap",
            "Customer Erase Install (IPSW)",
            "Erase",
            &[kernel, os, ramdisk],
        ),
        identity(
            SYNTHETIC_MODEL,
            "stelatestap",
            "Customer Upgrade Install (IPSW)",
            "Update",
            &[kernel, update],
        ),
        identity(
            SYNTHETIC_MODEL,
            "stelatestap",
            "macOS Customer",
            "Erase",
            &[kernel],
        ),
    ]
    .concat();
    let build_manifest = format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?><plist version=\"1.0\"><dict>\
         <key>ProductVersion</key><string>0.0.1</string>\
         <key>ProductBuildVersion</key><string>{build}</string>\
         <key>SupportedProductTypes</key><array><string>{SYNTHETIC_MODEL}</string>\
         <string>Other1,1</string></array>\
         <key>BuildIdentities</key><array>{identities}</array></dict></plist>"
    );
    let os_bytes: Vec<u8> = (0..1_500_000u32)
        .map(|i| ((i / 7) ^ (i >> 11) ^ (i >> 17)) as u8)
        .collect();
    let mut writer = zip::ZipWriter::new(File::create_new(path)?);
    let stored = SimpleFileOptions::default().compression_method(CompressionMethod::Stored);
    let deflated = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
    let members: [(&str, Vec<u8>, SimpleFileOptions); 10] = [
        (BUILD_MANIFEST, build_manifest.into_bytes(), deflated),
        (kernel.1, vec![0x5a; 70_000], stored.large_file(true)),
        (os.1, os_bytes, deflated),
        (ramdisk.1, b"restore ramdisk".to_vec(), stored),
        (update.1, b"update ramdisk".to_vec(), stored),
        (
            "Firmware/Manifests/restore/Customer/apticket.stelatestap.im4m",
            b"global".to_vec(),
            stored,
        ),
        ("Firmware/unrelated.bin", vec![1; 10], stored),
        ("RestoreVersion.plist", b"restore version".to_vec(), stored),
        ("BootabilityBundle/Restore/Bootability/", Vec::new(), stored),
        (
            "BootabilityBundle/Restore/Bootability/brain",
            b"bootability".to_vec(),
            deflated,
        ),
    ];
    for (name, bytes, options) in members {
        writer.start_file(name, options)?;
        writer.write_all(&bytes)?;
    }
    writer.finish()?.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use stela_formats::payload::Purpose;

    fn options<'a>(image: &'a Path, root: &'a Path, layout: PayloadLayout) -> WriteOptions<'a> {
        WriteOptions {
            image,
            model: SYNTHETIC_MODEL,
            root,
            layout,
            slice_bytes: 400_000,
            fetch_keys: true,
        }
    }

    #[test]
    fn split_payloads_slice_members_and_record_every_identity() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let image = dir.path().join("Synthetic.ipsw");
        synthetic_ipsw(&image)?;
        let root = dir.path().join("usb");
        let report = write(&options(&image, &root, PayloadLayout::Split))?;
        assert_eq!(report["members"], 7);
        let directory = root.join(images_directory(SYNTHETIC_BUILD));
        let manifest = PayloadManifest::parse(&fs::read(directory.join(MANIFEST_FILE))?).unwrap();
        assert_eq!(manifest.model, SYNTHETIC_MODEL);
        assert_eq!(manifest.device_class, "stelatestap");
        let purposes: Vec<(u32, Purpose)> = manifest
            .identities
            .iter()
            .map(|i| (i.index, i.purpose))
            .collect();
        assert_eq!(
            purposes,
            [
                (1, Purpose::Erase),
                (2, Purpose::Update),
                (3, Purpose::Installed)
            ]
        );
        assert!(manifest.block_hashes.is_none());
        assert_eq!(manifest.global_manifests.len(), 1);
        assert!(manifest.aea_keys.is_empty());
        assert_eq!(manifest.source.sha256, Some(file_sha256(&image)?.1));
        // Each member's slices join back to its hashed bytes.
        let mut archive = ZipArchive::new(File::open(&image)?)?;
        for member in &manifest.members {
            let mut joined = Vec::new();
            for index in 0..slice_count(member.bytes, 400_000) {
                let slice = fs::read(directory.join(slice_path(&member.name, index)))?;
                assert!(slice.len() as u64 <= 400_000);
                joined.extend(slice);
            }
            let mut original = Vec::new();
            archive.by_name(&member.name)?.read_to_end(&mut original)?;
            assert_eq!(joined, original);
            assert_eq!(Some(stela_formats::sha256(&joined)), member.sha256);
            assert_eq!(Some(hex::encode(Sha384::digest(&joined))), member.sha384);
        }
        assert!(directory.join("Firmware/stelatest-os.dmg.003").exists());
        assert!(!directory.join("Firmware/unrelated.bin.000").exists());
        // Every identity carries what restored asks for outside the manifest.
        for identity in &manifest.identities {
            for name in [
                "RestoreVersion.plist",
                "BootabilityBundle/Restore/Bootability/brain",
            ] {
                assert!(identity.members.iter().any(|m| m == name), "{name}");
            }
            assert!(!identity.members.iter().any(|m| m.ends_with('/')));
        }
        // A second write never replaces a payload.
        assert!(write(&options(&image, &root, PayloadLayout::Split)).is_err());
        Ok(())
    }

    #[test]
    fn trimmed_archive_preserves_recovery_inventory() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let image = dir.path().join("Synthetic.ipsw");
        synthetic_ipsw(&image)?;
        let trimmed = dir.path().join("Recovery.ipsw");
        let copied = trim(
            File::open(&image)?,
            SYNTHETIC_MODEL,
            &trimmed,
            "Copying boot files",
        )?;
        assert!(copied > 0 && !trimmed.with_extension("part").exists());
        let mut archive = ZipArchive::new(File::open(&trimmed)?)?;
        assert_eq!(archive.comment(), RECOVERY_COMMENT);
        let names: BTreeSet<String> = archive.file_names().map(str::to_owned).collect();
        for name in [
            BUILD_MANIFEST,
            "kernelcache.release.stelatest",
            "stelatest-ramdisk.dmg",
            "RestoreVersion.plist",
            "Firmware/Manifests/restore/Customer/apticket.stelatestap.im4m",
        ] {
            assert!(names.contains(name), "{name}");
        }
        // The system image, the update ramdisk and other files stay behind.
        for name in [
            "Firmware/stelatest-os.dmg",
            "stelatest-update.dmg",
            "Firmware/unrelated.bin",
        ] {
            assert!(!names.contains(name), "{name}");
        }
        // Every member is stored, with the source's bytes.
        let mut original = ZipArchive::new(File::open(&image)?)?;
        for name in &names {
            let mut kept = Vec::new();
            let mut member = archive.by_name(name)?;
            assert_eq!(
                member.compression(),
                zip::CompressionMethod::Stored,
                "{name}"
            );
            member.read_to_end(&mut kept)?;
            let mut source = Vec::new();
            original.by_name(name)?.read_to_end(&mut source)?;
            assert_eq!(kept, source, "{name}");
        }
        // Its inventory names only the erase and installed identities.
        let recovery = inventory(&trimmed, SYNTHETIC_MODEL, KeyOptions::default())?;
        let purposes: Vec<Purpose> = recovery
            .manifest
            .identities
            .iter()
            .map(|i| i.purpose)
            .collect();
        assert_eq!(purposes, [Purpose::Erase, Purpose::Installed]);
        assert!(
            !recovery
                .manifest
                .members
                .iter()
                .any(|m| m.name == "Firmware/stelatest-os.dmg")
        );
        // A full IPSW's inventory is unchanged.
        let full = inventory(&image, SYNTHETIC_MODEL, KeyOptions::default())?;
        assert_eq!(full.manifest.identities.len(), 3);
        Ok(())
    }

    #[test]
    fn archive_payloads_copy_the_ipsw_and_inventory_hashes_in_place() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let image = dir.path().join("Synthetic.ipsw");
        synthetic_ipsw(&image)?;
        let root = dir.path().join("ntfs");
        write(&options(&image, &root, PayloadLayout::Archive))?;
        let directory = root.join(images_directory(SYNTHETIC_BUILD));
        let manifest = PayloadManifest::parse(&fs::read(directory.join(MANIFEST_FILE))?).unwrap();
        assert_eq!(
            manifest.layout,
            Layout::Archive {
                path: format!("{}/Synthetic.ipsw", images_directory(SYNTHETIC_BUILD))
            }
        );
        assert_eq!(
            fs::read(directory.join("Synthetic.ipsw"))?,
            fs::read(&image)?
        );
        let Inventory {
            report,
            manifest: in_place,
            blocks,
        } = inventory(
            &image,
            SYNTHETIC_MODEL,
            KeyOptions {
                fetch: true,
                output: None,
                input: None,
            },
        )?;
        assert!(manifest.block_hashes.is_none());
        assert_eq!(report["payload"]["members"], json!(manifest.members));
        // The in-place manifest describes the same archive bytes.
        assert_eq!(in_place.members, manifest.members);
        assert_eq!(in_place.source, manifest.source);
        // Its block hashes cover every member's blocks, in order.
        let entry = in_place.block_hashes.as_ref().unwrap();
        assert_eq!(entry.sha256, stela_formats::sha256(&blocks));
        assert_eq!(blocks.len() as u64, in_place.hash_blocks() * 32);
        let mut archive = ZipArchive::new(File::open(&image)?)?;
        let mut at = 0;
        for member in &in_place.members {
            let mut bytes = Vec::new();
            archive.by_name(&member.name)?.read_to_end(&mut bytes)?;
            assert_eq!(
                member.sha384.as_deref(),
                Some(hex::encode(Sha384::digest(&bytes)).as_str())
            );
            for block in bytes.chunks(HASH_BLOCK_BYTES as usize) {
                assert_eq!(blocks[at..at + 32], Sha256::digest(block)[..]);
                at += 32;
            }
        }
        assert_eq!(at, blocks.len());
        assert!(inventory(&image, "Missing1,1", KeyOptions::default()).is_err());
        Ok(())
    }

    #[test]
    fn remote_inventory_hashes_uncovered_members() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let image = dir.path().join("Synthetic.ipsw");
        synthetic_ipsw(&image)?;
        let bytes = fs::metadata(&image)?.len();
        let layout = Layout::Remote {
            url: "http://example.test/Synthetic.ipsw".into(),
            etag: "\"e\"".into(),
        };
        let remote = remote_inventory(
            File::open(&image)?,
            RemoteInventoryOptions {
                image: &image,
                bytes,
                layout: layout.clone(),
                model: SYNTHETIC_MODEL,
                keys_in: None,
            },
        )?;
        let manifest = &remote.manifest;
        assert_eq!(manifest.layout, layout);
        assert_eq!(
            (manifest.source.bytes, &manifest.source.sha256),
            (bytes, &None)
        );
        let full = inventory(&image, SYNTHETIC_MODEL, KeyOptions::default())?.manifest;
        let names =
            |m: &PayloadManifest| m.members.iter().map(|m| m.name.clone()).collect::<Vec<_>>();
        assert_eq!(names(manifest), names(&full));
        // The components the BuildManifest digests pass through; the rest
        // keep the in-place inventory's hashes.
        for (member, hashed) in manifest.members.iter().zip(&full.members) {
            let component = [
                "kernelcache.release.stelatest",
                "Firmware/stelatest-os.dmg",
                "stelatest-ramdisk.dmg",
                "stelatest-update.dmg",
            ]
            .contains(&member.name.as_str());
            assert_eq!(member.bytes, hashed.bytes);
            if component {
                assert_eq!(
                    (&member.sha256, &member.sha384),
                    (&None, &None),
                    "{}",
                    member.name
                );
            } else {
                assert_eq!(member, hashed);
            }
        }
        assert_eq!(remote.blocks.len() as u64, manifest.hash_blocks() * 32);
        assert!(
            remote_inventory(
                File::open(&image)?,
                RemoteInventoryOptions {
                    image: &image,
                    bytes,
                    layout: Layout::Staged,
                    model: SYNTHETIC_MODEL,
                    keys_in: None
                }
            )
            .is_err()
        );
        Ok(())
    }
}
