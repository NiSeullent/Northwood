//! The kernel boot bundle of an installed system, built from its APFS
//! container's Preboot volume (`stela_formats::preboot`), in `no_std` so
//! `stela.efi` builds the same bundle `stela-dev boot prepare --disk` writes.
//!
//! The volume group is the one NVRAM's boot volume names
//! (`stela_loader::boot_selection`), else any. Within a group the boot
//! directory is named as LLB names it: by the LocalPolicy's `nsih`
//! (`stela_loader::local_policy`), and when no such directory exists by
//! `boot/active`; a `--boot-hash` names it instead. The LocalPolicy's
//! signature is the SEP's, so without a [`PolicyAuthority`] its `nsih` only
//! narrows the choice. Whatever names it, a candidate
//! `<volume-group>/boot/<hash>/` checks by evidence when all three hold,
//! and exactly one candidate must:
//!
//! - its Image4 objects carry one manifest whose SHA-384 is the directory's
//!   name: `nsih` is the SHA-384 of the next stage's Image4 manifest (Apple
//!   Platform Security, "Contents of a LocalPolicy file");
//! - its `isys` object, the System volume's root hash, has the digest that
//!   manifest records for `isys` (SHA-384 of the `IM4P`, as the 26A434
//!   BuildManifest's `SystemVolume` digest is of its `.root_hash` file);
//! - the hash in that object equals the seal of the file system the volume
//!   group's System volume roots from (`apfs_root_to_xid`). The payload is
//!   the 208-byte metadata-hash record `apfs_prepare_cryptex -M` writes:
//!   `u32` version 2, `u32` 0, `u32` hash type (`apfs_hash_type_t`), `u32`
//!   hash length, then the hash; it matches the APFS seal for the IPSW's
//!   sealed AppCryptex and for volumes sealed by that tool.
//!
//! Otherwise every candidate is refused with what failed.
//!
//! The boot objects are the Image4 files of the boot directory, chosen by
//! payload type; each is split into its payload, decoded unchanged, and its
//! personalized manifest. The kernel collection's manifest is the boot
//! manifest. The product version and build come from the volume group's
//! `SystemVersion.plist` on Preboot, its Cryptex1's when it has one, the model
//! and board name from the DeviceTree, and chip and board identifiers from
//! the boot manifest.
use crate::aux_kc::{self, Kext, Linked};
use crate::{Error, Result, Within};
use crate::{
    apple_device_tree::DeviceTree,
    boot_bundle::{self, Bundle, Input, Inputs, KernelInput, Options, RootHash, Source, Target},
    boot_policy::{KextGrant, SealedVolume},
    boot_selection::{BootVolume, VolumeType, uuid_text},
    im4m, im4p,
    local_policy::{LocalPolicy, PolicyAuthority},
    root_selection::Root,
};
use alloc::{format, string::String, vec::Vec};
use stela_contracts::{refuse, require};
use {
    stela_formats::apfs,
    stela_formats::apfs::Container,
    stela_formats::apfs::Volume,
    stela_formats::apfs::btree::Hash,
    stela_formats::img4,
    stela_formats::payload::ReadAt,
    stela_formats::preboot,
    stela_formats::preboot::BootSet,
    stela_formats::preboot::{CRYPTEX_SYSTEM_VERSION, SYSTEM_VERSION},
    stela_formats::sha256,
};

/// The variant recorded for a bundle read from an installed system.
pub const INSTALLED_SYSTEM: &str = "installed system (Preboot)";
pub const KERNEL_LIMIT: u64 = 512 * 1024 * 1024;
pub const COMPONENT_LIMIT: u64 = 64 * 1024 * 1024;
const SEP_FIRMWARE_TYPE: &[u8; 4] = b"sepi";
/// The System volume's root hash, as an Image4 object.
const ROOT_HASH_TYPE: &[u8; 4] = b"isys";
/// The BaseSystem's root hash, as an Image4 object.
const BASE_SYSTEM_ROOT_HASH_TYPE: &[u8; 4] = b"csys";
const ROOT_HASH_RECORD_VERSION: u32 = 2;
/// The record's header: version, reserved, hash type and hash length.
const ROOT_HASH_HEADER_BYTES: usize = 16;

/// Why a boot directory is, or is not, the installed system's.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Evidence {
    pub directory: String,
    /// SHA-384 of the manifest the directory's objects carry.
    pub manifest_sha384: Option<String>,
    /// The root hash its `isys` object records.
    pub root_hash: Option<String>,
    /// The seal root of its System volume's root file system.
    pub system_seal: Option<String>,
    /// What named the directory: a LocalPolicy, `boot/active`, the
    /// `--boot-hash`, or nothing (evidence alone).
    pub named_by: Option<String>,
    /// Every requirement that failed; empty when the directory checks.
    pub problems: Vec<String>,
}

/// What to boot from a disk: everything is optional.
#[derive(Default, Clone, Copy)]
pub struct Request<'a> {
    /// The `boot/<hash>` directory, overriding every other name.
    pub boot_hash: Option<&'a str>,
    /// The model the system must be, such as `Mac14,3`.
    pub model: Option<&'a str>,
    /// NVRAM's boot volume: its partition and volume group.
    pub volume: Option<BootVolume>,
    /// Verifies LocalPolicy signatures; the SEP model's.
    pub authority: Option<&'a dyn PolicyAuthority>,
    /// Kexts to link against the system's boot collection (`aux_kc`).
    pub kexts: &'a [Kext<'a>],
}

/// The collection of `request`'s kexts, linked against `kernel`, and what
/// admits it.
fn auxiliary_collection(kernel: &[u8], request: &Request) -> Result<Option<(Decoded, KextGrant)>> {
    if request.kexts.is_empty() {
        return Ok(None);
    }
    let linked = aux_kc::link(kernel, request.kexts)?;
    let source = linked.source(request.kexts);
    let grant = linked.grant();
    let bytes = linked.collection;
    Ok(Some((Decoded { source, bytes }, grant)))
}

/// The hash type and hash of a root-hash record: an `isys` object's payload,
/// or a `csys` one's (BaseSystem's), which has the same layout.
pub fn root_hash_record(payload: &[u8]) -> Result<(Hash, &[u8])> {
    let word = |at: usize| -> Result<u32> {
        Ok(u32::from_le_bytes(
            payload
                .get(at..at + 4)
                .ok_or(refuse!("truncated root-hash record"))?
                .try_into()
                .unwrap(),
        ))
    };
    let version = word(0)?;
    require!(
        version == ROOT_HASH_RECORD_VERSION,
        "root-hash record version {version} is not {ROOT_HASH_RECORD_VERSION}",
        version,
        ROOT_HASH_RECORD_VERSION
    );
    let hash = Hash::from_type(word(8)?)?;
    let length = word(12)? as usize;
    require!(
        length == hash.size(),
        "root-hash record holds {length} bytes for a {hash} hash",
        length,
        ?hash
    );
    let bytes = payload
        .get(ROOT_HASH_HEADER_BYTES..ROOT_HASH_HEADER_BYTES + length)
        .ok_or(refuse!("truncated root-hash record"))?;
    Ok((hash, bytes))
}

fn evaluate<R: ReadAt>(
    container: &mut Container<R>,
    preboot: &Volume,
    set: &BootSet,
) -> Result<Evidence> {
    let mut evidence = Evidence {
        directory: set.directory(),
        manifest_sha384: None,
        root_hash: None,
        system_seal: None,
        named_by: None,
        problems: Vec::new(),
    };
    let mut manifests: Vec<Vec<u8>> = Vec::new();
    for object in &set.objects {
        if let Some(manifest) = preboot::manifest(container, preboot, object)?
            && !manifests.contains(&manifest)
        {
            manifests.push(manifest);
        }
    }
    let manifest = match &manifests[..] {
        [manifest] => {
            let digest = hex::encode_upper(im4m::sha384(manifest));
            if !digest.eq_ignore_ascii_case(&set.boot_hash) {
                evidence.problems.push(format!(
                    "its name is not the SHA-384 of its objects' manifest ({digest})"
                ));
            }
            evidence.manifest_sha384 = Some(digest);
            Some(im4m::parse(manifest)?)
        }
        [] => {
            evidence
                .problems
                .push("no object carries a manifest".into());
            None
        }
        _ => {
            evidence.problems.push(format!(
                "its objects carry {} different manifests",
                manifests.len()
            ));
            None
        }
    };
    let root_hash = match set.object(ROOT_HASH_TYPE) {
        Ok(Some(object)) => Some(object.clone()),
        Ok(None) => {
            evidence
                .problems
                .push("it has no isys root-hash object".into());
            None
        }
        Err(error) => {
            evidence.problems.push(format!("{error}"));
            None
        }
    };
    let record = match root_hash {
        Some(object) => {
            let file = preboot::read(container, preboot, &object)?;
            let parts = img4::parse(&file).within(|| refuse!("{path}", path = object.path))?;
            let digest = manifest
                .as_ref()
                .and_then(|m| m.object("isys"))
                .and_then(|o| o.properties.iter().find(|p| p.name == "DGST"))
                .map(|p| &p.value);
            match digest {
                Some(im4m::Value::Data(digest))
                    if digest.as_slice() == im4m::sha384(parts.payload).as_slice() => {}
                Some(_) => evidence
                    .problems
                    .push("its manifest's isys digest is not the root-hash object's".into()),
                None if manifest.is_some() => evidence
                    .problems
                    .push("its manifest records no isys digest".into()),
                None => {}
            }
            let im4p = im4p::parse(parts.payload)?;
            require!(
                im4p.compression_algorithm.is_none(),
                "{path}: a compressed root-hash object",
                path = object.path
            );
            let (hash, bytes) =
                root_hash_record(im4p.payload).within(|| refuse!("{path}", path = object.path))?;
            evidence.root_hash = Some(hex::encode(bytes));
            Some((hash, bytes.to_vec()))
        }
        None => None,
    };
    match preboot::system_seal(container, &set.volume_group, evidence.root_hash.as_deref()) {
        Ok(Some(seal)) => {
            evidence.system_seal = Some(hex::encode(&seal.root));
            if let Some((hash, bytes)) = &record
                && (*hash != seal.hash || *bytes != seal.root)
            {
                evidence.problems.push(
                    "its root hash is not the seal of the System volume's root file system".into(),
                );
            }
        }
        Ok(None) => evidence
            .problems
            .push("its System volume's root file system is not sealed".into()),
        Err(error) => evidence.problems.push(format!("{error}")),
    }
    Ok(evidence)
}

/// The directory LLB would boot in `group`, and what named it: the
/// LocalPolicy's `nsih` when that directory exists, else `boot/active`.
fn named<R: ReadAt>(
    container: &mut Container<R>,
    preboot: &Volume,
    group: &str,
    sets: &[BootSet],
    authority: Option<&dyn PolicyAuthority>,
    notes: &mut Vec<String>,
) -> Result<Option<(String, String)>> {
    let exists = |hash: &str| {
        sets.iter()
            .any(|s| s.volume_group == group && s.boot_hash.eq_ignore_ascii_case(hash))
    };
    let files = preboot::local_policies(container, preboot, group)?;
    let mut policies = Vec::new();
    for (path, bytes) in &files {
        match LocalPolicy::parse(path, bytes) {
            Ok(policy) => policies.push(policy),
            Err(error) => notes.push(format!("{path}: {error:#}")),
        }
    }
    match &mut policies[..] {
        [] => {}
        [policy] => {
            let label = match authority {
                Some(authority) => match policy.verify(authority) {
                    Ok(()) => Some("verified"),
                    Err(error) => {
                        notes.push(format!("{error:#}; its nsih is not used"));
                        None
                    }
                },
                None => Some("signature not verified"),
            };
            let group_matches = policy
                .volume_group
                .is_none_or(|vuid| uuid_text(&vuid).eq_ignore_ascii_case(group));
            if !group_matches {
                notes.push(format!("{} names another volume group", policy.path));
            }
            if let (Some(label), Some(nsih), true) = (label, &policy.nsih, group_matches) {
                if exists(nsih) {
                    return Ok(Some((
                        nsih.clone(),
                        format!("LocalPolicy {} ({label})", policy.path),
                    )));
                }
                notes.push(format!("{}: no boot/{nsih} directory", policy.path));
            }
        }
        several => notes.push(format!(
            "{group} has {} LocalPolicies; none names the boot directory",
            several.len()
        )),
    }
    match preboot::active(container, preboot, group)? {
        Some(hash) if exists(&hash) => Ok(Some((hash, format!("{group}/boot/active")))),
        Some(hash) => {
            notes.push(format!("{group}/boot/active names a missing boot/{hash}"));
            Ok(None)
        }
        None => Ok(None),
    }
}

/// The boot directory the evidence shows to be the installed system's,
/// among those `request` names.
pub fn select<R: ReadAt>(
    container: &mut Container<R>,
    request: &Request,
) -> Result<(Volume, BootSet, Evidence)> {
    let (preboot, sets) = preboot::candidates(container)?;
    let listed = || {
        sets.iter()
            .map(BootSet::directory)
            .collect::<Vec<_>>()
            .join(", ")
    };
    let wanted = request.volume.map(|v| uuid_text(&v.volume_group));
    let mut groups: Vec<String> = Vec::new();
    for set in &sets {
        if wanted
            .as_ref()
            .is_none_or(|g| g.eq_ignore_ascii_case(&set.volume_group))
            && !groups.contains(&set.volume_group)
        {
            groups.push(set.volume_group.clone());
        }
    }
    require!(
        !groups.is_empty(),
        "NVRAM boots volume group {unwrap_or_default}, which has no boot directory here; the candidates are {listed}",
        unwrap_or_default = wanted.unwrap_or_default(),
        listed = listed()
    );
    let mut notes = Vec::new();
    let mut chosen: Vec<(BootSet, Option<String>)> = Vec::new();
    for group in &groups {
        let in_group = sets.iter().filter(|s| &s.volume_group == group);
        if let Some(hash) = request.boot_hash {
            chosen.extend(
                in_group
                    .filter(|s| s.boot_hash.eq_ignore_ascii_case(hash))
                    .map(|s| (s.clone(), Some("--boot-hash".into()))),
            );
            continue;
        }
        match named(
            container,
            &preboot,
            group,
            &sets,
            request.authority,
            &mut notes,
        )? {
            Some((hash, by)) => chosen.extend(
                in_group
                    .filter(|s| s.boot_hash.eq_ignore_ascii_case(&hash))
                    .map(|s| (s.clone(), Some(by.clone()))),
            ),
            None => chosen.extend(in_group.map(|s| (s.clone(), None))),
        }
    }
    require!(
        !chosen.is_empty(),
        "no boot directory is named {unwrap_or_default}; the candidates are {listed}",
        unwrap_or_default = request.boot_hash.unwrap_or_default(),
        listed = listed()
    );
    let mut qualified = Vec::new();
    let mut refused = Vec::new();
    for (set, by) in chosen {
        let mut evidence = evaluate(container, &preboot, &set)?;
        evidence.named_by = by;
        if evidence.problems.is_empty() {
            qualified.push((set, evidence));
        } else {
            refused.push(format!(
                "{}: {}",
                evidence.directory,
                evidence.problems.join("; ")
            ));
        }
    }
    refused.extend(notes);
    match qualified.len() {
        1 => {
            let (set, evidence) = qualified.remove(0);
            Ok((preboot, set, evidence))
        }
        0 => Err(refuse!(
            "no boot directory is shown to be the installed system's ({refused})",
            refused = refused.join(" | ")
        )
        .into()),
        count => {
            let directories = qualified
                .iter()
                .map(|(set, _)| set.directory())
                .collect::<Vec<_>>();
            Err(refuse!("{count} boot directories ({directories}) each match their System volume, and neither a \
                 LocalPolicy nor boot/active names one ({refused})", count, directories = directories.join(", "), refused = refused.join(" | ")).into())
        }
    }
}

/// A bundle and where on the container its objects came from.
pub struct Installed {
    pub bundle: Bundle,
    pub preboot: Volume,
    pub set: BootSet,
    pub evidence: Evidence,
}

/// The installed loader's inputs. Its constructor stays in this module
/// where the owned kernel and its exact decode receipt originate.
pub(crate) struct BundleInputs<'a>(Inputs<'a>);

impl<'a> BundleInputs<'a> {
    pub(crate) fn into_inputs(self) -> Inputs<'a> {
        self.0
    }
}

pub struct Decoded {
    bytes: Vec<u8>,
    source: Source,
}

impl Decoded {
    fn input(&self) -> Input<'_> {
        Input {
            bytes: &self.bytes,
            source: self.source.clone(),
        }
    }
}

struct Disk<'a, R: ReadAt> {
    container: &'a mut Container<R>,
    preboot: Volume,
    set: BootSet,
}

impl<R: ReadAt> Disk<'_, R> {
    fn read(&mut self, object: &preboot::Object) -> Result<Vec<u8>> {
        preboot::read(self.container, &self.preboot, object)
            .within(|| refuse!("{path}", path = object.path))
    }

    /// The single object of `kind`, decoded, and its manifest.
    fn decode(&mut self, kind: &[u8; 4], limit: u64) -> Result<(Decoded, Option<Vec<u8>>)> {
        let object = self.set.require(kind)?.clone();
        let streamed = if object.bytes >= im4p::stream::MIN_FILE_BYTES {
            let mut prefix = [0; im4p::stream::PREFIX_BYTES];
            let mut file =
                preboot::open_regular(self.container, &self.preboot, &object, &mut prefix)
                    .within(|| refuse!("{path}", path = object.path))?;
            im4p::stream::decode(&mut file, &prefix, limit)?
        } else {
            None
        };
        let decoded = if let Some(image) = streamed {
            image
        } else {
            let file = self.read(&object)?;
            let parts = img4::parse(&file).within(|| refuse!("{path}", path = object.path))?;
            let im4p = im4p::parse(parts.payload)?;
            let (bytes, _) = im4p.decode_payload(limit)?;
            im4p::DecodedImage {
                bytes,
                payload_type: im4p.payload_type,
                description: im4p.description,
                container_sha256: sha256(&file),
                decoded_sha256: None,
                manifest: parts.manifest.map(<[u8]>::to_vec),
            }
        };
        let decoded_sha256 = match decoded.decoded_sha256 {
            Some(digest) => digest,
            None => sha256(&decoded.bytes),
        };
        let source = Source {
            component: format!("Preboot {}", String::from_utf8_lossy(kind)),
            member: object.path,
            container_sha256: decoded.container_sha256,
            payload_type: decoded.payload_type,
            description: decoded.description,
            decoded_bytes: decoded.bytes.len() as u64,
            decoded_sha256,
        };
        Ok((
            Decoded {
                bytes: decoded.bytes,
                source,
            },
            decoded.manifest,
        ))
    }

    /// The build the boot set's iBoot names itself with.
    fn iboot_build(&mut self) -> Result<Option<String>> {
        let Some(object) = self.set.object(b"ibot")?.cloned() else {
            return Ok(None);
        };
        let file = self.read(&object)?;
        let parts = img4::parse(&file).within(|| refuse!("{path}", path = object.path))?;
        Ok(Some(crate::im4p::parse(parts.payload)?.description))
    }

    /// An optional object kept as its encoded payload, as the SEP firmware is.
    fn opaque(&mut self, kind: &[u8; 4]) -> Result<Option<Decoded>> {
        let Some(object) = self.set.object(kind)?.cloned() else {
            return Ok(None);
        };
        let file = self.read(&object)?;
        let parts = img4::parse(&file).within(|| refuse!("{path}", path = object.path))?;
        Ok(Some(Decoded {
            source: Source {
                component: format!("Preboot {}", String::from_utf8_lossy(kind)),
                member: object.path.clone(),
                container_sha256: sha256(&file),
                payload_type: "opaque".into(),
                description: String::new(),
                decoded_bytes: parts.payload.len() as u64,
                decoded_sha256: sha256(parts.payload),
            },
            bytes: parts.payload.to_vec(),
        }))
    }

    /// The machine's personalized Cryptex1 manifest.
    fn cryptex1_manifest(&mut self, ecid: u64) -> Result<Option<Decoded>> {
        let ticket =
            preboot::cryptex1_ticket(self.container, &self.preboot, &self.set.volume_group, ecid)
                .within(|| refuse!("the Cryptex1 manifest"))?;
        Ok(ticket.map(|(path, bytes)| Decoded {
            source: Source {
                component: "Cryptex1 manifest".into(),
                member: path,
                container_sha256: sha256(&bytes),
                payload_type: "im4m".into(),
                description: "the personalized Cryptex1 manifest".into(),
                decoded_bytes: bytes.len() as u64,
                decoded_sha256: sha256(&bytes),
            },
            bytes,
        }))
    }

    /// `ProductVersion` and `ProductBuildVersion` of the volume group.
    fn system_version(&mut self) -> Result<(String, String)> {
        let group = &self.set.volume_group;
        let device = &mut self.container.device;
        let mut path = format!("{group}/{CRYPTEX_SYSTEM_VERSION}");
        let mut found = self.preboot.lookup(device, &path, true);
        if matches!(found, Err(apfs::ApfsError::NotFound(_))) {
            path = format!("{group}/{SYSTEM_VERSION}");
            found = self.preboot.lookup(device, &path, true);
        }
        let inode = found.within(|| refuse!("{path}", path))?;
        let bytes = self
            .preboot
            .read(device, &inode)
            .within(|| refuse!("{path}", path))?;
        let text =
            core::str::from_utf8(&bytes).map_err(|_| refuse!("{path} is not UTF-8", path))?;
        let field = |key: &str| -> Result<String> {
            plist_string(text, key)
                .within(|| refuse!("{path}", path))?
                .ok_or_else(|| refuse!("{path} has no {key}", path, key).into())
        };
        Ok((field("ProductVersion")?, field("ProductBuildVersion")?))
    }
}

/// The `<string>` value of `key` in the top-level dictionary of an XML
/// property list; `None` when the key is absent. Binary property lists are
/// refused.
pub(crate) fn plist_string(text: &str, key: &str) -> Result<Option<String>> {
    require!(
        !text.starts_with("bplist"),
        "binary property lists are unsupported"
    );
    require!(
        text.contains("<plist") && text.contains("<dict>"),
        "not an XML property list"
    );
    let marker = format!("<key>{key}</key>");
    let Some(at) = text.find(&marker) else {
        return Ok(None);
    };
    let rest = text[at + marker.len()..].trim_start();
    let value = rest
        .strip_prefix("<string>")
        .and_then(|rest| rest.split_once("</string>"))
        .map(|(value, _)| value)
        .ok_or_else(|| refuse!("{key} is not a string", key))?;
    let mut decoded = String::new();
    let mut rest = value;
    while let Some(at) = rest.find('&') {
        decoded.push_str(&rest[..at]);
        let (entity, after) = rest[at..]
            .split_once(';')
            .ok_or(refuse!("unterminated XML entity"))?;
        decoded.push(match entity {
            "&amp" => '&',
            "&lt" => '<',
            "&gt" => '>',
            "&quot" => '"',
            "&apos" => '\'',
            _ => return Err(refuse!("unsupported XML entity {entity};", entity).into()),
        });
        rest = after;
    }
    decoded.push_str(rest);
    Ok(Some(decoded))
}

fn root_string(tree: &DeviceTree, property: &str) -> Result<String> {
    tree.property("/", property)
        .and_then(|p| p.as_str())
        .map(Into::into)
        .ok_or_else(|| refuse!("the DeviceTree root has no {property}", property).into())
}

fn manifest_id(manifest: Option<&im4m::Manifest>, key: &str) -> Result<Option<u32>> {
    match manifest.and_then(|m| m.property(key)) {
        Some(im4m::Value::Integer(value)) => Ok(Some(u32::try_from(*value).map_err(|_| {
            refuse!(
                "boot manifest {key} {value:#x} exceeds u32",
                key,
                value = *value
            )
        })?)),
        Some(_) => Err(refuse!("boot manifest {key} is not an integer", key).into()),
        None => Ok(None),
    }
}

/// Build the boot bundle of the installed system on `container`, as
/// `request` names it; its `model`, when given, must match the DeviceTree's.
pub fn prepare<R: ReadAt>(
    container: &mut Container<R>,
    request: &Request,
    options: &Options,
    sep_firmware: Option<&Decoded>,
) -> Result<Installed> {
    let model = request.model;
    let (preboot, set, evidence) = select(container, request)?;
    let mut disk = Disk {
        container,
        preboot,
        set,
    };
    let (kernel, boot_manifest) = disk.decode(b"krnl", KERNEL_LIMIT)?;
    let (sptm, _) = disk.decode(b"sptm", COMPONENT_LIMIT)?;
    let (txm, _) = disk.decode(b"trxm", COMPONENT_LIMIT)?;
    let (device_tree, _) = disk.decode(b"dtre", COMPONENT_LIMIT)?;
    let (trust_cache, _) = disk.decode(b"trst", COMPONENT_LIMIT)?;
    // The root pivots through BaseSystem (`Inputs::pivot_root_hash`), whose
    // code, the dyld shared cache among it, this trust cache lists.
    let base_system_trust_cache = disk
        .set
        .object(b"bstc")?
        .is_some()
        .then(|| disk.decode(b"bstc", COMPONENT_LIMIT))
        .transpose()?
        .map(|(decoded, _)| decoded);
    let boot_set_sep = disk.opaque(SEP_FIRMWARE_TYPE)?;
    let sep = boot_set_sep.as_ref().or(sep_firmware);
    // The evidence already required this object and checked its signature.
    let root_hash = disk.opaque(ROOT_HASH_TYPE)?;
    let pivot_root_hash = disk.opaque(BASE_SYSTEM_ROOT_HASH_TYPE)?;
    let firmware_version = disk.iboot_build()?;
    let cryptex1_manifest = options
        .ecid
        .map(|ecid| disk.cryptex1_manifest(ecid))
        .transpose()?
        .flatten();
    let (product_version, product_build) = disk.system_version()?;
    let tree = DeviceTree::parse(&device_tree.bytes)?;
    let product_type = root_string(&tree, "model")?;
    if let Some(wanted) = model {
        require!(
            wanted == product_type,
            "the disk's system is a {product_type}, not a {wanted}",
            product_type,
            wanted
        );
    }
    let target_type = root_string(&tree, "target-type")?;
    let manifest = boot_manifest.as_deref().map(im4m::parse).transpose()?;
    let boot_manifest = boot_manifest.map(|bytes| Decoded {
        source: Source {
            component: "Boot manifest".into(),
            member: kernel.source.member.clone(),
            container_sha256: sha256(&bytes),
            payload_type: "im4m".into(),
            description: "the kernel collection's personalized manifest".into(),
            decoded_bytes: bytes.len() as u64,
            decoded_sha256: sha256(&bytes),
        },
        bytes,
    });
    let (auxiliary, kexts) = auxiliary_collection(&kernel.bytes, request)?.unzip();
    let kext_code =
        Linked::trust_cache(request.kexts)?.map(|(source, bytes)| Decoded { source, bytes });
    let inputs = Inputs {
        target: Target {
            product_type,
            // The model the install personalized this system for, which a
            // development restore changes.
            device_class: cryptex1_manifest
                .as_ref()
                .and_then(|ticket| preboot::cryptex1_ticket_model(&ticket.source.member))
                .map_or_else(
                    || format!("{}ap", target_type.to_ascii_lowercase()),
                    String::from,
                ),
            product_version,
            product_build,
            variant: INSTALLED_SYSTEM.into(),
            chip_id: manifest_id(manifest.as_ref(), "CHIP")?,
            board_id: manifest_id(manifest.as_ref(), "BORD")?,
        },
        kernel_collection: KernelInput {
            bytes: kernel.bytes.into(),
            source: kernel.source,
        },
        auxiliary_collection: auxiliary.as_ref().map(Decoded::input),
        sptm: sptm.input(),
        txm: txm.input(),
        device_tree: device_tree.input(),
        static_trust_cache: trust_cache.input(),
        sep_firmware: sep.map(Decoded::input),
        boot_manifest: boot_manifest.as_ref().map(Decoded::input),
        cryptex1_manifest: cryptex1_manifest.as_ref().map(Decoded::input),
        extra_trust_caches: base_system_trust_cache
            .iter()
            .chain(&kext_code)
            .map(Decoded::input)
            .collect(),
        ramdisk: None,
        // The installed volumes hold none: the engine reads the board's flash.
        board_firmware: Vec::new(),
        root_hash: Some(RootHash {
            volume: SealedVolume::System,
            im4p: root_hash.as_ref().map(Decoded::input),
            snapshot: evidence
                .root_hash
                .as_deref()
                .map(preboot::root_snapshot_name),
        }),
        pivot_root_hash: pivot_root_hash.as_ref().map(Decoded::input),
        firmware_version: firmware_version.as_deref(),
    };
    let mut options = options.placed_at(inputs.dram_base()?);
    options.security.kexts = kexts;
    options.boot_command = Some(crate::handoff::installed_command(
        options.boot_command,
        &options.boot_args,
    )?);
    // iBoot appends -x for the safe entry command (0x29a34..0x29abc).
    if options.boot_command == Some(crate::handoff::BootCommand::FsbootSafe)
        && !options
            .boot_args
            .split_ascii_whitespace()
            .any(|arg| arg == "-x")
    {
        options.boot_args.push_str(" -x");
    }
    options.root = Some(Root::System {
        volume_group: disk.set.volume_group.clone(),
        boot_hash: disk.set.boot_hash.clone(),
    });
    let mut bundle = boot_bundle::prepare_moving_kernel(BundleInputs(inputs), &options)?;
    if !evidence
        .named_by
        .as_deref()
        .is_some_and(|by| by.ends_with("(verified)"))
    {
        bundle.manifest.unresolved.push(format!(
            "no SEP-verified LocalPolicy names {}: it was chosen by {} and its evidence (its \
             manifest's SHA-384, its isys digest and its System volume's seal); the \
             LocalPolicy's other hashes are not inputs",
            disk.set.directory(),
            evidence.named_by.as_deref().unwrap_or("evidence alone")
        ));
    }
    Ok(Installed {
        bundle,
        preboot: disk.preboot,
        set: disk.set,
        evidence,
    })
}

/// The installed system on `source`: a disk with a GUID partition table or
/// one APFS container. With a boot volume, only its partition is read and
/// its type must be APFS; otherwise exactly one container must hold a
/// bootable system. Returns the container's byte offset and the bundle.
pub fn prepare_disk<R: ReadAt>(
    source: &mut R,
    request: &Request,
    options: &Options,
) -> Result<(u64, Installed)> {
    let mut partitions = apfs::container_partitions(source)?;
    if let Some(volume) = &request.volume {
        require!(
            volume.kind == VolumeType::Apfs,
            "NVRAM boots a {kind} partition, not an APFS container",
            kind = alloc::format!("{:?}", volume.kind)
        );
        partitions.retain(|(_, guid)| guid.is_none_or(|g| g.0 == volume.partition));
        require!(
            !partitions.is_empty(),
            "NVRAM boots partition {partition}, which this disk does not have",
            partition = uuid_text(&volume.partition)
        );
    }
    let sep_firmware = preboot::sep_firmware(&mut *source)
        .within(|| refuse!("the SEP firmware"))?
        .map(|(path, bytes)| {
            let digest = sha256(&bytes);
            Decoded {
                source: Source {
                    component: "SEP firmware".into(),
                    member: path,
                    container_sha256: digest.clone(),
                    payload_type: "img4".into(),
                    description: "the iBootSystemContainer's SEP firmware".into(),
                    decoded_bytes: bytes.len() as u64,
                    decoded_sha256: digest,
                },
                bytes,
            }
        });
    let mut found = Vec::new();
    let mut failures = Vec::new();
    for (offset, _) in partitions {
        let result = Container::open(&mut *source, offset)
            .map_err(Error::from)
            .and_then(|mut container| {
                prepare(&mut container, request, options, sep_firmware.as_ref())
            });
        match result {
            Ok(installed) => found.push((offset, installed)),
            Err(error) => failures.push(format!("container at {offset:#x}: {error:#}")),
        }
    }
    match found.len() {
        1 => Ok(found.remove(0)),
        0 => Err(refuse!(
            "no installed system to boot: {failures}",
            failures = failures.join("; ")
        )
        .into()),
        count => {
            let offsets = found
                .iter()
                .map(|(offset, _)| format!("{offset:#x}"))
                .collect::<Vec<_>>();
            Err(refuse!("{count} containers hold a bootable system (at {offsets}); NVRAM's boot-volume must name one", count, offsets = offsets.join(", ")).into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_root_hash_records_as_apfs_prepare_cryptex_writes_them() {
        let hash: Vec<u8> = (0..32).collect();
        let mut record = vec![0u8; 208];
        for (at, word) in [(0, 2u32), (8, 1), (12, 32)] {
            record[at..at + 4].copy_from_slice(&word.to_le_bytes());
        }
        record[16..48].copy_from_slice(&hash);
        let (kind, bytes) = root_hash_record(&record).unwrap();
        assert_eq!((kind, bytes), (Hash::Sha256, hash.as_slice()));
        let mut wrong = record.clone();
        wrong[0] = 3;
        assert!(root_hash_record(&wrong).is_err());
        let mut short = record;
        short[12] = 48;
        assert!(root_hash_record(&short).is_err());
    }

    #[test]
    fn reads_strings_from_xml_property_lists() {
        let text = "<?xml version=\"1.0\"?>\n<plist version=\"1.0\">\n<dict>\n\
            \t<key>ProductBuildVersion</key>\n\t<string>26A434</string>\n\
            \t<key>ProductName</key>\n\t<string>mac&amp;OS &lt;x&gt;</string>\n\
            \t<key>Count</key>\n\t<integer>3</integer>\n</dict>\n</plist>\n";
        assert_eq!(
            plist_string(text, "ProductBuildVersion")
                .unwrap()
                .as_deref(),
            Some("26A434")
        );
        assert_eq!(
            plist_string(text, "ProductName").unwrap().as_deref(),
            Some("mac&OS <x>")
        );
        assert_eq!(plist_string(text, "ProductVersion").unwrap(), None);
        assert!(plist_string(text, "Count").is_err());
        assert!(plist_string("bplist00", "ProductVersion").is_err());
    }
}
