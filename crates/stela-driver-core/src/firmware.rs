// SPDX-License-Identifier: GPL-2.0-or-later
//! Exact host-device firmware resources, independent of a filesystem or OS loader.
//!
//! A setup/boot loader supplies owned [`Resource`] buffers and their manifests to
//! [`Catalog::from_handoff`], then keeps the catalog while drivers borrow images.
//! The host-only manifest and reserved-page boot handoff supply it separately
//! from guest Apple boot bundles.
//! No vendor firmware is embedded here. Length and SHA-256 checks establish
//! integrity against the supplied manifest, **not** vendor authentication. A
//! caller that changes both the manifest and the bytes can pass those checks.
//! Device drivers must still validate vendor signatures, headers, firmware APIs,
//! calibration and upload/reset protocols before accepting a device as ready.
//!
//! Consumer requirements were checked against Linux commit
//! `a804c5742bff224d4cb4d25dd383554d598be32b`: `iwl-drv.c` builds a hardware
//! prefix plus API filename (`cfg/ax210.c` selects API 89); `btintel.c` selects
//! separate `.sfi`/`.ddc` names using silicon revision tuples; SOF's `loader.c`
//! and `sof-pci-dev.c` select platform firmware/topology paths. This native
//! facility incorporates no Linux code and never guesses a compatible variant.
//! Reference paths at that pin:
//! - `drivers/net/wireless/intel/iwlwifi/{iwl-drv.c,cfg/ax210.c}`
//! - `drivers/bluetooth/btintel.c`
//! - `sound/soc/sof/{loader.c,sof-pci-dev.c}`
//!
//! An image cannot outlive its provider:
//! ```compile_fail
//! use stela_driver_core::firmware::{Catalog, Provider, Requirement, Target};
//! let image = {
//!     let catalog = Catalog::from_handoff(vec![]).unwrap();
//!     catalog.load(Requirement {
//!         name: "vendor/fw.bin",
//!         target: Target { driver: "wifi", hardware: "1234:5678",
//!                          revision: "a0", board: None },
//!         minimum_bytes: 1,
//!     }).unwrap()
//! };
//! let _ = image.bytes();
//! ```

pub mod smbios;

use alloc::{string::String, vec::Vec};
use core::fmt;
use stela_crypto::{Sha256, digest::Digest};

/// An owned diagnostic name. Copying borrowed bytes uses a fallible reserve;
/// OOM keeps the original byte count and SHA-256 instead of aborting. This
/// fallback is explicitly unavailable diagnostic data, never a resource identity.
/// Resource selection and valid-name acceptance are unchanged.
#[derive(PartialEq, Eq)]
pub enum ErrorName {
    Available(String),
    AllocationFailed {
        original_bytes: usize,
        sha256: [u8; 32],
    },
}

impl ErrorName {
    /// Exact complete bytes when available; OOM never substitutes a prefix.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Available(name) => Some(name),
            Self::AllocationFailed { .. } => None,
        }
    }
    /// Only a complete diagnostic name can match a suffix.
    pub fn ends_with(&self, suffix: &str) -> bool {
        self.as_str().is_some_and(|name| name.ends_with(suffix))
    }
}
impl From<&str> for ErrorName {
    fn from(name: &str) -> Self {
        let mut owned = String::new();
        if owned.try_reserve_exact(name.len()).is_err() {
            return Self::AllocationFailed {
                original_bytes: name.len(),
                sha256: Sha256::digest(name.as_bytes()).into(),
            };
        }
        owned.push_str(name);
        Self::Available(owned)
    }
}
impl From<String> for ErrorName {
    fn from(name: String) -> Self {
        Self::Available(name)
    }
}
impl Clone for ErrorName {
    fn clone(&self) -> Self {
        match self {
            Self::Available(name) => Self::from(name.as_str()),
            Self::AllocationFailed {
                original_bytes,
                sha256,
            } => Self::AllocationFailed {
                original_bytes: *original_bytes,
                sha256: *sha256,
            },
        }
    }
}
impl PartialEq<str> for ErrorName {
    fn eq(&self, name: &str) -> bool {
        self.as_str() == Some(name)
    }
}
impl PartialEq<&str> for ErrorName {
    fn eq(&self, name: &&str) -> bool {
        self == *name
    }
}
impl PartialEq<String> for ErrorName {
    fn eq(&self, name: &String) -> bool {
        self == name.as_str()
    }
}
impl fmt::Display for ErrorName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Available(name) => fmt::Debug::fmt(name, f),
            Self::AllocationFailed {
                original_bytes,
                sha256,
            } => {
                write!(
                    f,
                    "<name unavailable: allocation failed; original bytes={original_bytes}; sha256="
                )?;
                for byte in sha256 {
                    write!(f, "{byte:02x}")?;
                }
                write!(f, ">")
            }
        }
    }
}
impl fmt::Debug for ErrorName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

/// Driver-defined, canonical identity labels, compared byte for byte.
/// `board: None` is an exact board-independent binding, never a wildcard.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Target<'a> {
    pub driver: &'a str,
    pub hardware: &'a str,
    /// Includes every silicon/ABI revision component the driver requires.
    pub revision: &'a str,
    pub board: Option<&'a str>,
}

/// An exact filename and target selected by the device driver.
#[derive(Clone, Copy, Debug)]
pub struct Requirement<'a> {
    pub name: &'a str,
    pub target: Target<'a>,
    /// Nonzero lower bound, before device-specific header parsing.
    pub minimum_bytes: usize,
}

/// Source identity kept with every accepted image; not an authentication proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Provenance<'a> {
    /// Package name or source URL.
    pub source: &'a str,
    /// Immutable source revision or vendor release identifier.
    pub revision: &'a str,
    /// Exact relative member path in the source package.
    pub member: &'a str,
    pub license: &'a str,
}

/// Loader-to-driver handoff. Construction consumes `bytes` without copying them;
/// descriptor strings need only live through [`Catalog::from_handoff`].
/// Expected length and digest must come from the loader's selected manifest.
pub struct Resource<'a, B = Vec<u8>> {
    pub name: &'a str,
    pub target: Target<'a>,
    pub provenance: Provenance<'a>,
    pub expected_length: usize,
    pub expected_sha256: [u8; 32],
    pub bytes: B,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    pub name: ErrorName,
    pub kind: ErrorKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ErrorKind {
    InvalidName,
    InvalidTarget(&'static str),
    InvalidProvenance(&'static str),
    InvalidRequirement,
    Missing,
    WrongTarget,
    Truncated {
        expected: usize,
        actual: usize,
    },
    LengthMismatch {
        expected: usize,
        actual: usize,
    },
    DigestMismatch {
        expected: [u8; 32],
        actual: [u8; 32],
    },
    Duplicate,
    Conflict,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "firmware {}: {}", self.name, self.kind)
    }
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let hex = |f: &mut fmt::Formatter<'_>, digest: &[u8; 32]| {
            digest.iter().try_for_each(|byte| write!(f, "{byte:02x}"))
        };
        match self {
            Self::InvalidName => f.write_str("invalid resource name"),
            Self::InvalidTarget(field) => write!(f, "invalid target {field}"),
            Self::InvalidProvenance(field) => write!(f, "invalid provenance {field}"),
            Self::InvalidRequirement => f.write_str("invalid requirement"),
            Self::Missing => f.write_str("not supplied"),
            Self::WrongTarget => f.write_str("supplied only for another target"),
            Self::Truncated { expected, actual } => {
                write!(f, "{actual} bytes, shorter than {expected}")
            }
            Self::LengthMismatch { expected, actual } => {
                write!(f, "{actual} bytes, longer than {expected}")
            }
            Self::DigestMismatch { expected, actual } => {
                f.write_str("SHA-256 ")?;
                hex(f, actual)?;
                f.write_str(" instead of ")?;
                hex(f, expected)
            }
            Self::Duplicate => f.write_str("supplied twice"),
            Self::Conflict => f.write_str("supplied with conflicting manifests"),
        }
    }
}

impl core::error::Error for Error {}

#[derive(Debug, PartialEq, Eq)]
struct OwnedTarget {
    driver: String,
    hardware: String,
    revision: String,
    board: Option<String>,
}

impl OwnedTarget {
    fn view(&self) -> Target<'_> {
        Target {
            driver: &self.driver,
            hardware: &self.hardware,
            revision: &self.revision,
            board: self.board.as_deref(),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
struct OwnedProvenance {
    source: String,
    revision: String,
    member: String,
    license: String,
}

impl OwnedProvenance {
    fn view(&self) -> Provenance<'_> {
        Provenance {
            source: &self.source,
            revision: &self.revision,
            member: &self.member,
            license: &self.license,
        }
    }
}

#[derive(Debug)]
enum Bytes {
    Owned(Vec<u8>),
    Persistent(&'static [u8]),
}

impl AsRef<[u8]> for Bytes {
    fn as_ref(&self) -> &[u8] {
        match self {
            Self::Owned(bytes) => bytes,
            Self::Persistent(bytes) => bytes,
        }
    }
}

#[derive(Debug)]
struct Entry {
    name: String,
    target: OwnedTarget,
    provenance: OwnedProvenance,
    sha256: [u8; 32],
    bytes: Bytes,
}

/// Immutable verified resource catalog. Failed construction publishes no entries.
/// Loading never performs filesystem IO, allocates image copies or tries fallback
/// resource names. Drivers keep responsibility for choosing compatible APIs.
pub struct Catalog {
    entries: Vec<Entry>,
}

impl fmt::Debug for Catalog {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (images, bytes) = self.inventory();
        f.debug_struct("Catalog")
            .field("images", &images)
            .field("bytes", &bytes)
            .finish()
    }
}

/// Lifetime-checked, immutable image lookup. Implementations must preserve exact
/// name/target matching and integrity checks before publishing an [`Image`].
/// Other providers can select/delegate to validated catalogs; images cannot be
/// constructed from unchecked bytes through this interface.
pub trait Provider {
    fn load(&self, requirement: Requirement<'_>) -> Result<Image<'_>, Error>;
}

/// Borrow the boot handoff's optional catalog through the same exact provider
/// contract. Absence behaves like an empty catalog, including validation of
/// malformed requirements before reporting a missing resource.
pub struct HandoffProvider<'a> {
    catalog: Option<&'a Catalog>,
}

impl<'a> HandoffProvider<'a> {
    pub const fn new(catalog: Option<&'a Catalog>) -> Self {
        Self { catalog }
    }
}

impl Provider for HandoffProvider<'_> {
    fn load(&self, requirement: Requirement<'_>) -> Result<Image<'_>, Error> {
        static EMPTY: Catalog = Catalog {
            entries: Vec::new(),
        };
        self.catalog.unwrap_or(&EMPTY).load(requirement)
    }
}

/// An immutable image borrowed from a catalog; no device upload has occurred.
#[derive(Clone, Copy)]
pub struct Image<'a> {
    entry: &'a Entry,
}

impl<'a> Image<'a> {
    pub fn name(self) -> &'a str {
        &self.entry.name
    }
    pub fn target(self) -> Target<'a> {
        self.entry.target.view()
    }
    pub fn provenance(self) -> Provenance<'a> {
        self.entry.provenance.view()
    }
    pub fn sha256(self) -> [u8; 32] {
        self.entry.sha256
    }
    pub fn bytes(self) -> &'a [u8] {
        self.entry.bytes.as_ref()
    }
}

impl fmt::Debug for Image<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Image")
            .field("name", &self.name())
            .field("target", &self.target())
            .field("provenance", &self.provenance())
            .field("sha256", &self.sha256())
            .field("length", &self.bytes().len())
            .finish()
    }
}

impl Catalog {
    pub fn from_handoff(resources: Vec<Resource<'_>>) -> Result<Self, Error> {
        Self::from_resources(resources, Bytes::Owned)
    }

    /// Permanent immutable loader pages or static data, without an image copy.
    /// The caller must obtain static references; constructing references
    /// from physical addresses remains the platform's unsafe mapping boundary.
    pub fn from_persistent_handoff(
        resources: Vec<Resource<'_, &'static [u8]>>,
    ) -> Result<Self, Error> {
        Self::from_resources(resources, Bytes::Persistent)
    }

    /// A catalog holding `image`'s verified bytes under the exact `name` and
    /// `target` a driver asked for: the image's bytes and digest were checked
    /// when its own catalog was built, so neither is copied nor hashed again.
    /// The provenance stays the image's.
    pub fn rebind(image: Image<'static>, name: &str, target: Target<'_>) -> Result<Self, Error> {
        Self::rebind_all([(image, name, target)])
    }

    /// [`Self::rebind`] for several images, each under its own name and
    /// target, which must differ.
    pub fn rebind_all<'a>(
        bindings: impl IntoIterator<Item = (Image<'static>, &'a str, Target<'a>)>,
    ) -> Result<Self, Error> {
        let mut entries: Vec<Entry> = Vec::new();
        for (image, name, target) in bindings {
            validate(name, target)?;
            if entries
                .iter()
                .any(|entry| entry.name == name && entry.target.view() == target)
            {
                return Err(error(name, ErrorKind::Duplicate));
            }
            let provenance = image.provenance();
            entries.push(Entry {
                name: name.into(),
                target: OwnedTarget {
                    driver: target.driver.into(),
                    hardware: target.hardware.into(),
                    revision: target.revision.into(),
                    board: target.board.map(String::from),
                },
                provenance: OwnedProvenance {
                    source: provenance.source.into(),
                    revision: provenance.revision.into(),
                    member: provenance.member.into(),
                    license: provenance.license.into(),
                },
                sha256: image.sha256(),
                bytes: Bytes::Persistent(image.bytes()),
            });
        }
        Ok(Self { entries })
    }

    /// Number and total bytes of immutable resources, for boot diagnostics.
    pub fn inventory(&self) -> (usize, usize) {
        (
            self.entries.len(),
            self.entries
                .iter()
                .map(|entry| entry.bytes.as_ref().len())
                .sum(),
        )
    }

    fn from_resources<B>(
        resources: Vec<Resource<'_, B>>,
        wrap: impl Fn(B) -> Bytes,
    ) -> Result<Self, Error> {
        let mut entries: Vec<Entry> = Vec::with_capacity(resources.len());
        for resource in resources {
            let fail = |kind| error(resource.name, kind);
            validate(resource.name, resource.target)?;
            let provenance = resource.provenance;
            for (field, value) in [
                ("source", provenance.source),
                ("revision", provenance.revision),
                ("license", provenance.license),
            ] {
                if !label(value) {
                    return Err(fail(ErrorKind::InvalidProvenance(field)));
                }
            }
            if !path(provenance.member) {
                return Err(fail(ErrorKind::InvalidProvenance("member")));
            }
            let bytes = wrap(resource.bytes);
            let actual = bytes.as_ref().len();
            if actual < resource.expected_length {
                return Err(fail(ErrorKind::Truncated {
                    expected: resource.expected_length,
                    actual,
                }));
            }
            if actual != resource.expected_length {
                return Err(fail(ErrorKind::LengthMismatch {
                    expected: resource.expected_length,
                    actual,
                }));
            }
            let sha256: [u8; 32] = Sha256::digest(bytes.as_ref()).into();
            if sha256 != resource.expected_sha256 {
                return Err(fail(ErrorKind::DigestMismatch {
                    expected: resource.expected_sha256,
                    actual: sha256,
                }));
            }
            if let Some(previous) = entries
                .iter()
                .find(|entry| entry.name == resource.name && entry.target.view() == resource.target)
            {
                return Err(fail(
                    if previous.sha256 == sha256
                        && previous.bytes.as_ref() == bytes.as_ref()
                        && previous.provenance.view() == provenance
                    {
                        ErrorKind::Duplicate
                    } else {
                        ErrorKind::Conflict
                    },
                ));
            }
            entries.push(Entry {
                name: resource.name.into(),
                target: OwnedTarget {
                    driver: resource.target.driver.into(),
                    hardware: resource.target.hardware.into(),
                    revision: resource.target.revision.into(),
                    board: resource.target.board.map(String::from),
                },
                provenance: OwnedProvenance {
                    source: provenance.source.into(),
                    revision: provenance.revision.into(),
                    member: provenance.member.into(),
                    license: provenance.license.into(),
                },
                sha256,
                bytes,
            });
        }
        Ok(Self { entries })
    }
}

impl Provider for Catalog {
    fn load(&self, requirement: Requirement<'_>) -> Result<Image<'_>, Error> {
        validate(requirement.name, requirement.target)?;
        let fail = |kind| error(requirement.name, kind);
        if requirement.minimum_bytes == 0 {
            return Err(fail(ErrorKind::InvalidRequirement));
        }
        let mut named = false;
        for entry in &self.entries {
            if entry.name != requirement.name {
                continue;
            }
            named = true;
            if entry.target.view() != requirement.target {
                continue;
            }
            if entry.bytes.as_ref().len() < requirement.minimum_bytes {
                return Err(fail(ErrorKind::Truncated {
                    expected: requirement.minimum_bytes,
                    actual: entry.bytes.as_ref().len(),
                }));
            }
            return Ok(Image { entry });
        }
        Err(fail(if named {
            ErrorKind::WrongTarget
        } else {
            ErrorKind::Missing
        }))
    }
}

fn error(name: &str, kind: ErrorKind) -> Error {
    Error {
        name: name.into(),
        kind,
    }
}

/// Validate the exact relative resource name before consulting a provider.
pub fn validate_name(name: &str) -> Result<(), Error> {
    if !path(name) {
        return Err(error(name, ErrorKind::InvalidName));
    }
    Ok(())
}

fn validate(name: &str, target: Target<'_>) -> Result<(), Error> {
    validate_name(name)?;
    for (field, value) in [
        ("driver", target.driver),
        ("hardware", target.hardware),
        ("revision", target.revision),
    ] {
        if !label(value) {
            return Err(error(name, ErrorKind::InvalidTarget(field)));
        }
    }
    if target.board.is_some_and(|board| !label(board)) {
        return Err(error(name, ErrorKind::InvalidTarget("board")));
    }
    Ok(())
}

fn label(value: &str) -> bool {
    !value.is_empty() && !value.chars().any(char::is_control)
}

fn path(value: &str) -> bool {
    label(value)
        && !value.contains('\\')
        && value
            .split('/')
            .all(|part| !matches!(part, "" | "." | ".."))
}

#[cfg(test)]
mod tests;
