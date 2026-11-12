//! The complete host firmware bundle: every pinned recipe's file, whatever
//! computer made the USB. Files are stored once by content (`<digest>.bin`)
//! beside their license notices, under [`DIRECTORY`] on the USB's data
//! volume, and [`MANIFEST`] lists them. A boot reads only the files the PCI
//! functions it observes need ([`select`]), and [`BundleProvider`] gives a
//! driver's exact request the file its recipe pins.
use super::{
    Error, MAX_IMAGE_BYTES, MAX_MANIFEST_BYTES, fail, logical_path,
    recipes::{self, Artifact, Plan, Recipe},
    requirements::{Inventory, Request},
    storage_path, valid_label,
};
use alloc::{boxed::Box, format, string::String, vec::Vec};
use core::cell::RefCell;
use serde::{Deserialize, Serialize};
use stela_driver_core::firmware::{
    self as driver, Catalog, ErrorKind, Image, Provider, Requirement,
};

/// The bundle's directory on its volume.
pub const DIRECTORY: &str = "stela/firmware";
/// The bundle's manifest on its volume.
pub const MANIFEST: &str = "stela/firmware/bundle.json";
/// At most this many files and as many notices.
pub const MAX_OBJECTS: usize = 256;
/// The target a bundle file has in the boot's catalog, under the name
/// [`name`] gives its digest. Only [`BundleProvider`] asks for it.
pub const TARGET: driver::Target<'static> = driver::Target {
    driver: "stela-bundle",
    hardware: "content",
    revision: "sha256",
    board: None,
};

/// Every recipe family, in the order a volume short of room keeps them: the
/// small network and Bluetooth files first, the GPU's large images last.
pub const FAMILIES: [(&str, &[Recipe]); 8] = [
    ("Wi-Fi regulatory database", recipes::REGULATORY),
    ("Realtek Ethernet", recipes::REALTEK),
    ("Realtek Wi-Fi", recipes::REALTEK_WIFI),
    ("MediaTek Wi-Fi", recipes::MEDIATEK_WIFI),
    ("Intel Wi-Fi", recipes::INTEL_WIFI),
    ("MediaTek Bluetooth", recipes::MEDIATEK_BLUETOOTH),
    ("Intel Bluetooth", recipes::INTEL_RSA),
    ("NVIDIA GSP-RM", recipes::GSP),
];

/// Every recipe of the catalog.
pub fn recipes() -> impl Iterator<Item = &'static Recipe> {
    FAMILIES.iter().flat_map(|(_, recipes)| recipes.iter())
}

/// Each distinct file of the catalog once, by digest, in [`FAMILIES`]
/// order with each family's files smallest first.
pub fn objects() -> Vec<&'static Artifact> {
    let mut objects: Vec<&'static Artifact> = Vec::new();
    for (_, family) in FAMILIES {
        let start = objects.len();
        for recipe in family {
            if !objects
                .iter()
                .any(|seen| seen.sha256 == recipe.image.sha256)
            {
                objects.push(&recipe.image);
            }
        }
        objects[start..].sort_by_key(|artifact| (artifact.length, artifact.sha256));
    }
    objects
}

/// Each distinct license notice of the catalog once, by digest.
pub fn notices() -> Vec<&'static Artifact> {
    let mut notices: Vec<&'static Artifact> = Vec::new();
    for recipe in recipes() {
        if !notices
            .iter()
            .any(|seen| seen.sha256 == recipe.license.sha256)
        {
            notices.push(recipe.license);
        }
    }
    notices
}

/// The complete bundle's file count and bytes, notices aside.
pub fn size() -> (usize, usize) {
    let objects = objects();
    (
        objects.len(),
        objects.iter().map(|artifact| artifact.length).sum(),
    )
}

/// The catalog name of the file with `sha256`.
pub fn name(sha256: &str) -> String {
    format!("sha256/{sha256}")
}

/// A file's path in [`DIRECTORY`]: its digest's first half.
pub fn object_path(sha256: &str) -> String {
    format!("{}.bin", &sha256[..32.min(sha256.len())])
}

/// A notice's path in [`DIRECTORY`].
pub fn notice_path(sha256: &str) -> String {
    format!("licenses/{}.txt", &sha256[..32.min(sha256.len())])
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Bundle<'a> {
    pub schema: u32,
    #[serde(borrow)]
    pub objects: Vec<Object<'a>>,
    #[serde(borrow)]
    pub notices: Vec<Notice<'a>>,
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Object<'a> {
    pub path: &'a str,
    pub sha256: &'a str,
    pub length: usize,
    pub source: &'a str,
    pub revision: &'a str,
    pub member: &'a str,
    /// The path of its notice.
    pub license: &'a str,
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Notice<'a> {
    pub path: &'a str,
    pub sha256: &'a str,
    pub length: usize,
    pub source: &'a str,
    pub revision: &'a str,
    pub member: &'a str,
}

fn digest(sha256: &str) -> bool {
    sha256.len() == 64
        && sha256
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// The manifest of a bundle of `files`, each with its license notice.
pub fn manifest(files: &[(&Artifact, &Artifact)]) -> Result<Vec<u8>, Error> {
    let mut notices: Vec<&Artifact> = Vec::new();
    for (_, notice) in files {
        if !notices.iter().any(|seen| seen.sha256 == notice.sha256) {
            notices.push(notice);
        }
    }
    let objects: Vec<_> = files
        .iter()
        .map(|(object, notice)| {
            serde_json::json!({"path":object_path(object.sha256),"sha256":object.sha256,
                "length":object.length,"source":object.source,"revision":object.revision,
                "member":object.member,"license":notice_path(notice.sha256)})
        })
        .collect();
    let notices: Vec<_> = notices
        .iter()
        .map(|notice| {
            serde_json::json!({"path":notice_path(notice.sha256),"sha256":notice.sha256,
                "length":notice.length,"source":notice.source,"revision":notice.revision,
                "member":notice.member})
        })
        .collect();
    let bytes = serde_json::to_vec_pretty(
        &serde_json::json!({"schema":1,"objects":objects,"notices":notices}),
    )
    .map_err(|e| Error(format!("host firmware bundle manifest: {e}")))?;
    Bundle::parse(&bytes)?;
    Ok(bytes)
}

/// The complete bundle's files with their notices, in [`objects`] order.
pub fn complete() -> Vec<(&'static Artifact, &'static Artifact)> {
    objects()
        .into_iter()
        .filter_map(|object| {
            let recipe = recipes().find(|recipe| recipe.image.sha256 == object.sha256)?;
            Some((object, recipe.license))
        })
        .collect()
}

impl<'a> Bundle<'a> {
    pub fn parse(bytes: &'a [u8]) -> Result<Self, Error> {
        if bytes.len() > MAX_MANIFEST_BYTES {
            return Err(fail("host firmware bundle manifest exceeds 128 KiB"));
        }
        let bundle: Self = serde_json::from_slice(bytes)
            .map_err(|e| Error(format!("host firmware bundle manifest: {e}")))?;
        if bundle.schema != 1 {
            return Err(fail("unsupported host firmware bundle schema"));
        }
        if bundle.objects.len() > MAX_OBJECTS || bundle.notices.len() > MAX_OBJECTS {
            return Err(fail("host firmware bundle lists too many files"));
        }
        for (index, notice) in bundle.notices.iter().enumerate() {
            if !digest(notice.sha256)
                || notice.path != notice_path(notice.sha256)
                || notice.length == 0
                || notice.length > MAX_IMAGE_BYTES
                || !valid_label(notice.source)
                || !valid_label(notice.revision)
                || !logical_path(notice.member)
                || bundle.notices[..index]
                    .iter()
                    .any(|other| other.sha256 == notice.sha256)
            {
                return Err(fail("invalid host firmware bundle notice"));
            }
        }
        for (index, object) in bundle.objects.iter().enumerate() {
            if !digest(object.sha256)
                || object.path != object_path(object.sha256)
                || !storage_path(object.path)
                || object.length == 0
                || object.length > MAX_IMAGE_BYTES
                || !valid_label(object.source)
                || !valid_label(object.revision)
                || !logical_path(object.member)
                || !bundle
                    .notices
                    .iter()
                    .any(|notice| notice.path == object.license)
                || bundle.objects[..index]
                    .iter()
                    .any(|other| other.sha256 == object.sha256 || other.path == object.path)
            {
                return Err(fail("invalid host firmware bundle file"));
            }
        }
        Ok(bundle)
    }

    /// The file with `sha256`, when the bundle holds it.
    pub fn object(&self, sha256: &str) -> Option<&Object<'a>> {
        self.objects.iter().find(|object| object.sha256 == sha256)
    }

    /// The catalog's files the bundle lacks.
    pub fn missing(&self) -> Vec<&'static Artifact> {
        objects()
            .into_iter()
            .filter(|artifact| self.object(artifact.sha256).is_none())
            .collect()
    }
}

/// The schema 1 manifest a boot hands the engine: `exact`'s images, then
/// each of `selected` under its digest's [`name`] and [`TARGET`], in that
/// order, which the image regions follow.
pub fn handoff_manifest(
    exact: Option<&super::Manifest<'_>>,
    selected: &[&Object<'_>],
) -> Result<Vec<u8>, Error> {
    let mut images = Vec::new();
    for image in exact.iter().flat_map(|manifest| &manifest.images) {
        images.push(serde_json::json!({"name":image.name,"path":image.path,
            "target":{"driver":image.target.driver,"hardware":image.target.hardware,
                "revision":image.target.revision,"board":image.target.board},
            "provenance":{"source":image.provenance.source,"revision":image.provenance.revision,
                "member":image.provenance.member,"license":image.provenance.license},
            "length":image.length,"sha256":image.sha256}));
    }
    for object in selected {
        images.push(serde_json::json!({"name":name(object.sha256),
            "path":format!("bundle/{}", object.path),
            "target":{"driver":TARGET.driver,"hardware":TARGET.hardware,
                "revision":TARGET.revision,"board":null},
            "provenance":{"source":object.source,"revision":object.revision,
                "member":object.member,"license":object.license},
            "length":object.length,"sha256":object.sha256}));
    }
    let bytes = serde_json::to_vec(&serde_json::json!({"schema":1,"images":images}))
        .map_err(|e| Error(format!("host firmware handoff manifest: {e}")))?;
    super::Manifest::parse(&bytes)?;
    Ok(bytes)
}

/// One PCI function a boot observed: IDs and base class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pci {
    pub vendor: u16,
    pub device: u16,
    pub class: u8,
}

/// The files a PC needs and why, and its devices of a supported family
/// whose files no recipe pins.
#[derive(Debug, Default)]
pub struct Selection {
    /// Each file once, with the device that asked first.
    pub wanted: Vec<(&'static Artifact, String)>,
    /// Devices of a supported family that no recipe covers, each once.
    pub unknown: Vec<String>,
}

impl Selection {
    fn add(&mut self, artifact: &'static Artifact, why: &str) {
        if !self
            .wanted
            .iter()
            .any(|(seen, _)| seen.sha256 == artifact.sha256)
        {
            self.wanted.push((artifact, why.into()));
        }
    }

    fn plan(&mut self, inventory: &Inventory, why: &str) {
        if let Ok(plan) = Plan::select(inventory) {
            for selection in plan.selections() {
                self.add(&selection.recipe.image, why);
            }
        }
    }

    fn members(&mut self, recipes: &'static [Recipe], members: &[&str], why: &str) {
        for recipe in recipes {
            if members.is_empty() || members.contains(&recipe.image.member) {
                self.add(&recipe.image, why);
            }
        }
    }

    /// The bytes of the wanted files.
    pub fn bytes(&self) -> usize {
        self.wanted
            .iter()
            .map(|(artifact, _)| artifact.length)
            .sum()
    }
}

/// The files a PC with these PCI functions needs. USB Bluetooth controllers
/// are not on PCI before the boot's own USB stack runs, so their files,
/// about 2 MB in all, are always selected.
pub fn select(functions: &[Pci]) -> Selection {
    let mut selection = Selection::default();
    selection.members(recipes::MEDIATEK_BLUETOOTH, &[], "usb bluetooth");
    selection.members(recipes::INTEL_RSA, &[], "usb bluetooth");
    for function in functions {
        let why = format!("pci {:04x}:{:04x}", function.vendor, function.device);
        let mut inventory = Inventory::default();
        if function.vendor == 0x10de && function.class == 0x03 {
            match recipes::nvidia_chip(function.device) {
                Some(chip) if recipes::require_gsp(&mut inventory, chip).is_ok() => {
                    selection.plan(&inventory, &why);
                }
                _ => {
                    if !selection.unknown.contains(&why) {
                        selection.unknown.push(why.clone());
                    }
                }
            }
            continue;
        }
        let wifi = match (function.vendor, function.device) {
            (0x10ec, 0x8168 | 0x8161) => {
                selection.members(recipes::REALTEK, &[], &why);
                false
            }
            (0x10ec, 0xc822 | 0xc82f) => {
                selection.members(
                    recipes::REALTEK_WIFI,
                    &["rtw88/rtw8822c_fw.bin", "rtw88/rtw8822c_wow_fw.bin"],
                    &why,
                );
                true
            }
            (0x10ec, 0xb852 | 0xb85b) => {
                selection.members(recipes::REALTEK_WIFI, &["rtw89/rtw8852b_fw-2.bin"], &why);
                true
            }
            (vendor, device) => {
                recipes::require_wifi(&mut inventory, vendor, device).unwrap_or(false)
            }
        };
        if wifi {
            selection.plan(&inventory, &why);
            selection.members(recipes::REGULATORY, &[], &why);
        }
    }
    selection
}

/// Gives a driver's exact request the bundle file its recipe pins, when the
/// catalog has no exact binding for it: the file, found by digest, is bound
/// once to the request's own name and target, so the driver sees the image
/// it asked for. Call only under the host drivers' lock, like
/// [`super::requirements::RecordingProvider`].
pub struct BundleProvider {
    inner: &'static dyn Provider,
    pin: fn(&Request) -> Option<&'static Artifact>,
    bound: RefCell<Vec<&'static Catalog>>,
}

fn pinned(request: &Request) -> Option<&'static Artifact> {
    recipes::recipe(request).map(|recipe| &recipe.image)
}

impl BundleProvider {
    pub fn new(inner: &'static dyn Provider) -> Self {
        Self::with_pins(inner, pinned)
    }

    /// With `pin` in place of the recipes: the file a request binds to.
    fn with_pins(
        inner: &'static dyn Provider,
        pin: fn(&Request) -> Option<&'static Artifact>,
    ) -> Self {
        Self {
            inner,
            pin,
            bound: RefCell::new(Vec::new()),
        }
    }
}

impl Provider for BundleProvider {
    fn load(&self, requirement: Requirement<'_>) -> Result<Image<'_>, driver::Error> {
        let exact = self.inner.load(requirement);
        match &exact {
            Err(error) if matches!(error.kind, ErrorKind::Missing | ErrorKind::WrongTarget) => {}
            _ => return exact,
        }
        for catalog in self.bound.borrow().iter() {
            if let Ok(image) = catalog.load(requirement) {
                return Ok(image);
            }
        }
        let Some(pin) = (self.pin)(&Request::from(requirement)) else {
            return exact;
        };
        let name = name(pin.sha256);
        let stored = match self.inner.load(Requirement {
            name: &name,
            target: TARGET,
            minimum_bytes: requirement.minimum_bytes,
        }) {
            Ok(stored) => stored,
            Err(error) if error.kind == ErrorKind::Missing => return exact,
            Err(mut error) => {
                error.name = requirement.name.into();
                return Err(error);
            }
        };
        if pin
            .verify_digest(stored.bytes().len(), stored.sha256())
            .is_err()
        {
            return exact;
        }
        let catalog: &'static Catalog = Box::leak(Box::new(Catalog::rebind(
            stored,
            requirement.name,
            requirement.target,
        )?));
        self.bound.borrow_mut().push(catalog);
        catalog.load(requirement)
    }
}

#[cfg(test)]
mod tests;
