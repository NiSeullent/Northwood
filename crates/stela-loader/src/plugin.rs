//! Links the Metal plugin: a fat `MH_BUNDLE` of its arm64 and arm64e objects,
//! with chained fixups and an ad-hoc signature.
//!
//! No released `ld64.lld` links arm64e's authenticated pointers, and a
//! plugin needs little of a general linker: one image with no exports, whose
//! imports dyld finds wherever the frameworks the process loaded define them
//! (`BIND_SPECIAL_DYLIB_FLAT_LOOKUP`), so nothing here reads a framework.
//! Layouts follow `mach-o/loader.h`, `mach-o/fixup-chains.h` and
//! `cs_blobs.h` at `xnu-12377.121.6`.
use crate::aux_kc::{
    fit,
    relocatable::{self, Definition, Kind, Object, Target},
    unit::{branch, page, page_offset, uuid},
};
use crate::trust_cache::CodeHash;
use crate::{Result, Within};
use object::{LittleEndian as LE, macho};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};
use std::iter;
use stela_contracts::{refuse, require};

// Segments align to the macOS arm64 page.
use stela_contracts::bytes::GUEST_PAGE_BYTES as PAGE;
/// A signature hashes the file in units of this.
const HASH_PAGE: u64 = 0x1000;
const HASH_PAGE_SHIFT: u8 = 12;
const HASH_BYTES: usize = 32;

/// `dyld_chained_import.lib_ordinal` of `BIND_SPECIAL_DYLIB_FLAT_LOOKUP`.
const FLAT_LOOKUP: u32 = 0xfe;
const DYLD_CHAINED_IMPORT: u32 = 1;
const DYLD_CHAINED_PTR_64_OFFSET: u16 = 6;
const DYLD_CHAINED_PTR_ARM64E_USERLAND: u16 = 9;
const DYLD_CHAINED_PTR_START_NONE: u16 = 0xffff;
/// macOS 27.0, as `LC_BUILD_VERSION` encodes it.
const MACOS: u32 = 27 << 16;

const CSMAGIC_EMBEDDED_SIGNATURE: u32 = 0xfade_0cc0;
const CSMAGIC_CODEDIRECTORY: u32 = 0xfade_0c02;
const CS_SUPPORTSEXECSEG: u32 = 0x20400;
const CS_ADHOC: u32 = 2;
const CS_LINKER_SIGNED: u32 = 0x2_0000;
const CS_HASHTYPE_SHA256: u8 = 2;
/// `CS_CodeDirectory` through `execSegFlags`.
const CODE_DIRECTORY_BYTES: usize = 88;

pub const CODE_RESOURCES_PATH: &str = "Contents/_CodeSignature/CodeResources";

/// The external files sealed into each slice's code directory.
#[derive(Clone, Copy)]
pub struct BundleSeal {
    info: [u8; HASH_BYTES],
    resources: [u8; HASH_BYTES],
}

impl BundleSeal {
    pub fn new(info: &[u8], resources: &[u8]) -> Self {
        Self {
            info: Sha256::digest(info).into(),
            resources: Sha256::digest(resources).into(),
        }
    }

    /// Special slots -3 (resources), -2 (no requirements) and -1 (Info.plist).
    fn slots(self) -> [[u8; HASH_BYTES]; 3] {
        [self.resources, [0; HASH_BYTES], self.info]
    }
}

/// What every process that loads the plugin has loaded already, which the
/// image names so its imports need nothing else.
const LIBRARIES: [&str; 5] = [
    "/usr/lib/libSystem.B.dylib",
    "/usr/lib/libobjc.A.dylib",
    "/System/Library/Frameworks/Foundation.framework/Versions/C/Foundation",
    "/System/Library/Frameworks/IOKit.framework/Versions/A/IOKit",
    "/System/Library/Frameworks/Metal.framework/Versions/A/Metal",
];

/// `adrp x16, 0`.
const ADRP_X16: u32 = 0x9000_0010;
/// `ldr x16, [x16, #0]`.
const LDR_X16_X16: u32 = 0xf940_0210;
/// `br x16`.
const BR_X16: u32 = 0xd61f_0200;
/// `adrp x17, 0`.
const ADRP_X17: u32 = 0x9000_0011;
/// `add x17, x17, #0`.
const ADD_X17_X17: u32 = 0x9100_0231;
/// `ldr x16, [x17]`.
const LDR_X16_X17: u32 = 0xf940_0230;
/// `braa x16, x17`.
const BRAA_X16_X17: u32 = 0xd71f_0a11;
const STUB_BYTES: u64 = 12;
const AUTH_STUB_BYTES: u64 = 16;
/// Instruction key A: what `braa` authenticates with.
const KEY_IA: u8 = 0;

const FIXUPS_COMMAND: usize = 16;
const SYMTAB_COMMAND: usize = 24;
const DYSYMTAB_COMMAND: usize = 80;
const UUID_COMMAND: usize = 24;
const BUILD_COMMAND: usize = 24;
const SIGNATURE_COMMAND: usize = 16;

/// The plugin's two segments that hold sections.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Region {
    Text,
    Data,
}

/// Where an object's section goes, or `None` for unwind and debug metadata:
/// the plugin never unwinds.
fn region_of(section: &relocatable::Section<'_>) -> Result<Option<Region>> {
    Ok(Some(
        match (section.segment.as_str(), section.name.as_str()) {
            ("__LD", "__compact_unwind") | ("__TEXT", "__eh_frame") | ("__DWARF", _) => {
                return Ok(None);
            }
            ("__TEXT", _) => Region::Text,
            ("__DATA" | "__DATA_CONST", _) => Region::Data,
            (segment, name) => {
                return Err(
                    refuse!("{segment},{name} has no place in a plugin", segment, name).into(),
                );
            }
        },
    ))
}

#[derive(Debug)]
struct Section {
    region: Region,
    name: String,
    flags: u32,
    align: u32,
    bytes: Vec<u8>,
    zerofill: bool,
    address: u64,
}

/// What a reference resolves to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Value {
    Local { section: usize, offset: u64 },
    Import(u32),
    Absolute(u64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Signature {
    key: u8,
    address_diversity: bool,
    diversity: u16,
}

/// What a data pointer becomes at load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Fixup {
    Rebase {
        target: u64,
        signature: Option<Signature>,
    },
    Bind {
        import: u32,
        addend: i64,
        signature: Option<Signature>,
    },
}

/// One slice being linked.
struct Slice<'a> {
    arm64e: bool,
    subtype: u32,
    objects: Vec<Object<'a>>,
    /// Each object section's output section and offset, `None` if dropped.
    placements: Vec<Vec<Option<(usize, u64)>>>,
    sections: Vec<Section>,
    globals: HashMap<String, (usize, usize)>,
    /// Imported names, in table order, and whether each may stay unresolved.
    imports: BTreeMap<String, bool>,
    import_indices: HashMap<String, u32>,
    /// The index of each branched-to import's stub and pointer slot.
    stubs: HashMap<u32, u64>,
    /// The index of each value's slot in `__got`.
    got: HashMap<Value, u64>,
    /// The stubs' section and their slots' section.
    stub_sections: Option<(usize, usize)>,
    got_section: Option<usize>,
    fixups: BTreeMap<u64, Fixup>,
}

impl<'a> Slice<'a> {
    fn new(objects: &'a [&'a [u8]]) -> Result<Self> {
        let objects = objects
            .iter()
            .enumerate()
            .map(|(index, object)| {
                relocatable::parse(object).within(|| refuse!("reading object {index}", index))
            })
            .collect::<Result<Vec<_>>>()?;
        let first = objects.first().ok_or(refuse!("a slice with no objects"))?;
        let (arm64e, subtype) = (first.is_arm64e(), first.subtype);
        require!(
            objects.iter().all(|object| object.subtype == subtype),
            "a slice's objects differ in architecture"
        );
        let mut slice = Self {
            arm64e,
            subtype,
            objects,
            placements: Vec::new(),
            sections: Vec::new(),
            globals: HashMap::new(),
            imports: BTreeMap::new(),
            import_indices: HashMap::new(),
            stubs: HashMap::new(),
            got: HashMap::new(),
            stub_sections: None,
            got_section: None,
            fixups: BTreeMap::new(),
        };
        slice.place()?;
        slice.bind()?;
        slice.synthesize()?;
        Ok(slice)
    }

    /// Merges the objects' sections, by region and name. Every object says
    /// the same `__objc_imageinfo`, and one image has only one.
    fn place(&mut self) -> Result<()> {
        for object in &self.objects {
            let mut placements = Vec::with_capacity(object.sections.len());
            for input in &object.sections {
                let Some(region) = region_of(input)? else {
                    placements.push(None);
                    continue;
                };
                let found = self
                    .sections
                    .iter()
                    .position(|section| section.region == region && section.name == input.name);
                let index = match found {
                    Some(index) => {
                        if input.name == "__objc_imageinfo" {
                            require!(
                                self.sections[index].bytes == input.data,
                                "objects disagree on the Objective-C image info"
                            );
                            placements.push(Some((index, 0)));
                            continue;
                        }
                        index
                    }
                    None => {
                        self.sections.push(Section {
                            region,
                            name: input.name.clone(),
                            flags: input.flags,
                            align: 0,
                            bytes: Vec::new(),
                            zerofill: input.zerofill,
                            address: 0,
                        });
                        self.sections.len() - 1
                    }
                };
                let output = &mut self.sections[index];
                require!(
                    input.align <= 14,
                    "{section} aligns past a page",
                    section = input.name
                );
                output.align = output.align.max(input.align);
                let offset = output.bytes.len().next_multiple_of(1 << input.align);
                output.bytes.resize(offset, 0);
                if input.zerofill {
                    let size: usize = fit(input.size, "a section's size")?;
                    output.bytes.resize(offset + size, 0);
                } else {
                    output.bytes.extend_from_slice(input.data);
                }
                placements.push(Some((index, offset as u64)));
            }
            self.placements.push(placements);
        }
        Ok(())
    }

    /// Binds each external name to its definition, a strong one over a weak
    /// one, and names what no object defines as an import.
    fn bind(&mut self) -> Result<()> {
        for (object_index, object) in self.objects.iter().enumerate() {
            for (symbol_index, symbol) in object.symbols.iter().enumerate() {
                let defined = matches!(
                    symbol.definition,
                    Definition::Section { .. } | Definition::Absolute(_)
                );
                if !symbol.external || !defined {
                    continue;
                }
                if let Some(&(other, index)) = self.globals.get(&symbol.name) {
                    let existing = &self.objects[other].symbols[index];
                    require!(
                        existing.weak || symbol.weak,
                        "{symbol} is defined twice",
                        symbol = symbol.name
                    );
                    if symbol.weak {
                        continue;
                    }
                }
                self.globals
                    .insert(symbol.name.clone(), (object_index, symbol_index));
            }
        }
        for object in &self.objects {
            for symbol in &object.symbols {
                let Definition::Undefined { weak } = symbol.definition else {
                    continue;
                };
                if !self.globals.contains_key(&symbol.name) {
                    let entry = self.imports.entry(symbol.name.clone()).or_insert(weak);
                    *entry &= weak;
                }
            }
        }
        self.import_indices = self
            .imports
            .keys()
            .enumerate()
            .map(|(index, name)| Ok((name.clone(), fit(index, "an import's index")?)))
            .collect::<Result<_>>()?;
        Ok(())
    }

    /// Adds the stubs and slots the relocations need.
    fn synthesize(&mut self) -> Result<()> {
        let (mut stubs, mut got) = (HashMap::new(), HashMap::new());
        for object in 0..self.objects.len() {
            for section in 0..self.objects[object].sections.len() {
                if self.placements[object][section].is_none() {
                    continue;
                }
                for relocation in &self.objects[object].sections[section].relocations {
                    match relocation.kind {
                        Kind::Branch26 => {
                            if let Value::Import(import) =
                                self.resolve(object, relocation.target)?
                            {
                                let next = stubs.len() as u64;
                                stubs.entry(import).or_insert(next);
                            }
                        }
                        Kind::GotPage21 | Kind::GotPageOffset12 => {
                            let value = self.resolve(object, relocation.target)?;
                            let next = got.len() as u64;
                            got.entry(value).or_insert(next);
                        }
                        _ => {}
                    }
                }
            }
        }
        let code = macho::S_ATTR_PURE_INSTRUCTIONS | macho::S_ATTR_SOME_INSTRUCTIONS;
        if !stubs.is_empty() {
            let (name, slots, bytes) = if self.arm64e {
                ("__auth_stubs", "__auth_got", AUTH_STUB_BYTES)
            } else {
                ("__stubs", "__stub_got", STUB_BYTES)
            };
            let stub_section =
                self.synthetic(Region::Text, name, code, 2, stubs.len() as u64 * bytes);
            let slot_section = self.synthetic(Region::Data, slots, 0, 3, stubs.len() as u64 * 8);
            self.stub_sections = Some((stub_section, slot_section));
        }
        if !got.is_empty() {
            self.got_section =
                Some(self.synthetic(Region::Data, "__got", 0, 3, got.len() as u64 * 8));
        }
        self.stubs = stubs;
        self.got = got;
        Ok(())
    }

    fn synthetic(
        &mut self,
        region: Region,
        name: &str,
        flags: u32,
        align: u32,
        bytes: u64,
    ) -> usize {
        self.sections.push(Section {
            region,
            name: name.to_owned(),
            flags,
            align,
            bytes: vec![0; bytes as usize],
            zerofill: false,
            address: 0,
        });
        self.sections.len() - 1
    }

    /// The section order of the file: zerofill last in its segment, where the
    /// file holds none of it.
    fn order(&self) -> Vec<usize> {
        let mut order: Vec<usize> = (0..self.sections.len()).collect();
        order.sort_by_key(|&index| (self.sections[index].region, self.sections[index].zerofill));
        order
    }

    fn resolve(&self, object: usize, target: Target) -> Result<Value> {
        let Target::Symbol(index) = target else {
            return Err(
                refuse!("a section-relative reference where only pointers have them").into(),
            );
        };
        let symbol = &self.objects[object].symbols[index];
        if !symbol.external {
            return self.define(object, index);
        }
        if let Some(&(object, index)) = self.globals.get(&symbol.name) {
            return self.define(object, index);
        }
        let import = self.import_indices.get(&symbol.name).ok_or_else(|| {
            refuse!(
                "{symbol} is neither defined nor imported",
                symbol = symbol.name
            )
        })?;
        Ok(Value::Import(*import))
    }

    fn define(&self, object: usize, index: usize) -> Result<Value> {
        let symbol = &self.objects[object].symbols[index];
        match symbol.definition {
            Definition::Section { section, address } => self
                .locate(object, section, address)
                .within(|| refuse!("defining {symbol}", symbol = symbol.name)),
            Definition::Absolute(value) => Ok(Value::Absolute(value)),
            Definition::Undefined { .. } | Definition::Debug => {
                Err(refuse!("{symbol} is not a definition", symbol = symbol.name).into())
            }
        }
    }

    /// The output location of `address` in an object's section.
    fn locate(&self, object: usize, section: usize, address: u64) -> Result<Value> {
        let input = &self.objects[object].sections[section];
        let (output, offset) = self.placements[object][section].ok_or_else(|| {
            refuse!(
                "a reference into {segment},{section}, which plugins drop",
                segment = input.segment,
                section = input.name
            )
        })?;
        let within = address
            .checked_sub(input.address)
            .filter(|&within| within <= input.size)
            .ok_or_else(|| {
                refuse!(
                    "{address:#x} lies outside {section}",
                    address,
                    section = input.name
                )
            })?;
        Ok(Value::Local {
            section: output,
            offset: offset + within,
        })
    }

    fn address(&self, value: Value) -> Result<u64> {
        match value {
            Value::Local { section, offset } => Ok(self.sections[section].address + offset),
            Value::Import(_) | Value::Absolute(_) => {
                Err(refuse!("{value} is not in the plugin", ?value).into())
            }
        }
    }

    fn stub(&self, import: u32) -> Result<u64> {
        let (stubs, _) = self.stub_sections.ok_or(refuse!("no stubs"))?;
        let index = self.stubs.get(&import).ok_or(refuse!("no stub"))?;
        let bytes = if self.arm64e {
            AUTH_STUB_BYTES
        } else {
            STUB_BYTES
        };
        Ok(self.sections[stubs].address + bytes * index)
    }

    /// Records what `value + addend` needs at `offset` in `section`, or stores
    /// it if it is absolute.
    fn pointer(
        &mut self,
        section: usize,
        offset: u64,
        value: Value,
        addend: i64,
        signature: Option<Signature>,
    ) -> Result<()> {
        require!(
            self.sections[section].region == Region::Data,
            "a pointer in {section}, which the image maps read-only",
            section = self.sections[section].name
        );
        let address = self.sections[section].address + offset;
        require!(address.is_multiple_of(8), "a misaligned pointer");
        let fixup = match value {
            Value::Local { .. } => Fixup::Rebase {
                target: self
                    .address(value)?
                    .checked_add_signed(addend)
                    .ok_or(refuse!("a pointer before the image"))?,
                signature,
            },
            Value::Import(import) => Fixup::Bind {
                import,
                addend,
                signature,
            },
            Value::Absolute(raw) => {
                require!(signature.is_none(), "a signed absolute pointer");
                let at: usize = fit(offset, "a relocation offset")?;
                let bytes = raw.wrapping_add_signed(addend).to_le_bytes();
                self.sections[section].bytes[at..at + 8].copy_from_slice(&bytes);
                return Ok(());
            }
        };
        require!(
            self.fixups.insert(address, fixup).is_none(),
            "two pointers at {address:#x}",
            address
        );
        Ok(())
    }

    /// Fills the stubs and their slots.
    fn fill_stubs(&mut self) -> Result<()> {
        if let Some((stubs, slots)) = self.stub_sections {
            let (stub_base, slot_base) =
                (self.sections[stubs].address, self.sections[slots].address);
            let signature = self.arm64e.then_some(Signature {
                key: KEY_IA,
                address_diversity: true,
                diversity: 0,
            });
            for (import, index) in self.stubs.clone() {
                let slot = slot_base + 8 * index;
                let (bytes, words): (u64, Vec<u32>) = if self.arm64e {
                    let place = stub_base + AUTH_STUB_BYTES * index;
                    let words = vec![
                        page(ADRP_X17, place, slot)?,
                        page_offset(ADD_X17_X17, slot)?,
                        LDR_X16_X17,
                        BRAA_X16_X17,
                    ];
                    (AUTH_STUB_BYTES, words)
                } else {
                    let place = stub_base + STUB_BYTES * index;
                    let words = vec![
                        page(ADRP_X16, place, slot)?,
                        page_offset(LDR_X16_X16, slot)?,
                        BR_X16,
                    ];
                    (STUB_BYTES, words)
                };
                let at: usize = fit(bytes * index, "a stub's offset")?;
                let code: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
                self.sections[stubs].bytes[at..at + code.len()].copy_from_slice(&code);
                self.pointer(slots, 8 * index, Value::Import(import), 0, signature)?;
            }
        }
        if let Some(got) = self.got_section {
            for (value, index) in self.got.clone() {
                self.pointer(got, 8 * index, value, 0, None)?;
            }
        }
        Ok(())
    }

    fn apply(&mut self) -> Result<()> {
        for object in 0..self.objects.len() {
            for input in 0..self.objects[object].sections.len() {
                let Some((section, start)) = self.placements[object][input] else {
                    continue;
                };
                for index in 0..self.objects[object].sections[input].relocations.len() {
                    let relocation = self.objects[object].sections[input].relocations[index];
                    self.relocate(object, section, start, relocation)?;
                }
            }
        }
        Ok(())
    }

    fn relocate(
        &mut self,
        object: usize,
        section: usize,
        start: u64,
        relocation: relocatable::Relocation,
    ) -> Result<()> {
        let offset = start + relocation.offset;
        let at: usize = fit(offset, "a relocation offset")?;
        let place = self.sections[section].address + offset;
        let word = |slice: &Self| -> Result<u32> {
            let bytes = slice.sections[section]
                .bytes
                .get(at..at + 4)
                .and_then(|bytes| <[u8; 4]>::try_from(bytes).ok())
                .ok_or(refuse!("an instruction past its section"))?;
            Ok(u32::from_le_bytes(bytes))
        };
        let patched = match relocation.kind {
            Kind::Pointer | Kind::Authenticated { .. } => {
                let signature = match relocation.kind {
                    Kind::Authenticated {
                        key,
                        address_diversity,
                        diversity,
                    } => Some(Signature {
                        key,
                        address_diversity,
                        diversity,
                    }),
                    _ => None,
                };
                let (value, addend) = match relocation.target {
                    // The bytes in place hold the target's object address.
                    Target::Section(target) => {
                        (self.locate(object, target, relocation.addend as u64)?, 0)
                    }
                    Target::Symbol(_) => {
                        (self.resolve(object, relocation.target)?, relocation.addend)
                    }
                };
                return self.pointer(section, offset, value, addend, signature);
            }
            Kind::Difference { minus, bytes } => {
                let plus = self.address(self.resolve(object, relocation.target)?)?;
                let minus = self.address(self.resolve(object, minus)?)?;
                let difference = (plus as i64)
                    .wrapping_sub(minus as i64)
                    .wrapping_add(relocation.addend);
                let field = &mut self.sections[section].bytes[at..at + usize::from(bytes)];
                if bytes == 4 {
                    let narrow: i32 = fit(difference, "a 32-bit difference")?;
                    field.copy_from_slice(&narrow.to_le_bytes());
                } else {
                    field.copy_from_slice(&difference.to_le_bytes());
                }
                return Ok(());
            }
            Kind::Branch26 => {
                let target = match self.resolve(object, relocation.target)? {
                    Value::Import(import) => self.stub(import)?,
                    value => self.address(value)?,
                };
                branch(
                    word(self)?,
                    place,
                    target.wrapping_add_signed(relocation.addend),
                )?
            }
            Kind::Page21 | Kind::PageOffset12 => {
                let target = self
                    .address(self.resolve(object, relocation.target)?)
                    .within(|| refuse!("a page reference out of the plugin"))?
                    .wrapping_add_signed(relocation.addend);
                if relocation.kind == Kind::Page21 {
                    page(word(self)?, place, target)?
                } else {
                    page_offset(word(self)?, target)?
                }
            }
            Kind::GotPage21 | Kind::GotPageOffset12 => {
                require!(relocation.addend == 0, "a GOT reference with an addend");
                let value = self.resolve(object, relocation.target)?;
                let got = self.got_section.ok_or(refuse!("no GOT"))?;
                let index = self.got.get(&value).ok_or(refuse!("no GOT slot"))?;
                let target = self.sections[got].address + 8 * index;
                if relocation.kind == Kind::GotPage21 {
                    page(word(self)?, place, target)?
                } else {
                    page_offset(word(self)?, target)?
                }
            }
        };
        self.sections[section].bytes[at..at + 4].copy_from_slice(&patched.to_le_bytes());
        Ok(())
    }

    fn format(&self) -> u16 {
        if self.arm64e {
            DYLD_CHAINED_PTR_ARM64E_USERLAND
        } else {
            DYLD_CHAINED_PTR_64_OFFSET
        }
    }

    /// The pointer `fixup` stores, `next` strides before the next in its page.
    fn encode(&self, fixup: Fixup, next: u64) -> Result<u64> {
        if self.arm64e {
            require!(next < 1 << 11, "a chain link past its field");
            return match fixup {
                Fixup::Rebase {
                    target,
                    signature: None,
                } => {
                    require!(target < 1 << 43, "a rebase past its field");
                    Ok(target | next << 51)
                }
                Fixup::Rebase {
                    target,
                    signature: Some(signature),
                } => {
                    require!(target < 1 << 32, "an authenticated rebase past its field");
                    Ok(target | signed(signature) | next << 51 | 1 << 63)
                }
                Fixup::Bind {
                    import,
                    addend,
                    signature: None,
                } => {
                    require!(import < 1 << 16, "an import past its field");
                    require!(
                        (-(1 << 18)..1 << 18).contains(&addend),
                        "an addend past its field"
                    );
                    let addend = (addend as u64) & 0x7_ffff;
                    Ok(u64::from(import) | addend << 32 | next << 51 | 1 << 62)
                }
                Fixup::Bind {
                    import,
                    addend,
                    signature: Some(signature),
                } => {
                    require!(import < 1 << 16, "an import past its field");
                    require!(addend == 0, "an authenticated bind with an addend");
                    Ok(u64::from(import) | signed(signature) | next << 51 | 3 << 62)
                }
            };
        }
        require!(next < 1 << 12, "a chain link past its field");
        match fixup {
            Fixup::Rebase {
                signature: Some(_), ..
            }
            | Fixup::Bind {
                signature: Some(_), ..
            } => Err(refuse!("a signed pointer in an arm64 slice").into()),
            Fixup::Rebase { target, .. } => {
                require!(target < 1 << 36, "a rebase past its field");
                Ok(target | next << 51)
            }
            Fixup::Bind { import, addend, .. } => {
                require!(import < 1 << 24, "an import past its field");
                let addend: u8 = fit(addend, "a bind's addend")?;
                Ok(u64::from(import) | u64::from(addend) << 24 | next << 51 | 1 << 63)
            }
        }
    }

    /// Gives `region`'s sections their addresses from `start`, and answers
    /// where they end and where their bytes in the file do.
    fn lay_out(&mut self, order: &[usize], region: Region, start: u64) -> (u64, u64) {
        let (mut end, mut file_end) = (start, start);
        for &index in order {
            let section = &mut self.sections[index];
            if section.region != region {
                continue;
            }
            section.address = end.next_multiple_of(1 << section.align);
            end = section.address + section.bytes.len() as u64;
            if !section.zerofill {
                file_end = end;
            }
        }
        (end, file_end)
    }

    /// Lays the sections out, applies every reference and writes the Mach-O.
    /// Addresses in the file's two section segments equal file offsets.
    fn link(
        mut self,
        identifier: &str,
        uuid: [u8; 16],
        seal: Option<BundleSeal>,
    ) -> Result<Vec<u8>> {
        let order = self.order();
        let commands = commands_bytes(order.len());
        let header_bytes = (size_of::<macho::MachHeader64<LE>>() + commands) as u64;
        let (text_end, _) = self.lay_out(&order, Region::Text, header_bytes);
        let text_bytes = text_end.next_multiple_of(PAGE);
        let (data_end, data_file_end) = self.lay_out(&order, Region::Data, text_bytes);
        let data_bytes = (data_end - text_bytes).next_multiple_of(PAGE).max(PAGE);
        let data_file_bytes = (data_file_end - text_bytes)
            .next_multiple_of(PAGE)
            .max(PAGE);
        let linkedit = text_bytes + data_file_bytes;

        self.fill_stubs()?;
        self.apply()?;
        let starts = self.chain(text_bytes, data_bytes)?;
        let fixups = self.fixups_blob(&starts, text_bytes)?;
        // dyld validates pointer alignment even for the empty symbol/string tables.
        let strings = (linkedit + fixups.len() as u64).next_multiple_of(8);
        let signature_offset = (strings + 1).next_multiple_of(16);
        let signature_bytes = signature_size(identifier, signature_offset, seal);
        let end = signature_offset + signature_bytes;

        let sections = |region: Region| -> Vec<&Section> {
            order
                .iter()
                .map(|&index| &self.sections[index])
                .filter(|section| section.region == region)
                .collect()
        };
        let mut file = vec![0u8; fit(signature_offset, "the plugin's size")?];
        let mut output = Header {
            file: &mut file,
            at: 0,
        };
        output.mach_header(self.subtype, commands);
        output.segment(
            Segment {
                name: "__TEXT",
                address: 0,
                size: text_bytes,
                offset: 0,
                file_size: text_bytes,
                protection: macho::VM_PROT_READ | macho::VM_PROT_EXECUTE,
            },
            &sections(Region::Text),
        );
        output.segment(
            Segment {
                name: "__DATA",
                address: text_bytes,
                size: data_bytes,
                offset: text_bytes,
                file_size: data_file_bytes,
                protection: macho::VM_PROT_READ | macho::VM_PROT_WRITE,
            },
            &sections(Region::Data),
        );
        output.segment(
            Segment {
                name: "__LINKEDIT",
                address: text_bytes + data_bytes,
                size: (end - linkedit).next_multiple_of(PAGE),
                offset: linkedit,
                file_size: end - linkedit,
                protection: macho::VM_PROT_READ,
            },
            &[],
        );
        output.fixups(
            fit(linkedit, "the fixups' offset")?,
            fit(fixups.len(), "the fixups' size")?,
        );
        output.symbol_tables(fit(strings, "the string table's offset")?);
        output.uuid(uuid);
        output.build_version();
        output.libraries();
        output.signature(
            fit(signature_offset, "the signature's offset")?,
            fit(signature_bytes, "the signature's size")?,
        );
        require!(output.at as u64 == header_bytes, "the load commands' size");

        for section in self.sections.iter().filter(|section| !section.zerofill) {
            let at: usize = fit(section.address, "a section's address")?;
            file[at..at + section.bytes.len()].copy_from_slice(&section.bytes);
        }
        let start: usize = fit(linkedit, "the link edit's offset")?;
        file[start..start + fixups.len()].copy_from_slice(&fixups);
        let signature = sign(&file, identifier, text_bytes, seal);
        require!(
            signature.len() as u64 == signature_bytes,
            "the signature's size"
        );
        file.extend_from_slice(&signature);
        Ok(file)
    }

    /// Links each page's pointers into a chain and answers each page's first
    /// in the `bytes` of the data segment at `data`.
    fn chain(&mut self, data: u64, bytes: u64) -> Result<Vec<u16>> {
        let stride = if self.arm64e { 8 } else { 4 };
        let pages = (bytes / PAGE) as usize;
        let mut starts = vec![DYLD_CHAINED_PTR_START_NONE; pages];
        let fixups: Vec<(u64, Fixup)> = self
            .fixups
            .iter()
            .map(|(&at, &fixup)| (at, fixup))
            .collect();
        for (index, &(address, fixup)) in fixups.iter().enumerate() {
            let offset = address - data;
            let page_index = (offset / PAGE) as usize;
            if index == 0 || (fixups[index - 1].0 - data) / PAGE != offset / PAGE {
                starts[page_index] = (offset % PAGE) as u16;
            }
            let next = fixups
                .get(index + 1)
                .filter(|(following, _)| (following - data) / PAGE == offset / PAGE)
                .map_or(0, |(following, _)| (following - address) / stride);
            let raw = self.encode(fixup, next)?;
            let section = self
                .sections
                .iter_mut()
                .find(|section| {
                    (section.address..section.address + section.bytes.len() as u64)
                        .contains(&address)
                })
                .ok_or(refuse!("a pointer outside every section"))?;
            require!(!section.zerofill, "a pointer in zero-filled memory");
            let at = (address - section.address) as usize;
            section.bytes[at..at + 8].copy_from_slice(&raw.to_le_bytes());
        }
        Ok(starts)
    }

    /// `LC_DYLD_CHAINED_FIXUPS`' payload.
    fn fixups_blob(&self, starts: &[u16], data: u64) -> Result<Vec<u8>> {
        const HEADER: u32 = 28;
        // `dyld_chained_starts_in_image`: three segments, the second with pointers.
        const IMAGE: u32 = 4 + 3 * 4;
        const SEGMENT: u32 = 22;
        let segment_bytes = SEGMENT + 2 * starts.len() as u32;
        let imports_offset = (HEADER + IMAGE + segment_bytes).next_multiple_of(4);
        let symbols_offset = imports_offset + 4 * self.imports.len() as u32;
        let mut blob = Vec::new();
        let word = |blob: &mut Vec<u8>, value: u32| blob.extend_from_slice(&value.to_le_bytes());
        for value in [
            0,
            HEADER,
            imports_offset,
            symbols_offset,
            fit(self.imports.len(), "the import count")?,
            DYLD_CHAINED_IMPORT,
            0,
        ] {
            word(&mut blob, value);
        }
        for value in [3, 0, IMAGE, 0] {
            word(&mut blob, value);
        }
        word(&mut blob, segment_bytes);
        blob.extend_from_slice(&(PAGE as u16).to_le_bytes());
        blob.extend_from_slice(&self.format().to_le_bytes());
        blob.extend_from_slice(&data.to_le_bytes());
        word(&mut blob, 0);
        blob.extend_from_slice(&(starts.len() as u16).to_le_bytes());
        for start in starts {
            blob.extend_from_slice(&start.to_le_bytes());
        }
        blob.resize(imports_offset as usize, 0);
        let mut names = Vec::new();
        for (name, &weak) in &self.imports {
            let offset: u32 = fit(names.len(), "an import name's offset")?;
            require!(offset < 1 << 23, "the import names past their field");
            word(&mut blob, FLAT_LOOKUP | u32::from(weak) << 8 | offset << 9);
            names.extend_from_slice(name.as_bytes());
            names.push(0);
        }
        blob.extend_from_slice(&names);
        Ok(blob)
    }
}

fn signed(signature: Signature) -> u64 {
    u64::from(signature.diversity) << 32
        | u64::from(signature.address_diversity) << 48
        | u64::from(signature.key) << 49
}

fn dylib_command_bytes(name: &str) -> usize {
    (24 + name.len() + 1).next_multiple_of(8)
}

/// The size of the load commands of a slice of `sections` sections.
fn commands_bytes(sections: usize) -> usize {
    let segments =
        3 * size_of::<macho::SegmentCommand64<LE>>() + sections * size_of::<macho::Section64<LE>>();
    let libraries: usize = LIBRARIES.iter().map(|name| dylib_command_bytes(name)).sum();
    segments
        + libraries
        + FIXUPS_COMMAND
        + SYMTAB_COMMAND
        + DYSYMTAB_COMMAND
        + UUID_COMMAND
        + BUILD_COMMAND
        + SIGNATURE_COMMAND
}

/// A segment's load command.
struct Segment {
    name: &'static str,
    address: u64,
    size: u64,
    offset: u64,
    file_size: u64,
    protection: u32,
}

/// Writes the header and load commands.
struct Header<'a> {
    file: &'a mut [u8],
    at: usize,
}

impl Header<'_> {
    fn words(&mut self, values: &[u32]) {
        for value in values {
            self.file[self.at..self.at + 4].copy_from_slice(&value.to_le_bytes());
            self.at += 4;
        }
    }

    fn quads(&mut self, values: &[u64]) {
        for value in values {
            self.file[self.at..self.at + 8].copy_from_slice(&value.to_le_bytes());
            self.at += 8;
        }
    }

    fn name(&mut self, name: &str) {
        self.file[self.at..self.at + name.len()].copy_from_slice(name.as_bytes());
        self.at += 16;
    }

    fn mach_header(&mut self, subtype: u32, commands: usize) {
        self.words(&[
            macho::MH_MAGIC_64,
            macho::CPU_TYPE_ARM64,
            subtype,
            macho::MH_BUNDLE,
            (3 + 6 + LIBRARIES.len()) as u32,
            commands as u32,
            macho::MH_DYLDLINK | macho::MH_TWOLEVEL,
            0,
        ]);
    }

    fn segment(&mut self, segment: Segment, sections: &[&Section]) {
        let bytes = size_of::<macho::SegmentCommand64<LE>>()
            + sections.len() * size_of::<macho::Section64<LE>>();
        self.words(&[macho::LC_SEGMENT_64, bytes as u32]);
        self.name(segment.name);
        self.quads(&[
            segment.address,
            segment.size,
            segment.offset,
            segment.file_size,
        ]);
        self.words(&[
            segment.protection,
            segment.protection,
            sections.len() as u32,
            0,
        ]);
        for section in sections {
            self.name(&section.name);
            self.name(segment.name);
            self.quads(&[section.address, section.bytes.len() as u64]);
            let offset = if section.zerofill {
                0
            } else {
                section.address as u32
            };
            self.words(&[offset, section.align, 0, 0, section.flags, 0, 0, 0]);
        }
    }

    fn fixups(&mut self, offset: u32, bytes: u32) {
        self.words(&[
            macho::LC_DYLD_CHAINED_FIXUPS,
            FIXUPS_COMMAND as u32,
            offset,
            bytes,
        ]);
    }

    /// No symbols: nothing resolves against the plugin by name.
    fn symbol_tables(&mut self, strings: u32) {
        self.words(&[
            macho::LC_SYMTAB,
            SYMTAB_COMMAND as u32,
            strings,
            0,
            strings,
            1,
        ]);
        self.words(&[macho::LC_DYSYMTAB, DYSYMTAB_COMMAND as u32]);
        self.words(&[0; 18]);
    }

    fn uuid(&mut self, uuid: [u8; 16]) {
        self.words(&[macho::LC_UUID, UUID_COMMAND as u32]);
        self.file[self.at..self.at + 16].copy_from_slice(&uuid);
        self.at += 16;
    }

    fn build_version(&mut self) {
        self.words(&[
            macho::LC_BUILD_VERSION,
            BUILD_COMMAND as u32,
            macho::PLATFORM_MACOS,
            MACOS,
            MACOS,
            0,
        ]);
    }

    fn libraries(&mut self) {
        for name in LIBRARIES {
            let bytes = dylib_command_bytes(name);
            self.words(&[macho::LC_LOAD_DYLIB, bytes as u32, 24, 2, 1 << 16, 1 << 16]);
            self.file[self.at..self.at + name.len()].copy_from_slice(name.as_bytes());
            self.at += bytes - 24;
        }
    }

    fn signature(&mut self, offset: u32, bytes: u32) {
        self.words(&[
            macho::LC_CODE_SIGNATURE,
            SIGNATURE_COMMAND as u32,
            offset,
            bytes,
        ]);
    }
}

/// The signature's size for a file signed to `code_limit`.
fn signature_size(identifier: &str, code_limit: u64, seal: Option<BundleSeal>) -> u64 {
    let slots = code_limit.div_ceil(HASH_PAGE);
    let special = seal.map_or(0, |seal| size_of_val(&seal.slots()));
    20 + (CODE_DIRECTORY_BYTES + identifier.len() + 1 + special) as u64 + slots * HASH_BYTES as u64
}

/// The ad-hoc signature of `file`, whose first `text_bytes` are executable.
fn sign(file: &[u8], identifier: &str, text_bytes: u64, seal: Option<BundleSeal>) -> Vec<u8> {
    let slots = file.len().div_ceil(HASH_PAGE as usize);
    let special = seal.map(BundleSeal::slots);
    let special = special.as_slice().as_flattened();
    let hash_offset = CODE_DIRECTORY_BYTES + identifier.len() + 1 + size_of_val(special);
    let directory_bytes = hash_offset + slots * HASH_BYTES;
    let mut directory = Vec::with_capacity(directory_bytes);
    let word = |bytes: &mut Vec<u8>, value: u32| bytes.extend_from_slice(&value.to_be_bytes());
    let quad = |bytes: &mut Vec<u8>, value: u64| bytes.extend_from_slice(&value.to_be_bytes());
    word(&mut directory, CSMAGIC_CODEDIRECTORY);
    word(&mut directory, directory_bytes as u32);
    word(&mut directory, CS_SUPPORTSEXECSEG);
    word(
        &mut directory,
        if seal.is_some() {
            CS_ADHOC
        } else {
            CS_ADHOC | CS_LINKER_SIGNED
        },
    );
    word(&mut directory, hash_offset as u32);
    word(&mut directory, CODE_DIRECTORY_BYTES as u32);
    word(&mut directory, special.len() as u32);
    word(&mut directory, slots as u32);
    word(&mut directory, file.len() as u32);
    directory.extend_from_slice(&[HASH_BYTES as u8, CS_HASHTYPE_SHA256, 0, HASH_PAGE_SHIFT]);
    word(&mut directory, 0);
    word(&mut directory, 0);
    word(&mut directory, 0);
    word(&mut directory, 0);
    quad(&mut directory, 0);
    quad(&mut directory, 0);
    quad(&mut directory, text_bytes);
    // `execSegFlags`: a plugin is no main binary.
    quad(&mut directory, 0);
    directory.extend_from_slice(identifier.as_bytes());
    directory.push(0);
    directory.extend_from_slice(special.as_flattened());
    for chunk in file.chunks(HASH_PAGE as usize) {
        directory.extend_from_slice(&Sha256::digest(chunk));
    }
    let mut blob = Vec::with_capacity(20 + directory.len());
    word(&mut blob, CSMAGIC_EMBEDDED_SIGNATURE);
    word(&mut blob, 20 + directory.len() as u32);
    word(&mut blob, 1);
    word(&mut blob, 0);
    word(&mut blob, 20);
    blob.extend_from_slice(&directory);
    blob
}

/// Each slice's CDHash in a fat plugin [`link`] wrote: the truncated SHA-256
/// of its code directory, which a trust cache lists.
pub fn code_hashes(file: &[u8]) -> Result<Vec<CodeHash>> {
    code_hashes_with_seal(file, None)
}

/// The verified CDHashes of a bundle, including its Info.plist and resource seal.
pub fn bundle_code_hashes(file: &[u8], seal: BundleSeal) -> Result<Vec<CodeHash>> {
    code_hashes_with_seal(file, Some(seal))
}

fn code_hashes_with_seal(file: &[u8], seal: Option<BundleSeal>) -> Result<Vec<CodeHash>> {
    use object::read::macho::{FatArch, MachHeader, MachOFatFile32};

    let fat = MachOFatFile32::parse(file).within(|| refuse!("the plugin is not a fat file"))?;
    require!(!fat.arches().is_empty(), "the plugin has no slices");
    fat.arches()
        .iter()
        .map(|arch| {
            let slice = arch.data(file)?;
            let header = macho::MachHeader64::<LE>::parse(slice, 0)?;
            require!(
                header.cputype(LE) == arch.cputype()
                    && header.cpusubtype(LE) == arch.cpusubtype()
                    && header.filetype(LE) == macho::MH_BUNDLE,
                "a plugin slice disagrees with its fat header"
            );
            let mut commands = header.load_commands(LE, slice, 0)?;
            let mut signature = None;
            while let Some(command) = commands.next()? {
                if command.cmd() == macho::LC_CODE_SIGNATURE {
                    require!(signature.is_none(), "a plugin slice has two signatures");
                    let data: &macho::LinkeditDataCommand<LE> = command.data()?;
                    signature = Some((
                        data.dataoff.get(LE) as usize,
                        data.datasize.get(LE) as usize,
                    ));
                }
            }
            let (offset, bytes) = signature.ok_or(refuse!("a plugin slice is unsigned"))?;
            let signed = slice
                .get(..offset)
                .ok_or(refuse!("a plugin signature offset exceeds its slice"))?;
            let blob = slice
                .get(offset..)
                .filter(|blob| blob.len() == bytes)
                .ok_or(refuse!("a plugin signature exceeds its slice"))?;
            code_hash(signed, blob, seal)
        })
        .collect()
}

/// Validate the SHA-256 signature layout this linker writes before trusting
/// its directory. Every range stays inside its containing slice or blob.
fn code_hash(signed: &[u8], blob: &[u8], seal: Option<BundleSeal>) -> Result<CodeHash> {
    fn word(bytes: &[u8], at: usize) -> Result<usize> {
        let value = bytes
            .get(at..)
            .and_then(|tail| tail.get(..4))
            .ok_or(refuse!("a plugin signature is truncated"))?;
        Ok(u32::from_be_bytes(value.try_into().unwrap()) as usize)
    }
    require!(
        word(blob, 0)? == CSMAGIC_EMBEDDED_SIGNATURE as usize
            && word(blob, 4)? == blob.len()
            && word(blob, 8)? == 1
            && word(blob, 12)? == 0
            && word(blob, 16)? == 20,
        "a plugin slice's signature is malformed"
    );
    let directory = blob
        .get(20..)
        .ok_or(refuse!("a plugin signature has no directory"))?;
    let special = seal.map(BundleSeal::slots);
    let special = special.as_slice().as_flattened();
    require!(
        directory.len() >= CODE_DIRECTORY_BYTES
            && word(directory, 0)? == CSMAGIC_CODEDIRECTORY as usize
            && word(directory, 4)? == directory.len()
            && word(directory, 8)? == CS_SUPPORTSEXECSEG as usize
            && word(directory, 24)? == special.len()
            && word(directory, 32)? == signed.len()
            && directory[36..40] == [HASH_BYTES as u8, CS_HASHTYPE_SHA256, 0, HASH_PAGE_SHIFT],
        "a plugin slice's code directory is malformed or unsupported"
    );
    let start = word(directory, 16)?;
    let identifier = word(directory, 20)?;
    let hashes = directory
        .get(start..)
        .filter(|_| start >= CODE_DIRECTORY_BYTES)
        .ok_or(refuse!("a plugin's code hashes exceed its directory"))?;
    let special_start = start
        .checked_sub(size_of_val(special))
        .ok_or(refuse!("a plugin's bundle hashes exceed its directory"))?;
    require!(
        identifier == CODE_DIRECTORY_BYTES
            && directory
                .get(identifier..special_start)
                .is_some_and(|name| name.last() == Some(&0))
            && hashes.len().is_multiple_of(HASH_BYTES)
            && hashes.len() / HASH_BYTES == word(directory, 28)?
            && word(directory, 28)? == signed.len().div_ceil(HASH_PAGE as usize),
        "a plugin's code hash table is malformed"
    );
    require!(
        directory[special_start..start] == *special.as_flattened(),
        "a plugin's Info.plist or CodeResources hash differs from its signature"
    );
    for (page, hash) in signed
        .chunks(HASH_PAGE as usize)
        .zip(hashes.as_chunks::<HASH_BYTES>().0)
    {
        require!(
            Sha256::digest(page).as_slice() == hash,
            "a plugin's signed page has changed"
        );
    }
    let mut hash = CodeHash::default();
    hash.copy_from_slice(&Sha256::digest(directory)[..size_of::<CodeHash>()]);
    Ok(hash)
}

/// Links each slice's objects and joins the slices into one fat file, whose
/// bytes go in the bundle as its executable. Each slice's signature names
/// `identifier`.
pub fn link(identifier: &str, slices: &[Vec<&[u8]>]) -> Result<Vec<u8>> {
    link_slices(identifier, slices, None)
}

/// Links a bundle whose code directory also seals Info.plist and CodeResources.
pub fn link_bundle(identifier: &str, slices: &[Vec<&[u8]>], seal: BundleSeal) -> Result<Vec<u8>> {
    link_slices(identifier, slices, Some(seal))
}

fn link_slices(
    identifier: &str,
    slices: &[Vec<&[u8]>],
    seal: Option<BundleSeal>,
) -> Result<Vec<u8>> {
    let mut linked = Vec::with_capacity(slices.len());
    for objects in slices {
        let slice = Slice::new(objects)?;
        let subtype = slice.subtype;
        let uuid = uuid(iter::once(identifier.as_bytes()).chain(objects.iter().copied()));
        linked.push((subtype, slice.link(identifier, uuid, seal)?));
    }
    let mut file = Vec::new();
    let mut word = |value: u32| file.extend_from_slice(&value.to_be_bytes());
    word(macho::FAT_MAGIC);
    word(fit(linked.len(), "the slice count")?);
    let mut offset = PAGE;
    for (subtype, slice) in &linked {
        word(macho::CPU_TYPE_ARM64);
        word(*subtype);
        word(fit(offset, "a slice's offset")?);
        word(fit(slice.len(), "a slice's size")?);
        word(PAGE.trailing_zeros());
        offset = (offset + slice.len() as u64).next_multiple_of(PAGE);
    }
    for (_, slice) in linked {
        file.resize(file.len().next_multiple_of(PAGE as usize), 0);
        file.extend_from_slice(&slice);
    }
    Ok(file)
}

#[cfg(test)]
mod tests;
