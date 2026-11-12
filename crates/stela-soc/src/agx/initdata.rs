//! The initdata root and the objects it names, as the 26A434 kext builds
//! them. The layout differs from Asahi's 13.x
//! `InitData` (`unk_buf`, runtime pointers, globals, firmware status, UAT
//! level info), so nothing here reuses it; each object is named by the kext
//! field that owns it.
//!
//! Known (`AGXG14G`, kernel file offsets):
//! - The root is one page (`AGXArmFirmware::allocFirmwareData` 0x1b5bdc0).
//!   `AGXArmFirmware::initFirmwareData` (0x1b5b35c..0x1b5b4f0) writes
//!   the family's magic at +0x00 ([`super::generation::Family::
//!   initdata_magic`]), the firmware VA of the page at
//!   `AGXArmFirmware+0x1660` at +0x08, the 64-bit firmware VAs cached at
//!   `AGXFirmware+0x830`, +0x358, +0x848, +0x928, +0x930 and +0x368 at +0x18,
//!   +0x20, +0xa8, +0xb0, +0xb8 and +0xc0, the words 0 and 1 at +0x28 and
//!   +0x2c, and 0x68 bytes of configuration (+0x18..+0x80 of the accelerator's
//!   object at +0x18a38) at +0x30..+0x98. Nothing else of the root is written.
//! - `AGXFirmware::allocFirmwareData` (0x1b359f0) sizes those objects from a
//!   table of (CPU pointer, firmware VA, bytes): the ring directory 0x4c8
//!   (`+0x810`/`+0x830`), 0xe48 (`+0x350`/`+0x358`), 0x20 (`+0x840`/`+0x848`),
//!   0xe850 (`+0x918`/`+0x928`), 0x698 (`+0x920`/`+0x930`) and 0x60
//!   (`+0x360`/`+0x368`).
//! - The ring directory's +0x20 is the AP-to-firmware descriptor table
//!   (`AGXFirmware+0x818` = directory + 0x20): for each submission index
//!   0..4 a 0x60-byte group of three descriptors for the TA (+0x00), 3D
//!   (+0x20) and CL (+0x40) data-master rings (the rings at `AGXFirmware+0x380`,
//!   +0x420 and +0x4c0, stride 0x28, that `AGXFirmware::submitTA/3D/CLChannel`
//!   0x1b32b2c/0x1b328d4/0x1b32568 index by their submission index), then the
//!   device-control ring's at table +0x180 (directory +0x1a0). The directory's
//!   +0x1c0..+0x200 holds eight firmware VAs for the firmware-to-AP rings
//!   (`AGXFirmware+0x820`, written at 0x1b349b0..0x1b34bcc); the first two
//!   name a 0x30-byte state block and a 0x4800-byte array, the firmware event
//!   ring (256 entries of 0x48).
//!
//! The internal fields of the other objects are not modeled: the firmware
//! model reads only the root and the ring directory.

use super::firmware::Pipe;
use super::ring::{self, Descriptor, Shared};
use crate::Result;
use stela_contracts::refuse;

/// Bytes of the root the kext writes (through +0xc0's pointer).
pub const ROOT_BYTES: usize = 0xc8;
/// The words at +0x28 and +0x2c.
const ROOT_CONSTANTS: (u32, u32) = (0, 1);
/// Bytes of the configuration copy at +0x30.
pub const CONFIG_BYTES: usize = 0x68;

/// The AP-to-firmware descriptor table's offset in the ring directory.
const DESCRIPTOR_TABLE: u64 = 0x20;
/// Bytes of one submission index's group of three work descriptors.
const WORK_GROUP: u64 = 0x60;
/// Submission indices per work kind.
pub const WORK_INDICES: u8 = 4;
/// The device-control descriptor's offset in the ring directory.
const DEVICE_CONTROL: u64 = DESCRIPTOR_TABLE + 0x180;
/// The firmware event ring's state and entry pointers in the directory.
const EVENT_STATE: u64 = 0x1c0;
const EVENT_ENTRIES: u64 = 0x1c8;

/// The initdata root.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Root {
    /// +0x08: the one-page object at `AGXArmFirmware+0x1660`.
    pub page_1660: u64,
    /// +0x18: the 0x4c8-byte ring directory (`AGXFirmware+0x830`).
    pub rings: u64,
    /// +0x20: the 0xe48-byte object at `AGXFirmware+0x358`.
    pub object_358: u64,
    /// +0xa8: the 0x20-byte object at `AGXFirmware+0x848`.
    pub object_848: u64,
    /// +0xb0: the 0xe850-byte object at `AGXFirmware+0x928`.
    pub object_928: u64,
    /// +0xb8: the 0x698-byte object at `AGXFirmware+0x930`.
    pub object_930: u64,
    /// +0xc0: the 0x60-byte object at `AGXFirmware+0x368`.
    pub object_368: u64,
    /// +0x30..+0x98: the accelerator configuration copy.
    pub config: [u8; CONFIG_BYTES],
}

impl Root {
    /// Parse the root's bytes, refusing a root that is not the kext's: its
    /// first word must be the family's `magic`.
    pub fn parse(bytes: &[u8; ROOT_BYTES], magic: u64) -> Result<Self> {
        let u64_at = |at: usize| u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap());
        let u32_at = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        if u64_at(0) != magic {
            if let Some((build, family)) =
                stela_soc_data::agx::generation::unchecked_initdata(u64_at(0))
            {
                return Err(refuse!(
                    "GFX initdata root is a later kext interface revision whose layouts the model has not checked",
                    build = %build,
                    family = %family,
                    found = %format_args!("{:#018x}", u64_at(0))
                ));
            }
            return Err(refuse!(
                "GFX initdata root lacks its magic",
                found = %format_args!("{:#018x}", u64_at(0)),
                second = %format_args!("{:#018x}", u64_at(8))
            ));
        }
        if (u32_at(0x28), u32_at(0x2c)) != ROOT_CONSTANTS {
            return Err(refuse!(
                "GFX initdata root's words at +0x28 are not 0 and 1"
            ));
        }
        let root = Self {
            page_1660: u64_at(0x08),
            rings: u64_at(0x18),
            object_358: u64_at(0x20),
            object_848: u64_at(0xa8),
            object_928: u64_at(0xb0),
            object_930: u64_at(0xb8),
            object_368: u64_at(0xc0),
            config: bytes[0x30..0x30 + CONFIG_BYTES].try_into().unwrap(),
        };
        let pointers = [
            root.page_1660,
            root.rings,
            root.object_358,
            root.object_848,
            root.object_928,
            root.object_930,
            root.object_368,
        ];
        if pointers.contains(&0) {
            return Err(refuse!("GFX initdata root leaves an object pointer empty"));
        }
        Ok(root)
    }

    /// Read the root at `va`, whose first word must be `magic`.
    pub fn read(memory: &mut dyn Shared, va: u64, magic: u64) -> Result<Self> {
        let mut bytes = [0; ROOT_BYTES];
        memory.read(va, &mut bytes)?;
        Self::parse(&bytes, magic)
    }

    /// The device-control ring.
    pub fn device_control(&self, memory: &mut dyn Shared) -> Result<Descriptor> {
        Descriptor::read(memory, self.rings + DEVICE_CONTROL)
    }

    /// The data-master ring of `pipe` at submission `index` (0..4).
    pub fn work(&self, memory: &mut dyn Shared, pipe: Pipe, index: u8) -> Result<Descriptor> {
        if index >= WORK_INDICES {
            return Err(refuse!("GFX work submission index names no ring"));
        }
        let va = self.rings
            + DESCRIPTOR_TABLE
            + u64::from(index) * WORK_GROUP
            + pipe as u64 * ring::DESCRIPTOR_BYTES as u64;
        Descriptor::read(memory, va)
    }

    /// The firmware event ring: (state, entries) firmware VAs.
    pub fn event_ring(&self, memory: &mut dyn Shared) -> Result<(u64, u64)> {
        let state = memory.read_u64(self.rings + EVENT_STATE)?;
        let entries = memory.read_u64(self.rings + EVENT_ENTRIES)?;
        if state == 0 || entries == 0 {
            return Err(refuse!("GFX firmware event ring is not populated"));
        }
        Ok((state, entries))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::agx::ring::tests::Flat;
    pub(crate) use stela_soc_data::agx::generation::MAGIC_G13_G14G as MAGIC;

    pub(crate) const BASE: u64 = 0xffff_ffa0_0000_0000;
    pub(crate) const ROOT: u64 = BASE;
    pub(crate) const RINGS: u64 = BASE + 0x1000;

    /// A root at [`ROOT`] whose ring directory is at [`RINGS`] and whose other
    /// objects follow it.
    pub(crate) fn kext_root(memory: &mut Flat) {
        memory.put_u64(ROOT, MAGIC);
        memory.put_u64(ROOT + 0x08, BASE + 0x2000);
        memory.put_u64(ROOT + 0x18, RINGS);
        memory.put_u64(ROOT + 0x20, BASE + 0x3000);
        memory.put_u64(ROOT + 0x28, 1 << 32);
        memory.put(ROOT + 0x30, &[0x5a; CONFIG_BYTES]);
        for (n, at) in [0xa8, 0xb0, 0xb8, 0xc0].into_iter().enumerate() {
            memory.put_u64(ROOT + at, BASE + 0x4000 + n as u64 * 0x1000);
        }
    }

    #[test]
    fn the_root_names_its_objects_by_the_kext_fields() {
        let mut memory = Flat::new(BASE, 0x10000);
        kext_root(&mut memory);
        let root = Root::read(&mut memory, ROOT, MAGIC).unwrap();
        assert_eq!(root.rings, RINGS);
        assert_eq!(root.page_1660, BASE + 0x2000);
        assert_eq!(root.object_358, BASE + 0x3000);
        assert_eq!(root.object_368, BASE + 0x7000);
        assert_eq!(root.config, [0x5a; CONFIG_BYTES]);
    }

    #[test]
    fn a_root_that_is_not_the_kexts_refuses() {
        let mut memory = Flat::new(BASE, 0x10000);
        kext_root(&mut memory);
        memory.put_u64(ROOT + 0x28, 0);
        assert!(Root::read(&mut memory, ROOT, MAGIC).is_err());
        kext_root(&mut memory);
        memory.put_u64(ROOT + 0xb0, 0);
        assert!(Root::read(&mut memory, ROOT, MAGIC).is_err());
        kext_root(&mut memory);
        memory.put_u64(ROOT, 0);
        assert_eq!(
            Root::read(&mut memory, ROOT, MAGIC).unwrap_err().what(),
            "GFX initdata root lacks its magic"
        );
    }

    #[test]
    fn a_later_kext_revisions_root_refuses_by_its_revision() {
        let mut memory = Flat::new(BASE, 0x10000);
        kext_root(&mut memory);
        memory.put_u64(ROOT, 0x0c8a_8342_8392_04c8);
        assert_eq!(
            Root::read(&mut memory, ROOT, MAGIC).unwrap_err().what(),
            "GFX initdata root is a later kext interface revision whose layouts the model has not checked"
        );
    }

    #[test]
    fn the_ring_directory_places_work_device_control_and_events() {
        let mut memory = Flat::new(BASE, 0x10000);
        kext_root(&mut memory);
        let root = Root::read(&mut memory, ROOT, MAGIC).unwrap();
        // 3D, submission index 2.
        memory.put_descriptor(RINGS + 0x20 + 2 * 0x60 + 0x20, BASE + 0x8000, BASE + 0x9000);
        memory.put_descriptor(RINGS + 0x1a0, BASE + 0x8100, BASE + 0xa000);
        memory.put_u64(RINGS + 0x1c0, BASE + 0x8200);
        memory.put_u64(RINGS + 0x1c8, BASE + 0xb000);
        let work = root.work(&mut memory, Pipe::Fragment, 2).unwrap();
        assert_eq!((work.state, work.entries), (BASE + 0x8000, BASE + 0x9000));
        let control = root.device_control(&mut memory).unwrap();
        assert_eq!(control.state, BASE + 0x8100);
        assert_eq!(
            root.event_ring(&mut memory).unwrap(),
            (BASE + 0x8200, BASE + 0xb000)
        );
        // Other slots are empty, and index 4 names no ring.
        assert!(root.work(&mut memory, Pipe::Vertex, 2).is_err());
        assert!(root.work(&mut memory, Pipe::Fragment, 4).is_err());
    }
}
