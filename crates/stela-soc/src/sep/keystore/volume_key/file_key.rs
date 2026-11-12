//! Per-file keys: the keys APFS gives each file on a volume whose class keys
//! are loaded, wrapped under the file's protection class.
//!
//! **Wire.** 26A434 kernel file offsets. `fs_new_key` (0x2d4f958) and
//! `fs_unwrap_key` (0x2d22d14) send the u64 1 every client request starts
//! with, the keybag handle `_fs_handle` (0x2d54e88) finds for the file's
//! volume, and the volume's encoded parameters
//! (`_encode_pfk_params_public` 0x2d5fb5c): a DER dictionary with the
//! 16-byte octet strings `persona_uuid` when the parameters' flags bit 0 is
//! set and `volume_uuid` when bit 1 is, which `_volume_handle` (0x2d5249c)
//! sets for a volume mapped to handle -5; no flags encode no dictionary.
//! - NEW_KEY 0x10 (`__ipc_new_pfk_data_v2` 0x2d59448, codec 0x2d3f6d0),
//!   variant 2 from `new_key` (0x2d51d48) for request type 1: u64, handle,
//!   parameters, class, flags, a 28-byte file-system context and an empty
//!   blob -> key, IV key, a blob the caller takes no buffer for, wrapped
//!   key, another such blob, flags and the wrapped key's class, which lands
//!   at `aks_wrapped_key_s` + 12. `check_class` (0x2d25df4), APFS's class
//!   availability check, sends the same command with flags 0, no context
//!   and no buffer for any reply field: no key is made, and the class is
//!   available when its volume's class keys are loaded.
//! - UNWRAP_KEY 0x09 (`__ipc_unwrap_pfk_data_v1` 0x2d58d74, codec
//!   0x2d3e22c), variant 1 from `unwrap_data` (0x2d25f78) for request type
//!   1: u64, handle, parameters, the wrapped key's class, wrapped key, flags
//!   -> key, IV key, flags.
//!
//! The request flags are 2 for a new key and 0x102 for an unwrap, with bit
//! 0 set when the caller takes no key. The reply flags' bit 1 becomes the raw
//! key's `CP_RAW_KEY_WRAPPEDKEY` (0x2d51f28, 0x2d26354). Other request types
//! send other variants (unwrap variant 0), which refuse.
//!
//! **Rewrap.** REWRAP 0x0f (`__ipc_rewrap_pfk_data_v3` 0x2d591e0, codec
//! 0x2d3efdc), variant 3 from `rewrap_data_with_opts` (0x2d51bf8): u64, an
//! empty blob, parameters, two handles, the old and new classes, two empty
//! blobs, wrapped key, flags, two file-system contexts -> wrapped key, class.
//! `fs_rewrap_key` (0x2d22f50) sends flags 0, `fs_rewrap_ekwk` (0x2d50820)
//! 0x200 and `fs_migrate_media_key_to_class` (0x2d4ffa0) 0x40, the last with
//! no parameters: the key stays and its wrapping moves to the new class.
//! `fs_rewrap_ek` (26B123 0xfffffe0009d6f6b4), which APFS's `icp_clone_ek`
//! sends when a write splits a cloned file's extents, fills the two blobs
//! with the old and new ephemeral keys, the classes with theirs, and sends
//! flags 0x100: the wrapped EK moves from the old EKWK to the new. APFS
//! panics ("cannot rewrap ek") when that fails.
//!
//! **Scalable keys.** APFS gives a file an EK, its file key, wrapped by an
//! EKWK, wrapped under the protection class (`new_skey` 0x491a7d0), with
//! the unwrapped EKWK as an ephemeral key it keeps in memory (at most 64
//! bytes, `alloc_new_ekwk`). Request type 2 sets flags bit 8:
//! - `fs_new_ekwk_ek` 0x2d50254: NEW_KEY with no trailing blob -> key, IV
//!   key, ephemeral key, wrapped EK, wrapped EKWK, in the blobs `new_key`
//!   (0x2d51d48) gives `fs_new_ekwk_ek`'s third, sixth and fourth arguments.
//! - `fs_new_ek` 0x2d50484: NEW_KEY class 0, the ephemeral key as the
//!   trailing blob -> key, IV key, wrapped EK in the wrapped key's place.
//! - UNWRAP_KEY variant 2 (`__ipc_unwrap_pfk_data_v2` 0x2d58f50): u64,
//!   handle, parameters, class, wrapped EK, wrapped EKWK, ephemeral key,
//!   flags -> key, IV key, ephemeral key, flags. `fs_unwrap_ekwk` 0x2d23238
//!   sends the wrapped EKWK under its class, with a wrapped EK if it wants
//!   the key too; `fs_unwrap_ek` 0x2d50660 sends class 0, a wrapped EK and
//!   the ephemeral key.
//!
//! **Keys.** The key is the inline engine's wrapped form, as for a VEK. The
//! IV key is 16 bytes derived from the key, so an unwrap returns what the new
//! key did: `_return_ek` (0x491b02c, `new_skey`'s path) panics on any other
//! length and gives it to `cpx`; the inline engine takes its tweak from the
//! command and never uses it. Loading a volume's class keys (protection
//! operation 18) keeps a class root derived from its VEK until the sepOS
//! restarts or the VEK's state is dropped; each class key derives from it. A wrapped key
//! is Stela's format: magic, class and volume UUID, then the key sealed
//! under that class key, so it opens only on its machine, volume and class
//! once the volume's class keys are loaded. A wrapped EKWK is such a key; an
//! EK is sealed under its EKWK, and an ephemeral key is the EKWK sealed under
//! a key of the machine's. The volume comes from the
//! parameters' `volume_uuid`; a request without one refuses. Keybag
//! handles, personas and file-system contexts are not
//! interpreted, and classes differ only in their keys: none is ever locked,
//! so class A and B keys stay available. That is reported once.
use super::super::super::Entropy;
use super::super::Fields;
use super::super::dictionary::octet_strings;
use super::{KEY_BYTES, Reply, UUID_BYTES, VolumeKeys, derive, random};
use crate::Result;
use crate::inline_key::{SEALED_BYTES, open, seal};
use alloc::{string::String, vec::Vec};
use stela_contracts::refuse;
use zeroize::Zeroizing;

pub(super) const UNWRAP_KEY: u8 = 0x09;
pub(super) const REWRAP: u8 = 0x0f;
pub(super) const NEW_KEY: u8 = 0x10;

const NEW_KEY_VARIANT: u32 = 2;
const UNWRAP_KEY_VARIANT: u32 = 1;
const UNWRAP_SCALABLE_VARIANT: u32 = 2;
const REWRAP_VARIANT: u32 = 3;
/// The u64 `new_key` and `unwrap_data` send first.
const CLIENT: u64 = 1;
/// Request flags: the caller takes no key; it wants a SEP-wrapped key; the
/// request is of type 2.
const NO_KEY: u32 = 1;
const WRAPPED_KEY: u32 = 1 << 1;
const SCALABLE: u32 = 1 << 8;
/// Rewrap flags: `fs_migrate_media_key_to_class`'s and `fs_rewrap_ekwk`'s.
const REWRAP_MEDIA_KEY: u32 = 1 << 6;
const REWRAP_EKWK: u32 = 1 << 9;
/// `check_class`'s request flags.
const CHECK_CLASS: u32 = 0;
/// Reply flags: the key is SEP-wrapped.
const SEP_WRAPPED: u32 = 1 << 1;
const CONTEXT_BYTES: usize = 28;

const MAGIC: [u8; 4] = *b"SFK1";
/// Magic, class, volume UUID; then the sealed key.
const HEADER: usize = 4 + 4 + UUID_BYTES;
pub(super) const WRAPPED_BYTES: usize = HEADER + SEALED_BYTES;
const EK_MAGIC: [u8; 4] = *b"SEK1";
pub(super) const WRAPPED_EK_BYTES: usize = EK_MAGIC.len() + SEALED_BYTES;

const CLASS_ROOT_DOMAIN: &[u8] = b"Stela SEP class keys v1\0";
const CLASS_KEY_DOMAIN: &[u8] = b"Stela SEP class key v1\0";
const FILE_KEY_DOMAIN: &[u8] = b"Stela SEP per-file key v1\0";
const EK_DOMAIN: &[u8] = b"Stela SEP EK v1\0";
const EPHEMERAL_DOMAIN: &[u8] = b"Stela SEP ephemeral key v1\0";
const IV_KEY_DOMAIN: &[u8] = b"Stela SEP per-file IV key v1\0";
/// `iv_key_len`, the only length `_return_ek` accepts.
const IV_KEY_BYTES: usize = 16;

pub(super) const UNINTERPRETED: &str = "SEP per-file keys: keybag handles, personas, \
     and file-system contexts are not interpreted, and no class is ever locked";

pub(super) fn recognizes(command: u8) -> bool {
    matches!(command, NEW_KEY | UNWRAP_KEY)
}

/// A volume's loaded class keys.
pub(super) struct ClassKeys {
    volume: [u8; UUID_BYTES],
    root: Zeroizing<[u8; 32]>,
}

impl ClassKeys {
    pub(super) fn new(volume: [u8; UUID_BYTES], vek: &[u8; KEY_BYTES]) -> Self {
        Self {
            volume,
            root: derive(vek, CLASS_ROOT_DOMAIN, &volume),
        }
    }

    pub(super) fn volume(&self) -> [u8; UUID_BYTES] {
        self.volume
    }

    fn key(&self, class: u32) -> Zeroizing<[u8; 32]> {
        derive(&self.root, CLASS_KEY_DOMAIN, &class.to_le_bytes())
    }
}

impl VolumeKeys {
    pub(super) fn file_key(
        &mut self,
        command: u8,
        variant: u32,
        fields: &mut Fields<'_>,
        entropy: &mut dyn Entropy,
        notes: &mut dyn FnMut(String),
    ) -> Result<Vec<u8>> {
        let expected = match command {
            NEW_KEY => variant == NEW_KEY_VARIANT,
            _ => matches!(variant, UNWRAP_KEY_VARIANT | UNWRAP_SCALABLE_VARIANT),
        };
        if !expected {
            return Err(refuse!("SEP per-file key command variant is not modeled"));
        }
        if fields.lword()? != CLIENT {
            return Err(refuse!(
                "SEP per-file key request does not start with the kernel's u64"
            ));
        }
        let _handle = fields.word()?;
        let volume = volume(fields.blob()?)?;
        let class = fields.word()?;
        let mut reply = Reply::default();
        if command == UNWRAP_KEY && variant == UNWRAP_SCALABLE_VARIANT {
            reply = self.unwrap_scalable(volume, class, fields)?;
            self.report(UNINTERPRETED, notes);
            return Ok(reply.0);
        }
        if command == NEW_KEY {
            let request = fields.word()?;
            let context = fields.blob()?.len();
            let trailing = fields.blob()?;
            if request & SCALABLE != 0 {
                flags(request)?;
                if context != CONTEXT_BYTES {
                    return Err(refuse!("SEP new per-file key context is not modeled"));
                }
                fields.end()?;
                reply = self.new_scalable(volume, class, trailing, entropy)?;
                self.report(UNINTERPRETED, notes);
                return Ok(reply.0);
            }
            let trailing = trailing.len();
            if class == 0 {
                return Err(refuse!("SEP per-file key names no protection class"));
            }
            if request == CHECK_CLASS && context == 0 && trailing == 0 {
                fields.end()?;
                self.class_keys(volume)?;
                for _ in 0..5 {
                    reply.blob(&[]);
                }
                reply.word(0);
                reply.word(class);
                self.report(UNINTERPRETED, notes);
                return Ok(reply.0);
            }
            flags(request)?;
            if context != CONTEXT_BYTES || trailing != 0 {
                return Err(refuse!(
                    "SEP new per-file key context or trailing blob is not modeled"
                ));
            }
            fields.end()?;
            let classes = self.class_keys(volume)?;
            let key = random(entropy)?;
            let wrapped = wrap(classes, class, &key);
            self.key_reply(&mut reply, &key)?;
            reply.blob(&[]);
            reply.blob(&wrapped);
            reply.blob(&[]);
            reply.word(SEP_WRAPPED);
            reply.word(class);
        } else {
            if class == 0 {
                return Err(refuse!("SEP per-file key names no protection class"));
            }
            let wrapped = fields.blob()?;
            flags(fields.word()?)?;
            fields.end()?;
            let key = unwrap(self.class_keys(volume)?, class, wrapped)?;
            self.key_reply(&mut reply, &key)?;
            reply.word(SEP_WRAPPED);
        }
        self.report(UNINTERPRETED, notes);
        Ok(reply.0)
    }

    /// Move a wrapped key from class `old` to class `new`.
    pub(super) fn rewrap(
        &mut self,
        variant: u32,
        fields: &mut Fields<'_>,
        notes: &mut dyn FnMut(String),
    ) -> Result<Vec<u8>> {
        if variant != REWRAP_VARIANT {
            return Err(refuse!("SEP rewrap command variant is not modeled"));
        }
        if fields.lword()? != CLIENT {
            return Err(refuse!(
                "SEP rewrap request does not start with the kernel's u64"
            ));
        }
        let first = fields.blob()?.len();
        let parameters = fields.blob()?;
        let (_handle, _new_handle) = (fields.word()?, fields.word()?);
        let (old, new) = (fields.word()?, fields.word()?);
        let (ephemeral, new_ephemeral) = (fields.blob()?, fields.blob()?);
        let wrapped = fields.blob()?;
        let request = fields.word()?;
        let (_context, _new_context) = (fields.blob()?, fields.blob()?);
        fields.end()?;
        let scalable = request == SCALABLE;
        let ephemerals = [ephemeral, new_ephemeral].map(|key| !key.is_empty());
        if first != 0 || ephemerals != [scalable; 2] {
            return Err(refuse!("SEP rewrap request's other keys are not modeled"));
        }
        if new == 0 && !scalable {
            return Err(refuse!("SEP rewrap names no new class"));
        }
        let mut reply = Reply::default();
        match request {
            SCALABLE => {
                let key = unwrap_ek(&*self.open_ephemeral(ephemeral)?, wrapped)?;
                reply.blob(&wrap_ek(&*self.open_ephemeral(new_ephemeral)?, &key));
            }
            REWRAP_MEDIA_KEY => reply.blob(&self.rewrap_media_key(old, new, wrapped)?),
            0 | REWRAP_EKWK => {
                let classes = self.class_keys(volume(parameters)?)?;
                reply.blob(&wrap(classes, new, &*unwrap(classes, old, wrapped)?));
            }
            _ => return Err(refuse!("SEP rewrap flags are not modeled")),
        }
        reply.word(new);
        self.report(UNINTERPRETED, notes);
        Ok(reply.0)
    }

    /// A new EK under the EKWK the `ephemeral` key holds, or with no
    /// ephemeral key under a new EKWK of `class`.
    fn new_scalable(
        &self,
        volume: [u8; UUID_BYTES],
        class: u32,
        ephemeral: &[u8],
        entropy: &mut dyn Entropy,
    ) -> Result<Reply> {
        let key = random(entropy)?;
        let mut reply = Reply::default();
        if ephemeral.is_empty() {
            if class == 0 {
                return Err(refuse!("SEP new EKWK names no protection class"));
            }
            let classes = self.class_keys(volume)?;
            let wrapping = random(entropy)?;
            self.key_reply(&mut reply, &key)?;
            reply.blob(&self.ephemeral(&wrapping));
            reply.blob(&wrap_ek(&wrapping, &key));
            reply.blob(&wrap(classes, class, &wrapping));
        } else {
            if class != 0 {
                return Err(refuse!("SEP new EK names a protection class"));
            }
            let wrapping = self.open_ephemeral(ephemeral)?;
            self.key_reply(&mut reply, &key)?;
            reply.blob(&[]);
            reply.blob(&wrap_ek(&wrapping, &key));
            reply.blob(&[]);
        }
        reply.word(SEP_WRAPPED);
        reply.word(class);
        Ok(reply)
    }

    /// UNWRAP_KEY variant 2's reply to its `fields` after the class: the
    /// EKWK comes from the ephemeral key, or from its wrapped form under
    /// `class`, which then answers an ephemeral key.
    fn unwrap_scalable(
        &self,
        volume: [u8; UUID_BYTES],
        class: u32,
        fields: &mut Fields<'_>,
    ) -> Result<Reply> {
        let wrapped_ek = fields.blob()?;
        let wrapped_ekwk = fields.blob()?;
        let ephemeral = fields.blob()?;
        let request = fields.word()?;
        flags(request)?;
        fields.end()?;
        let (wrapping, opened) = if ephemeral.is_empty() {
            let wrapping = unwrap(self.class_keys(volume)?, class, wrapped_ekwk)?;
            let opened = self.ephemeral(&wrapping);
            (wrapping, Some(opened))
        } else {
            (self.open_ephemeral(ephemeral)?, None)
        };
        let mut reply = Reply::default();
        if request & NO_KEY != 0 || wrapped_ek.is_empty() {
            reply.blob(&[]);
            reply.blob(&[]);
        } else {
            self.key_reply(&mut reply, &*unwrap_ek(&wrapping, wrapped_ek)?)?;
        }
        reply.blob(opened.as_ref().map_or(&[], |key| key.as_slice()));
        reply.word(SEP_WRAPPED);
        Ok(reply)
    }

    fn ephemeral(&self, wrapping: &[u8; KEY_BYTES]) -> [u8; SEALED_BYTES] {
        seal(&self.ephemeral_key(), EPHEMERAL_DOMAIN, &[], wrapping)
    }

    fn open_ephemeral(&self, ephemeral: &[u8]) -> Result<Zeroizing<[u8; KEY_BYTES]>> {
        let sealed = ephemeral
            .try_into()
            .map_err(|_| refuse!("SEP ephemeral key is not a Stela key"))?;
        open(&self.ephemeral_key(), EPHEMERAL_DOMAIN, &[], sealed)
            .map_err(|_| refuse!("SEP ephemeral key does not open on this machine"))
    }

    fn ephemeral_key(&self) -> Zeroizing<[u8; 32]> {
        derive(&self.root, EPHEMERAL_DOMAIN, &[])
    }

    fn class_keys(&self, volume: [u8; UUID_BYTES]) -> Result<&ClassKeys> {
        self.class_keys
            .iter()
            .find(|keys| keys.volume == volume)
            .ok_or(refuse!("SEP per-file key volume has no loaded class keys"))
    }

    /// The key and its IV key.
    fn key_reply(&self, reply: &mut Reply, key: &[u8; KEY_BYTES]) -> Result<()> {
        reply.blob(&self.engine.wrap(key)?);
        reply.blob(&derive(key, IV_KEY_DOMAIN, &[])[..IV_KEY_BYTES]);
        Ok(())
    }
}

/// The volume UUID the encoded parameters name.
fn volume(parameters: &[u8]) -> Result<[u8; UUID_BYTES]> {
    let entries = octet_strings(parameters)?;
    let uuid = entries
        .iter()
        .find(|(key, _)| *key == b"volume_uuid")
        .ok_or(refuse!("SEP per-file key parameters name no volume"))?
        .1;
    if entries.iter().any(|(key, value)| {
        !matches!(*key, b"volume_uuid" | b"persona_uuid") || value.len() != UUID_BYTES
    }) {
        return Err(refuse!("SEP per-file key parameters are not modeled"));
    }
    Ok(uuid.try_into().unwrap())
}

/// Request flags: a SEP-wrapped key, optionally not taken.
fn flags(flags: u32) -> Result<()> {
    if flags & WRAPPED_KEY == 0 || flags & !(NO_KEY | WRAPPED_KEY | SCALABLE) != 0 {
        return Err(refuse!("SEP per-file key request flags are not modeled"));
    }
    Ok(())
}

fn header(class: u32, volume: [u8; UUID_BYTES]) -> [u8; HEADER] {
    let mut header = [0; HEADER];
    header[..4].copy_from_slice(&MAGIC);
    header[4..8].copy_from_slice(&class.to_le_bytes());
    header[8..].copy_from_slice(&volume);
    header
}

fn wrap(classes: &ClassKeys, class: u32, key: &[u8; KEY_BYTES]) -> Vec<u8> {
    let header = header(class, classes.volume);
    let mut wrapped = header.to_vec();
    wrapped.extend_from_slice(&seal(&classes.key(class), FILE_KEY_DOMAIN, &header, key));
    wrapped
}

fn wrap_ek(wrapping: &[u8; KEY_BYTES], key: &[u8; KEY_BYTES]) -> Vec<u8> {
    let mut wrapped = EK_MAGIC.to_vec();
    wrapped.extend_from_slice(&seal(wrapping, EK_DOMAIN, &EK_MAGIC, key));
    wrapped
}

fn unwrap_ek(wrapping: &[u8; KEY_BYTES], wrapped: &[u8]) -> Result<Zeroizing<[u8; KEY_BYTES]>> {
    let (Some(sealed), true) = (
        wrapped
            .strip_prefix(&EK_MAGIC)
            .and_then(|sealed| sealed.try_into().ok()),
        wrapped.len() == WRAPPED_EK_BYTES,
    ) else {
        return Err(refuse!("SEP wrapped EK is not a Stela key"));
    };
    open(wrapping, EK_DOMAIN, &EK_MAGIC, sealed)
        .map_err(|_| refuse!("SEP wrapped EK does not open under its EKWK"))
}

fn unwrap(classes: &ClassKeys, class: u32, wrapped: &[u8]) -> Result<Zeroizing<[u8; KEY_BYTES]>> {
    if wrapped.len() != WRAPPED_BYTES || wrapped[..HEADER] != header(class, classes.volume) {
        return Err(refuse!(
            "SEP wrapped per-file key is not a Stela key of this class and volume"
        ));
    }
    open(
        &classes.key(class),
        FILE_KEY_DOMAIN,
        &wrapped[..HEADER],
        wrapped[HEADER..].try_into().unwrap(),
    )
    .map_err(|_| refuse!("SEP wrapped per-file key does not open under its class key"))
}
