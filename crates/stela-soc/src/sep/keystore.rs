//! The AppleKeyStore service, `sks`, seen from the SEP.
//!
//! The AP side is `AppleKeyStore` in the 26A434 kernel (file offsets).
//!
//! **Frame.** `sep_deliver_msg_gated` 0x2d27158..0x2d27168 sends byte 1 the
//! command (seven bits), byte 2 a sequence number from an atomic counter
//! (0x2d26b14), bytes 6..7 the OOL request length and every other byte zero.
//! `sep_action` 0x2d27d4c..0x2d27da0 takes a frame with byte 1 bit 7 set as
//! the reply. The waiter (0x2d27470..0x2d27490) requires its sequence in
//! byte 2 and reads byte 3 as the status, bytes 4..5 as reply call flags
//! (`_msg_process_ipc_call_flags` 0x2d4b0bc) and bytes 6..7 as the reply's
//! OOL length, which must fit its receive buffer (0x2d2764c). A nonzero
//! status, sign-extended, becomes the request's result (0x2d275e8). In
//! [`Frame`] fields: tag = command, plus [`REPLY`] in a reply; opcode = the
//! sequence; parameter = the status; data = flags | length << 16.
//!
//! **Message.** Each command's `_code_ipc_*` codec (`_code_ipc_set_env`
//! 0x2d43040, `_code_ipc_get_capabilities` 0x2d47c20) writes:
//! - a header (`_ipc_header` 0x2d4b190): a u32 length L, 72 for version 1
//!   and 80 for version 2 (`_ipc_header_effective_size` 0x2d4b060), then L
//!   bytes of `_gen_ipc_header` 0x2d4ae80's record: a 16-byte digest, the
//!   u32 version, a u64 time and u32 flags at message offsets 4, 20, 24 and
//!   32, then further client fields;
//! - the digest, SHA-256 over everything from the version to the message
//!   end, truncated to 16 bytes (`_payload_hash` 0x2d4b7ac with
//!   `ccsha256_di`); flags bit 0 skips it, and bit 1 announces an extended
//!   header in a version-2 reply (`_ipc_msg_process_extended_header`
//!   0x2d5df14);
//! - a u32 **command variant** (`_word` 0x2d5dba4), not a result: startup
//!   uses 0, create uses 4 and copy uses 1 in the AP builders below;
//!   status is the mailbox parameter, sign-extended by the AP;
//! - the command's fields: u64 (`_lword` 0x2d5dd70) and blobs (`_blob`
//!   0x2d5dbfc: a u32 length, the bytes, zero padding to four).
//!
//! A reply decoder consumes exactly the reply's length (0x2d431b0).
//!
//! **Startup.** `init_sep_endpoint` 0x2d27eac sends GET_CAPABILITIES (0x4d,
//! `__ipc_get_capabilities` 0x2d5c954) with u64 1 and an empty blob; the
//! reply carries variant 0, a u64 and a blob. The u64 is the SEP's highest
//! IPC header version: the kernel negotiates the lower of it and 2
//! (0x2d27f9c..0x2d27fac, "negotiated to ipc header theirs:v%llu, ours:v%u")
//! and falls back to version 1 on failure. It then sends SET_ENVIRONMENT
//! (0x2a, `__ipc_set_env` 0x2d3b69c from `set_env` 0x2d2a80c) with u64 1 and
//! a 1036-byte environment: u32 1, the u32 `/defaults` `cpx-encryption-mode`
//! (0 when absent), u32 4 (6 when `set_env`'s Boolean is set), then zeros.
//! Its reply carries variant 0 alone.
//!
//! This SEP parses both header versions, so it reports version 2, and it
//! records the environment, which nothing consumes yet. [`keybag`] creates,
//! saves, loads and unloads user keybags; a request that needs a passcode
//! KEK is answered after its derivation, which runs over later SEP steps
//! ([`KeyStore::resume`]). [`volume_key`] serves volume, class and per-file
//! keys. The failure answer is the status
//! [`FAILURE`], Stela's choice of generic failure: the AP sign-extends
//! 0xff to -1, then maps it to IOReturn 0xe00002bc (0x2d3a324, table
//! 0x725990). No sepOS-specific authorization error is invented.
mod dictionary;
mod keybag;
mod reference_key;
mod system_key;
mod volume_key;

use super::{
    endpoint::Service,
    message::Frame,
    observe::Journal,
    transaction::{Answer, Payload, Refusal},
};
use crate::Result;
use alloc::{string::String, vec, vec::Vec};
use stela_contracts::refuse;
use stela_crypto::Sha256;
use stela_crypto::digest::Digest;
use zeroize::Zeroizing;

/// Stela's catalog ID for the service, after the xART pair.
pub const ID: u8 = 0x12;
pub const NAME: u32 = u32::from_be_bytes(*b"sks\0");
/// `sep_endpoint` 0x2d574a0..0x2d574c8 and 0x2d576c8 size each OOL buffer
/// to the advertised maximum pages. Two 16 KiB pages each way is Stela's
/// choice, not a recovered sepOS capacity.
pub const CAPACITIES: [u8; 4] = [2, 2, 2, 2];

/// Byte 1's reply bit; the command is the low seven bits.
pub const REPLY: u8 = 0x80;
const LENGTH_SHIFT: u32 = 16;
const CALL_FLAGS: u32 = 0xffff;
/// Stela's refusal status: -1, mapped by the AP to generic IOReturn failure.
pub const FAILURE: u8 = 0xff;

/// Whether `command`'s request can carry a passcode or another secret:
/// create (0x01), change lock state (0x04), change secret (0x07), make
/// system (0x0d), the escrow stash (0x13, 0x14), device state transition
/// (0x18) and verify (0x21). Observation records only their lengths.
pub fn carries_secret(command: u8) -> bool {
    matches!(
        command,
        0x01 | 0x04 | 0x07 | 0x0d | 0x13 | 0x14 | 0x18 | 0x21
    )
}

pub const GET_CAPABILITIES: u8 = 0x4d;
pub const SET_ENVIRONMENT: u8 = 0x2a;
/// `effaceable_init` (0x2d258c8) sends NULL_D_KEY (`__ipc_null_d_key`
/// 0x2d3a618) with variant 0 and u64 1 when `/defaults` has
/// `no-effaceable-storage`; the reply is the variant alone. This SEP never
/// keeps keys in effaceable storage, so it acknowledges.
pub const NULL_D_KEY: u8 = 0x0a;
/// `get_device_state` (0x2d21fe8) sends GET_DEVICE_STATE
/// (`__ipc_get_device_state` 0x2d59b34, `_v1` 0x2d59c24, codec 0x2d40a28):
/// u64 session, u32 handle, then u32 0, or a blob and a u32 in variant 1.
/// The reply is a blob, the DER dictionary `decode_extended_state`
/// (0x2d5f570) reads.
pub const GET_DEVICE_STATE: u8 = 0x19;
/// `__ipc_client_terminate_v1` (0x2d59e14, codec 0x2d40e88): three u64s and
/// a u32 in variant 1 and no reply fields. This SEP keeps nothing per client,
/// so it acknowledges.
pub const CLIENT_TERMINATE: u8 = 0x1b;
/// CLIENT_TERMINATE's reply fields, which `client_terminate` passes to
/// `handle_device_state_return`: two u64s of state and the keybag they
/// belong to, -1 as the kernel starts it, for none.
const TERMINATED: [u8; 20] = {
    let mut fields = [0; 20];
    fields[16] = 0xff;
    fields[17] = 0xff;
    fields[18] = 0xff;
    fields[19] = 0xff;
    fields
};
/// `ss`: never locked and no passcode, from the keybag state flags
/// `keybag_state_no_pin` (bit 1) and `keybag_state_been_unlocked` (bit 2).
/// The kernel passes `ss` unchanged as selector 7's scalar
/// (`handleUserClientSelector` case 7 stores `decode_extended_state` field 0,
/// `der_key_state_state` = "ss"), and `MKBDeviceUnlockedSinceBoot`,
/// `MKBUserUnlockedSinceBoot` and secd's "Keybag never unlocked" test bit 2.
const UNLOCKED_WITHOUT_PASSCODE: u8 = 0b0110;

const LENGTH_BYTES: usize = 4;
const DIGEST_START: usize = 4;
const DIGEST_BYTES: usize = 16;
const VERSION: usize = 20;
const FLAGS: usize = 32;
const HEADER_V1: usize = 72;
const HEADER_V2: usize = 80;
const VARIANT_BYTES: usize = 4;
/// The highest IPC header version this SEP parses.
const HEADER_VERSION: u64 = 2;
/// The u64 the kernel sends with both startup commands; its meaning is
/// not recovered, so no other value is accepted.
const STARTUP_ARGUMENT: u64 = 1;
const ENVIRONMENT_BYTES: usize = 1036;
const ENVIRONMENT_VERSION: u32 = 1;
const ENVIRONMENT_MODES: [u32; 2] = [4, 6];

pub fn service() -> Service {
    Service {
        id: ID,
        name: NAME,
        capacities: CAPACITIES,
    }
}

/// The failure reply to `frame`: its command with the reply bit, its
/// sequence, the [`FAILURE`] status and no OOL bytes.
pub fn failure(frame: Frame) -> Frame {
    Frame {
        tag: frame.tag | REPLY,
        parameter: FAILURE,
        data: 0,
        ..frame
    }
}

/// A keybag refusal's answer: a wrong passcode with
/// [`keybag::BAD_PASSCODE_STATUS`], anything else with [`FAILURE`].
fn keybag_refusal(frame: Frame, reason: stela_contracts::Refusal) -> Refusal {
    let mut answer = failure(frame);
    if reason.what() == keybag::BAD_PASSCODE {
        answer.parameter = keybag::BAD_PASSCODE_STATUS;
    }
    Refusal::answered(reason, Answer::frame(answer))
}

/// The environment the kernel set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Environment {
    /// `/defaults` `cpx-encryption-mode`, 0 when the DeviceTree has none.
    pub cpx_encryption_mode: u32,
    /// 4, or 6 when `set_env`'s Boolean is set; its meaning is not
    /// recovered.
    pub mode: u32,
}

#[derive(Default)]
pub struct KeyStore {
    environment: Option<Environment>,
    system_keys: Option<system_key::SystemKeys>,
    volume_keys: Option<volume_key::VolumeKeys>,
    diagnostics: Journal,
    /// Commands already reported as unimplemented.
    unimplemented: Vec<u8>,
    /// Commands whose refusal was already reported.
    refused: Vec<u8>,
    /// Whether the unmodeled lock states were reported.
    state_reported: bool,
    keybags: Option<keybag::Keybags>,
    /// The request whose passcode derivation is running.
    deferred: Option<Deferred>,
}

/// A request answered once its derivation completes: its frame, and its
/// message's header and variant for the reply. No request secret is kept.
struct Deferred {
    frame: Frame,
    prefix: Vec<u8>,
    pending: alloc::boxed::Box<keybag::Pending>,
}

impl KeyStore {
    pub(super) fn with_system_keys(root: Zeroizing<[u8; 32]>) -> Self {
        Self {
            keybags: Some(keybag::Keybags::new(&root)),
            system_keys: Some(system_key::SystemKeys::new(root)),
            ..Self::default()
        }
    }

    /// As [`Self::with_system_keys`], with keybags created at `iterations`.
    #[cfg(test)]
    pub(super) fn with_keybag_iterations(root: Zeroizing<[u8; 32]>, iterations: u32) -> Self {
        Self {
            keybags: Some(keybag::Keybags::with_iterations(&root, iterations)),
            ..Self::with_system_keys(root)
        }
    }

    /// Whether a request's passcode derivation is running: the SEP keeps
    /// stepping ([`Self::resume`]) and answers nothing else until it ends.
    pub fn deriving(&self) -> bool {
        self.deferred.is_some()
    }

    /// The frame of the request whose derivation is running.
    pub fn deferred_frame(&self) -> Option<Frame> {
        self.deferred.as_ref().map(|deferred| deferred.frame)
    }

    /// Run one bounded slice of the running derivation. When it completes,
    /// write the reply through `payload` and return the reply frame, or the
    /// refusal; until then, `None`.
    pub fn resume(&mut self, payload: &mut dyn Payload) -> Option<Result<Frame, Refusal>> {
        // A derivation is only ever started by the keybags.
        let keybags = self.keybags.as_mut()?;
        let deferred = self.deferred.as_mut()?;
        let done = keybags.step(&mut deferred.pending)?;
        for line in keybags.drain() {
            self.diagnostics.record(line);
        }
        let deferred = self.deferred.take()?;
        let frame = deferred.frame;
        let refuse = |reason: stela_contracts::Refusal| {
            Refusal::answered(reason, Answer::frame(failure(frame)))
        };
        let result = done
            .map_err(|reason| keybag_refusal(frame, reason))
            .and_then(|fields| {
                let reply = reply_message(&deferred.prefix, &fields);
                payload.output(&reply).map_err(refuse)?;
                Ok(reply_frame(frame, &reply))
            });
        self.note_refusal(frame, result.as_ref().err());
        Some(result)
    }

    /// The state dictionary of a keybag: none is ever locked, as
    /// [`volume_key`] keeps every class available, and no passcode exists.
    /// Handles are not interpreted. Reported once.
    fn device_state(&mut self, variant: u32, fields: &mut Fields<'_>) -> Result<Vec<u8>> {
        let _session = fields.lword()?;
        let handle = fields.word()? as i32;
        match variant {
            0 => {}
            1 => {
                let _options = fields.blob()?;
            }
            _ => return Err(refuse!("SEP keystore device state variant is not modeled")),
        }
        let _flags = fields.word()?;
        fields.end()?;
        let zero = || dictionary::number(0);
        // A loaded user keybag answers with its own lock state.
        let state = match self.keybags.as_ref().and_then(|bags| bags.state(handle)) {
            Some(state) => state as u8,
            None => {
                self.report_unmodeled_state();
                UNLOCKED_WITHOUT_PASSCODE
            }
        };
        Ok(dictionary::dictionary(&[
            (b"sas", zero()),
            (b"sb", zero()),
            (b"sbo", zero()),
            (b"sfa", zero()),
            (b"sfao", zero()),
            (b"sgpe", zero()),
            (b"sgs", zero()),
            (b"sls", zero()),
            (b"sms", zero()),
            (b"srcd", zero()),
            (b"ss", dictionary::number(state)),
        ]))
    }

    fn report_unmodeled_state(&mut self) {
        if !self.state_reported {
            self.state_reported = true;
            self.diagnostics.record(alloc::string::String::from(
                "SEP keystore: every keybag but a loaded user keybag reports unlocked without a passcode",
            ));
        }
    }

    /// Name an unimplemented command in the diagnostics, once per command.
    fn report_unimplemented(&mut self, command: u8) {
        if !self.unimplemented.contains(&command) {
            self.unimplemented.push(command);
            self.diagnostics.record(alloc::format!(
                "SEP keystore: command {command:#04x} is not implemented"
            ));
        }
    }

    /// Serve FileVault volume keys under the machine's system-key `root`,
    /// with a fresh inline wrapping key in `engine` for this sepOS start.
    pub(super) fn with_volume_keys(
        mut self,
        root: &[u8; 32],
        engine: crate::inline_key::Engine,
        entropy: &mut dyn super::Entropy,
    ) -> Result<Self> {
        self.volume_keys = Some(volume_key::VolumeKeys::new(root, engine, entropy)?);
        Ok(self)
    }

    #[cfg(test)]
    pub fn environment(&self) -> Option<Environment> {
        self.environment
    }

    /// Drain bounded, value-free metadata for unsupported system-key requests.
    /// Independent of opt-in raw traffic observation.
    pub(super) fn diagnostics(&mut self) -> impl Iterator<Item = String> + '_ {
        self.diagnostics.drain()
    }

    /// Answer one AP request: read its message, write the reply message
    /// and return the reply frame, or `None` while its passcode derivation
    /// runs ([`Self::resume`]). The first refusal of each command is named
    /// in the diagnostics, unless the command reports its own.
    pub fn begin(
        &mut self,
        frame: Frame,
        payload: &mut dyn Payload,
        entropy: &mut dyn super::Entropy,
    ) -> Result<Option<Frame>, Refusal> {
        let result = if self.deferred.is_some() {
            Err(Refusal::unanswerable(
                "SEP keystore request arrived during a derivation",
            ))
        } else {
            self.serve(frame, payload, entropy)
        };
        self.note_refusal(frame, result.as_ref().err());
        result
    }

    /// [`Self::begin`], then any derivation to its end, for tests: sepOS
    /// runs one bounded slice per DMA step instead.
    #[cfg(test)]
    pub fn request(
        &mut self,
        frame: Frame,
        payload: &mut dyn Payload,
        entropy: &mut dyn super::Entropy,
    ) -> Result<Frame, Refusal> {
        if let Some(reply) = self.begin(frame, payload, entropy)? {
            return Ok(reply);
        }
        loop {
            if let Some(result) = self.resume(payload) {
                return result;
            }
            if !self.deriving() {
                return Err(Refusal::unanswerable(
                    "SEP keystore derivation ended without an answer",
                ));
            }
        }
    }

    fn note_refusal(&mut self, frame: Frame, refusal: Option<&Refusal>) {
        if let Some(refusal) = refusal
            && frame.tag != system_key::PERFORM_OPERATION
            && !self.unimplemented.contains(&frame.tag)
            && !self.refused.contains(&frame.tag)
        {
            self.refused.push(frame.tag);
            self.diagnostics.record(alloc::format!(
                "SEP keystore: command {:#04x}: {}",
                frame.tag,
                refusal.reason
            ));
        }
    }

    fn serve(
        &mut self,
        frame: Frame,
        payload: &mut dyn Payload,
        entropy: &mut dyn super::Entropy,
    ) -> Result<Option<Frame>, Refusal> {
        let refuse = |reason: stela_contracts::Refusal| {
            Refusal::answered(reason, Answer::frame(failure(frame)))
        };
        if frame.tag & REPLY != 0 {
            return Err(Refusal::unanswerable(
                "SEP keystore frame from the AP carries the reply bit",
            ));
        }
        if frame.parameter != 0 || frame.data & CALL_FLAGS != 0 {
            return Err(refuse(refuse!(
                "SEP keystore request status or call flags are not modeled"
            )));
        }
        let command = frame.tag;
        if !matches!(
            command,
            GET_CAPABILITIES
                | SET_ENVIRONMENT
                | NULL_D_KEY
                | GET_DEVICE_STATE
                | CLIENT_TERMINATE
                | system_key::PERFORM_OPERATION
        ) && !keybag::recognizes(command)
            && !volume_key::recognizes(command)
        {
            self.report_unimplemented(command);
            return Err(refuse(refuse!("SEP keystore command is not implemented")));
        }
        // The request may hold a passcode: its bytes are zeroed when dropped.
        let mut bytes = Zeroizing::new(vec![0; (frame.data >> LENGTH_SHIFT) as usize]);
        payload.input(&mut bytes).map_err(refuse)?;
        let message = Message::parse(&bytes).map_err(refuse)?;
        let mut fields = message.fields();
        if command == system_key::PERFORM_OPERATION {
            let operation = match system_key::operation(message.variant(), &mut fields) {
                Ok(operation) => operation,
                Err(reason) => {
                    self.diagnostics.record(system_key::diagnostic(
                        message.variant(),
                        &mut message.fields(),
                    ));
                    return Err(refuse(reason));
                }
            };
            let keys = self.system_keys.as_mut().ok_or_else(|| {
                refuse(refuse!("SEP keystore requires provisioned machine roots"))
            })?;
            let public = match keys.perform(operation, entropy) {
                Ok(bytes) => bytes,
                Err(reason) => {
                    self.diagnostics.record(system_key::diagnostic(
                        message.variant(),
                        &mut message.fields(),
                    ));
                    return Err(refuse(reason));
                }
            };
            let mut reply_fields = (public.len() as u32).to_le_bytes().to_vec();
            reply_fields.extend_from_slice(&public);
            reply_fields.resize(reply_fields.len().next_multiple_of(4), 0);
            let reply = message.reply(&reply_fields);
            payload.output(&reply).map_err(refuse)?;
            return Ok(Some(reply_frame(frame, &reply)));
        }
        if volume_key::recognizes(command) {
            let keys = self
                .volume_keys
                .as_mut()
                .ok_or_else(|| refuse(refuse!("SEP keystore requires provisioned volume keys")))?;
            let diagnostics = &mut self.diagnostics;
            let reply_fields = keys
                .request(
                    command,
                    message.variant(),
                    &mut fields,
                    entropy,
                    &mut |note| diagnostics.record(note),
                )
                .map_err(refuse)?;
            let reply = message.reply(&reply_fields);
            payload.output(&reply).map_err(refuse)?;
            return Ok(Some(reply_frame(frame, &reply)));
        }
        if command == GET_DEVICE_STATE {
            let state = self
                .device_state(message.variant(), &mut fields)
                .map_err(refuse)?;
            let mut reply_fields = (state.len() as u32).to_le_bytes().to_vec();
            reply_fields.extend_from_slice(&state);
            reply_fields.resize(reply_fields.len().next_multiple_of(4), 0);
            let reply = message.reply(&reply_fields);
            payload.output(&reply).map_err(refuse)?;
            return Ok(Some(reply_frame(frame, &reply)));
        }
        if command == CLIENT_TERMINATE {
            if message.variant() != 1 {
                return Err(refuse(refuse!(
                    "SEP keystore client termination variant is not modeled"
                )));
            }
            fields.lword().map_err(refuse)?;
            fields.end().map_err(refuse)?;
            let reply = message.reply(&TERMINATED);
            payload.output(&reply).map_err(refuse)?;
            return Ok(Some(reply_frame(frame, &reply)));
        }
        if keybag::recognizes(command) {
            let Some(keybags) = self.keybags.as_mut() else {
                // The layout is checked first, so a malformed request is
                // named as such.
                keybag::Request::parse(command, message.variant(), &mut fields).map_err(refuse)?;
                return Err(refuse(refuse!(
                    "SEP keystore requires provisioned machine roots"
                )));
            };
            let diagnostics = &mut self.diagnostics;
            let outcome = keybags
                .request(
                    command,
                    message.variant(),
                    &mut fields,
                    entropy,
                    &mut |note| diagnostics.record(note),
                )
                .map_err(|reason| keybag_refusal(frame, reason))?;
            for line in keybags.drain() {
                self.diagnostics.record(line);
            }
            return match outcome {
                keybag::Outcome::Reply(reply_fields) => {
                    let reply = message.reply(&reply_fields);
                    payload.output(&reply).map_err(refuse)?;
                    Ok(Some(reply_frame(frame, &reply)))
                }
                keybag::Outcome::Derive(pending) => {
                    self.deferred = Some(Deferred {
                        frame,
                        prefix: message.prefix().to_vec(),
                        pending,
                    });
                    Ok(None)
                }
            };
        }
        if message.variant() != 0 {
            return Err(refuse(refuse!(
                "SEP keystore startup command variant is not zero"
            )));
        }
        if fields.lword().map_err(refuse)? != STARTUP_ARGUMENT {
            return Err(refuse(refuse!(
                "SEP keystore startup command argument is not the kernel's"
            )));
        }
        if command == NULL_D_KEY {
            fields.end().map_err(refuse)?;
            let reply = message.reply(&[]);
            payload.output(&reply).map_err(refuse)?;
            return Ok(Some(reply_frame(frame, &reply)));
        }
        let blob = fields.blob().map_err(refuse)?;
        fields.end().map_err(refuse)?;
        let (reply, environment) = if command == GET_CAPABILITIES {
            if !blob.is_empty() {
                return Err(refuse(refuse!(
                    "SEP keystore capability query carries a blob"
                )));
            }
            let mut fields = HEADER_VERSION.to_le_bytes().to_vec();
            fields.extend_from_slice(&0u32.to_le_bytes());
            (message.reply(&fields), None)
        } else {
            (
                message.reply(&[]),
                Some(Environment::parse(blob).map_err(refuse)?),
            )
        };
        payload.output(&reply).map_err(refuse)?;
        if environment.is_some() {
            self.environment = environment;
        }
        Ok(Some(reply_frame(frame, &reply)))
    }
}

/// The reply frame to `frame`, whose reply message `reply` is written.
fn reply_frame(frame: Frame, reply: &[u8]) -> Frame {
    Frame {
        tag: frame.tag | REPLY,
        parameter: 0,
        data: (reply.len() as u32) << LENGTH_SHIFT,
        ..frame
    }
}

/// The production passcode iterations, for tests outside this module.
#[cfg(test)]
pub fn keybag_iterations() -> u32 {
    keybag::ITERATIONS
}

/// The iterations per SEP step, for tests outside this module.
#[cfg(test)]
pub fn keybag_slice() -> u32 {
    keybag::SLICE
}

/// The at-rest statement, for tests outside this module.
#[cfg(test)]
pub fn keybag_at_rest() -> &'static str {
    keybag::AT_REST
}

impl Environment {
    fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != ENVIRONMENT_BYTES {
            return Err(refuse!("SEP keystore environment is not 1036 bytes"));
        }
        let word = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        if word(0) != ENVIRONMENT_VERSION
            || !ENVIRONMENT_MODES.contains(&word(8))
            || bytes[12..].iter().any(|&byte| byte != 0)
        {
            return Err(refuse!(
                "SEP keystore environment is outside the kernel's set_env fields"
            ));
        }
        Ok(Self {
            cpx_encryption_mode: word(4),
            mode: word(8),
        })
    }
}

fn u32_at(bytes: &[u8], at: usize) -> Result<u32> {
    stela_contracts::bytes::get_le32(bytes, at)
        .ok_or(refuse!("SEP keystore IPC message ends inside its header"))
}

/// The truncated SHA-256 over everything from the header's version on.
fn digest(message: &[u8]) -> [u8; DIGEST_BYTES] {
    Sha256::digest(&message[VERSION..])[..DIGEST_BYTES]
        .try_into()
        .unwrap()
}

/// A protected IPC message. Variant interpretation belongs to its command.
struct Message<'a> {
    bytes: &'a [u8],
    /// Where the variant starts: after the length and L header bytes.
    body: usize,
}

impl<'a> Message<'a> {
    fn parse(bytes: &'a [u8]) -> Result<Self> {
        let header = u32_at(bytes, 0)? as usize;
        match (u32_at(bytes, VERSION)?, header) {
            (1, HEADER_V1) | (2, HEADER_V2) => {}
            _ => {
                return Err(refuse!(
                    "SEP keystore IPC header is neither 72-byte v1 nor 80-byte v2"
                ));
            }
        }
        let body = LENGTH_BYTES + header;
        if bytes.len() < body + VARIANT_BYTES {
            return Err(refuse!(
                "SEP keystore IPC message ends inside its header or variant"
            ));
        }
        if u32_at(bytes, FLAGS)? != 0 {
            return Err(refuse!(
                "SEP keystore unprotected or extended IPC headers are not modeled"
            ));
        }
        if digest(bytes) != bytes[DIGEST_START..DIGEST_START + DIGEST_BYTES] {
            return Err(refuse!("SEP keystore IPC message digest does not match"));
        }
        Ok(Self { bytes, body })
    }

    fn variant(&self) -> u32 {
        // parse checked that the complete variant word is present.
        u32_at(self.bytes, self.body).unwrap()
    }

    fn fields(&self) -> Fields<'a> {
        Fields {
            bytes: self.bytes,
            at: self.body + VARIANT_BYTES,
        }
    }

    /// The length, header and variant: everything a reply repeats.
    fn prefix(&self) -> &'a [u8] {
        &self.bytes[..self.body + VARIANT_BYTES]
    }

    /// The reply: this request's header and variant, then `fields`, with
    /// the digest recomputed over the reply.
    fn reply(&self, fields: &[u8]) -> Vec<u8> {
        reply_message(self.prefix(), fields)
    }
}

/// A reply message: a request's `prefix` ([`Message::prefix`]), then
/// `fields`, with the digest recomputed over the reply.
fn reply_message(prefix: &[u8], fields: &[u8]) -> Vec<u8> {
    let mut reply = prefix.to_vec();
    reply.extend_from_slice(fields);
    let digest = digest(&reply);
    reply[DIGEST_START..DIGEST_START + DIGEST_BYTES].copy_from_slice(&digest);
    reply
}

/// The command fields after the variant, read in order.
struct Fields<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Fields<'a> {
    fn take(&mut self, count: usize) -> Result<&'a [u8]> {
        let field = self
            .at
            .checked_add(count)
            .and_then(|end| self.bytes.get(self.at..end))
            .ok_or(refuse!("SEP keystore IPC message ends inside a field"))?;
        self.at += count;
        Ok(field)
    }

    fn word(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn lword(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn blob(&mut self) -> Result<&'a [u8]> {
        let length = u32::from_le_bytes(self.take(4)?.try_into().unwrap()) as usize;
        let padded = length
            .checked_next_multiple_of(4)
            .ok_or(refuse!("SEP keystore IPC blob length overflows"))?;
        Ok(&self.take(padded)?[..length])
    }

    fn end(&self) -> Result<()> {
        if self.at == self.bytes.len() {
            Ok(())
        } else {
            Err(refuse!(
                "SEP keystore IPC message has bytes after its fields"
            ))
        }
    }
}

#[cfg(test)]
pub(crate) mod tests;
