extern crate std;

use super::*;
use alloc::{boxed::Box, vec, vec::Vec};

fn blob(bytes: &mut Vec<u8>, value: &[u8]) {
    bytes.extend_from_slice(&(value.len() as u32).to_le_bytes());
    bytes.extend_from_slice(value);
    bytes.resize(bytes.len().next_multiple_of(4), 0);
}

pub(in super::super) fn body(command: u8) -> (u32, Vec<u8>) {
    let mut bytes = 0x1234_5678_9abc_def0u64.to_le_bytes().to_vec();
    let variant = match command {
        CREATE => {
            bytes.extend_from_slice(&7u32.to_le_bytes());
            bytes.extend_from_slice(&(-1i32).to_le_bytes());
            for value in [&b"passcode"[..], &b"acm"[..], &b"x"[..], &b"fourth"[..]] {
                blob(&mut bytes, value);
            }
            bytes.extend_from_slice(&0x100u64.to_le_bytes());
            4
        }
        COPY => {
            bytes.extend_from_slice(&(-11i32).to_le_bytes());
            blob(&mut bytes, b"query");
            1
        }
        LOAD => {
            blob(&mut bytes, b"opaque serialized bag");
            0
        }
        UNLOAD | UUID => {
            bytes.extend_from_slice(&(-11i32).to_le_bytes());
            0
        }
        _ => unreachable!(),
    };
    (variant, bytes)
}

#[test]
fn keybag_variants_have_command_specific_lengths_and_no_accepted_prefixes_or_trailers() {
    for command in [CREATE, COPY, LOAD, UNLOAD, UUID] {
        let (variant, bytes) = body(command);
        let decode = |variant, bytes: &[u8]| {
            Request::parse(command, variant, &mut Fields { bytes, at: 0 }).map(|_| ())
        };
        assert_eq!(decode(variant, &bytes), Ok(()));
        assert!(decode(variant + 1, &bytes).is_err());
        for end in 0..bytes.len() {
            assert!(
                decode(variant, &bytes[..end]).is_err(),
                "command {command} prefix {end}"
            );
        }
        let mut trailing = bytes.clone();
        trailing.extend_from_slice(&[0; 4]);
        assert!(decode(variant, &trailing).is_err());
    }
    // Both length addition and source bounds are checked before slicing.
    for length in [u32::MAX, u32::MAX - 1, 0x7fff_ffff] {
        let mut bytes = vec![0; 8];
        bytes.extend_from_slice(&length.to_le_bytes());
        assert!(
            Request::parse(
                LOAD,
                0,
                &mut Fields {
                    bytes: &bytes,
                    at: 0
                }
            )
            .is_err()
        );
    }
}

#[test]
#[ignore = "oracle: STELA_TEST_AKS_TEXT"]
fn original_libaks_builders_bind_public_selectors_and_uuid_capacity() {
    use sha2::{Digest, Sha256};
    let bytes = std::fs::read(std::env::var("STELA_TEST_AKS_TEXT").unwrap()).unwrap();
    assert_eq!(bytes.len(), 579_648);
    assert_eq!(
        alloc::format!("{:x}", Sha256::digest(&bytes)),
        "3a688ee90028e1dc3a8e735a272821a6c90ea8479d598fa29846cad598bdbc52"
    );
    let movz = |at: usize, register: u32, value: u32| {
        let word = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        assert_eq!(word & 0xffe0_001f, 0x5280_0000 | register);
        assert_eq!((word >> 5) & 0xffff, value);
    };
    movz(0x62148, 1, 2); // create selector
    movz(0x6214c, 3, 7); // create's seven scalar inputs
    movz(0x623f0, 1, 6); // load selector
    movz(0x62594, 1, 3); // save selector
    movz(0x62680, 1, 0x17); // UUID selector
    movz(0x62654, 8, 16); // UUID output capacity
}

#[test]
#[ignore = "oracle: STELA_TEST_AKS_KERNEL"]
fn original_keybag_builders_bind_commands_variants_and_field_helpers() {
    use sha2::{Digest, Sha256};
    let bytes = std::fs::read(std::env::var("STELA_TEST_AKS_KERNEL").unwrap()).unwrap();
    assert_eq!(bytes.len(), 125_026_304);
    assert_eq!(
        alloc::format!("{:x}", Sha256::digest(&bytes)),
        "06846c718a67715f540c7f5a5d5a183da6185d94b6fc01f1ce57a30b601be44d"
    );
    let word = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
    for (at, command) in [
        (0x2d58500, CREATE),
        (0x2d5887c, COPY),
        (0x2d3a45c, LOAD),
        (0x2d3a5bc, UNLOAD),
        (0x2d58a40, UUID),
    ] {
        // MOVZ W1,#command in the AP-to-SEP wrapper, not the public
        // IOConnect selector (which has a different command number).
        assert_eq!(word(at) & 0xffe0_001f, 0x5280_0001);
        assert_eq!((word(at) >> 5) & 0xffff, u32::from(command));
    }
    assert_eq!(word(0x725a40), 4); // create wrapper's variant initializer
    assert_eq!(word(0x725870), 1); // copy wrapper's variant initializer
    let helpers = |start: usize, end: usize| {
        (start..end)
            .step_by(4)
            .filter_map(|at| {
                let instruction = word(at);
                if instruction >> 26 != 0b100101 {
                    return None;
                }
                // BL's signed immediate26, in units of four bytes.
                let displacement = ((instruction << 6) as i32 >> 4) as isize;
                Some(at.wrapping_add_signed(displacement))
            })
            .collect::<Vec<_>>()
    };
    let word_helper = 0x2d5dba4;
    let lword = 0x2d5dd70;
    let blob = 0x2d5dbfc;
    let descriptor = 0x2d5ddc8;
    assert_eq!(
        helpers(0x2d3cf28, 0x2d3d010),
        [
            lword,
            word_helper,
            word_helper,
            blob,
            blob,
            blob,
            descriptor,
            lword
        ]
    );
    assert_eq!(helpers(0x2d3d278, 0x2d3d2b0), [lword, word_helper]);
    assert_eq!(helpers(0x2d3d2c4, 0x2d3d2d4), [blob]);
    assert_eq!(helpers(0x2d3d458, 0x2d3d48c), [lword, blob]);
    assert_eq!(helpers(0x2d3d8f0, 0x2d3d924), [lword, word_helper]);
    assert_eq!(helpers(0x2d3daa4, 0x2d3dad4), [lword, word_helper]);
    // The refusal byte 0xff is sign-extended to -1 by sep_deliver_msg,
    // then the generic IOReturn failure entry is selected by this table.
    assert_eq!(word(0x725990 + 38 * 4), 0xe000_02bc);
}

// ---- Behaviour: requests built as the 26A434 kernel codecs lay them out
// (see the codec ranges in the parent module); no captured passcodes.

use super::super::tests::{Buffers, frame, message};
use super::super::{
    DIGEST_BYTES, DIGEST_START, FAILURE, KeyStore, Message, REPLY, digest, failure,
};
use crate::sep::message::Frame;
use crate::sep::transaction::Refusal;
use alloc::string::String;
use zeroize::Zeroizing;

const ROOT: [u8; 32] = [1; 32];
/// Test bags use few iterations; production uses [`ITERATIONS`].
const TEST_ITERATIONS: u32 = 3 * SLICE + 5;
const SECRET: &[u8] = b"correct horse 7";

/// A request message: header `version`, `variant`, then `body`.
fn request(version: u32, variant: u32, body: &[u8]) -> Vec<u8> {
    let mut request = message(version, 0, 1, &[]);
    let start = Message::parse(&request).unwrap().body;
    request.truncate(start);
    request.extend_from_slice(&variant.to_le_bytes());
    request.extend_from_slice(body);
    let hash = digest(&request);
    request[DIGEST_START..DIGEST_START + DIGEST_BYTES].copy_from_slice(&hash);
    request
}

const SESSION: u64 = 0x0000_0001_0000_01f5;

/// Create, variant 4, as `create_keybag` sends `aks_create_bag`'s call.
fn create_body(kind: u32, parent: u32, passcode: &[u8], acm: &[u8], flags: u64) -> Vec<u8> {
    let mut bytes = SESSION.to_le_bytes().to_vec();
    bytes.extend_from_slice(&kind.to_le_bytes());
    bytes.extend_from_slice(&parent.to_le_bytes());
    for value in [passcode, acm, &[][..], &[][..]] {
        blob(&mut bytes, value);
    }
    bytes.extend_from_slice(&flags.to_le_bytes());
    bytes
}

fn handle_body(handle: i32) -> Vec<u8> {
    let mut bytes = SESSION.to_le_bytes().to_vec();
    bytes.extend_from_slice(&handle.to_le_bytes());
    bytes
}

fn copy_body(handle: i32) -> Vec<u8> {
    let mut bytes = handle_body(handle);
    blob(&mut bytes, &[]);
    bytes
}

fn load_body(saved: &[u8]) -> Vec<u8> {
    let mut bytes = SESSION.to_le_bytes().to_vec();
    blob(&mut bytes, saved);
    bytes
}

fn counter() -> impl FnMut(&mut [u8]) -> crate::Result<()> {
    let mut next = 0u8;
    move |bytes: &mut [u8]| {
        for byte in bytes {
            next = next.wrapping_add(1);
            *byte = next;
        }
        Ok(())
    }
}

type TestEntropy = Box<dyn FnMut(&mut [u8]) -> crate::Result<()>>;

struct Sep {
    keystore: KeyStore,
    entropy: TestEntropy,
}

impl Sep {
    fn new(root: [u8; 32]) -> Self {
        Self {
            keystore: KeyStore::with_keybag_iterations(Zeroizing::new(root), TEST_ITERATIONS),
            entropy: Box::new(counter()),
        }
    }

    /// Send `command` with `body` in header `version`; the reply's fields
    /// after header and variant, or the refusal.
    fn ask(
        &mut self,
        version: u32,
        command: u8,
        variant: u32,
        body: &[u8],
    ) -> Result<Vec<u8>, Refusal> {
        let input = request(version, variant, body);
        let sent = frame(command, 7, input.len());
        let mut buffers = Buffers {
            input: input.clone(),
            output: Vec::new(),
        };
        let reply = self
            .keystore
            .request(sent, &mut buffers, &mut self.entropy)?;
        assert_eq!(reply.tag, command | REPLY);
        assert_eq!((reply.opcode, reply.parameter), (7, 0));
        assert_eq!((reply.data >> 16) as usize, buffers.output.len());
        // The reply repeats the request's header and variant, then its
        // fields, under a fresh digest.
        let start = Message::parse(&buffers.output).unwrap().body + 4;
        assert_eq!(buffers.output[20..start], input[20..start]);
        Ok(buffers.output[start..].to_vec())
    }

    fn create(&mut self, version: u32, passcode: &[u8]) -> i32 {
        let fields = self
            .ask(
                version,
                CREATE,
                4,
                &create_body(0, u32::MAX, passcode, &[], 0),
            )
            .unwrap();
        // The handle, then an empty descriptor blob.
        assert_eq!(fields.len(), 8);
        assert_eq!(fields[4..], [0; 4]);
        i32::from_le_bytes(fields[..4].try_into().unwrap())
    }

    fn save(&mut self, version: u32, handle: i32) -> Vec<u8> {
        let fields = self.ask(version, COPY, 1, &copy_body(handle)).unwrap();
        let length = u32::from_le_bytes(fields[..4].try_into().unwrap()) as usize;
        assert_eq!(length, BLOB_BYTES);
        assert_eq!(fields.len(), 4 + length.next_multiple_of(4));
        fields[4..4 + length].to_vec()
    }

    fn load(&mut self, saved: &[u8]) -> Result<i32, Refusal> {
        let fields = self.ask(2, LOAD, 0, &load_body(saved))?;
        assert_eq!(fields.len(), 4);
        Ok(i32::from_le_bytes(fields.try_into().unwrap()))
    }

    fn uuid(&mut self, handle: i32) -> Result<Vec<u8>, Refusal> {
        let fields = self.ask(1, UUID, 0, &handle_body(handle))?;
        assert_eq!(fields[..4], 16u32.to_le_bytes());
        Ok(fields[4..].to_vec())
    }

    fn notes(&mut self) -> Vec<String> {
        self.keystore.diagnostics().collect()
    }

    fn bags(&self) -> &Keybags {
        self.keystore.keybags.as_ref().unwrap()
    }
}

#[test]
fn pbkdf2_slices_match_the_rfc_6070_style_sha256_vectors() {
    // PBKDF2-HMAC-SHA256, dkLen 32 (the RFC 6070 inputs; published SHA-256
    // results).
    for (iterations, expected) in [
        (
            1,
            "120fb6cffcf8b32c43e7225256c4f837a86548c92ccc35480805987cb70be17b",
        ),
        (
            2,
            "ae4d0c95af6b46d32d0adff928f06dd02a303f8ef3c251dfd6e2d85a95474c43",
        ),
        (
            4096,
            "c5e478d59288c841aa530db6845c4c8d962893a001ce4e11a4963873aa98134a",
        ),
    ] {
        for slice in [1, 7, SLICE] {
            let mut derivation = Derivation::new(b"password", b"salt", iterations);
            let mut steps = 1;
            while !derivation.step(slice) {
                steps += 1;
            }
            assert_eq!(hex::encode(derivation.key()), expected);
            assert_eq!(steps, (iterations - 1).div_ceil(slice).max(1));
        }
    }
}

#[test]
fn a_login_creates_saves_and_reloads_the_bag_in_both_header_versions() {
    for version in [1, 2] {
        let mut sep = Sep::new(ROOT);
        let handle = sep.create(version, SECRET);
        assert_eq!(handle, FIRST_HANDLE);
        let saved = sep.save(version, handle);
        // Saving again gives the same bytes: nothing changed.
        assert_eq!(sep.save(version, handle), saved);
        // The passcode and the seeds never appear in the saved bag.
        assert!(!saved.windows(SECRET.len()).any(|w| w == SECRET));
        let uuid = sep.uuid(handle).unwrap();
        assert_eq!(uuid, saved[16..32]);
        // Unload drops only that handle.
        assert!(
            sep.ask(version, UNLOAD, 0, &handle_body(handle))
                .unwrap()
                .is_empty()
        );
        assert!(sep.uuid(handle).is_err());
        // The next boot: a fresh SEP with the same roots loads it.
        let mut next = Sep::new(ROOT);
        let loaded = next.load(&saved).unwrap();
        assert_eq!(next.uuid(loaded).unwrap(), uuid);
        assert_eq!(next.save(2, loaded), saved);
        let bag = next.bags().find(loaded).unwrap();
        assert!(bag.header.has_passcode());
        assert!(bag.passcode.is_none(), "a loaded bag starts locked");
        assert_eq!(bag.header.iterations, TEST_ITERATIONS);
        // Only the right passcode opens its passcode seed.
        for (secret, opens) in [
            (SECRET, true),
            (&b"correct horse 8"[..], false),
            (&b""[..], false),
        ] {
            let keybags = next.bags();
            let mut derivation = keybags.derivation(&bag.header, secret);
            while !derivation.step(SLICE) {}
            let key = keybags.passcode_key(&bag.header, Some(derivation.key()));
            let opened = open(
                &key,
                PASSCODE_DOMAIN,
                &bag.header.bytes(),
                &bag.passcode_sealed,
            );
            assert_eq!(opened.is_ok(), opens);
        }
        let notes = next.notes();
        assert_eq!(notes, [AT_REST, LOADED]);
        assert!(!notes.iter().any(|n| n.contains("correct horse")));
    }
}

#[test]
fn an_empty_passcode_bag_is_answered_at_once_and_opens_unlocked() {
    let mut sep = Sep::new(ROOT);
    let input = request(2, 4, &create_body(0, u32::MAX, b"", &[], 0));
    let mut buffers = Buffers {
        input,
        output: Vec::new(),
    };
    let sent = frame(CREATE, 3, buffers.input.len());
    let reply = sep
        .keystore
        .begin(sent, &mut buffers, &mut sep.entropy)
        .unwrap();
    assert!(reply.is_some());
    assert!(!sep.keystore.deriving());
    let saved = sep.save(1, FIRST_HANDLE);
    let mut next = Sep::new(ROOT);
    let loaded = next.load(&saved).unwrap();
    let bag = next.bags().find(loaded).unwrap();
    assert!(!bag.header.has_passcode());
    assert!(bag.passcode.is_some());
}

#[test]
fn a_bag_that_does_not_open_is_refused_once_and_never_replaced() {
    let mut sep = Sep::new(ROOT);
    let handle = sep.create(2, SECRET);
    let saved = sep.save(2, handle);
    sep.notes();
    // Another machine's roots, a changed byte anywhere, a short or long
    // blob: each refuses, with no handle and no new bag.
    let mut other = Sep::new([2; 32]);
    assert!(other.load(&saved).is_err());
    let mut cases = Vec::new();
    for at in [0, 8, 12, 20, 33, 40, HEADER_BYTES + 3, BLOB_BYTES - 1] {
        let mut bad = saved.clone();
        bad[at] ^= 1;
        cases.push(bad);
    }
    cases.push(saved[..BLOB_BYTES - 1].to_vec());
    let mut long = saved.clone();
    long.push(0);
    cases.push(long);
    for bad in &cases {
        let refusal = sep.load(bad).unwrap_err();
        assert_eq!(refusal.answer.unwrap().frame.parameter, FAILURE);
    }
    assert_eq!(sep.bags().bags.len(), 1);
    // Reported once, in words, on each SEP.
    let notes = sep.notes();
    assert_eq!(notes.iter().filter(|n| *n == UNOPENABLE).count(), 1);
    assert_eq!(other.notes().iter().filter(|n| *n == UNOPENABLE).count(), 1);
    // The original still loads.
    assert!(sep.load(&saved).is_ok());
}

#[test]
fn create_refuses_every_shape_outside_the_login_call_before_drawing_entropy() {
    for (body, what) in [
        (create_body(7, u32::MAX, SECRET, &[], 0), "type 0x7"),
        (create_body(0, 0, SECRET, &[], 0), "parent 0x0"),
        (create_body(0, u32::MAX, SECRET, &[], 0x100), "flags 0x100"),
        (
            create_body(0, u32::MAX, SECRET, b"acm", 0),
            "ACM blob length 0x3",
        ),
    ] {
        let mut keystore = KeyStore::with_keybag_iterations(Zeroizing::new(ROOT), TEST_ITERATIONS);
        for _ in 0..2 {
            let input = request(2, 4, &body);
            let sent = frame(CREATE, 1, input.len());
            let mut buffers = Buffers {
                input,
                output: Vec::new(),
            };
            let refusal = keystore
                .request(
                    sent,
                    &mut buffers,
                    &mut |_: &mut [u8]| -> crate::Result<()> {
                        panic!("entropy drawn for a refused create")
                    },
                )
                .unwrap_err();
            assert_eq!(refusal.answer.unwrap().frame, failure(sent));
            assert!(buffers.output.is_empty());
        }
        assert!(keystore.keybags.as_ref().unwrap().bags.is_empty());
        let notes: Vec<_> = keystore.diagnostics().collect();
        // The shape once, with its value, then the command's refusal once.
        assert_eq!(notes.len(), 2, "{notes:?}");
        assert!(notes[0].contains(what), "{notes:?}");
        assert!(!notes.iter().any(|n| n.contains("correct horse")));
    }
}

#[test]
fn unknown_handles_and_save_queries_refuse() {
    let mut sep = Sep::new(ROOT);
    for (command, variant, body) in [
        (COPY, 1, copy_body(FIRST_HANDLE)),
        (UNLOAD, 0, handle_body(FIRST_HANDLE)),
        (UUID, 0, handle_body(-501)),
    ] {
        assert!(sep.ask(2, command, variant, &body).is_err());
    }
    let handle = sep.create(2, b"");
    let mut query = handle_body(handle);
    blob(&mut query, b"q");
    assert!(sep.ask(2, COPY, 1, &query).is_err());
}

#[test]
fn a_passcode_derivation_runs_in_bounded_slices_and_holds_the_queue() {
    let mut keystore = KeyStore::with_keybag_iterations(Zeroizing::new(ROOT), ITERATIONS);
    let input = request(2, 4, &create_body(0, u32::MAX, SECRET, &[], 0));
    let sent = frame(CREATE, 9, input.len());
    let mut buffers = Buffers {
        input,
        output: Vec::new(),
    };
    let mut entropy = counter();
    assert_eq!(keystore.begin(sent, &mut buffers, &mut entropy), Ok(None));
    assert!(keystore.deriving());
    assert_eq!(keystore.deferred_frame(), Some(sent));
    // Another request during the derivation is not taken.
    assert!(
        keystore
            .begin(frame(UUID, 10, 0), &mut buffers, &mut entropy)
            .is_err()
    );
    // Each step runs at most SLICE iterations; its time stays far under the
    // few-millisecond bound on one exit (generous in unoptimized builds).
    let bound = if cfg!(debug_assertions) {
        std::time::Duration::from_millis(25)
    } else {
        std::time::Duration::from_millis(3)
    };
    let mut durations = Vec::new();
    let reply = loop {
        let started = std::time::Instant::now();
        let result = keystore.resume(&mut buffers);
        durations.push(started.elapsed());
        match result {
            Some(result) => break result.unwrap(),
            None => assert!(keystore.deriving()),
        }
    };
    assert_eq!(
        u32::try_from(durations.len()).unwrap(),
        (ITERATIONS - 1).div_ceil(SLICE)
    );
    // The median step, since a loaded host can preempt single steps.
    durations.sort_unstable();
    let typical = durations[durations.len() / 2];
    assert!(typical < bound, "median step {typical:?}");
    assert!(!keystore.deriving());
    assert_eq!(
        reply,
        Frame {
            tag: CREATE | REPLY,
            data: reply.data,
            ..sent
        }
    );
    assert_eq!(
        buffers.output[buffers.output.len() - 8..],
        [16, 0, 0, 0, 0, 0, 0, 0]
    );
    assert!(keystore.resume(&mut buffers).is_none());
}

fn bind_body(handle: i32, special: i32, secret: &[u8], flags: u64) -> Vec<u8> {
    let mut bytes = handle_body(handle);
    bytes.extend_from_slice(&special.to_le_bytes());
    blob(&mut bytes, secret);
    bytes.extend_from_slice(&flags.to_le_bytes());
    bytes
}

/// Change lock state v1: session, handle, operation, secret, flags 0 (as
/// every selector but sel 12, and sel 12 for `aks_unlock_bag`, send it).
fn lock_body(handle: i32, operation: u32, secret: &[u8]) -> Vec<u8> {
    lock_body_with(handle, operation, secret, 0)
}

fn lock_body_with(handle: i32, operation: u32, secret: &[u8], flags: u64) -> Vec<u8> {
    let mut bytes = handle_body(handle);
    bytes.extend_from_slice(&operation.to_le_bytes());
    blob(&mut bytes, secret);
    bytes.extend_from_slice(&flags.to_le_bytes());
    bytes
}

fn transition_body(handle: i32, transition: u32, secret: &[u8]) -> Vec<u8> {
    transition_body_with(handle, transition, 0, secret)
}

fn transition_body_with(handle: i32, transition: u32, argument: u64, secret: &[u8]) -> Vec<u8> {
    let mut bytes = handle_body(handle);
    bytes.extend_from_slice(&transition.to_le_bytes());
    bytes.extend_from_slice(&argument.to_le_bytes());
    blob(&mut bytes, secret);
    bytes
}

/// The session handle of uid 501, as sel 5 forms it.
const SESSION_BAG: i32 = -501;

impl Sep {
    fn bind(&mut self, handle: i32) -> Result<(), Refusal> {
        let fields = self.ask(2, MAKE_SYSTEM, 1, &bind_body(handle, SESSION_BAG, &[], 0))?;
        assert!(fields.is_empty());
        Ok(())
    }

    /// Change lock state: (state, flags).
    fn lock_state(&mut self, operation: u32, secret: &[u8]) -> Result<(u32, u64), Refusal> {
        let fields = self.ask(
            2,
            CHANGE_LOCK_STATE,
            1,
            &lock_body(SESSION_BAG, operation, secret),
        )?;
        assert_eq!(fields.len(), 12);
        Ok((
            u32::from_le_bytes(fields[..4].try_into().unwrap()),
            u64::from_le_bytes(fields[4..].try_into().unwrap()),
        ))
    }

    /// GET_DEVICE_STATE variant 0's `ss`.
    fn ss(&mut self, handle: i32) -> u8 {
        let mut body = handle_body(handle);
        body.extend_from_slice(&0u32.to_le_bytes());
        let fields = self
            .ask(2, super::super::GET_DEVICE_STATE, 0, &body)
            .unwrap();
        let length = u32::from_le_bytes(fields[..4].try_into().unwrap()) as usize;
        let dictionary = &fields[4..4 + length];
        assert_eq!(dictionary[length - 7..length - 1], *b"\x0c\x02ss\x02\x01");
        dictionary[length - 1]
    }
}

const LOCKED_WITH_PASSCODE: u32 = 1;
const UNLOCKED_SINCE_LOAD: u32 = BEEN_UNLOCKED;

#[test]
fn a_first_login_creates_saves_binds_and_unlocks_and_the_next_boot_loads_and_unlocks() {
    let mut sep = Sep::new(ROOT);
    // Before the login, the session handle has no bag: today's answer.
    assert_eq!(sep.ss(SESSION_BAG), 0b0110);
    assert!(sep.lock_state(QUERY, &[]).is_err());
    let handle = sep.create(2, SECRET);
    let saved = sep.save(2, handle);
    sep.bind(handle).unwrap();
    // aks_get_system's query: bound, locked until the passcode opens it.
    assert_eq!(sep.lock_state(QUERY, &[]), Ok((LOCKED_WITH_PASSCODE, 0)));
    assert_eq!(sep.ss(SESSION_BAG), 1);
    // A wrong passcode answers status -8 and leaves it locked.
    let refusal = sep.lock_state(UNLOCK, b"wrong horse").unwrap_err();
    assert_eq!(refusal.answer.unwrap().frame.parameter, BAD_PASSCODE_STATUS);
    assert_eq!(sep.lock_state(QUERY, &[]), Ok((LOCKED_WITH_PASSCODE, 0)));
    // The right one unlocks and asks the kernel to notify the change.
    assert_eq!(
        sep.lock_state(UNLOCK, SECRET),
        Ok((UNLOCKED_SINCE_LOAD, NOTIFY_LOCK_STATE))
    );
    assert_eq!(sep.ss(SESSION_BAG), BEEN_UNLOCKED as u8);
    assert_eq!(sep.lock_state(UNLOCK, SECRET), Ok((UNLOCKED_SINCE_LOAD, 0)));
    // applekeystored then unloads its creation handle; the bound bag stays.
    assert!(
        sep.ask(2, UNLOAD, 0, &handle_body(handle))
            .unwrap()
            .is_empty()
    );
    assert_eq!(sep.ss(SESSION_BAG), BEEN_UNLOCKED as u8);
    // Lock, then loginwindow's unlock transition with the passcode.
    assert_eq!(
        sep.lock_state(LOCK, &[]),
        Ok((LOCKED_WITH_PASSCODE | BEEN_UNLOCKED, NOTIFY_LOCK_STATE))
    );
    let fields = sep
        .ask(
            2,
            DEVICE_STATE_TRANSITION,
            0,
            &transition_body(SESSION_BAG, 0, SECRET),
        )
        .unwrap();
    assert_eq!(fields[..8], NOTIFY_LOCK_STATE.to_le_bytes());
    assert_eq!(fields[8..], [0; 8]);
    assert_eq!(sep.ss(SESSION_BAG), BEEN_UNLOCKED as u8);
    let refusal = sep
        .ask(
            2,
            DEVICE_STATE_TRANSITION,
            0,
            &transition_body(SESSION_BAG, 0, b"x"),
        )
        .unwrap_err();
    assert_eq!(refusal.answer.unwrap().frame.parameter, BAD_PASSCODE_STATUS);
    // The next boot: load, bind, unlock.
    let mut next = Sep::new(ROOT);
    let loaded = next.load(&saved).unwrap();
    next.bind(loaded).unwrap();
    assert_eq!(next.lock_state(QUERY, &[]), Ok((LOCKED_WITH_PASSCODE, 0)));
    assert_eq!(
        next.lock_state(UNLOCK, SECRET),
        Ok((UNLOCKED_SINCE_LOAD, NOTIFY_LOCK_STATE))
    );
    // Binding the same bag again changes nothing.
    next.bind(loaded).unwrap();
    assert_eq!(next.ss(SESSION_BAG), BEEN_UNLOCKED as u8);
    // applekeystored's reset (the user's passcode no longer opens the old
    // bag, after a password change the SEP was not told of): a new bag,
    // bound over the old one, locked until the new passcode opens it.
    // (The test entropy repeats per SEP: shift it, for a new UUID.)
    (next.entropy)(&mut [0; 7]).unwrap();
    let other = next.create(2, b"new horse 9");
    assert_ne!(next.uuid(other).unwrap(), next.uuid(loaded).unwrap());
    next.bind(other).unwrap();
    assert_eq!(next.uuid(SESSION_BAG).unwrap(), next.uuid(other).unwrap());
    assert_eq!(next.lock_state(QUERY, &[]), Ok((LOCKED_WITH_PASSCODE, 0)));
    assert!(next.lock_state(UNLOCK, SECRET).is_err());
    assert_eq!(
        next.lock_state(UNLOCK, b"new horse 9"),
        Ok((UNLOCKED_SINCE_LOAD, NOTIFY_LOCK_STATE))
    );
    let notes = next.notes();
    assert!(!notes.iter().any(|n| n.contains("horse")), "{notes:?}");
    for line in [AT_REST, LOADED, BOUND, UNLOCKED, CREATED, REPLACED] {
        assert_eq!(notes.iter().filter(|n| *n == line).count(), 1, "{notes:?}");
    }
}

/// Rollback: with the saved bag deleted (or not opening here), the next
/// boot answers exactly as before keybags existed: the session handle has
/// no bag, reports unlocked without a passcode, and the query is refused.
#[test]
fn without_its_saved_bag_the_next_boot_answers_as_before_keybags() {
    let mut sep = Sep::new(ROOT);
    let handle = sep.create(2, SECRET);
    let saved = sep.save(2, handle);
    sep.bind(handle).unwrap();
    assert_eq!(sep.ss(SESSION_BAG), 1);
    for mut next in [Sep::new(ROOT), Sep::new([2; 32])] {
        // Deleted: nothing is loaded. Another machine's keys: the load is
        // refused and nothing is bound.
        // Either way, until applekeystored binds a bag, the session
        // handle answers as before.
        let _ = next.load(&saved);
        assert_eq!(next.ss(SESSION_BAG), 0b0110);
        assert!(next.lock_state(QUERY, &[]).is_err());
        assert!(next.lock_state(UNLOCK, SECRET).is_err());
    }
    let mut fresh = Sep::new(ROOT);
    assert_eq!(fresh.ss(SESSION_BAG), 0b0110);
    assert!(fresh.lock_state(QUERY, &[]).is_err());
    // The next login simply creates a new bag.
    let handle = fresh.create(2, SECRET);
    fresh.bind(handle).unwrap();
    assert_eq!(
        fresh.lock_state(UNLOCK, SECRET),
        Ok((UNLOCKED_SINCE_LOAD, NOTIFY_LOCK_STATE))
    );
}

#[test]
fn bind_lock_state_and_transition_refuse_shapes_outside_the_login_calls() {
    let mut sep = Sep::new(ROOT);
    let handle = sep.create(2, b"");
    for (command, variant, body) in [
        (MAKE_SYSTEM, 1, bind_body(handle, SESSION_BAG, b"s", 0)),
        (MAKE_SYSTEM, 1, bind_body(handle, SESSION_BAG, &[], 0x100)),
        (MAKE_SYSTEM, 1, bind_body(handle, -1, &[], 0)),
        (MAKE_SYSTEM, 1, bind_body(handle, 0, &[], 0)),
        (MAKE_SYSTEM, 1, bind_body(SESSION_BAG, -502, &[], 0)),
        (MAKE_SYSTEM, 1, bind_body(handle + 1, SESSION_BAG, &[], 0)),
        (CHANGE_LOCK_STATE, 1, lock_body(handle, 2, &[])),
        (CHANGE_LOCK_STATE, 1, lock_body(handle, QUERY, b"s")),
        (CHANGE_LOCK_STATE, 1, lock_body(handle, LOCK, b"s")),
        (
            CHANGE_LOCK_STATE,
            1,
            lock_body_with(handle, QUERY, &[], 0x100),
        ),
        (
            CHANGE_LOCK_STATE,
            1,
            lock_body_with(handle, UNLOCK, &[], 0x800),
        ),
        (DEVICE_STATE_TRANSITION, 0, transition_body(handle, 10, &[])),
        (DEVICE_STATE_TRANSITION, 0, transition_body(handle, 2, &[])),
        (DEVICE_STATE_TRANSITION, 0, transition_body(handle, 1, b"s")),
        (DEVICE_STATE_TRANSITION, 0, transition_body(handle, 5, b"s")),
        (
            DEVICE_STATE_TRANSITION,
            0,
            transition_body_with(handle, 1, 1, &[]),
        ),
    ] {
        assert!(sep.ask(2, command, variant, &body).is_err());
    }
    // Variants the kernel does not send for these commands.
    assert!(
        sep.ask(
            2,
            MAKE_SYSTEM,
            0,
            &bind_body(handle, SESSION_BAG, &[], 0)[..20]
        )
        .is_err()
    );
    assert!(
        sep.ask(2, CHANGE_LOCK_STATE, 0, &lock_body(handle, QUERY, &[]))
            .is_err()
    );
    assert!(sep.bags().find(SESSION_BAG).is_err());
    // A bag without a passcode is unlocked; an empty-passcode unlock
    // answers at once.
    let fields = sep
        .ask(2, CHANGE_LOCK_STATE, 1, &lock_body(handle, UNLOCK, &[]))
        .unwrap();
    assert_eq!(fields[..4], (NO_PIN | BEEN_UNLOCKED).to_le_bytes());
}

fn change_body(handle: i32, old: &[u8], new: &[u8], third: &[u8], words: [u32; 3]) -> Vec<u8> {
    let mut bytes = handle_body(handle);
    for value in [old, new, third, &[][..]] {
        blob(&mut bytes, value);
    }
    for word in words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    bytes
}

/// `aks_change_secret`'s words: byte 0xff, generation option 0, flags 0.
const CHANGE_WORDS: [u32; 3] = [0xff, 0, 0];

impl Sep {
    /// Change secret on `handle`: the reply's generation.
    fn change(&mut self, handle: i32, old: &[u8], new: &[u8]) -> Result<u32, Refusal> {
        let fields = self.ask(
            2,
            CHANGE_SECRET,
            3,
            &change_body(handle, old, new, &[], CHANGE_WORDS),
        )?;
        // The u32, then an empty descriptor blob.
        assert_eq!(fields.len(), 8);
        assert_eq!(fields[4..], [0; 4]);
        Ok(u32::from_le_bytes(fields[..4].try_into().unwrap()))
    }
}

const NEW_SECRET: &[u8] = b"battery staple 4";

#[test]
fn a_password_change_rewraps_the_bag_so_only_the_new_passcode_unlocks_it() {
    let mut sep = Sep::new(ROOT);
    let handle = sep.create(2, SECRET);
    sep.bind(handle).unwrap();
    assert_eq!(
        sep.lock_state(UNLOCK, SECRET).unwrap().0,
        UNLOCKED_SINCE_LOAD
    );
    sep.ask(2, UNLOAD, 0, &handle_body(handle)).unwrap();
    // applekeystored: change on the session's bag, then save and write.
    assert_eq!(sep.change(SESSION_BAG, SECRET, NEW_SECRET), Ok(1));
    // The bound bag stays unlocked: its seed did not change.
    assert_eq!(sep.ss(SESSION_BAG), BEEN_UNLOCKED as u8);
    let saved = sep.save(2, SESSION_BAG);
    assert!(!saved.windows(NEW_SECRET.len()).any(|w| w == NEW_SECRET));
    // Locked again, only the new passcode opens it.
    sep.lock_state(LOCK, &[]).unwrap();
    let refusal = sep.lock_state(UNLOCK, SECRET).unwrap_err();
    assert_eq!(refusal.answer.unwrap().frame.parameter, BAD_PASSCODE_STATUS);
    assert_eq!(
        sep.lock_state(UNLOCK, NEW_SECRET),
        Ok((UNLOCKED_SINCE_LOAD, NOTIFY_LOCK_STATE))
    );
    // The next boot loads the changed bag: the new passcode, generation 1.
    let mut next = Sep::new(ROOT);
    let loaded = next.load(&saved).unwrap();
    assert_eq!(next.bags().find(loaded).unwrap().header.generation, 1);
    next.bind(loaded).unwrap();
    assert!(next.lock_state(UNLOCK, SECRET).is_err());
    assert_eq!(
        next.lock_state(UNLOCK, NEW_SECRET),
        Ok((UNLOCKED_SINCE_LOAD, NOTIFY_LOCK_STATE))
    );
    // Changing again from the new passcode works; from the old one not.
    assert!(next.change(SESSION_BAG, SECRET, b"third").is_err());
    assert_eq!(next.change(SESSION_BAG, NEW_SECRET, b"third"), Ok(2));
    let notes = sep.notes();
    assert!(notes.iter().any(|n| n == CHANGED), "{notes:?}");
    assert!(
        !notes
            .iter()
            .any(|n| n.contains("horse") || n.contains("staple"))
    );
}

#[test]
fn a_failed_change_leaves_the_bag_exactly_as_it_was() {
    let mut sep = Sep::new(ROOT);
    let handle = sep.create(2, SECRET);
    sep.bind(handle).unwrap();
    let before = sep.save(2, SESSION_BAG);
    // A wrong old passcode: status -8, nothing re-sealed.
    let refusal = sep
        .ask(
            2,
            CHANGE_SECRET,
            3,
            &change_body(SESSION_BAG, b"wrong", NEW_SECRET, &[], CHANGE_WORDS),
        )
        .unwrap_err();
    assert_eq!(refusal.answer.unwrap().frame.parameter, BAD_PASSCODE_STATUS);
    assert_eq!(sep.save(2, SESSION_BAG), before);
    assert_eq!(sep.save(2, handle), before);
    // Shapes aks_change_secret never sends refuse before any derivation.
    for (third, words) in [
        (&b"x"[..], CHANGE_WORDS),
        (&[][..], [0, 0, 0]),
        (&[][..], [0xff, 1, 0]),
        (&[][..], [0xff, 0, 8]),
    ] {
        assert!(
            sep.ask(
                2,
                CHANGE_SECRET,
                3,
                &change_body(SESSION_BAG, SECRET, NEW_SECRET, third, words)
            )
            .is_err()
        );
        assert!(!sep.keystore.deriving());
    }
    // Other variants and unknown handles refuse.
    assert!(
        sep.ask(
            2,
            CHANGE_SECRET,
            2,
            &change_body(SESSION_BAG, SECRET, NEW_SECRET, &[], CHANGE_WORDS)
        )
        .is_err()
    );
    assert!(sep.change(-502, SECRET, NEW_SECRET).is_err());
    assert_eq!(sep.save(2, SESSION_BAG), before);
    sep.lock_state(LOCK, &[]).unwrap();
    assert!(sep.lock_state(UNLOCK, SECRET).is_ok());
}

#[test]
fn a_change_to_and_from_an_empty_passcode_keeps_the_bag_open_correctly() {
    let mut sep = Sep::new(ROOT);
    let handle = sep.create(2, b"");
    assert!(sep.change(handle, b"x", NEW_SECRET).is_err());
    assert_eq!(sep.change(handle, b"", NEW_SECRET), Ok(1));
    let saved = sep.save(2, handle);
    let mut next = Sep::new(ROOT);
    let loaded = next.load(&saved).unwrap();
    assert!(next.bags().find(loaded).unwrap().passcode.is_none());
    next.bind(loaded).unwrap();
    assert!(next.lock_state(UNLOCK, b"").is_err());
    next.lock_state(UNLOCK, NEW_SECRET).unwrap();
    assert_eq!(next.change(SESSION_BAG, NEW_SECRET, b""), Ok(2));
    assert_eq!(next.ss(SESSION_BAG), (NO_PIN | BEEN_UNLOCKED) as u8);
    let saved = next.save(2, SESSION_BAG);
    let mut third = Sep::new(ROOT);
    let loaded = third.load(&saved).unwrap();
    assert!(third.bags().find(loaded).unwrap().passcode.is_some());
}

#[test]
fn a_change_runs_both_derivations_in_bounded_slices() {
    let mut keystore = KeyStore::with_keybag_iterations(Zeroizing::new(ROOT), ITERATIONS);
    let mut entropy = counter();
    let mut ask = |keystore: &mut KeyStore, command: u8, variant: u32, body: &[u8]| {
        let input = request(2, variant, body);
        let mut buffers = Buffers {
            input: input.clone(),
            output: Vec::new(),
        };
        keystore
            .request(frame(command, 1, input.len()), &mut buffers, &mut entropy)
            .unwrap();
        buffers.output
    };
    let created = ask(
        &mut keystore,
        CREATE,
        4,
        &create_body(0, u32::MAX, SECRET, &[], 0),
    );
    let handle = i32::from_le_bytes(
        created[created.len() - 8..created.len() - 4]
            .try_into()
            .unwrap(),
    );
    let input = request(
        2,
        3,
        &change_body(handle, SECRET, NEW_SECRET, &[], CHANGE_WORDS),
    );
    let sent = frame(CHANGE_SECRET, 2, input.len());
    let mut buffers = Buffers {
        input,
        output: Vec::new(),
    };
    let mut entropy = counter();
    assert_eq!(keystore.begin(sent, &mut buffers, &mut entropy), Ok(None));
    let bound = if cfg!(debug_assertions) {
        std::time::Duration::from_millis(25)
    } else {
        std::time::Duration::from_millis(3)
    };
    let mut durations = Vec::new();
    let reply = loop {
        let started = std::time::Instant::now();
        let result = keystore.resume(&mut buffers);
        durations.push(started.elapsed());
        if let Some(result) = result {
            break result.unwrap();
        }
        assert!(keystore.deriving());
    };
    // The old KEK, then the new one: each its own run of slices.
    assert_eq!(
        u32::try_from(durations.len()).unwrap(),
        2 * (ITERATIONS - 1).div_ceil(SLICE)
    );
    // The median step, since a loaded host can preempt single steps.
    durations.sort_unstable();
    let typical = durations[durations.len() / 2];
    assert!(typical < bound, "median step {typical:?}");
    assert_eq!(reply.tag, CHANGE_SECRET | REPLY);
    assert_eq!(
        buffers.output[buffers.output.len() - 8..],
        [1, 0, 0, 0, 0, 0, 0, 0]
    );
}

/// Change lock state as the 26A434 and 26B123 codecs lay it out: the
/// secret is followed by the u64 flags word, so the shape without it (the
/// one build-57 parsed, which refused every guest request with "bytes
/// after its fields") and any trailer are refused.
#[test]
fn change_lock_state_carries_the_trailing_flags_word() {
    let full = lock_body(SESSION_BAG, QUERY, &[]);
    let decode = |bytes: &[u8]| {
        Request::parse(CHANGE_LOCK_STATE, 1, &mut Fields { bytes, at: 0 }).map(|_| ())
    };
    assert_eq!(decode(&full), Ok(()));
    for end in 0..full.len() {
        assert!(decode(&full[..end]).is_err(), "prefix {end}");
    }
    let mut trailing = full.clone();
    trailing.extend_from_slice(&[0; 4]);
    assert!(decode(&trailing).is_err());
    let verify = verify_body(SESSION_BAG, SECRET, &[], VERIFY_FLAGS);
    let decode = |bytes: &[u8]| Request::parse(VERIFY, 1, &mut Fields { bytes, at: 0 }).map(|_| ());
    assert_eq!(decode(&verify), Ok(()));
    for end in 0..verify.len() {
        assert!(decode(&verify[..end]).is_err(), "verify prefix {end}");
    }
    let mut fields = Fields {
        bytes: &verify,
        at: 0,
    };
    assert!(Request::parse(VERIFY, 0, &mut fields).is_err());
}

/// The device state transitions the kernel sends besides unlock: sel 8's
/// lock (1) and the lock assertion hold (5) and drop (6). An assertion
/// keeps the bag open across a lock until the last one is dropped.
#[test]
fn lock_assertions_defer_a_lock_until_the_last_drop() {
    let mut sep = Sep::new(ROOT);
    let handle = sep.create(2, SECRET);
    sep.bind(handle).unwrap();
    let transition = |sep: &mut Sep, handle: i32, transition: u32| -> Result<u64, Refusal> {
        let fields = sep.ask(
            2,
            DEVICE_STATE_TRANSITION,
            0,
            &transition_body(handle, transition, &[]),
        )?;
        assert_eq!(fields.len(), 16);
        // No deadline: the kernel times the assertion itself.
        assert_eq!(fields[8..], [0; 8]);
        Ok(u64::from_le_bytes(fields[..8].try_into().unwrap()))
    };
    let session = |sep: &mut Sep, which: u32| transition(sep, SESSION_BAG, which);
    // A locked bag cannot be held open.
    let refusal = session(&mut sep, HOLD_ASSERTION).unwrap_err();
    assert_eq!(refusal.answer.unwrap().frame.parameter, FAILURE);
    assert_eq!(sep.ss(SESSION_BAG), 1);
    sep.lock_state(UNLOCK, SECRET).unwrap();
    assert_eq!(session(&mut sep, HOLD_ASSERTION), Ok(0));
    assert_eq!(session(&mut sep, HOLD_ASSERTION), Ok(0));
    // The lock waits for the assertions; the bag stays open.
    assert_eq!(session(&mut sep, LOCK_TRANSITION), Ok(0));
    assert_eq!(sep.ss(SESSION_BAG), 4);
    assert_eq!(session(&mut sep, DROP_ASSERTION), Ok(0));
    assert_eq!(sep.ss(SESSION_BAG), 4);
    // The last drop applies the lock and asks the kernel to notify it.
    assert_eq!(session(&mut sep, DROP_ASSERTION), Ok(NOTIFY_LOCK_STATE));
    assert_eq!(sep.ss(SESSION_BAG), 5);
    assert!(session(&mut sep, DROP_ASSERTION).is_err());
    // The passcode opens it again; a lock without assertions is at once.
    sep.lock_state(UNLOCK, SECRET).unwrap();
    assert_eq!(session(&mut sep, LOCK_TRANSITION), Ok(NOTIFY_LOCK_STATE));
    assert_eq!(session(&mut sep, LOCK_TRANSITION), Ok(0));
    // An unlock while a lock waits cancels it.
    sep.lock_state(UNLOCK, SECRET).unwrap();
    assert_eq!(session(&mut sep, HOLD_ASSERTION), Ok(0));
    assert_eq!(sep.lock_state(LOCK, &[]), Ok((UNLOCKED_SINCE_LOAD, 0)));
    sep.lock_state(UNLOCK, SECRET).unwrap();
    assert_eq!(session(&mut sep, DROP_ASSERTION), Ok(0));
    assert_eq!(sep.ss(SESSION_BAG), 4);
    // The creation handle is its own instance: it holds no assertion.
    assert!(transition(&mut sep, handle, DROP_ASSERTION).is_err());
    // A bag without a passcode never locks.
    let open = sep.create(2, b"");
    assert_eq!(transition(&mut sep, open, LOCK_TRANSITION), Ok(0));
    assert_eq!(sep.ss(open), (NO_PIN | BEEN_UNLOCKED) as u8);
}

fn verify_body(handle: i32, secret: &[u8], acm: &[u8], flags: u64) -> Vec<u8> {
    let mut bytes = handle_body(handle);
    blob(&mut bytes, secret);
    blob(&mut bytes, acm);
    bytes.extend_from_slice(&flags.to_le_bytes());
    bytes
}

/// Verify (0x21) checks the passcode through the same derivation as an
/// unlock and changes nothing: a locked bag stays locked.
#[test]
fn verify_checks_the_passcode_without_unlocking() {
    let mut sep = Sep::new(ROOT);
    let handle = sep.create(2, SECRET);
    sep.bind(handle).unwrap();
    let verify = |sep: &mut Sep, handle: i32, secret: &[u8], acm: &[u8], flags: u64| {
        sep.ask(2, VERIFY, 1, &verify_body(handle, secret, acm, flags))
    };
    assert_eq!(
        verify(&mut sep, SESSION_BAG, SECRET, &[], VERIFY_FLAGS),
        Ok(vec![0; 8])
    );
    assert_eq!(sep.ss(SESSION_BAG), 1);
    let refusal = verify(&mut sep, SESSION_BAG, b"wrong horse", &[], VERIFY_FLAGS).unwrap_err();
    assert_eq!(refusal.answer.unwrap().frame.parameter, BAD_PASSCODE_STATUS);
    for (acm, flags) in [
        (&[][..], 0),
        (&[][..], VERIFY_FLAGS | 0x80),
        (&[][..], VERIFY_FLAGS | 0x100),
        (&b"acm"[..], VERIFY_FLAGS),
    ] {
        let refusal = verify(&mut sep, SESSION_BAG, SECRET, acm, flags).unwrap_err();
        assert_eq!(refusal.answer.unwrap().frame.parameter, FAILURE);
    }
    // No bag at the handle: refused, as before keybags.
    assert!(verify(&mut sep, -502, SECRET, &[], VERIFY_FLAGS).is_err());
    // A bag without a passcode verifies only the empty secret.
    let open = sep.create(2, b"");
    assert_eq!(
        verify(&mut sep, open, &[], &[], VERIFY_FLAGS),
        Ok(vec![0; 8])
    );
    let refusal = verify(&mut sep, open, b"x", &[], VERIFY_FLAGS).unwrap_err();
    assert_eq!(refusal.answer.unwrap().frame.parameter, BAD_PASSCODE_STATUS);
    let notes = sep.notes();
    assert!(!notes.iter().any(|n| n.contains("horse")), "{notes:?}");
    assert_eq!(
        notes.iter().filter(|n| *n == VERIFIED).count(),
        1,
        "{notes:?}"
    );
}
