use super::*;
use crate::inline_key;
use alloc::string::String;
use core::sync::atomic::{AtomicU32, Ordering};

const VOLUME: [u8; 16] = [0x11; 16];
const USER: [u8; 16] = [0x22; 16];

/// Distinct bytes on every draw, across requests.
fn entropy() -> impl FnMut(&mut [u8]) -> Result<()> {
    static NEXT: AtomicU32 = AtomicU32::new(0);
    |bytes: &mut [u8]| {
        for byte in bytes {
            let next = NEXT.fetch_add(1, Ordering::Relaxed);
            *byte = (next ^ next >> 8 ^ next >> 16) as u8;
        }
        Ok(())
    }
}

fn keys(root: u8) -> (VolumeKeys, Engine) {
    let engine = Engine::default();
    let keys = VolumeKeys::new(&[root; 32], engine.clone(), &mut entropy()).unwrap();
    (keys, engine)
}

/// A request's fields after the handle and parameters every command sends.
#[derive(Default)]
struct Request(Reply);

impl Request {
    fn new() -> Self {
        let mut request = Self::default();
        request.0.lword(0x55);
        request.0.blob(b"parameters");
        request
    }
    fn word(mut self, value: u32) -> Self {
        self.0.word(value);
        self
    }
    fn lword(mut self, value: u64) -> Self {
        self.0.lword(value);
        self
    }
    fn blob(mut self, bytes: &[u8]) -> Self {
        self.0.blob(bytes);
        self
    }
}

fn send(
    keys: &mut VolumeKeys,
    command: u8,
    variant: u32,
    request: Request,
    notes: &mut Vec<String>,
) -> Result<Vec<u8>> {
    let bytes = request.0.0;
    let mut fields = Fields {
        bytes: &bytes,
        at: 0,
    };
    keys.request(command, variant, &mut fields, &mut entropy(), &mut |note| {
        notes.push(note)
    })
}

fn reply_fields(reply: &[u8]) -> Fields<'_> {
    Fields {
        bytes: reply,
        at: 0,
    }
}

/// NEW_VEK's reply: the wrapped VEK and the inline key area.
fn new_vek(keys: &mut VolumeKeys, secret: &[u8], kek: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let request = Request::new().blob(secret).blob(kek).blob(&VOLUME);
    let reply = send(keys, NEW_VEK, 0, request, &mut Vec::new()).unwrap();
    let mut fields = reply_fields(&reply);
    let wrapped = fields.blob().unwrap().to_vec();
    let area = fields.blob().unwrap().to_vec();
    assert_eq!(fields.word().unwrap(), SEP_WRAPPED);
    fields.end().unwrap();
    (wrapped, area)
}

fn unwrap(
    keys: &mut VolumeKeys,
    variant: u32,
    secret: &[u8],
    kek: &[u8],
    vek: &[u8],
) -> Result<Vec<u8>> {
    let request = Request::new()
        .word(KEK_TYPE)
        .blob(secret)
        .blob(kek)
        .blob(vek)
        .blob(&[]);
    let reply = send(keys, UNWRAP_VEK, variant, request, &mut Vec::new())?;
    let mut fields = reply_fields(&reply);
    let area = fields.blob().unwrap().to_vec();
    assert_eq!(fields.word().unwrap(), SEP_WRAPPED);
    if variant == 1 {
        assert!(fields.blob().unwrap().is_empty());
    }
    fields.end().unwrap();
    Ok(area)
}

fn state(keys: &mut VolumeKeys, vek: &[u8]) -> Vec<u8> {
    let request = Request::new().blob(vek);
    let reply = send(keys, GET_BLOB_STATE, 0, request, &mut Vec::new()).unwrap();
    let mut fields = reply_fields(&reply);
    let state = fields.blob().unwrap().to_vec();
    fields.end().unwrap();
    state
}

/// The cipher an inline key area names, as ANS sees it: a data unit's
/// ciphertext.
fn cipher(engine: &Engine, area: &[u8]) -> Vec<u8> {
    let key = engine.unwrap(area.try_into().unwrap()).unwrap();
    let mut data = alloc::vec![0x5a; inline_key::DATA_UNIT_BYTES];
    key.transform(3, &mut data, true).unwrap();
    data
}

#[test]
fn sizes_name_the_wrapped_blobs_apfs_allocates() {
    let (mut keys, _) = keys(1);
    for (kind, size) in [(WRAPPED_VEK, VEK_BYTES), (WRAPPED_KEK, KEK_BYTES)] {
        let request = Request::new().word(kind);
        let reply = send(&mut keys, GET_SIZE, 0, request, &mut Vec::new()).unwrap();
        assert_eq!(reply, (size as u64).to_le_bytes());
    }
    let request = Request::new().word(3);
    assert!(send(&mut keys, GET_SIZE, 0, request, &mut Vec::new()).is_err());
    let request = Request::new().word(WRAPPED_VEK);
    assert!(send(&mut keys, GET_SIZE, 1, request, &mut Vec::new()).is_err());
}

#[test]
fn a_system_vek_commits_reports_its_state_and_unlocks_to_the_same_key() {
    let (mut keys, engine) = keys(1);
    let (vek, area) = new_vek(&mut keys, &[], &[]);
    assert_eq!(vek.len(), VEK_BYTES);
    assert_eq!(area.len(), inline_key::KEY_AREA_BYTES);

    let mut notes = Vec::new();
    for _ in 0..2 {
        let request = Request::new()
            .word(COMMIT)
            .lword(0)
            .blob(&[])
            .blob(&[])
            .blob(&vek);
        let reply = send(&mut keys, SET_PROTECTION, 0, request, &mut notes).unwrap();
        assert_eq!(reply, 0u32.to_le_bytes());
    }
    // Creating the VEK already reported the uninterpreted fields.
    assert_eq!(notes, [UNRECORDED_COMMIT]);

    assert_eq!(
        state(&mut keys, &vek),
        dictionary(&[
            (b"f", number(SYSTEM_PROTECTED as u8)),
            (b"id", tagged(4, &[0; 16])),
            (b"kid", tagged(4, &VOLUME)),
            (b"kt", number(1)),
            (b"kv", number(1)),
            (b"sr", tagged(4, &[0; 8])),
        ])
    );
    for variant in [0, 1] {
        let again = unwrap(&mut keys, variant, &[], &[], &vek).unwrap();
        assert_eq!(cipher(&engine, &again), cipher(&engine, &area));
    }
    // A system VEK has no user seal for a KEK to open.
    let kek = new_kek(&mut keys, b"secret");
    assert!(unwrap(&mut keys, 0, b"secret", &kek, &vek).is_err());
}

fn new_kek(keys: &mut VolumeKeys, secret: &[u8]) -> Vec<u8> {
    let request = Request::new().word(KEK_TYPE).blob(secret).blob(&USER);
    let reply = send(keys, NEW_KEK, 0, request, &mut Vec::new()).unwrap();
    let mut fields = reply_fields(&reply);
    let kek = fields.blob().unwrap().to_vec();
    fields.end().unwrap();
    assert_eq!(kek.len(), KEK_BYTES);
    kek
}

#[test]
fn a_kek_bound_vek_unlocks_only_with_its_secret() {
    let (mut keys, engine) = keys(1);
    let kek = new_kek(&mut keys, b"secret");
    let (vek, area) = new_vek(&mut keys, b"secret", &kek);
    assert_eq!(
        state(&mut keys, &vek),
        dictionary(&[
            (b"f", number(USER_PROTECTED as u8)),
            (b"id", tagged(4, &USER)),
            (b"kid", tagged(4, &VOLUME)),
            (b"kt", number(1)),
            (b"kv", number(1)),
            (b"sr", tagged(4, &[0; 8])),
        ])
    );
    let again = unwrap(&mut keys, 0, b"secret", &kek, &vek).unwrap();
    assert_eq!(cipher(&engine, &again), cipher(&engine, &area));
    assert!(unwrap(&mut keys, 0, b"wrong", &kek, &vek).is_err());
    assert!(unwrap(&mut keys, 0, &[], &[], &vek).is_err());
    // Creating a VEK needs the KEK's secret too.
    let request = Request::new().blob(b"wrong").blob(&kek).blob(&VOLUME);
    assert!(send(&mut keys, NEW_VEK, 0, request, &mut Vec::new()).is_err());
    // Another KEK does not open it, even with its own secret.
    let other = new_kek(&mut keys, b"other");
    assert!(unwrap(&mut keys, 0, b"other", &other, &vek).is_err());
}

#[test]
fn user_protection_binds_a_verifier_and_keeps_the_system_seal() {
    let (mut keys, engine) = keys(1);
    let (system, area) = new_vek(&mut keys, &[], &[]);
    let request = Request::new()
        .word(VERIFIER_TYPE)
        .blob(b"password")
        .blob(&USER);
    let reply = send(&mut keys, NEW_KEK, 0, request, &mut Vec::new()).unwrap();
    let mut fields = reply_fields(&reply);
    let verifier = fields.blob().unwrap().to_vec();
    let protect = |keys: &mut VolumeKeys, secret: &[u8]| {
        let request = Request::new()
            .word(SET_USER_PROTECTION)
            .lword(0)
            .blob(secret)
            .blob(&verifier)
            .blob(&system);
        send(keys, SET_PROTECTION, 0, request, &mut Vec::new())
    };
    assert!(protect(&mut keys, b"wrong").is_err());
    let reply = protect(&mut keys, b"password").unwrap();
    let mut fields = reply_fields(&reply);
    let both = fields.blob().unwrap().to_vec();
    fields.end().unwrap();
    assert_eq!(both.len(), VEK_BYTES);
    let protected = Vek::parse(&both).unwrap();
    assert_eq!(protected.flags, SYSTEM_SEAL | USER_SEAL);
    assert_eq!(protected.volume(), VOLUME);
    assert_eq!(protected.kek, USER);
    let cipher_of = |area: &[u8]| cipher(&engine, area);
    let by_system = unwrap(&mut keys, 0, &[], &[], &both).unwrap();
    let by_user = unwrap(&mut keys, 0, b"password", &verifier, &both).unwrap();
    assert_eq!(cipher_of(&by_system), cipher_of(&area));
    assert_eq!(cipher_of(&by_user), cipher_of(&area));
    assert!(unwrap(&mut keys, 0, b"wrong", &verifier, &both).is_err());
}

#[test]
fn a_verifier_checks_its_secret_without_a_vek() {
    let (mut keys, _) = keys(1);
    let request = Request::new()
        .word(VERIFIER_TYPE)
        .blob(b"password")
        .blob(&USER);
    let reply = send(&mut keys, NEW_KEK, 0, request, &mut Vec::new()).unwrap();
    let mut fields = reply_fields(&reply);
    let verifier = fields.blob().unwrap().to_vec();
    fields.end().unwrap();
    assert_eq!(verifier.len(), KEK_BYTES);
    let verify = |keys: &mut VolumeKeys, secret: &[u8]| {
        let request = Request::new()
            .word(VERIFIER_TYPE)
            .blob(secret)
            .blob(&verifier)
            .blob(&[])
            .blob(&[]);
        send(keys, UNWRAP_VEK, 0, request, &mut Vec::new())
    };
    let reply = verify(&mut keys, b"password").unwrap();
    let mut fields = reply_fields(&reply);
    assert!(fields.blob().unwrap().is_empty());
    assert_eq!(fields.word().unwrap(), 0);
    fields.end().unwrap();
    assert!(verify(&mut keys, b"wrong").is_err());
}

#[test]
fn another_machine_or_a_changed_blob_does_not_unlock() {
    let (mut keys, _) = keys(1);
    let (vek, _) = new_vek(&mut keys, &[], &[]);
    let (mut other, _) = self::keys(2);
    assert!(unwrap(&mut other, 0, &[], &[], &vek).is_err());
    for index in [0, 5, 9, 30, VEK_HEADER + 1, VEK_BYTES - 1] {
        let mut bad = vek.clone();
        bad[index] ^= 1;
        assert!(
            unwrap(&mut keys, 0, &[], &[], &bad).is_err(),
            "byte {index}"
        );
    }
    assert!(unwrap(&mut keys, 0, &[], &[], &vek[1..]).is_err());
    let kek = new_kek(&mut keys, b"secret");
    let mut bad = kek.clone();
    bad[12] ^= 1;
    assert!(unwrap(&mut keys, 0, b"secret", &bad, &vek).is_err());
}

fn protect(
    keys: &mut VolumeKeys,
    operation: u32,
    blobs: [&[u8]; 3],
    notes: &mut Vec<String>,
) -> Result<Vec<u8>> {
    let request = Request::new()
        .word(operation)
        .lword(0)
        .blob(blobs[0])
        .blob(blobs[1])
        .blob(blobs[2]);
    send(keys, SET_PROTECTION, 0, request, notes)
}

#[test]
fn state_operations_check_their_key_and_report_what_is_not_kept() {
    let (mut keys, _) = keys(1);
    let (vek, _) = new_vek(&mut keys, &[], &[]);
    let kek = new_kek(&mut keys, b"secret");
    let mut notes = Vec::new();
    for (operation, blobs) in [
        (REMOVE_VEK_STATE, [&[][..], &[], &vek]),
        (COMMIT_KEK, [&[], &kek, &[]]),
        (RETAIN_UNLOCK_RECORD, [&[], &kek, &[]]),
        (RELEASE_UNLOCK_RECORD, [&[], &kek, &[]]),
    ] {
        let reply = protect(&mut keys, operation, blobs, &mut notes).unwrap();
        assert_eq!(reply, 0u32.to_le_bytes(), "operation {operation}");
    }
    assert_eq!(notes, [UNRECORDED_COMMIT, UNRETAINED]);
    // A key in the wrong slot, or not a Stela key, refuses.
    for (operation, blobs) in [
        (REMOVE_VEK_STATE, [&[][..], &kek, &[]]),
        (COMMIT_KEK, [&[], &[], &kek]),
        (RETAIN_UNLOCK_RECORD, [&[], &vek, &[]]),
        (RELEASE_UNLOCK_RECORD, [&kek, &kek, &[]]),
    ] {
        assert!(protect(&mut keys, operation, blobs, &mut notes).is_err());
    }
}

#[test]
fn loading_class_keys_unlocks_the_vek_and_reports_the_sealed_system_bit() {
    let (mut keys, _) = keys(1);
    let (system, _) = new_vek(&mut keys, &[], &[]);
    let kek = new_kek(&mut keys, b"secret");
    let (user, _) = new_vek(&mut keys, b"secret", &kek);
    let load = |keys: &mut VolumeKeys, value: u64, blobs: [&[u8]; 3], notes: &mut Vec<String>| {
        let request = Request::new()
            .word(LOAD_CLASS_KEYS)
            .lword(value)
            .blob(blobs[0])
            .blob(blobs[1])
            .blob(blobs[2]);
        send(keys, SET_PROTECTION, 0, request, notes)
    };
    let mut notes = Vec::new();
    for (value, blobs) in [
        (0, [&[][..], &[], &system]),
        (SEALED_SYSTEM, [&[], &[], &system]),
        (0, [b"secret", &kek, &user]),
    ] {
        let reply = load(&mut keys, value, blobs, &mut notes).unwrap();
        assert_eq!(reply, 0u32.to_le_bytes());
    }
    assert_eq!(notes, [UNREAD_SEALED_SYSTEM]);
    // The wrong secret, a user VEK without its KEK, another machine's VEK
    // or another u64 bit refuses.
    for (value, blobs) in [
        (0, [&b"wrong"[..], &kek, &user]),
        (0, [&[], &[], &user]),
        (1, [&[], &[], &system]),
    ] {
        assert!(load(&mut keys, value, blobs, &mut notes).is_err());
    }
    let (mut other, _) = self::keys(2);
    assert!(load(&mut other, 0, [&[], &[], &system], &mut notes).is_err());
}

/// Loads the class keys of `vek`, a system-protected VEK.
fn load_class_keys(keys: &mut VolumeKeys, vek: &[u8]) {
    let reply = protect(keys, LOAD_CLASS_KEYS, [&[], &[], vek], &mut Vec::new()).unwrap();
    assert_eq!(reply, 0u32.to_le_bytes());
}

/// Encoded per-file key parameters naming `volume`, and a persona.
fn parameters(volume: &[u8]) -> Vec<u8> {
    dictionary(&[
        (b"persona_uuid", tagged(4, &USER)),
        (b"volume_uuid", tagged(4, volume)),
    ])
}

/// A per-file key request's fields: the u64 1, a handle, the parameters
/// and the class.
fn file_request(volume: &[u8], class: u32) -> Request {
    Request::default()
        .lword(1)
        .word(u32::MAX - 4)
        .blob(&parameters(volume))
        .word(class)
}

fn send_file_key(
    keys: &mut VolumeKeys,
    command: u8,
    request: Request,
    notes: &mut Vec<String>,
) -> Result<Vec<u8>> {
    let variant = if command == file_key::NEW_KEY { 2 } else { 1 };
    send(keys, command, variant, request, notes)
}

/// NEW_KEY's reply: the inline key area and the wrapped key.
fn new_file_key(keys: &mut VolumeKeys, volume: &[u8], class: u32) -> Result<(Vec<u8>, Vec<u8>)> {
    let request = file_request(volume, class).word(2).blob(&[0; 28]).blob(&[]);
    let reply = send_file_key(keys, file_key::NEW_KEY, request, &mut Vec::new())?;
    let mut fields = reply_fields(&reply);
    let area = fields.blob().unwrap().to_vec();
    assert_eq!(fields.blob().unwrap().len(), 16);
    assert!(fields.blob().unwrap().is_empty());
    let wrapped = fields.blob().unwrap().to_vec();
    assert!(fields.blob().unwrap().is_empty());
    assert_eq!(fields.word().unwrap(), 1 << 1);
    assert_eq!(fields.word().unwrap(), class);
    fields.end().unwrap();
    Ok((area, wrapped))
}

fn unwrap_file_key(
    keys: &mut VolumeKeys,
    volume: &[u8],
    class: u32,
    wrapped: &[u8],
) -> Result<Vec<u8>> {
    let request = file_request(volume, class).blob(wrapped).word(0x102);
    let reply = send_file_key(keys, file_key::UNWRAP_KEY, request, &mut Vec::new())?;
    let mut fields = reply_fields(&reply);
    let area = fields.blob().unwrap().to_vec();
    assert_eq!(fields.blob().unwrap().len(), 16);
    assert_eq!(fields.word().unwrap(), 1 << 1);
    fields.end().unwrap();
    Ok(area)
}

#[test]
fn a_per_file_key_unwraps_to_the_same_inline_key_only_under_its_volume_class_keys() {
    let (mut keys, engine) = keys(1);
    let (vek, _) = new_vek(&mut keys, &[], &[]);
    assert!(new_file_key(&mut keys, &VOLUME, 4).is_err());
    load_class_keys(&mut keys, &vek);
    let (area, wrapped) = new_file_key(&mut keys, &VOLUME, 4).unwrap();
    assert_eq!(wrapped.len(), file_key::WRAPPED_BYTES);
    let again = unwrap_file_key(&mut keys, &VOLUME, 4, &wrapped).unwrap();
    assert_eq!(cipher(&engine, &again), cipher(&engine, &area));
    let (other_area, _) = new_file_key(&mut keys, &VOLUME, 4).unwrap();
    assert_ne!(cipher(&engine, &other_area), cipher(&engine, &area));
    // Another class, volume, machine or a changed key does not unwrap.
    assert!(unwrap_file_key(&mut keys, &VOLUME, 3, &wrapped).is_err());
    assert!(unwrap_file_key(&mut keys, &USER, 4, &wrapped).is_err());
    let mut bad = wrapped.clone();
    bad[30] ^= 1;
    assert!(unwrap_file_key(&mut keys, &VOLUME, 4, &bad).is_err());
    let (mut other, _) = self::keys(2);
    let (other_vek, _) = new_vek(&mut other, &[], &[]);
    load_class_keys(&mut other, &other_vek);
    assert!(unwrap_file_key(&mut other, &VOLUME, 4, &wrapped).is_err());
    // Dropping the VEK's state unloads its class keys until they load again.
    protect(
        &mut keys,
        REMOVE_VEK_STATE,
        [&[], &[], &vek],
        &mut Vec::new(),
    )
    .unwrap();
    assert!(unwrap_file_key(&mut keys, &VOLUME, 4, &wrapped).is_err());
    load_class_keys(&mut keys, &vek);
    let again = unwrap_file_key(&mut keys, &VOLUME, 4, &wrapped).unwrap();
    assert_eq!(cipher(&engine, &again), cipher(&engine, &area));
    // So does the unmount's unload.
    let blobs = [&[][..], &[], &vek];
    protect(&mut keys, UNLOAD_CLASS_KEYS, blobs, &mut Vec::new()).unwrap();
    assert!(unwrap_file_key(&mut keys, &VOLUME, 4, &wrapped).is_err());
}

/// A NEW_KEY for a scalable key: its five blobs.
fn new_scalable(keys: &mut VolumeKeys, class: u32, ephemeral: &[u8]) -> Result<[Vec<u8>; 5]> {
    let request = file_request(&VOLUME, class)
        .word(0x102)
        .blob(&[0; 28])
        .blob(ephemeral);
    let reply = send_file_key(keys, file_key::NEW_KEY, request, &mut Vec::new())?;
    let mut fields = reply_fields(&reply);
    let blobs = core::array::from_fn(|_| fields.blob().unwrap().to_vec());
    assert_eq!(fields.word().unwrap(), 1 << 1);
    assert_eq!(fields.word().unwrap(), class);
    fields.end().unwrap();
    Ok(blobs)
}

/// An UNWRAP_KEY variant 2: its key, IV key and ephemeral key.
fn unwrap_scalable(
    keys: &mut VolumeKeys,
    class: u32,
    [ek, ekwk, ephemeral]: [&[u8]; 3],
    flags: u32,
) -> Result<[Vec<u8>; 3]> {
    let request = file_request(&VOLUME, class)
        .blob(ek)
        .blob(ekwk)
        .blob(ephemeral)
        .word(flags);
    let reply = send(keys, file_key::UNWRAP_KEY, 2, request, &mut Vec::new())?;
    let mut fields = reply_fields(&reply);
    let blobs = core::array::from_fn(|_| fields.blob().unwrap().to_vec());
    assert_eq!(fields.word().unwrap(), 1 << 1);
    fields.end().unwrap();
    Ok(blobs)
}

#[test]
fn a_scalable_key_unwraps_through_its_ekwk_and_its_ephemeral_key() {
    let (mut keys, engine) = keys(1);
    let (vek, _) = new_vek(&mut keys, &[], &[]);
    assert!(new_scalable(&mut keys, 4, &[]).is_err());
    load_class_keys(&mut keys, &vek);
    let [area, iv, ephemeral, ek, ekwk] = new_scalable(&mut keys, 4, &[]).unwrap();
    assert_eq!(iv.len(), 16);
    assert_eq!(ekwk.len(), file_key::WRAPPED_BYTES);
    assert!(ekwk.len() <= 168 && ek.len() <= 168 && ephemeral.len() <= 64);
    assert_eq!(ek.len(), file_key::WRAPPED_EK_BYTES);
    // The wrapped EKWK under its class gives the ephemeral key, and the key
    // too when the wrapped EK comes with it.
    let [no_key, no_iv, opened] = unwrap_scalable(&mut keys, 4, [&[], &ekwk, &[]], 0x103).unwrap();
    assert!(no_key.is_empty() && no_iv.is_empty());
    assert_eq!(opened, ephemeral);
    let [again, same_iv, _] = unwrap_scalable(&mut keys, 4, [&ek, &ekwk, &[]], 0x102).unwrap();
    assert_eq!(cipher(&engine, &again), cipher(&engine, &area));
    assert_eq!(same_iv, iv);
    // So does the ephemeral key, with no class.
    let [again, _, none] = unwrap_scalable(&mut keys, 0, [&ek, &[], &ephemeral], 0x102).unwrap();
    assert_eq!(cipher(&engine, &again), cipher(&engine, &area));
    assert!(none.is_empty());
    // The EKWK wraps more EKs, in the wrapped key's place.
    let [other, _, no_ephemeral, other_ek, none] = new_scalable(&mut keys, 0, &ephemeral).unwrap();
    assert!(none.is_empty() && no_ephemeral.is_empty());
    assert_ne!(cipher(&engine, &other), cipher(&engine, &area));
    let [again, ..] = unwrap_scalable(&mut keys, 0, [&other_ek, &[], &ephemeral], 0x102).unwrap();
    assert_eq!(cipher(&engine, &again), cipher(&engine, &other));
    // Nothing else opens them.
    assert!(unwrap_scalable(&mut keys, 3, [&ek, &ekwk, &[]], 0x102).is_err());
    let [_, _, foreign, ..] = new_scalable(&mut keys, 4, &[]).unwrap();
    assert!(unwrap_scalable(&mut keys, 0, [&ek, &[], &foreign], 0x102).is_err());
    assert!(unwrap_scalable(&mut keys, 0, [&ek, &[], &ek], 0x102).is_err());
    assert!(new_scalable(&mut keys, 4, &ephemeral).is_err());
    assert!(new_scalable(&mut keys, 0, &[]).is_err());
}

/// A keychain command on `blob` under `class`; its reply's fields.
fn keychain(keys: &mut VolumeKeys, command: u8, class: u32, blob: &[u8]) -> Result<Vec<u8>> {
    let mut request = Reply::default();
    request.lword(0x9e40_bd95_86a2_5553);
    request.word(-6i32 as u32);
    request.word(class);
    request.blob(blob);
    request.word(0x5a);
    let mut fields = reply_fields(&request.0);
    keys.request(command, 0, &mut fields, &mut entropy(), &mut |_| {})
}

#[test]
fn a_keychain_key_opens_under_the_class_that_wrapped_it() {
    let (mut keys, _) = keys(1);
    let key = [7; 32];
    let reply = keychain(&mut keys, keychain::KC_WRAP, 11, &key).unwrap();
    let mut fields = reply_fields(&reply);
    let wrapped = fields.blob().unwrap().to_vec();
    assert_eq!(wrapped.len(), 40);
    assert_ne!(&wrapped[8..], key);
    assert_eq!(fields.word().unwrap(), 0x5a);
    fields.end().unwrap();
    let reply = keychain(&mut keys, keychain::KC_UNWRAP, 11, &wrapped).unwrap();
    let mut fields = reply_fields(&reply);
    assert_eq!(fields.blob().unwrap(), key);
    assert!(fields.blob().unwrap().is_empty());
    assert_eq!(fields.word().unwrap(), 0x5a);
    fields.end().unwrap();
    assert!(keychain(&mut keys, keychain::KC_UNWRAP, 7, &wrapped).is_err());
    assert!(keychain(&mut keys, keychain::KC_WRAP, 11, &key[1..]).is_err());
    assert!(keychain(&mut keys, keychain::KC_UNWRAP, 11, &wrapped[1..]).is_err());
}

#[test]
fn a_class_check_succeeds_without_a_key_once_its_volume_class_keys_load() {
    let (mut keys, _) = keys(1);
    let (vek, _) = new_vek(&mut keys, &[], &[]);
    let check = |keys: &mut VolumeKeys| {
        let request = file_request(&VOLUME, 3).word(0).blob(&[]).blob(&[]);
        send_file_key(keys, file_key::NEW_KEY, request, &mut Vec::new())
    };
    assert!(check(&mut keys).is_err());
    load_class_keys(&mut keys, &vek);
    let reply = check(&mut keys).unwrap();
    let mut fields = reply_fields(&reply);
    for _ in 0..5 {
        assert!(fields.blob().unwrap().is_empty());
    }
    assert_eq!(fields.word().unwrap(), 0);
    assert_eq!(fields.word().unwrap(), 3);
    fields.end().unwrap();
    // A class check carries no context or trailing blob.
    let request = file_request(&VOLUME, 3).word(0).blob(&[0; 28]).blob(&[]);
    assert!(send_file_key(&mut keys, file_key::NEW_KEY, request, &mut Vec::new()).is_err());
}

#[test]
fn per_file_keys_report_what_they_do_not_interpret_once_and_refuse_unmodeled_requests() {
    let (mut keys, _) = keys(1);
    let (vek, _) = new_vek(&mut keys, &[], &[]);
    load_class_keys(&mut keys, &vek);
    let mut notes = Vec::new();
    for _ in 0..2 {
        let request = file_request(&VOLUME, 3).word(3).blob(&[0; 28]).blob(&[]);
        send_file_key(&mut keys, file_key::NEW_KEY, request, &mut notes).unwrap();
    }
    assert_eq!(notes, [file_key::UNINTERPRETED]);
    let new_key = |request: Request| request.word(2).blob(&[0; 28]).blob(&[]);
    for request in [
        // Class 0, no volume, another key or a short UUID in the parameters.
        new_key(file_request(&VOLUME, 0)),
        new_key(Request::default().lword(1).word(0).blob(&[]).word(3)),
        new_key(
            Request::default()
                .lword(1)
                .word(0)
                .blob(&dictionary(&[(b"other", tagged(4, &VOLUME))]))
                .word(3),
        ),
        new_key(file_request(&VOLUME[1..], 3)),
        // Another u64, flags without a SEP-wrapped key or with unknown bits,
        // a short context, a trailing blob or trailing bytes.
        new_key(
            Request::default()
                .lword(2)
                .word(0)
                .blob(&parameters(&VOLUME))
                .word(3),
        ),
        file_request(&VOLUME, 3)
            .word(0x100)
            .blob(&[0; 28])
            .blob(&[]),
        file_request(&VOLUME, 3).word(6).blob(&[0; 28]).blob(&[]),
        file_request(&VOLUME, 3).word(2).blob(&[0; 27]).blob(&[]),
        file_request(&VOLUME, 3).word(2).blob(&[0; 28]).blob(b"x"),
        new_key(file_request(&VOLUME, 3)).word(0),
    ] {
        assert!(send_file_key(&mut keys, file_key::NEW_KEY, request, &mut notes).is_err());
    }
    let request = new_key(file_request(&VOLUME, 3));
    assert!(send(&mut keys, file_key::NEW_KEY, 1, request, &mut notes).is_err());
    let (_, wrapped) = new_file_key(&mut keys, &VOLUME, 3).unwrap();
    let request = file_request(&VOLUME, 3).blob(&wrapped).word(0x102);
    assert!(send(&mut keys, file_key::UNWRAP_KEY, 2, request, &mut notes).is_err());
}

#[test]
fn octet_string_dictionaries_read_back_and_malformed_ones_refuse() {
    use super::super::dictionary::octet_strings;
    let long = [7; 200];
    let bytes = dictionary(&[(b"a", tagged(4, &[1, 2])), (b"b", tagged(4, &long))]);
    let entries = octet_strings(&bytes).unwrap();
    assert_eq!(entries, [(&b"a"[..], &[1, 2][..]), (&b"b"[..], &long[..])]);
    assert!(octet_strings(&[]).unwrap().is_empty());
    let malformed: [&[u8]; 5] = [
        &bytes[..bytes.len() - 1],
        &[bytes.as_slice(), &[0]].concat(),
        &dictionary(&[(b"a", number(1))]),
        &[0x30, 0],
        &[0x31, 0x81, 1, 0],
    ];
    for bad in malformed {
        assert!(octet_strings(bad).is_err());
    }
}

fn send_media(keys: &mut VolumeKeys, command: u8, request: Request) -> Result<Vec<u8>> {
    // Media requests start with u64 1, not a handle and parameters.
    let mut fields_request = Request::default().lword(1);
    fields_request.0.0.extend_from_slice(&request.0.0);
    send(keys, command, 1, fields_request, &mut Vec::new())
}

/// NEW_MEDIA_KEY's reply: the inline key area and the wrapped key.
fn new_media_key(keys: &mut VolumeKeys, class: u32) -> (Vec<u8>, Vec<u8>) {
    let request = Request::default().word(u32::MAX).word(class);
    let reply = send_media(keys, media::NEW_MEDIA_KEY, request).unwrap();
    let mut fields = reply_fields(&reply);
    let area = media_key_fields(&mut fields);
    let wrapped = fields.blob().unwrap().to_vec();
    fields.end().unwrap();
    (area, wrapped)
}

fn media_key_fields(fields: &mut Fields<'_>) -> Vec<u8> {
    let area = fields.blob().unwrap().to_vec();
    assert!(fields.blob().unwrap().is_empty());
    assert_eq!(fields.word().unwrap(), 1 << 1);
    area
}

fn unwrap_media_key(
    keys: &mut VolumeKeys,
    wrapped: &[u8],
    handle: u32,
    class: u32,
) -> Result<Vec<u8>> {
    let request = Request::default().blob(wrapped).word(handle).word(class);
    let reply = send_media(keys, media::UNWRAP_MEDIA_KEY, request)?;
    let mut fields = reply_fields(&reply);
    let area = media_key_fields(&mut fields);
    fields.end().unwrap();
    Ok(area)
}

#[test]
fn a_media_key_unwraps_to_the_same_inline_key_only_on_its_machine_and_class() {
    let (mut keys, engine) = keys(1);
    let (area, wrapped) = new_media_key(&mut keys, 0);
    assert_eq!(wrapped.len(), 40);
    let again = unwrap_media_key(&mut keys, &wrapped, u32::MAX, 0).unwrap();
    assert_eq!(cipher(&engine, &again), cipher(&engine, &area));
    assert!(unwrap_media_key(&mut keys, &wrapped, u32::MAX, 1).is_err());
    let (mut other, _) = self::keys(2);
    assert!(unwrap_media_key(&mut other, &wrapped, u32::MAX, 0).is_err());
    let mut bad = wrapped.clone();
    bad[20] ^= 1;
    assert!(unwrap_media_key(&mut keys, &bad, u32::MAX, 0).is_err());
    assert!(unwrap_media_key(&mut keys, &wrapped[1..], u32::MAX, 0).is_err());
    // Another variant or client word refuses.
    let request = Request::default().lword(2).word(u32::MAX).word(0);
    assert!(send(&mut keys, media::NEW_MEDIA_KEY, 1, request, &mut Vec::new()).is_err());
    let request = Request::default().lword(1).word(u32::MAX).word(0);
    assert!(send(&mut keys, media::NEW_MEDIA_KEY, 0, request, &mut Vec::new()).is_err());
}

/// A REWRAP request with no file-system contexts.
fn rewrap(
    keys: &mut VolumeKeys,
    parameters: &[u8],
    classes: (u32, u32),
    wrapped: &[u8],
    flags: u32,
) -> Result<Vec<u8>> {
    rewrap_with(keys, parameters, classes, [&[], &[]], wrapped, flags)
}

/// A REWRAP request with the old and new ephemeral keys `ephemerals`.
fn rewrap_with(
    keys: &mut VolumeKeys,
    parameters: &[u8],
    (old, new): (u32, u32),
    [ephemeral, new_ephemeral]: [&[u8]; 2],
    wrapped: &[u8],
    flags: u32,
) -> Result<Vec<u8>> {
    let request = Request::default()
        .lword(1)
        .blob(&[])
        .blob(parameters)
        .word(u32::MAX - 4)
        .word(u32::MAX - 4)
        .word(old)
        .word(new)
        .blob(ephemeral)
        .blob(new_ephemeral)
        .blob(wrapped)
        .word(flags)
        .blob(&[0; 28])
        .blob(&[]);
    let reply = send(keys, file_key::REWRAP, 3, request, &mut Vec::new())?;
    let mut fields = reply_fields(&reply);
    let rewrapped = fields.blob().unwrap().to_vec();
    assert_eq!(fields.word().unwrap(), new);
    fields.end().unwrap();
    Ok(rewrapped)
}

#[test]
fn a_media_key_migrates_to_a_keybag_class_and_opens_with_its_handle() {
    let (mut keys, engine) = keys(1);
    let (area, wrapped) = new_media_key(&mut keys, 0);
    assert!(rewrap(&mut keys, &[], (1, 14), &wrapped, 0x40).is_err());
    assert!(rewrap(&mut keys, &[], (0, 0), &wrapped, 0x40).is_err());
    let migrated = rewrap(&mut keys, &[], (0, 14), &wrapped, 0x40).unwrap();
    assert_eq!(migrated.len(), wrapped.len());
    let again = unwrap_media_key(&mut keys, &migrated, u32::MAX - 4, 14).unwrap();
    assert_eq!(cipher(&engine, &again), cipher(&engine, &area));
    assert!(unwrap_media_key(&mut keys, &migrated, u32::MAX, 0).is_err());
    assert!(rewrap(&mut keys, &[], (0, 14), &wrapped, 0x80).is_err());
}

#[test]
fn a_per_file_key_rewraps_to_another_class_of_its_volume() {
    let (mut keys, engine) = keys(1);
    let (vek, _) = new_vek(&mut keys, &[], &[]);
    load_class_keys(&mut keys, &vek);
    let (area, wrapped) = new_file_key(&mut keys, &VOLUME, 4).unwrap();
    let parameters = parameters(&VOLUME);
    assert!(rewrap(&mut keys, &parameters, (3, 2), &wrapped, 0).is_err());
    let moved = rewrap(&mut keys, &parameters, (4, 2), &wrapped, 0).unwrap();
    let again = unwrap_file_key(&mut keys, &VOLUME, 2, &moved).unwrap();
    assert_eq!(cipher(&engine, &again), cipher(&engine, &area));
    assert!(unwrap_file_key(&mut keys, &VOLUME, 4, &moved).is_err());
    let moved = rewrap(&mut keys, &parameters, (2, 3), &moved, 1 << 9).unwrap();
    assert!(unwrap_file_key(&mut keys, &VOLUME, 3, &moved).is_ok());
    assert!(rewrap(&mut keys, &[], (3, 2), &moved, 0).is_err());
}

#[test]
fn a_scalable_key_rewraps_from_one_ekwk_to_another_as_a_clone_splits() {
    // fs_rewrap_ek: the old and new ephemeral keys in the two blobs, the
    // wrapped EK and flags 0x100.
    let (mut keys, engine) = keys(1);
    let (vek, _) = new_vek(&mut keys, &[], &[]);
    load_class_keys(&mut keys, &vek);
    let parameters = parameters(&VOLUME);
    let [area, _, ephemeral, ek, _] = new_scalable(&mut keys, 4, &[]).unwrap();
    let [_, _, new_ephemeral, _, _] = new_scalable(&mut keys, 2, &[]).unwrap();
    let ephemerals = [&ephemeral[..], &new_ephemeral];
    let moved = rewrap_with(&mut keys, &parameters, (4, 2), ephemerals, &ek, 0x100).unwrap();
    assert_eq!(moved.len(), file_key::WRAPPED_EK_BYTES);
    let [again, ..] = unwrap_scalable(&mut keys, 0, [&moved, &[], &new_ephemeral], 0x102).unwrap();
    assert_eq!(cipher(&engine, &again), cipher(&engine, &area));
    assert!(unwrap_scalable(&mut keys, 0, [&moved, &[], &ephemeral], 0x102).is_err());
    // Classes are the ephemeral keys' and need not be named.
    let back = [&new_ephemeral[..], &ephemeral];
    let moved = rewrap_with(&mut keys, &parameters, (0, 0), back, &moved, 0x100).unwrap();
    let [again, ..] = unwrap_scalable(&mut keys, 0, [&moved, &[], &ephemeral], 0x102).unwrap();
    assert_eq!(cipher(&engine, &again), cipher(&engine, &area));
    // A missing ephemeral key, keys with other flags, a wrong EKWK or a
    // foreign key refuse.
    let one = [&ephemeral[..], &[]];
    assert!(rewrap_with(&mut keys, &parameters, (4, 2), one, &ek, 0x100).is_err());
    assert!(rewrap(&mut keys, &parameters, (4, 2), &ek, 0x100).is_err());
    assert!(rewrap_with(&mut keys, &parameters, (4, 2), ephemerals, &ek, 0).is_err());
    let swapped = [&new_ephemeral[..], &ephemeral];
    assert!(rewrap_with(&mut keys, &parameters, (2, 4), swapped, &ek, 0x100).is_err());
    let foreign = [&ek[..], &new_ephemeral];
    assert!(rewrap_with(&mut keys, &parameters, (4, 2), foreign, &ek, 0x100).is_err());
}

#[test]
fn unmodeled_operations_and_fields_refuse() {
    let (mut keys, _) = keys(1);
    let (vek, _) = new_vek(&mut keys, &[], &[]);
    let mut notes = Vec::new();
    let request = Request::new()
        .word(4)
        .lword(0)
        .blob(&[])
        .blob(&[])
        .blob(&vek);
    assert!(send(&mut keys, SET_PROTECTION, 0, request, &mut notes).is_err());
    assert!(notes.iter().any(|note| note.contains("operation 4")));
    // ACM authorization, an unknown KEK type and a secret without a KEK.
    let request = Request::new()
        .word(KEK_TYPE)
        .blob(&[])
        .blob(&[])
        .blob(&vek)
        .blob(b"acm");
    assert!(send(&mut keys, UNWRAP_VEK, 0, request, &mut Vec::new()).is_err());
    let request = Request::new().word(1).blob(b"secret").blob(&USER);
    assert!(send(&mut keys, NEW_KEK, 0, request, &mut Vec::new()).is_err());
    let request = Request::new().blob(b"secret").blob(&[]).blob(&VOLUME);
    assert!(send(&mut keys, NEW_VEK, 0, request, &mut Vec::new()).is_err());
    // A short UUID and trailing bytes.
    let request = Request::new().blob(&[]).blob(&[]).blob(&VOLUME[1..]);
    assert!(send(&mut keys, NEW_VEK, 0, request, &mut Vec::new()).is_err());
    let request = Request::new().word(WRAPPED_KEK).word(0);
    assert!(send(&mut keys, GET_SIZE, 0, request, &mut Vec::new()).is_err());
}

#[test]
fn sidp_status_reports_nothing_provisioned_in_the_layout_libaks_decodes() {
    let (mut keys, _) = keys(1);
    let mut notes = Vec::new();
    let reply = protect(&mut keys, SIDP_STATUS, [&[], &[], &[]], &mut notes).unwrap();
    // The reply blob: its length, then SEQUENCE { OCTET STRING (21 zeros) }.
    let mut expected = 25u32.to_le_bytes().to_vec();
    expected.extend_from_slice(&[0x30, 23, 0x04, 21]);
    expected.extend_from_slice(&[0; SIDP_STATUS_BYTES]);
    expected.resize(expected.len().next_multiple_of(4), 0);
    assert_eq!(reply, expected);
    assert_eq!(notes, [UNINTERPRETED, NO_SIDP]);
    // Any input is not the call libaks makes.
    let (vek, _) = new_vek(&mut keys, &[], &[]);
    assert!(protect(&mut keys, SIDP_STATUS, [&[], &[], &vek], &mut notes).is_err());
    let request = Request::new()
        .word(SIDP_STATUS)
        .lword(1)
        .blob(&[])
        .blob(&[])
        .blob(&[]);
    assert!(send(&mut keys, SET_PROTECTION, 0, request, &mut notes).is_err());
}
