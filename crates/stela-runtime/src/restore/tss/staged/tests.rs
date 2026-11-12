use super::*;
use crate::restore::{
    answers::{Answers, Plan},
    tss::{LOCAL_POLICY, Signer, request_root_ticket},
};
use alloc::rc::Rc;
use core::cell::Cell;

/// The 26B123 J473 install run's FirmwareUpdaterData and
/// RecoveryOSLocalPolicy arguments with its BuildManifest identities 20 and
/// 140, trimmed to what the requests read. Its ECID is made up, with as
/// many decimal digits as the real one, so the request sizes are the run's.
const RUN: &[u8] = include_bytes!("../../../../tests/fixtures/tss-install-26B123.plist");
const RUN_ECID: u64 = 0xc118_e934_9665_abb9;

fn run() -> Value {
    plist::decode(RUN, plist::Limits::default()).unwrap()
}

fn identity(run: &Value, key: &str, index: usize) -> Identity {
    Identity {
        index,
        value: run.required(key).unwrap().clone(),
    }
}

fn body(value: &Value) -> Vec<u8> {
    plist::encode_xml(value, plist::Limits::default()).unwrap()
}

fn sent(run: &Value, kind: &str) -> usize {
    run.required("SentBytes")
        .unwrap()
        .required(kind)
        .unwrap()
        .uint()
        .unwrap() as usize
}

#[test]
fn the_recorded_install_requests_rebuild_byte_for_byte_from_setup_values() {
    let run = run();
    let restore = identity(&run, "RestoreIdentity", 20);
    let installed = identity(&run, "InstalledIdentity", 140);
    // Cryptex1: the restore host's request from the guest's arguments, and
    // the one setup builds from the installed identity, the ECID and BNCH.
    let recorded = run.required("CryptexArguments").unwrap();
    let tags: Vec<&str> = recorded
        .required("DeviceGeneratedTags")
        .unwrap()
        .required("BuildIdentityTags")
        .unwrap()
        .array()
        .unwrap()
        .iter()
        .map(|tag| tag.string().unwrap())
        .collect();
    assert_eq!(tags, CRYPTEX_IDENTITY_TAGS);
    let bnch: [u8; 32] = recorded
        .required("MessageArgInfo")
        .unwrap()
        .required("Nonce")
        .unwrap()
        .data()
        .unwrap()
        .try_into()
        .unwrap();
    let guest = body(&cryptex_request(&restore, RUN_ECID, recorded).unwrap());
    let built = cryptex_args(&installed, RUN_ECID, &bnch).unwrap();
    // The guest's own part of the request is the same dictionary.
    assert_eq!(
        built.required("DeviceGeneratedRequest").unwrap(),
        recorded.required("DeviceGeneratedRequest").unwrap()
    );
    let setup = body(&cryptex_request(&restore, RUN_ECID, &built).unwrap());
    assert_eq!(setup, guest);
    assert_eq!(setup.len(), sent(&run, "Cryptex1"));
    // LocalPolicy: the constant volume UUID and the two digests.
    let recorded = run.required("PolicyArguments").unwrap();
    assert_eq!(
        recorded
            .required("Ap,VolumeUUID")
            .unwrap()
            .string()
            .unwrap(),
        RECOVERY_VOLUME_UUID
    );
    let digest = |key| -> [u8; 48] {
        recorded
            .required(key)
            .unwrap()
            .data()
            .unwrap()
            .try_into()
            .unwrap()
    };
    let built = recovery_policy_args_from_digests(
        &digest("Ap,NextStageIM4MHash"),
        &digest("Ap,RecoveryOSPolicyNonceHash"),
    );
    assert_eq!(&built, recorded);
    let guest = body(&local_policy_request(&restore, RUN_ECID, recorded).unwrap());
    let setup = body(&local_policy_request(&restore, RUN_ECID, &built).unwrap());
    assert_eq!(setup, guest);
    assert_eq!(setup.len(), sent(&run, "RecoveryOSLocalPolicy"));
}

/// The same check against the real BuildManifest, AP ticket and serial
/// lines of a recorded run, which stay on this computer:
/// `STELA_TSS_RUN=<dir>` with `BuildManifest.plist`, `ap.im4m`,
/// `ecid` (hex), `cryptex.plist` and `policy.plist` (the recorded
/// arguments) and `sent` (the three POST sizes: root, cryptex, policy).
#[test]
#[ignore = "oracle: STELA_TSS_RUN"]
fn a_recorded_run_rebuilds_from_its_own_files() {
    extern crate std;
    let Some(dir) = std::env::var_os("STELA_TSS_RUN") else {
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let read = |name: &str| std::fs::read(dir.join(name)).unwrap();
    let manifest = read("BuildManifest.plist");
    let limits = plist::Limits {
        bytes: manifest.len(),
        nodes: 4_000_000,
        depth: 64,
    };
    let identities: Vec<Identity> = plist::decode(&manifest, limits)
        .unwrap()
        .required("BuildIdentities")
        .unwrap()
        .array()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(index, value)| Identity {
            index,
            value: value.clone(),
        })
        .collect();
    let restore = &identities[20];
    let installed = crate::restore::answers::installed_identity(&identities, restore).unwrap();
    let ecid = u64::from_str_radix(std::str::from_utf8(&read("ecid")).unwrap().trim(), 16).unwrap();
    let ap_ticket = read("ap.im4m");
    let sizes: Vec<usize> = std::str::from_utf8(&read("sent"))
        .unwrap()
        .split_whitespace()
        .map(|n| n.parse().unwrap())
        .collect();
    let decode = |name| plist::decode(&read(name), plist::Limits::default()).unwrap();
    let cryptex = decode("cryptex.plist");
    let policy = decode("policy.plist");
    let (_, nonces) = Ticket::booted(ap_ticket.clone(), restore, ecid).unwrap();
    let root = body(&root_ticket_request(restore, ecid, &nonces, Root::RecoveryOs).unwrap());
    assert_eq!(root.len(), sizes[0]);
    let guest = body(&cryptex_request(restore, ecid, &cryptex).unwrap());
    let setup = body(
        &cryptex_request(
            restore,
            ecid,
            &cryptex_args(&installed, ecid, &nonces.ap).unwrap(),
        )
        .unwrap(),
    );
    assert!(setup == guest, "the Cryptex1 request differs");
    assert_eq!(setup.len(), sizes[1]);
    let next_stage: [u8; 48] = Sha384::digest(&ap_ticket).into();
    assert_eq!(
        policy
            .required("Ap,NextStageIM4MHash")
            .unwrap()
            .data()
            .unwrap(),
        next_stage
    );
    let nonce_hash: [u8; 48] = policy
        .required("Ap,RecoveryOSPolicyNonceHash")
        .unwrap()
        .data()
        .unwrap()
        .try_into()
        .unwrap();
    let built = recovery_policy_args_from_digests(&next_stage, &nonce_hash);
    let guest = body(&local_policy_request(restore, ecid, &policy).unwrap());
    let setup = body(&local_policy_request(restore, ecid, &built).unwrap());
    assert!(setup == guest, "the LocalPolicy request differs");
    assert_eq!(setup.len(), sizes[2]);
}

const ECID: u64 = 0x77;

fn nonces() -> Nonces {
    Nonces {
        ap: [0xa5; 32],
        sep: [0x5a; 20],
    }
}

/// Apple's signing service as the runtime's other tests fake it: a ticket
/// for each kind of request, personalized for the ECID it names.
struct Apple {
    posts: Rc<Cell<usize>>,
}

impl Transport for Apple {
    fn exchange(
        &mut self,
        _: &str,
        _: &str,
        _: &[(&str, &str)],
        _: &str,
        body: &[u8],
    ) -> Result<Option<Vec<u8>>> {
        use crate::restore::ticket::tests::{bound_ticket, ticket_for};
        self.posts.set(self.posts.get() + 1);
        let request = plist::decode(body, plist::Limits::default())?;
        let ecid = request.required("ApECID")?.bits()?;
        let answer = if request.get("@Cryptex1,Ticket").is_some() {
            Value::dict([("Cryptex1,Ticket", Value::Data(alloc::vec![9, 8, 7]))])
        } else if request.get("Ap,LocalPolicy").is_some() {
            let ticket = ticket_for(b"lpol", &LOCAL_POLICY, Some(ecid));
            Value::dict([("ApImg4Ticket", Value::Data(ticket))])
        } else {
            let ticket = bound_ticket(b"krnl", b"k", Some(ecid), Some(&nonces()));
            Value::dict([("ApImg4Ticket", Value::Data(ticket))])
        };
        let mut response = b"STATUS=0&MESSAGE=SUCCESS&REQUEST_STRING=".to_vec();
        response.extend(plist::encode_xml(&answer, plist::Limits::default())?);
        Ok(Some(response))
    }
}

/// The install's way out with no network: every request refuses.
struct Unplugged(Rc<Cell<usize>>);

impl Transport for Unplugged {
    fn exchange(
        &mut self,
        _: &str,
        _: &str,
        _: &[(&str, &str)],
        _: &str,
        _: &[u8],
    ) -> Result<Option<Vec<u8>>> {
        self.0.set(self.0.get() + 1);
        Err(Error::Invalid("no network"))
    }
}

#[test]
fn staged_answers_serve_the_install_without_a_post() {
    let store = crate::restore::store::tests::fixture(b"x", false);
    let restore = store.identity().clone();
    let ap_ticket =
        crate::restore::ticket::tests::bound_ticket(b"krnl", b"k", Some(ECID), Some(&nonces()));
    let policy_nonce = [0x3c; 32];
    let setup = Setup {
        restore: &restore,
        installed: &restore,
        ecid: ECID,
        ap_ticket: &ap_ticket,
        policy_nonce: &policy_nonce,
    };
    let posts = Rc::new(Cell::new(0));
    let set = setup
        .stage(&mut Apple {
            posts: posts.clone(),
        })
        .unwrap();
    assert_eq!(posts.get(), 3);
    assert!(set.complete());
    let set = Set::decode(&set.encode().unwrap()).unwrap();
    assert!(set.complete());
    assert_eq!(set.policy_nonce, Some(policy_nonce));

    // The install: the restore host builds each request from the guest's
    // arguments and the staged set answers it; nothing leaves the PC.
    let offline = Rc::new(Cell::new(0));
    let used = Rc::new(core::cell::RefCell::new(Vec::new()));
    let log = used.clone();
    let mut answers = Answers::new(store);
    answers.signer = Some(Signer {
        ecid: ECID,
        nonces: Nonces {
            ap: [0; 32],
            sep: [0; 20],
        },
        transport: Box::new(Staging::new(
            set.clone(),
            Box::new(Unplugged(offline.clone())),
            Box::new(move |kind, _| log.borrow_mut().push(kind)),
        )),
    });
    answers.boot_with(ap_ticket.clone()).unwrap();
    let cryptex = Value::dict([
        ("DataType", Value::from("FirmwareUpdaterData")),
        (
            "Arguments",
            cryptex_args(&restore, ECID, &nonces().ap).unwrap(),
        ),
    ]);
    let Plan::Reply(reply) = answers.plan(&cryptex).unwrap() else {
        panic!()
    };
    assert_eq!(
        reply
            .required("FirmwareResponseData")
            .unwrap()
            .required("Cryptex1,Ticket")
            .unwrap()
            .data()
            .unwrap(),
        [9, 8, 7]
    );
    let policy = Value::dict([
        ("DataType", Value::from("RecoveryOSLocalPolicy")),
        ("Arguments", recovery_policy_args(&ap_ticket, &policy_nonce)),
    ]);
    assert!(matches!(answers.plan(&policy), Ok(Plan::Reply(_))));
    let mut signer = answers.signer.take().unwrap();
    request_root_ticket(&mut signer, &restore, Root::RecoveryOs).unwrap();
    assert_eq!(offline.get(), 0);
    assert_eq!(
        *used.borrow(),
        [Kind::Cryptex, Kind::RecoveryPolicy, Kind::RecoveryRoot]
    );
    // A request the set does not hold, as another policy nonce makes, takes
    // the inner transport.
    let other = recovery_policy_args(&ap_ticket, &[0x3d; 32]);
    assert!(crate::restore::tss::request_local_policy(&mut signer, &restore, &other).is_err());
    assert_eq!(offline.get(), 1);
}

#[test]
fn an_incomplete_or_damaged_set_is_not_complete_or_refuses() {
    let mut set = Set::default();
    set.insert(Kind::Cryptex, b"a", b"answer".to_vec()).unwrap();
    assert!(set.insert(Kind::Cryptex, b"b", b"answer".to_vec()).is_err());
    set.insert(Kind::RecoveryRoot, b"c", b"answer".to_vec())
        .unwrap();
    set.insert(Kind::RecoveryPolicy, b"d", b"answer".to_vec())
        .unwrap();
    // Without the SEP's nonce the policy request cannot match.
    assert!(!set.complete());
    set.policy_nonce = Some([1; 32]);
    assert!(set.complete());
    assert_eq!(set.answer(b"a"), Some((Kind::Cryptex, &b"answer"[..])));
    assert_eq!(set.answer(b"e"), None);
    let text = String::from_utf8(set.encode().unwrap()).unwrap();
    let damaged = text.replacen("<string>Cryptex1</string>", "<string>Other</string>", 1);
    assert!(Set::decode(damaged.as_bytes()).is_err());
    let twice = text.replacen(
        "<string>RecoveryOSRootTicket</string>",
        "<string>Cryptex1</string>",
        1,
    );
    assert!(Set::decode(twice.as_bytes()).is_err());
}

#[test]
fn an_update_neither_checks_the_staged_set_nor_goes_offline_with_it() {
    let mut set = Set::default();
    for (kind, body) in [
        (Kind::Cryptex, b"a"),
        (Kind::RecoveryRoot, b"b"),
        (Kind::RecoveryPolicy, b"c"),
    ] {
        set.insert(kind, body, b"answer".to_vec()).unwrap();
    }
    let partial = set.clone();
    set.policy_nonce = Some([1; 32]);
    // An update with a complete erase set: no dry run (it would report a
    // mismatch), and a network for the update's own requests.
    for remote in [false, true] {
        let update = Use::of(Some(&set), true, remote);
        assert_eq!(update, Use::Update);
        assert!(!update.checks() && !update.offline());
    }
    assert_eq!(Use::of(Some(&set), false, false), Use::Offline);
    assert!(Use::Offline.checks() && Use::Offline.offline());
    assert_eq!(Use::of(Some(&set), false, true), Use::Checked);
    assert!(Use::Checked.checks() && !Use::Checked.offline());
    for update in [false, true] {
        assert_eq!(Use::of(Some(&partial), update, false), Use::Network);
        assert_eq!(Use::of(None, update, false), Use::Network);
    }
    assert!(!Use::Network.checks() && !Use::Network.offline());
}

#[test]
fn a_cryptex_udid_is_the_chip_and_ecid() {
    assert_eq!(
        udid(0x8112, 0x0102_0304_0506_0708).unwrap(),
        [0, 0, 0, 0, 0, 0, 0x81, 0x12, 1, 2, 3, 4, 5, 6, 7, 8]
    );
}

#[test]
fn the_dry_run_accepts_the_staged_set_and_refuses_any_drift() {
    use crate::restore::ticket::tests::bound_ticket;
    let store = crate::restore::store::tests::fixture(b"x", false);
    let restore = store.identity().clone();
    let ap_ticket = bound_ticket(b"krnl", b"k", Some(ECID), Some(&nonces()));
    let policy_nonce = [0x3c; 32];
    let setup = Setup {
        restore: &restore,
        installed: &restore,
        ecid: ECID,
        ap_ticket: &ap_ticket,
        policy_nonce: &policy_nonce,
    };
    let posts = Rc::new(Cell::new(0));
    let set = setup.stage(&mut Apple { posts }).unwrap();
    assert_eq!(set.check(&setup), Ok(()));

    // Another policy nonce on the SEP side.
    let other_nonce = [0x3d; 32];
    let drift = Setup {
        policy_nonce: &other_nonce,
        ..setup
    };
    assert_eq!(set.check(&drift).unwrap_err().0, Kind::RecoveryPolicy);
    // Another machine: its requests have no stored answer.
    let other_ticket = bound_ticket(b"krnl", b"k", Some(ECID + 1), Some(&nonces()));
    let drift = Setup {
        ecid: ECID + 1,
        ap_ticket: &other_ticket,
        ..setup
    };
    assert_eq!(set.check(&drift).unwrap_err().0, Kind::RecoveryRoot);
    // Another AP ticket with the same nonces changes the policy digest.
    let other_ticket = bound_ticket(b"krnl", b"j", Some(ECID), Some(&nonces()));
    let drift = Setup {
        ap_ticket: &other_ticket,
        ..setup
    };
    assert_eq!(set.check(&drift).unwrap_err().0, Kind::RecoveryPolicy);
    // A damaged answer in a file that still decodes.
    let text = String::from_utf8(set.encode().unwrap()).unwrap();
    let damaged = Set::decode(text.replacen("<data>", "<data>AAAA", 1).as_bytes()).unwrap();
    assert!(damaged.check(&setup).is_err());
    // A set without the nonce cannot match the policy request.
    let mut bare = set.clone();
    bare.policy_nonce = None;
    assert_eq!(bare.check(&setup).unwrap_err().0, Kind::RecoveryPolicy);
}
