//! Signing answers fetched before the install: setup builds, from values it
//! knows, the exact request bodies a restore sends for the recoveryOS root
//! ticket, the Cryptex1 ticket and the recoveryOS LocalPolicy, asks Apple for
//! each and stores the answers; the install then answers an identical body
//! from the stored set without the network.
//!
//! Every value these requests bind is known at setup (26A434
//! `restored_external`, 26B123 J473 install run):
//! - recoveryOS root ticket: the restore identity, the ECID and the AP
//!   ticket's `BNCH`/`snon`, which setup draws ([`super::root_ticket_request`]);
//! - Cryptex1: restored's `_personalize_splat_ticket` takes `Nonce` from the
//!   booted AP ticket's `BNCH` (R 0x1000343f8..0x100034420), `Cryptex1,UDID`
//!   is the chip ID and ECID, and every other value is the installed
//!   identity's ([`cryptex_args`]);
//! - LocalPolicy: `Ap,NextStageIM4MHash` is the SHA-384 of the AP ticket
//!   (R 0x10002b938), `Ap,RecoveryOSPolicyNonceHash` the SHA-384 of the
//!   recoveryOS policy nonce Stela's SEP model proposes, which setup draws
//!   and the install's SEP takes, and `Ap,VolumeUUID` a constant string
//!   (R 0x10002b9d8..0x10002b9f0, [`RECOVERY_VOLUME_UUID`]).
//!
//! A later build that changes any of this sends a different body; it then
//! takes the network like any request ([`Staging`]), or refuses without one.
use super::{
    super::{Error, Result, plist, plist::Value, store::Identity, ticket::Ticket},
    CONTENT_TYPE, HEADERS, HOST, Nonces, PATH, Root, Transport, answer, cryptex_request,
    local_policy_request, root_ticket_request, ticket,
};
use alloc::{boxed::Box, collections::BTreeMap, string::String, vec::Vec};
use sha2::{Digest, Sha384};

/// The boot bundle file that holds the staged answers.
pub const FILE: &str = "tss-answers.plist";
const VERSION: u64 = 1;
const MAX_FILE_BYTES: usize = 256 * 1024;

/// The `Ap,VolumeUUID` restored names in its recoveryOS LocalPolicy request:
/// the system recoveryOS UUID, a constant CFString (26A434 R 0x100253780), not
/// a volume the install creates. The 26B123 install sent the same string.
pub const RECOVERY_VOLUME_UUID: &str = "3D3287DE-280D-4619-AAAB-D97469CA9C71";

/// The `BuildIdentityTags` restored lists in its Cryptex1
/// `FirmwareUpdaterData` request (26B123 J473 install run): the installed
/// identity keys and manifest entries its `DeviceGeneratedRequest` copies.
pub const CRYPTEX_IDENTITY_TAGS: [&str; 33] = [
    "ApSecurityDomain",
    "Cryptex1,ChipID",
    "Cryptex1,Type",
    "Cryptex1,SubType",
    "Cryptex1,ProductClass",
    "Cryptex1,UseProductClass",
    "Cryptex1,NonceDomain",
    "Cryptex1,LiveNonceDomain",
    "Cryptex1,ExclaveNonceDomain",
    "Cryptex1,ExclaveLiveNonceDomain",
    "Cryptex1,Version",
    "Cryptex1,PreauthorizationVersion",
    "Cryptex1,FakeRoot",
    "UniqueBuildID",
    "Cryptex1,DataOnly",
    "Cryptex1,KeyAlgorithm",
    "Cryptex1,SystemOS",
    "Cryptex1,SystemVolume",
    "Cryptex1,SystemTrustCache",
    "Cryptex1,RosettaOS",
    "Cryptex1,RosettaVolume",
    "Cryptex1,RosettaTrustCache",
    "Cryptex1,AppOS",
    "Cryptex1,AppVolume",
    "Cryptex1,AppTrustCache",
    "Cryptex1,MobileAssetBrainOS",
    "Cryptex1,MobileAssetBrainVolume",
    "Cryptex1,MobileAssetBrainTrustCache",
    "Cryptex1,CryptexInfoPlist",
    "Cryptex1,GenericDmg",
    "Cryptex1,GenericTrustCache",
    "Cryptex1,GenericVolume",
    "Cryptex1,GenericIntegrityCatalog",
];

/// The three requests an install sends, as the stored set names them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Kind {
    RecoveryRoot,
    Cryptex,
    RecoveryPolicy,
}

impl Kind {
    pub const ALL: [Kind; 3] = [Kind::RecoveryRoot, Kind::Cryptex, Kind::RecoveryPolicy];

    pub fn name(self) -> &'static str {
        match self {
            Kind::RecoveryRoot => "RecoveryOSRootTicket",
            Kind::Cryptex => "Cryptex1",
            Kind::RecoveryPolicy => "RecoveryOSLocalPolicy",
        }
    }

    fn parse(name: &str) -> Result<Self> {
        Self::ALL
            .into_iter()
            .find(|kind| kind.name() == name)
            .ok_or(Error::Invalid("staged answer kind"))
    }
}

/// restored's `RecoveryOSLocalPolicy` arguments from the AP ticket's and the
/// recoveryOS policy nonce's SHA-384 digests.
pub fn recovery_policy_args_from_digests(next_stage: &[u8; 48], nonce_hash: &[u8; 48]) -> Value {
    Value::dict([
        ("Ap,NextStageIM4MHash", Value::Data(next_stage.to_vec())),
        (
            "Ap,RecoveryOSPolicyNonceHash",
            Value::Data(nonce_hash.to_vec()),
        ),
        ("Ap,VolumeUUID", Value::from(RECOVERY_VOLUME_UUID)),
    ])
}

/// restored's `RecoveryOSLocalPolicy` arguments for an install booted under
/// `ap_ticket` whose SEP proposes `policy_nonce`.
pub fn recovery_policy_args(ap_ticket: &[u8], policy_nonce: &[u8; 32]) -> Value {
    recovery_policy_args_from_digests(
        &Sha384::digest(ap_ticket).into(),
        &Sha384::digest(policy_nonce).into(),
    )
}

/// `Cryptex1,UDID`: four zero bytes, the chip ID and the ECID, big-endian.
fn udid(chip: u64, ecid: u64) -> Result<Vec<u8>> {
    let chip = u32::try_from(chip).map_err(|_| Error::Invalid("chip ID width"))?;
    let mut out = Vec::with_capacity(16);
    out.extend_from_slice(&[0; 4]);
    out.extend_from_slice(&chip.to_be_bytes());
    out.extend_from_slice(&ecid.to_be_bytes());
    Ok(out)
}

/// restored's Cryptex1 `FirmwareUpdaterData` arguments for the machine
/// `ecid` booted under an AP ticket whose `BNCH` is `ap_nonce`, installing
/// `installed` (the `macOS Customer` identity of the restore's board).
pub fn cryptex_args(installed: &Identity, ecid: u64, ap_nonce: &[u8; 32]) -> Result<Value> {
    let chip = installed.number("ApChipID")?;
    let mut generated = BTreeMap::new();
    generated.insert("@Cryptex1,Ticket".into(), Value::Boolean(true));
    for tag in CRYPTEX_IDENTITY_TAGS {
        if let Some(value) = installed.value.get(tag) {
            generated.insert(tag.into(), value.clone());
        } else if let Ok(entry) = installed.component(tag)
            && let Some(digest) = entry.get("Digest")
        {
            generated.insert(tag.into(), Value::dict([("Digest", digest.clone())]));
        }
    }
    generated.insert("Cryptex1,Nonce".into(), Value::Data(ap_nonce.to_vec()));
    generated.insert("Cryptex1,ProductionMode".into(), Value::Boolean(true));
    generated.insert("Cryptex1,UDID".into(), Value::Data(udid(chip, ecid)?));
    Ok(Value::dict([
        ("MessageArgUpdaterName", Value::from("Cryptex1")),
        ("MessageArgType", Value::from("FirmwareResponseData")),
        (
            "MessageArgInfo",
            Value::dict([
                ("ApChipId", Value::Integer(chip)),
                ("ApECID", Value::Signed(ecid as i64)),
                ("ApProductionMode", Value::Boolean(true)),
                ("Nonce", Value::Data(ap_nonce.to_vec())),
            ]),
        ),
        ("DeviceGeneratedRequest", Value::Dict(generated)),
        (
            "DeviceGeneratedTags",
            Value::dict([
                (
                    "BuildIdentityTags",
                    Value::Array(CRYPTEX_IDENTITY_TAGS.map(Value::from).to_vec()),
                ),
                (
                    "ResponseTags",
                    Value::Array(alloc::vec![Value::from("Cryptex1,Ticket")]),
                ),
            ]),
        ),
    ]))
}

/// One request an install will send, as bytes on the wire.
pub struct Request {
    pub kind: Kind,
    pub body: Vec<u8>,
}

/// What setup knows when it stages an install's signing answers.
pub struct Setup<'a> {
    /// The identity the install restores with (its bundle's).
    pub restore: &'a Identity,
    /// The identity it installs (`macOS Customer`).
    pub installed: &'a Identity,
    pub ecid: u64,
    /// Apple's AP ticket the install boots under.
    pub ap_ticket: &'a [u8],
    /// The recoveryOS policy nonce the install's SEP proposes first.
    pub policy_nonce: &'a [u8; 32],
}

impl Setup<'_> {
    /// The AP ticket's nonces, as the install takes them (`Answers::boot_with`).
    fn nonces(&self) -> Result<Nonces> {
        Ticket::booted(self.ap_ticket.to_vec(), self.restore, self.ecid).map(|(_, nonces)| nonces)
    }

    /// Whether `response` is an answer the install accepts for a request of
    /// `kind`, checked as the install checks it.
    fn validate(&self, kind: Kind, response: &[u8], nonces: &Nonces) -> Result<()> {
        match kind {
            Kind::RecoveryRoot => {
                Ticket::personalized(ticket(response)?, self.restore, self.ecid, Some(nonces))
                    .map(drop)
            }
            Kind::Cryptex => match answer(response, "Cryptex1,Ticket")?.is_empty() {
                true => Err(Error::Invalid("Apple's Cryptex1 ticket is empty")),
                false => Ok(()),
            },
            Kind::RecoveryPolicy => {
                Ticket::personalized(ticket(response)?, self.restore, self.ecid, None).map(drop)
            }
        }
    }

    /// The exact bodies of the three requests, built by the functions the
    /// install's restore host uses.
    pub fn requests(&self) -> Result<Vec<Request>> {
        let nonces = self.nonces()?;
        let encode = |value: &Value| plist::encode_xml(value, plist::Limits::default());
        let root = root_ticket_request(self.restore, self.ecid, &nonces, Root::RecoveryOs)?;
        let cryptex = cryptex_request(
            self.restore,
            self.ecid,
            &cryptex_args(self.installed, self.ecid, &nonces.ap)?,
        )?;
        let policy = local_policy_request(
            self.restore,
            self.ecid,
            &recovery_policy_args(self.ap_ticket, self.policy_nonce),
        )?;
        Ok(alloc::vec![
            Request {
                kind: Kind::RecoveryRoot,
                body: encode(&root)?,
            },
            Request {
                kind: Kind::Cryptex,
                body: encode(&cryptex)?,
            },
            Request {
                kind: Kind::RecoveryPolicy,
                body: encode(&policy)?,
            },
        ])
    }

    /// Send each request through `transport` and keep Apple's answers, each
    /// checked as the install checks it, with the policy nonce.
    pub fn stage(&self, transport: &mut dyn Transport) -> Result<Set> {
        let nonces = self.nonces()?;
        let mut set = Set {
            policy_nonce: Some(*self.policy_nonce),
            ..Set::default()
        };
        for request in self.requests()? {
            let response = transport
                .exchange(HOST, PATH, &HEADERS, CONTENT_TYPE, &request.body)?
                .ok_or(Error::Pending)?;
            self.validate(request.kind, &response, &nonces)?;
            set.insert(request.kind, &request.body, response)?;
        }
        Ok(set)
    }
}

/// Stored answers keyed by the SHA-256 of the request body they answer, and
/// the recoveryOS policy nonce the requests were built for.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Set {
    pub policy_nonce: Option<[u8; 32]>,
    answers: BTreeMap<String, (Kind, Vec<u8>)>,
}

impl Set {
    pub fn insert(&mut self, kind: Kind, body: &[u8], response: Vec<u8>) -> Result<()> {
        if self.answers.values().any(|(k, _)| *k == kind) {
            return Err(Error::Invalid("duplicate staged answer kind"));
        }
        self.answers
            .insert(stela_formats::sha256(body), (kind, response));
        Ok(())
    }

    /// Whether every request an install sends has an answer, and the SEP has
    /// the nonce the policy request binds.
    pub fn complete(&self) -> bool {
        self.policy_nonce.is_some()
            && Kind::ALL
                .iter()
                .all(|kind| self.answers.values().any(|(k, _)| k == kind))
    }

    /// Dry run of an install's signing requests before it changes a disk:
    /// the set was staged for this SEP policy nonce, and every body `setup`
    /// builds has a stored answer of its kind that opens as the install
    /// opens it. Returns the first request that fails, and why.
    pub fn check(&self, setup: &Setup<'_>) -> core::result::Result<(), (Kind, Error)> {
        if self.policy_nonce.as_ref() != Some(setup.policy_nonce) {
            return Err((
                Kind::RecoveryPolicy,
                Error::Invalid("the staged recoveryOS policy nonce differs"),
            ));
        }
        let nonces = setup.nonces().map_err(|e| (Kind::RecoveryRoot, e))?;
        for request in setup.requests().map_err(|e| (Kind::RecoveryRoot, e))? {
            let (kind, response) = self.answer(&request.body).ok_or((
                request.kind,
                Error::Invalid("no staged answer for this request"),
            ))?;
            if kind != request.kind {
                return Err((
                    request.kind,
                    Error::Invalid("staged answer of another kind"),
                ));
            }
            setup
                .validate(kind, response, &nonces)
                .map_err(|e| (kind, e))?;
        }
        Ok(())
    }

    /// The stored answer to `body`, and its kind.
    pub fn answer(&self, body: &[u8]) -> Option<(Kind, &[u8])> {
        self.answers
            .get(&stela_formats::sha256(body))
            .map(|(kind, response)| (*kind, response.as_slice()))
    }

    pub fn encode(&self) -> Result<Vec<u8>> {
        let mut fields = BTreeMap::new();
        fields.insert("Version".into(), Value::Integer(VERSION));
        if let Some(nonce) = &self.policy_nonce {
            fields.insert("RecoveryPolicyNonce".into(), Value::Data(nonce.to_vec()));
        }
        let answers = self
            .answers
            .iter()
            .map(|(digest, (kind, response))| {
                Value::dict([
                    ("Kind", Value::from(kind.name())),
                    ("RequestSHA256", Value::from(digest.as_str())),
                    ("Response", Value::Data(response.clone())),
                ])
            })
            .collect();
        fields.insert("Answers".into(), Value::Array(answers));
        plist::encode_xml(&Value::Dict(fields), plist::Limits::default())
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() > MAX_FILE_BYTES {
            return Err(Error::Limit("staged answers file bytes"));
        }
        let value = plist::decode(bytes, plist::Limits::default())?;
        if value.required("Version")?.uint()? != VERSION {
            return Err(Error::Invalid("staged answers version"));
        }
        let policy_nonce = value
            .get("RecoveryPolicyNonce")
            .map(|nonce| {
                <[u8; 32]>::try_from(nonce.data()?)
                    .map_err(|_| Error::Invalid("staged recoveryOS policy nonce is not 32 bytes"))
            })
            .transpose()?;
        let mut set = Set {
            policy_nonce,
            ..Set::default()
        };
        for entry in value.required("Answers")?.array()? {
            let kind = Kind::parse(entry.required("Kind")?.string()?)?;
            let digest = entry.required("RequestSHA256")?.string()?;
            if digest.len() != 64 || !digest.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err(Error::Invalid("staged request digest"));
            }
            if set.answers.values().any(|(k, _)| *k == kind) || set.answers.contains_key(digest) {
                return Err(Error::Invalid("duplicate staged answer"));
            }
            let response = entry.required("Response")?.data()?.to_vec();
            set.answers.insert(digest.into(), (kind, response));
        }
        Ok(set)
    }
}

/// What a boot does with the staged answers on its USB.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Use {
    /// No complete set: Apple's signing server answers over the network.
    Network,
    /// A complete set for an erase install whose image downloads: checked
    /// before the guest starts, with the network open for the image.
    Checked,
    /// A complete set for an erase install from the USB's image: checked
    /// before the guest starts, and no network is opened.
    Offline,
    /// An update: the set holds the erase install's requests, which an
    /// update does not send, so it needs the signing server.
    Update,
}

impl Use {
    pub fn of(set: Option<&Set>, update: bool, remote_image: bool) -> Self {
        match (set.is_some_and(Set::complete), update, remote_image) {
            (false, _, _) => Self::Network,
            (true, true, _) => Self::Update,
            (true, false, true) => Self::Checked,
            (true, false, false) => Self::Offline,
        }
    }

    /// Whether the install opens no network.
    pub fn offline(self) -> bool {
        self == Self::Offline
    }

    /// Whether the set's dry run ([`Set::check`]) runs before the guest.
    pub fn checks(self) -> bool {
        matches!(self, Self::Checked | Self::Offline)
    }
}

/// Hears each stored answer a [`Staging`] transport uses: its kind and the
/// SHA-256 of the request body.
pub type Report = Box<dyn FnMut(Kind, &str)>;

/// A signing transport that answers a request whose body has a stored
/// answer, and passes every other one to `inner` (the network, or the
/// refusal without one). `report` hears each stored answer used.
pub struct Staging {
    set: Set,
    inner: Box<dyn Transport>,
    report: Report,
}

impl Staging {
    pub fn new(set: Set, inner: Box<dyn Transport>, report: Report) -> Self {
        Self { set, inner, report }
    }
}

impl Transport for Staging {
    fn exchange(
        &mut self,
        host: &str,
        path: &str,
        headers: &[(&str, &str)],
        content_type: &str,
        body: &[u8],
    ) -> Result<Option<Vec<u8>>> {
        if (host, path) == (HOST, PATH)
            && let Some((kind, response)) = self.set.answer(body)
        {
            (self.report)(kind, &stela_formats::sha256(body));
            return Ok(Some(response.to_vec()));
        }
        self.inner.exchange(host, path, headers, content_type, body)
    }
}

#[cfg(test)]
mod tests;
