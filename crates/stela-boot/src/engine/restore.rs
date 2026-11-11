//! The restore host: Stela's native restore client driving the guest's
//! `restored` over the USB controller's device port.
//!
//! Two sources feed it. The install medium (`--install`) is the IPSW as
//! pieces on the USB's FAT32 partition. [`Mode::Install`] erases the stick
//! SSD and installs recoveryOS without the macOS system image; macOS is
//! installed from recoveryOS. [`Mode::Update`] (`--update-restore`) runs
//! Apple's stock update. The opt-in, unchecked option study
//! (`--restore-study`) reads an exact IPSW extent on an explicitly selected
//! raw partition with the study's options ([`Mode::Study`]). Neither source
//! is an ANS namespace or a pointer into translated memory. Live USB
//! validation and the complete intact SEP trust route remain research
//! requirements.
use super::{
    bundle::{self, Bytes},
    network,
    remote::Remote,
};
use crate::Result;
use crate::{
    handoff::Handoff,
    output,
    system::{RESTORE_BLOCKS_FILE, RESTORE_INVENTORY_FILE},
};
use alloc::{boxed::Box, collections::VecDeque, format, rc::Rc, string::String, vec::Vec};
use core::{cell::RefCell, fmt::Write, task::Poll};
use stela_contracts::Refusal;
use stela_contracts::refuse;
use stela_drivers::block::{Ahead, BlockDevice, ReadAhead};
use {
    stela_formats::payload::PayloadManifest, stela_formats::payload::Purpose,
    stela_formats::payload::ReadAt, stela_runtime::net::Transfer,
    stela_runtime::restore::answers::Answers, stela_runtime::restore::driver::Driver,
    stela_runtime::restore::driver::Ports, stela_runtime::restore::proxy::Control,
    stela_runtime::restore::proxy::Destination, stela_runtime::restore::proxy::Event as ProxyEvent,
    stela_runtime::restore::proxy::Peer, stela_runtime::restore::proxy::Version,
    stela_runtime::restore::route::Route, stela_runtime::restore::source::Extent,
    stela_runtime::restore::store::Store, stela_runtime::restore::store::Verify,
    stela_runtime::restore::stream::ByteStream, stela_soc::machine::AppleDevices,
    stela_soc::machine::MuxBackpressure, stela_soc::machine::MuxStream,
    stela_soc::machine::MuxStreamStats, stela_soc::machine::UsbClient, stela_soc::machine::UsbId,
};

/// The bundle's boot manifest (`stela_loader::boot_bundle`).
const BOOT_MANIFEST_FILE: &str = "boot-manifest.im4m";

/// Which restore the host runs, and how it reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// The user's install: the inventory's Erase identity, with the macOS
    /// system image when the payload holds it ([`Route::Stock`]), else
    /// without ([`Route::Recovery`]).
    Install,
    /// The original restore's explicit development-board policy: keep and
    /// report FDR failures, but allow restored to continue past them.
    DevelopmentInstall,
    /// The same, with the inventory's Update identity: restored_update
    /// installs over the installed system and keeps its data.
    Update,
    /// The option study, with its explicit identity.
    Study,
}

impl Mode {
    fn label(self) -> &'static str {
        match self {
            Self::Install => "INSTALL",
            Self::DevelopmentInstall => "DEVELOPMENT INSTALL",
            Self::Update => "UPDATE",
            Self::Study => "RESTORE STUDY",
        }
    }
}

/// Any IPSW source, its errors described; a remote image's reader also
/// reads ahead between the restore's requests ([`Source::pump`]).
pub struct Source {
    reader: Box<dyn ReadAt<Error = Refusal>>,
    remote: Option<Rc<RefCell<Remote>>>,
    /// The partition's read-ahead ([`Source::extent`]).
    ahead: Option<Ahead>,
}

struct Described<R>(R);

impl<R: ReadAt> ReadAt for Described<R> {
    type Error = Refusal;
    fn len(&self) -> u64 {
        self.0.len()
    }
    fn read_at(&mut self, offset: u64, buffer: &mut [u8]) -> Result<()> {
        self.0
            .read_at(offset, buffer)
            .map_err(|error| refuse!("payload read at {offset:#x}: {error}", offset, ?error))
    }
}

/// A shared remote reader, borrowed for each read.
struct Borrowed(Rc<RefCell<Remote>>);

impl ReadAt for Borrowed {
    type Error = stela_runtime::net::NetError;
    fn len(&self) -> u64 {
        self.0.borrow().len()
    }
    fn read_at(&mut self, offset: u64, buffer: &mut [u8]) -> Result<(), Self::Error> {
        self.0.borrow_mut().read_at(offset, buffer)
    }
}

impl Source {
    pub fn new<R: ReadAt + 'static>(reader: R) -> Self {
        Self {
            reader: Box::new(Described(reader)),
            remote: None,
            ahead: None,
        }
    }

    pub fn remote(remote: Remote) -> Self {
        let remote = Rc::new(RefCell::new(remote));
        Self {
            reader: Box::new(Described(Borrowed(Rc::clone(&remote)))),
            remote: Some(remote),
            ahead: None,
        }
    }

    /// Reads the image ahead without waiting.
    pub fn pump(&mut self) {
        if let Some(remote) = &self.remote {
            remote.borrow_mut().pump();
        }
        if let Some(ahead) = &self.ahead {
            ahead.pump();
        }
    }

    /// The study's raw source: the IPSW from the partition's first byte,
    /// read ahead in the host storage's largest commands.
    pub fn extent(partition: Box<dyn BlockDevice>, bytes: u64) -> Result<Self> {
        let ahead = Ahead::new(ReadAhead::new(
            partition,
            READ_AHEAD_BYTES,
            READ_AHEAD_DEPTH,
        ));
        let extent = Extent::new(Bytes::new(ahead.clone()), bytes)
            .map_err(|error| refuse!("restore source extent: {error}", %error))?;
        let mut source = Self::new(extent);
        source.ahead = Some(ahead);
        Ok(source)
    }
}

impl ReadAt for Source {
    type Error = Refusal;
    fn len(&self) -> u64 {
        self.reader.len()
    }
    fn read_at(&mut self, offset: u64, buffer: &mut [u8]) -> Result<()> {
        self.reader.read_at(offset, buffer)
    }
}

/// The restore inventory the loader read.
pub fn inventory(handoff: &Handoff) -> Result<PayloadManifest> {
    let file = handoff.files[..handoff.file_count]
        .iter()
        .find(|f| f.name() == RESTORE_INVENTORY_FILE)
        .ok_or(refuse!("restore inventory was not loaded"))?;
    PayloadManifest::parse(bundle::contents(file))
        .map_err(|error| refuse!("restore inventory: {error}", %error))
}

/// A retained file the loader read, excluding bytes now owned by the guest.
pub fn inventory_file<'a>(handoff: &'a Handoff, name: &str) -> Option<&'a [u8]> {
    handoff.files[..handoff.file_count]
        .iter()
        .find(|f| f.name() == name)
        .filter(|file| {
            let start = file.contents.physical;
            let end = start.saturating_add(file.contents.bytes);
            !handoff.bundle_layout.backing().iter().any(|part| {
                let physical = part.region.physical;
                start < physical.saturating_add(part.region.bytes) && physical < end
            })
        })
        .map(bundle::contents)
}

/// The install's signing answers setup fetched from Apple
/// (`tss::staged::FILE`), when the bundle holds them. A damaged file is
/// reported and the install takes the network as without one.
pub fn staged_answers(handoff: &Handoff) -> Option<stela_runtime::restore::tss::staged::Set> {
    let bytes = inventory_file(handoff, stela_runtime::restore::tss::staged::FILE)?;
    stela_runtime::restore::tss::staged::Set::decode(bytes)
        .inspect_err(|error| {
            output::arguments(format_args!("TSS STAGED refusal: {error}\n"));
        })
        .ok()
}

pub struct Experiment {
    usb: UsbId,
    driver: Driver<Source, MuxStream>,
    proxy: Option<ReverseProxy>,
    terminal: bool,
    /// The restore failed: why, and the counter then. The board stops
    /// [`FAILURE_GRACE_MS`] later.
    failure: Option<(String, u64)>,
    ledger: usize,
    mode: Mode,
    /// The mux's backpressure as the last poll found it.
    backpressure: MuxBackpressure,
    asr_stream: Option<MuxStreamStats>,
}

/// How long the guest runs on after its restore fails. A failure is often
/// the guest panicking, which takes the USB link down first; its report
/// reaches the console after that.
const FAILURE_GRACE_MS: u64 = 10_000;

/// The install medium's read-ahead: requests of 1 MiB, the most a USB
/// disk's command moves, four kept ahead.
const READ_AHEAD_BYTES: usize = 1 << 20;
const READ_AHEAD_DEPTH: usize = 4;

/// Chunks of the image each service reads ahead, in one read the size of
/// the store's largest, which the host's storage moves in its largest
/// commands.
const PREFETCH_CHUNKS: usize = 8;

/// Native services kept for the whole restore session.
pub struct Services {
    pub signer: Option<stela_runtime::restore::tss::Signer>,
    pub fdr_storage: Option<Box<dyn BlockDevice>>,
    pub network: Option<network::Shared>,
}

/// libimobiledevice uses one 1 MiB buffer per direction. Keep the same finite
/// bound while keeping bytes under either guest or host TCP backpressure.
const PROXY_BUFFER: usize = network::RELAY_BUFFER;
const PROXY_READ: usize = 64 << 10;
const PROXY_CONNECTIONS: usize = 4;
// libimobiledevice's reference client gives each control receive 20 seconds.
// A translated restore guest can need materially longer than a host-native
// device to parse and answer the v2 plist, so use that same protocol bound.
const PROXY_NEGOTIATION_MILLIS: u64 = 20_000;

struct MuxIo<'a, 'b> {
    ports: &'a mut UsbClient<'b>,
    handle: MuxStream,
}
impl ByteStream for MuxIo<'_, '_> {
    fn read(&mut self, bytes: &mut [u8]) -> Poll<stela_runtime::restore::Result<usize>> {
        Ports::read(self.ports, self.handle, bytes)
    }
    fn write(&mut self, bytes: &[u8]) -> Poll<stela_runtime::restore::Result<usize>> {
        Ports::write(self.ports, self.handle, bytes)
    }
}

struct ProxyConnection {
    guest_port: u16,
    version: Version,
    guest: Option<MuxStream>,
    peer: Peer,
    peer_state: &'static str,
    destination: Option<Destination>,
    remote: Option<Transfer>,
    to_remote: VecDeque<u8>,
    to_guest: VecDeque<u8>,
    guest_closed: bool,
    remote_closed: bool,
    closing: bool,
    out_bytes: u64,
    in_bytes: u64,
}

impl ProxyConnection {
    fn new(guest_port: u16, version: Version) -> Self {
        let peer = Peer::new(version);
        let peer_state = peer.state();
        Self {
            guest_port,
            version,
            guest: None,
            peer,
            peer_state,
            destination: None,
            remote: None,
            to_remote: VecDeque::with_capacity(PROXY_BUFFER),
            to_guest: VecDeque::with_capacity(PROXY_BUFFER),
            guest_closed: false,
            remote_closed: false,
            closing: false,
            out_bytes: 0,
            in_bytes: 0,
        }
    }

    fn refuse(&mut self, reason: impl core::fmt::Display) {
        let destination = self
            .destination
            .as_ref()
            .map(|d| format!("{}:{}", d.host, d.port))
            .unwrap_or_else(|| String::from("unparsed destination"));
        output::arguments(format_args!(
            "RESTORE PROXY connection {destination} refused: {reason}\n"
        ));
        self.closing = true;
    }
}

impl ProxyConnection {
    /// One bounded operation in each direction. `true` means both owners were
    /// retired and the scheduler may remove this connection.
    fn poll(&mut self, ports: &mut UsbClient<'_>, network: &network::Shared) -> bool {
        if self.guest.is_none() && !self.closing {
            match Ports::open(ports, self.guest_port) {
                Poll::Pending => return false,
                Poll::Ready(Ok(handle)) => self.guest = Some(handle),
                Poll::Ready(Err(error)) => self.refuse(error),
            }
        }
        if self.destination.is_none()
            && !self.closing
            && let Some(handle) = self.guest
        {
            let mut stream = MuxIo { ports, handle };
            let result = self.peer.poll(&mut stream);
            let state = self.peer.state();
            if state != self.peer_state {
                output::arguments(format_args!(
                    "RESTORE PROXY peer port={} protocol={:?} state={state}\n",
                    self.guest_port, self.version
                ));
                self.peer_state = state;
            }
            match result {
                Poll::Pending => return false,
                Poll::Ready(Err(_)) if self.peer.ping_answered() => {
                    output::arguments(format_args!(
                        "RESTORE PROXY ping complete protocol={:?}\n",
                        self.version
                    ));
                    self.closing = true;
                }
                Poll::Ready(Err(error)) => {
                    if let Some(header) = self.peer.observed_header() {
                        output::arguments(format_args!(
                            "RESTORE PROXY framing header={:02x}:{:02x}:{:02x}\n",
                            header[0], header[1], header[2]
                        ));
                    }
                    self.refuse(error);
                }
                Poll::Ready(Ok(destination)) => {
                    output::arguments(format_args!(
                        "RESTORE PROXY connect {}:{} protocol={:?}\n",
                        destination.host, destination.port, self.version
                    ));
                    match network
                        .0
                        .borrow_mut()
                        .connect(&destination.host, destination.port)
                    {
                        Ok(remote) => {
                            self.remote = Some(remote);
                            self.destination = Some(destination);
                        }
                        Err(error) => {
                            self.destination = Some(destination);
                            self.refuse(error);
                        }
                    }
                }
            }
        }
        if !self.closing {
            let handle = self.guest.expect("opened before relay");

            if self.to_remote.is_empty() && !self.guest_closed {
                let mut bytes = [0; PROXY_READ];
                match Ports::read(ports, handle, &mut bytes) {
                    Poll::Pending => {}
                    Poll::Ready(Ok(0)) => self.guest_closed = true,
                    Poll::Ready(Ok(n)) if n <= bytes.len() => {
                        self.to_remote.extend(&bytes[..n]);
                        self.out_bytes += n as u64;
                    }
                    Poll::Ready(Ok(_)) => self.refuse("invalid guest proxy read length"),
                    Poll::Ready(Err(error)) => self.refuse(error),
                }
            }
            if !self.to_remote.is_empty() && !self.closing {
                let sent = {
                    let remote = self.remote.as_mut().expect("connected before relay");
                    let (first, _) = self.to_remote.as_slices();
                    network.0.borrow_mut().relay_write(remote, first)
                };
                match sent {
                    Ok(0) => {}
                    Ok(n) if n <= self.to_remote.len() => {
                        self.to_remote.drain(..n);
                    }
                    Ok(_) => self.refuse("invalid remote proxy write length"),
                    Err(error) => self.refuse(error),
                }
            }
            if self.to_guest.is_empty() && !self.remote_closed && !self.closing {
                let to_guest = &mut self.to_guest;
                let mut closed = false;
                let received = {
                    let remote = self.remote.as_mut().expect("connected before relay");
                    network.0.borrow_mut().relay_read(remote, |bytes, eof| {
                        if to_guest.len().saturating_add(bytes.len()) > PROXY_BUFFER {
                            return Err(stela_runtime::net::NetError::Invalid(
                                "reverse proxy receive buffer",
                            ));
                        }
                        to_guest.extend(bytes);
                        closed |= eof;
                        Ok(eof)
                    })
                };
                match received {
                    Ok(_) => self.remote_closed |= closed,
                    Err(error) => self.refuse(error),
                }
            }
            if !self.to_guest.is_empty() && !self.closing {
                let written = {
                    let (first, _) = self.to_guest.as_slices();
                    Ports::write(ports, handle, first)
                };
                match written {
                    Poll::Pending => {}
                    Poll::Ready(Ok(n)) if n != 0 && n <= self.to_guest.len() => {
                        self.to_guest.drain(..n);
                        self.in_bytes += n as u64;
                    }
                    Poll::Ready(Ok(_)) => self.refuse("invalid guest proxy write length"),
                    Poll::Ready(Err(error)) => self.refuse(error),
                }
            }
            self.closing |= (self.guest_closed && self.to_remote.is_empty())
                || (self.remote_closed && self.to_guest.is_empty());
        }
        if self.closing {
            if let Some(remote) = self.remote.take() {
                network.0.borrow_mut().close(remote);
            }
            if let Some(handle) = self.guest {
                match Ports::close(ports, handle) {
                    Poll::Pending => return false,
                    Poll::Ready(Err(error)) => output::arguments(format_args!(
                        "RESTORE PROXY guest close refused: {error}\n"
                    )),
                    Poll::Ready(Ok(())) => {}
                }
                self.guest = None;
            }
            let destination = self
                .destination
                .as_ref()
                .map(|d| format!("{}:{}", d.host, d.port))
                .unwrap_or_else(|| String::from("unparsed destination"));
            output::arguments(format_args!(
                "RESTORE PROXY closed {destination} out={} in={}\n",
                self.out_bytes, self.in_bytes
            ));
            return true;
        }
        false
    }
}

struct ReverseProxy {
    control: Control<MuxStream>,
    connections: Vec<ProxyConnection>,
    network: network::Shared,
    stopped: bool,
    restarting: bool,
    tsc_hz: u64,
    negotiation_deadline: u64,
    control_progress: (Version, u8, u64, u64),
    waiting_reported: bool,
}
impl ReverseProxy {
    fn new(network: network::Shared, tsc_hz: u64) -> Self {
        let control = Control::new();
        let now = super::host::rdtsc();
        output::arguments(format_args!(
            "RESTORE PROXY enabled control_port={} negotiation_ms={PROXY_NEGOTIATION_MILLIS}\n",
            stela_runtime::restore::proxy::CONTROL_PORT
        ));
        Self {
            control_progress: control.progress(),
            control,
            connections: Vec::new(),
            network,
            stopped: false,
            restarting: false,
            tsc_hz,
            negotiation_deadline: now.saturating_add(
                (tsc_hz / 1000)
                    .saturating_mul(PROXY_NEGOTIATION_MILLIS)
                    .max(1),
            ),
            waiting_reported: false,
        }
    }

    fn arm_negotiation_deadline(&mut self, now: u64) {
        self.negotiation_deadline = now.saturating_add(
            (self.tsc_hz / 1000)
                .saturating_mul(PROXY_NEGOTIATION_MILLIS)
                .max(1),
        );
    }

    fn poll(&mut self, ports: &mut UsbClient<'_>) {
        let now = super::host::rdtsc();
        if self.restarting {
            match self.control.restart(ports) {
                Poll::Pending => {}
                Poll::Ready(Ok(())) => {
                    self.restarting = false;
                    self.control_progress = self.control.progress();
                    self.arm_negotiation_deadline(now);
                }
                Poll::Ready(Err(error)) => {
                    output::arguments(format_args!(
                        "RESTORE PROXY control retirement refused: {error}\n"
                    ));
                    self.stopped = true;
                }
            }
        }
        if !self.stopped {
            let result = if self.restarting {
                Poll::Pending
            } else {
                self.control.poll(ports)
            };
            match result {
                Poll::Pending => {
                    if !self.restarting && !self.control.ready() {
                        let progress = self.control.progress();
                        if progress != self.control_progress {
                            self.control_progress = progress;
                            self.arm_negotiation_deadline(now);
                        } else if now >= self.negotiation_deadline {
                            if !self.waiting_reported {
                                output::arguments(format_args!(
                                    "RESTORE PROXY waiting for guest service: control negotiation timed out\n"
                                ));
                                self.waiting_reported = true;
                            }
                            self.restarting = true;
                        }
                    }
                }
                Poll::Ready(Ok(ProxyEvent::Connection { port, version })) => {
                    if self.waiting_reported {
                        output::arguments(format_args!(
                            "RESTORE PROXY control available protocol={version:?}\n"
                        ));
                        self.waiting_reported = false;
                    }
                    if self.connections.len() == PROXY_CONNECTIONS {
                        output::arguments(format_args!(
                            "RESTORE PROXY refused connection: {PROXY_CONNECTIONS} active\n"
                        ));
                        self.stopped = true;
                    } else {
                        self.connections.push(ProxyConnection::new(port, version));
                    }
                }
                Poll::Ready(Ok(ProxyEvent::Shutdown)) => {
                    output::arguments(format_args!("RESTORE PROXY shutdown\n"));
                    self.stopped = true;
                    for connection in &mut self.connections {
                        connection.closing = true;
                    }
                }
                Poll::Ready(Err(error)) => {
                    if !self.waiting_reported {
                        output::arguments(format_args!(
                            "RESTORE PROXY waiting for guest service: {error}\n"
                        ));
                        self.waiting_reported = true;
                    }
                    self.restarting = true;
                }
            }
        }
        let mut at = 0;
        while at < self.connections.len() {
            if self.connections[at].poll(ports, &self.network) {
                self.connections.remove(at);
            } else {
                at += 1;
            }
        }
    }
}
impl Experiment {
    /// The study runs `identity`; the install the inventory's one Erase
    /// identity, and the update its one Update identity. The payload is
    /// checked block by block as the restore reads it when its hashes are
    /// recorded, and every member before StartRestore otherwise, unless
    /// `verify` is [`Verify::OnRead`], which refuses a payload without them.
    pub fn new(
        source: Source,
        usb: UsbId,
        handoff: &'static Handoff,
        mode: Mode,
        identity: Option<usize>,
        verify: Verify,
        services: Services,
    ) -> Result<Self> {
        let Services {
            signer,
            fdr_storage,
            network,
        } = services;
        let manifest = inventory(handoff)?;
        let purpose = match mode {
            Mode::Update => Purpose::Update,
            Mode::Install | Mode::DevelopmentInstall | Mode::Study => Purpose::Erase,
        };
        let identity = match identity {
            Some(identity) => identity,
            None => manifest
                .restore_identity(purpose)
                .map_err(|error| refuse!("restore identity: {error}", %error))?
                as usize,
        };
        // Recorded block hashes check each block as the restore reads it;
        // without them every member is read whole first, to build them.
        let recorded = manifest.block_hashes.is_some();
        let verify = match (recorded, verify) {
            (true, _) => Verify::OnRead,
            (false, Verify::OnRead) => {
                return Err(refuse!(
                    "--verify-on-read needs the payload's recorded block hashes, and this restore inventory names none"
                ));
            }
            (false, Verify::Full) => Verify::Full,
        };
        output::arguments(format_args!(
            "{} build={} source_sha256={} manifest_sha256={} identity={} bytes={} hashes={} verify={}\n",
            match mode {
                Mode::Study => "RESTORE STUDY UNQUALIFIED",
                Mode::DevelopmentInstall => "DEVELOPMENT INSTALL TSS-ENABLED FDR-OFFLINE",
                mode => mode.label(),
            },
            manifest.product_build,
            manifest.source.sha256.as_deref().unwrap_or("-"),
            manifest.source.build_manifest_sha256,
            identity,
            manifest.source.bytes,
            if recorded { "recorded" } else { "scan" },
            match verify {
                Verify::Full => "full",
                Verify::OnRead => "on-read",
            }
        ));
        let mut store = Store::open(source, manifest, identity)
            .map_err(|error| refuse!("restore store: {error}", %error))?;
        store.set_verify(verify);
        if recorded {
            // Both modes check reads against the manifest's recorded hashes.
            let blocks = inventory_file(handoff, RESTORE_BLOCKS_FILE).ok_or(refuse!(
                "the restore inventory's block hashes were not loaded"
            ))?;
            store
                .adopt_static_block_hashes(blocks)
                .map_err(|error| refuse!("restore block hashes: {error}", %error))?;
        }
        let mut answers = Answers::new(store);
        let route = match mode {
            Mode::Install if answers.has_system_image() => Route::Stock,
            Mode::Install => Route::Recovery,
            Mode::DevelopmentInstall if !answers.has_system_image() => {
                return Err(refuse!(
                    "--development-restore needs a payload containing the macOS system image"
                ));
            }
            Mode::DevelopmentInstall => Route::Development,
            Mode::Update => Route::Stock,
            Mode::Study => Route::StelaFirmwareStudy,
        };
        if mode == Mode::DevelopmentInstall && network.is_some() {
            return Err(refuse!(
                "the development restore refuses an FDR reverse-proxy network"
            ));
        }
        if let Some(bytes) = inventory_file(handoff, stela_runtime::restore::fdr::FILE) {
            if mode == Mode::DevelopmentInstall {
                return Err(refuse!(
                    "the development restore refuses an FDR diagnostic/archive"
                ));
            }
            let placement = super::fdr::Placement::parse(bytes)?;
            answers.fdr_diagnostic = Some(
                stela_runtime::restore::fdr::Diagnostic::decode(placement.diagnostic)
                    .map_err(|error| refuse!("FDR diagnostic refused: {error}", %error))?,
            );
            let device =
                fdr_storage.ok_or(refuse!("the required FDR storage volume was not opened"))?;
            answers.fdr_commit = Some(super::fdr::writer(device, &placement)?);
            output::arguments(format_args!(
                "FDR DIAGNOSTIC bytes={} sha256={} original Memory-store import; durable host commit enabled; factory trust unqualified\n",
                placement.diagnostic.len(),
                stela_formats::sha256(placement.diagnostic),
            ));
        }
        if signer.is_some() {
            answers.signer = signer;
            // The bundle boots under the AP ticket Apple personalized for it,
            // which also answers RootTicket.
            let booted = inventory_file(handoff, BOOT_MANIFEST_FILE)
                .ok_or(refuse!("install: the bundle has no boot manifest"))?;
            answers.boot_with(booted.to_vec()).map_err(|error| refuse!("install: the bundle's boot manifest is not personalized for this machine: {error}", %error))?;
        }
        // Serve the system-image content-key GET from the inventory's keys.
        answers
            .stage_system_image_keys()
            .map_err(|error| refuse!("system image keys: {error}", %error))?;
        if let Some(wall) = handoff.wall {
            // The firmware's time at the handoff, carried forward on the TSC.
            let hz = handoff.tsc_hz;
            answers.clock = Some(Box::new(move || {
                wall.unix_seconds_at(super::host::rdtsc(), hz)
            }));
        }
        let driver = Driver::new(answers, route)
            .map_err(|error| refuse!("restore driver: {error}", %error))?;
        Ok(Self {
            usb,
            driver,
            proxy: network.map(|network| ReverseProxy::new(network, handoff.tsc_hz)),
            terminal: false,
            failure: None,
            ledger: 0,
            mode,
            backpressure: MuxBackpressure::default(),
            asr_stream: None,
        })
    }
    /// Read ahead from the host's storage, without the guest's devices.
    pub fn pump(&mut self) {
        let source = self.driver.answers.store.reader_mut();
        if self.terminal {
            // The service owns the restore and its network here. Retire the
            // remote range outside the device lock and leave queued disk I/O
            // to its existing owner.
            if let Some(remote) = source.remote.take() {
                remote.borrow_mut().retire();
            }
        } else {
            source.pump();
        }
    }

    /// Read ahead, verified, what the restore sends next, without the
    /// guest's devices: [`Self::poll`] then only copies it to the guest.
    pub fn prefetch(&mut self) {
        if !self.terminal {
            self.driver.prefetch(PREFETCH_CHUNKS);
        }
    }

    /// Answer the guest's last request, without the guest's devices: the
    /// answers sign with Apple's TSS and read the image over the network,
    /// which every PE's device access would otherwise wait out.
    pub fn work(&mut self) {
        if !self.terminal {
            self.driver.work();
        }
    }

    /// The image sent so far and the share of polls the host was behind
    /// (no chunk read ahead) or the guest was (the mux took no more), while
    /// the restore sends one; and the writes the guest's receive window and
    /// the mux's transmit queue each refused, with the largest window.
    pub fn asr_report(&self) -> Option<String> {
        let asr = self.driver.asr()?;
        let (sent, size) = asr.progress();
        let backpressure = self.backpressure;
        let mut report = format!(
            "ASR HOST sent={}MiB/{}MiB polls_host_behind={} polls_guest_full={} \
             refused_window={} refused_queue={} largest_window={}KiB",
            sent >> 20,
            size >> 20,
            asr.starved,
            asr.full,
            backpressure.window,
            backpressure.queue,
            backpressure.largest_window >> 10
        );
        if let Some(payload) = asr.payload_progress() {
            let _ = write!(
                report,
                " payload_verified_bytes={} payload_staged_bytes={} payload_accepted_bytes={}",
                payload.verified_ahead_bytes, payload.staged_bytes, payload.accepted_bytes,
            );
        }
        if let Some(stream) = self.asr_stream {
            let _ = write!(
                report,
                " mux_accepted_bytes={} mux_acked_bytes={} mux_window_bytes={} \
                 mux_outstanding_bytes={} mux_zero_window_writes={} mux_full_window_writes={} \
                 mux_credit_wait_ticks={} mux_credit_waiting={}",
                stream.accepted_bytes,
                stream.acknowledged_bytes,
                stream.window_bytes,
                stream.outstanding_bytes,
                stream.zero_window_writes,
                stream.full_window_writes,
                stream.credit_wait_ticks,
                stream.credit_waiting,
            );
        }
        Some(report)
    }

    /// Serve the guest's restore client; [`Self::pump`] read ahead and
    /// [`Self::work`] answered its last request first.
    pub fn poll(&mut self, devices: &mut AppleDevices) {
        if let Some((message, at)) = &self.failure
            && output::millis_since(*at) >= FAILURE_GRACE_MS
        {
            output::fatal(message);
        }
        if self.terminal {
            return;
        }
        let label = self.mode.label();
        let mut client = devices
            .usb_client(self.usb)
            .unwrap_or_else(|e| output::fatal(e));
        if let Some(proxy) = &mut self.proxy {
            proxy.poll(&mut client);
        }
        let result = self.driver.poll_prefetched(&mut client);
        self.backpressure = client.backpressure();
        self.asr_stream = self
            .driver
            .asr_transfer()
            .and_then(|(_, handle)| client.stream_stats(handle?));
        for message in &self.driver.session.ledger()[self.ledger..] {
            output::arguments(format_args!("{label} ledger={message:?}\n"));
        }
        self.ledger = self.driver.session.ledger().len();
        match result {
            Poll::Pending => {}
            Poll::Ready(Ok(())) => {
                self.terminal = true;
                output::arguments(format_args!(
                    "{}\n",
                    match self.mode {
                        Mode::Install =>
                            "INSTALL protocol complete; restart into recoveryOS (Stela)",
                        Mode::DevelopmentInstall => {
                            "DEVELOPMENT INSTALL protocol complete; TSS enabled, FDR offline/unprovisioned; installed boot remains unqualified"
                        }
                        Mode::Update => "UPDATE protocol complete; reboot into macOS (Stela)",
                        Mode::Study => {
                            "RESTORE STUDY protocol complete; installed boot is unqualified"
                        }
                    }
                ));
            }
            Poll::Ready(Err(error)) => {
                output::arguments(format_args!("{label} REFUSED {error}\n"));
                // The restore has no automatic reboot/retry or fallback: the
                // board stops once the guest has had time to report why.
                self.terminal = true;
                self.failure = Some((
                    format!(
                        "{} failed: {}",
                        match self.mode {
                            Mode::Install => "the install",
                            Mode::DevelopmentInstall => "the development install",
                            Mode::Update => "the update",
                            Mode::Study => "the restore study",
                        },
                        error.summary()
                    ),
                    output::now(),
                ));
            }
        }
    }
}
