//! The mount's own network client over the host's first network adapter:
//! the restore client's signing-server requests (TSS), the
//! connectivity check (`--network-check`) and a remote install image's
//! ranged downloads (`engine::remote`), which share it.
//!
//! UEFI's Simple Network Protocol ends with boot services, and the request
//! is made during the restore, long after ExitBootServices, so the client
//! runs the runtime's dual-stack client (`stela_runtime::net`) over the adapter's
//! after-EBS driver (`stela_drivers::net::virtio`, `stela_drivers::net::igc`
//! for an Intel I225/I226, `stela_drivers::net::e1000e` for an Intel 82574L,
//! `stela_drivers::net::igb` for supported Intel 82575–I211 copper adapters,
//! or `stela_drivers::usb::cdc` for a USB Ethernet
//! adapter). Its registers and DMA pages are identity-mapped in the
//! loader's root, so every frame moves inside that root
//! (`storage::RestoreRoot`), as block and input I/O do.
mod security;
use super::storage::RestoreRoot;
use crate::Refusal;
use crate::Result;
use crate::output;
use alloc::{collections::VecDeque, format, rc::Rc, string::String, sync::Arc, vec::Vec};
use core::{cell::RefCell, mem::ManuallyDrop};
use stela_contracts::refuse;
use stela_drivers::{
    Clock,
    net::{Link, LinkError, Mac},
    probe::Adapter,
    usb::xhci::SlotId,
};
use stela_runtime::net::{
    HttpPost, HttpRequest, HttpTransfer, NetError, Stack, TlsUnavailable, Transfer,
    stack::DOWNLOAD_BUFFER,
};
use stela_soc::machine::AppleDevices;

/// One restore-time opaque TCP relay window. TLS stays end to end between the
/// original Apple guest process and its requested server.
pub const RELAY_BUFFER: usize = 1 << 20;

/// No adapter a native driver runs: what to connect, as the screen says it.
const NO_ADAPTER: &str = "No supported network adapter. Connect wired Ethernet (most Intel, Realtek, Broadcom and Aquantia adapters, or a Realtek or ASIX USB adapter), or set Wi-Fi with `stela setup configure --wifi SSID --usb USB` on another computer.";
/// What the screen says when DHCP gives the first adapter no lease.
const NO_LEASE: &str = "No automatic network address. Connect Ethernet or Wi-Fi to a network with DHCPv4 or IPv6 router advertisements, or set an address with `stela setup configure --ipv4 ADDRESS/PREFIX --usb USB` on another computer. Stateful DHCPv6 is unsupported.";

/// The host's network adapters and the loader root they are reached in.
/// Adapters left in it are dropped in that root.
pub struct Links {
    pub devices: Vec<Adapter>,
    pub root: u64,
}

impl Drop for Links {
    fn drop(&mut self) {
        let devices = core::mem::take(&mut self.devices);
        if !devices.is_empty() {
            in_loader_root(self.root, || drop(devices));
        }
    }
}

/// Run `release` in the loader's root, the only root that maps the
/// adapters' registers and DMA pages: dropping an adapter can drive its
/// device (`Igc` stops its controller), and outside that root its first
/// register access faults. A processor already inside the host drivers
/// (`storage::open`'s probe) is in that root and holds their lock.
fn in_loader_root(root: u64, release: impl FnOnce()) {
    let _root = (!super::storage::host_drivers_held_here()).then(|| {
        // SAFETY: the loader root `storage::open` captured, which maps the
        // adapters; the processor's pinned record is active, as for every
        // `RootLink` call.
        unsafe { RestoreRoot::enter(root) }
    });
    release();
}

struct RootLink {
    /// Dropped only in the loader's root ([`in_loader_root`]).
    device: ManuallyDrop<Adapter>,
    root: u64,
}

impl RootLink {
    fn new(device: Adapter, root: u64) -> Self {
        Self {
            device: ManuallyDrop::new(device),
            root,
        }
    }

    fn usb_slot(&self) -> Option<SlotId> {
        match &*self.device {
            Adapter::Usb(link) => Some(link.slot_id()),
            Adapter::UsbVendor(link) => Some(link.slot_id()),
            _ => None,
        }
    }
}

impl Drop for RootLink {
    fn drop(&mut self) {
        // SAFETY: taken once, here; the field is never used again.
        let device = unsafe { ManuallyDrop::take(&mut self.device) };
        in_loader_root(self.root, || drop(device));
    }
}

impl Link for RootLink {
    fn mac(&self) -> Mac {
        self.device.mac()
    }
    fn carrier(&mut self) -> bool {
        // SAFETY: the loader root `storage::open` captured maps this
        // device.
        let _root = unsafe { RestoreRoot::enter(self.root) };
        self.device.carrier()
    }
    fn transmit(&mut self, frame: &[u8]) -> Result<(), LinkError> {
        // SAFETY: as in `carrier`.
        let _root = unsafe { RestoreRoot::enter(self.root) };
        self.device.transmit(frame)
    }
    fn receive(&mut self) -> Result<Option<Vec<u8>>, LinkError> {
        // SAFETY: as in `carrier`.
        let _root = unsafe { RestoreRoot::enter(self.root) };
        self.device.receive()
    }
    fn receive_reusing(&mut self, frame: &mut Vec<u8>) -> Result<bool, LinkError> {
        // SAFETY: as in `carrier`.
        let _root = unsafe { RestoreRoot::enter(self.root) };
        self.device.receive_reusing(frame)
    }
    fn signals(&self) -> bool {
        self.device.signals()
    }
    fn due_us(&self) -> Option<u64> {
        self.device.due_us()
    }
}

/// The PC's clock, moving the screen's running step as the stack waits.
struct Ticking(Clock);

impl stela_runtime::net::Clock for Ticking {
    fn now_us(&self) -> Result<u64, stela_runtime::net::ClockError> {
        crate::status::tick();
        self.0
            .try_now_us()
            .map_err(stela_runtime::net::ClockError::Calibration)
    }
    /// The adapter's completion signal wakes the boot processor, which runs
    /// the stack.
    fn sleep(&self, until_us: u64) -> Result<(), stela_runtime::net::ClockError> {
        let now = self
            .0
            .try_now_us()
            .map_err(stela_runtime::net::ClockError::Calibration)?;
        if until_us > now {
            super::host::sleep_for(until_us - now);
        }
        Ok(())
    }
}

pub struct HostNetwork {
    stack: Stack<RootLink, Ticking>,
    security: Result<Arc<stela_tls::ClientConfig>, TlsUnavailable>,
    /// The address and adapter, for the screen.
    description: String,
}

impl HostNetwork {
    /// Try each native adapter until DHCPv4 or IPv6 SLAAC provides a usable
    /// address. A failed adapter keeps its diagnostic while the next is tried.
    pub fn open(
        mut links: Links,
        clock: Clock,
        handoff: &crate::handoff::Handoff,
        options: &stela_boot_core::options::System,
    ) -> Result<Self> {
        let security = security::native(handoff);
        match &security {
            Ok(_) => output::arguments(format_args!(
                "NETWORK TLS certificate time: the firmware RTC carried forward on the invariant TSC\n"
            )),
            Err(unavailable) => {
                output::arguments(format_args!("NETWORK TLS unavailable: {unavailable}\n"))
            }
        }
        let mut failures = Vec::new();
        // Each adapter is owned by a `RootLink` or still by `links` on every
        // path out, so it is dropped in the loader's root.
        while !links.devices.is_empty() {
            let link = RootLink::new(links.devices.remove(0), links.root);
            let driver = link.device.driver();
            let location = String::from(link.device.location());
            output::arguments(format_args!(
                "NETWORK candidate driver={driver} adapter={location} mac={}\n",
                link.mac()
            ));
            crate::status::progress(&format!("adapter {location}"));
            let mut seed = [0u8; 8];
            crate::random::fill(&mut seed)
                .map_err(|_| refuse!("secure host entropy is unavailable for the network stack"))?;
            let mut stack = Stack::new(link, Ticking(clock), u64::from_le_bytes(seed))
                .map_err(|error| refuse!("host network stack: {error}", %error))?;
            if let Some(ipv4) = options.ipv4 {
                stack
                    .set_ipv4(ipv4.address, ipv4.prefix, options.ipv4_gateway)
                    .map_err(|error| refuse!("host network --ipv4 {ipv4}: {error}", %error))?;
            }
            if !options.dns.is_empty() {
                stack
                    .set_dns(&options.dns)
                    .map_err(|error| refuse!("host network --dns: {error}", %error))?;
            }
            match stack.automatic() {
                Ok(config) => {
                    output::arguments(format_args!("NETWORK driver={driver}\n"));
                    output::arguments(format_args!(
                        "NETWORK adapter={location} addresses={:?} dns={:?}\n",
                        config.addresses, config.dns
                    ));
                    let description = format!("{:?} on adapter {location}", config.addresses);
                    return Ok(Self {
                        stack,
                        description,
                        security,
                    });
                }
                Err(error) => {
                    output::arguments(format_args!("NETWORK adapter {location} failed: {error}\n"));
                    failures.push(format!("{location}: {error}"));
                    // Its driver stops it, in the loader's root (`RootLink`).
                    drop(stack);
                    output::arguments(format_args!("NETWORK adapter {location} released\n"));
                }
            }
        }
        if failures.is_empty() {
            Err(refuse!("{NO_ADAPTER}", NO_ADAPTER))
        } else {
            Err(refuse!(
                "{NO_LEASE} {failures}",
                NO_LEASE,
                failures = failures.join("; ")
            ))
        }
    }

    /// The address and adapter, as the screen names them.
    pub fn describe(&self) -> &str {
        &self.description
    }

    /// The connectivity check's POST (`--network-check`), waiting for its
    /// response. Redirects keep the full resulting URL and can never
    /// downgrade an HTTPS hop.
    #[cfg(feature = "diagnostics")]
    pub fn post_url(
        &mut self,
        url: &str,
        headers: &[(&str, &str)],
        content_type: &str,
        body: &[u8],
    ) -> Result<Vec<u8>, NetError> {
        let result = self
            .stack
            .post_url(url, headers, content_type, body, self.security.clone());
        match &result {
            Ok(reply) => output::arguments(format_args!(
                "NETWORK POST {url} sent={} received={}\n",
                body.len(),
                reply.len()
            )),
            Err(error) => output::arguments(format_args!("NETWORK POST {url} refused: {error}\n")),
        }
        result
    }

    /// Starts a POST without waiting ([`Self::read_post`]).
    pub fn open_post(
        &mut self,
        url: &str,
        headers: &[(&str, &str)],
        content_type: &str,
        body: &[u8],
    ) -> Result<HttpPost, NetError> {
        self.stack
            .open_post(url, headers, content_type, body, self.security.clone())
            .inspect_err(|error| {
                output::arguments(format_args!(
                    "NETWORK POST {url} refused: {error}
"
                ))
            })
    }

    /// The POST's response body once it has arrived, `None` while it is on
    /// its way.
    pub fn read_post(&mut self, post: &mut HttpPost) -> Result<Option<Vec<u8>>, NetError> {
        let result = self.stack.read_post(post);
        let url = post.url();
        match &result {
            Ok(Some(reply)) => output::arguments(format_args!(
                "NETWORK POST {url} sent={} received={}
",
                post.sent(),
                reply.len()
            )),
            Ok(None) => {}
            Err(error) => output::arguments(format_args!(
                "NETWORK POST {url} refused: {error}
"
            )),
        }
        result
    }

    pub fn close_post(&mut self, post: HttpPost) {
        self.stack.close_post(post);
    }

    pub fn check_url(&self, url: &str) -> Result<(), NetError> {
        let destination = stela_runtime::net::http::Url::parse(url).map_err(NetError::Url)?;
        if destination.scheme == stela_runtime::net::http::Scheme::Https {
            self.security
                .as_ref()
                .map_err(|error| NetError::TlsUnavailable(*error))?;
        }
        Ok(())
    }
    pub fn download(
        &mut self,
        url: &str,
        first: u64,
        last: u64,
        etag: &str,
    ) -> Result<HttpTransfer, NetError> {
        self.open_download(url, first, last, Some(etag))
    }

    fn open_download(
        &mut self,
        url: &str,
        first: u64,
        last: u64,
        etag: Option<&str>,
    ) -> Result<HttpTransfer, NetError> {
        self.stack.open_http(
            url,
            HttpRequest::range(first, last, etag),
            DOWNLOAD_BUFFER,
            self.security.clone(),
        )
    }

    /// Read one bounded public object prefix through the restore download path.
    #[cfg(feature = "diagnostics")]
    pub fn check_public_range(&mut self, url: &str) -> Result<(u64, Vec<u8>), NetError> {
        use stela_runtime::net::http::{Ranged, Scheme, Url, head};

        const LAST_BYTE: u64 = 4095;
        if Url::parse(url).map_err(NetError::Url)?.scheme != Scheme::Https {
            return Err(NetError::Invalid("the range check requires HTTPS"));
        }
        let mut transfer = self.open_download(url, 0, LAST_BYTE, None)?;
        let mut range = None;
        let mut body = Vec::new();
        let start = output::now();
        let result = loop {
            let received = self.read_download(&mut transfer, |bytes, closed| {
                if range.is_none() {
                    // Stack::read_http delivers the complete accepted header in
                    // the first callback, including any initial body bytes.
                    let response = head(bytes, closed)?
                        .ok_or(NetError::Truncated("HTTPS range check header"))?;
                    let Some((0, last, size)) = response.range else {
                        return Err(NetError::Invalid("HTTPS range check response extent"));
                    };
                    if last != (size - 1).min(LAST_BYTE) {
                        return Err(NetError::Invalid("HTTPS range check response extent"));
                    }
                    range = Some(Ranged::new(0, last));
                }
                range
                    .as_mut()
                    .expect("range header checked above")
                    .feed(bytes, closed, |data| body.extend_from_slice(data))
            });
            match received {
                Ok(true) => break Ok(()),
                Ok(false) if output::millis_since(start) < 30_000 => {
                    if let Err(error) = self.wait(50_000) {
                        break Err(error);
                    }
                }
                Ok(false) => break Err(NetError::Timeout("HTTPS range check")),
                Err(error) => break Err(error),
            }
        };
        self.close_download(transfer);
        result?;
        let size = range
            .and_then(|range| range.size)
            .ok_or(NetError::Truncated("HTTPS range check body"))?;
        Ok((size, body))
    }

    /// Exercise pinned restore downloads together, retaining only their prefixes.
    #[cfg(feature = "diagnostics")]
    pub fn check_exact_ranges(
        &mut self,
        checks: &[stela_boot_core::options::NetworkCheckRange],
    ) -> bool {
        use stela_runtime::net::http::Ranged;

        const SAMPLE_BYTES: u64 = 16 * 1024;
        let mut transfers: Vec<_> = checks
            .iter()
            .map(|check| {
                let range =
                    Ranged::with_identity(check.first, check.last, &check.etag, check.total)?;
                let transfer = self.download(&check.url, check.first, check.last, &check.etag)?;
                Ok::<_, NetError>((transfer, range, Vec::new(), None))
            })
            .collect();
        let start = output::now();
        let mut wait_error = None;
        loop {
            let mut pending = false;
            for (check, transfer) in checks.iter().zip(&mut transfers) {
                let Ok((transfer, range, prefix, result)) = transfer else {
                    continue;
                };
                if result.is_some() {
                    continue;
                }
                let sample = (check.last - check.first + 1).min(SAMPLE_BYTES) as usize;
                match self.read_download(transfer, |bytes, closed| {
                    let done = range.feed(bytes, closed, |data| {
                        let take = data.len().min(sample - prefix.len());
                        prefix.extend_from_slice(&data[..take]);
                    })?;
                    Ok(done || prefix.len() == sample)
                }) {
                    Ok(true) => *result = Some(Ok(())),
                    Ok(false) => pending = true,
                    Err(error) => *result = Some(Err(error)),
                }
            }
            if !pending || output::millis_since(start) >= 35_000 {
                break;
            }
            if let Err(error) = self.wait(50_000) {
                wait_error = Some(error);
                break;
            }
        }
        let mut passed = true;
        for (index, (check, transfer)) in checks.iter().zip(transfers).enumerate() {
            let (header, prefix, result) = match transfer {
                Ok((transfer, range, prefix, result)) => {
                    self.close_download(transfer);
                    (range.size.is_some(), prefix, result)
                }
                Err(error) => (false, Vec::new(), Some(Err(error))),
            };
            let elapsed_ms = output::millis_since(start);
            if matches!(result, Some(Ok(()))) {
                output::arguments(format_args!(
                    "NETWORK EXACT RANGE id={index} first={} last={} object={} status=206 bytes={} sha256={} elapsed_ms={elapsed_ms}\n",
                    check.first,
                    check.last,
                    check.total,
                    prefix.len(),
                    stela_formats::sha256(&prefix),
                ));
                continue;
            }
            passed = false;
            let timeout = NetError::Timeout("exact range check");
            let error = result
                .as_ref()
                .and_then(|result| result.as_ref().err())
                .or(wait_error.as_ref())
                .unwrap_or(&timeout);
            output::arguments(format_args!(
                "NETWORK EXACT RANGE id={index} first={} last={} accepted_header={header} bytes={} elapsed_ms={elapsed_ms} error={error}\n",
                check.first,
                check.last,
                prefix.len(),
            ));
        }
        passed
    }

    /// Fill the first range's TCP receive queue, abort it, then reopen on the
    /// same network stack as a remote restore does on a distant archive read.
    #[cfg(feature = "diagnostics")]
    pub fn check_reopened_range(
        &mut self,
        first: &stela_boot_core::options::NetworkCheckRange,
        next: &stela_boot_core::options::NetworkCheckRange,
    ) -> Result<bool, NetError> {
        use stela_runtime::net::http::Ranged;

        let mut range = Ranged::with_identity(first.first, first.last, &first.etag, first.total)?;
        let mut transfer = self.download(&first.url, first.first, first.last, &first.etag)?;
        let start = output::now();
        let primed = (|| {
            let mut consumed = 0usize;
            while consumed == 0 {
                let ended = self.read_download(&mut transfer, |bytes, closed| {
                    range.feed(bytes, closed, |body| consumed += body.len())
                })?;
                if ended {
                    return Err(NetError::Invalid("the prime range completed before abort"));
                }
                if output::millis_since(start) >= 30_000 {
                    return Err(NetError::Timeout("range reopen prime body"));
                }
                if consumed == 0 {
                    self.wait(50_000)?;
                }
            }
            loop {
                self.stack.poll()?;
                let (queued, capacity) =
                    self.stack
                        .http_receive_queue(&transfer)
                        .ok_or(NetError::Invalid(
                            "range reopen prime requires a plain socket",
                        ))?;
                if queued == capacity {
                    return Ok((queued, consumed));
                }
                if output::millis_since(start) >= 30_000 {
                    return Err(NetError::Timeout("range reopen prime receive queue"));
                }
                self.wait(50_000)?;
            }
        })();
        self.close_download(transfer);
        let (queued, consumed) = primed?;
        output::arguments(format_args!(
            "NETWORK EXACT RANGE REOPEN aborted first={} consumed_bytes={consumed} full_queue_bytes={queued}\n",
            first.first,
        ));
        Ok(self.check_exact_ranges(core::slice::from_ref(next)))
    }

    pub fn read_download(
        &mut self,
        transfer: &mut HttpTransfer,
        take: impl FnMut(&[u8], bool) -> Result<bool, NetError>,
    ) -> Result<bool, NetError> {
        self.stack.read_http(transfer, take)
    }
    pub fn close_download(&mut self, transfer: HttpTransfer) {
        self.stack.close_http(transfer);
    }

    pub fn close(&mut self, transfer: Transfer) {
        self.stack.close(transfer);
    }

    /// Waits up to `micros` for the adapter's next frame (`Stack::wait`).
    pub fn wait(&mut self, micros: u64) -> Result<(), NetError> {
        self.stack.wait(micros)
    }

    /// Opens one opaque TCP stream for the restore reverse proxy.
    pub fn connect(&mut self, host: &str, port: u16) -> Result<Transfer, NetError> {
        self.stack.open_raw(host, port, RELAY_BUFFER)
    }

    /// Sends caller-kept opaque bytes, returning zero under backpressure.
    pub fn relay_write(
        &mut self,
        transfer: &mut Transfer,
        bytes: &[u8],
    ) -> Result<usize, NetError> {
        self.stack.write(transfer, bytes)
    }

    /// Receives opaque bytes. `take` owns framing and reports when the remote
    /// close has completed the relay.
    pub fn relay_read(
        &mut self,
        transfer: &mut Transfer,
        take: impl FnMut(&[u8], bool) -> Result<bool, NetError>,
    ) -> Result<bool, NetError> {
        self.stack.read(transfer, take)
    }
}

/// The host network, shared by the signing transport and a remote install
/// image; only the service reaches it, which one core at a time holds, so a
/// borrow never overlaps another.
#[derive(Clone)]
pub struct Shared(pub Rc<RefCell<HostNetwork>>);

/// The restore client's way to Apple's signing service (TSS): one POST
/// at a time, moved along by each host service without waiting, so no
/// guest processor stalls while Apple answers.
pub struct Signing {
    network: Shared,
    /// The POST in flight and the body it sends.
    exchange: Option<(Vec<u8>, HttpPost)>,
}

impl Signing {
    pub fn new(network: Shared) -> Self {
        Self {
            network,
            exchange: None,
        }
    }
}

impl stela_runtime::restore::tss::Transport for Signing {
    fn exchange(
        &mut self,
        host: &str,
        path: &str,
        headers: &[(&str, &str)],
        content_type: &str,
        body: &[u8],
    ) -> stela_runtime::restore::Result<Option<Vec<u8>>> {
        let failed = |error: NetError| {
            stela_runtime::restore::Error::Transport(refuse!("signing request: {error}", %error))
        };
        let mut network = self.network.0.borrow_mut();
        if let Some((sent, _)) = &self.exchange
            && sent != body
        {
            let (_, post) = self.exchange.take().unwrap();
            network.close_post(post);
        }
        let (_, post) = match &mut self.exchange {
            Some(exchange) => exchange,
            None => {
                let url = format!("http://{host}{path}");
                let post = network
                    .open_post(&url, headers, content_type, body)
                    .map_err(failed)?;
                self.exchange.insert((body.to_vec(), post))
            }
        };
        let result = network.read_post(post);
        if let Ok(None) = result {
            return Ok(None);
        }
        let (_, post) = self.exchange.take().unwrap();
        network.close_post(post);
        result.map_err(failed)
    }
}

/// The signing transport when the host network did not open: every request
/// is refused with the reason, so a restore stops at its first signing
/// request.
pub struct Offline(pub Refusal);

impl stela_runtime::restore::tss::Transport for Offline {
    fn exchange(
        &mut self,
        host: &str,
        path: &str,
        _headers: &[(&str, &str)],
        _content_type: &str,
        _body: &[u8],
    ) -> stela_runtime::restore::Result<Option<Vec<u8>>> {
        Err(stela_runtime::restore::Error::Transport(
            refuse!("no host network for http://{host}{path}", host, path).because(self.0.clone()),
        ))
    }
}

/// The guest's USB Ethernet adapter bridged to a host adapter of its own
/// (`--guest-network`). The adapter keeps the host adapter's station
/// address, so the guest is that adapter on the host's network and the
/// adapter needs no promiscuous mode. Frames move between engine slices.
pub struct GuestNetwork {
    link: Option<RootLink>,
    root: u64,
    /// Late adapters can wait for the guest hub to acknowledge a disconnect.
    available: VecDeque<RootLink>,
    up: Option<bool>,
    /// The first link refusal, reported once.
    refused: bool,
    /// Frames read from the host adapter that the guest's receive queue
    /// had no room for yet, delivered first at the next turn.
    received: VecDeque<Vec<u8>>,
}

/// Frames read from the host adapter in one service turn at most, outside
/// the devices' lock: the rest wait in the adapter's ring.
const RECEIVE_BURST: usize = 64;

impl GuestNetwork {
    /// The second adapter when the mount's own client (`wants_client`)
    /// keeps the first, otherwise the first. Keep the guest port reserved if
    /// no host link is free yet.
    pub fn open(links: &mut Links, wants_client: bool) -> Self {
        let index = usize::from(wants_client);
        let link = (index < links.devices.len())
            .then(|| RootLink::new(links.devices.remove(index), links.root));
        if let Some(link) = &link {
            Self::report_adapter(link);
        } else {
            output::arguments(format_args!(
                "GUEST NETWORK waiting for a USB network adapter\n"
            ));
        }
        Self {
            link,
            root: links.root,
            available: VecDeque::new(),
            up: None,
            refused: false,
            received: VecDeque::new(),
        }
    }

    pub fn mac(&self) -> Option<Mac> {
        self.link.as_ref().map(Link::mac)
    }

    fn report_adapter(link: &RootLink) {
        output::arguments(format_args!(
            "GUEST NETWORK adapter {} mac={}\n",
            link.device.location(),
            link.mac()
        ));
    }

    /// Retire exact generations before forwarding another frame. Drop host
    /// owners outside the guest devices' lock and signal the guest hub even
    /// if a replacement has already arrived in the same poll.
    pub(super) fn usb_changed(
        &mut self,
        board: &super::parallel::Board,
        added: Vec<Adapter>,
        removed: &[SlotId],
    ) {
        let retired = |link: &RootLink| link.usb_slot().is_some_and(|slot| removed.contains(&slot));
        if self.link.as_ref().is_some_and(retired) {
            board.serve(AppleDevices::detach_network);
            self.link = None;
            self.received.clear();
            self.refused = false;
            output::arguments(format_args!("GUEST NETWORK host USB adapter removed\n"));
            self.status(None);
        }
        self.available.retain(|link| !retired(link));
        self.available.extend(
            added
                .into_iter()
                .map(|link| RootLink::new(link, self.root))
                .filter(|link| !retired(link)),
        );
    }

    pub(super) fn has_work(&self) -> bool {
        self.link.is_some() || !self.available.is_empty()
    }

    /// Publish a new MAC as a fresh virtual USB function. The hub defers
    /// publication while the previous disconnect is pending at the guest.
    pub(super) fn activate(&mut self, board: &super::parallel::Board) -> bool {
        if self.link.is_some() {
            return true;
        }
        while let Some(link) = self.available.front() {
            let mac = link.mac();
            match board.serve(|devices| devices.attach_network(mac)) {
                Ok(true) => {
                    Self::report_adapter(link);
                    self.link = self.available.pop_front();
                    self.up = None;
                    return true;
                }
                Ok(false) => return false,
                Err(error) => {
                    output::arguments(format_args!("GUEST NETWORK hotplug refused: {error}\n"));
                    self.available.pop_front();
                }
            }
        }
        false
    }

    /// Report the guest's link state as `status` gives it.
    pub fn status(&mut self, status: Option<(bool, stela_soc::machine::UsbNetworkCounters)>) {
        let up = status.is_some_and(|(up, _)| up);
        if self.up != Some(up) {
            self.up = Some(up);
            let counters = status.map(|(_, counters)| counters).unwrap_or_default();
            output::arguments(format_args!(
                "GUEST NETWORK link {} {counters:?}\n",
                if up { "up" } else { "down" }
            ));
        }
    }

    /// Read frames from the host adapter while the guest's receive queue
    /// had `room` and nothing read earlier still waits, without the devices'
    /// lock: the adapter's MMIO and DMA can wait milliseconds on the host.
    pub fn receive(&mut self, room: bool) {
        if !room || !self.received.is_empty() {
            return;
        }
        while self.received.len() < RECEIVE_BURST {
            let Some(link) = self.link.as_mut() else {
                break;
            };
            match link.receive() {
                Ok(Some(frame)) => self.received.push_back(frame),
                Ok(None) => break,
                Err(error) => {
                    self.refuse(error.into());
                    break;
                }
            }
        }
    }

    /// Give the guest the frames read so far while its receive queue has
    /// room. The guest's queues are the only bound: the rest wait here and
    /// then in the host adapter's ring, which drops on overflow as the
    /// hardware does.
    pub fn deliver(&mut self, devices: &mut AppleDevices) {
        while !self.received.is_empty() && devices.network_room() {
            if let Some(frame) = self.received.pop_front() {
                // Without an attached adapter the frame has nowhere to go.
                let _ = devices.network_deliver_owned(frame);
            }
        }
    }

    /// The frames the guest queued, taken from its devices, for
    /// [`Self::transmit`] to send without them.
    pub fn queued(devices: &mut AppleDevices) -> Vec<Vec<u8>> {
        core::iter::from_fn(|| devices.network_take()).collect()
    }

    /// Send in queue order and return the last frame's reusable storage.
    pub fn transmit(&mut self, frames: Vec<Vec<u8>>) -> Option<Vec<u8>> {
        let mut reusable = None;
        for frame in frames {
            let Some(link) = self.link.as_mut() else {
                return reusable;
            };
            if let Err(error) = link.transmit(&frame) {
                self.refuse(error.into());
            }
            reusable = Some(frame);
        }
        reusable
    }

    fn refuse(&mut self, error: NetError) {
        if !core::mem::replace(&mut self.refused, true) {
            output::arguments(format_args!("GUEST NETWORK refused: {error}\n"));
        }
    }
}
