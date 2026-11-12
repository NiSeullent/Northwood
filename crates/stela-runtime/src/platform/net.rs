//! Dual-stack HTTP and TLS transport over kept native Ethernet adapters.
//!
//! smoltcp owns ARP, IPv4/IPv6, DHCPv4, DNS and TCP. SLAAC/RDNSS admission
//! includes duplicate-address detection; the URL-aware request layer keeps
//! redirect policy and range validators. TLS uses the shared protocol provider
//! and requires owned entropy and a UTC-correlated continuous clock.
//! Missing native capabilities refuse HTTPS; they cannot select plaintext.
//! Stateful DHCPv6 is unsupported. Raw guest relay bytes stay opaque.
use alloc::string::String;
use core::fmt;
use stela_drivers::net::LinkError;

pub mod http;
pub mod ranges;
pub mod stack;
pub mod tls;

pub use smoltcp::wire::Ipv4Address as Ipv4;
pub use stack::{
    Config, HttpPost, HttpRequest, HttpTransfer, SecureTransfer, Stack, TlsUnavailable, Transfer,
};

/// Why a network operation failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NetError {
    /// The link's exact PCI access/capability failure.
    /// This is terminal for automatic transport retry, including uncertain writes.
    Pci(stela_drivers::block::BlockError),
    /// The link adapter failed or refused a frame.
    Link(String),
    /// The owned transport clock refused; this stack cannot be reused.
    Clock(ClockError),
    /// The link reports no carrier.
    NoCarrier,
    /// IPv6 duplicate-address detection found another owner.
    DuplicateAddress(core::net::Ipv6Addr),
    /// No DHCP server answered.
    Dhcp(&'static str),
    /// The resolver failed: no server, no answer, or an error code.
    Dns(String),
    /// The peer refused or reset the connection.
    Reset,
    /// Something did not finish before its deadline.
    Timeout(&'static str),
    /// The HTTP response was malformed or not one this client accepts.
    Http(String),
    /// The TLS engine refused the peer, record or required time/entropy input.
    Tls(String),
    /// A connection ended inside an otherwise valid framed body or record.
    Truncated(&'static str),
    TlsUnavailable(TlsUnavailable),
    Url(http::UrlError),
    Redirect(http::RedirectError),
    ProxyAuthentication,
    NetworkAuthentication,
    /// The server answered with a non-success HTTP status.
    Status(u16),
    /// A request the stack cannot express (bad URL, oversized message).
    Invalid(&'static str),
}

impl NetError {
    /// Retry only a transient transport/server failure. Identity, certificate,
    /// capability, framing and redirect policy failures need a changed input.
    pub fn retryable(&self) -> bool {
        matches!(
            self,
            Self::Link(_)
                | Self::NoCarrier
                | Self::Dhcp(_)
                | Self::Dns(_)
                | Self::Reset
                | Self::Timeout(_)
                | Self::Truncated(_)
                | Self::Status(408 | 429 | 500 | 502 | 503 | 504)
        )
    }
}

/// Attempts per request, as the setup CLI makes them: with waits of 1, 2, 4
/// and 8 s between them, a network that drops for about 15 s costs a pause.
pub const ATTEMPTS: u32 = 5;

/// One request's bounded retries on a caller's millisecond clock. It never
/// waits: the caller asks [`Backoff::ready`] on each service step, so no
/// processor stalls while the next attempt is due.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Backoff {
    attempt: u32,
    not_before_ms: u64,
}

impl Default for Backoff {
    fn default() -> Self {
        Self::new()
    }
}

impl Backoff {
    pub const fn new() -> Self {
        Self {
            attempt: 1,
            not_before_ms: 0,
        }
    }

    /// The attempt that runs next, from 1.
    pub fn attempt(&self) -> u32 {
        self.attempt
    }

    /// Whether the next attempt may start at `now_ms`.
    pub fn ready(&self, now_ms: u64) -> bool {
        now_ms >= self.not_before_ms
    }

    /// The attempt failed with `error` at `now_ms`: the wait in milliseconds
    /// before the next one, or `None` when `error` is not transient or the
    /// attempts are spent.
    pub fn failed(&mut self, error: &NetError, now_ms: u64) -> Option<u64> {
        if !error.retryable() || self.attempt >= ATTEMPTS {
            return None;
        }
        let wait = 1000 << (self.attempt - 1);
        self.attempt += 1;
        self.not_before_ms = now_ms.saturating_add(wait);
        Some(wait)
    }
}

impl fmt::Display for NetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pci(why) => write!(f, "network PCI: {why}"),
            Self::Link(why) => write!(f, "network link: {why}"),
            Self::Clock(why) => write!(f, "network clock: {why}"),
            Self::NoCarrier => f.write_str("network link has no carrier"),
            Self::DuplicateAddress(address) => {
                write!(f, "IPv6 address {address} is already in use")
            }
            Self::Dhcp(why) => write!(f, "DHCP: {why}"),
            Self::Dns(why) => write!(f, "DNS: {why}"),
            Self::Reset => f.write_str("connection refused or reset by peer"),
            Self::Timeout(what) => write!(f, "timed out: {what}"),
            Self::Http(why) => write!(f, "HTTP: {why}"),
            Self::Tls(why) => write!(f, "TLS: {why}"),
            Self::Truncated(what)=>write!(f,"connection ended inside {what}"),
            Self::TlsUnavailable(why)=>write!(f,"TLS unavailable: {why}"),
            Self::Url(why)=>write!(f,"URL: {why}"),
            Self::Redirect(why)=>write!(f,"redirect: {why}"),
            Self::ProxyAuthentication=>f.write_str("the network requires proxy authentication (HTTP 407); no proxy credentials are configured"),
            Self::NetworkAuthentication=>f.write_str("the network requires authentication (HTTP 511); complete network login before installing"),
            Self::Status(code) => write!(f, "HTTP status {code}"),
            Self::Invalid(why) => write!(f, "invalid request: {why}"),
        }
    }
}

impl From<LinkError> for NetError {
    fn from(error: LinkError) -> Self {
        match error {
            LinkError::Pci(error) => Self::Pci(error),
            LinkError::Adapter(why) => Self::Link(why),
            LinkError::Invalid(why) => Self::Invalid(why),
            LinkError::NoCarrier => Self::NoCarrier,
        }
    }
}

/// A failed time observation never becomes a usable timestamp.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClockError {
    Unavailable,
    Native(stela_drivers::platform::work::WorkError),
    Calibration(stela_drivers::ClockError),
    Regressed { previous: u64, current: u64 },
    OutOfRange(u64),
}
impl fmt::Display for ClockError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unavailable => f.write_str("continuous clock unavailable"),
            Self::Native(error) => write!(f, "native source refused: {error:?}"),
            Self::Calibration(error) => write!(f, "TSC calibration refused: {error:?}"),
            Self::Regressed { previous, current } => write!(
                f,
                "clock regressed from {previous} to {current} microseconds"
            ),
            Self::OutOfRange(value) => {
                write!(f, "clock value {value} exceeds the transport time range")
            }
        }
    }
}
impl core::error::Error for ClockError {}
/// Microseconds from one monotonic origin. Native implementations
/// preserve source/context errors instead of substituting a timestamp.
pub trait Clock {
    fn now_us(&self) -> Result<u64, ClockError>;
    /// Sleep until an interrupt or `until_us`; no-sleep clocks return at once.
    fn sleep(&self, until_us: u64) -> Result<(), ClockError> {
        let _ = until_us;
        Ok(())
    }
}
impl Clock for stela_drivers::Clock {
    fn now_us(&self) -> Result<u64, ClockError> {
        self.try_now_us().map_err(ClockError::Calibration)
    }
}

#[cfg(test)]
mod pci_error_tests {
    use super::*;
    use stela_drivers::{
        block::BlockError,
        bus::pci::{ConfigError, ConfigFailure},
    };

    #[test]
    fn link_pci_refusal_retains_exact_reason_and_effect_without_transport_retry() {
        for error in [
            ConfigError::before_effect(ConfigFailure::ForeignGrant),
            ConfigError::write_may_have_applied(
                4,
                2,
                ConfigFailure::Mapping(stela_drivers::bus::mmio::owned::Error::Unavailable),
            ),
        ] {
            let cause = BlockError::Config(error);
            let converted = NetError::from(LinkError::Pci(cause.clone()));
            assert_eq!(converted, NetError::Pci(cause));
            assert!(!converted.retryable());
            let NetError::Pci(BlockError::Config(actual)) = converted else {
                panic!("typed PCI failure lost");
            };
            assert_eq!(actual.reason, error.reason);
            assert_eq!(actual.effect, error.effect);
        }
    }

    #[test]
    fn legacy_link_carrier_and_invalid_conversions_keep_their_retry_policy() {
        let adapter = NetError::from(LinkError::Adapter(String::from("modeled link error")));
        assert_eq!(adapter, NetError::Link(String::from("modeled link error")));
        assert!(adapter.retryable());
        assert_eq!(NetError::from(LinkError::NoCarrier), NetError::NoCarrier);
        assert!(NetError::NoCarrier.retryable());
        assert_eq!(
            NetError::from(LinkError::Invalid("frame")),
            NetError::Invalid("frame")
        );
        assert!(!NetError::Invalid("frame").retryable());
    }
}

#[cfg(test)]
mod backoff_tests {
    use super::*;
    use alloc::vec::Vec;

    #[test]
    fn a_transient_failure_is_retried_four_times_with_doubling_waits() {
        let mut backoff = Backoff::new();
        assert!(backoff.ready(0));
        let mut now = 100;
        let mut waits = Vec::new();
        while let Some(wait) = backoff.failed(&NetError::Reset, now) {
            assert!(!backoff.ready(now + wait - 1));
            assert!(backoff.ready(now + wait));
            waits.push(wait);
            now += wait;
        }
        assert_eq!(waits, [1000, 2000, 4000, 8000]);
        assert_eq!(backoff.attempt(), ATTEMPTS);
    }

    #[test]
    fn a_failure_that_needs_another_input_is_not_retried() {
        let mut backoff = Backoff::new();
        assert_eq!(backoff.failed(&NetError::Status(400), 0), None);
        assert_eq!(backoff.failed(&NetError::Invalid("url"), 0), None);
        assert_eq!(backoff.attempt(), 1);
        assert_eq!(backoff.failed(&NetError::Status(503), 0), Some(1000));
    }
}
