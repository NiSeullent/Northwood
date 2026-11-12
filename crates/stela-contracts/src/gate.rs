//! A call gate: native code a guest thread enters through a device access,
//! reaching that thread's memory as its own unprivileged loads and stores
//! would.
use alloc::{sync::Arc, vec::Vec};
use core::num::NonZeroU64;

/// Why the caller's memory refused an access.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallerFault {
    /// The address is unmapped, or the caller may not access it so.
    Denied { address: u64 },
    /// The address maps something other than ordinary RAM.
    NotRam { address: u64 },
    /// The caller's translation uses a configuration the model does not
    /// implement.
    Unsupported(&'static str),
}

/// The address space of the thread whose device access is being served,
/// checked with EL0 permissions in its current translation regime.
pub trait CallerMemory {
    fn read(&mut self, address: u64, buffer: &mut [u8]) -> Result<(), CallerFault>;
    /// Writes nothing unless every byte is writable RAM.
    fn write(&mut self, address: u64, data: &[u8]) -> Result<(), CallerFault>;
}

/// Wired guest RAM: contiguous guest physical memory, in whole pages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryRange {
    pub address: u64,
    pub bytes: u64,
}

/// The native service a gate enters, by connection. Calls run on the PEs
/// that make them, outside the board's lock, so one call's length is no
/// other PE's; the service orders them itself.
pub trait GateService: Send + Sync {
    fn connect(&self) -> Result<NonZeroU64, &'static str>;
    /// Releases what the connection holds; in-flight work completes.
    fn disconnect(&self, connection: NonZeroU64);
    /// Runs the call whose record is at `record` in `caller`, completing
    /// the record; fails only when the record is out of reach.
    fn serve(
        &self,
        connection: NonZeroU64,
        record: u64,
        caller: &mut dyn CallerMemory,
    ) -> Result<(), CallerFault>;
    /// Gives `connection` the guest RAM of `ranges`, which the kernel keeps
    /// wired until [`Self::retired`] answers `token`. Answers its handle.
    fn import(
        &self,
        connection: NonZeroU64,
        ranges: &[MemoryRange],
        token: NonZeroU64,
    ) -> Result<NonZeroU64, &'static str>;
    /// The token of imported memory the GPU no longer reaches.
    fn retired(&self) -> Option<NonZeroU64>;
    /// Whether the service's owner closed it and its GPU acknowledged idle,
    /// which is what lets a gate release what its connections hold.
    fn quiesced(&self) -> bool;
    /// Whether the GPU completed work since this last answered true. Under
    /// the board's lock, so it answers at once.
    fn signaled(&self) -> bool;
    /// Whether submitted work is not yet seen complete: the gate then
    /// advances again soon, so that [`Self::signaled`] notices a completion
    /// the service reads from the GPU's own writes, on whichever PE comes.
    fn awaits_completion(&self) -> bool {
        false
    }
    /// Told what the kext found of the Metal plugin it installs.
    fn bundle(&self, report: BundleReport);
    /// Told that the gate dropped a request its PE's store made before the
    /// store's step took it, and why: the caller found its record unserved.
    fn dropped(&self, _reason: &'static str) {}
}

/// What the kext found of the Metal plugin `/Library/GPUBundles` holds,
/// against the one it carries, whose code the boot's trust cache lists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BundleReport {
    /// Every file was the kext's.
    Current,
    /// The kext rewrote the files that differed.
    Updated,
    /// The plugin's directories are absent, and the kernel cannot create
    /// directories for a kext.
    Missing,
    /// Reading or writing a file failed.
    Failed,
}

impl BundleReport {
    /// The report a store of `value` to [`window::BUNDLE`] makes.
    pub fn from_raw(value: u64) -> Option<Self> {
        match value {
            window::BUNDLE_CURRENT => Some(Self::Current),
            window::BUNDLE_UPDATED => Some(Self::Updated),
            window::BUNDLE_MISSING => Some(Self::Missing),
            window::BUNDLE_FAILED => Some(Self::Failed),
            _ => None,
        }
    }
}

/// What a store to a gate window asks of the service.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GateRequest {
    Open,
    Close(NonZeroU64),
    Serve {
        gate: usize,
        connection: NonZeroU64,
        record: u64,
    },
    Import {
        connection: NonZeroU64,
        ranges: Vec<MemoryRange>,
        token: NonZeroU64,
    },
    Retire,
    Bundle(BundleReport),
}

/// A request's result, for the gate at `window` of the board to record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GateOutcome {
    pub window: usize,
    pub answer: GateAnswer,
}

/// What a request came to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateAnswer {
    Opened(Option<NonZeroU64>),
    Closed,
    Served {
        gate: usize,
        connection: NonZeroU64,
        /// The call record the store named, so the gate keeps its fault for
        /// that call alone.
        record: u64,
        fault: Option<CallerFault>,
    },
    Imported(Option<NonZeroU64>),
    Retired(Option<NonZeroU64>),
    Reported,
}

/// A store's request, taken from the gate to run in that store's own step
/// outside the board's lock.
pub struct GateCall {
    window: usize,
    service: Arc<dyn GateService>,
    request: GateRequest,
}

impl GateCall {
    pub fn new(window: usize, service: Arc<dyn GateService>, request: GateRequest) -> Self {
        Self {
            window,
            service,
            request,
        }
    }

    pub fn run(self, caller: &mut dyn CallerMemory) -> GateOutcome {
        let answer = match self.request {
            GateRequest::Open => GateAnswer::Opened(self.service.connect().ok()),
            GateRequest::Close(connection) => {
                self.service.disconnect(connection);
                GateAnswer::Closed
            }
            GateRequest::Serve {
                gate,
                connection,
                record,
            } => GateAnswer::Served {
                gate,
                connection,
                record,
                fault: self.service.serve(connection, record, caller).err(),
            },
            GateRequest::Import {
                connection,
                ranges,
                token,
            } => GateAnswer::Imported(self.service.import(connection, &ranges, token).ok()),
            GateRequest::Retire => GateAnswer::Retired(self.service.retired()),
            GateRequest::Bundle(report) => {
                self.service.bundle(report);
                GateAnswer::Reported
            }
        };
        GateOutcome {
            window: self.window,
            answer,
        }
    }
}

/// Defines a block of 64-bit layout constants and [`LAYOUT`] listing them,
/// from which the kext's header is generated.
macro_rules! layout {
    ($($(#[doc = $doc:literal])* $name:ident = $value:expr;)*) => {
        $($(#[doc = $doc])* pub const $name: u64 = $value;)*
        /// Every constant above, by name.
        pub const LAYOUT: &[(&str, u64)] = &[$((stringify!($name), $name)),*];
    };
}

/// A StelaGPU gate window at `/arm-io/stela-gpu<n>`, as the runtime models
/// it and the StelaGPU kext drives it.
///
/// Page 0 is the kext's control page. Page `n` (1 to [`CONNECTIONS`]) is
/// connection `n`'s gate, which the kext maps into the one task it opened
/// the connection for. Every register is 64-bit.
pub mod window {
    layout! {
        /// The guest's page: the unit one task maps.
        PAGE = 0x4000;
        /// Connections the window has gates for.
        CONNECTIONS = 255;
        /// The window: the control page and every gate.
        BYTES = PAGE * (CONNECTIONS + 1);

        /// Control: a store opens a connection.
        OPEN = 0x00;
        /// Control: the gate page of the last open, 0 when it failed.
        OPENED = 0x08;
        /// Control: a store of a gate page closes its connection.
        CLOSE = 0x10;
        /// Control: the number of gate pages.
        COUNT = 0x18;
        /// Control: 1 while completed work is unacknowledged.
        STATUS = 0x20;
        /// Control: a store acknowledges completed work.
        ACK = 0x28;
        /// Control: the guest physical address of an import's next range.
        IMPORT_ADDRESS = 0x30;
        /// Control: a store of that range's length adds it to the import.
        IMPORT_BYTES = 0x38;
        /// Control: the nonzero token `RETIRED` will name the import by.
        IMPORT_TOKEN = 0x40;
        /// Low bit of the kext's aligned descriptor token. Set only when
        /// its descriptor was wired for both reads and writes. The full
        /// token, including this bit, must be returned on retirement.
        IMPORT_TOKEN_WRITABLE = 0x1;
        /// Control: a store of a gate page imports the ranges added for its
        /// connection. Only the kext reaches this page, so only the kernel
        /// names physical memory.
        IMPORT = 0x48;
        /// Control: the handle of the last import, 0 when it failed.
        IMPORTED = 0x50;
        /// Control: a store looks for retired memory.
        RETIRE = 0x58;
        /// Control: the token that found, 0 for none.
        RETIRED = 0x60;
        /// Control: a store reports what the kext found of the Metal
        /// plugin, one of the `BUNDLE_` answers, before it publishes the
        /// accelerator.
        BUNDLE = 0x68;

        /// Gate: a store names a call record.
        CALL = 0x00;
        /// Gate: why the last call's record was out of reach, 0 for none.
        FAULT = 0x08;

        /// `FAULT`: the record is unmapped or not the caller's to access.
        FAULT_DENIED = 1;
        /// `FAULT`: the record is not RAM.
        FAULT_NOT_RAM = 2;
        /// `FAULT`: the caller's translation is one the model lacks.
        FAULT_UNSUPPORTED = 3;

        /// `BUNDLE`: every file of the plugin was the kext's.
        BUNDLE_CURRENT = 1;
        /// `BUNDLE`: the kext rewrote the files that differed.
        BUNDLE_UPDATED = 2;
        /// `BUNDLE`: the plugin's directories are absent.
        BUNDLE_MISSING = 3;
        /// `BUNDLE`: reading or writing a file of the plugin failed.
        BUNDLE_FAILED = 4;
    }
}
