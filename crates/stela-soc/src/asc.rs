//! The ASC coprocessor wrapper (`iop,ascwrap-v4`; ADT `/arm-io/<iop>` `reg[0]`):
//! the CPU control of an RTKit coprocessor and its AP mailbox.
//!
//! The coprocessor's own software is emulated natively behind [`Firmware`],
//! at its AP-facing protocol; the wrapper only carries 128-bit messages and
//! starts it.
//!
//! Documented by Linux `drivers/soc/apple/rtkit-helper.c` (`APPLE_ASC_CPU_CONTROL`
//! 0x44, `RUN` bit 4) and `drivers/mailbox/apple-mailbox.c` (the `asc` variant,
//! 0x8000 into the wrapper): A2I (AP to coprocessor) control 0x110, send
//! 0x800 and 0x808; I2A control 0x114, receive 0x830 and 0x838; control bit 16
//! FULL and bit 17 EMPTY; a message is two 64-bit words, the payload and the
//! endpoint in the low byte of the second, sent by the store of 0x808 and
//! consumed by the load of 0x838. The 26A434 J473AP `AppleASCWrapV4` accesses
//! exactly these registers (`_inbox` stores 0x8800 and 0x8808, `_outbox` loads
//! 0x8830 and 0x8838, `_getInboxEmpty`/`_getInboxFull` read 0x8110 bits 17/16,
//! `_runCPU` sets or clears 0x44 bit 4 and on stop also clears bit 5).
//! `getMailboxBulk` takes the number of waiting messages from I2A control
//! bits 23:20. Each received second word also snapshots the pre-pop count in
//! bits 55:52 (`getMailboxBulk`, kernel file 0x1badaa8); the driver uses
//! that snapshot to stop a bulk read after its last message. The full and
//! empty queries treat bits 19:18 as errors,
//! which the model never raises.
//!
//! The ADT lists four interrupts in the order of their AIC numbers rotated by
//! one pair (J473AP PMP: 0x275, 0x274, 0x277, 0x276); `AppleA7IOP` takes
//! index 1 as inbox empty and index 2 as outbox not empty. With Linux's order
//! of a wrapper's four consecutive lines (send empty, send not empty, receive
//! empty, receive not empty), index 0 is inbox not empty and index 3 outbox
//! empty. Every line is a level.
//!
//! Inferred, not documented:
//! - the native firmware takes each message as it is sent, so the inbox is
//!   always empty, which is the fastest permitted completion; a message sent
//!   while the coprocessor is stopped is refused, since nothing would drain it;
//! - `AppleASCWrapV4::_enableOutbox` sets 0x8114 bit 0 and leaves interrupt
//!   enables to its no-op `_enableOutboxInterrupt`, so bit 0 enables the
//!   outbox itself: the coprocessor's messages wait in it, invisible, until
//!   the AP enables it. 0x8110 bit 0 (`_getKICInboxEnabled`) is held;
//! - a load of an empty outbox is refused (the hardware underflows, which
//!   `AppleA7IOP::_outboxUnderflowCheck` treats as an error);
//! - clearing RUN halts native firmware rather than resetting it. A firmware
//!   must admit kept stop/resume explicitly; unsupported reset and
//!   power-loss flows still refuse. Mailbox controls, staged payload and
//!   queued responses survive a halt.
use crate::Result;
use alloc::boxed::Box;
use alloc::collections::{BTreeSet, VecDeque};
use alloc::string::String;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::any::Any;
use stela_contracts::Refusal;
use stela_contracts::dma::Port;
use stela_contracts::refuse;
use {
    stela_contracts::bus::Access, stela_contracts::bus::Direction, stela_contracts::bus::Lifecycle,
    stela_contracts::bus::LifecycleState, stela_contracts::bus::MmioDevice,
    stela_contracts::bus::MmioError, stela_contracts::bus::Transition, stela_contracts::bus::Width,
    stela_contracts::bus::refuse,
};
use {stela_contracts::interrupt::AicInput, stela_contracts::interrupt::Wire};

pub mod entry;
pub use entry::{Auxiliary, EntryFetch, EntryHandoff};

const NAME: &str = "asc";

const CPU_CONTROL: u64 = 0x44;
const RUN: u64 = 1 << 4;
/// Cleared with RUN on stop; its meaning is not documented, so it is held.
const CPU_CONTROL_HELD: u64 = 1 << 5;
const IDLE_STATUS: u64 = 0x8000;
const A2I_CONTROL: u64 = 0x8110;
const I2A_CONTROL: u64 = 0x8114;
const A2I_SEND0: u64 = 0x8800;
const A2I_SEND1: u64 = 0x8808;
const I2A_RECV0: u64 = 0x8830;
const I2A_RECV1: u64 = 0x8838;
const CONTROL_ENABLE: u64 = 1;
const CONTROL_FULL: u64 = 1 << 16;
const CONTROL_EMPTY: u64 = 1 << 17;
/// The I2A control's count of waiting messages, bits 23:20.
const OCCUPANCY_SHIFT: u32 = 20;
const OCCUPANCY_MAX: usize = 0xf;
const CONTROL_STATUS: u64 = CONTROL_FULL | CONTROL_EMPTY | (0xf << OCCUPANCY_SHIFT);
/// The second message word holds the endpoint in its low byte.
const ENDPOINT: u64 = 0xff;

/// The wrapper's register span, through the mailbox.
pub const WRAPPER_BYTES: u64 = 0x8840;

/// One mailbox message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Message {
    pub endpoint: u8,
    pub payload: u64,
}

/// A machine transition a coprocessor carries out for the AP, such as the
/// SMC's restart and standby commands: the device and its reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransitionRequest {
    pub device: &'static str,
    pub transition: Transition,
    pub reason: &'static str,
}

/// The coprocessor's software, emulated natively.
pub trait Firmware: Any {
    /// Fetch through this backend's live allocation and selected IOP mapping
    /// owners. Success requires known instruction bytes, resolved PA,
    /// current mapping/ownership and applicable fetch access. A source hash,
    /// numerical address relation or native protocol peer is not that proof.
    /// ASC checks the result at handoff and before every start/resume; the
    /// same backend must keep the owners through its execution/use boundary.
    fn fetch_entry(&mut self, _iop: u64) -> Result<EntryFetch> {
        Err(refuse!(
            "original coprocessor instruction fetch ownership is not modeled"
        ))
    }

    /// Validate loss of native volatile firmware state. Physical reset is
    /// distinct from an RTKit sleep request: no mailbox ACK is generated.
    fn check_reset(&self) -> Result<()> {
        Err(refuse!("coprocessor volatile-state reset is not modeled"))
    }
    /// Cold reset after `check_reset` succeeded, without starting execution.
    fn reset(&mut self) -> Result<()> {
        Err(refuse!("coprocessor volatile-state reset is not modeled"))
    }
    /// The coprocessor starts running; messages it sends go to `outbox`.
    /// Start/resume/receive output is provisional until Ok; a refusal must
    /// keep retryable native state and must not commit its success ACK.
    fn start(&mut self, outbox: &mut Vec<Message>) -> Result<()>;
    /// Halt CPU execution while keeping firmware state. Validate before
    /// changing state: a refusal leaves RUN and the mailbox untouched.
    /// Autonomous hardware work must be settled or kept by its owner.
    fn stop(&mut self) -> Result<()> {
        Err(refuse!("firmware retained CPU stop is not modeled"))
    }
    /// Resume after an accepted halt, preserving protocol state.
    /// This is distinct from cold `start`; refusal leaves the CPU halted.
    fn resume(&mut self, _outbox: &mut Vec<Message>) -> Result<()> {
        Err(refuse!("firmware retained CPU resume is not modeled"))
    }
    /// The AP sent `message`.
    fn receive(&mut self, message: Message, outbox: &mut Vec<Message>) -> Result<()>;
    /// The coprocessor has bus-master work queued (DMA it began when it
    /// received a message), to run before the AP's next mailbox access. The
    /// default coprocessor masters no memory.
    fn dma_pending(&self) -> bool {
        false
    }
    /// Perform that queued DMA through `port`, pushing committed mailbox
    /// messages to `outbox`. Earlier output remains committed if a later
    /// transfer fails; keep unfinished native work for retry.
    fn dma(&mut self, _port: &mut Port<'_, '_>, _outbox: &mut Vec<Message>) -> Result<()> {
        Ok(())
    }
    /// Committed asynchronous messages, collected at each device advance.
    /// Output appended before a later error remains visible; keep only
    /// unfinished native work so retry never duplicates an earlier event.
    fn poll(&mut self, _outbox: &mut Vec<Message>) -> Result<()> {
        Ok(())
    }
    /// A machine transition the AP's last message asked for, once.
    fn take_transition(&mut self) -> Option<TransitionRequest> {
        None
    }
    /// Requests the firmware answered degraded on its own transports (not
    /// a refused mailbox message, which the wrapper reports), each reported
    /// once, since the last call.
    fn take_unmodeled(&mut self) -> Vec<String> {
        Vec::new()
    }
    /// The coprocessor's own error reply to `message`, a request it does
    /// not implement (`reason`). `None`, the default, when its protocol has
    /// no error reply for that request: the request is then left
    /// unanswered, so the client's own timeout path runs.
    fn error_reply(&mut self, _message: Message, _reason: &Refusal) -> Option<Vec<Message>> {
        None
    }
}

/// The wrapper's interrupt lines, in ADT `interrupts` order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Lines {
    pub inbox_not_empty: Option<AicInput>,
    pub inbox_empty: Option<AicInput>,
    pub outbox_not_empty: Option<AicInput>,
    pub outbox_empty: Option<AicInput>,
}

impl Lines {
    /// The lines of an ADT `interrupts` list of four AIC inputs.
    pub fn from_interrupts(irqs: [AicInput; 4]) -> Self {
        Self {
            inbox_not_empty: Some(irqs[0]),
            inbox_empty: Some(irqs[1]),
            outbox_not_empty: Some(irqs[2]),
            outbox_empty: Some(irqs[3]),
        }
    }
}

pub struct Asc {
    auxiliary: Auxiliary,
    firmware: Box<dyn Firmware>,
    lifecycle: LifecycleState,
    running: bool,
    /// RUN was cleared after firmware admitted a CPU halt.
    halted: bool,
    cpu_control: u64,
    a2i_control: u64,
    outbox_enabled: bool,
    send0: Option<u64>,
    /// The coprocessor's messages; the head is visible while the outbox is
    /// enabled.
    outbox: VecDeque<Message>,
    lines: Lines,
    /// Levels last reported, in [`Lines`] order.
    levels: [Option<bool>; 4],
    /// The endpoints and refusals answered degraded so far.
    reported: BTreeSet<(u8, &'static str)>,
    /// Reports not yet taken ([`Self::take_unmodeled`]).
    unmodeled: Vec<String>,
}

impl Asc {
    /// A wrapper whose coprocessor `running` already (started by the boot
    /// firmware) or waits for the AP to set RUN.
    pub fn new(mut firmware: Box<dyn Firmware>, running: bool, lines: Lines) -> Result<Self> {
        let mut outbox = Vec::new();
        if running {
            firmware.start(&mut outbox)?;
        }
        Ok(Self {
            auxiliary: Auxiliary::unknown(running),
            firmware,
            lifecycle: LifecycleState {
                powered: true,
                clocked: true,
                quiesced: false,
                reset: false,
                retention: false,
            },
            running,
            halted: false,
            cpu_control: if running { RUN } else { 0 },
            a2i_control: 0,
            outbox_enabled: false,
            send0: None,
            outbox: outbox.into(),
            lines,
            levels: [None; 4],
            reported: BTreeSet::new(),
            unmodeled: Vec::new(),
        })
    }

    /// The requests answered degraded for the first time since the last
    /// call.
    pub fn take_unmodeled(&mut self) -> Vec<String> {
        let mut lines = core::mem::take(&mut self.unmodeled);
        lines.extend(self.firmware.take_unmodeled());
        lines
    }

    /// Emulated boot-chain handoff: native firmware is running and both
    /// mailboxes are enabled before the AP attaches. RTBuddy's preloaded
    /// path waits for HELLO without reinitializing those controls.
    pub fn booted(firmware: Box<dyn Firmware>, lines: Lines) -> Result<Self> {
        let mut asc = Self::new(firmware, true, lines)?;
        asc.a2i_control = CONTROL_ENABLE;
        asc.outbox_enabled = true;
        Ok(asc)
    }

    /// The ASCv4 provider's second register aperture. It shares this owner's
    /// lifecycle; do not bind a second PMGR lifecycle owner to the handle.
    /// Native protocol firmware does not imply an ARM entry-register value.
    #[cfg(test)]
    pub fn auxiliary(&self) -> Auxiliary {
        self.auxiliary.clone()
    }

    /// Check the backend's current owned entry fetch, then establish
    /// the independent register handoff. The supplied mapping describes all
    /// three address domains and the expected original instruction.
    /// This never loads or executes ARM firmware, or derives an entry from a
    /// native Rust implementation address.
    #[cfg(test)]
    pub fn establish_entry(&mut self, handoff: EntryHandoff) -> Result<()> {
        self.check_entry_fetch(handoff)?;
        self.auxiliary.establish(handoff)
    }

    fn check_entry_fetch(&mut self, handoff: EntryHandoff) -> Result<()> {
        let fetched = self.firmware.fetch_entry(handoff.iop_entry())?;
        if fetched.physical != handoff.physical_entry() {
            return Err(refuse!(
                "ASC actual entry translation disagrees with owned physical backing"
            ));
        }
        if fetched.instruction != handoff.instruction() {
            return Err(refuse!(
                "ASC fetched entry instruction disagrees with original source"
            ));
        }
        Ok(())
    }

    /// Read-only mailbox state for boot diagnostics.
    pub fn diagnostic(&self) -> alloc::string::String {
        alloc::format!(
            "run={} a2i={:#x} i2a_enabled={} send={:x?} outbox={:x?} levels={:?}",
            self.running,
            self.a2i_control,
            self.outbox_enabled,
            self.send0,
            self.outbox,
            self.levels
        )
    }

    /// A native coprocessor's typed state, for its DMA boundary. The wrapper
    /// still owns all mailbox and interrupt effects; firmware cannot replace it.
    pub fn firmware<F: Firmware>(&self) -> Option<&F> {
        (&*self.firmware as &dyn Any).downcast_ref()
    }

    /// Whether the coprocessor has bus-master work to run this step.
    pub fn dma_pending(&self) -> bool {
        self.execution_enabled() && self.firmware.dma_pending()
    }

    /// Run the coprocessor's queued DMA through `port`; any messages it
    /// produces enter the outbox and raise the mailbox interrupt, exactly as
    /// a received message's replies do.
    pub fn service_dma(&mut self, port: &mut Port<'_, '_>) -> Result<()> {
        if !self.execution_enabled() {
            return Err(refuse!("mailbox DMA on a stopped coprocessor"));
        }
        let mut produced = Vec::new();
        let result = self.firmware.dma(port, &mut produced);
        // Firmware may have committed earlier DMA writes before a later
        // transfer faults. Its already-produced notifications are equally
        // committed and must remain visible even when this step fails.
        self.outbox.extend(produced);
        self.update_lines(port.access);
        result
    }

    /// The native coprocessor, for board events it reports; what it sends
    /// in response leaves through the mailbox at the next device advance.
    pub fn firmware_mut<F: Firmware>(&mut self) -> Option<&mut F> {
        (&mut *self.firmware as &mut dyn Any).downcast_mut()
    }

    fn visible(&self) -> bool {
        self.outbox_enabled && !self.outbox.is_empty()
    }

    /// CPU-driven work, including display scanout outside mailbox DMA,
    /// may run only while every physical execution gate is released.
    pub fn execution_enabled(&self) -> bool {
        self.running
            && self.lifecycle.powered
            && self.lifecycle.clocked
            && !self.lifecycle.quiesced
            && !self.lifecycle.reset
    }

    fn accessible(&self) -> Result<()> {
        if !self.lifecycle.powered {
            return Err(refuse!("ASC power is gated"));
        }
        if !self.lifecycle.clocked {
            return Err(refuse!("ASC clock is gated"));
        }
        if self.lifecycle.reset {
            return Err(refuse!("ASC reset is asserted"));
        }
        Ok(())
    }

    fn update_lines(&mut self, access: &mut Access<'_>) {
        let visible = self.visible();
        let wanted = [
            (self.lines.inbox_not_empty, false),
            (self.lines.inbox_empty, true),
            (self.lines.outbox_not_empty, visible),
            (self.lines.outbox_empty, !visible),
        ];
        for (index, (line, level)) in wanted.into_iter().enumerate() {
            if self.levels[index] != Some(level) {
                self.levels[index] = Some(level);
                if let Some(irq) = line {
                    access.interrupts.set_level(Wire::Aic(irq), level);
                }
            }
        }
    }

    fn deliver(&mut self, result: Result<Vec<Message>>) -> Result<()> {
        self.outbox.extend(result?);
        Ok(())
    }

    fn send(&mut self, endpoint_word: u64) -> Result<()> {
        if self.lifecycle.quiesced {
            return Err(refuse!("ASC inbox submission while PMGR quiesced"));
        }
        if !self.running {
            return Err(refuse!("mailbox send to a stopped coprocessor"));
        }
        if endpoint_word & !ENDPOINT != 0 {
            return Err(refuse!(
                "mailbox message bits above the endpoint are not modeled"
            ));
        }
        let payload = self
            .send0
            .ok_or(refuse!("mailbox send without its payload word"))?;
        let message = Message {
            endpoint: endpoint_word as u8,
            payload,
        };
        let mut replies = Vec::new();
        let result = match self.firmware.receive(message, &mut replies) {
            Ok(()) => Ok(replies),
            // A request the firmware does not implement gets the firmware's
            // own error reply where its protocol has one, else none, and is
            // reported once per endpoint and refusal.
            Err(reason) => {
                let error = self.firmware.error_reply(message, &reason);
                if self.reported.insert((message.endpoint, reason.what())) {
                    self.unmodeled.push(alloc::format!(
                        "endpoint {:#x} payload {:#018x}: {reason}; {}",
                        message.endpoint,
                        message.payload,
                        if error.is_some() {
                            "answered with the firmware's error reply"
                        } else {
                            "left unanswered for the client's timeout"
                        }
                    ));
                }
                Ok(error.unwrap_or_default())
            }
        };
        self.deliver(result)?;
        self.send0 = None;
        Ok(())
    }

    fn write_cpu_control(&mut self, value: u64) -> Result<()> {
        if value & !(RUN | CPU_CONTROL_HELD) != 0 {
            return Err(refuse!("ASC CPU control bit is not modeled"));
        }
        let run = value & RUN != 0;
        if run && !self.running && self.lifecycle.quiesced {
            return Err(refuse!("ASC CPU start while PMGR quiesced"));
        }
        if self.running && !run {
            self.firmware.stop()?;
            self.running = false;
            self.halted = true;
        }
        if run && !self.running {
            if let Some(handoff) = self.auxiliary.check_start()? {
                self.check_entry_fetch(handoff)?;
            }
            let mut started = Vec::new();
            let result = if self.halted {
                self.firmware.resume(&mut started)
            } else {
                self.firmware.start(&mut started)
            }
            .map(|()| started);
            self.deliver(result)?;
            self.running = true;
            self.halted = false;
        }
        self.cpu_control = value;
        self.auxiliary.set_running(self.running);
        Ok(())
    }
}

impl Lifecycle for Asc {
    fn check_lifecycle(&self, proposed: LifecycleState) -> Result<()> {
        let destructive = proposed.reset && !self.lifecycle.reset
            || !proposed.powered && self.lifecycle.powered && !proposed.retention;
        if destructive {
            // A posted word or committed notification still belongs to its
            // sender. Do not erase it or complete the PMGR request first.
            if self.send0.is_some() || !self.outbox.is_empty() {
                return Err(refuse!(
                    "ASC reset or power loss with owned mailbox messages"
                ));
            }
            self.firmware.check_reset()?;
        }
        Ok(())
    }

    fn set_lifecycle(&mut self, proposed: LifecycleState, access: &mut Access<'_>) {
        let destructive = proposed.reset && !self.lifecycle.reset
            || !proposed.powered && self.lifecycle.powered && !proposed.retention;
        if destructive {
            self.firmware.reset().expect("ASC reset was preflighted");
            self.running = false;
            self.halted = false;
            self.cpu_control = 0;
            self.a2i_control = 0;
            self.outbox_enabled = false;
        }
        self.lifecycle = proposed;
        self.auxiliary.set_lifecycle(proposed);
        // Gates freeze execution; neither they nor gate release
        // synthesize a firmware sleep/wake message or alter FIFO ownership.
        self.update_lines(access);
    }
}

impl MmioDevice for Asc {
    fn lifecycle(&mut self) -> Option<&mut dyn Lifecycle> {
        Some(self)
    }
    fn name(&self) -> &'static str {
        NAME
    }

    fn read(
        &mut self,
        offset: u64,
        width: Width,
        access: &mut Access<'_>,
    ) -> Result<u64, MmioError> {
        let fail = |reason: Refusal| refuse(NAME, offset, width, Direction::Read, reason);
        self.accessible().map_err(fail)?;
        let value = match (offset, width) {
            (CPU_CONTROL, Width::Word) => self.cpu_control,
            // AppleASCWrapV4::_isIdle (kernel file 0x1bad724) consumes
            // (IDLE_STATUS & 3) != 0. Emulated die choice: bit 0 denotes
            // quiescent native mailboxes, with no partially posted message
            // or pending response. Unknown writes remain refused.
            (IDLE_STATUS, Width::Word) => u64::from(self.send0.is_none() && self.outbox.is_empty()),
            (A2I_CONTROL, Width::Word) => self.a2i_control | CONTROL_EMPTY,
            (I2A_CONTROL, Width::Word) => {
                let waiting = if self.visible() { self.outbox.len() } else { 0 };
                u64::from(self.outbox_enabled)
                    | (waiting.min(OCCUPANCY_MAX) as u64) << OCCUPANCY_SHIFT
                    | if waiting == 0 { CONTROL_EMPTY } else { 0 }
            }
            (I2A_RECV0, Width::Double) => self
                .visible()
                .then(|| self.outbox[0].payload)
                .ok_or_else(|| fail(refuse!("load of an empty outbox")))?,
            (I2A_RECV1, Width::Double) => {
                if !self.visible() {
                    return Err(fail(refuse!("load of an empty outbox")));
                }
                let waiting = self.outbox.len().min(OCCUPANCY_MAX) as u64;
                let message = self.outbox.pop_front().expect("visible outbox has a head");
                self.update_lines(access);
                u64::from(message.endpoint) | waiting << (32 + OCCUPANCY_SHIFT)
            }
            _ => return Err(fail(refuse!("ASC register without a model"))),
        };
        Ok(value)
    }

    fn write(
        &mut self,
        offset: u64,
        width: Width,
        value: u64,
        access: &mut Access<'_>,
    ) -> Result<(), MmioError> {
        let fail = |reason: Refusal| refuse(NAME, offset, width, Direction::Write, reason);
        self.accessible().map_err(fail)?;
        match (offset, width) {
            (CPU_CONTROL, Width::Word) => self.write_cpu_control(value).map_err(fail)?,
            (A2I_CONTROL, Width::Word) if value & !(CONTROL_ENABLE | CONTROL_STATUS) == 0 => {
                self.a2i_control = value & CONTROL_ENABLE
            }
            (I2A_CONTROL, Width::Word) if value & !(CONTROL_ENABLE | CONTROL_STATUS) == 0 => {
                // _enableOutbox (kernel file 0x1badc28..0x1badc94)
                // preserves read-only FIFO status in its RMW store.
                self.outbox_enabled = value & CONTROL_ENABLE != 0;
            }
            (A2I_CONTROL | I2A_CONTROL, Width::Word) => {
                return Err(fail(refuse!("mailbox control field without a model")));
            }
            (A2I_SEND0, Width::Double) => self.send0 = Some(value),
            (A2I_SEND1, Width::Double) => {
                self.send(value).map_err(fail)?;
                if let Some(request) = self.firmware.take_transition() {
                    access.request_transition(request.device, request.transition, request.reason);
                }
            }
            _ => return Err(fail(refuse!("ASC register without a model"))),
        }
        self.update_lines(access);
        Ok(())
    }
    fn advance(&mut self, access: &mut Access<'_>) -> Result<Option<u64>, MmioError> {
        if self.execution_enabled() {
            let mut sent = Vec::new();
            let result = self.firmware.poll(&mut sent);
            self.outbox.extend(sent);
            self.update_lines(access);
            result.map_err(|reason| refuse(NAME, 0, Width::Double, Direction::Read, reason))?;
            return Ok(None);
        }
        self.update_lines(access);
        Ok(None)
    }
}

#[cfg(test)]
mod tests;
