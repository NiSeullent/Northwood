//! The GFX coprocessor's AP-facing firmware protocol, modeled natively on two
//! RTKit application endpoints behind the generic ASC mailbox and RTKit
//! management ([`super::super::rtkit`]). Apple's `armfw_g14g` is not executed.
//!
//! Known (26A434 `AGXG14G`/`AGXFirmwareKextG14GRTBuddy`, unslid VAs):
//! - The kext matches two services named `GFXEndpoint1` and `GFXEndpoint2`
//!   (`AGXFirmwareKextRTBuddy::bootFirmware` 0x1ae9c88) and keeps endpoint
//!   **0x20** as the message endpoint and **0x21** as the doorbell
//!   (`matchedGFXEndpointGated` 0x1aea368). RTBuddy numbers application
//!   endpoint `n` as `Endpoint(n - 0x1f)`, so 0x20 is Endpoint1.
//! - After both endpoints start, the kext sends the initdata pointer on 0x20
//!   as `0x81 << 48 | (initdata_fw_va & ((1 << 44) - 1))`
//!   (`AGXArmFirmware::notifyFirmwareStarted` 0x1b44234).
//! - Work and control are asynchronous notes on 0x21:
//!   `(0x80 | type & 0x3f) << 48 | (index & 7) << 2 | kind`. A work doorbell
//!   (type 3) names the pipe in `kind` (TA 0, 3D 1, CL 2) and the submission
//!   index 0..4 in `index` (`AGXArmFirmware::submitTA/3D/CLChannel`
//!   0x1b56fb4/0x1b56e74/0x1b56d40); the device-control doorbell is the
//!   firmware-control type 4 with the low byte 0x11
//!   (`AGXArmFirmware::submitDeviceControl` 0x1b56bc0). The generic
//!   `kickFirmware` (0x1b56c18) forms the same layout with `kind` 0.
//! - Firmware-to-AP events arrive on 0x20 and the kext accepts only
//!   `(msg >> 48) & 0x3f == 2`, the 0x42 event doorbell
//!   (`AGXArmFirmware::receivedMessageFromAKF` 0x1b44480).
//!
//! - The initdata root the pointer names is one page whose first word is the
//!   magic `0x0c8a_6342_8392_04c8` (`AGXArmFirmware::initFirmwareData`
//!   0x1b5b36c, the page at `firmware+0x1630` that `notifyFirmwareStarted`
//!   names).
//!
//! [`Coprocessor`] is the whole GFX coprocessor behind the ASC: RTKit with
//! these endpoints, plus the firmware's bus-master accesses through the UAT in
//! the firmware context. On the initdata message it reads the root
//! ([`super::initdata::Root`]) and refuses one that is not the kext's. On the
//! device-control doorbell it consumes the device-control ring: every type
//! the 26A434 kext sends is decoded against the firmware's dispatcher, its
//! shared effects reproduced and its entry retired and counted; the types
//! whose effects need a unit or firmware thread the model lacks refuse by
//! name and stay unconsumed. On a work doorbell it runs the named data-master ring: each
//! entry's channel runs its pending commands in order. The signal commands
//! complete ([`super::signal`]), with their events posted on the firmware
//! event ring ([`super::event`]) and each announced by the 0x42 message on
//! 0x20; every command that needs the GPU refuses by its type and stays
//! pending, so no GPU work is retired and no completion is invented.
//!
//! Work that stays pending makes no progress, so the kext's progress timer
//! ends it: `AGXArmFirmware::stopFirmwareForRecoveryInspection` (0x1b57a6c)
//! sends `0x86 << 48 | generation` on 0x21 and panics unless the firmware's
//! generation word reads `generation + 1` and it reports itself stopped. The
//! firmware drops the work it held and follows the stop-state handshake
//! ([`Stop`]). The words are Known from the kext's readers and writers of
//! them; that nothing else of the firmware's recovery block matters to a stop
//! the kext asked for is Inferred.

use super::super::asc::{Firmware, Message as Mail};
use super::super::rtkit::{Endpoint, RtKit};
use super::channel;
use super::event;
use super::generation::{Family, Layout, RecoveryWords};
use super::initdata::{self, Root};
use super::ring::{self, Kind, Shared};
use super::signal;
use super::uat;
use crate::Result;
use alloc::boxed::Box;
use alloc::collections::{BTreeMap, VecDeque};
use alloc::rc::Rc;
use alloc::vec::Vec;
use core::cell::RefCell;
use stela_contracts::Refusal;
use stela_contracts::dma::Port;
use stela_contracts::refuse;
use stela_formats::builds::Build;

/// An RTKit application endpoint with its number, as [`RtKit::new`] takes them.
///
/// [`RtKit::new`]: super::super::rtkit::RtKit::new
pub type NumberedEndpoint = (u8, Box<dyn Endpoint>);

/// The message endpoint (`GFXEndpoint1`).
pub const MESSAGE_ENDPOINT: u8 = 0x20;
/// The doorbell endpoint (`GFXEndpoint2`).
pub const DOORBELL_ENDPOINT: u8 = 0x21;

const TYPE_SHIFT: u32 = 48;
/// The initdata message type on 0x20.
const MSG_INIT: u64 = 0x81;
/// Low 44 bits of the initdata message carry the firmware VA.
const INIT_VA_BITS: u64 = (1 << 44) - 1;
/// A doorbell's high bit (set in `0x80 | type`).
const DOORBELL_FLAG: u64 = 0x80;
const TYPE_MASK: u64 = 0x3f;
/// Work-queue doorbell (ring a pipe's work).
const DOORBELL_WORK: u64 = 3;
/// Firmware-control doorbell.
const DOORBELL_FWCTL: u64 = 4;
/// The doorbell that stops the firmware for recovery, naming the generation.
const DOORBELL_STOP: u64 = 6;
/// The firmware-control doorbell's low byte that names the device-control
/// ring (index 4, kind 1).
const FWCTL_DEVICE_CONTROL: u64 = 0x11;
/// The device-control power-transition notice (`AGXAccelerator::
/// powerStateDidChangeTo`/`WillChangeTo` 0x1af8508/0x1af861c).
const DEVICE_CONTROL_POWER: u32 = 0x28;
/// The device-control types the firmware counts once handled, one bit per
/// type (its byte table at 0xffffff80000593a0, read at 0xffffff8000003a2c).
const COUNTED_DEVICE_CONTROL: u64 = 0x1fff_3bfb_f3c7_bffe;
/// The ring directory's firmware VA of the HWDSID table (handlers
/// 0xffffff80000043a0 and 0xffffff8000003ba4).
const DSID_TABLE: u64 = 0x2c8;
/// The data masters the kext keeps a histogram for: TA, 3D and CL, whose
/// 0x30-byte histograms follow the enables byte.
const HISTOGRAM_MASTERS: u32 = 3;
/// Bytes of the power-counter sample buffer.
const POWER_COUNTER_BYTES: usize = 0x1800;
/// The timed-out word of an `AGXLateEvalWait`'s firmware object
/// (handler 0xffffff8000003d34).
const LATE_EVAL_TIMED_OUT: u64 = 0x38;
/// The event type the firmware would send on 0x20 (`0x42`).
pub const EVENT_TYPE: u64 = 0x42;

/// The work pipe a doorbell names (its low two bits): the TA (vertex), 3D
/// (fragment) and CL (compute) data masters. The value is also the
/// data-master entry's command type (`encodeAcceleratorRingCommand`
/// 0x1b56be8, called with 0, 1 and 2 by `AGXFirmware::submitTA/3D/CLChannel`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pipe {
    Vertex = 0,
    Fragment = 1,
    Compute = 2,
}

impl Pipe {
    fn from_kind(kind: u64) -> Option<Pipe> {
        match kind {
            0 => Some(Pipe::Vertex),
            1 => Some(Pipe::Fragment),
            2 => Some(Pipe::Compute),
            _ => None,
        }
    }
}

/// A device-control command the model carries out, by its shared effects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Command {
    Power(PowerNotice),
    /// The firmware has no handler for the type, or keeps every effect in
    /// its own data or registers: nothing the AP shares changes.
    Private,
    /// 0xe: set a DSID's requested priority, release the DSID once the
    /// firmware's word matches it, then write `value` at `pointer`.
    DsidPriority {
        dsid: u8,
        priority: u32,
        pointer: u64,
        value: u32,
    },
    /// 0x14: clear a DSID's table slot, then write `value` at `pointer`.
    DsidDeprioritize {
        dsid: u8,
        pointer: u64,
        value: u32,
    },
    /// 0x1a operation 3 or 4: enable or disable a data master's histogram.
    Histogram {
        master: u32,
        enable: bool,
    },
    /// 0x27: mark a late-evaluation wait's firmware object timed out.
    LateEvalTimeout {
        wait: u64,
    },
    /// 0x32: empty the power-counter sample buffer.
    PowerCounters,
    /// 0x34/0x35: a RIART word in the root's +0x20 object.
    Riart {
        word: Riart,
        value: u32,
    },
}

/// The RIART words device control sets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Riart {
    ULevelsDisabled,
    TDisabled,
}

/// A power-transition notice the AP sent on the device-control ring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerNotice {
    /// Word 1 is 0: the GPU's power state changed to a powered one.
    PoweredOn,
    /// Word 1 is 1: the GPU is about to change to power state 0.
    PoweringOff,
}

/// Firmware work a doorbell queued, run at the next DMA step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Job {
    DeviceControl,
    Work(Pipe, u8),
    Stop(u64),
}

/// Where a stopped firmware waits in the kext's recovery.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stop {
    /// Stop state 1, until the kext writes 2.
    PreRecovery,
    /// Stop state 3, until the kext writes 0.
    PostRecovery,
}

/// The firmware's half of the UAT handoff header SPTM builds in `gfx-handoff`:
/// the magic it publishes at +8 once it runs. `sptm_uat_unmap_begin` (sptm
/// 0xfffffff0270c5e5c, `uat.c:2807`) panics unless it reads this value; Public:
/// Asahi `Handoff::magic_fw`, `PPL_MAGIC`.
const HANDOFF_MICRO_MAGIC: (u64, u64) = (8, 0x4b1d_0000_0000_0002);

// Every modeled interface's original firmware binds root+a8 to this 0x20-byte
// shared status object. Startup clears ready/generation, publishes powered,
// then publishes ready after native initialization.
const STATUS_POWERED: u64 = 0x10;
const STATUS_READY: u64 = 0x14;
const STATUS_GENERATION: u64 = 0x1c;
const STARTUP_STATUS: [(u64, u32); 4] = [
    (STATUS_READY, 0),
    (STATUS_GENERATION, 0),
    (STATUS_POWERED, 1),
    (STATUS_READY, 1),
];

/// One accepted signal command, including ordered effects and its exact
/// ring retirement. The owner survives every fallible DMA boundary.
#[derive(Debug)]
struct ActiveCommand {
    actions: Vec<signal::Action>,
    next: usize,
    publication: Option<event::Publication>,
    retirement: channel::Retirement,
}

impl ActiveCommand {
    /// GPU-work commands refuse by name here: only signal commands are
    /// accepted.
    fn accept(
        memory: &mut dyn Shared,
        command: channel::Command,
        retirement: channel::Retirement,
    ) -> Result<Self> {
        Ok(Self {
            actions: signal::prepare(memory, &command)?,
            next: 0,
            publication: None,
            retirement,
        })
    }

    fn advance(
        &mut self,
        memory: &mut dyn Shared,
        root: &Root,
        outbox: &mut Vec<Mail>,
        events: &mut u64,
    ) -> Result<bool> {
        while let Some(&action) = self.actions.get(self.next) {
            match action {
                signal::Action::Event(event) => {
                    if self.publication.is_none() {
                        self.publication = event::Publication::reserve(memory, root, event)?;
                        if self.publication.is_none() {
                            return Ok(false);
                        }
                    }
                    self.publication.as_mut().unwrap().commit(memory)?;
                    self.publication = None;
                    outbox.push(Mail {
                        endpoint: MESSAGE_ENDPOINT,
                        payload: EVENT_TYPE << TYPE_SHIFT,
                    });
                    *events += 1;
                }
                write => write.write(memory)?,
            }
            self.next += 1;
        }
        self.retirement.commit(memory)?;
        Ok(true)
    }
}

/// A data-master entry keeps the descriptor, channel and final read-index
/// publication until the channel's accepted commands have all completed.
#[derive(Debug)]
struct ActiveWork {
    descriptor: ring::Descriptor,
    next_read: u32,
    submission: channel::Submission,
    command: Option<ActiveCommand>,
    retiring: bool,
}

/// A RAM-only effect of an accepted device-control entry. Multi-page
/// buffers own their bytes through a partial-prefix fault; scalar stores
/// must satisfy the same single-page contract as signal publications.
#[derive(Debug)]
struct ControlWrite {
    address: u64,
    bytes: Vec<u8>,
}

impl ControlWrite {
    fn new(address: u64, bytes: &[u8]) -> Result<Self> {
        address
            .checked_add(bytes.len() as u64 - 1)
            .ok_or(refuse!("GFX device-control output range wraps"))?;
        if bytes.len() <= 8 {
            ring::scalar_range(address, bytes.len() as u64)?;
        }
        Ok(Self {
            address,
            bytes: bytes.to_vec(),
        })
    }
}

/// The entry, read-derived values and exact retirement stay accepted until
/// all effects commit. Local notice/counter effects have their own phase so
/// a final consumer-index fault cannot count or enqueue them twice.
#[derive(Debug)]
struct ActiveControl {
    descriptor: ring::Descriptor,
    next_read: u32,
    kind: u32,
    power: Option<PowerNotice>,
    dsid_priority: Option<ControlPriority>,
    writes: Vec<ControlWrite>,
    count: Option<ControlCount>,
    next: usize,
    local_committed: bool,
}

/// Preserve the original priority store -> matching-word read -> optional
/// clear. A read fault observes the already committed priority store; a
/// later clear/write fault never repeats that store or rediscovers the match.
#[derive(Debug)]
struct ControlPriority {
    slot: u64,
    priority: u32,
    matched: Option<u32>,
    phase: u8,
}
impl ControlPriority {
    fn commit(&mut self, memory: &mut dyn Shared) -> Result<()> {
        if self.phase == 0 {
            memory.write(self.slot, &self.priority.to_le_bytes())?;
            self.phase = 1;
        }
        if self.phase == 1 {
            self.matched = Some(memory.read_u32(self.slot + 4)?);
            self.phase = 2;
        }
        if self.phase == 2 {
            if self.matched == Some(self.priority) && self.priority != 0 {
                memory.write(self.slot, &[0; 8])?;
            }
            self.phase = 3;
        }
        Ok(())
    }
}

/// The original common-tail RMW follows the command's writes. Capture its
/// result before attempting the write, so a write fault never increments
/// twice; a read fault leaves prior command effects committed and owned.
#[derive(Debug)]
struct ControlCount {
    address: u64,
    value: Option<u32>,
    committed: bool,
}

impl ControlCount {
    fn commit(&mut self, memory: &mut dyn Shared) -> Result<()> {
        if self.committed {
            return Ok(());
        }
        let value = match self.value {
            Some(value) => value,
            None => {
                let value = memory.read_u32(self.address)?.wrapping_add(1);
                self.value = Some(value);
                value
            }
        };
        memory.write(self.address, &value.to_le_bytes())?;
        self.committed = true;
        Ok(())
    }
}

impl ActiveControl {
    fn accept(
        memory: &mut dyn Shared,
        root: &Root,
        layout: &Layout,
        descriptor: ring::Descriptor,
        read: u32,
        entry: &[u8],
    ) -> Result<Self> {
        let kind = u32::from_le_bytes(entry[..4].try_into().unwrap());
        let mut writes = Vec::new();
        let mut power = None;
        let mut dsid_priority = None;
        let mut put = |address: u64, bytes: &[u8]| -> Result<()> {
            writes.push(ControlWrite::new(address, bytes)?);
            Ok(())
        };
        let address = |base: u64, offset: u64| {
            base.checked_add(offset)
                .ok_or(refuse!("GFX device-control output address wraps"))
        };
        match device_control_entry(entry)? {
            Command::Power(notice) => power = Some(notice),
            Command::Private => {}
            Command::DsidPriority {
                dsid,
                priority,
                pointer,
                value,
            } => {
                let slot = dsid_slot(memory, root, layout, dsid)?;
                ring::scalar_range(slot, 8)?;
                dsid_priority = Some(ControlPriority {
                    slot,
                    priority,
                    matched: None,
                    phase: 0,
                });
                put(pointer, &value.to_le_bytes())?;
            }
            Command::DsidDeprioritize {
                dsid,
                pointer,
                value,
            } => {
                let slot = dsid_slot(memory, root, layout, dsid)?;
                put(slot, &[0; 8])?;
                put(pointer, &value.to_le_bytes())?;
            }
            Command::Histogram { master, enable } => {
                let at = address(root.rings, layout.histogram_enables)?;
                let mut enables = [0];
                memory.read(at, &mut enables)?;
                let bit = 1 << master;
                enables[0] = if enable {
                    enables[0] | bit
                } else {
                    enables[0] & !bit
                };
                put(at, &enables)?;
            }
            Command::LateEvalTimeout { wait } => {
                put(address(wait, LATE_EVAL_TIMED_OUT)?, &1u32.to_le_bytes())?;
            }
            Command::PowerCounters => {
                put(
                    address(root.object_928, layout.power_counters)?,
                    &[0; POWER_COUNTER_BYTES],
                )?;
                put(
                    address(root.object_928, layout.power_counter_samples)?,
                    &[0; 4],
                )?;
            }
            Command::Riart { word, value } => {
                let offset = match word {
                    Riart::ULevelsDisabled => layout.riart_u_levels_disabled,
                    Riart::TDisabled => layout.riart_t_disabled,
                };
                put(address(root.object_358, offset)?, &value.to_le_bytes())?;
            }
        }
        let count = if COUNTED_DEVICE_CONTROL
            .checked_shr(kind)
            .is_some_and(|bits| bits & 1 != 0)
        {
            let address = address(root.object_928, layout.device_control_count)?;
            ring::scalar_range(address, 4)?;
            Some(ControlCount {
                address,
                value: None,
                committed: false,
            })
        } else {
            None
        };
        ring::scalar_range(descriptor.state, 4)?;
        Ok(Self {
            descriptor,
            next_read: ring::next(read),
            kind,
            power,
            dsid_priority,
            writes,
            count,
            next: 0,
            local_committed: false,
        })
    }
}

/// The GFX firmware protocol state shared by its two endpoints.
#[derive(Debug)]
pub struct State {
    /// The AGX family whose kext and firmware fix the interface.
    family: &'static Family,
    /// The firmware VA of the initdata root, once the kext sends it, as the
    /// 44-bit field of the message.
    pub initdata_va: Option<u64>,
    /// The root was validated through the UAT and startup status committed.
    pub initdata_checked: bool,
    /// The last attempt to validate the root and commit the startup status
    /// was refused: the start does not go on, so the firmware runs no
    /// doorbell and has not published itself in the handoff.
    start_refused: bool,
    /// The validated root, kept even if startup-status DMA later faults.
    pub root: Option<Root>,
    startup_word: usize,
    /// Work doorbells received, in order, as (pipe, submission index).
    pub work: Vec<(Pipe, u8)>,
    /// Count of device-control doorbells received.
    pub fwctl: u64,
    /// Power-transition notices consumed, in order.
    pub power: Vec<PowerNotice>,
    /// Device-control entries consumed, by type.
    pub device_control: BTreeMap<u32, u64>,
    /// Channel commands completed.
    pub commands: u64,
    /// Firmware events posted.
    pub events: u64,
    jobs: VecDeque<Job>,
    active_work: Option<ActiveWork>,
    active_control: Option<ActiveControl>,
    stop: Option<Stop>,
    /// The `gfx-handoff` page, until the firmware publishes its magic there.
    handoff: Option<u64>,
}

impl State {
    /// A fresh protocol state for `family`'s interface.
    fn new(family: &'static Family) -> Self {
        State {
            family,
            initdata_va: None,
            initdata_checked: false,
            start_refused: false,
            root: None,
            startup_word: 0,
            work: Vec::new(),
            fwctl: 0,
            power: Vec::new(),
            device_control: BTreeMap::new(),
            commands: 0,
            events: 0,
            jobs: VecDeque::new(),
            active_work: None,
            active_control: None,
            stop: None,
            handoff: None,
        }
    }

    fn init(&mut self, payload: u64) -> Result<()> {
        if payload >> TYPE_SHIFT != MSG_INIT {
            return Err(refuse!(
                "AGX message endpoint models only the 0x81 initdata message"
            ));
        }
        if payload & !(MSG_INIT << TYPE_SHIFT | INIT_VA_BITS) != 0 {
            return Err(refuse!(
                "AGX initdata message has bits outside its 44-bit address field"
            ));
        }
        if self.initdata_va.is_some() {
            return Err(refuse!("AGX initdata sent twice"));
        }
        self.initdata_va = Some(payload & INIT_VA_BITS);
        Ok(())
    }

    fn doorbell(&mut self, payload: u64) -> Result<()> {
        let hi = payload >> TYPE_SHIFT;
        if hi & DOORBELL_FLAG == 0 {
            return Err(refuse!("AGX doorbell without its 0x80 type flag"));
        }
        if hi == DOORBELL_FLAG | DOORBELL_STOP {
            // A stop takes the firmware off whatever it held: the kext's
            // recovery owns those channels from here.
            self.jobs.retain(|job| matches!(job, Job::DeviceControl));
            self.active_work = None;
            self.jobs.push_back(Job::Stop(payload & INIT_VA_BITS));
            return Ok(());
        }
        // The kext forms (0x80 | type) << 48 | (index & 7) << 2 | kind;
        // nothing else is set.
        let type_bits = (DOORBELL_FLAG | TYPE_MASK) << TYPE_SHIFT;
        if payload & !(type_bits | 0x1f) != 0 {
            return Err(refuse!(
                "AGX doorbell has bits outside the type, index and kind"
            ));
        }
        let low = payload & 0x1f;
        let (index, kind) = ((low >> 2) as u8, low & 3);
        match hi & TYPE_MASK {
            DOORBELL_WORK => {
                let pipe =
                    Pipe::from_kind(kind).ok_or(refuse!("AGX work doorbell names no pipe"))?;
                if index >= initdata::WORK_INDICES {
                    return Err(refuse!("AGX work doorbell names no submission index"));
                }
                self.work.push((pipe, index));
                self.jobs.push_back(Job::Work(pipe, index));
                Ok(())
            }
            DOORBELL_FWCTL if low == FWCTL_DEVICE_CONTROL => {
                self.fwctl += 1;
                self.jobs.push_back(Job::DeviceControl);
                Ok(())
            }
            DOORBELL_FWCTL => Err(refuse!(
                "AGX firmware-control doorbell other than device control (0x11) is not modeled"
            )),
            _ => Err(refuse!("AGX doorbell type is not modeled")),
        }
    }

    /// The firmware has a bus-master access to make.
    fn dma_pending(&self) -> bool {
        (self.initdata_va.is_some() && (!self.initdata_checked || self.handoff.is_some()))
            || !self.jobs.is_empty()
            || self.active_work.is_some()
            || self.active_control.is_some()
            || self.stop.is_some()
    }

    /// Whether an IOP sleep must wait for the firmware: a start or a stop
    /// still in progress, or control work it holds. A start whose root was
    /// refused does not progress, and none of its queued doorbells can run.
    fn sleep_waits(&self) -> bool {
        !self.start_refused && self.dma_pending()
    }

    /// Read the root, then run the queued doorbells in order.
    fn dma(&mut self, memory: &mut dyn Shared, outbox: &mut Vec<Mail>) -> Result<()> {
        let started = self.start(memory);
        self.start_refused = started.is_err();
        started?;
        self.work(memory, outbox)
    }

    /// Validate the root the kext sent and commit the startup status and the
    /// handoff magic, once.
    fn start(&mut self, memory: &mut dyn Shared) -> Result<()> {
        if let Some(va) = self.initdata_va.filter(|_| !self.initdata_checked) {
            if self.root.is_none() {
                let magic = self.family.initdata_magic.ok_or(refuse!(
                    "the GPU family's initdata root magic is not recovered"
                ))?;
                self.root = Some(Root::read(memory, uat::canonical_firmware_va(va), magic)?);
            }
            let status = self
                .root
                .as_ref()
                .expect("validated initdata root")
                .object_848;
            status
                .checked_add(0x20)
                .ok_or(refuse!("GFX firmware status range wraps"))?;
            while let Some(&(offset, value)) = STARTUP_STATUS.get(self.startup_word) {
                memory.write(status + offset, &value.to_le_bytes())?;
                self.startup_word += 1;
            }
            self.initdata_checked = true;
        }
        if self.initdata_checked
            && let Some(handoff) = self.handoff
        {
            let (offset, magic) = HANDOFF_MICRO_MAGIC;
            memory.write_physical(handoff + offset, &magic.to_le_bytes())?;
            self.handoff = None;
        }
        Ok(())
    }

    /// Run the queued doorbells in order.
    fn work(&mut self, memory: &mut dyn Shared, outbox: &mut Vec<Mail>) -> Result<()> {
        while let Some(&job) = self.jobs.front() {
            let root = self
                .root
                .clone()
                .ok_or(refuse!("GFX doorbell before the initdata root"))?;
            if self.stopped(memory, &root)? {
                return Ok(());
            }
            let complete = match job {
                Job::DeviceControl => {
                    self.device_control(memory, &root)?;
                    true
                }
                Job::Work(pipe, index) => self.work_ring(memory, &root, pipe, index, outbox)?,
                Job::Stop(generation) => {
                    self.stop_for_recovery(memory, &root, generation)?;
                    true
                }
            };
            if !complete {
                return Ok(());
            }
            self.jobs.pop_front();
        }
        if let Some(root) = self.root.clone() {
            self.stopped(memory, &root)?;
        }
        Ok(())
    }

    /// Report the stop, the generation word last: it is what the kext polls.
    fn stop_for_recovery(
        &mut self,
        memory: &mut dyn Shared,
        root: &Root,
        generation: u64,
    ) -> Result<()> {
        let words = self.recovery_words()?;
        memory.write(root.object_928 + words.stopped, &1u32.to_le_bytes())?;
        memory.write(root.object_928 + words.stop_state, &1u32.to_le_bytes())?;
        let next = generation + 1;
        memory.write(root.object_928 + words.generation, &next.to_le_bytes())?;
        self.stop = Some(Stop::PreRecovery);
        Ok(())
    }

    /// Follow the kext through recovery. True while the firmware stays stopped.
    fn stopped(&mut self, memory: &mut dyn Shared, root: &Root) -> Result<bool> {
        let Some(stop) = self.stop else {
            return Ok(false);
        };
        let words = self.recovery_words()?;
        let state = memory.read_u32(root.object_928 + words.stop_state)?;
        match (stop, state) {
            (Stop::PreRecovery, 1) | (Stop::PostRecovery, 3) => Ok(true),
            (Stop::PreRecovery, 2) => {
                memory.write(root.object_928 + words.stopped, &0u32.to_le_bytes())?;
                memory.write(root.object_928 + words.stop_state, &3u32.to_le_bytes())?;
                self.stop = Some(Stop::PostRecovery);
                Ok(true)
            }
            (Stop::PostRecovery, 0) => {
                self.stop = None;
                Ok(false)
            }
            _ => Err(refuse!(
                "GFX stop state is not one the kext's recovery writes",
                state
            )),
        }
    }

    fn recovery_words(&self) -> Result<RecoveryWords> {
        self.family
            .model()?
            .layout
            .recovery
            .ok_or(refuse!("the GPU family's recovery words are not recovered"))
    }

    /// Consume the device-control ring up to its write index.
    fn device_control(&mut self, memory: &mut dyn Shared, root: &Root) -> Result<()> {
        let layout: &Layout = &self.family.model()?.layout;
        loop {
            if self.active_control.is_none() {
                let descriptor = root.device_control(memory)?;
                let (read, write) = descriptor.indices(memory)?;
                if read == write {
                    return Ok(());
                }
                let entry = descriptor.entry(memory, Kind::DeviceControl, read)?;
                self.active_control = Some(ActiveControl::accept(
                    memory, root, layout, descriptor, read, &entry,
                )?);
            }
            let active = self.active_control.as_mut().unwrap();
            if let Some(priority) = active.dsid_priority.as_mut() {
                priority.commit(memory)?;
            }
            while let Some(write) = active.writes.get(active.next) {
                memory.write(write.address, &write.bytes)?;
                active.next += 1;
            }
            if let Some(count) = active.count.as_mut() {
                count.commit(memory)?;
            }
            if !active.local_committed {
                if let Some(notice) = active.power {
                    self.power.push(notice);
                }
                *self.device_control.entry(active.kind).or_default() += 1;
                active.local_committed = true;
            }
            active.descriptor.retire(memory, active.next_read)?;
            self.active_control = None;
        }
    }

    /// Run a data-master ring's pending entries. Each names a channel; the
    /// channel's pending commands run in order, each retired from the
    /// channel's command ring once complete, and the entry is retired once
    /// the entry's captured command tail is reached. A command the model does not execute
    /// refuses by name and stays pending, as does its entry.
    fn work_ring(
        &mut self,
        memory: &mut dyn Shared,
        root: &Root,
        pipe: Pipe,
        index: u8,
        outbox: &mut Vec<Mail>,
    ) -> Result<bool> {
        loop {
            let active = match &mut self.active_work {
                Some(active) => active,
                None => {
                    let descriptor = root.work(memory, pipe, index)?;
                    let (read, write) = descriptor.indices(memory)?;
                    if read == write {
                        return Ok(true);
                    }
                    let entry = descriptor.entry(memory, Kind::DataMaster, read)?;
                    let command = u32::from_le_bytes(entry[0x10..0x14].try_into().unwrap());
                    if command != pipe as u32 {
                        return Err(refuse!(
                            "GFX data-master entry's command type is not its ring's pipe"
                        ));
                    }
                    let channel_state = u64::from_le_bytes(entry[0x08..0x10].try_into().unwrap());
                    let queue = channel::Queue::read(memory, channel_state)?;
                    let target = u16::from_le_bytes(entry[0x14..0x16].try_into().unwrap());
                    let submission = queue.submission(memory, channel_state, target)?;
                    ring::scalar_range(descriptor.state + ring::READ, 4)?;
                    self.active_work.insert(ActiveWork {
                        descriptor,
                        next_read: ring::next(read),
                        submission,
                        command: None,
                        retiring: false,
                    })
                }
            };
            if active.retiring {
                active.descriptor.retire(memory, active.next_read)?;
                self.active_work = None;
                continue;
            }
            active.submission.publish(memory)?;
            loop {
                let command = match &mut active.command {
                    Some(command) => command,
                    None => {
                        let Some(va) = active.submission.next(memory)? else {
                            // The target is held even if this final DMA
                            // fails. It cannot retire a later entry on retry.
                            active.retiring = true;
                            break;
                        };
                        let command = channel::Command::read(memory, va)?;
                        if command.kind == channel::Kind::DashboardSignal
                            && !self.family.dashboard_signals
                        {
                            return Err(refuse!(
                                "GFX dashboard signal command: this GPU family's firmware has no dashboard-signal handler"
                            ));
                        }
                        let retirement = active.submission.retirement();
                        active
                            .command
                            .insert(ActiveCommand::accept(memory, command, retirement)?)
                    }
                };
                if !command.advance(memory, root, outbox, &mut self.events)? {
                    return Ok(false);
                }
                active.command = None;
                active.submission.consumed();
                self.commands += 1;
            }
        }
    }
}

/// The HWDSID table slot of `dsid`, through the ring directory's table VA.
fn dsid_slot(memory: &mut dyn Shared, root: &Root, layout: &Layout, dsid: u8) -> Result<u64> {
    let pointer = root
        .rings
        .checked_add(DSID_TABLE)
        .ok_or(refuse!("GFX HWDSID table pointer wraps"))?;
    pointer
        .checked_add(7)
        .ok_or(refuse!("GFX HWDSID table pointer wraps"))?;
    let table = memory.read_u64(pointer)?;
    if table == 0 {
        return Err(refuse!("GFX ring directory names no HWDSID table"));
    }
    let offset = u64::from(dsid)
        .checked_mul(layout.dsid_slot_bytes)
        .ok_or(refuse!("GFX HWDSID table slot offset wraps"))?;
    let slot = table
        .checked_add(offset)
        .ok_or(refuse!("GFX HWDSID table slot wraps"))?;
    let extent = layout
        .dsid_slot_bytes
        .checked_sub(1)
        .ok_or(refuse!("GFX HWDSID table has zero-sized slots"))?;
    slot.checked_add(extent)
        .ok_or(refuse!("GFX HWDSID table slot extent wraps"))?;
    Ok(slot)
}

/// Every byte of `entry` after the type and outside `fields`, each a
/// half-open (start, end) byte range, is zero.
fn only(entry: &[u8], fields: &[(usize, usize)]) -> bool {
    entry.iter().enumerate().skip(4).all(|(at, &b)| {
        b == 0
            || fields
                .iter()
                .any(|&(start, end)| (start..end).contains(&at))
    })
}

/// Decode one device-control entry (`AGFIAcceleratorDeviceControlEntry`,
/// 0x40 bytes, the type at +0) into its shared effects. Each type's fields
/// are those its 26A434 kext builder writes; every other byte is zero, as
/// the builders zero the entry first. The firmware handlers are the 26A434
/// dispatcher's (0xffffff8000003a1c, a jump table over types 4..0x3b at
/// 0xffffff8000004c48).
fn device_control_entry(entry: &[u8]) -> Result<Command> {
    let word = |at: usize| u32::from_le_bytes(entry[at..at + 4].try_into().unwrap());
    let u64_at = |at: usize| u64::from_le_bytes(entry[at..at + 8].try_into().unwrap());
    let form = |fields: &[(usize, usize)], beyond: &'static str| {
        if only(entry, fields) {
            Ok(Command::Private)
        } else {
            Err(Refusal::new(beyond))
        }
    };
    let boolean = |beyond: &'static str| {
        if word(4) > 1 {
            Err(refuse!("GFX device-control flag word is not 0 or 1"))
        } else {
            form(&[(4, 8)], beyond)
        }
    };
    match word(0) {
        DEVICE_CONTROL_POWER => {
            if entry[8..].iter().any(|&b| b != 0) {
                return Err(refuse!(
                    "GFX device-control power notice has fields beyond its word"
                ));
            }
            match word(4) {
                0 => Ok(Command::Power(PowerNotice::PoweredOn)),
                1 => Ok(Command::Power(PowerNotice::PoweringOff)),
                _ => Err(refuse!(
                    "GFX device-control power notice names no transition"
                )),
            }
        }
        // The dispatcher has no handler for these: they go straight to its
        // common tail. 7 is the device-state query; no 26A434 firmware code
        // writes its report at +0xe82c of the root's +0xb0 object, so the
        // section mask stays as the AP left it. 0x23 (FW utilisation
        // monitor), 0x2d (LUD hints), 0x12 (VKSCDM reset), 0x11 (VKSCDM
        // distribution), 0x1c (GVDM mode) and 0x31 (AVMR) are likewise
        // unhandled by this firmware.
        0x7 | 0x12 | 0x23 | 0x2d => form(&[], "GFX device control has fields beyond its type"),
        0x11 => form(
            &[(4, 0x14)],
            "GFX VKSCDM distribution has fields beyond its state",
        ),
        0x1c => form(&[(4, 8)], "GFX GVDM mode has fields beyond its word"),
        0x31 => form(
            &[(4, 0xd)],
            "GFX AVMR control has fields beyond its overrides",
        ),
        // Handlers whose effects stay in the firmware's own data, its
        // power-management threads or its registers. 0xb/0xc/0xd wake the
        // power thread for new hot-spot offsets and DPE/PPT parameters; 0x15
        // and 0x30 copy initdata configuration (the UMA recall words the kext
        // wrote at +0xe1c of the root's +0x20 object) into firmware data; 0x22
        // copies the display power words at +0x988 of the same object; 0x1b
        // only cleans the firmware's cached SFE tier-3 and THM counts at
        // directory +0x47d/+0x475, which no unit of the model ever raises;
        // 0x33 issues the register-window instruction-cache invalidate, which
        // the model's GPU, holding no instruction cache, has nothing to act on.
        // The rest set firmware power, QoS and scheduling state: 0xa idle
        // power-off, 0xf/0x10 CDM distribution, 0x13 resource release, 0x16/
        // 0x17 default QoS, 0x18 FRG power nap, 0x19 GIFAF hint, 0x25 QoS
        // knob, 0x38 power-counter selection (the list the kext wrote at
        // +0xca34 of the root's +0xb0 object) and 0x3b remote IOP power,
        // whose handler reads its words and does nothing.
        0xb | 0xc | 0xd | 0x10 | 0x15 | 0x1b | 0x22 | 0x30 | 0x38 => {
            form(&[], "GFX device control has fields beyond its type")
        }
        0xa => boolean("GFX idle power-off has fields beyond its flag"),
        0xf => form(
            &[(4, 0x1c)],
            "GFX CDM distribution has fields beyond its state",
        ),
        0x13 => form(
            &[(8, 0x14)],
            "GFX resource release has fields beyond its slots",
        ),
        0x16 | 0x17 => form(&[(4, 0xc)], "GFX default QoS has fields beyond its value"),
        0x18 => {
            if word(4) > 1 {
                return Err(refuse!("GFX FRG power nap flag is not 0 or 1"));
            }
            form(
                &[(4, 0x10)],
                "GFX FRG power nap has fields beyond its words",
            )
        }
        0x19 => boolean("GFX GIFAF streaming hint has fields beyond its flag"),
        0x25 => {
            // The firmware aborts on a data master other than 0..2
            // (0xffffff8000004c28).
            if entry[4] > 2 {
                return Err(refuse!(
                    "GFX QoS knob names no data master the firmware accepts"
                ));
            }
            form(&[(4, 0xa)], "GFX QoS knob has fields beyond its knobs")
        }
        0x33 => form(
            &[(4, 0xc)],
            "GFX ICache invalidate has fields beyond its words",
        ),
        0x3b => form(
            &[(4, 0x10)],
            "GFX remote IOP power notice has fields beyond its words",
        ),
        // Handlers with shared effects.
        0xe => {
            if !only(entry, &[(4, 0x15)]) {
                return Err(refuse!("GFX DSID priority has fields beyond its slot"));
            }
            if u64_at(4) == 0 {
                return Err(refuse!("GFX DSID priority names no completion word"));
            }
            Ok(Command::DsidPriority {
                dsid: entry[0x14],
                priority: word(0x10),
                pointer: u64_at(4),
                value: word(0xc),
            })
        }
        0x14 => {
            if !only(entry, &[(4, 0x10), (0x14, 0x15)]) {
                return Err(refuse!("GFX DSID deprioritize has fields beyond its slot"));
            }
            if u64_at(4) == 0 {
                return Err(refuse!("GFX DSID deprioritize names no completion word"));
            }
            Ok(Command::DsidDeprioritize {
                dsid: entry[0x14],
                pointer: u64_at(4),
                value: word(0xc),
            })
        }
        0x1a => {
            if !only(entry, &[(4, 9)]) {
                return Err(refuse!(
                    "GFX context-switch histogram has fields beyond its operation"
                ));
            }
            let master = word(4);
            if master >= HISTOGRAM_MASTERS {
                return Err(refuse!("GFX context-switch histogram names no data master"));
            }
            match entry[8] {
                // 1 (get), 2 (reset) and 5 only clean the firmware's cached
                // histogram for the AP to read; the model never switches a
                // data master's context, so its bins stay as they are.
                1 | 2 | 5 => Ok(Command::Private),
                3 => Ok(Command::Histogram {
                    master,
                    enable: true,
                }),
                4 => Ok(Command::Histogram {
                    master,
                    enable: false,
                }),
                _ => Err(refuse!("GFX context-switch histogram names no operation")),
            }
        }
        0x27 => {
            if !only(entry, &[(4, 0x20)]) {
                return Err(refuse!("GFX late-eval timeout has fields beyond its wait"));
            }
            match u64_at(4) {
                0 => Err(refuse!("GFX late-eval timeout names no wait")),
                wait => Ok(Command::LateEvalTimeout { wait }),
            }
        }
        0x32 => {
            form(&[], "GFX power-counter gather has fields beyond its type")?;
            Ok(Command::PowerCounters)
        }
        0x34 => {
            form(
                &[(4, 8)],
                "GFX RIART U-level mask has fields beyond its word",
            )?;
            Ok(Command::Riart {
                word: Riart::ULevelsDisabled,
                value: word(4),
            })
        }
        0x35 => {
            boolean("GFX RIART T-disable has fields beyond its flag")?;
            Ok(Command::Riart {
                word: Riart::TDisabled,
                value: word(4),
            })
        }
        // Refused by name.
        0x4 | 0x5 => Err(refuse!(
            "GFX device control 0x4/0x5 starts or stops performance-counter sampling, which programs counter hardware the model lacks"
        )),
        // AGXAccelerator::allocateMemoryEvent answers a host-allocation
        // request the firmware raised; the model raises none.
        0x8 | 0x26 => Err(refuse!(
            "GFX device control 0x8/0x26 answers a firmware host-allocation request the model never makes"
        )),
        0x9 => Err(refuse!(
            "GFX device control 0x9 answers a firmware event the model never posts and wakes a firmware scheduler thread it does not run"
        )),
        0x1f => Err(refuse!(
            "GFX device control 0x1f updates an app role in the firmware scheduler's context registry, which the model does not keep"
        )),
        // An AFR perf-state override the 26A434 firmware lacks: its handler
        // only logs a warning, and firmware log records are excluded, so the
        // entry is consumed.
        0x24 => form(&[(4, 8)], "GFX AFR perf state has fields beyond its word"),
        0x29 => Err(refuse!(
            "GFX device control 0x29 hands an exiting context to a firmware teardown thread the model does not run"
        )),
        0x2c => Err(refuse!(
            "GFX device control 0x2c (external CS or kill request) is a fatal unsupported command in the 26A434 firmware"
        )),
        _ => Err(refuse!(
            "GFX device-control type is not one the 26A434 kext sends"
        )),
    }
}

/// A handle to the shared protocol state, for the DMA boundary and tests.
#[derive(Clone)]
pub struct Protocol(Rc<RefCell<State>>);

impl Protocol {
    /// The protocol of `family`'s firmware.
    pub fn new(family: &'static Family) -> Self {
        Protocol(Rc::new(RefCell::new(State::new(family))))
    }

    pub fn with<R>(&self, f: impl FnOnce(&State) -> R) -> R {
        f(&self.0.borrow())
    }
}

/// The GFX coprocessor: RTKit with the GFX endpoints and the firmware's DMA.
pub struct Coprocessor {
    rtkit: RtKit,
    protocol: Protocol,
    power: Option<super::sgx::FirmwarePower>,
}

impl Coprocessor {
    /// `family`'s GFX coprocessor, whose RTKit manages its power ordinarily
    /// (`power-managed`), speaking `build`'s RTKit version.
    pub fn new(family: &'static Family, build: &Build) -> Result<Self> {
        let (endpoints, protocol) = endpoints(family);
        let rtkit = RtKit::new(endpoints, build.rtkit_version)?;
        Ok(Self {
            rtkit,
            protocol,
            power: None,
        })
    }

    /// Attach the native power/register owner of the mounted SGX instance.
    /// A checked physical binding does not make unknown callbacks successful.
    pub fn with_power(
        family: &'static Family,
        power: super::sgx::FirmwarePower,
        build: &Build,
    ) -> Result<Self> {
        if family != power.family() {
            return Err(refuse!(
                "GFX firmware power owner belongs to a different GPU family"
            ));
        }
        let mut coprocessor = Self::new(family, build)?;
        coprocessor.power = Some(power);
        Ok(coprocessor)
    }

    /// Have the firmware publish itself in the UAT handoff at `handoff` as it
    /// starts.
    pub fn with_handoff(self, handoff: u64) -> Self {
        self.protocol.0.borrow_mut().handoff = Some(handoff);
        self
    }
    #[cfg(test)]
    pub fn protocol(&self) -> &Protocol {
        &self.protocol
    }
}

impl Firmware for Coprocessor {
    fn check_reset(&self) -> Result<()> {
        if let Some(power) = &self.power {
            power.check_reset()?;
        }
        self.rtkit.check_reset()
    }
    fn reset(&mut self) -> Result<()> {
        self.check_reset()?;
        self.rtkit.reset()
    }
    fn start(&mut self, outbox: &mut Vec<Mail>) -> Result<()> {
        self.rtkit.start(outbox)
    }
    fn receive(&mut self, message: Mail, outbox: &mut Vec<Mail>) -> Result<()> {
        if is_iop_sleep(message) && self.protocol.with(|state| state.initdata_va.is_some()) {
            // Original native power callbacks depend on the AP hardware
            // config, update shared powered/generation words and restore GPU
            // registers. None of that runs: `error_reply` acknowledges the
            // request once the firmware owns no work.
            return Err(refuse!(
                "GFX initialized IOP sleep runs no native power callbacks"
            ));
        }
        if matches!(message.endpoint, MESSAGE_ENDPOINT | DOORBELL_ENDPOINT)
            && !matches!(
                self.rtkit.iop_power(),
                super::super::rtkit::POWER_ON | 0x220
            )
        {
            return Err(refuse!(
                "GFX application message while the IOP is not awake"
            ));
        }
        self.rtkit.receive(message, outbox)
    }
    fn dma_pending(&self) -> bool {
        self.protocol.with(State::dma_pending)
    }
    fn dma(&mut self, port: &mut Port<'_, '_>, outbox: &mut Vec<Mail>) -> Result<()> {
        self.protocol.0.borrow_mut().dma(port, outbox)
    }
    /// The degraded answer to an initialized IOP sleep: the management ACK
    /// with the firmware's state kept, without the native callbacks. Work the
    /// firmware holds goes as it does at a stop, because sleep ends its claim
    /// on those channels; a start or a stop still in progress stays unanswered.
    /// A start that was refused is not in progress: its sleep is acknowledged.
    fn error_reply(&mut self, message: Mail, _reason: &Refusal) -> Option<Vec<Mail>> {
        if !is_iop_sleep(message) {
            return None;
        }
        {
            let mut state = self.protocol.0.borrow_mut();
            state.jobs.retain(|job| !matches!(job, Job::Work(..)));
            state.active_work = None;
            if state.sleep_waits() {
                return None;
            }
        }
        let mut replies = Vec::new();
        self.rtkit.receive(message, &mut replies).ok()?;
        Some(replies)
    }
}

/// A management request for one of the IOP's sleep states.
fn is_iop_sleep(message: Mail) -> bool {
    message.endpoint == 0
        && message.payload >> 52 & 0xf == 6
        && matches!(message.payload & !(0xf << 52), 0x201 | 0x202 | 0)
}

/// The 0x20 message endpoint.
struct Message(Protocol);
/// The 0x21 doorbell endpoint.
struct Doorbell(Protocol);

impl Endpoint for Message {
    fn check_reset(&self) -> Result<()> {
        if self.0.with(State::dma_pending) {
            return Err(refuse!(
                "GFX physical reset with owned initdata or queued channel work"
            ));
        }
        Ok(())
    }
    fn reset(&mut self) -> Result<()> {
        self.check_reset()?;
        let mut state = self.0.0.borrow_mut();
        state.initdata_va = None;
        state.initdata_checked = false;
        state.start_refused = false;
        state.root = None;
        state.startup_word = 0;
        // Historical counters/journals remain diagnostic evidence. Channel
        // addresses and validation are cold-reset session state.
        Ok(())
    }
    fn start(&mut self, _outbox: &mut Vec<u64>) -> Result<()> {
        Ok(())
    }
    fn receive(&mut self, payload: u64, _outbox: &mut Vec<u64>) -> Result<()> {
        self.0.0.borrow_mut().init(payload)
    }
}

impl Endpoint for Doorbell {
    fn check_reset(&self) -> Result<()> {
        Message(self.0.clone()).check_reset()
    }
    fn reset(&mut self) -> Result<()> {
        Message(self.0.clone()).reset()
    }
    fn start(&mut self, _outbox: &mut Vec<u64>) -> Result<()> {
        Ok(())
    }
    fn receive(&mut self, payload: u64, _outbox: &mut Vec<u64>) -> Result<()> {
        self.0.0.borrow_mut().doorbell(payload)
    }
}

/// The GFX coprocessor's application endpoints and a handle to their shared
/// state. Pass the endpoints to [`super::super::rtkit::RtKit::new`].
pub fn endpoints(family: &'static Family) -> (Vec<NumberedEndpoint>, Protocol) {
    let protocol = Protocol::new(family);
    let endpoints: Vec<NumberedEndpoint> = alloc::vec![
        (
            MESSAGE_ENDPOINT,
            Box::new(Message(protocol.clone())) as Box<dyn Endpoint>,
        ),
        (DOORBELL_ENDPOINT, Box::new(Doorbell(protocol.clone()))),
    ];
    (endpoints, protocol)
}

#[cfg(test)]
mod tests;
