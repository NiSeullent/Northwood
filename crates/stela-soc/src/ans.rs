//! ANS2's AP-facing NVMe controller; no coprocessor firmware is executed.
//!
//! Evidence: Linux `drivers/nvme/host/apple.c` documents linear SQ doorbells
//! (0x2490c/0x24910), 64 shared tags, 128-byte NVMMU TCBs at bases
//! 0x28108/0x28110, invalidation/status 0x28118/0x28120, and BOOT_STATUS
//! 0x1300 = 0xde71ce55. NVMe 1.4 sections 3, 4, 5 and 6 define the base
//! registers, PRPs, admin commands, completions and NVM read/write/flush.
//! The 26A434 J473AP kernel drives an ANS2 with `nvme-linear-sq` through
//! `AppleANS3CGv2Controller` (`probe` 0x3dfa2f4), a subclass of
//! `AppleANS2CGv2Controller` (file 0x3de4954) whose `GetIOSQES`
//! (0x3de51d0) returns 6. The register accessors at 0x3ddee24..64 use ANS
//! `reg[3]`; its `nvme-interrupt-idx` selects the completion interrupt.
//! The controller's values must satisfy SPTM's NVMe validation, and cover
//! every register the drivers reach through restore, boot, sleep, wake and
//! shutdown.
//!
//! Stela's die choices: NVMe 1.3, one I/O queue, explicit linear geometry,
//! 4/16 KiB queue pages, fixed 4 KiB NVMMU data pages, 4 KiB LBAs, 1 MiB maximum transfer, the
//! host disk's volatile write cache, reported as the host's driver finds it, no
//! optional encryption/SHA/firmware-slot updates or asynchronous events.
//! The restore mode's boot-from-host image download is recorded, not run,
//! and so are the restore updater's NAND tunable tables. Identity
//! strings identify Stela rather than impersonating a physical Apple SSD.
//! The boot stage starts RTKit before handoff. NVMe itself uses MMIO; the
//! RTKit coprocessor lists one application endpoint, [`RTKIT_ENDPOINT`],
//! which carries no messages but must exist for the driver to start.
//! Undescribed registers, commands, fields and cryptographic effects refuse.
mod bfh;
mod commands;
mod copy;
mod debug;
mod die;

mod host;
mod namespace;
mod prp;
mod tunables;

use crate::Result;
use crate::inline_key;
use alloc::{
    boxed::Box,
    collections::{BTreeMap, VecDeque},
    format,
    string::String,
    vec,
    vec::Vec,
};
pub use bfh::BootImage;
#[cfg(test)]
use bfh::Identity;
use copy::Copied;
pub use copy::ReadCopy;
use namespace::Medium;
use prp::dma_error;
use stela_contracts::Refusal;
use stela_contracts::refuse;
use stela_drivers::block::BlockDevice;
use stela_soc_data::ans::Description;
use stela_soc_data::ans::{NVMMU_CONTROL, QueueGeometry};
use {crate::sart::Sart, stela_contracts::dma::Port};
use {
    stela_contracts::bus::Access, stela_contracts::bus::Direction, stela_contracts::bus::Lifecycle,
    stela_contracts::bus::LifecycleState, stela_contracts::bus::MmioDevice,
    stela_contracts::bus::MmioError, stela_contracts::bus::Width, stela_contracts::bus::refuse,
    stela_contracts::interrupt::AicInput, stela_contracts::interrupt::Wire,
};

const NAME: &str = "ans2";
/// The oldest commands a diagnostic lists.
const DIAGNOSTIC_COMMANDS: usize = 8;
/// Historical default software tag slots; configured dies use QueueGeometry.
#[cfg(test)]
pub const TAGS: u16 = 64;
/// Entries one queue may hold, CAP.MQES + 1. SPTM `INIT` (0x270c79d0)
/// requires the kernel's queue entries to equal `nvme-queue-entries` (64),
/// and `AppleANS3CGv2Controller::GetNVMeSPTMQueueEntries` (0x3db9270)
/// passes `min(MQES + 1, 257) - 1` (0x3de4aa8..ae0 with its 256 pending
/// commands, 0x3db9264), so this default's MQES is 64. The kernel sizes every
/// ring with that minimum (`SetNVMeProperties` 0x3db3850, read by
/// `CreateSubmissionQueue` 0x3ddd1dc and `CreateCompletionQueue`
/// 0x3ddd648): 65-entry admin and I/O rings, while tags stay below
/// [`TAGS`] (SPTM `MAP_PAGES` 0x270c8178).
#[cfg(test)]
pub const QUEUE_ENTRIES: u16 = 65;
/// Where SPTM finds the VDMA translation cache in ANS2 `reg[13]` when the
/// node carries `nvme-vdma-wa` (`nvme_bootstrap` 0x270c8cac..cdc maps the
/// bank and keeps it + 0x20000), and the cache's span: one 32-byte entry
/// per tag (`UNMAP_PAGES` 0x270c7c20..30 reads tag << 5).
pub const VDMA_CACHE_OFFSET: u64 = 0x2_0000;
/// Historical default span; placement uses QueueGeometry::vdma_bytes.
#[cfg(test)]
pub const VDMA_CACHE_BYTES: u64 = TAGS as u64 * VDMA_ENTRY_BYTES;
/// The device offset the board maps the VDMA cache window at, past the
/// NVMe registers.
pub const VDMA_CACHE: u64 = 0x10_0000;
const VDMA_ENTRY_BYTES: u64 = 32;
const CAP: u64 = 0;
const VS: u64 = 8;
const INTMS: u64 = 0xc;
const INTMC: u64 = 0x10;
const CC: u64 = 0x14;
const CSTS: u64 = 0x1c;
const AQA: u64 = 0x24;
const ASQ: u64 = 0x28;
const ACQ: u64 = 0x30;
const ADMIN_HEAD: u64 = 0x1004;
const IO_HEAD: u64 = 0x100c;
const IO_SQ: u64 = 0x1200;
const IO_CQ: u64 = 0x1208;
const TAG_STATUS: u64 = 0x28000;
const MAX_PENDING: u64 = 0x1210;
const BOOT_STATUS: u64 = 0x1300;
const MODESEL: u64 = 0x1304;
const BASE_CID: u64 = 0x1308;
const SLEEP_NOTIFY: u64 = 0x1314;
const AUTO_QUEUE: u64 = 0x13c8;
const LINEAR: u64 = 0x24908;
const ADMIN_SUBMIT: u64 = 0x2490c;
const IO_SUBMIT: u64 = 0x24910;
const NVMMU_CONTROL_REG: u64 = 0x28100;
const ADMIN_TCB: u64 = 0x28108;
const IO_TCB: u64 = 0x28110;
const TCB_INVALIDATE: u64 = 0x28118;
const TCB_STATUS: u64 = 0x28120;
const ENABLE: u32 = 1;
const SHUTDOWN: u32 = 3 << 14;
/// MODESEL values `PreCCEnableHook` writes before enabling: 0 regular, 4
/// in the restore environment (`AppleANS2NVMeController::PreCCEnableHook`
/// 0x3dbdfb8: 4 when `-restore` set its boot-from-host flag, 0x3dbb6b8).
const MODE_REGULAR: u32 = 0;
const MODE_RESTORE: u32 = 4;
const CONFIG_MASK: u32 = ENABLE | (0xf << 7) | SHUTDOWN | (0xf << 16) | (0xf << 20);
const ENTRY_SIZES: u32 = (6 << 16) | (4 << 20);
const READY: u32 = 1;
const SHUTDOWN_COMPLETE: u32 = 2 << 2;
const VERSION: u64 = 0x0001_0300;
const BOOT_OK: u64 = 0xde71_ce55;
const COMMAND_BYTES: u64 = 64;
const COMPLETION_BYTES: u64 = 16;
const TCB_BYTES: u64 = 128;
/// SPTM NVMMU TCB PRPs always describe 4 KiB pages, independent of CC.MPS.
const NVMMU_PAGE_BYTES: u64 = 4096;

/// ANS2's RTKit application endpoint. RTBuddy publishes it at roll call as
/// the service `ANS2Endpoint1` (`RTBuddyEndpointService::createName`
/// 0x4665818: role + "Endpoint" + endpoint - 0x1f).
/// `AppleANS2NVMeController::start` blocks until that service matches
/// (`waitForMatchingService` 0x3dbb308, before `WaitDeviceActive`), binds
/// it with no message handler (`initForOwner` 0x3dbb344) and only attaches
/// its own power-state action to it (`InitializePowerManagement`
/// 0x3dbf668), so no message crosses it in either direction.
pub const RTKIT_ENDPOINT: u8 = 0x20;

/// The coprocessor side of [`RTKIT_ENDPOINT`]: enabling it is accepted and
/// any message on it refuses.
#[derive(Debug, Default)]
pub struct Endpoint;

impl super::rtkit::Endpoint for Endpoint {
    fn check_reset(&self) -> Result<()> {
        Ok(())
    }
    fn reset(&mut self) -> Result<()> {
        Ok(())
    }
    fn start(&mut self, _outbox: &mut alloc::vec::Vec<u64>) -> Result<()> {
        Ok(())
    }
    fn receive(&mut self, _payload: u64, _outbox: &mut alloc::vec::Vec<u64>) -> Result<()> {
        Err(refuse!(
            "ANS2 RTKit endpoint 0x20 has no AP message protocol"
        ))
    }
}

#[derive(Clone, Copy, Default)]
struct Queue {
    sq: u64,
    cq: u64,
    sq_entries: u16,
    cq_entries: u16,
    head: u16,
    tail: u16,
    pending: u16,
    phase: bool,
    sq_enabled: bool,
    cq_enabled: bool,
    interrupt: bool,
}

impl Queue {
    fn acknowledge(&mut self, head: u16) -> Result<()> {
        if !self.cq_enabled || head >= self.cq_entries {
            return Err(refuse!("ANS CQ head outside an enabled queue"));
        }
        let consumed = (head + self.cq_entries - self.head) % self.cq_entries;
        let consumed = if consumed == 0 && self.pending == self.cq_entries {
            self.cq_entries
        } else {
            consumed
        };
        if consumed > self.pending {
            return Err(refuse!("ANS CQ head consumes completions not produced"));
        }
        self.pending -= consumed;
        self.head = head;
        Ok(())
    }
}

/// Last DMA submission, kept for an explicit refusal diagnostic. This
/// never grants keys or substitutes a successful completion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Submission {
    pub queue: usize,
    pub tag: u16,
    pub command: [u8; 64],
    pub tcb: [u8; 128],
    /// The tunnel request the command named, once it was read.
    pub tunnel: Option<[u8; 80]>,
}

/// A command whose host transfer runs, with what completes it.
struct Inflight {
    queue: usize,
    command: commands::Command,
    wait: commands::Wait,
}

/// A command whose completion awaits room in its completion queue.
struct Completed {
    queue: usize,
    command: commands::Command,
    outcome: commands::Outcome,
    /// Whether its host transfer completed after it executed: a read copy
    /// that fails then ends the command without a completion, as a failed
    /// copy in place there does; one that executed at once fails alone.
    after_host: bool,
}

pub struct Ans {
    lifecycle: LifecycleState,
    geometry: QueueGeometry,
    medium: Medium,
    config: u32,
    status: u32,
    interrupt_mask: u32,
    irq: Option<AicInput>,
    irq_level: bool,
    aqa: u32,
    queues: [Queue; 2],
    tcb_bases: [u64; 2],
    nvmmu_control: u16,
    valid: Vec<bool>,
    /// When each tag's command was submitted, in counter ticks.
    submitted: Vec<u64>,
    /// Submitted commands not yet executed, in order.
    pending: VecDeque<(usize, u16)>,
    /// Commands whose host transfers run ([`host::HostDisk`]), by host key.
    inflight: BTreeMap<u32, Inflight>,
    /// When to look for the host transfers, in counter ticks
    /// ([`host::HostDisk::next_look`]).
    look: u64,
    /// Executed commands, in the order they completed, whose completions
    /// await room in their queues.
    completed: VecDeque<Completed>,
    linear: bool,
    /// LINEAR as the boot chain hands the controller over, which a power-on
    /// after a power loss starts from again.
    boot_linear: bool,
    /// I/O queue attributes at 0x1210: 0's-based SQ entries in bits 15:0
    /// and CQ entries in bits 31:16, as AQA. Both writers size their rings
    /// one larger than the field: `AppleANS2CGv2Controller::SetupIOQARegister`
    /// (0x3de5a6c) passes ring entries - 1 through SPTM `IOQA` (0x270c7470),
    /// and `AppleANS2NVMeController::SetupIOQARegister` (0x3dc15d4) writes
    /// 64 for its 65-entry rings. None until programmed.
    queue_attributes: Option<u32>,
    modesel: u32,
    auto_queue: bool,
    shutdown: bool,
    /// NVMe feature state, indexed by the small standard feature IDs.
    features: [u32; 12],
    debug: debug::DebugService,
    timestamp: u64,
    os_build: [u8; 16],
    /// The host clocks of the last device-time tunnel (request type 6).
    device_time: Option<commands::DeviceTime>,
    /// The uncommitted boot-from-host download and the last image it
    /// committed, which survives controller resets.
    download: bfh::Download,
    boot_image: Option<BootImage>,
    /// NAND tunable tables accepted over the tunnel ([`tunables`]).
    tunables: tunables::Tables,
    last_submission: Option<Submission>,
    /// The inline key engine the SEP provisions (`inline_key`).
    inline: inline_key::Engine,
    /// Whether a completed read's copy to guest RAM is left to
    /// [`Self::take_copies`] ([`copy`]); otherwise it runs in place.
    defer_copies: bool,
    /// The read copies left since the last [`Self::take_copies`].
    copies: Vec<ReadCopy>,
}

impl Ans {
    /// Every feature's default: only Volatile Write Cache is nonzero,
    /// enabled where the host disk has one.
    fn default_features(medium: &Medium) -> [u32; 12] {
        let mut features = [0; 12];
        features[commands::VOLATILE_WRITE_CACHE] = u32::from(medium.device.write_cache());
        features
    }

    #[cfg(test)]
    pub fn new(
        backing: Box<dyn BlockDevice>,
        namespaces: &[Description],
        irq: Option<AicInput>,
    ) -> Result<Self> {
        Self::with_geometry(backing, namespaces, irq, QueueGeometry::DEFAULT)
    }

    pub fn with_geometry(
        backing: Box<dyn BlockDevice>,
        namespaces: &[Description],
        irq: Option<AicInput>,
        geometry: QueueGeometry,
    ) -> Result<Self> {
        let medium = Medium::new(backing, namespaces)?;
        let features = Self::default_features(&medium);
        Ok(Self {
            lifecycle: LifecycleState {
                powered: true,
                clocked: true,
                quiesced: false,
                reset: false,
                retention: false,
            },
            geometry,
            medium,
            config: 0,
            status: 0,
            interrupt_mask: 0,
            irq,
            irq_level: false,
            aqa: 0,
            queues: [Queue::default(); 2],
            tcb_bases: [0; 2],
            nvmmu_control: NVMMU_CONTROL,
            valid: vec![false; geometry.tag_slots() as usize],
            submitted: vec![0; geometry.tag_slots() as usize],
            pending: VecDeque::new(),
            inflight: BTreeMap::new(),
            look: 0,
            completed: VecDeque::new(),
            linear: false,
            boot_linear: false,
            queue_attributes: None,
            modesel: 0,
            auto_queue: false,
            shutdown: false,
            features,
            debug: debug::DebugService::default(),
            timestamp: 0,
            os_build: [0; 16],
            device_time: None,
            download: bfh::Download::default(),
            boot_image: None,
            tunables: tunables::Tables::default(),
            last_submission: None,
            inline: inline_key::Engine::default(),
            defer_copies: false,
            copies: Vec::new(),
        })
    }

    /// The identity of the last committed boot-from-host image.
    #[cfg(test)]
    pub fn boot_image(&self) -> Option<&BootImage> {
        self.boot_image.as_ref()
    }

    /// The identity of the last tunable table of each accepted kind,
    /// keyed by its tunnel request type.
    #[cfg(test)]
    pub fn tunable_tables(&self) -> &[(u32, Identity)] {
        self.tunables.recorded()
    }

    pub fn last_submission(&self) -> Option<&Submission> {
        self.last_submission.as_ref()
    }

    /// Whether a transfer at counter `now` would advance a command: one
    /// waits to execute or to complete, or host work runs and has completed
    /// or it is time to look for it.
    pub fn pending(&self, now: u64) -> bool {
        self.executing()
            && (self.shutdown
                || !self.pending.is_empty()
                || !self.completed.is_empty()
                || (self.host_work() && (now >= self.look || self.medium.device.posted())))
    }

    /// Whether the host disk runs anything: a command's transfers or its
    /// background discards.
    fn host_work(&self) -> bool {
        !self.inflight.is_empty() || self.medium.device.discarding()
    }

    /// When to look for the running host transfers, if any run: at once
    /// when one has completed.
    pub fn host_look(&self) -> Option<u64> {
        if !self.host_work() {
            return None;
        }
        Some(if self.medium.device.posted() {
            0
        } else {
            self.look
        })
    }

    /// The commands the controller owns at counter `now`, oldest first, each
    /// as queue:tag/stage/age in counter ticks: queued to execute, running
    /// its host transfer, its read data on its way to guest RAM (`copy`) or
    /// completed while its CQ is full. Then each CQ's
    /// unconsumed completions, the interrupt and the host disk: what tells
    /// a lost command from a lost completion or interrupt.
    pub fn diagnostic(&self, now: u64) -> String {
        let age = |tag: u16| now.saturating_sub(self.submitted[usize::from(tag)]);
        let mut owned: Vec<(u64, &str, usize, u16)> = self
            .pending
            .iter()
            .map(|&(queue, tag)| (age(tag), "queued", queue, tag))
            .chain(self.inflight.values().map(|run| {
                let tag = run.command.cid();
                (age(tag), "host", run.queue, tag)
            }))
            .chain(self.completed.iter().map(|done| {
                let tag = done.command.cid();
                let stage = if done.outcome.copy.is_some() {
                    "copy"
                } else {
                    "cq-full"
                };
                (age(tag), stage, done.queue, tag)
            }))
            .collect();
        owned.sort_unstable_by_key(|&(age, ..)| core::cmp::Reverse(age));
        let oldest = owned
            .iter()
            .take(DIAGNOSTIC_COMMANDS)
            .map(|(age, stage, queue, tag)| format!("{queue}:{tag:#x}/{stage}/{age}"))
            .collect::<Vec<_>>()
            .join(",");
        let queues = self
            .queues
            .iter()
            .map(|q| format!("{}/{}@{}-{}", q.pending, q.cq_entries, q.head, q.tail))
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "owned={} oldest=[{oldest}] cq=[{queues}] irq={} mask={:#x} look={:#x} executing={} \
             lifecycle={:?} host=[{}]",
            owned.len(),
            self.irq_level,
            self.interrupt_mask,
            self.look,
            self.executing(),
            self.lifecycle,
            self.medium.device.diagnostic(now),
        )
    }

    fn executing(&self) -> bool {
        self.lifecycle.powered
            && self.lifecycle.clocked
            && !self.lifecycle.quiesced
            && !self.lifecycle.reset
    }

    fn accessible(&self) -> Result<()> {
        if !self.lifecycle.powered {
            return Err(refuse!("ANS power is gated"));
        }
        if !self.lifecycle.clocked {
            return Err(refuse!("ANS clock is gated"));
        }
        if self.lifecycle.reset {
            return Err(refuse!("ANS reset is asserted"));
        }
        Ok(())
    }

    /// Logical blocks erased and not written since, which read as zeros.
    #[cfg(test)]
    pub fn erased_blocks(&self) -> u64 {
        self.medium.erased_blocks()
    }

    fn page(&self) -> u64 {
        4096 << ((self.config >> 7) & 0xf)
    }

    fn base(&self, offset: u64) -> Option<u64> {
        Some(match offset {
            ASQ => self.queues[0].sq,
            ACQ => self.queues[0].cq,
            IO_SQ => self.queues[1].sq,
            IO_CQ => self.queues[1].cq,
            ADMIN_TCB => self.tcb_bases[0],
            IO_TCB => self.tcb_bases[1],
            _ => return None,
        })
    }

    /// SPTM 26A434 VA 0xfffffff0270c7250/0x270c6fd0 programs the
    /// linear I/O queue bases; 0x270c7470 supplies the SQ/CQ entry counts.
    /// Auto management (CGv2 0x3de5ad8) activates that single queue without
    /// Create SQ/CQ commands. Existing manually created queues use the same
    /// submission, completion, bounds and interrupt machinery.
    fn refresh_auto_queue(&mut self) {
        if !self.auto_queue || self.config & ENABLE == 0 {
            return;
        }
        let Some(attributes) = self.queue_attributes else {
            return;
        };
        let sq_entries = (attributes & 0xffff) as u16 + 1;
        let cq_entries = (attributes >> 16) as u16 + 1;
        let q = &mut self.queues[1];
        if q.sq != 0 && q.cq != 0 && !q.sq_enabled && !q.cq_enabled {
            q.sq_entries = sq_entries;
            q.cq_entries = cq_entries;
            q.sq_enabled = true;
            q.cq_enabled = true;
            q.interrupt = true;
            q.phase = true;
        }
    }

    fn update_irq(&mut self, access: &mut Access<'_>) {
        let level = self.interrupt_mask & 1 == 0
            && self.queues.iter().any(|q| q.interrupt && q.pending != 0);
        if self.irq_level != level {
            self.irq_level = level;
            if let Some(irq) = self.irq {
                access.interrupts.set_level(Wire::Aic(irq), level);
            }
        }
    }

    /// The boot chain's hand-off under `/defaults nvme-iboot-sptm-security`:
    /// the boot stage has put the controller in linear submission mode, so
    /// the kernel never writes LINEAR (`AppleANS3CGv2Controller::EnableLinearSQ`
    /// 0x3db9064 returns at once with that flag) and SPTM, which also reads
    /// it, writes no linear register either.
    pub fn handed_off_linear(mut self) -> Self {
        self.linear = true;
        self.boot_linear = true;
        self
    }

    /// Unwrap inline keys with `engine`, which the SEP provisions.
    pub fn with_inline_engine(mut self, engine: inline_key::Engine) -> Self {
        self.inline = engine;
        self
    }

    /// Leave each completed read's copy to guest RAM to
    /// [`Self::take_copies`] from now on ([`copy`]).
    pub fn defer_copies(&mut self) {
        self.defer_copies = true;
    }

    /// The read copies the transfers since the last call left: each must
    /// run ([`ReadCopy::run`]) for its command to complete.
    pub fn take_copies(&mut self) -> impl Iterator<Item = ReadCopy> + '_ {
        self.copies.drain(..)
    }

    fn configure(&mut self, value: u32) -> Result<()> {
        if value & !CONFIG_MASK != 0 {
            return Err(refuse!("ANS CC fields without a model"));
        }
        if value & ENABLE != 0 {
            if value & (0xff << 16) != ENTRY_SIZES || !matches!((value >> 7) & 0xf, 0 | 2) {
                return Err(refuse!(
                    "ANS CC requires 64-byte SQEs, 16-byte CQEs and 4/16 KiB pages"
                ));
            }
            if self.config & ENABLE != 0 && (value ^ self.config) & !SHUTDOWN != 0 {
                return Err(refuse!("ANS CC changed while enabled"));
            }
            if self.config & ENABLE == 0 {
                let sq_entries = ((self.aqa & 0xfff) + 1) as u16;
                let cq_entries = (((self.aqa >> 16) & 0xfff) + 1) as u16;
                let page = 4096 << ((value >> 7) & 0xf);
                let queue = self.queues[0];
                if sq_entries > self.geometry.ring_entries()
                    || cq_entries > self.geometry.ring_entries()
                    || queue.sq == 0
                    || queue.cq == 0
                    || !queue.sq.is_multiple_of(page)
                    || !queue.cq.is_multiple_of(page)
                {
                    return Err(refuse!(
                        "ANS admin queues are missing, unaligned or too deep"
                    ));
                }
                self.queues[0] = Queue {
                    sq_entries,
                    cq_entries,
                    phase: true,
                    sq_enabled: true,
                    cq_enabled: true,
                    interrupt: true,
                    ..queue
                };
                self.status = READY;
            }
            // Normal (01b), abrupt (10b, `HandleAbruptShutdown` 0x3ddbb14)
            // and suspend (11b) notifications all complete once every
            // accepted command has and the backing has flushed: the die
            // keeps no volatile state that one would lose and another not.
            if value & SHUTDOWN != 0 && self.config & SHUTDOWN == 0 {
                self.shutdown = true;
            }
        } else {
            if !self.pending.is_empty() || !self.inflight.is_empty() || !self.completed.is_empty() {
                return Err(refuse!("ANS disable with unprocessed submissions"));
            }
            self.status = 0;
            self.queues = [Queue::default(); 2];
            self.valid.fill(false);
            self.shutdown = false;
            // NVMe 1.4 sections 5.11 and 7.3.2: a controller-level reset
            // discards an uncommitted image download and returns every
            // feature to its default.
            self.download = bfh::Download::default();
            self.features = Self::default_features(&self.medium);
        }
        self.config = value;
        self.refresh_auto_queue();
        Ok(())
    }

    /// Every command the controller owns, by queue and tag: submitted, its
    /// host transfer running, or its completion not yet published.
    fn outstanding(&self) -> impl Iterator<Item = (usize, u16)> + '_ {
        self.pending
            .iter()
            .copied()
            .chain(
                self.inflight
                    .values()
                    .map(|run| (run.queue, run.command.cid())),
            )
            .chain(
                self.completed
                    .iter()
                    .map(|done| (done.queue, done.command.cid())),
            )
    }

    fn submit(&mut self, queue: usize, tag: u16, now: u64) -> Result<()> {
        if self.status != READY || self.shutdown || !self.linear || !self.queues[queue].sq_enabled {
            return Err(refuse!(
                "ANS submission requires a ready controller and an enabled linear SQ"
            ));
        }
        if tag >= self.geometry.tag_slots()
            || tag >= self.queues[queue].sq_entries
            || self.valid[tag as usize]
        {
            return Err(refuse!(
                "ANS tag outside handoff/SQ capacity or still in use"
            ));
        }
        if self.tcb_bases[queue] == 0 {
            return Err(refuse!("ANS submission without an NVMMU TCB array"));
        }
        let limit = (queue == 1)
            .then_some(self.queue_attributes)
            .flatten()
            .map(|attributes| (attributes & 0xffff) as usize + 1);
        if limit
            .is_some_and(|limit| self.outstanding().filter(|&(q, _)| q == queue).count() >= limit)
        {
            return Err(refuse!(
                "ANS pending commands exceed the programmed queue limit"
            ));
        }
        self.valid[tag as usize] = true;
        self.submitted[tag as usize] = now;
        self.pending.push_back((queue, tag));
        Ok(())
    }

    /// Complete submitted commands before the next guest access. The NVMe
    /// engine reaches queues, TCBs, PRPs and data at physical addresses
    /// through `port`: the driver gives them no SART entries
    /// (`AllocateNVMeQueue` 0x3dbe804 and the request pool create their
    /// buffers without a mapper; `GetIOMappers` 0x3dbceb4). Only tunnel
    /// buffers, which the coprocessor reaches at raw addresses carried in the
    /// command (`AllocateTunnelBuffer` 0x3dbe83c), are checked against `sart`.
    /// NVMMU TCB PRPs are authoritative; their encryption and hash fields
    /// must be zero until those engines are modeled, apart from the
    /// CDW14/CDW15 copy the CG v2 driver places at TCB 0x38.
    /// Check a submitted command against its TCB and carry it out.
    fn start(
        &mut self,
        queue: usize,
        tag: u16,
        command: &mut commands::Command,
        tcb: &[u8; 128],
        port: &mut Port<'_, '_>,
        sart: &mut Sart,
    ) -> Result<commands::Executed> {
        // `NVMeCoastGuardSetTCBEntry` (CG v2, 0x3de4bd8..0x3de4be4)
        // copies CDW14 and CDW15 to TCB 0x38 and 0x3c; earlier
        // submissions leave them zero. It then copies the request's key
        // area to 0x40.
        let mirror = &tcb[56..64];
        if tcb[0] != 0 && tcb[0] != command.opcode()
            || u16::from(tcb[2]) != tag
            || tcb[3] != 0
            || mirror != [0; 8] && mirror != &command.0[56..64]
            || tcb[6..24].iter().chain(&tcb[40..56]).any(|&b| b != 0)
        {
            return Err(refuse!(
                "ANS NVMMU TCB opcode, tag, encryption or hash fields are not modeled"
            ));
        }
        // SPTM's MAP_PAGES writes CDW0 bits 12:8 to TCB byte 1
        // (`270c8130`). Bits 1:0 are the request's IODirection, a grant
        // of device writes (bit 0) and reads (bit 1); a read into an
        // in-out descriptor grants both, so the command's own direction
        // need only be within the grant. Bits 3:2 are its inline crypto
        // mode, which names the key in the key area: wrapped by the
        // SEP's inline engine, or raw (`inline_key`).
        let flags = tcb[1];
        let area = tcb[64..].try_into().unwrap();
        let key = match (flags & commands::CRYPTO) >> 2 {
            0 if tcb[64..].iter().all(|&b| b == 0) => None,
            inline_key::WRAPPED => Some(self.inline.unwrap(area)?),
            inline_key::RAW => Some(inline_key::InlineKey::raw(area)?),
            _ => {
                return Err(refuse!(
                    "ANS NVMMU inline crypto mode or key area is not modeled"
                ));
            }
        };
        command.validate()?;
        let direction = command.direction(queue)?;
        if flags & !(3 | commands::CRYPTO) != 0 || direction & !flags & 3 != 0 {
            return Err(refuse!(
                "ANS NVMMU TCB DMA permissions do not grant the command's direction"
            ));
        }
        if command.0[1] != 0 && command.0[1] != flags & commands::CRYPTO {
            return Err(refuse!(
                "ANS command and NVMMU TCB inline crypto modes differ"
            ));
        }
        if queue != 0 && matches!(command.opcode(), 1 | 2 | 5) && tcb[4..6] != command.0[48..50] {
            return Err(refuse!("ANS NVMMU length differs from the command"));
        }
        command.0[24..40].copy_from_slice(&tcb[24..40]);
        self.execute(queue, command, key, port, sart)
    }

    pub fn run(&mut self, port: &mut Port<'_, '_>, sart: &mut Sart) -> Result<(), MmioError> {
        if !self.executing() {
            return Err(refuse(
                NAME,
                ADMIN_SUBMIT,
                Width::Word,
                Direction::Write,
                "ANS DMA while PMGR inhibits execution",
            ));
        }
        let address = |base: u64, bytes| {
            base.checked_add(bytes)
                .ok_or(refuse!("ANS DMA address wraps"))
        };
        let refusal = |queue: usize, reason: Refusal| {
            refuse(
                NAME,
                if queue == 0 { ADMIN_SUBMIT } else { IO_SUBMIT },
                Width::Word,
                Direction::Write,
                reason,
            )
        };
        // Host transfers that completed end their commands.
        let now = port.access.now;
        for (key, done) in self.medium.device.poll(now) {
            let Some(inflight) = self.inflight.remove(&key) else {
                return Err(refusal(1, refuse!("ANS host transfer for no command")));
            };
            let queue = inflight.queue;
            let outcome = self
                .finish(inflight.wait, Some(done), port)
                .map_err(|reason| refusal(queue, reason))?;
            self.completed.push_back(Completed {
                queue,
                command: inflight.command,
                outcome,
                after_host: true,
            });
        }
        // Submitted commands execute in order; a host transfer runs while
        // the next ones execute.
        let mut refused = None;
        while let Some(&(queue, tag)) = self.pending.front() {
            let error = |reason: Refusal| refusal(queue, reason);
            let q = self.queues[queue];
            if !q.cq_enabled {
                return Err(error(refuse!("ANS completion queue is disabled")));
            }
            let mut raw = [0; 64];
            port.read(
                address(q.sq, u64::from(tag) * COMMAND_BYTES).map_err(error)?,
                &mut raw,
            )
            .map_err(dma_error)
            .map_err(error)?;
            let mut command = commands::Command(raw);
            if command.cid() != tag {
                return Err(error(refuse!(
                    "ANS SQ command ID differs from its linear tag"
                )));
            }
            let mut tcb = [0; 128];
            port.read(
                address(self.tcb_bases[queue], u64::from(tag) * TCB_BYTES).map_err(error)?,
                &mut tcb,
            )
            .map_err(dma_error)
            .map_err(error)?;
            self.last_submission = Some(Submission {
                queue,
                tag,
                command: command.0,
                tcb,
                tunnel: None,
            });
            let executed = match self.start(queue, tag, &mut command, &tcb, port, sart) {
                Ok(executed) => executed,
                Err(reason) => {
                    // A command the model cannot carry out fails alone: the
                    // guest gets its status and the queues go on.
                    self.pending.pop_front();
                    self.completed.push_back(Completed {
                        queue,
                        command,
                        outcome: commands::Outcome::unsupported(),
                        after_host: false,
                    });
                    refused.get_or_insert(error(reason));
                    continue;
                }
            };
            self.pending.pop_front();
            match executed {
                commands::Executed::Done(outcome) => self.completed.push_back(Completed {
                    queue,
                    command,
                    outcome,
                    after_host: false,
                }),
                commands::Executed::Host(wait) => {
                    let key = (queue as u32) << 16 | u32::from(tag);
                    self.inflight.insert(
                        key,
                        Inflight {
                            queue,
                            command,
                            wait,
                        },
                    );
                }
            }
        }
        if self.host_work() {
            self.look = self.medium.device.next_look(now);
        }
        // Completions publish in the order their commands completed, each
        // while its queue has room: queue backpressure leaves every
        // completed tag owned until the guest advances CQ head, and no
        // completion is overwritten.
        let mut index = 0;
        while index < self.completed.len() {
            let Completed {
                queue,
                ref command,
                ref outcome,
                after_host,
            } = self.completed[index];
            let error = |reason: Refusal| refusal(queue, reason);
            let q = self.queues[queue];
            if !q.cq_enabled {
                return Err(error(refuse!("ANS completion queue is disabled")));
            }
            if q.pending == q.cq_entries {
                index += 1;
                continue;
            }
            // A read whose data is still on its way holds back every
            // completion after it, which keeps their order as it was.
            if let Some(copy) = &outcome.copy {
                match copy.outcome() {
                    Copied::Running => break,
                    Copied::Done => {}
                    // As a copy in place that failed after its host
                    // transfer: the command ends without a completion and
                    // the transfer reports why.
                    Copied::Failed(reason) if after_host => {
                        self.completed.remove(index);
                        return Err(error(reason));
                    }
                    // As one that failed as it executed: the command fails
                    // alone, with the status of one the model cannot carry
                    // out, and the queues go on.
                    Copied::Failed(reason) => {
                        self.completed[index].outcome = commands::Outcome::unsupported();
                        refused.get_or_insert(error(reason));
                        continue;
                    }
                }
            }
            let tag = command.cid();
            let mut completion = [0; 16];
            completion[..4].copy_from_slice(&outcome.result.to_le_bytes());
            // Linear SQs have no advancing head; the consumed slot is tag.
            completion[8..10].copy_from_slice(&tag.to_le_bytes());
            completion[10..12].copy_from_slice(&(queue as u16).to_le_bytes());
            completion[12..14].copy_from_slice(&tag.to_le_bytes());
            completion[14..]
                .copy_from_slice(&((outcome.status << 1) | u16::from(q.phase)).to_le_bytes());
            port.write(
                address(q.cq, u64::from(q.tail) * COMPLETION_BYTES).map_err(error)?,
                &completion,
            )
            .map_err(dma_error)
            .map_err(error)?;
            let q = &mut self.queues[queue];
            q.pending += 1;
            q.tail += 1;
            if q.tail == q.cq_entries {
                q.tail = 0;
                q.phase = !q.phase;
            }
            self.completed.remove(index);
            self.update_irq(port.access);
        }
        // Discards still queued for a pause in transfers go to the disk
        // before the shutdown completes: its flush does not send them, and
        // the guest was told their blocks read as deallocated.
        if self.shutdown
            && self.pending.is_empty()
            && self.inflight.is_empty()
            && self.completed.is_empty()
            && !self.medium.device.discarding()
        {
            self.medium.device.flush().map_err(|_| {
                refuse(
                    NAME,
                    CC,
                    Width::Word,
                    Direction::Write,
                    "ANS backing flush failed during shutdown",
                )
            })?;
            self.shutdown = false;
            self.status = READY | SHUTDOWN_COMPLETE;
        }
        refused.map_or(Ok(()), Err)
    }
}

impl Ans {
    /// Whether the controller may lose power: its driver's shutdown
    /// notification completed (CSTS.SHST, which `run` sets only once every
    /// accepted command has completed and the backing flushed) and it owns
    /// no command or discard since.
    fn clean_for_power_loss(&self) -> Result<()> {
        if self.status & SHUTDOWN_COMPLETE == 0 || self.shutdown {
            return Err(refuse!(
                "ANS power loss before its driver's shutdown notification completed"
            ));
        }
        if self.outstanding().next().is_some() || self.medium.device.discarding() {
            return Err(refuse!(
                "ANS power loss with commands or discards outstanding"
            ));
        }
        Ok(())
    }

    /// A power loss after a clean shutdown: every register, queue, feature,
    /// tunable table and uncommitted download is gone, as on a controller the
    /// boot chain has just handed over. The medium and the committed boot
    /// image persist in NAND and LINEAR returns to the hand-off's value, so a
    /// power-on brings the controller up as at boot.
    fn lose_power(&mut self) {
        self.config = 0;
        self.status = 0;
        self.interrupt_mask = 0;
        self.aqa = 0;
        self.queues = [Queue::default(); 2];
        self.tcb_bases = [0; 2];
        self.nvmmu_control = NVMMU_CONTROL;
        self.valid.fill(false);
        self.submitted.fill(0);
        self.linear = self.boot_linear;
        self.queue_attributes = None;
        self.modesel = 0;
        self.auto_queue = false;
        self.shutdown = false;
        self.features = Self::default_features(&self.medium);
        self.device_time = None;
        self.download = bfh::Download::default();
        self.tunables = tunables::Tables::default();
        self.last_submission = None;
    }

    /// A power loss: the rail drops without retention, from on or from a
    /// retained power-down.
    fn loses_power(&self, proposed: LifecycleState) -> bool {
        (self.lifecycle.powered || self.lifecycle.retention)
            && !proposed.powered
            && !proposed.retention
    }
}

impl Lifecycle for Ans {
    fn check_lifecycle(&self, proposed: LifecycleState) -> Result<()> {
        if proposed.reset && !self.lifecycle.reset {
            // CC.EN reset is modeled separately in configure(). The PMGR
            // bank includes the protected linear/NVMMU handoff, loader and
            // NAND firmware state: CC reset does not prove their rail/reset
            // persistence. Keep all commands/completions owned on refusal.
            return Err(refuse!(
                "ANS physical reset requires protected register and NAND firmware restore semantics"
            ));
        }
        if self.loses_power(proposed) {
            // Keep all commands/completions owned on refusal.
            self.clean_for_power_loss()?;
        }
        Ok(())
    }
    fn set_lifecycle(&mut self, proposed: LifecycleState, access: &mut Access<'_>) {
        if self.loses_power(proposed) {
            self.lose_power();
        }
        self.lifecycle = proposed;
        self.update_irq(access);
    }
}

impl MmioDevice for Ans {
    fn lifecycle(&mut self) -> Option<&mut dyn Lifecycle> {
        Some(self)
    }
    fn name(&self) -> &'static str {
        NAME
    }
    fn read(&mut self, offset: u64, width: Width, _: &mut Access<'_>) -> Result<u64, MmioError> {
        self.accessible()
            .map_err(|reason| refuse(NAME, offset, width, Direction::Read, reason))?;
        if let Some(base) = self.base(offset & !7) {
            return match (offset % 8, width) {
                (0, Width::Double) => Ok(base),
                (0 | 4, Width::Word) => Ok((base >> ((offset % 8) * 8)) & 0xffff_ffff),
                _ => Err(refuse(
                    NAME,
                    offset,
                    width,
                    Direction::Read,
                    "ANS base register width or alignment",
                )),
            };
        }
        // Before invalidating a tag, SPTM's UNMAP_PAGES (0x270c7c18..30,
        // with nvme-vdma-wa) requires the tag's VDMA cache entry to hold no
        // DMA permissions (bits 9:8 of its first word, the TCB's direction
        // bits). The controller caches no translation once a command
        // completes, so that entry reads 0; every other word, and the entry
        // of a tag whose command is still pending, has no model.
        if (VDMA_CACHE..VDMA_CACHE + self.geometry.vdma_bytes()).contains(&offset) {
            let tag = (offset - VDMA_CACHE) / VDMA_ENTRY_BYTES;
            let pending = self.outstanding().any(|(_, t)| u64::from(t) == tag);
            if width != Width::Word || !offset.is_multiple_of(VDMA_ENTRY_BYTES) || pending {
                return Err(refuse(
                    NAME,
                    offset,
                    width,
                    Direction::Read,
                    "ANS VDMA cache word or pending entry without a model",
                ));
            }
            return Ok(0);
        }
        // SPTM 0x270c7c40 invalidates a tag then checks its three-bit
        // status in a four-bit lane (eight tags per word). The controller
        // has no cached translations after a synchronous DMA completes.
        if (TAG_STATUS..TAG_STATUS + self.geometry.status_bytes()).contains(&offset)
            && offset.is_multiple_of(4)
            && width == Width::Word
        {
            let first = ((offset - TAG_STATUS) * 2) as u16;
            return Ok(self
                .outstanding()
                .filter(|(_, tag)| (first..first + 8).contains(tag))
                .fold(0, |bits, (_, tag)| bits | (1 << ((tag - first) * 4))));
        }
        let value = match (offset, width) {
            (CAP, Width::Double) => self.geometry.capabilities(),
            (CAP, Width::Word) => self.geometry.capabilities() & u64::from(u32::MAX),
            (4, Width::Word) => self.geometry.capabilities() >> 32,
            (VS, Width::Word) => VERSION,
            (INTMS | INTMC, Width::Word) => u64::from(self.interrupt_mask),
            (CC, Width::Word) => u64::from(self.config),
            (CSTS, Width::Word) => u64::from(self.status),
            (AQA, Width::Word) => u64::from(self.aqa),
            (ASQ, Width::Double) => self.queues[0].sq,
            (ACQ, Width::Double) => self.queues[0].cq,
            (ADMIN_TCB, Width::Double) => self.tcb_bases[0],
            (IO_TCB, Width::Double) => self.tcb_bases[1],
            (BOOT_STATUS, Width::Word) => BOOT_OK,
            (BASE_CID, Width::Word) => 0, // Stela assigns tags starting at zero.
            // `SetupANS2SleepNotifier` (0x3dbc540) subscribes to system power
            // changes and tunnels sleep notices to the controller only when
            // bits 1:0 read 1. Stela's die takes no such notices.
            (SLEEP_NOTIFY, Width::Word) => 0,
            (MODESEL, Width::Word) => u64::from(self.modesel),
            (AUTO_QUEUE, Width::Word) => u64::from(self.auto_queue),
            (MAX_PENDING, Width::Word) => u64::from(self.queue_attributes.unwrap_or(0)),
            (LINEAR, Width::Word) => u64::from(self.linear),
            (NVMMU_CONTROL_REG, Width::Word) => u64::from(self.nvmmu_control),
            (TCB_STATUS, Width::Word) => 0, // No cached translations: invalidation completes synchronously.
            _ => {
                return Err(refuse(
                    NAME,
                    offset,
                    width,
                    Direction::Read,
                    "ANS register or width without a model",
                ));
            }
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
        self.accessible()
            .map_err(|reason| refuse(NAME, offset, width, Direction::Write, reason))?;
        if self.lifecycle.quiesced && matches!(offset, ADMIN_SUBMIT | IO_SUBMIT) {
            return Err(refuse(
                NAME,
                offset,
                width,
                Direction::Write,
                "ANS submission while PMGR quiesced",
            ));
        }
        let error = |reason: Refusal| refuse(NAME, offset, width, Direction::Write, reason);
        if let Some(old) = self.base(offset & !7) {
            let base = offset & !7;
            if self.config & ENABLE != 0 && !matches!(base, IO_SQ | IO_CQ) {
                return Err(error(refuse!("ANS queue/TCB base changed while enabled")));
            }
            if self.outstanding().next().is_some() || self.queues[1].pending != 0 {
                return Err(error(refuse!("ANS base changed with outstanding commands")));
            }
            // SPTM 0x270c88c4..0x270c88e8 and 0x270c76cc..0x270c7750
            // issue low/high 32-bit stores to TCB and queue addresses.
            let value = match (offset % 8, width) {
                (0, Width::Double) => value,
                (0 | 4, Width::Word) if value <= u64::from(u32::MAX) => {
                    let shift = (offset % 8) * 8;
                    (old & !(0xffff_ffff << shift)) | (value << shift)
                }
                _ => return Err(error(refuse!("ANS base register width or alignment"))),
            };
            if !value.is_multiple_of(4096) {
                return Err(error(refuse!("ANS queue/TCB base is not page-aligned")));
            }
            let bytes = match base {
                ADMIN_TCB | IO_TCB => u64::from(self.geometry.tag_slots()) * TCB_BYTES,
                ASQ | IO_SQ => u64::from(self.geometry.ring_entries()) * COMMAND_BYTES,
                ACQ | IO_CQ => u64::from(self.geometry.ring_entries()) * COMPLETION_BYTES,
                _ => unreachable!(),
            };
            if value.checked_add(bytes).is_none() {
                return Err(error(refuse!("ANS declared queue/TCB extent wraps")));
            }
            match base {
                ASQ => self.queues[0].sq = value,
                ACQ => self.queues[0].cq = value,
                IO_SQ => self.queues[1].sq = value,
                IO_CQ => self.queues[1].cq = value,
                ADMIN_TCB => self.tcb_bases[0] = value,
                IO_TCB => self.tcb_bases[1] = value,
                _ => unreachable!(),
            }
            self.refresh_auto_queue();
            return Ok(());
        }
        if width != Width::Word || value > u64::from(u32::MAX) {
            return Err(error(refuse!("ANS register requires a 32-bit word")));
        }
        let value = value as u32;
        match offset {
            CC => self.configure(value).map_err(error)?,
            INTMS | INTMC => {
                // NVMe's full 32-bit interrupt mask: W1S/W1C. Only
                // vector zero has a source in this single-vector die.
                if offset == INTMS {
                    self.interrupt_mask |= value;
                } else {
                    self.interrupt_mask &= !value;
                }
            }
            AQA if self.config & ENABLE == 0 && value & 0xf000_f000 == 0 => self.aqa = value,
            ADMIN_HEAD | IO_HEAD => {
                let head = u16::try_from(value)
                    .map_err(|_| error(refuse!("ANS CQ head exceeds 16 bits")))?;
                self.queues[usize::from(offset == IO_HEAD)]
                    .acknowledge(head)
                    .map_err(error)?;
            }
            LINEAR if value <= 1 && self.outstanding().next().is_none() => self.linear = value != 0,
            MAX_PENDING
                if value & 0xffff < u32::from(self.geometry.ring_entries())
                    && value >> 16 < u32::from(self.geometry.ring_entries()) =>
            {
                let q = self.queues[1];
                if q.sq_enabled && q.sq_entries != (value as u16 + 1)
                    || q.cq_enabled && q.cq_entries != ((value >> 16) as u16 + 1)
                    || self.outstanding().any(|(queue, _)| queue == 1)
                {
                    return Err(error(refuse!(
                        "ANS IOQA changes an active queue or outstanding submission"
                    )));
                }
                self.queue_attributes = Some(value)
            }
            MODESEL
                if matches!(value, MODE_REGULAR | MODE_RESTORE) && self.config & ENABLE == 0 =>
            {
                // A die choice: Stela's controller runs no host-loaded
                // firmware, so the restore mode serves the same commands;
                // its boot-from-host image download stays an unmodeled
                // admin command.
                self.modesel = value
            }
            AUTO_QUEUE if value <= 1 => self.auto_queue = value != 0,
            NVMMU_CONTROL_REG if value == u32::from(NVMMU_CONTROL) && self.config & ENABLE == 0 => {
                self.nvmmu_control = value as u16
            }
            ADMIN_SUBMIT | IO_SUBMIT => self
                .submit(
                    usize::from(offset == IO_SUBMIT),
                    u16::try_from(value).map_err(|_| error(refuse!("ANS tag exceeds 16 bits")))?,
                    access.now,
                )
                .map_err(error)?,
            TCB_INVALIDATE if value < u32::from(self.geometry.tag_slots()) => {
                if self.outstanding().any(|(_, tag)| u32::from(tag) == value) {
                    return Err(error(refuse!("ANS invalidation of a pending command")));
                }
                self.valid[value as usize] = false;
            }
            _ => {
                return Err(error(refuse!(
                    "ANS register, field or state without a model"
                )));
            }
        }
        self.refresh_auto_queue();
        self.update_irq(access);
        Ok(())
    }
}

#[cfg(test)]
mod tests;
