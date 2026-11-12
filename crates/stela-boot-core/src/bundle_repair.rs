//! The automatic repair of a missing Metal plugin, `StelaGPU.bundle` in
//! `/Library/GPUBundles` on the Data volume.
//!
//! The StelaGPU kext writes the plugin's files from the copy it carries, but
//! the kernel gives a kext no way to create a directory (`files.h`), so on a
//! freshly installed system, whose Data volume holds only Apple's empty
//! `/Library/GPUBundles`, it reports the plugin missing. Nothing else can
//! create them before the first boot: the Data volume is encrypted, Stela's
//! APFS code only reads, and the restore runs Apple's own programs. A root
//! process of a normal boot is refused too: `/Library/GPUBundles` is
//! restricted, and the GPU policy's SIP grant keeps filesystem protection on.
//!
//! So a normal boot that finds the plugin missing records
//! [`BUNDLE_REPAIR_PENDING`] in NVRAM at its stop and restarts. The loader
//! turns that into one single-user boot with filesystem protection off
//! (`--bundle-repair`) and records [`BUNDLE_REPAIR_ATTEMPTED`] before it
//! boots, so neither a failure nor a power loss in the middle repeats the
//! repair by itself. In that boot the engine types [`steps`] at the guest's
//! root console ([`Driver`]), as the manual remedy does: it mounts the Data
//! volume, creates the plugin's directories root-owned and 0755, marks them
//! restricted, syncs, remounts it read-only and restarts quickly
//! ([`REBOOT`]). The next boot's kext then creates the
//! files in place. Only mounting and the directories are required: a
//! failing optional step is logged and the repair goes on. A failed
//! required step still syncs and restarts, and a repair whose shell stops
//! answering restarts the PC, so the next normal boot always tries the kext
//! once. A boot that finds the plugin installed removes the record; one
//! that finds it missing after an attempt stops with the manual remedy
//! ([`decide`]).
//!
//! The manual remedy's `xattr -w com.apple.rootless ...` is not a step: it
//! failed with EPERM in every manual installation (runs 20261106-070021,
//! 20261106-083218, 20261107-060417, and in this repair's first run), even
//! with filesystem protection off, and Metal loaded the plugin each time
//! without it. Whether Metal needs the restricted flag is unproven: every
//! manual installation set it, so the repair tries it but does not depend
//! on it.
use alloc::{
    format,
    string::{String, ToString},
    vec::Vec,
};
use stela_loader::boot_selection::{
    BUNDLE_REPAIR, BUNDLE_REPAIR_ATTEMPTED, BUNDLE_REPAIR_PENDING, lookup,
};
use stela_soc_data::nor::nvram::{Partition, Variables};

/// The plugin's parent on the Data volume, as the kext looks for it.
pub const PLUGINS: &str = "/System/Volumes/Data/Library/GPUBundles";

/// The plugin's directories, parents first: the bundle layout
/// `cargo xtask guest` builds (`Contents/Info.plist`,
/// `Contents/MacOS/StelaGPU`), whose files the kext writes.
pub const DIRECTORIES: [&str; 3] = [
    "StelaGPU.bundle",
    "StelaGPU.bundle/Contents",
    "StelaGPU.bundle/Contents/MacOS",
];

/// What the kext found of the plugin (the gate's `BUNDLE` report).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Report {
    Current,
    Updated,
    Missing,
    Failed,
}

/// Which boot reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Boot {
    /// A multi-user boot, whose Metal clients need the plugin.
    Normal,
    /// A single-user boot an operator asked for (`--single-user`).
    SingleUser,
    /// The automatic repair's maintenance boot (`--bundle-repair`).
    Repair,
}

/// [`BUNDLE_REPAIR`] as NVRAM holds it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Record {
    None,
    Pending,
    Attempted,
    /// A value Stela never writes.
    Other,
}

impl Record {
    pub fn read(variables: &Variables) -> Self {
        match lookup(variables, BUNDLE_REPAIR)
            .map(|value| value.strip_suffix(b"\0").unwrap_or(value))
        {
            None => Self::None,
            Some(BUNDLE_REPAIR_PENDING) => Self::Pending,
            Some(BUNDLE_REPAIR_ATTEMPTED) => Self::Attempted,
            Some(_) => Self::Other,
        }
    }
}

/// What NVRAM's record becomes at the stop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Change {
    /// Ask the next boot to repair.
    Request,
    /// The plugin is installed: forget any repair.
    Forget,
}

impl Change {
    /// Apply to `variables`; whether they changed, when the bank must be
    /// saved.
    pub fn apply(self, variables: &mut Variables) -> Result<bool, String> {
        match self {
            Self::Request => {
                if Record::read(variables) == Record::Pending {
                    return Ok(false);
                }
                variables
                    .set(Partition::System, BUNDLE_REPAIR, BUNDLE_REPAIR_PENDING)
                    .map_err(|error| format!("{error:?}"))?;
                Ok(true)
            }
            Self::Forget => {
                let mut changed = false;
                for partition in [Partition::System, Partition::Common] {
                    changed |= variables.remove(partition, BUNDLE_REPAIR).is_some();
                }
                Ok(changed)
            }
        }
    }
}

/// What the engine does with a report.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decision {
    /// The plugin is in place: boot on, applying the change at the stop.
    Continue(Option<Change>),
    /// A single-user boot logs the report; its console may install it.
    Log,
    /// Record the request and restart into the repair.
    Repair,
    /// Stop with the manual remedy; `attempted` when the automatic repair
    /// already ran and did not help, or cannot run.
    StopMissing { attempted: bool },
    /// The kext failed to write the plugin; its lines say why.
    StopFailed,
}

/// Decide what `report` means for `boot`, given NVRAM's `record` and
/// whether this boot can write NVRAM at its stop (`persistent`).
pub fn decide(report: Report, boot: Boot, record: Record, persistent: bool) -> Decision {
    match (report, boot) {
        (Report::Current | Report::Updated, _) => {
            Decision::Continue((record != Record::None).then_some(Change::Forget))
        }
        (Report::Missing | Report::Failed, Boot::SingleUser | Boot::Repair) => Decision::Log,
        (Report::Failed, Boot::Normal) => Decision::StopFailed,
        (Report::Missing, Boot::Normal) => match record {
            Record::None | Record::Other if persistent => Decision::Repair,
            // A pending request this boot did not take means NVRAM was not
            // read for it; another request would loop.
            Record::None | Record::Other | Record::Pending => {
                Decision::StopMissing { attempted: false }
            }
            Record::Attempted => Decision::StopMissing { attempted: true },
        },
    }
}

/// One console command and the seconds it may take.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Step {
    pub command: String,
    pub seconds: u64,
    /// A failure ends the repair; otherwise it is logged and the next step
    /// goes on.
    pub required: bool,
}

/// The repair's commands, in order, each answered by its exit status.
pub fn steps() -> Vec<Step> {
    let directories = DIRECTORIES
        .iter()
        .map(|directory| format!("{PLUGINS}/{directory}"))
        .collect::<Vec<_>>()
        .join(" ");
    let step = |command: String, seconds| Step {
        command,
        seconds,
        required: true,
    };
    let optional = |command: String, seconds| Step {
        command,
        seconds,
        required: false,
    };
    Vec::from([
        // The Data volume, as the single-user banner says for a machine
        // with a SEP.
        step("/sbin/mount -P 1".into(), 120),
        step("/usr/libexec/init_data_protection".into(), 180),
        step("/sbin/mount -P 2".into(), 180),
        step(format!("/bin/test -d {PLUGINS}"), 30),
        // Each named last, so `-m` gives each its mode.
        step(format!("/bin/mkdir -p -m 0755 {directories}"), 30),
        step(format!("/usr/sbin/chown root:wheel {directories}"), 30),
        optional(format!("/usr/bin/chflags restricted {directories}"), 30),
        step("/bin/sync".into(), 120),
        // Commits the volume and marks it clean, so the quick restart below
        // leaves nothing for the next mount to replay. Optional: the sync
        // already put every change on disk, and a busy volume refuses it.
        optional("/sbin/mount -u -r /System/Volumes/Data".into(), 60),
    ])
}

/// The command that ends the repair; the guest's restart restarts the PC.
///
/// `-q` is RB_QUICK: XNU's `reboot_kernel` then only syncs and goes to
/// `PEHaltRestart`, skipping the process kill and `vfs_unmountall`. The
/// power tree's shutdown still runs there, so the NVMe driver still sends
/// its shutdown notification and the controller flushes. The full path
/// stalled in single-user mode: run 20261111-150448 stopped in
/// `vfs_unmountall(system)` at `unmount(0) /System/Volumes/xarts`, before
/// APFS began that unmount, until the 300 s bound ran out.
pub const REBOOT: &str = "/sbin/reboot -q";
/// What ends a repair whose required step failed: what was written stays.
pub const SYNC_AND_REBOOT: &str = "/bin/sync; /sbin/reboot -q";
/// The line the single-user shell's banner starts with.
pub const BANNER: &str = "*** Single-user boot ***";
/// How long the single-user banner may take to appear, from the start.
pub const BANNER_SECONDS: u64 = 1800;
/// How long to wait after the banner before the first command: launchd
/// starts the shell after it, and input before the shell reads may be
/// flushed.
pub const SETTLE_SECONDS: u64 = 5;
/// How long the guest may take to restart after [`REBOOT`] in all.
pub const RESTART_SECONDS: u64 = 120;
/// How long the guest's console may stay silent after [`REBOOT`]: the
/// restart path logs each stage, and a stalled one prints nothing more.
pub const QUIET_SECONDS: u64 = 30;

/// Why the repair failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Failure {
    /// No single-user banner in time.
    NoShell,
    /// Step `index` printed no exit status in time.
    Timeout { index: usize },
    /// Step `index` exited with `status`.
    Exit { index: usize, status: u32 },
    /// The guest did not restart after the repair.
    NoRestart,
}

impl core::fmt::Display for Failure {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NoShell => write!(f, "no single-user shell within {BANNER_SECONDS} s"),
            Self::Timeout { index } => {
                write!(f, "step {} printed no exit status in time", index + 1)
            }
            Self::Exit { index, status } => {
                write!(f, "step {} exited with status {status}", index + 1)
            }
            Self::NoRestart => write!(
                f,
                "macOS did not restart (console silent for {QUIET_SECONDS} s, or over {RESTART_SECONDS} s in all)"
            ),
        }
    }
}

/// What the driver did, for the log.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Event {
    Shell,
    Sent {
        index: usize,
        command: String,
    },
    Passed {
        index: usize,
    },
    /// An optional step failed; the repair goes on.
    Skipped {
        index: usize,
        status: u32,
    },
    /// The restart was typed; `repaired` when every required step passed.
    Rebooting {
        repaired: bool,
    },
    Failed(Failure),
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum State {
    Banner {
        deadline: u64,
    },
    Settle {
        until: u64,
    },
    Step {
        index: usize,
        deadline: u64,
    },
    /// `quiet` moves with each console line, never past `deadline`.
    Rebooting {
        quiet: u64,
        deadline: u64,
    },
    Failed,
}

/// Types [`steps`] at the guest's console, one at a time, each followed by
/// an `echo` of its exit status under a marker, and goes on only when that
/// line says 0. Times are milliseconds on any monotonic clock.
pub struct Driver {
    steps: Vec<Step>,
    state: State,
    input: Vec<u8>,
    sent: usize,
    events: Vec<Event>,
}

const MILLIS: u64 = 1000;

impl Driver {
    pub fn new(steps: Vec<Step>, now: u64) -> Self {
        Self {
            steps,
            state: State::Banner {
                deadline: now + BANNER_SECONDS * MILLIS,
            },
            input: Vec::new(),
            sent: 0,
            events: Vec::new(),
        }
    }

    fn marker(index: usize) -> String {
        format!("STELA-REPAIR-{}-rc=", index + 1)
    }

    fn send(&mut self, index: usize, now: u64) {
        let step = &self.steps[index];
        let line = format!("{}; echo {}$?\n", step.command, Self::marker(index));
        self.input.extend_from_slice(line.as_bytes());
        self.events.push(Event::Sent {
            index,
            command: step.command.clone(),
        });
        self.state = State::Step {
            index,
            deadline: now + step.seconds * MILLIS,
        };
    }

    fn fail(&mut self, failure: Failure) {
        self.events.push(Event::Failed(failure));
        self.state = State::Failed;
    }

    /// A complete line the guest's console printed.
    pub fn line(&mut self, text: &str, now: u64) {
        match self.state {
            State::Banner { .. } if text.contains(BANNER) => {
                self.events.push(Event::Shell);
                self.state = State::Settle {
                    until: now + SETTLE_SECONDS * MILLIS,
                };
            }
            State::Step { index, .. } => {
                let marker = Self::marker(index);
                // The shell's echo of the command holds the marker before
                // `$?`; only the `echo`'s output has digits after it.
                let Some(at) = text.find(&marker) else {
                    return;
                };
                let digits: String = text[at + marker.len()..]
                    .chars()
                    .take_while(char::is_ascii_digit)
                    .collect();
                let Ok(status) = digits.parse::<u32>() else {
                    return;
                };
                if status == 0 {
                    self.events.push(Event::Passed { index });
                } else if self.steps[index].required {
                    // The shell answers: keep what was written and restart,
                    // so the next boot tries the kext.
                    self.events
                        .push(Event::Failed(Failure::Exit { index, status }));
                    return self.reboot(SYNC_AND_REBOOT, false, now);
                } else {
                    self.events.push(Event::Skipped { index, status });
                }
                self.advance(index + 1, now);
            }
            State::Rebooting { deadline, .. } => {
                self.state = State::Rebooting {
                    quiet: deadline.min(now + QUIET_SECONDS * MILLIS),
                    deadline,
                };
            }
            _ => {}
        }
    }

    /// Send step `index`, or after the last, the restart.
    fn advance(&mut self, index: usize, now: u64) {
        if index < self.steps.len() {
            return self.send(index, now);
        }
        self.reboot(REBOOT, true, now);
    }

    fn reboot(&mut self, command: &str, repaired: bool, now: u64) {
        self.input.extend_from_slice(command.as_bytes());
        self.input.push(b'\n');
        self.events.push(Event::Rebooting { repaired });
        self.state = State::Rebooting {
            quiet: now + QUIET_SECONDS * MILLIS,
            deadline: now + RESTART_SECONDS * MILLIS,
        };
    }

    /// Advance the clock: the settle's end and every deadline.
    pub fn tick(&mut self, now: u64) {
        match self.state {
            State::Banner { deadline } if now >= deadline => self.fail(Failure::NoShell),
            State::Settle { until } if now >= until => self.advance(0, now),
            State::Step { index, deadline } if now >= deadline => {
                self.fail(Failure::Timeout { index })
            }
            State::Rebooting { quiet, .. } if now >= quiet => self.fail(Failure::NoRestart),
            _ => {}
        }
    }

    /// The next byte to type, if any. A failed repair types nothing more.
    pub fn next_byte(&mut self) -> Option<u8> {
        if self.state == State::Failed {
            return None;
        }
        let byte = *self.input.get(self.sent)?;
        self.sent += 1;
        if self.sent == self.input.len() {
            self.input.clear();
            self.sent = 0;
        }
        Some(byte)
    }

    /// What happened since the last call.
    pub fn take_events(&mut self) -> Vec<Event> {
        core::mem::take(&mut self.events)
    }

    /// Whether the repair stopped without a guest restart on its way: the
    /// shell did not answer or macOS did not restart. The engine then
    /// restarts the PC itself.
    pub fn failed(&self) -> bool {
        self.state == State::Failed
    }
}

/// The log line for `event`.
pub fn describe(event: &Event) -> String {
    match event {
        Event::Shell => "BUNDLE REPAIR single-user shell up".to_string(),
        Event::Sent { index, command } => {
            format!("BUNDLE REPAIR step {} sent: {command}", index + 1)
        }
        Event::Passed { index } => format!("BUNDLE REPAIR step {} ok", index + 1),
        Event::Skipped { index, status } => format!(
            "BUNDLE REPAIR step {} exited with status {status}; optional, going on",
            index + 1
        ),
        Event::Rebooting { repaired: true } => {
            "BUNDLE REPAIR directories created; restarting".to_string()
        }
        Event::Rebooting { repaired: false } => {
            "BUNDLE REPAIR syncing and restarting; the next boot tries the kext once".to_string()
        }
        Event::Failed(failure) => format!("BUNDLE REPAIR failed: {failure}"),
    }
}

#[cfg(test)]
mod tests;
