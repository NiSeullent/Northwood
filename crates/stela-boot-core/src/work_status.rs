//! What the boot screen says when the engine's owned work stacks, its
//! service clock or the disks' start are unavailable. The screen is the only
//! record before the boot log on the USB opens again, so each line names the
//! cause and, where the user can act, the firmware setting by its usual names.
use alloc::{format, string::String};

/// Why the boot processor's owned work stacks could not be prepared, by what
/// a user could check or report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cause {
    /// The loader's records of the stacks, processors or pages differ from
    /// what the engine finds: a Stela fault, not a setting.
    Records,
    /// The engine is not running on the processor the firmware booted.
    Processor,
    /// The time-stamp counter's rate was not measured.
    Clock,
    /// The stacks are not mapped as the loader left them.
    Mapping,
    /// The firmware's memory-type ranges (MTRRs) leave the stacks' memory
    /// other than write-back.
    Cache,
    /// Stela's own code is not kept where the firmware loaded it.
    Code,
    /// The boot processor refused to run a continuation on the stacks.
    Continuation,
    /// Memory for the stacks' records ran out.
    Memory,
}

/// The reason line for `cause`, `detail` naming the refusal the log has.
pub fn reason(cause: Cause, detail: &str) -> String {
    let what = match cause {
        Cause::Records => "the loader's records of its work stacks are not intact (a Stela fault)",
        Cause::Processor => "the engine is not on the processor the firmware booted",
        Cause::Clock => "the processor's time-stamp counter rate was not measured",
        Cause::Mapping => "the work stacks are not mapped as the loader left them",
        Cause::Cache => {
            "the firmware's memory type ranges (MTRRs) do not make the work stacks' memory \
             write-back; a firmware (BIOS) update may correct them"
        }
        Cause::Code => "Stela's code is not kept where the firmware loaded it",
        Cause::Continuation => "the boot processor refused to run on the work stacks",
        Cause::Memory => "there was no memory left for the work stacks' records",
    };
    format!("{what} ({detail})")
}

/// The screen line for a boot whose disks and USB devices start on the boot
/// processor's own stack because the owned work stacks are unavailable for
/// `reason`: each device keeps its own start limit, but the firmware's ACPI
/// methods and the GPU need those stacks.
pub fn foreground(reason: &str) -> String {
    format!(
        "Starting the disks and USB devices without Stela's work stacks, so without the \
         firmware's ACPI methods and without a GPU: {reason}"
    )
}

/// The screen line for a GPU left unstarted because the owned work stacks are
/// unavailable for `reason`.
pub fn gpu_without_work(reason: &str) -> String {
    format!("the GPU needs Stela's work stacks, which this PC did not allow: {reason}")
}

/// The screen line for a GPU left unstarted because no steady service clock
/// is installed: `invariant_tsc` says whether the processor's time-stamp
/// counter runs at one rate in every power state, `detail` names the refusal.
pub fn gpu_without_clock(invariant_tsc: bool, detail: &str) -> String {
    if invariant_tsc {
        format!("the GPU needs a steady clock and the processor's counter was refused ({detail})")
    } else {
        format!(
            "the GPU needs a steady clock: this processor's time-stamp counter changes rate and \
             the HPET is missing; enable HPET (High Precision Event Timer) in the firmware setup \
             ({detail})"
        )
    }
}

/// The fail line for a disk and USB start that did not complete: `timed_out`
/// says whether it outran the whole start's limit of `limit_s` seconds,
/// which only a stall outside every device's own start limit reaches.
pub fn devices_failed(timed_out: bool, limit_s: u64, detail: &str) -> String {
    if timed_out {
        format!(
            "the disks and USB devices did not finish starting within {limit_s} s, outside \
             every device's own limit ({detail})"
        )
    } else {
        format!("the disks and USB devices could not start on Stela's work stacks ({detail})")
    }
}

/// The fail line for a boot whose loader prepared no PCI device access.
pub const NO_DEVICE_ACCESS: &str = "the loader prepared no PCI device access, so no disk can be \
     opened: the firmware has no valid ACPI MCFG table or no memory below 4 GiB for device DMA";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_cause_names_itself_and_its_detail() {
        for cause in [
            Cause::Records,
            Cause::Processor,
            Cause::Clock,
            Cause::Mapping,
            Cause::Cache,
            Cause::Code,
            Cause::Continuation,
            Cause::Memory,
        ] {
            let line = reason(cause, "Work(TimedOut)");
            assert!(line.ends_with(" (Work(TimedOut))"), "{line}");
            assert!(!line.contains("owned continuation"), "{line}");
        }
        assert!(reason(Cause::Cache, "x").contains("MTRRs"));
        assert!(reason(Cause::Cache, "x").contains("firmware (BIOS) update"));
    }

    #[test]
    fn the_foreground_start_says_what_it_lacks() {
        let line = foreground("the engine is not on the processor the firmware booted (x)");
        assert!(
            line.contains("without the firmware's ACPI methods"),
            "{line}"
        );
        assert!(line.contains("without a GPU"), "{line}");
        assert!(line.ends_with("booted (x)"), "{line}");
        let gpu = gpu_without_work("r");
        assert!(
            gpu.starts_with("the GPU needs Stela's work stacks"),
            "{gpu}"
        );
        assert!(gpu.ends_with(": r"), "{gpu}");
    }

    #[test]
    fn a_missing_clock_names_the_firmware_setting_only_when_it_helps() {
        let varying = gpu_without_clock(false, "Unavailable");
        assert!(
            varying.contains("enable HPET (High Precision Event Timer)"),
            "{varying}"
        );
        let steady = gpu_without_clock(true, "Unavailable");
        assert!(!steady.contains("HPET"), "{steady}");
        assert!(steady.ends_with("(Unavailable)"), "{steady}");
    }

    #[test]
    fn a_failed_start_says_whether_it_stalled() {
        let stalled = devices_failed(true, 300, "Work(TimedOut)");
        assert!(stalled.contains("within 300 s"), "{stalled}");
        let refused = devices_failed(false, 300, "Work(Busy)");
        assert!(refused.contains("could not start"), "{refused}");
        assert!(refused.ends_with("(Work(Busy))"), "{refused}");
        assert!(NO_DEVICE_ACCESS.contains("ACPI MCFG"));
    }
}
