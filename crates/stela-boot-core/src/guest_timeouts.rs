//! XNU machine timeouts as boot-args give them.
//!
//! An `ml-timeout-NAME=VALUE` boot-arg is in picoseconds, whatever unit the
//! timeout is stored in (`osfmk/kern/startup.h`, "Machine Timeouts are
//! ALWAYS in picoseconds in the device tree or boot-args"). XNU converts an
//! overriding value to the stored unit, timebase ticks for `debug-ack`, and
//! keeps a default as it is: 26B123's `debug-ack` default is 1,440,000
//! ticks, 60 ms at 24 MHz.

//!
//! XNU also multiplies each timeout, its default or an override, by
//! `ml-timeout-NAME-scale` and by `ml-timeout-global-scale` (integers,
//! default 1; a product that overflows saturates), or by `NAME-scale` and
//! `global-scale` in `/machine-timeouts` or `/chosen/machine-timeouts`.
//! 26B123's kernel declares these timeouts: `lock` (6,000,000 ticks,
//! 250 ms), `ticket-lock` (3,000,000, 125 ms), `lock-panic` (12,582,912,
//! the mutex interlock's 0.5 s), `debug-ack`, `dtrace-spin-threshold` and
//! four `report-`/`trace-phy-read`/`write-delay` (default 0, off). Its
//! RELEASE build has no interrupt-masked or preemption-disable timeout.

use alloc::{format, string::String};

const PICOSECONDS_PER_SECOND: u64 = 1_000_000_000_000;

/// `seconds` as a machine timeout boot-arg's value.
pub const fn picoseconds(seconds: u64) -> u64 {
    seconds.saturating_mul(PICOSECONDS_PER_SECOND)
}

/// How long XNU's debugger cross-call waits for every PE (`debug-ack`),
/// in seconds. XNU's default, 60 ms, is short for PEs that answer at their
/// next exit. It must end well before the watchdog mechanism a panic arms
/// ([`WATCHDOG_MECHANISM_SECONDS`]): a PE that never answers (one parked
/// in SPTM because another CPU owns the panic) holds the panicking PE in
/// this wait, and only after it does XNU count PEs in SPTM as answered
/// ("found CPU %d in SPTM") and print the panic. A wait as long as the
/// mechanism's lets the reset come first, with no panic text.
pub const DEBUG_ACK_SECONDS: u64 = 2;

/// The watchdog mechanism's timeout: `AppleARMWatchdogTimer::
/// _armWatchdogMechanism` arms uptime plus 10 s, rounded down to whole
/// seconds, so the reset may come as little as 9 s after the arm.
pub const WATCHDOG_MECHANISM_SECONDS: u64 = 10;

/// The timeouts by which one core times another's lock hold.
pub const LOCK_TIMEOUTS: [&str; 3] = ["ticket-lock", "lock", "lock-panic"];

/// The modeled chips' performance core clock: Apple M2's (and M2 Ultra's)
/// 3.49 GHz, the speed XNU's lock timeouts assume of a holder.
pub const MODELED_CORE_HZ: u64 = 3_490_000_000;

/// The largest scale: 1 s for a ticket spinlock.
pub const MAX_LOCK_SCALE: u64 = 8;

/// The factor the lock timeouts get on a host whose cores run at
/// `core_hz`, as [`measure_core_hz`] measures it: how many times slower
/// than the modeled chip's they are, rounded up, so 1 on a host at least
/// as fast. A translated holder runs its critical section that much
/// slower; a holder the host stops is the count's to hold (`stela_engine`
/// `clock`). A rate within 5% of a step takes the lower one, so that the
/// measurement's noise does not move a host between two scales. An
/// unmeasured rate changes nothing.
pub fn lock_scale(core_hz: u64) -> u64 {
    if core_hz == 0 {
        return 1;
    }
    MODELED_CORE_HZ
        .div_ceil(core_hz.saturating_add(core_hz / 20))
        .clamp(1, MAX_LOCK_SCALE)
}

/// The additions one timed run of the reference loop makes, each waiting
/// for the one before: 2,000,000 core cycles, 0.5 ms at 4 GHz.
pub const REFERENCE_ADDS: u64 = 2_000_000;

/// Timed runs of the reference loop; the fastest counts.
pub const REFERENCE_RUNS: u32 = 8;

/// The core clock that `adds` dependent additions taking `ticks` of a TSC
/// at `tsc_hz` show: one addition a cycle, as on every x86-64 core.
pub fn core_hz(adds: u64, ticks: u64, tsc_hz: u64) -> u64 {
    if ticks == 0 {
        return 0;
    }
    u64::try_from(u128::from(adds) * u128::from(tsc_hz) / u128::from(ticks)).unwrap_or(u64::MAX)
}

/// The clock this core sustains, measured against its TSC at `tsc_hz`:
/// the fastest of [`REFERENCE_RUNS`] runs of [`REFERENCE_ADDS`] dependent
/// additions, about 4 ms in all. Unlike the TSC's nominal rate it shows
/// the core's turbo or power-saving clock and, under a hypervisor, time
/// the host gives other work. The fastest run leaves out interrupts and
/// the clock's ramp, so it is stable to a few percent from boot to boot.
/// It does not see IPC, which translated code's speed also follows.
#[cfg(target_arch = "x86_64")]
pub fn measure_core_hz(tsc_hz: u64) -> u64 {
    let ticks = (0..REFERENCE_RUNS)
        .map(|_| {
            let start = fenced_tsc();
            dependent_adds(REFERENCE_ADDS / 8);
            fenced_tsc().wrapping_sub(start)
        })
        .min()
        .unwrap_or(0);
    core_hz(REFERENCE_ADDS, ticks, tsc_hz)
}

/// The TSC, read after every earlier instruction and before every later
/// one.
#[cfg(target_arch = "x86_64")]
fn fenced_tsc() -> u64 {
    use core::arch::x86_64::{_mm_lfence, _rdtsc};
    // SAFETY: LFENCE and RDTSC are baseline x86-64 instructions that touch
    // no memory.
    #[allow(unused_unsafe)]
    unsafe {
        _mm_lfence();
        let tsc = _rdtsc();
        _mm_lfence();
        tsc
    }
}

/// `iterations` times eight additions, each needing the one before.
#[cfg(target_arch = "x86_64")]
fn dependent_adds(iterations: u64) {
    let mut sum = 0u64;
    // SAFETY: register arithmetic only.
    unsafe {
        core::arch::asm!(
            "2:",
            "add {sum}, {left}",
            "add {sum}, {left}",
            "add {sum}, {left}",
            "add {sum}, {left}",
            "add {sum}, {left}",
            "add {sum}, {left}",
            "add {sum}, {left}",
            "add {sum}, {left}",
            "dec {left}",
            "jnz 2b",
            sum = inout(reg) sum,
            left = inout(reg) iterations => _,
            options(nomem, nostack),
        );
    }
    core::hint::black_box(sum);
}

/// The boot-args that scale the lock timeouts by `scale`, each with a
/// leading space; none at 1, which leaves XNU's defaults.
pub fn lock_scale_arguments(scale: u64) -> String {
    if scale <= 1 {
        return String::new();
    }
    LOCK_TIMEOUTS
        .iter()
        .map(|name| format!(" ml-timeout-{name}-scale={scale}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_host_at_least_as_fast_as_the_modeled_chip_keeps_the_timeouts() {
        assert_eq!(lock_scale(4_700_041_194), 1);
        assert_eq!(lock_scale(MODELED_CORE_HZ), 1);
        assert_eq!(lock_scale_arguments(1), "");
        // Unmeasured: unchanged.
        assert_eq!(lock_scale(0), 1);
    }

    #[test]
    fn a_slower_host_scales_the_lock_timeouts_by_its_deficit_rounded_up() {
        assert_eq!(lock_scale(3_000_000_000), 2);
        assert_eq!(lock_scale(1_100_000_000), 4);
        assert_eq!(lock_scale(100_000_000), MAX_LOCK_SCALE);
        // Within 5% of a step: the lower scale, whichever way noise goes.
        assert_eq!(lock_scale(3_400_000_000), 1);
        assert_eq!(lock_scale(1_700_000_000), 2);
        assert_eq!(lock_scale(1_600_000_000), 3);
        assert_eq!(
            lock_scale_arguments(2),
            " ml-timeout-ticket-lock-scale=2 ml-timeout-lock-scale=2 \
             ml-timeout-lock-panic-scale=2"
        );
    }

    #[test]
    fn dependent_additions_show_the_core_clock() {
        // 2,000,000 additions in 1,000,000 ticks of a 2 GHz TSC: 4 GHz.
        assert_eq!(
            core_hz(REFERENCE_ADDS, 1_000_000, 2_000_000_000),
            4_000_000_000
        );
        assert_eq!(core_hz(REFERENCE_ADDS, 0, 2_000_000_000), 0);
        assert_eq!(lock_scale(core_hz(REFERENCE_ADDS, 0, 1)), 1, "unmeasured");
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn this_host_s_core_runs_the_reference_loop_near_its_tsc() {
        // As cycles per TSC tick: a core between a tenth and four times its
        // TSC's rate, near the same from one measurement to the next even
        // while other work loads the host.
        let first = measure_core_hz(1_000_000_000);
        assert!((100_000_000..4_000_000_000).contains(&first), "{first}");
        let again = measure_core_hz(1_000_000_000);
        assert!(first.abs_diff(again) < first / 4, "{first} then {again}");
    }

    #[test]
    fn the_debugger_cross_call_gives_up_long_before_the_watchdog_mechanism() {
        // The panic's own printing, a nested SPTM query per silent PE and
        // the paniclog flush need time after the wait, within the 9 s the
        // rounded-down mechanism deadline may leave.
        let shortest_mechanism = WATCHDOG_MECHANISM_SECONDS - 1;
        assert!(DEBUG_ACK_SECONDS * 4 <= shortest_mechanism);
        // Still far above XNU's 60 ms default and a 0.5 s engine exit.
        assert!(picoseconds(DEBUG_ACK_SECONDS) >= 1_000_000_000_000);
    }

    #[test]
    fn a_boot_arg_timeout_is_in_picoseconds() {
        assert_eq!(picoseconds(10), 10_000_000_000_000);
        // What 24 MHz ticks XNU makes of it: 10 s.
        assert_eq!(picoseconds(10) / 1_000_000 * 24, 240_000_000);
    }

    #[test]
    fn the_old_tick_count_was_240_microseconds() {
        // 240,000,000 read as picoseconds, which XNU does.
        assert_eq!(240_000_000 * 24_000_000 / PICOSECONDS_PER_SECOND, 5_760);
    }

    #[test]
    fn a_timeout_too_long_saturates() {
        assert_eq!(picoseconds(u64::MAX), u64::MAX);
    }
}
