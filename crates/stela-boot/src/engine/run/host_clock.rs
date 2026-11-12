// SPDX-License-Identifier: GPL-2.0-or-later
use crate::output;
use core::arch::x86_64::__cpuid;
use stela_boot_core::msr::{APERF, MPERF};

/// Interval samples for the boot processor, which owns the heartbeat.
pub(super) struct Samples {
    actual: u64,
    reference: u64,
}

impl Samples {
    pub(super) fn start() -> Option<Self> {
        let vendor = __cpuid(0);
        let intel = [vendor.ebx, vendor.edx, vendor.ecx] == [0x756e6547, 0x49656e69, 0x6c65746e]; // GenuineIntel
        let virtual_cpu = __cpuid(1).ecx & (1 << 31) != 0;
        (intel && !virtual_cpu && APERF.implemented()).then(Self::read)
    }

    fn read() -> Self {
        // SAFETY: this path runs at CPL0 on the same boot processor. Its
        // Intel CPUID.06H:ECX[0] enumerates both MSRs before the first read.
        unsafe {
            Self {
                actual: APERF.read(),
                reference: MPERF.read(),
            }
        }
    }

    pub(super) fn report(&mut self, tsc_hz: u64) {
        let previous = core::mem::replace(self, Self::read());
        let (Some(actual), Some(reference)) = (
            self.actual.checked_sub(previous.actual),
            self.reference.checked_sub(previous.reference),
        ) else {
            return;
        };
        if reference == 0 {
            return;
        }
        let ratio_milli = u128::from(actual) * 1000 / u128::from(reference);
        let active_mhz =
            u128::from(actual) * u128::from(tsc_hz) / (u128::from(reference) * 1_000_000);
        output::arguments(format_args!(
            "HOST CLOCK cpu=bsp aperf_delta={actual} mperf_delta={reference} ratio_milli={ratio_milli} active_mhz={active_mhz}\n"
        ));
    }
}
