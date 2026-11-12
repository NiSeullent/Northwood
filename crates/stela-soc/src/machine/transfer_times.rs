//! The host time each part of a board transfer ([`Devices::transfer`])
//! takes, kept while the host gives a clock (`--profile`). A transfer runs
//! after every device write, under the devices' lock that every PE's device
//! access waits for, and for the home machine at the DMA deadline: this
//! names the master whose step holds it.
//!
//! Without a clock a part costs one branch.
//!
//! [`Devices::transfer`]: stela_engine::machine::Devices::transfer

/// The parts, in the order a transfer runs them: its entry (time and
/// lifecycle), each master's DMA step, and the wires it settles.
pub const TRANSFER_PARTS: [&str; 9] = [
    "entry", "pmp", "aes", "storage", "gfx", "sep", "dcp", "usb", "settle",
];

/// One part's calls, host ticks in all and the longest call.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PartTime {
    pub calls: u64,
    pub ticks: u64,
    pub longest: u64,
}

#[derive(Default)]
pub(super) struct TransferTimes {
    clock: Option<fn() -> u64>,
    parts: [PartTime; TRANSFER_PARTS.len()],
}

/// The time since the last part ended.
pub(super) struct Lap {
    clock: Option<fn() -> u64>,
    last: u64,
}

impl TransferTimes {
    pub(super) fn set_clock(&mut self, clock: Option<fn() -> u64>) {
        self.clock = clock;
    }

    /// A lap from now.
    pub(super) fn lap(&self) -> Lap {
        Lap {
            clock: self.clock,
            last: self.clock.map_or(0, |clock| clock()),
        }
    }

    /// Each part's time since the last call, by [`TRANSFER_PARTS`].
    pub(super) fn take(&mut self) -> [PartTime; TRANSFER_PARTS.len()] {
        core::mem::take(&mut self.parts)
    }
}

impl Lap {
    /// Part `part` (an index of [`TRANSFER_PARTS`]) ended now.
    pub(super) fn end(&mut self, times: &mut TransferTimes, part: usize) {
        let Some(clock) = self.clock else {
            return;
        };
        let now = clock();
        let took = now.wrapping_sub(self.last);
        self.last = now;
        let time = &mut times.parts[part];
        time.calls += 1;
        time.ticks = time.ticks.wrapping_add(took);
        time.longest = time.longest.max(took);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::machine::AppleDevices;
    use core::sync::atomic::{AtomicU64, Ordering};
    use stela_contracts::dma::test_support::Ram;
    use stela_engine::machine::Devices;

    static TICKS: AtomicU64 = AtomicU64::new(0);

    /// Advances by 10 a read.
    fn clock() -> u64 {
        TICKS.fetch_add(10, Ordering::Relaxed) + 10
    }

    #[test]
    fn a_clock_times_each_transfer_part_and_none_costs_nothing() {
        let mut devices = AppleDevices::new();
        let mut ram = Ram::new(0x8_0000_0000, 0x1000);
        devices.transfer(0, &mut ram, 0).unwrap();
        assert_eq!(devices.take_transfer_times(), [PartTime::default(); 9]);
        devices.time_transfers(Some(clock));
        devices.transfer(0, &mut ram, 1).unwrap();
        devices.transfer(0, &mut ram, 2).unwrap();
        let times = devices.take_transfer_times();
        // Each part read the clock once a transfer, one read after the
        // last: 10 ticks each.
        assert!(
            times.iter().all(|time| *time
                == PartTime {
                    calls: 2,
                    ticks: 20,
                    longest: 10
                }),
            "{times:?}"
        );
        assert_eq!(devices.take_transfer_times(), [PartTime::default(); 9]);
        devices.time_transfers(None);
        devices.transfer(0, &mut ram, 3).unwrap();
        assert_eq!(devices.take_transfer_times(), [PartTime::default(); 9]);
    }

    struct Sink;
    impl crate::uart::ConsoleSink for Sink {
        fn transmit(&mut self, _: u8) -> Result<(), crate::uart::ConsoleBusy> {
            Ok(())
        }
    }

    struct Locked(Ram);
    impl stela_engine::machine::engine::LockedMemory for Locked {
        fn under(
            &mut self,
            _: stela_engine::machine::mmu::MemoryLocks,
        ) -> &mut dyn stela_engine::machine::engine::Memory {
            &mut self.0
        }
    }

    /// A console byte's write settles its wires without the masters' steps,
    /// except every `QUIET_WRITES + 1`th; a transfer of its own runs them.
    #[test]
    fn a_console_write_runs_the_masters_only_every_few_writes() {
        const UART: u64 = 0x2_3520_0000;
        let mut devices = AppleDevices::new();
        devices
            .add_uart(
                Sink,
                &[crate::machine::Range {
                    physical: UART,
                    bytes: 0x4000,
                }],
                None,
            )
            .unwrap();
        let mut ram = Locked(Ram::new(0x8_0000_0000, 0x1000));
        devices.time_transfers(Some(clock));
        let writes = super::super::QUIET_WRITES as u64 + 4;
        for now in 1..=writes {
            devices
                .write(0, UART + 0x20, 4, u64::from(b'a'), now)
                .unwrap();
            devices.settle_write(0, now, &mut ram).transfer.unwrap();
        }
        let times = devices.take_transfer_times();
        // Every write settled its wires; the masters ran once, at the write
        // after a full run of quiet ones.
        assert_eq!(times[0].calls, writes, "{times:?}");
        assert_eq!(times[8].calls, writes, "{times:?}");
        assert!(times[1..8].iter().all(|time| time.calls == 1), "{times:?}");
        // A transfer of its own, as at a DMA deadline, always runs them.
        devices.transfer(0, &mut ram.0, writes + 1).unwrap();
        let times = devices.take_transfer_times();
        assert!(times.iter().all(|time| time.calls == 1), "{times:?}");
    }
}
