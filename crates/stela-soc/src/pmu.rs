//! Dialog SPMI PMU (`pmu,spmi`): a byte-addressed register file behind an
//! SPMI slave ID, holding the boot-state scratchpads and, on the primary
//! PMU service, the 32768 Hz real-time clock and its alarm. Secondary PMUs
//! can own the same register layout without owning the system service.
//!
//! Evidence (26A434 KC VAs, unslid): `AppleDialogSPMIPMU::_readConfiguration`
//! 0xfffffe0009dd5014 reads the `info-*` windows; `_readRegs` 0xfffffe0009dd6610
//! and `_writeRegs` 0xfffffe0009dd64d4 use SPMI extended-long reads and writes
//! of consecutive addresses; `_readBootKey`/`_writeBootKey`
//! (0xfffffe0009dd713c/0xfffffe0009dd7210) address `info-leg_scrpad` plus a key
//! offset below 0x20. `AppleDialogSPMIPMURTC::_readRTCUpcountTicks`
//! 0xfffffe0009ddee60 reads six bytes at `info-rtc` and takes the 48-bit
//! little-endian value shifted right by one as 32768 Hz ticks;
//! `_readCurrentOffsetTicks` 0xfffffe0009dddee4 and `_writeCurrentOffsetTicks`
//! 0xfffffe0009dde0d4 keep a signed 48-bit tick offset at `info-clock_offset`
//! (`<base 6>`); `getUTCTimeOfDay` 0xfffffe0009dde9e4 returns
//! `(upcount + offset) >> 15` Unix seconds. Without `info-clock_offset` (the
//! J700AP Baku primary), the RTC `_readConfiguration` (mac17p 26A434 VA
//! 0xfffffe0009a25e60..25ed4) logs "Read RTC offset from leg_scrpad" and
//! takes `info-leg_scrpad` + 4 and + 0x15 instead; `_readCurrentOffsetTicks`
//! (0xfffffe0009a264b0..2655c) reads four little-endian bytes there as the
//! offset's seconds and two bytes at + 0x15 as its 15-bit fraction (low byte,
//! then seven bits), and sign-extends `seconds << 15 | fraction` from 47
//! bits; `_writeCurrentOffsetTicks` (0xfffffe0009a2661c..266d0) writes the
//! same fields. Those bytes are legacy-scratch storage. `updatePMSetting`
//! 0xfffffe0009dd44d8 read-modify-writes the byte at `info-pm_setting`.
//!
//! The alarm (KC file offsets): `AppleDialogSPMIPMURTC::_readConfiguration`
//! (0x2dd9874) takes the alarm's registers from `info-rtc_alarm_event`,
//! `_alarm_offset`, `_irq_mask_offset` and `_alarm_ctrl`, and its bits from
//! `info-rtc_alarm_event_mask` (0x1 when absent, as on J473AP),
//! `_alarm_mask`, `_alarm_monitor_mask` and `_alarm_ctrl_en_mask`.
//! `programRTCAlarm` (0x2ddab18) clears a latched event by writing its bit
//! to the event register, writes the alarm time as four little-endian bytes
//! at the alarm offset, the alarm mask to the IRQ mask register to mask the
//! alarm's interrupt or zero to unmask it, and the monitor bit, with the
//! enable bit for an alarm, to the control register. The time is in
//! seconds of the upcount (upcount ticks >> 15): `enableDateTimeAlarm`
//! (0x2ddad80) subtracts the clock offset's seconds from a calendar time,
//! and `scheduleRTCWakeAlarm` (0x2ddb1bc) compares it with the upcount's.
//! `checkRTCWakeEvent` (0x2ddb450) reports an RTC wake when every event-mask
//! bit is set in the event register.
//!
//! The model maps explicit register owners. Ordinary scratchpads and PM
//! settings begin with declared zero storage; these are not observed Apple
//! reset values. Outside-scratch fault logs require `with_fault_history` and
//! every initial byte explicitly supplied. Only the source's complete captured
//! snapshot acknowledgement is checked; arbitrary W1C, unchecked partial
//! writes,
//! physical fault production and reset semantics remain unsupported. A log
//! inside scratch shares its canonical bytes and source zero-clear behavior.
//! The off-to-wake source (`info-off2wake_source`, on Chapel, Baku and
//! Woodcote) is a second such owner: `_readOff2WakeSource` (mac15s 26A434 VA
//! 0xfffffe0009d57ea0) reads its byte-counted extent and, when asked to
//! clear, writes the same bytes back (`..d57f30..d57f68`), the fault log's
//! snapshot acknowledgement ([`stela_soc_data::pmu::Retained`], `with_wake_source_history`).
//! The legacy scratchpad is 0x100 bytes, the boot keys'
//! extent rounded to the 256-byte page at whose end the RTC block begins
//! (inferred). The upcount follows the system counter and refuses writes;
//! the clock offset starts at the host's wall time when the engine knows it
//! and otherwise at zero, a PMU whose RTC lost power. The upcount goes on
//! across power cycles from the machine state's anchor
//! ([`RtcEpoch::anchored`]), as a Mac's PMIC keeps counting while the
//! machine is off, and starts at zero without one. An enabled alarm
//! latches its event when the upcount's seconds reach the alarm time, once
//! per programming: writing the time or control arms it only for a time
//! still ahead (inferred: a comparator that fires on reaching the time, so
//! a time already passed never fires and `programRTCAlarm`'s clear of a
//! fired event does not retrigger it). The event register is
//! write-one-to-clear. The PMU's interrupt output ([`Interrupt`]) is
//! asserted while the event is latched and the IRQ mask leaves it
//! unmasked; where it leads is the board's. Every other address NACKs.
//!
//! Chapel (T603x/T604x), Abbey (M4/M5) and Woodcote declare the same byte for
//! `info-rtc_alarm_ctrl` and `info-rtc_alarm_event` (`0x2000`), with event
//! mask 0x2 beside monitor 0x1 and enable 0x40. The same `programRTCAlarm`
//! (mac15s 26A434 VA 0xfffffe0009d5f4e8) reads the event byte and, when an
//! event-mask bit is set, writes the byte it read back before it writes the
//! control (`..d5f590..d5f5d4`, `..d5f6d8..d5f70c`); `checkRTCWakeEvent`
//! (0xfffffe0009d5fe20) tests the event-mask bits of that byte. On a shared
//! byte that write-back keeps the control bits it read and clears the event
//! bits it carries, so the byte reads as control | event, its event-mask bits
//! are write-one-to-clear and its other bits are the control. The event and
//! control fields must be disjoint; the control field keeps the checked
//! monitor / monitor+enable encodings of a separate control register.
//!
//! Chapel and later PMICs declare a firmware mailbox: `fw-mcom-address`,
//! `fw-host-irq-address` and, from M4, `fw-mcom-int-index` naming its entry
//! in `interrupts`. Its AP consumer is `ApplePMUFirmwareDriver` (mac16g
//! 26A434; absent from mac15s): `start` (VA 0xfffffe0009be9418) reads the
//! three as 16-bit values, `mailboxTransactionGated` (0xfffffe0009bea2cc)
//! moves a five-byte message at +0, a byte at +5, five bytes at +6 and a
//! byte at +0xa around the mailbox interrupt, and `writeHostIRQGated`
//! (0xfffffe0009bea894) writes the doorbell byte. The answers are PMIC
//! firmware's, which does not execute here, so these 11 + 1 bytes are
//! explicit unsupported requirements and the mailbox vector is never driven.
//!
//! A PMU without RTC registers can still declare an output (`interrupts`
//! on the 26A434 Simetra, Vale, Spitfire, Portier, Club and Jerez PMUs,
//! from T8103 to T6050). The AP consumer of every PMU output is
//! `AppleDialogSPMIPMU::_interruptAction` (mac15g 26A434 VA 0xfffffe0009d63d80),
//! registered by `start` (0xfffffe0009d6317c..d6319c) for any provider with
//! `interrupts`: it only calls the RTC child's `programRTCAlarm(0)`, and with
//! no RTC child it does nothing beyond a debug log. The RTC alarm is the only
//! output source this model has; firmware, fault and rail events have no
//! modeled producer (as for the fault log), so such a PMU's output is an
//! actual, permanently deasserted wire rather than an unchecked one.
use crate::Result;
use stela_contracts::refuse;
use stela_soc_data::pmu::ALARM_BYTES;
#[cfg(test)]
use stela_soc_data::pmu::DEFAULT_EVENT_MASK;
#[cfg(test)]
use stela_soc_data::pmu::LEGACY_SCRATCH_BYTES;
use stela_soc_data::pmu::OFFSET_BYTES;
use stela_soc_data::pmu::UPCOUNT_BYTES;
mod clock;
mod fault;
mod firmware;
mod startup;
pub use clock::{RtcAnchor, RtcClock, RtcEpoch};

use super::spmi::{SlaveError, SpmiSlave};
#[cfg(test)]
use crate::counter::FREQUENCY_HZ;
use alloc::{rc::Rc, vec::Vec};
use core::cell::Cell;
use stela_soc_data::pmu::{Configuration, FirmwareStartup, Resources};

/// The RTC's tick rate.
pub const RTC_HZ: u64 = 32_768;
/// Upcount ticks per second of the alarm's time.
const SECOND_SHIFT: u32 = 15;
/// The firmware mailbox's message, doorbell, response and status bytes
/// (see the module comment).
use stela_soc_data::pmu::{ClockOffset, Rtc, WallClock, Window};

/// The PMU's interrupt output, shared with whatever the board connects it
/// to: asserted while an alarm event is latched and unmasked.
#[derive(Debug, Clone, Default)]
pub struct Interrupt(Rc<Cell<bool>>);

impl Interrupt {
    pub fn asserted(&self) -> bool {
        self.0.get()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Byte {
    Storage,
    /// A byte of the history owner at this index.
    History(usize, u16),
    Upcount(u16),
    AlarmControl,
    AlarmEvent,
    /// One byte holding both the control and the event bits.
    AlarmShared,
    IrqMask,
}

pub struct Pmu {
    /// Storage windows, including the RTC's storage registers.
    windows: Vec<Window>,
    rtc: Option<Rtc>,
    clock: RtcClock,
    monitor: u8,
    identity: Option<Window>,
    mailbox: Option<[Window; 2]>,
    /// The outside-scratch fault log and off-to-wake source, where declared.
    histories: Vec<fault::FaultLog>,
    firmware: Option<firmware::FirmwareMemory>,
    startup: Option<FirmwareStartup>,
    loader: Option<startup::Loader>,
    /// Bytes of every storage window, in window order.
    storage: Vec<Vec<u8>>,
    alarm_control: u8,
    alarm_event: u8,
    /// The alarm has not fired since its time or control was last written.
    armed: bool,
    interrupt: Interrupt,
}

impl Pmu {
    /// `windows` are the scratchpad and log windows; the RTC adds its own.
    #[cfg(test)]
    pub fn new(windows: Vec<Window>, rtc: Option<Rtc>, wall: Option<WallClock>) -> Self {
        // Original J473 legacy API; new board admission supplies the mask explicitly.
        let config = Configuration::new(windows, rtc, u8::from(rtc.is_some()))
            .expect("valid PMU register configuration");
        Self::configured(config, wall).expect("representable PMU initial wall clock")
    }

    /// Instantiate only validated byte resources. Initial wall time is a declared
    /// emulated resource value, never a captured PMIC identity or firmware image.
    #[cfg(test)]
    pub fn configured(config: Configuration, wall: Option<WallClock>) -> Result<Self> {
        Self::with_clock(
            config,
            RtcClock::new(RtcEpoch {
                counter: 0,
                ticks: 0,
            })?,
            wall,
        )
    }

    /// Bind a declared oscillator owner; the primary's SMC helper must keep
    /// this same handle, never derive a second epoch from its own uptime.
    pub fn with_clock(
        config: Configuration,
        clock: RtcClock,
        wall: Option<WallClock>,
    ) -> Result<Self> {
        let Resources {
            mut windows,
            rtc,
            monitor,
            identity,
            mailbox,
            fault_history,
            wake_source,
            firmware,
            startup,
            native_loader,
            execution_epoch,
        } = config.into_resources();
        let loader = if native_loader {
            Some(startup::Loader::new(
                startup.expect("validated native loader profile"),
                usize::from(
                    firmware
                        .as_ref()
                        .expect("validated native SRAM")
                        .ports
                        .bytes,
                ),
                execution_epoch,
            )?)
        } else {
            None
        };
        if let Some(rtc) = rtc {
            if let Some(base) = rtc.clock_offset.register() {
                windows.push(Window {
                    base,
                    bytes: OFFSET_BYTES,
                });
            }
            for (base, bytes) in [(rtc.alarm_offset, ALARM_BYTES), (rtc.irq_mask, 1)] {
                windows.push(Window { base, bytes });
            }
        }
        let storage = windows
            .iter()
            .map(|w| alloc::vec![0; w.bytes.into()])
            .collect();
        let mut pmu = Self {
            windows,
            rtc,
            clock,
            monitor,
            identity,
            mailbox,
            histories: fault_history
                .into_iter()
                .chain(wake_source)
                .map(Into::into)
                .collect(),
            firmware: firmware.map(Into::into),
            startup,
            loader,
            storage,
            alarm_control: 0,
            alarm_event: 0,
            armed: false,
            interrupt: Interrupt::default(),
        };
        if let (Some(rtc), Some(wall)) = (rtc, wall) {
            let upcount = pmu.clock.check(wall.counter)?;
            let target = i128::from(wall.unix_seconds) * i128::from(RTC_HZ);
            let offset = target - i128::from(upcount);
            match rtc.clock_offset {
                ClockOffset::Register(base) => {
                    if !(-(1i128 << 47)..(1i128 << 47)).contains(&offset) {
                        return Err(refuse!(
                            "PMU initial RTC offset exceeds signed 48-bit resource"
                        ));
                    }
                    let bytes = (offset as i64).to_le_bytes();
                    pmu.store(base, &bytes[..OFFSET_BYTES.into()]);
                }
                ClockOffset::LegacyScratch(_) => {
                    // `_readCurrentOffsetTicks` sign-extends 47 bits.
                    if !(-(1i128 << 46)..(1i128 << 46)).contains(&offset) {
                        return Err(refuse!(
                            "PMU initial RTC offset exceeds signed 47-bit resource"
                        ));
                    }
                    let [seconds, fraction] = rtc.clock_offset.legacy_fields().expect("validated");
                    let offset = offset as i64;
                    pmu.store(seconds.base, &((offset >> 15) as u32).to_le_bytes());
                    pmu.store(fraction.base, &[offset as u8, (offset >> 8) as u8 & 0x7f]);
                }
            }
        }
        Ok(pmu)
    }

    /// The kept oscillator owner, when this PMU has RTC resources.
    pub fn rtc_clock(&self) -> Option<RtcClock> {
        self.rtc.map(|_| self.clock.clone())
    }

    /// The PMU's interrupt output.
    pub fn interrupt(&self) -> Interrupt {
        self.interrupt.clone()
    }

    fn cell(&self, address: u16) -> Option<Byte> {
        if let Some((index, history)) = self
            .histories
            .iter()
            .enumerate()
            .find(|(_, history)| history.window.contains(address))
        {
            return Some(Byte::History(index, address - history.window.base));
        }
        if let Some(rtc) = self.rtc {
            if address.wrapping_sub(rtc.upcount) < UPCOUNT_BYTES {
                return Some(Byte::Upcount(address - rtc.upcount));
            }
            if address == rtc.alarm_control {
                return Some(if rtc.shares_event() {
                    Byte::AlarmShared
                } else {
                    Byte::AlarmControl
                });
            }
            if address == rtc.alarm_event {
                return Some(Byte::AlarmEvent);
            }
            if address == rtc.irq_mask {
                return Some(Byte::IrqMask);
            }
        }
        self.windows
            .iter()
            .any(|w| w.contains(address))
            .then_some(Byte::Storage)
    }
    fn byte(&mut self, address: u16) -> &mut u8 {
        let (index, window) = self
            .windows
            .iter()
            .enumerate()
            .find(|(_, w)| w.contains(address))
            .expect("storage cell");
        &mut self.storage[index][usize::from(address - window.base)]
    }
    fn store(&mut self, base: u16, bytes: &[u8]) {
        for (i, &value) in bytes.iter().enumerate() {
            *self.byte(base + i as u16) = value;
        }
    }
    fn check_unqualified(&self, address: u16, bytes: usize) -> Result<(), SlaveError> {
        for address in Self::span(address, bytes)? {
            if self.identity.is_some_and(|window| window.contains(address)) {
                return Err(SlaveError::Unsupported(refuse!(
                    "PMU declared identity register contents are unqualified"
                )));
            }
            if self
                .mailbox
                .is_some_and(|windows| windows.iter().any(|window| window.contains(address)))
            {
                return Err(SlaveError::Unsupported(refuse!(
                    "PMU firmware mailbox requires executing PMIC firmware"
                )));
            }
        }
        Ok(())
    }
    fn span(address: u16, bytes: usize) -> Result<impl Iterator<Item = u16>, SlaveError> {
        let bytes = u32::try_from(bytes).map_err(|_| SlaveError::Nack)?;
        let end = u32::from(address)
            .checked_add(bytes)
            .ok_or(SlaveError::Nack)?;
        if end > 0x1_0000 {
            return Err(SlaveError::Nack);
        }
        Ok((u32::from(address)..end).map(|a| a as u16))
    }

    fn check_firmware_startup(&self, address: u16, bytes: usize) -> Result<(), SlaveError> {
        let end = u64::from(address).saturating_add(bytes.max(1) as u64);
        if self.loader.is_none()
            && self.startup.is_some_and(|profile| {
                profile.windows().iter().any(|window| {
                    u64::from(address) < u64::from(window.base) + u64::from(window.bytes)
                        && u64::from(window.base) < end
                })
            })
        {
            return Err(SlaveError::Unsupported(refuse!(
                "PMU firmware startup requires the actual protected-control, execution and completion owner"
            )));
        }
        Ok(())
    }

    /// Check the shared oscillator before publishing any command effects.
    fn publish_time(&mut self, now: u64) -> Result<(), SlaveError> {
        self.check_time(now)?;
        let native = self
            .loader
            .as_mut()
            .map(|loader| loader.prepare_time(now))
            .transpose()
            .map_err(SlaveError::Unsupported)?
            .flatten();
        if self.rtc.is_some() {
            self.clock.publish(now).map_err(SlaveError::Unsupported)?;
        }
        if let Some(native) = native {
            native.commit();
        }
        self.update(now);
        Ok(())
    }

    /// The counter value at which the enabled alarm latches its event, if
    /// it is enabled, armed and has not.
    fn alarm_deadline(&mut self) -> Option<u64> {
        let rtc = self.rtc?;
        if !self.armed || self.alarm_event & rtc.event_mask != 0 {
            return None;
        }
        self.alarm_time()
    }

    /// The counter value of the enabled alarm's time.
    fn alarm_time(&mut self) -> Option<u64> {
        let rtc = self.rtc?;
        if self.alarm_control & rtc.alarm_enable == 0 {
            return None;
        }
        let mut time = [0u8; ALARM_BYTES as usize];
        for (i, byte) in time.iter_mut().enumerate() {
            *byte = *self.byte(rtc.alarm_offset + i as u16);
        }
        let upcount = u64::from(u32::from_le_bytes(time)) << SECOND_SHIFT;
        self.clock.counter_for(upcount)
    }

    /// Latch the alarm's event once its time is reached, and drive the
    /// interrupt output.
    fn update(&mut self, now: u64) {
        let Some(rtc) = self.rtc else {
            return;
        };
        if self
            .alarm_deadline()
            .is_some_and(|deadline| now >= deadline)
        {
            self.alarm_event |= rtc.event_mask;
            self.armed = false;
        }
        let masked = *self.byte(rtc.irq_mask) & rtc.interrupt_mask != 0;
        self.interrupt
            .0
            .set(self.alarm_event & rtc.event_mask != 0 && !masked);
    }
}

impl SpmiSlave for Pmu {
    fn read(&mut self, address: u16, data: &mut [u8], now: u64) -> Result<(), SlaveError> {
        self.check_unqualified(address, data.len())?;
        self.check_firmware_startup(address, data.len())?;
        if let Some(loader) = &self.loader
            && loader.claims(address, data.len())
        {
            let bytes = loader
                .read(address, data.len())
                .map_err(SlaveError::Unsupported)?;
            self.publish_time(now)?;
            data.copy_from_slice(&bytes);
            return Ok(());
        }
        if let Some(memory) = &self.firmware
            && memory.claims(address, data.len())
        {
            if let Some(loader) = &self.loader {
                loader
                    .check_memory(false)
                    .map_err(SlaveError::Unsupported)?;
            }
            let transfer = memory
                .prepare(address, data.len(), firmware::Direction::Read)
                .map_err(SlaveError::Unsupported)?;
            self.publish_time(now)?;
            self.firmware
                .as_mut()
                .expect("firmware owner")
                .read(transfer, data);
            return Ok(());
        }
        let cells: Vec<_> = Self::span(address, data.len())?
            .map(|a| self.cell(a).map(|cell| (a, cell)))
            .collect::<Option<_>>()
            .ok_or(SlaveError::Nack)?;
        for (index, history) in self.histories.iter().enumerate() {
            if cells
                .iter()
                .any(|(_, cell)| matches!(cell, Byte::History(owner, _) if *owner == index))
            {
                history.check_read().map_err(SlaveError::Unsupported)?;
            }
        }
        self.publish_time(now)?;
        // One sample serves every upcount byte of the access.
        let raw = if self.rtc.is_some() {
            (self
                .clock
                .published_ticks()
                .map_err(SlaveError::Unsupported)?
                << 1)
                .to_le_bytes()
        } else {
            [0; 8]
        };
        for ((at, cell), out) in cells.into_iter().zip(data.iter_mut()) {
            *out = match cell {
                Byte::Storage | Byte::IrqMask => *self.byte(at),
                Byte::History(index, offset) => self.histories[index].byte(offset),
                Byte::Upcount(i) => raw[usize::from(i)],
                Byte::AlarmControl => self.alarm_control,
                Byte::AlarmEvent => self.alarm_event,
                Byte::AlarmShared => self.alarm_control | self.alarm_event,
            };
        }
        for history in &mut self.histories {
            history.capture(address, data.len());
        }
        Ok(())
    }
    fn write(&mut self, address: u16, data: &[u8], now: u64) -> Result<(), SlaveError> {
        self.check_unqualified(address, data.len())?;
        self.check_firmware_startup(address, data.len())?;
        if let Some(loader) = &self.loader
            && loader.claims(address, data.len())
        {
            let write = loader
                .prepare(
                    address,
                    data,
                    self.firmware.as_ref().expect("native loader SRAM"),
                    now,
                )
                .map_err(SlaveError::Unsupported)?;
            self.publish_time(now)?;
            let stop = matches!(write, startup::Write::Stop);
            self.loader.as_mut().expect("native loader").commit(write);
            if stop {
                self.firmware
                    .as_mut()
                    .expect("native loader SRAM")
                    .forget_address();
            }
            return Ok(());
        }
        if let Some(memory) = &self.firmware
            && memory.claims(address, data.len())
        {
            let address_byte = memory
                .check_address(address, data)
                .map_err(SlaveError::Unsupported)?;
            if let Some(loader) = &self.loader {
                // The address pair only positions the port. Permit
                // a new readback cursor after sealing while rejecting every
                // data mutation of the protected image.
                loader
                    .check_memory(address_byte.is_none())
                    .map_err(SlaveError::Unsupported)?;
            }
            let transfer = if address_byte.is_none() {
                Some(
                    memory
                        .prepare(address, data.len(), firmware::Direction::Write)
                        .map_err(SlaveError::Unsupported)?,
                )
            } else {
                None
            };
            self.publish_time(now)?;
            let memory = self.firmware.as_mut().expect("firmware owner");
            if let Some(byte) = address_byte {
                memory.address(address, byte);
            }
            if let Some(transfer) = transfer {
                if let Some(loader) = &mut self.loader {
                    loader.written(transfer.range.clone());
                }
                memory.write(transfer, data);
            }
            return Ok(());
        }
        let cells: Vec<_> = Self::span(address, data.len())?
            .map(|a| self.cell(a).map(|cell| (a, cell)))
            .collect::<Option<_>>()
            .ok_or(SlaveError::Nack)?;
        if cells
            .iter()
            .any(|&(_, cell)| matches!(cell, Byte::Upcount(_)))
        {
            return Err(SlaveError::Nack);
        }
        let mut owners = cells.iter().filter_map(|(_, cell)| match cell {
            Byte::History(index, _) => Some(*index),
            _ => None,
        });
        let acknowledged = owners.next();
        if let Some(index) = acknowledged {
            if owners.any(|other| other != index) {
                return Err(SlaveError::Unsupported(refuse!(
                    "PMU acknowledge spanning two retained-history owners is unqualified"
                )));
            }
            self.histories[index]
                .check_acknowledge(address, data)
                .map_err(SlaveError::Unsupported)?;
        }
        // Preflight the complete transfer before time, storage, event or output effects.
        if let Some(rtc) = self.rtc {
            let control = |value| value == self.monitor || value == self.monitor | rtc.alarm_enable;
            for ((_, cell), &value) in cells.iter().zip(data) {
                let qualified = match cell {
                    Byte::AlarmControl => control(value),
                    Byte::AlarmEvent => value & !rtc.event_mask == 0,
                    Byte::AlarmShared => control(value & !rtc.event_mask),
                    Byte::IrqMask => value == 0 || value == rtc.interrupt_mask,
                    _ => true,
                };
                if !qualified {
                    return Err(SlaveError::Unsupported(refuse!(
                        "PMU RTC control/event/mask field is unqualified"
                    )));
                }
            }
        }
        self.publish_time(now)?;
        let alarm = self
            .rtc
            .map(|rtc| rtc.alarm_offset..rtc.alarm_offset + ALARM_BYTES);
        let mut programmed = false;
        for ((at, cell), &value) in cells.into_iter().zip(data) {
            match cell {
                Byte::Storage | Byte::IrqMask => {
                    *self.byte(at) = value;
                    programmed |= alarm.as_ref().is_some_and(|time| time.contains(&at));
                }
                Byte::AlarmControl => {
                    self.alarm_control = value;
                    programmed = true;
                }
                Byte::AlarmEvent => self.alarm_event &= !value,
                Byte::AlarmShared => {
                    let events = self.rtc.expect("RTC byte").event_mask;
                    self.alarm_control = value & !events;
                    self.alarm_event &= !(value & events);
                    programmed = true;
                }
                Byte::History(..) => {} // One complete snapshot is acknowledged below.
                Byte::Upcount(_) => unreachable!(),
            }
        }
        if let Some(index) = acknowledged {
            self.histories[index].acknowledge(address, data.len());
        }
        if programmed {
            self.armed = self.alarm_time().is_some_and(|time| time > now);
        }
        self.update(now);
        Ok(())
    }
    fn check_time(&self, now: u64) -> Result<(), SlaveError> {
        if self.rtc.is_some() {
            self.clock
                .check(now)
                .map(|_| ())
                .map_err(SlaveError::Unsupported)?;
        }
        self.loader
            .as_ref()
            .map_or(Ok(()), |loader| loader.check_time(now))
            .map_err(SlaveError::Unsupported)
    }
    /// The RTC alarm is the only modeled output source; without an RTC the
    /// output stays deasserted (see the module comment).
    fn interrupt_level(&self) -> Option<bool> {
        Some(self.interrupt.asserted())
    }
    fn interrupt_armed(&self) -> bool {
        self.armed || self.interrupt.asserted()
    }
    fn advance(&mut self, now: u64) -> Option<u64> {
        self.publish_time(now)
            .expect("SPMI autonomous time preflight completed");
        self.alarm_deadline()
    }
}

#[cfg(test)]
mod tests;
