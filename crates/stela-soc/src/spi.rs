//! Apple `spi-1,spimc`, version 1, byte PIO.
//!
//! Sources: m1n1 `proxyclient/m1n1/hw/spi.py` (v1 STATUS.TX_COMPLETE
//! is bit 22, unlike later Linux controllers), Linux `drivers/spi/spi-apple.c`
//! (FIFO depth 16 and register fields), and 26A434 kernel
//! `AppleSPIMCController::_executeSPICommandPIO` file 0x2dbde3c,
//! `_interruptActionSubr` 0x2dbf68c, `_setInternalCS` 0x2dc00cc,
//! `_getTxFIFOCount` 0x2dc02b8 and `_getRxFIFOCount` 0x2dc0348.
//! The shifter runs eagerly on accesses until a FIFO stalls: no host I/O or
//! transfer-time guarantee. The boot stage leaves it stopped, CS inactive,
//! FIFOs/counts/configuration zero (Stela boot-chain choices). Only byte,
//! MSB-first PIO is supported; DMA and wider words refuse.
//!
//! Linux `spi-apple.c` (the same M1-family controller) also names SHIFTCFG
//! 0x150 and DELAY_PRE/POST 0x160/0x168, which 26A434
//! `AppleSPIMCController::_configureHardwareDelay` (file 0x2dbcf34) uses: its
//! software-delay branch clears SHIFTCFG.OVERRIDE_CS (bit 24) and both delay
//! enables by RMW. The boot stage leaves all three zero (Linux's own init
//! state). Delays only change timing, which the eager shifter never
//! guarantees, so their words are held. Only OVERRIDE_CS may change in
//! SHIFTCFG, and only while an external GPIO drives the chip select, since
//! the internal CS output is then disconnected from the attached device.
use crate::Result;
use alloc::vec::Vec;
use stela_contracts::Refusal;
use stela_contracts::refuse;
use {
    super::nor::Nor, stela_contracts::bus::Access, stela_contracts::bus::Direction,
    stela_contracts::bus::Lifecycle, stela_contracts::bus::LifecycleState,
    stela_contracts::bus::MmioDevice, stela_contracts::bus::MmioError, stela_contracts::bus::Width,
    stela_contracts::bus::refuse, stela_contracts::interrupt::AicInput,
    stela_contracts::interrupt::Wire, stela_soc_data::nor::Image,
};

const NAME: &str = "Apple SPI v1";
const CTRL: u64 = 0x00;
const CFG: u64 = 0x04;
const STATUS: u64 = 0x08;
const PIN: u64 = 0x0c;
const TXDATA: u64 = 0x10;
const RXDATA: u64 = 0x20;
const CLKDIV: u64 = 0x30;
const RXCNT: u64 = 0x34;
const DELAY: u64 = 0x38;
const TXCNT: u64 = 0x4c;
const FIFOSTAT: u64 = 0x10c;
const SHIFTCFG: u64 = 0x150;
const DELAY_PRE: u64 = 0x160;
const DELAY_POST: u64 = 0x168;
const OVERRIDE_CS: u32 = 1 << 24;
const RUN: u32 = 1;
const TX_RESET: u32 = 1 << 2;
const RX_RESET: u32 = 1 << 3;
const CS_HIGH: u32 = 1 << 1;
const FIFO_DEPTH: usize = 16;
const RX_COMPLETE: u32 = 1;
const THRESHOLD: u32 = 1 << 1;
const TX_COMPLETE: u32 = 1 << 22;
const MODE: u32 = 3 << 5;
const IRQ_MODE: u32 = 1 << 5;
const IE_RX_COMPLETE: u32 = 1 << 7;
const IE_THRESHOLD: u32 = 1 << 8;
const IE_TX_COMPLETE: u32 = 1 << 21;
const FIFO_THRESHOLD: u32 = 3 << 17;
const CONFIG_MASK: u32 = 6 | MODE | IE_RX_COMPLETE | IE_THRESHOLD | IE_TX_COMPLETE | FIFO_THRESHOLD;
const TRANSFER_CONFIGURATION: u8 = 3; // CTRL and CFG.
/// SHIFTCFG and DELAY_PRE/POST: a power cycle or reset restores the boot
/// stage's zero words, which `_configureHardwareDelay` read-modify-writes
/// after power-up (file 0x2dbcf34) without writing them first.
const RESET_CONFIGURATION: u8 = 0xe0;

fn configuration_bit(offset: u64) -> u8 {
    match offset {
        CTRL => 1,
        CFG => 2,
        PIN => 4,
        CLKDIV => 8,
        DELAY => 16,
        SHIFTCFG => 32,
        DELAY_PRE => 64,
        DELAY_POST => 128,
        _ => 0,
    }
}

/// A byte FIFO of [`FIFO_DEPTH`]: a copy of its value, not of the heap,
/// is what a refused transfer restores.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Fifo {
    bytes: [u8; FIFO_DEPTH],
    start: usize,
    len: usize,
}

impl Fifo {
    fn len(&self) -> usize {
        self.len
    }
    fn is_empty(&self) -> bool {
        self.len == 0
    }
    fn front(&self) -> Option<u8> {
        (self.len != 0).then(|| self.bytes[self.start])
    }
    fn pop_front(&mut self) -> Option<u8> {
        let byte = self.front()?;
        self.start = (self.start + 1) % FIFO_DEPTH;
        self.len -= 1;
        Some(byte)
    }
    /// Callers check for room first.
    fn push_back(&mut self, byte: u8) {
        assert!(self.len < FIFO_DEPTH, "SPI FIFO overflow");
        self.bytes[(self.start + self.len) % FIFO_DEPTH] = byte;
        self.len += 1;
    }
    fn push_front(&mut self, byte: u8) {
        assert!(self.len < FIFO_DEPTH, "SPI FIFO overflow");
        self.start = (self.start + FIFO_DEPTH - 1) % FIFO_DEPTH;
        self.bytes[self.start] = byte;
        self.len += 1;
    }
    fn clear(&mut self) {
        *self = Self::default();
    }
}

#[cfg(feature = "native-profile")]
#[derive(Default)]
struct RegisterProfile {
    reads: u64,
    writes: u64,
    rxdata_reads: u64,
    txdata_writes: u64,
}

pub struct Spi {
    nor: Nor,
    irq: AicInput,
    control: u32,
    config: u32,
    pin: u32,
    divider: u32,
    delay: u32,
    shift: u32,
    cs_delays: [u32; 2],
    status: u32,
    tx_count: u32,
    rx_count: u32,
    tx: Fifo,
    rx: Fifo,
    /// External CS wiring disables the internal PIN output's connection.
    external_cs: bool,
    lifecycle: LifecycleState,
    configuration: u8,
    #[cfg(feature = "native-profile")]
    profile: RegisterProfile,
}
impl Spi {
    pub fn new(image: Image, irq: AicInput, external_cs: bool) -> Self {
        Self {
            nor: Nor::new(image),
            irq,
            control: 0,
            config: 0,
            pin: CS_HIGH,
            divider: 0,
            delay: 0,
            shift: 0,
            cs_delays: [0; 2],
            status: 0,
            tx_count: 0,
            rx_count: 0,
            tx: Fifo::default(),
            rx: Fifo::default(),
            external_cs,
            lifecycle: LifecycleState {
                powered: true,
                clocked: true,
                quiesced: false,
                reset: false,
                retention: false,
            },
            configuration: u8::MAX,
            #[cfg(feature = "native-profile")]
            profile: RegisterProfile::default(),
        }
    }
    /// The NOR image as programs and erases left it.
    pub fn image(&self) -> &Image {
        self.nor.image()
    }
    /// The NOR sectors changed since the last call (`Nor::take_dirty`).
    pub fn take_dirty(&mut self) -> Vec<(usize, Vec<u8>)> {
        self.nor.take_dirty()
    }
    /// Successful SPI register accesses and NOR commands completed at CS
    /// deassertion since the last report. No register or payload values.
    #[cfg(feature = "native-profile")]
    pub(crate) fn take_profile(&mut self) -> alloc::string::String {
        let spi = core::mem::take(&mut self.profile);
        let nor = self.nor.take_profile();
        alloc::format!(
            "mmio_reads={} mmio_writes={} rxdata_reads={} txdata_writes={} \
             reads={} read_bytes={} programs={} program_bytes={} erases={} erase_bytes={} status_bytes={}",
            spi.reads,
            spi.writes,
            spi.rxdata_reads,
            spi.txdata_writes,
            nor.reads,
            nor.read_bytes,
            nor.programs,
            nor.program_bytes,
            nor.erases,
            nor.erase_bytes,
            nor.status_bytes,
        )
    }
    /// Board wiring, independent of command/transfer boundaries. External
    /// GPIO's output is active low, just as PIN.CS is for internal selection.
    pub(crate) fn check_cs(&self, high: bool) -> Result<()> {
        self.nor.check_select(!high)
    }
    pub fn set_cs(&mut self, high: bool) -> Result<()> {
        self.nor.select(!high)
    }
    fn error(
        offset: u64,
        width: Width,
        direction: Direction,
        reason: impl Into<Refusal>,
    ) -> MmioError {
        refuse(NAME, offset, width, direction, reason)
    }
    fn threshold(&self) -> bool {
        let threshold = match (self.config & FIFO_THRESHOLD) >> 17 {
            0 => 8,
            1 => 4,
            2 => 1,
            _ => return false,
        };
        self.config & MODE == IRQ_MODE
            && (self.rx.len() >= threshold
                || self.tx_count != 0 && FIFO_DEPTH - self.tx.len() >= threshold)
    }
    fn status(&self) -> u32 {
        self.status | if self.threshold() { THRESHOLD } else { 0 }
    }
    fn interrupt(&self, access: &mut Access<'_>) {
        let status = self.status();
        let enabled = (status & RX_COMPLETE != 0 && self.config & IE_RX_COMPLETE != 0)
            || (status & THRESHOLD != 0 && self.config & IE_THRESHOLD != 0)
            || (status & TX_COMPLETE != 0 && self.config & IE_TX_COMPLETE != 0);
        access
            .interrupts
            .set_level(Wire::Aic(self.irq), self.active() && enabled);
    }

    fn mmio_ready(&self) -> Result<()> {
        if !self.lifecycle.powered {
            return Err(refuse!("SPI power domain is off"));
        }
        if !self.lifecycle.clocked {
            return Err(refuse!("SPI clock is gated"));
        }
        if self.lifecycle.reset {
            return Err(refuse!("SPI physical reset is asserted"));
        }
        Ok(())
    }

    fn active(&self) -> bool {
        self.mmio_ready().is_ok() && !self.lifecycle.quiesced
    }
    fn pump(&mut self) -> Result<()> {
        // An idle controller moves nothing: skip the rollback copies that
        // every device advance would otherwise take.
        if !self.active() || self.control & RUN == 0 {
            return Ok(());
        }
        let checkpoint = self.nor.checkpoint();
        let (tx, rx, tx_count, rx_count, status) =
            (self.tx, self.rx, self.tx_count, self.rx_count, self.status);
        if let Err(error) = self.pump_inner() {
            self.nor.restore(checkpoint);
            self.tx = tx;
            self.rx = rx;
            self.tx_count = tx_count;
            self.rx_count = rx_count;
            self.status = status;
            return Err(error);
        }
        Ok(())
    }
    fn pump_inner(&mut self) -> Result<()> {
        if !self.active() || self.control & RUN == 0 {
            return Ok(());
        }
        if self.configuration & TRANSFER_CONFIGURATION != TRANSFER_CONFIGURATION {
            return Err(refuse!(
                "SPI control and transfer configuration require initialization"
            ));
        }
        while self.tx_count != 0 || self.rx_count != 0 {
            if self.rx_count != 0 && self.rx.len() == FIFO_DEPTH {
                break;
            }
            let byte = if self.tx_count != 0 {
                let Some(byte) = self.tx.front() else {
                    break;
                };
                byte
            } else {
                0xff
            };
            let incoming = self.nor.shift(byte)?;
            if self.tx_count != 0 {
                self.tx.pop_front();
                self.tx_count -= 1;
                if self.tx_count == 0 {
                    self.status |= TX_COMPLETE;
                }
            }
            if self.rx_count != 0 {
                self.rx.push_back(incoming);
                self.rx_count -= 1;
                if self.rx_count == 0 {
                    self.status |= RX_COMPLETE;
                }
            }
        }
        Ok(())
    }
}

impl Lifecycle for Spi {
    fn check_lifecycle(&self, proposed: LifecycleState) -> Result<()> {
        if (self.lifecycle.powered && !proposed.powered && !proposed.retention
            || !self.lifecycle.reset && proposed.reset)
            && !self.external_cs
        {
            self.nor.check_select(false)?;
        }
        Ok(())
    }

    fn set_lifecycle(&mut self, proposed: LifecycleState, access: &mut Access<'_>) {
        if self.lifecycle.powered && !proposed.powered && !proposed.retention
            || !self.lifecycle.reset && proposed.reset
        {
            if !self.external_cs {
                // Internal CS rising may commit an accepted NOR
                // program prefix. Its preflight ran before PMGR completion.
                self.nor.select(false).expect("checked internal SPI CS");
            }
            // External GPIO CS and the flash die are separate hardware:
            // keep their transaction, image, WEL and dirty sectors.
            self.control = 0;
            self.status = 0;
            self.tx_count = 0;
            self.rx_count = 0;
            self.tx.clear();
            self.rx.clear();
            self.shift = 0;
            self.cs_delays = [0; 2];
            self.configuration = RESET_CONFIGURATION;
        }
        self.lifecycle = proposed;
        self.interrupt(access);
    }
}

impl MmioDevice for Spi {
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
        let error = |reason: Refusal| Self::error(offset, width, Direction::Read, reason);
        if width != Width::Word || !offset.is_multiple_of(4) {
            return Err(error(refuse!(
                "SPI registers require aligned 32-bit accesses"
            )));
        }
        self.mmio_ready().map_err(error)?;
        let bit = configuration_bit(offset);
        if bit != 0 && self.configuration & bit == 0 {
            return Err(error(refuse!(
                "SPI configuration reset value has not been qualified"
            )));
        }
        let value = match offset {
            CTRL => self.control,
            CFG => self.config,
            STATUS => self.status(),
            PIN => self.pin,
            CLKDIV => self.divider,
            DELAY => self.delay,
            SHIFTCFG => self.shift,
            DELAY_PRE => self.cs_delays[0],
            DELAY_POST => self.cs_delays[1],
            TXCNT => self.tx_count,
            RXCNT => self.rx_count,
            FIFOSTAT => {
                ((self.tx.len() as u32) << 8)
                    | ((self.rx.len() as u32) << 24)
                    | if self.tx.len() == FIFO_DEPTH {
                        1 << 4
                    } else {
                        0
                    }
                    | if self.rx.is_empty() { 1 << 20 } else { 0 }
            }
            RXDATA => {
                let byte = self
                    .rx
                    .front()
                    .ok_or_else(|| error(refuse!("SPI RX FIFO underflow")))?;
                // Consume only after validating the next shift. The incoming
                // command byte is validated by TXDATA, before it is queued.
                self.rx.pop_front();
                if let Err(reason) = self.pump() {
                    self.rx.push_front(byte);
                    return Err(error(reason));
                }
                u32::from(byte)
            }
            _ => return Err(error(refuse!("unmodeled SPI register"))),
        };
        self.interrupt(access);
        #[cfg(feature = "native-profile")]
        {
            self.profile.reads = self.profile.reads.saturating_add(1);
            self.profile.rxdata_reads = self
                .profile
                .rxdata_reads
                .saturating_add(u64::from(offset == RXDATA));
        }
        Ok(u64::from(value))
    }
    fn write(
        &mut self,
        offset: u64,
        width: Width,
        value: u64,
        access: &mut Access<'_>,
    ) -> Result<(), MmioError> {
        let error = |reason: Refusal| Self::error(offset, width, Direction::Write, reason);
        if width != Width::Word || !offset.is_multiple_of(4) {
            return Err(error(refuse!(
                "SPI registers require aligned 32-bit accesses"
            )));
        }
        self.mmio_ready().map_err(error)?;
        let value = value as u32;
        let saved = (
            self.control,
            self.tx_count,
            self.rx_count,
            self.status,
            self.config,
            self.divider,
            self.delay,
            self.tx,
            self.rx,
            self.configuration,
        );
        match offset {
            CTRL => {
                if value & !(RUN | RX_RESET | TX_RESET) != 0 {
                    return Err(error(refuse!("unmodeled SPI control bits")));
                }
                if value & TX_RESET != 0 {
                    self.tx.clear();
                }
                if value & RX_RESET != 0 {
                    self.rx.clear();
                }
                self.control = value & RUN;
            }
            CFG => {
                if value & !CONFIG_MASK != 0 || !matches!(value & MODE, 0 | IRQ_MODE) {
                    return Err(error(refuse!(
                        "SPI DMA, LSB-first, wide words or unknown configuration"
                    )));
                }
                self.config = value;
            }
            STATUS => self.status &= !value, // v1 W1C, including inactive error flags.
            PIN => {
                if value & !3 != 0 {
                    return Err(error(refuse!("unmodeled SPI pin bits")));
                }
                if !self.external_cs {
                    self.set_cs(value & CS_HIGH != 0).map_err(error)?;
                }
                self.pin = value;
            }
            TXDATA => {
                if value > 0xff {
                    return Err(error(refuse!("SPI byte FIFO write exceeds 8 bits")));
                }
                if self.tx.len() == FIFO_DEPTH {
                    return Err(error(refuse!("SPI TX FIFO overflow")));
                }
                self.tx.push_back(value as u8);
            }
            CLKDIV => {
                if value > 0x7ff {
                    return Err(error(refuse!("SPI divider exceeds 11 bits")));
                }
                self.divider = value;
            }
            DELAY => {
                if value > 0xffff {
                    return Err(error(refuse!("SPI delay exceeds 16 bits")));
                }
                self.delay = value;
            }
            SHIFTCFG => {
                if (value ^ self.shift) & !OVERRIDE_CS != 0 {
                    return Err(error(refuse!("SPI shifter configuration is not modeled")));
                }
                if value & OVERRIDE_CS != 0 && !self.external_cs {
                    return Err(error(refuse!(
                        "shifter-driven internal SPI chip select is not modeled"
                    )));
                }
                self.shift = value;
            }
            DELAY_PRE => self.cs_delays[0] = value,
            DELAY_POST => self.cs_delays[1] = value,
            TXCNT => {
                self.tx_count = value;
                self.status &= !TX_COMPLETE;
            }
            RXCNT => {
                self.rx_count = value;
                self.status &= !RX_COMPLETE;
            }
            _ => return Err(error(refuse!("unmodeled or read-only SPI register"))),
        }
        self.configuration |= configuration_bit(offset);
        if matches!(offset, CTRL | TXDATA | TXCNT | RXCNT)
            && let Err(reason) = self.pump()
        {
            (
                self.control,
                self.tx_count,
                self.rx_count,
                self.status,
                self.config,
                self.divider,
                self.delay,
                self.tx,
                self.rx,
                self.configuration,
            ) = saved;
            return Err(error(reason));
        }
        self.interrupt(access);
        #[cfg(feature = "native-profile")]
        {
            self.profile.writes = self.profile.writes.saturating_add(1);
            self.profile.txdata_writes = self
                .profile
                .txdata_writes
                .saturating_add(u64::from(offset == TXDATA));
        }
        Ok(())
    }

    fn advance(&mut self, access: &mut Access<'_>) -> Result<Option<u64>, MmioError> {
        self.pump()
            .map_err(|reason| Self::error(CTRL, Width::Word, Direction::Write, reason))?;
        self.interrupt(access);
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use stela_contracts::interrupt::InterruptLatch;
    fn write(spi: &mut Spi, offset: u64, value: u32, access: &mut Access<'_>) {
        spi.write(offset, Width::Word, u64::from(value), access)
            .unwrap();
    }

    #[test]
    fn the_fifo_keeps_order_across_its_wrap_and_a_copy_restores_it() {
        let mut fifo = Fifo::default();
        for byte in 0..FIFO_DEPTH as u8 {
            fifo.push_back(byte);
        }
        assert_eq!(fifo.len(), FIFO_DEPTH);
        let full = fifo;
        for byte in 0..10 {
            assert_eq!(fifo.pop_front(), Some(byte));
        }
        // The next pushes wrap past the end of the storage.
        for byte in 100..109 {
            fifo.push_back(byte);
        }
        fifo.push_front(9);
        let order: alloc::vec::Vec<_> = core::iter::from_fn(|| fifo.pop_front()).collect();
        let expected: alloc::vec::Vec<u8> = (9..FIFO_DEPTH as u8).chain(100..109).collect();
        assert_eq!(order, expected);
        assert!(fifo.is_empty() && fifo.front().is_none());
        fifo = full;
        assert_eq!((fifo.len(), fifo.front()), (FIFO_DEPTH, Some(0)));
    }

    #[test]
    fn kept_pauses_preserve_queued_shifts_and_pending_interrupts() {
        let irq = AicInput::new(0, 749);
        let mut spi = Spi::new(Image::fresh(), irq, false);
        let mut latch = InterruptLatch::new();
        write(&mut spi, PIN, 0, &mut Access::new(0, 0, &mut latch));
        write(
            &mut spi,
            CFG,
            IRQ_MODE | IE_TX_COMPLETE,
            &mut Access::new(0, 0, &mut latch),
        );
        let active = spi.lifecycle;
        let paused = LifecycleState {
            quiesced: true,
            ..active
        };
        spi.check_lifecycle(paused).unwrap();
        spi.set_lifecycle(paused, &mut Access::new(0, 0, &mut latch));
        write(&mut spi, TXCNT, 1, &mut Access::new(0, 0, &mut latch));
        write(&mut spi, TXDATA, 0x06, &mut Access::new(0, 0, &mut latch));
        write(&mut spi, CTRL, RUN, &mut Access::new(0, 0, &mut latch));
        assert_eq!((spi.tx_count, spi.tx.front()), (1, Some(0x06)));
        assert_eq!(spi.status, 0);
        spi.advance(&mut Access::new(0, 0, &mut latch)).unwrap();
        assert_eq!(spi.tx_count, 1);
        spi.set_lifecycle(active, &mut Access::new(0, 0, &mut latch));
        spi.advance(&mut Access::new(0, 0, &mut latch)).unwrap();
        assert!(spi.tx.is_empty());
        assert!(latch.is_asserted(Wire::Aic(irq)));
        for paused in [
            LifecycleState {
                clocked: false,
                ..active
            },
            LifecycleState {
                powered: false,
                retention: true,
                ..active
            },
        ] {
            spi.check_lifecycle(paused).unwrap();
            spi.set_lifecycle(paused, &mut Access::new(0, 0, &mut latch));
            assert!(!latch.is_asserted(Wire::Aic(irq)));
            assert!(
                spi.read(STATUS, Width::Word, &mut Access::new(0, 0, &mut latch))
                    .is_err()
            );
            spi.set_lifecycle(active, &mut Access::new(0, 0, &mut latch));
            assert!(latch.is_asserted(Wire::Aic(irq)));
        }
        write(&mut spi, PIN, CS_HIGH, &mut Access::new(0, 0, &mut latch));
        spi.nor.select(true).unwrap();
        spi.nor.shift(0x05).unwrap();
        assert_eq!(
            spi.nor.shift(0xff).unwrap() & 2,
            2,
            "WREN committed at the actual CS edge"
        );
    }

    #[test]
    fn destructive_preflight_keeps_incomplete_internal_cs_transactions_unchanged() {
        let irq = AicInput::new(0, 749);
        let mut spi = Spi::new(Image::fresh(), irq, false);
        spi.nor.select(true).unwrap();
        spi.nor.shift(0x02).unwrap();
        let active = spi.lifecycle;
        for proposed in [
            LifecycleState {
                reset: true,
                ..active
            },
            LifecycleState {
                powered: false,
                ..active
            },
        ] {
            assert_eq!(
                spi.check_lifecycle(proposed),
                Err(refuse!("incomplete NOR command at CS rising"))
            );
            assert_eq!(spi.lifecycle, active);
            assert!(spi.nor.selected());
            assert!(spi.take_dirty().is_empty());
        }
    }

    #[test]
    fn destructive_edges_preserve_the_die_and_only_cs_edges_commit_programs() {
        for external_cs in [false, true] {
            for reset in [false, true] {
                let irq = AicInput::new(0, 749);
                let mut spi = Spi::new(Image::fresh(), irq, external_cs);
                spi.nor.select(true).unwrap();
                spi.nor.shift(0x06).unwrap();
                spi.nor.select(false).unwrap();
                spi.nor.select(true).unwrap();
                for byte in [0x02, 0, 0, 0x20, 0xa5] {
                    spi.nor.shift(byte).unwrap();
                }
                let mut latch = InterruptLatch::new();
                let mut a = Access::new(0, 0, &mut latch);
                let active = spi.lifecycle;
                let down = LifecycleState {
                    powered: reset,
                    reset,
                    ..active
                };
                spi.check_lifecycle(down).unwrap();
                spi.set_lifecycle(down, &mut a);
                assert_eq!(spi.nor.selected(), external_cs);
                assert_eq!(
                    spi.image().bytes()[0x20],
                    if external_cs { 0xff } else { 0xa5 }
                );
                if external_cs {
                    assert!(spi.take_dirty().is_empty());
                    spi.set_cs(true).unwrap();
                }
                assert_eq!(spi.image().bytes()[0x20], 0xa5);
                assert_eq!(spi.take_dirty().len(), 1);
                spi.set_lifecycle(down, &mut a);
                assert!(
                    spi.take_dirty().is_empty(),
                    "held reset cannot replay the CS edge"
                );
                assert!(spi.read(CTRL, Width::Word, &mut a).is_err());
                spi.set_lifecycle(active, &mut a);
                assert!(spi.read(CTRL, Width::Word, &mut a).is_err());
                write(&mut spi, CTRL, 0, &mut a);
                assert_eq!(spi.read(CTRL, Width::Word, &mut a), Ok(0));
                assert!(spi.read(CFG, Width::Word, &mut a).is_err());
                assert_eq!(spi.read(DELAY_PRE, Width::Word, &mut a), Ok(0));
            }
        }
    }
    #[test]
    fn long_read_stalls_at_fifo_capacity_and_interrupt_clears_after_drain() {
        use stela_soc_data::nor::CAPACITY;
        let image = Image::from_bytes((0..CAPACITY).map(|i| i as u8).collect()).unwrap();
        let irq = AicInput::new(0, 749);
        let mut spi = Spi::new(image, irq, false);
        let mut latch = InterruptLatch::new();
        {
            let mut a = Access::new(0, 0, &mut latch);
            write(&mut spi, PIN, 0, &mut a);
            write(&mut spi, RXCNT, 4, &mut a);
            write(&mut spi, TXCNT, 4, &mut a);
            for b in [3, 0, 0, 0] {
                write(&mut spi, TXDATA, b, &mut a);
            }
            write(&mut spi, CTRL, RUN, &mut a);
            for _ in 0..4 {
                spi.read(RXDATA, Width::Word, &mut a).unwrap();
            }
            write(&mut spi, CTRL, TX_RESET | RX_RESET, &mut a);
            write(&mut spi, RXCNT, 300, &mut a);
            write(&mut spi, TXCNT, 300, &mut a);
            write(
                &mut spi,
                CFG,
                IRQ_MODE | IE_RX_COMPLETE | IE_THRESHOLD | 2 << 17,
                &mut a,
            );
            for _ in 0..FIFO_DEPTH {
                write(&mut spi, TXDATA, 0xff, &mut a);
            }
            write(&mut spi, CTRL, RUN, &mut a);
            assert_eq!(spi.rx.len(), FIFO_DEPTH);
            assert_eq!(spi.rx_count, 284);
            let mut sent = FIFO_DEPTH;
            for i in 0..300 {
                while sent < 300 && spi.tx.len() < FIFO_DEPTH {
                    write(&mut spi, TXDATA, 0xff, &mut a);
                    sent += 1;
                }
                assert_eq!(
                    spi.read(RXDATA, Width::Word, &mut a),
                    Ok(u64::from(i as u8))
                );
                assert!(spi.rx.len() <= FIFO_DEPTH);
            }
            assert_eq!((spi.rx_count, spi.tx_count), (0, 0));
        }
        assert!(latch.is_asserted(Wire::Aic(irq)));
        let mut a = Access::new(0, 0, &mut latch);
        write(&mut spi, STATUS, RX_COMPLETE | TX_COMPLETE, &mut a);
        assert!(!latch.is_asserted(Wire::Aic(irq)));
    }

    #[test]
    fn unsupported_nor_byte_does_not_start_or_consume_a_transfer() {
        let mut spi = Spi::new(Image::fresh(), AicInput::new(0, 749), false);
        let mut latch = InterruptLatch::new();
        let mut a = Access::new(0, 0, &mut latch);
        write(&mut spi, PIN, 0, &mut a);
        write(&mut spi, TXCNT, 1, &mut a);
        write(&mut spi, TXDATA, 0x5a, &mut a);
        assert!(spi.write(CTRL, Width::Word, 1, &mut a).is_err());
        assert_eq!((spi.control, spi.tx_count), (0, 1));
        assert_eq!(spi.tx.front(), Some(0x5a));
        write(&mut spi, CTRL, TX_RESET, &mut a);
        write(&mut spi, TXDATA, 0x06, &mut a);
        write(&mut spi, CTRL, RUN, &mut a);
        write(&mut spi, PIN, CS_HIGH, &mut a);
        assert_eq!(spi.nor.shift(0), Ok(0xff));
    }

    #[test]
    fn kernel_pio_jedec_sequence_and_w1c_interrupt() {
        let irq = AicInput::new(0, 749);
        let mut spi = Spi::new(Image::fresh(), irq, false);
        let mut latch = InterruptLatch::new();
        let mut a = Access::new(0, 0, &mut latch);
        write(&mut spi, PIN, 0, &mut a);
        write(&mut spi, CTRL, TX_RESET | RX_RESET, &mut a);
        write(&mut spi, RXCNT, 4, &mut a);
        write(&mut spi, TXCNT, 4, &mut a);
        write(&mut spi, CFG, IRQ_MODE | IE_RX_COMPLETE | 2 << 17, &mut a);
        for byte in [0x9f, 0xff, 0xff, 0xff] {
            write(&mut spi, TXDATA, byte, &mut a);
        }
        write(&mut spi, CTRL, RUN, &mut a);
        assert_eq!(
            spi.read(STATUS, Width::Word, &mut a).unwrap() & (1 | 1 << 22),
            1 | 1 << 22
        );

        let bytes: alloc::vec::Vec<_> = (0..4)
            .map(|_| spi.read(RXDATA, Width::Word, &mut a).unwrap())
            .collect();
        assert_eq!(bytes, [0xff, 0xef, 0x60, 0x17]);
        write(&mut spi, PIN, CS_HIGH, &mut a);
        write(&mut spi, STATUS, 0x400023, &mut a);
        assert_eq!(spi.read(STATUS, Width::Word, &mut a), Ok(0));
        #[cfg(feature = "native-profile")]
        {
            assert!(spi.read(RXDATA, Width::Word, &mut a).is_err());
            assert!(spi.write(TXDATA, Width::Word, 0x100, &mut a).is_err());
            assert!(
                spi.take_profile()
                    .starts_with("mmio_reads=6 mmio_writes=12 rxdata_reads=4 txdata_writes=4 ")
            );
            assert!(
                spi.take_profile()
                    .starts_with("mmio_reads=0 mmio_writes=0 rxdata_reads=0 txdata_writes=0 ")
            );
        }
        assert!(!latch.is_asserted(Wire::Aic(irq)));
    }
    #[test]
    fn unknown_modes_and_fifo_errors_refuse() {
        let mut spi = Spi::new(Image::fresh(), AicInput::new(0, 1), false);
        let mut latch = InterruptLatch::new();
        let mut a = Access::new(0, 0, &mut latch);
        assert!(spi.write(CFG, Width::Word, 2 << 5, &mut a).is_err());
        assert!(spi.read(RXDATA, Width::Word, &mut a).is_err());
        for _ in 0..FIFO_DEPTH {
            write(&mut spi, TXDATA, 0xff, &mut a);
        }
        assert!(spi.write(TXDATA, Width::Word, 0xff, &mut a).is_err());
        assert_eq!(spi.tx.len(), FIFO_DEPTH);
        assert!(spi.read(0x134, Width::Word, &mut a).is_err());
        assert!(spi.write(CTRL, Width::Double, 1, &mut a).is_err());
    }
    #[test]
    fn shifter_cs_override_and_delays_are_kept_configuration() {
        let mut latch = InterruptLatch::new();
        let mut access = Access::new(0, 0, &mut latch);
        let mut external = Spi::new(Image::fresh(), AicInput::new(0, 1), true);
        assert_eq!(external.read(SHIFTCFG, Width::Word, &mut access), Ok(0));
        external
            .write(SHIFTCFG, Width::Word, OVERRIDE_CS.into(), &mut access)
            .unwrap();
        external
            .write(SHIFTCFG, Width::Word, 0, &mut access)
            .unwrap();
        assert!(
            external
                .write(SHIFTCFG, Width::Word, 1, &mut access)
                .is_err()
        );
        external
            .write(DELAY_PRE, Width::Word, 0x823 | 9 << 16, &mut access)
            .unwrap();
        assert_eq!(
            external.read(DELAY_PRE, Width::Word, &mut access),
            Ok(0x823 | 9 << 16)
        );
        external
            .write(DELAY_POST, Width::Word, 0, &mut access)
            .unwrap();
        let mut internal = Spi::new(Image::fresh(), AicInput::new(0, 1), false);
        assert!(
            internal
                .write(SHIFTCFG, Width::Word, OVERRIDE_CS.into(), &mut access)
                .is_err()
        );
        assert_eq!(internal.read(SHIFTCFG, Width::Word, &mut access), Ok(0));
    }
}
