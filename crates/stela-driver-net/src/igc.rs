//! Intel I225 and I226 Ethernet controllers, as a polled raw Ethernet
//! [`Link`]: the 2.5GBASE-T controllers most current PC boards carry.
//!
//! Reset, NVM and PHY setup follow the pinned Linux `igc` shared code.
//! The declared boot-lifetime profile uses one polled receive/transmit queue,
//! ordinary Ethernet, autonegotiated 10/100/1000/2500 copper with Linux's
//! default full flow control, and no EEE.
//! Management-owned PHYs, blank NVM, failed register reads and incomplete
//! reset/MDIO/semaphore exchanges fail before any new DMA ring is published.
mod hardware;
mod link;

pub use hardware::{FlowControl, Negotiated, Statistics};

use super::{
    Link, LinkError, MAX_FRAME, Mac,
    advanced::{
        self, QUEUE_ENABLE, QUEUE_WAIT_US, RX_DD, RX_EOP, RXDCTL_THRESHOLDS, SRRCTL_ONE_BUFFER,
        TXDCTL_THRESHOLDS, TransmitQueue, transmitted,
    },
    check_frame,
    ring::{DESCRIPTOR, Ring},
};
use crate::{
    Handoff,
    block::BlockError,
    bus::{
        dma::DmaAllocator,
        mmio::Mmio,
        pci::{self, Function},
    },
    registers,
};
use alloc::{format, string::String, vec::Vec};
use core::{
    mem::ManuallyDrop,
    sync::atomic::{Ordering, fence},
};

const VENDOR: u16 = 0x8086;
/// igc.h IGC_DEFAULT_RXD and IGC_DEFAULT_TXD.
const DESCRIPTORS: usize = 256;
/// The I220, I221, I225 and I226 parts with a programmed NVM (`igc_hw.h`).
const DEVICES: [u16; 14] = [
    0x15f2, 0x15f3, 0x15f7, 0x15f8, 0x3100, 0x3101, 0x3102, 0x5502, 0x5503, 0x0d9f, 0x125b, 0x125c,
    0x125d, 0x125e,
];

const STATUS: usize = 0x0008;
/// Interrupt registers igc_configure_msix and igc_irq_enable program.
const ICR: usize = 0x1500;
const IMS: usize = 0x1508;
const GPIE: usize = 0x1514;
const EIMS: usize = 0x1524;
const EIAC: usize = 0x152c;
const EIAM: usize = 0x1530;
const EITR0: usize = 0x1680;
const IVAR0: usize = 0x1700;
const IVAR_MISC: usize = 0x1740;
const GPIE_NSICR: u32 = 1;
const GPIE_MSIX_MODE: u32 = 1 << 4;
const GPIE_EIAME: u32 = 1 << 30;
const GPIE_PBA: u32 = 1 << 31;
const IVAR_VALID: u32 = 0x80;
const IMS_LSC: u32 = 1 << 2;
/// igc.h IGC_START_ITR, with IGC_EITR_CNT_IGNR as igc_write_itr sets it.
const START_ITR: u32 = 648 | 1 << 31;
/// The one MSI-X vector: queue 0's receive and the link causes.
const VECTOR: u32 = 0;
/// An MSI-X table entry's bytes.
const MSIX_ENTRY_BYTES: u64 = 16;
const RCTL: usize = 0x0100;
const TCTL: usize = 0x0400;
const RAL: usize = 0x5400;
const RAH: usize = 0x5404;
/// Queue 0's receive and transmit registers.
const RDBAL: usize = 0xc000;
const RDBAH: usize = 0xc004;
const RDLEN: usize = 0xc008;
const SRRCTL: usize = 0xc00c;
const RDH: usize = 0xc010;
const RDT: usize = 0xc018;
const RXDCTL: usize = 0xc028;
const TDBAL: usize = 0xe000;
const TDBAH: usize = 0xe004;
const TDLEN: usize = 0xe008;
const TDH: usize = 0xe010;
const TDT: usize = 0xe018;
const TXDCTL: usize = 0xe028;
/// The register file the driver touches.
const REGISTER_BYTES: u64 = 0x1_2018; // Includes EERD for validated programmed NVM.

const RAH_AV: u32 = 1 << 31;
const STATUS_TXOFF: u32 = 1 << 4;
/// igc_clean_tx_irq's hang check: a descriptor the controller has held this
/// long, scaled by igc_watchdog_task's tx_timeout_factor at 10 Mb/s.
const TRANSMIT_HANG_US: u64 = 1_000_000;
const RCTL_EN: u32 = 1 << 1;
const RCTL_MPE: u32 = 1 << 4;
const RCTL_BAM: u32 = 1 << 15;
const RCTL_SECRC: u32 = 1 << 26;
const TCTL_EN: u32 = 1 << 1;
const TCTL_PSP: u32 = 1 << 3;
const TCTL_CT: u32 = 0xff << 4;
const TCTL_RTLC: u32 = 1 << 24;
const COLLISION_THRESHOLD: u32 = 15 << 4;
const RX_RXE: u32 = 1 << 31;
/// igc_xmit_frame: TCTL.PSP still requires at least 17 DMA bytes.
const MIN_TRANSMIT: usize = 17;
const TRANSMIT_QUEUE: TransmitQueue = TransmitQueue {
    head: TDH,
    tail: TDT,
    control: TXDCTL,
};

pub fn matches(function: &Function) -> bool {
    matches_identity(&function.identity())
}

/// Candidate selection from copied identity DATA; owned attachment still
/// revalidates configuration and platform requirements.
pub fn matches_identity(function: &crate::bus::pci::Identity) -> bool {
    function.vendor == VENDOR && DEVICES.contains(&function.device) && function.class == [2, 0, 0]
}

fn prepare_link(
    device: u16,
    disable: impl FnOnce() -> Result<(), BlockError>,
) -> Result<(), BlockError> {
    // Pinned igc_base.c::igc_is_device_id_i226. LMVP5503 is not
    // guessed into this exact donor erratum predicate.
    if matches!(device, 0x125b | 0x125c | 0x125d | 0x3102) {
        disable()?;
    }
    Ok(())
}

/// The station address in receive address register 0, which the NVM
/// auto-read loads: RAL holds bytes 0-3, RAH bytes 4-5 and the valid bit.
fn station(low: u32, high: u32) -> Option<Mac> {
    if high == u32::MAX || low == u32::MAX || high & RAH_AV == 0 {
        return None;
    }
    let [a, b, c, d] = low.to_le_bytes();
    let [e, f, _, _] = high.to_le_bytes();
    Some(Mac([a, b, c, d, e, f])).filter(|mac| mac.0 != [0; 6] && mac.0[0] & 1 == 0)
}

/// A written-back receive descriptor's frame length, if the descriptor is
/// done; `Some(None)` for a frame to drop: one with errors, or one longer
/// than a buffer (long packets are off, so it cannot fit).
fn received(status: u32, length: u16) -> Option<Option<usize>> {
    if status & RX_DD == 0 {
        return None;
    }
    let length = usize::from(length);
    let good = status & RX_EOP != 0 && status & RX_RXE == 0 && length > 0 && length <= MAX_FRAME;
    Some(good.then_some(length))
}

pub struct Igc {
    device: Mmio,
    /// Released only after a completed stop: an unstopped controller may
    /// still write either ring, so Drop then keeps both allocations.
    receive: ManuallyDrop<Ring>,
    transmit: ManuallyDrop<Ring>,
    /// Since when the transmit ring has had no free descriptor.
    stalled: Option<u64>,
    /// Discard the rest of a split/oversized frame through its EOP.
    discarding: bool,
    /// An absent/failed MMIO function requires a fresh attachment.
    faulted: bool,
    /// Carrier supervision; its negotiated link while the carrier is up.
    link: link::Supervisor,
    statistics: Statistics,
    mac: Mac,
    clock: crate::timing::Timing,
    timing_failure: Option<BlockError>,
    pub location: String,
    /// Internal PHY identifier, including its silicon revision.
    pub phy_identity: u32,
    /// Whether MSI-X vector 0 sends the handoff's completion signal.
    signalled: bool,
}

impl Igc {
    /// Attach a physical integrated copper controller after ExitBootServices.
    ///
    /// # Safety
    /// The caller exclusively owns the controller, PCI segment configuration
    /// and firmware/ACPI ASPM policy. The handoff describes mapped ECAM,
    /// MMIO and boot-lifetime DMA ranges. No firmware or other driver may
    /// access these resources or restore disabled link states.
    pub unsafe fn new(
        function: &mut Function,
        handoff: &Handoff,
        dma: &mut dyn DmaAllocator,
        minimum: alloc::rc::Rc<dyn crate::bus::pci::power::Timer>,
    ) -> Result<Self, BlockError> {
        let clock = crate::timing::Timing::checked(handoff.clock, minimum)?;
        if !matches(function) {
            return Err(BlockError::Unsupported(
                "not an Intel I225/I226 copper candidate",
            ));
        }
        prepare_link(function.device, || {
            // SAFETY: the caller grants exclusive segment/ASPM ownership;
            // apply the erratum before BAR probing, decode or bus mastering.
            unsafe { pci::disable_aspm_l1_2_link(function, handoff) }
        })?;
        function.power_up(clock.minimum())?;
        let bars = function.bars()?;
        let device = registers(handoff, bars[0], 0, REGISTER_BYTES)?;
        function.enable_polled()?;
        let signalled = match (handoff.completion, function.interrupts()?.msix) {
            (Some(message), Some(msix)) => {
                let table = registers(
                    handoff,
                    bars[usize::from(msix.table_bar)],
                    u64::from(msix.table_offset),
                    MSIX_ENTRY_BYTES,
                )?;
                function.enable_msix(msix, || {
                    table.write32(0, message.address as u32);
                    table.write32(4, (message.address >> 32) as u32);
                    table.write32(8, message.data);
                    table.write32(12, 0);
                    Ok(())
                })?;
                true
            }
            _ => false,
        };
        let mut hardware = hardware::Hardware::new(&device, &clock)?;
        let phy_identity = hardware.initialize()?;
        let mac = station(device.read32(RAL), device.read32(RAH)).ok_or(
            BlockError::Unsupported("Intel I225/I226 without a valid station address"),
        )?;
        let mut net = Self {
            device,
            receive: ManuallyDrop::new(Ring::with(dma, DESCRIPTORS)?),
            transmit: ManuallyDrop::new(Ring::with(dma, DESCRIPTORS)?),
            stalled: None,
            discarding: false,
            faulted: false,
            timing_failure: None,
            link: link::Supervisor::new(),
            statistics: Statistics::default(),
            mac,
            clock,
            location: format!("{}", function.address),
            phy_identity,
            signalled,
        };
        net.start_receive()?;
        net.start_transmit()?;
        Ok(net)
    }

    fn start_receive(&mut self) -> Result<(), BlockError> {
        let device = &self.device;
        device.write32(RCTL, 0);
        self.receive.program(
            |offset, value| device.write32(offset, value),
            RDBAL,
            RDBAH,
            RDLEN,
        );
        device.write32(SRRCTL, SRRCTL_ONE_BUFFER);
        device.write32(RDH, 0);
        device.write32(RDT, 0);
        device.write32(RXDCTL, RXDCTL_THRESHOLDS | QUEUE_ENABLE);
        if !self.clock.wait(QUEUE_WAIT_US, || {
            let value = device.read32(RXDCTL);
            value != u32::MAX && value & QUEUE_ENABLE != 0
        }) {
            return Err(BlockError::Unsupported(
                "Intel I225/I226 receive queue did not enable",
            ));
        }
        hardware::flush_receive_fifo(&self.device, &self.clock)?;
        for index in 0..DESCRIPTORS {
            advanced::offer_receive(&mut self.receive, index);
        }
        // Head equal to tail is an empty ring: the controller owns every
        // descriptor but the one before the head.
        fence(Ordering::SeqCst);
        self.device.write32(RDT, (DESCRIPTORS - 1) as u32);
        // Link has no multicast-list callback. Admit all multicast, as the
        // donor's IFF_ALLMULTI path does, so IPv6/other multicast reaches it.
        self.device
            .write32(RCTL, RCTL_EN | RCTL_MPE | RCTL_BAM | RCTL_SECRC);
        if self.signalled {
            self.arm_interrupts();
        }
        Ok(())
    }

    /// igc_configure_msix and igc_irq_enable for one vector: queue 0's
    /// receive and the link causes send it, auto-masked until the next
    /// empty receive re-arms it.
    fn arm_interrupts(&self) {
        let device = &self.device;
        device.write32(GPIE, GPIE_MSIX_MODE | GPIE_PBA | GPIE_EIAME | GPIE_NSICR);
        device.write32(IVAR_MISC, (VECTOR | IVAR_VALID) << 8);
        device.write32(IVAR0, VECTOR | IVAR_VALID);
        device.write32(EITR0, START_ITR);
        device.write32(EIAC, device.read32(EIAC) | 1 << VECTOR);
        device.write32(EIAM, device.read32(EIAM) | 1 << VECTOR);
        device.write32(EIMS, 1 << VECTOR);
        device.write32(IMS, IMS_LSC);
    }

    fn start_transmit(&mut self) -> Result<(), BlockError> {
        let device = &self.device;
        self.transmit.program(
            |offset, value| device.write32(offset, value),
            TDBAL,
            TDBAH,
            TDLEN,
        );
        device.write32(TDT, 0);
        device.write32(TDH, 0);
        device.write32(TXDCTL, TXDCTL_THRESHOLDS | QUEUE_ENABLE);
        if !self.clock.wait(QUEUE_WAIT_US, || {
            let value = device.read32(TXDCTL);
            value != u32::MAX && value & QUEUE_ENABLE != 0
        }) {
            return Err(BlockError::Unsupported(
                "Intel I225/I226 transmit queue did not enable",
            ));
        }
        let control = device.read32(TCTL);
        if control == u32::MAX {
            return Err(BlockError::Device { status: control });
        }
        let control = control & !TCTL_CT;
        device.write32(
            TCTL,
            control | TCTL_PSP | TCTL_RTLC | TCTL_EN | COLLISION_THRESHOLD,
        );
        Ok(())
    }

    /// Linux igc_down: record the counters, then stop the controller. Errors
    /// leave its DMA state uncertain.
    fn stop(&mut self) -> Result<(), BlockError> {
        if let Some(error) = &self.timing_failure {
            return Err(error.clone());
        }
        let _ = self.statistics.accumulate(&self.device);
        hardware::Hardware::new(&self.device, &self.clock)?.stop()
    }

    /// Linux igc_reinit_locked, as igc_reset_task runs it for a transmit
    /// hang: a completed stop, full reinitialization, and both rings
    /// restarted empty on their existing buffers. The carrier then follows
    /// a fresh link check. Any failure leaves the adapter faulted.
    fn reinitialize(&mut self) -> Result<(), BlockError> {
        let result = (|| {
            self.stop()?;
            self.phy_identity = hardware::Hardware::new(&self.device, &self.clock)?.initialize()?;
            self.receive.reset();
            self.transmit.reset();
            self.discarding = false;
            self.stalled = None;
            self.link = link::Supervisor::new();
            self.start_receive()?;
            self.start_transmit()
        })();
        self.faulted |= result.is_err();
        if let Err(error) = &result
            && matches!(error, BlockError::PciPower(_))
        {
            self.timing_failure = Some(error.clone());
        }
        result
    }

    /// The negotiated link while the carrier is up: speed, duplex and the
    /// pause frames both sides agreed on.
    pub fn negotiated(&self) -> Option<Negotiated> {
        self.link.carrier()
    }

    /// The MAC counters accumulated since attachment, after reading the
    /// controller's clear-on-read registers (Linux igc_update_stats).
    pub fn statistics(&mut self) -> Result<Statistics, LinkError> {
        if self.faulted || self.statistics.accumulate(&self.device).is_err() {
            self.faulted = true;
            return Err(LinkError::Adapter(format!(
                "Intel I225/I226 {} lost register access; reinitialization required",
                self.location
            )));
        }
        Ok(self.statistics)
    }

    /// Report a lost function, and discard the transmit work a lost link
    /// strands (Linux `igc_watchdog_task` resets for it). Returns the carrier.
    fn watchdog(&mut self) -> Result<bool, LinkError> {
        if let Some(error) = &self.timing_failure {
            return Err(LinkError::Pci(error.clone()));
        }
        let change = match self.faulted {
            true => link::Change::None,
            false => match self
                .link
                .poll(&self.device, &self.clock, self.clock.now_us())
            {
                Ok(change) => change,
                Err(error) => {
                    self.faulted = true;
                    if matches!(error, BlockError::PciPower(_)) {
                        self.timing_failure = Some(error.clone());
                        return Err(LinkError::Pci(error));
                    }
                    link::Change::None
                }
            },
        };
        if change == link::Change::Down && self.transmit.pending() {
            let device = self.device;
            let flushed = TRANSMIT_QUEUE.flush(
                &mut self.transmit,
                self.clock.nominal(),
                |offset| device.read32(offset),
                |offset, value| device.write32(offset, value),
            );
            // A queue that will not stop needs the reset Linux uses here.
            if !flushed
                && let Err(error) = self.reinitialize()
                && matches!(error, BlockError::PciPower(_))
            {
                self.timing_failure = Some(error.clone());
                return Err(LinkError::Pci(error));
            }
        }
        match self.faulted {
            true => Err(LinkError::Adapter(format!(
                "Intel I225/I226 {} lost register access; reinitialization required",
                self.location
            ))),
            false => Ok(self.link.carrier().is_some()),
        }
    }
}

impl Igc {
    /// A full transmit ring. Linux treats descriptors held past the hang
    /// timeout as a hang unless PAUSE frames hold the transmitter off
    /// (STATUS.TXOFF), and resets the controller for it.
    fn transmit_stalled(&mut self) -> LinkError {
        let now = self.clock.now_us();
        let since = *self.stalled.get_or_insert(now);
        let status = self.device.read32(STATUS);
        let factor = match self.link.carrier() {
            Some(link) if link.speed == 10 => 14,
            _ => 1,
        };
        let held = now.saturating_sub(since);
        if status == u32::MAX {
            self.faulted = true;
        } else if status & STATUS_TXOFF == 0 && held >= TRANSMIT_HANG_US * factor {
            let reset = self.reinitialize();
            if let Err(error) = &reset
                && matches!(error, BlockError::PciPower(_))
            {
                return LinkError::Pci(error.clone());
            }
            return LinkError::Adapter(format!(
                "Intel I225/I226 {} transmit hang; controller reset {}",
                self.location,
                if reset.is_ok() { "completed" } else { "failed" }
            ));
        }
        LinkError::Adapter(format!(
            "Intel I225/I226 {} transmit ring full{}",
            self.location,
            if status & STATUS_TXOFF != 0 && status != u32::MAX {
                " while paused by flow control"
            } else {
                ""
            }
        ))
    }
}

impl Drop for Igc {
    /// Linux igc_down then igc_release_hw_control. The rings are freed only
    /// once the controller verifiably stopped; otherwise they stay allocated.
    ///
    /// This drives the controller's registers, so the owner drops it where
    /// its MMIO window is mapped; anywhere else the first read faults.
    fn drop(&mut self) {
        if self.stop().is_ok() {
            let _ =
                hardware::Hardware::new(&self.device, &self.clock).map(|hw| hw.release_control());
            // SAFETY: the completed MAC reset stopped every descriptor and
            // buffer access, and neither ring is used after this.
            unsafe {
                ManuallyDrop::drop(&mut self.receive);
                ManuallyDrop::drop(&mut self.transmit);
            }
        }
    }
}

impl Link for Igc {
    fn mac(&self) -> Mac {
        self.mac
    }

    fn carrier(&mut self) -> bool {
        self.watchdog().unwrap_or(false)
    }

    fn transmit(&mut self, frame: &[u8]) -> Result<(), LinkError> {
        check_frame(frame)?;
        if !self.watchdog()? {
            return Err(LinkError::NoCarrier);
        }
        if !self.transmit.free(self.clock.nominal(), transmitted) {
            return Err(self.transmit_stalled());
        }
        self.stalled = None;
        advanced::queue_frame(&mut self.transmit, frame, frame.len().max(MIN_TRANSMIT));
        self.device.write32(TDT, self.transmit.next as u32);
        Ok(())
    }

    fn signals(&self) -> bool {
        self.signalled
    }

    fn receive(&mut self) -> Result<Option<Vec<u8>>, LinkError> {
        self.receive_frame(<[u8]>::to_vec)
    }

    /// The stack's receive path: the frame is copied into its reused
    /// buffer, so a received frame allocates nothing.
    fn receive_reusing(&mut self, frame: &mut Vec<u8>) -> Result<bool, LinkError> {
        self.receive_frame(|bytes| {
            frame.clear();
            frame.extend_from_slice(bytes);
        })
        .map(|received| received.is_some())
    }
}

impl Igc {
    /// The next whole frame the controller wrote back, passed to `take`
    /// while its buffer is still held.
    fn receive_frame<T>(
        &mut self,
        mut take: impl FnMut(&[u8]) -> T,
    ) -> Result<Option<T>, LinkError> {
        self.watchdog()?;
        loop {
            let index = self.receive.next;
            let base = index * DESCRIPTOR;
            fence(Ordering::SeqCst);
            let status: u32 = self.receive.descriptors.read(base + 8);
            let length: u16 = self.receive.descriptors.read(base + 12);
            let Some(frame) = received(status, length) else {
                if let Some(tail) = self.receive.idle() {
                    fence(Ordering::SeqCst);
                    self.device.write32(RDT, tail);
                }
                if self.signalled {
                    // igc_msix_other reads the link cause, clearing it so
                    // the next change sends the vector; igc_ring_irq_enable
                    // unmasks the vector, auto-masked when it fired.
                    let link = self.device.read32(ICR) & IMS_LSC != 0;
                    self.device.write32(EIMS, 1 << VECTOR);
                    if link {
                        // This read cleared the cause the watchdog reads.
                        self.link.link_changed();
                        self.watchdog()?;
                    }
                }
                return Ok(None);
            };
            let discarded = self.discarding || status & RX_EOP == 0;
            self.discarding = status & RX_EOP == 0;
            let frame = frame
                .filter(|_| !discarded)
                .map(|length| take(&self.receive.buffer(index)[..length]));
            advanced::offer_receive(&mut self.receive, index);
            self.receive.advance();
            if let Some(tail) = self.receive.offered(index) {
                fence(Ordering::SeqCst);
                self.device.write32(RDT, tail);
            }
            if frame.is_some() {
                return Ok(frame);
            }
        }
    }
}

#[cfg(test)]
use crate::Clock;

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::rc::Rc;

    /// An I225 over modeled registers and DMA memory, both rings started
    /// and the link up.
    const LINK: Negotiated = Negotiated {
        speed: 2500,
        full_duplex: true,
        flow_control: FlowControl::Full,
    };

    struct Fixture {
        net: Igc,
        registers: Vec<u32>,
        _backing: Vec<u64>,
    }

    impl Fixture {
        fn new() -> Self {
            use crate::bus::dma::RegionAllocator;
            let mut backing = vec![0u64; 256 * 1024];
            let start = backing.as_mut_ptr() as u64;
            // SAFETY: backing remains live until all modeled DMA buffers drop.
            let mut dma =
                unsafe { RegionAllocator::new(start..start + (backing.len() * 8) as u64) };
            let mut registers = vec![0u32; REGISTER_BYTES as usize / 4];
            // SAFETY: the vector is this fixture's exclusive live MMIO model.
            let device =
                unsafe { Mmio::new(registers.as_mut_ptr() as u64, size_of_val(&registers[..])) };
            let mut net = Igc {
                device,
                receive: ManuallyDrop::new(Ring::with(&mut dma, DESCRIPTORS).unwrap()),
                transmit: ManuallyDrop::new(Ring::with(&mut dma, DESCRIPTORS).unwrap()),
                stalled: None,
                discarding: false,
                faulted: false,
                timing_failure: None,
                link: link::Supervisor::established(LINK),
                statistics: Statistics::default(),
                mac: Mac([2, 0, 0, 0, 0, 1]),
                clock: crate::timing::model(Clock {
                    ticks_per_microsecond: 1,
                }),
                location: "model".into(),
                phy_identity: 0x67c9dc01,
                signalled: false,
            };
            net.start_receive().unwrap();
            net.start_transmit().unwrap();
            Self {
                net,
                registers,
                _backing: backing,
            }
        }
    }

    /// An empty receive unmasks the vector, auto-masked when it fired.
    #[test]
    fn empty_receive_unmasks_the_signal_vector() {
        let mut fixture = Fixture::new();
        fixture.net.signalled = true;
        assert!(fixture.net.signals());
        assert_eq!(fixture.net.receive(), Ok(None));
        assert_eq!(fixture.registers[EIMS / 4], 1 << VECTOR);
    }

    #[test]
    fn link_loss_discards_queued_transmit_work_and_refuses_new_frames() {
        let mut fixture = Fixture::new();
        for _ in 0..3 {
            fixture.net.transmit(&[1; 60]).unwrap();
        }
        assert_eq!(fixture.registers[TDT / 4], 3);
        // A link change whose PHY check finds no link (here: no PHY answers).
        let _clock = crate::test_clock::virtualize();
        fixture.registers[ICR / 4] = 1 << 2;
        assert_eq!(fixture.net.transmit(&[1; 60]), Err(LinkError::NoCarrier));
        assert_eq!(fixture.net.negotiated(), None);
        assert!(!fixture.net.transmit.pending());
        assert_eq!(fixture.registers[TDT / 4], 0);
        assert_ne!(fixture.registers[TXDCTL / 4] & QUEUE_ENABLE, 0);
        fixture.registers[ICR / 4] = 0;
        fixture.net.link = link::Supervisor::established(LINK);
        fixture.net.transmit(&[2; 60]).unwrap();
        assert_eq!(fixture.registers[TDT / 4], 1);
    }

    #[test]
    fn a_held_ring_resets_only_after_the_hang_timeout_and_never_while_paused() {
        let _clock = crate::test_clock::virtualize();
        let error = |fixture: &mut Fixture| match fixture.net.transmit(&[2; 60]) {
            Err(LinkError::Adapter(message)) => message,
            other => panic!("{other:?}"),
        };
        let full = || {
            let mut fixture = Fixture::new();
            for _ in 0..DESCRIPTORS - 1 {
                fixture.net.transmit(&[1; 60]).unwrap();
            }
            fixture
        };
        // A descriptor held past one second is a hang; the reset then runs
        // (and here cannot complete).
        let mut fixture = full();
        let mut attempts = 0;
        let message = loop {
            attempts += 1;
            let message = error(&mut fixture);
            if !message.ends_with("transmit ring full") {
                break message;
            }
        };
        assert!(attempts > 1);
        assert!(message.ends_with("transmit hang; controller reset failed"));
        assert!(fixture.net.faulted);
        assert!(error(&mut fixture).ends_with("reinitialization required"));

        // PAUSE frames from the partner hold the transmitter off indefinitely;
        // time held while paused still counts once they stop, as in Linux.
        let mut fixture = full();
        fixture.registers[STATUS / 4] = STATUS_TXOFF;
        for _ in 0..attempts * 2 {
            assert!(
                error(&mut fixture).ends_with("transmit ring full while paused by flow control")
            );
        }
        assert!(!fixture.net.faulted);
        fixture.registers[STATUS / 4] = 0;
        assert!(error(&mut fixture).ends_with("transmit hang; controller reset failed"));
    }

    #[test]
    fn statistics_accumulate_until_the_function_is_lost() {
        let mut fixture = Fixture::new();
        fixture.registers[0x4074 / 4] = 3; // GPRC
        fixture.registers[0x40c0 / 4] = 0x100; // TORL
        assert_eq!(fixture.net.statistics().unwrap().good_received, 3);
        let statistics = fixture.net.statistics().unwrap();
        assert_eq!(statistics.good_received, 6);
        assert_eq!(statistics.total_octets_received, 0x200);
        fixture.registers[0x0008 / 4] = u32::MAX; // STATUS
        assert!(fixture.net.statistics().is_err());
        assert!(!fixture.net.carrier());
    }

    #[test]
    fn the_station_address_is_receive_address_zero_when_valid() {
        let mac = station(0x7856_3412, RAH_AV | 0xbc9a);
        assert_eq!(mac, Some(Mac([0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc])));
        assert_eq!(station(0x7856_3412, 0xbc9a), None);
        assert_eq!(station(0, RAH_AV), None);
        assert_eq!(station(0x7856_3412, u32::MAX), None);
    }

    #[test]
    fn a_frame_is_one_advanced_data_descriptor_with_its_crc_inserted() {
        let (command, status) = advanced::transmit_words(60);
        assert_eq!(command, 0x2b30_003c);
        assert_eq!(status, 60 << 14);
        assert_eq!(status & 1, 0); // DD clear until the controller writes back
    }

    #[test]
    fn written_back_frames_are_taken_or_dropped() {
        assert_eq!(received(0, 60), None);
        assert_eq!(received(RX_DD | RX_EOP, 60), Some(Some(60)));
        assert_eq!(received(RX_DD | RX_EOP | RX_RXE, 60), Some(None));
        assert_eq!(received(RX_DD, 60), Some(None));
        assert_eq!(received(RX_DD | RX_EOP, 4096), Some(None));
    }

    #[test]
    fn only_programmed_i225_and_i226_parts_match() {
        assert!(DEVICES.contains(&0x125c));
        assert!(!DEVICES.contains(&0x125f));
        assert!(!DEVICES.contains(&0x15fd));
    }

    #[test]
    fn source_exact_i226_link_erratum_runs_before_attachment_and_propagates_failure() {
        use core::cell::Cell;
        for device in DEVICES {
            let called = Cell::new(false);
            let result = prepare_link(device, || {
                called.set(true);
                Err(BlockError::Timeout)
            });
            let required = [0x125b, 0x125c, 0x125d, 0x3102].contains(&device);
            assert_eq!(called.get(), required);
            assert_eq!(
                result,
                if required {
                    Err(BlockError::Timeout)
                } else {
                    Ok(())
                }
            );
        }
    }

    #[test]
    fn packet_rings_wrap_and_fragment_tails_are_not_delivered_as_frames() {
        let mut fixture = Fixture::new();
        let (net, registers) = (&mut fixture.net, &mut fixture.registers);
        assert_ne!(registers[RCTL / 4] & RCTL_MPE, 0);
        // 256 sixteen-byte descriptors each, as IGC_DEFAULT_RXD/TXD.
        assert_eq!([registers[RDLEN / 4], registers[TDLEN / 4]], [4096; 2]);
        let mut reused = Vec::with_capacity(MAX_FRAME);
        let storage = reused.as_ptr();
        for sequence in 0..(DESCRIPTORS * 3) {
            let frame = vec![sequence as u8; 60 + sequence % 128];
            net.transmit(&frame).unwrap();
            let index = sequence % DESCRIPTORS;
            assert_eq!(&net.transmit.buffer(index)[..frame.len()], frame);
            assert_eq!(
                net.transmit.descriptors.read::<u64>(index * DESCRIPTOR),
                net.transmit.address(index)
            );
            assert_eq!(
                net.transmit.descriptors.read::<u32>(index * DESCRIPTOR + 8),
                advanced::transmit_words(frame.len()).0
            );
            // Modeled hardware writes completion into the same TD.
            net.transmit
                .descriptors
                .write(index * DESCRIPTOR + 12, 1u32); // DD
            net.receive.buffer(index)[..frame.len()].copy_from_slice(&frame);
            net.receive
                .descriptors
                .write(index * DESCRIPTOR + 8, RX_DD | RX_EOP);
            net.receive
                .descriptors
                .write(index * DESCRIPTOR + 12, frame.len() as u16);
            if sequence % 2 == 0 {
                assert_eq!(net.receive().unwrap(), Some(frame));
            } else {
                // The stack's path refills its own buffer in place.
                assert_eq!(net.receive_reusing(&mut reused), Ok(true));
                assert_eq!(reused, frame);
                assert_eq!(reused.as_ptr(), storage);
            }
            // Linux's IGC_RX_BUFFER_WRITE: one tail write per sixteen.
            if (sequence + 1) % 16 == 0 {
                assert_eq!(registers[RDT / 4], index as u32);
            }
        }
        for (index, status) in [(0, RX_DD), (1, RX_DD | RX_EOP), (2, RX_DD | RX_EOP)] {
            net.receive.buffer(index)[..60].fill(index as u8);
            net.receive
                .descriptors
                .write(index * DESCRIPTOR + 8, status);
            net.receive
                .descriptors
                .write(index * DESCRIPTOR + 12, 60u16);
        }
        assert_eq!(net.receive().unwrap(), Some(vec![2; 60]));
        assert_eq!(net.receive.next, 3);
        assert_eq!(net.negotiated(), Some(LINK));
        registers[ICR / 4] = u32::MAX;
        assert!(!net.carrier());
        registers[ICR / 4] = 0;
        assert!(!net.carrier());
        assert!(net.transmit(&[0; 60]).is_err());
        assert!(net.receive().is_err());
        reused.clone_from(&vec![7; 60]);
        assert!(net.receive_reusing(&mut reused).is_err());
        assert_eq!(reused, vec![7; 60]);
        assert!(station(u32::MAX, u32::MAX).is_none());
        assert_eq!(received(RX_DD | RX_EOP, 0), Some(None));
        assert_eq!(received(RX_DD | RX_EOP, (MAX_FRAME + 1) as u16), Some(None));
    }

    #[test]
    fn short_transmits_publish_the_psp_minimum_with_zeroed_padding_and_reject_bad_headers() {
        let mut fixture = Fixture::new();
        let net = &mut fixture.net;
        // Reuse every ring buffer after nonzero old packets have occupied it.
        for sequence in 0..(DESCRIPTORS * 2) {
            let index = sequence % DESCRIPTORS;
            net.transmit.buffer(index).fill(0xa5);
            let length = 14 + sequence % 4;
            let frame = vec![sequence as u8; length];
            net.transmit(&frame).unwrap();
            assert_eq!(&net.transmit.buffer(index)[..length], frame);
            assert!(
                net.transmit.buffer(index)[length..17]
                    .iter()
                    .all(|&b| b == 0)
            );
            assert_eq!(
                net.transmit.descriptors.read::<u32>(index * DESCRIPTOR + 8),
                advanced::transmit_words(17).0
            );
            assert_eq!(
                net.transmit
                    .descriptors
                    .read::<u32>(index * DESCRIPTOR + 12),
                advanced::transmit_words(17).1
            );
            net.transmit
                .descriptors
                .write(index * DESCRIPTOR + 12, 1u32); // DD
        }
        let tail = net.transmit.next;
        let before = net.transmit.buffer(tail).to_vec();
        let descriptors = net.transmit.descriptors.bytes().to_vec();
        for length in 0..14 {
            assert!(net.transmit(&vec![0; length]).is_err());
            assert_eq!(net.transmit.next, tail);
            assert_eq!(net.transmit.buffer(tail), before);
            assert_eq!(net.transmit.descriptors.bytes(), descriptors);
        }
    }

    /// Registers of the stop sequence, private to `hardware`.
    const CTRL: usize = 0;
    const EECD: usize = 0x10;
    const CTRL_EXT: usize = 0x18;
    const IMC: usize = 0x150c;
    const EIMC: usize = 0x1528;
    const MASTER_DISABLE: u32 = 1 << 2;
    const MAC_RESET: u32 = 1 << 26;
    const AUTO_READ_DONE: u32 = 1 << 9;
    const DRV_LOAD: u32 = 1 << 28;

    /// A minimum-delay model whose waits stand for the controller: a MAC
    /// reset completes, and reloads the NVM, during the next wait. The
    /// first time it completes one it records RCTL, TCTL and CTRL.
    struct Resetting {
        registers: *mut u32,
        before_reset: Rc<core::cell::Cell<Option<[u32; 3]>>>,
    }
    // SAFETY: isolated virtual-time model over the fixture's register
    // vector, which outlives every wait; no hardware wait is granted.
    unsafe impl crate::bus::pci::power::Timer for Resetting {
        fn check(&self) -> Result<(), crate::bus::pci::power::Error> {
            Ok(())
        }
        fn wait_us(&self, _: u64) -> Result<(), crate::bus::pci::power::Error> {
            let register = |offset: usize| self.registers.wrapping_add(offset / 4);
            // SAFETY: offsets inside the fixture's live register vector.
            unsafe {
                let control = register(CTRL).read_volatile();
                if control & MAC_RESET != 0 {
                    if self.before_reset.get().is_none() {
                        self.before_reset.set(Some([
                            register(RCTL).read_volatile(),
                            register(TCTL).read_volatile(),
                            control,
                        ]));
                    }
                    register(CTRL).write_volatile(control & !MAC_RESET);
                    let eecd = register(EECD).read_volatile();
                    register(EECD).write_volatile(eecd | AUTO_READ_DONE);
                }
            }
            Ok(())
        }
    }

    /// Dropping a running adapter is Linux igc_down and
    /// igc_release_hw_control: receive and transmit off and the bus master
    /// drained before the MAC reset, every interrupt masked, and the
    /// controller handed back to firmware only after the reset completed.
    #[test]
    fn dropping_a_running_adapter_stops_it_before_releasing_control() {
        let mut fixture = Fixture::new();
        let before_reset = Rc::new(core::cell::Cell::new(None));
        fixture.net.clock = crate::timing::Timing::checked(
            Clock {
                ticks_per_microsecond: 1,
            },
            Rc::new(Resetting {
                registers: fixture.registers.as_mut_ptr(),
                before_reset: before_reset.clone(),
            }),
        )
        .unwrap();
        fixture.registers[CTRL_EXT / 4] = DRV_LOAD | 1;
        assert_ne!(fixture.registers[RCTL / 4] & RCTL_EN, 0);
        assert_ne!(fixture.registers[TCTL / 4] & TCTL_EN, 0);
        let Fixture {
            net,
            registers,
            _backing,
        } = fixture;
        drop(net);
        let [rctl, tctl, control] = before_reset.get().expect("a MAC reset was requested");
        assert_eq!(rctl & RCTL_EN, 0);
        assert_eq!(tctl & TCTL_EN, 0);
        assert_ne!(control & MASTER_DISABLE, 0);
        assert_eq!(registers[IMC / 4], u32::MAX);
        assert_eq!(registers[EIMC / 4], u32::MAX);
        assert_eq!(registers[CTRL_EXT / 4], 1);
    }

    /// A reset that never completes is no stop: the controller may still
    /// write its rings, so it keeps them and firmware is not handed control.
    #[test]
    fn dropping_an_adapter_whose_reset_never_completes_keeps_control() {
        let mut fixture = Fixture::new();
        fixture.registers[CTRL_EXT / 4] = DRV_LOAD;
        let Fixture {
            net,
            registers,
            _backing,
        } = fixture;
        drop(net);
        assert_ne!(registers[CTRL / 4] & MAC_RESET, 0);
        assert_eq!(registers[CTRL_EXT / 4], DRV_LOAD);
    }

    /// A function that is gone reads all ones: dropping it writes nothing.
    #[test]
    fn dropping_a_lost_adapter_writes_no_register() {
        let mut fixture = Fixture::new();
        fixture.registers.fill(u32::MAX);
        let Fixture {
            net,
            registers,
            _backing,
        } = fixture;
        drop(net);
        assert!(registers.iter().all(|&value| value == u32::MAX));
    }

    #[test]
    fn minimum_failure_during_reinitialization_is_typed_and_blocks_later_device_effects() {
        let mut fixture = Fixture::new();
        let (timing, failed) = crate::timing::controlled_model(Clock {
            ticks_per_microsecond: 1,
        });
        fixture.net.clock = timing;
        failed.set(true);
        let expected = BlockError::PciPower(crate::bus::pci::power::Error::Clock);
        assert_eq!(fixture.net.reinitialize(), Err(expected.clone()));
        let observed = fixture.registers.clone();
        for _ in 0..3 {
            assert!(!fixture.net.carrier());
            assert_eq!(
                fixture.net.transmit(&[1; 60]),
                Err(LinkError::Pci(expected.clone()))
            );
            assert_eq!(fixture.net.receive(), Err(LinkError::Pci(expected.clone())));
        }
        assert_eq!(fixture.registers, observed);
    }
}
