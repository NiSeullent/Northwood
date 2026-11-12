// SPDX-License-Identifier: GPL-2.0-only
// Copyright(c) 2007 - 2018 Intel Corporation.
// Copyright (c) 2026 Stela contributors
//! Intel igb copper Ethernet: 82575/82576/82580, I350 and I210/I211.
//!
//! Native Stela adaptation of Linux's `drivers/net/ethernet/intel/igb` at
//! `a804c5742bff224d4cb4d25dd383554d598be32b`. Reset, EEPROM/iNVM,
//! firmware synchronization, PHY and PLL algorithms keep the GPL terms;
//! source hashes and supported subsets are in NOTICE.
//!
//! One polled advanced-descriptor queue, ordinary Ethernet frames, no
//! offloads, SR-IOV, timestamping, EEE, fiber, SerDes or SGMII. Matching a
//! copper PCI ID still requires a supported PHY and valid programmed NVM.
//! DMA remains allocated until reboot, including after errors or drop.
use super::{
    HEADER, Link, LinkError, MAX_FRAME, MIN_FRAME, Mac,
    advanced::{
        self, QUEUE_ENABLE, RX_DD, RX_EOP, RXDCTL_THRESHOLDS, SRRCTL_ONE_BUFFER, TXDCTL_THRESHOLDS,
        TransmitQueue, transmitted,
    },
    check_frame,
    ring::{DESCRIPTOR, DESCRIPTORS, Ring},
};
use crate::{
    Handoff,
    block::BlockError,
    bus::{
        dma::DmaAllocator,
        mmio::Mmio,
        pci::{ConfigSpace, Function},
    },
    registers as mapped_registers,
};
use alloc::{format, string::String, vec::Vec};
use core::sync::atomic::{Ordering, fence};

mod hardware;
mod nvm;
mod phy;
mod registers;
use hardware::{Hardware, PciCapabilities};
use registers::*;

/// Receive descriptor errors that drop a frame: RXE, IPE, TCPE and CE;
/// checksum errors are irrelevant with checksum offload disabled.
const RX_FRAME_ERRORS: u32 = 0x9700_0000;
const TRANSMIT_QUEUE: TransmitQueue = TransmitQueue {
    head: TDH,
    tail: TDT,
    control: TXDCTL,
};

/// Interrupt registers igb_configure_msix and igb_irq_enable program.
const IMS: usize = 0x00d0;
const EICS: usize = 0x1520;
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
/// igb.h IGB_START_ITR, with E1000_EITR_CNT_IGNR as igb_write_itr sets it.
const START_ITR: u32 = 648 | 1 << 31;
/// The one MSI-X vector: queue 0's receive and the link causes.
const VECTOR: u32 = 0;
/// An MSI-X table entry's bytes.
const MSIX_ENTRY_BYTES: u64 = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Chip {
    I82575,
    I82576,
    I82580,
    I350,
    I210,
    I211,
}

impl Chip {
    fn device(id: u16) -> Option<Self> {
        match id {
            0x10a7 | 0x10d6 => Some(Self::I82575),
            0x10c9 | 0x10e8 | 0x1526 | 0x150a => Some(Self::I82576),
            0x150e | 0x1516 => Some(Self::I82580),
            0x1521 => Some(Self::I350),
            0x1533 | 0x157b => Some(Self::I210),
            0x1539 => Some(Self::I211),
            _ => None,
        }
    }
    fn modern(self) -> bool {
        matches!(self, Self::I210 | Self::I211)
    }
    fn old_reset(self) -> bool {
        matches!(self, Self::I82575 | Self::I82576)
    }
    fn queues(self) -> usize {
        match self {
            Self::I82575 | Self::I210 => 4,
            Self::I82576 => 16,
            Self::I82580 | Self::I350 => 8,
            Self::I211 => 2,
        }
    }
    fn receive_addresses(self) -> usize {
        match self {
            Self::I82576 | Self::I82580 => 24,
            Self::I350 => 32,
            _ => 16,
        }
    }
    fn name(self) -> &'static str {
        match self {
            Self::I82575 => "82575",
            Self::I82576 => "82576",
            Self::I82580 => "82580",
            Self::I350 => "I350",
            Self::I210 => "I210",
            Self::I211 => "I211",
        }
    }
}

pub fn matches<C: ConfigSpace>(function: &Function<C>) -> bool {
    matches_identity(&function.identity())
}

/// Candidate selection from copied identity DATA; owned attachment still
/// revalidates configuration and platform requirements.
pub fn matches_identity(function: &crate::bus::pci::Identity) -> bool {
    function.vendor == 0x8086
        && function.class == [2, 0, 0]
        && Chip::device(function.device).is_some()
}

/// Linux continues after these firmware completion indications time out,
/// then checks NVM and PHY transactions. Keep those diagnostics visible.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InitializationWarnings {
    pub nvm_autoload_timeout: bool,
    pub phy_configuration_timeout: bool,
}

pub(crate) trait RegisterIo {
    fn read(&self, offset: usize) -> u32;
    fn write(&self, offset: usize, value: u32);
}
impl RegisterIo for Mmio {
    fn read(&self, offset: usize) -> u32 {
        self.read32(offset)
    }
    fn write(&self, offset: usize, value: u32) {
        self.write32(offset, value);
    }
}
pub(crate) trait Timer {
    fn delay(&self, microseconds: u64) -> Result<(), BlockError>;
    fn wait(&self, microseconds: u64, ready: impl FnMut() -> bool) -> bool;
}
impl Timer for crate::timing::Timing {
    fn delay(&self, us: u64) -> Result<(), BlockError> {
        crate::timing::Timing::delay(self, us)
    }
    fn wait(&self, us: u64, ready: impl FnMut() -> bool) -> bool {
        crate::timing::Timing::wait(self, us, ready)
    }
}

pub struct Igb {
    device: Mmio,
    receive: Ring,
    transmit: Ring,
    dropping: bool,
    /// Register reads return all ones: the function is gone.
    faulted: bool,
    mac: Mac,
    clock: crate::timing::Timing,
    pub location: String,
    pub warnings: InitializationWarnings,
    /// Whether MSI-X vector 0 sends the handoff's completion signal.
    signalled: bool,
}

impl Igb {
    /// # Safety
    /// The caller exclusively owns this physical PCI function after
    /// ExitBootServices and supplies its mapped MMIO/DMA handoff.
    /// No firmware or other driver may drive the function afterwards.
    pub unsafe fn new(
        function: &mut Function,
        handoff: &Handoff,
        dma: &mut dyn DmaAllocator,
        minimum: alloc::rc::Rc<dyn crate::bus::pci::power::Timer>,
    ) -> Result<Self, BlockError> {
        let clock = crate::timing::Timing::checked(handoff.clock, minimum)?;
        if !matches(function) {
            return Err(BlockError::Unsupported("not an igb copper controller"));
        }
        let chip = Chip::device(function.device).unwrap();
        let bytes = if chip.modern() { 0x2_0000 } else { 0x1_0000 };
        function.power_up(clock.minimum())?;
        let bars = function.bars()?;
        let device = mapped_registers(handoff, bars[0], 0, bytes)?;
        let capabilities = PciCapabilities::read(function)?;
        function.enable_polled()?;
        // The 82575 maps queues to vectors through MSIXBM, which this
        // driver leaves alone: it stays polled.
        let msix = function
            .interrupts()?
            .msix
            .filter(|_| !matches!(chip, Chip::I82575));
        let signalled = match (handoff.completion, msix) {
            (Some(message), Some(msix)) => {
                let table = mapped_registers(
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
        let mut hardware = Hardware::new(&device, &clock, chip);
        let (mac, warnings) = hardware.initialize(function.config(), capabilities)?;
        let mut net = Self {
            device,
            receive: Ring::new(dma)?,
            transmit: Ring::new(dma)?,
            dropping: false,
            faulted: false,
            mac,
            clock,
            location: format!("{} Intel {}", function.address, chip.name()),
            warnings,
            signalled,
        };
        start_queues(&net.device, &net.clock, &mut net.receive, &net.transmit)?;
        if signalled {
            arm_interrupts(&net.device);
        }
        Ok(net)
    }

    /// Report a lost function, and discard the transmit work a lost link
    /// strands (Linux `igb_watchdog_task` resets for it). Returns the carrier.
    fn watchdog(&mut self) -> Result<bool, LinkError> {
        let status = self.device.read32(STATUS);
        self.faulted |= status == u32::MAX;
        if !self.faulted && status & STATUS_LU == 0 && self.transmit.pending() {
            let device = self.device;
            self.faulted = !TRANSMIT_QUEUE.flush(
                &mut self.transmit,
                self.clock.nominal(),
                |offset| device.read32(offset),
                |offset, value| device.write32(offset, value),
            );
        }
        match self.faulted {
            true => Err(LinkError::Adapter(format!(
                "{} lost register access; reinitialization required",
                self.location
            ))),
            false => Ok(status & STATUS_LU != 0),
        }
    }
}

/// igb_configure_msix and igb_irq_enable for one vector: queue 0's receive
/// and the link causes send it, auto-masked until the next empty receive
/// re-arms it. Queue 0 takes IVAR0's low byte on every chip but the 82575.
fn arm_interrupts(device: &Mmio) {
    device.write32(GPIE, GPIE_MSIX_MODE | GPIE_PBA | GPIE_EIAME | GPIE_NSICR);
    device.write32(IVAR_MISC, (VECTOR | IVAR_VALID) << 8);
    device.write32(IVAR0, VECTOR | IVAR_VALID);
    device.write32(EITR0, START_ITR);
    device.write32(EIAC, device.read32(EIAC) | 1 << VECTOR);
    device.write32(EIAM, device.read32(EIAM) | 1 << VECTOR);
    device.write32(EIMS, 1 << VECTOR);
    device.write32(IMS, IMS_LSC);
}

fn start_queues(
    io: &impl RegisterIo,
    clock: &impl Timer,
    receive: &mut Ring,
    transmit: &Ring,
) -> Result<(), BlockError> {
    let write = |offset, value| io.write(offset, value);
    receive.program(write, RDBAL, RDBAH, RDLEN);
    transmit.program(write, TDBAL, TDBAH, TDLEN);
    io.write(RDH, 0);
    io.write(RDT, 0);
    io.write(TDH, 0);
    io.write(TDT, 0);
    io.write(SRRCTL, SRRCTL_ONE_BUFFER);
    // One-at-a-time write-back keeps polling live even for isolated packets.
    io.write(RXDCTL, RXDCTL_THRESHOLDS | QUEUE_ENABLE);
    io.write(TXDCTL, TXDCTL_THRESHOLDS | QUEUE_ENABLE);
    let enabled = clock.wait(20_000, || {
        io.read(RXDCTL) & QUEUE_ENABLE != 0 && io.read(TXDCTL) & QUEUE_ENABLE != 0
    });
    if !enabled {
        io.write(RXDCTL, 0);
        io.write(TXDCTL, 0);
        io.write(RCTL, 0);
        io.write(TCTL, TCTL_PSP);
        return Err(BlockError::Timeout);
    }
    for index in 0..DESCRIPTORS {
        advanced::offer_receive(receive, index);
    }
    fence(Ordering::SeqCst);
    io.write(RDT, (DESCRIPTORS - 1) as u32);
    io.write(RCTL, RCTL_EN | RCTL_MPE | RCTL_BAM | RCTL_SECRC);
    io.write(TCTL, TCTL_EN | TCTL_PSP | TCTL_RTLC | TCTL_COLLISIONS);
    io.read(STATUS);
    Ok(())
}

impl Link for Igb {
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
            return Err(LinkError::Adapter(format!(
                "{} transmit ring stalled",
                self.location
            )));
        }
        advanced::queue_frame(&mut self.transmit, frame, frame.len().max(MIN_FRAME));
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

impl Igb {
    /// The next good whole frame the controller wrote back, passed to
    /// `take` while its buffer is still held.
    fn receive_frame<T>(
        &mut self,
        mut take: impl FnMut(&[u8]) -> T,
    ) -> Result<Option<T>, LinkError> {
        self.watchdog()?;
        for _ in 0..DESCRIPTORS {
            let index = self.receive.next;
            let status: u32 = self.receive.descriptors.read(index * DESCRIPTOR + 8);
            if status & RX_DD == 0 {
                if let Some(tail) = self.receive.idle() {
                    fence(Ordering::SeqCst);
                    self.device.write32(RDT, tail);
                }
                if self.signalled {
                    // igb_msix_other reads the link cause, clearing it so
                    // the next change sends the vector; igb_ring_irq_enable
                    // unmasks the vector, auto-masked when it fired.
                    let link = self.device.read32(ICR) & IMS_LSC != 0;
                    self.device.write32(EIMS, 1 << VECTOR);
                    if link {
                        self.watchdog()?;
                    }
                }
                return Ok(None);
            }
            fence(Ordering::Acquire);
            let length = usize::from(
                self.receive
                    .descriptors
                    .read::<u16>(index * DESCRIPTOR + 12),
            );
            let eop = status & RX_EOP != 0;
            let good = !self.dropping
                && eop
                && status & RX_FRAME_ERRORS == 0
                && (HEADER..=MAX_FRAME).contains(&length);
            let frame = good.then(|| take(&self.receive.buffer(index)[..length]));
            self.dropping = !eop;
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
        if let Some(tail) = self.receive.idle() {
            fence(Ordering::SeqCst);
            self.device.write32(RDT, tail);
        }
        if self.signalled {
            // Frames stopped at the bound wait with the vector masked; a
            // software cause sends it again, as igb_watchdog_task does.
            self.device.write32(EICS, 1 << VECTOR);
            self.device.write32(EIMS, 1 << VECTOR);
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
use crate::Clock;
