// SPDX-License-Identifier: GPL-2.0-or-later
// Sequences follow Linux `drivers/net/ethernet/atheros/atl1c`, Copyright (c)
// 2008 Atheros Communications Inc., Copyright (c) 2007 Atheros Corporation.
// Copyright (c) 2026 Stela contributors
//! Atheros AR8131/AR8132 (L1C/L2C), AR8151 (L1D v1.0 and v2.0) and AR8152
//! (L2C_B v1.1 and L2C_B2 v2.0) Ethernet, following Linux's `atl1c` driver.
//! The newer AR8161/AR8162/AR8171/AR8172 and Killer parts are [`crate::alx`].
//!
//! One polled transmit queue and one receive queue, without checksum, TSO
//! or VLAN offloads and without interrupts: the interrupt mask stays zero
//! and the PHY link is sampled at most every 10 ms. Probe follows
//! `atl1c_probe` (PCIe reset with ASPM off, PHY reset, MAC reset, PHY
//! autonegotiation, permanent address) and open follows `atl1c_up`.
//! A link going down runs `atl1c_check_link_status`'s MAC reset and
//! reconfiguration; a link coming up sets ASPM and starts the MAC.
//! All multicast is accepted, as `IFF_ALLMULTI` selects.
//!
//! Differences from Linux: the PCI owner keeps the command register (it
//! enables memory decoding and bus mastering and disables INTx for this
//! polled driver) and the AER severity mask; the AR8151 v2.0 multi-queue
//! variant that answers `MT_MAGIC` is refused; an adapter without a
//! valid station address is refused instead of given a random one; a
//! received frame spread over several free descriptors is dropped (the
//! 1522-byte buffers hold every standard frame). Jumbo frames,
//! Wake-on-LAN and suspend are outside this port. DMA buffers live until
//! reboot.
//!
//! Tested only against the register and DMA model in `atl1c/tests.rs`; no
//! atl1c hardware was available.
use super::{Link, LinkError, MAX_FRAME, MIN_FRAME, Mac, check_frame};
use crate::{
    Handoff,
    alx::Registers,
    block::BlockError,
    bus::{
        dma::{DmaAllocator, DmaBuffer},
        mmio::Mmio,
        pci::{ConfigSpace, Function},
    },
    ring::{Ring, write_transmit_frame},
    timing::Timing,
};
use alloc::{format, string::String, vec::Vec};
use core::sync::atomic::{Ordering, fence};

mod hardware;
mod registers;
#[cfg(test)]
mod tests;

use hardware::{Chip, Hardware, Kind, LinkState, RX_RING, Speed, TX_RING};
use registers::*;

/// `PCI_VENDOR_ID_ATTANSIC`.
const VENDOR: u16 = 0x1969;
/// `atl1c_pci_tbl`: L2C, L1C, L2C_B, L2C_B2, L1D and L1D_2_0.
pub const DEVICES: [u16; 6] = [0x1062, 0x1063, 0x2060, 0x2062, 0x1073, 0x1083];
/// The register window: every register this port names lies below it.
const REGISTER_BYTES: u64 = 0x2000;
const LINK_POLL_US: u64 = 10_000;
const TRANSMIT_WAIT_US: u64 = 100_000;
const DESCRIPTOR: usize = 16;
const RFD: usize = 8;
const FCS: usize = 4;

/// `atl1c_patch_assign`'s platform table: device, revision, subsystem
/// vendor and subsystem device that take the link threshold patch.
const LINK_PATCH: [(u16, u8, u16, u16); 19] = [
    (0x2060, 0xc1, 0x1019, 0x8152),
    (0x2060, 0xc1, 0x1019, 0x2060),
    (0x2060, 0xc1, 0x1019, 0xe000),
    (0x2062, 0xc0, 0x1019, 0x8152),
    (0x2062, 0xc0, 0x1019, 0x2062),
    (0x2062, 0xc0, 0x1458, 0xe000),
    (0x2062, 0xc1, 0x1019, 0x8152),
    (0x2062, 0xc1, 0x1019, 0x2062),
    (0x2062, 0xc1, 0x1458, 0xe000),
    (0x2062, 0xc1, 0x1565, 0x2802),
    (0x2062, 0xc1, 0x1565, 0x2801),
    (0x1073, 0xc0, 0x1019, 0x8151),
    (0x1073, 0xc0, 0x1019, 0x1073),
    (0x1073, 0xc0, 0x1458, 0xe000),
    (0x1083, 0xc0, 0x1458, 0xe000),
    (0x1083, 0xc0, 0x1019, 0x8151),
    (0x1083, 0xc0, 0x1019, 0x1083),
    (0x1083, 0xc0, 0x1462, 0x7680),
    (0x1083, 0xc0, 0x1565, 0x2803),
];

pub fn matches<C: ConfigSpace>(function: &Function<C>) -> bool {
    matches_identity(&function.identity())
}

/// Candidate selection from copied identity data.
pub fn matches_identity(identity: &crate::bus::pci::Identity) -> bool {
    identity.vendor == VENDOR && DEVICES.contains(&identity.device) && identity.class == [2, 0, 0]
}

fn link_patch(device: u16, revision: u8, subsystem: u32) -> bool {
    let (vendor, id) = (subsystem as u16, (subsystem >> 16) as u16);
    LINK_PATCH.contains(&(device, revision, vendor, id))
}

/// `atl1c_configure_tx`'s read request correction: the DMA read block is
/// `pcie_get_readrq() >> 8` at most 1024 bytes' code, and a request under
/// 512 bytes is raised to 512. Answers the block and the new request code.
fn dma_read_block(control: u16) -> (u32, Option<u16>) {
    let bytes = 128u32 << ((control >> 12) & 7);
    let block = (bytes >> 8).min(3);
    if block < DEVICE_CTRL_MAXRRS_MIN {
        (DEVICE_CTRL_MAXRRS_MIN, Some(DEVICE_CTRL_MAXRRS_MIN as u16))
    } else {
        (block, None)
    }
}

/// The transmit rings (the used one and the always-programmed idle
/// high-priority one), the receive free and return rings, and the
/// driver's indices into them.
struct Queues {
    transmit: Ring,
    idle: DmaBuffer,
    /// Receive free descriptors: eight bytes each in the ring's descriptor
    /// storage, one 2 KiB buffer each.
    free: Ring,
    returned: DmaBuffer,
    tx_write: usize,
    tx_read: usize,
    rx_write: usize,
    rx_read: usize,
    rrd_read: usize,
}

impl Queues {
    fn new(dma: &mut dyn DmaAllocator) -> Result<Self, BlockError> {
        let transmit = Ring::with(dma, TX_RING)?;
        let idle = dma.allocate(TX_RING * DESCRIPTOR, 4096)?;
        let free = Ring::with(dma, RX_RING)?;
        let returned = dma.allocate(RX_RING * DESCRIPTOR, 4096)?;
        if idle.len() != TX_RING * DESCRIPTOR || returned.len() != RX_RING * DESCRIPTOR {
            return Err(BlockError::Unsupported("atl1c ring allocation"));
        }
        // One high-address register serves each direction's rings.
        if idle.bus_address() >> 32 != transmit.base() >> 32
            || returned.bus_address() >> 32 != free.base() >> 32
        {
            return Err(BlockError::Unsupported(
                "atl1c rings across a 4 GiB boundary",
            ));
        }
        Ok(Self {
            transmit,
            idle,
            free,
            returned,
            tx_write: 0,
            tx_read: 0,
            rx_write: 0,
            rx_read: 0,
            rrd_read: 0,
        })
    }

    /// `atl1c_reset_dma_ring` and `atl1c_init_ring_ptrs`.
    fn reset(&mut self) {
        self.transmit.descriptors.bytes_mut().fill(0);
        self.idle.bytes_mut().fill(0);
        self.free.descriptors.bytes_mut().fill(0);
        self.returned.bytes_mut().fill(0);
        self.tx_write = 0;
        self.tx_read = 0;
        self.rx_write = 0;
        self.rx_read = 0;
        self.rrd_read = 0;
    }

    /// `atl1c_alloc_rx_buffer`: offer every free slot up to the one before
    /// the next to clean, and answer the producer index to publish.
    fn refill(&mut self) -> Option<usize> {
        let mut current = self.rx_write;
        let mut next = (current + 1) % RX_RING;
        let mut count = 0;
        while next != self.rx_read {
            let address = self.free.address(current);
            self.free.descriptors.write(current * RFD, address);
            current = next;
            next = (next + 1) % RX_RING;
            count += 1;
        }
        if count == 0 {
            return None;
        }
        fence(Ordering::SeqCst);
        self.rx_write = current;
        Some(current)
    }

    /// `atl1c_tpd_avail`.
    fn transmit_available(&self) -> usize {
        if self.tx_write >= self.tx_read {
            TX_RING + self.tx_read - self.tx_write - 1
        } else {
            self.tx_read - self.tx_write - 1
        }
    }
}

pub struct Atl1c<R: Registers = Mmio> {
    hardware: Hardware<R>,
    queues: Queues,
    mac: Mac,
    /// When the PHY link was last sampled; `None` forces a sample.
    sampled: Option<u64>,
    /// Whether the link state was ever acted on; the first sample without
    /// a link runs the link-down path once, as `atl1c_up` does.
    known: bool,
    failure: Option<BlockError>,
    stopped: bool,
    pub phy: u32,
    pub location: String,
}

impl Atl1c {
    /// Initialize one exclusively owned function after ExitBootServices.
    ///
    /// # Safety
    /// The caller exclusively owns the controller and its PCI configuration,
    /// and the handoff mapped its BAR0 uncached. `dma` is identity-mapped
    /// DMA memory.
    pub unsafe fn new(
        function: &mut Function,
        handoff: &Handoff,
        dma: &mut dyn DmaAllocator,
        minimum: alloc::rc::Rc<dyn crate::bus::pci::power::Timer>,
    ) -> Result<Self, BlockError> {
        let clock = Timing::checked(handoff.clock, minimum)?;
        if !matches(function) {
            return Err(BlockError::Unsupported("not an atl1c controller"));
        }
        let kind = Kind::from_device(function.device)
            .ok_or(BlockError::Unsupported("not an atl1c controller"))?;
        function.power_up(clock.minimum())?;
        let pcie = function
            .capabilities()?
            .into_iter()
            .find(|&(_, id)| id == 0x10)
            .map(|(offset, _)| u16::from(offset))
            .ok_or(BlockError::Unsupported("atl1c without a PCIe capability"))?;
        let revision = function.config().read8(0x08)?;
        let subsystem = function.config().read32(0x2c)?;
        let bars = function.bars()?;
        let device = crate::registers(handoff, bars[0], 0, REGISTER_BYTES)?;
        if kind == Kind::L1d2 && device.read32(MT_MAGIC_REG) == MT_MAGIC {
            return Err(BlockError::Unsupported(
                "atl1c multi-queue MT variant not implemented",
            ));
        }
        function.enable_polled()?;
        // atl1c_reset_pcie: clear the device status error bits.
        function.config().write16(pcie + 0x0a, 0x000f)?;
        if function.device == 0x2062 && revision == L2CB_V21 {
            // atl1c_patch_assign: the indirect PCIe miscellaneous control.
            function
                .config()
                .write32(PCIE_IND_ACC_ADDR, PCIE_DEV_MISC_CTRL)?;
            let misc = function.config().read32(PCIE_IND_ACC_DATA)? & !0x100;
            function
                .config()
                .write32(PCIE_IND_ACC_ADDR, PCIE_DEV_MISC_CTRL)?;
            function.config().write32(PCIE_IND_ACC_DATA, misc)?;
        }
        let control = function.config().read16(pcie + 8)?;
        let (dmar_block, raised) = dma_read_block(control);
        if let Some(code) = raised {
            function
                .config()
                .write16(pcie + 8, (control & !0x7000) | (code << 12))?;
        }
        let chip = Chip {
            kind,
            device: function.device,
            revision,
            link_patch: link_patch(function.device, revision, subsystem),
        };
        let location = format!("{} Atheros {:04x}", function.address, function.device);
        Self::bring_up(device, clock, chip, dma, dmar_block, location)
    }
}

impl<R: Registers> Atl1c<R> {
    /// `atl1c_probe` from the PCIe reset on, then `atl1c_open`.
    fn bring_up(
        registers: R,
        clock: Timing,
        chip: Chip,
        dma: &mut dyn DmaAllocator,
        dmar_block: u32,
        location: String,
    ) -> Result<Self, BlockError> {
        let hardware = Hardware::new(registers, clock, chip, dmar_block);
        hardware.reset_pcie()?;
        hardware.reset_phy()?;
        hardware.reset_mac()?;
        let phy = hardware.init_phy()?;
        let mac = hardware.permanent_mac()?;
        hardware.set_mac(mac);
        let queues = Queues::new(dma)?;
        let mut adapter = Self {
            hardware,
            queues,
            mac,
            sampled: None,
            known: false,
            failure: None,
            stopped: false,
            phy,
            location,
        };
        adapter.configure();
        adapter.check_link()?;
        Ok(adapter)
    }

    /// `atl1c_configure`: ring indices, receive filter, receive buffers,
    /// then `atl1c_configure_mac`.
    fn configure(&mut self) {
        self.queues.reset();
        self.hardware.receive_filter();
        if let Some(index) = self.queues.refill() {
            self.hardware.produce_receive(index);
        }
        let queues = &self.queues;
        self.hardware.configure_mac(|hardware| {
            hardware.ring_pointers(
                queues.transmit.base(),
                queues.idle.bus_address(),
                queues.free.base(),
                queues.returned.bus_address(),
            )
        });
    }

    /// `atl1c_check_link_status`, run when the sampled state changes.
    fn check_link(&mut self) -> Result<bool, BlockError> {
        self.sampled = Some(self.hardware.timer.now_us());
        let new = self.hardware.read_phy_link()?;
        let old = self.hardware.link;
        let known = core::mem::replace(&mut self.known, true);
        if known && old == new {
            return Ok(new.speed.is_some());
        }
        self.hardware.link = new;
        match new.speed {
            None => {
                self.hardware.hibernate = true;
                self.hardware.reset_mac()?;
                self.hardware.set_aspm(None, true, true);
                self.hardware.post_phy_link(None)?;
                self.configure();
            }
            Some(speed) => {
                self.hardware.hibernate = false;
                self.hardware.set_aspm(Some(speed), true, true);
                self.hardware.post_phy_link(Some(speed))?;
                self.hardware.start_mac();
            }
        }
        Ok(new.speed.is_some())
    }

    /// `atl1c_down` then `atl1c_up`, for a return descriptor naming a
    /// free descriptor that does not exist.
    fn reinit(&mut self) -> Result<(), BlockError> {
        self.halt()?;
        self.configure();
        self.known = false;
        self.sampled = None;
        Ok(())
    }

    /// `atl1c_down`'s hardware part: interrupts masked, ASPM off, MAC reset.
    fn halt(&mut self) -> Result<(), BlockError> {
        self.hardware.mask_interrupts();
        self.hardware.set_aspm(None, false, false);
        self.hardware.reset_mac()?;
        self.hardware.timer.delay(1000)?;
        self.hardware.link = LinkState::default();
        self.queues.reset();
        Ok(())
    }

    fn checked(&mut self) -> Result<(), LinkError> {
        if self.stopped {
            return Err(LinkError::Adapter(format!("{} is stopped", self.location)));
        }
        match &self.failure {
            Some(error) => Err(LinkError::Adapter(format!("{}: {error}", self.location))),
            None => Ok(()),
        }
    }

    fn fail(&mut self, error: BlockError) -> LinkError {
        let message = LinkError::Adapter(format!("{}: {error}", self.location));
        self.failure = Some(error);
        message
    }

    fn carrier_now(&mut self) -> Result<bool, LinkError> {
        let fresh = self
            .sampled
            .is_some_and(|last| self.hardware.timer.now_us().wrapping_sub(last) < LINK_POLL_US);
        if fresh {
            return Ok(self.hardware.link.speed.is_some());
        }
        self.check_link().map_err(|error| self.fail(error))
    }

    /// `atl1c_clean_tx`: retire what the controller consumed; a consumer
    /// index past the ring is ignored, as Linux does.
    fn reclaim(&mut self) {
        let consumed = self.hardware.transmit_consumer();
        if consumed < TX_RING {
            self.queues.tx_read = consumed;
        }
    }

    /// Stop both queues and the MAC, as `atl1c_down` does.
    pub fn stop(&mut self) -> Result<(), BlockError> {
        self.stopped = true;
        self.halt()
    }

    /// The resolved link speed in Mb/s, `None` while down.
    pub fn speed(&self) -> Option<u32> {
        self.hardware.link.speed.map(|speed| match speed {
            Speed::Ten => 10,
            Speed::Hundred => 100,
            Speed::Thousand => 1000,
        })
    }
}

impl<R: Registers> Link for Atl1c<R> {
    fn mac(&self) -> Mac {
        self.mac
    }

    fn carrier(&mut self) -> bool {
        self.checked().is_ok() && self.carrier_now().unwrap_or(false)
    }

    fn transmit(&mut self, frame: &[u8]) -> Result<(), LinkError> {
        check_frame(frame)?;
        self.checked()?;
        if !self.carrier_now()? {
            return Err(LinkError::NoCarrier);
        }
        self.reclaim();
        if self.queues.transmit_available() == 0 {
            let clock = self.hardware.timer.nominal();
            let waited = clock.wait(TRANSMIT_WAIT_US, || {
                self.reclaim();
                self.queues.transmit_available() != 0
            });
            if !waited {
                return Err(LinkError::Adapter(format!(
                    "{} transmit ring stalled",
                    self.location
                )));
            }
        }
        let index = self.queues.tx_write;
        let length = frame.len().max(MIN_FRAME);
        write_transmit_frame(self.queues.transmit.buffer(index), frame, length);
        let address = self.queues.transmit.address(index);
        let descriptors = &mut self.queues.transmit.descriptors;
        let base = index * DESCRIPTOR;
        descriptors.write(base, length as u16);
        descriptors.write(base + 2, 0u16);
        descriptors.write(base + 4, TPD_EOP);
        descriptors.write(base + 8, address);
        self.queues.tx_write = (index + 1) % TX_RING;
        fence(Ordering::SeqCst);
        self.hardware.produce_transmit(self.queues.tx_write);
        Ok(())
    }

    fn receive(&mut self) -> Result<Option<Vec<u8>>, LinkError> {
        self.receive_frame(|bytes| {
            let mut data = Vec::new();
            data.try_reserve_exact(bytes.len()).ok()?;
            data.extend_from_slice(bytes);
            Some(data)
        })
    }

    /// The stack's receive path: the frame is copied into its reused
    /// buffer, so a received frame allocates nothing once it has grown.
    fn receive_reusing(&mut self, frame: &mut Vec<u8>) -> Result<bool, LinkError> {
        self.receive_frame(|bytes| {
            frame
                .try_reserve(bytes.len().saturating_sub(frame.len()))
                .ok()?;
            frame.clear();
            frame.extend_from_slice(bytes);
            Some(())
        })
        .map(|received| received.is_some())
    }
}

impl<R: Registers> Atl1c<R> {
    /// `atl1c_clean_rx`: the next good frame, passed to `take` while its
    /// buffer is still held; `take` answers `None` when it cannot store it.
    fn receive_frame<T>(
        &mut self,
        mut take: impl FnMut(&[u8]) -> Option<T>,
    ) -> Result<Option<T>, LinkError> {
        self.checked()?;
        if !self.carrier_now()? {
            return Ok(None);
        }
        let mut cleaned = false;
        let mut frame = None;
        while frame.is_none() {
            let base = self.queues.rrd_read * DESCRIPTOR;
            let status = self.queues.returned.read::<u32>(base + 12);
            if status & RRS_RXD_UPDATED == 0 {
                break;
            }
            fence(Ordering::Acquire);
            let word0 = self.queues.returned.read::<u32>(base);
            let count = field(word0, RRS_RX_RFD_CNT) as usize;
            let index = field(word0, RRS_RX_RFD_INDEX) as usize;
            if count == 0 || count > RX_RING || index >= RX_RING {
                self.reinit().map_err(|error| self.fail(error))?;
                return Ok(None);
            }
            // atl1c_clean_rrd: one return descriptor per free descriptor.
            for _ in 0..count {
                let slot = self.queues.rrd_read * DESCRIPTOR + 12;
                let word3 = self.queues.returned.read::<u32>(slot);
                self.queues.returned.write(slot, word3 & !RRS_RXD_UPDATED);
                self.queues.rrd_read = (self.queues.rrd_read + 1) % RX_RING;
            }
            let good = status & (RRS_RX_ERR_SUM | RRS_802_3_LEN_ERR) == 0 && count == 1;
            if good {
                let length = ((status & RRS_PKT_SIZE_MASK) as usize).saturating_sub(FCS);
                if (super::HEADER..=MAX_FRAME).contains(&length) {
                    let data =
                        take(&self.queues.free.buffer(index)[..length]).ok_or_else(|| {
                            LinkError::Adapter(format!(
                                "{} receive allocation failed",
                                self.location
                            ))
                        })?;
                    frame = Some(data);
                }
            }
            // atl1c_clean_rfd: the free descriptors go back to the driver.
            self.queues.rx_read = (index + count) % RX_RING;
            cleaned = true;
        }
        if cleaned && let Some(index) = self.queues.refill() {
            self.hardware.produce_receive(index);
        }
        Ok(frame)
    }
}
