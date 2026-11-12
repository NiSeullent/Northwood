// SPDX-License-Identifier: GPL-2.0-or-later AND ISC
// Copyright (c) 2013, 2021 Johannes Berg <johannes@sipsolutions.net>
// Copyright (c) 2012 Qualcomm Atheros, Inc.
// Copyright (c) 2026 Stela contributors
//
// Permission to use, copy, modify, and/or distribute this software for any
// purpose with or without fee is hereby granted, provided that the above
// copyright notice and this permission notice appear in all copies.
//
// THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
// WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
// MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
// ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
// WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
// ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
// OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
//! Qualcomm Atheros AR8161/AR8162/AR8171/AR8172 and Killer E2200/E2400/
//! E2500 Ethernet, ported from Linux `drivers/net/ethernet/atheros/alx`
//! (`main.c`, `hw.c`, `hw.h`, `reg.h`) at the pinned revision.
//!
//! One polled transmit queue and one receive queue, without checksum, TSO
//! or VLAN offloads and without interrupts: the interrupt mask stays zero
//! and the PHY link is sampled at most every 10 ms, as `alx_check_link`
//! reads it. Probe follows `alx_probe` (PCIe reset, PHY reset unless
//! firmware configured it, MAC reset, autonegotiation setup, permanent
//! address, PHY identity) and open follows `__alx_open`. Link up and down
//! run `alx_check_link`'s MAC start and MAC reset/reconfiguration, and a
//! receive return descriptor out of order runs `alx_reinit`, as Linux's
//! reset work does. All multicast is accepted, as `IFF_ALLMULTI` selects.
//!
//! Linux sets the PCI command register's I/O enable and clears its INTx
//! disable in `alx_reset_pcie`; this polled driver leaves the command
//! register to the PCI owner, which enables memory decoding and bus
//! mastering. Jumbo frames, Wake-on-LAN and suspend are outside this port.
//! DMA buffers live until reboot.
use super::{Link, LinkError, MAX_FRAME, MIN_FRAME, Mac, check_frame};
use crate::{
    Handoff,
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

pub use hardware::Registers;
use hardware::{Chip, Hardware, LinkState, RX_RING, TX_RING};
use registers::*;

/// `PCI_VENDOR_ID_ATTANSIC`.
const VENDOR: u16 = 0x1969;
/// `alx_pci_tbl`: AR8161, Killer E2200, E2400, E2500, AR8162, AR8171 and
/// AR8172.
pub const DEVICES: [u16; 7] = [0x1091, 0xe091, 0xe0a1, 0xe0b1, 0x1090, 0x10a1, 0x10a0];
/// The register window: every register this port names lies below it.
const REGISTER_BYTES: u64 = 0x2000;
/// How long a PHY link sample stays current.
const LINK_POLL_US: u64 = 10_000;
/// `ALX_RX_ALLOC_THRESH`: cleaned receive slots handed back at once.
const RX_ALLOC_THRESH: usize = 32;
const TRANSMIT_WAIT_US: u64 = 100_000;
const DESCRIPTOR: usize = 16;
const RFD: usize = 8;
const FCS: usize = 4;
/// MMD 3's devices-in-package registers, which `alx_get_phy_info` reads.
const MDIO_DEVS1: u16 = 5;
const MDIO_DEVS2: u16 = 6;

pub fn matches<C: ConfigSpace>(function: &Function<C>) -> bool {
    matches_identity(&function.identity())
}

/// Candidate selection from copied identity data.
pub fn matches_identity(identity: &crate::bus::pci::Identity) -> bool {
    identity.vendor == VENDOR && DEVICES.contains(&identity.device) && identity.class == [2, 0, 0]
}

/// The receive free and return rings and the transmit ring, with the
/// driver's indices into them (`alx_rx_queue`, `alx_tx_queue`).
struct Queues {
    transmit: Ring,
    /// Receive free descriptors: eight bytes each in the ring's
    /// descriptor storage, one 2 KiB buffer each.
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
        let free = Ring::with(dma, RX_RING)?;
        let returned = dma.allocate(RX_RING * DESCRIPTOR, 4096)?;
        if returned.len() != RX_RING * DESCRIPTOR {
            return Err(BlockError::Unsupported(
                "alx receive return ring allocation",
            ));
        }
        // One high-address register serves both receive rings.
        if returned.bus_address() >> 32 != free.base() >> 32 {
            return Err(BlockError::Unsupported(
                "alx receive rings across a 4 GiB boundary",
            ));
        }
        Ok(Self {
            transmit,
            free,
            returned,
            tx_write: 0,
            tx_read: 0,
            rx_write: 0,
            rx_read: 0,
            rrd_read: 0,
        })
    }

    /// `alx_free_buffers` and the index part of `alx_init_ring_ptrs`.
    fn reset(&mut self) {
        self.transmit.descriptors.bytes_mut().fill(0);
        self.free.descriptors.bytes_mut().fill(0);
        self.returned.bytes_mut().fill(0);
        self.tx_write = 0;
        self.tx_read = 0;
        self.rx_write = 0;
        self.rx_read = 0;
        self.rrd_read = 0;
    }

    /// `alx_refill_rx_ring`: hand every free slot up to the one before
    /// the next to read back, and return the producer index to publish.
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

    /// `alx_tpd_avail`.
    fn transmit_available(&self) -> usize {
        if self.tx_write >= self.tx_read {
            TX_RING + self.tx_read - self.tx_write - 1
        } else {
            self.tx_read - self.tx_write - 1
        }
    }
}

/// What the PHY link sampling last found and when.
#[derive(Default)]
struct Sampling {
    last: Option<u64>,
}

pub struct Alx<R: Registers = Mmio> {
    hardware: Hardware<R>,
    queues: Queues,
    mac: Mac,
    /// `pcie_get_readrq() >> 8` once the read request size is corrected,
    /// which later configurations read.
    read_request: u32,
    sampling: Sampling,
    /// A failure that left the controller's state unknown.
    failure: Option<BlockError>,
    stopped: bool,
    pub phy: u32,
    pub location: String,
}

impl Alx {
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
            return Err(BlockError::Unsupported("not an alx controller"));
        }
        function.power_up(clock.minimum())?;
        let mut pcie = None;
        let mut power = false;
        for (offset, id) in function.capabilities()? {
            match id {
                0x10 => pcie = Some(u16::from(offset)),
                1 => power = true,
                _ => {}
            }
        }
        if !power {
            return Err(BlockError::Unsupported(
                "alx without a power management capability",
            ));
        }
        let pcie = pcie.ok_or(BlockError::Unsupported("alx without a PCIe capability"))?;
        let revision = function.config().read8(0x08)?;
        let subsystem = function.config().read32(0x2c)?;
        let chip = Chip {
            device: function.device,
            revision,
            // alx_init_sw: an AR8161 rev 0 with Atheros subsystem 0091.
            link_patch: function.device == 0x1091 && subsystem == 0x0091_1969 && revision == 0,
        };
        if chip.rev() > REV_C0 {
            return Err(BlockError::Unsupported("unrecognized alx chip revision"));
        }
        let bars = function.bars()?;
        let device = crate::registers(handoff, bars[0], 0, REGISTER_BYTES)?;
        function.enable_polled()?;
        // alx_configure_basic: a read request size under 512 bytes is set
        // back to 512; the first configuration still uses the old value.
        let control = function.config().read16(pcie + 8)?;
        let original = (128u32 << ((control >> 12) & 7)) >> 8;
        if original < DEV_CTRL_MAXRRS_MIN {
            function.config().write16(
                pcie + 8,
                (control & !0x7000) | ((DEV_CTRL_MAXRRS_MIN as u16) << 12),
            )?;
        }
        let location = format!(
            "{} Qualcomm Atheros {:04x}",
            function.address, function.device
        );
        Self::bring_up(device, clock, chip, dma, original, location)
    }
}

impl<R: Registers> Alx<R> {
    /// `alx_probe` from the PCIe reset on, then `__alx_open`.
    fn bring_up(
        registers: R,
        clock: Timing,
        chip: Chip,
        dma: &mut dyn DmaAllocator,
        read_request: u32,
        location: String,
    ) -> Result<Self, BlockError> {
        let mut hardware = Hardware::new(registers, clock, chip, read_request);
        hardware.reset_pcie()?;
        let configured = hardware.phy_configured()?;
        if !configured {
            hardware.reset_phy()?;
        }
        hardware.reset_mac()?;
        if !configured {
            hardware.setup_speed_duplex()?;
        }
        let mac = hardware.permanent_mac()?;
        let phy = hardware.phy_identity()?;
        hardware.read_ext(3, MDIO_DEVS1)?;
        hardware.read_ext(3, MDIO_DEVS2)?;
        let queues = Queues::new(dma)?;
        let mut alx = Self {
            hardware,
            queues,
            mac,
            read_request: read_request.max(DEV_CTRL_MAXRRS_MIN),
            sampling: Sampling::default(),
            failure: None,
            stopped: false,
            phy,
            location,
        };
        // __alx_open: configure, ring pointers, old interrupts cleared.
        alx.hardware.configure(mac);
        alx.hardware.read_request = alx.read_request;
        alx.reinit_rings();
        alx.hardware.clear_interrupts();
        alx.check_link()?;
        Ok(alx)
    }

    /// `alx_reinit_rings`.
    fn reinit_rings(&mut self) {
        self.queues.reset();
        self.hardware.ring_pointers(
            self.queues.transmit.base(),
            self.queues.returned.bus_address(),
            self.queues.free.base(),
        );
        if let Some(index) = self.queues.refill() {
            self.hardware.produce_receive(index);
        }
    }

    /// `alx_check_link`, which a 10 ms sample cache keeps off the packet
    /// path.
    fn check_link(&mut self) -> Result<bool, BlockError> {
        let now = self.hardware.timer.now_us();
        self.sampling.last = Some(now);
        self.hardware.clear_phy_interrupt()?;
        let old = self.hardware.link;
        let new = self.hardware.read_phy_link()?;
        self.hardware.link = new;
        if old.speed == new.speed {
            return Ok(new.speed.is_some());
        }
        if new.speed.is_some() {
            self.hardware.post_phy_link()?;
            self.hardware.enable_aspm(true, true);
            self.hardware.start_mac();
        } else {
            // A MAC reset loses every setting; transmit work is dropped.
            self.hardware.reset_mac()?;
            self.reinit_rings();
            self.hardware.configure(self.mac);
            self.hardware.enable_aspm(false, true);
            self.hardware.post_phy_link()?;
        }
        Ok(new.speed.is_some())
    }

    /// `alx_reinit`: halt, then reactivate, as the reset work does.
    fn reinit(&mut self) -> Result<(), BlockError> {
        self.hardware.link = LinkState::default();
        self.hardware.reset_mac()?;
        self.hardware.enable_aspm(false, false);
        self.reinit_rings();
        self.hardware.configure(self.mac);
        self.hardware.clear_interrupts();
        self.sampling.last = None;
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
            .sampling
            .last
            .is_some_and(|last| self.hardware.timer.now_us().wrapping_sub(last) < LINK_POLL_US);
        if fresh {
            return Ok(self.hardware.link.speed.is_some());
        }
        self.check_link().map_err(|error| self.fail(error))
    }

    /// `alx_clean_tx_irq`: retire what the controller consumed.
    fn reclaim(&mut self) {
        let consumed = self.hardware.transmit_consumer();
        if consumed < TX_RING {
            self.queues.tx_read = consumed;
        }
    }

    /// Stop both queues and the MAC, as `alx_halt` does.
    pub fn stop(&mut self) -> Result<(), BlockError> {
        self.stopped = true;
        self.hardware.link = LinkState::default();
        self.hardware.reset_mac()?;
        self.hardware.enable_aspm(false, false);
        Ok(())
    }
}

impl<R: Registers> Link for Alx<R> {
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

impl<R: Registers> Alx<R> {
    /// The next good frame the controller returned, passed to `take` while
    /// its buffer is still held; `take` answers `None` when it cannot store
    /// the frame.
    fn receive_frame<T>(
        &mut self,
        mut take: impl FnMut(&[u8]) -> Option<T>,
    ) -> Result<Option<T>, LinkError> {
        self.checked()?;
        self.carrier_now()?;
        let mut cleaned = 0;
        let mut frame = None;
        while frame.is_none() {
            let base = self.queues.rrd_read * DESCRIPTOR;
            let status = self.queues.returned.read::<u32>(base + 12);
            if status & RRD_UPDATED == 0 {
                break;
            }
            fence(Ordering::Acquire);
            self.queues.returned.write(base + 12, status & !RRD_UPDATED);
            let word0 = self.queues.returned.read::<u32>(base);
            if field(word0, RRD_SI) as usize != self.queues.rx_read || field(word0, RRD_NOR) != 1 {
                // alx_clean_rx_irq schedules the reset work.
                self.reinit().map_err(|error| self.fail(error))?;
                return Ok(None);
            }
            let index = self.queues.rx_read;
            if status & (RRD_ERR_RES | RRD_ERR_LEN) != 0 {
                self.queues.returned.write(base + 12, 0u32);
            } else {
                let length = ((status & RRD_PKTLEN) as usize).saturating_sub(FCS);
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
            self.queues.rx_read = (index + 1) % RX_RING;
            self.queues.rrd_read = (self.queues.rrd_read + 1) % RX_RING;
            cleaned += 1;
            if cleaned > RX_ALLOC_THRESH {
                cleaned = 0;
                if let Some(index) = self.queues.refill() {
                    self.hardware.produce_receive(index);
                }
            }
        }
        if cleaned != 0
            && let Some(index) = self.queues.refill()
        {
            self.hardware.produce_receive(index);
        }
        Ok(frame)
    }
}
