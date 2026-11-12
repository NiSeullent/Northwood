// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (c) 2026 Stela contributors
//! A register and DMA peer for the alx port: MDIO with the debug and
//! extended PHY spaces, the self-clearing MAC reset, the station address
//! loader, and transmit/receive rings in host memory.
use super::*;
use crate::{Clock, bus::dma::RegionAllocator};
use alloc::{collections::BTreeMap, rc::Rc, vec};
use core::cell::RefCell;

const STATION: [u8; 6] = [0x00, 0x0b, 0x6a, 0xf6, 0x00, 0xdc];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phy {
    Clause22(u16),
    Debug(u16),
    Extended(u8, u16),
}

#[derive(Default)]
struct Model {
    registers: BTreeMap<usize, u32>,
    phy: BTreeMap<u16, u16>,
    debug: BTreeMap<u16, u16>,
    extended: BTreeMap<(u8, u16), u16>,
    phy_writes: Vec<(Phy, u16)>,
    writes: Vec<(usize, u32)>,
    /// Station address the eFuse loader provides on SLD_START.
    efuse: Option<[u32; 2]>,
    consume_transmit: bool,
    sent: Vec<Vec<u8>>,
    /// The next receive free descriptor and return slot the peer uses.
    rx_next: usize,
    mac_resets: usize,
}

impl Model {
    fn get(&self, offset: usize) -> u32 {
        self.registers.get(&offset).copied().unwrap_or(0)
    }

    fn mdio(&mut self, value: u32) -> u32 {
        let read = value & MDIO_OP_READ != 0;
        let data = value as u16;
        let target = if value & MDIO_MODE_EXT != 0 {
            let extension = self.get(MDIO_EXTN);
            Phy::Extended((extension >> 16) as u8, extension as u16)
        } else {
            let register = ((value >> 16) & 0x1f) as u16;
            match register {
                MII_DBG_DATA => Phy::Debug(self.phy.get(&MII_DBG_ADDR).copied().unwrap_or(0)),
                _ => Phy::Clause22(register),
            }
        };
        let result = if read {
            match target {
                Phy::Clause22(register) => self.phy.get(&register).copied().unwrap_or(0),
                Phy::Debug(register) => self.debug.get(&register).copied().unwrap_or(0),
                Phy::Extended(device, register) => {
                    self.extended.get(&(device, register)).copied().unwrap_or(0)
                }
            }
        } else {
            self.phy_writes.push((target, data));
            match target {
                Phy::Clause22(MII_BMCR) => {
                    self.phy.insert(MII_BMCR, data & !BMCR_RESET);
                }
                Phy::Clause22(register) => {
                    self.phy.insert(register, data);
                }
                Phy::Debug(register) => {
                    self.debug.insert(register, data);
                }
                Phy::Extended(device, register) => {
                    self.extended.insert((device, register), data);
                }
            }
            0
        };
        (value & !(MDIO_BUSY | 0xffff)) | u32::from(result)
    }

    fn address(&self, low: usize, high: usize) -> u64 {
        u64::from(self.get(low)) | u64::from(self.get(high)) << 32
    }

    /// The controller sends every descriptor up to the producer index.
    fn transmit(&mut self, producer: usize) {
        if !self.consume_transmit {
            return;
        }
        let ring = self.address(TPD_PRI0_ADDR_LO, TX_BASE_ADDR_HI);
        let mut consumer = self.get(TPD_PRI0_CIDX) as usize;
        while consumer != producer {
            let descriptor = (ring + (consumer * DESCRIPTOR) as u64) as *const u8;
            // SAFETY: the driver published this descriptor in the test's
            // live DMA memory, which outlives the peer.
            let (length, word1, buffer) = unsafe {
                (
                    core::ptr::read_volatile(descriptor.cast::<u16>()),
                    core::ptr::read_volatile(descriptor.add(4).cast::<u32>()),
                    core::ptr::read_volatile(descriptor.add(8).cast::<u64>()),
                )
            };
            assert_eq!(word1, TPD_EOP, "no offloads, one descriptor a frame");
            // SAFETY: the descriptor names its live DMA buffer.
            let bytes =
                unsafe { core::slice::from_raw_parts(buffer as *const u8, usize::from(length)) };
            self.sent.push(bytes.to_vec());
            consumer = (consumer + 1) % TX_RING;
        }
        self.registers.insert(TPD_PRI0_CIDX, consumer as u32);
    }

    /// DMA one frame into the next free buffer and return descriptor.
    fn deliver(&mut self, frame: &[u8], status: u32, slot: Option<usize>) -> bool {
        let index = self.rx_next;
        if index == self.get(RFD_PIDX) as usize {
            return false;
        }
        let free = self.address(RFD_ADDR_LO, RX_BASE_ADDR_HI);
        let returned = self.address(RRD_ADDR_LO, RX_BASE_ADDR_HI);
        // SAFETY: the driver published this free descriptor and its buffer
        // in the test's live DMA memory.
        unsafe {
            let buffer =
                core::ptr::read_volatile((free + (index * RFD) as u64) as *const u64) as *mut u8;
            assert_ne!(buffer as u64, 0, "an offered buffer");
            core::ptr::copy_nonoverlapping(frame.as_ptr(), buffer, frame.len());
            let descriptor = (returned + (index * DESCRIPTOR) as u64) as *mut u32;
            let slot = slot.unwrap_or(index) as u32;
            descriptor.write_volatile((slot << 20) | (1 << 16));
            descriptor
                .add(3)
                .write_volatile(RRD_UPDATED | status | (frame.len() + FCS) as u32);
        }
        self.rx_next = (index + 1) % RX_RING;
        true
    }
}

#[derive(Clone)]
struct Peer(Rc<RefCell<Model>>);

impl Registers for Peer {
    fn read16(&self, offset: usize) -> u16 {
        self.0.borrow().get(offset) as u16
    }
    fn read32(&self, offset: usize) -> u32 {
        self.0.borrow().get(offset)
    }
    fn write16(&self, offset: usize, value: u16) {
        let mut model = self.0.borrow_mut();
        model.writes.push((offset, u32::from(value)));
        model.registers.insert(offset, u32::from(value));
        if offset == TPD_PRI0_PIDX {
            model.transmit(usize::from(value));
        }
    }
    fn write32(&self, offset: usize, value: u32) {
        let mut model = self.0.borrow_mut();
        model.writes.push((offset, value));
        let stored = match offset {
            MDIO => model.mdio(value),
            MASTER if value & MASTER_DMA_MAC_RST != 0 => {
                model.mac_resets += 1;
                model.registers.insert(RFD_PIDX, 0);
                model.registers.insert(TPD_PRI0_CIDX, 0);
                model.rx_next = 0;
                value & !MASTER_DMA_MAC_RST
            }
            SLD if value & SLD_START != 0 => {
                if let Some([low, high]) = model.efuse {
                    model.registers.insert(STAD0, low);
                    model.registers.insert(STAD1, high);
                }
                value & !SLD_START
            }
            _ => value,
        };
        model.registers.insert(offset, stored);
    }
}

struct Fixture {
    model: Rc<RefCell<Model>>,
    _memory: Vec<u64>,
    dma: RegionAllocator,
}

impl Fixture {
    fn new() -> Self {
        let mut memory = vec![0u64; 400_000];
        let start = memory.as_mut_ptr() as u64;
        // SAFETY: the fixture keeps this memory alive beyond the driver.
        let dma = unsafe { RegionAllocator::new(start..start + 3_200_000) };
        let mut model = Model {
            consume_transmit: true,
            ..Model::default()
        };
        model.registers.insert(STAD0, 0x6af6_00dc);
        model.registers.insert(STAD1, 0x000b);
        model.registers.insert(SRAM5, 0x400);
        model.phy.insert(MII_PHYSID1, 0x004d);
        model.phy.insert(MII_PHYSID2, 0xd074);
        Self {
            model: Rc::new(RefCell::new(model)),
            _memory: memory,
            dma,
        }
    }

    fn chip(device: u16, revision: u8) -> Chip {
        Chip {
            device,
            revision,
            link_patch: false,
        }
    }

    fn attach(&mut self, chip: Chip) -> Result<Alx<Peer>, BlockError> {
        let clock = crate::timing::model(Clock {
            ticks_per_microsecond: 1,
        });
        Alx::bring_up(
            Peer(self.model.clone()),
            clock,
            chip,
            &mut self.dma,
            1,
            String::from("peer"),
        )
    }

    fn link(&self, pssr: Option<u16>) {
        let mut model = self.model.borrow_mut();
        match pssr {
            Some(value) => {
                model.phy.insert(MII_BMSR, BMSR_LSTATUS);
                model.phy.insert(MII_GIGA_PSSR, value);
            }
            None => {
                model.phy.insert(MII_BMSR, 0);
            }
        }
    }
}

fn sample<R: Registers>(alx: &mut Alx<R>) -> bool {
    alx.sampling.last = None;
    alx.carrier()
}

const GIGABIT_FULL: u16 = GIGA_PSSR_SPD_DPLX_RESOLVED | GIGA_PSSR_1000MBS | GIGA_PSSR_DPLX;

#[test]
fn the_device_table_is_linuxs_alx_pci_table() {
    let identity = |vendor, device, class| crate::bus::pci::Identity {
        address: crate::bus::pci::Address {
            segment: 0,
            bus: 3,
            device: 0,
            function: 0,
        },
        vendor,
        device,
        class,
    };
    for device in [0x1091, 0xe091, 0xe0a1, 0xe0b1, 0x1090, 0x10a1, 0x10a0] {
        assert!(matches_identity(&identity(0x1969, device, [2, 0, 0])));
    }
    // atl1c and atl1e parts share the vendor and stay unclaimed.
    for device in [0x1062, 0x1063, 0x1073, 0x1083, 0x2060, 0x2062, 0x1026] {
        assert!(!matches_identity(&identity(0x1969, device, [2, 0, 0])));
    }
    assert!(!matches_identity(&identity(0x1969, 0x1091, [2, 0x80, 0])));
}

#[test]
fn attach_resets_the_phy_advertises_and_opens_one_polled_queue_pair() {
    let mut fixture = Fixture::new();
    let alx = fixture.attach(Fixture::chip(0x1091, REV_B0 << 3)).unwrap();
    assert_eq!(alx.mac(), Mac(STATION));
    assert_eq!(alx.phy, 0x004d_d074);
    let model = fixture.model.borrow();
    // alx_reset_phy's defaults, alx_setup_speed_duplex's advertisement.
    for expected in [
        (Phy::Debug(MIIDBG_LEGCYPS), LEGCYPS_DEF),
        (Phy::Extended(MIIEXT_ANEG, MIIEXT_LOCAL_EEEADV), 0),
        (
            Phy::Extended(MIIEXT_ANEG, MIIEXT_NLP78),
            MIIEXT_NLP78_120M_DEF,
        ),
        (Phy::Clause22(MII_IER), IER_LINK_UP | IER_LINK_DOWN),
        (Phy::Clause22(MII_ADVERTISE), 0x21e1),
        (Phy::Clause22(MII_CTRL1000), ADVERTISE_1000FULL),
        (Phy::Clause22(MII_BMCR), 0x9200),
        (Phy::Clause22(MII_DBG_ADDR), PHY_INITED),
    ] {
        assert!(model.phy_writes.contains(&expected), "{expected:x?}");
    }
    assert_eq!(field(model.get(DRV), DRV_PHY), 0xb8);
    assert_eq!(model.get(IMR), 0, "polled: no interrupt source enabled");
    assert_eq!(model.get(RXQ0) & RXQ0_RSS_HASH_EN, 0);
    assert_ne!(model.get(MAC_CTRL) & MAC_CTRL_MULTIALL_EN, 0);
    assert_eq!(model.get(MAC_CTRL) & (MAC_CTRL_RX_EN | MAC_CTRL_TX_EN), 0);
    assert_eq!(model.get(STAD0), 0x6af6_00dc);
    assert_eq!(model.get(RFD_BUF_SZ), 1544);
    assert_eq!(model.get(SRAM9), SRAM_LOAD_PTR);
    assert_eq!(model.get(RFD_PIDX) as usize, RX_RING - 1);
    assert_eq!(model.get(TPD_RING_SZ) as usize, TX_RING);
    // The first configuration keeps the read request found, 256 bytes.
    let dma = model
        .writes
        .iter()
        .find(|(offset, _)| *offset == DMA)
        .unwrap()
        .1;
    assert_eq!((dma >> DMA_RREQ_BLEN_SHIFT) & 0x3f, 1);
    assert_eq!(
        (dma >> DMA_RCHNL_SEL_SHIFT) & 3,
        3,
        "rev B0 has four channels"
    );
}

#[test]
fn a_phy_firmware_configured_keeps_its_advertisement() {
    let mut fixture = Fixture::new();
    {
        let mut model = fixture.model.borrow_mut();
        model.registers.insert(PHY_CTRL, PHY_CTRL_DSPRST_OUT);
        model.registers.insert(DRV, 0x1700_0000);
        model.phy.insert(MII_DBG_ADDR, PHY_INITED);
    }
    fixture.attach(Fixture::chip(0x1091, 0)).unwrap();
    let model = fixture.model.borrow();
    assert!(
        !model
            .phy_writes
            .iter()
            .any(|(target, _)| matches!(target, Phy::Clause22(MII_BMCR) | Phy::Debug(_))),
        "neither alx_reset_phy nor alx_setup_speed_duplex runs"
    );
}

#[test]
fn fast_ethernet_parts_do_not_advertise_gigabit() {
    let mut fixture = Fixture::new();
    fixture.attach(Fixture::chip(0x1090, 0)).unwrap();
    let model = fixture.model.borrow();
    assert!(model.phy_writes.contains(&(Phy::Clause22(MII_CTRL1000), 0)));
    assert_eq!(model.get(RXQ0) & 3, 0, "no 100M ASPM threshold");
}

#[test]
fn the_station_address_falls_back_to_the_efuse_and_otherwise_refuses() {
    let mut fixture = Fixture::new();
    {
        let mut model = fixture.model.borrow_mut();
        model.registers.insert(STAD0, 0);
        model.registers.insert(STAD1, 0);
        model.efuse = Some([0x6af6_00dc, 0x000b]);
    }
    assert_eq!(
        fixture.attach(Fixture::chip(0x10a1, 0)).unwrap().mac(),
        Mac(STATION)
    );

    let mut fixture = Fixture::new();
    {
        let mut model = fixture.model.borrow_mut();
        model.registers.insert(STAD0, 0);
        model.registers.insert(STAD1, 0);
    }
    assert!(matches!(
        fixture.attach(Fixture::chip(0x10a1, 0)),
        Err(BlockError::Unsupported(_))
    ));
}

#[test]
fn link_up_starts_the_mac_at_the_resolved_speed_and_down_resets_it() {
    let mut fixture = Fixture::new();
    let mut alx = fixture.attach(Fixture::chip(0x1091, REV_B0 << 3)).unwrap();
    assert!(!sample(&mut alx));
    fixture.link(Some(GIGABIT_FULL));
    assert!(sample(&mut alx));
    {
        let model = fixture.model.borrow();
        let mac = model.get(MAC_CTRL);
        assert_ne!(mac & MAC_CTRL_RX_EN, 0);
        assert_ne!(mac & MAC_CTRL_TX_EN, 0);
        assert_ne!(mac & MAC_CTRL_FULLD, 0);
        assert_eq!(field(mac, MAC_CTRL_SPEED), MAC_CTRL_SPEED_1000);
        assert_ne!(model.get(RXQ0) & RXQ0_EN, 0);
        assert_ne!(model.get(TXQ0) & TXQ0_EN, 0);
        assert_ne!(model.get(PMCTRL) & (PMCTRL_L0S_EN | PMCTRL_L1_EN), 0);
        // alx_post_phy_link on rev B0: a short cable.
        assert!(
            model
                .phy_writes
                .contains(&(Phy::Debug(MIIDBG_AZ_ANADECT), AZ_ANADECT_DEF))
        );
    }
    alx.transmit(&[0x11; 60]).unwrap();
    let resets = fixture.model.borrow().mac_resets;
    fixture.link(None);
    assert!(!sample(&mut alx));
    {
        let model = fixture.model.borrow();
        assert_eq!(model.mac_resets, resets + 1);
        assert_eq!(model.get(MAC_CTRL) & (MAC_CTRL_RX_EN | MAC_CTRL_TX_EN), 0);
        assert_eq!(model.get(PMCTRL) & PMCTRL_L0S_EN, 0);
        assert_ne!(model.get(PMCTRL) & PMCTRL_L1_EN, 0);
        assert_eq!(
            model.get(RFD_PIDX) as usize,
            RX_RING - 1,
            "rings reinitialized"
        );
    }
    assert_eq!(alx.queues.tx_write, 0);
    assert_eq!(alx.transmit(&[0x22; 60]), Err(LinkError::NoCarrier));
    // Half duplex 100 Mb/s after reconnecting.
    fixture.link(Some(GIGA_PSSR_SPD_DPLX_RESOLVED | GIGA_PSSR_100MBS));
    assert!(sample(&mut alx));
    let mac = fixture.model.borrow().get(MAC_CTRL);
    assert_eq!(mac & MAC_CTRL_FULLD, 0);
    assert_eq!(field(mac, MAC_CTRL_SPEED), MAC_CTRL_SPEED_10_100);
}

#[test]
fn an_unresolved_phy_speed_fails_the_link() {
    let mut fixture = Fixture::new();
    let mut alx = fixture.attach(Fixture::chip(0x1091, 0)).unwrap();
    fixture.link(Some(GIGA_PSSR_1000MBS));
    assert!(!sample(&mut alx));
    assert!(matches!(alx.receive(), Err(LinkError::Adapter(_))));
}

#[test]
fn transmit_fills_one_descriptor_per_frame_and_pads_short_frames() {
    let mut fixture = Fixture::new();
    let mut alx = fixture.attach(Fixture::chip(0x1091, 0)).unwrap();
    fixture.link(Some(GIGABIT_FULL));
    assert!(sample(&mut alx));
    let frame: Vec<u8> = (0..20u8).collect();
    alx.transmit(&frame).unwrap();
    alx.transmit(&[0x5a; MAX_FRAME]).unwrap();
    let model = fixture.model.borrow();
    assert_eq!(model.sent.len(), 2);
    assert_eq!(&model.sent[0][..20], &frame[..]);
    assert_eq!(model.sent[0].len(), MIN_FRAME);
    assert!(model.sent[0][20..].iter().all(|&byte| byte == 0));
    assert_eq!(model.sent[1], vec![0x5a; MAX_FRAME]);
    assert_eq!(model.get(TPD_PRI0_PIDX), 2);
    drop(model);
    assert_eq!(
        alx.transmit(&[0; MAX_FRAME + 1]),
        Err(LinkError::Invalid("frame exceeds the Ethernet maximum"))
    );
}

#[test]
fn a_full_ring_waits_for_the_consumer_and_then_reports_a_stall() {
    let mut fixture = Fixture::new();
    let mut alx = fixture.attach(Fixture::chip(0x1091, 0)).unwrap();
    fixture.link(Some(GIGABIT_FULL));
    assert!(sample(&mut alx));
    fixture.model.borrow_mut().consume_transmit = false;
    for _ in 0..TX_RING - 1 {
        alx.transmit(&[1; 60]).unwrap();
    }
    assert!(matches!(alx.transmit(&[1; 60]), Err(LinkError::Adapter(_))));
    // The consumer catches up: a descriptor frees.
    fixture
        .model
        .borrow_mut()
        .registers
        .insert(TPD_PRI0_CIDX, 1);
    alx.transmit(&[2; 60]).unwrap();
}

#[test]
fn receive_returns_good_frames_without_fcs_and_drops_errors() {
    let mut fixture = Fixture::new();
    let mut alx = fixture.attach(Fixture::chip(0x1091, 0)).unwrap();
    fixture.link(Some(GIGABIT_FULL));
    assert!(sample(&mut alx));
    assert_eq!(alx.receive(), Ok(None));
    let frame: Vec<u8> = (0..64u8).collect();
    {
        let mut model = fixture.model.borrow_mut();
        assert!(model.deliver(&[0xee; 64], RRD_ERR_LEN, None));
        assert!(model.deliver(&frame, 0, None));
    }
    assert_eq!(alx.receive(), Ok(Some(frame)));
    assert_eq!(alx.receive(), Ok(None));
    assert_eq!(alx.queues.rx_read, 2);
    // The cleaned slots go back to the controller.
    assert_eq!(fixture.model.borrow().get(RFD_PIDX), 1);
}

#[test]
fn receive_keeps_up_across_the_ring_wrap() {
    let mut fixture = Fixture::new();
    let mut alx = fixture.attach(Fixture::chip(0x1091, 0)).unwrap();
    fixture.link(Some(GIGABIT_FULL));
    assert!(sample(&mut alx));
    let mut reused = Vec::with_capacity(MAX_FRAME);
    let storage = reused.as_ptr();
    for round in 0..(RX_RING * 2) {
        let frame = vec![round as u8; 100];
        assert!(fixture.model.borrow_mut().deliver(&frame, 0, None));
        if round % 2 == 0 {
            assert_eq!(alx.receive(), Ok(Some(frame)));
        } else {
            // The stack's path refills its own buffer in place.
            assert_eq!(alx.receive_reusing(&mut reused), Ok(true));
            assert_eq!(reused, frame);
            assert_eq!(reused.as_ptr(), storage);
        }
    }
    assert_eq!(alx.receive_reusing(&mut reused), Ok(false));
    assert_eq!(reused, vec![(RX_RING * 2 - 1) as u8; 100]);
    alx.stop().unwrap();
    assert!(alx.receive_reusing(&mut reused).is_err());
    assert_eq!(reused, vec![(RX_RING * 2 - 1) as u8; 100]);
}

#[test]
fn an_out_of_order_return_descriptor_reinitializes_the_queues() {
    let mut fixture = Fixture::new();
    let mut alx = fixture.attach(Fixture::chip(0x1091, 0)).unwrap();
    fixture.link(Some(GIGABIT_FULL));
    assert!(sample(&mut alx));
    let resets = fixture.model.borrow().mac_resets;
    assert!(fixture.model.borrow_mut().deliver(&[1; 64], 0, Some(7)));
    assert_eq!(alx.receive(), Ok(None));
    assert_eq!(fixture.model.borrow().mac_resets, resets + 1);
    assert_eq!(alx.queues.rx_read, 0);
    // The next sample finds the link and starts the MAC again.
    assert!(alx.carrier());
    assert_ne!(fixture.model.borrow().get(MAC_CTRL) & MAC_CTRL_RX_EN, 0);
    let frame = vec![9; 64];
    assert!(fixture.model.borrow_mut().deliver(&frame, 0, None));
    assert_eq!(alx.receive(), Ok(Some(frame)));
}

#[test]
fn stop_resets_the_mac_and_refuses_further_frames() {
    let mut fixture = Fixture::new();
    let mut alx = fixture.attach(Fixture::chip(0x1091, 0)).unwrap();
    fixture.link(Some(GIGABIT_FULL));
    assert!(sample(&mut alx));
    let resets = fixture.model.borrow().mac_resets;
    alx.stop().unwrap();
    {
        let model = fixture.model.borrow();
        assert_eq!(model.mac_resets, resets + 1);
        assert_eq!(model.get(MAC_CTRL) & (MAC_CTRL_RX_EN | MAC_CTRL_TX_EN), 0);
        assert_eq!(model.get(RXQ0) & RXQ0_EN, 0);
        assert_eq!(model.get(PMCTRL) & (PMCTRL_L0S_EN | PMCTRL_L1_EN), 0);
        assert_eq!(model.get(MSIX_MASK), u32::MAX);
    }
    assert!(!alx.carrier());
    assert!(matches!(alx.transmit(&[0; 60]), Err(LinkError::Adapter(_))));
    assert!(matches!(alx.receive(), Err(LinkError::Adapter(_))));
}

#[test]
fn revision_a_cr_parts_keep_aspm_off_across_the_mac_reset() {
    let mut fixture = Fixture::new();
    fixture
        .model
        .borrow_mut()
        .registers
        .insert(PMCTRL, PMCTRL_L1_EN | PMCTRL_L0S_EN);
    // Revision A1 with the CR bit set.
    fixture
        .attach(Fixture::chip(0x1091, (REV_A1 << 3) | 1))
        .unwrap();
    let model = fixture.model.borrow();
    let reset = model
        .writes
        .iter()
        .position(|&(offset, value)| offset == MASTER && value & MASTER_DMA_MAC_RST != 0)
        .unwrap();
    let before = model.writes[..reset]
        .iter()
        .rev()
        .find(|(offset, _)| *offset == PMCTRL)
        .unwrap()
        .1;
    assert_eq!(before & (PMCTRL_L1_EN | PMCTRL_L0S_EN), 0);
    assert!(
        model.writes[reset..]
            .iter()
            .any(|&(offset, value)| offset == MASTER && value & MASTER_PCLKSEL_SRDS != 0)
    );
}
