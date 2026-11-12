// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (c) 2026 Stela contributors
//! A register and DMA peer for the atl1c port: MDIO with the debug and
//! extended PHY spaces, the self-clearing soft reset, the EEPROM station
//! address loader, and transmit/receive rings in host memory.
use super::*;
use crate::{Clock, bus::dma::RegionAllocator};
use alloc::{collections::BTreeMap, rc::Rc, vec};
use core::cell::RefCell;

const STATION: [u8; 6] = [0x00, 0x13, 0x74, 0x01, 0x02, 0x03];

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
    /// The MDIO clock selection of every PHY access.
    clocks: Vec<u32>,
    writes: Vec<(usize, u32)>,
    /// Station address the EEPROM loader provides on `TWSI_CTRL_SW_LDSTART`.
    eeprom: Option<[u32; 2]>,
    consume_transmit: bool,
    sent: Vec<Vec<u8>>,
    rx_next: usize,
    soft_resets: usize,
    /// Idle status bits that stay set: a unit that never stops.
    busy: u32,
}

impl Model {
    fn get(&self, offset: usize) -> u32 {
        self.registers.get(&offset).copied().unwrap_or(0)
    }

    fn mdio(&mut self, value: u32) -> u32 {
        self.clocks.push((value >> MDIO_CTRL_CLK_SEL_SHIFT) & 7);
        let read = value & MDIO_CTRL_OP_READ != 0;
        let data = value as u16;
        let target = if value & MDIO_CTRL_MODE_EXT != 0 {
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
        (value & !(MDIO_CTRL_BUSY | MDIO_CTRL_START | 0xffff)) | u32::from(result)
    }

    fn address(&self, low: usize, high: usize) -> u64 {
        u64::from(self.get(low)) | u64::from(self.get(high)) << 32
    }

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

    /// DMA one frame into the next offered buffer and return descriptor;
    /// `index` overrides the free descriptor the return descriptor names.
    fn deliver(&mut self, frame: &[u8], status: u32, index: Option<usize>, count: u32) -> bool {
        let slot = self.rx_next;
        if slot == self.get(MB_RFD0_PROD_IDX) as usize {
            return false;
        }
        let free = self.address(RFD0_HEAD_ADDR_LO, RX_BASE_ADDR_HI);
        let returned = self.address(RRD0_HEAD_ADDR_LO, RX_BASE_ADDR_HI);
        // SAFETY: the driver published this free descriptor and its buffer
        // in the test's live DMA memory.
        unsafe {
            let buffer =
                core::ptr::read_volatile((free + (slot * RFD) as u64) as *const u64) as *mut u8;
            assert_ne!(buffer as u64, 0, "an offered buffer");
            core::ptr::copy_nonoverlapping(frame.as_ptr(), buffer, frame.len());
            let descriptor = (returned + (slot * DESCRIPTOR) as u64) as *mut u32;
            let named = index.unwrap_or(slot) as u32;
            descriptor.write_volatile((named << 20) | (count << 16));
            descriptor
                .add(3)
                .write_volatile(RRS_RXD_UPDATED | status | (frame.len() + FCS) as u32);
        }
        self.rx_next = (slot + 1) % RX_RING;
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
        let model = self.0.borrow();
        match offset {
            IDLE_STATUS => model.busy,
            _ => model.get(offset),
        }
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
            MDIO_CTRL => model.mdio(value),
            MASTER_CTRL if value & MASTER_CTRL_SOFT_RST != 0 => {
                model.soft_resets += 1;
                model.registers.insert(MB_RFD0_PROD_IDX, 0);
                model.registers.insert(TPD_PRI0_CIDX, 0);
                model.rx_next = 0;
                value & !MASTER_CTRL_SOFT_RST
            }
            TWSI_CTRL if value & TWSI_CTRL_SW_LDSTART != 0 => {
                if let Some([low, high]) = model.eeprom {
                    model.registers.insert(MAC_STA_ADDR, low);
                    model.registers.insert(MAC_STA_ADDR + 4, high);
                }
                value & !TWSI_CTRL_SW_LDSTART
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
        model.registers.insert(MAC_STA_ADDR, 0x7401_0203);
        model.registers.insert(MAC_STA_ADDR + 4, 0x0013);
        model.phy.insert(MII_PHYSID1, 0x004d);
        model.phy.insert(MII_PHYSID2, 0xd024);
        Self {
            model: Rc::new(RefCell::new(model)),
            _memory: memory,
            dma,
        }
    }

    fn chip(device: u16) -> Chip {
        Chip {
            kind: Kind::from_device(device).unwrap(),
            device,
            revision: 0xc0,
            link_patch: false,
        }
    }

    fn attach_chip(&mut self, chip: Chip) -> Result<Atl1c<Peer>, BlockError> {
        let clock = crate::timing::model(Clock {
            ticks_per_microsecond: 1,
        });
        Atl1c::bring_up(
            Peer(self.model.clone()),
            clock,
            chip,
            &mut self.dma,
            2,
            String::from("peer"),
        )
    }

    fn attach(&mut self, device: u16) -> Result<Atl1c<Peer>, BlockError> {
        self.attach_chip(Self::chip(device))
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

fn sample<R: Registers>(adapter: &mut Atl1c<R>) -> bool {
    adapter.sampled = None;
    adapter.carrier()
}

const GIGABIT_FULL: u16 = GIGA_PSSR_SPD_DPLX_RESOLVED | GIGA_PSSR_1000MBS | GIGA_PSSR_DPLX;
const FAST_FULL: u16 = GIGA_PSSR_SPD_DPLX_RESOLVED | GIGA_PSSR_100MBS | GIGA_PSSR_DPLX;

#[test]
fn the_device_table_is_linuxs_atl1c_pci_table() {
    let identity = |vendor, device, class| crate::bus::pci::Identity {
        address: crate::bus::pci::Address {
            segment: 0,
            bus: 2,
            device: 0,
            function: 0,
        },
        vendor,
        device,
        class,
    };
    for device in [0x1062, 0x1063, 0x2060, 0x2062, 0x1073, 0x1083] {
        assert!(matches_identity(&identity(0x1969, device, [2, 0, 0])));
        assert!(!crate::alx::matches_identity(&identity(
            0x1969,
            device,
            [2, 0, 0]
        )));
    }
    // alx parts and atl1e/atl1 parts stay with other drivers.
    for device in [0x1091, 0x1090, 0x10a1, 0xe091, 0x1026, 0x1048] {
        assert!(!matches_identity(&identity(0x1969, device, [2, 0, 0])));
    }
    assert!(!matches_identity(&identity(0x1969, 0x1063, [2, 0x80, 0])));
    assert!(!matches_identity(&identity(0x10ec, 0x1063, [2, 0, 0])));
}

#[test]
fn the_read_request_correction_and_link_patch_table_match_linux() {
    // 512 bytes: block code 2, unchanged.
    assert_eq!(dma_read_block(2 << 12), (2, None));
    // 4096 bytes: limited to 1024's code.
    assert_eq!(dma_read_block(5 << 12), (3, None));
    // 128 and 256 bytes are raised to 512.
    assert_eq!(dma_read_block(0), (2, Some(2)));
    assert_eq!(dma_read_block(1 << 12), (2, Some(2)));
    assert!(link_patch(0x1083, 0xc0, 0x7680_1462));
    assert!(link_patch(0x2060, 0xc1, 0x8152_1019));
    assert!(!link_patch(0x2060, 0xc0, 0x8152_1019));
    assert!(!link_patch(0x1063, 0xc0, 0x8152_1019));
}

#[test]
fn attach_resets_the_phy_advertises_and_opens_one_polled_queue_pair() {
    let mut fixture = Fixture::new();
    let adapter = fixture.attach(0x1063).unwrap();
    assert_eq!(adapter.mac(), Mac(STATION));
    assert_eq!(adapter.phy, 0x004d_d024);
    assert_eq!(adapter.speed(), None);
    let model = fixture.model.borrow();
    for expected in [
        (Phy::Debug(MIIDBG_LEGCYPS), L1C_LEGCYPS_DEF),
        (Phy::Debug(MIIDBG_SYSMODCTRL), SYSMODCTRL_IECHOADJ_DEF),
        (Phy::Debug(MIIDBG_ANACTRL), ANACTRL_DEF),
        (Phy::Debug(MIIDBG_SRDSYSMOD), SRDSYSMOD_DEF),
        (Phy::Debug(MIIDBG_TST10BTCFG), TST10BTCFG_DEF),
        (Phy::Debug(MIIDBG_TST100BTCFG), 0xe1ac),
        (Phy::Clause22(MII_IER), IER_LINK_UP | IER_LINK_DOWN),
        (Phy::Clause22(MII_ADVERTISE), 0x0de0),
        (Phy::Clause22(MII_CTRL1000), 0x0300),
        (Phy::Clause22(MII_BMCR), 0x9200),
    ] {
        assert!(model.phy_writes.contains(&expected), "{expected:x?}");
    }
    // L1C has no EEE: the extended space stays untouched.
    assert!(
        !model
            .phy_writes
            .iter()
            .any(|(target, _)| matches!(target, Phy::Extended(..)))
    );
    // The PHY reset pulse: analog reset first, then the external reset.
    let resets: Vec<u32> = model
        .writes
        .iter()
        .filter(|(offset, _)| *offset == GPHY_CTRL)
        .map(|&(_, value)| value)
        .collect();
    assert_eq!(resets.len(), 2);
    assert_eq!(resets[0] & GPHY_CTRL_EXT_RESET, 0);
    assert_ne!(resets[0] & (GPHY_CTRL_SEL_ANA_RST | GPHY_CTRL_HIB_EN), 0);
    assert_ne!(resets[1] & GPHY_CTRL_EXT_RESET, 0);
    // First generation: receive detection forced, no 25 MHz wake bit.
    assert_ne!(model.get(PCIE_PHYMISC) & PCIE_PHYMISC_FORCE_RCV_DET, 0);
    assert_eq!(model.get(IMR), 0, "polled: no interrupt source enabled");
    assert_eq!(model.get(MAC_STA_ADDR), 0x7401_0203);
    assert_eq!(model.get(MAC_STA_ADDR + 4), 0x0013);
    assert_ne!(model.get(MAC_CTRL) & MAC_CTRL_MC_ALL_EN, 0);
    assert_ne!(model.get(MAC_CTRL) & MAC_CTRL_SPEED_MODE_SW, 0);
    assert_eq!(model.get(MAC_CTRL) & (MAC_CTRL_RX_EN | MAC_CTRL_TX_EN), 0);
    assert_eq!(model.get(RX_BUF_SIZE), 1522);
    assert_eq!(model.get(MTU), 1522);
    assert_eq!(model.get(LOAD_PTR), 1);
    assert_eq!(model.get(MB_RFD0_PROD_IDX) as usize, RX_RING - 1);
    assert_eq!(model.get(TPD_RING_SIZE) as usize, TX_RING);
    assert_ne!(
        model.get(TPD_PRI1_ADDR_LO),
        0,
        "the idle ring is programmed"
    );
    assert_eq!(model.get(TXQ_CTRL), L1C_TXQ_CFGV);
    assert_eq!(model.get(RXQ_CTRL), (8 << 20) | 3, "gigabit ASPM limit");
    assert_eq!(
        model.get(DMA_CTRL),
        4 | (1 << 10) | (2 << 4) | (4 << 16) | (15 << 11)
    );
    assert_eq!(model.get(IRQ_MODRT_TIMER_INIT), 1000 | (200 << 16));
    assert_eq!(model.get(TX_TSO_OFFLOAD_THRESH), 896);
    // Probe's MAC reset and the link-down reset of atl1c_up.
    assert_eq!(model.soft_resets, 2);
    // Link down: ASPM L1 allowed, L0s not.
    assert_ne!(model.get(PM_CTRL) & PM_CTRL_CLK_SWH_L1, 0);
    assert_eq!(model.get(PM_CTRL) & PM_CTRL_ASPM_L0S_EN, 0);
}

#[test]
fn fast_ethernet_l2c_b_parts_use_their_sram_layout_and_no_gigabit() {
    let mut fixture = Fixture::new();
    fixture
        .model
        .borrow_mut()
        .registers
        .insert(SERDES, SERDES_PHY_CLK_SLOWDOWN | SERDES_MAC_CLK_SLOWDOWN);
    fixture.attach(0x2060).unwrap();
    let model = fixture.model.borrow();
    assert!(model.phy_writes.contains(&(Phy::Clause22(MII_CTRL1000), 0)));
    assert_eq!(model.get(SRAM_RXF_LEN), 0x02a0);
    assert_eq!(model.get(SRAM_TRD_ADDR), 0x03df_03c0);
    assert_eq!(model.get(TXQ_CTRL), L2CB_TXQ_CFGV);
    assert_eq!(model.get(RXQ_CTRL), 8 << 20, "no gigabit ASPM limit");
    assert_eq!(
        model.get(SERDES) & (SERDES_PHY_CLK_SLOWDOWN | SERDES_MAC_CLK_SLOWDOWN),
        0
    );
    // The revision 1.0 PCIe patch and the clock switch fix.
    assert_eq!(field(model.get(PCIE_PHYMISC2), PCIE_PHYMISC2_CDR_BW), 3);
    assert_ne!(model.get(LINK_CTRL) & LINK_CTRL_EXT_SYNC, 0);
    assert_ne!(model.get(MASTER_CTRL) & MASTER_CTRL_WAKEN_25M, 0);
    assert!(
        model
            .phy_writes
            .contains(&(Phy::Debug(MIIDBG_CABLE1TH_DET), CABLE1TH_DET_EN))
    );
}

#[test]
fn l1d_2_disables_eee_slows_its_clocks_and_uses_the_slow_mdio_clock_while_down() {
    let mut fixture = Fixture::new();
    let mut adapter = fixture.attach(0x1083).unwrap();
    {
        let model = fixture.model.borrow();
        assert!(
            model
                .phy_writes
                .contains(&(Phy::Debug(MIIDBG_LEGCYPS), L1D_LEGCYPS_DEF))
        );
        assert!(
            model
                .phy_writes
                .contains(&(Phy::Extended(MIIEXT_ANEG, MIIEXT_LOCAL_EEEADV), 0))
        );
        assert!(
            model
                .phy_writes
                .contains(&(Phy::Extended(MIIEXT_PCS, MIIEXT_CLDCTRL3), L2CB_CLDCTRL3))
        );
        assert_ne!(
            model.get(SERDES) & (SERDES_PHY_CLK_SLOWDOWN | SERDES_MAC_CLK_SLOWDOWN),
            0
        );
        assert!(
            model
                .clocks
                .iter()
                .all(|&clock| clock == MDIO_CTRL_CLK_25_128)
        );
        assert_eq!(model.get(RXQ_CTRL), 8 << 20, "L1D_2 has no ASPM limit");
    }
    fixture.model.borrow_mut().clocks.clear();
    fixture.link(Some(GIGABIT_FULL));
    assert!(sample(&mut adapter));
    // The link-up cable length check, then the fast clock.
    let model = fixture.model.borrow();
    assert!(
        model
            .phy_writes
            .contains(&(Phy::Debug(MIIDBG_AZ_ANADECT), AZ_ANADECT_DEF))
    );
    assert_eq!(model.clocks.last(), Some(&MDIO_CTRL_CLK_25_4));
}

#[test]
fn the_station_address_comes_from_the_eeprom_load_or_the_adapter_is_refused() {
    let mut fixture = Fixture::new();
    {
        let mut model = fixture.model.borrow_mut();
        model.registers.insert(MAC_STA_ADDR, 0);
        model.registers.insert(MAC_STA_ADDR + 4, 0);
        model.registers.insert(TWSI_DEBUG, TWSI_DEBUG_DEV_EXIST);
        model.eeprom = Some([0x7401_0203, 0x0013]);
    }
    assert_eq!(fixture.attach(0x1062).unwrap().mac(), Mac(STATION));
    {
        let model = fixture.model.borrow();
        // L2C: the OTP clock on for the load and off again.
        let otp: Vec<u32> = model
            .writes
            .iter()
            .filter(|(offset, _)| *offset == OTP_CTRL)
            .map(|&(_, value)| value)
            .collect();
        assert_eq!(otp, vec![OTP_CTRL_CLK_EN, 0]);
    }

    // L2C_B2 raises the PHY voltage around the load.
    let mut fixture = Fixture::new();
    {
        let mut model = fixture.model.borrow_mut();
        model.registers.insert(MAC_STA_ADDR, 0);
        model.registers.insert(MAC_STA_ADDR + 4, 0);
        model.registers.insert(MASTER_CTRL, MASTER_CTRL_OTP_SEL);
        model.eeprom = Some([0x7401_0203, 0x0013]);
    }
    fixture.attach(0x2062).unwrap();
    assert!(
        fixture
            .model
            .borrow()
            .phy_writes
            .contains(&(Phy::Debug(MIIDBG_VOLT_CTRL), VOLT_CTRL_SWLOWEST))
    );

    let mut fixture = Fixture::new();
    {
        let mut model = fixture.model.borrow_mut();
        model.registers.insert(MAC_STA_ADDR, 0);
        model.registers.insert(MAC_STA_ADDR + 4, 0);
    }
    assert!(matches!(
        fixture.attach(0x1063),
        Err(BlockError::Unsupported(_))
    ));
}

#[test]
fn a_mac_that_stays_busy_after_reset_refuses_the_adapter() {
    let mut fixture = Fixture::new();
    fixture.model.borrow_mut().busy = IDLE_STATUS_RXMAC_BUSY;
    assert!(matches!(
        fixture.attach(0x1063),
        Err(BlockError::Unsupported(_))
    ));
}

#[test]
fn link_up_starts_the_mac_at_the_resolved_speed_and_down_resets_it() {
    let mut fixture = Fixture::new();
    let mut adapter = fixture.attach(0x1073).unwrap();
    assert!(!sample(&mut adapter));
    let resets = fixture.model.borrow().soft_resets;
    assert_eq!(resets, 2, "no reset while the link stays down");
    fixture.link(Some(GIGABIT_FULL));
    assert!(sample(&mut adapter));
    assert_eq!(adapter.speed(), Some(1000));
    {
        let model = fixture.model.borrow();
        let mac = model.get(MAC_CTRL);
        for bit in [
            MAC_CTRL_RX_EN,
            MAC_CTRL_TX_EN,
            MAC_CTRL_DUPLX,
            MAC_CTRL_ADD_CRC,
            MAC_CTRL_PAD,
            MAC_CTRL_BC_EN,
        ] {
            assert_ne!(mac & bit, 0, "{bit:#x}");
        }
        assert_eq!(field(mac, MAC_CTRL_SPEED), MAC_CTRL_SPEED_1000);
        assert_eq!(field(mac, MAC_CTRL_PRMLEN), 7);
        assert_ne!(model.get(RXQ_CTRL) & RXQ_CTRL_EN, 0);
        assert_ne!(model.get(TXQ_CTRL) & TXQ_CTRL_EN, 0);
        // L1D with a link: L0s and L1 on, serdes off in L1.
        let power = model.get(PM_CTRL);
        assert_ne!(power & PM_CTRL_ASPM_L0S_EN, 0);
        assert_ne!(power & PM_CTRL_ASPM_L1_EN, 0);
        assert_eq!(
            field(power, PM_CTRL_L1_ENTRY_TIMER),
            L1C_PM_CTRL_L1_ENTRY_TM
        );
    }
    adapter.transmit(&[0x11; 60]).unwrap();
    fixture.link(None);
    assert!(!sample(&mut adapter));
    {
        let model = fixture.model.borrow();
        assert_eq!(model.soft_resets, resets + 1);
        assert_eq!(model.get(MAC_CTRL) & (MAC_CTRL_RX_EN | MAC_CTRL_TX_EN), 0);
        assert_eq!(model.get(PM_CTRL) & PM_CTRL_ASPM_L0S_EN, 0);
        assert_eq!(model.get(MB_RFD0_PROD_IDX) as usize, RX_RING - 1);
    }
    assert_eq!(adapter.queues.tx_write, 0);
    assert_eq!(adapter.transmit(&[0x22; 60]), Err(LinkError::NoCarrier));
    fixture.link(Some(GIGA_PSSR_SPD_DPLX_RESOLVED | GIGA_PSSR_100MBS));
    assert!(sample(&mut adapter));
    let mac = fixture.model.borrow().get(MAC_CTRL);
    assert_eq!(mac & MAC_CTRL_DUPLX, 0);
    assert_eq!(field(mac, MAC_CTRL_SPEED), MAC_CTRL_SPEED_10_100);
}

#[test]
fn listed_platforms_adjust_the_phy_thresholds_at_100_mbps() {
    let mut fixture = Fixture::new();
    let mut chip = Fixture::chip(0x1073);
    chip.link_patch = true;
    let mut adapter = fixture.attach_chip(chip).unwrap();
    assert!(
        fixture
            .model
            .borrow()
            .phy_writes
            .contains(&(Phy::Debug(MIIDBG_MSE16DB), L1D_MSE16DB_DOWN))
    );
    fixture.link(Some(FAST_FULL));
    assert!(sample(&mut adapter));
    let model = fixture.model.borrow();
    assert!(
        model
            .phy_writes
            .contains(&(Phy::Debug(MIIDBG_MSE16DB), L1D_MSE16DB_UP))
    );
    assert!(
        model
            .phy_writes
            .contains(&(Phy::Debug(MIIDBG_SYSMODCTRL), L1D_SYSMODCTRL_IECHOADJ_DEF))
    );
}

#[test]
fn an_unresolved_phy_speed_fails_the_link() {
    let mut fixture = Fixture::new();
    let mut adapter = fixture.attach(0x1063).unwrap();
    fixture.link(Some(GIGA_PSSR_1000MBS));
    assert!(!sample(&mut adapter));
    assert!(matches!(adapter.receive(), Err(LinkError::Adapter(_))));
}

#[test]
fn transmit_fills_one_descriptor_per_frame_and_pads_short_frames() {
    let mut fixture = Fixture::new();
    let mut adapter = fixture.attach(0x1063).unwrap();
    fixture.link(Some(GIGABIT_FULL));
    let frame: Vec<u8> = (0..20u8).collect();
    adapter.transmit(&frame).unwrap();
    adapter.transmit(&[0x5a; MAX_FRAME]).unwrap();
    let model = fixture.model.borrow();
    assert_eq!(model.sent.len(), 2);
    assert_eq!(&model.sent[0][..20], &frame[..]);
    assert_eq!(model.sent[0].len(), MIN_FRAME);
    assert!(model.sent[0][20..].iter().all(|&byte| byte == 0));
    assert_eq!(model.sent[1], vec![0x5a; MAX_FRAME]);
    assert_eq!(model.get(TPD_PRI0_PIDX), 2);
    drop(model);
    assert_eq!(
        adapter.transmit(&[0; MAX_FRAME + 1]),
        Err(LinkError::Invalid("frame exceeds the Ethernet maximum"))
    );
}

#[test]
fn a_full_ring_waits_for_the_consumer_and_then_reports_a_stall() {
    let mut fixture = Fixture::new();
    let mut adapter = fixture.attach(0x1063).unwrap();
    fixture.link(Some(GIGABIT_FULL));
    fixture.model.borrow_mut().consume_transmit = false;
    for _ in 0..TX_RING - 1 {
        adapter.transmit(&[1; 60]).unwrap();
    }
    assert!(matches!(
        adapter.transmit(&[1; 60]),
        Err(LinkError::Adapter(_))
    ));
    fixture
        .model
        .borrow_mut()
        .registers
        .insert(TPD_PRI0_CIDX, 1);
    adapter.transmit(&[2; 60]).unwrap();
}

#[test]
fn receive_returns_good_frames_without_fcs_and_drops_errors() {
    let mut fixture = Fixture::new();
    let mut adapter = fixture.attach(0x1063).unwrap();
    fixture.link(Some(GIGABIT_FULL));
    assert_eq!(adapter.receive(), Ok(None));
    let frame: Vec<u8> = (0..64u8).collect();
    {
        let mut model = fixture.model.borrow_mut();
        assert!(model.deliver(&[0xee; 64], RRS_RX_ERR_SUM, None, 1));
        assert!(model.deliver(&[0xdd; 64], RRS_802_3_LEN_ERR, None, 1));
        assert!(model.deliver(&frame, 0, None, 1));
    }
    assert_eq!(adapter.receive(), Ok(Some(frame)));
    assert_eq!(adapter.receive(), Ok(None));
    assert_eq!(adapter.queues.rx_read, 3);
    assert_eq!(fixture.model.borrow().get(MB_RFD0_PROD_IDX), 2);
}

#[test]
fn receive_keeps_up_across_the_ring_wrap_into_the_reused_buffer() {
    let mut fixture = Fixture::new();
    let mut adapter = fixture.attach(0x1063).unwrap();
    fixture.link(Some(GIGABIT_FULL));
    let mut reused = Vec::with_capacity(MAX_FRAME);
    let storage = reused.as_ptr();
    for round in 0..(RX_RING * 2) {
        let frame = vec![round as u8; 100];
        assert!(fixture.model.borrow_mut().deliver(&frame, 0, None, 1));
        if round % 2 == 0 {
            assert_eq!(adapter.receive(), Ok(Some(frame)));
        } else {
            assert_eq!(adapter.receive_reusing(&mut reused), Ok(true));
            assert_eq!(reused, frame);
            assert_eq!(reused.as_ptr(), storage);
        }
    }
    assert_eq!(adapter.receive_reusing(&mut reused), Ok(false));
    adapter.stop().unwrap();
    assert!(adapter.receive_reusing(&mut reused).is_err());
    assert_eq!(reused, vec![(RX_RING * 2 - 1) as u8; 100]);
}

#[test]
fn a_frame_over_several_free_descriptors_is_dropped_and_its_descriptors_returned() {
    let mut fixture = Fixture::new();
    let mut adapter = fixture.attach(0x1063).unwrap();
    fixture.link(Some(GIGABIT_FULL));
    {
        let mut model = fixture.model.borrow_mut();
        assert!(model.deliver(&[0x33; 64], 0, Some(0), 2));
        // The second return descriptor of that frame.
        assert!(model.deliver(&[0x33; 64], 0, Some(1), 0));
    }
    let frame = vec![0x44; 80];
    assert!(fixture.model.borrow_mut().deliver(&frame, 0, None, 1));
    assert_eq!(adapter.receive(), Ok(Some(frame)));
    assert_eq!(adapter.queues.rx_read, 3);
}

#[test]
fn a_return_descriptor_naming_no_free_descriptor_reinitializes_the_queues() {
    let mut fixture = Fixture::new();
    let mut adapter = fixture.attach(0x1063).unwrap();
    fixture.link(Some(GIGABIT_FULL));
    assert!(sample(&mut adapter));
    let resets = fixture.model.borrow().soft_resets;
    assert!(
        fixture
            .model
            .borrow_mut()
            .deliver(&[1; 64], 0, Some(RX_RING + 4), 1)
    );
    assert_eq!(adapter.receive(), Ok(None));
    assert_eq!(fixture.model.borrow().soft_resets, resets + 1);
    assert_eq!(adapter.queues.rx_read, 0);
    assert!(adapter.carrier());
    assert_ne!(fixture.model.borrow().get(MAC_CTRL) & MAC_CTRL_RX_EN, 0);
    let frame = vec![9; 64];
    assert!(fixture.model.borrow_mut().deliver(&frame, 0, None, 1));
    assert_eq!(adapter.receive(), Ok(Some(frame)));
}

#[test]
fn stop_masks_interrupts_resets_the_mac_and_refuses_further_frames() {
    let mut fixture = Fixture::new();
    let mut adapter = fixture.attach(0x1063).unwrap();
    fixture.link(Some(GIGABIT_FULL));
    assert!(sample(&mut adapter));
    let resets = fixture.model.borrow().soft_resets;
    adapter.stop().unwrap();
    {
        let model = fixture.model.borrow();
        assert_eq!(model.soft_resets, resets + 1);
        assert_eq!(model.get(IMR), 0);
        assert_eq!(model.get(ISR), ISR_DIS_INT);
        assert_eq!(model.get(MAC_CTRL) & (MAC_CTRL_RX_EN | MAC_CTRL_TX_EN), 0);
        assert_eq!(model.get(RXQ_CTRL) & RXQ_CTRL_EN, 0);
        assert_eq!(
            model.get(PM_CTRL) & (PM_CTRL_ASPM_L0S_EN | PM_CTRL_ASPM_L1_EN),
            0
        );
    }
    assert!(!adapter.carrier());
    assert!(matches!(
        adapter.transmit(&[0; 60]),
        Err(LinkError::Adapter(_))
    ));
    assert!(matches!(adapter.receive(), Err(LinkError::Adapter(_))));
}
