// SPDX-License-Identifier: GPL-2.0-only
// Copyright (c) 2026 Stela contributors
//! Stateful register/firmware peer, separate from the driver's algorithms.
use super::*;
use crate::bus::{dma::RegionAllocator, pci::Address};
use alloc::{collections::BTreeMap, rc::Rc};
use core::cell::{Cell, RefCell};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Fault {
    None,
    Master,
    Reset,
    Firmware,
    Semaphore,
    Mdio,
    MdioTimeout,
    MdioRegister,
    Eeprom,
    QueueEnable,
    QueueDisable,
    PhyReset,
    NeverPll,
    D3,
    D0,
}

struct Model {
    registers: RefCell<BTreeMap<usize, u32>>,
    phy: RefCell<BTreeMap<(u16, u16), u16>>,
    eeprom: RefCell<Vec<u16>>,
    writes: RefCell<Vec<(usize, u32)>>,
    mdio_writes: RefCell<Vec<(u16, u16, u16)>>,
    nvm_reads: RefCell<Vec<u16>>,
    page: Cell<u16>,
    phy_id: u32,
    fault: Cell<Fault>,
    power: Cell<u16>,
    cycles: Cell<usize>,
    lock_after: Cell<usize>,
}

impl Model {
    const MAC: Mac = Mac([2, 0x11, 0x22, 0x33, 0x44, 0x55]);
    fn new(chip: Chip, phy_id: u32, flash: bool) -> Rc<Self> {
        let model = Rc::new(Self {
            registers: RefCell::new(BTreeMap::new()),
            phy: RefCell::new(BTreeMap::new()),
            eeprom: RefCell::new(vec![0; 512]),
            writes: RefCell::new(Vec::new()),
            mdio_writes: RefCell::new(Vec::new()),
            nvm_reads: RefCell::new(Vec::new()),
            page: Cell::new(0),
            phy_id,
            fault: Cell::new(Fault::None),
            power: Cell::new(0),
            cycles: Cell::new(0),
            lock_after: Cell::new(0),
        });
        model.set(
            EECD,
            EECD_PRESENT
                | EECD_AUTO_RD
                | (3 << 11)
                | if flash && chip.modern() {
                    EECD_FLASH
                } else {
                    0
                },
        );
        model.set(EEMNGCTL, 0xf << 18);
        model.set(EEMNGCTL_I210, 1 << 18);
        model.set(RAL, 0x3322_1102);
        model.set(RAH, (1 << 31) | 0x5544);
        model.set(MDICNFG, 0x0020_0010);
        model.set(WUC, 0x1234);
        model
            .phy
            .borrow_mut()
            .insert((0, PHY_ID1), (phy_id >> 16) as u16);
        model
            .phy
            .borrow_mut()
            .insert((0, PHY_ID2), phy_id as u16 | 1);
        model
            .phy
            .borrow_mut()
            .insert((0, PHY_CONTROL), PHY_POWERDOWN | PHY_ISOLATE | PHY_LOOPBACK);
        model.phy.borrow_mut().insert((0, PHY_GIGABIT), 0x1f00);
        model.set(INVM, (0x1102 << 16) | 1);
        model.set(INVM + 4, (0x3322 << 16) | (1 << 9) | 1);
        model.set(INVM + 8, (0x5544 << 16) | (2 << 9) | 1);
        model.set(INVM + 12, (0x202f << 16) | (0x0a << 9) | 1);
        model.set(INVM + 16, 0);
        model.fix_checksums();
        model
    }
    fn set(&self, register: usize, value: u32) {
        self.registers.borrow_mut().insert(register, value);
    }
    fn get(&self, register: usize) -> u32 {
        self.registers.borrow().get(&register).copied().unwrap_or(0)
    }
    fn fix_checksums(&self) {
        let mut words = self.eeprom.borrow_mut();
        for base in [0, 0x80, 0xc0, 0x100] {
            let checksum = words[base..base + 0x3f]
                .iter()
                .fold(0u16, |sum, &word| sum.wrapping_add(word));
            words[base + 0x3f] = NVM_SUM.wrapping_sub(checksum);
        }
    }
    fn phy_value(&self, register: u16) -> u16 {
        self.phy.borrow().get(&(0, register)).copied().unwrap_or(0)
    }
    fn assert_released(&self) {
        assert_eq!(self.get(SW_FW_SYNC) & 0xffff, 0);
        assert_eq!(self.get(SWSM), 0);
    }
}

impl RegisterIo for Model {
    fn read(&self, offset: usize) -> u32 {
        assert_eq!(self.power.get(), 0, "MMIO read while in PCI D3");
        if offset == SWSM {
            if self.fault.get() == Fault::Semaphore {
                return 1;
            }
            let value = self.get(offset);
            self.set(offset, value | 1); // read-to-set SMBI latch
            return value;
        }
        if offset == SW_FW_SYNC && self.fault.get() == Fault::Firmware {
            return 0xffff_0000;
        }
        if offset == STATUS && self.fault.get() == Fault::Master {
            return self.get(offset) | STATUS_MASTER;
        }
        self.get(offset)
    }

    fn write(&self, offset: usize, mut value: u32) {
        assert_eq!(self.power.get(), 0, "MMIO write while in PCI D3");
        self.writes.borrow_mut().push((offset, value));
        if offset == CTRL && value & CTRL_RST != 0 && self.fault.get() != Fault::Reset {
            value &= !(CTRL_RST | CTRL_MASTER_DISABLE);
            self.set(RXDCTL, QUEUE_ENABLE);
            self.set(TXDCTL, QUEUE_ENABLE);
        }
        if offset == STATUS {
            value = self.get(STATUS) & !value;
        }
        if (offset == RXDCTL || offset == TXDCTL) && self.fault.get() == Fault::QueueEnable {
            value &= !QUEUE_ENABLE;
        }
        if offset == RXDCTL && self.fault.get() == Fault::QueueDisable {
            value |= QUEUE_ENABLE;
        }
        if offset == EERD {
            let word = ((value >> 2) & 0x3fff) as u16;
            self.nvm_reads.borrow_mut().push(word);
            if self.fault.get() != Fault::Eeprom {
                let payload = self
                    .eeprom
                    .borrow()
                    .get(usize::from(word))
                    .copied()
                    .unwrap_or(u16::MAX);
                value = (u32::from(payload) << 16) | (u32::from(word) << 2) | 2;
            }
        }
        if offset == MDIC && self.fault.get() != Fault::MdioTimeout {
            let register = ((value >> 16) & 31) as u16;
            if self.fault.get() == Fault::Mdio || value & (31 << 21) != MDIC_PHY {
                value |= MDIC_ERROR;
            } else if value & MDIC_READ != 0 {
                let page = if register < 16 { 0 } else { self.page.get() };
                let payload = if self.page.get() == 0xfc && register == 14 {
                    if self.fault.get() == Fault::NeverPll
                        || self.cycles.get() < self.lock_after.get()
                    {
                        0xff
                    } else {
                        0x12
                    }
                } else {
                    self.phy
                        .borrow()
                        .get(&(page, register))
                        .copied()
                        .unwrap_or(0)
                };
                value = (value & !0xffff) | u32::from(payload);
            } else {
                let mut data = value as u16;
                let page = self.page.get();
                self.mdio_writes.borrow_mut().push((page, register, data));
                if (register == PHY_PAGE_IGP && self.phy_id == PHY_IGP3)
                    || (register == PHY_PAGE_M88
                        && matches!(self.phy_id, PHY_M88 | PHY_I347 | PHY_I210))
                {
                    self.page.set(data);
                }
                if register == PHY_CONTROL {
                    data &= !PHY_RESTART;
                    if self.fault.get() != Fault::PhyReset {
                        data &= !PHY_RESET;
                    }
                }
                let page = if register < 16 { 0 } else { page };
                self.phy.borrow_mut().insert((page, register), data);
            }
            value |= MDIC_READY;
            if self.fault.get() == Fault::MdioRegister {
                value ^= 1 << 16;
            }
        }
        self.set(offset, value);
    }
}

#[derive(Default)]
struct ModelClock {
    delays: RefCell<Vec<u64>>,
}
impl Timer for ModelClock {
    fn delay(&self, microseconds: u64) -> Result<(), BlockError> {
        self.delays.borrow_mut().push(microseconds);
        Ok(())
    }
    fn wait(&self, _: u64, mut ready: impl FnMut() -> bool) -> bool {
        (0..8).any(|_| ready())
    }
}

struct Config {
    words: [u32; 128],
    io: Rc<Model>,
    writes: Vec<(u16, u16)>,
}
impl Config {
    fn new(io: &Rc<Model>) -> Self {
        let mut words = [0; 128];
        words[0] = 0x10c9_8086;
        words[1] = 1 << 20;
        words[2] = 0x0200_0000;
        words[13] = 0x40;
        words[0x40 / 4] = 0x0002_8010; // PCIe v2, next capability at 0x80
        words[0x80 / 4] = 1;
        words[0x84 / 4] = 1 << 15; // PM capability; PME_Status latched
        Self {
            words,
            io: Rc::clone(io),
            writes: Vec::new(),
        }
    }
}
impl ConfigSpace for Config {
    fn read8(&self, offset: u16) -> core::result::Result<u8, crate::bus::pci::ConfigError> {
        Ok((self.words[usize::from(offset / 4)] >> ((offset & 3) * 8)) as u8)
    }
    fn read16(&self, offset: u16) -> core::result::Result<u16, crate::bus::pci::ConfigError> {
        Ok((self.words[usize::from(offset / 4)] >> ((offset & 3) * 8)) as u16)
    }
    fn read32(&self, offset: u16) -> core::result::Result<u32, crate::bus::pci::ConfigError> {
        Ok(self.words[usize::from(offset / 4)] >> ((offset & 3) * 8))
    }
    fn write8(&mut self, _: u16, _: u8) -> core::result::Result<(), crate::bus::pci::ConfigError> {
        Err(crate::bus::pci::ConfigError::before_effect(
            crate::bus::pci::ConfigFailure::UnqualifiedControl,
        ))
    }
    fn write32(
        &mut self,
        _: u16,
        _: u32,
    ) -> core::result::Result<(), crate::bus::pci::ConfigError> {
        Err(crate::bus::pci::ConfigError::before_effect(
            crate::bus::pci::ConfigFailure::UnqualifiedControl,
        ))
    }
    fn write16(
        &mut self,
        offset: u16,
        mut value: u16,
    ) -> core::result::Result<(), crate::bus::pci::ConfigError> {
        self.writes.push((offset, value));
        if offset == 0x84 {
            let old = self.read16(offset)?;
            if self.io.fault.get() == Fault::D3 && value & 3 == 3 {
                value &= !3;
            }
            if self.io.fault.get() == Fault::D0 && old & 3 == 3 && value & 3 == 0 {
                value |= 3;
            }
            if old & 3 == 3 && value & 3 == 0 {
                self.io.cycles.set(self.io.cycles.get() + 1);
                // D3/D0 releases the PHY reset through the NVM autoload.
                self.io.set(CTRL, self.io.get(CTRL) & !CTRL_PHY_RST);
            }
            self.io.power.set(value & 3);
            value = (value & !(1 << 15)) | (old & (1 << 15) & !value);
        }
        let shift = (offset & 2) * 8;
        let word = &mut self.words[usize::from(offset / 4)];
        *word = (*word & !(0xffff << shift)) | u32::from(value) << shift;

        Ok(())
    }
}

fn initialize(io: &Rc<Model>, chip: Chip) -> Result<(Mac, InitializationWarnings), BlockError> {
    let clock = ModelClock::default();
    let mut config = Config::new(io);
    let capabilities = PciCapabilities {
        pcie: Some(0x40),
        power: Some(0x80),
    };
    Hardware::new(io.as_ref(), &clock, chip).initialize(&mut config, capabilities)
}

#[test]
fn every_supported_copper_phy_branch_initializes_without_publishing_dma() {
    for (chip, phy, flash) in [
        (Chip::I82575, PHY_IGP3, true),
        (Chip::I82576, PHY_IGP3, true),
        (Chip::I82576, PHY_M88, true),
        (Chip::I82576, PHY_I347, true),
        (Chip::I82580, PHY_82580, true),
        (Chip::I350, PHY_I350, true),
        (Chip::I210, PHY_I210, true),
        (Chip::I210, PHY_I210, false),
        (Chip::I211, PHY_I210, false),
    ] {
        let io = Model::new(chip, phy, flash);
        let (mac, warnings) = initialize(&io, chip).unwrap();
        assert_eq!(mac, Model::MAC, "{chip:?}");
        assert_eq!(warnings, InitializationWarnings::default());
        assert_eq!(io.phy_value(PHY_ADVERTISE), 0x01e1);
        assert_eq!(io.phy_value(PHY_GIGABIT) & 0x1f00, 0x0200);
        assert_eq!(
            io.phy_value(PHY_CONTROL) & (PHY_POWERDOWN | PHY_ISOLATE | PHY_LOOPBACK | PHY_RESET),
            0
        );
        assert_eq!(io.phy_value(PHY_CONTROL) & PHY_AUTONEG, PHY_AUTONEG);
        assert!(
            !io.writes
                .borrow()
                .iter()
                .any(|&(offset, _)| matches!(offset, RDBAL | RDBAH | TDBAL | TDBAH))
        );
        io.assert_released();
        match phy {
            PHY_IGP3 => assert_eq!(io.phy_value(18) & 0x1000, 0x1000),
            PHY_M88 => assert_eq!(io.phy_value(20) & 0x70, 0x70),
            PHY_I347 | PHY_I210 => assert_eq!(io.phy_value(PHY_SPECIFIC) & 0x7862, 0x5860),
            PHY_82580 | PHY_I350 => {
                assert_eq!(io.phy_value(22) & 0x8c00, 0x8c00);
                assert_eq!(io.phy_value(18) & 0x0600, 0x0400);
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn reset_stops_queues_without_device_reset() {
    for chip in [
        Chip::I82575,
        Chip::I82576,
        Chip::I82580,
        Chip::I350,
        Chip::I210,
        Chip::I211,
    ] {
        let phy = match chip {
            Chip::I82575 | Chip::I82576 => PHY_IGP3,
            Chip::I82580 => PHY_82580,
            Chip::I350 => PHY_I350,
            _ => PHY_I210,
        };
        let io = Model::new(chip, phy, chip != Chip::I211);
        initialize(&io, chip).unwrap();
        let writes = io.writes.borrow();
        assert!(
            !writes
                .iter()
                .any(|&(offset, value)| offset == CTRL && value & (1 << 29) != 0)
        );
        assert_eq!(
            writes.iter().any(|&(offset, _)| offset == STATUS),
            !chip.old_reset()
        );
        for queue in 0..chip.queues() {
            assert!(writes.contains(&(queue_register(RXDCTL, queue), 0)));
            assert!(writes.contains(&(queue_register(TXDCTL, queue), 0)));
        }
    }
}

#[test]
fn missing_firmware_completion_bits_are_visible_after_real_nvm_and_phy_validation() {
    let io = Model::new(Chip::I82576, PHY_IGP3, true);
    io.set(EECD, io.get(EECD) & !EECD_AUTO_RD);
    io.set(EEMNGCTL, 0);
    let (mac, warnings) = initialize(&io, Chip::I82576).unwrap();
    assert_eq!(mac, Model::MAC);
    assert_eq!(
        warnings,
        InitializationWarnings {
            nvm_autoload_timeout: true,
            phy_configuration_timeout: true
        }
    );
    assert!(io.nvm_reads.borrow().contains(&NVM_CHECKSUM));
    assert_eq!(io.phy_value(PHY_ADVERTISE), 0x01e1);
}

#[test]
fn master_reset_and_queue_disable_timeouts_stop_before_nvm_or_phy_access() {
    for fault in [Fault::Master, Fault::Reset, Fault::QueueDisable] {
        let io = Model::new(Chip::I82576, PHY_IGP3, true);
        io.fault.set(fault);
        assert_eq!(initialize(&io, Chip::I82576), Err(BlockError::Timeout));
        assert!(io.nvm_reads.borrow().is_empty());
        assert!(io.mdio_writes.borrow().is_empty());
    }
}

#[test]
fn lock_and_mdio_failures_are_bounded() {
    for fault in [
        Fault::Firmware,
        Fault::Semaphore,
        Fault::Mdio,
        Fault::MdioTimeout,
        Fault::MdioRegister,
    ] {
        let io = Model::new(Chip::I82576, PHY_IGP3, true);
        io.fault.set(fault);
        assert!(initialize(&io, Chip::I82576).is_err());
        if fault != Fault::Semaphore {
            io.assert_released();
        }
        assert_eq!(io.phy_value(PHY_ADVERTISE), 0);
    }
}

#[test]
fn i210_recovers_a_stale_smbi_latch_only_once() {
    let io = Model::new(Chip::I210, PHY_I210, true);
    io.set(SWSM, 1);
    let clock = ModelClock::default();
    let mut hardware = Hardware::new(io.as_ref(), &clock, Chip::I210);
    assert!(hardware.read_phy(PHY_ID1).is_ok());
    io.assert_released();
    io.set(SWSM, 1);
    assert_eq!(hardware.read_phy(PHY_ID1), Err(BlockError::Timeout));
    assert_eq!(io.get(SWSM), 1);
}

#[test]
fn unknown_phy_media_and_firmware_owned_reset_do_not_start_a_link() {
    let unknown = Model::new(Chip::I82576, 0x1234_5670, true);
    assert!(initialize(&unknown, Chip::I82576).is_err());
    assert_eq!(unknown.phy_value(PHY_ADVERTISE), 0);
    let firmware_owned = Model::new(Chip::I82576, PHY_IGP3, true);
    firmware_owned.set(MANC, MANC_BLOCK_PHY);
    assert!(initialize(&firmware_owned, Chip::I82576).is_err());
    assert!(firmware_owned.mdio_writes.borrow().is_empty());
    let media = Model::new(Chip::I82576, PHY_IGP3, true);
    media.set(CTRL_EXT, CTRL_EXT_MODE);
    assert!(initialize(&media, Chip::I82576).is_err());
    assert!(media.nvm_reads.borrow().is_empty());
    let mismatch = Model::new(Chip::I350, PHY_82580, true);
    assert!(initialize(&mismatch, Chip::I350).is_err());
}

#[test]
fn all_i350_checksum_regions_and_82580_compatibility_mode_are_checked() {
    for chip in [Chip::I350, Chip::I82580] {
        let phy = if chip == Chip::I350 {
            PHY_I350
        } else {
            PHY_82580
        };
        let io = Model::new(chip, phy, true);
        io.eeprom.borrow_mut()[3] = 0x8000;
        io.fix_checksums();
        io.eeprom.borrow_mut()[0xc0] ^= 1;
        assert!(initialize(&io, chip).is_err());
        assert_eq!(io.phy_value(PHY_ADVERTISE), 0);
        io.assert_released();
    }
    let old_mode = Model::new(Chip::I82580, PHY_82580, true);
    old_mode.eeprom.borrow_mut()[0xc0] ^= 1;
    initialize(&old_mode, Chip::I82580).unwrap();
    assert!(!old_mode.nvm_reads.borrow().contains(&0xc0));
}

#[test]
fn nvm_failures_and_unprogrammed_addresses_are_explicit() {
    let bad_checksum = Model::new(Chip::I82576, PHY_IGP3, true);
    bad_checksum.eeprom.borrow_mut()[0] ^= 1;
    assert!(initialize(&bad_checksum, Chip::I82576).is_err());
    let stalled = Model::new(Chip::I82576, PHY_IGP3, true);
    stalled.fault.set(Fault::Eeprom);
    assert_eq!(initialize(&stalled, Chip::I82576), Err(BlockError::Timeout));
    stalled.assert_released();
    let absent = Model::new(Chip::I82575, PHY_IGP3, true);
    absent.set(EECD, EECD_AUTO_RD);
    assert!(initialize(&absent, Chip::I82575).is_err());
    let no_mac = Model::new(Chip::I82576, PHY_IGP3, true);
    no_mac.set(RAL, 0);
    no_mac.set(RAH, 1 << 31);
    assert!(initialize(&no_mac, Chip::I82576).is_err());
    let no_invm = Model::new(Chip::I211, PHY_I210, false);
    no_invm.set(INVM, 0);
    assert!(initialize(&no_invm, Chip::I211).is_err());
    assert!(no_invm.mdio_writes.borrow().is_empty());
    let mut pointer = Model::new(Chip::I82576, PHY_IGP3, true);
    Rc::get_mut(&mut pointer).unwrap().eeprom.borrow_mut()[0x37] = 0x4000;
    pointer.fix_checksums();
    assert!(initialize(&pointer, Chip::I82576).is_err());
}

#[test]
fn alternate_mac_uses_the_lan_function_and_nvm_pointer() {
    let io = Model::new(Chip::I82576, PHY_IGP3, true);
    io.set(STATUS, 1 << 2);
    io.eeprom.borrow_mut()[0x37] = 0x80;
    io.eeprom.borrow_mut()[0x83..0x86].copy_from_slice(&[0x7702, 0x8899, 0xaabb]);
    io.fix_checksums();
    let (mac, _) = initialize(&io, Chip::I82576).unwrap();
    assert_eq!(mac, Mac([2, 0x77, 0x99, 0x88, 0xbb, 0xaa]));
    assert!(io.nvm_reads.borrow().contains(&0x83));
    assert!(
        io.writes
            .borrow()
            .iter()
            .any(|&(offset, value)| offset == SW_FW_SYNC && value == 4)
    );
}

#[test]
fn invm_parser_stops_at_uninitialized_record() {
    let io = Model::new(Chip::I211, PHY_I210, false);
    for index in 0..INVM_WORDS {
        io.set(INVM + 4 * index, 0);
    }
    io.set(INVM, 2);
    io.set(INVM + 4, (0x1111 << 16) | (7 << 9) | 1); // payload is not an autoload record
    io.set(INVM + 8, 4);
    for index in 3..11 {
        io.set(INVM + 4 * index, (0x2222 << 16) | (7 << 9) | 1);
    }
    io.set(INVM + 44, (0x3333 << 16) | (7 << 9) | 1);
    let clock = ModelClock::default();
    let hardware = Hardware::new(io.as_ref(), &clock, Chip::I211);
    assert_eq!(hardware.invm_word(7), Some(0x3333));
    assert_eq!(hardware.invm_word(8), None);
    io.set(INVM + 44, 0);
    io.set(INVM + 48, (0x4444 << 16) | (7 << 9) | 1);
    assert_eq!(hardware.invm_word(7), None);
}

#[test]
fn flashless_pll_recovery_cycles_d3_and_restores_state() {
    let io = Model::new(Chip::I211, PHY_I210, false);
    io.lock_after.set(1);
    let clock = ModelClock::default();
    let mut config = Config::new(&io);
    let old_mdio = io.get(MDICNFG);
    let old_wake = io.get(WUC);
    let old_control = io.get(CTRL);
    let old_extended = io.get(CTRL_EXT);
    Hardware::new(io.as_ref(), &clock, Chip::I211)
        .pll_workaround(
            &mut config,
            PciCapabilities {
                pcie: Some(0x40),
                power: Some(0x80),
            },
        )
        .unwrap();
    assert_eq!(io.cycles.get(), 1);
    assert_eq!(io.power.get(), 0);
    assert_eq!(io.page.get(), 0);
    assert_eq!(io.get(MDICNFG), old_mdio);
    assert_eq!(io.get(WUC), old_wake);
    assert_eq!(io.get(CTRL), old_control);
    assert_eq!(io.get(CTRL_EXT), old_extended);
    assert_ne!(config.read16(0x84).unwrap() & (1 << 15), 0);
    assert!(
        config
            .writes
            .iter()
            .all(|&(offset, value)| offset != 0x84 || value & (1 << 15) == 0)
    );
    assert!(
        io.writes
            .borrow()
            .contains(&(EEARBC, (0x0a << 4) | (0x203f << 16)))
    );
    io.assert_released();
}

#[test]
fn pll_lock_failure_restores_state_and_a_refused_d0_stops_mmio_access() {
    for fault in [Fault::NeverPll, Fault::D3, Fault::D0] {
        let io = Model::new(Chip::I211, PHY_I210, false);
        io.lock_after.set(1);
        io.fault.set(fault);
        let clock = ModelClock::default();
        let mut config = Config::new(&io);
        assert!(
            Hardware::new(io.as_ref(), &clock, Chip::I211)
                .pll_workaround(
                    &mut config,
                    PciCapabilities {
                        pcie: Some(0x40),
                        power: Some(0x80)
                    }
                )
                .is_err()
        );
        if fault != Fault::D0 {
            assert_eq!(io.power.get(), 0);
            assert_eq!(io.page.get(), 0);
            assert_eq!(io.get(WUC), 0x1234);
        } else {
            assert_eq!(io.power.get(), 3);
        }
    }
}

#[test]
fn reset_programs_old_pcie_v2_completion_timeout_with_a_halfword_store() {
    let io = Model::new(Chip::I82576, PHY_IGP3, true);
    io.set(GCR, GCR_V2 | GCR_RESEND);
    let clock = ModelClock::default();
    let mut config = Config::new(&io);
    let caps = PciCapabilities {
        pcie: Some(0x40),
        power: Some(0x80),
    };
    Hardware::new(io.as_ref(), &clock, Chip::I82576)
        .initialize(&mut config, caps)
        .unwrap();
    assert!(config.writes.contains(&(0x68, 5)));
    assert_eq!(io.get(GCR) & GCR_RESEND, 0);
}

#[test]
fn mac_filter_capacity_and_i82575_management_fifo_flush_are_chip_specific() {
    for chip in [Chip::I82575, Chip::I82576, Chip::I350, Chip::I210] {
        let phy = if chip.old_reset() {
            PHY_IGP3
        } else if chip == Chip::I350 {
            PHY_I350
        } else {
            PHY_I210
        };
        let io = Model::new(chip, phy, true);
        io.set(MANC, MANC_TCO);
        initialize(&io, chip).unwrap();
        let writes = io.writes.borrow();
        assert_eq!(
            writes
                .iter()
                .any(|&(offset, _)| offset == receive_address(23) + 4),
            matches!(chip, Chip::I82576 | Chip::I350)
        );
        assert_eq!(
            writes
                .iter()
                .any(|&(offset, _)| offset == receive_address(31) + 4),
            chip == Chip::I350
        );
        assert_eq!(
            writes.contains(&(RCTL, RCTL_LPE | RCTL_EN)),
            chip == Chip::I82575
        );
    }
}

/// An empty receive unmasks the vector, auto-masked when it fired.
#[test]
fn empty_receive_unmasks_the_signal_vector() {
    let mut fixture = DmaFixture::new();
    fixture.net.signalled = true;
    assert!(fixture.net.signals());
    assert_eq!(fixture.net.receive(), Ok(None));
    assert_eq!(fixture.registers[EIMS / 4], 1 << VECTOR);
}

struct DmaFixture {
    net: Igb,
    registers: Vec<u32>,
    _memory: Vec<u64>,
}

impl DmaFixture {
    fn new() -> Self {
        let mut registers = vec![0; 0x2_0000 / 4];
        let mut memory = vec![u64::MAX; 200_000];
        let physical = memory.as_mut_ptr() as u64;
        // SAFETY: the fixture keeps these DMA pages alive beyond the driver.
        let mut dma = unsafe { RegionAllocator::new(physical..physical + 1_600_000) };
        let mut net = Igb {
            // SAFETY: this fixture exclusively owns the complete register backing.
            device: unsafe {
                Mmio::new(registers.as_mut_ptr() as u64, size_of_val(&registers[..]))
            },
            receive: Ring::new(&mut dma).unwrap(),
            transmit: Ring::new(&mut dma).unwrap(),
            dropping: false,
            faulted: false,
            mac: Model::MAC,
            clock: crate::timing::model(Clock {
                ticks_per_microsecond: 1,
            }),
            location: String::from("DMA peer"),
            warnings: InitializationWarnings::default(),
            signalled: false,
        };
        start_queues(&net.device, &net.clock, &mut net.receive, &net.transmit).unwrap();
        registers[STATUS / 4] = STATUS_LU;
        Self {
            net,
            registers,
            _memory: memory,
        }
    }

    fn deliver(&mut self, index: usize, frame: &[u8], status: u32, length: u16) {
        self.net.receive.buffer(index)[..frame.len()].copy_from_slice(frame);
        self.net
            .receive
            .descriptors
            .write(index * DESCRIPTOR + 12, length);
        // DMA writes payload and metadata before publishing DD.
        self.net
            .receive
            .descriptors
            .write(index * DESCRIPTOR + 8, status);
    }
}

#[test]
fn advanced_dma_setup_preserves_full_addresses_and_unambiguous_ring_ownership() {
    let fixture = DmaFixture::new();
    let base = |low: usize, high: usize| {
        u64::from(fixture.registers[low / 4]) | u64::from(fixture.registers[high / 4]) << 32
    };
    assert_eq!(
        base(RDBAL, RDBAH),
        fixture.net.receive.descriptors.physical()
    );
    assert_eq!(
        base(TDBAL, TDBAH),
        fixture.net.transmit.descriptors.physical()
    );
    assert_eq!(
        fixture.registers[RDLEN / 4],
        (DESCRIPTORS * DESCRIPTOR) as u32
    );
    assert_eq!(
        fixture.registers[TDLEN / 4],
        (DESCRIPTORS * DESCRIPTOR) as u32
    );
    assert_eq!(fixture.registers[SRRCTL / 4], 0x0200_0002);
    assert_eq!(fixture.registers[RDT / 4], (DESCRIPTORS - 1) as u32);
    assert_eq!(fixture.registers[TDT / 4], 0);
    for index in 0..DESCRIPTORS {
        assert_eq!(
            fixture
                .net
                .receive
                .descriptors
                .read::<u64>(index * DESCRIPTOR),
            fixture.net.receive.address(index)
        );
        assert_eq!(
            fixture
                .net
                .receive
                .descriptors
                .read::<u64>(index * DESCRIPTOR + 8),
            0
        );
    }
}

#[test]
fn failed_queue_enable_never_rings_the_receive_doorbell() {
    let mut fixture = DmaFixture::new();
    let io = Model::new(Chip::I82576, PHY_IGP3, true);
    io.fault.set(Fault::QueueEnable);
    assert_eq!(
        start_queues(
            io.as_ref(),
            &ModelClock::default(),
            &mut fixture.net.receive,
            &fixture.net.transmit
        ),
        Err(BlockError::Timeout)
    );
    assert!(
        !io.writes
            .borrow()
            .contains(&(RDT, (DESCRIPTORS - 1) as u32))
    );
    assert_eq!(io.get(RXDCTL) & QUEUE_ENABLE, 0);
    assert_eq!(io.get(TXDCTL) & QUEUE_ENABLE, 0);
    assert_eq!(io.get(RCTL), 0);
    assert_eq!(io.get(TCTL) & TCTL_EN, 0);
}

#[test]
fn link_loss_discards_queued_transmit_work_and_absent_functions_report_no_link() {
    let mut fixture = DmaFixture::new();
    for _ in 0..3 {
        fixture.net.transmit(&[1; 60]).unwrap();
    }
    fixture.registers[STATUS / 4] = 0;
    assert!(!fixture.net.carrier());
    assert!(!fixture.net.transmit.pending());
    assert_eq!(fixture.registers[TDT / 4], 0);
    assert_ne!(fixture.registers[TXDCTL / 4] & QUEUE_ENABLE, 0);
    assert_eq!(fixture.net.transmit(&[1; 60]), Err(LinkError::NoCarrier));
    fixture.registers[STATUS / 4] = u32::MAX;
    assert!(!fixture.net.carrier());
    assert!(matches!(fixture.net.receive(), Err(LinkError::Adapter(_))));
}

#[test]
fn full_transmit_ring_refuses_overwrite_and_recovers_only_completed_slots() {
    let mut fixture = DmaFixture::new();
    for _ in 0..DESCRIPTORS - 1 {
        fixture.net.transmit(&[7; 60]).unwrap();
    }
    let descriptors = fixture.net.transmit.descriptors.bytes().to_vec();
    let buffers: Vec<Vec<u8>> = (0..DESCRIPTORS)
        .map(|index| fixture.net.transmit.buffer(index).to_vec())
        .collect();
    assert!(fixture.net.transmit(&[8; 60]).is_err());
    assert_eq!(fixture.net.transmit.descriptors.bytes(), descriptors);
    for (index, buffer) in buffers.iter().enumerate() {
        assert_eq!(fixture.net.transmit.buffer(index), &buffer[..]);
    }
    fixture.net.transmit.descriptors.write(12, 1u32);
    fixture.net.transmit(&[9; 60]).unwrap();
    assert_eq!(fixture.net.transmit.next, 0);
    assert_eq!(fixture.net.transmit.clean, 1);
    assert!(!fixture.net.transmit.reclaim(transmitted));
}

#[test]
fn seeded_ethernet_payloads_survive_both_rings_for_many_wraps() {
    let mut fixture = DmaFixture::new();
    let mut seed = 0x9e37_79b9u32;
    for sequence in 0..DESCRIPTORS * 17 {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        let length = 14 + seed as usize % (MAX_FRAME - 13);
        let frame: Vec<_> = (0..length)
            .map(|offset| ((seed >> (offset % 4 * 8)) as u8).wrapping_add(offset as u8))
            .collect();
        let index = sequence % DESCRIPTORS;
        fixture.net.transmit(&frame).unwrap();
        let padded = length.max(MIN_FRAME);
        assert_eq!(
            fixture
                .net
                .transmit
                .descriptors
                .read::<u64>(index * DESCRIPTOR),
            fixture.net.transmit.address(index)
        );
        assert_eq!(
            fixture
                .net
                .transmit
                .descriptors
                .read::<u32>(index * DESCRIPTOR + 8),
            0x2b30_0000 | padded as u32
        );
        assert_eq!(
            fixture
                .net
                .transmit
                .descriptors
                .read::<u32>(index * DESCRIPTOR + 12),
            (padded as u32) << 14
        );
        assert_eq!(&fixture.net.transmit.buffer(index)[..length], frame);
        assert!(
            fixture.net.transmit.buffer(index)[length..padded]
                .iter()
                .all(|&byte| byte == 0)
        );
        fixture
            .net
            .transmit
            .descriptors
            .write(index * DESCRIPTOR + 12, 1u32);
        fixture.deliver(index, &frame, 3, length as u16);
        assert_eq!(fixture.net.receive().unwrap(), Some(frame));
        assert_eq!(
            fixture
                .net
                .receive
                .descriptors
                .read::<u64>(index * DESCRIPTOR + 8),
            0
        );
        if (sequence + 1) % 16 == 0 {
            assert_eq!(fixture.registers[RDT / 4], index as u32);
        }
    }
    assert_eq!(fixture.net.transmit.next, 0);
    assert_eq!(fixture.net.receive.next, 0);
}

#[test]
fn receive_discards_whole_fragment_chains_crc_errors_and_out_of_bounds_lengths() {
    let mut fixture = DmaFixture::new();
    fixture.deliver(0, &[1; 60], 1, 60); // fragment without EOP
    fixture.deliver(1, &[2; 60], 3, 60); // last fragment is not a standalone frame
    fixture.deliver(2, &[3; 60], 0x0100_0003, 60); // FCS error
    fixture.deliver(3, &[4; 60], 3, 4096); // metadata must not produce an out-of-bounds slice
    fixture.deliver(4, &[5; 13], 3, 13);
    fixture.deliver(5, &[6; 60], 3, 60);
    // The stack's reused buffer takes only the good frame, in place.
    let mut reused = Vec::with_capacity(MAX_FRAME);
    let storage = reused.as_ptr();
    assert_eq!(fixture.net.receive_reusing(&mut reused), Ok(true));
    assert_eq!(reused, vec![6; 60]);
    assert_eq!(reused.as_ptr(), storage);
    assert_eq!(fixture.net.receive.next, 6);
    assert_eq!(fixture.registers[RDT / 4], (DESCRIPTORS - 1) as u32);
    assert_eq!(fixture.net.receive().unwrap(), None);
    assert_eq!(fixture.registers[RDT / 4], 5);
}

#[test]
fn checksum_flags_do_not_substitute_for_a_software_checksum_when_offload_is_disabled() {
    let mut fixture = DmaFixture::new();
    fixture.deliver(0, &[7; 60], 0x6000_0003, 60);
    assert_eq!(fixture.net.receive().unwrap(), Some(vec![7; 60]));
    let tail = fixture.net.transmit.next;
    for invalid in [vec![], vec![0; 13], vec![0; MAX_FRAME + 1]] {
        assert!(matches!(
            fixture.net.transmit(&invalid),
            Err(LinkError::Invalid(_))
        ));
        assert_eq!(fixture.net.transmit.next, tail);
    }
}

#[test]
fn an_uncompleted_phy_software_reset_fails_before_autonegotiation() {
    let io = Model::new(Chip::I210, PHY_I210, true);
    io.fault.set(Fault::PhyReset);
    assert_eq!(initialize(&io, Chip::I210), Err(BlockError::Timeout));
    assert_eq!(io.phy_value(PHY_ADVERTISE), 0);
    io.assert_released();
}

#[test]
fn device_matching_is_exact_and_does_not_accept_fiber_sgmii_vfs_or_igc() {
    let io = Model::new(Chip::I82576, PHY_IGP3, true);
    for id in [
        0x10a7, 0x10d6, 0x10c9, 0x10e8, 0x1526, 0x150a, 0x150e, 0x1516, 0x1521, 0x1533, 0x157b,
        0x1539,
    ] {
        let mut config = Config::new(&io);
        config.words[0] = u32::from(id as u16) << 16 | 0x8086;
        let function = Function::new(
            Address {
                segment: 0,
                bus: 0,
                device: 1,
                function: 0,
            },
            config,
        )
        .unwrap();
        assert!(matches(&function), "{id:04x}");
    }
    for (vendor, id, class) in [
        (0x8086, 0x10e6, 0x0200_0000),
        (0x8086, 0x1511, 0x0200_0000),
        (0x8086, 0x10ca, 0x0200_0000),
        (0x8086, 0x125c, 0x0200_0000),
        (0x1234, 0x10c9, 0x0200_0000),
        (0x8086, 0x10c9, 0x0100_0000),
    ] {
        let mut config = Config::new(&io);
        config.words[0] = (id << 16) | vendor;
        config.words[2] = class;
        let function = Function::new(
            Address {
                segment: 0,
                bus: 0,
                device: 1,
                function: 0,
            },
            config,
        )
        .unwrap();
        assert!(!matches(&function));
    }
}
