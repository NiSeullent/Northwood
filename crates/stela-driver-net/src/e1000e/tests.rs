use super::*;
use crate::bus::{
    dma::RegionAllocator,
    pci::{Address, ConfigSpace},
};
use alloc::collections::BTreeMap;
use core::cell::RefCell;

#[derive(Default)]
struct Model {
    registers: RefCell<BTreeMap<usize, u32>>,
    writes: RefCell<Vec<(usize, u32)>>,
    phy: RefCell<[u16; 32]>,
    ownership_blocked: bool,
    master_stuck: bool,
    reset_stuck: bool,
    mdic_timeout: bool,
    mdic_error: bool,
    wrong_register: bool,
}

impl Model {
    fn ready() -> Self {
        let model = Self::default();
        model.registers.borrow_mut().insert(EECD, EECD_AUTO_RD);
        model.phy.borrow_mut()[usize::from(PHY_ID1)] = (PHY_BM_REV2 >> 16) as u16;
        model.phy.borrow_mut()[usize::from(PHY_ID2)] = PHY_BM_REV2 as u16;
        model
    }
}

impl Registers for Model {
    fn read(&self, offset: usize) -> u32 {
        self.registers.borrow().get(&offset).copied().unwrap_or(0)
    }

    fn write(&self, offset: usize, mut value: u32) {
        self.writes.borrow_mut().push((offset, value));
        if offset == EXTCNF_CTRL && self.ownership_blocked {
            value &= !MDIO_OWN;
        }
        if offset == CTRL {
            if value & CTRL_RST != 0 && !self.reset_stuck {
                value &= !CTRL_RST;
            }
            if value & CTRL_MASTER_DISABLE != 0 && !self.master_stuck {
                self.registers.borrow_mut().insert(STATUS, 0);
            }
        }
        if offset == MDIC && !self.mdic_timeout {
            let register = ((value & MDIC_REGISTER) >> 16) as usize;
            if self.mdic_error {
                value |= MDIC_ERROR;
            } else if value & MDIC_READ != 0 {
                value |= u32::from(self.phy.borrow()[register]);
            } else {
                let data = value as u16;
                self.phy.borrow_mut()[register] = if register == usize::from(PHY_CONTROL) {
                    data & !(PHY_RESET | PHY_RESTART)
                } else {
                    data
                };
            }
            if self.wrong_register {
                value ^= 1 << 16;
            }
            value |= MDIC_READY;
        }
        self.registers.borrow_mut().insert(offset, value);
    }
}

#[derive(Default)]
struct ModelTimer {
    delays: RefCell<Vec<u64>>,
}

impl Timer for ModelTimer {
    fn delay(&self, microseconds: u64) -> Result<(), BlockError> {
        self.delays.borrow_mut().push(microseconds);
        Ok(())
    }

    fn wait(&self, _: u64, mut ready: impl FnMut() -> bool) -> bool {
        (0..4).any(|_| ready())
    }
}

#[test]
fn reset_drains_dma_reloads_nvm_and_releases_mdio() {
    let model = Model::ready();
    model.registers.borrow_mut().insert(STATUS, STATUS_MASTER);
    model.registers.borrow_mut().insert(TCTL, TCTL_EN);
    let timer = ModelTimer::default();
    assert_eq!(reset(&model, &timer), Ok(()));
    assert_eq!(*timer.delays.borrow(), [10_000, 25_000]);
    assert_eq!(model.read(EXTCNF_CTRL) & MDIO_OWN, 0);
    assert_eq!(model.read(RCTL), 0);
    assert_eq!(model.read(TCTL) & TCTL_EN, 0);
    let writes = model.writes.borrow();
    let own = writes
        .iter()
        .position(|&(at, value)| at == EXTCNF_CTRL && value & MDIO_OWN != 0)
        .unwrap();
    let rst = writes
        .iter()
        .position(|&(at, value)| at == CTRL && value & CTRL_RST != 0)
        .unwrap();
    assert!(own < rst);
}

#[test]
fn a_bus_master_that_will_not_drain_is_not_reset() {
    let model = Model {
        master_stuck: true,
        ..Model::ready()
    };
    model.registers.borrow_mut().insert(STATUS, STATUS_MASTER);
    assert_eq!(
        reset(&model, &ModelTimer::default()),
        Err(BlockError::Timeout)
    );
    assert!(
        !model
            .writes
            .borrow()
            .iter()
            .any(|&(at, value)| at == CTRL && value & CTRL_RST != 0)
    );
}

#[test]
fn ownership_and_nvm_failures_do_not_look_successful() {
    let ownership = Model {
        ownership_blocked: true,
        ..Model::ready()
    };
    assert_eq!(
        reset(&ownership, &ModelTimer::default()),
        Err(BlockError::Timeout)
    );
    assert!(
        !ownership
            .writes
            .borrow()
            .iter()
            .any(|&(at, value)| at == CTRL && value & CTRL_RST != 0)
    );
    let nvm = Model::default();
    assert_eq!(
        reset(&nvm, &ModelTimer::default()),
        Err(BlockError::Timeout)
    );
    assert_eq!(nvm.read(EXTCNF_CTRL) & MDIO_OWN, 0);
    let stuck = Model {
        reset_stuck: true,
        ..Model::ready()
    };
    assert_eq!(
        reset(&stuck, &ModelTimer::default()),
        Err(BlockError::Timeout)
    );
    assert_eq!(stuck.read(EXTCNF_CTRL) & MDIO_OWN, 0);
}

#[test]
fn phy_setup_programs_autonegotiation_and_releases_the_firmware_lock() {
    let model = Model::ready();
    model.phy.borrow_mut()[usize::from(PHY_CONTROL)] = PHY_POWERDOWN | PHY_LOOPBACK | PHY_ISOLATE;
    assert_eq!(configure_phy(&model, &ModelTimer::default()), Ok(()));
    assert_eq!(model.read(EXTCNF_CTRL) & MDIO_OWN, 0);
    let phy = model.phy.borrow();
    assert_eq!(phy[usize::from(PHY_ADVERTISE)], 0x01e1);
    assert_eq!(phy[usize::from(PHY_GIGABIT)], 0x0200);
    assert_eq!(phy[usize::from(PHY_CONTROL)], PHY_AUTONEG);
    assert_eq!(
        phy[usize::from(PHY_SPECIFIC)] & (PHY_AUTO_MDIX | PHY_DOWNSHIFT),
        PHY_AUTO_MDIX | PHY_DOWNSHIFT
    );
    assert_eq!(phy[usize::from(PHY_BIAS1)], 3);
}

#[test]
fn mdio_timeout_error_wrong_response_and_unknown_phy_are_explicit() {
    for model in [
        Model {
            mdic_timeout: true,
            ..Model::ready()
        },
        Model {
            mdic_error: true,
            ..Model::ready()
        },
        Model {
            wrong_register: true,
            ..Model::ready()
        },
        Model::default(),
    ] {
        assert!(configure_phy(&model, &ModelTimer::default()).is_err());
        assert_eq!(model.read(EXTCNF_CTRL) & MDIO_OWN, 0);
    }
}

/// An empty receive enables the receive cause again, auto-cleared when
/// the vector fired.
#[test]
fn empty_receive_enables_the_receive_cause_again() {
    let mut fixture = Fixture::new();
    fixture.net.signalled = true;
    assert!(fixture.net.signals());
    assert_eq!(fixture.net.receive(), Ok(None));
    assert_eq!(fixture.registers[IMS / 4], IMS_RXQ0);
}

struct Fixture {
    net: E1000e,
    registers: Vec<u32>,
    _memory: Vec<u64>,
}

impl Fixture {
    fn new() -> Self {
        let mut registers = vec![0u32; REGISTER_BYTES as usize / 4];
        let mut memory = vec![u64::MAX; 200_000];
        let start = memory.as_mut_ptr() as u64;
        // SAFETY: the fixture owns the memory for the driver's whole lifetime.
        let mut allocator = unsafe { RegionAllocator::new(start..start + 1_600_000) };
        // SAFETY: this aligned backing covers every register used by the driver.
        let device =
            unsafe { Mmio::new(registers.as_mut_ptr() as u64, size_of_val(&registers[..])) };
        let mut net = E1000e {
            device,
            receive: Ring::new(&mut allocator).unwrap(),
            transmit: Ring::new(&mut allocator).unwrap(),
            dropping_receive: false,
            mac: Mac([2, 0, 0, 0, 0, 1]),
            clock: crate::timing::model(Clock {
                ticks_per_microsecond: 1,
            }),
            watchdog_at: u64::MAX,
            phy_hangs: 0,
            faulted: false,
            timing_failure: None,
            location: String::from("model"),
            signalled: false,
        };
        net.start();
        registers[STATUS / 4] = STATUS_LU;
        Self {
            net,
            registers,
            _memory: memory,
        }
    }

    fn inject(&mut self, index: usize, length: u16, status: u8, errors: u8, byte: u8) {
        self.net.receive.buffer(index).fill(byte);
        self.net
            .receive
            .descriptors
            .write(index * DESCRIPTOR + 8, length);
        self.net
            .receive
            .descriptors
            .write(index * DESCRIPTOR + 13, errors);
        // Hardware makes DD visible after the data and metadata writes.
        self.net
            .receive
            .descriptors
            .write(index * DESCRIPTOR + 12, status);
    }
}

#[test]
fn dma_rings_use_legacy_descriptors_and_a_single_empty_slot() {
    let mut fixture = Fixture::new();
    assert_eq!(
        fixture.registers[RDLEN / 4],
        (DESCRIPTORS * DESCRIPTOR) as u32
    );
    assert_eq!(
        fixture.registers[TDLEN / 4],
        (DESCRIPTORS * DESCRIPTOR) as u32
    );
    assert_eq!(fixture.registers[RDT / 4], (DESCRIPTORS - 1) as u32);
    assert_eq!(fixture.registers[RFCTL / 4] & RFCTL_EXTEN, 0);
    for _ in 0..DESCRIPTORS - 1 {
        fixture.net.transmit(&[7; 60]).unwrap();
    }
    let before = fixture.net.transmit.descriptors.bytes().to_vec();
    assert!(fixture.net.transmit(&[8; 60]).is_err());
    assert_eq!(fixture.net.transmit.descriptors.bytes(), before);
    assert_eq!(fixture.net.transmit.next, DESCRIPTORS - 1);
    // Completion of the oldest descriptor permits only that buffer to be reused.
    fixture.net.transmit.descriptors.write(12, TX_DONE);
    fixture.net.transmit(&[9; 60]).unwrap();
    assert_eq!(fixture.net.transmit.next, 0);
    assert_eq!(fixture.net.transmit.clean, 1);
    assert!(!fixture.net.transmit.reclaim(transmitted));
}

#[test]
fn transmit_padding_descriptors_and_wrap_are_real_dma_state() {
    let mut fixture = Fixture::new();
    for index in 0..DESCRIPTORS * 3 {
        let descriptor = index % DESCRIPTORS;
        fixture.net.transmit(&[0xa5; 14]).unwrap();
        assert_eq!(&fixture.net.transmit.buffer(descriptor)[..14], &[0xa5; 14]);
        assert_eq!(&fixture.net.transmit.buffer(descriptor)[14..60], &[0; 46]);
        assert_eq!(
            fixture
                .net
                .transmit
                .descriptors
                .read::<u32>(descriptor * DESCRIPTOR + 8),
            60 | TX_EOP | TX_IFCS | TX_REPORT
        );
        fixture
            .net
            .transmit
            .descriptors
            .write(descriptor * DESCRIPTOR + 12, TX_DONE);
    }
    assert_eq!(fixture.net.transmit.next, 0);
    assert_eq!(fixture.registers[TDT / 4], 0);
    let before = fixture.net.transmit.next;
    for bad in [vec![], vec![0; 13], vec![0; MAX_FRAME + 1]] {
        assert!(matches!(
            fixture.net.transmit(&bad),
            Err(LinkError::Invalid(_))
        ));
        assert_eq!(fixture.net.transmit.next, before);
    }
}

#[test]
fn receive_drops_errors_oversize_and_entire_fragment_chains() {
    let mut fixture = Fixture::new();
    fixture.inject(0, 60, RX_DONE, 0, 1); // first fragment
    fixture.inject(1, 60, RX_DONE | RX_EOP, 0, 2); // not a standalone packet
    fixture.inject(2, 60, RX_DONE | RX_EOP, 1, 3); // bad FCS
    fixture.inject(3, 4096, RX_DONE | RX_EOP, 0, 4); // don't slice out of bounds
    fixture.inject(4, 60, RX_DONE | RX_EOP, 0, 5);
    // The stack's reused buffer takes only the good frame, in place.
    let mut reused = Vec::with_capacity(MAX_FRAME);
    let storage = reused.as_ptr();
    assert_eq!(fixture.net.receive_reusing(&mut reused), Ok(true));
    assert_eq!(reused, vec![5; 60]);
    assert_eq!(reused.as_ptr(), storage);
    assert_eq!(fixture.net.receive.next, 5);
    // The tail waits for a batch, or for the ring to run dry.
    assert_eq!(fixture.registers[RDT / 4], (DESCRIPTORS - 1) as u32);
    assert_eq!(
        fixture
            .net
            .receive
            .descriptors
            .read::<u64>(4 * DESCRIPTOR + 8),
        0
    );
    assert_eq!(fixture.net.receive().unwrap(), None);
    assert_eq!(fixture.registers[RDT / 4], 4);
}

#[test]
fn receive_wraps_and_reoffers_every_descriptor_before_the_doorbell() {
    let mut fixture = Fixture::new();
    for count in 0..DESCRIPTORS * 3 {
        let index = count % DESCRIPTORS;
        fixture.inject(index, 60, RX_DONE | RX_EOP, 0, count as u8);
        assert_eq!(fixture.net.receive().unwrap(), Some(vec![count as u8; 60]));
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
        // Linux's E1000_RX_BUFFER_WRITE: one tail write per sixteen.
        if (count + 1) % 16 == 0 {
            assert_eq!(fixture.registers[RDT / 4], index as u32);
        }
    }
    assert_eq!(fixture.net.receive.next, 0);
}

#[test]
fn link_loss_discards_queued_transmit_work_and_refuses_new_frames() {
    let mut fixture = Fixture::new();
    for _ in 0..3 {
        fixture.net.transmit(&[1; 60]).unwrap();
    }
    assert_eq!(fixture.registers[TDT / 4], 3);
    fixture.registers[STATUS / 4] = 0;
    assert!(!fixture.net.carrier());
    assert!(!fixture.net.transmit.pending());
    assert_eq!(fixture.registers[TDT / 4], 0);
    assert!(fixture.registers[TCTL / 4] & TCTL_EN != 0);
    assert_eq!(fixture.net.transmit(&[1; 60]), Err(LinkError::NoCarrier));
    fixture.registers[STATUS / 4] = STATUS_LU;
    fixture.net.transmit(&[2; 60]).unwrap();
    assert_eq!(fixture.registers[TDT / 4], 1);
}

#[test]
fn an_absent_function_is_never_reported_as_linked() {
    let mut fixture = Fixture::new();
    fixture.registers[STATUS / 4] = u32::MAX;
    assert!(!fixture.net.carrier());
    assert!(matches!(
        fixture.net.transmit(&[1; 60]),
        Err(LinkError::Adapter(_))
    ));
    fixture.registers[STATUS / 4] = STATUS_LU;
    assert!(!fixture.net.carrier());
}

#[test]
fn a_phy_hangs_only_when_both_error_counters_saturate() {
    for (errors, idle, hung) in [
        (u16::MAX, PHY_IDLE_ERRORS, true),
        (u16::MAX, 0x7f, false),
        (0xfffe, PHY_IDLE_ERRORS, false),
    ] {
        let model = Model::ready();
        model.phy.borrow_mut()[usize::from(PHY_RECEIVE_ERRORS)] = errors;
        model.phy.borrow_mut()[usize::from(PHY_GIGABIT_STATUS)] = idle;
        assert_eq!(phy_hung(&model, &ModelTimer::default()), hung);
        assert_eq!(model.read(EXTCNF_CTRL) & MDIO_OWN, 0);
    }
}

#[test]
fn station_addresses_reject_invalid_zero_and_multicast_nvm() {
    assert_eq!(
        station(0x5634_1202, RAH_VALID | 0xbc9a),
        Some(Mac([2, 0x12, 0x34, 0x56, 0x9a, 0xbc]))
    );
    assert_eq!(station(0x5634_1202, 0xbc9a), None);
    assert_eq!(station(0, RAH_VALID), None);
    assert_eq!(station(u32::MAX, u32::MAX), None);
}

struct Config(u32, u32);

impl ConfigSpace for Config {
    fn read8(&self, offset: u16) -> Result<u8, crate::bus::pci::ConfigError> {
        Ok(((match offset & !3 {
            0 => self.0,
            8 => self.1,
            _ => 0,
        }) >> ((offset & 3) * 8)) as u8)
    }
    fn read16(&self, offset: u16) -> Result<u16, crate::bus::pci::ConfigError> {
        Ok(((match offset & !3 {
            0 => self.0,
            8 => self.1,
            _ => 0,
        }) >> ((offset & 3) * 8)) as u16)
    }
    fn read32(&self, offset: u16) -> Result<u32, crate::bus::pci::ConfigError> {
        Ok((match offset & !3 {
            0 => self.0,
            8 => self.1,
            _ => 0,
        }) >> ((offset & 3) * 8))
    }
    fn write8(&mut self, _: u16, _: u8) -> Result<(), crate::bus::pci::ConfigError> {
        Err(crate::bus::pci::ConfigError::before_effect(
            crate::bus::pci::ConfigFailure::UnqualifiedControl,
        ))
    }
    fn write16(&mut self, _: u16, _: u16) -> Result<(), crate::bus::pci::ConfigError> {
        Err(crate::bus::pci::ConfigError::before_effect(
            crate::bus::pci::ConfigFailure::UnqualifiedControl,
        ))
    }
    fn write32(&mut self, _: u16, _: u32) -> Result<(), crate::bus::pci::ConfigError> {
        Err(crate::bus::pci::ConfigError::before_effect(
            crate::bus::pci::ConfigFailure::UnqualifiedControl,
        ))
    }
}

#[test]
fn match_is_restricted_to_82574l_and_ethernet_class() {
    for (id, class, expected) in [
        (0x10d3_8086, 0x0200_0000, true),
        (0x10d3_8086, 0, false),
        (0x15b8_8086, 0x0200_0000, false),
        (0x10d3_1234, 0x0200_0000, false),
    ] {
        let function = crate::bus::pci::Function::new(
            Address {
                segment: 0,
                bus: 0,
                device: 1,
                function: 0,
            },
            Config(id, class),
        )
        .unwrap();
        assert_eq!(matches(&function), expected);
    }
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
    assert_eq!(
        fixture.net.reinitialize(),
        Err(LinkError::Pci(expected.clone()))
    );
    // Completed RX and an IRQ-capable path must also remain fenced.
    fixture.inject(0, 60, RX_DONE | RX_EOP, 0, 7);
    fixture.net.signalled = true;
    let receive = fixture.net.receive.descriptors.bytes().to_vec();
    let observed = fixture.registers.clone();
    for _ in 0..3 {
        assert!(!fixture.net.carrier());
        assert_eq!(
            fixture.net.transmit(&[1; 60]),
            Err(LinkError::Pci(expected.clone()))
        );
        assert_eq!(fixture.net.receive(), Err(LinkError::Pci(expected.clone())));
    }
    assert_eq!(fixture.net.receive.descriptors.bytes(), receive);
    assert_eq!(fixture.registers, observed);
}
