extern crate std;
use super::super::owned::test_support::{Counters, fixture};
use super::*;
use crate::bus::pci::regs;
use alloc::sync::Arc;
use core::{cell::Cell, sync::atomic::Ordering};

const ADDRESS: Address = Address {
    segment: 0xfffe,
    bus: 0xe7,
    device: 29,
    function: 6,
};

struct Config {
    words: [u32; 1024],
    last_read: Cell<u16>,
    /// COMMAND bits that ignore writes, as hardwired bits do.
    command_fixed: u16,
}
impl ConfigSpace for Config {
    fn read32(&self, offset: u16) -> Result<u32, crate::bus::pci::ConfigError> {
        Ok({
            self.last_read.set(offset);
            self.words[usize::from(offset / 4)]
        })
    }
    fn write32(&mut self, offset: u16, value: u32) -> Result<(), crate::bus::pci::ConfigError> {
        let index = usize::from(offset / 4);
        let fixed = if index == usize::from(regs::COMMAND / 4) {
            u32::from(self.command_fixed)
        } else {
            0
        };
        self.words[index] = value & !fixed | self.words[index] & fixed;
        Ok(())
    }

    fn read8(&self, offset: u16) -> Result<u8, crate::bus::pci::ConfigError> {
        Ok((self.read32(offset & !3)? >> ((offset & 3) * 8)) as u8)
    }

    fn write8(&mut self, offset: u16, value: u8) -> Result<(), crate::bus::pci::ConfigError> {
        let shift = (offset & 3) * 8;
        let word = self.read32(offset & !3)? & !(0xff << shift) | u32::from(value) << shift;
        self.write32(offset & !3, word)
    }

    fn read16(&self, offset: u16) -> Result<u16, crate::bus::pci::ConfigError> {
        Ok((self.read32(offset & !3)? >> ((offset & 2) * 8)) as u16)
    }

    fn write16(&mut self, offset: u16, value: u16) -> Result<(), crate::bus::pci::ConfigError> {
        let shift = (offset & 2) * 8;
        let word = self.read32(offset & !3)? & !(0xffff << shift) | u32::from(value) << shift;
        self.write32(offset & !3, word)
    }
}
fn configure<C: ConfigSpace>(config: &mut C) {
    config.write32(0, 0x2c05_10de).unwrap();
    config.write32(8, 0x0300_0000).unwrap();
    config
        .write16(regs::COMMAND, regs::COMMAND_INTX_DISABLE)
        .unwrap();
    // INTA, as a function that can raise INTx.
    config.write8(regs::INTERRUPT_PIN, 1).unwrap();
}
fn endpoint() -> Function<Config> {
    let mut config = Config {
        words: [0; 1024],
        last_read: Cell::new(0),
        command_fixed: 0,
    };
    configure(&mut config);
    Function::new(ADDRESS, config).unwrap()
}
fn cap(function: &mut Function<Config>, offset: u16, id: u16, next: u16) {
    function
        .config()
        .write16(regs::STATUS, regs::STATUS_CAP_LIST)
        .unwrap();
    function.config().write8(0x34, offset as u8).unwrap();
    function.config().write16(offset, id | next << 8).unwrap();
    if id == PM {
        function.config().write16(offset + 2, 3).unwrap();
    }
}
fn ext(function: &mut Function<Config>, offset: u16, id: u16, version: u32, next: u16) {
    function
        .config()
        .write32(
            offset,
            u32::from(id) | version << 16 | u32::from(next) << 20,
        )
        .unwrap();
}
fn release(function: Function<OwnedConfig>, counters: &Arc<Counters>) {
    assert_eq!(counters.resource_drops.load(Ordering::Relaxed), 0);
    // SAFETY: fixture has no device, IRQ or workers and no config access
    // remains after consuming the original function owner.
    unsafe { super::super::owned::test_support::retire_owned(function.into_config()) }.unwrap();
    assert_eq!(counters.unmaps.load(Ordering::Relaxed), 1);
    assert_eq!(counters.resource_drops.load(Ordering::Relaxed), 1);
    assert_eq!(counters.platform_drops.load(Ordering::Relaxed), 1);
}

#[test]
fn owned_config_binds_full_segment_rid_and_keeps_original_epoch() {
    let (mut config, counters) = fixture(ADDRESS);
    configure(&mut config);
    let function = Function::new(ADDRESS, config).unwrap();
    // SAFETY: isolated RAM fixture has no active engine, alias, VF or IRQ;
    // its real registry owners are held, without native validation.
    let mut claim =
        match unsafe { StoppedRequester::from_stopped_pci(function, FunctionRole::Physical) } {
            Ok(claim) => claim,
            Err((error, _)) => panic!("valid owned endpoint: {error:?}"),
        };
    assert_eq!(claim.address(), ADDRESS);
    assert_eq!(claim.requester(), 0xe7ee);
    assert_eq!(claim.check_stopped(), Ok(()));
    assert_eq!(counters.resource_drops.load(Ordering::Relaxed), 0);
    assert_eq!(counters.platform_drops.load(Ordering::Relaxed), 0);
    release(claim.into_function(), &counters);
}

fn bound_fixture() -> (StoppedRequester, RequesterBinding, Arc<Counters>) {
    let (mut config, counters) = fixture(ADDRESS);
    configure(&mut config);
    let function = Function::new(ADDRESS, config).unwrap();
    // SAFETY: isolated registry-owned RAM; no device/IRQ/worker.
    let claim =
        match unsafe { StoppedRequester::from_stopped_pci(function, FunctionRole::Physical) } {
            Ok(claim) => claim,
            Err((error, _)) => panic!("fixture claim: {error:?}"),
        };
    let (function, binding) = match claim.into_dma_binding() {
        Ok(pair) => pair,
        Err((error, _)) => panic!("fixture binding: {error:?}"),
    };
    (function, binding, counters)
}

#[test]
fn binding_keeps_original_config_and_native_root_until_explicit_release() {
    let (mut claim, binding, counters) = bound_fixture();
    assert_eq!(claim.check_stopped(), Ok(()));
    let (error, claim) = match claim.into_dma_binding() {
        Err(error) => error,
        Ok(_) => panic!("second safe DMA binding accepted"),
    };
    assert_eq!(error, ClaimError::AlreadyBound);
    let mut function = claim.into_function();
    assert_eq!(binding.address(), ADDRESS);
    assert_eq!(binding.requester(), 0xe7ee);
    assert_eq!(binding.check_live(), Ok(()));
    assert_eq!(function.config().read32(0).unwrap(), 0x2c05_10de);
    let mut config = function.into_config();
    assert_eq!(config.matches_binding(&binding), Ok(true));
    // SAFETY: stopped synchronous fixture; Busy must refuse before unmap.
    assert_eq!(unsafe { config.retire() }, Err(MappingError::Busy));
    assert_eq!(counters.unmaps.load(Ordering::Relaxed), 0);
    let (other, other_counters) = fixture(ADDRESS);
    // Same BDF and even the fixture's same numeric root identity do not
    // reproduce the original Permission allocation.
    assert_eq!(other.matches_binding(&binding), Ok(false));
    drop(binding);
    // SAFETY: explicit fixture release follows the final binding.
    unsafe { super::super::owned::test_support::retire_owned(config) }.unwrap();
    assert_eq!(counters.unmaps.load(Ordering::Relaxed), 1);
    assert_eq!(counters.resource_drops.load(Ordering::Relaxed), 1);
    // SAFETY: independent original synchronous fixture mapping.
    unsafe { super::super::owned::test_support::retire_owned(other) }.unwrap();
    assert_eq!(other_counters.unmaps.load(Ordering::Relaxed), 1);
}

#[test]
fn dropping_original_config_permanently_quarantines_the_bound_epoch() {
    let (function, binding, counters) = bound_fixture();
    drop(function);
    assert!(binding.check_live().is_err());
    assert_eq!(counters.unmaps.load(Ordering::Relaxed), 0);
    drop(binding);
    assert_eq!(counters.unmaps.load(Ordering::Relaxed), 0);
    assert_eq!(counters.resource_drops.load(Ordering::Relaxed), 0);
    assert_eq!(counters.platform_drops.load(Ordering::Relaxed), 0);
}

#[test]
fn binding_recheck_refusal_returns_the_unchanged_original_function() {
    let (mut config, counters) = fixture(ADDRESS);
    configure(&mut config);
    let function = Function::new(ADDRESS, config).unwrap();
    // SAFETY: isolated synchronous stopped fixture, as above.
    let mut claim =
        match unsafe { StoppedRequester::from_stopped_pci(function, FunctionRole::Physical) } {
            Ok(claim) => claim,
            Err((error, _)) => panic!("fixture claim: {error:?}"),
        };
    // Private test access models a changed hardware observation.
    claim
        .function
        .config()
        .write16(
            regs::COMMAND,
            regs::COMMAND_INTX_DISABLE | regs::COMMAND_MASTER,
        )
        .unwrap();
    let (error, claim) = match claim.into_dma_binding() {
        Err(error) => error,
        Ok(_) => panic!("active requester accepted"),
    };
    assert_eq!(
        error,
        ClaimError::Active {
            offset: regs::COMMAND
        }
    );
    let mut function = claim.into_function();
    assert_eq!(function.address, ADDRESS);
    assert_eq!(
        function.config().read16(regs::COMMAND).unwrap(),
        regs::COMMAND_INTX_DISABLE | regs::COMMAND_MASTER
    );
    function
        .config()
        .write16(regs::COMMAND, regs::COMMAND_INTX_DISABLE)
        .unwrap();
    release(function, &counters);
}

#[test]
fn rejected_roles_and_mutable_public_metadata_return_the_original_owner() {
    for role in [FunctionRole::Virtual, FunctionRole::Unverified] {
        let (mut config, counters) = fixture(ADDRESS);
        configure(&mut config);
        let function = Function::new(ADDRESS, config).unwrap();
        // SAFETY: fixture is drained; unsupported role makes no grant.
        let (error, function) = match unsafe { StoppedRequester::from_stopped_pci(function, role) }
        {
            Err(rejected) => rejected,
            Ok(_) => panic!("unsupported role admitted"),
        };
        assert_eq!(error, ClaimError::UnsupportedRole(role));
        release(function, &counters);
    }
    let (mut config, counters) = fixture(ADDRESS);
    configure(&mut config);
    let mut function = Function::new(ADDRESS, config).unwrap();
    function.address.bus ^= 1;
    // SAFETY: drained fixture, mismatched metadata must fail.
    let (error, function) =
        match unsafe { StoppedRequester::from_stopped_pci(function, FunctionRole::Physical) } {
            Err(rejected) => rejected,
            Ok(_) => panic!("forged metadata admitted"),
        };
    assert_eq!(error, ClaimError::IdentityChanged);
    release(function, &counters);
}

#[test]
fn device_identity_interrupt_gating_and_header_are_observed_not_assumed() {
    assert_eq!(validate_stopped(&mut endpoint()), Ok(()));
    for (offset, value, expected) in [
        (0, u32::MAX, ClaimError::ReadFailed { offset: 0 }),
        (0, 0x2c03_10de, ClaimError::IdentityChanged),
        (8, u32::MAX, ClaimError::ReadFailed { offset: 8 }),
        (8, 0x0400_0000, ClaimError::IdentityChanged),
        (4, 0x0404, ClaimError::Active { offset: 4 }),
        (4, 0, ClaimError::Active { offset: 4 }),
        (4, 0xffff, ClaimError::ReadFailed { offset: 4 }),
    ] {
        let mut function = endpoint();
        function.config().write32(offset, value).unwrap();
        assert_eq!(validate_stopped(&mut function), Err(expected));
    }
    // A conventional function has no extended space: its first extended
    // header reads all-ones.
    let mut function = endpoint();
    function.config().write32(0x100, u32::MAX).unwrap();
    assert_eq!(validate_stopped(&mut function), Ok(()));
    let mut function = endpoint();
    function.config().write8(0x0e, 1).unwrap();
    assert_eq!(
        validate_stopped(&mut function),
        Err(ClaimError::UnsupportedHeader(1))
    );
    // Without an INTx pin, INTx Disable may stay clear; Bus Master may not.
    let mut function = endpoint();
    function.config().write8(regs::INTERRUPT_PIN, 0).unwrap();
    function.config().write16(regs::COMMAND, 0).unwrap();
    assert_eq!(validate_stopped(&mut function), Ok(()));
    function.config().write16(regs::COMMAND, 4).unwrap();
    assert_eq!(
        validate_stopped(&mut function),
        Err(ClaimError::Active { offset: 4 })
    );
    let mut function = endpoint();
    function.class = [0xff; 3];
    function.config().write32(8, u32::MAX).unwrap();
    assert_eq!(
        validate_stopped(&mut function),
        Err(ClaimError::ReadFailed { offset: 8 })
    );
}

#[test]
fn msi_layouts_pcie_versions_and_no_snoop_require_complete_extents() {
    for (flags, length) in [(0, 10), (0x80, 14), (0x100, 20), (0x180, 24)] {
        let mut function = endpoint();
        cap(&mut function, 0x40, MSI, 0);
        function.config().write16(0x42, flags).unwrap();
        assert_eq!(validate_stopped(&mut function), Ok(()));
        function.config().write16(0x42, flags | 1).unwrap();
        assert_eq!(
            validate_stopped(&mut function),
            Err(ClaimError::Active { offset: 0x42 })
        );
        function.config().write16(0x42, flags).unwrap();
        let collision = 0x40 + ((length - 1) / 4) * 4;
        function
            .config()
            .write16(0x40, MSI | collision << 8)
            .unwrap();
        function.config().write16(collision, PM).unwrap();
        function.config().write16(collision + 2, 3).unwrap();
        assert_eq!(
            validate_stopped(&mut function),
            Err(ClaimError::MalformedCapability { offset: collision })
        );
    }
    for (kind, version, bytes) in [(0, 1, 20), (1, 2, 52), (9, 1, 12), (9, 2, 44)] {
        let mut function = endpoint();
        cap(&mut function, 0x40, PCIE, 0);
        function
            .config()
            .write16(0x42, version | kind << 4)
            .unwrap();
        assert_eq!(validate_stopped(&mut function), Ok(()));
        function.config().write16(0x48, 1 << 11).unwrap();
        assert_eq!(
            validate_stopped(&mut function),
            Err(ClaimError::Active { offset: 0x48 })
        );
        // A full layout one DWORD beyond the conventional region is refused
        // before Device Control could be read from unrelated space.
        let offset = 260 - bytes;
        let mut function = endpoint();
        cap(&mut function, offset, PCIE, 0);
        function
            .config()
            .write16(offset + 2, version | kind << 4)
            .unwrap();
        assert_eq!(
            validate_stopped(&mut function),
            Err(ClaimError::MalformedCapability { offset })
        );
        assert!(function.config().last_read.get() <= offset);
    }
    let mut function = endpoint();
    cap(&mut function, 0x40, MSIX, 0);
    function.config().write16(0x42, 0x8000).unwrap();
    assert_eq!(
        validate_stopped(&mut function),
        Err(ClaimError::Active { offset: 0x42 })
    );
}

#[test]
fn linked_zero_cycles_overlaps_unknowns_and_unmasked_pointers_are_refused() {
    for pointer in [0x3f, 0x41, 0xff] {
        let mut function = endpoint();
        function
            .config()
            .write16(regs::STATUS, regs::STATUS_CAP_LIST)
            .unwrap();
        function.config().write8(0x34, pointer).unwrap();
        assert!(matches!(
            validate_stopped(&mut function),
            Err(ClaimError::MalformedCapability { .. })
        ));
    }
    for (first, second) in [(0x40, 0x44), (0x44, 0x40), (0x40, 0x40)] {
        let mut function = endpoint();
        cap(&mut function, first, PM, second);
        if first != second {
            function.config().write16(second, PM).unwrap();
            function.config().write16(second + 2, 3).unwrap();
        }
        assert!(matches!(
            validate_stopped(&mut function),
            Err(ClaimError::MalformedCapability { .. })
        ));
    }
    let mut unknown = endpoint();
    cap(&mut unknown, 0x40, 0x7f, 0);
    assert_eq!(
        validate_stopped(&mut unknown),
        Err(ClaimError::UnsupportedCapability {
            id: 0x7f,
            extended: false
        })
    );
    let mut unknown = endpoint();
    ext(&mut unknown, 0x100, 0x7fff, 1, 0);
    assert_eq!(
        validate_stopped(&mut unknown),
        Err(ClaimError::UnsupportedCapability {
            id: 0x7fff,
            extended: true
        })
    );
    let mut gpu = endpoint();
    ext(&mut gpu, 0x100, 0, 1, 0x140);
    ext(&mut gpu, 0x140, 0x01, 2, 0x180);
    ext(&mut gpu, 0x180, 0x15, 1, 0x1c0);
    ext(&mut gpu, 0x1c0, 0x10, 1, 0);
    assert_eq!(validate_stopped(&mut gpu), Ok(()));
    gpu.config().write16(0x1c0 + 8, 1).unwrap();
    assert_eq!(
        validate_stopped(&mut gpu),
        Err(ClaimError::Active { offset: 0x1c8 })
    );
    for next in [0x104, 0x101, 0xfff, 0x100, 0x120] {
        let mut function = endpoint();
        ext(&mut function, 0x100, ATS, 1, next);
        assert!(validate_stopped(&mut function).is_err());
    }
}

#[test]
fn pcix_versioned_bodies_cannot_hide_an_overlapping_capability() {
    for version in 0..=2 {
        let mut function = endpoint();
        cap(&mut function, 0x40, PCIX, 0);
        function.config().write16(0x42, version << 12).unwrap();
        assert_eq!(validate_stopped(&mut function), Ok(()));
    }
    let mut function = endpoint();
    cap(&mut function, 0x40, PCIX, 0);
    function.config().write16(0x42, 3 << 12).unwrap();
    assert_eq!(
        validate_stopped(&mut function),
        Err(ClaimError::UnsupportedVersion {
            id: PCIX,
            version: 3
        })
    );
    for version in [1, 2] {
        for reverse in [false, true] {
            let mut function = endpoint();
            cap(&mut function, 0x40, PCIX, if reverse { 0 } else { 0x48 });
            function.config().write16(0x42, version << 12).unwrap();
            function
                .config()
                .write16(0x48, MSI | if reverse { 0x40 << 8 } else { 0 })
                .unwrap();
            if reverse {
                function.config().write8(0x34, 0x48).unwrap();
            }
            assert!(matches!(
                validate_stopped(&mut function),
                Err(ClaimError::MalformedCapability { .. })
            ));
        }
    }
    for version in 0..=7 {
        let mut function = endpoint();
        cap(&mut function, 0x40, PM, 0);
        function.config().write16(0x42, version).unwrap();
        let result = validate_stopped(&mut function);
        if (1..=3).contains(&version) {
            assert_eq!(result, Ok(()));
        } else {
            assert_eq!(
                result,
                Err(ClaimError::UnsupportedVersion {
                    id: PM,
                    version: version as u8
                })
            );
        }
    }
}

#[test]
fn translation_features_require_inactive_controls_and_stopped_pri() {
    for (id, control, active) in [(ATS, 6, 0x8000), (PRI, 4, 1), (PRI, 4, 2), (PASID, 6, 1)] {
        let mut function = endpoint();
        ext(&mut function, 0x100, id, 1, 0);
        if id == PRI {
            function.config().write16(0x106, 0x100).unwrap();
        }
        assert_eq!(validate_stopped(&mut function), Ok(()));
        function.config().write16(0x100 + control, active).unwrap();
        assert_eq!(
            validate_stopped(&mut function),
            Err(ClaimError::Active {
                offset: 0x100 + control
            })
        );
        function
            .config()
            .write16(0x100 + control, u16::MAX)
            .unwrap();
        assert_eq!(
            validate_stopped(&mut function),
            Err(ClaimError::ReadFailed {
                offset: 0x100 + control
            })
        );
    }
    let mut function = endpoint();
    ext(&mut function, 0x100, PRI, 1, 0);
    assert_eq!(
        validate_stopped(&mut function),
        Err(ClaimError::Active { offset: 0x106 })
    );
    for (id, bytes) in [(ATS, 8), (PRI, 16), (PASID, 8)] {
        let mut function = endpoint();
        ext(&mut function, 0x100, ATS, 1, 4100 - bytes);
        ext(&mut function, 4100 - bytes, id, 1, 0);
        assert!(matches!(
            validate_stopped(&mut function),
            Err(ClaimError::MalformedCapability { .. })
        ));
        let mut function = endpoint();
        ext(&mut function, 0x100, id, 2, 0);
        assert_eq!(
            validate_stopped(&mut function),
            Err(ClaimError::UnsupportedVersion { id, version: 2 })
        );
    }
}

#[test]
fn pcie_pending_transactions_refuse_a_stopped_claim_and_bounded_wait() {
    let mut function = endpoint();
    cap(&mut function, 0x40, PCIE, 0);
    function.config().write16(0x42, 2).unwrap();
    function.config().write16(0x4a, 1 << 5).unwrap();
    let mut checks = 0;
    assert_eq!(
        wait_for_pending_transactions(&*function.config(), || {
            checks += 1;
            checks >= 4
        }),
        Err(ClaimError::Active { offset: 0x4a })
    );
    assert_eq!(
        validate_stopped(&mut function),
        Err(ClaimError::Active { offset: 0x4a })
    );
    function.config().write16(0x4a, u16::MAX).unwrap();
    assert_eq!(
        wait_for_pending_transactions(&*function.config(), || true),
        Err(ClaimError::ReadFailed { offset: 0x4a })
    );
    function.config().write16(0x4a, 0).unwrap();
    assert_eq!(validate_stopped(&mut function), Ok(()));
    assert_eq!(
        wait_for_pending_transactions(&*function.config(), || panic!("idle status needs no wait")),
        Ok(())
    );
}

#[test]
fn pcie_pending_wait_observes_completion_before_admitting_the_requester() {
    struct Completing {
        config: Config,
        pending: Cell<u8>,
    }
    impl ConfigSpace for Completing {
        fn read32(&self, offset: u16) -> Result<u32, crate::bus::pci::ConfigError> {
            Ok({
                let word = self.config.read32(offset)?;
                if offset != 0x48 {
                    return Ok(word);
                }
                let pending = self.pending.get();
                self.pending.set(pending.saturating_sub(1));
                if pending != 0 {
                    word | (1 << 21)
                } else {
                    word & !(1 << 21)
                }
            })
        }
        fn write32(&mut self, offset: u16, value: u32) -> Result<(), crate::bus::pci::ConfigError> {
            self.config.write32(offset, value)?;
            Ok(())
        }

        fn read8(&self, offset: u16) -> Result<u8, crate::bus::pci::ConfigError> {
            Ok((self.read32(offset & !3)? >> ((offset & 3) * 8)) as u8)
        }

        fn write8(&mut self, offset: u16, value: u8) -> Result<(), crate::bus::pci::ConfigError> {
            let shift = (offset & 3) * 8;
            let word = self.read32(offset & !3)? & !(0xff << shift) | u32::from(value) << shift;
            self.write32(offset & !3, word)
        }

        fn read16(&self, offset: u16) -> Result<u16, crate::bus::pci::ConfigError> {
            Ok((self.read32(offset & !3)? >> ((offset & 2) * 8)) as u16)
        }

        fn write16(&mut self, offset: u16, value: u16) -> Result<(), crate::bus::pci::ConfigError> {
            let shift = (offset & 2) * 8;
            let word = self.read32(offset & !3)? & !(0xffff << shift) | u32::from(value) << shift;
            self.write32(offset & !3, word)
        }
    }
    let mut function = endpoint();
    cap(&mut function, 0x40, PCIE, 0);
    function.config().write16(0x42, 2).unwrap();
    let mut function = Function::new(
        ADDRESS,
        Completing {
            config: function.into_config(),
            pending: Cell::new(2),
        },
    )
    .unwrap();
    let mut checks = 0;
    assert_eq!(
        wait_for_pending_transactions(&*function.config(), || {
            checks += 1;
            checks >= 4
        }),
        Ok(())
    );
    assert_eq!(validate_stopped(&mut function), Ok(()));
}

fn containment_plan(config: &Config) -> Result<containment::Plan, ClaimError> {
    containment::Plan::read(|offset, width| {
        Ok(match width {
            1 => u32::from(config.read8(offset)?),
            2 => u32::from(config.read16(offset)?),
            4 => config.read32(offset)?,
            _ => panic!("unexpected PCI read width"),
        })
    })
}

#[test]
fn containment_only_writes_the_exact_disabled_command_and_msi_controls() {
    use crate::bus::pci::ConfigEffect;

    let mut function = endpoint();
    cap(&mut function, 0x50, MSI, 0);
    let config = function.config();
    config.write16(regs::COMMAND, 7).unwrap();
    config.write16(0x52, 0x81).unwrap();
    let plan = containment_plan(config).unwrap();
    assert!(plan.permits(4, 2, 0x403));
    assert!(plan.permits(0x52, 2, 0x80));
    assert!(!plan.permits(4, 4, 0x403));
    assert!(!plan.permits(0x52, 2, 0x81));
    assert_eq!(
        plan.apply(config),
        Ok(ConfigEffect::WriteMayHaveApplied {
            offset: 0x52,
            width: 2
        })
    );
    assert_eq!(config.read16(regs::COMMAND), Ok(0x403));
    assert_eq!(config.read16(regs::STATUS), Ok(regs::STATUS_CAP_LIST));
    assert_eq!(config.read16(0x52), Ok(0x80));
}

#[test]
fn containment_readback_refuses_msi_reenabled_after_the_stop() {
    use crate::bus::pci::{ConfigEffect, ConfigFailure};

    let mut function = endpoint();
    cap(&mut function, 0x50, MSI, 0);
    let config = function.config();
    config.write16(regs::COMMAND, 7).unwrap();
    config.write16(0x52, 0x81).unwrap();
    let plan = containment_plan(config).unwrap();
    plan.apply(config).unwrap();
    assert_eq!(plan.check_disabled(|offset| config.read16(offset)), Ok(()));

    config.write16(0x52, 0x81).unwrap();
    let failure = plan
        .check_disabled(|offset| config.read16(offset))
        .unwrap_err();
    assert_eq!(failure.effect, ConfigEffect::None);
    assert!(matches!(
        failure.reason,
        ConfigFailure::Readback {
            offset: 0x52,
            expected: 0x80,
            observed: 0x81,
            ..
        }
    ));
    assert_eq!(config.read16(0x52), Ok(0x81));
}

#[test]
fn containment_refuses_changed_control_before_the_first_write() {
    use crate::bus::pci::{ConfigEffect, ConfigFailure};

    let mut function = endpoint();
    cap(&mut function, 0x50, MSI, 0);
    let config = function.config();
    config.write16(regs::COMMAND, 7).unwrap();
    config.write16(0x52, 1).unwrap();
    let plan = containment_plan(config).unwrap();
    config.write16(0x52, 0x81).unwrap();
    let failure = plan.apply(config).unwrap_err();
    assert_eq!(failure.effect, ConfigEffect::None);
    assert!(matches!(
        failure.reason,
        ConfigFailure::Readback { offset: 0x52, .. }
    ));
    assert_eq!(config.read16(regs::COMMAND), Ok(7));
}

#[test]
fn containment_refuses_to_replay_function_or_pri_reset_commands() {
    use crate::bus::pci::{ConfigError, ConfigFailure};

    let mut function = endpoint();
    cap(&mut function, 0x40, PCIE, 0);
    function.config().write16(0x42, 2).unwrap();
    let refused = ClaimError::Config(ConfigError::before_effect(
        ConfigFailure::UnqualifiedControl,
    ));
    function.config().write16(0x48, 1 << 15).unwrap();
    assert!(matches!(containment_plan(function.config()), Err(error) if error == refused));

    function.config().write16(0x48, 0).unwrap();
    ext(&mut function, 0x100, PRI, 1, 0);
    function.config().write16(0x104, 2).unwrap();
    assert!(matches!(containment_plan(function.config()), Err(error) if error == refused));
}

#[test]
fn containment_accepts_a_hardwired_clear_command_without_an_interrupt_pin() {
    use crate::bus::pci::ConfigEffect;

    // A host bridge implements no COMMAND bit and no INTx pin.
    let mut function = endpoint();
    let config = function.config();
    config.write8(regs::INTERRUPT_PIN, 0).unwrap();
    config.write16(regs::COMMAND, 0).unwrap();
    config.command_fixed = u16::MAX;
    let plan = containment_plan(config).unwrap();
    assert!(plan.permits(4, 2, u32::from(regs::COMMAND_INTX_DISABLE)));
    assert_eq!(
        plan.apply(config),
        Ok(ConfigEffect::WriteMayHaveApplied {
            offset: regs::COMMAND,
            width: 2
        })
    );
    assert_eq!(config.read16(regs::COMMAND), Ok(0));
    assert_eq!(plan.check_disabled(|offset| config.read16(offset)), Ok(()));
}

#[test]
fn containment_sets_intx_disable_where_writable_even_without_a_pin() {
    let mut function = endpoint();
    let config = function.config();
    config.write8(regs::INTERRUPT_PIN, 0).unwrap();
    config.write16(regs::COMMAND, 6).unwrap();
    let plan = containment_plan(config).unwrap();
    plan.apply(config).unwrap();
    assert_eq!(config.read16(regs::COMMAND), Ok(0x402));
    assert_eq!(plan.check_disabled(|offset| config.read16(offset)), Ok(()));
}

#[test]
fn containment_stops_a_normal_endpoint_with_an_interrupt_pin() {
    let mut function = endpoint();
    let config = function.config();
    config.write16(regs::COMMAND, 6).unwrap();
    let plan = containment_plan(config).unwrap();
    plan.apply(config).unwrap();
    assert_eq!(config.read16(regs::COMMAND), Ok(0x402));
    assert_eq!(plan.check_disabled(|offset| config.read16(offset)), Ok(()));
    // A later clear of INTx Disable on a function with a pin is refused.
    config.write16(regs::COMMAND, 2).unwrap();
    assert!(plan.check_disabled(|offset| config.read16(offset)).is_err());
}

#[test]
fn containment_refuses_an_interrupt_pin_that_ignores_intx_disable() {
    use crate::bus::pci::{ConfigEffect, ConfigFailure};

    let mut function = endpoint();
    let config = function.config();
    config.write16(regs::COMMAND, 2).unwrap();
    config.command_fixed = regs::COMMAND_INTX_DISABLE;
    let plan = containment_plan(config).unwrap();
    let failure = plan.apply(config).unwrap_err();
    assert_eq!(
        failure.effect,
        ConfigEffect::WriteMayHaveApplied {
            offset: regs::COMMAND,
            width: 2
        }
    );
    assert_eq!(
        failure.reason,
        ConfigFailure::Readback {
            offset: regs::COMMAND,
            width: 2,
            expected: 0x402,
            observed: 2,
        }
    );
}

#[test]
fn containment_refuses_bus_master_that_stays_set_without_a_pin() {
    use crate::bus::pci::ConfigFailure;

    let mut function = endpoint();
    let config = function.config();
    config.write8(regs::INTERRUPT_PIN, 0).unwrap();
    config.write16(regs::COMMAND, 4).unwrap();
    config.command_fixed = regs::COMMAND_MASTER;
    let plan = containment_plan(config).unwrap();
    let failure = plan.apply(config).unwrap_err();
    assert!(matches!(
        failure.reason,
        ConfigFailure::Readback {
            offset: regs::COMMAND,
            observed: 0x404,
            ..
        }
    ));
}

#[test]
fn containment_plan_refuses_when_the_interrupt_pin_cannot_be_read() {
    let mut function = endpoint();
    let config = function.config();
    let refused = ClaimError::ReadFailed {
        offset: regs::INTERRUPT_PIN,
    };
    let result = containment::Plan::read(|offset, width| {
        if offset == regs::INTERRUPT_PIN {
            return Err(refused);
        }
        Ok(match width {
            1 => u32::from(config.read8(offset).unwrap()),
            2 => u32::from(config.read16(offset).unwrap()),
            _ => config.read32(offset).unwrap(),
        })
    });
    assert!(matches!(result, Err(error) if error == refused));
}
