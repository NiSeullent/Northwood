//! Exclusive stopped PCI requester kept by its VT-d context. A numeric RID
//! or a cache-policy bit never grants this ownership or DMA coherency.
use crate::vtd::VtdError;
use stela_drivers::bus::pci::{
    CapabilityError, ConfigSpace, Function, owned::OwnedConfig, regs, requester::RequesterBinding,
};

pub struct RequesterClaim {
    rid: u16,
    owner: Owner,
}
impl core::fmt::Debug for RequesterClaim {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RequesterClaim")
            .field("requester", &self.rid)
            .finish_non_exhaustive()
    }
}
enum Owner {
    Pci(Function<OwnedConfig>),
    /// A driver's binding: the driver keeps the configuration owner, and
    /// with it bus mastering.
    Binding(RequesterBinding),
    #[cfg(test)]
    Model(ModelOwner),
}
#[cfg(test)]
struct ModelOwner {
    released: alloc::sync::Arc<core::sync::atomic::AtomicU64>,
    check: Option<alloc::boxed::Box<dyn FnMut()>>,
}
#[cfg(test)]
impl ModelOwner {
    fn check(&mut self) {
        if let Some(check) = self.check.as_mut() {
            check();
        }
    }
}
#[cfg(test)]
impl Drop for ModelOwner {
    fn drop(&mut self) {
        self.released
            .fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    }
}
impl RequesterClaim {
    /// Consume the configuration owner, never a supplied numeric RID.
    ///
    /// # Safety
    /// `function` is the uniquely enumerated ECAM function, including
    /// unmodified address/identity metadata validated against the complete
    /// platform route inventory; no reconstructed/forged Function checks.
    /// Vendor engines/DMA/posted requests and IRQ/work/reset paths are
    /// stopped and drained. This requester has coherent WB RAM DMA,
    /// independently of its CPU cache attributes and VT-d table coherency.
    /// No aliases, firmware/SMM/reset/config owner can change its identity or
    /// DMA policy. The UC config/root resource owner remains permanently
    /// through every domain or quarantine; the current opt-in fixture
    /// uses its permanently host-check root. Native production attach
    /// needs the canonical owned MMIO/root-epoch adapter, not this contract
    /// inferred from an ECAM address. No operation/final release runs in IRQ.
    pub unsafe fn from_stopped_pci(
        function: Function<OwnedConfig>,
    ) -> Result<Self, (VtdError, Function<OwnedConfig>)> {
        if function.address.segment != 0 || function.address.bus != 0 {
            return Err((VtdError::Unsupported, function));
        }
        let rid = u16::from(function.address.device) * 8 + u16::from(function.address.function);
        if function.address.device >= 32 || function.address.function >= 8 {
            return Err((VtdError::Unsupported, function));
        }
        let function = core::mem::ManuallyDrop::new(function);
        let mut original = core::mem::ManuallyDrop::new(alloc::boxed::Box::new(function));
        if let Err(error) = Self::validate_stopped(&mut original) {
            return Err((
                error,
                core::mem::ManuallyDrop::into_inner(*core::mem::ManuallyDrop::into_inner(original)),
            ));
        }
        Ok(Self {
            rid,
            owner: Owner::Pci(core::mem::ManuallyDrop::into_inner(
                *core::mem::ManuallyDrop::into_inner(original),
            )),
        })
    }
    /// Keep a driver's binding of its stopped function.
    ///
    /// # Safety
    /// The function is stopped as [`StoppedRequester::from_stopped_pci`]
    /// requires, and its driver enables bus mastering only after mapping
    /// its DMA through the domain this claim attaches to.
    ///
    /// [`StoppedRequester::from_stopped_pci`]: stela_drivers::bus::pci::requester::StoppedRequester::from_stopped_pci
    pub unsafe fn from_binding(
        binding: RequesterBinding,
    ) -> Result<Self, (VtdError, RequesterBinding)> {
        let binding = core::mem::ManuallyDrop::new(binding);
        let original = core::mem::ManuallyDrop::new(alloc::boxed::Box::new(binding));
        if let Err(error) = original.check_live() {
            return Err((
                VtdError::Config(stela_drivers::bus::pci::ConfigError::before_effect(
                    stela_drivers::bus::pci::ConfigFailure::Mapping(error),
                )),
                core::mem::ManuallyDrop::into_inner(*core::mem::ManuallyDrop::into_inner(original)),
            ));
        }
        let binding =
            core::mem::ManuallyDrop::into_inner(*core::mem::ManuallyDrop::into_inner(original));
        Ok(Self {
            rid: binding.requester(),
            owner: Owner::Binding(binding),
        })
    }
    pub fn requester(&self) -> u16 {
        self.rid
    }
    pub(super) fn check_stopped(&mut self) -> Result<(), VtdError> {
        let function = match &mut self.owner {
            Owner::Pci(f) => f,
            Owner::Binding(binding) => {
                return binding.check_live().map_err(|error| {
                    VtdError::Config(stela_drivers::bus::pci::ConfigError::before_effect(
                        stela_drivers::bus::pci::ConfigFailure::Mapping(error),
                    ))
                });
            }
            #[cfg(test)]
            Owner::Model(model) => {
                model.check();
                return Ok(());
            }
        };
        Self::validate_stopped(function)
    }
    fn validate_stopped<C: ConfigSpace>(function: &mut Function<C>) -> Result<(), VtdError> {
        let expected = u32::from(function.vendor) | u32::from(function.device) << 16;
        let id = function.config().read32(0)?;
        let command = function.config().read16(4)?;
        if id == u32::MAX || command == u16::MAX {
            return Err(VtdError::RegisterRead);
        }
        let intx_quiet = command & regs::COMMAND_INTX_DISABLE != 0
            || !regs::intx_disable_required(function.config().read8(regs::INTERRUPT_PIN)?);
        if id != expected
            || command & regs::COMMAND_MASTER != 0
            || !intx_quiet
            || function.config().read8(0x0e)? & 0x7f != 0
        {
            return Err(VtdError::Busy);
        }
        // Walk the complete checked list; observed capability data never
        // substitutes for the caller's original ownership/exclusion grant.
        let status = function.config().read16(6)?;
        if status == u16::MAX {
            return Err(VtdError::RegisterRead);
        }
        let failed = |error| match error {
            CapabilityError::Config { error, .. } => VtdError::Config(error),
            CapabilityError::FailedRead { .. } => VtdError::RegisterRead,
            _ => VtdError::Unsupported,
        };
        let config = &*function.config();
        let read16 = |offset| match config.read16(offset)? {
            u16::MAX => Err(VtdError::RegisterRead),
            value => Ok(value),
        };
        let mut pcie = false;
        for entry in config.capabilities() {
            let entry = entry.map_err(failed)?;
            let offset = entry.offset;
            let (register, mask) = match entry.id {
                0 => return Err(VtdError::Unsupported),
                5 => (offset + 2, 1),
                0x11 => (offset + 2, 1 << 15),
                0x10 => {
                    if core::mem::replace(&mut pcie, true) {
                        return Err(VtdError::Unsupported);
                    }
                    (offset + 8, 1 << 11)
                }
                _ => continue,
            };
            if read16(register)? & mask != 0 {
                return Err(VtdError::Busy);
            }
        }
        // A zero extended header can describe absence under the original
        // grant. An all-ones PCIe observation cannot establish that absence.
        if pcie && config.read32(0x100)? == u32::MAX {
            return Err(VtdError::RegisterRead);
        }
        for entry in config.extended_capabilities() {
            let entry = entry.map_err(failed)?;
            let (register, mask) = match entry.id {
                0x000f => (entry.offset + 6, 1 << 15),
                0x0013 => (entry.offset + 4, 1),
                0x001b => (entry.offset + 6, 1),
                _ => continue,
            };
            if read16(register)? & mask != 0 {
                return Err(VtdError::Unsupported);
            }
        }
        Ok(())
    }
    /// Bus-master gating is not device completion or quiescence.
    pub(super) fn set_bus_master(&mut self, enabled: bool) -> Result<(), VtdError> {
        let function = match &mut self.owner {
            Owner::Pci(function) => function,
            Owner::Binding(_) => return Err(VtdError::Unsupported),
            #[cfg(test)]
            Owner::Model(model) => {
                model.check();
                return Ok(());
            }
        };
        function.set_bus_master(enabled).map_err(VtdError::Config)
    }
    pub fn into_function(self) -> Result<Function<OwnedConfig>, (VtdError, Self)> {
        match self.owner {
            Owner::Pci(function) => Ok(function),
            owner => Err((
                VtdError::Unsupported,
                Self {
                    rid: self.rid,
                    owner,
                },
            )),
        }
    }
    #[cfg(test)]
    pub(super) fn model(
        rid: u16,
        released: alloc::sync::Arc<core::sync::atomic::AtomicU64>,
    ) -> Self {
        Self {
            rid,
            owner: Owner::Model(ModelOwner {
                released,
                check: None,
            }),
        }
    }
    #[cfg(test)]
    pub(super) fn model_with_check(
        rid: u16,
        released: alloc::sync::Arc<core::sync::atomic::AtomicU64>,
        check: alloc::boxed::Box<dyn FnMut()>,
    ) -> Self {
        Self {
            rid,
            owner: Owner::Model(ModelOwner {
                released,
                check: Some(check),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use stela_drivers::bus::pci::Address;
    struct Config([u32; 1024]);
    impl ConfigSpace for Config {
        fn read32(&self, offset: u16) -> Result<u32, stela_drivers::bus::pci::ConfigError> {
            Ok(self.0[usize::from(offset / 4)])
        }
        fn write32(
            &mut self,
            offset: u16,
            value: u32,
        ) -> Result<(), stela_drivers::bus::pci::ConfigError> {
            self.0[usize::from(offset / 4)] = value;
            Ok(())
        }

        fn read8(&self, offset: u16) -> Result<u8, stela_drivers::bus::pci::ConfigError> {
            Ok((self.read32(offset & !3)? >> ((offset & 3) * 8)) as u8)
        }

        fn write8(
            &mut self,
            offset: u16,
            value: u8,
        ) -> Result<(), stela_drivers::bus::pci::ConfigError> {
            let shift = (offset & 3) * 8;
            let word = self.read32(offset & !3)? & !(0xff << shift) | u32::from(value) << shift;
            self.write32(offset & !3, word)
        }

        fn read16(&self, offset: u16) -> Result<u16, stela_drivers::bus::pci::ConfigError> {
            Ok((self.read32(offset & !3)? >> ((offset & 2) * 8)) as u16)
        }

        fn write16(
            &mut self,
            offset: u16,
            value: u16,
        ) -> Result<(), stela_drivers::bus::pci::ConfigError> {
            let shift = (offset & 2) * 8;
            let word = self.read32(offset & !3)? & !(0xffff << shift) | u32::from(value) << shift;
            self.write32(offset & !3, word)
        }
    }
    fn function() -> Function<Config> {
        let mut c = Config([0; 1024]);
        c.0[0] = 0x11e8_1234;
        c.0[1] = 1 << 10;
        // Interrupt Pin INTA.
        c.0[0x3c / 4] = 1 << 8;
        Function::new(
            Address {
                segment: 0,
                bus: 0,
                device: 2,
                function: 0,
            },
            c,
        )
        .unwrap()
    }
    #[test]
    fn config_identity_mastering_and_interrupt_policy_are_checked() {
        let mut f = function();
        assert_eq!(RequesterClaim::validate_stopped(&mut f), Ok(()));
        for bad in [4, 0, u16::MAX] {
            f.config().write16(4, bad).unwrap();
            assert!(RequesterClaim::validate_stopped(&mut f).is_err());
        }
        // Without an INTx pin a hardwired-clear INTx Disable is quiet; Bus
        // Master still refuses.
        f.config().write8(0x3d, 0).unwrap();
        f.config().write16(4, 0).unwrap();
        assert_eq!(RequesterClaim::validate_stopped(&mut f), Ok(()));
        f.config().write16(4, 4).unwrap();
        assert_eq!(
            RequesterClaim::validate_stopped(&mut f),
            Err(VtdError::Busy)
        );
        f.config().write8(0x3d, 1).unwrap();
        f.config().write16(4, 1 << 10).unwrap();
        f.config().write32(0, 0x11e9_1234).unwrap();
        assert_eq!(
            RequesterClaim::validate_stopped(&mut f),
            Err(VtdError::Busy)
        );
        for (id, offset, active) in [(5, 0x42, 1), (0x11, 0x42, 1 << 15), (0x10, 0x48, 1 << 11)] {
            let mut f = function();
            f.config().write16(6, 16).unwrap();
            f.config().write8(0x34, 0x40).unwrap();
            f.config().write16(0x40, id).unwrap();
            f.config().write16(offset, active).unwrap();
            assert_eq!(
                RequesterClaim::validate_stopped(&mut f),
                Err(VtdError::Busy)
            );
            f.config().write16(offset, 0).unwrap();
            assert_eq!(RequesterClaim::validate_stopped(&mut f), Ok(()));
        }
    }
    #[test]
    fn malformed_or_failed_capability_reads_do_not_check_a_claim() {
        for pointer in [0x41, 0x3c, 0xff] {
            let mut f = function();
            f.config().write16(6, 16).unwrap();
            f.config().write8(0x34, pointer).unwrap();
            assert_eq!(
                RequesterClaim::validate_stopped(&mut f),
                Err(VtdError::Unsupported)
            );
        }
        let mut f = function();
        f.config().write16(6, 16).unwrap();
        f.config().write8(0x34, 0x40).unwrap();
        f.config().write16(0x40, 0x4005).unwrap();
        assert_eq!(
            RequesterClaim::validate_stopped(&mut f),
            Err(VtdError::Unsupported)
        );
        f.config().write16(0x40, 0x10).unwrap();
        f.config().write32(0x100, u32::MAX).unwrap();
        assert_eq!(
            RequesterClaim::validate_stopped(&mut f),
            Err(VtdError::RegisterRead)
        );
        for (id, control, mask) in [(0xf, 0x106, 1 << 15), (0x13, 0x104, 1), (0x1b, 0x106, 1)] {
            let mut f = function();
            f.config().write32(0x100, id | 1 << 16).unwrap();
            f.config().write16(control, mask).unwrap();
            assert_eq!(
                RequesterClaim::validate_stopped(&mut f),
                Err(VtdError::Unsupported)
            );
            f.config().write16(control, u16::MAX).unwrap();
            assert_eq!(
                RequesterClaim::validate_stopped(&mut f),
                Err(VtdError::RegisterRead)
            );
            f.config().write16(control, 0).unwrap();
            assert_eq!(RequesterClaim::validate_stopped(&mut f), Ok(()));
        }
        let mut f = function();
        f.config().write32(0x100, 0x1001_000f).unwrap();
        assert_eq!(
            RequesterClaim::validate_stopped(&mut f),
            Err(VtdError::Unsupported)
        );
    }
}
