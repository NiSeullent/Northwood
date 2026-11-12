// SPDX-License-Identifier: GPL-2.0-or-later

extern crate std;

use super::*;
use crate::bus::{
    dma::{
        mapping::{Direction, DmaDomain, DmaError, DomainMapping},
        pages::DmaPages,
    },
    lpss::tests::native_fixture_for,
    pci::{
        Address,
        owned::test_support::fixture_pointer,
        requester::{
            FunctionRole, RequesterBinding,
            preparation::{AttachmentBackend, DetachError, Root},
        },
    },
};
use alloc::{boxed::Box, rc::Rc, sync::Arc, vec::Vec};
use core::{
    cell::{Cell, RefCell},
    ops::Range,
    sync::atomic::Ordering,
};

struct UnboundRoot(Rc<Cell<usize>>);

impl Drop for UnboundRoot {
    fn drop(&mut self) {
        self.0.set(self.0.get() + 1);
    }
}

// SAFETY: a model with no hardware, asynchronous owners or bindings. This
// stage is forbidden to reach either of the original-root operations.
unsafe impl Root for UnboundRoot {
    fn drain(&self) -> Result<(), DmaError> {
        panic!("LPSS reset observation cannot drain the original root");
    }

    unsafe fn bind(&self, _: RequesterBinding) -> Result<Box<dyn AttachmentBackend>, DmaError> {
        panic!("LPSS reset observation cannot publish a new root binding");
    }
}

fn preparing(function: Function<OwnedConfig>) -> (PreparingRequester, Rc<Cell<usize>>) {
    let drops = Rc::new(Cell::new(0));
    let root = Box::new(UnboundRoot(drops.clone()));
    // SAFETY: the same isolated owned PCI fixture, without competing firmware,
    // configuration, interrupt or MMIO users. No DMA grant is inferred.
    let requester = unsafe { PreparingRequester::new(function, FunctionRole::Physical, root) }
        .unwrap_or_else(|_| panic!("retained fixture configuration"));
    (requester, drops)
}

#[test]
fn gemini_lake_readback_holds_the_function_and_idma_in_reset() {
    for device in [
        0x31ac, 0x31ae, 0x31b0, 0x31b2, 0x31b4, 0x31b6, 0x31b8, 0x31ba,
    ] {
        let (function, mapping, config, bar) = native_fixture_for(device);
        mapping.write32(RESET, RESET_FUNCTION_AND_IDMA).unwrap();
        mapping.write32(REMAP_LOW, 0x0123_4000).unwrap();
        mapping.write32(0x6c, 1).unwrap();
        let reset = mapping.address(RESET, 4).unwrap().cast::<u32>();
        let remap = mapping.address(REMAP_LOW, 4).unwrap().cast::<u32>();
        let child = mapping.address(0x6c, 4).unwrap().cast::<u32>();
        let (requester, drops) = preparing(function);
        {
            // SAFETY: synchronous owned BAR and function fixture in D0.
            let _held = unsafe { hold_inherited_reset(requester, mapping) }
                .unwrap_or_else(|refusal| panic!("GLK reset: {:?}", refusal.error));
        }
        // SAFETY: no other user of the original model registers. The staged
        // reset keeps the same owned mapping live even after abandonment.
        unsafe {
            assert_eq!(reset.as_ptr().read_volatile(), 0);
            assert_eq!(remap.as_ptr().read_volatile(), 0x0123_4000);
            assert_eq!(child.as_ptr().read_volatile(), 1);
        }
        assert_eq!(drops.get(), 0);
        assert_eq!(config.unmaps.load(Ordering::Relaxed), 0);
        assert_eq!(bar.unmaps.load(Ordering::Relaxed), 0);
    }
}

#[test]
fn unrelated_identities_or_invalid_lpss_type_do_not_write_reset() {
    for (device, caps) in [
        (0x31c2, 0), // GLK SPI
        (0x9d60, 0), // Sunrise Point I²C
        (0x31ba, u32::MAX),
        (0x31ba, 1 << 4),
    ] {
        let (function, mapping, config, bar) = native_fixture_for(device);
        mapping.write32(CAPS, caps).unwrap();
        mapping.write32(RESET, RESET_FUNCTION_AND_IDMA).unwrap();
        let reset = mapping.address(RESET, 4).unwrap().cast::<u32>();
        let (requester, drops) = preparing(function);
        // SAFETY: same isolated config and BAR; identity and type are
        // deliberately unsupported before any native reset write.
        let refusal = unsafe { hold_inherited_reset(requester, mapping) }
            .err()
            .unwrap();
        assert_eq!(refusal.error, Error::Unsupported);
        drop(refusal);
        // SAFETY: the refused attempt retains the original fixture mapping.
        assert_eq!(
            unsafe { reset.as_ptr().read_volatile() },
            RESET_FUNCTION_AND_IDMA
        );
        assert_eq!(drops.get(), 0);
        assert_eq!(config.unmaps.load(Ordering::Relaxed), 0);
        assert_eq!(bar.unmaps.load(Ordering::Relaxed), 0);
    }
}

#[test]
fn d3_or_uncontained_pci_configuration_does_not_write_reset() {
    for command in [
        regs::COMMAND_MEMORY,
        regs::COMMAND_INTX_DISABLE,
        regs::COMMAND_MEMORY | regs::COMMAND_INTX_DISABLE,
    ] {
        let (mut function, mapping, _, _) = native_fixture_for(0x31ba);
        function.config().write16(regs::COMMAND, command).unwrap();
        if command == regs::COMMAND_MEMORY | regs::COMMAND_INTX_DISABLE {
            // A valid PM capability that reports D3hot. The cold stage must
            // not enter MMIO or try to power up the inherited function.
            function
                .config()
                .write16(regs::STATUS, regs::STATUS_CAP_LIST)
                .unwrap();
            function.config().write8(0x34, 0x40).unwrap();
            function.config().write16(0x40, 1).unwrap();
            function.config().write16(0x42, 3).unwrap();
            function.config().write16(0x44, 3).unwrap();
        }
        mapping.write32(RESET, RESET_FUNCTION_AND_IDMA).unwrap();
        let reset = mapping.address(RESET, 4).unwrap().cast::<u32>();
        let (requester, _) = preparing(function);
        // SAFETY: same isolated original fixture. Its deliberately invalid
        // configuration must be refused before any register access.
        let refusal = unsafe { hold_inherited_reset(requester, mapping) }
            .err()
            .unwrap();
        let expected = if command == regs::COMMAND_MEMORY {
            Error::Claim(ClaimError::Active {
                offset: regs::COMMAND,
            })
        } else {
            Error::Configuration
        };
        assert_eq!(refusal.error, expected);
        // SAFETY: the refused attempt keeps the original RAM register.
        assert_eq!(
            unsafe { reset.as_ptr().read_volatile() },
            RESET_FUNCTION_AND_IDMA
        );
    }
}

#[test]
fn denied_mmio_before_or_after_reset_keeps_original_owners() {
    for ordinal in 1..=5 {
        let (function, mapping, config, bar) = native_fixture_for(0x31ba);
        mapping.write32(RESET, RESET_FUNCTION_AND_IDMA).unwrap();
        let reset = mapping.address(RESET, 4).unwrap().cast::<u32>();
        let cut = bar.contexts.load(Ordering::Relaxed) + ordinal;
        bar.fail_context_at.store(cut, Ordering::Relaxed);
        let (requester, drops) = preparing(function);
        // SAFETY: same isolated original mapping with a deliberate denial
        // injected in its real per-access context gate.
        let refusal = unsafe { hold_inherited_reset(requester, mapping) }
            .err()
            .unwrap();
        assert_eq!(
            refusal.error,
            Error::Mmio(crate::bus::mmio::owned::Error::Unavailable)
        );
        drop(refusal);
        assert_eq!(bar.contexts.load(Ordering::Relaxed), cut);
        let expected = if ordinal <= 3 {
            RESET_FUNCTION_AND_IDMA
        } else {
            0
        };
        // SAFETY: original synchronous fixture remains held; this is only
        // an observation of the mock's state after refusal.
        assert_eq!(unsafe { reset.as_ptr().read_volatile() }, expected);
        assert_eq!(drops.get(), 0);
        assert_eq!(config.unmaps.load(Ordering::Relaxed), 0);
        assert_eq!(bar.unmaps.load(Ordering::Relaxed), 0);
    }
}

#[test]
fn reset_readback_must_follow_the_write_and_show_every_bit_asserted() {
    for observed in [0, 1, 2, 4, u32::MAX] {
        let written = Cell::new(false);
        let result = assert_reset(
            || {
                assert!(!written.replace(true));
                Ok(())
            },
            || {
                assert!(written.get());
                Ok(observed)
            },
        );
        assert_eq!(
            result,
            if observed == 0 {
                Ok(())
            } else {
                Err(Error::Register {
                    offset: RESET,
                    expected: 0,
                    observed,
                })
            }
        );
    }
    assert_eq!(
        assert_reset(|| Err(Error::Io), || panic!("read after failed write")),
        Err(Error::Io)
    );
}

#[derive(Clone, Copy)]
enum RootResult {
    Bind,
    RefuseDrain,
    RefuseBind,
    Unwind,
}

#[derive(Default)]
struct RootTrace {
    calls: RefCell<Vec<&'static str>>,
    drops: Cell<usize>,
}

struct TracedRoot {
    address: Address,
    result: RootResult,
    trace: Rc<RootTrace>,
}

impl Drop for TracedRoot {
    fn drop(&mut self) {
        self.trace.drops.set(self.trace.drops.get() + 1);
    }
}

// SAFETY: an isolated synchronous RAM model. Drain and binding are
// independent modeled acknowledgements for the original function.
unsafe impl Root for TracedRoot {
    fn drain(&self) -> Result<(), DmaError> {
        self.trace.calls.borrow_mut().push("drain");
        match self.result {
            RootResult::RefuseDrain => Err(DmaError::DrainUnavailable),
            RootResult::Unwind => panic!("injected original-root unwind"),
            _ => Ok(()),
        }
    }

    unsafe fn bind(
        &self,
        binding: RequesterBinding,
    ) -> Result<Box<dyn AttachmentBackend>, DmaError> {
        assert_eq!(binding.address(), self.address);
        binding.check_live().unwrap();
        assert_eq!(&*self.trace.calls.borrow(), &["drain"]);
        self.trace.calls.borrow_mut().push("bind");
        if matches!(self.result, RootResult::RefuseBind) {
            return Err(DmaError::MappingFailed);
        }
        Ok(Box::new(NoDmaAttachment {
            _binding: binding,
            _original: self.trace.clone(),
        }))
    }
}

struct NoDmaAttachment {
    _binding: RequesterBinding,
    _original: Rc<RootTrace>,
}

// SAFETY: this fixture exposes no DMA mapping or IRQ and retains its
// original binding. Detach refuses before any effect.
unsafe impl AttachmentBackend for NoDmaAttachment {
    fn domain(&self) -> Arc<dyn DmaDomain> {
        Arc::new(NoDma)
    }

    unsafe fn detach(&mut self) -> Result<(), DetachError> {
        Err(DetachError::Unexposed(DmaError::IommuUnavailable))
    }
}

struct NoDma;

// SAFETY: the fixture creates no mappings or DMA visibility promises.
unsafe impl DmaDomain for NoDma {
    unsafe fn map(&self, _: &DmaPages, _: Direction, _: u64) -> Result<DomainMapping, DmaError> {
        Err(DmaError::IommuUnavailable)
    }

    unsafe fn unmap(&self, _: DomainMapping) -> Result<(), DmaError> {
        Err(DmaError::Unmapped)
    }

    unsafe fn sync_for_device(
        &self,
        _: DomainMapping,
        _: Range<usize>,
        _: Direction,
    ) -> Result<(), DmaError> {
        Err(DmaError::Unmapped)
    }

    unsafe fn sync_for_cpu(
        &self,
        _: DomainMapping,
        _: Range<usize>,
        _: Direction,
    ) -> Result<(), DmaError> {
        Err(DmaError::Unmapped)
    }
}

fn traced(
    function: Function<OwnedConfig>,
    result: RootResult,
) -> (PreparingRequester, Rc<RootTrace>) {
    let trace = Rc::new(RootTrace::default());
    let root = Box::new(TracedRoot {
        address: function.address,
        result,
        trace: trace.clone(),
    });
    // SAFETY: this exact paired model has no outside firmware, IRQ, work,
    // reset or DMA writers. Its configuration and old root are exclusive.
    let requester = unsafe { PreparingRequester::new(function, FunctionRole::Physical, root) }
        .unwrap_or_else(|_| panic!("retained fixture configuration"));
    (requester, trace)
}

#[test]
fn finish_drains_then_binds_the_original_requester_and_returns_its_exact_bar() {
    let (function, mapping, config, bar) = native_fixture_for(0x31ba);
    let address = function.address;
    let bar_span = mapping.physical_span().unwrap();
    let original = mapping.address(0, WINDOW_BYTES).unwrap();
    let (requester, trace) = traced(function, RootResult::Bind);
    // SAFETY: this isolated fixture has no real device, competing owners or DMA.
    let held = unsafe { hold_inherited_reset(requester, mapping) }
        .unwrap_or_else(|failure| panic!("{:?}", failure.error));
    assert!(trace.calls.borrow().is_empty());
    // SAFETY: all former producer roles are absent in this model. The same
    // fixture owns the exact original resources through finalization.
    let Finished { bound, mapping } =
        unsafe { held.finish() }.unwrap_or_else(|failure| panic!("{:?}", failure.error));
    assert_eq!(&*trace.calls.borrow(), &["drain", "bind"]);
    assert_eq!(bound.requester.address(), address);
    assert_eq!(mapping.address(0, WINDOW_BYTES).unwrap(), original);
    assert_eq!(mapping.physical_span().unwrap(), bar_span);
    assert_eq!(mapping.read32(RESET), Ok(0));
    let mut function = bound.requester.into_function();
    assert!(bound.attachment.matches_config(function.config()).unwrap());
    assert_eq!(
        function.config().read16(regs::COMMAND).unwrap(),
        regs::COMMAND_MEMORY | regs::COMMAND_INTX_DISABLE
    );
    assert_eq!(config.unmaps.load(Ordering::Relaxed), 0);
    assert_eq!(bar.unmaps.load(Ordering::Relaxed), 0);
}

#[test]
fn finish_refuses_reset_or_bar_drift_without_writing_or_reaching_the_root() {
    for reset_drift in [false, true] {
        let (mut function, mapping, config, bar) = native_fixture_for(0x31ba);
        let bar0 = fixture_pointer(function.config(), 0x10).cast::<u32>();
        let reset = mapping.address(RESET, 4).unwrap().cast::<u32>();
        // SAFETY: this raw pointer addresses the isolated fixture's original RAM.
        let original_bar = unsafe { bar0.as_ptr().read_volatile() };
        let (requester, trace) = traced(function, RootResult::Bind);
        // SAFETY: same exclusively owned RAM fixture, in D0 with no producers.
        let held = unsafe { hold_inherited_reset(requester, mapping) }
            .unwrap_or_else(|failure| panic!("{:?}", failure.error));
        let (observed_bar, observed_reset, error) = if reset_drift {
            (
                original_bar,
                RESET_FUNCTION_AND_IDMA,
                Error::Register {
                    offset: RESET,
                    expected: 0,
                    observed: RESET_FUNCTION_AND_IDMA,
                },
            )
        } else {
            (original_bar ^ 0x1000, 0, Error::Mapping)
        };
        // SAFETY: inject register drift in the isolated original RAM without
        // an actual reset, reassignment, or asynchronous device producer.
        unsafe {
            bar0.as_ptr().write_volatile(observed_bar);
            reset.as_ptr().write_volatile(observed_reset);
        }
        // SAFETY: the RAM owners stay exclusive and no producer can restart.
        let failure = unsafe { held.finish() }
            .err()
            .expect("stale observation refused");
        assert_eq!(failure.error, FinishError::Recheck(error));
        drop(failure);
        assert!(trace.calls.borrow().is_empty());
        // SAFETY: refusal keeps the two original fixture mappings.
        unsafe {
            assert_eq!(bar0.as_ptr().read_volatile(), observed_bar);
            assert_eq!(reset.as_ptr().read_volatile(), observed_reset);
        }
        assert_eq!(trace.drops.get(), 0);
        assert_eq!(config.unmaps.load(Ordering::Relaxed), 0);
        assert_eq!(bar.unmaps.load(Ordering::Relaxed), 0);
    }
}

#[test]
fn finish_quarantines_both_owners_on_root_refusal_or_unwind() {
    for (result, error, calls) in [
        (
            RootResult::RefuseDrain,
            DmaError::DrainUnavailable,
            &["drain"][..],
        ),
        (
            RootResult::RefuseBind,
            DmaError::MappingFailed,
            &["drain", "bind"][..],
        ),
        (
            RootResult::Unwind,
            DmaError::DrainUnavailable,
            &["drain"][..],
        ),
    ] {
        let (function, mapping, config, bar) = native_fixture_for(0x31ba);
        let reset = mapping.address(RESET, 4).unwrap().cast::<u32>();
        let (requester, trace) = traced(function, result);
        // SAFETY: synchronous original RAM model, no physical requester.
        let held = unsafe { hold_inherited_reset(requester, mapping) }
            .unwrap_or_else(|failure| panic!("{:?}", failure.error));
        let finish = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // SAFETY: all asynchronous roles are absent and this fixture
            // keeps every original owner until the failed transition.
            unsafe { held.finish() }
        }));
        if matches!(result, RootResult::Unwind) {
            assert!(finish.is_err());
        } else {
            let failure = finish.unwrap().err().expect("root refused");
            assert_eq!(
                failure.error,
                FinishError::Finalize(preparation::Error::Dma(error))
            );
            drop(failure);
        }
        assert_eq!(&*trace.calls.borrow(), calls);
        // SAFETY: the original fixture's BAR stays kept on failure/unwind.
        assert_eq!(unsafe { reset.as_ptr().read_volatile() }, 0);
        assert_eq!(trace.drops.get(), 0);
        assert_eq!(config.unmaps.load(Ordering::Relaxed), 0);
        assert_eq!(bar.unmaps.load(Ordering::Relaxed), 0);
    }
}
