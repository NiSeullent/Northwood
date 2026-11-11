// SPDX-License-Identifier: GPL-2.0-or-later
//! Gemini Lake LPSS inherited engine reset, before any requester handoff.
//! Register layout and GLK PCI IDs were checked against Linux
//! cc9de5d4b5ab6459f5774946aa3195ad51236fa8, drivers/mfd/intel-lpss.c
//! and drivers/mfd/intel-lpss-pci.c.

use super::*;
use crate::bus::{
    mmio::owned::ResourceKind,
    pci::{
        CAPABILITY_POWER,
        requester::{
            ClaimError,
            preparation::{self, BoundRequester, PreparingRequester},
            validate_stopped,
        },
    },
};

type Resources = (PreparingRequester, Mapping);

/// The function and iDMA reset was observed asserted. Its original
/// configuration, root and BAR remain held. This does not attest to prior
/// interrupt/work retirement or original-root DMA completion.
pub struct ResetHeld {
    resources: ManuallyDrop<Resources>,
}

/// The refused attempt retains the same original resources.
pub struct Refusal {
    pub error: Error,
    _resources: ManuallyDrop<Resources>,
}

/// The original requester, its bound attachment and the exact LPSS BAR.
pub struct Finished {
    pub bound: BoundRequester,
    pub mapping: Mapping,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FinishError {
    Recheck(Error),
    Finalize(preparation::Error),
}

/// The failed transition keeps the exact original BAR and requester graph.
pub struct FinishRefusal {
    pub error: FinishError,
    _resources: FinishResources,
}

enum FinishResources {
    Recheck {
        _held: ResetHeld,
    },
    Finalize {
        _failed: ManuallyDrop<(preparation::Failure, Mapping)>,
    },
}

impl ResetHeld {
    /// Recheck the reset and consume the original root's completion path.
    ///
    /// # Safety
    /// Independently establish that all firmware, IRQ delivery, work and
    /// submission owners of this exact function have retired. Bypass and
    /// peer traffic are excluded or drained. The same exclusive configuration,
    /// BAR, power and clock epoch persists, with AML/SMM, PUNIT and all other
    /// reset, power or MMIO writers excluded. The controller cannot restart
    /// during finalization. The caller keeps all prior DMA backing through
    /// completion or quarantine, and observes the original root's context.
    /// Reset readback does not supply any of these guarantees.
    pub unsafe fn finish(mut self) -> Result<Finished, FinishRefusal> {
        let (requester, mapping) = &mut *self.resources;
        let rechecked = check_pci(requester.function_mut(), mapping)
            .and_then(|()| check_reset(mapping.read32(RESET).map_err(Error::Mmio)?));
        if let Err(error) = rechecked {
            return Err(FinishRefusal {
                error: FinishError::Recheck(error),
                _resources: FinishResources::Recheck { _held: self },
            });
        }

        let (requester, mapping) = ManuallyDrop::into_inner(self.resources);
        // Keep the original BAR before invoking the root callbacks, including
        // if their finalization unwinds.
        let mapping = ManuallyDrop::new(mapping);
        // SAFETY: the reset and contained PCI state were rechecked in this
        // original epoch. The caller supplies independent producer retirement
        // and bypass/peer exclusion, with the old DMA backing and BAR kept.
        let receipt = unsafe { requester.acknowledge_quiescence() };
        match receipt.finalize() {
            Ok(bound) => Ok(Finished {
                bound,
                mapping: ManuallyDrop::into_inner(mapping),
            }),
            Err(failed) => Err(FinishRefusal {
                error: FinishError::Finalize(failed.error),
                _resources: FinishResources::Finalize {
                    _failed: ManuallyDrop::new((failed, ManuallyDrop::into_inner(mapping))),
                },
            }),
        }
    }
}

/// Assert the inherited LPSS I²C function and iDMA reset and read it back.
///
/// # Safety
/// The caller owns the same original 31xx requester and complete exclusive UC
/// BAR0 through the live original epoch. The platform has excluded competing
/// firmware, AML/SMM, PUNIT and other reset/power/MMIO users. It keeps the
/// power epoch stable and the controller clock live in D0, has independently
/// contained interrupt delivery, and keeps all inherited DMA backing.
/// Configuration is already contained. No child/worker is published. Do not
/// use the resulting observation as a root drain or a permission to resume DMA.
pub unsafe fn hold_inherited_reset(
    requester: PreparingRequester,
    mapping: Mapping,
) -> Result<ResetHeld, Refusal> {
    let mut resources = ManuallyDrop::new((requester, mapping));
    let result = (|| {
        let (requester, mapping) = &mut *resources;
        let function = requester.function_mut();
        let profile = Profile::select(Identity {
            vendor: function.vendor,
            device: function.device,
            subsystem_vendor: 0,
            subsystem_device: 0,
        })?;
        if profile.kind != ProfileKind::GeminiLake {
            return Err(Error::Unsupported);
        }
        check_pci(function, mapping)?;
        let caps = mapping.read32(CAPS).map_err(Error::Mmio)?;
        if caps == u32::MAX || caps & CAPS_TYPE != 0 {
            return Err(Error::Unsupported);
        }
        // intel_lpss_init_dev holds both LPSS_PRIV_RESETS_FUNC bits and
        // LPSS_PRIV_RESETS_IDMA low by writing zero. Here the reset stays
        // asserted until the caller has independent quiescence evidence.
        assert_reset(
            || mapping.write32(RESET, 0).map_err(Error::Mmio),
            || mapping.read32(RESET).map_err(Error::Mmio),
        )?;
        check_pci(function, mapping)
    })();
    match result {
        Ok(()) => Ok(ResetHeld { resources }),
        Err(error) => Err(Refusal {
            error,
            _resources: resources,
        }),
    }
}

fn check_pci(function: &mut Function<OwnedConfig>, mapping: &Mapping) -> Result<(), Error> {
    // This existing check is PCI configuration only. It checks identity and
    // that PCI mastering, MSI/MSI-X and other request sources stay disabled.
    validate_stopped(function).map_err(Error::Claim)?;
    if let Some((offset, _)) = function
        .capabilities()
        .map_err(|error| Error::Claim(ClaimError::from(error)))?
        .into_iter()
        .find(|&(_, id)| id == CAPABILITY_POWER)
    {
        // PCI PMCSR is at capability + 4. State 0 means D0. The owned
        // platform excludes D3cold when a function has no PM capability.
        let power = function.config().read16(u16::from(offset) + 4)?;
        if power & 3 != 0 {
            return Err(Error::Configuration);
        }
    }
    if mapping.kind().map_err(Error::Mmio)? != ResourceKind::Registers {
        return Err(Error::Mapping);
    }
    native_window(function.config(), mapping)?;
    Ok(())
}

fn assert_reset(
    write: impl FnOnce() -> Result<(), Error>,
    read: impl FnOnce() -> Result<u32, Error>,
) -> Result<(), Error> {
    write()?;
    check_reset(read()?)
}

fn check_reset(observed: u32) -> Result<(), Error> {
    if observed != 0 {
        return Err(Error::Register {
            offset: RESET,
            expected: 0,
            observed,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests;
