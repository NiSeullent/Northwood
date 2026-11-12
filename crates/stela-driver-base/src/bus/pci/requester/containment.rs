//! Exact PCI control writes derived from a read-only configuration census.
use super::{ATS, ClaimError, MSI, MSIX, PASID, PCIE, PRI};
use crate::bus::{
    mmio::owned::Error as MappingError,
    pci::{
        ConfigEffect, ConfigError, ConfigFailure, ConfigSpace,
        capability::{Capabilities, ExtendedCapabilities, Layout, Transport},
        regs,
    },
};
use alloc::vec::Vec;
use core::cell::RefCell;

#[derive(Clone, Copy)]
struct Control {
    offset: u16,
    original: u16,
    disabled: u16,
    /// Bits of `disabled` the function may leave clear (see
    /// regs::command_optional_bits); every other bit must read back exactly.
    optional: u16,
}

/// This plan describes PCI-level controls only. It conveys no physical
/// ownership, engine stop, or authority to release old DMA or IRQ state.
pub struct Plan(Vec<Control>);

impl Plan {
    pub fn read(read: impl FnMut(u16, u8) -> Result<u32, ClaimError>) -> Result<Self, ClaimError> {
        let reader = RefCell::new(read);
        let read = |offset, width| reader.borrow_mut()(offset, width);
        let mut plan = Self(Vec::new());
        let command = read(regs::COMMAND, 2)? as u16;
        if command == u16::MAX {
            return Err(ClaimError::ReadFailed {
                offset: regs::COMMAND,
            });
        }
        // INTx Disable is asked of every function and required of those
        // with an INTx pin; Bus Master must read back clear on all of them.
        let pin = read(regs::INTERRUPT_PIN, 1)? as u8;
        plan.push(
            regs::COMMAND,
            command,
            command & !regs::COMMAND_MASTER | regs::COMMAND_INTX_DISABLE,
            regs::command_optional_bits(pin),
        )?;

        let mut express = false;
        let mut seen = 0u32;
        for entry in Capabilities::new(Transport(read)) {
            let entry = entry?;
            let id = u16::from(entry.id);
            let (relative, clear, event) = match id {
                MSI => (2, 1, 0),
                MSIX => (2, 1 << 15, 0),
                PCIE => {
                    express = true;
                    // Device Control bit 15 can initiate FLR or bridge retry.
                    (8, 1 << 11, 1 << 15)
                }
                _ => continue,
            };
            if entry.layout != Layout::Sized {
                return Err(ClaimError::UnsupportedCapability {
                    id,
                    extended: false,
                });
            }
            record_capability(&mut seen, id, entry.offset, relative, entry.bytes)?;
            plan.disable(read, entry.offset + relative, clear, event)?;
        }

        if express && read(0x100, 4)? != u32::MAX {
            let mut seen = 0u32;
            for entry in ExtendedCapabilities::new(Transport(read)) {
                let entry = entry?;
                let (relative, clear, event) = match entry.id {
                    ATS => (6, 1 << 15, 0),
                    PASID => (6, 1, 0),
                    PRI => (4, 1, 2), // Reset is a command, not persistent state.
                    _ => continue,
                };
                if entry.version != 1 {
                    return Err(ClaimError::UnsupportedVersion {
                        id: entry.id,
                        version: entry.version,
                    });
                }
                if entry.layout != Layout::Sized {
                    return Err(ClaimError::MalformedCapability {
                        offset: entry.offset,
                    });
                }
                record_capability(&mut seen, entry.id, entry.offset, relative, entry.bytes)?;
                plan.disable(read, entry.offset + relative, clear, event)?;
            }
        }
        Ok(plan)
    }

    fn disable(
        &mut self,
        mut read: impl FnMut(u16, u8) -> Result<u32, ClaimError>,
        offset: u16,
        clear: u16,
        event: u16,
    ) -> Result<(), ClaimError> {
        let original = read(offset, 2)? as u16;
        if original == u16::MAX {
            return Err(ClaimError::ReadFailed { offset });
        }
        if original & event != 0 {
            return Err(ClaimError::Config(ConfigError::before_effect(
                ConfigFailure::UnqualifiedControl,
            )));
        }
        self.push(offset, original, original & !clear, 0)
    }

    fn push(
        &mut self,
        offset: u16,
        original: u16,
        disabled: u16,
        optional: u16,
    ) -> Result<(), ClaimError> {
        if self.0.iter().any(|control| control.offset == offset) {
            return Err(ClaimError::MalformedCapability { offset });
        }
        self.0
            .try_reserve(1)
            .map_err(|_| ClaimError::Mapping(MappingError::Exhausted))?;
        self.0.push(Control {
            offset,
            original,
            disabled,
            optional,
        });
        Ok(())
    }

    /// Does this exact write narrow one observed control?
    pub fn permits(&self, offset: u16, width: usize, value: u32) -> bool {
        width == 2
            && self.0.iter().any(|control| {
                control.offset == offset
                    && control.original != control.disabled
                    && u32::from(control.disabled) == value
            })
    }

    /// Preflight every old control before issuing any write to this function.
    /// Recheck each changed word and retain the last write on a later refusal.
    pub fn apply(&self, config: &mut impl ConfigSpace) -> Result<ConfigEffect, ConfigError> {
        for &control in &self.0 {
            verify(
                control.offset,
                control.original,
                0,
                config.read16(control.offset)?,
            )?;
        }
        let mut effect = ConfigEffect::None;
        for &control in &self.0 {
            if control.original != control.disabled {
                config
                    .write16(control.offset, control.disabled)
                    .map_err(|error| with_prefix(error, effect))?;
                effect = ConfigEffect::WriteMayHaveApplied {
                    offset: control.offset,
                    width: 2,
                };
                config
                    .read16(control.offset)
                    .and_then(|observed| control.verify_disabled(observed))
                    .map_err(|error| with_prefix(error, effect))?;
            }
        }
        Ok(effect)
    }

    /// Recheck the stopped controls with a read-only configuration view.
    pub fn check_disabled(
        &self,
        mut read: impl FnMut(u16) -> Result<u16, ConfigError>,
    ) -> Result<(), ConfigError> {
        for control in &self.0 {
            control.verify_disabled(read(control.offset)?)?;
        }
        Ok(())
    }
}

fn record_capability(
    seen: &mut u32,
    id: u16,
    offset: u16,
    relative: u16,
    bytes: u16,
) -> Result<(), ClaimError> {
    let bit = 1 << id;
    if relative + 2 > bytes || *seen & bit != 0 {
        return Err(ClaimError::MalformedCapability { offset });
    }
    *seen |= bit;
    Ok(())
}

impl Control {
    /// A stopped control reads back its disabled value; an optional bit may
    /// read back clear instead.
    fn verify_disabled(&self, observed: u16) -> Result<(), ConfigError> {
        verify(self.offset, self.disabled, self.optional, observed)
    }
}

/// Compare every bit outside `optional`.
fn verify(offset: u16, expected: u16, optional: u16, observed: u16) -> Result<(), ConfigError> {
    if (expected ^ observed) & !optional != 0 {
        return Err(ConfigError::before_effect(ConfigFailure::Readback {
            offset,
            width: 2,
            expected: u32::from(expected),
            observed: u32::from(observed),
        }));
    }
    Ok(())
}

fn with_prefix(error: ConfigError, prefix: ConfigEffect) -> ConfigError {
    match error.effect {
        ConfigEffect::None => ConfigError {
            effect: prefix,
            ..error
        },
        ConfigEffect::WriteMayHaveApplied { .. } => error,
    }
}
