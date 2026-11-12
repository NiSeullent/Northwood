// SPDX-License-Identifier: GPL-2.0-or-later

//! A generation of the registered GPA-to-host backing map.
use super::{ExternalDmaLoan, RegisteredSystemRam};
use crate::memory::domain::{ExecutionLoanAttempt, MemoryExecutionLoan};
use stela_compiler::context::memory::MemoryStatus;

pub(super) struct RamBinding<'a> {
    pub(super) ram: &'a RegisteredSystemRam,
    pub(super) generation: u64,
}

impl RegisteredSystemRam {
    /// Keep the installed RAM domain through a serialized CPU invocation.
    /// This excludes mapping services; the caller must already have excluded
    /// concurrent raw writers and admitted the CPU's architectural access.
    pub fn cpu_execution_loan(&self) -> Result<MemoryExecutionLoan, MemoryStatus> {
        self.domain.execution_loan()
    }

    /// Apply the same admission rules as [Self::cpu_execution_loan] and
    /// report an active mapping service separately from an admission error.
    pub fn try_cpu_execution_loan(&self) -> Result<ExecutionLoanAttempt, MemoryStatus> {
        self.domain.try_execution_loan()
    }

    pub(super) fn binding(&self) -> Result<RamBinding<'_>, MemoryStatus> {
        Ok(RamBinding {
            ram: self,
            generation: self.generation()?,
        })
    }

    /// Resolve resident guest RAM for a CPU already holding an execution loan
    /// from this exact RAM owner.
    pub fn cpu_binding<'a>(
        &'a self,
        execution: &'a MemoryExecutionLoan,
    ) -> Result<CpuRamBinding<'a>, MemoryStatus> {
        if !execution.belongs_to(&self.domain) {
            return Err(MemoryStatus::Domain);
        }
        Ok(CpuRamBinding {
            binding: self.binding()?,
            _execution: execution,
        })
    }
}

impl RamBinding<'_> {
    pub(super) fn validate(&self) -> Result<(), MemoryStatus> {
        if self.ram.generation()? == self.generation {
            Ok(())
        } else {
            Err(MemoryStatus::Domain)
        }
    }

    pub(super) fn check_range(&self, physical: u64, bytes: usize) -> Result<(), MemoryStatus> {
        self.ram.check_frame_residency(physical, bytes)?;
        self.ram.preflight_geometry(physical, bytes)
    }

    fn physical_run(&self, guest: u64, max_bytes: u64) -> Result<Option<(u64, u64)>, MemoryStatus> {
        if max_bytes == 0 {
            return Ok(None);
        }
        let Some((region, offset)) = self.ram.region(guest) else {
            return Ok(None);
        };
        let mut bytes = max_bytes.min((region.bytes - offset) as u64);
        if let Some(frame) = self.ram.nonresident_frame(guest, bytes as usize) {
            if frame <= guest {
                return Err(MemoryStatus::Nonresident);
            }
            bytes = frame - guest;
        }
        Ok(Some((region.host_pa + offset as u64, bytes)))
    }
}

/// Registered RAM placement held for the lifetime of an execution loan.
pub struct CpuRamBinding<'a> {
    binding: RamBinding<'a>,
    _execution: &'a MemoryExecutionLoan,
}

/// Classification of a complete physical range under a live RAM binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RamPageKind {
    Resident,
    Nonresident { frame: u64 },
    Unmapped,
}

impl CpuRamBinding<'_> {
    pub fn generation(&self) -> u64 {
        self.binding.generation
    }

    /// Return the contiguous resident host physical run starting at `guest`,
    /// limited by `max_bytes`. Empty requests and unregistered addresses
    /// return `None`; a nonresident starting frame returns `Nonresident`.
    /// Placement metadata grants no CPU dereference or mapping permission.
    pub fn physical_run(
        &self,
        guest: u64,
        max_bytes: u64,
    ) -> Result<Option<(u64, u64)>, MemoryStatus> {
        self.binding.validate()?;
        self.binding.physical_run(guest, max_bytes)
    }

    /// Classify the entire nonempty range without reading its backing. A
    /// partially registered range cannot be classified as an unmapped page.
    pub fn classify_page(&self, guest: u64, bytes: usize) -> Result<RamPageKind, MemoryStatus> {
        self.binding.validate()?;
        let offset = bytes.checked_sub(1).ok_or(MemoryStatus::InvalidRequest)?;
        let offset = u64::try_from(offset).map_err(|_| MemoryStatus::Translation)?;
        let last = guest.checked_add(offset).ok_or(MemoryStatus::Translation)?;
        let ram = self.binding.ram;
        let index = ram
            .regions
            .partition_point(|region| region.source_pa + region.bytes as u64 <= guest);
        let Some(first) = ram
            .regions
            .get(index)
            .filter(|region| region.source_pa <= last)
        else {
            return Ok(RamPageKind::Unmapped);
        };
        if guest < first.source_pa {
            return Err(MemoryStatus::Translation);
        }
        if last >= first.source_pa + first.bytes as u64 {
            ram.preflight_geometry(guest, bytes)?;
        }
        Ok(match ram.nonresident_frame(guest, bytes) {
            Some(frame) => RamPageKind::Nonresident { frame },
            None => RamPageKind::Resident,
        })
    }
}

impl ExternalDmaLoan<'_> {
    pub fn generation(&self) -> u64 {
        self.binding.generation
    }

    /// Visit the exact host physical runs in request order under the DMA pins.
    /// Use the visitor only for private planning and discard it on error.
    /// Keep the loan through publication; this does not authorize DMA or prove
    /// that a previously published mapping can be retired.
    pub fn visit_physical_runs(&self, mut visit: impl FnMut(u64, u64)) -> Result<(), MemoryStatus> {
        self.binding.validate()?;
        for range in self.ranges {
            let mut guest = range.address;
            let mut remaining = range.bytes;
            while remaining != 0 {
                let (host, bytes) = self
                    .binding
                    .physical_run(guest, remaining)?
                    .ok_or(MemoryStatus::Translation)?;
                visit(host, bytes);
                remaining -= bytes;
                if remaining != 0 {
                    guest += bytes;
                }
            }
        }
        self.binding.validate()
    }
}

#[cfg(test)]
impl RegisteredSystemRam {
    /// Seed the modeled post-release state before any execution loan. The
    /// production retained-frame token keeps its writer service while absent.
    pub(crate) fn model_set_nonresident(&self, physical: u64) {
        self.model_set_frame_residency(physical, false);
    }

    pub(crate) fn model_set_resident(&self, physical: u64) {
        self.model_set_frame_residency(physical, true);
    }

    fn model_set_frame_residency(&self, physical: u64, resident: bool) {
        use super::descriptor_frames::FRAME_BYTES;

        assert!(physical.is_multiple_of(FRAME_BYTES as u64));
        let _service = self.try_service().unwrap().expect("idle modeled RAM");
        let (region, offset) = self.region(physical).expect("registered modeled frame");
        assert!(FRAME_BYTES <= region.bytes - offset);
        assert!((region.host_pa + offset as u64).is_multiple_of(FRAME_BYTES as u64));
        let frames = self
            .descriptor_frames
            .as_ref()
            .expect("tracked modeled RAM");
        let _stripe = frames.try_close(physical).expect("idle modeled frame");
        let index = region.wire_index(physical);
        if resident {
            frames.mark_resident(index);
        } else {
            frames.mark_nonresident(index);
        }
    }
}

#[cfg(test)]
mod tests;
