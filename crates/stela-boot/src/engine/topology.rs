//! The started executors that host guest PEs, joined to the assigned topology
//! `stela_boot_core::assigned_topology` plans from a boot bundle's source.
use crate::Result;
use alloc::vec::Vec;
pub(crate) use stela_boot_core::assigned_topology::*;
use stela_boot_core::{record::Capture, topology as native};
use stela_contracts::refuse;
use stela_soc_data::board::assigned::{AssignedCore, RetainedRoster};

/// The guest's prefix of the started executors. Host-only workers stay out
/// of the Apple topology plan. Each retained AP must have acknowledged its
/// private installed Host and CPUID identity through Executors::start.
pub(crate) struct AssignedRoster<'a> {
    _bsp: BspRoster<'a>,
    _executors: &'a crate::native_smp::executor::Executors,
    cores: Vec<AssignedCore>,
}
impl<'a> AssignedRoster<'a> {
    pub(crate) fn new(
        capture: &'a Capture,
        executors: &'a crate::native_smp::executor::Executors,
        _owner: &super::host::HostOwner,
        guest_workers: usize,
    ) -> Result<Self> {
        if !(1..=executors.started()).contains(&guest_workers) {
            return Err(refuse!(
                "guest worker count exceeds the started host roster"
            ));
        }
        let bsp = BspRoster::current(capture, 0)?;
        let mut cores = Vec::with_capacity(guest_workers);
        cores.extend_from_slice(bsp.cores());
        // The validated slot order selects distinct physical cores before SMT
        // siblings. Source class placement runs within this guest prefix.
        for (index, processor) in executors.processors()[..guest_workers - 1]
            .iter()
            .enumerate()
        {
            let worker = index + 1;
            if processor.slot() != worker {
                return Err(refuse!("assigned started worker roster has a slot gap"));
            }
            let index = *capture
                .selected
                .get(worker)
                .filter(|_| worker < capture.core_count as usize)
                .ok_or(refuse!("assigned worker exceeds captured host threads"))?
                as usize;
            let source = *capture
                .processors
                .get(index)
                .filter(|_| index < capture.processor_count as usize)
                .ok_or(refuse!("assigned worker has no captured source processor"))?;
            if !source.available()
                || source.flags & native::BSP != 0
                || processor.apic_id() != source.apic_id
                || !processor.observation().matches(source)
            {
                return Err(refuse!(
                    "assigned worker differs from actual started AP identity"
                ));
            }
            cores.push(AssignedCore {
                worker,
                package: source.location.package,
                core: source.location.core,
                apic: u32::try_from(source.apic_id)
                    .map_err(|_| refuse!("assigned APIC exceeds transport width"))?,
                started: true,
            });
        }
        Ok(Self {
            _bsp: bsp,
            _executors: executors,
            cores,
        })
    }
}
impl RetainedRoster for AssignedRoster<'_> {
    fn revision(&self) -> u64 {
        1
    }
    fn cores(&self) -> &[AssignedCore] {
        &self.cores
    }
}
