// SPDX-License-Identifier: GPL-2.0-or-later

//! Legacy T8112 PMGR complex roster for the J473 normal-harvesting source.
//!
//! 26B123 mac14g ApplePMGR::initDriver (VA 0xfffffe0009b8e3e0) reads one
//! little-endian u32 per physical complex, using its low u16 as the core
//! count. AppleT8110PMGR::initDriver (0xfffffe0009fbc194) stores XNU's
//! ml_get_topology_info() as its ACC topology. A stale PMGR roster can
//! therefore enumerate more CPU IDs during restore than the pruned XNU
//! topology accepts (setCPUCoreEnergyAccumulator, 0xfffffe0009fc01a4).
//! AppleT8110PMGR::restoreHW (0xfffffe0009fbc998 and 0xfffffe0009fbc9d8)
//! always enables both physical complexes, so each retains a nonempty
//! prefix, and their power-controller records stay intact.
use super::*;
use stela_soc_data::board::assigned::ResourceProfile;

const PMGR: &str = "/arm-io/pmgr";
const CLUSTERS: &str = "clusters";
const DEVICES: &str = "devices";
const PERF_DOMAINS: &str = "perf-domains";
const DEVICE_BYTES: usize = 48;
const PERF_DOMAIN_BYTES: usize = 28;
const DEVICE_NAME: usize = 32;
const VIRTUAL: u8 = 1 << 4;
const CORES_PER_COMPLEX: usize = 4;
const CORE_DOMAIN: [u8; 2] = [2, 5];
const PERF_DOMAIN: usize = 2;
const SOURCE: [u8; 8] = [4, 0, 0, 0, 4, 0, 0, 0];
const AFFINITIES: [u32; 8] = [0, 1, 2, 3, 0x100, 0x101, 0x102, 0x103];
const CORE_NAMES: [&[u8]; 8] = [
    b"ECPU0", b"ECPU1", b"ECPU2", b"ECPU3", b"PCPU0", b"PCPU1", b"PCPU2", b"PCPU3",
];
const CPM_NAMES: [&[u8]; 2] = [b"ECPM", b"PCPM"];

#[cfg(test)]
mod tests;

pub(super) fn apply(
    profile: ResourceProfile,
    original: &DeviceTree,
    live: &DeviceTree,
    cpus: &[CpuIdMapping],
    projected: &mut DeviceTree,
) -> Result<()> {
    if (profile.soc, profile.board) != (0x8112, 0x24) {
        return Ok(());
    }
    let original_cpus = original
        .node(CPUS)
        .ok_or(refuse!("J473 PMGR projection has no original CPUs"))?;
    if original_cpus.children.len() == AFFINITIES.len() && cpus.len() == AFFINITIES.len() {
        return Ok(());
    }
    let original_pmgr = original
        .node(PMGR)
        .ok_or(refuse!("J473 PMGR projection has no original PMGR"))?;
    let live_pmgr = live
        .node(PMGR)
        .ok_or(refuse!("J473 PMGR projection has no live PMGR"))?;
    require!(
        original_pmgr.property(CLUSTERS) == live_pmgr.property(CLUSTERS),
        "J473 PMGR clusters changed before CPU projection"
    );
    if original_pmgr.property(CLUSTERS).is_none() {
        return Ok(());
    }
    require!(
        bytes(original_pmgr, "compatible")? == b"pmgr1,t8112\0"
            && original_pmgr.property("compatible") == live_pmgr.property("compatible")
            && original_pmgr.property("soc-clusters").is_none()
            && live_pmgr.property("soc-clusters").is_none()
            && original_pmgr.property("acc-harvesting").is_none()
            && live_pmgr.property("acc-harvesting").is_none()
            && bytes(original_pmgr, CLUSTERS)? == SOURCE,
        "unqualified J473 PMGR complex roster"
    );
    require!(
        original_cpus.children.len() == AFFINITIES.len() && cpus.len() <= AFFINITIES.len(),
        "J473 PMGR projection requires the complete eight-CPU source"
    );
    for (index, (cpu, reg)) in original_cpus.children.iter().zip(AFFINITIES).enumerate() {
        let kind = if reg < 0x100 { b"E\0" } else { b"P\0" };
        require!(
            word(cpu, "reg")? == reg
                && word(cpu, "cpu-id")? == index as u32
                && bytes(cpu, "cluster-type")? == kind,
            "J473 PMGR complex roster differs from the original CPU order"
        );
    }
    let efficiency = cpus.iter().take_while(|cpu| cpu.reg < 0x100).count();
    let performance = cpus.len() - efficiency;
    require!(
        (1..=CORES_PER_COMPLEX).contains(&efficiency)
            && (1..=CORES_PER_COMPLEX).contains(&performance),
        "reduced J473 PMGR restore requires nonempty efficiency and performance complexes"
    );
    let retained = (0..efficiency).chain(CORES_PER_COMPLEX..CORES_PER_COMPLEX + performance);
    let mut keep = [false; AFFINITIES.len()];
    for (logical, (cpu, source_index)) in cpus.iter().zip(retained).enumerate() {
        require!(
            cpu.reg == AFFINITIES[source_index]
                && cpu.original_id == source_index as u32
                && cpu.projected_id == logical as u32
                // T8112 special-cases logical 6/7 for the last two P-core
                // energy banks, so only original IDs can use those paths.
                && (logical < 6 || source_index == logical),
            "reduced J473 PMGR complex roster requires both original CPU prefixes in order"
        );
        keep[source_index] = true;
    }
    let devices = project_devices(same_bytes(original_pmgr, live_pmgr, DEVICES)?, &keep)?;
    validate_domains(same_bytes(original_pmgr, live_pmgr, PERF_DOMAINS)?)?;
    let clusters = [
        (efficiency as u32).to_le_bytes(),
        (performance as u32).to_le_bytes(),
    ]
    .concat();
    let candidate = projected
        .node_mut(PMGR)
        .ok_or(refuse!("J473 PMGR projection lost the candidate PMGR"))?;
    candidate.set_property(CLUSTERS, clusters)?;
    candidate.set_property(DEVICES, devices)?;
    Ok(())
}

fn same_bytes<'a>(original: &'a Node, live: &Node, key: &str) -> Result<&'a [u8]> {
    require!(
        original.property(key) == live.property(key),
        "J473 PMGR {key} changed before CPU projection",
        key
    );
    bytes(original, key)
}

fn project_devices(raw: &[u8], keep: &[bool; AFFINITIES.len()]) -> Result<Vec<u8>> {
    // In normal mode the ApplePMGR device walker creates CPUCore entries
    // only for physical devices in a CPU performance domain. Domain zero
    // skips that classification (0xfffffe0009b91dd4..0xfffffe0009b91e50,
    // 0xfffffe0009b92414). Both CPM records and every device name, numeric
    // ID, parent and power-state register remain as in the original table.
    let mut expected = CORE_NAMES.into_iter().chain(CPM_NAMES);
    let (records, tail) = raw.as_chunks::<DEVICE_BYTES>();
    require!(
        tail.is_empty() && records.len() >= 10,
        "incomplete J473 PMGR devices"
    );
    for (index, record) in records[..10].iter().enumerate() {
        let name = expected.next().expect("eight cores and two CPMs");
        let complex = if index < CORE_NAMES.len() {
            index / CORES_PER_COMPLEX
        } else {
            index - CORE_NAMES.len()
        };
        require!(
            record[..3] == [0, 0, CORE_DOMAIN[complex]]
                && record[4..10] == [0; 6]
                && record[10..12] == [index as u8, 5]
                && record[15] == 0
                && record[25] == 0
                && record[26..28] == ((index + 1) as u16).to_le_bytes()
                && record[DEVICE_NAME..].starts_with(name)
                && record[DEVICE_NAME + name.len()..]
                    .iter()
                    .all(|&byte| byte == 0),
            "unqualified J473 PMGR physical CPU device {index}",
            index
        );
    }
    require!(
        records[10..].iter().all(|record| {
            let name = &record[DEVICE_NAME..];
            record[0] & VIRTUAL != 0
                || [
                    b"CPU".as_slice(),
                    b"ECPU",
                    b"PCPU",
                    b"MCPU",
                    b"CPM",
                    b"ECPM",
                    b"PCPM",
                    b"MCPM",
                ]
                .iter()
                .all(|prefix| !name.starts_with(prefix))
        }),
        "unqualified additional J473 PMGR CPU device"
    );
    let mut projected = raw.to_vec();
    for (record, &kept) in projected
        .as_chunks_mut::<DEVICE_BYTES>()
        .0
        .iter_mut()
        .zip(keep)
    {
        if !kept {
            record[PERF_DOMAIN] = 0;
        }
    }
    Ok(projected)
}

fn validate_domains(raw: &[u8]) -> Result<()> {
    // Descriptor byte 2 selects CPUComplex allocation (0xfffffe0009b90104).
    let (records, tail) = raw.as_chunks::<PERF_DOMAIN_BYTES>();
    require!(tail.is_empty(), "incomplete J473 PMGR perf-domains");
    let mut complexes = records.iter().filter(|record| record[2] == 1);
    let expected = [(CORE_DOMAIN[0], b"ECPU\0"), (CORE_DOMAIN[1], b"PCPU\0")];
    for (id, name) in expected {
        let record = complexes
            .next()
            .ok_or(refuse!("missing J473 PMGR complex descriptor"))?;
        require!(
            record[3] == id && record[12..].starts_with(name),
            "unqualified J473 PMGR complex descriptor"
        );
    }
    require!(
        complexes.next().is_none(),
        "additional J473 PMGR complex descriptor"
    );
    Ok(())
}
