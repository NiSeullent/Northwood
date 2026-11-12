//! Selected original T8112 resources. Membership is derived by the same
//! normal-mode resolver for each PE, preserving the running boot PE.
//! Capacities stay inside the complete original eight-PE/two-complex envelope;
//! this adapter cannot expand an Apple source profile to additional host cores.
use super::*;
use crate::Result;
use stela_contracts::refuse;

const J473_AFFINITIES: [u32; 8] = [0, 1, 2, 3, 0x100, 0x101, 0x102, 0x103];

/// AppleT8110PMGR::restoreHW unconditionally enables both CPU complexes
/// (26B123 mac14g 0xfffffe0009fbc998 and 0xfffffe0009fbc9d8). The legacy
/// guest device-table walker assigns CPU numbers in physical record order.
fn j473_reduced_affinities(
    root: &Node<'_>,
    nodes: &[Node<'_>],
    board: u32,
    boot_reg: u32,
    count: usize,
) -> Result<Option<Vec<u32>>> {
    if board != 0x24 || count == J473_AFFINITIES.len() {
        return Ok(None);
    }
    let pmgr = root
        .child("arm-io")
        .and_then(|io| io.child("pmgr"))
        .ok_or(refuse!("assigned J473 PMGR is absent"))?;
    let Some(clusters) = pmgr.property("clusters") else {
        return Ok(None);
    };
    if clusters != [4, 0, 0, 0, 4, 0, 0, 0]
        || pmgr.property("compatible") != Some(b"pmgr1,t8112\0")
        || pmgr.property("soc-clusters").is_some()
        || pmgr.property("acc-harvesting").is_some()
    {
        return Err(refuse!("assigned J473 PMGR complex roster is unqualified"));
    }
    if count < 2 {
        return Err(refuse!(
            "assigned J473 PMGR restore requires a worker for each of its two CPU complexes"
        ));
    }
    if boot_reg != J473_AFFINITIES[0] {
        return Err(refuse!(
            "assigned J473 PMGR reduction requires the original first efficiency CPU as BSP"
        ));
    }
    for (index, (node, reg)) in nodes.iter().zip(J473_AFFINITIES).enumerate() {
        let kind = if reg < 0x100 { b"E\0" } else { b"P\0" };
        if word(node, "reg")? != reg
            || word(node, "cpu-id")? != index as u32
            || property(node, "cluster-type")? != kind
        {
            return Err(refuse!(
                "assigned J473 PMGR reduction differs from the original CPU order"
            ));
        }
    }
    let efficiency = (count - 1).min(4);
    let performance = count - efficiency;
    Ok(Some(
        J473_AFFINITIES[..efficiency]
            .iter()
            .chain(&J473_AFFINITIES[4..4 + performance])
            .copied()
            .collect(),
    ))
}

impl<'a> BspSource<'a> {
    /// Construct metadata for `count` assigned workers, BSP first. The
    /// caller must keep a started physical roster and join native permission.
    /// Selection keeps the source boot PE first and each complex contiguous
    /// for XNU's L2-sharing interval consumer, preserving source order within
    /// each complex after the boot member.
    /// No running state, power ACK, interrupt or executor is manufactured here.
    pub fn t8112_assigned(
        original: &'a [u8],
        prepared: &[u8],
        identity: SourceIdentity,
        boot_reg: u32,
        acc_harvesting: u32,
        count: usize,
    ) -> Result<Self> {
        let root =
            Node::root(original).ok_or(refuse!("assigned original source tree is malformed"))?;
        let cpus = root
            .child("cpus")
            .ok_or(refuse!("assigned original source lacks CPUs"))?;
        let nodes = cpus
            .children()
            .ok_or(refuse!("assigned original CPU list is malformed"))?;
        // The normal-mode association resolver checks this original family
        // as two four-core complexes. Counts beyond this envelope require a
        // separate whole-consumer capacity proof, never cloning a source slot.
        if nodes.len() != 8 || count == 0 || count > nodes.len() {
            return Err(refuse!(
                "assigned T8112 worker count exceeds the qualified source envelope"
            ));
        }
        let regs = match j473_reduced_affinities(
            &root,
            &nodes,
            identity.profile.board,
            boot_reg,
            count,
        )? {
            Some(regs) => regs,
            None => {
                let mut regs = vec![boot_reg];
                let boot_complex = boot_reg >> 8;
                for same_complex in [true, false] {
                    for node in &nodes {
                        let reg = word(node, "reg")?;
                        if reg != boot_reg
                            && (reg >> 8 == boot_complex) == same_complex
                            && regs.len() < count
                        {
                            regs.push(reg);
                        }
                    }
                }
                regs
            }
        };
        let mut result = Self::t8112(original, prepared, identity, boot_reg, acc_harvesting)?;
        result.remap_member(0);
        for (ordinal, &reg) in regs.iter().enumerate().skip(1) {
            let mut member =
                Self::t8112_member(original, prepared, identity, boot_reg, reg, acc_harvesting)?;
            member.remap_member(ordinal as u32);
            for cluster in member.clusters {
                if let Some(old) = result.clusters.iter().find(|old| old.id == cluster.id) {
                    if old != &cluster {
                        return Err(refuse!(
                            "assigned source cluster metadata conflicts across PEs"
                        ));
                    }
                } else {
                    result.clusters.push(cluster);
                }
            }
            for cache in member.caches {
                if let Some(old) = result.caches.iter().find(|old| old.id == cache.id) {
                    if old != &cache {
                        return Err(refuse!(
                            "assigned source cache metadata conflicts across PEs"
                        ));
                    }
                } else {
                    result.caches.push(cache);
                }
            }
            for power in member.powers {
                if let Some(old) = result.powers.iter_mut().find(|old| old.id == power.id) {
                    for core in power.cores {
                        if old.cores.contains(&core) {
                            return Err(refuse!("assigned source repeats core power membership"));
                        }
                        old.cores.push(core);
                    }
                    for cluster in power.clusters {
                        if !old.clusters.contains(&cluster) {
                            old.clusters.push(cluster);
                        }
                    }
                } else {
                    result.powers.push(power);
                }
            }
            result.kinds.extend(member.kinds);
            result.slots.extend(member.slots);
            result.interrupts.extend(member.interrupts);
            result.starts.extend(member.starts);
        }
        for capacity in &mut result.capacities {
            capacity.maximum = Some(match capacity.kind {
                CapacityKind::KernelClusters
                | CapacityKind::PlatformClusters
                | CapacityKind::KernelPhysicalCluster => 2,
                _ => 8,
            });
            capacity.provenance = "T8112 normal-mode original eight-PE/two-complex source envelope; no source expansion";
        }
        result.provenance = "T8112 selected original normal-mode associations; actual worker and platform adoption required";
        Ok(result)
    }

    fn remap_member(&mut self, ordinal: u32) {
        let slot = GuestSlotId(ordinal);
        let cluster = ClusterId(self.slots[0].reg >> 8);
        self.kinds[0].id = KindId(ordinal);
        self.slots[0].id = slot;
        self.slots[0].kind = Some(KindId(ordinal));
        self.slots[0].cluster = cluster;
        self.slots[0].interrupt = Some(InterruptId(ordinal));
        self.slots[0].start = Some(StartId(ordinal));
        self.clusters[0].id = cluster;
        self.clusters[0].cache = Some(CacheId(cluster.0));
        self.caches[0].id = CacheId(cluster.0);
        self.caches[0].cluster = cluster;
        self.interrupts[0].id = InterruptId(ordinal);
        self.starts[0].id = StartId(ordinal);
        self.powers[0].cores = vec![slot];
        self.powers[1].clusters = vec![cluster];
    }

    /// Original node name for this exact selected resource slot.
    pub fn name_for(&self, slot: GuestSlotId) -> Result<&str> {
        let reg = self
            .slots
            .iter()
            .find(|member| member.id == slot)
            .ok_or(refuse!("assigned source slot is absent"))?
            .reg;
        let root =
            Node::root(self.source).ok_or(refuse!("assigned retained source is malformed"))?;
        let cpus = root
            .child("cpus")
            .ok_or(refuse!("assigned retained source lacks CPUs"))?;
        let nodes = cpus
            .children()
            .ok_or(refuse!("assigned retained CPU list is malformed"))?;
        let node = nodes
            .iter()
            .find(|node| word(node, "reg") == Ok(reg))
            .ok_or(refuse!("assigned retained source PE is absent"))?;
        text(node, "name")
    }
}
