// SPDX-License-Identifier: GPL-2.0-or-later

use super::*;
use stela_contracts::Chip;
use stela_soc_data::board::assigned::RegisterRange;

fn original() -> DeviceTree {
    let mut pmgr = Node::named("pmgr").unwrap();
    pmgr.set_property("compatible", b"pmgr1,t8112\0".to_vec())
        .unwrap();
    pmgr.set_property(CLUSTERS, SOURCE.to_vec()).unwrap();
    pmgr.set_property("reg", [0x21; 32].to_vec()).unwrap();
    pmgr.set_property("hw-dpe-reg", [0x37; 40].to_vec())
        .unwrap();
    let mut devices = Vec::new();
    for (index, name) in CORE_NAMES.into_iter().chain(CPM_NAMES).enumerate() {
        let mut row = [0; DEVICE_BYTES];
        row[2] = CORE_DOMAIN[if index < 8 { index / 4 } else { index - 8 }];
        row[10] = index as u8;
        row[11] = 5;
        row[26..28].copy_from_slice(&((index + 1) as u16).to_le_bytes());
        row[DEVICE_NAME..DEVICE_NAME + name.len()].copy_from_slice(name);
        devices.extend(row);
    }
    let mut virtual_row = [0; DEVICE_BYTES];
    virtual_row[0] = VIRTUAL;
    virtual_row[DEVICE_NAME..DEVICE_NAME + 8].copy_from_slice(b"PCPU_RAM");
    devices.extend(virtual_row);
    pmgr.set_property(DEVICES, devices).unwrap();
    let mut domains = Vec::new();
    for (id, kind, name) in [
        (1, 0, b"SOC\0\0".as_slice()),
        (2, 1, b"ECPU\0"),
        (5, 1, b"PCPU\0"),
    ] {
        let mut row = [0; PERF_DOMAIN_BYTES];
        row[2] = kind;
        row[3] = id;
        row[12..17].copy_from_slice(name);
        domains.extend(row);
    }
    pmgr.set_property(PERF_DOMAINS, domains).unwrap();
    let mut io = Node::named("arm-io").unwrap();
    io.children.push(pmgr);

    let mut cpus = Node::named("cpus").unwrap();
    for (index, reg) in AFFINITIES.into_iter().enumerate() {
        let mut cpu = Node::named(&format!("cpu{index}")).unwrap();
        cpu.set_property("cpu-id", (index as u32).to_le_bytes().to_vec())
            .unwrap();
        cpu.set_property("reg", reg.to_le_bytes().to_vec()).unwrap();
        let kind = if index < 4 { b"E\0" } else { b"P\0" };
        cpu.set_property("cluster-type", kind.to_vec()).unwrap();
        cpus.children.push(cpu);
    }

    let mut root = Node::named("device-tree").unwrap();
    root.children.extend([cpus, io]);
    DeviceTree { root }
}

fn profile() -> ResourceProfile {
    ResourceProfile {
        chip: Chip::AppleM2,
        soc: 0x8112,
        board: 0x24,
        input_set_sha256: [1; 32],
        adapted_set_sha256: None,
    }
}

fn mappings(indices: impl IntoIterator<Item = usize>) -> Vec<CpuIdMapping> {
    indices
        .into_iter()
        .enumerate()
        .map(|(logical, index)| CpuIdMapping {
            slot: GuestSlotId(logical as u32),
            path: format!("{CPUS}/cpu{index}"),
            reg: AFFINITIES[index],
            original_id: index as u32,
            projected_id: logical as u32,
            watchdog_record: [0; 16],
            coresight: RegisterRange { base: 0, bytes: 0 },
        })
        .collect()
}

#[test]
fn original_roster_remains_byte_exact_and_reduced_prefix_preserves_domain_routes() {
    let original = original();
    for (selected, clusters) in [
        (&[0, 4][..], [1u32, 1u32]),
        (&[0, 1, 2, 4], [3, 1]),
        (&[0, 1, 2, 3, 4], [4, 1]),
        (&[0, 1, 2, 3, 4, 5, 6], [4, 3]),
        (&[0, 1, 2, 3, 4, 5, 6, 7], [4, 4]),
    ] {
        let mut candidate = original.clone();
        apply(
            profile(),
            &original,
            &original,
            &mappings(selected.iter().copied()),
            &mut candidate,
        )
        .unwrap();
        assert_eq!(
            bytes(candidate.node(PMGR).unwrap(), CLUSTERS).unwrap(),
            clusters.map(u32::to_le_bytes).concat()
        );
        let before = original.node(PMGR).unwrap();
        let after = candidate.node(PMGR).unwrap();
        let raw_devices = bytes(before, DEVICES).unwrap();
        let raw_domains = bytes(before, PERF_DOMAINS).unwrap();
        let devices = bytes(after, DEVICES).unwrap();
        let domains = bytes(after, PERF_DOMAINS).unwrap();
        for (index, (old, new)) in raw_devices
            .as_chunks::<DEVICE_BYTES>()
            .0
            .iter()
            .zip(devices.as_chunks::<DEVICE_BYTES>().0)
            .enumerate()
        {
            if index < AFFINITIES.len() && !selected.contains(&index) {
                let mut expected = *old;
                expected[PERF_DOMAIN] = 0;
                assert_eq!(*new, expected);
            } else {
                assert_eq!(old, new);
            }
        }
        assert_eq!(domains, raw_domains);
        if selected.len() == AFFINITIES.len() {
            assert_eq!(candidate.serialize(), original.serialize());
        }
        candidate
            .set_property(PMGR, CLUSTERS, SOURCE.to_vec())
            .unwrap();
        candidate
            .set_property(PMGR, DEVICES, raw_devices.to_vec())
            .unwrap();
        assert_eq!(candidate, original);
    }
}

#[test]
fn malformed_or_unqualified_cpu_rows_refuse_before_mutation() {
    let original = original();
    let mut live = original.clone();
    let mut candidate = original.clone();
    let kept = mappings([0, 1, 2, 4]);
    let raw = bytes(original.node(PMGR).unwrap(), DEVICES).unwrap();
    let mut devices = raw.to_vec();
    devices[DEVICE_BYTES * 7 + 26] = 0xff;
    live.set_property(PMGR, DEVICES, devices).unwrap();
    assert!(apply(profile(), &original, &live, &kept, &mut candidate).is_err());
    assert_eq!(candidate, original);

    let mut devices = raw.to_vec();
    devices[DEVICE_BYTES * 10] = 0;
    live.set_property(PMGR, DEVICES, devices).unwrap();
    assert!(apply(profile(), &live, &live, &kept, &mut candidate).is_err());
    assert_eq!(candidate, original);

    let mut live = original.clone();
    live.node_mut(PMGR).unwrap().remove_property(PERF_DOMAINS);
    assert!(apply(profile(), &live, &live, &kept, &mut candidate).is_err());
    assert_eq!(candidate, original);
}

#[test]
fn other_boards_preserve_their_pm_clusters_and_ambiguous_j473_assignments_refuse() {
    let original = original();
    let mut candidate = original.clone();
    let mut other = profile();
    other.board = 0x25;
    apply(other, &original, &original, &mappings(4..8), &mut candidate).unwrap();
    assert_eq!(candidate, original);

    for selected in [
        &[0][..],
        &[4, 5, 6, 7],
        &[0, 1, 2, 3],
        &[1, 0, 2, 4],
        &[0, 2, 3, 4],
        &[0, 1, 2, 4, 5, 6, 7],
    ] {
        assert!(
            apply(
                profile(),
                &original,
                &original,
                &mappings(selected.iter().copied()),
                &mut candidate
            )
            .unwrap_err()
            .to_string()
            .contains("J473 PMGR")
        );
        assert_eq!(candidate, original);
    }

    let mut live = original.clone();
    live.set_property(
        PMGR,
        CLUSTERS,
        [4u32.to_le_bytes(), 3u32.to_le_bytes()].concat(),
    )
    .unwrap();
    assert!(
        apply(profile(), &original, &live, &mappings(0..4), &mut candidate)
            .unwrap_err()
            .to_string()
            .contains("changed before CPU projection")
    );
    assert_eq!(candidate, original);
}
