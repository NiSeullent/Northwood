// SPDX-License-Identifier: GPL-2.0-or-later

//! Geometry for placing boot-bundle source bytes directly in guest RAM.
use alloc::{collections::BTreeSet, vec::Vec};
use stela_formats::boot_bundle::{Manifest, schema::Region};

/// Consecutive bytes in both a bundle file and guest RAM. `head` is the
/// byte offset of the large-page interior within the group.
#[derive(Debug, PartialEq, Eq)]
pub struct Group<'a> {
    pub file_offset: u64,
    pub bytes: u64,
    pub guest_offset: u64,
    pub regions: Vec<&'a Region>,
    pub head: u64,
    pub interior: u64,
}

/// Find the groups of a file that can be staged at matching host and guest
/// offsets. The selected regions must tile the entire source file exactly
/// and may not overlap any other guest region or contain a patch or zero
/// tail. Groups without a large-page interior have `head = interior = 0`.
/// This is a geometry check. The file bytes still require verification.
pub fn plan_file<'a>(
    manifest: &'a Manifest,
    name: &str,
    page: u64,
    large: u64,
) -> Option<Vec<Group<'a>>> {
    let base = manifest.options.dram_base;
    if name.is_empty()
        || !page.is_power_of_two()
        || !large.is_power_of_two()
        || large < page
        || !base.is_multiple_of(large)
    {
        return None;
    }
    let mut files = manifest
        .files
        .iter()
        .filter(|file| file.name.eq_ignore_ascii_case(name));
    let file = files.next()?;
    if file.name != name || files.next().is_some() || file.bytes == 0 {
        return None;
    }

    let mut selected = Vec::new();
    let mut names = BTreeSet::new();
    for region in &manifest.regions {
        if !region.file.eq_ignore_ascii_case(name) {
            continue;
        }
        if region.file != name || region.name.is_empty() || !names.insert(region.name.as_str()) {
            return None;
        }
        selected.push(region);
    }
    if manifest
        .regions
        .iter()
        .any(|region| region.file != name && names.contains(region.name.as_str()))
        || manifest.patches.iter().any(|patch| {
            patch.file.eq_ignore_ascii_case(name) || names.contains(patch.region.as_str())
        })
        || !guest_spans_disjoint(manifest, name)?
    {
        return None;
    }

    selected.sort_unstable_by_key(|region| region.file_offset);
    let mut groups: Vec<Group<'a>> = Vec::new();
    let mut file_end = 0u64;
    for region in selected {
        let guest_offset = region.guest_pa.checked_sub(base)?;
        if region.file_offset != file_end
            || region.bytes == 0
            || region.bytes != region.memory_bytes
            || !region.bytes.is_multiple_of(page)
            || !guest_offset.is_multiple_of(page)
        {
            return None;
        }
        file_end = file_end.checked_add(region.bytes)?;
        if let Some(group) = groups.last_mut()
            && group.guest_offset.checked_add(group.bytes)? == guest_offset
        {
            group.bytes = group.bytes.checked_add(region.bytes)?;
            group.regions.push(region);
        } else {
            groups.push(Group {
                file_offset: region.file_offset,
                bytes: region.bytes,
                guest_offset,
                regions: alloc::vec![region],
                head: 0,
                interior: 0,
            });
        }
    }
    if file_end != file.bytes {
        return None;
    }
    for group in &mut groups {
        let end = group.guest_offset.checked_add(group.bytes)?;
        let (start, interior) = crate::ram::aligned_range(group.guest_offset, end, large);
        if interior != 0 {
            group.head = start - group.guest_offset;
            group.interior = interior;
        }
    }
    Some(groups)
}

/// The selected guest extents cannot share any byte with another region.
/// Other regions retain their existing placement rules.
fn guest_spans_disjoint(manifest: &Manifest, name: &str) -> Option<bool> {
    let mut spans = Vec::with_capacity(manifest.regions.len());
    for region in &manifest.regions {
        if region.memory_bytes != 0 {
            spans.push((
                region.guest_pa,
                region.guest_pa.checked_add(region.memory_bytes)?,
                region.file == name,
            ));
        }
    }
    spans.sort_unstable_by_key(|&(start, _, _)| start);
    let mut prior_end = 0;
    let mut selected_end = 0;
    for (start, end, selected) in spans {
        if start < selected_end || (selected && start < prior_end) {
            return Some(false);
        }
        prior_end = prior_end.max(end);
        if selected {
            selected_end = end;
        }
    }
    Some(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use stela_formats::boot_bundle::{Encoding, Patch, PatchSource, SCHEMA_VERSION, schema::File};

    const FILE: &str = "kernelcache.img";
    const PAGE: u64 = 0x1000;
    const LARGE: u64 = 0x4000;

    fn manifest(base: u64, bytes: u64, regions: &[(&str, u64, u64, u64)]) -> Manifest {
        let regions: Vec<_> = regions
            .iter()
            .map(|(name, offset, guest, bytes)| {
                json!({
                    "name": name, "file": FILE, "file_offset": offset,
                    "guest_pa": guest, "bytes": bytes, "memory_bytes": bytes, "sha256": ""
                })
            })
            .collect();
        let json = json!({
            "schema_version": SCHEMA_VERSION,
            "options": {"dram_base": base, "dram_size": 0x40000000, "phys_base": base},
            "files": [{"name": FILE, "bytes": bytes, "sha256": ""}],
            "regions": regions,
            "entry": {"image": "test", "virtual_address": 0, "physical_address": base},
            "entry_x0": base, "boot_args": {"command_line": ""},
            "top_of_kernel_data": base, "patches": []
        });
        Manifest::parse(&serde_json::to_vec(&json).unwrap()).unwrap()
    }

    fn synthetic() -> Manifest {
        manifest(
            0x8000,
            0x8000,
            &[
                ("third", 0x7000, 0x11000, 0x1000),
                ("second", 0x2000, 0xb000, 0x5000),
                ("first", 0, 0x9000, 0x2000),
            ],
        )
    }

    #[test]
    fn groups_follow_file_order_and_require_guest_adjacency() {
        let mut m = synthetic();
        // File order need not be guest address order.
        m.regions[0].guest_pa = 0x8000;
        let groups = plan_file(&m, FILE, PAGE, LARGE).unwrap();
        assert_eq!(
            groups,
            [
                Group {
                    file_offset: 0,
                    bytes: 0x7000,
                    guest_offset: 0x1000,
                    regions: alloc::vec![&m.regions[2], &m.regions[1]],
                    head: 0x3000,
                    interior: 0x4000,
                },
                Group {
                    file_offset: 0x7000,
                    bytes: 0x1000,
                    guest_offset: 0,
                    regions: alloc::vec![&m.regions[0]],
                    head: 0,
                    interior: 0,
                }
            ]
        );
        assert!(core::ptr::eq(groups[0].regions[0], &m.regions[2]));
    }

    #[test]
    fn bootkc_26b123_geometry_has_three_independent_interiors() {
        let m = manifest(
            0x800000000,
            0x7788000,
            &[
                ("BootKC-rx", 0x1ae4000, 0x802094000, 0x3a90000),
                ("BootKC-bx", 0x5574000, 0x805b24000, 0x8000),
                ("BootKC-ro", 0, 0x805b2c000, 0x1a70000),
                ("BootKC-rs", 0x1a70000, 0x80759c000, 0x74000),
                ("BootKC-rw", 0x557c000, 0x8077a4000, 0x798000),
                ("BootKC-le", 0x5d14000, 0x807f3c000, 0x1a74000),
            ],
        );
        let groups = plan_file(&m, FILE, PAGE, 0x200000).unwrap();
        assert_eq!(groups.len(), 3);
        let geometry: Vec<_> = groups
            .iter()
            .map(|group| {
                (
                    group.file_offset,
                    group.bytes,
                    group.guest_offset,
                    group.head,
                    group.interior,
                )
            })
            .collect();
        assert_eq!(
            geometry,
            [
                (0, 0x1ae4000, 0x5b2c000, 0xd4000, 26 << 20),
                (0x1ae4000, 0x3a98000, 0x2094000, 0x16c000, 56 << 20),
                (0x557c000, 0x220c000, 0x77a4000, 0x5c000, 32 << 20),
            ]
        );
        assert_eq!(
            groups.iter().map(|group| group.interior).sum::<u64>(),
            114 << 20
        );
    }

    #[test]
    fn file_tile_refuses_holes_repeats_zero_tails_and_wrong_lengths() {
        for offset in [0x1000, 0x2001, 0x3000, u64::MAX] {
            let mut m = synthetic();
            m.regions[1].file_offset = offset;
            assert!(plan_file(&m, FILE, PAGE, LARGE).is_none());
        }
        for bytes in [0, 0x7000, 0x9000] {
            let mut m = synthetic();
            m.files[0].bytes = bytes;
            assert!(plan_file(&m, FILE, PAGE, LARGE).is_none());
        }
        let mut m = synthetic();
        m.regions[0].bytes = 0;
        m.regions[0].memory_bytes = 0;
        assert!(plan_file(&m, FILE, PAGE, LARGE).is_none());
        m.regions[0].bytes = 0x1000;
        m.regions[0].memory_bytes = 0x2000;
        assert!(plan_file(&m, FILE, PAGE, LARGE).is_none());
    }

    #[test]
    fn aliases_repeated_region_names_and_selected_patches_refuse() {
        let mut m = synthetic();
        assert!(plan_file(&m, "other.img", PAGE, LARGE).is_none());
        assert!(plan_file(&m, "KernelCache.img", PAGE, LARGE).is_none());
        for name in [FILE, "KernelCache.img"] {
            m.files.push(File {
                name: name.into(),
                bytes: 0x8000,
                sha256: "".into(),
            });
            assert!(plan_file(&m, FILE, PAGE, LARGE).is_none());
            m.files.pop();
        }
        let mut alias = m.regions[0].clone();
        alias.file = "KernelCache.img".into();
        alias.name = "alias".into();
        alias.guest_pa = 0x13000;
        m.regions.push(alias);
        assert!(plan_file(&m, FILE, PAGE, LARGE).is_none());
        m.regions.pop();
        let mut repeated = m.clone();
        repeated.regions[0].name = repeated.regions[1].name.clone();
        assert!(plan_file(&repeated, FILE, PAGE, LARGE).is_none());
        let mut unrelated = m.regions[0].clone();
        unrelated.file = "other.img".into();
        unrelated.guest_pa = 0x13000;
        m.regions.push(unrelated);
        assert!(plan_file(&m, FILE, PAGE, LARGE).is_none());
        m.regions.pop();
        let patch = |file: &str, region: &str| Patch {
            source: PatchSource::DramSize,
            required: true,
            region: region.into(),
            file: file.into(),
            file_offset: 0,
            bytes: 8,
            encoding: Encoding::U64Le,
            field: "test".into(),
            prepared_hex: "".into(),
            rule: "".into(),
        };
        for (file, region) in [
            (FILE, "other"),
            ("KernelCache.img", "other"),
            ("other.img", "first"),
        ] {
            m.patches = alloc::vec![patch(file, region)];
            assert!(plan_file(&m, FILE, PAGE, LARGE).is_none());
        }
        m.patches = alloc::vec![patch("other.img", "other")];
        assert!(plan_file(&m, FILE, PAGE, LARGE).is_some());
    }

    #[test]
    fn guest_overlap_counts_other_regions_full_memory_extents() {
        let mut m = synthetic();
        m.regions[0].guest_pa = 0xa000;
        assert!(plan_file(&m, FILE, PAGE, LARGE).is_none());
        m.regions[0].guest_pa = 0x11000;
        let mut other = m.regions[0].clone();
        other.name = "reserved".into();
        other.file = "other.img".into();
        other.guest_pa = 0x8000;
        other.bytes = 0;
        other.memory_bytes = 0x2000;
        m.regions.push(other);
        assert!(plan_file(&m, FILE, PAGE, LARGE).is_none());
        m.regions[3].memory_bytes = 0x1000;
        assert!(plan_file(&m, FILE, PAGE, LARGE).is_some());
    }

    #[test]
    fn alignment_and_overflow_checks_refuse_malformed_geometry() {
        let m = synthetic();
        for (page, large) in [
            (0, LARGE),
            (3, LARGE),
            (LARGE, PAGE),
            (PAGE, 0),
            (PAGE, 0x3000),
        ] {
            assert!(plan_file(&m, FILE, page, large).is_none());
        }
        let mut base = m.clone();
        base.options.dram_base += PAGE;
        assert!(plan_file(&base, FILE, PAGE, LARGE).is_none());
        for address in [0x7000, 0x9001, u64::MAX - PAGE + 1] {
            let mut m = m.clone();
            m.regions[2].guest_pa = address;
            assert!(plan_file(&m, FILE, PAGE, LARGE).is_none());
        }
        let mut bytes = m;
        bytes.regions[0].bytes -= 1;
        bytes.regions[0].memory_bytes -= 1;
        bytes.files[0].bytes -= 1;
        assert!(plan_file(&bytes, FILE, PAGE, LARGE).is_none());
    }
}
