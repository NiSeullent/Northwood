use super::*;

#[test]
fn fragmented_firmware_can_retry_with_a_smaller_contiguous_heap() {
    let free = 3478 * MIB;
    let preferred = sizes(free, 4, 4).unwrap();
    let fallback = fragmented_sizes(free, 4, 4).unwrap();
    assert_eq!(preferred.profile, Profile::Compact);
    assert_eq!(fallback.profile, Profile::Paged);
    assert_eq!(fallback.heap, 640 * MIB);
    assert!(fallback.heap < preferred.heap);
    assert!(fallback.guest > preferred.guest);
    assert!(fragmented_sizes(MIN_PHYSICAL - 1, 4, 4).is_err());
}
use alloc::string::ToString;

/// Free RAM a PC of `installed` GiB shows Stela: firmware keeps about 3 %.
fn free(installed: u64) -> u64 {
    installed * GIB / 32 * 31
}

#[test]
fn every_host_size_keeps_the_parts_in_bounds_and_the_guest_above_its_floor() {
    for (installed, runners) in [
        (8, 2),
        (8, 4),
        (8, 8),
        (16, 8),
        (32, 8),
        (128, 8),
        (128, 64),
        (2048, 8),
        (2048, 64),
    ] {
        let free = free(installed);
        let sizes = sizes(free, runners, runners).unwrap();
        assert_eq!(sizes.profile, Profile::Standard, "{installed} GiB");
        let floor = HEAP_FLOOR + sizes.code.window - FLOOR_WINDOW;
        assert!((floor..=HEAP_CEILING.max(floor)).contains(&sizes.heap));
        assert!(sizes.guest >= GUEST_FLOOR);
        assert_eq!(
            sizes.heap + sizes.code.bytes + sizes.pool + sizes.guest,
            free
        );
        // The pool maps each sharer's guest DRAM once, or its base and a
        // halves' sizes if less, up to its cap.
        let sharers = (runners as u64).min(POOL_SHARERS);
        let (lower, upper) = pool_halves(sizes.guest);
        let each = (sizes.guest / POOL_SHARE).min(lower + upper);
        let wanted = sharers * each;
        let cap = sharers * POOL_PER_RUNNER;
        let floor = POOL_FLOOR.max(runners as u64 * POOL_RUNNER_FLOOR);
        assert!(
            sizes.pool == cap || sizes.pool + ALIGNMENT * 2 >= wanted,
            "{installed} GiB"
        );
        assert!(sizes.pool <= wanted.max(floor) || sizes.pool == cap);
    }
}

#[test]
fn an_8_gib_host_gives_the_guest_what_the_parts_leave() {
    let sizes = sizes(free(8), 8, 8).unwrap();
    // The floor less the 16 MiB the 496 MiB window lacks of 512 MiB.
    assert_eq!(sizes.code.window, 496 * MIB);
    assert_eq!(sizes.heap, HEAP_FLOOR - 16 * MIB);
    assert_eq!(sizes.code.bytes, 512 * MIB);
    // 8 runners' sixty-fourths of the guest's 4 GiB, not of all 7.75 GiB.
    assert_eq!(sizes.pool, 512 * MIB);
    assert_eq!(sizes.guest, 4112 * MIB);
}

#[test]
fn a_16_gib_host_takes_96_mib_per_runner_for_its_window_and_raises_the_heap_floor() {
    let sizes = sizes(free(16), 8, 8).unwrap();
    // 8 runners' 96 MiB (768 MiB, under a sixteenth's 992 MiB), and each
    // executor's 2 MiB of its own.
    assert_eq!(sizes.code.window, 768 * MIB);
    assert_eq!(sizes.code.bytes, 784 * MIB);
    // The floor plus a byte per window byte past 512 MiB: a twenty-fourth,
    // the window and 16 MiB per PE are under it.
    assert_eq!(sizes.heap, HEAP_FLOOR + 256 * MIB);
    assert!(sizes.guest > 10 * GIB);
}

/// A 12 GiB host with 8 cores (11514 MiB free) takes a 704 MiB window,
/// 192 MiB more than 64 MiB per runner gave, and as much more heap.
#[test]
fn a_12_gib_8_core_host_takes_a_larger_window_and_heap_from_its_guest() {
    let sizes = sizes(11514 * MIB, 8, 8).unwrap();
    assert_eq!(
        [sizes.heap, sizes.code.bytes, sizes.pool, sizes.guest].map(|b| b / MIB),
        [3008, 720, 722, 7064]
    );
}

/// A host whose guest the larger window would put under its floor keeps
/// 64 MiB per runner, and boots where it booted before.
#[test]
fn a_host_short_of_the_larger_window_falls_back_to_the_smaller() {
    // Two runners near the floor: 192 MiB of window would leave the guest
    // under 3 GiB, 128 MiB does not.
    let short = (5 * GIB / MIB..8 * GIB / MIB)
        .step_by(16)
        .map(|free| free * MIB)
        .find(|&free| {
            let large = estimate_with(free, 2, 2, WINDOW_PER_RUNNER).unwrap();
            let small = estimate_with(free, 2, 2, SMALL_WINDOW_PER_RUNNER).unwrap();
            large.guest < GUEST_FLOOR && small.guest >= GUEST_FLOOR
        })
        .expect("a host the fallback lets boot");
    let sizes = sizes(short, 2, 2).unwrap();
    assert_eq!(sizes.code.window, 2 * SMALL_WINDOW_PER_RUNNER);
    assert!(sizes.guest >= GUEST_FLOOR);
    // With a little more RAM the larger window fits.
    let roomy = sizes_of(short + 512 * MIB);
    assert_eq!(roomy.code.window, 2 * WINDOW_PER_RUNNER);
}

fn sizes_of(free: u64) -> Sizes {
    sizes(free, 2, 2).unwrap()
}

#[test]
fn a_128_gib_host_grows_the_heap_with_its_share_window_and_pes() {
    let sizes = sizes(free(128), 8, 8).unwrap();
    let heap = free(128) / HEAP_SHARE + HEAP_PER_WINDOW * 768 * MIB + 8 * PE_HEAP;
    assert_eq!(sizes.heap, heap / ALIGNMENT * ALIGNMENT);
    assert!(HEAP_FLOOR < sizes.heap && sizes.heap < HEAP_CEILING);
    assert_eq!(sizes.code.bytes, 784 * MIB);
    // Each runner's halves for the guest's 115 GiB, not a 64th.
    assert_eq!(sizes.pool, 2022 * MIB);
    assert!(sizes.pool < 8 * POOL_PER_RUNNER);
    assert!(sizes.guest > 100 * GIB);
}

/// Heap, code, pool and guest MiB for `runners` PEs at `free` GiB. At 4 GiB
/// the compact profile fits 2 and 8 runners, and requests for 24 or 64 use
/// the paged profile with four workers.
#[test]
fn stelas_share_stops_growing_per_pe_past_what_each_pe_takes() {
    for (runners, free, expected) in [
        (2, 4, [1518, 68, 76, 2434]),
        (2, 8, [2496, 196, 166, 5334]),
        (2, 16, [2496, 196, 198, 13494]),
        (2, 30, [2496, 196, 240, 27788]),
        (2, 128, [5684, 196, 524, 124668]),
        (8, 4, [1528, 144, 268, 2156]),
        (8, 8, [2816, 528, 538, 4310]),
        (8, 16, [3072, 784, 776, 11752]),
        (8, 30, [3072, 784, 942, 25922]),
        (8, 128, [6356, 784, 2068, 121864]),
        (24, 4, [1152, 72, 64, 2792]),
        (24, 8, [2816, 560, 768, 4048]),
        (24, 16, [3328, 1072, 770, 11214]),
        (24, 30, [4224, 1968, 916, 23612]),
        (24, 128, [7892, 2096, 2034, 119050]),
        (64, 4, [1152, 72, 64, 2792]),
        (64, 8, [3520, 640, 1024, 3008]),
        (64, 16, [3328, 1152, 2048, 9856]),
        (64, 30, [4224, 2048, 2048, 22400]),
        (64, 128, [8532, 2176, 2048, 118316]),
    ] {
        let sizes = sizes(free * GIB, runners, runners).unwrap();
        let split = [sizes.heap, sizes.code.bytes, sizes.pool, sizes.guest].map(|b| b / MIB);
        assert_eq!(split, expected, "{runners} PEs, {free} GiB");
        // The pool's total stays within an eighth of the guest, or each
        // reserved runner's floor, whatever the requested PE count.
        assert!(
            sizes.pool <= (sizes.guest / 8).max(sizes.cores as u64 * 32 * MIB),
            "{runners} {free}"
        );
    }
}

/// The splits of measured boots' free RAM on 8 PEs, and the guest MiB each
/// boot gave:
/// - the 8 GiB VM, `HOST MEMORY free=7354M heap=2816M code=464M pool=452M
///   guest=3622M`: 58 MiB more guest;
/// - an 8 GB PC (7544 MiB free): 44 MiB more;
/// - the 12 GiB VM, `free=11514M heap=3008M code=720M pool=864M
///   guest=6922M`: 42 MiB more;
/// - the 16 GiB VM, `free=15710M heap=3280M code=992M pool=1032M
///   guest=10406M`: 596 MiB more;
/// - the 32 GB PC, `free=30242M heap=3672M code=1040M pool=2836M
///   guest=22694M`: 2726 MiB more.
#[test]
fn measured_boots_split_as_expected() {
    for (free, expected) in [
        (7354, [2752, 464, 458, 3680]),
        (HOST_8_GB / MIB, [2768, 480, 476, 3820]),
        (11514, [3008, 720, 722, 7064]),
        (15710, [3072, 784, 768, 11086]),
        (30242, [3072, 784, 938, 25448]),
    ] {
        let sizes = sizes(free * MIB, 8, 8).unwrap();
        assert_eq!(
            [sizes.heap, sizes.code.bytes, sizes.pool, sizes.guest].map(|b| b / MIB),
            expected,
            "{free} MiB free"
        );
    }
}

/// Installed GiB and cores to the split, and the guest MiB the sizing of
/// v0.1.0 gave (8 GiB: +372 at 2 cores, +226 at 4, +16 at 8; 16 GiB: +560
/// at 4, +878 at 8; 32 GiB: +2990; 64 GiB: +8160; 128 GiB: +12302).
#[test]
fn host_sizes_split_as_expected() {
    for (installed, cores, expected, before) in [
        (8, 2, [2496, 196, 158, 5086], 4714),
        (8, 4, [2688, 392, 284, 4572], 4346),
        (8, 8, [2800, 512, 512, 4112], 4096),
        (16, 4, [2688, 392, 392, 12400], 11800),
        (16, 8, [3072, 784, 770, 11246], 10284),
        (32, 8, [3072, 784, 954, 26934], 23920),
        (64, 8, [3540, 784, 1316, 57848], 49784),
        (128, 8, [6186, 784, 2022, 117984], 106012),
    ] {
        let sizes = sizes(free(installed), cores, cores).unwrap();
        let split = [sizes.heap, sizes.code.bytes, sizes.pool, sizes.guest].map(|b| b / MIB);
        assert_eq!(split, expected, "{installed} GiB, {cores} cores");
        assert!(split[3] > before, "{installed} GiB, {cores} cores");
        // From 16 GiB up each runner's user half holds at least the 52 MiB
        // the 12 GiB VM filled, beside its kernel half.
        if installed >= 16 {
            let per = sizes.pool / cores as u64;
            assert!(per - upper_half(per, sizes.guest, 4096) >= 52 * MIB);
        }
    }
}

#[test]
fn a_2_tib_host_caps_every_part() {
    let sizes = sizes(free(2048), 8, 8).unwrap();
    assert_eq!(sizes.heap, HEAP_CEILING);
    assert_eq!(sizes.code.bytes, 784 * MIB);
    assert_eq!(sizes.pool, 8 * POOL_PER_RUNNER);
    assert_eq!(sizes.guest, free(2048) - 24 * GIB - 784 * MIB);
}

#[test]
fn the_heap_grows_with_the_host_between_its_bounds() {
    let small = sizes(free(32), 8, 8).unwrap().heap;
    let large = sizes(free(64), 8, 8).unwrap().heap;
    assert!(HEAP_FLOOR < small && small < large && large < HEAP_CEILING);
}

#[test]
fn below_two_gib_refuses_with_the_host_minimum() {
    let refusal = sizes(MIN_PHYSICAL - ALIGNMENT, 8, 8).unwrap_err();
    assert_eq!(
        refusal.to_string(),
        "this PC has 1.9 GB of RAM free; Stela needs at least 2 GB free to start the guest"
    );
    assert!(sizes(MIN_PHYSICAL - 1, 1, 1).is_err());
    assert_eq!(sizes(MIN_PHYSICAL, 1, 1).unwrap().profile, Profile::Paged);
}

#[test]
fn paged_reserves_backed_ram_and_valid_native_partitions_from_two_gib() {
    use crate::native_resources::partition;
    use crate::topology::MAX_CORES;

    for cores in [1, 2, 4, 6, 8, 16, MAX_CORES] {
        for runners in [1, cores] {
            for free in (MIN_PHYSICAL..=4 * GIB).step_by(ALIGNMENT as usize) {
                let sizes = sizes(free, cores, runners).unwrap();
                if sizes.profile != Profile::Paged {
                    assert_eq!(sizes.cores, cores);
                    continue;
                }
                assert_eq!(sizes.cores, cores.min(4));
                assert!(
                    sizes.guest >= GIB,
                    "{free} bytes, {cores} cores, {runners} runners"
                );
                assert!(sizes.heap.is_multiple_of(ALIGNMENT));
                assert!(sizes.code.bytes.is_multiple_of(ALIGNMENT));
                assert!(sizes.pool.is_multiple_of(ALIGNMENT));
                assert!(sizes.guest.is_multiple_of(ALIGNMENT));
                assert_eq!(
                    sizes.heap
                        + sizes.code.bytes
                        + sizes.pool
                        + sizes.guest
                        + PAGED_HANDOFF_RESERVE,
                    free
                );
                let resident = guest_dram(4 * GIB, sizes.guest);
                assert_eq!(resident.reported, sizes.guest);
                assert_eq!(pack_budget(free, cores, runners), 0);
                let running = runners.min(sizes.cores);
                for slot in 0..sizes.cores {
                    partition(sizes.cores, running, slot, sizes.code.bytes, sizes.pool).unwrap();
                }
            }
        }
    }
}

#[test]
fn lenovo_reported_free_ram_gets_a_paged_budget() {
    let sizes = sizes(2764 * MIB, 4, 4).unwrap();
    assert_eq!(sizes.profile, Profile::Paged);
    assert_eq!(sizes.cores, 4);
    assert_eq!(
        [sizes.heap, sizes.code.bytes, sizes.pool, sizes.guest].map(|bytes| bytes / MIB),
        [640, 72, 64, 1972]
    );
}

#[test]
fn paged_reaches_the_compact_guest_floor_before_the_boundary() {
    for (cores, threshold_mib) in [(2, 3300), (4, 3336)] {
        let threshold = threshold_mib * MIB;
        let below = sizes(threshold - ALIGNMENT, cores, cores).unwrap();
        let at = sizes(threshold, cores, cores).unwrap();
        assert_eq!(below.profile, Profile::Paged);
        assert_eq!(at.profile, Profile::Compact);
        assert_eq!(below.guest, GUEST_COMPACT_FLOOR);
        assert_eq!(at.guest, GUEST_COMPACT_FLOOR);
        assert!(below.heap < at.heap);
    }
}

/// A 4 GB PC (3968 MiB free, the firmware keeping 3 %) boots in the compact
/// profile on any core count it has, every part at or above its floor and
/// the guest above its 2 GiB.
#[test]
fn a_4_gb_pc_boots_compact() {
    for cores in [1, 2, 4, 6, 8] {
        let sizes = sizes(free(4), cores, cores).unwrap();
        assert_eq!(sizes.profile, Profile::Compact, "{cores} cores");
        assert!(
            sizes.guest >= GUEST_COMPACT_FLOOR,
            "{cores} cores: {sizes:?}"
        );
        assert!(
            sizes.heap >= HEAP_COMPACT_FLOOR + sizes.code.window + cores as u64 * PE_HEAP,
            "{cores} cores: {sizes:?}"
        );
        assert!(sizes.pool >= POOL_FLOOR.max(cores as u64 * POOL_COMPACT_RUNNER_FLOOR));
        assert!(sizes.code.window <= 128 * MIB);
        assert!(sizes.heap.is_multiple_of(ALIGNMENT));
        assert!(sizes.pool.is_multiple_of(ALIGNMENT));
        assert_eq!(
            sizes.heap + sizes.code.bytes + sizes.pool + sizes.guest,
            free(4),
            "{cores} cores"
        );
    }
    // Four cores: the 64 MiB window and 8 MiB of private code, the pool's
    // 64 MiB floor, the heap's 1152 MiB floor (1 GiB, the window's records
    // and 16 MiB per PE) and the guest's 2 GiB leave 632 MiB, split between
    // the heap and the guest's side; the pool takes its sixty-fourths of
    // that side.
    let sizes = sizes(free(4), 4, 4).unwrap();
    assert_eq!(
        [sizes.heap, sizes.code.bytes, sizes.pool, sizes.guest].map(|b| b / MIB),
        [1468, 72, 142, 2286]
    );
}

/// A 4 GB PC whose firmware keeps more (3.5 GiB free) still boots on four
/// cores, the guest at its floor and a little more.
#[test]
fn a_tight_4_gb_pc_keeps_the_guest_its_floor() {
    let tight = sizes(3584 * MIB, 4, 4).unwrap();
    assert_eq!(tight.profile, Profile::Compact);
    assert_eq!(
        [tight.heap, tight.code.bytes, tight.pool, tight.guest].map(|b| b / MIB),
        [1276, 72, 130, 2106]
    );
    // Right at the floors everything is a floor: the guest exactly 2 GiB.
    let least = sizes_at(3336 * MIB);
    assert_eq!(least.guest, GUEST_COMPACT_FLOOR);
    assert_eq!(least.heap, 1152 * MIB);
    assert_eq!(least.pool, 64 * MIB);
    assert_eq!(sizes(3335 * MIB, 4, 4).unwrap().profile, Profile::Paged);
}

fn sizes_at(free: u64) -> Sizes {
    sizes(free, 4, 4).unwrap()
}

/// The pack's budget requires the standard profile, so a compact host
/// reads none: its RAM is the guest's and the heap's.
#[test]
fn compact_hosts_read_no_pack() {
    assert_eq!(pack_budget(free(4), 4, 4), 0);
    assert_eq!(pack_budget(free(4), 8, 8), 0);
    assert_eq!(pack_budget(6 * GIB, 8, 8), 0);
}

/// Where the standard profile begins to fit, the compact one hands over
/// without a large step in the heap or the guest.
#[test]
fn the_compact_profile_hands_over_to_the_standard_one() {
    let threshold = (4 * GIB / MIB..8 * GIB / MIB)
        .map(|mib| mib * MIB)
        .find(|&free| estimate_guest(free, 8, 8).is_ok_and(|guest| guest >= GUEST_FLOOR))
        .expect("a host the standard profile fits");
    let below = sizes(threshold - MIB, 8, 8).unwrap();
    let at = sizes(threshold, 8, 8).unwrap();
    assert_eq!(below.profile, Profile::Compact);
    assert_eq!(at.profile, Profile::Standard);
    assert!(below.guest >= GUEST_COMPACT_FLOOR && at.guest >= GUEST_FLOOR);
    assert!(below.heap.abs_diff(at.heap) < GIB, "{below:?} {at:?}");
    assert!(below.guest.abs_diff(at.guest) < GIB, "{below:?} {at:?}");
    // Between the compact floors and that threshold every host boots, and
    // every part keeps within the RAM.
    for free in (3600 * MIB..threshold).step_by(7 * MIB as usize) {
        let sizes = sizes(free, 8, 8).unwrap();
        assert_eq!(sizes.profile, Profile::Compact, "{free}");
        assert!(sizes.guest >= GUEST_COMPACT_FLOOR, "{free}");
        assert_eq!(
            sizes.heap + sizes.code.bytes + sizes.pool + sizes.guest,
            free
        );
    }
}

#[test]
fn parked_executors_add_only_their_reserves() {
    let parked = sizes(free(16), 12, 8).unwrap();
    let rest = free(16) - parked.heap - parked.code.bytes;
    let share = 8 * rest / (POOL_SHARE + 8);
    let grown = 8 * (2048 * (LOWER_BASE + UPPER_BASE) + 3 * rest) / (2048 + 3 * 8);
    let running = share.min(grown) / ALIGNMENT * ALIGNMENT;
    assert_eq!(parked.pool, running + 4 * 4 * MIB);
}

/// The free RAM an 8 GB host with eight cores shows the loader with no pack
/// read (`HOST MEMORY free=7544M`).
const HOST_8_GB: u64 = 7544 * MIB;

#[test]
fn an_8_gb_pc_keeps_a_sixteenth_for_the_pack() {
    let budget = pack_budget(HOST_8_GB, 8, 8);
    assert_eq!(budget, 470 * MIB);
    // An 845 MB pack, which stopped such a host's boots, no longer fits whole,
    // and what is read of it leaves the parts and the guest's floor.
    assert!(budget < 845_496_619);
    let sizes = sizes(HOST_8_GB - budget - PACK_MARGIN, 8, 8).unwrap();
    assert!(sizes.guest > GUEST_FLOOR + 200 * MIB, "{sizes:?}");
    assert_eq!(pack_budget(free(8), 8, 8), 496 * MIB);
}

#[test]
fn a_16_gb_pc_keeps_about_a_gigabyte_for_the_pack() {
    assert_eq!(pack_budget(free(16), 8, 8), 992 * MIB);
}

#[test]
fn a_128_gb_pc_keeps_more_for_the_pack() {
    let budget = pack_budget(free(128), 16, 16);
    assert_eq!(budget, 7936 * MIB);
    let sizes = sizes(free(128) - budget - PACK_MARGIN, 16, 16).unwrap();
    assert!(sizes.guest > 72 * GIB, "{sizes:?}");
}

#[test]
fn the_pack_gets_only_what_the_parts_and_the_guest_leave() {
    // Just above what eight cores need for the standard profile, the pack
    // gets the rest, not its share; below it, nothing, and the boot goes on
    // without a pack (in the compact profile, nearer the floors).
    let needed = (6000..7200)
        .map(|mib| mib * MIB)
        .find(|&free| estimate_guest(free, 8, 8).is_ok_and(|guest| guest >= GUEST_FLOOR))
        .unwrap();
    assert_eq!(pack_budget(needed, 8, 8), 0);
    // 100 MiB more leaves the pack about that much, less the margin: the
    // most that still fits.
    let budget = pack_budget(needed + 100 * MIB, 8, 8);
    assert!((PACK_FLOOR..=100 * MIB).contains(&budget), "{budget}");
    let fits = sizes(needed + 100 * MIB - budget - PACK_MARGIN, 8, 8).unwrap();
    assert_eq!(fits.profile, Profile::Standard);
    let short = sizes(needed + 100 * MIB - budget - ALIGNMENT - PACK_MARGIN, 8, 8).unwrap();
    assert_eq!(short.profile, Profile::Compact);
    assert_eq!(pack_budget(4 * GIB, 8, 8), 0);
    for cores in [1, 2, 4, 8, 16, 64] {
        for free in (2 * GIB..40 * GIB).step_by(37 * MIB as usize) {
            let budget = pack_budget(free, cores, cores);
            assert!(budget.is_multiple_of(ALIGNMENT));
            assert!(budget <= free / PACK_SHARE);
            if budget != 0 {
                assert!(budget >= PACK_FLOOR);
                let left = free - budget - PACK_MARGIN;
                assert!(sizes(left, cores, cores).is_ok(), "{free} {cores}");
            }
        }
    }
}

#[test]
fn the_guest_reports_the_dram_that_backs_it_not_the_hosts() {
    // The 8 GiB bench: 3920 MiB backed of 8190 MiB installed.
    let dram = guest_dram(0x1_ffe0_0000, 0xf500_0000);
    assert_eq!(dram.reported, 0xf500_0000);
    assert_eq!(dram.kept, 0x1_0ae0_0000);
    // `memSize` (the managed range above the carveouts and below the
    // framebuffer) stays within it, and tag storage sized from it lies at
    // the managed range's top.
    let dram_base = 0x100_0000_0000;
    let phys_base = dram_base + 32 * MIB;
    let managed = dram.reported - 32 * MIB - 16 * MIB;
    let value = stela_contracts::tag_storage::hand_off(phys_base, managed, dram.reported).unwrap();
    let base = stela_contracts::tag_storage::base(value);
    assert!(base >= phys_base);
    assert!(base + stela_contracts::tag_storage::bytes(dram.reported) <= phys_base + managed);
    // A host that backs everything it has keeps nothing.
    assert_eq!(guest_dram(8 * GIB, 8 * GIB).kept, 0);
}

/// Each runner's pool splits into the kernel half's 32 MiB and 1024th of
/// the guest and the user half's 48 MiB and 2048th; a pool smaller than
/// both (a sixty-fourth of a small guest, at 8 GiB) splits in their ratio.
#[test]
fn each_runners_pool_gives_the_kernel_half_what_it_needs_and_the_user_half_the_rest() {
    const PAGE: u64 = 4096;
    const ROOTS: u64 = 64 * PAGE;
    for (installed, sized) in [(8, false), (12, true), (16, true), (64, true)] {
        let sizes = sizes(free(installed), 8, 8).unwrap();
        let shadow = sizes.pool / 8 / PAGE * PAGE - ROOTS;
        let upper = sizes.profile.upper_shadow(shadow, sizes.guest, PAGE);
        let lower = shadow - upper;
        let (lower_wanted, upper_wanted) = pool_halves(sizes.guest);
        assert!(upper.is_multiple_of(PAGE));
        assert!(upper <= upper_wanted, "{installed} GiB");
        if sized {
            // Both sizes, less the roots and the pool's 2 MiB rounding.
            assert!(upper + ROOTS + 2 * MIB >= upper_wanted, "{installed} GiB");
            assert!(lower + ROOTS + 2 * MIB >= lower_wanted, "{installed} GiB");
        } else {
            // Short of both: each the part's share in their ratio.
            let ratio = |half: u64| half as f64 / (lower_wanted + upper_wanted) as f64;
            let share = |bytes: u64| bytes as f64 / shadow as f64;
            assert!(
                (share(upper) - ratio(upper_wanted)).abs() < 0.01,
                "{installed} GiB"
            );
            assert!(lower > upper, "{installed} GiB");
        }
    }
    // The 16 GiB host's kernel half: 32 MiB and a MiB per GiB of its
    // 11 GiB guest.
    let sizes = sizes(free(16), 8, 8).unwrap();
    let upper = upper_half(sizes.pool / 8 / 4096 * 4096 - ROOTS, sizes.guest, PAGE);
    assert!((42..=43).contains(&(upper / MIB)));
    // A tiny pool's halves never go below a page each.
    assert_eq!(upper_half(2 * PAGE, 64 * GIB, PAGE), PAGE);
}

#[test]
fn low_memory_profiles_keep_the_even_shadow_pool_split() {
    const PAGE: u64 = 4096;
    const ROOTS: u64 = 64 * PAGE;
    for (runners, profile) in [(8, Profile::Compact), (24, Profile::Paged)] {
        let sizes = sizes(4 * GIB, runners, runners).unwrap();
        assert_eq!(sizes.profile, profile);
        let shadow = sizes.pool / sizes.cores as u64 / PAGE * PAGE - ROOTS;
        assert_eq!(
            profile.upper_shadow(shadow, sizes.guest, PAGE),
            (shadow / PAGE).div_ceil(2) * PAGE,
        );
        assert_eq!(profile.upper_shadow(5 * PAGE, sizes.guest, PAGE), 3 * PAGE);
    }
}
