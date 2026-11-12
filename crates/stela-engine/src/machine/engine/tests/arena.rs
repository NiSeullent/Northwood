//! The board's code arena ([`board::Arena`]): a host's whole code window
//! that the engine allocates itself, which every machine of a board shares.
use super::super::board;
use super::super::parallel::{Hub, Kick};
use super::*;

const RET: u32 = 0xd65f_03c0;
const ADD: u32 = 0x9100_0400; // add x0, x0, #1
/// A window of four parts of a page each: a few blocks fill a part.
const WINDOW: usize = 4 * 4096;

struct NoKick;
impl Kick for NoKick {
    fn kick(&self, _pe: usize) {}
}

/// Memory a test's window lies in. Tests run no native code, so it needs
/// no execute permission: its two views are one.
fn window(memory: &mut [u8]) -> CodeWindow {
    let address = memory.as_mut_ptr() as usize;
    CodeWindow {
        exec: address,
        write: address,
        bytes: memory.len(),
    }
}

/// A machine of RET instructions over four pages whose code goes to the
/// window over `memory`, alone.
fn arena_machine(memory: &mut [u8]) -> Machine<FakeHost, NoDevices> {
    arena_machine_of(&vec![RET; 4 * FETCH_PAGE as usize / 4], memory)
}

/// A machine of `words` whose code goes to the window over `memory`.
fn arena_machine_of(words: &[u32], memory: &mut [u8]) -> Machine<FakeHost, NoDevices> {
    let mut machine = machine(words);
    machine.code = Arc::new(board::Code::new(
        board::slot_capacity(Some(WINDOW)),
        chain::MIN_LOOKUP_ENTRIES,
        Some(window(memory)),
    ));
    machine
}

/// Each live block's code address: none is shared, as a part is reused only
/// after its blocks went.
fn live_codes(machine: &Machine<FakeHost, NoDevices>) -> Vec<usize> {
    let blocks = &machine.code.blocks;
    let mut codes: Vec<usize> = (0..blocks.len())
        .filter_map(|slot| blocks.installed(slot))
        .map(|block| block.code.0)
        .collect();
    codes.sort_unstable();
    codes
}

/// The ring's current generation: parts filled so far.
fn generation(machine: &Machine<FakeHost, NoDevices>) -> u32 {
    machine
        .code
        .arena
        .as_ref()
        .expect("an arena")
        .ring
        .generation()
}

#[test]
fn the_serial_lane_installs_into_its_arena_and_retires_parts_in_turn() {
    let mut memory = vec![0u8; WINDOW];
    let range = memory.as_ptr() as usize..memory.as_ptr() as usize + WINDOW;
    let mut machine = arena_machine(&mut memory);
    let base = machine.cpu.pc;
    let mut fetched = 0;
    while generation(&machine) < 8 {
        assert!(fetched < 20_000, "the ring goes round");
        let pc = base + 4 * (fetched % 4096);
        let Ok(Fetched::Block(slot)) = machine.fetch_block(pc, false) else {
            panic!("a block at {pc:#x}")
        };
        let block = machine.code.blocks.installed(slot).expect("a live block");
        assert!(range.contains(&block.code.0));
        fetched += 1;
    }
    assert!(machine.statistics.code_evictions >= 4);
    // Every byte went to the window, none to the host's arena.
    assert!(machine.host.installed.is_empty());
    assert_eq!(machine.statistics.code_discards.iter().sum::<u64>(), 0);
    // No two live blocks share code: a retired part's blocks went first.
    let codes = live_codes(&machine);
    assert!(codes.windows(2).all(|pair| pair[0] != pair[1]));
    // Retired slots served later blocks.
    assert!(machine.code.blocks.len() < fetched as usize);
    // Every indexed block is live.
    let cache = machine.code.cache();
    assert!(
        cache
            .index
            .sweep(None)
            .all(|(_, slot)| machine.code.blocks.record(slot).live())
    );
}

/// Two machines of one board: the second joins the first's code.
fn shared_pair(
    memory: &mut [u8],
) -> (
    Machine<FakeHost, NoDevices>,
    Machine<FakeHost, NoDevices>,
    Arc<Hub>,
) {
    shared_pair_of(&vec![RET; 4 * FETCH_PAGE as usize / 4], memory)
}

/// [`shared_pair`] of machines of `words`.
fn shared_pair_of(
    words: &[u32],
    memory: &mut [u8],
) -> (
    Machine<FakeHost, NoDevices>,
    Machine<FakeHost, NoDevices>,
    Arc<Hub>,
) {
    let hub = Hub::new(2, NoKick);
    let mut first = arena_machine_of(words, memory);
    let mut second = machine(words);
    second.code = Arc::clone(&first.code);
    // A board's machines share their numbering ([`numbering`]).
    second.variants = Arc::clone(&first.variants);
    second.stored_reads = Arc::clone(&first.stored_reads);
    second.stored_writes = Arc::clone(&first.stored_writes);
    first.code_hub = Some(Arc::clone(&hub));
    second.code_hub = Some(Arc::clone(&hub));
    (first, second, hub)
}

#[test]
fn a_block_one_pe_installed_runs_on_another_and_the_first_install_wins() {
    let mut memory = vec![0u8; WINDOW];
    let (mut first, mut second, _hub) = shared_pair(&mut memory);
    let pc = first.cpu.pc;
    let Ok(Fetched::Block(slot)) = first.fetch_block(pc, false) else {
        panic!("a block")
    };
    // The other PE finds it in the board's index or table and compiles
    // nothing.
    let Ok(Fetched::Block(found)) = second.fetch_block(pc, false) else {
        panic!("a block")
    };
    assert_eq!(found, slot);
    assert_eq!(second.statistics.compiled, 0);
    assert_eq!(second.statistics.fetches[4], 0);
    // A PE that compiled the block too, as it would have racing the
    // first, keeps the first's: its own slot goes back, its code stays
    // unreachable until its part goes.
    let block = &first.code.blocks[slot];
    let (physical, variant) = (block.physical, first.block_variant(slot));
    let Ok(Fetched::Block(raced)) = second.compile(pc, physical, variant) else {
        panic!("a block")
    };
    assert_eq!(raced, slot);
    assert_eq!(second.statistics.lost_installs, 1);
    assert_eq!(first.code.blocks.live(), 1);
    assert_eq!(first.code.cache().index.len(), 1);
}

#[test]
fn a_part_is_reused_only_after_every_pe_left_its_section() {
    let mut memory = vec![0u8; WINDOW];
    let (mut first, _second, hub) = shared_pair(&mut memory);
    // PE 1 runs native code: inside its section, where it may hold any
    // code address it read.
    hub.board.grace.enter(1);
    // The counter moves on with each read, so each wait round ends soon;
    // PE 1 leaves at the 50th turn of waiting.
    first.host.counter_step = first.profile.counter_frequency / 50;
    let leaver = Arc::clone(&hub);
    first.host.on_relax = Some(Box::new(move |turns| {
        if turns == 50 {
            leaver.board.grace.leave(1);
        }
    }));
    let base = first.cpu.pc;
    let mut fetched = 0;
    // The ring goes round once while PE 1 is inside: the first part
    // retired after PE 1 entered is the first reuse that waits.
    while generation(&first) < 8 {
        assert!(fetched < 20_000, "the ring goes round");
        let pc = base + 4 * (fetched % 4096);
        let Ok(Fetched::Block(_)) = first.fetch_block(pc, false) else {
            panic!("a block at {pc:#x}")
        };
        fetched += 1;
    }
    // The reuse waited for PE 1 to leave, through overdue rounds, which the
    // board reported once.
    assert!(first.host.relaxes >= 50);
    assert!(!hub.board.overdue(), "the overrun was reported before");
    let reported = first.take_unmodeled();
    assert_eq!(reported.len(), 1);
    let codes = live_codes(&first);
    assert!(codes.windows(2).all(|pair| pair[0] != pair[1]));
}

#[test]
fn an_invalidated_page_leaves_every_pe_and_its_code_stays_until_its_part_goes() {
    let mut memory = vec![0u8; WINDOW];
    let (mut first, mut second, hub) = shared_pair(&mut memory);
    let pc = first.cpu.pc;
    let Ok(Fetched::Block(slot)) = first.fetch_block(pc, false) else {
        panic!("a block")
    };
    assert!(matches!(second.fetch_block(pc, false), Ok(Fetched::Block(found)) if found == slot));
    // PE 1 runs the block while PE 0 posts an IC IVAU of its page; each
    // applies it to its own caches, the board's index forgets the page once.
    hub.board.grace.enter(1);
    let physical = first.code.blocks[slot].physical;
    first.invalidate_code_page(physical);
    second.invalidate_code_page(physical);
    assert_eq!(first.statistics.code_invalidations, 1);
    assert_eq!(second.statistics.code_invalidations, 1);
    assert_eq!(first.code.cache().index.len(), 0);
    // The running block's code is not taken away.
    assert!(first.code.blocks.record(slot).live());
    // Both PEs compile the page again; the second finds the first's.
    let Ok(Fetched::Block(again)) = first.fetch_block(pc, false) else {
        panic!("a block")
    };
    assert_ne!(again, slot);
    assert!(matches!(second.fetch_block(pc, false), Ok(Fetched::Block(found)) if found == again));
    hub.board.grace.leave(1);
}

#[test]
fn a_posted_invalidation_takes_away_a_block_published_after_the_posters_forget() {
    let mut memory = vec![0u8; WINDOW];
    let (mut first, mut second, _hub) = shared_pair(&mut memory);
    let pc = first.cpu.pc;
    let Ok(Fetched::Block(slot)) = first.fetch_block(pc, false) else {
        panic!("a block")
    };
    let physical = first.code.blocks[slot].physical;
    // PE 0 writes the page and forgets it on the board at its IC IVAU.
    // PE 1 read the old bytes before the write and publishes their block
    // only now, as a compile outside its section may.
    first.invalidate_code_page(physical);
    assert_eq!(first.code.cache().index.len(), 0);
    let variant = first.block_variant(slot);
    let Ok(Fetched::Block(stale)) = second.compile(pc, physical, variant) else {
        panic!("a block")
    };
    first.host.ram[0..4].copy_from_slice(&ADD.to_le_bytes());
    assert_eq!(first.code.cache().index.page(physical).count(), 1);
    assert_eq!(first.stale_code().stale, 1);
    // PE 1 applies the post after it published: the block goes.
    second.invalidate_posted_code_page(physical);
    assert_eq!(first.code.cache().index.page(physical).count(), 0);
    assert!(first.code.blocks.record(stale).live(), "its code stays");
    assert_eq!(first.stale_code().stale, 0);
    assert_eq!(second.statistics.code_invalidations, 1);
}

/// A guest access fault at `offset` of a block's native code.
fn access_fault(offset: usize) -> NativeFault {
    NativeFault {
        origin: crate::machine::NativeFaultOrigin::Exception,
        trap: 13,
        error_code: 0,
        spills: stela_compiler::block::HostSpillSnapshot::new(),
        registers: stela_compiler::block::HostGprSnapshot::legacy([0; 16]),
        offset,
        address: 0x7a_80d4_0000,
        write: true,
        cause: FaultCause::NonCanonical,
    }
}

#[test]
fn a_pe_faults_in_a_block_another_pe_retired_and_still_finds_its_rows() {
    let mut memory = vec![0u8; WINDOW];
    let (mut first, mut second, hub) = shared_pair(&mut memory);
    let pc = first.cpu.pc;
    let Ok(Fetched::Block(slot)) = first.fetch_block(pc, false) else {
        panic!("a block")
    };
    // PE 0 runs the block, inside its section, while PE 1 takes the
    // block's part away as the ring retires it.
    hub.board.grace.enter(0);
    let code = first.code.blocks[slot].code.0;
    let arena = first.code.arena.as_ref().expect("an arena");
    let part = (code - arena.window.exec) / arena.part_bytes;
    let range = arena.window.exec + part * arena.part_bytes
        ..arena.window.exec + (part + 1) * arena.part_bytes;
    second.evict_code(range);
    assert!(!first.code.blocks.record(slot).live());
    // The slot still holds the block, and its rows, until PE 0 passes.
    assert!(first.code.blocks.get(slot).is_some());
    assert!(first.code.blocks.recovery().get(&slot).is_some());
    let step = first.native_fault(slot, access_fault(0), 0, None);
    assert!(
        !matches!(&step, Err(Stop::NativeFailure(failure))
            if failure.reason == NativeFailureReason::UnknownCode),
        "{step:?}"
    );
    hub.board.grace.leave(0);
}

#[test]
fn a_lost_install_says_so_and_holds_no_rows() {
    let mut memory = vec![0u8; WINDOW];
    let (mut first, mut second, _hub) = shared_pair(&mut memory);
    let pc = first.cpu.pc;
    let Ok(Fetched::Block(slot)) = first.fetch_block(pc, false) else {
        panic!("a block")
    };
    let (physical, variant) = (first.code.blocks[slot].physical, first.block_variant(slot));
    let lost = first.code.blocks.len();
    second.current = 1;
    let Ok(Fetched::Block(held)) = second.compile(pc, physical, variant) else {
        panic!("a block")
    };
    assert_eq!(held, slot);
    let report = second.board_slot(lost);
    assert!(report.lost && !report.live && !report.rows, "{report:?}");
    assert_eq!(report.installer, 1);
}

#[test]
fn a_device_stub_one_pe_learnt_is_a_stub_on_every_pe() {
    const LDR: u32 = 0xb940_0016; // ldr w22, [x0]
    let mut memory = vec![0u8; WINDOW];
    let (mut first, mut second, _hub) = shared_pair(&mut memory);
    let pc = first.cpu.pc;
    let Ok(Fetched::Block(slot)) = first.fetch_block(pc, false) else {
        panic!("a block")
    };
    let physical = first.code.blocks[slot].physical;
    first.host.ram[0..4].copy_from_slice(&LDR.to_le_bytes());
    second.host.ram[0..4].copy_from_slice(&LDR.to_le_bytes());
    // PE 0 met a device at the load: the page compiles again, the load's
    // block a stub whose System exit is the site's access.
    first.learn_device_site(physical);
    let Ok(Fetched::Block(stub)) = first.fetch_block(pc, false) else {
        panic!("a stub")
    };
    assert!(first.device_stub(stub, pc));
    // PE 1 never met the device and runs PE 0's stub: its exit is the
    // site's access there too, not a System instruction.
    let Ok(Fetched::Block(found)) = second.fetch_block(pc, false) else {
        panic!("a block")
    };
    assert_eq!(found, stub);
    assert!(!second.device_site(physical));
    assert!(second.device_stub(found, pc));
}

#[test]
fn a_pe_that_joins_the_board_reads_the_count_natively_as_its_compiler_does() {
    const MRS_CNTVCT: u32 = 0xd53b_e04a; // mrs x10, cntvct_el0
    let words = [ADD, MRS_CNTVCT, RET];
    let mut memory = vec![0u8; WINDOW];
    let hub = Hub::new(2, NoKick);
    let mut first = arena_machine_of(&words, &mut memory);
    first.host.tsc_counter = Some((5, 7));
    let pc = first.cpu.pc;
    let Ok(Fetched::Block(slot)) = first.fetch_block(pc, false) else {
        panic!("a block")
    };
    // Its count read completes in a thunk that reads the PE's clock
    // reader, which compiling it set up on PE 0.
    assert!(!first.code.blocks[slot].successors.is_empty());
    assert!(first.clock.reads_natively());
    hub.share_code(&first.code);
    // PE 1 starts on the board's code and compiles nothing: it runs PE 0's
    // thunk, which must find a reader that follows the host counter.
    let mut second = machine(&words);
    second.host.tsc_counter = Some((5, 7));
    second.code = Arc::clone(&first.code);
    second.variants = Arc::clone(&first.variants);
    assert!(!second.clock.reads_natively());
    second.join_code(&hub);
    let Ok(Fetched::Block(found)) = second.fetch_block(pc, false) else {
        panic!("a block")
    };
    assert_eq!(found, slot);
    assert_eq!(second.statistics.compiled, 0);
    assert!(second.clock.reads_natively());
    assert!(second.count_reads_natively());
}

#[test]
fn a_hub_notes_the_pages_of_blocks_compiled_before_it() {
    let mut memory = vec![0u8; WINDOW];
    let mut first = arena_machine(&mut memory);
    let pc = first.cpu.pc;
    let Ok(Fetched::Block(slot)) = first.fetch_block(pc, false) else {
        panic!("a block")
    };
    let physical = first.code.blocks[slot].physical;
    let hub = Hub::new(2, NoKick);
    assert!(!hub.has_code(physical));
    hub.note_code(&first.code);
    assert!(hub.has_code(physical));
}

#[test]
fn a_word_admitted_as_udf_is_no_stale_code() {
    const UNALLOCATED: u32 = 0xe7ff_deff; // admitted as UDF
    let mut memory = vec![0u8; WINDOW];
    let mut machine = arena_machine_of(&[ADD, UNALLOCATED], &mut memory);
    let pc = machine.cpu.pc;
    let Ok(Fetched::Block(slot)) = machine.fetch_block(pc, false) else {
        panic!("a block")
    };
    assert_eq!(machine.code.blocks[slot].instructions, 2);
    assert_ne!(
        machine.host.ram[4..8],
        machine.admit(UNALLOCATED).to_le_bytes()
    );
    let report = machine.stale_code();
    assert_eq!((report.stale, report.admitted), (0, 1), "{report:?}");
}

/// A machine alone or one of a pair whose board gives only `slots` block
/// slots: the slots run out long before the window does.
fn starved(machine: &mut Machine<FakeHost, NoDevices>, memory: &mut [u8], slots: u32) {
    machine.code = Arc::new(board::Code::new(
        slots,
        chain::MIN_LOOKUP_ENTRIES,
        Some(window(memory)),
    ));
}

#[test]
fn a_board_out_of_slots_moves_its_ring_on_instead_of_stopping() {
    let mut memory = vec![0u8; WINDOW];
    let mut machine = arena_machine(&mut memory);
    starved(&mut machine, &mut memory, 8);
    let hub = Hub::new(2, NoKick);
    machine.code_hub = Some(Arc::clone(&hub));
    let base = machine.cpu.pc;
    for fetched in 0..64 {
        let pc = base + 4 * fetched;
        let Ok(Fetched::Block(slot)) = machine.fetch_block(pc, false) else {
            panic!("a block at {pc:#x}")
        };
        assert!(machine.code.blocks.installed(slot).is_some());
    }
    let report = machine.code.report();
    assert!(report.forced > 0, "{report:?}");
    assert_eq!(report.forced, report.retired);
    assert!(machine.code.blocks.len() <= 8);
    // Nothing was discarded: the ring gave the slots back.
    assert_eq!(machine.statistics.code_discards.iter().sum::<u64>(), 0);
    let codes = live_codes(&machine);
    assert!(codes.windows(2).all(|pair| pair[0] != pair[1]));
    let cache = machine.code.cache();
    assert!(
        cache
            .index
            .sweep(None)
            .all(|(_, slot)| machine.code.blocks.record(slot).live())
    );
}

/// A page of two blocks, the first ending in a direct branch to the
/// second (`b +8` over a hole), and the second's address.
fn chained_page() -> (Vec<u32>, u64) {
    let mut words = vec![RET; 4 * FETCH_PAGE as usize / 4];
    words[..5].copy_from_slice(&[ADD, 0x1400_0002, 0, ADD, 0x17ff_fffc]);
    (words, 12)
}

/// The part range of the block in `slot`.
fn part_of(machine: &Machine<FakeHost, NoDevices>, slot: usize) -> core::ops::Range<usize> {
    let arena = machine.code.arena.as_ref().expect("an arena");
    let code = machine.code.blocks[slot].code.0;
    let part = (code - arena.window.exec) / arena.part_bytes;
    arena.window.exec + part * arena.part_bytes..arena.window.exec + (part + 1) * arena.part_bytes
}

/// PE 0 runs A1 -> A2 by a direct link; PE 1's IC IVAU forgets the page.
/// PE 0, before it reads the post, still finds A1 in its jump cache: A1
/// takes no lookup entry again and links nothing, and its link out is cut,
/// so when A2's part and slot go no live block jumps into them (build 35:
/// a PE ran a link into a reused part, `CodeExtent` 854 KiB off).
#[test]
fn a_forgotten_block_keeps_no_link_and_takes_no_lookup_entry() {
    let (words, second_offset) = chained_page();
    let mut memory = vec![0u8; WINDOW];
    let (mut first, mut second, _hub) = shared_pair_of(&words, &mut memory);
    let base = first.cpu.pc;
    let Ok(Fetched::Block(a1)) = first.fetch_block(base, false) else {
        panic!("a block")
    };
    let Ok(Fetched::Block(a2)) = second.fetch_block(base + second_offset, false) else {
        panic!("a block")
    };
    assert!(first.link(a1, base + second_offset, ExitKind::Direct));
    assert!(first.code.blocks.record(a1).chained(0));
    let physical = first.code.blocks[a1].physical;
    second.invalidate_code_page(physical);
    assert!(
        !first.code.blocks.record(a1).chained(0),
        "the forget cuts A1's link"
    );
    // PE 0's jump cache still names A1.
    assert!(matches!(first.fetch_block(base, false), Ok(Fetched::Block(found)) if found == a1));
    let key = target_key(&first, base);
    let entry = first.code.table.find(target_slot(&first, base), base, key);
    assert!(
        !(entry.pc == base && entry.block as usize == a1),
        "a forgotten block took a lookup entry"
    );
    assert!(!first.link(a1, base + second_offset, ExitKind::Direct));
    // A2's part goes: no live block chains into it.
    let range = part_of(&first, a2);
    second.evict_code(range);
    for slot in 0..first.code.blocks.len() {
        let Some(block) = first.code.blocks.installed(slot) else {
            continue;
        };
        let record = first.code.blocks.record(slot);
        assert!(
            (0..block.successors.len()).all(|position| !record.chained(position)),
            "slot {slot} still chains into a part taken away"
        );
    }
}

/// A guest fault in a block another PE installed, reached by a direct
/// link, finds that block's rows; a fault outside the block it names says
/// which block's code holds the address.
#[test]
fn a_fault_in_a_linked_block_of_another_pe_finds_its_rows_and_a_stray_one_names_its_owner() {
    let (words, second_offset) = chained_page();
    let mut memory = vec![0u8; WINDOW];
    let (mut first, mut second, _hub) = shared_pair_of(&words, &mut memory);
    let base = first.cpu.pc;
    second.current = 1;
    let Ok(Fetched::Block(a2)) = second.fetch_block(base + second_offset, false) else {
        panic!("a block")
    };
    let Ok(Fetched::Block(a1)) = first.fetch_block(base, false) else {
        panic!("a block")
    };
    assert!(first.link(a1, base + second_offset, ExitKind::Direct));
    let step = first.native_fault(a2, access_fault(0), 0, None);
    assert!(
        !matches!(&step, Err(Stop::NativeFailure(failure))
            if matches!(failure.reason,
                NativeFailureReason::UnknownCode | NativeFailureReason::CodeExtent)),
        "{step:?}"
    );
    // A fault at A1's code, reported against A2.
    let offset = first.code.blocks[a1]
        .code
        .0
        .wrapping_sub(first.code.blocks[a2].code.0);
    let Err(Stop::NativeFailure(failure)) = first.native_fault(a2, access_fault(offset), 0, None)
    else {
        panic!("a failure")
    };
    assert_eq!(failure.reason, NativeFailureReason::CodeExtent);
    let owner = failure.owner.expect("the owner of the address");
    assert_eq!(owner.slot, a1 as u64);
    assert!(owner.live);
    assert!(failure.board.is_some());
}

/// A part of blocks compiled across eight pages in turn, each page's
/// blocks linked to each other, is taken away in steps that walk each
/// page's index list once: its visits grow with its blocks and their
/// pages, not with blocks times pages, and no step holds more than a
/// bounded number (build 37: 32-slot steps over interleaved pages walked
/// every page at every step, 8 s of lock holds a part).
#[test]
fn a_part_of_interleaved_pages_retires_in_steps_that_visit_each_page_once() {
    const PAGES: usize = 8;
    const B8: u32 = 0x1400_0002; // b .+8
    let group = [B8, 0, RET, 0];
    let words: Vec<u32> = group
        .iter()
        .copied()
        .cycle()
        .take(PAGES * FETCH_PAGE as usize / 4)
        .collect();
    let mut memory = vec![0u8; 4 << 18];
    let mut machine = machine(&words);
    machine.code = Arc::new(board::Code::new(
        board::slot_capacity(Some(memory.len())),
        chain::MIN_LOOKUP_ENTRIES,
        Some(window(&mut memory)),
    ));
    let base = machine.cpu.pc;
    let groups = FETCH_PAGE / 16;
    let mut first = None;
    'fill: for group in 0..groups {
        for page in 0..PAGES as u64 {
            let pc = base + page * FETCH_PAGE + 16 * group;
            let Ok(Fetched::Block(from)) = machine.fetch_block(pc, false) else {
                panic!("a block")
            };
            assert!(matches!(
                machine.fetch_block(pc + 8, false),
                Ok(Fetched::Block(_))
            ));
            machine.link(from, pc + 8, ExitKind::Direct);
            first.get_or_insert(from);
            if generation(&machine) > 0 {
                break 'fill;
            }
        }
    }
    let first = first.expect("blocks");
    let range = part_of(&machine, first);
    let evicted: Vec<usize> = (0..machine.code.blocks.len())
        .filter(|&slot| {
            machine
                .code
                .blocks
                .installed(slot)
                .is_some_and(|block| range.contains(&block.code.0))
        })
        .collect();
    assert!(
        evicted.len() > 64,
        "a part of many blocks: {}",
        evicted.len()
    );
    let indexed = machine.code.cache().index.sweep(None).count();
    machine.evict_code(range);
    let events = &machine.code.events;
    let visits = events.retire_visits.load(Ordering::Relaxed);
    let step = events.retire_step_visits.load(Ordering::Relaxed);
    // Each page's list once, each block once, and the cell steps.
    assert!(
        visits <= (evicted.len() + indexed) as u64 + 4 * 256,
        "{visits} visits for {} blocks, {indexed} indexed",
        evicted.len()
    );
    // A step stays within its budget, but for one page index list's share.
    let budget = super::super::code_cache::STEP_VISITS;
    assert!(step <= budget + FETCH_PAGE / 4, "{step} visits in one step");
    assert!(
        evicted
            .iter()
            .all(|&slot| !machine.code.blocks.record(slot).live())
    );
    // No live block chains into a block that went.
    for slot in 0..machine.code.blocks.len() {
        let Some(block) = machine.code.blocks.installed(slot) else {
            continue;
        };
        let record = machine.code.blocks.record(slot);
        for (position, exit) in block.successors.iter().enumerate() {
            if record.chained(position) {
                let physical = block.physical + (exit.target.0 - block.pc);
                let target = machine.code.find(exit.target.0, physical, block.variant);
                assert!(target.is_some_and(|target| !evicted.contains(&target)));
            }
        }
    }
}

/// A full ring leaves the part it retires to the installs after: they take
/// its steps, and the ring moves on to it only once they are done.
#[test]
fn a_retirement_is_taken_in_steps_by_later_installs() {
    let mut memory = vec![0u8; WINDOW];
    let mut machine = arena_machine(&mut memory);
    // Each counter read passes the budget: an install takes one step.
    machine.host.counter_step = 1 << 32;
    let base = machine.cpu.pc;
    let mut pending = 0;
    for fetched in 0..4096 {
        let pc = base + 4 * (fetched % 4096);
        assert!(matches!(
            machine.fetch_block(pc, false),
            Ok(Fetched::Block(_))
        ));
        if machine.code.retire_pending.load(Ordering::Relaxed) {
            pending += 1;
        }
        if generation(&machine) >= 8 {
            break;
        }
    }
    assert!(generation(&machine) >= 8, "the ring goes round");
    assert!(
        pending > 0,
        "a retirement outlived the install that began it"
    );
    let codes = live_codes(&machine);
    assert!(codes.windows(2).all(|pair| pair[0] != pair[1]));
}

/// Acquisitions of the board's code lock at `site` so far.
fn acquired(machine: &Machine<FakeHost, NoDevices>, site: board::LockSite) -> u64 {
    machine.code.cache().lock.sites[site as usize].acquired
}

/// A jump cache hit on a block the index no longer holds (an IC IVAU on
/// another PE forgot its page) adds no block table entry, and learns so
/// from the finder without the code lock (build 39: one `target` lock per
/// jump cache hit, 12-20 thousand a second at an idle login window).
#[test]
fn a_jump_cache_hit_on_a_forgotten_block_takes_no_code_lock() {
    let mut memory = vec![0u8; WINDOW];
    let (mut first, mut second, _hub) = shared_pair(&mut memory);
    let pc = first.cpu.pc;
    let Ok(Fetched::Block(slot)) = first.fetch_block(pc, false) else {
        panic!("a block")
    };
    second.invalidate_code_page(first.code.blocks[slot].physical);
    let before = acquired(&first, board::LockSite::Target);
    for _ in 0..16 {
        let jumped = first.statistics.fetches[2];
        assert!(matches!(first.fetch_block(pc, false), Ok(Fetched::Block(found)) if found == slot));
        assert_eq!(first.statistics.fetches[2], jumped + 1, "a jump cache hit");
    }
    assert_eq!(acquired(&first, board::LockSite::Target), before);
    let key = target_key(&first, pc);
    assert_ne!(
        first.code.table.find(target_slot(&first, pc), pc, key).pc,
        pc
    );
    assert_eq!(first.code.report().targets.unindexed, 16);
}

/// A block whose block table entry keeps going (here forgotten by hand,
/// on a board by other inserts into its set) is added again at its first
/// few jump cache hits, then at one in [`board::REMEMBER_EVERY`]: the lock
/// is taken a bounded share of the hits.
#[test]
fn a_block_whose_entry_keeps_going_takes_the_code_lock_a_bounded_share_of_its_hits() {
    const HITS: u64 = 256;
    let mut memory = vec![0u8; WINDOW];
    let mut machine = arena_machine(&mut memory);
    let pc = machine.cpu.pc;
    let Ok(Fetched::Block(slot)) = machine.fetch_block(pc, false) else {
        panic!("a block")
    };
    let key = target_key(&machine, pc);
    let frame = chain::key_frame(key);
    let before = acquired(&machine, board::LockSite::Target);
    for _ in 0..HITS {
        machine
            .code
            .cache()
            .targets
            .forget_frames(frame..=frame, |_| true);
        assert!(
            matches!(machine.fetch_block(pc, false), Ok(Fetched::Block(found)) if found == slot)
        );
    }
    let taken = acquired(&machine, board::LockSite::Target) - before;
    // The fill's own insert was the first of the block's free ones.
    let due = u64::from(board::REMEMBERS_FREE) - 1 + HITS / u64::from(board::REMEMBER_EVERY);
    assert!(taken <= due, "{taken} acquisitions for {HITS} hits");
    let report = machine.code.report();
    assert_eq!(report.targets.inserted, taken + 1);
    assert_eq!(report.targets.deferred, HITS - taken);
}

/// Pages of two-block groups, each group's first block ending in a direct
/// branch to its second (`b .+8` over a hole): words for `pages` pages.
fn linked_groups(pages: usize) -> Vec<u32> {
    const B8: u32 = 0x1400_0002; // b .+8
    [B8, 0, RET, 0]
        .iter()
        .copied()
        .cycle()
        .take(pages * FETCH_PAGE as usize / 4)
        .collect()
}

/// A machine of `words` over a window of `memory`, with as many slots.
fn windowed(words: &[u32], memory: &mut [u8]) -> Machine<FakeHost, NoDevices> {
    let mut machine = machine(words);
    machine.code = Arc::new(board::Code::new(
        board::slot_capacity(Some(memory.len())),
        chain::MIN_LOOKUP_ENTRIES,
        Some(window(memory)),
    ));
    machine
}

/// A page whose index list is far longer than a step's work, of which a
/// part takes a few blocks: each step of the retirement does bounded
/// work, whatever the page's list, and no live block chains into a block
/// that went (build 39: `evict` holds of 3-11 ms, a step walking a whole
/// page's list and table sets for each of its pages).
#[test]
fn a_page_with_a_long_index_list_retires_in_steps_of_bounded_work() {
    const B4: u32 = 0x1400_0001; // b .+4
    const BUDGET: u64 = 64;
    let words: Vec<u32> = [B4, RET]
        .iter()
        .copied()
        .cycle()
        .take(FETCH_PAGE as usize / 4)
        .collect();
    let mut memory = vec![0u8; 1 << 17];
    let mut machine = windowed(&words, &mut memory);
    Arc::get_mut(&mut machine.code).expect("alone").step_visits = BUDGET;
    let base = machine.cpu.pc;
    // The branches' targets first, so a part of them goes while the
    // branches linked into them stay.
    for pair in 0..FETCH_PAGE / 8 {
        assert!(matches!(
            machine.fetch_block(base + 8 * pair + 4, false),
            Ok(Fetched::Block(_))
        ));
    }
    for pair in 0..FETCH_PAGE / 8 {
        let pc = base + 8 * pair;
        let Ok(Fetched::Block(from)) = machine.fetch_block(pc, false) else {
            panic!("a block")
        };
        assert!(machine.link(from, pc + 4, ExitKind::Direct));
    }
    let listed = machine.code.cache().index.page(base).count();
    assert_eq!(listed as u64, FETCH_PAGE / 4);
    // The part of a block in the middle: a few of the page's blocks.
    // Slot 0 is the first target, at `base + 4`.
    let (physical, variant) = (
        machine.code.blocks[0].physical - 4,
        machine.code.blocks[0].variant,
    );
    let quarter = FETCH_PAGE / 4 + 4;
    let middle = machine
        .code
        .find(base + quarter, physical + quarter, variant)
        .expect("indexed");
    let range = part_of(&machine, middle);
    let evicted: Vec<usize> = (0..machine.code.blocks.len())
        .filter(|&slot| {
            machine
                .code
                .blocks
                .installed(slot)
                .is_some_and(|block| range.contains(&block.code.0))
        })
        .collect();
    assert!(evicted.len() < listed / 4, "{} of {listed}", evicted.len());
    let linked_in = (0..machine.code.blocks.len())
        .filter(|&slot| !evicted.contains(&slot) && machine.code.blocks.record(slot).chained(0))
        .filter(|&slot| {
            let block = &machine.code.blocks[slot];
            let target = machine
                .code
                .find(block.pc + 4, block.physical + 4, block.variant);
            target.is_some_and(|target| evicted.contains(&target))
        })
        .count();
    assert!(linked_in > 0, "blocks that stay link into blocks that go");
    machine.evict_code(range);
    let step = machine
        .code
        .events
        .retire_step_visits
        .load(Ordering::Relaxed);
    assert!(
        step <= BUDGET + listed as u64 / 16 + 32,
        "{step} visits in one step for a list of {listed}"
    );
    assert!(
        evicted
            .iter()
            .all(|&slot| !machine.code.blocks.record(slot).live())
    );
    assert_eq!(
        machine.code.cache().index.page(base).count(),
        listed - evicted.len()
    );
    for slot in 0..machine.code.blocks.len() {
        let Some(block) = machine.code.blocks.installed(slot) else {
            continue;
        };
        let record = machine.code.blocks.record(slot);
        for (position, exit) in block.successors.iter().enumerate() {
            if record.chained(position) {
                let physical = block.physical + (exit.target.0 - block.pc);
                let target = machine.code.find(exit.target.0, physical, block.variant);
                assert!(target.is_some_and(|target| !evicted.contains(&target)));
            }
        }
    }
    // Their lookup entries went with them.
    for &slot in &evicted {
        let block = &machine.code.blocks[slot];
        let key = machine.block_key(block.physical, block.variant);
        let table = &machine.code.table;
        let entry = table.find(table.slot(block.pc, key), block.pc, key);
        assert!(!(entry.pc == block.pc && entry.block as usize == slot));
    }
}

/// The code lock work an install takes: one hold, in which the fetch that
/// compiled the block also puts in its block table entry. A fetch of a
/// block another PE installed takes none.
#[test]
fn an_install_takes_the_code_lock_once_and_a_found_block_never() {
    const BLOCKS: u64 = 64;
    let mut memory = vec![0u8; 4 << 18];
    let (mut first, mut second, _hub) = shared_pair_of(&linked_groups(1), &mut memory);
    let base = first.cpu.pc;
    let total = |machine: &Machine<FakeHost, NoDevices>| machine.code.cache().lock.acquired;
    let before = total(&first);
    for group in 0..BLOCKS {
        assert!(matches!(
            first.fetch_block(base + 16 * group, false),
            Ok(Fetched::Block(_))
        ));
    }
    // One `cache()` read per call of `total`.
    let taken = total(&first) - before - 1;
    assert_eq!(taken, BLOCKS, "{taken} acquisitions for {BLOCKS} installs");
    assert_eq!(acquired(&first, board::LockSite::Target), 0);
    for group in 0..BLOCKS {
        let pc = base + 16 * group;
        let key = target_key(&first, pc);
        let entry = first.code.table.find(target_slot(&first, pc), pc, key);
        assert_eq!((entry.pc, entry.key), (pc, key));
    }
    assert_eq!(first.code.report().targets.inserted, BLOCKS);
    let before = total(&second);
    for group in 0..BLOCKS {
        assert!(matches!(
            second.fetch_block(base + 16 * group, false),
            Ok(Fetched::Block(_))
        ));
    }
    assert_eq!(total(&second) - before - 1, 0);
}

/// A direct exit within its page to a PC that has no block yet waits for
/// the fetch that compiles it without taking the code lock (the finder
/// says so); that fetch's install and its link take one hold each, and the
/// block, which only the link enters, takes no block table entry. An exit
/// to a PC that has a block links at once, in one hold.
#[test]
fn an_exit_to_a_pc_without_a_block_waits_for_it_without_the_code_lock() {
    let mut memory = vec![0u8; 4 << 18];
    let (mut first, _second, _hub) = shared_pair_of(&linked_groups(1), &mut memory);
    let base = first.cpu.pc;
    let Ok(Fetched::Block(from)) = first.fetch_block(base, false) else {
        panic!("a block")
    };
    let total = |machine: &Machine<FakeHost, NoDevices>| machine.code.cache().lock.acquired;
    let before = total(&first);
    assert!(!first.link(from, base + 8, ExitKind::Direct));
    assert_eq!(total(&first) - before - 1, 0);
    assert!(
        first
            .direct_exit
            .is_some_and(|(block, _, target, _)| block == from && target == base + 8)
    );
    let before = total(&first);
    let Ok(Fetched::Block(to)) = first.fetch_block(base + 8, false) else {
        panic!("a block")
    };
    assert_eq!(
        total(&first) - before - 1,
        2,
        "the install's and the link's"
    );
    assert!(first.code.blocks.record(from).chained(0));
    let pc = base + 8;
    let key = target_key(&first, pc);
    let entry = first.code.table.find(target_slot(&first, pc), pc, key);
    assert!(!(entry.pc == pc && entry.block as usize == to));
    let Ok(Fetched::Block(again)) = first.fetch_block(base + 16, false) else {
        panic!("a block")
    };
    assert!(matches!(
        first.fetch_block(base + 24, false),
        Ok(Fetched::Block(_))
    ));
    let before = acquired(&first, board::LockSite::Link);
    assert!(first.link(again, base + 24, ExitKind::Direct));
    assert_eq!(acquired(&first, board::LockSite::Link), before + 1);
}

/// A link that a retirement cut, whose target went, and that another PE
/// made again from the same block to the target compiled anew enters the
/// new block, as its bytes say: the source keeps its incarnation and is
/// chained again, so what a PE remembered of the first link no longer
/// names it (the race the concurrent test met while it remembered links).
#[test]
fn a_link_cut_from_a_retired_target_and_made_again_enters_the_new_block() {
    let words = linked_groups(4);
    let mut memory = vec![0u8; WINDOW];
    let (mut first, mut second, _hub) = shared_pair_of(&words, &mut memory);
    second.current = 1;
    let base = first.cpu.pc;
    let Ok(Fetched::Block(target)) = first.fetch_block(base + 8, false) else {
        panic!("a block")
    };
    // The source goes to the next part: the target's part goes alone.
    let start = generation(&first);
    let mut group = 1;
    while generation(&first) == start {
        assert!(matches!(
            first.fetch_block(base + 16 * group, false),
            Ok(Fetched::Block(_))
        ));
        group += 1;
    }
    let Ok(Fetched::Block(source)) = first.fetch_block(base, false) else {
        panic!("a block")
    };
    assert_ne!(part_of(&first, source), part_of(&first, target));
    assert!(first.link(source, base + 8, ExitKind::Direct));
    let code = Arc::clone(&first.code);
    let entry = code.blocks[target].code.0 as u64;
    assert_eq!(entered(&code, source, 0), Some((target, entry)));
    let incarnation = code.blocks.record(source).incarnation();
    let entered_then = code.blocks.record(target).incarnation();

    second.evict_code(part_of(&first, target));
    assert!(!code.blocks.record(target).live(), "the target went");
    assert!(
        !code.blocks.record(source).chained(0),
        "its link was cut first"
    );
    let Ok(Fetched::Block(again)) = second.fetch_block(base + 8, false) else {
        panic!("a block")
    };
    assert_ne!(
        (again, code.blocks.record(again).incarnation()),
        (target, entered_then)
    );
    assert!(second.link(source, base + 8, ExitKind::Direct));
    // The same source, chained again: only its bytes tell the links apart.
    let record = code.blocks.record(source);
    assert_eq!(record.incarnation(), incarnation);
    assert!(record.chained(0));
    let entry = code.blocks[again].code.0 as u64;
    assert_eq!(entered(&code, source, 0), Some((again, entry)));
    assert_eq!(enters_live_block(&code, source, 0), Ok(()));
}

/// What a looking PE held of a block it fetched inside its section: the
/// slot, its PC, its code and the ring's generation then. Until the PE
/// passes, the slot holds the block and the code's part is not filled
/// again.
struct Held {
    slot: usize,
    pc: u64,
    address: usize,
    generation: u32,
}

/// Whether the ring filled the part of `held`'s code again since it was
/// fetched: the part's last fill by then was at most a lap before.
fn refilled(code: &board::Code, held: &Held) -> bool {
    let arena = code.arena.as_ref().expect("an arena");
    let part = arena.part_of(held.address).expect("code in the window");
    arena.ring.holds(part) > held.generation
}

/// `bytes` bytes of the arena's code at executable `address`, if they lie
/// in its window. Read under the board's lock, which every link and cut
/// holds, so a link's bytes are whole.
fn arena_bytes(arena: &board::Arena, address: u64, bytes: usize) -> Option<Vec<u8>> {
    let start = usize::try_from(address).ok()?;
    arena.part_of(start)?;
    arena.part_of(start.checked_add(bytes.checked_sub(1)?)?)?;
    let alias = arena.window.write + (start - arena.window.exec);
    Some(
        (0..bytes)
            // SAFETY: inside the window, which the test's memory backs for
            // the board's life.
            .map(|offset| unsafe { core::ptr::read_volatile((alias + offset) as *const u8) })
            .collect(),
    )
}

/// The block and code entry that the link code at `at` enters, read back
/// from its bytes as `encode` writes them for a block and an entry: each
/// field lies where two encodings that differ in it alone first differ,
/// and every byte must be what `encode` writes for the fields read. `near`
/// is an address in a jump's reach of `at`. `None` when the bytes hold no
/// such link.
fn read_link(
    arena: &board::Arena,
    at: u64,
    near: u64,
    encode: impl Fn(u32, u64) -> Option<Vec<u8>>,
) -> Option<(usize, u64)> {
    let reference = encode(0, near)?;
    let differs = |other: Vec<u8>| reference.iter().zip(&other).position(|(a, b)| a != b);
    let block_at = differs(encode(1, near)?)?;
    let entry_at = differs(encode(0, near + 1)?)?;
    let bytes = arena_bytes(arena, at, reference.len())?;
    let field = |bytes: &[u8], at: usize| {
        u32::from_le_bytes(bytes[at..at + 4].try_into().expect("a 32-bit field"))
    };
    let block = field(&bytes, block_at);
    let moved = field(&bytes, entry_at).wrapping_sub(field(&reference, entry_at)) as i32;
    let entry = near.wrapping_add_signed(i64::from(moved));
    (encode(block, entry)? == bytes).then_some((block as usize, entry))
}

/// The block and code that block `slot`'s exit at `position` enters by the
/// bytes a core runs: its patch point, then the slot's body it falls into
/// or the thunk it jumps to ([`chain::link`], [`chain::thunk`]). `None`
/// when they hold no link.
fn entered(code: &board::Code, slot: usize, position: usize) -> Option<(usize, u64)> {
    let arena = code.arena.as_ref().expect("an arena");
    let block = &code.blocks[slot];
    let exit = block.successors[position];
    let word = exit.word();
    let retired = u32::from(block.instructions) - u32::from(word.skipped);
    let near = block.code.0 as u64;
    let (point, fallback) = match exit.slot {
        0 => (
            chain::direct_patch(exit.site),
            chain::Fallback::Return(word),
        ),
        bytes => (
            chain::slot_patch(exit.site - u64::from(bytes)),
            chain::Fallback::Jump(exit.site),
        ),
    };
    let instruction = arena_bytes(arena, point, chain::PATCH_BYTES)?;
    if exit.slot != 0 && instruction == chain::NOP5 {
        let at = exit.site - u64::from(exit.slot);
        let body = point + chain::PATCH_BYTES as u64;
        return read_link(arena, body, near, |block, entry| {
            chain::link(at, exit.slot.into(), retired, block, entry).map(|(_, bytes)| bytes)
        });
    }
    // Else a `jmp rel32` to a thunk.
    let (&0xe9, distance) = instruction.split_first()? else {
        return None;
    };
    let distance = i32::from_le_bytes(distance.try_into().ok()?);
    let thunk = (point + chain::PATCH_BYTES as u64).wrapping_add_signed(i64::from(distance));
    read_link(arena, thunk, near, |block, entry| {
        chain::thunk(thunk, retired, block, entry, fallback)
    })
}

/// Check that block `slot`'s chained exit at `position` enters, by the
/// bytes a core runs, a live block of its target at that block's code:
/// never a block a retirement took, nor code a part's refill reused.
fn enters_live_block(code: &board::Code, slot: usize, position: usize) -> Result<(), String> {
    let block = &code.blocks[slot];
    let target = block.successors[position].target.0;
    let (entered, entry) = entered(code, slot, position)
        .ok_or_else(|| format!("block {slot}'s chained exit {position} holds no link"))?;
    match code.blocks.installed(entered) {
        Some(found)
            if found.pc == target
                && found.variant == block.variant
                && found.code.0 as u64 == entry =>
        {
            Ok(())
        }
        found => Err(format!(
            "block {slot}'s exit {position} to {target:#x} enters slot {entered} at {entry:#x}, \
             which holds {:?}",
            found.map(|found| (found.pc, found.code.0))
        )),
    }
}

/// A PE's section, left however its thread ends: a PE that panicked
/// inside would hold every grace wait up.
struct Section<'a> {
    grace: &'a board::Grace,
    pe: usize,
}

impl<'a> Section<'a> {
    fn enter(grace: &'a board::Grace, pe: usize) -> Self {
        grace.enter(pe);
        Self { grace, pe }
    }
}

impl Drop for Section<'_> {
    fn drop(&mut self) {
        self.grace.leave(self.pe);
    }
}

/// Three PEs of one board on threads. PE 0 installs blocks of many pages
/// round a small ring, so parts retire in steps all the time, and links
/// them; PEs 1 and 2 fetch and link a few hot groups and, before each
/// pass of their section, check that every block they fetched still holds
/// its PC and code, and that each hot block's link, as the bytes a core
/// runs say, enters a live block of its target. No PE ever meets a block a
/// retirement took (lock-free fetch lookup, stepwise retirement, cut
/// links). A link is read from its bytes, not remembered: between two
/// checks a retirement may cut it and take its target, and another PE link
/// the same block again to the target compiled anew.
#[test]
fn pes_that_look_up_and_link_never_meet_a_block_a_concurrent_retirement_took() {
    const PES: usize = 3;
    const PAGES: usize = 4;
    const ROUNDS: usize = 20_000;
    const HOT: u64 = 8;
    // Indexed blocks a looker checks each round beyond its own, in turn.
    const SWEPT: usize = 16;
    let words = linked_groups(PAGES);
    let mut memory = vec![0u8; 8 * 4096];
    let template = windowed(&words, &mut memory);
    let hub = Hub::new(PES, NoKick);
    let code = Arc::clone(&template.code);
    let variants = Arc::clone(&template.variants);
    let reads = Arc::clone(&template.stored_reads);
    let writes = Arc::clone(&template.stored_writes);
    let base = template.cpu.pc;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    let make = |pe: usize| {
        let mut machine = machine(&words);
        machine.code = Arc::clone(&code);
        machine.variants = Arc::clone(&variants);
        machine.stored_reads = Arc::clone(&reads);
        machine.stored_writes = Arc::clone(&writes);
        machine.code_hub = Some(Arc::clone(&hub));
        machine.current = pe;
        // Each install takes one retirement step.
        machine.host.counter_step = machine.profile.counter_frequency;
        machine
    };
    std::thread::scope(|scope| {
        let installer = scope.spawn(|| {
            let mut machine = make(0);
            let grace = &hub.board.grace;
            let section = Section::enter(grace, 0);
            let groups = PAGES as u64 * FETCH_PAGE / 16;
            for round in 0..ROUNDS * 4 {
                if std::time::Instant::now() > deadline {
                    break;
                }
                grace.pass(0);
                let pc = base + 16 * (round as u64 * 7 % groups);
                let Ok(Fetched::Block(from)) = machine.fetch_block(pc, false) else {
                    panic!("a block at {pc:#x}")
                };
                if matches!(machine.fetch_block(pc + 8, false), Ok(Fetched::Block(_))) {
                    machine.link(from, pc + 8, ExitKind::Direct);
                }
            }
            drop(section);
            generation(&machine)
        });
        let lookers: Vec<_> = (1..PES)
            .map(|pe| {
                let make = &make;
                let hub = &hub;
                scope.spawn(move || {
                    let mut machine = make(pe);
                    let code = Arc::clone(&machine.code);
                    let grace = &hub.board.grace;
                    let mut checked = 0u64;
                    let mut linked = 0u64;
                    let mut cursor = None;
                    let section = Section::enter(grace, pe);
                    for round in 0..ROUNDS {
                        if std::time::Instant::now() > deadline {
                            break;
                        }
                        grace.pass(pe);
                        let pc = base + 16 * (round as u64 % HOT);
                        // The target in one round of three, before its
                        // source: it is older than its source as often as
                        // not, and goes while the source stays.
                        if round % 3 == 0 {
                            assert!(matches!(
                                machine.fetch_block(pc + 8, false),
                                Ok(Fetched::Block(_))
                            ));
                        }
                        let waits = code.events.grace_waits.load(Ordering::Relaxed);
                        let Ok(Fetched::Block(slot)) = machine.fetch_block(pc, false) else {
                            panic!("a block at {pc:#x}")
                        };
                        let block = code.blocks.get(slot).expect("a block");
                        assert_eq!(block.pc, pc, "PE {pe} fetched another block");
                        assert!(
                            !code.blocks.record(slot).lost(),
                            "PE {pe} fetched a lost install"
                        );
                        let held = Held {
                            slot,
                            pc,
                            address: block.code.0,
                            generation: generation(&machine),
                        };
                        let page = block.physical - (pc - base);
                        let variant = block.variant;
                        machine.link(slot, pc + 8, ExitKind::Direct);
                        // A wait for a grace (its own install) let go of
                        // what it held: no check then.
                        let still_held = code.events.grace_waits.load(Ordering::Relaxed) == waits;
                        if still_held {
                            let block = code.blocks.get(held.slot).expect("a block");
                            assert_eq!(block.pc, held.pc, "PE {pe}: slot reused while held");
                            assert!(
                                !refilled(&code, &held),
                                "PE {pe}: its part filled again while held"
                            );
                            checked += 1;
                        }
                        // Under the lock: each hot block that chains its
                        // exit, the one fetched and the next few indexed
                        // blocks of any PE's enter a live block of their
                        // target (a link to an older block of the PC is cut
                        // before either goes).
                        let mut cache = code.cache();
                        let swept: Vec<_> = cache.index.sweep(cursor).take(SWEPT).collect();
                        cursor = swept
                            .last()
                            .filter(|_| swept.len() == SWEPT)
                            .map(|&(at, _)| at);
                        let hot = (0..HOT).filter_map(|group| {
                            cache.index.get(
                                &code.blocks,
                                base + 16 * group,
                                page + 16 * group,
                                variant,
                            )
                        });
                        let mut sources: Vec<usize> = hot.collect();
                        sources.extend(still_held.then_some(slot));
                        sources.extend(swept.iter().map(|&(_, source)| source));
                        sources.sort_unstable();
                        sources.dedup();
                        for source in sources {
                            if code.blocks.record(source).chained(0) {
                                if let Err(error) = enters_live_block(&code, source, 0) {
                                    panic!("PE {pe}: {error}");
                                }
                                linked += 1;
                            }
                        }
                        drop(cache);
                    }
                    drop(section);
                    (checked, linked)
                })
            })
            .collect();
        let generations = installer.join().expect("the installer");
        let (checked, linked) = lookers
            .into_iter()
            .map(|looker| looker.join().expect("a looker"))
            .fold((0, 0), |(checked, linked), (more, links)| {
                (checked + more, linked + links)
            });
        assert!(
            std::time::Instant::now() <= deadline,
            "the PEs ran past their time"
        );
        assert!(
            generations >= 16,
            "the ring went round: {generations} parts"
        );
        assert!(checked > 0, "lookups were checked");
        assert!(linked > 0, "links were checked");
    });
    // Once the retirements end, no live block chains into one that went,
    // and every indexed block's code lies in a part that still holds it.
    let mut machine = make(0);
    machine.retire_steps(u64::MAX);
    let arena = code.arena.as_ref().expect("an arena");
    for slot in 0..code.blocks.len() {
        let Some(block) = code.blocks.installed(slot) else {
            continue;
        };
        if code.find(block.pc, block.physical, block.variant) == Some(slot) {
            let part = arena.part_of(block.code.0).expect("code in the window");
            let held = board::Place {
                generation: arena.ring.holds(part),
                part: part as u32,
                offset: 0,
            };
            assert!(
                arena.ring.live(held),
                "indexed block {slot} lies in part {part}, which went"
            );
        }
        let record = code.blocks.record(slot);
        for (position, exit) in block.successors.iter().enumerate() {
            if record.chained(position) {
                if let Err(error) = enters_live_block(&code, slot, position) {
                    panic!("after the retirements: {error}");
                }
                let physical = block.physical + (exit.target.0 - block.pc);
                assert!(code.find(exit.target.0, physical, block.variant).is_some());
            }
        }
    }
}

#[test]
fn the_board_counts_why_it_compiles_a_block_again() {
    let mut memory = vec![0u8; WINDOW];
    let mut machine = arena_machine(&mut memory);
    let base = machine.cpu.pc;
    let churn = |machine: &Machine<FakeHost, NoDevices>| machine.code.cache().churn;
    // A loop over more blocks than the ring holds: once round, each block
    // is compiled again after its part went.
    let mut fetched = 0;
    while generation(&machine) < 24 {
        assert!(fetched < 60_000, "the ring goes round");
        let pc = base + 4 * (fetched % 512);
        assert!(matches!(
            machine.fetch_block(pc, false),
            Ok(Fetched::Block(_))
        ));
        fetched += 1;
    }
    let counts = churn(&machine);
    assert_eq!(counts.installs, machine.install_causes().installs);
    // Ghosts are direct-mapped: one a collision lost counts as new.
    assert!(
        counts.new >= 512 && counts.new < counts.installs / 2,
        "{counts:?} fetched {fetched}"
    );
    assert!(counts.again[super::super::churn::Gone::Ring as usize] > 0);
    assert_eq!(
        counts.installs,
        counts.new + counts.again.iter().sum::<u64>(),
        "every install is new or again"
    );
    // Most come back within a few turns of the ring of four parts.
    assert!(counts.ages[0] + counts.ages[1] > counts.ages[2]);
    assert_eq!((counts.aliases, counts.variants), (0, 0));

    // An IC IVAU of unchanged bytes: compiled again, the same bytes.
    let pc = base + 4 * (fetched % 512);
    let Ok(Fetched::Block(slot)) = machine.fetch_block(pc, false) else {
        panic!("a block at {pc:#x}")
    };
    let before = churn(&machine);
    let physical = machine.code.blocks[slot].physical;
    machine.invalidate_code_page(physical);
    assert!(matches!(
        machine.fetch_block(pc, false),
        Ok(Fetched::Block(_))
    ));
    let after = churn(&machine).since(&before);
    assert_eq!(
        after.again[super::super::churn::Gone::Invalidated as usize],
        1
    );
    assert_eq!(after.same_bytes, 1);
}

/// A machine of `words` whose code goes to a window of all of `memory`.
fn arena_machine_over(words: &[u32], memory: &mut [u8]) -> Machine<FakeHost, NoDevices> {
    let mut machine = machine(words);
    machine.code = Arc::new(board::Code::new(
        board::slot_capacity(Some(memory.len())),
        chain::MIN_LOOKUP_ENTRIES,
        Some(window(memory)),
    ));
    machine
}

/// The slot a fetch at `pc` runs.
fn fetched(machine: &mut Machine<FakeHost, NoDevices>, pc: u64) -> usize {
    match machine.fetch_block(pc, false) {
        Ok(Fetched::Block(slot)) => slot,
        _ => panic!("a block at {pc:#x}"),
    }
}

#[test]
fn hot_blocks_survive_many_turns_of_cold_code_in_the_tenured_generation() {
    // Sixteen parts of a page: about 128 blocks each, and cold code that
    // comes back only after eight turns, past the age that promotes.
    let mut memory = vec![0u8; 16 * 4096];
    let words = vec![RET; 16 * 1024];
    let mut machine = arena_machine_over(&words, &mut memory);
    let base = machine.cpu.pc;
    const HOT: u64 = 64;
    let cold = |at: u64| base + 4 * (HOT + at % (words.len() as u64 - HOT));
    let mut at = 0;
    // Cold code turns the ring while the hot blocks keep being entered:
    // their first part goes, they come back tenured and stay.
    let mut run = |machine: &mut Machine<FakeHost, NoDevices>, turns: u32| {
        let until = generation(machine) + turns * 16;
        let mut hot = Vec::new();
        while generation(machine) < until {
            for _ in 0..32 {
                fetched(machine, cold(at));
                at += 1;
            }
            hot = (0..HOT)
                .map(|block| fetched(machine, base + 4 * block))
                .collect();
        }
        hot
    };
    let warm = run(&mut machine, 4);
    let arena = machine.code.arena.as_ref().expect("an arena");
    assert!(warm.iter().all(|&slot| {
        let part = arena
            .part_of(machine.code.blocks[slot].code.0)
            .expect("code in the window");
        arena.ring.kind(part) == board::TENURED
    }));
    let before = machine.install_causes().installs;
    let after = run(&mut machine, 10);
    // Ten turns of the ring later the hot blocks are the same blocks.
    assert_eq!(warm, after);
    let report = machine.code.report();
    assert!(report.promoted >= HOT);
    assert!(report.outs[board::Why::Young as usize] >= 100);
    assert!(machine.install_causes().installs - before >= 10 * 16 * 64);
    assert_eq!(report.generations[board::TENURED], 1);

    // An IC IVAU of a tenured page still drops its blocks: the next fetch
    // compiles the block again, and the part counts the hole.
    let slot = after[0];
    let physical = machine.code.blocks[slot].physical;
    machine.invalidate_code_page(physical);
    assert!(
        machine.code.blocks.installed(slot).is_some(),
        "its code stays"
    );
    let again = fetched(&mut machine, base);
    assert_ne!(again, slot);
    let churn = machine.code.cache().churn;
    assert!(churn.again[super::super::churn::Gone::Invalidated as usize] >= 1);
}

/// Fill `stream`'s open part of `ring` and open the next, taking away at
/// once whatever part that chose: the parts chosen and why.
fn turn(ring: &board::Ring, grace: &board::Grace, stream: usize) -> Vec<(u32, board::Why)> {
    let mut chosen = Vec::new();
    while let Ok(place) = ring.take(stream, 64, 16) {
        ring.release(place);
    }
    let full = ring.head(stream);
    loop {
        match ring.advance(stream, full, grace, None) {
            board::Advance::Moved => return chosen,
            board::Advance::Retire(part, why) => {
                chosen.push((part, why));
                ring.retired(part, grace.snapshot());
                if ring.head(stream) != full {
                    return chosen;
                }
            }
            other => panic!("a ring nobody holds: {other:?}"),
        }
    }
}

/// The open part of `stream`.
fn open_part(ring: &board::Ring, stream: usize) -> usize {
    (ring.head(stream) & 63) as usize
}

#[test]
fn the_tenured_part_entered_least_goes_first_and_age_and_holes_bound_the_rest() {
    let grace = board::Grace::new(1);
    let ring = board::Ring::new(16, 1024);
    assert_eq!(ring.young_parts, 4);
    // Young code fills the ring alone: first in, first out.
    let mut young = Vec::new();
    for _ in 0..20 {
        young.extend(turn(&ring, &grace, board::YOUNG));
    }
    assert!(young.iter().all(|&(_, why)| why == board::Why::Young));
    assert!(young.len() >= 4);
    // Tenured code takes parts; the young generation shrinks to its share.
    let mut tenured = Vec::new();
    for _ in 0..10 {
        tenured.push(open_part(&ring, board::TENURED));
        turn(&ring, &grace, board::TENURED);
        turn(&ring, &grace, board::YOUNG);
    }
    let tenured: Vec<usize> = tenured.into_iter().skip(1).collect();
    let [young_parts, tenured_parts] = ring.owned();
    assert!(young_parts <= 5, "young keeps its share: {young_parts}");
    assert!(tenured_parts >= 9);
    // Every full tenured part but one is entered.
    let cold = tenured[3];
    for &part in &tenured {
        if part != cold {
            for _ in 0..10 {
                ring.entered(part);
            }
        }
    }
    // Young code past its share goes first; then the tenured choice.
    let chosen = loop {
        let chosen = turn(&ring, &grace, board::TENURED);
        if let Some(&choice) = chosen.iter().find(|&&(_, why)| why != board::Why::Young) {
            break choice;
        }
    };
    assert_eq!(chosen, (cold as u32, board::Why::Cold));

    // A part whose blocks mostly left the index goes before the rest.
    let holey = tenured[5];
    for _ in 0..100 {
        ring.installed(holey);
    }
    for _ in 0..80 {
        ring.dropped(holey);
    }
    let mut chosen = Vec::new();
    while chosen.is_empty() {
        chosen = turn(&ring, &grace, board::YOUNG);
    }
    assert_eq!(chosen, [(holey as u32, board::Why::Holes)]);

    // Young code alone for longer than the age cap: the oldest tenured
    // parts go however often they are entered.
    let mut aged = Vec::new();
    for _ in 0..(board::AGE_TURNS + 1) * 16 {
        for part in 0..16 {
            ring.entered(part);
        }
        aged.extend(turn(&ring, &grace, board::YOUNG));
    }
    assert!(aged.iter().any(|&(_, why)| why == board::Why::Aged));
    let outs = ring.outs();
    assert!(outs.iter().all(|&count| count > 0), "{outs:?}");
}

#[test]
fn a_part_another_part_jumps_into_waits_for_it() {
    let grace = board::Grace::new(1);
    let ring = board::Ring::new(16, 1024);
    for _ in 0..4 {
        turn(&ring, &grace, board::TENURED);
    }
    // The first tenured part's code jumps into the third's: the third is
    // never chosen while the first holds code, however cold it is.
    let [first, _, third] = [0, 1, 2].map(|at| {
        (0..16)
            .filter(|&part| ring.kind(part) == board::TENURED)
            .min_by_key(|&part| (ring.holds(part), part))
            .map(|part| part + at)
            .unwrap()
    });
    ring.refer(first, third);
    for part in 0..16 {
        if part != third {
            ring.entered(part);
        }
    }
    let mut chosen = Vec::new();
    for _ in 0..40 {
        chosen.extend(turn(&ring, &grace, board::TENURED));
        if chosen.iter().any(|&(part, _)| part as usize == first) {
            break;
        }
    }
    let first_out = chosen
        .iter()
        .position(|&(part, _)| part as usize == first)
        .expect("the first part goes");
    assert!(
        !chosen[..first_out]
            .iter()
            .any(|&(part, _)| part as usize == third)
    );
}

/// A tenured part that stayed open while young code turned the ring past
/// the age cap is old when a block takes its last bytes. The block's own
/// PAC thunk then finds the part full and opens the next, and that advance
/// chooses a part to go: never the one the block's install still writes,
/// which a choice by age counted from the part's opening took (finding 1
/// of the engine audit: the block was then published into a part already
/// taken away, and later ran another block's bytes).
#[test]
fn a_block_whose_thunk_fills_its_part_is_published_in_a_part_that_stays() {
    const RETAA: u32 = 0xd65f_0bff;
    let words = [RETAA, RET].repeat(2 * FETCH_PAGE as usize / 8);
    // The block's code as a machine alone compiles it, and its frame.
    let (block_bytes, physical) = {
        let mut memory = vec![0u8; 8 * 4096];
        let mut probe = arena_machine_over(&words, &mut memory);
        let base = probe.cpu.pc;
        let slot = fetched(&mut probe, base);
        let recovery = probe.code.blocks.recovery();
        let bytes = recovery.get(&slot).map(|owner| owner.code_bytes());
        (bytes.expect("rows"), probe.code.blocks[slot].physical)
    };
    let mut memory = vec![0u8; 8 * 4096];
    let mut machine = arena_machine_over(&words, &mut memory);
    let code = Arc::clone(&machine.code);
    let arena = code.arena.as_ref().expect("an arena");
    let ring = &arena.ring;
    assert_eq!(ring.parts, 8);
    let grace = board::Grace::new(1);
    // The tenured generation opens; young code turns the ring past its age.
    let none = ring.head(board::TENURED);
    assert_eq!(
        ring.advance(board::TENURED, none, &grace, None),
        board::Advance::Moved
    );
    let tenured = open_part(ring, board::TENURED);
    let opened = ring.holds(tenured);
    while ring.generation() < opened + board::AGE_TURNS * ring.parts {
        turn(ring, &grace, board::YOUNG);
    }
    // All of the tenured part but the block's bytes is taken.
    let block_at = arena.part_bytes - block_bytes.next_multiple_of(16);
    let filler = ring.take(board::TENURED, block_at, 16).expect("room");
    ring.release(filler);
    // The ring took the block lately: it is compiled tenured.
    let pc = machine.cpu.pc;
    let context = machine.current_variant(false, false);
    let guarded_page = machine.fetch_guarded(pc);
    let variant = machine.page_variant(
        Variant {
            guarded_page,
            ..context
        },
        physical,
    );
    let id = machine.variant_id(variant);
    let identity = super::super::churn::identity(pc, physical, id.0);
    let gone = super::super::churn::Gone::Ring;
    code.ghosts.gone(
        &mut Default::default(),
        identity,
        0,
        ring.generation(),
        gone,
    );
    let outs: u64 = ring.outs().iter().sum();
    let slot = fetched(&mut machine, pc);
    let block = &code.blocks[slot];
    let part = arena.part_of(block.code.0).expect("code in the window");
    assert_eq!(part, tenured, "the block took the tenured part");
    assert_eq!(block.code.0 - arena.range(part as u32).start, block_at);
    // Its thunk moved the ring on, which chose a part to go, not this one.
    assert_eq!(ring.outs().iter().sum::<u64>(), outs + 1);
    let held = board::Place {
        generation: ring.holds(part),
        part: part as u32,
        offset: 0,
    };
    assert_eq!(ring.retired_through(part), 0, "the block's part went");
    assert!(ring.live(held), "the block's part was chosen to go");
    assert_eq!(code.report().redone, 0);
    assert_eq!(machine.code.find(pc, physical, id), Some(slot));
}

/// Another PE's advance chooses a block's part between the block's take
/// and its publish, and the part is taken away: its retirement listed the
/// part's blocks without this one, so the block is not published there
/// but installed again, in a part that stays.
#[test]
fn a_block_whose_part_is_chosen_before_it_publishes_is_installed_again() {
    let mut memory = vec![0u8; WINDOW];
    let mut machine = arena_machine(&mut memory);
    let code = Arc::clone(&machine.code);
    let other = Arc::clone(&code);
    let chosen = Arc::new(core::sync::atomic::AtomicU32::new(u32::MAX));
    let noted = Arc::clone(&chosen);
    // The first install's write: another PE fills the part, moves the ring
    // on and chooses the part, then retires it (it holds no listed block).
    machine.host.on_serialize = Some(Box::new(move |count| {
        if count != 1 {
            return;
        }
        let ring = &other.arena.as_ref().expect("an arena").ring;
        let grace = board::Grace::new(1);
        let part = open_part(ring, board::YOUNG) as u32;
        while let Ok(place) = ring.take(board::YOUNG, 16, 16) {
            ring.release(place);
        }
        let full = ring.head(board::YOUNG);
        assert_eq!(
            ring.advance(board::YOUNG, full, &grace, None),
            board::Advance::Moved
        );
        assert!(ring.choose_part(part));
        ring.retired(part, grace.snapshot());
        noted.store(part, Ordering::Relaxed);
    }));
    let pc = machine.cpu.pc;
    let slot = fetched(&mut machine, pc);
    let gone = chosen.load(Ordering::Relaxed);
    assert_ne!(gone, u32::MAX, "the hook ran");
    let arena = code.arena.as_ref().expect("an arena");
    let part = arena
        .part_of(code.blocks[slot].code.0)
        .expect("code in the window");
    assert_ne!(
        part as u32, gone,
        "a block published into a part taken away"
    );
    assert_eq!(code.report().redone, 1);
    let block = &code.blocks[slot];
    assert_eq!(
        code.find(pc, block.physical, block.variant),
        Some(slot),
        "the block installed again is the one indexed"
    );
    assert_eq!(live_codes(&machine), [block.code.0]);
}
