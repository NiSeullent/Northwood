//! Host model of process switching through one PE's ITLB: address spaces
//! take turns, each fetching its hot user pages at the same VAs (every
//! process's main binary and stack sit at one address), and every turn
//! fetches the kernel's hot pages in the upper half, global as XNU maps its
//! text. The PE fetches through the engine's own context, ITLB and TLBI
//! paths ([`Machine::contexts`], [`Machine::current_key`],
//! [`Machine::translate_page`], [`Machine::tlb_maintained`]); only the walk
//! is left out, as its result is the block key the model names.
//!
//! `cargo test -p stela-engine --release --lib process_switching_miss_rates
//! -- --ignored --nocapture` prints ITLB misses per switch for each count of
//! processes, with per-page TLBIs of single ASIDs between turns, and with a
//! TLBI of everything every so often (which the architecture requires to
//! drop everything of the regime).

use super::*;
use mmu::tlbi::{Everywhere, Reach};

const MSR_TTBR0_EL1_X0: u32 = 0xd518_2000;
/// Hot pages of each process, at the same VAs in every one.
const USER_PAGES: u64 = 48;
/// Hot kernel pages each turn runs (system call, interrupt, switch paths).
const KERNEL_PAGES: u64 = 64;
const USER_BASE: u64 = 0x1_0000_0000;
const KERNEL_BASE: u64 = 0xffff_fe00_1000_0000;

/// What one run measured.
#[derive(Debug, Clone, Copy, Default)]
struct Rates {
    switches: u64,
    user_misses: u64,
    kernel_misses: u64,
}

impl Rates {
    fn per_switch(self) -> (f64, f64) {
        let switches = self.switches.max(1) as f64;
        (
            self.user_misses as f64 / switches,
            self.kernel_misses as f64 / switches,
        )
    }
}

/// A deterministic choice of the next process: mostly the recent ones, as
/// a desktop's few busy processes run while hundreds wake now and then.
struct Picks(u64);

impl Picks {
    fn next(&mut self, processes: u64) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let draw = self.0 >> 33;
        // Three in four turns go to the busiest eighth.
        let busy = (processes / 8).max(1);
        if !draw.is_multiple_of(4) {
            (draw >> 2) % busy
        } else {
            (draw >> 2) % processes
        }
    }
}

/// Fetch `pc` at `el` in the current contexts: whether the ITLB held it.
fn touch(machine: &mut Machine<FakeHost, NoDevices>, el: u8, pc: u64, key: u64) -> bool {
    let variant = machine.current_variant(false, false);
    let contexts = machine.contexts(el, variant);
    machine.chain.set_contexts(contexts);
    if machine.current_key(pc) == Some(key) {
        return true;
    }
    let half = chain::half(pc);
    machine.fetched_through(half, pc, FETCH_PAGE, half == 1);
    // The walk may have forgotten contexts; the fetch records the page in
    // the ones it runs in after.
    let contexts = machine.contexts(el, variant);
    machine.chain.set_contexts(contexts);
    machine.translate_page(pc, key);
    false
}

/// The 8-bit ASID of process `process` (XNU's on an 8-bit chip shares one
/// between processes past 255).
fn asid(process: u64) -> u64 {
    process % 255 + 1
}

/// Run `switches` turns among `processes`: a TLBI of one page of one ASID
/// after each turn with `page_tlbis`, and a TLBI of everything every
/// `all_every` turns when nonzero.
fn run(processes: u64, switches: u64, page_tlbis: bool, all_every: u64) -> Rates {
    let mut machine = machine(&[]);
    let kernel_el = machine.cpu.pstate.el;
    let ttbr0 = machine.cpu.read(SystemRegister::TTBR0_EL1);
    let mut picks = Picks(processes ^ 0x5eed);
    let mut rates = Rates::default();
    // Process `p`'s user page `n` maps frame `p * USER_PAGES + n`.
    let user_key =
        |process: u64, page: u64| chain::block_key((process * USER_PAGES + page) << 12, 2);
    let kernel_key = |page: u64| chain::block_key((1 << 30 | page) << 12, 2);
    for turn in 0..switches {
        let process = picks.next(processes);
        let table = ttbr0 & 0xffff_ffff_ffff & !0xfff_ffff | (process + 1) << 16;
        machine.cpu.state.x[0] = table | asid(process) << 48;
        machine.system_instruction(MSR_TTBR0_EL1_X0).unwrap();
        rates.switches += 1;
        for page in 0..KERNEL_PAGES / 2 {
            let pc = KERNEL_BASE + page * FETCH_PAGE;
            if !touch(&mut machine, kernel_el, pc, kernel_key(page)) {
                rates.kernel_misses += 1;
            }
        }
        for page in 0..USER_PAGES {
            let pc = USER_BASE + page * FETCH_PAGE;
            if !touch(&mut machine, 0, pc, user_key(process, page)) {
                rates.user_misses += 1;
            }
        }
        for page in KERNEL_PAGES / 2..KERNEL_PAGES {
            let pc = KERNEL_BASE + page * FETCH_PAGE;
            if !touch(&mut machine, kernel_el, pc, kernel_key(page)) {
                rates.kernel_misses += 1;
            }
        }
        if page_tlbis {
            // A page of a heap or a stack (not a hot code page), of one
            // process, as `vm_map` changes make thousands of times a second.
            let victim = picks.next(processes);
            let reach = Reach::Containing(USER_BASE + (USER_PAGES + turn % 64) * FETCH_PAGE);
            machine.tlb_maintained(
                reach,
                Everywhere::Addresses(reach),
                Some(asid(victim) as u16),
            );
            // And one hot page of one process, which that process refills.
            let reach = Reach::Containing(USER_BASE + (turn % USER_PAGES) * FETCH_PAGE);
            machine.tlb_maintained(
                reach,
                Everywhere::Addresses(reach),
                Some(asid(victim) as u16),
            );
        }
        if all_every != 0 && turn % all_every == all_every - 1 {
            machine.tlb_maintained(Reach::All, Everywhere::All, None);
        }
    }
    rates
}

/// Prints misses per switch; the assertions only check the model runs.
#[test]
#[ignore = "manual: opt-in"]
fn process_switching_miss_rates() {
    const SWITCHES: u64 = 20_000;
    println!("processes  tlbis  all_every  user/switch  kernel/switch");
    for &processes in &[16, 48, 96, 192, 384] {
        for (page_tlbis, all_every) in [(false, 0), (true, 0), (true, 22), (true, 60)] {
            let rates = run(processes, SWITCHES, page_tlbis, all_every);
            let (user, kernel) = rates.per_switch();
            println!(
                "{processes:9}  {:5}  {all_every:9}  {user:11.2}  {kernel:13.2}",
                page_tlbis
            );
            assert_eq!(rates.switches, SWITCHES);
        }
    }
}

/// The context of TTBR0 value `table` with ASID `asid` at `el`.
fn context_of(
    machine: &mut Machine<FakeHost, NoDevices>,
    el: u8,
    table: u64,
    asid: u64,
) -> [u64; 2] {
    let variant = machine.current_variant(false, false);
    machine.cpu.state.x[0] = table & !(0xffff << 48) | asid << 48;
    machine.system_instruction(MSR_TTBR0_EL1_X0).unwrap();
    machine.contexts(el, variant)
}

/// Put an ITLB entry of `page` in `context`, the set's other entry kept.
fn put(machine: &mut Machine<FakeHost, NoDevices>, page: u64, context: u64) {
    let slot = machine.itlb.page_slot(page, context);
    machine
        .itlb
        .insert(slot, chain::LookupEntry::page(page, context, 1));
}

/// Whether the ITLB holds `page` in `context`.
fn held(machine: &Machine<FakeHost, NoDevices>, page: u64, context: u64) -> bool {
    let slot = machine.itlb.page_slot(page, context);
    machine.itlb.find(slot, page, context).key == context
}

/// A TLBI by VA and ASID drops the page in that ASID's contexts alone,
/// though other ASIDs' contexts share their salt: with more address spaces
/// than salts, a salt is never what decides. An ASID an 8-bit TCR.AS gave
/// two address spaces drops the page in both.
#[test]
fn a_tlbi_by_va_and_asid_spares_other_asids_that_share_its_salt() {
    const PAGE: u64 = 0x8000;
    let mut machine = machine(&[]);
    let el = machine.cpu.pstate.el;
    let ttbr0 = machine.cpu.read(SystemRegister::TTBR0_EL1);
    let contexts: Vec<(u64, u64)> = (1..=70u64)
        .map(|asid| {
            (
                asid,
                context_of(&mut machine, el, ttbr0 + asid * 0x10_0000, asid)[0],
            )
        })
        .collect();
    let salt = |context: u64| context % chain::LOOKUP_SALTS as u64;
    let (&(asid, mine), &(other_asid, other)) = contexts
        .iter()
        .flat_map(|a| contexts.iter().map(move |b| (a, b)))
        .find(|(a, b)| a.0 != b.0 && salt(a.1) == salt(b.1))
        .expect("70 address spaces share a salt among 64");
    assert_ne!(asid, other_asid);
    // A third address space reusing `asid` (an 8-bit ASID's wrap).
    let reused = context_of(&mut machine, el, ttbr0 + 0x7f0_0000, asid + 256)[0];
    machine.fetched_through(0, PAGE, 0x4000, false);
    for context in [mine, other, reused] {
        put(&mut machine, PAGE, context);
    }
    assert!(held(&machine, PAGE, mine) && held(&machine, PAGE, other));
    let reach = Reach::Containing(PAGE);
    machine.tlb_maintained(reach, Everywhere::Addresses(reach), Some(asid as u16));
    assert!(!held(&machine, PAGE, mine));
    assert!(!held(&machine, PAGE, reused));
    assert!(held(&machine, PAGE, other));
    assert_eq!(
        machine.statistics.itlb_sources[ItlbSource::DropAsid as usize],
        2
    );
}

/// A process's context outlives a thousand others' and finds its pages
/// again; past the half's bound the least recently used goes with its
/// entries, and its key takes a new number whose ITLB entries it lacks.
#[test]
fn contexts_outlive_a_thousand_address_spaces_then_go_least_recently_used() {
    const PAGE: u64 = 0x8000;
    let mut machine = machine(&[]);
    let el = machine.cpu.pstate.el;
    let ttbr0 = machine.cpu.read(SystemRegister::TTBR0_EL1);
    let table = |process: u64| ttbr0 + (process + 1) * 0x10_0000;
    let first = context_of(&mut machine, el, table(0), 1);
    put(&mut machine, PAGE, first[0]);
    for process in 1..=1000 {
        context_of(&mut machine, el, table(process), process % 250 + 1);
    }
    assert_eq!(context_of(&mut machine, el, table(0), 1), first);
    assert!(held(&machine, PAGE, first[0]));
    assert_eq!(machine.statistics.itlb_drops[2], 0);
    for process in 1001..=1000 + contexts::LOWER_CONTEXTS as u64 {
        context_of(&mut machine, el, table(process), process % 250 + 1);
    }
    assert!(machine.statistics.itlb_drops[2] > 0);
    assert!(machine.spaces[0].live(first[0]).is_none());
    let renewed = context_of(&mut machine, el, table(0), 1);
    assert_ne!(renewed[0], first[0]);
    assert_eq!(renewed[1], first[1]);
    assert!(machine.spaces[0].len() <= contexts::LOWER_CONTEXTS);
    // The entry of the context that went is never found for the new one.
    machine.chain.set_contexts(renewed);
    assert_eq!(machine.current_key(PAGE), None);
}

/// Forms without an address remove what [`mmu::tlbi::Scope::covers`]
/// says: in the host (EL2&0), ALLE1, VMALLS12E1 and IPAS2E1 reach no entry
/// of its regime and leave its contexts; ALLE2 and VMALLE1 (EL2&0 in the
/// host) forget them. A TLBI of everything a PE's mail kept forgets every
/// context.
#[test]
fn a_tlbi_of_everything_forgets_only_the_regimes_it_reaches() {
    const HCR_E2H: u64 = 1 << 34;
    const HCR_TGE: u64 = 1 << 27;
    const TLBI_ALLE1: u32 = 0xd50c_879f;
    const TLBI_VMALLS12E1: u32 = 0xd50c_87df;
    const TLBI_IPAS2E1_X0: u32 = 0xd50c_8420;
    const TLBI_ALLE2: u32 = 0xd50c_871f;
    const TLBI_VMALLE1: u32 = 0xd508_871f;
    let mut machine = machine(&[]);
    machine.cpu.pstate.el = 2;
    let hcr = machine.cpu.read(SystemRegister::HCR_EL2) | HCR_E2H | HCR_TGE;
    machine.cpu.registers.set(SystemRegister::HCR_EL2, 0, hcr);
    machine.controls_changed();
    assert_eq!(machine.cpu.tlbi_current().regime, mmu::Regime::El20);
    let variant = machine.current_variant(false, false);
    let contexts = |machine: &mut Machine<FakeHost, NoDevices>| {
        [machine.contexts(2, variant), machine.contexts(0, variant)]
    };
    let before = contexts(&mut machine);
    for word in [TLBI_ALLE1, TLBI_VMALLS12E1, TLBI_IPAS2E1_X0] {
        machine.cpu.state.x[0] = 0x1234;
        machine.system_instruction(word).unwrap();
        assert_eq!(contexts(&mut machine), before, "{word:#x}");
    }
    assert_eq!(
        machine.statistics.itlb_sources[ItlbSource::AllKept as usize],
        3
    );
    let mut last = before;
    for word in [TLBI_ALLE2, TLBI_VMALLE1] {
        machine.system_instruction(word).unwrap();
        let after = contexts(&mut machine);
        for (was, now) in last.iter().flatten().zip(after.iter().flatten()) {
            assert_ne!(was, now, "{word:#x}");
        }
        last = after;
    }
    assert_eq!(
        machine.statistics.itlb_sources[ItlbSource::AllLocal as usize],
        2
    );
    // The contexts of another regime (EL1&0, made outside the host) go
    // with a TLBI that reaches nothing of the current one; the current
    // regime's stay. One that reaches it forgets every context.
    let host = |machine: &mut Machine<FakeHost, NoDevices>, on: bool| {
        machine.cpu.pstate.el = if on { 2 } else { 1 };
        let hcr = machine.cpu.read(SystemRegister::HCR_EL2) & !HCR_TGE;
        let hcr = if on { hcr | HCR_TGE } else { hcr };
        machine.cpu.registers.set(SystemRegister::HCR_EL2, 0, hcr);
        machine.controls_changed();
    };
    host(&mut machine, false);
    assert_eq!(machine.cpu.tlbi_current().regime, mmu::Regime::El10);
    let guest = machine.contexts(1, variant);
    host(&mut machine, true);
    let kept = machine.contexts(2, variant);
    machine.tlb_maintained(Reach::Nothing, Everywhere::All, None);
    assert_eq!(
        machine.statistics.itlb_sources[ItlbSource::AllForgot as usize],
        1
    );
    assert_eq!(machine.contexts(2, variant), kept);
    host(&mut machine, false);
    let renewed = machine.contexts(1, variant);
    assert!(renewed[0] != guest[0] && renewed[1] != guest[1]);
    host(&mut machine, true);
    assert_eq!(machine.contexts(2, variant), kept);
    machine.tlb_maintained(Reach::All, Everywhere::All, None);
    let after = machine.contexts(2, variant);
    assert!(after[0] != kept[0] && after[1] != kept[1]);
}

/// More translation input values than [`CONTEXT_FRAMES`] forget every
/// context, and both halves' contexts are then of the same generation:
/// neither names a context the other's frame forgot.
#[test]
fn a_full_frame_table_forgets_every_context_and_names_live_ones() {
    const MSR_MAIR_EL1_X0: u32 = 0xd518_a200;
    let mut machine = machine(&[]);
    let el = machine.cpu.pstate.el;
    let variant = machine.current_variant(false, false);
    let first = machine.contexts(el, variant);
    let others = machine.statistics.lookup_forgets[3];
    for mair in 1..=CONTEXT_FRAMES as u64 + 1 {
        machine.cpu.state.x[0] = mair;
        machine.system_instruction(MSR_MAIR_EL1_X0).unwrap();
        let contexts = machine.contexts(el, variant);
        for (spaces, context) in machine.spaces.iter().zip(contexts) {
            assert!(spaces.live(context).is_some());
        }
    }
    assert!(machine.statistics.lookup_forgets[3] > others);
    assert!(machine.spaces[0].live(first[0]).is_none());
    assert!(machine.context_frames.len() <= CONTEXT_FRAMES);
}

/// Host nanoseconds of an engine fetch that refills the ITLB after a TLBI
/// of everything, against one that finds its page, and of the walk alone.
#[test]
#[ignore = "manual: opt-in"]
fn itlb_refill_cost() {
    use std::time::Instant;
    const RET: u32 = 0xd65f_03c0;
    const PAGES: u64 = 8;
    let mut words = vec![0u32; (PAGES * 0x1000 / 4) as usize];
    for page in 0..PAGES {
        words[(page * 0x400) as usize] = RET;
    }
    let mut machine = machine(&words);
    let frames: Vec<u64> = (0..PAGES).map(|page| BASE + page * 0x1000).collect();
    paged_space(&mut machine, 0, &frames);
    let fetch = |machine: &mut Machine<FakeHost, NoDevices>, page: u64| {
        assert!(matches!(
            machine.fetch_block(BASE + page * 0x1000, false),
            Ok(Fetched::Block(_))
        ));
    };
    for page in 0..PAGES {
        fetch(&mut machine, page);
    }
    const ROUNDS: u32 = 20_000;
    let time = |machine: &mut Machine<FakeHost, NoDevices>, forget: bool| {
        let started = Instant::now();
        for _ in 0..ROUNDS {
            if forget {
                machine.tlb_maintained(Reach::All, Everywhere::All, None);
            }
            for page in 0..PAGES {
                fetch(machine, page);
            }
        }
        started.elapsed().as_nanos() as f64 / f64::from(ROUNDS) / PAGES as f64
    };
    let hit = time(&mut machine, false);
    let refill = time(&mut machine, true);
    let walk = {
        let started = Instant::now();
        for _ in 0..ROUNDS {
            machine.advance_translation_generation();
            for page in 0..PAGES {
                assert!(machine.fetch_physical(BASE + page * 0x1000).is_ok());
            }
        }
        started.elapsed().as_nanos() as f64 / f64::from(ROUNDS) / PAGES as f64
    };
    let forget = {
        let started = Instant::now();
        for _ in 0..ROUNDS * 8 {
            machine.tlb_maintained(Reach::All, Everywhere::All, None);
        }
        started.elapsed().as_nanos() as f64 / f64::from(ROUNDS * 8)
    };
    let variant = machine.current_variant(false, false);
    let el = machine.cpu.pstate.el;
    let keys: Vec<u64> = (0..PAGES)
        .map(|page| machine.current_key(BASE + page * 0x1000).expect("held"))
        .collect();
    let contexts = {
        let started = Instant::now();
        for _ in 0..ROUNDS {
            machine.tlb_maintained(Reach::All, Everywhere::All, None);
            let contexts = machine.contexts(el, variant);
            machine.chain.set_contexts(contexts);
        }
        started.elapsed().as_nanos() as f64 / f64::from(ROUNDS) - forget
    };
    let entries = {
        let started = Instant::now();
        for _ in 0..ROUNDS {
            machine.tlb_maintained(Reach::All, Everywhere::All, None);
            let contexts = machine.contexts(el, variant);
            machine.chain.set_contexts(contexts);
            for page in 0..PAGES {
                machine.translate_page(BASE + page * 0x1000, keys[page as usize]);
            }
        }
        (started.elapsed().as_nanos() as f64 / f64::from(ROUNDS) - forget - contexts) / PAGES as f64
    };
    println!(
        "fetch found {hit:.0} ns, refill {refill:.0} ns per page (incl. 1/{PAGES} of a forget of {forget:.0} ns and of contexts {contexts:.0} ns), walk {walk:.0} ns, ITLB fill {entries:.0} ns"
    );
}

/// After a TLBI of everything, the first fetch in a context made anew for
/// a forgotten context's key walks again the pages that context took and
/// gives the ITLB those that translate, now: a page unmapped since is left
/// to its fetch (no fault, no state change), and a page remapped since
/// takes its new frame's key, never the old one.
#[test]
fn contexts_made_anew_walk_again_their_predecessors_pages() {
    const RET: u32 = 0xd65f_03c0;
    const PAGES: u64 = 6;
    let mut words = vec![0u32; (PAGES * 0x1000 / 4) as usize];
    for page in 0..PAGES {
        words[(page * 0x400) as usize] = RET;
    }
    let mut machine = machine(&words);
    let frames: Vec<u64> = (0..PAGES).map(|page| BASE + page * 0x1000).collect();
    paged_space(&mut machine, 0, &frames);
    let pc = |page: u64| BASE + page * 0x1000;
    let fetch = |machine: &mut Machine<FakeHost, NoDevices>, page: u64| {
        assert!(matches!(
            machine.fetch_block(pc(page), false),
            Ok(Fetched::Block(_))
        ));
    };
    for page in 0..PAGES {
        fetch(&mut machine, page);
    }
    let keys: Vec<u64> = (0..PAGES)
        .map(|page| machine.current_key(pc(page)).expect("held"))
        .collect();
    let before = machine.chain.contexts();
    machine.tlb_maintained(Reach::All, Everywhere::All, None);
    // Page 4 is unmapped and page 5 maps page 3's frame since the TLBI.
    let [_, _, l3] = SPACES_TABLES[0];
    write_entry(&mut machine, l3 + 8 * 4, 0);
    write_entry(&mut machine, l3 + 8 * 5, page_descriptor(frames[3], false));
    let (esr, far, elr) = (
        machine.cpu.read(SystemRegister::ESR_EL1),
        machine.cpu.read(SystemRegister::FAR_EL1),
        machine.cpu.read(SystemRegister::ELR_EL1),
    );
    let misses = machine.statistics.itlb[0];
    fetch(&mut machine, 0);
    assert_ne!(machine.chain.contexts(), before);
    assert_eq!(machine.statistics.itlb[0], misses);
    assert_eq!(
        machine.statistics.itlb_sources[ItlbSource::Refilled as usize],
        5
    );
    for page in 0..4 {
        assert_eq!(machine.current_key(pc(page)), Some(keys[page as usize]));
    }
    assert_eq!(machine.current_key(pc(4)), None);
    assert_eq!(machine.current_key(pc(5)), Some(keys[3]));
    assert_ne!(keys[5], keys[3]);
    assert_eq!(
        (
            machine.cpu.read(SystemRegister::ESR_EL1),
            machine.cpu.read(SystemRegister::FAR_EL1),
            machine.cpu.read(SystemRegister::ELR_EL1),
            machine.cpu.pc,
        ),
        (esr, far, elr, machine.cpu.pc)
    );
    // The pages it found need no walk of their own.
    for page in 1..4 {
        fetch(&mut machine, page);
    }
    assert_eq!(machine.statistics.itlb[0], misses);
}

/// A TLBI by VA and ASID of a lower-half page reaches that ASID's entries
/// and, in every context, the entries whose walk ended in a global leaf,
/// as a global entry matches every ASID (Arm `TLBIMatch`). Another ASID's
/// non-global entry of the same page stays, even inside the span of global
/// pages (a process's own copy of a shared library page).
#[test]
fn a_tlbi_by_va_and_asid_reaches_other_asids_only_at_their_global_entries() {
    const LOW: u64 = 0x1_8000_0000;
    const HIGH: u64 = 0x2_8000_0000;
    const OWN: u64 = 0x2_0000_0000;
    let mut machine = machine(&[]);
    let el = machine.cpu.pstate.el;
    let ttbr0 = machine.cpu.read(SystemRegister::TTBR0_EL1);
    let mine = context_of(&mut machine, el, ttbr0 + 0x10_0000, 5)[0];
    let other = context_of(&mut machine, el, ttbr0 + 0x20_0000, 6)[0];
    let third = context_of(&mut machine, el, ttbr0 + 0x30_0000, 7)[0];
    machine.fetched_through(0, LOW, 0x4000, true);
    machine.fetched_through(0, HIGH, 0x4000, true);
    machine.fetched_through(0, OWN, 0x4000, false);
    let put_global = |machine: &mut Machine<FakeHost, NoDevices>, page: u64, context: u64| {
        let slot = machine.itlb.page_slot(page, context);
        let entry = chain::LookupEntry::global_page(page, context, 1);
        machine.itlb.insert(slot, entry);
    };
    for context in [mine, other] {
        put(&mut machine, OWN, context);
        put_global(&mut machine, LOW, context);
    }
    // The third process maps LOW privately (nG=1).
    put(&mut machine, LOW, third);
    let tlbi = |machine: &mut Machine<FakeHost, NoDevices>, page: u64, asid: Option<u16>| {
        let reach = Reach::Containing(page);
        machine.tlb_maintained(reach, Everywhere::Addresses(reach), asid);
    };
    tlbi(&mut machine, OWN, Some(5));
    assert!(!held(&machine, OWN, mine));
    assert!(held(&machine, OWN, other));
    tlbi(&mut machine, LOW, Some(5));
    assert!(!held(&machine, LOW, mine));
    assert!(!held(&machine, LOW, other));
    assert!(held(&machine, LOW, third));
    // Its own ASID reaches its private entry.
    tlbi(&mut machine, LOW, Some(7));
    assert!(!held(&machine, LOW, third));
    // A TLBI by VA of all ASIDs reaches every context's entry.
    put(&mut machine, OWN, mine);
    tlbi(&mut machine, OWN, None);
    assert!(!held(&machine, OWN, mine));
    assert!(!held(&machine, OWN, other));
}
