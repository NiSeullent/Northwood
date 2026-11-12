//! `--profile`'s `CHAIN CHECK` line: what one chained transfer costs on this
//! PE, measured once at its start with the engine's own emitted transfer code
//! (`stela_compiler::x86_64::chain`) on this core's stack, heap and code
//! arena, and nothing else of the engine's.
//!
//! Four rings of 32 stub blocks, each block only its exit, alternate between
//! two guest pages as far transfers do:
//! - `near`: thunks ([`chain::thunk`]): the budget, the gate and a jump, as
//!   direct exits in their page take;
//! - `cell`: cell tails ([`chain::cell_tail`]) through the cell probe, as
//!   direct exits out of their page take;
//! - `table`: lookup tails ([`chain::lookup_tail`]) through the table probe,
//!   as RET, BR and BLR take;
//! - `stamped`: stamped tails ([`chain::stamped_tail`]), as hot direct exits
//!   out of their page take: a direct jump while this PE's stamp holds, after
//!   the first transfer's stamping cell probe.
//!
//! Each runs until its budget of transfers is spent; the line
//! (`stela_boot_core::chain_check::transfer_line`) gives the best of three
//! runs in ns per transfer, the share of the stamped ring's transfers its
//! stamps sent straight on, each transfer as a multiple of this core's own
//! host CALL and RET pair, then the pieces of a probe transfer apart
//! ([`RAW`]). Every figure is measured here, so the line needs no outside
//! baseline.
use super::local;
use crate::{apic::rdtsc, output};
use alloc::vec::Vec;
use stela_compiler::{
    Anchor, GuestState,
    block::ExitKind,
    x86_64::{
        chain::{self, ChainControl, ExitWord, LookupEntry, LookupTable},
        pinned,
    },
};
use stela_engine::machine::{Host, NativeCode};

/// Blocks per ring: half in each page.
const BLOCKS: usize = 32;
/// Transfers per timed run.
const TRANSFERS: u64 = 1 << 20;
/// Bytes per stub block, per tail and per thunk.
const STRIDE: usize = 256;
const _: () = assert!(STRIDE >= chain::STAMPED_TAIL_BYTES);
const PAGES: [u64; 2] = [0x1_8000_0000, 0x1_8000_1000];
const DISPATCHER: usize = 0;
const ENTRY: usize = 0x1000;
const BLOCKS_AT: usize = 0x2000;
const TAILS_AT: usize = BLOCKS_AT + BLOCKS * STRIDE;
const RAW_AT: usize = TAILS_AT + BLOCKS * STRIDE;
const BYTES: usize = RAW_AT + 0x400;
/// Loop iterations of a raw kernel, each eight of its operations.
const LOOPS: u64 = 1 << 17;
/// Short runs of a raw kernel, and the loop iterations of each.
const BURSTS: usize = 4096;
const BURST_LOOPS: u64 = 64;

/// The pieces of a probe transfer apart, each eight times a loop
/// (`fn(loops: u64, data: *mut u64)`): a host CALL and RET pair on this
/// stack; PUSH of two memory words and two POPs; a read-modify-write of one
/// heap word, as the budget's; an indirect jump to the next instruction, as
/// a tail's into its block.
const RAW: [(&str, &[u8]); 4] = [
    ("callret", &[0xe8, 0, 0, 0, 0]),
    ("push", &[0xff, 0x36, 0xff, 0x76, 0x08, 0x58, 0x58]),
    ("rmw", &[0x48, 0x83, 0x2e, 0x01]),
    // lea r9, [rip + 3]; jmp r9
    ("jump", &[0x4c, 0x8d, 0x0d, 0x03, 0, 0, 0, 0x41, 0xff, 0xe1]),
];

/// Raw kernel `kernel` of [`RAW`]: its operation eight times, `dec rdi`,
/// `jnz` back, `ret`; for `callret`, then the `ret` the calls reach.
fn raw_code(kernel: usize) -> Vec<u8> {
    let (_, op) = RAW[kernel];
    let mut code = loop_code(op);
    if kernel == 0 {
        // Each call's rel32: the `ret` after the loop's.
        let target = code.len();
        for call in 0..8 {
            let end = 5 * (call + 1);
            code[end - 4..end].copy_from_slice(&((target - end) as i32).to_le_bytes());
        }
        code.push(0xc3);
    }
    code
}

/// `op` eight times, `dec rdi`, `jnz` back, `ret`.
fn loop_code(op: &[u8]) -> Vec<u8> {
    let mut code = Vec::new();
    for _ in 0..8 {
        code.extend_from_slice(op);
    }
    code.extend([0x48, 0xff, 0xcf, 0x0f, 0x85]);
    let back = -(code.len() as i32 + 4);
    code.extend(back.to_le_bytes());
    code.push(0xc3);
    code
}

/// `HOST BRANCH`'s kernels: an indirect jump to the next instruction (the
/// same target every time, so a predicting core predicts it) and a direct
/// jump to the next instruction.
const BRANCH: [&[u8]; 2] = [
    // lea r9, [rip + 3]; jmp r9
    &[0x4c, 0x8d, 0x0d, 0x03, 0, 0, 0, 0x41, 0xff, 0xe1],
    // jmp +0
    &[0xe9, 0, 0, 0, 0],
];
/// `HOST BRANCH`'s bursts, and the loop iterations (eight jumps each) of
/// one: about 0.4 ms in all where an indirect jump costs 3 ns.
const BRANCH_BURSTS: usize = 256;
const BRANCH_LOOPS: u64 = 64;
/// Code bytes for one `HOST BRANCH` kernel.
const BRANCH_BYTES: usize = 256;

/// Print `HOST BRANCH`: this core's best ns per predicted indirect jump and
/// per direct jump over short bursts (a burst no interruption spoiled), so
/// every boot log, bare metal too, shows whether indirect jumps are
/// predicted on this host (FACTS item 11 measured 14 cycles each in a
/// guest). Under a millisecond, once, on the boot PE.
pub(crate) fn host_branch(host: &mut impl Host) {
    let Ok(installed) = host.install(&[0xcc; BRANCH_BYTES]) else {
        output::arguments(format_args!(
            "HOST BRANCH unavailable: no room in the host's code space\n"
        ));
        return;
    };
    let hz = local::state().tsc_hz.max(1);
    let mut ns = [0.0_f64; 2];
    for (slot, op) in BRANCH.iter().enumerate() {
        let code = loop_code(op);
        let mut bytes = alloc::vec![0xcc_u8; BRANCH_BYTES];
        bytes[..code.len()].copy_from_slice(&code);
        host.patch(installed, &bytes);
        let mut data = 0_u64;
        type Raw = unsafe extern "sysv64" fn(u64, *mut u64);
        // SAFETY: just installed: a loop of `loops` rounds of jumps within
        // its own code that touches no memory, then returns.
        let run: Raw = unsafe { core::mem::transmute(installed.0) };
        let mut best = u64::MAX;
        for _ in 0..BRANCH_BURSTS {
            let started = rdtsc();
            // SAFETY: as above.
            unsafe { run(BRANCH_LOOPS, &mut data) };
            best = best.min(rdtsc().wrapping_sub(started));
        }
        ns[slot] = best as f64 * 1e9 / hz as f64 / (BRANCH_LOOPS * 8) as f64;
    }
    let line = stela_boot_core::chain_check::host_branch_line(ns[0], ns[1]);
    output::arguments(format_args!("{line}\n"));
}

#[derive(Clone, Copy)]
enum Ring {
    Near,
    Cell,
    Table,
    Stamped,
}

fn pc(block: usize) -> u64 {
    PAGES[block % 2] + 4 * block as u64
}

/// The ring's code, for an arena region at `base`: the dispatcher, the
/// entry, the blocks and their tails or thunks. `None` when an emitter
/// refuses (out of reach).
fn ring_code(ring: Ring, base: u64, table: &mut LookupTable, key: u64) -> Option<Vec<u8>> {
    let mut code = alloc::vec![0xcc_u8; BYTES];
    let mut put = |at: usize, bytes: &[u8]| code[at..at + bytes.len()].copy_from_slice(bytes);
    let dispatcher = chain::lookup_dispatcher(0)?;
    put(DISPATCHER, &dispatcher);
    let block_at = |block: usize| base + (BLOCKS_AT + block * STRIDE) as u64;
    let tail_at = |block: usize| base + (TAILS_AT + block * STRIDE) as u64;
    let word = ExitWord::from(ExitKind::Direct);
    for block in 0..BLOCKS {
        let next = (block + 1) % BLOCKS;
        let at = block_at(block);
        let tail = tail_at(block);
        // The block: `movabs rax, next pc; jmp tail`, as a direct exit.
        let mut exit = alloc::vec![0x48, 0xb8];
        exit.extend(pc(next).to_le_bytes());
        exit.extend(chain::jump(at + exit.len() as u64, tail)?);
        put(BLOCKS_AT + block * STRIDE, &exit);
        let tail_code = match ring {
            Ring::Near => chain::thunk(
                tail,
                1,
                next as u32,
                block_at(next),
                chain::Fallback::Return(word),
            )?,
            Ring::Cell => {
                let current = table.find(table.slot(pc(next), key), pc(next), key);
                let cell = table.link_cell(tail, pc(next), Some(current));
                let probe = base + (DISPATCHER + chain::CELL_PROBE_ENTRY) as u64;
                chain::cell_tail(tail, cell, word, 1, probe)?.to_vec()
            }
            Ring::Table => {
                let probe = base + (DISPATCHER + chain::PROBE_ENTRY) as u64;
                chain::lookup_tail(tail, word, 1, probe)?.to_vec()
            }
            Ring::Stamped => {
                let current = table.find(table.slot(pc(next), key), pc(next), key);
                let cell = table.link_cell(tail, pc(next), Some(current));
                let number = table.site_cell_number(tail)?;
                table.note_stamped(number);
                let probe = base + (DISPATCHER + chain::STAMPING_PROBE_ENTRY) as u64;
                chain::stamped_tail(
                    tail,
                    cell,
                    number,
                    pc(next),
                    word,
                    1,
                    next as u32,
                    block_at(next),
                    probe,
                )?
                .to_vec()
            }
        };
        put(TAILS_AT + block * STRIDE, &tail_code);
    }
    let entry_at = base + ENTRY as u64;
    let body_at = entry_at + pinned::entered(&[]).len() as u64;
    put(ENTRY, &pinned::entered(&chain::jump(body_at, block_at(0))?));
    Some(code)
}

/// Run each ring on this PE and print `CHAIN CHECK`.
pub(crate) fn run(pe: usize, host: &mut impl Host) {
    let Ok(installed) = host.install(&[0xcc; BYTES]) else {
        output::arguments(format_args!(
            "CHAIN CHECK pe={pe} unavailable: no room in the host's code space\n"
        ));
        return;
    };
    let mut results = [0.0_f64; 4];
    let mut stamped_hits = 0;
    for (slot, ring) in [Ring::Near, Ring::Cell, Ring::Table, Ring::Stamped]
        .into_iter()
        .enumerate()
    {
        match measure(ring, host, installed) {
            Some((ns, hits)) => {
                results[slot] = ns;
                if matches!(ring, Ring::Stamped) {
                    stamped_hits = hits;
                }
            }
            None => {
                output::arguments(format_args!(
                    "CHAIN CHECK pe={pe} unavailable: an emitter refused\n"
                ));
                return;
            }
        }
    }
    let mut raw = Vec::new();
    for (kernel, (name, _)) in RAW.iter().enumerate() {
        let code = raw_code(kernel);
        let mut bytes = alloc::vec![0xcc_u8; BYTES];
        bytes[RAW_AT..RAW_AT + code.len()].copy_from_slice(&code);
        host.patch(installed, &bytes);
        let mut data = alloc::boxed::Box::new([0_u64; 8]);
        type Raw = unsafe extern "sysv64" fn(u64, *mut u64);
        // SAFETY: just installed: a loop of `loops` rounds that touches
        // only `data` and this stack, then returns.
        let run: Raw = unsafe { core::mem::transmute(installed.0 + RAW_AT) };
        let hz = local::state().tsc_hz.max(1);
        let mut best = u64::MAX;
        for _ in 0..3 {
            let started = rdtsc();
            // SAFETY: as above.
            unsafe { run(LOOPS, data.as_mut_ptr()) };
            best = best.min(rdtsc().wrapping_sub(started));
        }
        let ns = best as f64 * 1e9 / hz as f64 / (LOOPS * 8) as f64;
        // The fastest of many short bursts: a burst that no VM exit (and
        // its predictor flush) interrupted. Equal to `ns` when the core
        // predicts as well throughout; far below it when only exits spoil it.
        let mut burst = u64::MAX;
        for _ in 0..BURSTS {
            let started = rdtsc();
            // SAFETY: as above.
            unsafe { run(BURST_LOOPS, data.as_mut_ptr()) };
            burst = burst.min(rdtsc().wrapping_sub(started));
        }
        let burst = burst as f64 * 1e9 / hz as f64 / (BURST_LOOPS * 8) as f64;
        raw.push((*name, ns, burst));
    }
    let (efer, spec_ctrl, cr4) = speculation_controls();
    output::arguments(format_args!(
        "CHAIN CHECK pe={pe} code={:#x} efer={efer:#x} spec_ctrl={spec_ctrl} cr4={cr4:#x}\n",
        installed.0
    ));
    let transfers = [
        ("near", results[0]),
        ("cell", results[1]),
        ("table", results[2]),
        ("stamped", results[3]),
    ];
    let line = stela_boot_core::chain_check::transfer_line(pe, &transfers, stamped_hits, &raw);
    output::arguments(format_args!("{line}\n"));
}

/// One ring's best ns per transfer, in the region `installed`, and the
/// percentage of its transfers that stamps sent straight on.
fn measure(ring: Ring, host: &mut impl Host, installed: NativeCode) -> Option<(f64, u64)> {
    let base = installed.0 as u64;
    let mut table = LookupTable::with_entries(chain::MIN_LOOKUP_ENTRIES);
    let mut itlb = LookupTable::itlb();
    let context = 1_u64 << chain::LOOKUP_SALT_BITS;
    let key = chain::block_key(0x4000_0000, 2);
    for block in 0..BLOCKS {
        let at = base + (BLOCKS_AT + block * STRIDE) as u64;
        let entry = LookupEntry {
            pc: pc(block),
            key,
            entry: at,
            block: block as u32,
            targets: 0b1111,
        };
        table.insert(table.slot(pc(block), key), entry);
        itlb.insert(
            itlb.page_slot(pc(block), context),
            LookupEntry::page(pc(block) & !0xfff, context, key),
        );
    }
    // As the engine runs: the cell probe counts in the stamped cells.
    let cells = table.stamped_cells();
    let stamps = table.new_stamps();
    let code = ring_code(ring, base, &mut table, key)?;
    host.patch(installed, &code);
    let mut control = alloc::boxed::Box::new(ChainControl {
        contexts: [context, context + 1],
        salts: [itlb.salt(context), itlb.salt(context + 1)],
        table: table.address(),
        itlb: itlb.address(),
        masks: [table.mask(), itlb.mask()],
        stamps,
        cells,
        ..ChainControl::default()
    });
    let mut state = alloc::boxed::Box::new(GuestState::default());
    state.anchors[Anchor::Control] = &raw mut *control as u64;
    type Entry = unsafe extern "sysv64" fn(*mut GuestState) -> u128;
    // SAFETY: installed above: the pinned entry, then a ring whose every
    // transfer stays in this region until the budget ends it with a return.
    let entry: Entry = unsafe { core::mem::transmute(base as usize + ENTRY) };
    let hz = local::state().tsc_hz.max(1);
    let mut best = u64::MAX;
    for _ in 0..3 {
        control.budget = TRANSFERS;
        let started = rdtsc();
        // SAFETY: as above; the ring touches only `state`, `control`, the
        // tables and this stack.
        core::hint::black_box(unsafe { entry(&raw mut *state) });
        best = best.min(rdtsc().wrapping_sub(started));
    }
    // SAFETY: `stamps` is the table's, which lives.
    let hits = unsafe { chain::stamp_counts(stamps) }[chain::StampCount::Hits as usize];
    drop((table, itlb));
    Some((
        best as f64 * 1e9 / hz as f64 / TRANSFERS as f64,
        hits * 100 / (3 * TRANSFERS),
    ))
}

/// EFER (AutoIBRS is bit 21), IA32_SPEC_CTRL (IBRS bit 0, STIBP 1, SSBD 2,
/// PSFD 7) where CPUID enumerates it, else "none", and CR4 (CET is bit 23):
/// what this processor runs the transfer code under. PRED_CMD is write-only.
fn speculation_controls() -> (u64, alloc::string::String, u64) {
    use core::arch::x86_64::{__cpuid, __cpuid_count};
    // Intel IBRS (CPUID 7 EDX 26); AMD IBRS (8000_0008h EBX 14), STIBP (15),
    // SSBD (24): each implies SPEC_CTRL (the set KVM also exposes it for).
    let intel = __cpuid(0).eax >= 7 && __cpuid_count(7, 0).edx & 1 << 26 != 0;
    let amd = __cpuid(0x8000_0000).eax >= 0x8000_0008
        && __cpuid(0x8000_0008).ebx & (1 << 14 | 1 << 15 | 1 << 24) != 0;
    // SAFETY: CPL0; EFER exists in long mode, SPEC_CTRL where enumerated;
    // reads have no effect.
    let efer = unsafe { crate::cpu::msr::EFER.read() };
    let spec_ctrl = if intel || amd {
        // SAFETY: as above.
        alloc::format!("{:#x}", unsafe { crate::cpu::msr::rdmsr(0x48) })
    } else {
        alloc::string::String::from("none")
    };
    let cr4: u64;
    // SAFETY: reading CR4 at CPL0 has no effect.
    unsafe {
        core::arch::asm!("mov {}, cr4", out(reg) cr4, options(nomem, nostack, preserves_flags))
    };
    (efer, spec_ctrl, cr4)
}
