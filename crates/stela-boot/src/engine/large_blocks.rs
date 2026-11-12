//! The engine heap's large blocks and the code that made them: each live
//! allocation of at least `heap::TRACKED_BYTES`, with the image offsets on
//! its allocating stack, for `HEAP LARGE` lines that `stela.pdb` resolves.
//! The heap calls [`allocated_or_freed`] only for such blocks, a few
//! hundred live at most, so the cost stays off every ordinary allocation.
use crate::output;
use alloc::{format, string::String, vec::Vec};
use core::fmt::Write;
use core::sync::atomic::{AtomicU64, Ordering};

/// Live large blocks kept; past them a block is only counted.
const SLOTS: usize = 256;
/// Image offsets kept from each allocating stack.
const WORDS: usize = 12;
/// The stack words searched for them.
const SCANNED: usize = 512;

struct Slot {
    /// The payload's address; zero for a free slot.
    pointer: AtomicU64,
    bytes: AtomicU64,
    words: [AtomicU64; WORDS],
}

static BLOCKS: [Slot; SLOTS] = [const {
    Slot {
        pointer: AtomicU64::new(0),
        bytes: AtomicU64::new(0),
        words: [const { AtomicU64::new(0) }; WORDS],
    }
}; SLOTS];
/// Large blocks made while every slot was taken, which no line names: a
/// count of blocks made, not of live ones.
static UNTRACKED: AtomicU64 = AtomicU64::new(0);
/// Large blocks whose stack scan found no code offset.
static EMPTY: AtomicU64 = AtomicU64::new(0);
/// Blocks made or freed so far, and the count the last report saw.
static CHANGES: AtomicU64 = AtomicU64::new(0);
static REPORTED: AtomicU64 = AtomicU64::new(0);
/// Heartbeats between reports, and the reports asked for so far.
const REPORT_CALLS: u64 = 6;
static CALLS: AtomicU64 = AtomicU64::new(0);

/// Lets the heap tell this module of its large blocks.
pub(super) fn install() {
    stela_boot_core::heap::set_large_hook(allocated_or_freed);
}

/// The heap's large-block hook. It runs on any processor, outside the
/// heap's lock, so it neither allocates nor waits.
fn allocated_or_freed(pointer: usize, bytes: usize, allocated: bool) {
    let pointer = pointer as u64;
    CHANGES.fetch_add(1, Ordering::Relaxed);
    if !allocated {
        // The heap calls this before the block can be handed out again.
        if let Some(slot) = BLOCKS
            .iter()
            .find(|slot| slot.pointer.load(Ordering::Relaxed) == pointer)
        {
            slot.bytes.store(0, Ordering::Relaxed);
            slot.pointer.store(0, Ordering::Release);
        }
        return;
    }
    let Some(slot) = BLOCKS.iter().find(|slot| {
        slot.pointer
            .compare_exchange(0, pointer, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
    }) else {
        UNTRACKED.fetch_add(1, Ordering::Relaxed);
        return;
    };
    let mut words = [0; WORDS];
    if !stack_offsets(&mut words) {
        EMPTY.fetch_add(1, Ordering::Relaxed);
    }
    for (word, offset) in slot.words.iter().zip(words) {
        word.store(offset, Ordering::Relaxed);
    }
    slot.bytes.store(bytes as u64, Ordering::Release);
}

/// The image offsets among the stack's words from the stack pointer on, at
/// most [`SCANNED`] words: its own page, which is mapped, then the page above
/// through a read that a fault ends, as that page may be the stack's guard.
/// Whether any was found.
fn stack_offsets(out: &mut [u64; WORDS]) -> bool {
    const PAGE: u64 = 4096;
    unsafe extern "C" {
        static __ImageBase: u8;
    }
    let base = &raw const __ImageBase as u64;
    // SAFETY: the mapped PE headers of this image: `e_lfanew`, then the
    // optional header's SizeOfCode and BaseOfCode.
    let (code, size) = unsafe {
        let header = u64::from((base as *const u32).byte_add(0x3c).read_unaligned());
        let pe = (base + header) as *const u32;
        (
            u64::from(pe.byte_add(0x2c).read_unaligned()),
            u64::from(pe.byte_add(0x1c).read_unaligned()),
        )
    };
    let stack: *const u64;
    // SAFETY: reads the stack pointer.
    unsafe {
        core::arch::asm!("mov {}, rsp", out(reg) stack, options(nomem, nostack, preserves_flags));
    }
    let own = ((PAGE - stack as u64 % PAGE) / 8) as usize;
    let mut kept = 0;
    for word in 0..SCANNED {
        let value = if word < own {
            // SAFETY: within the stack pointer's own page.
            unsafe { stack.add(word).read() }
        } else {
            let mut value = 0u64;
            // SAFETY: the engine's host fault handler is active (the hook is
            // installed after it), and `value` is this frame's own; a fault
            // ends the read with bytes unread.
            let unread = unsafe {
                stela_boot_core::nofault_read::read(
                    (&raw mut value).cast(),
                    stack.wrapping_add(word).cast(),
                    8,
                )
            };
            if unread != 0 {
                break;
            }
            value
        };
        let offset = value.wrapping_sub(base);
        if offset.wrapping_sub(code) < size {
            out[kept] = offset;
            kept += 1;
            if kept == WORDS {
                break;
            }
        }
    }
    kept != 0
}

/// `HEAP LARGE blocks=31 bytes=612M untracked=0 empty_stacks=0`, then each live block,
/// largest first, as `HEAP LARGE 64M stack 0x… 0x…`: every
/// [`REPORT_CALLS`]th call, when blocks were made or freed since the last
/// report.
pub(super) fn report() {
    if !CALLS
        .fetch_add(1, Ordering::Relaxed)
        .is_multiple_of(REPORT_CALLS)
    {
        return;
    }
    let changes = CHANGES.load(Ordering::Relaxed);
    if REPORTED.swap(changes, Ordering::Relaxed) == changes {
        return;
    }
    let mut blocks: Vec<(u64, [u64; WORDS])> = BLOCKS
        .iter()
        .filter(|slot| slot.pointer.load(Ordering::Acquire) != 0)
        .map(|slot| {
            let bytes = slot.bytes.load(Ordering::Acquire);
            (
                bytes,
                core::array::from_fn(|at| slot.words[at].load(Ordering::Relaxed)),
            )
        })
        .filter(|&(bytes, _)| bytes != 0)
        .collect();
    blocks.sort_unstable_by_key(|&(bytes, _)| core::cmp::Reverse(bytes));
    output::arguments(format_args!("{}\n", summary(&blocks)));
    for (bytes, words) in &blocks {
        output::arguments(format_args!("{}\n", line(*bytes, words)));
    }
}

fn summary(blocks: &[(u64, [u64; WORDS])]) -> String {
    let bytes: u64 = blocks.iter().map(|&(bytes, _)| bytes).sum();
    format!(
        "HEAP LARGE blocks={} bytes={}M untracked={} empty_stacks={}",
        blocks.len(),
        bytes >> 20,
        UNTRACKED.load(Ordering::Relaxed),
        EMPTY.load(Ordering::Relaxed)
    )
}

fn line(bytes: u64, words: &[u64; WORDS]) -> String {
    let mut line = format!("HEAP LARGE {}M stack", bytes >> 20);
    for &offset in words.iter().filter(|&&offset| offset != 0) {
        let _ = write!(line, " {offset:#x}");
    }
    line
}
