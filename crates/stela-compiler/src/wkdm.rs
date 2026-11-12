//! Apple WKdm page compression instructions, WKDMC and WKDMD.
//!
//! The words sit in the reserved A64 space below AMX (Asahi Linux, "Apple
//! Proprietary Instructions"): `0x00200800 | Xd << 5 | Xs` compresses and
//! `0x00200c00 | Xd << 5 | Xs` decompresses. Xs holds the source address and
//! receives the result; Xd holds the destination address. The 26A434
//! kernels of M2 to M6 use both with registers X0/X1 (VM compressor), X8/X9
//! (hibernation image writer) and X19/X20 (hibernation restore), which
//! confirms the register fields.
//!
//! The result register layout is recovered from those callers:
//!
//! - bits 7:0, compress only: the compressed size in 64-byte units, less
//!   one; a size of 16 KiB means the page did not compress;
//! - bit 8, compress only: every 32-bit word of the page is equal, so the
//!   caller keeps that value and the compressed size is 0;
//! - bits 13:12: [`Status`];
//! - bits 49:32: a population count, which the kernel stores with the slot
//!   at compression and compares with the one decompression returns.
//!
//! The hardware's compressed format is not public. [`codec`] defines this
//! machine's own WKdm-structured format; only WKDMD on the same machine
//! reads what WKDMC writes (see the module's evidence).
pub mod codec;

use serde::Serialize;

/// Bits 31:11 of both words.
const SPACE_MASK: u32 = 0xffff_f800;
const SPACE: u32 = 0x0020_0800;
/// Bit 10 selects decompression.
const DECOMPRESS_BIT: u32 = 1 << 10;
const REGISTER_MASK: u32 = 0x1f;
const DESTINATION_SHIFT: u32 = 5;

/// WKDMC X0, X1 and WKDMD X0, X1, the VM compressor's words.
pub const COMPRESS_X0_X1: u32 = 0x0020_0820;
pub const DECOMPRESS_X0_X1: u32 = 0x0020_0c20;

/// The page the instructions work on: macOS's 16 KiB page, the only size
/// the size field (at most 256 units of 64 bytes) can describe.
pub const PAGE_BYTES: usize = 16 * 1024;
pub const PAGE_WORDS: usize = PAGE_BYTES / 4;
/// Compressed data is 64-byte aligned (XNU `__PLATFORM_WKDM_ALIGNMENT_
/// BOUNDARY__` in `H13.h`, `H16.h` and `H17.h`) and sized in 64-byte units.
pub const UNIT_BYTES: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Operation {
    Compress,
    Decompress,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct Instruction {
    pub operation: Operation,
    /// Xs: the source address, then the result. Field value 31 is kept as
    /// decoded; whether it names XZR or SP is not established.
    pub source: u8,
    /// Xd: the destination address.
    pub destination: u8,
}

impl Instruction {
    /// Ignored low bits of the source and destination addresses (Asahi
    /// Linux: "bottom bits ignored"): a page for the uncompressed side and
    /// a 64-byte unit for the compressed side.
    pub const fn address_masks(self) -> (u64, u64) {
        const PAGE: u64 = !(PAGE_BYTES as u64 - 1);
        const UNIT: u64 = !(UNIT_BYTES as u64 - 1);
        match self.operation {
            Operation::Compress => (PAGE, UNIT),
            Operation::Decompress => (UNIT, PAGE),
        }
    }
}

pub const fn decode(word: u32) -> Option<Instruction> {
    if word & SPACE_MASK != SPACE {
        return None;
    }
    Some(Instruction {
        operation: if word & DECOMPRESS_BIT == 0 {
            Operation::Compress
        } else {
            Operation::Decompress
        },
        source: (word & REGISTER_MASK) as u8,
        destination: (word >> DESTINATION_SHIFT & REGISTER_MASK) as u8,
    })
}

/// The status field, bits 13:12. The 26A434 VM compressor panics on 1
/// after a compression; after a decompression, on an M2 whose `arm-io`
/// `chip-revision` is 0, it retries once from a copy of the source
/// ("Recursive WKDM workaround"), and otherwise fails. The hibernation
/// callers test bits 14:12 and stop on any nonzero value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Complete = 0,
    Error = 1,
    /// The page did not compress and the destination holds nothing useful
    /// (compression), or the source is not valid compressed data
    /// (decompression, this machine's choice).
    Incompressible = 2,
    /// The page did not compress and was copied to the destination; the
    /// kernel then skips its own copy.
    CopiedIncompressible = 3,
}

const SIZE_MASK: u64 = 0xff;
const SINGLE_VALUE: u64 = 1 << 8;
const STATUS_SHIFT: u32 = 12;
const POPULATION_SHIFT: u32 = 32;
/// 18 bits: 0 to 8 * 16384 set bits (XNU `C_SLOT_NO_POPCOUNT`, `(16 <<
/// C_SLOT_C_SIZE_BITS) - 1` with 14 size bits, is the all-ones "unknown").
pub const POPULATION_MASK: u64 = (1 << 18) - 1;

/// What the population count field counts. XNU's platform headers say
/// which: `__APPLE_WKDM_POPCNT_COMPRESSED_DATA__` is 0 for H13 (M1,
/// `xnu-8792`) and 1 for H16 and H17 (M4 and M5, `xnu-12377`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Population {
    /// The uncompressed page: the source of WKDMC, the destination of WKDMD.
    Uncompressed,
    /// The compressed data, padded to whole units: what WKDMC writes, what
    /// WKDMD reads.
    Compressed,
}

/// The result register value.
pub fn result(status: Status, units: Option<usize>, single_value: bool, population: u32) -> u64 {
    let size = units.map_or(0, |units| (units as u64 - 1) & SIZE_MASK);
    let single_value = if single_value { SINGLE_VALUE } else { 0 };
    size | single_value
        | (status as u64) << STATUS_SHIFT
        | (u64::from(population) & POPULATION_MASK) << POPULATION_SHIFT
}

/// Set bits in `bytes`.
pub fn population(bytes: &[u8]) -> u32 {
    // Eight bytes per count: a page is 2048 counts, not 16384, and one
    // POPCNT each where the host has it.
    let (chunks, rest) = bytes.as_chunks::<8>();
    let wide = if popcnt::available() {
        popcnt::count(chunks)
    } else {
        population_words(chunks)
    };
    wide + rest.iter().map(|byte| byte.count_ones()).sum::<u32>()
}

/// The set bits of `chunks`, in whatever instructions the caller's target
/// features allow.
#[inline(always)]
fn population_words(chunks: &[[u8; 8]]) -> u32 {
    chunks
        .iter()
        .map(|chunk| u64::from_ne_bytes(*chunk).count_ones())
        .sum()
}

/// The POPCNT tier of [`population`]: the baseline x86-64 target counts
/// bits in about 15 instructions per word without it.
mod popcnt {
    use core::sync::atomic::{AtomicU8, Ordering::Relaxed};

    /// 0 not asked yet, 1 absent, 2 present.
    static PRESENT: AtomicU8 = AtomicU8::new(0);

    pub(super) fn available() -> bool {
        match PRESENT.load(Relaxed) {
            0 => {
                let present = detect();
                PRESENT.store(if present { 2 } else { 1 }, Relaxed);
                present
            }
            known => known == 2,
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn detect() -> bool {
        // Leaf 1 exists on every x86-64 processor; ECX bit 23 is POPCNT.
        core::arch::x86_64::__cpuid(1).ecx & (1 << 23) != 0
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn detect() -> bool {
        false
    }

    pub(super) fn count(chunks: &[[u8; 8]]) -> u32 {
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: `available` saw POPCNT in CPUID before any caller
            // reaches this.
            unsafe { with_popcnt(chunks) }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            super::population_words(chunks)
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "popcnt")]
    unsafe fn with_popcnt(chunks: &[[u8; 8]]) -> u32 {
        super::population_words(chunks)
    }

    #[cfg(test)]
    pub(super) fn both(chunks: &[[u8; 8]]) -> (u32, Option<u32>) {
        (
            super::population_words(chunks),
            available().then(|| count(chunks)),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_the_register_fields_of_every_kernel_site() {
        // M2-M6 26A434: the VM compressor, the hibernation writer and the
        // hibernation restore.
        for (word, operation, source, destination) in [
            (COMPRESS_X0_X1, Operation::Compress, 0, 1),
            (DECOMPRESS_X0_X1, Operation::Decompress, 0, 1),
            (0x0020_0928, Operation::Compress, 8, 9),
            (0x0020_0e93, Operation::Decompress, 19, 20),
        ] {
            assert_eq!(
                decode(word),
                Some(Instruction {
                    operation,
                    source,
                    destination
                }),
                "{word:#x}"
            );
        }
        // MUL53 below and AMX above are other spaces.
        for word in [0x0020_07ff, 0x0020_1000, 0x0020_1420, 0x0030_0820] {
            assert_eq!(decode(word), None, "{word:#x}");
        }
        assert_eq!(decode(0x0020_0fff).unwrap().destination, 31);
    }

    #[test]
    fn address_masks_keep_a_page_and_a_unit() {
        let compress = decode(COMPRESS_X0_X1).unwrap().address_masks();
        assert_eq!(compress, (!0x3fff, !0x3f));
        let decompress = decode(DECOMPRESS_X0_X1).unwrap().address_masks();
        assert_eq!(decompress, (!0x3f, !0x3fff));
    }

    #[test]
    fn result_fields_follow_the_kernel_decoding() {
        // The kernel: size = ((r & 0xff) << 6) + 64; single value r & 0x100;
        // status (r >> 12) & 3; population (r >> 32) & 0x3ffff.
        let value = result(Status::Complete, Some(3), false, 0x2_0000);
        assert_eq!(((value & 0xff) << 6) + 64, 3 * 64);
        assert_eq!(value >> 12 & 3, 0);
        assert_eq!(value >> 32 & 0x3ffff, 0x2_0000);
        assert_eq!(value & !(0xff | 0x3_ffff << 32), 0);
        let single = result(Status::Complete, None, true, 7);
        assert_eq!(single, 0x100 | 7 << 32);
        assert_eq!(
            result(Status::Incompressible, None, false, 0) >> 12 & 7,
            2,
            "the hibernation callers' 0x7000 test fails it"
        );
        assert_eq!(result(Status::Complete, Some(255), false, 0) & 0xff, 254);
    }
}
