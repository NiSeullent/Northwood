//! The guest kernel's panic region (`/pram reg`, carveout 98, which the
//! loader publishes), read without the guest's help when it went down
//! without printing a panic: XNU writes its panic log there before the
//! console sees it, and a CPU that never answers the debugger cross-call
//! (one parked in SPTM, say) keeps the console from ever seeing it.
//!
//! Layout: XNU `osfmk/kern/debug.h` `struct embedded_panic_header`, packed:
//! `eph_magic` u32 at 0 (`EMBEDDED_PANIC_MAGIC`, 'CNUF'), `eph_crc` at 4,
//! `eph_version` at 8, `eph_panic_flags` u64 at 12, `eph_panic_log_offset`
//! u32 at 20 and `eph_panic_log_len` u32 at 24. `PE_init_panicheader` sets
//! the magic and the log offset (past the header) when a CPU starts to
//! panic; the length is written only when the log is flushed, so the text
//! is read up to its first NUL instead.

use alloc::string::String;

/// `EMBEDDED_PANIC_MAGIC`.
pub const MAGIC: u32 = 0x4655_4e43;
/// Bytes of the header that this reader needs.
pub const HEADER_BYTES: usize = 28;
/// The most text one report shows.
pub const TEXT_LIMIT: usize = 2048;

/// What the panic region held.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Contents {
    /// No panic header: the kernel has not begun a panic (or never mapped
    /// the region).
    NoHeader,
    /// A header whose log offset lies outside the region.
    BadOffset(u32),
    /// A header with no log text yet.
    Empty,
    /// The log text, with its unprintable bytes shown as `.`, cut at
    /// [`TEXT_LIMIT`]; `truncated` says whether more followed.
    Text { text: String, truncated: bool },
}

/// The panic log `region` (the start of the panic region) holds.
pub fn read(region: &[u8]) -> Contents {
    let word = |at: usize| {
        region
            .get(at..at + 4)
            .map(|bytes| u32::from_le_bytes(bytes.try_into().expect("4 bytes")))
    };
    if word(0) != Some(MAGIC) {
        return Contents::NoHeader;
    }
    let Some(offset) = word(20) else {
        return Contents::NoHeader;
    };
    let start = offset as usize;
    if start < HEADER_BYTES || start >= region.len() {
        return Contents::BadOffset(offset);
    }
    let log = &region[start..];
    let end = log.iter().position(|&byte| byte == 0).unwrap_or(log.len());
    let shown = end.min(TEXT_LIMIT);
    if shown == 0 {
        return Contents::Empty;
    }
    let text = log[..shown]
        .iter()
        .map(|&byte| {
            if byte.is_ascii_graphic() || b" \t\n".contains(&byte) {
                char::from(byte)
            } else {
                '.'
            }
        })
        .collect();
    Contents::Text {
        text,
        truncated: end > shown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{vec, vec::Vec};

    /// A region with the header `PE_init_panicheader` writes and the log
    /// 0x331 bytes in, where 26A434 XNU's `debug_log_init` puts it.
    fn region(log: &[u8]) -> Vec<u8> {
        let mut region = vec![0u8; 0x1000];
        region[..4].copy_from_slice(&MAGIC.to_le_bytes());
        region[20..24].copy_from_slice(&0x331u32.to_le_bytes());
        region[0x331..0x331 + log.len()].copy_from_slice(log);
        region
    }

    #[test]
    fn the_log_after_the_header_is_read_to_its_first_nul() {
        let contents = read(&region(
            b"panic(cpu 2 caller 0xfffffe0037a1d94): cannot rewrap ek 12, 3\n\0stale",
        ));
        assert_eq!(
            contents,
            Contents::Text {
                text: "panic(cpu 2 caller 0xfffffe0037a1d94): cannot rewrap ek 12, 3\n".into(),
                truncated: false,
            }
        );
    }

    #[test]
    fn a_region_without_the_magic_has_no_panic() {
        let mut bytes = region(b"panic(cpu 0 caller 0x1): old\n");
        bytes[0] = 0;
        assert_eq!(read(&bytes), Contents::NoHeader);
        assert_eq!(read(&[]), Contents::NoHeader);
        assert_eq!(read(&MAGIC.to_le_bytes()), Contents::NoHeader);
    }

    #[test]
    fn a_header_without_text_or_with_a_wild_offset_says_so() {
        assert_eq!(read(&region(b"")), Contents::Empty);
        let mut bytes = region(b"x");
        bytes[20..24].copy_from_slice(&0x2000u32.to_le_bytes());
        assert_eq!(read(&bytes), Contents::BadOffset(0x2000));
        bytes[20..24].copy_from_slice(&4u32.to_le_bytes());
        assert_eq!(read(&bytes), Contents::BadOffset(4));
    }

    #[test]
    fn long_or_binary_text_is_cut_and_masked() {
        let mut log = vec![b'a'; TEXT_LIMIT + 10];
        log[1] = 0x07;
        let Contents::Text { text, truncated } = read(&region(&log)) else {
            panic!("text");
        };
        assert!(truncated);
        assert_eq!(text.len(), TEXT_LIMIT);
        assert!(text.starts_with("a.a"));
    }
}
