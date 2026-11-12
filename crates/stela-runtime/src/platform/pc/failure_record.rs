//! A failed start's record, kept in the machine state's tail of the
//! partition Install macOS chose, for a stop the USB's own log cannot keep:
//! the log lives on the stick, and when Stela's drivers do not find the
//! stick, the lines after boot services end have nowhere to go. The chosen
//! partition is often found then, as it is on another bus, so the engine
//! writes the step, its message and the disk account here, and the next
//! boot's firmware-side code, which reads the stick through the firmware,
//! copies it into the USB's log and clears it.
//!
//! The region starts at [`OFFSET`] of the machine state, after the RTC
//! anchor, and is [`BYTES`] long. The installer zeroes the whole tail, so a
//! fresh partition holds no record.
//!
//! | Offset | Bytes | Field |
//! | --- | --- | --- |
//! | 0 | 8 | magic `STELAFRC` |
//! | 8 | 4 | version, 1 |
//! | 12 | 4 | text length |
//! | 16 | 32 | SHA-256 of the text |
//! | 48 | length | UTF-8 text, lines ending in `\n` |
use alloc::{string::String, vec, vec::Vec};
use stela_crypto::Sha256;
use stela_crypto::digest::Digest;

/// Where the region starts in the machine state: the first 4 KiB boundary
/// after the RTC anchor, which every block size up to 4096 divides.
pub const OFFSET: u64 = super::rtc_anchor::PARTITION_BYTES.next_multiple_of(4096);
/// The region's bytes.
pub const BYTES: u64 = 64 << 10;
const _: () = assert!(OFFSET + BYTES <= super::machine_state::MACHINE_STATE_BYTES);
const MAGIC: &[u8; 8] = b"STELAFRC";
const VERSION: u32 = 1;
const HEADER: usize = 48;
/// The most text a record holds.
pub const TEXT_BYTES: usize = BYTES as usize - HEADER;

/// The region holding `text`, cut at a character boundary to
/// [`TEXT_BYTES`].
pub fn encode(text: &str) -> Vec<u8> {
    let mut end = text.len().min(TEXT_BYTES);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    let text = &text.as_bytes()[..end];
    let mut region = vec![0; BYTES as usize];
    region[..8].copy_from_slice(MAGIC);
    region[8..12].copy_from_slice(&VERSION.to_le_bytes());
    region[12..16].copy_from_slice(&(text.len() as u32).to_le_bytes());
    region[16..48].copy_from_slice(&Sha256::digest(text));
    region[HEADER..HEADER + text.len()].copy_from_slice(text);
    region
}

/// The record `region` holds; `None` for a blank region, another layout or
/// a damaged record.
pub fn decode(region: &[u8]) -> Option<String> {
    if region.len() < HEADER || &region[..8] != MAGIC {
        return None;
    }
    let word = |at: usize| u32::from_le_bytes(region[at..at + 4].try_into().expect("4 bytes"));
    let length = word(12) as usize;
    if word(8) != VERSION || length > TEXT_BYTES || HEADER + length > region.len() {
        return None;
    }
    let text = &region[HEADER..HEADER + length];
    if Sha256::digest(text)[..] != region[16..48] {
        return None;
    }
    String::from_utf8(text.to_vec()).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_record_reads_back_and_blank_or_damaged_regions_hold_none() {
        let text = "STEP Starting the PC's disks and USB devices\nHOST DISKS Disk SATA\n";
        let region = encode(text);
        assert_eq!(region.len() as u64, BYTES);
        assert_eq!(decode(&region).as_deref(), Some(text));
        assert_eq!(decode(&vec![0; BYTES as usize]), None);
        let mut damaged = region.clone();
        damaged[HEADER] ^= 1;
        assert_eq!(decode(&damaged), None);
        let mut long = String::new();
        while long.len() <= TEXT_BYTES {
            long.push('\u{e9}');
        }
        let kept = decode(&encode(&long)).unwrap();
        assert!(kept.len() <= TEXT_BYTES && long.starts_with(&kept));
        assert_eq!(OFFSET % 4096, 0);
    }
}
