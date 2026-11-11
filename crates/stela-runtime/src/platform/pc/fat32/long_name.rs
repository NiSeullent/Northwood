// SPDX-License-Identifier: GPL-2.0-or-later
//! VFAT long directory entries and their binding to the following short entry.
use super::{ENTRY_BYTES, le16};

const LAST_ENTRY: u8 = 0x40;
const ORDER_MASK: u8 = 0x1f;
const ENTRY_LIMIT: usize = 20;
const WORDS_PER_ENTRY: usize = 13;
pub(super) const NAME_LIMIT: usize = 255;
const WORD_OFFSETS: [usize; WORDS_PER_ENTRY] = [1, 3, 5, 7, 9, 14, 16, 18, 20, 22, 24, 28, 30];

pub(super) fn valid_component(name: &str) -> bool {
    !name.is_empty()
        && !name.ends_with(['.', ' '])
        && name.encode_utf16().count() <= NAME_LIMIT
        && !name.chars().any(|character| {
            character.is_control()
                || matches!(character, '"' | '*' | ':' | '<' | '>' | '?' | '\\' | '|')
        })
}

pub(super) fn checksum(short_name: &[u8]) -> u8 {
    short_name
        .iter()
        .fold(0u8, |sum, byte| sum.rotate_right(1).wrapping_add(*byte))
}

pub(super) struct Entries {
    words: [u16; ENTRY_LIMIT * WORDS_PER_ENTRY],
    total: usize,
    remaining: usize,
    checksum: u8,
}

impl Entries {
    pub(super) const fn new() -> Self {
        Self {
            words: [0; ENTRY_LIMIT * WORDS_PER_ENTRY],
            total: 0,
            remaining: 0,
            checksum: 0,
        }
    }

    pub(super) fn clear(&mut self) {
        self.total = 0;
        self.remaining = 0;
    }

    pub(super) fn push(&mut self, entry: &[u8; ENTRY_BYTES]) {
        let flags = entry[0];
        let order = usize::from(flags & ORDER_MASK);
        if flags & !(LAST_ENTRY | ORDER_MASK) != 0
            || !(1..=ENTRY_LIMIT).contains(&order)
            || entry[12] != 0
            || le16(entry, 26) != 0
        {
            self.clear();
            return;
        }
        if flags & LAST_ENTRY != 0 {
            self.total = order;
            self.remaining = order;
            self.checksum = entry[13];
        }
        if order != self.remaining || entry[13] != self.checksum {
            self.clear();
            return;
        }
        let start = (order - 1) * WORDS_PER_ENTRY;
        for (index, offset) in WORD_OFFSETS.into_iter().enumerate() {
            self.words[start + index] = le16(entry, offset);
        }
        self.remaining -= 1;
    }

    pub(super) fn matches(&self, name: &str, short_name: &[u8]) -> bool {
        if self.total == 0 || self.remaining != 0 || self.checksum != checksum(short_name) {
            return false;
        }
        let words = &self.words[..self.total * WORDS_PER_ENTRY];
        let length = words
            .iter()
            .position(|&word| word == 0)
            .unwrap_or(words.len());
        if length == 0
            || length > NAME_LIMIT
            || words[..length].contains(&0xffff)
            || length < words.len() && words[length + 1..].iter().any(|&word| word != 0xffff)
        {
            return false;
        }
        fn fold(word: u16) -> u16 {
            match u8::try_from(word) {
                Ok(byte) => u16::from(byte.to_ascii_uppercase()),
                Err(_) => word,
            }
        }
        words[..length]
            .iter()
            .copied()
            .map(fold)
            .eq(name.encode_utf16().map(fold))
    }
}
