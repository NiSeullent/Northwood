//! The Apple AES engine of S8000 layout (`aes,s8000`; ADT `/arm-io/aes`), the
//! DMA engine behind XNU's IOAESAccelerator service.
//!
//! Documented by the qemu-t8030 `apple_aes` model (register and command
//! layouts) and checked against the macOS 26 J473 AppleS8000AESAccelerator
//! driver, which reads `aes-version` from the ADT. The first `reg` range
//! holds the block registers, 32-bit:
//! - CONTROL (0x8): START 0 and STOP 1 start and stop the command FIFO.
//! - STATUS (0xc): COMMAND_FIFO_ENABLED 6; from version 2 the driver waits
//!   for bits 7 and 8 (TEXT and KEY_UNWRAP DPA random seeded), from version 3
//!   for bits 7 to 13, after it enables the engine.
//! - INT_STATUS (0x18, write one to clear) and INT_ENABLE (0x1c):
//!   COMMAND_FIFO_LOW 0, COMMAND_FIFO_OVERFLOW 1, INVALID_COMMAND 2, AXI read
//!   and write response errors 3 and 4, FLAG_COMMAND 5, INVALID_DATA_LENGTH
//!   6, and per key context (bits 7 and 13 up) KEY_DISABLED +0 and
//!   KEY_INVALID_AT_USE_TIME +3. The interrupt is any enabled status bit.
//! - WATERMARKS (0x20): COMMAND_FIFO_LOW [6:0], in words.
//! - COMMAND_FIFO_STATUS (0x24): LOW 0 (level below the watermark), EMPTY 1,
//!   FULL 2, OVERFLOW 3, LEVEL [15:8] in words. The driver counts free space
//!   as 128 words less LEVEL.
//! - FLAG_COMMAND (0x30): the code [7:0] of the last FLAG command; the driver
//!   reads it again while [15:8] is nonzero.
//! - CLEAR_FIFO (0x38): bit 0 empties the command FIFO (versions 0 and 1).
//! - COMMAND_FIFO (0x200, 0x100 in version 0): each word written joins the
//!   FIFO.
//!
//! The second range holds the key-disable words. The word at
//! `sioaesdisable-offset` (default 0) has a sticky bit per hardware key:
//! UID1 0, GID0 1, GID1 2 and a fourth key 3 (IOAESAccelerator handle 3000),
//! each set by writing it as one. A key whose ADT names its own word
//! (`sioaesdisableuid1-offset`, `-gid0-`, `-gid1-`; 0x40, 0x44, 0x48 on M3
//! and M4, 4, 8, 0xc on M5 and M6) is enabled while that word reads
//! 0xa050c030, and writing 0xa55ac33c disables it (`OverrideRegMap`,
//! `_isKeyDisabled`, `disableHardwareKey`). The layout follows the ADT, not
//! `aes-version`: version 3 M3 boards have the separate words too.
//!
//! The 26A434 driver is one binary for every board (M2 J473, M3 J613, M4
//! J773g, M5 J813 and M6 J873g, identical code apart from relocation and
//! the M6 build's lack of pointer authentication). Its only `aes-version`
//! branches: the FIFO at 0x100 in version 0; CLEAR_FIFO before version 2;
//! the seeded STATUS bits and interrupt masks from version 2; a third
//! `reg` range, the DKey range, from version 5 (`start`); its distribution
//! register at 4 in version 5 and 8 from version 6 (`_distribute_dkey`);
//! and the H16 scrambler option from version 5 (`getSupportedOptions`).
//! Distribution writes one and waits up to 5 s for bit 0 to clear; it
//! precedes command 13 (H16 scrambled key) for AppleAVD, whose payload
//! seed the driver first writes to the block's 0x50-0x5c. The ADT
//! `address-width` (default 36) sets the high address fields: 10 bits at
//! 42 (every M3 to M6 board, whose `dart-sio` window starts at 2^40),
//! 8 bits otherwise.
//!
//! Commands are an opcode [31:28] and its operands, in FIFO words:
//! - KEY (1): context 27, select [26:24] (0 software, 1 UID1, 2 and 3 GID0
//!   and GID1, 4 to 7 HDCP), length [23:22] (128, 192, 256 bits), WRAPPED
//!   21, ENCRYPT 20, function [19:18] (0 none), mode [17:16] (ECB, CBC,
//!   CTR), then a software key's words.
//! - IV (2): context [27:26], then four words.
//! - DATA (5): key context 27, IV context [26:25], length [23:0] in bytes,
//!   then the source's high address bits from bit 16 and the destination's
//!   from bit 0 (8 or 10 bits each), the source's and the destination's
//!   low 32 bits: I/O addresses through the engine's IOMMU stream.
//! - STORE_IV (6): context [27:26], destination high bits from bit 0, then
//!   its low 32 bits: writes the context's chaining value.
//! - FLAG (8): code [7:0], STOP 26, INTERRUPT 27.
//!
//! Inferred, not documented: the engine has no DPA countermeasure to seed,
//! so the seeded STATUS bits read set from reset; commands run as soon as
//! the FIFO is started and holds a whole one, so the FIFO drains before the
//! guest's next access; CTR increments its 128-bit counter big-endian; a
//! hardware key the key-disable register disables raises KEY_DISABLED and
//! leaves its context without a key, and DATA with no key raises
//! KEY_INVALID_AT_USE_TIME, transferring nothing; a FLAG with STOP stops the
//! FIFO like CONTROL STOP; a key's own disable word reads zero once
//! disabled (the driver only compares it with the enabled value); the
//! per-key bit word reports a key with its own word in that key's bit, and
//! writing such a bit there is refused. Stela has no Apple GID keys, so
//! GID0, GID1 and the fourth key read disabled; UID1 is the board's own
//! per-installation key when it has one, else disabled too. VERSION,
//! CONFIG, KEY_ID, AXI_STATUS, COMMAND_FIFO_COUNT, SKG_KEY, the history
//! FIFO, the tag registers (0x50-0x6c), RESET and RESET_AES, wrapped keys,
//! key functions, HDCP keys, the DSB, SKG, WRITE_REG and H16 scrambled-key
//! commands, DKey distribution, DATA bit 24 and the reserved bits of the
//! address words are refused until their values or behavior are known.
use crate::Result;
use alloc::collections::VecDeque;
#[cfg(not(feature = "std"))]
use alloc::vec;
use stela_contracts::Refusal;
use stela_contracts::refuse;
use stela_crypto::cipher::{BlockDecrypt, BlockEncrypt, KeyInit, generic_array::GenericArray};
use stela_soc_data::aes::DEFAULT_ADDRESS_WIDTH;
#[cfg(test)]
use stela_soc_data::aes::DKEY_DISTRIBUTE_V5;
#[cfg(test)]
use stela_soc_data::aes::DKEY_DISTRIBUTE_V6;
use {
    stela_contracts::bus::Access, stela_contracts::bus::Direction, stela_contracts::bus::Lifecycle,
    stela_contracts::bus::LifecycleState, stela_contracts::bus::MmioDevice,
    stela_contracts::bus::MmioError, stela_contracts::bus::Width, stela_contracts::bus::refuse,
};
use {stela_contracts::dma::DmaError, stela_contracts::dma::Port};
use {stela_contracts::interrupt::AicInput, stela_contracts::interrupt::Wire};

const NAME: &str = "aes-s8000";

/// Bytes of the block register range; the key-disable range follows it,
/// then the DKey range from version 5.
pub const BLOCK_BYTES: u64 = 0x4000;
const SOURCE_HIGH_SHIFT: u32 = 16;
const CONTROL: u64 = 0x8;
const STATUS: u64 = 0xc;
const INT_STATUS: u64 = 0x18;
const INT_ENABLE: u64 = 0x1c;
const WATERMARKS: u64 = 0x20;
const FIFO_STATUS: u64 = 0x24;
const FLAG_COMMAND: u64 = 0x30;
const CLEAR_FIFO: u64 = 0x38;

const CONTROL_START: u32 = 1;
const CONTROL_STOP: u32 = 1 << 1;
const STATUS_FIFO_ENABLED: u32 = 1 << 6;
const STATUS_SEEDED: u32 = 0x3f80;

const INT_FIFO_LOW: u32 = 1;
const INT_FIFO_OVERFLOW: u32 = 1 << 1;
const INT_INVALID_COMMAND: u32 = 1 << 2;
const INT_AXI_READ: u32 = 1 << 3;
const INT_AXI_WRITE: u32 = 1 << 4;
const INT_FLAG: u32 = 1 << 5;
const INT_INVALID_LENGTH: u32 = 1 << 6;
const INT_KEY_DISABLED: [u32; 2] = [1 << 7, 1 << 13];
const INT_KEY_INVALID_AT_USE: [u32; 2] = [1 << 10, 1 << 16];
const INT_KNOWN: u32 = 0x7_ffff;

const FIFO_LOW: u32 = 1;
const FIFO_EMPTY: u32 = 1 << 1;
const FIFO_FULL: u32 = 1 << 2;
const FIFO_OVERFLOW: u32 = 1 << 3;
const FIFO_LEVEL_SHIFT: u32 = 8;
const FIFO_WORDS: usize = 128;
const WATERMARK: u32 = 0x7f;

/// Key-disable bits: UID1, GID0, GID1 and the fourth hardware key.
const DISABLE_UID1: u32 = 1;
const DISABLE_KNOWN: u32 = 0xf;
const DISABLE_ALL_BUT_UID1: u32 = DISABLE_KNOWN & !DISABLE_UID1;
/// What a key's own disable word reads while the key is enabled.
const KEY_ENABLED: u32 = 0xa050_c030;
/// The value that disables a key through its own word.
const KEY_DISABLE: u32 = 0xa55a_c33c;
/// What a key's own disable word reads once disabled (inferred).
const KEY_DISABLED: u32 = 0;

const OPCODE_KEY: u32 = 1;
const OPCODE_IV: u32 = 2;
const OPCODE_DSB: u32 = 3;
const OPCODE_SKG: u32 = 4;
const OPCODE_DATA: u32 = 5;
const OPCODE_STORE_IV: u32 = 6;
const OPCODE_WRITE_REG: u32 = 7;
const OPCODE_FLAG: u32 = 8;
const OPCODE_SCRAMBLED_KEY: u32 = 0xd;

const OPCODE_SHIFT: u32 = 28;
const OPCODE_FIELD: u32 = 0xf << OPCODE_SHIFT;
const KEY_WRAPPED: u32 = 1 << 21;
const KEY_ENCRYPT: u32 = 1 << 20;
const FLAG_STOP: u32 = 1 << 26;
const FLAG_INTERRUPT: u32 = 1 << 27;
const DATA_LENGTH: u32 = 0xff_ffff;
/// DATA bit 24, which the driver never sets.
const DATA_RESERVED: u32 = 1 << 24;
const STORE_IV_CONTEXT_SHIFT: u32 = 26;
const CONTEXT_MASK: u32 = 3;

/// Operand words of the longest command, a KEY with a 256-bit key.
const MAX_OPERANDS: usize = 8;
const BLOCK: usize = 16;
/// Bytes a DATA command moves per DMA read and write.
const CHUNK: usize = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Ecb,
    Cbc,
    Ctr,
}

enum Cipher {
    Aes128(stela_crypto::Aes128),
    Aes192(stela_crypto::Aes192),
    Aes256(stela_crypto::Aes256),
}

impl Cipher {
    fn new(key: &[u8]) -> Option<Self> {
        Some(match key.len() {
            16 => Self::Aes128(stela_crypto::Aes128::new_from_slice(key).ok()?),
            24 => Self::Aes192(stela_crypto::Aes192::new_from_slice(key).ok()?),
            32 => Self::Aes256(stela_crypto::Aes256::new_from_slice(key).ok()?),
            _ => return None,
        })
    }

    fn encrypt(&self, block: &mut [u8; BLOCK]) {
        let block = GenericArray::from_mut_slice(block);
        match self {
            Self::Aes128(cipher) => cipher.encrypt_block(block),
            Self::Aes192(cipher) => cipher.encrypt_block(block),
            Self::Aes256(cipher) => cipher.encrypt_block(block),
        }
    }

    fn decrypt(&self, block: &mut [u8; BLOCK]) {
        let block = GenericArray::from_mut_slice(block);
        match self {
            Self::Aes128(cipher) => cipher.decrypt_block(block),
            Self::Aes192(cipher) => cipher.decrypt_block(block),
            Self::Aes256(cipher) => cipher.decrypt_block(block),
        }
    }
}

struct Key {
    cipher: Cipher,
    mode: Mode,
    encrypt: bool,
}

impl Key {
    /// Transform `data` in place, whole blocks, chaining through `iv`.
    fn apply(&self, iv: &mut [u8; BLOCK], data: &mut [u8]) {
        for block in data.as_chunks_mut::<BLOCK>().0 {
            match (self.mode, self.encrypt) {
                (Mode::Ecb, true) => self.cipher.encrypt(block),
                (Mode::Ecb, false) => self.cipher.decrypt(block),
                (Mode::Cbc, true) => {
                    xor(block, iv);
                    self.cipher.encrypt(block);
                    *iv = *block;
                }
                (Mode::Cbc, false) => {
                    let input = *block;
                    self.cipher.decrypt(block);
                    xor(block, iv);
                    *iv = input;
                }
                (Mode::Ctr, _) => {
                    let mut stream = *iv;
                    self.cipher.encrypt(&mut stream);
                    xor(block, &stream);
                    *iv = (u128::from_be_bytes(*iv).wrapping_add(1)).to_be_bytes();
                }
            }
        }
    }
}

fn xor(block: &mut [u8; BLOCK], other: &[u8; BLOCK]) {
    for (byte, other) in block.iter_mut().zip(other) {
        *byte ^= other;
    }
}

use stela_soc_data::aes::Config;
#[cfg(test)]
use stela_soc_data::aes::KeyDisable;

/// Where a device offset lands.
enum Space {
    Block(u64),
    KeyDisable(u64),
    DKey(u64),
}

pub struct Aes {
    lifecycle: LifecycleState,
    config: Config,
    irq: Option<AicInput>,
    irq_level: bool,
    uid: Option<[u8; 32]>,
    disabled: u32,
    started: bool,
    /// Unsupported DMA may already have committed RAM and chaining state.
    /// Keep the cause until a checked recovery can authorize execution.
    dma_failure: Option<Refusal>,
    fifo: VecDeque<u32>,
    overflow: bool,
    watermark: u32,
    int_status: u32,
    int_enable: u32,
    flag: u32,
    keys: [Option<Key>; 2],
    ivs: [[u8; BLOCK]; 4],
}

impl Aes {
    /// An engine as the ADT describes it in `config`, interrupting on
    /// `irq`, whose UID1 key is `uid` (disabled without one).
    pub fn new(config: Config, irq: Option<AicInput>, uid: Option<[u8; 32]>) -> Result<Self> {
        config.validate()?;
        let disabled = match uid {
            Some(_) => DISABLE_ALL_BUT_UID1,
            None => DISABLE_KNOWN,
        };
        Ok(Self {
            lifecycle: LifecycleState {
                powered: true,
                clocked: true,
                quiesced: false,
                reset: false,
                retention: false,
            },
            config,
            irq,
            irq_level: false,
            uid,
            disabled,
            started: false,
            dma_failure: None,
            fifo: VecDeque::with_capacity(FIFO_WORDS),
            overflow: false,
            watermark: 0,
            int_status: 0,
            int_enable: 0,
            flag: 0,
            keys: [None, None],
            ivs: [[0; BLOCK]; 4],
        })
    }

    fn space(&self, offset: u64) -> Space {
        let disable = BLOCK_BYTES + self.config.key_disable.bytes;
        if offset < BLOCK_BYTES {
            Space::Block(offset)
        } else if offset < disable {
            Space::KeyDisable(offset - BLOCK_BYTES)
        } else {
            Space::DKey(offset - disable)
        }
    }

    /// The key-disable bit of the key whose own word is at `offset`.
    fn own_word(&self, offset: u64) -> Option<u32> {
        let keys = self.config.key_disable.keys;
        let index = keys
            .iter()
            .position(|&word| word.map(u64::from) == Some(offset))?;
        Some(1 << index)
    }

    /// The key-disable bits of the keys that have their own words.
    fn own_words(&self) -> u32 {
        let keys = self.config.key_disable.keys.iter();
        keys.enumerate()
            .filter(|(_, word)| word.is_some())
            .fold(0, |bits, (index, _)| bits | 1 << index)
    }

    fn read_key_disable(&self, offset: u64) -> Option<u32> {
        if offset == u64::from(self.config.key_disable.word) {
            return Some(self.disabled);
        }
        let bit = self.own_word(offset)?;
        Some(if self.disabled & bit != 0 {
            KEY_DISABLED
        } else {
            KEY_ENABLED
        })
    }

    fn write_key_disable(&mut self, offset: u64, value: u32) -> Result<()> {
        if offset == u64::from(self.config.key_disable.word) {
            if value & !DISABLE_KNOWN != 0 {
                return Err(refuse!("AES key-disable reserved bits"));
            }
            if value & self.own_words() != 0 {
                return Err(refuse!(
                    "AES key-disable bit of a key with its own disable word"
                ));
            }
            self.disabled |= value;
            return Ok(());
        }
        let bit = self
            .own_word(offset)
            .ok_or(refuse!("unknown AES key-disable register"))?;
        if value != KEY_DISABLE {
            return Err(refuse!(
                "AES key-disable word value other than the disable value"
            ));
        }
        self.disabled |= bit;
        Ok(())
    }

    /// Why an access at `offset` in the DKey range is refused.
    fn dkey_refusal(&self, offset: u64) -> &'static str {
        if !self.config.has_dkey() {
            "unknown AES register"
        } else if offset == self.config.dkey_distribute() {
            "AES DKey distribution to AppleAVD"
        } else {
            "unknown AES DKey register"
        }
    }

    fn executing(&self) -> bool {
        self.accessible().is_ok() && !self.lifecycle.quiesced
    }

    fn loses_power(&self, proposed: LifecycleState) -> bool {
        self.lifecycle.powered && !proposed.powered && !proposed.retention
    }

    fn accessible(&self) -> Result<()> {
        if !self.lifecycle.powered {
            return Err(refuse!("AES power domain is off"));
        }
        if !self.lifecycle.clocked {
            return Err(refuse!("AES clock is gated"));
        }
        if self.lifecycle.reset {
            return Err(refuse!("AES physical reset is asserted"));
        }
        Ok(())
    }

    fn register_accessible(&self, space: &Space) -> Result<()> {
        // J473's v3 key-disable word is in the separate SIO aperture.
        // 26B123 writes it after PMGR has gated the command engine. The
        // word only adds sticky restrictions and grants no command access.
        if matches!(space, Space::KeyDisable(_))
            && self.config.version == 3
            && self.config.address_width == DEFAULT_ADDRESS_WIDTH
            && self.config.key_disable.word == 0
            && self.config.key_disable.keys == [None; 3]
            && !self.lifecycle.reset
        {
            return Ok(());
        }
        self.accessible()
    }

    fn command_fifo(&self) -> u64 {
        if self.config.version == 0 {
            0x100
        } else {
            0x200
        }
    }

    fn low(&self) -> bool {
        (self.fifo.len() as u32) < self.watermark
    }

    fn status(&self) -> u32 {
        self.int_status | if self.low() { INT_FIFO_LOW } else { 0 }
    }

    fn fifo_status(&self) -> u32 {
        let level = self.fifo.len();
        let mut status = (level as u32) << FIFO_LEVEL_SHIFT;
        for (set, bit) in [
            (self.low(), FIFO_LOW),
            (level == 0, FIFO_EMPTY),
            (level >= FIFO_WORDS, FIFO_FULL),
            (self.overflow, FIFO_OVERFLOW),
        ] {
            if set {
                status |= bit;
            }
        }
        status
    }

    fn signal(&mut self, access: &mut Access<'_>) {
        let level = self.status() & self.int_enable != 0;
        if level != self.irq_level {
            self.irq_level = level;
            if let Some(irq) = self.irq {
                access.interrupts.set_level(Wire::Aic(irq), level);
            }
        }
    }

    /// Whether commands wait for [`Self::run`].
    pub fn pending(&self) -> bool {
        self.dma_failure.is_none() && self.executing() && self.started && !self.fifo.is_empty()
    }

    /// Run every whole command at the head of the started FIFO, moving data
    /// through `port`. A command the model does not implement is refused
    /// and stays queued. An unsupported DMA transfer keeps its cause and
    /// command without replay: accepted prefixes and chaining state cannot be
    /// undone or restarted by FIFO submission, START or CLEAR_FIFO.
    pub fn run(&mut self, port: &mut Port<'_, '_>) -> Result<(), MmioError> {
        if !self.executing() {
            return Err(refuse(
                NAME,
                self.command_fifo(),
                Width::Word,
                Direction::Write,
                "AES DMA while PMGR inhibits execution",
            ));
        }
        if let Some(reason) = self.dma_failure.clone() {
            return Err(refuse(
                NAME,
                self.command_fifo(),
                Width::Word,
                Direction::Write,
                reason,
            ));
        }
        while self.started {
            let Some(&command) = self.fifo.front() else {
                break;
            };
            let words = self.words(command)?;
            if self.fifo.len() < words {
                break;
            }
            let mut operands = [0; MAX_OPERANDS];
            for (slot, &word) in operands.iter_mut().zip(self.fifo.iter().skip(1)) {
                *slot = word;
            }
            self.execute(command, &operands[..words - 1], port)?;
            self.fifo.drain(..words);
        }
        self.signal(port.access);
        Ok(())
    }

    /// FIFO words of the command `command` begins, itself included.
    fn words(&self, command: u32) -> Result<usize, MmioError> {
        let fail = |reason| {
            refuse(
                NAME,
                self.command_fifo(),
                Width::Word,
                Direction::Write,
                reason,
            )
        };
        Ok(match command >> OPCODE_SHIFT {
            OPCODE_KEY if (command >> 24) & 7 == 0 => 1 + key_bytes(command) / 4,
            OPCODE_KEY => 1,
            OPCODE_IV => 5,
            OPCODE_DATA => 4,
            OPCODE_STORE_IV => 2,
            OPCODE_FLAG => 1,
            OPCODE_DSB => return Err(fail("AES DSB command")),
            OPCODE_SKG => return Err(fail("AES SKG command")),
            OPCODE_WRITE_REG => return Err(fail("AES WRITE_REG command")),
            OPCODE_SCRAMBLED_KEY => return Err(fail("AES H16 scrambled-key command")),
            _ => 1,
        })
    }

    fn execute(
        &mut self,
        command: u32,
        operands: &[u32],
        port: &mut Port<'_, '_>,
    ) -> Result<(), MmioError> {
        let fail = |reason| {
            refuse(
                NAME,
                self.command_fifo(),
                Width::Word,
                Direction::Write,
                reason,
            )
        };
        match command >> OPCODE_SHIFT {
            OPCODE_KEY => {
                if command & KEY_WRAPPED != 0 {
                    return Err(fail("AES wrapped keys"));
                }
                if (command >> 18) & 3 != 0 {
                    return Err(fail("AES key functions"));
                }
                let context = (command >> 27 & 1) as usize;
                let mode = match (command >> 16) & 3 {
                    0 => Mode::Ecb,
                    1 => Mode::Cbc,
                    2 => Mode::Ctr,
                    _ => {
                        self.keys[context] = None;
                        self.int_status |= INT_INVALID_COMMAND;
                        return Ok(());
                    }
                };
                let length = key_bytes(command);
                let mut material = [0; 32];
                match (command >> 24) & 7 {
                    0 => {
                        for (bytes, word) in
                            material.as_chunks_mut::<4>().0.iter_mut().zip(operands)
                        {
                            bytes.copy_from_slice(&word.to_le_bytes());
                        }
                    }
                    select @ 1..=3 => {
                        let bit = 1 << (select - 1);
                        match self.uid.filter(|_| self.disabled & bit == 0) {
                            Some(uid) => material = uid,
                            None => {
                                self.keys[context] = None;
                                self.int_status |= INT_KEY_DISABLED[context];
                                return Ok(());
                            }
                        }
                    }
                    _ => return Err(fail("AES HDCP keys")),
                }
                self.keys[context] = match Cipher::new(&material[..length]) {
                    Some(cipher) => Some(Key {
                        cipher,
                        mode,
                        encrypt: command & KEY_ENCRYPT != 0,
                    }),
                    None => {
                        self.int_status |= INT_INVALID_COMMAND;
                        None
                    }
                };
            }
            OPCODE_IV => {
                let context = (command >> 26 & 3) as usize;
                for (bytes, word) in self.ivs[context]
                    .as_chunks_mut::<4>()
                    .0
                    .iter_mut()
                    .zip(operands)
                {
                    bytes.copy_from_slice(&word.to_le_bytes());
                }
            }
            OPCODE_DATA => self.data(command, operands, port)?,
            OPCODE_STORE_IV => {
                let high = self.config.high_mask();
                let fields = OPCODE_FIELD | CONTEXT_MASK << STORE_IV_CONTEXT_SHIFT | high;
                if command & !fields != 0 {
                    return Err(fail("AES STORE_IV reserved bits"));
                }
                let context = (command >> STORE_IV_CONTEXT_SHIFT & CONTEXT_MASK) as usize;
                let destination = u64::from(command & high) << 32 | u64::from(operands[0]);
                let iv = self.ivs[context];
                if let Err(error) = port.write(destination, &iv) {
                    self.dma_failed(error, INT_AXI_WRITE)?;
                }
            }
            OPCODE_FLAG => {
                self.flag = command & 0xff;
                if command & FLAG_INTERRUPT != 0 {
                    self.int_status |= INT_FLAG;
                }
                if command & FLAG_STOP != 0 {
                    self.started = false;
                }
            }
            _ => self.int_status |= INT_INVALID_COMMAND,
        }
        Ok(())
    }

    fn data(
        &mut self,
        command: u32,
        operands: &[u32],
        port: &mut Port<'_, '_>,
    ) -> Result<(), MmioError> {
        let fail = |reason| {
            refuse(
                NAME,
                self.command_fifo(),
                Width::Word,
                Direction::Write,
                reason,
            )
        };
        let length = (command & DATA_LENGTH) as usize;
        let context = (command >> 27 & 1) as usize;
        let iv_context = (command >> 25 & CONTEXT_MASK) as usize;
        let [upper, source, destination] = [operands[0], operands[1], operands[2]];
        let high = self.config.high_mask();
        if command & DATA_RESERVED != 0 {
            return Err(fail("AES DATA bit 24"));
        }
        if upper & !(high << SOURCE_HIGH_SHIFT | high) != 0 {
            return Err(fail("AES DATA address word reserved bits"));
        }
        let source = u64::from(upper >> SOURCE_HIGH_SHIFT & high) << 32 | u64::from(source);
        let destination = u64::from(upper & high) << 32 | u64::from(destination);
        if !length.is_multiple_of(BLOCK) {
            self.int_status |= INT_INVALID_LENGTH;
            return Ok(());
        }
        let Some(key) = &self.keys[context] else {
            self.int_status |= INT_KEY_INVALID_AT_USE[context];
            return Ok(());
        };
        let mut iv = self.ivs[iv_context];
        let mut buffer = vec![0; length.min(CHUNK)];
        let mut done = 0;
        let mut failed = None;
        while done < length {
            let chunk = &mut buffer[..(length - done).min(CHUNK)];
            if let Err(error) = port.read(source + done as u64, chunk) {
                failed = Some((error, INT_AXI_READ));
                break;
            }
            key.apply(&mut iv, chunk);
            if let Err(error) = port.write(destination + done as u64, chunk) {
                failed = Some((error, INT_AXI_WRITE));
                break;
            }
            done += chunk.len();
        }
        self.ivs[iv_context] = iv;
        match failed {
            Some((error, interrupt)) => self.dma_failed(error, interrupt),
            None => Ok(()),
        }
    }

    /// An AXI error interrupt for a DMA the IOMMU faulted or that missed
    /// RAM; a stream configuration the model lacks is refused.
    fn dma_failed(&mut self, error: DmaError, interrupt: u32) -> Result<(), MmioError> {
        match error {
            DmaError::Fault(_) | DmaError::Memory => {
                self.int_status |= interrupt;
                Ok(())
            }
            DmaError::Unsupported(reason) => {
                self.dma_failure = Some(reason.clone());
                Err(refuse(
                    NAME,
                    self.command_fifo(),
                    Width::Word,
                    Direction::Write,
                    reason,
                ))
            }
        }
    }
}

/// Bytes of the key a KEY command's length field selects.
fn key_bytes(command: u32) -> usize {
    match (command >> 22) & 3 {
        0 => 16,
        1 => 24,
        _ => 32,
    }
}

impl Lifecycle for Aes {
    fn check_lifecycle(&self, proposed: LifecycleState) -> Result<()> {
        if !self.lifecycle.reset && proposed.reset {
            return Err(refuse!(
                "AES physical reset requires qualified key-disable, context and entropy restore semantics"
            ));
        }
        if self.loses_power(proposed) && (!self.fifo.is_empty() || self.dma_failure.is_some()) {
            return Err(refuse!(
                "AES nonretained power loss with outstanding commands or failed DMA"
            ));
        }
        Ok(())
    }

    fn set_lifecycle(&mut self, proposed: LifecycleState, access: &mut Access<'_>) {
        // A nonretained supply loss on an idle engine resets ordinary
        // state. Hardware key restrictions stay sticky. Retained gates
        // preserve commands and interrupts until the engine can run again.
        if self.loses_power(proposed) {
            debug_assert!(self.fifo.is_empty() && self.dma_failure.is_none());
            self.started = false;
            self.overflow = false;
            self.watermark = 0;
            self.int_status = 0;
            self.int_enable = 0;
            self.flag = 0;
            self.keys = [None, None];
            self.ivs = [[0; BLOCK]; 4];
        }
        self.lifecycle = proposed;
        self.signal(access);
    }
}

impl MmioDevice for Aes {
    fn lifecycle(&mut self) -> Option<&mut dyn Lifecycle> {
        Some(self)
    }
    fn name(&self) -> &'static str {
        NAME
    }

    fn read(
        &mut self,
        offset: u64,
        width: Width,
        _access: &mut Access<'_>,
    ) -> Result<u64, MmioError> {
        let fail = |reason: Refusal| refuse(NAME, offset, width, Direction::Read, reason);
        let space = self.space(offset);
        self.register_accessible(&space).map_err(fail)?;
        if width != Width::Word {
            return Err(fail(refuse!("AES registers are 32-bit")));
        }
        let offset = match space {
            Space::Block(offset) => offset,
            Space::KeyDisable(at) => {
                let value = self.read_key_disable(at);
                return value
                    .map(u64::from)
                    .ok_or_else(|| fail(refuse!("unknown AES key-disable register")));
            }
            Space::DKey(at) => return Err(fail(self.dkey_refusal(at).into())),
        };
        let value = match offset {
            CONTROL => 0,
            STATUS => STATUS_SEEDED | if self.started { STATUS_FIFO_ENABLED } else { 0 },
            INT_STATUS => self.status(),
            INT_ENABLE => self.int_enable,
            WATERMARKS => self.watermark,
            FIFO_STATUS => self.fifo_status(),
            FLAG_COMMAND => self.flag,
            _ => return Err(fail(refuse!("unknown AES register"))),
        };
        Ok(u64::from(value))
    }

    fn write(
        &mut self,
        offset: u64,
        width: Width,
        value: u64,
        access: &mut Access<'_>,
    ) -> Result<(), MmioError> {
        let fail = |reason: Refusal| refuse(NAME, offset, width, Direction::Write, reason);
        let space = self.space(offset);
        self.register_accessible(&space).map_err(fail)?;
        if self.lifecycle.quiesced
            && (offset == self.command_fifo()
                || offset == CONTROL && value == u64::from(CONTROL_START))
        {
            return Err(fail(refuse!("AES command submission while PMGR quiesced")));
        }
        if width != Width::Word {
            return Err(fail(refuse!("AES registers are 32-bit")));
        }
        let value = value as u32;
        let offset = match space {
            Space::Block(offset) => offset,
            Space::KeyDisable(at) => return self.write_key_disable(at, value).map_err(fail),
            Space::DKey(at) => return Err(fail(self.dkey_refusal(at).into())),
        };
        match offset {
            CONTROL => match value {
                CONTROL_START => self.started = true,
                CONTROL_STOP => self.started = false,
                _ => return Err(fail(refuse!("AES CONTROL other than START or STOP"))),
            },
            INT_STATUS => self.int_status &= !value,
            INT_ENABLE if value & !INT_KNOWN == 0 => self.int_enable = value,
            INT_ENABLE => return Err(fail(refuse!("AES INT_ENABLE reserved bits"))),
            WATERMARKS => self.watermark = value & WATERMARK,
            CLEAR_FIFO => {
                if value & 1 != 0 {
                    self.fifo.clear();
                    self.overflow = false;
                }
            }
            _ if offset == self.command_fifo() => {
                if let Some(reason) = self.dma_failure.clone() {
                    return Err(fail(reason));
                }
                if self.fifo.len() < FIFO_WORDS {
                    self.fifo.push_back(value);
                } else {
                    self.overflow = true;
                    self.int_status |= INT_FIFO_OVERFLOW;
                }
            }
            _ => return Err(fail(refuse!("unknown AES register"))),
        }
        self.signal(access);
        Ok(())
    }
}

#[cfg(test)]
mod tests;
