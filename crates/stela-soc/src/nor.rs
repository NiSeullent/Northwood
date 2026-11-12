//! Single-wire, 24-bit SPI NOR behind `spi::Spi`.
//!
//! Contract: 26A434 `AppleARMSPIFlashController::_readJedecId` file
//! 0x1c7d364 (9f), `_readRDSR` 0x1c7d1b8 (05), `_writeEnable`
//! 0x1c7d048 (06/04), `_programBytes` 0x1c7c894 (02), `_eraseSector`
//! 0x1c7c480 (20), `_eraseBlock` 0x1c7c288 (d8), `readBuffer`
//! 0x1c9ffdc (03). `_identifyNORDevice` at 0x1c7b7a0 selects ef6017's
//! geometry through 0x1c7ba1c..0x1c7ba28 -> 0x1c7bfe4 -> 0x1c7c030;
//! the constants at 0x1759f0 are 2048 sectors, 4096 bytes, 256-byte pages
//! and 50. The former ef6514 selects only 256 sectors at 0x175a20 (1 MiB).
//! Stela reports the matching identifier for its 8 MiB die, with 256-byte
//! pages, 4 KiB sectors and 64 KiB blocks. It is not a claim about the
//! physical board's flash vendor. The 26A434 J473 `mBoot` image's table at
//! file 0x1b0110 supplies the 8 MiB capacity and partition regions below;
//! boot firmware normalizes the DeviceTree's placeholder partition `reg`s
//! and publishes that resolved geometry to the kernel.
//!
//! Commands commit on CS rising. This functional die completes program and
//! erase at that boundary (zero busy latency), so WIP is always clear;
//! status WEL follows WREN/WRDI and is consumed by program/erase/WRSR.
//! Unsupported commands refuse before their byte is accepted.
use crate::Result;
use alloc::collections::BTreeSet;
#[cfg(test)]
use alloc::vec;
use alloc::vec::Vec;
use stela_contracts::refuse;

pub mod persist;

#[cfg(test)]
use stela_soc_data::nor::ANVRAM;
use stela_soc_data::nor::BLOCK_SIZE;
use stela_soc_data::nor::CAPACITY;
use stela_soc_data::nor::ERASED;
use stela_soc_data::nor::Image;
use stela_soc_data::nor::JEDEC_ID;
use stela_soc_data::nor::PAGE_SIZE;
use stela_soc_data::nor::SECTOR_SIZE;
#[cfg(test)]
use stela_soc_data::nor::SYSCFG;
#[cfg(test)]
use stela_soc_data::nor::nvram;

const WEL: u8 = 2;
const READ: u8 = 0x03;
const FAST_READ: u8 = 0x0b;
const PROGRAM: u8 = 0x02;
const ERASE_SECTOR: u8 = 0x20;
const ERASE_BLOCK: u8 = 0xd8;
const READ_STATUS: u8 = 0x05;
const WRITE_STATUS: u8 = 0x01;
const WRITE_ENABLE: u8 = 0x06;
const WRITE_DISABLE: u8 = 0x04;
const READ_ID: u8 = 0x9f;

/// A command between CS edges. Copy, with its page buffer inline, so the
/// controller checkpoints it for each transfer without allocating.
#[derive(Clone, Copy)]
struct Transaction {
    command: u8,
    position: usize,
    address: usize,
    data: [u8; PAGE_SIZE],
    length: usize,
}

impl Transaction {
    fn data(&self) -> &[u8] {
        &self.data[..self.length]
    }

    fn push(&mut self, byte: u8) -> Result<()> {
        if self.length == PAGE_SIZE {
            return Err(refuse!("NOR program exceeds the modeled page buffer"));
        }
        self.data[self.length] = byte;
        self.length += 1;
        Ok(())
    }
}

pub(crate) struct Checkpoint(Option<Transaction>);

#[cfg(feature = "native-profile")]
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct NorProfile {
    pub(crate) reads: u64,
    pub(crate) read_bytes: u64,
    pub(crate) programs: u64,
    pub(crate) program_bytes: u64,
    pub(crate) erases: u64,
    pub(crate) erase_bytes: u64,
    pub(crate) status_bytes: u64,
}

pub struct Nor {
    image: Image,
    enabled: bool,
    selected: bool,
    transaction: Option<Transaction>,
    /// Sectors committed programs and erases changed since the host last
    /// took them.
    dirty: BTreeSet<usize>,
    #[cfg(feature = "native-profile")]
    profile: NorProfile,
}
impl Nor {
    pub fn new(image: Image) -> Self {
        Self {
            image,
            enabled: false,
            selected: false,
            transaction: None,
            dirty: BTreeSet::new(),
            #[cfg(feature = "native-profile")]
            profile: NorProfile::default(),
        }
    }
    #[cfg(feature = "native-profile")]
    pub(crate) fn take_profile(&mut self) -> NorProfile {
        core::mem::take(&mut self.profile)
    }
    /// The changed sectors, each as (offset, its current bytes).
    pub fn take_dirty(&mut self) -> Vec<(usize, Vec<u8>)> {
        core::mem::take(&mut self.dirty)
            .into_iter()
            .map(|sector| {
                let offset = sector * SECTOR_SIZE;
                (
                    offset,
                    self.image.bytes()[offset..offset + SECTOR_SIZE].to_vec(),
                )
            })
            .collect()
    }
    pub(crate) fn checkpoint(&self) -> Checkpoint {
        Checkpoint(self.transaction)
    }
    pub(crate) fn restore(&mut self, checkpoint: Checkpoint) {
        self.transaction = checkpoint.0;
    }
    pub fn image(&self) -> &Image {
        &self.image
    }
    pub fn selected(&self) -> bool {
        self.selected
    }

    pub(crate) fn check_select(&self, selected: bool) -> Result<()> {
        if !selected && let Some(t) = &self.transaction {
            let required = match t.command {
                READ | PROGRAM | ERASE_SECTOR | ERASE_BLOCK => 3,
                FAST_READ => 4,
                WRITE_STATUS => 1,
                _ => 0,
            };
            if t.position < required {
                return Err(refuse!("incomplete NOR command at CS rising"));
            }
            if t.command == PROGRAM && t.length == 0 {
                return Err(refuse!("empty NOR page program"));
            }
            if t.command == WRITE_STATUS && t.data() != [0] {
                return Err(refuse!("NOR status protection fields are not modeled"));
            }
        }
        Ok(())
    }

    /// CS active low; a rising edge terminates the command, keeping all
    /// bytes between falling/rising edges across controller transfers.
    pub fn select(&mut self, selected: bool) -> Result<()> {
        if self.selected == selected {
            return Ok(());
        }
        self.check_select(selected)?;
        if !selected && let Some(t) = self.transaction.take() {
            match t.command {
                WRITE_ENABLE => self.enabled = true,
                WRITE_DISABLE => self.enabled = false,
                #[cfg(feature = "native-profile")]
                READ | FAST_READ => {
                    let prefix = if t.command == READ { 3 } else { 4 };
                    self.profile.reads = self.profile.reads.saturating_add(1);
                    self.profile.read_bytes = self
                        .profile
                        .read_bytes
                        .saturating_add((t.position - prefix) as u64);
                }
                #[cfg(feature = "native-profile")]
                READ_STATUS => {
                    self.profile.status_bytes =
                        self.profile.status_bytes.saturating_add(t.position as u64);
                }
                PROGRAM if self.enabled => {
                    let base = (t.address % CAPACITY) & !(PAGE_SIZE - 1);
                    let page = &mut self.image.bytes_mut()[base..base + PAGE_SIZE];
                    let mut changed = false;
                    for (i, &byte) in t.data().iter().enumerate() {
                        let stored = &mut page[(t.address + i) & (PAGE_SIZE - 1)];
                        let programmed = *stored & byte;
                        changed |= programmed != *stored;
                        *stored = programmed;
                    }
                    if changed {
                        self.dirty.insert(base / SECTOR_SIZE);
                    }
                    self.enabled = false;
                    #[cfg(feature = "native-profile")]
                    {
                        self.profile.programs = self.profile.programs.saturating_add(1);
                        self.profile.program_bytes =
                            self.profile.program_bytes.saturating_add(t.length as u64);
                    }
                }
                ERASE_SECTOR | ERASE_BLOCK if self.enabled => {
                    let size = if t.command == ERASE_SECTOR {
                        SECTOR_SIZE
                    } else {
                        BLOCK_SIZE
                    };
                    let base = (t.address % CAPACITY) & !(size - 1);
                    let sectors =
                        self.image.bytes_mut()[base..base + size].chunks_exact_mut(SECTOR_SIZE);
                    for (index, sector) in sectors.enumerate() {
                        if sector.iter().any(|&byte| byte != ERASED) {
                            sector.fill(ERASED);
                            self.dirty.insert(base / SECTOR_SIZE + index);
                        }
                    }
                    self.enabled = false;
                    #[cfg(feature = "native-profile")]
                    {
                        self.profile.erases = self.profile.erases.saturating_add(1);
                        self.profile.erase_bytes =
                            self.profile.erase_bytes.saturating_add(size as u64);
                    }
                }
                WRITE_STATUS if self.enabled => self.enabled = false,
                _ => (), // JEDEC ignores mutations when WEL is clear.
            }
        }
        self.selected = selected;
        Ok(())
    }

    /// One MSB-first SPI byte; an unselected die leaves MISO pulled high.
    pub fn shift(&mut self, byte: u8) -> Result<u8> {
        if !self.selected {
            return Ok(ERASED);
        }
        let Some(t) = &mut self.transaction else {
            if !matches!(
                byte,
                READ | FAST_READ
                    | PROGRAM
                    | ERASE_SECTOR
                    | ERASE_BLOCK
                    | READ_STATUS
                    | WRITE_STATUS
                    | WRITE_ENABLE
                    | WRITE_DISABLE
                    | READ_ID
            ) {
                return Err(refuse!(
                    "unsupported NOR opcode (SFDP, quad and vendor commands are not modeled)"
                ));
            }
            self.transaction = Some(Transaction {
                command: byte,
                position: 0,
                address: 0,
                data: [0; PAGE_SIZE],
                length: 0,
            });
            return Ok(ERASED);
        };
        let result = match t.command {
            READ_ID => JEDEC_ID
                .get(t.position)
                .copied()
                .ok_or(refuse!("NOR ID has only three modeled bytes"))?,
            READ_STATUS => {
                if self.enabled {
                    WEL
                } else {
                    0
                }
            }
            READ | FAST_READ | PROGRAM | ERASE_SECTOR | ERASE_BLOCK if t.position < 3 => {
                t.address = t.address << 8 | usize::from(byte);
                ERASED
            }
            FAST_READ if t.position == 3 => ERASED,
            READ | FAST_READ => {
                let prefix = if t.command == READ { 3 } else { 4 };
                self.image.bytes()[(t.address + t.position - prefix) % CAPACITY]
            }
            PROGRAM => {
                t.push(byte)?;
                ERASED
            }
            WRITE_STATUS if t.position == 0 && byte == 0 => {
                t.push(byte)?;
                ERASED
            }
            _ => return Err(refuse!("unsupported NOR command length or status fields")),
        };
        t.position += 1;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn command(nor: &mut Nor, bytes: &[u8]) -> Vec<u8> {
        nor.select(true).unwrap();
        let result = bytes.iter().map(|&b| nor.shift(b).unwrap()).collect();
        nor.select(false).unwrap();
        result
    }
    #[test]
    fn inactive_nvram_bank_program_block_erase_wrdi_and_fast_read() {
        let mut nor = Nor::new(Image::fresh());
        let inactive = ANVRAM.0 as usize + nvram::BANK_SIZE;
        let addressed = |command, offset: usize, tail: &[u8]| {
            let mut bytes = vec![
                command,
                (offset >> 16) as u8,
                (offset >> 8) as u8,
                offset as u8,
            ];
            bytes.extend_from_slice(tail);
            bytes
        };
        command(&mut nor, &[WRITE_ENABLE]);
        command(&mut nor, &[WRITE_DISABLE]);
        command(&mut nor, &addressed(PROGRAM, inactive, &[0x42]));
        assert_eq!(nor.image.bank(1).unwrap()[0], ERASED);
        command(&mut nor, &[WRITE_ENABLE]);
        command(&mut nor, &[WRITE_STATUS, 0]);
        assert_eq!(command(&mut nor, &[READ_STATUS, 0xff])[1], 0);
        command(&mut nor, &[WRITE_ENABLE]);
        command(&mut nor, &addressed(PROGRAM, inactive, &[0x42]));
        assert_eq!(
            command(&mut nor, &addressed(FAST_READ, inactive, &[0, 0xff]))[5],
            0x42
        );
        assert_eq!(nor.image.bank(1).unwrap()[0], 0x42);
        command(&mut nor, &[WRITE_ENABLE]);
        command(&mut nor, &addressed(ERASE_BLOCK, inactive + 0x2100, &[]));
        assert!(nor.image.bank(1).unwrap().iter().all(|&b| b == ERASED));
        assert_eq!(nvram::validate(nor.image.bank(0).unwrap()), Ok(()));
    }

    #[test]
    fn jedec_status_program_erase_and_read_share_the_image() {
        let mut nor = Nor::new(Image::fresh());
        let id = command(&mut nor, &[READ_ID, 0xff, 0xff, 0xff]);
        assert_eq!(id, [0xff, 0xef, 0x60, 0x17]);
        assert_eq!(1usize << u32::from(id[3]), nor.image.bytes().len());
        command(&mut nor, &[PROGRAM, 0, 0, 0, 0x12]);
        assert_eq!(nor.image.bytes()[0], ERASED);
        command(&mut nor, &[WRITE_ENABLE]);
        assert_eq!(
            command(&mut nor, &[READ_STATUS, 0xff, 0xff]),
            [0xff, WEL, WEL]
        );
        command(&mut nor, &[PROGRAM, 0, 0, 0, 0x12, 0x34]);
        assert_eq!(
            command(&mut nor, &[READ, 0, 0, 0, 0xff, 0xff]),
            [0xff, 0xff, 0xff, 0xff, 0x12, 0x34]
        );
        assert_eq!(command(&mut nor, &[READ_STATUS, 0]), [0xff, 0]);
        command(&mut nor, &[WRITE_ENABLE]);
        command(&mut nor, &[PROGRAM, 0, 0, 0, 0xff]);
        assert_eq!(nor.image.bytes()[0], 0x12);
        command(&mut nor, &[WRITE_ENABLE]);
        command(&mut nor, &[ERASE_SECTOR, 0, 1, 0]);
        assert!(
            nor.image.bytes()[..SECTOR_SIZE]
                .iter()
                .all(|&b| b == ERASED)
        );
        #[cfg(feature = "native-profile")]
        {
            // A fast read's dummy byte does not count as payload. The
            // write attempted with WEL clear did not commit a program.
            command(&mut nor, &[FAST_READ, 0, 0, 0, 0, 0xff]);
            assert_eq!(
                nor.take_profile(),
                NorProfile {
                    reads: 2,
                    read_bytes: 3,
                    programs: 2,
                    program_bytes: 3,
                    erases: 1,
                    erase_bytes: SECTOR_SIZE as u64,
                    status_bytes: 3,
                }
            );
            assert_eq!(nor.take_profile(), NorProfile::default());
        }
    }
    #[test]
    fn committed_programs_and_erases_report_their_sectors_once() {
        let mut nor = Nor::new(Image::fresh());
        command(&mut nor, &[PROGRAM, 0, 0x10, 0, 0]);
        assert!(nor.take_dirty().is_empty());
        command(&mut nor, &[WRITE_ENABLE]);
        command(&mut nor, &[PROGRAM, 0, 0x10, 0, 0x5a]);
        let dirty = nor.take_dirty();
        assert_eq!(dirty.len(), 1);
        assert_eq!((dirty[0].0, dirty[0].1[0]), (0x1000, 0x5a));
        assert!(nor.take_dirty().is_empty());
        for address in [0x00, 0xa0] {
            command(&mut nor, &[WRITE_ENABLE]);
            command(&mut nor, &[PROGRAM, 0x01, address, 0, 0x5a]);
        }
        command(&mut nor, &[WRITE_ENABLE]);
        command(&mut nor, &[PROGRAM, 0x01, 0, 0, ERASED]);
        assert_eq!(command(&mut nor, &[READ_STATUS, 0]), [ERASED, 0]);
        for _ in 0..2 {
            command(&mut nor, &[WRITE_ENABLE]);
            command(&mut nor, &[ERASE_BLOCK, 0x01, 0, 0]);
            assert_eq!(command(&mut nor, &[READ_STATUS, 0]), [ERASED, 0]);
        }
        let erased = nor.take_dirty();
        assert_eq!(erased.len(), 2);
        assert_eq!((erased[0].0, erased[1].0), (0x10000, 0x1a000));
        assert!(
            erased
                .iter()
                .all(|(_, bytes)| bytes.iter().all(|&byte| byte == ERASED))
        );
        assert!(nor.take_dirty().is_empty());
    }

    #[test]
    fn suppressing_no_op_stores_preserves_the_reloaded_image() {
        use stela_drivers::block::RamDisk;

        let mut disk = RamDisk::new("nor", CAPACITY as u64 / 512, 512);
        let (image, _) = persist::load(&mut disk).unwrap();
        let mut nor = Nor::new(image);
        command(&mut nor, &[WRITE_ENABLE]);
        command(&mut nor, &[ERASE_BLOCK, 0x01, 0, 0]);
        assert!(nor.take_dirty().is_empty());
        let program = [PROGRAM, 0x01, 0, 255, 0x12, 0x34];
        command(&mut nor, &[WRITE_ENABLE]);
        command(&mut nor, &program);
        let writes = nor.take_dirty();
        assert_eq!(writes.len(), 1);
        for (offset, bytes) in writes {
            persist::store(&mut disk, offset, &bytes).unwrap();
        }
        command(&mut nor, &[WRITE_ENABLE]);
        command(&mut nor, &program);
        assert!(nor.take_dirty().is_empty());
        assert_eq!(command(&mut nor, &[READ_STATUS, 0]), [ERASED, 0]);
        let (loaded, fresh) = persist::load(&mut disk).unwrap();
        assert!(!fresh);
        assert_eq!(loaded.bytes(), nor.image().bytes());
        assert_eq!(loaded.bytes()[0x100ff], 0x12);
        assert_eq!(loaded.bytes()[0x10000], 0x34);
    }

    #[test]
    fn the_newest_valid_bank_is_current() {
        let image = Image::fresh();
        assert_eq!(image.current_bank(), Ok(0));
        let newer = nvram::empty_bank_of(2);
        let image = image.with_proxy(1, &newer).unwrap();
        assert_eq!(image.current_bank(), Ok(1));
        let tied = image.with_proxy(0, &newer).unwrap();
        assert_eq!(tied.current_bank(), Ok(1));
        let blank = Image::from_bytes(vec![ERASED; CAPACITY]).unwrap();
        assert!(blank.current_bank().is_err());
    }

    #[test]
    fn invalid_banks_supply_an_empty_proxy_without_rewriting_the_flash() {
        let mut image = Image::fresh()
            .with_proxy(1, &nvram::empty_bank_of(2))
            .unwrap();
        // A nonempty bank whose payload was not programmed completely.
        for bank in 0..nvram::BANK_COUNT {
            let start = ANVRAM.0 as usize + bank as usize * nvram::BANK_SIZE;
            image.bytes_mut()[start + PAGE_SIZE..start + nvram::BANK_SIZE].fill(ERASED);
        }
        image.bytes_mut()[SYSCFG[0].0 as usize + 10] = 0x42;
        let before = image.clone();
        assert!(image.current_bank().is_err());
        let (bank, proxy) = image.boot_bank().unwrap();
        assert_eq!(bank, nvram::CURRENT_BANK);
        assert_eq!(proxy, nvram::empty_bank());
        assert_eq!(image, before);
        // A valid bank always takes precedence over damaged ones.
        let valid = image.with_proxy(1, &nvram::empty_bank_of(3)).unwrap();
        assert_eq!(valid.boot_bank().unwrap(), (1, nvram::empty_bank_of(3)));
    }

    #[test]
    fn erased_flash_recovers_but_unidentified_data_and_corrupt_headers_refuse() {
        let mut image = Image::from_bytes(vec![ERASED; CAPACITY]).unwrap();
        assert_eq!(image.boot_bank().unwrap().1, nvram::empty_bank());
        image.bytes_mut()[12] = 0x42;
        assert!(image.boot_bank().is_err());
        let mut image = Image::fresh();
        image.bytes_mut()[ANVRAM.0 as usize + 1] ^= 1;
        assert!(image.boot_bank().is_err());
    }

    #[test]
    fn a_guest_save_after_recovery_replaces_only_its_selected_bank() {
        let mut image = Image::fresh();
        image.bytes_mut()[ANVRAM.0 as usize + 100] ^= 1;
        let (_, proxy) = image.boot_bank().unwrap();
        let mut variables = nvram::Variables::parse(&proxy).unwrap();
        variables
            .set(nvram::Partition::System, b"boot-command", b"recover")
            .unwrap();
        let saved = variables.encode().unwrap();
        let untouched = image.bank(0).unwrap().to_vec();
        let mut nor = Nor::new(image);
        let bank = ANVRAM.0 as usize + nvram::BANK_SIZE;
        command(&mut nor, &[WRITE_ENABLE]);
        command(
            &mut nor,
            &[
                ERASE_BLOCK,
                (bank >> 16) as u8,
                (bank >> 8) as u8,
                bank as u8,
            ],
        );
        for (page, bytes) in saved.chunks(PAGE_SIZE).enumerate() {
            let address = ANVRAM.0 as usize + nvram::BANK_SIZE + page * PAGE_SIZE;
            let mut program = vec![
                PROGRAM,
                (address >> 16) as u8,
                (address >> 8) as u8,
                address as u8,
            ];
            program.extend_from_slice(bytes);
            command(&mut nor, &[WRITE_ENABLE]);
            command(&mut nor, &program);
        }
        assert_eq!(nor.image.current_bank(), Ok(1));
        assert_eq!(nor.image.boot_bank().unwrap(), (1, saved));
        assert_eq!(nor.image.bank(0).unwrap(), untouched);
    }

    #[test]
    fn nvram_image_has_one_valid_bank_and_other_regions_are_erased() {
        let image = Image::fresh();
        assert_eq!(nvram::validate(image.bank(0).unwrap()), Ok(()));
        assert!(image.bank(1).unwrap().iter().all(|&b| b == ERASED));
        assert!(
            image.bytes()[..ANVRAM.0 as usize]
                .iter()
                .all(|&b| b == ERASED)
        );
        assert!(
            image.bytes()[ANVRAM.0 as usize + nvram::BANK_SIZE..]
                .iter()
                .all(|&b| b == ERASED)
        );
    }
    #[test]
    fn page_wrap_and_refusal_preserve_flash_until_cs_rises() {
        let mut nor = Nor::new(Image::fresh());
        command(&mut nor, &[WRITE_ENABLE]);
        nor.select(true).unwrap();
        for b in [PROGRAM, 0, 0, 255, 0x12, 0x34] {
            nor.shift(b).unwrap();
        }
        assert_eq!(nor.image.bytes()[255], ERASED);
        nor.select(false).unwrap();
        assert_eq!(nor.image.bytes()[255], 0x12);
        assert_eq!(nor.image.bytes()[0], 0x34);
        nor.select(true).unwrap();
        assert!(nor.shift(0x5a).is_err());
        assert_eq!(nor.shift(READ_ID), Ok(ERASED));
    }
    #[test]
    fn a_restored_checkpoint_drops_the_program_bytes_after_it() {
        let mut nor = Nor::new(Image::fresh());
        command(&mut nor, &[WRITE_ENABLE]);
        nor.select(true).unwrap();
        for b in [PROGRAM, 0, 0, 0, 0x12] {
            nor.shift(b).unwrap();
        }
        let checkpoint = nor.checkpoint();
        nor.shift(0x34).unwrap();
        nor.restore(checkpoint);
        nor.shift(0x56).unwrap();
        nor.select(false).unwrap();
        assert_eq!(nor.image.bytes()[..3], [0x12, 0x56, ERASED]);
        // A full page refuses its next byte.
        command(&mut nor, &[WRITE_ENABLE]);
        nor.select(true).unwrap();
        for b in [PROGRAM, 0, 1, 0].into_iter().chain([0; PAGE_SIZE]) {
            nor.shift(b).unwrap();
        }
        assert!(nor.shift(0).is_err());
    }
}
