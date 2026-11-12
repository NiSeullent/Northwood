use super::*;
use stela_contracts::interrupt::InterruptLatch;
use {
    stela_contracts::dma::Iommu, stela_contracts::dma::Memory,
    stela_contracts::dma::test_support::Ram,
};

const RAM: u64 = 0x8_0000_0000;
const FIFO: u64 = 0x200;
const SOURCE: u64 = RAM + 0x1000;
const DESTINATION: u64 = RAM + 0x2000;

/// The per-key bit word of the J473 layout, which has no others.
const DISABLE_WORD: u64 = BLOCK_BYTES;
/// J773g `dart-sio` `vm-base`: I/O addresses above 40 bits.
const VM_BASE: u64 = 0x100_0000_0000;

/// Every stream reaches RAM at its I/O address less `iova`, plus [`RAM`].
struct Window {
    iova: u64,
}
impl Iommu for Window {
    fn page(&self) -> u64 {
        0x4000
    }
    fn translate(
        &mut self,
        _: &mut dyn Memory,
        _: usize,
        iova: u64,
        _: u64,
        _: bool,
        _: &mut Access<'_>,
    ) -> Result<u64, DmaError> {
        Ok(iova - self.iova + RAM)
    }
}

/// The J473 ADT: version 3, no `address-width`, one key-disable word.
fn j473() -> Config {
    Config {
        version: 3,
        address_width: DEFAULT_ADDRESS_WIDTH,
        key_disable: KeyDisable {
            bytes: 0x8000,
            word: 0,
            keys: [None; 3],
        },
    }
}

/// The J773g ADT: version 5, 42-bit addresses, a word per key.
fn j773g() -> Config {
    Config {
        version: 5,
        address_width: 42,
        key_disable: KeyDisable {
            bytes: 0x8000,
            word: 0,
            keys: [Some(0x40), Some(0x44), Some(0x48)],
        },
    }
}

fn hex(text: &str) -> alloc::vec::Vec<u8> {
    (0..text.len())
        .step_by(2)
        .map(|at| u8::from_str_radix(&text[at..at + 2], 16).unwrap())
        .collect()
}

fn words(bytes: &[u8]) -> impl Iterator<Item = u32> + '_ {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word))
}

struct Bench {
    aes: Aes,
    ram: Ram,
    latch: InterruptLatch,
    /// The I/O address of [`RAM`].
    iova: u64,
}

impl Bench {
    /// The J473 driver's `_enableAES(true)` on a version 3 engine.
    fn new(uid: Option<[u8; 32]>) -> Self {
        Self::with(j473(), uid, RAM)
    }
    /// `_enableAES(true)` on an engine of `config` whose stream maps
    /// RAM at `iova`.
    fn with(config: Config, uid: Option<[u8; 32]>, iova: u64) -> Self {
        let mut bench = Self {
            aes: Aes::new(config, Some(AicInput::new(0, 0x2f7)), uid).unwrap(),
            ram: Ram::new(RAM, 0x4000),
            latch: InterruptLatch::new(),
            iova,
        };
        assert_eq!(bench.read(STATUS) & STATUS_SEEDED, STATUS_SEEDED);
        let status = bench.read(INT_STATUS);
        bench.write(INT_STATUS, u64::from(status));
        bench.write(INT_ENABLE, 0x3effe);
        bench.write(WATERMARKS, 0x40);
        bench.write(CONTROL, u64::from(CONTROL_START));
        bench
    }
    /// The I/O address of `physical`.
    fn io(&self, physical: u64) -> u64 {
        physical - RAM + self.iova
    }
    /// A 32-bit read, or a write of `value`, that may be refused.
    fn access(&mut self, offset: u64, value: Option<u64>) -> Result<u64, MmioError> {
        let mut access = Access::new(0, 0, &mut self.latch);
        match value {
            Some(value) => self
                .aes
                .write(offset, Width::Word, value, &mut access)
                .map(|()| value),
            None => self.aes.read(offset, Width::Word, &mut access),
        }
    }
    fn read(&mut self, offset: u64) -> u32 {
        let mut access = Access::new(0, 0, &mut self.latch);
        self.aes.read(offset, Width::Word, &mut access).unwrap() as u32
    }
    fn write(&mut self, offset: u64, value: u64) {
        let mut access = Access::new(0, 0, &mut self.latch);
        self.aes
            .write(offset, Width::Word, value, &mut access)
            .unwrap();
    }
    /// Push `command` and run what the FIFO holds, as the engine does
    /// after each device write.
    fn push(&mut self, command: impl IntoIterator<Item = u32>) -> Result<(), MmioError> {
        for word in command {
            self.write(FIFO, u64::from(word));
        }
        let mut access = Access::new(0, 0, &mut self.latch);
        let mut port = Port {
            iommu: &mut Window { iova: self.iova },
            stream: 1,
            memory: &mut self.ram,
            access: &mut access,
        };
        self.aes.run(&mut port)
    }
    fn data(&mut self, command: u32, length: usize) -> Result<(), MmioError> {
        let [source, destination] = [self.io(SOURCE), self.io(DESTINATION)];
        let upper = ((source >> 32) << SOURCE_HIGH_SHIFT | destination >> 32) as u32;
        self.push([
            OPCODE_DATA << OPCODE_SHIFT | command | length as u32,
            upper,
            source as u32,
            destination as u32,
        ])
    }
    fn bytes(&mut self, physical: u64, length: usize) -> alloc::vec::Vec<u8> {
        let mut bytes = vec![0; length];
        assert!(self.ram.read(physical, &mut bytes));
        bytes
    }
}

fn check_unsupported_dart_dma_prefix(write_failure: bool, version: u32) {
    use super::super::dart::Dart;
    use stela_soc_data::dart::Profile;
    const PAGE: u64 = 0x4000;
    const INPUT: u64 = 0x200_7000;
    let mut config = j473();
    config.version = version;
    let mut bench = Bench::with(config, None, RAM);
    bench.ram = Ram::new(RAM, (6 * PAGE) as usize);
    let entry = |address: u64| address >> 4 | 1;
    bench.ram.put_u64(RAM + 8, entry(RAM + PAGE));
    for (slot, page, unsupported) in [
        (1, 2, false),
        (2, 3, !write_failure),
        (3, 4, false),
        (4, 5, write_failure),
    ] {
        bench.ram.put_u64(
            RAM + PAGE + slot * 8,
            entry(RAM + page * PAGE) | 0xfff << 40 | if unsupported { 1 << 9 } else { 0 },
        );
    }
    let length = if write_failure { CHUNK } else { 2 * CHUNK };
    let committed = if write_failure { BLOCK } else { CHUNK };
    let destination = if write_failure {
        0x200_fff0
    } else {
        0x200_c000
    };
    let physical_destination = RAM + 4 * PAGE + (destination & (PAGE - 1));
    let plain = vec![0x11; length];
    assert!(bench.ram.write(RAM + 3 * PAGE - CHUNK as u64, &plain));
    let key = hex("2b7e151628aed2a6abf7158809cf4f3c");
    bench
        .push(
            core::iter::once(OPCODE_KEY << OPCODE_SHIFT | KEY_ENCRYPT | 1 << 16).chain(words(&key)),
        )
        .unwrap();
    let iv = hex("000102030405060708090a0b0c0d0e0f");
    bench
        .push(core::iter::once(OPCODE_IV << OPCODE_SHIFT).chain(words(&iv)))
        .unwrap();
    let mut dart = Dart::for_profile(Profile::T8110, 16, None).unwrap();
    {
        let mut access = Access::new(0, 0, &mut bench.latch);
        for (offset, value) in [(0xc00, 2), (0x1004, 1), (0x1404, RAM >> 14 << 2 | 1)] {
            dart.write(offset, Width::Word, value, &mut access).unwrap();
        }
    }
    for word in [
        OPCODE_DATA << OPCODE_SHIFT | length as u32,
        0,
        INPUT as u32,
        destination as u32,
    ] {
        bench.write(FIFO, u64::from(word));
    }
    let run = |bench: &mut Bench, dart: &mut Dart| {
        let mut access = Access::new(0, 0, &mut bench.latch);
        bench.aes.run(&mut Port {
            iommu: dart,
            stream: 1,
            memory: &mut bench.ram,
            access: &mut access,
        })
    };
    assert_eq!(
        reason(run(&mut bench, &mut dart)),
        "DART descriptor contains unmodeled attributes"
    );
    let prefix = bench.bytes(physical_destination, committed);
    assert_ne!(prefix, plain[..committed]);
    assert_eq!(
        bench.bytes(physical_destination + committed as u64, BLOCK),
        [0; BLOCK]
    );
    let chaining = bench.aes.ivs[0];
    // A natural next guest FIFO store must not authorize DATA offset zero again.
    let submitted = bench.access(
        FIFO,
        Some(u64::from(
            OPCODE_FLAG << OPCODE_SHIFT | FLAG_INTERRUPT | 0x2a,
        )),
    );
    assert!(run(&mut bench, &mut dart).is_err());
    assert_eq!(
        bench.bytes(physical_destination, committed),
        prefix,
        "CBC prefix replay, write_failure={write_failure}"
    );
    assert_eq!(bench.aes.ivs[0], chaining);
    assert_eq!(bench.read(FLAG_COMMAND), 0);
    assert!(!bench.aes.pending());
    assert!(submitted.is_err());
    if version < 2 {
        bench.write(CLEAR_FIFO, 1);
        bench.write(CONTROL, u64::from(CONTROL_STOP));
        bench.write(CONTROL, u64::from(CONTROL_START));
        assert!(!bench.aes.pending());
        assert!(run(&mut bench, &mut dart).is_err());
        assert_eq!(bench.bytes(physical_destination, committed), prefix);
        assert_eq!(bench.aes.ivs[0], chaining);
    }
}

#[test]
fn unsupported_dart_read_after_prefix_never_replays() {
    check_unsupported_dart_dma_prefix(false, 3);
}

#[test]
fn unsupported_dart_write_after_prefix_never_replays() {
    check_unsupported_dart_dma_prefix(true, 3);
}

#[test]
fn unsupported_dma_replay_is_not_authorized_by_fifo_clear_or_start() {
    check_unsupported_dart_dma_prefix(false, 1);
}

/// NIST SP 800-38A F.2.1 (CBC-AES128.Encrypt), the chaining value the
/// engine stores afterwards, and the FLAG that completes the request.
#[test]
fn software_keys_encrypt_cbc_store_the_chain_and_flag_completion() {
    let mut bench = Bench::new(None);
    let key = hex("2b7e151628aed2a6abf7158809cf4f3c");
    let plain = hex("6bc1bee22e409f96e93d7e117393172aae2d8a571e03ac9c9eb76fac45af8e51");
    assert!(bench.ram.write(SOURCE, &plain));
    // KEY context 0, software, 128-bit, encrypt, CBC.
    let command = OPCODE_KEY << 28 | KEY_ENCRYPT | 1 << 16;
    bench
        .push(core::iter::once(command).chain(words(&key)))
        .unwrap();
    let iv = hex("000102030405060708090a0b0c0d0e0f");
    bench
        .push(core::iter::once(OPCODE_IV << 28 | 1 << 26).chain(words(&iv)))
        .unwrap();
    bench.data(1 << 25, 32).unwrap();
    assert_eq!(
        bench.bytes(DESTINATION, 32),
        hex("7649abac8119b246cee98e9b12e9197d5086cb9b507219ee95db113a917678b2")
    );
    bench
        .push([
            OPCODE_STORE_IV << 28 | 1 << 26 | (RAM >> 32) as u32,
            RAM as u32,
        ])
        .unwrap();
    assert_eq!(
        bench.bytes(RAM, 16),
        hex("5086cb9b507219ee95db113a917678b2")
    );
    assert!(!bench.aes.irq_level);
    bench
        .push([OPCODE_FLAG << 28 | FLAG_INTERRUPT | 0x2a])
        .unwrap();
    assert_eq!(bench.read(FLAG_COMMAND), 0x2a);
    assert_eq!(bench.read(INT_STATUS) & INT_FLAG, INT_FLAG);
    assert!(bench.aes.irq_level);
    bench.write(INT_STATUS, u64::from(INT_FLAG));
    assert!(!bench.aes.irq_level);
    assert_eq!(bench.read(FIFO_STATUS), FIFO_LOW | FIFO_EMPTY);
}

#[test]
fn idle_nonretained_power_gate_clears_context_and_preserves_hardware_disables() {
    let mut bench = Bench::new(Some([0x69; 32]));
    let active = bench.aes.lifecycle;
    let key = hex("2b7e151628aed2a6abf7158809cf4f3c");
    let iv = hex("000102030405060708090a0b0c0d0e0f");
    bench
        .push(core::iter::once(OPCODE_KEY << 28 | KEY_ENCRYPT).chain(words(&key)))
        .unwrap();
    bench
        .push(core::iter::once(OPCODE_IV << 28).chain(words(&iv)))
        .unwrap();
    bench
        .push([OPCODE_FLAG << 28 | FLAG_INTERRUPT | 0x2a])
        .unwrap();
    assert!(bench.aes.keys[0].is_some());
    assert_ne!(bench.aes.ivs[0], [0; BLOCK]);
    let irq = Wire::Aic(AicInput::new(0, 0x2f7));
    assert!(bench.latch.is_asserted(irq));

    let off = LifecycleState {
        powered: false,
        clocked: false,
        ..active
    };
    bench.aes.check_lifecycle(off).unwrap();
    bench
        .aes
        .set_lifecycle(off, &mut Access::new(0, 0, &mut bench.latch));
    assert!(bench.access(STATUS, None).is_err());
    assert!(!bench.latch.is_asserted(irq));
    assert_eq!(bench.read(DISABLE_WORD), DISABLE_ALL_BUT_UID1);
    bench.write(DISABLE_WORD, u64::from(DISABLE_UID1));
    assert!(bench.access(DISABLE_WORD + 4, Some(0)).is_err());
    assert!(
        bench
            .access(CONTROL, Some(u64::from(CONTROL_START)))
            .is_err()
    );

    bench.aes.check_lifecycle(active).unwrap();
    bench
        .aes
        .set_lifecycle(active, &mut Access::new(0, 0, &mut bench.latch));
    assert_eq!(bench.read(DISABLE_WORD), DISABLE_KNOWN);
    assert_eq!(bench.read(STATUS), STATUS_SEEDED);
    assert_eq!(bench.read(INT_ENABLE), 0);
    assert_eq!(bench.read(INT_STATUS), 0);
    assert_eq!(bench.read(WATERMARKS), 0);
    assert_eq!(bench.read(FLAG_COMMAND), 0);
    assert!(bench.aes.keys.iter().all(Option::is_none));
    assert_eq!(bench.aes.ivs, [[0; BLOCK]; 4]);
    assert!(
        bench
            .aes
            .check_lifecycle(LifecycleState {
                reset: true,
                ..active
            })
            .is_err()
    );
}

#[test]
fn kept_gates_preserve_keys_chaining_queued_dma_and_completed_interrupts() {
    let mut bench = Bench::new(Some([0x69; 32]));
    let key = hex("2b7e151628aed2a6abf7158809cf4f3c");
    bench
        .push(core::iter::once(OPCODE_KEY << 28 | KEY_ENCRYPT | 1 << 16).chain(words(&key)))
        .unwrap();
    let iv = hex("000102030405060708090a0b0c0d0e0f");
    bench
        .push(core::iter::once(OPCODE_IV << 28).chain(words(&iv)))
        .unwrap();
    bench
        .push([OPCODE_FLAG << 28 | FLAG_INTERRUPT | 0x2a])
        .unwrap();
    let plain = hex("6bc1bee22e409f96e93d7e117393172a");
    assert!(bench.ram.write(SOURCE, &plain));
    assert!(bench.ram.write(DESTINATION, &[0xa5; 16]));
    for word in [
        OPCODE_DATA << 28 | 16,
        ((SOURCE >> 32) << 16 | DESTINATION >> 32) as u32,
        SOURCE as u32,
        DESTINATION as u32,
    ] {
        bench.write(FIFO, u64::from(word));
    }
    let active = bench.aes.lifecycle;
    let fifo = bench.aes.fifo.clone();
    let ivs = bench.aes.ivs;
    let disabled = bench.aes.disabled;
    for paused in [
        LifecycleState {
            clocked: false,
            ..active
        },
        LifecycleState {
            quiesced: true,
            ..active
        },
        LifecycleState {
            powered: false,
            retention: true,
            ..active
        },
    ] {
        bench.aes.check_lifecycle(paused).unwrap();
        bench
            .aes
            .set_lifecycle(paused, &mut Access::new(0, 0, &mut bench.latch));
        assert!(!bench.aes.pending());
        assert!(bench.push([]).is_err());
        assert!(
            bench
                .aes
                .write(
                    FIFO,
                    Width::Word,
                    0,
                    &mut Access::new(0, 0, &mut bench.latch)
                )
                .is_err()
        );
        assert_eq!(bench.aes.fifo, fifo);
        assert_eq!(bench.aes.ivs, ivs);
        assert_eq!(bench.aes.disabled, disabled);
        assert_eq!(bench.bytes(DESTINATION, 16), [0xa5; 16]);
        assert!(bench.latch.is_asserted(Wire::Aic(AicInput::new(0, 0x2f7))));
        bench
            .aes
            .set_lifecycle(active, &mut Access::new(0, 0, &mut bench.latch));
        assert!(bench.aes.pending());
    }
    for destructive in [
        LifecycleState {
            reset: true,
            ..active
        },
        LifecycleState {
            powered: false,
            ..active
        },
    ] {
        assert!(bench.aes.check_lifecycle(destructive).is_err());
        assert_eq!(bench.aes.lifecycle, active);
        assert_eq!(bench.aes.fifo, fifo);
        assert_eq!(bench.aes.uid, Some([0x69; 32]));
    }
    bench.push([]).unwrap();
    assert_eq!(
        bench.bytes(DESTINATION, 16),
        hex("7649abac8119b246cee98e9b12e9197d")
    );
    assert!(bench.aes.fifo.is_empty());
    let completed_iv = bench.aes.ivs;
    bench.push([]).unwrap();
    assert_eq!(
        bench.aes.ivs, completed_iv,
        "gate release cannot replay DMA"
    );
    assert_eq!(bench.read(FLAG_COMMAND), 0x2a);
}

/// NIST SP 800-38A F.5.2 (CTR-AES128.Decrypt) and F.2.2
/// (CBC-AES128.Decrypt), through the second key and IV contexts.
#[test]
fn ctr_and_cbc_decrypt_through_other_contexts() {
    let mut bench = Bench::new(None);
    let key = hex("2b7e151628aed2a6abf7158809cf4f3c");
    let ctr = OPCODE_KEY << 28 | 1 << 27 | 2 << 16;
    bench
        .push(core::iter::once(ctr).chain(words(&key)))
        .unwrap();
    let counter = hex("f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff");
    bench
        .push(core::iter::once(OPCODE_IV << 28 | 3 << 26).chain(words(&counter)))
        .unwrap();
    let cipher = hex("874d6191b620e3261bef6864990db6ce9806f66b7970fdff8617187bb9fffdff");
    assert!(bench.ram.write(SOURCE, &cipher));
    bench.data(1 << 27 | 3 << 25, 32).unwrap();
    assert_eq!(
        bench.bytes(DESTINATION, 32),
        hex("6bc1bee22e409f96e93d7e117393172aae2d8a571e03ac9c9eb76fac45af8e51")
    );
    assert_eq!(bench.aes.ivs[3][15], 0x01, "the counter advanced twice");
    let cbc = OPCODE_KEY << 28 | 1 << 16;
    bench
        .push(core::iter::once(cbc).chain(words(&key)))
        .unwrap();
    let iv = hex("000102030405060708090a0b0c0d0e0f");
    bench
        .push(core::iter::once(OPCODE_IV << 28).chain(words(&iv)))
        .unwrap();
    assert!(
        bench
            .ram
            .write(SOURCE, &hex("7649abac8119b246cee98e9b12e9197d"))
    );
    bench.data(0, 16).unwrap();
    assert_eq!(
        bench.bytes(DESTINATION, 16),
        hex("6bc1bee22e409f96e93d7e117393172a")
    );
    assert_eq!(bench.read(INT_STATUS) & !INT_FIFO_LOW, 0);
}

#[test]
fn hardware_keys_follow_the_key_disable_register() {
    let uid = [0x5a; 32];
    let mut bench = Bench::new(Some(uid));
    assert_eq!(bench.read(DISABLE_WORD), DISABLE_ALL_BUT_UID1);
    assert!(bench.ram.write(SOURCE, &[0; 16]));
    // UID1, 256-bit ECB encrypt matches the same key given in software.
    bench
        .push([OPCODE_KEY << 28 | 1 << 24 | 2 << 22 | KEY_ENCRYPT])
        .unwrap();
    bench.data(0, 16).unwrap();
    let hardware = bench.bytes(DESTINATION, 16);
    let software = OPCODE_KEY << 28 | 2 << 22 | KEY_ENCRYPT;
    bench
        .push(core::iter::once(software).chain(words(&uid)))
        .unwrap();
    bench.data(0, 16).unwrap();
    assert_eq!(bench.bytes(DESTINATION, 16), hardware);
    assert_ne!(hardware, [0; 16]);
    // GID0 is disabled; so is UID1 once software disables it.
    bench
        .push([OPCODE_KEY << 28 | 2 << 24 | KEY_ENCRYPT])
        .unwrap();
    assert_eq!(
        bench.read(INT_STATUS) & INT_KEY_DISABLED[0],
        INT_KEY_DISABLED[0]
    );
    assert!(bench.aes.irq_level);
    bench.data(0, 16).unwrap();
    assert_ne!(bench.read(INT_STATUS) & INT_KEY_INVALID_AT_USE[0], 0);
    bench.write(INT_STATUS, u64::from(INT_KNOWN));
    bench.write(DISABLE_WORD, u64::from(DISABLE_UID1));
    assert_eq!(bench.read(DISABLE_WORD), DISABLE_KNOWN);
    bench.push([OPCODE_KEY << 28 | 1 << 27 | 1 << 24]).unwrap();
    assert_eq!(
        bench.read(INT_STATUS) & INT_KEY_DISABLED[1],
        INT_KEY_DISABLED[1]
    );
    // Without a UID every hardware key reads disabled.
    assert_eq!(Bench::new(None).read(DISABLE_WORD), DISABLE_KNOWN);
}

#[test]
fn stopped_fifos_queue_and_unknown_commands_are_refused() {
    let mut bench = Bench::new(None);
    bench.write(CONTROL, u64::from(CONTROL_STOP));
    bench.push([OPCODE_FLAG << 28; 3]).unwrap();
    assert_eq!(bench.read(FIFO_STATUS), 3 << FIFO_LEVEL_SHIFT | FIFO_LOW);
    assert_eq!(bench.read(STATUS) & STATUS_FIFO_ENABLED, 0);
    bench.write(CONTROL, u64::from(CONTROL_START));
    bench.push([]).unwrap();
    assert_eq!(bench.read(FIFO_STATUS) & FIFO_EMPTY, FIFO_EMPTY);
    // A partial command waits for the rest of its words.
    bench.push([OPCODE_IV << 28, 1, 2]).unwrap();
    assert_eq!(bench.read(FIFO_STATUS) >> FIFO_LEVEL_SHIFT, 3);
    bench.push([3, 4]).unwrap();
    assert_eq!(bench.read(FIFO_STATUS) >> FIFO_LEVEL_SHIFT, 0);
    assert!(bench.push([OPCODE_DSB << 28]).is_err());
    bench.write(CLEAR_FIFO, 1);
    assert!(
        bench
            .push([OPCODE_KEY << 28 | KEY_WRAPPED | 1 << 24])
            .is_err()
    );
    bench.write(CLEAR_FIFO, 1);
    let mut access = Access::new(0, 0, &mut bench.latch);
    assert!(bench.aes.read(0x0, Width::Word, &mut access).is_err());
    assert!(bench.aes.write(0x50, Width::Word, 0, &mut access).is_err());
}

fn reason(result: Result<impl core::fmt::Debug, MmioError>) -> &'static str {
    match result {
        Err(MmioError::Unsupported { reason, .. }) => reason.what(),
        other => panic!("not refused: {other:?}"),
    }
}

/// NIST SP 800-38A F.2.1 through the J773g `dart-sio` window above
/// 2^40: the 10-bit high address fields of `address-width` 42.
#[test]
fn wide_address_engines_move_data_and_chaining_values_above_40_bits() {
    let mut bench = Bench::with(j773g(), None, VM_BASE);
    let key = hex("2b7e151628aed2a6abf7158809cf4f3c");
    let plain = hex("6bc1bee22e409f96e93d7e117393172aae2d8a571e03ac9c9eb76fac45af8e51");
    assert!(bench.ram.write(SOURCE, &plain));
    let command = OPCODE_KEY << OPCODE_SHIFT | KEY_ENCRYPT | 1 << 16;
    bench
        .push(core::iter::once(command).chain(words(&key)))
        .unwrap();
    let iv = hex("000102030405060708090a0b0c0d0e0f");
    bench
        .push(core::iter::once(OPCODE_IV << OPCODE_SHIFT).chain(words(&iv)))
        .unwrap();
    // The driver's `_prepareTransfer` word: 0x100 above bit 16 and 0.
    assert_eq!(bench.io(SOURCE) >> 32, 0x100);
    bench.data(0, 32).unwrap();
    assert_eq!(
        bench.bytes(DESTINATION, 32),
        hex("7649abac8119b246cee98e9b12e9197d5086cb9b507219ee95db113a917678b2")
    );
    let chain = bench.io(RAM);
    bench
        .push([
            OPCODE_STORE_IV << OPCODE_SHIFT | (chain >> 32) as u32,
            chain as u32,
        ])
        .unwrap();
    assert_eq!(
        bench.bytes(RAM, 16),
        hex("5086cb9b507219ee95db113a917678b2")
    );
    assert_eq!(bench.read(INT_STATUS) & !INT_FIFO_LOW, 0);
}

/// Bits the established encodings leave clear stay queued and refuse:
/// a narrow engine has no high address bits 8 and 9.
#[test]
fn reserved_command_bits_refuse_and_stay_queued() {
    let mut bench = Bench::with(j473(), None, VM_BASE);
    let key = hex("2b7e151628aed2a6abf7158809cf4f3c");
    bench
        .push(core::iter::once(OPCODE_KEY << OPCODE_SHIFT | KEY_ENCRYPT).chain(words(&key)))
        .unwrap();
    assert_eq!(
        reason(bench.data(0, 16)),
        "AES DATA address word reserved bits"
    );
    assert_eq!(bench.read(FIFO_STATUS) >> FIFO_LEVEL_SHIFT, 4);
    bench.write(CLEAR_FIFO, 1);
    let mut bench = Bench::new(None);
    assert_eq!(reason(bench.data(DATA_RESERVED, 16)), "AES DATA bit 24");
    bench.write(CLEAR_FIFO, 1);
    let store = OPCODE_STORE_IV << OPCODE_SHIFT | 1 << 8;
    assert_eq!(reason(bench.push([store, 0])), "AES STORE_IV reserved bits");
    bench.write(CLEAR_FIFO, 1);
    let scrambled = OPCODE_SCRAMBLED_KEY << OPCODE_SHIFT;
    assert_eq!(
        reason(bench.push([scrambled])),
        "AES H16 scrambled-key command"
    );
}

/// The J773g (0x40, 0x44, 0x48) and J813 (4, 8, 0xc) key words.
#[test]
fn keys_with_their_own_disable_words_follow_the_adt_offsets() {
    for keys in [[0x40, 0x44, 0x48], [4, 8, 0xc]] {
        let mut config = j773g();
        config.key_disable.keys = keys.map(Some);
        let [uid1, gid0, gid1] = keys.map(|offset| BLOCK_BYTES + u64::from(offset));
        let uid = [0x5a; 32];
        let mut bench = Bench::with(config, Some(uid), VM_BASE);
        assert_eq!(bench.read(uid1), KEY_ENABLED);
        assert_eq!(bench.read(gid0), KEY_DISABLED);
        assert_eq!(bench.read(gid1), KEY_DISABLED);
        assert_eq!(bench.read(DISABLE_WORD), DISABLE_ALL_BUT_UID1);
        // UID1 encrypts as the same key given in software.
        assert!(bench.ram.write(SOURCE, &[0; 16]));
        let hardware = OPCODE_KEY << OPCODE_SHIFT | 1 << 24 | 2 << 22 | KEY_ENCRYPT;
        bench.push([hardware]).unwrap();
        bench.data(0, 16).unwrap();
        let encrypted = bench.bytes(DESTINATION, 16);
        let software = OPCODE_KEY << OPCODE_SHIFT | 2 << 22 | KEY_ENCRYPT;
        bench
            .push(core::iter::once(software).chain(words(&uid)))
            .unwrap();
        bench.data(0, 16).unwrap();
        assert_eq!(bench.bytes(DESTINATION, 16), encrypted);
        // Only the disable value disables, and only through the word.
        assert_eq!(
            reason(bench.access(uid1, Some(u64::from(KEY_ENABLED)))),
            "AES key-disable word value other than the disable value"
        );
        assert_eq!(
            reason(bench.access(DISABLE_WORD, Some(u64::from(DISABLE_UID1)))),
            "AES key-disable bit of a key with its own disable word"
        );
        assert_eq!(bench.read(uid1), KEY_ENABLED);
        bench.write(uid1, u64::from(KEY_DISABLE));
        assert_eq!(bench.read(uid1), KEY_DISABLED);
        assert_eq!(bench.read(DISABLE_WORD), DISABLE_KNOWN);
        bench.push([hardware]).unwrap();
        assert_ne!(bench.read(INT_STATUS) & INT_KEY_DISABLED[0], 0);
        // The fourth key keeps its bit in the shared word.
        bench.write(DISABLE_WORD, 1 << 3);
        assert_eq!(
            reason(bench.access(BLOCK_BYTES + 0x10, None)),
            "unknown AES key-disable register"
        );
    }
    // J473 names no own words.
    let mut bench = Bench::new(None);
    assert_eq!(
        reason(bench.access(BLOCK_BYTES + 0x40, None)),
        "unknown AES key-disable register"
    );
}

/// `_distribute_dkey` writes the third range's 4 in version 5 and 8 in
/// version 6; the distribution to AppleAVD is not modeled.
#[test]
fn dkey_distribution_refuses_at_its_version_register() {
    let dkey = BLOCK_BYTES + 0x8000;
    let mut v5 = Bench::with(j773g(), None, VM_BASE);
    assert_eq!(
        reason(v5.access(dkey + DKEY_DISTRIBUTE_V5, Some(1))),
        "AES DKey distribution to AppleAVD"
    );
    assert_eq!(
        reason(v5.access(dkey + DKEY_DISTRIBUTE_V6, None)),
        "unknown AES DKey register"
    );
    let mut config = j773g();
    config.version = 6;
    let mut v6 = Bench::with(config, None, VM_BASE);
    assert_eq!(
        reason(v6.access(dkey + DKEY_DISTRIBUTE_V6, Some(1))),
        "AES DKey distribution to AppleAVD"
    );
    assert_eq!(
        reason(Bench::new(None).access(dkey + DKEY_DISTRIBUTE_V5, Some(1))),
        "unknown AES register"
    );
    assert!(config.has_dkey() && !j473().has_dkey());
}

#[test]
fn configs_without_an_established_contract_refuse() {
    let with = |change: fn(&mut Config)| {
        let mut config = j773g();
        change(&mut config);
        Aes::new(config, None, None).err().map(|error| error.what())
    };
    assert_eq!(with(|_| {}), None);
    assert_eq!(with(|config| config.address_width = 40), None);
    for version in [4, 7] {
        let mut config = j773g();
        config.version = version;
        assert_eq!(
            Aes::new(config, None, None).err().map(|error| error.what()),
            Some("AES aes-version without an established layout")
        );
    }
    assert_eq!(
        with(|config| config.address_width = 41),
        Some("AES address-width without an established address field width")
    );
    let outside = Some("AES key-disable word outside its range or unaligned");
    assert_eq!(
        with(|config| config.key_disable.keys[0] = Some(0x42)),
        outside
    );
    assert_eq!(with(|config| config.key_disable.word = 0x7ffd), outside);
    assert_eq!(
        with(|config| config.key_disable.keys[2] = Some(0x8000)),
        outside
    );
    assert_eq!(
        with(|config| config.key_disable.keys[1] = Some(0)),
        Some("AES key-disable words overlap")
    );
}
