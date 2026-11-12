use super::*;
use alloc::{vec, vec::Vec};
use stela_drivers::block::{BlockError, RamDisk};
use {
    crate::sart::Sart, stela_contracts::dma::Memory, stela_contracts::dma::Physical,
    stela_contracts::dma::test_support::Ram, stela_contracts::interrupt::InterruptLatch,
    stela_soc_data::sart::Bounds,
};
const RAM: u64 = 0x8_0000_0000;
const SQ: u64 = RAM;
const CQ: u64 = RAM + 0x1000;
const TCB: u64 = RAM + 0x2000;
const IOSQ: u64 = RAM + 0x4000;
const IOCQ: u64 = RAM + 0x5000;
const IOTCB: u64 = RAM + 0x6000;
const DATA: u64 = RAM + 0x9000;
const IRQ: AicInput = AicInput::new(0, 724);
/// CC.SHN for a suspend: `IONVMeController::ShutdownNVMe(inSuspend)`
/// (0x3ddbb48) passes `inSuspend` to `SendShutdownNotification`
/// (0x3ddb908), which writes 11b (0x3ddb9dc) for it and 01b otherwise.
/// NVMe reserves 11b; Apple's controllers take it as the suspend notice.
const SUSPEND: u32 = 3 << 14;

/// One NVMe submission's command fields.
struct Command {
    opcode: u8,
    nsid: u32,
    prp: u64,
    words: [u32; 6],
}

struct Rig {
    ans: Ans,
    sart: Sart,
    ram: Ram,
    latch: InterruptLatch,
}
impl Rig {
    fn new() -> Self {
        Self::with_geometry(QueueGeometry::DEFAULT)
    }
    fn with_geometry(geometry: QueueGeometry) -> Self {
        Self::with_disk(Box::new(RamDisk::new("partition-id", 32, 4096)), geometry)
    }
    fn with_disk(disk: Box<dyn BlockDevice>, geometry: QueueGeometry) -> Self {
        let mut rig = Self {
            ans: Ans::with_geometry(
                disk,
                &[Description {
                    id: 1,
                    kind: 1,
                    blocks: 0,
                }],
                Some(IRQ),
                geometry,
            )
            .unwrap(),
            sart: Sart::new(Bounds::Exclusive),
            ram: Ram::new(RAM, 0x100000),
            latch: InterruptLatch::new(),
        };
        let mut access = Access::new(0, 0, &mut rig.latch);
        for (offset, value) in [(0x40, RAM / 4096), (0x80, 0x20), (0, 0xff)] {
            rig.sart
                .write(offset, Width::Word, value, &mut access)
                .unwrap();
        }
        rig.enable();
        rig
    }
    /// The driver's bring-up: admin queues, TCB bases, LINEAR and CC.EN.
    fn enable(&mut self) {
        let geometry = self.ans.geometry;
        let mut access = Access::new(0, 0, &mut self.latch);
        for (offset, width, value) in [
            (
                AQA,
                Width::Word,
                (1 << 16) | u64::from(geometry.handoff_entries()),
            ),
            (ASQ, Width::Double, SQ),
            (ACQ, Width::Double, CQ),
            (ADMIN_TCB, Width::Double, TCB),
            (IO_TCB, Width::Double, IOTCB),
            (LINEAR, Width::Word, 1),
            (CC, Width::Word, u64::from(ENTRY_SIZES | ENABLE)),
        ] {
            self.ans.write(offset, width, value, &mut access).unwrap();
        }
    }
    fn read(&mut self, offset: u64) -> u64 {
        self.ans
            .read(offset, Width::Word, &mut Access::new(0, 0, &mut self.latch))
            .unwrap()
    }
    fn write(&mut self, offset: u64, value: u64) -> Result<(), MmioError> {
        self.ans.write(
            offset,
            Width::Word,
            value,
            &mut Access::new(0, 0, &mut self.latch),
        )
    }
    fn run(&mut self) -> Result<(), MmioError> {
        let mut access = Access::new(0, 0, &mut self.latch);
        self.ans.run(
            &mut Port {
                iommu: &mut Physical,
                stream: 0,
                memory: &mut self.ram,
                access: &mut access,
            },
            &mut self.sart,
        )
    }
    fn submit(
        &mut self,
        queue: usize,
        tag: u16,
        opcode: u8,
        nsid: u32,
        prp: u64,
        words: [u32; 6],
    ) -> Result<(), MmioError> {
        let command = Command {
            opcode,
            nsid,
            prp,
            words,
        };
        self.submit_granting(queue, tag, command, None)
    }
    /// `submit` with TCB DMA permissions `grant` in place of the
    /// command's own direction.
    fn submit_granting(
        &mut self,
        queue: usize,
        tag: u16,
        command: Command,
        grant: Option<u8>,
    ) -> Result<(), MmioError> {
        self.submit_keyed(queue, tag, command, grant, 0, [0; 64])
    }
    /// `submit_granting` with inline crypto `mode` and key `area` in the
    /// TCB, as SPTM's MAP_PAGES copies them.
    fn submit_keyed(
        &mut self,
        queue: usize,
        tag: u16,
        command: Command,
        grant: Option<u8>,
        mode: u8,
        area: [u8; 64],
    ) -> Result<(), MmioError> {
        let Command {
            opcode,
            nsid,
            prp,
            words,
        } = command;
        let mut cmd = [0u8; 64];
        cmd[0] = opcode;
        cmd[2..4].copy_from_slice(&tag.to_le_bytes());
        cmd[4..8].copy_from_slice(&nsid.to_le_bytes());
        cmd[24..32].copy_from_slice(&prp.to_le_bytes());
        for (i, word) in words.iter().enumerate() {
            cmd[40 + 4 * i..44 + 4 * i].copy_from_slice(&word.to_le_bytes());
        }
        let mut tcb = [0u8; 128];
        tcb[2] = tag as u8;
        tcb[4..6].copy_from_slice(&(words[2] as u16).to_le_bytes());
        tcb[24..32].copy_from_slice(&prp.to_le_bytes());
        tcb[1] = grant.unwrap_or_else(|| commands::Command(cmd).direction(queue).unwrap_or(0));
        tcb[1] |= mode << 2;
        tcb[56..64].copy_from_slice(&cmd[56..64]);
        tcb[64..].copy_from_slice(&area);
        let (sq, tcbs, doorbell) = if queue == 0 {
            (SQ, TCB, ADMIN_SUBMIT)
        } else {
            (IOSQ, IOTCB, IO_SUBMIT)
        };
        assert!(self.ram.write(sq + u64::from(tag) * 64, &cmd));
        assert!(self.ram.write(tcbs + u64::from(tag) * 128, &tcb));
        self.write(doorbell, u64::from(tag))?;
        self.run()
    }
    fn retire(&mut self, queue: usize, tag: u16) {
        let head = (self.ans.queues[queue].head + 1) % self.ans.queues[queue].cq_entries;
        self.write(TCB_INVALIDATE, u64::from(tag)).unwrap();
        self.write(
            if queue == 0 { ADMIN_HEAD } else { IO_HEAD },
            u64::from(head),
        )
        .unwrap();
    }
    fn io_queues(&mut self) {
        self.submit(0, 0, 5, 0, IOCQ, [1 | (3 << 16), 3, 0, 0, 0, 0])
            .unwrap();
        self.retire(0, 0);
        self.submit(
            0,
            1,
            1,
            0,
            IOSQ,
            [
                1 | (u32::from(QUEUE_ENTRIES - 1) << 16),
                1 | (1 << 16),
                0,
                0,
                0,
                0,
            ],
        )
        .unwrap();
        self.retire(0, 1);
    }
    fn bytes(&mut self, address: u64, len: usize) -> alloc::vec::Vec<u8> {
        let mut data = vec![0; len];
        assert!(self.ram.read(address, &mut data));
        data
    }
}

#[test]
fn sptm_split_bases_auto_queue_and_packed_invalidation_status() {
    let mut rig = Rig::new();
    rig.write(CC, 0).unwrap();
    for (register, address) in [
        (ASQ, SQ),
        (ACQ, CQ),
        (ADMIN_TCB, TCB),
        (IO_TCB, IOTCB),
        (IO_SQ, IOSQ),
        (IO_CQ, IOCQ),
    ] {
        rig.write(register, address & 0xffff_ffff).unwrap();
        rig.write(register + 4, address >> 32).unwrap();
        assert_eq!(
            rig.ans.read(
                register,
                Width::Double,
                &mut Access::new(0, 0, &mut rig.latch)
            ),
            Ok(address)
        );
        assert_eq!(
            rig.ans.read(
                register + 4,
                Width::Word,
                &mut Access::new(0, 0, &mut rig.latch)
            ),
            Ok(address >> 32)
        );
    }
    rig.write(MAX_PENDING, 3 | (3 << 16)).unwrap();
    rig.write(AUTO_QUEUE, 1).unwrap();
    rig.write(CC, u64::from(ENTRY_SIZES | ENABLE)).unwrap();
    assert!(rig.ans.queues[1].sq_enabled && rig.ans.queues[1].cq_enabled);
    rig.submit(1, 2, 2, 1, DATA, [0; 6]).unwrap();
    assert_eq!(rig.bytes(IOCQ + 14, 2), [1, 0]);
    rig.retire(1, 2);
    rig.write(IO_SUBMIT, 2).unwrap();
    assert_eq!(
        rig.ans.read(
            TAG_STATUS,
            Width::Word,
            &mut Access::new(0, 0, &mut rig.latch)
        ),
        Ok(1 << 8)
    );
    assert!(rig.write(TCB_INVALIDATE, 2).is_err());
    rig.run().unwrap();
    rig.retire(1, 2);
    assert_eq!(
        rig.ans.read(
            TAG_STATUS,
            Width::Word,
            &mut Access::new(0, 0, &mut rig.latch)
        ),
        Ok(0)
    );
}

#[test]
fn private_identify_matches_features_and_partition_capacity() {
    let mut rig = Rig::new();
    // The original AP builder selects CNS 1 for its NSID-zero query.
    rig.submit(0, 0, 6, 0, DATA, [1, 0, 0, 0, 0, 0]).unwrap();
    assert_eq!(rig.bytes(DATA + 0x204, 4), 1u32.to_le_bytes());
    assert_eq!(&rig.bytes(DATA + 0xc00, 16)[..14], b"Stela ANS2 1.0");
    assert_eq!(rig.bytes(DATA + 0xc1c, 2), [0, 1]);
    assert_eq!(rig.bytes(DATA + 0xc21, 3), [1, 0, 1]);
    assert_eq!(rig.bytes(DATA + 0xe06, 2), 1u16.to_le_bytes());
    let descriptor = u64::from_le_bytes(rig.bytes(DATA + 0xd00, 8).try_into().unwrap());
    rig.retire(0, 0);
    rig.submit(0, 1, 0xa, 0, 0, [0xc0, 0, 0, 0, 0, 0]).unwrap();
    assert_eq!(rig.bytes(CQ + 16, 4), (descriptor as u32).to_le_bytes());
    rig.retire(0, 1);
    rig.submit(0, 0, 0xa, 0, 0, [0xc1, 0, 0, 0, 0, 0]).unwrap();
    assert_eq!(rig.bytes(CQ, 4), ((descriptor >> 32) as u32).to_le_bytes());
    rig.retire(0, 0);
    rig.submit(0, 1, 0xa, 0, 0, [0xc2, 0, 0, 0, 0, 0]).unwrap();
    assert_eq!(rig.bytes(CQ + 16, 4), [0; 4]);
    rig.retire(0, 1);
    assert!(rig.submit(0, 0, 9, 0, 0, [0xc0, 0, 0, 0, 0, 0]).is_err());
}

#[test]
fn namespace_zero_is_invalid_for_namespace_identify() {
    let mut rig = Rig::new();
    let sentinel = vec![0xa5; commands::IDENTIFY_BYTES];
    assert!(rig.ram.write(DATA, &sentinel));
    rig.submit(0, 0, 6, 0, DATA, [0; 6]).unwrap();
    assert_eq!(rig.bytes(DATA, commands::IDENTIFY_BYTES), sentinel);
    let status = u16::from_le_bytes(rig.bytes(CQ + 14, 2).try_into().unwrap()) >> 1;
    assert_eq!(status, commands::INVALID_NAMESPACE);
}

#[test]
fn pmgr_gates_preserve_completed_irq_and_queued_dma_until_execution_resumes() {
    let mut rig = Rig::new();
    rig.submit(0, 0, 6, 0, DATA, [1, 0, 0, 0, 0, 0]).unwrap();
    assert!(rig.latch.is_asserted(Wire::Aic(IRQ)));
    let completion = rig.bytes(CQ, 16);
    // Queue another accepted request without running its DMA yet.
    rig.write(ADMIN_SUBMIT, 1).unwrap();
    let active = rig.ans.lifecycle;
    for gated in [
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
            clocked: false,
            retention: true,
            ..active
        },
    ] {
        rig.ans.check_lifecycle(gated).unwrap();
        rig.ans
            .set_lifecycle(gated, &mut Access::new(0, 0, &mut rig.latch));
        assert!(!rig.ans.pending(0));
        assert!(rig.run().is_err());
        assert_eq!(rig.ans.pending.len(), 1);
        assert_eq!(rig.bytes(CQ, 16), completion);
        assert!(rig.latch.is_asserted(Wire::Aic(IRQ)));
        assert!(
            rig.ans
                .check_lifecycle(LifecycleState {
                    reset: true,
                    ..active
                })
                .is_err()
        );
        rig.ans
            .set_lifecycle(active, &mut Access::new(0, 0, &mut rig.latch));
        assert!(rig.ans.pending(0));
    }
    // A malformed not-yet-filled SQE remains owned rather than being
    // silently completed by gate release.
    assert!(rig.run().is_err());
    assert_eq!(rig.ans.pending.len(), 1);
    assert_eq!(rig.bytes(CQ, 16), completion);
}

#[test]
fn debug_service_masks_round_trip_through_no_dma_completions() {
    let mut rig = Rig::new();
    for (selector, mask) in [(0xb00, 1), (0xb01, 0)] {
        rig.submit(0, 0, 0xc2, 0, 0, [0, 0, 1, selector, mask, 0])
            .unwrap();
        rig.retire(0, 0);
        let slot = rig.ans.queues[0].tail;
        rig.submit(0, 1, 0xc2, 0, 0, [0, 0, 0, selector, 0, 0])
            .unwrap();
        assert_eq!(rig.bytes(CQ + u64::from(slot) * 16, 4), mask.to_le_bytes());
        rig.retire(0, 1);
    }
    assert_eq!(rig.bytes(DATA, 16), [0; 16]);
    // A debug request carries neither a namespace nor a PRP transfer.
    for (namespace, prp) in [(1, 0), (0, DATA)] {
        let mut rig = Rig::new();
        assert!(
            rig.submit(0, 0, 0xc2, namespace, prp, [0, 0, 1, 0xb00, 1, 0])
                .is_err()
        );
        assert_eq!(rig.ans.debug, debug::DebugService::default());
        assert_eq!(admin_status(&mut rig, 0), UNSUPPORTED);
    }
    for words in [
        [1, 0, 1, 0xb00, 1, 0],
        [0, 1, 1, 0xb00, 1, 0],
        [0, 0, 1, 0xb00, 1, 1],
        [0, 0, 0, 0xb00, 1, 0],
    ] {
        let mut rig = Rig::new();
        assert!(rig.submit(0, 0, 0xc2, 0, 0, words).is_err());
        assert_eq!(rig.ans.debug, debug::DebugService::default());
    }
    for (operation, selector, value, expected_status) in [
        (1, 0x100, 0, 2),
        (1, 0xb20, 0x56, 2),
        (0, 0xa20, 0, 0),
        (1, 0xa20, 0, 2),
    ] {
        let mut rig = Rig::new();
        rig.submit(0, 0, 0xc2, 0, 0, [0, 0, operation, selector, value, 0])
            .unwrap();
        let status = u16::from_le_bytes(rig.bytes(CQ + 14, 2).try_into().unwrap()) >> 1;
        assert_eq!(status, expected_status);
        assert_eq!(rig.bytes(CQ, 4), [0; 4]);
    }
}

#[test]
fn identify_completions_wrap_phase_and_interrupt_is_level_masked() {
    let mut rig = Rig::new();
    rig.submit(0, 0, 6, 0, DATA, [1, 0, 0, 0, 0, 0]).unwrap();
    assert_eq!(&rig.bytes(DATA + 24, 10), b"Stela ANS2");
    assert_eq!(rig.bytes(DATA, 4), vec![0x6b, 0x10, 0x6b, 0x10]);
    assert_eq!(rig.bytes(DATA + 260, 1), vec![3]);
    let cqe = rig.bytes(CQ, 16);
    assert_eq!(cqe[14], 1);
    assert_eq!(&cqe[8..14], &[0; 6]);
    assert!(rig.latch.is_asserted(Wire::Aic(IRQ)));
    rig.write(INTMS, u64::from(u32::MAX)).unwrap();
    assert_eq!(rig.ans.interrupt_mask, u32::MAX);
    assert!(!rig.latch.is_asserted(Wire::Aic(IRQ)));
    rig.write(INTMC, 1).unwrap();
    assert_eq!(rig.ans.interrupt_mask, u32::MAX - 1);
    assert!(rig.latch.is_asserted(Wire::Aic(IRQ)));
    assert!(rig.write(ADMIN_SUBMIT, 0).is_err());
    rig.retire(0, 0);
    assert!(!rig.latch.is_asserted(Wire::Aic(IRQ)));
    rig.submit(0, 1, 6, 1, DATA, [0; 6]).unwrap();
    assert_eq!(rig.bytes(DATA + 130, 1), vec![12]);
    rig.retire(0, 1);
    rig.submit(0, 0, 6, 0, DATA, [1, 0, 0, 0, 0, 0]).unwrap();
    assert_eq!(
        rig.bytes(CQ + 14, 2),
        vec![0, 0],
        "phase flips after wrapping CQ"
    );
    assert!(rig.write(ADMIN_HEAD, 2).is_err());
}

/// TCB byte 1 grants DMA directions: a read whose descriptor is in-out
/// grants both and runs; a grant without the command's direction
/// refuses.
#[test]
fn tcb_dma_permissions_must_grant_the_command_direction() {
    let mut rig = Rig::new();
    rig.io_queues();
    let pattern = vec![0x5a; 4096];
    assert!(rig.ram.write(DATA, &pattern));
    rig.submit(1, 2, 1, 1, DATA, [3, 0, 0, 0, 0, 0]).unwrap();
    rig.retire(1, 2);
    assert!(rig.ram.write(DATA, &[0; 4096]));
    rig.submit_granting(
        1,
        3,
        Command {
            opcode: 2,
            nsid: 1,
            prp: DATA,
            words: [3, 0, 0, 0, 0, 0],
        },
        Some(3),
    )
    .unwrap();
    rig.retire(1, 3);
    assert_eq!(rig.bytes(DATA, 4096), pattern);
    let error = rig
        .submit_granting(
            1,
            4,
            Command {
                opcode: 2,
                nsid: 1,
                prp: DATA,
                words: [3, 0, 0, 0, 0, 0],
            },
            Some(2),
        )
        .unwrap_err();
    assert!(
        format!("{error:?}").contains("do not grant the command's direction"),
        "{error:?}"
    );
}

#[test]
fn writes_reads_compare_fua_flush_and_namespace_range() {
    let mut rig = Rig::new();
    rig.io_queues();
    let pattern = vec![0xa5; 4096];
    assert!(rig.ram.write(DATA, &pattern));
    rig.submit(1, 2, 1, 1, DATA, [3, 0, 1 << 30, 0, 0, 0])
        .unwrap();
    rig.retire(1, 2);
    assert!(rig.ram.write(DATA, &[0; 4096]));
    // An access-frequency hint (CDW13 bits 3:0) changes nothing.
    rig.submit(1, 3, 2, 1, DATA, [3, 0, 0, 2, 0, 0]).unwrap();
    rig.retire(1, 3);
    assert_eq!(rig.bytes(DATA, 4096), pattern);
    rig.submit(1, 4, 5, 1, DATA, [3, 0, 0, 0, 0, 0]).unwrap();
    rig.retire(1, 4);
    assert!(rig.ram.write(DATA, &[1; 4096]));
    rig.submit(1, 5, 5, 1, DATA, [3, 0, 0, 0, 0, 0]).unwrap();
    assert_eq!(
        u16::from_le_bytes(rig.bytes(IOCQ + 3 * 16 + 14, 2).try_into().unwrap()) >> 1,
        0x285
    );
    rig.retire(1, 5);
    rig.submit(1, 6, 0, 1, 0, [0; 6]).unwrap();
    rig.retire(1, 6);
    // A barrier Flush (CDW10 bit 0) completes as a flush; other CDW10
    // bits refuse.
    rig.submit(1, 7, 0, 1, 0, [1, 0, 0, 0, 0, 0]).unwrap();
    rig.retire(1, 7);
    let mut refused = Rig::new();
    refused.io_queues();
    assert!(refused.submit(1, 0, 0, 1, 0, [2, 0, 0, 0, 0, 0]).is_err());
    // Apple's priority tier, a write's quick completion and the passive
    // bit are hints too; the bits above them refuse.
    let mut hinted = Rig::new();
    hinted.io_queues();
    hinted
        .submit(1, 0, 2, 1, DATA, [3, 0, 0, 1 << 12 | 3 << 8 | 2, 0, 0])
        .unwrap();
    hinted.retire(1, 0);
    hinted
        .submit(1, 1, 1, 1, DATA, [3, 0, 0, 1 << 11 | 2, 0, 0])
        .unwrap();
    hinted.retire(1, 1);
    let mut refused = Rig::new();
    refused.io_queues();
    assert!(
        refused
            .submit(1, 0, 1, 1, DATA, [3, 0, 0, 1 << 13, 0, 0])
            .is_err()
    );
    rig.submit(1, 8, 2, 1, DATA, [32, 0, 0, 0, 0, 0]).unwrap();
    assert_eq!(
        u16::from_le_bytes(rig.bytes(IOCQ + 2 * 16 + 14, 2).try_into().unwrap()) >> 1,
        0x80
    );
}

#[test]
fn inline_keyed_writes_store_ciphertext_and_reads_return_plaintext() {
    let engine = crate::inline_key::Engine::default();
    engine.provision(zeroize::Zeroizing::new([9; 32]));
    let mut rig = Rig::new();
    rig.ans.inline = engine.clone();
    rig.io_queues();
    let key = [0x3c; 32];
    let area = engine.wrap(&key).unwrap();
    let mut raw = [0; 64];
    raw[32..].copy_from_slice(&key);
    let command = |opcode, unit: u32| Command {
        opcode,
        nsid: 1,
        prp: DATA,
        words: [3, 0, 0, 0, unit, 0],
    };
    let plain: alloc::vec::Vec<u8> = (0..4096).map(|i| (i * 7) as u8).collect();
    assert!(rig.ram.write(DATA, &plain));
    rig.submit_keyed(1, 2, command(1, 7), None, 1, area)
        .unwrap();
    rig.retire(1, 2);
    // The disk holds the data unit's ciphertext.
    assert!(rig.ram.write(DATA, &[0; 4096]));
    rig.submit(1, 3, 2, 1, DATA, [3, 0, 0, 0, 0, 0]).unwrap();
    rig.retire(1, 3);
    let mut expected = plain.clone();
    crate::inline_key::InlineKey::raw(&raw)
        .unwrap()
        .transform(7, &mut expected, true)
        .unwrap();
    assert_eq!(rig.bytes(DATA, 4096), expected);
    // A keyed read decrypts it, with the wrapped or the raw key, but only
    // as its own data unit.
    for (tag, mode, area) in [(4, 1, area), (5, 2, raw)] {
        assert!(rig.ram.write(DATA, &[0; 4096]));
        rig.submit_keyed(1, tag, command(2, 7), None, mode, area)
            .unwrap();
        rig.retire(1, tag);
        assert_eq!(rig.bytes(DATA, 4096), plain);
    }
    rig.submit_keyed(1, 6, command(2, 8), None, 1, area)
        .unwrap();
    rig.retire(1, 6);
    assert_ne!(rig.bytes(DATA, 4096), plain);

    // Keys from another boot, mode 3, a key without a mode, a data unit
    // without a key and keyed flushes refuse.
    let next_boot = crate::inline_key::Engine::default();
    next_boot.provision(zeroize::Zeroizing::new([8; 32]));
    let refusals: [(Command, u8, [u8; 64], Option<crate::inline_key::Engine>); 5] = [
        (command(2, 7), 1, area, Some(next_boot)),
        (command(2, 7), 3, area, None),
        (command(2, 7), 0, area, None),
        (command(2, 7), 0, [0; 64], None),
        (
            Command {
                opcode: 0,
                nsid: 1,
                prp: 0,
                words: [0; 6],
            },
            1,
            area,
            None,
        ),
    ];
    for (command, mode, area, engine) in refusals {
        let mut refused = Rig::new();
        refused.ans.inline = engine.unwrap_or_else(|| rig.ans.inline.clone());
        refused.io_queues();
        assert!(
            refused
                .submit_keyed(1, 0, command, None, mode, area)
                .is_err()
        );
    }
}

/// A RAM disk with a volatile write cache that counts its flushes.
struct Cached {
    disk: RamDisk,
    flushes: alloc::sync::Arc<core::sync::atomic::AtomicUsize>,
}

impl BlockDevice for Cached {
    fn identity(&self) -> &stela_drivers::block::Identity {
        self.disk.identity()
    }
    fn read(&mut self, lba: u64, buffer: &mut [u8]) -> Result<(), BlockError> {
        self.disk.read(lba, buffer)
    }
    fn write(&mut self, lba: u64, data: &[u8]) -> Result<(), BlockError> {
        self.disk.write(lba, data)
    }
    fn flush_cache(&mut self) -> Result<(), BlockError> {
        self.flushes
            .fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        Ok(())
    }
}

#[test]
fn a_host_write_cache_is_reported_and_flushed_only_when_the_guest_asks() {
    let flushes = alloc::sync::Arc::new(core::sync::atomic::AtomicUsize::new(0));
    let mut disk = RamDisk::new("partition-id", 32, 4096);
    disk.identity.features.write_cache = true;
    let disk = Box::new(Cached {
        disk,
        flushes: flushes.clone(),
    });
    let mut rig = Rig::with_disk(disk, QueueGeometry::DEFAULT);
    let flushed = || flushes.load(core::sync::atomic::Ordering::Relaxed);
    rig.submit(0, 0, 6, 0, DATA, [1, 0, 0, 0, 0, 0]).unwrap();
    assert_eq!(rig.bytes(DATA + 525, 1), [1], "VWC");
    rig.retire(0, 0);
    rig.submit(0, 1, 0xa, 0, 0, [6, 0, 0, 0, 0, 0]).unwrap();
    assert_eq!(rig.bytes(CQ + 16, 4), 1u32.to_le_bytes(), "enabled");
    rig.retire(0, 1);
    rig.io_queues();
    // A write completes cached; Force Unit Access and Flush reach the
    // host's cache.
    rig.submit(1, 0, 1, 1, DATA, [3, 0, 0, 0, 0, 0]).unwrap();
    rig.retire(1, 0);
    assert_eq!(flushed(), 0);
    rig.submit(1, 1, 1, 1, DATA, [3, 0, 1 << 30, 0, 0, 0])
        .unwrap();
    rig.retire(1, 1);
    assert_eq!(flushed(), 1);
    rig.submit(1, 2, 0, 1, 0, [0; 6]).unwrap();
    rig.retire(1, 2);
    assert_eq!(flushed(), 2);
    // With this controller's cache off, each write is durable.
    rig.submit(0, 2, 9, 0, 0, [6, 0, 0, 0, 0, 0]).unwrap();
    rig.retire(0, 2);
    rig.submit(1, 3, 1, 1, DATA, [3, 0, 0, 0, 0, 0]).unwrap();
    rig.retire(1, 3);
    assert_eq!(flushed(), 3);
}

/// The status of a command the model cannot carry out.
const UNSUPPORTED: u16 = 1;

/// The status field of the admin completion in `slot`.
fn admin_status(rig: &mut Rig, slot: u64) -> u16 {
    let entry = rig.bytes(CQ + slot * 16, 16);
    u16::from_le_bytes([entry[14], entry[15]]) >> 1
}

#[test]
fn an_unknown_admin_opcode_fails_alone() {
    let mut rig = Rig::new();
    assert!(rig.submit(0, 0, 0xff, 0, 0, [0; 6]).is_err());
    assert_eq!(admin_status(&mut rig, 0), UNSUPPORTED);
    // The refusal is reported once and the next command runs.
    rig.run().unwrap();
    rig.submit(0, 1, 6, 0, DATA, [1, 0, 0, 0, 0, 0]).unwrap();
    assert_eq!(admin_status(&mut rig, 1), 0);
}

#[test]
fn the_health_log_reads_as_a_new_disk_and_others_are_absent() {
    let mut rig = Rig::new();
    // Log 2, 128 dwords, as IONVMeController asks for it.
    rig.submit(0, 0, 2, u32::MAX, DATA, [0x007f_0002, 0, 0, 0, 0, 0])
        .unwrap();
    assert_eq!(admin_status(&mut rig, 0), 0);
    let log = rig.bytes(DATA, 512);
    assert_eq!((log[0], log[3], log[4], log[5]), (0, 100, 10, 0));
    // A log it lacks, and more of the log than there is.
    for (words, status) in [(0x007f_0001, 0x109), (0x0080_0002, 2)] {
        let mut rig = Rig::new();
        rig.submit(0, 0, 2, u32::MAX, DATA, [words, 0, 0, 0, 0, 0])
            .unwrap();
        assert_eq!(admin_status(&mut rig, 0), status);
    }
}

#[test]
fn protected_tcb_commands_fail_alone() {
    let mut rig = Rig::new();
    assert!(rig.ram.write(TCB + 64, &[1]));
    // Install an ordinary identify first, then change its protected TCB
    // before the deferred DMA service runs.
    let mut cmd = [0; 64];
    cmd[0] = 6;
    cmd[24..32].copy_from_slice(&DATA.to_le_bytes());
    cmd[40] = 1;
    assert!(rig.ram.write(SQ, &cmd));
    assert!(rig.ram.write(TCB + 1, &[1]));
    assert!(rig.ram.write(TCB + 24, &DATA.to_le_bytes()));
    rig.write(ADMIN_SUBMIT, 0).unwrap();
    assert!(rig.run().is_err());
    let rejected = rig.ans.last_submission().unwrap();
    assert_eq!((rejected.queue, rejected.tag), (0, 0));
    assert_eq!(rejected.command[0], 6);
    assert_eq!(rejected.tcb[64], 1);
    assert_eq!(admin_status(&mut rig, 0), UNSUPPORTED);
    // TCB 0x38..0x40 holds zero or a copy of CDW14 and CDW15.
    let mut rig = Rig::new();
    assert!(rig.ram.write(SQ, &cmd));
    assert!(rig.ram.write(TCB + 1, &[1]));
    assert!(rig.ram.write(TCB + 24, &DATA.to_le_bytes()));
    assert!(rig.ram.write(TCB + 60, &[1]));
    rig.write(ADMIN_SUBMIT, 0).unwrap();
    assert!(rig.run().is_err());
    assert_eq!(admin_status(&mut rig, 0), UNSUPPORTED);
}

#[test]
fn queues_and_tcbs_bypass_the_sart() {
    // Queues, TCBs and PRP data are physical: an Identify completes
    // with every SART entry disabled.
    let mut rig = Rig::new();
    rig.sart
        .write(0, Width::Word, 0, &mut Access::new(0, 0, &mut rig.latch))
        .unwrap();
    rig.submit(0, 0, 6, 0, DATA, [1, 0, 0, 0, 0, 0]).unwrap();
    assert_ne!(rig.bytes(CQ, 16), vec![0; 16]);
    assert!(rig.latch.is_asserted(Wire::Aic(IRQ)));
}

#[test]
fn queue_depth_satisfies_the_sptm_queue_entries_contract() {
    // ANS3CGv2 hands SPTM INIT min(MQES + 1, 256 + 1) - 1, which must
    // equal the tree's nvme-queue-entries.
    const NVME_QUEUE_ENTRIES: u64 = 64;
    let mut rig = Rig::new();
    let cap = rig
        .ans
        .read(CAP, Width::Double, &mut Access::new(0, 0, &mut rig.latch))
        .unwrap();
    assert_eq!((cap & 0xffff) + 1, u64::from(QUEUE_ENTRIES));
    assert_eq!(((cap & 0xffff) + 1).min(256 + 1) - 1, NVME_QUEUE_ENTRIES);
    // The kernel's 65-entry admin rings fit; a deeper ring does not.
    rig.write(CC, 0).unwrap();
    for (base, address) in [(ASQ, SQ), (ACQ, CQ)] {
        rig.ans
            .write(
                base,
                Width::Double,
                address,
                &mut Access::new(0, 0, &mut rig.latch),
            )
            .unwrap();
    }
    rig.write(AQA, 65 | 65 << 16).unwrap();
    assert!(rig.write(CC, u64::from(ENTRY_SIZES | ENABLE)).is_err());
    rig.write(AQA, 64 | 64 << 16).unwrap();
    rig.write(CC, u64::from(ENTRY_SIZES | ENABLE)).unwrap();
    assert_eq!(rig.ans.queues[0].sq_entries, QUEUE_ENTRIES);
    assert_eq!(rig.ans.queues[0].cq_entries, QUEUE_ENTRIES);
    // Create I/O CQ: QSIZE 65 (66 entries) fails, QSIZE 64 succeeds.
    rig.submit(0, 0, 5, 0, IOCQ, [1 | (65 << 16), 3, 0, 0, 0, 0])
        .unwrap();
    assert_ne!(rig.bytes(CQ + 14, 2), [1, 0]);
    rig.retire(0, 0);
    rig.submit(0, 1, 5, 0, IOCQ, [1 | (64 << 16), 3, 0, 0, 0, 0])
        .unwrap();
    assert_eq!(rig.bytes(CQ + 16 + 14, 2), [1, 0]);
    assert_eq!(rig.ans.queues[1].cq_entries, QUEUE_ENTRIES);
    // Tags stay below 64 whatever the ring depth.
    assert!(rig.write(ADMIN_SUBMIT, u64::from(TAGS)).is_err());
    // I/O queue attributes are 0's based like AQA.
    assert!(rig.write(MAX_PENDING, 65).is_err());
    rig.write(MAX_PENDING, 64 | 64 << 16).unwrap();
}

#[test]
fn restore_mode_suspend_shutdown_and_the_sleep_notice_capability() {
    let mut rig = Rig::new();
    // The restore boot's MODESEL 4 before enabling, then CC.EN.
    rig.write(CC, 0).unwrap();
    for (base, address) in [(ASQ, SQ), (ACQ, CQ)] {
        rig.ans
            .write(
                base,
                Width::Double,
                address,
                &mut Access::new(0, 0, &mut rig.latch),
            )
            .unwrap();
    }
    rig.write(MODESEL, u64::from(MODE_RESTORE)).unwrap();
    rig.write(CC, u64::from(ENTRY_SIZES | ENABLE)).unwrap();
    assert_eq!(rig.read(CSTS), u64::from(READY));
    assert_eq!(rig.read(MODESEL), u64::from(MODE_RESTORE));
    assert_eq!(rig.read(SLEEP_NOTIFY), 0);
    // A suspend notification completes like a normal one.
    rig.write(CC, u64::from(ENTRY_SIZES | ENABLE | SUSPEND))
        .unwrap();
    rig.run().unwrap();
    assert_eq!(rig.read(CSTS), u64::from(READY | SHUTDOWN_COMPLETE));
    // Clearing SHN keeps the completed status (`HandleAbruptShutdown`
    // 0x3dc0fc8 clears it and polls SHST for 2).
    rig.write(CC, u64::from(ENTRY_SIZES | ENABLE)).unwrap();
    rig.run().unwrap();
    assert_eq!(rig.read(CSTS), u64::from(READY | SHUTDOWN_COMPLETE));
    // The abrupt notification after it completes again.
    rig.write(CC, u64::from(ENTRY_SIZES | ENABLE | 2 << 14))
        .unwrap();
    rig.run().unwrap();
    assert_eq!(rig.read(CSTS), u64::from(READY | SHUTDOWN_COMPLETE));
    // The hand-off leaves a fresh controller in linear mode.
    let disk = Box::new(RamDisk::new("partition-id", 32, 4096));
    let namespaces = [Description {
        id: 1,
        kind: 1,
        blocks: 0,
    }];
    assert!(!Ans::new(disk, &namespaces, None).unwrap().linear);
    let disk = Box::new(RamDisk::new("partition-id", 32, 4096));
    let ans = Ans::new(disk, &namespaces, None)
        .unwrap()
        .handed_off_linear();
    assert!(ans.linear);
}

/// Disable, select MODESEL 4 and re-enable, as the restore boot does.
fn restore_mode(rig: &mut Rig) {
    reenable(rig, MODE_RESTORE);
}

/// `IONVMeController::EnableController` (0x3ddaec0): CC 0, the admin
/// queues, `PreCCEnableHook`'s MODESEL (0x3dbdfb8), then CC.EN.
fn reenable(rig: &mut Rig, mode: u32) {
    rig.write(CC, 0).unwrap();
    for (base, address) in [(ASQ, SQ), (ACQ, CQ)] {
        rig.ans
            .write(
                base,
                Width::Double,
                address,
                &mut Access::new(0, 0, &mut rig.latch),
            )
            .unwrap();
    }
    rig.write(MODESEL, u64::from(mode)).unwrap();
    rig.write(CC, u64::from(ENTRY_SIZES | ENABLE)).unwrap();
}

/// `update_NAND_firmware` on this die. `nvmefwupdater --query` reads the
/// MSP type (selector 10), boot-loads the loader (selector 4) and reads
/// the NAND descriptor (selector 5). restored finds no package named
/// after that descriptor and runs `--skip`: selectors 3, 8 and 7 reach
/// no register, and `SetNVMeState(1, 1)` (selector 6, 0x3dc9aec)
/// re-initializes the controller after `SetBFHMode(false)` cleared the
/// MODESEL 4 flag (`InitializeNVMe` 0x3dbdf28).
#[test]
fn restore_nand_query_then_skip_returns_the_die_to_regular_mode() {
    let mut rig = Rig::new();
    restore_mode(&mut rig);
    rig.io_queues();
    // The completion status and result of admin command `tag`.
    let admin = |rig: &mut Rig, tag: u16, opcode: u8, nsid: u32, prp: u64, words| {
        let slot = rig.ans.queues[0].tail;
        rig.submit(0, tag, opcode, nsid, prp, words).unwrap();
        let cqe = rig.bytes(CQ + u64::from(slot) * 16, 16);
        rig.retire(0, tag);
        (
            u16::from_le_bytes([cqe[14], cqe[15]]) >> 1,
            u32::from_le_bytes(cqe[..4].try_into().unwrap()),
        )
    };
    let feature = |rig: &mut Rig, tag, id| admin(rig, tag, 0xa, 0, 0, [id, 0, 0, 0, 0, 0]);
    assert_eq!(feature(&mut rig, 2, 0xc2), (0, 0), "GetMSPType: part 0");
    let loader: Vec<u8> = (0..0x1000u32).map(|i| (i * 13 + 1) as u8).collect();
    assert!(rig.ram.write(DATA, &loader));
    let download = [0x3ff, 0, 0, 0, 0, bfh::crc32c(&loader)];
    assert_eq!(admin(&mut rig, 3, 0x11, 0, DATA, download).0, 0);
    assert_eq!(admin(&mut rig, 4, 0x10, 0, 0, [1 << 3, 0, 0, 0, 0, 0]).0, 0);
    let (_, low) = feature(&mut rig, 5, 0xc0);
    let (_, high) = feature(&mut rig, 6, 0xc1);
    assert_eq!(u64::from(high) << 32 | u64::from(low), die::DESCRIPTOR);

    reenable(&mut rig, MODE_REGULAR);
    assert_eq!(rig.read(CSTS), u64::from(READY));
    assert_eq!(rig.read(MODESEL), u64::from(MODE_REGULAR));
    assert!(
        !rig.ans.queues[1].sq_enabled,
        "the reset dropped the I/O pair"
    );
    rig.io_queues();
    // `IdentifyNamespaces`, then `CheckClogVersionMismatch`'s
    // namespace-1 Identify (0x3df6c90), which must not answer Invalid
    // Namespace, and the type `IsFWNamespaceTypeMismatch` compares.
    assert_eq!(admin(&mut rig, 2, 6, 0, DATA, [1, 0, 0, 0, 0, 0]).0, 0);
    assert_eq!(rig.bytes(DATA + 0x204, 4), 1u32.to_le_bytes());
    assert_eq!(admin(&mut rig, 3, 6, 1, DATA, [0; 6]).0, 0);
    assert_eq!(rig.bytes(DATA + 0x180, 1), [1]);
    // Regular mode serves no boot-from-host command; the loader's
    // record stays.
    assert!(rig.submit(0, 4, 0x11, 0, DATA, download).is_err());
    assert_eq!(rig.ans.boot_image().unwrap().sections.len(), 1);
}

#[test]
fn boot_from_host_download_is_ordered_crc_checked_and_recorded() {
    let image: Vec<u8> = (0..0x1800u32).map(|i| (i * 7 + 3) as u8).collect();
    let chunk = |rig: &mut Rig, tag, offset: usize, len: usize, crc: u32| {
        assert!(rig.ram.write(DATA, &image[offset..offset + len]));
        let words = [len as u32 / 4 - 1, offset as u32 / 4, 0, 0, 0, crc];
        rig.submit(0, tag, 0x11, 0, DATA, words)
    };
    let crc = |offset: usize, len: usize| bfh::crc32c(&image[offset..offset + len]);
    // Outside the restore mode the commands refuse.
    let mut rig = Rig::new();
    assert!(chunk(&mut rig, 0, 0, 0x1000, crc(0, 0x1000)).is_err());
    let mut rig = Rig::new();
    restore_mode(&mut rig);
    // The parameter block, then the loader in two chunks.
    assert!(rig.ram.write(DATA + 0x2000, b"FWUP"));
    rig.submit(0, 0, 0xd1, 0, DATA + 0x2000, [0, 0, 0, 0, 0, 0])
        .unwrap();
    assert_eq!(rig.bytes(CQ + 14, 2), [1, 0]);
    rig.retire(0, 0);
    chunk(&mut rig, 1, 0, 0x1000, crc(0, 0x1000)).unwrap();
    assert_eq!(rig.bytes(CQ + 16 + 14, 2), [1, 0]);
    rig.retire(0, 1);
    // The two-entry ring wraps: success with the inverted phase.
    chunk(&mut rig, 2, 0x1000, 0x800, crc(0x1000, 0x800)).unwrap();
    assert_eq!(rig.bytes(CQ + 14, 2), [0, 0]);
    rig.retire(0, 2);
    // Slot 0, commit action 1 records the identity and succeeds.
    rig.submit(0, 3, 0x10, 0, 0, [1 << 3, 0, 0, 0, 0, 0])
        .unwrap();
    assert_eq!(rig.bytes(CQ + 16 + 14, 2), [0, 0]);
    rig.retire(0, 3);
    let recorded = rig.ans.boot_image().unwrap();
    assert_eq!(recorded.sections.len(), 1);
    assert_eq!(recorded.sections[0].0, 0);
    assert_eq!(recorded.sections[0].1.bytes, 0x1800);
    assert_eq!(recorded.sections[0].1.crc32c, crc(0, 0x1800));
    assert_eq!(recorded.params.as_ref().unwrap().bytes, 4);
    // The identity survives a controller reset.
    restore_mode(&mut rig);
    assert!(rig.ans.boot_image().is_some());

    // A CRC mismatch, a gap, an out-of-order section and an
    // unsupported commit action each stop without completing.
    let refused = |words: [u32; 6], opcode| {
        let mut rig = Rig::new();
        restore_mode(&mut rig);
        assert!(rig.ram.write(DATA, &image[..0x1000]));
        rig.submit(0, 0, opcode, 0, DATA, words).is_err()
    };
    assert!(refused([0x3ff, 0, 0, 0, 0, crc(0, 0x1000) ^ 1], 0x11));
    assert!(refused([0x3ff, 4, 0, 0, 0, crc(0, 0x1000)], 0x11));
    assert!(refused([0x3ff, 0, 0, 0, 1, crc(0, 0x1000)], 0x11));
    assert!(refused([0x3ff, 0, 0, 0, 3, crc(0, 0x1000)], 0x11));
    assert!(refused([0x3ff, 0, 1, 0, 0, crc(0, 0x1000)], 0x11));
    assert!(refused([2 << 3, 0, 0, 0, 0, 0], 0x10));
    assert!(refused([1 << 3, 0, 0, 0, 0, 0], 0x10));
    // A controller reset discards an uncommitted download.
    let mut rig = Rig::new();
    restore_mode(&mut rig);
    chunk(&mut rig, 0, 0, 0x1000, crc(0, 0x1000)).unwrap();
    rig.retire(0, 0);
    restore_mode(&mut rig);
    assert!(
        rig.submit(0, 0, 0x10, 0, 0, [1 << 3, 0, 0, 0, 0, 0])
            .is_err()
    );
}

#[test]
fn shutdown_deletes_the_io_sq_before_its_cq() {
    let mut rig = Rig::new();
    rig.io_queues();
    // The status of admin command `tag`, retired after it completes.
    let delete = |rig: &mut Rig, tag: u16, opcode: u8, id: u32| {
        let slot = rig.ans.queues[0].tail;
        rig.submit(0, tag, opcode, 0, 0, [id, 0, 0, 0, 0, 0])
            .unwrap();
        let status = rig.bytes(CQ + u64::from(slot) * 16 + 14, 2);
        rig.retire(0, tag);
        u16::from_le_bytes([status[0], status[1]]) >> 1
    };
    // The CQ cannot go while its SQ exists; queue 2 does not exist.
    assert_eq!(delete(&mut rig, 2, 4, 1), 0x10c);
    assert_eq!(delete(&mut rig, 3, 0, 2), 0x101);
    assert_eq!(delete(&mut rig, 4, 0, 1), 0);
    assert!(!rig.ans.queues[1].sq_enabled);
    assert_eq!(delete(&mut rig, 5, 0, 1), 0x101);
    assert_eq!(delete(&mut rig, 6, 4, 1), 0);
    assert!(!rig.ans.queues[1].cq_enabled);
    // A deletion carrying a data pointer is outside the contract.
    assert!(rig.submit(0, 7, 4, 0, DATA, [1, 0, 0, 0, 0, 0]).is_err());
}

#[test]
fn full_queues_do_not_overwrite_and_invalid_config_has_no_effect() {
    let mut rig = Rig::new();
    rig.submit(0, 0, 6, 0, DATA, [1, 0, 0, 0, 0, 0]).unwrap();
    rig.submit(0, 1, 6, 0, DATA, [1, 0, 0, 0, 0, 0]).unwrap();
    let before = rig.bytes(CQ, 32);
    rig.submit(0, 2, 6, 0, DATA, [1, 0, 0, 0, 0, 0]).unwrap();
    // The third executes; its completion waits for room.
    assert!(rig.ans.pending.is_empty());
    assert_eq!(rig.ans.completed.len(), 1);
    assert_eq!(rig.bytes(CQ, 32), before);
    assert!(
        rig.write(CC, u64::from(ENTRY_SIZES | ENABLE | 1 << 12))
            .is_err()
    );
    assert_eq!(rig.ans.config, ENTRY_SIZES | ENABLE);
    assert!(rig.write(MODESEL, 1).is_err());
    // MODESEL applies at enable.
    assert!(rig.write(MODESEL, u64::from(MODE_RESTORE)).is_err());
    assert!(rig.write(AUTO_QUEUE, 2).is_err());
    assert!(rig.write(ADMIN_HEAD, 3).is_err());
}
const LARGE_CQ: u64 = RAM + 0x10000;
const LARGE_TCB: u64 = RAM + 0x20000;
const LARGE_DATA: u64 = RAM + 0x70000;
fn large(handoff: u32) -> Rig {
    large_mps(handoff, 0)
}
fn large_mps(handoff: u32, mps: u32) -> Rig {
    let geometry = QueueGeometry::linear(handoff).unwrap();
    let mut rig = Rig::with_geometry(geometry);
    rig.write(CC, 0).unwrap();
    for (register, address) in [(ASQ, SQ), (ACQ, LARGE_CQ), (ADMIN_TCB, LARGE_TCB)] {
        rig.ans
            .write(
                register,
                Width::Double,
                address,
                &mut Access::new(0, 0, &mut rig.latch),
            )
            .unwrap();
    }
    rig.write(NVMMU_CONTROL_REG, 63).unwrap();
    rig.write(AQA, handoff as u64 | (handoff as u64) << 16)
        .unwrap();
    rig.write(CC, u64::from(ENTRY_SIZES | ENABLE | mps << 7))
        .unwrap();
    rig
}
fn large_command(rig: &mut Rig, tag: u16, opcode: u8, prp: u64, words: [u32; 6]) {
    let mut cmd = [0u8; 64];
    cmd[0] = opcode;
    cmd[2..4].copy_from_slice(&tag.to_le_bytes());
    for (i, value) in words.into_iter().enumerate() {
        cmd[40 + i * 4..44 + i * 4].copy_from_slice(&value.to_le_bytes());
    }
    let mut tcb = [0; 128];
    tcb[1] = 3;
    tcb[2] = tag as u8;
    tcb[24..32].copy_from_slice(&prp.to_le_bytes());
    assert!(rig.ram.write(SQ + u64::from(tag) * 64, &cmd));
    assert!(rig.ram.write(LARGE_TCB + u64::from(tag) * 128, &tcb));
}
#[test]
fn dynamic_linear_geometry_wraps_non_power_of_two_rings_and_stalls_without_overwrite() {
    for handoff in [1, 253, 254, 256] {
        let mut rig = large(handoff);
        assert_eq!(rig.read(CAP) & 0xffff, u64::from(handoff));
        assert_eq!(rig.read(NVMMU_CONTROL_REG), 63);
        for index in 0..handoff + 1 {
            let tag = (index % handoff) as u16;
            if index >= handoff {
                rig.write(TCB_INVALIDATE, u64::from(tag)).unwrap();
            }
            large_command(&mut rig, tag, 6, LARGE_DATA, [1, 0, 0, 0, 0, 0]);
            rig.write(ADMIN_SUBMIT, u64::from(tag)).unwrap();
            rig.run().unwrap();
        }
        assert_eq!(rig.ans.queues[0].tail, 0);
        assert!(!rig.ans.queues[0].phase);
        assert_eq!(rig.ans.queues[0].pending, (handoff + 1) as u16);
        let before = rig.bytes(LARGE_CQ, (handoff as usize + 1) * 16);
        let tag = ((handoff + 1) % handoff) as u16;
        rig.write(TCB_INVALIDATE, u64::from(tag)).unwrap();
        large_command(&mut rig, tag, 6, LARGE_DATA, [1, 0, 0, 0, 0, 0]);
        rig.write(ADMIN_SUBMIT, u64::from(tag)).unwrap();
        rig.run().unwrap();
        assert_eq!(rig.bytes(LARGE_CQ, before.len()), before);
        assert_eq!(rig.ans.completed.len(), 1);
        assert!(rig.write(CC, 0).is_err());
        rig.write(ADMIN_HEAD, 1).unwrap();
        rig.run().unwrap();
        assert!(rig.ans.pending.is_empty());
        assert!(rig.ans.completed.is_empty());
        assert_eq!(rig.bytes(LARGE_CQ + 14, 2), [0, 0]);
        assert!(rig.write(ADMIN_SUBMIT, u64::from(handoff)).is_err());
    }
}
#[test]
fn large_queue_creation_auto_queue_and_live_resize_use_one_geometry() {
    let mut rig = large(253);
    let io_sq = RAM + 0x30000;
    let io_cq = RAM + 0x40000;
    for (tag, opcode, prp, words) in [
        (0, 5, io_cq, [1 | (253 << 16), 3, 0, 0, 0, 0]),
        (1, 1, io_sq, [1 | (253 << 16), 1 | (1 << 16), 0, 0, 0, 0]),
    ] {
        large_command(&mut rig, tag, opcode, prp, words);
        rig.write(ADMIN_SUBMIT, u64::from(tag)).unwrap();
        rig.run().unwrap();
        rig.retire(0, tag);
    }
    assert_eq!(rig.ans.queues[1].sq_entries, 254);
    assert_eq!(rig.ans.queues[1].cq_entries, 254);
    rig.write(MAX_PENDING, 253 | (253 << 16)).unwrap();
    let attributes = rig.ans.queue_attributes;
    assert!(rig.write(MAX_PENDING, 252 | (253 << 16)).is_err());
    assert_eq!(rig.ans.queue_attributes, attributes);
    assert!(rig.write(MAX_PENDING, 254 | (253 << 16)).is_err());
    rig.write(CC, 0).unwrap();
    assert_eq!(rig.read(CAP) & 0xffff, 253);
    assert!(rig.write(NVMMU_CONTROL_REG, 62).is_err());
    assert_eq!(rig.read(NVMMU_CONTROL_REG), 63);
    // Auto management programs the same two ring sizes before EN.
    for (register, address) in [(ASQ, SQ), (ACQ, LARGE_CQ), (IO_SQ, io_sq), (IO_CQ, io_cq)] {
        rig.ans
            .write(
                register,
                Width::Double,
                address,
                &mut Access::new(0, 0, &mut rig.latch),
            )
            .unwrap();
    }
    rig.write(AUTO_QUEUE, 1).unwrap();
    rig.write(CC, u64::from(ENTRY_SIZES | ENABLE)).unwrap();
    assert_eq!(rig.ans.queues[1].sq_entries, 254);
    assert_eq!(rig.ans.queues[1].cq_entries, 254);
}
struct FaultWrites<'a> {
    ram: &'a mut Ram,
    at: u64,
    fail: bool,
}
impl Memory for FaultWrites<'_> {
    fn read(&mut self, p: u64, b: &mut [u8]) -> bool {
        self.ram.read(p, b)
    }
    fn write(&mut self, p: u64, b: &[u8]) -> bool {
        if self.fail && p == self.at {
            self.fail = false;
            false
        } else {
            self.ram.write(p, b)
        }
    }
}
#[test]
fn completed_effects_survive_cq_fault_retention_and_retry_without_reexecution() {
    let mut rig = large(253);
    let tag = 252;
    large_command(&mut rig, tag, 0xc2, 0, [0, 0, 1, 0xb00, 1, 0]);
    rig.write(ADMIN_SUBMIT, u64::from(tag)).unwrap();
    let mut access = Access::new(0, 0, &mut rig.latch);
    let mut memory = FaultWrites {
        ram: &mut rig.ram,
        at: LARGE_CQ,
        fail: true,
    };
    assert!(
        rig.ans
            .run(
                &mut Port {
                    iommu: &mut Physical,
                    stream: 0,
                    memory: &mut memory,
                    access: &mut access
                },
                &mut rig.sart
            )
            .is_err()
    );
    // Executed once: its completion stays owned until published.
    assert!(rig.ans.pending.is_empty());
    assert_eq!(rig.ans.completed.len(), 1);
    assert_eq!(rig.ans.queues[0].pending, 0);
    assert_eq!(rig.ans.queues[0].tail, 0);
    assert!(!rig.latch.is_asserted(Wire::Aic(IRQ)));
    assert_eq!(
        rig.read(TAG_STATUS + 4 * (u64::from(tag) / 8)),
        1 << ((tag % 8) * 4)
    );
    assert!(rig.write(TCB_INVALIDATE, u64::from(tag)).is_err());
    let live = rig.ans.lifecycle;
    let stopped = LifecycleState {
        powered: false,
        clocked: false,
        quiesced: true,
        retention: true,
        ..live
    };
    rig.ans.check_lifecycle(stopped).unwrap();
    rig.ans
        .set_lifecycle(stopped, &mut Access::new(0, 0, &mut rig.latch));
    assert!(rig.run().is_err());
    assert_eq!(rig.ans.completed.len(), 1);
    assert!(
        rig.ans
            .check_lifecycle(LifecycleState {
                retention: false,
                ..stopped
            })
            .is_err()
    );
    rig.ans
        .set_lifecycle(live, &mut Access::new(0, 0, &mut rig.latch));
    // A changed source command cannot replace already executed effects.
    large_command(&mut rig, tag, 0xff, 0, [0; 6]);
    rig.run().unwrap();
    assert!(rig.ans.pending.is_empty());
    assert!(rig.ans.completed.is_empty());
    assert_eq!(rig.bytes(LARGE_CQ + 12, 4), [252, 0, 1, 0]);
    assert!(rig.latch.is_asserted(Wire::Aic(IRQ)));
}
#[test]
fn partial_data_dma_fails_its_command() {
    let mut rig = large_mps(253, 2);
    let tag = 252;
    large_command(&mut rig, tag, 6, LARGE_DATA + 4092, [1, 0, 0, 0, 0, 0]);
    assert!(rig.ram.write(
        LARGE_TCB + u64::from(tag) * 128 + 32,
        &(LARGE_DATA + 4096).to_le_bytes()
    ));
    rig.write(ADMIN_SUBMIT, u64::from(tag)).unwrap();
    let mut access = Access::new(0, 0, &mut rig.latch);
    let mut memory = FaultWrites {
        ram: &mut rig.ram,
        at: LARGE_DATA + 4096,
        fail: true,
    };
    assert!(
        rig.ans
            .run(
                &mut Port {
                    iommu: &mut Physical,
                    stream: 0,
                    memory: &mut memory,
                    access: &mut access
                },
                &mut rig.sart
            )
            .is_err()
    );
    assert_eq!(rig.bytes(LARGE_DATA + 4092, 4), [0x6b, 0x10, 0x6b, 0x10]);
    let entry = rig.bytes(LARGE_CQ, 16);
    assert_eq!(u16::from_le_bytes([entry[14], entry[15]]) >> 1, UNSUPPORTED);
    assert!(rig.ans.pending.is_empty());
    assert!(rig.ans.completed.is_empty());
    assert_eq!(rig.ans.queues[0].pending, 1);
}
#[test]
fn declared_extents_and_per_queue_tag_bounds_refuse_before_mutation() {
    let mut rig = large(256);
    rig.write(CC, 0).unwrap();
    for register in [ASQ, ACQ, ADMIN_TCB, IO_SQ, IO_CQ, IO_TCB] {
        let before = rig.ans.base(register).unwrap();
        assert!(
            rig.ans
                .write(
                    register,
                    Width::Double,
                    !4095,
                    &mut Access::new(0, 0, &mut rig.latch)
                )
                .is_err()
        );
        assert_eq!(rig.ans.base(register), Some(before));
    }
    for (register, address) in [(ASQ, SQ), (ACQ, LARGE_CQ)] {
        rig.ans
            .write(
                register,
                Width::Double,
                address,
                &mut Access::new(0, 0, &mut rig.latch),
            )
            .unwrap();
    }
    rig.write(AQA, 1 | (1 << 16)).unwrap();
    rig.write(CC, u64::from(ENTRY_SIZES | ENABLE)).unwrap();
    assert!(rig.write(ADMIN_SUBMIT, 2).is_err());
    assert!(!rig.ans.valid[2]);
    assert!(rig.ans.pending.is_empty());
}
#[test]
fn nvmmu_data_pages_are_4k_with_16k_controller_queue_alignment() {
    let mut reference = large(253);
    large_command(&mut reference, 252, 6, LARGE_DATA, [1, 0, 0, 0, 0, 0]);
    reference.write(ADMIN_SUBMIT, 252).unwrap();
    reference.run().unwrap();
    let expected = reference.bytes(LARGE_DATA, 4096);
    let mut rig = large_mps(253, 2);
    let first = LARGE_DATA + 4092;
    let second = LARGE_DATA + 0x8000;
    assert!(rig.ram.write(LARGE_DATA + 4096, &vec![0x6a; 4092]));
    large_command(&mut rig, 252, 6, first, [1, 0, 0, 0, 0, 0]);
    assert!(
        rig.ram
            .write(LARGE_TCB + 252 * 128 + 32, &second.to_le_bytes())
    );
    rig.write(ADMIN_SUBMIT, 252).unwrap();
    rig.run().unwrap();
    let mut actual = rig.bytes(first, 4);
    actual.extend(rig.bytes(second, 4092));
    assert_eq!(actual, expected);
    assert_eq!(rig.bytes(LARGE_DATA + 4096, 4092), vec![0x6a; 4092]);
    assert_eq!(rig.ans.page(), 16384);
    assert_eq!(rig.ans.queues[0].pending, 1);
}

/// A completion's command ID and status at slot `slot` of the I/O CQ.
fn io_completion(rig: &mut Rig, slot: u64) -> (u16, u16) {
    let entry = rig.bytes(IOCQ + slot * 16, 16);
    (
        u16::from_le_bytes([entry[12], entry[13]]),
        u16::from_le_bytes([entry[14], entry[15]]) >> 1,
    )
}

#[test]
fn a_deferred_read_completes_after_its_copy_and_holds_later_completions_behind_it() {
    let mut rig = Rig::new();
    rig.ans.defer_copies();
    rig.io_queues();
    let pattern: Vec<u8> = (0..4096).map(|i| (i * 13) as u8).collect();
    assert!(rig.ram.write(DATA, &pattern));
    rig.submit(1, 2, 1, 1, DATA, [3, 0, 0, 0, 0, 0]).unwrap();
    assert!(
        rig.ans.take_copies().next().is_none(),
        "a write copies in place"
    );
    rig.retire(1, 2);
    assert!(rig.ram.write(DATA, &[0; 4096]));
    rig.submit(1, 3, 2, 1, DATA, [3, 0, 0, 0, 0, 0]).unwrap();
    // The read executed; its data and completion wait for the copy.
    let copies: Vec<ReadCopy> = rig.ans.take_copies().collect();
    assert_eq!(copies.len(), 1);
    assert_eq!(rig.bytes(DATA, 4096), vec![0; 4096]);
    assert_eq!(rig.ans.queues[1].pending, 0);
    // A flush after it completes behind it, never ahead.
    rig.submit(1, 4, 0, 1, 0, [0; 6]).unwrap();
    assert_eq!(rig.ans.queues[1].pending, 0);
    assert!(rig.ans.pending(0));
    for copy in copies {
        copy.run(&mut rig.ram);
    }
    assert_eq!(rig.bytes(DATA, 4096), pattern);
    rig.run().unwrap();
    assert_eq!(rig.ans.queues[1].pending, 2);
    assert_eq!(io_completion(&mut rig, 1), (3, 0));
    assert_eq!(io_completion(&mut rig, 2), (4, 0));
}

#[test]
fn deferred_keyed_reads_decrypt_in_the_copy_as_in_place() {
    let engine = crate::inline_key::Engine::default();
    engine.provision(zeroize::Zeroizing::new([9; 32]));
    let area = engine.wrap(&[0x3c; 32]).unwrap();
    let command = |opcode| Command {
        opcode,
        nsid: 1,
        prp: DATA,
        words: [3, 0, 0, 0, 7, 0],
    };
    let plain: Vec<u8> = (0..4096).map(|i| (i * 7) as u8).collect();
    let mut read = Vec::new();
    for defer in [false, true] {
        let mut rig = Rig::new();
        rig.ans.inline = engine.clone();
        if defer {
            rig.ans.defer_copies();
        }
        rig.io_queues();
        assert!(rig.ram.write(DATA, &plain));
        rig.submit_keyed(1, 2, command(1), None, 1, area).unwrap();
        rig.retire(1, 2);
        assert!(rig.ram.write(DATA, &[0; 4096]));
        rig.submit_keyed(1, 3, command(2), None, 1, area).unwrap();
        for copy in rig.ans.take_copies().collect::<Vec<_>>() {
            copy.run(&mut rig.ram);
        }
        rig.run().unwrap();
        assert_eq!(io_completion(&mut rig, 1), (3, 0));
        read.push(rig.bytes(DATA, 4096));
    }
    assert_eq!(read[0], plain);
    assert_eq!(read[1], plain);
}

#[test]
fn a_deferred_read_whose_copy_fails_ends_as_a_failed_copy_in_place() {
    let outside = RAM + 0x10_0000;
    let mut seen = Vec::new();
    for defer in [false, true] {
        let mut rig = Rig::new();
        if defer {
            rig.ans.defer_copies();
        }
        rig.io_queues();
        let submitted = rig.submit(1, 2, 2, 1, outside, [3, 0, 0, 0, 0, 0]);
        let result = if defer {
            assert!(submitted.is_ok());
            assert_eq!(rig.ans.queues[1].pending, 0);
            for copy in rig.ans.take_copies().collect::<Vec<_>>() {
                copy.run(&mut rig.ram);
            }
            rig.run()
        } else {
            submitted
        };
        assert!(result.is_err(), "defer={defer}");
        assert!(rig.ans.completed.is_empty());
        assert!(rig.ans.inflight.is_empty());
        let pending = rig.ans.queues[1].pending;
        let first = (pending != 0).then(|| io_completion(&mut rig, 1));
        if pending != 0 {
            rig.retire(1, 2);
        }
        // The next command runs and completes.
        rig.submit(1, 3, 0, 1, 0, [0; 6]).unwrap();
        seen.push((pending, first, rig.ans.queues[1].pending));
    }
    assert_eq!(seen[0], seen[1]);
}

/// Host micro-benchmark: the time a keyed 4 KiB read holds the devices'
/// lock with its copy in place, against its run under the lock and its
/// copy outside. `cargo test --release -p stela-soc --lib read_lock_costs
/// -- --ignored --nocapture`.
#[test]
#[ignore = "manual: opt-in"]
fn read_lock_costs() {
    let engine = crate::inline_key::Engine::default();
    engine.provision(zeroize::Zeroizing::new([9; 32]));
    let area = engine.wrap(&[0x3c; 32]).unwrap();
    let command = |opcode| Command {
        opcode,
        nsid: 1,
        prp: DATA,
        words: [3, 0, 0, 0, 7, 0],
    };
    const ROUNDS: u32 = 2000;
    let mut times = [0u128; 3];
    for defer in [false, true] {
        let mut rig = Rig::new();
        rig.ans.inline = engine.clone();
        if defer {
            rig.ans.defer_copies();
        }
        rig.io_queues();
        rig.submit_keyed(1, 2, command(1), None, 1, area).unwrap();
        rig.retire(1, 2);
        for round in 0..ROUNDS {
            let tag = 3 + (round % 2) as u16;
            let started = std::time::Instant::now();
            rig.submit_keyed(1, tag, command(2), None, 1, area).unwrap();
            let under = started.elapsed().as_nanos();
            if defer {
                times[1] += under;
                let copies: Vec<ReadCopy> = rig.ans.take_copies().collect();
                let started = std::time::Instant::now();
                for copy in copies {
                    copy.run(&mut rig.ram);
                }
                times[2] += started.elapsed().as_nanos();
                let started = std::time::Instant::now();
                rig.run().unwrap();
                times[1] += started.elapsed().as_nanos();
            } else {
                times[0] += under;
            }
            rig.retire(1, tag);
        }
    }
    let ns = |nanos: u128| nanos / u128::from(ROUNDS);
    std::println!(
        "keyed 4 KiB read: in place {} ns under the lock; deferred {} ns under it, {} ns copy outside",
        ns(times[0]),
        ns(times[1]),
        ns(times[2]),
    );
}

/// The driver's clean shutdown: I/O queues deleted, then CC.SHN normal
/// until CSTS.SHST reports it complete.
fn shut_down(rig: &mut Rig) {
    for (tag, opcode) in [(2, 0), (3, 4)] {
        rig.submit(0, tag, opcode, 0, 0, [1, 0, 0, 0, 0, 0])
            .unwrap();
        rig.retire(0, tag);
    }
    rig.write(CC, u64::from(ENTRY_SIZES | ENABLE | 1 << 14))
        .unwrap();
    rig.run().unwrap();
    assert_eq!(rig.read(CSTS), u64::from(READY | SHUTDOWN_COMPLETE));
}

#[test]
fn a_clean_shutdown_powers_off_and_a_power_on_comes_up_as_at_boot() {
    for handed_off in [false, true] {
        let mut rig = Rig::new();
        rig.ans.boot_linear = handed_off;
        rig.io_queues();
        shut_down(&mut rig);
        let live = rig.ans.lifecycle;
        let off = LifecycleState {
            powered: false,
            clocked: false,
            ..live
        };
        rig.ans.check_lifecycle(off).unwrap();
        rig.ans
            .set_lifecycle(off, &mut Access::new(0, 0, &mut rig.latch));
        // The off state reads back: the controller does not answer.
        assert_eq!(rig.ans.lifecycle, off);
        assert!(
            rig.ans
                .read(CSTS, Width::Word, &mut Access::new(0, 0, &mut rig.latch))
                .is_err()
        );
        assert!(!rig.latch.is_asserted(Wire::Aic(IRQ)));
        // Power returns to a controller as the boot chain hands it over:
        // nothing the driver programmed survives.
        rig.ans.check_lifecycle(live).unwrap();
        rig.ans
            .set_lifecycle(live, &mut Access::new(0, 0, &mut rig.latch));
        assert_eq!(rig.read(CSTS), 0);
        assert_eq!(rig.read(CC), 0);
        assert_eq!(rig.read(LINEAR), u64::from(handed_off));
        assert_eq!(rig.read(AQA), 0);
        for queue in rig.ans.queues {
            assert!(!queue.sq_enabled && !queue.cq_enabled && queue.sq == 0);
        }
        // The driver's normal bring-up then works as at boot.
        rig.enable();
        assert_eq!(rig.read(CSTS), u64::from(READY));
        rig.submit(0, 0, 6, 0, DATA, [1, 0, 0, 0, 0, 0]).unwrap();
        assert_eq!(rig.bytes(CQ + 14, 2), [1, 0]);
        rig.retire(0, 0);
        rig.io_queues();
        assert!(rig.ans.queues[1].sq_enabled && rig.ans.queues[1].cq_enabled);
    }
}

#[test]
fn power_loss_with_work_outstanding_or_before_the_shutdown_completes_is_refused() {
    let mut rig = Rig::new();
    let live = rig.ans.lifecycle;
    let off = LifecycleState {
        powered: false,
        clocked: false,
        ..live
    };
    // No shutdown notification at all.
    let refusal = rig.ans.check_lifecycle(off).unwrap_err();
    assert!(
        refusal
            .to_string()
            .contains("shutdown notification completed")
    );
    // A notification that has not completed: an accepted command waits.
    rig.write(ADMIN_SUBMIT, 1).unwrap();
    rig.write(CC, u64::from(ENTRY_SIZES | ENABLE | 1 << 14))
        .unwrap();
    let refusal = rig.ans.check_lifecycle(off).unwrap_err();
    assert!(
        refusal
            .to_string()
            .contains("shutdown notification completed")
    );
    assert_eq!(rig.ans.pending.len(), 1);
    // A completed shutdown with a command still owned.
    rig.ans.status = READY | SHUTDOWN_COMPLETE;
    rig.ans.shutdown = false;
    let refusal = rig.ans.check_lifecycle(off).unwrap_err();
    assert!(refusal.to_string().contains("outstanding"));
    assert_eq!(rig.ans.pending.len(), 1);
    // A retained power-down and a physical reset keep their own answers.
    rig.ans
        .check_lifecycle(LifecycleState {
            retention: true,
            ..off
        })
        .unwrap();
    assert!(
        rig.ans
            .check_lifecycle(LifecycleState {
                reset: true,
                ..live
            })
            .unwrap_err()
            .to_string()
            .contains("physical reset")
    );
}
