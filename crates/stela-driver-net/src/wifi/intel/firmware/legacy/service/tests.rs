// SPDX-License-Identifier: GPL-2.0-or-later
use super::*;
use crate::{
    block::BlockError,
    wifi::intel::{
        dma::tests::{Domain, Source},
        firmware::{
            Firmware, SEC_INIT,
            legacy::{
                Cpu, InitError, InitStatus, InitStep, LoadPlan,
                tests::{image, section},
            },
        },
    },
};
use alloc::{rc::Rc, vec::Vec};
use core::{cell::RefCell, num::NonZeroUsize, sync::atomic::Ordering};

const TX_CONFIG: u32 = 0x1e20;
const ENABLE: u32 = 0x8010_0000;
const UNRELATED_IRQ: u32 = 1 << 7;
const UNRELATED_FH: u32 = 1 << 16;

#[derive(Clone, Copy)]
enum Completion {
    Delayed,
    Fault,
    Never,
}

struct State {
    csr: u32,
    fh: u32,
    ict: u32,
    gp: u32,
    mode: u32,
    writes: Vec<(u32, u32)>,
    delays: usize,
    submissions: usize,
    dma_live: bool,
    completion: Completion,
    uncertain_enable: bool,
    stuck_completion: bool,
}

#[derive(Clone)]
struct Io {
    state: Rc<RefCell<State>>,
    domain: Arc<Domain>,
}

impl Io {
    fn new(domain: Arc<Domain>, completion: Completion) -> Self {
        Self {
            state: Rc::new(RefCell::new(State {
                csr: UNRELATED_IRQ,
                fh: UNRELATED_FH,
                ict: 0,
                gp: MAC_CLOCK,
                mode: 0x20,
                writes: Vec::new(),
                delays: 0,
                submissions: 0,
                dma_live: false,
                completion,
                uncertain_enable: false,
                stuck_completion: false,
            })),
            domain,
        }
    }
}

impl Csr for Io {
    fn read32(&mut self, register: u32) -> Result<u32, AccessError> {
        let state = self.state.borrow();
        Ok(match register {
            INT => state.csr,
            FH_INT => state.fh,
            INT_MASK => INT_FH_TX,
            ICT => state.ict,
            GP_CONTROL => state.gp,
            PRPH_RDATA => state.mode,
            _ => panic!("unexpected read {register:#x}"),
        })
    }

    fn write32(&mut self, register: u32, value: u32) -> Result<(), AccessError> {
        let mut state = self.state.borrow_mut();
        state.writes.push((register, value));
        match register {
            INT if !state.stuck_completion => state.csr &= !value,
            FH_INT if !state.stuck_completion => state.fh &= !value,
            GP_CONTROL => state.gp = value,
            PRPH_RADDR | PRPH_WADDR => {
                assert_eq!(value, 0x0300_1ff8);
                assert_ne!(state.gp & MAC_ACCESS, 0);
            }
            PRPH_WDATA => {
                assert!(!state.dma_live, "never change addressing during DMA");
                state.mode = value;
            }
            TX_CONFIG if value == ENABLE => {
                assert!(!state.dma_live);
                assert_ne!(state.gp & MAC_ACCESS, 0);
                assert_eq!(state.csr & INT_FH_TX, 0);
                assert_eq!(state.fh & FH_TX, 0);
                let records = self.domain.records.lock().unwrap();
                assert_eq!(records.live.len(), 1);
                assert_eq!(records.syncs, state.submissions + 1);
                assert_eq!(records.directions.last(), Some(&Direction::ToDevice));
                let destination = state
                    .writes
                    .iter()
                    .rev()
                    .find(|(reg, _)| *reg == 0x19c8)
                    .unwrap()
                    .1;
                assert_eq!(
                    state.mode & EXTENDED_SRAM != 0,
                    (0x40000..0x58000).contains(&destination)
                );
                state.dma_live = true;
                state.submissions += 1;
                if state.uncertain_enable {
                    return Err(AccessError::Mapping);
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn delay_us(&mut self, microseconds: u32) -> Result<(), AccessError> {
        let mut state = self.state.borrow_mut();
        if microseconds == COMPLETION_POLL_US {
            assert_eq!(state.gp & MAC_ACCESS, 0);
            assert_eq!(self.domain.records.lock().unwrap().live.len(), 1);
            state.delays += 1;
            match state.completion {
                Completion::Delayed => {
                    state.csr |= INT_FH_TX;
                    state.fh |= 1;
                    state.dma_live = false;
                }
                Completion::Fault => {
                    state.csr |= INT_FH_TX | INT_HW_ERR;
                    state.fh |= 1 | FH_ERROR;
                }
                Completion::Never => {}
            }
        }
        Ok(())
    }
}

// SAFETY: this synchronous model owns its register state and serializes all
// CSR accesses on one thread. The tests explicitly stop modeled DMA.
unsafe impl BootIo for Io {
    fn write8(&mut self, _: u32, _: u8) -> Result<(), AccessError> {
        panic!("byte register access is outside the service protocol")
    }

    fn reset_function(&mut self) -> Result<bool, BlockError> {
        panic!("a transport failure never triggers an unverified reset")
    }
}

fn transfer(destination: u32) -> Transfer<'static> {
    Transfer {
        destination,
        data: b"firmware",
    }
}

#[test]
fn fresh_completion_drains_each_pin_and_restores_extended_mode() {
    let source = Source::default();
    let domain = Arc::new(Domain::default());
    let io = Io::new(domain.clone(), Completion::Delayed);
    let state = io.state.clone();
    state.borrow_mut().csr |= INT_FH_TX;
    state.borrow_mut().fh |= 1;
    // SAFETY: the model is quiescent and has a direct, dedicated CSR route.
    let mut channel = unsafe { ServiceChannel::new(io, &source, domain.clone()) };
    channel.transfer(transfer(0x41000)).unwrap();
    assert_eq!(state.borrow().mode, 0x20);
    state.borrow_mut().mode |= EXTENDED_SRAM;
    channel.transfer(transfer(0x410000)).unwrap();
    assert!(channel.finish().is_ok());
    let state = state.borrow();
    assert_eq!(state.submissions, 2);
    assert_eq!(state.delays, 2, "stale completion never finishes a new DMA");
    assert_eq!(state.mode, 0x21);
    assert_eq!(state.csr, UNRELATED_IRQ);
    assert_eq!(state.fh, UNRELATED_FH);
    let regs = domain.records.lock().unwrap();
    assert_eq!(regs.unmaps, 2);
    assert!(regs.live.is_empty());
    assert_eq!(regs.copied, [b"firmware".to_vec(), b"firmware".to_vec()]);
    assert_eq!(source.released.load(Ordering::Relaxed), 2);
}

#[test]
fn timeout_retains_the_exact_mapping_until_proven_stop() {
    let source = Source::default();
    let domain = Arc::new(Domain::default());
    let io = Io::new(domain.clone(), Completion::Never);
    let state = io.state.clone();
    // SAFETY: the model is quiescent and has a direct, dedicated CSR route.
    let mut channel = unsafe { ServiceChannel::new(io, &source, domain.clone()) };
    assert_eq!(channel.transfer(transfer(0x41000)), Err(Error::Timeout));
    assert_eq!(state.borrow().delays, COMPLETION_POLLS);
    assert_eq!(state.borrow().mode, 0x21);
    assert_eq!(channel.transfer(transfer(0x41000)), Err(Error::Unavailable));
    assert!(matches!(channel.finish(), Err(Error::Unavailable)));
    assert_eq!(source.released.load(Ordering::Relaxed), 0);
    assert_eq!(domain.records.lock().unwrap().live.len(), 1);
    state.borrow_mut().dma_live = false;
    // SAFETY: the synchronous model has now stopped the only submitted DMA.
    assert!(unsafe { channel.reclaim_after_stop() }.is_ok());
    assert_eq!(source.released.load(Ordering::Relaxed), 1);
    assert!(domain.records.lock().unwrap().live.is_empty());
}

#[test]
fn init_progress_follows_dma_completion_for_each_cpu() {
    let mut bytes = image();
    section(&mut bytes, SEC_INIT, super::super::CPU_SEPARATOR, &[]);
    section(&mut bytes, SEC_INIT, 0xc0080000, b"abc");
    let firmware = Firmware::parse(&bytes).unwrap();
    let plan = LoadPlan::from_firmware(&firmware).unwrap();
    let source = Source::default();
    let domain = Arc::new(Domain::default());
    let io = Io::new(domain.clone(), Completion::Delayed);
    let state = io.state.clone();
    // SAFETY: this model uses the dedicated FH-TX route for each transfer.
    // The regular interrupt handoff is tracked in the test, outside CSR I/O.
    let mut channel = unsafe { ServiceChannel::new(io, &source, domain.clone()) };
    let mut load_status = 0;
    let mut interrupt_ready = None;
    let mut completed = Vec::new();
    plan.upload_init(
        NonZeroUsize::new(2).unwrap(),
        |action| -> Result<(), Error> {
            match action {
                InitStep::CheckFresh => assert_eq!(load_status, 0),
                InitStep::Transfer { transfer, .. } => channel.transfer(transfer)?,
                InitStep::SectionLoaded { status, .. } => {
                    assert!(!state.borrow().dma_live);
                    assert!(domain.records.lock().unwrap().live.is_empty());
                    assert_eq!(load_status, status.before);
                    load_status = status.after;
                }
                InitStep::EnableInterrupts(cpu) => interrupt_ready = Some(cpu),
                InitStep::CpuComplete { cpu, status } => {
                    assert_eq!(interrupt_ready.take(), Some(cpu));
                    assert_eq!(load_status, status.before);
                    load_status = status.after;
                    completed.push((cpu, status.after));
                }
            }
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(completed, [(Cpu::First, 0xffff), (Cpu::Second, u32::MAX)]);
    assert_eq!(state.borrow().submissions, 3);
    assert_eq!(source.released.load(Ordering::Relaxed), 3);
    assert!(channel.finish().is_ok());
}

#[test]
fn failed_init_chunk_suppresses_progress_and_leaves_its_pin_owned() {
    let mut bytes = image();
    section(&mut bytes, SEC_INIT, 0x41000, b"abcd");
    let firmware = Firmware::parse(&bytes).unwrap();
    let plan = LoadPlan::from_firmware(&firmware).unwrap();
    let source = Source::default();
    let domain = Arc::new(Domain::default());
    let io = Io::new(domain.clone(), Completion::Delayed);
    let state = io.state.clone();
    // SAFETY: the model is quiescent and has a direct, dedicated CSR route.
    let mut channel = unsafe { ServiceChannel::new(io, &source, domain.clone()) };
    let mut acknowledged = Vec::new();
    let result = plan.upload_init(NonZeroUsize::new(2).unwrap(), |action| {
        if let InitStep::Transfer { transfer, .. } = action {
            if state.borrow().submissions == 2 {
                state.borrow_mut().completion = Completion::Fault;
            }
            channel.transfer(transfer)?;
        } else {
            acknowledged.push(action);
        }
        Ok(())
    });
    assert!(matches!(
        result,
        Err(InitError::Step(Error::DeviceFault { .. }))
    ));
    assert_eq!(
        acknowledged,
        [
            InitStep::CheckFresh,
            InitStep::SectionLoaded {
                cpu: Cpu::First,
                section: 0,
                status: InitStatus {
                    before: 0,
                    after: 1
                },
            },
        ]
    );
    assert_eq!(state.borrow().submissions, 3);
    assert_eq!(source.released.load(Ordering::Relaxed), 2);
    assert_eq!(domain.records.lock().unwrap().live.len(), 1);
    state.borrow_mut().dma_live = false;
    // SAFETY: the synchronous model has now stopped the only in-flight DMA.
    assert!(unsafe { channel.reclaim_after_stop() }.is_ok());
}

#[test]
fn uncertain_enable_quarantines_both_the_pin_and_the_requester() {
    if crate::test_process::isolate_model() {
        return;
    }
    let source = Source::default();
    let domain = Arc::new(Domain::default());
    let io = Io::new(domain.clone(), Completion::Never);
    let state = io.state.clone();
    state.borrow_mut().uncertain_enable = true;
    // SAFETY: the model is quiescent and has a direct, dedicated CSR route.
    let mut channel = unsafe { ServiceChannel::new(io, &source, domain.clone()) };
    assert_eq!(
        channel.transfer(transfer(0x41000)),
        Err(Error::Access(AccessError::Mapping))
    );
    drop(channel);
    assert!(state.borrow().dma_live);
    assert_eq!(domain.records.lock().unwrap().live.len(), 1);
    assert_eq!(source.released.load(Ordering::Relaxed), 0);
    assert_eq!(
        Rc::strong_count(&state),
        2,
        "the requester owner was retained"
    );
}

#[test]
fn fault_precedes_completion_and_failed_invalidation_can_retry() {
    let source = Source::default();
    let domain = Arc::new(Domain::default());
    domain.records.lock().unwrap().unmap_fail = Some(0);
    let io = Io::new(domain.clone(), Completion::Fault);
    let state = io.state.clone();
    // SAFETY: the model is quiescent and has a direct, dedicated CSR route.
    let mut channel = unsafe { ServiceChannel::new(io, &source, domain.clone()) };
    assert!(matches!(
        channel.transfer(transfer(0x41000)),
        Err(Error::DeviceFault { .. })
    ));
    assert_ne!(state.borrow().csr & INT_FH_TX, 0);
    assert_eq!(domain.records.lock().unwrap().unmaps, 0);
    state.borrow_mut().dma_live = false;
    // SAFETY: the synchronous model has now stopped the only submitted DMA.
    let result = unsafe { channel.reclaim_after_stop() };
    assert!(matches!(
        result,
        Err(Error::Dma(dma::Error::Dma(
            crate::bus::dma::mapping::DmaError::InvalidationFailed
        )))
    ));
    assert_eq!(source.released.load(Ordering::Relaxed), 0);
    assert_eq!(domain.records.lock().unwrap().live.len(), 1);
    // SAFETY: the device is still stopped and the failed unmap kept its lease.
    assert!(unsafe { channel.reclaim_after_stop() }.is_ok());
    assert_eq!(source.released.load(Ordering::Relaxed), 1);
}

#[test]
fn an_uncleared_cause_or_wrong_route_never_publishes_dma() {
    for error in [Error::CompletionNotCleared, Error::InterruptMode] {
        let source = Source::default();
        let domain = Arc::new(Domain::default());
        let io = Io::new(domain.clone(), Completion::Never);
        let state = io.state.clone();
        state.borrow_mut().csr |= INT_FH_TX;
        state.borrow_mut().stuck_completion = true;
        if error == Error::InterruptMode {
            state.borrow_mut().ict = ICT_ENABLED;
        }
        // SAFETY: the model is quiescent and keeps ownership. Its wrong-route
        // case deliberately simulates an inconsistent ICT mode readback.
        let mut channel = unsafe { ServiceChannel::new(io, &source, domain.clone()) };
        assert_eq!(channel.transfer(transfer(0x41000)), Err(error));
        assert_eq!(state.borrow().submissions, 0);
        assert_eq!(source.released.load(Ordering::Relaxed), 1);
        assert!(domain.records.lock().unwrap().live.is_empty());
        // SAFETY: no modeled DMA was ever submitted.
        assert!(unsafe { channel.reclaim_after_stop() }.is_ok());
    }
}
