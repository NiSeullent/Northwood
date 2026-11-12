// SPDX-License-Identifier: GPL-2.0-or-later

use super::*;
use alloc::{collections::VecDeque, rc::Rc, vec::Vec};
use core::cell::RefCell;
use stela_driver_base::test_clock;

const CLOCK: Clock = Clock {
    ticks_per_microsecond: 100,
};
const SET_FALLBACK: u32 = 0x8000_8000;
const CLEAR_FALLBACK: u32 = 0x8000_0000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Event {
    Read(usize),
    Write(usize, u32),
}

#[derive(Default)]
struct Script {
    reads: VecDeque<Result<u32, owned::Error>>,
    fallback: u32,
    read_us: u64,
    read_delays: VecDeque<u64>,
    writes: usize,
    deny_write: Option<usize>,
    events: Vec<Event>,
    fallback_set_at: Option<u64>,
    fallback_waits: Vec<u64>,
    drops: usize,
}

struct Model(Rc<RefCell<Script>>);

impl Registers for Model {
    fn read32(&mut self, offset: usize) -> Result<u32, owned::Error> {
        let mut script = self.0.borrow_mut();
        script.events.push(Event::Read(offset));
        if let Some(set_at) = script.fallback_set_at.take() {
            script.fallback_waits.push(CLOCK.now_us() - set_at);
        }
        let delay = script.read_delays.pop_front().unwrap_or(script.read_us);
        if delay != 0 {
            test_clock::advance(CLOCK, delay);
        }
        script.reads.pop_front().unwrap_or(Ok(script.fallback))
    }

    fn write32(&mut self, offset: usize, value: u32) -> Result<(), owned::Error> {
        let mut script = self.0.borrow_mut();
        script.events.push(Event::Write(offset, value));
        script.writes += 1;
        if script.deny_write == Some(script.writes) {
            return Err(owned::Error::Unavailable);
        }
        if value == SET_FALLBACK {
            script.fallback_set_at = Some(CLOCK.now_us());
        }
        Ok(())
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        self.0.borrow_mut().drops += 1;
    }
}

fn fixture(reads: &[u32], fallback: u32) -> (Model, Rc<RefCell<Script>>) {
    let script = Rc::new(RefCell::new(Script {
        reads: reads.iter().copied().map(Ok).collect(),
        fallback,
        ..Script::default()
    }));
    (Model(script.clone()), script)
}

fn start(model: Model, domain: ForcewakeDomain) -> Result<Awake<Model>, Refusal<Model>> {
    // SAFETY: exclusively owned synchronous synthetic registers on this
    // thread's model clock. No physical MMIO, firmware, or other actor.
    unsafe { acquire(model, CLOCK, domain) }
}

#[test]
fn all_three_domains_wait_before_request_and_return_same_owner_after_release() {
    let _time = test_clock::virtualize();
    for domain in [
        ForcewakeDomain::Render,
        ForcewakeDomain::Gt,
        ForcewakeDomain::Media,
    ] {
        // Bit 1 belongs to the other request class. It stays set throughout.
        let (model, script) = fixture(&[2, 2, 3, 3, 2], 2);
        let awake = start(model, domain).unwrap_or_else(|refusal| panic!("{:?}", refusal.error));
        assert_eq!(script.borrow().drops, 0);
        let released = awake
            .release()
            .unwrap_or_else(|refusal| panic!("{:?}", refusal.error));
        assert!(Rc::ptr_eq(&released.0, &script));
        let pair = domain.registers();
        assert_eq!(
            script.borrow().events,
            [
                Event::Read(pair.acknowledge),
                Event::Write(pair.request, 0x0001_0001),
                Event::Read(pair.acknowledge),
                Event::Read(pair.acknowledge),
                Event::Write(pair.request, 0x0001_0000),
                Event::Read(pair.acknowledge),
                Event::Read(pair.acknowledge),
            ]
        );
        drop(released);
        assert_eq!(script.borrow().drops, 1);
    }
}

#[test]
fn glk_collision_pulses_use_the_same_domain_before_or_after_kernel_request() {
    let _time = test_clock::virtualize();
    for domain in [
        ForcewakeDomain::Render,
        ForcewakeDomain::Gt,
        ForcewakeDomain::Media,
    ] {
        let pair = domain.registers();
        let ack = Event::Read(pair.acknowledge);
        let fallback_on = Event::Write(pair.request, SET_FALLBACK);
        let fallback_off = Event::Write(pair.request, CLEAR_FALLBACK);
        let kernel_on = Event::Write(pair.request, 0x0001_0001);
        let kernel_off = Event::Write(pair.request, 0x0001_0000);
        for (reads, read_delays, expected_events) in [
            (
                [KERNEL, KERNEL, FALLBACK, FALLBACK, 0, 0, KERNEL, 0],
                [ACK_TIMEOUT_US, 0],
                [
                    ack,
                    ack,
                    fallback_on,
                    ack,
                    ack,
                    fallback_off,
                    ack,
                    ack,
                    kernel_on,
                    ack,
                    kernel_off,
                    ack,
                ],
            ),
            (
                [
                    0,
                    0,
                    0,
                    FALLBACK | KERNEL,
                    FALLBACK | KERNEL,
                    KERNEL,
                    KERNEL,
                    0,
                ],
                [0, ACK_TIMEOUT_US],
                [
                    ack,
                    kernel_on,
                    ack,
                    ack,
                    fallback_on,
                    ack,
                    ack,
                    fallback_off,
                    ack,
                    ack,
                    kernel_off,
                    ack,
                ],
            ),
        ] {
            let (model, script) = fixture(&reads, 0);
            script.borrow_mut().read_delays = read_delays.into();
            let awake =
                start(model, domain).unwrap_or_else(|refusal| panic!("{:?}", refusal.error));
            let owner = awake
                .release()
                .unwrap_or_else(|refusal| panic!("{:?}", refusal.error));
            let observed = script.borrow();
            assert_eq!(observed.events, expected_events);
            let [waited] = observed.fallback_waits.as_slice() else {
                panic!("one collision pulse");
            };
            assert!((10..18).contains(waited), "waited {waited} µs");
            assert!(Rc::ptr_eq(&owner.0, &script));
        }
    }
}

#[test]
fn ten_acknowledged_pulses_are_the_retry_limit_with_increasing_delays() {
    let _time = test_clock::virtualize();
    let domain = ForcewakeDomain::Gt;
    let pair = domain.registers();
    for delayed_kernel_ack in [false, true] {
        let mut reads = alloc::vec![0, 0];
        for _ in 0..10 {
            reads.extend([0, FALLBACK, FALLBACK, 0]);
        }
        if delayed_kernel_ack {
            reads.extend([KERNEL, 0]);
        }
        let (model, script) = fixture(&reads, 0);
        {
            let mut script = script.borrow_mut();
            script.read_us = 10_000;
            script.read_delays = alloc::vec![0; reads.len()].into();
            script.read_delays[1] = ACK_TIMEOUT_US;
        }
        let outcome = start(model, domain);
        if delayed_kernel_ack {
            let awake = outcome.unwrap_or_else(|refusal| panic!("{:?}", refusal.error));
            let owner = awake
                .release()
                .unwrap_or_else(|refusal| panic!("{:?}", refusal.error));
            assert!(Rc::ptr_eq(&owner.0, &script));
            drop(owner);
        } else {
            let refusal = outcome.err().unwrap();
            assert_eq!(refusal.error, Error::Timeout(Phase::Acquire));
            drop(refusal);
        }
        let observed = script.borrow();
        let writes = observed
            .events
            .iter()
            .copied()
            .filter(|event| matches!(event, Event::Write(..)))
            .collect::<Vec<_>>();
        assert_eq!(writes[0], Event::Write(pair.request, 0x0001_0001));
        assert_eq!(writes.len(), if delayed_kernel_ack { 22 } else { 21 });
        for pulse in writes[1..=20].chunks_exact(2) {
            assert_eq!(
                pulse,
                [
                    Event::Write(pair.request, SET_FALLBACK),
                    Event::Write(pair.request, CLEAR_FALLBACK)
                ]
            );
        }
        assert_eq!(observed.fallback_waits.len(), 10);
        for (index, &elapsed) in observed.fallback_waits.iter().enumerate() {
            let minimum = (index as u64 + 1) * 10;
            assert!(
                (minimum..minimum + 8).contains(&elapsed),
                "pulse {index} waited {elapsed} µs"
            );
        }
        assert_eq!(observed.drops, usize::from(delayed_kernel_ack));
    }
}

#[test]
fn fallback_ack_failure_or_lost_post_clear_observation_retains_the_owner() {
    let _time = test_clock::virtualize();
    for (reads, fallback, error, writes) in [
        (&[0, 0, 0][..], 0, Error::Timeout(Phase::FallbackSet), 2),
        (
            &[0, 0, 0, u32::MAX][..],
            0,
            Error::UnreadableAck(Phase::FallbackSet),
            2,
        ),
        (
            &[0, 0, 0, FALLBACK | KERNEL, FALLBACK | KERNEL][..],
            FALLBACK | KERNEL,
            Error::Timeout(Phase::FallbackRelease),
            3,
        ),
        (
            &[0, 0, 0, FALLBACK | KERNEL, FALLBACK | KERNEL, KERNEL][..],
            FALLBACK | KERNEL,
            Error::Timeout(Phase::Acquire),
            3,
        ),
    ] {
        let (model, script) = fixture(reads, fallback);
        {
            let mut script = script.borrow_mut();
            script.read_us = 10_000;
            script.read_delays = alloc::vec![0; reads.len()].into();
            script.read_delays[1] = ACK_TIMEOUT_US;
        }
        let refusal = start(model, ForcewakeDomain::Media).err().unwrap();
        assert_eq!(refusal.error, error);
        drop(refusal);
        assert_eq!(script.borrow().writes, writes);
        assert_eq!(script.borrow().drops, 0);
    }
}

#[test]
fn uncalibrated_clock_or_stale_ack_cannot_publish_a_request() {
    let _time = test_clock::virtualize();
    let domain = ForcewakeDomain::Render;
    let (model, script) = fixture(&[], 0);
    // SAFETY: isolated synthetic provider. The deliberately invalid
    // calibration is checked before touching even the model.
    let refusal = unsafe {
        acquire(
            model,
            Clock {
                ticks_per_microsecond: 0,
            },
            domain,
        )
    }
    .err()
    .unwrap();
    assert_eq!(refusal.error, Error::Clock(ClockError::Uncalibrated));
    drop(refusal);
    assert!(script.borrow().events.is_empty());
    assert_eq!(script.borrow().drops, 0);

    let (model, script) = fixture(&[], KERNEL | FALLBACK);
    script.borrow_mut().read_us = 10_000;
    let refusal = start(model, domain).err().unwrap();
    assert_eq!(refusal.error, Error::Timeout(Phase::FallbackIdle));
    assert_eq!(script.borrow().writes, 0);
    assert!(!script.borrow().events.is_empty());
}

#[test]
fn timeout_keeps_uncertain_requests_and_rejects_a_late_ack() {
    let _time = test_clock::virtualize();
    let domain = ForcewakeDomain::Media;
    let (model, script) = fixture(&[0, 0, 0, KERNEL], FALLBACK);
    script.borrow_mut().read_us = 20_000;
    let refusal = start(model, domain).err().unwrap();
    assert_eq!(refusal.error, Error::Timeout(Phase::FallbackIdle));
    drop(refusal);
    assert_eq!(script.borrow().writes, 1);
    assert_eq!(script.borrow().drops, 0);

    let (model, script) = fixture(&[0, KERNEL], KERNEL);
    script.borrow_mut().read_us = 10_000;
    let awake = start(model, domain).unwrap_or_else(|refusal| panic!("{:?}", refusal.error));
    let refusal = awake.release().err().unwrap();
    assert_eq!(refusal.error, Error::Timeout(Phase::Release));
    drop(refusal);
    assert_eq!(script.borrow().writes, 2);
    assert_eq!(script.borrow().drops, 0);
}

#[test]
fn all_ones_never_acknowledges_any_phase() {
    let _time = test_clock::virtualize();
    for (reads, phase, writes) in [
        (&[u32::MAX][..], Phase::Idle, 0),
        (&[0, u32::MAX][..], Phase::Acquire, 1),
        (&[0, KERNEL, u32::MAX][..], Phase::Release, 2),
    ] {
        let (model, script) = fixture(reads, 0);
        let refusal = match start(model, ForcewakeDomain::Gt) {
            Ok(awake) => awake.release().err().unwrap(),
            Err(refusal) => refusal,
        };
        assert_eq!(refusal.error, Error::UnreadableAck(phase));
        drop(refusal);
        assert_eq!(script.borrow().writes, writes);
        assert_eq!(script.borrow().drops, 0);
    }
}

#[test]
fn transport_denial_and_abandonment_issue_no_cleanup_write() {
    let _time = test_clock::virtualize();
    let domain = ForcewakeDomain::Render;
    let pair = domain.registers();
    for (write, phase) in [(1, Phase::Acquire), (2, Phase::Release)] {
        let (model, script) = fixture(&[0, KERNEL], 0);
        script.borrow_mut().deny_write = Some(write);
        let refusal = match start(model, domain) {
            Ok(awake) => awake.release().err().unwrap(),
            Err(refusal) => refusal,
        };
        assert_eq!(
            refusal.error,
            Error::Access {
                phase,
                offset: pair.request,
                error: owned::Error::Unavailable,
            }
        );
        drop(refusal);
        assert_eq!(script.borrow().writes, write);
        assert_eq!(script.borrow().drops, 0);
    }
    let (model, script) = fixture(&[0], 0);
    script
        .borrow_mut()
        .reads
        .push_back(Err(owned::Error::Unavailable));
    let refusal = start(model, domain).err().unwrap();
    assert_eq!(
        refusal.error,
        Error::Access {
            phase: Phase::Acquire,
            offset: pair.acknowledge,
            error: owned::Error::Unavailable,
        }
    );
    drop(refusal);
    assert_eq!(script.borrow().writes, 1);
    assert_eq!(script.borrow().drops, 0);

    let (model, script) = fixture(&[0, KERNEL], 0);
    let awake = start(model, domain).unwrap_or_else(|refusal| panic!("{:?}", refusal.error));
    drop(awake);
    assert_eq!(script.borrow().writes, 1);
    assert_eq!(script.borrow().drops, 0);
}
