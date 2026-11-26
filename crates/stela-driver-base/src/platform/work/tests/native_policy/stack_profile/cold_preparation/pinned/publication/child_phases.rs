//! Actual guarded native transitions; no original C task graph is modeled ready.
use super::*;
use crate::platform::work::DormantGeneralStack;
struct Clock {
    cpu: Arc<dyn Cpu>,
    start: std::time::Instant,
}
// SAFETY: genuine hosted elapsed nanoseconds on the same owned fixture CPU;
// bounded read has no callback or permission/flag mutation.
unsafe impl service::Clock for Clock {
    fn cpu(&self) -> &Arc<dyn Cpu> {
        &self.cpu
    }
    fn now_ns(&self) -> Result<u64, WorkError> {
        self.cpu.check_cpu()?;
        u64::try_from(self.start.elapsed().as_nanos()).map_err(|_| WorkError::Exhausted)
    }
}
struct Idle {
    clock: Arc<dyn service::Clock>,
    thread: std::thread::Thread,
}
// SAFETY: actual same-thread park/unpark event provider with the same monotonic
// clock. Real events are consumed by park, no flags or callback are fabricated.
unsafe impl service::Idle for Idle {
    fn clock(&self) -> &Arc<dyn service::Clock> {
        &self.clock
    }
    fn notify(&self) -> Result<(), WorkError> {
        self.thread.unpark();
        Ok(())
    }
    fn wait_until_ns(&self, deadline: Option<u64>) -> Result<(), WorkError> {
        if let Some(deadline) = deadline {
            if let Some(delay) = deadline.checked_sub(self.clock.now_ns()?) {
                std::thread::park_timeout(std::time::Duration::from_nanos(delay));
            }
        } else {
            std::thread::park();
        }
        Ok(())
    }
}
struct TerminalBinding {
    base: Binding,
    phase: AtomicUsize,
}
// SAFETY: same-host-thread immutable real stack identity plus bounded atomics;
// no original C/provider/allocator, owner drop, reentry or switch occurs in hooks.
unsafe impl SwitchBinding for TerminalBinding {
    unsafe fn attach(&self, v: &TaskContext) -> Result<(), WorkError> {
        // SAFETY: exact current fixture stack; delegate original real-RSP check.
        unsafe { self.base.attach(v) }
    }
    unsafe fn still_blocked(&self, v: &TaskContext) -> Result<bool, WorkError> {
        // SAFETY: exact callback-scoped view on actual retained current stack.
        unsafe { self.base.still_blocked(v) }
    }
    unsafe fn leave(&self, v: &TaskContext, r: LeaveReason) -> Result<(), WorkError> {
        // SAFETY: exact callback-scoped view and original outgoing stack.
        unsafe { self.base.leave(v, r) }
    }
    unsafe fn arrive(&self, v: &TaskContext, n: bool) -> Result<(), WorkError> {
        // SAFETY: exact callback-scoped destination view on real kept stack.
        unsafe { self.base.arrive(v, n) }
    }
    unsafe fn complete(&self, _: &TaskContext) -> Result<(), WorkError> {
        panic!("terminal task used ordinary callback completion on dying stack")
    }
    unsafe fn terminal_prepare(&self, v: &TaskContext) -> Result<(), WorkError> {
        self.base.verify(v, true);
        assert_eq!(self.phase.swap(1, Ordering::AcqRel), 0);
        Ok(())
    }
    unsafe fn terminal_successor(&self, v: &TaskContext) -> Result<(), WorkError> {
        self.base.verify(v, false);
        assert_eq!(self.phase.swap(2, Ordering::AcqRel), 1);
        Ok(())
    }
    unsafe fn detach(&self, v: &TaskContext) -> Result<(), WorkError> {
        self.base.verify(v, false);
        assert_eq!(self.phase.swap(3, Ordering::AcqRel), 2);
        Ok(())
    }
}
struct ChildPhases {
    base: Arc<PinScene>,
    domain: Mutex<Option<Arc<service::Domain<64>>>>,
    dormant: UnsafeCell<Option<DormantGeneralStack<'static, 64>>>,
    binding: Mutex<Option<(Arc<TerminalBinding>, WakeHandle<64>)>>,
    calls: AtomicUsize,
}
// SAFETY: passive retention only; every affine cell is touched solely by the
// same real Runtime fixture CPU, never remotely and never as native authority.
unsafe impl Send for ChildPhases {}
// SAFETY: same single-host-thread stack-switch fixture, no concurrent cell use.
unsafe impl Sync for ChildPhases {}
unsafe extern "C" fn service_done(p: *mut c_void) {
    // SAFETY: exact actual service frame and Queue-retained fixture/code.
    let f = unsafe { &*p.cast::<ChildPhases>() };
    let r = f.base.runtime();
    let binding = Arc::new(Binding {
        id: AtomicU64::new(0),
        blocked: AtomicBool::new(false),
    });
    let wake = r.bind_current(binding.clone()).unwrap();
    *f.base.binding.lock().unwrap() = Some((binding, wake));
    let domain = f.domain.lock().unwrap().as_ref().unwrap().clone();
    domain.bind_current(r).unwrap();
}
unsafe extern "C" fn child(p: *mut c_void) {
    // SAFETY: exact fixture and code retained by actual Queue before publication.
    let f = unsafe { &*p.cast::<ChildPhases>() };
    let r = f.base.runtime();
    assert_eq!(f.base.entered.load(Ordering::Acquire), 0);
    {
        let binding = Arc::new(TerminalBinding {
            base: Binding {
                id: AtomicU64::new(0),
                blocked: AtomicBool::new(false),
            },
            phase: AtomicUsize::new(0),
        });
        let wake = r.bind_current(binding.clone()).unwrap();
        *f.binding.lock().unwrap() = Some((binding, wake));
    }
    f.calls.fetch_add(1, Ordering::AcqRel);
    // SAFETY: fixture retains ALL owners outside skipped frames. This callback
    // holds only nonowning references now; no guard/lock/Arc/borrowed process
    // permit survives. Genuine C terminal semantics remain a separate producer.
    unsafe { r.terminate_current() }
}
unsafe extern "C" fn prepare(p: *mut c_void) {
    // SAFETY: kept fixture on actual creator stack through exact native join.
    let f = unsafe { &*p.cast::<ChildPhases>() };
    // SAFETY: fixture retains exact Executor and all RAM/code until successor.
    let r: Pin<&'static Runtime<64>> = unsafe {
        Pin::new_unchecked(&*(f.base.runtime.load(Ordering::Acquire) as *const Runtime<64>))
    };
    let authority = f.base.authority();
    let owner = r
        .reserve_general(&authority, kernel())
        .unwrap()
        .into_owned()
        .unwrap();
    let prepared = take_prepared(owner, &authority);
    let pin = prepared
        .pin(&authority)
        .unwrap_or_else(|_| panic!("pin refused"));
    let mut ready = pin
        .reconstruct(&authority)
        .unwrap_or_else(|_| panic!("frame refused"));
    let rank = scheduling::Snapshot {
        base: scheduling::BasePolicy::Normal,
        effective: 120,
    };
    ready.reserve_child_rank(&authority, rank).unwrap();
    assert_eq!(
        ready.reserve_child_rank(
            &authority,
            scheduling::Snapshot {
                effective: 2,
                ..rank
            }
        ),
        Err(WorkError::InvalidTicket)
    );
    assert_eq!(f.calls.load(Ordering::Acquire), 0);
    // SAFETY: exact kept fixture graph; native mechanics only, no original C
    // child/source/PID is manufactured or claimed by accepting this native job.
    let dormant = unsafe { ready.publish_dormant(&authority, rank, child, p) }
        .unwrap_or_else(|_| panic!("dormant publication refused"));
    assert_eq!(f.calls.load(Ordering::Acquire), 0);
    // SAFETY: same actual fixture CPU, no peer borrows this retained cell.
    unsafe {
        *f.dormant.get() = Some(dormant);
    }
}
#[test]
fn dormant_job_waits_for_actual_activation_then_terminal_joins_on_real_successor() {
    let dropped = Arc::new(AtomicUsize::new(0));
    let cpu = Arc::new(ModelCpu {
        model: test_support::CpuModel::new(),
        dropped: dropped.clone(),
        mode: Mode::Bytes,
        watch: AtomicU64::new(0),
        watch_state: AtomicU64::new(0),
        restored: AtomicBool::new(false),
    });
    let base = Arc::new(PinScene {
        runtime: AtomicU64::new(0),
        cpu: cpu.clone(),
        case: Case::Good,
        active: AtomicBool::new(true),
        entered: AtomicUsize::new(0),
        exports: AtomicUsize::new(0),
        pins_dropped: Arc::new(AtomicUsize::new(0)),
        marked: AtomicUsize::new(0),
        binding: Mutex::new(None),
        held_view: Mutex::new(None),
    });
    let f = Arc::new(ChildPhases {
        base: base.clone(),
        domain: Mutex::new(None),
        dormant: UnsafeCell::new(None),
        binding: Mutex::new(None),
        calls: AtomicUsize::new(0),
    });
    // SAFETY: same real host thread, exact code/fixture and guarded mmap owners.
    let q = unsafe { Queue::<64>::new(cpu.clone(), f.clone()) }.unwrap();
    let executor =
        Executor::new_hosted_profile_for_test(q, mappings(&base, &dropped), policy(), profile())
            .unwrap();
    let r = executor.runtime();
    base.runtime
        .store(r.get_ref() as *const Runtime<64> as u64, Ordering::Release);
    let clock: Arc<dyn service::Clock> = Arc::new(Clock {
        cpu: cpu.clone(),
        start: std::time::Instant::now(),
    });
    let idle = Arc::new(Idle {
        clock: clock.clone(),
        thread: std::thread::current(),
    });
    // This fixture's service returns after binding. A periodic tick could
    // otherwise wake its closed continuation while the child is dispatched.
    let domain = service::Domain::tickless(clock, idle).unwrap();
    r.attach_service_domain(domain.clone()).unwrap();
    *f.domain.lock().unwrap() = Some(domain.clone());
    // SAFETY: exact retained fixture/code on its real reserved service stack.
    let service = unsafe { r.start_service(service_done, Arc::as_ptr(&f) as *mut c_void) }.unwrap();
    assert!(r.step().unwrap());
    assert!(r.queue.completed(service).unwrap());
    // SAFETY: actual fixture kept through creator/child/publication joins.
    unsafe {
        r.call_foreground(
            prepare,
            Arc::as_ptr(&f) as *mut c_void,
            cpu.ticks() + 1_000_000_000,
        )
    }
    .unwrap();
    // SAFETY: same CPU after real creator return, no affine peer borrows cell.
    let dormant = unsafe { (*f.dormant.get()).take() }.unwrap();
    for _ in 0..4 {
        assert!(!r.step().unwrap());
    }
    assert_eq!(f.calls.load(Ordering::Acquire), 0);
    assert!(!r.queue.completed(dormant.assigned().ticket()).unwrap());
    let authority = base.authority();
    base.active.store(false, Ordering::Release);
    // SAFETY: valid retained graph; denied real model scope must retain Dormant.
    let dormant = match unsafe { dormant.activate(&authority) } {
        Err((d, WorkError::Unavailable)) => d,
        _ => panic!("revoked activation accepted"),
    };
    base.active.store(true, Ordering::Release);
    drop(authority);
    assert_eq!(f.calls.load(Ordering::Acquire), 0);
    let authority = base.authority();
    // SAFETY: native mechanics source explicitly reaches its first runnable
    // wake here. No original Linux source/publication readiness is claimed.
    let published = unsafe { dormant.activate(&authority) }.unwrap();
    drop(authority);
    assert_eq!(f.calls.load(Ordering::Acquire), 0);
    r.drain(cpu.ticks() + 1_000_000_000).unwrap();
    assert_eq!(f.calls.load(Ordering::Acquire), 1);
    assert!(r.queue.completed(published.assigned().ticket()).unwrap());
    let (binding, wake) = f.binding.lock().unwrap().take().unwrap();
    assert_eq!(binding.phase.load(Ordering::Acquire), 2);
    assert_eq!(wake.wake(), Err(WorkError::Closed));
    domain.stop_producers().unwrap();
    // SAFETY: actual reserved service callback really bound/returned and its
    // exact job closed above; no original C/event producer exists in fixture.
    unsafe { domain.retire() }.unwrap();
    let (service_binding, service_wake) = base.binding.lock().unwrap().take().unwrap();
    // SAFETY: actual closed service returned physically, source producers stopped.
    drop(unsafe { r.detach_binding(service_wake.context_id()) }.unwrap());
    drop(service_binding);
    drop(service_wake);
    r.queue.close().unwrap();
    // SAFETY: stopped producers, exact real child returned physically. Pin and
    // binding still held: neither native completion nor source-hook metadata
    // may authorize Runtime or Queue retirement or physical stack reuse.
    assert_eq!(unsafe { r.retire() }, Err(WorkError::Pending));
    // SAFETY: independently checked stopped graph, exact live pin remains.
    assert_eq!(unsafe { r.queue.retire() }, Err(WorkError::Pending));
    // SAFETY: no foreign C/raw users; attached binding must still refuse release.
    let published = match unsafe { published.release() } {
        Err((p, WorkError::Busy)) => p,
        _ => panic!("terminal binding bypassed"),
    };
    // SAFETY: real physical successor, fixture producers stopped, no C users.
    drop(unsafe { r.detach_binding(wake.context_id()) }.unwrap());
    assert_eq!(binding.phase.load(Ordering::Acquire), 3);
    drop(binding);
    drop(wake);
    // SAFETY: all actual fixture owners/views/users joined, no C graph exists.
    unsafe { published.release() }.unwrap();
    assert_eq!(base.pins_dropped.load(Ordering::Acquire), 1);
    // SAFETY: exact native graph fully joined and admission closed.
    unsafe { r.retire() }.unwrap();
    drop(executor);
    assert_eq!(dropped.load(Ordering::Acquire), 32);
}
