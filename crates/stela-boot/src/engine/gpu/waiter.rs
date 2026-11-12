//! A Linux task that waits on a Metal device's completion timeline, so its
//! service looks for completed work as soon as some completes.
//!
//! The service only asks the timeline's value, and nouveau arms a fence's
//! interrupt only for a waiter: without one, completed work would wait for
//! the next guest poll or the pump's fallback look. The device exports its
//! timeline as a DRM syncobj; the waiter imports it into a render node file
//! of its own and blocks in `DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT` for the next
//! point, waiting for its submission too. Its own handle keeps the syncobj
//! alive whatever becomes of the device, and each wait ends within
//! [`WAIT_NS`] while work is outstanding, else [`IDLE_WAIT_NS`], so a
//! waiter a newer device or the stop replaced leaves.
//!
//! Each point it sees signaled goes straight to the device's service
//! (`SharedService::completed_through`), which tells the guest at once even
//! while a call, such as a compile, holds the service.
//!
//! Its file is a nouveau client with RM objects (device, subdevice, VA
//! space). The GPU's stop waits for every waiter to close it
//! ([`wait_gone`]): frees that reach nouveau after the stop began are refused,
//! and GSP-RM is then asked to unload with that client still allocated.
use super::{client, gate, look_again};
use alloc::boxed::Box;
use core::{
    ffi::{c_int, c_uint, c_void},
    ptr,
    sync::atomic::{AtomicU32, AtomicU64, Ordering},
};

/// The waiters' generation: a waiter of another one leaves.
static GENERATION: AtomicU64 = AtomicU64::new(0);

/// The generation of the waiter that waits on its timeline now, 0 for none.
static WATCHING: AtomicU64 = AtomicU64::new(0);

/// Waiters started that have not yet closed their file.
static RUNNING: AtomicU32 = AtomicU32::new(0);

/// Counts a waiter as running until it is dropped, after its file closed.
struct Running;

impl Running {
    fn start() -> Self {
        RUNNING.fetch_add(1, Ordering::AcqRel);
        Self
    }
}

impl Drop for Running {
    fn drop(&mut self) {
        RUNNING.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Whether the newest waiter waits on its timeline: every fence of it that
/// signals, in an interrupt or not, ends its wait and has the services look.
pub(super) fn watching() -> bool {
    let generation = GENERATION.load(Ordering::Acquire);
    generation != 0 && WATCHING.load(Ordering::Acquire) == generation
}

/// The longest one wait lasts while the service awaits a completion: a
/// stalled timeline is noticed after it.
const WAIT_NS: u64 = 100_000_000;
/// The longest one wait lasts while the service awaits none. A wait ends
/// as soon as its point signals, whatever its timeout, and waits for the
/// point's submission too, so a longer one loses no completion; an idle
/// desktop then wakes the waiter once a second instead of ten times, each
/// wake about 1 ms of the boot PE. A submission during an idle wait that
/// never signals is noticed after this instead of [`WAIT_NS`].
const IDLE_WAIT_NS: u64 = 1_000_000_000;
/// Failed waits in a row after which a waiter leaves: its syncobj or file
/// is unusable.
const FAILURES: u32 = 8;

const SYNCOBJ_DESTROY: u32 = 0xc008_64c0;
const SYNCOBJ_FD_TO_HANDLE: u32 = 0xc010_64c2;
const SYNCOBJ_TIMELINE_WAIT: u32 = 0xc030_64ca;
const SYNCOBJ_QUERY: u32 = 0xc018_64cb;
/// `DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT`: a point not yet submitted is
/// waited for rather than refused.
const WAIT_FOR_SUBMIT: u32 = 1 << 1;
const ETIME: i64 = 62;
const EINTR: i64 = 4;

/// `struct drm_syncobj_handle`.
#[repr(C)]
struct Handle {
    handle: u32,
    flags: u32,
    fd: i32,
    pad: u32,
}

/// `struct drm_syncobj_timeline_wait`.
#[repr(C)]
struct TimelineWait {
    handles: u64,
    points: u64,
    timeout_nsec: i64,
    count_handles: u32,
    flags: u32,
    first_signaled: u32,
    pad: u32,
    deadline_nsec: u64,
}

/// `struct drm_syncobj_timeline_array`.
#[repr(C)]
struct TimelineArray {
    handles: u64,
    points: u64,
    count_handles: u32,
    flags: u32,
}

/// What a waiter takes: the exported timeline, its generation, and the
/// service and timeline it publishes for.
struct Watch {
    fd: c_int,
    generation: u64,
    service: usize,
    timeline: u64,
    _running: Running,
}

/// Waits on `fd`, the completion timeline `timeline` of service `service`'s
/// device, which the waiter owns; earlier waiters leave. A waiter that
/// cannot start closes `fd`.
pub(super) fn start(fd: c_int, service: usize, timeline: u64) {
    let generation = GENERATION.fetch_add(1, Ordering::AcqRel) + 1;
    let watch = Box::into_raw(Box::new(Watch {
        fd,
        generation,
        service,
        timeline,
        _running: Running::start(),
    }));
    if client::spawn(run, watch.cast()) != 0 {
        // SAFETY: the thread was refused, so the box is still ours.
        let watch = unsafe { Box::from_raw(watch) };
        client::close_descriptor(watch.fd);
    }
}

/// Every waiter leaves within [`IDLE_WAIT_NS`]: the GPUs stop.
pub(super) fn stop() {
    GENERATION.fetch_add(1, Ordering::AcqRel);
}

/// The longest the stop waits for the waiters to close their files: one
/// wait of [`IDLE_WAIT_NS`], its syncobj's destruction and the file's
/// release.
const GONE_NS: u64 = IDLE_WAIT_NS + 400_000_000;
const _: () = assert!(WAIT_NS <= IDLE_WAIT_NS && GONE_NS > IDLE_WAIT_NS);

/// Has every waiter leave and waits, within [`GONE_NS`], until each closed
/// its file, so its nouveau client is freed before the GPU stops; says
/// whether they did. On a Linux task that may sleep.
pub(super) fn wait_gone() -> bool {
    unsafe extern "C" {
        fn msleep(msecs: c_uint);
    }
    stop();
    let deadline = client::now_ns().saturating_add(GONE_NS);
    while RUNNING.load(Ordering::Acquire) != 0 && client::now_ns() < deadline {
        // SAFETY: Linux's sleep, on the caller's task, which may sleep.
        unsafe { msleep(1) };
    }
    RUNNING.load(Ordering::Acquire) == 0
}

unsafe extern "C" fn run(argument: *mut c_void) {
    // SAFETY: `start` gave this thread the box.
    let watch = unsafe { Box::from_raw(argument.cast::<Watch>()) };
    let Some(file) = client::open_node(client::RENDER_MINOR) else {
        client::close_descriptor(watch.fd);
        return;
    };
    let mut import = Handle {
        handle: 0,
        flags: 0,
        fd: watch.fd,
        pad: 0,
    };
    let imported = client::control(
        file,
        SYNCOBJ_FD_TO_HANDLE,
        ptr::from_mut(&mut import).cast(),
    );
    client::close_descriptor(watch.fd);
    if imported == 0 {
        wait(file, import.handle, &watch);
        let mut destroy = [import.handle, 0];
        client::control(file, SYNCOBJ_DESTROY, destroy.as_mut_ptr().cast());
    }
    client::close_descriptor(file);
}

/// Waits for each next point of `handle`, publishing each and having the
/// services look, until another generation starts or waits keep failing.
fn wait(file: c_int, handle: u32, watch: &Watch) {
    let generation = watch.generation;
    let mut next = value(file, handle).unwrap_or(0) + 1;
    let mut failures = 0;
    WATCHING.store(generation, Ordering::Release);
    while GENERATION.load(Ordering::Acquire) == generation && failures < FAILURES {
        let mut point = next;
        let mut handles = handle;
        let timeout = if gate::awaits_completion(watch.service) {
            WAIT_NS
        } else {
            IDLE_WAIT_NS
        };
        let mut request = TimelineWait {
            handles: ptr::from_mut(&mut handles) as u64,
            points: ptr::from_mut(&mut point) as u64,
            timeout_nsec: client::now_ns().saturating_add(timeout) as i64,
            count_handles: 1,
            flags: WAIT_FOR_SUBMIT,
            first_signaled: 0,
            pad: 0,
            deadline_nsec: 0,
        };
        match client::control(
            file,
            SYNCOBJ_TIMELINE_WAIT,
            ptr::from_mut(&mut request).cast(),
        ) {
            0 => {
                failures = 0;
                // Every point through the newest completed one is done.
                let reached = value(file, handle).unwrap_or(next).max(next);
                next = reached + 1;
                gate::completed_through(watch.service, watch.timeline, reached);
                look_again();
            }
            result if result == -ETIME || result == -EINTR => {
                failures = 0;
                // A stalled timeline with work outstanding, as after a
                // failed threaded submission: the device look notices the
                // loss no point or interrupt will signal. Never while idle.
                if result == -ETIME && gate::awaits_completion(watch.service) {
                    look_again();
                }
            }
            _ => failures += 1,
        }
    }
    // A newer waiter may already have stored its own.
    let _ = WATCHING.compare_exchange(generation, 0, Ordering::AcqRel, Ordering::Relaxed);
}

/// The newest point of `handle` that signaled.
fn value(file: c_int, handle: u32) -> Option<u64> {
    let mut handles = handle;
    let mut point = 0u64;
    let mut query = TimelineArray {
        handles: ptr::from_mut(&mut handles) as u64,
        points: ptr::from_mut(&mut point) as u64,
        count_handles: 1,
        flags: 0,
    };
    (client::control(file, SYNCOBJ_QUERY, ptr::from_mut(&mut query).cast()) == 0).then_some(point)
}
