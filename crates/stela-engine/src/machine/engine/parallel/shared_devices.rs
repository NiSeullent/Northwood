//! The board's devices shared by the PE threads: the locked board and its
//! device-bus calls.

use super::super::{GateCall, GateOutcome, LocalRegisters, PassiveRegisters, UnlockedDma};
use super::*;
use core::mem::ManuallyDrop;

pub(super) struct Board<D> {
    pub(super) devices: Mutex<BoardState<D>>,
    pub(super) hub: Arc<Hub>,
    /// Calls that found the devices' lock held, by [`Call`].
    pub(super) waits: [AtomicU64; Call::ALL.len()],
    /// By [`Call`], the host ticks ([`Hub::ticks`]) such calls waited for
    /// the lock, the ticks calls held it, and the longest one held it.
    pub(super) waited: [AtomicU64; Call::ALL.len()],
    pub(super) held: [AtomicU64; Call::ALL.len()],
    pub(super) longest: [AtomicU64; Call::ALL.len()],
    /// By [`Call`], the calls that changed another PE's lines and the host
    /// ticks they spent waking those PEs once the lock was released
    /// ([`wake_after`]): a sleeping PE's host core takes an interrupt.
    pub(super) woken: [AtomicU64; Call::ALL.len()],
    pub(super) wake_ticks: [AtomicU64; Call::ALL.len()],
    /// The latest count any machine passed, which every later call sees at
    /// least: the PEs' counters agree only to within their reads.
    pub(super) now: AtomicU64,
    /// Each PE's latched pins as every call left them ([`pack`]), which
    /// another PE's machine reads without the lock: only the home machine
    /// advances the devices with time.
    pub(super) pins: Box<[AtomicU8]>,
    /// The devices' DMA deadline (`u64::MAX`: none) and a reset they
    /// requested, as each call left them: the home machine asks at every
    /// step, without the lock.
    pub(super) dma: AtomicU64,
    /// The devices' next deadline in time (`u64::MAX`: none) as the last
    /// advance with time left it, written under the lock: a PE other than home advances them once
    /// it passed while home is outside its run loop
    /// ([`SharedDevices::advance_for_home`]).
    pub(super) time: AtomicU64,
    /// Locked calls so far, bumped as each releases the lock: the home
    /// machine keeps the deadline its last advance gave while this moved
    /// by its own call alone ([`Devices::generation`]).
    pub(super) generation: AtomicU64,
    pub(super) reset: Mutex<Option<Reset>>,
    pub(super) resetting: AtomicBool,
    /// PE starts the devices made, which every machine asks for at every
    /// wait, without the lock; `starting` changes only under `starts`.
    pub(super) starts: Mutex<VecDeque<Start>>,
    pub(super) starting: AtomicBool,
    /// The SoC sleeps until the host's power button
    /// ([`Devices::waits_on_host`]).
    pub(super) asleep: AtomicBool,
    /// The IMPLEMENTATION DEFINED registers (op0 3, CRn 15) the devices
    /// serve, by op1, CRm and op2 ([`register_bit`]): a PE's machine asks
    /// for every one it reaches, and the others take no lock.
    pub(super) registers: [u64; 16],
}

pub(super) struct BoardState<D> {
    devices: D,
    /// Each PE's device lines as last seen, under the same lock as the devices.
    lines: Vec<Lines>,
}

/// Pins as [`Board::pins`] holds them: IRQ and FIQ, or [`UNLATCHED`] for
/// devices whose lines need an advance to read.
pub(super) fn pack(pins: Option<Lines>) -> u8 {
    pins.map_or(UNLATCHED, |lines| {
        u8::from(lines.irq) | u8::from(lines.fiq) << 1
    })
}

pub(super) const UNLATCHED: u8 = u8::MAX;

/// The transfers a call makes on its own PE to publish what its copies
/// outside the devices' lock came to, while each leaves more
/// ([`SharedDevices::run_unlocked`]).
const UNLOCKED_ROUNDS: usize = 4;

pub(super) fn unpack(pins: u8) -> Option<Lines> {
    (pins != UNLATCHED).then_some(Lines {
        irq: pins & 1 != 0,
        fiq: pins & 2 != 0,
    })
}

/// The bit of `registers` for an IMPLEMENTATION DEFINED encoding.
pub(super) fn register_bit(encoding: Encoding) -> Option<usize> {
    (encoding.op0() == 3 && encoding.crn() == 15).then(|| {
        usize::from(encoding.op1()) << 7
            | usize::from(encoding.crm()) << 3
            | usize::from(encoding.op2())
    })
}

impl<D: Devices> Board<D> {
    /// Host work done outside the devices' lock completed something they
    /// should look at: the home machine takes its next transfer at once,
    /// which publishes their real deadline again. Told as a line change, which
    /// its sleep checks, so a wake sent just before it marks itself asleep
    /// is not lost.
    pub(super) fn call_home(&self) {
        let home = self.hub.home.load(Ordering::Acquire);
        self.dma.store(0, Ordering::Release);
        self.hub.lines_changed_unwoken(home);
        self.hub.wake(home);
    }

    pub(super) fn lock(&self, call: Call) -> Held<'_, D> {
        let (state, since, waited) = match self.devices.try_lock() {
            Ok(state) => (state, self.hub.ticks(), 0),
            Err(_) => {
                self.waits[call as usize].fetch_add(1, Ordering::Relaxed);
                let asked = self.hub.ticks();
                let state = self.devices.lock().unwrap_or_else(|e| e.into_inner());
                let since = self.hub.ticks();
                let waited = since.wrapping_sub(asked);
                self.waited[call as usize].fetch_add(waited, Ordering::Relaxed);
                (state, since, waited)
            }
        };
        Held {
            board: self,
            state: ManuallyDrop::new(state),
            call,
            since,
            waited,
            wakes: 0,
            pe: usize::MAX,
        }
    }

    /// Record what the machines ask of `devices` at every step: the home
    /// machine its resets, DMA and host inputs, and any its starts and the
    /// host's service. Every call's release runs it, so it writes only what
    /// changed: the machines read these without the lock, and an unchanged
    /// store would still take each one's cache line from every reader.
    /// Whether the home machine must be woken, which the caller does once
    /// it has released the lock.
    pub(super) fn publish(&self, devices: &mut D) -> bool {
        if let Some(reset) = devices.take_reset() {
            self.reset
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .get_or_insert(reset);
            self.resetting.store(true, Ordering::Release);
        }
        if let Some(start) = devices.take_start() {
            let mut starts = self.starts.lock().unwrap_or_else(|e| e.into_inner());
            starts.push_back(start);
            starts.extend(core::iter::from_fn(|| devices.take_start()));
            self.starting.store(true, Ordering::Release);
        }
        // A deadline earlier than the last, such as a host transfer that
        // completed on another core, wakes the home machine if it sleeps.
        let dma = devices.dma_deadline().unwrap_or(u64::MAX);
        let published = self.dma.load(Ordering::Acquire);
        let wake_home = published > dma;
        if published != dma {
            self.dma.store(dma, Ordering::Release);
        }
        // The latched pins alone: each PE adds the FIQ its own handle
        // drives ([`LocalRegisters::fiq`]), which changes without the lock.
        for pe in 0..self.pins.len() {
            self.publish_pins(pe, devices.latched_pins(pe));
        }
        let asleep = devices.waits_on_host();
        if self.asleep.load(Ordering::Relaxed) != asleep {
            self.asleep.store(asleep, Ordering::Release);
            if !asleep {
                // Each PE's own FIQ ([`LocalRegisters::fiq`]) was held low
                // while the SoC slept, and no latched pin tells it that one
                // set meanwhile is now up: every PE looks at its lines
                // again. Woken under the lock, as the SoC wakes rarely.
                for pe in 0..self.hub.pes() {
                    self.hub.lines_changed_unwoken(pe);
                    self.hub.wake(pe);
                }
            }
        }
        if devices.take_host_service() {
            self.hub.request_service();
        }
        wake_home
    }

    /// Publish the devices' next deadline in time.
    fn publish_time(&self, deadline: Option<u64>) {
        let time = deadline.unwrap_or(u64::MAX);
        if self.time.load(Ordering::Relaxed) != time {
            self.time.store(time, Ordering::Release);
        }
    }

    /// The devices' lock for `call` if no call holds it now.
    fn try_lock(&self, call: Call) -> Option<Held<'_, D>> {
        let state = self.devices.try_lock().ok()?;
        Some(Held {
            board: self,
            state: ManuallyDrop::new(state),
            call,
            since: self.hub.ticks(),
            waited: 0,
            wakes: 0,
            pe: usize::MAX,
        })
    }

    /// Publish the snapshot a remote PE reads without the devices' lock.
    pub(super) fn publish_pins(&self, pe: usize, lines: Option<Lines>) {
        let pins = &self.pins[pe];
        let packed = pack(lines);
        if pins.load(Ordering::Relaxed) != packed {
            pins.store(packed, Ordering::Release);
        }
    }

    /// The oldest start [`Self::publish`] kept.
    pub(super) fn take_start(&self) -> Option<Start> {
        if !self.starting.load(Ordering::Acquire) {
            return None;
        }
        let mut starts = self.starts.lock().unwrap_or_else(|e| e.into_inner());
        let start = starts.pop_front();
        self.starting.store(!starts.is_empty(), Ordering::Release);
        start
    }
}

/// The devices while a call holds their lock; its release publishes what
/// the call left for the home machine ([`Board::publish`]).
pub(super) struct Held<'a, D: Devices> {
    board: &'a Board<D>,
    /// Released in [`Drop`] before the host's work after the call.
    state: ManuallyDrop<crate::platform::sync::MutexGuard<'a, BoardState<D>>>,
    /// What the call is for, and [`Hub::ticks`] when it took the lock.
    call: Call,
    since: u64,
    /// The ticks the call waited for the lock.
    waited: u64,
    /// The PEs to wake once the lock is released, by bit ([`wake_bit`]): a
    /// wake can cost the host an interrupt to another core, which no other
    /// PE should wait for the lock behind.
    wakes: u64,
    /// The PE whose call this is, or `usize::MAX` when the caller did not
    /// say: the host work its release leaves ([`Kick::released`]) runs only
    /// on a PE outside its run loop ([`Hub::serves_released`]).
    pe: usize,
}

impl<D: Devices> Held<'_, D> {
    /// The count the devices see for a call at `now`, at least as high as
    /// every earlier call's. Only this guard can advance the count.
    fn now(&self, now: u64) -> u64 {
        let previous = self.board.now.load(Ordering::Relaxed);
        if now > previous {
            self.board.now.store(now, Ordering::Release);
        }
        previous.max(now)
    }

    /// Tell every PE but `cpu` whose lines changed since it last saw them.
    /// Every change reaches the devices under their lock, so a PE learns of
    /// each from the call that made it, and needs no device deadline of its
    /// own: only the home machine advances the devices with time.
    fn notify(&mut self, cpu: usize, now: u64) {
        let BoardState {
            devices,
            lines: seen,
        } = &mut **self.state;
        if let Some(own) = seen.get_mut(cpu)
            && let Some(lines) = devices.latched_pins(cpu)
        {
            // The caller reads its own lines after the call.
            *own = lines;
        }
        // PEs whose own FIQ word the call set (a deferred fast IPI that
        // fired): they read it themselves, so only the epoch tells them.
        // The caller reads its own after the call.
        devices.take_signalled(cpu);
        for pe in self.board.hub.others(cpu) {
            // The latched pins alone: each PE adds the FIQ it reads itself
            // ([`LocalRegisters::fiq`]), which changes without the lock.
            let pins = devices.latched_pins(pe);
            let lines = match pins {
                Some(lines) => lines,
                None => devices.lines(pe, now),
            };
            // The receiver may read its pins as soon as the epoch changes,
            // before this call releases the devices' lock.
            self.board.publish_pins(pe, pins);
            let signalled = devices.take_signalled(pe);
            if lines != seen[pe] || signalled {
                seen[pe] = lines;
                self.board.hub.lines_changed_unwoken(pe);
                match wake_bit(pe) {
                    0 => self.board.hub.wake(pe),
                    bit => self.wakes |= bit,
                }
            }
        }
    }
}

/// A PE's bit in [`Held::wakes`]; none for a PE past its width.
fn wake_bit(pe: usize) -> u64 {
    u32::try_from(pe)
        .ok()
        .and_then(|pe| 1u64.checked_shl(pe))
        .unwrap_or(0)
}

/// Wake each PE of `wakes` ([`wake_bit`]), after the lock is released.
fn wake_after(hub: &Hub, wakes: u64) {
    let mut left = wakes;
    while left != 0 {
        hub.wake(left.trailing_zeros() as usize);
        left &= left - 1;
    }
}

impl<D: Devices> core::ops::Deref for Held<'_, D> {
    type Target = D;
    fn deref(&self) -> &D {
        &self.state.devices
    }
}

impl<D: Devices> core::ops::DerefMut for Held<'_, D> {
    fn deref_mut(&mut self) -> &mut D {
        &mut self.state.devices
    }
}

impl<D: Devices> Drop for Held<'_, D> {
    fn drop(&mut self) {
        if self.board.publish(&mut self.state.devices) {
            let home = self.board.hub.home.load(Ordering::Acquire);
            match wake_bit(home) {
                0 => self.board.hub.wake(home),
                bit => self.wakes |= bit,
            }
        }
        // Only the holder writes these.
        let held = self.board.hub.ticks().wrapping_sub(self.since);
        let call = self.call as usize;
        self.board.held[call].fetch_add(held, Ordering::Relaxed);
        if held > self.board.longest[call].load(Ordering::Relaxed) {
            self.board.longest[call].store(held, Ordering::Relaxed);
        }
        // Before the release: a reader that saw the lock free after this
        // call sees its bump.
        self.board.generation.fetch_add(1, Ordering::AcqRel);
        // SAFETY: dropped once, here, and not used after.
        unsafe { ManuallyDrop::drop(&mut self.state) };
        if self.wakes != 0 {
            let hub = &self.board.hub;
            let start = hub.ticks();
            wake_after(hub, self.wakes);
            let spent = hub.ticks().wrapping_sub(start);
            self.board.woken[call].fetch_add(1, Ordering::Relaxed);
            self.board.wake_ticks[call].fetch_add(spent, Ordering::Relaxed);
        }
        // Host work completed outside the lock: the home machine takes its
        // next transfer at once, which publishes the real deadline again.
        // A PE inside its run loop leaves that work, which can wait on a
        // host device, to the next release by a PE outside it or to the
        // host's own service: meanwhile it would read no mail, and every
        // maintenance broadcast would wait for it. A PE outside it for a
        // guest exit (a storage doorbell's) hands the work to a PE that
        // sleeps waiting for an interrupt, whose idle core does it at its
        // host service, and goes on with its guest; with none asleep it does
        // the work itself, as before.
        let hub = &self.board.hub;
        if hub.serves_released(self.pe)
            && hub.kick.release_due()
            && !hub.hand_off_service(self.pe)
            && hub.kick.released()
        {
            self.board.call_home();
        }
    }
}

// SAFETY: the devices are reached only under `devices`' lock, so state they
// share through `Rc` or `RefCell` is never touched by two cores at once, and
// no reference into them outlives the guard.
unsafe impl<D> Send for Board<D> {}
// SAFETY: shared references rely on the same argument as `Send` above.
unsafe impl<D> Sync for Board<D> {}

/// How long calls of one [`Call`] waited for the devices' lock and held it
/// ([`SharedDevices::lock_times`]), in host ticks.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LockTimes {
    pub waited: u64,
    pub held: u64,
    pub longest: u64,
    /// Calls that changed another PE's lines, and the ticks waking those PEs.
    pub woken: u64,
    pub wake: u64,
}

/// The board's devices, shared by every PE's machine.
pub struct SharedDevices<D> {
    pub(super) board: Arc<Board<D>>,
    /// The machine that serves resets and DMA.
    pub(super) home: bool,
    /// Each PE's [`Devices::local_registers`], once asked for.
    local: Vec<Option<Option<Arc<dyn LocalRegisters>>>>,
    /// [`Devices::passive_registers`], asked for once the board is built.
    passive: Option<Arc<dyn PassiveRegisters>>,
    /// The ticks this handle's calls waited for the lock since
    /// [`Devices::take_lock_wait`].
    waited: AtomicU64,
}

impl<D: Devices> SharedDevices<D> {
    /// The home machine's handle.
    pub fn new(mut devices: D, hub: Arc<Hub>) -> Self {
        let pes = hub.pes();
        devices.defer_dma();
        let passive = devices.passive_registers();
        let mut registers = [0u64; 16];
        for bit in 0..1 << 10 {
            let (op1, crm, op2) = ((bit >> 7) as u8, (bit >> 3 & 15) as u8, (bit & 7) as u8);
            if devices.serves_register(Encoding::new(3, op1, 15, crm, op2)) {
                registers[bit / 64] |= 1 << (bit % 64);
            }
        }
        let board = Board {
            devices: Mutex::new(BoardState {
                devices,
                lines: alloc::vec![Lines::default(); pes],
            }),
            hub,
            waits: Default::default(),
            waited: Default::default(),
            held: Default::default(),
            longest: Default::default(),
            woken: Default::default(),
            wake_ticks: Default::default(),
            now: AtomicU64::new(0),
            pins: (0..pes).map(|_| AtomicU8::new(UNLATCHED)).collect(),
            dma: AtomicU64::new(u64::MAX),
            time: AtomicU64::new(u64::MAX),
            generation: AtomicU64::new(0),
            reset: Mutex::new(None),
            resetting: AtomicBool::new(false),
            starts: Mutex::new(VecDeque::new()),
            starting: AtomicBool::new(false),
            asleep: AtomicBool::new(false),
            registers,
        };
        drop(board.lock(Call::Other));
        Self {
            board: Arc::new(board),
            home: true,
            local: Vec::new(),
            passive,
            waited: AtomicU64::new(0),
        }
    }

    /// A handle for another PE's machine.
    pub fn worker(&self) -> Self {
        Self {
            board: Arc::clone(&self.board),
            home: false,
            local: Vec::new(),
            passive: self.passive.clone(),
            waited: AtomicU64::new(0),
        }
    }

    pub fn hub(&self) -> &Arc<Hub> {
        &self.board.hub
    }

    /// Run `f` on the devices while no PE reaches them.
    pub fn with<R>(&self, f: impl FnOnce(&mut D) -> R) -> R {
        self.call(Call::Other, f)
    }

    /// Run `f` on the devices if no PE reaches them now, else `None`: for
    /// work that can wait, which a PE whose device access waits on the host
    /// (a disk transfer's MMIO, say) would otherwise hold up.
    pub fn try_with<R>(&self, f: impl FnOnce(&mut D) -> R) -> Option<R> {
        let mut held = self.board.try_lock(Call::Other)?;
        Some(f(&mut held))
    }

    /// Advance the devices to `now` for the home machine, if their deadline
    /// in time passed while it is outside its run loop (serving the host,
    /// compiling, a heartbeat) and no call holds them: vblanks, device
    /// timers and completions then come on time whatever home does. Home
    /// advances them as ever once it runs; the devices see one monotonic
    /// count ([`Held::now`]) from whichever PE asks.
    fn advance_for_home(&self, now: u64) {
        let board = &self.board;
        if board.time.load(Ordering::Acquire) > now || !board.hub.home_outside() {
            return;
        }
        let Some(mut devices) = board.try_lock(Call::Deadline) else {
            return;
        };
        let now = devices.now(now);
        if board.time.load(Ordering::Relaxed) > now {
            return;
        }
        let deadline = devices.deadline(now);
        board.publish_time(deadline);
        devices.notify(usize::MAX, now);
    }

    /// When a PE other than home next looks at the devices' deadline: at
    /// it, at most 10 ms on, or 1 ms on while it is overdue and home is in
    /// its run loop, which advances them.
    fn look_again(&self, now: u64) -> u64 {
        let milli = self.board.hub.counts_per_milli().max(1);
        let time = self.board.time.load(Ordering::Acquire);
        if time > now {
            time.min(now.saturating_add(10 * milli))
        } else {
            now.saturating_add(milli)
        }
    }

    pub(super) fn call<R>(&self, call: Call, f: impl FnOnce(&mut D) -> R) -> R {
        f(&mut self.lock(call))
    }

    /// The board's lock for `call`, counting the ticks it waited.
    fn lock(&self, call: Call) -> Held<'_, D> {
        let held = self.board.lock(call);
        if held.waited != 0 {
            self.waited.fetch_add(held.waited, Ordering::Relaxed);
        }
        held
    }

    /// Calls that found the devices' lock held since the board was made, by
    /// [`Call`].
    pub fn waits(&self) -> [u64; Call::ALL.len()] {
        self.board
            .waits
            .each_ref()
            .map(|waits| waits.load(Ordering::Relaxed))
    }

    /// By [`Call`], the host ticks ([`Kick::ticks`]) calls have waited for
    /// the devices' lock and held it since the board was made, and the
    /// longest any one held it since this was last asked.
    pub fn lock_times(&self) -> [LockTimes; Call::ALL.len()] {
        core::array::from_fn(|call| LockTimes {
            waited: self.board.waited[call].load(Ordering::Relaxed),
            held: self.board.held[call].load(Ordering::Relaxed),
            woken: self.board.woken[call].load(Ordering::Relaxed),
            wake: self.board.wake_ticks[call].load(Ordering::Relaxed),
            longest: self.board.longest[call].swap(0, Ordering::Relaxed),
        })
    }

    /// Host work done outside the devices' lock, such as a staged disk's
    /// step, completed something they should look at ([`Board::call_home`]).
    pub fn call_home(&self) {
        self.board.call_home();
    }

    /// Serve the host's inputs with `f`, which can change any PE's lines.
    pub fn serve<R>(&self, f: impl FnOnce(&mut D) -> R) -> R {
        let now = self.board.now.load(Ordering::Acquire);
        self.changing(Call::Home, usize::MAX, now, |devices, _| f(devices))
    }

    /// Run `f` on the devices at `now`, as [`Held::now`] gives it.
    pub(super) fn at<R>(&self, call: Call, now: u64, f: impl FnOnce(&mut D, u64) -> R) -> R {
        let mut devices = self.lock(call);
        let now = devices.now(now);
        f(&mut devices, now)
    }

    /// PE `cpu`'s own registers, without the devices' lock once it has its
    /// handle.
    pub(super) fn local_registers_of(&mut self, cpu: usize) -> Option<&Arc<dyn LocalRegisters>> {
        if self.local.len() <= cpu {
            self.local.resize_with(cpu + 1, || None);
        }
        if self.local[cpu].is_none() {
            self.local[cpu] = Some(self.with(|devices| devices.local_registers(cpu)));
        }
        self.local[cpu].as_ref()?.as_ref()
    }

    /// PE `cpu`'s FIQ from its own handle ([`LocalRegisters::fiq`]), held
    /// low while the SoC sleeps, as the latched pins are.
    fn signalled_fiq(&mut self, cpu: usize) -> bool {
        !self.board.asleep.load(Ordering::Acquire)
            && self
                .local_registers_of(cpu)
                .is_some_and(|local| local.fiq())
    }

    /// A call that can change devices: run it, then tell every other PE
    /// whose lines it changed.
    pub(super) fn changing<R>(
        &self,
        call: Call,
        cpu: usize,
        now: u64,
        f: impl FnOnce(&mut D, u64) -> R,
    ) -> R {
        let mut devices = self.lock(call);
        devices.pe = cpu;
        let now = devices.now(now);
        let result = f(&mut devices, now);
        devices.notify(cpu, now);
        result
    }

    /// Run the data copies a transfer by PE `cpu` at `now` left
    /// ([`Devices::take_unlocked`]) on `memory`, outside the devices' lock,
    /// then transfer again at once on this PE: that publishes what they came
    /// to, such as a read's completion and its interrupt, and tells the PEs
    /// whose lines it changed. No other PE need come by: an idle home
    /// machine may be about to halt, and a wake sent before it marks itself
    /// asleep would be lost. The first refusal of those transfers.
    fn run_unlocked(
        &self,
        cpu: usize,
        now: u64,
        mut copies: Vec<Box<dyn UnlockedDma>>,
        memory: &mut dyn super::super::Memory,
    ) -> Result<(), DeviceError> {
        let mut result = Ok(());
        for _ in 0..UNLOCKED_ROUNDS {
            if copies.is_empty() {
                return result;
            }
            for copy in copies.drain(..) {
                copy.run(memory);
            }
            let (transfer, more) = self.changing(Call::Transfer, cpu, now, |devices, now| {
                (devices.transfer(cpu, memory, now), devices.take_unlocked())
            });
            if result.is_ok() {
                result = transfer;
            }
            copies = more;
        }
        // Copies kept coming: run the last ones and leave their publication
        // to the home machine, told as a line change, which its sleep checks.
        if !copies.is_empty() {
            for copy in copies {
                copy.run(memory);
            }
            self.board.call_home();
        }
        result
    }
}

impl<D: Devices> Devices for SharedDevices<D> {
    fn pc_sample_target(&self, physical: u64, bytes: u8) -> Option<usize> {
        self.with(|devices| devices.pc_sample_target(physical, bytes))
    }
    fn observe_pc_sample(&mut self, physical: u64, pc: u64) {
        self.with(|devices| devices.observe_pc_sample(physical, pc));
    }
    fn read(&mut self, cpu: usize, physical: u64, bytes: u8, now: u64) -> Result<u64, DeviceError> {
        if let Some(value) = self.passive.as_ref().and_then(|p| p.read(physical, bytes)) {
            return Ok(value);
        }
        self.changing(Call::Access, cpu, now, |devices, now| {
            devices.read(cpu, physical, bytes, now)
        })
    }
    /// A passive register ([`Devices::passive_registers`]) is read without
    /// the lock: its value is what the last write published, and its read
    /// changes nothing the lock would settle.
    fn read_scalar(&mut self, cpu: usize, physical: u64, bytes: u8, now: u64) -> ScalarRead {
        if let Some(value) = self.passive.as_ref().and_then(|p| p.read(physical, bytes)) {
            return ScalarRead::Served(Ok(value));
        }
        self.changing(Call::Access, cpu, now, |devices, now| {
            devices.read_scalar(cpu, physical, bytes, now)
        })
    }
    fn requires_canonical_read(&self, cpu: usize, physical: u64, shape: DeviceReadShape) -> bool {
        self.with(|devices| devices.requires_canonical_read(cpu, physical, shape))
    }
    fn requires_canonical_write(&self, cpu: usize, physical: u64, bytes: u8) -> bool {
        self.with(|devices| devices.requires_canonical_write(cpu, physical, bytes))
    }
    fn read_canonical(
        &mut self,
        cpu: usize,
        physical: u64,
        bytes: u8,
        now: u64,
        memory: &mut dyn CanonicalDeviceMemory,
    ) -> Result<ConsumedScalar, DeviceError> {
        self.changing(Call::Access, cpu, now, |devices, now| {
            devices.read_canonical(cpu, physical, bytes, now, memory)
        })
    }
    fn read_pair_canonical(
        &mut self,
        cpu: usize,
        physical: u64,
        bytes: u8,
        now: u64,
        memory: &mut dyn CanonicalDeviceMemory,
    ) -> Result<ConsumedPair, DeviceError> {
        self.changing(Call::Access, cpu, now, |devices, now| {
            devices.read_pair_canonical(cpu, physical, bytes, now, memory)
        })
    }
    fn write_canonical(
        &mut self,
        cpu: usize,
        physical: u64,
        bytes: u8,
        value: u64,
        now: u64,
        owners: &mut dyn CanonicalDeviceWrite,
    ) -> Result<ConsumedWrite, DeviceError> {
        self.changing(Call::Access, cpu, now, |devices, now| {
            devices.write_canonical(cpu, physical, bytes, value, now, owners)
        })
    }
    fn write_pair_canonical(
        &mut self,
        cpu: usize,
        physical: u64,
        bytes: u8,
        values: [u64; 2],
        now: u64,
        owners: &mut dyn CanonicalDeviceWrite,
    ) -> Result<ConsumedWrite, DeviceError> {
        self.changing(Call::Access, cpu, now, |devices, now| {
            devices.write_pair_canonical(cpu, physical, bytes, values, now, owners)
        })
    }
    fn read_pair(
        &mut self,
        cpu: usize,
        physical: u64,
        bytes: u8,
        now: u64,
    ) -> Result<Option<ConsumedPair>, DeviceError> {
        self.changing(Call::Access, cpu, now, |devices, now| {
            devices.read_pair(cpu, physical, bytes, now)
        })
    }
    fn write(
        &mut self,
        cpu: usize,
        physical: u64,
        bytes: u8,
        value: u64,
        now: u64,
    ) -> Result<(), DeviceError> {
        self.changing(Call::Access, cpu, now, |devices, now| {
            devices.write(cpu, physical, bytes, value, now)
        })
    }
    fn write_with_processors(
        &mut self,
        cpu: usize,
        physical: u64,
        bytes: u8,
        value: u64,
        now: u64,
        processors: super::super::ProcessorWrite<'_>,
    ) -> Result<(), DeviceError> {
        self.changing(Call::Access, cpu, now, |devices, now| {
            devices.write_with_processors(cpu, physical, bytes, value, now, processors)
        })
    }
    fn write_plain(
        &mut self,
        cpu: usize,
        physical: u64,
        bytes: u8,
        value: u64,
        now: u64,
        processors: super::super::ProcessorWrite<'_>,
    ) -> Option<Result<(), DeviceError>> {
        self.changing(Call::Access, cpu, now, |devices, now| {
            devices.write_plain(cpu, physical, bytes, value, now, processors)
        })
    }
    /// The home machine's asking advances the devices to `now`, which can
    /// change any PE's lines; another PE's machine reads its latched pins.
    fn lines(&mut self, cpu: usize, now: u64) -> Lines {
        if !self.home
            && let Some(lines) = self.pins(cpu)
        {
            return lines;
        }
        let mut devices = self.lock(Call::Lines);
        let now = devices.now(now);
        let lines = devices.lines(cpu, now);
        devices.state.lines[cpu] = lines;
        devices.notify(cpu, now);
        lines
    }
    /// The pins every call left, with the FIQ the PE's own handle drives
    /// ([`LocalRegisters::fiq`]): an Apple fast IPI sent without the lock.
    fn pins(&mut self, cpu: usize) -> Option<Lines> {
        let mut lines = unpack(self.board.pins.get(cpu)?.load(Ordering::Acquire))?;
        lines.fiq |= self.signalled_fiq(cpu);
        Some(lines)
    }
    /// The home machine advances the devices with time and tells the other
    /// PEs what that changes ([`Self::notify`]); their machines wait for
    /// those changes instead, and wake for no device deadline. One that
    /// asks while awake advances them when their deadline passed and home
    /// is outside its run loop ([`Self::advance_for_home`]).
    fn deadline(&mut self, now: u64) -> Option<u64> {
        if !self.home {
            self.advance_for_home(now);
            return None;
        }
        let mut devices = self.lock(Call::Deadline);
        let now = devices.now(now);
        let deadline = devices.deadline(now);
        devices.notify(usize::MAX, now);
        self.board.publish_time(deadline);
        deadline
    }
    /// The home machine's [`Self::lines`] and [`Self::deadline`] under one
    /// lock, from one advance of the devices. Another PE's machine gets its
    /// pins and when to look again ([`Self::look_again`]).
    fn lines_and_deadline(&mut self, cpu: usize, now: u64) -> (Lines, Option<u64>) {
        if !self.home {
            self.advance_for_home(now);
            return (self.lines(cpu, now), Some(self.look_again(now)));
        }
        let mut devices = self.lock(Call::Lines);
        let now = devices.now(now);
        let (lines, deadline) = devices.lines_and_deadline(cpu, now);
        devices.state.lines[cpu] = lines;
        devices.notify(cpu, now);
        self.board.publish_time(deadline);
        (lines, deadline)
    }
    fn system_register(
        &mut self,
        cpu: usize,
        encoding: Encoding,
        write: Option<u64>,
        now: u64,
    ) -> Option<Result<u64, &'static str>> {
        if let Some(bit) = register_bit(encoding)
            && self.board.registers[bit / 64] & 1 << (bit % 64) == 0
        {
            return None;
        }
        if let Some(value) = self
            .local_registers_of(cpu)
            .and_then(|local| local.register(encoding, write))
        {
            return Some(Ok(value));
        }
        // A signal to another PE that needs no lock (an immediate fast
        // IPI): the target looks at its lines again at once. One to a CPU
        // the board has but no PE runs (a DeviceTree listing more CPUs than
        // PEs) is refused, which the engine reports once and goes on: no PE
        // reads that CPU's word, so the signal is dropped, as a locked one
        // is ([`Held::notify`] reaches only PEs).
        if let Some(value) = write
            && let Some(sent) = self
                .local_registers_of(cpu)
                .and_then(|local| local.send(encoding, value))
        {
            if let Some(pe) = sent {
                if pe >= self.board.hub.pes() {
                    return Some(Err("signal to a CPU that no PE runs is dropped"));
                }
                let hub = &self.board.hub;
                hub.lines_changed_unwoken(pe);
                // Timed as a locked call's wakes are (`DEVICE WAKES
                // register=`): a sleeping target's core takes an interrupt,
                // which this exit pays.
                let start = hub.ticks();
                hub.wake(pe);
                let spent = hub.ticks().wrapping_sub(start);
                let call = Call::Register as usize;
                self.board.woken[call].fetch_add(1, Ordering::Relaxed);
                self.board.wake_ticks[call].fetch_add(spent, Ordering::Relaxed);
            }
            return Some(Ok(0));
        }
        if write.is_none() {
            return self.at(Call::Register, now, |devices, now| {
                devices.system_register(cpu, encoding, None, now)
            });
        }
        self.changing(Call::Register, cpu, now, |devices, now| {
            devices.system_register(cpu, encoding, write, now)
        })
    }
    /// Only the home machine takes a reset, which stops every PE.
    fn take_reset(&mut self) -> Option<Reset> {
        let resetting = &self.board.resetting;
        if !self.home
            || !resetting.load(Ordering::Relaxed)
            || !resetting.swap(false, Ordering::Acquire)
        {
            return None;
        }
        self.board
            .reset
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
    }
    fn take_start(&mut self) -> Option<Start> {
        self.board.take_start()
    }
    fn reset_vector(&self, cpu: usize) -> Option<u64> {
        self.with(|devices| devices.reset_vector(cpu))
    }
    fn processor_power(&mut self, cpu: usize, on: bool) {
        let now = self.board.now.load(Ordering::Acquire);
        self.changing(Call::Other, cpu, now, |devices, _| {
            devices.processor_power(cpu, on)
        });
    }
    fn irq_steering(&mut self, cpu: usize, refuses: bool, now: u64) {
        self.changing(Call::Other, cpu, now, |devices, now| {
            devices.irq_steering(cpu, refuses, now)
        });
    }
    /// Only the home machine takes the host's power button.
    fn waits_on_host(&self) -> bool {
        self.home && self.board.asleep.load(Ordering::Acquire)
    }
    /// Any PE that waits for an interrupt can serve the host.
    fn take_host_service(&mut self) -> bool {
        self.board.hub.take_service()
    }
    fn memory_lock(&self) -> mmu::MemoryLocks {
        self.with(|devices| devices.memory_lock())
    }
    fn register_storage(&mut self, cpu: usize, encoding: Encoding) -> Option<u64> {
        self.local_registers_of(cpu)?.storage(encoding)
    }
    fn serves_register(&self, encoding: Encoding) -> bool {
        register_bit(encoding)
            .is_none_or(|bit| self.board.registers[bit / 64] & 1 << (bit % 64) != 0)
    }
    /// Only the home machine moves devices' asynchronous DMA, through its
    /// host's canonical RAM owner.
    fn generation(&self) -> Option<u64> {
        self.home
            .then(|| self.board.generation.load(Ordering::Acquire))
    }
    fn dma_deadline(&self) -> Option<u64> {
        if !self.home {
            return None;
        }
        let dma = self.board.dma.load(Ordering::Acquire);
        (dma != u64::MAX).then_some(dma)
    }
    fn transfer(
        &mut self,
        cpu: usize,
        memory: &mut dyn super::super::Memory,
        now: u64,
    ) -> Result<(), DeviceError> {
        let (result, copies) = self.changing(Call::Transfer, cpu, now, |devices, now| {
            (devices.transfer(cpu, memory, now), devices.take_unlocked())
        });
        let published = self.run_unlocked(cpu, now, copies, memory);
        result.and(published)
    }
    fn take_call(&mut self, cpu: usize) -> Option<GateCall> {
        self.with(|devices| devices.take_call(cpu))
    }
    fn finish_call(&mut self, outcome: GateOutcome) {
        self.with(|devices| devices.finish_call(outcome));
    }
    /// The lock, the DMA and the call after a write, under one lock.
    fn settle_write(
        &mut self,
        cpu: usize,
        now: u64,
        memory: &mut dyn super::super::LockedMemory,
    ) -> super::super::Settled {
        let (mut settled, copies) = self.changing(Call::Transfer, cpu, now, |devices, now| {
            (
                devices.settle_write(cpu, now, memory),
                devices.take_unlocked(),
            )
        });
        if !copies.is_empty() {
            let published = self.run_unlocked(cpu, now, copies, memory.under(settled.after));
            if settled.transfer.is_ok() {
                settled.transfer = published;
            }
        }
        settled
    }
    fn device_name(&self, physical: u64) -> Option<&'static str> {
        self.with(|devices| devices.device_name(physical))
    }
    fn take_lock_wait(&mut self) -> u64 {
        self.waited.swap(0, Ordering::Relaxed)
    }
}
