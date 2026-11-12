//! One kept timing owner for a driver: the nominal clock its deadlines are
//! counted against, and the qualified provider of the minimum delays its
//! device needs at least.
use crate::{
    Clock,
    block::BlockError,
    bus::{i2c, pci::power},
};
use alloc::rc::Rc;

/// The nominal clock alone grants no minimum wait: waits a device needs at
/// least go through the kept provider.
#[derive(Clone)]
pub struct Timing<N = Clock> {
    nominal: N,
    minimum: Rc<dyn power::Timer>,
}

impl<N> Timing<N> {
    /// Associate an existing provider; this performs no device effect and
    /// grants no clock, CPU, root or hardware ownership.
    pub fn new(nominal: N, minimum: Rc<dyn power::Timer>) -> Self {
        Self { nominal, minimum }
    }

    /// [`Self::new`] once the provider answers.
    pub fn checked(nominal: N, minimum: Rc<dyn power::Timer>) -> Result<Self, BlockError> {
        minimum.check()?;
        Ok(Self::new(nominal, minimum))
    }

    pub fn minimum(&self) -> &dyn power::Timer {
        self.minimum.as_ref()
    }

    /// Wait at least `us` microseconds.
    pub fn delay(&self, us: u64) -> Result<(), BlockError> {
        self.minimum.wait_us(us).map_err(BlockError::PciPower)
    }
}

impl<N: Copy> Timing<N> {
    pub fn nominal(&self) -> N {
        self.nominal
    }
}

impl Timing<Clock> {
    pub fn now_us(&self) -> u64 {
        self.nominal.now_us()
    }

    /// [`Clock::wait`] on the nominal clock: a deadline, not a minimum.
    #[track_caller]
    pub fn wait(&self, us: u64, ready: impl FnMut() -> bool) -> bool {
        self.nominal.wait(us, ready)
    }
}

impl<N: i2c::Timer> i2c::Timer for Timing<N> {
    fn now_us(&self) -> u64 {
        self.nominal.now_us()
    }
    fn relax(&self) {
        self.nominal.relax();
    }
}

// SAFETY: all admission and minimum waits delegate to the same retained
// provider; the independently supplied nominal clock grants nothing.
unsafe impl<N> power::Timer for Timing<N> {
    fn check(&self) -> Result<(), power::Error> {
        self.minimum.check()
    }
    fn wait_us(&self, us: u64) -> Result<(), power::Error> {
        self.minimum.wait_us(us)
    }
}
