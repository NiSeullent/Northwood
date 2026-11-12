//! A spin wait with a deadline: `stela-boot` takes the host drivers' lock
//! with it, so a holder that never lets go ends in a named stop instead of
//! a processor spinning forever.

/// How a [`spin`] ended, with the counter ticks it waited.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Spin {
    /// `take` succeeded.
    Taken(u64),
    /// `limit` ticks passed without `take` succeeding.
    Overrun(u64),
}

/// Call `take` until it succeeds or `limit` ticks of `now` have passed
/// since the first call, running `relax` between attempts. The counter may
/// wrap. `take` is tried once more after the deadline, so a holder that
/// lets go during the last `relax` is still taken.
pub fn spin(
    mut take: impl FnMut() -> bool,
    mut now: impl FnMut() -> u64,
    limit: u64,
    mut relax: impl FnMut(),
) -> Spin {
    let started = now();
    loop {
        if take() {
            return Spin::Taken(now().wrapping_sub(started));
        }
        relax();
        let waited = now().wrapping_sub(started);
        if waited >= limit {
            return if take() {
                Spin::Taken(waited)
            } else {
                Spin::Overrun(waited)
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::cell::Cell;

    #[test]
    fn a_free_lock_is_taken_at_once() {
        let relaxes = Cell::new(0);
        let spun = spin(|| true, || 7, 10, || relaxes.set(relaxes.get() + 1));
        assert_eq!(spun, Spin::Taken(0));
        assert_eq!(relaxes.get(), 0);
    }

    #[test]
    fn a_lock_released_before_the_deadline_is_taken() {
        let tries = Cell::new(0u32);
        let clock = Cell::new(0u64);
        let spun = spin(
            || {
                tries.set(tries.get() + 1);
                tries.get() == 4
            },
            || clock.get(),
            100,
            || clock.set(clock.get() + 10),
        );
        assert_eq!(spun, Spin::Taken(30));
    }

    #[test]
    fn a_lock_never_released_overruns_once_the_limit_passes() {
        let clock = Cell::new(u64::MAX - 15);
        let relaxes = Cell::new(0u32);
        let spun = spin(
            || false,
            || clock.get(),
            50,
            || {
                relaxes.set(relaxes.get() + 1);
                clock.set(clock.get().wrapping_add(10));
            },
        );
        // The counter wrapped on the way; the wait is still bounded.
        assert_eq!(spun, Spin::Overrun(50));
        assert_eq!(relaxes.get(), 5);
    }

    #[test]
    fn a_release_during_the_last_relax_is_still_taken() {
        let clock = Cell::new(0u64);
        let spun = spin(
            || clock.get() >= 20,
            || clock.get(),
            20,
            || clock.set(clock.get() + 10),
        );
        assert_eq!(spun, Spin::Taken(20));
    }
}
