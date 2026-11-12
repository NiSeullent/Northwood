//! Placing a firmware allocation in write-back memory. Firmware MTRRs may
//! leave some RAM uncached (a variable-range cover that stops short of the
//! top of memory, or a non-WB range over RAM); the engine refuses to run its
//! stacks there. Before ExitBootServices the loader can instead take the
//! pages again until they land in write-back memory, keeping the refused
//! pages allocated meanwhile so the firmware cannot hand them back.
use alloc::vec::Vec;

/// Allocations tried before the first is kept as it is.
pub const ATTEMPTS: usize = 8;

/// How an allocation was placed.
#[derive(Debug, PartialEq, Eq)]
pub struct Placement {
    /// The physical base kept.
    pub base: u64,
    /// The memory types of `base`: `Some(true)` write-back, `Some(false)` not
    /// (every attempt was refused, and the engine's own check reports it),
    /// `None` not decidable here (the engine's own check decides).
    pub writeback: Option<bool>,
    /// Refused allocations, for the caller to free.
    pub rejected: Vec<u64>,
}

/// Allocate with `allocate` until `writeback` accepts a base, at most
/// [`ATTEMPTS`] times. A failed allocation after a refused one ends the
/// search with the first refused base kept.
pub fn place<E>(
    mut allocate: impl FnMut() -> Result<u64, E>,
    mut writeback: impl FnMut(u64) -> Option<bool>,
) -> Result<Placement, E> {
    let mut rejected = Vec::new();
    for _ in 0..ATTEMPTS {
        let base = match allocate() {
            Ok(base) => base,
            Err(error) if rejected.is_empty() => return Err(error),
            Err(_) => break,
        };
        match writeback(base) {
            Some(false) => rejected.push(base),
            decided => {
                return Ok(Placement {
                    base,
                    writeback: decided,
                    rejected,
                });
            }
        }
    }
    let base = rejected.remove(0);
    Ok(Placement {
        base,
        writeback: Some(false),
        rejected,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A PC whose MTRRs cover RAM with WB only below 64 GiB (one WB range
    /// 0-64 GiB, UC default), as firmware does with 96 GiB of DDR5 when it
    /// runs out of variable ranges: the firmware hands out the top first.
    #[test]
    fn a_reservation_above_the_writeback_cover_moves_below_it() {
        let mut next = [80u64 << 30, 72 << 30, 40 << 30].into_iter();
        let placed = place(|| next.next().ok_or(()), |base| Some(base < 64 << 30)).unwrap();
        assert_eq!(placed.base, 40 << 30);
        assert_eq!(placed.writeback, Some(true));
        assert_eq!(placed.rejected, [80 << 30, 72 << 30]);
    }

    #[test]
    fn undecidable_types_keep_the_first_allocation() {
        let placed = place(|| Ok::<_, ()>(1 << 32), |_| None).unwrap();
        assert_eq!(placed.base, 1 << 32);
        assert_eq!(placed.writeback, None);
        assert!(placed.rejected.is_empty());
    }

    #[test]
    fn no_writeback_memory_keeps_the_first_and_frees_the_rest() {
        let mut base = 0u64;
        let placed = place(
            || {
                base += 1 << 30;
                Ok::<_, ()>(base)
            },
            |_| Some(false),
        )
        .unwrap();
        assert_eq!(placed.base, 1 << 30);
        assert_eq!(placed.writeback, Some(false));
        assert_eq!(placed.rejected.len(), ATTEMPTS - 1);
    }

    #[test]
    fn exhausted_firmware_after_a_refusal_keeps_the_refused_pages() {
        let mut next = [8u64 << 30].into_iter();
        let placed = place(|| next.next().ok_or("out"), |_| Some(false)).unwrap();
        assert_eq!(placed.base, 8 << 30);
        assert_eq!(placed.writeback, Some(false));
        assert!(placed.rejected.is_empty());
        assert_eq!(place(|| Err::<u64, _>("out"), |_| Some(true)), Err("out"));
    }
}
