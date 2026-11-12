//! The `CHAIN CHECK` transfer line `stela-boot` prints under `--profile`:
//! only values measured on this core. Each transfer cost is also shown as a
//! multiple of this core's own host CALL and RET pair, so the line reads
//! without an outside baseline: a ratio well above the usual few
//! CALL/RET pairs places a cost in this machine's environment rather than
//! in the engine's state.
use alloc::{format, string::String};

/// The line for PE `pe`: each transfer's ns (`transfers`, name and ns), the
/// share of the stamped ring's transfers that stamps sent straight on, each
/// transfer as a multiple of the long `callret` kernel, then each raw
/// kernel's long and burst ns (`raw`, name, long, burst). A ratio is `-`
/// when `callret` was not measured or took no time.
pub fn transfer_line(
    pe: usize,
    transfers: &[(&str, f64)],
    stamped_hits: u64,
    raw: &[(&str, f64, f64)],
) -> String {
    let mut line = format!("CHAIN CHECK pe={pe} ns/transfer");
    for (name, ns) in transfers {
        line.push_str(&format!(" {name}={ns:.2}"));
    }
    line.push_str(&format!(" stamped_hits={stamped_hits}%; x_callret"));
    let callret = raw
        .iter()
        .find(|(name, ..)| *name == "callret")
        .map(|&(_, long, _)| long)
        .filter(|&long| long > 0.0);
    for (name, ns) in transfers {
        match callret {
            Some(callret) => line.push_str(&format!(" {name}={:.1}", ns / callret)),
            None => line.push_str(&format!(" {name}=-")),
        }
    }
    line.push_str("; ns/op long/burst");
    for (name, long, burst) in raw {
        line.push_str(&format!(" {name}={long:.2}/{burst:.2}"));
    }
    line
}

/// The `HOST BRANCH` line every boot prints once, from the boot PE's start:
/// this core's ns per predicted indirect jump (a register jump to the same
/// next instruction each time) and per direct jump, each the best of short
/// bursts, so a bare-metal log shows whether indirect jumps cost what direct
/// ones do there. `ratio` is `-` when the direct jump took no time.
pub fn host_branch_line(indirect_ns: f64, direct_ns: f64) -> String {
    let ratio = if direct_ns > 0.0 {
        format!("{:.1}", indirect_ns / direct_ns)
    } else {
        String::from("-")
    };
    format!("HOST BRANCH indirect_ns={indirect_ns:.2} direct_ns={direct_ns:.2} ratio={ratio}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_branch_names_both_costs_and_their_ratio() {
        assert_eq!(
            host_branch_line(3.08, 0.29),
            "HOST BRANCH indirect_ns=3.08 direct_ns=0.29 ratio=10.6"
        );
        assert_eq!(
            host_branch_line(0.3, 0.0),
            "HOST BRANCH indirect_ns=0.30 direct_ns=0.00 ratio=-"
        );
    }

    #[test]
    fn transfers_are_measured_values_and_multiples_of_callret() {
        let line = transfer_line(
            3,
            &[("near", 1.6), ("cell", 6.4)],
            99,
            &[("callret", 0.8, 0.75), ("jump", 0.3, 0.25)],
        );
        assert_eq!(
            line,
            "CHAIN CHECK pe=3 ns/transfer near=1.60 cell=6.40 stamped_hits=99%; \
             x_callret near=2.0 cell=8.0; ns/op long/burst callret=0.80/0.75 jump=0.30/0.25"
        );
    }

    #[test]
    fn no_callret_gives_no_ratio() {
        let line = transfer_line(0, &[("near", 1.0)], 0, &[("callret", 0.0, 0.0)]);
        assert!(line.contains("x_callret near=-;"), "{line}");
        let line = transfer_line(0, &[("near", 1.0)], 0, &[]);
        assert!(line.contains("x_callret near=-;"), "{line}");
    }
}
