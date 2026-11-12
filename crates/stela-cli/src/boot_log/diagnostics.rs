// SPDX-License-Identifier: GPL-2.0-or-later

//! Memory and swap observations from a single boot in a persisted USB log.
use super::{boot_number, timestamp};

#[derive(Default)]
pub(crate) struct Diagnostics {
    pub(crate) usable_mib: Option<u64>,
    pub(crate) minimum_mib: Option<u64>,
    pub(crate) profile: Option<&'static str>,
    pub(crate) estimate: bool,
    pub(crate) memory_refused: bool,
    pub(crate) swap: Option<String>,
    pub(crate) swap_refused: bool,
}

impl Diagnostics {
    pub(crate) fn latest(bytes: &[u8], overwritten: bool) -> Self {
        let mut diagnostics = Self::default();
        let mut saw_mount = false;
        for bytes in bytes
            .split_inclusive(|&byte| byte == b'\n')
            .skip(usize::from(overwritten))
        {
            if !bytes.ends_with(b"\n") {
                break;
            }
            let Ok(text) = std::str::from_utf8(bytes) else {
                continue;
            };
            let (_, line) = timestamp(text.trim_ascii());
            if boot_number(line).is_some() {
                diagnostics = Self::default();
                saw_mount = false;
            } else if line.starts_with("STELA UEFI MOUNT version=") {
                if saw_mount || diagnostics.usable_mib.is_some() || diagnostics.swap.is_some() {
                    diagnostics = Self::default();
                }
                saw_mount = true;
            } else {
                diagnostics.observe(line);
            }
        }
        diagnostics
    }

    fn observe(&mut self, line: &str) {
        if let Some(memory) = line.strip_prefix("HOST MEMORY ") {
            if let Some(free) = field(memory, "free")
                .and_then(|s| s.strip_suffix('M'))
                .and_then(decimal)
            {
                self.usable_mib = Some(free);
                self.estimate = false;
            }
            if let Some(profile) = field(memory, "profile").and_then(profile) {
                self.profile = Some(profile);
            }
        } else if let Some(memory) = line.strip_prefix("CHECK memory ") {
            self.memory_check(memory);
        } else if let Some(reason) = line.strip_prefix("STELA FATAL ")
            && ((reason.contains("of RAM free") && reason.contains("needs at least"))
                || reason.starts_with("no contiguous firmware RAM extent"))
        {
            self.memory_refused = true;
            self.profile = None;
        } else if let Some(swap) = line.strip_prefix("HOST SWAP ") {
            self.runtime_swap(swap);
        } else if let Some(swap) = line.strip_prefix("CHECK swap ") {
            self.swap_check(swap);
        }
    }

    fn memory_check(&mut self, text: &str) {
        let Some((verdict @ ("PASS" | "FAIL"), fields)) = text.split_once(' ') else {
            return;
        };
        if let Some(estimate) = field(fields, "estimate") {
            // Standalone Diagnostics measures before loading the guest.
            self.estimate = true;
            self.usable_mib = field(fields, "usable").and_then(decimal);
            self.minimum_mib = field(fields, "minimum").and_then(decimal);
            self.profile = profile(estimate);
        } else {
            self.usable_mib = field(fields, "usable")
                .and_then(hex_mib)
                .or(self.usable_mib);
            self.minimum_mib = field(fields, "minimum").and_then(hex_mib);
            self.profile = field(fields, "profile").and_then(profile).or(self.profile);
        }
        self.memory_refused = verdict == "FAIL";
        if self.memory_refused {
            self.profile = None;
        }
    }

    fn runtime_swap(&mut self, text: &str) {
        if let Some(location) = field(text, "candidate").and_then(swap_location) {
            let capacity = field(text, "bytes")
                .and_then(decimal)
                .map(|bytes| bytes >> 20);
            let suffix = if field(text, "mode") == Some("resident") {
                "; resident mode"
            } else {
                ""
            };
            self.swap = Some(store(
                &format!("{location} candidate verified"),
                capacity,
                suffix,
            ));
            self.swap_refused = false;
        } else if text.starts_with("no verified store") {
            let label = if self.swap_refused {
                "store refused"
            } else {
                "no verified store"
            };
            let suffix = if field(text, "mode") == Some("resident") {
                "; resident mode"
            } else {
                ""
            };
            self.swap = Some(format!("{label}{suffix}"));
        } else if text.starts_with("refused ")
            || text.starts_with("GPT candidate refused:")
            || text.starts_with("volume candidate refused:")
        {
            self.swap = Some("store refused".into());
            self.swap_refused = true;
        }
    }

    fn swap_check(&mut self, text: &str) {
        if let Some(details) = text.strip_prefix("PASS ") {
            let (location, capacity) = if details.starts_with("partition=") {
                ("partition", field(details, "usable"))
            } else if details.starts_with("boot data volume=") {
                ("data volume", field(details, "capacity"))
            } else {
                return;
            };
            self.swap = Some(store(
                &format!("{location} verified"),
                capacity.and_then(decimal),
                " (read-only diagnostic)",
            ));
            self.swap_refused = false;
        } else if text.starts_with("FAIL ") {
            self.swap = Some("store refused by diagnostic check".into());
            self.swap_refused = true;
        } else if let Some(reason) = text.strip_prefix("SKIP ") {
            self.swap = Some(if reason.starts_with("no prepared swap ") {
                "no prepared store found by diagnostic check".into()
            } else {
                "store not checked by Diagnostics".into()
            });
            self.swap_refused = false;
        }
    }
}

fn field<'a>(text: &'a str, name: &str) -> Option<&'a str> {
    text.split_ascii_whitespace()
        .filter_map(|part| part.split_once('='))
        .find_map(|(key, value)| (key == name).then(|| value.trim_end_matches(',')))
}

fn decimal(value: &str) -> Option<u64> {
    value.parse().ok()
}

fn hex_mib(value: &str) -> Option<u64> {
    u64::from_str_radix(value.strip_prefix("0x")?, 16)
        .ok()
        .map(|bytes| bytes >> 20)
}

fn profile(name: &str) -> Option<&'static str> {
    match name {
        "standard" => Some("Standard"),
        "compact" => Some("Compact"),
        "paged" => Some("Paged"),
        _ => None,
    }
}

fn swap_location(name: &str) -> Option<&'static str> {
    match name {
        "partition" => Some("partition"),
        "volume" => Some("data volume"),
        _ => None,
    }
}

fn store(label: &str, capacity: Option<u64>, suffix: &str) -> String {
    match capacity {
        Some(capacity) => format!("{label} ({capacity} MiB){suffix}"),
        None => format!("{label}{suffix}"),
    }
}

#[cfg(test)]
mod tests {
    use super::Diagnostics;

    #[test]
    fn a_new_boot_never_inherits_old_memory_or_swap() {
        let log = b"=== STELA BOOT 1 ===\nSTELA UEFI MOUNT version=0.1.0\n\
            HOST MEMORY free=2406M profile=paged\n\
            HOST SWAP candidate=partition bytes=34359738368 mode=resident\n\
            === STELA BOOT 2 ===\nSTELA UEFI MOUNT version=0.1.0\n";
        let report = Diagnostics::latest(log, false);
        assert_eq!(report.usable_mib, None);
        assert_eq!(report.profile, None);
        assert_eq!(report.swap, None);
    }

    #[test]
    fn memory_check_and_resident_candidate_are_separate_observations() {
        let log = b"[ 12.800] HOST MEMORY free=2406M profile=paged\r\n\
            [ 12.801] CHECK memory PASS usable=0x96600000 profile=paged cores=4\r\n\
            [ 15.100] HOST SWAP GPT candidate refused: bad header\r\n\
            [ 16.100] HOST SWAP candidate=volume bytes=34359738368 mode=resident\r\n";
        let report = Diagnostics::latest(log, false);
        assert_eq!(report.usable_mib, Some(2406));
        assert_eq!(report.profile, Some("Paged"));
        assert!(!report.estimate);
        assert_eq!(
            report.swap.as_deref(),
            Some("data volume candidate verified (32768 MiB); resident mode")
        );
        assert!(!report.swap_refused);
    }

    #[test]
    fn reports_diagnostic_estimate_and_read_only_swap_check() {
        let log = b"STELA UEFI MOUNT version=0.1.0\n\
            CHECK memory PASS usable=2364 MiB, minimum=2048 MiB, one-worker estimate=paged\n\
            CHECK swap PASS partition=example usable=14 MiB (read only)\n";
        let report = Diagnostics::latest(log, false);
        assert_eq!(report.usable_mib, Some(2364));
        assert_eq!(report.minimum_mib, Some(2048));
        assert_eq!(report.profile, Some("Paged"));
        assert!(report.estimate);
        assert_eq!(
            report.swap.as_deref(),
            Some("partition verified (14 MiB) (read-only diagnostic)")
        );
    }

    #[test]
    fn ignores_wrapped_prefix_invalid_utf8_and_incomplete_tail() {
        let log = b"HOST MEMORY free=8000M profile=standard\n\
            HOST MEMORY free=1446M before_reclaim=1344M\n\
            CHECK memory FAIL usable=0x5a600000 minimum=0x80000000 cores=1\n\
            HOST SWAP candidate=\xff\n\
            HOST MEMORY free=8000M profile=standard";
        let report = Diagnostics::latest(log, true);
        assert_eq!(report.usable_mib, Some(1446));
        assert_eq!(report.minimum_mib, Some(2048));
        assert_eq!(report.profile, None);
        assert!(report.memory_refused);
        assert_eq!(report.swap, None);
    }

    #[test]
    fn recognizes_older_memory_refusals_without_a_check_record() {
        let log = b"HOST MEMORY free=1914M cores=2\n\
            STELA FATAL this PC has 1.8 GB of RAM free; Stela needs at least 2 GB free to start the guest\n";
        let report = Diagnostics::latest(log, false);
        assert_eq!(report.usable_mib, Some(1914));
        assert_eq!(report.profile, None);
        assert!(report.memory_refused);
    }
}
