// SPDX-License-Identifier: GPL-2.0-or-later

//! Host-side measurement extraction. Report order follows the log and
//! each run is kept separate, even when two logs reuse the same boot number.
use super::{boot_number, timestamp};
use std::{
    fmt::Display,
    io::{self, BufRead, Write},
};

#[cfg(test)]
mod tests;

pub(super) fn csv_header(output: &mut impl Write) -> io::Result<()> {
    writeln!(
        output,
        "source,run,boot,line,report,section,metric,value,unit"
    )
}

pub(super) fn write_report(
    mut input: impl BufRead,
    output: &mut impl Write,
    source: &str,
    overwritten: bool,
    csv: bool,
) -> io::Result<()> {
    if !csv {
        writeln!(output, "Performance summary for {source}")?;
        writeln!(
            output,
            "Run | Boot | Profile | Free/guest MiB | Beats | Compiled | Mean instr/s | \
             Peak sampled engine MiB | Service gap max us | Beat step max us | Capture"
        )?;
    }
    let source = source.replace('"', "\"\"");
    let mut run = 1;
    let mut stats = Stats {
        partial: overwritten,
        ..Stats::default()
    };
    let mut bytes = Vec::new();
    let mut line_number = 0;
    // A wrapped ring can begin partway through a line. Discard that line
    // even if its suffix resembles one of the reports.
    if overwritten && read_line(&mut input, &mut bytes)?.is_some() {
        line_number += 1;
    }
    while let Some(complete) = read_line(&mut input, &mut bytes)? {
        line_number += 1;
        let Ok(line) = std::str::from_utf8(&bytes) else {
            stats.partial = true;
            continue;
        };
        if !complete {
            stats.partial = true;
            continue;
        }
        let (stamp, line) = timestamp(line.trim_ascii());
        let boot = boot_number(line);
        let mount = line.starts_with("STELA UEFI MOUNT version=");
        if boot.is_some() || mount {
            // A persistent log writes its numbered marker before the mount
            // banner. In serial output the banner is the boot marker.
            if stats.started() && (boot.is_some() || stats.mount || stats.samples != 0) {
                if !csv {
                    stats.write(output, run)?;
                }
                run += 1;
                stats = Stats::default();
            }
            stats.mount |= mount;
            stats.boot = boot.or(stats.boot);
        } else if line.starts_with("[boot log dropped ") || line.starts_with("[serial: ") {
            stats.partial = true;
        } else if let Some(report) = Report::parse(line, stamp) {
            if let Some(sample) = report.sample() {
                stats.observe(report, &sample);
                if csv {
                    // Keep file order. HOST SERVICE also reports at shutdown,
                    // so a row's line is more precise than a heartbeat index.
                    let boot = stats
                        .boot
                        .map(|number| number.to_string())
                        .unwrap_or_default();
                    for metric in sample {
                        writeln!(
                            output,
                            "\"{source}\",{run},{boot},{line_number},{},{},{},{},{}",
                            report.name, metric.section, metric.name, metric.value, metric.unit,
                        )?;
                    }
                }
            } else {
                stats.partial = true;
            }
        }
    }
    if !csv {
        stats.write(output, run)?;
    }
    Ok(())
}

// A serial capture can include arbitrary output. Drain an oversized line
// while bounding memory use, and ignore an unterminated tail.
fn read_line(input: &mut impl BufRead, line: &mut Vec<u8>) -> io::Result<Option<bool>> {
    const MAX_LINE_BYTES: usize = 64 << 10;
    line.clear();
    let mut too_long = false;
    loop {
        let available = input.fill_buf()?;
        if available.is_empty() {
            return Ok((too_long || !line.is_empty()).then_some(false));
        }
        let newline = available.iter().position(|&byte| byte == b'\n');
        let length = newline.map_or(available.len(), |at| at + 1);
        too_long |= line.len() + length > MAX_LINE_BYTES;
        if !too_long {
            line.extend_from_slice(&available[..length]);
        }
        input.consume(length);
        if newline.is_some() {
            return Ok(Some(!too_long));
        }
    }
}

#[derive(Default)]
struct Stats {
    mount: bool,
    boot: Option<u64>,
    partial: bool,
    samples: u64,
    profile: Option<String>,
    free: Option<u128>,
    guest: Option<u128>,
    beats: u64,
    compiled: Option<u128>,
    ips_total: u128,
    ips_count: u128,
    engine_peak: Option<u128>,
    service_gap: Option<u128>,
    beat_step: Option<u128>,
}

impl Stats {
    fn started(&self) -> bool {
        self.mount || self.boot.is_some() || self.samples != 0 || self.partial
    }

    fn observe(&mut self, report: Report<'_>, metrics: &[Metric<'_>]) {
        self.samples += 1;
        let number = |name, unit| {
            metrics
                .iter()
                .find(|metric| {
                    metric.name == name && metric.unit == unit && metric.section.is_empty()
                })?
                .value
                .parse::<u128>()
                .ok()
        };
        match report.name {
            "HOST MEMORY" => {
                if let Some(profile) = metrics.iter().find(|metric| metric.name == "profile") {
                    self.profile = Some(profile.value.to_owned());
                }
                self.free = number("free", "MiB").or(self.free);
                self.guest = number("guest", "MiB").or(self.guest);
            }
            "BEAT" => {
                self.beats += 1;
                if let Some(compiled) = number("compiled", "count") {
                    self.compiled = Some(self.compiled.unwrap_or(0).saturating_add(compiled));
                }
            }
            "GUEST PERFORMANCE" => {
                if let Some(ips) = number("instructions_per_second", "instructions/s") {
                    self.ips_total = self.ips_total.saturating_add(ips);
                    self.ips_count += 1;
                }
            }
            "ENGINE MEMORY" => {
                let footprint = [
                    "blocks",
                    "owners",
                    "successors",
                    "rows",
                    "index",
                    "lookup",
                    "pages",
                ]
                .into_iter()
                .try_fold(0u128, |sum, key| {
                    Some(sum.saturating_add(number(key, "MiB")?))
                });
                self.engine_peak = self.engine_peak.max(footprint);
            }
            "HOST SERVICE" => {
                self.service_gap = self.service_gap.max(number("maximum_gap_us", "us"));
            }
            "BEAT COST" => {
                self.beat_step = self.beat_step.max(number("longest_step", "us"));
            }
            "BOOT STAGE" | "HOST CLOCK" | "PES CODE" | "SPI NOR PROFILE" => {}
            _ => unreachable!("recognized report"),
        }
    }

    fn write(&self, output: &mut impl Write, run: u64) -> io::Result<()> {
        let capture = match (self.partial, self.samples) {
            (true, 0) => "partial, no samples",
            (true, _) => "partial",
            (false, 0) => "no samples",
            (false, _) => "samples present",
        };
        let beats = (self.beats != 0).then_some(self.beats);
        writeln!(
            output,
            "{run} | {} | {} | {}/{} | {} | {} | {} | {} | {} | {} | {capture}",
            shown(self.boot),
            self.profile.as_deref().unwrap_or("-"),
            shown(self.free),
            shown(self.guest),
            shown(beats),
            shown(self.compiled),
            shown(self.ips_total.checked_div(self.ips_count)),
            shown(self.engine_peak),
            shown(self.service_gap),
            shown(self.beat_step),
        )
    }
}

fn shown(value: Option<impl Display>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "-".into())
}

#[derive(Clone, Copy)]
struct Report<'a> {
    name: &'static str,
    fields: &'a str,
    timestamp: Option<&'a str>,
}

impl<'a> Report<'a> {
    fn parse(line: &'a str, timestamp: Option<&'a str>) -> Option<Self> {
        let prefixes = [
            ("BEAT COST", "BEAT COST us "),
            ("BOOT STAGE", "BOOT STAGE "),
            ("HOST MEMORY", "HOST MEMORY "),
            ("GUEST PERFORMANCE", "GUEST PERFORMANCE "),
            ("ENGINE MEMORY", "ENGINE MEMORY "),
            ("HOST CLOCK", "HOST CLOCK "),
            ("HOST SERVICE", "HOST SERVICE "),
            ("PES CODE", "PES CODE live/slots/pages/releases "),
            ("SPI NOR PROFILE", "SPI NOR PROFILE "),
            ("BEAT", "BEAT "),
        ];
        prefixes.into_iter().find_map(|(name, prefix)| {
            let fields = line.strip_prefix(prefix)?;
            if name == "PES CODE"
                && !fields
                    .split_ascii_whitespace()
                    .any(|word| word.starts_with("board="))
            {
                return None;
            }
            Some(Self {
                name,
                fields,
                timestamp,
            })
        })
    }

    fn sample(self) -> Option<Vec<Metric<'a>>> {
        // BEAT can precede other prose. Confirm the numeric anchor so it
        // cannot turn an unrelated line into a heartbeat.
        let (section, anchor, unit) = match self.name {
            "HOST MEMORY" if self.fields.starts_with("heap_contiguous_unavailable=") => {
                ("", "heap_contiguous_unavailable", "MiB")
            }
            "HOST MEMORY" => ("", "free", "MiB"),
            "BEAT" => ("", "blocks", "count"),
            "GUEST PERFORMANCE" => ("", "retired", "count"),
            "ENGINE MEMORY" => ("", "pe", "count"),
            "HOST SERVICE" => ("", "boundaries", "count"),
            "BEAT COST" => ("", "longest_step", "us"),
            "HOST CLOCK" => ("", "mperf_delta", "count"),
            "BOOT STAGE" => ("", "name", "text"),
            "PES CODE" => ("board", "generation", "count"),
            "SPI NOR PROFILE" => ("", "counter", "hex"),
            _ => unreachable!("recognized report"),
        };
        let metrics = self.metrics();
        metrics
            .iter()
            .any(|metric| metric.name == anchor && metric.unit == unit && metric.section == section)
            .then_some(metrics)
    }

    fn metrics(self) -> Vec<Metric<'a>> {
        let mut metrics = Vec::new();
        if let Some(value) = self.timestamp {
            metrics.push(Metric {
                section: "",
                name: "timestamp",
                value,
                unit: "s",
            });
        }
        let mut section = "";
        for word in self.fields.split_ascii_whitespace() {
            if self.name == "HOST SERVICE" {
                section = match (word, section) {
                    ("services", _) => "drivers",
                    ("longest_us", "drivers") => "driver_longest",
                    ("total_us", "driver_longest") => "driver_total",
                    ("longest_us", _) => "longest",
                    ("total_us", _) => "total",
                    _ => section,
                };
            }
            let Some((name, mut value)) = word.split_once('=') else {
                continue;
            };
            if self.name == "PES CODE" {
                if let Some((section, tuple, names)) = board_tuple(name, value) {
                    append_board_tuple(&mut metrics, section, tuple, names);
                }
                continue;
            }
            if let Some((section, tuple, names)) = tuple(name, value) {
                metrics.extend(
                    tuple
                        .split('/')
                        .zip(names)
                        .filter_map(|(value, &(name, unit))| field(section, name, value, unit)),
                );
                continue;
            }
            if name == "discards"
                && let Some((count, causes)) = value.split_once('(')
            {
                value = count;
                metrics.extend(causes.strip_suffix(')').into_iter().flat_map(|causes| {
                    causes.split(',').filter_map(|cause| {
                        let (name, value) = cause.split_once('=')?;
                        field("discard_cause", name, value, "count")
                    })
                }));
            }
            let section = if self.name == "HOST SERVICE" && name.starts_with("input_") {
                "input"
            } else {
                section
            };
            if name == "host_drivers_held_us"
                && let Some((elapsed, _site)) = value.split_once('@')
            {
                value = elapsed;
            }
            metrics.extend(field(section, name, value, self.unit(section, name)));
        }
        metrics
    }

    fn unit(self, section: &str, name: &str) -> &'static str {
        if name.ends_with("_us")
            || matches!(
                section,
                "longest" | "total" | "driver_longest" | "driver_total"
            )
            || self.name == "BEAT COST" && name != "steps"
        {
            "us"
        } else if name.ends_with("_ms") {
            "ms"
        } else if self.name == "SPI NOR PROFILE" && name.ends_with("_bytes") {
            "bytes"
        } else {
            match name {
                "profile" | "name" | "cpu" => "text",
                "active_mhz" => "MHz",
                "ratio_milli" => "milli-ratio",
                "instructions_per_second" => "instructions/s",
                "instructions_per_block_milli" => "milli-instructions/block",
                "ticks/block" => "ticks/block",
                _ => "count",
            }
        }
    }
}

#[derive(Clone, Copy)]
struct Metric<'a> {
    section: &'static str,
    name: &'a str,
    value: &'a str,
    unit: &'static str,
}

type TupleFields = &'static [(&'static str, &'static str)];

fn board_tuple<'a>(name: &str, value: &'a str) -> Option<(&'static str, &'a str, TupleFields)> {
    Some(match name {
        "board" => (
            "board",
            value,
            &[
                ("window", "MiB"),
                ("parts", "count"),
                ("generation", "count"),
            ],
        ),
        "churn" => (
            "churn",
            value.strip_prefix("new/ring/ivau/slot/other:")?,
            &[
                ("new", "count"),
                ("ring", "count"),
                ("ivau", "count"),
                ("slot", "count"),
                ("other", "count"),
            ],
        ),
        "lock" => (
            "lock",
            value.strip_prefix("acquired/contended/waited_us/longest_us:")?,
            &[
                ("acquired", "count"),
                ("contended", "count"),
                ("waited", "us"),
                ("longest", "us"),
            ],
        ),
        _ => return None,
    })
}

fn append_board_tuple<'a>(
    metrics: &mut Vec<Metric<'a>>,
    section: &'static str,
    values: &'a str,
    names: TupleFields,
) {
    let start = metrics.len();
    let mut values = values.split('/');
    for &(name, unit) in names {
        let Some(metric) = values
            .next()
            .and_then(|value| field(section, name, value, unit))
        else {
            break;
        };
        if metric.unit != unit {
            break;
        }
        metrics.push(metric);
    }
    if metrics.len() != start + names.len() || values.next().is_some() {
        metrics.truncate(start);
    }
}

fn tuple<'a>(name: &'a str, value: &'a str) -> Option<(&'static str, &'a str, TupleFields)> {
    Some(match name {
        "heap" if value.contains('/') => ("heap", value, &[("live", "MiB"), ("total", "MiB")]),
        "shadow_peak" => (
            "shadow",
            value,
            &[
                ("lower_peak", "MiB"),
                ("upper_peak", "MiB"),
                ("pool", "MiB"),
            ],
        ),
        "code_peak" => ("code", value, &[("peak", "MiB"), ("window", "MiB")]),
        "loops" => (
            "loops",
            value,
            &[
                ("compiled", "count"),
                ("entries", "count"),
                ("extra_retired", "count"),
            ],
        ),
        "gap" => (
            "gap",
            value.strip_prefix("host_us/slice_us/kind:")?,
            &[("host", "us"), ("slice", "us"), ("kind", "text")],
        ),
        "host_drivers_waited_us" => ("driver_wait", value, &[("total", "us"), ("longest", "us")]),
        _ => return None,
    })
}

fn field<'a>(
    section: &'static str,
    name: &'a str,
    value: &'a str,
    unit: &'static str,
) -> Option<Metric<'a>> {
    if name.is_empty()
        || !name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"_/-".contains(&byte))
    {
        return None;
    }
    let (value, unit) = if unit == "text" {
        if value.is_empty()
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || b"_-".contains(&byte))
        {
            return None;
        }
        (value, unit)
    } else if let Some(value) = value.strip_suffix('M') {
        (value, "MiB")
    } else if let Some(value) = value.strip_suffix('%') {
        (value, "percent")
    } else if let Some(hex) = value.strip_prefix("0x") {
        u128::from_str_radix(hex, 16).ok()?;
        return Some(Metric {
            section,
            name,
            value,
            unit: "hex",
        });
    } else {
        (value, unit)
    };
    if unit != "text" {
        value.parse::<u128>().ok()?;
    }
    Some(Metric {
        section,
        name,
        value,
        unit,
    })
}
