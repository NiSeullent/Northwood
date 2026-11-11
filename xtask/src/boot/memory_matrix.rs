// SPDX-License-Identifier: GPL-2.0-or-later

//! Reproduce the memory admission checks with the system-kernel fixture.
use super::{IDLE_EXIT, KERNEL_BANNER, MAX_TIMEOUT_SECONDS, Run, binding, launch, verify_kernel};
use anyhow::{Context, Result, ensure};
use clap::{Args, FromArgMatches};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    process::ExitCode,
};

const LOW_MEMORY_MIB: u32 = 2048;
const STANDARD_MEMORY_MIB: u32 = 8192;
const REFUSAL_EXIT: u8 = (0x11 << 1) | 1;

#[derive(Args)]
pub struct Arguments {
    /// Optimized EFI image to test. Every case uses the same copy.
    #[arg(long)]
    image: PathBuf,
    /// Fresh directory for per-case logs and matrix.json.
    #[arg(long)]
    output: PathBuf,
    /// Per-case timeout in seconds.
    #[arg(long, default_value_t = MAX_TIMEOUT_SECONDS)]
    timeout: f64,
}

#[derive(Deserialize)]
struct QemuExit {
    qemu_returncode: u8,
    timed_out: bool,
}

#[derive(Debug, Serialize)]
struct Verification {
    outcome: &'static str,
    usable_mib: u32,
    profile: Option<&'static str>,
    qemu_returncode: u8,
}

pub(super) fn run(arguments: Arguments) -> Result<ExitCode> {
    ensure!(
        arguments.timeout > 0.0 && arguments.timeout <= MAX_TIMEOUT_SECONDS,
        "timeout must be in (0, {MAX_TIMEOUT_SECONDS}] seconds"
    );
    ensure!(
        arguments.image.is_file(),
        "the supplied EFI image is not a file"
    );
    fs::create_dir(&arguments.output)
        .with_context(|| format!("create {}", arguments.output.display()))?;
    // The original may be replaced by another build before this matrix ends.
    let image = arguments.output.join("input.efi");
    fs::copy(&arguments.image, &image)?;
    let image_binding = binding(&image)?;
    let mut rows = Vec::new();
    let mut failures = 0;
    for memory_mib in [LOW_MEMORY_MIB, 2560, 3072, 4096, STANDARD_MEMORY_MIB] {
        for host_cores in [2, 4] {
            let directory = format!("{memory_mib}m-{host_cores}c");
            let output = arguments.output.join(&directory);
            let result = fixture_run(&image, &output, memory_mib, host_cores, arguments.timeout);
            let mut row = json!({
                "memory_mib": memory_mib,
                "host_cores": host_cores,
                "directory": directory,
                "passed": result.is_ok(),
            });
            match result {
                Ok(verification) => {
                    println!(
                        "{memory_mib} MiB, {host_cores} vCPUs: {} ({} MiB usable{})",
                        verification.outcome,
                        verification.usable_mib,
                        verification
                            .profile
                            .map_or("".to_owned(), |p| format!(", {p}")),
                    );
                    row["verification"] = json!(verification);
                }
                Err(error) => {
                    failures += 1;
                    eprintln!("{memory_mib} MiB, {host_cores} vCPUs: {error:#}");
                    row["error"] = json!(format!("{error:#}"));
                }
            }
            rows.push(row);
        }
    }
    let matrix = json!({
        "image": image_binding,
        "fixture": "system-kernel (one guest worker)",
        "passed": failures == 0,
        "runs": rows,
    });
    let report = arguments.output.join("matrix.json");
    fs::write(&report, format!("{matrix:#}\n"))?;
    ensure!(
        failures == 0,
        "{failures} matrix cases failed, see {}",
        report.display()
    );
    Ok(ExitCode::SUCCESS)
}

fn fixture_run(
    image: &Path,
    output: &Path,
    memory_mib: u32,
    host_cores: u8,
    timeout: f64,
) -> Result<Verification> {
    // Reuse the normal boot command's defaults, including the authored ARM
    // image, serial check, firmware search and debug-exit device.
    let matches = Run::augment_args(clap::Command::new("boot-run")).try_get_matches_from([
        OsString::from("boot-run"),
        "--image".into(),
        image.as_os_str().into(),
        "--output".into(),
        output.as_os_str().into(),
        "--memory".into(),
        format!("{memory_mib}M").into(),
        "--host-cores".into(),
        host_cores.to_string().into(),
        "--timeout".into(),
        timeout.to_string().into(),
    ])?;
    launch(Run::from_arg_matches(&matches)?, false)?;
    let exit: QemuExit = serde_json::from_slice(&fs::read(output.join("run.json"))?)?;
    let serial = fs::read_to_string(output.join("serial.txt"))?;
    verify(memory_mib, host_cores, exit, &serial)
}

fn field<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    line.split_ascii_whitespace().find_map(|field| {
        let (name, value) = field.split_once('=')?;
        (name == key).then_some(value)
    })
}

fn verify(memory_mib: u32, host_cores: u8, exit: QemuExit, serial: &str) -> Result<Verification> {
    ensure!(!exit.timed_out, "QEMU timed out");
    let memory = serial
        .lines()
        .filter_map(|line| line.strip_prefix("HOST MEMORY "));
    let usable_mib = memory
        .clone()
        .find(|line| field(line, "cores").is_some())
        .and_then(|line| field(line, "free"))
        .and_then(|free| free.strip_suffix('M'))
        .context("missing measured usable RAM before reservation")?
        .parse::<u32>()
        .context("invalid measured usable RAM")?;
    ensure!(
        usable_mib <= memory_mib,
        "reported {usable_mib} MiB usable in a {memory_mib} MiB VM"
    );
    let profile = memory.filter_map(|line| field(line, "profile")).last();
    let fatal = serial.lines().find(|line| line.starts_with("STELA FATAL "));
    let qemu_returncode = exit.qemu_returncode;
    if usable_mib < LOW_MEMORY_MIB {
        ensure!(
            memory_mib == LOW_MEMORY_MIB,
            "only {usable_mib} MiB of RAM usable"
        );
        let refusal = format!(
            "STELA FATAL this PC has {}.{} GB of RAM free; Stela needs at least 2 GB free to start the guest",
            usable_mib / 1024,
            usable_mib % 1024 * 10 / 1024,
        );
        ensure!(
            fatal == Some(refusal.as_str())
                && qemu_returncode == REFUSAL_EXIT
                && profile.is_none()
                && serial.contains("STELA STOPPED code=0x11;")
                && !serial.contains(KERNEL_BANNER),
            "{usable_mib} MiB usable, expected a pre-guest RAM refusal; exit={qemu_returncode}, fatal={fatal:?}",
        );
        return Ok(Verification {
            outcome: "refused as expected",
            usable_mib,
            profile: None,
            qemu_returncode,
        });
    }
    let profile = match profile {
        Some("standard") => "standard",
        Some("compact") => "compact",
        Some("paged") => "paged",
        _ => anyhow::bail!("no recognized memory profile at {usable_mib} MiB; fatal={fatal:?}"),
    };
    ensure!(
        memory_mib != STANDARD_MEMORY_MIB || profile == "standard",
        "8 GiB VM selected {profile}, expected standard"
    );
    ensure!(
        usable_mib > 3072 || profile == "paged",
        "{usable_mib} MiB usable selected {profile}, expected paged"
    );
    ensure!(
        qemu_returncode == IDLE_EXIT && fatal.is_none() && !serial.contains("STELA PANIC "),
        "expected fixture idle exit {IDLE_EXIT}, got {qemu_returncode}; fatal={fatal:?}",
    );
    let cores = serial
        .lines()
        .find_map(|line| line.strip_prefix("NATIVE INVENTORY "))
        .and_then(|line| field(line, "physical_cores"))
        .and_then(|count| count.parse::<u8>().ok());
    ensure!(
        cores == Some(host_cores),
        "expected {host_cores} firmware-visible cores, got {cores:?}"
    );
    verify_kernel(serial).map_err(anyhow::Error::msg)?;
    Ok(Verification {
        outcome: "fixture passed",
        usable_mib,
        profile: Some(profile),
        qemu_returncode,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boot::{FAILED_EXIT, KERNEL_RESULTS};
    use std::fmt::Write;

    fn exit(code: u8) -> QemuExit {
        QemuExit {
            qemu_returncode: code,
            timed_out: false,
        }
    }

    fn verify_two_cores(memory_mib: u32, exit: QemuExit, serial: &str) -> Result<Verification> {
        verify(memory_mib, 2, exit, serial)
    }

    #[test]
    fn refusal_requires_the_measured_floor_and_the_specific_fatal() {
        let log = concat!(
            "HOST MEMORY free=1920M before_reclaim=1800M cores=1\n",
            "STELA FATAL this PC has 1.8 GB of RAM free; Stela needs at least 2 GB free to start the guest\n",
            "STELA STOPPED code=0x11; power off to continue\n",
        );
        assert!(verify_two_cores(2048, exit(REFUSAL_EXIT), log).is_ok());
        assert!(verify_two_cores(2048, exit(FAILED_EXIT), log).is_err());
        assert!(verify_two_cores(2560, exit(REFUSAL_EXIT), log).is_err());
        assert!(
            verify_two_cores(2048, exit(REFUSAL_EXIT), &log.replace("1920M", "2048M")).is_err()
        );
        assert!(
            verify_two_cores(
                2048,
                exit(REFUSAL_EXIT),
                &log.replace("RAM free", "disk free")
            )
            .is_err()
        );
    }

    #[test]
    fn admission_needs_the_full_kernel_and_the_standard_control() {
        let mut log = format!(
            "HOST MEMORY free=2048M cores=1\nHOST MEMORY free=2048M profile=paged\n\
             NATIVE INVENTORY firmware_processors=2 physical_cores=2 requested=1\n\
             {KERNEL_BANNER}\nSTOP idle\n"
        );
        assert!(verify_two_cores(2048, exit(IDLE_EXIT), &log).is_err());
        for (index, expected) in KERNEL_RESULTS {
            writeln!(log, "X{index}: {:#x}", expected.unwrap_or(1)).unwrap();
        }
        assert!(verify_two_cores(2048, exit(IDLE_EXIT), &log).is_ok());
        assert!(verify(2048, 4, exit(IDLE_EXIT), &log).is_err());
        assert!(verify_two_cores(2048, exit(IDLE_EXIT), &log.replace("paged", "compact")).is_err());
        let standard = log.replace("2048M", "8060M");
        assert!(verify_two_cores(8192, exit(IDLE_EXIT), &standard).is_err());
        assert!(
            verify_two_cores(
                8192,
                exit(IDLE_EXIT),
                &standard.replace("paged", "standard")
            )
            .is_ok()
        );
    }
}
