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

const MINIMUM_MEMORY_MIB: u32 = 8192;
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
    usable_mib: Option<u32>,
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
    for memory_mib in [
        4096,
        7168,
        MINIMUM_MEMORY_MIB - 1,
        MINIMUM_MEMORY_MIB,
        12288,
    ] {
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
                    print!(
                        "{memory_mib} MiB, {host_cores} vCPUs: {}",
                        verification.outcome
                    );
                    if let (Some(usable), Some(profile)) =
                        (verification.usable_mib, verification.profile)
                    {
                        print!(" ({usable} MiB usable, {profile})");
                    }
                    println!();
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
    let qemu_returncode = exit.qemu_returncode;
    let fatal = serial.lines().find(|line| line.starts_with("STELA FATAL "));
    let sufficient = memory_mib >= MINIMUM_MEMORY_MIB;
    let admission = if sufficient { "PASS" } else { "FAIL" };
    let prefix = format!("CHECK installed-memory {admission} ");
    ensure!(
        serial.lines().any(|line| {
            line.starts_with(&prefix) && field(line, "minimum") == Some("0x200000000")
        }),
        "{memory_mib} MiB installed, expected the RAM check to report {admission}; exit={qemu_returncode}, fatal={fatal:?}",
    );
    if !sufficient {
        ensure!(
            fatal.is_some()
                && qemu_returncode == REFUSAL_EXIT
                && serial.contains("STELA STOPPED code=0x11;")
                && !serial.contains(KERNEL_BANNER),
            "{memory_mib} MiB installed, expected an early RAM refusal; exit={qemu_returncode}, fatal={fatal:?}",
        );
        return Ok(Verification {
            outcome: "refused as expected",
            usable_mib: None,
            profile: None,
            qemu_returncode,
        });
    }
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
    ensure!(
        profile == Some("standard"),
        "{usable_mib} MiB usable selected {profile:?}, expected standard; fatal={fatal:?}",
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
        usable_mib: Some(usable_mib),
        profile: Some("standard"),
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
    fn refusal_requires_the_installed_memory_marker_and_fatal_exit() {
        let log = concat!(
            "CHECK installed-memory FAIL installed=0x1fff00000 minimum=0x200000000\n",
            "STELA FATAL at least 8 GiB of installed RAM is required\n",
            "STELA STOPPED code=0x11; power off to continue\n",
        );
        assert!(verify_two_cores(8191, exit(REFUSAL_EXIT), log).is_ok());
        assert!(verify_two_cores(8191, exit(FAILED_EXIT), log).is_err());
        assert!(verify_two_cores(8192, exit(REFUSAL_EXIT), log).is_err());
        assert!(
            verify_two_cores(
                8191,
                exit(REFUSAL_EXIT),
                &log.replace("0x200000000", "0x80000000")
            )
            .is_err()
        );
        assert!(
            verify_two_cores(8191, exit(REFUSAL_EXIT), &format!("{log}{KERNEL_BANNER}\n")).is_err()
        );
    }

    #[test]
    fn admission_needs_the_full_kernel_and_the_standard_control() {
        let mut log = format!(
            "CHECK installed-memory PASS installed=0x200000000 minimum=0x200000000\n\
             HOST MEMORY free=8060M cores=1\nHOST MEMORY free=8060M profile=standard\n\
             NATIVE INVENTORY firmware_processors=2 physical_cores=2 requested=1\n\
             {KERNEL_BANNER}\nSTOP idle\n"
        );
        assert!(verify_two_cores(8192, exit(IDLE_EXIT), &log).is_err());
        for (index, expected) in KERNEL_RESULTS {
            writeln!(log, "X{index}: {:#x}", expected.unwrap_or(1)).unwrap();
        }
        assert!(verify_two_cores(8192, exit(IDLE_EXIT), &log).is_ok());
        assert!(verify(8192, 4, exit(IDLE_EXIT), &log).is_err());
        assert!(
            verify_two_cores(8192, exit(IDLE_EXIT), &log.replace("standard", "compact")).is_err()
        );
    }
}
