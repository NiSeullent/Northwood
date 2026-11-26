// SPDX-License-Identifier: GPL-2.0-or-later
//! Verify active host workers with the authored shared ARM kernel and keep
//! emulated timings separate from hardware runs.

use super::{IDLE_EXIT, KERNEL_OPTIONS, MAX_TIMEOUT_SECONDS, Run, binding, kernel, launch};
use anyhow::{Context, Result, ensure};
use clap::{Args, FromArgMatches};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    collections::{BTreeMap, BTreeSet},
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    process::ExitCode,
};

const PHYSICAL_CORES: u16 = 32;
const LOGICAL_CPUS: u16 = 2 * PHYSICAL_CORES;

#[derive(Args)]
pub struct Arguments {
    /// Northwood EFI built with the diagnostics feature.
    #[arg(long)]
    image: PathBuf,
    /// Fresh directory for serial logs, run reports and the matrix report.
    #[arg(long)]
    output: PathBuf,
    /// Native worker counts to try on each 32-core, 64-thread QEMU topology.
    #[arg(long, value_delimiter = ',', default_value = "32,64", value_parser = clap::value_parser!(u16).range(2..=64))]
    workers: Vec<u16>,
    /// QEMU RAM per run.
    #[arg(long, default_value = "12G")]
    memory: String,
    /// Rebuild the VM and run each configuration this many times.
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u8).range(1..=20))]
    repeats: u8,
    /// Run with KVM and the host CPU. TCG timings describe emulation.
    #[arg(long)]
    kvm: bool,
    /// Require all eligible APs to complete the translation-proof workload.
    #[arg(long)]
    require_translation_proof: bool,
    /// Per-run timeout in seconds.
    #[arg(long, default_value_t = MAX_TIMEOUT_SECONDS)]
    timeout: f64,
}

#[derive(Clone, Copy, Serialize)]
struct Case {
    sockets: u16,
    workers: u16,
}

#[derive(Deserialize)]
struct QemuRun {
    qemu_returncode: u8,
    timed_out: bool,
    wall_seconds: f64,
    qemu_version: String,
}

#[derive(Debug, Serialize)]
struct Measurement {
    scheduler_ticks: u64,
    tsc_hz: u64,
    scheduled_instructions: u64,
    scheduler_seconds: f64,
    guest_instructions_per_second: f64,
}

#[derive(Debug, Serialize)]
struct TranslationProof {
    eligible_workers: u64,
    completed_workers: u64,
    proven_blocks: u64,
    elapsed_ms: u64,
}

#[derive(Debug, Serialize)]
struct Evidence {
    active_workers: u16,
    packages_used: usize,
    cores_used: usize,
    smt_siblings_used: usize,
    atomic_updates: u64,
    scheduler_rounds: u64,
    total_retired: u64,
    measurement: Option<Measurement>,
    translation_proof: Option<TranslationProof>,
}

#[derive(Serialize)]
struct Sample {
    qemu_version: String,
    boot_and_fixture_seconds: f64,
    evidence: Evidence,
}

pub(super) fn run(arguments: Arguments) -> Result<ExitCode> {
    ensure!(
        arguments.timeout > 0.0 && arguments.timeout <= MAX_TIMEOUT_SECONDS,
        "timeout must be in (0, {MAX_TIMEOUT_SECONDS}] seconds"
    );
    ensure!(arguments.image.is_file(), "EFI image must be a file");
    let worker_counts = arguments.workers.iter().copied().collect::<BTreeSet<_>>();
    ensure!(
        worker_counts.len() == arguments.workers.len(),
        "worker counts must be distinct"
    );
    fs::create_dir(&arguments.output)
        .with_context(|| format!("create {}", arguments.output.display()))?;
    let image = arguments.output.join("input.efi");
    fs::copy(&arguments.image, &image)?;
    let module = arguments.output.join("input-kernel.bin");
    fs::write(&module, kernel()?)?;
    let mut rows = Vec::new();
    let mut failures = 0;
    for sockets in [1, 2] {
        for &workers in &worker_counts {
            let case = Case { sockets, workers };
            let mut boot_times = Vec::new();
            let mut scheduler_times = Vec::new();
            let mut instruction_rates = Vec::new();
            let mut samples = Vec::new();
            for repeat in 1..=arguments.repeats {
                let directory = format!("{sockets}s-{workers}w-{repeat}");
                let output = arguments.output.join(&directory);
                match fixture_run(&arguments, &image, &module, &output, case) {
                    Ok(sample) => {
                        boot_times.push(sample.boot_and_fixture_seconds);
                        if let Some(timing) = &sample.evidence.measurement {
                            scheduler_times.push(timing.scheduler_seconds);
                            instruction_rates.push(timing.guest_instructions_per_second);
                        }
                        println!(
                            "{directory}: {} active workers, {} packages, {:.3}s boot and fixture",
                            sample.evidence.active_workers,
                            sample.evidence.packages_used,
                            sample.boot_and_fixture_seconds
                        );
                        samples.push(
                            json!({ "directory": directory, "passed": true, "sample": sample }),
                        );
                    }
                    Err(error) => {
                        failures += 1;
                        eprintln!("{directory}: {error:#}");
                        samples.push(json!({ "directory": directory, "passed": false, "error": format!("{error:#}") }));
                    }
                }
            }
            rows.push(json!({
                "case": case,
                "samples": samples,
                "median_boot_and_fixture_seconds": median(&mut boot_times),
                "median_scheduler_seconds": median(&mut scheduler_times),
                "median_guest_instructions_per_second": median(&mut instruction_rates),
            }));
        }
    }
    let report = arguments.output.join("matrix.json");
    let matrix = json!({
        "image": binding(&image)?,
        "fixture": "authored native shared ARM kernel",
        "accelerator": if arguments.kvm { "kvm" } else { "tcg" },
        "physical_cores": PHYSICAL_CORES,
        "logical_cpus": LOGICAL_CPUS,
        "memory": arguments.memory,
        "passed": failures == 0,
        "runs": rows,
    });
    fs::write(&report, format!("{matrix:#}\n"))?;
    ensure!(
        failures == 0,
        "{failures} native-worker runs failed, see {}",
        report.display()
    );
    Ok(ExitCode::SUCCESS)
}

fn median(values: &mut [f64]) -> Option<f64> {
    values.sort_unstable_by(f64::total_cmp);
    let middle = values.len() / 2;
    values.get(middle).map(|middle_value| {
        if values.len().is_multiple_of(2) {
            (values[middle - 1] + middle_value) / 2.0
        } else {
            *middle_value
        }
    })
}

fn fixture_run(
    arguments: &Arguments,
    image: &Path,
    module: &Path,
    output: &Path,
    case: Case,
) -> Result<Sample> {
    let mut argv = vec![
        OsString::from("boot-run"),
        "--image".into(),
        image.as_os_str().into(),
        "--module".into(),
        module.as_os_str().into(),
        "--output".into(),
        output.as_os_str().into(),
        "--memory".into(),
        arguments.memory.clone().into(),
        "--host-sockets".into(),
        case.sockets.to_string().into(),
        "--host-cores".into(),
        (PHYSICAL_CORES / case.sockets).to_string().into(),
        "--host-threads".into(),
        "2".into(),
        "--timeout".into(),
        arguments.timeout.to_string().into(),
        "--append".into(),
        format!(
            "{KERNEL_OPTIONS} --native-shared-check --native-cores {}",
            case.workers
        )
        .into(),
    ];
    if arguments.kvm {
        argv.push("--kvm".into());
    }
    let matches = Run::augment_args(clap::Command::new("boot-run")).try_get_matches_from(argv)?;
    launch(Run::from_arg_matches(&matches)?, false)?;
    let run: QemuRun = serde_json::from_slice(&fs::read(output.join("run.json"))?)?;
    let serial = fs::read_to_string(output.join("serial.txt"))?;
    let fatal = serial.lines().find(|line| line.starts_with("STELA FATAL "));
    ensure!(
        !run.timed_out && run.qemu_returncode == IDLE_EXIT && fatal.is_none(),
        "QEMU exit {}, timed_out={}; {}",
        run.qemu_returncode,
        run.timed_out,
        fatal.unwrap_or("no fatal message")
    );
    ensure!(
        run.wall_seconds.is_finite() && run.wall_seconds > 0.0,
        "invalid QEMU run duration"
    );
    let evidence = verify(case, &serial)?;
    ensure!(
        !arguments.require_translation_proof || evidence.translation_proof.is_some(),
        "missing verified translation-proof workload"
    );
    Ok(Sample {
        qemu_version: run.qemu_version,
        boot_and_fixture_seconds: run.wall_seconds,
        evidence,
    })
}

fn field<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    line.split_ascii_whitespace().find_map(|word| {
        let (name, value) = word.split_once('=')?;
        (name == key).then_some(value)
    })
}

fn number(line: &str, key: &str) -> Result<u64> {
    field(line, key)
        .with_context(|| format!("missing {key} in {line}"))?
        .parse()
        .with_context(|| format!("invalid {key} in {line}"))
}

fn verify(case: Case, serial: &str) -> Result<Evidence> {
    let workers = u64::from(case.workers);
    let inventory = serial
        .lines()
        .find(|line| line.starts_with("NATIVE INVENTORY "))
        .context("missing firmware processor inventory")?;
    for (key, expected) in [
        ("firmware_processors", u64::from(LOGICAL_CPUS)),
        ("physical_cores", u64::from(PHYSICAL_CORES)),
        ("logical", u64::from(LOGICAL_CPUS)),
        ("requested", workers),
    ] {
        ensure!(
            number(inventory, key)? == expected,
            "expected inventory {key}={expected}: {inventory}"
        );
    }
    if field(inventory, "runners").is_some() {
        ensure!(
            number(inventory, "runners")? == workers,
            "the fixture did not provision every requested guest runner"
        );
    }

    let mut slots = BTreeSet::new();
    let mut apic_ids = BTreeSet::new();
    let mut packages = BTreeSet::new();
    let mut cores = BTreeMap::<[u64; 5], BTreeSet<u64>>::new();
    for line in serial
        .lines()
        .filter(|line| line.starts_with("NATIVE SOURCE slot="))
    {
        let Some(apic) = field(line, "apic") else {
            continue;
        };
        let slot = number(line, "slot")?;
        ensure!(slots.insert(slot), "duplicated native source slot {slot}");
        let apic = u32::from_str_radix(
            apic.strip_prefix("0x").context("missing APIC hex prefix")?,
            16,
        )
        .context("invalid APIC ID")?;
        ensure!(apic_ids.insert(apic), "duplicated native APIC ID {apic:#x}");
        let package = number(line, "package")?;
        packages.insert(package);
        let core = [
            package,
            number(line, "module")?,
            number(line, "tile")?,
            number(line, "die")?,
            number(line, "core")?,
        ];
        ensure!(
            cores
                .entry(core)
                .or_default()
                .insert(number(line, "thread")?),
            "duplicated SMT thread identity in physical core {core:?}"
        );
    }
    ensure!(
        slots.iter().copied().eq(0..workers),
        "native source slots do not cover {workers} workers"
    );
    let expected_cores = usize::from(case.workers.min(PHYSICAL_CORES));
    ensure!(
        cores.len() == expected_cores && cores.values().all(|threads| threads.len() <= 2),
        "expected {expected_cores} distinct physical cores with at most two selected siblings each"
    );
    if case.workers >= PHYSICAL_CORES {
        ensure!(
            packages.len() == usize::from(case.sockets),
            "workers covered {} packages, expected {}",
            packages.len(),
            case.sockets
        );
    }
    let mut started_ap_slots = BTreeSet::new();
    for line in serial
        .lines()
        .filter(|line| line.starts_with("NATIVE HOST slot="))
    {
        let slot = number(line, "slot")?;
        ensure!(
            number(line, "doorbells")? > 0 && started_ap_slots.insert(slot),
            "missing first doorbell or duplicate receipt for native worker {slot}"
        );
    }
    ensure!(
        started_ap_slots.iter().copied().eq(1..workers),
        "startup receipts do not cover all {} secondary workers",
        workers - 1
    );

    let pass = serial
        .lines()
        .find(|line| line.starts_with("CHECK native-shared-kernel PASS "))
        .context("missing verified shared-kernel result")?;
    for (key, expected) in [
        ("actual_frames", workers),
        ("CPU_starts", workers - 1),
        ("consumed_device_writes", workers * 2),
        ("CPU_controller_commits", workers * 4),
    ] {
        ensure!(
            number(pass, key)? == expected,
            "expected shared-kernel {key}={expected}"
        );
    }
    ensure!(
        field(pass, "canonical_RAM") == Some("shared")
            && field(pass, "full_GPR_SIMD_FP_state") == Some("preserved"),
        "shared memory and complete guest state were not verified"
    );
    let atomic_updates = number(pass, "lse_updates")?;
    let scheduler_rounds = number(pass, "scheduler_rounds")?;
    ensure!(
        atomic_updates > 0 && atomic_updates.is_multiple_of(workers) && scheduler_rounds > 0,
        "shared-kernel work counters are inconsistent"
    );
    let mut total_retired = 0u64;
    let mut frames = 0;
    for line in serial
        .lines()
        .filter(|line| line.starts_with("NATIVE SCHEDULER frame pc="))
    {
        let retired = number(line, "retired")?;
        ensure!(
            retired > 0,
            "a translated frame never retired an instruction"
        );
        total_retired = total_retired
            .checked_add(retired)
            .context("retired instruction total overflow")?;
        frames += 1;
    }
    ensure!(
        frames == workers,
        "expected {workers} translated frame records, got {frames}"
    );
    let measurement = match (
        field(pass, "scheduler_ticks"),
        field(pass, "tsc_hz"),
        field(pass, "scheduled_instructions"),
    ) {
        (None, None, None) => None,
        (Some(_), Some(_), Some(_)) => {
            let scheduler_ticks = number(pass, "scheduler_ticks")?;
            let tsc_hz = number(pass, "tsc_hz")?;
            let scheduled_instructions = number(pass, "scheduled_instructions")?;
            ensure!(
                scheduler_ticks > 0
                    && tsc_hz > 0
                    && (1..=total_retired).contains(&scheduled_instructions),
                "invalid scheduler timing counters"
            );
            let scheduler_seconds = scheduler_ticks as f64 / tsc_hz as f64;
            Some(Measurement {
                scheduler_ticks,
                tsc_hz,
                scheduled_instructions,
                scheduler_seconds,
                guest_instructions_per_second: scheduled_instructions as f64 / scheduler_seconds,
            })
        }
        _ => anyhow::bail!("incomplete scheduler timing counters"),
    };
    Ok(Evidence {
        active_workers: case.workers,
        packages_used: packages.len(),
        cores_used: cores.len(),
        smt_siblings_used: usize::from(case.workers) - cores.len(),
        atomic_updates,
        scheduler_rounds,
        total_retired,
        measurement,
        translation_proof: translation_proof(serial, workers)?,
    })
}

fn translation_proof(serial: &str, active_workers: u64) -> Result<Option<TranslationProof>> {
    let Some(pass) = serial
        .lines()
        .find(|line| line.starts_with("CHECK native-translation-proof PASS "))
    else {
        return Ok(None);
    };
    let guest_reserved = if active_workers == u64::from(LOGICAL_CPUS) {
        8
    } else {
        1
    };
    let eligible_workers = active_workers - guest_reserved;
    let completed_workers = number(pass, "completed")?;
    let proven_blocks = number(pass, "proven")?;
    ensure!(
        number(pass, "logical")? == active_workers
            && number(pass, "guest_reserved")? == guest_reserved
            && number(pass, "workers")? == eligible_workers
            && completed_workers == eligible_workers
            && proven_blocks == eligible_workers * 32,
        "translation proof cohort is incomplete or inconsistent: {pass}"
    );
    Ok(Some(TranslationProof {
        eligible_workers,
        completed_workers,
        proven_blocks,
        elapsed_ms: number(pass, "ms")?,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Write;

    fn log(case: Case) -> String {
        let mut log = format!(
            "NATIVE INVENTORY firmware_processors=64 physical_cores=32 logical=64 requested={}\n",
            case.workers
        );
        for slot in 0..case.workers {
            let core = slot % PHYSICAL_CORES;
            let thread = slot / PHYSICAL_CORES;
            let cores_per_socket = PHYSICAL_CORES / case.sockets;
            writeln!(
                log,
                "NATIVE SOURCE slot={slot} apic={:#x} package={} module=0 tile=0 die=0 core={} thread={thread}",
                2 * core + thread,
                core / cores_per_socket,
                core % cores_per_socket
            ).unwrap();
            if slot != 0 {
                writeln!(log, "NATIVE HOST slot={slot} doorbells=1").unwrap();
            }
            writeln!(log, "NATIVE SCHEDULER frame pc=0x1 retired=500").unwrap();
        }
        let workers = case.workers;
        writeln!(
            log,
            "CHECK native-shared-kernel PASS actual_frames={workers} CPU_starts={} consumed_device_writes={} CPU_controller_commits={} canonical_RAM=shared full_GPR_SIMD_FP_state=preserved lse_updates={} scheduler_rounds=8",
            workers - 1, workers * 2, workers * 4, workers * 192,
        ).unwrap();
        log
    }

    #[test]
    fn counts_active_workers_and_both_smt_siblings_across_sockets() {
        let case = Case {
            sockets: 2,
            workers: 64,
        };
        let evidence = verify(case, &log(case)).unwrap();
        assert_eq!(evidence.active_workers, 64);
        assert_eq!(evidence.packages_used, 2);
        assert_eq!(evidence.cores_used, 32);
        assert_eq!(evidence.smt_siblings_used, 32);
        assert_eq!(evidence.total_retired, 32000);
        assert!(evidence.measurement.is_none());
        assert!(evidence.translation_proof.is_none());
        assert!(
            verify(
                case,
                &log(case).replace("slot=63 apic=0x3f", "slot=63 apic=0x3e")
            )
            .is_err()
        );
        assert!(
            verify(
                case,
                &log(case).replace("slot=63 doorbells=1", "slot=63 doorbells=0")
            )
            .is_err()
        );
        assert!(
            verify(
                case,
                &log(case).replace("actual_frames=64", "actual_frames=63")
            )
            .is_err()
        );
    }

    #[test]
    fn translation_proof_requires_every_eligible_worker_and_block() {
        let case = Case {
            sockets: 2,
            workers: 64,
        };
        let log = format!(
            "{}CHECK native-translation-proof PASS logical=64 guest_reserved=8 workers=56 completed=56 proven=1792 ms=340\n",
            log(case)
        );
        let proof = verify(case, &log).unwrap().translation_proof.unwrap();
        assert_eq!(proof.completed_workers, 56);
        assert_eq!(proof.proven_blocks, 1792);
        assert_eq!(proof.elapsed_ms, 340);
        assert!(verify(case, &log.replace("completed=56", "completed=55")).is_err());
    }

    #[test]
    fn separates_scheduler_time_from_boot_and_rejects_partial_metrics() {
        let case = Case {
            sockets: 1,
            workers: 32,
        };
        let log = log(case).replace(
            "scheduler_rounds=8",
            "scheduler_rounds=8 scheduler_ticks=100000000 tsc_hz=2000000000 scheduled_instructions=10000"
        );
        let evidence = verify(case, &log).unwrap();
        assert_eq!(evidence.smt_siblings_used, 0);
        let measurement = evidence.measurement.unwrap();
        assert_eq!(measurement.scheduler_seconds, 0.05);
        assert_eq!(measurement.guest_instructions_per_second, 200000.0);
        assert!(verify(case, &log.replace(" tsc_hz=2000000000", "")).is_err());
        assert_eq!(median(&mut [3.0, 1.0, 9.0, 2.0]), Some(2.5));
        assert_eq!(median(&mut []), None);
    }
}
