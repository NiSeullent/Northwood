//! `cargo xtask cpu-reference`: run the same A64 cases at EL0 on an Apple
//! silicon Mac and through Stela, and compare their final states.
//!
//! The Mac back end builds `cpu_reference/runner.c` there over ssh and
//! streams the case records through it. Stela's back end compiles each case
//! and runs it through the engine's block runtime. `--record` keeps the
//! Mac's outcomes as the regression table `cpu_reference/apple-m1-pro.tsv`,
//! which the xtask tests check Stela against without the Mac.
use anyhow::{Context, Result, bail, ensure};
use clap::Args;
use std::{
    collections::BTreeMap,
    io::{Read, Write},
    process::{Command, Stdio},
};
use stela_compiler::TargetCpu;

mod corpus;
mod record;
mod stela;

use record::{Case, Ending, INPUT_BYTES, OUTPUT_BYTES};

const RUNNER: &str = include_str!("cpu_reference/runner.c");
const REMOTE_DIR: &str = "stela-cpu-reference";
const TABLE: &str = include_str!("cpu_reference/apple-m1-pro.tsv");
const TABLE_PATH: &str = "xtask/src/cpu_reference/apple-m1-pro.tsv";

/// The runner is an arm64 process, whose PAC keys macOS disables, so PACIA
/// and AUTIA leave every pointer; Stela runs arm64e processes with keys, so
/// a non-canonical pointer fails authentication.
const ARM64_KEYS_DISABLED: &str = "arm64 process: keys disabled on the Mac";
/// Cases where Stela differs from the recorded Mac on purpose, and why.
pub(crate) const KNOWN_DIFFERENCES: [(&str, &str); 5] = [
    ("pacia.autia.1", ARM64_KEYS_DISABLED),
    ("pacia.autia.2", ARM64_KEYS_DISABLED),
    ("pacia.autia.3", ARM64_KEYS_DISABLED),
    ("pacia.autia.4", ARM64_KEYS_DISABLED),
    // FEAT_AFP's FIZ, AH and NEP are M4 and later; Stela's process profile
    // keeps them. The M1 Pro also keeps AArch32's Len and Stride, RES0
    // without AArch32, which Stela reads as zero.
    (
        "fpcr.all",
        "FPCR fields: FEAT_AFP (M4 on) and the M1's Len and Stride",
    ),
];

fn known_difference(name: &str) -> Option<&'static str> {
    KNOWN_DIFFERENCES
        .iter()
        .find(|(known, _)| *known == name)
        .map(|(_, reason)| *reason)
}

#[derive(Args)]
pub struct CpuReference {
    /// The Mac to run the cases on, as ssh names it (default: `STELA_MAC`).
    #[arg(long)]
    mac: Option<String>,
    /// Compare Stela with the recorded table instead of running the Mac.
    #[arg(long, conflicts_with = "record")]
    recorded: bool,
    /// Rewrite the recorded table from this Mac run.
    #[arg(long)]
    record: bool,
    /// Only the cases whose name contains this text.
    #[arg(long)]
    only: Option<String>,
    /// Compile for this host-supported x86 target instead of the detected one.
    #[arg(long)]
    target: Option<TargetCpu>,
    /// List the case names and exit.
    #[arg(long)]
    list: bool,
}

impl CpuReference {
    pub fn run(self) -> Result<()> {
        let cases: Vec<Case> = corpus::all()
            .into_iter()
            .filter(|case| {
                self.only
                    .as_ref()
                    .is_none_or(|text| case.name.contains(text))
            })
            .collect();
        if self.list {
            for case in &cases {
                println!("{}", case.name);
            }
            println!("{} cases", cases.len());
            return Ok(());
        }
        ensure!(!cases.is_empty(), "no case matches");
        let host = TargetCpu::detect();
        let target = self.target.unwrap_or(host);
        let target_id = target.id();
        ensure!(
            host.supports(target),
            "host {} cannot run target {target_id}",
            host.id()
        );
        let (reference, source) = if self.recorded {
            let table = recorded()?;
            let outcomes = cases
                .iter()
                .map(|case| {
                    table
                        .get(case.name.as_str())
                        .map(|outcome| outcome.to_string())
                        .with_context(|| {
                            format!("{} is not recorded; run with --record", case.name)
                        })
                })
                .collect::<Result<Vec<_>>>()?;
            (outcomes, "recorded")
        } else {
            let host = crate::tools::mac_host(self.mac.clone())?;
            let endings = mac(&host, &cases)?;
            let outcomes: Vec<String> = cases
                .iter()
                .zip(&endings)
                .map(|(case, ending)| record::outcome(case, ending))
                .collect();
            if self.record {
                ensure!(self.only.is_none(), "--record keeps every case");
                write_table(&host, &cases, &outcomes)?;
            }
            (outcomes, "mac")
        };
        let stela: Vec<String> = stela::run(&cases, target)?
            .iter()
            .zip(&cases)
            .map(|(ending, case)| record::outcome(case, ending))
            .collect();
        let ledger = ledger(&cases, &reference, &stela)?;
        let differences: Vec<usize> = (0..cases.len())
            .filter(|&index| reference[index] != stela[index])
            .collect();
        let (known, unknown): (Vec<usize>, Vec<usize>) = differences
            .iter()
            .partition(|&&index| known_difference(&cases[index].name).is_some());
        for &index in unknown.iter().take(40) {
            println!(
                "{}\n  {source}: {}\n  stela: {}",
                cases[index].name, reference[index], stela[index]
            );
        }
        for &index in &known {
            let name = &cases[index].name;
            println!("{name}: known ({})", known_difference(name).unwrap());
        }
        println!(
            "target {target_id}: {} cases, {} match, {} differ, {} known; ledger {}",
            cases.len(),
            cases.len() - differences.len(),
            unknown.len(),
            known.len(),
            ledger.display()
        );
        ensure!(
            unknown.is_empty(),
            "Stela differs from the {source} outcomes"
        );
        Ok(())
    }
}

/// Build the runner on `host` and run every case there.
fn mac(host: &str, cases: &[Case]) -> Result<Vec<Ending>> {
    let build = format!(
        "set -e; mkdir -p {REMOTE_DIR}; cat > {REMOTE_DIR}/runner.c; \
         clang -O1 -Wall -Wextra -Werror -arch arm64 {REMOTE_DIR}/runner.c -o {REMOTE_DIR}/runner"
    );
    let mut child = ssh(host, &build).stdin(Stdio::piped()).spawn()?;
    child.stdin.take().unwrap().write_all(RUNNER.as_bytes())?;
    let status = child.wait()?;
    ensure!(
        status.success(),
        "building the runner on {host} failed: {status}"
    );
    let mut child = ssh(host, &format!("{REMOTE_DIR}/runner"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    let mut input = child.stdin.take().unwrap();
    let records: Vec<[u8; INPUT_BYTES]> = cases.iter().map(Case::encode).collect();
    let writer = std::thread::spawn(move || -> std::io::Result<()> {
        for record in records {
            input.write_all(&record)?;
        }
        Ok(())
    });
    let mut output = Vec::new();
    child.stdout.take().unwrap().read_to_end(&mut output)?;
    writer.join().expect("writer thread")?;
    let status = child.wait()?;
    ensure!(status.success(), "the runner on {host} failed: {status}");
    ensure!(
        output.len() == cases.len() * OUTPUT_BYTES,
        "the runner returned {} bytes for {} cases",
        output.len(),
        cases.len()
    );
    Ok(output
        .chunks(OUTPUT_BYTES)
        .map(record::decode_output)
        .collect())
}

fn ssh(host: &str, script: &str) -> Command {
    let mut command = Command::new("ssh");
    command.args(["-o", "BatchMode=yes", host, script]);
    command
}

/// The recorded table: case name to outcome.
fn recorded() -> Result<BTreeMap<&'static str, &'static str>> {
    let mut table = BTreeMap::new();
    for line in TABLE
        .lines()
        .filter(|line| !line.starts_with('#') && !line.is_empty())
    {
        let Some((name, outcome)) = line.split_once('\t') else {
            bail!("malformed recorded line {line:?}");
        };
        table.insert(name, outcome);
    }
    Ok(table)
}

fn write_table(host: &str, cases: &[Case], outcomes: &[String]) -> Result<()> {
    let identity = Command::new("ssh")
        .args([
            "-o",
            "BatchMode=yes",
            host,
            "sysctl -n machdep.cpu.brand_string hw.model; sw_vers -productVersion; sw_vers -buildVersion",
        ])
        .output()?;
    let identity = String::from_utf8_lossy(&identity.stdout).replace('\n', " ");
    let mut text = format!(
        "# Generated by `cargo xtask cpu-reference --record`: each case's outcome at EL0\n\
         # in an arm64 process on {}\n# Outcome: the registers and buffer bytes that changed, or SIGNAL@word.\n",
        identity.trim()
    );
    for (case, outcome) in cases.iter().zip(outcomes) {
        text.push_str(&format!("{}\t{outcome}\n", case.name));
    }
    let path = crate::tools::root().join(TABLE_PATH);
    std::fs::write(&path, text)?;
    println!("recorded {}", path.display());
    Ok(())
}

fn ledger(cases: &[Case], reference: &[String], stela: &[String]) -> Result<std::path::PathBuf> {
    let directory = crate::tools::root().join(".stela/reference/cpu");
    std::fs::create_dir_all(&directory)?;
    let seconds = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs();
    let path = directory.join(format!("{seconds}.tsv"));
    let mut text = String::from("# case\treference\tstela\n");
    for (index, case) in cases.iter().enumerate() {
        text.push_str(&format!(
            "{}\t{}\t{}\n",
            case.name, reference[index], stela[index]
        ));
    }
    std::fs::write(&path, text)?;
    Ok(path)
}

#[cfg(test)]
mod tests;
