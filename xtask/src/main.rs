//! Repository development tasks: `cargo xtask <task>`.
use clap::{Parser, Subcommand};
use std::process::ExitCode;

mod amx;
mod boot;
mod catalog;
mod check;
mod cpu_reference;
mod dist;
mod gpu;
mod guest;
mod oracles;
mod pages;
mod restore_dry_run;
mod sysregs;
mod tools;

#[derive(Parser)]
#[command(about = "Stela development tasks")]
struct Cli {
    #[command(subcommand)]
    task: Task,
}

#[derive(Subcommand)]
enum Task {
    /// Check formatting, run strict host and freestanding Clippy, then test.
    Check {
        /// Format sources before checking.
        #[arg(long)]
        fix: bool,
        /// Run check lanes, build jobs, and tests one at a time.
        #[arg(long)]
        low_memory: bool,
    },
    /// Build the release `stela` with its matching `stela.efi` embedded.
    Dist,
    /// Generate or check the Northwood PCI driver catalog.
    Catalog(catalog::Catalog),
    /// Run ignored oracle tests whose declared inputs are present and write a dated report.
    Oracles(oracles::Oracles),
    /// Build, run and debug the UEFI mount in x86 QEMU.
    #[command(subcommand)]
    Boot(boot::Command),
    /// Build native GPU components and report their remaining platform imports.
    #[command(subcommand)]
    Gpu(gpu::Command),
    /// Build the ARM64e components Stela installs in macOS: the StelaGPU
    /// kext's objects, against `third_party/xnu`, carrying the newest
    /// Stela.app package (or `STELA_APP`'s) when there is one.
    Guest,
    /// Build Stela.app (`guest/stela-app`) on this Mac or, over ssh, on the
    /// Mac `--mac` or `STELA_MAC` names, into `target/stela-app`, where
    /// `cargo xtask guest` finds it.
    App {
        /// The Mac's ssh name (default: `STELA_MAC`).
        #[arg(long)]
        mac: Option<String>,
    },
    /// Compare Stela's AMX model with the corsix/amx emulator on QEMU.
    AmxReference(amx::AmxReference),
    /// Disassemble, size and time the native AMX kernels against the model.
    AmxKernels(amx::kernels::AmxKernels),
    /// Run A64 cases on an Apple silicon Mac and through Stela, and compare.
    CpuReference(cpu_reference::CpuReference),
    /// Drive the whole restore host against the real IPSW, natively, with
    /// every request restored can make, and check each answer.
    RestoreDryRun(restore_dry_run::RestoreDryRun),
    /// Generate the System register accessors from Arm's AARCHMRS.
    Sysregs(sysregs::Sysregs),
}

fn main() -> ExitCode {
    let result = match Cli::parse().task {
        Task::Check { fix, low_memory } => check::run(fix, low_memory).map(|()| ExitCode::SUCCESS),
        Task::Dist => dist::run().map(|()| ExitCode::SUCCESS),
        Task::Catalog(task) => task.run().map(|()| ExitCode::SUCCESS),
        Task::Oracles(task) => task.run().map(|()| ExitCode::SUCCESS),
        Task::Boot(command) => boot::run(command),
        Task::Gpu(command) => gpu::run(command).map(|()| ExitCode::SUCCESS),
        Task::Guest => guest::run().map(|()| ExitCode::SUCCESS),
        Task::App { mac } => {
            let mac = mac.or_else(|| std::env::var(tools::MAC_VARIABLE).ok());
            guest::app::build(tools::root(), mac.as_deref().filter(|mac| !mac.is_empty()))
                .map(|_| ExitCode::SUCCESS)
        }
        Task::AmxReference(task) => task.run().map(|()| ExitCode::SUCCESS),
        Task::AmxKernels(task) => task.run().map(|()| ExitCode::SUCCESS),
        Task::CpuReference(task) => task.run().map(|()| ExitCode::SUCCESS),
        Task::RestoreDryRun(task) => task.run(),
        Task::Sysregs(task) => task.run().map(|()| ExitCode::SUCCESS),
    };
    result.unwrap_or_else(|error| {
        eprintln!("error: {error:#}");
        ExitCode::FAILURE
    })
}
