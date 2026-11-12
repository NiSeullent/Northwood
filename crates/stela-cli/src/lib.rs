//! The `stela` command line prepares a USB that boots Stela to install macOS
//! on a PC and reads the boot log. Research and development commands belong to
//! `stela-dev`, which uses this library's boot-bundle and restore-image tools.
use clap::{FromArgMatches, Parser, Subcommand};
use std::process::ExitCode;

pub mod apfs;
pub mod args;
pub mod boot;
mod boot_log;
pub mod efi;
pub use stela_gpu_maintenance as gpu;
mod net;
mod progress;
mod report;
pub mod restore;
mod setup;

pub use setup::InstallMedium;

/// The help's examples, with this computer's spelling of a whole disk.
fn examples() -> String {
    let disk = setup::host::DISK_EXAMPLE;
    format!(
        "Examples:
  stela setup                       Create a macOS installer USB, step by step
  stela setup --usb {disk:<15} Erase that USB and write the installer, downloading macOS
  stela setup status --usb VOLUME   Check an installer USB by its mounted STELA volume"
    )
}
const ABOUT: &str = "Prepare a USB that boots Stela to install macOS on a PC";

#[derive(Parser)]
#[command(
    name = "stela",
    version,
    about = ABOUT,
    after_help = examples(),
    arg_required_else_help = true
)]
#[cfg_attr(
    windows,
    allow(
        dead_code,
        reason = "Windows runs Installer directly; the shared command grammar remains covered by tests"
    )
)]
pub(crate) struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
#[cfg_attr(
    windows,
    allow(
        dead_code,
        reason = "Windows runs Installer directly; the shared command grammar remains covered by tests"
    )
)]
pub(crate) enum Command {
    /// Create a macOS installer USB interactively, or manage an existing USB.
    Setup(Box<setup::Setup>),
    /// Print every boot log stela.efi kept on a disk image or a raw partition:
    /// it finds each machine-state partition, or SSD partition tail, itself.
    BootLog(boot_log::BootLog),
    /// Summarize recorded Linux PCI and USB inventories as a Markdown report.
    Report(report::Report),
    /// Inspect Northwood configuration files.
    Config {
        #[command(subcommand)]
        command: ConfigCommand,
    },
}

#[derive(Subcommand)]
pub(crate) enum ConfigCommand {
    /// Validate an XML nwconfig.plist and its effective boot options.
    Validate { file: std::path::PathBuf },
}

// Windows ships the direct installer; other hosts expose these subcommands.
#[cfg_attr(
    windows,
    allow(
        dead_code,
        reason = "Windows runs Installer directly; the shared command grammar remains covered by tests"
    )
)]
impl Command {
    fn run(self) -> anyhow::Result<()> {
        match self {
            Self::Setup(command) => command.run(),
            Self::BootLog(command) => command.run(),
            Self::Report(command) => command.run(),
            Self::Config {
                command: ConfigCommand::Validate { file },
            } => stela_boot_core::config_validation::validate_file(&file),
        }
    }
}

/// The portable installer and reporting commands embedded by another CLI.
/// Implements the same Clap grammar as `stela`'s portable entry point.
pub struct HostCommand(Command);

impl FromArgMatches for HostCommand {
    fn from_arg_matches(matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        Command::from_arg_matches(matches).map(Self)
    }

    fn from_arg_matches_mut(matches: &mut clap::ArgMatches) -> Result<Self, clap::Error> {
        Command::from_arg_matches_mut(matches).map(Self)
    }

    fn update_from_arg_matches(&mut self, matches: &clap::ArgMatches) -> Result<(), clap::Error> {
        self.0.update_from_arg_matches(matches)
    }

    fn update_from_arg_matches_mut(
        &mut self,
        matches: &mut clap::ArgMatches,
    ) -> Result<(), clap::Error> {
        self.0.update_from_arg_matches_mut(matches)
    }
}

impl Subcommand for HostCommand {
    fn augment_subcommands(command: clap::Command) -> clap::Command {
        Command::augment_subcommands(command)
    }

    fn augment_subcommands_for_update(command: clap::Command) -> clap::Command {
        Command::augment_subcommands_for_update(command)
    }

    fn has_subcommand(name: &str) -> bool {
        Command::has_subcommand(name)
    }
}

impl HostCommand {
    /// Run a parsed command and report its result with Stela's CLI behavior.
    pub fn run(self) -> ExitCode {
        finish(self.0.run())
    }
}

/// Parse the command line and run it, reporting errors on stderr.
pub fn main() -> ExitCode {
    #[cfg(windows)]
    let result = installer();
    #[cfg(not(windows))]
    let result = commands();
    finish(result)
}

fn finish(result: anyhow::Result<()>) -> ExitCode {
    let code = match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!(
                "{} {error:#}",
                console::style("error:").for_stderr().red().bold()
            );
            ExitCode::FAILURE
        }
    };
    setup::hold_window(code == ExitCode::SUCCESS);
    code
}

#[cfg(not(windows))]
fn commands() -> anyhow::Result<()> {
    Cli::parse().command.run()
}

/// `stela.exe` is the installer alone: running it opens the setup wizard, and
/// its arguments are `stela setup`'s own flags and commands.
#[cfg(windows)]
#[derive(Parser)]
#[command(name = "stela", version, about = ABOUT)]
struct Installer {
    #[command(flatten)]
    setup: setup::Setup,
}

#[cfg(windows)]
fn installer() -> anyhow::Result<()> {
    let arguments: Vec<_> = std::env::args_os().collect();
    if arguments.get(1).is_some_and(|word| word == "config") {
        Cli::parse_from(arguments).command.run()
    } else {
        Installer::parse_from(installer_arguments(arguments))
            .setup
            .run()
    }
}

/// `stela.exe`'s arguments without a leading `setup`, so the `stela setup
/// ...` commands that messages and the docs name work as typed.
#[cfg(any(windows, test))]
fn installer_arguments(
    arguments: impl IntoIterator<Item = std::ffi::OsString>,
) -> Vec<std::ffi::OsString> {
    let mut arguments: Vec<_> = arguments.into_iter().collect();
    if arguments.get(1).is_some_and(|word| word == "setup") {
        arguments.remove(1);
    }
    arguments
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    fn parse(words: &[&str]) -> Command {
        Cli::try_parse_from(std::iter::once("stela").chain(words.iter().copied()))
            .unwrap()
            .command
    }

    #[test]
    #[cfg(windows)]
    fn windows_executable_exposes_installer_commands() {
        // No arguments opens the wizard; setup's own flags and commands stay
        // scriptable.
        Installer::command()
            .try_get_matches_from(["stela"])
            .unwrap();
        let scripted = Installer::command()
            .try_get_matches_from(["stela", "--ipsw", "a.ipsw", "--profile"])
            .unwrap();
        assert!(scripted.get_flag("profile") && scripted.contains_id("ipsw"));
        Installer::try_parse_from(["stela", "status", "--usb", "E:\\"]).unwrap();
        Installer::try_parse_from(installer_arguments(
            ["stela", "setup", "status", "--usb", "E:\\"].map(Into::into),
        ))
        .unwrap();
        let help = Installer::command().render_help().to_string();
        assert!(help.contains("status"), "{help}");
        Installer::command().debug_assert();
    }

    #[test]
    fn a_leading_setup_word_is_dropped_for_the_installer() {
        let words =
            |list: &[&str]| -> Vec<std::ffi::OsString> { list.iter().map(Into::into).collect() };
        assert_eq!(
            installer_arguments(words(&["stela", "setup", "status", "--usb", "E:\\"])),
            words(&["stela", "status", "--usb", "E:\\"])
        );
        assert_eq!(installer_arguments(words(&["stela"])), words(&["stela"]));
        assert_eq!(
            installer_arguments(words(&["stela", "--usb", "setup"])),
            words(&["stela", "--usb", "setup"])
        );
    }

    #[test]
    fn help_lists_shipped_commands() {
        let help = Cli::command().render_help().to_string();
        assert!(
            help.contains("setup")
                && help.contains("boot-log")
                && help.contains("config")
                && !help.contains("\n  gpu"),
            "{help}"
        );
        for command in [
            "compile-block",
            "boot",
            "efi",
            "install-medium",
            "cpu-info",
            "gpu",
        ] {
            assert!(
                Cli::try_parse_from(["stela", command, "--help"]).is_err(),
                "{command}"
            );
        }
        Cli::command().debug_assert();
    }

    #[test]
    fn portable_commands_dispatch() -> anyhow::Result<()> {
        let directory = tempfile::tempdir()?;
        let missing = directory.path().join("missing.img");
        let error = parse(&["boot-log", missing.to_str().unwrap()])
            .run()
            .unwrap_err();
        assert!(error.to_string().contains("cannot open"), "{error:#}");
        assert!(error.downcast_ref::<std::io::Error>().is_some());
        assert!(matches!(
            parse(&["setup", "--usb", missing.to_str().unwrap(), "--dry-run"]),
            Command::Setup(_)
        ));
        Ok(())
    }

    #[test]
    fn setup_without_a_usb_enters_the_wizard_and_preserves_test_options() {
        assert!(matches!(parse(&["setup"]), Command::Setup(_)));
        assert!(matches!(
            parse(&["setup", "--ipsw", "local.ipsw", "--profile"]),
            Command::Setup(_)
        ));
    }

    #[test]
    fn setup_actions_are_subcommands_that_take_only_their_own_options() {
        let usb = ["--usb", "/Volumes/STELA"];
        for words in [
            &["status"][..],
            &["repair", "--ipsw", "a.ipsw", "--erase-target"],
            &["update", "stela", "--efi", "stela.efi"],
            &["update", "os", "--ipsw", "a.ipsw"],
            &["update", "stela", "--fetch"],
            &["uninstall"],
            &["nvram", "next-boot", "recovery", "--install", "macOS"],
            &["nvram", "reset", "--nor", "sdc3"],
            &["state", "backup", "state.bin"],
            &["state", "restore", "state.bin", "--nor", "sdc3"],
            &["boot-log"],
            &["configure", "--no-profile", "--chooser-timeout", "0"],
            &["host-firmware", "install", "package"],
            &[
                "host-firmware",
                "fetch",
                "requests.json",
                "--cache",
                "cache",
            ],
        ] {
            let line = ["stela", "setup"].into_iter().chain(words.iter().copied());
            Cli::try_parse_from(line.chain(usb))
                .unwrap_or_else(|error| panic!("{words:?}: {error}"));
        }
        // Each action takes neither creation's flags nor another action's.
        for words in [
            &["status", "--target", "sdc2"][..],
            &["uninstall", "--ipsw", "a.ipsw"],
            &["configure", "--profile", "--no-profile"],
            &["state", "backup", "state.bin", "--chooser-timeout", "0"],
            &["--target", "sdc2", "status"],
            &["status", "repair"],
            &["--fetch"],
            &["update", "os", "--fetch"],
            &["doctor"],
        ] {
            let line = ["stela", "setup"].into_iter().chain(words.iter().copied());
            assert!(Cli::try_parse_from(line.chain(usb)).is_err(), "{words:?}");
        }
    }
}
