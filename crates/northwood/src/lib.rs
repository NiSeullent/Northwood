// SPDX-License-Identifier: GPL-2.0-or-later
//! Host-side installer and configuration tools for Northwood.
use clap::{Parser, Subcommand};
use std::{path::PathBuf, process::ExitCode};

mod import;
mod oc;

#[derive(Parser)]
#[command(name = "northwood", version, about)]
pub struct Northwood {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Read an existing OpenCore plist and export Northwood configuration hints.
    ImportOc {
        #[arg(long)]
        config: PathBuf,
        /// Optional existing directory with ACPI, Kexts and Drivers subdirectories.
        #[arg(long)]
        efi: Option<PathBuf>,
        #[arg(long)]
        out: PathBuf,
    },
    /// Inspect Northwood configuration files.
    #[cfg(not(feature = "installer"))]
    Config {
        #[command(subcommand)]
        command: ConfigCommand,
    },
    #[cfg(feature = "installer")]
    #[command(flatten)]
    Host(stela_cli::HostCommand),
}

#[cfg(not(feature = "installer"))]
#[derive(Subcommand)]
enum ConfigCommand {
    /// Validate an XML nwconfig.plist and its effective boot options.
    Validate { file: PathBuf },
}

impl Northwood {
    pub fn run(self) -> ExitCode {
        let result = match self.command {
            Command::ImportOc { config, efi, out } => import::write(&config, efi.as_deref(), &out)
                .map(|paths| {
                    for path in paths {
                        println!("wrote {}", path.display());
                    }
                }),
            #[cfg(not(feature = "installer"))]
            Command::Config {
                command: ConfigCommand::Validate { file },
            } => stela_boot_core::config_validation::validate_file(&file),
            #[cfg(feature = "installer")]
            Command::Host(command) => return command.run(),
        };
        match result {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("error: {error:#}");
                ExitCode::FAILURE
            }
        }
    }
}

#[cfg(test)]
mod cli_tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn lists_configuration_tools() {
        Northwood::command().debug_assert();
        let help = Northwood::command().render_help().to_string();
        assert!(help.contains("import-oc") && help.contains("config"));
    }

    #[test]
    fn reports_cross_constraint_in_a_nondefault_boot_entry() {
        use std::io::Write as _;

        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(
            br#"<plist><dict><key>Boot</key><dict>
            <key>Bundle</key><string>boot/fixture</string>
            <key>Entries</key><array>
                <dict><key>Name</key><string>Normal</string></dict>
                <dict><key>Name</key><string>Repair</string>
                    <key>Overrides</key><dict><key>Boot</key><dict>
                        <key>VerifyOnRead</key><true/>
                    </dict></dict>
                </dict>
            </array></dict></dict></plist>"#,
        )
        .unwrap();
        let error = stela_boot_core::config_validation::validate_file(file.path()).unwrap_err();
        let message = format!("{error:#}");
        assert!(message.contains("Boot/Entries[1] (\"Repair\"): Boot/VerifyOnRead"));
        assert!(message.contains("--verify-on-read applies to --install and --restore-study"));
    }

    #[cfg(feature = "installer")]
    #[test]
    fn shares_installer_and_report_grammar() {
        let help = Northwood::command().render_help().to_string();
        for command in ["setup", "boot-log", "report", "config"] {
            assert!(help.contains(command), "{help}");
            let error = Northwood::try_parse_from(["northwood", command, "--help"])
                .err()
                .unwrap();
            assert_eq!(error.kind(), clap::error::ErrorKind::DisplayHelp);
            assert!(error.to_string().contains(&format!("northwood {command}")));
        }
        assert!(
            Northwood::try_parse_from([
                "northwood",
                "setup",
                "update",
                "stela",
                "--usb",
                "VOLUME",
                "--fetch",
            ])
            .is_ok()
        );
    }
}
