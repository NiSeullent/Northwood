// SPDX-License-Identifier: GPL-2.0-or-later
//! The `northwood` command's entry point.
use clap::Parser;
use std::process::ExitCode;

fn main() -> ExitCode {
    northwood::Northwood::parse().run()
}
