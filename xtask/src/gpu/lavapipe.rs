// SPDX-License-Identifier: GPL-2.0-or-later
//! Hosted Mesa software Vulkan build for CPU validation.

use super::{
    fetch::checkout_lavapipe_mesa,
    nvk::{self, object_inputs},
    support::{build_directory, capture, tool},
    toolchain::GENERATORS,
};
use crate::tools::{binding, exec};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, btree_map::Entry},
    fs,
    io::{self, Write},
    path::Path,
    process::Command,
};

const OPTIONS: &[&str] = &[
    "--buildtype=release",
    "--libdir=lib",
    "--wrap-mode=nodownload",
    "-Doptimization=2",
    "-Dgallium-drivers=llvmpipe",
    "-Dvulkan-drivers=swrast",
    "-Dplatforms=",
    "-Dglx=disabled",
    "-Degl=disabled",
    "-Dgbm=disabled",
    "-Dglvnd=disabled",
    "-Dopengl=false",
    "-Dgles1=disabled",
    "-Dgles2=disabled",
    "-Dllvm=enabled",
    "-Dshared-llvm=enabled",
    "-Dbuild-tests=false",
    "-Dxmlconfig=disabled",
    "-Dlibunwind=disabled",
    "-Dvalgrind=disabled",
    "-Dvideo-codecs=",
    "-Dgallium-rusticl=false",
];

const DRIVER: &str = "src/gallium/targets/lavapipe/libvulkan_lvp.so";

#[derive(Deserialize)]
struct CompileCommand {
    directory: String,
    file: String,
    output: String,
    command: String,
}

#[derive(PartialEq, Serialize)]
struct PlannedObject {
    group: &'static str,
    directory: String,
    source: String,
    arguments: Vec<String>,
}

/// Inventory of a single combined Meson graph before the COFF retargeting
/// and archive selection. Multiple compile outputs for one source stay distinct.
pub fn combined_plan(build: &Path) -> Result<()> {
    let ninja = tool("ninja", GENERATORS)?;
    let nvk = object_inputs(build, &ninja, None, nvk::DRIVER)?;
    let lavapipe = object_inputs(build, &ninja, None, DRIVER)?;
    ensure!(
        !nvk.is_empty() && !lavapipe.is_empty(),
        "missing driver object inputs"
    );
    let database = build.join("compile_commands.json");
    let commands: Vec<CompileCommand> = serde_json::from_slice(&fs::read(&database)?)
        .with_context(|| format!("invalid compile database {}", database.display()))?;
    let mut objects = BTreeMap::new();
    for command in commands {
        let group = match (
            nvk.contains(&command.output),
            lavapipe.contains(&command.output),
        ) {
            (true, true) => "shared",
            (true, false) => "nvk_only",
            (false, true) => "lavapipe_only",
            (false, false) => continue,
        };
        let arguments = shlex::split(&command.command)
            .with_context(|| format!("invalid compile command for {}", command.output))?;
        let mut outputs = arguments.iter().enumerate().filter(|(_, arg)| *arg == "-o");
        ensure!(
            outputs
                .next()
                .and_then(|(at, _)| arguments.get(at + 1).map(String::as_str))
                == Some(command.output.as_str())
                && outputs.next().is_none(),
            "compile command does not uniquely produce {}",
            command.output
        );
        let object = PlannedObject {
            group,
            directory: command.directory,
            source: command.file,
            arguments,
        };
        match objects.entry(command.output) {
            Entry::Vacant(entry) => {
                entry.insert(object);
            }
            Entry::Occupied(entry) => ensure!(
                entry.get() == &object,
                "conflicting compile commands for {}",
                entry.key()
            ),
        }
    }
    for output in nvk.union(&lavapipe) {
        ensure!(
            objects.contains_key(output.as_str()),
            "missing compile command for {output}"
        );
    }
    let mut stdout = io::stdout().lock();
    serde_json::to_writer(&mut stdout, &objects)?;
    writeln!(stdout)?;
    Ok(())
}

pub fn build(jobs: u16) -> Result<()> {
    ensure!(
        cfg!(all(target_os = "linux", target_arch = "x86_64")),
        "the hosted lavapipe build requires an x86-64 Linux host"
    );
    let meson = tool("meson", GENERATORS)?;
    let ninja = tool("ninja", GENERATORS)?;
    let meson_version = capture(Command::new(&meson).arg("--version"))?;
    let ninja_version = capture(Command::new(&ninja).arg("--version"))?;

    let output = build_directory("lavapipe");
    fs::create_dir_all(&output)?;
    let run = tempfile::Builder::new()
        .prefix("run-")
        .tempdir_in(output)?
        .keep();
    println!("Hosted lavapipe build: {}", run.display());
    let source = run.join("source");
    let provenance = checkout_lavapipe_mesa(&source)?;
    let build = run.join("build");
    let prefix = run.join("install");
    exec(
        Command::new(&meson)
            .env("NINJA", &ninja)
            .args(["setup"])
            .arg(&build)
            .arg(&source)
            .arg("--prefix")
            .arg(&prefix)
            .args(OPTIONS),
    )?;
    exec(
        Command::new(&ninja)
            .arg("-C")
            .arg(&build)
            .arg("-j")
            .arg(jobs.to_string()),
    )?;
    exec(
        Command::new(&meson)
            .env_remove("DESTDIR")
            .args(["install", "--no-rebuild", "-C"])
            .arg(&build),
    )?;

    let icd = prefix.join("share/vulkan/icd.d/lvp_icd.x86_64.json");
    let library = prefix.join("lib/libvulkan_lvp.so");
    let icd_manifest: Value = serde_json::from_slice(&fs::read(&icd)?)?;
    ensure!(
        icd_manifest["ICD"]["library_path"]
            .as_str()
            .map(std::path::Path::new)
            == Some(library.as_path()),
        "the installed ICD manifest must point to {}",
        library.display()
    );
    let manifest = run.join("lavapipe.json");
    let evidence = json!({
        "host": "linux-x86_64",
        "source": provenance,
        "meson": meson_version.trim(),
        "ninja": ninja_version.trim(),
        "meson_options": OPTIONS,
        "meson_log": binding(&build.join("meson-logs/meson-log.txt"))?,
        "icd": binding(&icd)?,
        "library": binding(&library)?,
    });
    fs::write(&manifest, serde_json::to_vec_pretty(&evidence)?)?;
    println!("Hosted lavapipe ICD: {}", icd.display());
    println!("Hosted lavapipe manifest: {}", manifest.display());
    Ok(())
}
