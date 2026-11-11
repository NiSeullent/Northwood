//! A run's report: what the guest printed checked against what the run
//! asked for, written to `run.json`, and the run's exit code.
use super::{
    FAILED_EXIT, IDLE_EXIT, PayloadMode, Run, TIMEOUT_EXIT, audio_check, binding, disks,
    first_guest_panic, first_unsupported, host_firmware, network, preboot::Preboot,
    qemu_accelerator, staged, storage::Paths, verify_kernel,
};
use anyhow::{Result, bail, ensure};
use serde_json::{Value, json};
use std::{
    fs,
    path::PathBuf,
    process::{ExitCode, ExitStatus},
    time::Instant,
};

/// What a finished run leaves for its report.
pub(super) struct Outcome {
    pub(super) paths: Paths,
    pub(super) preboot: Option<Preboot>,
    pub(super) disks: Vec<disks::Disk>,
    pub(super) attached: Vec<(PathBuf, String)>,
    pub(super) network: network::Check,
    /// `stela.cfg`.
    pub(super) options: String,
    pub(super) command_line: Vec<String>,
    /// The launch-time bindings of QEMU, the application and `source.bin`.
    pub(super) qemu: Value,
    pub(super) qemu_version: String,
    pub(super) qemu_process_image: Option<Value>,
    pub(super) image: Value,
    pub(super) source: Option<Value>,
    pub(super) started: Instant,
    /// QEMU's exit, or `None` when the run timed out.
    pub(super) status: Option<ExitStatus>,
}

/// Write `run.json`, optionally print it, then fail on a check that did not pass.
pub(super) fn finish(arguments: &Run, outcome: Outcome, print: bool) -> Result<ExitCode> {
    let Outcome {
        paths,
        preboot,
        disks,
        attached,
        network,
        options,
        command_line,
        qemu,
        qemu_version,
        qemu_process_image,
        image,
        source,
        started,
        status,
    } = outcome;
    let output = &paths.output;
    let mut code = match status {
        Some(status) => status.code().map_or(TIMEOUT_EXIT, |code| code as u8),
        None => TIMEOUT_EXIT,
    };
    let qemu_returncode = code;
    let serial_text = fs::read_to_string(&paths.serial).unwrap_or_default();
    let mount_exit = arguments
        .audio_check
        .map(|_| audio_check::guest_exit(&serial_text));
    if let Some(result) = &mount_exit {
        code = result.as_ref().copied().unwrap_or(FAILED_EXIT);
    }
    let unsupported = first_unsupported(&serial_text);
    // The installed-system scenario ends at the bundle's entry: what the
    // engine meets after it is reported but does not fail the run.
    let mut entered = false;
    let verification = if let Some(preboot) = &preboot {
        let checked = preboot.verify(&serial_text);
        entered = checked.is_ok();
        Some(
            checked
                .unwrap_or_else(|error| json!({ "passed": false, "reason": format!("{error:#}") })),
        )
    } else if arguments.bundle.is_none() && arguments.module.is_none() && !arguments.guestless() {
        let checked = verify_kernel(&serial_text);
        if checked.is_err() && code == IDLE_EXIT {
            code = FAILED_EXIT;
        }
        Some(checked.map_or_else(
            |error| json!({ "passed": false, "reason": error }),
            |()| json!({ "passed": true }),
        ))
    } else {
        None
    };
    // Before the outputs are bound, as it writes the peer's snapshot.
    let network = network.verify(arguments, output, &serial_text)?;
    let outputs: Vec<Value> = [
        "serial.txt",
        "stdout.txt",
        "stderr.txt",
        "network-check-peer.txt",
        audio_check::WAV,
    ]
    .into_iter()
    .map(|name| output.join(name))
    .filter(|path| path.exists())
    .map(|path| binding(&path))
    .collect::<Result<_>>()?;
    let storage = match arguments.storage_check {
        true => Some(disks::verify(&disks, &paths.serial)),
        false => None,
    };
    let host_firmware = arguments
        .host_firmware
        .as_ref()
        .map(|_| host_firmware::report(&paths.volume, &serial_text))
        .transpose()?;
    let audio = arguments.audio_check.map(|mode| {
        if let Some(Err(error)) = &mount_exit {
            bail!("invalid guest exit status: {error:#}");
        }
        audio_check::normal_exit(
            qemu_returncode,
            &fs::read_to_string(output.join("stdout.txt"))?,
        )?;
        let mut report = audio_check::verify(mode, &serial_text, output)?;
        report["qemu_shutdown"] = json!("acknowledged-qmp-quit");
        report["mount_returncode"] =
            json!(mount_exit.as_ref().and_then(|result| result.as_ref().ok()));
        Ok::<_, anyhow::Error>(report)
    });
    let network_report = network.as_ref().map(|result| match result {
        Ok(report) => report.clone(),
        Err(error) => json!({ "passed": false, "reason": error.to_string() }),
    });
    let mut report = json!({
        "command": command_line,
        "qemu": qemu,
        "qemu_version": qemu_version,
        "host_firmware": host_firmware,
        "qemu_process_image": qemu_process_image,
        "image": image,
        "source": source,
        "usb_image": arguments.usb_image,
        "usb_serial": arguments.usb_image.as_ref().map(|_| &arguments.usb_serial),
        "usb_writable": arguments.usb_writable,
        "no_la57": arguments.no_la57,
        "qemu_host_cores": arguments.host_cores,
        "qemu_host_threads": arguments.host_threads,
        "qemu_accelerator": qemu_accelerator(arguments.icount, arguments.kvm),
        "options": options,
        "scenario": verification,
        "bundle": arguments.bundle,
        "preboot_disk": arguments.preboot_disk,
        "directory_volume": arguments.directory_volume,
        "restore_manifest": arguments.restore_manifest,
        "verify_on_read": arguments.verify_on_read,
        "storage_disks": attached
            .iter()
            .map(|(path, serial)| json!({ "path": path, "serial": serial }))
            .collect::<Vec<_>>(),
        "first_unsupported": unsupported,
        "first_guest_panic": first_guest_panic(&serial_text),
        "returncode": code,
        "qemu_returncode": qemu_returncode,
        "payload": arguments.payload.map(|mode| {
            matches!(mode, PayloadMode::Stage | PayloadMode::StagePath)
        }),
        "staging": paths.target.exists().then(|| staged(&paths.target)).transpose()?,
        "timed_out": status.is_none(),
        "wall_seconds": started.elapsed().as_secs_f64(),
        "outputs": outputs,
        "audio": audio.as_ref().map(|result| match result {
            Ok(report) => report.clone(),
            Err(error) => json!({ "passed": false, "reason": format!("{error:#}") }),
        }),
        "storage": storage.as_ref().map(|result| match result {
            Ok(disks) => disks.clone(),
            Err(error) => json!({ "error": format!("{error:#}") }),
        }),
        "network": network_report,
    });
    report["hpet"] = json!(arguments.hpet_enabled());
    fs::write(output.join("run.json"), format!("{report:#}\n"))?;
    if print {
        println!("{report:#}");
    }
    if let Some(Err(error)) = storage {
        return Err(error.context("host storage check failed"));
    }
    ensure!(
        report["host_firmware"].is_null() || report["host_firmware"]["passed"] == true,
        "host firmware did not survive the before/after EBS verification"
    );
    if let Some(Err(error)) = network {
        bail!("host network check failed: {error}");
    }
    if let Some(Err(error)) = audio {
        return Err(error.context("native audio check failed"));
    }
    Ok(ExitCode::from(match (preboot.is_some(), entered) {
        (true, true) => 0,
        (true, false) => FAILED_EXIT,
        (false, _) => code,
    }))
}
