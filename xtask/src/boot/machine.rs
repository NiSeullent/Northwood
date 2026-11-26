//! A run's QEMU machine: its CPU, memory, firmware, devices other than
//! storage and network, console and debugging.
use super::host_display;
use super::{
    Display, Firmware, Iommu, Run, UsbDiskModel, option_path, overlay, qemu_accelerator, qemu_cpu,
    qemu_machine,
};
use anyhow::{Result, ensure};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command as Process,
};

/// Paths used to create or reopen QEMU firmware variables.
pub(super) struct VariableStoreOptions<'a> {
    pub(super) qemu: &'a Path,
    pub(super) firmware: &'a Firmware,
    pub(super) output: &'a Path,
}

/// The firmware's variable store and its format. A qcow2 stick keeps its
/// variables in a qcow2 of its own, so `savevm` can checkpoint the whole
/// machine.
pub(super) fn variables(
    arguments: &Run,
    options: VariableStoreOptions<'_>,
) -> Result<(PathBuf, &'static str)> {
    let VariableStoreOptions {
        qemu,
        firmware,
        output,
    } = options;
    let checkpointed = arguments
        .usb_image
        .as_ref()
        .is_some_and(|image| image.extension().is_some_and(|e| e == "qcow2"));
    let variables = if checkpointed {
        let variables = arguments
            .usb_image
            .as_ref()
            .unwrap()
            .with_extension("vars.qcow2");
        if !variables.exists() {
            overlay(qemu, &firmware.variables, &variables)?;
        }
        (variables, "qcow2")
    } else {
        let variables = output.join("variables.fd");
        fs::copy(&firmware.variables, &variables)?;
        (variables, "raw")
    };
    ensure!(
        arguments.loadvm.is_none() || checkpointed,
        "--loadvm needs a .qcow2 --usb-image"
    );
    Ok(variables)
}

/// QEMU binary, firmware state and guest execution class.
pub(super) struct MachineOptions<'a> {
    pub(super) qemu: &'a Path,
    pub(super) firmware: &'a Firmware,
    pub(super) variables: &'a Path,
    pub(super) format: &'a str,
    pub(super) system: bool,
    pub(super) bundle: bool,
}

/// QEMU with the machine, its CPUs and memory, and the firmware.
pub(super) fn command(arguments: &Run, options: MachineOptions<'_>) -> Process {
    let MachineOptions {
        qemu,
        firmware,
        variables,
        format,
        system,
        bundle,
    } = options;
    let usb = arguments.usb_image.is_some();
    let mut command = Process::new(qemu);
    // A USB's startup choices take a keyboard. System mode places the guest
    // beside the window with LA57, which the selected CPU exposes unless
    // `--no-la57` hides it. KVM's host model hides the invariant TSC, as
    // migration could change its rate; a run never migrates, and the mount
    // counts time and dates TLS certificates by it, as on a PC.
    let cpu = qemu_cpu(
        arguments
            .cpu
            .as_deref()
            .or(arguments.kvm.then_some("host,+invtsc")),
        arguments.no_la57,
        usb || system,
    );
    let keyboard = if usb { "on" } else { "off" };
    let memory = arguments.memory.clone().unwrap_or_else(|| {
        match (bundle, system) {
            (true, _) => "12G",
            (false, true) => "8G",
            (false, false) => "1G",
        }
        .into()
    });
    command
        .arg("-machine")
        .arg(qemu_machine(keyboard, arguments.hpet_enabled()))
        .args([
            "-accel",
            qemu_accelerator(arguments.icount, arguments.kvm),
            "-cpu",
            &cpu,
            "-smp",
        ])
        .arg(smp(arguments))
        .args(
            arguments
                .icount
                .map(|shift| ["-icount".into(), format!("shift={shift},sleep=off")])
                .into_iter()
                .flatten(),
        )
        .args(["-m", &memory, "-nodefaults", "-no-reboot"])
        // Every change to a mapped page of guest RAM flushes the guest's
        // whole mapping, which under nested virtualization discards the
        // outer hypervisor's shadow tables: fault it all in before it runs.
        .args(arguments.kvm.then_some("-mem-prealloc"))
        .args(
            arguments
                .loadvm
                .iter()
                .flat_map(|tag| ["-loadvm", tag.as_str()]),
        )
        .arg("-drive")
        .arg(format!(
            "if=pflash,format=raw,readonly=on,file={}",
            option_path(&firmware.code)
        ))
        .arg("-drive")
        .arg(format!(
            "if=pflash,format={format},file={}",
            option_path(variables)
        ));
    // An IOMMU precedes the PCI devices it translates for.
    match arguments.iommu {
        Some(Iommu::Intel) => command.args(["-device", "intel-iommu,intremap=off"]),
        Some(Iommu::Amd) => command.args(["-device", "amd-iommu,intremap=off"]),
        None => &mut command,
    };
    command
}

pub(super) fn smp(arguments: &Run) -> String {
    let sockets = u32::from(arguments.host_sockets);
    let cores = u32::from(arguments.host_cores);
    let threads = u32::from(arguments.host_threads);
    let cpus = sockets * cores * threads;
    format!("cpus={cpus},sockets={sockets},cores={cores},threads={threads}")
}

/// How the mount stops QEMU: the debug-exit port, or QMP for an audio check.
pub(super) fn attach_exit(arguments: &Run, output: &Path, command: &mut Process) {
    if let Some(mode) = arguments.audio_check {
        mode.attach(command, output);
        // isa-debug-exit calls exit() before the audio backend finalizes its
        // WAV. Quit through QMP after the mount stops so cleanup can finish.
        command.args(["-qmp", "stdio"]);
    } else {
        command.args(["-device", "isa-debug-exit,iobase=0xf4,iosize=4"]);
    }
}

/// The display a bundle's kernel draws on through the mount's GOP, a panel
/// of the host monitor's mode and refresh ([`host_display`]), and
/// `--input`'s virtio keyboard and tablet.
pub(super) fn attach_display(arguments: &Run, bundle: bool, command: &mut Process) {
    if bundle {
        let panel = host_display::host().and_then(host_display::emulated);
        command.arg("-device").arg(host_display::vga_device(panel));
    }
    if arguments.input {
        command.args([
            "-device",
            "virtio-keyboard-pci",
            "-device",
            "virtio-tablet-pci",
        ]);
    }
}

/// The root port `--vfio-gpu` hangs behind. An explicit slot, past those the
/// other parts number in order, leaves their addresses as they were.
const VFIO_PORT: &str = "pcie-root-port,id=host-gpu-port,chassis=1,slot=1,addr=0x1c";

/// `--vfio-gpu`'s functions, the given one first: all of its slot, since
/// they share a reset and usually an IOMMU group. Each must be vfio-pci's.
pub(super) fn vfio_functions(function: &str, devices: &Path) -> Result<Vec<String>> {
    let (slot, _) = function.rsplit_once('.').expect("parsed as a PCI function");
    let mut functions = Vec::new();
    for entry in fs::read_dir(devices)? {
        let name = entry?.file_name().to_string_lossy().into_owned();
        if name
            .rsplit_once('.')
            .is_some_and(|(other, _)| other == slot)
        {
            let driver = fs::read_link(devices.join(&name).join("driver")).ok();
            ensure!(
                driver.as_deref().and_then(Path::file_name) == Some("vfio-pci".as_ref()),
                "{name} is not bound to vfio-pci: run through xtask/scripts/vfio-gpu.sh"
            );
            functions.push(name);
        }
    }
    ensure!(
        functions.iter().any(|name| name == function),
        "the host has no PCI function {function}"
    );
    functions.sort_by_key(|name| (name != function, name.clone()));
    Ok(functions)
}

/// The QEMU devices of `functions`, in one multifunction slot of the port.
///
/// No option ROM: Stela drives the card with no firmware help, and a card nothing
/// has posted since it lost power hangs the firmware in the ROM's GOP driver.
pub(super) fn vfio_devices(functions: &[String]) -> Vec<String> {
    let mut devices = vec![VFIO_PORT.to_owned()];
    devices.extend(functions.iter().enumerate().map(|(index, host)| {
        let multifunction = if index == 0 { ",multifunction=on" } else { "" };
        format!("vfio-pci,host={host},bus=host-gpu-port,addr=0.{index},rombar=0{multifunction}")
    }));
    devices
}

/// `--vfio-gpu`'s root port and the card's functions.
pub(super) fn attach_vfio_gpu(arguments: &Run, command: &mut Process) -> Result<()> {
    if let Some(function) = &arguments.vfio_gpu {
        let functions = vfio_functions(function, Path::new("/sys/bus/pci/devices"))?;
        for device in vfio_devices(&functions) {
            command.arg("-device").arg(device);
        }
    }
    Ok(())
}

/// `--host-audio`'s HD Audio controller and codec.
pub(super) fn attach_audio(arguments: &Run, command: &mut Process) {
    if arguments.host_audio {
        command.args([
            "-audiodev",
            "none,id=host-audio",
            "-device",
            "intel-hda,id=host-hda",
            "-device",
            "hda-duplex,bus=host-hda.0,audiodev=host-audio",
        ]);
    }
}

/// The host's xHCI controller, when a USB device needs it. Eight USB2 root
/// ports seat the boot stick, the input devices and lent devices with none
/// behind a hub QEMU adds when the ports run out, which runs at full speed.
pub(super) fn attach_usb(arguments: &Run, command: &mut Process) {
    let host_usb = arguments.usb_image.is_some()
        || arguments.usb_input
        || arguments.usb_network
        || !arguments.usb_disk.is_empty()
        || !arguments.usb_host.is_empty();
    if host_usb {
        command.args([
            "-device",
            if arguments.usb_disk_model == Some(UsbDiskModel::UasUsb2) {
                "qemu-xhci,id=host-usb,p2=8,p3=0"
            } else {
                "qemu-xhci,id=host-usb,p2=8"
            },
        ]);
    }
}

/// `--usb-input`'s keyboard, tablet and mouse.
pub(super) fn attach_usb_input(arguments: &Run, command: &mut Process) {
    if arguments.usb_input {
        for device in ["usb-kbd", "usb-tablet", "usb-mouse"] {
            command
                .arg("-device")
                .arg(format!("{device},bus=host-usb.0"));
        }
    }
}

/// `--usb-host`'s devices of this PC.
pub(super) fn attach_usb_host(arguments: &Run, command: &mut Process) {
    for (vendor, product) in &arguments.usb_host {
        command.arg("-device").arg(format!(
            "usb-host,vendorid={vendor:#06x},productid={product:#06x},bus=host-usb.0"
        ));
    }
}

/// QEMU's SDL window, drawn with GL. QEMU draws its window on its main
/// thread while holding the lock every device access takes; measured, the
/// GTK window held it up to 70 ms at a time (it paints the window in
/// software), and this one a few milliseconds.
const WINDOW: &str = "sdl,gl=on";

/// The UART, logged to `serial`, and the monitor.
pub(super) fn attach_console(arguments: &Run, serial: &Path, command: &mut Process) {
    if let Some(port) = arguments.console {
        let display = match arguments.display {
            Display::None => "none",
            Display::Window if cfg!(target_os = "macos") => "cocoa",
            Display::Window => WINDOW,
        };
        command
            .args(["-display", display, "-chardev"])
            .arg(format!(
                "socket,id=uart,host=127.0.0.1,port={port},server=on,wait=off,logfile={}",
                option_path(serial)
            ))
            .args(["-serial", "chardev:uart"]);
    } else {
        attach_uart(arguments, serial, command);
    }
    match arguments.monitor {
        Some(port) => command
            .arg("-monitor")
            .arg(format!("telnet:127.0.0.1:{port},server=on,wait=off")),
        None => command.args(["-monitor", "none"]),
    };
    if let Some(port) = arguments.qmp {
        command
            .arg("-qmp")
            .arg(format!("tcp:127.0.0.1:{port},server=on,wait=off"));
    }
}

/// The UART on the run's display, logged to `serial`.
fn attach_uart(arguments: &Run, serial: &Path, command: &mut Process) {
    match arguments.display {
        Display::None => command
            .args(["-display", "none"])
            .arg("-serial")
            .arg(format!("file:{}", serial.display())),
        // Cocoa sizes its text console from the chardev; GTK's VTE console
        // sizes itself and refuses the options.
        Display::Window if cfg!(target_os = "macos") => command
            .args(["-display", "cocoa", "-chardev"])
            .arg(format!(
                "vc,id=uart,cols=132,rows=50,logfile={}",
                option_path(serial)
            ))
            .args(["-serial", "chardev:uart"]),
        Display::Window => command
            .args(["-display", WINDOW, "-chardev"])
            .arg(format!("vc,id=uart,logfile={}", option_path(serial)))
            .args(["-serial", "chardev:uart"]),
    };
}

/// `--gdb`: a paused machine waiting for the debugger.
pub(super) fn attach_debugger(arguments: &Run, command: &mut Process) {
    if let Some(port) = arguments.gdb {
        command
            .arg("-gdb")
            .arg(format!("tcp:127.0.0.1:{port}"))
            .arg("-S");
        eprintln!("paused: gdb -ex 'target remote 127.0.0.1:{port}'");
    }
}
