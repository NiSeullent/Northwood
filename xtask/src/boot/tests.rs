//! Tests for boot commands and run reports.
use super::*;

#[test]
fn usb_hpet_handoff_needs_standalone_usb_diagnostics_before_work() {
    use clap::Parser;
    let prefix = ["xtask", "boot", "run", "--output", "/tmp/usb-hpet"];
    let parse = |options: Vec<&str>| {
        let parsed = crate::Cli::try_parse_from(prefix.into_iter().chain(options))?;
        let crate::Task::Boot(Command::Run(run)) = parsed.task else {
            panic!("boot run")
        };
        Ok::<_, clap::Error>(run)
    };
    assert!(!parse(vec![]).unwrap().hpet_enabled());
    assert!(parse(vec!["--hpet"]).is_err());
    let usb_line = "--usb-reset-check --usb-own-hpet --storage-check TEST:02020202-0202-0202-0202-020202020202";
    let usb = parse(vec!["--append", usb_line]).unwrap();
    assert!(usb.hpet_enabled());
    assert!(usb_hpet_preflight(&usb).is_ok());
    let virtual_clock = parse(vec!["--append", usb_line, "--icount", "1"]).unwrap();
    assert!(usb_hpet_preflight(&virtual_clock).is_err());
    let system = parse(vec!["--append", "--system --usb-own-hpet"]).unwrap();
    assert!(usb_hpet_preflight(&system).is_err());
    let off = qemu_machine("off", false);
    assert_eq!(
        off,
        "q35,smm=off,sata=off,usb=off,smbus=off,hpet=off,i8042=off,vmport=off"
    );
    assert_eq!(
        qemu_machine("off", true),
        off.replace("hpet=off", "hpet=on")
    );
    assert_eq!(
        qemu_machine("on", false),
        off.replace("i8042=off", "i8042=on")
    );
}

#[test]
fn audio_fixture_exposes_hpet_with_smm_disabled() {
    use clap::Parser;
    let parsed = crate::Cli::try_parse_from([
        "xtask",
        "boot",
        "run",
        "--output",
        "/tmp/audio-hpet",
        "--audio-check",
        "transport-duplex",
    ])
    .unwrap();
    let crate::Task::Boot(Command::Run(run)) = parsed.task else {
        panic!("boot run")
    };
    let command = machine::command(
        &run,
        machine::MachineOptions {
            qemu: Path::new("qemu-system-x86_64"),
            firmware: &Firmware {
                code: "firmware".into(),
                variables: "firmware-variables.fd".into(),
            },
            variables: Path::new("variables.fd"),
            format: "raw",
            system: true,
            bundle: false,
        },
    );
    let machine = command
        .get_args()
        .skip_while(|arg| *arg != "-machine")
        .nth(1)
        .unwrap();
    assert_eq!(machine, qemu_machine("off", true).as_str());
}

#[test]
fn parallel_cpu_default_and_icount_select_distinct_modes() {
    assert_eq!(qemu_accelerator(None, false), "tcg,thread=multi");
    for shift in 0..=10 {
        assert_eq!(qemu_accelerator(Some(shift), false), "tcg,thread=single");
    }
    assert_eq!(qemu_accelerator(None, true), "kvm");
}

#[test]
fn usb_host_takes_hex_vendor_and_product() {
    assert_eq!(parse_usb_id("13d3:3602"), Ok((0x13d3, 0x3602)));
    assert_eq!(parse_usb_id("0x0A12:0x0001"), Ok((0x0a12, 0x0001)));
    for bad in [
        "13d3",
        "13d3:",
        ":3602",
        "13d3:3602:1",
        "g000:0001",
        "10000:1",
    ] {
        assert!(parse_usb_id(bad).is_err(), "{bad}");
    }
}

#[test]
fn vfio_gpu_needs_kvm_and_a_full_pci_function() {
    use clap::Parser;
    let prefix = ["xtask", "boot", "run", "--output", "/tmp/vfio"];
    let parse = |options: Vec<&str>| crate::Cli::try_parse_from(prefix.into_iter().chain(options));
    assert_eq!(
        parse_pci_function("0000:01:00.0").as_deref(),
        Ok("0000:01:00.0")
    );
    assert_eq!(
        parse_pci_function("0000:0A:1f.7").as_deref(),
        Ok("0000:0a:1f.7")
    );
    for bad in ["01:00.0", "0000:01:00.8", "0000:01:00", "0000-01-00.0", ""] {
        assert!(parse_pci_function(bad).is_err(), "{bad}");
    }
    assert!(parse(vec!["--vfio-gpu", "0000:01:00.0"]).is_err());
    assert!(parse(vec!["--kvm", "--vfio-gpu", "0000:01:00.0"]).is_ok());
    assert!(
        parse(vec![
            "--kvm",
            "--vfio-gpu",
            "0000:01:00.0",
            "--iommu",
            "amd"
        ])
        .is_err()
    );
}

#[test]
fn vfio_gpu_takes_its_whole_slot_bound_to_vfio_pci() {
    let root = std::env::temp_dir().join(format!("stela-vfio-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    let device = |name: &str, driver: &str| {
        let directory = root.join(name);
        fs::create_dir_all(&directory).unwrap();
        std::os::unix::fs::symlink(format!("/drivers/{driver}"), directory.join("driver")).unwrap();
    };
    device("0000:01:00.0", "vfio-pci");
    device("0000:01:00.1", "vfio-pci");
    device("0000:02:00.0", "nvme");
    let functions = machine::vfio_functions("0000:01:00.0", &root).unwrap();
    assert_eq!(functions, ["0000:01:00.0", "0000:01:00.1"]);
    assert_eq!(
        machine::vfio_devices(&functions),
        [
            "pcie-root-port,id=host-gpu-port,chassis=1,slot=1,addr=0x1c",
            "vfio-pci,host=0000:01:00.0,bus=host-gpu-port,addr=0.0,rombar=0,multifunction=on",
            "vfio-pci,host=0000:01:00.1,bus=host-gpu-port,addr=0.1,rombar=0",
        ]
    );
    // The function asked for leads its slot.
    assert_eq!(
        machine::vfio_functions("0000:01:00.1", &root).unwrap(),
        ["0000:01:00.1", "0000:01:00.0"]
    );
    assert!(machine::vfio_functions("0000:03:00.0", &root).is_err());
    device("0000:01:00.2", "snd_hda_intel");
    assert!(machine::vfio_functions("0000:01:00.0", &root).is_err());
    fs::remove_dir_all(&root).unwrap();
}

#[test]
fn cpu_selection_preserves_model_and_baseline_overrides() {
    assert_eq!(qemu_cpu(None, false, true), "max,vendor=GenuineIntel");
    assert_eq!(qemu_cpu(None, false, false), "qemu64");
    assert_eq!(qemu_cpu(Some("Nehalem"), true, true), "Nehalem,la57=off");
    assert_eq!(
        qemu_cpu(Some("max,vendor=AuthenticAMD"), false, true),
        "max,vendor=AuthenticAMD"
    );
}

#[test]
fn run_report_keeps_degraded_requirements_and_the_first_guest_panic() {
    let serial = concat!(
        "GUEST| panic handlers ready\r\n",
        "UNMODELED pc=0x1000 device write at PA 0x2000: missing lifecycle\r\n",
        "GUEST| panic(cpu 0 caller 0x3000): first failure\r\n",
        "UNMODELED pc=0x4000 debug sample: PE is off\r\n",
        "GUEST| panic: nested failure\r\n",
        "STOP UNSUPPORTED a later terminal requirement\r\n",
        "STOP reset: amcc: DRAM access disabled\r\n",
    );
    assert_eq!(
        first_unsupported(serial),
        Some("pc=0x1000 device write at PA 0x2000: missing lifecycle")
    );
    assert_eq!(
        first_guest_panic(serial),
        Some("panic(cpu 0 caller 0x3000): first failure")
    );
    assert_eq!(
        first_unsupported("STOP UNSUPPORTED pc=0x1000 instruction\n"),
        Some("pc=0x1000 instruction")
    );
    assert_eq!(
        first_guest_panic("GUEST| panic: first failure\n"),
        Some("panic: first failure")
    );
    for ordinary in [
        "GUEST| UNMODELED guest text\n",
        "GUEST| panic flow initialized\n",
        "STOP reset: ordinary restart\n",
    ] {
        assert_eq!(first_unsupported(ordinary), None);
        assert_eq!(first_guest_panic(ordinary), None);
    }
}

#[test]
fn a_passed_check_does_not_hide_a_real_guest_panic() {
    assert_eq!(
        first_guest_panic("CHECK storage PASS\nGUEST| panic(cpu 0): failed\n"),
        Some("panic(cpu 0): failed")
    );
    assert_eq!(
        first_guest_panic("GUEST| panic handlers ready\nGUEST| panic flow initialized\n"),
        None
    );
}

#[test]
fn qemu_host_inventory_does_not_synthesize_a_native_worker_grant() {
    use clap::Parser;
    let prefix = ["xtask", "boot", "run", "--output", "/tmp/owned-cores"];
    let parsed =
        crate::Cli::try_parse_from(prefix.into_iter().chain(["--host-cores", "2"])).unwrap();
    let crate::Task::Boot(Command::Run(run)) = parsed.task else {
        panic!("boot run")
    };
    assert_eq!(run.host_cores, 2);
    assert_eq!(run.host_threads, 1);
    assert_eq!(machine::smp(&run), "cpus=2,sockets=1,cores=2,threads=1");
    assert!(run.append.is_none());
    let parsed = crate::Cli::try_parse_from(prefix.into_iter().chain([
        "--host-sockets",
        "2",
        "--host-cores",
        "16",
        "--host-threads",
        "2",
    ]))
    .unwrap();
    let crate::Task::Boot(Command::Run(run)) = parsed.task else {
        panic!("boot run")
    };
    assert_eq!(machine::smp(&run), "cpus=64,sockets=2,cores=16,threads=2");
    assert!(run.append.is_none());
    assert!(
        crate::Cli::try_parse_from(prefix.into_iter().chain(["--host-threads", "0",])).is_err()
    );
    for flag in ["--host-cores", "--host-sockets"] {
        for invalid in ["0", "65", "-1", "two"] {
            assert!(crate::Cli::try_parse_from(prefix.into_iter().chain([flag, invalid])).is_err());
        }
    }
}

#[test]
fn attached_storage_and_directory_volume_have_explicit_cli_contracts() {
    use clap::Parser;
    let cli = crate::Cli::try_parse_from([
        "xtask",
        "boot",
        "run",
        "--bundle",
        "/tmp/bundle",
        "--output",
        "/tmp/run",
        "--storage-disk",
        "/tmp/disk.raw",
        "--storage-serial",
        "ANS-1",
        "--directory-volume",
        "--append",
        "--system --storage ANS-1:GUID",
    ])
    .unwrap();
    let crate::Task::Boot(Command::Run(run)) = cli.task else {
        panic!("boot run");
    };
    assert_eq!(run.storage_serial, ["ANS-1"]);
    assert_eq!(run.storage_disk, [Path::new("/tmp/disk.raw")]);
    assert!(run.directory_volume);
    assert_eq!(run.append.as_deref(), Some("--system --storage ANS-1:GUID"));
    for extra in [
        vec!["--storage-disk", "/tmp/disk", "--storage-check"],
        vec!["--storage-serial", "ANS"],
        vec!["--directory-volume"],
        vec!["--network-check", "http://example.test/"],
        vec!["--usb-storage-disk", "/tmp/disk", "--storage-check"],
        vec!["--network", "--usb-network"],
        vec!["--network-model", "e1000e"],
        vec!["--network-self-check"],
        vec![
            "--network",
            "--network-self-check",
            "--network-check",
            "http://localhost/",
        ],
        vec!["--network", "--network-model", "unknown"],
        vec!["--usb-disk", "/tmp/disk", "--storage-check"],
        vec![
            "--storage-disk",
            "/tmp/disk",
            "--storage-serial",
            "bad,drive=x",
        ],
    ] {
        let args: Vec<_> = ["xtask", "boot", "run", "--output", "/tmp/run"]
            .into_iter()
            .chain(extra)
            .collect();
        assert!(crate::Cli::try_parse_from(args).is_err());
    }
}

#[cfg(any(target_os = "macos", target_os = "linux"))]
#[test]
fn process_image_identifies_the_executing_binary() {
    assert_eq!(
        fs::canonicalize(process_image(std::process::id()).unwrap()).unwrap(),
        fs::canonicalize(std::env::current_exe().unwrap()).unwrap(),
    );
}

#[test]
fn audio_modes_select_explicit_evidence_backends() {
    use clap::Parser;
    for (extra, expected) in [
        (vec!["--audio-check"], audio_check::Mode::Duplex),
        (
            vec!["--audio-check", "playback"],
            audio_check::Mode::Playback,
        ),
        (vec!["--audio-check", "duplex"], audio_check::Mode::Duplex),
    ] {
        let args: Vec<_> = ["xtask", "boot", "run", "--output", "/tmp/run"]
            .into_iter()
            .chain(extra)
            .collect();
        let cli = crate::Cli::try_parse_from(args).unwrap();
        let crate::Task::Boot(Command::Run(run)) = cli.task else {
            panic!("boot run");
        };
        assert_eq!(run.audio_check, Some(expected));
    }
}

#[test]
fn network_models_select_only_implemented_pci_drivers() {
    use clap::Parser;
    for (argument, expected) in [
        ("virtio", NetworkModel::Virtio),
        ("e1000e", NetworkModel::E1000e),
        ("igb", NetworkModel::Igb),
    ] {
        let cli = crate::Cli::try_parse_from([
            "xtask",
            "boot",
            "run",
            "--output",
            "/tmp/run",
            "--network",
            "--network-model",
            argument,
            "--network-check",
            "http://localhost/check",
        ])
        .unwrap();
        let crate::Task::Boot(Command::Run(run)) = cli.task else {
            panic!("boot run");
        };
        assert_eq!(run.network_model, Some(expected));
    }
}

#[test]
fn storage_disks_pair_with_distinct_serials() {
    let file = std::env::current_exe().unwrap();
    let one = storage_disks(std::slice::from_ref(&file), &[]).unwrap();
    assert_eq!(one, [(file.clone(), "STELAANS".to_owned())]);
    let two = [file.clone(), file.clone()];
    let serials = ["TARGET".to_owned(), "SOURCE".to_owned()];
    assert_eq!(storage_disks(&two, &serials).unwrap()[1].1, "SOURCE");
    assert!(storage_disks(&two, &[]).is_err());
    assert!(storage_disks(&two, &["A".to_owned(), "A".to_owned()]).is_err());
    assert!(storage_disks(&[file.with_extension("missing")], &[]).is_err());
}

#[test]
fn option_paths_double_their_commas() {
    let path = std::path::Path::new("/tmp/bundles/Mac14,2/variables.fd");
    assert_eq!(option_path(path), "/tmp/bundles/Mac14,,2/variables.fd");
}

#[test]
fn storage_mdts_is_optional_typed_and_requires_a_disk() {
    use clap::Parser;
    let base = ["xtask", "boot", "run", "--output", "/tmp/run"];
    for value in ["1", "7", "9"] {
        let cli = crate::Cli::try_parse_from(base.into_iter().chain([
            "--storage-disk",
            "/tmp/disk.raw",
            "--storage-mdts",
            value,
        ]))
        .unwrap();
        let crate::Task::Boot(Command::Run(run)) = cli.task else {
            panic!("boot run");
        };
        assert_eq!(run.storage_mdts, Some(value.parse().unwrap()));
    }
    let cli =
        crate::Cli::try_parse_from(base.into_iter().chain(["--storage-disk", "/tmp/disk.raw"]))
            .unwrap();
    let crate::Task::Boot(Command::Run(run)) = cli.task else {
        panic!("boot run");
    };
    assert_eq!(run.storage_mdts, None);
    for extra in [
        vec!["--storage-mdts", "9"],
        vec!["--storage-disk", "/tmp/disk.raw", "--storage-mdts", "0"],
        vec!["--storage-disk", "/tmp/disk.raw", "--storage-mdts", "10"],
        vec!["--storage-disk", "/tmp/disk.raw", "--storage-mdts", "256"],
        vec!["--storage-disk", "/tmp/disk.raw", "--storage-mdts", "-1"],
        vec![
            "--storage-disk",
            "/tmp/disk.raw",
            "--storage-mdts",
            "11,serial=x",
        ],
    ] {
        assert!(crate::Cli::try_parse_from(base.into_iter().chain(extra)).is_err());
    }
}
