//! Prepare the kernel boot bundle that the UEFI mount hands to the ARM kernel.
use crate::{
    args::parse_address,
    restore::{self, components},
};
use anyhow::{Context, Result, bail, ensure};
use clap::{Args, Subcommand};
use std::{fs::OpenOptions, io::Write, path::PathBuf};
use stela_formats::payload::Purpose;
use stela_loader::aux_kc::{Linked, TRUSTED_CODE_FILE};
use stela_loader::boot_bundle::{
    self, Input, Inputs, Options, OverlayProperty, OwnedRamdisk, RAMDISK_FILE, RootHash, Source,
    Target,
};
use stela_loader::boot_policy::{
    KextGrant, OsEnvironment, SealedVolume, SecurityMode, SecurityPolicy,
};
use stela_loader::handoff::{self, BootCommand as FirmwareBootCommand};
use stela_loader::im4m;
use stela_loader::root_selection::Root;
use stela_loader::trust_cache::CodeHash;

mod board_firmware;
mod disk;
mod ticket;

const MANIFEST_FILE: &str = "bundle.json";
const KERNEL_LIMIT: u64 = 512 * 1024 * 1024;
const COMPONENT_LIMIT: u64 = 64 * 1024 * 1024;
/// The largest BuildManifest a personalized restore reads.
const BUILD_MANIFEST_LIMIT: u64 = 64 * 1024 * 1024;
const RAMDISK_LIMIT: u64 = 2 * 1024 * 1024 * 1024;
/// The OS environment of a restore ramdisk: its `/osenvironments` child
/// turns off the SEP firmware load and protected data access and selects
/// ephemeral storage.
const RESTORE_ENVIRONMENT: OsEnvironment = OsEnvironment::DarwinosRamdisk;
/// The console boot-args when `--boot-args` gives none: verbose boot, with
/// the kernel console and panic text on the modeled `uart0`, so every run's
/// log keeps the guest's own account of its boot.
const DEFAULT_BOOT_ARGS: &str = "-v serial=3 serial-device-name=uart0";
/// The boot-args a restore host has iBoot pass to the restore ramdisk
/// (idevicerestore `restore.c`): the ramdisk root, NAND reformat permission,
/// the progress UI and restore mode. `-restore` is read by the kernel and
/// about twenty kexts; IOUSBDeviceConfigurator, for one, leaves the USB
/// device controllers to restored only in restore mode.
const RESTORE_MODE_ARGS: &[&str] = &[
    "rd=md0",
    "nand-enable-reformat=0x1",
    "-progress",
    "-restore",
];

/// The BuildManifest components, and their IM4P types, a boot mode loads.
struct Roles {
    kernel: &'static str,
    device_tree: &'static str,
    trust_cache: (&'static str, &'static str),
    sptm: &'static str,
    txm: &'static str,
    sep: &'static str,
    /// An IM4P ramdisk component and its type.
    ramdisk: Option<(&'static str, &'static str)>,
    /// The sealed volume the kernel roots from, by its root-hash component.
    root_hash: Option<(&'static str, SealedVolume)>,
    /// Whether the identity's global manifest admits every boot object of
    /// the mode. The restore objects sit outside it: that manifest is the
    /// installed system's root ticket, which the restore host answers with.
    manifest_admits_all: bool,
}

const INSTALLED: Roles = Roles {
    kernel: "KernelCache",
    device_tree: "DeviceTree",
    trust_cache: ("StaticTrustCache", "trst"),
    sptm: "Ap,SecurePageTableMonitor",
    txm: "Ap,TrustedExecutionMonitor",
    sep: "SEP",
    ramdisk: None,
    root_hash: Some(("SystemVolume", SealedVolume::System)),
    manifest_admits_all: true,
};

const RESTORE: Roles = Roles {
    kernel: "RestoreKernelCache",
    device_tree: "RestoreDeviceTree",
    trust_cache: ("RestoreTrustCache", "rtsc"),
    sptm: "Ap,RestoreSecurePageTableMonitor",
    txm: "Ap,RestoreTrustedExecutionMonitor",
    sep: "RestoreSEP",
    ramdisk: Some(("RestoreRamDisk", "rdsk")),
    root_hash: None,
    manifest_admits_all: false,
};

#[derive(Subcommand)]
pub enum BootCommand {
    /// Lay out a model's kernel collection, SPTM, TXM, DeviceTree and trust
    /// cache in guest physical memory for `stela.efi`.
    Prepare(Box<Prepare>),
    /// Check a prepared bundle's DeviceTree against the engine's board
    /// models (`machine::soc::board`) and mount its whole machine
    /// (`machine::mount`), as `stela.efi` does before entry, and print the
    /// board the engine would run or the reason it refuses.
    Check(Check),
}

#[derive(Args)]
pub struct Check {
    /// A directory `boot prepare` wrote.
    bundle: PathBuf,
}

impl Check {
    fn run(self) -> Result<()> {
        let tree = std::fs::read(
            self.bundle
                .join(stela_loader::boot_bundle::DEVICE_TREE_FILE),
        )
        .with_context(|| format!("{} has no DeviceTree", self.bundle.display()))?;
        let mut board =
            stela_soc_data::board::board(&tree).context("the engine refuses this board")?;
        let gpu_gate = gpu_gate(&tree)?;
        let mount = mount(&self.bundle, &tree, &mut board)
            .context("the engine refuses to mount this board")?;
        let hex = |value: u64| format!("{value:#x}");
        let report = serde_json::json!({
            "chip": board.cpus.chip.name(),
            "processors": board.cpus.processors.iter().map(|pe| serde_json::json!({
                "midr": hex(pe.midr),
                "mpidr": hex(pe.mpidr),
                "clock_hz": pe.clock,
                "caches": pe.caches.is_some(),
            })).collect::<Vec<_>>(),
            "aic": {
                "physical": hex(board.aic.physical),
                "inputs": board.aic.inputs,
                "dies": board.aic.layout.dies(),
            },
            "lockdown": board.lockdown.iter().map(|block| serde_json::json!({
                "block": block.name,
                "physical": hex(block.physical),
                "control": hex(block.control),
            })).collect::<Vec<_>>(),
            "cpu_start": board.cpu_start.map(|(physical, clusters)| serde_json::json!({
                "physical": hex(physical),
                "clusters": clusters,
            })),
            "power_states": board.power_states.as_ref().map(|states| states.registers.len()),
            "clusters": board.clusters.len(),
            "memory_cache": board.memory_cache.is_some(),
            "gpu_gate": gpu_gate,
            "mount": mount,
        });
        println!("{}", serde_json::to_string_pretty(&report)?);
        Ok(())
    }
}

/// Where a boot that opened a GPU puts its first StelaGPU gate in `tree`, as
/// `stela.efi` does before it models the board, and whether the board's AIC
/// then takes the gate's input (vector V is die V / MAX_IRQ, IRQ V % MAX_IRQ).
fn gpu_gate(tree: &[u8]) -> Result<serde_json::Value> {
    use stela_loader::apple_device_tree::DeviceTree;
    let mut tree = DeviceTree::parse(tree).map_err(|error| anyhow::anyhow!("{error}"))?;
    let placement = match stela_loader::stela_gpu::place_gpu_gate(
        &mut tree,
        0,
        stela_contracts::gate::window::BYTES,
    ) {
        Ok(placement) => placement,
        Err(error) => return Ok(serde_json::json!({ "refusal": error.to_string() })),
    };
    let hidden = stela_loader::devices::hide_agx(&mut tree);
    let board = stela_soc_data::board::board(&tree.serialize());
    Ok(serde_json::json!({
        "physical": format!("{:#x}", placement.physical),
        "vector": placement.irq,
        "hidden": hidden,
        "board": match board {
            Ok(board) => {
                let per_die = u32::from(board.aic.layout.max_irq());
                serde_json::json!({
                    "die": placement.irq / per_die,
                    "input": placement.irq % per_die,
                    "inputs": board.aic.inputs,
                    "dies": board.aic.layout.dies(),
                })
            }
            Err(error) => serde_json::json!({ "refusal": error.to_string() }),
        },
    }))
}

/// Mount `board`'s whole machine as `stela.efi` does, with stand-in host
/// inputs: a sparse disk for any internal SSD, a monitor that discards
/// frames, zero entropy, a zero ECID when the bundle names none and no
/// cables. Returns the mount's `DEVICES` lines and the
/// PMGR domains whose lifecycle stays refused.
fn mount(
    bundle: &std::path::Path,
    tree: &[u8],
    board: &mut stela_soc_data::board::Board,
) -> Result<serde_json::Value> {
    use stela_runtime::machine::mount;
    use stela_soc::machine::{
        ConsoleBusy, ConsoleSink, Scanout, ScanoutFormat, ScanoutMode, UsbCables,
    };
    struct Discard;
    /// An internal SSD of zeros that keeps what is written.
    struct Sparse {
        identity: stela_drivers::block::Identity,
        written: std::collections::BTreeMap<u64, Vec<u8>>,
    }
    impl stela_drivers::block::BlockDevice for Sparse {
        fn identity(&self) -> &stela_drivers::block::Identity {
            &self.identity
        }
        fn read(
            &mut self,
            lba: u64,
            buffer: &mut [u8],
        ) -> std::result::Result<(), stela_drivers::block::BlockError> {
            self.identity.check(lba, buffer.len())?;
            let size = self.identity.block_size as usize;
            for (index, block) in buffer.chunks_mut(size).enumerate() {
                match self.written.get(&(lba + index as u64)) {
                    Some(bytes) => block.copy_from_slice(bytes),
                    None => block.fill(0),
                }
            }
            Ok(())
        }
        fn write(
            &mut self,
            lba: u64,
            data: &[u8],
        ) -> std::result::Result<(), stela_drivers::block::BlockError> {
            self.identity.check_write(lba, data.len())?;
            let size = self.identity.block_size as usize;
            for (index, block) in data.chunks(size).enumerate() {
                self.written.insert(lba + index as u64, block.to_vec());
            }
            Ok(())
        }
    }
    impl ConsoleSink for Discard {
        fn transmit(&mut self, _: u8) -> std::result::Result<(), ConsoleBusy> {
            Ok(())
        }
    }
    impl Scanout for Discard {
        fn mode(&self) -> ScanoutMode {
            ScanoutMode {
                width: 1920,
                height: 1080,
                format: ScanoutFormat::Bgra8888,
            }
        }
        fn present(&mut self, _: &[u8]) -> std::result::Result<(), stela_contracts::Refusal> {
            Ok(())
        }
    }
    let manifest = std::fs::read(bundle.join("bundle.json"))
        .with_context(|| format!("{} has no manifest", bundle.display()))?;
    let manifest = stela_formats::boot_bundle::Manifest::parse(&manifest)
        .map_err(|error| anyhow::anyhow!("bundle manifest: {error:?}"))?;
    // A bundle prepared for no particular machine has no ECID; the SEP
    // takes a stand-in one, as the SSD takes a stand-in disk.
    if let Some(sep) = board.sep.as_mut() {
        sep.ecid.get_or_insert(0);
    }
    let unavailable = match board.power_states.as_mut() {
        Some(states) => states.apply_admission(tree, &manifest.hidden_devices)?,
        None => 0,
    };
    // Board device firmware travels in the bundle by guest file name; the
    // boot images themselves are not device firmware.
    let files = manifest
        .files
        .iter()
        .filter(|file| file.bytes <= 16 << 20)
        .map(|file| Ok((file.name.clone(), std::fs::read(bundle.join(&file.name))?)))
        .collect::<std::io::Result<Vec<_>>>()
        .context("bundle file")?;
    let firmware = |name: &str| {
        files
            .iter()
            .find(|(file, _)| file == name)
            .map(|(_, bytes)| bytes.as_slice())
    };
    let mut log = Vec::new();
    let (devices, _) = mount::devices(
        board,
        Discard,
        mount::Inputs {
            programmed: mount::memory_locks(
                manifest
                    .read_only_region
                    .map(|region| (region.base, region.bytes)),
                &manifest.core_locks.unwrap_or_default(),
            ),
            reset: Some(manifest.entry.physical_address),
            now: 0,
            wall: None,
            rtc_anchor: None,
            backing: board.ans.is_some().then(|| {
                Box::new(Sparse {
                    identity: stela_drivers::block::Identity {
                        transport: stela_drivers::block::Transport::Nvme,
                        location: "check".into(),
                        model: "sparse".into(),
                        serial: "STELACHECK".into(),
                        blocks: 256 << 18,
                        block_size: 4096,
                        read_only: false,
                        features: Default::default(),
                    },
                    written: Default::default(),
                }) as Box<dyn stela_drivers::block::BlockDevice>
            }),
            sep_state: None,
            sep_observe: false,
            sep_recovery_policy_nonce: None,
            entropy: Box::new(|bytes: &mut [u8]| {
                bytes.fill(0);
                Ok(())
            }),
            restore: false,
            input: false,
            usb: UsbCables::default(),
            scanout: Some(Box::new(Discard)),
            firmware: &firmware,
            build: manifest
                .target
                .as_ref()
                .map(|target| stela_formats::builds::find(&target.product_build))
                .transpose()
                .map_err(anyhow::Error::msg)?,
        },
        &mut |arguments| log.push(arguments.to_string().trim_end().to_owned()),
    )?;
    Ok(serde_json::json!({
        "devices": {
            "gpio": board.gpio.is_some(),
            "spi_nor": board.spi_nor.is_some(),
            "i2c": board.i2cs.len(),
            "typec_phys": board.typec_phys.len(),
            "darts": board.darts.len(),
            "ans": board.ans.is_some(),
            "smc": board.smc.is_some(),
            "pmp": board.pmp.is_some(),
            "gfx": board.gfx.is_some(),
            "dcp": board.dcp.is_some(),
            "sep": board.sep.is_some(),
            "usb": board.usbs.len(),
            "spmi": board.spmis.len(),
            "pmgr": board.power_states.is_some(),
            "mcc": board.memory_cache.is_some(),
            "amcc": board.memory_controller.is_some(),
            "cpu_clusters": board.clusters.len(),
            "cpu_start": board.cpu_start.is_some(),
            "lockdown": board.lockdown.len(),
            "wake_masks": board.wake_masks.is_some(),
        },
        "log": log,
        "pmgr_unavailable_resources": unavailable,
        "pmgr_lifecycle_refusals": devices
            .power_refusals()
            .iter()
            .map(|(address, reason)| format!("{address:#x} {reason}"))
            .collect::<Vec<_>>(),
    }))
}

#[derive(Args)]
pub struct Prepare {
    #[arg(skip)]
    quiet: bool,
    #[arg(long, required_unless_present = "disk", conflicts_with = "disk")]
    ipsw: Option<PathBuf>,
    /// The `Ap,ProductType`, such as Mac14,3.
    #[arg(long, required_unless_present = "disk")]
    model: Option<String>,
    /// Build the bundle from an installed system's Preboot volume on this
    /// disk image instead of an IPSW.
    #[arg(long, conflicts_with_all = ["restore", "variant", "cryptex_trust_caches"])]
    disk: Option<PathBuf>,
    /// The `boot/<hash>` directory to use when the Preboot volume has several.
    #[arg(long, requires = "disk")]
    boot_hash: Option<String>,
    /// The NVRAM `boot-volume` to boot, `TYPE:PARTITION:GROUP`, choosing the
    /// disk's container and volume group as LLB does.
    #[arg(long, requires = "disk")]
    boot_volume: Option<String>,
    /// Boot the macOS update staged on the disk's Update volume
    /// (`stela_loader::upgrade`) instead of its installed system.
    #[arg(long, requires = "disk", conflicts_with_all = ["boot_hash"])]
    upgrade: bool,
    /// Boot a recoveryOS the disk holds (`stela_loader::recovery`) instead
    /// of its installed system: `paired`, its container's Recovery volume,
    /// or `system`, System Recovery in its RecoveryOSContainer.
    #[arg(long, value_enum, requires = "disk", conflicts_with_all = ["boot_hash", "upgrade"])]
    disk_recovery: Option<DiskRecovery>,
    /// A new directory for the bundle.
    #[arg(long)]
    output: PathBuf,
    /// Boot the restore ramdisk with the identity's restore objects instead
    /// of the installed system.
    #[arg(long)]
    restore: bool,
    /// With `--restore`: the model's update restore identity (its
    /// `RestoreBehavior` is `Update`), whose ramdisk runs restored_update
    /// over the installed system and keeps its data.
    #[arg(long, requires = "restore", conflicts_with = "variant")]
    update: bool,
    /// The restore identity by its variant, such as `Customer Upgrade
    /// Install (IPSW)`; by default the model's erase restore identity with
    /// `--restore`, and its installed system's otherwise.
    #[arg(long)]
    variant: Option<String>,
    /// The OS environment iBoot enters (`/chosen osenvironment` and its
    /// `/osenvironments` child): `darwinos-ramdisk` with `--restore`, none
    /// (macOS) otherwise.
    #[arg(long, value_parser = |name: &str| stela_loader::boot_policy::os_environment(name).map_err(|e| e.to_string()))]
    osenvironment: Option<OsEnvironment>,
    /// Guest DRAM base; the target SoC's by default (0x800000000 on the M1
    /// and M2, 0x10000000000 on every other SoC).
    #[arg(long, value_parser = parse_address)]
    dram_base: Option<u64>,
    /// Guest DRAM size assumed until the mount patches it at boot.
    #[arg(long, value_parser = parse_address, default_value = "0x200000000")]
    dram_size: u64,
    /// Lowest address handed to the kernel; the DRAM base by default, and
    /// always past iBoot's carveouts.
    #[arg(long, value_parser = parse_address)]
    phys_base: Option<u64>,
    /// Distance of the first region above the physical base.
    #[arg(long, value_parser = parse_address, default_value = "0")]
    load_offset: u64,
    /// The kernel command line, `-v` by default. `--restore` adds the
    /// arguments its boot mode needs.
    #[arg(long)]
    boot_args: Option<String>,
    /// The boot policy expressed to SPTM, TXM and the kernel.
    #[arg(long, value_enum, default_value = "reduced")]
    security: Security,
    /// System Integrity Protection bits (`lp-sip0`).
    #[arg(long, value_parser = parse_address, default_value = "0")]
    sip_bits: u64,
    /// Override the template's `/chosen sepfw-load-at-boot`.
    #[arg(long)]
    sepfw_load_at_boot: Option<bool>,
    /// Also place the identity's Cryptex1 trust caches in the boot
    /// `TrustCache` region after the static one.
    #[arg(long)]
    cryptex_trust_caches: bool,
    /// JSON list of DeviceTree values from device models, such as
    /// `[{"path": "/chosen/lock-regs/amcc", "property": "aperture-count", "u32": 1}]`.
    #[arg(long)]
    dt_overlay: Option<PathBuf>,
    /// Explicit `/cpus` child of the boot CPU; defaults to the first
    /// performance CPU in the original DeviceTree order.
    #[arg(long)]
    boot_cpu: Option<String>,
    /// The generic timer rate the machine presents, in Hz.
    #[arg(long, value_parser = parse_address, default_value = "24000000")]
    timebase_frequency: u64,
    /// A board configuration value for DeviceTree fill-ins, as KEY=HEX
    /// (for example SrNm=...); repeat for each key.
    #[arg(long = "syscfg", value_parser = parse_keyed_bytes)]
    syscfg: Vec<(String, Vec<u8>)>,
    /// A MAC address for DeviceTree fill-ins, as NAME=HEX with optional
    /// colons (for example ethaddr=02:00:00:00:00:01); repeat for each.
    #[arg(long = "macaddr", value_parser = parse_keyed_bytes)]
    mac_addresses: Vec<(String, Vec<u8>)>,
    /// The machine's ECID, as hex: a nonzero per-machine value the install
    /// tooling generates and keeps (`/chosen unique-chip-id`, `die-id`,
    /// `unique-device-id`). Without it those rows stay unresolved.
    #[arg(long, value_parser = parse_ecid)]
    ecid: Option<u64>,
    /// A firmware image the board's own devices run, kept under its guest
    /// file name (such as `USB-C_HPM,47.bin`); repeat for each image. Each
    /// takes the place of the same-named image the DeviceTree's board names;
    /// the others are taken from the IPSW's restore ramdisk.
    #[arg(long, conflicts_with = "disk")]
    board_firmware: Vec<PathBuf>,
    /// A kext to link into an auxiliary kernel collection against the boot
    /// collection: the directory `cargo xtask guest` builds, holding its
    /// `Info.plist` and arm64e objects (such as `target/guest/StelaGPU.kext`),
    /// which the installed system keeps in `/Library/Extensions`; repeat for
    /// each kext.
    #[arg(long = "kext", conflicts_with_all = ["upgrade", "disk_recovery"])]
    kexts: Vec<PathBuf>,
}

/// The recoveryOS copy `--disk-recovery` boots.
#[derive(Clone, Copy, clap::ValueEnum)]
pub(crate) enum DiskRecovery {
    Paired,
    System,
}

fn parse_ecid(value: &str) -> Result<u64, String> {
    let digits = value.strip_prefix("0x").unwrap_or(value);
    match u64::from_str_radix(digits, 16) {
        Ok(0) => Err("an ECID is nonzero".into()),
        Ok(ecid) => Ok(ecid),
        Err(error) => Err(error.to_string()),
    }
}

fn parse_keyed_bytes(value: &str) -> Result<(String, Vec<u8>), String> {
    let (key, bytes) = value.split_once('=').ok_or("expected KEY=HEX")?;
    if key.is_empty() {
        return Err("empty key".into());
    }
    let bytes = hex::decode(bytes.replace(':', "")).map_err(|error| error.to_string())?;
    Ok((key.to_owned(), bytes))
}

/// Run `boot prepare` with these arguments, exactly as the command line
/// parses them, so callers get the same defaults.
pub(crate) fn prepare(arguments: &[String]) -> Result<()> {
    #[derive(clap::Parser)]
    struct Line {
        #[command(flatten)]
        prepare: Prepare,
    }
    let mut line = <Line as clap::Parser>::try_parse_from(
        std::iter::once("prepare".to_owned()).chain(arguments.iter().cloned()),
    )?;
    line.prepare.quiet = true;
    line.prepare.run()
}

impl BootCommand {
    pub fn run(self) -> Result<()> {
        match self {
            Self::Prepare(command) => (*command).run(),
            Self::Check(command) => command.run(),
        }
    }
}

struct Decoded {
    bytes: Vec<u8>,
    source: Source,
    /// The BuildManifest digest of the component, as a hex string.
    declared_digest: Option<String>,
}

#[derive(Clone, Copy, clap::ValueEnum)]
enum Security {
    Reduced,
    Permissive,
}

/// One `--dt-overlay` entry; exactly one value field is present.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct OverlayEntry {
    path: String,
    property: String,
    u32: Option<u32>,
    u64: Option<u64>,
    hex: Option<String>,
    string: Option<String>,
}

fn read_overlay(path: &std::path::Path) -> Result<Vec<OverlayProperty>> {
    let entries: Vec<OverlayEntry> = stela_engine::read_json(path)?;
    entries
        .into_iter()
        .map(|entry| {
            let mut values = [
                entry.u32.map(|v| v.to_le_bytes().to_vec()),
                entry.u64.map(|v| v.to_le_bytes().to_vec()),
                entry.hex.map(hex::decode).transpose()?,
                entry.string.map(|v| [v.as_bytes(), &[0]].concat()),
            ]
            .into_iter()
            .flatten();
            let value = values.next().with_context(|| {
                format!("overlay {} {} has no value", entry.path, entry.property)
            })?;
            ensure!(
                values.next().is_none(),
                "overlay {} {} has several values",
                entry.path,
                entry.property
            );
            Ok(OverlayProperty {
                path: entry.path,
                property: entry.property,
                value,
            })
        })
        .collect()
}

/// The identity's global manifests sit at `Firmware/Manifests/restore/[cryptex1/]VARIANT/apticket.CLASS.im4m`,
/// with `Info.MacOSVariant` as VARIANT when the identity has one, as the
/// restore host's RootTicket answer chooses it: restored's `update_ticket`
/// compares that ticket's SHA-384 with `/chosen/boot-manifest-hash`.
fn manifest_member(cryptex1: bool, variant: &str, device_class: &str) -> String {
    let scope = if cryptex1 { "cryptex1/" } else { "" };
    format!("Firmware/Manifests/restore/{scope}{variant}/apticket.{device_class}.im4m")
}

/// The boot objects the manifest does not admit, by component. A manifest
/// admits an object when one of its entries' `DGST` is the object's
/// BuildManifest digest. When `admits_all`, an object it does not admit
/// fails, naming the entry of the object's IM4P type when there is one.
fn check_digests(
    manifest: &im4m::Manifest,
    decoded: &[&Decoded],
    admits_all: bool,
) -> Result<Vec<String>> {
    let digest = |object: &im4m::Object| {
        object
            .properties
            .iter()
            .find(|p| p.name == "DGST")
            .map(|p| p.value.clone())
    };
    let mut outside = vec![];
    for item in decoded {
        let Some(declared) = &item.declared_digest else {
            continue;
        };
        let declared = im4m::Value::Data(hex::decode(declared)?);
        if manifest
            .objects
            .iter()
            .any(|object| digest(object).as_ref() == Some(&declared))
        {
            continue;
        }
        ensure!(
            !admits_all,
            "{}",
            match manifest.object(&item.source.payload_type) {
                Some(object) => format!(
                    "manifest {} digest differs from the BuildManifest {}",
                    object.tag, item.source.component
                ),
                None => format!("the manifest does not admit {}", item.source.component),
            }
        );
        outside.push(item.source.component.clone());
    }
    Ok(outside)
}

fn parse_id(value: Option<&str>) -> Result<Option<u32>> {
    value
        .map(|v| {
            u32::from_str_radix(v.trim_start_matches("0x").trim_start_matches("0X"), 16)
                .with_context(|| format!("bad restore identity value {v}"))
        })
        .transpose()
}

/// `console` followed by each of the boot mode's arguments whose name it does
/// not already give, so an explicit `name=value` keeps its value.
pub(super) fn boot_args(console: &str, mode: &[&str]) -> String {
    let name = |argument: &str| argument.split('=').next().unwrap_or(argument).to_owned();
    let given: Vec<String> = console.split_whitespace().map(name).collect();
    let mut line = console.trim().to_owned();
    for argument in mode.iter().filter(|a| !given.contains(&name(a))) {
        if !line.is_empty() {
            line.push(' ');
        }
        line.push_str(argument);
    }
    line
}

/// A `--kext` directory's `Info.plist`, objects, name and trusted code.
struct KextBundle {
    info: Vec<u8>,
    objects: Vec<Vec<u8>>,
    name: String,
    trusted_code: Vec<CodeHash>,
}

/// `bundles` as the linker takes them.
fn kexts(bundles: &[KextBundle]) -> Vec<stela_loader::aux_kc::Kext<'_>> {
    bundles
        .iter()
        .map(|bundle| stela_loader::aux_kc::Kext {
            info: &bundle.info,
            objects: bundle.objects.iter().map(Vec::as_slice).collect(),
            bundle_path: format!("/Library/Extensions/{}", bundle.name),
            trusted_code: bundle.trusted_code.clone(),
        })
        .collect()
}

impl Prepare {
    /// The `--kext` bundles' contents.
    fn kext_bundles(&self) -> Result<Vec<KextBundle>> {
        let mut bundles = Vec::with_capacity(self.kexts.len());
        for directory in &self.kexts {
            let read = |path: &std::path::Path| {
                std::fs::read(path).with_context(|| format!("cannot read {}", path.display()))
            };
            let name = directory
                .file_name()
                .and_then(|name| name.to_str())
                .with_context(|| format!("{} has no UTF-8 name", directory.display()))?;
            let mut objects = Vec::new();
            for entry in std::fs::read_dir(directory)
                .with_context(|| format!("cannot list {}", directory.display()))?
            {
                let path = entry?.path();
                if path.extension().is_some_and(|extension| extension == "o") {
                    objects.push(path);
                }
            }
            objects.sort();
            let objects = objects
                .iter()
                .map(|path| read(path))
                .collect::<Result<Vec<_>>>()?;
            let trusted = directory.join(TRUSTED_CODE_FILE);
            let trusted_code = match std::fs::read(&trusted) {
                Ok(bytes) => stela_loader::trust_cache::code_hashes(&bytes)
                    .with_context(|| format!("{}: invalid trusted code", trusted.display()))?,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => Vec::new(),
                Err(error) => {
                    return Err(error)
                        .with_context(|| format!("cannot read {}", trusted.display()));
                }
            };
            bundles.push(KextBundle {
                info: read(&directory.join("Info.plist"))?,
                objects,
                name: name.to_owned(),
                trusted_code,
            });
        }
        Ok(bundles)
    }

    /// The trust cache module of the code the `--kext` bundles install.
    fn kext_code(kexts: &[stela_loader::aux_kc::Kext<'_>]) -> Result<Option<Decoded>> {
        Ok(Linked::trust_cache(kexts)?.map(|(source, bytes)| Decoded {
            source,
            bytes,
            declared_digest: None,
        }))
    }

    /// The `--kext` bundles linked against `kernel` (`stela_loader::aux_kc`).
    fn auxiliary_collection(
        kernel: &[u8],
        kexts: &[stela_loader::aux_kc::Kext<'_>],
    ) -> Result<Option<(Decoded, KextGrant)>> {
        if kexts.is_empty() {
            return Ok(None);
        }
        let linked = stela_loader::aux_kc::link(kernel, kexts)?;
        let source = linked.source(kexts);
        let grant = linked.grant();
        let bytes = linked.collection;
        let declared_digest = None;
        Ok(Some((
            Decoded {
                source,
                bytes,
                declared_digest,
            },
            grant,
        )))
    }

    fn run(self) -> Result<()> {
        ensure!(
            !self.output.exists(),
            "output exists; choose a new --output directory"
        );
        if let Some(disk) = &self.disk {
            return self.run_disk(disk);
        }
        let ipsw = self.ipsw.clone().context("--ipsw is required")?;
        let model = self.model.clone().context("--model is required")?;
        // The profile's identities and components, as setup chooses them.
        let profile = components::Profile::of(&model)?;
        let report = restore::archive::manifest(&ipsw, None, false)?;
        let (roles, purpose) = match self.restore {
            true if self.update => (RESTORE, Purpose::Update),
            true => (RESTORE, Purpose::Erase),
            false => (INSTALLED, Purpose::Installed),
        };
        let identity = match self.variant.as_deref() {
            Some(variant) => components::variant(&report, &profile, variant)?,
            None => components::identity(&report, &profile, purpose)?,
        };
        let owned = components::identities(&report, &profile);
        let component = |name: &str, payload_type: Option<&str>, limit: u64| -> Result<Decoded> {
            // A component absent from the selected identity is taken from
            // another identity of the same profile, and named with its
            // variant.
            let (owner, entry) = components::component(&report, &profile, identity, name)?;
            let container = restore::archive::read_member(&ipsw, &entry.path, limit)?;
            let label = if owner.variant == identity.variant {
                name.to_owned()
            } else {
                format!("{name} ({})", owner.variant)
            };
            let (bytes, container_sha256, payload_type, description, decoded_sha256) =
                match payload_type {
                    Some(expected) => {
                        let (bytes, receipt) = stela_loader::im4p::decode_owned(container, limit)?;
                        if receipt.payload_type != expected {
                            bail!(
                                "{name} is IM4P type {}, expected {expected}",
                                receipt.payload_type
                            );
                        }
                        (
                            bytes,
                            receipt.container_sha256,
                            receipt.payload_type,
                            receipt.description,
                            receipt.decoded_sha256,
                        )
                    }
                    None => {
                        let digest = stela_formats::sha256(&container);
                        (
                            container,
                            digest.clone(),
                            "opaque".into(),
                            String::new(),
                            digest,
                        )
                    }
                };
            let source = Source {
                component: label,
                member: entry.path.clone(),
                container_sha256,
                payload_type,
                description,
                decoded_bytes: bytes.len() as u64,
                decoded_sha256,
            };
            Ok(Decoded {
                bytes,
                source,
                declared_digest: entry.declared_digest_hex.clone(),
            })
        };
        let kernel = component(roles.kernel, Some("krnl"), KERNEL_LIMIT)?;
        let sptm = component(roles.sptm, Some("sptm"), COMPONENT_LIMIT)?;
        let txm = component(roles.txm, Some("trxm"), COMPONENT_LIMIT)?;
        let device_tree = component(roles.device_tree, Some("dtre"), COMPONENT_LIMIT)?;
        let (trust_cache_name, trust_cache_type) = roles.trust_cache;
        let trust_cache = component(trust_cache_name, Some(trust_cache_type), COMPONENT_LIMIT)?;
        let ramdisk = roles
            .ramdisk
            .map(|(name, payload_type)| component(name, Some(payload_type), RAMDISK_LIMIT))
            .transpose()?;
        // The root hash stays an IM4P: iBoot forwards it with its manifest.
        let root_hash = match roles.root_hash {
            None => None,
            Some((name, volume)) => Some((
                volume,
                owned
                    .iter()
                    .any(|i| i.components.contains_key(name))
                    .then(|| component(name, None, COMPONENT_LIMIT))
                    .transpose()?,
            )),
        };
        let sep = owned
            .iter()
            .any(|i| i.components.contains_key(roles.sep))
            .then(|| component(roles.sep, None, COMPONENT_LIMIT))
            .transpose()?;
        let global_manifest = |cryptex1: bool| -> Result<Option<Decoded>> {
            let variant = identity.macos_variant.as_ref().unwrap_or(&identity.variant);
            let member = manifest_member(cryptex1, variant, &identity.device_class);
            if !restore::archive::has_member(&ipsw, &member)? {
                return Ok(None);
            }
            let bytes = restore::archive::read_member(&ipsw, &member, COMPONENT_LIMIT)?;
            let digest = stela_formats::sha256(&bytes);
            let source = Source {
                component: if cryptex1 {
                    "Cryptex1 manifest"
                } else {
                    "Boot manifest"
                }
                .into(),
                member,
                container_sha256: digest.clone(),
                payload_type: "im4m".into(),
                description: String::new(),
                decoded_bytes: bytes.len() as u64,
                decoded_sha256: digest,
            };
            Ok(Some(Decoded {
                bytes,
                source,
                declared_digest: None,
            }))
        };
        // A restore for a named machine boots under the AP root ticket
        // Apple's signing service (TSS) personalizes for its ECID; the
        // install answers restored's RootTicket with the same ticket.
        let mut staged = None;
        let boot_manifest = match (self.restore, self.ecid) {
            (true, Some(ecid)) => {
                let manifest = restore::archive::read_member(
                    &ipsw,
                    "BuildManifest.plist",
                    BUILD_MANIFEST_LIMIT,
                )?;
                eprintln!(
                    "Sending ECID {ecid:#x}, fresh nonces and the component hashes to Apple's \
                     signing service (gs.apple.com) for the restore's AP root ticket"
                );
                let bytes = ticket::personalized(&manifest, identity.index, ecid)?;
                // An erase install's other signing requests bind only values
                // known now: their answers go on the stick, so the install
                // needs no network.
                if !self.update {
                    eprintln!(
                        "Sending the same ECID and the install's recoveryOS, Cryptex1 and \
                         LocalPolicy requests to Apple's signing service now, so the install \
                         does not ask for them"
                    );
                    staged = ticket::staged(&manifest, identity.index, ecid, &bytes)
                        .inspect_err(|error| {
                            eprintln!(
                                "note: {error:#}; installing macOS will ask Apple's signing service"
                            )
                        })
                        .ok();
                }
                let digest = stela_formats::sha256(&bytes);
                Some(Decoded {
                    source: Source {
                        component: "Boot manifest".into(),
                        member: "gs.apple.com personalized AP ticket".into(),
                        container_sha256: digest.clone(),
                        payload_type: "im4m".into(),
                        description: String::new(),
                        decoded_bytes: bytes.len() as u64,
                        decoded_sha256: digest,
                    },
                    bytes,
                    declared_digest: None,
                })
            }
            _ => global_manifest(false)?,
        };
        let cryptex1_manifest = global_manifest(true)?;
        let mut cryptex_trust_caches = vec![];
        if self.cryptex_trust_caches {
            for name in identity.components.keys() {
                if name.starts_with("Cryptex1,") && name.ends_with("TrustCache") {
                    cryptex_trust_caches.push(component(name, None, COMPONENT_LIMIT).and_then(
                        |raw| {
                            let (bytes, receipt) =
                                stela_loader::im4p::decode_owned(raw.bytes, COMPONENT_LIMIT)?;
                            Ok(Decoded {
                                source: Source {
                                    payload_type: receipt.payload_type,
                                    description: receipt.description,
                                    decoded_bytes: bytes.len() as u64,
                                    decoded_sha256: receipt.decoded_sha256,
                                    ..raw.source
                                },
                                bytes,
                                declared_digest: raw.declared_digest,
                            })
                        },
                    )?);
                }
            }
        }
        let mut outside_manifest = vec![];
        if let Some(manifest) = &boot_manifest {
            outside_manifest = check_digests(
                &im4m::parse(&manifest.bytes)?,
                &[&kernel, &sptm, &txm, &device_tree, &trust_cache],
                roles.manifest_admits_all,
            )?;
        }
        if let Some(manifest) = &cryptex1_manifest {
            let decoded: Vec<_> = cryptex_trust_caches.iter().collect();
            check_digests(&im4m::parse(&manifest.bytes)?, &decoded, true)?;
        }
        fn input(decoded: &Decoded) -> Input<'_> {
            Input {
                bytes: &decoded.bytes,
                source: decoded.source.clone(),
            }
        }
        let board_firmware = self
            .board_firmware
            .iter()
            .map(|path| {
                let bytes = std::fs::read(path)
                    .with_context(|| format!("cannot read {}", path.display()))?;
                let name = path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .with_context(|| format!("{} has no UTF-8 file name", path.display()))?
                    .to_owned();
                let digest = stela_formats::sha256(&bytes);
                let source = Source {
                    component: "BoardFirmware".into(),
                    member: path.canonicalize()?.display().to_string(),
                    container_sha256: digest.clone(),
                    payload_type: "raw".into(),
                    description: format!("board device firmware {name}"),
                    decoded_bytes: bytes.len() as u64,
                    decoded_sha256: digest,
                };
                Ok((
                    name,
                    Decoded {
                        bytes,
                        source,
                        declared_digest: None,
                    },
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let bundles = self.kext_bundles()?;
        let extensions = kexts(&bundles);
        let (auxiliary, kexts) = Self::auxiliary_collection(&kernel.bytes, &extensions)?.unzip();
        let kext_code = Self::kext_code(&extensions)?;
        let inputs = Inputs {
            target: Target {
                product_type: model.clone(),
                device_class: identity.device_class.clone(),
                product_version: report.product_version.clone(),
                product_build: report.product_build.clone(),
                variant: identity.variant.clone(),
                chip_id: parse_id(identity.ap_chip_id.as_deref())?,
                board_id: parse_id(identity.ap_board_id.as_deref())?,
            },
            kernel_collection: input(&kernel).into(),
            auxiliary_collection: auxiliary.as_ref().map(input),
            sptm: input(&sptm),
            txm: input(&txm),
            device_tree: input(&device_tree),
            static_trust_cache: input(&trust_cache),
            sep_firmware: sep.as_ref().map(input),
            boot_manifest: boot_manifest.as_ref().map(input),
            cryptex1_manifest: cryptex1_manifest.as_ref().map(input),
            extra_trust_caches: cryptex_trust_caches
                .iter()
                .chain(&kext_code)
                .map(input)
                .collect(),
            ramdisk: None,
            root_hash: root_hash.as_ref().map(|(volume, decoded)| RootHash {
                volume: *volume,
                im4p: decoded.as_ref().map(input),
                snapshot: None,
            }),
            pivot_root_hash: None,
            firmware_version: None,
            board_firmware: board_firmware
                .iter()
                .map(|(name, decoded)| (name.clone(), input(decoded)))
                .collect(),
        };
        let mut options = self.options(inputs.dram_base()?)?;
        options.security.kexts = kexts;
        let ramdisk_source = ramdisk.as_ref().map(|ramdisk| ramdisk.source.clone());
        let mut bundle = match ramdisk {
            Some(Decoded { bytes, source, .. }) => {
                boot_bundle::prepare_owned(&inputs, OwnedRamdisk { bytes, source }, &options)?
            }
            None => boot_bundle::prepare(&inputs, &options)?,
        };
        if let Some(bytes) = staged {
            let digest = stela_formats::sha256(&bytes);
            let source = Source {
                component: "Signing answers".into(),
                member: "gs.apple.com staged install answers".into(),
                container_sha256: digest.clone(),
                payload_type: "plist".into(),
                description: "Apple's answers to the install's recoveryOS root ticket, \
                              Cryptex1 and recoveryOS LocalPolicy requests"
                    .into(),
                decoded_bytes: bytes.len() as u64,
                decoded_sha256: digest,
            };
            bundle.append_file(
                stela_runtime::restore::tss::staged::FILE.into(),
                bytes,
                source,
            )?;
        }
        let mut unresolved = vec![];
        // The board the bundle's DeviceTree describes names its images, each
        // die the one its own node or its flash's names; the same IPSW's
        // restore ramdisk holds those not given by hand.
        match board_firmware::needed(&bundle) {
            Ok(names) => {
                let names: Vec<String> = names
                    .into_iter()
                    .filter(|name| !board_firmware.iter().any(|(given, _)| given == name))
                    .collect();
                if !names.is_empty() {
                    let separate_ramdisk;
                    let (bytes, source) = match &ramdisk_source {
                        Some(source) if source.component == components::RESTORE_RAMDISK => {
                            let (_, bytes) = bundle
                                .files
                                .iter()
                                .find(|(name, _)| name == RAMDISK_FILE)
                                .context("the bundle has no restore ramdisk")?;
                            (bytes.as_slice(), source)
                        }
                        _ => {
                            separate_ramdisk = component(
                                components::RESTORE_RAMDISK,
                                Some("rdsk"),
                                RAMDISK_LIMIT,
                            )?;
                            (separate_ramdisk.bytes.as_slice(), &separate_ramdisk.source)
                        }
                    };
                    let sourced = board_firmware::from_ramdisk(
                        bytes,
                        source,
                        board_firmware::DIRECTORY,
                        &names,
                    )?;
                    for (name, Decoded { bytes, source, .. }) in sourced {
                        bundle.append_board_firmware(name, bytes, source)?;
                    }
                }
            }
            Err(error) if board_firmware.is_empty() => unresolved.push(format!(
                "board firmware: the DeviceTree is not a board the engine models ({error})"
            )),
            Err(_) => {}
        }
        bundle.manifest.unresolved.extend(unresolved);
        bundle
            .manifest
            .unresolved
            .extend(outside_manifest.iter().map(|component| {
                format!("{component}: outside the boot manifest, which is the installed system's")
            }));
        let provenance = serde_json::json!({
            "archive": {
                "path": ipsw.canonicalize()?.display().to_string(),
                "build_manifest_sha256": report.manifest_sha256,
            }
        });
        self.write(&bundle, provenance)?;
        Ok(())
    }

    /// The bundle options from the command line, with the boot-args, OS
    /// environment and root of the selected boot mode.
    fn options(&self, board_dram_base: Option<u64>) -> Result<Options> {
        let dram_base = self
            .dram_base
            .or(board_dram_base)
            .unwrap_or(Options::default().dram_base);
        Ok(Options {
            dram_base,
            dram_size: self.dram_size,
            phys_base: self.phys_base.unwrap_or(dram_base),
            load_offset: self.load_offset,
            boot_args: boot_args(
                self.boot_args.as_deref().unwrap_or(DEFAULT_BOOT_ARGS),
                if self.restore { RESTORE_MODE_ARGS } else { &[] },
            ),
            osenvironment: self
                .osenvironment
                .or(self.restore.then_some(RESTORE_ENVIRONMENT)),
            // --restore replaces the host-received iBEC path, for both
            // erase and update IPSW identities. Disk updates are assigned
            // Upgrade by stela_loader::upgrade instead.
            boot_command: Some(match self.restore {
                true => FirmwareBootCommand::TetheredRestore,
                false => handoff::installed_command(
                    None,
                    self.boot_args.as_deref().unwrap_or(DEFAULT_BOOT_ARGS),
                )?,
            }),
            // A restore ramdisk roots from itself as `md0`; an installed
            // system's root is named from its disk (`stela_loader::installed`).
            root: self.restore.then_some(Root::Ramdisk),
            ecid: self.ecid,
            boot_cpu: self.boot_cpu.clone(),
            timebase_frequency: self.timebase_frequency,
            syscfg: self.syscfg.iter().cloned().collect(),
            mac_addresses: self.mac_addresses.iter().cloned().collect(),
            security: SecurityPolicy {
                mode: match self.security {
                    Security::Reduced => SecurityMode::Reduced,
                    Security::Permissive => SecurityMode::Permissive,
                },
                sip_bits: self.sip_bits,
                sepfw_load_at_boot: self.sepfw_load_at_boot,
                kexts: None,
            },
            overlay: self
                .dt_overlay
                .as_deref()
                .map(read_overlay)
                .transpose()?
                .unwrap_or_default(),
        })
    }

    /// Write the bundle files and `bundle.json`, with `provenance` naming
    /// where the inputs came from.
    fn write(&self, bundle: &boot_bundle::Bundle, provenance: serde_json::Value) -> Result<()> {
        std::fs::create_dir_all(&self.output)?;
        for (name, bytes) in &bundle.files {
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(self.output.join(name))?
                .write_all(bytes)?;
        }
        let mut written = serde_json::to_value(&bundle.manifest)?;
        if let (Some(object), serde_json::Value::Object(extra)) =
            (written.as_object_mut(), provenance)
        {
            object.extend(extra);
        }
        stela_engine::write_json(&self.output.join(MANIFEST_FILE), &written)?;
        let manifest = &bundle.manifest;
        if !self.quiet
            && let Some(target) = &manifest.target
        {
            eprintln!(
                "{} {} {}: {} regions to {:#x}; entry {} at {:#x}, x0 {:#x}; {}",
                target.product_type,
                target.product_build,
                target.variant,
                manifest.regions.len(),
                manifest.top_of_kernel_data,
                manifest.entry.image,
                manifest.entry.physical_address,
                manifest.entry_x0,
                self.output.join(MANIFEST_FILE).display()
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ecid_requires_nonzero_hex() {
        assert_eq!(parse_ecid("0x700cde23bbf1c97d"), Ok(0x700c_de23_bbf1_c97d));
        assert_eq!(parse_ecid("ABCDEF"), Ok(0xab_cdef));
        assert!(parse_ecid("0").is_err());
        assert!(parse_ecid("0x").is_err());
        assert!(parse_ecid("12345678901234567").is_err());
    }

    #[test]
    fn manifest_admits_external_objects_by_digest() {
        let object = |tag: &str, digest: u8| im4m::Object {
            tag: tag.into(),
            properties: vec![im4m::Property {
                name: "DGST".into(),
                value: im4m::Value::Data(vec![digest; 48]),
            }],
        };
        let manifest = im4m::Manifest {
            version: 0,
            properties: vec![],
            objects: vec![object("krnl", 1), object("rdtr", 2)],
            sha384: String::new(),
            digest: [0; 48],
            signature_validated: false,
        };
        let decoded = |component: &str, payload_type: &str, digest: u8| Decoded {
            bytes: vec![],
            source: Source {
                component: component.into(),
                member: String::new(),
                container_sha256: String::new(),
                payload_type: payload_type.into(),
                description: String::new(),
                decoded_bytes: 0,
                decoded_sha256: String::new(),
            },
            declared_digest: Some(hex::encode([digest; 48])),
        };
        let kernel = decoded("KernelCache", "krnl", 1);
        let restore_kernel = decoded("RestoreKernelCache", "krnl", 3);
        let restore_tree = decoded("RestoreDeviceTree", "dtre", 2);
        assert!(
            check_digests(&manifest, &[&kernel], true)
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            check_digests(&manifest, &[&restore_kernel, &restore_tree], false).unwrap(),
            ["RestoreKernelCache"]
        );
        let error = check_digests(&manifest, &[&restore_kernel], true).unwrap_err();
        assert!(error.to_string().contains("manifest krnl digest differs"));
    }

    #[test]
    fn boot_modes_add_their_arguments_after_the_console_arguments() {
        assert_eq!(
            boot_args(DEFAULT_BOOT_ARGS, &[]),
            "-v serial=3 serial-device-name=uart0"
        );
        assert_eq!(
            boot_args(DEFAULT_BOOT_ARGS, RESTORE_MODE_ARGS),
            "-v serial=3 serial-device-name=uart0 rd=md0 nand-enable-reformat=0x1 -progress -restore"
        );
    }

    #[test]
    fn explicit_boot_argument_is_preserved() {
        assert_eq!(
            boot_args("serial=3 rd=md1 -restore", RESTORE_MODE_ARGS),
            "serial=3 rd=md1 -restore nand-enable-reformat=0x1 -progress"
        );
    }

    #[test]
    fn boot_cpu_omission_and_an_explicit_cpu0_remain_distinct() {
        use clap::Parser;
        #[derive(Parser)]
        struct Line {
            #[command(flatten)]
            prepare: Prepare,
        }
        let base = [
            "prepare", "--ipsw", "a.ipsw", "--model", "Mac14,3", "--output", "out",
        ];
        let automatic = Line::try_parse_from(base).unwrap().prepare;
        assert_eq!(automatic.boot_cpu, None);
        assert_eq!(automatic.options(None).unwrap().boot_cpu, None);
        let explicit = Line::try_parse_from(base.into_iter().chain(["--boot-cpu", "cpu0"]))
            .unwrap()
            .prepare;
        assert_eq!(explicit.boot_cpu.as_deref(), Some("cpu0"));
        assert_eq!(
            explicit.options(None).unwrap().boot_cpu.as_deref(),
            Some("cpu0")
        );
    }

    #[test]
    fn restore_roots_from_md0() {
        use clap::Parser;
        #[derive(Parser)]
        struct Line {
            #[command(flatten)]
            prepare: Prepare,
        }
        let parse = |mode: &str| {
            Line::try_parse_from([
                "prepare", "--ipsw", "a.ipsw", "--model", "Mac14,3", "--output", "out", mode,
            ])
            .map(|line| line.prepare)
        };
        assert!(parse("--recovery").is_err());
        let restore = parse("--restore").unwrap();
        assert_eq!(
            restore.options(None).unwrap().boot_command,
            Some(FirmwareBootCommand::TetheredRestore)
        );
        assert_eq!(restore.options(None).unwrap().root, Some(Root::Ramdisk));
        // `--update` picks the update restore identity, and only with
        // `--restore`.
        assert!(parse("--update").is_err());
        let update = Line::try_parse_from([
            "prepare",
            "--ipsw",
            "a.ipsw",
            "--model",
            "Mac14,3",
            "--output",
            "out",
            "--restore",
            "--update",
        ])
        .unwrap()
        .prepare;
        assert!(update.update);
        assert_eq!(
            update.options(None).unwrap().boot_command,
            Some(FirmwareBootCommand::TetheredRestore)
        );
        assert_eq!(update.options(None).unwrap().root, Some(Root::Ramdisk));
        // Overriding the environment and arguments does not claim
        // that an iBEC image was received from a restore host.
        let manual = Line::try_parse_from([
            "prepare",
            "--ipsw",
            "a.ipsw",
            "--model",
            "Mac14,3",
            "--output",
            "out",
            "--osenvironment",
            "darwinos-ramdisk",
            "--boot-args",
            "rd=md0 -restore",
        ])
        .unwrap()
        .prepare;
        assert_eq!(
            manual.options(None).unwrap().boot_command,
            Some(FirmwareBootCommand::Fsboot)
        );
    }
}
