//! Semantic syntax and configuration failures for the shared option grammar.
use super::{BootCommand, Guid, Target};
use core::fmt;

/// The partition argument whose producer rejected its syntax.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetArgument {
    Storage,
    Nor,
    Preboot,
    RestoreSource,
}

/// The GUID argument whose producer rejected its syntax.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuidArgument {
    InstallSource,
    HostNorFile,
}

/// An option syntax/value failure or an invalid complete configuration.
/// Upstream partition/GUID/origin causes remain explicit producer leaves until
/// their owning modules expose typed errors; owned grammar errors are variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    AudioMode,
    DumpRegister,
    Number,
    UrlScheme,
    UrlPath,
    UrlHost,
    Url(stela_runtime::net::http::UrlError),
    SystemPrefix,
    DuplicateNativeCores,
    NativeCoreCount,
    NativeCoreCapacity,
    DuplicateGuestPes,
    GuestPeCount,
    MissingRestoreSource,
    DuplicateRestoreSource,
    MissingRestoreManifest,
    DuplicateRestoreManifest,
    MissingInstallManifest,
    DuplicateInstallMedium,
    MissingInstallSource,
    DuplicateInstallSource,
    MissingRestoreIdentity,
    RestoreIdentityCapacity,
    DuplicateRestoreIdentity,
    MissingBundleDirectory,
    MissingUsbStorage,
    DuplicateUsbStorage,
    MissingWifiNetwork,
    DuplicateWifi,
    MissingWifiKey,
    DuplicateWifiKey,
    WifiCountry,
    DuplicateWifiCountry,
    Usb4Approve,
    DuplicateUsb4Approve,
    Ipv4,
    DuplicateIpv4,
    Ipv4Gateway,
    DuplicateIpv4Gateway,
    Dns,
    DuplicateDns,
    DuplicateAudioCheck,
    DuplicateAudioHpet,
    DuplicateNativePciIoapic,
    MissingAudioMode,
    MissingNetworkUrl,
    DuplicateNetworkCheck,
    NetworkCheckGetHttps,
    MissingNetworkCheckRange,
    NetworkCheckRangeExtent,
    NetworkCheckRangeEtag,
    NetworkCheckRangeLimit,
    NetworkCheckReopen,
    MissingEcid,
    Ecid,
    MissingStorage,
    AnsBacking,
    MissingRamStorage,
    MissingNor,
    DuplicateNor,
    MissingPreboot,
    MissingWatchRange,
    WatchRange,
    MissingBootVolume,
    BootVolume,
    DuplicateBootVolume,
    MissingBootCommand,
    BootCommandByte,
    DuplicateBootCommand,
    RecoverySelection,
    MissingDumpRange,
    DumpRange,
    MissingNumericValue,
    UnknownSystemOption,
    HostSeconds,
    HostPrefix,
    HostNorSpecification,
    UnknownHostOption,
    WifiNetwork,
    DuplicateNativeShared,
    DuplicateNativeBootstrap,
    Constraint(Constraint),
    Target {
        argument: TargetArgument,
        source: <Target as core::str::FromStr>::Err,
    },
    Guid {
        argument: GuidArgument,
        source: <Guid as core::str::FromStr>::Err,
    },
    BootCommandOrigin {
        value: u8,
        source: <BootCommand as TryFrom<u8>>::Error,
    },
}

/// A rule checked against the complete typed record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Constraint {
    UnresolvedDisk,
    RestoreConfiguration,
    UnexpectedRestoreInputs,
    WifiDependency,
    Ipv4Gateway,
    InstallSourceDependency,
    InstallConfiguration,
    UpdateDependency,
    DevelopmentRestore,
    VerifyDependency,
    PrebootSelection,
    UpgradeSelection,
    BundleRepairSelection,
    RecoverySafeMode,
    DramAlignment,
    AudioHpetDependency,
    NativePciIoapicExclusive,
}
impl Error {
    /// The established stela.cfg diagnostic at presentation-only callers.
    pub const fn message(self) -> &'static str {
        match self {
            Self::AudioMode => {
                "audio check mode must be playback, duplex, transport-playback or transport-duplex"
            }
            Self::DumpRegister => "a dump register is x0 to x30",
            Self::Number => "invalid number in system options",
            Self::UrlScheme => "URL scheme must be http or https",
            Self::UrlPath => "URL without a path",
            Self::UrlHost => "URL without a host",
            Self::Url(error) => error.message(),
            Self::SystemPrefix => "system options must start with --system",
            Self::DuplicateNativeCores => "--native-cores must occur once",
            Self::NativeCoreCount => "--native-cores needs a processor count",
            Self::NativeCoreCapacity => {
                "--native-cores is outside the supported processor capacity"
            }
            Self::DuplicateGuestPes => "--guest-pes must occur once",
            Self::GuestPeCount => "--guest-pes needs a processor count",
            Self::MissingRestoreSource => "missing restore source partition",
            Self::DuplicateRestoreSource => "duplicate restore source",
            Self::MissingRestoreManifest => "missing restore manifest path",
            Self::DuplicateRestoreManifest => "duplicate restore manifest",
            Self::MissingInstallManifest => "missing install medium manifest",
            Self::DuplicateInstallMedium => "duplicate install medium",
            Self::MissingInstallSource => "missing install medium partition",
            Self::DuplicateInstallSource => "duplicate install medium partition",
            Self::MissingRestoreIdentity => "missing restore identity index",
            Self::RestoreIdentityCapacity => "restore identity too large",
            Self::DuplicateRestoreIdentity => "duplicate restore identity",
            Self::MissingBundleDirectory => "missing bundle directory",
            Self::MissingUsbStorage => "missing USB storage disk serial",
            Self::DuplicateUsbStorage => "duplicate USB storage disk",
            Self::MissingWifiNetwork => "--wifi needs a network name",
            Self::DuplicateWifi => "duplicate --wifi",
            Self::MissingWifiKey => "--wifi-key needs an ESP path",
            Self::DuplicateWifiKey => "duplicate --wifi-key",
            Self::WifiCountry => "--wifi-country needs a two-letter country",
            Self::DuplicateWifiCountry => "duplicate --wifi-country",
            Self::Usb4Approve => "--usb4-approve needs a device UUID",
            Self::DuplicateUsb4Approve => "duplicate --usb4-approve",
            Self::Ipv4 => "--ipv4 needs a host ADDRESS/PREFIX, such as 192.168.1.20/24",
            Self::DuplicateIpv4 => "duplicate --ipv4",
            Self::Ipv4Gateway => "--ipv4-gateway needs an IPv4 address",
            Self::DuplicateIpv4Gateway => "duplicate --ipv4-gateway",
            Self::Dns => "--dns needs 1 to 3 distinct server addresses separated by commas",
            Self::DuplicateDns => "duplicate --dns",
            Self::DuplicateAudioCheck => "duplicate audio check",
            Self::DuplicateAudioHpet => "duplicate audio HPET handoff",
            Self::DuplicateNativePciIoapic => "duplicate native PCI IOAPIC handoff",
            Self::MissingAudioMode => "missing audio check mode",
            Self::MissingNetworkUrl => "missing network check URL",
            Self::DuplicateNetworkCheck => "duplicate network check",
            Self::NetworkCheckGetHttps => "--network-check-get requires an HTTPS URL",
            Self::MissingNetworkCheckRange => {
                "--network-check-range requires URL FIRST LAST TOTAL ETAG"
            }
            Self::NetworkCheckRangeExtent => "--network-check-range requires FIRST <= LAST < TOTAL",
            Self::NetworkCheckRangeEtag => {
                "--network-check-range requires a nonempty strong quoted ETag"
            }
            Self::NetworkCheckRangeLimit => {
                "at most two --network-check-range options are supported"
            }
            Self::NetworkCheckReopen => {
                "--network-check-reopen requires two ranges of the same pinned HTTP object"
            }
            Self::MissingEcid => "missing ECID",
            Self::Ecid => "--ecid takes one nonzero ECID",
            Self::MissingStorage => "missing ANS storage partition",
            Self::AnsBacking => "system mode accepts exactly one ANS backing",
            Self::MissingRamStorage => "missing RAM storage size",
            Self::MissingNor => "missing NOR partition",
            Self::DuplicateNor => "duplicate NOR partition",
            Self::MissingPreboot => "missing Preboot disk",
            Self::MissingWatchRange => "missing watch range",
            Self::WatchRange => "a watch is BASE,BYTES",
            Self::MissingBootVolume => "missing boot volume",
            Self::BootVolume => "--boot-volume is not TYPE:PARTITION:GROUP",
            Self::DuplicateBootVolume => "duplicate boot volume",
            Self::MissingBootCommand => "missing boot command",
            Self::BootCommandByte => "boot command is not a byte",
            Self::DuplicateBootCommand => "duplicate boot command",
            Self::RecoverySelection => {
                "--paired-recovery and --system-recovery name one recoveryOS"
            }
            Self::MissingDumpRange => "missing dump range",
            Self::DumpRange => "a dump is BASE,BYTES",
            Self::MissingNumericValue => "missing system option value",
            Self::UnknownSystemOption => "unknown system option",
            Self::HostSeconds => "host check: a time option takes a positive number of seconds",
            Self::HostPrefix => "host check: stela.cfg must start with --host-check",
            Self::HostNorSpecification => "host check: --nor-file takes GUID:/PATH",
            Self::UnknownHostOption => "host check: unknown option",
            Self::WifiNetwork => "--wifi needs a 1 to 32 byte network name, with %XX escapes",
            Self::DuplicateNativeShared => "duplicate native shared check",
            Self::DuplicateNativeBootstrap => "duplicate native bootstrap check",
            Self::Constraint(rule) => rule.message(),
            Self::Target { source, .. } | Self::Guid { source, .. } => source,
            Self::BootCommandOrigin { source, .. } => source,
        }
    }
}
impl Constraint {
    pub const fn message(self) -> &'static str {
        match self {
            Self::UnresolvedDisk => "@disk names a disk Install macOS has not chosen",
            Self::RestoreConfiguration => {
                "--restore-study requires --bundle, --storage, --restore-source, --restore-manifest and --restore-identity; refuses --preboot/--survey"
            }
            Self::UnexpectedRestoreInputs => "restore inputs require explicit --restore-study",
            Self::WifiDependency => "--wifi-key and --wifi-country configure a --wifi network",
            Self::Ipv4Gateway => "--ipv4-gateway needs --ipv4 and another address of its subnet",
            Self::InstallSourceDependency => {
                "--install-source names the partition of an --install medium"
            }
            Self::InstallConfiguration => {
                "--install requires --bundle, --storage and --nor; refuses --restore-study, --preboot and --survey"
            }
            Self::UpdateDependency => "--update-restore applies to --install",
            Self::DevelopmentRestore => {
                "--development-restore requires an erase --install and refuses --update-restore"
            }
            Self::VerifyDependency => "--verify-on-read applies to --install and --restore-study",
            Self::PrebootSelection => {
                "--boot-volume, --boot-command, --safe-boot, --maintenance-write, --upgrade, --paired-recovery and --system-recovery \
                 choose a --preboot boot"
            }
            Self::UpgradeSelection => "--upgrade is neither a safe-mode boot nor a recoveryOS",
            Self::BundleRepairSelection => {
                "--bundle-repair is a macOS boot, neither an update nor a recoveryOS"
            }
            Self::RecoverySafeMode => "a recoveryOS boot is not a safe-mode boot",
            Self::DramAlignment => "guest DRAM base must be 2 MiB aligned",
            Self::AudioHpetDependency => "--audio-own-hpet requires --audio-check",
            Self::NativePciIoapicExclusive => {
                "--native-pci-own-ioapic requires the normal system platform path"
            }
        }
    }
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.message())
    }
}
impl fmt::Display for Constraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.message())
    }
}
impl core::error::Error for Error {}
impl core::error::Error for Constraint {}
