//! The contract between `stela setup` and the `stela.efi` it installs.
//!
//! - **Interface marker.** `stela.efi` carries one [`MARKER_BYTES`]-byte
//!   record, [`MAGIC`] followed by the [`Interface`] it was built with: the
//!   boot-bundle schema it reads, the revision of the `stela.cfg` and
//!   [`CHOICES_FILE`] options setup writes, and the layout of the
//!   machine-state partition (the NOR image, then the SEP's area). Setup
//!   reads it from an image ([`find`]) and refuses an image whose interface
//!   differs from its own, or that has none, instead of installing a
//!   `stela.efi` that would misread what setup wrote.
//! - **Startup choices.** With several systems, or to reach Install macOS
//!   without a host computer, setup also writes [`CHOICES_FILE`]: a timeout
//!   line and one entry per line, each a name, a tab and a complete
//!   `stela.cfg` option line. The first entry is the default, the one
//!   `stela.cfg` itself holds. `stela.efi` lists the entries, waits up to the
//!   timeout for a digit ([`opening`]; 0 waits only when a key is already
//!   pressed) and boots the chosen line in place of `stela.cfg`.
use crate::Result;
use alloc::{format, string::String, vec::Vec};
use stela_contracts::refuse;

pub mod layout;

/// The marker's first bytes.
pub const MAGIC: [u8; 16] = *b"STELA-MOUNT-IF\x00\x01";
const FIELDS: usize = 4;
pub const MARKER_BYTES: usize = MAGIC.len() + 4 * FIELDS;

/// The revision of the option lines setup writes for `stela.efi`: bump it
/// when an option changes meaning or a required one is added. Revision 3:
/// `--nor` may name the `--storage` partition, whose tail then holds the
/// machine state. Revision 4: `--update-restore` restores the install
/// medium's Update identity, and the startup choices hold up to
/// [`MAX_CHOICES`] entries, chosen with `1`-`9` and then `a`-`z`. Revision
/// 5: `--external-disks` makes every host disk the engine leaves the guest's
/// external drive, and `--storage` may share the USB with the install medium.
/// Revision 6: [`CHOSEN_DISK`] stands for the partition Install macOS chose.
/// Revision 7: erase installs select `--development-restore` with `--tss`,
/// keeping the original DEV-board FDR failure policy without an FDR proxy.
/// Revision 8: `--recovery-bundle` is gone; a recoveryOS boot reads the
/// disk's own recoveryOS or is refused. Revision 9: the data partition's
/// bundles, install medium and translation store are under `stela/`, not
/// `EFI/stela/`. Revision 10: `--wifi SSID`, `--wifi-key PATH` and
/// `--wifi-country XX` join a Wi-Fi network for the mount's network.
/// Revision 11: `--ipv4 ADDRESS/PREFIX`, `--ipv4-gateway ADDRESS` and
/// `--dns SERVERS` set the mount network's addresses by hand. Revision 12:
/// `--usb4-approve UUID` admits a USB4 device's PCIe tunnel without an IOMMU.
/// Revision 13: `--tss` is gone; every `--install` signs with Apple's
/// signing service.
pub const OPTIONS_REVISION: u32 = 13;

/// The startup choices, beside `stela.cfg` on the ESP.
pub const CHOICES_FILE: &str = "\\EFI\\stela\\choices.cfg";
/// The USB's own boot log beside them: setup preallocates it, contiguous,
/// with `stela_runtime::pc::boot_log::LOG_BYTES`, and `stela.efi` keeps its
/// whole output there from its first line, before and after boot services
/// end, so every run leaves a log on the USB itself.
pub const BOOT_LOG_FILE: &str = "\\EFI\\stela\\boot.log";
/// Plain-text snapshots of the firmware phase on the USB's data partition.
pub const VOLUME_LOG_DIRECTORY: &str = "stela/logs";
pub const VOLUME_LOG_SLOTS: [&str; 4] = ["boot-0.log", "boot-1.log", "boot-2.log", "boot-3.log"];
pub const VOLUME_LOG_BYTES: usize = 256 << 10;
pub const VOLUME_LOG_PREFIX: &str = "STELA loader boot log #";
pub const VOLUME_LOG_EMPTY: &str = "STELA loader boot log #0 (unused)\n";

/// The boot sequence in a slot's first line. Zero names a prepared slot.
pub fn volume_log_sequence(header: &[u8]) -> Option<u64> {
    let line = header.split(|&byte| byte == b'\n').next()?;
    let sequence = core::str::from_utf8(line)
        .ok()?
        .strip_prefix(VOLUME_LOG_PREFIX)?
        .split_once(' ')?
        .0;
    sequence.parse().ok()
}

/// The word that stands, in a startup choice's options, for the partition
/// `stela.efi`'s Install macOS chose for the Mac's internal SSD
/// (`SERIAL:GUID`), which it keeps in [`DISK_FILE`].
pub const CHOSEN_DISK: &str = "@disk";
/// The chosen partition, beside `stela.cfg`.
pub const DISK_FILE: &str = "\\EFI\\stela\\disk.cfg";

/// `options` with each [`CHOSEN_DISK`] replaced by `disk`; `None` when they
/// name the chosen disk and none was chosen.
pub fn with_disk(options: &str, disk: Option<&str>) -> Option<String> {
    let mut words = Vec::new();
    for word in options.split_ascii_whitespace() {
        words.push(match word {
            CHOSEN_DISK => disk?,
            word => word,
        });
    }
    Some(words.join(" "))
}

/// How the startup menu opens for a boot nothing holds at the menu.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Opening {
    /// The first entry starts at once unless a key is held or pressed
    /// within a short glance, as a Mac shows its startup options only when
    /// asked.
    Glance,
    /// A countdown of this many seconds starts the first entry.
    Countdown(u32),
    /// The menu waits for a choice.
    Wait,
}

/// How the menu opens with the file's `timeout_seconds`, `first_mac` telling
/// whether its first entry starts the Mac (`Some`) and whether that Mac can
/// start now (`true`), and `ready_macs` Mac entries that can start now. Only
/// a real choice keeps the countdown: one ready Mac, the first entry, starts
/// at once, and Install macOS and recovery stay a key away. A timeout of 0
/// reaches the menu only when a key was already pressed, which asked for it.
pub fn opening(timeout_seconds: u32, first_mac: Option<bool>, ready_macs: usize) -> Opening {
    match first_mac {
        None => Opening::Wait,
        Some(_) if timeout_seconds == 0 => Opening::Wait,
        Some(true) if ready_macs == 1 => Opening::Glance,
        Some(_) => Opening::Countdown(timeout_seconds),
    }
}

/// The keys that choose entries, in order: one per entry.
const KEYS: &[u8; 35] = b"123456789abcdefghijklmnopqrstuvwxyz";
/// The most entries the keys choose among.
pub const MAX_CHOICES: usize = KEYS.len();
/// The longest name; names are printable ASCII.
pub const MAX_NAME: usize = 40;
/// The longest timeout `stela.efi` waits.
pub const MAX_TIMEOUT_SECONDS: u32 = 60;
const TIMEOUT: &str = "timeout";

/// What one `stela.efi` build reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Interface {
    /// `stela_formats::boot_bundle::SCHEMA_VERSION`.
    pub bundle_schema: u32,
    pub options_revision: u32,
    /// The NOR image at the machine-state partition's start.
    pub nor_bytes: u32,
    /// The SEP's area after it.
    pub sep_bytes: u32,
}

/// The interface of this source tree.
pub const CURRENT: Interface = Interface {
    bundle_schema: stela_formats::boot_bundle::SCHEMA_VERSION,
    options_revision: OPTIONS_REVISION,
    nor_bytes: stela_soc_data::nor::CAPACITY as u32,
    sep_bytes: stela_formats::machine_state::SEP_BYTES as u32,
};

impl Interface {
    fn fields(self) -> [u32; FIELDS] {
        [
            self.bundle_schema,
            self.options_revision,
            self.nor_bytes,
            self.sep_bytes,
        ]
    }
}

/// The marker record `stela.efi` embeds.
pub const fn marker(interface: Interface) -> [u8; MARKER_BYTES] {
    let mut bytes = [0u8; MARKER_BYTES];
    let mut at = 0;
    while at < MAGIC.len() {
        bytes[at] = MAGIC[at];
        at += 1;
    }
    let fields = [
        interface.bundle_schema,
        interface.options_revision,
        interface.nor_bytes,
        interface.sep_bytes,
    ];
    let mut field = 0;
    while field < FIELDS {
        let value = fields[field].to_le_bytes();
        let mut byte = 0;
        while byte < 4 {
            bytes[MAGIC.len() + 4 * field + byte] = value[byte];
            byte += 1;
        }
        field += 1;
    }
    bytes
}

/// The interface an image's marker records. `None` when it has no marker or
/// markers that disagree.
pub fn find(image: &[u8]) -> Option<Interface> {
    let mut found: Option<Interface> = None;
    let mut at = 0;
    while let Some(offset) = image[at..]
        .windows(MAGIC.len())
        .position(|window| window == MAGIC)
    {
        let start = at + offset + MAGIC.len();
        let fields = image.get(start..start + 4 * FIELDS)?;
        let word = |n: usize| u32::from_le_bytes(fields[4 * n..4 * n + 4].try_into().unwrap());
        let interface = Interface {
            bundle_schema: word(0),
            options_revision: word(1),
            nor_bytes: word(2),
            sep_bytes: word(3),
        };
        match found {
            Some(earlier) if earlier != interface => return None,
            _ => found = Some(interface),
        }
        at = start;
    }
    found
}

/// Why `found` cannot be installed by a setup built with `expected`, or
/// `None` when it can.
pub fn mismatch(expected: Interface, found: Option<Interface>) -> Option<String> {
    let Some(found) = found else {
        return Some(
            "it carries no Stela mount interface marker, so it predates this stela or is not \
             stela.efi"
                .into(),
        );
    };
    let names = [
        "boot-bundle schema",
        "stela.cfg option revision",
        "NOR image bytes",
        "SEP state bytes",
    ];
    let differences: Vec<String> = names
        .iter()
        .zip(expected.fields().into_iter().zip(found.fields()))
        .filter(|(_, (expected, found))| expected != found)
        .map(|(name, (expected, found))| format!("{name} {found}, not {expected}"))
        .collect();
    (!differences.is_empty()).then(|| differences.join(", "))
}

/// The user-facing restore option: `--install MANIFEST` names the install
/// medium's `payload.json`.
pub const INSTALL_OPTION: &str = "--install";
/// The install restores with the inventory's Update identity, over the
/// installed system and keeping its data, instead of erasing the target.
pub const UPDATE_RESTORE_OPTION: &str = "--update-restore";

/// One startup choice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Choice {
    pub name: String,
    pub options: String,
}

/// What a startup choice starts, read from its options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChoiceKind {
    /// The Mac, from the startup disk NVRAM names, as a Mac starts
    /// (`--system` with `--preboot`).
    Mac,
    /// The restore that lays out the Mac's SSD ([`INSTALL_OPTION`]).
    Setup,
    /// The restore that updates the installed system in place
    /// ([`UPDATE_RESTORE_OPTION`]).
    Update,
    /// Anything else, such as a host check.
    Other,
}

impl Choice {
    pub fn kind(&self) -> ChoiceKind {
        let mut words = self.options.split_ascii_whitespace();
        if words.next() != Some("--system") {
            return ChoiceKind::Other;
        }
        let has = |name: &str| words.clone().any(|word| word == name);
        match (
            has(INSTALL_OPTION),
            has(UPDATE_RESTORE_OPTION),
            has(crate::boot_selection::PREBOOT_OPTION),
        ) {
            (true, true, _) => ChoiceKind::Update,
            (true, false, _) => ChoiceKind::Setup,
            (false, _, true) => ChoiceKind::Mac,
            (false, _, false) => ChoiceKind::Other,
        }
    }
}

/// The startup choices: the default first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Choices {
    pub timeout_seconds: u32,
    pub entries: Vec<Choice>,
}

fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= MAX_NAME
        && name.bytes().all(|b| b.is_ascii_graphic() || b == b' ')
        && name.trim() == name
}

impl Choices {
    /// The file's text; refuses what [`Choices::parse`] would.
    pub fn encode(&self) -> Result<String> {
        self.check()?;
        let mut text = format!("{TIMEOUT} {}\n", self.timeout_seconds);
        for entry in &self.entries {
            if entry.options.contains(['\n', '\r', '\t']) || entry.options.trim().is_empty() {
                return Err(refuse!("a startup choice's options are not one line").into());
            }
            text.push_str(&format!("{}\t{}\n", entry.name, entry.options.trim()));
        }
        Ok(text)
    }

    fn check(&self) -> Result<()> {
        if self.entries.is_empty() || self.entries.len() > MAX_CHOICES {
            return Err(refuse!("startup choices hold one to 35 entries").into());
        }
        if self.timeout_seconds > MAX_TIMEOUT_SECONDS {
            return Err(refuse!("the startup choice timeout is longer than a minute").into());
        }
        for (n, entry) in self.entries.iter().enumerate() {
            if !valid_name(&entry.name) {
                return Err(refuse!("a startup choice name is not short printable ASCII").into());
            }
            if self.entries[..n].iter().any(|e| e.name == entry.name) {
                return Err(refuse!("two startup choices have the same name").into());
            }
        }
        Ok(())
    }

    /// Parse the file: `timeout N`, then `NAME<TAB>OPTIONS` lines; blank
    /// lines and lines starting with `#` are ignored.
    pub fn parse(text: &str) -> Result<Self> {
        let mut timeout = None;
        let mut entries = Vec::new();
        for line in text.lines() {
            let line = line.trim_end_matches('\r');
            if line.trim().is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some(value) = line.strip_prefix(TIMEOUT).and_then(|v| v.strip_prefix(' ')) {
                if timeout.is_some() {
                    return Err(refuse!("choices.cfg has two timeout lines").into());
                }
                timeout = Some(
                    value
                        .trim()
                        .parse()
                        .map_err(|_| refuse!("choices.cfg's timeout is not a number"))?,
                );
                continue;
            }
            let (name, options) = line.split_once('\t').ok_or(refuse!(
                "a choices.cfg entry is not NAME, a tab and options"
            ))?;
            if options.trim().is_empty() {
                return Err(refuse!("a choices.cfg entry has no options").into());
            }
            entries.push(Choice {
                name: name.into(),
                options: options.trim().into(),
            });
        }
        let choices = Self {
            timeout_seconds: timeout.ok_or(refuse!("choices.cfg has no timeout line"))?,
            entries,
        };
        choices.check()?;
        Ok(choices)
    }

    /// The key that chooses entry `index`: `1`-`9`, then `a`-`z`.
    pub fn key(index: usize) -> Option<char> {
        KEYS.get(index).map(|&key| char::from(key))
    }

    /// The entry a key chooses: its [`key`](Self::key), in either case, or
    /// Enter for the default.
    pub fn chosen(&self, key: char) -> Option<usize> {
        match key {
            '\r' | '\n' => Some(0),
            _ => {
                let key = u8::try_from(key.to_ascii_lowercase()).ok()?;
                let index = KEYS.iter().position(|&k| k == key)?;
                (index < self.entries.len()).then_some(index)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn the_chosen_disk_stands_in_for_every_word_that_names_it() {
        let options = "--system --storage @disk --nor @disk --guest-network";
        assert_eq!(
            with_disk(options, Some("S1:0C81")).as_deref(),
            Some("--system --storage S1:0C81 --nor S1:0C81 --guest-network")
        );
        assert_eq!(with_disk(options, None), None);
        assert_eq!(
            with_disk("--system --bundle x", None).as_deref(),
            Some("--system --bundle x")
        );
    }

    #[test]
    fn only_a_real_choice_counts_down() {
        // One installed Mac starts at once; the menu is a key away.
        assert_eq!(opening(5, Some(true), 1), Opening::Glance);
        // None ready, or several, is a choice to make.
        assert_eq!(opening(5, Some(false), 0), Opening::Countdown(5));
        assert_eq!(opening(5, Some(true), 2), Opening::Countdown(5));
        assert_eq!(opening(5, Some(false), 1), Opening::Countdown(5));
        // A first entry that is not the Mac, such as Install macOS while no
        // macOS is installed, waits for a choice.
        assert_eq!(opening(5, None, 0), Opening::Wait);
        assert_eq!(opening(5, None, 1), Opening::Wait);
        // A timeout of 0 opened the menu because a key asked for it.
        assert_eq!(opening(0, Some(true), 1), Opening::Wait);
        assert_eq!(opening(0, Some(true), 2), Opening::Wait);
    }

    #[test]
    fn the_marker_round_trips_and_disagreeing_markers_count_as_none() {
        let mut image = b"MZ....".to_vec();
        image.extend_from_slice(&marker(CURRENT));
        image.extend_from_slice(b"tail");
        assert_eq!(find(&image), Some(CURRENT));
        assert_eq!(mismatch(CURRENT, find(&image)), None);
        // The same record twice is still one interface.
        image.extend_from_slice(&marker(CURRENT));
        assert_eq!(find(&image), Some(CURRENT));
        let other = Interface {
            bundle_schema: CURRENT.bundle_schema + 1,
            ..CURRENT
        };
        let mut newer = b"MZ".to_vec();
        newer.extend_from_slice(&marker(other));
        let reason = mismatch(CURRENT, find(&newer)).unwrap();
        assert!(reason.contains("boot-bundle schema"), "{reason}");
        image.extend_from_slice(&marker(other));
        assert_eq!(find(&image), None);
        assert!(mismatch(CURRENT, find(b"MZ")).unwrap().contains("no Stela"));
        // A truncated marker is no marker.
        assert_eq!(find(&marker(CURRENT)[..MARKER_BYTES - 1]), None);
    }

    #[test]
    fn a_choice_is_classified_by_its_options() {
        let kind = |options: &str| {
            Choice {
                name: "x".into(),
                options: options.into(),
            }
            .kind()
        };
        let disk = "--system --storage A:B --nor A:B";
        assert_eq!(
            kind(&format!("{disk} --preboot A:B --ecid 1")),
            ChoiceKind::Mac
        );
        assert_eq!(kind(&format!("{disk} --bundle r")), ChoiceKind::Other);
        assert_eq!(
            kind(&format!("{disk} --bundle s --install p.json")),
            ChoiceKind::Setup
        );
        assert_eq!(
            kind(&format!(
                "{disk} --bundle s --install p.json --update-restore"
            )),
            ChoiceKind::Update
        );
        assert_eq!(kind("--host-check"), ChoiceKind::Other);
    }

    #[test]
    fn choices_round_trip_and_keys_choose_entries() {
        let choices = Choices {
            timeout_seconds: 5,
            entries: vec![
                Choice {
                    name: "macOS".into(),
                    options: "--system --preboot A:B".into(),
                },
                Choice {
                    name: "macOS - Install macOS".into(),
                    options: "--system --bundle \\EFI\\x --install p.json".into(),
                },
            ],
        };
        let text = choices.encode().unwrap();
        assert_eq!(Choices::parse(&text).unwrap(), choices);
        assert_eq!(
            Choices::parse(&format!("# comment\r\n\r\n{text}")).unwrap(),
            choices
        );
        assert_eq!(choices.chosen('\r'), Some(0));
        assert_eq!(choices.chosen('2'), Some(1));
        assert_eq!(choices.chosen('3'), None);
        assert_eq!(choices.chosen('x'), None);
        assert_eq!(choices.chosen('0'), None);
        // Past nine entries, letters choose.
        let many = Choices {
            timeout_seconds: 0,
            entries: (0..MAX_CHOICES)
                .map(|n| Choice {
                    name: format!("entry {n}"),
                    options: "--system".into(),
                })
                .collect(),
        };
        assert_eq!(Choices::parse(&many.encode().unwrap()).unwrap(), many);
        assert_eq!(Choices::key(9), Some('a'));
        assert_eq!(Choices::key(MAX_CHOICES - 1), Some('z'));
        assert_eq!(Choices::key(MAX_CHOICES), None);
        assert_eq!(many.chosen('a'), Some(9));
        assert_eq!(many.chosen('Z'), Some(MAX_CHOICES - 1));
        let mut too_many = many.clone();
        too_many.entries.push(Choice {
            name: "one more".into(),
            options: "--system".into(),
        });
        assert!(too_many.encode().is_err());
        for bad in [
            "macOS\t--system\n",
            "timeout 5\n",
            "timeout x\nmacOS\t--system\n",
            "timeout 61\nmacOS\t--system\n",
            "timeout 1\nmacOS --system\n",
            "timeout 1\nmacOS\t \n",
            "timeout 1\nA\t--system\nA\t--system\n",
            "timeout 1\ntimeout 2\nA\t--system\n",
        ] {
            assert!(Choices::parse(bad).is_err(), "{bad:?}");
        }
        let mut long = choices.clone();
        long.entries[0].name = "x".repeat(MAX_NAME + 1);
        assert!(long.encode().is_err());
        long.entries[0].name = "macOS".into();
        long.entries[0].options = "a\nb".into();
        assert!(long.encode().is_err());
    }
}
