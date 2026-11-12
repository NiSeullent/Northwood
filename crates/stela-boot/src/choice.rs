//! The startup menu (`stela_loader::mount_interface`): when the boot volume
//! has `choices.cfg`, `stela.efi` lists on the framebuffer, in Stela's look
//! ([`ui`]), the entries that can start now, then the PC firmware's own
//! boot entries (`firmware::entries`), with one highlighted. A macOS entry
//! is left out until a macOS is installed or while Install macOS has not
//! prepared its SSD or the disk it chose is gone (`chosen_disk::Stale`),
//! and Install macOS once it has (`selection::ready`). Up and Down move the
//! highlight, Enter starts the highlighted entry, and an entry's key (`1`-`9`,
//! then `a`-`z`) starts it at once; a firmware entry restarts the PC into it.
//! Diagnostics builds add a hardware report after the existing choices.
//! When the first entry, the one `stela.cfg` holds, is the only Mac that
//! can start now, it starts at once unless a key is held or pressed within
//! [`GLANCE`], as a Mac shows its startup options only when asked
//! (`mount_interface::opening`). When there is a choice to make, a
//! countdown from the file's timeout starts the first entry; any key stops
//! it. A timeout of 0 opens the menu only when a key is already pressed.
//! A file that cannot be read or parsed is reported and
//! `stela.cfg` boots, so a damaged file never stops the default boot. A
//! machine that can only be set up waits at the menu with no countdown and
//! its setup entry highlighted ([`hold`]).
use crate::{
    console::{self, Framebuffer, font},
    firmware::{
        entries::{self, Entry},
        esp,
    },
    output,
    selection::{self, Readiness},
    status,
    ui::{self, Canvas},
};
use alloc::{format, string::String, vec::Vec};
use core::time::Duration;
use stela_contracts::refuse;
use stela_loader::mount_interface::{
    CHOSEN_DISK, ChoiceKind, Choices, Opening,
    layout::{CHOICES, SELECTED_DISK},
    opening, with_disk,
};
use uefi::{
    boot::{self, EventType, TimerTrigger, Tpl},
    fs::FileSystem,
    proto::console::text::{Key, ScanCode},
};

const TICK: Duration = Duration::from_secs(1);
/// How long a key may still come to open the menu when the first entry
/// starts without one: longer than a held key's repeat and a USB keyboard's
/// polling, short enough not to be a wait.
const GLANCE: Duration = Duration::from_millis(250);

pub(crate) enum Press {
    Up,
    Down,
    Left,
    Right,
    Home,
    End,
    Delete,
    Escape,
    Enter,
    Character(char),
    Other,
}

/// What a wait for a key ended with.
pub(crate) enum Wait {
    Key(Press),
    /// The key event fired without a key.
    Nothing,
    Timeout,
    /// The console cannot be waited on.
    NoInput,
}

/// The next key, waiting up to `timeout` (forever with `None`).
pub(crate) fn key(timeout: Option<Duration>) -> Wait {
    let Ok(keys) = uefi::system::with_stdin(|input| input.wait_for_key_event()) else {
        return Wait::NoInput;
    };
    let timer = match timeout {
        Some(duration) => {
            // SAFETY: a plain timer event without a notification function.
            let Ok(timer) =
                (unsafe { boot::create_event(EventType::TIMER, Tpl::APPLICATION, None, None) })
            else {
                return Wait::NoInput;
            };
            if boot::set_timer(&timer, TimerTrigger::Relative(duration)).is_err() {
                let _ = boot::close_event(timer);
                return Wait::NoInput;
            }
            Some(timer)
        }
        None => None,
    };
    let mut events = Vec::from([keys]);
    // SAFETY: the copy is only waited on and closed once, below.
    events.extend(timer.as_ref().map(|t| unsafe { t.unsafe_clone() }));
    let signaled = boot::wait_for_event(&events).ok();
    if let Some(timer) = timer {
        let _ = boot::close_event(timer);
    }
    match signaled {
        Some(0) => pressed().map_or(Wait::Nothing, Wait::Key),
        Some(_) => Wait::Timeout,
        None => Wait::NoInput,
    }
}

/// A key already waiting, without waiting.
fn pressed() -> Option<Press> {
    Some(
        match uefi::system::with_stdin(|input| input.read_key()).ok()?? {
            Key::Printable(character) => match char::from(character) {
                '\r' | '\n' => Press::Enter,
                character => Press::Character(character),
            },
            Key::Special(ScanCode::UP) => Press::Up,
            Key::Special(ScanCode::DOWN) => Press::Down,
            Key::Special(ScanCode::LEFT) => Press::Left,
            Key::Special(ScanCode::RIGHT) => Press::Right,
            Key::Special(ScanCode::HOME) => Press::Home,
            Key::Special(ScanCode::END) => Press::End,
            Key::Special(ScanCode::DELETE) => Press::Delete,
            Key::Special(ScanCode::ESCAPE) => Press::Escape,
            Key::Special(_) => Press::Other,
        },
    )
}

/// The volume's startup choices; `None` when it has none or they cannot be
/// read, which is reported.
pub(crate) fn read_legacy(volume: &mut FileSystem) -> Option<Choices> {
    let path = esp::find(volume, CHOICES).ok()??;
    let choices = volume
        .read(&path)
        .map_err(|_| refuse!("cannot read choices.cfg"))
        .and_then(|bytes| String::from_utf8(bytes).map_err(|_| refuse!("choices.cfg is not UTF-8")))
        .and_then(|text| Ok(Choices::parse(&text)?));
    match choices {
        Ok(choices) => Some(choices),
        Err(error) => {
            output::arguments(format_args!("CHOICES refused: {error}\n"));
            None
        }
    }
}

fn read(volume: &mut FileSystem) -> Option<Choices> {
    let legacy = read_legacy(volume);
    crate::firmware::northwood_choices(volume, legacy)
}

fn saved_disk(volume: &mut FileSystem) -> Result<Option<String>, &'static str> {
    let Some(path) =
        esp::find(volume, SELECTED_DISK).map_err(|_| "cannot locate the saved install target")?
    else {
        return Ok(None);
    };
    let bytes = volume
        .read(&path)
        .map_err(|_| "cannot read the saved install target")?;
    let text = String::from_utf8(bytes).map_err(|_| "saved install target is not UTF-8")?;
    Ok(Some(text))
}

/// The partition Install macOS chose (`SERIAL:GUID`), when it chose one.
pub fn chosen_disk(volume: &mut FileSystem) -> Option<String> {
    let text = saved_disk(volume).ok()??;
    let disk = text.trim();
    match disk.parse::<stela_runtime::pc::storage::Target>() {
        Ok(_) => Some(disk.into()),
        Err(reason) => {
            output::arguments(format_args!("CHOSEN DISK refused: {reason}\n"));
            None
        }
    }
}

#[cfg(feature = "diagnostics")]
pub(crate) fn diagnostic_disk(
    volume: &mut FileSystem,
) -> Result<Option<stela_runtime::pc::storage::Target>, &'static str> {
    saved_disk(volume)?
        .map(|text| text.trim().parse())
        .transpose()
}

/// `options` with the chosen disk in place of [`CHOSEN_DISK`], or as they
/// are while none is chosen, which the default boot then holds for.
pub fn resolve(volume: &mut FileSystem, options: Vec<u8>) -> Vec<u8> {
    let Ok(text) = core::str::from_utf8(&options) else {
        return options;
    };
    match with_disk(text, chosen_disk(volume).as_deref()) {
        Some(resolved) => resolved.into_bytes(),
        None => options,
    }
}

/// Whether `options` still name a disk Install macOS has not chosen.
pub fn unchosen(options: &[u8]) -> bool {
    core::str::from_utf8(options)
        .is_ok_and(|text| text.split_ascii_whitespace().any(|w| w == CHOSEN_DISK))
}

/// The heading over the firmware's entries.
const PC: &str = "This PC";
const DIAGNOSTICS: &str = "Diagnostics";

/// The SSD (`--storage`) a startup choice's options name.
fn storage(options: &str) -> Option<&str> {
    let mut words = options.split_ascii_whitespace();
    words.find(|word| *word == "--storage")?;
    words.next()
}

/// One entry of the menu.
enum Item {
    /// Entry `n` of the startup choices.
    Choice(usize, ChoiceKind),
    /// Entry `n`, an Install macOS that chooses its disk, on a disk other
    /// than the one chosen before; named.
    Another(usize, String),
    /// One of the PC firmware's own boot entries.
    Firmware(Entry),
    Diagnostics,
}

/// The menu.
struct Menu<'a> {
    choices: &'a Choices,
    /// The partition Install macOS chose, when it chose one.
    disk: Option<String>,
    /// The saved target is present and no ready Mac entry uses it.
    /// [`Item::Another`] opens the picker while this shortcut is available.
    present: bool,
    items: Vec<Item>,
    selected: usize,
    /// Seconds before the first entry starts, while the countdown runs.
    countdown: Option<u32>,
    /// The first entry starts unless a key asks for the menu.
    glance: bool,
    /// Why the last firmware entry chosen did not start.
    notice: Option<&'static str>,
    /// Why the machine can only be set up, shown while nothing else is.
    held: Option<&'a str>,
}

impl<'a> Menu<'a> {
    /// The menu of `choices`: those that can start now, then the firmware's
    /// entries. `held` is why the machine can only be set up.
    fn new(
        choices: &'a Choices,
        held: Option<&'a str>,
        disk: Option<String>,
        allow_diagnostics: bool,
    ) -> Self {
        // A choice on a disk not chosen yet waits for Install macOS.
        let resolved: Vec<Option<String>> = choices
            .entries
            .iter()
            .map(|entry| with_disk(&entry.options, disk.as_deref()))
            .collect();
        let readiness: Vec<Readiness> = choices
            .entries
            .iter()
            .zip(&resolved)
            .map(|(entry, options)| match options {
                Some(options) => selection::ready(options, entry.kind()),
                None => Readiness::Unprepared,
            })
            .collect();
        // The SSDs a macOS choice is known to start from,
        // which Install macOS has prepared already.
        let prepared: Vec<Option<&str>> = choices
            .entries
            .iter()
            .zip(&readiness)
            .zip(&resolved)
            .filter(|((_, r), _)| **r == Readiness::Ready)
            .map(|(_, options)| options.as_deref().and_then(storage))
            .collect();
        // A disk used by a ready Mac entry stays in the picker. A disk
        // without such an entry may still hold data, so reuse asks too.
        let present = disk
            .as_deref()
            .is_some_and(|disk| !prepared.contains(&Some(disk)) && selection::chosen_present(disk));
        let mut items = Vec::new();
        for (n, entry) in choices.entries.iter().enumerate() {
            let kind = entry.kind();
            let left_out = match (kind, readiness[n]) {
                (ChoiceKind::Mac, Readiness::NotInstalled) => Some("no macOS is installed"),
                (ChoiceKind::Mac, Readiness::Unprepared) => Some("its SSD is not prepared"),
                (ChoiceKind::Mac, Readiness::Stale(stale)) => Some(stale.reason()),
                // Install macOS that chooses its disk is always offered.
                (ChoiceKind::Setup, _)
                    if !unchosen(entry.options.as_bytes())
                        && prepared.contains(&storage(&entry.options)) =>
                {
                    Some("its SSD is prepared")
                }
                _ => None,
            };
            match left_out {
                Some(reason) => output::arguments(format_args!(
                    "CHOICE {} ({}) left out: {reason}\n",
                    n + 1,
                    entry.name
                )),
                None => {
                    items.push(Item::Choice(n, kind));
                    if present && kind == ChoiceKind::Setup && unchosen(entry.options.as_bytes()) {
                        items.push(Item::Another(n, format!("{} on another disk", entry.name)));
                    }
                }
            }
        }
        // A setup without its restore choice still lists its own.
        if items.is_empty() {
            items.extend(
                choices
                    .entries
                    .iter()
                    .enumerate()
                    .map(|(n, entry)| Item::Choice(n, entry.kind())),
            );
        }
        items.extend(entries::list().into_iter().map(Item::Firmware));
        if allow_diagnostics
            && cfg!(feature = "diagnostics")
            && !choices
                .entries
                .iter()
                .any(|entry| stela_boot_core::options::HostCheck::parse(&entry.options).is_ok())
        {
            items.push(Item::Diagnostics);
        }
        let selected = match held {
            Some(_) => items
                .iter()
                .position(|item| matches!(item, Item::Choice(_, ChoiceKind::Setup)))
                .unwrap_or(0),
            None => 0,
        };
        // The first entry starts by itself only when it starts the Mac,
        // with no countdown when that Mac is the only one ready.
        let first_mac = match items.first() {
            Some(Item::Choice(n, ChoiceKind::Mac)) => Some(readiness[*n] == Readiness::Ready),
            _ => None,
        };
        let ready_macs = items
            .iter()
            .filter(|item| {
                matches!(item, Item::Choice(n, ChoiceKind::Mac) if readiness[*n] == Readiness::Ready)
            })
            .count();
        let opens = match held {
            Some(_) => Opening::Wait,
            None => opening(choices.timeout_seconds, first_mac, ready_macs),
        };
        Self {
            choices,
            disk,
            present,
            countdown: match opens {
                Opening::Countdown(seconds) => Some(seconds),
                Opening::Glance | Opening::Wait => None,
            },
            glance: opens == Opening::Glance,
            items,
            selected,
            notice: None,
            held,
        }
    }

    fn name(&self, item: &'a Item) -> &'a str {
        match item {
            Item::Choice(n, _) => &self.choices.entries[*n].name,
            Item::Another(_, name) => name,
            Item::Firmware(entry) => &entry.name,
            Item::Diagnostics => DIAGNOSTICS,
        }
    }

    /// The entries under the header, the highlighted one raised with an
    /// orange mark, then the countdown or a notice.
    fn draw(&self, canvas: &mut Canvas) {
        const ROW: usize = 36;
        let (width, height) = canvas.size();
        canvas.fill(0, 0, width, height, ui::BACKGROUND);
        let top = canvas.header() + 24;
        let column = width.saturating_sub(32).min(440);
        let left = (width - column) / 2;
        let fits = (height.saturating_sub(top + 120) / ROW).max(1);
        let first = stela_boot_core::screen::window(self.selected, self.items.len(), fits);
        let mut y = top;
        let mut firmware = false;
        for index in (first..self.items.len()).take(fits) {
            let item = &self.items[index];
            if !firmware && matches!(item, Item::Firmware(_) | Item::Diagnostics) {
                firmware = true;
                y += 16;
                canvas.text(left + 16, y, PC, 1, ui::DIM);
                y += 28;
            }
            if index == self.selected {
                canvas.panel(left, y, column, ROW - 4, 8, ui::RAISED);
                canvas.panel(left + 6, y + 9, 3, ROW - 22, 1, ui::ORANGE);
            }
            canvas.text(left + 20, y + 8, self.name(item), 1, ui::TEXT);
            y += ROW;
        }
        let (status, colour) = match (self.notice, self.countdown, self.held) {
            (Some(notice), _, _) => (String::from(notice), ui::ORANGE),
            (None, Some(seconds), _) => (
                format!("{} in {seconds}", self.name(&self.items[0])),
                ui::DIM,
            ),
            (None, None, Some(held)) => (String::from(held), ui::DIM),
            (None, None, None) => return,
        };
        // A status of two lines raises its first line by one row.
        let lines = status.lines().count();
        for (n, line) in status.lines().enumerate() {
            let y = height.saturating_sub(56 + (lines - 1 - n) * 24);
            canvas.text_centred(width / 2, y, line, 1, colour);
        }
    }

    /// The chosen item; `None` without console input.
    fn run(&mut self) -> Option<usize> {
        console::release();
        let (screen, font) = Framebuffer::from_firmware().zip(font::console())?;
        let mut canvas = Canvas::new(screen, font);
        loop {
            self.draw(&mut canvas);
            canvas.show();
            let wait = key(self.countdown.map(|_| TICK));
            if let Wait::Key(_) = wait {
                self.countdown = None;
                self.notice = None;
            }
            let items = self.items.len();
            match wait {
                Wait::Key(Press::Up) => self.selected = (self.selected + items - 1) % items,
                Wait::Key(Press::Down) => self.selected = (self.selected + 1) % items,
                Wait::Key(Press::Enter) => return Some(self.selected),
                Wait::Key(Press::Character(key)) => {
                    let chosen = (0..items).find(|&n| Choices::key(n) == Some(key));
                    if chosen.is_some() {
                        return chosen;
                    }
                }
                Wait::Key(_) | Wait::Nothing => {}
                Wait::Timeout => match self.countdown {
                    Some(1) => return Some(0),
                    Some(seconds) => self.countdown = Some(seconds - 1),
                    None => {}
                },
                Wait::NoInput => return None,
            }
        }
    }
}

/// The selected option line and the menu name that supplied it.
pub(crate) struct Selection {
    pub(crate) options: Vec<u8>,
    pub(crate) entry: Option<String>,
}

/// Log the chosen entry; its options, on `disk`.
fn chosen(choices: &Choices, index: usize, disk: Option<&str>) -> Selection {
    let entry = &choices.entries[index];
    output::arguments(format_args!("CHOICE {} ({})\n", index + 1, entry.name));
    status::info(&format!("Starting {}", entry.name));
    Selection {
        options: with_disk(&entry.options, disk)
            .unwrap_or_else(|| entry.options.clone())
            .into_bytes(),
        entry: Some(entry.name.clone()),
    }
}

/// Whether a key asks for the menu: one already pressed or held, or one
/// pressed within [`GLANCE`].
fn asked() -> bool {
    pressed().is_some() || matches!(key(Some(GLANCE)), Wait::Key(_))
}

/// Confirm erasing the saved target. Selecting Install macOS alone grants
/// no permission to blank it, even if the startup check cannot boot it.
fn confirm_again(disk: &str) -> bool {
    let Some((serial, partition)) = disk.rsplit_once(':') else {
        return false;
    };
    let Some((screen, font)) = Framebuffer::from_firmware().zip(font::console()) else {
        return false;
    };
    let mut canvas = Canvas::new(screen, font);
    let (width, height) = canvas.size();
    let middle = width / 2;
    let mut erase = false;
    loop {
        canvas.fill(0, 0, width, height, ui::BACKGROUND);
        let top = canvas.header() + 24;
        canvas.text_centred(middle, top, "Erase and install macOS?", 1, ui::TEXT);
        canvas.text_centred(middle, top + 30, "Disk serial", 1, ui::DIM);
        canvas.text_centred(middle, top + 50, serial, 1, ui::TEXT);
        canvas.text_centred(middle, top + 80, "Partition GUID", 1, ui::DIM);
        canvas.text_centred(middle, top + 100, partition, 1, ui::TEXT);
        canvas.text_centred(
            middle,
            top + 130,
            "Erases this partition and its machine state.",
            1,
            ui::ORANGE,
        );
        let back_colour = if erase { ui::TEXT } else { ui::ORANGE };
        let erase_colour = if erase { ui::ORANGE } else { ui::TEXT };
        canvas.text_centred(width / 4, top + 170, "Back", 1, back_colour);
        canvas.text_centred(
            width - width / 4,
            top + 170,
            "Erase and Install",
            1,
            erase_colour,
        );
        canvas.show();
        match key(None) {
            Wait::Key(Press::Left) => erase = false,
            Wait::Key(Press::Right) => erase = true,
            Wait::Key(Press::Enter) => return erase,
            Wait::Key(Press::Escape) | Wait::NoInput => return false,
            _ => {}
        }
    }
}

/// Wait for a boot choice. Setup with no reusable saved target and the
/// "on another disk" item open the picker. Reusing a saved target asks
/// for confirmation before blanking it. Leaving either returns to the menu.
fn menu(volume: &mut FileSystem, choices: &Choices, mut menu: Menu<'_>) -> Option<Selection> {
    loop {
        let index = menu.run()?;
        let (n, picker) = match &menu.items[index] {
            Item::Another(n, _) => (*n, true),
            Item::Choice(n, ChoiceKind::Setup)
                if unchosen(choices.entries[*n].options.as_bytes()) =>
            {
                (*n, !menu.present)
            }
            Item::Choice(n, _) => return Some(chosen(choices, *n, menu.disk.as_deref())),
            Item::Firmware(entry) => {
                output::arguments(format_args!("CHOICE firmware entry ({})\n", entry.name));
                menu.notice = Some(entries::start(entry.number));
                continue;
            }
            Item::Diagnostics => {
                output::arguments(format_args!("CHOICE {DIAGNOSTICS}\n"));
                status::info("Starting Diagnostics");
                return Some(Selection {
                    options: stela_boot_core::options::HOST_CHECK.as_bytes().to_vec(),
                    entry: None,
                });
            }
        };
        if !picker {
            let disk = menu.disk.as_deref()?;
            if !confirm_again(disk) {
                continue;
            }
            if !selection::chosen_present(disk) {
                menu.notice = Some("The disk chosen before changed or is no longer unique.");
                menu.present = false;
                continue;
            }
            match crate::installer::again(disk) {
                Ok(()) => return Some(chosen(choices, n, Some(disk))),
                Err(error) => {
                    output::arguments(format_args!("INSTALLER again refused: {error}\n"));
                    menu.notice = Some("The disk chosen before could not be prepared again.");
                    continue;
                }
            }
        }
        let Some(disk) = crate::installer::choose() else {
            continue;
        };
        let recorded = esp::destination(volume, SELECTED_DISK)
            .is_ok_and(|path| volume.write(path, format!("{disk}\n")).is_ok());
        if !recorded {
            menu.notice = Some("The chosen disk could not be recorded on the USB.");
            continue;
        }
        output::arguments(format_args!("CHOSEN DISK {disk}\n"));
        return Some(chosen(choices, n, Some(&disk)));
    }
}

/// The options to boot: the chosen entry's, or `options` (`stela.cfg`).
/// `held` is why the default boot can only wait for the setup restore, when
/// it can.
pub(crate) fn choose(volume: &mut FileSystem, options: Vec<u8>, held: Option<&str>) -> Selection {
    if let Some(reason) = held {
        return hold_menu(volume, reason, true).unwrap_or_else(|| without_choices(volume, options));
    }
    let Some(choices) = read(volume) else {
        return without_choices(volume, options);
    };
    // Diagnostic media allow the same brief key window even with a zero
    // timeout, so the memory refusal's next step works on those media.
    if choices.timeout_seconds == 0
        && !(if cfg!(feature = "diagnostics") {
            asked()
        } else {
            pressed().is_some()
        })
    {
        return chosen(&choices, 0, chosen_disk(volume).as_deref());
    }
    let opened = Menu::new(&choices, None, chosen_disk(volume), true);
    if opened.glance
        && !asked()
        && let Some(Item::Choice(n, _)) = opened.items.first()
    {
        return chosen(&choices, *n, opened.disk.as_deref());
    }
    menu(volume, &choices, opened)
        .unwrap_or_else(|| chosen(&choices, 0, chosen_disk(volume).as_deref()))
}

/// Make Diagnostics reachable even on media with only a default option line.
fn without_choices(volume: &mut FileSystem, options: Vec<u8>) -> Selection {
    if cfg!(feature = "diagnostics")
        && let Ok(text) = core::str::from_utf8(&options)
        && stela_boot_core::options::System::is_system(text)
        && asked()
    {
        let choices = Choices {
            timeout_seconds: 0,
            entries: alloc::vec![stela_loader::mount_interface::Choice {
                name: String::from("Continue startup"),
                options: String::from(text),
            }],
        };
        let opened = Menu::new(&choices, None, chosen_disk(volume), true);
        if let Some(selected) = menu(volume, &choices, opened)
            && selected.options.as_slice() == stela_boot_core::options::HOST_CHECK.as_bytes()
        {
            return selected;
        }
    }
    Selection {
        options,
        entry: None,
    }
}

/// Wait at the menu after `reason`, with no countdown, as LLB waits in
/// recovery mode for a restore host: the machine can only be set up. The
/// setup entry starts highlighted; any entry may be chosen, and one that
/// cannot boot either returns here. Returns the chosen entry's options;
/// `None` when the volume has no choices or no console input to choose with.
pub(crate) fn hold(volume: &mut FileSystem, reason: &str) -> Option<Selection> {
    hold_menu(volume, reason, false)
}

fn hold_menu(volume: &mut FileSystem, reason: &str, allow_diagnostics: bool) -> Option<Selection> {
    let choices = read(volume)?;
    output::arguments(format_args!("HOLD {reason}\n"));
    let held = Menu::new(
        &choices,
        Some(reason),
        chosen_disk(volume),
        allow_diagnostics,
    );
    menu(volume, &choices, held)
}
