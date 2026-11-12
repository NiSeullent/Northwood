//! Terminal-only progress; redirected commands keep their ordinary output.
//! Once the setup wizard has drawn its timeline, each phase continues it: a
//! running step on the rail, then the finished step it leaves behind.
use indicatif::{HumanBytes, ProgressBar, ProgressStyle};
use std::{
    io::{self, IsTerminal, Read, Write},
    path::Path,
    sync::OnceLock,
    time::Duration,
};

/// The wizard timeline's styled glyphs: a running, finished and stopped
/// step, and the rail between steps.
pub(crate) struct Rail {
    pub(crate) active: String,
    pub(crate) done: String,
    pub(crate) stopped: String,
    pub(crate) bar: String,
}

static RAIL: OnceLock<Rail> = OnceLock::new();

/// Draw every later phase on `rail`.
pub(crate) fn follow(rail: Rail) {
    let _ = RAIL.set(rail);
}

/// The start of a step's first line: its glyph on the rail, if any.
fn step(glyph: fn(&Rail) -> &str) -> String {
    RAIL.get()
        .map(|rail| format!("{}  ", glyph(rail)))
        .unwrap_or_default()
}

/// The start of a step's later lines: the rail, if any.
fn detail() -> String {
    RAIL.get()
        .map(|rail| format!("{}  ", rail.bar))
        .unwrap_or_default()
}

/// The columns left free at the right edge.
const MARGIN: &str = "  ";

pub(crate) struct Phase {
    bar: Option<ProgressBar>,
    label: String,
    finished: bool,
}

impl Phase {
    pub(crate) fn bytes(label: impl Into<String>, bytes: u64) -> Self {
        let label = label.into();
        let bar = io::stderr().is_terminal().then(|| {
            let template = format!(
                "{}{{msg}}\n{}{{wide_bar:.173/239}} {{percent:>3}}%  {{bytes}}/{{total_bytes}}  \
                 {{bytes_per_sec}}  ETA {{eta_precise}}{MARGIN}",
                step(|rail| &rail.active),
                detail(),
            );
            let bar = ProgressBar::new(bytes).with_style(
                ProgressStyle::with_template(&template)
                    .expect("a fixed progress template")
                    .progress_chars("█▌░"),
            );
            bar.set_message(label.clone());
            bar.enable_steady_tick(Duration::from_millis(200));
            bar
        });
        Self {
            bar,
            label,
            finished: false,
        }
    }

    pub(crate) fn spinner(label: impl Into<String>) -> Self {
        let label = label.into();
        let bar = io::stderr().is_terminal().then(|| {
            let template = format!("{}{{msg}}  {{elapsed:.dim}}", step(|rail| &rail.active));
            let bar = ProgressBar::new_spinner().with_style(
                ProgressStyle::with_template(&template).expect("a fixed progress template"),
            );
            bar.set_message(label.clone());
            bar.enable_steady_tick(Duration::from_secs(1));
            bar
        });
        Self {
            bar,
            label,
            finished: false,
        }
    }

    pub(crate) fn advance(&self, bytes: u64) {
        if let Some(bar) = &self.bar {
            bar.inc(bytes);
        }
    }

    pub(crate) fn position(&self, bytes: u64) {
        if let Some(bar) = &self.bar {
            bar.set_position(bytes);
            bar.reset_eta();
        }
    }

    pub(crate) fn add_work(&self, bytes: u64) {
        if let Some(bar) = &self.bar {
            bar.set_length(bar.length().unwrap_or(0).saturating_add(bytes));
        }
    }

    pub(crate) fn finish(mut self) {
        if let Some(bar) = &self.bar {
            // A one-line summary replaces the finished timers. Off the
            // timeline, a step without a size leaves nothing behind.
            match (bar.length(), RAIL.get()) {
                (None, None) => bar.finish_and_clear(),
                (None, Some(_)) => {
                    bar.set_style(summary(|rail| &rail.done));
                    bar.finish_with_message(self.label.clone());
                }
                (Some(total), _) => {
                    bar.set_style(summary(|rail| &rail.done));
                    bar.finish_with_message(format!("{} · {}", self.label, HumanBytes(total)));
                }
            }
        }
        self.finished = true;
    }
}

/// A finished or stopped step: its glyph and message, then the rail.
fn summary(glyph: fn(&Rail) -> &str) -> ProgressStyle {
    let rail = RAIL
        .get()
        .map(|rail| format!("\n{}", rail.bar))
        .unwrap_or_default();
    ProgressStyle::with_template(&format!("{}{{msg}}{rail}", step(glyph)))
        .expect("a fixed progress template")
}

impl Drop for Phase {
    fn drop(&mut self) {
        if !self.finished
            && let Some(bar) = &self.bar
        {
            bar.set_style(summary(|rail| &rail.stopped));
            bar.abandon_with_message(format!("{} stopped", self.label));
        }
    }
}

pub(crate) fn copy(from: &Path, to: &Path, progress: &Phase) -> io::Result<()> {
    let mut input = std::fs::File::open(from)?;
    let mut output = std::fs::File::create(to)?;
    let mut buffer = vec![0; 1024 * 1024];
    loop {
        let count = match input.read(&mut buffer) {
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            result => result?,
        };
        if count == 0 {
            break;
        }
        output.write_all(&buffer[..count])?;
        progress.advance(count as u64);
    }
    output.sync_all()?;
    drop_cached(&output);
    Ok(())
}

/// Drop `file`'s written pages from the cache after `sync_all`, so a check
/// that reads the file back reads what the disk holds rather than what was
/// written.
pub(crate) fn drop_cached(file: &std::fs::File) {
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::io::AsRawFd;
        // SAFETY: an advice call on a descriptor `file` keeps open.
        unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
    }
    #[cfg(not(target_os = "linux"))]
    let _ = file;
}
