//! Upstream sources: the git submodules in `third_party/`, each at the
//! commit its gitlink pins. `fetch` checks each one out shallowly, limited to
//! the patterns in `third_party/<name>.sparse` where that file exists, and
//! applies Stela's series from `third_party/patches/<name>/` as commits on
//! the local branch `stela`. `export` writes a submodule's commits since its
//! pin back to that series, each file keeping its short name while its
//! commit's subject stays the same, with the repository's configured identity
//! as the author of Stela's own patches and every such date in UTC.
use super::support::capture;
use crate::tools::{self, exec};
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

/// The series applied last and the commit it produced, in the submodule's
/// git directory.
const STATE: &str = "stela-series";

/// The committer of the series commits `checkout` applies.
const COMMITTER_EMAIL: &str = "stela@localhost";

struct Submodule {
    name: String,
    /// The checkout, relative to the repository root.
    path: String,
    url: String,
    /// The commit the superproject's gitlink pins.
    pin: String,
}

impl Submodule {
    fn directory(&self) -> PathBuf {
        tools::root().join(&self.path)
    }

    fn patches_directory(&self) -> PathBuf {
        tools::root().join("third_party/patches").join(&self.name)
    }

    /// Non-cone sparse-checkout patterns, or `None` for the whole tree.
    fn sparse(&self) -> Result<Option<String>> {
        let path = tools::root()
            .join("third_party")
            .join(format!("{}.sparse", self.name));
        match fs::read_to_string(&path) {
            Ok(text) => Ok(Some(
                text.lines()
                    .filter(|line| !line.is_empty() && !line.starts_with('#'))
                    .map(|line| format!("{line}\n"))
                    .collect(),
            )),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error).with_context(|| path.display().to_string()),
        }
    }

    fn patches(&self) -> Result<Vec<PathBuf>> {
        let mut patches = match fs::read_dir(self.patches_directory()) {
            Ok(entries) => entries
                .map(|entry| Ok(entry?.path()))
                .collect::<Result<Vec<_>>>()?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Vec::new(),
            Err(error) => return Err(error.into()),
        };
        patches.retain(|path| path.extension().is_some_and(|e| e == "patch"));
        patches.sort();
        Ok(patches)
    }

    /// Binds the pin, sparse scope and patch series.
    fn series(&self) -> Result<String> {
        let mut digest = Sha256::new();
        digest.update(self.pin.as_bytes());
        digest.update(self.sparse()?.unwrap_or_default().as_bytes());
        for patch in self.patches()? {
            digest.update(patch.file_name().unwrap().as_encoded_bytes());
            digest.update(Sha256::digest(fs::read(&patch)?));
        }
        Ok(format!("{:x}", digest.finalize()))
    }

    fn git(&self) -> Command {
        git(&self.directory())
    }

    /// Whether the checkout has a commit, which a fetch stopped before its
    /// first checkout leaves it without.
    fn checked_out(&self) -> Result<bool> {
        Ok(self
            .git()
            .args(["rev-parse", "--verify", "-q", "HEAD"])
            .stdout(Stdio::null())
            .status()?
            .success())
    }

    fn head(&self) -> Result<String> {
        Ok(capture(self.git().args(["rev-parse", "HEAD"]))?
            .trim()
            .to_owned())
    }

    fn state_path(&self) -> Result<PathBuf> {
        let directory = capture(self.git().args(["rev-parse", "--absolute-git-dir"]))?;
        Ok(Path::new(directory.trim()).join(STATE))
    }

    fn state(&self) -> Result<Option<(String, String)>> {
        let Ok(text) = fs::read_to_string(self.state_path()?) else {
            return Ok(None);
        };
        Ok(text
            .trim()
            .split_once(' ')
            .map(|(series, head)| (series.to_owned(), head.to_owned())))
    }

    fn record(&self, series: &str) -> Result<()> {
        fs::write(self.state_path()?, format!("{series} {}\n", self.head()?))?;
        Ok(())
    }

    /// Whether the checkout already holds the pinned commit.
    fn has_pin(&self) -> Result<bool> {
        Ok(self
            .git()
            .args(["cat-file", "-e", &format!("{}^{{commit}}", self.pin)])
            .stderr(Stdio::null())
            .status()?
            .success())
    }

    /// Refuses a gitlink that names a commit `checkout` made by applying the
    /// series: no clone can fetch it. `git commit -a` or `git add` records one
    /// when the index has lost `hold_pin`'s flag.
    fn ensure_upstream_pin(&self) -> Result<()> {
        let committer = capture(
            self.git()
                .args(["log", "-1", "--format=%ce", &self.pin, "--"]),
        )?;
        ensure!(
            committer.trim() != COMMITTER_EMAIL,
            "{} pins {}, a local commit of Stela's patch series; restore its upstream pin \
             with `git update-index --cacheinfo 160000,<pin>,{}`",
            self.path,
            self.pin,
            self.path
        );
        Ok(())
    }

    /// Keeps `git commit -a` and `git add` from recording the checkout's HEAD,
    /// a local commit of the series, as the gitlink. The flag lives in the
    /// index, so each fetch sets it again; a new pin goes in with
    /// `git update-index --cacheinfo`.
    fn hold_pin(&self) -> Result<()> {
        exec(git(tools::root()).args(["update-index", "--assume-unchanged", "--", &self.path]))
    }

    /// Commits or tracked edits that `export` has not written to the series.
    /// A checkout at its bare pin, as `git submodule update` leaves it, has none.
    fn unexported(&self) -> Result<bool> {
        let dirty = !capture(
            self.git()
                .args(["status", "--porcelain", "--untracked-files=no"]),
        )?
        .is_empty();
        let head = self.head()?;
        let recorded = self.state()?.map(|(_, head)| head);
        Ok(dirty || (head != self.pin && recorded.as_ref() != Some(&head)))
    }
}

fn git(directory: &Path) -> Command {
    let mut command = Command::new("git");
    command.current_dir(directory);
    command
}

fn submodules() -> Result<Vec<Submodule>> {
    let root = tools::root();
    let config = |key: &str| -> Result<String> {
        Ok(
            capture(git(root).args(["config", "-f", ".gitmodules", "--get", key]))?
                .trim()
                .to_owned(),
        )
    };
    let paths = capture(git(root).args([
        "config",
        "-f",
        ".gitmodules",
        "--get-regexp",
        r"^submodule\..*\.path$",
    ]))?;
    let mut submodules = Vec::new();
    for line in paths.lines() {
        let (key, path) = line.split_once(' ').context("malformed .gitmodules")?;
        let name = key
            .strip_prefix("submodule.")
            .and_then(|key| key.strip_suffix(".path"))
            .context("malformed .gitmodules")?;
        // The gitlink in the index, as `git submodule update` reads it.
        let entry = capture(git(root).args(["ls-files", "--stage", "--", path]))?;
        let pin = entry
            .strip_prefix("160000 ")
            .and_then(|entry| entry.split_whitespace().next())
            .with_context(|| format!("{path} is not a submodule in the index"))?;
        submodules.push(Submodule {
            name: name.to_owned(),
            path: path.to_owned(),
            url: config(&format!("submodule.{name}.url"))?,
            pin: pin.to_owned(),
        });
    }
    Ok(submodules)
}

/// Creates the checkout the way `git submodule update --init` would, but
/// sparse before anything is checked out, then moves its git directory
/// into the superproject's.
fn initialize(submodule: &Submodule) -> Result<()> {
    let root = tools::root();
    let url = format!("submodule.{}.url", submodule.name);
    if !git(root)
        .args(["config", "--get", &url])
        .stdout(Stdio::null())
        .status()?
        .success()
    {
        exec(git(root).args(["submodule", "init", "--", &submodule.path]))?;
    }
    let directory = submodule.directory();
    fs::create_dir_all(&directory)?;
    exec(git(&directory).args(["init", "-q"]))?;
    // Pinned sources hash and build the same on every host.
    exec(git(&directory).args(["config", "core.autocrlf", "false"]))?;
    exec(git(&directory).args(["remote", "add", "origin", &submodule.url]))?;
    first_checkout(submodule)
}

fn first_checkout(submodule: &Submodule) -> Result<()> {
    checkout(submodule)?;
    exec(git(tools::root()).args(["submodule", "absorbgitdirs", "--", &submodule.path]))
}

fn apply_patches(directory: &Path, name: &str, patches: &[PathBuf]) -> Result<()> {
    if patches.is_empty() {
        return Ok(());
    }
    let applied = exec(
        git(directory)
            .args(["-c", "user.name=Stela", "-c"])
            .arg(format!("user.email={COMMITTER_EMAIL}"))
            .args(["am", "-q", "--3way", "--committer-date-is-author-date"])
            .args(patches),
    );
    if applied.is_err() {
        // Leave the checkout at its pin, with no `am` left to block the next.
        exec(git(directory).args(["am", "--abort"]))?;
    }
    applied.with_context(|| format!("third_party/patches/{name} no longer applies"))
}

fn set_sparse_checkout(directory: &Path, patterns: &str) -> Result<()> {
    let mut sparse = git(directory)
        .args(["sparse-checkout", "set", "--no-cone", "--stdin"])
        .stdin(Stdio::piped())
        .spawn()?;
    sparse
        .stdin
        .take()
        .unwrap()
        .write_all(patterns.as_bytes())?;
    ensure!(sparse.wait()?.success(), "git sparse-checkout failed");
    Ok(())
}

/// Checks out the pin with its sparse scope and applies the series, fetching
/// from the URL `.gitmodules` names now.
fn checkout(submodule: &Submodule) -> Result<()> {
    println!("GPU source {}: {}", submodule.name, submodule.pin);
    exec(git(tools::root()).args(["submodule", "sync", "-q", "--", &submodule.path]))?;
    match submodule.sparse()? {
        Some(patterns) => set_sparse_checkout(&submodule.directory(), &patterns)?,
        None => exec(submodule.git().args(["sparse-checkout", "disable"]))?,
    }
    if !submodule.has_pin()? {
        exec(submodule.git().args([
            "fetch",
            "-q",
            "--depth=1",
            "--filter=blob:none",
            "origin",
            &submodule.pin,
        ]))?;
    }
    exec(
        submodule
            .git()
            .args(["checkout", "-q", "-f", "-B", "stela", &submodule.pin]),
    )?;
    apply_patches(
        &submodule.directory(),
        &submodule.name,
        &submodule.patches()?,
    )?;
    submodule.record(&submodule.series()?)
}

/// Prepares a private pinned Mesa checkout with the lavapipe ICD target.
pub(super) fn checkout_lavapipe_mesa(directory: &Path) -> Result<Value> {
    let mesa = submodules()?
        .into_iter()
        .find(|submodule| submodule.name == "mesa")
        .context("no Mesa submodule in .gitmodules")?;
    println!("GPU source {}: {}", mesa.name, mesa.pin);
    fs::create_dir(directory)?;
    exec(git(directory).args(["init", "-q"]))?;
    exec(git(directory).args(["config", "core.autocrlf", "false"]))?;
    let mut sparse = mesa.sparse()?;
    if let Some(patterns) = &mut sparse {
        patterns.push_str("/src/gallium/targets/lavapipe/\n");
        set_sparse_checkout(directory, patterns)?;
    }
    exec(git(directory).args([
        "fetch",
        "-q",
        "--depth=1",
        "--no-tags",
        &mesa.url,
        &mesa.pin,
    ]))?;
    exec(git(directory).args(["checkout", "-q", "--detach", &mesa.pin]))?;

    let patches = mesa.patches()?;
    let inputs = patches
        .iter()
        .map(|patch| tools::binding(patch))
        .collect::<Result<Vec<_>>>()?;
    apply_patches(directory, &mesa.name, &patches)?;
    for (patch, expected) in patches.iter().zip(&inputs) {
        ensure!(
            tools::binding(patch)? == *expected,
            "{} changed during checkout",
            patch.display()
        );
    }
    let head = capture(git(directory).args(["rev-parse", "HEAD"]))?;
    Ok(json!({
        "pin": mesa.pin,
        "patched_commit": head.trim(),
        "sparse_patterns": sparse.as_deref().map(|patterns| patterns.lines().collect::<Vec<_>>()),
        "patches": inputs,
    }))
}

/// Installs the pinned tools, then checks out every submodule at its pin with
/// Stela's series.
pub fn fetch(force: bool) -> Result<()> {
    super::toolchain::install()?;
    for submodule in submodules()? {
        submodule.hold_pin()?;
        if !submodule.directory().join(".git").exists() {
            initialize(&submodule)?;
            continue;
        }
        if !submodule.checked_out()? {
            first_checkout(&submodule)?;
            continue;
        }
        if submodule.has_pin()? {
            submodule.ensure_upstream_pin()?;
        }
        let unexported = submodule.unexported()?;
        if !unexported && submodule.state()? == Some((submodule.series()?, submodule.head()?)) {
            continue;
        }
        ensure!(
            force || !unexported,
            "{} has commits or edits not in third_party/patches/{}; run `cargo xtask gpu export` \
             or pass --force to discard them",
            submodule.path,
            submodule.name
        );
        checkout(&submodule)?;
    }
    Ok(())
}

/// Writes each submodule's commits since its pin to its patch series.
pub fn export() -> Result<()> {
    let identity = identity()?;
    for submodule in submodules()? {
        // A submodule never checked out has nothing to export.
        if !submodule.directory().join(".git").exists() {
            continue;
        }
        ensure!(
            capture(
                submodule
                    .git()
                    .args(["status", "--porcelain", "--untracked-files=no"])
            )?
            .is_empty(),
            "{} has uncommitted edits; commit them first",
            submodule.path
        );
        ensure!(
            submodule
                .git()
                .args(["merge-base", "--is-ancestor", &submodule.pin, "HEAD"])
                .status()?
                .success(),
            "{} is not based on its pinned commit {}",
            submodule.path,
            submodule.pin
        );
        let patches = submodule.patches_directory();
        let mut names = HashMap::new();
        for patch in submodule.patches()? {
            names.insert(subject(&patch)?, name(&patch));
            fs::remove_file(patch)?;
        }
        if submodule.head()? != submodule.pin {
            fs::create_dir_all(&patches)?;
            exec(
                submodule
                    .git()
                    .args([
                        "format-patch",
                        "-q",
                        "--no-numbered",
                        "--numbered-files",
                        "--zero-commit",
                        // Abbreviated blob names grow with the checkout.
                        "--full-index",
                        "--no-signature",
                    ])
                    .arg("-o")
                    .arg(&patches)
                    .arg(format!("{}..HEAD", submodule.pin)),
            )?;
            for number in 1.. {
                let patch = patches.join(number.to_string());
                if !patch.exists() {
                    break;
                }
                let subject = subject(&patch)?;
                let name = names.remove(&subject).unwrap_or_else(|| slug(&subject));
                let text = publish_header(&fs::read_to_string(&patch)?, &identity)
                    .with_context(|| patch.display().to_string())?;
                fs::write(patches.join(format!("{number:04}-{name}.patch")), text)?;
                fs::remove_file(patch)?;
            }
        } else if patches.is_dir() {
            fs::remove_dir(&patches)?;
        }
        submodule.record(&submodule.series()?)?;
    }
    Ok(())
}

/// The repository's configured `user.name <user.email>`, which `export`
/// writes as the author of Stela's own patches.
fn identity() -> Result<String> {
    let field = |key: &str| -> Result<String> {
        let value = capture(git(tools::root()).args(["config", key])).with_context(|| {
            format!("set git config {key}; export writes it as the patch author")
        })?;
        Ok(value.trim().to_owned())
    };
    Ok(format!(
        "{} <{}>",
        field("user.name")?,
        field("user.email")?
    ))
}

/// A patch with its header's `From:` set to `identity` and its `Date:` set to
/// the same instant in UTC, so the series carries neither a submodule commit's
/// own author address nor its time zone. A patch whose message has a
/// `Signed-off-by:` trailer is upstream work and keeps its author and date.
fn publish_header(patch: &str, identity: &str) -> Result<String> {
    let message = patch.split("\n---\n").next().unwrap_or(patch);
    if message
        .lines()
        .any(|line| line.starts_with("Signed-off-by: "))
    {
        return Ok(patch.to_owned());
    }
    let mut text = String::with_capacity(patch.len());
    let mut header = true;
    for line in patch.split_inclusive('\n') {
        let content = line.trim_end_matches('\n');
        header &= !content.is_empty();
        if header && content.starts_with("From: ") {
            text.push_str(&format!("From: {identity}\n"));
        } else if header && let Some(date) = content.strip_prefix("Date: ") {
            text.push_str(&format!("Date: {}\n", utc_date(date)?));
        } else {
            text.push_str(line);
        }
    }
    Ok(text)
}

const WEEKDAYS: [&str; 7] = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const MONTHS: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

/// An RFC 2822 date as `git format-patch` writes it (`Tue, 4 Nov 2025
/// 21:16:44 -0500`), converted to the same instant in UTC.
fn utc_date(date: &str) -> Result<String> {
    let parse = || -> Option<i64> {
        let fields: Vec<&str> = date.split_whitespace().collect();
        let [_, day, month, year, time, zone] = fields[..] else {
            return None;
        };
        let month = MONTHS.iter().position(|m| *m == month)? as i64 + 1;
        let mut clock = time.split(':').map(str::parse::<i64>);
        let (hour, minute, second) = (
            clock.next()?.ok()?,
            clock.next()?.ok()?,
            clock.next()?.ok()?,
        );
        let sign = match zone.as_bytes().first()? {
            b'+' => 1,
            b'-' => -1,
            _ => return None,
        };
        let zone: i64 = zone.get(1..)?.parse().ok()?;
        let offset = sign * ((zone / 100) * 3600 + (zone % 100) * 60);
        let days = days_from_civil(year.parse().ok()?, month, day.parse().ok()?);
        Some(days * 86400 + hour * 3600 + minute * 60 + second - offset)
    };
    let seconds = parse().with_context(|| format!("unreadable patch date {date:?}"))?;
    let days = seconds.div_euclid(86400);
    let time = seconds.rem_euclid(86400);
    let (year, month, day) = civil_from_days(days);
    Ok(format!(
        "{}, {day} {} {year} {:02}:{:02}:{:02} +0000",
        WEEKDAYS[(days + 4).rem_euclid(7) as usize],
        MONTHS[month as usize - 1],
        time / 3600,
        time / 60 % 60,
        time % 60
    ))
}

/// Days since 1970-01-01 of a proleptic Gregorian date.
fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let year = if month <= 2 { year - 1 } else { year };
    let era = year.div_euclid(400);
    let year_of_era = year - era * 400;
    let day_of_year = (153 * ((month + 9) % 12) + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146097 + day_of_era - 719468
}

/// The proleptic Gregorian date of a day count since 1970-01-01.
fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let days = days + 719468;
    let era = days.div_euclid(146097);
    let day_of_era = days - era * 146097;
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36524 - day_of_era / 146096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let shifted = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * shifted + 2) / 5 + 1;
    let month = if shifted < 10 {
        shifted + 3
    } else {
        shifted - 9
    };
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    (year, month, day)
}

/// A patch's subject, unfolded and without `[PATCH]`.
fn subject(patch: &Path) -> Result<String> {
    let text = fs::read_to_string(patch)?;
    let mut lines = text
        .lines()
        .skip_while(|line| !line.starts_with("Subject: "));
    let first = lines
        .next()
        .with_context(|| format!("{} has no subject", patch.display()))?;
    let mut subject = first["Subject: ".len()..]
        .trim_start_matches("[PATCH] ")
        .to_owned();
    for line in lines.take_while(|line| line.starts_with(' ')) {
        subject.push_str(line);
    }
    Ok(subject)
}

/// A patch file's name between its number and extension, which `export`
/// keeps while the commit's subject stays the same.
fn name(patch: &Path) -> String {
    let file = patch.file_stem().unwrap().to_string_lossy();
    file.split_once('-')
        .map_or(&*file, |(_, name)| name)
        .to_owned()
}

/// A new patch's name: the first three words of its subject's summary.
fn slug(subject: &str) -> String {
    let summary = subject
        .split_once(": ")
        .map_or(subject, |(_, summary)| summary);
    summary
        .replace('\'', "")
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|word| !word.is_empty())
        .take(3)
        .map(str::to_ascii_lowercase)
        .collect::<Vec<_>>()
        .join("-")
}

/// Each submodule's origin, pin, checked-out commit and series, for receipts.
pub fn origins() -> Result<Value> {
    submodules()?
        .iter()
        .map(|submodule| {
            let patches = submodule
                .patches()?
                .iter()
                .map(|patch| tools::binding(patch))
                .collect::<Result<Vec<_>>>()?;
            Ok(json!({
                "name": submodule.name, "url": submodule.url, "pin": submodule.pin,
                "head": submodule.head()?, "patches": patches,
            }))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_stopped_first_fetch_is_a_checkout_without_a_commit() {
        let directory = tempfile::tempdir().unwrap();
        let submodule = Submodule {
            name: "stopped".into(),
            path: directory.path().to_str().unwrap().into(),
            url: String::new(),
            pin: String::new(),
        };
        exec(submodule.git().args(["init", "-q"])).unwrap();
        assert!(!submodule.checked_out().unwrap());
        exec(submodule.git().args([
            "-c",
            "user.name=Stela",
            "-c",
            "user.email=stela@localhost",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "first",
        ]))
        .unwrap();
        assert!(submodule.checked_out().unwrap());
    }

    #[test]
    fn exported_headers_carry_the_configured_identity_in_utc() -> Result<()> {
        let identity = "Stela Author <author@example.com>";
        let patch = "From 0000000000000000000000000000000000000000 Mon Sep 17 00:00:00 2001\n\
                     From: Someone Else <someone@example.org>\n\
                     Date: Tue, 4 Nov 2025 21:16:44 -0500\n\
                     Subject: [PATCH] nouveau: name each GSP\n\n\
                     From: a body line\nDate: a body line\n---\n \
                     a | 1 +\n";
        assert_eq!(
            publish_header(patch, identity)?,
            "From 0000000000000000000000000000000000000000 Mon Sep 17 00:00:00 2001\n\
             From: Stela Author <author@example.com>\n\
             Date: Wed, 5 Nov 2025 02:16:44 +0000\n\
             Subject: [PATCH] nouveau: name each GSP\n\n\
             From: a body line\nDate: a body line\n---\n \
             a | 1 +\n"
        );
        // Upstream work keeps its author and date.
        let upstream = patch.replace(
            "\n---\n",
            "\nSigned-off-by: Someone Else <someone@example.org>\n---\n",
        );
        assert_eq!(publish_header(&upstream, identity)?, upstream);
        // Already in UTC, the header stays the same.
        let published = publish_header(patch, identity)?;
        assert_eq!(publish_header(&published, identity)?, published);
        assert_eq!(
            utc_date("Wed, 31 Dec 2025 20:00:00 -0500")?,
            "Thu, 1 Jan 2026 01:00:00 +0000"
        );
        assert_eq!(
            utc_date("Thu, 1 Jan 2026 00:30:00 +0100")?,
            "Wed, 31 Dec 2025 23:30:00 +0000"
        );
        assert_eq!(
            utc_date("Thu, 29 Feb 2024 23:59:59 -0730")?,
            "Fri, 1 Mar 2024 07:29:59 +0000"
        );
        assert_eq!(
            utc_date("Sun, 15 Jun 2025 15:37:57 +0200")?,
            "Sun, 15 Jun 2025 13:37:57 +0000"
        );
        assert!(utc_date("Tue, 4 Nov 2025 21:16:44").is_err());
        Ok(())
    }

    #[test]
    fn names_follow_subjects() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let patch = directory.path().join("0002-stela-compilers.patch");
        fs::write(
            &patch,
            "From 0000000000000000000000000000000000000000 Mon Sep 17 00:00:00 2001\n\
             Subject: [PATCH] meson: build Stela's shader compiler libraries without API\n \
             drivers\n\n---\n",
        )?;
        assert_eq!(
            subject(&patch)?,
            "meson: build Stela's shader compiler libraries without API drivers"
        );
        assert_eq!(name(&patch), "stela-compilers");
        assert_eq!(slug(&subject(&patch)?), "build-stelas-shader");
        assert_eq!(slug("brw: Add Xe3P (GFX_VERx10 == 350)"), "add-xe3p-gfx");
        Ok(())
    }
}
