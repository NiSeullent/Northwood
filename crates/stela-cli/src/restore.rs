//! Discover signed macOS IPSWs through IPSW.me and download originals from Apple.
pub mod aea;
pub mod archive;
pub mod components;
pub mod disk;
mod history;
pub mod manifest;
pub mod payload;
pub mod remote;
use anyhow::{Context, Result, bail, ensure};
use plist::Value;
use reqwest::{Url, blocking::Client, header, redirect::Policy};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::{File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    time::Duration,
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Image {
    pub version: String,
    pub build: String,
    pub models: Vec<String>,
    pub url: String,
    pub sha1: Option<String>,
}

pub fn apple_url(url: &Url) -> bool {
    url.scheme() == "https"
        && url.host_str().is_some_and(|h| {
            h == "apple.com"
                || h.ends_with(".apple.com")
                || h == "cdn-apple.com"
                || h.ends_with(".cdn-apple.com")
        })
        && url.username().is_empty()
        && url.password().is_none()
}

fn client() -> Result<Client> {
    Ok(Client::builder()
        .user_agent(concat!("stela/", env!("CARGO_PKG_VERSION")))
        .connect_timeout(Duration::from_secs(30))
        .timeout(Duration::from_secs(24 * 60 * 60))
        .redirect(Policy::custom(|attempt| {
            if attempt.previous().len() >= 5 {
                attempt.error("too many redirects")
            } else if !apple_url(attempt.url()) {
                attempt.error("redirect outside Apple's HTTPS servers")
            } else {
                attempt.follow()
            }
        }))
        .build()?)
}

pub fn catalog() -> Result<Vec<Image>> {
    Ok(release_catalog()?.images.clone())
}

pub(crate) fn model_names() -> Result<BTreeMap<String, String>> {
    Ok(release_catalog()?.models.clone())
}

fn release_catalog() -> Result<&'static history::Releases> {
    static CATALOG: std::sync::OnceLock<history::Releases> = std::sync::OnceLock::new();
    if CATALOG.get().is_none() {
        let _ = CATALOG.set(history::load()?);
    }
    Ok(CATALOG.get().expect("a successfully loaded catalog"))
}

fn string(dict: &plist::Dictionary, key: &str) -> Result<String> {
    dict.get(key)
        .and_then(Value::as_string)
        .map(str::to_owned)
        .with_context(|| format!("catalog is missing {key}"))
}

pub fn parse_catalog(bytes: &[u8]) -> Result<Vec<Image>> {
    let value = Value::from_reader(std::io::Cursor::new(bytes))?;
    let versions = value
        .as_dictionary()
        .and_then(|d| d.get("MobileDeviceSoftwareVersionsByVersion"))
        .and_then(Value::as_dictionary)
        .context("unrecognized Apple catalog schema")?;
    let mut images: BTreeMap<(String, String), Image> = BTreeMap::new();
    for version in versions.values() {
        let Some(models) = version
            .as_dictionary()
            .and_then(|v| v.get("MobileDeviceSoftwareVersions"))
            .and_then(Value::as_dictionary)
        else {
            continue;
        };
        for (model, builds) in models {
            let Some(builds) = builds.as_dictionary() else {
                continue;
            };
            for (key, build) in builds {
                let Some(mut build) = build.as_dictionary() else {
                    continue;
                };
                if key == "Unknown" {
                    let Some(universal) = build.get("Universal").and_then(Value::as_dictionary)
                    else {
                        continue;
                    };
                    build = universal;
                }
                let Some(restore) = build.get("Restore").and_then(Value::as_dictionary) else {
                    continue;
                };
                let url = string(restore, "FirmwareURL")?;
                ensure!(
                    apple_url(&Url::parse(&url)?),
                    "catalog supplied a non-Apple HTTPS URL"
                );
                let build = string(restore, "BuildVersion")?;
                let version = string(restore, "ProductVersion")?;
                let sha1 = restore
                    .get("FirmwareSHA1")
                    .and_then(Value::as_string)
                    .map(str::to_lowercase);
                if let Some(digest) = &sha1 {
                    ensure!(
                        digest.len() == 40 && digest.bytes().all(|b| b.is_ascii_hexdigit()),
                        "invalid catalog SHA-1"
                    );
                }
                let entry = images.entry((build.clone(), url.clone())).or_insert(Image {
                    version: version.clone(),
                    build,
                    models: vec![],
                    url,
                    sha1: sha1.clone(),
                });
                ensure!(entry.version == version, "inconsistent catalog version");
                match (&entry.sha1, &sha1) {
                    (Some(existing), Some(new)) => {
                        ensure!(existing == new, "conflicting catalog digests")
                    }
                    (None, Some(_)) => entry.sha1 = sha1,
                    _ => {}
                }
                if !entry.models.contains(model) {
                    entry.models.push(model.clone());
                }
            }
        }
    }
    ensure!(
        !images.is_empty(),
        "Apple catalog contains no restore images"
    );
    let mut result: Vec<_> = images.into_values().collect();
    for image in &mut result {
        image.models.sort();
    }
    Ok(result)
}

pub fn select(images: &[Image], build: Option<&str>, model: Option<&str>) -> Result<Image> {
    let candidates: Vec<_> = images
        .iter()
        .filter(|i| {
            build.is_none_or(|b| b == i.build)
                && model.is_none_or(|m| i.models.iter().any(|v| v == m))
        })
        .collect();
    ensure!(
        !candidates.is_empty(),
        "no matching image; run `stela-dev images list`"
    );
    if build.is_some() {
        ensure!(
            candidates.len() == 1,
            "build has multiple images; specify --model"
        );
        return Ok(candidates[0].clone());
    }
    // Numeric version ordering avoids 27.10 sorting below 27.9.
    fn version(v: &str) -> Vec<u64> {
        v.split('.').map(|p| p.parse().unwrap_or(0)).collect()
    }
    let newest = candidates
        .iter()
        .map(|i| version(&i.version))
        .max()
        .unwrap();
    let latest: Vec<_> = candidates
        .into_iter()
        .filter(|i| version(&i.version) == newest)
        .collect();
    ensure!(
        latest.len() == 1,
        "latest version has multiple builds/images; specify --build and --model"
    );
    Ok(latest[0].clone())
}

#[derive(Debug, Serialize, Deserialize)]
struct Partial {
    image: Image,
    total_bytes: Option<u64>,
    /// The object's strong ETag, else its Last-Modified date, as the first
    /// answer named it: a resumed request sends it as `If-Range`, so a
    /// changed object answers whole and the download restarts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    validator: Option<String>,
}

/// The validator [`Partial`] keeps from an answer's headers.
fn validator(headers: &header::HeaderMap) -> Option<String> {
    let value = |name| {
        headers
            .get(name)
            .and_then(|value: &header::HeaderValue| value.to_str().ok())
            .map(str::to_owned)
    };
    value(header::ETAG)
        .filter(|etag| !etag.starts_with("W/"))
        .or_else(|| value(header::LAST_MODIFIED))
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Receipt {
    pub schema_version: u32,
    pub image: Image,
    pub bytes: u64,
    pub sha256: String,
    pub catalog_sha1_verified: bool,
}

fn sidecar(path: &Path, suffix: &str) -> PathBuf {
    let mut value = path.as_os_str().to_owned();
    value.push(suffix);
    value.into()
}

/// The download receipt beside an image (`IMAGE.json`), when there is one;
/// it must name the image's own size.
pub fn receipt(image: &Path) -> Result<Option<Receipt>> {
    let path = sidecar(image, ".json");
    if !path.is_file() {
        return Ok(None);
    }
    let receipt: Receipt = stela_engine::read_json(&path)
        .with_context(|| format!("{} is not a download receipt", path.display()))?;
    let bytes = std::fs::metadata(image)?.len();
    ensure!(
        bytes == receipt.bytes,
        "{} holds {bytes} bytes, but its download receipt {} names {}; download it again",
        image.display(),
        path.display(),
        receipt.bytes
    );
    Ok(Some(receipt))
}

/// A Content-Range must cover exactly the remaining suffix of the image.
fn range_total(value: &str, offset: u64) -> Result<u64> {
    let (span, total) = value
        .strip_prefix("bytes ")
        .context("invalid Content-Range unit")?
        .split_once('/')
        .context("invalid Content-Range")?;
    let (start, end) = span.split_once('-').context("invalid Content-Range span")?;
    let (start, end, total): (u64, u64, u64) = (start.parse()?, end.parse()?, total.parse()?);
    ensure!(
        start == offset && end >= start && end.checked_add(1) == Some(total),
        "incorrect or incomplete resumed range"
    );
    Ok(total)
}

/// Copy `image`'s recovery members for `model` from Apple's servers by
/// byte ranges into the archive `output` ([`payload::trim`]). Returns the
/// bytes written for the members.
pub fn fetch_recovery(image: &Image, model: &str, output: &Path) -> Result<u64> {
    let url = Url::parse(&image.url)?;
    ensure!(
        apple_url(&url),
        "image must be hosted on Apple's HTTPS servers"
    );
    payload::trim(
        remote::Remote::open(client()?, &image.url)?,
        model,
        output,
        "Downloading boot files",
    )
}

/// The bytes [`fetch_recovery`] would copy of `image` for `model`, and the
/// whole image's, from its ZIP directory read by byte ranges.
pub fn recovery_size(image: &Image, model: &str) -> Result<(u64, u64)> {
    let url = Url::parse(&image.url)?;
    ensure!(
        apple_url(&url),
        "image must be hosted on Apple's HTTPS servers"
    );
    payload::recovery_bytes(remote::Remote::open(client()?, &image.url)?, model)
}

/// The restore inventory of `image` for `model` that `stela.efi` streams
/// from Apple's servers during the install ([`payload::remote_inventory`]).
/// It is read here over HTTPS and records the plain-HTTP twin the mount,
/// which has no TLS, reads, once that twin serves the same object.
pub fn remote_inventory(
    image: &Image,
    model: &str,
    keys_in: Option<&Path>,
) -> Result<payload::Inventory> {
    let url = Url::parse(&image.url)?;
    ensure!(
        apple_url(&url),
        "image must be hosted on Apple's HTTPS servers"
    );
    let http = plain_http(&url)?;
    let source = remote::Remote::open(client()?, &image.url)?;
    let twin_client = Client::builder()
        .user_agent(concat!("stela/", env!("CARGO_PKG_VERSION")))
        .connect_timeout(Duration::from_secs(30))
        .timeout(Duration::from_secs(60))
        .redirect(Policy::none())
        .build()?;
    let twin = remote::Remote::open(twin_client, &http)?;
    ensure!(
        (twin.etag(), twin.size()) == (source.etag(), source.size()),
        "{http} serves another object than {}",
        image.url
    );
    let layout = stela_formats::payload::Layout::Remote {
        url: http,
        etag: source.etag().into(),
    };
    let name = url
        .path_segments()
        .and_then(|mut segments| segments.next_back())
        .context("the image URL names no file")?;
    let bytes = source.size();
    payload::remote_inventory(
        source,
        payload::RemoteInventoryOptions {
            image: Path::new(name),
            bytes,
            layout,
            model,
            keys_in,
        },
    )
}

/// Apple's update CDN serves each object over plain HTTP from its `-http`
/// host as well.
fn plain_http(url: &Url) -> Result<String> {
    ensure!(
        url.host_str() == Some("updates.cdn-apple.com"),
        "{url} is not on Apple's update CDN, the one that also serves plain HTTP"
    );
    let mut http = url.clone();
    http.set_scheme("http")
        .map_err(|()| anyhow::anyhow!("{url} has no plain-HTTP form"))?;
    http.set_host(Some("updates-http.cdn-apple.com"))?;
    Ok(http.into())
}

pub fn download(image: &Image, output: &Path) -> Result<Receipt> {
    let url = Url::parse(&image.url)?;
    ensure!(
        apple_url(&url),
        "image must be hosted on Apple's HTTPS servers"
    );
    download_with_client(image, output, &client()?)
}

pub(crate) fn open_image(image: &Image) -> Result<remote::Remote> {
    ensure!(
        apple_url(&Url::parse(&image.url)?),
        "image must be hosted on Apple's HTTPS servers"
    );
    remote::Remote::open(client()?, &image.url)
}

fn download_with_client(image: &Image, output: &Path, client: &Client) -> Result<Receipt> {
    ensure!(
        !output.exists() && !sidecar(output, ".json").exists(),
        "output or receipt already exists; choose another path"
    );
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let part = sidecar(output, ".part");
    let metadata = sidecar(output, ".part.json");
    let mut offset = 0;
    let mut previous_total = None;
    let mut previous_validator = None;
    if part.exists() {
        // A partial download resumes only as the same catalog entry and
        // within its size; any other is removed and the download restarts.
        let resumable = stela_engine::read_json::<Partial>(&metadata)
            .ok()
            .filter(|previous| previous.image == *image)
            .and_then(|previous| {
                let bytes = part.metadata().ok()?.len();
                previous
                    .total_bytes
                    .is_none_or(|total| bytes <= total)
                    .then_some((bytes, previous.total_bytes, previous.validator))
            });
        match resumable {
            Some((bytes, total, validator)) => {
                (offset, previous_total, previous_validator) = (bytes, total, validator)
            }
            None => remove_partial(&part, &metadata)?,
        }
        if offset > 0 && previous_total == Some(offset) {
            // Recover interruption after the last byte but before publishing.
            return publish(image, output, &part, &metadata, offset);
        }
    }
    let mut response = crate::net::send("downloading macOS", || {
        let request = client
            .get(&image.url)
            .header(header::ACCEPT_ENCODING, "identity");
        match (offset, &previous_validator) {
            (0, _) => request,
            (_, None) => request.header(header::RANGE, format!("bytes={offset}-")),
            (_, Some(validator)) => request
                .header(header::RANGE, format!("bytes={offset}-"))
                .header(header::IF_RANGE, validator),
        }
    })?
    .error_for_status()?;
    let answered = validator(response.headers());
    let total;
    if response.status() == reqwest::StatusCode::PARTIAL_CONTENT {
        total = Some(range_total(
            response
                .headers()
                .get(header::CONTENT_RANGE)
                .context("missing Content-Range")?
                .to_str()?,
            offset,
        )?);
        if (previous_total.is_some() && total != previous_total)
            || (previous_validator.is_some()
                && answered.is_some()
                && answered != previous_validator)
        {
            // Another object than the partial's, though the server sent a
            // range of it: start over from its first byte.
            drop(response);
            remove_partial(&part, &metadata)?;
            eprintln!("note: the image changed on Apple's server; downloading it anew");
            return download_with_client(image, output, client);
        }
        if let Some(length) = response.content_length() {
            ensure!(
                length.checked_add(offset) == total,
                "range length disagrees with Content-Length"
            );
        }
    } else if response.status() == reqwest::StatusCode::OK {
        offset = 0; // Server ignored Range: safely restart, never append a full image.
        total = response.content_length();
    } else {
        bail!("unexpected download status: {}", response.status());
    }
    if let Some(total) = total {
        let directory = output.parent().unwrap_or(Path::new("."));
        crate::setup::ensure_room(directory, "Downloading macOS", total - offset)?;
    }
    stela_engine::write_json(
        &metadata,
        &Partial {
            image: image.clone(),
            total_bytes: total,
            validator: answered,
        },
    )?;
    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(offset == 0)
        .append(offset > 0)
        .open(&part)?;
    let mut written = offset;
    let progress = total
        .map(|total| {
            let progress = crate::progress::Phase::bytes("Downloading macOS", total);
            progress.position(offset);
            progress
        })
        .unwrap_or_else(|| crate::progress::Phase::spinner("Downloading macOS"));
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let n = response.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        file.write_all(&buffer[..n])?;
        written += n as u64;
        progress.advance(n as u64);
        ensure!(
            total.is_none_or(|t| written <= t),
            "response exceeds advertised size"
        );
    }
    file.sync_all()?;
    drop(file);
    ensure!(
        written > 0 && total.is_none_or(|t| t == written),
        "incomplete image; rerun to resume"
    );
    progress.finish();
    publish(image, output, &part, &metadata, written)
}

/// Remove a partial download and its metadata, as far as they exist.
fn remove_partial(part: &Path, metadata: &Path) -> Result<()> {
    for path in [part, metadata] {
        match std::fs::remove_file(path) {
            Err(error) if error.kind() != std::io::ErrorKind::NotFound => {
                return Err(error).with_context(|| format!("cannot remove {}", path.display()));
            }
            _ => {}
        }
    }
    Ok(())
}

fn publish(
    image: &Image,
    output: &Path,
    part: &Path,
    metadata: &Path,
    written: u64,
) -> Result<Receipt> {
    let mut file = File::open(part)?;
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut sha256 = Sha256::new();
    let mut sha1 = sha1::Sha1::new();
    let progress = crate::progress::Phase::bytes("Verifying downloaded macOS", written);
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        sha256.update(&buffer[..n]);
        sha1.update(&buffer[..n]);
        progress.advance(n as u64);
    }
    if let Some(expected) = &image.sha1
        && hex::encode(sha1.finalize()) != *expected
    {
        remove_partial(part, metadata)?;
        bail!(
            "the downloaded macOS image does not match Apple's checksum and was removed; run \
             setup again to download it anew"
        );
    }
    let receipt = Receipt {
        schema_version: 1,
        image: image.clone(),
        bytes: written,
        sha256: hex::encode(sha256.finalize()),
        catalog_sha1_verified: image.sha1.is_some(),
    };
    // Do not clobber an output created while the download was running.
    std::fs::hard_link(part, output).context("publish completed image")?;
    stela_engine::write_json(&sidecar(output, ".json"), &receipt)?;
    std::fs::remove_file(part)?;
    std::fs::remove_file(metadata)?;
    progress.finish();
    Ok(receipt)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn validates_hosts_and_ranges() {
        assert!(apple_url(
            &Url::parse("https://updates.cdn-apple.com/file").unwrap()
        ));
        assert!(!apple_url(
            &Url::parse("https://apple.com.evil.test/file").unwrap()
        ));
        assert!(!apple_url(
            &Url::parse("http://updates.cdn-apple.com/file").unwrap()
        ));
        assert_eq!(range_total("bytes 5-9/10", 5).unwrap(), 10);
        assert!(range_total("bytes 0-9/10", 5).is_err());
        assert!(range_total("bytes 5-8/10", 5).is_err());
    }
    #[test]
    fn receipt_requires_image_size() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let image = dir.path().join("Restore.ipsw");
        std::fs::write(&image, b"ipsw")?;
        assert!(receipt(&image)?.is_none());
        let written = Receipt {
            schema_version: 1,
            image: Image {
                version: "27.0.1".into(),
                build: "26A434".into(),
                models: vec!["Mac14,3".into()],
                url: "https://updates.cdn-apple.com/Restore.ipsw".into(),
                sha1: Some("0".repeat(40)),
            },
            bytes: 4,
            sha256: "0".repeat(64),
            catalog_sha1_verified: true,
        };
        stela_engine::write_json(&sidecar(&image, ".json"), &written)?;
        assert_eq!(receipt(&image)?.unwrap().image, written.image);
        std::fs::write(&image, b"ipsw!")?;
        let error = receipt(&image).unwrap_err().to_string();
        assert!(error.contains("names 4"), "{error}");
        Ok(())
    }
    #[test]
    fn parses_real_feed_structure_and_unknown_universal() -> Result<()> {
        let bytes = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/catalog.plist"
        ));
        let images = parse_catalog(bytes)?;
        assert_eq!(images.len(), 2);
        assert_eq!(
            select(&images, None, Some("VirtualMac2,1"))?.version,
            "27.10"
        );
        assert_eq!(select(&images, Some("TEST1"), None)?.models.len(), 2);
        assert_eq!(
            select(&images, Some("TEST1"), None)?.sha1,
            Some("1".repeat(40))
        );
        assert!(parse_catalog(b"invalid").is_err());
        Ok(())
    }
    fn serve(reply: String) -> (String, std::thread::JoinHandle<String>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let thread = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = vec![];
            let mut byte = [0];
            while !request.ends_with(b"\r\n\r\n") {
                stream.read_exact(&mut byte).unwrap();
                request.push(byte[0]);
            }
            stream.write_all(reply.as_bytes()).unwrap();
            String::from_utf8(request).unwrap()
        });
        (format!("http://{address}/fixture.ipsw"), thread)
    }
    #[test]
    fn resumes_and_verifies_image() -> Result<()> {
        let (url, server) = serve("HTTP/1.1 206 Partial Content\r\nContent-Length: 5\r\nContent-Range: bytes 5-9/10\r\nConnection: close\r\n\r\n56789".into());
        let image = Image {
            version: "test".into(),
            build: "TEST".into(),
            models: vec![],
            url,
            sha1: Some(hex::encode(sha1::Sha1::digest(b"0123456789"))),
        };
        let dir = tempfile::tempdir()?;
        let out = dir.path().join("test.ipsw");
        std::fs::write(sidecar(&out, ".part"), b"01234")?;
        stela_engine::write_json(
            &sidecar(&out, ".part.json"),
            &Partial {
                image: image.clone(),
                total_bytes: Some(10),
                validator: None,
            },
        )?;
        let receipt = download_with_client(&image, &out, &Client::new())?;
        assert_eq!(std::fs::read(&out)?, b"0123456789");
        assert!(receipt.catalog_sha1_verified);
        assert!(
            server
                .join()
                .unwrap()
                .to_lowercase()
                .contains("range: bytes=5-")
        );
        Ok(())
    }

    #[test]
    fn completed_partial_is_published_without_another_request() -> Result<()> {
        let image = Image {
            version: "test".into(),
            build: "TEST".into(),
            models: vec![],
            url: "http://127.0.0.1:1/unused".into(),
            sha1: Some(hex::encode(sha1::Sha1::digest(b"complete"))),
        };
        let dir = tempfile::tempdir()?;
        let out = dir.path().join("test.ipsw");
        std::fs::write(sidecar(&out, ".part"), b"complete")?;
        stela_engine::write_json(
            &sidecar(&out, ".part.json"),
            &Partial {
                image: image.clone(),
                total_bytes: Some(8),
                validator: None,
            },
        )?;
        let receipt = download_with_client(&image, &out, &Client::new())?;
        assert_eq!(receipt.bytes, 8);
        assert_eq!(std::fs::read(out)?, b"complete");
        Ok(())
    }
    #[test]
    fn a_partial_of_another_catalog_entry_is_removed_and_the_download_restarts() -> Result<()> {
        let (url, server) =
            serve("HTTP/1.1 200 OK\r\nContent-Length: 3\r\nConnection: close\r\n\r\nabc".into());
        let image = Image {
            version: "test".into(),
            build: "TEST".into(),
            models: vec![],
            url,
            sha1: Some(hex::encode(sha1::Sha1::digest(b"abc"))),
        };
        let dir = tempfile::tempdir()?;
        let out = dir.path().join("test.ipsw");
        std::fs::write(sidecar(&out, ".part"), b"0123456789")?;
        stela_engine::write_json(
            &sidecar(&out, ".part.json"),
            &Partial {
                image: Image {
                    build: "OTHER".into(),
                    ..image.clone()
                },
                total_bytes: Some(10),
                validator: None,
            },
        )?;
        download_with_client(&image, &out, &Client::new())?;
        assert_eq!(std::fs::read(&out)?, b"abc");
        assert!(!sidecar(&out, ".part").exists());
        // The restart asks for the whole image.
        assert!(!server.join().unwrap().to_lowercase().contains("range:"));
        Ok(())
    }

    #[test]
    fn ignored_range_restarts_and_bad_digest_never_publishes() -> Result<()> {
        for bad in [false, true] {
            let (url, server) = serve(
                "HTTP/1.1 200 OK\r\nContent-Length: 3\r\nConnection: close\r\n\r\nabc".into(),
            );
            let image = Image {
                version: "test".into(),
                build: "TEST".into(),
                models: vec![],
                url,
                sha1: bad.then(|| "0".repeat(40)),
            };
            let dir = tempfile::tempdir()?;
            let out = dir.path().join("test.ipsw");
            std::fs::write(sidecar(&out, ".part"), b"old")?;
            stela_engine::write_json(
                &sidecar(&out, ".part.json"),
                &Partial {
                    image: image.clone(),
                    total_bytes: None,
                    validator: None,
                },
            )?;
            let result = download_with_client(&image, &out, &Client::new());
            if bad {
                let error = result.unwrap_err().to_string();
                assert!(error.contains("was removed"), "{error}");
                assert!(!out.exists());
                assert!(!sidecar(&out, ".part").exists());
                assert!(!sidecar(&out, ".part.json").exists());
            } else {
                result?;
                assert_eq!(std::fs::read(out)?, b"abc");
            }
            server.join().unwrap();
        }
        Ok(())
    }

    #[test]
    fn a_resumed_request_names_the_object_and_a_changed_one_restarts() -> Result<()> {
        // The object changed: the server answers `If-Range` with all of it.
        let (url, server) = serve(
            "HTTP/1.1 200 OK\r\nETag: \"new\"\r\nContent-Length: 3\r\nConnection: close\r\n\r\nabc"
                .into(),
        );
        let image = Image {
            version: "test".into(),
            build: "TEST".into(),
            models: vec![],
            url,
            sha1: Some(hex::encode(sha1::Sha1::digest(b"abc"))),
        };
        let dir = tempfile::tempdir()?;
        let out = dir.path().join("test.ipsw");
        std::fs::write(sidecar(&out, ".part"), b"01")?;
        stela_engine::write_json(
            &sidecar(&out, ".part.json"),
            &Partial {
                image: image.clone(),
                total_bytes: Some(10),
                validator: Some("\"old\"".into()),
            },
        )?;
        download_with_client(&image, &out, &Client::new())?;
        assert_eq!(std::fs::read(&out)?, b"abc");
        let request = server.join().unwrap().to_lowercase();
        assert!(request.contains("range: bytes=2-"), "{request}");
        assert!(request.contains("if-range: \"old\""), "{request}");
        // The answer's validator is kept for the next resume, a strong
        // ETag before Last-Modified.
        let mut headers = header::HeaderMap::new();
        headers.insert(
            header::LAST_MODIFIED,
            "Tue, 10 Nov 2026 00:00:00 GMT".parse()?,
        );
        assert_eq!(
            validator(&headers).as_deref(),
            Some("Tue, 10 Nov 2026 00:00:00 GMT")
        );
        headers.insert(header::ETAG, "W/\"weak\"".parse()?);
        assert_eq!(
            validator(&headers).as_deref(),
            Some("Tue, 10 Nov 2026 00:00:00 GMT")
        );
        headers.insert(header::ETAG, "\"strong\"".parse()?);
        assert_eq!(validator(&headers).as_deref(), Some("\"strong\""));
        Ok(())
    }
}
