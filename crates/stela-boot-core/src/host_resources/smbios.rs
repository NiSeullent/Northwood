// SPDX-License-Identifier: GPL-2.0-or-later
//! Pre-EBS physical-host SMBIOS source acquisition only. UEFI 2.10 §2.3.4
//! grants identity mappings only for described memory; §4.6 config GUIDs and
//! DSP0134 §5.2 authorize reading their published tables. Never use this after
//! root replacement/EBS, or for undescribed/reserved/MMIO/protected memory.
use crate::handoff_records::Region;
use alloc::vec::Vec;
use stela_crypto::Sha256;
use stela_crypto::digest::Digest;
use stela_drivers::firmware::smbios::{self, Error, Format, MAX_ENTRY_BYTES};
use uefi_raw::table::boot::MemoryType;

use stela_contracts::bytes::HOST_PAGE_BYTES as PAGE;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Stage {
    Select,
    EntryPrefix,
    Entry,
    Table,
    Validate,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Failure {
    pub stage: Stage,
    pub reason: Error,
    pub status: Option<u64>,
    pub address: u64,
    pub bytes: u64,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum State {
    NotCaptured,
    Unavailable,
    Failed(Failure),
    Present,
}
#[derive(Clone, Copy)]
pub struct Capture {
    pub state: State,
    pub region: Region,
    pub format: Option<Format>,
    pub source_address: u64,
    pub sha256: [u8; 32],
    pub entry_sha256: [u8; 32],
    pub table_sha256: [u8; 32],
}
impl Capture {
    pub const fn empty() -> Self {
        Self {
            state: State::NotCaptured,
            region: Region {
                physical: 0,
                bytes: 0,
            },
            format: None,
            source_address: 0,
            sha256: [0; 32],
            entry_sha256: [0; 32],
            table_sha256: [0; 32],
        }
    }
}
pub fn failure(stage: Stage, reason: Error, address: u64, bytes: usize) -> Failure {
    Failure {
        stage,
        reason,
        status: None,
        address,
        bytes: bytes as u64,
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Selection {
    primary: (Format, u64),
    legacy: Option<u64>,
}

pub fn select(entries: impl Iterator<Item = (Format, u64)>) -> Result<Option<Selection>, Error> {
    let mut old = None;
    let mut modern = None;
    for (format, address) in entries {
        let slot = match format {
            Format::Smbios2 => &mut old,
            Format::Smbios3 => &mut modern,
        };
        if slot.replace(address).is_some() {
            return Err(Error::AmbiguousConfiguration);
        }
    }
    Ok(modern
        .map(|address| Selection {
            primary: (Format::Smbios3, address),
            legacy: old,
        })
        .or_else(|| {
            old.map(|address| Selection {
                primary: (Format::Smbios2, address),
                legacy: None,
            })
        }))
}
#[derive(Clone, Copy)]
pub struct Span {
    pub start: u64,
    pub pages: u64,
    pub kind: MemoryType,
    pub attributes: u64,
}
/// Prove full source extent against a fresh firmware memory map. No
/// address is converted to a pointer until this grant succeeds. Only published
/// data/ACPI RAM is eligible; overlapping descriptors are ambiguous, not grants.
pub fn grant(
    spans: impl Iterator<Item = Span> + Clone,
    address: u64,
    bytes: usize,
    linear_bits: u32,
) -> Result<(), Error> {
    let end = address
        .checked_add(bytes as u64)
        .ok_or(Error::AddressOverflow)?;
    if address == 0 || bytes == 0 {
        return Err(Error::UnreadableRange);
    }
    if !matches!(linear_bits, 47 | 56) || end > (1u64 << linear_bits) {
        return Err(Error::NoncanonicalAddress);
    }
    let mut cursor = address;
    while cursor < end {
        let mut covered = None;
        for span in spans.clone() {
            let top = span
                .pages
                .checked_mul(PAGE)
                .and_then(|size| span.start.checked_add(size))
                .ok_or(Error::AddressOverflow)?;
            if span.start <= cursor && cursor < top {
                if covered.is_some() {
                    return Err(Error::UnreadableRange);
                }
                if !matches!(
                    span.kind,
                    MemoryType::BOOT_SERVICES_DATA
                        | MemoryType::RUNTIME_SERVICES_DATA
                        | MemoryType::ACPI_RECLAIM
                        | MemoryType::ACPI_NON_VOLATILE
                ) || span.attributes & 0x2000 != 0
                {
                    return Err(Error::UnreadableRange);
                }
                covered = Some(top.min(end));
            }
        }
        let top = covered.ok_or(Error::UnreadableRange)?;
        if spans
            .clone()
            .any(|span| span.start > cursor && span.start < top)
        {
            return Err(Error::UnreadableRange);
        }
        cursor = top;
    }
    Ok(())
}
pub fn capture_with(
    selected: Result<Option<Selection>, Error>,
    mut read: impl FnMut(u64, &mut [u8], Stage) -> Result<(), Failure>,
    allocate: impl FnOnce(usize) -> Result<&'static mut [u8], Failure>,
) -> Capture {
    let mut capture = Capture::empty();
    let Selection { primary, legacy } = match selected {
        Ok(Some(selected)) => selected,
        Ok(None) => {
            capture.state = State::Unavailable;
            return capture;
        }
        Err(error) => {
            capture.state = State::Failed(failure(Stage::Select, error, 0, 0));
            return capture;
        }
    };
    let (mut format, mut address) = primary;
    let mut source = read_source(format, address, &mut read);
    // Some firmware publishes a modern entry whose table is in unreadable
    // legacy memory. A separately published 2.x entry can supply the data
    // under the same memory-map and format checks. Never scan low memory.
    if let Err(Failure {
        stage: Stage::Table,
        reason: Error::UnreadableRange,
        ..
    }) = &source
        && let Some(legacy) = legacy
        && let Ok(alternate) = read_source(Format::Smbios2, legacy, &mut read)
    {
        format = Format::Smbios2;
        address = legacy;
        source = Ok(alternate);
    }
    capture.format = Some(format);
    capture.source_address = address;
    let result = source.and_then(|(entry, bytes)| {
        // Reserve permanent handoff memory only for a complete, valid source.
        let output = allocate(bytes.len())?;
        if output.len() != bytes.len() {
            return Err(failure(
                Stage::Table,
                Error::AllocationFailed,
                entry.table_address,
                entry.table_bytes,
            ));
        }
        output.copy_from_slice(&bytes);
        let length = entry.entry_bytes;
        capture.sha256 = Sha256::digest(&*output).into();
        capture.entry_sha256 = Sha256::digest(&output[..length]).into();
        capture.table_sha256 = Sha256::digest(&output[length..]).into();
        capture.region = Region {
            physical: output.as_ptr() as u64,
            bytes: output.len() as u64,
        };
        Ok(())
    });
    capture.state = match result {
        Ok(()) => State::Present,
        Err(error) => State::Failed(error),
    };
    capture
}

fn read_source(
    format: Format,
    address: u64,
    read: &mut impl FnMut(u64, &mut [u8], Stage) -> Result<(), Failure>,
) -> Result<(smbios::Entry, Vec<u8>), Failure> {
    let mut prefix = [0; 8];
    read(address, &mut prefix, Stage::EntryPrefix)?;
    let length = smbios::entry_length(&prefix, format)
        .map_err(|error| failure(Stage::Entry, error, address, prefix.len()))?;
    let mut raw = [0; MAX_ENTRY_BYTES];
    read(address, &mut raw[..length], Stage::Entry)?;
    if raw[..8] != prefix {
        return Err(failure(Stage::Entry, Error::ChangedEntry, address, length));
    }
    let entry = smbios::entry(&raw[..length], format).map_err(|error| {
        let reported_bytes = match error {
            Error::TableBound { bytes } => bytes,
            _ => length,
        };
        failure(Stage::Entry, error, address, reported_bytes)
    })?;
    let mut bytes = Vec::new();
    let total = length + entry.table_bytes;
    bytes.try_reserve_exact(total).map_err(|_| {
        failure(
            Stage::Table,
            Error::AllocationFailed,
            entry.table_address,
            entry.table_bytes,
        )
    })?;
    bytes.extend_from_slice(&raw[..length]);
    bytes.resize(total, 0);
    read(entry.table_address, &mut bytes[length..], Stage::Table)?;
    smbios::identity(entry, &bytes[length..]).map_err(|error| {
        failure(
            Stage::Validate,
            error,
            entry.table_address,
            entry.table_bytes,
        )
    })?;
    Ok((entry, bytes))
}
pub fn record(
    phase: &str,
    capture: &Capture,
    identity: Option<smbios::SystemIdentity<'_>>,
) -> serde_json::Value {
    let hex = |bytes: &[u8]| -> alloc::string::String {
        bytes.iter().map(|b| alloc::format!("{b:02x}")).collect()
    };
    let field = |value: smbios::Observation<'_>| match value {
        smbios::Observation::NotObserved => serde_json::json!({"state":"not-observed"}),
        smbios::Observation::Absent => serde_json::json!({"state":"absent"}),
        smbios::Observation::Value(bytes) => {
            serde_json::json!({"state":"value","bytes":bytes.len(),"sha256":hex(&Sha256::digest(bytes))})
        }
    };
    let state = match capture.state {
        State::NotCaptured => serde_json::json!({"state":"not-captured"}),
        State::Unavailable => {
            serde_json::json!({"state":"unavailable","reason":"no-configuration-table"})
        }
        State::Failed(error) => {
            serde_json::json!({"state":"failed","stage":alloc::format!("{:?}",error.stage),"reason":alloc::format!("{:?}",error.reason),"status":error.status,"source_address":error.address,"reported_bytes":error.bytes})
        }
        State::Present => {
            serde_json::json!({"state":"present","bytes":capture.region.bytes,"sha256":hex(&capture.sha256),"entry_sha256":hex(&capture.entry_sha256),"table_sha256":hex(&capture.table_sha256)})
        }
    };
    let identity = identity.unwrap_or_default();
    serde_json::json!({"phase":phase,"source_guid":capture.format.map(Format::guid),"source_entry_address":capture.source_address,"resource":if capture.state==State::Present {Some(serde_json::json!({"name":smbios::NAME,"target":{"driver":smbios::TARGET.driver,"hardware":smbios::TARGET.hardware,"revision":smbios::TARGET.revision,"board":smbios::TARGET.board},"source":capture.format.map(Format::source)}))}else{None},"observation":state,"system_vendor":field(identity.system_vendor),"product_name":field(identity.product_name),"product_version":field(identity.product_version),"board_vendor":field(identity.board_vendor)})
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{boxed::Box, vec, vec::Vec};
    use stela_drivers::firmware::smbios::MAX_TABLE_BYTES;
    fn fixtures() -> (Vec<u8>, Vec<u8>) {
        let mut table = vec![1, 8, 1, 0, 1, 2, 3, 0];
        table.extend_from_slice(b"SYNTH-VENDOR\0SYNTH-PRODUCT\0SYNTH-VERSION\0\0");
        table.extend_from_slice(&[127, 4, 255, 255, 0, 0]);
        let mut entry = vec![0; 24];
        entry[..5].copy_from_slice(b"_SM3_");
        entry[6] = 24;
        entry[7] = 3;
        entry[8] = 8;
        entry[10] = 1;
        entry[12..16].copy_from_slice(&(table.len() as u32).to_le_bytes());
        entry[16..24].copy_from_slice(&0x2000u64.to_le_bytes());
        entry[5] = 0u8.wrapping_sub(entry.iter().fold(0u8, |a, b| a.wrapping_add(*b)));
        (entry, table)
    }
    fn span(start: u64, pages: u64) -> Span {
        Span {
            start,
            pages,
            kind: MemoryType::RUNTIME_SERVICES_DATA,
            attributes: 0,
        }
    }
    fn legacy_entry(table: &[u8]) -> [u8; 31] {
        let mut entry = [0; 31];
        entry[..4].copy_from_slice(b"_SM_");
        entry[5..8].copy_from_slice(&[31, 2, 8]);
        entry[8..10].copy_from_slice(&(table.len() as u16).to_le_bytes());
        entry[16..21].copy_from_slice(b"_DMI_");
        entry[22..24].copy_from_slice(&(table.len() as u16).to_le_bytes());
        entry[24..28].copy_from_slice(&0x4000u32.to_le_bytes());
        entry[28..30].copy_from_slice(&2u16.to_le_bytes());
        entry[21] = 0u8.wrapping_sub(entry[16..].iter().fold(0u8, |a, b| a.wrapping_add(*b)));
        entry[4] = 0u8.wrapping_sub(entry.iter().fold(0u8, |a, b| a.wrapping_add(*b)));
        entry
    }
    #[test]
    fn read_grants_require_full_unambiguous_readable_canonical_data_ram() {
        assert_eq!(
            grant(
                [span(0x1000, 1), span(0x2000, 1)].into_iter(),
                0x1fff,
                2,
                47
            ),
            Ok(())
        );
        for kind in [
            MemoryType::RESERVED,
            MemoryType::MMIO,
            MemoryType::CONVENTIONAL,
            MemoryType::UNUSABLE,
            MemoryType::BOOT_SERVICES_CODE,
            MemoryType::LOADER_DATA,
        ] {
            let mut s = span(0x1000, 1);
            s.kind = kind;
            assert_eq!(
                grant([s].into_iter(), 0x1000, 1, 47),
                Err(Error::UnreadableRange)
            );
        }
        let mut s = span(0x1000, 1);
        s.attributes = 0x2000;
        assert_eq!(
            grant([s].into_iter(), 0x1000, 1, 47),
            Err(Error::UnreadableRange)
        );
        assert_eq!(
            grant([span(0x1000, 1)].into_iter(), 0x1fff, 2, 47),
            Err(Error::UnreadableRange)
        );
        assert_eq!(
            grant(
                [span(0x1000, 2), span(0x2000, 1)].into_iter(),
                0x1000,
                8192,
                47
            ),
            Err(Error::UnreadableRange)
        );
        assert_eq!(
            grant([span(0x1000, u64::MAX)].into_iter(), 0x1000, 1, 47),
            Err(Error::AddressOverflow)
        );
        assert_eq!(
            grant([].into_iter(), u64::MAX, 2, 47),
            Err(Error::AddressOverflow)
        );
        assert_eq!(
            grant([].into_iter(), 1 << 47, 1, 47),
            Err(Error::NoncanonicalAddress)
        );
        assert_eq!(
            grant([span(1 << 48, 1)].into_iter(), 1 << 48, 1, 56),
            Ok(())
        );
    }
    #[test]
    fn published_selection_prefers_modern_and_rejects_ambiguity() {
        assert_eq!(select([].into_iter()), Ok(None));
        assert_eq!(
            select([(Format::Smbios2, 1), (Format::Smbios3, 2)].into_iter()),
            Ok(Some(Selection {
                primary: (Format::Smbios3, 2),
                legacy: Some(1),
            }))
        );
        assert_eq!(
            select([(Format::Smbios3, 1), (Format::Smbios3, 2)].into_iter()),
            Err(Error::AmbiguousConfiguration)
        );
        let c = capture_with(
            Ok(None),
            |_, _, _| panic!("unavailable must not read"),
            |_| panic!("unavailable must not allocate"),
        );
        assert_eq!(c.state, State::Unavailable);
        let json = record("before-ebs", &c, None);
        assert_eq!(json["system_vendor"]["state"], "not-observed");
        assert_eq!(json["product_name"]["state"], "not-observed");
        assert_eq!(json["product_version"]["state"], "not-observed");
        let c = capture_with(
            select([(Format::Smbios3, 1)].into_iter()),
            |address, _, stage| {
                Err(Failure {
                    status: Some(0x8000_0000_0000_0007),
                    ..failure(stage, Error::UnreadableRange, address, 8)
                })
            },
            |_| panic!(),
        );
        assert!(matches!(
            c.state,
            State::Failed(Failure {
                stage: Stage::EntryPrefix,
                status: Some(0x8000_0000_0000_0007),
                ..
            })
        ));
        assert_eq!(c.region.bytes, 0);
    }
    #[test]
    fn source_capture_copies_once_validates_before_publication_and_logs_no_strings() {
        let (entry, table) = fixtures();
        let mut calls = 0;
        let mut allocations = 0;
        let c = capture_with(
            select([(Format::Smbios3, 0x1000), (Format::Smbios2, 0x3000)].into_iter()),
            |address, output, _| {
                calls += 1;
                let source = if address == 0x1000 {
                    &entry
                } else {
                    assert_eq!(address, 0x2000);
                    &table
                };
                output.copy_from_slice(&source[..output.len()]);
                Ok(())
            },
            |bytes| {
                allocations += 1;
                Ok(Box::leak(vec![0; bytes].into_boxed_slice()))
            },
        );
        assert_eq!(calls, 3);
        assert_eq!(allocations, 1);
        assert_eq!(c.state, State::Present);
        // SAFETY: pure fixture allocation is leaked and immutable after capture.
        let bytes = unsafe {
            core::slice::from_raw_parts(c.region.physical as *const u8, c.region.bytes as usize)
        };
        assert_eq!(&bytes[..entry.len()], entry);
        assert_eq!(&bytes[entry.len()..], table);
        let mut captured = smbios::Captured::new(bytes).unwrap();
        let catalog = stela_drivers::firmware::Catalog::from_persistent_handoff(vec![
            captured.take_resource().unwrap(),
        ])
        .unwrap();
        let snapshot = smbios::load(&catalog).unwrap();
        assert_eq!(snapshot.image.bytes().as_ptr(), bytes.as_ptr());
        let json = record("after-ebs", &c, Some(snapshot.identity));
        let text = json.to_string();
        assert!(!text.contains("SYNTH-VENDOR"));
        assert!(!text.contains("SYNTH-PRODUCT"));
        assert!(!text.contains("SYNTH-VERSION"));
        assert_eq!(json["system_vendor"]["state"], "value");
    }
    #[test]
    fn unreadable_ami_table_recovers_only_with_an_independent_valid_entry() {
        let (mut modern, table) = fixtures();
        let reported_address = 0x000e_d9b0u64;
        let reported_bytes = 5530u32;
        modern[12..16].copy_from_slice(&reported_bytes.to_le_bytes());
        modern[16..24].copy_from_slice(&reported_address.to_le_bytes());
        modern[5] = 0;
        modern[5] = 0u8.wrapping_sub(modern.iter().fold(0u8, |a, b| a.wrapping_add(*b)));
        let legacy = legacy_entry(&table);
        let primary_failure = failure(
            Stage::Table,
            Error::UnreadableRange,
            reported_address,
            reported_bytes as usize,
        );
        for alternate_readable in [true, false] {
            let mut allocations = 0;
            let mut spans = [span(0x1000, 1), span(0x3000, 1), span(0x4000, 1)];
            if !alternate_readable {
                spans[2].kind = MemoryType::RESERVED;
            }
            let captured = capture_with(
                select([(Format::Smbios3, 0x1000), (Format::Smbios2, 0x3000)].into_iter()),
                |address, output, stage| {
                    grant(spans.into_iter(), address, output.len(), 47)
                        .map_err(|reason| failure(stage, reason, address, output.len()))?;
                    let source = match address {
                        0x1000 => modern.as_slice(),
                        0x3000 => legacy.as_slice(),
                        0x4000 if alternate_readable => table.as_slice(),
                        _ => panic!("unpublished source"),
                    };
                    output.copy_from_slice(&source[..output.len()]);
                    Ok(())
                },
                |bytes| {
                    allocations += 1;
                    assert_eq!(bytes, legacy.len() + table.len());
                    Ok(Box::leak(vec![0; bytes].into_boxed_slice()))
                },
            );
            if alternate_readable {
                assert_eq!(captured.state, State::Present);
                assert_eq!(captured.format, Some(Format::Smbios2));
                assert_eq!(captured.source_address, 0x3000);
                assert_eq!(allocations, 1);
            } else {
                assert_eq!(captured.state, State::Failed(primary_failure));
                assert_eq!(captured.format, Some(Format::Smbios3));
                assert_eq!(captured.source_address, 0x1000);
                assert_eq!(allocations, 0);
            }
        }
    }
    #[test]
    fn changed_malformed_or_oversized_sources_do_not_publish_or_retry() {
        for changed in 0..5 {
            let (mut entry, mut table) = fixtures();
            if changed == 0 {
                entry[5] ^= 1;
            }
            if changed == 1 {
                entry[12..16].copy_from_slice(&((MAX_TABLE_BYTES + 1) as u32).to_le_bytes());
                entry[5] = 0;
                entry[5] = 0u8.wrapping_sub(entry.iter().fold(0u8, |a, b| a.wrapping_add(*b)));
            }
            if changed == 2 {
                table[4] = 255;
            }
            let mut calls = 0;
            let c = capture_with(
                select([(Format::Smbios3, 0x1000)].into_iter()),
                |address, output, stage| {
                    calls += 1;
                    let raw = if address == 0x1000 { &entry } else { &table };
                    output.copy_from_slice(&raw[..output.len()]);
                    if changed == 3 && calls == 2 {
                        output[7] ^= 1;
                    }
                    if changed == 4 && stage == Stage::Table {
                        return Err(Failure {
                            status: Some(42),
                            ..failure(stage, Error::UnreadableRange, address, output.len())
                        });
                    }
                    Ok(())
                },
                |_| panic!("invalid source must not reserve handoff memory"),
            );
            assert!(matches!(c.state, State::Failed(_)));
            assert_eq!(c.region.bytes, 0);
            assert!(calls <= 3);
            if changed == 1 {
                assert!(
                    matches!(c.state,State::Failed(Failure { reason: Error::TableBound { bytes }, .. }) if bytes == MAX_TABLE_BYTES+1)
                );
                if let State::Failed(error) = c.state {
                    assert_eq!(error.bytes, (MAX_TABLE_BYTES + 1) as u64);
                }
            }
            if changed == 3 {
                assert!(matches!(
                    c.state,
                    State::Failed(Failure {
                        reason: Error::ChangedEntry,
                        ..
                    })
                ));
            }
            if changed == 4 {
                assert!(matches!(
                    c.state,
                    State::Failed(Failure {
                        stage: Stage::Table,
                        status: Some(42),
                        ..
                    })
                ));
            }
        }
    }
}
