// SPDX-License-Identifier: GPL-2.0-or-later
use super::*;

fn notification(kind: u16, data: &[u8]) -> Vec<u8> {
    let mut payload = Vec::from(kind.to_le_bytes());
    payload.extend((data.len() as u16).to_le_bytes());
    payload.extend_from_slice(data);
    payload
}

#[test]
fn retains_latest_exact_bytes_in_runtime_order() {
    let mut database = Database::default();
    for (kind, data) in [
        (Kind::Papd, &[2, 0, 0xa2][..]),
        (Kind::TxPower, &[3, 0, 0, 0, 0x19, 0]),
        (Kind::NonChannel, &[0xc0]),
        (Kind::Papd, &[0, 0, 0xa0]),
        (Kind::Configuration, &[0xcf]),
        (Kind::Papd, &[2, 0, 0xb2]),
    ] {
        database.record(&notification(kind.id(), data)).unwrap();
    }
    let sections: Vec<_> = database
        .sections()
        .map(|section| (section.kind, section.group, section.data))
        .collect();
    assert_eq!(
        sections,
        [
            (Kind::Configuration, None, &[0xcf][..]),
            (Kind::NonChannel, None, &[0xc0]),
            (Kind::Papd, Some(0), &[0, 0, 0xa0]),
            (Kind::Papd, Some(2), &[2, 0, 0xb2]),
            (Kind::TxPower, Some(3), &[3, 0, 0, 0, 0x19, 0]),
        ]
    );
}

#[test]
fn rejects_inconsistent_headers_groups_and_packets() {
    let mut database = Database::default();
    assert_eq!(
        database.record(&[1, 0, 1]),
        Err(Error::Length {
            expected: 4,
            actual: 3,
        })
    );
    assert_eq!(
        database.record(&[1, 0, 0, 0, 9]),
        Err(Error::Length {
            expected: 4,
            actual: 5,
        })
    );
    assert_eq!(
        database.record(&notification(3, &[0])),
        Err(Error::UnknownType(3))
    );
    assert_eq!(
        database.record(&notification(5, &[1, 0, 0, 0, 0])),
        Err(Error::GroupLength {
            kind: Kind::TxPower,
            minimum: 6,
            actual: 5,
        })
    );
    database.record(&notification(4, &[2, 0])).unwrap();
    assert_eq!(
        database.record(&notification(4, &[3, 0])),
        Err(Error::GroupOrder {
            kind: Kind::Papd,
            highest: 2,
            actual: 3,
        })
    );
    let size = MAX_SECTION_BYTES + 1;
    assert_eq!(
        database.record(&notification(1, &vec![0; size])),
        Err(Error::SectionSize(size))
    );
    assert_eq!(database.sections().len(), 1);
}

#[test]
fn sparse_ids_and_replacements_stay_within_one_rfh_ring() {
    let mut database = Database::default();
    let mut data = vec![0; MAX_SECTION_BYTES];
    for id in (1..=ENTRIES).rev() {
        data[..2].copy_from_slice(&id.to_le_bytes());
        database.record(&notification(4, &data)).unwrap();
    }
    data[..2].copy_from_slice(&0u16.to_le_bytes());
    assert_eq!(
        database.record(&notification(4, &data)),
        Err(Error::DatabaseFull)
    );
    database.record(&notification(4, &[1, 0])).unwrap();
    database.record(&notification(4, &[0, 0])).unwrap();
    assert_eq!(database.sections().len(), usize::from(ENTRIES) + 1);

    let mut sparse = Database::default();
    sparse.record(&notification(4, &[0xff, 0xff])).unwrap();
    assert_eq!(sparse.sections().len(), 1);
}
