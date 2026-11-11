// SPDX-License-Identifier: GPL-2.0-or-later
use super::*;
use crate::wifi::intel::{command::Command, nvm::AddressSource};

fn reply(section: u16, offset: u16, length: u16, status: u16) -> [u8; 8] {
    let mut header = [0; 8];
    for (field, word) in header
        .chunks_exact_mut(2)
        .zip([offset, length, section, status])
    {
        field.copy_from_slice(&word.to_le_bytes());
    }
    header
}

#[test]
fn read_uses_the_wide_group_and_bounds_section_offset_and_length() {
    let request = Request::new(Section::PHY_SKU, 0x0800, CHUNK_BYTES).unwrap();
    assert_eq!(
        (request.section(), request.offset()),
        (Section::PHY_SKU, 0x0800)
    );
    let command = Command::encode(request.id(), 0, 0x123, &request.payload()).unwrap();
    assert_eq!(
        command.bytes(),
        &[0x88, 1, 0x23, 0, 8, 0, 0, 0, 0, 0, 12, 0, 0, 8, 0, 8]
    );
    assert_eq!(Section::new(0).unwrap().id(), 0);
    assert_eq!(Section::new(13), Err(Error::Section(13)));
    assert!(Request::new(Section::SW, 0x7800, CHUNK_BYTES).is_ok());
    for (offset, length) in [(0, 0), (0, CHUNK_BYTES + 1), (0x7801, CHUNK_BYTES)] {
        assert_eq!(
            Request::new(Section::SW, offset, length),
            Err(Error::Range { offset, length })
        );
    }
}

#[test]
fn replies_require_matching_fields_and_exact_payload_size() {
    let request = Request::new(Section::SW, CHUNK_BYTES, 2).unwrap();
    let header = reply(1, CHUNK_BYTES, 2, 0);
    let bytes = [header.as_slice(), &[0xab, 0xcd]].concat();
    assert_eq!(request.decode(&bytes), Ok(Read::Data(&[0xab, 0xcd])));
    for length in 0..bytes.len() {
        assert!(matches!(
            request.decode(&bytes[..length]),
            Err(Error::Length { .. })
        ));
    }
    assert!(matches!(
        request.decode(&[bytes.as_slice(), &[0]].concat()),
        Err(Error::Length { .. })
    ));
    assert_eq!(
        request.decode(&reply(3, CHUNK_BYTES, 0, 0)),
        Err(Error::ReplySection {
            expected: Section::SW,
            actual: 3,
        })
    );
    assert_eq!(
        request.decode(&reply(1, 0, 0, 0)),
        Err(Error::ReplyOffset {
            expected: CHUNK_BYTES,
            actual: 0,
        })
    );
    assert_eq!(
        request.decode(&reply(1, CHUNK_BYTES, 3, 0)),
        Err(Error::ReplyRange {
            requested: 2,
            actual: 3,
        })
    );
    assert_eq!(
        request.decode(&reply(1, CHUNK_BYTES, 0, 7)),
        Err(Error::Status(7))
    );
    assert_eq!(request.decode(&reply(1, CHUNK_BYTES, 0, 1)), Ok(Read::End));
    assert_eq!(
        Request::new(Section::SW, 0, CHUNK_BYTES)
            .unwrap()
            .decode(&reply(1, 0, 0, 1)),
        Ok(Read::Absent)
    );
}

#[test]
fn identity_uses_extended_fields_and_prefers_valid_strap_to_otp() {
    let sw = [0x1d, 0x0a, 0x02, 0x01, 0, 0, 0xb5, 0xff, 0, 0];
    let phy_sku = [0x3c, 0x5b, 0x10, 0xa7, 0x2f, 0, 0, 0];
    let strap = [0x0223_4567, 0x89ab];
    let otp = [0x0467_89ab, 0xcdef];
    let identity = Identity::decode(&sw, &phy_sku, strap, Some(otp)).unwrap();
    assert_eq!(identity.address.bytes, [2, 0x23, 0x45, 0x67, 0x89, 0xab]);
    assert_eq!(identity.address.source, AddressSource::Strap);
    assert_eq!(identity.nvm_version, 0x0102_0a1d);
    assert_eq!(identity.hardware_addresses, 5);
    assert_eq!(identity.radio_configuration, 0xa710_5b3c);
    assert_eq!(identity.radio_type, 0x105);
    assert_eq!(identity.radio_step, 0xb);
    assert_eq!(identity.radio_dash, 3);
    assert_eq!(identity.radio_flavor, 0xc);
    assert_eq!((identity.tx_antennas, identity.rx_antennas), (7, 0xa));
    assert_eq!(identity.sku_flags, 0x2f);

    let from_otp = Identity::decode(&sw, &phy_sku, [0, 0], Some(otp)).unwrap();
    assert_eq!(from_otp.address.source, AddressSource::Otp);
    assert_eq!(
        Identity::decode(&sw, &phy_sku, [0, 0], None),
        Err(Error::HardwareAddress)
    );
    assert!(matches!(
        Identity::decode(&sw[..9], &phy_sku, strap, None),
        Err(Error::IdentitySection {
            section: Section::SW,
            ..
        })
    ));
    assert!(matches!(
        Identity::decode(&sw, &phy_sku[..7], strap, None),
        Err(Error::IdentitySection {
            section: Section::PHY_SKU,
            ..
        })
    ));
}
