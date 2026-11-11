// SPDX-License-Identifier: GPL-2.0-or-later

//! Wire messages for connectionless Ethernet Remote NDIS 1.0 over USB.
//!
//! Based on the public [Remote NDIS messaging specification][messages] and
//! its [USB data padding rules][padding]. The caller owns USB control
//! transfers, negotiated transfer limits and device lifecycle.
//!
//! [messages]: https://learn.microsoft.com/windows-hardware/drivers/network/remote-ndis-messaging
//! [padding]: https://learn.microsoft.com/windows-hardware/drivers/network/usb-short-packets

use crate::net::HEADER as ETHERNET_HEADER;

const PACKET: u32 = 1;
const INITIALIZE: u32 = 2;
const QUERY: u32 = 4;
const SET: u32 = 5;
const COMPLETION: u32 = 0x8000_0000;
const CONNECTIONLESS: u32 = 1;
const ETHERNET: u32 = 0;
const VERSION: (u32, u32) = (1, 0);

const HEADER: usize = 8;
const COMPLETION_HEADER: usize = 16;
const INITIALIZE_REPLY: usize = 52;
const QUERY_REPLY: usize = 24;
const OID_HEADER: usize = 28;
const PACKET_HEADER: usize = 44;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    ShortMessage,
    InvalidLength,
    UnexpectedType(u32),
    UnexpectedRequest(u32),
    DeviceStatus(u32),
    UnsupportedInitialize,
    InvalidOffset,
    ShortFrame,
    InvalidUsbPacketSize,
    BufferTooSmall,
}

/// Device limits for data transfers sent by the host.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Initialize {
    pub max_packets_per_transfer: u32,
    pub max_transfer_size: u32,
    /// Alignment in bytes between messages in a host multipacket transfer.
    pub packet_alignment: u32,
}

/// One received Ethernet frame and the unparsed remainder of its USB transfer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Packet<'a> {
    pub frame: &'a [u8],
    pub remainder: &'a [u8],
}

/// Advertise the host's maximum size for a single bulk IN transfer.
pub fn initialize_request(request_id: u32, max_transfer_size: u32) -> [u8; 24] {
    let mut request = [0; 24];
    write_u32(&mut request, 0, INITIALIZE);
    write_u32(&mut request, 4, 24);
    write_u32(&mut request, 8, request_id);
    write_u32(&mut request, 12, VERSION.0);
    write_u32(&mut request, 16, VERSION.1);
    write_u32(&mut request, 20, max_transfer_size);
    request
}

pub fn initialize_reply(response: &[u8], request_id: u32) -> Result<Initialize, Error> {
    let message = checked_completion(response, INITIALIZE, request_id)?;
    if message.len() < INITIALIZE_REPLY {
        return Err(Error::ShortMessage);
    }
    let version = (read_u32(message, 16), read_u32(message, 20));
    let max_packets_per_transfer = read_u32(message, 32);
    let max_transfer_size = read_u32(message, 36);
    let alignment_factor = read_u32(message, 40);
    if version != VERSION
        || read_u32(message, 24) != CONNECTIONLESS
        || read_u32(message, 28) != ETHERNET
        || max_packets_per_transfer == 0
        || max_transfer_size < (PACKET_HEADER + ETHERNET_HEADER) as u32
        || alignment_factor > 7
    {
        return Err(Error::UnsupportedInitialize);
    }
    Ok(Initialize {
        max_packets_per_transfer,
        max_transfer_size,
        packet_alignment: 1 << alignment_factor,
    })
}

/// Encode an OID query into `out` and return the number of bytes to send.
pub fn query_request(
    request_id: u32,
    oid: u32,
    input: &[u8],
    out: &mut [u8],
) -> Result<usize, Error> {
    oid_request(QUERY, request_id, oid, input, out)
}

pub fn query_reply(response: &[u8], request_id: u32) -> Result<&[u8], Error> {
    let message = checked_completion(response, QUERY, request_id)?;
    if message.len() < QUERY_REPLY {
        return Err(Error::ShortMessage);
    }
    let len = read_u32(message, 16);
    if len == 0 {
        return Ok(&[]);
    }
    payload(message, QUERY_REPLY, read_u32(message, 20), len)
}

/// Encode an OID set into `out` and return the number of bytes to send.
pub fn set_request(
    request_id: u32,
    oid: u32,
    value: &[u8],
    out: &mut [u8],
) -> Result<usize, Error> {
    oid_request(SET, request_id, oid, value, out)
}

pub fn set_reply(response: &[u8], request_id: u32) -> Result<(), Error> {
    checked_completion(response, SET, request_id).map(|_| ())
}

/// Frame one outgoing Ethernet packet and return its USB bulk OUT length.
/// The caller supplies the endpoint's maximum packet size in bytes.
pub fn encode_packet(frame: &[u8], usb_max_packet: u16, out: &mut [u8]) -> Result<usize, Error> {
    let transfer_len = packet_transfer_len(frame.len(), usb_max_packet)?;
    let len = PACKET_HEADER + frame.len();
    let out = out.get_mut(..transfer_len).ok_or(Error::BufferTooSmall)?;
    out[..PACKET_HEADER].fill(0);
    write_u32(out, 0, PACKET);
    write_u32(out, 4, len as u32);
    write_u32(out, 8, (PACKET_HEADER - HEADER) as u32);
    write_u32(out, 12, frame.len() as u32);
    out[PACKET_HEADER..len].copy_from_slice(frame);
    out[len..].fill(0);
    Ok(transfer_len)
}

pub(super) fn packet_transfer_len(frame_len: usize, usb_max_packet: u16) -> Result<usize, Error> {
    if frame_len < ETHERNET_HEADER {
        return Err(Error::ShortFrame);
    }
    if usb_max_packet == 0 {
        return Err(Error::InvalidUsbPacketSize);
    }
    let len = wire_length(PACKET_HEADER, frame_len)?;
    // The byte that terminates a full USB packet is outside MessageLength.
    let padding = usize::from(len.is_multiple_of(usize::from(usb_max_packet)));
    len.checked_add(padding).ok_or(Error::InvalidLength)
}

/// Decode one frame from a bulk IN transfer, including an aggregated transfer.
/// Call again with `remainder`. Trailing USB zero padding ends the transfer.
pub fn decode_packet(transfer: &[u8]) -> Result<Option<Packet<'_>>, Error> {
    if transfer.is_empty() || (transfer[0] == 0 && transfer.iter().all(|&byte| byte == 0)) {
        return Ok(None);
    }
    let message = checked_message(transfer, PACKET, PACKET_HEADER)?;
    let offset = read_u32(message, 8);
    if !offset.is_multiple_of(4) {
        return Err(Error::InvalidOffset);
    }
    let len = read_u32(message, 12);
    if len < ETHERNET_HEADER as u32 {
        return Err(Error::ShortFrame);
    }
    // Per-packet metadata is not used by the Ethernet link. The payload
    // must be wholly contained in this message, excluding the next header.
    let frame = payload(message, PACKET_HEADER, offset, len)?;
    Ok(Some(Packet {
        frame,
        remainder: &transfer[message.len()..],
    }))
}

fn oid_request(
    message_type: u32,
    request_id: u32,
    oid: u32,
    input: &[u8],
    out: &mut [u8],
) -> Result<usize, Error> {
    let len = wire_length(OID_HEADER, input.len())?;
    let out = out.get_mut(..len).ok_or(Error::BufferTooSmall)?;
    let offset = if input.is_empty() {
        0
    } else {
        OID_HEADER - HEADER
    };
    write_u32(out, 0, message_type);
    write_u32(out, 4, len as u32);
    write_u32(out, 8, request_id);
    write_u32(out, 12, oid);
    write_u32(out, 16, input.len() as u32);
    write_u32(out, 20, offset as u32);
    write_u32(out, 24, 0);
    out[OID_HEADER..].copy_from_slice(input);
    Ok(len)
}

fn wire_length(header: usize, payload: usize) -> Result<usize, Error> {
    header
        .checked_add(payload)
        .filter(|&len| u32::try_from(len).is_ok())
        .ok_or(Error::InvalidLength)
}

fn checked_message(input: &[u8], expected_type: u32, minimum: usize) -> Result<&[u8], Error> {
    if input.len() < HEADER {
        return Err(Error::ShortMessage);
    }
    let message_type = read_u32(input, 0);
    if message_type != expected_type {
        return Err(Error::UnexpectedType(message_type));
    }
    let len = read_u32(input, 4) as usize;
    if len < minimum {
        return Err(Error::InvalidLength);
    }
    input.get(..len).ok_or(Error::ShortMessage)
}

fn checked_completion(response: &[u8], expected: u32, request_id: u32) -> Result<&[u8], Error> {
    let message = checked_message(response, expected | COMPLETION, COMPLETION_HEADER)?;
    let actual = read_u32(message, 8);
    if actual != request_id {
        return Err(Error::UnexpectedRequest(actual));
    }
    let status = read_u32(message, 12);
    if status != 0 {
        return Err(Error::DeviceStatus(status));
    }
    Ok(message)
}

fn payload(message: &[u8], minimum: usize, offset: u32, len: u32) -> Result<&[u8], Error> {
    // Packet and OID offsets are both relative to byte 8 of the message.
    let start = HEADER
        .checked_add(offset as usize)
        .ok_or(Error::InvalidOffset)?;
    if start < minimum {
        return Err(Error::InvalidOffset);
    }
    let end = start
        .checked_add(len as usize)
        .ok_or(Error::InvalidLength)?;
    message.get(start..end).ok_or(Error::InvalidLength)
}

fn write_u32(output: &mut [u8], offset: usize, value: u32) {
    output[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    let word = [
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ];
    u32::from_le_bytes(word)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words<const N: usize>(values: &[u32]) -> [u8; N] {
        assert_eq!(N, values.len() * 4);
        let mut bytes = [0; N];
        for (field, value) in bytes.chunks_exact_mut(4).zip(values) {
            field.copy_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn initialize_checks_request_version_medium_and_limits() {
        let request = initialize_request(7, 4096);
        assert_eq!(request, words(&[2, 24, 7, 1, 0, 4096]));
        let reply = words::<52>(&[0x8000_0002, 52, 7, 0, 1, 0, 1, 0, 2, 4096, 3, 0, 0]);
        assert_eq!(
            initialize_reply(&reply, 7),
            Ok(Initialize {
                max_packets_per_transfer: 2,
                max_transfer_size: 4096,
                packet_alignment: 8
            })
        );
        assert_eq!(
            initialize_reply(&reply, 8),
            Err(Error::UnexpectedRequest(7))
        );
        assert_eq!(initialize_reply(&reply[..51], 7), Err(Error::ShortMessage));
        for (offset, field) in [(16, 2), (24, 2), (28, 1), (32, 0), (36, 57), (40, 8)] {
            let mut invalid = reply;
            invalid[offset..offset + 4].copy_from_slice(&u32::to_le_bytes(field));
            assert_eq!(
                initialize_reply(&invalid, 7),
                Err(Error::UnsupportedInitialize)
            );
        }
    }

    #[test]
    fn query_and_set_serialize_relative_oid_offsets() {
        let mut output = [0xff; 40];
        let size = query_request(9, 0x0101_0102, &[], &mut output).unwrap();
        assert_eq!(
            output[..size],
            words::<28>(&[4, 28, 9, 0x0101_0102, 0, 0, 0])
        );
        let size = set_request(10, 0x0001_010e, &[0x0d, 0, 0, 0], &mut output).unwrap();
        assert_eq!(
            output[..size],
            words::<32>(&[5, 32, 10, 0x0001_010e, 4, 20, 0, 0x0d])
        );
        assert_eq!(output[size], 0xff);
        assert_eq!(
            set_request(10, 1, &[1], &mut output[..28]),
            Err(Error::BufferTooSmall)
        );
    }

    #[test]
    fn control_replies_bound_data_and_surface_status() {
        let query = words::<32>(&[0x8000_0004, 30, 9, 0, 6, 16, 0x0302_0100, 0xaaaa_0504]);
        assert_eq!(query_reply(&query, 9), Ok([0, 1, 2, 3, 4, 5].as_slice()));
        let overruns_message = words::<32>(&[0x8000_0004, 30, 9, 0, 7, 16, 0, 0]);
        assert_eq!(query_reply(&overruns_message, 9), Err(Error::InvalidLength));
        let points_into_header = words::<28>(&[0x8000_0004, 28, 9, 0, 4, 12, 0]);
        assert_eq!(
            query_reply(&points_into_header, 9),
            Err(Error::InvalidOffset)
        );
        let empty = words::<24>(&[0x8000_0004, 24, 9, 0, 0, 0]);
        assert_eq!(query_reply(&empty, 9), Ok([].as_slice()));
        let success = words::<16>(&[0x8000_0005, 16, 10, 0]);
        let failure = words::<16>(&[0x8000_0005, 16, 10, 0xc000_00bb]);
        assert_eq!(set_reply(&success, 10), Ok(()));
        assert_eq!(
            set_reply(&failure, 10),
            Err(Error::DeviceStatus(0xc000_00bb))
        );
        assert_eq!(
            query_reply(&success, 10),
            Err(Error::UnexpectedType(0x8000_0005))
        );
    }

    #[test]
    fn usb_terminator_is_outside_message_length() {
        let frame = [0x42; 20];
        let mut out = [0xcc; 66];
        assert_eq!(encode_packet(&frame, 64, &mut out), Ok(65));
        assert_eq!(
            &out[..44],
            &words::<44>(&[1, 64, 36, 20, 0, 0, 0, 0, 0, 0, 0])
        );
        assert_eq!(out[64], 0);
        assert_eq!(out[65], 0xcc);
        let packet = decode_packet(&out[..65]).unwrap().unwrap();
        assert_eq!(packet.frame, frame);
        assert_eq!(decode_packet(packet.remainder), Ok(None));
        assert_eq!(
            encode_packet(&frame, 64, &mut out[..64]),
            Err(Error::BufferTooSmall)
        );
        assert_eq!(
            encode_packet(&frame, 0, &mut out),
            Err(Error::InvalidUsbPacketSize)
        );
    }

    #[test]
    fn bulk_in_walks_message_lengths_and_padding() {
        let mut transfer = [0; 132];
        let first = words::<44>(&[1, 64, 36, 14, 0, 0, 0, 0, 0, 0, 0]);
        let second = words::<44>(&[1, 64, 40, 16, 0, 0, 0, 0, 0, 0, 0]);
        transfer[..44].copy_from_slice(&first);
        transfer[44..58].fill(0x11);
        transfer[58..64].fill(0xfe);
        transfer[64..108].copy_from_slice(&second);
        transfer[112..128].fill(0x22);
        let packet = decode_packet(&transfer).unwrap().unwrap();
        assert_eq!(packet.frame, &[0x11; 14]);
        let packet = decode_packet(packet.remainder).unwrap().unwrap();
        assert_eq!(packet.frame, &[0x22; 16]);
        assert_eq!(decode_packet(packet.remainder), Ok(None));
        transfer[131] = 1;
        assert_eq!(decode_packet(&transfer[128..]), Err(Error::ShortMessage));
    }

    #[test]
    fn invalid_packets_cannot_cross_a_message_boundary() {
        let mut transfer = [0; 100];
        for (len, offset, frame_len, error) in [
            (43, 36, 20, Error::InvalidLength),
            (101, 36, 20, Error::ShortMessage),
            (64, 32, 20, Error::InvalidOffset),
            (64, 37, 20, Error::InvalidOffset),
            (64, 36, 21, Error::InvalidLength),
            (64, u32::MAX - 3, 20, Error::InvalidLength),
            (64, 36, 13, Error::ShortFrame),
        ] {
            transfer[..44].copy_from_slice(&words::<44>(&[
                1, len, offset, frame_len, 0, 0, 0, 0, 0, 0, 0,
            ]));
            assert_eq!(decode_packet(&transfer), Err(error));
        }
    }
}
