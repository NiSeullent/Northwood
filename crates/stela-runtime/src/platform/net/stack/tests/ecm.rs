// SPDX-License-Identifier: GPL-2.0-or-later
//! The runtime network client plays the USB host. A private Ethernet peer
//! sees only frames assembled by ECM and sends replies through ECM bulk IN.
use super::*;
use smoltcp::{
    phy::ChecksumCapabilities,
    wire::{
        ArpOperation, ArpPacket, ArpRepr, DHCP_CLIENT_PORT, DHCP_SERVER_PORT, DhcpMessageType,
        DhcpPacket, DhcpRepr, EthernetFrame, EthernetProtocol, EthernetRepr, Icmpv4DstUnreachable,
        Icmpv4Packet, Icmpv4Repr, IpProtocol, Ipv4Packet, Ipv4Repr, UDP_HEADER_LEN, UdpPacket,
        UdpRepr,
    },
};
use stela_soc::usb::{
    BusPeer, MAX_PACKET, Reply, Token,
    control::{SET_ADDRESS, SET_CONFIGURATION, SET_INTERFACE},
    ecm::{Counters, Ecm},
};

const BULK_IN: u8 = 0x81;
const BULK_OUT: u8 = 0x02;
const CLIENT_ETH: EthernetAddress = EthernetAddress(CLIENT.0);
const SERVER_ETH: EthernetAddress = EthernetAddress(SERVER);
const PROBE_PORTS: UdpRepr = UdpRepr {
    src_port: 49153,
    dst_port: 32767,
};
const PROBE_DATA: &[u8] = b"ecm!";

#[derive(Debug, PartialEq, Eq)]
enum Outbound {
    Dhcp(DhcpMessageType),
    Ipv6,
    ArpReply,
    PortUnreachable,
}

struct Peer {
    ecm: Ecm,
    outbound: Vec<Outbound>,
}

impl Peer {
    fn new() -> Self {
        let mut ecm = Ecm::new(CLIENT).unwrap();
        for mut setup in [
            [0, SET_ADDRESS, 3, 0, 0, 0, 0, 0],
            [0, SET_CONFIGURATION, 1, 0, 0, 0, 0, 0],
            [1, SET_INTERFACE, 1, 0, 1, 0, 0, 0],
        ] {
            assert_eq!(ecm.packet(0, Token::Setup, &mut setup), Ok(Reply::Ack(8)));
            assert_eq!(ecm.packet(0, Token::In, &mut []), Ok(Reply::Ack(0)));
        }
        assert!(ecm.up());
        Self {
            ecm,
            outbound: Vec::new(),
        }
    }

    fn from_guest(&mut self, frame: &[u8]) {
        let eth = EthernetFrame::new_checked(frame).unwrap();
        assert_eq!(eth.src_addr(), CLIENT_ETH);
        let checksums = ChecksumCapabilities::default();
        match eth.ethertype() {
            EthernetProtocol::Ipv4 => {
                let packet = Ipv4Packet::new_checked(eth.payload()).unwrap();
                let ip = Ipv4Repr::parse(&packet, &checksums).unwrap();
                match ip.next_header {
                    IpProtocol::Udp => self.dhcp(eth.dst_addr(), ip, packet.payload()),
                    IpProtocol::Icmp => {
                        assert_eq!(eth.dst_addr(), SERVER_ETH);
                        assert_eq!((ip.src_addr, ip.dst_addr), (CLIENT_IP, SERVER_IP));
                        let packet = Icmpv4Packet::new_checked(packet.payload()).unwrap();
                        let Icmpv4Repr::DstUnreachable {
                            reason,
                            header,
                            data,
                        } = Icmpv4Repr::parse(&packet, &checksums).unwrap()
                        else {
                            panic!("expected an ICMP error for the peer's probe");
                        };
                        assert_eq!(reason, Icmpv4DstUnreachable::PortUnreachable);
                        assert_eq!((header.src_addr, header.dst_addr), (SERVER_IP, CLIENT_IP));
                        assert_eq!(header.next_header, IpProtocol::Udp);
                        let quoted = UdpPacket::new_checked(data).unwrap();
                        assert_eq!(
                            (quoted.src_port(), quoted.dst_port()),
                            (PROBE_PORTS.src_port, PROBE_PORTS.dst_port)
                        );
                        assert_eq!(quoted.payload(), PROBE_DATA);
                        self.outbound.push(Outbound::PortUnreachable);
                    }
                    other => panic!("unexpected IPv4 protocol: {other:?}"),
                }
            }
            EthernetProtocol::Arp => {
                let arp = ArpPacket::new_checked(eth.payload()).unwrap();
                assert_eq!(
                    ArpRepr::parse(&arp).unwrap(),
                    ArpRepr::EthernetIpv4 {
                        operation: ArpOperation::Reply,
                        source_hardware_addr: CLIENT_ETH,
                        source_protocol_addr: CLIENT_IP,
                        target_hardware_addr: SERVER_ETH,
                        target_protocol_addr: SERVER_IP,
                    }
                );
                self.outbound.push(Outbound::ArpReply);
            }
            EthernetProtocol::Ipv6 => self.outbound.push(Outbound::Ipv6),
            other => panic!("unexpected EtherType: {other:?}"),
        }
    }

    fn dhcp(&mut self, destination: EthernetAddress, ip: Ipv4Repr, data: &[u8]) {
        assert_eq!(destination, EthernetAddress::BROADCAST);
        assert_eq!(
            (ip.src_addr, ip.dst_addr),
            (Ipv4::UNSPECIFIED, Ipv4::BROADCAST)
        );
        let udp = UdpPacket::new_checked(data).unwrap();
        assert_eq!(
            (udp.src_port(), udp.dst_port()),
            (DHCP_CLIENT_PORT, DHCP_SERVER_PORT)
        );
        let packet = DhcpPacket::new_checked(udp.payload()).unwrap();
        let request = DhcpRepr::parse(&packet).unwrap();
        assert_eq!(request.client_hardware_address, CLIENT_ETH);
        let message_type = match request.message_type {
            DhcpMessageType::Discover => DhcpMessageType::Offer,
            DhcpMessageType::Request => {
                assert_eq!(request.requested_ip, Some(CLIENT_IP));
                assert_eq!(request.server_identifier, Some(SERVER_IP));
                DhcpMessageType::Ack
            }
            other => panic!("unexpected DHCP message: {other:?}"),
        };
        self.outbound.push(Outbound::Dhcp(request.message_type));
        let reply = DhcpRepr {
            message_type,
            your_ip: CLIENT_IP,
            server_ip: SERVER_IP,
            server_identifier: Some(SERVER_IP),
            router: Some(SERVER_IP),
            subnet_mask: Some(Ipv4::new(255, 255, 255, 0)),
            dns_servers: Some(core::iter::once(SERVER_IP).collect()),
            lease_duration: Some(120),
            client_identifier: None,
            requested_ip: None,
            parameter_request_list: None,
            max_size: None,
            ..request
        };
        let data_len = reply.buffer_len();
        let frame = udp_frame(
            EthernetAddress::BROADCAST,
            Ipv4::BROADCAST,
            UdpRepr {
                src_port: DHCP_SERVER_PORT,
                dst_port: DHCP_CLIENT_PORT,
            },
            data_len,
            |bytes| reply.emit(&mut DhcpPacket::new_unchecked(bytes)).unwrap(),
        );
        self.ecm.deliver(&frame);
    }
}

impl Link for Peer {
    fn mac(&self) -> Mac {
        CLIENT
    }

    fn carrier(&mut self) -> bool {
        self.ecm.up()
    }

    fn transmit(&mut self, frame: &[u8]) -> Result<(), LinkError> {
        for chunk in frame.chunks(MAX_PACKET) {
            assert_eq!(
                self.ecm.packet(BULK_OUT, Token::Out, &mut chunk.to_vec()),
                Ok(Reply::Ack(chunk.len()))
            );
        }
        if frame.len().is_multiple_of(MAX_PACKET) {
            assert_eq!(
                self.ecm.packet(BULK_OUT, Token::Out, &mut []),
                Ok(Reply::Ack(0))
            );
        }
        let assembled = self
            .ecm
            .take()
            .expect("the peer must see a complete bulk OUT frame");
        assert_eq!(assembled, frame);
        self.from_guest(&assembled);
        Ok(())
    }

    fn receive(&mut self) -> Result<Option<Vec<u8>>, LinkError> {
        let mut frame = Vec::new();
        loop {
            let mut packet = [0; MAX_PACKET];
            match self.ecm.packet(BULK_IN, Token::In, &mut packet).unwrap() {
                Reply::Nak if frame.is_empty() => return Ok(None),
                Reply::Ack(bytes) => {
                    frame.extend_from_slice(&packet[..bytes]);
                    if bytes < packet.len() {
                        return Ok(Some(frame));
                    }
                }
                reply => panic!("incomplete bulk IN frame: {reply:?}"),
            }
        }
    }
}

fn ethernet_frame(
    destination: EthernetAddress,
    protocol: EthernetProtocol,
    length: usize,
    emit: impl FnOnce(&mut [u8]),
) -> Vec<u8> {
    let mut bytes = vec![0; EthernetFrame::<&[u8]>::buffer_len(length)];
    let mut frame = EthernetFrame::new_unchecked(&mut bytes);
    EthernetRepr {
        src_addr: SERVER_ETH,
        dst_addr: destination,
        ethertype: protocol,
    }
    .emit(&mut frame);
    emit(frame.payload_mut());
    bytes
}

fn udp_frame(
    destination: EthernetAddress,
    address: Ipv4,
    ports: UdpRepr,
    length: usize,
    emit: impl FnOnce(&mut [u8]),
) -> Vec<u8> {
    let ip = Ipv4Repr {
        src_addr: SERVER_IP,
        dst_addr: address,
        next_header: IpProtocol::Udp,
        payload_len: UDP_HEADER_LEN + length,
        hop_limit: 64,
    };
    ethernet_frame(
        destination,
        EthernetProtocol::Ipv4,
        ip.buffer_len() + ip.payload_len,
        |payload| {
            let mut packet = Ipv4Packet::new_unchecked(payload);
            let checksums = ChecksumCapabilities::default();
            ip.emit(&mut packet, &checksums);
            ports.emit(
                &mut UdpPacket::new_unchecked(packet.payload_mut()),
                &SERVER_IP.into(),
                &address.into(),
                length,
                emit,
                &checksums,
            );
        },
    )
}

#[test]
fn dhcp_and_leased_ipv4_move_ethernet_frames_through_ecm_bulk_endpoints() {
    let mut stack = Stack::new(Peer::new(), TestClock::default(), 7).unwrap();
    assert_eq!(
        stack.dhcp().unwrap(),
        &Config {
            address: Ipv4Cidr::new(CLIENT_IP, 24),
            router: Some(SERVER_IP),
            dns: vec![SERVER_IP],
        }
    );
    {
        let peer = stack.link_mut();
        assert_eq!(
            peer.outbound,
            [
                Outbound::Dhcp(DhcpMessageType::Discover),
                Outbound::Ipv6,
                Outbound::Dhcp(DhcpMessageType::Request)
            ]
        );
        assert_eq!(
            peer.ecm.counters(),
            Counters {
                to_host: 3,
                to_guest: 2,
                ..Counters::default()
            }
        );

        let arp = ArpRepr::EthernetIpv4 {
            operation: ArpOperation::Request,
            source_hardware_addr: SERVER_ETH,
            source_protocol_addr: SERVER_IP,
            target_hardware_addr: EthernetAddress([0; 6]),
            target_protocol_addr: CLIENT_IP,
        };
        peer.ecm.deliver(&ethernet_frame(
            EthernetAddress::BROADCAST,
            EthernetProtocol::Arp,
            arp.buffer_len(),
            |data| {
                arp.emit(&mut ArpPacket::new_unchecked(data));
            },
        ));
        peer.ecm.deliver(&udp_frame(
            CLIENT_ETH,
            CLIENT_IP,
            PROBE_PORTS,
            PROBE_DATA.len(),
            |data| data.copy_from_slice(PROBE_DATA),
        ));
    }
    stack.poll().unwrap();
    let peer = stack.link_mut();
    assert_eq!(
        &peer.outbound[3..],
        [Outbound::ArpReply, Outbound::PortUnreachable]
    );
    assert_eq!(
        peer.ecm.counters(),
        Counters {
            to_host: 5,
            to_guest: 4,
            ..Counters::default()
        }
    );
}
