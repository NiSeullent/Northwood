// SPDX-License-Identifier: GPL-2.0-or-later AND GPL-2.0-only
// NCM mode negotiation adapted from Linux cdc_ncm.c:
// Copyright (C) ST-Ericsson 2010-2012
// Original author: Hans Petter Selasky <hans.petter.selasky@stericsson.com>
//! USB CDC Ethernet adapters as host network links: the Ethernet Control
//! Model (CDC ECM 1.2) and the Network Control Model (CDC NCM 1.0), the two
//! standard USB Ethernet classes.
//!
//! The communications interface (class 2, subclass 6 for ECM or 0x0d for
//! NCM) names its data interface in its Union descriptor and its station
//! address in the Ethernet Networking descriptor's iMACAddress string.
//! Attach configures the data interface's alternate setting with two bulk
//! endpoints, selects it (SET_INTERFACE), and sets the packet filter to
//! directed, broadcast and all-multicast frames (ECM 6.2.4; NCM only when
//! its functional descriptor's bmNetworkCapabilities offers the request).
//!
//! ECM moves one frame per bulk transfer: each received frame is one bulk
//! IN transfer ended by a short packet (3.3.1); each sent frame is one bulk
//! OUT transfer. NCM moves transfer blocks (`ntb`): while the data
//! interface is still at alternate setting 0, attach reads the function's
//! NTB parameters (GetNtbParameters), selects CRC-off and NTB16 when those
//! controls are supported, and caps its IN blocks at
//! [`NCM_IN_BLOCK`] bytes (SetNtbInputSize); each IN block may carry several
//! frames, and each sent frame is one OUT block. Either way a transfer that
//! is a multiple of the endpoint's packet size is followed by a zero-length
//! packet, except an NCM block of exactly the function's largest OUT size.
//! NetworkConnection notifications on the interrupt endpoint set the
//! carrier (ECM 6.3.1, and alike in NCM); until one arrives the link is
//! taken to be up. Frames above the 1514-byte Ethernet maximum are refused.
//! RNDIS shares the Ethernet link and bulk endpoints. Its own control
//! messages supply the station address and enable packet reception.
mod ntb;

use super::descriptor::{
    DEVICE_TO_HOST, Endpoint, Interface, RECIPIENT_INTERFACE, SET_INTERFACE, TYPE_CLASS, setup,
};
use super::rndis;
use super::xhci::{Device, DeviceAccess, Plan, Shared, SlotId, TransferTicket};
use crate::{
    Clock,
    block::BlockError,
    bus::dma::{DmaAllocator, DmaBuffer},
    net::{Link, LinkError, MAX_FRAME, Mac},
};
use alloc::{collections::VecDeque, format, string::String, vec::Vec};

pub const COMMUNICATIONS: u8 = 2;
/// The Ethernet Control Model subclass.
pub const ECM: u8 = 6;
/// The Network Control Model subclass.
pub const NCM: u8 = 0x0d;
/// Class requests and notifications address the communications interface.
const CLASS_INTERFACE: u8 = TYPE_CLASS | RECIPIENT_INTERFACE;
pub(crate) const DATA_CLASS: u8 = 10;
const CS_INTERFACE: u8 = 0x24;
const UNION: u8 = 0x06;
const ETHERNET: u8 = 0x0f;
/// The NCM functional descriptor's subtype.
const NCM_FUNCTIONAL: u8 = 0x1a;
/// bmNetworkCapabilities: SetEthernetPacketFilter is supported.
const NCM_PACKET_FILTER: u8 = 1 << 0;
/// bmNetworkCapabilities: CRC mode can be selected.
const NCM_CRC_MODE: u8 = 1 << 4;
/// bmNetworkCapabilities: SetNtbInputSize takes 8 bytes, the size and
/// wNtbInMaxDatagrams.
const NCM_INPUT_SIZE_8: u8 = 1 << 5;
const SET_ETHERNET_PACKET_FILTER: u8 = 0x43;
const GET_NTB_PARAMETERS: u8 = 0x80;
const SET_NTB_FORMAT: u8 = 0x84;
const SET_NTB_INPUT_SIZE: u8 = 0x86;
const SET_CRC_MODE: u8 = 0x8a;
const NTB16_FORMAT: u16 = 0;
const CRC_NOT_APPENDED: u16 = 0;
/// Directed, broadcast and all-multicast frames.
const FILTER: u16 = 0x04 | 0x08 | 0x02;
const NETWORK_CONNECTION: u8 = 0x00;
/// An ECM frame's DMA buffer.
const FRAME_BYTES: usize = 2048;
/// The largest NCM IN block this driver takes: room for ten full frames.
pub const NCM_IN_BLOCK: usize = 16 * 1024;
const SEND_US: u64 = 1_000_000;
const RNDIS_WIRELESS: u8 = 0xe0;
const RNDIS_IN_BYTES: usize = 16 * 1024;
const RNDIS_SEND_COMMAND: u8 = 0;
const RNDIS_GET_RESPONSE: u8 = 1;
const RNDIS_RESPONSE_BYTES: u16 = 1024;
const RNDIS_CONTROL_US: u64 = 10_000_000;
const RNDIS_RESPONSE_AVAILABLE: [u8; 8] = [1, 0, 0, 0, 0, 0, 0, 0];
const RNDIS_CURRENT_ADDRESS: u32 = 0x0101_0102;
const RNDIS_CURRENT_PACKET_FILTER: u32 = 0x0001_010e;
const RNDIS_FILTER: u32 = 0x01 | 0x04 | 0x08;

/// The CDC Ethernet model an adapter speaks.
#[derive(Clone, Copy, Debug)]
enum Model {
    Ecm,
    /// With its functional descriptor's bmNetworkCapabilities.
    Ncm(u8),
    Rndis,
}

/// How frames travel on the data interface.
enum Framing {
    Ecm,
    /// NTB16 blocks, with the next OUT block's wSequence.
    Ncm(ntb::Parameters, u16),
    Rndis(usize),
}

/// The interfaces and endpoints one adapter uses.
#[derive(Clone, Debug)]
pub struct Binding {
    model: Model,
    control: u8,
    data: u8,
    alternate: u8,
    mac_string: u8,
    input: Endpoint,
    output: Endpoint,
    notify: Option<Endpoint>,
}

impl Binding {
    /// The data interface and alternate setting selected by this binding.
    pub(super) fn data_setting(&self) -> (u8, u8) {
        (self.data, self.alternate)
    }

    pub(super) fn endpoints(&self) -> Vec<Endpoint> {
        let mut endpoints = alloc::vec![self.input, self.output];
        endpoints.extend(self.notify);
        endpoints
    }

    pub(super) fn preference(&self) -> u8 {
        match self.model {
            Model::Rndis => 1,
            Model::Ecm => 2,
            Model::Ncm(_) => 3,
        }
    }
}

pub(super) fn is_rndis(interface: &Interface) -> bool {
    // Microsoft's ACM descriptor and the Android wireless-controller variant.
    matches!(
        (interface.class, interface.subclass, interface.protocol),
        (COMMUNICATIONS, 2, 0xff) | (RNDIS_WIRELESS, 1, 3)
    )
}

pub(super) fn plan(interface: &Interface, all: &[Interface]) -> Result<Plan, &'static str> {
    let descriptor = |kind: u8, bytes: usize| {
        interface
            .extra
            .iter()
            .find(|d| d.len() >= bytes && d[1] == CS_INTERFACE && d[2] == kind)
    };
    let model = match (interface.class, interface.subclass) {
        _ if is_rndis(interface) => Model::Rndis,
        (COMMUNICATIONS, ECM) => Model::Ecm,
        (COMMUNICATIONS, NCM) => Model::Ncm(
            descriptor(NCM_FUNCTIONAL, 6)
                .ok_or("NCM interface without its functional descriptor")?[5],
        ),
        _ => return Err("not a CDC Ethernet interface"),
    };
    let union = descriptor(UNION, 5).ok_or("CDC Ethernet interface without a Union descriptor")?;
    let data = union[4];
    let mac_string = if matches!(model, Model::Rndis) {
        let unions = interface
            .extra
            .iter()
            .filter(|d| d.get(1..3) == Some([CS_INTERFACE, UNION].as_slice()))
            .count();
        if unions != 1
            || union.len() != 5
            || union[3] != interface.number
            || data == interface.number
        {
            return Err("RNDIS requires a Union descriptor naming one separate data interface");
        }
        0
    } else {
        descriptor(ETHERNET, 13)
            .ok_or("CDC Ethernet interface without an Ethernet Networking descriptor")?[3]
    };
    let setting = all
        .iter()
        .find(|i| {
            i.number == data
                && i.class == DATA_CLASS
                && i.endpoints.len() >= 2
                && (!matches!(model, Model::Rndis)
                    || (i.subclass == 0 && i.protocol == 0 && i.endpoints.len() == 2))
        })
        .ok_or("CDC Ethernet data interface without a bulk setting")?;
    let bulk = |is_in: bool| {
        setting
            .endpoints
            .iter()
            .copied()
            .find(|e| e.kind() == 2 && e.is_in() == is_in)
    };
    let (Some(input), Some(output)) = (bulk(true), bulk(false)) else {
        return Err("CDC Ethernet data interface without two bulk endpoints");
    };
    let notify = interface
        .endpoints
        .iter()
        .copied()
        .find(|e| e.kind() == 3 && e.is_in());
    if matches!(model, Model::Rndis)
        && (interface.endpoints.len() != 1
            || notify.is_none_or(|endpoint| endpoint.max_packet & 0x7ff < 8))
    {
        return Err("RNDIS requires an interrupt IN response endpoint");
    }
    Ok(Plan::Ethernet(Binding {
        model,
        control: interface.number,
        data,
        alternate: setting.alternate,
        mac_string,
        input,
        output,
        notify,
    }))
}

fn parse_mac(text: &str) -> Option<Mac> {
    if text.len() != 12 {
        return None;
    }
    let mut mac = [0u8; 6];
    for (n, byte) in mac.iter_mut().enumerate() {
        *byte = u8::from_str_radix(text.get(2 * n..2 * n + 2)?, 16).ok()?;
    }
    Some(Mac(mac))
}

fn read_mac(
    c: &mut DeviceAccess<'_>,
    device: &Device,
    binding: &Binding,
) -> Result<Mac, BlockError> {
    let text = device
        .language
        .and_then(|language| c.string(language, binding.mac_string))
        .ok_or(BlockError::Unsupported(
            "CDC Ethernet adapter without a MAC address string",
        ))?;
    parse_mac(&text).ok_or(BlockError::Unsupported("CDC Ethernet MAC address string"))
}

fn rndis_exchange(
    c: &mut DeviceAccess<'_>,
    binding: &Binding,
    status: &DmaBuffer,
    clock: Clock,
    command: &[u8],
) -> Result<Vec<u8>, BlockError> {
    let control = u16::from(binding.control);
    let notify = binding
        .notify
        .ok_or(BlockError::Protocol("missing RNDIS notification endpoint"))?;
    let clock_error = |_| BlockError::Unsupported("RNDIS requires a calibrated clock");
    clock
        .check_duration(RNDIS_CONTROL_US)
        .map_err(clock_error)?;
    let start = clock.try_start().map_err(clock_error)?;
    c.control(
        setup(
            CLASS_INTERFACE,
            RNDIS_SEND_COMMAND,
            0,
            control,
            command.len() as u16,
        ),
        command,
    )?;
    // RNDIS control requests and the eight-byte RESPONSE_AVAILABLE are
    // defined in Microsoft Learn's Control Channel Characteristics.
    let get = setup(
        DEVICE_TO_HOST | CLASS_INTERFACE,
        RNDIS_GET_RESPONSE,
        0,
        control,
        RNDIS_RESPONSE_BYTES,
    );
    for _ in 0..4 {
        let elapsed = clock.try_elapsed_us(start).map_err(clock_error)?;
        let remaining = RNDIS_CONTROL_US.saturating_sub(elapsed);
        if remaining == 0 {
            break;
        }
        let bytes = match c.transfer(
            notify.dci(),
            status.physical(),
            RNDIS_RESPONSE_AVAILABLE.len(),
            remaining,
        ) {
            Ok(bytes) => bytes,
            Err(error) => {
                // A synchronous error normally retires the TD. If Stop or
                // Set-Dequeue failed, fence this generation. The boot DMA
                // allocator retains the original grant after attach fails.
                if c.endpoint_ready(notify.dci()).is_err() {
                    c.quarantine();
                }
                return Err(error);
            }
        };
        if status.bytes().get(..bytes) != Some(RNDIS_RESPONSE_AVAILABLE.as_slice()) {
            return Err(BlockError::Protocol(
                "malformed RNDIS response notification",
            ));
        }
        let reply = c.control(get, &[])?;
        if clock.try_elapsed_us(start).map_err(clock_error)? >= RNDIS_CONTROL_US {
            break;
        }
        // The device can signal an unsolicited status or an empty response.
        if reply == [0] || reply.starts_with(&7u32.to_le_bytes()) {
            continue;
        }
        return Ok(reply);
    }
    Err(BlockError::Timeout)
}

fn rndis_reply_error(error: rndis::Error) -> BlockError {
    match error {
        rndis::Error::DeviceStatus(status) => BlockError::Device { status },
        rndis::Error::UnsupportedInitialize => {
            BlockError::Unsupported("unsupported RNDIS Ethernet profile")
        }
        _ => BlockError::Protocol("invalid RNDIS control message"),
    }
}

fn rndis_initialize(
    c: &mut DeviceAccess<'_>,
    binding: &Binding,
    status: &DmaBuffer,
    clock: Clock,
) -> Result<(Mac, Framing), BlockError> {
    let init = rndis::initialize_request(1, RNDIS_IN_BYTES as u32);
    let response = rndis_exchange(c, binding, status, clock, &init)?;
    let limits = rndis::initialize_reply(&response, 1).map_err(rndis_reply_error)?;
    let send_bytes = rndis::packet_transfer_len(MAX_FRAME, binding.output.max_packet & 0x7ff)
        .map_err(rndis_reply_error)?;
    if send_bytes > limits.max_transfer_size as usize {
        return Err(BlockError::Unsupported(
            "RNDIS OUT transfer cannot fit a maximum Ethernet frame",
        ));
    }

    let mut query = [0; 28];
    let length = rndis::query_request(2, RNDIS_CURRENT_ADDRESS, &[], &mut query)
        .map_err(rndis_reply_error)?;
    let response = rndis_exchange(c, binding, status, clock, &query[..length])?;
    let address = rndis::query_reply(&response, 2).map_err(rndis_reply_error)?;
    let mac = Mac(address
        .try_into()
        .map_err(|_| BlockError::Protocol("invalid RNDIS MAC address length"))?);
    if mac == Mac::default() || mac.is_multicast() {
        return Err(BlockError::Protocol("invalid RNDIS station address"));
    }
    Ok((mac, Framing::Rndis(send_bytes)))
}

pub struct UsbNet {
    controller: Shared,
    id: SlotId,
    binding: Binding,
    framing: Framing,
    mac: Mac,
    receive: DmaBuffer,
    /// Frames of an aggregated receive transfer not yet taken.
    pending: VecDeque<Vec<u8>>,
    send: DmaBuffer,
    status: DmaBuffer,
    queued: Option<TransferTicket>,
    notify_queued: Option<TransferTicket>,
    connected: Option<bool>,
    pub location: String,
}

impl UsbNet {
    /// The addressed device generation that owns this link.
    pub fn slot_id(&self) -> SlotId {
        self.id
    }

    pub(super) fn new(
        controller: Shared,
        device: &Device,
        binding: Binding,
        dma: &mut dyn DmaAllocator,
    ) -> Result<Self, BlockError> {
        let mut rndis_status = None;
        let (mac, framing) = {
            let mut host = controller.borrow_mut();
            let clock = host.clock;
            let mut c = host.device(device.id)?;
            let control = u16::from(binding.control);
            // NCM's mode and size requests precede the data setting.
            match binding.model {
                Model::Rndis => {
                    let status = dma.allocate(64, 64)?;
                    let initialized = rndis_initialize(&mut c, &binding, &status, clock)?;
                    rndis_status = Some(status);
                    initialized
                }
                Model::Ecm => (read_mac(&mut c, device, &binding)?, Framing::Ecm),
                Model::Ncm(capabilities) => {
                    let mac = read_mac(&mut c, device, &binding)?;
                    let request = setup(
                        DEVICE_TO_HOST | CLASS_INTERFACE,
                        GET_NTB_PARAMETERS,
                        0,
                        control,
                        ntb::PARAMETER_BYTES as u16,
                    );
                    let reply = c.control(request, &[])?;
                    let mut parameters =
                        ntb::Parameters::parse(&reply).map_err(BlockError::Unsupported)?;
                    if capabilities & NCM_CRC_MODE != 0 {
                        let request =
                            setup(CLASS_INTERFACE, SET_CRC_MODE, CRC_NOT_APPENDED, control, 0);
                        c.control(request, &[])?;
                    }
                    // NTB16 is the reset default. Like Linux cdc_ncm_init,
                    // explicitly select it when both formats are offered.
                    if parameters.supports_ntb32 {
                        let request =
                            setup(CLASS_INTERFACE, SET_NTB_FORMAT, NTB16_FORMAT, control, 0);
                        c.control(request, &[])?;
                    }
                    if parameters.in_max > NCM_IN_BLOCK {
                        parameters.in_max = NCM_IN_BLOCK;
                        let mut size = (NCM_IN_BLOCK as u32).to_le_bytes().to_vec();
                        if capabilities & NCM_INPUT_SIZE_8 != 0 {
                            // wNtbInMaxDatagrams 0: no limit.
                            size.extend_from_slice(&[0; 4]);
                        }
                        let request = setup(
                            CLASS_INTERFACE,
                            SET_NTB_INPUT_SIZE,
                            0,
                            control,
                            size.len() as u16,
                        );
                        c.control(request, &size)?;
                    }
                    (mac, Framing::Ncm(parameters, 0))
                }
            }
        };
        let (receive_bytes, send_bytes) = match &framing {
            Framing::Ecm => (FRAME_BYTES, FRAME_BYTES),
            Framing::Rndis(bytes) => (RNDIS_IN_BYTES, *bytes),
            Framing::Ncm(parameters, _) => (
                parameters.in_max,
                parameters
                    .encoded_len(MAX_FRAME)
                    .ok_or(BlockError::Unsupported(
                        "NCM OUT parameters cannot fit the maximum Ethernet frame",
                    ))?,
            ),
        };
        // Both one-TD buffers must stay inside one 64 KiB xHCI window. The
        // validated NTB16 OUT size is below 64 KiB, and IN is capped at 16 KiB;
        // next-power-of-two alignment therefore contains every transfer.
        let receive_align = receive_bytes
            .checked_next_power_of_two()
            .ok_or(BlockError::NoDmaMemory)?;
        let send_align = send_bytes
            .checked_next_power_of_two()
            .ok_or(BlockError::NoDmaMemory)?;
        let receive = dma.allocate(receive_bytes, receive_align)?;
        let send = dma.allocate(send_bytes, send_align)?;
        let status = match rndis_status {
            Some(status) => status,
            None => dma.allocate(64, 64)?,
        };
        // Do not activate the data setting if any bounded DMA allocation failed.
        {
            let mut host = controller.borrow_mut();
            let clock = host.clock;
            let mut c = host.device(device.id)?;
            let control = u16::from(binding.control);
            if binding.alternate != 0 {
                let request = setup(
                    1,
                    SET_INTERFACE,
                    u16::from(binding.alternate),
                    u16::from(binding.data),
                    0,
                );
                c.control(request, &[])?;
            }
            let filter = match binding.model {
                Model::Ecm => true,
                Model::Ncm(capabilities) => capabilities & NCM_PACKET_FILTER != 0,
                Model::Rndis => false,
            };
            if filter {
                let request = setup(
                    CLASS_INTERFACE,
                    SET_ETHERNET_PACKET_FILTER,
                    FILTER,
                    control,
                    0,
                );
                c.control(request, &[])?;
            }
            if matches!(binding.model, Model::Rndis) {
                let mut command = [0; 32];
                let bytes = rndis::set_request(
                    3,
                    RNDIS_CURRENT_PACKET_FILTER,
                    &RNDIS_FILTER.to_le_bytes(),
                    &mut command,
                )
                .map_err(rndis_reply_error)?;
                let reply = rndis_exchange(&mut c, &binding, &status, clock, &command[..bytes])?;
                rndis::set_reply(&reply, 3).map_err(rndis_reply_error)?;
            }
        }
        let location = if matches!(binding.model, Model::Rndis) {
            format!("{} ({}, RNDIS {mac})", device.location, device.name())
        } else {
            format!("{} ({})", device.location, device.name())
        };
        Ok(Self {
            controller,
            id: device.id,
            binding,
            framing,
            mac,
            receive,
            pending: VecDeque::new(),
            send,
            status,
            queued: None,
            notify_queued: None,
            connected: None,
            location,
        })
    }

    /// Takes any NetworkConnection notifications that arrived.
    fn notifications(&mut self) -> Result<(), BlockError> {
        let mut host = self.controller.borrow_mut();
        let mut c = host.device(self.id)?;
        let Some(notify) = self.binding.notify else {
            return Ok(());
        };
        let length = usize::from(notify.max_packet & 0x7ff).min(64);
        for _ in 0..4 {
            if self.notify_queued.is_none() {
                self.notify_queued =
                    Some(c.submit_ticket(notify.dci(), self.status.physical(), length)?);
            }
            let Some(result) = c.completed_ticket(self.notify_queued.as_ref().unwrap()) else {
                return Ok(());
            };
            self.notify_queued = None;
            if let Ok(bytes) = result
                && bytes >= 8
            {
                let header = &self.status.bytes()[..8];
                if header[0] == DEVICE_TO_HOST | CLASS_INTERFACE && header[1] == NETWORK_CONNECTION
                {
                    self.connected = Some(header[2] != 0);
                }
            }
        }
        Ok(())
    }
}

fn link_error(error: BlockError) -> LinkError {
    LinkError::Adapter(format!("USB Ethernet: {error}"))
}

impl Link for UsbNet {
    fn mac(&self) -> Mac {
        self.mac
    }

    fn carrier(&mut self) -> bool {
        if matches!(self.framing, Framing::Rndis(_)) {
            // Link indications use RNDIS control messages, so the USB
            // connection is the available carrier signal here.
            return self
                .controller
                .borrow_mut()
                .device(self.id)
                .and_then(|mut c| c.connected())
                .unwrap_or(false);
        }
        self.notifications().is_ok() && self.connected.unwrap_or(true)
    }

    fn transmit(&mut self, frame: &[u8]) -> Result<(), LinkError> {
        // A previous failed cancellation may still own this OUT buffer.
        let mut host = self.controller.borrow_mut();
        let mut c = host.device(self.id).map_err(link_error)?;
        c.endpoint_ready(self.binding.output.dci())
            .map_err(link_error)?;
        if frame.len() > MAX_FRAME {
            return Err(LinkError::Adapter(String::from(
                "frame above the Ethernet maximum",
            )));
        }
        // The length of the one OUT transfer, and the NCM block size that
        // needs no zero-length packet after it.
        let output = self.binding.output;
        let (length, whole) = match &mut self.framing {
            Framing::Ecm => {
                self.send.bytes_mut()[..frame.len()].copy_from_slice(frame);
                (frame.len(), None)
            }
            Framing::Ncm(parameters, sequence) => {
                let block = parameters
                    .encode(*sequence, frame)
                    .filter(|block| block.len() <= self.send.len())
                    .ok_or_else(|| {
                        LinkError::Adapter(String::from("frame above the NCM OUT block size"))
                    })?;
                *sequence = sequence.wrapping_add(1);
                self.send.bytes_mut()[..block.len()].copy_from_slice(&block);
                (block.len(), Some(parameters.out_max))
            }
            Framing::Rndis(_) => {
                let bytes =
                    rndis::encode_packet(frame, output.max_packet & 0x7ff, self.send.bytes_mut())
                        .map_err(|error| LinkError::Adapter(format!("RNDIS packet: {error:?}")))?;
                // The codec already supplies the USB short-packet byte.
                (bytes, Some(bytes))
            }
        };
        let physical = self.send.physical();
        let sent = c
            .transfer(output.dci(), physical, length, SEND_US)
            .map_err(link_error)?;
        if sent != length {
            return Err(LinkError::Adapter(String::from(
                "short USB Ethernet OUT transfer",
            )));
        }
        let packet = usize::from(output.max_packet & 0x7ff).max(1);
        if length.is_multiple_of(packet) && whole != Some(length) {
            c.transfer(output.dci(), physical, 0, SEND_US)
                .map_err(link_error)?;
        }
        Ok(())
    }

    fn receive(&mut self) -> Result<Option<Vec<u8>>, LinkError> {
        let mut host = self.controller.borrow_mut();
        let mut c = host.device(self.id).map_err(link_error)?;
        if let Some(frame) = self.pending.pop_front() {
            return Ok(Some(frame));
        }
        let input = self.binding.input;
        let capacity = self.receive.bytes().len();
        if self.queued.is_none() {
            self.queued = Some(
                c.submit_ticket(input.dci(), self.receive.physical(), capacity)
                    .map_err(link_error)?,
            );
        }
        let Some(result) = c.completed_ticket(self.queued.as_ref().unwrap()) else {
            return Ok(None);
        };
        self.queued = None;
        let bytes = match result {
            Ok(bytes) => &self.receive.bytes()[..bytes],
            Err(error) => {
                let _ = c.clear(&input);
                return Err(link_error(error));
            }
        };
        // A malformed block, like an oversized frame, is dropped as a NIC
        // drops a frame that fails its checks.
        let frames = match self.framing {
            Framing::Ecm => alloc::vec![bytes],
            Framing::Ncm(..) => ntb::decode(bytes).unwrap_or_default(),
            Framing::Rndis(_) => {
                let mut frames = Vec::new();
                let mut remainder = bytes;
                loop {
                    match rndis::decode_packet(remainder) {
                        Ok(Some(packet)) => {
                            frames.push(packet.frame);
                            remainder = packet.remainder;
                        }
                        Ok(None) => break frames,
                        Err(_) => break Vec::new(),
                    }
                }
            }
        };
        self.pending.extend(
            frames
                .into_iter()
                .filter(|frame| !frame.is_empty() && frame.len() <= MAX_FRAME)
                .map(<[u8]>::to_vec),
        );
        Ok(self.pending.pop_front())
    }
}

#[cfg(test)]
pub(crate) mod tests;
