//! usbmux stream host. Protocol evidence: usbmuxd 3ded00c9985a5108cfc7591a
//! `src/device.c` send_packet, send_tcp, device_version_input and
//! device_tcp_input. Network byte order, v1/v2 framing, v2 setup byte 07,
//! TCP-without-IP and window scaling by eight are independent of USB packets.
//! Like that peer, closure uses RST; FIN is not advertised as supported.
use crate::Result;
use alloc::{collections::VecDeque, vec::Vec};
use stela_contracts::refuse;

pub const MAX_FRAME: usize = 65536;
// usbmuxd usb.h USB_MTU; receive MRU is independently 64 KiB.
const TX_MTU: usize = 3 * 16384;
const MAX_STREAMS: usize = 8;
const MAX_QUEUED: usize = 64;
/// The transmit queue's bytes: several full segments, so the bus finds the
/// next ready as it finishes one, as usbmuxd keeps a transfer per packet.
const MAX_QUEUED_BYTES: usize = 4 * TX_MTU;
const RX_WINDOW: usize = 65536;
const MAGIC: u32 = 0xfeed_face;
const VERSION: u32 = 0;
const CONTROL: u32 = 1;
const SETUP: u32 = 2;
const TCP: u32 = 6;
const SYN: u8 = 2;
const RST: u8 = 4;
const ACK: u8 = 16;
const TCP_HEADER: usize = 20;
const VERSION_HEADER: usize = 8;
const MUX2_HEADER: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    Negotiating,
    Active,
    Failed,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamState {
    Connecting,
    Open,
    Refused,
    Reset,
    Closed,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamId {
    generation: u64,
    port: u16,
}
struct Stream {
    id: StreamId,
    destination: u16,
    state: StreamState,
    next: u32,
    acknowledged: u32,
    received: u32,
    window: u32,
    input: VecDeque<u8>,
    accepted_bytes: u64,
    acknowledged_bytes: u64,
    zero_window_writes: u64,
    full_window_writes: u64,
    credit_wait_since: Option<u64>,
    credit_wait_ticks: u64,
    /// An ACK or RST owed to the peer while the transmit queue was full.
    /// TCP lets one later ACK cover every byte received meanwhile.
    reply: Option<u8>,
}
impl Stream {
    fn outstanding(&self) -> u32 {
        self.next.wrapping_sub(self.acknowledged)
    }
}

pub struct Mux {
    state: State,
    version: u32,
    generation: u64,
    next_port: u16,
    tx_sequence: u16,
    rx_sequence: u16,
    receive: Vec<u8>,
    outgoing: VecDeque<Vec<u8>>,
    queued_bytes: usize,
    streams: Vec<Stream>,
    logs: VecDeque<Vec<u8>>,
    /// TCP packets shorter than a TCP header, which usbmuxd drops
    /// ("Incoming TCP packet is too small").
    short_tcp: u64,
    backpressure: Backpressure,
    observed_at: u64,
}
/// What held back [`Mux::write`] since the connection began: the writes the
/// peer's receive window refused and those the transmit queue did, and the
/// largest window the peer advertised.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Backpressure {
    pub window: u64,
    pub queue: u64,
    pub largest_window: u32,
}
/// Per-stream transmit accounting. Byte counts include framing above TCP.
/// Credit wait starts at an attempted write and ends at an observed window
/// update or ACK that gives the sender room. Time uses the USB session clock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamStats {
    pub accepted_bytes: u64,
    pub acknowledged_bytes: u64,
    pub window_bytes: u32,
    pub outstanding_bytes: u32,
    pub zero_window_writes: u64,
    pub full_window_writes: u64,
    pub credit_wait_ticks: u64,
    pub credit_waiting: bool,
}
impl Default for Mux {
    fn default() -> Self {
        Self::new()
    }
}
impl Mux {
    pub fn new() -> Self {
        Self::with_generation(0)
    }
    fn with_generation(generation: u64) -> Self {
        let mut mux = Self {
            state: State::Negotiating,
            version: 0,
            generation,
            next_port: 1,
            tx_sequence: 0,
            rx_sequence: u16::MAX,
            receive: Vec::new(),
            outgoing: VecDeque::new(),
            queued_bytes: 0,
            streams: Vec::new(),
            logs: VecDeque::new(),
            short_tcp: 0,
            backpressure: Backpressure::default(),
            observed_at: 0,
        };
        let mut version = Vec::new();
        version.extend_from_slice(&2u32.to_be_bytes());
        version.extend_from_slice(&0u32.to_be_bytes());
        version.extend_from_slice(&0u32.to_be_bytes());
        mux.emit(VERSION, &version, &[])
            .expect("one version frame fits the empty queue");
        mux
    }
    /// Invalidate every stream handle and discard all old wire work.
    pub fn reset(&mut self) -> Result<()> {
        let generation = self
            .generation
            .checked_add(1)
            .ok_or(refuse!("usbmux connection generation exhausted"))?;
        *self = Self::with_generation(generation);
        Ok(())
    }
    pub fn state(&self) -> State {
        self.state
    }
    pub fn backpressure(&self) -> Backpressure {
        self.backpressure
    }
    pub(super) fn observe_time(&mut self, now: u64) {
        self.observed_at = now;
    }
    pub(super) fn stream_stats(&self, id: StreamId) -> Option<StreamStats> {
        let stream = &self.streams[self.stream(id).ok()?];
        if stream.state != StreamState::Open {
            return None;
        }
        let waiting = stream
            .credit_wait_since
            .map_or(0, |since| self.observed_at.saturating_sub(since));
        Some(StreamStats {
            accepted_bytes: stream.accepted_bytes,
            acknowledged_bytes: stream.acknowledged_bytes,
            window_bytes: stream.window,
            outstanding_bytes: stream.outstanding(),
            zero_window_writes: stream.zero_window_writes,
            full_window_writes: stream.full_window_writes,
            credit_wait_ticks: stream.credit_wait_ticks.saturating_add(waiting),
            credit_waiting: stream.credit_wait_since.is_some(),
        })
    }
    pub fn take_packet(&mut self) -> Option<Vec<u8>> {
        let packet = self.outgoing.pop_front()?;
        self.queued_bytes -= packet.len();
        self.flush_replies();
        Some(packet)
    }
    /// Queue owed replies ahead of further writes as space frees up.
    fn flush_replies(&mut self) {
        for index in 0..self.streams.len() {
            let stream = &self.streams[index];
            if let Some(flags) = stream.reply
                && self.can_emit(self.header() + TCP_HEADER)
            {
                let next = stream.next;
                self.send_tcp(index, flags, next, &[])
                    .expect("space was checked");
            }
        }
    }
    pub fn take_log(&mut self) -> Option<Vec<u8>> {
        self.logs.pop_front()
    }
    fn header(&self) -> usize {
        if self.version == 2 {
            MUX2_HEADER
        } else {
            VERSION_HEADER
        }
    }
    fn can_emit(&self, bytes: usize) -> bool {
        self.outgoing.len() < MAX_QUEUED
            && bytes <= MAX_FRAME
            && self.queued_bytes + bytes <= MAX_QUEUED_BYTES
    }
    fn emit(&mut self, protocol: u32, prefix: &[u8], data: &[u8]) -> Result<()> {
        let header = self.header();
        let bytes = header + prefix.len() + data.len();
        if bytes > TX_MTU || !self.can_emit(bytes) {
            return Err(refuse!("usbmux transmit queue backpressure"));
        }
        let mut packet = Vec::with_capacity(bytes);
        packet.extend_from_slice(&protocol.to_be_bytes());
        packet.extend_from_slice(&(bytes as u32).to_be_bytes());
        if self.version == 2 {
            if protocol == SETUP {
                self.tx_sequence = 0;
                self.rx_sequence = u16::MAX;
            }
            packet.extend_from_slice(&MAGIC.to_be_bytes());
            packet.extend_from_slice(&self.tx_sequence.to_be_bytes());
            packet.extend_from_slice(&self.rx_sequence.to_be_bytes());
            self.tx_sequence = self.tx_sequence.wrapping_add(1);
        }
        packet.extend_from_slice(prefix);
        packet.extend_from_slice(data);
        self.outgoing.push_back(packet);
        self.queued_bytes += bytes;
        Ok(())
    }
    fn stream(&self, id: StreamId) -> Result<usize> {
        self.streams
            .iter()
            .position(|stream| stream.id == id)
            .ok_or(refuse!("unknown or stale usbmux stream"))
    }
    pub fn stream_state(&self, id: StreamId) -> Result<StreamState> {
        Ok(self.streams[self.stream(id)?].state)
    }
    /// Space for one SYN/RST/window ACK. Native poll adapters check before
    /// open/read/close, whose original APIs report queue backpressure as Err.
    pub fn control_ready(&self) -> bool {
        self.can_emit(self.header() + TCP_HEADER)
    }
    pub fn open(&mut self, destination: u16) -> Result<StreamId> {
        if self.state != State::Active || destination == 0 {
            return Err(refuse!("usbmux is not active or port is zero"));
        }
        if self.streams.len() == MAX_STREAMS {
            return Err(refuse!(
                "usbmux stream capacity exhausted; release closed streams"
            ));
        }
        let port = self.next_port;
        if port == 0 {
            return Err(refuse!(
                "usbmux source port space exhausted; reconnect required"
            ));
        }
        if !self.can_emit(self.header() + TCP_HEADER) {
            return Err(refuse!("usbmux transmit queue backpressure"));
        }
        let id = StreamId {
            generation: self.generation,
            port,
        };
        self.streams.push(Stream {
            id,
            destination,
            state: StreamState::Connecting,
            next: 1,
            acknowledged: 0,
            received: 0,
            window: 0,
            input: VecDeque::new(),
            accepted_bytes: 0,
            acknowledged_bytes: 0,
            zero_window_writes: 0,
            full_window_writes: 0,
            credit_wait_since: None,
            credit_wait_ticks: 0,
            reply: None,
        });
        let index = self.streams.len() - 1;
        // The SYN sequence is zero. As in usbmuxd, it consumes one byte.
        self.send_tcp(index, SYN, 0, &[])?;
        self.next_port = port.wrapping_add(1);
        Ok(id)
    }
    fn send_tcp(&mut self, index: usize, flags: u8, sequence: u32, data: &[u8]) -> Result<()> {
        let stream = &self.streams[index];
        let header = tcp_header(
            stream.id.port,
            stream.destination,
            sequence,
            stream.received,
            flags,
            ((RX_WINDOW - stream.input.len()) >> 8) as u16,
        );
        self.emit(TCP, &header, data)?;
        // Every segment carries the current ACK, and an RST ends the stream.
        self.streams[index].reply = None;
        Ok(())
    }
    /// Accepts a bounded prefix according to the peer's scaled byte window.
    /// No hidden unbounded application send queue exists.
    pub fn write(&mut self, id: StreamId, bytes: &[u8]) -> Result<usize> {
        let index = self.stream(id)?;
        let stream = &self.streams[index];
        if stream.state != StreamState::Open {
            return Err(refuse!("usbmux stream is not open"));
        }
        let available = stream.window.saturating_sub(stream.outstanding()) as usize;
        let count = bytes
            .len()
            .min(available)
            .min(TX_MTU - self.header() - TCP_HEADER);
        if bytes.is_empty() {
            return Ok(0);
        }
        if count == 0 {
            self.backpressure.window += 1;
            let stream = &mut self.streams[index];
            if stream.window == 0 {
                stream.zero_window_writes = stream.zero_window_writes.saturating_add(1);
            } else {
                stream.full_window_writes = stream.full_window_writes.saturating_add(1);
            }
            stream.credit_wait_since.get_or_insert(self.observed_at);
            return Ok(0);
        }
        if !self.can_emit(self.header() + TCP_HEADER + count) {
            self.backpressure.queue += 1;
            return Ok(0);
        }
        let sequence = stream.next;
        self.send_tcp(index, ACK, sequence, &bytes[..count])?;
        let stream = &mut self.streams[index];
        stream.next = sequence.wrapping_add(count as u32);
        stream.accepted_bytes = stream.accepted_bytes.saturating_add(count as u64);
        Ok(count)
    }
    /// Reading reopens the receive window by a real TCP ACK. Backpressure
    /// refuses before draining the stream, so the caller can retry later.
    pub fn read(&mut self, id: StreamId, output: &mut [u8]) -> Result<usize> {
        let index = self.stream(id)?;
        let stream = &self.streams[index];
        let count = output.len().min(stream.input.len());
        if count == 0 {
            return Ok(0);
        }
        if !self.can_emit(self.header() + TCP_HEADER) {
            return Err(refuse!("usbmux receive-window ACK backpressure"));
        }
        for byte in &mut output[..count] {
            *byte = self.streams[index].input.pop_front().unwrap();
        }
        self.send_tcp(index, ACK, self.streams[index].next, &[])?;
        Ok(count)
    }
    pub fn close(&mut self, id: StreamId) -> Result<()> {
        let index = self.stream(id)?;
        if matches!(
            self.streams[index].state,
            StreamState::Connecting | StreamState::Open
        ) {
            self.send_tcp(index, RST, self.streams[index].next, &[])?;
            self.streams[index].state = StreamState::Closed;
        }
        Ok(())
    }
    pub fn release(&mut self, id: StreamId) -> Result<()> {
        let index = self.stream(id)?;
        if matches!(
            self.streams[index].state,
            StreamState::Connecting | StreamState::Open
        ) {
            return Err(refuse!("close usbmux stream before releasing it"));
        }
        self.streams.remove(index);
        Ok(())
    }
    /// Consumes fragmented/coalesced bulk bytes, including empty USB ZLPs.
    /// Parsing is length-bounded before reserving space. A protocol error
    /// terminates this connection; reset is the only recovery path.
    pub fn receive(&mut self, bytes: &[u8]) -> Result<()> {
        if self.state == State::Failed {
            return Err(refuse!("usbmux connection requires reset"));
        }
        let result = self.receive_inner(bytes);
        if result.is_err() {
            self.state = State::Failed;
        }
        self.flush_replies();
        result
    }
    fn frame_length(&self, bytes: &[u8]) -> Result<usize> {
        let length = u32::from_be_bytes(bytes[4..8].try_into().unwrap()) as usize;
        if length < self.header() || length > MAX_FRAME {
            return Err(refuse!("invalid usbmux frame length"));
        }
        Ok(length)
    }
    fn receive_inner(&mut self, mut bytes: &[u8]) -> Result<()> {
        while !bytes.is_empty() {
            // ACKs and other complete frames can be handled directly in
            // the bulk transfer that delivered them.
            if self.receive.is_empty() && bytes.len() >= VERSION_HEADER {
                let length = self.frame_length(bytes)?;
                if let Some((packet, rest)) = bytes.split_at_checked(length) {
                    self.input(packet)?;
                    bytes = rest;
                    continue;
                }
            }
            if self.receive.len() < VERSION_HEADER {
                let missing = VERSION_HEADER - self.receive.len();
                let (header, rest) = bytes.split_at(missing.min(bytes.len()));
                self.receive.extend_from_slice(header);
                bytes = rest;
                if self.receive.len() < VERSION_HEADER {
                    break;
                }
            }
            let length = self.frame_length(&self.receive)?;
            let missing = length - self.receive.len();
            let (fragment, rest) = bytes.split_at(missing.min(bytes.len()));
            self.receive.extend_from_slice(fragment);
            bytes = rest;
            if self.receive.len() == length {
                let mut packet = core::mem::take(&mut self.receive);
                let result = self.input(&packet);
                packet.clear();
                self.receive = packet;
                result?;
            }
        }
        Ok(())
    }
    fn input(&mut self, packet: &[u8]) -> Result<()> {
        let protocol = u32::from_be_bytes(packet[..4].try_into().unwrap());
        let header = self.header();
        if self.version == 2 {
            // Match the pinned usbmuxd, which sets the magic only on what it
            // sends (device.c:222) and ignores it on input: the 26A434 device
            // mux sends 0 there. The reply echo is the incoming rx_seq field
            // (device.c:797), not its tx_seq field.
            self.rx_sequence = u16::from_be_bytes(packet[14..16].try_into().unwrap());
        }
        let payload = &packet[header..];
        match protocol {
            VERSION if self.state == State::Negotiating => {
                if payload.len() != 12 {
                    return Err(refuse!("invalid usbmux version payload length"));
                }
                // The pinned usbmuxd checks only the major version
                // (device.c `device_version_input`): the 26A434 device mux
                // leaves the padding word uninitialized.
                let major = u32::from_be_bytes(payload[..4].try_into().unwrap());
                if !matches!(major, 1 | 2) {
                    return Err(refuse!("unsupported usbmux version"));
                }
                self.version = major;
                if major == 2 {
                    self.emit(SETUP, &[7], &[])?;
                }
                self.state = State::Active;
            }
            CONTROL if self.state == State::Active => {
                if self.logs.len() == 8 || payload.len() > 4096 {
                    return Err(refuse!("usbmux control-log capacity exceeded"));
                }
                self.logs.push_back(payload.to_vec());
            }
            TCP if self.state == State::Active => self.input_tcp(payload)?,
            _ => return Err(refuse!("unexpected usbmux protocol or connection state")),
        }
        Ok(())
    }
    /// TCP input as usbmuxd's device_data_input and device_tcp_input take
    /// it: the data follows a fixed 20-byte header whatever its data offset
    /// says, and neither the offset nor the urgent pointer is checked. A
    /// packet shorter than the header is dropped.
    fn input_tcp(&mut self, payload: &[u8]) -> Result<()> {
        if payload.len() < TCP_HEADER {
            self.short_tcp += 1;
            return Ok(());
        }
        let source = u16::from_be_bytes(payload[..2].try_into().unwrap());
        let destination = u16::from_be_bytes(payload[2..4].try_into().unwrap());
        let sequence = u32::from_be_bytes(payload[4..8].try_into().unwrap());
        let acknowledgement = u32::from_be_bytes(payload[8..12].try_into().unwrap());
        let flags = payload[13];
        let window = u32::from(u16::from_be_bytes(payload[14..16].try_into().unwrap())) << 8;
        let data = &payload[TCP_HEADER..];
        let Some(index) = self
            .streams
            .iter()
            .position(|stream| stream.id.port == destination && stream.destination == source)
        else {
            // With the queue full the RST is dropped, as a lost segment
            // would be; the peer's next segment draws another.
            if flags & RST == 0 && self.can_emit(self.header() + TCP_HEADER) {
                self.emit(
                    TCP,
                    &tcp_header(destination, source, 0, sequence, RST, 0),
                    &[],
                )?;
            }
            return Ok(());
        };
        let stream = &mut self.streams[index];
        let connecting = stream.state == StreamState::Connecting;
        if flags & RST != 0 {
            stream.reply = None;
            stream.state = if stream.state == StreamState::Connecting {
                StreamState::Refused
            } else {
                StreamState::Reset
            };
            return Ok(());
        }
        if matches!(
            stream.state,
            StreamState::Closed | StreamState::Reset | StreamState::Refused
        ) {
            return Ok(());
        }
        let acknowledged = acknowledgement.wrapping_sub(stream.acknowledged);
        if acknowledged > stream.outstanding() {
            return Err(refuse!("usbmux TCP ACK outside sent sequence range"));
        }
        // Flags other than the state's one expected value tear the stream
        // down with an RST, as usbmuxd's connection_teardown does.
        let expected = if connecting { SYN | ACK } else { ACK };
        if flags != expected {
            stream.state = StreamState::Reset;
            stream.reply = Some(RST);
            return Ok(());
        }
        if stream.state == StreamState::Connecting {
            if acknowledgement != 1 || sequence != 0 || !data.is_empty() {
                return Err(refuse!("invalid usbmux SYN-ACK"));
            }
            stream.received = 1;
            stream.state = StreamState::Open;
        } else {
            if sequence != stream.received {
                return Err(refuse!("out-of-order usbmux TCP byte sequence"));
            }
            if data.len() > RX_WINDOW - stream.input.len() {
                return Err(refuse!("usbmux peer exceeded receive window"));
            }
            stream.input.extend(data.iter().copied());
            stream.received = stream.received.wrapping_add(data.len() as u32);
            stream.acknowledged_bytes = stream
                .acknowledged_bytes
                .saturating_add(u64::from(acknowledged));
        }
        stream.acknowledged = acknowledgement;
        stream.window = window;
        if window > stream.outstanding()
            && let Some(since) = stream.credit_wait_since.take()
        {
            stream.credit_wait_ticks = stream
                .credit_wait_ticks
                .saturating_add(self.observed_at.saturating_sub(since));
        }
        self.backpressure.largest_window = self.backpressure.largest_window.max(window);
        if connecting || !data.is_empty() {
            stream.reply = Some(ACK);
        }
        Ok(())
    }
}
fn tcp_header(
    source: u16,
    destination: u16,
    sequence: u32,
    acknowledgement: u32,
    flags: u8,
    window: u16,
) -> [u8; TCP_HEADER] {
    let mut header = [0; TCP_HEADER];
    header[..2].copy_from_slice(&source.to_be_bytes());
    header[2..4].copy_from_slice(&destination.to_be_bytes());
    header[4..8].copy_from_slice(&sequence.to_be_bytes());
    header[8..12].copy_from_slice(&acknowledgement.to_be_bytes());
    header[12] = 5 << 4;
    header[13] = flags;
    header[14..16].copy_from_slice(&window.to_be_bytes());
    header
}

/// A mux connected to lockdownd's port, for clients driven through it.
#[cfg(any(test, feature = "test-support"))]
pub mod test_support {
    use super::*;
    use alloc::vec;
    pub(crate) fn wire(protocol: u32, payload: &[u8], version: u32) -> Vec<u8> {
        let mut mux = Mux::new();
        mux.outgoing.clear();
        mux.queued_bytes = 0;
        mux.version = version;
        mux.emit(protocol, payload, &[]).unwrap();
        mux.take_packet().unwrap()
    }
    pub(crate) fn version(major: u32) -> Vec<u8> {
        let mut data = vec![0; 12];
        data[..4].copy_from_slice(&major.to_be_bytes());
        wire(VERSION, &data, 0)
    }
    pub fn connected() -> (Mux, StreamId) {
        let mut mux = Mux::new();
        mux.take_packet();
        mux.receive(&version(2)).unwrap();
        mux.take_packet();
        let id = mux.open(62078).unwrap();
        mux.take_packet();
        let mut reply = wire(TCP, &tcp_header(62078, id.port, 0, 1, SYN | ACK, 256), 2);
        reply[14..16].copy_from_slice(&0u16.to_be_bytes());
        mux.receive(&reply).unwrap();
        mux.take_packet();
        (mux, id)
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::{connected, version, wire};
    use super::*;
    #[test]
    fn tcp_input_follows_usbmuxd_header_and_flag_handling() {
        const FIN: u8 = 1;
        // A packet shorter than a TCP header is dropped, as usbmuxd drops it.
        let (mut mux, id) = connected();
        mux.receive(&wire(TCP, &[0; 12], 2)).unwrap();
        assert_eq!(mux.short_tcp, 1);
        assert_eq!(mux.state(), State::Active);
        // The data offset and urgent pointer are not checked: data follows
        // the fixed 20-byte header.
        let mut header = tcp_header(62078, id.port, 1, 1, ACK, 256);
        header[12] = 0;
        header[18..20].copy_from_slice(&[0, 9]);
        let mut packet = header.to_vec();
        packet.extend_from_slice(b"ok");
        mux.receive(&wire(TCP, &packet, 2)).unwrap();
        mux.take_packet();
        let mut read = [0; 2];
        assert_eq!(mux.read(id, &mut read), Ok(2));
        assert_eq!(&read, b"ok");
        // Any flags but ACK on an open stream tear it down with an RST.
        while mux.take_packet().is_some() {}
        mux.receive(&wire(
            TCP,
            &tcp_header(62078, id.port, 3, 1, FIN | ACK, 256),
            2,
        ))
        .unwrap();
        let reset = mux.take_packet().unwrap();
        assert_eq!(reset[MUX2_HEADER + 13], RST);
        assert_eq!(mux.stream_state(id), Ok(StreamState::Reset));
    }
    #[test]
    fn pinned_wire_vectors_version_setup_syn_and_scaled_window() {
        let mut mux = Mux::new();
        assert_eq!(
            mux.take_packet().unwrap(),
            [0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0]
        );
        // The device's padding word is not its version.
        let mut reply = version(2);
        let padding = reply.len() - 4;
        reply[padding] = 0x1a;
        for byte in reply {
            mux.receive(&[byte]).unwrap();
        }
        assert_eq!(
            mux.take_packet().unwrap(),
            [
                0, 0, 0, 2, 0, 0, 0, 17, 0xfe, 0xed, 0xfa, 0xce, 0, 0, 0xff, 0xff, 7
            ]
        );
        let id = mux.open(62078).unwrap();
        let syn = mux.take_packet().unwrap();
        assert_eq!(&syn[16..20], &[0, 1, 0xf2, 0x7e]);
        assert_eq!(&syn[28..32], &[0x50, 2, 1, 0]);
        assert_eq!(mux.stream_state(id), Ok(StreamState::Connecting));
    }
    #[test]
    fn simultaneous_ports_fragmentation_coalescing_and_rst() {
        let (mut mux, id) = connected();
        assert_eq!(mux.write(id, b"hello"), Ok(5));
        mux.take_packet();
        let other = mux.open(12345).unwrap();
        mux.take_packet();
        let mut body = tcp_header(62078, id.port, 1, 6, ACK, 0).to_vec();
        body.extend_from_slice(b"reply");
        let mut bytes = wire(TCP, &body, 2);
        bytes.extend(wire(TCP, &tcp_header(12345, other.port, 0, 0, RST, 0), 2));
        for chunk in bytes.chunks(3) {
            mux.receive(chunk).unwrap();
        }
        assert_eq!(mux.write(id, b"blocked"), Ok(0));
        let mut out = [0; 20];
        assert_eq!(mux.read(id, &mut out), Ok(5));
        assert_eq!(&out[..5], b"reply");
        assert_eq!(mux.stream_state(other), Ok(StreamState::Refused));
        mux.reset().unwrap();
        assert!(mux.stream_state(id).is_err());
    }
    #[test]
    fn complete_and_split_frames_switch_header_sizes_during_negotiation() {
        let mut bytes = version(2);
        bytes.extend(wire(CONTROL, b"first", 2));
        bytes.extend(wire(CONTROL, b"second", 2));
        for split in 0..=bytes.len() {
            let mut mux = Mux::new();
            mux.take_packet().unwrap();
            mux.receive(&bytes[..split]).unwrap();
            mux.receive(&bytes[split..]).unwrap();
            assert_eq!(mux.state(), State::Active);
            assert_eq!(mux.take_packet().unwrap()[MUX2_HEADER..], [7]);
            assert_eq!(mux.take_log().as_deref(), Some(b"first".as_slice()));
            assert_eq!(mux.take_log().as_deref(), Some(b"second".as_slice()));
            assert!(mux.receive.is_empty());
        }
    }
    #[test]
    fn sequence_wrap_windows_and_close() {
        let (mut mux, id) = connected();
        let stream = &mut mux.streams[0];
        stream.next = u32::MAX - 1;
        stream.acknowledged = u32::MAX - 1;
        stream.received = u32::MAX;
        assert_eq!(mux.write(id, b"wrap"), Ok(4));
        mux.take_packet();
        let mut payload = tcp_header(62078, id.port, u32::MAX, 2, ACK, 1).to_vec();
        payload.extend_from_slice(b"ab");
        mux.receive(&wire(TCP, &payload, 2)).unwrap();
        assert_eq!(mux.streams[0].received, 1);
        assert_eq!(mux.streams[0].acknowledged, 2);
        let stats = mux.stream_stats(id).unwrap();
        assert_eq!(stats.accepted_bytes, 4);
        assert_eq!(stats.acknowledged_bytes, 4);
        assert_eq!(stats.outstanding_bytes, 0);
        mux.tx_sequence = u16::MAX;
        mux.take_packet();
        mux.write(id, b"a").unwrap();
        mux.write(id, b"b").unwrap();
        assert_eq!(&mux.take_packet().unwrap()[12..14], &u16::MAX.to_be_bytes());
        assert_eq!(&mux.take_packet().unwrap()[12..14], &[0, 0]);
        mux.close(id).unwrap();
        assert_eq!(mux.stream_state(id), Ok(StreamState::Closed));
        mux.release(id).unwrap();
        assert!(mux.read(id, &mut [0]).is_err());
    }
    #[test]
    fn malformed_lengths_and_acks_fail_connection_but_magic_is_ignored() {
        for length in [0, 7, (MAX_FRAME + 1) as u32, u32::MAX] {
            let mut mux = Mux::new();
            let mut bytes = [0; 8];
            bytes[4..].copy_from_slice(&length.to_be_bytes());
            assert!(mux.receive(&bytes).is_err());
            assert_eq!(mux.state(), State::Failed);
        }
        let (mut mux, id) = connected();
        assert!(
            mux.receive(&wire(TCP, &tcp_header(62078, id.port, 1, 123, ACK, 1), 2))
                .is_err()
        );
        // The device's magic field is not checked: 26A434 sends 0.
        let (mut mux, _) = connected();
        let mut zero = wire(CONTROL, b"log", 2);
        zero[8..12].fill(0);
        mux.receive(&zero).unwrap();
        let mut mux = Mux::new();
        mux.take_packet();
        mux.receive(&version(1)).unwrap();
        assert!(mux.take_packet().is_none());
        assert!(mux.open(62078).is_ok());
    }
    #[test]
    fn unknown_ports_get_rst_and_capacity_is_bounded() {
        let (mut mux, _) = connected();
        mux.receive(&wire(TCP, &tcp_header(1082, 99, 123, 0, SYN, 1), 2))
            .unwrap();
        let reset = mux.take_packet().unwrap();
        assert_eq!(reset[29], RST);
        assert_eq!(&reset[24..28], &123u32.to_be_bytes());
        for _ in 1..MAX_STREAMS {
            mux.open(62078).unwrap();
        }
        assert!(mux.open(62078).is_err());
        while mux.emit(CONTROL, b"log", &[]).is_ok() {}
        assert!(mux.queued_bytes <= MAX_QUEUED_BYTES);
        assert!(mux.outgoing.len() <= MAX_QUEUED);
    }
    #[test]
    fn backpressure_names_the_window_or_the_queue_that_refused_a_write() {
        let (mut mux, id) = connected();
        assert_eq!(mux.backpressure().largest_window, 256 << 8);
        mux.streams[0].window = 4;
        assert_eq!(mux.write(id, b"abcdef"), Ok(4));
        assert_eq!(mux.write(id, b"ef"), Ok(0));
        assert_eq!(mux.write(id, b""), Ok(0));
        mux.streams[0].window = u32::MAX;
        while mux.write(id, &[0; 4096]).unwrap() != 0 {}
        let backpressure = mux.backpressure();
        assert_eq!((backpressure.window, backpressure.queue), (1, 1));
    }

    #[test]
    fn stream_stats_time_credit_waits_and_exclude_other_streams() {
        let (mut mux, id) = connected();
        let other = mux.open(12345).unwrap();
        mux.take_packet().unwrap();
        mux.receive(&wire(
            TCP,
            &tcp_header(12345, other.port, 0, 1, SYN | ACK, 1),
            2,
        ))
        .unwrap();
        mux.take_packet().unwrap();
        assert_eq!(mux.stream_stats(other).unwrap().acknowledged_bytes, 0);

        let ack = |mux: &mut Mux, sequence, window| {
            mux.receive(&wire(
                TCP,
                &tcp_header(62078, id.port, 1, sequence, ACK, window),
                2,
            ))
            .unwrap();
        };
        ack(&mut mux, 1, 0);
        mux.observe_time(10);
        assert_eq!(mux.write(id, b"blocked"), Ok(0));
        mux.observe_time(13);
        assert_eq!(mux.write(id, b"retry"), Ok(0));
        assert_eq!(mux.write(other, b"other"), Ok(5));
        mux.take_packet().unwrap();
        let waiting = mux.stream_stats(id).unwrap();
        assert_eq!(waiting.zero_window_writes, 2);
        assert_eq!(waiting.credit_wait_ticks, 3);
        assert!(waiting.credit_waiting);

        mux.observe_time(22);
        ack(&mut mux, 1, 1); // A duplicate ACK changes just the advertised window.
        let ready = mux.stream_stats(id).unwrap();
        assert_eq!(ready.acknowledged_bytes, 0);
        assert_eq!(ready.credit_wait_ticks, 12);
        assert!(!ready.credit_waiting);
        let data = [0x3c; 256];
        assert_eq!(mux.write(id, &data), Ok(data.len()));
        let packet = mux.take_packet().unwrap();
        assert_eq!(&packet[MUX2_HEADER + TCP_HEADER..], &data);

        mux.observe_time(25);
        assert_eq!(mux.write(id, b"blocked again"), Ok(0));
        mux.observe_time(31);
        ack(&mut mux, 129, 1);
        assert_eq!(
            mux.stream_stats(id),
            Some(StreamStats {
                accepted_bytes: 256,
                acknowledged_bytes: 128,
                window_bytes: 256,
                outstanding_bytes: 128,
                zero_window_writes: 2,
                full_window_writes: 1,
                credit_wait_ticks: 18,
                credit_waiting: false,
            })
        );
        let other = mux.stream_stats(other).unwrap();
        assert_eq!(other.accepted_bytes, 5);
        assert_eq!(other.acknowledged_bytes, 0);
        assert_eq!(other.credit_wait_ticks, 0);
    }
    #[test]
    fn replies_owed_while_writes_fill_the_queue_are_coalesced_not_fatal() {
        let (mut mux, id) = connected();
        mux.streams[0].window = u32::MAX;
        while mux.write(id, &[0; 4096]).unwrap() != 0 {}
        while mux.write(id, b"x").unwrap() != 0 {}
        while mux.emit(CONTROL, &[], &[]).is_ok() {}
        assert!(!mux.control_ready());
        let sent = mux.streams[0].next;
        for sequence in [1, 3] {
            let mut segment = tcp_header(62078, id.port, sequence, sent, ACK, 256).to_vec();
            segment.extend_from_slice(b"ok");
            mux.receive(&wire(TCP, &segment, 2)).unwrap();
        }
        assert_eq!(mux.state(), State::Active);
        // The first freed space carries one bare ACK covering both segments,
        // queued ahead of any further write.
        mux.take_packet();
        let ack = &mux.outgoing.back().unwrap()[MUX2_HEADER..];
        assert_eq!(ack.len(), TCP_HEADER);
        assert_eq!(&ack[8..12], &5u32.to_be_bytes());
        assert_eq!(ack[13], ACK);
        while mux.take_packet().is_some() {}
        assert!(mux.outgoing.is_empty());
    }
    #[test]
    fn complete_data_and_reset_vectors_match_executed_pinned_usbmuxd() {
        let (mut mux, id) = connected();
        assert_eq!(mux.write(id, b"hello"), Ok(5));
        let data = mux.take_packet().unwrap();
        assert_eq!(
            hex::encode(data),
            "0000000600000029feedface000300000001f27e0000000100000001501001000000000068656c6c6f"
        );
        mux.streams[0].next = u32::MAX;
        mux.tx_sequence = u16::MAX;
        mux.close(id).unwrap();
        assert_eq!(
            hex::encode(mux.take_packet().unwrap()),
            "0000000600000024feedfaceffff00000001f27effffffff000000015004010000000000"
        );
    }
}
