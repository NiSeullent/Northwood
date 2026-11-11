//! Joins the USB and mux state machines with bounded queues. The machine
//! owner supplies one bus step; clients only enqueue/read bytes and cannot
//! recursively enter translated code while device memory is borrowed.
use super::{
    BusPeer, MAX_PACKET,
    host::{Descriptor, Host, TransferStatus},
    mux::{Backpressure, Mux, State, StreamId, StreamState, StreamStats},
    plist::{self, Channel},
};
use crate::Result;
use alloc::{collections::VecDeque, vec::Vec};
use stela_contracts::refuse;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Log {
    Descriptor(Descriptor),
    MuxControl(Vec<u8>),
    QueryTypeReply(Vec<u8>),
    /// The leading bytes of a bulk IN transfer the mux refused, so the
    /// refusal that follows can be checked against the device's framing.
    MuxRejected(Vec<u8>),
}

/// Bytes of a refused transfer kept for [`Log::MuxRejected`].
const REJECTED_BYTES: usize = 96;
struct Probe {
    stream: Option<StreamId>,
    request: Vec<u8>,
    sent: usize,
    reply: Channel,
}
pub struct Session {
    host: Host,
    mux: Mux,
    generation: Option<u64>,
    descriptors_logged: bool,
    input: Option<u64>,
    output: Option<u64>,
    logs: VecDeque<Log>,
    probe: Option<Probe>,
    /// A bulk transfer moved bytes since [`Session::take_due`]: the client
    /// may have bytes to read or room to write.
    due: bool,
}
impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}
impl Session {
    #[cfg(any(test, feature = "test-support"))]
    pub fn test_mux(mux: Mux) -> Self {
        Self { mux, ..Self::new() }
    }
    pub fn new() -> Self {
        Self {
            host: Host::new(),
            mux: Mux::new(),
            generation: None,
            descriptors_logged: false,
            input: None,
            output: None,
            logs: VecDeque::new(),
            probe: None,
            due: false,
        }
    }
    pub fn attach(&mut self, generation: u64) -> Result<()> {
        if self.generation == Some(generation) {
            return Ok(());
        }
        self.detach()?;
        self.generation = Some(generation);
        self.host.attach(generation);
        Ok(())
    }
    pub fn detach(&mut self) -> Result<()> {
        self.detach_reset(Mux::reset)
    }

    fn detach_reset(&mut self, reset_mux: impl FnOnce(&mut Mux) -> Result<()>) -> Result<()> {
        if self.generation.is_some() {
            // Failed invalidation must remain required on the next attempt;
            // an old connection can never become a successful detach retry.
            reset_mux(&mut self.mux)?;
            self.generation = None;
        }
        self.host.detach();
        while self.host.completion().is_some() {}
        self.descriptors_logged = false;
        self.input = None;
        self.output = None;
        if let Some(probe) = self.probe.as_mut() {
            probe.stream = None;
            probe.sent = 0;
            probe.reply = Channel::default();
        }
        Ok(())
    }
    pub fn open(&mut self, port: u16) -> Result<StreamId> {
        self.mux.open(port)
    }
    /// Negotiation readiness, without opening a speculative stream.
    pub fn mux_state(&self) -> State {
        self.mux.state()
    }
    pub fn control_ready(&self) -> bool {
        self.mux.control_ready()
    }
    pub fn backpressure(&self) -> Backpressure {
        self.mux.backpressure()
    }
    pub fn stream_stats(&self, stream: StreamId) -> Option<StreamStats> {
        self.mux.stream_stats(stream)
    }
    pub fn state(&self, stream: StreamId) -> Result<StreamState> {
        self.mux.stream_state(stream)
    }
    pub fn write(&mut self, stream: StreamId, bytes: &[u8]) -> Result<usize> {
        self.mux.write(stream, bytes)
    }
    pub fn read(&mut self, stream: StreamId, bytes: &mut [u8]) -> Result<usize> {
        self.mux.read(stream, bytes)
    }
    pub fn close(&mut self, stream: StreamId) -> Result<()> {
        self.mux.close(stream)
    }
    pub fn release(&mut self, stream: StreamId) -> Result<()> {
        self.mux.release(stream)
    }
    /// Whether a bulk transfer moved bytes since the last call.
    pub fn take_due(&mut self) -> bool {
        core::mem::take(&mut self.due)
    }
    pub fn take_log(&mut self) -> Option<Log> {
        self.logs.pop_front()
    }
    /// An explicitly requested diagnostic using the same public stream path
    /// as a later restore client. It records the guest's opaque plist bytes.
    pub fn query_type(&mut self) -> Result<()> {
        if self.probe.is_some() {
            return Err(refuse!("QueryType probe already pending"));
        }
        self.probe = Some(Probe {
            stream: None,
            request: plist::frame(plist::QUERY_TYPE)?,
            sent: 0,
            reply: Channel::default(),
        });
        Ok(())
    }
    /// One packet on the bus. Whether another call at the same `now` can
    /// move one ([`Host::advance`]), or this call submitted a transfer.
    pub fn advance(&mut self, peer: &mut impl BusPeer, now: u64) -> Result<bool> {
        if self.generation.is_none() {
            return Ok(false);
        }
        self.mux.observe_time(now);
        // Log backpressure cannot discard a diagnostic or grow without bound.
        if self.logs.len() >= 32 {
            return Ok(false);
        }
        let moved = self.host.advance(peer, now)?;
        while let Some(completion) = self.host.completion() {
            let input = self.input == Some(completion.id);
            let output = self.output == Some(completion.id);
            if !input && !output {
                return Err(refuse!("USB session received a stale completion"));
            }
            if input {
                self.input = None;
            } else {
                self.output = None;
            }
            match completion.status {
                TransferStatus::Complete => {
                    if input && let Err(error) = self.mux.receive(&completion.bytes) {
                        let kept = completion.bytes.len().min(REJECTED_BYTES);
                        self.logs
                            .push_back(Log::MuxRejected(completion.bytes[..kept].to_vec()));
                        return Err(error);
                    }
                    self.due |= output || !completion.bytes.is_empty();
                }
                status => {
                    return Err(refuse!(
                        "USB mux bulk transfer failed",
                        ?status,
                        input,
                        transferred = completion.transferred
                    ));
                }
            }
        }
        if !self.host.ready() {
            return Ok(moved);
        }
        if !self.descriptors_logged {
            for descriptor in self.host.descriptors() {
                self.logs.push_back(Log::Descriptor(descriptor.clone()));
            }
            self.descriptors_logged = true;
        }
        while let Some(log) = self.mux.take_log() {
            self.logs.push_back(Log::MuxControl(log));
        }
        self.advance_probe()?;
        let mut submitted = false;
        if self.input.is_none() {
            self.input = Some(self.host.submit_in(MAX_PACKET)?);
            submitted = true;
        }
        if self.output.is_none()
            && let Some(packet) = self.mux.take_packet()
        {
            // A full-size last USB packet is followed by a ZLP; the mux
            // frame length remains independent of that USB terminator.
            let terminate = packet.len().is_multiple_of(MAX_PACKET);
            self.output = Some(self.host.submit_out_owned(packet, terminate)?);
            submitted = true;
        }
        Ok(moved || submitted)
    }
    fn advance_probe(&mut self) -> Result<()> {
        if self.mux.state() != State::Active {
            return Ok(());
        }
        let Some(probe) = self.probe.as_mut() else {
            return Ok(());
        };
        let stream = match probe.stream {
            Some(id) => id,
            None => {
                let id = self.mux.open(62078)?;
                probe.stream = Some(id);
                id
            }
        };
        match self.mux.stream_state(stream)? {
            StreamState::Connecting => return Ok(()),
            StreamState::Open => {}
            _ => return Err(refuse!("QueryType stream was refused or reset")),
        }
        if probe.sent < probe.request.len() {
            probe.sent += self.mux.write(stream, &probe.request[probe.sent..])?;
        }
        let mut bytes = [0; MAX_PACKET];
        let count = self.mux.read(stream, &mut bytes)?;
        if count != 0 {
            let (consumed, reply) = probe.reply.receive(&bytes[..count])?;
            if consumed != count {
                return Err(refuse!("unexpected bytes following QueryType reply"));
            }
            if let Some(reply) = reply {
                self.mux.close(stream)?;
                self.mux.release(stream)?;
                self.logs.push_back(Log::QueryTypeReply(reply));
                self.probe = None;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    #[test]
    fn detach_retries_failed_invalidation_and_success_invalidates_stale_handles_once() {
        use crate::usb::mux::test_support::connected;
        let (mux, stream) = connected();
        let mut session = Session::test_mux(mux);
        session.attach(7).unwrap();
        assert_eq!(session.state(stream), Ok(StreamState::Open));
        assert_eq!(
            session.detach_reset(|_| Err(refuse!("injected reset refusal"))),
            Err(refuse!("injected reset refusal"))
        );
        assert_eq!(session.generation, Some(7));
        assert_eq!(session.state(stream), Ok(StreamState::Open));
        session.detach().unwrap();
        assert_eq!(session.generation, None);
        assert!(session.state(stream).is_err());
        assert!(session.mux.take_packet().is_some()); // One fresh Version frame.
        assert!(session.mux.take_packet().is_none());
        session.detach().unwrap();
        assert!(session.mux.take_packet().is_none()); // No second invalidation.
        assert!(session.state(stream).is_err());
    }

    /// Synthetic mux wire fixture, never represented as an observed guest.
    fn packet(protocol: u32, payload: &[u8], v2: bool) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&protocol.to_be_bytes());
        bytes.extend_from_slice(&((if v2 { 16 } else { 8 }) + payload.len() as u32).to_be_bytes());
        if v2 {
            bytes.extend_from_slice(&0xfeed_faceu32.to_be_bytes());
            bytes.extend_from_slice(&[0, 0, 0, 0]);
        }
        bytes.extend_from_slice(payload);
        bytes
    }
    #[test]
    fn query_type_uses_port_62078_and_logs_only_received_plist() {
        let mut session = Session::new();
        session.mux.take_packet();
        session.query_type().unwrap();
        session
            .mux
            .receive(&packet(0, &[0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0], false))
            .unwrap();
        session.mux.take_packet();
        session.advance_probe().unwrap();
        let syn = session.mux.take_packet().unwrap();
        assert_eq!(&syn[18..20], &62078u16.to_be_bytes());
        let mut tcp = vec![0; 20];
        tcp[..2].copy_from_slice(&62078u16.to_be_bytes());
        tcp[2..4].copy_from_slice(&syn[16..18]);
        tcp[11] = 1;
        tcp[12] = 0x50;
        tcp[13] = 0x12;
        tcp[14] = 1;
        session.mux.receive(&packet(6, &tcp, true)).unwrap();
        session.mux.take_packet();
        session.advance_probe().unwrap();
        let query = session.mux.take_packet().unwrap();
        assert_eq!(&query[36..], &plist::frame(plist::QUERY_TYPE).unwrap());
        let response=b"<plist version=\"1.0\"><dict><key>Type</key><string>com.apple.mobile.restored</string><key>RestoreProtocolVersion</key><integer>15</integer></dict></plist>";
        tcp[4..8].copy_from_slice(&1u32.to_be_bytes());
        tcp[8..12].copy_from_slice(&((query.len() - 36 + 1) as u32).to_be_bytes());
        tcp[13] = 0x10;
        tcp.extend_from_slice(&plist::frame(response).unwrap());
        session.mux.receive(&packet(6, &tcp, true)).unwrap();
        session.advance_probe().unwrap();
        assert_eq!(
            session.take_log(),
            Some(Log::QueryTypeReply(response.to_vec()))
        );
        assert!(session.probe.is_none());
    }
}
