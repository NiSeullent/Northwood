//! Restored uses big-endian u32 length prefixes; reverse proxy uses little
//! endian. Each poll attempts one bounded read/write and keeps partial I/O.
use super::{
    Error, Result,
    plist::{self, Limits, Value},
};
use alloc::vec::Vec;
use core::task::Poll;
use stela_contracts::refuse;

/// Integration boundary for the native mux API. Pending means no progress;
/// Ready(Ok(0)) means EOF on read, an error on a nonempty write.
/// Adapters must never report more bytes than the supplied slice.
pub trait ByteStream {
    fn read(&mut self, bytes: &mut [u8]) -> Poll<Result<usize>>;
    fn write(&mut self, bytes: &[u8]) -> Poll<Result<usize>>;
}
/// Opening each DataPort produces an independent stream. Reconnect for proxy
/// fallback: protocol 1 must not reuse a half-negotiated protocol-2 stream.
pub trait Mux {
    type Stream: ByteStream;
    fn connect(&mut self, port: u16) -> Poll<Result<Self::Stream>>;
}
#[derive(Debug, Clone, Copy)]
pub enum Endian {
    Big,
    Little,
}
impl Endian {
    fn encode(self, n: u32) -> [u8; 4] {
        match self {
            Self::Big => n.to_be_bytes(),
            Self::Little => n.to_le_bytes(),
        }
    }
    fn decode(self, n: [u8; 4]) -> u32 {
        match self {
            Self::Big => u32::from_be_bytes(n),
            Self::Little => u32::from_le_bytes(n),
        }
    }
}
#[derive(Default)]
pub struct Output {
    bytes: Vec<u8>,
    at: usize,
    /// Every byte the stream has taken.
    written: u64,
}
impl Output {
    pub fn is_pending(&self) -> bool {
        self.at < self.bytes.len()
    }
    pub fn written(&self) -> u64 {
        self.written
    }
    /// Bytes in the current buffer the stream has yet to accept.
    pub(super) fn remaining(&self) -> usize {
        self.bytes.len() - self.at
    }
    /// Take the emptied buffer after a successful send.
    pub(super) fn take_completed_buffer(&mut self) -> Vec<u8> {
        debug_assert!(!self.is_pending());
        core::mem::take(&mut self.bytes)
    }
    pub fn queue(&mut self, bytes: Vec<u8>, limit: usize) -> Result<()> {
        if self.is_pending() {
            return Err(Error::Invalid("output already pending"));
        }
        if bytes.len() > limit {
            return Err(Error::Limit("output queue"));
        }
        self.bytes = bytes;
        self.at = 0;
        Ok(())
    }
    pub fn poll(&mut self, stream: &mut impl ByteStream) -> Poll<Result<()>> {
        if !self.is_pending() {
            return Poll::Ready(Ok(()));
        }
        match stream.write(&self.bytes[self.at..]) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Err(e)) => Poll::Ready(Err(e)),
            Poll::Ready(Ok(n)) => {
                if n == 0 || n > self.bytes.len() - self.at {
                    return Poll::Ready(Err(Error::Transport(refuse!("invalid/zero write"))));
                }
                self.at += n;
                self.written += n as u64;
                if self.is_pending() {
                    Poll::Pending
                } else {
                    self.bytes.clear();
                    self.at = 0;
                    Poll::Ready(Ok(()))
                }
            }
        }
    }
}
pub struct Framed {
    endian: Endian,
    limits: Limits,
    header: [u8; 4],
    header_at: usize,
    body: Vec<u8>,
    body_at: usize,
    output: Output,
    failed: bool,
}
impl Framed {
    pub fn new(endian: Endian, limits: Limits) -> Self {
        Self {
            endian,
            limits,
            header: [0; 4],
            header_at: 0,
            body: Vec::new(),
            body_at: 0,
            output: Output::default(),
            failed: false,
        }
    }
    pub fn queue(&mut self, v: &Value, binary: bool) -> Result<()> {
        if self.failed {
            return Err(Error::Invalid("framing stream failed"));
        }
        let body = if binary {
            plist::encode_binary(v, self.limits)?
        } else {
            plist::encode_xml(v, self.limits)?
        };
        let length =
            u32::try_from(body.len()).map_err(|_| Error::Limit("plist frame u32 length"))?;
        let mut bytes = self.endian.encode(length).to_vec();
        bytes.extend_from_slice(&body);
        self.output
            .queue(bytes, self.limits.bytes.saturating_add(4))
    }
    /// Queue consecutive binary FileData frames and return the final frame's
    /// byte offset in the send buffer.
    pub(super) fn queue_file_data<'a>(
        &mut self,
        chunks: impl IntoIterator<Item = &'a [u8]>,
    ) -> Result<usize> {
        if self.failed {
            return Err(Error::Invalid("framing stream failed"));
        }
        if self.output.is_pending() {
            return Err(Error::Invalid("output already pending"));
        }
        let out = &mut self.output.bytes;
        let mut last = 0;
        for data in chunks {
            let at = out.len();
            last = at;
            out.resize(at + 4, 0);
            match plist::append_file_data(out, data, self.limits) {
                Ok(length) => out[at..at + 4].copy_from_slice(&self.endian.encode(length)),
                Err(error) => {
                    out.clear();
                    return Err(error);
                }
            }
        }
        Ok(last)
    }
    pub fn poll_send(&mut self, s: &mut impl ByteStream) -> Poll<Result<()>> {
        if self.failed {
            return Poll::Ready(Err(Error::Invalid("framing stream failed")));
        }
        let r = self.output.poll(s);
        if matches!(r, Poll::Ready(Err(_))) {
            self.failed = true;
        }
        r
    }
    pub fn sending(&self) -> bool {
        self.output.is_pending()
    }
    pub fn written(&self) -> u64 {
        self.output.written()
    }
    pub fn poll_receive(&mut self, s: &mut impl ByteStream) -> Poll<Result<Value>> {
        if self.failed {
            return Poll::Ready(Err(Error::Invalid("framing stream failed")));
        }
        let r = self.receive(s);
        if matches!(r, Poll::Ready(Err(_))) {
            self.failed = true;
        }
        r
    }
    fn receive(&mut self, s: &mut impl ByteStream) -> Poll<Result<Value>> {
        let header = self.header_at < 4;
        let target = if header {
            &mut self.header[self.header_at..]
        } else {
            &mut self.body[self.body_at..]
        };
        let n = match s.read(target) {
            Poll::Pending => return Poll::Pending,
            Poll::Ready(Err(e)) => return Poll::Ready(Err(e)),
            Poll::Ready(Ok(n)) => n,
        };
        if n == 0 || n > target.len() {
            return Poll::Ready(Err(Error::Transport("truncated/invalid plist read".into())));
        }
        if header {
            self.header_at += n;
            if self.header_at == 4 {
                let size = self.endian.decode(self.header) as usize;
                if size == 0 || size > self.limits.bytes {
                    return Poll::Ready(Err(Error::Limit("plist frame length")));
                }
                self.body.resize(size, 0);
            }
            return Poll::Pending;
        }
        self.body_at += n;
        if self.body_at < self.body.len() {
            return Poll::Pending;
        }
        let result = plist::decode(&self.body, self.limits);
        self.header_at = 0;
        self.body_at = 0;
        self.body.clear();
        Poll::Ready(result)
    }
}
/// ASR plists are unframed XML documents. Read a byte at a time so a coalesced
/// next document is left on the stream, without guessing USB packet boundaries.
/// Scheduling can use a buffered ByteStream adapter to avoid per-byte mux work.
pub struct XmlStream {
    bytes: Vec<u8>,
    limits: Limits,
}
impl XmlStream {
    pub fn new(limits: Limits) -> Self {
        Self {
            bytes: Vec::new(),
            limits,
        }
    }
    pub fn poll(&mut self, s: &mut impl ByteStream) -> Poll<Result<Value>> {
        let mut byte = [0];
        match s.read(&mut byte) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Err(e)) => Poll::Ready(Err(e)),
            Poll::Ready(Ok(1)) => {
                if self.bytes.len() == self.limits.bytes {
                    return Poll::Ready(Err(Error::Limit("ASR XML bytes")));
                }
                self.bytes.push(byte[0]);
                if self.bytes.ends_with(b"</plist>") {
                    let b = core::mem::take(&mut self.bytes);
                    Poll::Ready(plist::decode(&b, self.limits))
                } else {
                    Poll::Pending
                }
            }
            _ => Poll::Ready(Err(Error::Transport(
                "truncated/invalid ASR XML read".into(),
            ))),
        }
    }
}
/// Fixed-size incremental reads for raw protocol fields, never read-ahead.
#[derive(Default)]
pub(crate) struct Input {
    bytes: Vec<u8>,
    at: usize,
}
impl Input {
    pub(crate) fn poll(&mut self, s: &mut impl ByteStream, n: usize) -> Poll<Result<Vec<u8>>> {
        if n == 0 || n > 1024 {
            return Poll::Ready(Err(Error::Limit("raw protocol field")));
        }
        if self.bytes.is_empty() {
            self.bytes.resize(n, 0);
        }
        if self.bytes.len() != n {
            return Poll::Ready(Err(Error::Invalid("raw read length changed")));
        }
        let read = match s.read(&mut self.bytes[self.at..]) {
            Poll::Pending => return Poll::Pending,
            Poll::Ready(v) => v?,
        };
        if read == 0 || read > n - self.at {
            return Poll::Ready(Err(Error::Transport("truncated raw field".into())));
        }
        self.at += read;
        if self.at < n {
            Poll::Pending
        } else {
            self.at = 0;
            Poll::Ready(Ok(core::mem::take(&mut self.bytes)))
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    pub struct Memory {
        pub input: Vec<u8>,
        pub at: usize,
        pub output: Vec<u8>,
        pub chunk: usize,
        pub pending: bool,
    }
    impl Memory {
        pub fn new(input: Vec<u8>, chunk: usize) -> Self {
            Self {
                input,
                at: 0,
                output: Vec::new(),
                chunk,
                pending: false,
            }
        }
    }
    impl ByteStream for Memory {
        fn read(&mut self, b: &mut [u8]) -> Poll<Result<usize>> {
            self.pending = !self.pending;
            if self.pending {
                return Poll::Pending;
            }
            let n = b.len().min(self.chunk).min(self.input.len() - self.at);
            b[..n].copy_from_slice(&self.input[self.at..self.at + n]);
            self.at += n;
            Poll::Ready(Ok(n))
        }
        fn write(&mut self, b: &[u8]) -> Poll<Result<usize>> {
            self.pending = !self.pending;
            if self.pending {
                return Poll::Pending;
            }
            let n = b.len().min(self.chunk);
            self.output.extend_from_slice(&b[..n]);
            Poll::Ready(Ok(n))
        }
    }
    pub fn receive(f: &mut Framed, s: &mut Memory) -> Result<Value> {
        for _ in 0..1_000_000 {
            if let Poll::Ready(v) = f.poll_receive(s) {
                return v;
            }
        }
        panic!("receive did not terminate")
    }
    #[test]
    fn stream_framing_survives_fragmentation_and_short_writes() {
        for endian in [Endian::Big, Endian::Little] {
            for binary in [false, true] {
                let value = Value::dict([("Request", Value::from("QueryType"))]);
                let mut f = Framed::new(endian, Limits::default());
                let mut s = Memory::new(Vec::new(), 1);
                for _ in 0..2 {
                    f.queue(&value, binary).unwrap();
                    while f.sending() {
                        if let Poll::Ready(r) = f.poll_send(&mut s) {
                            r.unwrap();
                        }
                    }
                }
                s.input = core::mem::take(&mut s.output);
                assert_eq!(receive(&mut f, &mut s).unwrap(), value);
                assert_eq!(receive(&mut f, &mut s).unwrap(), value);
                assert!(receive(&mut f, &mut s).is_err());
            }
        }
    }
    #[test]
    fn file_data_reuses_the_frame_buffer_and_preserves_the_wire_bytes() {
        let limits = Limits::default();
        for endian in [Endian::Big, Endian::Little] {
            let mut frame = Framed::new(endian, limits);
            let mut stream = Memory::new(Vec::new(), 31);
            for len in [8192, 0, 1, 14, 15, 8192] {
                let data = vec![0x5a; len];
                let expected = plist::encode_binary(
                    &Value::dict([("FileData", Value::Data(data.clone()))]),
                    limits,
                )
                .unwrap();
                let buffer = frame.output.bytes.as_ptr();
                let capacity = frame.output.bytes.capacity();
                frame.queue_file_data([data.as_slice()]).unwrap();
                if capacity >= frame.output.bytes.len() {
                    assert_eq!(frame.output.bytes.as_ptr(), buffer);
                }
                assert!(frame.queue_file_data([b"pending".as_slice()]).is_err());
                while frame.sending() {
                    if let Poll::Ready(result) = frame.poll_send(&mut stream) {
                        result.unwrap();
                    }
                }
                assert_eq!(&stream.output[..4], &endian.encode(expected.len() as u32));
                assert_eq!(&stream.output[4..], expected);
                stream.output.clear();
            }
        }
        let mut limited = Framed::new(
            Endian::Big,
            Limits {
                bytes: 100,
                ..limits
            },
        );
        assert!(
            limited
                .queue_file_data([b"ok".as_slice(), &[0; 100]])
                .is_err()
        );
        assert!(limited.queue_file_data([[0; 40].as_slice()]).is_err());
        assert!(!limited.sending());
    }
    #[test]
    fn untrusted_lengths_fail_before_body_allocation() {
        let mut f = Framed::new(
            Endian::Big,
            Limits {
                bytes: 32,
                ..Limits::default()
            },
        );
        let mut s = Memory::new(u32::MAX.to_be_bytes().to_vec(), 4);
        assert!(receive(&mut f, &mut s).is_err());
        assert!(f.body.is_empty());
        assert!(f.queue(&Value::Boolean(true), false).is_err());
    }
}
