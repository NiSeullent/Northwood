//! Request redirects and refusal controls through smoltcp connections.
use super::*;
struct Peer {
    lan: Lan,
    replies: Vec<Vec<u8>>,
    requests: Vec<Vec<u8>>,
    request: Vec<u8>,
    sent: usize,
    responding: bool,
}
impl Peer {
    fn new(clock: TestClock, replies: Vec<Vec<u8>>) -> Self {
        Self {
            lan: Lan::new(0, clock),
            replies,
            requests: Vec::new(),
            request: Vec::new(),
            sent: 0,
            responding: false,
        }
    }
    fn step(&mut self) {
        let socket = self.lan.sockets.get_mut::<tcp::Socket>(self.lan.listener);
        if socket.state() == tcp::State::Closed {
            socket.listen(80).unwrap();
            self.request.clear();
            self.sent = 0;
            self.responding = false;
        }
        let now = Instant::from_micros(self.lan.clock.0.get() as i64);
        self.lan
            .iface
            .poll(now, &mut self.lan.server, &mut self.lan.sockets);
        let socket = self.lan.sockets.get_mut::<tcp::Socket>(self.lan.listener);
        while socket.can_recv() {
            socket
                .recv(|bytes| {
                    self.request.extend_from_slice(bytes);
                    (bytes.len(), ())
                })
                .unwrap();
        }
        if !self.responding && self.request.windows(4).any(|w| w == b"\r\n\r\n") {
            self.responding = true;
            self.requests.push(self.request.clone());
        }
        if self.responding {
            let reply = &self.replies[(self.requests.len() - 1).min(self.replies.len() - 1)];
            if socket.can_send() && self.sent < reply.len() {
                self.sent += socket.send_slice(&reply[self.sent..]).unwrap();
            }
            if self.sent == reply.len() {
                socket.close();
            }
        }
        self.lan
            .iface
            .poll(now, &mut self.lan.server, &mut self.lan.sockets);
    }
}
impl Link for Peer {
    fn mac(&self) -> Mac {
        CLIENT
    }
    fn transmit(&mut self, frame: &[u8]) -> Result<(), LinkError> {
        self.lan.client.transmit(frame)
    }
    fn receive(&mut self) -> Result<Option<Vec<u8>>, LinkError> {
        if self.lan.client.into.borrow().0.is_empty() {
            self.step();
        }
        self.lan.client.receive()
    }
}
fn stack(replies: Vec<Vec<u8>>) -> Stack<Peer, TestClock> {
    let clock = TestClock::default();
    let mut stack = Stack::new(Peer::new(clock.clone(), replies), clock, 7).unwrap();
    stack
        .set_config(Config {
            address: Ipv4Cidr::new(CLIENT_IP, 24),
            router: None,
            dns: Vec::new(),
        })
        .unwrap();
    stack
}
fn sockets(stack: &Stack<Peer, TestClock>) -> usize {
    stack
        .sockets
        .iter()
        .filter(|(_, s)| matches!(s, smoltcp::socket::Socket::Tcp(_)))
        .count()
}
fn failure(stack: &mut Stack<Peer, TestClock>, transfer: &mut HttpTransfer) -> NetError {
    for _ in 0..100_000 {
        if let Err(error) =
            stack.read_http(transfer, |_, _| panic!("refused response cannot publish"))
        {
            return error;
        }
    }
    panic!("expected bounded refusal")
}
#[test]
fn relative_redirect_rebuilds_host_path_and_pins_the_final_range() {
    let mut stack=stack(vec![b"HTTP/1.1 302 Found\r\nLocation: next?sig=%2b%3D\r\nContent-Length: 0\r\n\r\n".to_vec(),b"HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 0-1/10\r\nContent-Length: 2\r\nETag: \"v1\"\r\n\r\nab".to_vec()]);
    let mut transfer = stack
        .open_http(
            "http://10.0.2.2/start",
            HttpRequest::range(0, 1, Some("\"v1\"")),
            1024,
            Err(TlsUnavailable::CorrelatedClock),
        )
        .unwrap();
    let mut ranged = http::Ranged::with_identity(0, 1, "\"v1\"", 10).unwrap();
    let mut body = Vec::new();
    let mut done = false;
    for _ in 0..100_000 {
        done = stack
            .read_http(&mut transfer, |bytes, eof| {
                ranged.feed(bytes, eof, |b| body.extend_from_slice(b))
            })
            .unwrap();
        if done {
            break;
        }
    }
    assert!(done);
    assert_eq!(body, b"ab");
    assert_eq!(transfer.url(), "http://10.0.2.2/next?sig=%2b%3D");
    assert_eq!(stack.link_mut().requests.len(), 2);
    let final_request = core::str::from_utf8(&stack.link_mut().requests[1]).unwrap();
    assert!(final_request.starts_with("GET /next?sig=%2b%3D HTTP/1.1\r\nHost: 10.0.2.2\r\n"));
    assert!(final_request.contains("If-Range: \"v1\"\r\n"));
    assert_eq!(sockets(&stack), 0);
    stack.close_http(transfer);
    assert_eq!(sockets(&stack), 0);
}

#[test]
fn an_aborted_full_receive_queue_releases_the_peer_for_the_next_range() {
    let mut large = b"HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 0-8191/8192\r\nContent-Length: 8192\r\nETag: \"v1\"\r\n\r\n".to_vec();
    large.resize(large.len() + 8192, b'x');
    let small = b"HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 4096-4097/8192\r\nContent-Length: 2\r\nETag: \"v1\"\r\n\r\nab".to_vec();
    let mut stack = stack(vec![large, small]);
    let mut first = stack
        .open_http(
            "http://10.0.2.2/object",
            HttpRequest::range(0, 8191, Some("\"v1\"")),
            512,
            Err(TlsUnavailable::CorrelatedClock),
        )
        .unwrap();
    let mut range = http::Ranged::with_identity(0, 8191, "\"v1\"", 8192).unwrap();
    let mut consumed = 0;
    for _ in 0..1024 {
        assert!(
            !stack
                .read_http(&mut first, |bytes, closed| {
                    range.feed(bytes, closed, |body| consumed += body.len())
                })
                .unwrap()
        );
        if consumed != 0 {
            break;
        }
    }
    assert!(consumed > 0 && consumed < 8192);
    for _ in 0..1024 {
        stack.poll().unwrap();
        if stack.http_receive_queue(&first) == Some((512, 512)) {
            break;
        }
    }
    assert_eq!(stack.http_receive_queue(&first), Some((512, 512)));
    stack.close_http(first);
    assert_eq!(sockets(&stack), 0);

    let mut second = stack
        .open_http(
            "http://10.0.2.2/object",
            HttpRequest::range(4096, 4097, Some("\"v1\"")),
            512,
            Err(TlsUnavailable::CorrelatedClock),
        )
        .unwrap();
    let mut range = http::Ranged::with_identity(4096, 4097, "\"v1\"", 8192).unwrap();
    let mut body = Vec::new();
    for _ in 0..1024 {
        if stack
            .read_http(&mut second, |bytes, closed| {
                range.feed(bytes, closed, |bytes| body.extend_from_slice(bytes))
            })
            .unwrap()
        {
            break;
        }
    }
    assert_eq!(body, b"ab");
    assert_eq!(stack.link_mut().requests.len(), 2);
}
#[test]
fn authentication_failures_are_explicit() {
    for (status, expected) in [
        (407, NetError::ProxyAuthentication),
        (511, NetError::NetworkAuthentication),
    ] {
        let mut stack = stack(vec![
            alloc::format!("HTTP/1.1 {status} Authentication\r\nContent-Length: 0\r\n\r\n")
                .into_bytes(),
        ]);
        let mut transfer = stack
            .open_http(
                "http://10.0.2.2/p",
                HttpRequest::range(0, 1, None),
                1024,
                Err(TlsUnavailable::CorrelatedClock),
            )
            .unwrap();
        assert_eq!(failure(&mut stack, &mut transfer), expected);
        assert_eq!(sockets(&stack), 0);
        stack.close_http(transfer);
    }
}
#[test]
fn refused_redirects_send_nothing_further() {
    let redirect = |status: &str, location: &str| {
        alloc::format!("HTTP/1.1 {status}\r\nLocation: {location}\r\nContent-Length: 0\r\n\r\n")
            .into_bytes()
    };
    let tls = NetError::TlsUnavailable(TlsUnavailable::CorrelatedClock);
    for (reply, request, expected) in [
        // An HTTPS upgrade without trusted time never falls back to plain TCP.
        (
            redirect("302 Found", "https://10.0.2.2/next"),
            HttpRequest::range(0, 1, None),
            tls.clone(),
        ),
        // A POST upgraded to HTTPS still requires its real TLS capability.
        (
            redirect("307 Temporary Redirect", "https://10.0.2.2/next"),
            HttpRequest::post(&[], "text/plain", b"body"),
            tls,
        ),
        // A cross-origin POST redirect never replays its body or credentials.
        (
            redirect("307 Temporary Redirect", "http://10.0.2.3/next"),
            HttpRequest::post(
                &[("Authorization", "Bearer private")],
                "text/plain",
                b"private-body",
            ),
            NetError::Redirect(http::RedirectError::Origin),
        ),
    ] {
        let mut stack = stack(vec![reply]);
        let mut transfer = stack
            .open_http(
                "http://10.0.2.2/p",
                request,
                1024,
                Err(TlsUnavailable::CorrelatedClock),
            )
            .unwrap();
        assert_eq!(failure(&mut stack, &mut transfer), expected);
        assert_eq!(stack.link_mut().requests.len(), 1);
        assert_eq!(sockets(&stack), 0);
        stack.close_http(transfer);
    }
}

#[test]
fn truncated_pinned_range_resumes_only_its_missing_bytes() {
    let mut stack=stack(vec![b"HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 0-1/10\r\nContent-Length: 2\r\nETag: \"v1\"\r\n\r\na".to_vec(),b"HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 1-1/10\r\nContent-Length: 1\r\nETag: \"v1\"\r\n\r\nb".to_vec()]);
    let mut transfer = stack
        .open_http(
            "http://10.0.2.2/p",
            HttpRequest::range(0, 1, Some("\"v1\"")),
            1024,
            Err(TlsUnavailable::CorrelatedClock),
        )
        .unwrap();
    let mut body = Vec::new();
    let mut ranged = http::Ranged::with_identity(0, 1, "\"v1\"", 10).unwrap();
    let mut failure = None;
    for _ in 0..100_000 {
        if let Err(error) = stack.read_http(&mut transfer, |bytes, eof| {
            ranged.feed(bytes, eof, |b| body.extend_from_slice(b))
        }) {
            failure = Some(error);
            break;
        }
    }
    let failure = failure.unwrap();
    assert!(matches!(failure, NetError::Truncated("HTTP range body")));
    assert!(failure.retryable());
    assert_eq!(body, b"a");
    assert_eq!(sockets(&stack), 0);
    stack.close_http(transfer);
    let mut transfer = stack
        .open_http(
            "http://10.0.2.2/p",
            HttpRequest::range(1, 1, Some("\"v1\"")),
            1024,
            Err(TlsUnavailable::CorrelatedClock),
        )
        .unwrap();
    let mut ranged = http::Ranged::with_identity(1, 1, "\"v1\"", 10).unwrap();
    let mut done = false;
    for _ in 0..100_000 {
        done = stack
            .read_http(&mut transfer, |bytes, eof| {
                ranged.feed(bytes, eof, |b| body.extend_from_slice(b))
            })
            .unwrap();
        if done {
            break;
        }
    }
    assert!(done);
    assert_eq!(body, b"ab");
    assert!(
        core::str::from_utf8(&stack.link_mut().requests[1])
            .unwrap()
            .contains("Range: bytes=1-1\r\nIf-Range: \"v1\"")
    );
    stack.close_http(transfer);
}
