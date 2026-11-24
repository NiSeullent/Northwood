// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (C) 2015 Intel Corporation
//! btbcm USB reset/identification, HCD replay and finalize over real HCI events.
use super::{
    AppleReport, Attempt, Error, Hcd, Identity, LocalVersion, OwnedReport, Result, UsbIdentity,
};
use crate::{
    firmware::{ErrorKind as ResourceKind, Provider, Requirement},
    usb::bluetooth::{
        FirmwareReceipt, Hci, Io, Packet, State,
        command::{self, Response},
    },
};
use alloc::{boxed::Box, vec::Vec};

const RESET: u16 = 0x0c03;
const READ_NAME: u16 = 0x0c14;
const READ_VERSION: u16 = 0x1001;
const READ_ADDRESS: u16 = 0x1009;
const SET_ADDRESS: u16 = 0xfc01;
const DOWNLOAD_MINIDRIVER: u16 = 0xfc2e;
const READ_USB_PRODUCT: u16 = 0xfc5a;
const READ_FEATURES: u16 = 0xfc6e;
const READ_VERBOSE: u16 = 0xfc79;
const COMMAND_US: u64 = 5_000_000;
const ATTEMPT_US: u64 = 120_000_000;
const MAX_ASYNC_PACKETS: usize = 64;

#[expect(
    clippy::result_large_err,
    reason = "The shared error preserves exact source and cleanup causes without allocating on propagation."
)]
pub(super) fn run<B: Io>(
    hci: &mut Hci<B>,
    provider: &dyn Provider,
    address: Option<[u8; 6]>,
) -> Result<OwnedReport> {
    if !super::is_candidate(hci.binding.device()) {
        return Err(Error::Unsupported("USB PatchRAM identity/interface"));
    }
    initialize(hci, |engine| engine.run(provider, address))
}

#[expect(
    clippy::result_large_err,
    reason = "The shared error preserves exact source and cleanup causes without allocating on propagation."
)]
pub(in crate::usb::bluetooth) fn run_apple<B: Io>(hci: &mut Hci<B>) -> Result<AppleReport> {
    initialize(hci, |engine| engine.apple())
}

#[expect(
    clippy::result_large_err,
    reason = "The shared error preserves exact source and cleanup causes without allocating on propagation."
)]
fn initialize<B: Io, R>(
    hci: &mut Hci<B>,
    setup: impl FnOnce(&mut Engine<'_, B>) -> Result<R>,
) -> Result<R> {
    hci.open()?;
    let start = hci.io.now_us();
    let mut engine = Engine {
        hci,
        start,
        credits: 1,
        attempt: Attempt::default(),
    };
    match setup(&mut engine) {
        Ok(report) => Ok(report),
        Err(cause) => {
            let cause = Error::Attempt {
                cause: Box::new(cause),
                observation: Box::new(engine.attempt),
            };
            if engine.hci.state() == State::Open
                && let Err(cleanup) = engine.hci.close()
            {
                return Err(Error::Cleanup {
                    cause: Box::new(cause),
                    cleanup,
                });
            }
            Err(cause)
        }
    }
}
struct Engine<'a, B: Io> {
    hci: &'a mut Hci<B>,
    start: u64,
    credits: u8,
    attempt: Attempt,
}
impl<B: Io> Engine<'_, B> {
    #[expect(
        clippy::result_large_err,
        reason = "The shared error preserves exact source and cleanup causes without allocating on propagation."
    )]
    fn within_attempt(&self) -> Result<()> {
        if self.hci.io.now_us().wrapping_sub(self.start) >= ATTEMPT_US {
            Err(Error::Timeout("complete initialization attempt"))
        } else {
            Ok(())
        }
    }
    #[expect(
        clippy::result_large_err,
        reason = "The shared error preserves exact source and cleanup causes without allocating on propagation."
    )]
    fn asynchronous(&mut self, packet: Packet) -> Result<()> {
        if self.attempt.asynchronous.len() >= MAX_ASYNC_PACKETS {
            let (Packet::Event(bytes) | Packet::Acl(bytes) | Packet::Diagnostic(bytes)) = packet;
            return Err(Error::protocol(
                "bounded asynchronous initialization queue",
                &bytes,
            ));
        }
        self.attempt.asynchronous.push(packet);
        Ok(())
    }
    #[expect(
        clippy::result_large_err,
        reason = "The shared error preserves exact source and cleanup causes without allocating on propagation."
    )]
    fn command(&mut self, opcode: u16, parameters: &[u8]) -> Result<Vec<u8>> {
        self.within_attempt()?;
        let packet =
            command::packet(opcode, parameters).map_err(|why| Error::protocol(why, parameters))?;
        let start = self.hci.io.now_us();
        let mut sent = false;
        loop {
            self.within_attempt()?;
            if self.hci.io.now_us().wrapping_sub(start) >= COMMAND_US {
                return Err(Error::Timeout(if sent {
                    "genuine command completion"
                } else {
                    "genuine command credits"
                }));
            }
            if !sent && self.credits != 0 {
                self.hci.submit_command(&packet)?;
                self.credits -= 1;
                sent = true;
            }
            let mut completed = None;
            for received in self.hci.poll()? {
                let Packet::Event(event) = received else {
                    self.asynchronous(received)?;
                    continue;
                };
                match command::response(&event, opcode)
                    .map_err(|why| Error::protocol(why, &event))?
                {
                    Response::Credits(credits) => self.credits = credits,
                    Response::Complete { .. } | Response::Status { .. }
                        if !sent || completed.is_some() =>
                    {
                        return Err(Error::protocol(
                            "unsolicited or duplicate command response",
                            &event,
                        ));
                    }
                    Response::Complete { credits, returned } => {
                        self.credits = credits;
                        if returned[0] != 0 {
                            return Err(Error::Controller {
                                opcode,
                                status: returned[0],
                                parameters: returned.to_vec(),
                            });
                        }
                        completed = Some(returned.to_vec());
                    }
                    Response::Status { credits, status } => {
                        self.credits = credits;
                        if status != 0 {
                            return Err(Error::Controller {
                                opcode,
                                status,
                                parameters: event[2..].to_vec(),
                            });
                        }
                    }
                    Response::Other => self.asynchronous(Packet::Event(event))?,
                }
            }
            if let Some(returned) = completed {
                return Ok(returned);
            }
            crate::waiting();
            core::hint::spin_loop();
        }
    }
    #[expect(
        clippy::result_large_err,
        reason = "The shared error preserves exact source and cleanup causes without allocating on propagation."
    )]
    fn delay(&mut self, minimum: u64) -> Result<()> {
        let start = self.hci.io.now_us();
        while self.hci.io.now_us().wrapping_sub(start) < minimum {
            self.within_attempt()?;
            for packet in self.hci.poll()? {
                if let Packet::Event(ref event) = packet {
                    match command::response(event, 0).map_err(|why| Error::protocol(why, event))? {
                        Response::Credits(credits) => {
                            self.credits = credits;
                            continue;
                        }
                        Response::Other => {}
                        _ => {
                            return Err(Error::protocol(
                                "command completion while settling controller",
                                event,
                            ));
                        }
                    }
                }
                self.asynchronous(packet)?;
            }
            crate::waiting();
            core::hint::spin_loop();
        }
        self.within_attempt()
    }
    #[expect(
        clippy::result_large_err,
        reason = "The shared error preserves exact source and cleanup causes without allocating on propagation."
    )]
    fn exact<const N: usize>(&mut self, opcode: u16) -> Result<[u8; N]> {
        let p = self.command(opcode, &[])?;
        p.as_slice()
            .try_into()
            .map_err(|_| Error::protocol("vendor response length", &p))
    }
    #[expect(
        clippy::result_large_err,
        reason = "The shared error preserves exact source and cleanup causes without allocating on propagation."
    )]
    fn reset(&mut self) -> Result<()> {
        let p = self.command(RESET, &[])?;
        if p != [0] {
            return Err(Error::protocol("Reset response length", &p));
        }
        self.delay(100_000)
    }
    #[expect(
        clippy::result_large_err,
        reason = "The shared error preserves exact source and cleanup causes without allocating on propagation."
    )]
    fn name(&mut self) -> Result<Vec<u8>> {
        let p = self.command(READ_NAME, &[])?;
        if p.len() != 249 {
            return Err(Error::protocol("Read Local Name length", &p));
        }
        Ok(p)
    }
    #[expect(
        clippy::result_large_err,
        reason = "The shared error preserves exact source and cleanup causes without allocating on propagation."
    )]
    fn address(&mut self) -> Result<[u8; 6]> {
        let p = self.exact::<7>(READ_ADDRESS)?;
        Ok(p[1..].try_into().unwrap())
    }
    #[expect(
        clippy::result_large_err,
        reason = "The shared error preserves exact source and cleanup causes without allocating on propagation."
    )]
    fn apple(&mut self) -> Result<AppleReport> {
        self.reset()?;
        // Linux btbcm_setup_apple uses these reads for diagnostics. A real
        // controller rejection is optional, but transport/protocol faults
        // still take the normal retirement path.
        self.attempt.verbose = optional(self.exact(READ_VERBOSE))?;
        self.attempt.product = optional(self.exact(READ_USB_PRODUCT))?;
        self.attempt.features = optional(self.exact(READ_FEATURES))?;
        self.attempt.local_name = optional(self.name())?;
        Ok(AppleReport {
            usb: UsbIdentity::read(self.hci.binding.device()),
            interface: self.hci.binding.interface(),
            verbose: self.attempt.verbose,
            product: self.attempt.product,
            features: self.attempt.features,
            local_name: self.attempt.local_name.take(),
            command_credits: self.credits,
            asynchronous: core::mem::take(&mut self.attempt.asynchronous),
            strict_duplicate_filter: true,
        })
    }
    #[expect(
        clippy::result_large_err,
        reason = "The shared error preserves exact source and cleanup causes without allocating on propagation."
    )]
    fn run(
        &mut self,
        provider: &dyn Provider,
        platform_address: Option<[u8; 6]>,
    ) -> Result<OwnedReport> {
        if let Some(address) = platform_address
            && invalid_address(address)
        {
            return Err(Error::InvalidAddress(address));
        }
        self.reset()?;
        let version = LocalVersion::parse(&self.command(READ_VERSION, &[])?)?;
        self.attempt.version = Some(version.clone());
        let verbose = self.exact(READ_VERBOSE)?;
        self.attempt.verbose = Some(verbose);
        let features = self.exact(READ_FEATURES)?;
        self.attempt.features = Some(features);
        let local_name = self.name()?;
        self.attempt.local_name = Some(local_name.clone());
        let product = self.exact(READ_USB_PRODUCT)?;
        self.attempt.product = Some(product);
        let before = Identity {
            usb: UsbIdentity::read(self.hci.binding.device()),
            version,
            verbose,
            features,
            local_name,
            product,
        };
        self.attempt.identity = Some(before.clone());
        let hardware = before.usb.hardware_label();
        let revision = before.revision_label();
        let target = before.target(&hardware, &revision);
        let mut firmware = None;
        let mut missing = None;
        for name in before.firmware_names() {
            match provider.load(Requirement {
                name: &name,
                target,
                minimum_bytes: 3,
            }) {
                Ok(image) => {
                    firmware = Some(image);
                    break;
                }
                Err(error) if error.kind == ResourceKind::Missing => missing = Some(error),
                Err(error) => return Err(Error::Resource(error)),
            }
        }
        let firmware = firmware.ok_or_else(|| Error::Resource(missing.unwrap()))?;
        self.attempt.firmware = Some(FirmwareReceipt::from(firmware));
        let patch = Hcd::parse(firmware.bytes())?;
        let p = self.command(DOWNLOAD_MINIDRIVER, &[])?;
        if p != [0] {
            return Err(Error::protocol("Download Minidriver response length", &p));
        }
        self.attempt.download_mode_entered = true;
        self.delay(50_000)?;
        for record in patch.records() {
            self.command(record.opcode, record.parameters)?;
            self.attempt.patch_commands += 1;
        }
        self.delay(250_000)?;
        self.reset()?;
        let after = LocalVersion::parse(&self.command(READ_VERSION, &[])?)?;
        self.attempt.after = Some(after.clone());
        let after_features = self.exact(READ_FEATURES)?;
        self.attempt.after_features = Some(after_features);
        let after_local_name = self.name()?;
        self.attempt.after_local_name = Some(after_local_name.clone());
        let mut address = self.address()?;
        self.attempt.address = Some(address);
        if invalid_address(address) {
            let supplied = platform_address.ok_or(Error::InvalidAddress(address))?;
            let p = self.command(SET_ADDRESS, &supplied)?;
            if p != [0] {
                return Err(Error::protocol("Set BD_ADDR response length", &p));
            }
            address = self.address()?;
            self.attempt.address = Some(address);
            if address != supplied {
                return Err(Error::protocol(
                    "controller did not retain supplied BD_ADDR",
                    &address,
                ));
            }
        }
        Ok(OwnedReport {
            before,
            after,
            address,
            firmware: Box::new(firmware.into()),
            patch_commands: self.attempt.patch_commands,
            command_credits: self.credits,
            asynchronous: core::mem::take(&mut self.attempt.asynchronous),
            strict_duplicate_filter: true,
            after_features,
            after_local_name,
        })
    }
}

#[expect(
    clippy::result_large_err,
    reason = "The shared error preserves exact source and cleanup causes without allocating on propagation."
)]
fn optional<T>(result: Result<T>) -> Result<Option<T>> {
    match result {
        Err(Error::Controller { .. }) => Ok(None),
        other => other.map(Some),
    }
}

// Exact little-endian defaults checked by btbcm_check_bdaddr, plus absent
// all-zero/all-ones public addresses. Never manufacture a replacement identity.
fn invalid_address(address: [u8; 6]) -> bool {
    (address == [0; 6] || address == [0xff; 6])
        || [
            [0x00, 0xa0, 0x02, 0x70, 0x20, 0x00],
            [0x00, 0x00, 0xa0, 0x02, 0x70, 0x20],
            [0x79, 0x56, 0x00, 0xa0, 0x76, 0x20],
            [0x00, 0x00, 0x00, 0xb3, 0x24, 0x43],
            [0x00, 0x00, 0x00, 0xb1, 0x30, 0x43],
            [0x00, 0x00, 0x00, 0xb0, 0x34, 0x43],
            [0xac, 0x1f, 0x00, 0xc5, 0x45, 0x43],
            [0xac, 0x1f, 0x12, 0xa0, 0x43, 0x43],
            [0xaa; 6],
            [0xac, 0x1f, 0x00, 0x1b, 0x34, 0x43],
        ]
        .contains(&address)
}
