// SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause
// Copyright (C) 2005-2014, 2018-2026 Intel Corporation
// Copyright (C) 2013-2015 Intel Mobile Communications GmbH
// Copyright (C) 2016-2017 Intel Deutschland GmbH
//! API 46 ALIVE and INIT_COMPLETE for the separate INIT and REGULAR images.
//! The wire layout and wait behavior follow `fw/api/alive.h`,
//! `fw/api/commands.h`, `fw/notif-wait.c` and `mvm/fw.c` at Linux
//! cc9de5d4b5ab6459f5774946aa3195ad51236fa8.

use super::{API, Firmware};
use crate::wifi::intel::receive::{self, ALIVE_COMMAND, ALIVE_GROUP, Lmac, Umac, alive_flags};
use stela_driver_base::{Clock, ClockError, Deadline};

const DEFAULT_GROUP: u8 = 1;
const INIT_COMPLETE_COMMAND: u8 = 4;
const V3_BYTES: usize = 68;
pub(super) const TIMEOUT_US: u64 = 2_000_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Image {
    Init,
    Regular,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Report {
    pub flags: u16,
    pub lmac: Lmac,
    pub umac: Umac,
}

impl Report {
    pub(super) fn scheduler_base(self) -> u32 {
        // scd_base_ptr in iwl_lmac_debug_addrs for the v3 ALIVE layout.
        self.lmac.debug_addresses[5]
    }
}

/// One packet from an RX buffer the transport has reclaimed for the CPU.
/// `generation` comes from the owning RX queue at publication, independent
/// of the firmware command sequence or reported subtype. `index` is the
/// used-ring position in that generation's one-shot queue.
pub(super) struct Completion<'a> {
    pub(super) generation: u64,
    pub(super) index: u16,
    pub(super) group: u8,
    pub(super) command: u8,
    pub(super) payload: &'a [u8],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Progress {
    Pending,
    Malformed { actual: usize },
    Alive { image: Image, report: Report },
    InitComplete,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    FirmwareApi(u32),
    Version(Option<u8>),
    Clock(ClockError),
    State,
    Generation { previous: u64, next: u64 },
    Notification { image: Image, error: receive::Error },
    Timeout(Image),
    InitCompleteTimeout,
}

#[derive(Clone, Copy, Debug)]
enum Notification {
    Alive(Image),
    InitComplete { first_index: u16 },
}

#[derive(Debug)]
enum State {
    Waiting {
        notification: Notification,
        deadline: Deadline,
    },
    InitAlive,
    InitComplete,
    RegularAlive,
    Failed,
}

/// One two-image boot attempt. The transport owns RX, device stop, and DMA
/// lifetime. A new attempt requires fresh transport generations.
#[derive(Debug)]
pub struct Sequence {
    clock: Clock,
    generation: u64,
    state: State,
}

pub(super) fn validate_firmware(firmware: &Firmware<'_>) -> Result<(), Error> {
    let api = firmware.api_version();
    if api != API {
        return Err(Error::FirmwareApi(api));
    }
    let version = firmware
        .command_version(ALIVE_GROUP, ALIVE_COMMAND)
        .map(|entry| entry.notification_version);
    if version != Some(3) {
        return Err(Error::Version(version));
    }
    Ok(())
}

impl Sequence {
    /// Start the INIT deadline after the firmware start is acknowledged.
    /// The transport must be ready to capture ALIVE before it starts INIT.
    pub(super) fn wait_init(
        firmware: &Firmware<'_>,
        clock: Clock,
        generation: u64,
    ) -> Result<Self, Error> {
        validate_firmware(firmware)?;
        Ok(Self {
            clock,
            generation,
            state: State::Waiting {
                notification: Notification::Alive(Image::Init),
                deadline: clock.try_deadline(TIMEOUT_US).map_err(Error::Clock)?,
            },
        })
    }

    /// Arm immediately before submitting the PHY configuration command.
    /// The transport samples the first uncompleted index of the one-shot
    /// RX ring at this boundary. Earlier completions cannot advance INIT.
    /// The calibration deadline starts when this wait is armed.
    pub(super) fn wait_init_complete(
        &mut self,
        generation: u64,
        first_index: u16,
    ) -> Result<(), Error> {
        if !matches!(self.state, State::InitAlive) {
            return Err(Error::State);
        }
        if !self.accepts_generation(generation) {
            return Err(Error::Generation {
                previous: self.generation,
                next: generation,
            });
        }
        self.state = State::Waiting {
            notification: Notification::InitComplete { first_index },
            deadline: self.clock.try_deadline(TIMEOUT_US).map_err(Error::Clock)?,
        };
        Ok(())
    }

    pub(super) fn accepts_generation(&self, generation: u64) -> bool {
        self.generation == generation
    }

    pub(super) fn check_regular_generation(&self, generation: u64) -> Result<(), Error> {
        if !matches!(self.state, State::InitComplete) {
            return Err(Error::State);
        }
        if generation <= self.generation {
            return Err(Error::Generation {
                previous: self.generation,
                next: generation,
            });
        }
        Ok(())
    }

    /// Call after INIT_COMPLETE, acknowledged device stop, and REGULAR
    /// start on a fresh RX queue. The queue must be ready to capture ALIVE
    /// before REGULAR starts.
    pub(super) fn wait_regular_after_init_stop(&mut self, generation: u64) -> Result<(), Error> {
        self.check_regular_generation(generation)?;
        self.state = State::Waiting {
            notification: Notification::Alive(Image::Regular),
            deadline: self.clock.try_deadline(TIMEOUT_US).map_err(Error::Clock)?,
        };
        self.generation = generation;
        Ok(())
    }

    /// Observe an optional completed RX packet or check for timeout alone.
    /// Wrong-generation packets and unrelated commands leave the wait armed.
    /// A matching ALIVE packet with a malformed length can be followed by a
    /// valid packet within the original deadline. Expiration is checked at
    /// poll time and retires the wait before considering the supplied packet.
    pub(super) fn poll(&mut self, completion: Option<Completion<'_>>) -> Result<Progress, Error> {
        let State::Waiting {
            notification,
            deadline,
        } = self.state
        else {
            return Err(Error::State);
        };
        if deadline.passed() {
            let error = match notification {
                Notification::Alive(image) => Error::Timeout(image),
                Notification::InitComplete { .. } => Error::InitCompleteTimeout,
            };
            self.state = State::Failed;
            return Err(error);
        }
        let Some(completion) = completion else {
            return Ok(Progress::Pending);
        };
        let command = match notification {
            Notification::Alive(_) => ALIVE_COMMAND,
            Notification::InitComplete { first_index } => {
                if completion.index < first_index {
                    return Ok(Progress::Pending);
                }
                INIT_COMPLETE_COMMAND
            }
        };
        if !self.accepts_generation(completion.generation)
            || completion.command != command
            || !matches!(completion.group, ALIVE_GROUP | DEFAULT_GROUP)
        {
            return Ok(Progress::Pending);
        }
        let Notification::Alive(image) = notification else {
            // Linux's INIT_COMPLETE callback checks the command ID without
            // interpreting a payload. The RX framing has already been checked.
            self.state = State::InitComplete;
            return Ok(Progress::InitComplete);
        };
        self.observe_alive(image, completion.payload)
    }

    fn observe_alive(&mut self, image: Image, payload: &[u8]) -> Result<Progress, Error> {
        if payload.len() != V3_BYTES {
            return Ok(Progress::Malformed {
                actual: payload.len(),
            });
        }
        let flags = match alive_flags(payload) {
            Ok(flags) => flags,
            Err(error) => {
                self.state = State::Failed;
                return Err(Error::Notification { image, error });
            }
        };
        let report = Report {
            flags,
            lmac: Lmac::decode(&payload[4..52]),
            umac: Umac::decode(&payload[52..68]),
        };
        self.state = match image {
            Image::Init => State::InitAlive,
            Image::Regular => State::RegularAlive,
        };
        Ok(Progress::Alive { image, report })
    }
}
