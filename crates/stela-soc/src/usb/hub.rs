//! A native USB 2.0 high-speed hub with a single transaction translator
//! (USB 2.0 chapter 11), as a host port's [`Function`] whose downstream
//! ports carry other functions.
//!
//! Evidence: USB 2.0 11.23 (hub descriptors and the device, qualifier and
//! configuration a high-speed hub reports, 11.23.1), 11.24.2 (hub class
//! requests), 11.24.2.6-7 (hub and port status and change bits), 11.5
//! (downstream port states) and 11.12.1 (the status change endpoint's
//! bitmap). On J473 macOS `AppleUSB20Hub` (KC `0x451599c`) matches
//! bDeviceClass 9, bDeviceSubClass 0.
//!
//! The hub reports individual port power switching, individual
//! over-current protection, software-controllable port indicators and a TT
//! think time of 8 full-speed bit times; its ports are removable. Port
//! reset completes at once: the port is Enabled with C_PORT_RESET set and
//! its function reset, as at the end of the 10 ms reset signaling
//! (11.5.1.5). Resume likewise completes at once with C_PORT_SUSPEND.
//!
//! The xHCI controller above routes a slot to a downstream function by its
//! route string, so the hub carries no split transactions itself: its TT
//! holds no buffered state, which makes CLEAR_TT_BUFFER and RESET_TT
//! complete with nothing to clear. STOP_TT (11.24.2.11) halts the TT until
//! RESET_TT, and while it is stopped the low- and full-speed functions it
//! serves take no transactions. GET_TT_STATE (11.24.2.6) returns, for a
//! stopped TT with TT_Flags 0, the hub's own state format: [`TT_STATE`].
//!
//! SET_FEATURE(PORT_TEST) (11.24.2.13, the test selectors of table 11-24)
//! puts a disabled, disconnected or suspended port in the Testing state
//! (11.5): PORT_TEST is set and the port forwards nothing, and only a reset
//! of the hub itself, or a configuration change, which powers its ports
//! off, ends it. SET_FEATURE(PORT_INDICATOR) takes the selectors of table
//! 11-25; a nonzero selector sets PORT_INDICATOR (software control), and
//! selector 0 or CLEAR_FEATURE(PORT_INDICATOR) returns the indicator to
//! automatic. The indicator is a light with no other effect.
//!
//! Refused (STALL), as request errors: SET_HUB_DEPTH, which USB 3.2
//! 10.16.2.9 defines for Enhanced SuperSpeed hubs only (a USB 2.0 hub
//! has no request 12, and `AppleUSB20Hub` never sends it), and setting
//! the hub's or a port's change features, which 11.24.2.12-13 make
//! optional.
use super::{
    BusPeer, Function, Reply, Speed, Token,
    control::{
        self, CLASS, CLEAR_FEATURE, Control, DEVICE, DEVICE_QUALIFIER, Descriptors, ENDPOINT,
        ENDPOINT_HALT, GET_DESCRIPTOR, GET_INTERFACE, GET_STATUS, INTERFACE, Outcome,
        RECIPIENT_DEVICE, RECIPIENT_ENDPOINT, RECIPIENT_INTERFACE, RECIPIENT_OTHER, Requests,
        SET_FEATURE, SET_INTERFACE, STANDARD, Setup, Stall,
    },
};
use crate::Result;
use alloc::{format, string::String, vec, vec::Vec};
use core::fmt::Write;
use stela_contracts::refuse;

/// The most ports one status-change byte reports.
pub const MAX_PORTS: usize = 7;
const STATUS_ENDPOINT: u8 = 0x81;
const HUB_DESCRIPTOR: u8 = 0x29;
/// Individual port power switching, individual over-current protection,
/// port indicators (11.23.2.1: wHubCharacteristics D1..0, D4..3, D7).
const CHARACTERISTICS: u16 = 0x0089;
/// Power-on to power-good, in 2 ms units.
const POWER_ON_TO_GOOD: u8 = 5;
const CONTROLLER_CURRENT_MA: u8 = 50;
/// High speed: 2^(12-1) microframes; full speed: 255 frames.
const HIGH_SPEED_INTERVAL: u8 = 12;
const FULL_SPEED_INTERVAL: u8 = 255;

const CLEAR_TT_BUFFER: u8 = 8;
const RESET_TT: u8 = 9;
const GET_TT_STATE: u8 = 10;
const STOP_TT: u8 = 11;
/// The hub's GET_TT_STATE reply (a vendor-specific format, 11.24.2.6):
/// byte 0 is 1 while the TT is stopped, byte 1 the transactions it holds
/// buffered (always 0: the controller above carries the splits), bytes
/// 2..3 reserved.
const TT_STATE: [u8; 4] = [1, 0, 0, 0];
/// Table 11-24: Test_J, Test_K, Test_SE0_NAK, Test_Packet,
/// Test_Force_Enable.
const TEST_SELECTORS: core::ops::RangeInclusive<u8> = 1..=5;
/// Table 11-25: automatic, amber, green, off.
const INDICATOR_SELECTORS: core::ops::RangeInclusive<u8> = 0..=3;

// Port features (11.24.2, table 11-17) and status bits (11.24.2.7.1).
const PORT_CONNECTION: u16 = 0;
const PORT_ENABLE: u16 = 1;
const PORT_SUSPEND: u16 = 2;
const PORT_RESET: u16 = 4;
const PORT_POWER: u16 = 8;
const PORT_LOW_SPEED: u16 = 9;
const PORT_HIGH_SPEED: u16 = 10;
const PORT_TEST: u16 = 11;
const PORT_INDICATOR: u16 = 12;
/// The feature selectors of PORT_TEST and PORT_INDICATOR (table 11-17).
const TEST_FEATURE: u16 = 21;
const INDICATOR_FEATURE: u16 = 22;
const C_PORT_CONNECTION: u16 = 16;
const C_PORT_ENABLE: u16 = 17;
const C_PORT_SUSPEND: u16 = 18;
const C_PORT_OVER_CURRENT: u16 = 19;
const C_PORT_RESET: u16 = 20;
const C_HUB_LOCAL_POWER: u16 = 0;
const C_HUB_OVER_CURRENT: u16 = 1;

const fn bit(feature: u16) -> u16 {
    1 << feature
}

/// The functions on a hub's downstream ports, by zero-based port index.
pub trait Ports {
    fn count(&self) -> usize;
    fn speed(&self, port: usize) -> Speed;
    /// Whether a function is cabled to the port now.
    fn attached(&self, _port: usize) -> bool {
        true
    }
    fn function(&mut self, port: usize) -> &mut dyn Function;
    fn function_ref(&self, port: usize) -> &dyn Function;
}

#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
struct Port {
    status: u16,
    change: u16,
}

impl Port {
    fn testing(&self) -> bool {
        self.status & bit(PORT_TEST) != 0
    }

    fn disconnect(&mut self) {
        if self.status & bit(PORT_CONNECTION) != 0 {
            self.status &= bit(PORT_POWER) | bit(PORT_TEST);
            self.change |= bit(PORT_CONNECTION);
        }
    }
}

struct State {
    ports: Vec<Port>,
    halted: bool,
    /// STOP_TT halted the TT; RESET_TT restarts it.
    tt_stopped: bool,
}

/// The hub's class and interface requests over its ports.
struct Context<'a, P: Ports> {
    state: &'a mut State,
    functions: &'a mut P,
}

pub struct Hub<P: Ports> {
    control: Control,
    state: State,
    /// The downstream functions.
    pub ports: P,
}

fn descriptors(serial: &str) -> Descriptors {
    let device = [
        18,
        DEVICE,
        0x00,
        0x02, // USB 2.0
        9,
        0,
        1, // hub, single TT
        control::EP0_PACKET as u8,
        0x09,
        0x12, // vendor 0x1209 (pid.codes)
        0x01,
        0x00, // its test product ID
        0x00,
        0x01,
        1,
        2,
        3,
        1,
    ];
    let configuration = |interval: u8| {
        control::configuration(
            1,
            0,
            &[
                &[9, INTERFACE, 0, 0, 1, 9, 0, 0, 0],
                &[7, ENDPOINT, STATUS_ENDPOINT, 3, 1, 0, interval],
            ],
        )
    };
    // At full speed a hub has no TT: protocol 0 (11.23.1).
    let qualifier = [
        10,
        DEVICE_QUALIFIER,
        0x00,
        0x02,
        9,
        0,
        0,
        control::EP0_PACKET as u8,
        1,
        0,
    ];
    Descriptors {
        device,
        configuration: configuration(HIGH_SPEED_INTERVAL),
        other_speed: Some((qualifier, configuration(FULL_SPEED_INTERVAL))),
        strings: vec![
            String::from("Stela"),
            String::from("Stela USB 2.0 Hub"),
            String::from(serial),
        ],
    }
}

impl<P: Ports> Hub<P> {
    pub fn new(ports: P, serial: &str) -> Result<Self> {
        if !(1..=MAX_PORTS).contains(&ports.count()) {
            return Err(refuse!("a hub has one to seven downstream ports"));
        }
        Ok(Self {
            control: Control::new(descriptors(serial)),
            state: State {
                ports: vec![Port::default(); ports.count()],
                halted: false,
                tt_stopped: false,
            },
            ports,
        })
    }

    fn context(&mut self) -> (&mut Control, Context<'_, P>) {
        (
            &mut self.control,
            Context {
                state: &mut self.state,
                functions: &mut self.ports,
            },
        )
    }

    /// The configuration and each port's status and change bits, for the
    /// boot log's heartbeat.
    pub fn summary(&self) -> String {
        let mut line = format!("configuration={} ports=[", self.control.configuration());
        for port in &self.state.ports {
            let _ = write!(line, " {:#x}/{:#x}", port.status, port.change);
        }
        line.push_str(" ]");
        line
    }

    /// Brings each powered port's connection status in line with its cable
    /// (11.24.2.7.1.1): a change sets C_PORT_CONNECTION, and a
    /// disconnection disables the port.
    pub(crate) fn detect(&mut self) {
        for (n, port) in self.state.ports.iter_mut().enumerate() {
            if port.status & bit(PORT_POWER) == 0 || port.testing() {
                continue;
            }
            let attached = self.ports.attached(n);
            let connected = port.status & bit(PORT_CONNECTION) != 0;
            if attached && !connected {
                port.status |= bit(PORT_CONNECTION);
                if self.ports.speed(n) == Speed::Low {
                    port.status |= bit(PORT_LOW_SPEED);
                }
                port.change |= bit(PORT_CONNECTION);
            } else if !attached {
                port.disconnect();
            }
        }
    }

    /// Publish removal before another function can take this slot.
    pub(crate) fn disconnect(&mut self, port: usize) {
        debug_assert!(!self.ports.attached(port));
        self.state.ports[port].disconnect();
    }

    /// A vacant port can accept a function once it reports no connection change.
    pub(crate) fn can_connect(&self, port: usize) -> bool {
        let state = &self.state.ports[port];
        !state.testing() && state.change & bit(PORT_CONNECTION) == 0
    }

    /// The status change bitmap (11.12.4): bit 0 the hub, bit n port n.
    fn changes(&self) -> u8 {
        self.state
            .ports
            .iter()
            .enumerate()
            .filter(|(_, port)| port.change != 0)
            .fold(0, |bitmap, (n, _)| bitmap | 1 << (n + 1))
    }

    /// A port whose function takes transactions: enabled, neither
    /// suspended nor testing, and for a low- or full-speed function a
    /// running TT.
    fn usable(&self, port: u8) -> Option<usize> {
        let n = usize::from(port).checked_sub(1)?;
        let status = self.state.ports.get(n)?.status;
        if !self.ports.attached(n) {
            return None;
        }
        let enabled = bit(PORT_CONNECTION) | bit(PORT_ENABLE) | bit(PORT_POWER);
        let split = self.ports.speed(n) != Speed::High;
        (status & enabled == enabled
            && status & (bit(PORT_SUSPEND) | bit(PORT_TEST)) == 0
            && !(split && self.state.tt_stopped))
            .then_some(n)
    }
}

/// The port wIndex names, from 1.
fn port(ports: &mut [Port], index: u16) -> Result<(usize, &mut Port), Stall> {
    let n = usize::from(index as u8).checked_sub(1).ok_or(Stall)?;
    if index >> 8 != 0 {
        return Err(Stall);
    }
    ports.get_mut(n).map(|p| (n, p)).ok_or(Stall)
}

impl<P: Ports> Context<'_, P> {
    fn port(&mut self, index: u16) -> Result<(usize, &mut Port), Stall> {
        port(&mut self.state.ports, index)
    }

    fn set_port_feature(&mut self, setup: Setup) -> Result<Outcome, Stall> {
        // PORT_TEST and PORT_INDICATOR carry a selector in wIndex 15:8.
        let selector = (setup.index >> 8) as u8;
        let index = match setup.value {
            TEST_FEATURE | INDICATOR_FEATURE => setup.index & 0xff,
            _ => setup.index,
        };
        let (n, port) = port(&mut self.state.ports, index)?;
        if port.testing() {
            // A port in test mode leaves it only through a hub reset.
            return Err(Stall);
        }
        let powered = port.status & bit(PORT_POWER) != 0;
        match setup.value {
            PORT_POWER => port.status |= bit(PORT_POWER),
            PORT_RESET
                if powered
                    && port.status & bit(PORT_CONNECTION) != 0
                    && self.functions.attached(n) =>
            {
                port.status &= !(bit(PORT_SUSPEND) | bit(PORT_HIGH_SPEED));
                port.status |= bit(PORT_ENABLE);
                if self.functions.speed(n) == Speed::High {
                    port.status |= bit(PORT_HIGH_SPEED);
                }
                port.change |= bit(PORT_RESET);
                self.functions.function(n).reset().map_err(|_| Stall)?;
            }
            // Reset of an unconnected or unpowered port has no effect.
            PORT_RESET => {}
            PORT_SUSPEND if port.status & bit(PORT_ENABLE) != 0 => port.status |= bit(PORT_SUSPEND),
            PORT_SUSPEND => {}
            // Only a Disabled, Disconnected or Suspended port enters test
            // mode (11.24.2.13).
            TEST_FEATURE if TEST_SELECTORS.contains(&selector) => {
                let enabled = port.status & bit(PORT_ENABLE) != 0;
                let suspended = port.status & bit(PORT_SUSPEND) != 0;
                if enabled && !suspended {
                    return Err(Stall);
                }
                port.status |= bit(PORT_TEST);
            }
            // The selected color is a light with no other effect.
            INDICATOR_FEATURE if INDICATOR_SELECTORS.contains(&selector) => {
                if selector == 0 {
                    port.status &= !bit(PORT_INDICATOR);
                } else {
                    port.status |= bit(PORT_INDICATOR);
                }
            }
            _ => return Err(Stall),
        }
        Ok(Outcome::Done)
    }

    fn clear_port_feature(&mut self, setup: Setup) -> Result<Outcome, Stall> {
        let (_, port) = self.port(setup.index)?;
        if port.testing() {
            return Err(Stall);
        }
        match setup.value {
            PORT_ENABLE => port.status &= !(bit(PORT_ENABLE) | bit(PORT_SUSPEND)),
            PORT_SUSPEND if port.status & bit(PORT_SUSPEND) != 0 => {
                port.status &= !bit(PORT_SUSPEND);
                port.change |= bit(PORT_SUSPEND);
            }
            PORT_SUSPEND => {}
            // Power off disconnects; a reconnection follows power on.
            PORT_POWER => port.status = 0,
            INDICATOR_FEATURE => port.status &= !bit(PORT_INDICATOR),
            C_PORT_CONNECTION | C_PORT_ENABLE | C_PORT_SUSPEND | C_PORT_OVER_CURRENT
            | C_PORT_RESET => port.change &= !bit(setup.value - C_PORT_CONNECTION),
            _ => return Err(Stall),
        }
        Ok(Outcome::Done)
    }

    /// The single TT's port argument is 1 (11.24.2.6, 11.24.2.11).
    fn single_tt(setup: Setup) -> Result<(), Stall> {
        (setup.index == 1).then_some(()).ok_or(Stall)
    }
}

impl<P: Ports> Requests for Context<'_, P> {
    fn configure(&mut self, value: u8) -> Result<(), Stall> {
        self.state.halted = false;
        if value == 0 {
            // Back to Address: ports lose power (11.11).
            self.state.ports.fill(Port::default());
            self.state.tt_stopped = false;
        }
        Ok(())
    }

    fn request(&mut self, setup: Setup, _data: &[u8]) -> Result<Outcome, Stall> {
        let read = setup.device_to_host();
        match (setup.kind(), setup.recipient(), setup.request, read) {
            (CLASS, RECIPIENT_DEVICE, GET_DESCRIPTOR, true)
                if setup.value >> 8 == u16::from(HUB_DESCRIPTOR) && setup.value & 0xff == 0 =>
            {
                let count = self.state.ports.len() as u8;
                let [low, high] = CHARACTERISTICS.to_le_bytes();
                Ok(Outcome::Data(vec![
                    9,
                    HUB_DESCRIPTOR,
                    count,
                    low,
                    high,
                    POWER_ON_TO_GOOD,
                    CONTROLLER_CURRENT_MA,
                    0,    // every port removable
                    0xff, // PortPwrCtrlMask, all ones (11.23.2.1)
                ]))
            }
            (CLASS, RECIPIENT_DEVICE, GET_STATUS, true) if setup.value == 0 => {
                // Local power good, no over-current, no changes.
                Ok(Outcome::Data(vec![0; 4]))
            }
            (CLASS, RECIPIENT_DEVICE, CLEAR_FEATURE, false)
                if matches!(setup.value, C_HUB_LOCAL_POWER | C_HUB_OVER_CURRENT) =>
            {
                Ok(Outcome::Done)
            }
            (CLASS, RECIPIENT_OTHER, GET_STATUS, true) if setup.value == 0 => {
                let (_, port) = self.port(setup.index)?;
                let mut bytes = port.status.to_le_bytes().to_vec();
                bytes.extend_from_slice(&port.change.to_le_bytes());
                Ok(Outcome::Data(bytes))
            }
            (CLASS, RECIPIENT_OTHER, SET_FEATURE, false) => self.set_port_feature(setup),
            (CLASS, RECIPIENT_OTHER, CLEAR_FEATURE, false) => self.clear_port_feature(setup),
            (CLASS, RECIPIENT_OTHER, CLEAR_TT_BUFFER | RESET_TT, false) => {
                // Single TT: wIndex is the port for RESET_TT, 1 for
                // CLEAR_TT_BUFFER (11.24.2.3, 11.24.2.9).
                match setup.request {
                    RESET_TT => {
                        self.port(setup.index)?;
                        self.state.tt_stopped = false;
                        Ok(Outcome::Done)
                    }
                    _ => Ok(Outcome::Done),
                }
            }
            (CLASS, RECIPIENT_OTHER, STOP_TT, false) if setup.value == 0 => {
                Self::single_tt(setup)?;
                self.state.tt_stopped = true;
                Ok(Outcome::Done)
            }
            // No TT_Flags are defined, and only a stopped TT reports its
            // state.
            (CLASS, RECIPIENT_OTHER, GET_TT_STATE, true)
                if setup.value == 0 && self.state.tt_stopped =>
            {
                Self::single_tt(setup)?;
                Ok(Outcome::Data(TT_STATE.to_vec()))
            }
            (STANDARD, RECIPIENT_INTERFACE, GET_STATUS, true) if setup.index == 0 => {
                Ok(Outcome::Data(vec![0, 0]))
            }
            (STANDARD, RECIPIENT_INTERFACE, GET_INTERFACE, true) if setup.index == 0 => {
                Ok(Outcome::Data(vec![0]))
            }
            (STANDARD, RECIPIENT_INTERFACE, SET_INTERFACE, false)
                if setup.index == 0 && setup.value == 0 =>
            {
                self.state.halted = false;
                Ok(Outcome::Done)
            }
            (STANDARD, RECIPIENT_ENDPOINT, GET_STATUS, true)
                if setup.index == u16::from(STATUS_ENDPOINT) =>
            {
                Ok(Outcome::Data(vec![u8::from(self.state.halted), 0]))
            }
            (STANDARD, RECIPIENT_ENDPOINT, CLEAR_FEATURE | SET_FEATURE, false)
                if setup.index == u16::from(STATUS_ENDPOINT) && setup.value == ENDPOINT_HALT =>
            {
                self.state.halted = setup.request == SET_FEATURE;
                Ok(Outcome::Done)
            }
            _ => Err(Stall),
        }
    }
}

impl<P: Ports> BusPeer for Hub<P> {
    /// A reset from the hub's own upstream port: Default state, every
    /// downstream port unpowered (11.24.2.7.1).
    fn reset(&mut self) -> Result<()> {
        let (control, mut context) = self.context();
        control.reset(&mut context)?;
        self.state.ports.fill(Port::default());
        self.state.halted = false;
        self.state.tt_stopped = false;
        Ok(())
    }

    fn packet(&mut self, endpoint: u8, token: Token, data: &mut [u8]) -> Result<Reply> {
        match (endpoint, token) {
            (0, _) => {
                let (control, mut context) = self.context();
                let reply = control.packet(token, data, &mut context);
                self.detect();
                reply
            }
            (_, Token::Setup) => Err(refuse!("SETUP to a non-control endpoint")),
            (STATUS_ENDPOINT, Token::In) if self.control.configuration() != 0 => {
                if self.state.halted {
                    return Ok(Reply::Stall);
                }
                self.detect();
                match self.changes() {
                    0 => Ok(Reply::Nak),
                    bitmap if !data.is_empty() => {
                        data[0] = bitmap;
                        Ok(Reply::Ack(1))
                    }
                    _ => Err(refuse!("status change IN without a buffer")),
                }
            }
            _ => Ok(Reply::Stall),
        }
    }
}

impl<P: Ports> Function for Hub<P> {
    fn set_time(&mut self, now: u64) {
        for n in 0..self.ports.count() {
            if self.ports.attached(n) {
                self.ports.function(n).set_time(now);
            }
        }
        self.detect();
    }

    fn ready_at(&self, endpoint: u8) -> Option<u64> {
        let configured = self.control.configuration() != 0 && !self.state.halted;
        let attach = self.state.ports.iter().enumerate().any(|(n, port)| {
            port.status & bit(PORT_POWER) != 0
                && self.ports.attached(n) != (port.status & bit(PORT_CONNECTION) != 0)
        });
        (endpoint == STATUS_ENDPOINT && configured && (self.changes() != 0 || attach)).then_some(0)
    }

    fn downstream(&mut self, port: u8) -> Option<&mut dyn Function> {
        let n = self.usable(port)?;
        Some(self.ports.function(n))
    }

    fn downstream_ref(&self, port: u8) -> Option<&dyn Function> {
        let n = self.usable(port)?;
        Some(self.ports.function_ref(n))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::usb::control::{
        GET_CONFIGURATION, SET_ADDRESS, SET_CONFIGURATION,
        tests::{setup, transfer},
    };

    /// A function that counts its resets and answers one IN endpoint.
    #[derive(Default)]
    struct Probe {
        resets: usize,
        now: u64,
    }
    impl BusPeer for Probe {
        fn reset(&mut self) -> Result<()> {
            self.resets += 1;
            Ok(())
        }
        fn packet(&mut self, _: u8, _: Token, _: &mut [u8]) -> Result<Reply> {
            Ok(Reply::Ack(0))
        }
    }
    impl Function for Probe {
        fn set_time(&mut self, now: u64) {
            self.now = now;
        }
        fn ready_at(&self, _: u8) -> Option<u64> {
            None
        }
    }

    struct Two {
        functions: [Probe; 2],
        speeds: [Speed; 2],
        attached: [bool; 2],
    }
    impl Ports for Two {
        fn count(&self) -> usize {
            2
        }
        fn speed(&self, port: usize) -> Speed {
            self.speeds[port]
        }
        fn attached(&self, port: usize) -> bool {
            self.attached[port]
        }
        fn function(&mut self, port: usize) -> &mut dyn Function {
            &mut self.functions[port]
        }
        fn function_ref(&self, port: usize) -> &dyn Function {
            &self.functions[port]
        }
    }

    fn configured() -> Hub<Two> {
        let mut hub = Hub::new(
            Two {
                functions: Default::default(),
                speeds: [Speed::Full, Speed::High],
                attached: [true, true],
            },
            "HUB1",
        )
        .unwrap();
        let device = transfer(&mut hub, setup(0x80, GET_DESCRIPTOR, 0x100, 0, 18), &[]).unwrap();
        assert_eq!((device[4], device[5], device[6]), (9, 0, 1));
        transfer(&mut hub, setup(0, SET_ADDRESS, 3, 0, 0), &[]).unwrap();
        transfer(&mut hub, setup(0, SET_CONFIGURATION, 1, 0, 0), &[]).unwrap();
        assert_eq!(
            transfer(&mut hub, setup(0x80, GET_CONFIGURATION, 0, 0, 1), &[]).unwrap(),
            [1]
        );
        hub
    }

    fn port_status(hub: &mut Hub<Two>, port: u16) -> (u16, u16) {
        let bytes = transfer(hub, setup(0xa3, GET_STATUS, 0, port, 4), &[]).unwrap();
        (
            u16::from_le_bytes([bytes[0], bytes[1]]),
            u16::from_le_bytes([bytes[2], bytes[3]]),
        )
    }

    fn status_change(hub: &mut Hub<Two>) -> Reply {
        let mut byte = [0u8; 1];
        let reply = hub.packet(STATUS_ENDPOINT, Token::In, &mut byte).unwrap();
        if let Reply::Ack(1) = reply {
            return Reply::Ack(usize::from(byte[0]));
        }
        reply
    }

    #[test]
    fn descriptors_describe_a_two_port_single_tt_hub() {
        let mut hub = configured();
        let hub_descriptor =
            transfer(&mut hub, setup(0xa0, GET_DESCRIPTOR, 0x2900, 0, 71), &[]).unwrap();
        assert_eq!(hub_descriptor, [9, 0x29, 2, 0x89, 0, 5, 50, 0, 0xff]);
        let qualifier = transfer(&mut hub, setup(0x80, GET_DESCRIPTOR, 0x600, 0, 10), &[]).unwrap();
        assert_eq!(qualifier[6], 0, "no TT at full speed");
        let config = transfer(&mut hub, setup(0x80, GET_DESCRIPTOR, 0x200, 0, 255), &[]).unwrap();
        assert_eq!(&config[9..18], &[9, INTERFACE, 0, 0, 1, 9, 0, 0, 0]);
        assert_eq!(config[24], HIGH_SPEED_INTERVAL);
        assert_eq!(
            transfer(&mut hub, setup(0xa0, GET_STATUS, 0, 0, 4), &[]).unwrap(),
            [0; 4]
        );
    }

    #[test]
    fn powered_ports_connect_reset_and_route_to_their_functions() {
        let mut hub = configured();
        assert_eq!(status_change(&mut hub), Reply::Nak);
        assert!(hub.downstream(1).is_none());
        for port in 1..=2 {
            transfer(&mut hub, setup(0x23, SET_FEATURE, PORT_POWER, port, 0), &[]).unwrap();
        }
        assert_eq!(hub.ready_at(STATUS_ENDPOINT), Some(0));
        assert_eq!(status_change(&mut hub), Reply::Ack(0b110));
        assert_eq!(port_status(&mut hub, 1), (0x0101, 0x0001));
        transfer(
            &mut hub,
            setup(0x23, CLEAR_FEATURE, C_PORT_CONNECTION, 1, 0),
            &[],
        )
        .unwrap();
        transfer(&mut hub, setup(0x23, SET_FEATURE, PORT_RESET, 1, 0), &[]).unwrap();
        assert_eq!(port_status(&mut hub, 1), (0x0103, 0x0010));
        assert_eq!(hub.ports.functions[0].resets, 1);
        transfer(
            &mut hub,
            setup(0x23, CLEAR_FEATURE, C_PORT_RESET, 1, 0),
            &[],
        )
        .unwrap();
        transfer(&mut hub, setup(0x23, SET_FEATURE, PORT_RESET, 2, 0), &[]).unwrap();
        // The high-speed function reports high speed once enabled.
        assert_eq!(port_status(&mut hub, 2).0, 0x0503);
        assert!(hub.downstream(1).is_some() && hub.downstream_ref(2).is_some());
        assert!(hub.downstream(3).is_none() && hub.downstream(0).is_none());
        hub.set_time(77);
        assert_eq!(hub.ports.functions[1].now, 77);
        // Suspend hides the function until resume, which reports a change.
        transfer(&mut hub, setup(0x23, SET_FEATURE, PORT_SUSPEND, 1, 0), &[]).unwrap();
        assert!(hub.downstream(1).is_none());
        transfer(
            &mut hub,
            setup(0x23, CLEAR_FEATURE, PORT_SUSPEND, 1, 0),
            &[],
        )
        .unwrap();
        assert_eq!(port_status(&mut hub, 1), (0x0103, 0x0004));
        assert!(hub.downstream(1).is_some());
    }

    #[test]
    fn detach_power_off_and_unsupported_features() {
        let mut hub = configured();
        transfer(&mut hub, setup(0x23, SET_FEATURE, PORT_POWER, 1, 0), &[]).unwrap();
        transfer(&mut hub, setup(0x23, SET_FEATURE, PORT_RESET, 1, 0), &[]).unwrap();
        hub.ports.attached[0] = false;
        assert_eq!(hub.ready_at(STATUS_ENDPOINT), Some(0));
        hub.set_time(1);
        assert_eq!(port_status(&mut hub, 1), (0x0100, 0x0011));
        assert!(hub.downstream(1).is_none());
        transfer(&mut hub, setup(0x23, CLEAR_FEATURE, PORT_POWER, 1, 0), &[]).unwrap();
        assert_eq!(port_status(&mut hub, 1).0, 0);
        // Test selector 0 or 6, indicator selector 4, GET_TT_STATE of a
        // running TT and SET_HUB_DEPTH are request errors.
        for (feature, index) in [
            (TEST_FEATURE, 0x0001),
            (TEST_FEATURE, 0x0601),
            (INDICATOR_FEATURE, 0x0401),
        ] {
            let request = setup(0x23, SET_FEATURE, feature, index, 0);
            assert!(transfer(&mut hub, request, &[]).is_err());
        }
        assert!(transfer(&mut hub, setup(0xa3, GET_TT_STATE, 0, 1, 4), &[]).is_err());
        assert!(transfer(&mut hub, setup(0x20, 12, 1, 0, 0), &[]).is_err());
        assert!(transfer(&mut hub, setup(0x23, SET_FEATURE, PORT_POWER, 3, 0), &[]).is_err());
        transfer(&mut hub, setup(0x23, CLEAR_TT_BUFFER, 0x0081, 1, 0), &[]).unwrap();
        transfer(&mut hub, setup(0x23, RESET_TT, 0, 1, 0), &[]).unwrap();
        // A halted status endpoint STALLs until cleared.
        transfer(&mut hub, setup(0x02, SET_FEATURE, 0, 0x81, 0), &[]).unwrap();
        assert_eq!(status_change(&mut hub), Reply::Stall);
        transfer(&mut hub, setup(0x02, CLEAR_FEATURE, 0, 0x81, 0), &[]).unwrap();
        assert_eq!(status_change(&mut hub), Reply::Ack(0b10));
        hub.reset().unwrap();
        assert_eq!(hub.state.ports, [Port::default(); 2]);
    }

    /// Both ports powered, connected and reset; port 1 full speed, port 2
    /// high speed.
    fn enabled() -> Hub<Two> {
        let mut hub = configured();
        for port in 1..=2 {
            transfer(&mut hub, setup(0x23, SET_FEATURE, PORT_POWER, port, 0), &[]).unwrap();
            transfer(&mut hub, setup(0x23, SET_FEATURE, PORT_RESET, port, 0), &[]).unwrap();
        }
        hub
    }

    #[test]
    fn test_mode_holds_a_port_until_the_hub_resets() {
        let mut hub = enabled();
        // An enabled port refuses test mode; a suspended one enters it.
        let test_packet = setup(0x23, SET_FEATURE, TEST_FEATURE, 0x0401, 0);
        assert!(transfer(&mut hub, test_packet, &[]).is_err());
        transfer(&mut hub, setup(0x23, SET_FEATURE, PORT_SUSPEND, 1, 0), &[]).unwrap();
        transfer(&mut hub, test_packet, &[]).unwrap();
        assert_ne!(port_status(&mut hub, 1).0 & bit(PORT_TEST), 0);
        assert!(hub.downstream(1).is_none());
        // Port requests other than GET_STATUS fail until the hub resets.
        for request in [
            setup(0x23, CLEAR_FEATURE, PORT_SUSPEND, 1, 0),
            setup(0x23, SET_FEATURE, PORT_RESET, 1, 0),
        ] {
            assert!(transfer(&mut hub, request, &[]).is_err());
        }
        // A disconnected port enters test mode too; the hub reset ends both.
        hub.ports.attached[1] = false;
        hub.set_time(1);
        transfer(
            &mut hub,
            setup(0x23, SET_FEATURE, TEST_FEATURE, 0x0102, 0),
            &[],
        )
        .unwrap();
        hub.reset().unwrap();
        assert!(hub.state.ports.iter().all(|port| !port.testing()));
    }

    #[test]
    fn port_indicators_follow_software_selectors() {
        let mut hub = enabled();
        let green = setup(0x23, SET_FEATURE, INDICATOR_FEATURE, 0x0202, 0);
        transfer(&mut hub, green, &[]).unwrap();
        assert_eq!(port_status(&mut hub, 2).0, 0x1503);
        // Selector 0 and CLEAR_FEATURE both return to automatic.
        let automatic = setup(0x23, SET_FEATURE, INDICATOR_FEATURE, 0x0002, 0);
        transfer(&mut hub, automatic, &[]).unwrap();
        assert_eq!(port_status(&mut hub, 2).0, 0x0503);
        transfer(&mut hub, green, &[]).unwrap();
        let clear = setup(0x23, CLEAR_FEATURE, INDICATOR_FEATURE, 2, 0);
        transfer(&mut hub, clear, &[]).unwrap();
        assert_eq!(port_status(&mut hub, 2).0, 0x0503);
        // The function still takes transactions.
        assert!(hub.downstream(2).is_some());
    }

    #[test]
    fn a_stopped_tt_reports_its_state_and_cuts_off_split_functions() {
        let mut hub = enabled();
        assert!(transfer(&mut hub, setup(0x23, STOP_TT, 0, 2, 0), &[]).is_err());
        transfer(&mut hub, setup(0x23, STOP_TT, 0, 1, 0), &[]).unwrap();
        // The full-speed function behind the TT is cut off; the high-speed
        // one is not.
        assert!(hub.downstream(1).is_none());
        assert!(hub.downstream(2).is_some());
        let state = transfer(&mut hub, setup(0xa3, GET_TT_STATE, 0, 1, 4), &[]).unwrap();
        assert_eq!(state, TT_STATE);
        assert!(transfer(&mut hub, setup(0xa3, GET_TT_STATE, 1, 1, 4), &[]).is_err());
        transfer(&mut hub, setup(0x23, RESET_TT, 0, 1, 0), &[]).unwrap();
        assert!(hub.downstream(1).is_some());
        assert!(transfer(&mut hub, setup(0xa3, GET_TT_STATE, 0, 1, 4), &[]).is_err());
    }
}
