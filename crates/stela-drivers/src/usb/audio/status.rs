// SPDX-License-Identifier: GPL-2.0-or-later
//! Optional UAC1/UAC2 AudioControl status interrupts. Layout/receive/requeue reference:
//! Linux sound/usb/mixer.c, Copyright (c) 2002 Takashi Iwai, and original audio.c
//! contributions by Alan Cox and Thomas Sailer. See NOTICE.
//! Notifications invalidate observations; they never supply a new control value.
use super::{Binding, Entity, Error, Graph, UAC1, UAC2};
use crate::{
    bus::dma::{DmaAllocator, DmaBuffer, RegionAllocator},
    usb::descriptor::Endpoint,
    usb::xhci::{self, Shared, SlotId, TransferTicket},
};
use alloc::{collections::VecDeque, vec::Vec};

const MESSAGE_BYTES: usize = 6;
const MAX_PACKET: usize = 1024;

/// A changed attribute, not its replacement value. Requery through the existing
/// descriptor-checked CUR/RANGE control APIs before changing audio policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StatusAttribute {
    Current,
    Range,
}
/// An invalidation from an advertised readable control of this AC function.
/// Receipt does not authorize changing a live PCM clock, alternate or mixer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StatusNotice {
    pub attribute: StatusAttribute,
    pub entity: u8,
    pub selector: u8,
    pub channel: u8,
    pub raw: [u8; MESSAGE_BYTES],
}

/// UAC1 reports an entire entity, without a selector, channel or replacement
/// value. The memory flag names a memory-space change; it grants no vendor
/// memory protocol or permission to change a running stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Uac1StatusNotice {
    pub entity: u8,
    pub memory_changed: bool,
    pub interrupt_pending: bool,
    pub raw: [u8; 2],
}

/// The protocol determines the wire layout and invalidation scope.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StatusEvent {
    Uac1(Uac1StatusNotice),
    Uac2(StatusNotice),
}

#[derive(Clone)]
pub(super) struct StatusPlan {
    endpoint: Endpoint,
    graph: Graph,
    connectors: super::connector::ConnectorPlan,
}
impl StatusPlan {
    pub(super) fn discover(binding: &Binding, configuration: &[u8]) -> Result<Option<Self>, Error> {
        if !matches!(binding.graph.protocol, UAC1 | UAC2) {
            return Ok(None);
        }
        let interfaces = crate::usb::descriptor::interfaces(configuration);
        // Binding::parse already checked complete configuration/interface counts,
        // and associations before native attachment. Status endpoint aliases are
        // checked separately below (the binding only checked streaming DCIs).
        let control = interfaces
            .iter()
            .find(|i| i.number == binding.graph.control_interface && i.alternate == 0)
            .ok_or(Error::Descriptor("missing audio status control interface"))?;
        if control.class != super::CLASS
            || control.subclass != super::CONTROL_SUBCLASS
            || control.protocol != binding.graph.protocol
        {
            return Err(Error::Descriptor("foreign audio status control interface"));
        }
        if control.endpoints.is_empty() {
            return Ok(None);
        }
        if control.endpoints.len() != 1 {
            return Err(Error::Unsupported("multiple audio status endpoints"));
        }
        let endpoint = control.endpoints[0];
        if !endpoint.is_in()
            || endpoint.attributes != 3
            || endpoint.address & 0x70 != 0
            || endpoint.address & 15 == 0
            || endpoint.max_packet & 0x1800 != 0
            || usize::from(endpoint.max_packet)
                < if binding.graph.protocol == UAC1 {
                    2
                } else {
                    MESSAGE_BYTES
                }
            || usize::from(endpoint.max_packet) > MAX_PACKET
            || !matches!(binding.speed, xhci::FULL_SPEED | xhci::HIGH_SPEED)
        {
            return Err(Error::Unsupported("audio status endpoint profile"));
        }
        endpoint.validate_context(binding.speed)?;
        if interfaces.iter().any(|i| {
            (i.number != control.number || i.alternate != 0)
                && i.endpoints.iter().any(|ep| ep.dci() == endpoint.dci())
        }) {
            return Err(Error::Descriptor(
                "audio status endpoint aliases another interface",
            ));
        }
        Ok(Some(Self {
            endpoint,
            graph: binding.graph.clone(),
            connectors: super::connector::ConnectorPlan::discover(binding, configuration)?,
        }))
    }
    fn message_bytes(&self) -> usize {
        if self.graph.protocol == UAC1 {
            2
        } else {
            MESSAGE_BYTES
        }
    }
    fn event(&self, raw: &[u8]) -> Result<StatusEvent, Error> {
        if self.graph.protocol == UAC1 {
            let raw: [u8; 2] = raw
                .try_into()
                .map_err(|_| Error::Packet("truncated UAC1 status word"))?;
            // bStatusType: bits 7/6 are pending/memory; bits 5/4 are
            // reserved; bits 3:0 identify AC, AS-interface or AS-endpoint.
            // AS origins need separate interface/endpoint status protocols.
            if raw[0] & 0x3f != 0 {
                return Err(Error::Unsupported("non-AC/reserved UAC1 status origin"));
            }
            self.graph.entity(raw[1])?;
            Ok(StatusEvent::Uac1(Uac1StatusNotice {
                entity: raw[1],
                memory_changed: raw[0] & 0x40 != 0,
                interrupt_pending: raw[0] & 0x80 != 0,
                raw,
            }))
        } else {
            let raw = raw
                .try_into()
                .map_err(|_| Error::Packet("truncated UAC2 status message"))?;
            self.notice(raw).map(StatusEvent::Uac2)
        }
    }
    fn notice(&self, raw: [u8; MESSAGE_BYTES]) -> Result<StatusNotice, Error> {
        if raw[0] != 0 {
            return Err(Error::Unsupported(
                "vendor/endpoint/reserved UAC2 status notice",
            ));
        }
        let attribute = match raw[1] {
            1 => StatusAttribute::Current,
            2 => StatusAttribute::Range,
            _ => return Err(Error::Unsupported("UAC2 memory/unknown status attribute")),
        };
        if raw[4] != self.graph.control_interface || raw[5] == 0 {
            return Err(Error::Descriptor("foreign UAC2 status origin"));
        }
        let selector = raw[3];
        let channel = raw[2];
        let access = |bits: u32, count: u8| {
            if selector == 0 || selector > count {
                return false;
            }
            matches!((bits >> (u32::from(selector - 1) * 2)) & 3, 1 | 3)
        };
        let readable = match self.graph.entity(raw[5])? {
            Entity::ClockSource { controls, .. } | Entity::ClockMultiplier { controls, .. } => {
                channel == 0 && access(u32::from(*controls), 2)
            }
            Entity::ClockSelector { controls, .. } | Entity::Selector { controls, .. } => {
                channel == 0 && access(u32::from(*controls), 1)
            }
            Entity::Feature { controls, .. }
                if selector <= 2 || super::FeatureControl::from_selector(selector).is_some() =>
            {
                controls
                    .get(usize::from(channel))
                    .is_some_and(|bits| access(*bits, 16))
            }
            Entity::Input { id, .. } | Entity::Output { id, .. } => {
                selector == 2 && channel == 0 && self.connectors.supports(*id)
            }
            // Mixer MCN uses a different address convention.
            _ => return Err(Error::Unsupported("UAC2 status entity control layout")),
        };
        if !readable {
            return Err(Error::Descriptor(
                "unadvertised UAC2 status control/channel",
            ));
        }
        let range_supported = matches!(self.graph.entity(raw[5])?,
            Entity::ClockSource { .. } if selector == 1)
            || matches!(self.graph.entity(raw[5])?, Entity::Feature { .. }
                if selector == 2 || super::FeatureControl::from_selector(selector).is_some_and(|feature| feature.has_range()));
        if attribute == StatusAttribute::Range && !range_supported {
            return Err(Error::Unsupported("UAC2 status RANGE on CUR-only control"));
        }
        Ok(StatusNotice {
            attribute,
            entity: raw[5],
            selector,
            channel,
            raw,
        })
    }
}

/// Separate, opt-in native owner for one descriptor-selected interrupt endpoint.
/// Exactly one receive can be pending. One complete packet (at most 512 events)
/// supplies bounded backpressure: drain it before polling rearms the endpoint.
/// All DMA allocations live for the boot even if this object is dropped. Drop
/// does not retire hardware or free its endpoint; call `stop` for acknowledged
/// teardown. Failed teardown keeps the poisoned owner for platform recovery. A poisoned native endpoint cannot be revived here.
pub struct StatusNotifications {
    shared: Shared,
    id: SlotId,
    plan: StatusPlan,
    _arena: DmaBuffer,
    allocator: RegionAllocator,
    packet: DmaBuffer,
    ticket: Option<TransferTicket>,
    notices: VecDeque<StatusEvent>,
    configured: bool,
    poisoned: bool,
    cancelled: bool,
}
impl StatusNotifications {
    pub(super) fn new(
        shared: Shared,
        id: SlotId,
        plan: StatusPlan,
        dma: &mut dyn DmaAllocator,
    ) -> Result<Self, Error> {
        shared.borrow_mut().device(id)?;
        let arena = dma.allocate(65536, 65536)?;
        // SAFETY: exclusively reserved live boot memory, never rewound.
        let allocator = unsafe { RegionAllocator::within(&arena) };
        let bytes = usize::from(plan.endpoint.max_packet);
        // One ordinary TD cannot cross a 64 KiB boundary. A power-of-two
        // alignment at least the bounded packet size proves that for any DMA
        // allocator placement, including fragmented physical reservations.
        let packet = dma.allocate(bytes, bytes.next_power_of_two().max(64))?;
        Ok(Self {
            shared,
            id,
            plan,
            _arena: arena,
            allocator,
            packet,
            ticket: None,
            notices: VecDeque::new(),
            configured: false,
            poisoned: false,
            cancelled: false,
        })
    }
    /// Configure and arm. An uncertain publication leaves this object poisoned,
    /// keeping its DMA and staged endpoint for `stop` attempt; unresolved native failures require platform recovery.
    pub fn start(&mut self) -> Result<(), Error> {
        let mut controller = self.shared.borrow_mut();
        let mut device = controller.device(self.id)?;
        if self.poisoned {
            return Err(Error::Poisoned);
        }
        if self.configured {
            return Err(Error::Busy);
        }
        let endpoints = [self.plan.endpoint];
        device.preflight_endpoints(&endpoints)?;
        self.configured = true;
        if let Err(error) = device.configure(&endpoints, None, &mut self.allocator) {
            self.poisoned = true;
            return Err(error.into());
        }
        self.notices.clear();
        self.cancelled = false;
        match device.submit_ticket(
            self.plan.endpoint.dci(),
            self.packet.physical(),
            self.packet.len(),
        ) {
            Ok(ticket) => self.ticket = Some(ticket),
            Err(error) => {
                self.poisoned = true;
                return Err(error.into());
            }
        }
        Ok(())
    }
    /// Nonblocking: a NAK supplies no event and preserves the original ticket.
    /// A malformed complete packet publishes no prefix and may be polled again.
    pub fn poll(&mut self) -> Result<usize, Error> {
        let mut controller = self.shared.borrow_mut();
        let mut device = controller.device(self.id)?;
        if self.poisoned {
            return Err(Error::Poisoned);
        }
        if !self.configured {
            return Err(Error::NotRunning);
        }
        if !self.notices.is_empty() {
            return Ok(0);
        }
        if self.ticket.is_none() {
            match device.submit_ticket(
                self.plan.endpoint.dci(),
                self.packet.physical(),
                self.packet.len(),
            ) {
                Ok(ticket) => self.ticket = Some(ticket),
                Err(error) => {
                    self.poisoned = true;
                    return Err(error.into());
                }
            }
        }
        let Some(done) = device.completed_ticket(self.ticket.as_ref().unwrap()) else {
            return Ok(0);
        };
        let bytes = match done {
            Ok(bytes) => bytes,
            Err(error) => {
                self.poisoned = true;
                return Err(error.into());
            }
        };
        self.ticket = None;
        let message_bytes = self.plan.message_bytes();
        if bytes > self.packet.len() || !bytes.is_multiple_of(message_bytes) {
            return Err(Error::Packet("truncated audio status packet"));
        }
        let mut notices = Vec::with_capacity(bytes / message_bytes);
        for offset in (0..bytes).step_by(message_bytes) {
            let mut raw = [0; MESSAGE_BYTES];
            for (n, value) in raw.iter_mut().take(message_bytes).enumerate() {
                *value = self.packet.read::<u8>(offset + n);
            }
            notices.push(self.plan.event(&raw[..message_bytes])?);
        }
        let count = notices.len();
        self.notices.extend(notices);
        Ok(count)
    }
    /// UAC2 compatibility accessor. UAC1 has entity-wide events; use
    /// `next_event` for those. A protocol mismatch never consumes the queue.
    pub fn next_notice(&mut self) -> Result<Option<StatusNotice>, Error> {
        self.shared.borrow_mut().device(self.id)?;
        if self.poisoned {
            return Err(Error::Poisoned);
        }
        if self.plan.graph.protocol != UAC2 {
            return Err(Error::Unsupported(
                "UAC1 status requires entity event accessor",
            ));
        }
        match self.notices.pop_front() {
            Some(StatusEvent::Uac2(notice)) => Ok(Some(notice)),
            None => Ok(None),
            Some(StatusEvent::Uac1(_)) => unreachable!("protocol-bound status queue"),
        }
    }
    /// Current identity is checked even when the queue is empty or cached.
    pub fn next_event(&mut self) -> Result<Option<StatusEvent>, Error> {
        self.shared.borrow_mut().device(self.id)?;
        if self.poisoned {
            return Err(Error::Poisoned);
        }
        Ok(self.notices.pop_front())
    }
    /// Retire pending DMA, drain events and drop exactly this endpoint context.
    /// Only acknowledgement permits queue/ticket reset and subsequent restart.
    pub fn stop(&mut self) -> Result<(), Error> {
        let mut controller = self.shared.borrow_mut();
        let mut device = controller.device(self.id)?;
        // A terminal transfer error may require Reset Endpoint and Set
        // Dequeue before ordinary submission-readiness permits Drop. The exact
        // ticket authorizes cancellation without weakening that fence.
        if !self.cancelled
            && let Some(ticket) = &self.ticket
        {
            if let Err(error) = device.cancel_ticket(ticket) {
                self.poisoned = true;
                return Err(error.into());
            }
            self.cancelled = true;
        }
        // Reset Endpoint retires xHC DMA, but the device's endpoint HALT is a
        // separate USB feature. An unpolled STALL can race cancellation, so clear
        // it for every configured endpoint, including already-clear devices.
        // Require EP0 acknowledgement, including a later explicit retry.
        if self.configured
            && let Err(error) = device.clear(&self.plan.endpoint)
        {
            self.poisoned = true;
            return Err(error.into());
        }
        if self.configured
            && let Err(error) = device.deconfigure(&[self.plan.endpoint])
        {
            self.poisoned = true;
            return Err(error.into());
        }
        self.ticket = None;
        self.notices.clear();
        self.configured = false;
        self.poisoned = false;
        self.cancelled = false;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
