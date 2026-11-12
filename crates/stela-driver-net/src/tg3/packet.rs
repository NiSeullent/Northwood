// SPDX-License-Identifier: GPL-2.0-only
//! Single-queue BCM5720/5761/5762 A0 copper Ethernet backend.
use super::{Delay, Error, Hardware, ReceiveFilter, Registers};
use crate::bus::pci::requester::preparation::Attachment;
use crate::{
    HEADER, Link, LinkError, MAX_FRAME, MIN_FRAME, Mac,
    bus::dma::{
        mapping::DmaDomain,
        memory::{MemoryError, PageSource},
    },
    check_frame,
};
use alloc::{sync::Arc, vec::Vec};
mod memory;
use super::geometry::Geometry;
#[cfg(test)]
use memory::RX;
use memory::{Memory, TX};

const SD_STATUS_ERROR: u32 = 4;

#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    Building,
    Running,
    Faulted,
    Stopped,
    PhySuspended,
}
/// PCI/config/MMIO ownership, isolated requester domain and pinned rings
/// form one lifetime. An uncertain epoch keeps all of them. No IRQs or worker
/// callbacks are registered by this polled backend.
pub struct Packet<R: Registers, D: Delay> {
    hardware: Hardware<R, D>,
    memory: Option<Memory>,
    domain: Arc<dyn DmaDomain>,
    attachment: Option<Attachment>,
    mac: Mac,
    board: Option<super::transport::board::Board>,
    carrier_up: bool,
    state: State,
    failure: Option<Error>,
    mbuf_low_water_events: u64,
    tx_producer: usize,
    tx_consumer: usize,
    rx_producer: usize,
    rx_consumer: usize,
    /// True only after all preconditions for attempted device containment have
    /// been observed; arbitrary unsupported hardware is never stopped here.
    identified: bool,
    /// An attach error can keep partial DMA allocations. Only a fully
    /// initialized packet owner may rebuild those rings for another epoch.
    restartable: bool,
    geometry: Geometry,
    receive_filter: ReceiveFilter,
    /// The last received frame. The stack's path swaps it with its own
    /// buffer, so steady-state receive allocates nothing, and a failed
    /// receive never touches the caller's buffer.
    received: Vec<u8>,
}
impl<R: Registers, D: Delay> Packet<R, D> {
    /// Attach an exclusively owned stopped requester, with real NVM
    /// bootcode already present. Errors return the whole epoch.
    ///
    /// # Safety
    /// `hardware` owns the BCM5720/5761/5762 function/config/UC MAC/APE resources
    /// described by `Registers`, powered in PCI D0. `domain` is the
    /// isolated requester domain for that function/aliases; it keeps its root
    /// and performs real invalidation, coherence and payload write drain.
    /// `pages` supplies kept pins, never synthetic physical addresses.
    /// Inherited DMA is stopped/drained, BusMaster clear, MSI/MSI-X disabled and
    /// posted messages/CPU vectors/workers drained before this call. Those
    /// original interrupt exclusions remain held in the register
    /// owner through retirement; no other provider may re-enable them.
    /// The kept timer is the original provider's calibrated clock;
    /// its rate and CPU/power context remain valid through all settling waits.
    /// All calls/releases run in the original provider's checked context.
    /// A numeric BDF, arbitrary Arc or fake domain cannot grant this permission.
    #[allow(
        clippy::result_large_err,
        reason = "attach failure returns the complete original hardware/DMA epoch without a fallible error-path allocation"
    )]
    pub unsafe fn attach(
        hardware: Hardware<R, D>,
        pages: &dyn PageSource,
        domain: Arc<dyn DmaDomain>,
    ) -> Result<Self, (Error, Self)> {
        let mut value = Self {
            hardware,
            memory: None,
            domain,
            attachment: None,
            mac: Mac::default(),
            board: None,
            carrier_up: false,
            state: State::Building,
            failure: None,
            mbuf_low_water_events: 0,
            tx_producer: 0,
            tx_consumer: 0,
            rx_producer: Geometry::of(super::Asic::Bcm5720).rx - 1,
            rx_consumer: 0,
            identified: false,
            restartable: false,
            geometry: Geometry::of(super::Asic::Bcm5720),
            receive_filter: ReceiveFilter::default(),
            received: Vec::new(),
        };
        let result = (|| {
            let board = value.hardware.packet_prepare()?;
            value.identified = true;
            value.mac = board.mac;
            value.board = Some(board);
            value.geometry =
                Geometry::of(value.hardware.identity().ok_or(Error::NotIdentified)?.asic);
            value.rx_producer = value.geometry.rx - 1;
            value.hardware.packet_reset(board)?;
            match Memory::new(pages, value.domain.clone(), value.geometry) {
                Ok(memory) => value.memory = Some(memory),
                Err((error, memory)) => {
                    value.memory = Some(memory);
                    return Err(error);
                }
            }
            let addresses = value.memory.as_ref().unwrap().addresses()?;
            value
                .hardware
                .packet_start(board, addresses, &value.receive_filter)?;
            Ok(())
        })();
        match result {
            Ok(()) => {
                value.restartable = true;
                value.state = State::Running;
                Ok(value)
            }
            Err(error) => {
                value.fault(error);
                Err((error, value))
            }
        }
    }
    /// Attach through the unique control produced by cold original-root
    /// finalization. Retirement detaches it after every DMA mapping retires.
    /// # Safety
    /// All `attach` requirements hold, and `attachment` is paired with this
    /// exact original hardware/configuration epoch. The inherited interrupt
    /// exclusions remain held through detach; this backend installs no IRQs.
    #[expect(
        clippy::result_large_err,
        reason = "attach errors retain the complete original hardware and attachment"
    )]
    pub unsafe fn attach_bound(
        hardware: Hardware<R, D>,
        pages: &dyn PageSource,
        attachment: Attachment,
    ) -> Result<Self, (Error, Self)> {
        // SAFETY: the caller supplies the same original stopped hardware,
        // IRQ/work exclusion and unique attachment. Keep it through all errors.
        match unsafe { Self::attach(hardware, pages, attachment.domain()) } {
            Ok(mut packet) => {
                packet.attachment = Some(attachment);
                Ok(packet)
            }
            Err((error, mut packet)) => {
                packet.attachment = Some(attachment);
                Err((error, packet))
            }
        }
    }

    /// Program an acknowledged hardware receive policy in this live epoch.
    /// Already completed frames are not retracted. On error, the uncertain
    /// original epoch is held and packet publication remains fenced.
    pub fn set_receive_filter(&mut self, filter: ReceiveFilter) -> Result<(), Error> {
        self.check()?;
        filter.validate()?;
        match self.hardware.packet_receive_filter(&filter) {
            Ok(()) => {
                self.receive_filter = filter;
                Ok(())
            }
            Err(error) => Err(self.fault(error)),
        }
    }
    /// Last acknowledged request; multicast collisions/unused unicast slots
    /// still have the pinned hardware semantics described by ReceiveFilter.
    pub fn receive_filter(&self) -> &ReceiveFilter {
        &self.receive_filter
    }
    pub fn failure(&self) -> Option<Error> {
        self.failure
    }
    /// Successfully handled BCM5720/5762 A0 MBUF low-water discard indications,
    /// acknowledged through the real W1C register. Multiple drops can coalesce
    /// into one event; this is not an exact packet-loss counter. Faulted cause
    /// or acknowledgement observations remain in `failure`. Saturates at u64::MAX.
    pub fn mbuf_low_water_events(&self) -> u64 {
        self.mbuf_low_water_events
    }
    fn fault(&mut self, error: Error) -> Error {
        self.failure.get_or_insert(error);
        self.state = State::Faulted;
        error
    }
    fn check(&self) -> Result<(), Error> {
        if self.state == State::Running {
            Ok(())
        } else {
            Err(Error::Stopped)
        }
    }
    fn poll(&mut self) -> Result<usize, Error> {
        self.check()?;
        self.hardware.request_status()?;
        let (flags, tx, rx) = self.memory.as_ref().unwrap().status()?;
        if flags & SD_STATUS_ERROR != 0 && self.hardware.packet_attention()? {
            self.mbuf_low_water_events = self.mbuf_low_water_events.saturating_add(1);
        }
        if tx >= TX || rx >= self.geometry.returns {
            return Err(Error::RingCorrupt);
        }
        let done = (tx + TX - self.tx_consumer) % TX;
        let pending = (self.tx_producer + TX - self.tx_consumer) % TX;
        if done > pending {
            return Err(Error::RingCorrupt);
        }
        for _ in 0..done {
            self.memory
                .as_ref()
                .unwrap()
                .complete_tx(self.tx_consumer)?;
            self.tx_consumer = (self.tx_consumer + 1) % TX;
        }
        Ok(rx)
    }
    /// Stop MAC/DMA engines, observe BusMaster clear and complete the original
    /// remapper's posted payload drain. A failed attempt remains retryable only
    /// for containment; it can never resume submission or invent completion.
    pub fn stop(&mut self) -> Result<(), Error> {
        if matches!(self.state, State::Stopped | State::PhySuspended) {
            return Ok(());
        }
        self.state = State::Faulted;
        if !self.identified {
            return Err(self.failure.unwrap_or(Error::NotIdentified));
        }
        let result = (|| {
            self.hardware.packet_stop()?;
            // SAFETY: real engines acknowledged stopped, original BME clear;
            // isolated domain and every packet/status target remain.
            unsafe {
                self.domain.drain_posted_writes()?;
            }
            Ok(())
        })();
        match result {
            Ok(()) => {
                self.state = State::Stopped;
                Ok(())
            }
            Err(error) => Err(self.fault(error)),
        }
    }
    /// Stop/drain packets and put the admitted copper PHY into its real low
    /// power state. The original PCI function stays in D0 with every mapping
    /// kept. This supplies no PCI/system sleep or wake-on-LAN operation.
    pub fn suspend_phy(&mut self) -> Result<(), Error> {
        if self.state == State::PhySuspended {
            return Ok(());
        }
        if !self.restartable {
            return Err(Error::Stopped);
        }
        self.stop()?;
        self.state = State::Faulted;
        self.carrier_up = false;
        match self.hardware.power_down_packet_phy() {
            Ok(()) => {
                self.state = State::PhySuspended;
                Ok(())
            }
            Err(error) => Err(self.fault(error)),
        }
    }
    /// Restore this same PHY and packet epoch through the full PHY/core reset
    /// and queue rebuild. The kept platform owner must still hold PCI D0.
    pub fn resume_phy(&mut self) -> Result<(), Error> {
        if self.state != State::PhySuspended {
            return Err(Error::Stopped);
        }
        self.restart()
    }
    /// Release only after stop/drain and acknowledged removal of every real
    /// mapping. An unmap error returns this same epoch with original owners.
    #[allow(
        clippy::result_large_err,
        reason = "failed retirement must return the same retained hardware/DMA epoch for retry without allocating"
    )]
    pub fn retire(mut self) -> Result<R, (Error, Self)> {
        let result = self.stop().and_then(|()| {
            // Unmap can succeed for some rings before another mapping fails.
            // That owner is retryable only for retirement, never restart.
            self.restartable = false;
            if let Some(memory) = &mut self.memory {
                memory.retire()?;
            }
            if let Some(attachment) = &mut self.attachment {
                // SAFETY: stop observed vendor engine/BME retirement and the
                // original-root payload drain, every mapping is now unmapped,
                // and inherited IRQ/work exclusions remain held by the owner.
                unsafe {
                    attachment.detach()?;
                }
            }
            Ok(())
        });
        match result {
            Ok(()) => {
                self.memory.take();
                self.attachment.take();
                Ok(self.hardware.release_registers())
            }
            Err(error) => {
                self.failure.get_or_insert(error);
                Err((error, self))
            }
        }
    }
    /// Rebuild an explicitly stopped or failed packet queue in the same held-D0
    /// hardware owner. Outstanding frames are discarded only after
    /// engine stop, BusMaster-clear readback and posted-write drain. A failure
    /// keeps every mapping and prohibits packet I/O; another containment or
    /// restart attempt cannot bypass uncertain transport/reset ownership.
    pub fn restart(&mut self) -> Result<(), Error> {
        if !self.restartable {
            return Err(Error::Stopped);
        }
        self.stop()?;
        self.state = State::Faulted;
        self.carrier_up = false;
        match self.rebuild_after_drain() {
            Ok(()) => {
                self.failure = None;
                self.state = State::Running;
                Ok(())
            }
            Err(error) => Err(self.fault(error)),
        }
    }
    fn rebuild_after_drain(&mut self) -> Result<(), Error> {
        let board = self.board.ok_or(Error::NotIdentified)?;
        self.hardware.packet_reset(board)?;
        // SAFETY: the old epoch is fully stopped/drained and the chip reset;
        // the same real mappings remain, with no intervening new DMA source.
        unsafe {
            self.memory.as_mut().unwrap().reset_after_drain()?;
        }
        let addresses = self.memory.as_ref().unwrap().addresses()?;
        self.hardware
            .packet_start(board, addresses, &self.receive_filter)?;
        self.tx_producer = 0;
        self.tx_consumer = 0;
        self.rx_producer = self.geometry.rx - 1;
        self.rx_consumer = 0;
        Ok(())
    }
    fn transmit_inner(&mut self, frame: &[u8]) -> Result<(), Error> {
        self.poll()?;
        if (self.tx_producer + 1) % TX == self.tx_consumer {
            return Err(Error::RingFull);
        }
        self.memory.as_ref().unwrap().tx(self.tx_producer, frame)?;
        self.tx_producer = (self.tx_producer + 1) % TX;
        self.hardware.tx_producer(self.tx_producer)
    }
    /// Whether a frame arrived; it is left in `self.received`.
    fn receive_inner(&mut self) -> Result<bool, Error> {
        let producer = self.poll()?;
        // A finite pass drops damaged frames without blocking on new traffic.
        for _ in 0..self.geometry.returns {
            if self.rx_consumer == producer {
                return Ok(false);
            }
            let (_, frame) = self
                .memory
                .as_mut()
                .unwrap()
                .receive(self.rx_consumer, &mut self.received)?;
            // The next spare producer slot must really be CPU-owned; the
            // memory layer refuses an out-of-order return which violates it.
            self.memory.as_mut().unwrap().offer(self.rx_producer)?;
            self.rx_producer = (self.rx_producer + 1) % self.geometry.rx;
            self.hardware.rx_producer(self.rx_producer)?;
            self.rx_consumer = (self.rx_consumer + 1) % self.geometry.returns;
            self.hardware.rx_consumer(self.rx_consumer)?;
            if frame {
                return Ok(true);
            }
        }
        Err(Error::RingCorrupt)
    }
    /// Runs one receive; `Ok(true)` leaves the frame in `self.received`.
    fn receive_checked(&mut self) -> Result<bool, LinkError> {
        if !self.carrier() {
            return if let Some(error) = self.failure {
                Err(error.into_link())
            } else {
                Ok(false)
            };
        }
        self.receive_inner().map_err(|error| {
            self.fault(error);
            error.into_link()
        })
    }
}
impl<R: Registers, D: Delay> Link for Packet<R, D> {
    fn mac(&self) -> Mac {
        self.mac
    }
    fn carrier(&mut self) -> bool {
        if self.check().is_err() {
            return false;
        }
        match self.hardware.packet_carrier() {
            Ok(carrier) => {
                if !carrier && self.carrier_up {
                    self.carrier_up = false;
                    if let Err(error) = self.restart() {
                        self.fault(error);
                        return false;
                    }
                }
                self.carrier_up = carrier;
                carrier
            }
            Err(error) => {
                self.fault(error);
                false
            }
        }
    }
    fn transmit(&mut self, frame: &[u8]) -> Result<(), LinkError> {
        check_frame(frame)?;
        if !self.carrier() {
            return if let Some(error) = self.failure {
                Err(error.into_link())
            } else {
                Err(LinkError::NoCarrier)
            };
        }
        match self.transmit_inner(frame) {
            Ok(()) => Ok(()),
            Err(Error::RingFull) => Err(LinkError::Adapter("tg3 transmit ring full".into())),
            Err(error) => {
                self.fault(error);
                Err(error.into_link())
            }
        }
    }
    fn receive(&mut self) -> Result<Option<Vec<u8>>, LinkError> {
        Ok(self
            .receive_checked()?
            .then(|| core::mem::take(&mut self.received)))
    }
    /// The stack's path: the received frame and the caller's buffer trade
    /// places, so a received frame allocates nothing once both have grown.
    fn receive_reusing(&mut self, frame: &mut Vec<u8>) -> Result<bool, LinkError> {
        let received = self.receive_checked()?;
        if received {
            core::mem::swap(frame, &mut self.received);
        }
        Ok(received)
    }
}
#[cfg(test)]
mod tests;
