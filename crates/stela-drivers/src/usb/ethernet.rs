//! Vendor USB Ethernet backends. Candidate interfaces become usable links only
//! after exact chip identification and revision-specific initialization.
//! RTL8152 01/02/07, RTL8153A 03/04/05/06, RTL8153B 08/09, RTL8153C 14
//! and RTL8156A 10/11 and RTL8156B 12/13/15 have distinct implementations.
//! Revision15 follows the donor's gigabit capability policy. ASIX AX88179/178A
//! uses its own control protocol and packet framing behind the shared DMA path.
mod asix;
pub mod firmware;
mod framing;
mod identity;
mod minimum;
mod registers;
use minimum::MinimumControl;
pub mod rtl815x;

use super::descriptor::{Endpoint, Interface};
use super::xhci::{Device, Shared, SlotId, TransferTicket};
use crate::{
    block::BlockError,
    bus::dma::{DmaAllocator, DmaBuffer},
    net::{Link, LinkError, Mac},
};
use alloc::{collections::VecDeque, format, string::String, vec::Vec};
#[cfg(test)]
pub(crate) use asix::LegacyModel;
#[cfg(test)]
pub(crate) use asix::model::Hardware as AsixModel;
#[cfg(test)]
pub(crate) fn asix_legacy_ids() -> Vec<(u16, u16)> {
    asix::legacy_ids()
        .iter()
        .map(|entry| (entry.0, entry.1))
        .collect()
}
use core::sync::atomic::{Ordering, fence};
pub use firmware::Error;
#[cfg(test)]
pub(crate) use identity::USB_IDS;
pub use identity::matches;
use identity::policy;
use rtl815x::{Chip, Revision};

/// Whether a vendor Ethernet backend's ID table lists the device. Only listed
/// devices are adapters; other vendor-class interfaces, such as LED or sensor
/// controllers, belong to no driver here and get no chip identification.
pub fn listed(descriptor: &[u8; 18]) -> bool {
    matches(descriptor) || asix::listed(descriptor)
}

#[cfg(test)]
pub(crate) fn asix_ids() -> Vec<(u16, u16)> {
    asix::ids().iter().map(|entry| (entry.0, entry.1)).collect()
}

/// Synchronous endpoint-zero requests. Success includes the exact transferred
/// byte count; short replies/writes are errors, never invented register values.
pub trait Control {
    fn control(&mut self, request: [u8; 8], output: &[u8]) -> Result<Vec<u8>, BlockError>;
    /// Nominal polling cadence only; never authorizes a hardware minimum.
    fn delay_us(&mut self, microseconds: u64);
    /// Required physical quiet interval from this actual device's owner.
    fn minimum_wait(&mut self, microseconds: u64) -> Result<(), BlockError>;
}

#[derive(Clone, Debug)]
pub struct Binding {
    input: Endpoint,
    output: Endpoint,
    kind: Kind,
    setting: (u8, u8),
}

#[derive(Clone, Copy, Debug)]
enum Kind {
    Realtek(Revision),
    Asix(asix::Identity),
}

impl Binding {
    pub(super) fn new(device: &Device, interface: &Interface) -> Result<Self, &'static str> {
        let kind = if let Some(identity) = asix::identify(&device.descriptor, interface) {
            if matches!(identity, asix::Identity::Ax88772 { .. })
                && !matches!(
                    device.speed,
                    super::xhci::FULL_SPEED | super::xhci::HIGH_SPEED
                )
            {
                return Err("AX88772 requires USB full or high speed");
            }
            Kind::Asix(identity)
        } else {
            let revision = device
                .vendor_ethernet_revision
                .map(Revision)
                .ok_or("USB Ethernet chip revision not identified")?;
            if !matches(&device.descriptor) || !revision.implemented() {
                return Err("RTL815x chip revision initialization not implemented");
            }
            if interface.class != 0xff || interface.alternate != 0 || interface.number != 0 {
                return Err("RTL815x unsupported vendor interface setting");
            }
            Kind::Realtek(revision)
        };
        let input: Vec<_> = interface
            .endpoints
            .iter()
            .copied()
            .filter(|e| e.kind() == 2 && e.is_in())
            .collect();
        let output: Vec<_> = interface
            .endpoints
            .iter()
            .copied()
            .filter(|e| e.kind() == 2 && !e.is_in())
            .collect();
        if input.len() != 1 || output.len() != 1 {
            return Err("USB vendor Ethernet requires unique bulk IN/OUT endpoints");
        }
        // rtl_check_vendor_ok fixes the controller's RX/TX pipe numbers.
        // Carrier is polled, so the unused interrupt endpoint is not required.
        if matches!(kind, Kind::Realtek(_))
            && (input[0].address != 0x81 || output[0].address != 0x02)
        {
            return Err("RTL815x requires RX endpoint1 and TX endpoint2");
        }
        Ok(Self {
            input: input[0],
            output: output[0],
            kind,
            setting: (interface.number, interface.alternate),
        })
    }

    pub(super) fn setting(&self) -> (u8, u8) {
        self.setting
    }

    pub(super) fn observed_vendor(&self) -> bool {
        matches!(self.kind, Kind::Realtek(_))
    }
    pub(super) fn asix_candidate(&self) -> bool {
        matches!(self.kind, Kind::Asix(_))
    }

    pub(super) fn endpoints(&self) -> Vec<Endpoint> {
        vec![self.input, self.output]
    }
}

pub(super) struct HostControl {
    pub(super) controller: Shared,
    pub(super) id: SlotId,
}

impl Control for HostControl {
    fn control(&mut self, request: [u8; 8], output: &[u8]) -> Result<Vec<u8>, BlockError> {
        let mut controller = self.controller.borrow_mut();
        let mut c = controller.device(self.id)?;
        if !c.connected()? {
            return Err(BlockError::Unsupported("USB Ethernet disconnected"));
        }
        c.control(request, output)
    }
    fn delay_us(&mut self, microseconds: u64) {
        self.controller.borrow().clock.delay(microseconds);
    }
    fn minimum_wait(&mut self, microseconds: u64) -> Result<(), BlockError> {
        let mut controller = self.controller.borrow_mut();
        controller.device(self.id)?.connected()?;
        controller.minimum_wait(microseconds)
    }
}

enum Backend {
    Realtek(Chip<HostControl>),
    Asix(asix::Device<HostControl>),
}
impl Backend {
    fn receive_bytes(&self) -> usize {
        match self {
            Self::Realtek(chip) => chip.receive_bytes(),
            Self::Asix(chip) => chip.receive_bytes(),
        }
    }
    fn carrier(&mut self) -> Result<bool, BlockError> {
        match self {
            Self::Realtek(chip) => chip.carrier(),
            Self::Asix(chip) => chip.carrier(),
        }
    }
    fn link_down(&mut self) -> Result<(), BlockError> {
        match self {
            Self::Realtek(c) => c.stop(),
            Self::Asix(c) => c.link_down(),
        }
    }
    fn stop(&mut self) -> Result<(), BlockError> {
        match self {
            Self::Realtek(chip) => chip.stop(),
            Self::Asix(chip) => chip.stop(),
        }
    }
    fn encode(&self, frame: &[u8], output: &mut [u8], packet: u16) -> Result<usize, &'static str> {
        match self {
            Self::Realtek(_) => framing::encode(frame, output),
            Self::Asix(chip) => chip.encode(frame, output, packet),
        }
    }
    fn decode(&mut self, bytes: &[u8]) -> Result<Vec<Vec<u8>>, &'static str> {
        match self {
            Self::Realtek(_) => framing::decode(bytes),
            Self::Asix(chip) => chip.decode(bytes),
        }
    }
}

/// One receive TD and one synchronous transmit TD; DMA allocations persist
/// for the boot even after failed cancellation, as required by xHCI ownership.
pub struct UsbEthernet {
    controller: Shared,
    id: SlotId,
    binding: Binding,
    chip: Backend,
    mac: Mac,
    receive: DmaBuffer,
    send: DmaBuffer,
    queued: Option<TransferTicket>,
    pending: VecDeque<Vec<u8>>,
    carrier: bool,
    closed: bool,
    stopped: bool,
    pub firmware: Option<firmware::Receipt>,
    pub location: String,
}

impl UsbEthernet {
    /// The addressed device generation that owns this link.
    pub fn slot_id(&self) -> SlotId {
        self.id
    }

    pub fn driver(&self) -> &'static str {
        match self.chip {
            Backend::Realtek(_) => "usb-rtl815x",
            Backend::Asix(_) => "usb-asix",
        }
    }

    pub(super) fn new(
        controller: Shared,
        device: &Device,
        binding: Binding,
        dma: &mut dyn DmaAllocator,
        provider: &dyn crate::firmware::Provider,
    ) -> Result<Self, Error> {
        controller.borrow_mut().device(device.id)?;
        if binding.setting.1 != 0 {
            controller.borrow_mut().device(device.id)?.control(
                super::descriptor::setup(
                    1,
                    super::descriptor::SET_INTERFACE,
                    u16::from(binding.setting.1),
                    u16::from(binding.setting.0),
                    0,
                ),
                &[],
            )?;
        }
        let io = HostControl {
            controller: controller.clone(),
            id: device.id,
        };
        let (chip, mac, firmware, family) = match binding.kind {
            Kind::Realtek(revision) => {
                let firmware = firmware::load(provider, &device.descriptor, revision)?;
                let (chip, mac) = Chip::attach_with_policy(
                    io,
                    revision,
                    device.speed,
                    firmware.as_ref().map(|(_, parsed)| parsed),
                    policy(device),
                )?;
                let family = if revision.is_8153a() {
                    "RTL8153A"
                } else if revision.is_8153b() {
                    "RTL8153B"
                } else if revision.is_8153c() {
                    "RTL8153C"
                } else if revision.is_8156a() {
                    "RTL8156A"
                } else if revision.0 == 0x7420 {
                    "RTL815x tcr-7420 (gigabit)"
                } else if revision.is_8156b() {
                    "RTL8156B"
                } else {
                    "RTL8152"
                };
                let receipt = firmware.map(|(image, _)| firmware::Receipt::from_image(image));
                (Backend::Realtek(chip), mac, receipt, String::from(family))
            }
            Kind::Asix(identity) => {
                let (chip, mac) = asix::Device::attach(io, identity)?;
                let family = match identity {
                    asix::Identity::Ax88179 => "AX88179",
                    asix::Identity::Ax88178a => "AX88178A",
                    asix::Identity::Ax88772 { .. } => "AX88772/A/B",
                };
                let register = if matches!(identity, asix::Identity::Ax88772 { .. }) {
                    "smsr"
                } else {
                    "general"
                };
                let family = format!(
                    "{family} {register}-{:02x} phy-{:08x}",
                    chip.identity().0,
                    chip.identity().1
                );
                (Backend::Asix(chip), mac, None, family)
            }
        };
        let receive_bytes = chip.receive_bytes();
        Ok(Self {
            controller,
            id: device.id,
            binding,
            chip,
            mac,
            receive: dma.allocate(receive_bytes, receive_bytes.next_power_of_two())?,
            send: dma.allocate(2048, 2048)?,
            queued: None,
            pending: VecDeque::new(),
            carrier: false,
            closed: false,
            stopped: false,
            firmware,
            location: format!("{} {family}", device.location),
        })
    }

    fn open(&mut self) -> Result<(), BlockError> {
        if self.closed {
            return Err(BlockError::Unsupported("USB Ethernet closed"));
        }
        if !self.controller.borrow_mut().device(self.id)?.connected()? {
            self.closed = true;
            return Err(BlockError::Unsupported("USB Ethernet disconnected"));
        }
        Ok(())
    }

    fn quiesce(&mut self) -> Result<(), BlockError> {
        let mut controller = self.controller.borrow_mut();
        let mut c = controller.device(self.id)?;
        c.cancel(self.binding.input.dci())?;
        self.queued = None;
        c.cancel(self.binding.output.dci())?;
        self.pending.clear();
        Ok(())
    }

    fn update_carrier(&mut self) -> Result<bool, BlockError> {
        self.open()?;
        let linked = self.chip.carrier()?;
        if self.carrier
            && !linked
            && let Err(error) = self.quiesce().and_then(|()| self.chip.link_down())
        {
            self.closed = true;
            self.carrier = false;
            return Err(error);
        }
        self.carrier = linked;
        Ok(linked)
    }

    /// Stop requires acknowledged cancellation of both bulk endpoints before
    /// the device reset; failed stops remain closed and never reuse DMA.
    pub fn stop(&mut self) -> Result<(), BlockError> {
        self.controller.borrow_mut().device(self.id)?;
        if self.stopped {
            return Ok(());
        }
        self.closed = true;
        self.carrier = false;
        self.quiesce()?;
        self.chip.stop()?;
        self.stopped = true;
        Ok(())
    }
}

impl Drop for UsbEthernet {
    fn drop(&mut self) {
        // Failure cannot release boot-lifetime DMA or claim device quiescence.
        let _ = self.stop();
    }
}

fn link_error(error: BlockError) -> LinkError {
    LinkError::Adapter(format!("USB Ethernet: {error}"))
}

impl Link for UsbEthernet {
    fn mac(&self) -> Mac {
        self.mac
    }
    fn carrier(&mut self) -> bool {
        self.update_carrier().unwrap_or(false)
    }
    fn transmit(&mut self, frame: &[u8]) -> Result<(), LinkError> {
        if !self.update_carrier().map_err(link_error)? {
            return Err(LinkError::Adapter("USB Ethernet carrier down".into()));
        }
        let mut controller = self.controller.borrow_mut();
        let mut c = controller.device(self.id).map_err(link_error)?;
        c.endpoint_ready(self.binding.output.dci())
            .map_err(link_error)?;
        let bytes = self
            .chip
            .encode(frame, self.send.bytes_mut(), self.binding.output.max_packet)
            .map_err(|why| LinkError::Adapter(why.into()))?;
        fence(Ordering::Release);
        let actual = match c.transfer(
            self.binding.output.dci(),
            self.send.physical(),
            bytes,
            1_000_000,
        ) {
            Ok(actual) => actual,
            Err(error) => {
                // A STALL completes the request but the halted hardware can
                // still reference its old TD until Reset/Set Dequeue succeeds.
                c.clear(&self.binding.output).map_err(link_error)?;
                return Err(link_error(error));
            }
        };
        if actual != bytes {
            return Err(LinkError::Adapter("USB Ethernet short TX transfer".into()));
        }
        // The vendor protocol carries its own lengths; Linux submits no ZLP.
        Ok(())
    }
    fn receive(&mut self) -> Result<Option<Vec<u8>>, LinkError> {
        if !self.update_carrier().map_err(link_error)? {
            return Ok(None);
        }
        if let Some(frame) = self.pending.pop_front() {
            return Ok(Some(frame));
        }
        let mut controller = self.controller.borrow_mut();
        let mut c = controller.device(self.id).map_err(link_error)?;
        let endpoint = self.binding.input;
        if self.queued.is_none() {
            let token = c
                .submit_ticket(endpoint.dci(), self.receive.physical(), self.receive.len())
                .map_err(link_error)?;
            self.queued = Some(token);
        }
        let Some(result) = c.completed_ticket(self.queued.as_ref().unwrap()) else {
            return Ok(None);
        };
        self.queued = None;
        let actual = match result {
            Ok(bytes) => bytes,
            Err(error) => {
                c.clear(&endpoint).map_err(link_error)?;
                return Err(link_error(error));
            }
        };
        if actual > self.receive.len() {
            return Err(LinkError::Adapter(
                "USB Ethernet RX completion exceeds DMA".into(),
            ));
        }
        fence(Ordering::Acquire);
        self.pending.extend(
            self.chip
                .decode(&self.receive.bytes()[..actual])
                .map_err(|why| LinkError::Adapter(why.into()))?,
        );
        Ok(self.pending.pop_front())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    pub(crate) mod generation;
    use super::*;

    #[test]
    #[ignore = "oracle: STELA_RTL815X_IDS_ORACLE"]
    fn unchanged_linux_rtl815x_exact_usb_candidates_and_flags() {
        extern crate std;
        use sha2::{Digest, Sha256};
        let ids =
            std::fs::read_to_string(std::env::var("STELA_RTL815X_IDS_ORACLE").unwrap()).unwrap();
        assert_eq!(
            format!("{:x}", Sha256::digest(ids.as_bytes())),
            "e722f13b67f558590bbe36bfa80ca12174007f5f8a2a661a2f0d84bfd8b3aab1"
        );
        let expected: Vec<_> = ids
            .lines()
            .map(|line| {
                let (vid, pid) = line.split_once(' ').unwrap();
                (
                    u16::from_str_radix(vid, 16).unwrap(),
                    u16::from_str_radix(pid, 16).unwrap(),
                )
            })
            .collect();
        assert_eq!(USB_IDS, expected);
        for (vid, pid) in expected {
            let mut descriptor = [0; 18];
            descriptor[8..10].copy_from_slice(&vid.to_le_bytes());
            descriptor[10..12].copy_from_slice(&pid.to_le_bytes());
            assert!(matches(&descriptor));
            descriptor[8..10].copy_from_slice(&(vid ^ 0x8000).to_le_bytes());
            assert!(!matches(&descriptor));
            descriptor[8..10].copy_from_slice(&vid.to_le_bytes());
            descriptor[10..12].copy_from_slice(&0xffffu16.to_le_bytes());
            assert!(!matches(&descriptor));
        }
    }
}
