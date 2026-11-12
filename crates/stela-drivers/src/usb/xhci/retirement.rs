// SPDX-License-Identifier: GPL-2.0-only
//! Whole-slot retirement before Disable Slot (xHCI 4.6.4).
//!
//! Linux xhci_free_dev receives a device whose traffic was already stopped by
//! usb_disconnect. Native enumeration failures must establish that same fact;
//! an unpublished descriptor inventory does not mean EP0 stopped running.
use super::*;
use reset::Ownership;

type Failure = (&'static str, BlockError);
const ENDPOINT_DISABLED: u32 = 0;

/// Minted only after the whole-slot retirement boundary. Its fields are
/// private to this module, so a software status or slot number cannot return DMA.
pub(super) struct Unpublished {
    slot: u8,
    state: Slot,
}

impl Unpublished {
    pub(super) fn into_slot(self) -> (u8, Slot) {
        (self.slot, self.state)
    }
}

/// Retired root-leaf owner, never reconstructed from a current ledger.
/// The original Slot stays in Controller until physical reset and final health.
struct RetiredRoot {
    id: SlotId,
    attachment: Attachment,
}

/// Complete published storage returned only by RetiredRoot::reset.
pub(super) struct Published {
    slot: u8,
    state: Slot,
}
impl Published {
    pub(super) fn into_slot(self) -> (u8, Slot) {
        (self.slot, self.state)
    }
}

/// The mutable controller borrow is held from the last acknowledged
/// endpoint retirement through Disable Slot. No caller can publish another TD,
/// run private recovery control, or substitute a different device generation
/// between those operations. This permission is private and cannot be copied.
struct Quiesced<'a> {
    controller: &'a mut Controller,
    slot: u8,
    ownership: Ownership,
}

impl Controller {
    /// End this reserved BOT attachment without resetting or re-enumerating
    /// its port. The class caller retains its transport and payload buffers.
    #[expect(
        clippy::result_large_err,
        reason = "retain the operation and the exact controller error"
    )]
    pub(in crate::usb) fn retire_reserved_bot(
        &mut self,
        attachment: crate::usb::storage::ReservedBotAttachment<'_>,
    ) -> Result<(), crate::usb::storage::UsbRetirementError> {
        use crate::usb::storage::UsbRetirementError;
        const PREFLIGHT: &str = "preflight reserved BOT slot retirement";
        let refused = |error| UsbRetirementError::preflight(PREFLIGHT, error);
        let (device, endpoints) = attachment.parts();
        self.hardware.ready().map_err(refused)?;
        self.reset_preflight(device, endpoints).map_err(refused)?;
        let slot = device.id.hardware_slot();
        let state = self.slots[usize::from(slot)].as_ref().unwrap();
        if state.attachment.parent.is_some()
            || state.consumer != identity::Consumer::Native
            || state.endpoints[1].is_none()
            || state.retired_rings.iter().any(Option::is_some)
        {
            return Err(refused(BlockError::Unsupported(
                "BOT retirement requires one original native root leaf",
            )));
        }
        if self.error.is_some()
            || self.command_poisoned
            || self.command_ring_stopped
            || self.outstanding.is_some()
            || self.queued_resets.contains(&device.id)
            || self.posted_for(device.id)
            || state.endpoints.iter().flatten().any(|endpoint| {
                !endpoint.enabled
                    || endpoint.poisoned
                    || endpoint.in_flight != 0
                    || !endpoint.ring.normal.idle()
                    || endpoint.streams.is_some()
                    || endpoint.iso.is_some()
            })
        {
            return Err(refused(BlockError::Unsupported(
                "BOT slot or controller still has pending work",
            )));
        }
        // This same mutable borrow fences aliases before the first Stop and
        // stays held through the acknowledged Disable and DCBAA clear.
        let uncertain = |(operation, cause)| UsbRetirementError {
            operation,
            cause,
            ownership_uncertain: true,
        };
        self.disable_slot(slot, Ownership::Retired)
            .map_err(uncertain)?;
        self.dcbaa
            .sync_dirty()
            .map_err(|cause| uncertain(("publish retired BOT DCBAA clear", cause)))
    }

    /// Exclusive root-leaf reset is the only published storage return producer.
    /// Its private witness keeps the original generation/current/attachment
    /// through whole-slot retirement and postsettle physical reset.
    #[expect(
        clippy::result_large_err,
        reason = "The shared error preserves exact source and cleanup causes without allocating on propagation."
    )]
    pub(super) fn retire_reset_root(&mut self, id: SlotId) -> Result<Published, Failure> {
        let slot = self
            .validate_id(id)
            .map_err(|e| ("validate reusable generation", e))?;
        let state = self.slots[usize::from(slot)].as_ref().unwrap();
        let attachment = state.attachment;
        if attachment.parent.is_some()
            || state.parent_id.is_some()
            || attachment.tier != 0
            || attachment.route != 0
            || attachment.translator.is_some()
            || !self.retry_resources.can_offer(slot, state)
        {
            return Err((
                "preflight reusable root storage",
                BlockError::Unsupported(
                    "USB reusable reset requires one complete root-leaf resource pool",
                ),
            ));
        }
        self.disable_slot(slot, Ownership::Retired)?;
        RetiredRoot { id, attachment }.reset(self)
    }

    /// Retire every endpoint and its ordered events, then acknowledge
    /// Disable before clearing DCBAA. Generic Disable keeps published owners
    /// permanently; only unpublished EP0-only owners enter the retry cache.
    /// The separate RetiredRoot witness additionally requires physical reset
    /// and final health before Published can return complete root-leaf storage.
    #[expect(
        clippy::result_large_err,
        reason = "The shared error preserves exact source and cleanup causes without allocating on propagation."
    )]
    pub(super) fn disable_slot(&mut self, slot: u8, ownership: Ownership) -> Result<(), Failure> {
        Quiesced::acquire(self, slot, ownership)?.disable()
    }
}

impl RetiredRoot {
    #[expect(
        clippy::result_large_err,
        reason = "The shared error preserves exact source and cleanup causes without allocating on propagation."
    )]
    fn reset(self, controller: &mut Controller) -> Result<Published, Failure> {
        let slot = self.id.hardware_slot();
        let index = usize::from(slot);
        // The typed witness binds the Disable to these original owners;
        // the current grant is still present before the physical reset.
        if !controller.known_id(self.id)
            || controller.slot_ownership[index] != Ownership::Retired
            || controller.dcbaa.read::<u64>(index * 8) != 0
            || controller
                .current_port_owner(None, self.attachment.root_port)
                .map_err(|e| ("retain reusable physical owner", e))?
                != Some(self.id)
        {
            return Err((
                "retain reusable physical owner",
                BlockError::Unsupported("USB reusable reset lost the original retired attachment"),
            ));
        }
        controller
            .reset_root_device(self.attachment)
            .map_err(|e| ("acknowledge reusable root reset", e))?;
        // reset_root_device includes its settling delay. Recheck health
        // and link after that delay, before erasing a single retired DMA byte.
        controller
            .ready()
            .map_err(|e| ("post-reset reusable controller health", e))?;
        let at = controller.port_offset(self.attachment.root_port);
        let status = controller.registers.read32(at);
        if status & (CCS | PED | PP) != CCS | PED | PP
            || status & (PR | PORT_CHANGES | (1 << 3)) != 0
            || (status >> PORT_SPEED_SHIFT & PORT_SPEED_FIELD) as u8 != self.attachment.speed
            || status & PLS != 0
            || controller
                .power
                .port_owner(None, self.attachment.root_port)
                .is_some()
            || controller
                .pending
                .iter()
                .any(|e| e.kind() == TRANSFER_EVENT && e.slot() == slot)
        {
            return Err((
                "verify reusable root reset settlement",
                BlockError::Unsupported("USB reusable reset lacks a stable original physical link"),
            ));
        }
        let state = controller.slots[index].take().unwrap();
        controller.slot_ownership[index] = Ownership::Vacant;
        Ok(Published { slot, state })
    }
}

impl<'a> Quiesced<'a> {
    #[expect(
        clippy::result_large_err,
        reason = "The shared error preserves exact source and cleanup causes without allocating on propagation."
    )]
    fn acquire(
        controller: &'a mut Controller,
        slot: u8,
        ownership: Ownership,
    ) -> Result<Self, Failure> {
        let index = usize::from(slot);
        if slot == 0
            || index >= controller.slots.len()
            || controller.slot_ownership.get(index) != Some(&Ownership::Enabled)
            || !matches!(ownership, Ownership::Vacant | Ownership::Retired)
        {
            return Err((
                "preflight Disable Slot ownership",
                BlockError::Unsupported("invalid xHCI slot retirement owner"),
            ));
        }
        let state = controller.slots.get_mut(index).and_then(Option::as_mut);
        let unaddressed = if let Some(state) = state {
            if ownership == Ownership::Vacant
                && (state.inventory.is_some()
                    || state.consumer != identity::Consumer::Unclaimed
                    || state
                        .endpoints
                        .iter()
                        .enumerate()
                        .any(|(dci, endpoint)| dci != 1 && endpoint.is_some())
                    || state.retired_rings.iter().any(Option::is_some)
                    || state.endpoints[1]
                        .as_ref()
                        .is_none_or(|ep0| ep0.streams.is_some() || ep0.iso.is_some()))
            {
                return Err((
                    "preflight unpublished Disable Slot",
                    BlockError::Unsupported("published xHCI slot cannot become vacant"),
                ));
            }
            // Address Device has not succeeded and no EP0 doorbell could have
            // been published while this initial generation fence was held.
            // Such a failed command may leave the zeroed EP0 context Disabled;
            // Stop/Set Dequeue would be illegal in that hardware state.
            let unaddressed = ownership == Ownership::Vacant
                && state.retiring
                && state.inventory.is_none()
                && state.consumer == identity::Consumer::Unclaimed
                && state.current_configuration == Some(0)
                && state.endpoints.iter().skip(2).all(Option::is_none)
                && state.endpoints[1].as_ref().is_some_and(|endpoint| {
                    endpoint.ring.index == 0
                        && endpoint.ring.cycle
                        && endpoint.in_flight == 0
                        && endpoint.streams.is_none()
                        && endpoint.iso.is_none()
                });
            state.retiring = true;
            unaddressed
        } else {
            // The Enable Slot grant may need returning when reservation failed
            // before any contexts were published. Never infer this from absence
            // of a software record if DCBAA still names a DMA allocation.
            if ownership != Ownership::Vacant || controller.dcbaa.read::<u64>(index * 8) != 0 {
                return Err((
                    "preflight unaddressed Disable Slot",
                    BlockError::Unsupported("xHCI slot allocation ownership unresolved"),
                ));
            }
            true
        };
        controller
            .ready()
            .map_err(|error| ("pre-Disable Slot controller status", error))?;
        let context = controller.context();
        let mut failure = None;
        for dci in 1..32usize {
            let Some(state) = controller.slots[index].as_ref() else {
                break;
            };
            if state.endpoints[dci].is_none() {
                continue;
            }
            fence(Ordering::Acquire);
            let endpoint_state = state
                ._output
                .coherent_u32(dci * context)
                .map_err(|error| ("observe actual endpoint DMA context", error))?
                & 7;
            if unaddressed && dci == 1 && endpoint_state == ENDPOINT_DISABLED {
                continue;
            }
            // The complete software endpoint set is visited even after one
            // contributor fails. No Disable Slot is published on partial success.
            if let Err(error) = controller.cancel_inner(slot, dci as u8) {
                failure.get_or_insert(error);
            }
        }
        if let Some(error) = failure {
            return Err(("retire every endpoint before Disable Slot", error));
        }
        controller
            .ready()
            .map_err(|error| ("pre-Disable Slot event retirement", error))?;
        // Recheck device contexts after every cancellation and its event
        // drain, rather than accepting a software queue count as an idle proof.
        if let Some(state) = controller.slots[index].as_ref() {
            fence(Ordering::Acquire);
            for (dci, endpoint) in state.endpoints.iter().enumerate().skip(1) {
                if let Some(endpoint) = endpoint {
                    let actual = state
                        ._output
                        .coherent_u32(dci * context)
                        .map_err(|error| ("observe actual endpoint DMA context", error))?
                        & 7;
                    if endpoint.poisoned
                        || ownership == Ownership::Vacant
                            && (endpoint.in_flight != 0 || !endpoint.ring.normal.idle())
                        || !(actual == ENDPOINT_STOPPED
                            || unaddressed && dci == 1 && actual == ENDPOINT_DISABLED)
                    {
                        return Err((
                            "verify stopped endpoint before Disable Slot",
                            BlockError::Unsupported("xHCI slot endpoint retirement unresolved"),
                        ));
                    }
                }
            }
        }
        Ok(Self {
            controller,
            slot,
            ownership,
        })
    }

    #[expect(
        clippy::result_large_err,
        reason = "The shared error preserves exact source and cleanup causes without allocating on propagation."
    )]
    fn disable(self) -> Result<(), Failure> {
        let Self {
            controller,
            slot,
            ownership,
        } = self;
        let completion = controller
            .command(0, 0, reset::DISABLE_SLOT << 10 | u32::from(slot) << 24)
            .map_err(|error| ("acknowledge Disable Slot", error))?;
        if completion.slot() != slot {
            controller.command_poisoned = true;
            return Err((
                "acknowledge Disable Slot identity",
                BlockError::Unsupported("Disable Slot completion named another slot"),
            ));
        }
        #[cfg(test)]
        {
            controller.test_retired_numbers[usize::from(slot)] = true;
        }
        controller
            .ready()
            .map_err(|error| ("post-Disable Slot controller status", error))?;
        controller
            .pending
            .retain(|event| !(event.kind() == TRANSFER_EVENT && event.slot() == slot));
        controller.slot_ownership[usize::from(slot)] = ownership;
        controller.dcbaa.write(usize::from(slot) * 8, 0u64);
        if ownership == Ownership::Vacant
            && let Some(state) = controller.slots[usize::from(slot)].take()
        {
            controller
                .retry_resources
                .retain(Unpublished { slot, state });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
