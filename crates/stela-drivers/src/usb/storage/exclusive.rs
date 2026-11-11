// SPDX-License-Identifier: GPL-2.0-or-later
//! Reserve the class transport of one idle BOT disk before publication.
//!
//! The controller still owns slot, topology and DMA lifetimes. A reservation
//! holds the transport while those lifetimes are qualified separately.

use super::{
    Disk, UsbDisk, UsbStorage, UsbTransport,
    bot::{Bot, Native},
    media::TransportOwner,
};
use crate::{
    block::{BlockError, Identity},
    usb::{
        descriptor::{Endpoint, interfaces},
        xhci::Device,
    },
};
use alloc::rc::Rc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReservationError {
    UnsupportedTransport,
    DifferentOwner,
    Aliased,
    Busy,
    UnsupportedMedia,
    Stale,
}

/// Class-level custody of one BOT transport. The reservation has no block,
/// SCSI or rescan interface. Actual I/O needs controller ownership for the
/// same attachment, including recovery and DMA retirement.
#[must_use]
pub struct UsbDiskReservation {
    transport: Option<Bot<Native>>,
    identity: Identity,
    retirement_uncertain: bool,
}

/// An exact attachment that completed whole-slot retirement. This keeps the
/// original transport owner and grants no DMA reuse or block I/O.
#[must_use]
pub struct RetiredUsbDisk {
    original: UsbDiskReservation,
}

impl RetiredUsbDisk {
    pub fn identity(&self) -> &Identity {
        self.original.identity()
    }
}

#[derive(Debug)]
pub struct UsbRetirementError {
    pub operation: &'static str,
    pub cause: BlockError,
    /// A retirement attempt passed preflight and could have touched hardware.
    pub ownership_uncertain: bool,
}

impl UsbRetirementError {
    pub(in crate::usb) fn preflight(operation: &'static str, cause: BlockError) -> Self {
        Self {
            operation,
            cause,
            ownership_uncertain: false,
        }
    }
}

/// Only the consumed class owner can construct this attachment proof.
pub(in crate::usb) struct ReservedBotAttachment<'a> {
    device: &'a Device,
    endpoints: &'a [Endpoint; 2],
}

impl<'a> ReservedBotAttachment<'a> {
    pub(in crate::usb) fn parts(self) -> (&'a Device, &'a [Endpoint; 2]) {
        (self.device, self.endpoints)
    }
}

impl UsbDiskReservation {
    pub fn identity(&self) -> &Identity {
        &self.identity
    }

    /// Stop and disable this original xHCI slot. Preflight refusal issues no
    /// command. If a command fails after fencing, the original slot and BOT
    /// buffers stay held. The returned reservation cannot resume I/O.
    #[expect(
        clippy::result_large_err,
        reason = "keep the original reservation and failure together"
    )]
    pub fn retire_xhci(mut self) -> Result<RetiredUsbDisk, (Self, UsbRetirementError)> {
        match self.retire_slot() {
            Ok(()) => Ok(RetiredUsbDisk { original: self }),
            Err(error) => {
                self.retirement_uncertain |= error.ownership_uncertain;
                Err((self, error))
            }
        }
    }

    #[expect(
        clippy::result_large_err,
        reason = "preserve the exact retirement failure"
    )]
    fn retire_slot(&self) -> Result<(), UsbRetirementError> {
        const PREFLIGHT: &str = "preflight reserved BOT interface retirement";
        if self.retirement_uncertain {
            return Err(UsbRetirementError {
                operation: PREFLIGHT,
                cause: BlockError::Unsupported("previous BOT retirement is unresolved"),
                ownership_uncertain: true,
            });
        }
        let transport = self.transport.as_ref().expect("kept BOT transport");
        let device = transport.device();
        let settings = interfaces(&device.configuration);
        let [setting] = settings.as_slice() else {
            return Err(UsbRetirementError::preflight(
                PREFLIGHT,
                BlockError::Unsupported("BOT device has other interface settings"),
            ));
        };
        if device.id != transport.io.id
            || setting.number != transport.interface()
            || setting.alternate != 0
            || !super::bulk_only(setting)
            || setting.endpoints.len() != transport.io.endpoints.len()
            || !transport
                .io
                .endpoints
                .iter()
                .all(|endpoint| setting.endpoints.contains(endpoint))
        {
            return Err(UsbRetirementError::preflight(
                PREFLIGHT,
                BlockError::Unsupported("BOT interface or endpoint ownership changed"),
            ));
        }
        let mut controller = transport.io.controller.try_borrow_mut().map_err(|_| {
            UsbRetirementError::preflight(
                PREFLIGHT,
                BlockError::Unsupported("xHCI controller is borrowed"),
            )
        })?;
        controller.retire_reserved_bot(ReservedBotAttachment {
            device,
            endpoints: &transport.io.endpoints,
        })
    }
}

impl Drop for UsbDiskReservation {
    fn drop(&mut self) {
        if self.retirement_uncertain {
            // A failed Stop or Disable cannot return original DMA backing.
            core::mem::forget(self.transport.take());
        }
    }
}

impl UsbStorage {
    /// Consume the unpublished interface and its sole fixed writable LUN.
    /// Refusal returns both original handles. Admission only observes the
    /// existing software state, without submitting or polling commands.
    #[expect(
        clippy::result_large_err,
        reason = "return both owners without allocation on refusal"
    )]
    pub fn try_reserve_disk(
        self,
        disk: UsbDisk,
    ) -> Result<UsbDiskReservation, (Self, UsbDisk, ReservationError)> {
        if let Err(error) = self.check_disk_reservation(&disk) {
            return Err((self, disk, error));
        }
        let UsbTransport::Bot(Disk { bot, identity, .. }) = disk.disk else {
            unreachable!("admission checked the transport");
        };
        drop(self);
        // The two exact strong references were checked above. These Rc
        // owners are confined to the current thread and no callback ran.
        let owner = Rc::into_inner(bot)
            .expect("admission excluded other class owners")
            .into_inner();
        Ok(UsbDiskReservation {
            transport: Some(owner.transport),
            identity,
            retirement_uncertain: false,
        })
    }

    fn check_disk_reservation(&self, disk: &UsbDisk) -> Result<(), ReservationError> {
        let (TransportOwner::Bot(owner), UsbTransport::Bot(disk)) = (&self.owner, &disk.disk)
        else {
            return Err(ReservationError::UnsupportedTransport);
        };
        if !Rc::ptr_eq(owner, &disk.bot) {
            return Err(ReservationError::DifferentOwner);
        }
        if Rc::strong_count(owner) != 2 || Rc::weak_count(owner) != 0 {
            return Err(ReservationError::Aliased);
        }
        let owner = owner.try_borrow().map_err(|_| ReservationError::Busy)?;
        let bot = &owner.transport;
        if bot.max_lun() != 0 || disk.lun != 0 {
            return Err(ReservationError::UnsupportedMedia);
        }
        if bot.command_pending() {
            return Err(ReservationError::Busy);
        }
        if bot.poisoned() {
            return Err(ReservationError::Stale);
        }
        owner.check_reservation(disk)?;
        let controller = bot
            .io
            .controller
            .try_borrow()
            .map_err(|_| ReservationError::Busy)?;
        controller
            .validate_device_id(bot.io.id)
            .map_err(|_| ReservationError::Stale)
    }
}
