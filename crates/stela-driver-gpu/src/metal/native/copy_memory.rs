// SPDX-License-Identifier: GPL-2.0-or-later
//! Private GPU backing and a canonical copy lease for each guest wire.

use crate::metal::{
    error::{Error, Result, ensure, require},
    vulkan::backend::GuestRows,
};
use alloc::{boxed::Box, collections::BTreeMap};
use ash::vk;
use core::num::NonZeroU64;
use stela_contracts::gate::{MemoryRange, window::IMPORT_TOKEN_WRITABLE};

/// Canonical access to the pages a guest wire keeps live. A failed
/// publication may have changed a prefix; callers must not retry it.
pub trait GuestCopyLease: Send + Sync {
    fn read_into(&self, offset: u64, output: &mut [u8]) -> Result<()>;
    fn publish_from(&self, offset: u64, source: &[u8]) -> Result<()>;
}

/// Admits ordered guest physical ranges without lending an alias address
/// to the Vulkan driver.
pub trait GuestCopyProvider: Send + Sync {
    /// # Safety
    /// A trusted guest gate supplied these ranges from a bidirectionally
    /// wired descriptor. It remains wired until the backend retires the
    /// corresponding token. The provider must independently validate
    /// placement and permissions through its canonical RAM owner.
    unsafe fn lease(&self, ranges: &[MemoryRange]) -> Result<Box<dyn GuestCopyLease>>;
}

pub(super) struct Import {
    pub(super) lease: Box<dyn GuestCopyLease>,
    pub(super) token: NonZeroU64,
    pub(super) bytes: u64,
}

impl Import {
    pub(super) fn check_rows(&self, rows: GuestRows) -> Result<usize> {
        let last = u64::from(rows.count.saturating_sub(1));
        let end = last
            .checked_mul(rows.stride)
            .and_then(|at| at.checked_add(rows.offset))
            .and_then(|at| at.checked_add(rows.bytes));
        ensure(
            rows.count != 0
                && rows.bytes != 0
                && rows.bytes <= rows.stride
                && end.is_some_and(|end| end <= self.bytes),
            "copied guest buffer row span",
        )?;
        usize::try_from(rows.bytes).map_err(|_| Error::Invalid("copied guest buffer row length"))
    }
}

pub(super) struct Imports {
    provider: Box<dyn GuestCopyProvider>,
    pub(super) owned: BTreeMap<u64, Import>,
    pub(super) checkpoint: Option<(vk::Semaphore, u64)>,
}

impl Imports {
    #[cfg(any(test, feature = "metal-linked"))]
    pub(super) fn new(provider: Box<dyn GuestCopyProvider>) -> Self {
        Self {
            provider,
            owned: BTreeMap::new(),
            checkpoint: None,
        }
    }

    /// # Safety
    /// The gate's token authorizes this live wire, as required by
    /// [`GuestCopyProvider::lease`].
    pub(super) unsafe fn acquire(
        &self,
        ranges: &[MemoryRange],
        token: NonZeroU64,
    ) -> Result<Import> {
        let bits = token.get();
        require(
            bits & IMPORT_TOKEN_WRITABLE != 0 && bits & !IMPORT_TOKEN_WRITABLE != 0,
            "copied guest memory requires a bidirectional guest wire",
        )?;
        let bytes = ranges.iter().try_fold(0u64, |total, range| {
            ensure(
                range.bytes != 0 && range.address.checked_add(range.bytes).is_some(),
                "copied guest range",
            )?;
            total
                .checked_add(range.bytes)
                .ok_or(Error::Invalid("copied guest memory length"))
        })?;
        ensure(bytes != 0, "empty copied guest memory")?;
        // SAFETY: this is the trusted tagged wire admitted by the caller.
        let lease = unsafe { self.provider.lease(ranges) }?;
        Ok(Import {
            lease,
            token,
            bytes,
        })
    }
}
