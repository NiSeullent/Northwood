// SPDX-License-Identifier: GPL-2.0-or-later
//! Ownership of guest pages, resident frames, and swap slots for demand paging.
//!
//! The runtime serializes access to this table and performs I/O outside its
//! lock. It must quiesce all users of a victim frame and invalidate cached
//! translations before claiming it for eviction. A claimed frame can be
//! inspected until eviction completes. Finish any request only when its I/O
//! has completed or been cancelled and drained, including late DMA access.

use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};

#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Page(pub u32);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Frame(pub u32);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Slot(pub u32);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    Geometry,
    Allocation,
    InvalidPage,
    NotResident,
    Pinned,
    PinOverflow,
    SwapFull,
    SequenceOverflow,
    StaleRequest,
    AlreadyPrepared,
    InvalidOutcome,
    Io,
    Checksum,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Location {
    Zero,
    Resident(Frame),
    Swapped,
    InTransit,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Access {
    Read,
    Write,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PageInSource {
    Zero,
    Swap { slot: Slot, checksum: u32 },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PageInOutcome {
    FilledZero,
    Read { checksum: u32 },
    Failed,
}

#[derive(Debug)]
#[must_use]
pub enum PageIn {
    Resident(Frame),
    InTransit,
    NeedsFrame,
    Load(PageInRequest),
}

#[derive(Debug)]
#[must_use = "finish the request to publish or release its frame"]
pub struct PageInRequest {
    page: Page,
    frame: Frame,
    source: PageInSource,
    owner: u64,
    sequence: u64,
}

impl PageInRequest {
    pub fn page(&self) -> Page {
        self.page
    }

    pub fn frame(&self) -> Frame {
        self.frame
    }

    pub fn source(&self) -> PageInSource {
        self.source
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EvictionAction {
    DiscardZero,
    ReuseSwap,
    Write { slot: Slot },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EvictionOutcome {
    Discarded,
    Written { checksum: u32 },
    Failed,
}

#[derive(Debug)]
#[must_use = "finish the request to commit or roll back eviction"]
pub struct EvictionRequest {
    page: Page,
    frame: Frame,
    owner: u64,
    sequence: u64,
}

impl EvictionRequest {
    pub fn page(&self) -> Page {
        self.page
    }

    pub fn frame(&self) -> Frame {
        self.frame
    }
}

#[derive(Debug)]
#[must_use = "unpin the frame when direct access is finished"]
pub struct Pin {
    page: Page,
    frame: Frame,
    owner: u64,
}

impl Pin {
    pub fn frame(&self) -> Frame {
        self.frame
    }
}

#[derive(Clone, Copy)]
struct Stored {
    slot: Slot,
    checksum: u32,
}

#[derive(Clone, Copy)]
struct Resident {
    frame: Frame,
    stored: Option<Stored>,
    pins: u32,
    table: bool,
    dirty: bool,
    referenced: bool,
}

impl Resident {
    fn is_pinned(self) -> bool {
        self.table || self.pins != 0
    }
}

#[derive(Clone, Copy)]
enum PageState {
    Zero,
    Swapped(Stored),
    Resident(Resident),
    Loading {
        frame: Frame,
        stored: Option<Stored>,
        sequence: u64,
    },
    Evicting {
        resident: Resident,
        action: Option<EvictionAction>,
        sequence: u64,
    },
}

impl PageState {
    fn location(self) -> Location {
        match self {
            Self::Zero => Location::Zero,
            Self::Swapped(_) => Location::Swapped,
            Self::Resident(resident) => Location::Resident(resident.frame),
            Self::Loading { .. } | Self::Evicting { .. } => Location::InTransit,
        }
    }
}

fn next_owner() -> Result<u64, Error> {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.try_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
        .map_err(|_| Error::SequenceOverflow)
}

fn reserve<T>(count: usize) -> Result<Vec<T>, Error> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| Error::Allocation)?;
    Ok(values)
}

struct Slots {
    used: Vec<u64>,
    hint: usize,
}

impl Slots {
    fn new(count: usize) -> Result<Self, Error> {
        let words = count.div_ceil(64);
        let mut used = reserve(words)?;
        used.resize(words, 0);
        let tail = count % 64;
        if tail != 0 {
            // Mark the padding of the last word occupied.
            used[words - 1] = u64::MAX << tail;
        }
        Ok(Self { used, hint: 0 })
    }

    fn take(&mut self) -> Option<Slot> {
        let words = self.used.len();
        for step in 0..words {
            let word = (self.hint + step) % words;
            let available = !self.used[word];
            if available != 0 {
                let bit = available.trailing_zeros();
                self.used[word] |= 1 << bit;
                self.hint = word;
                return Some(Slot((word * 64 + bit as usize) as u32));
            }
        }
        None
    }

    fn release(&mut self, slot: Slot) {
        let index = slot.0 as usize;
        let word = index / 64;
        let bit = 1 << (index % 64);
        debug_assert_ne!(self.used[word] & bit, 0);
        self.used[word] &= !bit;
        self.hint = word;
    }
}

pub struct Pager {
    pages: Vec<PageState>,
    frames: Vec<Option<Page>>,
    free: Vec<Frame>,
    slots: Slots,
    clock: usize,
    owner: u64,
    sequence: u64,
}

impl Pager {
    /// Reserves all metadata before any pages or I/O requests are active.
    pub fn new(page_count: u32, frame_count: u32, slot_count: u32) -> Result<Self, Error> {
        if frame_count == 0 || page_count < frame_count {
            return Err(Error::Geometry);
        }

        let mut pages = reserve(page_count as usize)?;
        pages.resize(page_count as usize, PageState::Zero);
        let mut frames = reserve(frame_count as usize)?;
        frames.resize(frame_count as usize, None);
        let mut free = reserve(frame_count as usize)?;
        free.extend((0..frame_count).rev().map(Frame));
        Ok(Self {
            pages,
            frames,
            free,
            slots: Slots::new(slot_count as usize)?,
            clock: 0,
            owner: next_owner()?,
            sequence: 0,
        })
    }

    fn next_sequence(&mut self) -> Result<u64, Error> {
        // A delayed completion must not complete a later request for the same page.
        self.sequence = self
            .sequence
            .checked_add(1)
            .ok_or(Error::SequenceOverflow)?;
        Ok(self.sequence)
    }

    pub fn free_frames(&self) -> usize {
        self.free.len()
    }

    fn state(&mut self, page: Page) -> Result<&mut PageState, Error> {
        self.pages
            .get_mut(page.0 as usize)
            .ok_or(Error::InvalidPage)
    }

    pub fn locate(&self, page: Page) -> Result<Location, Error> {
        self.pages
            .get(page.0 as usize)
            .map(|state| state.location())
            .ok_or(Error::InvalidPage)
    }

    /// Call when exposing a frame to the guest or to a direct host access.
    /// Writable mappings conservatively dirty the page when installed.
    pub fn access(&mut self, page: Page, access: Access) -> Result<Location, Error> {
        let state = self.state(page)?;
        if let PageState::Resident(resident) = state {
            resident.referenced = true;
            resident.dirty |= access == Access::Write;
        }
        Ok(state.location())
    }

    /// Direct access to a raw host pointer must keep this pin until finished.
    pub fn pin(&mut self, page: Page, access: Access) -> Result<Pin, Error> {
        let owner = self.owner;
        let PageState::Resident(resident) = self.state(page)? else {
            return Err(Error::NotResident);
        };
        resident.pins = resident.pins.checked_add(1).ok_or(Error::PinOverflow)?;
        resident.referenced = true;
        resident.dirty |= access == Access::Write;
        Ok(Pin {
            page,
            frame: resident.frame,
            owner,
        })
    }

    pub fn unpin(&mut self, pin: Pin) -> Result<(), Error> {
        if pin.owner != self.owner {
            return Err(Error::StaleRequest);
        }
        let PageState::Resident(resident) = self.state(pin.page)? else {
            return Err(Error::StaleRequest);
        };
        if resident.frame != pin.frame || resident.pins == 0 {
            return Err(Error::StaleRequest);
        }
        resident.pins -= 1;
        Ok(())
    }

    /// P1 keeps guest page-table frames resident for the lifetime of the boot.
    pub fn mark_table(&mut self, page: Page) -> Result<(), Error> {
        let PageState::Resident(resident) = self.state(page)? else {
            return Err(Error::NotResident);
        };
        resident.table = true;
        Ok(())
    }

    /// Reserves a frame for a fill or read. Another request for the same
    /// page sees `InTransit` until the original request completes.
    pub fn begin_page_in(&mut self, page: Page) -> Result<PageIn, Error> {
        let state = *self.state(page)?;
        let stored = match state {
            PageState::Zero => None,
            PageState::Swapped(stored) => Some(stored),
            PageState::Resident(resident) => return Ok(PageIn::Resident(resident.frame)),
            PageState::Loading { .. } | PageState::Evicting { .. } => return Ok(PageIn::InTransit),
        };
        let Some(&frame) = self.free.last() else {
            return Ok(PageIn::NeedsFrame);
        };
        let sequence = self.next_sequence()?;
        self.free.pop();
        self.frames[frame.0 as usize] = Some(page);
        self.pages[page.0 as usize] = PageState::Loading {
            frame,
            stored,
            sequence,
        };
        let source = stored.map_or(PageInSource::Zero, |stored| PageInSource::Swap {
            slot: stored.slot,
            checksum: stored.checksum,
        });
        Ok(PageIn::Load(PageInRequest {
            page,
            frame,
            source,
            owner: self.owner,
            sequence,
        }))
    }

    /// A failed fill or read leaves the prior zero state or swap copy intact.
    /// Report failure only once the I/O and any DMA into the frame have drained.
    pub fn finish_page_in(
        &mut self,
        request: &PageInRequest,
        outcome: PageInOutcome,
    ) -> Result<Frame, Error> {
        if request.owner != self.owner {
            return Err(Error::StaleRequest);
        }
        let PageState::Loading {
            frame,
            stored,
            sequence,
        } = *self.state(request.page)?
        else {
            return Err(Error::StaleRequest);
        };
        if request.sequence != sequence {
            return Err(Error::StaleRequest);
        }

        let result = match (stored, outcome) {
            (_, PageInOutcome::Failed) => Err(Error::Io),
            (None, PageInOutcome::FilledZero) => Ok(()),
            (Some(stored), PageInOutcome::Read { checksum }) if checksum == stored.checksum => {
                Ok(())
            }
            (Some(_), PageInOutcome::Read { .. }) => Err(Error::Checksum),
            _ => Err(Error::InvalidOutcome),
        };
        if let Err(error) = result {
            self.pages[request.page.0 as usize] =
                stored.map_or(PageState::Zero, PageState::Swapped);
            self.release_frame(frame);
            return Err(error);
        }

        self.pages[request.page.0 as usize] = PageState::Resident(Resident {
            frame,
            stored,
            pins: 0,
            table: false,
            dirty: false,
            referenced: true,
        });
        Ok(frame)
    }

    fn release_frame(&mut self, frame: Frame) {
        self.frames[frame.0 as usize] = None;
        self.free.push(frame);
    }

    /// Second-chance clock over physical frames. Advances the hand even if
    /// the caller cannot evict this victim, for example when swap is full.
    /// The returned page is only a candidate until `begin_eviction`.
    pub fn next_victim(&mut self) -> Option<Page> {
        let frame_count = self.frames.len();
        for _ in 0..2 {
            for _ in 0..frame_count {
                let owner = self.frames[self.clock];
                self.clock = (self.clock + 1) % frame_count;
                let Some(page) = owner else {
                    continue;
                };
                let PageState::Resident(resident) = &mut self.pages[page.0 as usize] else {
                    continue;
                };
                if resident.is_pinned() {
                    continue;
                }
                if resident.referenced {
                    resident.referenced = false;
                    continue;
                }
                return Some(page);
            }
        }
        None
    }

    /// Claim the page's current frame after stopping its users and invalidating
    /// its mappings. Use the returned request to identify the frame to inspect.
    /// Accesses through this table now see
    /// `InTransit`, and the runtime must keep prior raw users quiescent.
    pub fn begin_eviction(&mut self, page: Page) -> Result<EvictionRequest, Error> {
        let PageState::Resident(resident) = *self.state(page)? else {
            return Err(Error::NotResident);
        };
        if resident.is_pinned() {
            return Err(Error::Pinned);
        }

        let sequence = self.next_sequence()?;
        self.pages[page.0 as usize] = PageState::Evicting {
            resident,
            action: None,
            sequence,
        };
        Ok(EvictionRequest {
            page,
            frame: resident.frame,
            owner: self.owner,
            sequence,
        })
    }

    /// Decide the I/O for a claimed frame. Set `content_is_zero` only if an
    /// inspection of this request's entire frame, after it was claimed, found
    /// zeros. `SwapFull` restores the resident state and retires the request.
    pub fn prepare_eviction(
        &mut self,
        request: &EvictionRequest,
        content_is_zero: bool,
    ) -> Result<EvictionAction, Error> {
        let (resident, action) = self.eviction_state(request)?;
        if action.is_some() {
            return Err(Error::AlreadyPrepared);
        }

        let action = match (content_is_zero, resident.dirty, resident.stored) {
            (true, _, _) | (_, false, None) => EvictionAction::DiscardZero,
            (_, false, Some(_)) => EvictionAction::ReuseSwap,
            (_, true, Some(stored)) => EvictionAction::Write { slot: stored.slot },
            (_, true, None) => {
                let Some(slot) = self.slots.take() else {
                    self.pages[request.page.0 as usize] = PageState::Resident(resident);
                    return Err(Error::SwapFull);
                };
                EvictionAction::Write { slot }
            }
        };
        self.pages[request.page.0 as usize] = PageState::Evicting {
            resident,
            action: Some(action),
            sequence: request.sequence,
        };
        Ok(action)
    }

    fn eviction_state(
        &mut self,
        request: &EvictionRequest,
    ) -> Result<(Resident, Option<EvictionAction>), Error> {
        if request.owner != self.owner {
            return Err(Error::StaleRequest);
        }
        let PageState::Evicting {
            resident,
            action,
            sequence,
        } = *self.state(request.page)?
        else {
            return Err(Error::StaleRequest);
        };
        if request.sequence != sequence {
            return Err(Error::StaleRequest);
        }
        Ok((resident, action))
    }

    /// A successful eviction requires a prepared action. Use `Failed` to
    /// release an unprepared claim, or after a failed I/O has fully drained.
    /// The resident frame remains authoritative after failure, even if a
    /// partial write damaged an older swap copy.
    pub fn finish_eviction(
        &mut self,
        request: &EvictionRequest,
        outcome: EvictionOutcome,
    ) -> Result<(), Error> {
        let (resident, action) = self.eviction_state(request)?;
        let next = match (action, outcome) {
            (_, EvictionOutcome::Failed) => Err(Error::Io),
            (Some(EvictionAction::DiscardZero), EvictionOutcome::Discarded) => {
                if let Some(stored) = resident.stored {
                    self.slots.release(stored.slot);
                }
                Ok(PageState::Zero)
            }
            (Some(EvictionAction::ReuseSwap), EvictionOutcome::Discarded) => resident
                .stored
                .map(PageState::Swapped)
                .ok_or(Error::InvalidOutcome),
            (Some(EvictionAction::Write { slot }), EvictionOutcome::Written { checksum }) => {
                Ok(PageState::Swapped(Stored { slot, checksum }))
            }
            _ => Err(Error::InvalidOutcome),
        };
        match next {
            Ok(state) => {
                self.pages[request.page.0 as usize] = state;
                self.release_frame(resident.frame);
                Ok(())
            }
            Err(error) => {
                if let Some(EvictionAction::Write { slot }) = action
                    && resident.stored.is_none()
                {
                    self.slots.release(slot);
                }
                self.pages[request.page.0 as usize] = PageState::Resident(resident);
                Err(error)
            }
        }
    }
}
