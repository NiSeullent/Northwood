// SPDX-License-Identifier: GPL-2.0-or-later
use super::*;

fn start(pager: &mut Pager, page: u32) -> PageInRequest {
    let PageIn::Load(request) = pager.begin_page_in(Page(page)).unwrap() else {
        panic!("page {page} did not start loading");
    };
    request
}

fn fill(pager: &mut Pager, page: u32) -> Frame {
    let request = start(pager, page);
    assert_eq!(request.source(), PageInSource::Zero);
    pager
        .finish_page_in(&request, PageInOutcome::FilledZero)
        .unwrap()
}

fn write_out(pager: &mut Pager, page: u32, slot: u32, checksum: u32) {
    pager.access(Page(page), Access::Write).unwrap();
    let request = pager.begin_eviction(Page(page)).unwrap();
    assert_eq!(
        pager.prepare_eviction(&request, false),
        Ok(EvictionAction::Write { slot: Slot(slot) })
    );
    pager
        .finish_eviction(&request, EvictionOutcome::Written { checksum })
        .unwrap();
}

#[test]
fn geometry_and_indexes_are_checked() {
    assert!(matches!(Pager::new(0, 0, 0), Err(Error::Geometry)));
    assert!(matches!(Pager::new(2, 3, 0), Err(Error::Geometry)));

    let mut pager = Pager::new(2, 1, 0).unwrap();
    assert_eq!(pager.locate(Page(2)), Err(Error::InvalidPage));
    assert_eq!(
        pager.access(Page(u32::MAX), Access::Write),
        Err(Error::InvalidPage)
    );
    assert!(matches!(
        pager.begin_page_in(Page(2)),
        Err(Error::InvalidPage)
    ));
    assert!(matches!(
        pager.begin_eviction(Page(2)),
        Err(Error::InvalidPage)
    ));
    assert!(matches!(
        pager.pin(Page(2), Access::Read),
        Err(Error::InvalidPage)
    ));
    assert_eq!(pager.mark_table(Page(2)), Err(Error::InvalidPage));
    assert_eq!(pager.free_frames(), 1);
}

#[test]
fn duplicate_loads_coalesce_and_failure_releases_the_frame() {
    let mut pager = Pager::new(2, 1, 0).unwrap();
    let request = start(&mut pager, 0);
    assert_eq!(request.page(), Page(0));
    assert_eq!(request.frame(), Frame(0));
    assert_eq!(pager.locate(Page(0)), Ok(Location::InTransit));
    assert!(matches!(
        pager.begin_page_in(Page(0)),
        Ok(PageIn::InTransit)
    ));
    assert!(matches!(
        pager.begin_page_in(Page(1)),
        Ok(PageIn::NeedsFrame)
    ));
    assert_eq!(pager.next_victim(), None);
    assert_eq!(
        pager.finish_page_in(&request, PageInOutcome::Failed),
        Err(Error::Io)
    );
    assert_eq!(pager.locate(Page(0)), Ok(Location::Zero));
    assert_eq!(pager.free_frames(), 1);

    let retry = start(&mut pager, 0);
    assert_eq!(
        pager.finish_page_in(&request, PageInOutcome::FilledZero),
        Err(Error::StaleRequest)
    );
    assert_eq!(
        pager.finish_page_in(&retry, PageInOutcome::FilledZero),
        Ok(Frame(0))
    );
    assert_eq!(pager.free_frames(), 0);
    assert!(matches!(
        pager.begin_page_in(Page(0)),
        Ok(PageIn::Resident(Frame(0)))
    ));
}

#[test]
fn wrong_or_corrupt_reads_keep_the_swap_copy() {
    let mut pager = Pager::new(2, 1, 1).unwrap();
    fill(&mut pager, 0);
    write_out(&mut pager, 0, 0, 0x1234);

    let request = start(&mut pager, 0);
    assert_eq!(
        request.source(),
        PageInSource::Swap {
            slot: Slot(0),
            checksum: 0x1234
        }
    );
    assert_eq!(
        pager.finish_page_in(&request, PageInOutcome::Read { checksum: 0x1235 }),
        Err(Error::Checksum),
    );
    assert_eq!(pager.locate(Page(0)), Ok(Location::Swapped));
    assert_eq!(pager.free_frames(), 1);

    let request = start(&mut pager, 0);
    assert_eq!(
        pager.finish_page_in(&request, PageInOutcome::FilledZero),
        Err(Error::InvalidOutcome)
    );
    let retry = start(&mut pager, 0);
    assert_eq!(
        retry.source(),
        PageInSource::Swap {
            slot: Slot(0),
            checksum: 0x1234
        }
    );
    assert_eq!(
        pager.finish_page_in(&retry, PageInOutcome::Read { checksum: 0x1234 }),
        Ok(Frame(0))
    );
}

#[test]
fn failed_eviction_holds_the_frame_until_completion_and_releases_the_new_slot() {
    let mut pager = Pager::new(2, 1, 1).unwrap();
    let frame = fill(&mut pager, 0);
    pager.access(Page(0), Access::Write).unwrap();
    let request = pager.begin_eviction(Page(0)).unwrap();
    assert_eq!(request.frame(), frame);
    assert_eq!(request.page(), Page(0));
    assert_eq!(
        pager.prepare_eviction(&request, false),
        Ok(EvictionAction::Write { slot: Slot(0) })
    );
    assert!(matches!(
        pager.begin_page_in(Page(0)),
        Ok(PageIn::InTransit)
    ));
    assert!(matches!(
        pager.begin_page_in(Page(1)),
        Ok(PageIn::NeedsFrame)
    ));
    assert_eq!(pager.next_victim(), None);
    assert_eq!(
        pager.finish_eviction(&request, EvictionOutcome::Failed),
        Err(Error::Io)
    );
    assert_eq!(pager.locate(Page(0)), Ok(Location::Resident(frame)));
    assert_eq!(pager.free_frames(), 0);
    let retry = pager.begin_eviction(Page(0)).unwrap();
    assert_eq!(
        pager.prepare_eviction(&retry, false),
        Ok(EvictionAction::Write { slot: Slot(0) })
    );
    assert_eq!(
        pager.finish_eviction(&request, EvictionOutcome::Written { checksum: 0 }),
        Err(Error::StaleRequest),
    );
    assert_eq!(
        pager.finish_eviction(&retry, EvictionOutcome::Discarded),
        Err(Error::InvalidOutcome)
    );
    write_out(&mut pager, 0, 0, 42);
    assert_eq!(pager.locate(Page(0)), Ok(Location::Swapped));
    assert_eq!(pager.free_frames(), 1);
}

#[test]
fn clean_swap_copy_is_reused_and_failed_dirty_overwrite_is_retried() {
    let mut pager = Pager::new(2, 1, 1).unwrap();
    fill(&mut pager, 0);
    write_out(&mut pager, 0, 0, 1);

    let request = start(&mut pager, 0);
    pager
        .finish_page_in(&request, PageInOutcome::Read { checksum: 1 })
        .unwrap();
    pager.access(Page(0), Access::Read).unwrap();
    let request = pager.begin_eviction(Page(0)).unwrap();
    assert_eq!(
        pager.prepare_eviction(&request, false),
        Ok(EvictionAction::ReuseSwap)
    );
    pager
        .finish_eviction(&request, EvictionOutcome::Discarded)
        .unwrap();

    let request = start(&mut pager, 0);
    pager
        .finish_page_in(&request, PageInOutcome::Read { checksum: 1 })
        .unwrap();
    pager.access(Page(0), Access::Write).unwrap();
    let request = pager.begin_eviction(Page(0)).unwrap();
    assert_eq!(
        pager.prepare_eviction(&request, false),
        Ok(EvictionAction::Write { slot: Slot(0) })
    );
    assert_eq!(
        pager.finish_eviction(&request, EvictionOutcome::Failed),
        Err(Error::Io)
    );
    let retry = pager.begin_eviction(Page(0)).unwrap();
    assert_eq!(
        pager.prepare_eviction(&retry, false),
        Ok(EvictionAction::Write { slot: Slot(0) })
    );
    pager
        .finish_eviction(&retry, EvictionOutcome::Written { checksum: 2 })
        .unwrap();
    assert_eq!(
        start(&mut pager, 0).source(),
        PageInSource::Swap {
            slot: Slot(0),
            checksum: 2
        }
    );
}

#[test]
fn zero_evictions_work_with_full_or_no_swap() {
    let mut pager = Pager::new(3, 2, 1).unwrap();
    fill(&mut pager, 0);
    fill(&mut pager, 1);
    write_out(&mut pager, 0, 0, 1);

    // A fresh unmodified frame needs no slot even though the store is full.
    let request = pager.begin_eviction(Page(1)).unwrap();
    assert_eq!(
        pager.prepare_eviction(&request, false),
        Ok(EvictionAction::DiscardZero)
    );
    pager
        .finish_eviction(&request, EvictionOutcome::Discarded)
        .unwrap();
    fill(&mut pager, 2);
    pager.access(Page(2), Access::Write).unwrap();
    let request = pager.begin_eviction(Page(2)).unwrap();
    assert_eq!(
        pager.prepare_eviction(&request, false),
        Err(Error::SwapFull)
    );
    assert_eq!(
        pager.prepare_eviction(&request, true),
        Err(Error::StaleRequest)
    );
    assert!(matches!(pager.locate(Page(2)), Ok(Location::Resident(_))));

    let request = start(&mut pager, 0);
    pager
        .finish_page_in(&request, PageInOutcome::Read { checksum: 1 })
        .unwrap();
    pager.access(Page(0), Access::Write).unwrap();
    let request = pager.begin_eviction(Page(0)).unwrap();
    assert_eq!(
        pager.prepare_eviction(&request, true),
        Ok(EvictionAction::DiscardZero)
    );
    pager
        .finish_eviction(&request, EvictionOutcome::Discarded)
        .unwrap();
    assert_eq!(pager.locate(Page(0)), Ok(Location::Zero));
    write_out(&mut pager, 2, 0, 2);

    let mut no_swap = Pager::new(1, 1, 0).unwrap();
    fill(&mut no_swap, 0);
    no_swap.access(Page(0), Access::Write).unwrap();
    let request = no_swap.begin_eviction(Page(0)).unwrap();
    assert_eq!(
        no_swap.prepare_eviction(&request, false),
        Err(Error::SwapFull)
    );
    let request = no_swap.begin_eviction(Page(0)).unwrap();
    assert_eq!(
        no_swap.prepare_eviction(&request, true),
        Ok(EvictionAction::DiscardZero)
    );
    no_swap
        .finish_eviction(&request, EvictionOutcome::Discarded)
        .unwrap();
    assert_eq!(no_swap.locate(Page(0)), Ok(Location::Zero));
}

#[test]
fn slot_bitmap_wraps_and_never_allocates_padding() {
    let mut slots = Slots::new(65).unwrap();
    for slot in 0..65 {
        assert_eq!(slots.take(), Some(Slot(slot)));
    }
    assert_eq!(slots.take(), None);
    slots.release(Slot(64));
    slots.release(Slot(0));
    assert_eq!(slots.take(), Some(Slot(0)));
    assert_eq!(slots.take(), Some(Slot(64)));
    assert_eq!(slots.take(), None);
}

#[test]
fn clock_gives_a_referenced_frame_a_second_chance() {
    let mut pager = Pager::new(4, 3, 0).unwrap();
    for page in 0..3 {
        assert_eq!(fill(&mut pager, page), Frame(page));
    }
    // The first scan clears all three and returns to the first frame.
    assert_eq!(pager.next_victim(), Some(Page(0)));
    pager.access(Page(1), Access::Read).unwrap();
    assert_eq!(pager.next_victim(), Some(Page(2)));
    assert_eq!(pager.next_victim(), Some(Page(0)));
    assert_eq!(pager.next_victim(), Some(Page(1)));
}

#[test]
fn nested_and_table_pins_exclude_victims() {
    let mut pager = Pager::new(2, 2, 1).unwrap();
    fill(&mut pager, 0);
    fill(&mut pager, 1);
    pager.mark_table(Page(0)).unwrap();
    let first = pager.pin(Page(1), Access::Read).unwrap();
    let second = pager.pin(Page(1), Access::Write).unwrap();
    assert_eq!(first.frame(), Frame(1));
    assert_eq!(pager.next_victim(), None);
    assert!(matches!(pager.begin_eviction(Page(0)), Err(Error::Pinned)));
    pager.unpin(first).unwrap();
    assert_eq!(pager.next_victim(), None);
    assert!(matches!(pager.begin_eviction(Page(1)), Err(Error::Pinned)));
    pager.unpin(second).unwrap();
    assert_eq!(pager.next_victim(), Some(Page(1)));
    let request = pager.begin_eviction(Page(1)).unwrap();
    assert_eq!(
        pager.prepare_eviction(&request, false),
        Ok(EvictionAction::Write { slot: Slot(0) })
    );
    assert!(matches!(
        pager.pin(Page(1), Access::Read),
        Err(Error::NotResident)
    ));
    assert_eq!(pager.mark_table(Page(1)), Err(Error::NotResident));
}

#[test]
fn completions_cannot_target_another_pager() {
    let mut a = Pager::new(1, 1, 0).unwrap();
    let mut b = Pager::new(1, 1, 0).unwrap();
    let request_a = start(&mut a, 0);
    let request_b = start(&mut b, 0);
    assert_eq!(
        b.finish_page_in(&request_a, PageInOutcome::FilledZero),
        Err(Error::StaleRequest)
    );
    assert_eq!(
        a.finish_page_in(&request_a, PageInOutcome::FilledZero),
        Ok(Frame(0))
    );
    assert_eq!(
        b.finish_page_in(&request_b, PageInOutcome::FilledZero),
        Ok(Frame(0))
    );
    let request_a = a.begin_eviction(Page(0)).unwrap();
    let request_b = b.begin_eviction(Page(0)).unwrap();
    a.prepare_eviction(&request_a, false).unwrap();
    b.prepare_eviction(&request_b, false).unwrap();
    assert_eq!(
        b.finish_eviction(&request_a, EvictionOutcome::Discarded),
        Err(Error::StaleRequest)
    );
    assert_eq!(
        a.finish_eviction(&request_a, EvictionOutcome::Discarded),
        Ok(())
    );
    assert_eq!(
        b.finish_eviction(&request_b, EvictionOutcome::Discarded),
        Ok(())
    );
}

#[test]
fn zero_inspection_belongs_to_one_eviction_claim_even_after_frame_reuse() {
    let mut pager = Pager::new(1, 1, 1).unwrap();
    let page = Page(0);
    let frame = fill(&mut pager, 0);
    let previous = pager.begin_eviction(page).unwrap();
    assert_eq!(previous.frame(), frame);
    assert_eq!(pager.access(page, Access::Write), Ok(Location::InTransit));
    assert_eq!(
        pager.prepare_eviction(&previous, true),
        Ok(EvictionAction::DiscardZero)
    );
    pager
        .finish_eviction(&previous, EvictionOutcome::Discarded)
        .unwrap();

    assert_eq!(fill(&mut pager, 0), frame);
    pager.access(page, Access::Write).unwrap();
    let current = pager.begin_eviction(page).unwrap();
    assert_eq!(current.frame(), frame);
    assert_eq!(
        pager.prepare_eviction(&previous, true),
        Err(Error::StaleRequest)
    );
    assert_eq!(
        pager.prepare_eviction(&current, false),
        Ok(EvictionAction::Write { slot: Slot(0) })
    );
    assert_eq!(
        pager.prepare_eviction(&current, true),
        Err(Error::AlreadyPrepared)
    );
    pager
        .finish_eviction(&current, EvictionOutcome::Written { checksum: 42 })
        .unwrap();
    assert_eq!(pager.locate(page), Ok(Location::Swapped));
}
