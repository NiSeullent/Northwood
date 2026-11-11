// SPDX-License-Identifier: GPL-2.0-or-later
use super::*;
use crate::wifi::intel::{
    dma::tests::{Domain, Source},
    firmware::legacy::nvm::Section,
};

#[derive(Default)]
struct Doorbell {
    posted: Option<u32>,
}

impl Csr for Doorbell {
    fn read32(&mut self, register: u32) -> Result<u32, AccessError> {
        assert_eq!(register, service::INT_MASK);
        Ok(0)
    }

    fn write32(&mut self, register: u32, value: u32) -> Result<(), AccessError> {
        assert_eq!(register, WRITE_POINTER);
        self.posted = Some(value);
        Err(AccessError::Unavailable)
    }

    fn delay_us(&mut self, _: u32) -> Result<(), AccessError> {
        unreachable!()
    }
}

#[test]
fn gen1_tfd_packs_the_high_iova_nibble_and_rejects_an_out_of_range_tb() {
    let tfd = command_tfd(
        Region {
            address: 0xf_1234_5678,
            bytes: 16,
        },
        None,
    )
    .unwrap();
    assert_eq!(
        &tfd[..10],
        &[0, 0, 0, 1, 0x78, 0x56, 0x34, 0x12, 0x0f, 0x01]
    );
    assert!(tfd[10..].iter().all(|&byte| byte == 0));
    assert_eq!(
        command_tfd(
            Region {
                address: FH_DMA_LIMIT - 15,
                bytes: 16,
            },
            None,
        ),
        Err(Error::Address)
    );
}

#[test]
fn uncertain_doorbell_preserves_the_same_pending_command_and_all_pins() {
    let source = Source::default();
    let domain = Arc::new(Domain::default());
    let mut queue = Queue::prepare(&source, domain.clone()).unwrap();
    // SAFETY: this synchronous fixture owns a fresh stopped queue and admits
    // only this queue's DMA and modeled doorbell.
    unsafe {
        assert_eq!(queue.expose_for_preparation().unwrap(), queue.layout());
        queue.admit_prepared().unwrap();
    }
    let request = Request::new(Section::SW, 0, 2048).unwrap();
    let mut io = Doorbell::default();
    assert!(matches!(
        queue.submit_nvm_read(&mut io, request),
        Err(Error::Service(_))
    ));
    assert_eq!(io.posted, Some(1));
    assert_eq!(queue.pending.as_ref().unwrap().ticket.sequence(), 0);
    assert_eq!(domain.records.lock().unwrap().live.len(), 5);
    assert_eq!(queue.submit_nvm_read(&mut io, request), Err(Error::State));
    // SAFETY: the fixture records publication without any asynchronous DMA.
    unsafe { queue.release_after_stop() }.unwrap();
    assert!(domain.records.lock().unwrap().live.is_empty());
}

#[test]
fn uncertain_paging_doorbell_keeps_the_command_tail_until_successful_invalidation() {
    let source = Source::default();
    let domain = Arc::new(Domain::default());
    let dma_domain: Arc<dyn DmaDomain> = domain.clone();
    let mut queue = Queue::prepare(&source, dma_domain.clone()).unwrap();
    // SAFETY: the synchronous model has a stopped requester and only this queue.
    unsafe {
        queue.expose_for_preparation().unwrap();
        queue.admit_prepared().unwrap();
    }
    let mut io = Doorbell::default();
    let payload = [0x5a; paging::PAYLOAD_BYTES];
    assert!(matches!(
        queue.submit_paging(&mut io, &source, &dma_domain, &payload),
        Err(Error::Service(_))
    ));
    assert_eq!(io.posted, Some(1));
    let remainder = queue.pending.as_ref().unwrap().remainder.as_ref().unwrap();
    assert_eq!(remainder.region().bytes, 132);
    {
        let mut records = domain.records.lock().unwrap();
        assert_eq!(records.live.len(), 6);
        assert_eq!(records.directions.last(), Some(&Direction::ToDevice));
        records.unmap_fail = Some(5);
    }
    // SAFETY: the model has no asynchronous DMA or pending workers.
    assert!(matches!(
        unsafe { queue.release_after_stop() },
        Err(Error::Dma(dma::Error::Dma(_)))
    ));
    assert_eq!(domain.records.lock().unwrap().live.len(), 1);
    assert!(queue.pending.as_ref().unwrap().remainder.is_some());
    // SAFETY: retry with the same requester still stopped.
    unsafe { queue.release_after_stop() }.unwrap();
    assert!(domain.records.lock().unwrap().live.is_empty());
}
