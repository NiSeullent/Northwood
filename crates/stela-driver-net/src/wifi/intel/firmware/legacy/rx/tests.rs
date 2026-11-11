// SPDX-License-Identifier: GPL-2.0-or-later
use super::*;
use crate::wifi::intel::dma::tests::{Domain, Source, read_back};
use core::sync::atomic::AtomicUsize;

mod notifications;

const ADDRESS_LIMIT: u64 = (1 << 36) - 1;

pub(super) struct Model {
    pub(super) queue: Queue,
    pub(super) domain: Arc<Domain>,
    layout: Layout,
    released: Arc<AtomicUsize>,
}

impl Model {
    pub(super) fn new(generation: u64) -> Self {
        let source = Source::default();
        let domain = Arc::new(Domain::default());
        domain.records.lock().unwrap().coherent = true;
        let mut queue = Queue::prepare(&source, domain.clone(), ADDRESS_LIMIT, generation).unwrap();
        // SAFETY: this fixture owns a fresh synchronous requester and the
        // sole queue for this modeled generation. No physical device exists.
        let layout = unsafe { queue.expose() }.unwrap();
        Self {
            queue,
            domain,
            layout,
            released: source.released,
        }
    }

    pub(super) fn write_buffer(&mut self, vid: u16, bytes: &[u8]) {
        let (address, posted_vid) = self.free_descriptor(vid - 1);
        assert_eq!(posted_vid, vid);
        self.write_dma(address, 0, bytes, true);
    }

    pub(super) fn used(&mut self, slot: u16, word: u32) {
        self.write_dma(
            self.layout.used,
            usize::from(slot) * 4,
            &word.to_le_bytes(),
            false,
        );
    }

    pub(super) fn producer(&mut self, closed: u32) {
        self.write_dma(self.layout.status, 0, &closed.to_le_bytes(), false);
    }

    pub(super) fn stop(&mut self) -> Result<(), DmaError> {
        // SAFETY: all test DMA and IRQ work is synchronous, and this call
        // ends the exclusive requester's modeled access to the queue.
        unsafe { self.queue.release_after_stop() }
    }

    fn free_descriptor(&self, slot: u16) -> (u64, u16) {
        let bytes = read_back(&self.queue.free, usize::from(slot) * 8, 8);
        let word = u64::from_le_bytes(bytes.try_into().unwrap());
        (word & !0xfff, (word & 0xfff) as u16)
    }

    fn write_dma(&self, iova: u64, offset: usize, bytes: &[u8], complete: bool) {
        // The test Domain hands out ordinary IOVAs at 0x100000 + id*0x10000.
        let id = (iova - 0x100000) / 0x10000;
        assert_eq!(iova, 0x100000 + id * 0x10000);
        let mut records = self.domain.records.lock().unwrap();
        assert!(matches!(
            records.permissions.get(&id),
            Some(Direction::FromDevice | Direction::Bidirectional)
        ));
        let pointer = records.live[&id].address(offset, bytes.len()).unwrap();
        // SAFETY: the locked fixture mapping is live and device writable.
        // The source bytes are independent and the test performs no concurrent DMA.
        unsafe {
            core::ptr::copy_nonoverlapping(bytes.as_ptr(), pointer.as_ptr(), bytes.len());
        }
        if complete {
            assert!(records.completed.insert(id));
        }
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        if self.queue.state != State::Stopped {
            let _ = self.stop();
        }
    }
}

#[test]
fn posted_layout_and_closed_index_gate_reads_and_payload_ownership() {
    let mut model = Model::new(41);
    assert_eq!(model.layout.write, 504);
    assert_eq!(model.layout.free, model.queue.free.region().address);
    assert_eq!(
        model.free_descriptor(0),
        (model.queue.buffers[0].chunk.region().address, 1)
    );
    assert_eq!(
        model.free_descriptor(503),
        (model.queue.buffers[503].chunk.region().address, 504),
    );
    assert_eq!(model.free_descriptor(504), (0, 0));
    assert_eq!(model.free_descriptor(511), (0, 0));
    {
        let mut records = model.domain.records.lock().unwrap();
        assert_eq!(records.live.len(), 507);
        assert_eq!(
            records.directions[504..],
            [
                Direction::ToDevice,
                Direction::Bidirectional,
                Direction::Bidirectional,
            ]
        );
        assert_eq!(records.coherent_reads.len(), 2);
        records.coherent_reads.clear();
    }

    // Device-written used data and a completed payload do not give the CPU
    // ownership until the coherent producer advances beyond that slot.
    model.write_buffer(2, b"first");
    model.used(0, 0xdead_b002);
    assert!(model.queue.complete().unwrap().is_none());
    model.producer(0xabcd_0000); // Closed frame number advances independently.
    assert!(model.queue.complete().unwrap().is_none());
    {
        let records = model.domain.records.lock().unwrap();
        assert_eq!(records.cpu_syncs, 0);
        assert_eq!(records.coherent_reads, [(507, 0), (507, 0)]);
    }

    model.write_buffer(1, b"second");
    model.used(1, 1);
    model.producer(0xabcd_0201); // A0 producer wrap quirk; low nine bits give slot 1.
    let completed = model.queue.complete().unwrap().unwrap();
    assert_eq!((completed.generation, completed.index), (41, 0));
    assert_eq!(&completed.bytes[..5], b"first");
    assert!(model.queue.complete().unwrap().is_none());
    assert_eq!(model.domain.records.lock().unwrap().cpu_syncs, 1);

    model.producer(2);
    let completed = model.queue.complete().unwrap().unwrap();
    assert_eq!((completed.generation, completed.index), (41, 1));
    assert_eq!(&completed.bytes[..6], b"second");
    assert!(model.queue.complete().unwrap().is_none());
    assert_eq!(model.domain.records.lock().unwrap().cpu_syncs, 2);
}

#[test]
fn out_of_pool_and_duplicate_vids_fault_without_an_extra_cpu_sync() {
    for vid in [0, POSTED + 1, 0xfff] {
        let mut model = Model::new(8);
        model.used(0, u32::from(vid));
        model.producer(1);
        assert!(matches!(model.queue.complete(), Err(Error::CompletionId(id)) if id == vid));
        assert!(matches!(model.queue.complete(), Err(Error::State)));
        assert_eq!(model.domain.records.lock().unwrap().cpu_syncs, 0);
    }

    let mut model = Model::new(9);
    model.write_buffer(1, b"owned");
    model.used(0, 1);
    model.used(1, 1);
    model.producer(2);
    assert!(model.queue.complete().unwrap().is_some());
    assert!(matches!(
        model.queue.complete(),
        Err(Error::CompletionId(1))
    ));
    assert_eq!(model.domain.records.lock().unwrap().cpu_syncs, 1);
    assert!(matches!(model.queue.complete(), Err(Error::State)));
}

#[test]
fn impossible_producer_and_backward_progress_never_consume_a_payload() {
    let mut model = Model::new(1);
    model.write_buffer(1, b"not a receipt");
    model.used(0, 1);
    model.producer(u32::from(POSTED + 1));
    assert!(
        matches!(model.queue.complete(), Err(Error::Producer { previous: 0, observed }) if observed == POSTED + 1)
    );
    assert_eq!(model.domain.records.lock().unwrap().cpu_syncs, 0);
    assert!(matches!(model.queue.complete(), Err(Error::State)));

    let mut model = Model::new(2);
    model.write_buffer(1, b"first");
    model.used(0, 1);
    model.producer(2);
    assert!(model.queue.complete().unwrap().is_some());
    model.producer(1);
    assert!(matches!(
        model.queue.complete(),
        Err(Error::Producer {
            previous: 2,
            observed: 1
        })
    ));
    assert_eq!(model.domain.records.lock().unwrap().cpu_syncs, 1);
    assert!(matches!(model.queue.complete(), Err(Error::State)));
}

#[test]
fn cpu_visibility_failure_faults_queue_and_stop_retries_invalidation() {
    let mut model = Model::new(5);
    let all_mappings = model.domain.records.lock().unwrap().live.len();
    model.write_buffer(1, b"device payload");
    model.used(0, 1);
    model.producer(1);
    model.domain.records.lock().unwrap().cpu_sync_fail = Some(0);
    assert!(matches!(
        model.queue.complete(),
        Err(Error::Dma(dma::Error::Dma(DmaError::VisibilityFailed)))
    ));
    assert!(matches!(model.queue.complete(), Err(Error::State)));
    assert_eq!(model.queue.read, 0);
    assert_eq!(model.released.load(Ordering::Relaxed), 0);

    model.domain.records.lock().unwrap().unmap_fail = Some(1);
    assert_eq!(model.stop(), Err(DmaError::InvalidationFailed));
    assert_eq!(
        model.domain.records.lock().unwrap().live.len(),
        all_mappings - 1
    );
    assert_eq!(model.released.load(Ordering::Relaxed), 0);
    assert!(matches!(model.queue.complete(), Err(Error::State)));
    model.stop().unwrap();
    {
        let records = model.domain.records.lock().unwrap();
        assert_eq!(records.unmaps, all_mappings + 1);
        assert!(records.live.is_empty());
    }
    let released = model.released.clone();
    drop(model);
    assert_eq!(released.load(Ordering::Relaxed), all_mappings);
}

#[test]
fn backend_without_coherent_observation_cannot_prepare_the_queue() {
    let source = Source::default();
    let domain = Arc::new(Domain::default());
    assert!(matches!(
        Queue::prepare(&source, domain.clone(), ADDRESS_LIMIT, 1),
        Err(Error::Dma(dma::Error::Dma(DmaError::CoherenceUnavailable)))
    ));
    let records = domain.records.lock().unwrap();
    assert!(records.live.is_empty());
    assert_eq!(source.released.load(Ordering::Relaxed), records.maps);
}
