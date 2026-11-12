// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 Stela contributors
//! Independently authored register/DMA peer derived from Linux atlantic's
//! hardware contract. Tests are source-only until the coordinator runs them.
extern crate std;
use super::*;
use crate::bus::{
    dma::{
        mapping::{Direction, DomainMapping},
        memory::{CacheAttribute, CpuMapping, PageBacking, PageSource, PinnedPages},
        pages::DmaPages,
    },
    pci::Address,
};
use alloc::{
    alloc::{alloc_zeroed, dealloc},
    boxed::Box,
    collections::BTreeMap,
    rc::Rc,
};
use core::{
    alloc::Layout,
    cell::{Cell, RefCell, UnsafeCell},
    ops::Range,
    ptr::NonNull,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
};
use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    sync::Mutex,
};

struct Ram {
    pointer: NonNull<u8>,
    physical: Vec<u64>,
    layout: Layout,
    dropped: Arc<AtomicUsize>,
}
// SAFETY: test-only uniquely kept stable allocations; physical labels are
// symbolic and never passed to native hardware or any real remapping provider.
unsafe impl Send for Ram {}
// SAFETY: only raw access, synchronized by the synchronous test model.
unsafe impl Sync for Ram {}
// SAFETY: zeroed page-aligned model storage, immutable labels and kept owner.
unsafe impl PageBacking for Ram {
    fn cpu_base(&self) -> NonNull<u8> {
        self.pointer
    }
    fn physical_pages(&self) -> &[u64] {
        &self.physical
    }
    fn page_shift(&self) -> u8 {
        12
    }
    fn cache_attribute(&self) -> CacheAttribute {
        CacheAttribute::WriteBack
    }
}
impl Drop for Ram {
    fn drop(&mut self) {
        self.dropped.fetch_add(1, Ordering::Relaxed);
        // SAFETY: the pointer came from the matching allocation with this layout and is freed once.
        unsafe { dealloc(self.pointer.as_ptr(), self.layout) }
    }
}
struct Pages {
    dropped: Arc<AtomicUsize>,
    next: Cell<u64>,
}
impl PageSource for Pages {
    fn allocate(
        &self,
        bytes: usize,
        alignment: usize,
        attribute: CacheAttribute,
    ) -> core::result::Result<PinnedPages, MemoryError> {
        assert_eq!(attribute, CacheAttribute::WriteBack);
        let bytes = bytes.div_ceil(4096) * 4096;
        let layout = Layout::from_size_align(bytes, alignment).unwrap();
        // SAFETY: the layout has a nonzero size.
        let p = NonNull::new(unsafe { alloc_zeroed(layout) }).ok_or(MemoryError::OutOfMemory)?;
        let base = self.next.get();
        self.next.set(base + bytes as u64);
        PinnedPages::new(
            Box::new(Ram {
                pointer: p,
                physical: (0..bytes / 4096)
                    .map(|i| base + (i * 4096) as u64)
                    .collect(),
                layout,
                dropped: self.dropped.clone(),
            }),
            bytes,
            alignment,
        )
    }
}
#[derive(Default)]
struct Domain {
    maps: Mutex<BTreeMap<u64, CpuMapping>>,
    config: Mutex<Option<Arc<NativePm>>>,
    next: AtomicUsize,
    visibility: Mutex<Vec<(bool, u64, Range<usize>)>>,
    drain_count: AtomicUsize,
    engine_epoch: AtomicUsize,
    drained_epoch: AtomicUsize,
    engines: AtomicUsize,
    pending_dma: AtomicUsize,
    published: Mutex<BTreeMap<u64, u8>>,
    defer_writebacks: AtomicBool,
    posted: Mutex<Vec<(u64, Vec<u8>)>>,
    fail_drain: AtomicBool,
    fail_unmap: AtomicBool,
    unmap_calls: AtomicUsize,
    fail_unmap_at: AtomicUsize,
    fail_sync: AtomicBool,
    incoherent: AtomicBool,
}
impl Domain {
    fn require_bus_master_disabled(&self) {
        // Keep the very same backing installed beneath OwnedConfig. This is
        // a device-side observation of the native Command word, never a
        // caller-supplied quiescence flag or a second configuration writer.
        let config = self.config.lock().unwrap().as_ref().unwrap().clone();
        assert_eq!(
            config.command() & 4,
            0,
            "PCI bus mastering must be disabled before DMA drain or unmap"
        );
        let readback = {
            let accesses = config.accesses.lock().unwrap();
            let last_write = accesses.iter().rposition(|access| {
                matches!(
                    access,
                    ConfigAccess::Write8(4 | 5, _)
                        | ConfigAccess::Write16(4, _)
                        | ConfigAccess::Write32(4, _)
                )
            });
            last_write.is_some_and(|index| {
                let cleared = match accesses[index] {
                    ConfigAccess::Write16(4, value) => value & 4 == 0,
                    _ => false,
                };
                let acknowledged = accesses[index + 1..].iter().any(|access| match access {
                    ConfigAccess::Read16(4, value) => *value != u16::MAX && *value & 4 == 0,
                    _ => false,
                });
                cleared && acknowledged
            })
        };
        assert!(
            readback,
            "PCI bus-master clear requires caller readback before DMA drain or unmap"
        );
    }
    fn raw(&self, address: u64, bytes: usize) -> NonNull<u8> {
        let maps = self.maps.lock().unwrap();
        for (&base, mapping) in maps.iter() {
            if address >= base && address + bytes as u64 <= base + mapping.bytes() as u64 {
                return mapping.address((address - base) as usize, bytes).unwrap();
            }
        }
        panic!("model DMA outside an actual retained IOVA: {address:x}/{bytes}")
    }
    fn visible(&self, address: u64, bytes: usize) {
        let mut published = self.published.lock().unwrap();
        for n in 0..bytes {
            let at = address + n as u64;
            // SAFETY: the pointer addresses live memory this test's model owns, in bounds, and
            // nothing accesses it concurrently.
            let actual = unsafe { self.raw(at, 1).as_ptr().read_volatile() };
            assert_eq!(
                published.remove(&at),
                Some(actual),
                "DMA without fresh visibility at {at:x}"
            );
        }
    }
    fn read32(&self, address: u64) -> u32 {
        self.visible(address, 4);
        // SAFETY: the pointer addresses live memory this test's model owns, in bounds, and nothing
        // accesses it concurrently.
        unsafe { self.raw(address, 4).as_ptr().cast::<u32>().read_volatile() }
    }
    fn read64(&self, address: u64) -> u64 {
        self.visible(address, 8);
        // SAFETY: the pointer addresses live memory this test's model owns, in bounds, and nothing
        // accesses it concurrently.
        unsafe { self.raw(address, 8).as_ptr().cast::<u64>().read_volatile() }
    }
    fn writeback(&self, address: u64, bytes: &[u8]) {
        if self.defer_writebacks.load(Ordering::Relaxed) {
            self.posted.lock().unwrap().push((address, bytes.to_vec()));
        } else {
            self.store(address, bytes);
        }
    }
    fn store(&self, address: u64, bytes: &[u8]) {
        for (i, b) in bytes.iter().enumerate() {
            // SAFETY: the pointer addresses live memory this test's model owns, in bounds, and
            // nothing accesses it concurrently.
            unsafe { self.raw(address + i as u64, 1).as_ptr().write_volatile(*b) }
        }
    }
    fn complete_posted(&self) {
        for (address, bytes) in self.posted.lock().unwrap().drain(..) {
            self.store(address, &bytes);
        }
    }
    fn put32(&self, address: u64, value: u32) {
        self.writeback(address, &value.to_le_bytes());
    }
    fn packet(&self, address: u64, length: usize) -> Vec<u8> {
        self.visible(address, length);
        (0..length)
            // SAFETY: the pointer addresses live memory this test's model owns, in bounds, and
            // nothing accesses it concurrently.
            .map(|n| unsafe { self.raw(address + n as u64, 1).as_ptr().read_volatile() })
            .collect()
    }
    fn put_packet(&self, address: u64, bytes: &[u8]) {
        self.visible(address, memory::BUFFER);
        self.writeback(address, bytes);
    }
}
// SAFETY: explicit independent synchronous model, never native DMA. Each mapping
// keeps its exact pins, has a unique symbolic IOVA and checked cache accesses.
unsafe impl DmaDomain for Domain {
    unsafe fn map(
        &self,
        pages: &DmaPages,
        _: Direction,
        limit: u64,
    ) -> core::result::Result<DomainMapping, DmaError> {
        let id = self.next.fetch_add(1, Ordering::Relaxed) + 1;
        let address = 0x1_0000_0000 + (id as u64) * 0x10_0000;
        assert!(address + pages.bytes() as u64 - 1 <= limit);
        self.maps.lock().unwrap().insert(
            address,
            pages
                .as_single()
                .unwrap()
                .cpu_mapping(CacheAttribute::WriteBack)
                .unwrap(),
        );
        Ok(DomainMapping {
            id: id as u64,
            address,
            bytes: pages.bytes(),
        })
    }
    unsafe fn unmap(&self, m: DomainMapping) -> core::result::Result<(), DmaError> {
        let call = self.unmap_calls.fetch_add(1, Ordering::Relaxed) + 1;
        if self.fail_unmap.load(Ordering::Relaxed)
            || self.fail_unmap_at.load(Ordering::Relaxed) == call
        {
            return Err(DmaError::InvalidationFailed);
        }
        self.require_bus_master_disabled();
        assert!(self.drain_count.load(Ordering::Relaxed) > 0);
        assert_eq!(self.engines.load(Ordering::Relaxed), 0);
        assert_eq!(self.pending_dma.load(Ordering::Relaxed), 0);
        assert!(self.posted.lock().unwrap().is_empty());
        assert_eq!(
            self.drained_epoch.load(Ordering::Relaxed),
            self.engine_epoch.load(Ordering::Relaxed),
            "historical drain cannot release a newer engine epoch"
        );
        assert!(self.maps.lock().unwrap().remove(&m.address).is_some());
        Ok(())
    }
    unsafe fn sync_for_device(
        &self,
        m: DomainMapping,
        r: Range<usize>,
        _: Direction,
    ) -> core::result::Result<(), DmaError> {
        if self.fail_sync.load(Ordering::Relaxed) {
            return Err(DmaError::VisibilityFailed);
        }
        assert!(r.end <= m.bytes);
        let mut published = self.published.lock().unwrap();
        for n in r.clone() {
            let address = m.address + n as u64;
            // SAFETY: the pointer addresses live memory this test's model owns, in bounds, and
            // nothing accesses it concurrently.
            let byte = unsafe { self.raw(address, 1).as_ptr().read_volatile() };
            published.insert(address, byte);
        }
        self.visibility.lock().unwrap().push((false, m.address, r));
        Ok(())
    }
    unsafe fn sync_for_cpu(
        &self,
        m: DomainMapping,
        r: Range<usize>,
        _: Direction,
    ) -> core::result::Result<(), DmaError> {
        assert!(r.end <= m.bytes);
        self.visibility.lock().unwrap().push((true, m.address, r));
        Ok(())
    }
    unsafe fn read_coherent_u32(
        &self,
        m: DomainMapping,
        offset: usize,
        _: Direction,
    ) -> core::result::Result<u32, DmaError> {
        if self.incoherent.load(Ordering::Relaxed) {
            return Err(DmaError::CoherenceUnavailable);
        }
        // SAFETY: the mapping's range lies inside this test's live pages, and the model writes
        // nothing concurrently.
        Ok(unsafe {
            self.raw(m.address + offset as u64, 4)
                .as_ptr()
                .cast::<u32>()
                .read_volatile()
        })
    }
    unsafe fn drain_posted_writes(&self) -> core::result::Result<(), DmaError> {
        if self.fail_drain.load(Ordering::Relaxed) {
            return Err(DmaError::DrainUnavailable);
        }
        assert_eq!(
            self.engines.load(Ordering::Relaxed),
            0,
            "drain while engines can still issue DMA"
        );
        self.require_bus_master_disabled();
        // Pending device writebacks complete at the real modeled drain boundary.
        self.complete_posted();
        self.pending_dma.store(0, Ordering::Relaxed);
        self.drained_epoch
            .store(self.engine_epoch.load(Ordering::Relaxed), Ordering::Relaxed);
        self.drain_count.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum Fault {
    None,
    Reset,
    HostBoot,
    BootFailure,
    BootAllOnes,
    RamSemaphore,
    RamCompletion,
    FirmwareAck,
    Cache,
    QueueStop,
    Removed,
    Snapshot,
}
struct Peer {
    regs: RefCell<BTreeMap<usize, u32>>,
    ram: RefCell<BTreeMap<u32, u32>>,
    writes: RefCell<Vec<(usize, u32)>>,
    fault: Cell<Fault>,
    profile: Profile,
    elapsed: Cell<u64>,
    timing: Cell<bool>,
    waits: RefCell<Vec<u64>>,
    fail_wait: Cell<Option<usize>>,
    domain: Arc<Domain>,
    sent: RefCell<Vec<Vec<u8>>>,
    tx_head: Cell<usize>,
    rx_head: Cell<usize>,
    semaphores: RefCell<BTreeMap<usize, bool>>,
    second_lock_busy: Cell<bool>,
    fail_read: Cell<Option<usize>>,
    fail_write: Cell<Option<usize>>,
    failed_release_reached_device: Cell<bool>,
    failed_acquire_reached_device: Cell<bool>,
    reads: RefCell<Vec<usize>>,
    reset_count: Cell<usize>,
    ack_delay: Cell<u64>,
    pending_ack: RefCell<Vec<(u64, usize, u32)>>,
    tx_delay: Cell<bool>,
    pending_tx: Cell<Option<usize>>,
    native_pm: RefCell<Option<Arc<NativePm>>>,
}
impl Peer {
    fn new(profile: Profile, domain: Arc<Domain>) -> Rc<Self> {
        let peer = Rc::new(Self {
            regs: RefCell::new(BTreeMap::new()),
            ram: RefCell::new(BTreeMap::new()),
            writes: RefCell::new(Vec::new()),
            fault: Cell::new(Fault::None),
            profile,
            elapsed: Cell::new(0),
            timing: Cell::new(true),
            waits: RefCell::new(Vec::new()),
            fail_wait: Cell::new(None),
            domain,
            sent: RefCell::new(Vec::new()),
            tx_head: Cell::new(0),
            rx_head: Cell::new(0),
            semaphores: RefCell::new(BTreeMap::new()),
            second_lock_busy: Cell::new(false),
            fail_read: Cell::new(None),
            fail_write: Cell::new(None),
            failed_release_reached_device: Cell::new(false),
            failed_acquire_reached_device: Cell::new(false),
            reads: RefCell::new(Vec::new()),
            reset_count: Cell::new(0),
            ack_delay: Cell::new(200),
            pending_ack: RefCell::new(Vec::new()),
            tx_delay: Cell::new(false),
            pending_tx: Cell::new(None),
            native_pm: RefCell::new(None),
        });
        peer.set(0x1c, 0xa);
        peer.set(0x18, 0x0301005a);
        peer.set(0x388, 1);
        peer.set(0x364, 0x10000);
        peer.set(0x360, 0x20000);
        peer.set(0x334, 0x30000);
        peer.ram.borrow_mut().insert(0x100a0, 0x02112233);
        peer.ram.borrow_mut().insert(0x100a4, 0x44550000);
        peer.set(0x12008, 0x33221102);
        peer.set(0x1200c, 0x5544);
        peer.set(0x13000, 0x10001);
        peer.set(0x13004, 0x11_0201);
        peer.set(0x13010, 1);
        peer.set(0x13774, 0x1004);
        peer.set(0x13778, 0);
        peer.set(0x1377c, 0x08020000); // ART16..79
        peer.set(0x5a10, 1);
        peer
    }
    fn set(&self, offset: usize, value: u32) {
        self.regs.borrow_mut().insert(offset, value);
    }
    fn get(&self, offset: usize) -> u32 {
        self.regs.borrow().get(&offset).copied().unwrap_or(0)
    }
    fn address(&self, base: usize) -> u64 {
        u64::from(self.get(base)) | (u64::from(self.get(base + 4)) << 32)
    }
    fn acknowledge_later(&self, register: usize, value: u32) {
        self.pending_ack.borrow_mut().push((
            self.elapsed.get() + self.ack_delay.get(),
            register,
            value,
        ));
    }
    fn transmit(&self, tail: usize) {
        let base = self.address(0x7c00);
        assert_ne!(
            self.get(0x7c08) & 0x80000000,
            0,
            "TX before descriptor enable"
        );
        assert_eq!(
            self.get(0x7c08) & 0x1ff8,
            64,
            "ring size must be descriptor count, not byte count"
        );
        assert_ne!(self.domain.engines.load(Ordering::Relaxed) & 1, 0);
        let mut head = self.tx_head.get();
        while head != tail {
            let descriptor = base + (head * 16) as u64;
            let control = self.domain.read32(descriptor + 8);
            let length = ((control >> 4) & 0xffff) as usize;
            assert_eq!(control & 0xffff000f, 0x08200001);
            assert_eq!(self.domain.read32(descriptor + 12), (length as u32) << 14);
            self.sent
                .borrow_mut()
                .push(self.domain.packet(self.domain.read64(descriptor), length));
            self.domain.put32(descriptor + 8, control | (1 << 20));
            self.domain.pending_dma.fetch_add(1, Ordering::Relaxed);
            head = (head + 1) % 64;
        }
        self.tx_head.set(head);
        self.set(0x7c0c, head as u32);
    }
    fn receive(&self, bytes: &[u8], kind: u32, status: u16, length: u16) {
        assert_ne!(self.domain.engines.load(Ordering::Relaxed) & 2, 0);
        let index = self.rx_head.get();
        assert_ne!(
            index,
            self.get(0x5b10) as usize,
            "device ran beyond offered RX tail"
        );
        let desc = self.address(0x5b00) + (index * 16) as u64;
        let address = self.domain.read64(desc);
        assert_eq!(
            self.domain.read64(desc + 8),
            0,
            "receive descriptor must have its stale status cleared"
        );
        self.domain.put_packet(address, bytes);
        self.domain.put32(desc, kind);
        self.domain
            .put32(desc + 8, u32::from(status) | (u32::from(length) << 16));
        self.rx_head.set((index + 1) % 64);
        self.domain.pending_dma.fetch_add(1, Ordering::Relaxed);
    }
}
// SAFETY: stateful, independent symbolic register peer; all operations are
// synchronous and cannot name or access any physical device.
unsafe impl Registers for Rc<Peer> {
    fn read(&self, offset: usize) -> Result<u32> {
        if let Some(pm) = self.native_pm.borrow().as_ref() {
            assert_eq!(
                pm.actual_state.load(Ordering::Relaxed),
                0,
                "powered-off MMIO read"
            );
        }
        if self.fault.get() == Fault::Removed {
            return Ok(u32::MAX);
        }
        self.reads.borrow_mut().push(offset);
        if self.fail_read.get() == Some(offset) {
            if matches!(offset, 0x3ac | 0x3b0) && self.failed_acquire_reached_device.get() {
                self.semaphores.borrow_mut().insert(offset, true);
            }
            return Err(Error::Unsupported("injected native MMIO read failure"));
        }
        if matches!(offset, 0x3a8 | 0x3ac | 0x3b0) {
            if self.fault.get() == Fault::RamSemaphore
                || (offset == 0x3b0 && self.second_lock_busy.get())
            {
                return Ok(0);
            }
            let mut semaphores = self.semaphores.borrow_mut();
            let held = semaphores.entry(offset).or_insert(false);
            if *held {
                return Ok(0);
            }
            *held = true;
            return Ok(1);
        }
        if offset == 0x13000 && self.fault.get() == Fault::Snapshot {
            return Ok(0x20001);
        }
        Ok(self.get(offset))
    }
    fn write(&self, offset: usize, value: u32) -> Result<()> {
        if let Some(pm) = self.native_pm.borrow().as_ref() {
            assert_eq!(
                pm.actual_state.load(Ordering::Relaxed),
                0,
                "powered-off MMIO write"
            );
        }
        self.writes.borrow_mut().push((offset, value));
        if self.fail_write.get() == Some(offset) {
            if matches!(offset, 0x3ac | 0x3b0) && self.failed_release_reached_device.get() {
                self.semaphores.borrow_mut().insert(offset, false);
            }
            return Err(Error::Unsupported("injected native MMIO write failure"));
        }
        if matches!(offset, 0x3a8 | 0x3ac | 0x3b0) && value == 1 {
            assert_eq!(
                self.semaphores.borrow_mut().insert(offset, false),
                Some(true),
                "release must own this read-to-acquire semaphore"
            );
            return Ok(());
        }
        if (offset == 0 && value & 0x8000 != 0) || (offset == 0x3040 && value == 1) {
            self.reset_count.set(self.reset_count.get() + 1);
            assert_eq!(self.domain.engines.load(Ordering::Relaxed), 0);
            self.semaphores.borrow_mut().clear();
            self.pending_ack.borrow_mut().clear();
            self.pending_tx.set(None);
            self.tx_head.set(0);
            self.rx_head.set(0);
            self.set(0x7c0c, 0);
        }
        if offset == 0x3040 && value == 1 {
            self.set(
                offset,
                if self.fault.get() == Fault::Reset {
                    0
                } else if self.fault.get() == Fault::BootAllOnes {
                    u32::MAX
                } else if self.fault.get() == Fault::BootFailure {
                    0x11000000
                } else {
                    0x81000000
                },
            );
            if self.fault.get() == Fault::HostBoot {
                self.set(0xf00, 1);
            }
            return Ok(());
        }
        if offset == 0x404 && value == 0x40e0 && self.fault.get() != Fault::Reset {
            self.set(0x388, 1);
        }
        if offset == 0x404 && value == 0x180e0 && self.fault.get() != Fault::Reset {
            self.set(0x704, 0x10);
        }
        if offset == 0x200 && value == 0x8000 && self.fault.get() != Fault::RamCompletion {
            let address = self.get(0x208);
            self.set(0x20c, self.ram.borrow().get(&address).copied().unwrap_or(0));
            self.set(0x208, address + 4);
        }
        if offset == 0x36c && self.fault.get() != Fault::FirmwareAck {
            self.acknowledge_later(0x374, value);
        }
        if offset == 0xe00 {
            self.set(0xe04, 1);
            if self.fault.get() != Fault::FirmwareAck {
                self.acknowledge_later(0xe04, 0);
            }
        }
        if offset == 0x5a00 {
            self.set(
                0x5a10,
                if self.fault.get() == Fault::Cache {
                    0
                } else {
                    1
                },
            );
        }
        if offset == 0x54d0 {
            self.set(offset, value & !(1 << 5));
            return Ok(());
        }
        if matches!(offset, 0x7c08 | 0x5b08)
            && value & (1 << 31) == 0
            && self.fault.get() == Fault::QueueStop
        {
            self.set(offset, value | (1 << 31));
            return Ok(());
        }
        if offset == 0x7c08
            && value & (1 << 31) == 0
            && let Some(tail) = self.pending_tx.take()
        {
            self.transmit(tail);
        }
        self.set(offset, value);
        if matches!(offset, 0x7c08 | 0x5b08) {
            let bit = if offset == 0x7c08 { 1 } else { 2 };
            if value & (1 << 31) != 0 {
                let previous = self.domain.engines.fetch_or(bit, Ordering::Relaxed);
                if previous == 0 {
                    self.domain.engine_epoch.fetch_add(1, Ordering::Relaxed);
                }
            } else {
                // Finish any delayed descriptor fetch before acknowledging
                // engine disable. Posted writebacks still require a drain.
                self.domain.engines.fetch_and(!bit, Ordering::Relaxed);
            }
        }
        if offset == 0x7c10 && value as usize != self.tx_head.get() {
            if self.tx_delay.get() {
                self.pending_tx.set(Some(value as usize));
            } else {
                self.transmit(value as usize);
            }
        }
        Ok(())
    }
}
// SAFETY: explicit isolated device-model clock. Successful symbolic waits
// advance this same peer's time and acknowledge its pending hardware.
unsafe impl Timer for Peer {
    fn check(&self) -> core::result::Result<(), crate::bus::pci::power::Error> {
        if self.timing.get() {
            Ok(())
        } else {
            Err(crate::bus::pci::power::Error::Clock)
        }
    }
    fn wait_us(&self, us: u64) -> core::result::Result<(), crate::bus::pci::power::Error> {
        self.check()?;
        self.waits.borrow_mut().push(us);
        if self.fail_wait.get() == Some(self.waits.borrow().len()) {
            return Err(crate::bus::pci::power::Error::Clock);
        }
        self.elapsed.set(self.elapsed.get() + us);
        self.pending_ack
            .borrow_mut()
            .retain(|&(deadline, register, value)| {
                if self.elapsed.get() >= deadline {
                    self.set(register, value);
                    false
                } else {
                    true
                }
            });
        Ok(())
    }
}
fn hardware(peer: &Rc<Peer>) -> Hardware<Rc<Peer>, Rc<Peer>> {
    Hardware {
        registers: peer.clone(),
        timer: peer.clone(),
        reset_ownership: Cell::new(firmware::ResetOwnership::default()),
    }
}
type Fixture = (
    Aqtion<Rc<Peer>, Rc<Peer>, ControlledConfig>,
    Rc<Peer>,
    Arc<Domain>,
    Arc<AtomicUsize>,
);
fn setup(device: u16) -> Fixture {
    setup_with_memory(device, true)
}
fn setup_with_memory(device: u16, allocate_memory: bool) -> Fixture {
    let profile = Profile::identify(0x1d6a, device).unwrap();
    let domain = Arc::new(Domain::default());
    let dropped = Arc::new(AtomicUsize::new(0));
    let pages = Pages {
        dropped: dropped.clone(),
        next: Cell::new(0x40000000),
    };
    let peer = Peer::new(profile, domain.clone());
    let addr = Address {
        segment: 0,
        bus: 1,
        device: 0,
        function: 0,
    };
    let (function, config) = native_config(addr, device);
    *domain.config.lock().unwrap() = Some(config.clone());
    *peer.native_pm.borrow_mut() = Some(config);
    let owner = Aqtion {
        function: Some(function),
        hardware: Some(hardware(&peer)),
        domain: Some(domain.clone()),
        memory: allocate_memory.then(|| Memory::new(&pages, domain.clone()).unwrap()),
        profile,
        firmware: None,
        mac: Mac::default(),
        state: State::Stopped,
        power: PciPower::D0,
        tx_tail: 0,
        tx_head: 0,
        rx_head: 0,
        dropping: false,
    };
    (owner, peer, domain, dropped)
}
#[test]
fn exact_profiles_do_not_admit_usb_fiber_or_unimplemented_silicon_families() {
    if crate::test_process::isolate_model() {
        return;
    }
    for id in [
        0x07b1, 0x87b1, 0x08b1, 0x88b1, 0x11b1, 0x91b1, 0x12b1, 0x92b1, 0x00c0, 0x94c0, 0x93c0,
        0x04c0, 0x14c0, 0x34c0,
    ] {
        assert!(Profile::identify(0x1d6a, id).is_some());
    }
    for (vendor, id) in [
        (0x1d6a, 0x00b1),
        (0x1d6a, 0xd107),
        (0x1d6a, 0x12c0),
        (0x1d6a, 0x11c0),
        (0x2eca, 0xc101),
        (0x8086, 0x07b1),
    ] {
        assert_eq!(Profile::identify(vendor, id), None);
    }
}
#[test]
fn b1_and_antigua_rings_transfer_and_wrap() {
    if crate::test_process::isolate_model() {
        return;
    }
    for id in [0x07b1, 0x08b1, 0x11b1, 0x12b1, 0x04c0, 0x93c0] {
        let (mut net, peer, domain, dropped) = setup(id);
        net.start().unwrap();
        assert_eq!(net.mac(), Mac([2, 0x11, 0x22, 0x33, 0x44, 0x55]));
        if peer.profile.generation == Generation::Atlantic {
            peer.set(0x370, 0x100);
        } else {
            peer.set(0x13014, 0x830);
        }
        for n in 0..130 {
            let frame = vec![(n % 251) as u8; 14 + n % 1500];
            net.transmit(&frame).unwrap();
            let sent = peer.sent.borrow().last().unwrap().clone();
            assert_eq!(&sent[..frame.len()], frame);
            assert!(sent[frame.len()..].iter().all(|v| *v == 0));
            peer.receive(&frame, 0, 3, frame.len() as u16);
            assert_eq!(net.receive().unwrap(), Some(frame));
        }
        net.retire().unwrap();
        assert_eq!(net.state(), State::Stopped);
        assert!(domain.maps.lock().unwrap().is_empty());
        assert_eq!(dropped.load(Ordering::Relaxed), 4);
        assert!(domain.drain_count.load(Ordering::Relaxed) > 0);
    }
}
#[test]
fn bad_rx_metadata_and_fragment_chains_are_not_invented_as_frames() {
    if crate::test_process::isolate_model() {
        return;
    }
    let (mut net, peer, _, _) = setup(0x07b1);
    net.start().unwrap();
    peer.set(0x370, 0x100);
    for (kind, status, len) in [
        (0, 1, 60),
        (0, 3, 60),
        (0, 7, 60),
        (0x1000, 3, 60),
        (0, 3, 4097),
        (0, 3, 13),
        (0, 0x1003, 60),
    ] {
        peer.receive(&[9; 60], kind, status, len);
    }
    let mut reused = vec![0x52; 1514];
    let buffer = reused.as_ptr();
    peer.receive(&[7; 60], 0, 3, 60);
    assert_eq!(net.receive_reusing(&mut reused), Ok(true));
    assert_eq!(reused, [7; 60]);
    assert_eq!(reused.as_ptr(), buffer);
    assert_eq!(net.receive_reusing(&mut reused), Ok(false));
    assert_eq!(reused, [7; 60]);
    peer.receive(&[8; 60], 0, 3, 60);
    assert_eq!(net.receive().unwrap(), Some(vec![8; 60]));
    assert_eq!(net.receive().unwrap(), None);
    net.retire().unwrap();
}
#[test]
fn stop_and_dma_drain_failures_keep_all_resources_for_retry() {
    if crate::test_process::isolate_model() {
        return;
    }
    let (mut net, peer, domain, dropped) = setup(0x04c0);
    net.start().unwrap();
    peer.fault.set(Fault::QueueStop);
    assert!(net.stop().is_err());
    assert_eq!(net.state(), State::Faulted);
    assert_eq!(dropped.load(Ordering::Relaxed), 0);
    peer.fault.set(Fault::None);
    domain.fail_drain.store(true, Ordering::Relaxed);
    assert_eq!(net.stop(), Err(Error::Dma(DmaError::DrainUnavailable)));
    assert_eq!(dropped.load(Ordering::Relaxed), 0);
    domain.fail_drain.store(false, Ordering::Relaxed);
    net.stop().unwrap();
    domain.fail_unmap.store(true, Ordering::Relaxed);
    assert_eq!(net.retire(), Err(Error::Dma(DmaError::InvalidationFailed)));
    assert_eq!(dropped.load(Ordering::Relaxed), 0);
    domain.fail_unmap.store(false, Ordering::Relaxed);
    net.retire().unwrap();
    assert_eq!(dropped.load(Ordering::Relaxed), 4);
}
#[test]
fn firmware_failures_are_bounded_and_do_not_publish_a_working_link() {
    if crate::test_process::isolate_model() {
        return;
    }
    for (id, fault) in [
        (0x07b1, Fault::RamSemaphore),
        (0x07b1, Fault::RamCompletion),
        (0x04c0, Fault::Reset),
        (0x04c0, Fault::HostBoot),
        (0x04c0, Fault::BootFailure),
        (0x04c0, Fault::Snapshot),
        (0x04c0, Fault::FirmwareAck),
        (0x07b1, Fault::Cache),
    ] {
        let (mut net, peer, domain, dropped) = setup(id);
        peer.fault.set(fault);
        assert!(net.start().is_err());
        assert_eq!(net.state(), State::Faulted);
        assert!(!net.carrier());
        assert_eq!(dropped.load(Ordering::Relaxed), 0);
        assert!(peer.elapsed.get() < 20_000_000);
        peer.fault.set(Fault::None);
        net.retire().unwrap();
        assert_eq!(domain.maps.lock().unwrap().len(), 0);
    }
}
#[test]
fn loss_visibility_and_invalid_hardware_heads_are_terminal_until_recovery() {
    if crate::test_process::isolate_model() {
        return;
    }
    let (mut net, peer, domain, _) = setup(0x07b1);
    net.start().unwrap();
    peer.set(0x370, 0x100);
    peer.set(0x7c0c, 64);
    assert_eq!(net.send(&[1; 60]), Err(Error::RingHead(64)));
    peer.set(0x7c0c, 0);
    domain.fail_sync.store(true, Ordering::Relaxed);
    assert!(net.transmit(&[1; 60]).is_err());
    assert_eq!(net.state(), State::Faulted);
    domain.fail_sync.store(false, Ordering::Relaxed);
    net.restart().unwrap();
    peer.fault.set(Fault::Removed);
    assert!(net.receive().is_err());
    assert_eq!(net.state(), State::Faulted);
    peer.fault.set(Fault::None);
    net.retire().unwrap();
}
#[test]
fn unsupported_coherent_dma_cannot_be_silently_replaced_with_a_fence() {
    if crate::test_process::isolate_model() {
        return;
    }
    let (mut net, _, domain, _) = setup(0x07b1);
    domain.incoherent.store(true, Ordering::Relaxed);
    assert_eq!(net.start(), Err(Error::Dma(DmaError::CoherenceUnavailable)));
    domain.incoherent.store(false, Ordering::Relaxed);
    net.retire().unwrap();
}

#[test]
fn ordinary_suspend_stops_dma_before_d3_and_keeps_the_resumable_epoch() {
    if crate::test_process::isolate_model() {
        return;
    }
    let (mut net, peer, domain, dropped) = setup(0x07b1);
    let config = net.function.as_mut().unwrap().config();
    config.write16(6, 1 << 4).unwrap();
    config.write8(0x34, 0x40).unwrap();
    config.write16(0x40, 1).unwrap();
    config.write16(0x44, 0x8100).unwrap();
    net.start().unwrap();
    net.suspend().unwrap();
    assert_eq!(net.state(), State::Suspended);
    assert!(domain.drain_count.load(Ordering::Relaxed) > 0);
    assert_eq!(
        net.function
            .as_mut()
            .unwrap()
            .config()
            .read16(0x44)
            .unwrap(),
        3
    );
    let writes = peer.writes.borrow().len();
    assert!(!net.carrier());
    assert!(net.receive().is_err());
    net.retire().unwrap();
    assert_eq!(
        peer.writes.borrow().len(),
        writes,
        "suspended teardown must not access powered-off MMIO"
    );
    assert_eq!(dropped.load(Ordering::Relaxed), 4);
}

// Native volatile storage shared with the device/DMA peer. The platform only
// supplies RAM; ControlledConfig below models synchronous device responses at
// the ConfigSpace boundary. Neither component grants physical permission.
#[repr(align(4096))]
struct ConfigPage([u8; 4096]);
struct ConfigRam(UnsafeCell<ConfigPage>);
// SAFETY: same isolated raw-only configuration RAM and serialized fixture users;
// no controller/Source/Authority or safe mutable-memory borrow is retained here.
unsafe impl Send for ConfigRam {}
// SAFETY: same actual RAM-only exclusion contract as Send.
unsafe impl Sync for ConfigRam {}
impl ConfigRam {
    fn pointer(&self) -> NonNull<u8> {
        // SAFETY: retained exact UnsafeCell allocation, no native address cast.
        unsafe { NonNull::new_unchecked(core::ptr::addr_of_mut!((*self.0.get()).0).cast()) }
    }
}
struct NativePm {
    page: Arc<ConfigRam>,
    mode: Mutex<PmMode>,
    accesses: Mutex<Vec<ConfigAccess>>,
    actual_state: AtomicUsize,
    unmaps: AtomicUsize,
    fail_unmap: AtomicBool,
    deny_context: AtomicBool,
    retire_after: Mutex<Option<Arc<AtomicBool>>>,
}
// SAFETY: test-only stable RAM, serialized native volatile accesses and locked metadata.
unsafe impl Send for NativePm {}
// SAFETY: no references into the UnsafeCell storage are exposed.
unsafe impl Sync for NativePm {}
impl NativePm {
    fn power_state(&self) -> u16 {
        // SAFETY: this device-side observer owns the same aligned model ECAM
        // RAM as the configuration mapping; the fixture serializes access.
        unsafe {
            self.pointer()
                .as_ptr()
                .add(0x44)
                .cast::<u16>()
                .read_volatile()
                & 3
        }
    }

    fn pointer(&self) -> NonNull<u8> {
        // SAFETY: the model device, its pages and its domain belong to this test alone; no other
        // driver, worker or DMA touches them.
        self.page.pointer()
    }
    fn pm_accesses_since(&self, start: usize) -> Vec<ConfigAccess> {
        self.accesses.lock().unwrap()[start..]
            .iter()
            .copied()
            .filter(|access| {
                matches!(
                    access,
                    ConfigAccess::Read16(0x44, _) | ConfigAccess::Write16(0x44, _)
                )
            })
            .collect()
    }
    fn command(&self) -> u16 {
        // SAFETY: stable aligned native configuration RAM kept by this
        // model owner; serialized observation, with no write or access callback.
        unsafe { self.pointer().as_ptr().add(4).cast::<u16>().read_volatile() }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PmMode {
    Normal,
    RefuseD0,
    LoseD3Readback,
    RefuseD3,
    Removed,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ConfigAccess {
    Read8(u16, u8),
    Read16(u16, u16),
    Read32(u16, u32),
    Write8(u16, u8),
    Write16(u16, u16),
    Write32(u16, u32),
}
// Owns exactly the original native accessor. No copied configuration handle,
// test-only production hook, mapping callback or asynchronous writer is used.
// Response overrides are modeled firmware behavior, not native ECAM evidence.
struct ControlledConfig {
    native: OwnedConfig,
    peer: Arc<NativePm>,
}
impl ConfigSpace for ControlledConfig {
    fn read8(&self, offset: u16) -> core::result::Result<u8, crate::bus::pci::ConfigError> {
        let native = self.native.read8(offset)?;
        let value = if *self.peer.mode.lock().unwrap() == PmMode::Removed {
            u8::MAX
        } else {
            native
        };
        self.peer
            .accesses
            .lock()
            .unwrap()
            .push(ConfigAccess::Read8(offset, value));
        Ok(value)
    }
    fn read16(&self, offset: u16) -> core::result::Result<u16, crate::bus::pci::ConfigError> {
        let native = self.native.read16(offset)?;
        let mode = *self.peer.mode.lock().unwrap();
        let value = if mode == PmMode::Removed
            || (mode == PmMode::LoseD3Readback && offset == 0x44 && native & 3 == 3)
        {
            u16::MAX
        } else {
            native
        };
        self.peer
            .accesses
            .lock()
            .unwrap()
            .push(ConfigAccess::Read16(offset, value));
        Ok(value)
    }
    fn read32(&self, offset: u16) -> core::result::Result<u32, crate::bus::pci::ConfigError> {
        let native = self.native.read32(offset)?;
        let value = if *self.peer.mode.lock().unwrap() == PmMode::Removed {
            u32::MAX
        } else {
            native
        };
        self.peer
            .accesses
            .lock()
            .unwrap()
            .push(ConfigAccess::Read32(offset, value));
        Ok(value)
    }
    fn write8(
        &mut self,
        offset: u16,
        value: u8,
    ) -> core::result::Result<(), crate::bus::pci::ConfigError> {
        self.native.write8(offset, value)?;
        self.peer
            .accesses
            .lock()
            .unwrap()
            .push(ConfigAccess::Write8(offset, value));
        Ok(())
    }
    fn write16(
        &mut self,
        offset: u16,
        value: u16,
    ) -> core::result::Result<(), crate::bus::pci::ConfigError> {
        self.native.write16(offset, value)?;
        self.peer
            .accesses
            .lock()
            .unwrap()
            .push(ConfigAccess::Write16(offset, value));
        if offset == 0x44 {
            match (*self.peer.mode.lock().unwrap(), value & 3) {
                (PmMode::RefuseD0, 0) => {
                    self.native
                        .write16(offset, (value & !3) | 3)
                        .map_err(|error| {
                            crate::bus::pci::ConfigError::write_may_have_applied(
                                offset,
                                2,
                                error.reason,
                            )
                        })?
                }
                (PmMode::RefuseD3, 3) => {
                    self.native.write16(offset, value & !3).map_err(|error| {
                        crate::bus::pci::ConfigError::write_may_have_applied(
                            offset,
                            2,
                            error.reason,
                        )
                    })?
                }
                _ => {}
            }
            // Synchronous device response through the same accessor.
            // This internal read is absent from the caller trace.
            self.peer.actual_state.store(
                usize::from(
                    self.native.read16(offset).map_err(|error| {
                        crate::bus::pci::ConfigError::write_may_have_applied(
                            offset,
                            2,
                            error.reason,
                        )
                    })? & 3,
                ),
                Ordering::Relaxed,
            );
        }
        Ok(())
    }
    fn write32(
        &mut self,
        offset: u16,
        value: u32,
    ) -> core::result::Result<(), crate::bus::pci::ConfigError> {
        self.native.write32(offset, value)?;
        self.peer
            .accesses
            .lock()
            .unwrap()
            .push(ConfigAccess::Write32(offset, value));
        Ok(())
    }
}
struct ConfigResource(
    owned::PhysicalSpan,
    owned::model::MappingRetention<Arc<ConfigRam>>,
);

// SAFETY: symbolic test ECAM span, backed only by the fixture RAM.
unsafe impl owned::ResourceBacking for ConfigResource {
    fn begin_mapping(
        &self,
    ) -> core::result::Result<crate::bus::pci::assignment::OpaqueUse, owned::Error> {
        self.1.begin_mapping()
    }

    fn physical_span(&self) -> owned::PhysicalSpan {
        self.0
    }
    fn kind(&self) -> owned::ResourceKind {
        owned::ResourceKind::Configuration
    }
}
// This unique hosted configuration actor consumes the original RAM controller.
// Resource inventories retain only ConfigResource's noncyclic projection.
struct PmCustody {
    owner: owned::model::MappingOwner<Arc<ConfigRam>>,
    completion: Option<Arc<PmCompletion>>,
    retention: owned::model::MappingRetention<Arc<ConfigRam>>,
    resource: owned::Resource,
    address: Address,
    window: crate::bus::pci::EcamWindow,
}
struct PmCompletion {
    retention: owned::model::MappingRetention<Arc<ConfigRam>>,
    fault: stela_driver_base::sync::FirstFault<crate::bus::pci::ConfigError>,
}
// SAFETY: actual isolated RAM/context retention only, outside physical backing
// and affine controls. No provider/fault-schedule callback or per-use allocation.
unsafe impl crate::bus::pci::owned::grant::CompletionCheck for PmCompletion {
    fn postvalidate(&self) -> core::result::Result<(), crate::bus::pci::ConfigError> {
        use crate::bus::pci::{ConfigError, ConfigFailure};
        let result = self.retention.check_context().map_err(|error| {
            ConfigError::before_effect(match error {
                owned::Error::Assignment(error) => ConfigFailure::Ownership(error),
                error => ConfigFailure::Mapping(error),
            })
        });
        if let Err(error) = result {
            self.fault.latch(error);
        }
        result
    }
    fn failure(&self) -> stela_driver_base::sync::FaultState<crate::bus::pci::ConfigError> {
        self.fault.observe()
    }
}
impl Drop for PmCustody {
    fn drop(&mut self) {
        self.owner.abandon_admission();
        // Also cover direct model-custody abandonment outside DriverGrant.
        // The original preowned validator cannot become a per-access final drop.
        if let Some(completion) = self.completion.take() {
            core::mem::forget(completion);
        }
    }
}
struct PmEpoch(owned::model::MappingRetention<Arc<ConfigRam>>);
// SAFETY: this is only the same actual isolated RAM/count gate. It never
// retains a config writer, controller, native Source or Mapping/View backedge.
unsafe impl crate::bus::pci::owned::grant::ConfigEpoch for PmEpoch {
    fn check(&self) -> core::result::Result<(), owned::Error> {
        self.0.check_admission()
    }
    fn retain(
        &self,
    ) -> core::result::Result<Box<dyn crate::bus::pci::owned::grant::ConfigEpoch>, owned::Error>
    {
        self.0.check_admission()?;
        Ok(Box::new(Self(self.0.clone())))
    }
    fn same_epoch(
        &self,
        other: &dyn crate::bus::pci::owned::grant::ConfigEpoch,
    ) -> core::result::Result<bool, owned::Error> {
        self.0.check_admission()?;
        let Some(other) = other.as_any().downcast_ref::<Self>() else {
            return Ok(false);
        };
        other.0.check_admission()?;
        Ok(self.0.same_owner(&other.0))
    }
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}
// SAFETY: this cfg-only fixture owns the actual original model allocation and
// sole controller. The same resource and gate are consumed with its mapping;
// symbolic geometry grants no native/reset/IRQ/fabric/physical authority.
unsafe impl crate::bus::pci::owned::grant::ConfigCustody for PmCustody {
    fn address(&self) -> Address {
        self.address
    }
    fn window(&self) -> crate::bus::pci::EcamWindow {
        self.window
    }
    fn resource(&self) -> &owned::Resource {
        &self.resource
    }
    fn begin(
        &self,
        operation: crate::bus::pci::owned::grant::Operation,
    ) -> core::result::Result<
        crate::bus::pci::owned::grant::ConfigOperation,
        crate::bus::pci::ConfigError,
    > {
        use crate::bus::pci::{ConfigError, ConfigFailure, owned::grant::Operation};
        let (offset, width) = match operation {
            Operation::Read { offset, width } | Operation::Write { offset, width, .. } => {
                (offset, width)
            }
        };
        if !matches!(width, 1 | 2 | 4)
            || !usize::from(offset).is_multiple_of(width)
            || usize::from(offset)
                .checked_add(width)
                .is_none_or(|end| end > 4096)
        {
            return Err(ConfigError::before_effect(ConfigFailure::Access {
                offset,
                width,
            }));
        }
        let completion = self
            .completion
            .as_ref()
            .ok_or(ConfigError::before_effect(ConfigFailure::ForeignGrant))?;
        match crate::bus::pci::owned::grant::CompletionCheck::failure(completion.as_ref()) {
            stela_driver_base::sync::FaultState::Ready(error) => return Err(error),
            stela_driver_base::sync::FaultState::Publishing => {
                return Err(ConfigError::before_effect(ConfigFailure::FaultPublishing));
            }
            stela_driver_base::sync::FaultState::Empty => {}
        }
        let count = self
            .retention
            .begin_use(crate::bus::pci::assignment::UseKind::Configuration)
            .map_err(|error| {
                ConfigError::before_effect(match error {
                    owned::Error::Assignment(error) => ConfigFailure::Ownership(error),
                    error => ConfigFailure::Mapping(error),
                })
            })?;
        // SAFETY: actual original RAM gate and its preowned context validator
        // were created together; no source/control callback enters physical P.
        Ok(unsafe {
            crate::bus::pci::owned::grant::ConfigOperation::new(count, completion.clone())
        })
    }
    fn retirement_context(&self) -> core::result::Result<(), owned::Error> {
        self.retention.check_context()
    }
    fn retain_epoch(
        &self,
    ) -> core::result::Result<Box<dyn crate::bus::pci::owned::grant::ConfigEpoch>, owned::Error>
    {
        self.retention.check_admission()?;
        Ok(Box::new(PmEpoch(self.retention.clone())))
    }
}
// SAFETY: exclusive original hosted RAM fixture, not a native phase promotion.
unsafe impl crate::bus::pci::owned::grant::DriverCustody for PmCustody {}
// SAFETY: isolated registry, stable aligned native RAM, no physical device.
// check_context is a mapping-context callback, not a register-access callback.
unsafe impl owned::MappingPlatform for NativePm {
    fn check_context(&self, _: owned::AddressSpace) -> core::result::Result<(), owned::Error> {
        if self.deny_context.load(Ordering::Relaxed) {
            return Err(owned::Error::Unavailable);
        }
        Ok(())
    }
    unsafe fn map(
        &self,
        resource: &owned::Resource,
        request: owned::MapRequest,
    ) -> core::result::Result<owned::InstalledMapping, owned::MapFailure> {
        assert!(resource.backing().is::<ConfigResource>());
        assert_eq!(request.physical, resource.physical_span());
        Ok(owned::InstalledMapping {
            token: request.id,
            cpu_base: self.pointer(),
            physical: request.physical,
            attribute: request.attribute,
            protection: request.protection,
        })
    }
    unsafe fn unmap(
        &self,
        request: owned::MapRequest,
        mapping: owned::InstalledMapping,
    ) -> core::result::Result<(), owned::Error> {
        assert_eq!(request.id, mapping.token);
        if let Some(bar_retired) = self.retire_after.lock().unwrap().as_ref() {
            assert!(
                bar_retired.load(Ordering::Relaxed),
                "configuration before BAR retirement"
            );
        }
        self.unmaps.fetch_add(1, Ordering::Relaxed);
        if self.fail_unmap.load(Ordering::Relaxed) {
            return Err(owned::Error::UnmapFailed);
        }
        Ok(())
    }
}
fn native_config(address: Address, device: u16) -> (Function<ControlledConfig>, Arc<NativePm>) {
    let model = Arc::new(NativePm {
        page: Arc::new(ConfigRam(UnsafeCell::new(ConfigPage([0; 4096])))),
        mode: Mutex::new(PmMode::Normal),
        accesses: Mutex::new(Vec::new()),
        actual_state: AtomicUsize::new(0),
        unmaps: AtomicUsize::new(0),
        fail_unmap: AtomicBool::new(false),
        deny_context: AtomicBool::new(false),
        retire_after: Mutex::new(None),
    });
    let span = owned::PhysicalSpan::new(0x8010_0000, 4096).unwrap();
    let space = owned::AddressSpace {
        identity: 197,
        physical_bits: 48,
        virtual_bits: 48,
    };
    let authority =
        // SAFETY: the model device, its pages and its domain belong to this test alone; no other
        // driver, worker or DMA touches them.
        unsafe { owned::Authority::isolated_model(model.clone(), space, Vec::new()) }.unwrap();
    // SAFETY: SAME original real model ECAM allocation, not its symbolic PA;
    // hosted accesses are serialized and no native/affine resource exists.
    let mapping_owner = unsafe {
        owned::model::MappingOwner::from_owned(model.page.clone(), |owner| {
            if (owner.pointer().as_ptr() as usize).is_multiple_of(4096) {
                Ok(())
            } else {
                Err(crate::bus::pci::assignment::Error::Context)
            }
        })
    };
    let retention = mapping_owner.retention().unwrap();
    let resource = owned::Resource::new(Box::new(ConfigResource(span, retention.clone())));
    let window = crate::bus::pci::EcamWindow {
        base: 0x8000_0000,
        segment: 0,
        start_bus: 1,
        end_bus: 1,
    };
    // SAFETY: the model device, its pages and its domain belong to this test alone; no other
    // driver, worker or DMA touches them.
    let custody = Box::new(PmCustody {
        owner: mapping_owner,
        completion: Some(Arc::new(PmCompletion {
            retention: retention.clone(),
            fault: stela_driver_base::sync::FirstFault::new(),
        })),
        retention,
        resource: resource.clone(),
        address,
        window,
    });
    let close = custody.owner.admission_close().unwrap();
    // SAFETY: the close is from the same original custody owner before mapping.
    let grant = unsafe { crate::bus::pci::owned::grant::DriverGrant::from_boxed(custody, close) };
    let prepared = OwnedConfig::prepare(grant);
    let mapping = authority
        .map(
            resource,
            0,
            4096,
            owned::CacheAttribute::Uncached,
            owned::Protection::ReadWrite,
        )
        .unwrap();
    let mut config = match prepared.attach(mapping) {
        Ok(config) => config,
        Err((error, _parts)) => {
            panic!("native PM fixture: {error:?}")
        }
    };
    config
        .write32(0, (u32::from(device) << 16) | 0x1d6a)
        .unwrap();
    config.write32(8, 0x02000000).unwrap();
    config.write16(6, 1 << 4).unwrap();
    config.write8(0x34, 0x40).unwrap();
    config.write16(0x40, 1).unwrap();
    config.write16(0x44, 0).unwrap();
    let config = ControlledConfig {
        native: config,
        peer: model.clone(),
    };
    (Function::new(address, config).unwrap(), model)
}

// Preserve these historical selector names. Their native volatile transport is
// real RAM; refused/absent device responses are explicitly modeled by the wrapper.
#[test]
fn native_failed_d0_resume_is_retryable_and_never_uses_powered_off_mmio() {
    if crate::test_process::isolate_model() {
        return;
    }
    let (mut net, peer, domain, dropped) = setup(0x07b1);
    let pm = peer.native_pm.borrow().as_ref().unwrap().clone();
    net.start().unwrap();
    net.suspend().unwrap();
    let writes = peer.writes.borrow().len();
    let reads = peer.reads.borrow().len();
    *pm.mode.lock().unwrap() = PmMode::RefuseD0;
    let access_start = pm.accesses.lock().unwrap().len();
    assert_eq!(
        net.resume_owned(),
        Err(Error::Pci(crate::block::BlockError::Unsupported(
            "PCI function did not enter D0"
        )))
    );
    assert_eq!(
        pm.pm_accesses_since(access_start),
        [
            ConfigAccess::Read16(0x44, 3),
            ConfigAccess::Write16(0x44, 0),
            ConfigAccess::Read16(0x44, 3),
        ]
    );
    assert!(net.function.is_some() && net.hardware.is_some() && net.domain.is_some());
    assert!(net.memory.is_some());
    assert_eq!(domain.maps.lock().unwrap().len(), 4);
    assert_eq!(dropped.load(Ordering::Relaxed), 0);
    assert_eq!(net.state(), State::Suspended);
    assert_eq!(net.power, PciPower::Unknown);
    assert_eq!(pm.actual_state.load(Ordering::Relaxed), 3);
    net.stop().unwrap();
    assert!(net.restart().is_err());
    assert_eq!(peer.writes.borrow().len(), writes);
    assert_eq!(peer.reads.borrow().len(), reads);
    *pm.mode.lock().unwrap() = PmMode::Normal;
    let access_start = pm.accesses.lock().unwrap().len();
    net.resume_owned().unwrap();
    assert_eq!(
        pm.pm_accesses_since(access_start),
        [
            ConfigAccess::Read16(0x44, 3),
            ConfigAccess::Write16(0x44, 0),
            ConfigAccess::Read16(0x44, 0),
        ]
    );
    assert_eq!(net.power, PciPower::D0);
    assert_eq!(net.state(), State::Running);
    assert_eq!(domain.engine_epoch.load(Ordering::Relaxed), 2);
    peer.set(0x370, 0x100);
    net.transmit(&[7; 60]).unwrap();
    net.retire().unwrap();
}

#[test]
fn native_d3_readback_loss_keeps_drained_retirement_and_d0_retry_safe() {
    if crate::test_process::isolate_model() {
        return;
    }
    for retire in [false, true] {
        let (mut net, peer, domain, dropped) = setup(0x04c0);
        let pm = peer.native_pm.borrow().as_ref().unwrap().clone();
        net.start().unwrap();
        *pm.mode.lock().unwrap() = PmMode::LoseD3Readback;
        let access_start = pm.accesses.lock().unwrap().len();
        assert_eq!(net.suspend(), Err(Error::Removed));
        assert_eq!(
            pm.pm_accesses_since(access_start),
            [
                ConfigAccess::Read16(0x44, 0),
                ConfigAccess::Write16(0x44, 3),
                ConfigAccess::Read16(0x44, u16::MAX),
            ]
        );
        assert!(net.function.is_some() && net.hardware.is_some() && net.domain.is_some());
        assert!(net.memory.is_some());
        assert_eq!(domain.maps.lock().unwrap().len(), 4);
        assert_eq!(dropped.load(Ordering::Relaxed), 0);
        assert_eq!(domain.drained_epoch.load(Ordering::Relaxed), 1);
        assert_eq!(net.power, PciPower::Unknown);
        assert_eq!(net.state(), State::Stopped);
        assert_eq!(pm.actual_state.load(Ordering::Relaxed), 3);
        let writes = peer.writes.borrow().len();
        let reads = peer.reads.borrow().len();
        net.stop().unwrap();
        assert!(net.restart().is_err());
        if retire {
            net.retire().unwrap();
            assert_eq!(dropped.load(Ordering::Relaxed), 4);
            assert!(domain.maps.lock().unwrap().is_empty());
        }
        assert_eq!(peer.writes.borrow().len(), writes);
        assert_eq!(peer.reads.borrow().len(), reads);
        if !retire {
            *pm.mode.lock().unwrap() = PmMode::Normal;
            // The model answers config again; the native PMCSR is D3hot.
            net.resume_owned().unwrap();
            assert_eq!(net.power, PciPower::D0);
            assert_eq!(net.state(), State::Running);
            assert_eq!(domain.engine_epoch.load(Ordering::Relaxed), 2);
            net.retire().unwrap();
            assert_eq!(dropped.load(Ordering::Relaxed), 4);
        }
    }
}

#[test]
fn initial_native_d0_failure_keeps_uninitialized_owner_without_mmio() {
    if crate::test_process::isolate_model() {
        return;
    }
    let (mut net, peer, domain, dropped) = setup_with_memory(0x07b1, false);
    let pm = peer.native_pm.borrow().as_ref().unwrap().clone();
    // Model attach before Memory allocation, using its power helper.
    net.state = State::Faulted;
    net.function
        .as_mut()
        .unwrap()
        .config()
        .write16(0x44, 3)
        .unwrap();
    *pm.mode.lock().unwrap() = PmMode::RefuseD0;
    let access_start = pm.accesses.lock().unwrap().len();
    assert_eq!(
        power_up(net.function.as_mut().unwrap(), &peer, &mut net.power),
        Err(Error::Pci(crate::block::BlockError::Unsupported(
            "PCI function did not enter D0"
        )))
    );
    assert_eq!(
        pm.pm_accesses_since(access_start),
        [
            ConfigAccess::Read16(0x44, 3),
            ConfigAccess::Write16(0x44, 0),
            ConfigAccess::Read16(0x44, 3),
        ]
    );
    assert_eq!(pm.actual_state.load(Ordering::Relaxed), 3);
    assert!(net.function.is_some() && net.hardware.is_some() && net.domain.is_some());
    assert!(net.memory.is_none());
    assert!(domain.maps.lock().unwrap().is_empty());
    assert_eq!(net.power, PciPower::Unknown);
    assert!(net.stop().is_err());
    assert!(net.restart().is_err());
    assert!(net.retire().is_err());
    assert!(peer.writes.borrow().is_empty());
    assert!(peer.reads.borrow().is_empty());
    assert_eq!(dropped.load(Ordering::Relaxed), 0);
}

#[test]
fn modeled_removed_config_resume_keeps_owners_and_never_uses_powered_off_mmio() {
    if crate::test_process::isolate_model() {
        return;
    }
    for device in [0x07b1, 0x04c0] {
        let (mut net, peer, domain, dropped) = setup(device);
        let pm = peer.native_pm.borrow().as_ref().unwrap().clone();
        net.start().unwrap();
        net.suspend().unwrap();
        let reads = peer.reads.borrow().len();
        let writes = peer.writes.borrow().len();
        let access_start = pm.accesses.lock().unwrap().len();
        *pm.mode.lock().unwrap() = PmMode::Removed;
        assert_eq!(net.resume_owned(), Err(Error::Removed));
        assert_eq!(
            &pm.accesses.lock().unwrap()[access_start..],
            &[ConfigAccess::Read16(4, u16::MAX)]
        );
        assert_eq!(net.state(), State::Suspended);
        assert_eq!(net.power, PciPower::Unknown);
        assert_eq!(pm.actual_state.load(Ordering::Relaxed), 3);
        assert!(net.function.is_some() && net.hardware.is_some() && net.domain.is_some());
        assert!(net.memory.is_some());
        assert_eq!(domain.maps.lock().unwrap().len(), 4);
        assert_eq!(dropped.load(Ordering::Relaxed), 0);
        assert!(net.restart().is_err());
        assert_eq!(peer.reads.borrow().len(), reads);
        assert_eq!(peer.writes.borrow().len(), writes);
        *pm.mode.lock().unwrap() = PmMode::Normal;
        net.resume_owned().unwrap();
        assert_eq!(net.power, PciPower::D0);
        assert_eq!(domain.engine_epoch.load(Ordering::Relaxed), 2);
        net.retire().unwrap();
        assert_eq!(dropped.load(Ordering::Relaxed), 4);
    }
}

fn firmware4(peer: &Peer, flb: bool) {
    peer.set(0x18, 0x0401005a);
    peer.set(0x348, 1);
    peer.set(0x388, u32::from(!flb));
}
fn reset_releases(peer: &Peer) -> Vec<usize> {
    peer.writes
        .borrow()
        .iter()
        .filter_map(|&(register, value)| {
            (matches!(register, 0x3ac | 0x3b0) && value == 1).then_some(register)
        })
        .collect()
}
#[test]
fn reset_semaphores_are_read_to_acquire_and_write_to_release() {
    if crate::test_process::isolate_model() {
        return;
    }
    let (_, peer, _, _) = setup(0x07b1);
    assert_eq!(peer.read(0x3ac), Ok(1));
    assert_eq!(peer.read(0x3ac), Ok(0));
    peer.write(0x3ac, 1).unwrap();
    assert_eq!(peer.read(0x3ac), Ok(1));
}
#[test]
fn firmware4_second_lock_timeout_releases_only_the_acquired_first_lock_then_retries() {
    if crate::test_process::isolate_model() {
        return;
    }
    for flb in [false, true] {
        let (_, peer, _, _) = setup(0x07b1);
        let hw = hardware(&peer);
        firmware4(&peer, flb);
        peer.second_lock_busy.set(true);
        assert_eq!(
            hw.reset(peer.profile),
            Err(Error::Timeout { register: 0x3b0 })
        );
        assert_eq!(reset_releases(&peer), vec![0x3ac]);
        assert_eq!(peer.reset_count.get(), 0);
        peer.second_lock_busy.set(false);
        hw.reset(peer.profile).unwrap();
        assert_eq!(peer.reset_count.get(), if flb { 2 } else { 1 });
        // Successful reset consumed the locks; a new epoch can acquire again.
        hw.reset(peer.profile).unwrap();
        assert!(peer.semaphores.borrow().values().all(|held| !held));
    }
}
#[test]
fn firmware4_pre_reset_mmio_errors_release_both_locks_in_reverse_order() {
    if crate::test_process::isolate_model() {
        return;
    }
    for flb in [false, true] {
        for failed_read in [0x53c, 0] {
            let (_, peer, _, _) = setup(0x07b1);
            let hw = hardware(&peer);
            firmware4(&peer, flb);
            peer.fail_read.set(Some(failed_read));
            assert!(hw.reset(peer.profile).is_err());
            assert_eq!(peer.reset_count.get(), 0);
            assert_eq!(reset_releases(&peer), vec![0x3b0, 0x3ac]);
            peer.fail_read.set(None);
            hw.reset(peer.profile).unwrap();
        }
        let (_, peer, _, _) = setup(0x07b1);
        let hw = hardware(&peer);
        firmware4(&peer, flb);
        peer.fail_write.set(Some(0x404));
        assert!(hw.reset(peer.profile).is_err());
        assert_eq!(reset_releases(&peer), vec![0x3b0, 0x3ac]);
        peer.fail_write.set(None);
        hw.reset(peer.profile).unwrap();
    }
}
#[test]
fn firmware4_uncertain_semaphore_release_or_acquire_blocks_reacquisition() {
    if crate::test_process::isolate_model() {
        return;
    }
    for (failed_release, reached_device) in
        [(0x3ac, false), (0x3b0, false), (0x3ac, true), (0x3b0, true)]
    {
        let (_, peer, _, _) = setup(0x07b1);
        let hw = hardware(&peer);
        firmware4(&peer, false);
        peer.fail_read.set(Some(0x53c));
        peer.fail_write.set(Some(failed_release));
        peer.failed_release_reached_device.set(reached_device);
        assert!(hw.reset(peer.profile).is_err());
        assert_eq!(reset_releases(&peer), vec![0x3b0, 0x3ac]);
        assert_eq!(
            peer.semaphores.borrow().get(&failed_release),
            Some(&!reached_device)
        );
        // A different owner can acquire after a release whose response failed.
        peer.semaphores.borrow_mut().insert(failed_release, true);
        peer.fail_read.set(None);
        peer.fail_write.set(None);
        let reads = peer.reads.borrow().len();
        let writes = peer.writes.borrow().len();
        assert!(hw.reset(peer.profile).is_err());
        assert_eq!(peer.reads.borrow().len(), reads);
        assert_eq!(peer.writes.borrow().len(), writes);
    }
    for failed_acquire in [0x3ac, 0x3b0] {
        for reached_device in [false, true] {
            let (_, peer, _, _) = setup(0x07b1);
            let hw = hardware(&peer);
            firmware4(&peer, false);
            peer.fail_read.set(Some(failed_acquire));
            peer.failed_acquire_reached_device.set(reached_device);
            assert!(hw.reset(peer.profile).is_err());
            assert_eq!(
                reset_releases(&peer),
                if failed_acquire == 0x3b0 {
                    vec![0x3ac]
                } else {
                    vec![]
                }
            );
            assert_eq!(
                peer.semaphores
                    .borrow()
                    .get(&failed_acquire)
                    .copied()
                    .unwrap_or(false),
                reached_device
            );
            peer.fail_read.set(None);
            let reads = peer.reads.borrow().len();
            let writes = peer.writes.borrow().len();
            assert!(hw.reset(peer.profile).is_err());
            assert_eq!(peer.reads.borrow().len(), reads);
            assert_eq!(peer.writes.borrow().len(), writes);
        }
    }
}
#[test]
fn firmware4_reset_store_uncertainty_and_flb_post_first_reset_abort_keep_ownership() {
    if crate::test_process::isolate_model() {
        return;
    }
    for flb in [false, true] {
        let (_, peer, _, _) = setup(0x07b1);
        let hw = hardware(&peer);
        firmware4(&peer, flb);
        peer.fail_write.set(Some(0));
        assert!(hw.reset(peer.profile).is_err());
        assert!(reset_releases(&peer).is_empty());
        peer.fail_write.set(None);
        assert!(hw.reset(peer.profile).is_err());
    }
    let (_, peer, _, _) = setup(0x07b1);
    let hw = hardware(&peer);
    firmware4(&peer, true);
    peer.fail_write.set(Some(0x520));
    assert!(hw.reset(peer.profile).is_err());
    assert_eq!(peer.reset_count.get(), 1);
    assert!(reset_releases(&peer).is_empty());
    peer.fail_write.set(None);
    assert!(hw.reset(peer.profile).is_err());
    assert_eq!(peer.reset_count.get(), 1);
}
#[test]
fn restart_after_delayed_traffic_drains_each_epoch_and_rebuilds_device_heads() {
    if crate::test_process::isolate_model() {
        return;
    }
    for id in [0x07b1, 0x04c0] {
        let (mut net, peer, domain, dropped) = setup(id);
        net.start().unwrap();
        if peer.profile.generation == Generation::Atlantic {
            peer.set(0x370, 0x100);
        } else {
            peer.set(0x13014, 0x830);
        }
        net.transmit(&[3; 60]).unwrap();
        peer.receive(&[5; 60], 0, 3, 60);
        assert_eq!(net.receive().unwrap(), Some(vec![5; 60]));
        peer.tx_delay.set(true);
        domain.defer_writebacks.store(true, Ordering::Relaxed);
        net.transmit(&[7; 60]).unwrap();
        assert_eq!(peer.pending_tx.get(), Some(2));
        assert!(domain.pending_dma.load(Ordering::Relaxed) > 0);
        net.restart().unwrap();
        assert!(peer.pending_tx.get().is_none());
        assert_eq!(peer.tx_head.get(), 0);
        assert_eq!(peer.rx_head.get(), 0);
        assert_eq!(domain.pending_dma.load(Ordering::Relaxed), 0);
        assert_eq!(domain.engine_epoch.load(Ordering::Relaxed), 2);
        assert_eq!(domain.drained_epoch.load(Ordering::Relaxed), 1);
        peer.tx_delay.set(false);
        domain.defer_writebacks.store(false, Ordering::Relaxed);
        assert!(domain.posted.lock().unwrap().is_empty());
        net.transmit(&[9; 60]).unwrap();
        peer.receive(&[11; 60], 0, 3, 60);
        assert_eq!(net.receive().unwrap(), Some(vec![11; 60]));
        net.retire().unwrap();
        assert_eq!(domain.drained_epoch.load(Ordering::Relaxed), 2);
        assert_eq!(domain.pending_dma.load(Ordering::Relaxed), 0);
        assert_eq!(dropped.load(Ordering::Relaxed), 4);
        assert_eq!(
            peer.sent.borrow().as_slice(),
            &[vec![3; 60], vec![7; 60], vec![9; 60]]
        );
    }
}
#[test]
fn delayed_firmware_ack_is_required_before_link_shutdown_finishes() {
    if crate::test_process::isolate_model() {
        return;
    }
    for id in [0x07b1, 0x04c0] {
        let (mut net, peer, _, _) = setup(id);
        net.start().unwrap();
        peer.ack_delay.set(500);
        let before = peer.elapsed.get();
        net.stop().unwrap();
        assert!(peer.elapsed.get() >= before + 500);
        assert!(peer.pending_ack.borrow().is_empty());
        net.retire().unwrap();
    }
}
fn assert_bus_master_rejection(body: impl FnOnce()) {
    let rejection = catch_unwind(AssertUnwindSafe(body))
        .expect_err("model accepted DMA teardown with PCI bus mastering enabled");
    let message = rejection
        .downcast::<std::string::String>()
        .expect("expected the model's BusMaster assertion");
    assert!(message.contains("PCI bus mastering must be disabled before DMA drain or unmap"));
}
#[test]
fn model_rejects_dma_drain_before_native_bus_master_clear_and_keeps_pending_writes() {
    if crate::test_process::isolate_model() {
        return;
    }
    for device in [0x07b1, 0x04c0] {
        let (mut net, peer, domain, dropped) = setup(device);
        net.start().unwrap();
        domain.defer_writebacks.store(true, Ordering::Relaxed);
        peer.receive(&[19; 60], 0, 3, 60);
        net.hw().stop_queues().unwrap();
        assert_eq!(domain.engines.load(Ordering::Relaxed), 0);
        let mappings = domain.maps.lock().unwrap().len();
        let pending = domain.pending_dma.load(Ordering::Relaxed);
        let posted = domain.posted.lock().unwrap().clone();
        assert_eq!(mappings, 4);
        assert!(pending > 0);
        assert!(!posted.is_empty());
        assert_ne!(peer.native_pm.borrow().as_ref().unwrap().command() & 4, 0);

        // Wrong order: queue shutdown alone must not admit the drain. Catch
        // rejection so this control also proves all DMA evidence is held.
        // SAFETY: the model device is stopped and drained, so nothing still uses what this
        // releases.
        assert_bus_master_rejection(|| unsafe { domain.drain_posted_writes().unwrap() });
        assert_eq!(domain.drain_count.load(Ordering::Relaxed), 0);
        assert_eq!(domain.drained_epoch.load(Ordering::Relaxed), 0);
        assert_eq!(domain.pending_dma.load(Ordering::Relaxed), pending);
        assert_eq!(*domain.posted.lock().unwrap(), posted);
        assert_eq!(domain.maps.lock().unwrap().len(), mappings);
        assert_eq!(dropped.load(Ordering::Relaxed), 0);

        // The real production path now clears and reads back Command before
        // entering this same peer drain. No test flag certifies quiescence.
        net.retire().unwrap();
        assert_eq!(peer.native_pm.borrow().as_ref().unwrap().command() & 4, 0);
        assert_eq!(domain.drained_epoch.load(Ordering::Relaxed), 1);
        assert_eq!(domain.pending_dma.load(Ordering::Relaxed), 0);
        assert!(domain.posted.lock().unwrap().is_empty());
        assert!(domain.maps.lock().unwrap().is_empty());
        assert_eq!(dropped.load(Ordering::Relaxed), 4);
    }
}
#[test]
fn model_rejects_dma_drain_and_unmap_without_bus_master_readback() {
    if crate::test_process::isolate_model() {
        return;
    }
    for device in [0x07b1, 0x04c0] {
        for unmap in [false, true] {
            let (mut net, peer, domain, dropped) = setup(device);
            net.start().unwrap();
            if unmap {
                net.stop().unwrap();
            } else {
                domain.defer_writebacks.store(true, Ordering::Relaxed);
                peer.receive(&[23; 60], 0, 3, 60);
                net.hw().stop_queues().unwrap();
            }
            let drains = domain.drain_count.load(Ordering::Relaxed);
            let epoch = domain.drained_epoch.load(Ordering::Relaxed);
            let pending = domain.pending_dma.load(Ordering::Relaxed);
            let posted = domain.posted.lock().unwrap().clone();
            assert_eq!(domain.maps.lock().unwrap().len(), 4);
            if !unmap {
                assert!(pending > 0);
                assert!(!posted.is_empty());
            }
            let config = net.function.as_mut().unwrap().config();
            let command = config.read16(4).unwrap();
            config.write16(4, command & !4).unwrap();
            // This device-side volatile sample is absent from the
            // caller trace: cleared RAM cannot substitute for driver readback.
            assert_eq!(peer.native_pm.borrow().as_ref().unwrap().command() & 4, 0);
            let rejection = catch_unwind(AssertUnwindSafe(|| {
                if unmap {
                    net.memory.as_mut().unwrap().retire().unwrap();
                } else {
                    // SAFETY: the model device is stopped and drained, so nothing still uses what
                    // this releases.
                    unsafe { domain.drain_posted_writes().unwrap() };
                }
            }))
            .expect_err("model accepted teardown without command readback");
            let message = rejection
                .downcast_ref::<std::string::String>()
                .map(|s| s.as_str())
                .or_else(|| rejection.downcast_ref::<&str>().copied())
                .expect("expected the readback assertion");
            assert!(message.contains(
                "PCI bus-master clear requires caller readback before DMA drain or unmap"
            ));
            assert_eq!(domain.drain_count.load(Ordering::Relaxed), drains);
            assert_eq!(domain.drained_epoch.load(Ordering::Relaxed), epoch);
            assert_eq!(domain.pending_dma.load(Ordering::Relaxed), pending);
            assert_eq!(*domain.posted.lock().unwrap(), posted);
            assert_eq!(domain.maps.lock().unwrap().len(), 4);
            assert_eq!(dropped.load(Ordering::Relaxed), 0);
            assert!(net.memory.is_some());
            // Re-enter production stop even if the control broke
            // an already stopped owner's readback assumption.
            net.state = State::Faulted;
            net.retire().unwrap();
            assert_eq!(domain.drain_count.load(Ordering::Relaxed), drains + 1);
            assert_eq!(domain.pending_dma.load(Ordering::Relaxed), 0);
            assert!(domain.posted.lock().unwrap().is_empty());
            assert!(domain.maps.lock().unwrap().is_empty());
            assert_eq!(dropped.load(Ordering::Relaxed), 4);
        }
    }
}

#[test]
fn model_rejects_unmap_after_native_bus_master_reenable_and_keeps_dma() {
    if crate::test_process::isolate_model() {
        return;
    }
    for device in [0x07b1, 0x04c0] {
        let (mut net, _, domain, dropped) = setup(device);
        net.start().unwrap();
        net.stop().unwrap();
        assert_eq!(domain.drained_epoch.load(Ordering::Relaxed), 1);
        let mappings = domain.maps.lock().unwrap().len();
        assert_eq!(mappings, 4);
        let config = net.function.as_mut().unwrap().config();
        let command = config.read16(4).unwrap();
        config.write16(4, command | 4).unwrap();
        assert_ne!(config.read16(4).unwrap() & 4, 0);

        // Violate the stopped owner's configuration invariant,
        // then try to reclaim through the real Memory/DmaMapping unmap path.
        assert_bus_master_rejection(|| net.memory.as_mut().unwrap().retire().unwrap());
        assert!(net.memory.is_some());
        assert_eq!(domain.maps.lock().unwrap().len(), mappings);
        assert_eq!(dropped.load(Ordering::Relaxed), 0);
        assert_eq!(domain.drain_count.load(Ordering::Relaxed), 1);

        // Force production stop to establish its boundary again rather than
        // relying on the Stopped state invalidated above.
        net.state = State::Faulted;
        net.retire().unwrap();
        assert_eq!(domain.drain_count.load(Ordering::Relaxed), 2);
        assert!(domain.maps.lock().unwrap().is_empty());
        assert_eq!(dropped.load(Ordering::Relaxed), 4);
    }
}
#[test]
#[should_panic(expected = "historical drain cannot release a newer engine epoch")]
fn model_rejects_historical_drain_after_a_new_engine_epoch() {
    let (mut net, _, _, _) = setup(0x07b1);
    net.start().unwrap();
    net.stop().unwrap();
    net.start().unwrap();
    net.hw().stop_queues().unwrap();
    // Satisfy the independent PCI boundary, but omit this epoch's
    // domain drain. The historical-drain control must reach that exact check.
    let config = net.function.as_mut().unwrap().config();
    let command = config.read16(4).unwrap();
    config.write16(4, command & !4).unwrap();
    assert_eq!(config.read16(4).unwrap() & 4, 0);
    net.memory.as_mut().unwrap().retire().unwrap();
}
fn omit_visibility_before_tx(payload: bool) {
    let (mut net, peer, domain, _) = setup(0x07b1);
    net.start().unwrap();
    net.memory.as_mut().unwrap().transmit(0, &[7; 60]).unwrap();
    let descriptor = peer.address(0x7c00);
    let address = if payload {
        // Inspect native fixture storage without performing a modeled DMA read.
        // SAFETY: the model device, its pages and its domain belong to this test alone; no other
        // driver, worker or DMA touches them.
        unsafe {
            domain
                .raw(descriptor, 8)
                .as_ptr()
                .cast::<u64>()
                .read_volatile()
        }
    } else {
        descriptor
    };
    domain.published.lock().unwrap().remove(&address);
    peer.write(0x7c10, 1).unwrap();
}
#[test]
#[should_panic(expected = "DMA without fresh visibility")]
fn model_rejects_tx_descriptor_dma_without_visibility() {
    omit_visibility_before_tx(false);
}
#[test]
#[should_panic(expected = "DMA without fresh visibility")]
fn model_rejects_tx_payload_dma_without_visibility() {
    omit_visibility_before_tx(true);
}

#[test]
fn delayed_rx_writeback_is_invisible_until_completion_and_stop_drains_pending_writes() {
    if crate::test_process::isolate_model() {
        return;
    }
    let (mut net, peer, domain, dropped) = setup(0x07b1);
    net.start().unwrap();
    peer.set(0x370, 0x100);
    domain.defer_writebacks.store(true, Ordering::Relaxed);
    peer.receive(&[13; 60], 0, 3, 60);
    assert_eq!(net.receive().unwrap(), None);
    assert!(!domain.posted.lock().unwrap().is_empty());
    domain.complete_posted();
    assert_eq!(net.receive().unwrap(), Some(vec![13; 60]));
    peer.receive(&[17; 60], 0, 3, 60);
    domain.fail_drain.store(true, Ordering::Relaxed);
    assert_eq!(net.retire(), Err(Error::Dma(DmaError::DrainUnavailable)));
    assert!(!domain.posted.lock().unwrap().is_empty());
    assert_eq!(dropped.load(Ordering::Relaxed), 0);
    domain.fail_drain.store(false, Ordering::Relaxed);
    net.retire().unwrap();
    assert!(domain.posted.lock().unwrap().is_empty());
    assert_eq!(dropped.load(Ordering::Relaxed), 4);
}

#[test]
fn firmware4_first_lock_timeout_does_not_release_another_owner() {
    if crate::test_process::isolate_model() {
        return;
    }
    let (_, peer, _, _) = setup(0x07b1);
    let hw = hardware(&peer);
    firmware4(&peer, false);
    peer.semaphores.borrow_mut().insert(0x3ac, true);
    assert_eq!(
        hw.reset(peer.profile),
        Err(Error::Timeout { register: 0x3ac })
    );
    assert!(reset_releases(&peer).is_empty());
    assert_eq!(peer.semaphores.borrow().get(&0x3ac), Some(&true));
    assert_eq!(peer.reset_count.get(), 0);
}

#[test]
fn native_refused_d3_requires_fresh_d0_acknowledgement_before_restart() {
    if crate::test_process::isolate_model() {
        return;
    }
    let (mut net, peer, domain, dropped) = setup(0x07b1);
    let pm = peer.native_pm.borrow().as_ref().unwrap().clone();
    net.start().unwrap();
    *pm.mode.lock().unwrap() = PmMode::RefuseD3;
    let access_start = pm.accesses.lock().unwrap().len();
    assert_eq!(
        net.suspend(),
        Err(Error::Readback {
            register: 0x44,
            observed: 0
        })
    );
    assert_eq!(
        pm.pm_accesses_since(access_start),
        [
            ConfigAccess::Read16(0x44, 0),
            ConfigAccess::Write16(0x44, 3),
            ConfigAccess::Read16(0x44, 0),
        ]
    );
    assert!(net.function.is_some() && net.hardware.is_some() && net.domain.is_some());
    assert!(net.memory.is_some());
    assert_eq!(domain.maps.lock().unwrap().len(), 4);
    assert_eq!(dropped.load(Ordering::Relaxed), 0);
    assert_eq!(pm.actual_state.load(Ordering::Relaxed), 0);
    assert_eq!(net.state(), State::Stopped);
    assert_eq!(net.power, PciPower::Unknown);
    let reads = peer.reads.borrow().len();
    let writes = peer.writes.borrow().len();
    assert!(net.restart().is_err());
    assert_eq!(peer.writes.borrow().len(), writes);
    assert_eq!(peer.reads.borrow().len(), reads);
    *pm.mode.lock().unwrap() = PmMode::Normal;
    let access_start = pm.accesses.lock().unwrap().len();
    net.resume_owned().unwrap();
    assert_eq!(
        pm.pm_accesses_since(access_start),
        [ConfigAccess::Read16(0x44, 0)]
    );
    assert_eq!(net.power, PciPower::D0);
    net.retire().unwrap();
}

// Authored unchanged-source comparison and parser checks: UNRUN. This peer is
// independent of the firmware/DMA model above.
mod registers_oracle {
    use super::super::registers as f;
    use super::*;
    use alloc::{collections::BTreeSet, string::String};

    const INITIAL: &[u32] = &[0, 0xa5c3_5a3c, 0x5a3c_a5c3];
    const RINGS: &[u32] = &[0, 7, 31];
    const FILTERS: &[u32] = &[0, 16, 32];
    const TCS: &[u32] = &[0, 1, 3];
    const SCALAR: &[u32] = &[0];
    const BIT: &[u32] = &[0, 1];
    const WORD: &[u32] = &[0, 0x1234_5678, u32::MAX];
    const TAIL: &[u32] = &[0, 63, 8191];
    const CASE_COUNT: usize = 600;

    #[derive(Clone, Copy)]
    enum Access {
        Bits,
        Word,
        Get(u32), // Independent declared result bound, not derived from Field.
    }
    struct RegisterCase {
        name: &'static str,
        field: Field,
        indexes: &'static [u32],
        values: &'static [u32],
        access: Access,
    }
    // Production constants only. Input domains are explicit donor-width
    // encodings, independent of Field addresses/masks/shifts. No expected result
    // or address is calculated from this table. The external comparison gets
    // them from original Linux; the verifier controls below use donor literals.
    const FIELDS: &[RegisterCase] = &[
        RegisterCase {
            name: "hw_atl_rdm_rx_desc_en_set",
            field: f::RDM_RX_DESC_EN_SET,
            indexes: RINGS,
            values: BIT,
            access: Access::Bits,
        },
        RegisterCase {
            name: "hw_atl_reg_rx_dma_desc_base_addresslswset",
            field: f::REG_RX_DMA_DESC_BASE_ADDRESSLSWSET,
            indexes: RINGS,
            values: WORD,
            access: Access::Word,
        },
        RegisterCase {
            name: "hw_atl_reg_rx_dma_desc_base_addressmswset",
            field: f::REG_RX_DMA_DESC_BASE_ADDRESSMSWSET,
            indexes: RINGS,
            values: WORD,
            access: Access::Word,
        },
        RegisterCase {
            name: "hw_atl_rdm_rx_desc_len_set",
            field: f::RDM_RX_DESC_LEN_SET,
            indexes: RINGS,
            values: &[0, 1, 1023],
            access: Access::Bits,
        },
        RegisterCase {
            name: "hw_atl_rdm_rx_desc_data_buff_size_set",
            field: f::RDM_RX_DESC_DATA_BUFF_SIZE_SET,
            indexes: RINGS,
            values: &[0, 1, 31],
            access: Access::Bits,
        },
        RegisterCase {
            name: "hw_atl_rdm_rx_desc_head_buff_size_set",
            field: f::RDM_RX_DESC_HEAD_BUFF_SIZE_SET,
            indexes: RINGS,
            values: &[0, 1, 31],
            access: Access::Bits,
        },
        RegisterCase {
            name: "hw_atl_rdm_rx_desc_head_splitting_set",
            field: f::RDM_RX_DESC_HEAD_SPLITTING_SET,
            indexes: RINGS,
            values: BIT,
            access: Access::Bits,
        },
        RegisterCase {
            name: "hw_atl_reg_rx_dma_desc_tail_ptr_set",
            field: f::RX_TAIL,
            indexes: RINGS,
            values: TAIL,
            access: Access::Word,
        },
        RegisterCase {
            name: "hw_atl_tdm_tx_desc_en_set",
            field: f::TDM_TX_DESC_EN_SET,
            indexes: RINGS,
            values: BIT,
            access: Access::Bits,
        },
        RegisterCase {
            name: "hw_atl_reg_tx_dma_desc_base_addresslswset",
            field: f::REG_TX_DMA_DESC_BASE_ADDRESSLSWSET,
            indexes: RINGS,
            values: WORD,
            access: Access::Word,
        },
        RegisterCase {
            name: "hw_atl_reg_tx_dma_desc_base_addressmswset",
            field: f::REG_TX_DMA_DESC_BASE_ADDRESSMSWSET,
            indexes: RINGS,
            values: WORD,
            access: Access::Word,
        },
        RegisterCase {
            name: "hw_atl_tdm_tx_desc_len_set",
            field: f::TDM_TX_DESC_LEN_SET,
            indexes: RINGS,
            values: &[0, 1, 1023],
            access: Access::Bits,
        },
        RegisterCase {
            name: "hw_atl_tdm_tx_desc_head_ptr_get",
            field: f::TX_HEAD,
            indexes: RINGS,
            values: SCALAR,
            access: Access::Get(8191),
        },
        RegisterCase {
            name: "hw_atl_reg_tx_dma_desc_tail_ptr_set",
            field: f::TX_TAIL,
            indexes: RINGS,
            values: TAIL,
            access: Access::Word,
        },
        RegisterCase {
            name: "hw_atl_rpfl2_uc_flr_en_set",
            field: f::RPFL2_UC_FLR_EN_SET,
            indexes: FILTERS,
            values: BIT,
            access: Access::Bits,
        },
        RegisterCase {
            name: "hw_atl_rpfl2unicast_flr_act_set",
            field: f::RPFL2UNICAST_FLR_ACT_SET,
            indexes: FILTERS,
            values: &[0, 1, 7],
            access: Access::Bits,
        },
        RegisterCase {
            name: "hw_atl_rpfl2unicast_dest_addresslsw_set",
            field: f::RPFL2UNICAST_DEST_ADDRESSLSW_SET,
            indexes: FILTERS,
            values: WORD,
            access: Access::Word,
        },
        RegisterCase {
            name: "hw_atl_rpfl2unicast_dest_addressmsw_set",
            field: f::RPFL2UNICAST_DEST_ADDRESSMSW_SET,
            indexes: FILTERS,
            values: &[0, 1, 65535],
            access: Access::Bits,
        },
        RegisterCase {
            name: "hw_atl_rpb_rx_pkt_buff_size_per_tc_set",
            field: f::RPB_RX_PKT_BUFF_SIZE_PER_TC_SET,
            indexes: TCS,
            values: &[0, 1, 511],
            access: Access::Bits,
        },
        RegisterCase {
            name: "hw_atl_rpb_rx_buff_hi_threshold_per_tc_set",
            field: f::RPB_RX_BUFF_HI_THRESHOLD_PER_TC_SET,
            indexes: TCS,
            values: &[0, 1, 16383],
            access: Access::Bits,
        },
        RegisterCase {
            name: "hw_atl_tpb_tx_pkt_buff_size_per_tc_set",
            field: f::TPB_TX_PKT_BUFF_SIZE_PER_TC_SET,
            indexes: TCS,
            values: &[0, 1, 255],
            access: Access::Bits,
        },
        RegisterCase {
            name: "hw_atl_tpb_tx_buff_hi_threshold_per_tc_set",
            field: f::TPB_TX_BUFF_HI_THRESHOLD_PER_TC_SET,
            indexes: TCS,
            values: &[0, 1, 8191],
            access: Access::Bits,
        },
        RegisterCase {
            name: "hw_atl_rdm_rx_dma_desc_cache_init_done_get",
            field: f::RDM_RX_DMA_DESC_CACHE_INIT_DONE_GET,
            indexes: SCALAR,
            values: SCALAR,
            access: Access::Get(1),
        },
        RegisterCase {
            name: "hw_atl2_rpfl2_uc_flr_tag_set",
            field: f::ATL2_RPFL2_UC_FLR_TAG_SET,
            indexes: FILTERS,
            values: &[0, 1, 63],
            access: Access::Bits,
        },
        RegisterCase {
            name: "hw_atl2_rpfl2_bc_flr_tag_set",
            field: f::ATL2_RPFL2_BC_FLR_TAG_SET,
            indexes: SCALAR,
            values: &[0, 1, 63],
            access: Access::Bits,
        },
        RegisterCase {
            name: "hw_atl2_rpf_rss_hash_type_set",
            field: f::ATL2_RPF_RSS_HASH_TYPE_SET,
            indexes: SCALAR,
            values: &[0, 1, 511],
            access: Access::Bits,
        },
        RegisterCase {
            name: "hw_atl2_rpf_act_rslvr_section_en_set",
            field: f::ATL2_RPF_ACT_RSLVR_SECTION_EN_SET,
            indexes: SCALAR,
            values: &[0, 1, 65535],
            access: Access::Bits,
        },
        RegisterCase {
            name: "hw_atl2_rpf_act_rslvr_section_en_get",
            field: f::ATL2_RPF_ACT_RSLVR_SECTION_EN_GET,
            indexes: SCALAR,
            values: SCALAR,
            access: Access::Get(65535),
        },
    ];

    #[derive(Debug)]
    struct Row {
        case: usize,
        index: u32,
        initial: u32,
        value: u32,
        address: u32,
        word: u32,
        result: u32,
    }
    type ParseResult<T> = core::result::Result<T, &'static str>;

    fn parse_row(line: &str) -> ParseResult<Row> {
        let columns: Vec<_> = line.split(' ').collect();
        if columns.len() != 9 {
            return Err("column count");
        }
        let case = FIELDS
            .iter()
            .position(|s| s.name == columns[0])
            .ok_or("unknown helper")?;
        let mut numbers = [0u32; 8];
        for (number, column) in numbers.iter_mut().zip(&columns[1..]) {
            if column.is_empty() || !column.bytes().all(|c| c.is_ascii_digit()) {
                return Err("non-decimal column");
            }
            *number = column.parse().map_err(|_| "u32 overflow")?;
        }
        let [index, initial, value, address, word, result, reads, writes] = numbers;
        let spec = &FIELDS[case];
        if !spec.indexes.contains(&index)
            || !INITIAL.contains(&initial)
            || !spec.values.contains(&value)
        {
            return Err("input outside declared cases");
        }
        if address >= 0x10000 || !address.is_multiple_of(4) {
            return Err("register outside aligned BAR");
        }
        match spec.access {
            Access::Get(max) => {
                if reads != 1 || writes != 0 || word != initial || result > max {
                    return Err("invalid getter trace");
                }
            }
            Access::Word => {
                if reads != 0 || writes != 1 || result != 0 {
                    return Err("invalid full-word setter trace");
                }
            }
            Access::Bits => {
                if reads != 1 || writes > 1 || result != 0 || (writes == 0) != (word == initial) {
                    return Err("invalid RMW setter trace");
                }
            }
        }
        Ok(Row {
            case,
            index,
            initial,
            value,
            address,
            word,
            result,
        })
    }

    fn parse_trace(text: &str) -> ParseResult<Vec<Row>> {
        let text = text.strip_suffix('\n').ok_or("missing final newline")?;
        let lines: Vec<_> = text.split('\n').collect();
        if lines.len() != CASE_COUNT + 2
            || lines[0] != "AQTION_REGISTERS_V1 600"
            || lines[CASE_COUNT + 1] != "END 600"
        {
            return Err("trace framing/count");
        }
        if FIELDS.len() != 28 {
            return Err("helper inventory");
        }
        let mut remaining = BTreeSet::new();
        let mut names = BTreeSet::new();
        for (case, spec) in FIELDS.iter().enumerate() {
            if !names.insert(spec.name) {
                return Err("duplicate helper inventory");
            }
            for &index in spec.indexes {
                for &initial in INITIAL {
                    for &value in spec.values {
                        if !remaining.insert((case, index, initial, value)) {
                            return Err("duplicate input inventory");
                        }
                    }
                }
            }
        }
        if remaining.len() != CASE_COUNT {
            return Err("case inventory");
        }
        let mut rows = Vec::with_capacity(CASE_COUNT);
        for line in &lines[1..=CASE_COUNT] {
            let row = parse_row(line)?;
            if !remaining.remove(&(row.case, row.index, row.initial, row.value)) {
                return Err("duplicate or unexpected case");
            }
            rows.push(row);
        }
        if !remaining.is_empty() {
            return Err("missing cases");
        }
        Ok(rows)
    }

    struct RegisterPeer {
        address: usize,
        word: Cell<u32>,
        reads: Cell<u32>,
        writes: Cell<u32>,
    }
    // SAFETY: isolated synchronous scalar test state, no physical MMIO/DMA. Each
    // callback checks the exact donor-observed address and keeps the raw word.
    unsafe impl Registers for RegisterPeer {
        fn read(&self, offset: usize) -> Result<u32> {
            assert_eq!(offset, self.address, "donor read address");
            assert_eq!(self.writes.get(), 0, "unexpected read after write");
            self.reads.set(self.reads.get() + 1);
            Ok(self.word.get())
        }
        fn write(&self, offset: usize, value: u32) -> Result<()> {
            assert_eq!(offset, self.address, "donor write address");
            self.writes.set(self.writes.get() + 1);
            self.word.set(value);
            Ok(())
        }
    }
    struct NoDelay;
    // SAFETY: scalar-only isolated peer; any attempted wait traps.
    unsafe impl Timer for NoDelay {
        fn check(&self) -> core::result::Result<(), crate::bus::pci::power::Error> {
            Ok(())
        }
        fn wait_us(&self, _: u64) -> core::result::Result<(), crate::bus::pci::power::Error> {
            panic!("LLH field operations must not delay")
        }
    }

    fn compare_trace(text: &str) {
        let rows = parse_trace(text).expect("complete, strictly framed Linux trace");
        assert_eq!(rows.len(), CASE_COUNT);
        for row in rows {
            let spec = &FIELDS[row.case];
            let hw = Hardware {
                registers: RegisterPeer {
                    address: row.address as usize,
                    word: Cell::new(row.initial),
                    reads: Cell::new(0),
                    writes: Cell::new(0),
                },
                timer: NoDelay,
                reset_ownership: Cell::new(firmware::ResetOwnership::default()),
            };
            match spec.access {
                Access::Get(_) => assert_eq!(
                    hw.get(spec.field, row.index as usize).unwrap(),
                    row.result,
                    "getter {} index {} initial {:08x}",
                    spec.name,
                    row.index,
                    row.initial
                ),
                Access::Bits | Access::Word => {
                    hw.set(spec.field, row.index as usize, row.value).unwrap()
                }
            }
            assert_eq!(hw.registers.word.get(), row.word, "raw final word: {row:?}");
            let (reads, writes) = (hw.registers.reads.get(), hw.registers.writes.get());
            assert!(reads + writes > 0, "empty Rust trace: {row:?}");
            // Linux may suppress unchanged RMW writes, while Hardware::set writes
            // unconditionally. Check Rust's own access shape, not equal counts.
            match spec.access {
                Access::Get(_) => assert_eq!((reads, writes), (1, 0)),
                Access::Word => assert_eq!((reads, writes), (0, 1)),
                Access::Bits => assert_eq!((reads, writes), (1, 1)),
            }
        }
    }

    #[test]
    #[ignore = "oracle: STELA_AQTION_REGISTERS_ORACLE"]
    fn unchanged_linux_aqtion_registers() {
        if crate::test_process::isolate_model() {
            return;
        }
        let path = std::env::var_os("STELA_AQTION_REGISTERS_ORACLE")
            .expect("set STELA_AQTION_REGISTERS_ORACLE");
        assert!(
            std::fs::metadata(&path).unwrap().len() <= 256_000,
            "oversized trace"
        );
        compare_trace(&std::fs::read_to_string(path).unwrap());
    }

    // Synthetic verifier controls, not executed original-Linux evidence. These
    // (base, stride, mask, shift) literals were transcribed from the pinned
    // hw_atl{,2}_llh_internal.h macros used by the 28 original helpers above.
    // Keep them independent of Field and Hardware::set/get, including when a
    // production constant changes. Full-word helpers ignore mask/shift.
    fn control_trace() -> String {
        const DONOR_ENCODINGS: &[(u32, u32, u32, u32)] = &[
            (0x5b08, 0x20, 0x8000_0000, 31), // RDM_DESCDEN
            (0x5b00, 0x20, u32::MAX, 0),     // RX_DMA_DESC_BASE_ADDRLSW
            (0x5b04, 0x20, u32::MAX, 0),     // RX_DMA_DESC_BASE_ADDRMSW
            (0x5b08, 0x20, 0x0000_1ff8, 3),  // RDM_DESCDLEN
            (0x5b18, 0x20, 0x0000_001f, 0),  // RDM_DESCDDATA_SIZE
            (0x5b18, 0x20, 0x0000_1f00, 8),  // RDM_DESCDHDR_SIZE
            (0x5b08, 0x20, 0x1000_0000, 28), // RDM_DESCDHDR_SPLIT
            (0x5b10, 0x20, u32::MAX, 0),     // RX_DMA_DESC_TAIL_PTR
            (0x7c08, 0x40, 0x8000_0000, 31), // TDM_DESCDEN
            (0x7c00, 0x40, u32::MAX, 0),     // TX_DMA_DESC_BASE_ADDRLSW
            (0x7c04, 0x40, u32::MAX, 0),     // TX_DMA_DESC_BASE_ADDRMSW
            (0x7c08, 0x40, 0x0000_1ff8, 3),  // TDM_DESCDLEN
            (0x7c0c, 0x40, 0x0000_1fff, 0),  // TDM_DESCDHD
            (0x7c10, 0x40, u32::MAX, 0),     // TX_DMA_DESC_TAIL_PTR
            (0x5114, 0x08, 0x8000_0000, 31), // RPFL2UC_ENF
            (0x5114, 0x08, 0x0007_0000, 16), // RPFL2UC_ACTF
            (0x5110, 0x08, u32::MAX, 0),     // RPFL2UC_DAFLSW
            (0x5114, 0x08, 0x0000_ffff, 0),  // RPFL2UC_DAFMSW
            (0x5710, 0x10, 0x0000_01ff, 0),  // RPB_RXBBUF_SIZE
            (0x5714, 0x10, 0x3fff_0000, 16), // RPB_RXBHI_THRESH
            (0x7910, 0x10, 0x0000_00ff, 0),  // TPB_TXBBUF_SIZE
            (0x7914, 0x10, 0x1fff_0000, 16), // TPB_TXBHI_THRESH
            (0x5a10, 0x00, 0x0000_0001, 0),  // RDM_RX_DMA_DESC_CACHE_INIT_DONE
            (0x5114, 0x08, 0x0fc0_0000, 22), // ATL2_RPFL2UC_TAG
            (0x50f0, 0x00, 0x0000_003f, 0),  // ATL2_RPF_L2_BC_TAG
            (0x54c8, 0x00, 0x0000_01ff, 0),  // ATL2_RPF_PIF_RPF_RSS_HASH_TYPEI
            (0x6ff0, 0x00, 0x0000_ffff, 0),  // ATL2_RPF_REC_TAB_EN (set)
            (0x6ff0, 0x00, 0x0000_ffff, 0),  // ATL2_RPF_REC_TAB_EN (get)
        ];
        assert_eq!(DONOR_ENCODINGS.len(), FIELDS.len());
        let mut trace = String::from("AQTION_REGISTERS_V1 600\n");
        for (spec, &(base, stride, mask, shift)) in FIELDS.iter().zip(DONOR_ENCODINGS) {
            for &index in spec.indexes {
                for &initial in INITIAL {
                    for &value in spec.values {
                        let (word, result, reads, writes) = match spec.access {
                            Access::Get(_) => (initial, (initial & mask) >> shift, 1, 0),
                            Access::Word => (value, 0, 0, 1),
                            Access::Bits => {
                                let word = (initial & !mask) | (value << shift);
                                (word, 0, 1, u32::from(word != initial))
                            }
                        };
                        trace.push_str(&format!(
                            "{} {index} {initial} {value} {} {word} {result} {reads} {writes}\n",
                            spec.name,
                            base + index * stride
                        ));
                    }
                }
            }
        }
        trace.push_str("END 600\n");
        trace
    }

    #[test]
    fn accepts_complete_aqtion_register_control_trace() {
        if crate::test_process::isolate_model() {
            return;
        }
        let trace = control_trace();
        assert_eq!(parse_trace(&trace).unwrap().len(), CASE_COUNT);
        compare_trace(&trace);
    }

    #[test]
    fn rejects_parseable_aqtion_register_comparison_corruptions() {
        if crate::test_process::isolate_model() {
            return;
        }
        let trace = control_trace();
        compare_trace(&trace); // A baseline failure cannot count as rejection.
        // Every replacement keeps a complete valid input inventory and passes
        // the parser. Only the real Hardware comparison can reject these rows.
        // Exercise addresses for every access class, words for both setter
        // classes, and in-range results for all three getters.
        for (original, corrupted) in [
            (
                "hw_atl_rdm_rx_desc_en_set 0 0 0 23304 0 0 1 0\n",
                "hw_atl_rdm_rx_desc_en_set 0 0 0 23308 0 0 1 0\n",
            ),
            (
                "hw_atl_reg_rx_dma_desc_base_addresslswset 0 0 0 23296 0 0 0 1\n",
                "hw_atl_reg_rx_dma_desc_base_addresslswset 0 0 0 23300 0 0 0 1\n",
            ),
            (
                "hw_atl_tdm_tx_desc_head_ptr_get 0 0 0 31756 0 0 1 0\n",
                "hw_atl_tdm_tx_desc_head_ptr_get 0 0 0 31760 0 0 1 0\n",
            ),
            (
                "hw_atl_rdm_rx_desc_en_set 0 0 1 23304 2147483648 0 1 1\n",
                "hw_atl_rdm_rx_desc_en_set 0 0 1 23304 2147483649 0 1 1\n",
            ),
            (
                "hw_atl_reg_rx_dma_desc_base_addresslswset 0 0 0 23296 0 0 0 1\n",
                "hw_atl_reg_rx_dma_desc_base_addresslswset 0 0 0 23296 1 0 0 1\n",
            ),
            (
                "hw_atl_tdm_tx_desc_head_ptr_get 0 0 0 31756 0 0 1 0\n",
                "hw_atl_tdm_tx_desc_head_ptr_get 0 0 0 31756 0 1 1 0\n",
            ),
            (
                "hw_atl_rdm_rx_dma_desc_cache_init_done_get 0 0 0 23056 0 0 1 0\n",
                "hw_atl_rdm_rx_dma_desc_cache_init_done_get 0 0 0 23056 0 1 1 0\n",
            ),
            (
                "hw_atl2_rpf_act_rslvr_section_en_get 0 0 0 28656 0 0 1 0\n",
                "hw_atl2_rpf_act_rslvr_section_en_get 0 0 0 28656 0 1 1 0\n",
            ),
        ] {
            assert_eq!(trace.matches(original).count(), 1, "control target");
            let mutated = trace.replacen(original, corrupted, 1);
            assert_eq!(parse_trace(&mutated).unwrap().len(), CASE_COUNT);
            assert!(
                std::panic::catch_unwind(|| compare_trace(&mutated)).is_err(),
                "comparison accepted corruption: {corrupted}"
            );
        }
    }

    #[test]
    fn rejects_malformed_aqtion_register_oracle_rows_and_incomplete_coverage() {
        if crate::test_process::isolate_model() {
            return;
        }
        const VALID: &str = "hw_atl_rdm_rx_desc_en_set 0 0 0 23304 0 0 1 0";
        assert_eq!(parse_row(VALID).unwrap().address, 23304);
        for bad in [
            "", // Empty and non-decimal/trailing columns.
            "unknown 0 0 0 23304 0 0 1 0",
            "hw_atl_rdm_rx_desc_en_set 0 0 0 23304 0 0 1",
            "hw_atl_rdm_rx_desc_en_set 0 0 0 23304 0 0 1 0 junk",
            "hw_atl_rdm_rx_desc_en_set +0 0 0 23304 0 0 1 0",
            "hw_atl_rdm_rx_desc_en_set 0 0x0 0 23304 0 0 1 0",
            "hw_atl_rdm_rx_desc_en_set 0 0 0 4294967296 0 0 1 0",
            "hw_atl_rdm_rx_desc_en_set 32 0 0 23304 0 0 1 0",
            "hw_atl_rdm_rx_desc_en_set 0 4294967295 0 23304 0 0 1 0",
            "hw_atl_rdm_rx_desc_en_set 0 0 2 23304 0 0 1 0",
            "hw_atl_rdm_rx_desc_en_set 0 0 0 65536 0 0 1 0",
            "hw_atl_rdm_rx_desc_en_set 0 0 0 23305 0 0 1 0",
            "hw_atl_rdm_rx_desc_en_set 0 0 0 23304 0 0 0 0",
            "hw_atl_rdm_rx_desc_en_set 0 0 0 23304 0 0 1 2",
            "hw_atl_rdm_rx_desc_en_set 0 0 0 23304 0 1 1 0",
            "hw_atl_rdm_rx_desc_en_set 0 0 0 23304 1 0 1 0",
            "hw_atl_reg_rx_dma_desc_base_addresslswset 0 0 0 23296 0 0 1 1",
            "hw_atl_reg_rx_dma_desc_base_addresslswset 0 0 0 23296 0 0 0 0",
            "hw_atl_reg_rx_dma_desc_base_addresslswset 0 0 0 23296 0 0 0 2",
            "hw_atl_reg_rx_dma_desc_base_addresslswset 0 0 0 23296 0 1 0 1",
            "hw_atl_tdm_tx_desc_head_ptr_get 0 0 0 31756 0 0 0 0",
            "hw_atl_tdm_tx_desc_head_ptr_get 0 0 0 31756 0 0 2 0",
            "hw_atl_tdm_tx_desc_head_ptr_get 0 0 0 31756 0 0 1 1",
            "hw_atl_tdm_tx_desc_head_ptr_get 0 0 0 31756 1 0 1 0",
            "hw_atl_tdm_tx_desc_head_ptr_get 0 0 0 31756 0 8192 1 0",
        ] {
            assert!(parse_row(bad).is_err(), "accepted malformed row: {bad}");
        }
        let repeated = core::iter::repeat_n(VALID, CASE_COUNT)
            .collect::<Vec<_>>()
            .join("\n");
        let duplicate = format!("AQTION_REGISTERS_V1 600\n{repeated}\nEND 600\n");
        assert_eq!(
            parse_trace(&duplicate).unwrap_err(),
            "duplicate or unexpected case"
        );
        let missing = format!("AQTION_REGISTERS_V1 600\n{VALID}\nEND 600\n");
        assert_eq!(parse_trace(&missing).unwrap_err(), "trace framing/count");
        assert_eq!(
            parse_trace(&(duplicate.clone() + "junk\n")).unwrap_err(),
            "trace framing/count"
        );
        assert_eq!(
            parse_trace(duplicate.trim_end_matches('\n')).unwrap_err(),
            "missing final newline"
        );
        assert!(parse_trace("").is_err());
    }
}

mod timing;

mod retirement;

mod reset_source;

mod liveness;

mod ownership;

#[test]
fn config_context_denial_keeps_exact_error_and_initial_owner_without_mmio() {
    use crate::bus::pci::{ConfigError, ConfigFailure};
    let (mut net, peer, domain, dropped) = setup(0x07b1);
    let pm = peer.native_pm.borrow().as_ref().unwrap().clone();
    let reads = peer.reads.borrow().len();
    let writes = peer.writes.borrow().len();
    let config_accesses = pm.accesses.lock().unwrap().len();
    let expected = Error::Config(ConfigError::before_effect(ConfigFailure::Mapping(
        owned::Error::Unavailable,
    )));
    pm.deny_context.store(true, Ordering::Relaxed);
    assert_eq!(net.start(), Err(expected.clone()));
    assert_eq!(
        adapter(expected),
        LinkError::Pci(crate::block::BlockError::Config(
            ConfigError::before_effect(ConfigFailure::Mapping(owned::Error::Unavailable))
        ))
    );
    assert_eq!(net.state(), State::Faulted);
    assert!(
        net.function.is_some()
            && net.hardware.is_some()
            && net.domain.is_some()
            && net.memory.is_some()
    );
    assert_eq!(peer.reads.borrow().len(), reads);
    assert_eq!(peer.writes.borrow().len(), writes);
    assert_eq!(pm.accesses.lock().unwrap().len(), config_accesses);
    assert_eq!(pm.unmaps.load(Ordering::Relaxed), 0);
    drop(net);
    assert_eq!(pm.unmaps.load(Ordering::Relaxed), 0);
    assert_eq!(dropped.load(Ordering::Relaxed), 0);
    assert_eq!(domain.maps.lock().unwrap().len(), 4);
}
