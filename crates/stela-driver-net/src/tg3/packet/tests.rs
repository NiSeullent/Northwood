// SPDX-License-Identifier: GPL-2.0-only
//! Independent device peer with separate device-visible RAM and nonidentity
//! IOVAs. All tests are authored UNRUN until the coordinator obtains a lease.
extern crate std;
use super::*;
use crate::bus::dma::{
    mapping::{Direction, DmaError, DomainMapping},
    memory::{CacheAttribute, CpuMapping, PageBacking, PinnedPages},
    pages::DmaPages,
};
use alloc::{
    alloc::{alloc_zeroed, dealloc},
    boxed::Box,
    collections::BTreeMap,
    vec,
};
use core::{
    alloc::Layout,
    cell::Cell,
    ops::Range,
    ptr::NonNull,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
};
use std::sync::Mutex;

struct Ram {
    address: NonNull<u8>,
    physical: Vec<u64>,
    layout: Layout,
    drops: Arc<AtomicUsize>,
}
// SAFETY: symbolic test-only RAM, serialized raw accesses and stable backing.
unsafe impl Send for Ram {}
// SAFETY: symbolic test-only RAM; no native DMA or safe references alias it.
unsafe impl Sync for Ram {}
// SAFETY: exact page aligned allocation and immutable kept symbolic labels.
unsafe impl PageBacking for Ram {
    fn cpu_base(&self) -> NonNull<u8> {
        self.address
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
        self.drops.fetch_add(1, Ordering::Relaxed);
        // SAFETY: the pointer came from the matching allocation with this layout and is freed once.
        unsafe { dealloc(self.address.as_ptr(), self.layout) };
    }
}
struct Pages {
    next: Cell<u64>,
    drops: Arc<AtomicUsize>,
    allocated: Cell<usize>,
    fail_after: Option<usize>,
}
impl PageSource for Pages {
    fn allocate(
        &self,
        bytes: usize,
        alignment: usize,
        cache: CacheAttribute,
    ) -> Result<PinnedPages, MemoryError> {
        assert_eq!(cache, CacheAttribute::WriteBack);
        let count = self.allocated.get();
        if self.fail_after == Some(count) {
            return Err(MemoryError::OutOfMemory);
        }
        self.allocated.set(count + 1);
        let bytes = bytes.div_ceil(4096) * 4096;
        let layout = Layout::from_size_align(bytes, alignment).unwrap();
        let address =
            // SAFETY: the layout has a nonzero size.
            NonNull::new(unsafe { alloc_zeroed(layout) }).ok_or(MemoryError::OutOfMemory)?;
        let base = self.next.get();
        self.next.set(base + bytes as u64);
        PinnedPages::new(
            Box::new(Ram {
                address,
                physical: (0..bytes / 4096).map(|i| base + i as u64 * 4096).collect(),
                layout,
                drops: self.drops.clone(),
            }),
            bytes,
            alignment,
        )
    }
}
/// Device-completed ranges the CPU may take back, as the furthest end
/// completed from each start address.
#[derive(Default)]
struct Completions {
    ends: BTreeMap<u64, u64>,
    longest: u64,
}
impl Completions {
    fn complete(&mut self, range: Range<u64>) {
        self.longest = self.longest.max(range.end - range.start);
        let end = self.ends.entry(range.start).or_insert(range.end);
        *end = (*end).max(range.end);
    }
    /// A range handed back to the device ends every completion overlapping it.
    fn revoke(&mut self, range: Range<u64>) {
        let overlapping: Vec<u64> = self
            .ends
            .range(range.start.saturating_sub(self.longest)..range.end)
            .filter(|&(_, &end)| end > range.start)
            .map(|(&start, _)| start)
            .collect();
        for start in overlapping {
            self.ends.remove(&start);
        }
    }
    fn covers(&self, range: Range<u64>) -> bool {
        self.ends
            .range(range.start.saturating_sub(self.longest)..=range.start)
            .any(|(_, &end)| end >= range.end)
    }
}
struct Map {
    cpu: CpuMapping,
    device: Vec<u8>,
    visible: Vec<bool>,
}
struct Domain {
    peer: Arc<Mutex<Device>>,
    maps: Mutex<BTreeMap<u64, Map>>,
    next: AtomicUsize,
    posted: Mutex<Vec<(u64, Vec<u8>)>>,
    completions: Mutex<Completions>,
    fail_drain: AtomicBool,
    fail_unmap: AtomicBool,
    fail_unmap_id: AtomicUsize,
    fail_sync: AtomicBool,
    incoherent: AtomicBool,
    defer: AtomicBool,
    drains: AtomicUsize,
    drained: AtomicBool,
}
impl Domain {
    fn new(peer: Arc<Mutex<Device>>) -> Self {
        Self {
            peer,
            maps: Mutex::new(BTreeMap::new()),
            next: AtomicUsize::new(0),
            posted: Mutex::new(Vec::new()),
            completions: Mutex::new(Completions::default()),
            fail_drain: AtomicBool::new(false),
            fail_unmap: AtomicBool::new(false),
            fail_unmap_id: AtomicUsize::new(0),
            fail_sync: AtomicBool::new(false),
            incoherent: AtomicBool::new(false),
            defer: AtomicBool::new(false),
            drains: AtomicUsize::new(0),
            drained: AtomicBool::new(false),
        }
    }
    fn visible(&self, at: u64, bytes: usize) -> Vec<u8> {
        let mut maps = self.maps.lock().unwrap();
        let (&base, map) = maps.range_mut(..=at).next_back().unwrap();
        let start = (at - base) as usize;
        assert!(
            start + bytes <= map.device.len(),
            "device read outside IOVA"
        );
        assert!(
            map.visible[start..start + bytes].iter().all(|v| *v),
            "missing sync_for_device"
        );
        map.visible[start..start + bytes].fill(false);
        map.device[start..start + bytes].to_vec()
    }
    fn read32(&self, at: u64) -> u32 {
        u32::from_le_bytes(self.visible(at, 4).try_into().unwrap())
    }
    fn store(&self, at: u64, bytes: &[u8]) {
        let maps = self.maps.lock().unwrap();
        let (&base, map) = maps.range(..=at).next_back().unwrap();
        let address = map.cpu.address((at - base) as usize, bytes.len()).unwrap();
        // SAFETY: the CPU mapping covers these bytes, which `bytes` cannot alias.
        unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), address.as_ptr(), bytes.len()) };
    }
    fn write(&self, at: u64, bytes: &[u8]) {
        if self.defer.load(Ordering::Relaxed) {
            self.posted.lock().unwrap().push((at, bytes.to_vec()));
        } else {
            self.store(at, bytes);
        }
    }
    fn complete(&self, at: u64, bytes: usize) {
        self.completions
            .lock()
            .unwrap()
            .complete(at..at + bytes as u64);
    }
    fn require_stopped(&self) {
        let peer = self.peer.lock().unwrap();
        assert_eq!(peer.config[&4] & 4, 0, "real Command.BM still enabled");
        assert!(
            peer.command_clear_observed,
            "real BM clear readback missing"
        );
        for at in [
            0x468, 0x2c00, 0x2000, 0x2400, 0x2800, 0x3000, 0x1400, 0x1800, 0xc00, 0x4800, 0x1000,
            0x1c00, 0x45c, 0x3c00, 0x4c00,
        ] {
            assert_eq!(
                peer.mac.get(&at).copied().unwrap_or(0) & 2,
                0,
                "live engine {at:x}"
            );
        }
    }
}
// SAFETY: independently authored synchronous symbolic remapper; no native use.
unsafe impl DmaDomain for Domain {
    unsafe fn map(
        &self,
        pages: &DmaPages,
        _: Direction,
        limit: u64,
    ) -> Result<DomainMapping, DmaError> {
        let id = self.next.fetch_add(1, Ordering::Relaxed) + 1;
        let address = 0x2_0000_0000 + id as u64 * 0x100_0000;
        assert!(address + pages.bytes() as u64 <= limit);
        assert_ne!(address, pages.as_single().unwrap().physical_pages()[0]);
        let cpu = pages
            .as_single()
            .unwrap()
            .cpu_mapping(CacheAttribute::WriteBack)
            .unwrap();
        self.maps.lock().unwrap().insert(
            address,
            Map {
                cpu,
                device: vec![0xa5; pages.bytes()],
                visible: vec![false; pages.bytes()],
            },
        );
        Ok(DomainMapping {
            id: id as u64,
            address,
            bytes: pages.bytes(),
        })
    }
    unsafe fn unmap(&self, mapping: DomainMapping) -> Result<(), DmaError> {
        self.require_stopped();
        assert!(self.drained.load(Ordering::Relaxed));
        assert!(self.posted.lock().unwrap().is_empty());
        if self.fail_unmap.load(Ordering::Relaxed)
            || self.fail_unmap_id.load(Ordering::Relaxed) as u64 == mapping.id
        {
            return Err(DmaError::InvalidationFailed);
        }
        assert!(self.maps.lock().unwrap().remove(&mapping.address).is_some());
        Ok(())
    }
    unsafe fn sync_for_device(
        &self,
        mapping: DomainMapping,
        range: Range<usize>,
        _: Direction,
    ) -> Result<(), DmaError> {
        if self.fail_sync.load(Ordering::Relaxed) {
            return Err(DmaError::VisibilityFailed);
        }
        let mut maps = self.maps.lock().unwrap();
        let map = maps.get_mut(&mapping.address).unwrap();
        let start = mapping.address + range.start as u64;
        let end = mapping.address + range.end as u64;
        self.completions.lock().unwrap().revoke(start..end);
        let source = map.cpu.address(range.start, range.len()).unwrap();
        // SAFETY: the CPU mapping covers `range`, and the device copy is a
        // separate allocation of the mapping's length.
        unsafe {
            core::ptr::copy_nonoverlapping(
                source.as_ptr(),
                map.device[range.clone()].as_mut_ptr(),
                range.len(),
            )
        };
        map.visible[range].fill(true);
        Ok(())
    }
    unsafe fn sync_for_cpu(
        &self,
        mapping: DomainMapping,
        range: Range<usize>,
        _: Direction,
    ) -> Result<(), DmaError> {
        if self.drained.load(Ordering::Relaxed) {
            self.require_stopped();
            return Ok(());
        }
        let at = mapping.address + range.start as u64;
        let end = mapping.address + range.end as u64;
        assert!(
            self.completions.lock().unwrap().covers(at..end),
            "CPU access without device completion"
        );
        assert!(
            self.posted
                .lock()
                .unwrap()
                .iter()
                .all(|(p, b)| *p + b.len() as u64 <= at || *p >= end),
            "CPU access before delayed packet write completion"
        );
        Ok(())
    }
    unsafe fn read_coherent_u32(
        &self,
        mapping: DomainMapping,
        offset: usize,
        _: Direction,
    ) -> Result<u32, DmaError> {
        if self.incoherent.load(Ordering::Relaxed) {
            return Err(DmaError::CoherenceUnavailable);
        }
        let maps = self.maps.lock().unwrap();
        let map = maps.get(&mapping.address).unwrap();
        let pointer = map.cpu.address(offset, 4).unwrap();
        // SAFETY: the pointer addresses live memory this test's model owns, in bounds, and nothing
        // accesses it concurrently.
        Ok(u32::from_le(unsafe {
            pointer.as_ptr().cast::<u32>().read_volatile()
        }))
    }
    unsafe fn drain_posted_writes(&self) -> Result<(), DmaError> {
        self.require_stopped();
        if self.fail_drain.load(Ordering::Relaxed) {
            return Err(DmaError::DrainUnavailable);
        }
        for (at, bytes) in self.posted.lock().unwrap().drain(..) {
            self.store(at, &bytes);
        }
        self.drains.fetch_add(1, Ordering::Relaxed);
        self.drained.store(true, Ordering::Relaxed);
        Ok(())
    }
}
#[derive(Clone, Copy)]
enum DmaPath {
    Tx,
    Rx,
    Status,
}
#[derive(Default)]
struct Device {
    config: BTreeMap<u16, u32>,
    bdf_function: Option<u8>,
    clkreq_bug: bool,
    // Legacy fixtures isolate BD/DMA behavior; filter fixtures additionally
    // enable the independent physical MAC/hash matcher below.
    match_receive_filter: bool,
    ignore_config_write: Option<u16>,
    fail_config_write: Option<u16>,
    removed_config_read: Option<u16>,
    mac: BTreeMap<u32, u32>,
    sram: BTreeMap<u32, u32>,
    phy: BTreeMap<u8, u16>,
    phy_writes: Vec<(u8, u16)>,
    ignore_phy_write: Option<(u8, u16)>,
    ape: BTreeMap<u32, u32>,
    nvram: BTreeMap<u32, u32>,
    auxiliary: BTreeMap<u16, u16>,
    aux_select: u16,
    dsp: BTreeMap<u16, u16>,
    dsp_select: u16,
    command: Option<u32>,
    busy: bool,
    tx: usize,
    rx: usize,
    ret: usize,
    sent: Vec<Vec<u8>>,
    access: Vec<(bool, u32, u32)>,
    reset_ack: bool,
    phy_stuck: bool,
    nvram_stuck: bool,
    stop_stuck: Option<u32>,
    hold_tx: bool,
    command_clear_observed: bool,
    drops: usize,
    ignore_master_clear: bool,
    ignore_reserve_fix: bool,
    ignore_write: Option<u32>,
    reset_nvram: Option<u32>,
    fail_read_on: Option<(u32, usize)>,
    read_counts: BTreeMap<u32, usize>,
    fail_write: Option<u32>,
    removed_read: Option<u32>,
    attention_reassert: u32,
    dma_effects: usize,
}
impl Device {
    fn new() -> Self {
        let mut p = Self {
            reset_ack: true,
            ..Self::default()
        };
        for (at, value) in [
            (0, 0x165f_14e4),
            (4, 0x0010_0002),
            (8, 0x0200_0000),
            (0x34, 0x40),
            (0x40, 0x10),
            (0x48, 0x0810),
            (0x68, 0xf000_0000),
            (0xf4, 0x0572_0000),
        ] {
            p.config.insert(at, value);
        }
        // Signed board, copper, NIC, ST serial flash 256-byte page.
        for (at, value) in [
            (0xb54, 0x4b65_7654),
            (0xb58, 0),
            (0xb5c, 0x0001_0000),
            (0xd38, 0),
            (0xe00, 0x362c),
            (0xc14, 0x484b_0211),
            (0xc18, 0x2233_4455),
        ] {
            p.sram.insert(at, value);
        }
        p.mac.insert(0x7014, 0x0200_0000);
        // Independent arbitrary firmware reserve bits must survive the fix.
        p.mac.insert(0x4900, 0x0123_0840);
        let id = 0x5c0d_8b62u32;
        p.phy.insert(2, (id >> 10) as u16);
        p.phy
            .insert(3, (((id >> 16) & 0xfc00) | (id & 0x3ff)) as u16);
        p.phy.insert(1, 0x24);
        p.phy.insert(0x19, 0x700);
        p
    }
    fn phy_read(&self, at: u8) -> u16 {
        match at {
            1 if self.phy.get(&0).copied().unwrap_or(0) & 0x0800 != 0 => {
                self.phy.get(&1).copied().unwrap_or(0) & !4
            }
            0x18 => self.auxiliary.get(&self.aux_select).copied().unwrap_or(0),
            0x15 => self.dsp.get(&self.dsp_select).copied().unwrap_or(0),
            _ => self.phy.get(&at).copied().unwrap_or(0),
        }
    }
    fn phy_write(&mut self, at: u8, value: u16, domain: &Domain) {
        if matches!((at, value), (0x10, 8) | (0x18, 0x1b2) | (0, 0x800)) {
            assert_eq!(self.config[&4] & 4, 0, "PHY power before BME clear");
            assert!(self.command_clear_observed, "PHY power before BME readback");
            for at in [
                0x468, 0x2c00, 0x2000, 0x2400, 0x2800, 0x3000, 0x1400, 0x1800, 0xc00, 0x4800,
                0x1000, 0x1c00, 0x45c, 0x3c00, 0x4c00,
            ] {
                assert_eq!(self.mac.get(&at).copied().unwrap_or(0) & 2, 0);
            }
            assert!(
                domain.drained.load(Ordering::Relaxed),
                "PHY power before DMA drain"
            );
            assert!(domain.posted.lock().unwrap().is_empty());
        }
        self.phy_writes.push((at, value));
        if self.ignore_phy_write == Some((at, value)) {
            return;
        }
        if self.is_5761() && at == 0 && value == 0x800 {
            assert_eq!(
                self.mac[&0x360c] & 0x1f0000,
                0x110000,
                "BCM5761 PDOWN before actual 12.5MHz clock"
            );
        }
        match at {
            0 if value & 0x8000 != 0 => {
                self.phy.insert(0, if self.phy_stuck { value } else { 0 });
                if !self.phy_stuck {
                    // IEEE PHY reset removes PDOWN and restores LED/power
                    // controls; no PCI or packet completion is implied.
                    *self.phy.entry(0x10).or_default() &= !8;
                    *self.auxiliary.entry(2).or_default() &= !0x1b0;
                }
            }
            0x18 if value & 0x8007 == 7 => {
                self.aux_select = (value >> 12) & 7;
            }
            0x18 => {
                self.auxiliary.insert(value & 7, value & !0x8007);
            }
            0x17 => self.dsp_select = value,
            0x15 => {
                self.dsp.insert(self.dsp_select, value);
            }
            _ => {
                self.phy.insert(at, value);
            }
        }
    }
    fn read(&mut self, at: u32, domain: &Domain) -> Result<u32, Error> {
        let count = self.read_counts.entry(at).or_default();
        *count += 1;
        if self.fail_read_on == Some((at, *count)) {
            return Err(Error::Access);
        }
        if self.removed_read == Some(at) {
            return Ok(u32::MAX);
        }
        if self.config[&4] & 2 == 0 {
            return Err(Error::Access);
        }
        let value = match at {
            0x84 => self
                .sram
                .get(&self.mac.get(&0x7c).copied().unwrap_or(0))
                .copied()
                .unwrap_or(0),
            0x44c => {
                if self.busy {
                    self.busy = false;
                    self.command.unwrap()
                } else if let Some(command) = self.command {
                    let at = ((command >> 16) & 31) as u8;
                    let value = if command & 0x0800_0000 != 0 {
                        self.phy_read(at)
                    } else {
                        self.phy_write(at, command as u16, domain);
                        command as u16
                    };
                    self.command = None;
                    let reply = (command & !0x2000_ffff) | u32::from(value);
                    self.mac.insert(0x44c, reply);
                    reply
                } else {
                    self.mac.get(&at).copied().unwrap_or(0)
                }
            }
            _ => self.mac.get(&at).copied().unwrap_or(0),
        };
        self.access.push((false, at, value));
        Ok(value)
    }
    fn status(&self) -> u64 {
        (u64::from(self.mac[&0x3c38]) << 32) | u64::from(self.mac[&0x3c3c])
    }
    // These expectations use the pinned donor's numeric encodings rather
    // than the production constants. Check stored controls for EVERY
    // device effect; an attach phase or caller-supplied ready flag is no proof.
    fn require_bits(&self, at: u32, mask: u32, value: u32) {
        assert_eq!(
            self.mac.get(&at).copied().unwrap_or(0) & mask,
            value,
            "DMA control {at:x}"
        );
    }
    fn require_dma(&self, path: DmaPath) {
        if matches!(path, DmaPath::Tx) && self.clkreq_bug {
            let aux = self.phy[&0x19] & 0x700;
            assert_eq!(
                self.config[&0x50] & 0x100,
                if matches!(aux, 0x100 | 0x200 | 0x300 | 0x500) {
                    0
                } else {
                    0x100
                },
                "actual per-speed CLKREQ must prevent send BD corruption"
            );
        }
        assert_eq!(
            self.phy.get(&0).copied().unwrap_or(0) & 0x800,
            0,
            "PHY suspended"
        );
        assert_eq!(self.config[&4] & 6, 6, "actual memory decode and BME");
        assert_eq!(
            self.config[&0x68] & 0x98,
            0x98,
            "host word-swap/indirect access"
        );
        self.require_bits(0x4000, 3, 2); // MEMARB enabled, not reset.
        self.require_bits(0x6800, 0x34, 0x34); // data/nonframe word and byte swap.
        // Source profile initialization, including the offload mode bits
        // even though the public packet API emits only plain descriptors.
        self.require_bits(
            0x4800,
            0x1800_0000,
            if self.is_5761() {
                0x0800_0000
            } else {
                0x1800_0000
            },
        );
        if self.config[&0] == 0x1687_14e4 {
            self.require_5762_dma();
        } else if self.is_5761() {
            self.require_5761_dma();
        }
        if matches!(path, DmaPath::Tx | DmaPath::Rx) {
            let mode = match self.phy[&0x19] & 0x700 {
                0x100 | 0x300 => 6, // MII with half duplex.
                0x200 | 0x500 => 4, // MII full duplex.
                0x700 => 8,         // GMII full duplex.
                _ => panic!("packet DMA without an admitted negotiated PHY mode"),
            };
            self.require_bits(0x400, 14, mode);
            assert_eq!(self.phy[&4] & 0x1ff, 0x1e1);
            assert_eq!(self.phy[&9] & 0x300, 0x200, "no 1000half advertisement");
        }
        match path {
            DmaPath::Status => {
                self.require_bits(0x3c00, 0x187, if self.is_5761() { 0x102 } else { 0x106 });
                self.require_bits(0x4c00, 3, 2); // host status needs write DMA.
            }
            DmaPath::Tx => {
                self.require_bits(0x4400, 3, 2); // buffer manager.
                self.require_bits(0x400, 0xa0_0000, 0xa0_0000); // TX/frame engines.
                self.require_bits(0x45c, 0x103, 0x102); // TX, A0 MBUF lockup fix.
                self.require_bits(0x6800, 0x20000, 0x20000); // host send BDs.
                self.require_bits(0x4800, 3, 2); // read DMA.
                if self.config[&0] == 0x165f_14e4 {
                    self.require_bits(0x4900, 4, 4); // BCM5720 RDMA FIFO overflow fix.
                    self.require_bits(0x4910, 0xf0000, 0xf0000); // 4K BD/LSO boundaries.
                }
                for at in [0x1400, 0x1800, 0xc00, 0x1000, 0x1c00] {
                    self.require_bits(at, 3, 2);
                }
            }
            DmaPath::Rx => {
                self.require_bits(0x4400, 3, 2);
                self.require_bits(0x400, 0xc0_0000, 0xc0_0000); // RX/frame engines.
                self.require_bits(0x468, 0x403, 0x402); // RX and intact VLAN bytes.
                self.require_bits(0x4c00, 3, 2);
                self.require_bits(0x2400, 0x10003, if self.is_5761() { 2 } else { 0x10002 });
                for at in [0x2c00, 0x2000, 0x2800, 0x3000] {
                    self.require_bits(at, 3, 2);
                }
            }
        }
    }
    fn dma_read32(&mut self, path: DmaPath, domain: &Domain, at: u64) -> u32 {
        self.require_dma(path);
        let value = domain.read32(at);
        self.dma_effects += 1;
        value
    }
    fn dma_read(&mut self, path: DmaPath, domain: &Domain, at: u64, bytes: usize) -> Vec<u8> {
        self.require_dma(path);
        let value = domain.visible(at, bytes);
        self.dma_effects += 1;
        value
    }
    fn dma_write(&mut self, path: DmaPath, domain: &Domain, at: u64, bytes: &[u8]) {
        self.require_dma(path);
        domain.write(at, bytes);
        self.dma_effects += 1;
    }
    fn dma_put32(&mut self, path: DmaPath, domain: &Domain, at: u64, value: u32) {
        self.dma_write(path, domain, at, &value.to_le_bytes());
    }
    fn publish_indices(&mut self, domain: &Domain) {
        let at = self.status() + 16;
        let value = ((self.tx as u32) << 16) | self.ret as u32;
        self.dma_put32(DmaPath::Status, domain, at, value);
    }
    fn publish_attention(&mut self, domain: &Domain) {
        // Independent hardware causes produce the status error indication.
        let error = [0x3c48, 0x4804, 0x4c04]
            .iter()
            .any(|at| self.mac.get(at).copied().unwrap_or(0) != 0)
            || self.mac.get(&0x6004).copied().unwrap_or(0) & !1 != 0;
        self.dma_put32(
            DmaPath::Status,
            domain,
            self.status(),
            if error { 4 } else { 0 },
        );
    }
    fn transmit(&mut self, tail: usize, domain: &Domain) {
        if self.tx == tail || self.hold_tx {
            return;
        }
        let base = (u64::from(self.sram[&0x100]) << 32) | u64::from(self.sram[&0x104]);
        assert_eq!(self.sram[&0x108], 512 << 16);
        for _ in 0..512 {
            if self.tx == tail {
                return;
            }
            let at = base + (self.tx * 16) as u64;
            let high = self.dma_read32(DmaPath::Tx, domain, at);
            let low = self.dma_read32(DmaPath::Tx, domain, at + 4);
            let address = (u64::from(high) << 32) | u64::from(low);
            let command = self.dma_read32(DmaPath::Tx, domain, at + 8);
            assert_eq!(command & 0xffff, 0x84);
            assert_eq!(self.dma_read32(DmaPath::Tx, domain, at + 12), 0);
            let length = (command >> 16) as usize;
            assert!((60..=1514).contains(&length));
            let frame = self.dma_read(DmaPath::Tx, domain, address, length);
            self.sent.push(frame);
            self.require_dma(DmaPath::Tx);
            domain.complete(address, 4096);
            self.tx = (self.tx + 1) & 511;
            self.publish_indices(domain);
        }
        panic!("unbounded TX walk")
    }
    fn receive(&mut self, frame: &[u8], errors: u32, domain: &Domain) {
        if self.match_receive_filter && !self.accepts_destination(frame) {
            return;
        }
        let (rx_count, ret_count, flags) = if self.is_5761() {
            (512, 512, 512 << 16)
        } else {
            (2048, 4096, (2048 << 16) | (1536 << 2))
        };
        assert_eq!(self.mac[&0x2458], flags);
        let tail = self.mac[&0x26c] as usize;
        assert_ne!(self.rx, tail);
        let base = (u64::from(self.mac[&0x2450]) << 32) | u64::from(self.mac[&0x2454]);
        let at = base + (self.rx * 32) as u64;
        let high = self.dma_read32(DmaPath::Rx, domain, at);
        let low = self.dma_read32(DmaPath::Rx, domain, at + 4);
        let address = (u64::from(high) << 32) | u64::from(low);
        assert_eq!(self.dma_read32(DmaPath::Rx, domain, at + 8), 1536);
        assert_eq!(self.dma_read32(DmaPath::Rx, domain, at + 12), 4);
        let opaque = self.dma_read32(DmaPath::Rx, domain, at + 28);
        assert_eq!(opaque, 0x10000 | self.rx as u32);
        self.dma_read(DmaPath::Rx, domain, address, 4096); // offered page.
        let returns = (u64::from(self.sram[&0x200]) << 32) | u64::from(self.sram[&0x204]);
        assert_eq!(self.sram[&0x208], (ret_count as u32) << 16);
        let returned = returns + (self.ret * 32) as u64;
        self.dma_write(DmaPath::Rx, domain, address, frame);
        self.dma_write(
            DmaPath::Rx,
            domain,
            address + frame.len() as u64,
            &[0xde, 0xad, 0xbe, 0xef],
        );
        self.require_dma(DmaPath::Rx);
        domain.complete(address, 4096);
        for (offset, value) in [
            (8, frame.len() as u32 + 4),
            (12, if errors != 0 { 0x404 } else { 4 }),
            (20, errors),
            (28, opaque),
        ] {
            self.dma_put32(DmaPath::Rx, domain, returned + offset, value);
        }
        self.rx = (self.rx + 1) % rx_count;
        self.ret = (self.ret + 1) % ret_count;
        self.publish_indices(domain);
    }
}
struct Reg {
    peer: Arc<Mutex<Device>>,
    domain: Arc<Domain>,
}
impl Drop for Reg {
    fn drop(&mut self) {
        self.peer.lock().unwrap().drops += 1;
    }
}
struct Timer;
// SAFETY: isolated DMA/register peer only; no production timing permission.
unsafe impl Delay for Timer {
    fn check(&self) -> Result<(), crate::bus::pci::power::Error> {
        Ok(())
    }
    fn wait_us(&self, _: u64) -> Result<(), crate::bus::pci::power::Error> {
        Ok(())
    }
}
// SAFETY: isolated symbolic device only, no native permission.
unsafe impl Registers for Reg {
    fn function_number(&self) -> u8 {
        let peer = self.peer.lock().unwrap();
        if let Some(function) = peer.bdf_function {
            function
        } else if peer.is_5761() {
            0
        } else {
            3
        }
    } // Differs from internal port 0.
    fn config_read32(&mut self, at: u16) -> Result<u32, Error> {
        let mut p = self.peer.lock().unwrap();
        let value = if p.removed_config_read == Some(at) {
            u32::MAX
        } else {
            p.config.get(&at).copied().unwrap_or(0)
        };
        if at == 4 && value & 4 == 0 {
            p.command_clear_observed = true;
        }
        p.access.push((false, 0x10000 + u32::from(at), value));
        Ok(value)
    }
    fn config_write32(&mut self, at: u16, value: u32) -> Result<(), Error> {
        let mut p = self.peer.lock().unwrap();
        p.config.insert(at, value);
        p.access.push((true, 0x10000 + u32::from(at), value));
        Ok(())
    }
    fn config_write16(&mut self, at: u16, value: u16) -> Result<(), Error> {
        let mut p = self.peer.lock().unwrap();
        let shift = (at & 2) * 8;
        let word = p.config.get(&(at & !3)).copied().unwrap_or(0);
        if p.fail_config_write == Some(at) || p.ignore_config_write == Some(at) {
            p.access
                .push((true, 0x10000 + u32::from(at), u32::from(value)));
            return if p.fail_config_write == Some(at) {
                Err(Error::Access)
            } else {
                Ok(())
            };
        }
        if at == 4 && word & 4 != 0 && value & 4 == 0 && p.ignore_master_clear {
            return Ok(());
        }
        if at == 4 && value & 4 != 0 {
            self.domain.drained.store(false, Ordering::Relaxed);
        }
        p.config.insert(
            at & !3,
            (word & !(0xffff << shift)) | (u32::from(value) << shift),
        );
        if at == 4 {
            p.command_clear_observed = false;
        }
        p.access
            .push((true, 0x10000 + u32::from(at), u32::from(value)));
        Ok(())
    }
    fn read32(&mut self, at: u32) -> Result<u32, Error> {
        self.peer.lock().unwrap().read(at, &self.domain)
    }
    fn write32(&mut self, at: u32, value: u32) -> Result<(), Error> {
        let mut p = self.peer.lock().unwrap();
        if p.config[&4] & 2 == 0 {
            return Err(Error::Access);
        }
        if p.fail_write == Some(at) {
            return Err(Error::Access);
        }
        p.access.push((true, at, value));
        if p.ignore_write == Some(at) {
            return Ok(());
        }
        if p.is_5761() && at == 0x360c && value & 0x1f0000 == 0x110000 {
            assert_eq!(p.config[&4] & 4, 0, "5761 power clock before BME clear");
            assert!(p.command_clear_observed);
            assert!(self.domain.drained.load(Ordering::Relaxed));
            assert!(self.domain.posted.lock().unwrap().is_empty());
            assert_eq!(p.phy.get(&0).copied().unwrap_or(0) & 0x800, 0);
            assert_eq!(p.phy[&0x10] & 8, 8, "5761 clock before LED low power");
            assert_eq!(
                p.auxiliary[&2] & 0x1b0,
                0x1b0,
                "5761 clock before AUX low power"
            );
        }
        match at {
            0x84 => {
                let at = p.mac.get(&0x7c).copied().unwrap_or(0);
                p.sram.insert(at, value);
            }
            0x404 => {
                let old = p.mac.get(&at).copied().unwrap_or(0);
                p.mac.insert(at, old & !value);
            }
            0x3c48 => {
                let old = p.mac.get(&at).copied().unwrap_or(0);
                let reassert = p.attention_reassert;
                p.mac.insert(at, (old & !value) | reassert);
            }
            0x4900 if p.ignore_reserve_fix => {}
            0x3c00 if value & 8 != 0 => {
                // COALESCE_NOW is a hardware request, not a sticky enable.
                // Complete it synchronously in this model; later stop RMWs
                // must not issue a fictitious update after disabling HOSTCC.
                p.mac.insert(at, value & !8);
                p.publish_attention(&self.domain);
            }
            0x44c => {
                let expected_phy = if p.config[&0] == 0x1687_14e4 {
                    1
                } else {
                    ((p.sram[&0xe00] >> 30) + 1) & 31
                };
                assert_eq!(
                    (value >> 21) & 31,
                    expected_phy,
                    "MDIO must address the actual internal port"
                );
                p.command = Some(value);
                p.busy = true;
            }
            0x7020 if value == 2 => {
                p.mac.insert(at, 0x202);
            }
            0x7000 => {
                let address = p.mac.get(&0x700c).copied().unwrap_or(0);
                let data = p.nvram.get(&address).copied().unwrap_or(u32::MAX);
                p.mac.insert(0x7010, data);
                let stuck = p.nvram_stuck;
                p.mac.insert(at, if stuck { value & !8 } else { value | 8 });
            }
            0x6804 if value & 1 != 0 => {
                if p.reset_ack {
                    p.sram.insert(0xb50, !0x4b65_7654u32);
                }
                if let Some(config) = p.reset_nvram {
                    p.mac.insert(0x7014, config);
                }
                p.tx = 0;
                p.rx = 0;
                p.ret = 0;
                // No MMIO operation is allowed until config decode restored.
                let command = p.config[&4];
                p.config.insert(4, command & !2);
            }
            0x304 => {
                p.mac.insert(at, value);
                p.transmit(value as usize, &self.domain);
            }
            _ if p.stop_stuck == Some(at) && value & 2 == 0 => {}
            _ => {
                p.mac.insert(at, value);
            }
        }
        Ok(())
    }
    fn ape_read32(&mut self, at: u32) -> Result<u32, Error> {
        Ok(self.peer.lock().unwrap().ape.get(&at).copied().unwrap_or(0))
    }
    fn ape_write32(&mut self, at: u32, value: u32) -> Result<(), Error> {
        let mut p = self.peer.lock().unwrap();
        if (0x8400..0x8420).contains(&at) || (0x2c..0x4c).contains(&at) {
            p.ape.insert(at + 0x20, value);
        } else {
            p.ape.insert(at, 0);
        }
        Ok(())
    }
}
fn parts() -> (Hardware<Reg, Timer>, Pages, Arc<Domain>, Arc<Mutex<Device>>) {
    let peer = Arc::new(Mutex::new(Device::new()));
    let domain = Arc::new(Domain::new(peer.clone()));
    let pages = Pages {
        next: Cell::new(0x3000_0000),
        drops: Arc::new(AtomicUsize::new(0)),
        allocated: Cell::new(0),
        fail_after: None,
    };
    (
        // SAFETY: original register/domain/timer owners of this isolated peer.
        unsafe {
            Hardware::new(
                Reg {
                    peer: peer.clone(),
                    domain: domain.clone(),
                },
                Timer,
            )
        },
        pages,
        domain,
        peer,
    )
}
fn attach() -> (Packet<Reg, Timer>, Pages, Arc<Domain>, Arc<Mutex<Device>>) {
    let (hardware, pages, domain, peer) = parts();
    // SAFETY: the model device, its pages and its domain belong to this test alone; no other
    // driver, worker or DMA touches them.
    let packet = match unsafe { Packet::attach(hardware, &pages, domain.clone()) } {
        Ok(packet) => packet,
        Err((e, _)) => panic!("attach: {e:?}"),
    };
    (packet, pages, domain, peer)
}

#[test]
fn nonidentity_rings_transfer_frames_pad_strip_fcs_and_retire() {
    if crate::test_process::isolate_model() {
        return;
    }
    let (mut packet, pages, domain, peer) = attach();
    assert_eq!(packet.mac(), Mac([2, 0x11, 0x22, 0x33, 0x44, 0x55]));
    assert!(packet.carrier());
    let short = vec![0x53; 14];
    packet.transmit(&short).unwrap();
    assert_eq!(
        peer.lock().unwrap().sent[0],
        [&short[..], &[0; 46]].concat()
    );
    let frame = vec![0x86; 1514];
    peer.lock().unwrap().receive(&frame, 0, &domain);
    assert_eq!(packet.receive().unwrap(), Some(frame));
    assert_eq!(packet.receive().unwrap(), None);
    let registers = match packet.retire() {
        Ok(r) => r,
        Err((e, _)) => panic!("retire {e:?}"),
    };
    assert!(domain.maps.lock().unwrap().is_empty());
    assert_eq!(pages.drops.load(Ordering::Relaxed), 6);
    drop(registers);
    assert_eq!(peer.lock().unwrap().drops, 1);
}
#[test]
fn wraps_tx_and_rx_producer_return_indices() {
    if crate::test_process::isolate_model() {
        return;
    }
    let (mut packet, _, domain, peer) = attach();
    // One frame past the 512-entry send ring, then past the 2048-entry
    // standard and 4096-entry return rings.
    let mut reused = Vec::new();
    for value in 0..4097u32 {
        let mut frame = vec![0x2a; 60];
        frame[14..18].copy_from_slice(&value.to_le_bytes());
        if value <= 512 {
            packet.transmit(&frame).unwrap();
        }
        peer.lock().unwrap().receive(&frame, 0, &domain);
        if value % 2 == 0 {
            assert_eq!(packet.receive().unwrap(), Some(frame));
        } else {
            assert_eq!(packet.receive_reusing(&mut reused), Ok(true));
            assert_eq!(reused, frame);
        }
    }
    assert_eq!(peer.lock().unwrap().sent.len(), 513);
    assert_eq!(peer.lock().unwrap().tx, 1);
    assert_eq!(peer.lock().unwrap().ret, 1);
    // The stack's path trades buffers with the driver: two allocations
    // serve every frame once both have grown to the largest one.
    let mut storage = Vec::new();
    for value in 0..8u8 {
        let frame = vec![value; 1514 - usize::from(value)];
        peer.lock().unwrap().receive(&frame, 0, &domain);
        assert_eq!(packet.receive_reusing(&mut reused), Ok(true));
        assert_eq!(reused, frame);
        if value >= 2 && !storage.contains(&reused.as_ptr()) {
            storage.push(reused.as_ptr());
        }
    }
    assert_eq!(storage.len(), 2);
    assert_eq!(packet.receive_reusing(&mut reused), Ok(false));
    assert_eq!(reused, vec![7; 1507]);
    assert!(packet.retire().is_ok());
}
#[test]
fn only_real_completed_tx_slots_can_be_reused_and_full_is_recoverable() {
    if crate::test_process::isolate_model() {
        return;
    }
    let (mut packet, _, domain, peer) = attach();
    peer.lock().unwrap().hold_tx = true;
    for _ in 0..511 {
        packet.transmit(&[0xa5; 60]).unwrap();
    }
    assert!(packet.transmit(&[0xb6; 60]).is_err());
    assert_eq!(packet.failure(), None);
    {
        let mut p = peer.lock().unwrap();
        p.hold_tx = false;
        p.transmit(511, &domain);
    }
    packet.transmit(&[0xb6; 60]).unwrap();
    assert_eq!(peer.lock().unwrap().sent.len(), 512);
    assert!(packet.retire().is_ok());
}
#[test]
fn carrier_comes_from_phy_status_and_negotiated_mode() {
    if crate::test_process::isolate_model() {
        return;
    }
    let (mut packet, _, _, peer) = attach();
    peer.lock().unwrap().phy.insert(1, 0);
    assert!(!packet.carrier());
    assert_eq!(packet.transmit(&[0; 60]), Err(LinkError::NoCarrier));
    for (aux, port) in [(0x200, 4), (0x500, 4), (0x700, 8)] {
        let mut p = peer.lock().unwrap();
        p.phy.insert(1, 0x24);
        p.phy.insert(0x19, aux);
        drop(p);
        assert!(packet.carrier());
        assert_eq!(peer.lock().unwrap().mac[&0x400] & 12, port);
    }
    peer.lock().unwrap().phy.insert(0x19, 0x600);
    assert!(!packet.carrier());
    assert!(packet.retire().is_ok());
}
#[test]
fn corrupted_rx_is_dropped_and_invalid_opaque_poison_keeps_owners() {
    if crate::test_process::isolate_model() {
        return;
    }
    let (mut packet, pages, domain, peer) = attach();
    peer.lock().unwrap().receive(&[0x12; 60], 0x10000, &domain);
    assert_eq!(packet.receive().unwrap(), None);
    peer.lock().unwrap().receive(&[0x34; 60], 0, &domain);
    let returns = {
        let p = peer.lock().unwrap();
        (u64::from(p.sram[&0x200]) << 32) | u64::from(p.sram[&0x204])
    };
    domain.store(returns + 32 + 28, &0x20001u32.to_le_bytes());
    let mut reused = vec![9; 60];
    assert!(packet.receive_reusing(&mut reused).is_err());
    assert_eq!(reused, vec![9; 60]);
    assert_eq!(packet.failure(), Some(Error::RingCorrupt));
    drop(packet);
    assert_eq!(pages.drops.load(Ordering::Relaxed), 0);
    assert_eq!(peer.lock().unwrap().drops, 0);
    assert_eq!(domain.maps.lock().unwrap().len(), 6);
}
#[test]
fn late_payload_writes_drain_before_unmap_and_failure_keeps_exact_epoch() {
    if crate::test_process::isolate_model() {
        return;
    }
    let (packet, pages, domain, peer) = attach();
    domain.defer.store(true, Ordering::Relaxed);
    peer.lock().unwrap().receive(&[0x7b; 60], 0, &domain);
    assert!(!domain.posted.lock().unwrap().is_empty());
    domain.fail_drain.store(true, Ordering::Relaxed);
    let mut packet = match packet.retire() {
        Err((Error::Dma(DmaError::DrainUnavailable), p)) => p,
        _ => panic!("drain must refuse"),
    };
    assert_eq!(pages.drops.load(Ordering::Relaxed), 0);
    assert_eq!(domain.maps.lock().unwrap().len(), 6);
    assert!(packet.transmit(&[0; 60]).is_err());
    domain.fail_drain.store(false, Ordering::Relaxed);
    domain.fail_unmap.store(true, Ordering::Relaxed);
    let packet = match packet.retire() {
        Err((Error::Dma(DmaError::InvalidationFailed), p)) => p,
        _ => panic!("unmap must refuse"),
    };
    assert!(domain.posted.lock().unwrap().is_empty());
    assert_eq!(pages.drops.load(Ordering::Relaxed), 0);
    domain.fail_unmap.store(false, Ordering::Relaxed);
    assert!(packet.retire().is_ok());
    assert_eq!(pages.drops.load(Ordering::Relaxed), 6);
}
#[test]
fn a_refused_retirement_never_drains_or_unmaps() {
    if crate::test_process::isolate_model() {
        return;
    }
    // A live engine that does not stop, and a Command.BM clear the device
    // ignores: neither authorizes a drain or an unmap until it is cured.
    let stuck: fn(&mut Device, bool) = |peer, on| peer.stop_stuck = on.then_some(0x4800);
    let ignored: fn(&mut Device, bool) = |peer, on| peer.ignore_master_clear = on;
    for (fault, expected) in [
        (stuck, Error::StopTimeout(0x4800)),
        (ignored, Error::ConfigDidNotLatch),
    ] {
        let (packet, pages, domain, peer) = attach();
        fault(&mut peer.lock().unwrap(), true);
        let packet = match packet.retire() {
            Err((error, packet)) if error == expected => packet,
            _ => panic!("retirement must refuse with {expected:?}"),
        };
        assert_eq!(domain.drains.load(Ordering::Relaxed), 0);
        assert_eq!(pages.drops.load(Ordering::Relaxed), 0);
        fault(&mut peer.lock().unwrap(), false);
        assert!(packet.retire().is_ok());
    }
}

#[test]
fn reset_phy_nvm_and_coherence_failures_never_advertise_attachment() {
    if crate::test_process::isolate_model() {
        return;
    }
    for fault in 0..4 {
        let (hardware, pages, domain, peer) = parts();
        {
            let mut p = peer.lock().unwrap();
            match fault {
                0 => p.reset_ack = false,
                1 => p.phy_stuck = true,
                2 => {
                    p.sram.insert(0xc14, 0);
                    p.nvram_stuck = true;
                }
                _ => domain.incoherent.store(true, Ordering::Relaxed),
            }
        }
        // SAFETY: the model device, its pages and its domain belong to this test alone; no other
        // driver, worker or DMA touches them.
        let (error, mut packet) = match unsafe { Packet::attach(hardware, &pages, domain.clone()) }
        {
            Err(failed) => failed,
            Ok(_) => panic!("fault admitted"),
        };
        assert_eq!(
            error,
            match fault {
                0 => Error::FirmwareTimeout,
                1 => Error::PhyResetTimeout,
                2 => Error::NvramTimeout,
                _ => Error::Dma(DmaError::CoherenceUnavailable),
            }
        );
        assert!(!packet.carrier());
        assert!(packet.transmit(&[0; 60]).is_err());
        drop(packet);
        assert_eq!(peer.lock().unwrap().drops, 0);
    }
}
#[test]
fn nvm_fallback_uses_internal_port_and_exact_read_protocol() {
    if crate::test_process::isolate_model() {
        return;
    }
    let (hardware, pages, domain, peer) = parts();
    {
        let mut p = peer.lock().unwrap();
        p.sram.insert(0xc14, 0);
        p.sram.insert(0xe00, 0x8000_362c);
        // Internal port 2: MAC location 0x7c+0x18c, not BDF function 3.
        p.nvram.insert(0x208, 0x0212);
        p.nvram.insert(0x20c, 0x3456_789a);
    }
    // SAFETY: the model device, its pages and its domain belong to this test alone; no other
    // driver, worker or DMA touches them.
    let packet = match unsafe { Packet::attach(hardware, &pages, domain) } {
        Ok(p) => p,
        Err((e, _)) => panic!("{e:?}"),
    };
    assert_eq!(packet.mac(), Mac([2, 0x12, 0x34, 0x56, 0x78, 0x9a]));
    let p = peer.lock().unwrap();
    assert!(p.access.contains(&(true, 0x7000, 0x198)));
    assert!(p.access.contains(&(true, 0x700c, 0x208)));
    drop(p);
    assert!(packet.retire().is_ok());
}
#[test]
fn live_status_with_future_completion_cannot_retire_unsubmitted_buffer() {
    if crate::test_process::isolate_model() {
        return;
    }
    let (mut packet, _, domain, peer) = attach();
    let status = peer.lock().unwrap().status();
    domain.store(status + 16, &0x0001_0000u32.to_le_bytes());
    assert!(packet.receive().is_err());
    assert_eq!(packet.failure(), Some(Error::RingCorrupt));
    assert!(packet.retire().is_ok());
}

#[test]
fn link_loss_discards_old_tx_only_after_real_drain_and_rebuilds_rings() {
    if crate::test_process::isolate_model() {
        return;
    }
    let (mut packet, _, domain, peer) = attach();
    assert!(packet.carrier());
    peer.lock().unwrap().hold_tx = true;
    packet.transmit(&[0x11; 60]).unwrap();
    peer.lock().unwrap().phy.insert(1, 0);
    assert!(!packet.carrier());
    assert_eq!(domain.drains.load(Ordering::Relaxed), 1);
    assert_eq!(packet.tx_producer, 0);
    {
        let mut p = peer.lock().unwrap();
        p.phy.insert(1, 0x24);
        p.hold_tx = false;
    }
    packet.transmit(&[0x22; 60]).unwrap();
    assert_eq!(peer.lock().unwrap().sent, vec![vec![0x22; 60]]);
    peer.lock().unwrap().receive(&[0x33; 60], 0, &domain);
    assert_eq!(packet.receive().unwrap(), Some(vec![0x33; 60]));
    assert!(packet.retire().is_ok());
}
#[test]
fn visibility_failure_after_attachment_poison_keeps_for_real_retirement() {
    if crate::test_process::isolate_model() {
        return;
    }
    let (mut packet, pages, domain, _) = attach();
    domain.fail_sync.store(true, Ordering::Relaxed);
    assert!(packet.transmit(&[0x5a; 60]).is_err());
    assert_eq!(
        packet.failure(),
        Some(Error::Dma(DmaError::VisibilityFailed))
    );
    assert_eq!(packet.tx_producer, 0);
    assert_eq!(pages.drops.load(Ordering::Relaxed), 0);
    domain.fail_sync.store(false, Ordering::Relaxed);
    assert!(packet.retire().is_ok());
}
#[test]
fn peer_reads_published_device_snapshot_instead_of_unsynchronized_cpu_bytes() {
    if crate::test_process::isolate_model() {
        return;
    }
    let (_hardware, pages, domain, _) = parts();
    let pins = pages
        .allocate(4096, 4096, CacheAttribute::WriteBack)
        .unwrap();
    let cpu = pins.cpu_mapping(CacheAttribute::WriteBack).unwrap();
    let mapping = crate::bus::dma::mapping::DmaMapping::map(
        domain.clone(),
        pins,
        Direction::Bidirectional,
        u64::MAX,
    )
    .unwrap();
    let pointer = cpu.address(0, 4).unwrap();
    // SAFETY: the pointer addresses this test's own live pages, and the model writes nothing
    // concurrently.
    unsafe {
        pointer
            .as_ptr()
            .cast::<u32>()
            .write_volatile(0x11223344u32.to_le());
    }
    mapping.sync_for_device(0..4).unwrap();
    // SAFETY: the pointer addresses this test's own live pages, and the model writes nothing
    // concurrently.
    unsafe {
        pointer
            .as_ptr()
            .cast::<u32>()
            .write_volatile(0xaabbccddu32.to_le());
    }
    assert_eq!(domain.read32(mapping.address().unwrap()), 0x11223344);
    // This unpublished standalone model allocation is quarantined
    // on drop; it never impersonates a native completed mapping.
}
#[test]
fn peer_refuses_cpu_ownership_before_delayed_dma_payload_completion() {
    if crate::test_process::isolate_model() {
        return;
    }
    let (_packet, _, domain, peer) = attach();
    domain.defer.store(true, Ordering::Relaxed);
    peer.lock().unwrap().receive(&[0x43; 60], 0, &domain);
    let (mapping, bytes) = {
        let maps = domain.maps.lock().unwrap();
        let (&address, map) = maps.iter().last().unwrap();
        (
            DomainMapping {
                id: 6,
                address,
                bytes: map.device.len(),
            },
            map.device.len(),
        )
    };
    let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // SAFETY: intentional negative model control, no native use. There is
        // no completion visibility yet and the peer must reject this call.
        let _ =
            unsafe { domain.sync_for_cpu(mapping, 0..bytes.min(4096), Direction::Bidirectional) };
    }));
    assert!(refused.is_err());
}
#[test]
fn peer_rejects_drain_without_engine_stop_and_bus_master_clear() {
    if crate::test_process::isolate_model() {
        return;
    }
    let (_packet, _, domain, _) = attach();
    let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // SAFETY: intentional rejected symbolic negative control only.
        let _ = unsafe { domain.drain_posted_writes() };
    }));
    assert!(refused.is_err());
}

#[test]
fn partial_page_allocation_keeps_every_mapping_handle_for_explicit_retirement() {
    if crate::test_process::isolate_model() {
        return;
    }
    for allocated in 0..6 {
        let (hardware, mut pages, domain, peer) = parts();
        pages.fail_after = Some(allocated);
        // SAFETY: the model device, its pages and its domain belong to this test alone; no other
        // driver, worker or DMA touches them.
        let packet = match unsafe { Packet::attach(hardware, &pages, domain.clone()) } {
            Err((Error::Memory(MemoryError::OutOfMemory), packet)) => packet,
            _ => panic!("allocation failure must be returned"),
        };
        assert_eq!(domain.maps.lock().unwrap().len(), allocated);
        assert_eq!(pages.drops.load(Ordering::Relaxed), 0);
        let registers = match packet.retire() {
            Ok(r) => r,
            Err((e, _)) => panic!("{e:?}"),
        };
        assert_eq!(pages.drops.load(Ordering::Relaxed), allocated);
        assert!(domain.maps.lock().unwrap().is_empty());
        drop(registers);
        assert_eq!(peer.lock().unwrap().drops, 1);
    }
}

#[test]
fn rdma_fifo_fix_preserves_reserve_bits_and_is_observed_before_enable() {
    if crate::test_process::isolate_model() {
        return;
    }
    let (packet, _, _, peer) = attach();
    let p = peer.lock().unwrap();
    assert_eq!(p.mac[&0x4900], 0x0123_0844);
    let write = p
        .access
        .iter()
        .position(|a| *a == (true, 0x4900, 0x0123_0844))
        .unwrap();
    let readback = p
        .access
        .iter()
        .position(|a| *a == (false, 0x4900, 0x0123_0844))
        .unwrap();
    let enable = p
        .access
        .iter()
        .position(|a| a.0 && a.1 == 0x4800 && a.2 & 2 != 0)
        .unwrap();
    assert!(write < readback && readback < enable);
    drop(p);
    assert!(packet.retire().is_ok());
}
#[test]
fn rdma_fix_access_or_readback_failure_never_enables_read_dma() {
    if crate::test_process::isolate_model() {
        return;
    }
    for fault in 0..4 {
        let (hardware, pages, domain, peer) = parts();
        {
            let mut p = peer.lock().unwrap();
            match fault {
                0 => p.fail_read_on = Some((0x4900, 1)),
                1 => p.fail_write = Some(0x4900),
                2 => p.fail_read_on = Some((0x4900, 2)),
                _ => p.ignore_reserve_fix = true,
            }
        }
        // SAFETY: the model device, its pages and its domain belong to this test alone; no other
        // driver, worker or DMA touches them.
        let packet = match unsafe { Packet::attach(hardware, &pages, domain.clone()) } {
            Err((error, packet)) => {
                assert_eq!(
                    error,
                    if fault == 3 {
                        Error::ConfigDidNotLatch
                    } else {
                        Error::Access
                    }
                );
                packet
            }
            Ok(_) => panic!("missing RDMA workaround admitted"),
        };
        let mut p = peer.lock().unwrap();
        assert!(
            !p.access
                .iter()
                .any(|a| a.0 && a.1 == 0x4800 && a.2 & 2 != 0)
        );
        assert_eq!(p.dma_effects, 0);
        assert_eq!(p.drops, 0);
        assert_eq!(pages.drops.load(Ordering::Relaxed), 0);
        assert_eq!(domain.maps.lock().unwrap().len(), 6);
        p.fail_read_on = None;
        p.fail_write = None;
        drop(p);
        assert!(packet.retire().is_ok());
    }
}
#[test]
fn a0_mbuf_low_water_is_counted_acknowledged_and_packet_io_continues() {
    if crate::test_process::isolate_model() {
        return;
    }
    let (mut packet, _, domain, peer) = attach();
    peer.lock().unwrap().mac.insert(0x3c48, 0x40);
    assert_eq!(packet.receive().unwrap(), None);
    assert_eq!(packet.mbuf_low_water_events(), 1);
    assert_eq!(packet.failure(), None);
    let p = peer.lock().unwrap();
    assert_eq!(p.mac[&0x3c48], 0);
    let ack = p
        .access
        .iter()
        .position(|a| *a == (true, 0x3c48, 0x40))
        .unwrap();
    for at in [0x3c48, 0x6004, 0x4804, 0x4c04] {
        assert!(p.access[..ack].iter().any(|a| !a.0 && a.1 == at));
    }
    assert_eq!(p.access[ack + 1], (false, 0x3c48, 0));
    drop(p);
    packet.transmit(&[0x37; 60]).unwrap();
    peer.lock().unwrap().receive(&[0x73; 60], 0, &domain);
    assert_eq!(packet.receive().unwrap(), Some(vec![0x73; 60]));
    assert_eq!(packet.mbuf_low_water_events(), 1);
    assert!(packet.retire().is_ok());
}
#[test]
fn reasserted_low_water_remains_visible_and_counter_never_wraps() {
    if crate::test_process::isolate_model() {
        return;
    }
    let (mut packet, _, _, peer) = attach();
    {
        let mut p = peer.lock().unwrap();
        p.mac.insert(0x3c48, 0x40);
        p.attention_reassert = 0x40;
    }
    packet.mbuf_low_water_events = u64::MAX - 1;
    for _ in 0..2 {
        assert_eq!(packet.receive().unwrap(), None);
        assert_eq!(packet.mbuf_low_water_events(), u64::MAX);
        assert_eq!(peer.lock().unwrap().mac[&0x3c48], 0x40);
    }
    peer.lock().unwrap().attention_reassert = 0;
    assert_eq!(packet.receive().unwrap(), None);
    assert_eq!(peer.lock().unwrap().mac[&0x3c48], 0);
    assert_eq!(packet.failure(), None);
    assert!(packet.retire().is_ok());
}
#[test]
fn every_fatal_flow_message_and_dma_cause_remains_an_explicit_kept_fault() {
    if crate::test_process::isolate_model() {
        return;
    }
    // Preserve every original single-bit cause: four registers by 32 bits,
    // excluding only the flow low-water bit and message MSI request bit.
    // Also keep the combined flow mask to exercise simultaneous causes;
    // all-ones reads are independently classified as device removal.
    let single_bits = [0x3c48, 0x6004, 0x4804, 0x4c04]
        .into_iter()
        .flat_map(|at| (0..32).map(move |bit| (at, 1u32 << bit)))
        .filter(|&(at, value)| !((at == 0x3c48 && value == 0x40) || (at == 0x6004 && value == 1)));
    for (at, value) in single_bits.chain([(0x3c48, 0x7fff_ffbf)]) {
        {
            let (mut packet, pages, domain, peer) = attach();
            {
                let mut p = peer.lock().unwrap();
                p.mac.insert(0x3c48, 0x40);
                p.mac
                    .insert(at, value | if at == 0x3c48 { 0x40 } else { 0 });
            }
            assert!(packet.receive().is_err());
            assert_eq!(
                packet.failure(),
                Some(Error::Attention {
                    flow: if at == 0x3c48 { value | 0x40 } else { 0x40 },
                    message: if at == 0x6004 { value } else { 0 },
                    read_dma: if at == 0x4804 { value } else { 0 },
                    write_dma: if at == 0x4c04 { value } else { 0 },
                })
            );
            assert_eq!(packet.mbuf_low_water_events(), 0);
            assert!(
                !peer
                    .lock()
                    .unwrap()
                    .access
                    .iter()
                    .any(|a| a.0 && a.1 == 0x3c48)
            );
            assert_eq!(pages.drops.load(Ordering::Relaxed), 0);
            assert_eq!(domain.maps.lock().unwrap().len(), 6);
            assert!(packet.transmit(&[0; 60]).is_err());
            assert!(packet.retire().is_ok());
        }
    }
}
#[test]
fn stale_status_and_msi_request_alone_are_not_fatal_attention_causes() {
    if crate::test_process::isolate_model() {
        return;
    }
    let (mut packet, _, _, peer) = attach();
    assert_eq!(packet.hardware.packet_attention(), Ok(false));
    peer.lock().unwrap().mac.insert(0x6004, 1);
    assert_eq!(packet.hardware.packet_attention(), Ok(false));
    assert_eq!(packet.failure(), None);
    assert!(packet.retire().is_ok());
}
#[test]
fn attention_ack_failures_keep_owners() {
    if crate::test_process::isolate_model() {
        return;
    }
    // Cause reads, targeted W1C write, and ordering read after that write.
    for fault in 0..6 {
        let (mut packet, pages, domain, peer) = attach();
        {
            let mut p = peer.lock().unwrap();
            p.mac.insert(0x3c48, 0x40);
            if fault == 4 {
                p.fail_write = Some(0x3c48);
            } else {
                let at = [0x3c48, 0x6004, 0x4804, 0x4c04, 0, 0x3c48][fault];
                let count = p.read_counts.get(&at).copied().unwrap_or(0);
                p.fail_read_on = Some((at, count + if fault == 5 { 2 } else { 1 }));
            }
        }
        assert!(packet.receive().is_err());
        assert_eq!(packet.failure(), Some(Error::Access));
        assert_eq!(pages.drops.load(Ordering::Relaxed), 0);
        assert_eq!(domain.maps.lock().unwrap().len(), 6);
        let mut p = peer.lock().unwrap();
        p.fail_write = None;
        p.fail_read_on = None;
        drop(p);
        assert!(packet.retire().is_ok());
    }
}

fn rejected_dma_effect(
    packet: &Packet<Reg, Timer>,
    domain: &Domain,
    p: &mut Device,
    path: DmaPath,
) {
    let before = p.dma_effects;
    let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match path {
        DmaPath::Tx => p.transmit(1, domain),
        DmaPath::Rx => p.receive(&[0x19; 60], 0, domain),
        DmaPath::Status => p.publish_attention(domain),
    }));
    assert!(refused.is_err(), "missing device control admitted DMA");
    assert_eq!(
        p.dma_effects, before,
        "DMA effect before prerequisite refusal"
    );
    assert_eq!(p.tx, packet.tx_consumer);
    assert_eq!(p.rx, 0);
}
#[test]
fn peer_refuses_each_deleted_or_disabled_mac_dma_engine_and_workaround() {
    if crate::test_process::isolate_model() {
        return;
    }
    let common = [(0x4000, 2), (0x6800, 4), (0x6800, 0x10), (0x6800, 0x20)];
    let tx = [
        (0x4400, 2),
        (0x400, 0x20_0000),
        (0x400, 0x80_0000),
        (0x45c, 2),
        (0x45c, 0x100),
        (0x6800, 0x20000),
        (0x4800, 2),
        (0x4900, 4),
        (0x4910, 0x10000),
        (0x4910, 0x20000),
        (0x4910, 0x40000),
        (0x4910, 0x80000),
        (0x1400, 2),
        (0x1800, 2),
        (0xc00, 2),
        (0x1000, 2),
        (0x1c00, 2),
    ];
    let rx = [
        (0x4400, 2),
        (0x400, 0x40_0000),
        (0x400, 0x80_0000),
        (0x468, 2),
        (0x468, 0x400),
        (0x4c00, 2),
        (0x2400, 2),
        (0x2400, 0x10000),
        (0x2c00, 2),
        (0x2000, 2),
        (0x2800, 2),
        (0x3000, 2),
    ];
    let status = [(0x3c00, 2), (0x3c00, 4), (0x3c00, 0x100), (0x4c00, 2)];
    for (path, controls) in [
        (DmaPath::Tx, &tx[..]),
        (DmaPath::Rx, &rx[..]),
        (DmaPath::Status, &status[..]),
    ] {
        // A refusal leaves no DMA effect, so one attachment serves every
        // control once each is restored.
        let (packet, _, domain, peer) = attach();
        packet.memory.as_ref().unwrap().tx(0, &[0x91; 60]).unwrap();
        for &(at, bit) in common.iter().chain(controls) {
            for delete in [false, true] {
                // Guard is acquired outside catch_unwind, so the deliberate
                // model assertion cannot poison the mutex or hide cleanup.
                let mut p = peer.lock().unwrap();
                let original = p.mac[&at];
                if delete {
                    p.mac.remove(&at);
                } else {
                    p.mac.insert(at, original & !bit);
                }
                rejected_dma_effect(&packet, &domain, &mut p, path);
                p.mac.insert(at, original);
            }
        }
        assert!(packet.retire().is_ok());
    }
}
#[test]
fn peer_refuses_dma_without_pci_and_host_access_controls() {
    if crate::test_process::isolate_model() {
        return;
    }
    for path in [DmaPath::Tx, DmaPath::Rx, DmaPath::Status] {
        let (packet, _, domain, peer) = attach();
        packet.memory.as_ref().unwrap().tx(0, &[0x92; 60]).unwrap();
        for (at, bit) in [(4, 2), (4, 4), (0x68, 8), (0x68, 0x10), (0x68, 0x80)] {
            let mut p = peer.lock().unwrap();
            let original = p.config[&at];
            p.config.insert(at, original & !bit);
            rejected_dma_effect(&packet, &domain, &mut p, path);
            p.config.insert(at, original);
        }
        assert!(packet.retire().is_ok());
    }
}

#[test]
fn fatal_flow_observed_during_low_water_ack_is_never_discarded() {
    if crate::test_process::isolate_model() {
        return;
    }
    let (mut packet, pages, domain, peer) = attach();
    {
        let mut p = peer.lock().unwrap();
        p.mac.insert(0x3c48, 0x40);
        p.attention_reassert = 0x80;
    }
    assert!(packet.receive().is_err());
    assert_eq!(
        packet.failure(),
        Some(Error::Attention {
            flow: 0x80,
            message: 0,
            read_dma: 0,
            write_dma: 0,
        })
    );
    assert_eq!(peer.lock().unwrap().mac[&0x3c48], 0x80);
    assert_eq!(pages.drops.load(Ordering::Relaxed), 0);
    assert_eq!(domain.maps.lock().unwrap().len(), 6);
    assert!(packet.retire().is_ok());
}

mod bcm5761;
mod bcm5761_power;
mod bcm5762;
mod phy_link;
mod phy_power;
mod rdma_mode;
mod restart;

mod timing;

mod clkreq;

mod receive_filter;

mod attachment;
