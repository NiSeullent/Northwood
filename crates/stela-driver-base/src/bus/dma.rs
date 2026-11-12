//! Device-visible memory with distinct CPU mappings and bus addresses.
use crate::block::BlockError;
use alloc::vec::Vec;
use core::ops::Range;

mod allocator;
pub mod amd;
pub use allocator::{AllocationError, DomainAllocator};
pub mod mapping;
pub mod memory;
pub mod pages;
pub mod region;

mod sealed {
    pub trait DmaValue {}
}

/// Integer words whose every bit pattern is valid when written by a device.
/// Sealed so safe reads cannot manufacture a reference, `bool` or Rust enum
/// from arbitrary DMA bytes.
pub trait DmaValue: Copy + sealed::DmaValue {}

macro_rules! dma_values {
    ($($word:ty),+ $(,)?) => {$(
        impl sealed::DmaValue for $word {}
        impl DmaValue for $word {}
    )+};
}
dma_values!(
    u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize
);

/// A zeroed DMA allocation. The CPU view and device address may differ, and
/// physical pages need not be contiguous when a requester domain maps them.
pub struct DmaBuffer {
    memory: BufferMemory,
    bytes: usize,
}

enum BufferMemory {
    // Compatibility with RegionAllocator's explicitly unsafe identity contract.
    Identity(u64),
    // A RegionAllocator whose CPU view of bus address `bus` is at
    // `bus + offset` (wrapping), such as the engine's physmap, which every
    // engine page-table root maps.
    Window { bus: u64, offset: u64 },
    Domain(triomphe::Arc<allocator::Allocation>),
}

impl core::fmt::Debug for DmaBuffer {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DmaBuffer")
            .field("bus_address", &self.bus_address())
            .field("cpu_address", &self.as_ptr())
            .field("bytes", &self.bytes)
            .finish()
    }
}

impl DmaBuffer {
    /// The address programmed into this requester's device, never a CPU pointer.
    pub fn bus_address(&self) -> u64 {
        match &self.memory {
            BufferMemory::Identity(address) => *address,
            BufferMemory::Window { bus, .. } => *bus,
            BufferMemory::Domain(allocation) => allocation.bus_address(),
        }
    }

    /// Historical spelling of the device address. This need not be physical
    /// RAM and must never be converted into a CPU pointer; use `as_ptr`.
    pub fn physical(&self) -> u64 {
        self.bus_address()
    }

    pub fn len(&self) -> usize {
        self.bytes
    }

    pub fn is_empty(&self) -> bool {
        self.bytes == 0
    }

    pub fn as_ptr(&self) -> *mut u8 {
        match &self.memory {
            BufferMemory::Identity(address) => *address as *mut u8,
            BufferMemory::Window { bus, offset } => bus.wrapping_add(*offset) as *mut u8,
            BufferMemory::Domain(allocation) => allocation.cpu_address(self.bytes),
        }
    }

    pub fn bytes(&self) -> &[u8] {
        // SAFETY: the allocator keeps this exclusive CPU mapping for this buffer.
        unsafe { core::slice::from_raw_parts(self.as_ptr(), self.bytes) }
    }

    pub fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: as in `bytes`, and `&mut self` is the only reference.
        unsafe { core::slice::from_raw_parts_mut(self.as_ptr(), self.bytes) }
    }

    fn word<T: DmaValue>(&self, offset: usize) -> *mut T {
        assert!(
            offset <= self.bytes && size_of::<T>() <= self.bytes - offset,
            "DMA word outside buffer"
        );
        // SAFETY: bounds were checked without overflow. The allocator grants
        // this live RAM range for the rest of the boot.
        let pointer = unsafe { self.as_ptr().add(offset) }.cast::<T>();
        assert!(
            (pointer as usize).is_multiple_of(align_of::<T>()),
            "unaligned DMA word"
        );
        pointer
    }

    /// Volatile access for structures the device updates while the CPU polls.
    /// This checks both the complete word's bounds and its alignment.
    ///
    /// ```compile_fail
    /// use stela_drivers::bus::dma::DmaBuffer;
    /// fn invalid_boolean(buffer: &DmaBuffer) -> bool {
    ///     buffer.read(0)
    /// }
    /// ```
    pub fn read<T: DmaValue>(&self, offset: usize) -> T {
        // SAFETY: word checks bounds/alignment; every T bit pattern is valid.
        unsafe { self.word::<T>(offset).read_volatile() }
    }

    pub fn write<T: DmaValue>(&mut self, offset: usize, value: T) {
        // SAFETY: same checked word; &mut self owns this CPU write.
        unsafe { self.word::<T>(offset).write_volatile(value) }
    }
}

/// A source of DMA buffers.
pub trait DmaAllocator {
    /// `bytes` of zeroed memory aligned to `align`, a power of two.
    fn allocate(&mut self, bytes: usize, align: usize) -> Result<DmaBuffer, BlockError>;
}

/// Zeroed heap memory a test fixture hands out lowest first, aligned to
/// [`TestRegion::ALIGN`] so that whether an aligned allocation fits never
/// depends on where the heap placed the region. Freed on drop.
#[cfg(any(test, feature = "test-support"))]
pub struct TestRegion {
    layout: core::alloc::Layout,
    start: *mut u8,
    allocator: RegionAllocator,
}

#[cfg(any(test, feature = "test-support"))]
impl TestRegion {
    /// The largest alignment a driver requests: a 64 KiB transfer buffer.
    pub const ALIGN: usize = 64 * 1024;

    pub fn new(bytes: usize) -> Self {
        assert_ne!(bytes, 0, "an empty test DMA region");
        let layout = core::alloc::Layout::from_size_align(bytes, Self::ALIGN)
            .expect("test DMA region layout");
        // SAFETY: the layout has a non-zero size.
        let start = unsafe { alloc::alloc::alloc_zeroed(layout) };
        assert!(!start.is_null(), "test DMA region allocation");
        let base = start as u64;
        // SAFETY: the region owns this allocation until it drops, and test
        // peers reach it at the CPU's address.
        let allocator = unsafe { RegionAllocator::new(base..base + bytes as u64) };
        Self {
            layout,
            start,
            allocator,
        }
    }
}

#[cfg(any(test, feature = "test-support"))]
impl core::ops::Deref for TestRegion {
    type Target = RegionAllocator;
    fn deref(&self) -> &RegionAllocator {
        &self.allocator
    }
}

#[cfg(any(test, feature = "test-support"))]
impl core::ops::DerefMut for TestRegion {
    fn deref_mut(&mut self) -> &mut RegionAllocator {
        &mut self.allocator
    }
}

#[cfg(any(test, feature = "test-support"))]
impl DmaAllocator for TestRegion {
    fn allocate(&mut self, bytes: usize, align: usize) -> Result<DmaBuffer, BlockError> {
        self.allocator.allocate(bytes, align)
    }
}

#[cfg(any(test, feature = "test-support"))]
impl Drop for TestRegion {
    fn drop(&mut self) {
        // SAFETY: `start` came from `alloc_zeroed` with this layout.
        unsafe { alloc::alloc::dealloc(self.start, self.layout) }
    }
}

/// Identity-mapped DMA memory: the regions added to it, each handed out
/// front to back, and the ranges taken back once their device is done with
/// them, which later allocations use first.
///
/// A buffer is never taken back by dropping it: a driver may drop one its
/// device can still reach, after a timeout or a reset that failed, so only
/// [`Self::release`], whose caller proves the device stopped, returns it.
/// Memory never released is never handed out twice, and the alignment gaps
/// front-to-back allocation skips stay unused, as before release existed.
pub struct RegionAllocator {
    /// The regions added, in the order they were.
    regions: Vec<Range<u64>>,
    /// Each region's part not yet handed out, in the same order.
    untouched: Vec<Range<u64>>,
    /// The released ranges, sorted, disjoint and never adjacent.
    free: Vec<Range<u64>>,
    /// What is added to a bus address to reach its CPU view: zero for
    /// identity-mapped memory.
    offset: u64,
}

/// Why [`RegionAllocator::release`] refused a buffer, which it hands back.
#[derive(Debug)]
pub struct ReleaseRefused(pub DmaBuffer);

impl RegionAllocator {
    /// # Safety
    ///
    /// `region` must be RAM that only this allocator hands out, mapped
    /// writable at the same address, and reachable by every device's DMA.
    pub unsafe fn new(region: Range<u64>) -> Self {
        // SAFETY: forwarded; an identity view is offset zero.
        unsafe { Self::with_cpu_offset(region, 0) }
    }

    /// An allocator whose buffers the CPU reaches at their bus address plus
    /// `offset` (wrapping), not at the bus address itself: memory mapped
    /// through a window that every page-table root of its users keeps,
    /// rather than an identity map only some roots keep.
    ///
    /// # Safety
    ///
    /// `region` must be RAM that only this allocator hands out, reachable
    /// by every device's DMA at those addresses, and mapped writable at
    /// `address + offset` for every `address` in it, in every context that
    /// uses a buffer.
    pub unsafe fn with_cpu_offset(region: Range<u64>, offset: u64) -> Self {
        let mut allocator = Self {
            regions: Vec::new(),
            untouched: Vec::new(),
            free: Vec::new(),
            offset,
        };
        // SAFETY: forwarded.
        unsafe { allocator.add(region) };
        allocator
    }

    /// An allocator over all of `buffer`, whose buffers keep its CPU view:
    /// a driver's own arena carved from a buffer another allocator gave.
    ///
    /// # Safety
    ///
    /// `buffer` is kept for as long as this allocator's buffers are used,
    /// and nothing else uses its memory.
    pub unsafe fn within(buffer: &DmaBuffer) -> Self {
        let bus = buffer.bus_address();
        let offset = (buffer.as_ptr() as u64).wrapping_sub(bus);
        // SAFETY: a live DMA buffer, contiguous in bus and CPU addresses,
        // whose exclusive use the caller passes on.
        unsafe { Self::with_cpu_offset(bus..bus + buffer.len() as u64, offset) }
    }

    /// Grow by `region`. An empty region or one that overlaps a region
    /// already added is ignored.
    ///
    /// # Safety
    ///
    /// As for [`Self::new`], for `region`.
    pub unsafe fn add(&mut self, region: Range<u64>) {
        if region.start >= region.end
            || self
                .regions
                .iter()
                .any(|added| added.start < region.end && region.start < added.end)
        {
            return;
        }
        self.regions.push(region.clone());
        self.untouched.push(region);
    }

    /// The bytes free: released and never handed out.
    pub fn remaining(&self) -> u64 {
        self.free
            .iter()
            .chain(&self.untouched)
            .map(|range| range.end - range.start)
            .sum()
    }

    /// The bytes of the largest free range.
    pub fn largest(&self) -> u64 {
        self.free
            .iter()
            .chain(&self.untouched)
            .map(|range| range.end - range.start)
            .max()
            .unwrap_or(0)
    }

    /// Take `buffer` back for later allocations. A buffer this allocator did
    /// not hand out, or whose memory is already free, is refused.
    ///
    /// # Safety
    ///
    /// The device given `buffer`'s bus address has stopped reaching it: no
    /// command, ring or descriptor that names it is outstanding, and the
    /// device was told to forget it (its slot disabled, its queue deleted or
    /// its controller halted). No copy of its CPU pointer is used again.
    pub unsafe fn release(&mut self, buffer: DmaBuffer) -> Result<(), ReleaseRefused> {
        let start = match buffer.memory {
            BufferMemory::Identity(start) if self.offset == 0 => start,
            BufferMemory::Window { bus, offset } if offset == self.offset && offset != 0 => bus,
            _ => return Err(ReleaseRefused(buffer)),
        };
        let Some(end) = start.checked_add(buffer.bytes as u64) else {
            return Err(ReleaseRefused(buffer));
        };
        if start == end {
            return Ok(());
        }
        let owned = self
            .regions
            .iter()
            .any(|region| region.start <= start && end <= region.end);
        let free = self
            .free
            .iter()
            .chain(&self.untouched)
            .any(|range| range.start < end && start < range.end);
        if !owned || free {
            return Err(ReleaseRefused(buffer));
        }
        self.insert(start..end);
        Ok(())
    }

    /// Add `range`, which overlaps no free range, to the free list,
    /// merging it with its neighbours.
    fn insert(&mut self, range: Range<u64>) {
        let at = self.free.partition_point(|free| free.end < range.start);
        let mut merged = range;
        let mut end = at;
        while end < self.free.len() && self.free[end].start <= merged.end {
            merged.start = merged.start.min(self.free[end].start);
            merged.end = merged.end.max(self.free[end].end);
            end += 1;
        }
        self.free.splice(at..end, core::iter::once(merged));
    }
}

impl DmaAllocator for RegionAllocator {
    fn allocate(&mut self, bytes: usize, align: usize) -> Result<DmaBuffer, BlockError> {
        assert!(align.is_power_of_two());
        let bytes64 = bytes as u64;
        let fit = |range: &Range<u64>| {
            let start = range.start.checked_next_multiple_of(align as u64)?;
            let end = start.checked_add(bytes64)?;
            (end <= range.end).then_some(start)
        };
        let start = if let Some((index, start)) = self
            .free
            .iter()
            .enumerate()
            .find_map(|(index, range)| fit(range).map(|start| (index, start)))
        {
            // Released memory first, keeping the gap before it free.
            let range = self.free[index].clone();
            let end = start + bytes64;
            let mut rest = Vec::new();
            if range.start < start {
                rest.push(range.start..start);
            }
            if end < range.end {
                rest.push(end..range.end);
            }
            self.free.splice(index..=index, rest);
            start
        } else {
            let range = self
                .untouched
                .iter_mut()
                .find(|range| fit(range).is_some())
                .ok_or(BlockError::NoDmaMemory)?;
            let start = fit(range).ok_or(BlockError::NoDmaMemory)?;
            range.start = start + bytes64;
            start
        };
        let memory = match self.offset {
            0 => BufferMemory::Identity(start),
            offset => BufferMemory::Window { bus: start, offset },
        };
        let mut buffer = DmaBuffer { memory, bytes };
        buffer.bytes_mut().fill(0);
        Ok(buffer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    extern crate std;

    #[test]
    fn region_allocations_are_aligned_zeroed_and_bounded() {
        let mut backing: Vec<u64> = vec![u64::MAX; 4096];
        let start = backing.as_mut_ptr() as u64;
        // SAFETY: the vector outlives the allocator and nothing else uses it.
        let mut allocator = unsafe { RegionAllocator::new(start..start + 32768) };
        let first = allocator.allocate(100, 8).unwrap();
        assert_eq!(first.physical(), start);
        assert!(first.bytes().iter().all(|byte| *byte == 0));
        let page = allocator.allocate(4096, 4096).unwrap();
        assert!(page.physical().is_multiple_of(4096));
        assert!(page.physical() >= start + 100);
        assert_eq!(
            allocator.allocate(32768, 8).unwrap_err(),
            BlockError::NoDmaMemory
        );
        let mut word = allocator.allocate(8, 8).unwrap();
        word.write(0, 0x1234u32);
        assert_eq!(word.read::<u32>(0), 0x1234);
    }

    /// A region of `bytes` over a vector the test owns.
    fn region(backing: &mut Vec<u64>) -> (u64, RegionAllocator) {
        let start = backing.as_mut_ptr() as u64;
        let end = start + (backing.len() * 8) as u64;
        // SAFETY: the vector outlives the allocator and nothing else uses it.
        (start, unsafe { RegionAllocator::new(start..end) })
    }

    #[test]
    fn an_exhausted_region_refuses_until_memory_is_released() {
        let mut backing: Vec<u64> = vec![0; 4096];
        let (start, mut allocator) = region(&mut backing);
        let mut taken = Vec::new();
        while let Ok(buffer) = allocator.allocate(4096, 4096) {
            taken.push(buffer);
        }
        assert!(!taken.is_empty());
        assert!(allocator.largest() < 4096);
        assert_eq!(
            allocator.allocate(4096, 4096).unwrap_err(),
            BlockError::NoDmaMemory
        );
        let mut middle = taken.swap_remove(1);
        let address = middle.physical();
        middle.bytes_mut().fill(0xa5);
        // SAFETY: no device was given the buffer.
        unsafe { allocator.release(middle) }.unwrap();
        let again = allocator.allocate(4096, 4096).unwrap();
        assert_eq!(again.physical(), address);
        assert!(
            again.bytes().iter().all(|byte| *byte == 0),
            "reused memory is zeroed"
        );
        assert!(again.physical() >= start);
    }

    #[test]
    fn released_neighbours_merge_into_one_range() {
        let mut backing: Vec<u64> = vec![0; 3 * 512];
        let (_, mut allocator) = region(&mut backing);
        let total = allocator.remaining();
        let buffers: Vec<DmaBuffer> = (0..3)
            .map(|_| allocator.allocate(4096, 8).unwrap())
            .collect();
        assert_eq!(allocator.remaining(), total - 3 * 4096);
        for buffer in buffers.into_iter().rev() {
            // SAFETY: no device was given the buffers.
            unsafe { allocator.release(buffer) }.unwrap();
        }
        assert_eq!(allocator.remaining(), total);
        assert_eq!(allocator.largest(), total);
        allocator.allocate(3 * 4096, 8).unwrap();
    }

    #[test]
    fn a_foreign_or_already_free_buffer_is_refused() {
        let mut backing: Vec<u64> = vec![0; 1024];
        let (_, mut allocator) = region(&mut backing);
        let mut other: Vec<u64> = vec![0; 1024];
        let (_, mut stranger) = region(&mut other);
        let foreign = stranger.allocate(64, 8).unwrap();
        // SAFETY: refused before any effect.
        assert!(unsafe { allocator.release(foreign) }.is_err());
        let buffer = allocator.allocate(64, 8).unwrap();
        let copy = DmaBuffer {
            memory: BufferMemory::Identity(buffer.physical()),
            bytes: 64,
        };
        // SAFETY: no device was given the buffer.
        unsafe { allocator.release(buffer) }.unwrap();
        // SAFETY: refused before any effect.
        let refused = unsafe { allocator.release(copy) }.unwrap_err();
        assert_eq!(refused.0.len(), 64);
        assert_eq!(allocator.remaining(), 8192);
    }

    #[test]
    fn an_added_region_grows_the_pool() {
        let mut first: Vec<u64> = vec![0; 512];
        let (_, mut allocator) = region(&mut first);
        allocator.allocate(4096, 8).unwrap();
        assert!(allocator.allocate(4096, 8).is_err());
        let mut second: Vec<u64> = vec![0; 1024];
        let start = second.as_mut_ptr() as u64;
        // SAFETY: the vector outlives the allocator and nothing else uses it.
        unsafe { allocator.add(start..start + 8192) };
        // An overlapping region is not added twice.
        // SAFETY: as above; the call has no effect.
        unsafe { allocator.add(start + 4096..start + 12288) };
        assert_eq!(allocator.remaining(), 8192);
        let grown = allocator.allocate(8192, 8).unwrap();
        assert_eq!(grown.physical(), start);
    }

    #[test]
    fn a_device_plugged_a_thousand_times_leaves_the_pool_as_it_was() {
        let mut backing: Vec<u64> = vec![0; 64 * 512];
        let (_, mut allocator) = region(&mut backing);
        // Long-lived controller memory, then a device's contexts and rings.
        let _controller = allocator.allocate(3 * 4096, 4096).unwrap();
        let mut steady = None;
        for _ in 0..1000 {
            let device = [
                allocator.allocate(2048, 64).unwrap(),
                allocator.allocate(1024, 64).unwrap(),
                allocator.allocate(64, 16).unwrap(),
                allocator.allocate(4096, 4096).unwrap(),
            ];
            for buffer in device {
                // SAFETY: the modeled device was disabled first.
                unsafe { allocator.release(buffer) }.unwrap();
            }
            // The first plug skips an alignment gap in fresh memory; every
            // later one reuses what the one before released.
            let now = (allocator.remaining(), allocator.largest());
            assert_eq!(*steady.get_or_insert(now), now);
        }
    }

    #[test]
    fn without_release_allocations_never_go_below_earlier_ones() {
        let mut backing: Vec<u64> = vec![0; 32 * 512];
        let (_, mut allocator) = region(&mut backing);
        let mut end = 0;
        for (bytes, align) in [(1, 65536), (100, 8), (4096, 4096), (64, 64), (8, 8)] {
            let Ok(buffer) = allocator.allocate(bytes, align) else {
                continue;
            };
            assert!(buffer.physical() >= end, "a skipped gap was handed out");
            end = buffer.physical() + bytes as u64;
        }
    }

    #[test]
    fn integer_words_preserve_device_bits_and_neighbor_bytes() {
        let mut backing = [0u64; 4];
        let mut buffer = DmaBuffer {
            memory: BufferMemory::Identity(backing.as_mut_ptr() as u64),
            bytes: size_of_val(&backing),
        };
        buffer.bytes_mut().fill(0xa5);
        buffer.write(2, 0x1234u16);
        buffer.write(8, -2i32);
        buffer.write(24, u64::MAX);
        assert_eq!(buffer.read::<u16>(2), 0x1234);
        assert_eq!(buffer.read::<i32>(8), -2);
        assert_eq!(buffer.read::<u64>(24), u64::MAX);
        assert_eq!(&buffer.bytes()[..2], &[0xa5; 2]);
        assert_eq!(&buffer.bytes()[4..8], &[0xa5; 4]);
        assert_eq!(&buffer.bytes()[12..24], &[0xa5; 12]);
    }

    #[test]
    fn invalid_word_ranges_and_alignment_refuse_before_access() {
        let mut backing = [0xa5a5_a5a5_a5a5_a5a5u64; 4];
        let original = backing;
        let mut buffer = DmaBuffer {
            memory: BufferMemory::Identity(backing.as_mut_ptr() as u64),
            bytes: size_of_val(&backing),
        };
        for offset in [1, 29, 32, usize::MAX - 1, usize::MAX] {
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                    || buffer.read::<u32>(offset)
                ))
                .is_err()
            );
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    buffer.write(offset, 0u32)
                }))
                .is_err()
            );
        }
        // Aligned offsets do not repair an allocator's unaligned base.
        let unaligned = DmaBuffer {
            memory: BufferMemory::Identity(buffer.physical() + 1),
            bytes: buffer.len() - 1,
        };
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unaligned.read::<u32>(0)))
                .is_err()
        );
        assert_eq!(backing, original);
    }

    /// A window allocator's buffers carry bus addresses for the device and
    /// reach memory only through the CPU view; an identity allocator never
    /// takes one of them back, nor the reverse.
    #[test]
    fn a_window_allocator_reaches_memory_only_through_its_cpu_view() {
        const OFFSET: u64 = 0x1000_0000_0000;
        let mut backing: Vec<u64> = vec![u64::MAX; 1024];
        let cpu = backing.as_mut_ptr() as u64;
        let bus = cpu.wrapping_sub(OFFSET);
        // SAFETY: the vector outlives the allocator and nothing else uses it;
        // the bus addresses are never dereferenced.
        let mut allocator = unsafe { RegionAllocator::with_cpu_offset(bus..bus + 8192, OFFSET) };
        let mut buffer = allocator.allocate(64, 64).unwrap();
        assert_eq!(buffer.bus_address() % 64, 0);
        assert!((bus..bus + 8192).contains(&buffer.bus_address()));
        assert_eq!(
            buffer.as_ptr() as u64,
            buffer.bus_address().wrapping_add(OFFSET)
        );
        assert!(buffer.bytes().iter().all(|byte| *byte == 0));
        buffer.write(4, 0x5a5a_1234u32);
        assert_eq!(buffer.read::<u32>(4), 0x5a5a_1234);
        let mut identity_backing: Vec<u64> = vec![0; 1024];
        let (_, mut identity) = region(&mut identity_backing);
        // SAFETY: refused before any effect.
        let buffer = unsafe { identity.release(buffer) }.unwrap_err().0;
        let other = identity.allocate(64, 64).unwrap();
        // SAFETY: refused before any effect.
        assert!(unsafe { allocator.release(other) }.is_err());
        let total = allocator.remaining();
        // SAFETY: no device was given the buffer.
        unsafe { allocator.release(buffer) }.unwrap();
        assert_eq!(allocator.remaining(), total + 64);
    }

    /// An arena carved from a window buffer keeps that buffer's CPU view.
    #[test]
    fn an_arena_within_a_buffer_keeps_its_cpu_view() {
        const OFFSET: u64 = 0x2000_0000_0000;
        let mut backing: Vec<u64> = vec![0; 2048];
        let cpu = backing.as_mut_ptr() as u64;
        let bus = cpu.wrapping_sub(OFFSET);
        // SAFETY: as in the test above.
        let mut outer = unsafe { RegionAllocator::with_cpu_offset(bus..bus + 16384, OFFSET) };
        let arena = outer.allocate(4096, 4096).unwrap();
        // SAFETY: the arena is kept below and used by nothing else.
        let mut inner = unsafe { RegionAllocator::within(&arena) };
        let mut piece = inner.allocate(128, 64).unwrap();
        assert!(piece.bus_address() >= arena.bus_address());
        assert_eq!(
            piece.as_ptr() as u64,
            piece.bus_address().wrapping_add(OFFSET)
        );
        piece.write(0, 7u64);
        let at = (piece.bus_address() - arena.bus_address()) as usize;
        assert_eq!(arena.read::<u64>(at), 7);
        let mut identity_backing: Vec<u64> = vec![0; 512];
        let (_, mut identity) = region(&mut identity_backing);
        let plain = identity.allocate(512, 64).unwrap();
        let start = plain.bus_address();
        // SAFETY: as above.
        let mut nested = unsafe { RegionAllocator::within(&plain) };
        assert_eq!(nested.allocate(8, 8).unwrap().as_ptr() as u64, start);
    }
}
