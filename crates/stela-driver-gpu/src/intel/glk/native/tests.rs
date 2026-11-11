// SPDX-License-Identifier: GPL-2.0-or-later

extern crate std;

use super::*;
use alloc::{boxed::Box, sync::Arc, vec::Vec};
use core::{
    cell::RefCell,
    ptr::NonNull,
    sync::atomic::{AtomicUsize, Ordering::Relaxed},
};
use std::alloc::{Layout, alloc_zeroed, dealloc, handle_alloc_error};
use stela_driver_base::{
    bus::{
        mmio::owned::{
            AddressSpace, Authority, InstalledMapping, MapFailure, MapRequest, MappingPlatform,
            PAGE_BYTES, PhysicalSpan, ResourceBacking,
            model::{MappingOwner, MappingRetention},
        },
        pci::{
            Address,
            assignment::OpaqueUse,
            owned::test_support::{Counters, fixture},
            requester::FunctionRole,
        },
    },
    on_wait, test_clock,
};

const BASE: u64 = 0x1_d000_0000;
const BAR_BYTES: u64 = 16 << 20;
const BDF: Address = Address {
    segment: 0,
    bus: 0,
    device: 2,
    function: 0,
};
const CLOCK: Clock = Clock {
    ticks_per_microsecond: 100,
};
const BAR: Bar = Bar::Memory {
    address: BASE,
    size: BAR_BYTES,
    prefetchable: false,
    wide: true,
};

struct Ram(NonNull<u8>);

// SAFETY: isolated heap bytes and no safe references. Raw accesses and the
// synchronous peer are confined to the fixture's creating test thread.
unsafe impl Send for Ram {}
// SAFETY: same confinement; sharing only keeps the allocation alive.
unsafe impl Sync for Ram {}

impl Ram {
    fn layout() -> Layout {
        Layout::from_size_align(REGISTER_BYTES, PAGE_BYTES as usize).unwrap()
    }

    fn new() -> Self {
        // SAFETY: fixed nonempty page-aligned layout.
        let layout = Self::layout();
        let allocation = unsafe { alloc_zeroed(layout) };
        Self(NonNull::new(allocation).unwrap_or_else(|| handle_alloc_error(layout)))
    }

    fn word(&self, offset: usize) -> *mut u32 {
        assert!(offset < REGISTER_BYTES && offset.is_multiple_of(size_of::<u32>()));
        // SAFETY: within the original aligned retained allocation.
        unsafe { self.0.as_ptr().add(offset).cast() }
    }
}

impl Drop for Ram {
    fn drop(&mut self) {
        // SAFETY: last allocation owner, same layout.
        unsafe { dealloc(self.0.as_ptr(), Self::layout()) };
    }
}

struct BarBacking(MappingRetention<Arc<Ram>>);

// SAFETY: one immutable model resource retains the actual RAM assignment.
unsafe impl ResourceBacking for BarBacking {
    fn physical_span(&self) -> PhysicalSpan {
        PhysicalSpan::new(BASE, BAR_BYTES).unwrap()
    }

    fn kind(&self) -> ResourceKind {
        ResourceKind::Registers
    }

    fn begin_mapping(&self) -> Result<OpaqueUse, owned::Error> {
        self.0.begin_mapping()
    }
}

struct MappedBar {
    ram: Arc<Ram>,
    unmaps: Arc<AtomicUsize>,
}

// SAFETY: hosted BAR fixture with one installed mapping. Pointer lifetime is
// retained by the platform and resource. Context checks are pure.
unsafe impl MappingPlatform for MappedBar {
    fn check_context(&self, _: AddressSpace) -> Result<(), owned::Error> {
        Ok(())
    }

    unsafe fn map(
        &self,
        resource: &Resource,
        request: MapRequest,
    ) -> Result<InstalledMapping, MapFailure> {
        assert!(resource.backing().is::<BarBacking>());
        assert_eq!(request.physical.start(), BASE);
        assert!(request.physical.bytes() <= REGISTER_BYTES as u64);
        Ok(InstalledMapping {
            token: request.id,
            cpu_base: self.ram.0,
            physical: request.physical,
            attribute: request.attribute,
            protection: request.protection,
        })
    }

    unsafe fn unmap(
        &self,
        request: MapRequest,
        installed: InstalledMapping,
    ) -> Result<(), owned::Error> {
        assert_eq!(request.id, installed.token);
        assert_eq!(self.unmaps.fetch_add(1, Relaxed), 0);
        Ok(())
    }
}

struct PlatformOwner {
    bar: MappingOwner<Arc<Ram>>,
    drops: Arc<AtomicUsize>,
}

impl Drop for PlatformOwner {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Relaxed);
    }
}

struct Fixture {
    requester: StoppedRequester,
    config: Arc<Counters>,
    platform: PlatformOwner,
    resource: Resource,
    mapping: Mapping,
    ram: Arc<Ram>,
    unmaps: Arc<AtomicUsize>,
}

impl Fixture {
    fn new(device: u16, command: u16, pmcsr: u16, bytes: usize, protection: Protection) -> Self {
        let (mut config, counters) = fixture(BDF);
        config.write32(0, u32::from(device) << 16 | 0x8086).unwrap();
        config.write32(8, 0x0300_0001).unwrap();
        config.write16(regs::COMMAND, command).unwrap();
        config.write16(regs::STATUS, regs::STATUS_CAP_LIST).unwrap();
        config.write8(regs::CAPABILITY_LIST, 0x40).unwrap();
        config.write16(0x40, u16::from(CAPABILITY_POWER)).unwrap();
        config.write16(0x42, 3).unwrap();
        config.write16(0x44, pmcsr).unwrap();
        config.write32(BAR0, BASE as u32 | BAR_TYPE_64).unwrap();
        config.write32(BAR0 + 4, (BASE >> 32) as u32).unwrap();
        let function = Function::new(BDF, config).unwrap();
        // SAFETY: isolated function fixture, no real engine, firmware or DMA.
        let requester =
            unsafe { StoppedRequester::from_stopped_pci(function, FunctionRole::Physical) }
                .unwrap_or_else(|(error, _)| panic!("{error:?}"));
        let ram = Arc::new(Ram::new());
        let unmaps = Arc::new(AtomicUsize::new(0));
        // SAFETY: actual retained model allocation in the same test thread.
        let bar = unsafe { MappingOwner::from_owned(ram.clone(), |_| Ok(())) };
        let resource = Resource::new(Box::new(BarBacking(bar.retention().unwrap())));
        // SAFETY: isolated test platform, no native pages or MMIO.
        let authority = unsafe {
            Authority::isolated_model(
                Arc::new(MappedBar {
                    ram: ram.clone(),
                    unmaps: unmaps.clone(),
                }),
                AddressSpace {
                    identity: 99,
                    physical_bits: 48,
                    virtual_bits: 48,
                },
                Vec::new(),
            )
        }
        .unwrap();
        let mapping = authority
            .map(
                resource.clone(),
                0,
                bytes,
                CacheAttribute::Uncached,
                protection,
            )
            .unwrap();
        Self {
            requester,
            config: counters,
            platform: PlatformOwner {
                bar,
                drops: Arc::new(AtomicUsize::new(0)),
            },
            resource,
            mapping,
            ram,
            unmaps,
        }
    }

    fn attach(self) -> Result<Uncore<PlatformOwner>, AttachRefusal<PlatformOwner>> {
        // SAFETY: entirely isolated fixture. Negative cases exercise
        // admission only, before any access through the register transport.
        unsafe {
            Uncore::attach(
                self.requester,
                BAR,
                self.resource,
                self.mapping,
                self.platform,
                CLOCK,
            )
        }
    }
}

const CONTAINED: u16 = regs::COMMAND_INTX_DISABLE | regs::COMMAND_MEMORY;

struct Peer {
    ram: Arc<Ram>,
    domain: ForcewakeDomain,
    request_words: Vec<u32>,
    fault_on_request: Option<u32>,
}

std::thread_local! {
    static PEER: RefCell<Option<Peer>> = const { RefCell::new(None) };
}

fn service_peer() {
    PEER.with(|peer| {
        let mut peer = peer.borrow_mut();
        let Some(peer) = peer.as_mut() else {
            return;
        };
        let pair = peer.domain.registers();
        // SAFETY: the driver is paused on this same test thread. All bytes
        // belong to the retained hosted allocation.
        let request = unsafe { peer.ram.word(pair.request).read_volatile() };
        peer.request_words.push(request);
        let ack = if peer.fault_on_request == Some(request) {
            u32::MAX
        } else {
            2 | request & 1
        };
        // SAFETY: same synchronous model RAM and no concurrent accesses.
        unsafe { peer.ram.word(pair.acknowledge).write_volatile(ack) };
    });
}

fn run(
    uncore: Uncore<PlatformOwner>,
    ram: Arc<Ram>,
    domain: ForcewakeDomain,
    fault_on_request: Option<u32>,
) -> (
    Result<Uncore<PlatformOwner>, WakeRefusal<PlatformOwner>>,
    Vec<u32>,
) {
    let _time = test_clock::virtualize();
    on_wait(service_peer);
    PEER.with(|peer| {
        *peer.borrow_mut() = Some(Peer {
            ram,
            domain,
            request_words: Vec::new(),
            fault_on_request,
        });
    });
    let result = uncore.forcewake_cycle(domain);
    let request_words = PEER.with(|peer| peer.borrow_mut().take().unwrap().request_words);
    (result, request_words)
}

#[test]
fn owned_bar_cycles_and_returns_the_exact_original_graph() {
    for domain in [
        ForcewakeDomain::Render,
        ForcewakeDomain::Gt,
        ForcewakeDomain::Media,
    ] {
        let model = Fixture::new(0x3185, CONTAINED, 0, REGISTER_BYTES, Protection::ReadWrite);
        let (ram, unmaps) = (model.ram.clone(), model.unmaps.clone());
        let config = model.config.clone();
        let drops = model.platform.drops.clone();
        let resource = model.resource.clone();
        let uncore = model
            .attach()
            .unwrap_or_else(|refusal| panic!("{:?}", refusal.error));
        let (result, writes) = run(uncore, ram, domain, None);
        let released = result.unwrap_or_else(|refusal| panic!("{:?}", refusal.error));
        assert_eq!(writes, [0x0001_0001, 0x0001_0000]);
        assert_eq!(unmaps.load(Relaxed), 0);
        assert_eq!(config.unmaps.load(Relaxed), 0);
        assert_eq!(drops.load(Relaxed), 0);
        // SAFETY: fixture has observed release ACK and all raw peers ended.
        let mut parts = unsafe { released.into_parts() };
        assert!(parts.mapping.retains_resource(&resource).unwrap());
        assert_eq!(parts.function.identity().address, BDF);
        // SAFETY: no other model actors or pointer uses remain.
        unsafe {
            parts.mapping.retire().unwrap();
            parts.function.config().retire().unwrap();
        }
        parts.platform.bar.close().unwrap();
        drop(parts);
        assert_eq!(unmaps.load(Relaxed), 1);
        assert_eq!(config.unmaps.load(Relaxed), 1);
        assert_eq!(drops.load(Relaxed), 1);
    }
}

#[test]
fn owned_bar_attachment_rejects_wrong_function_decode_power_and_mapping() {
    for (device, command, pmcsr, bytes, protection, expected) in [
        (
            0x5917,
            CONTAINED,
            0,
            REGISTER_BYTES,
            Protection::ReadWrite,
            Error::UnsupportedFunction,
        ),
        (
            0x3185,
            regs::COMMAND_INTX_DISABLE,
            0,
            REGISTER_BYTES,
            Protection::ReadWrite,
            Error::DecodeDisabled,
        ),
        (
            0x3185,
            CONTAINED,
            3,
            REGISTER_BYTES,
            Protection::ReadWrite,
            Error::PowerState,
        ),
        (
            0x3185,
            CONTAINED,
            0,
            REGISTER_BYTES,
            Protection::ReadOnly,
            Error::Bar,
        ),
        (
            0x3185,
            CONTAINED,
            0,
            4096,
            Protection::ReadWrite,
            Error::Bar,
        ),
    ] {
        let model = Fixture::new(device, command, pmcsr, bytes, protection);
        let (ram, unmaps) = (model.ram.clone(), model.unmaps.clone());
        let refusal = model.attach().err().expect("attachment must refuse");
        assert_eq!(refusal.error, expected);
        assert_eq!(unmaps.load(Relaxed), 0);
        // SAFETY: no other model actor is using this word.
        assert_eq!(
            unsafe {
                ram.word(ForcewakeDomain::Gt.registers().request)
                    .read_volatile()
            },
            0
        );
    }
}

#[test]
fn numerically_identical_bar_from_another_reservation_is_rejected() {
    let mut model = Fixture::new(0x3185, CONTAINED, 0, REGISTER_BYTES, Protection::ReadWrite);
    let false_resource = Resource::new(Box::new(BarBacking(
        model.platform.bar.retention().unwrap(),
    )));
    assert_eq!(
        false_resource.physical_span(),
        model.resource.physical_span()
    );
    model.resource = false_resource;
    assert_eq!(model.attach().err().unwrap().error, Error::Bar);
}

#[test]
fn unreadable_ack_on_acquire_or_release_parks_config_bar_and_platform() {
    let writes = [0x0001_0001, 0x0001_0000];
    for (index, phase) in [forcewake::Phase::Acquire, forcewake::Phase::Release]
        .into_iter()
        .enumerate()
    {
        let model = Fixture::new(0x3185, CONTAINED, 0, REGISTER_BYTES, Protection::ReadWrite);
        let (ram, unmaps) = (model.ram.clone(), model.unmaps.clone());
        let config = model.config.clone();
        let drops = model.platform.drops.clone();
        let uncore = model
            .attach()
            .unwrap_or_else(|refusal| panic!("{:?}", refusal.error));
        let (result, observed) = run(uncore, ram, ForcewakeDomain::Gt, Some(writes[index]));
        let refusal = result.err().expect("uncertain request must refuse");
        assert_eq!(refusal.error, forcewake::Error::UnreadableAck(phase));
        assert_eq!(observed, writes[..=index]);
        drop(refusal);
        assert_eq!(unmaps.load(Relaxed), 0);
        assert_eq!(config.unmaps.load(Relaxed), 0);
        assert_eq!(drops.load(Relaxed), 0);
    }
}
