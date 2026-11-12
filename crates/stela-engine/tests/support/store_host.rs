//! Bounded native test host: only explicitly owned code/data/table pages.
use anyhow::{Result, ensure};
use std::{cell::Cell, collections::BTreeMap, rc::Rc};
use stela_compiler::{GuestState, TargetCpu};
use stela_contracts::Chip;
use stela_engine::{
    execution::{platform::Mapping as NativeMapping, reference::NativeRunner},
    machine::{
        Cpu, CpuProfile, FaultCause, Host, InstallError, Machine, NativeCode, NativeExit,
        NoDevices, Regime, exception::Pstate, identification, mmu, registers::encodings,
        system::Register,
    },
};
const PAGE: usize = 4096;
const PAGES: usize = 4;
const BYTES: usize = PAGE * PAGES;
/// Inaccessible bytes reserved on each side of a region. Native code
/// reaches guest addresses as host addresses, so an access the guest's
/// tables do not map must fault in the host too: past an unguarded region
/// it read or wrote whatever the process had mapped there (another
/// region, an engine allocation), which corrupted the run at random.
/// Every test loop walks from inside its region in steps far below this.
const GUARD: usize = 16 * PAGE;

struct Region {
    address: u64,
    readable: [Cell<bool>; PAGES],
}
impl Drop for Region {
    fn drop(&mut self) {
        let reservation = (self.address - GUARD as u64) as *mut libc::c_void;
        // SAFETY: this owner reserved the complete OS mapping, guards
        // included, and all clones keeping its address have dropped before
        // this final Rc owner.
        let unmapped = unsafe { libc::munmap(reservation, GUARD + BYTES + GUARD) };
        assert_eq!(unmapped, 0);
    }
}
impl Region {
    fn offset(&self, address: u64, bytes: usize) -> Option<usize> {
        let offset = usize::try_from(address.checked_sub(self.address)?).ok()?;
        let end = offset.checked_add(bytes)?;
        if end > BYTES {
            return None;
        }
        if bytes != 0 && (offset / PAGE..=(end - 1) / PAGE).any(|page| !self.readable[page].get()) {
            return None;
        }
        Some(offset)
    }
    fn read(&self, address: u64, out: &mut [u8]) -> bool {
        if self.offset(address, out.len()).is_none() {
            return false;
        }
        // SAFETY: the OS mapping and every touched page are readable;
        // this distinct Rust output does not alias the externally mapped RAM.
        unsafe {
            core::ptr::copy_nonoverlapping(address as *const u8, out.as_mut_ptr(), out.len())
        };
        true
    }
    fn write(&self, address: u64, bytes: &[u8]) -> bool {
        if self.offset(address, bytes.len()).is_none() {
            return false;
        }
        // SAFETY: this bounded private RW mapping is idle outside a native run;
        // no Rust data reference covers it and the source is separate storage.
        unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), address as *mut u8, bytes.len()) };
        true
    }
}
pub struct Mapping(pub u64, Rc<Region>);
impl Mapping {
    pub fn new() -> Result<Self> {
        // SAFETY: a fresh private reservation, never a fixed-address
        // overwrite: inaccessible guards around the region's bytes.
        let reservation = unsafe {
            libc::mmap(
                core::ptr::null_mut(),
                GUARD + BYTES + GUARD,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        };
        ensure!(
            reservation != libc::MAP_FAILED,
            "reserve test RAM: {}",
            std::io::Error::last_os_error()
        );
        let address = reservation as u64 + GUARD as u64;
        // SAFETY: the region's bytes inside the reservation just made.
        if unsafe { libc::mprotect(address as *mut _, BYTES, libc::PROT_READ | libc::PROT_WRITE) }
            != 0
        {
            let error = std::io::Error::last_os_error();
            // SAFETY: the reservation above, which nothing else refers to.
            unsafe { libc::munmap(reservation, GUARD + BYTES + GUARD) };
            anyhow::bail!("map test RAM: {error}");
        }
        let mapping = Self(
            address,
            Rc::new(Region {
                address,
                readable: std::array::from_fn(|_| Cell::new(true)),
            }),
        );
        ensure!(
            address != 0 && address < (1 << 47),
            "test mapping exceeds declared 47bit guest range"
        );
        Ok(mapping)
    }

    pub fn write(&self, address: u64, bytes: &[u8]) -> bool {
        self.1.write(address, bytes)
    }
    pub fn read(&self, address: u64, bytes: usize) -> Option<Vec<u8>> {
        let mut out = vec![0; bytes];
        self.1.read(address, &mut out).then_some(out)
    }
    pub fn protect(&self, offset: usize, bytes: usize, readable: bool) -> Result<()> {
        ensure!(
            offset.is_multiple_of(PAGE)
                && bytes.is_multiple_of(PAGE)
                && offset.checked_add(bytes).is_some_and(|end| end <= BYTES),
            "protection range"
        );
        let flags = if readable {
            libc::PROT_READ | libc::PROT_WRITE
        } else {
            libc::PROT_NONE
        };
        ensure!(
            // SAFETY: a page-aligned extent of this mapping; tests call only after native
            // return and restore rights before reading RAM.
            unsafe { libc::mprotect((self.0 + offset as u64) as *mut _, bytes, flags) } == 0,
            "protect test RAM: {}",
            std::io::Error::last_os_error()
        );
        for page in offset / PAGE..(offset + bytes) / PAGE {
            self.1.readable[page].set(readable);
        }
        Ok(())
    }
}

#[repr(C, align(4096))]
struct Table([u64; 512]);
struct Tables {
    root: u64,
    pages: BTreeMap<u64, Box<Table>>,
}
impl Tables {
    fn new(regions: &[Rc<Region>], vector: u64) -> Self {
        let mut tables = Self {
            root: 0,
            pages: BTreeMap::new(),
        };
        tables.root = tables.allocate();
        for region in regions {
            for offset in (0..BYTES).step_by(PAGE) {
                let address = region.address + offset as u64;
                let mut current = tables.root;
                for shift in [39, 30, 21] {
                    let index = ((address >> shift) & 511) as usize;
                    let mut entry = tables.pages[&current].0[index];
                    if entry == 0 {
                        entry = tables.allocate() | 3;
                        tables.pages.get_mut(&current).unwrap().0[index] = entry;
                    }
                    current = entry & !0xfff;
                }
                // AF, inner-shareable Normal-WB attribute0. Code/data are
                // EL0 RW. The ERET vector is privileged-only and EL0 XN;
                // architectural EL1 execution excludes EL0-writable pages.
                tables.pages.get_mut(&current).unwrap().0[((address >> 12) & 511) as usize] =
                    address
                        | 3
                        | (if region.address == vector {
                            1 << 54
                        } else {
                            1 << 6
                        })
                        | (3 << 8)
                        | (1 << 10);
            }
        }
        tables
    }
    fn allocate(&mut self) -> u64 {
        let page = Box::new(Table([0; 512]));
        let address = (&raw const *page) as u64;
        self.pages.insert(address, page);
        address
    }
    fn read(&self, address: u64, out: &mut [u8]) -> bool {
        let Some(page) = self.pages.get(&(address & !0xfff)) else {
            return false;
        };
        let offset = (address & 0xfff) as usize;
        if offset.checked_add(out.len()).is_none_or(|end| end > PAGE) {
            return false;
        }
        // SAFETY: initialized immutable table bytes, kept through the run,
        // copied to a disjoint caller buffer within this one page.
        unsafe {
            core::ptr::copy_nonoverlapping(
                (&raw const **page).cast::<u8>().add(offset),
                out.as_mut_ptr(),
                out.len(),
            )
        };
        true
    }
}

struct Executable {
    address: u64,
    bytes: usize,
}
impl Drop for Executable {
    fn drop(&mut self) {
        assert_eq!(
            // SAFETY: uniquely owns this complete page-rounded executable mapping.
            unsafe { libc::munmap(self.address as *mut _, self.bytes) },
            0
        );
    }
}
pub struct NativeHost {
    regions: Vec<Rc<Region>>,
    tables: Tables,
    code: Vec<Executable>,
    runner: NativeRunner,
    entry: NativeMapping,
    vector: u64,
}
impl NativeHost {
    pub fn lower_synchronous(&self) -> u64 {
        self.vector + 0x400
    }
}
impl Host for NativeHost {
    fn read_ram(&mut self, address: u64, out: &mut [u8]) -> bool {
        self.tables.read(address, out)
            || self.regions.iter().any(|region| region.read(address, out))
    }
    fn write_ram(&mut self, address: u64, bytes: &[u8]) -> bool {
        self.regions
            .iter()
            .any(|region| region.write(address, bytes))
    }
    fn install(&mut self, bytes: &[u8]) -> Result<NativeCode, InstallError> {
        let length = bytes.len().div_ceil(PAGE) * PAGE;
        // SAFETY: requests a new private RW mapping for this immutable install.
        let pointer = unsafe {
            libc::mmap(
                core::ptr::null_mut(),
                length,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        };
        if pointer == libc::MAP_FAILED {
            return Err(InstallError::Full);
        }
        let mapping = Executable {
            address: pointer as u64,
            bytes: length,
        };
        if pointer.is_null() {
            return Err(InstallError::Full);
        }
        // SAFETY: fresh RW mapping is disjoint from compiler output; no entry
        // escapes before its initialized bytes become RX.
        unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), pointer.cast(), bytes.len()) };
        // SAFETY: complete kept OS extent, idle and fully initialized.
        if unsafe { libc::mprotect(pointer, length, libc::PROT_READ | libc::PROT_EXEC) } != 0 {
            return Err(InstallError::Full);
        }
        self.code.push(mapping);
        self.runner.set_ranges(
            self.code
                .iter()
                .map(|code| (code.address, code.address + code.bytes as u64))
                .collect(),
        );
        Ok(NativeCode(pointer as usize))
    }
    fn read_native(&mut self, native: NativeCode, bytes: &mut [u8]) -> bool {
        let Some(end) = native.0.checked_add(bytes.len()) else {
            return false;
        };
        if !self.code.iter().any(|code| {
            native.0 as u64 >= code.address && end as u64 <= code.address + code.bytes as u64
        }) {
            return false;
        }
        // SAFETY: this owner keeps the idle readable executable mmap through
        // fault settlement; no address/reference escapes and discard owns munmap.
        unsafe {
            core::ptr::copy_nonoverlapping(native.0 as *const u8, bytes.as_mut_ptr(), bytes.len());
        }
        true
    }

    fn patch(&mut self, native: NativeCode, bytes: &[u8]) {
        let mapping = self
            .code
            .iter()
            .find(|code| {
                native.0 as u64 >= code.address
                    && (native.0 as u64)
                        .checked_add(bytes.len() as u64)
                        .is_some_and(|end| end <= code.address + code.bytes as u64)
            })
            .unwrap();
        // SAFETY: the mapping is idle; Machine patches only outside
        // invocation. No executable or byte reference escapes this owner.
        unsafe {
            assert_eq!(
                libc::mprotect(
                    mapping.address as *mut _,
                    mapping.bytes,
                    libc::PROT_READ | libc::PROT_WRITE
                ),
                0
            );
            core::ptr::copy_nonoverlapping(bytes.as_ptr(), native.0 as *mut u8, bytes.len());
            assert_eq!(
                libc::mprotect(
                    mapping.address as *mut _,
                    mapping.bytes,
                    libc::PROT_READ | libc::PROT_EXEC
                ),
                0
            );
        }
    }
    fn discard_code(&mut self) {
        self.runner.set_ranges(vec![]);
        self.code.clear();
    }
    unsafe fn execute(&mut self, native: NativeCode, state: &mut GuestState) -> NativeExit {
        // SAFETY: Machine keeps this installed block and its block ABI, all
        // reachable mappings and the distinct GuestState for the whole call.
        match unsafe { self.runner.run_through(self.entry.entry(), native.0, state) } {
            Ok(exit) => NativeExit::Exit {
                next_pc: exit.next_pc,
                kind: exit.kind,
            },
            Err(fault) => {
                let cause = fault.check_cause().unwrap_or_else(|| {
                    let permission = self.regions.iter().any(|region| {
                        fault.address >= region.address
                            && fault.address < region.address + BYTES as u64
                    });
                    FaultCause::Guest(mmu::Fault {
                        kind: if permission {
                            mmu::FaultKind::Permission
                        } else {
                            mmu::FaultKind::Translation
                        },
                        level: 3,
                        second_stage: None,
                    })
                });
                NativeExit::Fault(
                    fault.native(fault.rip.wrapping_sub(native.0 as u64) as usize, cause),
                )
            }
        }
    }
    fn report(&mut self, event: &stela_engine::machine::Event) {
        if let stela_engine::machine::Event::Exception { from_el, .. } = event {
            assert_eq!(*from_el, 0, "unexpected test-kernel exception: {event:?}");
        }
    }
    fn set_regime(&mut self, _: &Regime) {}
    fn invalidate_translations(&mut self) {}
    fn counter(&mut self) -> u64 {
        0
    }
    fn arm_timer(&mut self, _: Option<u64>) {}
    fn wait_for_interrupt(&mut self) {
        panic!("store validation never waits")
    }
}

pub fn machine(
    code: &Mapping,
    data: &Mapping,
    words: &[u32],
    target: TargetCpu,
) -> Result<Machine<NativeHost, NoDevices>> {
    ensure!(
        code.write(
            code.0,
            &words
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .collect::<Vec<_>>()
        ),
        "write code"
    );
    let vector = Mapping::new()?;
    ensure!(
        vector.write(vector.0 + 0x400, &0xd69f_03e0u32.to_le_bytes()),
        "write ERET vector"
    );
    let regions = vec![Rc::clone(&code.1), Rc::clone(&data.1), Rc::clone(&vector.1)];
    let tables = Tables::new(&regions, vector.0);
    let chip = Chip::AppleM2;
    let mut cpu = Cpu::of(
        chip,
        code.0,
        Pstate {
            el: 0,
            sp: false,
            daif: 0,
            ..Pstate::reset(0)
        },
    );
    for (register, value) in [
        (Register::SCTLR_EL1, 1 | (1 << 2) | (1 << 12)),
        (
            Register::TCR_EL1,
            17 | (1 << 8) | (1 << 10) | (3 << 12) | (1 << 23) | (5 << 32),
        ),
        (Register::MAIR_EL1, 0xff),
        (Register::TTBR0_EL1, tables.root),
        (Register::CPACR_EL1, 3 << 20),
        (Register::VBAR_EL1, vector.0),
        (Register::HCR_EL2, 1 << 31 | 1 << 40 | 1 << 41),
    ] {
        cpu.registers.set_raw(register, 0, value);
    }
    let mut fixed = identification::derive(chip, target, true, identification::Speculation::NONE);
    fixed.push((encodings::MIDR_EL1, 0x611f_0320));
    // 64-byte DC ZVA blocks, as an M2's.
    fixed.push((encodings::DCZID_EL0, 4));
    let profile = CpuProfile {
        name: "bounded native store validation",
        chip,
        counter_frequency: 24_000_000,
        fixed,
        processors: vec![],
    };
    let host = NativeHost {
        regions,
        tables,
        code: vec![],
        runner: NativeRunner::new(vec![])?,
        entry: NativeMapping::new(&stela_compiler::x86_64::pinned::entry())?,
        vector: vector.0,
    };
    let mut machine = Machine::new(cpu, host, NoDevices, profile, target);
    machine.set_breakpoints(&[
        machine.host.lower_synchronous(),
        code.0 + words.len() as u64 * 4,
    ]);
    Ok(machine)
}
