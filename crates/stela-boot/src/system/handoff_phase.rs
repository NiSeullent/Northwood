//! Boot files, final memory-map inventory and the engine transfer.
use super::*;
use crate::Result;
use stela_contracts::refuse;

pub(super) struct RetainedFiles {
    pub options: Region,
    pub source: Region,
    pub files: [File; MAX_FILES],
    pub file_count: usize,
}

pub(super) fn retain_config(config: Vec<u8>) -> Region {
    if config.is_empty() {
        return Region::default();
    }
    let region = Region {
        physical: config.as_ptr() as u64,
        bytes: config.len() as u64,
    };
    // The engine reads this immutable allocation through the shared physmap.
    core::mem::forget(config);
    region
}

/// Keep loader input allocations for engine reads through the physmap.
pub(super) fn retain_files(
    options: Vec<u8>,
    source: Vec<u8>,
    bundle: BundleFiles,
) -> Result<RetainedFiles> {
    let options_region = Region {
        physical: options.as_ptr() as u64,
        bytes: options.len() as u64,
    };
    let source_region = Region {
        physical: source.as_ptr() as u64,
        bytes: source.len() as u64,
    };
    let mut files = [File::default(); MAX_FILES];
    for (slot, (name, contents)) in files.iter_mut().zip(&bundle) {
        if name.len() > MAX_NAME {
            return Err(refuse!("bundle file name too long"));
        }
        slot.name[..name.len()].copy_from_slice(name.as_bytes());
        slot.name_bytes = name.len();
        slot.contents = *contents;
    }
    let file_count = bundle.len();
    // All stay allocated: the engine reads them through the physmap.
    core::mem::forget(options);
    core::mem::forget(source);
    core::mem::forget(bundle);
    Ok(RetainedFiles {
        options: options_region,
        source: source_region,
        files,
        file_count,
    })
}

pub(super) struct GuestMemory {
    pub ranges: [Region; MAX_GUEST_RANGES],
    pub count: usize,
    pub ledger: memory_map::Ledger,
}

/// Collect guest candidates from the final firmware map, by the same rules
/// as the loader's earlier budget.
pub(super) fn guest_memory(
    map: &impl MemoryMap,
    bundle_layout: &mut crate::handoff::BundleLayout,
) -> GuestMemory {
    let memory_map::Map {
        mut ranges,
        mut count,
        ledger,
    } = memory_map::summarize(map, None);
    // Inserting at increasing guest offsets keeps each earlier placement.
    let backing = &mut bundle_layout.backing[..bundle_layout.backing_count];
    backing.sort_unstable_by_key(|part| part.guest_offset);
    let mut adopted = 0;
    for index in 0..backing.len() {
        let part = backing[index];
        let name = if part.kernel_collection {
            "bootkc"
        } else {
            "ramdisk"
        };
        let Region { physical, bytes } = part.region;
        let Some(admitted) = stela_boot_core::ram::insert_backing(
            &mut ranges,
            count,
            part.guest_offset,
            (physical, bytes),
            LARGE,
        ) else {
            output::arguments(format_args!("HOST BUNDLE {name} backing kept as staging\n"));
            continue;
        };
        count = admitted;
        backing[adopted] = part;
        adopted += 1;
        output::arguments(format_args!(
            "HOST BUNDLE {name} adopted={bytes:#x} offset={:#x} physical={physical:#x}\n",
            part.guest_offset,
        ));
    }
    bundle_layout.backing_count = adopted;
    let mut guest = [Region::default(); MAX_GUEST_RANGES];
    for (region, &(physical, bytes)) in guest.iter_mut().zip(&ranges[..count]) {
        *region = Region { physical, bytes };
    }
    GuestMemory {
        ranges: guest,
        count,
        ledger,
    }
}

/// The live root, stack window and entry that receive the completed handoff.
pub(super) struct Transfer {
    pub placement: Placement,
    pub root: u64,
    pub trampoline: u64,
    pub engine_entry: u64,
    pub handoff: *mut Handoff,
}

impl Transfer {
    /// # Safety
    /// Boot services have ended. `root` maps this code and the trampoline,
    /// and the engine's image, stack, handoff and shared physmap.
    pub(super) unsafe fn enter(self) -> Result<core::convert::Infallible> {
        let Self {
            placement,
            root,
            trampoline,
            engine_entry,
            handoff,
        } = self;
        // SAFETY: the caller supplied live identity and engine-window mappings.
        unsafe {
            asm!("cli", options(nomem, nostack));
            paging::take_over()?;
            match placement {
                Placement::FiveLevel => la57::enable(root, trampoline),
                Placement::FourLevel { .. } => {
                    asm!("mov cr3, {}", in(reg) root, options(nostack, preserves_flags))
                }
            }
            asm!(
                "mov rsp, {stack}",
                "call {entry}",
                "ud2",
                stack = in(reg) layout::stack() + PAGE + STACK_BYTES,
                entry = in(reg) engine_entry,
                in("rdi") layout::physical(handoff as u64),
                options(noreturn),
            )
        }
    }
}
