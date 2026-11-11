//! The translation helper that blocks of the translated variant call for
//! each access outside the host's native ranges. It runs inside native code, as
//! the #PF handler does: it looks the access up in the software TLB
//! (`mmu::translated`), fills a missing page from the same software walk of
//! the guest's tables as a shadow fill ([`super::resolve`]), and returns
//! where the host reaches the access, at the guest address's offset in its
//! page, through the physmap or, across discontiguous pages, through its
//! two-page bounce slot. A guest fault, a device, a watched or tag-checked
//! page ends the block precisely, as an unresolved #PF does; a guest fault
//! leaves the instruction to the engine, which takes an Alignment fault
//! first.
//!
//! The call (`stela_compiler::x86_64`'s translated access stub) passes the
//! guest address in RCX and the access's size and flags in the two bytes
//! after its CALL, returns past them with the host address in RAX, and
//! keeps every other register, XMM registers and MXCSR included.
use super::{Contexts, FaultCause, Recorded, State, flags_pan, layout, resolve, state, tables};
use core::arch::global_asm;
use stela_engine::machine::mmu::translated::Lookup;

/// The access's flags after its size.
const WRITE: u8 = 1 << 0;
const UNPRIVILEGED: u8 = 1 << 1;
const ATOMIC: u8 = 1 << 2;
/// Fills one access can need: one per page, and a retry.
const FILLS: usize = 3;

/// The registers at the call, RAX through R15 by x86 register number apart
/// from RSP (which is just above `ret`), and the call's return address.
#[repr(C)]
struct Saved {
    rax: u64,
    rcx: u64,
    rdx: u64,
    rbx: u64,
    rbp: u64,
    rsi: u64,
    rdi: u64,
    r8: u64,
    r9: u64,
    r10: u64,
    r11: u64,
    r12: u64,
    r13: u64,
    r14: u64,
    r15: u64,
    ret: u64,
}

// This helper image is separate from the original host image kept by
// stela_enter_block. Normal return restores the guest image; abandon
// restores the original host image through the one real recovery continuation.
#[repr(C)]
struct HelperFooter {
    xstate: super::super::xstate::EntryMetadata,
    saved_stack: u64,
    result: u64,
}

global_asm!(
    r#"
.global stela_translation_helper
stela_translation_helper:
    push r15
    push r14
    push r13
    push r12
    push r11
    push r10
    push r9
    push r8
    push rdi
    push rsi
    push rbp
    push rbx
    push rdx
    push rcx
    push rax
    mov rbx, rsp
    cmp qword ptr gs:[{recovery}], 0
    je {refuse}
    cmp qword ptr gs:[{helper}], 0
    jne {refuse}
    mov r11, gs:[{image_bytes}]
    sub rsp, r11
    sub rsp, {footer_bytes}
    and rsp, -64
    lea rsi, [rsp + r11]
    mov [rsi + {saved_stack}], rbx
    mov rdi, rsp
    call {save}
    mov gs:[{helper}], rsp
    cld
    fninit
    ldmxcsr [rip + {default_mxcsr}]
    mov rdi, rbx
    call {translate}
    mov r11, gs:[{image_bytes}]
    mov [rsp + r11 + {result}], rax
    mov rdi, rsp
    lea rsi, [rsp + r11]
    call {restore}
    mov r11, gs:[{image_bytes}]
    mov rax, [rsp + r11 + {result}]
    mov rbx, [rsp + r11 + {saved_stack}]
    mov qword ptr gs:[{helper}], 0
    mov rsp, rbx
    add rsp, 8
    pop rcx
    pop rdx
    pop rbx
    pop rbp
    pop rsi
    pop rdi
    pop r8
    pop r9
    pop r10
    pop r11
    pop r12
    pop r13
    pop r14
    pop r15
    // Past the size and flags; the caller's exact Saved layout is unchanged.
    add qword ptr [rsp], 2
    ret
"#,
    recovery = const super::local::RECOVERY_RSP_OFFSET,
    helper = const super::local::HELPER_RSP_OFFSET,
    image_bytes = const super::local::XSTATE_IMAGE_BYTES_OFFSET,
    footer_bytes = const size_of::<HelperFooter>(),
    saved_stack = const core::mem::offset_of!(HelperFooter, saved_stack),
    result = const core::mem::offset_of!(HelperFooter, result),
    save = sym super::super::xstate::stela_xstate_save_owned,
    restore = sym super::super::xstate::stela_xstate_restore_owned,
    refuse = sym super::super::xstate::stela_xstate_boundary_refuse,
    default_mxcsr = sym super::super::xstate::DEFAULT_MXCSR,
    translate = sym translate,
);

unsafe extern "sysv64" {
    fn stela_translation_helper();
}

/// The slot whose address the translated variant's blocks call through.
static HELPER: unsafe extern "sysv64" fn() = stela_translation_helper;

/// The address of the 8 bytes holding the helper's entry point.
pub(super) fn slot() -> u64 {
    &raw const HELPER as u64
}

/// Two window pages whose leaf entries the helper points at the two guest
/// frames of an access across discontiguous pages, so that the access
/// reaches the guest's own memory.
pub(super) struct Bounce {
    /// The first page's address.
    pub(super) address: u64,
    /// The physical addresses of the two pages' leaf entries.
    pub(super) entries: [u64; 2],
    /// The host frame each page maps now.
    mapped: [u64; 2],
}

impl Bounce {
    pub(super) fn new(address: u64, entries: [u64; 2]) -> Self {
        Self {
            address,
            entries,
            mapped: [u64::MAX; 2],
        }
    }

    /// Drop both local aliases. The caller flushes this CPU's contexts.
    pub(super) fn revoke(&mut self) {
        for entry in self.entries {
            // SAFETY: these are the worker's own window leaves, through the
            // physmap. The worker is held outside the translation helper.
            unsafe { (layout::physical(entry) as *mut u64).write_volatile(0) };
        }
        self.mapped = [u64::MAX; 2];
    }

    /// Map host frames `first` and `second` at the slot, which every view
    /// shares and `current` is loaded; its address.
    fn map(&mut self, first: u64, second: u64, contexts: &mut Contexts, current: usize) -> u64 {
        for (index, frame) in [first, second].into_iter().enumerate() {
            if self.mapped[index] != frame {
                let entry = layout::physical(self.entries[index]) as *mut u64;
                // SAFETY: a leaf entry of the window's own tables, through
                // the physmap; only this helper maps the slot.
                unsafe {
                    entry.write_volatile(frame | tables::PRESENT | tables::WRITABLE | tables::NX)
                };
                contexts.forget(self.address + index as u64 * tables::PAGE, current);
                self.mapped[index] = frame;
            }
        }
        self.address
    }
}

/// The helper's Rust side: the host address of the access the saved
/// registers' call describes, or no return at all.
extern "sysv64" fn translate(saved: &mut Saved) -> u64 {
    let state = state();
    let address = saved.rcx;
    // SAFETY: the two bytes the stub placed after its CALL, in the arena.
    let [bytes, flags] = unsafe { (saved.ret as *const [u8; 2]).read_unaligned() };
    let write = flags & WRITE != 0;
    // Even a warm direct or split hit may only use aliases from the
    // admitted execution generation.
    if state.native_ram().is_err() {
        abandon(state, saved, address, write, FaultCause::RamBinding);
    }
    state.follow_native();
    state.follow_pan(flags_pan());
    let view = state.current as u8;
    let current = state.root_of(state.current);
    state.helper_calls += 1;
    for _ in 0..FILLS {
        match state.tlb.lookup(address, u64::from(bytes), write, view) {
            Lookup::Direct(host) => return layout::physical(host),
            // A page-crossing atomic crosses its 16-byte granule too: the
            // engine takes its Alignment fault.
            Lookup::Split { .. } if flags & ATOMIC != 0 => {
                abandon(state, saved, address, write, FaultCause::InvalidOpcode)
            }
            Lookup::Split { first, second } => {
                let Some(bounce) = state.bounce.as_mut() else {
                    abandon(state, saved, address, write, FaultCause::NonCanonical)
                };
                state.bounces += 1;
                let slot = bounce.map(first, second, &mut state.contexts, current);
                return slot + (address & (tables::PAGE - 1));
            }
            Lookup::Miss { address: missing } => {
                state.helper_fills += 1;
                match resolve(state, missing, write, flags & UNPRIVILEGED != 0) {
                    Ok((translation, host)) => {
                        state.legacy_translations_dirty = true;
                        state.tlb.fill(missing, host, translation.writable, view)
                    }
                    // The stub runs before the block's alignment check, and
                    // Arm takes an Alignment fault before a translation
                    // fault: the engine checks the instruction's accesses
                    // in that order.
                    Err(FaultCause::Guest(_)) => {
                        abandon(state, saved, missing, write, FaultCause::NonCanonical)
                    }
                    Err(cause) => abandon(state, saved, missing, write, cause),
                }
            }
        }
    }
    // A fill the next lookup does not find: the walk and the TLB disagree.
    abandon(state, saved, address, write, FaultCause::NonCanonical)
}

/// End the native block at the call and report `cause` to the engine, as
/// an unresolved fault does; RAX and RCX are the stub's scratch.
fn abandon(state: &mut State, saved: &Saved, address: u64, write: bool, cause: FaultCause) -> ! {
    let rsp = &raw const saved.ret as u64 + 8;
    state.recorded = Some(Recorded {
        rip: saved.ret,
        origin: stela_engine::machine::NativeFaultOrigin::TranslatedHelper,
        trap: 0,
        error_code: 0,
        address,
        write,
        cause,
        registers: stela_compiler::block::HostGprSnapshot::legacy([
            saved.rax, saved.rcx, saved.rdx, saved.rbx, rsp, saved.rbp, saved.rsi, saved.rdi,
            saved.r8, saved.r9, saved.r10, saved.r11, saved.r12, saved.r13, saved.r14, saved.r15,
        ]),
    });
    // SAFETY: the recovery path restores the stack `stela_enter_block` left
    // and returns from it; nothing on this stack is used again.
    unsafe { core::arch::asm!("jmp {}", sym super::stela_native_recover, options(noreturn)) }
}
