//! MRS, MSR and System instructions through the generated AARCHMRS accessors
//! (`machine::system`): this module is the [`Context`] they run against.
//!
//! - Identification registers hold the CPU profile's values; one without a
//!   value refuses the read (`RequirementKind::SystemRegister`).
//! - SP_ELx are the banked stack pointers, CNTx_CTL compute ISTATUS, ISR_EL1
//!   reports the pending lines, and FPCR/FPSR live in the guest state.
//! - Writes the engine cannot honor refuse: debug exceptions and breakpoints,
//!   and Apple VMSA_LOCK. SCTLR_ELx.EE and E0E are RES0 because the profile
//!   has no FEAT_MixedEnd.
//!
//! A refusal is reported once and the register behaves as storage
//! (`Machine::register_refused`).
use super::{
    Devices, Discard, Host, Machine, Step,
    mmu::{
        self, Access as MemoryAccess, Outcome as Translated,
        tlbi::{self, Reach, Scope},
    },
};
use alloc::vec::Vec;
use core::cell::Cell;
use stela_compiler::aarch64::system::registers::{Encoding, RegisterAccess};
use stela_compiler::x86_64::cache_clean;
use {
    crate::machine::CpuProfile, crate::machine::amu, crate::machine::amu::Amu,
    crate::machine::cpu::Cpu, crate::machine::cpu::Exception, crate::machine::cpu::HCR_AMO,
    crate::machine::cpu::HCR_FMO, crate::machine::cpu::HCR_IMO, crate::machine::cpu::HCR_TGE,
    crate::machine::cpu::HCR_VF, crate::machine::cpu::HCR_VI, crate::machine::cpu::HCR_VSE,
    crate::machine::cpu::TranslationRegime, crate::machine::debug, crate::machine::debug::DCC_FULL,
    crate::machine::exception, crate::machine::exception::Class, crate::machine::exception::DAIF_A,
    crate::machine::exception::DAIF_D, crate::machine::exception::DAIF_F,
    crate::machine::exception::DAIF_I, crate::machine::exception::Pstate as PstateBits,
    crate::machine::pmu::Pmu, crate::machine::registers::Refusal,
    crate::machine::registers::trap_iss, crate::machine::system::ACCESSORS,
    crate::machine::system::AtAccess, crate::machine::system::CacheOp,
    crate::machine::system::CacheScope, crate::machine::system::CacheType,
    crate::machine::system::Context, crate::machine::system::Effect,
    crate::machine::system::ErrorRecord, crate::machine::system::File,
    crate::machine::system::Limit, crate::machine::system::Operation,
    crate::machine::system::Pstate, crate::machine::system::Register,
    crate::machine::system::Stage, crate::machine::system::Tlbi,
    crate::machine::system::helpers::sign_extend, crate::machine::timer,
    crate::machine::timer::Channel, crate::machine::timer::agt::Bank, stela_contracts::Chip,
    stela_contracts::Feature,
};
mod register_context;

/// The FPCR fields of every chip: AHP, DN, FZ, RMode and FZ16 (FEAT_FP16),
/// and the trap enables IOE, DZE, OFE, UFE, IXE and IDE, which an M1 Pro
/// keeps and traps on (measured at EL0 with `cargo xtask cpu-reference`;
/// XNU reports `hw.optional.arm.FP_SyncExceptions` 1 there). The engine
/// takes the trap (`Machine::take_floating_point_exception`). FEAT_EBF16 and AArch32's
/// Len and Stride are on no chip.
const FPCR_FIELDS: u32 = 0x07c8_0000 | FPCR_TRAP_ENABLES;
/// FPCR's floating-point exception trap enables: IOE, DZE, OFE, UFE, IXE
/// and IDE.
const FPCR_TRAP_ENABLES: u32 = 0x9f00;
/// Whether software can set every FPCR trap enable, the test XNU makes for
/// `hw.optional.arm.FP_SyncExceptions` (`commpage_init_arm_optional_features_fpcr`).
pub const FP_TRAPS_WRITABLE: bool = FPCR_FIELDS & FPCR_TRAP_ENABLES == FPCR_TRAP_ENABLES;
/// FIZ, AH and NEP (FEAT_AFP, M4 and later). The compiled FP operations take
/// their effects from the stored FPCR.
const FPCR_ALTERNATE: u32 = 0b111;

/// The FPCR fields the PE whose register file is `registers` holds.
fn fpcr_fields(registers: &File) -> u32 {
    if registers.has(Feature::FEAT_AFP) {
        FPCR_FIELDS | FPCR_ALTERNATE
    } else {
        FPCR_FIELDS
    }
}

/// Registers whose value is the CPU profile's, read-only to software.
pub(super) fn identification(register: Register) -> bool {
    let name = register.name();
    name.starts_with("ID_")
        || matches!(
            register,
            Register::MIDR_EL1
                | Register::MPIDR_EL1
                | Register::REVIDR_EL1
                | Register::AIDR_EL1
                | Register::CTR_EL0
                | Register::DCZID_EL0
                | Register::GMID_EL1
                | Register::CLIDR_EL1
                | Register::CCSIDR_EL1
                | Register::MVFR0_EL1
                | Register::MVFR1_EL1
                | Register::MVFR2_EL1
                | Register::LORID_EL1
                | Register::MPAMIDR_EL1
                | Register::ERRIDR_EL1
                | Register::AMCFGR_EL0
                | Register::AMCGCR_EL0
                | Register::AMCG1IDR_EL0
                | Register::DBGAUTHSTATUS_EL1
                | Register::MDRAR_EL1
                | Register::RVBAR_EL2
                | Register::SMIDR_EL1
        )
}

/// HCR_EL2.AMVOFFEN (FEAT_AMUv1p1): reads of the architected activity
/// monitor counters at EL0 and EL1 outside the host subtract the virtual
/// offsets.
const HCR_AMVOFFEN: u32 = 51;

/// ACTLR_ELx bit 1, which XNU `xnu-12377.121.6` `proc_reg.h` names EnTSO.
const ACTLR_EN_TSO: u64 = 1 << 1;
/// ACTLR_ELx bit 9, which the 26A434 context switch sets to a Rosetta
/// thread's flag while the `icdsb` boot argument is nonzero.
const ACTLR_ICDSB: u64 = 1 << 9;
/// ACTLR_ELx bit 12, "IC IVAU Enable ASID" (Asahi
/// `hw/cpu/system-registers.md`), which the M3 kernels' CPU initialization
/// sets with AHCR_EL2 |= 0x1e under a word that is 1 in the image
/// (kernelcache.release.mac15g 0xfffffe000bcfd13c..0xfffffe000bcfd17c).
const ACTLR_IC_IVAU_ASID: u64 = 1 << 12;

/// The ACTLR_ELx bits that refuse by name, from XNU `proc_reg.h`: controls
/// of Apple features this machine's AIDR_EL1 does not advertise. The 26A434
/// kernel sets them only in translated x86-64 threads, and it executes an
/// x86-64 image only when AIDR_EL1 advertises TSO and APFLG, so no
/// component sets them here.
const ACTLR_REFUSED: [(u64, &str); 3] = [
    (
        1 << 4,
        "ACTLR EnAPFLG: the Rosetta flag extension, which AIDR_EL1 here does not advertise (APFLG, bit 19)",
    ),
    (
        1 << 5,
        "ACTLR EnAFP: Apple's alternate floating-point mode, which AIDR_EL1 here does not advertise (bit 23)",
    ),
    (
        1 << 6,
        "ACTLR EnPRSV: a mode AIDR_EL1 here does not advertise (PSRV, bit 20)",
    ),
];

/// Why a write that sets these bits of a register whose whole contents Arm
/// leaves IMPLEMENTATION DEFINED stops, or `None` when it completes (also
/// for every other register). The 26A434 census gives each bit its
/// treatment:
///
/// - AFSRn_ELx report additional fault status. This machine records none,
///   so exceptions leave them unchanged, and every bit holds what software
///   writes: the SPTM copies the guarded level's AFSR1 into AFSR1_EL2 and
///   the hypervisor saves and restores AFSRn_EL12, both expecting the value
///   back.
/// - ACTLR_ELx bit 1, EnTSO, asks for total store order. x86 hosts give
///   every translated plain access total store order already, and the
///   ordered accesses, atomics and barriers are at least as strong, so the
///   bit needs no further effect. Bit 9 is held: XNU's context switch copies a Rosetta thread's
///   flag into it while the `icdsb` boot argument (1 by default) is
///   nonzero, and the name ties it to instruction-cache maintenance and
///   DSB. Whatever ordering it adds to them, this machine already completes
///   every IC invalidation before the next instruction, and its only
///   consumer, the 26A434 Rosetta runtime, invalidates the code it writes
///   with the full DSB ISH, IC IVAU, DSB ISH, ISB sequence (`runtime`
///   `__text+0x318`), so no consumer needs a further effect (inference).
///   Bit 12 is held too: by its name it makes IC IVAU match the current
///   ASID, which can only narrow what an invalidation removes, and this
///   machine's IC IVAU removes every translation of the address whatever
///   the ASID, which Arm always permits, as instruction cache entries may be
///   invalidated at any time (inference).
///   XNU sets EnTSO with bit 4 (EnAPFLG), and with bits 5 and 6 (EnAFP,
///   EnPRSV) when AIDR_EL1 bit 23 enables that static branch, before
///   returning to EL0 in a translated x86-64 thread. It creates such a
///   thread only after exec's Rosetta check passes, which needs AIDR_EL1's
///   TSO and APFLG bits, and this machine advertises neither: each refuses
///   by name ([`ACTLR_REFUSED`]), and no component sets any other bit, so
///   they stop.
/// - AMAIR_ELx refine MAIR_ELx's attributes, and Arm lets that refinement
///   have no effect. Stela's memory model gives memory attributes none
///   beyond what MAIR_ELx selects (every PE shares coherent host RAM),
///   so every bit holds what software writes: no component writes AMAIR_EL1
///   or AMAIR_EL2, and the hypervisor round-trips a guest's AMAIR_EL12.
/// - HACR_EL2 holds every bit but the guest timer mask
///   (`machine::hacr`), which decides what each bit traps.
fn implementation_defined_refusal(register: Register, value: u64) -> Option<&'static str> {
    const UNMODELED: &str = "an IMPLEMENTATION DEFINED register bit without a modelled effect";
    match register {
        Register::ACTLR_EL1 | Register::ACTLR_EL2 => ACTLR_REFUSED
            .iter()
            .find(|(bit, _)| value & bit != 0)
            .map(|&(_, reason)| reason)
            .or(
                (value & !(ACTLR_EN_TSO | ACTLR_ICDSB | ACTLR_IC_IVAU_ASID) != 0)
                    .then_some(UNMODELED),
            ),
        Register::HACR_EL2 => crate::machine::hacr::unsupported_bits(value),
        _ => None,
    }
}

/// The MRS encoding of a register, from its generated accessor.
pub(super) fn encoding_of(register: Register) -> Option<Encoding> {
    let name = register.name();
    let accessor = ACCESSORS.iter().find(|accessor| {
        accessor.name.strip_prefix("MRS ") == Some(name) && accessor.index.is_empty()
    })?;
    let key = accessor.value;
    Some(Encoding::new(
        (key >> 14) as u8,
        ((key >> 11) & 7) as u8,
        ((key >> 7) & 15) as u8,
        ((key >> 3) & 15) as u8,
        (key & 7) as u8,
    ))
}

/// DataCacheWatchpointSize: the block a DC IVAC is watched as.
const DC_WATCH_BYTES: u64 = 64;

/// Writes that change which pages watchpoints keep out of the shadow tables
/// (`machine::debug::watched`).
fn watches(register: Register) -> bool {
    matches!(
        register,
        Register::DBGWCR_EL1
            | Register::DBGWVR_EL1
            | Register::MDSCR_EL1
            | Register::MDCR_EL2
            | Register::OSLSR_EL1
    )
}

/// Writes that change only the translation of the lower half (TTBR0 of
/// either regime): the host keeps its shadow of the upper half.
fn table_base(register: Register) -> bool {
    matches!(register, Register::TTBR0_EL1 | Register::TTBR0_EL2)
}

/// Writes that change translation: shadow roots and fetch translations go.
fn translates(register: Register) -> bool {
    matches!(
        register,
        Register::SCTLR_EL1
            | Register::SCTLR_EL2
            | Register::TCR_EL1
            | Register::TCR_EL2
            | Register::TTBR0_EL1
            | Register::TTBR0_EL2
            | Register::TTBR1_EL1
            | Register::TTBR1_EL2
            | Register::MAIR_EL1
            | Register::MAIR_EL2
            | Register::HCR_EL2
            | Register::VTCR_EL2
            | Register::VTTBR_EL2
    )
}

/// Whether the LORegion registers are RES0: LORID_EL1.LD (bits 23:16)
/// reports no LORegion descriptor.
fn no_lor_descriptors(cpu: &crate::machine::cpu::Cpu) -> bool {
    cpu.field(Register::LORID_EL1, 16, 8) == 0
}

fn lor_region(register: Register) -> bool {
    matches!(
        register,
        Register::LORC_EL1 | Register::LORSA_EL1 | Register::LOREA_EL1 | Register::LORN_EL1
    )
}

/// MPAMn_ELx.MPAMEN. Without EL3, MPAM2_EL2's is read/write and MPAM1_EL1's
/// reads it and ignores writes.
const MPAMEN: u64 = 1 << 63;
/// RMR_EL2.RR, the Warm reset request.
const RMR_RR: u64 = 1 << 1;
const SCTLR_DZE: u32 = 14;
const HCR_TDZ: u32 = 28;
/// HCR_EL2.DC: EL1&0 stage 1 behaves as disabled with Normal output.
const HCR_DC: u32 = 12;
const DCZID_DZP: u64 = 1 << 4;

/// Whether DC ZVA traps at the current EL, which DCZID_EL0.DZP reports:
/// SCTLR_EL1.DZE and HCR_EL2.TDZ at EL0 outside the host, SCTLR_EL2.DZE in
/// it, and HCR_EL2.TDZ at EL1.
fn zero_block_prohibited(cpu: &crate::machine::cpu::Cpu) -> bool {
    match cpu.pstate.el {
        0 if cpu.in_host() => cpu.field(Register::SCTLR_EL2, SCTLR_DZE, 1) == 0,
        0 => cpu.field(Register::SCTLR_EL1, SCTLR_DZE, 1) == 0 || cpu.hcr(HCR_TDZ),
        1 => cpu.hcr(HCR_TDZ),
        _ => false,
    }
}
/// DCZID_EL0.BS: log2 of the DC ZVA block in words, at most 9 (2 KiB).
const DCZID_BS: u64 = 0xf;
/// HFGITR_EL2.DCZVA (FEAT_FGT).
const HFGITR_DCZVA: u32 = 11;

/// The DC ZVA block, in bytes, while DC ZVA at the current EL completes by
/// zeroing it: the generated `DC ZVA` accessor neither traps (SCTLR_EL1.DZE
/// and HCR_EL2.TDZ, and HFGITR_EL2.DCZVA with FEAT_FGT, outside the host at
/// EL0 and at EL1; SCTLR_EL2.DZE at EL0 in the host) nor stops, and
/// [`Access::zero_block`] accepts the profile's DCZID_EL0. Native code
/// completes DC ZVA only then (`x86_64::zero_block`), and only while stage 1
/// is enabled or HCR_EL2.DC makes EL1&0 data Normal: otherwise every data
/// access is Device-nGnRnE (Arm `AArch64_S1DisabledOutput`), where DC ZVA
/// takes an Alignment fault.
pub(super) fn native_zero_block(cpu: &Cpu, missing: &[Register]) -> Option<u16> {
    let fine_grained = || {
        let implemented = cpu.registers.implemented(Feature::FEAT_FGT).ok()?;
        Some(implemented && cpu.field(Register::HFGITR_EL2, HFGITR_DCZVA, 1) != 0)
    };
    let enabled = |register| cpu.field(register, SCTLR_DZE, 1) != 0;
    let permitted = match cpu.pstate.el {
        0 if cpu.in_host() => enabled(Register::SCTLR_EL2),
        0 => enabled(Register::SCTLR_EL1) && !cpu.hcr(HCR_TDZ) && !fine_grained()?,
        1 => !cpu.hcr(HCR_TDZ) && !fine_grained()?,
        2 => true,
        _ => false,
    };
    let normal = cpu.regime_sctlr() & mmu::SCTLR_M != 0
        || cpu.regime() == TranslationRegime::El10 && !cpu.in_host() && cpu.hcr(HCR_DC);
    if !permitted || !normal || missing.contains(&Register::DCZID_EL0) {
        return None;
    }
    let bytes = 4u64 << (cpu.read(Register::DCZID_EL0) & DCZID_BS);
    (bytes <= ZEROES.len() as u64).then_some(bytes as u16)
}
static ZEROES: [u8; 2048] = [0; 2048];

/// SCTLR_ELx.UCI: EL0 may execute the DC clean operations and IC IVAU.
const SCTLR_UCI: u32 = 26;
/// HCR_EL2.TPCP, TPU and TOCU: trap the clean operations to the Point of
/// Coherency (and Persistence), and to the Point of Unification.
const HCR_TPCP: u32 = 23;
const HCR_TPU: u32 = 24;
const HCR_TOCU: u32 = 52;
/// HFGITR_EL2.DCCVAC, DCCIVAC and DCCVAU (FEAT_FGT).
const HFGITR_DCCVAC: u32 = 54;
const HFGITR_DCCIVAC: u32 = 10;
const HFGITR_DCCVAU: u32 = 7;

/// The DC clean operations at the current EL that only check that Xt
/// translates for a read, as [`Access::operation`] completes them on the
/// coherent host: those the generated `DC CVAC`, `DC CVAU` and `DC CIVAC`
/// accessors neither trap (SCTLR_EL1.UCI, then HCR_EL2.TPCP, or TPU and
/// TOCU for DC CVAU, then the HFGITR_EL2 bit with FEAT_FGT, outside the
/// host at EL0 and, without UCI, at EL1; SCTLR_EL2.UCI at EL0 in the host)
/// nor treat as a NOP (`treat_dc_as_nop` never does). Native code completes
/// these by reading the byte at Xt (`x86_64::cache_clean`).
pub(super) fn native_cache_clean(cpu: &Cpu) -> cache_clean::Operations {
    let Ok(fine_grained) = cpu.registers.implemented(Feature::FEAT_FGT) else {
        return cache_clean::Operations::NONE;
    };
    let hcr = |bit| cpu.hcr(bit);
    let fgt = |bit| fine_grained && cpu.field(Register::HFGITR_EL2, bit, 1) != 0;
    let guest = |el0: bool| {
        let uci = !el0 || cpu.field(Register::SCTLR_EL1, SCTLR_UCI, 1) != 0;
        [
            (
                cache_clean::Operations::CVAC,
                !hcr(HCR_TPCP) && !fgt(HFGITR_DCCVAC),
            ),
            (
                cache_clean::Operations::CVAU,
                !hcr(HCR_TPU) && !hcr(HCR_TOCU) && !fgt(HFGITR_DCCVAU),
            ),
            (
                cache_clean::Operations::CIVAC,
                !hcr(HCR_TPCP) && !fgt(HFGITR_DCCIVAC),
            ),
        ]
        .into_iter()
        .filter(|&(_, untrapped)| uci && untrapped)
        .fold(cache_clean::Operations::NONE, |set, (operation, _)| {
            set.with(operation)
        })
    };
    match cpu.pstate.el {
        0 if cpu.in_host() => {
            if cpu.field(Register::SCTLR_EL2, SCTLR_UCI, 1) != 0 {
                cache_clean::Operations::ALL
            } else {
                cache_clean::Operations::NONE
            }
        }
        0 => guest(true),
        1 => guest(false),
        2 => cache_clean::Operations::ALL,
        _ => cache_clean::Operations::NONE,
    }
}
const ISR_A: u64 = 1 << 8;
const ISR_I: u64 = 1 << 7;
const ISR_F: u64 = 1 << 6;

/// One MRS, MSR, SYS or SYSL in progress.
pub(super) struct Access<'a, H: Host, D: Devices> {
    machine: &'a mut Machine<H, D>,
    rt: u8,
    /// One counter sample for all timing calculations of this access.
    now: Option<u64>,
    result: Option<u64>,
    translation: Translation,
    regime: bool,
    /// RMR_EL2.RR requested a Warm reset (`Machine::warm_reset`).
    reset: bool,
    code: Code,
    refusal: Cell<Option<&'static str>>,
    /// The storage of the plain read the accessor made last.
    plain: Cell<Option<(Register, u64)>>,
    /// The timer bank the generic timer registers name (`timer::agt`).
    bank: Bank,
}

/// The cached translations an access makes stale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Translation {
    Unchanged,
    /// A TTBR0 write: the lower half's ([`table_base`]).
    TableBase,
    /// Another translation control: all.
    All,
    /// TLB maintenance: every fetch translation, of the shadow what the
    /// TLBI reaches, and of the lookup table what it reaches everywhere;
    /// and the scope other PEs apply, for one broadcast beyond this PE.
    Maintenance(Reach, tlbi::Everywhere, Option<u16>, Option<tlbi::Scope>),
}

/// HCR_EL2.FB: EL1 TLBIs broadcast Inner Shareable.
const HCR_FB: u32 = 9;

/// What `tlbi` removes from the shadow of `cpu`'s current context, and
/// from a cache of every context's translations. EL2 is enabled and
/// HCR_EL2.E2H is RES1 (`cpu`).
fn reach(cpu: &Cpu, tlbi: &Tlbi) -> (Reach, tlbi::Everywhere, Option<u16>, Option<tlbi::Scope>) {
    let context = tlbi::Context {
        el: cpu.pstate.el,
        el2: true,
        e2h: true,
        tge: cpu.hcr(HCR_TGE),
        fb: cpu.hcr(HCR_FB),
    };
    let scope = Scope::new(tlbi, context);
    let broadcast = super::parallel::broadcasts(&scope).then_some(scope);
    (
        scope.reach(&cpu.tlbi_current()),
        scope.everywhere(),
        scope.named_asid(),
        broadcast,
    )
}

/// The compiled code an instruction-cache invalidation makes stale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Code {
    Unchanged,
    /// IC IVAU of `address`, in the page holding `physical`.
    Page {
        address: u64,
        physical: u64,
    },
    /// IC IALLU and IC IALLUIS.
    All,
}

impl<H: Host, D: Devices> Access<'_, H, D> {
    fn now(&mut self) -> u64 {
        *self.now.get_or_insert_with(|| self.machine.now())
    }

    fn refuse(&self, reason: &'static str) {
        if self.refusal.get().is_none() {
            self.refusal.set(Some(reason));
        }
    }

    /// Arm `NVMem`: the 64-bit load or store that a register access at EL1
    /// becomes under HCR_EL2.NV2, at `SignExtend(VNCR_EL2.BADDR :: Zeros(12))
    /// + offset`. `EffectiveBADDR` makes bits 63:49 copy bit 48 (no
    /// FEAT_LVA; the RESS constraint). It is an EL2 access without PAN
    /// through the EL2&0 regime (`CreateAccDescNV2`). A translation fault is
    /// an NV2 Data Abort and a watchpoint an NV2 Watchpoint: the same-EL
    /// classes with ISS.VNCR (bit 13) set, taken to EL2 (Arm
    /// `AArch64_DataAbort`, `AArch64_FaultSyndrome`).
    fn nested_access(
        &mut self,
        offset: u64,
        write: bool,
        bytes: &mut [u8; 8],
    ) -> Result<(), Effect> {
        const VNCR: u32 = 1 << 13;
        let base = self.machine.cpu.read(Register::VNCR_EL2) & !0xfff;
        let address = sign_extend(base, 49).wrapping_add(offset);
        let pc = self.machine.cpu.pc;
        let watched = debug::Access {
            address,
            size: 8,
            read: !write,
            write,
            maintenance: false,
            el: 2,
        };
        if let Some(mut exception) = self.machine.watchpoint_exception(&watched, pc) {
            let iss = exception.syndrome as u32 & 0x01ff_ffff | VNCR;
            exception.syndrome = exception::syndrome(Class::WatchpointSame, iss);
            return Err(Effect::Exception(exception.to(2)));
        }
        let requester = mmu::Requester {
            el: 2,
            pan: false,
            unprivileged: false,
            guarded: false,
        };
        let access = if write {
            MemoryAccess::Write
        } else {
            MemoryAccess::Read
        };
        match self.translate(TranslationRegime::El20, requester, address, access) {
            Translated::Mapped(translation) => {
                let done = if write {
                    self.machine.host.write_ram(translation.physical, bytes)
                } else {
                    self.machine.host.read_ram(translation.physical, bytes)
                };
                if done {
                    Ok(())
                } else {
                    Err(Effect::Refused("a VNCR_EL2 access outside RAM"))
                }
            }
            Translated::Fault(fault) => {
                let iss = fault.data_syndrome(write, false) | VNCR;
                let mut exception = Exception::synchronous(Class::DataAbortSame, iss, pc).to(2);
                exception.fault_address = Some(address);
                Err(Effect::Exception(exception))
            }
            Translated::Unsupported(_) => Err(Effect::Refused(
                "a VNCR_EL2 access needs an unsupported translation",
            )),
        }
    }
}

impl<H: Host, D: Devices> Access<'_, H, D> {
    /// AMEVCNTR0<n>_EL0, or the group 0 enables AMCNTENSET0_EL0 and
    /// AMCNTENCLR0_EL0 report. With FEAT_AMUv1p1 and HCR_EL2.AMVOFFEN, a
    /// counter read at EL0 or EL1 outside the host returns the physical
    /// count less its virtual offset: AMEVCNTVOFF0<n>_EL2, or CNTVOFF_EL2
    /// for the constant frequency counter, which has no offset register.
    fn read_activity_monitor(&mut self, register: Register, index: u64) -> u64 {
        let now = if register == Register::AMEVCNTR0_EL0 {
            self.now()
        } else {
            0
        };
        let cpu = &self.machine.cpu;
        let Some(monitors) = cpu.amu.as_ref() else {
            self.refuse("activity monitors without their model");
            return 0;
        };
        if register != Register::AMEVCNTR0_EL0 {
            return monitors.enabled();
        }
        let count = monitors.count(index as usize, now, cpu.retired);
        let virtual_view = cpu.registers.has(Feature::FEAT_AMUv1p1)
            && cpu.pstate.el < 2
            && !cpu.in_host()
            && cpu.hcr(HCR_AMVOFFEN);
        if !virtual_view {
            return count;
        }
        let offset = if index == 1 {
            cpu.registers.get(Register::CNTVOFF_EL2, 0)
        } else {
            cpu.registers.get(Register::AMEVCNTVOFF0_EL2, index)
        };
        count.wrapping_sub(offset)
    }

    /// The group 0 counters and enables, which the activity monitors hold,
    /// and the group 1 enables, RAZ/WI with no group 1 counter; `None` for
    /// every other register.
    fn write_activity_monitor(
        &mut self,
        register: Register,
        index: u64,
        value: u64,
    ) -> Option<Result<(), Effect>> {
        let set = match register {
            Register::AMCNTENSET1_EL0 | Register::AMCNTENCLR1_EL0 => return Some(Ok(())),
            Register::AMCNTENSET0_EL0 => Some(true),
            Register::AMCNTENCLR0_EL0 => Some(false),
            Register::AMEVCNTR0_EL0 => None,
            _ => return None,
        };
        let (now, retired) = (self.now(), self.machine.cpu.retired);
        let Some(monitors) = self.machine.cpu.amu.as_mut() else {
            return Some(Err(Effect::Refused(
                "activity monitors without their model",
            )));
        };
        match set {
            Some(set) => monitors.set_enables(value, set, now, retired),
            None => monitors.set_count(index as usize, value, now, retired),
        }
        Some(Ok(()))
    }

    /// ISR_EL1: virtual interrupts while routed to EL2 from EL1 or EL0
    /// outside the host, physical lines otherwise.
    fn pending(&mut self) -> u64 {
        let lines = self.machine.lines();
        let cpu = &self.machine.cpu;
        let guest = cpu.pstate.el != 2 && !cpu.in_host();
        let line = |routing: u32, virtual_bit: u32, physical: bool| {
            if guest && cpu.hcr(routing) {
                cpu.virtual_pending(virtual_bit)
            } else {
                physical
            }
        };
        (if line(HCR_IMO, HCR_VI, lines.irq) {
            ISR_I
        } else {
            0
        }) | (if line(HCR_FMO, HCR_VF, lines.fiq) {
            ISR_F
        } else {
            0
        }) | (if line(HCR_AMO, HCR_VSE, false) {
            ISR_A
        } else {
            0
        })
    }

    fn translate(
        &mut self,
        regime: TranslationRegime,
        requester: mmu::Requester,
        address: u64,
        access: MemoryAccess,
    ) -> Translated {
        let controls = self.machine.cpu.controls(regime);
        mmu::translate(
            &controls,
            address,
            access,
            requester,
            &mut super::RamTables(&mut self.machine.host),
        )
    }

    /// The physical address cache maintenance by `address` acts on, or the
    /// fault its translation takes.
    fn check_maintenance(&mut self, address: u64, write: bool) -> Result<u64, Effect> {
        let regime = self.machine.cpu.regime();
        let requester = self.machine.cpu.requester();
        let access = if write {
            MemoryAccess::Write
        } else {
            MemoryAccess::Read
        };
        match self.translate(regime, requester, address, access) {
            Translated::Mapped(translation) => Ok(translation.physical),
            Translated::Fault(fault) => Err(Effect::Exception(
                self.machine
                    .data_abort_exception(address, fault, true, true),
            )),
            Translated::Unsupported(_) => Err(Effect::Refused(
                "cache maintenance address needs an unsupported translation",
            )),
        }
    }

    fn zero_block(&mut self, address: u64) -> Result<(), Effect> {
        let dczid = self.machine.cpu.read(Register::DCZID_EL0);
        if self.machine.profile_missing.contains(&Register::DCZID_EL0) {
            return Err(Effect::Refused("DC ZVA without a profile DCZID_EL0"));
        }
        let bytes = 4u64 << (dczid & DCZID_BS);
        if bytes > ZEROES.len() as u64 {
            return Err(Effect::Refused("DCZID_EL0.BS above Arm's 2 KiB maximum"));
        }
        let start = address & !(bytes - 1);
        let controls = self.machine.cpu.controls(self.machine.cpu.regime());
        let requester = self.machine.cpu.requester();
        // DC ZVA to Device memory takes an Alignment fault at the stage that
        // makes it Device (Arm `AArch64_S1HasAlignmentFaultDueToMemType` and
        // the stage-2 equivalent), with stage 1 disabled too.
        let shape = mmu::alignment::Shape {
            kind: mmu::alignment::Kind::Zero,
            bytes,
            pair: false,
        };
        let outcome = mmu::translate_checked(
            &controls,
            start,
            MemoryAccess::Write,
            requester,
            shape,
            &mut super::RamTables(&mut self.machine.host),
        );
        match outcome {
            Translated::Mapped(translation) => {
                let access = debug::Access {
                    address: start,
                    size: bytes,
                    read: false,
                    write: true,
                    maintenance: false,
                    el: self.machine.cpu.pstate.el,
                };
                if let Some(exception) = self
                    .machine
                    .watchpoint_exception(&access, self.machine.cpu.pc)
                {
                    return Err(Effect::Exception(exception));
                }
                // DC ZVA is tag checked (FEAT_MTE2); its fault reports Xt.
                match self.machine.tag_check(&access, requester, Some(address)) {
                    Ok(super::tags::TagCheck::Passed | super::tags::TagCheck::Untranslated) => {}
                    Ok(super::tags::TagCheck::Failed(exception)) => {
                        return Err(Effect::Exception(exception));
                    }
                    Err(_) => {
                        return Err(Effect::Refused(
                            "DC ZVA tag check needs an unsupported translation",
                        ));
                    }
                }
                if self
                    .machine
                    .host
                    .write_ram(translation.physical, &ZEROES[..bytes as usize])
                {
                    Ok(())
                } else {
                    Err(Effect::Refused("DC ZVA outside RAM"))
                }
            }
            Translated::Fault(fault) => Err(Effect::Exception(
                self.machine
                    .data_abort_exception(address, fault, true, false),
            )),
            Translated::Unsupported(_) => Err(Effect::Refused(
                "DC ZVA address needs an unsupported translation",
            )),
        }
    }

    /// `AArch64_AT`: S1E1 from the host becomes S1E2; S1E0 translates as
    /// unprivileged and S1E1R/W ignore PAN, S1E1RP/WP honor it. S1E* stops at
    /// the IPA, though its walk goes through stage 2; S12E* translates both
    /// stages. PAR_EL1 takes the result, except that an external abort, or a
    /// stage-2 fault on the walk of an AT executed at EL1, is taken as a
    /// Data Abort with CM set.
    fn address_translation(
        &mut self,
        stage: Stage,
        el: u64,
        access: AtAccess,
        address: u64,
    ) -> Result<(), Effect> {
        let cpu = &self.machine.cpu;
        let el = if el == 1 && cpu.in_host() && stage == Stage::S1 {
            2
        } else {
            el
        };
        let regime = if el == 2 || (el == 0 && cpu.in_host()) {
            TranslationRegime::El20
        } else {
            TranslationRegime::El10
        };
        let requester = mmu::Requester {
            el: if el == 0 { 1 } else { el as u8 },
            pan: matches!(access, AtAccess::ReadPAN | AtAccess::WritePAN) && cpu.pstate.pan,
            unprivileged: el == 0,
            guarded: cpu.pstate.guarded,
        };
        let memory = if matches!(access, AtAccess::Write | AtAccess::WritePAN) {
            MemoryAccess::Write
        } else {
            MemoryAccess::Read
        };
        let controls = self.machine.cpu.controls(regime);
        let tables = &mut super::RamTables(&mut self.machine.host);
        let outcome = match stage {
            Stage::S1 => mmu::translate_stage1(&controls, address, memory, requester, tables),
            Stage::S12 => mmu::translate_tables(&controls, address, memory, requester, tables),
        };
        if let Translated::Fault(fault) = outcome {
            let walk = fault.second_stage.is_some_and(|second| second.walk);
            if fault.kind == mmu::FaultKind::ExternalOnWalk
                || walk && self.machine.cpu.pstate.el == 1
            {
                return Err(Effect::Exception(
                    self.machine
                        .data_abort_exception(address, fault, true, true),
                ));
            }
        }
        let Some(par) = mmu::par(&outcome) else {
            return Err(Effect::Refused("AT needs an unsupported translation"));
        };
        self.machine.cpu.registers.set(Register::PAR_EL1, 0, par);
        Ok(())
    }
}

impl<H: Host, D: Devices> Machine<H, D> {
    /// Execute MRS, MSR, SYS or SYSL through the generated accessor for its
    /// encoding. `None` when no architectural accessor matches.
    pub(super) fn system_access(&mut self, word: u32, access: RegisterAccess) -> Option<Step> {
        let bank = timer::agt::selected(&self.cpu);
        self.system_access_as(word, access, access.encoding(), bank)
    }

    /// [`Self::system_access`] through the accessor of `encoding` on the
    /// timer bank `bank`, for an Apple view of an architectural register
    /// (`registers::Outcome::View`); a trap still reports the encoding
    /// `access` names.
    pub(super) fn system_access_as(
        &mut self,
        word: u32,
        access: RegisterAccess,
        encoding: Encoding,
        bank: Bank,
    ) -> Option<Step> {
        let read =
            access.direction() == stela_compiler::aarch64::system::registers::Direction::Read;
        let (accessor, index) = self.lookup.find(encoding, read)?;
        let mut context = Access {
            machine: self,
            rt: access.rt(),
            now: None,
            result: None,
            translation: Translation::Unchanged,
            regime: false,
            reset: false,
            code: Code::Unchanged,
            refusal: Cell::new(None),
            bank,
            plain: Cell::new(None),
        };
        let effect = match (accessor.run)(&mut context, index) {
            Ok(effect) | Err(effect) => effect,
        };
        let Access {
            result,
            translation,
            regime,
            reset,
            code,
            refusal,
            plain,
            ..
        } = context;
        self.plain_read = plain.into_inner();
        let refusal = refusal.into_inner().or(match effect {
            Effect::Refused(reason) => Some(reason),
            _ => None,
        });
        match translation {
            Translation::All => self.controls_changed(),
            Translation::TableBase => self.table_base_changed(),
            Translation::Maintenance(reach, everywhere, asid, broadcast) => {
                self.tlb_maintained(reach, everywhere, asid);
                if let Some(scope) = broadcast {
                    self.post(super::parallel::Maintenance::Tlb(scope));
                }
            }
            Translation::Unchanged if regime => self.sync_regime(),
            Translation::Unchanged => {}
        }
        match code {
            Code::Unchanged => {}
            Code::Page { address, physical } => {
                self.invalidate_code_page(physical);
                self.host.code_invalidated(Some(physical));
                self.post_code_page(physical);
                self.mirrors.open_code_page(address);
            }
            Code::All => {
                self.discard_code(Discard::Maintenance);
                self.host.code_invalidated(None);
                self.post(super::parallel::Maintenance::Code(None));
            }
        }
        // The state the accessor changed before refusing is resynchronized
        // above, whichever mode then answers the refusal.
        if let Some(reason) = refusal {
            return Some(self.register_refused(
                word,
                Some(access),
                Refusal::UnsupportedSetting(reason),
            ));
        }
        Some(match effect {
            Effect::Done => {
                match result {
                    Some(value) => {
                        let value = self.degraded_read(access, value);
                        if access.rt() != 31 {
                            self.cpu.state.x[usize::from(access.rt())] = value;
                        }
                    }
                    None => self.degraded_written(access),
                }
                if reset {
                    return Some(self.warm_reset());
                }
                self.next()
            }
            Effect::Undefined => self.undefined(),
            Effect::Trap { el, ec } => {
                // The generated SME traps are access traps, SMTC 0.
                let iss = match ec {
                    ec if ec == Class::SystemRegister as u64 => trap_iss(access),
                    ec if ec == Class::FloatingPointAccess as u64 => super::TRAPPED_AARCH64,
                    _ => 0,
                };
                let exception = Exception {
                    syndrome: (ec << 26) | (1 << 25) | u64::from(iss),
                    ..Exception::synchronous(Class::Unknown, 0, self.cpu.pc)
                }
                .to(el);
                self.take(exception);
                Ok(())
            }
            Effect::Exception(exception) => {
                self.take(exception);
                Ok(())
            }
            Effect::Refused(_) => unreachable!("handled above"),
        })
    }

    /// Fill the identification registers of the running PE from the CPU
    /// profile.
    pub(super) fn load_profile(&mut self) {
        identify(
            &self.profile,
            self.current,
            &mut self.cpu,
            &mut self.profile_missing,
        );
    }
}

/// Fill `cpu`'s identification registers with the profile's values for PE
/// number `pe` and hand VPIDR_EL2 and VMPIDR_EL2 the physical values, as boot
/// firmware does. `missing` collects the registers the profile has no value
/// for.
pub(super) fn identify(
    profile: &CpuProfile,
    pe: usize,
    cpu: &mut Cpu,
    missing: &mut Vec<Register>,
) {
    cpu.present(crate::machine::identification::withheld(
        profile.chip,
        &profile.fixed,
    ));
    // From the M3 on, CNTFRQ_EL0 of the Arm bank reads its scaled
    // frequency and the Apple bank counts the profile's (`timer::agt`).
    let frequency = if timer::agt::present(profile.chip) {
        cpu.apple
            .generic_timer
            .set_timebase(profile.counter_frequency);
        timer::agt::SCALED_HZ
    } else {
        profile.counter_frequency
    };
    cpu.registers.set_raw(Register::CNTFRQ_EL0, 0, frequency);
    let own = profile.processors.get(pe);
    cpu.pmu = own
        .and_then(|identity| identity.clock)
        .and_then(|clock| Pmu::for_chip(profile.chip, clock, profile.counter_frequency));
    cpu.aon = own
        .and_then(|identity| identity.clock)
        .and_then(|clock| crate::machine::aon::CoreActivity::new(clock, profile.counter_frequency));
    // The activity monitors count the board's core clock, or the counter's
    // frequency where the board gives none (`machine::amu`).
    cpu.amu = (cpu.registers.implemented(Feature::FEAT_AMUv1) == Ok(true)).then(|| {
        let clock = own.and_then(|identity| identity.clock);
        Amu::new(
            clock.unwrap_or(profile.counter_frequency),
            profile.counter_frequency,
        )
    });
    cpu.caches = own.and_then(|identity| identity.caches);
    for register in Register::ALL {
        if !identification(register) {
            continue;
        }
        let value = match (register, own) {
            (Register::MIDR_EL1, Some(identity)) => Some(identity.midr),
            (Register::MPIDR_EL1, Some(identity)) => Some(identity.mpidr),
            // CCSIDR_EL1 reads follow CSSELR_EL1 (`Access::read`).
            (Register::CLIDR_EL1, Some(identity)) if identity.caches.is_some() => {
                identity.caches.map(|caches| caches.clidr())
            }
            (Register::CCSIDR_EL1, Some(identity)) if identity.caches.is_some() => {
                identity.caches.map(|caches| caches.ccsidr(0))
            }
            _ => encoding_of(register).and_then(|encoding| {
                profile
                    .fixed
                    .iter()
                    .find(|(candidate, _)| *candidate == encoding)
                    .map(|(_, value)| *value)
            }),
        };
        match value {
            Some(value) => cpu.registers.set_raw(register, 0, value),
            // A layout the profile reserves entirely, such as the AArch32
            // ID registers without FEAT_AA32, has one value.
            None if cpu.registers.res0(register) | cpu.registers.res1(register) == u64::MAX => {
                let res1 = cpu.registers.res1(register);
                cpu.registers.set_raw(register, 0, res1);
            }
            None => missing.push(register),
        }
    }
    if cpu.registers.has(Feature::FEAT_SME) && !missing.contains(&Register::MPIDR_EL1) {
        let smidr = crate::machine::streaming::smidr(cpu.read(Register::MPIDR_EL1));
        cpu.registers.set_raw(Register::SMIDR_EL1, 0, smidr);
        missing.retain(|&register| register != Register::SMIDR_EL1);
    }
    for (virtual_register, physical) in [
        (Register::VPIDR_EL2, Register::MIDR_EL1),
        (Register::VMPIDR_EL2, Register::MPIDR_EL1),
    ] {
        if missing.contains(&physical) {
            missing.push(virtual_register);
        } else {
            let value = cpu.read(physical);
            cpu.registers.set_raw(virtual_register, 0, value);
        }
    }
}

#[cfg(test)]
mod tests;
