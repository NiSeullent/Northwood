//! Apple implementation-defined registers that the kernelcaches,
//! SPTMs and TXM of the Apple silicon chips access, each listed with its
//! consumer and the behavior that consumer needs. Registers with an
//! architectural or modeled effect have their own owners (`machine::registers`,
//! `pmu`, `amx`, `ctrr` and the board's fast-IPI registers); every encoding
//! neither lists refuses when a censused component accesses it, and is
//! UNDEFINED when none does.
//!
//! - `Storage`: software saves and restores the register around its own
//!   changes, or writes it and never reads it, so reads must return the last
//!   value written. A read before the first write refuses, because the reset
//!   value is unknown.
//! - `Tuning`: a core tuning ("chicken bit") register, named by m1n1
//!   `cpu_regs.h` or used only in the same boot-time pattern, or a
//!   performance hint to the core's predictors. Software reads it, sets or
//!   clears bits and writes it back, or saves and restores it.
//!   The reset value is unknown and taken as 0 (inference). No bit has an
//!   effect here.
//! - `Prediction`: source-checked predictor configuration. It preserves only
//!   the named fields and refuses unknown bits before modifying state. A
//!   prediction hint affects neither architectural results nor access permissions;
//!   no speculative timing model is claimed. Reset 0 remains an inference.
//! - `Scratch`: a register software keeps its own records in, with no
//!   hardware effect. Reads return the last value written, 0 from cold
//!   reset (inference).
//! - `Control`: a control register whose bits select features or events
//!   this machine does not model. It reads 0 from reset (inference) and
//!   takes only the entry's permitted bits (none for most); any other set
//!   bit stops, as for ACTLR_EL1.
//! - `Status`: the status of an event this machine never raises (an error,
//!   an uncore interrupt). It reads the entry's hand-off value (no event
//!   recorded, with the enables iBoot leaves set), and the writes software
//!   uses to clear it (write-one-to-clear) change nothing.
//! - `Record`: information about such an event. It reads 0, and since no
//!   consumer writes it, a write stops.
//! - `Counter`: an event counter whose controls (`Control` entries) accept
//!   only their disabled value here, or whose events the emulated die never
//!   has, so it never counts. It holds what software writes, 0 from reset
//!   (inference).
//! - `Energy`: a core energy accumulator of the emulated die
//!   ([`Registers::energy`]): it counts, in its consumer's units, the energy
//!   of a fixed modeled power per core type over the system counter's time
//!   since the PE first read it, so it never decreases. Writes stop.
//! - `Guest`: a VM's copy of Apple EL1 state, which the kernel's hypervisor
//!   loads before it runs a vCPU and saves afterwards. It is stored like
//!   `Storage`, but this machine applies none of it to a guest, so entering
//!   an EL1&0 guest of the EL2 host while one holds a nonzero value must
//!   stop. `Registers::guest_state` finds such a value, which the engine's
//!   ERET and GEXIT check before entering such a guest.
//! - `Unsupported`: its consumer relies on an effect this machine cannot
//!   provide; every access stops with the evidence as the reason.
//!
//! An entry applies on a chip only where that chip's own SPTM, TXM,
//! kernelcache or system kext collection in a supported build
//! (`stela_formats::builds`) accesses the encoding ([`census`], keyed by
//! `machine::Chip` as `Pmu::on` is), and, when it names chips
//! ([`Entry::chips`]), only on those: an encoding whose consumers differ by
//! chip (UPMCR0's 24-counter layout on the M6) has one entry per layout. The
//! entries cite the consumer on the chips whose components they were
//! established from; on another chip whose census shows the encoding, the same
//! behavior is an inference, except for the separately checked predictor
//! pair ([`objc`]). Entries span CRn 15
//! and CRn 11 (the M5 and M6 PL2 error records and M6 uncore selections).
//! An encoding the chip's components never access is not looked up: without
//! a model of its own it is UNDEFINED, Arm's behavior for an unallocated
//! encoding. That the chip allocates none of them is an inference from the
//! census (no component of a supported build needs one), not an observation.
//!
//! `Scope::Cluster` entries are one register per cluster, shared by its PEs
//! (`Registers::join`); the others are per PE. `el2` entries are UNDEFINED
//! below EL2. Every entry is hardware-unverified ([`VERIFICATION`]): no chip
//! has had the space enumerated, so neither an encoding's existence nor its
//! reset value is observed. The run reports every entry the guest used, so none of
//! these is silent.
use alloc::{boxed::Box, collections::BTreeMap, vec};
use stela_compiler::aarch64::system::registers::Encoding;
use stela_contracts::Chip;

mod census;
mod objc;

/// The verification state of every entry.
pub const VERIFICATION: &str =
    "hardware-unverified until a chip's enumeration observes the encoding and its reset value";

/// The CRn of most Apple registers (op0 3); the encodings below the table
/// macro are CRn 11's.
const IMPDEF_CRN: u8 = 15;

macro_rules! encodings {
    ($($name:ident = $op1:literal, $crm:literal, $op2:literal;)*) => {
        $(pub const $name: Encoding = Encoding::new(3, $op1, IMPDEF_CRN, $crm, $op2);)*
    };
}

// Names from XNU where its sources give one (xnu-6153.141.1 to
// xnu-12377.121.6 headers and strings), else from m1n1 (`cpu_regs.h`, else
// `proxyclient/m1n1/apple_regs.json`), else the encoding.
encodings! {
    HID0 = 0, 0, 0;
    EHID0 = 0, 0, 1;
    HID26 = 0, 0, 3;
    HID27 = 0, 0, 4;
    HID1 = 0, 1, 0;
    EHID1 = 0, 1, 1;
    EHID20 = 0, 1, 2;
    HID21 = 0, 1, 3;
    TUNING_C1_6 = 0, 1, 6;
    HID2 = 0, 2, 0;
    EHID2 = 0, 2, 1;
    TUNING_C2_3 = 0, 2, 3;
    TUNING_C2_4 = 0, 2, 4;
    TUNING_C2_5 = 0, 2, 5;
    HID3 = 0, 3, 0;
    HID4 = 0, 4, 0;
    EHID4 = 0, 4, 1;
    HID5 = 0, 5, 0;
    HID6 = 0, 6, 0;
    HID7 = 0, 7, 0;
    EHID7 = 0, 7, 1;
    HID9 = 0, 9, 0;
    EHID9 = 0, 9, 1;
    HID10 = 0, 10, 0;
    EHID10 = 0, 10, 1;
    HID11 = 0, 11, 0;
    HID18 = 0, 11, 2;
    EHID18 = 0, 11, 3;
    HID13 = 0, 14, 0;
    HID14 = 0, 15, 0;
    HID16 = 0, 15, 2;
    HID17 = 0, 15, 5;
    MEM_STALL_SELECT_0 = 1, 0, 3;
    MEM_STALL_SELECT_1 = 1, 1, 3;
    MEM_STALL_SELECT_2 = 1, 2, 3;
    MEM_STALL_SELECT_3 = 1, 3, 3;
    PMCR1_GL1 = 1, 8, 2;
    MEM_STALL_COUNTER_0 = 1, 8, 3;
    MEM_STALL_COUNTER_1 = 1, 9, 3;
    MEM_STALL_COUNTER_2 = 1, 10, 3;
    MEM_STALL_COUNTER_3 = 1, 11, 3;
    VM_COUNTER_OFFSET = 1, 9, 4;
    MEM_STALL_COUNTER_4 = 1, 12, 3;
    MEM_STALL_COUNTER_5 = 1, 13, 3;
    MEM_STALL_COUNTER_6 = 1, 14, 3;
    MEM_STALL_COUNTER_7 = 1, 15, 3;
    LSU_ERR_STS = 3, 0, 0;
    E_LSU_ERR_STS = 3, 2, 0;
    L2C_ERR_STS = 3, 8, 0;
    L2C_ERR_ADR = 3, 9, 0;
    L2C_ERR_INF = 3, 10, 0;
    FED_ERR_STS = 4, 0, 0;
    CORE_TUNING_1 = 4, 0, 1;
    E_FED_ERR_STS = 4, 0, 2;
    CORE_TUNING_3 = 4, 0, 3;
    VMSA_LOCK_EL12 = 4, 2, 6;
    AMX_CTX_EL1 = 4, 5, 0;
    SPRR_UMPRR_EL12 = 4, 8, 0;
    SPRR_UPERM_SH1_EL12 = 4, 8, 1;
    SPRR_UPERM_SH2_EL12 = 4, 8, 2;
    SPRR_UPERM_SH3_EL12 = 4, 8, 3;
    CTRR_A_LWR_EL12 = 4, 9, 0;
    CTRR_A_UPR_EL12 = 4, 9, 1;
    CTRR_B_LWR_EL12 = 4, 9, 2;
    CTRR_B_UPR_EL12 = 4, 9, 3;
    CTRR_CTL_EL12 = 4, 9, 4;
    CTRR_LOCK_EL12 = 4, 9, 5;
    SPTM_KEY_0 = 4, 10, 0;
    SPTM_KEY_1 = 4, 10, 1;
    SPTM_KEY_2 = 4, 10, 2;
    SPTM_KEY_3 = 4, 10, 3;
    SIQ_CFG_EL1 = 4, 10, 4;
    AVNCR_EL2 = 4, 10, 7;
    THREAD_CONTROL = 4, 12, 0;
    THREAD_KEY_0 = 4, 13, 6;
    THREAD_KEY_1 = 4, 13, 7;
    GUEST_C14_0 = 4, 14, 0;
    GUEST_C14_1 = 4, 14, 1;
    GUEST_C14_2 = 4, 14, 2;
    GUEST_C14_3 = 4, 14, 3;
    JIT_RANGE = 4, 15, 1;
    GUEST_C15_2 = 4, 15, 2;
    JIT_BOX_MODE = 4, 15, 4;
    GUEST_C15_5 = 4, 15, 5;
    JCTL_EL0 = 4, 15, 6;
    DPC_ERR_STS = 5, 0, 5;
    VM_TMR_LR_EL2 = 5, 1, 2;
    SWITCH_CLEARED = 5, 1, 4;
    WATCHDOG_DIAG0 = 5, 2, 6;
    ACC_CFG = 5, 4, 0;
    ACC_OVRD = 5, 6, 0;
    TRACE_CLEARED = 5, 10, 1;
    MMU_ERR_STS = 6, 0, 0;
    AFSR1_GL2 = 6, 0, 2;
    AFSR1_GL12 = 6, 0, 3;
    BP_OBJC_ADR_EL1 = 6, 0, 4;
    BP_OBJC_CTL_EL1 = 6, 0, 5;
    GUARDED_HPFAR = 6, 1, 1;
    E_MMU_ERR_STS = 6, 2, 0;
    APGAKEYLO_EL12 = 6, 2, 1;
    APGAKEYHI_EL12 = 6, 2, 2;
    KERNKEYLO_EL12 = 6, 2, 3;
    KERNKEYHI_EL12 = 6, 2, 4;
    AFPCR_EL0 = 6, 2, 5;
    SPRR_PMPRR_EL12 = 6, 6, 0;
    SPRR_PPERM_SH1_EL12 = 6, 6, 1;
    SPRR_PPERM_SH2_EL12 = 6, 6, 2;
    SPRR_PPERM_SH3_EL12 = 6, 6, 3;
    APIAKEYLO_EL12 = 6, 7, 0;
    APIAKEYHI_EL12 = 6, 7, 1;
    APIBKEYLO_EL12 = 6, 7, 2;
    APIBKEYHI_EL12 = 6, 7, 3;
    APDAKEYLO_EL12 = 6, 7, 4;
    APDAKEYHI_EL12 = 6, 7, 5;
    APDBKEYLO_EL12 = 6, 7, 6;
    APDBKEYHI_EL12 = 6, 7, 7;
    ASPSR_EL1 = 6, 8, 3;
    PMCR1_COMPANION = 6, 8, 5;
    GUEST_C12_7 = 6, 12, 7;
    APVMKEYLO_EL2 = 6, 14, 4;
    APVMKEYHI_EL2 = 6, 14, 5;
    ACTLR_EL12 = 6, 14, 6;
    APSTS_EL12 = 6, 14, 7;
    SPRR_CONFIG_EL12 = 6, 15, 4;
    SPRR_AMRANGE_EL12 = 6, 15, 5;
    CLUSTER_COUNTER_C0 = 7, 0, 2;
    AON_COUNTER_0 = 7, 0, 3;
    UPMCR0 = 7, 0, 4;
    UPMC8 = 7, 0, 5;
    CORE_NRG_ACC_DAT = 7, 1, 1;
    AON_SELECT_0 = 7, 1, 3;
    UPMESR0_EL1 = 7, 1, 4;
    UPMC9 = 7, 1, 5;
    CLUSTER_COUNTER_C2 = 7, 2, 2;
    SPTM_TUNING_SELECT = 7, 2, 0;
    AON_COUNTER_1 = 7, 2, 3;
    UPMC10 = 7, 2, 5;
    CORE_SRM_NRG_ACC_DAT = 7, 3, 1;
    AON_SELECT_1 = 7, 3, 3;
    UPMECM0_EL1 = 7, 3, 4;
    UPMC11 = 7, 3, 5;
    CLUSTER_COUNTER_C4 = 7, 4, 2;
    AON_COUNTER_2 = 7, 4, 3;
    UPMECM1_EL1 = 7, 4, 4;
    UPMC12 = 7, 4, 5;
    AON_ENABLE = 7, 5, 0;
    AON_SELECT_2 = 7, 5, 3;
    UPMPCM_EL1 = 7, 5, 4;
    UPMC13 = 7, 5, 5;
    CLUSTER_COUNTER_C6 = 7, 6, 2;
    AON_COUNTER_3 = 7, 6, 3;
    UPMSR = 7, 6, 4;
    UPMC14 = 7, 6, 5;
    AON_SELECT_3 = 7, 7, 3;
    UPMC0 = 7, 7, 4;
    UPMC15 = 7, 7, 5;
    CLUSTER_COUNTER_C8 = 7, 8, 2;
    UPMC1 = 7, 8, 4;
    UPMECM2_EL1 = 7, 8, 5;
    UPMC2 = 7, 9, 4;
    UPMECM3_EL1 = 7, 9, 5;
    CLUSTER_COUNTER_C10 = 7, 10, 2;
    UPMC3 = 7, 10, 4;
    UPMC4 = 7, 11, 4;
    UPMESR1_EL1 = 7, 11, 5;
    CLUSTER_COUNTER_C12 = 7, 12, 2;
    UPMC5 = 7, 12, 4;
    UPMC6 = 7, 13, 4;
    UPMC7 = 7, 14, 4;
    // The M3 and later chips'.
    INIT_C0_7 = 4, 0, 7;
    AHCR_EL2 = 4, 12, 1;
    CPM_PWRDN_CTL = 5, 6, 2;
    TRACE_C1_0 = 5, 1, 0;
    TRACE_C1_5 = 5, 1, 5;
    PROD_TRACE_CORE = 5, 1, 6;
    TRACE_C1_7 = 5, 1, 7;
    TRACE_C2_1 = 5, 2, 1;
    TRACE_C2_2 = 5, 2, 2;
    TRACE_C2_3 = 5, 2, 3;
    TRACE_C2_4 = 5, 2, 4;
    TRACE_C2_5 = 5, 2, 5;
    TRACE_C3_0 = 5, 3, 0;
    PROD_TRACE_MASK = 5, 3, 3;
    TRACE_C3_4 = 5, 3, 4;
    PROD_TRACE_BASE_0 = 5, 3, 6;
    PROD_TRACE_BASE_1 = 5, 3, 7;
    TRACE_C4_6 = 5, 4, 6;
    TRACE_VA_C4_7 = 5, 4, 7;
    TRACE_VA_C5_7 = 5, 5, 7;
    TRACE_C10_6 = 5, 10, 6;
    TRACE_C10_7 = 5, 10, 7;
    TRACE_C11_1 = 5, 11, 1;
    PROD_TRACE_ENABLE = 5, 11, 2;
    TRACE_C11_3 = 5, 11, 3;
    SW_TRACE_DATA = 5, 11, 6;
    AON_COUNTER_4 = 7, 8, 3;
    AON_SELECT_4 = 7, 9, 3;
    // The T6050's sixth to tenth core counters.
    AON_COUNTER_5 = 7, 10, 3;
    AON_SELECT_5 = 7, 11, 3;
    AON_COUNTER_6 = 7, 12, 3;
    AON_SELECT_6 = 7, 13, 3;
    AON_COUNTER_7 = 7, 14, 3;
    AON_SELECT_7 = 7, 15, 3;
    AON_COUNTER_8 = 7, 14, 0;
    AON_COUNTER_9 = 7, 14, 1;
    AON_SELECT_8 = 7, 15, 0;
    AON_SELECT_9 = 7, 15, 1;
    CLUSTER_COUNTER_C0_6 = 7, 0, 6;
    CLUSTER_COUNTER_C2_6 = 7, 2, 6;
    CLUSTER_COUNTER_C4_6 = 7, 4, 6;
    CLUSTER_COUNTER_C6_6 = 7, 6, 6;
    CLUSTER_COUNTER_C8_6 = 7, 8, 6;
    CLUSTER_COUNTER_C14_2 = 7, 14, 2;
    SPTM_CLEARED_C4_7 = 1, 4, 7;
    SPTM_CLEARED_C9_7 = 1, 9, 7;
    IRG_KEY_0 = 1, 11, 2;
    IRG_KEY_1 = 1, 12, 2;
    IRG_KEY_2 = 1, 13, 2;
    IRG_KEY_3 = 1, 14, 2;
    IRG_KEY_4 = 1, 15, 2;
    UNCORE_CONTROL_1 = 7, 10, 5;
    UNCORE_SELECT_C11_7 = 7, 11, 7;
    UNCORE_SELECT_C12_0 = 7, 12, 0;
    UNCORE_SELECT_C12_1 = 7, 12, 1;
    UNCORE_SELECT_C12_5 = 7, 12, 5;
    UNCORE_COUNTER_16 = 7, 12, 7;
    UNCORE_COUNTER_17 = 7, 13, 0;
    UNCORE_COUNTER_18 = 7, 13, 1;
    UNCORE_COUNTER_19 = 7, 13, 7;
    UNCORE_COUNTER_20 = 7, 14, 7;
    UNCORE_COUNTER_21 = 7, 15, 7;
    UNCORE_COUNTER_22 = 7, 10, 7;
    UNCORE_COUNTER_23 = 7, 10, 1;
}

// CRn 11: the M5 and M6 P-cluster L2 error records (names from the XNU
// format "PL2_ERR_STS/ADR/INF") and two M6 uncore core-mask selections.
pub const PL2_ERR_ADR: Encoding = Encoding::new(3, 1, 11, 4, 5);
pub const PL2_ERR_INF: Encoding = Encoding::new(3, 1, 11, 4, 6);
pub const PL2_ERR_STS: Encoding = Encoding::new(3, 1, 11, 4, 7);
pub const UNCORE_MASK_4: Encoding = Encoding::new(3, 6, 11, 12, 6);
pub const UNCORE_MASK_5: Encoding = Encoding::new(3, 6, 11, 12, 7);

/// The engine's tests still name SPRR_CONFIG_EL12 by its former name.
pub const GUEST_SPRR_C15_4: Encoding = SPRR_CONFIG_EL12;

/// UPMCR0's interrupt mode 4 (FIQ, bits 16-18) and per-counter PMI enables
/// (bits 20-35): m1n1 `UPMCR0_IMODE_T8020`, `UPMCR0_IMODE_FIQ` and
/// `UPMCR0_PMI_SHIFT_T8020`, for 16 counters. Its counter enables (bits
/// 0-15) and other interrupt modes are not permitted.
const UPMCR0_IMODE_FIQ: u64 = 4 << 16;
const UPMCR0_PMI_ENABLES: u64 = 0xffff << 20;
const UPMCR0_PERMITTED: u64 = UPMCR0_IMODE_FIQ | UPMCR0_PMI_ENABLES;
/// The chips with that 16-counter layout.
const UPMCR0_16_CHIPS: [Chip; 5] = [
    Chip::AppleM1,
    Chip::AppleM2,
    Chip::AppleM3,
    Chip::AppleM4,
    Chip::AppleM5,
];
/// The M6's 24 PMI enables (bits 24-47), which its XNU writes with no
/// interrupt-mode field ([`UNCORE_CONTROL_24`]); its counter enables (bits
/// 0-23) are not permitted.
const UPMCR0_PMI_ENABLES_24: u64 = 0xff_ffff << 24;

/// What iBoot leaves set in L2C_ERR_STS: undescribed enables in bits 34-38,
/// 40-43 and 60 (Asahi `hw/cpu/system-registers.md`).
const L2C_ERR_STS_HANDOFF: u64 = 0x1f << 34 | 0xf << 40 | 1 << 60;

/// PROD_TRACE_CORE bits 26-28, which XNU sets at CPU initialization under
/// its `APTMsgProdEL0` boot-arg, true by default ([`PROD_TRACE_CORE_EVIDENCE`]).
const PROD_TRACE_EL0_MESSAGES: u64 = 0x7 << 26;
/// PROD_TRACE_MASK bit 1, which the M3 SPTMs set on P cores at boot.
const PROD_TRACE_P_CORE_MASK: u64 = 1 << 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Class {
    Storage,
    Tuning,
    Prediction,
    Scratch,
    Control,
    Status,
    Record,
    Counter,
    Energy,
    Guest,
    Unsupported,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scope {
    Pe,
    Cluster,
}

pub struct Entry {
    pub encoding: Encoding,
    pub name: &'static str,
    pub class: Class,
    pub evidence: &'static str,
    pub scope: Scope,
    /// UNDEFINED below EL2.
    pub el2: bool,
    /// The bits a `Control` or `Prediction` entry takes.
    pub permitted: u64,
    /// Bits the entry holds whose effect this machine does not model, and
    /// what is missing: a write that sets one completes and is reported
    /// once ([`Entry::unmodeled_effect`]).
    pub unmodeled: Option<(u64, &'static str)>,
    /// What a `Status` entry reads: the enables iBoot hands off, no event.
    pub handoff: u64,
    /// The chips the entry is for, where their census shows the encoding.
    pub chips: &'static [Chip],
}

const fn entry(
    encoding: Encoding,
    name: &'static str,
    class: Class,
    evidence: &'static str,
) -> Entry {
    Entry {
        encoding,
        name,
        class,
        evidence,
        scope: Scope::Pe,
        el2: false,
        permitted: 0,
        unmodeled: None,
        handoff: 0,
        chips: &Chip::ALL,
    }
}

/// A `Control` entry that holds every value and reports a nonzero one once
/// as `missing`: its effect is not established.
const fn held(
    encoding: Encoding,
    name: &'static str,
    evidence: &'static str,
    missing: &'static str,
) -> Entry {
    entry(encoding, name, Class::Control, evidence)
        .permitting(u64::MAX)
        .unmodeled(u64::MAX, missing)
}

/// A processor-trace register ([`TRACE_UNMODELED`]).
const fn trace(encoding: Encoding, name: &'static str, evidence: &'static str) -> Entry {
    held(encoding, name, evidence, TRACE_UNMODELED)
}

impl Entry {
    const fn only(self, chips: &'static [Chip]) -> Self {
        Self { chips, ..self }
    }

    const fn handing_off(self, handoff: u64) -> Self {
        Self { handoff, ..self }
    }

    const fn cluster(self) -> Self {
        Self {
            scope: Scope::Cluster,
            ..self
        }
    }

    const fn el2(self) -> Self {
        Self { el2: true, ..self }
    }

    const fn permitting(self, permitted: u64) -> Self {
        Self { permitted, ..self }
    }

    const fn unmodeled(self, bits: u64, missing: &'static str) -> Self {
        Self {
            unmodeled: Some((bits, missing)),
            ..self
        }
    }

    /// What a write of `value` would need that this machine does not model.
    pub fn unmodeled_effect(&self, value: u64) -> Option<&'static str> {
        self.unmodeled
            .filter(|(bits, _)| value & bits != 0)
            .map(|(_, missing)| missing)
    }

    /// Whether the PE's value ([`Registers::storage`]) is the register:
    /// the classes that keep what is written, on this PE alone.
    const fn stored(&self) -> bool {
        matches!(
            self.class,
            Class::Storage
                | Class::Tuning
                | Class::Prediction
                | Class::Scratch
                | Class::Control
                | Class::Counter
                | Class::Guest
        ) && matches!(self.scope, Scope::Pe)
    }

    /// Whether a read that completes returns [`Registers::storage`]: the SRAM
    /// rail's accumulator always reads the 0 its word holds.
    pub fn reads_storage(&self) -> bool {
        self.stored() && self.encoding != TRACE_VA_C5_7 || self.encoding == CORE_SRM_NRG_ACC_DAT
    }

    /// The bits of a value that take a write more than a store to
    /// [`Registers::storage`], if a value without them takes no more.
    pub fn stored_write(&self) -> Option<u64> {
        let special = matches!(self.encoding, PMCR1_GL1 | SIQ_CFG_EL1);
        let checked = match self.class {
            Class::Control | Class::Prediction => !self.permitted,
            _ => 0,
        };
        let unmodeled = self.unmodeled.map_or(0, |(bits, _)| bits);
        (self.stored() && !special).then_some(checked | unmodeled)
    }
}

/// JIT box mode bit 0: the box is enforced. The kernel writes 2 for a task
/// without a JIT box, and SPTM clears this bit alone while it runs.
const JIT_BOX_ENABLE: u64 = 1;

const JIT_BOX_MODE_EVIDENCE: &str = "no public name; 26A434 XNU's JIT box (jitbox.c) writes a \
    task's mode to it before its range (S3_4_C15_C15_1): 2 for a task without a box, else \
    0x061e0c5b or 0x0e1e0c5b, then ISB (0xfffffe000bda55ec). A task gets a box only from \
    mmap with the com.apple.private.verified-jit entitlement (0xfffffe000c1a7c48, \
    0xfffffe000c1a7e44), and its EL0 faults (EC 0x3f, ISS 0x18-0x1f) raise \
    EXC_BAD_INSTRUCTION (0xfffffe000bd77d14). SPTM 26A434 writes 2 at boot (__text+0x40e4) and \
    saves it, clears bit 0 and restores it around its own execution with SCTLR_EL1 and TCR_EL1 \
    (__text+0x2124, __text+0x28ec): bit 0 enables the box (inference). Every value is held. \
    What an enabled box enforces is not documented, \
    so a value with bit 0 set is reported once as unmodeled";

/// What an enabled JIT box would need: its enforcement.
const JIT_BOX_UNMODELED: &str = "JIT box enabled; enforcement not modeled: JITs that keep to \
    the box run as on hardware, but the guest loses the box's hardening against a compromised \
    JIT";

const SPTM_KEY_WRITES: &str = "SPTM 26A434 writes it once per CPU with a PAC-key-derived \
    value after the architectural PAC keys and reads it nowhere; TXM and the kernel never touch it";

const M1N1_TUNING: &str = "XNU xnu-6153.141.1 arm64_common.h and m1n1 cpu_regs.h name it a \
    core tuning (HID) register and its chicken bits (prefetch, fusion, store and fill \
    disables, power-gating timings); SPTM 26A434 sets or clears bits with read-modify-write at \
    boot. They tune speculation and power only, so no bit has an effect here";

const XNU_THREAD_CONTROL: &str = "26A434 XNU saves it in every exception frame and \
    restores it on return (0xfffffe000bb9da44, 0xfffffe000bb9db48), SPTM 26A434 keeps it in \
    each context it resumes (__text+0x44a8), and CPU initialization clears bit 1 \
    (0xfffffe000bd80ed8) unless a patch alternative sets it, the bit ACTLR_EL1 names EnTSO; \
    AIDR_EL1 here advertises no TSO";

const XNU_AMX_CONTEXT: &str = "26A434 XNU CPU initialization writes MPIDR.Aff0 to it \
    under one boot configuration (0xfffffe000bd80f44) and reads it nowhere; m1n1 cpu_regs.h names \
    it AMX_CTX_EL1 beside AMX_CTL_EL1, while m1n1's apple_regs.json calls \
    it CORE_INDEX and Asahi's documentation says it is written with the core's index in its \
    cluster during initialization, which the value fits. Nothing reads it, so it is stored";

const SPTM_CORE_TUNING: &str = "no public name; SPTM 26A434 clears one bit with \
    read-modify-write at boot, selected by MPIDR core type next to its HID writes \
    (__text+0xb9a8, __text+0xb9d0), and AppleARM64ErrorHandler::_fedErrorHandler reads it \
    only after a front-end error (0xfffffe0009495114), which this machine never raises";

const PMCR1_COMPANION_EVIDENCE: &str = "no public name; SPTM 26A434 writes it once per CPU \
    at boot with the value it writes to PMCR1_GL1 (0x30300 or 0x3_0300_00ff_ff00, __text+0x1bebc) \
    and reads it nowhere; TXM and the kernel never touch it. Both values are PMCR1 mode-filter \
    patterns (m1n1 cpu_regs.h PMCR1 layout), so it is plausibly another guarded-level PMCR1 \
    (inference); stored";

const PMCR1_GL1_EVIDENCE: &str = "m1n1's apple_regs.json names it PMCR1_GL1, the \
    core PMU's mode filter for the guarded level: SPTM 26A434 writes 0x30300 (PMC0 and PMC1 \
    counting at EL0 and EL1) or a full enable pattern once per CPU at boot (__text+0x1bec8), and \
    the 26A434 kernel only saves and restores it with a VM's GL12 registers (0xfffffe000bd9eddc, \
    0xfffffe000bda2bb4). Stored, and machine::pmu filters counting at a guarded level by its \
    kernel field (pmu::Filter::Guarded)";

const ACC_CFG_EVIDENCE: &str = "m1n1 cpu_regs.h names it, with the branch-predictor sleep field \
    ACC_CFG_BP_SLEEP (bits 2-3), which 26A434 XNU sets with read-modify-write \
    (0xfffffe000bb9c8e0); the engine keeps no predictor state to retain";

const ACC_OVRD_EVIDENCE: &str = "m1n1 cpu_regs.h names it; XNU xnu-12377.121.6 \
    machine_routines_asm.s sets its deep-sleep fields (xnu-6153.141.1 arm64_common.h \
    ARM64_REG_ACC_OVRD_*) before a deep-sleep power-down, which acts only when the cluster \
    powers down, and core power-down stops (CYC_OVRD)";

/// The established SIQ_CFG_EL1 mode field. Only 0, 2 and 3 have checked
/// effects; mode 1 and every upper bit refuse before register mutation.
pub const SIQ_CFG_MODE: u64 = 0b11;

const SIQ_CFG: &str = "m1n1 cpu_regs.h names it SIQ_CFG_EL1, and Asahi's documentation says \
    the AICv2 sends a core that writes 3 to it no IRQs (FIQs are unaffected); 26A434 XNU sets bits \
    0-1 to a per-thread mode with read-modify-write at context switch (0xfffffe000bd7c388) and \
    before a power-down, and clears it elsewhere (0xfffffe000bd70b74). Modes 0/2 admit IRQs \
    and mode 3 keeps an AICv2 or AICv3 from sending the PE IRQs. Only these modes with zero \
    upper bits are held; mode 1 and upper-bit effects lack qualification and refuse before \
    mutation";

const XNU_THREAD_KEYS: &str = "26A434 XNU's context switch writes it with the thread's \
    APIBKey value plus 4 or 5, after APIBKey and APDBKey (0xfffffe000bb9bd38), and reads it \
    nowhere; the software PAC function, like for the Apple PAC controls, uses only the \
    architectural keys";

const WATCHDOG_DIAG: &str = "XNU xnu-12377.121.6 debug.c names it WATCHDOG_DIAG0, which the \
    Apple watchdog logging ORs its flags into (awl_set_scratch_reg_hv_bit); m1n1's \
    apple_regs.json calls it AWL_SCRATCH_EL1";

const XNU_SWITCH_CLEARS: &str = "26A434 XNU writes 0 to it at every context switch \
    (0xfffffe000bd8393c, beside S3_5_C15_C10_1) and reads it nowhere; SPTM and TXM never \
    touch it. Writing 0 can only clear state this machine does not have (inference)";

const XNU_TRACE_CLEARS: &str = "26A434 XNU writes 0 to it at every context switch \
    (0xfffffe000bd83940) and after appending a timestamp record to a per-CPU trace ring \
    (0xfffffe000bd7b0e0), and reads it nowhere; SPTM and TXM never touch it. Writing 0 can \
    only clear state this machine does not have (inference)";

const JCTL_EVIDENCE: &str = "XNU names it JCTL_EL0 (xnu-12377.121.6 tests/user_msrs.c, a \
    per-thread user register; sleh.c treats a fault with it in pointer signing mode as an \
    attack on WebKit), so WebKit's JIT plausibly sets it from EL0 (inference); m1n1 calls it \
    the JIT box \
    control. 26A434 XNU swaps a per-thread value through it at context switch \
    (0xfffffe000bd83950) and elsewhere reads it beside SPRR_PERM_EL0, treating the thread as \
    unrestricted when bits 14-15 are clear and SPRR_PERM_EL0 holds its default \
    (0xfffffe000bd787cc). Its layout is not public and what its modes enforce is not modeled: \
    held, and a nonzero value is reported once as unmodeled";

const JCTL_UNMODELED: &str = "JCTL_EL0 mode set; the pointer signing and JIT box restrictions it \
    selects are not modeled";

const ERROR_STATUS: &str = "an error status register (m1n1 cpu_regs.h names it; the E_ form \
    is the E-core's): 26A434 XNU's panic for an unhandled implementation-specific error \
    prints it (0xfffffe000c52dab8), and AppleARM64ErrorHandler::_platformErrorHandler reads \
    the form for the MPIDR core type and tests its valid bit 56 (0xfffffe0009494970). Both \
    run only after an SError or implementation-defined error exception, which this machine \
    never raises, so it reads 0: no error recorded";

const DPC_ERROR_STATUS: &str = "XNU xnu-6153.141.1 arm64_common.h names it \
    ARM64_REG_DPC_ERR_STS (HAS_DPC_ERR on H16 and H17); \
    26A434 XNU's panic for an unhandled implementation-specific error prints it as dpc_err_sts \
    (0xfffffe000c52dabc), and AppleARM64ErrorHandler::_platformErrorHandler reads it on both core \
    types and tests its valid bit 56 (0xfffffe000949497c). Both run only after an SError or \
    implementation-defined error exception, which this machine never raises, so it reads 0: no \
    error recorded";

const L2C_ERROR_STATUS: &str = "m1n1 cpu_regs.h names it, with L2C_ERR_STS_ENABLE_W1C; \
    26A434 XNU's error panic and AppleARM64ErrorHandler read it (0xfffffe000c52dad8, \
    0xfffffe00094949bc), _llcErrorHandler writes back the status bit it handled \
    (0xfffffe0009494ecc, 0xfffffe0009494f3c), write-one-to-clear, and _llcErrorClear writes \
    its argument (0xfffffe0009494e98); SPTM 26A434 reads it with L2C_ERR_ADR and L2C_ERR_INF \
    only for panic_with_register_state (__text+0x5ab90). Asahi's documentation makes it one \
    register per cluster and says iBoot hands it off with bits 34-38, 40-43 and 60 set, \
    undescribed enables. No error exception is ever raised here, so it reads those enables with \
    no error recorded, and the clearing writes change nothing";

const L2C_ERROR_RECORD: &str = "m1n1 cpu_regs.h names it; 26A434 XNU's error panic, \
    AppleARM64ErrorHandler::_platformErrorHandler (0xfffffe00094949c0) and SPTM 26A434 \
    (__text+0x5ab94) read it with L2C_ERR_STS to describe a recorded L2 error. None is ever \
    recorded here, so it reads 0; no consumer writes it";

const UNCORE_CONTROL: &str = "m1n1 cpu_regs.h UPMCR0 (T8020 layout: counter enables in bits \
    0-15, interrupt mode in bits 16-18 with 4 for FIQ, PMI enables from bit 20); 26A434 XNU's \
    uncore monitor writes 0xf_fff4_0000, every PMI enabled in FIQ mode, at CPU initialization \
    and when it resets the counters (0xfffffe000bd8bec8, 0xfffffe000bd8c314), adding counter \
    enables only while uncore counting is configured (0xfffffe000bd8bebc). Uncore events \
    are not modeled, so a counter enable or another interrupt mode stops. One per cluster";

const UNCORE_SELECT: &str = "m1n1 cpu_regs.h names this uncore monitor selection (event \
    numbers, core masks or the PMI core mask); 26A434 XNU writes it from its uncore \
    configuration at CPU initialization (0xfffffe000bd8bea0 to 0xfffffe000bd8bf14) and never \
    reads it. No uncore counter counts here, so the selection has no effect. One per cluster";

const UNCORE_STATUS: &str = "m1n1 cpu_regs.h UPMSR (bit 0 an uncore PMI is active, bits \
    2-17 the counters' overflow); 26A434 XNU's FIQ handler reads bit 0 to recognize an \
    uncore PMI (0xfffffe000bd7b400) and writes 0 after resetting the counters \
    (0xfffffe000bd8c450). No uncore counter counts here, so it reads 0. One per cluster";

const UNCORE_COUNTER: &str = "m1n1 cpu_regs.h names this uncore counter; 26A434 XNU's \
    uncore monitor zeroes, reads and restores it (0xfffffe000bd8c2b0, 0xfffffe000bd8bb3c, \
    0xfffffe000bd8d020). It counts only under a UPMCR0 counter enable, which stops here, so \
    it holds what software writes, 0 from reset (inference). One per cluster";

/// The selectors of the core performance counters, in counter order.
pub const AON_SELECTS: [Encoding; super::aon::SLOTS] = [
    AON_SELECT_0,
    AON_SELECT_1,
    AON_SELECT_2,
    AON_SELECT_3,
    AON_SELECT_4,
    AON_SELECT_5,
    AON_SELECT_6,
    AON_SELECT_7,
    AON_SELECT_8,
    AON_SELECT_9,
];

/// The core performance counters, in counter order.
pub const AON_COUNTERS: [Encoding; super::aon::SLOTS] = [
    AON_COUNTER_0,
    AON_COUNTER_1,
    AON_COUNTER_2,
    AON_COUNTER_3,
    AON_COUNTER_4,
    AON_COUNTER_5,
    AON_COUNTER_6,
    AON_COUNTER_7,
    AON_COUNTER_8,
    AON_COUNTER_9,
];

/// S3_7_C15_C5_0 bit 0, which enableCore sets and clears.
const AON_ENABLED: u64 = 1;

const AON_ENABLE_EVIDENCE: &str = "no public name; AppleT8112CLPC's \
    CorePerfSampler::readAONCounterEnable and writeAONCounterEnable read and write it \
    (0xfffffe0009f40a3c, 0xfffffe0009f40a48), and enableCore sets it after selecting three \
    core counters' events in S3_7_C15_C1_3, C3_3 and C5_3 (0xfffffe0009f40b14). What those \
    counters count is not documented: under bit 0 the cycle and instruction selectors count \
    (machine::aon, inference) and the others keep what software writes \
    (platform::apple::core_counters); enabling them is reported once";

const AON_SELECT_EVIDENCE: &str = "no public name; AppleT8112CLPC's \
    CorePerfSampler::enableCore writes the selectors 0x8000, 0x8100 and 0x8700 to \
    S3_7_C15_C1_3, C3_3 and C5_3 before setting S3_7_C15_C5_0 (0xfffffe0009f40afc) and never \
    reads them; they act only once that enable is set, whose counting is not modeled, so \
    they are stored";

const AON_COUNTER_EVIDENCE: &str = "no public name; AppleT8112CLPC's CorePerfSampler::sample \
    and SchedulerInterface::contextSwitch read it as a wrapping 32-bit count \
    (0xfffffe0009f39778, 0xfffffe0009f382ac), and other PEs read it through this PE's \
    cpu-impl-reg + 0x4000 (platform::apple::core_counters, which a board's PEs read it from). It \
    counts under S3_7_C15_C5_0 what its selector selects: cycles for 0x8000 and retired \
    instructions for 0x8100, added to what software wrote (machine::aon, inference); other \
    selectors count nothing";

const AON_FOURTH_EVIDENCE: &str = "no public name; the CorePerfSampler of AppleT6000CLPCv3, \
    AppleT6020CLPC and the M3-and-later CLPCs selects a fourth core counter: enableCore writes \
    0x8000, 0x8100, 0x8300 and 0x8700 to S3_7_C15_C1_3, C3_3, C5_3 and C7_3 before setting \
    S3_7_C15_C5_0 (kernelcache.release.mac14j 0xfffffe0009efa014, mac13j 0xfffffe0009e74e38), \
    and sample reads S3_7_C15_C6_3 after C0_3, C2_3 and C4_3 as a wrapping 32-bit count \
    (mac14j 0xfffffe0009ef29ac, mac13j 0xfffffe0009e6d85c). The selector acts and the \
    counter counts only under S3_7_C15_C5_0, whose counting is not modeled, so the selector \
    is stored and the counter holds what software wrote, 0 from reset (inference)";

/// What a running sampler's counters lack here.
const CORE_COUNTERS_UNMODELED: &str = "core performance counters enabled; selectors 0x8000 \
    and 0x8100 count cycles and retired instructions (inference, machine::aon), other events \
    are not counted: CLPC reads a zero fraction from them";

const MEM_STALL_UNMODELED: &str = "memory-stall counters started; events not counted: CLPC \
    reads no stalls, which only changes its efficiency-stall DVFS caps";

const MEM_STALL_SELECT_EVIDENCE: &str = "no public name; AppleT8112CLPC's \
    CoreMemStallSampler::enableCore writes two 32-bit selectors into each of S3_1_C15_C0_3 \
    and C1_3 (0x801 and 0x805, 0x807 and 0x3) with read-modify-write to start four \
    memory-stall counters, and 0 to stop them (0xfffffe0009f41790). The events are not \
    documented: the selectors are held and starting the counters is reported once";

const MEM_STALL_COUNTER_EVIDENCE: &str = "no public name; AppleT8112CLPC's \
    CoreMemStallSampler::sample reads it as a 48-bit count whose bit 47 is its overflow, \
    which it clears by writing 0 (0xfffffe0009f41878), and other PEs read and clear it through \
    this PE's cpu-impl-reg + 0x6030 (platform::apple::core_counters, which a board's PEs read it \
    from). It counts only under the S3_1_C15_C0_3 and C1_3 selectors, whose counting is not \
    modeled, so it holds what software wrote, 0 from reset (inference)";

const CLUSTER_COUNTER_EVIDENCE: &str = "no public name; AppleT8112CLPC's \
    ClusterPerfSampler::enableCluster sets the cluster's counter enable and seven event \
    selectors (0x800f, 0x810f, 0x9c0f, 0x9e0f, 0xbe0f, 0xbf0f, 0x8e0f) through the cluster's \
    MMIO (0xfffffe0009f407a4), ClusterPerfSampler::sample reads the PE's own cluster's seven \
    counters through S3_7_C15_C0_2 to C12_2 as 48-bit deltas and writes 0 to one whose bit 47 \
    overflowed (0xfffffe0009f68a8c, 0xfffffe0009f68bdc), and CLPC::sampleSystemBIU divides \
    the summed bus-interface delta by the elapsed time, clamped to elapsed time per cluster, as \
    a utilization (0xfffffe0009f6871c..0xfffffe0009f687cc). The emulated die's bus interface \
    carries no traffic (as the fabric bridges' counters, platform::apple::bridge), so they never \
    count: they hold what software writes, 0 from reset (inference). One per cluster";

const ENERGY_EVIDENCE: &str = "m1n1 cpu_regs.h names this core energy accumulator; \
    AppleT8112CLPC's context switch adds CORE_NRG_ACC_DAT and CORE_SRM_NRG_ACC_DAT, panics \
    if the sum ever decreases (0xfffffe0009f38328, 0xfffffe0009f38374) and stores min(delta, \
    2^32 - 1) * 0x199999a >> 30, the delta over 40, as the off-core thread's \
    perfcontrol_thread_data.energy_estimate_nj (0xfffffe0009f38390..0xfffffe0009f383ac; XNU \
    machine_switch_perfcontrol_context adds it to the thread's energy), so a count is 1/40 nJ; \
    AppleT8112PMGR reports them (0xfffffe0009fa45d4, 0xfffffe0009fa8ad0). The emulated die \
    accumulates a fixed modeled core power (Registers::energy), all of it on CORE_NRG_ACC_DAT";

const SPTM_HID_TUNING: &str = "XNU xnu-6153.141.1 arm64_common.h names S3_0_C15_C2_0 HID2 \
    (disMMUmtlbPrefetch, bit 13) and m1n1's apple_regs.json names C2_1 and C15_5 EHID2 and \
    HID17; the others have no public name. The 26A434 SPTMs set or clear its \
    bits with read-modify-write in their HID tuning sequences beside the m1n1-named HIDs, each \
    under a MIDR part-number or revision test: the T8103's and T6000's (sptm.t8103 file 0xaba4c, \
    0xabad0, 0xaba68 for C2_0, C2_1 and C15_5) and the T8122's (sptm.t8122 file 0xac050, 0xabd88, \
    0xac1bc and 0xac138 for C2_3, C2_4, C2_5 and C1_6); no other component touches it";

const SPTM_TUNING_SELECT_EVIDENCE: &str = "no public name; the 26A434 T8122 SPTM reads it in its \
    HID tuning sequence on MIDR part number 0x43 and sets S3_0_C15_C2_5 bit 0, HID7 bit 2 and \
    three more HID bits only when its bit 0 is set (sptm.t8122 file 0xac204, 0xac270); no \
    consumer writes it. It reads 0, so that optional tuning is not applied (inference)";

const MEM_STALL_WIDE_SELECT_EVIDENCE: &str = "no public name; the 26A434 CoreMemStallSampler of \
    AppleT6000CLPCv3, AppleT6020CLPC and every M3-and-later CLPC (S3_1_C15_C2_3; \
    kernelcache.release.mac14j 0xfffffe0009efa63c) and of the Ultras' AppleT6002CLPC and \
    AppleT6022CLPC and the T6050's AppleT6050dCLPC (S3_1_C15_C3_3; mac14j 0xfffffe0009f7dab0, \
    mac17j 0xfffffe0009cef4b8) writes two more 32-bit selectors \
    into it, with the read-modify-write AppleT8112CLPC uses for S3_1_C15_C0_3 and C1_3, and 0 \
    to stop them. The events are not documented: the selectors are held and starting the \
    counters is reported once";

const MEM_STALL_WIDE_COUNTER_EVIDENCE: &str = "no public name; the same samplers read it as a \
    48-bit count after S3_1_C15_C8_3 to C11_3 and clear its overflow bit 47 by writing 0 \
    (C12_3 and C13_3: mac14j 0xfffffe0009efa7b0; C14_3 and C15_3: mac14j \
    0xfffffe0009f7c420, mac17j 0xfffffe0009cede28). It counts only under its selectors, whose counting is not modeled, so \
    it holds what software wrote, 0 from reset (inference)";

const VM_COUNTER_OFFSET_EVIDENCE: &str = "no public name; the 26A434 T8122 SPTM writes 0 to it \
    right after CNTVOFF_EL2 when it starts at EL2 (sptm.t8122 file 0xab0cc, 0xab670), and the \
    T8122 kernel's hypervisor loads a vCPU's saved value less a global offset into it beside \
    MDCR_EL2 and APVMKEYLO_EL2 (kernelcache.release.mac15g file 0x4d1a91c) and reads it back \
    adding that offset (0x4d195e8); inference: a VM's counter offset. Stored; entering an EL1&0 \
    guest while it is nonzero stops at the ERET or GEXIT";

const AFPCR_EVIDENCE: &str = "XNU names it AFPCR_EL0 (xnu-7195.141.2 VMAPPLE.h) with DAZ \
    (bit 0) and FTZ (bit 1) (proc_reg.h), which Asahi's documentation says replace FPCR.FZ while \
    ACTLR_EL1.EnAFP is set; 26A434 XNU saves and restores it with ASPSR_EL1 for EnTSO and \
    EnAPFLG threads (0xfffffe000bb9ad7c, 0xfffffe000bb9b96c) and swaps a VM's value \
    (0xfffffe000bb9da34), and SPTM 26A434 restores it with each context it resumes \
    (__text+0x44a0). EnAFP refuses here (AIDR_EL1 does not advertise AFP), so only 0 is \
    accepted";

const ASPSR_EVIDENCE: &str = "XNU names it ASPSR_EL1 (xnu-7195.141.2 VMAPPLE.h), with \
    APSTATE G, P and A in bits 0-2 (xnu-12377.121.6 proc_reg.h): Apple state the PE records with \
    an exception, as ASPSR_GL1 does for the guarded level. 26A434 XNU saves its bits 1-2 \
    (APSTATE_AP_MASK) with AFPCR_EL0 in the exception frame of a thread that runs with ACTLR_EL1's \
    EnTSO and EnAPFLG and restores them on return (0xfffffe000bb9ad7c, 0xfffffe000bb9b96c), and \
    swaps a VM's values at guest exit and entry (0xfffffe000bb9da34, 0xfffffe000bb9b794). The PE \
    here records none (TSO and APFLG, which AIDR_EL1 does not advertise), so it reads 0 and only 0 \
    is accepted";

const AVNCR_EVIDENCE: &str = "m1n1 cpu_regs.h names it AVNCR_EL2, which m1n1's \
    hypervisor sets with VNCR_EL2 under NV2; SPTM 26A434 reads and writes it through accessors \
    beside VNCR_EL2's (__text+0x593b0, __text+0x593c8), stores a page's physical address in it \
    with VNCR_EL2's when HCR_EL2.NV2 is set (__text+0x58f28, __text+0x59e4c) and clears it before \
    entering EL1 (__text+0x49edc). The M2 has no FEAT_NV2, so only 0 is accepted";

const JIT_RANGE_EVIDENCE: &str = "no public name; 26A434 XNU's JIT box (jitbox.c) writes a \
    task's JIT range to it, a 32 MiB-aligned user address ORed with a size code, then ISB \
    (0xfffffe000bda5670), after its mode (S3_4_C15_C15_4), and never reads it. The context \
    switch does so when the incoming task's box or JCTL_EL0 differs from the CPU's \
    (0xfffffe000bd839f0), and so does an update of the running task's (0xfffffe000bda5964). \
    The range acts only under an enabled mode, whose enforcement is not modeled, so it is held";

const GUARDED_HPFAR_EVIDENCE: &str = "no public name; SPTM 26A434 copies it to HPFAR_EL2, \
    as it copies the GL bank's ESR and FAR to EL1's, when it reflects a guarded-level \
    exception to the kernel (__text+0x27c8); inference: the guarded level's HPFAR. No \
    stage-2 fault is taken at a guarded level while HCR_EL2.VM is clear, so it reads 0, and \
    with VM set a read stops (machine::registers); no consumer writes it";

const GUARDED_AFSR1_EVIDENCE: &str = "m1n1's apple_regs.json names it AFSR1_GL2; SPTM \
    26A434 copies it to AFSR1_EL2 when it reflects a guarded-level exception to the kernel \
    (__text+0x27d0). Its contents are implementation defined: as for \
    AFSRn_ELx, this machine records nothing in it, so it reads 0; no consumer writes it";

const GUEST_AFSR1_EVIDENCE: &str = "m1n1's apple_regs.json names it AFSR1_GL12; 26A434 \
    XNU's hypervisor saves and restores it with a vCPU's GL12 registers (0xfffffe000bd9ede4, \
    0xfffffe000bda2bbc). As for AFSRn_ELx, this machine records no fault \
    status in it, so it holds what the hypervisor writes, 0 from reset (inference)";

const VM_KEY_EVIDENCE: &str = "m1n1 cpu_regs.h names it; SPTM 26A434 clears it before \
    entering EL1 (__text+0x49ecc) and 26A434 XNU's hypervisor loads each VM's key into it \
    before running a vCPU (0xfffffe000bda2488). Like KERNELKEYLO/HI_EL1 it is stored: the \
    software PAC function uses only the architectural keys, so a guest's PACs stay \
    self-consistent but differ from hardware";

const VM_STATUS_EVIDENCE: &str = "m1n1 cpu_regs.h names it APSTS_EL12; SPTM 26A434 clears it \
    before entering EL1 (__text+0x49ed4) and 26A434 XNU's hypervisor loads a VM's value \
    before running a vCPU (0xfffffe000bda2498). A guest's APSTS_EL1 reads MKEYVld as the \
    host's does, so the value is only stored";

const GUEST_EVIDENCE: &str = "m1n1's apple_regs.json names most of these the EL12 views of \
    the SPRR, PAC key, VMSA lock and CTRR registers and VM_TMR_LR_EL2 (which Asahi's \
    documentation describes as recording a guest's pending virtual timer); the rest have no \
    public name. 26A434 XNU's hypervisor saves it with a vCPU's \
    EL12, GL12, SPRR, PAC or vGIC state after the guest runs (0xfffffe000bd9ea30) and loads \
    it before running one (0xfffffe000bd9fdc8), and SPTM 26A434 loads it from a saved \
    context before entering EL1, saves it on return and clears the PAC-group ones \
    (__text+0x49e80, __text+0x4a824, __text+0x4a120). A VM's copy of Apple EL1 state this \
    machine does not apply to a guest: stored; entering an EL1&0 guest while it is \
    nonzero stops at the ERET or GEXIT";

const AHCR_EVIDENCE: &str = "m1n1 cpu_regs.h names it AHCR_EL2, and Asahi's documentation \
    relates it to ACTLR_EL1 bit 12 (IC IVAU enable ASID); the M3 kernels' CPU initialization ORs \
    in 0x1e (bits 1-4) with read-modify-write while setting ACTLR_EL1 bit 12, under a word that \
    is 1 in the image and that no code writes by its address (kernelcache.release.mac15g \
    0xfffffe000bcfd154, 0xfffffe000bcfd15c, word 0xfffffe000c92025c; mac15j 0xfffffe000be79134; \
    mac15s 0xfffffe000bce5170; a qemu-sptm trace of the M3 boot reads 0 and writes 0x1e). Its \
    fields are not described: held, and a nonzero value is reported once as unmodeled";

const AHCR_UNMODELED: &str = "AHCR_EL2 bits set; their effect, related to ACTLR_EL1's IC IVAU \
    ASID enable, is not modeled";

const INIT_C0_7_EVIDENCE: &str = "no public name; the M6 kernel writes 1 to it, and never reads \
    it, at boot (kernelcache.release.mac18g 0xfffffe000c5ab068), at CPU initialization after the \
    counter redirect and before clearing ACTLR_EL1 bits 0 and 9-11 (0xfffffe000bde286c), the \
    place the M3 kernels set AHCR_EL2, and on CPU resume (0xfffffe000bdf4464). Its effect is not \
    described: held, and a nonzero value is reported once as unmodeled";

const INIT_C0_7_UNMODELED: &str = "S3_4_C15_C0_7 set; its effect is not documented";

const CPM_PWRDN_CTL_EVIDENCE: &str = "XNU's H16 and H17 headers name a CPM_PWRDN_CTL register \
    for deep sleep (xnu-12377.121.6 H16.h HAS_CPM_PWRDN_CTL) without its encoding, which this \
    use matches (inference): the M3-and-later kernels' arm64_prepare_for_sleep sets bits 0 \
    and 63 with read-modify-write only on the deep-sleep path, where XNU xnu-7195.141.2 sets \
    ACC_OVRD's deep-sleep fields, right before CYC_OVRD's power-down request and the WFI that \
    turns the core off (kernelcache.release.mac15g 0xfffffe000bb191b4; mac16g \
    0xfffffe000bc31640; mac17g 0xfffffe000bc7df80; mac18g 0xfffffe000bbc9978). Like ACC_OVRD it \
    acts only when the cluster powers down, which the emulated die models as the core turning \
    off at that WFI";

const AON_FIFTH_EVIDENCE: &str = "no public name; the CorePerfSampler of AppleT6031CLPC and the \
    M4-and-later CLPCs selects a fifth core counter: enableCore writes 0x8000, 0x8100, 0x8700, \
    0x8a00 and 0x8200 to S3_7_C15_C1_3, C3_3, C5_3, C7_3 and C9_3 before setting S3_7_C15_C5_0 \
    (kernelcache.release.mac16g 0xfffffe0009f73cc8; mac15j 0xfffffe0009f02b54), and sample reads \
    S3_7_C15_C8_3 after C6_3 (mac16g 0xfffffe0009f6bc4c; mac15j 0xfffffe0009efa9a0). The \
    selector acts and the counter counts only under S3_7_C15_C5_0, whose counting is not \
    modeled, so the selector is stored and the counter holds what software wrote, 0 from reset \
    (inference)";

const AON_TENTH_EVIDENCE: &str = "no public name; AppleT6050CLPC's CorePerfSampler selects \
    ten core counters: enableCore writes 0x8000, 0x8100, 0x8300, 0x8700, 0x8200, 0x9000, 0x9100, \
    0x9200, 0x9300 and 0x9400 to S3_7_C15_C1_3, C3_3, C5_3, C7_3, C9_3, C11_3, C13_3, C15_3, \
    C15_0 and C15_1 before setting S3_7_C15_C5_0 (kernelcache.release.mac17j \
    0xfffffe0009c08318..0xfffffe0009c0836c), and sample reads S3_7_C15_C10_3, C12_3, C14_3, \
    C14_0 and C14_1 after C0_3 to C8_3 as wrapping 32-bit counts (0xfffffe0009bff598.. \
    0xfffffe0009bff5f8); other PEs read the ninth and tenth through cpu-impl-reg + 0x3fc0 and \
    0x3fc8 (platform::apple::core_counters). The selectors act and the counters count only \
    under S3_7_C15_C5_0, whose counting is not modeled, so the selectors are stored and the \
    counters hold what software wrote, 0 from reset (inference)";

const CLUSTER_COUNTER_LATER_EVIDENCE: &str = "no public name; the M3-and-later \
    ClusterPerfSampler::sample reads it with the PE's own cluster's S3_7_C15_C*_2 counters as \
    a 48-bit delta and writes 0 when its bit 47 overflowed, as AppleT8112CLPC does \
    (AppleT8122CLPC kernelcache.release.mac15g 0xfffffe0009ec7eac: C4_6 0xfffffe0009ec8010, \
    C14_2 0xfffffe0009ec8064, C0_6 0xfffffe0009ec8084, C2_6 0xfffffe0009ec80a0, the clear \
    0xfffffe0009ec8280; C6_6 in AppleT6041CLPC mac16j 0xfffffe0009fb7a08, AppleT6050CLPC \
    mac17j 0xfffffe0009c3a4f8 and AppleT8152CLPC mac18g 0xfffffe0009fc92e0). 26B123 widens \
    the samplers: AppleT8112CLPC mac14g reads C14_2 at 0xfffffe0009f84710 and clears it at \
    0xfffffe0009f84a08, AppleT8103CLPCv3 mac13g at 0xfffffe0009e35408 and 0xfffffe0009e35628, \
    AppleT8122CLPC mac15g reads C6_6 at 0xfffffe0009ee4f00, and AppleT8152CLPC mac18g reads \
    C8_6 at 0xfffffe0009ffe694 and clears it at 0xfffffe0009ffead4. The emulated die's bus \
    interface carries no traffic, so it never counts: it holds what software writes, 0 from \
    reset (inference). One per cluster";

const PL2_ERROR_STATUS: &str = "26A434 XNU's peh.c names it in its \"PL2 %s error: ... \
    PL2_ERR_STS/ADR/INF\" panic, which prints S3_1_C11_C4_7, C4_5 and C4_6 in that order \
    (kernelcache.release.mac17g 0xfffffe000becee04); the platform error handler reads them on a \
    P core and writes the status back masked with 0x57f (mac17g 0xfffffe000bece65c, \
    0xfffffe000bece678; mac18g 0xfffffe000be1b990), write-one-to-clear. It runs only after an \
    SError or implementation-defined error exception, which this machine never raises, so it \
    reads 0: no error recorded, and the clearing write changes nothing";

const PL2_ERROR_RECORD: &str = "26A434 XNU's peh.c prints it with PL2_ERR_STS as the PL2 error's \
    address or information (kernelcache.release.mac17g 0xfffffe000bece660, 0xfffffe000bece668, \
    0xfffffe000becee04) after an SError or implementation-defined error exception, which this \
    machine never raises; none is recorded, so it reads 0, and no consumer writes it";

const SPTM_CLEARS: &str = "no public name; the M6 SPTM writes 0 to it once per PE while it \
    brings the PE up, after SCTLR_EL1 and SCTLR2_EL1 (sptm.t8152 file 0xbb4ec, 0xbb4f0), and no \
    component reads it. Writing 0 can only clear state this machine does not have (inference)";

const IRG_KEY_EVIDENCE: &str = "no public name; the M6 SPTM's SPTM_FUNCTIONID_PROGRAM_IRGKEY \
    handler (sptm.t8152 file 0xfa640, dispatch entry 0x1e7d0) writes (n + i) << 4 ORed with the \
    PE's logical ID to S3_1_C15_C11_2, C12_2, C13_2, C14_2 and C15_2, i = 0 to 4 (file 0xfa668 to \
    0xfa718), and no component reads them. By that name, keys of the IRG tag generator \
    (inference): they can change only which permitted tag IRG picks, which Arm leaves to the \
    implementation, so they are stored";

const UNCORE_CONTROL_24: &str = "m1n1 cpu_regs.h names UPMCR0; the M6 kernel's uncore monitor \
    has 24 counters (its read switch maps 16-23 to S3_7_C15_C12_7, C13_0, C13_1, C13_7, C14_7, \
    C15_7, C10_7 and C10_1, kernelcache.release.mac18g 0xfffffe000be00a1c) and writes \
    0xffff_ff00_0000, PMI enables for 24 counters from bit 24 with no interrupt-mode field, at \
    CPU initialization and when it resets the counters (0xfffffe000be00eb0, 0xfffffe000be012c8), \
    adding counter enables in bits 0-23 only while uncore counting is configured \
    (0xfffffe000be00e2c). Uncore events are not modeled, so a counter enable stops. One per \
    cluster";

const UNCORE_CONTROL_1_EVIDENCE: &str = "no public name; the M6 kernel writes 1 << 63 to it before \
    the uncore event selections only while uncore counting is configured \
    (kernelcache.release.mac18g 0xfffffe000be00dd0, 0xfffffe000be02858) and never reads it; \
    plausibly its interrupt control (inference). Held, and a nonzero value is reported once as \
    unmodeled. One per cluster";

const UNCORE_CONTROL_1_UNMODELED: &str = "M6 uncore monitor control set; uncore events are not \
    counted";

const UNCORE_SELECT_24: &str = "xnu-12377.121.6 monotonic_arm64.c:466-477 and 498-509 write \
    S3_7_C15_C11_7, C12_0, C12_1 and C12_5 as UPMESR2-5 (uc_event_regs[2..5]) and \
    S3_6_C11_C12_6 and C12_7 as UPMECM4-5, the event selections and core masks of counters \
    16-23 when UPMU_PMC_COUNT > 16, with plain writes because they are write-ignored on SoCs \
    with fewer than six CPUs per cluster; the M6 kernel writes them from its uncore \
    configuration only while uncore counting is configured (kernelcache.release.mac18g \
    0xfffffe000be00e4c to 0xfffffe000be00ea0) and never reads them. No uncore counter counts \
    here, so the selection has no effect. One per cluster";

const UNCORE_COUNTER_24: &str = "xnu-12377.121.6 monotonic_arm64.c:337-356 reads it as uncore \
    counter 16 to 23 when UPMU_PMC_COUNT > 16, as the M6 kernel's uncore monitor does \
    (kernelcache.release.mac18g 0xfffffe000be00a1c), which zeroes and restores it with UPMC0-15 \
    (0xfffffe000be0126c, 0xfffffe000be01b7c). It counts only under a UPMCR0 counter enable, \
    which stops here, so it holds what software writes, 0 from reset (inference). One per \
    cluster";

const TRACE_SESSION: &str = "no public name; AppleProcessorTrace programs it only from its \
    per-core trace-source methods (AppleProcessorTraceACC8_9_10Src and the other cores' \
    classes; kernelcache.release.mac16g 0xfffffe0009ca8df0 to 0xfffffe0009caa358, the same code \
    in mac16j, mac17g and mac18g), which cpu_broadcast_xcall runs when \
    startTracingOnClustersGated configures the cores (0xfffffe0009cbe3bc) and when a session \
    pauses, resumes, drains or stops. Only a user client's validated session \
    (validateConfig, 0xfffffe0009cbd490) or the apt_panic_trace boot-arg (start, \
    0xfffffe0009cb91d8) starts one. The trace unit is not modeled: held, and a nonzero value \
    is reported once as unmodeled";

const SPTM_TRACE: &str = "the 26A434 M3-and-later SPTMs program it only from their cputrace \
    endpoints, which AppleProcessorTrace calls while starting or stopping a session: \
    sptm_cputrace_set_base writes the stream bases S3_5_C15_C3_6 and C3_7, whose asserts name \
    PROD_TRC_STRM_BASE (sptm.t8122 file 0xbedd4, 0xbede8), the VA-stop endpoint clears \
    S3_5_C15_C11_2 (file 0xbec68), and the M5 and M6 sptm_cputrace_va_init and \
    sptm_cputrace_va_deinit program S3_5_C15_C4_7, C5_7 and C11_1, polling C5_7 bits 20-27 and \
    36 with 1 ms timeouts after requiring C11_2 clear and C10_6 bit 33 clear (sptm.t8142 file \
    0xbdea4, 0xbdb58). The trace unit is not modeled: held, and a nonzero value is reported \
    once as unmodeled; C5_7 reads nothing in flight and a requested drain complete, so those \
    polls end (machine::registers trace_unit_status)";

const PROD_TRACE_CORE_EVIDENCE: &str = "no public name; the M4-and-later kernels' CPU \
    initialization ORs in 0x1c000000 (bits 26-28) with read-modify-write while the \
    APTMsgProdEL0 boot-arg holds, true unless given as 0 (kernelcache.release.mac17g parse \
    0xfffffe000c60d160, write 0xfffffe000be94d80; mac16g 0xfffffe000be05658; mac18g \
    0xfffffe000bde2b38; a qemu-sptm trace of the M5 boot writes 0x1c000000), the bits that let \
    software trace messages into the product trace by that name (inference), and \
    AppleProcessorTrace's trace-source methods write it for a session (mac16g \
    0xfffffe0009ca9cf4). With no session the message bits have no effect, so they are held \
    silently; any other bit is the trace unit's, held and reported once as unmodeled";

const PROD_TRACE_MASK_EVIDENCE: &str = "no public name; the M3 SPTMs write 2 to it on P cores \
    (MPIDR bits 16-18 = 1) in their boot lockdown sequence (sptm.t8122 file 0xbbf40; t6030 \
    0xbbef8; t6031 0xbbe3c), and the M4-and-later AppleProcessorTrace writes 1 or 2 when a \
    session's trace source starts and 0 when it stops (kernelcache.release.mac16g \
    0xfffffe0009ca8e34, 0xfffffe0009ca97e0). It qualifies the trace a session produces \
    (inference), so without one bit 1 has no effect and is held silently; bit 0, which only a \
    starting session writes, is held and reported once as unmodeled";

const SW_TRACE_EVIDENCE: &str = "no public name; 26A434 XNU's thread_invoke (sched_prim.c) \
    writes four software trace messages, the threads' identifiers and AGTCNTVCT, to it at every \
    context switch while the APTMsgProdEL0 boot-arg holds (kernelcache.release.mac17g \
    0xfffffe000c60d2dc from 0xfffffe000bd3e164; mac16g 0xfffffe000c572fd0; mac18g \
    0xfffffe000c5564f0), and nothing reads it. A message enters the product trace only while a \
    session traces, which is not modeled, so it is stored (inference)";

/// What a processor-trace session lacks here.
const TRACE_UNMODELED: &str = "processor trace configured; the trace unit is not modeled, so a \
    session produces no trace data";

pub const TABLE: &[Entry] = &[
    entry(
        SWITCH_CLEARED,
        "S3_5_C15_C1_4 (unnamed)",
        Class::Control,
        XNU_SWITCH_CLEARS,
    ),
    entry(
        TRACE_CLEARED,
        "S3_5_C15_C10_1 (unnamed)",
        Class::Control,
        XNU_TRACE_CLEARS,
    ),
    held(JCTL_EL0, "JCTL_EL0", JCTL_EVIDENCE, JCTL_UNMODELED),
    entry(
        BP_OBJC_ADR_EL1,
        "BP_OBJC_ADR_EL1",
        Class::Prediction,
        objc::EVIDENCE,
    )
    .permitting(objc::ADDRESS),
    entry(
        BP_OBJC_CTL_EL1,
        "BP_OBJC_CTL_EL1",
        Class::Prediction,
        objc::EVIDENCE,
    )
    .permitting(objc::CONTROL),
    entry(ACC_CFG, "ACC_CFG", Class::Tuning, ACC_CFG_EVIDENCE),
    entry(ACC_OVRD, "ACC_OVRD", Class::Tuning, ACC_OVRD_EVIDENCE),
    entry(SIQ_CFG_EL1, "SIQ_CFG_EL1", Class::Control, SIQ_CFG).permitting(SIQ_CFG_MODE),
    entry(
        THREAD_KEY_0,
        "S3_4_C15_C13_6 (unnamed)",
        Class::Storage,
        XNU_THREAD_KEYS,
    ),
    entry(
        THREAD_KEY_1,
        "S3_4_C15_C13_7 (unnamed)",
        Class::Storage,
        XNU_THREAD_KEYS,
    ),
    entry(
        WATCHDOG_DIAG0,
        "WATCHDOG_DIAG0",
        Class::Scratch,
        WATCHDOG_DIAG,
    ),
    entry(
        THREAD_CONTROL,
        "S3_4_C15_C12_0 (unnamed)",
        Class::Control,
        XNU_THREAD_CONTROL,
    ),
    entry(
        CORE_TUNING_1,
        "S3_4_C15_C0_1 (unnamed)",
        Class::Tuning,
        SPTM_CORE_TUNING,
    ),
    entry(
        CORE_TUNING_3,
        "S3_4_C15_C0_3 (unnamed)",
        Class::Tuning,
        SPTM_CORE_TUNING,
    ),
    entry(HID0, "HID0", Class::Tuning, M1N1_TUNING),
    entry(EHID0, "EHID0", Class::Tuning, M1N1_TUNING),
    entry(EHID1, "EHID1", Class::Tuning, M1N1_TUNING),
    entry(EHID20, "EHID20", Class::Tuning, M1N1_TUNING),
    entry(HID6, "HID6", Class::Tuning, M1N1_TUNING),
    entry(EHID7, "EHID7", Class::Tuning, M1N1_TUNING),
    entry(
        TUNING_C1_6,
        "S3_0_C15_C1_6 (unnamed)",
        Class::Tuning,
        SPTM_HID_TUNING,
    ),
    entry(HID2, "HID2", Class::Tuning, SPTM_HID_TUNING),
    entry(EHID2, "EHID2", Class::Tuning, SPTM_HID_TUNING),
    entry(
        TUNING_C2_3,
        "S3_0_C15_C2_3 (unnamed)",
        Class::Tuning,
        SPTM_HID_TUNING,
    ),
    entry(
        TUNING_C2_4,
        "S3_0_C15_C2_4 (unnamed)",
        Class::Tuning,
        SPTM_HID_TUNING,
    ),
    entry(
        TUNING_C2_5,
        "S3_0_C15_C2_5 (unnamed)",
        Class::Tuning,
        SPTM_HID_TUNING,
    ),
    entry(HID17, "HID17", Class::Tuning, SPTM_HID_TUNING),
    entry(
        SPTM_TUNING_SELECT,
        "S3_7_C15_C2_0 (unnamed)",
        Class::Record,
        SPTM_TUNING_SELECT_EVIDENCE,
    ),
    entry(HID26, "HID26", Class::Tuning, M1N1_TUNING),
    entry(HID27, "HID27", Class::Tuning, M1N1_TUNING),
    entry(HID1, "HID1", Class::Tuning, M1N1_TUNING),
    entry(HID21, "HID21", Class::Tuning, M1N1_TUNING),
    entry(HID3, "HID3", Class::Tuning, M1N1_TUNING),
    entry(HID4, "HID4", Class::Tuning, M1N1_TUNING),
    entry(EHID4, "EHID4", Class::Tuning, M1N1_TUNING),
    entry(HID5, "HID5", Class::Tuning, M1N1_TUNING),
    entry(HID7, "HID7", Class::Tuning, M1N1_TUNING),
    entry(HID9, "HID9", Class::Tuning, M1N1_TUNING),
    entry(EHID9, "EHID9", Class::Tuning, M1N1_TUNING),
    entry(HID10, "HID10", Class::Tuning, M1N1_TUNING),
    entry(EHID10, "EHID10", Class::Tuning, M1N1_TUNING),
    entry(HID11, "HID11", Class::Tuning, M1N1_TUNING),
    entry(HID18, "HID18", Class::Tuning, M1N1_TUNING),
    entry(EHID18, "EHID18", Class::Tuning, M1N1_TUNING),
    entry(HID13, "HID13", Class::Tuning, M1N1_TUNING),
    entry(HID14, "HID14", Class::Tuning, M1N1_TUNING),
    entry(HID16, "HID16", Class::Tuning, M1N1_TUNING),
    entry(
        JIT_BOX_MODE,
        "S3_4_C15_C15_4 (unnamed)",
        Class::Control,
        JIT_BOX_MODE_EVIDENCE,
    )
    .permitting(u64::MAX)
    .unmodeled(JIT_BOX_ENABLE, JIT_BOX_UNMODELED),
    entry(
        JIT_RANGE,
        "S3_4_C15_C15_1 (unnamed)",
        Class::Storage,
        JIT_RANGE_EVIDENCE,
    ),
    entry(
        SPTM_KEY_0,
        "S3_4_C15_C10_0 (unnamed)",
        Class::Storage,
        SPTM_KEY_WRITES,
    ),
    entry(
        SPTM_KEY_1,
        "S3_4_C15_C10_1 (unnamed)",
        Class::Storage,
        SPTM_KEY_WRITES,
    ),
    entry(
        SPTM_KEY_2,
        "S3_4_C15_C10_2 (unnamed)",
        Class::Storage,
        SPTM_KEY_WRITES,
    ),
    entry(
        SPTM_KEY_3,
        "S3_4_C15_C10_3 (unnamed)",
        Class::Storage,
        SPTM_KEY_WRITES,
    ),
    entry(AMX_CTX_EL1, "AMX_CTX_EL1", Class::Storage, XNU_AMX_CONTEXT),
    entry(
        PMCR1_COMPANION,
        "S3_6_C15_C8_5 (unnamed)",
        Class::Storage,
        PMCR1_COMPANION_EVIDENCE,
    ),
    entry(PMCR1_GL1, "PMCR1_GL1", Class::Storage, PMCR1_GL1_EVIDENCE),
    // Error records.
    entry(LSU_ERR_STS, "LSU_ERR_STS", Class::Status, ERROR_STATUS),
    entry(E_LSU_ERR_STS, "E_LSU_ERR_STS", Class::Status, ERROR_STATUS),
    entry(FED_ERR_STS, "FED_ERR_STS", Class::Status, ERROR_STATUS),
    entry(E_FED_ERR_STS, "E_FED_ERR_STS", Class::Status, ERROR_STATUS),
    entry(MMU_ERR_STS, "MMU_ERR_STS", Class::Status, ERROR_STATUS),
    entry(E_MMU_ERR_STS, "E_MMU_ERR_STS", Class::Status, ERROR_STATUS),
    entry(DPC_ERR_STS, "DPC_ERR_STS", Class::Status, DPC_ERROR_STATUS),
    entry(L2C_ERR_STS, "L2C_ERR_STS", Class::Status, L2C_ERROR_STATUS)
        .cluster()
        .handing_off(L2C_ERR_STS_HANDOFF),
    entry(L2C_ERR_ADR, "L2C_ERR_ADR", Class::Record, L2C_ERROR_RECORD),
    entry(L2C_ERR_INF, "L2C_ERR_INF", Class::Record, L2C_ERROR_RECORD),
    // The uncore performance monitor, one per cluster.
    entry(UPMCR0, "UPMCR0", Class::Control, UNCORE_CONTROL)
        .cluster()
        .permitting(UPMCR0_PERMITTED)
        .only(&UPMCR0_16_CHIPS),
    entry(UPMCR0, "UPMCR0", Class::Control, UNCORE_CONTROL_24)
        .cluster()
        .permitting(UPMCR0_PMI_ENABLES_24)
        .only(&[Chip::AppleM6]),
    entry(UPMESR0_EL1, "UPMESR0_EL1", Class::Storage, UNCORE_SELECT).cluster(),
    entry(UPMESR1_EL1, "UPMESR1_EL1", Class::Storage, UNCORE_SELECT).cluster(),
    entry(UPMECM0_EL1, "UPMECM0_EL1", Class::Storage, UNCORE_SELECT).cluster(),
    entry(UPMECM1_EL1, "UPMECM1_EL1", Class::Storage, UNCORE_SELECT).cluster(),
    entry(UPMECM2_EL1, "UPMECM2_EL1", Class::Storage, UNCORE_SELECT).cluster(),
    entry(UPMECM3_EL1, "UPMECM3_EL1", Class::Storage, UNCORE_SELECT).cluster(),
    entry(UPMPCM_EL1, "UPMPCM_EL1", Class::Storage, UNCORE_SELECT).cluster(),
    entry(UPMSR, "UPMSR", Class::Status, UNCORE_STATUS).cluster(),
    entry(UPMC0, "UPMC0", Class::Counter, UNCORE_COUNTER).cluster(),
    entry(UPMC1, "UPMC1", Class::Counter, UNCORE_COUNTER).cluster(),
    entry(UPMC2, "UPMC2", Class::Counter, UNCORE_COUNTER).cluster(),
    entry(UPMC3, "UPMC3", Class::Counter, UNCORE_COUNTER).cluster(),
    entry(UPMC4, "UPMC4", Class::Counter, UNCORE_COUNTER).cluster(),
    entry(UPMC5, "UPMC5", Class::Counter, UNCORE_COUNTER).cluster(),
    entry(UPMC6, "UPMC6", Class::Counter, UNCORE_COUNTER).cluster(),
    entry(UPMC7, "UPMC7", Class::Counter, UNCORE_COUNTER).cluster(),
    entry(UPMC8, "UPMC8", Class::Counter, UNCORE_COUNTER).cluster(),
    entry(UPMC9, "UPMC9", Class::Counter, UNCORE_COUNTER).cluster(),
    entry(UPMC10, "UPMC10", Class::Counter, UNCORE_COUNTER).cluster(),
    entry(UPMC11, "UPMC11", Class::Counter, UNCORE_COUNTER).cluster(),
    entry(UPMC12, "UPMC12", Class::Counter, UNCORE_COUNTER).cluster(),
    entry(UPMC13, "UPMC13", Class::Counter, UNCORE_COUNTER).cluster(),
    entry(UPMC14, "UPMC14", Class::Counter, UNCORE_COUNTER).cluster(),
    entry(UPMC15, "UPMC15", Class::Counter, UNCORE_COUNTER).cluster(),
    // AppleT8112CLPC's core, memory-stall and cluster samplers and the
    // energy accumulators.
    entry(
        AON_ENABLE,
        "S3_7_C15_C5_0 (unnamed)",
        Class::Control,
        AON_ENABLE_EVIDENCE,
    )
    .permitting(AON_ENABLED)
    .unmodeled(AON_ENABLED, CORE_COUNTERS_UNMODELED),
    entry(
        AON_SELECT_0,
        "S3_7_C15_C1_3 (unnamed)",
        Class::Storage,
        AON_SELECT_EVIDENCE,
    ),
    entry(
        AON_SELECT_1,
        "S3_7_C15_C3_3 (unnamed)",
        Class::Storage,
        AON_SELECT_EVIDENCE,
    ),
    entry(
        AON_SELECT_2,
        "S3_7_C15_C5_3 (unnamed)",
        Class::Storage,
        AON_SELECT_EVIDENCE,
    ),
    entry(
        AON_COUNTER_0,
        "S3_7_C15_C0_3 (unnamed)",
        Class::Counter,
        AON_COUNTER_EVIDENCE,
    ),
    entry(
        AON_COUNTER_1,
        "S3_7_C15_C2_3 (unnamed)",
        Class::Counter,
        AON_COUNTER_EVIDENCE,
    ),
    entry(
        AON_COUNTER_2,
        "S3_7_C15_C4_3 (unnamed)",
        Class::Counter,
        AON_COUNTER_EVIDENCE,
    ),
    // The fourth core counter of the M1 and M2 Pro, Max and Ultra and the
    // M3-and-later CLPCs.
    entry(
        AON_SELECT_3,
        "S3_7_C15_C7_3 (unnamed)",
        Class::Storage,
        AON_FOURTH_EVIDENCE,
    ),
    entry(
        AON_COUNTER_3,
        "S3_7_C15_C6_3 (unnamed)",
        Class::Counter,
        AON_FOURTH_EVIDENCE,
    ),
    entry(
        MEM_STALL_SELECT_0,
        "S3_1_C15_C0_3 (unnamed)",
        Class::Control,
        MEM_STALL_SELECT_EVIDENCE,
    )
    .permitting(u64::MAX)
    .unmodeled(u64::MAX, MEM_STALL_UNMODELED),
    entry(
        MEM_STALL_SELECT_1,
        "S3_1_C15_C1_3 (unnamed)",
        Class::Control,
        MEM_STALL_SELECT_EVIDENCE,
    )
    .permitting(u64::MAX)
    .unmodeled(u64::MAX, MEM_STALL_UNMODELED),
    entry(
        MEM_STALL_SELECT_2,
        "S3_1_C15_C2_3 (unnamed)",
        Class::Control,
        MEM_STALL_WIDE_SELECT_EVIDENCE,
    )
    .permitting(u64::MAX)
    .unmodeled(u64::MAX, MEM_STALL_UNMODELED),
    entry(
        MEM_STALL_SELECT_3,
        "S3_1_C15_C3_3 (unnamed)",
        Class::Control,
        MEM_STALL_WIDE_SELECT_EVIDENCE,
    )
    .permitting(u64::MAX)
    .unmodeled(u64::MAX, MEM_STALL_UNMODELED),
    entry(
        MEM_STALL_COUNTER_4,
        "S3_1_C15_C12_3 (unnamed)",
        Class::Counter,
        MEM_STALL_WIDE_COUNTER_EVIDENCE,
    ),
    entry(
        MEM_STALL_COUNTER_5,
        "S3_1_C15_C13_3 (unnamed)",
        Class::Counter,
        MEM_STALL_WIDE_COUNTER_EVIDENCE,
    ),
    entry(
        MEM_STALL_COUNTER_6,
        "S3_1_C15_C14_3 (unnamed)",
        Class::Counter,
        MEM_STALL_WIDE_COUNTER_EVIDENCE,
    ),
    entry(
        MEM_STALL_COUNTER_7,
        "S3_1_C15_C15_3 (unnamed)",
        Class::Counter,
        MEM_STALL_WIDE_COUNTER_EVIDENCE,
    ),
    entry(
        MEM_STALL_COUNTER_0,
        "S3_1_C15_C8_3 (unnamed)",
        Class::Counter,
        MEM_STALL_COUNTER_EVIDENCE,
    ),
    entry(
        MEM_STALL_COUNTER_1,
        "S3_1_C15_C9_3 (unnamed)",
        Class::Counter,
        MEM_STALL_COUNTER_EVIDENCE,
    ),
    entry(
        MEM_STALL_COUNTER_2,
        "S3_1_C15_C10_3 (unnamed)",
        Class::Counter,
        MEM_STALL_COUNTER_EVIDENCE,
    ),
    entry(
        MEM_STALL_COUNTER_3,
        "S3_1_C15_C11_3 (unnamed)",
        Class::Counter,
        MEM_STALL_COUNTER_EVIDENCE,
    ),
    entry(
        CLUSTER_COUNTER_C0,
        "S3_7_C15_C0_2 (unnamed)",
        Class::Counter,
        CLUSTER_COUNTER_EVIDENCE,
    )
    .cluster(),
    entry(
        CLUSTER_COUNTER_C2,
        "S3_7_C15_C2_2 (unnamed)",
        Class::Counter,
        CLUSTER_COUNTER_EVIDENCE,
    )
    .cluster(),
    entry(
        CLUSTER_COUNTER_C4,
        "S3_7_C15_C4_2 (unnamed)",
        Class::Counter,
        CLUSTER_COUNTER_EVIDENCE,
    )
    .cluster(),
    entry(
        CLUSTER_COUNTER_C6,
        "S3_7_C15_C6_2 (unnamed)",
        Class::Counter,
        CLUSTER_COUNTER_EVIDENCE,
    )
    .cluster(),
    entry(
        CLUSTER_COUNTER_C8,
        "S3_7_C15_C8_2 (unnamed)",
        Class::Counter,
        CLUSTER_COUNTER_EVIDENCE,
    )
    .cluster(),
    entry(
        CLUSTER_COUNTER_C10,
        "S3_7_C15_C10_2 (unnamed)",
        Class::Counter,
        CLUSTER_COUNTER_EVIDENCE,
    )
    .cluster(),
    entry(
        CLUSTER_COUNTER_C12,
        "S3_7_C15_C12_2 (unnamed)",
        Class::Counter,
        CLUSTER_COUNTER_EVIDENCE,
    )
    .cluster(),
    entry(
        CORE_NRG_ACC_DAT,
        "CORE_NRG_ACC_DAT",
        Class::Energy,
        ENERGY_EVIDENCE,
    ),
    entry(
        CORE_SRM_NRG_ACC_DAT,
        "CORE_SRM_NRG_ACC_DAT",
        Class::Energy,
        ENERGY_EVIDENCE,
    ),
    // Per-thread Rosetta (TSO) state, nested-virtualization and
    // guarded-level syndromes.
    entry(ASPSR_EL1, "ASPSR_EL1", Class::Control, ASPSR_EVIDENCE),
    entry(AFPCR_EL0, "AFPCR_EL0", Class::Control, AFPCR_EVIDENCE),
    entry(AVNCR_EL2, "AVNCR_EL2", Class::Control, AVNCR_EVIDENCE).el2(),
    entry(
        GUARDED_HPFAR,
        "S3_6_C15_C1_1 (unnamed)",
        Class::Record,
        GUARDED_HPFAR_EVIDENCE,
    )
    .el2(),
    entry(
        AFSR1_GL2,
        "AFSR1_GL2",
        Class::Record,
        GUARDED_AFSR1_EVIDENCE,
    )
    .el2(),
    entry(
        AFSR1_GL12,
        "AFSR1_GL12",
        Class::Scratch,
        GUEST_AFSR1_EVIDENCE,
    )
    .el2(),
    // The hypervisor's VM state.
    entry(
        APVMKEYLO_EL2,
        "APVMKEYLO_EL2",
        Class::Storage,
        VM_KEY_EVIDENCE,
    )
    .el2(),
    entry(
        APVMKEYHI_EL2,
        "APVMKEYHI_EL2",
        Class::Storage,
        VM_KEY_EVIDENCE,
    )
    .el2(),
    entry(APSTS_EL12, "APSTS_EL12", Class::Storage, VM_STATUS_EVIDENCE).el2(),
    entry(
        VM_COUNTER_OFFSET,
        "S3_1_C15_C9_4 (unnamed)",
        Class::Guest,
        VM_COUNTER_OFFSET_EVIDENCE,
    )
    .el2(),
    guest(VMSA_LOCK_EL12, "VMSA_LOCK_EL12"),
    guest(SPRR_UMPRR_EL12, "SPRR_UMPRR_EL12"),
    guest(SPRR_UPERM_SH1_EL12, "SPRR_UPERM_SH1_EL12"),
    guest(SPRR_UPERM_SH2_EL12, "SPRR_UPERM_SH2_EL12"),
    guest(SPRR_UPERM_SH3_EL12, "SPRR_UPERM_SH3_EL12"),
    guest(CTRR_A_LWR_EL12, "CTRR_A_LWR_EL12"),
    guest(CTRR_A_UPR_EL12, "CTRR_A_UPR_EL12"),
    guest(CTRR_B_LWR_EL12, "CTRR_B_LWR_EL12"),
    guest(CTRR_B_UPR_EL12, "CTRR_B_UPR_EL12"),
    guest(CTRR_CTL_EL12, "CTRR_CTL_EL12"),
    guest(CTRR_LOCK_EL12, "CTRR_LOCK_EL12"),
    guest(GUEST_C14_0, "S3_4_C15_C14_0 (unnamed)"),
    guest(GUEST_C14_1, "S3_4_C15_C14_1 (unnamed)"),
    guest(GUEST_C14_2, "S3_4_C15_C14_2 (unnamed)"),
    guest(GUEST_C14_3, "S3_4_C15_C14_3 (unnamed)"),
    guest(GUEST_C15_2, "S3_4_C15_C15_2 (unnamed)"),
    guest(GUEST_C15_5, "S3_4_C15_C15_5 (unnamed)"),
    guest(VM_TMR_LR_EL2, "VM_TMR_LR_EL2"),
    guest(APGAKEYLO_EL12, "APGAKeyLo_EL12"),
    guest(APGAKEYHI_EL12, "APGAKeyHi_EL12"),
    guest(KERNKEYLO_EL12, "KERNKEYLO_EL12"),
    guest(KERNKEYHI_EL12, "KERNKEYHI_EL12"),
    guest(SPRR_PMPRR_EL12, "SPRR_PMPRR_EL12"),
    guest(SPRR_PPERM_SH1_EL12, "SPRR_PPERM_SH1_EL12"),
    guest(SPRR_PPERM_SH2_EL12, "SPRR_PPERM_SH2_EL12"),
    guest(SPRR_PPERM_SH3_EL12, "SPRR_PPERM_SH3_EL12"),
    guest(APIAKEYLO_EL12, "APIAKeyLo_EL12"),
    guest(APIAKEYHI_EL12, "APIAKeyHi_EL12"),
    guest(APIBKEYLO_EL12, "APIBKeyLo_EL12"),
    guest(APIBKEYHI_EL12, "APIBKeyHi_EL12"),
    guest(APDAKEYLO_EL12, "APDAKeyLo_EL12"),
    guest(APDAKEYHI_EL12, "APDAKeyHi_EL12"),
    guest(APDBKEYLO_EL12, "APDBKeyLo_EL12"),
    guest(APDBKEYHI_EL12, "APDBKeyHi_EL12"),
    guest(GUEST_C12_7, "S3_6_C15_C12_7 (unnamed)"),
    guest(SPRR_CONFIG_EL12, "SPRR_CONFIG_EL12"),
    guest(SPRR_AMRANGE_EL12, "SPRR_AMRANGE_EL12"),
    // The M3-and-later chips' boot and power registers.
    held(AHCR_EL2, "AHCR_EL2", AHCR_EVIDENCE, AHCR_UNMODELED).el2(),
    held(
        INIT_C0_7,
        "S3_4_C15_C0_7 (unnamed)",
        INIT_C0_7_EVIDENCE,
        INIT_C0_7_UNMODELED,
    ),
    entry(
        CPM_PWRDN_CTL,
        "S3_5_C15_C6_2 (XNU CPM_PWRDN_CTL)",
        Class::Tuning,
        CPM_PWRDN_CTL_EVIDENCE,
    ),
    entry(
        SPTM_CLEARED_C4_7,
        "S3_1_C15_C4_7 (unnamed)",
        Class::Control,
        SPTM_CLEARS,
    ),
    entry(
        SPTM_CLEARED_C9_7,
        "S3_1_C15_C9_7 (unnamed)",
        Class::Control,
        SPTM_CLEARS,
    ),
    entry(
        IRG_KEY_0,
        "S3_1_C15_C11_2 (unnamed)",
        Class::Storage,
        IRG_KEY_EVIDENCE,
    ),
    entry(
        IRG_KEY_1,
        "S3_1_C15_C12_2 (unnamed)",
        Class::Storage,
        IRG_KEY_EVIDENCE,
    ),
    entry(
        IRG_KEY_2,
        "S3_1_C15_C13_2 (unnamed)",
        Class::Storage,
        IRG_KEY_EVIDENCE,
    ),
    entry(
        IRG_KEY_3,
        "S3_1_C15_C14_2 (unnamed)",
        Class::Storage,
        IRG_KEY_EVIDENCE,
    ),
    entry(
        IRG_KEY_4,
        "S3_1_C15_C15_2 (unnamed)",
        Class::Storage,
        IRG_KEY_EVIDENCE,
    ),
    // The M5 and M6 P-cluster L2 error records.
    entry(PL2_ERR_STS, "PL2_ERR_STS", Class::Status, PL2_ERROR_STATUS),
    entry(PL2_ERR_ADR, "PL2_ERR_ADR", Class::Record, PL2_ERROR_RECORD),
    entry(PL2_ERR_INF, "PL2_ERR_INF", Class::Record, PL2_ERROR_RECORD),
    // The later CLPCs' fifth core counter and further cluster counters.
    entry(
        AON_SELECT_4,
        "S3_7_C15_C9_3 (unnamed)",
        Class::Storage,
        AON_FIFTH_EVIDENCE,
    ),
    entry(
        AON_COUNTER_4,
        "S3_7_C15_C8_3 (unnamed)",
        Class::Counter,
        AON_FIFTH_EVIDENCE,
    ),
    entry(
        AON_SELECT_5,
        "S3_7_C15_C11_3 (unnamed)",
        Class::Storage,
        AON_TENTH_EVIDENCE,
    ),
    entry(
        AON_SELECT_6,
        "S3_7_C15_C13_3 (unnamed)",
        Class::Storage,
        AON_TENTH_EVIDENCE,
    ),
    entry(
        AON_SELECT_7,
        "S3_7_C15_C15_3 (unnamed)",
        Class::Storage,
        AON_TENTH_EVIDENCE,
    ),
    entry(
        AON_SELECT_8,
        "S3_7_C15_C15_0 (unnamed)",
        Class::Storage,
        AON_TENTH_EVIDENCE,
    ),
    entry(
        AON_SELECT_9,
        "S3_7_C15_C15_1 (unnamed)",
        Class::Storage,
        AON_TENTH_EVIDENCE,
    ),
    entry(
        AON_COUNTER_5,
        "S3_7_C15_C10_3 (unnamed)",
        Class::Counter,
        AON_TENTH_EVIDENCE,
    ),
    entry(
        AON_COUNTER_6,
        "S3_7_C15_C12_3 (unnamed)",
        Class::Counter,
        AON_TENTH_EVIDENCE,
    ),
    entry(
        AON_COUNTER_7,
        "S3_7_C15_C14_3 (unnamed)",
        Class::Counter,
        AON_TENTH_EVIDENCE,
    ),
    entry(
        AON_COUNTER_8,
        "S3_7_C15_C14_0 (unnamed)",
        Class::Counter,
        AON_TENTH_EVIDENCE,
    ),
    entry(
        AON_COUNTER_9,
        "S3_7_C15_C14_1 (unnamed)",
        Class::Counter,
        AON_TENTH_EVIDENCE,
    ),
    cluster_counter(CLUSTER_COUNTER_C0_6, "S3_7_C15_C0_6 (unnamed)"),
    cluster_counter(CLUSTER_COUNTER_C2_6, "S3_7_C15_C2_6 (unnamed)"),
    cluster_counter(CLUSTER_COUNTER_C4_6, "S3_7_C15_C4_6 (unnamed)"),
    cluster_counter(CLUSTER_COUNTER_C6_6, "S3_7_C15_C6_6 (unnamed)"),
    cluster_counter(CLUSTER_COUNTER_C8_6, "S3_7_C15_C8_6 (unnamed)"),
    cluster_counter(CLUSTER_COUNTER_C14_2, "S3_7_C15_C14_2 (unnamed)"),
    // The M6's 24-counter uncore monitor, one per cluster.
    held(
        UNCORE_CONTROL_1,
        "S3_7_C15_C10_5 (unnamed)",
        UNCORE_CONTROL_1_EVIDENCE,
        UNCORE_CONTROL_1_UNMODELED,
    )
    .cluster(),
    uncore_select(UNCORE_SELECT_C11_7, "S3_7_C15_C11_7 (unnamed)"),
    uncore_select(UNCORE_SELECT_C12_0, "S3_7_C15_C12_0 (unnamed)"),
    uncore_select(UNCORE_SELECT_C12_1, "S3_7_C15_C12_1 (unnamed)"),
    uncore_select(UNCORE_SELECT_C12_5, "S3_7_C15_C12_5 (unnamed)"),
    uncore_select(UNCORE_MASK_4, "S3_6_C11_C12_6 (unnamed)"),
    uncore_select(UNCORE_MASK_5, "S3_6_C11_C12_7 (unnamed)"),
    uncore_counter(UNCORE_COUNTER_16, "S3_7_C15_C12_7 (unnamed)"),
    uncore_counter(UNCORE_COUNTER_17, "S3_7_C15_C13_0 (unnamed)"),
    uncore_counter(UNCORE_COUNTER_18, "S3_7_C15_C13_1 (unnamed)"),
    uncore_counter(UNCORE_COUNTER_19, "S3_7_C15_C13_7 (unnamed)"),
    uncore_counter(UNCORE_COUNTER_20, "S3_7_C15_C14_7 (unnamed)"),
    uncore_counter(UNCORE_COUNTER_21, "S3_7_C15_C15_7 (unnamed)"),
    uncore_counter(UNCORE_COUNTER_22, "S3_7_C15_C10_7 (unnamed)"),
    uncore_counter(UNCORE_COUNTER_23, "S3_7_C15_C10_1 (unnamed)"),
    // Processor trace: AppleProcessorTrace's sessions, the SPTMs' cputrace
    // endpoints and XNU's software trace messages.
    trace(TRACE_C1_0, "S3_5_C15_C1_0 (unnamed)", TRACE_SESSION),
    trace(TRACE_C1_5, "S3_5_C15_C1_5 (unnamed)", TRACE_SESSION),
    trace(TRACE_C1_7, "S3_5_C15_C1_7 (unnamed)", TRACE_SESSION),
    trace(TRACE_C2_1, "S3_5_C15_C2_1 (unnamed)", TRACE_SESSION),
    trace(TRACE_C2_2, "S3_5_C15_C2_2 (unnamed)", TRACE_SESSION),
    trace(TRACE_C2_3, "S3_5_C15_C2_3 (unnamed)", TRACE_SESSION),
    trace(TRACE_C2_4, "S3_5_C15_C2_4 (unnamed)", TRACE_SESSION),
    trace(TRACE_C2_5, "S3_5_C15_C2_5 (unnamed)", TRACE_SESSION),
    trace(TRACE_C3_0, "S3_5_C15_C3_0 (unnamed)", TRACE_SESSION),
    trace(TRACE_C3_4, "S3_5_C15_C3_4 (unnamed)", TRACE_SESSION),
    trace(TRACE_C4_6, "S3_5_C15_C4_6 (unnamed)", TRACE_SESSION),
    trace(TRACE_C10_6, "S3_5_C15_C10_6 (unnamed)", TRACE_SESSION),
    trace(TRACE_C10_7, "S3_5_C15_C10_7 (unnamed)", TRACE_SESSION),
    trace(TRACE_C11_1, "S3_5_C15_C11_1 (unnamed)", TRACE_SESSION),
    trace(TRACE_C11_3, "S3_5_C15_C11_3 (unnamed)", TRACE_SESSION),
    trace(
        PROD_TRACE_BASE_0,
        "S3_5_C15_C3_6 (SPTM PROD_TRC_STRM_BASE)",
        SPTM_TRACE,
    ),
    trace(
        PROD_TRACE_BASE_1,
        "S3_5_C15_C3_7 (SPTM PROD_TRC_STRM_BASE)",
        SPTM_TRACE,
    ),
    trace(PROD_TRACE_ENABLE, "S3_5_C15_C11_2 (unnamed)", SPTM_TRACE),
    trace(TRACE_VA_C4_7, "S3_5_C15_C4_7 (unnamed)", SPTM_TRACE),
    trace(TRACE_VA_C5_7, "S3_5_C15_C5_7 (unnamed)", SPTM_TRACE),
    entry(
        PROD_TRACE_CORE,
        "S3_5_C15_C1_6 (unnamed)",
        Class::Control,
        PROD_TRACE_CORE_EVIDENCE,
    )
    .permitting(u64::MAX)
    .unmodeled(!PROD_TRACE_EL0_MESSAGES, TRACE_UNMODELED),
    entry(
        PROD_TRACE_MASK,
        "S3_5_C15_C3_3 (unnamed)",
        Class::Control,
        PROD_TRACE_MASK_EVIDENCE,
    )
    .permitting(u64::MAX)
    .unmodeled(!PROD_TRACE_P_CORE_MASK, TRACE_UNMODELED),
    entry(
        SW_TRACE_DATA,
        "S3_5_C15_C11_6 (unnamed)",
        Class::Storage,
        SW_TRACE_EVIDENCE,
    ),
];

const fn guest_count() -> usize {
    let mut count = 0;
    let mut index = 0;
    while index < TABLE.len() {
        if matches!(TABLE[index].class, Class::Guest) {
            count += 1;
        }
        index += 1;
    }
    count
}

// Retain catalog order so the same register is reported when several hold
// guest state. Only these entries need inspection at exception return.
static GUEST_ENTRIES: [&Entry; guest_count()] = {
    let mut entries = [&TABLE[0]; guest_count()];
    let mut next = 0;
    let mut index = 0;
    while index < TABLE.len() {
        if matches!(TABLE[index].class, Class::Guest) {
            entries[next] = &TABLE[index];
            next += 1;
        }
        index += 1;
    }
    entries
};

const fn guest(encoding: Encoding, name: &'static str) -> Entry {
    entry(encoding, name, Class::Guest, GUEST_EVIDENCE).el2()
}

const fn cluster_counter(encoding: Encoding, name: &'static str) -> Entry {
    entry(
        encoding,
        name,
        Class::Counter,
        CLUSTER_COUNTER_LATER_EVIDENCE,
    )
    .cluster()
}

const fn uncore_select(encoding: Encoding, name: &'static str) -> Entry {
    entry(encoding, name, Class::Storage, UNCORE_SELECT_24).cluster()
}

const fn uncore_counter(encoding: Encoding, name: &'static str) -> Entry {
    entry(encoding, name, Class::Counter, UNCORE_COUNTER_24).cluster()
}

/// Whether `chip`'s SPTM, TXM, kernelcache or system kext collection
/// accesses `encoding` ([`census`]).
pub fn consumed(encoding: Encoding, chip: Chip) -> bool {
    census::CONSUMED[chip as usize]
        .binary_search(&encoding.selector())
        .is_ok()
}

/// The entry for `encoding` on `chip`: the table's entry for the chip,
/// where the chip's components access the encoding.
pub fn lookup(encoding: Encoding, chip: Chip) -> Option<&'static Entry> {
    TABLE
        .iter()
        .find(|entry| entry.encoding == encoding && entry.chips.contains(&chip))
        .filter(|_| consumed(encoding, chip))
}

/// XNU `MPIDR_PNE` ("pcore not ecore", xnu-7195.141.2 `apple_arm64_regs.h`):
/// MPIDR_EL1 bit 16, set on a P core.
const MPIDR_PNE: u64 = 1 << 16;
/// The energy accumulators' unit: AppleT8112CLPC divides their delta by 40
/// for `energy_estimate_nj` ([`ENERGY_EVIDENCE`]).
pub const ENERGY_UNITS_PER_NANOJOULE: u128 = 40;
const NANOJOULES_PER_MILLIWATT_SECOND: u128 = 1_000_000;
/// The emulated die's modeled power of a running core, E then P, in
/// milliwatts: Stela's fixed choice, not a measurement of any Apple core.
pub const CORE_MILLIWATTS: [u128; 2] = [250, 1000];

/// An accumulator's reading at system count `count`: the ticks since the
/// origin, none before it, times the scale over the frequency, saturated.
fn energy_reading(rate: stela_compiler::x86_64::register_read::Energy, count: u64) -> u64 {
    let ticks = u128::from(count.saturating_sub(rate.origin));
    let units = ticks * u128::from(rate.scale) / u128::from(rate.frequency);
    u64::try_from(units).unwrap_or(u64::MAX)
}

/// What an energy reading accumulates over: the system counter's value and
/// frequency and the reading PE's MPIDR_EL1.
#[derive(Debug, Clone, Copy)]
pub struct Activity {
    pub count: u64,
    pub frequency: u64,
    pub mpidr: u64,
}

/// The slots of a [`Values`]: CRn 15's encodings, then CRn 11's.
const SLOTS: usize = 2 << 10;

/// The slot of `encoding`, of the table's CRns.
const fn slot(encoding: Encoding) -> usize {
    let other = encoding.crn() != IMPDEF_CRN;
    (other as usize) << 10
        | (encoding.op1() as usize) << 7
        | (encoding.crm() as usize) << 3
        | encoding.op2() as usize
}

/// A PE's values, which stay where they are for native code to reach
/// ([`Registers::storage`]).
#[derive(Debug, Clone, PartialEq, Eq)]
struct Values {
    values: Box<[u64]>,
    /// A bit per slot, set by its first write.
    written: Box<[u64]>,
}

impl Default for Values {
    fn default() -> Self {
        Self {
            values: vec![0; SLOTS].into_boxed_slice(),
            written: vec![0; SLOTS / 64].into_boxed_slice(),
        }
    }
}

impl Values {
    fn get(&self, encoding: Encoding) -> Option<u64> {
        let slot = slot(encoding);
        (self.written[slot / 64] >> (slot % 64) & 1 != 0).then(|| self.values[slot])
    }

    fn set(&mut self, encoding: Encoding, value: u64) {
        let slot = slot(encoding);
        self.written[slot / 64] |= 1 << (slot % 64);
        self.values[slot] = value;
    }
}

/// Values and use counts of the table's registers. Native code completes
/// accesses of a register once the engine has completed one, which these
/// counts then leave out: they say which registers were used.
#[derive(Debug, Clone, Default)]
pub struct Registers {
    values: Values,
    /// The system count of the PE's first energy reading.
    energy_origin: Option<u64>,
    /// How CORE_NRG_ACC_DAT and CORE_SRM_NRG_ACC_DAT read, from the PE's
    /// first energy reading on, in that order: native code reads them too
    /// (`x86_64::register_read::energy_thunk`).
    energy_rates: [stela_compiler::x86_64::register_read::Energy; 2],
    /// The `Scope::Cluster` values, shared with the cluster's other PEs.
    cluster: super::shared::Shared<BTreeMap<u16, u64>>,
    pub reads: BTreeMap<u16, u64>,
    pub writes: BTreeMap<u16, u64>,
}

impl Registers {
    /// Share `other`'s cluster registers: a PE starting in a cluster that
    /// already has a powered PE.
    pub fn join(&mut self, other: &Self) {
        self.cluster = other.cluster.clone();
    }

    /// The cluster registers this PE shares, for a PE on another machine.
    pub(crate) fn cluster_registers(&self) -> super::shared::Shared<BTreeMap<u16, u64>> {
        self.cluster.clone()
    }

    /// Share `cluster`, the registers of a PE that runs on another machine.
    pub(crate) fn join_cluster(&mut self, cluster: super::shared::Shared<BTreeMap<u16, u64>>) {
        self.cluster = cluster;
    }

    fn stored(&self, entry: &Entry) -> Option<u64> {
        let key = entry.encoding.selector();
        match entry.scope {
            Scope::Pe => self.values.get(entry.encoding),
            Scope::Cluster => self.cluster.borrow().get(&key).copied(),
        }
    }

    /// The value software last wrote to the PE-scoped `encoding`, without
    /// counting a read.
    pub fn value(&self, encoding: Encoding) -> Option<u64> {
        self.values.get(encoding)
    }

    /// The enable and selectors of the CLPC core performance counters
    /// (`machine::aon`).
    pub fn aon_controls(&self) -> super::aon::Controls {
        let mut controls = super::aon::Controls {
            enabled: self.value(AON_ENABLE).unwrap_or(0) & AON_ENABLED != 0,
            ..Default::default()
        };
        for (selector, encoding) in controls.selectors.iter_mut().zip(AON_SELECTS) {
            *selector = self.value(encoding);
        }
        controls
    }

    /// The address of a [`Entry::stored`] entry's value on this PE, whose
    /// first access the engine completes.
    pub fn storage(&self, entry: &Entry) -> u64 {
        &self.values.values[slot(entry.encoding)] as *const u64 as u64
    }

    /// An `Energy` entry's reading: the energy the PE's core type's modeled
    /// power ([`CORE_MILLIWATTS`]) has drawn, in 1/40 nJ, over the system
    /// counter's time since the PE first read either accumulator, all of it
    /// on the core rail (CORE_NRG_ACC_DAT); the SRAM rail (CORE_SRM_NRG_ACC_DAT)
    /// draws none on the emulated die. It never decreases.
    pub fn energy(&mut self, entry: &Entry, activity: Activity) -> Result<u64, &'static str> {
        if entry.class != Class::Energy || activity.frequency == 0 {
            return Err("energy reading of a register that is no energy accumulator");
        }
        *self.reads.entry(entry.encoding.selector()).or_default() += 1;
        let origin = *self.energy_origin.get_or_insert(activity.count);
        let milliwatts = CORE_MILLIWATTS[usize::from(activity.mpidr & MPIDR_PNE != 0)];
        let core = milliwatts * NANOJOULES_PER_MILLIWATT_SECOND * ENERGY_UNITS_PER_NANOJOULE;
        for (rate, scale) in self.energy_rates.iter_mut().zip([core, 0]) {
            *rate = stela_compiler::x86_64::register_read::Energy {
                origin,
                scale: u64::try_from(scale).expect("a modeled power fits"),
                frequency: activity.frequency,
            };
        }
        let index = usize::from(entry.encoding != CORE_NRG_ACC_DAT);
        Ok(energy_reading(self.energy_rates[index], activity.count))
    }

    /// Where native code reads energy accumulator `entry` once the PE took
    /// an energy reading.
    pub fn energy_storage(&self, entry: &Entry) -> Option<u64> {
        let index = match entry.encoding {
            CORE_NRG_ACC_DAT => 0,
            CORE_SRM_NRG_ACC_DAT => 1,
            _ => return None,
        };
        let rate = &self.energy_rates[index];
        (self.energy_origin.is_some() && rate.frequency != 0)
            .then_some(rate as *const stela_compiler::x86_64::register_read::Energy as u64)
    }

    /// What energy accumulator `entry` reads at system count `count`, if
    /// native code reads it at [`Self::energy_storage`].
    pub fn energy_at(&self, entry: &Entry, count: u64) -> Option<u64> {
        let index = usize::from(entry.encoding != CORE_NRG_ACC_DAT);
        self.energy_storage(entry)
            .map(|_| energy_reading(self.energy_rates[index], count))
    }

    /// A read, or why it cannot complete.
    pub fn read(&mut self, entry: &Entry) -> Result<u64, &'static str> {
        match entry.class {
            Class::Unsupported => return Err(entry.evidence),
            Class::Energy => return Err("energy accumulator read without its activity"),
            _ => {}
        }
        let key = entry.encoding.selector();
        *self.reads.entry(key).or_default() += 1;
        match (self.stored(entry), entry.class) {
            (_, Class::Status) => Ok(entry.handoff),
            (_, Class::Record) => Ok(0),
            (Some(value), _) => Ok(value),
            (None, Class::Storage | Class::Guest) => {
                Err("read of an implementation-defined storage register before its first write")
            }
            (None, _) => Ok(0),
        }
    }

    /// A write, or why it cannot complete.
    pub fn write(&mut self, entry: &Entry, value: u64) -> Result<(), &'static str> {
        if entry.encoding == SIQ_CFG_EL1 && !matches!(value, 0 | 2 | 3) {
            return Err("SIQ_CFG_EL1 mode 1 or nonzero upper bits lack source-qualified effects");
        }
        match entry.class {
            Class::Unsupported => return Err(entry.evidence),
            Class::Record => {
                return Err("write of an implementation-defined record no consumer writes");
            }
            Class::Energy => {
                return Err("write of a core energy accumulator, which no consumer writes");
            }
            Class::Prediction if value & !entry.permitted != 0 => {
                return Err(
                    "ObjC predictor configuration contains a field without source-qualified semantics",
                );
            }
            Class::Control if value & !entry.permitted != 0 => {
                return Err(
                    "implementation-defined control bit for a feature or event this machine \
                     does not model",
                );
            }
            _ => {}
        }
        let key = entry.encoding.selector();
        *self.writes.entry(key).or_default() += 1;
        // A status register records nothing, so a clearing write leaves it 0.
        if entry.class != Class::Status {
            match entry.scope {
                Scope::Pe => self.values.set(entry.encoding, value),
                Scope::Cluster => {
                    self.cluster.borrow_mut().insert(key, value);
                }
            }
        }
        Ok(())
    }

    /// The first `Guest` register holding a nonzero value: state an EL1&0
    /// guest would run without.
    pub fn guest_state(&self) -> Option<&'static Entry> {
        GUEST_ENTRIES
            .iter()
            .copied()
            .find(|entry| self.stored(entry).is_some_and(|value| value != 0))
    }

    /// (entry, reads, writes) for every register the guest touched, once
    /// per encoding.
    pub fn used(&self) -> impl Iterator<Item = (&'static Entry, u64, u64)> + '_ {
        let first = |(index, entry): (usize, &'static Entry)| {
            TABLE[..index]
                .iter()
                .all(|other| other.encoding != entry.encoding)
                .then_some(entry)
        };
        TABLE
            .iter()
            .enumerate()
            .filter_map(first)
            .filter_map(|entry| {
                let key = entry.encoding.selector();
                let reads = self.reads.get(&key).copied().unwrap_or(0);
                let writes = self.writes.get(&key).copied().unwrap_or(0);
                (reads + writes > 0).then_some((entry, reads, writes))
            })
    }
}

#[cfg(test)]
mod tests;
