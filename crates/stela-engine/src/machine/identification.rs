//! The AArch64 identification registers the system engine presents.
//!
//! They advertise exactly what this engine implements, never more than the
//! chip implements: each field is set from one table row whose architectural
//! feature must be in the chip's Apple clang feature set ([`Chip::cap`],
//! from `clang -mcpu=apple-mN
//! --print-enabled-extensions`), or implied by it under the AARCHMRS
//! feature constraints (a generated [`Feature`] the chip has, such as the
//! M4's Armv8.7 FEAT_PAN3, which has no clang extension name), and whose
//! basis must hold here. An
//! instruction feature holds when its sample encodings compile for the
//! system-mode block profile on the host target; any other feature is an
//! engine property stated with its reason, holds only when the kernel
//! level is EL2 as a VHE host, or is a speculation guarantee the host gives
//! ([`Speculation`]). A field with no satisfied row reads its "not
//! implemented" (or "not disclosed") value.
use alloc::vec::Vec;
use stela_compiler::{
    GuestAddress, TargetCpu,
    aarch64::system::registers::Encoding,
    block::{BlockProfile, CompilationPlan, Features},
};
use {super::registers::encodings::*, stela_contracts::Chip, stela_contracts::Feature};

/// Why a row's level holds on this engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Basis {
    /// Every sample (clang-assembled A64 words) compiles for the system-mode
    /// profile on the host target, and the engine completes their exits.
    Instructions(&'static [u32]),
    /// An engine property, true or false, and why.
    Engine(bool, &'static str),
    /// Holds when the profile has EL2, which runs only as a VHE host
    /// (HCR_EL2.E2H set, EL1 never executing).
    Host(&'static str),
    /// Holds when the chip also has the named feature, an engine property
    /// with why: a field that describes the feature's instructions only
    /// within the other's.
    With(&'static str, &'static str),
    /// Holds when the host target has a data-independent-timing target
    /// ([`TargetCpu::data_independent_timing_target`]: it declares Intel's
    /// data operand independent timing and supports a tier the DIT audit
    /// covers), which the engine compiles PSTATE.DIT code for.
    Timing(&'static str),
    /// Holds when the host keeps branch predictions within one guest
    /// context ([`Speculation::contexts`]).
    Contexts(&'static str),
    /// Holds when no data a faulting access would load reaches later
    /// speculative execution ([`Speculation::faulting_data`]).
    FaultingData(&'static str),
    /// Holds on a chip with the Apple generic timer and its 1 GHz scaled
    /// Arm bank ([`super::timer::agt::present`]).
    GenericTimer(&'static str),
    /// Holds when the host's aligned 16-byte SSE accesses are single-copy
    /// atomic ([`TargetCpu::AtomicLoad16`]), which 16-byte-aligned X-register
    /// pairs compile to.
    Atomic16(&'static str),
    /// An instruction feature whose samples compile everywhere, as
    /// [`Basis::Instructions`], but which the field advertises only where the
    /// host has the named native tier (`None`: no tier lowers it natively),
    /// and why. Guest software that probes the field (the kernel derives its
    /// `hw.optional.arm.FEAT_*` sysctls and commpage bits from it) then takes
    /// its own fallback, which translates better than the feature's
    /// emulation. Steering only advises: the PE still runs the feature's
    /// instructions as the chip does, for code that uses them unprobed, so a
    /// steered feature is never [`withheld`] and every field of the feature
    /// reads absent with it ([`steered_off`]).
    Steered(&'static [u32], Option<TargetCpu>, &'static str),
}

/// One steered feature's advertisement on a host target ([`steering`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Steering {
    pub feature: &'static str,
    pub advertised: bool,
    /// The native tier the feature needs, or why it has none.
    pub reason: &'static str,
}

/// The speculation guarantees the host gives guest code, which the
/// ID_AA64PFR0_EL1 CSV2 and CSV3 fields disclose. Arm's 0 there means "not
/// disclosed", so a host that cannot establish the guarantee reports 0.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Speculation {
    /// FEAT_CSV2 and FEAT_CSV2_2: branch targets trained in one guest
    /// context cannot steer speculation in another. The bare-metal host
    /// issues an indirect branch prediction barrier and refills the return
    /// stack buffer at every guest context change
    /// (`engine::PredictionContext`, `Host::isolate_predictions`), where the
    /// CPU enumerates IBPB.
    pub contexts: bool,
    /// FEAT_CSV3: data loaded under speculation by an access with a
    /// permission or domain fault, or read from a System register the
    /// current EL cannot access, is not used by later speculative
    /// instructions. Native accesses fault through the shadow tables on a
    /// host whose loads forward no faulting data (`speculation::probe` on
    /// bare metal: Intel's RDCL_NO, MDS_NO, TAA_NO or no TSX, GDS_NO and
    /// RFDS_NO; AMD's TSA_SQ_NO and TSA_L1_NO with an always-serializing
    /// LFENCE); every access then masks its address into an unmapped entry
    /// on a mispredicted range check (`x86_64::RangeCheck`, checked
    /// at every access), the engine fences each guest RAM read behind its
    /// translation, and closed native System register cells hold no value.
    pub faulting_data: bool,
}

impl Speculation {
    /// A host that establishes no guarantee.
    pub const NONE: Self = Self {
        contexts: false,
        faulting_data: false,
    };
}

/// One level of one field. A field's rows are listed in increasing level;
/// the field reads the highest satisfied level, or `absent`.
#[derive(Debug, Clone, Copy)]
pub struct Row {
    pub register: Encoding,
    pub shift: u8,
    /// Architectural field width, including mixed-width SME, FP8 and AMU fields.
    pub width: u8,
    pub level: u64,
    pub absent: u64,
    /// The `FEAT_` tag the chip must have; `None` for fields the architecture
    /// fixes for any AArch64 PE.
    pub feature: Option<&'static str>,
    pub basis: Basis,
}

impl Row {
    /// Most feature fields have four bits; other layouts are explicit in TABLE.
    const fn with_width(mut self, width: u8) -> Self {
        self.width = width;
        self
    }

    fn mask(&self) -> u64 {
        u64::MAX >> (u64::BITS - u32::from(self.width))
    }
}

const fn row(
    register: Encoding,
    shift: u8,
    level: u64,
    absent: u64,
    feature: Option<&'static str>,
    basis: Basis,
) -> Row {
    Row {
        register,
        shift,
        width: 4,
        level,
        absent,
        feature,
        basis,
    }
}

const fn feature(
    register: Encoding,
    shift: u8,
    level: u64,
    name: &'static str,
    basis: Basis,
) -> Row {
    row(register, shift, level, 0, Some(name), basis)
}

const fn fixed(register: Encoding, shift: u8, level: u64, reason: &'static str) -> Row {
    row(register, shift, level, 0, None, Basis::Engine(true, reason))
}

/// The common four-bit ID field, used for explicit profile masks.
#[cfg(test)]
const ID_FIELD: u64 = 0xf;
/// ID_AA64DFR0_EL1.BRPs and WRPs, the breakpoint and watchpoint counts less
/// one.
const DFR0_BRPS: u8 = 12;
const DFR0_WRPS: u8 = 20;
/// ID_AA64PFR1_EL1.MTE, the memory tagging level.
const PFR1_MTE: u8 = 8;
/// ID_AA64PFR1_EL1.MTE_frac, the asynchronous Tag Check fault support.
const PFR1_MTE_FRAC: u8 = 40;
/// ID_AA64PFR1_EL1.SME, the Scalable Matrix Extension level.
const PFR1_SME: u8 = 24;
/// MTE_frac 0b1111: asynchronous Tag Check faults are not supported (no
/// FEAT_MTE_ASYNC).
const MTE_FRAC_SYNCHRONOUS_ONLY: u64 = 0b1111;

const fn engine(holds: bool, reason: &'static str) -> Basis {
    Basis::Engine(holds, reason)
}

const NESTING: &str = "HCR_EL2.NV traps and reports EL2 to a guest hypervisor at EL1, and \
     NV2 redirects its register accesses to VNCR_EL2 memory (NV1 is RAZ)";
const VHE_HOST: &str = "EL2 with E2H set and the EL1 register redirection";
const STAGE2_HOST: &str = "stage-2 translation of the EL1&0 regime";
const NO_MONITORS: &str = "no Arm performance or trace monitor is modeled";
const ACTIVITY_MONITORS: &str = "machine::amu counts the four architected group 0 events";
const VIRTUAL_OFFSETS: &str = "HCR_EL2.AMVOFFEN applies AMEVCNTVOFF0<n>_EL2 and CNTVOFF_EL2 \
     to EL0 and EL1 counter reads";
const PREDICTION_ISOLATION: &str = "the host isolates branch predictions at every guest \
     context change (IBPB and a return stack refill); SCXTNUM_ELx is held and distinguishes no context";
const FAULTING_DATA: &str = "the host's loads forward no data from a faulting access, every \
     native access masks its address on a mispredicted guest range check, the engine fences \
     guest RAM reads behind their translation, and closed native System register cells are \
     cleared";
const DATA_INDEPENDENT_TIMING: &str = "PSTATE.DIT blocks compile for the host's audited \
     data-independent-timing tier, whose lowerings of every listed instruction use only \
     Intel's data operand independent timing instructions";
const PAUTH_LR: &str = "PACM sets PSTATE.PACM for the next instruction, as IsPACMEnabled \
     allows, and the two-modifier forms and the PACM-modified PAC*SP, AUT*SP, RETA* and \
     *1716 forms compute ComputePAC2QARMA";
const SME_RUNS: &str = "the engine runs streaming mode, ZA and ZT0 at a 512-bit SVL and every \
     SVE and SME form of the chip's features, or takes UNDEFINED where Arm's decode does \
     (engine::scalable)";
const FPMR: &str = "FPMR holds its fields with the scalable state, which clears it as \
     PSTATE.SM changes, behind SCTLR_ELx.EnFPM and HCRX_EL2.EnFPM";
const FP8: &str = "the engine completes F1CVTL, F2CVTL, BF1CVTL, BF2CVTL and FCVTN (FP8) \
     with FPMR's formats and scales, FSCALE compiles, and the SME FP8 forms run in streaming \
     mode (engine::scalable)";
const MTE: &str = "IRG, ADDG and SUBG run in the engine under GCR_EL1 and RGSR_EL1, GMI and \
     SUBP(S) compile, and PSTATE.TCO has its MSR and MRS forms and is set by exception entry";
const MTE2: &str = "the engine keeps an Allocation Tag for each 16-byte granule of Tagged \
     memory, runs the tag loads and stores and DC GVA and GZVA, and checks the tags of \
     tag-checked accesses, synchronously";
const PREDICTION_RESTRICTION: &str = "CFP, DVP, CPP and (FEAT_SPECRES2) COSP RCTX run their \
     generated accessors, with SCTLR_ELx.EnRCTX and HFGITR_EL2 traps, and complete with no \
     effect: the engine keeps no prediction state a later context could use";

/// FEAT_BF16 has no native lowering on any host tier: BFDOT, BFMMLA and
/// BFMLAL round to odd in SSE2 sequences of 100 to 400 instructions, and a
/// zero or subnormal product input exits. Unadvertised, as on the M1, guest
/// libraries keep their single-precision paths.
const BF16_STEERING: &str = "no host tier lowers BF16 natively (AVX512-BF16 is unused)";
/// FJCVTZS completes natively only for a normal input whose truncation fits
/// 32 bits; minus zero, special and wrapping inputs exit. Unadvertised, a JIT
/// truncates with FCVTZS and its own native wrap.
const JSCVT_STEERING: &str = "FJCVTZS exits for minus zero, special and out-of-range inputs";

/// Every ID register the table defines; all their other fields read 0.
pub const REGISTERS: &[Encoding] = &[
    ID_AA64PFR0_EL1,
    ID_AA64PFR1_EL1,
    ID_AA64ISAR0_EL1,
    ID_AA64ISAR1_EL1,
    ID_AA64ISAR2_EL1,
    ID_AA64MMFR0_EL1,
    ID_AA64MMFR1_EL1,
    ID_AA64MMFR2_EL1,
    ID_AA64DFR0_EL1,
    ID_AA64DFR1_EL1,
    ID_AA64AFR0_EL1,
    ID_AA64AFR1_EL1,
    ID_AA64ZFR0_EL1,
    ID_AA64SMFR0_EL1,
    ID_AA64PFR2_EL1,
    ID_AA64FPFR0_EL1,
    ID_AA64DFR2_EL1,
    ID_AA64ISAR3_EL1,
    ID_AA64MMFR3_EL1,
    ID_AA64MMFR4_EL1,
    CTR_EL0,
    AIDR_EL1,
    // GMID_EL1.BS: LDGM and STGM move the tags of 64 bytes (FEAT_MTE2).
    GMID_EL1,
    // Without a board identity's caches (`machine::caches`) none are
    // visible: memory is coherent and CTR_EL0.IDC holds, so CLIDR_EL1
    // reports no cache level and LoC, LoUU and LoUIS 0, and CCSIDR_EL1
    // (UNKNOWN for an unimplemented cache) reads 0.
    CLIDR_EL1,
    CCSIDR_EL1,
    // No revision-specific information.
    REVIDR_EL1,
    // FEAT_LOR with no LORegions or descriptors (LORID_EL1 LR = LD = 0),
    // FEAT_MPAM with only PARTID 0 and PMG 0 and no MPAMHCR_EL2 or virtual
    // PARTID mapping, and FEAT_RAS with no error records (ERRIDR_EL1.NUM 0).
    LORID_EL1,
    MPAMIDR_EL1,
    ERRIDR_EL1,
    // No external debug ROM table.
    MDRAR_EL1,
    DBGAUTHSTATUS_EL1,
    // The activity monitors' configuration (`machine::amu`), 0 and
    // unreachable on a chip without them.
    AMCFGR_EL0,
    AMCGCR_EL0,
    AMCG1IDR_EL0,
];

pub const TABLE: &[Row] = &[
    // ID_AA64PFR0_EL1
    fixed(ID_AA64PFR0_EL1, 0, 1, "EL0 is AArch64-only"),
    fixed(ID_AA64PFR0_EL1, 4, 1, "EL1 is AArch64-only"),
    row(ID_AA64PFR0_EL1, 8, 1, 0, None, Basis::Host(VHE_HOST)),
    row(
        ID_AA64PFR0_EL1,
        16,
        0,
        0xf,
        Some("FEAT_FP"),
        Basis::Instructions(&[0x1e22_2820]),
    ),
    feature(
        ID_AA64PFR0_EL1,
        16,
        1,
        "FEAT_FP16",
        Basis::Instructions(&[0x1ee2_2820]),
    ),
    row(
        ID_AA64PFR0_EL1,
        20,
        0,
        0xf,
        Some("FEAT_AdvSIMD"),
        Basis::Instructions(&[0x4ea2_8420]),
    ),
    feature(
        ID_AA64PFR0_EL1,
        20,
        1,
        "FEAT_FP16",
        Basis::Instructions(&[0x4e42_1420]),
    ),
    feature(
        ID_AA64PFR0_EL1,
        28,
        1,
        "FEAT_RAS",
        engine(
            true,
            "no error records (ERRIDR_EL1.NUM 0, ERX* RAZ/WI); ESB and virtual SErrors",
        ),
    ),
    feature(
        ID_AA64PFR0_EL1,
        28,
        2,
        "FEAT_RASv1p1",
        engine(true, "ERXMISC2/3 and ERXPFG* of no error record, RAZ/WI"),
    ),
    feature(
        ID_AA64PFR0_EL1,
        28,
        3,
        "FEAT_RASv2",
        engine(
            true,
            "ERXGSR_EL1 of no error record group reads 0; ESB defers a virtual \
             SError as before",
        ),
    ),
    feature(
        ID_AA64PFR0_EL1,
        36,
        1,
        "FEAT_SEL2",
        engine(false, "no Secure state"),
    ),
    feature(
        ID_AA64PFR0_EL1,
        40,
        1,
        "FEAT_MPAM",
        engine(
            true,
            "MPAM v1.0 with only PARTID 0 and PMG 0, no MPAMHCR_EL2 or VPM",
        ),
    ),
    feature(
        ID_AA64PFR0_EL1,
        44,
        1,
        "FEAT_AMUv1",
        engine(true, ACTIVITY_MONITORS),
    ),
    feature(
        ID_AA64PFR0_EL1,
        44,
        2,
        "FEAT_AMUv1p1",
        engine(true, VIRTUAL_OFFSETS),
    ),
    // AMCFGR_EL0: N 3 (four counters), SIZE 63 (64 bits), HDBG 1 and NCG 0
    // (group 0 alone); AMCGCR_EL0: CG0NC 4 and CG1NC 0 (`machine::amu`).
    // AMCG1IDR_EL0 reads 0: no group 1 counter or offset.
    feature(
        AMCFGR_EL0,
        0,
        3,
        "FEAT_AMUv1",
        engine(true, ACTIVITY_MONITORS),
    )
    .with_width(8),
    feature(
        AMCFGR_EL0,
        8,
        63,
        "FEAT_AMUv1",
        engine(true, ACTIVITY_MONITORS),
    )
    .with_width(6),
    feature(
        AMCFGR_EL0,
        24,
        1,
        "FEAT_AMUv1",
        engine(true, ACTIVITY_MONITORS),
    )
    .with_width(1),
    feature(
        AMCGCR_EL0,
        0,
        4,
        "FEAT_AMUv1",
        engine(true, ACTIVITY_MONITORS),
    )
    .with_width(8),
    feature(
        ID_AA64PFR0_EL1,
        48,
        1,
        "FEAT_DIT",
        Basis::Timing(DATA_INDEPENDENT_TIMING),
    ),
    // CSV2 2 (FEAT_CSV2_2, which the chips' clang features name, above
    // Armv8.5's FEAT_CSV2), as the chip has it, where the host gives the
    // guarantee ([`Speculation`]). CSV2_frac stays 0, its only value with
    // CSV2 2. CSV3 1 likewise, where the host gives no faulting data to
    // speculation.
    row(
        ID_AA64PFR0_EL1,
        56,
        1,
        0,
        None,
        Basis::Contexts(PREDICTION_ISOLATION),
    ),
    feature(
        ID_AA64PFR0_EL1,
        56,
        2,
        "FEAT_CSV2_2",
        Basis::Contexts(PREDICTION_ISOLATION),
    ),
    // FEAT_CSV3 is mandatory from Armv8.5, which every chip's profile
    // implies, so no per-chip feature names it.
    row(
        ID_AA64PFR0_EL1,
        60,
        1,
        0,
        None,
        Basis::FaultingData(FAULTING_DATA),
    ),
    // ID_AA64PFR1_EL1
    feature(
        ID_AA64PFR1_EL1,
        0,
        1,
        "FEAT_BTI",
        engine(
            true,
            "the engine sets PSTATE.BTYPE at block exits and checks the first \
             instruction of a guarded-page block against it",
        ),
    ),
    feature(
        ID_AA64PFR1_EL1,
        4,
        1,
        "FEAT_SSBS",
        engine(true, "PSTATE.SSBS and MSR SSBS, #imm"),
    ),
    feature(
        ID_AA64PFR1_EL1,
        4,
        2,
        "FEAT_SSBS2",
        engine(true, "MRS and MSR of SSBS"),
    ),
    // ID_AA64PFR1_EL1.MTE and MTE_frac: synchronous Tag Check faults only
    // (MTE_frac 0b1111, no FEAT_MTE_ASYNC).
    feature(ID_AA64PFR1_EL1, PFR1_MTE, 1, "FEAT_MTE", engine(true, MTE)),
    feature(
        ID_AA64PFR1_EL1,
        PFR1_MTE,
        2,
        "FEAT_MTE2",
        engine(true, MTE2),
    ),
    feature(
        ID_AA64PFR1_EL1,
        PFR1_MTE_FRAC,
        MTE_FRAC_SYNCHRONOUS_ONLY,
        "FEAT_MTE2",
        engine(true, MTE2),
    ),
    feature(ID_AA64PFR1_EL1, 24, 1, "FEAT_SME", engine(true, SME_RUNS)),
    feature(ID_AA64PFR1_EL1, 24, 2, "FEAT_SME2", engine(true, SME_RUNS)),
    // ID_AA64SMFR0_EL1: F32F32, BI32I32, B16F32, F16F32, I8I32, F8F32,
    // F8F16, F16F16, B16B16, I16I32, F64F64, I16I64, SMEver and LUTv2, with
    // the levels AARCHMRS requires of each feature. FA64 (bit 63) reads 0:
    // no Apple chip has FEAT_SME_FA64.
    feature(ID_AA64SMFR0_EL1, 32, 1, "FEAT_SME", engine(true, SME_RUNS)).with_width(1),
    feature(ID_AA64SMFR0_EL1, 33, 1, "FEAT_SME2", engine(true, SME_RUNS)).with_width(1),
    feature(ID_AA64SMFR0_EL1, 34, 1, "FEAT_SME", engine(true, SME_RUNS)).with_width(1),
    feature(ID_AA64SMFR0_EL1, 35, 1, "FEAT_SME", engine(true, SME_RUNS)).with_width(1),
    feature(
        ID_AA64SMFR0_EL1,
        36,
        0xf,
        "FEAT_SME",
        engine(true, SME_RUNS),
    ),
    feature(
        ID_AA64SMFR0_EL1,
        40,
        1,
        "FEAT_SME_F8F32",
        engine(true, SME_RUNS),
    )
    .with_width(1),
    feature(
        ID_AA64SMFR0_EL1,
        41,
        1,
        "FEAT_SME_F8F16",
        engine(true, SME_RUNS),
    )
    .with_width(1),
    feature(
        ID_AA64SMFR0_EL1,
        42,
        1,
        "FEAT_SME_F16F16",
        engine(true, SME_RUNS),
    )
    .with_width(1),
    feature(
        ID_AA64SMFR0_EL1,
        43,
        1,
        "FEAT_SME_B16B16",
        engine(true, SME_RUNS),
    )
    .with_width(1),
    feature(
        ID_AA64SMFR0_EL1,
        44,
        0b0101,
        "FEAT_SME2",
        engine(true, SME_RUNS),
    ),
    feature(
        ID_AA64SMFR0_EL1,
        48,
        1,
        "FEAT_SME_F64F64",
        engine(true, SME_RUNS),
    )
    .with_width(1),
    feature(
        ID_AA64SMFR0_EL1,
        52,
        0xf,
        "FEAT_SME_I16I64",
        engine(true, SME_RUNS),
    ),
    feature(ID_AA64SMFR0_EL1, 56, 1, "FEAT_SME2", engine(true, SME_RUNS)),
    feature(
        ID_AA64SMFR0_EL1,
        56,
        2,
        "FEAT_SME2p1",
        engine(true, SME_RUNS),
    ),
    feature(
        ID_AA64SMFR0_EL1,
        60,
        1,
        "FEAT_SME_LUTv2",
        engine(true, SME_RUNS),
    )
    .with_width(1),
    // ID_AA64ZFR0_EL1 describes the SVE instructions of streaming mode:
    // SVEver reads 0, since no Apple chip has FEAT_SVE (AARCHMRS: FEAT_SVE2
    // if and only if SVEver is at least 1), and with FEAT_SME the BF16,
    // I8MM and B16B16 fields follow FEAT_BF16, FEAT_I8MM and FEAT_SVE_B16B16.
    feature(
        ID_AA64ZFR0_EL1,
        20,
        1,
        "FEAT_BF16",
        Basis::With("FEAT_SME", SME_RUNS),
    ),
    feature(
        ID_AA64ZFR0_EL1,
        24,
        1,
        "FEAT_SVE_B16B16",
        engine(true, SME_RUNS),
    ),
    feature(
        ID_AA64ZFR0_EL1,
        44,
        1,
        "FEAT_I8MM",
        Basis::With("FEAT_SME", SME_RUNS),
    ),
    // ID_AA64ISAR0_EL1
    feature(
        ID_AA64ISAR0_EL1,
        4,
        1,
        "FEAT_AES",
        Basis::Instructions(&[0x4e28_4820]),
    ),
    feature(
        ID_AA64ISAR0_EL1,
        4,
        2,
        "FEAT_PMULL",
        Basis::Instructions(&[0x0ee2_e020]),
    ),
    feature(
        ID_AA64ISAR0_EL1,
        8,
        1,
        "FEAT_SHA1",
        Basis::Instructions(&[0x5e02_0020]),
    ),
    feature(
        ID_AA64ISAR0_EL1,
        12,
        1,
        "FEAT_SHA256",
        Basis::Instructions(&[0x5e02_4020]),
    ),
    feature(
        ID_AA64ISAR0_EL1,
        12,
        2,
        "FEAT_SHA512",
        Basis::Instructions(&[0xce62_8020]),
    ),
    feature(
        ID_AA64ISAR0_EL1,
        16,
        1,
        "FEAT_CRC32",
        Basis::Instructions(&[0x9ac2_4c20]),
    ),
    feature(
        ID_AA64ISAR0_EL1,
        20,
        2,
        "FEAT_LSE",
        Basis::Instructions(&[0xf820_0041, 0xc8e0_fc41]),
    ),
    feature(
        ID_AA64ISAR0_EL1,
        28,
        1,
        "FEAT_RDM",
        Basis::Instructions(&[0x6e82_8420]),
    ),
    feature(
        ID_AA64ISAR0_EL1,
        32,
        1,
        "FEAT_SHA3",
        Basis::Instructions(&[0xce02_0c20]),
    ),
    feature(
        ID_AA64ISAR0_EL1,
        44,
        1,
        "FEAT_DotProd",
        Basis::Instructions(&[0x4e82_9420]),
    ),
    feature(
        ID_AA64ISAR0_EL1,
        48,
        1,
        "FEAT_FHM",
        Basis::Instructions(&[0x0e22_ec20]),
    ),
    feature(
        ID_AA64ISAR0_EL1,
        52,
        1,
        "FEAT_FlagM",
        Basis::Instructions(&[0xd500_401f]),
    ),
    feature(
        ID_AA64ISAR0_EL1,
        52,
        2,
        "FEAT_FlagM2",
        Basis::Instructions(&[0xd500_405f]),
    ),
    feature(
        ID_AA64ISAR0_EL1,
        56,
        1,
        "FEAT_TLBIOS",
        engine(true, "every TLBI invalidates"),
    ),
    feature(
        ID_AA64ISAR0_EL1,
        56,
        2,
        "FEAT_TLBIRANGE",
        engine(true, "every TLBI invalidates"),
    ),
    // ID_AA64ISAR1_EL1
    feature(
        ID_AA64ISAR1_EL1,
        0,
        1,
        "FEAT_DPB",
        Basis::Instructions(&[0xd50b_7c20]),
    ),
    feature(
        ID_AA64ISAR1_EL1,
        0,
        2,
        "FEAT_DPB2",
        Basis::Instructions(&[0xd50b_7d20]),
    ),
    // QARMA5 in the Classic policy; FPAC (level 4) needs faulting AUT.
    feature(
        ID_AA64ISAR1_EL1,
        4,
        1,
        "FEAT_PAuth",
        Basis::Instructions(&[0xdac1_0020]),
    ),
    feature(
        ID_AA64ISAR1_EL1,
        4,
        4,
        "FEAT_FPAC",
        engine(
            true,
            "PAuth2 with failed AUT* taking EC 0x1C; combined forms corrupt",
        ),
    ),
    feature(
        ID_AA64ISAR1_EL1,
        4,
        5,
        "FEAT_FPACCOMBINE",
        engine(
            true,
            "failed combined authentications (RETA*, BRA*, ERETA*, LDRA*) also take EC 0x1C",
        ),
    ),
    feature(
        ID_AA64ISAR1_EL1,
        4,
        6,
        "FEAT_PAuth_LR",
        engine(true, PAUTH_LR),
    ),
    feature(
        ID_AA64ISAR1_EL1,
        12,
        1,
        "FEAT_JSCVT",
        Basis::Steered(&[0x1e7e_0020], None, JSCVT_STEERING),
    ),
    feature(
        ID_AA64ISAR1_EL1,
        16,
        1,
        "FEAT_FCMA",
        Basis::Instructions(&[0x6e82_c420]),
    ),
    feature(
        ID_AA64ISAR1_EL1,
        20,
        1,
        "FEAT_LRCPC",
        Basis::Instructions(&[0xf8bf_c020]),
    ),
    feature(
        ID_AA64ISAR1_EL1,
        20,
        2,
        "FEAT_LRCPC2",
        Basis::Instructions(&[0xd940_0020]),
    ),
    feature(
        ID_AA64ISAR1_EL1,
        24,
        1,
        "FEAT_PAuth",
        Basis::Instructions(&[0x9ac2_3020]),
    ),
    feature(
        ID_AA64ISAR1_EL1,
        32,
        1,
        "FEAT_FRINTTS",
        Basis::Instructions(&[0x1e28_4020]),
    ),
    feature(
        ID_AA64ISAR1_EL1,
        36,
        1,
        "FEAT_SB",
        Basis::Instructions(&[0xd503_30ff]),
    ),
    feature(
        ID_AA64ISAR1_EL1,
        40,
        1,
        "FEAT_SPECRES",
        engine(true, PREDICTION_RESTRICTION),
    ),
    feature(
        ID_AA64ISAR1_EL1,
        40,
        2,
        "FEAT_SPECRES2",
        engine(true, PREDICTION_RESTRICTION),
    ),
    feature(
        ID_AA64ISAR1_EL1,
        44,
        1,
        "FEAT_BF16",
        Basis::Steered(&[0x6e42_fc20], None, BF16_STEERING),
    ),
    feature(
        ID_AA64ISAR1_EL1,
        52,
        1,
        "FEAT_I8MM",
        Basis::Instructions(&[0x4e82_a420]),
    ),
    feature(
        ID_AA64ISAR1_EL1,
        56,
        1,
        "FEAT_XS",
        engine(
            true,
            "DSB and TLBI nXS complete as DSB and TLBI, and translations report XS",
        ),
    ),
    // ID_AA64ISAR2_EL1
    feature(
        ID_AA64ISAR2_EL1,
        0,
        0b0010,
        "FEAT_WFxT",
        engine(
            true,
            "WFET and WFIT complete once the virtual count reaches Xt",
        ),
    ),
    // bc.ne (FEAT_HBC).
    feature(
        ID_AA64ISAR2_EL1,
        20,
        1,
        "FEAT_HBC",
        Basis::Instructions(&[0x5400_0051]),
    ),
    // abs x0, x1; cnt w0, w1; ctz x0, x1; smax x0, x1, #5; umin w0, w1, w2.
    feature(
        ID_AA64ISAR2_EL1,
        52,
        1,
        "FEAT_CSSC",
        Basis::Instructions(&[
            0xdac0_2020,
            0x5ac0_1c20,
            0xdac0_1820,
            0x91c0_1420,
            0x1ac2_6c20,
        ]),
    ),
    // luti2 and luti4 v0.16b, {v1.16b}, v2[0].
    feature(
        ID_AA64ISAR2_EL1,
        56,
        1,
        "FEAT_LUT",
        Basis::Instructions(&[0x4e82_1020, 0x4e42_2020]),
    ),
    // ID_AA64ISAR3_EL1: addpt, subpt (lsl #3), maddpt and msubpt (FEAT_CPA
    // without CPA2's checks); famax v0.4s and famin v0.8h, v1, v2.
    feature(
        ID_AA64ISAR3_EL1,
        0,
        1,
        "FEAT_CPA",
        Basis::Instructions(&[0x9a02_2020, 0xda02_2c20, 0x9b62_0c20, 0x9b62_8c20]),
    ),
    feature(
        ID_AA64ISAR3_EL1,
        4,
        1,
        "FEAT_FAMINMAX",
        Basis::Instructions(&[0x4ea2_dc20, 0x6ec2_1c20]),
    ),
    // PACM: 0b0010, a full PSTATE.PACM implementation.
    feature(
        ID_AA64ISAR3_EL1,
        12,
        2,
        "FEAT_PAuth_LR",
        engine(true, PAUTH_LR),
    ),
    // ID_AA64PFR2_EL1.FPMR and ID_AA64FPFR0_EL1: the FP8 conversions
    // (F8CVT) in both formats (F8E5M2, F8E4M3); no FP8 arithmetic feature.
    feature(ID_AA64PFR2_EL1, 32, 1, "FEAT_FPMR", engine(true, FPMR)),
    feature(ID_AA64FPFR0_EL1, 0, 1, "FEAT_FP8", engine(true, FP8)).with_width(1),
    feature(ID_AA64FPFR0_EL1, 1, 1, "FEAT_FP8", engine(true, FP8)).with_width(1),
    feature(ID_AA64FPFR0_EL1, 31, 1, "FEAT_FP8", engine(true, FP8)).with_width(1),
    feature(
        ID_AA64MMFR3_EL1,
        4,
        1,
        "FEAT_SCTLR2",
        engine(
            true,
            "SCTLR2_EL1 and SCTLR2_EL2 hold the fields the chip's features give \
             them, EnPACM and EnPACM0 on the M6, under HCRX_EL2.SCTLR2En",
        ),
    ),
    // Spec_FPACC, FEAT_FPACCOMBINE's field, reads 0: Arm's "does not
    // disclose" value. How the speculative use of a pointer that failed its
    // combined authentication changes cached state is the host's
    // speculation past the engine's check, which Stela does not
    // characterise.
    feature(
        ID_AA64MMFR3_EL1,
        60,
        0,
        "FEAT_FPACCOMBINE",
        engine(
            true,
            "the speculative effect of a failed authentication is not disclosed",
        ),
    ),
    // ID_AA64MMFR0_EL1
    fixed(ID_AA64MMFR0_EL1, 0, 0b0101, "48-bit physical addresses"),
    // SPTM 26A434 programs TCR.AS clear (`__text+0xb70c`), refuses an ASID whose
    // low byte is 0 (`__text+0x445f4`) and takes 16384 from the device tree's
    // `pmap-max-asids`; XNU reserves those ASIDs only while this field is 0.
    fixed(ID_AA64MMFR0_EL1, 4, 0, "8-bit ASIDs"),
    fixed(ID_AA64MMFR0_EL1, 20, 1, "16 KiB stage-1 granule"),
    row(
        ID_AA64MMFR0_EL1,
        24,
        0,
        0xf,
        None,
        engine(false, "the walker refuses 64 KiB granules"),
    ),
    row(
        ID_AA64MMFR0_EL1,
        28,
        0,
        0xf,
        None,
        engine(true, "4 KiB stage-1 granule"),
    ),
    // TGran16_2, TGran64_2 and TGran4_2: 0b0001 (not supported) without
    // EL2; with it, machine::mmu's stage 2 takes 16 KiB and 4 KiB granules
    // (0b0010) and refuses 64 KiB.
    fixed(ID_AA64MMFR0_EL1, 32, 1, "no stage 2 without EL2"),
    row(ID_AA64MMFR0_EL1, 32, 2, 1, None, Basis::Host(STAGE2_HOST)),
    fixed(
        ID_AA64MMFR0_EL1,
        36,
        1,
        "the walker refuses 64 KiB stage-2 granules",
    ),
    fixed(ID_AA64MMFR0_EL1, 40, 1, "no stage 2 without EL2"),
    row(ID_AA64MMFR0_EL1, 40, 2, 1, None, Basis::Host(STAGE2_HOST)),
    feature(
        ID_AA64MMFR0_EL1,
        56,
        1,
        "FEAT_FGT",
        engine(
            true,
            "the generated accessors apply HFG*TR, HDFG*TR and HAFGRTR; \
             HFGITR_EL2 also traps ERET and SVC",
        ),
    ),
    feature(
        ID_AA64MMFR0_EL1,
        56,
        2,
        "FEAT_FGT2",
        engine(
            true,
            "the generated accessors also apply HFG*TR2 and HDFG*TR2, whose one \
             control on the M6 is HFGRTR2_EL2.nERXGSR_EL1",
        ),
    ),
    feature(
        ID_AA64MMFR0_EL1,
        60,
        1,
        "FEAT_ECV",
        engine(
            true,
            "CNTPCTSS_EL0 and CNTVCTSS_EL0, EVNTIS on both event streams and \
             the EL1TVT and EL1TVCT traps; CNTPOFF_EL2 is FEAT_ECV_POFF, \
             outside the profile",
        ),
    ),
    // ID_AA64MMFR1_EL1
    feature(ID_AA64MMFR1_EL1, 8, 1, "FEAT_VHE", Basis::Host(VHE_HOST)),
    feature(
        ID_AA64MMFR1_EL1,
        40,
        1,
        "FEAT_HCX",
        engine(
            true,
            "HCRX_EL2 with its traps; its fields belong to features the chip has",
        ),
    ),
    // FEAT_HPDS (Armv8.1), FEAT_XNX, FEAT_TTCNP (Armv8.2), FEAT_S2FWB,
    // FEAT_TTL (Armv8.4) and FEAT_E0PD (Armv8.5) are mandatory in the
    // Armv8.6 profile, without an Apple clang feature name.
    row(
        ID_AA64MMFR1_EL1,
        12,
        1,
        0,
        None,
        engine(
            true,
            "TCR_ELx.HPD0 and HPD1 disable hierarchical permissions",
        ),
    ),
    row(
        ID_AA64MMFR1_EL1,
        28,
        1,
        0,
        None,
        Basis::Host("stage-2 XN[1:0] selects EL0 and EL1 execution"),
    ),
    feature(
        ID_AA64MMFR1_EL1,
        16,
        1,
        "FEAT_LOR",
        // No LORegions (LORID_EL1 0): LDLAR and STLLR order like LDAR and
        // STLR, which is always permitted.
        Basis::Instructions(&[0xc8df_7c20, 0xc89f_7c20]),
    ),
    feature(
        ID_AA64MMFR1_EL1,
        20,
        1,
        "FEAT_PAN",
        engine(true, "PSTATE.PAN and SPAN"),
    ),
    feature(
        ID_AA64MMFR1_EL1,
        20,
        2,
        "FEAT_PAN2",
        engine(true, "AT S1E1RP and S1E1WP"),
    ),
    feature(
        ID_AA64MMFR1_EL1,
        44,
        1,
        "FEAT_AFP",
        engine(
            true,
            "FPCR holds FIZ, AH and NEP, and the compiled FP operations follow them",
        ),
    ),
    feature(
        ID_AA64MMFR1_EL1,
        20,
        3,
        "FEAT_PAN3",
        engine(true, "SCTLR_ELx.EPAN extends PAN to EL0-executable memory"),
    ),
    // ID_AA64MMFR2_EL1
    row(
        ID_AA64MMFR2_EL1,
        0,
        1,
        0,
        None,
        engine(
            true,
            "CnP is a hint for TLBs that PEs share, and one PE shares none",
        ),
    ),
    row(
        ID_AA64MMFR2_EL1,
        40,
        1,
        0,
        None,
        Basis::Host("HCR_EL2.FWB lets stage 2 override stage-1 attributes"),
    ),
    row(
        ID_AA64MMFR2_EL1,
        48,
        1,
        0,
        None,
        engine(true, "TTL hints narrow the TLBI scope (machine::mmu::tlbi)"),
    ),
    row(
        ID_AA64MMFR2_EL1,
        60,
        1,
        0,
        None,
        engine(
            true,
            "TCR_ELx.E0PD0 and E0PD1 fault EL0 accesses at level 0",
        ),
    ),
    // FEAT_IDST (Armv8.4) and FEAT_EVT (Armv8.5), with the profile's
    // FEAT_EVT2, are mandatory in the Armv8.6 profile, without an Apple clang
    // feature name.
    row(
        ID_AA64MMFR2_EL1,
        36,
        1,
        0,
        None,
        engine(
            true,
            "EL0 reads of the ID space and accesses to unimplemented ID \
             registers trap with EC 0x18",
        ),
    ),
    row(
        ID_AA64MMFR2_EL1,
        56,
        1,
        0,
        None,
        Basis::Host("HCR_EL2.TID4, TICAB and TOCU trap through the generated accessors"),
    ),
    row(
        ID_AA64MMFR2_EL1,
        56,
        2,
        0,
        None,
        Basis::Host("HCR_EL2.TTLBIS and TTLBOS trap the Inner and Outer Shareable TLBIs"),
    ),
    feature(
        ID_AA64MMFR2_EL1,
        4,
        1,
        "FEAT_UAO",
        engine(
            true,
            "the engine performs LDTR/STTR with EL0 permissions unless \
             PSTATE.UAO makes them privileged",
        ),
    ),
    feature(ID_AA64MMFR2_EL1, 24, 1, "FEAT_NV", engine(true, NESTING)),
    feature(ID_AA64MMFR2_EL1, 24, 2, "FEAT_NV2", engine(true, NESTING)),
    feature(
        ID_AA64MMFR2_EL1,
        32,
        1,
        "FEAT_LSE2",
        Basis::Atomic16(
            "machine::mmu::alignment applies its alignment rules; a \
             16-byte-aligned LDP, STP, LDNP or STNP of X registers is one \
             MOVDQA, single-copy atomic on Intel and AMD hosts with AVX (SDM \
             Vol. 3A 9.1.1, APM Vol. 2 7.3.2); ordered and atomic accesses \
             keep their granule. On AMD a plain unaligned access inside its \
             16-byte granule that crosses an 8-byte boundary is two host \
             accesses, which concurrent PEs can observe torn",
        ),
    ),
    // ID_AA64DFR0_EL1: the Armv8.4 debug architecture (FEAT_Debugv8p4,
    // implied by Armv8.6), and the breakpoint and watchpoint counts of
    // machine::debug, which are Stela's choice.
    fixed(ID_AA64DFR0_EL1, 0, 0b1001, "self-hosted debug of Armv8.4"),
    fixed(
        ID_AA64DFR0_EL1,
        DFR0_BRPS,
        super::debug::BREAKPOINTS - 1,
        "breakpoints (machine::debug::BREAKPOINTS)",
    ),
    fixed(
        ID_AA64DFR0_EL1,
        DFR0_WRPS,
        super::debug::WATCHPOINTS - 1,
        "watchpoints (machine::debug::WATCHPOINTS)",
    ),
    fixed(
        ID_AA64DFR0_EL1,
        28,
        super::debug::CONTEXT_AWARE - 1,
        "context-aware breakpoints (machine::debug::CONTEXT_AWARE)",
    ),
    feature(
        ID_AA64DFR0_EL1,
        8,
        1,
        "FEAT_PMUv3",
        engine(false, NO_MONITORS),
    ),
    fixed(ID_AA64DFR0_EL1, 36, 0xf, "no OS Double Lock"),
    feature(
        ID_AA64DFR0_EL1,
        40,
        1,
        "FEAT_TRF",
        engine(false, NO_MONITORS),
    ),
    // CTR_EL0: bit 31 is RES1; 64-byte lines like the DC ZVA block; PIPT
    // instruction cache; instruction fetch needs IC (DIC 0) but no data
    // clean (IDC 1), because translation reads guest RAM directly.
    fixed(CTR_EL0, 0, 4, "64-byte instruction cache lines"),
    fixed(CTR_EL0, 14, 0b11, "PIPT instruction cache").with_width(2),
    fixed(CTR_EL0, 16, 4, "64-byte data cache lines"),
    fixed(CTR_EL0, 20, 4, "64-byte exclusives reservation granule"),
    fixed(CTR_EL0, 24, 4, "64-byte writeback granule"),
    fixed(
        CTR_EL0,
        28,
        1,
        "IDC: no data clean for instruction coherence",
    )
    .with_width(1),
    fixed(CTR_EL0, 31, 1, "RES1").with_width(1),
    // CTR_EL0.TminLine: the tag cache line is 64 bytes, as the data's.
    feature(CTR_EL0, 32, 4, "FEAT_MTE2", engine(true, MTE2)).with_width(6),
    feature(GMID_EL1, 0, 4, "FEAT_MTE2", engine(true, MTE2)),
    // ID_AA64MMFR4_EL1.E2H0 = 0b1110: HCR_EL2.E2H is RES1 (no FEAT_E2H0) and
    // HCR_EL2.NV1 is RES0 (RAZ where FEAT_NV is, an IMPLEMENTATION DEFINED
    // choice). The M2 itself predates the field.
    fixed(
        ID_AA64MMFR4_EL1,
        24,
        0b1110,
        "E2H is RES1 and there is no FEAT_NV1",
    ),
    // DBGAUTHSTATUS_EL1: Non-secure invasive and non-invasive debug are
    // implemented and externally disabled (no external debugger); no Secure,
    // Realm or Root state.
    fixed(
        DBGAUTHSTATUS_EL1,
        0,
        0b10,
        "external invasive debug disabled",
    )
    .with_width(2),
    fixed(
        DBGAUTHSTATUS_EL1,
        2,
        0b10,
        "external non-invasive debug disabled",
    )
    .with_width(2),
    // AIDR_EL1: Apple feature bits, named by m1n1 `cpu_regs.h`. Apple
    // features have no clang `FEAT_` tag; each bit is an engine property.
    // Bits m1n1 does not name read 0.
    fixed(
        AIDR_EL1,
        4,
        1,
        "AMX: machine::amx executes the AMX instructions",
    )
    .with_width(1),
    fixed(
        AIDR_EL1,
        16,
        1,
        "GXF: GENTER, GEXIT and the guarded-level banks",
    )
    .with_width(1),
    row(
        AIDR_EL1,
        0,
        1,
        0,
        None,
        engine(false, "MUL53 instructions are not modeled"),
    )
    .with_width(1),
    // WKdm: WKDMC and WKDMD (`stela_compiler::wkdm`), which every chip's
    // 26A434 kernel executes unless booted with `-vm_compressor_wk_sw`; it
    // does not read this bit.
    row(
        AIDR_EL1,
        1,
        1,
        0,
        None,
        Basis::Instructions(&[
            stela_compiler::wkdm::COMPRESS_X0_X1,
            stela_compiler::wkdm::DECOMPRESS_X0_X1,
        ]),
    )
    .with_width(1),
    row(
        AIDR_EL1,
        2,
        1,
        0,
        None,
        engine(false, "architectural state retention is not modeled"),
    )
    .with_width(1),
    row(
        AIDR_EL1,
        9,
        1,
        0,
        None,
        engine(false, "the ACTLR TSO mode switch is not modeled"),
    )
    .with_width(1),
    row(
        AIDR_EL1,
        19,
        1,
        0,
        None,
        engine(false, "APFLG is not modeled"),
    )
    .with_width(1),
    row(
        AIDR_EL1,
        20,
        1,
        0,
        None,
        engine(false, "PSRV is not modeled"),
    )
    .with_width(1),
    // Unnamed by m1n1: the 26A434 kernel enables the static branch that
    // adds ACTLR EnAFP and EnPRSV to a translated thread's EL0 return from
    // it (0xfffffe000c551258).
    row(
        AIDR_EL1,
        23,
        1,
        0,
        None,
        engine(
            false,
            "the alternate floating-point mode it selects is not modeled",
        ),
    )
    .with_width(1),
    // XNU's `AIDR_AGT` (xnu-12377.121.6 `tests/apple_generic_timer.c`):
    // CNTFRQ_EL0 reads 1 GHz, which m1n1 reads beside the 24 MHz Apple
    // timer on an M4 Pro (`src/hv/hv_sprr.c`).
    row(
        AIDR_EL1,
        32,
        1,
        0,
        None,
        Basis::GenericTimer("the Arm bank counts 1 GHz beside the Apple generic timer"),
    )
    .with_width(1),
];

/// The chip has `feature`: in its Apple clang feature set, or implied by it.
/// A feature only some chips' profiles have is the profile's: the M1 to M3
/// exclude the FEAT_AMUv1 their clang sets list.
fn has(chip: Chip, feature: &str) -> bool {
    match Feature::ALL.iter().find(|known| known.name() == feature) {
        Some(known) => known.on(chip),
        None => chip.cap().contains(&feature),
    }
}

fn compiles(samples: &[u32], target: TargetCpu) -> bool {
    samples.iter().all(|word| {
        CompilationPlan::new(
            &word.to_le_bytes(),
            GuestAddress(0x1000),
            &stela_compiler::Options::new(BlockProfile::new(Features::SYSTEM_MODE), target),
        )
        .and_then(CompilationPlan::compile)
        .is_ok()
    })
}

/// Whether a [`Basis::Steered`] row advertises its feature on `target`.
fn steered_holds(samples: &[u32], native: Option<TargetCpu>, target: TargetCpu) -> bool {
    native.is_some_and(|tier| target.supports(tier)) && compiles(samples, target)
}

/// Whether `feature` is steered ([`Basis::Steered`]): its instructions stay
/// admitted whatever its fields read.
fn steered(feature: &str) -> bool {
    TABLE
        .iter()
        .any(|row| row.feature == Some(feature) && matches!(row.basis, Basis::Steered(..)))
}

/// Whether a steered row of `feature` does not advertise it on `target`, so
/// that every field of the feature reads absent: a field describing the
/// feature's instructions within another feature (ID_AA64ZFR0_EL1.BF16 with
/// FEAT_SME) never claims what the main field hides.
fn steered_off(feature: &str, target: TargetCpu) -> bool {
    TABLE.iter().any(|row| {
        row.feature == Some(feature)
            && matches!(row.basis, Basis::Steered(samples, native, _)
                if !steered_holds(samples, native, target))
    })
}

/// The steered features `chip` has and whether `target` advertises them,
/// once per feature, in [`TABLE`] order: what the boot reports.
pub fn steering(chip: Chip, target: TargetCpu) -> Vec<Steering> {
    let mut steering: Vec<Steering> = Vec::new();
    for row in TABLE {
        let (Some(feature), Basis::Steered(_, _, reason)) = (row.feature, row.basis) else {
            continue;
        };
        if has(chip, feature) && steering.iter().all(|known| known.feature != feature) {
            steering.push(Steering {
                feature,
                advertised: !steered_off(feature, target),
                reason,
            });
        }
    }
    steering
}

fn holds(row: &Row, chip: Chip, target: TargetCpu, el2: bool, speculation: Speculation) -> bool {
    if row.feature.is_some_and(|feature| !has(chip, feature)) {
        return false;
    }
    if row
        .feature
        .is_some_and(|feature| steered_off(feature, target))
    {
        return false;
    }
    match row.basis {
        Basis::Steered(samples, native, _) => steered_holds(samples, native, target),
        Basis::Engine(holds, _) => holds,
        Basis::With(feature, _) => has(chip, feature),
        Basis::Host(_) => el2,
        Basis::Timing(_) => target.data_independent_timing_target().is_some(),
        Basis::Contexts(_) => speculation.contexts,
        Basis::FaultingData(_) => speculation.faulting_data,
        Basis::GenericTimer(_) => super::timer::agt::present(chip),
        Basis::Atomic16(_) => target.supports(TargetCpu::AtomicLoad16),
        Basis::Instructions(samples) => compiles(samples, target),
    }
}

/// The level [`derive`] presents in `register`'s field at `shift` on `chip`,
/// for a field whose rows are all engine properties: the same on every host
/// target and kernel level.
fn engine_field(chip: Chip, register: Encoding, shift: u8) -> u64 {
    let rows = || {
        TABLE
            .iter()
            .filter(move |row| row.register == register && row.shift == shift)
    };
    debug_assert!(rows().all(|row| matches!(row.basis, Basis::Engine(..))));
    rows()
        .filter(|row| row.feature.is_none_or(|feature| has(chip, feature)))
        .filter(|row| matches!(row.basis, Basis::Engine(true, _)))
        .fold(rows().next().map_or(0, |row| row.absent), |level, row| {
            level.max(row.level)
        })
}

/// The chip-varying features `chip` has that the identification `fixed`
/// withholds, one bit per [`Feature`] (`1 << feature as u64`): a feature
/// is withheld when one of the ID fields its rows describe reads below the
/// feature's lowest level there, among the rows whose containing feature
/// the PE presents. A streaming-only field does not constrain the ordinary
/// AdvSIMD feature when streaming is absent. A profile can withhold a feature
/// by masking its field (a validation machine's memory tagging), or because
/// the field's basis fails on this host. The PE treats it as absent, as a chip
/// without it. A register `fixed` has no value for withholds nothing. A
/// steered feature ([`Basis::Steered`]) is never withheld.
pub fn withheld(chip: Chip, fixed: &[(Encoding, u64)]) -> u64 {
    let mut withheld = 0;
    for feature in Feature::ALL.into_iter().filter(|feature| feature.on(chip)) {
        if !presents_feature(chip, feature.name(), fixed) {
            withheld |= 1 << feature as u64;
        }
    }
    withheld
}

/// Conditional rows describe an instruction feature inside another feature,
/// rather than making that containing feature a requirement for every use.
/// TABLE's containing-feature dependencies are acyclic (checked below).
fn presents_feature(chip: Chip, feature: &str, fixed: &[(Encoding, u64)]) -> bool {
    has(chip, feature)
        && (steered(feature)
            || TABLE
                .iter()
                .enumerate()
                .filter(|(_, row)| row.feature == Some(feature))
                .all(|(index, row)| {
                    if let Basis::With(containing, _) = row.basis
                        && !presents_feature(chip, containing, fixed)
                    {
                        return true;
                    }
                    presents(index, row, fixed)
                }))
}

/// Whether `fixed` presents `row` (at `index` in [`TABLE`]): its field
/// reads the level of this row or of a later row of the same field, or of
/// an earlier row of the same feature there (its lowest level).
fn presents(index: usize, row: &Row, fixed: &[(Encoding, u64)]) -> bool {
    let Some(value) = fixed
        .iter()
        .find(|(register, _)| *register == row.register)
        .map(|(_, value)| value >> row.shift & row.mask())
    else {
        return true;
    };
    let field = |other: &&Row| other.register == row.register && other.shift == row.shift;
    let first = TABLE
        .iter()
        .position(|other| field(&other) && other.feature == row.feature)
        .unwrap_or(index);
    TABLE[first..]
        .iter()
        .filter(field)
        .any(|other| other.level == value)
}

/// Whether `chip`'s profile implements memory tagging: ID_AA64PFR1_EL1.MTE
/// is nonzero.
pub fn memory_tagging(chip: Chip) -> bool {
    engine_field(chip, ID_AA64PFR1_EL1, PFR1_MTE) != 0
}

/// Whether `chip`'s profile implements the Scalable Matrix Extension:
/// ID_AA64PFR1_EL1.SME is nonzero.
pub fn streaming(chip: Chip) -> bool {
    engine_field(chip, ID_AA64PFR1_EL1, PFR1_SME) != 0
}

/// The breakpoint and watchpoint counts `chip`'s profile presents:
/// ID_AA64DFR0_EL1.BRPs and WRPs plus one.
pub fn breakpoints_and_watchpoints(chip: Chip) -> (u64, u64) {
    (
        engine_field(chip, ID_AA64DFR0_EL1, DFR0_BRPS) + 1,
        engine_field(chip, ID_AA64DFR0_EL1, DFR0_WRPS) + 1,
    )
}

/// Every register of [`REGISTERS`] with its value for `chip` on this host
/// target, with or without EL2, with the host's `speculation` guarantees.
pub fn derive(
    chip: Chip,
    target: TargetCpu,
    el2: bool,
    speculation: Speculation,
) -> Vec<(Encoding, u64)> {
    REGISTERS
        .iter()
        .map(|&register| {
            let mut value = 0u64;
            let mut fields: Vec<(u8, u64)> = Vec::new();
            for row in TABLE.iter().filter(|row| row.register == register) {
                let current = match fields.iter_mut().find(|(shift, _)| *shift == row.shift) {
                    Some(field) => field,
                    None => {
                        fields.push((row.shift, row.absent));
                        fields.last_mut().expect("just pushed")
                    }
                };
                if holds(row, chip, target, el2, speculation) {
                    current.1 = row.level;
                }
            }
            for (shift, field) in fields {
                value |= field << shift;
            }
            (register, value)
        })
        .collect()
}

#[cfg(test)]
mod tests;
