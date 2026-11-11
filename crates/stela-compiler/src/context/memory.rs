//! Native memory-service ABI used by scalable and exclusive accesses. Each
//! request describes one architectural element; data points to private staging.

use core::ffi::c_void;

mod operation;
pub use operation::{MemoryOperationClass, MemoryOrder};
pub mod address;
pub mod atomic;
pub mod exclusive;
pub mod instructions;
pub mod ordered;
pub use instructions::InstructionRequirements;

pub const MEMORY_CONTEXT_VERSION: u32 = 2;
pub const MEMORY_CONTEXT_BYTES: u32 = 64;
pub const MEMORY_CALLBACK_OFFSET: u32 = 8;
pub const MEMORY_REQUEST_BYTES: usize = 32;
pub const MEMORY_REQUEST_ADDRESS_OFFSET: u32 = 0;
pub const MEMORY_REQUEST_DATA_OFFSET: u32 = 8;
pub const MEMORY_REQUEST_LENGTH_OFFSET: u32 = 16;
pub const MEMORY_REQUEST_FLAGS_OFFSET: u32 = 20;
pub const MEMORY_REQUEST_PC_OFFSET: u32 = 24;

/// Access descriptor bits keep architectural intent across the native ABI.
pub mod flags {
    pub const WRITE: u32 = 1 << 0;
    pub const TAG_CHECKED: u32 = 1 << 1;
    pub const NON_TEMPORAL: u32 = 1 << 2;
    pub const SME: u32 = 1 << 3;
    /// CreateAccDescSVE with predicated=true; SME accesses never acquire this
    /// descriptor's special two-doubleword splitting rule.
    pub const SVE_PREDICATED: u32 = 1 << 4;
    pub const ASIMD: u32 = 1 << 5;
    /// Ordinary Mem accessor pair metadata. Atomic/exclusive pairs use their
    /// dedicated category and must never enter ordinary decomposition.
    pub const PAIR: u32 = 1 << 6;
    /// Reserved categories; ordinary dispatch rejects them.
    pub const ATOMIC: u32 = 1 << 7;
    pub const ORDERED: u32 = 1 << 8;
    pub const EXCLUSIVE: u32 = 1 << 9;
    pub const ORDINARY_KNOWN: u32 =
        WRITE | TAG_CHECKED | NON_TEMPORAL | SME | SVE_PREDICATED | ASIMD | PAIR;
    pub const KNOWN: u32 = ORDINARY_KNOWN | ATOMIC | ORDERED | EXCLUSIVE;
}

/// Requirements derived from and bound to one consumed native body. `None`
/// at a containing manifest means absent metadata, not these empty requirements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MemoryRequirements {
    /// Missing old artifact metadata deserializes to zero and must be refused.
    #[serde(default)]
    pub context_version: u32,
    #[serde(default, skip_serializing_if = "InstructionRequirements::is_empty")]
    pub required_instructions: InstructionRequirements,
    pub required_flags: u32,
    pub required_host_caps: u32,
    pub required_profile: u32,
    pub required_sp_alignment: SpAlignmentRequirement,
    /// Original source AddressAdd and constrained AddressIncrement effects.
    #[serde(
        default,
        skip_serializing_if = "address::AddressCheckRequirements::is_empty"
    )]
    pub required_address_checks: address::AddressCheckRequirements,
    /// FEAT_CPA arithmetic at the current EL.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub required_pointer_arithmetic: Option<crate::context::pointer::PointerArithmeticRequirement>,
}

impl Default for MemoryRequirements {
    fn default() -> Self {
        Self {
            context_version: MEMORY_CONTEXT_VERSION,
            required_instructions: InstructionRequirements::NONE,
            required_flags: 0,
            required_host_caps: 0,
            required_profile: 0,
            required_sp_alignment: SpAlignmentRequirement::Any,
            required_address_checks: Default::default(),
            required_pointer_arithmetic: None,
        }
    }
}

/// Original-SP checking specialized into one consumed native body. This is
/// a requirement on the effective CPU policy, never a copy of CPU control state.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SpAlignmentRequirement {
    #[default]
    Any,
    Enabled,
    Disabled,
}

/// The implementation choices the scalable memory code is built for: no
/// tag-check suppression for streaming SIMD/FP or SME LDR/STR, and
/// CHECKSPNONEACTIVE true. Implementations that suppress tag checks are not
/// supported.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryImplementationProfile {
    ScalableTagChecks,
    StreamingSveTagSuppression,
    SmeTagSuppression,
}

pub const MEMORY_IMPLEMENTATION_PROFILE_ID: &str =
    "scalable-memory-tags-checked-checkspnoneactive-v3";

impl MemoryRequirements {
    /// Combine requirements for one consumed body. Opposite SP specializations
    /// cannot share a single effective-control policy and must fail compilation.
    pub fn merge(self, other: Self) -> Option<Self> {
        if self.context_version != MEMORY_CONTEXT_VERSION
            || other.context_version != MEMORY_CONTEXT_VERSION
            || !self.required_instructions.valid()
            || !other.required_instructions.valid()
        {
            return None;
        }
        let required_sp_alignment = match (self.required_sp_alignment, other.required_sp_alignment)
        {
            (SpAlignmentRequirement::Any, requirement)
            | (requirement, SpAlignmentRequirement::Any) => requirement,
            (first, second) if first == second => first,
            _ => return None,
        };
        let required_pointer_arithmetic = match (
            self.required_pointer_arithmetic,
            other.required_pointer_arithmetic,
        ) {
            (None, required) | (required, None) => required,
            (Some(first), Some(second)) if first == second => Some(first),
            _ => return None,
        };
        let required_address_checks = self
            .required_address_checks
            .merge(other.required_address_checks)?;
        if let Some(required) = required_pointer_arithmetic
            && !required_address_checks.compatible_arithmetic(required)
        {
            return None;
        }
        Some(Self {
            context_version: MEMORY_CONTEXT_VERSION,
            required_instructions: self
                .required_instructions
                .union(other.required_instructions),
            required_flags: self.required_flags | other.required_flags,
            required_host_caps: self.required_host_caps | other.required_host_caps,
            required_profile: self.required_profile | other.required_profile,
            required_sp_alignment,
            required_address_checks,
            required_pointer_arithmetic,
        })
    }
}

pub mod host_caps {
    pub const ATOMIC_LOAD16: u32 = 1 << 0;
    pub const CX16: u32 = 1 << 1;
    pub const KNOWN: u32 = ATOMIC_LOAD16 | CX16;
}

pub mod profile_requirements {
    /// Native Normal-WB/SCA16 data access with AlignmentEnforced=false.
    /// CPA arithmetic/address specialization is separately typed and validated.
    /// Original-SP checking is separately specialized and validated.
    pub const NATIVE_ACCESS: u32 = 1 << 0;
    /// Physical allocation-tag granules keep guest/physical alignment.
    pub const TAG_GRANULES: u32 = 1 << 1;
    /// Require the fixed named implementation identity above. This is a body
    /// requirement, never a raw access descriptor or a copy of CPU controls.
    pub const SCALABLE_TAG_CHECKS: u32 = 1 << 2;
    /// Canonical native-system RAM permits an unchanged-value CMPXCHG16B
    /// read under the physical all-writer gate. The whole aligned host
    /// footprint must be RW; this grants no guest write permission.
    pub const SYSTEM_ATOMIC_READ_WRITEBACK: u32 = 1 << 8;
    pub const KNOWN: u32 =
        NATIVE_ACCESS | TAG_GRANULES | SCALABLE_TAG_CHECKS | SYSTEM_ATOMIC_READ_WRITEBACK;
}

/// Memory-service outcomes; a load commits only on Success.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryStatus {
    Success = 0,
    Translation = 1,
    Permission = 2,
    Alignment = 3,
    TagCheck = 4,
    Unsupported = 5,
    InvalidRequest = 6,
    Domain = 7,
    /// Original SP alignment fault, distinct from data-access alignment.
    StackAlignment = 8,
    /// A registered physical RAM frame is temporarily unavailable.
    Nonresident = 9,
}

impl MemoryStatus {
    pub fn from_code(code: u32) -> Option<Self> {
        [
            Self::Success,
            Self::Translation,
            Self::Permission,
            Self::Alignment,
            Self::TagCheck,
            Self::Unsupported,
            Self::InvalidRequest,
            Self::Domain,
            Self::StackAlignment,
            Self::Nonresident,
        ]
        .into_iter()
        .find(|status| *status as u32 == code)
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::Translation => "translation",
            Self::Permission => "permission",
            Self::Alignment => "alignment",
            Self::TagCheck => "tag check",
            Self::Unsupported => "unsupported",
            Self::InvalidRequest => "invalid request",
            Self::Domain => "domain",
            Self::StackAlignment => "stack alignment",
            Self::Nonresident => "nonresident RAM",
        }
    }
}

/// Buffer bytes are little endian. Reads may change staging before a later part
/// fails; callers must commit destinations only after every read succeeds.
#[repr(C)]
#[derive(Debug)]
pub struct MemoryRequest {
    pub address: u64,
    pub data: *mut u8,
    pub bytes: u32,
    pub flags: u32,
    pub pc: u64,
}

impl MemoryRequest {
    /// The ordinary Mem category only. Reserved atomic/ordered categories have
    /// separate exact operations and must never acquire mechanical splitting.
    pub const fn has_supported_ordinary_shape(&self) -> bool {
        if self.flags & !flags::ORDINARY_KNOWN != 0 {
            return false;
        }
        let asimd = self.flags & flags::ASIMD != 0;
        let pair = self.flags & flags::PAIR != 0;
        if (asimd || pair) && self.flags & (flags::SME | flags::SVE_PREDICATED) != 0 {
            return false;
        }
        if pair {
            matches!(self.bytes, 8 | 16) || self.bytes == 32 && asimd
        } else {
            matches!(self.bytes, 1 | 2 | 4 | 8 | 16)
        }
    }
}

#[cfg(target_arch = "x86_64")]
pub type MemoryCallback = unsafe extern "sysv64" fn(
    context: *mut MemoryContextView,
    request: *const MemoryRequest,
) -> u32;
/// Only x86-64 runs translated code; ARM hosts compile the type alone.
#[cfg(not(target_arch = "x86_64"))]
pub type MemoryCallback =
    unsafe extern "C" fn(context: *mut MemoryContextView, request: *const MemoryRequest) -> u32;

/// Borrowed dispatch view. Its owner keeps resolver, tag and monitor state.
/// Callback failures preserve a diagnostic record after native stack unwind.
#[repr(C)]
#[derive(Debug)]
pub struct MemoryContextView {
    pub version: u32,
    pub bytes: u32,
    pub callback: MemoryCallback,
    pub opaque: *mut c_void,
    pub fault_kind: u32,
    /// Dynamic fault direction, introduced by memory-context v2. OriginalSource
    /// keeps the source instruction's ordinary read/write classification.
    pub fault_access: u32,
    pub fault_address: u64,
    pub fault_pc: u64,
    pub completed_bytes: u64,
    pub reserved_tail: u64,
}

impl MemoryContextView {
    pub const fn has_supported_header(&self) -> bool {
        self.version == MEMORY_CONTEXT_VERSION
            && self.bytes >= MEMORY_CONTEXT_BYTES
            && matches!(self.fault_access, 0..=2)
            && (self.fault_kind != MemoryStatus::Success as u32 || self.fault_access == 0)
            && self.reserved_tail == 0
    }

    pub fn record(&mut self, status: MemoryStatus, address: u64, pc: u64, completed: u64) {
        self.fault_access = FaultAccess::OriginalSource as u32;
        self.fault_kind = status as u32;
        self.fault_address = address;
        self.fault_pc = pc;
        self.completed_bytes = completed;
    }
}

/// The reached side of a read-modify-write fault. Access intent does not change
/// DFSC/status, address, completed prefix or source PC.
#[repr(u32)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum FaultAccess {
    #[default]
    OriginalSource = 0,
    Read = 1,
    Write = 2,
}
impl FaultAccess {
    pub const fn from_code(code: u32) -> Option<Self> {
        match code {
            0 => Some(Self::OriginalSource),
            1 => Some(Self::Read),
            2 => Some(Self::Write),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::offset_of;

    #[test]
    fn named_stack_fault_preserves_old_codes_and_merge_rejects_opposite_sp_profiles() {
        for (code, status) in [
            MemoryStatus::Success,
            MemoryStatus::Translation,
            MemoryStatus::Permission,
            MemoryStatus::Alignment,
            MemoryStatus::TagCheck,
            MemoryStatus::Unsupported,
            MemoryStatus::InvalidRequest,
            MemoryStatus::Domain,
            MemoryStatus::StackAlignment,
            MemoryStatus::Nonresident,
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(status as u32, code as u32);
            assert_eq!(MemoryStatus::from_code(code as u32), Some(status));
        }
        assert_eq!(MemoryStatus::from_code(10), None);
        for first in [
            SpAlignmentRequirement::Any,
            SpAlignmentRequirement::Enabled,
            SpAlignmentRequirement::Disabled,
        ] {
            for second in [
                SpAlignmentRequirement::Any,
                SpAlignmentRequirement::Enabled,
                SpAlignmentRequirement::Disabled,
            ] {
                let a = MemoryRequirements {
                    required_flags: flags::WRITE,
                    required_sp_alignment: first,
                    ..Default::default()
                };
                let b = MemoryRequirements {
                    required_host_caps: host_caps::CX16,
                    required_profile: profile_requirements::NATIVE_ACCESS,
                    required_sp_alignment: second,
                    ..Default::default()
                };
                let merged = a.merge(b);
                let conflict = first != SpAlignmentRequirement::Any
                    && second != SpAlignmentRequirement::Any
                    && first != second;
                assert_eq!(merged.is_none(), conflict);
                if let Some(merged) = merged {
                    assert_eq!(merged.required_flags, flags::WRITE);
                    assert_eq!(merged.required_host_caps, host_caps::CX16);
                    assert_eq!(merged.required_profile, profile_requirements::NATIVE_ACCESS);
                    assert_eq!(
                        merged.required_sp_alignment,
                        if first == SpAlignmentRequirement::Any {
                            second
                        } else {
                            first
                        }
                    );
                }
            }
        }
    }

    #[test]
    fn ordinary_shape_matrix_never_admits_atomic_ordered_or_invalid_pair_widths() {
        let scalar: &[u32] = &[1, 2, 4, 8, 16];
        let pair: &[u32] = &[8, 16];
        let asimd_pair: &[u32] = &[8, 16, 32];
        let cases: &[(u32, &[u32])] = &[
            (0, scalar),
            (flags::SME, scalar),
            (flags::SVE_PREDICATED, scalar),
            (flags::SME | flags::SVE_PREDICATED, scalar),
            (flags::ASIMD, scalar),
            (flags::PAIR, pair),
            (flags::ASIMD | flags::PAIR, asimd_pair),
            (flags::PAIR | flags::SME, &[]),
            (flags::PAIR | flags::SVE_PREDICATED, &[]),
            (flags::ASIMD | flags::SME, &[]),
            (flags::ASIMD | flags::SVE_PREDICATED, &[]),
            (flags::ATOMIC, &[]),
            (flags::ORDERED, &[]),
            (flags::PAIR | flags::ATOMIC, &[]),
            (flags::ASIMD | flags::PAIR | flags::ORDERED, &[]),
            (1 << 31, &[]),
        ];
        for (descriptor, widths) in cases {
            for qualifiers in 0..8 {
                let flags = descriptor
                    | if qualifiers & 1 != 0 { flags::WRITE } else { 0 }
                    | if qualifiers & 2 != 0 {
                        flags::TAG_CHECKED
                    } else {
                        0
                    }
                    | if qualifiers & 4 != 0 {
                        flags::NON_TEMPORAL
                    } else {
                        0
                    };
                for bytes in [0, 1, 2, 4, 8, 16, 32, 64] {
                    let request = MemoryRequest {
                        address: 0,
                        data: core::ptr::null_mut(),
                        bytes,
                        flags,
                        pc: 0,
                    };
                    assert_eq!(
                        request.has_supported_ordinary_shape(),
                        widths.contains(&bytes),
                        "flags={flags:#x} bytes={bytes}"
                    );
                }
            }
        }
        assert_eq!(flags::PAIR, 1 << 6);
        assert_eq!(flags::ATOMIC, 1 << 7);
        assert_eq!(flags::ORDERED, 1 << 8);
        assert_eq!(flags::ORDINARY_KNOWN & (flags::ATOMIC | flags::ORDERED), 0);
    }

    #[test]
    fn request_and_callback_offsets_match_native_contract() {
        assert_eq!(size_of::<MemoryRequest>(), MEMORY_REQUEST_BYTES);
        assert_eq!(
            offset_of!(MemoryRequest, address),
            MEMORY_REQUEST_ADDRESS_OFFSET as usize
        );
        assert_eq!(
            offset_of!(MemoryRequest, data),
            MEMORY_REQUEST_DATA_OFFSET as usize
        );
        assert_eq!(
            offset_of!(MemoryRequest, bytes),
            MEMORY_REQUEST_LENGTH_OFFSET as usize
        );
        assert_eq!(
            offset_of!(MemoryRequest, flags),
            MEMORY_REQUEST_FLAGS_OFFSET as usize
        );
        assert_eq!(
            offset_of!(MemoryRequest, pc),
            MEMORY_REQUEST_PC_OFFSET as usize
        );
        assert_eq!(
            size_of::<MemoryContextView>(),
            MEMORY_CONTEXT_BYTES as usize
        );
        assert_eq!(
            offset_of!(MemoryContextView, callback),
            MEMORY_CALLBACK_OFFSET as usize
        );
    }
    #[test]
    fn context_v2_layout_and_requirement_versions_are_not_legacy_reserved_bytes() {
        assert_eq!(offset_of!(MemoryContextView, fault_kind), 24);
        assert_eq!(offset_of!(MemoryContextView, fault_access), 28);
        assert_eq!(offset_of!(MemoryContextView, completed_bytes), 48);
        let current = MemoryRequirements::default();
        assert_eq!(current.context_version, 2);
        for version in [0, 1, 3, u32::MAX] {
            let old = MemoryRequirements {
                context_version: version,
                ..current
            };
            assert!(old.merge(current).is_none());
            assert!(current.merge(old).is_none());
        }
        let lse = MemoryRequirements {
            required_instructions: InstructionRequirements::LSE,
            ..current
        };
        let rcpc = MemoryRequirements {
            required_instructions: InstructionRequirements::RCPC2,
            ..current
        };
        let merged = lse.merge(rcpc).unwrap();
        assert!(
            merged
                .required_instructions
                .contains(InstructionRequirements::LSE)
        );
        assert!(
            merged
                .required_instructions
                .contains(InstructionRequirements::RCPC2)
        );
        assert_eq!(FaultAccess::from_code(0), Some(FaultAccess::OriginalSource));
        assert_eq!(FaultAccess::from_code(1), Some(FaultAccess::Read));
        assert_eq!(FaultAccess::from_code(2), Some(FaultAccess::Write));
        for code in [3, u32::MAX] {
            assert!(FaultAccess::from_code(code).is_none());
        }
    }
}
