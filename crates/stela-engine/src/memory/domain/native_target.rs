//! A conservative limit, never CPU or memory execution permission. Keep the
//! memory guarantees and instruction recipes both sides have, which is what
//! blocks compile for. The current side's VEX recipes already need its
//! live YMM state, so a limit only removes them; AVX-512 state and DIT
//! cannot be enabled by a caller-provided declaration.
use stela_compiler::TargetCpu;

pub(in crate::memory) fn restrict(actual: TargetCpu, limit: TargetCpu) -> TargetCpu {
    let mut target = [TargetCpu::Sse42, TargetCpu::Sse41, TargetCpu::Ssse3]
        .into_iter()
        .find(|tier| actual.supports(*tier) && limit.supports(*tier))
        .unwrap_or(TargetCpu::Baseline);
    for (capability, with) in [
        (
            TargetCpu::Pclmulqdq,
            TargetCpu::with_pclmulqdq as fn(_) -> _,
        ),
        (TargetCpu::Popcnt, TargetCpu::with_popcnt),
        (TargetCpu::Aes, TargetCpu::with_aes),
        (TargetCpu::Sha, TargetCpu::with_sha),
        (TargetCpu::Gfni, TargetCpu::with_gfni),
        (TargetCpu::Lzcnt, TargetCpu::with_lzcnt),
        (TargetCpu::Cx8, TargetCpu::with_cx8),
        (TargetCpu::Cx16, TargetCpu::with_cx16),
        (TargetCpu::AtomicLoad16, TargetCpu::with_atomic_load16),
        (TargetCpu::LineAtomic, TargetCpu::with_line_atomic),
        // Scalar GPR extensions, which use no extended register state.
        (TargetCpu::Bmi1, TargetCpu::with_bmi1),
        (TargetCpu::Bmi2, TargetCpu::with_bmi2),
        (TargetCpu::FastPdep, TargetCpu::with_fast_pdep),
        (TargetCpu::Movbe, TargetCpu::with_movbe),
        // VEX vector recipes, which the current side admits only with
        // OS-enabled YMM state.
        (TargetCpu::Fma, TargetCpu::with_fma),
        (TargetCpu::F16c, TargetCpu::with_f16c),
        (TargetCpu::Avx2, TargetCpu::with_avx2),
    ] {
        if actual.supports(capability) && limit.supports(capability) {
            target = with(target);
        }
    }
    target
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every capability `restrict` keeps.
    const LEGACY: TargetCpu = TargetCpu::X86_64_V2
        .with_pclmulqdq()
        .with_aes()
        .with_sha()
        .with_gfni()
        .with_lzcnt()
        .with_atomic_load16()
        .with_line_atomic();

    #[test]
    fn neither_limit_nor_current_target_can_promote_the_other() {
        for left in [
            TargetCpu::Baseline,
            TargetCpu::Ssse3,
            TargetCpu::Sse41,
            TargetCpu::Sse42,
            LEGACY,
            TargetCpu::X86_64_V4,
        ] {
            for right in [
                TargetCpu::Baseline,
                TargetCpu::Ssse3,
                TargetCpu::Sse41,
                TargetCpu::Sse42,
                TargetCpu::Aes,
                TargetCpu::Sha,
                TargetCpu::Gfni,
                TargetCpu::Lzcnt,
                TargetCpu::AtomicLoad16,
                TargetCpu::LineAtomic,
                LEGACY,
            ] {
                let common = restrict(left, right);
                assert!(left.supports(common));
                assert!(right.supports(common));
                assert_eq!(common, restrict(right, left));
                assert_eq!(restrict(common, left), common);
            }
        }
    }

    #[test]
    fn independent_capability_loss_does_not_remove_other_admitted_recipes() {
        let actual = TargetCpu::Sse42.with_aes().with_sha().with_atomic_load16();
        let worker = TargetCpu::Sse41.with_aes().with_cx16();
        assert_eq!(restrict(actual, worker), TargetCpu::Sse41.with_aes());
        assert_eq!(
            restrict(TargetCpu::Lzcnt.with_gfni(), TargetCpu::Gfni),
            TargetCpu::Gfni
        );
        assert_eq!(
            restrict(TargetCpu::Baseline, TargetCpu::X86_64_V4),
            TargetCpu::Baseline
        );
    }

    #[test]
    fn caller_modes_and_extended_state_are_never_promoted_by_a_limit() {
        let rich = TargetCpu::X86_64_V4
            .with_doit()
            .with_line_atomic()
            .with_data_independent_timing();
        let common = restrict(rich, rich);
        assert!(!common.data_independent_timing());
        // VEX vector recipes stay only where both sides have them.
        for kept in [TargetCpu::Fma, TargetCpu::F16c, TargetCpu::Avx2] {
            assert!(common.supports(kept));
            assert!(!restrict(TargetCpu::X86_64_V2, rich).supports(kept));
            assert!(!restrict(rich, TargetCpu::X86_64_V2).supports(kept));
        }
        for excluded in [TargetCpu::Avx512f, TargetCpu::Doit] {
            assert!(!common.supports(excluded));
        }
        // Scalar GPR extensions use no extended state.
        for kept in [TargetCpu::Bmi1, TargetCpu::Bmi2, TargetCpu::Movbe] {
            assert!(common.supports(kept));
        }
    }

    #[test]
    fn line_atomicity_survives_only_the_current_and_common_worker_intersection() {
        use stela_compiler::{
            GuestAddress, Options,
            block::{BlockProfile, Features},
        };
        let intel = TargetCpu::X86_64_V2.with_line_atomic();
        let other = TargetCpu::X86_64_V2;
        assert_eq!(restrict(intel, intel), intel);
        assert_eq!(restrict(intel, other), other);
        assert_eq!(restrict(other, intel), other);
        assert!(!restrict(intel, intel).supports(TargetCpu::AtomicLoad16));

        // LDAPR X8, [X9] from the SPTM boot uses the normal direct lowering.
        let compile = |target| {
            stela_compiler::compile(
                &0xf8bf_c128u32.to_le_bytes(),
                GuestAddress(0x1000),
                &Options::new(BlockProfile::new(Features::SYSTEM_MODE), target),
            )
        };
        assert!(compile(restrict(intel, intel)).is_ok());
        assert!(compile(restrict(intel, other)).is_err());
    }
}
