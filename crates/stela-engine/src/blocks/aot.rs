//! Loading ahead-of-time compiled blocks into the block runtime.

use super::*;
use crate::artifact::{LoadedArtifact, prepare};
#[cfg(feature = "std")]
use std::path::{Path, PathBuf};
use stela_compiler::block::CompilationPlan;
use stela_contracts::{refuse, require};

const MAX_MANIFEST_CHECKSUM_BYTES: usize = 1024;

#[derive(Debug)]
pub(super) enum ArtifactPreparation {
    #[cfg(feature = "std")]
    FileCache(PathBuf),
    AheadOfTime,
    #[cfg(not(feature = "std"))]
    InMemory,
}

impl ArtifactPreparation {
    #[cfg(feature = "std")]
    pub(super) fn cache_root(&self, pc: GuestAddress) -> Result<&Path> {
        match self {
            Self::FileCache(root) => Ok(root),
            Self::AheadOfTime => Err(refuse!(
                "AOT block unavailable; runtime compilation is unsupported",
                pc = pc.0
            )
            .into()),
        }
    }
}

impl LoadedBlock {
    // Both intake paths share the same storage publication, checks and
    // insertion.
    pub(super) fn from_artifact(
        target: stela_compiler::TargetCpu,
        profile: BlockProfile,
        region: Rc<CodeRegion>,
        source_bytes: usize,
        prepared: LoadedArtifact,
        native: &[u8],
    ) -> Result<Self> {
        if !crate::platform::cpu::admitted(target) {
            return Err(crate::Error::HostLacks(target));
        }
        require!(source_bytes > 0, "compiled block has no source");
        let mapping = platform::Mapping::new(native)?;
        serialize_instruction_execution()?;
        Ok(Self {
            profile,
            region,
            source_bytes,
            prepared,
            mapping,
        })
    }
}

impl BlockRuntime {
    pub(super) fn with_preparation(
        address_space: u64,
        profile: BlockProfile,
        target: stela_compiler::TargetCpu,
        preparation: ArtifactPreparation,
    ) -> Self {
        Self {
            address_space,
            profile,
            target,
            preparation,
            next_generation: 0,
            regions: BTreeMap::new(),
            loaded: BTreeMap::new(),
            _serialized: PhantomData,
        }
    }

    /// A runtime that loads ahead-of-time artifacts. A missing body is refused
    /// before any disk-cache access or code generation.
    pub fn new_aot_for_target(
        address_space: u64,
        profile: BlockProfile,
        target: stela_compiler::TargetCpu,
    ) -> Self {
        Self::with_preparation(
            address_space,
            profile,
            target,
            ArtifactPreparation::AheadOfTime,
        )
    }

    /// Freestanding builds compile misses through the same compilation plan.
    /// There is no persistent cache without a filesystem; preloaded artifacts
    /// still work, and the strict AOT constructor still refuses misses.
    #[cfg(not(feature = "std"))]
    pub fn new_in_memory_for_target(
        address_space: u64,
        profile: BlockProfile,
        target: stela_compiler::TargetCpu,
    ) -> Self {
        Self::with_preparation(
            address_space,
            profile,
            target,
            ArtifactPreparation::InMemory,
        )
    }

    /// Load standard `manifest.json`, `manifest.sha256.json` and `native.bin`
    /// bytes. The expected identity, profile and requirements come from the
    /// compiler plan for this runtime's registered source; the source maps and
    /// native bytes are checked before the code is mapped. Nothing executes.
    pub fn preload_artifact(
        &mut self,
        pc: GuestAddress,
        manifest_bytes: &[u8],
        manifest_checksum_json: &[u8],
        native: &[u8],
    ) -> Result<()> {
        require!(pc.0.is_multiple_of(4), "unaligned preparation entry");
        require!(
            !self.loaded.contains_key(&pc.0),
            "native block already loaded"
        );
        let (base, region) = self
            .region(pc)
            .ok_or_else(|| refuse!("unmapped preparation entry", pc = pc.0))?;
        require!(region.end - pc.0 >= 4, "truncated preparation instruction");
        let offset = (pc.0 - base) as usize;
        let available = ((region.source.len() - offset) / 4).min(MAX_BLOCK_INSTRUCTIONS) * 4;
        let window = &region.source[offset..offset + available];
        let analysis = analyze_with_profile(window, pc, self.profile)?;
        let plan = CompilationPlan::new(
            window,
            pc,
            &stela_compiler::Options::new(self.profile, self.target),
        )?
        .with_resident_loops();
        let profile = plan.profile();
        let source_bytes = plan.source_bytes();
        require!(
            source_bytes == analysis.source_bytes,
            "prepared block analysis extent mismatch"
        );
        let expected_identity = crate::block_plan_identity(window, pc, self.target, &plan)?;

        // Preserve the file byte/provider rejection order, including checksum
        // codec before digest, then CPA specialization before fresh metadata.
        require!(
            manifest_bytes.len() <= prepare::MAX_CACHE_MANIFEST_BYTES,
            "cache file exceeds size limit"
        );
        require!(
            manifest_checksum_json.len() <= MAX_MANIFEST_CHECKSUM_BYTES,
            "cache file exceeds size limit"
        );
        let expected: String = serde_json::from_slice(manifest_checksum_json)?;
        prepare::validate_manifest_bytes(manifest_bytes, &expected)?;
        let manifest: crate::Manifest = serde_json::from_slice(manifest_bytes)?;
        require!(
            manifest.identity.memory_specialization_sha256
                == crate::memory_specialization_sha256(profile, manifest.memory_requirements)?,
            "cached memory specialization differs from complete requirements"
        );
        prepare::validate_preparation_metadata(
            &manifest,
            &expected_identity,
            plan.memory_requirements(),
        )?;
        prepare::validate_preparation_native(&manifest, native)?;
        prepare::validate_preparation_claim(&manifest)?;
        validate_source_map(&manifest, pc, source_bytes)?;
        let prepared = LoadedArtifact { manifest };
        let block = LoadedBlock::from_artifact(
            self.target,
            profile,
            Rc::clone(region),
            source_bytes,
            prepared,
            native,
        )?;
        self.loaded.insert(pc.0, block);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    //! Loading real compiler artifacts.
    use super::*;
    use anyhow::Result;
    use stela_compiler::{
        TargetCpu,
        context::memory::address::{AccessEl, AddressAddChecks, EffectiveAddressChecks},
    };

    const BASE: GuestAddress = GuestAddress(0x1000);
    const ADDRESS_SPACE: u64 = 7;
    const MOVZ_X0_42: u32 = 0xd280_0540;
    const B_NEXT: u32 = 0x1400_0001;
    const LDR_X0_X9_8: u32 = 0xf940_0520;
    const LDTR_X1_X9_8: u32 = 0xf840_8921;
    const RET_X30: u32 = 0xd65f_03c0;

    fn source(words: &[u32]) -> Vec<u8> {
        words.iter().flat_map(|word| word.to_le_bytes()).collect()
    }

    fn artifact(source: &[u8], profile: BlockProfile) -> Result<(crate::Manifest, Vec<u8>)> {
        let plan = CompilationPlan::new(
            source,
            BASE,
            &stela_compiler::Options::new(profile, TargetCpu::Baseline),
        )?
        .with_resident_loops();
        let identity = crate::block_plan_identity(source, BASE, TargetCpu::Baseline, &plan)?;
        let memory_requirements = plan.memory_requirements();
        let native = plan.compile()?.native;
        let manifest = crate::Manifest {
            identity,
            native_sha256: stela_formats::sha256(&native.x86_64_code),
            native_bytes: native.x86_64_code.len(),
            source_map: native.source_map,
            verified: false,
            memory_requirements,
        };
        Ok((manifest, native.x86_64_code))
    }

    fn serialized(manifest: &crate::Manifest) -> Result<(Vec<u8>, Vec<u8>)> {
        let bytes = serde_json::to_vec(manifest)?;
        let checksum = serde_json::to_vec(&stela_formats::sha256(&bytes))?;
        Ok((bytes, checksum))
    }

    fn runtime(source: &[u8], profile: BlockProfile) -> Result<BlockRuntime> {
        let mut runtime =
            BlockRuntime::new_aot_for_target(ADDRESS_SPACE, profile, TargetCpu::Baseline);
        runtime.register_code(BASE, source.to_vec())?;
        Ok(runtime)
    }

    fn outcome() -> RunOutcome {
        RunOutcome {
            reason: StopReason::BudgetExhausted,
            blocks_executed: 0,
            compiled_blocks: 0,
            disk_cache_hits: 0,
            memory_cache_hits: 0,
            last_exit: None,
            floating_point_helpers: 0,
            amx_operations: 0,
            amx_native_operations: 0,
            pointer_authentication_events: Vec::new(),
            clock_events: Vec::new(),
            thread_register_events: Vec::new(),
        }
    }

    #[test]
    fn standard_artifact_dispatches_from_memory_without_disk_or_compilation() -> Result<()> {
        let source = source(&[MOVZ_X0_42, B_NEXT]);
        let (manifest, native) = artifact(&source, BlockProfile::ORDINARY)?;
        let (bytes, checksum) = serialized(&manifest)?;
        let mut runtime = runtime(&source, BlockProfile::ORDINARY)?;
        runtime.preload_artifact(BASE, &bytes, &checksum, &native)?;
        assert_eq!(
            runtime.loaded_manifest(BASE).unwrap().identity,
            manifest.identity
        );
        #[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
        {
            let mut state = ThreadState {
                registers: stela_compiler::GuestState::default(),
                pc: BASE,
            };
            let stop = GuestAddress(BASE.0 + source.len() as u64);
            // SAFETY: the block only modifies registers and branches to the stop address.
            let observations = unsafe { runtime.run(&mut state, Some(stop), 1)? };
            assert_eq!(observations.reason, StopReason::StopAddress);
            assert_eq!(observations.memory_cache_hits, 1);
            assert_eq!(observations.disk_cache_hits, 0);
            assert_eq!(observations.compiled_blocks, 0);
            assert_eq!(observations.blocks_executed, 1);
            assert_eq!(state.registers.x[0], 42);
        }
        assert_eq!(runtime.loaded.len(), 1);
        assert!(
            runtime
                .preload_artifact(BASE, &bytes, &checksum, &native)
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn aot_miss_refuses_before_compiler_or_filesystem_preparation() -> Result<()> {
        // Unsupported instruction distinguishes availability refusal from invoking
        // the absent runtime miss compiler. Registered bytes remain unchanged.
        let source = [0; size_of::<u32>()];
        let mut runtime = runtime(&source, BlockProfile::ORDINARY)?;
        let error = runtime.prepare(BASE).unwrap_err();
        assert_eq!(
            error.to_string(),
            "AOT block unavailable; runtime compilation is unsupported (pc=0x1000)"
        );
        let mut observations = outcome();
        assert!(runtime.load(BASE, &mut observations).is_err());
        assert!(runtime.loaded.is_empty());
        assert_eq!(observations.memory_cache_hits, 0);
        assert_eq!(observations.disk_cache_hits, 0);
        assert_eq!(observations.compiled_blocks, 0);
        #[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
        {
            let mut state = ThreadState {
                registers: stela_compiler::GuestState::default(),
                pc: BASE,
            };
            let detail = stela_compiler::block::source_bytes(&source, BASE)
                .unwrap_err()
                .to_string();
            // SAFETY: dispatch rejects the unsupported word before execution.
            let stopped = unsafe { runtime.run(&mut state, None, 1)? };
            assert_eq!(stopped.reason, StopReason::Unsupported { detail });
            assert_eq!(stopped.blocks_executed, 0);
            runtime.register_code(BASE, MOVZ_X0_42.to_le_bytes().to_vec())?;
            // SAFETY: the valid register-only instruction has no preloaded body.
            let refusal = unsafe { runtime.run(&mut state, None, 1) }.unwrap_err();
            assert_eq!(refusal.to_string(), error.to_string());
            assert_eq!(state.pc, BASE);
            assert_eq!(state.registers.x[0], 0);
        }
        Ok(())
    }

    #[test]
    fn stale_source_native_map_and_verification_claim_cannot_publish_loaded_blocks() -> Result<()> {
        let source = source(&[MOVZ_X0_42, B_NEXT]);
        for change in 0..4 {
            let (mut manifest, native) = artifact(&source, BlockProfile::ORDINARY)?;
            match change {
                0 => manifest.identity.source_sha256.push('0'),
                1 => manifest.native_sha256.push('0'),
                2 => manifest.source_map[0].0 = GuestAddress(BASE.0 + 4),
                3 => manifest.verified = true,
                _ => unreachable!(),
            }
            let (bytes, checksum) = serialized(&manifest)?;
            let mut runtime = runtime(&source, BlockProfile::ORDINARY)?;
            assert!(
                runtime
                    .preload_artifact(BASE, &bytes, &checksum, &native)
                    .is_err()
            );
            assert!(runtime.loaded.is_empty());
        }
        Ok(())
    }

    #[test]
    fn complete_cpa_role_profile_is_recomputed_before_metadata_acceptance() -> Result<()> {
        use stela_compiler::block::CheckedMemoryProfile;
        let source = source(&[LDR_X0_X9_8, LDTR_X1_X9_8, RET_X30]);
        let checks =
            EffectiveAddressChecks::new(AddressAddChecks::Disabled, AddressAddChecks::Enabled);
        let profile = |normal, unprivileged| {
            BlockProfile::ORDINARY.with_checked_memory(
                CheckedMemoryProfile::SpAlignmentDisabled.with_address_checks(
                    normal,
                    unprivileged,
                    checks,
                ),
            )
        };
        let first = profile(AccessEl::El1, AccessEl::El0);
        let second = profile(AccessEl::El0, AccessEl::El1);
        let (manifest, native) = artifact(&source, first)?;
        let (second_manifest, _) = artifact(&source, second)?;
        assert_eq!(
            manifest.memory_requirements,
            second_manifest.memory_requirements
        );
        assert_ne!(
            manifest.identity.memory_specialization_sha256,
            second_manifest.identity.memory_specialization_sha256
        );
        let (bytes, checksum) = serialized(&manifest)?;
        let mut runtime = runtime(&source, second)?;
        let error = runtime
            .preload_artifact(BASE, &bytes, &checksum, &native)
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            "cached memory specialization differs from complete requirements"
        );
        assert!(runtime.loaded.is_empty());
        Ok(())
    }

    #[test]
    fn replacing_registered_source_retires_preloaded_native_generation() -> Result<()> {
        let source = source(&[MOVZ_X0_42, B_NEXT]);
        let (manifest, native) = artifact(&source, BlockProfile::ORDINARY)?;
        let (bytes, checksum) = serialized(&manifest)?;
        let mut runtime = runtime(&source, BlockProfile::ORDINARY)?;
        runtime.preload_artifact(BASE, &bytes, &checksum, &native)?;
        assert!(runtime.unregister_code(BASE));
        assert!(runtime.loaded.is_empty());
        runtime.register_code(BASE, [0; size_of::<u32>()].to_vec())?;
        assert!(
            runtime
                .preload_artifact(BASE, &bytes, &checksum, &native)
                .is_err()
        );
        assert!(runtime.loaded.is_empty());
        Ok(())
    }
}
