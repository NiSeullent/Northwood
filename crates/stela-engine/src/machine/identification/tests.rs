use super::*;

#[test]
fn containing_feature_dependencies_are_acyclic() {
    fn visit(feature: &str, path: &mut Vec<&'static str>) {
        for row in TABLE.iter().filter(|row| row.feature == Some(feature)) {
            if let Basis::With(containing, _) = row.basis {
                assert!(!path.contains(&containing), "feature cycle: {path:?}");
                path.push(containing);
                visit(containing, path);
                path.pop();
            }
        }
    }
    for row in TABLE {
        if let Some(feature) = row.feature {
            visit(feature, &mut vec![feature]);
        }
    }
}

#[test]
fn identification_fields_fit_their_independent_architectural_ranges() {
    for row in TABLE {
        assert!(row.width != 0 && u16::from(row.shift) + u16::from(row.width) <= 64);
        assert_eq!(row.level & !row.mask(), 0, "{row:?}");
        assert_eq!(row.absent & !row.mask(), 0, "{row:?}");
        for other in TABLE.iter().filter(|other| other.register == row.register) {
            if row.shift == other.shift {
                assert_eq!(row.width, other.width, "{row:?} {other:?}");
            } else {
                assert_eq!(
                    (row.mask() << row.shift) & (other.mask() << other.shift),
                    0,
                    "overlapping fields: {row:?} {other:?}"
                );
            }
        }
    }
}

#[test]
fn full_identification_keeps_mixed_width_instruction_features() {
    for chip in Chip::ALL {
        let fixed = derive(chip, TargetCpu::Baseline, true, Speculation::NONE);
        let absent = withheld(chip, &fixed);
        for feature in [
            Feature::FEAT_AMUv1,
            Feature::FEAT_AMUv1p1,
            Feature::FEAT_SME,
            Feature::FEAT_SME2,
            Feature::FEAT_SME_F64F64,
            Feature::FEAT_SME_F8F32,
            Feature::FEAT_SME_F8F16,
            Feature::FEAT_FP8,
        ] {
            assert_eq!(absent & (1 << feature as u64), 0, "{chip:?} {feature:?}");
        }
        // Clearing a field must deny its feature with the adjacent
        // fields kept. Full AMU SIZE above must read all six bits.
        for (register, shift, feature) in [
            (ID_AA64SMFR0_EL1, 32, Feature::FEAT_SME),
            (ID_AA64SMFR0_EL1, 33, Feature::FEAT_SME2),
            (ID_AA64FPFR0_EL1, 0, Feature::FEAT_FP8),
            (AMCFGR_EL0, 8, Feature::FEAT_AMUv1),
        ] {
            if !feature.on(chip) {
                continue;
            }
            let row = TABLE
                .iter()
                .find(|row| row.register == register && row.shift == shift)
                .unwrap();
            let mut masked = fixed.clone();
            for (candidate, value) in &mut masked {
                if *candidate == register {
                    *value &= !(row.mask() << shift);
                }
            }
            assert_ne!(
                withheld(chip, &masked) & (1 << feature as u64),
                0,
                "{chip:?} {feature:?}"
            );
        }
    }
}

#[test]
fn ordinary_matrix_and_bfloat_features_do_not_require_streaming() {
    const SMMLA: u32 = 0x4e9b_a781;
    const BFDOT: u32 = 0x6e42_fc20;
    let bit = |feature: Feature| 1u64 << feature as u64;
    // FEAT_BF16 is steered: never withheld, whatever its fields read.
    let ordinary = bit(Feature::FEAT_I8MM);
    for chip in Chip::ALL {
        let mut fixed = derive(chip, TargetCpu::Baseline, true, Speculation::NONE);
        let mut missing_streaming_fields = fixed.clone();
        for (register, value) in &mut missing_streaming_fields {
            if *register == ID_AA64ZFR0_EL1 {
                *value = 0;
            }
        }
        assert_eq!(
            withheld(chip, &missing_streaming_fields) & ordinary,
            if Feature::FEAT_SME.on(chip) {
                ordinary
            } else {
                0
            },
            "a presented containing feature requires its fields: {chip:?}"
        );
        for hidden_streaming in [false, true] {
            if hidden_streaming {
                for (register, value) in &mut fixed {
                    match *register {
                        ID_AA64PFR1_EL1 => *value &= !(ID_FIELD << PFR1_SME),
                        ID_AA64SMFR0_EL1 | ID_AA64ZFR0_EL1 => *value = 0,
                        _ => {}
                    }
                }
            }
            let withheld = withheld(chip, &fixed);
            assert_eq!(withheld & ordinary, 0, "{chip:?} {hidden_streaming}");
            let admission = chip.admit_table(withheld).unwrap();
            for word in [SMMLA, BFDOT] {
                assert_eq!(
                    admission.admit(word),
                    if chip == Chip::AppleM1 { 0 } else { word },
                    "{chip:?} {hidden_streaming} {word:#x}"
                );
            }
            if hidden_streaming && Feature::FEAT_SME.on(chip) {
                assert_ne!(withheld & bit(Feature::FEAT_SME), 0);
                assert_eq!(admission.admit(0x8081_2000), 0, "FMOPA stays absent");
            }
        }
        // Masking the ordinary field still denies the ordinary feature,
        // independently of whether streaming fields would present it.
        for (register, value) in &mut fixed {
            if *register == ID_AA64ISAR1_EL1 {
                *value &= !(ID_FIELD << 44 | ID_FIELD << 52);
            }
        }
        let expected = if chip == Chip::AppleM1 { 0 } else { ordinary };
        assert_eq!(withheld(chip, &fixed) & ordinary, expected, "{chip:?}");
        assert_eq!(withheld(chip, &fixed) & bit(Feature::FEAT_BF16), 0);
        let admission = chip.admit_table(withheld(chip, &fixed)).unwrap();
        assert_eq!(admission.admit(SMMLA), 0, "{chip:?}");
        assert_eq!(
            admission.admit(BFDOT),
            if chip == Chip::AppleM1 { 0 } else { BFDOT },
            "{chip:?}"
        );
    }
}

/// A field masked to 0 withholds the features its rows describe, and a
/// field reading a later row's level presents the earlier rows'.
#[test]
fn a_masked_field_withholds_its_features() {
    let full = derive(Chip::AppleM5, TargetCpu::Baseline, true, Speculation::NONE);
    let bit = |feature: Feature| 1u64 << feature as u64;
    let mte = bit(Feature::FEAT_MTE) | bit(Feature::FEAT_MTE2);
    assert_eq!(withheld(Chip::AppleM5, &full) & mte, 0);
    let mut masked = full.clone();
    for (register, value) in &mut masked {
        if *register == ID_AA64PFR1_EL1 {
            *value &= !(0xf << PFR1_MTE | 0xf << PFR1_MTE_FRAC);
        }
    }
    assert_eq!(
        withheld(Chip::AppleM5, &masked) & !withheld(Chip::AppleM5, &full),
        mte
    );
    // FEAT_MTE alone at level 1 presents FEAT_MTE, not FEAT_MTE2.
    for (register, value) in &mut masked {
        if *register == ID_AA64PFR1_EL1 {
            *value |= 1 << PFR1_MTE;
        }
    }
    assert_eq!(
        withheld(Chip::AppleM5, &masked) & mte,
        bit(Feature::FEAT_MTE2)
    );
    // A profile without the register withholds nothing.
    assert_eq!(withheld(Chip::AppleM5, &[]), 0);
}

#[test]
fn every_implemented_feature_that_varies_by_chip_is_identified() {
    // The varying features with no ID field level of their own.
    const NO_FIELD: [(&str, &str); 3] = [
        ("v8Ap6", "an architecture version, not a field"),
        ("v8Ap7", "an architecture version, not a field"),
        (
            "FEAT_PAuth2",
            "APA 3, below the FEAT_FPAC level 4 of every chip with it",
        ),
    ];
    for feature in Feature::ALL {
        if !Chip::ALL
            .iter()
            .any(|chip| crate::machine::capabilities::implemented(*chip, feature) == Ok(true))
        {
            continue;
        }
        let name = feature.name();
        assert!(
            TABLE.iter().any(|row| row.feature == Some(name))
                || NO_FIELD.iter().any(|(field, _)| *field == name),
            "{name} is implemented but has no identification row"
        );
    }
}

#[test]
fn minimal_feature_configurations_are_advertised() {
    let values = derive(
        crate::machine::test_chip(),
        TargetCpu::detect(),
        true,
        Speculation::NONE,
    );
    let value = |register| values.iter().find(|(r, _)| *r == register).unwrap().1;
    let field = |register, shift: u32| (value(register) >> shift) & 0xf;
    // RASv1p1 (RASv2 on the M6), MPAM v1.0 and the AMU (none on the M1
    // to M3, AMUv1p1 from the M4) in ID_AA64PFR0_EL1; LOR in MMFR1.
    let chip = crate::machine::test_chip();
    assert_eq!(
        field(ID_AA64PFR0_EL1, 28),
        if has(chip, "FEAT_RASv2") { 3 } else { 2 }
    );
    assert_eq!(field(ID_AA64PFR0_EL1, 40), 1);
    let amu = Feature::FEAT_AMUv1.on(chip);
    assert_eq!(
        field(ID_AA64PFR0_EL1, 44),
        if Feature::FEAT_AMUv1p1.on(chip) {
            2
        } else {
            u64::from(amu)
        }
    );
    // Four architected 64-bit counters and no auxiliary group.
    let (amcfgr, amcgcr) = if amu { (0x0100_3f03, 4) } else { (0, 0) };
    assert_eq!(value(AMCFGR_EL0), amcfgr);
    assert_eq!(value(AMCGCR_EL0), amcgcr);
    assert_eq!(value(AMCG1IDR_EL0), 0);
    // CSV2 and CSV3 (PFR0 56, 60) and CSV2_frac (PFR1 32) read 0 on a
    // host that gives no speculation guarantee. DIT (PFR0 48) needs a
    // host that explicitly declares DOIT; ordinary target detection does not.
    for shift in [48, 56, 60] {
        assert_eq!(field(ID_AA64PFR0_EL1, shift), 0, "PFR0 {shift}");
    }
    assert_eq!(field(ID_AA64PFR1_EL1, 32), 0);
    assert_eq!(field(ID_AA64MMFR1_EL1, 16), 1);
    assert_eq!(field(ID_AA64MMFR4_EL1, 24), 0b1110);
    for register in [
        LORID_EL1,
        MPAMIDR_EL1,
        ERRIDR_EL1,
        CLIDR_EL1,
        CCSIDR_EL1,
        MDRAR_EL1,
        REVIDR_EL1,
    ] {
        assert_eq!(value(register), 0, "{register:?}");
    }
    // ID_AA64ISAR3_EL1 describes only the M6's CPA, FAMINMAX, LUT and
    // PAuth_LR, which the instruction-field test checks.
    if !["FEAT_CPA", "FEAT_FAMINMAX", "FEAT_LUT", "FEAT_PAuth_LR"]
        .iter()
        .any(|feature| has(chip, feature))
    {
        assert_eq!(value(ID_AA64ISAR3_EL1), 0);
    }
    assert_eq!(value(DBGAUTHSTATUS_EL1), 0b1010);
}

#[test]
fn csv_fields_disclose_the_speculation_guarantees_the_host_gives() {
    const CSV2: u32 = 56;
    const CSV3: u32 = 60;
    const CSV2_FRAC: u32 = 32;
    let fields = |chip, speculation| {
        let values = derive(chip, TargetCpu::detect(), true, speculation);
        let value = |register| values.iter().find(|(r, _)| *r == register).unwrap().1;
        (
            value(ID_AA64PFR0_EL1) >> CSV2 & 0xf,
            value(ID_AA64PFR0_EL1) >> CSV3 & 0xf,
            value(ID_AA64PFR1_EL1) >> CSV2_FRAC & 0xf,
        )
    };
    // The chip has FEAT_CSV2_2 (CSV2 2, where CSV2_frac is 0) and, from
    // Armv8.5, FEAT_CSV3, each disclosed only where the host gives its
    // guarantee.
    let chip = crate::machine::test_chip();
    assert!(has(chip, "FEAT_CSV2_2"));
    assert_eq!(fields(chip, Speculation::NONE), (0, 0, 0));
    assert_eq!(
        fields(
            chip,
            Speculation {
                contexts: true,
                faulting_data: false
            }
        ),
        (2, 0, 0)
    );
    let faulting_data = |contexts| Speculation {
        contexts,
        faulting_data: true,
    };
    assert_eq!(fields(chip, faulting_data(false)), (0, 1, 0));
    assert_eq!(fields(chip, faulting_data(true)), (2, 1, 0));
}

#[test]
fn dit_is_advertised_only_with_a_data_independent_timing_target() {
    const DIT_SHIFT: u32 = 48;
    let dit = |chip, target| {
        let values = derive(chip, target, true, Speculation::NONE);
        let pfr0 = values
            .iter()
            .find(|(r, _)| *r == ID_AA64PFR0_EL1)
            .unwrap()
            .1;
        (pfr0 >> DIT_SHIFT) & 0xf
    };
    let chip = crate::machine::test_chip();
    let expected = u64::from(has(chip, "FEAT_DIT"));
    // A host that declares DOIT and supports an audited tier.
    for tier in TargetCpu::DATA_INDEPENDENT_TIMING_TIERS {
        assert_eq!(dit(chip, tier.with_doit()), expected, "{tier:?}");
        assert_eq!(dit(chip, tier), 0, "{tier:?} without DOIT");
    }
    // DOIT below every audited tier.
    assert_eq!(dit(chip, TargetCpu::Sse42.with_doit()), 0);
}

#[test]
fn advertised_instruction_fields_match_what_the_decoder_compiles() {
    let target = TargetCpu::detect();
    let values = derive(
        crate::machine::test_chip(),
        target,
        false,
        Speculation::NONE,
    );
    let value = |register| values.iter().find(|(r, _)| *r == register).unwrap().1;
    for row in TABLE {
        let Basis::Instructions(samples) = row.basis else {
            continue;
        };
        // Each row on the first chip with its feature.
        let chip = row.feature.map_or(Chip::DEFAULT, |feature| {
            Chip::ALL
                .into_iter()
                .find(|chip| has(*chip, feature))
                .unwrap()
        });
        let chip_values = derive(chip, target, false, Speculation::NONE);
        let chip_value = |register| chip_values.iter().find(|(r, _)| *r == register).unwrap().1;
        let field = (chip_value(row.register) >> row.shift) & row.mask();
        let compiles = samples.iter().all(|word| {
            CompilationPlan::new(
                &word.to_le_bytes(),
                GuestAddress(0x1000),
                &stela_compiler::Options::new(BlockProfile::new(Features::SYSTEM_MODE), target),
            )
            .and_then(CompilationPlan::compile)
            .is_ok()
        });
        // A satisfied row means the field reads at least its level; an
        // unsatisfied lowest row means it reads `absent`.
        if compiles {
            assert!(
                field == row.level || field > row.level && field != row.absent,
                "{chip:?} {:?} field {} reads {field}",
                row.register,
                row.shift
            );
        }
        let lowest = TABLE
            .iter()
            .filter(|other| other.register == row.register && other.shift == row.shift)
            .map(|other| other.level)
            .min()
            .unwrap();
        if !compiles && row.level == lowest {
            assert_eq!(field, row.absent, "{:?} field {}", row.register, row.shift);
        }
    }
    // Features the M2 lacks are never advertised: SVE stays 0, and
    // ZFR0 describes streaming SVE only on a chip with SME.
    assert_eq!(value(ID_AA64PFR0_EL1) >> 32 & 0xf, 0);
    if !has(crate::machine::test_chip(), "FEAT_SME") {
        assert_eq!(value(ID_AA64ZFR0_EL1), 0);
    }
}

#[test]
fn fixed_and_engine_fields() {
    let values = derive(
        crate::machine::test_chip(),
        TargetCpu::detect(),
        false,
        Speculation::NONE,
    );
    let value = |register| values.iter().find(|(r, _)| *r == register).unwrap().1;
    // EL0/EL1 AArch64, no EL2/EL3; 48-bit PA, 16-bit ASID, 16K and 4K but
    // not 64K granules; PAN2, or PAN3 from the M4; Armv8.0 debug without
    // double lock.
    let chip = crate::machine::test_chip();
    assert_eq!(value(ID_AA64PFR0_EL1) & 0xffff, 0x0011);
    assert_eq!(value(ID_AA64MMFR0_EL1) & 0xffff_ffff, 0x0f10_0005);
    assert_eq!(
        value(ID_AA64MMFR1_EL1) >> 20 & 0xf,
        if has(chip, "FEAT_PAN3") { 3 } else { 2 }
    );
    assert_eq!(value(ID_AA64DFR0_EL1) & 0xf0f0_f00f, 0x1030_5009);
    // Every other debug field reads its absent value: no PMU, trace,
    // SPE, BRBE or trace buffer, and no OS Double Lock (0xf). The
    // extended counts of ID_AA64DFR1_EL1 defer to ID_AA64DFR0_EL1, and
    // ID_AA64DFR2_EL1 describes only absent features.
    assert_eq!(value(ID_AA64DFR0_EL1), 0xf0_1030_5009);
    assert_eq!(value(ID_AA64DFR1_EL1), 0);
    assert_eq!(value(ID_AA64DFR2_EL1), 0);
    // With FEAT_MTE2, TminLine 4.
    let tag_line = if has(chip, "FEAT_MTE2") { 4 << 32 } else { 0 };
    assert_eq!(value(CTR_EL0), 0x9444_c004 | tag_line);
    let scaled = u64::from(crate::machine::timer::agt::present(chip)) << 32;
    assert_eq!(value(AIDR_EL1), 1 << 1 | 1 << 4 | 1 << 16 | scaled);
    assert_eq!(value(ID_AA64MMFR1_EL1) >> 8 & 0xf, 0);
    // No stage 2 without EL2; HPDS; CnP, TTL and E0PD.
    assert_eq!(value(ID_AA64MMFR0_EL1) >> 32 & 0xfff, 0x111);
    assert_eq!(value(ID_AA64MMFR1_EL1) >> 12 & 0xf, 1);
    assert_eq!(value(ID_AA64MMFR1_EL1) >> 28 & 0xf, 0);
    assert_eq!(
        value(ID_AA64MMFR2_EL1) & (0xf << 60 | 0xf << 48 | 0xf << 40 | 0xf),
        1 << 60 | 1 << 48 | 1
    );
    // With EL2: AArch64 EL2 and VHE.
    let system = derive(
        crate::machine::test_chip(),
        TargetCpu::detect(),
        true,
        Speculation::NONE,
    );
    let host = |register| system.iter().find(|(r, _)| *r == register).unwrap().1;
    assert_eq!(host(ID_AA64PFR0_EL1), value(ID_AA64PFR0_EL1) | 0x100);
    // With EL2, stage 2 adds 4 KiB and 16 KiB stage-2 granules, XNX and
    // FWB.
    assert_eq!(
        host(ID_AA64MMFR1_EL1),
        value(ID_AA64MMFR1_EL1) | 0x100 | 1 << 28
    );
    assert_eq!(
        host(ID_AA64MMFR0_EL1),
        value(ID_AA64MMFR0_EL1) + (1 << 32) + (1 << 40)
    );
    // FWB and FEAT_EVT2's HCR_EL2 traps; IDST without EL2.
    assert_eq!(
        host(ID_AA64MMFR2_EL1),
        value(ID_AA64MMFR2_EL1) | 1 << 40 | 2 << 56
    );
    assert_eq!(value(ID_AA64MMFR2_EL1) >> 36 & 0xf, 1);
    for &register in REGISTERS {
        if !matches!(
            register,
            ID_AA64PFR0_EL1 | ID_AA64MMFR0_EL1 | ID_AA64MMFR1_EL1 | ID_AA64MMFR2_EL1
        ) {
            assert_eq!(host(register), value(register));
        }
    }
}

#[test]
fn each_chip_advertises_only_its_own_features() {
    let field = |chip: Chip, register, shift: u32| {
        let values = derive(chip, TargetCpu::detect(), true, Speculation::NONE);
        let value = values.iter().find(|(r, _)| *r == register).unwrap().1;
        (value >> shift) & 0xf
    };
    // FGT and ECV (MMFR0) and I8MM (ISAR1) came with the M2, as did BF16,
    // which is steered (no host tier lowers it natively) and reads 0.
    for chip in Chip::ALL {
        assert_eq!(field(chip, ID_AA64ISAR1_EL1, 44), 0, "{chip:?}");
    }
    for (register, shift) in [
        (ID_AA64MMFR0_EL1, 56),
        (ID_AA64MMFR0_EL1, 60),
        (ID_AA64ISAR1_EL1, 52),
    ] {
        assert_eq!(
            field(Chip::AppleM1, register, shift),
            0,
            "{register:?} {shift}"
        );
        assert_ne!(
            field(Chip::AppleM2, register, shift),
            0,
            "{register:?} {shift}"
        );
    }
    // The M2's FPAC (APA 4 in ISAR1) reads as classic PAuth on the M1,
    // whose PACs the engine computes that way.
    assert_eq!(field(Chip::AppleM2, ID_AA64ISAR1_EL1, 4), 4);
    assert_eq!(field(Chip::AppleM1, ID_AA64ISAR1_EL1, 4), 1);
    // The M6 has FGT2 (MMFR0.FGT 2) and RASv2 (PFR0.RAS 3).
    assert_eq!(field(Chip::AppleM5, ID_AA64MMFR0_EL1, 56), 1);
    assert_eq!(field(Chip::AppleM6, ID_AA64MMFR0_EL1, 56), 2);
    assert_eq!(field(Chip::AppleM5, ID_AA64PFR0_EL1, 28), 2);
    assert_eq!(field(Chip::AppleM6, ID_AA64PFR0_EL1, 28), 3);
    // The M6's FEAT_PAuth_LR (APA 6, with ISAR3.PACM 2 for a full
    // PSTATE.PACM) needs FEAT_SCTLR2 (MMFR3.SCTLRX).
    assert_eq!(field(Chip::AppleM5, ID_AA64ISAR1_EL1, 4), 4);
    assert_eq!(field(Chip::AppleM6, ID_AA64ISAR1_EL1, 4), 6);
    for chip in Chip::ALL {
        let m6 = chip == Chip::AppleM6;
        assert_eq!(field(chip, ID_AA64ISAR3_EL1, 12), 2 * u64::from(m6));
        assert_eq!(field(chip, ID_AA64MMFR3_EL1, 4), u64::from(m6));
        // The M6's FEAT_FPMR (PFR2.FPMR) and FEAT_FP8 conversions
        // (FPFR0.F8CVT, F8E4M3 and F8E5M2).
        assert_eq!(field(chip, ID_AA64PFR2_EL1, 32), u64::from(m6));
        let fpfr0 = derive(chip, TargetCpu::detect(), true, Speculation::NONE)
            .into_iter()
            .find(|(register, _)| *register == ID_AA64FPFR0_EL1)
            .unwrap()
            .1;
        assert_eq!(fpfr0, if m6 { 1 << 31 | 0b11 } else { 0 }, "{chip:?}");
        // Spec_FPACC, present with the M6's FEAT_FPACCOMBINE, reads 0:
        // not disclosed.
        assert_eq!(field(chip, ID_AA64MMFR3_EL1, 60), 0);
        // The M5's and M6's FEAT_MTE2 with synchronous Tag Checks only
        // (PFR1.MTE 2, MTE_frac 0b1111), the 64-byte tag line of
        // CTR_EL0.TminLine and GMID_EL1.BS.
        let mte = matches!(chip, Chip::AppleM5 | Chip::AppleM6);
        assert_eq!(field(chip, ID_AA64PFR1_EL1, 8), 2 * u64::from(mte));
        assert_eq!(memory_tagging(chip), mte, "{chip:?}");
        assert_eq!(
            breakpoints_and_watchpoints(chip),
            (
                field(chip, ID_AA64DFR0_EL1, DFR0_BRPS.into()) + 1,
                field(chip, ID_AA64DFR0_EL1, DFR0_WRPS.into()) + 1
            )
        );
        assert_eq!(field(chip, ID_AA64PFR1_EL1, 40), 0xf * u64::from(mte));
        assert_eq!(field(chip, CTR_EL0, 32), 4 * u64::from(mte));
        assert_eq!(field(chip, GMID_EL1, 0), 4 * u64::from(mte));
    }
    // Apple clang drops SSBS from the M4 on (PFR1.SSBS).
    assert_ne!(field(Chip::AppleM3, ID_AA64PFR1_EL1, 4), 0);
    assert_eq!(field(Chip::AppleM4, ID_AA64PFR1_EL1, 4), 0);
    // Armv8.7 implies FEAT_PAN3 on the M4, though clang names no
    // extension for it (MMFR1.PAN).
    assert_eq!(field(Chip::AppleM3, ID_AA64MMFR1_EL1, 20), 2);
    assert_eq!(field(Chip::AppleM4, ID_AA64MMFR1_EL1, 20), 3);
    assert_eq!(field(Chip::AppleM3, ID_AA64MMFR1_EL1, 44), 0);
    assert_eq!(field(Chip::AppleM4, ID_AA64MMFR1_EL1, 44), 1);
    // Every chip has FEAT_LSE2 (MMFR2.AT), which a host with atomic
    // aligned 16-byte accesses implements.
    let lse2 = u64::from(TargetCpu::detect().supports(TargetCpu::AtomicLoad16));
    for chip in Chip::ALL {
        assert_eq!(field(chip, ID_AA64MMFR2_EL1, 32), lse2, "{chip:?}");
    }
    // Every chip restricts predictions (ISAR1.SPECRES 1); the M5 adds
    // COSP RCTX (2).
    for chip in Chip::ALL {
        let expected = if chip >= Chip::AppleM5 { 2 } else { 1 };
        assert_eq!(field(chip, ID_AA64ISAR1_EL1, 40), expected, "{chip:?}");
    }
    // CSSC and HBC came with the M5, LUT and FAMINMAX with the M6.
    for (register, shift, first) in [
        (ID_AA64ISAR2_EL1, 52, Chip::AppleM5),
        (ID_AA64ISAR2_EL1, 20, Chip::AppleM5),
        (ID_AA64ISAR2_EL1, 56, Chip::AppleM6),
        (ID_AA64ISAR3_EL1, 4, Chip::AppleM6),
        (ID_AA64ISAR3_EL1, 0, Chip::AppleM6),
    ] {
        for chip in Chip::ALL {
            let expected = u64::from(chip >= first);
            assert_eq!(field(chip, register, shift), expected, "{chip:?} {shift}");
        }
    }
    // The chip the tests run on advertises the prediction restrictions
    // of its FEAT_SPECRES2 and FEAT_LSE2's alignment.
    let chip = crate::machine::test_chip();
    let specres = 1 + u64::from(stela_contracts::Feature::FEAT_SPECRES2.on(chip));
    assert_eq!(field(chip, ID_AA64ISAR1_EL1, 40), specres);
    assert_eq!(field(chip, ID_AA64MMFR2_EL1, 32), lse2);
}

/// FEAT_LSE2 (MMFR2.AT) only where aligned 16-byte host accesses are
/// single-copy atomic, which 16-byte-aligned X-register pairs compile to.
#[test]
fn lse2_is_advertised_only_with_atomic_aligned_16_byte_accesses() {
    let at = |target: TargetCpu| {
        derive(crate::machine::test_chip(), target, true, Speculation::NONE)
            .into_iter()
            .find(|(register, _)| *register == ID_AA64MMFR2_EL1)
            .map_or(0, |(_, value)| value >> 32 & 0xf)
    };
    assert_eq!(at(TargetCpu::Baseline), 0);
    assert_eq!(at(TargetCpu::Baseline.with_atomic_load16()), 1);
}

/// ID_AA64MMFR2_EL1.VARange (bits 19:16) is 0 on every chip and host
/// target: development kernels assert it (XNU `vm_sanitize.c`, under
/// MACH_ASSERT; darwin-vm issue 14), and no chip has FEAT_LVA.
#[test]
fn every_chip_keeps_varange_0() {
    for chip in Chip::ALL {
        for target in [TargetCpu::Baseline, TargetCpu::detect()] {
            let mmfr2 = derive(chip, target, true, Speculation::NONE)
                .into_iter()
                .find(|(register, _)| *register == ID_AA64MMFR2_EL1)
                .map_or(0, |(_, value)| value);
            assert_eq!(mmfr2 >> 16 & 0xf, 0, "{chip:?} {target:?}");
        }
    }
}

#[test]
fn the_sme_chips_advertise_streaming_sve_and_sme_by_feature() {
    let value = |chip, register| {
        derive(chip, TargetCpu::detect(), true, Speculation::NONE)
            .into_iter()
            .find(|(r, _)| *r == register)
            .unwrap()
            .1
    };
    // PFR1.SME 2 (FEAT_SME2) from the M4. SMFR0: F32F32, BI32I32,
    // B16F32, F16F32, I8I32 0xf, I16I32 0b0101, F64F64, I16I64 0xf and
    // SMEver 1 on the M4; the M5 adds F16F16, B16B16 and SMEver 2
    // (FEAT_SME2p1), the M6 F8F32, F8F16 and LUTv2.
    // ZFR0: SVEver 0, I8MM, and B16B16 from the M5; BF16 reads 0 with the
    // steered FEAT_BF16.
    for (chip, sme, smfr0, zfr0) in [
        (Chip::AppleM1, 0, 0, 0),
        (Chip::AppleM2, 0, 0, 0),
        (Chip::AppleM3, 0, 0, 0),
        (Chip::AppleM4, 2, 0x01f1_50ff_0000_0000, 0x1000_0000_0000),
        (Chip::AppleM5, 2, 0x02f1_5cff_0000_0000, 0x1000_0100_0000),
        (Chip::AppleM6, 2, 0x12f1_5fff_0000_0000, 0x1000_0100_0000),
    ] {
        assert_eq!(value(chip, ID_AA64PFR1_EL1) >> 24 & 0xf, sme, "{chip:?}");
        assert_eq!(value(chip, ID_AA64SMFR0_EL1), smfr0, "{chip:?}");
        assert_eq!(value(chip, ID_AA64ZFR0_EL1), zfr0, "{chip:?}");
    }
}

#[test]
fn steered_features_follow_their_native_tier_and_stay_admitted() {
    const BFDOT: u32 = 0x6e42_fc20;
    let field = |chip, target, register, shift: u32| {
        let values = derive(chip, target, true, Speculation::NONE);
        values.iter().find(|(r, _)| *r == register).unwrap().1 >> shift & 0xf
    };
    let richest = TargetCpu::X86_64_V4
        .with_avx512bf16()
        .with_avx512fp16()
        .with_sha()
        .with_aes()
        .with_pclmulqdq();
    for target in [TargetCpu::X86_64_V2, TargetCpu::X86_64_V3, richest] {
        // No host tier lowers BF16 natively, so no host advertises it.
        assert_eq!(field(Chip::AppleM2, target, ID_AA64ISAR1_EL1, 44), 0);
        assert_eq!(field(Chip::AppleM2, target, ID_AA64ISAR1_EL1, 12), 0);
        assert_eq!(
            steering(Chip::AppleM2, target),
            [
                Steering {
                    feature: "FEAT_JSCVT",
                    advertised: false,
                    reason: JSCVT_STEERING,
                },
                Steering {
                    feature: "FEAT_BF16",
                    advertised: false,
                    reason: BF16_STEERING,
                },
            ]
        );
        // The PE still runs BFDOT for code that uses it unprobed.
        let fixed = derive(Chip::AppleM2, target, true, Speculation::NONE);
        let withheld = withheld(Chip::AppleM2, &fixed);
        assert_eq!(withheld & (1 << Feature::FEAT_BF16 as u64), 0);
        assert_eq!(
            Chip::AppleM2.admit_table(withheld).unwrap().admit(BFDOT),
            BFDOT
        );
    }
    // A chip without the feature reports no steering for it.
    assert!(
        steering(Chip::AppleM1, richest)
            .iter()
            .all(|entry| entry.feature != "FEAT_BF16")
    );
    // FJCVTZS still compiles and runs for code that uses it unprobed.
    assert!(compiles(&[0x1e7e_0020], TargetCpu::X86_64_V2));
    // A steered row with a native tier advertises only where the host has it.
    assert!(!steered_holds(
        &[BFDOT],
        Some(TargetCpu::Avx512bf16),
        TargetCpu::X86_64_V3
    ));
    assert!(steered_holds(
        &[BFDOT],
        Some(TargetCpu::Avx512bf16),
        richest
    ));
    assert!(!steered_holds(&[BFDOT], None, richest));
}
