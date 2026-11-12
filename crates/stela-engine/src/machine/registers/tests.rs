use super::*;
use crate::machine::cpu::HCR_TGE;
use stela_compiler::aarch64::system::registers::decode;

/// A write of an IMPLEMENTATION DEFINED register that only stores.
const STORED: Changes = Changes {
    translation: false,
    controls: false,
    permissions: false,
    regime: false,
    interrupts: false,
    floating_point: false,
    stored: false,
    reads_kept: true,
    pointer: false,
    unmodeled: None,
    irq_steering: None,
};

fn mrs(rt: u8, encoding: Encoding) -> RegisterAccess {
    decode(
        0xd530_0000
            | (u32::from(encoding.op0() & 1) << 19)
            | (u32::from(encoding.op1()) << 16)
            | (u32::from(encoding.crn()) << 12)
            | (u32::from(encoding.crm()) << 8)
            | (u32::from(encoding.op2()) << 5)
            | u32::from(rt),
    )
    .unwrap()
}

fn msr(rt: u8, encoding: Encoding) -> RegisterAccess {
    let read = mrs(rt, encoding);
    decode(
        0xd510_0000
            | (u32::from(encoding.op0() & 1) << 19)
            | (u32::from(encoding.op1()) << 16)
            | (u32::from(encoding.crn()) << 12)
            | (u32::from(encoding.crm()) << 8)
            | (u32::from(encoding.op2()) << 5)
            | u32::from(read.rt()),
    )
    .unwrap()
}

#[test]
fn standard_el2_key_names_join_typed_bank_without_inventing_halves_or_lower_bank_aliases() {
    let pairs = [
        [Register::APIAKeyLo_EL1, Register::APIAKeyHi_EL1],
        [Register::APIBKeyLo_EL1, Register::APIBKeyHi_EL1],
        [Register::APDAKeyLo_EL1, Register::APDAKeyHi_EL1],
        [Register::APDBKeyLo_EL1, Register::APDBKeyHi_EL1],
        [Register::APGAKeyLo_EL1, Register::APGAKeyHi_EL1],
    ];
    for chip in super::super::Chip::ALL {
        let mut cpu = Cpu::of(chip, 0, Pstate::reset(2));
        for (pair, [low, high]) in pairs.into_iter().enumerate() {
            cpu.registers.set(low, 0, 0x1111);
            cpu.registers.set(high, 0, 0x2222);
            assert!(cpu.read_el2_pointer_key(low).unwrap().is_err());
            assert!(cpu.write_el2_pointer_key(low, 0));
            assert_eq!(cpu.read_el2_pointer_key(low), Some(Ok(0)));
            assert!(cpu.read_el2_pointer_key(high).unwrap().is_err());
            assert!(cpu.apple.pointer_keys.pairs()[pair].is_none());
            assert!(cpu.write_el2_pointer_key(high, 100 + pair as u64));
            assert_eq!(
                cpu.apple.pointer_keys.pairs()[pair],
                Some([0, 100 + pair as u64])
            );
            let index = pair * 2;
            let direct = Encoding::new(3, 6, 15, 13 + (index / 8) as u8, (index % 8) as u8);
            cpu.state.x[0] = 200 + pair as u64;
            assert!(matches!(
                cpu.apple_access(msr(0, direct), 0),
                Outcome::Written(_)
            ));
            assert_eq!(cpu.read_el2_pointer_key(low), Some(Ok(200 + pair as u64)));
            assert!(matches!(
                cpu.apple_access(mrs(0, direct), 0),
                Outcome::Refused(_)
            ));
            assert_eq!(cpu.read(low), 0x1111);
            assert_eq!(cpu.read(high), 0x2222);
        }
        let el2_keys = cpu.apple.pointer_keys.pairs();
        let el12 = impdef::APGAKEYLO_EL12;
        if let Some(entry) = impdef::lookup(el12, chip) {
            // Existing EL12 Guest storage is held, not applied as PAC
            // state. A nonzero word remains an explicit guest-entry gap.
            assert_eq!(entry.class, impdef::Class::Guest);
            cpu.state.x[0] = 999;
            assert!(matches!(
                cpu.apple_access(msr(0, el12), 0),
                Outcome::Written(_)
            ));
            assert_eq!(cpu.apple_access(mrs(0, el12), 0), Outcome::Read(999));
            assert_eq!(
                cpu.apple.impdef.guest_state().map(|entry| entry.encoding),
                Some(el12)
            );
            assert!(matches!(
                cpu.apple_access(msr(31, el12), 0),
                Outcome::Written(_)
            ));
            assert!(cpu.apple.impdef.guest_state().is_none());
        } else {
            assert!(!impdef::consumed(el12, chip));
            assert_eq!(cpu.apple_access(msr(31, el12), 0), Outcome::Undefined);
        }
        assert_eq!(cpu.apple.pointer_keys.pairs(), el2_keys);
        assert!(cpu.read_el2_pointer_key(Register::HCR_EL2).is_none());
        assert!(!cpu.write_el2_pointer_key(Register::HCR_EL2, 0));
        let initialized = cpu.apple.pointer_keys.pairs();
        for el in [0, 1] {
            cpu.pstate.el = el;
            assert!(cpu.read_el2_pointer_key(Register::APGAKeyLo_EL1).is_none());
            assert!(!cpu.write_el2_pointer_key(Register::APGAKeyLo_EL1, 999));
            assert_eq!(cpu.apple.pointer_keys.pairs(), initialized);
        }
        let other = Cpu::of(chip, 0, Pstate::reset(2));
        assert_eq!(other.apple.pointer_keys.pairs(), [None; 5]);
    }
}

#[test]
fn physical_count_alias_uses_supplied_time_without_virtual_offset_or_tuning_coupling() {
    for chip in super::super::Chip::ALL {
        let mut cpu = Cpu::of(chip, 0, Pstate::reset(2));
        cpu.registers.set(Register::CNTVOFF_EL2, 0, 0x1234);
        // SIQ_CFG is a separate tuning word, not a count reset.
        let siq = impdef::SIQ_CFG_EL1;
        for tuning in [0, 2, 0] {
            cpu.state.x[5] = tuning;
            assert!(matches!(
                cpu.apple_access(msr(5, siq), 0),
                Outcome::Written(_)
            ));
            for physical in [0, 0x9876_5432_1234_5678, u64::MAX] {
                assert_eq!(
                    cpu.apple_access(mrs(0, ACNTPCT_EL0), physical),
                    Outcome::Read(physical)
                );
                assert_eq!(cpu.read(Register::CNTVOFF_EL2), 0x1234);
            }
        }
        assert_eq!(
            cpu.apple_access(msr(5, ACNTPCT_EL0), 99),
            Outcome::Undefined
        );
        assert_eq!(
            cpu.apple_access(mrs(0, ACNTPCT_EL0), 100),
            Outcome::Read(100)
        );
    }
}

#[test]
fn unqualified_lower_el_physical_alias_permissions_remain_named_for_both_directions() {
    for el in [0, 1] {
        let mut cpu = Cpu::new(0, Pstate::reset(el));
        for control in [0, u64::MAX] {
            cpu.registers.set(Register::CNTKCTL_EL1, 0, control);
            cpu.registers.set(Register::CNTHCTL_EL2, 0, control);
            for access in [mrs(0, ACNTPCT_EL0), msr(0, ACNTPCT_EL0)] {
                assert_eq!(
                    cpu.apple_access(access, 123),
                    Outcome::Refused(Refusal::UnsupportedSetting(
                        "ACNTPCT_EL0 access below EL2 lacks permission qualification"
                    ))
                );
            }
        }
    }
}

#[test]
fn sptms_gxf_finalization_locks_el2s_gxf_and_el1s_enable() {
    let mut cpu = Cpu::new(0, Pstate::reset(2));
    let write = |cpu: &mut Cpu, encoding, value| {
        cpu.state.x[5] = value;
        cpu.apple_access(msr(5, encoding), 0)
    };
    let written = |outcome| matches!(outcome, Outcome::Written(_));
    let refused = |outcome| matches!(outcome, Outcome::Refused(_));
    // SPTM enables GXF at EL2 and enters GL2.
    assert!(written(write(&mut cpu, SPRR_CONFIG_EL1, 1)));
    assert!(written(write(&mut cpu, GXF_CONFIG_EL1, 1)));
    assert!(cpu.guarded_enabled());
    cpu.pstate.guarded = true;
    // At GL2 it clears EL1's GXF and sets the undocumented bits.
    for encoding in [GXF_CONFIG_EL12, GXF_ENTER_EL12, GXF_ABORT_EL12] {
        assert!(written(write(&mut cpu, encoding, 0)));
    }
    assert!(written(write(&mut cpu, GXF_CONFIG_EL1, 0x6f)));
    assert_eq!(cpu.apple.gxf_config, [0, 0x6f]);
    assert_eq!(
        cpu.apple_access(mrs(0, GXF_CONFIG_EL1), 0),
        Outcome::Read(0x6f)
    );
    // Outside GL2, EL2's GXF registers and EL1's enable are locked, but
    // EL1's entry points are not.
    cpu.pstate.guarded = false;
    assert!(refused(write(&mut cpu, GXF_ENTER_EL1, 0x1000)));
    assert!(refused(write(&mut cpu, GXF_CONFIG_EL1, 0x6f)));
    assert!(refused(write(&mut cpu, GXF_CONFIG_EL12, 1)));
    assert!(written(write(&mut cpu, GXF_ENTER_EL12, 0x2000)));
    // Bits nobody sets refuse; EL1 reaches only its own configuration.
    let mut fresh = Cpu::new(0, Pstate::reset(2));
    assert!(refused(write(&mut fresh, GXF_CONFIG_EL1, 1 << 7)));
    let mut el1 = Cpu::new(0, Pstate::reset(1));
    assert_eq!(write(&mut el1, GXF_CONFIG_EL12, 0), Outcome::Undefined);
    assert!(written(write(&mut el1, GXF_CONFIG_EL1, 1)));
    assert_eq!(el1.apple.gxf_config, [1, 0]);
}

#[test]
fn cyc_ovrd_takes_xnus_sleep_and_restart_settings() {
    let mut cpu = Cpu::new(0, Pstate::reset(2));
    let mut write = |value| {
        cpu.state.x[5] = value;
        cpu.apple_access(msr(5, CYC_OVRD), 0)
    };
    let written = |outcome| matches!(outcome, Outcome::Written(_));
    // `ml_arm_sleep`: IRQ and FIQ disabled, then power-down with
    // retention disabled.
    assert!(written(write(0xa0_0000)));
    assert!(written(write(0x3a0_0001)));
    for value in [1 << 30, 1 << 20, 3 << 22, 1 << 24, 1 << 5] {
        assert!(matches!(write(value), Outcome::Refused(_)), "{value:#x}");
    }
    assert_eq!(
        cpu.apple_access(mrs(0, CYC_OVRD), 0),
        Outcome::Read(0x3a0_0001)
    );
    assert_eq!(cpu.wfi_mode(), CYC_OVRD_WFI_POWER_DOWN);
    let both = Lines {
        irq: true,
        fiq: true,
    };
    assert_eq!(cpu.core_lines(both), Lines::default());
    // The restart path clears both modes; `arm64_force_wfi_clock_gate`.
    cpu.state.x[5] = 0x200_0000;
    assert!(written(cpu.apple_access(msr(5, CYC_OVRD), 0)));
    assert_eq!(cpu.core_lines(both), both);
    assert_eq!(cpu.wfi_mode(), CYC_OVRD_WFI_CLOCK_GATE);
}

#[test]
fn sptms_spr_lockdown_commits_at_el2_and_then_holds() {
    use super::super::Chip;
    let write = |cpu: &mut Cpu, encoding, value| {
        cpu.state.x[5] = value;
        cpu.apple_access(msr(5, encoding), 0)
    };
    let written = |outcome| matches!(outcome, Outcome::Written(_));
    // XNU's MSR_RO_CTL_VAL, which the M1 and M2 SPTMs write: every
    // bit but 4, 27 and 47.
    const RO_CTL: u64 = 0xffff_7fff_f7ff_ffef;
    let mut cpu = Cpu::of(Chip::AppleM2, 0, Pstate::reset(2));
    assert_eq!(
        cpu.apple_access(mrs(0, SPR_LOCKDOWN_EL2), 0),
        Outcome::Read(0)
    );
    assert!(written(write(&mut cpu, SPR_LOCKDOWN_EL2, RO_CTL)));
    // Before the commit every group is writable.
    assert!(written(write(&mut cpu, impdef::HID1, 0)));
    assert!(written(write(&mut cpu, SPR_LOCKDOWN_COMMIT, 1)));
    assert_eq!(
        cpu.apple_access(mrs(0, SPR_LOCKDOWN_EL2), 0),
        Outcome::Read(RO_CTL)
    );
    assert_eq!(
        cpu.apple_access(mrs(0, SPR_LOCKDOWN_COMMIT), 0),
        Outcome::Read(1)
    );
    // A committed group's write is UNDEFINED and changes nothing; reads
    // and the groups XNU leaves writable complete.
    cpu.state.x[5] = 0x55;
    assert_eq!(
        cpu.apple_access(msr(5, impdef::HID1), 0),
        Outcome::Undefined
    );
    assert_eq!(cpu.apple_access(mrs(0, impdef::HID1), 0), Outcome::Read(0));
    for encoding in [impdef::HID4, CYC_OVRD, impdef::ACC_OVRD] {
        assert!(written(write(&mut cpu, encoding, 0)), "{encoding:?}");
    }
    cpu.pstate.el = 1;
    assert_eq!(write(&mut cpu, impdef::HID1, 1), Outcome::Undefined);
    cpu.pstate.el = 2;
    // The same values again complete; a change or a release stops.
    assert!(written(write(&mut cpu, SPR_LOCKDOWN_EL2, RO_CTL)));
    assert!(written(write(&mut cpu, SPR_LOCKDOWN_COMMIT, 1)));
    for (encoding, value) in [(SPR_LOCKDOWN_EL2, u64::MAX), (SPR_LOCKDOWN_COMMIT, 0)] {
        assert!(matches!(
            write(&mut cpu, encoding, value),
            Outcome::Refused(_)
        ));
    }
    // The M3 SPTMs add SPR_LOCKDOWN_EL1, all ones, and commit both banks
    // with 3 (sptm.t8122 file 0xbbf70..0xbbf80); bit 2 has no meaning.
    let mut m3 = Cpu::of(Chip::AppleM3, 0, Pstate::reset(2));
    assert!(written(write(&mut m3, SPR_LOCKDOWN_EL2, RO_CTL)));
    assert!(written(write(&mut m3, SPR_LOCKDOWN_EL1, u64::MAX)));
    assert!(written(write(&mut m3, SPR_LOCKDOWN_COMMIT, 3)));
    assert_eq!(m3.apple.spr_lockdown, [RO_CTL, u64::MAX]);
    assert!(matches!(
        write(&mut m3, SPR_LOCKDOWN_EL1, 0),
        Outcome::Refused(_)
    ));
    assert!(matches!(
        write(&mut m3, SPR_LOCKDOWN_COMMIT, 7),
        Outcome::Refused(_)
    ));
    // The M4-and-later `defeatureCore` loosens an uncommitted bank.
    let mut m4 = Cpu::of(Chip::AppleM4, 0, Pstate::reset(2));
    assert!(written(write(&mut m4, SPR_LOCKDOWN_EL2, 0)));
    assert_eq!(
        m4.apple_access(mrs(0, SPR_LOCKDOWN_EL2), 0),
        Outcome::Read(0)
    );
    // Every lockdown register is UNDEFINED below EL2.
    let mut el1 = Cpu::new(0, Pstate::reset(1));
    for encoding in [SPR_LOCKDOWN_EL2, SPR_LOCKDOWN_EL1, SPR_LOCKDOWN_COMMIT] {
        assert_eq!(write(&mut el1, encoding, 0), Outcome::Undefined);
    }
}

/// The M5 cputrace init and deinit polls complete: nothing is in
/// flight and a drain request reads complete.
#[test]
fn the_trace_units_polls_see_no_session() {
    use super::super::Chip;
    let mut cpu = Cpu::of(Chip::AppleM5, 0, Pstate::reset(2));
    let write = |cpu: &mut Cpu, value| {
        cpu.state.x[5] = value;
        assert!(matches!(
            cpu.apple_access(msr(5, impdef::TRACE_VA_C5_7), 0),
            Outcome::Written(_)
        ));
    };
    let read = |cpu: &mut Cpu| match cpu.apple_access(mrs(0, impdef::TRACE_VA_C5_7), 0) {
        Outcome::Read(value) => value,
        other => panic!("{other:?}"),
    };
    // va_init: request a drain, poll bit 36, enable.
    write(&mut cpu, TRACE_DRAIN);
    assert_ne!(read(&mut cpu) & TRACE_DRAINED, 0);
    write(&mut cpu, 3);
    // va_deinit: stop, poll the in-flight count, drain, disable.
    let value = read(&mut cpu) & !2;
    write(&mut cpu, value | TRACE_IN_FLIGHT);
    assert_eq!(read(&mut cpu) & TRACE_IN_FLIGHT, 0);
    let value = read(&mut cpu) | TRACE_DRAIN;
    write(&mut cpu, value);
    assert_ne!(read(&mut cpu) & TRACE_DRAINED, 0);
    let value = read(&mut cpu) & !1;
    write(&mut cpu, value);
    assert_eq!(read(&mut cpu), TRACE_DRAIN | TRACE_DRAINED);
}

/// PMCR1_EL12, the EL1&0 regime's PMCR1, is UNDEFINED below EL2.
#[test]
fn pmcr1_el12_is_undefined_below_el2() {
    let mut cpu = Cpu::new(0, Pstate::reset(1));
    let pmcr1_el12 = super::super::pmu::PMCR1_EL12;
    assert_eq!(cpu.apple_access(mrs(0, pmcr1_el12), 0), Outcome::Undefined);
    assert_eq!(cpu.apple_access(msr(0, pmcr1_el12), 0), Outcome::Undefined);
}

/// A SIQ_CFG_EL1 write tells the board whether the PE now refuses the
/// AIC's IRQs: mode 3 refuses, XNU's 0 and 2 accept.
#[test]
fn siq_cfg_steers_the_aics_irqs() {
    for chip in super::super::Chip::ALL {
        if !impdef::consumed(impdef::SIQ_CFG_EL1, chip) {
            continue;
        }
        for el in [1, 2] {
            let mut cpu = Cpu::of(chip, 0, Pstate::reset(el));
            for (value, refuses) in [(3, true), (2, false), (0, false), (3, true)] {
                cpu.state.x[5] = value;
                assert_eq!(
                    cpu.apple_access(msr(5, impdef::SIQ_CFG_EL1), 0),
                    Outcome::Written(Changes {
                        irq_steering: Some(refuses),
                        reads_kept: true,
                        ..Changes::default()
                    })
                );
                assert_eq!(
                    cpu.apple_access(mrs(0, impdef::SIQ_CFG_EL1), 0),
                    Outcome::Read(value)
                );
            }
        }
    }
}

#[test]
fn siq_cfg_refusals_preserve_held_state_counts_and_irq_effects() {
    let encoding = impdef::SIQ_CFG_EL1;
    let refused = Outcome::Refused(Refusal::UnsupportedSetting(
        "SIQ_CFG_EL1 mode 1 or nonzero upper bits lack source-qualified effects",
    ));
    for chip in super::super::Chip::ALL {
        if !impdef::consumed(encoding, chip) {
            continue;
        }
        for el in [1, 2] {
            let mut cpu = Cpu::of(chip, 0, Pstate::reset(el));
            cpu.state.x[5] = 1;
            let before = cpu.state.clone();
            assert_eq!(cpu.apple_access(msr(5, encoding), 0), refused);
            assert_eq!(cpu.apple.impdef.used().count(), 0);
            assert_eq!(cpu.state, before);
            cpu.state.x[5] = 3;
            assert_eq!(
                cpu.apple_access(msr(5, encoding), 0),
                Outcome::Written(Changes {
                    irq_steering: Some(true),
                    reads_kept: true,
                    ..Changes::default()
                })
            );
            // Test each upper bit with every otherwise-checked mode.
            // Refused writes produce no Changes, including no steering.
            for value in [1, u64::MAX]
                .into_iter()
                .chain((2..64).flat_map(|bit| [0, 2, 3].map(|mode| 1 << bit | mode)))
            {
                cpu.state.x[5] = value;
                let before = cpu.state.clone();
                let reads = cpu.apple.impdef.reads.clone();
                let writes = cpu.apple.impdef.writes.clone();
                assert_eq!(cpu.apple_access(msr(5, encoding), 0), refused);
                assert_eq!(cpu.state, before);
                assert_eq!(cpu.apple.impdef.reads, reads);
                assert_eq!(cpu.apple.impdef.writes, writes);
                assert_eq!(cpu.apple_access(mrs(0, encoding), 0), Outcome::Read(3));
            }
        }
    }
}

#[test]
fn siq_cfg_privilege_checks_precede_validation_and_xzr_clears() {
    let encoding = impdef::SIQ_CFG_EL1;
    for chip in super::super::Chip::ALL {
        if !impdef::consumed(encoding, chip) {
            continue;
        }
        let mut cpu = Cpu::of(chip, 0, Pstate::reset(1));
        cpu.registers.set(Register::HCR_EL2, 0, 1 << HCR_TIDCP);
        cpu.state.x[5] = u64::MAX;
        let before = cpu.state.clone();
        for access in [mrs(0, encoding), msr(5, encoding), msr(31, encoding)] {
            assert_eq!(cpu.apple_access(access, 0), Outcome::Trapped(Some(2)));
        }
        assert_eq!(cpu.state, before);
        assert_eq!(cpu.apple.impdef.used().count(), 0);
        cpu.pstate.el = 0;
        for host in [false, true] {
            cpu.registers.set(
                Register::HCR_EL2,
                0,
                (1 << HCR_TIDCP) | if host { 1 << HCR_TGE } else { 0 },
            );
            for access in [mrs(0, encoding), msr(5, encoding), msr(31, encoding)] {
                assert_eq!(cpu.apple_access(access, 0), Outcome::Undefined);
            }
        }
        assert_eq!(cpu.state, before);
        assert_eq!(cpu.apple.impdef.used().count(), 0);
        cpu.pstate.el = 2;
        assert_eq!(
            cpu.apple_access(msr(31, encoding), 0),
            Outcome::Written(Changes {
                irq_steering: Some(false),
                reads_kept: true,
                ..Changes::default()
            })
        );
        assert_eq!(cpu.apple_access(mrs(0, encoding), 0), Outcome::Read(0));
        assert_eq!(cpu.state, before);
    }
}

/// EL0 programs JCTL_EL0, the per-thread register XNU's user MSR test
/// leaves user-writable, on the chips whose components use it (M2 on).
#[test]
fn el0_reaches_jctl_on_the_chips_that_use_it() {
    use super::super::Chip;
    let mut m2 = Cpu::of(Chip::AppleM2, 0, Pstate::reset(0));
    m2.state.x[5] = 0x4000;
    assert!(matches!(
        m2.apple_access(msr(5, impdef::JCTL_EL0), 0),
        Outcome::Written(Changes {
            unmodeled: Some(_),
            ..
        })
    ));
    assert_eq!(
        m2.apple_access(mrs(0, impdef::JCTL_EL0), 0),
        Outcome::Read(0x4000)
    );
    assert_eq!(m2.apple_access(mrs(0, impdef::HID4), 0), Outcome::Undefined);
    let mut m1 = Cpu::of(Chip::AppleM1, 0, Pstate::reset(0));
    assert_eq!(
        m1.apple_access(mrs(0, impdef::JCTL_EL0), 0),
        Outcome::Undefined
    );
}

/// Bounded original-source integrity fixture. It decodes the original
/// accesses; it does not execute Apple code or check hardware effects.
#[cfg(feature = "std")]
#[test]
#[ignore = "oracle: STELA_TEST_EL0_OS_FILES"]
fn el0_apple_control_original_runtime_pins_the_user_afpcr_accesses() {
    let root = std::path::PathBuf::from(
        std::env::var_os("STELA_TEST_EL0_OS_FILES").expect("original OS-file extraction"),
    );
    let read = |path: &str, length, hash| {
        let path = root.join(path);
        let metadata = std::fs::symlink_metadata(&path).unwrap();
        assert!(
            metadata.file_type().is_file(),
            "original must be a regular file"
        );
        assert_eq!(metadata.len(), length);
        let bytes = std::fs::read(path).unwrap();
        assert_eq!(stela_formats::sha256(&bytes), hash);
        bytes
    };
    let version = read(
        "System/Library/CoreServices/SystemVersion.plist",
        604,
        "bd61e19dd67dd97ce9a27c79a2148d42cc9685a71c6f17ff27d62b819f39f983",
    );
    const BUILD: &[u8] = b"<string>26A434</string>";
    assert!(version.windows(BUILD.len()).any(|bytes| bytes == BUILD));
    let runtime = read(
        "usr/libexec/rosetta/runtime",
        231_808,
        "2adfa4a95ae456b4e44e316c4713d9dfcdd87d421a97b1b4c4fd848c562f3d28",
    );
    for (offset, expected) in [
        (0xdf8, msr(0, impdef::AFPCR_EL0)),
        (0xea4, mrs(2, impdef::AFPCR_EL0)),
        (0x203e0, mrs(25, impdef::AFPCR_EL0)),
    ] {
        let word = u32::from_le_bytes(runtime[offset..offset + 4].try_into().unwrap());
        assert_eq!(decode(word), Some(expected));
    }
}

#[test]
fn el0_apple_control_afpcr_uses_the_existing_pe_model_without_fp_effects() {
    use stela_contracts::Chip;

    let encoding = impdef::AFPCR_EL0;
    let key = encoding.selector();
    for chip in Chip::ALL {
        // All declared family profiles census AFPCR. Its owner is
        // physical PE state, not a new EL0, host or cluster bank.
        assert!(impdef::consumed(encoding, chip));
        assert_eq!(
            impdef::lookup(encoding, chip).unwrap().scope,
            impdef::Scope::Pe
        );
        let mut cpu = Cpu::of(chip, 0, Pstate::reset(0));
        cpu.state.fpcr = 0x0040_0000;
        cpu.state.fpsr = 0x0800_0001;
        cpu.state.x[5] = 0x1234;
        let before = cpu.state.clone();
        let aidr = cpu.read(Register::AIDR_EL1);
        for host in [false, true] {
            cpu.registers.set(
                Register::HCR_EL2,
                0,
                (1 << HCR_TIDCP) | if host { 1 << HCR_TGE } else { 0 },
            );
            assert_eq!(cpu.apple_access(mrs(0, encoding), 0), Outcome::Read(0));
            // XZR writes zero even while X5 holds unsupported controls.
            assert_eq!(
                cpu.apple_access(msr(31, encoding), 0),
                Outcome::Written(STORED)
            );
            let writes = cpu.apple.impdef.writes.clone();
            for value in [1, 2, 3, 1 << 63] {
                cpu.state.x[5] = value;
                let state = cpu.state.clone();
                assert!(matches!(
                    cpu.apple_access(msr(5, encoding), 0),
                    Outcome::Refused(Refusal::UnsupportedSetting(_))
                ));
                assert_eq!(cpu.state, state);
                assert_eq!(cpu.apple.impdef.writes, writes);
            }
            cpu.state.x[5] = before.x[5];
            assert_eq!(cpu.state, before);
            assert_eq!(cpu.read(Register::AIDR_EL1), aidr);
        }
        assert_eq!(cpu.apple.impdef.writes.get(&key), Some(&2));
        cpu.registers.set(Register::HCR_EL2, 0, 0);
        cpu.pstate.el = 1;
        assert_eq!(cpu.apple_access(mrs(0, encoding), 0), Outcome::Read(0));
        assert_eq!(cpu.apple.impdef.reads.get(&key), Some(&3));
        assert_eq!(cpu.apple.impdef.used().count(), 1);
        let mut peer = Cpu::of(chip, 1, Pstate::reset(0));
        peer.apple.impdef.join(&cpu.apple.impdef);
        assert!(peer.apple.impdef.reads.is_empty());
        assert!(peer.apple.impdef.writes.is_empty());
        assert_eq!(
            peer.apple_access(msr(31, encoding), 0),
            Outcome::Written(STORED)
        );
        assert_eq!(peer.apple.impdef.writes.get(&key), Some(&1));
        assert_eq!(cpu.apple.impdef.writes.get(&key), Some(&2));
    }
}

#[test]
fn el0_apple_control_afpcr_keeps_el1_tidcp_ordering_and_el2_access() {
    use stela_contracts::Chip;

    let encoding = impdef::AFPCR_EL0;
    for chip in Chip::ALL {
        let mut cpu = Cpu::of(chip, 0, Pstate::reset(1));
        cpu.state.x[5] = 3;
        let before = cpu.state.clone();
        cpu.registers.set(Register::HCR_EL2, 0, 1 << HCR_TIDCP);
        for access in [mrs(0, encoding), msr(5, encoding), msr(31, encoding)] {
            assert_eq!(cpu.apple_access(access, 0), Outcome::Trapped(Some(2)));
        }
        assert_eq!(cpu.state, before);
        assert_eq!(cpu.apple.impdef.used().count(), 0);
        // At EL2 TIDCP does not trap. The same zero-only physical
        // owner remains in use; no permissive floating-point substitute.
        cpu.pstate.el = 2;
        assert_eq!(cpu.apple_access(mrs(0, encoding), 0), Outcome::Read(0));
        assert!(matches!(
            cpu.apple_access(msr(5, encoding), 0),
            Outcome::Refused(Refusal::UnsupportedSetting(_))
        ));
        assert!(cpu.apple.impdef.writes.is_empty());
        assert_eq!(
            cpu.apple_access(msr(31, encoding), 0),
            Outcome::Written(STORED)
        );
        cpu.pstate.el = 1;
        cpu.registers.set(Register::HCR_EL2, 0, 0);
        assert_eq!(cpu.apple_access(mrs(0, encoding), 0), Outcome::Read(0));
        assert_eq!(cpu.apple.impdef.used().count(), 1);
        assert_eq!(cpu.state, before);
    }
}

#[test]
fn el0_apple_control_trace_permissions_remain_explicitly_unqualified() {
    use stela_contracts::Chip;

    let encoding = impdef::TRACE_CLEARED;
    for chip in Chip::ALL {
        let consumed = matches!(chip, Chip::AppleM1 | Chip::AppleM2);
        assert_eq!(impdef::consumed(encoding, chip), consumed);
        let mut cpu = Cpu::of(chip, 0, Pstate::reset(0));
        for host in [false, true] {
            cpu.registers
                .set(Register::HCR_EL2, 0, if host { 1 << HCR_TGE } else { 0 });
            for value in [0, 1] {
                cpu.state.x[5] = value;
                let before = cpu.state.clone();
                for access in [mrs(0, encoding), msr(5, encoding)] {
                    let expected = if consumed {
                        Outcome::Refused(Refusal::UnsupportedSetting(
                            "TRACE_CLEARED EL0 permission and trace side effects lack qualification",
                        ))
                    } else {
                        Outcome::Undefined
                    };
                    assert_eq!(cpu.apple_access(access, 0), expected);
                }
                assert_eq!(cpu.state, before);
                assert_eq!(cpu.apple.impdef.used().count(), 0);
            }
        }
        cpu.pstate.el = 1;
        cpu.registers.set(Register::HCR_EL2, 0, 1 << HCR_TIDCP);
        for access in [mrs(0, encoding), msr(31, encoding)] {
            assert_eq!(cpu.apple_access(access, 0), Outcome::Trapped(Some(2)));
        }
        assert_eq!(cpu.apple.impdef.used().count(), 0);
        cpu.registers.set(Register::HCR_EL2, 0, 0);
        // Original M1/M2 XNU can still clear the register at a
        // context switch. Non-censused chips keep the census inference.
        let expected = if consumed {
            Outcome::Written(STORED)
        } else {
            Outcome::Undefined
        };
        assert_eq!(cpu.apple_access(msr(31, encoding), 0), expected);
        if consumed {
            let writes = cpu.apple.impdef.writes.clone();
            cpu.state.x[5] = 1;
            assert!(matches!(
                cpu.apple_access(msr(5, encoding), 0),
                Outcome::Refused(Refusal::UnsupportedSetting(_))
            ));
            assert_eq!(cpu.apple.impdef.writes, writes);
        }
    }
}

/// CRn 11 reaches `machine::impdef` where an entry owns it, and stays
/// UNDEFINED on a chip whose components never access it.
#[test]
fn crn_11_encodings_reach_their_impdef_entries() {
    use super::super::Chip;
    let mut m5 = Cpu::of(Chip::AppleM5, 0, Pstate::reset(2));
    assert_eq!(
        m5.apple_access(mrs(0, impdef::PL2_ERR_STS), 0),
        Outcome::Read(0)
    );
    m5.state.x[5] = 0x57f;
    assert!(matches!(
        m5.apple_access(msr(5, impdef::PL2_ERR_STS), 0),
        Outcome::Written(_)
    ));
    let mut m2 = Cpu::of(Chip::AppleM2, 0, Pstate::reset(2));
    assert_eq!(
        m2.apple_access(mrs(0, impdef::PL2_ERR_STS), 0),
        Outcome::Undefined
    );
}

/// Exercise decoded MRS/MSR, including privilege routing before validation:
/// no rejected access can corrupt the PE's predictor configuration.
#[test]
fn objc_prediction_accesses_preserve_state_and_trap_before_writing() {
    use impdef::{BP_OBJC_ADR_EL1, BP_OBJC_CTL_EL1};
    use stela_contracts::Chip;

    for chip in Chip::ALL {
        let mut cpu = Cpu::of(chip, 0, Pstate::reset(1));
        for (encoding, value, unknown) in [
            (BP_OBJC_ADR_EL1, 0x00ab_cdef_1234_5678, 1 << 56),
            (BP_OBJC_CTL_EL1, 0x001a_bcde_f83b_107f, 1 << 22),
        ] {
            cpu.state.x[5] = value;
            if chip == Chip::AppleM1 {
                assert_eq!(cpu.apple_access(msr(5, encoding), 0), Outcome::Undefined);
                assert_eq!(cpu.apple_access(mrs(0, encoding), 0), Outcome::Undefined);
                assert_eq!(cpu.apple.impdef.used().count(), 0);
                continue;
            }
            assert_eq!(
                cpu.apple_access(msr(5, encoding), 0),
                Outcome::Written(STORED)
            );
            assert_eq!(cpu.apple_access(mrs(0, encoding), 0), Outcome::Read(value));
            let writes = cpu.apple.impdef.writes.clone();
            cpu.state.x[5] = value | unknown;
            assert!(matches!(
                cpu.apple_access(msr(5, encoding), 0),
                Outcome::Refused(Refusal::UnsupportedSetting(_))
            ));
            assert_eq!(cpu.apple.impdef.writes, writes);
            assert_eq!(cpu.apple_access(mrs(0, encoding), 0), Outcome::Read(value));

            cpu.registers.set(Register::HCR_EL2, 0, 1 << HCR_TIDCP);
            assert_eq!(
                cpu.apple_access(msr(5, encoding), 0),
                Outcome::Trapped(Some(2))
            );
            assert_eq!(
                cpu.apple_access(mrs(0, encoding), 0),
                Outcome::Trapped(Some(2))
            );
            assert_eq!(cpu.apple.impdef.writes, writes);
            cpu.registers.set(Register::HCR_EL2, 0, 0);
            cpu.pstate.el = 0;
            assert_eq!(cpu.apple_access(msr(5, encoding), 0), Outcome::Undefined);
            assert_eq!(cpu.apple_access(mrs(0, encoding), 0), Outcome::Undefined);
            assert_eq!(cpu.apple.impdef.writes, writes);
            // EL2's SPTM uses the same physical PE state. Writing XZR
            // clears it, even though X5 still contains an invalid value.
            cpu.pstate.el = 2;
            assert_eq!(cpu.apple_access(mrs(0, encoding), 0), Outcome::Read(value));
            assert_eq!(
                cpu.apple_access(msr(31, encoding), 0),
                Outcome::Written(STORED)
            );
            assert_eq!(cpu.apple_access(mrs(31, encoding), 0), Outcome::Read(0));
            cpu.pstate.el = 1;
        }
    }
}

#[test]
fn apple_registers_follow_the_vhe_names_and_tidcp() {
    let mut cpu = Cpu::new(0, Pstate::reset(2));
    let write = |cpu: &mut Cpu, encoding, value| {
        cpu.state.x[5] = value;
        cpu.apple_access(msr(5, encoding), 0)
    };
    // At EL2, APCTL_EL1 names APCTL_EL2 and APCTL_EL12 names EL1's.
    write(&mut cpu, APCTL_EL1, 2);
    write(&mut cpu, APCTL_EL12, 1);
    assert_eq!(cpu.apple.apctl, [1, 2]);
    // GXF entry points per level.
    write(&mut cpu, GXF_ENTER_EL1, 0x2000);
    write(&mut cpu, GXF_ENTER_EL12, 0x1000);
    assert_eq!(
        (cpu.apple.guarded[0].enter, cpu.apple.guarded[1].enter),
        (0x1000, 0x2000)
    );
    // GL12 names reach GL1's bank from EL2; TPIDR_GL2 is GL2's.
    write(&mut cpu, VBAR_GL12, 0x8800);
    write(&mut cpu, TPIDR_GL2, 7);
    assert_eq!(cpu.apple.guarded[0].vbar, 0x8800);
    assert_eq!(cpu.apple.guarded[1].tpidr, 7);
    // SPRR permissions per regime.
    write(&mut cpu, SPRR_CONFIG_EL1, 1);
    write(&mut cpu, SPRR_PERM_EL1, 0x22);
    write(&mut cpu, SPRR_PERM_EL12, 0x11);
    assert_eq!(
        (cpu.apple.sprr_perm[PERM_EL1], cpu.apple.sprr_perm[PERM_EL2]),
        (0x11, 0x22)
    );
    // EL2 names are UNDEFINED at EL1, and TIDCP traps the space to EL2.
    let mut el1 = Cpu::new(0, Pstate::reset(1));
    assert_eq!(el1.apple_access(mrs(0, APCTL_EL12), 0), Outcome::Undefined);
    el1.registers.set(Register::HCR_EL2, 0, 1 << HCR_TIDCP);
    assert_eq!(
        el1.apple_access(mrs(0, APCTL_EL1), 0),
        Outcome::Trapped(Some(2))
    );
    // TGE puts EL0 in the host.
    el1.registers.set(Register::HCR_EL2, 0, 1 << HCR_TGE);
    assert_eq!(el1.regime(), TranslationRegime::El10);
    // VMSA_LOCK_EL1 locks the writer's TTBR1 and is not released.
    assert!(matches!(
        write(&mut cpu, VMSA_LOCK_EL1, VMSA_LOCK_TTBR1),
        Outcome::Written(_)
    ));
    assert!(vmsa_locked(&cpu, Register::TTBR1_EL2, 0).is_some());
    assert!(vmsa_locked(&cpu, Register::TTBR1_EL1, 0).is_none());
    assert!(matches!(
        write(&mut cpu, VMSA_LOCK_EL1, 0),
        Outcome::Refused(_)
    ));
    // Undefined architectural encodings outside CRn 11/15.
    assert_eq!(
        cpu.apple_access(mrs(0, Encoding::new(3, 0, 13, 7, 7)), 0),
        Outcome::Undefined
    );
}

/// GL bank writes only store; a control write does not.
#[test]
fn guarded_bank_writes_report_that_they_only_store() {
    let mut cpu = Cpu::new(0, Pstate::reset(2));
    let stored = |cpu: &mut Cpu, encoding, value| {
        cpu.state.x[5] = value;
        match cpu.apple_access(msr(5, encoding), 0) {
            Outcome::Written(changes) => changes.stored,
            outcome => panic!("{outcome:?}"),
        }
    };
    assert!(!stored(&mut cpu, SPRR_CONFIG_EL1, 1));
    assert!(!stored(&mut cpu, GXF_CONFIG_EL1, 1));
    cpu.pstate.guarded = true;
    for encoding in [ELR_GL12, SPSR_GL12, GXF_ENTER_EL12] {
        assert!(stored(&mut cpu, encoding, 0x1000));
    }
}

#[test]
fn sprr_locks_bind_outside_the_guarded_level() {
    let mut cpu = Cpu::new(0, Pstate::reset(2));
    let write = |cpu: &mut Cpu, encoding, value| {
        cpu.state.x[5] = value;
        cpu.apple_access(msr(5, encoding), 0)
    };
    write(&mut cpu, SPRR_CONFIG_EL1, 0xfb);
    assert!(matches!(
        write(&mut cpu, SPRR_PERM_EL1, 1),
        Outcome::Refused(_)
    ));
    assert!(matches!(
        write(&mut cpu, SPRR_FIELDS_0, 1),
        Outcome::Refused(_)
    ));
    cpu.pstate.guarded = true;
    assert!(matches!(
        write(&mut cpu, SPRR_PERM_EL1, 1),
        Outcome::Written(_)
    ));
    assert!(matches!(
        write(&mut cpu, SPRR_CONFIG_EL1, 0xff),
        Outcome::Written(_)
    ));
    assert!(matches!(
        write(&mut cpu, SPRR_FIELDS_0, 1 << 18),
        Outcome::Written(_)
    ));
    assert_eq!(cpu.apple.sprr_fields[0], 1 << 18);
    // SPTM's lockdown: fields 2 and 9 of S3_6_C15_C3_1 keep those EL
    // fields of SPRR_PERM_EL1 writable outside GL.
    write(&mut cpu, SPRR_FIELDS_1, 0x40010);
    write(&mut cpu, SPRR_PERM_EL1, 0x2020_a52a_302a_bae6);
    cpu.pstate.guarded = false;
    for (value, allowed) in [
        (0x2020_a52a_302a_bbe6, true),  // index 2: EL r-- to rw-
        (0x2020_a53a_302a_bbe6, true),  // index 9: EL r-- to rw-
        (0x2020_a53a_302a_bfe6, false), // index 2's GL field
        (0x2020_a53a_302a_bbe7, false), // index 0
    ] {
        assert_eq!(
            matches!(write(&mut cpu, SPRR_PERM_EL1, value), Outcome::Written(_)),
            allowed,
            "{value:#x}"
        );
    }
    // SPRR_PERM_EL0 follows S3_6_C15_C3_0, here field 9 = 1 only: from
    // SPTM's default value, the TPRO toggle (index 9 r-- to rw-) passes.
    const DEFAULT_EL0: u64 = 0x2010_0020_3010_0000;
    const TPRO_RW: u64 = DEFAULT_EL0 | 1 << 36;
    const JIT_RW: u64 = DEFAULT_EL0 | 2 << 20;
    cpu.pstate.guarded = true;
    write(&mut cpu, SPRR_PERM_EL0, DEFAULT_EL0);
    cpu.pstate.guarded = false;
    assert!(matches!(
        write(&mut cpu, SPRR_PERM_EL0, TPRO_RW),
        Outcome::Written(_)
    ));
    assert!(matches!(
        write(&mut cpu, SPRR_PERM_EL0, TPRO_RW & !(2 << 36)),
        Outcome::Refused(_)
    ));
    assert!(matches!(
        write(&mut cpu, SPRR_PERM_EL0, JIT_RW),
        Outcome::Refused(_)
    ));
    // SPTM's value for a JIT space adds field 5 = 2: the JIT write toggle
    // (index 5 r-x to rw-) passes both ways, index 5's other EL bit and
    // its GL field stay locked.
    cpu.pstate.guarded = true;
    write(&mut cpu, SPRR_FIELDS_0, 2 << 10 | 1 << 18);
    cpu.pstate.guarded = false;
    for (value, allowed) in [
        (JIT_RW, true),
        (DEFAULT_EL0, true),
        (DEFAULT_EL0 & !(1 << 20), false),
        (DEFAULT_EL0 | 4 << 20, false),
    ] {
        assert_eq!(
            matches!(write(&mut cpu, SPRR_PERM_EL0, value), Outcome::Written(_)),
            allowed,
            "{value:#x}"
        );
    }
}

#[test]
fn checked_pmu_registers_trap_before_mutation_and_stay_undefined_at_el0() {
    for chip in super::super::Chip::ALL {
        let mut cpu = Cpu::of(chip, 0, Pstate::reset(2));
        cpu.pmu = Pmu::for_chip(chip, 1_000_000_000, 24_000_000);
        let mut encodings = alloc::vec![super::super::pmu::PMC[1]];
        if chip == super::super::Chip::AppleM6 {
            encodings.push(Encoding::new(3, 0, 11, 13, 2));
        }
        for encoding in encodings {
            cpu.pstate.el = 2;
            cpu.state.x[5] = 123;
            assert!(matches!(
                cpu.apple_access(msr(5, encoding), 0),
                Outcome::Written(_)
            ));
            cpu.pstate.el = 1;
            cpu.registers.set(Register::HCR_EL2, 0, 1 << HCR_TIDCP);
            cpu.state.x[5] = 456;
            assert_eq!(
                cpu.apple_access(msr(5, encoding), 0),
                Outcome::Trapped(Some(2))
            );
            assert_eq!(
                cpu.apple_access(mrs(0, encoding), 0),
                Outcome::Trapped(Some(2))
            );
            cpu.pstate.el = 0;
            assert_eq!(cpu.apple_access(msr(5, encoding), 0), Outcome::Undefined);
            assert_eq!(cpu.apple_access(mrs(0, encoding), 0), Outcome::Undefined);
            cpu.pstate.el = 2;
            assert_eq!(cpu.apple_access(mrs(0, encoding), 0), Outcome::Read(123));
        }
    }
}

#[test]
fn m4_ctrr_names_reach_the_vhe_copies_and_iboots_controls() {
    use super::super::{Chip, ctrr};
    let crn11 = |crm, op2| Encoding::new(3, 0, 11, crm, op2);
    let (lower_el1, lower_el12, lower_el2) = (crn11(0, 0), crn11(0, 4), crn11(1, 0));
    let handoff = ctrr::CoreLocks {
        locked: alloc::vec![ctrr::Locked {
            region: ctrr::LockRegion::CtrrC,
            first: 0x100_06fb_8000,
            last: 0x100_0710_bfff,
        }],
        legacy_locked: true,
        tag_offset: 0,
    };
    let mut cpu = Cpu::of(Chip::AppleM4, 0, Pstate::reset(2));
    cpu.ctrr.replace(ctrr::Cluster::handed_off(&handoff));
    cpu.ctrr_core = ctrr::Core::handed_off(&handoff);
    let write = |cpu: &mut Cpu, encoding, value| {
        cpu.state.x[5] = value;
        cpu.apple_access(msr(5, encoding), 0)
    };
    // SPTM's EL1-name write at EL2 lands in EL2's copy, which the kernel
    // reads by its EL2 name; EL1's copy stays clear.
    assert!(matches!(
        write(&mut cpu, lower_el1, 0x100_06fb_8000),
        Outcome::Written(Changes {
            translation: true,
            unmodeled: None,
            ..
        })
    ));
    assert_eq!(
        cpu.apple_access(mrs(0, lower_el2), 0),
        Outcome::Read(0x100_06fb_8000)
    );
    assert_eq!(cpu.apple_access(mrs(0, lower_el12), 0), Outcome::Read(0));
    // iBoot's locked per-cluster copy and CTRR-A/B controls.
    assert_eq!(
        cpu.apple_access(mrs(0, crn11(6, 7)), 0),
        Outcome::Read(0x100_0710_b000)
    );
    assert_eq!(
        cpu.apple_access(mrs(0, crn11(8, 2)), 0),
        Outcome::Read(ctrr::CTRR_HANDOFF)
    );
    let ctrr_a = Encoding::new(3, 4, 15, 2, 5);
    assert_eq!(
        cpu.apple_access(mrs(0, ctrr_a), 0),
        Outcome::Read(ctrr::CTRR_HANDOFF)
    );
    assert!(matches!(write(&mut cpu, ctrr_a, 0), Outcome::Refused(_)));
    // A CTXR enable completes with its effect reported.
    assert!(matches!(
        write(&mut cpu, crn11(5, 2), ctrr::CTXR_HANDOFF[0]),
        Outcome::Written(Changes {
            unmodeled: Some(_),
            ..
        })
    ));
    // At EL1 the EL1 name reaches EL1's copy; the EL12, EL2 and
    // per-cluster names are UNDEFINED.
    cpu.pstate.el = 1;
    assert_eq!(cpu.apple_access(mrs(0, lower_el1), 0), Outcome::Read(0));
    for encoding in [
        lower_el12,
        lower_el2,
        crn11(6, 6),
        Encoding::new(3, 4, 15, 11, 4),
    ] {
        assert_eq!(cpu.apple_access(mrs(0, encoding), 0), Outcome::Undefined);
    }
}

#[test]
fn tag_offset_reads_iboots_locked_value_at_el2() {
    use super::super::{Chip, ctrr, identification, tag_storage::TAG_OFFSET_EL2};
    const LOCKED: u64 = 0x8000_0101_e000_0000;
    for chip in Chip::ALL {
        let mut cpu = Cpu::of(chip, 0, Pstate::reset(2));
        cpu.core_locks = alloc::sync::Arc::new(ctrr::CoreLocks {
            tag_offset: LOCKED,
            ..ctrr::CoreLocks::default()
        });
        let outcome = cpu.apple_access(mrs(0, TAG_OFFSET_EL2), 0);
        if !identification::memory_tagging(chip) {
            // The census rules the other chips' encoding.
            assert_ne!(outcome, Outcome::Read(LOCKED), "{}", chip.name());
            continue;
        }
        assert_eq!(outcome, Outcome::Read(LOCKED));
        cpu.state.x[5] = LOCKED;
        assert!(matches!(
            cpu.apple_access(msr(5, TAG_OFFSET_EL2), 0),
            Outcome::Written(_)
        ));
        cpu.state.x[5] = 0;
        assert!(matches!(
            cpu.apple_access(msr(5, TAG_OFFSET_EL2), 0),
            Outcome::Refused(_)
        ));
        cpu.pstate.el = 1;
        assert_eq!(
            cpu.apple_access(mrs(0, TAG_OFFSET_EL2), 0),
            Outcome::Undefined
        );
    }
}

#[test]
fn every_crn11_encoding_the_m4_to_m6_sptms_and_kernels_access_reaches_ctrr() {
    use super::super::{Chip, ctrr};
    for chip in [Chip::AppleM4, Chip::AppleM5, Chip::AppleM6] {
        for crm in 0..16 {
            for op2 in 0..8 {
                let encoding = Encoding::new(3, 0, 11, crm, op2);
                // CRm 9 up holds TAG_OFFSET_EL2 (`machine::tag_storage`)
                // and the M6 kernel's other registers.
                if !impdef::consumed(encoding, chip) || crm >= 9 {
                    continue;
                }
                assert!(
                    ctrr::decode(chip, encoding).is_some(),
                    "{} {encoding:?}",
                    chip.name()
                );
            }
        }
    }
}

#[test]
fn legacy_actlr_alias_allocates_only_on_consuming_chips() {
    for chip in super::super::Chip::ALL {
        let mut cpu = Cpu::of(chip, 0, Pstate::reset(2));
        for operation in [mrs(0, impdef::ACTLR_EL12), msr(0, impdef::ACTLR_EL12)] {
            let outcome = cpu.apple_access(operation, 0);
            if impdef::consumed(impdef::ACTLR_EL12, chip) {
                assert_eq!(
                    outcome,
                    Outcome::View(encodings::ACTLR_EL12, agt::selected(&cpu))
                );
            } else {
                assert_eq!(outcome, Outcome::Undefined);
            }
            assert!(cpu.apple.impdef.used().next().is_none());
        }
    }
}

/// Every IMPLEMENTATION DEFINED encoding a chip's censused components
/// access has an owner: a modeled register here, CTRR, AMX, the core
/// PMU, the board's fast IPIs, the Apple generic timer or a
/// `machine::impdef` entry with a class other than `Unsupported`.
#[test]
fn every_encoding_the_components_access_has_a_model() {
    use crate::machine::fast_ipi as ipi;
    for chip in super::super::Chip::ALL {
        let mut owned = 0;
        for crn in [11, 15] {
            for selector in 0u16..1 << 10 {
                let field = |shift: u16, mask: u16| ((selector >> shift) & mask) as u8;
                let (op1, crm, op2) = (field(7, 7), field(3, 0xf), field(0, 7));
                let encoding = Encoding::new(3, op1, crn, crm, op2);
                if !impdef::consumed(encoding, chip)
                    || Pmu::owns_on(chip, encoding)
                    || ipi::Register::decode(encoding).is_some()
                    || super::super::timer::agt::decode(encoding, chip).is_some()
                {
                    continue;
                }
                owned += 1;
                let entry = impdef::lookup(encoding, chip);
                assert!(
                    entry.is_none_or(|entry| entry.class != impdef::Class::Unsupported),
                    "{} {encoding:?}",
                    chip.name()
                );
                // A modeled register may be UNDEFINED in this state
                // (SPRR's permissions while SPRR is disabled), but none
                // lacks a model.
                let mut cpu = Cpu::of(chip, 0, Pstate::reset(2));
                let outcome = cpu.apple_access(mrs(0, encoding), 0);
                assert!(
                    !matches!(
                        outcome,
                        Outcome::Refused(Refusal::Unimplemented)
                            | Outcome::Refused(Refusal::UnsupportedSetting(
                                "CRn=11 IMPLEMENTATION DEFINED register without a model"
                            ))
                    ),
                    "{} {encoding:?}",
                    chip.name()
                );
            }
        }
        assert!(owned > 150, "{}: {owned}", chip.name());
    }
}
