use super::*;
use crate::topology::{Cpuid, Msr, Snapshot};

#[test]
fn boot_elapsed_millis_uses_the_current_entry_epoch() {
    let hz = 2_294_686_699;
    let warm_entry = hz * 5700;
    let sample = warm_entry + hz / 4 + 1;
    assert_eq!(boot_elapsed_millis(warm_entry, sample, hz), 250);
    assert_eq!(boot_elapsed_millis(warm_entry, warm_entry - 1, hz), 0);
    assert_eq!(boot_elapsed_millis(warm_entry, sample, 0), 0);
}

fn fixture() -> Snapshot {
    let records = [
        Cpuid {
            leaf: 0,
            eax: 0x16,
            ebx: u32::from_le_bytes(*b"Genu"),
            edx: u32::from_le_bytes(*b"ineI"),
            ecx: u32::from_le_bytes(*b"ntel"),
            ..Cpuid::default()
        },
        Cpuid {
            leaf: 1,
            ebx: 7 << 24,
            edx: 1 << 4,
            ..Cpuid::default()
        },
        Cpuid {
            leaf: 5,
            ..Cpuid::default()
        },
        Cpuid {
            leaf: 6,
            eax: 1 << 2,
            ..Cpuid::default()
        },
        Cpuid {
            leaf: 7,
            ebx: 1 << 1,
            ..Cpuid::default()
        },
        Cpuid {
            leaf: 0xb,
            ..Cpuid::default()
        },
        Cpuid {
            leaf: 0x15,
            eax: 1,
            ebx: 125,
            ecx: 24_000_000,
            ..Cpuid::default()
        },
        Cpuid {
            leaf: 0x16,
            eax: 3000,
            ..Cpuid::default()
        },
        Cpuid {
            leaf: 0x8000_0000,
            eax: 0x8000_0007,
            ..Cpuid::default()
        },
        Cpuid {
            leaf: 0x8000_0001,
            edx: 1 << 27,
            ..Cpuid::default()
        },
        Cpuid {
            leaf: 0x8000_0007,
            edx: 1 << 8,
            ..Cpuid::default()
        },
    ];
    let mut snapshot = Snapshot::default();
    snapshot.records[..records.len()].copy_from_slice(&records);
    snapshot.count = records.len() as u32;
    snapshot.msrs[0] = Msr {
        index: 0x3b,
        value: 17,
    };
    snapshot.msr_count = 1;
    snapshot
}
fn record(snapshot: &mut Snapshot, leaf: u32) -> &mut Cpuid {
    snapshot
        .records
        .iter_mut()
        .find(|r| r.leaf == leaf)
        .unwrap()
}
#[test]
fn physical_bsp_feature_data_distinguishes_rate_reliability_and_deadline() {
    let mut snapshot = fixture();
    let policy = counter_policy(&snapshot, 1).unwrap();
    assert_eq!(policy.apic_id, 7);
    assert_eq!(policy.serialization, Serialization::Rdtscp);
    assert_eq!(
        policy.reliability,
        Reliability::AdjustmentMonitored { observed: 17 }
    );
    assert_eq!(policy.deadline, Deadline::Arat);
    let RateRequirement::HardwareCrystalRatio(rate) = policy.rate else {
        panic!("hardware crystal ratio")
    };
    assert_eq!(rate.hz(), Ok(3_000_000_000));
    assert_eq!(rate.nanoseconds(3_000_000_000), Ok(1_000_000_000));
    record(&mut snapshot, 0x15).ecx = 0;
    assert_eq!(
        counter_policy(&snapshot, 1).unwrap().rate,
        RateRequirement::RequiresObservedCalibration,
        "leaf16 nominal MHz is not an observed runtime rate"
    );
    record(&mut snapshot, 7).ebx = 0;
    record(&mut snapshot, 6).eax = 0;
    record(&mut snapshot, 0x8000_0001).edx = 0;
    let policy = counter_policy(&snapshot, 1).unwrap();
    assert_eq!(
        policy.reliability,
        Reliability::RequiresOwnedReferenceWatchdog
    );
    assert_eq!(policy.deadline, Deadline::RequiresOwnedBroadcast);
    assert_eq!(policy.serialization, Serialization::SerializingCpuid);
    assert_eq!(
        counter_policy(&fixture(), 5).unwrap().reliability,
        Reliability::RequiresOwnedReferenceWatchdog
    );
}
#[test]
fn missing_invariant_or_real_control_observation_never_becomes_a_policy_shortcut() {
    let mut snapshot = fixture();
    record(&mut snapshot, 1).edx = 0;
    assert_eq!(counter_policy(&snapshot, 1), Err(Error::TscUnavailable));
    let mut snapshot = fixture();
    record(&mut snapshot, 0x8000_0007).edx = 0;
    assert_eq!(
        counter_policy(&snapshot, 1),
        Err(Error::InvariantUnavailable)
    );
    let mut snapshot = fixture();
    record(&mut snapshot, 1).ecx = 1 << 31;
    assert_eq!(
        counter_policy(&snapshot, 1),
        Err(Error::HypervisorContractRequired)
    );
    let mut snapshot = fixture();
    snapshot.msr_count = 0;
    assert_eq!(counter_policy(&snapshot, 1), Err(Error::MissingMsr(0x3b)));
    let mut snapshot = fixture();
    snapshot.msrs[1] = snapshot.msrs[0];
    snapshot.msr_count = 2;
    assert_eq!(counter_policy(&snapshot, 1), Err(Error::DuplicateMsr(0x3b)));
    assert_eq!(
        counter_policy(&fixture(), 0),
        Err(Error::InvalidPackageCount)
    );
    let mut snapshot = fixture();
    snapshot.records[10] = snapshot.records[0];
    snapshot.count = 11;
    assert_eq!(counter_policy(&snapshot, 1), Err(Error::DuplicateCpuid(0)));
    let mut snapshot = fixture();
    snapshot.count = u32::MAX;
    assert_eq!(counter_policy(&snapshot, 1), Err(Error::MalformedCapture));
}
#[test]
fn pc10_policy_requires_package_observation_and_rejects_enabled_limit() {
    let mut snapshot = fixture();
    assert_eq!(
        hpet_power_constraint(&snapshot),
        Ok(HpetPowerConstraint::NoPc10Indication)
    );
    record(&mut snapshot, 1).ecx = 1 << 3;
    record(&mut snapshot, 5).ecx = 3;
    record(&mut snapshot, 5).edx = 1 << 28;
    assert_eq!(
        hpet_power_constraint(&snapshot),
        Err(Error::MissingMsr(0xe2))
    );
    snapshot.msrs[1] = Msr {
        index: 0xe2,
        value: 7,
    };
    snapshot.msr_count = 2;
    assert_eq!(
        hpet_power_constraint(&snapshot),
        Ok(HpetPowerConstraint::RequiresStablePackageLimit { observed: 7 })
    );
    snapshot.msrs[1].value = 8;
    assert_eq!(hpet_power_constraint(&snapshot), Err(Error::Pc10StopsHpet));
    record(&mut snapshot, 5).ecx = 1;
    assert_eq!(
        hpet_power_constraint(&snapshot),
        Ok(HpetPowerConstraint::NoPc10Indication)
    );
}
fn second() -> (Sample, Sample) {
    (
        Sample {
            tsc_before: 100,
            reference: 0,
            tsc_after: 110,
        },
        Sample {
            tsc_before: 3_000_000_100,
            reference: 3_579_545,
            tsc_after: 3_000_000_110,
        },
    )
}
#[test]
fn reference_brackets_preserve_uncertainty_and_cross_sample_consistency() {
    let (first, last) = second();
    let window = RateWindow::from_samples(
        first,
        last,
        0xffff_ffff,
        ReferenceScale::Hertz(3_579_545),
        0x20000,
    )
    .unwrap();
    assert_eq!(window.minimum_hz(), 2_999_999_990);
    assert_eq!(window.maximum_hz(), 3_000_000_010);
    assert!(window.contains(3_000_000_000));
    assert!(!window.contains(2_999_999_989));
    let hpet = RateWindow::from_samples(
        first,
        Sample {
            reference: 100_000_000,
            ..last
        },
        u64::MAX,
        ReferenceScale::PeriodFemtoseconds(10_000_000),
        0x20000,
    )
    .unwrap();
    assert_eq!(window, hpet);
    assert_eq!(window.intersect(hpet), Ok(window));
    let bad = RateWindow::from_samples(
        first,
        Sample {
            tsc_before: 2_000_000_100,
            tsc_after: 2_000_000_110,
            ..last
        },
        0xffff_ffff,
        ReferenceScale::Hertz(3_579_545),
        0x20000,
    )
    .unwrap();
    assert_eq!(window.intersect(bad), Err(Error::InconsistentRate));
}
#[test]
fn disturbed_backwards_zero_and_ambiguous_counter_data_fail_explicitly() {
    let (first, last) = second();
    assert_eq!(
        RateWindow::from_samples(
            first,
            Sample {
                tsc_before: 99,
                tsc_after: 100,
                ..last
            },
            0xffff_ffff,
            ReferenceScale::Hertz(3_579_545),
            0x20000
        ),
        Err(Error::Backwards)
    );
    assert_eq!(
        RateWindow::from_samples(
            first,
            Sample {
                tsc_after: last.tsc_before + 0x20000,
                ..last
            },
            0xffff_ffff,
            ReferenceScale::Hertz(3_579_545),
            0x20000
        ),
        Err(Error::Disturbed)
    );
    assert_eq!(
        RateWindow::from_samples(
            first,
            Sample {
                reference: 0,
                ..last
            },
            0xffff_ffff,
            ReferenceScale::Hertz(3_579_545),
            0x20000
        ),
        Err(Error::NoProgress)
    );
    assert_eq!(
        RateWindow::from_samples(first, last, 0xffff_ffff, ReferenceScale::Hertz(0), 0x20000),
        Err(Error::InvalidRate)
    );
    assert_eq!(wrapped_delta(0x00ff_ff00, 0x100, 0x00ff_ffff), Ok(512));
    assert_eq!(wrapped_delta(0, 1, 0xff), Err(Error::InvalidMask));
    assert_eq!(
        wrapped_delta(1 << 24, 0, 0x00ff_ffff),
        Err(Error::InvalidMask)
    );
    assert_eq!(RationalRate::new(1, 0), Err(Error::InvalidRate));
    assert_eq!(RationalRate::new(0, 1), Err(Error::InvalidRate));
    assert_eq!(
        RationalRate::new(1, u64::MAX)
            .unwrap()
            .nanoseconds(u64::MAX),
        Err(Error::Overflow)
    );
}

#[test]
fn missing_declared_topology_never_invents_a_legacy_cpu_identity() {
    let mut snapshot = fixture();
    record(&mut snapshot, 0xb).leaf = 0x1234;
    assert_eq!(counter_policy(&snapshot, 1), Err(Error::MalformedCapture));
}
