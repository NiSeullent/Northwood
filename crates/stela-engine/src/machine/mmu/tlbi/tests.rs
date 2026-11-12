use super::*;
use crate::machine::system::TlbiAttr;

fn tlbi(kind: TlbiKind, regime: Named, level: TlbiLevel, operand: u64) -> Tlbi {
    Tlbi {
        kind,
        regime,
        vmid: Some(5),
        broadcast: Broadcast::NSH,
        level,
        attr: TlbiAttr::AllAttr,
        operand,
    }
}

const GUEST: Context = Context {
    el: 1,
    el2: true,
    e2h: true,
    tge: false,
    fb: false,
};
const HOST: Context = Context {
    el: 2,
    tge: true,
    ..GUEST
};

/// A 16 KiB level-3 page of EL1&0 at `address`, VMID 5, ASID 7.
fn page(address: u64) -> Entry {
    Entry {
        regime: Regime::El10,
        vmid: Some(5),
        asid: Some(7),
        address,
        size_bits: 14,
        granule_bits: 14,
        level: 3,
        stage1: true,
        stage2: true,
        table: false,
    }
}

#[test]
fn only_a_lower_half_tlbi_with_an_asid_is_private() {
    const ASID_7: u64 = 7 << 48;
    let scope = |kind, regime, operand| {
        Scope::new(&tlbi(kind, regime, TlbiLevel::Any, operand), HOST).private()
    };
    assert_eq!(
        scope(TlbiKind::ASID, Named::EL10, ASID_7),
        Some(Private {
            regime: Regime::El20,
            asid: 7,
            addresses: None,
        })
    );
    assert_eq!(
        scope(TlbiKind::VA, Named::EL10, ASID_7 | 0x1234),
        Some(Private {
            regime: Regime::El20,
            asid: 7,
            addresses: Some((0x123_4000, 0x123_4000)),
        })
    );
    let upper = ASID_7 | (VA_OPERAND & !0xfff);
    assert_eq!(scope(TlbiKind::VA, Named::EL10, upper), None);
    assert_eq!(scope(TlbiKind::VAA, Named::EL10, 0x1234), None);
    assert_eq!(scope(TlbiKind::VMALL, Named::EL10, 0), None);
    let el2 = tlbi(TlbiKind::VA, Named::EL2, TlbiLevel::Any, 0x1234);
    let context = Context { e2h: false, ..HOST };
    assert_eq!(Scope::new(&el2, context).private(), None);
}

fn covers(tlbi: Tlbi, context: Context, entry: Entry) -> bool {
    Scope::new(&tlbi, context).covers(&entry)
}

#[test]
fn everywhere_names_va_forms_by_address_of_any_regime_or_asid() {
    let everywhere = |kind, regime, operand| {
        Scope::new(&tlbi(kind, regime, TlbiLevel::Any, operand), HOST).everywhere()
    };
    // VAE1 of ASID 9 at a kernel VA: the address alone, canonical.
    assert_eq!(
        everywhere(
            TlbiKind::VA,
            Named::EL10,
            9 << 48 | 0xffff_fe00_0000_4000 >> 12
        ),
        Everywhere::Addresses(Reach::Containing(0xffff_fe00_0000_4000))
    );
    assert_eq!(
        everywhere(TlbiKind::VAA, Named::EL2, 0x4000 >> 12),
        Everywhere::Addresses(Reach::Containing(0x4000))
    );
    assert_eq!(
        everywhere(TlbiKind::ASID, Named::EL10, 9 << 48),
        Everywhere::NonGlobal(9)
    );
    assert_eq!(everywhere(TlbiKind::VMALL, Named::EL10, 0), Everywhere::All);
    assert_eq!(everywhere(TlbiKind::ALL, Named::EL2, 0), Everywhere::All);
    // A range form with TG 0b00 names nothing.
    assert_eq!(
        everywhere(TlbiKind::RVA, Named::EL10, 0),
        Everywhere::Addresses(Reach::Nothing)
    );
}

#[test]
fn va_forms_match_address_asid_level_and_ttl() {
    super::super::test_translation_chip();
    let va = 0x1_2345_c000_u64;
    let entry = page(va);
    let vae1 = |asid: u64, ttl: u64, va: u64, level| {
        tlbi(
            TlbiKind::VA,
            Named::EL10,
            level,
            asid << 48 | ttl << 44 | va >> 12,
        )
    };
    assert!(covers(vae1(7, 0, va + 0x123, TlbiLevel::Any), GUEST, entry));
    assert!(!covers(vae1(8, 0, va, TlbiLevel::Any), GUEST, entry));
    assert!(!covers(
        vae1(7, 0, va + 0x4000, TlbiLevel::Any),
        GUEST,
        entry
    ));
    // A global entry matches any ASID; VAAE1 ignores the ASID.
    let global = Entry {
        asid: None,
        ..entry
    };
    assert!(covers(vae1(8, 0, va, TlbiLevel::Any), GUEST, global));
    let vaae1 = tlbi(
        TlbiKind::VAA,
        Named::EL10,
        TlbiLevel::Any,
        9 << 48 | va >> 12,
    );
    assert!(covers(vaae1, GUEST, entry));
    // A 32 MiB block matches any VA inside it.
    let block = Entry {
        size_bits: 25,
        level: 2,
        ..entry
    };
    assert!(covers(
        vae1(7, 0, va | 0x1ff_0000, TlbiLevel::Any),
        GUEST,
        block
    ));
    // Last level only skips walk-cache entries.
    let table = Entry {
        table: true,
        ..block
    };
    assert!(!covers(vae1(7, 0, va, TlbiLevel::Last), GUEST, table));
    assert!(covers(vae1(7, 0, va, TlbiLevel::Any), GUEST, table));
    // TTL 0b1011 (16 KiB, level 3) matches the page and not the block;
    // 0b0111 (4 KiB) matches neither; the reserved 0b1000 is no hint.
    assert!(covers(vae1(7, 0b1011, va, TlbiLevel::Any), GUEST, entry));
    assert!(!covers(vae1(7, 0b1011, va, TlbiLevel::Any), GUEST, block));
    assert!(!covers(vae1(7, 0b0111, va, TlbiLevel::Any), GUEST, entry));
    assert!(covers(vae1(7, 0b1000, va, TlbiLevel::Any), GUEST, block));
    // Another VMID, another regime, or a stage-2-only entry is untouched.
    assert!(!covers(
        vae1(7, 0, va, TlbiLevel::Any),
        GUEST,
        Entry {
            vmid: Some(6),
            ..entry
        }
    ));
    assert!(!covers(
        vae1(7, 0, va, TlbiLevel::Any),
        GUEST,
        Entry {
            regime: Regime::El20,
            ..entry
        }
    ));
    assert!(!covers(
        vae1(7, 0, va, TlbiLevel::Any),
        GUEST,
        Entry {
            stage1: false,
            ..entry
        }
    ));
    // The upper half compares VA[55:12].
    let high = 0xffff_fff0_0000_4000_u64;
    assert!(covers(
        vae1(7, 0, high & ((1 << 56) - 1), TlbiLevel::Any),
        GUEST,
        page(high)
    ));
}

#[test]
fn host_forms_name_the_el2_and_0_regime_without_a_vmid() {
    super::super::test_translation_chip();
    let host_entry = Entry {
        regime: Regime::El20,
        vmid: None,
        ..page(0x8000)
    };
    // VAE1, ASIDE1 and VMALLE1 in the host (E2H and TGE) act on EL2&0.
    let vae1 = tlbi(TlbiKind::VA, Named::EL10, TlbiLevel::Any, 7 << 48 | 0x8);
    assert!(covers(vae1, HOST, host_entry));
    assert!(!covers(vae1, HOST, page(0x8000)));
    let aside1 = tlbi(TlbiKind::ASID, Named::EL10, TlbiLevel::Any, 7 << 48);
    assert!(covers(aside1, HOST, host_entry));
    let vmalle1 = tlbi(TlbiKind::VMALL, Named::EL10, TlbiLevel::Any, 0);
    assert!(covers(vmalle1, HOST, host_entry));
    // VAE2 with E2H acts on EL2&0 whatever TGE holds.
    let vae2 = tlbi(TlbiKind::VA, Named::EL2, TlbiLevel::Any, 7 << 48 | 0x8);
    assert!(covers(vae2, GUEST, host_entry));
    // ASIDE1 removes only non-global entries of its ASID.
    assert!(!covers(
        aside1,
        HOST,
        Entry {
            asid: None,
            ..host_entry
        }
    ));
    assert!(!covers(
        aside1,
        HOST,
        Entry {
            asid: Some(8),
            ..host_entry
        }
    ));
}

#[test]
fn all_and_vmall_forms_follow_regime_and_vmid() {
    super::super::test_translation_chip();
    let entry = page(0x4000);
    let stage2_only = Entry {
        stage1: false,
        asid: None,
        ..entry
    };
    let vmalle1 = tlbi(TlbiKind::VMALL, Named::EL10, TlbiLevel::Any, 0);
    assert!(covers(vmalle1, GUEST, entry));
    assert!(!covers(vmalle1, GUEST, stage2_only));
    assert!(!covers(
        vmalle1,
        GUEST,
        Entry {
            vmid: Some(6),
            ..entry
        }
    ));
    let vmalls12 = tlbi(TlbiKind::VMALLS12, Named::EL10, TlbiLevel::Any, 0);
    assert!(covers(vmalls12, GUEST, entry) && covers(vmalls12, GUEST, stage2_only));
    assert!(!covers(
        vmalls12,
        GUEST,
        Entry {
            vmid: Some(6),
            ..entry
        }
    ));
    // ALLE1 removes every EL1&0 entry of every VMID; ALLE2 every EL2 and
    // EL2&0 entry.
    let alle1 = tlbi(TlbiKind::ALL, Named::EL10, TlbiLevel::Any, 0);
    assert!(covers(
        alle1,
        GUEST,
        Entry {
            vmid: Some(6),
            ..entry
        }
    ));
    let alle2 = tlbi(TlbiKind::ALL, Named::EL2, TlbiLevel::Any, 0);
    assert!(covers(
        alle2,
        GUEST,
        Entry {
            regime: Regime::El20,
            ..entry
        }
    ));
    assert!(covers(
        alle2,
        GUEST,
        Entry {
            regime: Regime::El2,
            ..entry
        }
    ));
    assert!(!covers(alle2, GUEST, entry));
}

#[test]
fn range_forms_decode_tg_scale_num_and_ttl() {
    super::super::test_translation_chip();
    // RVAE1 over 16 KiB pages: BaseADDR in 16 KiB units, NUM 1, SCALE 0:
    // (1 + 1) * 2^1 = 4 pages from VA 0x10_0000.
    let base = 0x10_0000_u64 >> 14;
    let rvae1 = |tg: u64, ttl: u64, scale: u64, num: u64, base: u64| {
        tlbi(
            TlbiKind::RVA,
            Named::EL10,
            TlbiLevel::Any,
            7 << 48 | tg << 46 | scale << 44 | num << 39 | ttl << 37 | base,
        )
    };
    let range = rvae1(0b10, 0, 0, 1, base);
    assert!(covers(range, GUEST, page(0x10_0000)));
    assert!(covers(range, GUEST, page(0x10_c000)));
    assert!(!covers(range, GUEST, page(0x11_0000)));
    assert!(!covers(range, GUEST, page(0xf_c000)));
    // A 32 MiB block overlapping the range is removed.
    let block = Entry {
        address: 0,
        size_bits: 25,
        level: 2,
        ..page(0)
    };
    assert!(covers(range, GUEST, block));
    // TTL 0b11 limits it to level-3 entries; TG must equal the entry's.
    assert!(!covers(rvae1(0b10, 0b11, 0, 1, base), GUEST, block));
    assert!(covers(
        rvae1(0b10, 0b11, 0, 1, base),
        GUEST,
        page(0x10_0000)
    ));
    assert!(!covers(rvae1(0b01, 0, 0, 1, 0x100), GUEST, page(0x10_0000)));
    // TG 0b00 is invalid and removes nothing.
    assert!(!covers(rvae1(0, 0, 0, 1, base), GUEST, page(0x10_0000)));
    // SCALE 1, NUM 0: 2^6 pages; BaseADDR's top bit extends the sign.
    let wide = rvae1(0b10, 0, 1, 0, base);
    assert!(covers(wide, GUEST, page(0x10_0000 + 63 * 0x4000)));
    assert!(!covers(wide, GUEST, page(0x10_0000 + 64 * 0x4000)));
    let high = 0xffff_ffff_ffff_0000_u64;
    let upper = rvae1(0b10, 0, 0, 0, (high >> 14) & ((1 << 37) - 1));
    assert!(covers(upper, GUEST, page(high & ((1 << 56) - 1))));
    assert_eq!(
        super::range(2 << 46 | (1 << 36)),
        Some((2, 0xfffc_0000_0000_0000, 0xfffc_0000_0000_8000))
    );
}

/// Every entry of the current context a TLBI covers is in its reach,
/// over random forms, operands, contexts and entries of each size.
#[test]
fn the_reach_holds_every_current_entry_the_scope_covers() {
    super::super::test_translation_chip();
    let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let kinds = [
        TlbiKind::ALL,
        TlbiKind::VMALL,
        TlbiKind::VMALLS12,
        TlbiKind::ASID,
        TlbiKind::VA,
        TlbiKind::VAA,
        TlbiKind::RVA,
        TlbiKind::RVAA,
        TlbiKind::IPAS2,
        TlbiKind::RIPAS2,
    ];
    let named = [Named::EL10, Named::EL20, Named::EL2];
    let regimes = [Regime::El10, Regime::El20, Regime::El2];
    let sizes = [
        (12, 12, 3),
        (14, 14, 3),
        (16, 16, 3),
        (21, 12, 2),
        (25, 14, 2),
        (36, 14, 1),
    ];
    let (mut covered, mut reaches) = ([0; 3], [0; 4]);
    for _ in 0..400_000 {
        let random = next();
        let context = Context {
            el: 1 + (random & 1) as u8,
            el2: true,
            e2h: true,
            tge: random & 2 != 0,
            fb: random & 4 != 0,
        };
        let regime = regimes[(random >> 3) as usize % 3];
        let current = Current {
            regime,
            asid: (regime != Regime::El2).then_some((random >> 8) as u16 & 3),
        };
        let (size_bits, granule_bits, level) = sizes[(random >> 16) as usize % sizes.len()];
        // Addresses near each other, in either half.
        let address = (next() & 0x3f_ffff_ffff) << 12 & !((1u64 << size_bits) - 1);
        let address = if random & 1 << 24 != 0 {
            address | 0xffff_ff00_0000_0000
        } else {
            address
        };
        let entry = Entry {
            regime,
            vmid: (regime == Regime::El10).then_some(5),
            asid: current.asid.filter(|_| random & 1 << 25 != 0),
            address,
            size_bits,
            granule_bits,
            level,
            stage1: true,
            stage2: random & 1 << 26 != 0,
            table: random & 1 << 27 != 0,
        };
        let kind = kinds[(random >> 32) as usize % kinds.len()];
        // The operand aims inside the entry or near it: its ASID and TTL
        // fields drawn small, range fields at random.
        let spread = if random & 1 << 28 != 0 {
            (1 << size_bits) - 1
        } else {
            0x3ff_ffff
        };
        let near = entry.address.wrapping_add(next() & spread) & ADDRESS_BITS;
        let operand = match kind {
            TlbiKind::RVA | TlbiKind::RVAA | TlbiKind::RIPAS2 => {
                let tg = next() % 4;
                let bits = granule(tg as u8).unwrap_or(12);
                // Ranges start up to 1 MiB before the entry.
                let below = entry.address.wrapping_sub(next() & 0xf_ffff);
                let base = (canonical(below & ADDRESS_BITS) >> bits) & RANGE_BASE;
                (random >> 40 & 3) << 48 | tg << 46 | (next() & 0x7f) << 37 | base
            }
            _ => (random >> 40 & 3) << 48 | (random >> 44 & 0xf) << 44 | near >> 12,
        };
        let tlbi = Tlbi {
            kind,
            regime: named[(random >> 48) as usize % 3],
            vmid: Some(5),
            broadcast: Broadcast::NSH,
            level: if random & 1 << 52 != 0 {
                TlbiLevel::Last
            } else {
                TlbiLevel::Any
            },
            attr: TlbiAttr::AllAttr,
            operand,
        };
        let scope = Scope::new(&tlbi, context);
        let reach = scope.reach(&current);
        reaches[match reach {
            Reach::Nothing => 0,
            Reach::Containing(_) => 1,
            Reach::Overlapping { .. } => 2,
            Reach::All => 3,
        }] += 1;
        if !scope.covers(&entry) {
            continue;
        }
        covered[match reach {
            Reach::Containing(_) => 1,
            Reach::Overlapping { .. } => 2,
            _ => 0,
        }] += 1;
        let last = entry.address | ((1 << size_bits) - 1);
        let held = match reach {
            Reach::Nothing => false,
            Reach::Containing(address) => (entry.address..=last).contains(&address),
            Reach::Overlapping { first, last: end } => first <= last && entry.address <= end,
            Reach::All => true,
        };
        assert!(held, "{tlbi:?} {context:?} {current:?} {entry:?} {reach:?}");
    }
    assert!(covered.iter().all(|count| *count > 1000), "{covered:?}");
    assert!(reaches.iter().all(|count| *count > 1000), "{reaches:?}");
}

#[test]
fn lower_only_reaches_name_no_upper_half_address() {
    assert!(Reach::Nothing.lower_only());
    assert!(Reach::Containing(0x1_0000).lower_only());
    assert!(!Reach::Containing(0xffff_fe00_0000_0000).lower_only());
    let range = |first, last| Reach::Overlapping { first, last };
    assert!(range(0x1000, 0x7f_ffff_ffff_ffff).lower_only());
    assert!(!range(0x1000, 0xff80_0000_0000_0000).lower_only());
    assert!(!Reach::All.lower_only());
}

#[test]
fn reaches_name_the_address_the_range_or_nothing_of_other_contexts() {
    super::super::test_translation_chip();
    let host = Current {
        regime: Regime::El20,
        asid: Some(7),
    };
    let reach = |kind, regime, operand| {
        Scope::new(&tlbi(kind, regime, TlbiLevel::Any, operand), HOST).reach(&host)
    };
    let high = 0xffff_fe00_0123_4000_u64;
    assert_eq!(
        reach(
            TlbiKind::VA,
            Named::EL10,
            9 << 48 | (high & ADDRESS_BITS) >> 12
        ),
        Reach::Containing(high)
    );
    assert_eq!(
        reach(TlbiKind::VAA, Named::EL10, 0x8),
        Reach::Containing(0x8000)
    );
    // ASIDE1 of another ASID reaches no entry: the global ones stay.
    assert_eq!(reach(TlbiKind::ASID, Named::EL10, 8 << 48), Reach::Nothing);
    assert_eq!(reach(TlbiKind::ASID, Named::EL10, 7 << 48), Reach::All);
    // Four 16 KiB pages from 0x10_0000.
    assert_eq!(
        reach(
            TlbiKind::RVA,
            Named::EL10,
            2 << 46 | 1 << 39 | 0x10_0000 >> 14
        ),
        Reach::Overlapping {
            first: 0x10_0000,
            last: 0x10_ffff
        }
    );
    // From EL2 without TGE, EL1 forms act on the guest's EL1&0 regime.
    let guest = Scope::new(
        &tlbi(TlbiKind::VA, Named::EL10, TlbiLevel::Any, 0x8),
        Context { tge: false, ..HOST },
    );
    assert_eq!(guest.reach(&host), Reach::Nothing);
    // A range of TG 0b00 is invalid.
    assert_eq!(reach(TlbiKind::RVA, Named::EL10, 0x8), Reach::Nothing);
    assert_eq!(reach(TlbiKind::ALL, Named::EL2, 0), Reach::All);
    assert_eq!(reach(TlbiKind::ALL, Named::EL10, 0), Reach::Nothing);
}

#[test]
fn ipa_forms_remove_only_stage_two_entries_and_fb_broadcasts() {
    super::super::test_translation_chip();
    let stage2_only = Entry {
        stage1: false,
        asid: None,
        size_bits: 12,
        granule_bits: 12,
        ..page(0x8_0000_1000)
    };
    let ipas2e1 = tlbi(
        TlbiKind::IPAS2,
        Named::EL10,
        TlbiLevel::Any,
        0x8_0000_1000 >> 12,
    );
    assert!(covers(ipas2e1, HOST, stage2_only));
    assert!(!covers(ipas2e1, HOST, page(0x8_0000_1000)));
    assert!(!covers(
        ipas2e1,
        HOST,
        Entry {
            vmid: Some(6),
            ..stage2_only
        }
    ));
    let ripas2e1 = tlbi(
        TlbiKind::RIPAS2,
        Named::EL10,
        TlbiLevel::Any,
        0b01 << 46 | 3 << 39 | 0x8_0000_0000 >> 12,
    );
    assert!(covers(ripas2e1, HOST, stage2_only));
    assert!(!covers(
        ripas2e1,
        HOST,
        Entry {
            address: 0x8_0000_8000,
            ..stage2_only
        }
    ));
    // HCR_EL2.FB makes VAE1 at EL1 Inner Shareable; IS forms stay so.
    let fb = Context { fb: true, ..GUEST };
    let vae1 = tlbi(TlbiKind::VA, Named::EL10, TlbiLevel::Any, 0);
    assert_eq!(Scope::new(&vae1, fb).broadcast, Broadcast::ISH);
    assert_eq!(Scope::new(&vae1, GUEST).broadcast, Broadcast::NSH);
    assert_eq!(
        Scope::new(&vae1, Context { fb: true, ..HOST }).broadcast,
        Broadcast::NSH
    );
}

/// A form without an address that reaches nothing of a context's regime
/// ([`Scope::reach`]) removes none of that regime's entries
/// ([`Scope::covers`]), global or of any ASID or VMID: what lets a cache of
/// many contexts keep that regime's.
#[test]
fn a_form_without_an_address_reaching_nothing_of_a_regime_removes_none_of_it() {
    use TlbiKind::{ALL, IPAS2, RIPAS2, VMALL, VMALLS12};
    for kind in [ALL, VMALL, VMALLS12, IPAS2, RIPAS2] {
        for named in [Named::EL10, Named::EL20, Named::EL2] {
            for context in [GUEST, HOST] {
                let scope = Scope::new(
                    &tlbi(kind, named, TlbiLevel::Any, 1 << 46 | 1 << 39),
                    context,
                );
                for regime in [Regime::El10, Regime::El20, Regime::El2] {
                    let current = Current {
                        regime,
                        asid: Some(7),
                    };
                    if scope.reach(&current) != Reach::Nothing {
                        continue;
                    }
                    for (vmid, asid, stage2) in [
                        (Some(5), Some(7), true),
                        (None, None, false),
                        (Some(6), Some(1), true),
                        (None, Some(7), false),
                    ] {
                        let entry = Entry {
                            regime,
                            vmid,
                            asid,
                            stage2,
                            ..page(0x10_0000)
                        };
                        assert!(
                            !scope.covers(&entry),
                            "{kind:?} {named:?} {regime:?} {entry:?}"
                        );
                    }
                }
            }
        }
    }
}

/// [`Scope::merged`] joins VA and range forms of one regime, VMID and
/// domain in one half, near each other, into a range removing at least
/// both; it refuses other forms, regimes and halves, and forms too far
/// apart for the gap.
#[test]
fn merged_scopes_join_near_va_forms_of_one_regime_and_half() {
    let scope =
        |kind, named, operand| Scope::new(&tlbi(kind, named, TlbiLevel::Last, operand), GUEST);
    let (a, b) = (
        scope(TlbiKind::VA, Named::EL10, 7 << 48 | 0x100),
        scope(TlbiKind::VA, Named::EL10, 7 << 48 | 0x103),
    );
    let merged = a.merged(b, 0x4000).expect("near pages of one ASID");
    assert_eq!(merged.named_asid(), Some(7));
    assert_eq!(merged.span(), Some((0x10_0000, 0x10_3000)));
    for address in [0x10_0000, 0x10_1000, 0x10_3fff] {
        for asid in [Some(7), None] {
            assert!(merged.covers(&Entry {
                asid,
                address,
                ..page(0)
            }));
        }
    }
    assert!(!merged.covers(&Entry {
        asid: Some(8),
        ..page(0x10_0000)
    }));
    assert!(!merged.covers(&Entry {
        asid: Some(7),
        ..page(0x10_4000)
    }));
    // Another ASID makes it every ASID's.
    let other = scope(TlbiKind::VA, Named::EL10, 8 << 48 | 0x101);
    assert_eq!(merged.merged(other, 0x4000).unwrap().named_asid(), None);
    // Too far for the gap, another regime, another half, an ASID form.
    assert!(
        a.merged(scope(TlbiKind::VA, Named::EL10, 7 << 48 | 0x200), 0x4000)
            .is_none()
    );
    assert!(
        a.merged(scope(TlbiKind::VA, Named::EL2, 0x101), u64::MAX)
            .is_none()
    );
    let upper = (0xffff_fe00_0010_0000_u64 & ((1 << 56) - 1)) >> 12;
    assert!(
        a.merged(scope(TlbiKind::VAA, Named::EL10, upper), u64::MAX)
            .is_none()
    );
    assert!(
        a.merged(scope(TlbiKind::ASID, Named::EL10, 7 << 48), u64::MAX)
            .is_none()
    );
}
