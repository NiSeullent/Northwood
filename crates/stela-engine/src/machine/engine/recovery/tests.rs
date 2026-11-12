use super::super::Fetched;
use super::super::tests::{BASE, machine, make_pending};

#[test]
fn legacy_first_fault_qualification_is_cached_and_reports_dispatcher_cost() {
    let program = [0x91000400u32, 0xf9400043, 0xd4000001];
    let input: Vec<u8> = program.into_iter().flat_map(u32::to_le_bytes).collect();
    let mut machine = machine(&program);
    let Ok(Fetched::Block(index)) = machine.fetch_block(BASE, false) else {
        panic!("actual emission");
    };
    make_pending(&mut machine, index, &input);
    let mut owner = machine.code.blocks.recovery().remove(&index).unwrap();
    let started = std::time::Instant::now();
    let first = owner.qualify(BASE, None).unwrap().rows();
    let cold = started.elapsed();
    assert!(
        first
            .iter()
            .any(|row| row.site.kind() == stela_compiler::block::RecoverySiteKind::GuestAccess)
    );
    let started = std::time::Instant::now();
    for _ in 0..32 {
        assert_eq!(owner.qualify(BASE, None).unwrap().rows(), first);
    }
    let cached = started.elapsed();
    assert_eq!(
        owner.qualification_attempts, 1,
        "only the first actual fault re-emits the plan"
    );
    eprintln!(
        "dispatcher recovery qualification: source={} native={} cold_ns={} cached_average_ns={} (current host; no native guest entered)",
        input.len(),
        owner.code_bytes(),
        cold.as_nanos(),
        cached.as_nanos() / 32
    );
}

use super::super::{Offer, Prepared, prepare};
use super::{Emission, StoredOrigin, Template};
use crate::translations::{Digest, Record, translator_id};
use alloc::{sync::Arc, vec::Vec};
use stela_compiler::block::{FaultRecovery, RecoveryBinding};

fn store_window() -> (
    Arc<super::Preparation>,
    Vec<u8>,
    stela_compiler::block::CompiledBlock,
) {
    let preparation = Arc::new(prepare::test_preparation());
    let source: Vec<u8> = [0xf900_0041u32; 256]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect();
    let Prepared::Block {
        compiled, input, ..
    } = preparation.prepare(&source, BASE, source.len()).unwrap()
    else {
        panic!("actual store-window compilation");
    };
    assert_eq!(input, source.len());
    assert!(!compiled.fault_recovery.is_empty());
    (preparation, source, compiled)
}

#[test]
fn a_compiled_owner_keeps_only_its_packed_rows() {
    let (preparation, source, compiled) = store_window();
    let expected = compiled.fault_recovery.clone();
    let unpacked = expected.len() * core::mem::size_of::<FaultRecovery>()
        + expected
            .iter()
            .map(|row| row.bindings.len() * core::mem::size_of::<RecoveryBinding>())
            .sum::<usize>();
    let native_bytes = compiled.native.x86_64_code.len();
    let mut owner =
        Template::compiled(Arc::new(Emission::new(source, compiled, preparation))).installed();
    assert_eq!(
        owner.code_bytes(),
        native_bytes,
        "the code's image, not a copy"
    );
    assert_eq!(owner.qualify(BASE, None).unwrap().rows(), expected);
    assert!(
        owner.test_packed_bytes() * 3 < unpacked,
        "{} packed bytes of {unpacked}",
        owner.test_packed_bytes()
    );
}

#[test]
fn exact_lazy_qualification_replaces_portable_bank_with_only_minted_rows() {
    let preparation = Arc::new(prepare::test_preparation());
    let mut page = [0u8; crate::translations::PAGE_BYTES];
    let source: Vec<u8> = [0x9100_0400u32, 0xf940_0043, 0xd400_0001]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect();
    page[..source.len()].copy_from_slice(&source);
    let key = Digest::of(&page);
    let context = preparation.context(BASE);
    let offer = Offer::compile(key, BASE, BASE, context, source, preparation.clone()).unwrap();
    let expected = &offer.block().fault_recovery;
    let code_bytes = offer.block().native.x86_64_code.len();
    assert!(!expected.is_empty());
    let translator = translator_id();
    let mut record = Record::new(translator, key);
    record.insert(offer.prove().unwrap());
    let record = Record::decode(&record.encode(), &translator, &key).unwrap();
    let origin = Arc::new(StoredOrigin {
        key,
        context,
        page: BASE,
        preparation,
        kept: None,
    });
    let mut owner = {
        let entry = record.find(0, &context).unwrap();
        let mut relocated = crate::translations::Relocated::default();
        assert!(entry.relocate(BASE, offer.input(), &mut relocated));
        Template::stored(entry, &relocated, origin).installed()
    };
    assert_eq!(owner.stored_key(), Some(key));
    assert_eq!(owner.code_bytes(), code_bytes);
    assert_eq!(
        &owner.qualify(BASE, Some(&record)).unwrap().rows(),
        expected
    );
    assert_eq!(owner.stored_key(), None);
    assert_eq!(owner.code_bytes(), code_bytes);
    assert_eq!(owner.qualification_attempts, 1);
    assert_eq!(&owner.qualify(BASE, None).unwrap().rows(), expected);
    assert_eq!(owner.qualification_attempts, 1);
}

/// An entry proven during the boot qualifies from itself, at the page it
/// was rebased to: the rows of the block compiled at that page.
#[test]
fn a_kept_entry_qualifies_at_its_new_page_without_a_record() {
    let preparation = Arc::new(prepare::test_preparation());
    let source: Vec<u8> = [0x1000_0040u32, 0xf940_0043, 0xd400_0001]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect();
    let key = Digest([7; 32]);
    let context = preparation.context(BASE);
    let moved = BASE + 0x5_4000;
    let at = |page| {
        Offer::compile(
            key,
            page,
            page,
            context,
            source.clone(),
            preparation.clone(),
        )
        .unwrap()
    };
    let entry = Arc::new(at(BASE).prove().unwrap());
    let expected = at(moved).block().fault_recovery.clone();
    assert!(!expected.is_empty());
    let origin = Arc::new(StoredOrigin {
        key,
        context,
        page: moved,
        preparation: preparation.clone(),
        kept: Some(Arc::clone(&entry)),
    });
    let mut relocated = crate::translations::Relocated::default();
    assert!(entry.relocate(moved, &source, &mut relocated));
    assert_eq!(relocated.code, at(moved).block().native.x86_64_code);
    let mut owner = Template::stored(&entry, &relocated, origin).installed();
    // No pack record is looked up for it.
    assert_eq!(owner.stored_key(), None);
    assert_eq!(&owner.qualify(moved, None).unwrap().rows(), &expected);
}

#[test]
fn fault_rows_survive_packing() {
    let program = [0x9100_0400u32, 0xf900_0020, 0xf940_0043, 0x9100_0c64].repeat(64);
    let mut description = prepare::test_preparation().description();
    description.checks = stela_compiler::x86_64::MemoryChecks::system(true, false, true, false);
    let preparation = Arc::new(super::Preparation::from_description(&description).unwrap());
    let source: Vec<u8> = program.into_iter().flat_map(u32::to_le_bytes).collect();
    let Prepared::Block {
        compiled, input, ..
    } = preparation.prepare(&source, BASE, source.len()).unwrap()
    else {
        panic!("actual load compilation");
    };
    assert_eq!(input, source.len());
    assert!(!compiled.fault_recovery.is_empty());
    let expected = compiled.fault_recovery.clone();
    let mut owner =
        Template::compiled(Arc::new(Emission::new(source, compiled, preparation))).installed();
    assert_eq!(owner.qualify(BASE, None).unwrap().rows(), expected);
}
