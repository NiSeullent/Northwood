use super::super::{
    FETCH_PAGE, FIRST_WINDOW_BYTES, Fetched, INSTRUCTION, tests::BASE, tests::machine,
};
use super::*;
use crate::translations;
use alloc::collections::BTreeMap;

/// A store in memory: records by key, the blocks offered to it, and
/// the blocks it has prepared by key, PC and context.
#[derive(Default)]
pub(crate) struct MemoryStore {
    pub(crate) records: BTreeMap<Digest, Arc<Record>>,
    pub(crate) offers: Vec<Offer>,
    pub(crate) prepared: BTreeMap<(Digest, u64, Digest), Arc<SharedTranslation>>,
}

impl Translations for MemoryStore {
    fn record(&mut self, key: &Digest) -> Option<Arc<Record>> {
        self.records.get(key).cloned()
    }
    fn prepared(
        &mut self,
        key: &Digest,
        pc: u64,
        context: &Digest,
    ) -> Option<Arc<SharedTranslation>> {
        self.prepared.get(&(*key, pc, *context)).cloned()
    }
    fn offer(&mut self, offer: Offer) {
        self.offers.push(offer);
    }
}

#[test]
fn sealed_shared_handoff_rejects_other_source_pc_and_profile_before_install() {
    let words = [0x91000400, 0xd4000001];
    let mut sender = machine(&words);
    sender.host.store = Some(MemoryStore::default());
    assert!(matches!(
        sender.fetch_block(BASE, false),
        Ok(Fetched::Block(_))
    ));
    let genuine = sender.host.store.as_mut().unwrap().offers.pop().unwrap();
    assert!(genuine.clone().shared().matches(
        &genuine.key,
        genuine.pc,
        &genuine.context,
        genuine.input()
    ));

    let mut different = machine(&[0x91000800, 0xd4000001]);
    different.host.store = Some(MemoryStore::default());
    assert!(matches!(
        different.fetch_block(BASE, false),
        Ok(Fetched::Block(_))
    ));
    let other_source = different
        .host
        .store
        .as_mut()
        .unwrap()
        .offers
        .pop()
        .unwrap()
        .shared();
    let other_pc = Offer::compile(
        genuine.key,
        genuine.page,
        genuine.pc + 4,
        genuine.context,
        genuine.input().to_vec(),
        Arc::clone(genuine.preparation()),
    )
    .unwrap()
    .shared();
    let mut description = genuine.preparation().description();
    description.fp = !description.fp;
    let other_preparation = Arc::new(Preparation::from_description(&description).unwrap());
    let other_context = other_preparation.context(genuine.page);
    let other_profile = Offer::compile(
        genuine.key,
        genuine.page,
        genuine.pc,
        other_context,
        genuine.input().to_vec(),
        other_preparation,
    )
    .unwrap()
    .shared();
    assert_eq!(
        other_profile.block().native.x86_64_code,
        genuine.block().native.x86_64_code,
        "identical scalar bytes do not prove profile identity"
    );
    assert!(
        Offer::compile(
            genuine.key,
            genuine.page,
            genuine.pc,
            other_context,
            genuine.input().to_vec(),
            Arc::clone(genuine.preparation())
        )
        .is_none()
    );

    for wrong in [other_source, other_pc, other_profile] {
        assert!(!wrong.matches(&genuine.key, genuine.pc, &genuine.context, genuine.input()));
        let mut store = MemoryStore::default();
        store
            .prepared
            .insert((genuine.key, genuine.pc, genuine.context), wrong);
        let mut receiver = machine(&words);
        receiver.host.store = Some(store);
        assert!(matches!(
            receiver.fetch_block(BASE, false),
            Ok(Fetched::Block(_))
        ));
        assert_eq!(
            (receiver.statistics.shared, receiver.statistics.compiled),
            (0, 1)
        );
        // The block compiled anew, then the SVC's thunk.
        assert_eq!(
            receiver.host.installed.first(),
            Some(&genuine.block().native.x86_64_code),
            "wrong handoff never reaches installation"
        );
        assert_eq!(receiver.host.installed.len(), 2);
    }
}

#[test]
fn cached_blocks_do_not_cross_presented_instruction_features() {
    use {
        crate::machine::identification, crate::machine::registers::encodings::ID_AA64ISAR1_EL1,
        stela_contracts::Chip,
    };
    use {crate::machine::system::Register, stela_contracts::Feature};

    const SMMLA: u32 = 0x4e9b_a781;
    let words = [0x9100_0400, SMMLA, 0xd400_0001];
    let build = |masked| {
        let template = super::super::tests::machine_on(Chip::AppleM2, &words);
        let mut profile = template.profile;
        profile.fixed = identification::derive(
            Chip::AppleM2,
            stela_compiler::TargetCpu::Baseline,
            true,
            identification::Speculation::NONE,
        );
        if masked {
            for (register, value) in &mut profile.fixed {
                if *register == ID_AA64ISAR1_EL1 {
                    *value &= !(0xf << 52);
                }
            }
        }
        let mut machine = Machine::new(
            template.cpu,
            template.host,
            crate::machine::NoDevices,
            profile,
            stela_compiler::TargetCpu::Baseline,
        );
        machine.cpu.registers.set(Register::CPACR_EL1, 0, 3 << 20);
        assert_eq!(
            machine.admit_table.withheld() & (1 << Feature::FEAT_I8MM as u64) != 0,
            masked
        );
        machine
    };
    for first_masked in [false, true] {
        let mut first = build(first_masked);
        first.host.store = Some(MemoryStore::default());
        assert!(matches!(
            first.fetch_block(BASE, false),
            Ok(Fetched::Block(_))
        ));
        let offer = first.host.store.as_mut().unwrap().offers.pop().unwrap();
        let mut record = Record::new(translations::translator_id(), offer.key);
        record.insert(proven(&offer));
        for shared in [false, true] {
            for same_profile in [false, true] {
                let mut store = MemoryStore::default();
                if shared {
                    store
                        .prepared
                        .insert((offer.key, offer.pc, offer.context), offer.clone().shared());
                } else {
                    store.records.insert(offer.key, Arc::new(record.clone()));
                }
                let mut next = build(if same_profile {
                    first_masked
                } else {
                    !first_masked
                });
                next.host.store = Some(store);
                assert!(matches!(
                    next.fetch_block(BASE, false),
                    Ok(Fetched::Block(_))
                ));
                assert_eq!(next.statistics.compiled, u64::from(!same_profile));
                assert_eq!(next.statistics.shared, u64::from(shared && same_profile));
                assert_eq!(next.statistics.stored, u64::from(!shared && same_profile));
            }
        }
    }
}

#[test]
fn a_block_another_thread_prepared_at_the_same_pc_is_installed_as_it_is() {
    // adr x0, #8; add x0, x0, #1; svc #0
    let words = [0x1000_0040, 0x9100_0400, 0xd400_0001];
    let mut first = machine(&words);
    first.host.store = Some(MemoryStore::default());
    assert!(first.fetch_block(BASE + 4, false).is_ok());
    let offer = first.host.store.as_mut().unwrap().offers.pop().unwrap();
    let mut store = MemoryStore::default();
    store
        .prepared
        .insert((offer.key, offer.pc, offer.context), offer.clone().shared());
    let mut second = machine(&words);
    second.host.store = Some(store);
    let Ok(Fetched::Block(index)) = second.fetch_block(BASE + 4, false) else {
        panic!("shared block")
    };
    assert_eq!(
        (second.statistics.compiled, second.statistics.shared),
        (0, 1)
    );
    assert_eq!(
        second.host.installed
            [second.code.blocks[index].code.0 / super::super::tests::NATIVE_CODE_STRIDE - 1],
        offer.block().native.x86_64_code
    );
    // Another PC of the same page is compiled.
    assert!(second.fetch_block(BASE, false).is_ok());
    assert_eq!(second.statistics.compiled, 1);
}

/// What the directory store's workers do with an offer.
fn proven(offer: &Offer) -> translations::Entry {
    offer.prove().unwrap()
}

#[test]
fn a_stored_block_of_the_same_page_bytes_is_installed_without_compiling() {
    // adr x0, #8; add x0, x0, #1; svc #0
    let words = [0x1000_0040, 0x9100_0400, 0xd400_0001];
    let mut first = machine(&words);
    first.host.store = Some(MemoryStore::default());
    let Ok(Fetched::Block(compiled)) = first.fetch_block(BASE + 4, false) else {
        panic!("block")
    };
    assert_eq!(
        (first.statistics.compiled, first.statistics.offered),
        (1, 1)
    );
    let offer = first.host.store.as_mut().unwrap().offers.pop().unwrap();
    assert_eq!((offer.page, offer.pc), (BASE, BASE + 4));
    let mut record = Record::new(translations::translator_id(), offer.key);
    record.insert(proven(&offer));
    let mut store = MemoryStore::default();
    store.records.insert(offer.key, Arc::new(record));
    let mut second = machine(&words);
    second.host.store = Some(store);
    let Ok(Fetched::Block(stored)) = second.fetch_block(BASE + 4, false) else {
        panic!("stored block")
    };
    assert_eq!(
        (second.statistics.compiled, second.statistics.stored),
        (0, 1)
    );
    let installed = |machine: &super::super::Machine<
        super::super::tests::FakeHost,
        crate::machine::NoDevices,
    >,
                     index: usize| {
        let block = &machine.code.blocks[index];
        (
            machine.host.installed[block.code.0 / super::super::tests::NATIVE_CODE_STRIDE - 1]
                .clone(),
            block.instructions,
            block.terminal,
            block.successors.len(),
        )
    };
    assert_eq!(installed(&first, compiled), installed(&second, stored));
    // Another entry PC, or the same page with other bytes, is compiled.
    assert!(matches!(
        second.fetch_block(BASE, false),
        Ok(Fetched::Block(_))
    ));
    assert_eq!(second.statistics.compiled, 1);
    let mut changed = machine(&[0x1000_0040, 0x9100_0800, 0xd400_0001]);
    changed.host.store = second.host.store.take();
    assert!(matches!(
        changed.fetch_block(BASE + 4, false),
        Ok(Fetched::Block(_))
    ));
    assert_eq!(
        (changed.statistics.compiled, changed.statistics.stored),
        (1, 0)
    );
}

#[test]
fn a_block_longer_than_the_first_window_compiles_and_is_stored_whole() {
    // 40 × add x0, x0, #1; svc #0
    const ADDS: usize = 40;
    let mut words = vec![0x9100_0400; ADDS];
    words.push(0xd400_0001);
    let block_bytes = (ADDS + 1) * INSTRUCTION as usize;
    assert!(block_bytes > FIRST_WINDOW_BYTES);
    let mut first = machine(&words);
    first.host.store = Some(MemoryStore::default());
    let Ok(Fetched::Block(index)) = first.fetch_block(BASE, false) else {
        panic!("block")
    };
    assert_eq!(first.code.blocks[index].source_bytes(), block_bytes);
    let offer = first.host.store.as_mut().unwrap().offers.pop().unwrap();
    assert_eq!(offer.input().len(), block_bytes);
    let mut record = Record::new(translations::translator_id(), offer.key);
    record.insert(proven(&offer));
    let mut store = MemoryStore::default();
    store.records.insert(offer.key, Arc::new(record));
    let mut second = machine(&words);
    second.host.store = Some(store);
    let Ok(Fetched::Block(stored)) = second.fetch_block(BASE, false) else {
        panic!("stored block")
    };
    assert_eq!(
        (second.statistics.compiled, second.statistics.stored),
        (0, 1)
    );
    assert_eq!(second.code.blocks[stored].source_bytes(), block_bytes);
    // A changed word past the first window, inside the block: the stored
    // block's bytes no longer match, and it is compiled.
    words[ADDS - 1] = 0x9100_0800;
    let mut changed = machine(&words);
    changed.host.store = second.host.store.take();
    assert!(changed.fetch_block(BASE, false).is_ok());
    assert_eq!(
        (changed.statistics.compiled, changed.statistics.stored),
        (1, 0)
    );
}

#[test]
fn a_page_written_without_invalidation_is_read_and_hashed_again() {
    // add x0, x0, #1; svc #0; add x0, x0, #2; svc #0
    let mut machine = machine(&[0x9100_0400, 0xd400_0001, 0x9100_0800, 0xd400_0001]);
    machine.host.store = Some(MemoryStore::default());
    assert!(machine.fetch_block(BASE, false).is_ok());
    let key = machine.pages[&BASE].key;
    // The guest rewrites the second block (add x0, x0, #3) with no IC
    // IVAU: it is compiled from, and offered with, the new bytes.
    machine.host.ram[8..12].copy_from_slice(&0x9100_0c00u32.to_le_bytes());
    assert!(machine.fetch_block(BASE + 8, false).is_ok());
    let offer = machine.host.store.as_mut().unwrap().offers.pop().unwrap();
    assert_ne!(offer.key, key);
    assert_eq!(offer.key, machine.pages[&BASE].key);
    assert_eq!(offer.input()[..4], 0x9100_0c00u32.to_le_bytes());
}

#[test]
fn an_invalidated_page_is_read_and_looked_up_again() {
    let mut machine = machine(&[0x9100_0400, 0xd400_0001]);
    machine.host.store = Some(MemoryStore::default());
    assert!(machine.fetch_block(BASE, false).is_ok());
    let key = machine.pages[&BASE].key;
    machine.host.ram[0] ^= 1;
    machine.invalidate_code_page(BASE);
    assert!(machine.pages.is_empty());
    assert!(machine.fetch_block(BASE, false).is_ok());
    assert_ne!(machine.pages[&BASE].key, key);
}

#[test]
fn the_least_recently_used_pages_are_let_go() {
    // A block in each of five pages: add x0, x0, #1; svc #0.
    const PAGE_WORDS: usize = FETCH_PAGE as usize / INSTRUCTION as usize;
    let mut words = vec![0; 4 * PAGE_WORDS + 2];
    for page in 0..5 {
        words[page * PAGE_WORDS..page * PAGE_WORDS + 2]
            .copy_from_slice(&[0x9100_0400, 0xd400_0001]);
    }
    let mut machine = machine(&words);
    machine.host.store = Some(MemoryStore::default());
    let page = |n: u64| BASE + n * FETCH_PAGE;
    for n in 0..4 {
        assert!(machine.fetch_block(page(n), false).is_ok());
    }
    // Page 0's second block uses it again: page 1 is the least
    // recently used when page 4 needs room.
    assert!(machine.fetch_block(page(0) + 4, false).is_ok());
    assert!(machine.fetch_block(page(4), false).is_ok());
    assert_eq!(
        machine.pages.keys().copied().collect::<Vec<_>>(),
        [page(0), page(2), page(3), page(4)]
    );
}

#[test]
fn a_page_let_go_and_read_again_takes_its_known_key_unless_it_changed() {
    // A block in each of five pages: add x0, x0, #1; svc #0.
    const PAGE_WORDS: usize = FETCH_PAGE as usize / INSTRUCTION as usize;
    let mut words = vec![0; 4 * PAGE_WORDS + 2];
    for page in 0..5 {
        words[page * PAGE_WORDS..page * PAGE_WORDS + 2]
            .copy_from_slice(&[0x9100_0400, 0xd400_0001]);
    }
    let mut machine = machine(&words);
    machine.host.store = Some(MemoryStore::default());
    let page = |n: u64| BASE + n * FETCH_PAGE;
    for n in 0..5 {
        assert!(machine.fetch_block(page(n), false).is_ok());
    }
    // Page 0 was let go; its second block reads it again.
    assert!(!machine.pages.contains_key(&page(0)));
    let key = Digest::of(&machine.host.ram[..FETCH_PAGE as usize]);
    assert!(machine.fetch_block(page(0) + 4, false).is_ok());
    assert_eq!(machine.statistics.known_keys, 1);
    assert_eq!(machine.pages[&page(0)].key, key);
    // A page written while let go is hashed again.
    for n in 1..5 {
        assert!(machine.fetch_block(page(n) + 4, false).is_ok());
    }
    assert!(!machine.pages.contains_key(&page(0)));
    let known = machine.statistics.known_keys;
    machine.host.ram[100] ^= 1;
    assert!(machine.fetch_block(page(0) + 8, false).is_ok());
    assert_eq!(machine.statistics.known_keys, known);
    let changed = Digest::of(&machine.host.ram[..FETCH_PAGE as usize]);
    assert_eq!(machine.pages[&page(0)].key, changed);
    // An IC IVAU forgets the known key too.
    machine.invalidate_code_page(page(0));
    assert!(machine.fetch_block(page(0) + 12, false).is_ok());
    assert_eq!(machine.statistics.known_keys, known);
}

#[test]
fn a_stored_install_relocates_into_the_machines_own_buffers() {
    // adr x0, #8; add x0, x0, #1; svc #0, twice in the page.
    let words = [
        0x1000_0040,
        0x9100_0400,
        0xd400_0001,
        0x1000_0040,
        0x9100_0400,
        0xd400_0001,
    ];
    let mut first = machine(&words);
    first.host.store = Some(MemoryStore::default());
    assert!(first.fetch_block(BASE, false).is_ok());
    assert!(first.fetch_block(BASE + 12, false).is_ok());
    let offers = core::mem::take(&mut first.host.store.as_mut().unwrap().offers);
    let mut record = Record::new(translations::translator_id(), offers[0].key);
    for offer in &offers {
        record.insert(proven(offer));
    }
    let mut store = MemoryStore::default();
    store.records.insert(offers[0].key, Arc::new(record));
    let mut second = machine(&words);
    second.host.store = Some(store);
    assert!(second.fetch_block(BASE, false).is_ok());
    let buffer = second.relocated.code.as_ptr();
    assert!(second.fetch_block(BASE + 12, false).is_ok());
    assert_eq!(second.statistics.stored, 2);
    assert_eq!(second.relocated.code.as_ptr(), buffer);
    // Each block's code is what the compiling machine installed.
    assert_eq!(second.host.installed, first.host.installed);
}

#[test]
fn with_memory_low_pages_stay_and_nothing_is_offered() {
    // A block in each of two pages: add x0, x0, #1; svc #0.
    const PAGE_WORDS: usize = FETCH_PAGE as usize / INSTRUCTION as usize;
    let mut words = vec![0; PAGE_WORDS + 2];
    for page in 0..2 {
        words[page * PAGE_WORDS..page * PAGE_WORDS + 2]
            .copy_from_slice(&[0x9100_0400, 0xd400_0001]);
    }
    let mut machine = machine(&words);
    machine.host.store = Some(MemoryStore::default());
    // Nothing to reclaim yet, which would forget the first page's block.
    machine.memory_release_blocks = usize::MAX;
    assert!(machine.fetch_block(BASE, false).is_ok());
    assert_eq!(machine.statistics.offered, 1);
    machine.host.memory_low = true;
    assert!(machine.fetch_block(BASE + FETCH_PAGE, false).is_ok());
    assert_eq!(
        machine.pages.keys().copied().collect::<Vec<_>>(),
        [BASE, BASE + FETCH_PAGE]
    );
    assert_eq!(machine.statistics.offered, 1);
    assert_eq!(machine.host.store.as_ref().unwrap().offers.len(), 1);
}

/// `PES STORE`: every install counts once, at its level, by where its
/// block came from: compiled with a store page, compiled without one,
/// relocated from the pack, or taken from another PE.
#[test]
fn installs_count_where_their_block_came_from_at_their_level() {
    use super::super::StoreSource;
    const USER: usize = 0;
    const KERNEL: usize = 1;
    let count = |source: StoreSource| source as usize;
    // adr x0, #8; add x0, x0, #1; svc #0
    let words = [0x1000_0040, 0x9100_0400, 0xd400_0001];
    let mut first = machine(&words);
    first.host.store = Some(MemoryStore::default());
    assert!(first.fetch_block(BASE + 4, false).is_ok());
    let mut sources = [[0; StoreSource::COUNT]; 2];
    sources[KERNEL][count(StoreSource::Compiled)] = 1;
    assert_eq!(first.statistics.store_sources, sources);

    let offer = first.host.store.as_mut().unwrap().offers.pop().unwrap();
    let mut record = Record::new(translations::translator_id(), offer.key);
    record.insert(proven(&offer));
    let mut store = MemoryStore::default();
    store.records.insert(offer.key, Arc::new(record));
    store
        .prepared
        .insert((offer.key, BASE + 8, offer.context), {
            let input = offer.input()[4..].to_vec();
            Offer::compile(
                offer.key,
                offer.page,
                BASE + 8,
                offer.context,
                input,
                Arc::clone(offer.preparation()),
            )
            .unwrap()
            .shared()
        });
    let mut second = machine(&words);
    second.host.store = Some(store);
    assert!(second.fetch_block(BASE + 4, false).is_ok());
    assert!(second.fetch_block(BASE + 8, false).is_ok());
    let mut sources = [[0; StoreSource::COUNT]; 2];
    sources[KERNEL][count(StoreSource::Pack)] = 1;
    sources[KERNEL][count(StoreSource::Shared)] = 1;
    assert_eq!(second.statistics.store_sources, sources);
    assert_eq!((second.statistics.stored, second.statistics.shared), (1, 1));

    // Without a store no page is keyed; at EL0 the counts go to the user row.
    let mut user = machine(&words);
    super::super::tests::normal_memory(&mut user, true);
    user.cpu.pstate.el = 0;
    assert!(matches!(
        user.fetch_block(BASE + 4, false),
        Ok(Fetched::Block(_))
    ));
    let mut sources = [[0; StoreSource::COUNT]; 2];
    sources[USER][count(StoreSource::Unstorable)] = 1;
    assert_eq!(user.statistics.store_sources, sources);
    assert_eq!(user.statistics.compiled, 1);
}

/// The same page content at four addresses: compiled at the first,
/// compiled and proven at the second, rebased at the third, where it is
/// what a compile there installs. A rewritten page is compiled again.
#[test]
fn a_block_compiled_at_two_addresses_is_rebased_at_a_third() {
    use super::super::tests::{FakeHost, NATIVE_CODE_STRIDE};
    const PAGE_WORDS: usize = FETCH_PAGE as usize / INSTRUCTION as usize;
    // adr x0, #8; add x0, x0, #1; svc #0, at the start of each page.
    let block = [0x1000_0040, 0x9100_0400, 0xd400_0001];
    let mut words = vec![0; 3 * PAGE_WORDS + block.len()];
    for page in 0..4 {
        words[page * PAGE_WORDS..page * PAGE_WORDS + block.len()].copy_from_slice(&block);
    }
    let page = |n: u64| BASE + n * FETCH_PAGE;
    let mut reusing = machine(&words);
    reusing.host.store = Some(MemoryStore::default());
    assert!(reusing.fetch_block(page(0), false).is_ok());
    assert_eq!(reusing.code.rebases.counts().proven, 0);
    assert!(reusing.fetch_block(page(1), false).is_ok());
    assert_eq!(reusing.statistics.compiled, 2);
    assert_eq!(reusing.code.rebases.counts().proven, 1);
    let Ok(Fetched::Block(rebased)) = reusing.fetch_block(page(2), false) else {
        panic!("rebased block")
    };
    assert_eq!(reusing.statistics.compiled, 2);
    let counts = reusing.code.rebases.counts();
    assert_eq!((counts.hits, counts.proven, counts.unprovable), (1, 1, 0));
    assert_eq!(
        reusing.statistics.store_sources[1][super::super::StoreSource::Rebased as usize],
        1
    );

    type Fake = super::super::Machine<FakeHost, crate::machine::NoDevices>;
    let installed = |machine: &Fake, index: usize| {
        let block = &machine.code.blocks[index];
        (
            machine.host.installed[block.code.0 / NATIVE_CODE_STRIDE - 1].clone(),
            block.pc,
            block.source_bytes(),
            block.instructions,
            block.terminal,
            block.successors.len(),
        )
    };
    let mut fresh = machine(&words);
    let Ok(Fetched::Block(compiled)) = fresh.fetch_block(page(2), false) else {
        panic!("compiled block")
    };
    assert_eq!(installed(&reusing, rebased), installed(&fresh, compiled));

    // The third page rewritten (add x0, x0, #2): another key, compiled.
    words[2 * PAGE_WORDS + 1] = 0x9100_0800;
    let at = 2 * FETCH_PAGE as usize + 4;
    reusing.host.ram[at..at + 4].copy_from_slice(&0x9100_0800u32.to_le_bytes());
    reusing.invalidate_code_page(page(2));
    let Ok(Fetched::Block(changed)) = reusing.fetch_block(page(2), false) else {
        panic!("compiled block")
    };
    assert_eq!(reusing.statistics.compiled, 3);
    assert_eq!(reusing.code.rebases.counts().hits, 1);
    let mut fresh = machine(&words);
    let Ok(Fetched::Block(compiled)) = fresh.fetch_block(page(2), false) else {
        panic!("compiled block")
    };
    assert_eq!(installed(&reusing, changed), installed(&fresh, compiled));
    // The fourth page still holds the proven content.
    assert!(reusing.fetch_block(page(3), false).is_ok());
    assert_eq!(
        (
            reusing.statistics.compiled,
            reusing.code.rebases.counts().hits
        ),
        (3, 2)
    );
}
