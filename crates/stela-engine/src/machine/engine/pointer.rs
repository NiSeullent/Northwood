//! The PAC results the engine computes, remembers and lets native code
//! complete (`x86_64::pointer`).
//!
//! A signing or authentication is a pure function of the five keys, the
//! chip's PAC version, the regime's TCR (Arm `AddPAC`, `Auth`) and its
//! operands, so [`PointerService`] remembers results in a memo tagged with
//! the generation of those inputs. It lives at a fixed host address: PAC
//! thunks read its memo and its tag words, and on a memo miss call
//! [`complete`] with it, which computes the result exactly as the engine's
//! exit would ([`PointerService::result`]) without leaving native code.
//! A key's tag word is nonzero only while the service is open and the key
//! was found enabled and untrapped under the current derived state, which
//! is what both paths require.
use crate::pauth;
use alloc::{boxed::Box, vec, vec::Vec};
use core::mem::offset_of;
use stela_compiler::{Anchor, Place, block::PointerAuthenticationKey, x86_64::pointer};

/// The distinct PAC inputs whose memo generations are kept.
pub(super) const POINTER_INPUTS_SEEN: usize = 16;

/// What a PAC result depends on besides its operands: the five keys, the
/// chip's PAC version and the regime's TCR (Arm `AddPAC`, `Auth`).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct PointerInputs {
    pub(super) keys: [Option<[u64; 2]>; 5],
    pub(super) version: pauth::Version,
    pub(super) tcr: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PointerOperation {
    Sign,
    Authenticate { combined: bool },
}

/// The [`pointer::MemoEntry`] kinds of the results native code does not
/// complete, plus 1 for a combined form: an authentication that failed
/// without a fault, and one that faults with the code in `value`.
const UNAUTHENTICATED: u64 = 3;
const FAULTED: u64 = 5;
/// The tag bits of an entry's kind.
const KIND: u64 = 0xf;

/// A result: the value and whether it authenticated (signings always do),
/// or the PAC failure code of a faulting authentication.
pub(super) type PointerResult = Result<(u64, bool), u8>;

/// The memo kind and value of an operation's result.
fn memo_result(operation: PointerOperation, result: PointerResult) -> (u64, u64) {
    let combined = u64::from(operation == PointerOperation::Authenticate { combined: true });
    match (operation, result) {
        (PointerOperation::Sign, Ok((value, _))) => (pointer::SIGNED, value),
        (_, Ok((value, true))) => (pointer::AUTHENTICATED, value),
        (_, Ok((value, false))) => (UNAUTHENTICATED + combined, value),
        (_, Err(code)) => (FAULTED + combined, u64::from(code)),
    }
}

/// The memo index of a key.
pub(super) fn memo_key(key: stela_contracts::pauth::Key) -> u8 {
    pointer::key_index(match key {
        stela_contracts::pauth::Key::Ia => PointerAuthenticationKey::Ia,
        stela_contracts::pauth::Key::Ib => PointerAuthenticationKey::Ib,
        stela_contracts::pauth::Key::Da => PointerAuthenticationKey::Da,
        stela_contracts::pauth::Key::Db => PointerAuthenticationKey::Db,
    })
}

fn key_of(key: PointerAuthenticationKey) -> stela_contracts::pauth::Key {
    match key {
        PointerAuthenticationKey::Ia => stela_contracts::pauth::Key::Ia,
        PointerAuthenticationKey::Ib => stela_contracts::pauth::Key::Ib,
        PointerAuthenticationKey::Da => stela_contracts::pauth::Key::Da,
        PointerAuthenticationKey::Db => stela_contracts::pauth::Key::Db,
    }
}

/// The engine's PAC context, memo and tag words, which native code reads.
pub(super) struct PointerService {
    /// The keys, version and layout, while open.
    context: Option<pauth::Context>,
    /// Results, each tagged with the generation of the inputs it was
    /// computed under; `generation` is that of `inputs`. Allocated before
    /// the first usable native tag and kept at that address for the owner.
    memo: Option<Box<[pointer::MemoEntry]>>,
    inputs: Option<PointerInputs>,
    pub(super) generation: u64,
    /// The generations of the most recent distinct inputs, most recent
    /// first, so entries stay valid across a return to earlier inputs (EL0
    /// and EL1 keys, guarded levels, PE switches).
    seen: Vec<(PointerInputs, u64)>,
    /// The last generation handed out; an evicted one is never reused.
    generations: u64,
    /// Each key's tag word for native code: zero until a PAC exit finds the
    /// key enabled and untrapped under the current memo, and again whenever
    /// derived state may have changed.
    pub(super) tags: [u64; pointer::TAG_WORDS],
    /// Instruction/data XPAC masks, independent of cryptographic key state.
    pub(super) strips: [pointer::strip::Cell; 2],
    /// [`SERVICE`]'s address, which blocks completing PAC instructions
    /// inside themselves call through (`x86_64::pointer::inline`).
    routine: u64,
    /// Results computed since boot (the cipher ran): each is a memo miss,
    /// of a block, a thunk or an engine exit.
    pub(super) fills: u64,
}

/// The routine a thunk calls on a memo miss.
pub(super) const SERVICE: pointer::ServiceRoutine = complete;

/// Where blocks find the service's tag words, strip cells and routine.
pub(super) const CELLS: pointer::inline::Cells = pointer::inline::Cells {
    tags: offset_of!(PointerService, tags) as u32,
    strips: offset_of!(PointerService, strips) as u32,
    routine: offset_of!(PointerService, routine) as u32,
};

/// Where native code finds the service's tag words.
pub(super) const TAGS: Place = Place::new(Anchor::Pointer, offset_of!(PointerService, tags));

/// Where native code finds the stripping cell of `data`.
pub(super) const fn strip_place(data: bool) -> Place {
    Place::new(
        Anchor::Pointer,
        offset_of!(PointerService, strips) + data as usize * size_of::<pointer::strip::Cell>(),
    )
}

impl PointerService {
    pub(super) fn new() -> Self {
        Self {
            context: None,
            memo: None,
            inputs: None,
            generation: 0,
            seen: Vec::new(),
            generations: 0,
            tags: [0; pointer::TAG_WORDS],
            strips: [pointer::strip::Cell::default(); 2],
            routine: SERVICE as usize as u64,
            fills: 0,
        }
    }

    /// Forget the context and close every key, as derived state changed.
    pub(super) fn close(&mut self) {
        self.context = None;
        self.close_native();
    }

    fn close_native(&mut self) -> [bool; 2] {
        self.tags = [0; pointer::TAG_WORDS];
        self.strips
            .each_mut()
            .map(|cell| core::mem::take(&mut cell.enabled) != 0)
    }

    /// Derived state changed, now with `inputs`: every tag word and
    /// stripping cell closes, and the context stays when it was made for
    /// these inputs (else it is made anew, with their memo generation).
    /// Returns which stripping cells were open, for the engine to reopen
    /// with [`Self::reopen`].
    pub(super) fn revalidate(&mut self, inputs: PointerInputs) -> [bool; 2] {
        let stripping = self.close_native();
        if self.context.is_none() || self.inputs != Some(inputs) {
            self.open(inputs);
        }
        stripping
    }

    /// Whether every key is closed to native code.
    pub(super) fn tags_closed(&self) -> bool {
        self.tags.iter().all(|&tag| tag == 0)
    }

    /// Whether native code had anything open: a tag word or a stripping cell.
    pub(super) fn any_open(&self) -> bool {
        self.tags.iter().any(|&tag| tag != 0) || self.strips.iter().any(|cell| cell.enabled != 0)
    }

    /// After [`Self::revalidate`], open `key`'s tag word as a PAC exit
    /// would: [`pointer::DISABLED`] while its SCTLR enable is clear, else,
    /// untrapped and present in the context, the memo's current tag.
    pub(super) fn reopen(
        &mut self,
        key: stela_contracts::pauth::Key,
        enabled: bool,
        trapped: bool,
    ) {
        let index = memo_key(key);
        self.tags[usize::from(index)] = if !enabled {
            pointer::DISABLED
        } else if !trapped
            && self
                .context
                .as_ref()
                .is_some_and(|context| context.has_key(key))
        {
            self.native_tag(index)
        } else {
            0
        };
    }

    /// After [`Self::revalidate`], open the generic key's tag word as a
    /// PACGA exit would: untrapped and present in the context, the memo's
    /// current tag.
    pub(super) fn reopen_generic(&mut self, trapped: bool) {
        let usable = !trapped
            && self
                .context
                .as_ref()
                .is_some_and(pauth::Context::has_generic_key);
        self.tags[usize::from(pointer::GENERIC_KEY)] = if usable {
            self.native_tag(pointer::GENERIC_KEY)
        } else {
            0
        };
    }

    /// After [`Self::revalidate`], reopen the stripping cell of `data`.
    pub(super) fn reopen_stripping(&mut self, data: bool) {
        if let Some(keep) = self
            .context
            .as_ref()
            .map(|context| context.stripping_masks(data))
        {
            self.strips[usize::from(data)] = pointer::strip::Cell {
                enabled: 1,
                keep,
                upper_extension: !keep[1],
            };
        }
    }

    pub(super) fn open_stripping(&mut self, context: &pauth::Context, data: bool) {
        let keep = context.stripping_masks(data);
        self.strips[usize::from(data)] = pointer::strip::Cell {
            enabled: 1,
            keep,
            upper_extension: !keep[1],
        };
    }

    pub(super) fn is_open(&self) -> bool {
        self.context.is_some()
    }

    /// Open the service with the current `inputs`.
    pub(super) fn open(&mut self, inputs: PointerInputs) {
        if self.inputs != Some(inputs) {
            self.generation = self.generation_of(inputs);
            self.inputs = Some(inputs);
        }
        let [ia, ib, da, db, generic] = inputs.keys;
        let mut keys = pauth::Keys::from_optional_words([ia, ib, da, db]);
        if let Some(generic) = generic {
            keys = keys.with_generic_words(generic);
        }
        self.context = Some(
            pauth::Context::new_with_version(0, keys, inputs.version)
                .with_translation_regime(inputs.tcr)
                .with_implementation_defined_cipher(),
        );
    }

    /// A usable tag permits a thunk to dereference the memo anchor. A
    /// closed or disabled tag is handled before that dereference.
    fn native_tag(&mut self, key: u8) -> u64 {
        if self.memo.is_none() {
            self.allocate_memo();
        }
        pointer::tag(self.generation, key, 0)
    }

    #[cold]
    fn allocate_memo(&mut self) {
        self.memo =
            Some(vec![pointer::MemoEntry::default(); pointer::MEMO_ENTRIES].into_boxed_slice());
    }

    /// What native code finds through [`Anchor::Pointer`] and
    /// [`Anchor::Memo`] on its next entry.
    pub(super) fn anchors(&self) -> (u64, u64) {
        (
            self as *const Self as u64,
            self.memo.as_ref().map_or(0, |memo| memo.as_ptr() as u64),
        )
    }

    /// One signing or authentication with the open context, remembered
    /// while its inputs stay the same, with `key` known enabled and
    /// untrapped, which opens its tag word. The memo slot comes from the
    /// pointer and modifier alone, so which slot is touched never depends on
    /// a key. A signing whose result strips back to its pointer also
    /// remembers that authenticating the result with the same key and
    /// modifier succeeds with that pointer (Arm `Auth` recomputes the PAC of
    /// the stripped pointer), which answers the epilogue of a signed frame;
    /// a success answers both the combined and the plain authentication.
    /// `None` when the service is closed or the context lacks the key.
    pub(super) fn result(
        &mut self,
        operation: PointerOperation,
        key: stela_contracts::pauth::Key,
        pointer: u64,
        modifier: u64,
    ) -> Option<PointerResult> {
        if !self.context.as_ref()?.has_key(key) {
            return None;
        }
        let index = memo_key(key);
        self.tags[usize::from(index)] = self.native_tag(index);
        if let Some(result) = self.remembered(operation, key, pointer, modifier) {
            return Some(result);
        }
        let context = self.context.as_ref()?;
        self.fills += 1;
        let result = match operation {
            PointerOperation::Sign => Ok((context.sign(key, pointer, modifier).ok()?, true)),
            PointerOperation::Authenticate { combined } => {
                match context.authenticate_operation(key, pointer, modifier, combined) {
                    Ok(result) => Ok((result.pointer, result.authenticated)),
                    Err(pauth::AuthenticationError::Failure { code }) => Err(code),
                    Err(
                        pauth::AuthenticationError::KeyUnavailable { .. }
                        | pauth::AuthenticationError::GenericKeyUnavailable,
                    ) => return None,
                }
            }
        };
        let strips_back = match (operation, result) {
            (PointerOperation::Sign, Ok((signed, _))) => {
                (context.strip(key, signed) == pointer).then_some(signed)
            }
            _ => None,
        };
        self.remember(operation, key, pointer, modifier, result);
        if let Some(signed) = strips_back {
            let authenticate = PointerOperation::Authenticate { combined: false };
            self.remember(authenticate, key, signed, modifier, Ok((pointer, true)));
        }
        Some(result)
    }

    /// PACGA with the open context, remembered as [`Self::result`] does,
    /// the key known untrapped. `None` when the context lacks the key.
    pub(super) fn generic_code(&mut self, data: u64, modifier: u64) -> Option<u64> {
        let key = pointer::GENERIC_KEY;
        let code = self.context.as_ref()?.generic_code(data, modifier).ok()?;
        self.fills += 1;
        let tag = self.native_tag(key);
        let memo = self.memo.as_mut().expect("memo for open tag");
        memo[pointer::slot(data, modifier)] = pointer::MemoEntry {
            pointer: data,
            modifier,
            tag: tag | pointer::SIGNED,
            value: code,
        };
        self.tags[usize::from(key)] = tag;
        Some(code)
    }

    /// The memo generation of `inputs`: the one they had if they are among
    /// the [`POINTER_INPUTS_SEEN`] most recent distinct inputs, else a new
    /// one. A result depends only on its inputs and operands, so an entry
    /// of a reused generation is still exact.
    fn generation_of(&mut self, inputs: PointerInputs) -> u64 {
        let generation = match self.seen.iter().position(|&(known, _)| known == inputs) {
            Some(index) => self.seen.remove(index).1,
            None => {
                self.generations += 1;
                self.generations
            }
        };
        self.seen.insert(0, (inputs, generation));
        self.seen.truncate(POINTER_INPUTS_SEEN);
        generation
    }

    /// The current memo's result of `operation` with `key` on these
    /// operands, if it has one: a success answers both authentication forms.
    pub(super) fn remembered(
        &self,
        operation: PointerOperation,
        key: stela_contracts::pauth::Key,
        pointer: u64,
        modifier: u64,
    ) -> Option<PointerResult> {
        let entry = self.memo.as_ref()?[pointer::slot(pointer, modifier)];
        let tag = pointer::tag(self.generation, memo_key(key), 0);
        if entry.pointer != pointer || entry.modifier != modifier || entry.tag & !KIND != tag {
            return None;
        }
        let kind = entry.tag & KIND;
        let combined = u64::from(operation == PointerOperation::Authenticate { combined: true });
        match operation {
            PointerOperation::Sign => (kind == pointer::SIGNED).then_some(Ok((entry.value, true))),
            PointerOperation::Authenticate { .. } => match kind {
                pointer::AUTHENTICATED => Some(Ok((entry.value, true))),
                kind if kind == UNAUTHENTICATED + combined => Some(Ok((entry.value, false))),
                kind if kind == FAULTED + combined => Some(Err(entry.value as u8)),
                _ => None,
            },
        }
    }

    fn remember(
        &mut self,
        operation: PointerOperation,
        key: stela_contracts::pauth::Key,
        pointer: u64,
        modifier: u64,
        result: PointerResult,
    ) {
        let (kind, value) = memo_result(operation, result);
        let memo = self.memo.as_mut().expect("memo for open tag");
        memo[pointer::slot(pointer, modifier)] = pointer::MemoEntry {
            pointer,
            modifier,
            tag: pointer::tag(self.generation, memo_key(key), kind),
            value,
        };
    }
}

translated_callee! {
/// A PAC thunk's memo miss (`x86_64::pointer::ServiceRoutine`): the result
/// of `request` on these operands while the requested key's tag word is
/// open, completed when it is a signing or a successful authentication.
/// A failed authentication is remembered for the exit the thunk then takes.
pub(super) fn complete(
    service: u64,
    request: u32,
    pointer: u64,
    modifier: u64,
) -> pointer::Completion {
    // SAFETY: thunks pass the address of the machine's boxed service, which
    // outlives every installed thunk; the engine does not touch it while
    // native code runs.
    let service = unsafe { &mut *(service as *mut PointerService) };
    if pointer::requests_generic_code(request) {
        if service.tags[usize::from(pointer::GENERIC_KEY)] == 0 {
            return pointer::Completion::default();
        }
        return service.generic_code(pointer, modifier).map_or_else(
            pointer::Completion::default,
            |value| pointer::Completion {
                completed: 1,
                value,
            },
        );
    }
    let (key, authenticate, combined) = pointer::requested(request);
    let key = key_of(key);
    if service.tags[usize::from(memo_key(key))] == 0 {
        return pointer::Completion::default();
    }
    let operation = if authenticate {
        PointerOperation::Authenticate { combined }
    } else {
        PointerOperation::Sign
    };
    match service.result(operation, key, pointer, modifier) {
        Some(Ok((value, true))) => pointer::Completion {
            completed: 1,
            value,
        },
        _ => pointer::Completion::default(),
    }
}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_memo_is_needed_only_for_a_usable_native_tag() {
        use stela_contracts::pauth::Key;

        let mut owner = Box::new(PointerService::new());
        let (service, memo) = owner.anchors();
        assert_eq!(memo, 0);
        let first = inputs(pauth::Version::Classic, 25 | 25 << 16);
        let mut unavailable = first;
        unavailable.keys = [None; 5];
        owner.open(unavailable);
        owner.reopen(Key::Ia, false, true);
        owner.reopen(Key::Ib, true, false);
        owner.reopen_generic(false);
        owner.reopen_stripping(true);
        assert_eq!(owner.tags, [pointer::DISABLED, 0, 0, 0, 0]);
        assert_eq!(owner.strips[1].enabled, 1);
        assert_eq!(owner.result(PointerOperation::Sign, Key::Ib, 4, 8), None);
        assert_eq!(owner.generic_code(4, 8), None);
        assert_eq!(owner.anchors(), (service, 0));

        owner.revalidate(first);
        owner.reopen(Key::Ia, true, true);
        owner.reopen_generic(true);
        assert_eq!(owner.anchors(), (service, 0));
        owner.reopen(Key::Ia, true, false);
        let memo = owner.anchors().1;
        assert_ne!(memo, 0);
        let tag = owner.tags[0];
        let result = owner.result(PointerOperation::Sign, Key::Ia, 4, 8).unwrap();
        owner.close();
        assert_eq!(owner.anchors(), (service, memo));
        owner.open(unavailable);
        owner.revalidate(first);
        owner.reopen(Key::Ia, true, false);
        assert_eq!(owner.anchors(), (service, memo));
        assert_eq!(owner.tags[0], tag);
        assert_eq!(
            owner.remembered(PointerOperation::Sign, Key::Ia, 4, 8),
            Some(result)
        );

        let mut other = Box::new(PointerService::new());
        other.open(first);
        assert_eq!(other.anchors().1, 0);
        other.reopen_generic(false);
        assert_ne!(other.anchors().1, 0);
        assert_ne!(other.anchors().1, memo);
    }

    #[test]
    fn fills_count_each_computed_result_once() {
        use stela_contracts::pauth::Key;

        let mut service = Box::new(PointerService::new());
        service.open(inputs(pauth::Version::Classic, 25 | 25 << 16));
        assert_eq!(service.fills, 0);
        let signed = service
            .result(PointerOperation::Sign, Key::Ib, 0x1000, 0x20)
            .unwrap();
        assert_eq!(service.fills, 1);
        // A remembered signing, and the authentication its signing filled.
        service.result(PointerOperation::Sign, Key::Ib, 0x1000, 0x20);
        let (pointer, _) = signed.unwrap();
        let authenticate = PointerOperation::Authenticate { combined: false };
        service.result(authenticate, Key::Ib, pointer, 0x20);
        assert_eq!(service.fills, 1);
        service.result(PointerOperation::Sign, Key::Ib, 0x2000, 0x20);
        service.generic_code(3, 4).unwrap();
        assert_eq!(service.fills, 3);
    }

    #[test]
    fn a_generic_code_opens_its_own_tag_and_completes_for_thunks() {
        let mut service = Box::new(PointerService::new());
        let mut input = inputs(pauth::Version::Classic, 25 | 25 << 16);
        input.keys[4] = Some([7, 9]);
        service.open(input);
        let state = service.anchors().0;
        let form = pointer::Form::of(
            stela_compiler::block::PointerAuthenticationOperation::GenericCode(
                stela_compiler::block::pauth_memory::GenericCode {
                    destination: 1,
                    source: 2,
                    modifier: 3,
                },
            ),
        )
        .unwrap();
        let (data, modifier) = (0x1234, 0x5678);
        // Closed, the service is not asked.
        assert_eq!(
            complete(state, form.request(), data, modifier),
            pointer::Completion::default()
        );
        let code = service.generic_code(data, modifier).unwrap();
        let tag = pointer::tag(service.generation, pointer::GENERIC_KEY, 0);
        assert_eq!(service.tags[usize::from(pointer::GENERIC_KEY)], tag);
        assert_eq!(service.tags[..4], [0; 4]);
        assert_eq!(
            service.memo.as_ref().unwrap()[pointer::slot(data, modifier)],
            pointer::MemoEntry {
                pointer: data,
                modifier,
                tag: tag | pointer::SIGNED,
                value: code,
            }
        );
        // Open, a memo miss completes through it.
        assert_eq!(
            complete(state, form.request(), data, modifier),
            pointer::Completion {
                completed: 1,
                value: code,
            }
        );
        service.reopen_generic(true);
        assert_eq!(service.tags, [0; pointer::TAG_WORDS]);
        service.reopen_generic(false);
        assert_eq!(service.tags[usize::from(pointer::GENERIC_KEY)], tag);
        // Without the key nothing opens.
        input.keys[4] = None;
        service.close();
        service.open(input);
        assert_eq!(service.generic_code(data, modifier), None);
        service.reopen_generic(false);
        assert_eq!(service.tags, [0; pointer::TAG_WORDS]);
    }

    #[test]
    fn absent_keys_never_open_native_tags_or_reuse_a_known_key_generation() {
        let mut service = PointerService::new();
        let mut input = inputs(pauth::Version::Classic, 25 | 25 << 16);
        input.keys = [None; 5];
        service.open(input);
        assert!(
            service
                .result(
                    PointerOperation::Sign,
                    stela_contracts::pauth::Key::Ia,
                    0x4000,
                    0x8000
                )
                .is_none()
        );
        assert_eq!(service.tags, [0; pointer::TAG_WORDS]);
        let unknown_generation = service.generation;
        input.keys[0] = Some([11, 21]);
        service.close();
        service.open(input);
        assert!(
            service
                .result(
                    PointerOperation::Sign,
                    stela_contracts::pauth::Key::Ia,
                    0x4000,
                    0x8000
                )
                .is_some()
        );
        assert_ne!(service.generation, unknown_generation);
        assert_eq!(service.tags[1..], [0; pointer::TAG_WORDS - 1]);
        service.close();
        input.keys[0] = None;
        service.open(input);
        assert_eq!(service.generation, unknown_generation);
        assert!(
            service
                .result(
                    PointerOperation::Sign,
                    stela_contracts::pauth::Key::Ia,
                    0x4000,
                    0x8000
                )
                .is_none()
        );
        assert_eq!(service.tags, [0; pointer::TAG_WORDS]);
    }

    #[test]
    fn stripping_cells_match_the_authoritative_layout_and_close_without_keys() {
        let mut service = PointerService::new();
        for t0 in [0, 16, 17, 25, 39, 63] {
            for t1 in [0, 16, 17, 25, 39, 63] {
                for controls in 0..16u64 {
                    let tcr = t0
                        | t1 << 16
                        | (controls & 1) << 37
                        | (controls & 2) << 37
                        | (controls & 4) << 49
                        | (controls & 8) << 49;
                    let context =
                        pauth::Context::new(0, pauth::Keys::from_optional_words([None; 4]))
                            .with_translation_regime(tcr);
                    for data in [false, true] {
                        service.open_stripping(&context, data);
                        assert!(
                            !service.is_open(),
                            "no cryptographic context or key is needed"
                        );
                        assert_eq!(service.tags, [0; pointer::TAG_WORDS]);
                        let cell = service.strips[usize::from(data)];
                        assert_eq!(cell.enabled, 1);
                        for pointer in [0, u64::MAX, 1 << 55, 1 << 63, 0xdead_1234_5678_abcd] {
                            let upper = pointer & (1 << 55) != 0;
                            let keep = cell.keep[usize::from(upper)];
                            let key = if data {
                                stela_contracts::pauth::Key::Da
                            } else {
                                stela_contracts::pauth::Key::Ia
                            };
                            assert_eq!(
                                (pointer & keep) | if upper { cell.upper_extension } else { 0 },
                                context.strip(key, pointer)
                            );
                        }
                    }
                    service.close();
                    assert!(service.strips.iter().all(|cell| cell.enabled == 0));
                }
            }
        }
    }

    fn inputs(version: pauth::Version, tcr: u64) -> PointerInputs {
        let key = |index: u64| [0x1122_3344_5566_7788 ^ index, 0x99aa_bbcc_ddee_ff00 + index];
        PointerInputs {
            keys: [key(1), key(2), key(3), key(4), key(5)].map(Some),
            version,
            tcr,
        }
    }

    fn request(key: PointerAuthenticationKey, operation: PointerOperation) -> u32 {
        let mut request = u32::from(pointer::key_index(key));
        if let PointerOperation::Authenticate { combined } = operation {
            request |= pointer::REQUEST_AUTHENTICATE;
            if combined {
                request |= pointer::REQUEST_COMBINED;
            }
        }
        request
    }

    /// The native completion answers exactly what the engine's exit
    /// computes, for every key, form and version and for kernel and user
    /// pointers, signed, corrupted and unsigned: it completes signings and
    /// successful authentications, refuses the rest, and refuses everything
    /// for a key whose tag word is closed.
    #[test]
    fn native_completions_equal_the_engine_results() {
        use PointerAuthenticationKey as Key;
        // T0SZ = T1SZ = 25 (39-bit VAs), TBI0 on.
        let tcr = 25 | 25 << 16 | 1 << 37;
        let operations = [
            PointerOperation::Sign,
            PointerOperation::Authenticate { combined: false },
            PointerOperation::Authenticate { combined: true },
        ];
        for version in [
            pauth::Version::Classic,
            pauth::Version::Enhanced,
            pauth::Version::Faulting,
            pauth::Version::FaultingCombined,
        ] {
            let inputs = inputs(version, tcr);
            let mut native = Box::new(PointerService::new());
            let mut engine = PointerService::new();
            native.open(inputs);
            engine.open(inputs);
            let state = native.anchors().0;
            for key in [Key::Ia, Key::Ib, Key::Da, Key::Db] {
                let pauth_key = key_of(key);
                let index = usize::from(pointer::key_index(key));
                for (step, base) in [0xffff_ff80_0012_3450_u64, 0x0000_0001_0000_8000]
                    .into_iter()
                    .enumerate()
                {
                    let modifier = 0xffff_ff80_0040_0000 + step as u64 * 0x10;
                    let signed = engine.result(PointerOperation::Sign, pauth_key, base, modifier);
                    let Some(Ok((signed, true))) = signed else {
                        panic!("signing {base:#x}: {signed:?}");
                    };
                    for operand in [base, signed, signed ^ (1 << 50), signed ^ (1 << 62)] {
                        for operation in operations {
                            native.tags = [0; pointer::TAG_WORDS];
                            let asked = complete(state, request(key, operation), operand, modifier);
                            assert_eq!(asked, pointer::Completion::default(), "a closed key");
                            native.reopen(pauth_key, true, false);
                            assert_ne!(native.tags[index], 0);
                            let expected = engine.result(operation, pauth_key, operand, modifier);
                            let asked = complete(state, request(key, operation), operand, modifier);
                            let completed = match expected {
                                Some(Ok((value, true))) => pointer::Completion {
                                    completed: 1,
                                    value,
                                },
                                _ => pointer::Completion::default(),
                            };
                            assert_eq!(asked, completed, "{version:?} {key:?} {operation:?}");
                            // The native call remembered what the exit then finds.
                            assert_eq!(
                                native.remembered(operation, pauth_key, operand, modifier),
                                expected
                            );
                        }
                    }
                }
            }
        }
    }

    /// A return to earlier inputs finds their generation again; other inputs
    /// get their own, and beyond the inputs kept a generation is new.
    #[test]
    fn generations_follow_the_inputs() {
        let mut service = PointerService::new();
        let tcr = 25 | 25 << 16;
        let (first, other) = (
            inputs(pauth::Version::Faulting, tcr),
            inputs(pauth::Version::Classic, tcr),
        );
        service.open(first);
        let generation = service.generation;
        service.open(other);
        assert_ne!(service.generation, generation);
        service.open(first);
        assert_eq!(service.generation, generation);
        for extra in 1..=POINTER_INPUTS_SEEN as u64 {
            service.open(inputs(pauth::Version::Faulting, tcr + (extra << 32)));
        }
        service.open(first);
        assert!(service.generation > generation);
        service.close();
        assert!(!service.is_open());
        assert_eq!(service.tags, [0; pointer::TAG_WORDS]);
        assert_eq!(
            service.result(
                PointerOperation::Sign,
                stela_contracts::pauth::Key::Ia,
                1,
                2
            ),
            None
        );
    }
}
