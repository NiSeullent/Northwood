//! Emitted byte buffers with instruction-boundary accounting.
use super::Asm;
use alloc::vec::Vec;

/// Emitted bytes and the number of instructions among them.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct Code {
    pub(super) bytes: Vec<u8>,
    pub(super) instructions: usize,
    pub(super) sites: Vec<EmissionSite>,
    /// Whether guest registers live in host registers ([`super::super::pinned`]).
    pub(super) pinned: bool,
    locals: Option<super::super::pinned::LocalRegisters>,
    /// The locals whose guest values only their hosts hold
    /// ([`super::super::pinned::Span`]), which a fault recovers from there.
    dirty: super::super::pinned::Dirty,
    /// The block's PC, in code whose exits keep a link slot
    /// ([`super::super::chain::link_slot`]).
    pub(super) linked_block: Option<u64>,
    /// Whether this code runs under PSTATE.DIT on a target that audits it.
    constant_time: bool,
    /// Index in the block's source map of the instruction being emitted.
    instruction: usize,
    /// Jumps to check failures, which the block places after its exits,
    /// with the instructions they fail and the locals dirty there.
    failures: Vec<(super::Patch, usize, super::super::pinned::Dirty)>,
    /// The masked-base plan for the access being emitted, and the plan the
    /// access applied ([`super::super::masked_base`]).
    base_request: Option<super::super::masked_base::Request>,
    base_applied: Option<super::super::masked_base::Plan>,
    /// Guest vectors kept in host XMM registers ([`super::super::vector_cache`]).
    pub(in crate::x86_64) vectors: super::super::vector_cache::VectorCache,
}

/// Exact faulting instruction interval, separate from source retirement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct EmissionSite {
    pub start: usize,
    pub end: usize,
    pub site: crate::block::RecoverySite,
    /// The host register that holds SP during the access, if not its own.
    pub stack: Option<u8>,
    /// The instruction this site belongs to, where the code is out of line.
    pub source: Option<usize>,
    /// The locals whose guest values only their hosts hold at the site.
    pub locals: super::super::pinned::Dirty,
}

/// A byte and instruction position captured at an emission boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Checkpoint {
    bytes: usize,
    pub(super) instructions: usize,
}

impl Checkpoint {
    pub(crate) const fn offset(self) -> usize {
        self.bytes
    }
}

/// A complete emitted fragment, including its exact instruction count.
/// Both endpoints belong to the same unmodified Code buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Span {
    start: Checkpoint,
    end: Checkpoint,
}
impl Span {
    pub(crate) const fn start(self) -> Checkpoint {
        self.start
    }
    pub(crate) const fn end(self) -> Checkpoint {
        self.end
    }
    pub(crate) const fn instructions(self) -> usize {
        self.end.instructions - self.start.instructions
    }
    pub(crate) const fn is_empty(self) -> bool {
        self.start.bytes == self.end.bytes
    }
}

/// Bytes a buffer starts with room for: most blocks' code, so emission
/// rarely grows it (growth was a fifth of compile time).
const INITIAL_BYTES: usize = 512;

impl Code {
    pub(crate) fn new() -> Self {
        Self {
            bytes: Vec::with_capacity(INITIAL_BYTES),
            instructions: 0,
            sites: Vec::new(),
            pinned: false,
            locals: None,
            dirty: Default::default(),
            constant_time: false,
            linked_block: None,
            instruction: 0,
            failures: Vec::new(),
            base_request: None,
            base_applied: None,
            vectors: Default::default(),
        }
    }
    /// Code that keeps [`super::super::pinned`] guest registers in host
    /// registers.
    pub(crate) fn pinned() -> Self {
        Self {
            pinned: true,
            ..Self::new()
        }
    }
    /// The instructions from here on have `index` in the block's source map.
    pub(crate) fn set_instruction(&mut self, index: usize) {
        self.instruction = index;
    }
    /// Offer the access the next instruction emits `request`
    /// ([`super::super::masked_base`]).
    pub(crate) fn set_base_request(&mut self, request: Option<super::super::masked_base::Request>) {
        self.base_request = request;
        self.base_applied = None;
    }
    /// The plan offered for the access at `address`, which the access then
    /// applies with [`Self::apply_base_plan`].
    pub(in crate::x86_64) fn base_plan(
        &self,
        address: super::super::ImmediateAddress,
    ) -> Option<super::super::masked_base::Plan> {
        self.base_request
            .filter(|request| request.is_for(address))
            .map(|request| request.plan)
    }
    /// The access applied the plan [`Self::base_plan`] gave.
    pub(crate) fn apply_base_plan(&mut self) {
        self.base_applied = self.base_request.take().map(|request| request.plan);
    }
    /// The plan the instruction applied, if any, ending the offer.
    pub(crate) fn finish_base_request(&mut self) -> Option<super::super::masked_base::Plan> {
        self.base_request = None;
        self.base_applied.take()
    }
    /// Fail the instruction being emitted if `failed` holds: a jump never
    /// taken costs no more than the mask a guest range check would pay for,
    /// and needs no address in RAX.
    pub(crate) fn check_failure(&mut self, failed: super::Cond) {
        assert!(
            !self.vectors.has_dirty(),
            "a check failure with V registers only in XMM"
        );
        let jump = self.asm().jcc(failed);
        self.failures.push((jump, self.instruction, self.dirty));
    }
    /// Place the check failures: a UD2 each, whose site names its instruction.
    pub(crate) fn place_failures(&mut self) {
        for (jump, instruction, dirty) in core::mem::take(&mut self.failures) {
            let mut asm = self.asm();
            asm.bind(jump);
            asm.guest_check_of(instruction);
            if let Some(site) = self.sites.last_mut() {
                site.locals = dirty;
            }
        }
    }
    /// Whether emitters must keep to data operand independent instructions.
    pub(crate) const fn constant_time(&self) -> bool {
        self.constant_time
    }
    /// Keep to data operand independent instructions, or not.
    pub(crate) fn set_constant_time(&mut self, constant_time: bool) {
        self.constant_time = constant_time;
    }
    /// Whether this code keeps guest registers in host registers.
    pub(crate) const fn is_pinned(&self) -> bool {
        self.pinned
    }
    /// The host register that holds `register` in this code.
    pub(crate) fn host(&self, register: crate::aarch64::Register) -> Option<super::Gpr> {
        self.pinned
            .then(|| super::super::pinned::host_or_local(register, self.locals))?
    }
    /// Keep guest registers in host registers from here on, or not.
    pub(crate) fn set_pinned(&mut self, pinned: bool) {
        self.pinned = pinned;
    }
    /// Map temporary guest registers only while emitting a pure pinned span.
    pub(in crate::x86_64) fn set_local_registers(
        &mut self,
        locals: Option<super::super::pinned::LocalRegisters>,
    ) {
        debug_assert!(self.pinned);
        self.locals = locals;
    }
    /// The locals a fault in what follows recovers from their hosts.
    pub(in crate::x86_64) fn set_dirty_locals(&mut self, dirty: super::super::pinned::Dirty) {
        self.dirty = dirty;
    }
    /// Give the exits from here on a link slot, in the block at `pc`.
    pub(crate) fn set_linked_block(&mut self, pc: u64) {
        self.linked_block = Some(pc);
    }
    /// An encoder that appends here and adds to this count.
    pub(crate) fn asm(&mut self) -> Asm<'_> {
        let guarded = self.vectors.occupied();
        let vectors_dirty = self.vectors.has_dirty();
        Asm {
            code: &mut self.bytes,
            instructions: 0,
            shared: Some(&mut self.instructions),
            sites: Some(&mut self.sites),
            pinned: self.pinned,
            locals: self.locals,
            dirty: self.dirty,
            linked_block: self.linked_block,
            guarded,
            vectors_dirty,
        }
    }
    /// [`Self::asm`] for the vector cache's own moves, which may write the
    /// pool registers that hold guest vectors.
    pub(in crate::x86_64) fn asm_unguarded(&mut self) -> Asm<'_> {
        let mut asm = self.asm();
        asm.guarded = 0;
        asm
    }
    /// A typed recipe that emits exactly one guest memory instruction.
    pub(crate) fn guest_access(&mut self, emit: impl FnOnce(&mut Self)) {
        assert!(
            !self.vectors.has_dirty(),
            "a guest access with V registers only in XMM"
        );
        let start = self.bytes.len();
        let count = self.instructions;
        emit(self);
        assert_eq!(
            self.instructions,
            count + 1,
            "one exact guest memory instruction"
        );
        let end = self.bytes.len();
        assert!(
            self.sites
                .last()
                .is_none_or(|previous| previous.end <= start),
            "fault sites do not nest"
        );
        self.sites.push(EmissionSite {
            start,
            end,
            site: crate::block::RecoverySite::guest_access(),
            stack: None,
            source: None,
            locals: self.dirty,
        });
    }
    pub(crate) fn sites(&self) -> &[EmissionSite] {
        &self.sites
    }
    /// The instructions emitted through [`Self::asm`].
    pub(crate) const fn instructions(&self) -> usize {
        self.instructions
    }
    pub(crate) fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
    /// Append `other`'s bytes and instructions.
    pub(crate) fn append(&mut self, other: Code) {
        debug_assert!(other.locals.is_none());
        let base = self.bytes.len();
        self.sites
            .extend(other.sites.into_iter().map(|site| EmissionSite {
                start: base + site.start,
                end: base + site.end,
                ..site
            }));
        self.failures.extend(
            other
                .failures
                .into_iter()
                .map(|(jump, instruction, dirty)| {
                    let moved = super::Patch {
                        at: base + jump.at,
                        ..jump
                    };
                    (moved, instruction, dirty)
                }),
        );
        self.bytes.extend(other.bytes);
        self.instructions += other.instructions;
    }
    /// The current emission boundary, for splitting or truncating this buffer.
    pub(crate) fn checkpoint(&self) -> Checkpoint {
        Checkpoint {
            bytes: self.bytes.len(),
            instructions: self.instructions,
        }
    }
    /// Record complete instructions emitted since this buffer's checkpoint.
    pub(crate) fn span_since(&self, start: Checkpoint) -> Span {
        let end = self.checkpoint();
        assert!(start.bytes <= end.bytes && start.instructions <= end.instructions);
        Span { start, end }
    }
    /// Copy a fragment from its unmodified source buffer. Internal branches and
    /// instruction-associated data move together; no byte decoding is involved.
    pub(crate) fn append_span(&mut self, source: &Code, span: Span) {
        assert!(
            span.end.bytes <= source.bytes.len() && span.end.instructions <= source.instructions
        );
        self.bytes
            .extend_from_slice(&source.bytes[span.start.bytes..span.end.bytes]);
        for site in &source.sites {
            if site.start < span.end.bytes && site.end > span.start.bytes {
                assert!(
                    site.start >= span.start.bytes && site.end <= span.end.bytes,
                    "span preserves complete faulting instructions"
                );
                let base = self.bytes.len() - (span.end.bytes - span.start.bytes);
                self.sites.push(EmissionSite {
                    start: base + site.start - span.start.bytes,
                    end: base + site.end - span.start.bytes,
                    ..*site
                });
            }
        }
        self.instructions += span.instructions();
    }
    /// Replace a fixed-size field without changing buffer length or count.
    pub(crate) fn patch<const N: usize>(&mut self, offset: usize, bytes: [u8; N]) {
        let end = offset
            .checked_add(N)
            .expect("patch field offset does not overflow");
        self.bytes
            .get_mut(offset..end)
            .expect("patch field lies in emitted code")
            .copy_from_slice(&bytes);
    }
    /// Move the instructions emitted after `at` into a separate buffer.
    pub(crate) fn split_off(&mut self, at: Checkpoint) -> Self {
        debug_assert!(
            self.locals.is_none(),
            "split after local register settlement"
        );
        assert!(at.bytes <= self.bytes.len() && at.instructions <= self.instructions);
        assert!(
            self.sites
                .iter()
                .all(|site| site.end <= at.bytes || site.start >= at.bytes),
            "split preserves complete faulting instructions"
        );
        let sites = self
            .sites
            .extract_if(.., |site| site.start >= at.bytes)
            .map(|site| EmissionSite {
                start: site.start - at.bytes,
                end: site.end - at.bytes,
                ..site
            })
            .collect();
        let failures = self
            .failures
            .extract_if(.., |(jump, _, _)| jump.at >= at.bytes)
            .map(|(jump, instruction, dirty)| {
                let moved = super::Patch {
                    at: jump.at - at.bytes,
                    ..jump
                };
                (moved, instruction, dirty)
            })
            .collect();
        let tail = Self {
            bytes: self.bytes.split_off(at.bytes),
            instructions: self.instructions - at.instructions,
            sites,
            pinned: self.pinned,
            locals: None,
            dirty: Default::default(),
            constant_time: self.constant_time,
            linked_block: self.linked_block,
            instruction: self.instruction,
            failures,
            base_request: None,
            base_applied: None,
            vectors: Default::default(),
        };
        self.instructions = at.instructions;
        tail
    }
    /// Discard the instructions emitted after `at`.
    pub(crate) fn truncate(&mut self, at: Checkpoint) {
        assert!(at.bytes <= self.bytes.len() && at.instructions <= self.instructions);
        assert!(
            self.sites
                .iter()
                .all(|site| site.end <= at.bytes || site.start >= at.bytes),
            "truncate preserves complete faulting instructions"
        );
        self.sites.retain(|site| site.end <= at.bytes);
        self.failures.retain(|(jump, _, _)| jump.at < at.bytes);
        self.bytes.truncate(at.bytes);
        self.instructions = at.instructions;
    }
    /// Remove one complete instruction whose bytes the caller identified.
    pub(crate) fn remove_instruction(&mut self, range: core::ops::Range<usize>) {
        assert!(!range.is_empty());
        let removed = range.len();
        // Sites and check jumps are appended in code order. Forwarding
        // removes an instruction near the end, so visit only the suffix.
        let mut removed_site = None;
        for (index, site) in self.sites.iter_mut().enumerate().rev() {
            if site.end <= range.start {
                break;
            }
            if site.start >= range.end {
                site.start -= removed;
                site.end -= removed;
            } else {
                assert!(
                    site.start == range.start && site.end == range.end,
                    "remove one complete faulting instruction"
                );
                removed_site = Some(index);
                break;
            }
        }
        if let Some(index) = removed_site {
            self.sites.remove(index);
        }
        for (jump, _, _) in self.failures.iter_mut().rev() {
            if jump.at < range.end {
                break;
            }
            jump.at -= removed;
        }
        self.bytes.drain(range);
        self.instructions = self
            .instructions
            .checked_sub(1)
            .expect("counted instruction");
    }
    /// Remove every byte and instruction.
    pub(crate) fn clear(&mut self) {
        self.bytes.clear();
        self.instructions = 0;
        self.sites.clear();
        self.instruction = 0;
        self.failures.clear();
    }
}

impl PartialEq<[u8]> for Code {
    fn eq(&self, other: &[u8]) -> bool {
        self.bytes == other
    }
}

impl PartialEq<Vec<u8>> for Code {
    fn eq(&self, other: &Vec<u8>) -> bool {
        self.bytes == *other
    }
}

impl<const N: usize> PartialEq<[u8; N]> for Code {
    fn eq(&self, other: &[u8; N]) -> bool {
        self.bytes == other
    }
}

impl PartialEq<&[u8]> for Code {
    fn eq(&self, other: &&[u8]) -> bool {
        self.bytes == *other
    }
}

impl PartialEq<Code> for [u8] {
    fn eq(&self, other: &Code) -> bool {
        self == other.bytes
    }
}

impl PartialEq<Code> for &[u8] {
    fn eq(&self, other: &Code) -> bool {
        *self == other.bytes
    }
}

impl PartialEq<Code> for Vec<u8> {
    fn eq(&self, other: &Code) -> bool {
        *self == other.bytes
    }
}

impl<const N: usize> PartialEq<Code> for [u8; N] {
    fn eq(&self, other: &Code) -> bool {
        self[..] == other.bytes
    }
}

impl From<Code> for Vec<u8> {
    fn from(code: Code) -> Self {
        code.bytes
    }
}

impl<const N: usize> TryFrom<Code> for [u8; N] {
    type Error = Vec<u8>;
    fn try_from(code: Code) -> Result<Self, Vec<u8>> {
        code.bytes.try_into()
    }
}

impl<'a> IntoIterator for &'a Code {
    type Item = &'a u8;
    type IntoIter = core::slice::Iter<'a, u8>;
    fn into_iter(self) -> Self::IntoIter {
        self.bytes.iter()
    }
}

impl IntoIterator for Code {
    type Item = u8;
    type IntoIter = alloc::vec::IntoIter<u8>;
    fn into_iter(self) -> Self::IntoIter {
        self.bytes.into_iter()
    }
}

impl core::ops::Deref for Code {
    type Target = Vec<u8>;
    fn deref(&self) -> &Vec<u8> {
        &self.bytes
    }
}
