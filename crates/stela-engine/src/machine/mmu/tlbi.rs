//! The exact scope of a TLB maintenance instruction (pseudocode
//! AArch64.TLBI_*, TLBIRange, AArch64.EffectiveBroadcast and TLBIMatch) for
//! the profile: no FEAT_XS, FEAT_D128, FEAT_LPA2, FEAT_RME or FEAT_TLBID.
//!
//! Removing more than the scope is always permitted, so a cache that keeps
//! no translation context may discard everything instead, and one that
//! holds a single context's translations may discard what [`Scope::reach`]
//! names.

use super::Regime;
use crate::machine::system::{Broadcast, Regime as Named, Tlbi, TlbiKind, TlbiLevel};

/// What a TLBI reaches of a PE that translates another context: at most
/// global entries in the lower half, at `addresses`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Private {
    pub regime: Regime,
    pub asid: u16,
    pub addresses: Option<(u64, u64)>,
}

/// A cached translation with the context it was made in (pseudocode
/// TLBRecord).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Entry {
    pub regime: Regime,
    /// The VMID of an EL1&0 entry made while EL2 is enabled (UseVMID).
    pub vmid: Option<u8>,
    /// The ASID of a non-global entry; `None` for a global entry and in the
    /// EL2 regime, which has no ASIDs.
    pub asid: Option<u16>,
    /// The input address: a VA, or an IPA for a stage-2-only entry.
    pub address: u64,
    /// log2 of the bytes the entry maps.
    pub size_bits: u32,
    /// log2 of the translation granule (12 or 14).
    pub granule_bits: u32,
    /// The lookup level of the descriptor it came from.
    pub level: i8,
    /// The entry holds stage-1 and stage-2 information respectively.
    pub stage1: bool,
    pub stage2: bool,
    /// A walk-cache entry for a table descriptor rather than a leaf.
    pub table: bool,
}

/// The PE state a TLBI's effect depends on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Context {
    /// The EL the instruction executes at.
    pub el: u8,
    /// EL2 is implemented and enabled (EL2Enabled).
    pub el2: bool,
    /// HCR_EL2.E2H and TGE.
    pub e2h: bool,
    pub tge: bool,
    /// HCR_EL2.FB: EL1 TLBIs are broadcast in the Inner Shareable domain.
    pub fb: bool,
}

/// The set of entries one TLBI removes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Scope {
    kind: TlbiKind,
    regime: Regime,
    /// The VMID compared, for the forms that compare one.
    vmid: Option<u8>,
    asid: u16,
    /// The first address, and for a range form the address after the last.
    address: u64,
    end: u64,
    /// TTL[3:0] of a VA or IPA form, or TG:TTL[1:0] of a range form.
    ttl: u8,
    level: TlbiLevel,
    /// The domain the operation reaches after HCR_EL2.FB. One PE makes every
    /// domain equivalent to the local one.
    pub broadcast: Broadcast,
    /// A range form with TG 0b00 removes nothing.
    valid: bool,
    /// A range of every entry overlapping it, whatever its granule and
    /// level hint: what [`Scope::merged`] makes of several TLBIs.
    merged: bool,
}

/// VA[55:0] and IPA[55:0] as compared by the matching.
const ADDRESS_BITS: u64 = (1 << 56) - 1;
const PAGE_SHIFT: u64 = 12;
const VA_OPERAND: u64 = (1 << 44) - 1;
const IPA_OPERAND: u64 = (1 << 40) - 1;
const RANGE_BASE: u64 = (1 << 37) - 1;

fn regime(named: Named) -> Regime {
    match named {
        Named::EL10 => Regime::El10,
        Named::EL20 => Regime::El20,
        Named::EL2 => Regime::El2,
    }
}

/// The granule a TG field selects; `None` for 0b00.
fn granule(tg: u8) -> Option<u32> {
    match tg & 3 {
        1 => Some(12),
        2 => Some(14),
        3 => Some(16),
        _ => None,
    }
}

/// TTL values that hint nothing (pseudocode ResTLBITTL without FEAT_LPA2).
fn reserved_ttl(ttl: u8) -> bool {
    matches!(ttl & 0xf, 0b0000..=0b0100 | 0b1000 | 0b1001 | 0b1100)
}

/// Range TTL values that hint nothing (pseudocode ResTLBIRTTL).
fn reserved_range_ttl(tg: u8, ttl: u8) -> bool {
    match ttl & 3 {
        0 => true,
        1 => granule(tg) == Some(14),
        _ => false,
    }
}

/// The first and end addresses of a range operand (pseudocode TLBIRange
/// without FEAT_LPA2): BaseADDR in units of the granule, sign-extended from
/// its top bit, and (NUM + 1) * 2^(5 * SCALE + 1) granules.
fn range(operand: u64) -> Option<(u8, u64, u64)> {
    let tg = (operand >> 46) as u8 & 3;
    let bits = granule(tg)?;
    let scale = (operand >> 44) & 3;
    let num = (operand >> 39) & 0x1f;
    let base = operand & RANGE_BASE;
    let top = bits + 36;
    let start = ((base << bits) as i64) << (63 - top) >> (63 - top);
    let start = start as u64;
    let end = start.wrapping_add((num + 1) << (5 * scale + 1 + u64::from(bits)));
    // A range that crosses bit 52 ends at the top of its half.
    let end = if (end ^ start) >> 52 & 1 != 0 {
        if start >> 52 & 1 != 0 {
            u64::MAX
        } else {
            (1 << 52) - 1
        }
    } else {
        end
    };
    Some((tg, start, end))
}

impl Scope {
    /// The scope of `tlbi` as the generated accessor decoded it, executed in
    /// `context`.
    pub fn new(tlbi: &Tlbi, context: Context) -> Self {
        let in_host = context.e2h && context.tge;
        let named = regime(tlbi.regime);
        let regime = match tlbi.kind {
            TlbiKind::VA | TlbiKind::RVA => match named {
                Regime::El10 if in_host => Regime::El20,
                Regime::El2 if context.e2h => Regime::El20,
                other => other,
            },
            TlbiKind::ASID | TlbiKind::VAA | TlbiKind::RVAA | TlbiKind::VMALL if in_host => {
                Regime::El20
            }
            _ => named,
        };
        let vmid = match tlbi.kind {
            TlbiKind::IPAS2 | TlbiKind::RIPAS2 | TlbiKind::VMALLS12 => {
                Some(tlbi.vmid.unwrap_or(0) as u8)
            }
            TlbiKind::ALL => None,
            _ if regime == Regime::El10 && context.el2 => Some(tlbi.vmid.unwrap_or(0) as u8),
            _ => None,
        };
        let operand = tlbi.operand;
        let mut scope = Self {
            kind: tlbi.kind,
            regime,
            vmid,
            asid: (operand >> 48) as u16,
            address: 0,
            end: 0,
            ttl: (operand >> 44) as u8 & 0xf,
            level: tlbi.level,
            broadcast: effective_broadcast(tlbi.broadcast, context),
            valid: true,
            merged: false,
        };
        match tlbi.kind {
            TlbiKind::VA | TlbiKind::VAA => {
                scope.address = (operand & VA_OPERAND) << PAGE_SHIFT;
            }
            TlbiKind::IPAS2 => scope.address = (operand & IPA_OPERAND) << PAGE_SHIFT,
            TlbiKind::RVA | TlbiKind::RVAA | TlbiKind::RIPAS2 => match range(operand) {
                Some((tg, start, end)) => {
                    scope.ttl = tg << 2 | (operand >> 37) as u8 & 3;
                    scope.address = start;
                    scope.end = end;
                }
                None => scope.valid = false,
            },
            _ => {}
        }
        scope
    }

    fn vmid_matches(self, entry: &Entry) -> bool {
        self.vmid == entry.vmid
    }

    /// The entry's VMID is either unused or the one this scope names.
    fn vmid_covers(self, entry: &Entry) -> bool {
        entry.vmid.is_none() || entry.vmid == self.vmid
    }

    /// The address bits above the entry's size match.
    fn address_matches(self, entry: &Entry) -> bool {
        let mask = ADDRESS_BITS & !((1u64 << entry.size_bits) - 1);
        self.address & mask == entry.address & mask
    }

    fn ttl_matches(self, entry: &Entry) -> bool {
        reserved_ttl(self.ttl)
            || granule(self.ttl >> 2) == Some(entry.granule_bits)
                && i8::try_from(self.ttl & 3) == Ok(entry.level)
    }

    fn range_matches(self, entry: &Entry) -> bool {
        let tg = self.ttl >> 2;
        let mask = (1u64 << entry.size_bits) - 1;
        let first = entry.address & !mask & ADDRESS_BITS;
        let last = (entry.address | mask) & ADDRESS_BITS;
        let hinted = self.merged
            || granule(tg) == Some(entry.granule_bits)
                && (reserved_range_ttl(tg, self.ttl)
                    || i8::try_from(self.ttl & 3) == Ok(entry.level));
        hinted && self.address & ADDRESS_BITS <= last && self.end & ADDRESS_BITS > first
    }

    fn level_matches(self, entry: &Entry) -> bool {
        self.level == TlbiLevel::Any || !entry.table
    }

    /// Whether the TLBI removes `entry` (pseudocode TLBIMatch).
    pub fn covers(self, entry: &Entry) -> bool {
        if !self.valid {
            return false;
        }
        let same_regime = self.regime == entry.regime;
        let first_stage = entry.stage1 && same_regime && self.vmid_matches(entry);
        let asid = match entry.asid {
            Some(asid) => asid == self.asid,
            None => true,
        };
        match self.kind {
            TlbiKind::ALL => {
                let el2 = |regime| matches!(regime, Regime::El20 | Regime::El2);
                same_regime || el2(self.regime) && el2(entry.regime)
            }
            TlbiKind::VMALL => first_stage,
            TlbiKind::VMALLS12 => same_regime && self.vmid_covers(entry),
            TlbiKind::ASID => first_stage && entry.asid == Some(self.asid),
            TlbiKind::VA => {
                first_stage
                    && asid
                    && self.address_matches(entry)
                    && self.ttl_matches(entry)
                    && self.level_matches(entry)
            }
            TlbiKind::VAA => {
                first_stage
                    && self.address_matches(entry)
                    && self.ttl_matches(entry)
                    && self.level_matches(entry)
            }
            TlbiKind::RVA => {
                first_stage && asid && self.range_matches(entry) && self.level_matches(entry)
            }
            TlbiKind::RVAA => first_stage && self.range_matches(entry) && self.level_matches(entry),
            TlbiKind::IPAS2 | TlbiKind::RIPAS2 => {
                let stage2_only = !entry.stage1 && entry.stage2 && same_regime;
                let address = if self.kind == TlbiKind::IPAS2 {
                    self.address_matches(entry) && self.ttl_matches(entry)
                } else {
                    self.range_matches(entry)
                };
                stage2_only && self.vmid_covers(entry) && address && self.level_matches(entry)
            }
        }
    }
}

/// The one context a cache that keeps no context per entry holds
/// translations of: the regime the PE translates in now, under the VMID it
/// uses now (the one every TLBI that names a VMID takes from VTTBR_EL2), with
/// non-global entries of one ASID.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Current {
    pub regime: Regime,
    /// The ASID of its non-global entries; `None` in the EL2 regime.
    pub asid: Option<u16>,
}

/// What a TLBI removes from a cache of the [`Current`] context's
/// stage-1 (or combined) translations, as a superset of its scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reach {
    Nothing,
    /// The entries whose input range holds this VA, canonical.
    Containing(u64),
    /// The entries whose input range overlaps `first..=last`, canonical VAs.
    Overlapping {
        first: u64,
        last: u64,
    },
    All,
}

impl Reach {
    /// Whether it removes nothing of the upper half (VA bit 55 set, TTBR1's
    /// range): it names no entry or only lower-half addresses.
    pub fn lower_only(self) -> bool {
        const UPPER: u64 = 1 << 55;
        match self {
            Self::Nothing => true,
            Self::Containing(address) => address & UPPER == 0,
            Self::Overlapping { first, last } => (first | last) & UPPER == 0,
            Self::All => false,
        }
    }
}

/// What a TLBI removes from a cache that keeps translations of many
/// contexts (regimes, ASIDs) without telling them apart, as a superset of
/// its scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Everywhere {
    /// The entries whose input range holds these VAs, whatever their
    /// context: the VA forms ([`Reach::Containing`] or
    /// [`Reach::Overlapping`]), or [`Reach::Nothing`].
    Addresses(Reach),
    /// The non-global entries of the ASID named (TLBI ASIDE1): a global
    /// entry or a non-global entry of another ASID stays (Arm `TLBIMatch`).
    NonGlobal(u16),
    /// Every entry.
    All,
}

/// VA[55:0] with bit 55 copied above: the address of either half.
pub(in crate::machine) fn canonical(address: u64) -> u64 {
    (((address << 8) as i64) >> 8) as u64
}

impl Scope {
    /// What this TLBI removes from a cache that keeps every context's
    /// translations ([`Everywhere`]): for the VA forms, of any regime or
    /// ASID, the entries at their address or range; for ASIDE1, the
    /// non-global entries; for the other forms, everything.
    pub fn everywhere(self) -> Everywhere {
        if !self.valid {
            return Everywhere::Addresses(Reach::Nothing);
        }
        match self.kind {
            TlbiKind::VA | TlbiKind::VAA => {
                Everywhere::Addresses(Reach::Containing(canonical(self.address)))
            }
            TlbiKind::RVA | TlbiKind::RVAA => Everywhere::Addresses(self.overlapping()),
            TlbiKind::ASID => Everywhere::NonGlobal(self.asid),
            _ => Everywhere::All,
        }
    }

    /// The ASID whose non-global entries this TLBI alone reaches: the VA and
    /// ASID forms that name one, whose VA forms reach global entries too.
    pub fn named_asid(self) -> Option<u16> {
        matches!(self.kind, TlbiKind::VA | TlbiKind::RVA | TlbiKind::ASID).then_some(self.asid)
    }

    /// The context this TLBI is private to, if it reaches nothing else but
    /// global entries in the lower half: a kernel's global upper half is
    /// every PE's, so the VA forms there are not.
    pub fn private(self) -> Option<Private> {
        if !self.valid || self.regime == Regime::El2 {
            return None;
        }
        let addresses = match self.kind {
            TlbiKind::ASID => None,
            TlbiKind::VA => Some((canonical(self.address), canonical(self.address))),
            TlbiKind::RVA => match self.overlapping() {
                Reach::Overlapping { first, last } => Some((first, last)),
                _ => return None,
            },
            _ => return None,
        };
        addresses
            .is_none_or(|(first, last)| (first | last) >> 63 == 0)
            .then_some(Private {
                regime: self.regime,
                asid: self.asid,
                addresses,
            })
    }

    /// The canonical first and last VAs a VA or range form names, which it
    /// removes the entries holding (VA) or overlapping (range).
    pub fn span(self) -> Option<(u64, u64)> {
        match self.kind {
            TlbiKind::VA | TlbiKind::VAA if self.valid => {
                let address = canonical(self.address);
                Some((address, address))
            }
            TlbiKind::RVA | TlbiKind::RVAA if self.valid => match self.overlapping() {
                Reach::Overlapping { first, last } => Some((first, last)),
                _ => None,
            },
            _ => None,
        }
    }

    /// One scope that removes at least what `self` and `other` remove
    /// (Arm permits removing more), when both are VA or range forms of one
    /// regime, VMID and domain whose spans lie in one half of the address
    /// space within `gap` bytes of each other: a range of every entry
    /// overlapping both spans and the bytes between, whatever its granule
    /// or level hint (both forms remove at least the entries holding or
    /// overlapping their own span), at any level unless both name the last
    /// level alone, of their ASID if both name one and the same, else of
    /// every ASID.
    pub fn merged(self, other: Self, gap: u64) -> Option<Self> {
        const UPPER: u64 = 1 << 55;
        let ((first, last), (other_first, other_last)) = (self.span()?, other.span()?);
        let same = self.regime == other.regime
            && self.vmid == other.vmid
            && self.broadcast == other.broadcast
            && (first ^ other_first) & UPPER == 0;
        let apart = first.max(other_first).saturating_sub(last.min(other_last));
        if !same || apart > gap {
            return None;
        }
        let (first, last) = (first.min(other_first), last.max(other_last));
        let named = |scope: Self| matches!(scope.kind, TlbiKind::VA | TlbiKind::RVA);
        let one_asid = named(self) && named(other) && self.asid == other.asid;
        Some(Self {
            kind: if one_asid {
                TlbiKind::RVA
            } else {
                TlbiKind::RVAA
            },
            address: first,
            // The address after the last; the end of the address space is
            // its own marker ([`Scope::overlapping`]).
            end: last.saturating_add(1),
            ttl: 0,
            level: if self.level == TlbiLevel::Last && other.level == TlbiLevel::Last {
                TlbiLevel::Last
            } else {
                TlbiLevel::Any
            },
            merged: true,
            ..self
        })
    }

    /// The canonical VAs a range form names.
    fn overlapping(self) -> Reach {
        Reach::Overlapping {
            first: canonical(self.address),
            last: if self.end == u64::MAX {
                u64::MAX
            } else {
                canonical(self.end.wrapping_sub(1))
            },
        }
    }

    /// What this TLBI removes from a cache of `current`'s translations: no
    /// entry of another regime, nothing of ASID forms for another ASID, all
    /// for the forms without an address, and for the address forms the
    /// entries at their address or range whatever their ASID, level or
    /// size. The IPA forms remove only stage-2-only entries
    /// ([`Scope::covers`]); they remove everything here, which the
    /// combined stage-1 and stage-2 entries a VMALLS12E1 must follow make
    /// harmless.
    pub fn reach(self, current: &Current) -> Reach {
        if !self.valid {
            return Reach::Nothing;
        }
        let same_regime = self.regime == current.regime;
        let all = |reaches: bool| if reaches { Reach::All } else { Reach::Nothing };
        match self.kind {
            TlbiKind::ALL => {
                let el2 = |regime| matches!(regime, Regime::El20 | Regime::El2);
                all(same_regime || el2(self.regime) && el2(current.regime))
            }
            TlbiKind::VMALL | TlbiKind::VMALLS12 | TlbiKind::IPAS2 | TlbiKind::RIPAS2 => {
                all(same_regime)
            }
            TlbiKind::ASID => all(same_regime && current.asid == Some(self.asid)),
            TlbiKind::VA | TlbiKind::VAA if same_regime => {
                Reach::Containing(canonical(self.address))
            }
            TlbiKind::RVA | TlbiKind::RVAA if same_regime => self.overlapping(),
            TlbiKind::VA | TlbiKind::VAA | TlbiKind::RVA | TlbiKind::RVAA => Reach::Nothing,
        }
    }
}

/// HCR_EL2.FB makes a local TLBI at EL1 Inner Shareable (pseudocode
/// AArch64.EffectiveBroadcast without FEAT_TLBID).
fn effective_broadcast(broadcast: Broadcast, context: Context) -> Broadcast {
    if broadcast == Broadcast::NSH && context.el == 1 && context.el2 && context.fb {
        Broadcast::ISH
    } else {
        broadcast
    }
}

#[cfg(test)]
mod tests;
