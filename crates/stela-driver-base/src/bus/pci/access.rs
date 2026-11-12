//! Fallible PCI configuration access, independent of the owned transport.
use super::assignment;
use crate::bus::mmio::owned::Error as MappingError;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConfigFailure {
    Window,
    Address,
    PhysicalSpan,
    CacheAttribute,
    Protection,
    CpuAddress,
    Access {
        offset: u16,
        width: usize,
    },
    Mapping(MappingError),
    Ownership(assignment::Error),
    ForeignGrant,
    UnqualifiedControl,
    /// Actual external Command admission refused. Its full native/core first
    /// cause remains retained by the original operation validator.
    CommandAdmission,
    /// Negative observation of the SAME producer publishing its first cause.
    /// This is not a completed physical cause and must not replace one.
    FaultPublishing,
    /// Before-effect/retained completion contention, not a physical cause.
    Busy,
    /// Caught interruption of this actual bounded configuration transport.
    Interrupted,
    BusMasterSealed,
    RoutingSealed,
    Readback {
        offset: u16,
        width: usize,
        expected: u32,
        observed: u32,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConfigEffect {
    /// No volatile configuration write was issued by this failed access.
    /// This does not assert that a vendor register read had no side effect.
    None,
    /// A volatile write was issued before validation failed. Keep the exact
    /// owner; this error supplies no permission to retry or roll back.
    WriteMayHaveApplied { offset: u16, width: u8 },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConfigError {
    pub reason: ConfigFailure,
    pub effect: ConfigEffect,
}
impl ConfigError {
    pub fn is_observation(self) -> bool {
        matches!(
            self.reason,
            ConfigFailure::FaultPublishing
                | ConfigFailure::Busy
                | ConfigFailure::Mapping(MappingError::Busy)
                | ConfigFailure::Ownership(assignment::Error::Busy { .. })
                | ConfigFailure::Mapping(MappingError::Assignment(assignment::Error::Busy { .. }))
        )
    }
    pub const fn before_effect(reason: ConfigFailure) -> Self {
        Self {
            reason,
            effect: ConfigEffect::None,
        }
    }
    pub const fn write_may_have_applied(offset: u16, width: u8, reason: ConfigFailure) -> Self {
        Self {
            reason,
            effect: ConfigEffect::WriteMayHaveApplied { offset, width },
        }
    }
}
/// A configuration register's name for the boot log.
pub struct Register(pub u16);
impl core::fmt::Display for Register {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let name = match self.0 {
            0x00 => "vendor and device ID",
            0x04 => "COMMAND",
            0x06 => "STATUS",
            0x08 => "class and revision",
            0x0c => "header type",
            0x18 => "bus numbers",
            _ => return write!(f, "register {:#x}", self.0),
        };
        write!(f, "{name} register ({:#04x})", self.0)
    }
}

/// A plain sentence: what was accessed, what was expected and read, and
/// whether a write may have taken effect.
impl core::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.reason {
            ConfigFailure::Readback {
                offset,
                width,
                expected,
                observed,
            } => {
                let digits = width * 2;
                write!(
                    f,
                    "the {} reads {observed:#0w$x} where {expected:#0w$x} was expected",
                    Register(offset),
                    w = digits + 2,
                )?;
            }
            ConfigFailure::Access { offset, width } => write!(
                f,
                "a {width}-byte access at {offset:#x} is misaligned or outside configuration space"
            )?,
            ConfigFailure::Mapping(error) => {
                write!(f, "the configuration space could not be mapped ({error:?})")?
            }
            ConfigFailure::UnqualifiedControl => {
                f.write_str("a control is in a state Stela does not change")?
            }
            reason => write!(f, "the configuration access failed ({reason:?})")?,
        }
        match self.effect {
            ConfigEffect::None => Ok(()),
            ConfigEffect::WriteMayHaveApplied { offset, .. } => {
                write!(f, " after Stela wrote the {}", Register(offset))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;

    #[test]
    fn a_readback_failure_reads_as_a_sentence() {
        let error = ConfigError::write_may_have_applied(
            4,
            2,
            ConfigFailure::Readback {
                offset: 4,
                width: 2,
                expected: 0x400,
                observed: 0,
            },
        );
        assert_eq!(
            format!("{error}"),
            "the COMMAND register (0x04) reads 0x0000 where 0x0400 was expected after \
             Stela wrote the COMMAND register (0x04)"
        );
        let error = ConfigError::before_effect(ConfigFailure::Readback {
            offset: 0x52,
            width: 2,
            expected: 0x80,
            observed: 0x81,
        });
        assert_eq!(
            format!("{error}"),
            "the register 0x52 reads 0x0081 where 0x0080 was expected"
        );
    }
}
