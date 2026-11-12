//! System-mode execution: one ARM PE with EL0, EL1 and EL2 running native
//! blocks from the shared compiler, with the MMU, exceptions, generic timers
//! and System registers implemented here and the x86 mechanisms supplied by a
//! [`Host`].
pub mod amu;
pub mod amx;
pub mod aon;
mod capabilities;
pub mod cpu;
pub mod ctrr;
pub mod debug;
pub mod engine;
pub mod exception;
pub mod fast_ipi;
pub mod hacr;
pub mod identification;
pub mod impdef;
pub mod mmu;
pub mod pmu;
pub mod pointer_keys;
pub mod registers;
pub mod shared;
pub mod streaming;
pub mod system;
pub mod tag_storage;
pub mod tags;
pub mod timer;
pub mod vgic;

pub use cpu::{Cpu, Exception};
#[cfg(test)]
pub(crate) use engine::test_preparation;
pub use engine::{
    ConsumedPair, Description, DeviceError, DeviceRegister, Devices, Event, FaultCause, Host,
    InstallError, Machine, NativeCode, NativeExit, NativeFault, NativeFaultOrigin, NativeView,
    NoDevices, Offer, PeState, PeerState, Preparation, Prepared, Refused, Regime, Requirement,
    RequirementKind, Reset, StaleCode, StaleOrigin, Statistics, Stop, Transition, Translations,
    window,
};
pub use registers::{CpuProfile, Lines};
#[cfg(test)]
pub(crate) use stela_contracts::chip::test_chip;
use stela_contracts::{Chip, caches};
