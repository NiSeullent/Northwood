//! The cases through Stela: each compiled by `stela_compiler` and run by the
//! engine's block runtime, with its FP, PAC and helper exits, at EL0 in the
//! process profile with software PAuth.
use super::record::{Case, Ending, MEMORY, State};
use anyhow::{Context, Result};
use stela_compiler::{GuestAddress, GuestState, TargetCpu};
use stela_engine::blocks::{BlockRuntime, StopReason, ThreadState};
use stela_engine::memory::{MemoryRegionSpec, OwnedMemory};
use stela_engine::pauth::{self, Keys};

const ADDRESS_SPACE: u64 = 0x5ce1;
/// Guest addresses equal host addresses in the direct profile: the buffer
/// is mapped here, at a 64-byte-aligned address as the runner's.
const MEMORY_BASE: u64 = 0x3c00_0000_0000;
const CODE_BASE: u64 = 0x10_0000;
/// Each case's region: its words and the branch that ends it.
const CODE_STRIDE: u64 = 0x200;
/// `B .+4`: ends the case's block; the runtime stops at the next PC.
const BRANCH_NEXT: u32 = 0x1400_0001;
/// Arbitrary process keys: the corpus only strips or signs and
/// authenticates in pairs, which give the same results under any key.
const KEY: [u64; 2] = [0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210];

/// Run every case and return how each ended.
pub fn run(cases: &[Case], target: TargetCpu) -> Result<Vec<Ending>> {
    let cache = tempfile::tempdir()?;
    let mut runtime = BlockRuntime::new_for_target(
        ADDRESS_SPACE,
        cache.path().into(),
        stela_compiler::block::BlockProfile::new(
            stela_compiler::block::Features::POINTER_AUTHENTICATION,
        ),
        target,
    );
    let mut memory = OwnedMemory::map(vec![MemoryRegionSpec {
        guest_address: MEMORY_BASE,
        virtual_bytes: 4096,
        initial_bytes: vec![0; 4096],
        readable: true,
        writable: true,
        executable: false,
    }])
    .context("cannot map the case buffer")?;
    let pauth = pauth::Context::new(
        ADDRESS_SPACE,
        Keys::from_words(KEY, KEY, KEY, KEY).with_generic_words(KEY),
    );
    let mut endings = Vec::with_capacity(cases.len());
    for (index, case) in cases.iter().enumerate() {
        let base = CODE_BASE + index as u64 * CODE_STRIDE;
        let mut words = case.words.clone();
        words.push(BRANCH_NEXT);
        let source = words.iter().flat_map(|word| word.to_le_bytes()).collect();
        runtime.register_code(GuestAddress(base), source)?;
        memory.write(MEMORY_BASE, &case.memory)?;
        let mut thread = ThreadState {
            registers: guest_state(case),
            pc: GuestAddress(base),
        };
        let stop = GuestAddress(base + 4 * words.len() as u64);
        // SAFETY: the corpus's accesses stay inside the mapped buffer, at
        // the offsets its memory registers hold; its code is registered and
        // nothing else mutates the buffer or the state meanwhile.
        let result = unsafe { runtime.run_with_pauth(&mut thread, Some(stop), 64, &pauth) };
        let ending = match result {
            Ok(outcome) => match outcome.reason {
                StopReason::StopAddress => {
                    let mut state = state(&thread.registers);
                    for register in 0..31 {
                        if case.memory_registers >> register & 1 != 0 {
                            state.x[register] = state.x[register].wrapping_sub(MEMORY_BASE);
                        }
                    }
                    let bytes = memory.read_snapshot(MEMORY_BASE, MEMORY)?;
                    Ending::Completed {
                        state: Box::new(state),
                        memory: Box::new(bytes.try_into().unwrap()),
                    }
                }
                // Both are SIGILL to a macOS process at the instruction.
                StopReason::FloatingPointTrapped { instruction_pc, .. } => Ending::Signal {
                    name: "SIGILL",
                    index: (instruction_pc.0 - base) / 4,
                },
                StopReason::Unsupported { .. } => Ending::Signal {
                    name: "SIGILL",
                    index: (thread.pc.0 - base) / 4,
                },
                reason => Ending::Other(format!("{reason:?}")),
            },
            Err(error) => Ending::Other(format!("{error:#}")),
        };
        endings.push(ending);
    }
    Ok(endings)
}

fn guest_state(case: &Case) -> GuestState {
    let mut registers = GuestState {
        x: case.state.x,
        nzcv: case.state.nzcv as u32,
        v: case.state.v,
        fpcr: case.state.fpcr as u32,
        fpsr: case.state.fpsr as u32,
        ..Default::default()
    };
    for register in 0..31 {
        if case.memory_registers >> register & 1 != 0 {
            registers.x[register] = registers.x[register].wrapping_add(MEMORY_BASE);
        }
    }
    registers
}

fn state(registers: &GuestState) -> State {
    State {
        x: registers.x,
        nzcv: u64::from(registers.nzcv),
        fpcr: u64::from(registers.fpcr),
        fpsr: u64::from(registers.fpsr),
        v: registers.v,
    }
}
