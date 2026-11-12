//! Per-PE implementation blocks: fast IPIs, CoreSight, lockdown, start words,
//! counters and the AIC.

use super::*;
use crate::Result;
use stela_contracts::refuse;

impl AppleDevices {
    /// Apple's fast-IPI registers for PEs with these MPIDR_EL1 values, in PE
    /// order: each PE's cluster and CPU number are its Aff1 and Aff0.
    pub fn add_fast_ipi(&mut self, mpidrs: &[u64]) {
        let cpus = mpidrs
            .iter()
            .map(|&mpidr| Affinity {
                cluster: (mpidr >> 8) as u8,
                cpu: mpidr as u8,
            })
            .collect();
        self.ipi = Some(FastIpi::new(cpus));
    }

    /// The lock pair at the end of one DT-placed CoreSight component page.
    pub fn add_coresight_lock(&mut self, physical: u64) -> Result<(), MapError> {
        let device = self.bus.add(Box::new(SoftwareLock::new()));
        self.bus.map(
            device,
            physical
                .checked_add(coresight::ACCESS)
                .ok_or(MapError::Overflow)?,
            8,
            coresight::ACCESS,
        )
    }

    /// The UTTDBG sampler of PE `cpu`, at its DT-placed UTT region base.
    pub fn add_coresight_sampler(&mut self, cpu: usize, physical: u64) -> Result<(), MapError> {
        let register = physical
            .checked_add(coresight::PC_SAMPLE)
            .ok_or(MapError::Overflow)?;
        let device = self.bus.add(Box::new(PcSampler::default()));
        self.bus.map(device, register, 8, coresight::PC_SAMPLE)?;
        self.pc_samplers.push((cpu, register, device));
        Ok(())
    }

    /// The lockdown registers of a CPU implementation block named `name`,
    /// with CONTROL at `control`, at `range`; a PE's `cpu-impl-reg` also
    /// holds that PE's RVBAR, which the boot firmware locked on `reset`, its
    /// run state CPU_STS and CORE_THRTL_CFG2.
    /// Only those registers are mapped: the block also holds the PE's or
    /// cluster's energy accumulators (`platform::apple::energy`).
    pub fn add_lockdown(
        &mut self,
        name: &'static str,
        control: u64,
        range: Range,
        reset: Option<(usize, u64)>,
    ) -> Result<(), MapError> {
        if let Some((cpu, _)) = reset
            && let Some(existing) = self.reset_vectors.get(cpu).copied().flatten()
        {
            return Err(MapError::Overlap { existing });
        }
        let mut model = Lockdown::new(name, control);
        let mut bindings = vec![(
            range
                .physical
                .checked_add(control)
                .ok_or(MapError::Overflow)?,
            lockdown::BYTES,
            control,
        )];
        if let Some((pe, entry)) = reset {
            model = model.with_reset_vector(entry);
            model.set_running(self.processor_running(pe));
            for offset in lockdown::PE_WORDS {
                let start = range
                    .physical
                    .checked_add(offset)
                    .ok_or(MapError::Overflow)?;
                bindings.push((start, lockdown::WORD_BYTES, offset));
            }
        }
        // Publish the reset-vector owner only after every bank fits.
        let device = self.bus.install(Box::new(model), &bindings)?;
        if let Some((pe, _)) = reset {
            if self.reset_vectors.len() <= pe {
                self.reset_vectors.resize(pe + 1, None);
            }
            self.reset_vectors[pe] = Some(device);
        }
        Ok(())
    }

    /// PE startup words when its original block has no legacy PIO lock pair.
    /// Publish its reset-vector binding only after all three words are mapped.
    pub fn add_processor_registers(
        &mut self,
        cpu: usize,
        range: Range,
        entry: u64,
    ) -> Result<(), MapError> {
        if let Some(existing) = self.reset_vectors.get(cpu).copied().flatten() {
            return Err(MapError::Overlap { existing });
        }
        if !range.physical.is_multiple_of(lockdown::WORD_BYTES) {
            return Err(MapError::Unaligned);
        }
        range
            .physical
            .checked_add(range.bytes)
            .ok_or(MapError::Overflow)?;
        let mut model = Lockdown::processor(entry);
        model.set_running(self.processor_running(cpu));
        let mut bindings = Vec::new();
        for offset in lockdown::PE_WORDS {
            if offset
                .checked_add(lockdown::WORD_BYTES)
                .is_none_or(|end| end > range.bytes)
            {
                return Err(MapError::Overflow);
            }
            let physical = range
                .physical
                .checked_add(offset)
                .ok_or(MapError::Overflow)?;
            bindings.push((physical, lockdown::WORD_BYTES, offset));
        }
        let device = self.bus.install(Box::new(model), &bindings)?;
        if self.reset_vectors.len() <= cpu {
            self.reset_vectors.resize(cpu + 1, None);
        }
        self.reset_vectors[cpu] = Some(device);
        Ok(())
    }

    /// The PMGR's CPU start registers for `clusters` clusters at `physical`,
    /// releasing the PEs with these MPIDR_EL1 values, in PE order, at the
    /// RVBARs of their `cpu-impl-reg` blocks.
    pub fn add_cpu_start(&mut self, physical: u64, clusters: u8, mpidrs: &[u64]) -> Result<()> {
        if self.cpu_start.is_some() {
            return Err(refuse!("CPU start owner is already bound"));
        }
        let device = self
            .bus
            .install(
                Box::new(CpuStart::new(clusters)),
                &[(physical, stela_soc_data::cpustart::bytes(clusters), 0)],
            )
            .map_err(|error| refuse!("CPU start placement: {error}", ?error))?;
        self.cpu_start = Some(CpuStarter {
            device,
            mpidrs: mpidrs.to_vec(),
        });
        Ok(())
    }

    /// PE `cpu`'s CLPC sampler counters of `layout`, whose windows lie in
    /// its `cpu-impl-reg` at `physical`; its MRS and MSR of the counters
    /// reach the same model (`system_register`).
    pub fn add_core_counters(
        &mut self,
        cpu: usize,
        physical: u64,
        layout: stela_soc_data::core_counters::Layout,
    ) -> Result<(), MapError> {
        let model = CoreCounters::new(layout);
        let passive: Vec<_> = model
            .passive_words()
            .map(|(offset, words, index)| (offset, Arc::clone(words), index))
            .collect();
        let device = self.bus.add(Box::new(model));
        for (offset, bytes) in layout.windows() {
            let start = physical.checked_add(offset).ok_or(MapError::Overflow)?;
            self.bus.map(device, start, bytes, offset)?;
        }
        // Each passive slot lies in a window just mapped.
        for (offset, words, index) in passive {
            self.passive.add(physical + offset, &words, index);
        }
        if self.core_counters.len() <= cpu {
            self.core_counters.resize(cpu + 1, None);
        }
        self.core_counters[cpu] = Some(device);
        Ok(())
    }

    /// PE `cpu`'s `cpu-impl-reg` CPU_STS follows its power.
    pub(super) fn set_processor_running(&mut self, cpu: usize, running: bool) {
        if let Some(block) = self.reset_vectors.get(cpu).copied().flatten()
            && let Some(lockdown) = self.bus.device_mut::<Lockdown>(block)
        {
            lockdown.set_running(running);
        }
    }

    /// The reset address in PE `cpu`'s `cpu-impl-reg` RVBAR.
    pub(super) fn locked_reset_vector(&self, cpu: usize) -> Option<u64> {
        let block = self.reset_vectors.get(cpu).copied().flatten()?;
        self.bus.device::<Lockdown>(block)?.reset_address()
    }

    /// The AIC at `physical`, serving this PE. NR_IRQ is each die's inputs
    /// (`machine::soc::InterruptController`), which the model reports in its
    /// INFO or CAP0 register (m1n1 `aic.c`, Linux `irq-apple-aic.c`). An
    /// AICv2's or AICv3's `layout` gives its MAX_IRQ and populated dies and
    /// says where its per-die arrays, timestamp slots and event register
    /// are; an AICv1's are fixed.
    pub fn add_aic(&mut self, physical: u64, nr_irq: u16, layout: AicLayout) -> Result<()> {
        self.add_aic_for_processors(physical, nr_irq, layout, 1)
    }

    /// Serve the unchanged board PE indices and default AICv2/v3 delivery to
    /// the selected boot PE. Group-bit affinity still requires its independent
    /// source-checked binding; this supplies no guessed target groups.
    pub fn add_aic_for_processors(
        &mut self,
        physical: u64,
        nr_irq: u16,
        layout: AicLayout,
        cpus: usize,
    ) -> Result<()> {
        if cpus == 0 || cpus > u64::BITS as usize || self.sleep.boot >= cpus {
            return Err(refuse!(
                "AIC boot PE is outside the served processor roster"
            ));
        }
        if self.aic.is_some() {
            return Err(refuse!("AIC owner is already bound"));
        }
        let (model, span, v1): (Box<dyn MmioDevice>, u64, bool) = match layout {
            AicLayout::V1 => {
                let aic = Aic1::new(nr_irq, cpus)
                    .map_err(|error| refuse!("AIC1 configuration: {error}", ?error))?;
                (Box::new(aic), Aic1::span(), true)
            }
            AicLayout::V2 {
                version,
                irq_config,
                max_irq,
                event,
                timestamps,
                timestamp_base,
                dies,
            } => {
                let config = Aic2Config {
                    version,
                    timestamps: usize::from(timestamps),
                    timestamp_base,
                    nr_irq,
                    max_irq,
                    irq_config,
                    event_offset: event,
                    cpus,
                    default_cpus: 1 << self.sleep.boot,
                };
                let aic = Aic2::stacked(config, dies)
                    .map_err(|error| refuse!("AIC2 configuration: {error}", ?error))?;
                let span = aic.span();
                (Box::new(aic), span, false)
            }
        };
        let id = self
            .bus
            .install(model, &[(physical, span, 0)])
            .map_err(|error| refuse!("AIC placement: {error}", ?error))?;
        let device = if v1 {
            Controller::V1(id)
        } else {
            Controller::V2(id)
        };
        self.aic = Some(device);
        // The AIC starts no DMA and holds no master's registers: its
        // writes (unmasks after each acknowledge, software IRQs, routes,
        // timestamps) change only interrupt state, which its lines carry.
        self.quiet.push(id);
        if matches!(device, Controller::V2(_)) {
            self.register_power_device(physical, id, false)?;
        }
        Ok(())
    }
}
